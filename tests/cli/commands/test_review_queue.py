"""Tests for aragora review-queue packet + settlement flows."""

from __future__ import annotations

import argparse
import io
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

import pytest

from aragora.cli.commands.review_queue import (
    ADVISORY_NOTE,
    HIGH_RISK_PATHS,
    LARGE_DIFF_THRESHOLD,
    MODEL_REVIEW_QUEUE_CAP,
    PARKED_LABELS,
    QueueItem,
    ReviewPacket,
    _build_merge_authorization_packet,
    _build_model_review_quorum,
    _build_packet,
    _build_queue,
    _classify_pr,
    _classify_model_review_tier,
    _extract_validation_commands,
    _filter_lanes,
    _GhError,
    _is_high_risk_path,
    _parse_pr_number,
    _record_external_settlement,
    _render_packet,
    _requested_action,
    _settle_packet,
    _subsystem_for,
    _summarize_checks,
    add_review_queue_parser,
    cmd_review_queue,
)
from aragora.review import (
    EvidenceKind,
    EvidenceRef,
    FindingCategory,
    FindingSeverity,
    Recommendation,
    ReviewerFinding,
    ReviewerOutput,
)
from aragora.swarm.pr_review_protocol import EXECUTED_PROTOCOL_STATUS
from aragora.triage.auto_handle_calibration import AutoHandleDriftAlert


# --- Synthetic PR payload builder ------------------------------------------


def _make_pr(
    *,
    number: int = 1,
    title: str = "test PR",
    state: str = "OPEN",
    merged_at: str = "",
    is_draft: bool = False,
    mergeable: str = "MERGEABLE",
    review_decision: str = "",
    labels: list[str] | None = None,
    additions: int = 10,
    deletions: int = 5,
    changed_files: int = 2,
    checks: list[dict[str, Any]] | None = None,
    files: list[str] | None = None,
    author: str = "an0mium",
    body: str = "",
) -> dict[str, Any]:
    """Build a synthetic gh-pr-list-style payload."""
    return {
        "number": number,
        "title": title,
        "url": f"https://github.com/synaptent/aragora/pull/{number}",
        "state": state,
        "mergedAt": merged_at,
        "headRefName": f"branch-{number}",
        "headRefOid": f"sha{number:08d}",
        "baseRefName": "main",
        "baseRefOid": "basesha0001",
        "isDraft": is_draft,
        "mergeable": mergeable,
        "reviewDecision": review_decision,
        "labels": [{"name": lab} for lab in (labels or [])],
        "author": {"login": author},
        "additions": additions,
        "deletions": deletions,
        "changedFiles": changed_files,
        "statusCheckRollup": checks
        or [{"name": "lint", "status": "COMPLETED", "conclusion": "SUCCESS"}],
        "files": [{"path": p} for p in (files or [])],
        "body": body,
    }


def _make_reviewer_output(
    *,
    slot_id: str,
    provider: str,
    family: str,
    recommendation: Recommendation,
) -> ReviewerOutput:
    return ReviewerOutput(
        reviewer_id=f"{provider}:{slot_id}",
        slot_id=slot_id,
        provider=provider,
        lens="core" if slot_id in {"logic", "security"} else "heterodox",
        family=family,
        recommendation_class=recommendation,
        confidence=0.63,
        summary=f"{slot_id} summary",
        top_findings=(
            ReviewerFinding(
                category=FindingCategory.VALIDATION,
                severity=FindingSeverity.MEDIUM,
                claim=f"{slot_id} reviewed the diff",
                evidence=(f"{slot_id} evidence",),
                files=(),
            ),
        ),
        evidence_refs=(
            EvidenceRef(
                kind=EvidenceKind.FILE,
                path=f"aragora/{slot_id}.py",
                line_range=(1, 2),
                quote="example",
            ),
        ),
        risk_flags=(),
        open_questions=(),
        round_index=1,
        latency_ms=100,
        cost_usd=0.2,
    )


def _write_admin_squash_receipt(
    review_queue_root: Path,
    *,
    pr_number: int,
    head_sha: str,
) -> Path:
    receipts_dir = review_queue_root / "receipts"
    receipts_dir.mkdir(parents=True, exist_ok=True)
    path = receipts_dir / f"pr-{pr_number}-recorded-{head_sha[:12]}-admin_squash_merge.json"
    path.write_text(
        json.dumps(
            {
                "pr_number": pr_number,
                "head_sha": head_sha,
                "action": "admin_squash_merge",
                "github_event": "ADMIN_SQUASH_MERGE",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _write_human_risk_settlement_receipt(
    review_queue_root: Path,
    *,
    pr_number: int,
    head_sha: str,
    action: str = "approve",
    github_event: str = "APPROVE",
    payload_pr_number: int | None = None,
) -> Path:
    receipts_dir = review_queue_root / "receipts"
    receipts_dir.mkdir(parents=True, exist_ok=True)
    path = receipts_dir / f"pr-{pr_number}-recorded-{head_sha[:12]}-approve.json"
    path.write_text(
        json.dumps(
            {
                "pr_number": pr_number if payload_pr_number is None else payload_pr_number,
                "head_sha": head_sha,
                "action": action,
                "github_event": github_event,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _dogfood_comment(
    body: str = "## Cross-author adversarial dogfood (Claude)\n6/6 pass",
) -> dict[str, Any]:
    return {"author": {"login": "an0mium"}, "body": body}


def _codex_openai_body(
    heading: str = "## Codex focused dogfood",
    body: str = "local checks pass",
) -> str:
    return (
        f"{heading}\n\n"
        "**Reviewer harness:** codex\n"
        "**Model family:** openai\n"
        "**Model id:** gpt-5-codex\n"
        "**Receipt artifact:** /tmp/codex-review.md\n\n"
        f"{body}"
    )


def _codex_openai_comment(
    *,
    heading: str = "## Codex focused dogfood",
    body: str = "local checks pass",
    created_at: str | None = None,
) -> dict[str, Any]:
    comment = _dogfood_comment(_codex_openai_body(heading=heading, body=body))
    if created_at is not None:
        comment["createdAt"] = created_at
    return comment


def _executed_protocol(*, dissent: bool = False) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": EXECUTED_PROTOCOL_STATUS,
        "validation_summary": {
            "reviewer_execution": {
                "status": EXECUTED_PROTOCOL_STATUS,
                "reviewer_count": 3,
                "reviewer_ids": ["claude:logic", "openai-api:security", "gemini:maintainability"],
                "providers": ["claude", "openai-api", "gemini"],
                "dissent_count": 1 if dissent else 0,
            }
        },
        "dissenting_views": [],
    }
    if dissent:
        payload["dissenting_views"] = [
            {
                "agent": "openai-api:security",
                "position": "request_changes",
                "reason": "security reviewer found a blocker",
            }
        ]
    return payload


# --- _summarize_checks -----------------------------------------------------


class TestSummarizeChecks:
    def test_all_green(self) -> None:
        checks = [
            {"status": "COMPLETED", "conclusion": "SUCCESS"},
            {"status": "COMPLETED", "conclusion": "SUCCESS"},
        ]
        summary, has_fail, has_pending = _summarize_checks(checks)
        assert "2/2 green" in summary
        assert not has_fail
        assert not has_pending

    def test_one_failing(self) -> None:
        checks = [
            {"status": "COMPLETED", "conclusion": "SUCCESS"},
            {"status": "COMPLETED", "conclusion": "FAILURE"},
        ]
        summary, has_fail, has_pending = _summarize_checks(checks)
        assert has_fail
        assert "1 failing" in summary

    def test_pending(self) -> None:
        checks = [
            {"status": "IN_PROGRESS", "conclusion": ""},
            {"status": "COMPLETED", "conclusion": "SUCCESS"},
        ]
        summary, has_fail, has_pending = _summarize_checks(checks)
        assert has_pending
        assert not has_fail
        assert "1 pending" in summary

    def test_state_based_green_rollups_are_counted_as_green(self) -> None:
        checks = [
            {"context": "lint", "state": "SUCCESS"},
            {"context": "ci/unit", "state": "SUCCESS"},
        ]
        summary, has_fail, has_pending = _summarize_checks(checks)
        assert summary == "2/2 green"
        assert not has_fail
        assert not has_pending

    def test_state_based_pending_and_failure_rollups_are_preserved(self) -> None:
        checks = [
            {"context": "lint", "state": "SUCCESS"},
            {"context": "ci/unit", "state": "PENDING"},
            {"context": "security", "state": "FAILURE"},
        ]
        summary, has_fail, has_pending = _summarize_checks(checks)
        assert summary == "1 failing / 3 total"
        assert has_fail
        assert has_pending

    def test_skipped_excluded_from_meaningful_total(self) -> None:
        checks = [
            {"status": "COMPLETED", "conclusion": "SUCCESS"},
            {"status": "COMPLETED", "conclusion": "SKIPPED"},
            {"status": "COMPLETED", "conclusion": "CANCELLED"},
        ]
        summary, has_fail, has_pending = _summarize_checks(checks)
        assert not has_fail
        assert not has_pending
        assert "1/1 green" in summary

    def test_current_merge_quorum_self_check_pending_is_ignored(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GITHUB_WORKFLOW", "Aragora Merge Quorum")
        monkeypatch.setenv("GITHUB_JOB", "merge-quorum")
        monkeypatch.setenv("GITHUB_RUN_ID", "26288586838")
        monkeypatch.setenv("GITHUB_REPOSITORY", "synaptent/aragora")
        monkeypatch.setenv("GITHUB_SERVER_URL", "https://github.com")
        checks = [
            {
                "name": "aragora-merge-quorum",
                "workflowName": "Aragora Merge Quorum",
                "status": "IN_PROGRESS",
                "conclusion": "",
                "detailsUrl": (
                    "https://github.com/synaptent/aragora/actions/runs/26288586838/job/77396719826"
                ),
            },
            {"name": "lint", "status": "COMPLETED", "conclusion": "SUCCESS"},
        ]
        summary, has_fail, has_pending = _summarize_checks(checks)
        assert summary == "1/1 green"
        assert not has_fail
        assert not has_pending

    def test_merge_quorum_pending_outside_current_run_still_blocks(self) -> None:
        checks = [
            {
                "name": "aragora-merge-quorum",
                "workflowName": "Aragora Merge Quorum",
                "status": "IN_PROGRESS",
                "conclusion": "",
                "detailsUrl": (
                    "https://github.com/synaptent/aragora/actions/runs/26288586838/job/77396719826"
                ),
            },
            {"name": "lint", "status": "COMPLETED", "conclusion": "SUCCESS"},
        ]
        summary, has_fail, has_pending = _summarize_checks(checks)
        assert summary == "1 pending / 2 total"
        assert not has_fail
        assert has_pending

    def test_completed_merge_quorum_failure_still_blocks(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GITHUB_WORKFLOW", "Aragora Merge Quorum")
        monkeypatch.setenv("GITHUB_JOB", "merge-quorum")
        monkeypatch.setenv("GITHUB_RUN_ID", "26288586838")
        monkeypatch.setenv("GITHUB_REPOSITORY", "synaptent/aragora")
        monkeypatch.setenv("GITHUB_SERVER_URL", "https://github.com")
        checks = [
            {
                "name": "aragora-merge-quorum",
                "workflowName": "Aragora Merge Quorum",
                "status": "COMPLETED",
                "conclusion": "FAILURE",
                "detailsUrl": (
                    "https://github.com/synaptent/aragora/actions/runs/26288586838/job/77396719826"
                ),
            },
            {"name": "lint", "status": "COMPLETED", "conclusion": "SUCCESS"},
        ]
        summary, has_fail, has_pending = _summarize_checks(checks)
        assert summary == "1 failing / 2 total"
        assert has_fail
        assert not has_pending

    def test_unrelated_failure_still_blocks_with_merge_quorum_present(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GITHUB_WORKFLOW", "Aragora Merge Quorum")
        monkeypatch.setenv("GITHUB_JOB", "merge-quorum")
        monkeypatch.setenv("GITHUB_RUN_ID", "26288586838")
        monkeypatch.setenv("GITHUB_REPOSITORY", "synaptent/aragora")
        monkeypatch.setenv("GITHUB_SERVER_URL", "https://github.com")
        checks = [
            {
                "name": "aragora-merge-quorum",
                "workflowName": "Aragora Merge Quorum",
                "status": "IN_PROGRESS",
                "conclusion": "",
                "detailsUrl": (
                    "https://github.com/synaptent/aragora/actions/runs/26288586838/job/77396719826"
                ),
            },
            {"name": "typecheck", "status": "COMPLETED", "conclusion": "FAILURE"},
            {"name": "lint", "status": "COMPLETED", "conclusion": "SUCCESS"},
        ]
        summary, has_fail, has_pending = _summarize_checks(checks)
        assert summary == "1 failing / 2 total"
        assert has_fail
        assert not has_pending

    def test_unrelated_pending_still_blocks_with_merge_quorum_present(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GITHUB_WORKFLOW", "Aragora Merge Quorum")
        monkeypatch.setenv("GITHUB_JOB", "merge-quorum")
        monkeypatch.setenv("GITHUB_RUN_ID", "26288586838")
        monkeypatch.setenv("GITHUB_REPOSITORY", "synaptent/aragora")
        monkeypatch.setenv("GITHUB_SERVER_URL", "https://github.com")
        checks = [
            {
                "name": "aragora-merge-quorum",
                "workflowName": "Aragora Merge Quorum",
                "status": "IN_PROGRESS",
                "conclusion": "",
                "detailsUrl": (
                    "https://github.com/synaptent/aragora/actions/runs/26288586838/job/77396719826"
                ),
            },
            {"name": "typecheck", "status": "IN_PROGRESS", "conclusion": ""},
            {"name": "lint", "status": "COMPLETED", "conclusion": "SUCCESS"},
        ]
        summary, has_fail, has_pending = _summarize_checks(checks)
        assert summary == "1 pending / 2 total"
        assert not has_fail
        assert has_pending

    def test_no_checks(self) -> None:
        summary, has_fail, has_pending = _summarize_checks([])
        assert summary == "no checks"
        assert not has_fail
        assert not has_pending

    def test_malformed_check_ignored(self) -> None:
        checks: list[Any] = ["not a dict", None, {"status": "COMPLETED", "conclusion": "SUCCESS"}]
        summary, _, _ = _summarize_checks(checks)
        assert "1/1 green" in summary

    def test_superseded_historical_failure_ignored_by_latest_check_identity(self) -> None:
        checks = [
            {
                "name": "build",
                "workflowName": "Build Documentation (PR Check)",
                "status": "COMPLETED",
                "conclusion": "FAILURE",
                "completedAt": "2026-04-19T00:39:02Z",
            },
            {
                "name": "build",
                "workflowName": "Build Documentation (PR Check)",
                "status": "COMPLETED",
                "conclusion": "SKIPPED",
                "completedAt": "2026-05-21T15:28:50Z",
            },
            {
                "name": "lint",
                "workflowName": "Lint",
                "status": "COMPLETED",
                "conclusion": "SUCCESS",
                "completedAt": "2026-05-21T15:29:19Z",
            },
        ]

        summary, has_fail, has_pending = _summarize_checks(checks)

        assert summary == "1/1 green"
        assert not has_fail
        assert not has_pending

    def test_latest_failure_for_same_check_identity_still_blocks(self) -> None:
        checks = [
            {
                "name": "build",
                "workflowName": "Build Documentation (PR Check)",
                "status": "COMPLETED",
                "conclusion": "SUCCESS",
                "completedAt": "2026-05-21T15:28:50Z",
            },
            {
                "name": "build",
                "workflowName": "Build Documentation (PR Check)",
                "status": "COMPLETED",
                "conclusion": "FAILURE",
                "completedAt": "2026-05-21T15:30:50Z",
            },
        ]

        summary, has_fail, has_pending = _summarize_checks(checks)

        assert summary == "1 failing / 1 total"
        assert has_fail
        assert not has_pending

    def test_merge_quorum_self_check_pending_outside_current_run_blocks(self) -> None:
        checks = [
            {
                "name": "aragora-merge-quorum",
                "workflowName": "Aragora Merge Quorum",
                "status": "IN_PROGRESS",
                "conclusion": "",
                "startedAt": "2026-05-22T14:08:46Z",
            },
            {
                "name": "lint",
                "workflowName": "Lint",
                "status": "COMPLETED",
                "conclusion": "SUCCESS",
                "completedAt": "2026-05-22T13:01:00Z",
            },
        ]

        summary, has_fail, has_pending = _summarize_checks(checks)

        assert summary == "1 pending / 2 total"
        assert not has_fail
        assert has_pending

    def test_merge_quorum_self_check_failure_is_preserved(self) -> None:
        checks = [
            {
                "name": "aragora-merge-quorum",
                "workflowName": "Aragora Merge Quorum",
                "status": "COMPLETED",
                "conclusion": "FAILURE",
                "completedAt": "2026-05-22T14:09:40Z",
            },
            {
                "name": "Generate & Validate",
                "workflowName": "OpenAPI Spec",
                "status": "COMPLETED",
                "conclusion": "SUCCESS",
                "completedAt": "2026-05-22T12:46:43Z",
            },
        ]

        summary, has_fail, has_pending = _summarize_checks(checks)

        assert summary == "1 failing / 2 total"
        assert has_fail
        assert not has_pending

    def test_merge_quorum_self_check_requires_matching_repo_url(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GITHUB_WORKFLOW", "Aragora Merge Quorum")
        monkeypatch.setenv("GITHUB_JOB", "merge-quorum")
        monkeypatch.setenv("GITHUB_RUN_ID", "26311249200")
        monkeypatch.setenv("GITHUB_REPOSITORY", "synaptent/aragora")
        monkeypatch.setenv("GITHUB_SERVER_URL", "https://github.com")
        checks = [
            {
                "name": "aragora-merge-quorum",
                "workflowName": "Aragora Merge Quorum",
                "status": "IN_PROGRESS",
                "conclusion": "",
                "detailsUrl": (
                    "https://github.com/fork/aragora/actions/runs/26311249200/job/77460233891"
                ),
            },
            {"name": "lint", "status": "COMPLETED", "conclusion": "SUCCESS"},
        ]

        summary, has_fail, has_pending = _summarize_checks(checks)

        assert summary == "1 pending / 2 total"
        assert not has_fail
        assert has_pending

    def test_merge_quorum_self_check_requires_path_match_not_query(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GITHUB_WORKFLOW", "Aragora Merge Quorum")
        monkeypatch.setenv("GITHUB_JOB", "merge-quorum")
        monkeypatch.setenv("GITHUB_RUN_ID", "26311249200")
        monkeypatch.setenv("GITHUB_REPOSITORY", "synaptent/aragora")
        checks = [
            {
                "name": "aragora-merge-quorum",
                "workflowName": "Aragora Merge Quorum",
                "status": "IN_PROGRESS",
                "conclusion": "",
                "detailsUrl": (
                    "https://github.com/synaptent/aragora/pull/7436?"
                    "next=/synaptent/aragora/actions/runs/26311249200/job/77460233891"
                ),
            },
            {"name": "lint", "status": "COMPLETED", "conclusion": "SUCCESS"},
        ]

        summary, has_fail, has_pending = _summarize_checks(checks)

        assert summary == "1 pending / 2 total"
        assert not has_fail
        assert has_pending

    def test_merge_quorum_self_check_requires_path_match_not_fragment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GITHUB_WORKFLOW", "Aragora Merge Quorum")
        monkeypatch.setenv("GITHUB_JOB", "merge-quorum")
        monkeypatch.setenv("GITHUB_RUN_ID", "26311249200")
        monkeypatch.setenv("GITHUB_REPOSITORY", "synaptent/aragora")
        checks = [
            {
                "name": "aragora-merge-quorum",
                "workflowName": "Aragora Merge Quorum",
                "status": "IN_PROGRESS",
                "conclusion": "",
                "detailsUrl": (
                    "https://github.com/synaptent/aragora/pull/7436#"
                    "/synaptent/aragora/actions/runs/26311249200/job/77460233891"
                ),
            },
            {"name": "lint", "status": "COMPLETED", "conclusion": "SUCCESS"},
        ]

        summary, has_fail, has_pending = _summarize_checks(checks)

        assert summary == "1 pending / 2 total"
        assert not has_fail
        assert has_pending

    def test_merge_quorum_self_check_requires_server_host(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GITHUB_WORKFLOW", "Aragora Merge Quorum")
        monkeypatch.setenv("GITHUB_JOB", "merge-quorum")
        monkeypatch.setenv("GITHUB_RUN_ID", "26311249200")
        monkeypatch.setenv("GITHUB_REPOSITORY", "synaptent/aragora")
        monkeypatch.setenv("GITHUB_SERVER_URL", "https://github.enterprise.example")
        checks = [
            {
                "name": "aragora-merge-quorum",
                "workflowName": "Aragora Merge Quorum",
                "status": "IN_PROGRESS",
                "conclusion": "",
                "detailsUrl": (
                    "https://github.com/synaptent/aragora/actions/runs/26311249200/job/77460233891"
                ),
            },
            {"name": "lint", "status": "COMPLETED", "conclusion": "SUCCESS"},
        ]

        summary, has_fail, has_pending = _summarize_checks(checks)

        assert summary == "1 pending / 2 total"
        assert not has_fail
        assert has_pending

    def test_merge_quorum_self_check_requires_repository_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GITHUB_WORKFLOW", "Aragora Merge Quorum")
        monkeypatch.setenv("GITHUB_JOB", "merge-quorum")
        monkeypatch.setenv("GITHUB_RUN_ID", "26311249200")
        monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
        checks = [
            {
                "name": "aragora-merge-quorum",
                "workflowName": "Aragora Merge Quorum",
                "status": "IN_PROGRESS",
                "conclusion": "",
                "detailsUrl": (
                    "https://github.com/synaptent/aragora/actions/runs/26311249200/job/77460233891"
                ),
            },
            {"name": "lint", "status": "COMPLETED", "conclusion": "SUCCESS"},
        ]

        summary, has_fail, has_pending = _summarize_checks(checks)

        assert summary == "1 pending / 2 total"
        assert not has_fail
        assert has_pending

    def test_similarly_named_check_in_other_workflow_still_blocks(self) -> None:
        checks = [
            {
                "name": "aragora-merge-quorum",
                "workflowName": "Release Readiness Gate",
                "status": "COMPLETED",
                "conclusion": "FAILURE",
                "completedAt": "2026-05-22T14:09:40Z",
            },
        ]

        summary, has_fail, has_pending = _summarize_checks(checks)

        assert summary == "1 failing / 1 total"
        assert has_fail
        assert not has_pending

    def test_cancelled_merge_quorum_check_blocks_settlement_summary(self) -> None:
        checks = [
            {
                "name": "aragora-merge-quorum",
                "workflowName": "Aragora Merge Quorum",
                "status": "COMPLETED",
                "conclusion": "CANCELLED",
                "completedAt": "2026-05-27T17:13:53Z",
            },
            {
                "name": "lint",
                "workflowName": "Tests",
                "status": "COMPLETED",
                "conclusion": "SUCCESS",
                "completedAt": "2026-05-27T17:14:53Z",
            },
        ]

        summary, has_fail, has_pending = _summarize_checks(checks)

        assert summary == "1 failing / 2 total"
        assert has_fail
        assert not has_pending


# --- _classify_pr lane logic -----------------------------------------------


class TestClassifyPR:
    def test_ready_now_when_all_green_small_diff(self) -> None:
        pr = _make_pr()
        item = _classify_pr(pr)
        assert item.lane == "ready_now"

    def test_state_based_green_rollups_are_ready_now(self) -> None:
        pr = _make_pr(
            checks=[
                {"context": "lint", "state": "SUCCESS"},
                {"context": "ci/unit", "state": "SUCCESS"},
            ]
        )
        item = _classify_pr(pr)
        assert item.lane == "ready_now"

    def test_draft_is_parked(self) -> None:
        pr = _make_pr(is_draft=True)
        item = _classify_pr(pr)
        assert item.lane == "parked"
        assert "draft" in item.lane_reason.lower()

    def test_parked_label_parks_pr(self) -> None:
        for label in PARKED_LABELS:
            pr = _make_pr(labels=[label])
            item = _classify_pr(pr)
            assert item.lane == "parked", f"label={label} should park PR"

    def test_merge_conflict_is_parked(self) -> None:
        pr = _make_pr(mergeable="CONFLICTING")
        item = _classify_pr(pr)
        assert item.lane == "parked"
        assert "conflict" in item.lane_reason.lower()

    def test_failing_check_is_repairable(self) -> None:
        pr = _make_pr(
            checks=[
                {"status": "COMPLETED", "conclusion": "SUCCESS"},
                {"status": "COMPLETED", "conclusion": "FAILURE"},
            ]
        )
        item = _classify_pr(pr)
        assert item.lane == "repairable"

    def test_pending_check_needs_attention(self) -> None:
        pr = _make_pr(
            checks=[
                {"status": "IN_PROGRESS", "conclusion": ""},
            ]
        )
        item = _classify_pr(pr)
        assert item.lane == "needs_attention"

    def test_state_based_pending_check_needs_attention(self) -> None:
        pr = _make_pr(
            checks=[
                {"context": "ci/unit", "state": "PENDING"},
            ]
        )
        item = _classify_pr(pr)
        assert item.lane == "needs_attention"

    def test_state_based_failure_is_repairable(self) -> None:
        pr = _make_pr(
            checks=[
                {"context": "ci/unit", "state": "FAILURE"},
            ]
        )
        item = _classify_pr(pr)
        assert item.lane == "repairable"

    def test_large_diff_needs_attention(self) -> None:
        pr = _make_pr(additions=LARGE_DIFF_THRESHOLD + 100, deletions=10)
        item = _classify_pr(pr)
        assert item.lane == "needs_attention"
        assert "large diff" in item.lane_reason.lower()

    def test_priority_order_draft_beats_failing(self) -> None:
        # A draft PR with failing checks should still be parked, not repairable.
        pr = _make_pr(
            is_draft=True,
            checks=[{"status": "COMPLETED", "conclusion": "FAILURE"}],
        )
        item = _classify_pr(pr)
        assert item.lane == "parked"

    def test_priority_order_conflict_beats_failing(self) -> None:
        # Conflict parks the PR even when checks also fail.
        pr = _make_pr(
            mergeable="CONFLICTING",
            checks=[{"status": "COMPLETED", "conclusion": "FAILURE"}],
        )
        item = _classify_pr(pr)
        assert item.lane == "parked"


# --- _filter_lanes ---------------------------------------------------------


class TestFilterLanes:
    def _items(self) -> list[QueueItem]:
        # Build 4 items, one per lane, smallest synthetic representation.
        out: list[QueueItem] = []
        for lane in ("ready_now", "needs_attention", "repairable", "parked"):
            out.append(
                _classify_pr(
                    _make_pr(
                        number=hash(lane) & 0xFFFF,
                        is_draft=(lane == "parked"),
                        checks=[{"status": "COMPLETED", "conclusion": "FAILURE"}]
                        if lane == "repairable"
                        else (
                            [{"status": "IN_PROGRESS", "conclusion": ""}]
                            if lane == "needs_attention"
                            else [{"status": "COMPLETED", "conclusion": "SUCCESS"}]
                        ),
                    )
                )
            )
        return out

    def test_ready_only(self) -> None:
        items = self._items()
        result = _filter_lanes(items, ready_only=True, include_parked=False)
        assert {it.lane for it in result} == {"ready_now"}

    def test_default_excludes_parked(self) -> None:
        items = self._items()
        result = _filter_lanes(items, ready_only=False, include_parked=False)
        assert "parked" not in {it.lane for it in result}

    def test_include_parked_keeps_all(self) -> None:
        items = self._items()
        result = _filter_lanes(items, ready_only=False, include_parked=True)
        assert "parked" in {it.lane for it in result}


# --- _subsystem_for + _is_high_risk_path -----------------------------------


class TestSubsystemAndRisk:
    @pytest.mark.parametrize(
        ("path", "expected"),
        [
            ("aragora/cli/commands/review_pr.py", "aragora/cli"),
            ("tests/cli/commands/test_review_queue.py", "tests/cli"),
            ("docs/CI_LANES.md", "docs"),
            ("scripts/automation_pr_preflight.sh", "scripts"),
            ("benchmarks/bench_readiness/README.md", "benchmarks"),
            (".github/workflows/test.yml", ".github"),
            ("README.md", "README.md"),
        ],
    )
    def test_subsystem_mapping(self, path: str, expected: str) -> None:
        assert _subsystem_for(path) == expected

    def test_high_risk_exact_paths(self) -> None:
        for path in HIGH_RISK_PATHS:
            assert _is_high_risk_path(path), f"{path} should be flagged"

    @pytest.mark.parametrize(
        "path",
        [
            "aragora/security/encryption.py",
            "aragora/auth/oidc.py",
            "aragora/blockchain/wallet.py",
            "aragora/rbac/checker.py",
            "scripts/auto_revert_main_required_failures.py",
            ".github/workflows/release.yml",
        ],
    )
    def test_high_risk_prefixes(self, path: str) -> None:
        assert _is_high_risk_path(path)

    def test_non_high_risk(self) -> None:
        assert not _is_high_risk_path("aragora/cli/commands/review_queue.py")
        assert not _is_high_risk_path("aragora/cli/commands/swarm.py")
        assert not _is_high_risk_path("docs/CI_LANES.md")
        assert not _is_high_risk_path("tests/cli/commands/test_review_queue.py")


# --- model-review tier + quorum -------------------------------------------


class TestModelReviewQuorum:
    @pytest.mark.parametrize(
        ("files", "expected_tier"),
        [
            (["docs/status/queue.md"], 0),
            (["tests/swarm/test_handoff_contract.py"], 0),
            (["AGENTS.md", "CLAUDE.md", "docs/COORDINATION.md"], 0),
            (["aragora/agents/router.py"], 1),
            (["aragora/cli/commands/swarm.py"], 2),
            (["scripts/publish_automation_handoffs.py"], 2),
            (["aragora/metrics/manifold_brier.py"], 3),
            (["aragora/debate/team_selector.py"], 3),
            (["aragora/reputation/store.py"], 3),
            (["aragora/auth/session.py"], 3),
            (["sdk/typescript/src/index.ts"], 3),
            ([".github/workflows/tests.yml"], 4),
            (["deploy/k8s/app.yaml"], 4),
            # Merge-authority self-modification: see TIER_4_PREFIXES rationale.
            (["aragora/cli/commands/review_queue.py"], 4),
            (["aragora/cli/parser.py"], 4),
        ],
    )
    def test_classifies_merge_tiers(self, files: list[str], expected_tier: int) -> None:
        tier, _, _ = _classify_model_review_tier(files, pr=_make_pr(files=files))
        assert tier == expected_tier

    @pytest.mark.parametrize(
        "title",
        [
            "[AGT-03] Calibration curve reporting for ManifoldBrierScorer",
            "[AGT-05] Wire enable_agt05_reputation_selection into TeamSelectionConfig",
            "fix: semantic scoring correction",
        ],
    )
    def test_classifies_semantic_titles_as_tier_three(self, title: str) -> None:
        files = ["aragora/agents/router.py"]
        tier, _, reason = _classify_model_review_tier(
            files,
            pr=_make_pr(title=title, files=files),
        )
        assert tier == 3
        assert "semantic" in reason

    def test_tier_zero_satisfied_by_one_dogfood_note(self) -> None:
        pr = _make_pr(files=["docs/status/report.md"])
        pr["comments"] = [_dogfood_comment()]
        quorum = _build_model_review_quorum(
            pr=pr,
            files=["docs/status/report.md"],
            protocol={"status": "metadata_heuristic"},
            machine_recommendation="approve_candidate",
            has_pending=False,
            has_failures=False,
        )
        assert quorum["tier"] == 0
        assert quorum["status"] == "satisfied"
        assert quorum["admin_squash_allowed"] is True

    def test_tier_two_requires_dogfood_even_with_executed_reviewers(self) -> None:
        pr = _make_pr(files=["aragora/cli/commands/swarm.py"])
        pr["comments"] = []
        quorum = _build_model_review_quorum(
            pr=pr,
            files=["aragora/cli/commands/swarm.py"],
            protocol=_executed_protocol(),
            machine_recommendation="approve_candidate",
            has_pending=False,
            has_failures=False,
        )
        assert quorum["tier"] == 2
        assert quorum["status"] == "needs_model_review_quorum"
        assert quorum["admin_squash_allowed"] is False
        assert "focused adversarial dogfood evidence is required" in quorum["reasons"]

    def test_tier_two_allows_admin_squash_when_quorum_and_dogfood_clean(self) -> None:
        pr = _make_pr(files=["aragora/cli/commands/swarm.py"])
        pr["comments"] = [_dogfood_comment()]
        quorum = _build_model_review_quorum(
            pr=pr,
            files=["aragora/cli/commands/swarm.py"],
            protocol=_executed_protocol(),
            machine_recommendation="approve_candidate",
            has_pending=False,
            has_failures=False,
        )
        assert quorum["status"] == "satisfied"
        assert quorum["verdict"] == "admin_squash_allowed"
        assert quorum["admin_squash_allowed"] is True
        assert set(quorum["counted_reviewer_ids"]) == {"claude", "gemini", "openai"}

    def test_duplicate_codex_comments_do_not_satisfy_tier_two_quorum(self) -> None:
        pr = _make_pr(files=["aragora/cli/commands/swarm.py"])
        pr["comments"] = [
            _codex_openai_comment(),
            {
                "author": {"login": "an0mium"},
                "body": _codex_openai_body(
                    heading="## Codex review",
                    body="LGTM after local dogfood.",
                ),
            },
            {
                "author": {"login": "an0mium"},
                "body": _codex_openai_body(
                    heading="## Codex review",
                    body="Second same-model note.",
                ),
            },
        ]
        quorum = _build_model_review_quorum(
            pr=pr,
            files=["aragora/cli/commands/swarm.py"],
            protocol={"status": "metadata_heuristic"},
            machine_recommendation="approve_candidate",
            has_pending=False,
            has_failures=False,
        )
        assert quorum["tier"] == 2
        assert quorum["status"] == "needs_model_review_quorum"
        assert quorum["admin_squash_allowed"] is False
        assert quorum["counted_reviewer_ids"] == ["openai"]
        assert "model quorum incomplete: 1/2 signal(s)" in quorum["reasons"]

    def test_codex_dogfood_and_grok_review_satisfy_tier_two_quorum(self) -> None:
        pr = _make_pr(files=["aragora/cli/commands/swarm.py"])
        pr["comments"] = [
            _codex_openai_comment(),
            {
                "author": {"login": "an0mium"},
                "body": "## Grok independent model review\nVerdict: approve.",
            },
        ]
        quorum = _build_model_review_quorum(
            pr=pr,
            files=["aragora/cli/commands/swarm.py"],
            protocol={"status": "metadata_heuristic"},
            machine_recommendation="approve_candidate",
            has_pending=False,
            has_failures=False,
        )
        assert quorum["tier"] == 2
        assert quorum["status"] == "satisfied"
        assert quorum["admin_squash_allowed"] is True
        assert quorum["counted_reviewer_ids"] == ["grok", "openai"]
        assert quorum["counted_model_families"] == ["grok", "openai"]
        assert quorum["dogfood_evidence"][0]["surface_reviewer_id"] == "codex"
        assert quorum["dogfood_evidence"][0]["model_family"] == "openai"

    def test_unknown_dogfood_does_not_count_or_satisfy_required_dogfood(self) -> None:
        pr = _make_pr(files=["aragora/cli/commands/swarm.py"])
        pr["comments"] = [
            _dogfood_comment("## Focused dogfood\nlocal checks pass"),
            {
                "author": {"login": "an0mium"},
                "body": "## Grok independent model review\nVerdict: approve.",
            },
        ]
        quorum = _build_model_review_quorum(
            pr=pr,
            files=["aragora/cli/commands/swarm.py"],
            protocol={"status": "metadata_heuristic"},
            machine_recommendation="approve_candidate",
            has_pending=False,
            has_failures=False,
        )
        assert quorum["tier"] == 2
        assert quorum["status"] == "needs_model_review_quorum"
        assert quorum["counted_reviewer_ids"] == ["grok"]
        assert "focused adversarial dogfood evidence is required" in quorum["reasons"]

    def test_branch_name_substring_in_body_does_not_phantom_tag_reviewer(self) -> None:
        """A comment that mentions ``codex/...`` branch in a code block but
        has no model-review heading must not be tagged as a Codex signal."""
        pr = _make_pr(files=["aragora/cli/commands/swarm.py"])
        pr["comments"] = [
            _codex_openai_comment(),
            {
                "author": {"login": "an0mium"},
                "body": (
                    "## Rebased over current main after queue drain\n"
                    "Rebased `codex/model-review-quorum-settlement` onto current "
                    "`origin/main` after #6783 and #6787 merged.\n"
                    "Conflict resolution kept both parser surfaces:\n"
                    "- `review-queue baseline` from #6783\n"
                    "- `review-queue merge-packet` from this PR\n"
                ),
            },
        ]
        quorum = _build_model_review_quorum(
            pr=pr,
            files=["aragora/cli/commands/swarm.py"],
            protocol={"status": "metadata_heuristic"},
            machine_recommendation="approve_candidate",
            has_pending=False,
            has_failures=False,
        )
        # The rebase note's heading does not contain a model name.  The
        # heuristic must NOT scan the entire body and pick up the branch
        # name ``codex/...`` in line 2 of the body.
        assert quorum["counted_reviewer_ids"] == ["openai"]
        # The dogfood evidence list should still include both comments
        # (the rebase note matches "rebased" → not a marker; "drain"
        # → not a marker; but the body does not actually contain any
        # of dogfood/adversarial/cross-author/recheck), so it is not
        # added to dogfood_evidence at all.
        dogfood_authors = [entry.get("reviewer_id") for entry in quorum["dogfood_evidence"]]
        assert "openai" in dogfood_authors
        # Ensure the rebase note didn't sneak into reviewer_signals.
        for sig in quorum["reviewer_signals"]:
            assert "rebase" not in (sig.get("summary", "") or "").lower()

    def test_inferrer_uses_first_heading_only(self) -> None:
        """If a comment's first heading does not name a model, the
        inferrer must fall back to first 200 chars and NOT scan the
        entire body."""
        from aragora.cli.commands.review_queue import _infer_model_reviewer_from_text

        body_with_phantom_codex_deep_in_body = (
            "## Cross-author adversarial dogfood (no model named in heading)\n"
            "Local checks pass.\n\n"
            + ("Filler line that does not name any model.\n" * 30)
            + "Reference: https://github.com/example/repo/blob/main/codex/x.py\n"
        )
        # No model name in heading or first 200 chars → unknown.
        assert (
            _infer_model_reviewer_from_text(body_with_phantom_codex_deep_in_body)
            == "unknown_model_reviewer"
        )

        body_with_codex_heading = "## Codex review\nVerdict: approve.\n"
        assert _infer_model_reviewer_from_text(body_with_codex_heading) == "codex"

        body_with_grok_in_lead = (
            "Grok independent semantic review of head SHA abc1234.\n"
            "No heading present; relying on first-200-chars fallback.\n"
        )
        assert _infer_model_reviewer_from_text(body_with_grok_in_lead) == "grok"

    def test_stale_comments_excluded_when_predate_head_commit(self) -> None:
        """Comments posted before the current head was committed must
        be excluded from quorum unless they explicitly cite the head SHA."""
        head_sha = "abcdef1234567890abcdef1234567890abcdef12"
        pr = _make_pr(files=["aragora/cli/commands/swarm.py"])
        pr["headRefOid"] = head_sha
        pr["commits"] = [
            {"oid": head_sha, "committedDate": "2026-04-28T20:00:00Z"},
        ]
        pr["comments"] = [
            # Posted BEFORE the head was committed → stale.
            {
                "author": {"login": "an0mium"},
                "body": _codex_openai_body(),
                "createdAt": "2026-04-28T18:00:00Z",
            },
            {
                "author": {"login": "an0mium"},
                "body": "## Grok independent model review\nVerdict: approve.",
                "createdAt": "2026-04-28T18:30:00Z",
            },
        ]
        quorum = _build_model_review_quorum(
            pr=pr,
            files=["aragora/cli/commands/swarm.py"],
            protocol={"status": "metadata_heuristic"},
            machine_recommendation="approve_candidate",
            has_pending=False,
            has_failures=False,
        )
        # Both stale → quorum empty.
        assert quorum["counted_reviewer_ids"] == []
        assert quorum["status"] == "needs_model_review_quorum"

    def test_fresh_comments_after_head_commit_count(self) -> None:
        head_sha = "abcdef1234567890abcdef1234567890abcdef12"
        pr = _make_pr(files=["aragora/cli/commands/swarm.py"])
        pr["headRefOid"] = head_sha
        pr["commits"] = [
            {"oid": head_sha, "committedDate": "2026-04-28T20:00:00Z"},
        ]
        pr["comments"] = [
            {
                "author": {"login": "an0mium"},
                "body": _codex_openai_body(),
                "createdAt": "2026-04-28T20:05:00Z",
            },
            {
                "author": {"login": "an0mium"},
                "body": "## Grok independent model review\nVerdict: approve.",
                "createdAt": "2026-04-28T20:10:00Z",
            },
        ]
        quorum = _build_model_review_quorum(
            pr=pr,
            files=["aragora/cli/commands/swarm.py"],
            protocol={"status": "metadata_heuristic"},
            machine_recommendation="approve_candidate",
            has_pending=False,
            has_failures=False,
        )
        assert quorum["counted_reviewer_ids"] == ["grok", "openai"]
        assert quorum["status"] == "satisfied"

    def test_stale_comment_with_head_sha_citation_still_counts(self) -> None:
        """A reviewer who explicitly cites the current head SHA in
        their body counts even if their createdAt predates the head."""
        head_sha = "abcdef1234567890abcdef1234567890abcdef12"
        pr = _make_pr(files=["aragora/cli/commands/swarm.py"])
        pr["headRefOid"] = head_sha
        pr["commits"] = [
            {"oid": head_sha, "committedDate": "2026-04-28T20:00:00Z"},
        ]
        pr["comments"] = [
            # Predates head BUT cites head SHA → grounded.
            {
                "author": {"login": "an0mium"},
                "body": _codex_openai_body(
                    body=f"Reviewed at head {head_sha[:7]} - local checks pass."
                ),
                "createdAt": "2026-04-28T18:00:00Z",
            },
            {
                "author": {"login": "an0mium"},
                "body": "## Grok independent model review\nVerdict: approve.",
                "createdAt": "2026-04-28T20:10:00Z",
            },
        ]
        quorum = _build_model_review_quorum(
            pr=pr,
            files=["aragora/cli/commands/swarm.py"],
            protocol={"status": "metadata_heuristic"},
            machine_recommendation="approve_candidate",
            has_pending=False,
            has_failures=False,
        )
        assert quorum["counted_reviewer_ids"] == ["grok", "openai"]
        assert quorum["status"] == "satisfied"

    def test_unresolved_dissent_forces_human_risk_settlement(self) -> None:
        pr = _make_pr(files=["aragora/cli/commands/swarm.py"])
        pr["comments"] = [_dogfood_comment()]
        quorum = _build_model_review_quorum(
            pr=pr,
            files=["aragora/cli/commands/swarm.py"],
            protocol=_executed_protocol(dissent=True),
            machine_recommendation="approve_candidate",
            has_pending=False,
            has_failures=False,
        )
        assert quorum["status"] == "unresolved_dissent"
        assert quorum["requires_human_risk_settlement"] is True
        assert quorum["admin_squash_allowed"] is False

    def test_needs_attention_risk_signal_can_still_admin_squash_when_quorum_clean(self) -> None:
        pr = _make_pr(files=["aragora/cli/commands/swarm.py"])
        pr["comments"] = [_dogfood_comment()]
        quorum = _build_model_review_quorum(
            pr=pr,
            files=["aragora/cli/commands/swarm.py"],
            protocol=_executed_protocol(),
            machine_recommendation="needs_human_attention",
            has_pending=False,
            has_failures=False,
        )
        assert quorum["status"] == "satisfied"
        assert quorum["admin_squash_allowed"] is True

    def test_draft_state_blocks_settlement_even_with_quorum(self) -> None:
        pr = _make_pr(files=["aragora/cli/commands/swarm.py"], is_draft=True)
        pr["comments"] = [_dogfood_comment()]
        quorum = _build_model_review_quorum(
            pr=pr,
            files=["aragora/cli/commands/swarm.py"],
            protocol=_executed_protocol(),
            machine_recommendation="needs_human_attention",
            has_pending=False,
            has_failures=False,
        )
        assert quorum["status"] == "repair_or_wait"
        assert quorum["admin_squash_allowed"] is False

    def test_merged_state_blocks_settlement_even_with_quorum(self) -> None:
        pr = _make_pr(
            files=["docs/README.md"],
            state="MERGED",
            merged_at="2026-05-27T00:19:36Z",
        )
        pr["comments"] = [_dogfood_comment("## Claude focused dogfood\npass")]
        quorum = _build_model_review_quorum(
            pr=pr,
            files=["docs/README.md"],
            protocol={"status": "metadata_heuristic"},
            machine_recommendation="approve_candidate",
            has_pending=False,
            has_failures=False,
        )
        assert quorum["status"] == "repair_or_wait"
        assert quorum["verdict"] == "not_ready_for_settlement"
        assert quorum["admin_squash_allowed"] is False
        assert "PR is MERGED; settlement applies only to open PRs" in quorum["reasons"]

    def test_tier_three_never_admin_squashes_without_human_risk_settlement(self) -> None:
        pr = _make_pr(files=["aragora/reputation/store.py"])
        pr["comments"] = [_dogfood_comment()]
        quorum = _build_model_review_quorum(
            pr=pr,
            files=["aragora/reputation/store.py"],
            protocol=_executed_protocol(),
            machine_recommendation="approve_candidate",
            has_pending=False,
            has_failures=False,
        )
        assert quorum["tier"] == 3
        assert quorum["status"] == "human_risk_settlement_required"
        assert quorum["admin_squash_allowed"] is False
        assert quorum["requires_human_risk_settlement"] is True

    def test_tier_three_human_risk_settlement_allows_admin_squash(self) -> None:
        pr = _make_pr(files=["aragora/reputation/store.py"])
        pr["comments"] = [_dogfood_comment()]
        quorum = _build_model_review_quorum(
            pr=pr,
            files=["aragora/reputation/store.py"],
            protocol=_executed_protocol(),
            machine_recommendation="approve_candidate",
            has_pending=False,
            has_failures=False,
            human_risk_settlement_recorded=True,
        )
        assert quorum["tier"] == 3
        assert quorum["status"] == "satisfied"
        assert quorum["verdict"] == "admin_squash_allowed"
        assert quorum["admin_squash_allowed"] is True
        assert quorum["requires_human_risk_settlement"] is False
        assert quorum["human_risk_settlement_recorded"] is True
        assert "exact-head human risk settlement receipt recorded" in quorum["reasons"]

    def test_human_risk_settlement_does_not_clear_unresolved_dissent(self) -> None:
        pr = _make_pr(files=["aragora/reputation/store.py"])
        pr["comments"] = [_dogfood_comment()]
        quorum = _build_model_review_quorum(
            pr=pr,
            files=["aragora/reputation/store.py"],
            protocol=_executed_protocol(dissent=True),
            machine_recommendation="approve_candidate",
            has_pending=False,
            has_failures=False,
            human_risk_settlement_recorded=True,
        )
        assert quorum["status"] == "unresolved_dissent"
        assert quorum["admin_squash_allowed"] is False
        assert quorum["requires_human_risk_settlement"] is True

    @pytest.mark.parametrize(
        ("has_failures", "has_pending"),
        [
            (True, False),
            (False, True),
        ],
    )
    def test_human_risk_settlement_does_not_clear_failing_or_pending_checks(
        self,
        *,
        has_failures: bool,
        has_pending: bool,
    ) -> None:
        pr = _make_pr(files=["aragora/reputation/store.py"])
        pr["comments"] = [_dogfood_comment()]
        quorum = _build_model_review_quorum(
            pr=pr,
            files=["aragora/reputation/store.py"],
            protocol=_executed_protocol(),
            machine_recommendation="approve_candidate",
            has_pending=has_pending,
            has_failures=has_failures,
            human_risk_settlement_recorded=True,
        )
        assert quorum["status"] == "repair_or_wait"
        assert quorum["admin_squash_allowed"] is False
        assert quorum["requires_human_risk_settlement"] is True

    def test_independent_model_review_comment_counts_as_quorum_signal(self) -> None:
        pr = _make_pr(files=["aragora/debate/team_selector.py"])
        pr["comments"] = [
            _codex_openai_comment(body="10/10 pass"),
            {
                "author": {"login": "an0mium"},
                "body": "## Grok independent semantic review\nVerdict: approve after human risk settlement.",
            },
        ]
        quorum = _build_model_review_quorum(
            pr=pr,
            files=["aragora/debate/team_selector.py"],
            protocol={"status": "metadata_heuristic"},
            machine_recommendation="approve_candidate",
            has_pending=False,
            has_failures=False,
        )
        assert quorum["tier"] == 3
        assert quorum["status"] == "human_risk_settlement_required"
        assert len(quorum["reviewer_signals"]) == 1
        assert quorum["reviewer_signals"][0]["reviewer_id"] == "grok"
        assert len(quorum["dogfood_evidence"]) == 1
        assert quorum["counted_reviewer_ids"] == ["grok", "openai"]

    def test_github_actions_advisory_review_does_not_count_as_model_signal(self) -> None:
        pr = _make_pr(files=["aragora/debate/team_selector.py"])
        pr["comments"] = [
            _dogfood_comment("## Codex focused dogfood\n10/10 pass"),
            {
                "author": {"login": "github-actions"},
                "body": "## Aragora Code Review\n\nAdvisory-only review. No issues found.",
            },
        ]
        quorum = _build_model_review_quorum(
            pr=pr,
            files=["aragora/debate/team_selector.py"],
            protocol={"status": "metadata_heuristic"},
            machine_recommendation="approve_candidate",
            has_pending=False,
            has_failures=False,
        )
        assert quorum["tier"] == 3
        assert quorum["status"] == "needs_model_review_quorum"
        assert len(quorum["reviewer_signals"]) == 0
        assert len(quorum["dogfood_evidence"]) == 1

    def test_current_head_review_pr_object_warns_that_comment_form_is_required(
        self,
    ) -> None:
        head_sha = "abcdef1234567890abcdef1234567890abcdef12"
        pr = _make_pr(files=["aragora/cli/commands/swarm.py"])
        pr["headRefOid"] = head_sha
        pr["commits"] = [
            {"oid": head_sha, "committedDate": "2026-04-28T20:00:00Z"},
        ]
        pr["comments"] = [
            {
                **_dogfood_comment("## Claude focused dogfood\n10/10 pass"),
                "createdAt": "2026-04-28T20:05:00Z",
            },
        ]
        pr["reviews"] = [
            {
                "author": {"login": "an0mium"},
                "body": (
                    "## Aragora review-pr: advisory pass\n\n"
                    "- Reviewer: `codex`\n"
                    "- Model family: `openai`\n"
                    "- Model id: `gpt-5-codex`\n"
                    f"- Head SHA: `{head_sha}`\n"
                    "- Final status: `passed`\n"
                ),
                "commit": {"oid": head_sha},
                "state": "COMMENTED",
                "submittedAt": "2026-04-28T20:10:00Z",
            }
        ]
        quorum = _build_model_review_quorum(
            pr=pr,
            files=["aragora/cli/commands/swarm.py"],
            protocol={"status": "metadata_heuristic"},
            machine_recommendation="approve_candidate",
            has_pending=False,
            has_failures=False,
        )
        assert quorum["counted_reviewer_ids"] == ["claude"]
        assert quorum["status"] == "needs_model_review_quorum"
        assert any(
            "GitHub review object from openai" in reason and "PR comment" in reason
            for reason in quorum["reasons"]
        )
        assert not any("GitHub review object from codex" in reason for reason in quorum["reasons"])

    def test_review_pr_object_with_router_reviewer_requires_model_family_metadata(
        self,
    ) -> None:
        head_sha = "abcdef1234567890abcdef1234567890abcdef12"
        pr = _make_pr(files=["aragora/cli/commands/swarm.py"])
        pr["headRefOid"] = head_sha
        pr["commits"] = [
            {"oid": head_sha, "committedDate": "2026-04-28T20:00:00Z"},
        ]
        pr["comments"] = [
            {
                **_dogfood_comment("## Claude focused dogfood\n10/10 pass"),
                "createdAt": "2026-04-28T20:05:00Z",
            },
        ]
        pr["reviews"] = [
            {
                "author": {"login": "an0mium"},
                "body": (
                    "## Aragora review-pr: advisory pass\n\n"
                    "- Reviewer: `codex`\n"
                    f"- Head SHA: `{head_sha}`\n"
                    "- Final status: `passed`\n"
                ),
                "commit": {"oid": head_sha},
                "state": "COMMENTED",
                "submittedAt": "2026-04-28T20:10:00Z",
            }
        ]
        quorum = _build_model_review_quorum(
            pr=pr,
            files=["aragora/cli/commands/swarm.py"],
            protocol={"status": "metadata_heuristic"},
            machine_recommendation="approve_candidate",
            has_pending=False,
            has_failures=False,
        )
        assert quorum["counted_reviewer_ids"] == ["claude"]
        assert any(
            "GitHub review object from codex lacks lineage-bound model family metadata" in reason
            for reason in quorum["reasons"]
        )

    def test_off_head_review_pr_object_does_not_warn_as_current_evidence(
        self,
    ) -> None:
        head_sha = "abcdef1234567890abcdef1234567890abcdef12"
        pr = _make_pr(files=["aragora/cli/commands/swarm.py"])
        pr["headRefOid"] = head_sha
        pr["commits"] = [
            {"oid": head_sha, "committedDate": "2026-04-28T20:00:00Z"},
        ]
        pr["comments"] = [
            {
                **_dogfood_comment("## Claude focused dogfood\n10/10 pass"),
                "createdAt": "2026-04-28T20:05:00Z",
            },
        ]
        pr["reviews"] = [
            {
                "author": {"login": "an0mium"},
                "body": (
                    "## Aragora review-pr: advisory pass\n\n"
                    "- Reviewer: `codex`\n"
                    "- Model family: `openai`\n"
                    "- Model id: `gpt-5-codex`\n"
                    "- Final status: `passed`\n"
                ),
                "commit": {"oid": "0000000000000000000000000000000000000000"},
                "state": "COMMENTED",
                "submittedAt": "2026-04-28T20:10:00Z",
            }
        ]
        quorum = _build_model_review_quorum(
            pr=pr,
            files=["aragora/cli/commands/swarm.py"],
            protocol={"status": "metadata_heuristic"},
            machine_recommendation="approve_candidate",
            has_pending=False,
            has_failures=False,
        )
        assert quorum["counted_reviewer_ids"] == ["claude"]
        assert not any("GitHub review object" in reason for reason in quorum["reasons"])

    # --- Finding 6: merge-authority self-modification elevation ------------

    def test_review_queue_self_modification_classified_tier_four(self) -> None:
        """``aragora/cli/commands/review_queue.py`` is the merge-authority code.

        Modifying it must elevate to Tier 4 so the quorum gating the change
        is not the version of the gate the change is trying to land.
        """
        files = ["aragora/cli/commands/review_queue.py"]
        tier, name, reason = _classify_model_review_tier(files, pr=_make_pr(files=files))
        assert tier == 4
        assert "destructive" in reason or "workflow" in reason or "preapproval" in name

    def test_tier_four_review_queue_blocks_admin_squash_even_with_full_quorum(
        self,
    ) -> None:
        """Even with executed protocol + full dogfood, Tier 4 self-modification
        cannot admin-squash on its own — human preapproval is required."""
        files = ["aragora/cli/commands/review_queue.py"]
        pr = _make_pr(files=files)
        pr["comments"] = [
            _codex_openai_comment(),
            {
                "author": {"login": "an0mium"},
                "body": "## Grok independent model review\nVerdict: approve.",
            },
        ]
        quorum = _build_model_review_quorum(
            pr=pr,
            files=files,
            protocol=_executed_protocol(),
            machine_recommendation="approve_candidate",
            has_pending=False,
            has_failures=False,
        )
        assert quorum["tier"] == 4
        assert quorum["admin_squash_allowed"] is False
        assert quorum["verdict"] == "tier_4_human_preapproval_required"
        assert quorum["requires_human_preapproval"] is True

    # --- Finding 2: source-side filter on _dogfood_evidence_from_comments ---

    def test_dogfood_with_unknown_model_is_excluded_at_source(self) -> None:
        """A dogfood comment whose first heading does not name a known model
        must not appear in ``dogfood_evidence`` at all (parallel to the signals
        path's behaviour). This keeps the evidence list interpretable for
        downstream consumers; the counting boundary already neutralised the
        inflation, but the source-side filter prevents misleading artifacts."""
        files = ["aragora/agents/router.py"]  # Tier 1
        pr = _make_pr(files=files)
        pr["comments"] = [
            {
                "author": {"login": "an0mium"},
                "body": "## Generic dogfood note (no model named)\n2 cases pass",
            },
        ]
        quorum = _build_model_review_quorum(
            pr=pr,
            files=files,
            protocol={"status": "metadata_heuristic"},
            machine_recommendation="approve_candidate",
            has_pending=False,
            has_failures=False,
        )
        assert quorum["dogfood_evidence"] == []
        # And quorum still incomplete because dogfood is required but absent.
        assert quorum["admin_squash_allowed"] is False

    @pytest.mark.parametrize("bot_login", ("github-actions", "github-actions[bot]"))
    def test_dogfood_from_github_actions_is_excluded_at_source(self, bot_login: str) -> None:
        """Bot-authored dogfood comments must not count as model evidence,
        mirroring the existing filter in ``_model_review_signals_from_comments``."""
        files = ["aragora/agents/router.py"]
        pr = _make_pr(files=files)
        pr["comments"] = [
            {
                "author": {"login": bot_login},
                "body": _codex_openai_body(body="automated regression sweep"),
            },
            _codex_openai_comment(),
        ]
        quorum = _build_model_review_quorum(
            pr=pr,
            files=files,
            protocol={"status": "metadata_heuristic"},
            machine_recommendation="approve_candidate",
            has_pending=False,
            has_failures=False,
        )
        # The bot comment is filtered; the real reviewer comment passes.
        assert len(quorum["dogfood_evidence"]) == 1
        assert quorum["dogfood_evidence"][0]["github_author"] == "an0mium"
        assert quorum["counted_reviewer_ids"] == ["openai"]

    def test_model_review_signal_from_github_actions_bot_is_excluded_at_source(
        self,
    ) -> None:
        """The real GitHub Actions bot login must not count as a model reviewer."""
        files = ["aragora/agents/router.py"]
        pr = _make_pr(files=files)
        pr["comments"] = [
            {
                "author": {"login": "github-actions[bot]"},
                "body": _codex_openai_body(
                    heading="## Codex review",
                    body="independent semantic review with structured lineage metadata",
                ),
            },
        ]
        quorum = _build_model_review_quorum(
            pr=pr,
            files=files,
            protocol={"status": "metadata_heuristic"},
            machine_recommendation="approve_candidate",
            has_pending=False,
            has_failures=False,
        )

        assert quorum["reviewer_signals"] == []
        assert quorum["dogfood_evidence"] == []
        assert quorum["counted_reviewer_ids"] == []

    # --- Plain-headed dogfood with body-named model (PR #7587 regression) ---
    #
    # A dogfood comment headed `## Focused adversarial dogfood` (no `(claude)`
    # in the heading) but whose BODY discloses the model family must count, so
    # long as it is head-grounded, not bot-authored, and carries the dogfood
    # tokens. This mirrors the model-review-signal recognizer, which already
    # reads the model family from structured metadata rather than the heading.

    def test_plain_headed_dogfood_with_model_family_line_counts(self) -> None:
        """`## Focused adversarial dogfood` heading + `Model family: claude`
        line in the body must be recognized as Claude dogfood evidence."""
        files = ["aragora/agents/router.py"]  # Tier 1
        pr = _make_pr(files=files)
        pr["comments"] = [
            {
                "author": {"login": "an0mium"},
                "body": (
                    "## Focused adversarial dogfood\n\n"
                    "**Reviewer harness:** claude-code\n"
                    "**Model family:** claude\n"
                    "**Model id:** claude-opus-4-8\n"
                    "**Receipt artifact:** /tmp/claude-dogfood.md\n\n"
                    "6/6 adversarial cases pass."
                ),
            },
        ]
        quorum = _build_model_review_quorum(
            pr=pr,
            files=files,
            protocol={"status": "metadata_heuristic"},
            machine_recommendation="approve_candidate",
            has_pending=False,
            has_failures=False,
        )
        assert len(quorum["dogfood_evidence"]) == 1
        entry = quorum["dogfood_evidence"][0]
        assert entry["model_family"] == "claude"
        assert entry["reviewer_id"] == "claude"
        assert "claude" in quorum["counted_reviewer_ids"]

    def test_plain_headed_dogfood_with_far_model_family_line_counts(self) -> None:
        """The `Model family:` disclosure may appear anywhere in the body,
        not only in the first lines after the heading."""
        files = ["aragora/agents/router.py"]
        pr = _make_pr(files=files)
        pr["comments"] = [
            {
                "author": {"login": "an0mium"},
                "body": (
                    "## Focused adversarial dogfood\n\n"
                    + ("Walked each adversarial case manually.\n" * 30)
                    + "\n**Model family:** openai\n"
                    + "**Model id:** gpt-5-codex\n"
                    + "**Receipt artifact:** /tmp/openai-dogfood.md\n"
                ),
            },
        ]
        quorum = _build_model_review_quorum(
            pr=pr,
            files=files,
            protocol={"status": "metadata_heuristic"},
            machine_recommendation="approve_candidate",
            has_pending=False,
            has_failures=False,
        )
        assert len(quorum["dogfood_evidence"]) == 1
        assert quorum["dogfood_evidence"][0]["model_family"] == "openai"
        assert "openai" in quorum["counted_reviewer_ids"]

    def test_plain_headed_dogfood_without_model_named_still_excluded(self) -> None:
        """A plain-headed dogfood comment that names NO model anywhere in the
        body must still be excluded — the relaxation only applies when a known
        model family is discoverable in the body."""
        files = ["aragora/agents/router.py"]
        pr = _make_pr(files=files)
        pr["comments"] = [
            {
                "author": {"login": "an0mium"},
                "body": "## Focused adversarial dogfood\n\n6/6 cases pass, no model named.",
            },
        ]
        quorum = _build_model_review_quorum(
            pr=pr,
            files=files,
            protocol={"status": "metadata_heuristic"},
            machine_recommendation="approve_candidate",
            has_pending=False,
            has_failures=False,
        )
        assert quorum["dogfood_evidence"] == []
        assert quorum["counted_reviewer_ids"] == []

    def test_plain_headed_dogfood_not_head_grounded_still_excluded(self) -> None:
        """Even with a body-named model, a stale (non-head-grounded) dogfood
        comment must NOT count — head-grounding is preserved."""
        head_sha = "abcdef1234567890abcdef1234567890abcdef12"
        files = ["aragora/agents/router.py"]
        pr = _make_pr(files=files)
        pr["headRefOid"] = head_sha
        pr["commits"] = [{"oid": head_sha, "committedDate": "2026-04-28T20:00:00Z"}]
        pr["comments"] = [
            {
                "author": {"login": "an0mium"},
                # Posted BEFORE head commit and does not cite the head SHA.
                "createdAt": "2026-04-28T18:00:00Z",
                "body": (
                    "## Focused adversarial dogfood\n\n"
                    "**Model family:** claude\n"
                    "**Model id:** claude-opus-4-8\n"
                    "**Receipt artifact:** /tmp/r.md\n\n"
                    "6/6 cases pass."
                ),
            },
        ]
        quorum = _build_model_review_quorum(
            pr=pr,
            files=files,
            protocol={"status": "metadata_heuristic"},
            machine_recommendation="approve_candidate",
            has_pending=False,
            has_failures=False,
        )
        assert quorum["dogfood_evidence"] == []
        assert quorum["counted_reviewer_ids"] == []

    def test_plain_headed_dogfood_from_github_actions_still_excluded(self) -> None:
        """Even with a body-named model, a github-actions-authored dogfood
        comment must NOT count — the bot exclusion is preserved."""
        files = ["aragora/agents/router.py"]
        pr = _make_pr(files=files)
        pr["comments"] = [
            {
                "author": {"login": "github-actions[bot]"},
                "body": (
                    "## Focused adversarial dogfood\n\n"
                    "**Model family:** claude\n"
                    "**Model id:** claude-opus-4-8\n"
                    "**Receipt artifact:** /tmp/r.md\n\n"
                    "6/6 cases pass."
                ),
            },
        ]
        quorum = _build_model_review_quorum(
            pr=pr,
            files=files,
            protocol={"status": "metadata_heuristic"},
            machine_recommendation="approve_candidate",
            has_pending=False,
            has_failures=False,
        )
        assert quorum["dogfood_evidence"] == []
        assert quorum["counted_reviewer_ids"] == []

    def test_plain_headed_dogfood_with_unknown_model_family_excluded(self) -> None:
        """A `Model family:` line that names something we cannot normalize to a
        known family must NOT count (no phantom inflation)."""
        files = ["aragora/agents/router.py"]
        pr = _make_pr(files=files)
        pr["comments"] = [
            {
                "author": {"login": "an0mium"},
                "body": (
                    "## Focused adversarial dogfood\n\n"
                    "**Model family:** acme-frontier-9000\n\n"
                    "6/6 cases pass."
                ),
            },
        ]
        quorum = _build_model_review_quorum(
            pr=pr,
            files=files,
            protocol={"status": "metadata_heuristic"},
            machine_recommendation="approve_candidate",
            has_pending=False,
            has_failures=False,
        )
        assert quorum["dogfood_evidence"] == []
        assert quorum["counted_reviewer_ids"] == []

    def test_dogfood_heading_body_family_conflict_is_not_counted(self) -> None:
        """Finding 1 (fail-closed bypass): when the heading names one family and
        post-heading metadata names a *conflicting* family, the original
        resolver blocks the comment with ``heading_model_family_conflict``. The
        body-family fallback must NOT override that block — the comment stays
        uncounted."""
        from aragora.cli.commands.review_queue import _resolve_dogfood_identity

        body = (
            "## Claude focused adversarial dogfood\n\n"
            "**Reviewer harness:** claude-code\n"
            "**Model family:** openai\n"  # conflicts with the heading's "claude"
            "**Model id:** gpt-5-codex\n"
            "**Receipt artifact:** /tmp/r.md\n\n"
            "6/6 cases pass."
        )
        identity = _resolve_dogfood_identity(body)
        assert "heading_model_family_conflict" in identity.identity_problems
        assert identity.identity_source != "dogfood_body_model_family"

        files = ["aragora/agents/router.py"]  # Tier 1
        pr = _make_pr(files=files)
        pr["comments"] = [{"author": {"login": "an0mium"}, "body": body}]
        quorum = _build_model_review_quorum(
            pr=pr,
            files=files,
            protocol={"status": "metadata_heuristic"},
            machine_recommendation="approve_candidate",
            has_pending=False,
            has_failures=False,
        )
        assert quorum["counted_reviewer_ids"] == []

    def test_dogfood_model_family_inside_code_fence_is_not_counted(self) -> None:
        """Finding 2 (code-fence inflation): a `Model family:` line that appears
        only inside a fenced code block (e.g. a pasted example template) must NOT
        be treated as a real disclosure."""
        from aragora.cli.commands.review_queue import _model_family_from_body

        body = (
            "## Focused adversarial dogfood\n\n"
            "Reviewers should disclose their model like this:\n\n"
            "```\n"
            "**Model family:** claude\n"
            "```\n\n"
            "6/6 cases pass (no real disclosure outside the fence)."
        )
        assert _model_family_from_body(body) == ""

        files = ["aragora/agents/router.py"]
        pr = _make_pr(files=files)
        pr["comments"] = [{"author": {"login": "an0mium"}, "body": body}]
        quorum = _build_model_review_quorum(
            pr=pr,
            files=files,
            protocol={"status": "metadata_heuristic"},
            machine_recommendation="approve_candidate",
            has_pending=False,
            has_failures=False,
        )
        assert quorum["dogfood_evidence"] == []
        assert quorum["counted_reviewer_ids"] == []

    def test_dogfood_model_family_in_inline_code_span_is_not_counted(self) -> None:
        """An inline-code back-ticked `Model family:` mention must not be parsed
        as a disclosure either."""
        from aragora.cli.commands.review_queue import _model_family_from_body

        body = (
            "## Focused adversarial dogfood\n\n"
            "I left a note saying `Model family: claude` as an example only.\n\n"
            "6/6 cases pass."
        )
        assert _model_family_from_body(body) == ""

    def test_dogfood_real_family_line_outside_fence_still_counts(self) -> None:
        """A genuine `Model family:` disclosure outside any code fence still
        counts even when an example fence is also present — the fence stripping
        must not eat the real line."""
        from aragora.cli.commands.review_queue import _model_family_from_body

        body = (
            "## Focused adversarial dogfood\n\n"
            "Template for reference:\n\n"
            "```\n"
            "**Model family:** <family>\n"
            "```\n\n"
            "**Model family:** claude\n"
            "**Receipt artifact:** /tmp/r.md\n\n"
            "6/6 cases pass."
        )
        assert _model_family_from_body(body) == "claude"

        files = ["aragora/agents/router.py"]
        pr = _make_pr(files=files)
        pr["comments"] = [{"author": {"login": "an0mium"}, "body": body}]
        quorum = _build_model_review_quorum(
            pr=pr,
            files=files,
            protocol={"status": "metadata_heuristic"},
            machine_recommendation="approve_candidate",
            has_pending=False,
            has_failures=False,
        )
        assert len(quorum["dogfood_evidence"]) == 1
        assert quorum["dogfood_evidence"][0]["model_family"] == "claude"
        assert "claude" in quorum["counted_reviewer_ids"]


# --- _parse_pr_number ------------------------------------------------------


class TestParsePRNumber:
    @pytest.mark.parametrize(
        ("ref", "expected"),
        [
            ("6280", 6280),
            ("#6280", 6280),
            ("https://github.com/synaptent/aragora/pull/6280", 6280),
            ("https://github.com/synaptent/aragora/pull/6280/", 6280),
        ],
    )
    def test_parses(self, ref: str, expected: int) -> None:
        assert _parse_pr_number(ref) == expected

    def test_rejects_invalid(self) -> None:
        with pytest.raises(_GhError):
            _parse_pr_number("not a number")


class TestValidationExtraction:
    def test_extracts_validation_bullets(self) -> None:
        body = """
## Summary
- one

## Validation
- `python3 -m pytest tests/cli/commands/test_review_queue.py -q`
- `bash scripts/automation_pr_preflight.sh origin/main HEAD`

## Notes
- later
"""
        assert _extract_validation_commands(body) == [
            "`python3 -m pytest tests/cli/commands/test_review_queue.py -q`",
            "`bash scripts/automation_pr_preflight.sh origin/main HEAD`",
        ]


# --- _build_queue + _build_packet (with mocked gh) -------------------------


class TestBuildQueueAndPacket:
    def test_merge_packet_explicit_pr_refs_do_not_hydrate_open_queue(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fail_build_queue(*_args: Any, **_kwargs: Any) -> list[QueueItem]:
            raise AssertionError("explicit --pr merge-packet must not call _build_queue")

        def fake_build_packet(ref: str, **_kwargs: Any) -> ReviewPacket:
            return ReviewPacket(
                pr_number=int(ref),
                title=f"PR {ref}",
                url=f"https://github.com/synaptent/aragora/pull/{ref}",
                head_sha="abc123",
                base_sha="def456",
                author="codex",
                is_draft=False,
                additions=1,
                deletions=1,
                changed_files=1,
                queue_bucket="ready_now",
                touched_subsystems=["scripts"],
                high_risk_paths_touched=[],
                validation=[],
                checks_summary="4/4 green",
                risk_flags=[],
                machine_recommendation="approve_candidate",
                machine_recommendation_reason="bounded test packet",
                packet_sha="sha256:test",
                generated_at="2026-05-30T00:00:00+00:00",
                model_review_quorum={
                    "tier": 0,
                    "tier_name": "Tier 0",
                    "status": "satisfied",
                    "verdict": "admin_squash_allowed",
                    "admin_squash_allowed": True,
                    "requires_human_risk_settlement": False,
                    "unresolved_dissent": False,
                    "reviewer_signals": [],
                    "dogfood_evidence": [],
                    "counted_reviewer_ids": ["codex"],
                    "reasons": ["docs/tests/status-only change"],
                },
            )

        monkeypatch.setattr("aragora.cli.commands.review_queue._build_queue", fail_build_queue)
        monkeypatch.setattr("aragora.cli.commands.review_queue._build_packet", fake_build_packet)

        packet = _build_merge_authorization_packet(
            pr_refs=["7528"],
            limit=30,
            repo_override=None,
        )

        assert packet["queue_pressure"] == {
            "current_open_prs": 1,
            "cap": MODEL_REVIEW_QUEUE_CAP,
            "active": False,
            "scope": "explicit_pr_refs",
        }
        assert packet["admin_squash_order"] == [7528]

    def test_build_queue_classifies_and_sorts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        prs = [
            _make_pr(number=10, is_draft=True),  # parked
            _make_pr(number=20),  # ready_now
            _make_pr(
                number=30,
                checks=[{"status": "COMPLETED", "conclusion": "FAILURE"}],
            ),  # repairable
            _make_pr(
                number=40,
                checks=[{"status": "IN_PROGRESS", "conclusion": ""}],
            ),  # needs_attention
        ]
        monkeypatch.setattr(
            "aragora.cli.commands.review_queue._gh_json",
            lambda args: prs,
        )
        items = _build_queue(limit=100)
        assert [it.number for it in items] == [20, 40, 30, 10]
        assert [it.lane for it in items] == [
            "ready_now",
            "needs_attention",
            "repairable",
            "parked",
        ]

    def test_build_packet_sets_recommendation(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pr_payload = _make_pr(
            number=6280,
            files=["aragora/cli/commands/review_pr.py", "tests/cli/commands/test_review_pr.py"],
            body=(
                "## Validation\n"
                "- `python3 -m pytest tests/cli/commands/test_review_pr.py -q`\n"
                "- `bash scripts/automation_pr_preflight.sh origin/main HEAD`\n"
            ),
        )
        monkeypatch.setattr(
            "aragora.cli.commands.review_queue._gh_json",
            lambda args: pr_payload,
        )
        packet = _build_packet("6280", repo_override=None)
        assert packet.pr_number == 6280
        assert packet.advisory_only is True
        assert packet.settlement_note == ADVISORY_NOTE
        assert packet.machine_recommendation == "approve_candidate"
        assert packet.queue_bucket == "ready_now"
        assert packet.base_sha == "basesha0001"
        assert packet.packet_sha.startswith("sha256:")
        assert "aragora/cli" in packet.touched_subsystems
        assert "tests/cli" in packet.touched_subsystems
        assert packet.high_risk_paths_touched == []
        assert packet.protocol["binding"]["repo"] == "synaptent/aragora"
        assert packet.protocol["binding"]["base_sha"] == "basesha0001"
        assert packet.protocol["availability_summary"]["total_slots"] == 5
        assert packet.protocol["recommendation_class"] == "approve_candidate"
        assert packet.model_review_quorum["tier"] == 2
        assert packet.model_review_quorum["status"] == "needs_model_review_quorum"
        assert packet.validation == [
            "`python3 -m pytest tests/cli/commands/test_review_pr.py -q`",
            "`bash scripts/automation_pr_preflight.sh origin/main HEAD`",
        ]

    def test_build_packet_accepts_state_based_green_rollups(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pr_payload = _make_pr(
            number=6280,
            checks=[
                {"context": "lint", "state": "SUCCESS"},
                {"context": "ci/unit", "state": "SUCCESS"},
            ],
            files=["aragora/cli/commands/review_pr.py"],
        )
        monkeypatch.setattr(
            "aragora.cli.commands.review_queue._gh_json",
            lambda args: pr_payload,
        )
        packet = _build_packet("6280", repo_override=None)
        assert packet.machine_recommendation == "approve_candidate"
        assert packet.checks_summary == "2/2 green"

    def test_merged_pr_fails_closed_before_admin_authorization(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pr_payload = _make_pr(
            number=7470,
            files=["docs/status/focus.md"],
            state="MERGED",
            merged_at="2026-05-27T00:19:36Z",
        )
        pr_payload["comments"] = [
            {
                "author": {"login": "an0mium"},
                "body": "## Grok independent model review\nVerdict: approve.",
            },
        ]
        monkeypatch.setattr(
            "aragora.cli.commands.review_queue._gh_json",
            lambda args: pr_payload,
        )
        packet = _build_packet("7470", repo_override=None)
        quorum = packet.model_review_quorum

        assert packet.machine_recommendation == "needs_human_attention"
        assert "PR is MERGED; settlement applies only to open PRs" in packet.risk_flags
        assert quorum["admin_squash_allowed"] is False
        assert quorum["status"] == "repair_or_wait"
        assert "PR is MERGED; settlement applies only to open PRs" in quorum["reasons"]

    def test_closed_pr_fails_closed_before_admin_authorization(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pr_payload = _make_pr(number=7471, files=["docs/status/closed.md"], state="CLOSED")
        pr_payload["comments"] = [
            {
                "author": {"login": "an0mium"},
                "body": "## Grok independent model review\nVerdict: approve.",
            },
        ]
        monkeypatch.setattr(
            "aragora.cli.commands.review_queue._gh_json",
            lambda args: pr_payload,
        )
        packet = _build_packet("7471", repo_override=None)

        assert packet.model_review_quorum["admin_squash_allowed"] is False
        assert packet.model_review_quorum["status"] == "repair_or_wait"
        assert (
            "PR is CLOSED; settlement applies only to open PRs"
            in packet.model_review_quorum["reasons"]
        )

    def test_open_authorized_pr_still_allows_admin_squash(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pr_payload = _make_pr(number=7472, files=["docs/status/open.md"])
        pr_payload["comments"] = [
            {
                "author": {"login": "an0mium"},
                "body": "## Grok independent model review\nVerdict: approve.",
            },
        ]
        monkeypatch.setattr(
            "aragora.cli.commands.review_queue._gh_json",
            lambda args: pr_payload,
        )
        packet = _build_packet("7472", repo_override=None)

        assert packet.machine_recommendation == "approve_candidate"
        assert packet.model_review_quorum["admin_squash_allowed"] is True
        assert packet.model_review_quorum["status"] == "satisfied"

    def test_open_pr_with_missing_check_rollup_fails_closed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pr_payload = _make_pr(number=7465, files=["docs/status/open.md"])
        pr_payload["statusCheckRollup"] = None
        pr_payload["comments"] = [
            _dogfood_comment("## Claude focused dogfood\npass"),
            {
                "author": {"login": "an0mium"},
                "body": "## Grok independent model review\nVerdict: approve.",
            },
        ]
        monkeypatch.setattr(
            "aragora.cli.commands.review_queue._gh_json",
            lambda args: pr_payload,
        )
        packet = _build_packet("7465", repo_override=None)

        assert packet.checks_summary == "no checks reported"
        assert packet.machine_recommendation == "needs_human_attention"
        assert "check rollup unavailable" in packet.risk_flags
        assert packet.check_surfaces["pr_rollup"] == {
            "available": False,
            "count": None,
            "summary": "no checks reported",
        }
        assert packet.model_review_quorum["admin_squash_allowed"] is False
        assert packet.model_review_quorum["status"] == "repair_or_wait"
        assert (
            "checks are unavailable; wait for GitHub check rollup before settlement"
            in packet.model_review_quorum["reasons"]
        )

    def test_missing_check_rollup_reports_direct_commit_check_surface(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pr_payload = _make_pr(number=7465, files=["docs/status/open.md"])
        pr_payload["statusCheckRollup"] = []
        pr_payload["comments"] = [
            _dogfood_comment("## Claude focused dogfood\npass"),
            {
                "author": {"login": "an0mium"},
                "body": "## Grok independent model review\nVerdict: approve.",
            },
        ]

        def fake_gh_json(args: list[str]) -> dict[str, Any]:
            if args[:2] == ["pr", "view"]:
                return pr_payload
            if args[:1] == ["api"] and "required_status_checks" in args[1]:
                return {"contexts": ["lint", "typecheck"]}
            if args[:1] == ["api"] and "check-runs" in args[1]:
                return {
                    "check_runs": [
                        {"name": "lint", "status": "completed", "conclusion": "success"},
                        {"name": "typecheck", "status": "completed", "conclusion": "success"},
                        {
                            "name": "Python SDK Tests (3.11)",
                            "status": "completed",
                            "conclusion": "failure",
                        },
                    ]
                }
            raise AssertionError(f"unexpected gh call: {args}")

        monkeypatch.setattr("aragora.cli.commands.review_queue._gh_json", fake_gh_json)

        packet = _build_packet("7465", repo_override=None)
        direct = packet.check_surfaces["direct_commit_check_runs"]

        assert packet.check_surfaces["pr_rollup"] == {
            "available": False,
            "count": 0,
            "summary": "no checks reported",
        }
        assert packet.checks_summary == "2/2 required green (direct check-runs fallback)"
        assert "check rollup unavailable" not in packet.risk_flags
        assert direct["total"] == 3
        assert direct["branch_protection_strict"] is False
        assert direct["successful_required_contexts"] == ["lint", "typecheck"]
        assert direct["missing_required_contexts"] == []
        assert direct["non_success_required_contexts"] == []
        assert direct["required_contexts_satisfied"] is True
        assert direct["non_green_sample"] == ["Python SDK Tests (3.11)"]
        assert (
            "non-required direct check-runs are non-green; "
            "fallback gates only branch-protection required contexts"
        ) in packet.risk_flags
        assert packet.check_surfaces["effective_gate"] == {
            "source": "direct_commit_check_runs",
            "summary": "2/2 required green (direct check-runs fallback)",
        }
        assert (
            "every branch-protection required context successful"
            in packet.check_surfaces["diagnosis"]
        )
        assert packet.machine_recommendation == "approve_candidate"
        assert (
            "non-required direct check-runs are non-green" in packet.machine_recommendation_reason
        )
        assert packet.model_review_quorum["admin_squash_allowed"] is True
        assert packet.model_review_quorum["status"] == "satisfied"
        assert not any(
            "checks are unavailable" in reason for reason in packet.model_review_quorum["reasons"]
        )
        rendered = io.StringIO()
        with redirect_stdout(rendered):
            _render_packet(packet)
        rendered_packet = rendered.getvalue()
        assert "check surfaces:" in rendered_packet
        assert "direct_commit_check_runs=3" in rendered_packet
        assert "diagnosis:" in rendered_packet
        assert "remediation:" in rendered_packet

    def test_non_required_rollup_failures_use_required_pr_checks_gate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pr_payload = _make_pr(
            number=7465,
            files=["docs/status/open.md"],
            checks=[
                {"name": "lint", "status": "COMPLETED", "conclusion": "SUCCESS"},
                {"name": "typecheck", "status": "COMPLETED", "conclusion": "SUCCESS"},
                {
                    "name": "Mac TypeScript SDK Shadow",
                    "status": "QUEUED",
                    "conclusion": "",
                },
                {"name": "Docs Consistency", "status": "COMPLETED", "conclusion": "FAILURE"},
            ],
        )
        pr_payload["comments"] = [
            _dogfood_comment("## Claude focused dogfood\npass"),
            {
                "author": {"login": "an0mium"},
                "body": "## Grok independent model review\nVerdict: approve.",
            },
        ]

        def fake_gh_json(args: list[str]) -> Any:
            if args[:2] == ["pr", "view"]:
                return pr_payload
            if args[:2] == ["pr", "checks"]:
                return [
                    {
                        "name": "lint",
                        "state": "SUCCESS",
                        "bucket": "pass",
                        "workflow": "Lint",
                    },
                    {
                        "name": "typecheck",
                        "state": "SUCCESS",
                        "bucket": "pass",
                        "workflow": "Lint",
                    },
                ]
            raise AssertionError(f"unexpected gh call: {args}")

        monkeypatch.setattr("aragora.cli.commands.review_queue._gh_json", fake_gh_json)

        packet = _build_packet("7465", repo_override=None)

        assert packet.checks_summary == "2/2 required green (required PR checks)"
        assert packet.check_surfaces["effective_gate"] == {
            "source": "required_pr_checks",
            "summary": "2/2 required green (required PR checks)",
        }
        assert (
            "non-required PR checks are non-green; "
            "effective gate uses branch-protection required checks"
        ) in packet.risk_flags
        assert "non-required PR checks are non-green" in packet.machine_recommendation_reason
        assert packet.machine_recommendation == "approve_candidate"
        assert packet.model_review_quorum["admin_squash_allowed"] is True
        assert packet.model_review_quorum["status"] == "satisfied"
        assert packet.check_surfaces["required_pr_checks"]["gate_selected"] is True

    def test_required_pr_checks_gate_ignores_stale_self_row_inside_quorum_run(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pr_payload = _make_pr(
            number=7465,
            files=["docs/status/open.md"],
            checks=[
                {"name": "lint", "status": "COMPLETED", "conclusion": "SUCCESS"},
                {"name": "typecheck", "status": "COMPLETED", "conclusion": "SUCCESS"},
                {
                    "name": "Mac TypeScript SDK Shadow",
                    "workflowName": "Self-Hosted Shadow CI",
                    "status": "QUEUED",
                    "conclusion": "",
                    "runner_id": 0,
                    "runner_name": "",
                    "queuedDurationSeconds": 7200,
                },
                {
                    "name": "Hetzner Offline Golden Path Shadow",
                    "workflowName": "Self-Hosted Shadow CI",
                    "status": "QUEUED",
                    "conclusion": "",
                    "runner_id": 0,
                    "runner_name": "",
                    "queuedDurationSeconds": 7200,
                },
                {
                    "name": "aragora-merge-quorum",
                    "workflowName": "Aragora Merge Quorum",
                    "status": "COMPLETED",
                    "conclusion": "FAILURE",
                },
            ],
        )
        pr_payload["comments"] = [
            _dogfood_comment("## Claude focused dogfood\npass"),
            {
                "author": {"login": "an0mium"},
                "body": "## Grok independent model review\nVerdict: approve.",
            },
        ]
        monkeypatch.setenv("GITHUB_WORKFLOW", "Aragora Merge Quorum")
        monkeypatch.setenv("GITHUB_JOB", "merge-quorum")

        def fake_gh_json(args: list[str]) -> Any:
            if args[:2] == ["pr", "view"]:
                return pr_payload
            if args[:2] == ["pr", "checks"]:
                return [
                    {
                        "name": "aragora-merge-quorum",
                        "state": "FAILURE",
                        "bucket": "fail",
                        "workflow": "Aragora Merge Quorum",
                        "link": "https://github.com/synaptent/aragora/actions/runs/old/job/1",
                    },
                    {
                        "name": "lint",
                        "state": "SUCCESS",
                        "bucket": "pass",
                        "workflow": "Lint",
                    },
                    {
                        "name": "typecheck",
                        "state": "SUCCESS",
                        "bucket": "pass",
                        "workflow": "Lint",
                    },
                ]
            raise AssertionError(f"unexpected gh call: {args}")

        monkeypatch.setattr("aragora.cli.commands.review_queue._gh_json", fake_gh_json)

        packet = _build_packet("7465", repo_override=None)
        required = packet.check_surfaces["required_pr_checks"]
        rollup = packet.check_surfaces["pr_rollup"]

        assert packet.checks_summary == "2/2 required green (required PR checks)"
        assert required["effective_total"] == 2
        assert required["failing_or_cancelled"] == []
        assert required["pending"] == []
        assert packet.machine_recommendation == "approve_candidate"
        assert packet.model_review_quorum["admin_squash_allowed"] is True
        assert packet.model_review_quorum["status"] == "satisfied"
        assert rollup["optional_runner_capacity_noise_count"] == 2
        assert rollup["optional_runner_capacity_noise_sample"] == [
            "Self-Hosted Shadow CI / Mac TypeScript SDK Shadow",
            "Self-Hosted Shadow CI / Hetzner Offline Golden Path Shadow",
        ]
        assert rollup["long_queued_self_hosted_shadow_without_runner_metadata_count"] == 0

    def test_required_gate_classifies_real_rollup_shaped_long_queued_shadows(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pr_payload = _make_pr(
            number=7465,
            files=["docs/status/open.md"],
            checks=[
                {"name": "lint", "status": "COMPLETED", "conclusion": "SUCCESS"},
                {"name": "typecheck", "status": "COMPLETED", "conclusion": "SUCCESS"},
                {
                    "__typename": "CheckRun",
                    "name": "Mac TypeScript SDK Shadow",
                    "workflowName": "Self-Hosted Shadow CI",
                    "status": "QUEUED",
                    "conclusion": "",
                    "createdAt": "2026-05-31T15:51:16Z",
                    "startedAt": "",
                    "detailsUrl": ("https://github.com/synaptent/aragora/actions/runs/1/job/10"),
                },
                {
                    "__typename": "CheckRun",
                    "name": "Hetzner Offline Golden Path Shadow",
                    "workflowName": "Self-Hosted Shadow CI",
                    "status": "QUEUED",
                    "conclusion": "",
                    "createdAt": "2026-05-31T15:51:16Z",
                    "startedAt": "",
                    "detailsUrl": ("https://github.com/synaptent/aragora/actions/runs/1/job/11"),
                },
            ],
        )
        pr_payload["comments"] = [
            _dogfood_comment("## Claude focused dogfood\npass"),
            {
                "author": {"login": "an0mium"},
                "body": "## Grok independent model review\nVerdict: approve.",
            },
        ]

        def fake_gh_json(args: list[str]) -> Any:
            if args[:2] == ["pr", "view"]:
                return pr_payload
            if args[:2] == ["pr", "checks"]:
                return [
                    {
                        "name": "lint",
                        "state": "SUCCESS",
                        "bucket": "pass",
                        "workflow": "Lint",
                    },
                    {
                        "name": "typecheck",
                        "state": "SUCCESS",
                        "bucket": "pass",
                        "workflow": "Lint",
                    },
                ]
            raise AssertionError(f"unexpected gh call: {args}")

        monkeypatch.setattr("aragora.cli.commands.review_queue._gh_json", fake_gh_json)

        packet = _build_packet("7465", repo_override=None)
        rollup = packet.check_surfaces["pr_rollup"]

        assert packet.checks_summary == "2/2 required green (required PR checks)"
        assert rollup["non_required_non_green_count"] == 2
        assert rollup["optional_runner_capacity_noise_count"] == 0
        assert rollup["optional_runner_capacity_noise_sample"] == []
        assert rollup["long_queued_self_hosted_shadow_without_runner_metadata_count"] == 2
        assert rollup["long_queued_self_hosted_shadow_without_runner_metadata_sample"] == [
            "Self-Hosted Shadow CI / Mac TypeScript SDK Shadow",
            "Self-Hosted Shadow CI / Hetzner Offline Golden Path Shadow",
        ]
        assert packet.model_review_quorum["admin_squash_allowed"] is True

        rendered = io.StringIO()
        with redirect_stdout(rendered):
            _render_packet(packet)
        rendered_packet = rendered.getvalue()
        assert "optional_runner_capacity_noise=" not in rendered_packet
        assert (
            "long_queued_self_hosted_shadow_without_runner_metadata=Self-Hosted Shadow CI"
            in rendered_packet
        )

    def test_required_pr_checks_gate_preserves_self_row_outside_quorum_run(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pr_payload = _make_pr(
            number=7465,
            files=["docs/status/open.md"],
            checks=[
                {
                    "name": "Mac TypeScript SDK Shadow",
                    "workflowName": "Self-Hosted Shadow CI",
                    "status": "QUEUED",
                    "conclusion": "",
                    "runner_id": 0,
                    "runner_name": "",
                    "queuedDurationSeconds": 7200,
                },
                {
                    "name": "Hetzner Offline Golden Path Shadow",
                    "workflowName": "Self-Hosted Shadow CI",
                    "status": "COMPLETED",
                    "conclusion": "CANCELLED",
                },
                {
                    "name": "Smoke Shadow",
                    "workflowName": "Self-Hosted Shadow CI",
                    "status": "COMPLETED",
                    "conclusion": "SKIPPED",
                },
                {
                    "name": "aragora-merge-quorum",
                    "workflowName": "Aragora Merge Quorum",
                    "status": "COMPLETED",
                    "conclusion": "FAILURE",
                },
                {"name": "lint", "status": "COMPLETED", "conclusion": "SUCCESS"},
            ],
        )
        pr_payload["comments"] = [
            _dogfood_comment("## Claude focused dogfood\npass"),
            {
                "author": {"login": "an0mium"},
                "body": "## Grok independent model review\nVerdict: approve.",
            },
        ]

        def fake_gh_json(args: list[str]) -> Any:
            if args[:2] == ["pr", "view"]:
                return pr_payload
            if args[:2] == ["pr", "checks"]:
                return [
                    {
                        "name": "aragora-merge-quorum",
                        "state": "FAILURE",
                        "bucket": "fail",
                        "workflow": "Aragora Merge Quorum",
                    },
                    {
                        "name": "lint",
                        "state": "SUCCESS",
                        "bucket": "pass",
                        "workflow": "Lint",
                    },
                ]
            raise AssertionError(f"unexpected gh call: {args}")

        monkeypatch.setattr("aragora.cli.commands.review_queue._gh_json", fake_gh_json)

        packet = _build_packet("7465", repo_override=None)
        rollup = packet.check_surfaces["pr_rollup"]

        assert "effective_gate" not in packet.check_surfaces
        assert rollup["optional_runner_capacity_noise_count"] == 1
        assert rollup["long_queued_self_hosted_shadow_without_runner_metadata_count"] == 0
        assert packet.machine_recommendation == "repair_first"
        assert packet.model_review_quorum["admin_squash_allowed"] is False
        assert packet.model_review_quorum["status"] == "repair_or_wait"

    def test_required_pr_checks_gate_keeps_non_self_required_failure_blocking(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pr_payload = _make_pr(
            number=7465,
            files=["docs/status/open.md"],
            checks=[
                {"name": "lint", "status": "COMPLETED", "conclusion": "SUCCESS"},
                {"name": "typecheck", "status": "COMPLETED", "conclusion": "FAILURE"},
                {
                    "name": "Mac TypeScript SDK Shadow",
                    "workflowName": "Self-Hosted Shadow CI",
                    "status": "QUEUED",
                    "conclusion": "",
                    "runner_id": 0,
                    "runner_name": "",
                    "queuedDurationSeconds": 7200,
                },
                {
                    "name": "aragora-merge-quorum",
                    "workflowName": "Aragora Merge Quorum",
                    "status": "COMPLETED",
                    "conclusion": "FAILURE",
                },
            ],
        )
        pr_payload["comments"] = [
            _dogfood_comment("## Claude focused dogfood\npass"),
            {
                "author": {"login": "an0mium"},
                "body": "## Grok independent model review\nVerdict: approve.",
            },
        ]
        monkeypatch.setenv("GITHUB_WORKFLOW", "Aragora Merge Quorum")
        monkeypatch.setenv("GITHUB_JOB", "merge-quorum")

        def fake_gh_json(args: list[str]) -> Any:
            if args[:2] == ["pr", "view"]:
                return pr_payload
            if args[:2] == ["pr", "checks"]:
                return [
                    {
                        "name": "aragora-merge-quorum",
                        "state": "FAILURE",
                        "bucket": "fail",
                        "workflow": "Aragora Merge Quorum",
                    },
                    {
                        "name": "lint",
                        "state": "SUCCESS",
                        "bucket": "pass",
                        "workflow": "Lint",
                    },
                    {
                        "name": "typecheck",
                        "state": "FAILURE",
                        "bucket": "fail",
                        "workflow": "Lint",
                    },
                ]
            raise AssertionError(f"unexpected gh call: {args}")

        monkeypatch.setattr("aragora.cli.commands.review_queue._gh_json", fake_gh_json)

        packet = _build_packet("7465", repo_override=None)
        required = packet.check_surfaces["required_pr_checks"]
        rollup = packet.check_surfaces["pr_rollup"]

        assert "effective_gate" not in packet.check_surfaces
        assert required["failing_or_cancelled"] == ["typecheck"]
        assert rollup["optional_runner_capacity_noise_count"] == 1
        assert rollup["optional_runner_capacity_noise_sample"] == [
            "Self-Hosted Shadow CI / Mac TypeScript SDK Shadow"
        ]
        assert rollup["long_queued_self_hosted_shadow_without_runner_metadata_count"] == 0
        assert packet.machine_recommendation == "repair_first"
        assert packet.model_review_quorum["admin_squash_allowed"] is False
        assert packet.model_review_quorum["status"] == "repair_or_wait"

    def test_required_pr_checks_gate_fails_closed_when_only_self_check_visible(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pr_payload = _make_pr(
            number=7465,
            files=["docs/status/open.md"],
            checks=[
                {"name": "Docs Consistency", "status": "COMPLETED", "conclusion": "FAILURE"},
            ],
        )
        pr_payload["comments"] = [
            _dogfood_comment("## Claude focused dogfood\npass"),
            {
                "author": {"login": "an0mium"},
                "body": "## Grok independent model review\nVerdict: approve.",
            },
        ]
        monkeypatch.setenv("GITHUB_WORKFLOW", "Aragora Merge Quorum")
        monkeypatch.setenv("GITHUB_JOB", "merge-quorum")
        monkeypatch.setenv("GITHUB_RUN_ID", "123456")
        monkeypatch.setenv("GITHUB_REPOSITORY", "synaptent/aragora")
        monkeypatch.setenv("GITHUB_SERVER_URL", "https://github.com")

        def fake_gh_json(args: list[str]) -> Any:
            if args[:2] == ["pr", "view"]:
                return pr_payload
            if args[:2] == ["pr", "checks"]:
                return [
                    {
                        "name": "aragora-merge-quorum",
                        "state": "PENDING",
                        "bucket": "pending",
                        "workflow": "Aragora Merge Quorum",
                        "link": "https://github.com/synaptent/aragora/actions/runs/123456/job/1",
                    },
                ]
            raise AssertionError(f"unexpected gh call: {args}")

        monkeypatch.setattr("aragora.cli.commands.review_queue._gh_json", fake_gh_json)

        packet = _build_packet("7465", repo_override=None)
        required = packet.check_surfaces["required_pr_checks"]

        assert required["effective_total"] == 0
        assert required["summary"] == "no required checks"
        assert required["gate_selected"] is False
        assert required["ignored_current_merge_quorum_self_check_count"] == 1
        assert "no effective branch-protection required checks" in required["gate_blocked_reason"]
        assert "effective_gate" not in packet.check_surfaces
        assert packet.model_review_quorum["admin_squash_allowed"] is False
        assert packet.model_review_quorum["status"] == "repair_or_wait"

    def test_merge_quorum_self_check_uses_required_gate_despite_shadow_jobs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pr_payload = _make_pr(
            number=7465,
            files=["pyproject.toml", "uv.lock"],
            checks=[
                {
                    "name": "aragora-merge-quorum",
                    "workflowName": "Aragora Merge Quorum",
                    "status": "QUEUED",
                    "conclusion": "",
                    "detailsUrl": (
                        "https://github.com/synaptent/aragora/actions/runs/123456/job/1"
                    ),
                },
                {
                    "name": "lint",
                    "workflowName": "Lint",
                    "status": "COMPLETED",
                    "conclusion": "SUCCESS",
                },
                {
                    "name": "TypeScript SDK Type Check",
                    "workflowName": "SDK Tests",
                    "status": "COMPLETED",
                    "conclusion": "SUCCESS",
                },
                {
                    "name": "Mac TypeScript SDK Shadow",
                    "workflowName": "Self-Hosted Shadow CI",
                    "status": "QUEUED",
                    "conclusion": "",
                    "runner_id": 0,
                    "runner_name": "",
                    "queuedDurationSeconds": 7200,
                },
                {
                    "name": "Hetzner Offline Golden Path Shadow",
                    "workflowName": "Self-Hosted Shadow CI",
                    "status": "QUEUED",
                    "conclusion": "",
                    "runner_id": 0,
                    "runner_name": "",
                    "queuedDurationSeconds": 7200,
                },
            ],
        )
        pr_payload["comments"] = [
            _codex_openai_comment(),
            {
                "author": {"login": "an0mium"},
                "body": "## Grok independent model review\nVerdict: approve.",
            },
        ]
        monkeypatch.setenv("GITHUB_WORKFLOW", "Aragora Merge Quorum")
        monkeypatch.setenv("GITHUB_JOB", "merge-quorum")
        monkeypatch.setenv("GITHUB_RUN_ID", "123456")
        monkeypatch.setenv("GITHUB_REPOSITORY", "synaptent/aragora")
        monkeypatch.setenv("GITHUB_SERVER_URL", "https://github.com")

        def fake_gh_json(args: list[str]) -> Any:
            if args[:2] == ["pr", "view"]:
                return pr_payload
            if args[:2] == ["pr", "checks"]:
                return [
                    {
                        "name": "aragora-merge-quorum",
                        "state": "PENDING",
                        "bucket": "pending",
                        "workflow": "Aragora Merge Quorum",
                        "link": "https://github.com/synaptent/aragora/actions/runs/123456/job/1",
                    },
                    {
                        "name": "lint",
                        "state": "SUCCESS",
                        "bucket": "pass",
                        "workflow": "Lint",
                    },
                    {
                        "name": "TypeScript SDK Type Check",
                        "state": "SUCCESS",
                        "bucket": "pass",
                        "workflow": "SDK Tests",
                    },
                ]
            raise AssertionError(f"unexpected gh call: {args}")

        monkeypatch.setattr("aragora.cli.commands.review_queue._gh_json", fake_gh_json)

        packet = _build_packet("7465", repo_override=None)
        required = packet.check_surfaces["required_pr_checks"]
        rollup = packet.check_surfaces["pr_rollup"]

        assert packet.checks_summary == "2/2 required green (required PR checks)"
        assert required["effective_total"] == 2
        assert required["ignored_current_merge_quorum_self_check_count"] == 1
        assert required["gate_selected"] is True
        assert required["pending"] == []
        assert rollup["non_required_non_green_count"] == 2
        assert rollup["non_required_non_green_sample"] == [
            "Self-Hosted Shadow CI / Mac TypeScript SDK Shadow",
            "Self-Hosted Shadow CI / Hetzner Offline Golden Path Shadow",
        ]
        assert rollup["optional_runner_capacity_noise_count"] == 2
        assert rollup["optional_runner_capacity_noise_sample"] == [
            "Self-Hosted Shadow CI / Mac TypeScript SDK Shadow",
            "Self-Hosted Shadow CI / Hetzner Offline Golden Path Shadow",
        ]
        assert rollup["long_queued_self_hosted_shadow_without_runner_metadata_count"] == 0
        assert packet.model_review_quorum["status"] == "satisfied"
        assert packet.model_review_quorum["admin_squash_allowed"] is True

        rendered = io.StringIO()
        with redirect_stdout(rendered):
            _render_packet(packet)
        rendered_packet = rendered.getvalue()
        assert "gate_selected=true" in rendered_packet
        assert "non_required_non_green_rollup=Self-Hosted Shadow CI" in rendered_packet
        assert "optional_runner_capacity_noise=Self-Hosted Shadow CI" in rendered_packet

    def test_required_pr_checks_unavailable_explains_gate_not_selected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pr_payload = _make_pr(
            number=7465,
            files=["pyproject.toml", "uv.lock"],
            checks=[
                {"name": "lint", "status": "COMPLETED", "conclusion": "SUCCESS"},
                {
                    "name": "Mac TypeScript SDK Shadow",
                    "workflowName": "Self-Hosted Shadow CI",
                    "status": "QUEUED",
                    "conclusion": "",
                },
            ],
        )
        pr_payload["comments"] = [
            _dogfood_comment("## Codex focused dogfood\npass"),
            {
                "author": {"login": "an0mium"},
                "body": "## Claude review\nVerdict: approve.",
            },
        ]

        def fake_gh_json(args: list[str]) -> Any:
            if args[:2] == ["pr", "view"]:
                return pr_payload
            if args[:2] == ["pr", "checks"]:
                raise _GhError("required checks request timed out")
            raise AssertionError(f"unexpected gh call: {args}")

        monkeypatch.setattr("aragora.cli.commands.review_queue._gh_json", fake_gh_json)

        packet = _build_packet("7465", repo_override=None)
        required = packet.check_surfaces["required_pr_checks"]

        assert required["available"] is False
        assert required["effective_total"] == 0
        assert required["gate_selected"] is False
        assert "cannot distinguish required checks" in required["gate_blocked_reason"]
        assert "required checks request timed out" in required["error"]
        assert "effective_gate" not in packet.check_surfaces
        assert packet.model_review_quorum["status"] == "repair_or_wait"
        assert packet.model_review_quorum["admin_squash_allowed"] is False

        rendered = io.StringIO()
        with redirect_stdout(rendered):
            _render_packet(packet)
        assert "required_gate_blocker:" in rendered.getvalue()

    def test_missing_check_rollup_uses_modern_checks_field_and_skipped_neutral(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pr_payload = _make_pr(number=7465, files=["docs/status/open.md"])
        pr_payload["statusCheckRollup"] = []
        pr_payload["comments"] = [
            _dogfood_comment("## Claude focused dogfood\npass"),
            {
                "author": {"login": "an0mium"},
                "body": "## Grok independent model review\nVerdict: approve.",
            },
        ]

        def fake_gh_json(args: list[str]) -> dict[str, Any]:
            if args[:2] == ["pr", "view"]:
                return pr_payload
            if args[:1] == ["api"] and "required_status_checks" in args[1]:
                return {
                    "contexts": [],
                    "checks": [
                        {"context": "lint", "app_id": 15368},
                        {"context": "typecheck", "app_id": 15368},
                    ],
                    "strict": False,
                }
            if args[:1] == ["api"] and "check-runs" in args[1]:
                return {
                    "check_runs": [
                        {
                            "name": "lint",
                            "status": "completed",
                            "conclusion": "skipped",
                            "app": {"id": 15368},
                        },
                        {
                            "name": "typecheck",
                            "status": "completed",
                            "conclusion": "neutral",
                            "app": {"id": 15368},
                        },
                    ]
                }
            raise AssertionError(f"unexpected gh call: {args}")

        monkeypatch.setattr("aragora.cli.commands.review_queue._gh_json", fake_gh_json)

        packet = _build_packet("7465", repo_override=None)
        direct = packet.check_surfaces["direct_commit_check_runs"]

        assert packet.checks_summary == "2/2 required green (direct check-runs fallback)"
        assert direct["required_contexts"] == ["lint", "typecheck"]
        assert direct["required_checks"] == [
            {"context": "lint", "app_id": 15368},
            {"context": "typecheck", "app_id": 15368},
        ]
        assert direct["successful_required_contexts"] == ["lint", "typecheck"]
        assert direct["non_success_required_contexts"] == []
        assert direct["required_contexts_satisfied"] is True
        assert packet.model_review_quorum["admin_squash_allowed"] is True

    def test_missing_check_rollup_fails_closed_when_required_app_binding_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pr_payload = _make_pr(number=7465, files=["docs/status/open.md"])
        pr_payload["statusCheckRollup"] = []
        pr_payload["comments"] = [
            _dogfood_comment("## Claude focused dogfood\npass"),
            {
                "author": {"login": "an0mium"},
                "body": "## Grok independent model review\nVerdict: approve.",
            },
        ]

        def fake_gh_json(args: list[str]) -> dict[str, Any]:
            if args[:2] == ["pr", "view"]:
                return pr_payload
            if args[:1] == ["api"] and "required_status_checks" in args[1]:
                return {
                    "contexts": [],
                    "checks": [{"context": "lint", "app_id": 15368}],
                    "strict": False,
                }
            if args[:1] == ["api"] and "check-runs" in args[1]:
                return {
                    "check_runs": [
                        {
                            "name": "lint",
                            "status": "completed",
                            "conclusion": "success",
                            "app": {"id": 99999},
                        },
                    ]
                }
            raise AssertionError(f"unexpected gh call: {args}")

        monkeypatch.setattr("aragora.cli.commands.review_queue._gh_json", fake_gh_json)

        packet = _build_packet("7465", repo_override=None)
        direct = packet.check_surfaces["direct_commit_check_runs"]

        assert direct["required_contexts"] == ["lint"]
        assert direct["required_checks"] == [{"context": "lint", "app_id": 15368}]
        assert direct["missing_required_contexts"] == ["lint"]
        assert direct["required_contexts_satisfied"] is False
        assert packet.model_review_quorum["admin_squash_allowed"] is False
        assert packet.model_review_quorum["status"] == "repair_or_wait"

    def test_missing_check_rollup_fails_closed_when_branch_protection_is_strict(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pr_payload = _make_pr(number=7465, files=["docs/status/open.md"])
        pr_payload["statusCheckRollup"] = []
        pr_payload["comments"] = [
            _dogfood_comment("## Claude focused dogfood\npass"),
            {
                "author": {"login": "an0mium"},
                "body": "## Grok independent model review\nVerdict: approve.",
            },
        ]

        def fake_gh_json(args: list[str]) -> dict[str, Any]:
            if args[:2] == ["pr", "view"]:
                return pr_payload
            if args[:1] == ["api"] and "required_status_checks" in args[1]:
                return {"contexts": ["lint", "typecheck"], "strict": True}
            if args[:1] == ["api"] and "check-runs" in args[1]:
                return {
                    "check_runs": [
                        {"name": "lint", "status": "completed", "conclusion": "success"},
                        {"name": "typecheck", "status": "completed", "conclusion": "success"},
                    ]
                }
            raise AssertionError(f"unexpected gh call: {args}")

        monkeypatch.setattr("aragora.cli.commands.review_queue._gh_json", fake_gh_json)

        packet = _build_packet("7465", repo_override=None)
        direct = packet.check_surfaces["direct_commit_check_runs"]

        assert packet.checks_summary == "no checks reported"
        assert direct["branch_protection_strict"] is True
        assert direct["successful_required_contexts"] == ["lint", "typecheck"]
        assert direct["required_contexts_satisfied"] is False
        assert packet.model_review_quorum["admin_squash_allowed"] is False
        assert packet.model_review_quorum["status"] == "repair_or_wait"
        assert "strict base freshness" in packet.check_surfaces["diagnosis"]

    def test_missing_check_rollup_fails_closed_when_base_ref_is_unknown(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pr_payload = _make_pr(number=7465, files=["docs/status/open.md"])
        pr_payload.pop("baseRefName", None)
        pr_payload["statusCheckRollup"] = []
        pr_payload["comments"] = [
            _dogfood_comment("## Claude focused dogfood\npass"),
            {
                "author": {"login": "an0mium"},
                "body": "## Grok independent model review\nVerdict: approve.",
            },
        ]

        def fake_gh_json(args: list[str]) -> dict[str, Any]:
            if args[:2] == ["pr", "view"]:
                return pr_payload
            if args[:1] == ["api"] and "required_status_checks" in args[1]:
                raise AssertionError("must not query a fabricated default base ref")
            if args[:1] == ["api"] and "check-runs" in args[1]:
                return {
                    "check_runs": [
                        {"name": "lint", "status": "completed", "conclusion": "success"},
                    ]
                }
            raise AssertionError(f"unexpected gh call: {args}")

        monkeypatch.setattr("aragora.cli.commands.review_queue._gh_json", fake_gh_json)

        packet = _build_packet("7465", repo_override=None)
        direct = packet.check_surfaces["direct_commit_check_runs"]

        assert direct["branch_protection_required_status_checks_available"] is False
        assert direct["required_contexts"] == []
        assert direct["required_contexts_satisfied"] is False
        assert packet.model_review_quorum["admin_squash_allowed"] is False
        assert packet.model_review_quorum["status"] == "repair_or_wait"

    def test_missing_check_rollup_fails_closed_when_required_status_fetch_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pr_payload = _make_pr(number=7465, files=["docs/status/open.md"])
        pr_payload["statusCheckRollup"] = []
        pr_payload["comments"] = [
            _dogfood_comment("## Claude focused dogfood\npass"),
            {
                "author": {"login": "an0mium"},
                "body": "## Grok independent model review\nVerdict: approve.",
            },
        ]

        def fake_gh_json(args: list[str]) -> dict[str, Any]:
            if args[:2] == ["pr", "view"]:
                return pr_payload
            if args[:1] == ["api"] and "required_status_checks" in args[1]:
                raise RuntimeError("branch protection endpoint timed out")
            if args[:1] == ["api"] and "check-runs" in args[1]:
                return {
                    "check_runs": [
                        {"name": "lint", "status": "completed", "conclusion": "success"},
                    ]
                }
            raise AssertionError(f"unexpected gh call: {args}")

        monkeypatch.setattr("aragora.cli.commands.review_queue._gh_json", fake_gh_json)

        packet = _build_packet("7465", repo_override=None)
        direct = packet.check_surfaces["direct_commit_check_runs"]

        assert packet.checks_summary == "no checks reported"
        assert packet.risk_flags == ["check rollup unavailable"]
        assert direct["branch_protection_required_status_checks_available"] is False
        assert direct["required_contexts"] == []
        assert direct["required_contexts_satisfied"] is False
        assert packet.model_review_quorum["admin_squash_allowed"] is False
        assert packet.model_review_quorum["status"] == "repair_or_wait"

    def test_missing_check_rollup_fails_closed_when_direct_check_fetch_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pr_payload = _make_pr(number=7465, files=["docs/status/open.md"])
        pr_payload["statusCheckRollup"] = []
        pr_payload["comments"] = [
            _dogfood_comment("## Claude focused dogfood\npass"),
            {
                "author": {"login": "an0mium"},
                "body": "## Grok independent model review\nVerdict: approve.",
            },
        ]

        def fake_gh_json(args: list[str]) -> dict[str, Any]:
            if args[:2] == ["pr", "view"]:
                return pr_payload
            if args[:1] == ["api"] and "required_status_checks" in args[1]:
                return {
                    "contexts": ["lint"],
                    "checks": [{"context": "lint", "app_id": None}],
                    "strict": False,
                }
            if args[:1] == ["api"] and "check-runs" in args[1]:
                raise RuntimeError("check-runs endpoint timed out")
            raise AssertionError(f"unexpected gh call: {args}")

        monkeypatch.setattr("aragora.cli.commands.review_queue._gh_json", fake_gh_json)

        packet = _build_packet("7465", repo_override=None)
        direct = packet.check_surfaces["direct_commit_check_runs"]

        assert packet.checks_summary == "no checks reported"
        assert packet.risk_flags == ["check rollup unavailable"]
        assert direct["branch_protection_required_status_checks_available"] is True
        assert direct["required_contexts"] == ["lint"]
        assert direct["missing_required_contexts"] == ["lint"]
        assert direct["required_contexts_satisfied"] is False
        assert packet.model_review_quorum["admin_squash_allowed"] is False
        assert packet.model_review_quorum["status"] == "repair_or_wait"

    def test_missing_check_rollup_fails_closed_when_required_context_missing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pr_payload = _make_pr(number=7465, files=["docs/status/open.md"])
        pr_payload["statusCheckRollup"] = []
        pr_payload["comments"] = [
            _dogfood_comment("## Claude focused dogfood\npass"),
            {
                "author": {"login": "an0mium"},
                "body": "## Grok independent model review\nVerdict: approve.",
            },
        ]

        def fake_gh_json(args: list[str]) -> dict[str, Any]:
            if args[:2] == ["pr", "view"]:
                return pr_payload
            if args[:1] == ["api"] and "required_status_checks" in args[1]:
                return {"contexts": ["lint", "typecheck"]}
            if args[:1] == ["api"] and "check-runs" in args[1]:
                return {
                    "check_runs": [
                        {"name": "lint", "status": "completed", "conclusion": "success"},
                    ]
                }
            raise AssertionError(f"unexpected gh call: {args}")

        monkeypatch.setattr("aragora.cli.commands.review_queue._gh_json", fake_gh_json)

        packet = _build_packet("7465", repo_override=None)
        direct = packet.check_surfaces["direct_commit_check_runs"]

        assert packet.checks_summary == "no checks reported"
        assert direct["missing_required_contexts"] == ["typecheck"]
        assert direct["required_contexts_satisfied"] is False
        assert packet.model_review_quorum["admin_squash_allowed"] is False
        assert packet.model_review_quorum["status"] == "repair_or_wait"
        assert (
            "checks are unavailable; wait for GitHub check rollup before settlement"
            in packet.model_review_quorum["reasons"]
        )

    def test_missing_check_rollup_fails_closed_when_required_context_not_successful(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pr_payload = _make_pr(number=7465, files=["docs/status/open.md"])
        pr_payload["statusCheckRollup"] = []
        pr_payload["comments"] = [
            _dogfood_comment("## Claude focused dogfood\npass"),
            {
                "author": {"login": "an0mium"},
                "body": "## Grok independent model review\nVerdict: approve.",
            },
        ]

        def fake_gh_json(args: list[str]) -> dict[str, Any]:
            if args[:2] == ["pr", "view"]:
                return pr_payload
            if args[:1] == ["api"] and "required_status_checks" in args[1]:
                return {"contexts": ["lint", "typecheck"]}
            if args[:1] == ["api"] and "check-runs" in args[1]:
                return {
                    "check_runs": [
                        {"name": "lint", "status": "completed", "conclusion": "success"},
                        {
                            "name": "typecheck",
                            "status": "completed",
                            "conclusion": "failure",
                        },
                    ]
                }
            raise AssertionError(f"unexpected gh call: {args}")

        monkeypatch.setattr("aragora.cli.commands.review_queue._gh_json", fake_gh_json)

        packet = _build_packet("7465", repo_override=None)
        direct = packet.check_surfaces["direct_commit_check_runs"]

        assert packet.checks_summary == "no checks reported"
        assert direct["non_success_required_contexts"] == ["typecheck"]
        assert direct["required_contexts_satisfied"] is False
        assert packet.model_review_quorum["admin_squash_allowed"] is False
        assert packet.model_review_quorum["status"] == "repair_or_wait"

    def test_missing_check_rollup_fails_closed_without_required_contexts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pr_payload = _make_pr(number=7465, files=["docs/status/open.md"])
        pr_payload["statusCheckRollup"] = []
        pr_payload["comments"] = [
            _dogfood_comment("## Claude focused dogfood\npass"),
            {
                "author": {"login": "an0mium"},
                "body": "## Grok independent model review\nVerdict: approve.",
            },
        ]

        def fake_gh_json(args: list[str]) -> dict[str, Any]:
            if args[:2] == ["pr", "view"]:
                return pr_payload
            if args[:1] == ["api"] and "required_status_checks" in args[1]:
                return {"contexts": []}
            if args[:1] == ["api"] and "check-runs" in args[1]:
                return {
                    "check_runs": [
                        {"name": "lint", "status": "completed", "conclusion": "success"},
                    ]
                }
            raise AssertionError(f"unexpected gh call: {args}")

        monkeypatch.setattr("aragora.cli.commands.review_queue._gh_json", fake_gh_json)

        packet = _build_packet("7465", repo_override=None)
        direct = packet.check_surfaces["direct_commit_check_runs"]

        assert direct["required_contexts"] == []
        assert direct["required_contexts_satisfied"] is False
        assert packet.model_review_quorum["admin_squash_allowed"] is False
        assert packet.model_review_quorum["status"] == "repair_or_wait"

    def test_cancelled_merge_quorum_blocks_admin_squash_authorization(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pr_payload = _make_pr(
            number=7456,
            files=["docs/status/open.md"],
            checks=[
                {
                    "name": "aragora-merge-quorum",
                    "workflowName": "Aragora Merge Quorum",
                    "status": "COMPLETED",
                    "conclusion": "CANCELLED",
                    "completedAt": "2026-05-27T17:13:53Z",
                },
                {
                    "name": "lint",
                    "workflowName": "Tests",
                    "status": "COMPLETED",
                    "conclusion": "SUCCESS",
                    "completedAt": "2026-05-27T17:14:53Z",
                },
            ],
        )
        pr_payload["comments"] = [
            {
                "author": {"login": "an0mium"},
                "body": "## Grok independent model review\nVerdict: approve.",
            },
        ]
        monkeypatch.setattr(
            "aragora.cli.commands.review_queue._gh_json",
            lambda args: pr_payload,
        )
        packet = _build_packet("7456", repo_override=None)
        quorum = packet.model_review_quorum

        assert packet.checks_summary == "1 failing / 2 total"
        assert packet.machine_recommendation == "repair_first"
        assert quorum["admin_squash_allowed"] is False
        assert quorum["status"] == "repair_or_wait"
        assert "checks are failing; repair before settlement" in quorum["reasons"]

    def test_inconsistent_open_state_with_merged_at_fails_closed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pr_payload = _make_pr(
            number=7473,
            files=["docs/status/stale.md"],
            merged_at="2026-05-27T00:19:36Z",
        )
        pr_payload["comments"] = [
            {
                "author": {"login": "an0mium"},
                "body": "## Grok independent model review\nVerdict: approve.",
            },
        ]
        monkeypatch.setattr(
            "aragora.cli.commands.review_queue._gh_json",
            lambda args: pr_payload,
        )
        packet = _build_packet("7473", repo_override=None)

        assert packet.model_review_quorum["admin_squash_allowed"] is False
        assert packet.model_review_quorum["status"] == "repair_or_wait"
        assert (
            "PR state is OPEN but mergedAt is set; settlement applies only to open unmerged PRs"
        ) in packet.model_review_quorum["reasons"]

    def test_merge_packet_omits_merged_pr_from_admin_squash_order(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pr_payload = _make_pr(
            number=7470,
            files=["docs/status/focus.md"],
            state="MERGED",
            merged_at="2026-05-27T00:19:36Z",
        )
        pr_payload["comments"] = [
            {
                "author": {"login": "an0mium"},
                "body": "## Grok independent model review\nVerdict: approve.",
            },
        ]
        monkeypatch.setattr(
            "aragora.cli.commands.review_queue._build_queue",
            lambda limit: [_classify_pr(_make_pr(number=7470))],
        )
        monkeypatch.setattr(
            "aragora.cli.commands.review_queue._gh_json",
            lambda args: pr_payload,
        )

        packet = _build_merge_authorization_packet(
            pr_refs=["7470"],
            limit=10,
            repo_override=None,
        )

        assert packet["entries"][0]["status"] == "repair_or_wait"
        assert packet["entries"][0]["admin_squash_allowed"] is False
        assert packet["admin_squash_order"] == []
        assert packet["not_ready"] == [7470]

    def test_merged_pr_with_exact_settlement_receipt_is_settled_noop(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        pr_payload = _make_pr(
            number=7447,
            files=["aragora/cli/commands/review_queue.py"],
            state="MERGED",
            merged_at="2026-05-27T02:57:31Z",
        )
        review_queue_root = tmp_path / "review-queue"
        _write_admin_squash_receipt(
            review_queue_root,
            pr_number=7447,
            head_sha=str(pr_payload["headRefOid"]),
        )
        monkeypatch.setattr(
            "aragora.cli.commands.review_queue._build_queue",
            lambda limit: [_classify_pr(_make_pr(number=7447))],
        )
        monkeypatch.setattr(
            "aragora.cli.commands.review_queue._gh_json",
            lambda args: pr_payload,
        )

        packet = _build_merge_authorization_packet(
            pr_refs=["7447"],
            limit=10,
            repo_override=None,
            review_queue_root=review_queue_root,
        )

        entry = packet["entries"][0]
        assert entry["status"] == "settled"
        assert entry["verdict"] == "already_merged_settlement_recorded"
        assert entry["admin_squash_allowed"] is False
        assert entry["requires_human_risk_settlement"] is False
        assert entry["reasons"] == [
            "workflow/deploy/destructive surface touched",
            "exact-head admin_squash_merge settlement receipt recorded",
        ]
        assert packet["admin_squash_order"] == []
        assert packet["human_risk_settlement_required"] == []
        assert packet["not_ready"] == []

    @pytest.mark.parametrize("github_event", ["APPROVE", "RECORDED_EXTERNAL_APPROVE"])
    def test_open_tier_three_with_exact_human_risk_receipt_is_authorized(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        github_event: str,
    ) -> None:
        pr_payload = _make_pr(
            number=7466,
            files=["aragora/reputation/store.py"],
        )
        pr_payload["comments"] = [
            _dogfood_comment("## Claude focused dogfood\npass"),
            {
                "author": {"login": "an0mium"},
                "body": "## Grok independent model review\nVerdict: approve.",
            },
        ]
        review_queue_root = tmp_path / "review-queue"
        _write_human_risk_settlement_receipt(
            review_queue_root,
            pr_number=7466,
            head_sha=str(pr_payload["headRefOid"]),
            github_event=github_event,
        )
        monkeypatch.setattr(
            "aragora.cli.commands.review_queue._build_queue",
            lambda limit: [_classify_pr(_make_pr(number=7466))],
        )
        monkeypatch.setattr(
            "aragora.cli.commands.review_queue._gh_json",
            lambda args: pr_payload,
        )

        packet = _build_merge_authorization_packet(
            pr_refs=["7466"],
            limit=10,
            repo_override=None,
            review_queue_root=review_queue_root,
        )

        entry = packet["entries"][0]
        assert entry["status"] == "satisfied"
        assert entry["verdict"] == "admin_squash_allowed"
        assert entry["admin_squash_allowed"] is True
        assert entry["requires_human_risk_settlement"] is False
        assert entry["reasons"] == [
            "semantic, persistence, security, API, or SDK surface touched",
            "exact-head human risk settlement receipt recorded",
        ]
        assert packet["admin_squash_order"] == [7466]
        assert packet["human_risk_settlement_required"] == []
        assert packet["not_ready"] == []

    @pytest.mark.parametrize(
        ("head_sha", "action", "github_event", "payload_pr_number"),
        [
            ("stale-head-sha", "approve", "APPROVE", None),
            (None, "request_changes", "APPROVE", None),
            (None, "approve", "COMMENT", None),
            (None, "approve", "APPROVE", 9999),
        ],
    )
    def test_open_tier_three_with_invalid_human_risk_receipt_stays_blocked(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        head_sha: str | None,
        action: str,
        github_event: str,
        payload_pr_number: int | None,
    ) -> None:
        pr_payload = _make_pr(
            number=7466,
            files=["aragora/reputation/store.py"],
        )
        pr_payload["comments"] = [
            _dogfood_comment("## Claude focused dogfood\npass"),
            {
                "author": {"login": "an0mium"},
                "body": "## Grok independent model review\nVerdict: approve.",
            },
        ]
        review_queue_root = tmp_path / "review-queue"
        _write_human_risk_settlement_receipt(
            review_queue_root,
            pr_number=7466,
            head_sha=head_sha or str(pr_payload["headRefOid"]),
            action=action,
            github_event=github_event,
            payload_pr_number=payload_pr_number,
        )
        monkeypatch.setattr(
            "aragora.cli.commands.review_queue._build_queue",
            lambda limit: [_classify_pr(_make_pr(number=7466))],
        )
        monkeypatch.setattr(
            "aragora.cli.commands.review_queue._gh_json",
            lambda args: pr_payload,
        )

        packet = _build_merge_authorization_packet(
            pr_refs=["7466"],
            limit=10,
            repo_override=None,
            review_queue_root=review_queue_root,
        )

        entry = packet["entries"][0]
        assert entry["status"] == "human_risk_settlement_required"
        assert entry["admin_squash_allowed"] is False
        assert entry["requires_human_risk_settlement"] is True
        assert packet["admin_squash_order"] == []
        assert packet["human_risk_settlement_required"] == [7466]
        assert packet["not_ready"] == []

    def test_merged_pr_with_stale_settlement_receipt_stays_fail_closed(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        pr_payload = _make_pr(
            number=7447,
            files=["docs/status/focus.md"],
            state="MERGED",
            merged_at="2026-05-27T02:57:31Z",
        )
        review_queue_root = tmp_path / "review-queue"
        _write_admin_squash_receipt(
            review_queue_root,
            pr_number=7447,
            head_sha="stale-head-sha",
        )
        monkeypatch.setattr(
            "aragora.cli.commands.review_queue._gh_json",
            lambda args: pr_payload,
        )

        packet = _build_packet(
            "7447",
            repo_override=None,
            review_queue_root=review_queue_root,
        )

        quorum = packet.model_review_quorum
        assert quorum["status"] == "repair_or_wait"
        assert quorum["verdict"] == "not_ready_for_settlement"
        assert quorum["admin_squash_allowed"] is False
        assert "PR is MERGED; settlement applies only to open PRs" in quorum["reasons"]

    def test_build_packet_preserves_completed_merge_quorum_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pr_payload = _make_pr(
            number=6281,
            checks=[
                {
                    "name": "aragora-merge-quorum",
                    "workflowName": "Aragora Merge Quorum",
                    "status": "COMPLETED",
                    "conclusion": "FAILURE",
                },
                {"name": "lint", "status": "COMPLETED", "conclusion": "SUCCESS"},
            ],
            files=["scripts/build_next_prompt.py"],
        )
        monkeypatch.setattr(
            "aragora.cli.commands.review_queue._gh_json",
            lambda args: pr_payload,
        )
        packet = _build_packet("6281", repo_override=None)
        assert packet.machine_recommendation == "repair_first"
        assert packet.checks_summary == "1 failing / 2 total"
        assert "checks failing (1 failing / 2 total)" in packet.risk_flags
        assert (
            "checks are failing; repair before settlement" in packet.model_review_quorum["reasons"]
        )

    def test_build_packet_preserves_non_self_check_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pr_payload = _make_pr(
            number=6282,
            checks=[
                {
                    "name": "aragora-merge-quorum",
                    "status": "COMPLETED",
                    "conclusion": "FAILURE",
                },
                {"name": "lint", "status": "COMPLETED", "conclusion": "FAILURE"},
                {"name": "typecheck", "status": "COMPLETED", "conclusion": "SUCCESS"},
            ],
            files=["scripts/build_next_prompt.py"],
        )
        monkeypatch.setattr(
            "aragora.cli.commands.review_queue._gh_json",
            lambda args: pr_payload,
        )
        packet = _build_packet("6282", repo_override=None)
        assert packet.machine_recommendation == "repair_first"
        assert packet.checks_summary == "2 failing / 3 total"
        assert "checks failing (2 failing / 3 total)" in packet.risk_flags
        assert (
            "checks are failing; repair before settlement" in packet.model_review_quorum["reasons"]
        )

    def test_build_packet_flags_high_risk_paths(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pr_payload = _make_pr(
            number=42,
            files=["aragora/security/encryption.py", "aragora/cli/commands/review_pr.py"],
        )
        monkeypatch.setattr(
            "aragora.cli.commands.review_queue._gh_json",
            lambda args: pr_payload,
        )
        packet = _build_packet("42", repo_override=None)
        assert packet.machine_recommendation == "needs_human_attention"
        assert "aragora/security/encryption.py" in packet.high_risk_paths_touched

    def test_build_packet_failures_recommend_repair(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pr_payload = _make_pr(
            number=99,
            checks=[{"status": "COMPLETED", "conclusion": "FAILURE"}],
        )
        monkeypatch.setattr(
            "aragora.cli.commands.review_queue._gh_json",
            lambda args: pr_payload,
        )
        packet = _build_packet("99", repo_override=None)
        assert packet.machine_recommendation == "repair_first"

    def test_build_packet_draft_needs_attention(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pr_payload = _make_pr(number=101, is_draft=True)
        monkeypatch.setattr(
            "aragora.cli.commands.review_queue._gh_json",
            lambda args: pr_payload,
        )
        packet = _build_packet("101", repo_override=None)
        assert packet.machine_recommendation == "needs_human_attention"
        assert "draft" in packet.machine_recommendation_reason.lower()
        assert "draft PR" in packet.risk_flags

    def test_build_packet_parked_label_needs_attention(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pr_payload = _make_pr(number=102, labels=["blocked"])
        monkeypatch.setattr(
            "aragora.cli.commands.review_queue._gh_json",
            lambda args: pr_payload,
        )
        packet = _build_packet("102", repo_override=None)
        assert packet.machine_recommendation == "needs_human_attention"
        assert "parked label" in packet.machine_recommendation_reason.lower()
        assert "parked label (blocked)" in packet.risk_flags

    def test_build_packet_pending_checks_need_attention(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pr_payload = _make_pr(
            number=100,
            checks=[{"status": "IN_PROGRESS", "conclusion": ""}],
        )
        monkeypatch.setattr(
            "aragora.cli.commands.review_queue._gh_json",
            lambda args: pr_payload,
        )
        packet = _build_packet("100", repo_override=None)
        assert packet.machine_recommendation == "needs_human_attention"
        assert "checks still pending" in packet.machine_recommendation_reason

    def test_build_packet_state_based_pending_checks_need_attention(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pr_payload = _make_pr(
            number=100,
            checks=[{"context": "ci/unit", "state": "PENDING"}],
        )
        monkeypatch.setattr(
            "aragora.cli.commands.review_queue._gh_json",
            lambda args: pr_payload,
        )
        packet = _build_packet("100", repo_override=None)
        assert packet.machine_recommendation == "needs_human_attention"
        assert "checks still pending" in packet.machine_recommendation_reason

    def test_build_packet_raises_when_pr_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "aragora.cli.commands.review_queue._gh_json",
            lambda args: None,
        )
        with pytest.raises(_GhError, match="not found"):
            _build_packet("9999", repo_override=None)

    def test_build_packet_can_execute_live_reviewers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pr_payload = _make_pr(
            number=6280,
            files=["aragora/cli/commands/review_pr.py"],
        )
        monkeypatch.setattr(
            "aragora.cli.commands.review_queue._gh_json",
            lambda args: pr_payload,
        )
        monkeypatch.setattr(
            "aragora.cli.commands.review_queue._gh_text",
            lambda args: (
                "diff --git a/aragora/cli/commands/review_pr.py b/aragora/cli/commands/review_pr.py"
            ),
        )
        outputs = [
            _make_reviewer_output(
                slot_id="logic",
                provider="claude",
                family="claude",
                recommendation=Recommendation.APPROVE_CANDIDATE,
            ),
            _make_reviewer_output(
                slot_id="security",
                provider="openai-api",
                family="gpt",
                recommendation=Recommendation.REPAIR_FIRST,
            ),
            _make_reviewer_output(
                slot_id="maintainability",
                provider="gemini-cli",
                family="gemini",
                recommendation=Recommendation.APPROVE_CANDIDATE,
            ),
        ]
        monkeypatch.setattr(
            "aragora.swarm.pr_review_protocol.PRReviewProtocol.execute_live_reviewers",
            lambda self, **kwargs: (outputs, []),
        )

        packet = _build_packet("6280", repo_override=None, execute_reviewers=True)

        assert packet.protocol["status"] == EXECUTED_PROTOCOL_STATUS
        assert packet.protocol["validation_summary"]["reviewer_execution"]["reviewer_count"] == 3
        assert len(packet.protocol["dissenting_views"]) == 1
        assert packet.model_review_quorum["unresolved_dissent"] is True


# --- JSON output schema ----------------------------------------------------


class TestJsonOutput:
    def test_queue_item_to_dict_keys(self) -> None:
        item = _classify_pr(_make_pr())
        d = item.to_dict()
        for key in (
            "number",
            "title",
            "url",
            "head_sha",
            "author",
            "is_draft",
            "mergeable",
            "labels",
            "additions",
            "deletions",
            "changed_files",
            "checks_summary",
            "lane",
            "lane_reason",
        ):
            assert key in d, f"QueueItem dict missing key: {key}"

    def test_packet_to_dict_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "aragora.cli.commands.review_queue._gh_json",
            lambda args: _make_pr(number=1, files=["aragora/cli/main.py"]),
        )
        packet = _build_packet("1", repo_override=None)
        d = packet.to_dict()
        for key in (
            "pr_number",
            "title",
            "url",
            "head_sha",
            "base_sha",
            "author",
            "is_draft",
            "additions",
            "deletions",
            "changed_files",
            "queue_bucket",
            "touched_subsystems",
            "high_risk_paths_touched",
            "validation",
            "checks_summary",
            "check_surfaces",
            "risk_flags",
            "machine_recommendation",
            "machine_recommendation_reason",
            "packet_sha",
            "generated_at",
            "protocol",
            "model_review_quorum",
            "advisory_only",
            "settlement_note",
        ):
            assert key in d, f"ReviewPacket dict missing key: {key}"
        # ReviewPacket.advisory_only must always be True (signature property).
        assert d["advisory_only"] is True
        assert d["protocol"]["binding"]["repo"] == "synaptent/aragora"
        assert d["model_review_quorum"]["version"] == "model_review_quorum.v1"

    def test_packet_json_round_trip(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "aragora.cli.commands.review_queue._gh_json",
            lambda args: _make_pr(number=1, files=["aragora/cli/main.py"]),
        )
        packet = _build_packet("1", repo_override=None)
        roundtrip = json.loads(json.dumps(packet.to_dict()))
        assert roundtrip["pr_number"] == 1
        assert roundtrip["advisory_only"] is True
        assert roundtrip["protocol"]["protocol_version"] == "pr_review_protocol.v1"
        assert "model_review_quorum" in roundtrip


# --- cmd_review_queue dispatch + parser ------------------------------------


class TestCommandDispatch:
    def test_parser_registers_build_packet_run_and_act(self) -> None:
        root = argparse.ArgumentParser()
        sub = root.add_subparsers()
        add_review_queue_parser(sub)
        # build invocation parses
        ns_build = root.parse_args(["review-queue", "build", "--limit", "5", "--json"])
        assert ns_build.review_queue_command == "build"
        assert ns_build.limit == 5
        assert ns_build.json is True
        # packet invocation parses
        ns_packet = root.parse_args(
            ["review-queue", "packet", "6280", "--execute-reviewers", "--json"]
        )
        assert ns_packet.review_queue_command == "packet"
        assert ns_packet.pr == "6280"
        assert ns_packet.execute_reviewers is True
        # merge-packet invocation parses
        ns_merge_packet = root.parse_args(["review-queue", "merge-packet", "--pr", "6280"])
        assert ns_merge_packet.review_queue_command == "merge-packet"
        assert ns_merge_packet.pr == ["6280"]
        ns_merge_packet_root = root.parse_args(
            [
                "review-queue",
                "merge-packet",
                "--pr",
                "6280",
                "--review-queue-root",
                "/tmp/review-queue",
            ]
        )
        assert ns_merge_packet_root.review_queue_command == "merge-packet"
        assert ns_merge_packet_root.review_queue_root == "/tmp/review-queue"
        # evidence-lint invocation parses
        ns_evidence_lint = root.parse_args(
            [
                "review-queue",
                "evidence-lint",
                "--pr",
                "6280",
                "--head-sha",
                "headsha123",
                "--body",
                "## Codex focused dogfood\nCurrent head: headsha123",
                "--json",
            ]
        )
        assert ns_evidence_lint.review_queue_command == "evidence-lint"
        assert ns_evidence_lint.pr == "6280"
        assert ns_evidence_lint.head_sha == "headsha123"
        assert ns_evidence_lint.body_file is None
        assert ns_evidence_lint.json is True
        # run invocation parses
        ns_run = root.parse_args(["review-queue", "run", "--limit", "3", "--ready-only"])
        assert ns_run.review_queue_command == "run"
        assert ns_run.limit == 3
        assert ns_run.ready_only is True
        # health invocation parses through the standalone command parser
        ns_health = root.parse_args(["review-queue", "health", "--json"])
        assert ns_health.review_queue_command == "health"
        assert ns_health.json_output is True
        # health-alert invocation parses through the standalone command parser
        ns_alert = root.parse_args(["review-queue", "health-alert", "--heartbeat", "--json"])
        assert ns_alert.review_queue_command == "health-alert"
        assert ns_alert.heartbeat is True
        assert ns_alert.json_output is True
        # act invocation parses
        ns_act = root.parse_args(
            ["review-queue", "act", "6280", "--request-changes", "--reason", "needs a test"]
        )
        assert ns_act.review_queue_command == "act"
        assert ns_act.pr == "6280"
        assert ns_act.request_changes is True
        assert ns_act.reason == "needs a test"
        # local-only external settlement recording parses
        ns_record = root.parse_args(
            [
                "review-queue",
                "record-settlement",
                "6280",
                "--head-sha",
                "headsha123",
                "--action",
                "admin_squash_merge",
                "--reason",
                "operator authorized exact-head merge",
                "--apply-post-merge-lane-audit",
                "--json",
            ]
        )
        assert ns_record.review_queue_command == "record-settlement"
        assert ns_record.pr == "6280"
        assert ns_record.head_sha == "headsha123"
        assert ns_record.action == "admin_squash_merge"
        assert ns_record.reason == "operator authorized exact-head merge"
        assert ns_record.apply_post_merge_lane_audit is True

    def test_cmd_review_queue_with_no_subcommand_returns_2(self) -> None:
        ns = argparse.Namespace(review_queue_command=None)
        rc = cmd_review_queue(ns)
        assert rc == 2

    def test_top_level_parser_registers_record_settlement(self) -> None:
        from aragora.cli.parser import build_parser

        parser = build_parser()
        ns = parser.parse_args(
            [
                "review-queue",
                "record-settlement",
                "6280",
                "--head-sha",
                "headsha123",
                "--action",
                "admin_squash_merge",
                "--reason",
                "operator authorized exact-head merge",
                "--apply-post-merge-lane-audit",
                "--json",
            ]
        )

        assert ns.command == "review-queue"
        assert ns.review_queue_command == "record-settlement"
        assert ns.pr == "6280"
        assert ns.head_sha == "headsha123"
        assert ns.action == "admin_squash_merge"
        assert ns.reason == "operator authorized exact-head merge"
        assert ns.apply_post_merge_lane_audit is True
        assert ns.json_output is True

    def test_top_level_parser_registers_evidence_lint(self) -> None:
        from aragora.cli.parser import build_parser

        parser = build_parser()
        ns = parser.parse_args(
            [
                "review-queue",
                "evidence-lint",
                "--pr",
                "7445",
                "--head-sha",
                "cd87c5a1b2db34f04167906553502db3ede9525e",
                "--body",
                "## Claude review\nCurrent head: cd87c5a1b2db34f04167906553502db3ede9525e",
                "--json",
            ]
        )

        assert ns.command == "review-queue"
        assert ns.review_queue_command == "evidence-lint"
        assert ns.pr == "7445"
        assert ns.head_sha == "cd87c5a1b2db34f04167906553502db3ede9525e"
        assert ns.body_file is None
        assert ns.json_output is True

    def test_evidence_lint_counts_current_head_dogfood(self) -> None:
        ns = argparse.Namespace(
            review_queue_command="evidence-lint",
            pr="7445",
            head_sha="cd87c5a1b2db34f04167906553502db3ede9525e",
            head_committed_at="2026-05-23T19:00:00Z",
            body=_codex_openai_body(
                body=(
                    "Current head: cd87c5a1b2db34f04167906553502db3ede9525e\n"
                    "Validation passed for the exact touched surface."
                )
            ),
            body_file=None,
            author="an0mium",
            json=True,
        )

        out = io.StringIO()
        with redirect_stdout(out):
            rc = cmd_review_queue(ns)

        payload = json.loads(out.getvalue())
        assert rc == 0
        assert payload["mode"] == "evidence_lint"
        assert payload["would_count"] is True
        assert payload["counted_reviewer_ids"] == ["openai"]
        assert payload["dogfood_evidence"][0]["reviewer_id"] == "openai"
        assert payload["current_head_grounding_method"] == "head_sha_citation"
        assert payload["problems"] == []

    def test_evidence_lint_rejects_github_actions_bot_author(self) -> None:
        ns = argparse.Namespace(
            review_queue_command="evidence-lint",
            pr="7445",
            head_sha="cd87c5a1b2db34f04167906553502db3ede9525e",
            head_committed_at="2026-05-23T19:00:00Z",
            body=_codex_openai_body(
                body=(
                    "Current head: cd87c5a1b2db34f04167906553502db3ede9525e\n"
                    "Automated structured evidence must remain advisory-only."
                )
            ),
            body_file=None,
            author="github-actions[bot]",
            json=True,
        )

        out = io.StringIO()
        with redirect_stdout(out):
            rc = cmd_review_queue(ns)

        payload = json.loads(out.getvalue())
        assert rc == 1
        assert payload["would_count"] is False
        assert payload["reviewer_signals"] == []
        assert payload["dogfood_evidence"] == []
        assert "github_actions_author_not_counted" in payload["problems"]
        assert "no_counted_model_family" in payload["problems"]

    def test_evidence_lint_rejects_ungrounded_comment(self) -> None:
        ns = argparse.Namespace(
            review_queue_command="evidence-lint",
            pr="7445",
            head_sha="cd87c5a1b2db34f04167906553502db3ede9525e",
            head_committed_at="2026-05-23T19:00:00Z",
            body="## Claude review\n\nLooks good, but no exact head citation.",
            body_file=None,
            author="an0mium",
            json=True,
        )

        out = io.StringIO()
        with redirect_stdout(out):
            rc = cmd_review_queue(ns)

        payload = json.loads(out.getvalue())
        assert rc == 1
        assert payload["would_count"] is False
        assert payload["current_head_grounding_method"] == "missing_head_sha_citation"
        assert "missing_current_head_grounding" in payload["problems"]
        assert "no_counted_model_reviewer" in payload["problems"]

    def test_evidence_lint_requires_head_sha_when_timestamp_omitted(self) -> None:
        ns = argparse.Namespace(
            review_queue_command="evidence-lint",
            pr="7445",
            head_sha="cd87c5a1b2db34f04167906553502db3ede9525e",
            head_committed_at="",
            body="## Codex focused dogfood\n\nValidation passed, but no SHA citation.",
            body_file=None,
            author="an0mium",
            json=True,
        )

        out = io.StringIO()
        with redirect_stdout(out):
            rc = cmd_review_queue(ns)

        payload = json.loads(out.getvalue())
        assert rc == 1
        assert payload["would_count"] is False
        assert payload["current_head_grounding_method"] == "missing_head_sha_citation"
        assert "missing_current_head_grounding" in payload["problems"]

    def test_evidence_lint_reads_body_file(self, tmp_path: Path) -> None:
        body_file = tmp_path / "comment.md"
        body_file.write_text(
            "## Claude review\n\n"
            "Current head: cd87c5a1b2db34f04167906553502db3ede9525e\n"
            "I reviewed the exact-head evidence-lint diff.",
            encoding="utf-8",
        )
        ns = argparse.Namespace(
            review_queue_command="evidence-lint",
            pr="7445",
            head_sha="cd87c5a1b2db34f04167906553502db3ede9525e",
            head_committed_at="2026-05-23T19:00:00Z",
            body=None,
            body_file=str(body_file),
            author="an0mium",
            json=True,
        )

        out = io.StringIO()
        with redirect_stdout(out):
            rc = cmd_review_queue(ns)

        payload = json.loads(out.getvalue())
        assert rc == 0
        assert payload["would_count"] is True
        assert payload["counted_reviewer_ids"] == ["claude"]
        assert payload["current_head_grounding_method"] == "head_sha_citation"
        assert payload["reviewer_signals"][0]["reviewer_id"] == "claude"


class TestSettlementHelpers:
    def test_requested_action(self) -> None:
        assert (
            _requested_action(argparse.Namespace(approve=True, request_changes=False, defer=False))
            == "approve"
        )
        assert (
            _requested_action(argparse.Namespace(approve=False, request_changes=True, defer=False))
            == "request_changes"
        )
        assert (
            _requested_action(argparse.Namespace(approve=False, request_changes=False, defer=True))
            == "defer"
        )

    def test_settle_packet_writes_receipt(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        recorded: list[list[str]] = []

        def _record_gh_text(args: list[str]) -> str:
            recorded.append(args)
            return ""

        monkeypatch.setattr(
            "aragora.cli.commands.review_queue._current_head_sha",
            lambda pr_number, repo_override=None: "headsha123",
        )
        monkeypatch.setattr(
            "aragora.cli.commands.review_queue._gh_text",
            _record_gh_text,
        )
        monkeypatch.setattr(
            "aragora.cli.commands.review_queue._github_actor",
            lambda: "an0mium",
        )
        packet = ReviewPacket(
            pr_number=6294,
            title="route PR-targeted handoffs out of boss queue",
            url="https://github.com/synaptent/aragora/pull/6294",
            head_sha="headsha123",
            base_sha="basesha123",
            author="codex",
            is_draft=False,
            additions=10,
            deletions=2,
            changed_files=1,
            queue_bucket="ready_now",
            touched_subsystems=["scripts"],
            high_risk_paths_touched=[],
            validation=["`python3 -m pytest -q tests/scripts/test_publish_automation_handoffs.py`"],
            checks_summary="5/5 green",
            risk_flags=[],
            machine_recommendation="approve_candidate",
            machine_recommendation_reason="all green, bounded diff, no high-risk paths",
            packet_sha="sha256:testpacket",
            generated_at="2026-04-19T05:00:00+00:00",
        )
        receipt = _settle_packet(
            packet=packet,
            action="approve",
            reason="looks bounded",
            repo_root=tmp_path,
            repo_override=None,
            session_id="session-1",
            elapsed_seconds=1.25,
        )
        assert recorded and "--approve" in recorded[0]
        assert receipt.actor == "an0mium"
        assert receipt.github_event == "APPROVE"
        assert receipt.receipt_path.endswith("pr-6294-session-1-approve.json")
        saved = json.loads(Path(receipt.receipt_path).read_text())
        assert saved["packet_sha"] == "sha256:testpacket"
        assert saved["reason"] == "looks bounded"

    def test_settle_packet_rejects_stale_head(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            "aragora.cli.commands.review_queue._current_head_sha",
            lambda pr_number, repo_override=None: "new-head",
        )
        packet = ReviewPacket(
            pr_number=1,
            title="stale",
            url="https://github.com/synaptent/aragora/pull/1",
            head_sha="old-head",
            base_sha="basesha123",
            author="codex",
            is_draft=False,
            additions=1,
            deletions=1,
            changed_files=1,
            queue_bucket="ready_now",
            touched_subsystems=["aragora/cli"],
            high_risk_paths_touched=[],
            validation=[],
            checks_summary="1/1 green",
            risk_flags=[],
            machine_recommendation="approve_candidate",
            machine_recommendation_reason="clean",
            packet_sha="sha256:testpacket",
            generated_at="2026-04-19T05:00:00+00:00",
        )
        with pytest.raises(_GhError, match="refresh the packet"):
            _settle_packet(
                packet=packet,
                action="approve",
                reason="",
                repo_root=tmp_path,
                repo_override=None,
                session_id="session-2",
            )

    def test_record_external_settlement_writes_local_receipt(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        def _fake_gh_json(args: list[str]) -> dict[str, Any]:
            if args[:2] == ["pr", "view"]:
                return {
                    "number": 6294,
                    "url": "https://github.com/synaptent/aragora/pull/6294",
                    "headRefOid": "headsha123",
                    "baseRefOid": "basesha123",
                    "state": "MERGED",
                    "mergedAt": "2026-05-10T08:00:00Z",
                }
            if args == ["api", "user"]:
                return {"login": "an0mium"}
            raise AssertionError(args)

        monkeypatch.setattr("aragora.cli.commands.review_queue._gh_json", _fake_gh_json)

        result = _record_external_settlement(
            pr_ref="6294",
            head_sha="headsha123",
            action="admin_squash_merge",
            reason="operator authorized exact-head merge",
            repo_root=tmp_path,
            repo_override=None,
            review_queue_root=None,
        )

        assert result.written is True
        assert result.idempotent is False
        assert result.receipt_sha256.startswith("sha256:")
        assert result.receipt.action == "admin_squash_merge"
        assert result.receipt.actor == "an0mium"
        assert result.receipt.reviewed_at == "2026-05-10T08:00:00+00:00"
        assert result.receipt.github_event == "ADMIN_SQUASH_MERGE"
        assert result.receipt.queue_bucket == "external_settlement"
        assert result.receipt.machine_recommendation == "operator_recorded_external_settlement"
        saved = json.loads(Path(result.receipt.receipt_path).read_text())
        assert saved["pr_number"] == 6294
        assert saved["head_sha"] == "headsha123"
        assert saved["packet_sha"].startswith("sha256:")

    def test_record_external_admin_merge_includes_post_merge_lane_audit(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        def _fake_gh_json(args: list[str]) -> dict[str, Any]:
            if args[:2] == ["pr", "view"]:
                return {
                    "number": 6294,
                    "url": "https://github.com/synaptent/aragora/pull/6294",
                    "headRefOid": "headsha123",
                    "baseRefOid": "basesha123",
                    "state": "MERGED",
                    "mergedAt": "2026-05-10T08:00:00Z",
                }
            if args == ["api", "user"]:
                return {"login": "an0mium"}
            raise AssertionError(args)

        monkeypatch.setattr("aragora.cli.commands.review_queue._gh_json", _fake_gh_json)
        audit_calls: list[tuple[int, bool]] = []

        def audit_provider(pr_number: int, apply: bool = False) -> dict[str, Any]:
            audit_calls.append((pr_number, apply))
            return {
                "finding_count": 1,
                "resolved_count": 0,
                "blocked_reason": None,
                "owner_steering_text": "",
                "owner_release_commands": [],
                "operator_apply_command": "python3 scripts/resolve_lane_conflicts.py --apply ...",
                "receipt_paths": [],
                "github_state": {"state": "MERGED", "mergeCommit": "merge-sha"},
                "audit_ok": True,
                "audit_applied": False,
            }

        result = _record_external_settlement(
            pr_ref="6294",
            head_sha="headsha123",
            action="admin_squash_merge",
            reason="operator authorized exact-head merge",
            repo_root=tmp_path,
            repo_override=None,
            review_queue_root=None,
            post_merge_lane_audit_provider=audit_provider,
        )

        assert audit_calls == [(6294, False)]
        assert result.post_merge_lane_audit is not None
        assert result.to_dict()["post_merge_lane_audit"]["operator_apply_command"].startswith(
            "python3 scripts/resolve_lane_conflicts.py"
        )
        saved = json.loads(Path(result.receipt.receipt_path).read_text())
        assert saved["post_merge_lane_audit"]["finding_count"] == 1

    def test_record_external_non_merge_settlement_does_not_run_post_merge_lane_audit(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        def _fake_gh_json(args: list[str]) -> dict[str, Any]:
            if args[:2] == ["pr", "view"]:
                return {
                    "number": 6294,
                    "url": "https://github.com/synaptent/aragora/pull/6294",
                    "headRefOid": "headsha123",
                    "baseRefOid": "basesha123",
                    "state": "OPEN",
                    "mergedAt": "",
                }
            if args == ["api", "user"]:
                return {"login": "an0mium"}
            raise AssertionError(args)

        monkeypatch.setattr("aragora.cli.commands.review_queue._gh_json", _fake_gh_json)

        def audit_provider(pr_number: int, apply: bool = False) -> dict[str, Any]:
            raise AssertionError("post-merge audit should only run for admin_squash_merge")

        result = _record_external_settlement(
            pr_ref="6294",
            head_sha="headsha123",
            action="comment",
            reason="operator recorded a comment",
            repo_root=tmp_path,
            repo_override=None,
            review_queue_root=None,
            post_merge_lane_audit_provider=audit_provider,
        )

        assert result.post_merge_lane_audit is None
        assert "post_merge_lane_audit" not in result.to_dict()

    def test_record_external_settlement_is_idempotent(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        def _fake_gh_json(args: list[str]) -> dict[str, Any]:
            if args[:2] == ["pr", "view"]:
                return {
                    "number": 6294,
                    "url": "https://github.com/synaptent/aragora/pull/6294",
                    "headRefOid": "headsha123",
                    "baseRefOid": "basesha123",
                    "state": "MERGED",
                    "mergedAt": "2026-05-10T08:00:00Z",
                }
            if args == ["api", "user"]:
                return {"login": "an0mium"}
            raise AssertionError(args)

        monkeypatch.setattr("aragora.cli.commands.review_queue._gh_json", _fake_gh_json)

        first = _record_external_settlement(
            pr_ref="6294",
            head_sha="headsha123",
            action="admin_squash_merge",
            reason="operator authorized exact-head merge",
            repo_root=tmp_path,
            repo_override=None,
            review_queue_root=None,
        )
        second = _record_external_settlement(
            pr_ref="6294",
            head_sha="headsha123",
            action="admin_squash_merge",
            reason="operator authorized exact-head merge",
            repo_root=tmp_path,
            repo_override=None,
            review_queue_root=None,
        )

        assert first.receipt.receipt_path == second.receipt.receipt_path
        assert second.written is False
        assert second.idempotent is True
        assert second.receipt_sha256 == first.receipt_sha256

    def test_record_external_settlement_rejects_conflicting_existing_receipt(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        def _fake_gh_json(args: list[str]) -> dict[str, Any]:
            if args[:2] == ["pr", "view"]:
                return {
                    "number": 6294,
                    "url": "https://github.com/synaptent/aragora/pull/6294",
                    "headRefOid": "headsha123",
                    "baseRefOid": "basesha123",
                    "state": "MERGED",
                    "mergedAt": "2026-05-10T08:00:00Z",
                }
            if args == ["api", "user"]:
                return {"login": "an0mium"}
            raise AssertionError(args)

        monkeypatch.setattr("aragora.cli.commands.review_queue._gh_json", _fake_gh_json)

        _record_external_settlement(
            pr_ref="6294",
            head_sha="headsha123",
            action="admin_squash_merge",
            reason="operator authorized exact-head merge",
            repo_root=tmp_path,
            repo_override=None,
            review_queue_root=None,
        )
        with pytest.raises(_GhError, match="conflicting settlement receipt"):
            _record_external_settlement(
                pr_ref="6294",
                head_sha="headsha123",
                action="admin_squash_merge",
                reason="different operator reason",
                repo_root=tmp_path,
                repo_override=None,
                review_queue_root=None,
            )

    def test_record_external_settlement_rejects_head_mismatch(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            "aragora.cli.commands.review_queue._gh_json",
            lambda args: {
                "number": 6294,
                "url": "https://github.com/synaptent/aragora/pull/6294",
                "headRefOid": "new-head",
                "baseRefOid": "basesha123",
                "state": "MERGED",
                "mergedAt": "2026-05-10T08:00:00Z",
            },
        )

        with pytest.raises(_GhError, match="exact externally settled head"):
            _record_external_settlement(
                pr_ref="6294",
                head_sha="headsha123",
                action="admin_squash_merge",
                reason="operator authorized exact-head merge",
                repo_root=tmp_path,
                repo_override=None,
                review_queue_root=None,
            )

    def test_record_external_settlement_rejects_unmerged_admin_merge(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            "aragora.cli.commands.review_queue._gh_json",
            lambda args: {
                "number": 6294,
                "url": "https://github.com/synaptent/aragora/pull/6294",
                "headRefOid": "headsha123",
                "baseRefOid": "basesha123",
                "state": "OPEN",
                "mergedAt": "",
            },
        )

        with pytest.raises(_GhError, match="require the PR to be MERGED"):
            _record_external_settlement(
                pr_ref="6294",
                head_sha="headsha123",
                action="admin_squash_merge",
                reason="operator authorized exact-head merge",
                repo_root=tmp_path,
                repo_override=None,
                review_queue_root=None,
            )

    def test_record_settlement_command_surfaces_github_unavailable(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            "aragora.cli.commands.review_queue.resolve_repo_root",
            lambda cwd: tmp_path,
        )
        monkeypatch.setattr(
            "aragora.cli.commands.review_queue._require_clean_worktree",
            lambda repo_root: None,
        )

        def _raise_gh(args: list[str]) -> None:
            raise _GhError("gh unavailable")

        monkeypatch.setattr("aragora.cli.commands.review_queue._gh_json", _raise_gh)
        ns = argparse.Namespace(
            review_queue_command="record-settlement",
            pr="6294",
            repo=None,
            head_sha="headsha123",
            action="admin_squash_merge",
            reason="operator authorized exact-head merge",
            review_queue_root=None,
            json=False,
        )

        err_buf = io.StringIO()
        with redirect_stderr(err_buf):
            rc = cmd_review_queue(ns)

        assert rc == 1
        assert "gh unavailable" in err_buf.getvalue()

    def test_record_settlement_command_records_audit_apply_failure_then_returns_1(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            "aragora.cli.commands.review_queue.resolve_repo_root",
            lambda cwd: tmp_path,
        )
        monkeypatch.setattr(
            "aragora.cli.commands.review_queue._require_clean_worktree",
            lambda repo_root: None,
        )

        def _fake_gh_json(args: list[str]) -> dict[str, Any]:
            if args[:2] == ["pr", "view"]:
                return {
                    "number": 6294,
                    "url": "https://github.com/synaptent/aragora/pull/6294",
                    "headRefOid": "headsha123",
                    "baseRefOid": "basesha123",
                    "state": "MERGED",
                    "mergedAt": "2026-05-10T08:00:00Z",
                }
            if args == ["api", "user"]:
                return {"login": "an0mium"}
            raise AssertionError(args)

        def _audit_failure(
            pr_number: int,
            *,
            repo_root: Path | None = None,
            apply: bool = False,
        ) -> dict[str, Any]:
            assert pr_number == 6294
            assert apply is True
            return {
                "finding_count": 1,
                "resolved_count": 0,
                "blocked_reason": "merge_commit_mismatch",
                "operator_apply_command": "python3 scripts/resolve_lane_conflicts.py --apply ...",
                "receipt_paths": [],
                "github_state": {"state": "MERGED", "mergeCommit": "merge-sha"},
                "audit_ok": False,
                "audit_applied": False,
                "audit_error": "merge_commit_mismatch",
            }

        monkeypatch.setattr("aragora.cli.commands.review_queue._gh_json", _fake_gh_json)
        monkeypatch.setattr(
            "aragora.cli.commands.review_queue.run_post_merge_lane_audit",
            _audit_failure,
        )
        ns = argparse.Namespace(
            review_queue_command="record-settlement",
            pr="6294",
            repo=None,
            head_sha="headsha123",
            action="admin_squash_merge",
            reason="operator authorized exact-head merge",
            review_queue_root=None,
            apply_post_merge_lane_audit=True,
            json=True,
        )

        out_buf = io.StringIO()
        with redirect_stdout(out_buf):
            rc = cmd_review_queue(ns)

        payload = json.loads(out_buf.getvalue())
        assert rc == 1
        assert payload["post_merge_lane_audit_failed"] is True
        assert payload["post_merge_lane_audit"]["blocked_reason"] == "merge_commit_mismatch"
        assert Path(payload["receipt_path"]).is_file()

    def test_build_command_renders_table(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "aragora.cli.commands.review_queue._gh_json",
            lambda args: [_make_pr(number=1)],
        )
        ns = argparse.Namespace(
            review_queue_command="build",
            limit=10,
            ready_only=False,
            include_parked=False,
            json=False,
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cmd_review_queue(ns)
        assert rc == 0
        out = buf.getvalue()
        assert "Review queue" in out
        assert "advisory only" in out

    def test_build_command_surfaces_active_auto_handle_drift_alerts(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "aragora.cli.commands.review_queue._gh_json",
            lambda args: [_make_pr(number=1)],
        )
        monkeypatch.setattr(
            "aragora.cli.commands.review_queue.AutoHandleCalibrationStore.list_active_alerts",
            lambda self, limit=3: [
                AutoHandleDriftAlert(
                    alert_id="alert-1",
                    auto_handle_path="fire_and_forget",
                    decision_class="tier=1|lanes=1|files=1|scope=aragora",
                    previous_success_rate=1.0,
                    current_success_rate=0.5,
                    window_days=30,
                    min_samples=20,
                    min_success_rate=0.95,
                    drift_threshold=0.05,
                    detected_at=0.0,
                    remediation_action="require_human_review_for_class",
                )
            ],
        )
        ns = argparse.Namespace(
            review_queue_command="build",
            limit=10,
            ready_only=False,
            include_parked=False,
            json=False,
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cmd_review_queue(ns)
        assert rc == 0
        out = buf.getvalue()
        assert "ACTIVE AUTO-HANDLE DRIFT ALERTS" in out
        assert "fire_and_forget" in out

    def test_build_command_warns_when_calibration_store_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "aragora.cli.commands.review_queue._gh_json",
            lambda args: [_make_pr(number=1)],
        )

        def _raise_store_error(self, limit=3):
            raise RuntimeError("db unavailable")

        monkeypatch.setattr(
            "aragora.cli.commands.review_queue.AutoHandleCalibrationStore.list_active_alerts",
            _raise_store_error,
        )
        ns = argparse.Namespace(
            review_queue_command="build",
            limit=10,
            ready_only=False,
            include_parked=False,
            json=False,
        )
        out_buf = io.StringIO()
        err_buf = io.StringIO()
        with redirect_stdout(out_buf), redirect_stderr(err_buf):
            rc = cmd_review_queue(ns)
        assert rc == 0
        assert "warning: auto-handle calibration unavailable: db unavailable" in err_buf.getvalue()

    def test_build_command_json_output(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "aragora.cli.commands.review_queue._gh_json",
            lambda args: [_make_pr(number=1), _make_pr(number=2)],
        )
        ns = argparse.Namespace(
            review_queue_command="build",
            limit=10,
            ready_only=False,
            include_parked=False,
            json=True,
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cmd_review_queue(ns)
        assert rc == 0
        payload = json.loads(buf.getvalue())
        assert isinstance(payload, list)
        assert {item["number"] for item in payload} == {1, 2}

    def test_packet_command_json_output(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "aragora.cli.commands.review_queue._gh_json",
            lambda args: _make_pr(number=42, files=["aragora/cli/main.py"]),
        )
        ns = argparse.Namespace(
            review_queue_command="packet",
            pr="42",
            repo=None,
            json=True,
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cmd_review_queue(ns)
        assert rc == 0
        payload = json.loads(buf.getvalue())
        assert payload["pr_number"] == 42
        assert payload["advisory_only"] is True
        assert payload["settlement_note"] == ADVISORY_NOTE

    def test_merge_packet_json_output_with_queue_pressure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        queue = [_classify_pr(_make_pr(number=i)) for i in range(1, MODEL_REVIEW_QUEUE_CAP + 2)]

        def _fake_build_packet(
            pr_ref: str,
            *,
            repo_override: str | None,
            execute_reviewers: bool = False,
        ) -> ReviewPacket:
            return ReviewPacket(
                pr_number=int(pr_ref),
                title="bounded docs",
                url=f"https://github.com/synaptent/aragora/pull/{pr_ref}",
                head_sha="headsha",
                base_sha="basesha",
                author="codex",
                is_draft=False,
                additions=1,
                deletions=1,
                changed_files=1,
                queue_bucket="ready_now",
                touched_subsystems=["docs"],
                high_risk_paths_touched=[],
                validation=[],
                checks_summary="5/5 green",
                risk_flags=[],
                machine_recommendation="approve_candidate",
                machine_recommendation_reason="clean",
                packet_sha="sha256:test",
                generated_at="2026-04-28T00:00:00+00:00",
                model_review_quorum={
                    "tier": 0,
                    "tier_name": "tier_0_docs_tests_status",
                    "status": "satisfied",
                    "verdict": "admin_squash_allowed",
                    "admin_squash_allowed": True,
                    "requires_human_risk_settlement": False,
                    "unresolved_dissent": False,
                    "reviewer_signals": [],
                    "dogfood_evidence": [{"reviewer_id": "claude"}],
                    "counted_reviewer_ids": ["claude"],
                    "reasons": ["docs/tests/status-only change"],
                },
            )

        monkeypatch.setattr("aragora.cli.commands.review_queue._build_queue", lambda limit: queue)
        monkeypatch.setattr(
            "aragora.cli.commands.review_queue._build_packet",
            _fake_build_packet,
        )
        ns = argparse.Namespace(
            review_queue_command="merge-packet",
            pr=[],
            repo=None,
            limit=10,
            execute_reviewers=False,
            json=True,
        )
        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = cmd_review_queue(ns)
        assert rc == 0
        payload = json.loads(buf.getvalue())
        assert payload["queue_pressure"]["active"] is True
        assert payload["queue_pressure"]["scope"] == "open_pr_queue"
        assert payload["admin_squash_order"] == list(range(1, MODEL_REVIEW_QUEUE_CAP + 2))
        assert payload["entries"][0]["verdict"] == "admin_squash_allowed"

    def test_act_command_requires_reason_for_request_changes(self) -> None:
        ns = argparse.Namespace(
            review_queue_command="act",
            pr="42",
            repo=None,
            approve=False,
            request_changes=True,
            defer=False,
            reason="",
            json=False,
        )
        rc = cmd_review_queue(ns)
        assert rc == 2
