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
    PARKED_LABELS,
    QueueItem,
    ReviewPacket,
    _build_packet,
    _build_queue,
    _classify_pr,
    _extract_validation_commands,
    _filter_lanes,
    _GhError,
    _is_high_risk_path,
    _parse_pr_number,
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
        "headRefName": f"branch-{number}",
        "headRefOid": f"sha{number:08d}",
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

    def test_no_checks(self) -> None:
        summary, has_fail, has_pending = _summarize_checks([])
        assert summary == "no checks"
        assert not has_fail
        assert not has_pending

    def test_malformed_check_ignored(self) -> None:
        checks: list[Any] = ["not a dict", None, {"status": "COMPLETED", "conclusion": "SUCCESS"}]
        summary, _, _ = _summarize_checks(checks)
        assert "1/1 green" in summary


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
        assert not _is_high_risk_path("docs/CI_LANES.md")
        assert not _is_high_risk_path("tests/cli/commands/test_review_queue.py")


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
            lambda args: "diff --git a/aragora/cli/commands/review_pr.py b/aragora/cli/commands/review_pr.py",
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
            "risk_flags",
            "machine_recommendation",
            "machine_recommendation_reason",
            "packet_sha",
            "generated_at",
            "protocol",
            "advisory_only",
            "settlement_note",
        ):
            assert key in d, f"ReviewPacket dict missing key: {key}"
        # ReviewPacket.advisory_only must always be True (signature property).
        assert d["advisory_only"] is True
        assert d["protocol"]["binding"]["repo"] == "synaptent/aragora"

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
        # run invocation parses
        ns_run = root.parse_args(["review-queue", "run", "--limit", "3", "--ready-only"])
        assert ns_run.review_queue_command == "run"
        assert ns_run.limit == 3
        assert ns_run.ready_only is True
        # act invocation parses
        ns_act = root.parse_args(
            ["review-queue", "act", "6280", "--request-changes", "--reason", "needs a test"]
        )
        assert ns_act.review_queue_command == "act"
        assert ns_act.pr == "6280"
        assert ns_act.request_changes is True
        assert ns_act.reason == "needs a test"

    def test_cmd_review_queue_with_no_subcommand_returns_2(self) -> None:
        ns = argparse.Namespace(review_queue_command=None)
        rc = cmd_review_queue(ns)
        assert rc == 2


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
