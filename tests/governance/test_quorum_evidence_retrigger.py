"""Governance tests for the quorum evidence re-trigger (B1).

These tests are the Tier 4 pre-approval regression target for the design
in ``docs/specs/QUORUM_EVIDENCE_RETRIGGER.md`` (root cause #1 in
``docs/governance/BOSS_LOOP_MERGE_GATE_RESILIENCE.md``: the enforcing
merge-quorum check always evaluates BEFORE evidence comments exist, so
every settlement pays a guaranteed stale failure plus a manual rerun).

They pin the structural contract of ``aragora-merge-quorum.yml``:

* ``issue_comment: [created]`` re-triggers evaluation when evidence
  comments arrive;
* the re-trigger path is guarded, debounced, and least-privileged;
* the enforcing evaluation job is untouched: no comment event reaches
  it, its permissions stay read-only, and its anti-doom-loop
  ``cancel-in-progress: false`` invariant is preserved.

The suite must FAIL against the pre-B1 workflow and PASS with the
change (RED/GREEN proof captured in the implementing PR).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

WORKFLOW_PATH = (
    Path(__file__).resolve().parents[2] / ".github" / "workflows" / "aragora-merge-quorum.yml"
)


@pytest.fixture(scope="module")
def workflow() -> dict[str, Any]:
    return yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def triggers(workflow: dict[str, Any]) -> dict[str, Any]:
    # YAML 1.1 parses the bare key ``on`` as boolean True.
    return workflow.get("on") or workflow[True]


@pytest.fixture(scope="module")
def retrigger_job(workflow: dict[str, Any]) -> dict[str, Any]:
    jobs = workflow["jobs"]
    assert "evidence-retrigger" in jobs, (
        "workflow must define the B1 evidence-retrigger job "
        "(docs/specs/QUORUM_EVIDENCE_RETRIGGER.md)"
    )
    return jobs["evidence-retrigger"]


@pytest.fixture(scope="module")
def enforcing_job(workflow: dict[str, Any]) -> dict[str, Any]:
    return workflow["jobs"]["merge-quorum"]


def _run_blocks(job: dict[str, Any]) -> str:
    return "\n".join(str(step.get("run", "")) for step in job.get("steps", []))


class TestIssueCommentTrigger:
    def test_issue_comment_created_is_a_trigger(self, triggers: dict[str, Any]) -> None:
        """Evidence comments must be able to re-trigger the workflow."""
        assert "issue_comment" in triggers
        assert triggers["issue_comment"]["types"] == ["created"]

    def test_existing_pull_request_trigger_is_preserved(self, triggers: dict[str, Any]) -> None:
        assert triggers["pull_request"]["types"] == [
            "opened",
            "synchronize",
            "reopened",
            "ready_for_review",
        ]


class TestDualEventPrResolution:
    def test_workflow_concurrency_group_resolves_issue_number(
        self, workflow: dict[str, Any]
    ) -> None:
        """Comment events must serialize per-PR, not in one global group."""
        group = workflow["concurrency"]["group"]
        assert "github.event.pull_request.number" in group
        assert "github.event.issue.number" in group

    def test_retrigger_job_resolves_pr_from_issue_number(
        self, retrigger_job: dict[str, Any]
    ) -> None:
        steps_env = "\n".join(str(step.get("env", "")) for step in retrigger_job.get("steps", []))
        assert "github.event.issue.number" in steps_env


class TestRetriggerGuards:
    def test_job_is_gated_to_pr_comments_only(self, retrigger_job: dict[str, Any]) -> None:
        """Declarative guard: issue.pull_request non-null, comment events only."""
        condition = str(retrigger_job.get("if", ""))
        assert "github.event_name == 'issue_comment'" in condition
        assert "github.event.issue.pull_request != null" in condition

    def test_github_actions_bot_comments_are_skipped(self, retrigger_job: dict[str, Any]) -> None:
        """Parser-excluded authors must not burn runs or create bot loops."""
        condition = str(retrigger_job.get("if", ""))
        assert "github.event.comment.user.login != 'github-actions[bot]'" in condition

    def test_guard_matches_known_reviewer_family_headings(
        self, retrigger_job: dict[str, Any]
    ) -> None:
        """The in-step guard mirrors the quorum parsers' family markers."""
        script = _run_blocks(retrigger_job)
        for family in ("claude", "grok", "gemini", "mistral", "openai", "codex", "factory"):
            assert family in script, f"guard regex must include reviewer family {family!r}"

    def test_guard_requires_open_non_draft_pr_and_stale_head_bound_run(
        self, retrigger_job: dict[str, Any]
    ) -> None:
        script = _run_blocks(retrigger_job)
        # PR open + non-draft.
        assert ".draft" in script
        assert ".state" in script
        # Re-run only the latest COMPLETED non-success run for the CURRENT head.
        assert "head_sha" in script
        assert "completed" in script
        assert "gh run rerun" in script

    def test_comment_body_enters_only_via_env(self, retrigger_job: dict[str, Any]) -> None:
        """Injection pin: comment markdown never interpolates into run: text."""
        script = _run_blocks(retrigger_job)
        assert "github.event.comment.body" not in script, (
            "comment body must reach the shell via env:, never inline ${{ }}"
        )
        steps_env: dict[str, Any] = {}
        for step in retrigger_job.get("steps", []):
            steps_env.update(step.get("env", {}) or {})
        assert any("github.event.comment.body" in str(value) for value in steps_env.values()), (
            "comment body must be provided to the guard step through env:"
        )


class TestDebounceConcurrency:
    def test_retrigger_concurrency_is_per_pr_and_cancels_in_progress(
        self, retrigger_job: dict[str, Any]
    ) -> None:
        concurrency = retrigger_job.get("concurrency") or {}
        assert "github.event.issue.number" in str(concurrency.get("group", ""))
        assert concurrency.get("cancel-in-progress") is True

    def test_enforcing_workflow_group_still_never_cancels_in_progress(
        self, workflow: dict[str, Any]
    ) -> None:
        """The anti-doom-loop invariant on the REQUIRED check is preserved."""
        assert workflow["concurrency"]["cancel-in-progress"] is False


class TestEnforcingJobUnchanged:
    def test_enforcing_job_excluded_from_issue_comment_events(
        self, enforcing_job: dict[str, Any]
    ) -> None:
        """A comment event must never produce a default-branch-bound evaluation."""
        assert "github.event_name != 'issue_comment'" in str(enforcing_job.get("if", ""))

    def test_workflow_level_permissions_remain_read_only(self, workflow: dict[str, Any]) -> None:
        assert workflow["permissions"] == {
            "contents": "read",
            "pull-requests": "read",
            "statuses": "read",
        }

    def test_enforcing_job_gains_no_write_permission(self, enforcing_job: dict[str, Any]) -> None:
        permissions = enforcing_job.get("permissions") or {}
        assert "write" not in str(permissions.values())

    def test_retrigger_write_surface_is_exactly_actions(
        self, retrigger_job: dict[str, Any]
    ) -> None:
        """actions:write (re-run our own evaluation) is the ONLY write scope."""
        permissions = retrigger_job.get("permissions") or {}
        writes = sorted(scope for scope, level in permissions.items() if level == "write")
        assert writes == ["actions"]
        assert permissions.get("contents") is None
        assert permissions.get("statuses") is None
