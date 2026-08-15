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
  ``cancel-in-progress: false`` invariant is preserved;
* BOTH retrigger surfaces (the in-file ``evidence-retrigger`` job and
  the standalone ``aragora-merge-quorum-retrigger.yml`` helper) select
  their rerun target deterministically: the newest completed non-draft
  head-bound evaluation ordered by ``(run_started_at, run_id,
  run_attempt)``, with concurrent comment bursts collapsing into a
  single rerun (``TestDeterministicNonDraftSelection``).

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
STANDALONE_WORKFLOW_PATH = (
    Path(__file__).resolve().parents[2]
    / ".github"
    / "workflows"
    / "aragora-merge-quorum-retrigger.yml"
)


@pytest.fixture(scope="module")
def workflow() -> dict[str, Any]:
    return yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def standalone_workflow() -> dict[str, Any]:
    return yaml.safe_load(STANDALONE_WORKFLOW_PATH.read_text(encoding="utf-8"))


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


class TestDeterministicNonDraftSelection:
    """Both retrigger surfaces select their rerun target deterministically.

    Regression target for the 2026-08-14 draft-success resurfacing incident
    on PR #9754: two pull_request evaluations existed at head ``ee63516c``
    (draft-era run 31772664823, ready-state run 31772790229), and a
    two-comment evidence burst rerain BOTH — the standalone helper picked
    the older draft-era run because the newest one was already re-running,
    and a rerun re-executes the run's ORIGINAL frozen event payload, so the
    draft short-circuit replayed a stale SUCCESS over the truthful
    ready-state ``human_risk_settlement_required`` result.

    The contract pinned here, for the in-file ``evidence-retrigger`` job AND
    the standalone ``aragora-merge-quorum-retrigger.yml`` helper alike:

    * enumerate ALL head-bound ``pull_request`` evaluations (full
      pagination reconciled against ``total_count``), never a single
      API-ordering-dependent first item;
    * exclude draft-frozen evaluations (runs created before the PR's newest
      ``ready_for_review`` transition);
    * order candidates by ``(run_started_at, run_id, run_attempt)`` and
      consider ONLY the newest survivor — an in-flight or already-green
      newest evaluation means no-op, never a fallback to an older run;
    * deduplicate concurrent comment-triggered retriggers into one rerun
      via a fresh pre-rerun status read plus rerun-rejection tolerance.
    """

    @pytest.fixture(scope="class")
    def retrigger_scripts(
        self, retrigger_job: dict[str, Any], standalone_workflow: dict[str, Any]
    ) -> dict[str, str]:
        return {
            "evidence-retrigger": _run_blocks(retrigger_job),
            "standalone": _run_blocks(standalone_workflow["jobs"]["retrigger"]),
        }

    def test_selection_enumerates_all_head_bound_runs(
        self, retrigger_scripts: dict[str, str]
    ) -> None:
        """Full pagination + total_count reconciliation, per surface."""
        for surface, script in retrigger_scripts.items():
            assert "--paginate" in script, f"{surface}: must enumerate ALL head-bound runs"
            assert "total_count" in script, (
                f"{surface}: enumeration must be reconciled against total_count"
            )

    def test_selection_orders_by_run_started_at_run_id_run_attempt(
        self, retrigger_scripts: dict[str, str]
    ) -> None:
        """The pinned deterministic ordering key, identical on both surfaces."""
        for surface, script in retrigger_scripts.items():
            assert "sort_by(.run_started_at, .id, .run_attempt)" in script, (
                f"{surface}: selection must order by (run_started_at, run_id, run_attempt)"
            )

    def test_selection_excludes_draft_frozen_evaluations(
        self, retrigger_scripts: dict[str, str]
    ) -> None:
        """Runs created before the newest ready_for_review transition are
        frozen draft payloads and must never be rerun targets."""
        for surface, script in retrigger_scripts.items():
            assert "ready_for_review" in script, (
                f"{surface}: selection must partition out draft-created evaluations"
            )
            assert "created_at" in script

    def test_nondeterministic_selections_are_gone(self, retrigger_scripts: dict[str, str]) -> None:
        """The two incident-era selections must not reappear."""
        for surface, script in retrigger_scripts.items():
            assert ".workflow_runs[0]" not in script, (
                f"{surface}: per_page=1 first-item selection is API-ordering-dependent"
            )
            assert "sort_by(.createdAt)" not in script, (
                f"{surface}: createdAt-only ordering ignores reruns and attempt order"
            )
            assert '.status=="completed")' not in script, (
                f"{surface}: filtering to completed runs BEFORE picking the newest "
                "falls back to older (draft-era) evaluations while the newest is "
                "re-running — the PR #9754 resurfacing mechanism"
            )

    def test_only_the_newest_evaluation_may_be_rerun(
        self, retrigger_scripts: dict[str, str]
    ) -> None:
        """Completed/success checks happen AFTER selection: an in-flight or
        green newest evaluation no-ops instead of falling back."""
        for surface, script in retrigger_scripts.items():
            assert script.count("gh run rerun") == 1, (
                f"{surface}: exactly one rerun path, for the selected newest run only"
            )
            assert '"$run_status" != "completed"' in script, (
                f"{surface}: in-flight newest evaluation must no-op, not fall back"
            )
            assert '"$run_conclusion" == "success"' in script, (
                f"{surface}: a green newest evaluation needs no recount"
            )

    def test_concurrent_retriggers_collapse_into_one_rerun(
        self, retrigger_scripts: dict[str, str]
    ) -> None:
        """All surfaces compute the same target; a fresh pre-rerun status
        read plus rerun-rejection tolerance turns burst losers into no-ops."""
        for surface, script in retrigger_scripts.items():
            assert "actions/runs/${run_id}" in script, (
                f"{surface}: must re-read the selected run's status just before rerun"
            )
            assert "concurrent retrigger won" in script
            assert "::warning::rerun request" in script, (
                f"{surface}: a rejected rerun (already re-running) must not be red noise"
            )

    def test_standalone_guards_open_non_draft_pr(self, standalone_workflow: dict[str, Any]) -> None:
        """The helper must observe the same gate-deferral rule as the job."""
        script = _run_blocks(standalone_workflow["jobs"]["retrigger"])
        assert ".draft" in script
        assert ".state" in script

    def test_issue_events_read_permission_present(
        self, retrigger_job: dict[str, Any], standalone_workflow: dict[str, Any]
    ) -> None:
        """The ready_for_review partition reads issue events on both surfaces."""
        job_permissions = retrigger_job.get("permissions") or {}
        assert job_permissions.get("issues") == "read"
        workflow_permissions = standalone_workflow.get("permissions") or {}
        assert workflow_permissions.get("issues") == "read"
        writes = sorted(scope for scope, level in workflow_permissions.items() if level == "write")
        assert writes == ["actions"], "helper workflow write surface stays exactly actions"
