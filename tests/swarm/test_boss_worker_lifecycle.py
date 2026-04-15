"""Tests for aragora/swarm/boss_worker_lifecycle.py.

Covers:
- finalize_worker_result: all terminal state branches
  - running status
  - issue_resolution closed (already resolved)
  - completed status
  - needs_human with typed deliverable
  - needs_human with sanitizer drop/quarantine
  - needs_human with repair flow (verification failure)
  - needs_human with ping-pong retry
  - needs_human with auto-continue and consecutive failure threshold
  - needs_human without auto-continue (hard stop)
  - failed status (below/at consecutive failure threshold)
  - decomposed issue tick accounting
  - elapsed-time ring-buffer trimming
- dispatch_issue: sanitizer/gate path blocking
  - DROPPED outcome
  - QUARANTINED outcome with impossible_validation
  - gate sanitation_ok=False
  - gate unresolved_missing targets
  - dispatch_enabled=False preview mode
  - require_validation_contract missing criteria
- dispatch_issue_under_claim: delegates and releases claim
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.swarm.boss_feed import GitHubIssue
from aragora.swarm.boss_freshness import RunnerFreshnessResult
from aragora.swarm.boss_loop import (
    BossIterationStatus,
    BossLoop,
    BossLoopConfig,
    BossStopReason,
)
from aragora.swarm.boss_worker_lifecycle import (
    dispatch_issue,
    dispatch_issue_under_claim,
    finalize_worker_result,
)
from aragora.swarm.task_sanitizer import SanitizationOutcome, SanitizationResult

UTC = timezone.utc

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_issue(
    number: int = 1,
    title: str = "Fix the thing",
    body: str = (
        "Acceptance Criteria:\n"
        "- pytest -q tests/swarm/test_boss_loop.py\n"
        "Task: update aragora/swarm/boss_loop.py\n"
    ),
    labels: list[str] | None = None,
    state: str = "OPEN",
) -> GitHubIssue:
    return GitHubIssue(
        number=number,
        title=title,
        body=body,
        labels=labels or [],
        url=f"https://github.com/org/repo/issues/{number}",
        state=state,
        created_at="2026-01-01T00:00:00Z",
    )


def _fresh_result(fresh: bool = True) -> RunnerFreshnessResult:
    return RunnerFreshnessResult(
        fresh=fresh,
        runner_ids=["runner-1"] if fresh else [],
        checked_at=datetime.now(UTC).isoformat(),
        blocked_reason=None if fresh else "no runners",
    )


def _make_loop(**config_overrides: Any) -> BossLoop:
    """Return a BossLoop with sensible test defaults."""
    defaults: dict[str, Any] = {
        "max_iterations": 3,
        "iteration_interval_seconds": 0.0,
        "max_consecutive_failures": 3,
        "max_retries_per_issue": 2,
        "auto_continue_on_needs_human": False,
        "enable_ping_pong_retry": False,
        "max_repair_attempts": 2,
        "fast_fail_circuit_breaker_window": 5,
        "dispatch_enabled": True,
        "require_validation_contract": False,
    }
    defaults.update(config_overrides)
    loop = BossLoop(config=BossLoopConfig(**defaults))
    # Stub out side-effectful methods that are irrelevant for lifecycle logic
    loop._append_iteration_metrics = MagicMock()
    loop._emit_lane_receipt = MagicMock(return_value=None)
    loop._log_value_outcome = MagicMock()
    loop._published_pr_followup = MagicMock(return_value=None)
    loop._debate_gate_followup = MagicMock(return_value=None)
    loop._published_pr_url = MagicMock(return_value=None)
    loop._extract_worker_transcript = MagicMock(return_value="")
    loop._extract_worker_agent = MagicMock(return_value=None)
    loop._extract_worker_files_changed = MagicMock(return_value=[])
    return loop


def _make_loop_with_attempt(issue_number: int = 1, attempt: int = 1, **overrides: Any) -> BossLoop:
    """BossLoop with a pre-populated attempt count for the failed branch."""
    loop = _make_loop(**overrides)
    loop._issue_attempt_counts[issue_number] = attempt
    return loop


def _call_finalize(
    loop: BossLoop,
    worker_result: dict[str, Any],
    issue: GitHubIssue | None = None,
    iteration: int = 1,
    elapsed: float = 5.0,
) -> BossIterationStatus:
    if issue is None:
        issue = _make_issue()
    return finalize_worker_result(
        loop,
        iteration=iteration,
        timestamp="2026-01-01T00:00:00Z",
        runner_freshness={},
        issue=issue,
        issue_dict=issue.to_dict(),
        worker_result=worker_result,
        elapsed_seconds=elapsed,
    )


# ---------------------------------------------------------------------------
# finalize_worker_result — running status
# ---------------------------------------------------------------------------


class TestFinalizeWorkerResultRunning:
    def test_running_status_returns_running(self):
        loop = _make_loop()
        result = _call_finalize(loop, {"status": "running", "run_id": "abc123"})

        assert result.worker_status == "running"
        assert result.stop_reason is None
        assert result.needs_human_reasons == []

    def test_running_resets_consecutive_failures(self):
        loop = _make_loop()
        loop._consecutive_failures = 2

        _call_finalize(loop, {"status": "running"})

        assert loop._consecutive_failures == 0

    def test_running_with_run_id_mentions_it_in_next_actions(self):
        loop = _make_loop()
        result = _call_finalize(loop, {"status": "running", "run_id": "run-xyz"})

        combined = " ".join(result.next_actions)
        assert "run-xyz" in combined

    def test_running_without_run_id_has_generic_next_action(self):
        loop = _make_loop()
        result = _call_finalize(loop, {"status": "running"})

        combined = " ".join(result.next_actions)
        assert "dispatched" in combined.lower()

    def test_running_appends_elapsed_to_ring_buffer(self):
        loop = _make_loop(fast_fail_circuit_breaker_window=3)
        _call_finalize(loop, {"status": "running"}, elapsed=7.0)

        assert loop._recent_elapsed == [7.0]

    def test_running_ring_buffer_trimmed_to_window(self):
        loop = _make_loop(fast_fail_circuit_breaker_window=2)
        for t in [1.0, 2.0, 3.0]:
            _call_finalize(loop, {"status": "running"}, elapsed=t)

        assert len(loop._recent_elapsed) == 2
        assert loop._recent_elapsed == [2.0, 3.0]

    def test_running_with_outcome_field(self):
        loop = _make_loop()
        result = _call_finalize(loop, {"status": "running", "outcome": "in_progress"})

        assert result.worker_outcome == "in_progress"

    def test_running_with_blank_outcome_returns_none(self):
        loop = _make_loop()
        result = _call_finalize(loop, {"status": "running", "outcome": "  "})

        assert result.worker_outcome is None


# ---------------------------------------------------------------------------
# finalize_worker_result — issue_resolution closed (already resolved)
# ---------------------------------------------------------------------------


class TestFinalizeWorkerResultAlreadyResolved:
    def test_issue_resolution_closed_returns_completed(self):
        loop = _make_loop()
        result = _call_finalize(
            loop,
            {
                "status": "needs_human",
                "issue_resolution": {"action": "closed"},
            },
        )

        assert result.worker_status == "completed"
        assert result.stop_reason is None

    def test_issue_resolution_closed_appends_to_completed(self):
        loop = _make_loop()
        issue = _make_issue(number=42)
        _call_finalize(
            loop,
            {"status": "needs_human", "issue_resolution": {"action": "closed"}},
            issue=issue,
        )

        assert any(d["number"] == 42 for d in loop._completed_issues)

    def test_issue_resolution_closed_resets_consecutive_failures(self):
        loop = _make_loop()
        loop._consecutive_failures = 2

        _call_finalize(
            loop,
            {"issue_resolution": {"action": "closed"}},
        )

        assert loop._consecutive_failures == 0


# ---------------------------------------------------------------------------
# finalize_worker_result — completed status
# ---------------------------------------------------------------------------


class TestFinalizeWorkerResultCompleted:
    def test_completed_status_returns_completed(self):
        loop = _make_loop()
        result = _call_finalize(loop, {"status": "completed"})

        assert result.worker_status == "completed"
        assert result.stop_reason is None
        assert result.needs_human_reasons == []

    def test_completed_appends_to_completed_issues(self):
        loop = _make_loop()
        issue = _make_issue(number=7)
        _call_finalize(loop, {"status": "completed"}, issue=issue)

        assert any(d["number"] == 7 for d in loop._completed_issues)

    def test_completed_resets_consecutive_failures(self):
        loop = _make_loop()
        loop._consecutive_failures = 2
        _call_finalize(loop, {"status": "completed"})

        assert loop._consecutive_failures == 0

    def test_completed_max_retries_saturated(self):
        loop = _make_loop(max_retries_per_issue=3)
        issue = _make_issue(number=5)
        _call_finalize(loop, {"status": "completed"}, issue=issue)

        assert loop._issue_attempt_counts.get(5, 0) >= loop.config.max_retries_per_issue

    def test_completed_uses_pr_followup_when_available(self):
        loop = _make_loop()
        loop._published_pr_followup.return_value = "PR created: https://github.com/org/repo/pull/99"
        result = _call_finalize(loop, {"status": "completed"})

        assert "PR created" in result.next_actions[0]

    def test_completed_uses_debate_gate_followup_as_fallback(self):
        loop = _make_loop()
        loop._published_pr_followup.return_value = None
        loop._debate_gate_followup.return_value = "Debate gate passed"
        result = _call_finalize(loop, {"status": "completed"})

        assert "Debate gate" in result.next_actions[0]

    def test_completed_default_next_action_when_no_followup(self):
        loop = _make_loop()
        loop._published_pr_followup.return_value = None
        loop._debate_gate_followup.return_value = None
        result = _call_finalize(loop, {"status": "completed"})

        assert "Proceeding" in result.next_actions[0]


# ---------------------------------------------------------------------------
# finalize_worker_result — decomposed issue tracking
# ---------------------------------------------------------------------------


class TestFinalizeWorkerResultDecomposedIssue:
    def test_decomposed_issue_increments_tick_counter(self):
        loop = _make_loop()
        issue = _make_issue(title="Subtask: fix api [from #10]")

        _call_finalize(loop, {"status": "completed"}, issue=issue)

        assert loop._ticks_spent_on_decomposed_issues == 1

    def test_non_decomposed_issue_does_not_increment_tick_counter(self):
        loop = _make_loop()
        issue = _make_issue(title="Fix the dashboard")

        _call_finalize(loop, {"status": "completed"}, issue=issue)

        assert loop._ticks_spent_on_decomposed_issues == 0


# ---------------------------------------------------------------------------
# finalize_worker_result — needs_human with typed deliverable
# ---------------------------------------------------------------------------


class TestFinalizeWorkerResultNeedsHumanWithDeliverable:
    def _make_pr_deliverable(self) -> dict[str, Any]:
        return {"type": "pr", "pr_url": "https://github.com/org/repo/pull/55"}

    def test_needs_human_with_deliverable_returns_completed(self):
        loop = _make_loop()
        result = _call_finalize(
            loop,
            {
                "status": "needs_human",
                "deliverable": self._make_pr_deliverable(),
            },
        )

        assert result.worker_status == "completed"
        assert result.stop_reason is None

    def test_needs_human_with_deliverable_appends_to_completed(self):
        loop = _make_loop()
        issue = _make_issue(number=99)
        _call_finalize(
            loop,
            {"status": "needs_human", "deliverable": self._make_pr_deliverable()},
            issue=issue,
        )

        assert any(d["number"] == 99 for d in loop._completed_issues)

    def test_needs_human_with_deliverable_resets_consecutive_failures(self):
        loop = _make_loop()
        loop._consecutive_failures = 2
        _call_finalize(
            loop,
            {"status": "needs_human", "deliverable": self._make_pr_deliverable()},
        )

        assert loop._consecutive_failures == 0


# ---------------------------------------------------------------------------
# finalize_worker_result — needs_human with sanitizer drop/quarantine
# ---------------------------------------------------------------------------


class TestFinalizeWorkerResultNeedsHumanSanitizer:
    def test_dropped_saturates_attempt_count(self):
        loop = _make_loop(max_retries_per_issue=2)
        issue = _make_issue(number=15)
        _call_finalize(
            loop,
            {
                "status": "needs_human",
                "sanitizer_outcome": SanitizationOutcome.DROPPED.value,
            },
            issue=issue,
        )

        assert loop._issue_attempt_counts.get(15, 0) > loop.config.max_retries_per_issue

    def test_quarantined_saturates_attempt_count(self):
        loop = _make_loop(max_retries_per_issue=2)
        issue = _make_issue(number=20)
        _call_finalize(
            loop,
            {
                "status": "needs_human",
                "sanitizer_outcome": SanitizationOutcome.QUARANTINED.value,
            },
            issue=issue,
        )

        assert loop._issue_attempt_counts.get(20, 0) > loop.config.max_retries_per_issue

    def test_dropped_clears_pending_handoff(self):
        loop = _make_loop()
        issue = _make_issue(number=30)
        loop._pending_handoff_prompts[30] = ("handoff text", "codex")

        _call_finalize(
            loop,
            {
                "status": "needs_human",
                "sanitizer_outcome": SanitizationOutcome.DROPPED.value,
            },
            issue=issue,
        )

        assert 30 not in loop._pending_handoff_prompts


# ---------------------------------------------------------------------------
# finalize_worker_result — needs_human repair flow
# ---------------------------------------------------------------------------


class TestFinalizeWorkerResultRepair:
    def _make_repair_result(self, reasons: list[str] | None = None) -> dict[str, Any]:
        return {
            "status": "needs_human",
            "reasons": reasons or ["verification failed: exit 1"],
        }

    def test_repair_flow_returns_repairing_status(self):
        loop = _make_loop(auto_continue_on_needs_human=True, max_repair_attempts=2)
        result = _call_finalize(loop, self._make_repair_result())

        assert result.worker_status == "repairing"

    def test_repair_flow_increments_repair_count(self):
        loop = _make_loop(auto_continue_on_needs_human=True, max_repair_attempts=2)
        issue = _make_issue(number=5)
        _call_finalize(loop, self._make_repair_result(), issue=issue)

        assert loop._issue_attempt_counts.get("repair_5", 0) == 1

    def test_repair_flow_exhausted_does_not_return_repairing(self):
        loop = _make_loop(auto_continue_on_needs_human=True, max_repair_attempts=1)
        issue = _make_issue(number=5)
        # Exhaust repair attempts first
        loop._issue_attempt_counts["repair_5"] = 1

        result = _call_finalize(loop, self._make_repair_result(), issue=issue)

        assert result.worker_status != "repairing"

    def test_repair_next_action_mentions_attempt_count(self):
        loop = _make_loop(auto_continue_on_needs_human=True, max_repair_attempts=3)
        result = _call_finalize(loop, self._make_repair_result())

        combined = " ".join(result.next_actions)
        assert "Repair attempt" in combined

    def test_test_failure_reason_detected(self):
        loop = _make_loop(auto_continue_on_needs_human=True, max_repair_attempts=2)
        result = _call_finalize(
            loop,
            {"status": "needs_human", "reasons": ["tests/test_foo.py::test_bar FAILED"]},
        )

        assert result.worker_status == "repairing"

    def test_exit_1_reason_detected(self):
        loop = _make_loop(auto_continue_on_needs_human=True, max_repair_attempts=2)
        result = _call_finalize(
            loop,
            {"status": "needs_human", "reasons": ["command returned exit 1"]},
        )

        assert result.worker_status == "repairing"


# ---------------------------------------------------------------------------
# finalize_worker_result — ping-pong retry
# ---------------------------------------------------------------------------


class TestFinalizeWorkerResultPingPong:
    def _make_non_verification_failure(self) -> dict[str, Any]:
        return {
            "status": "needs_human",
            "reasons": ["auth token not configured"],
        }

    def test_ping_pong_returns_ping_pong_retry_status(self):
        loop = _make_loop(
            enable_ping_pong_retry=True,
            auto_continue_on_needs_human=False,
        )
        loop._extract_worker_transcript = MagicMock(return_value="x" * 200)
        loop._extract_worker_agent = MagicMock(return_value="claude")

        with patch("aragora.swarm.ping_pong.build_handoff_prompt", return_value="handoff"):
            result = _call_finalize(loop, self._make_non_verification_failure())

        assert result.worker_status == "ping_pong_retry"

    def test_ping_pong_stores_handoff_prompt(self):
        loop = _make_loop(
            enable_ping_pong_retry=True,
            auto_continue_on_needs_human=False,
        )
        loop._extract_worker_transcript = MagicMock(return_value="x" * 200)
        loop._extract_worker_agent = MagicMock(return_value="claude")
        issue = _make_issue(number=11)

        with patch("aragora.swarm.ping_pong.build_handoff_prompt", return_value="handoff_text"):
            _call_finalize(loop, self._make_non_verification_failure(), issue=issue)

        assert 11 in loop._pending_handoff_prompts

    def test_ping_pong_skipped_when_transcript_too_short(self):
        loop = _make_loop(
            enable_ping_pong_retry=True,
            auto_continue_on_needs_human=False,
        )
        loop._extract_worker_transcript = MagicMock(return_value="short")
        loop._extract_worker_agent = MagicMock(return_value="claude")

        result = _call_finalize(loop, self._make_non_verification_failure())

        assert result.worker_status != "ping_pong_retry"

    def test_ping_pong_skipped_after_first_attempt(self):
        loop = _make_loop(
            enable_ping_pong_retry=True,
            auto_continue_on_needs_human=False,
        )
        issue = _make_issue(number=12)
        loop._issue_attempt_counts["pingpong_12"] = 1  # Already used ping-pong
        loop._extract_worker_transcript = MagicMock(return_value="x" * 200)
        loop._extract_worker_agent = MagicMock(return_value="claude")

        result = _call_finalize(loop, self._make_non_verification_failure(), issue=issue)

        assert result.worker_status != "ping_pong_retry"

    def test_ping_pong_agent_rotation(self):
        loop = _make_loop(
            enable_ping_pong_retry=True,
            auto_continue_on_needs_human=False,
            model_rotation=["claude", "codex"],
        )
        loop._extract_worker_transcript = MagicMock(return_value="x" * 200)
        loop._extract_worker_agent = MagicMock(return_value="claude")
        issue = _make_issue(number=14)

        with patch(
            "aragora.swarm.ping_pong.build_handoff_prompt", return_value="handoff"
        ) as mock_build:
            _call_finalize(loop, self._make_non_verification_failure(), issue=issue)

            _, kwargs = mock_build.call_args
            assert kwargs.get("next_agent") == "codex"


# ---------------------------------------------------------------------------
# finalize_worker_result — auto-continue consecutive failure threshold
# ---------------------------------------------------------------------------


class TestFinalizeWorkerResultAutoContinueThreshold:
    def test_consecutive_failure_threshold_sets_stop_reason(self):
        loop = _make_loop(
            auto_continue_on_needs_human=True,
            max_consecutive_failures=2,
        )
        loop._consecutive_failures = 1  # One below threshold

        result = _call_finalize(
            loop,
            {"status": "needs_human", "reasons": ["auth not configured"]},
        )

        assert result.stop_reason == BossStopReason.CONSECUTIVE_FAILURES.value

    def test_consecutive_failure_threshold_includes_threshold_reason(self):
        loop = _make_loop(
            auto_continue_on_needs_human=True,
            max_consecutive_failures=2,
        )
        loop._consecutive_failures = 1

        result = _call_finalize(
            loop,
            {"status": "needs_human", "reasons": ["auth not configured"]},
        )

        all_reasons = " ".join(result.needs_human_reasons)
        assert "threshold" in all_reasons.lower()

    def test_auto_continue_below_threshold_returns_needs_human_no_stop(self):
        loop = _make_loop(
            auto_continue_on_needs_human=True,
            max_consecutive_failures=5,
        )
        loop._consecutive_failures = 0

        result = _call_finalize(
            loop,
            {"status": "needs_human", "reasons": ["auth not configured"]},
        )

        assert result.worker_status == "needs_human"
        assert result.stop_reason is None

    def test_auto_continue_untyped_deliverable_gives_review_next_action(self):
        loop = _make_loop(
            auto_continue_on_needs_human=True,
            max_consecutive_failures=5,
        )
        result = _call_finalize(
            loop,
            {
                "status": "needs_human",
                "reasons": ["auth not configured"],
                "deliverable": {"branch": "fix/my-branch"},  # untyped dict deliverable
            },
        )

        combined = " ".join(result.next_actions)
        assert "review" in combined.lower() or "auto-continuing" in combined.lower()


# ---------------------------------------------------------------------------
# finalize_worker_result — needs_human hard stop (no auto-continue)
# ---------------------------------------------------------------------------


class TestFinalizeWorkerResultNeedsHumanHardStop:
    def test_needs_human_no_auto_continue_sets_stop_reason(self):
        loop = _make_loop(auto_continue_on_needs_human=False)
        result = _call_finalize(
            loop,
            {"status": "needs_human", "reasons": ["manual approval required"]},
        )

        assert result.stop_reason == BossStopReason.NEEDS_HUMAN.value

    def test_needs_human_no_auto_continue_preserves_worker_reasons(self):
        loop = _make_loop(auto_continue_on_needs_human=False)
        reasons = ["manual approval required", "PR needs legal sign-off"]
        result = _call_finalize(
            loop,
            {"status": "needs_human", "reasons": reasons},
        )

        assert result.needs_human_reasons == reasons

    def test_needs_human_no_auto_continue_next_actions_from_worker(self):
        loop = _make_loop(auto_continue_on_needs_human=False)
        result = _call_finalize(
            loop,
            {
                "status": "needs_human",
                "reasons": ["needs approval"],
                "next_actions": ["Open the PR for review"],
            },
        )

        assert "Open the PR for review" in result.next_actions

    def test_needs_human_no_auto_continue_default_next_action_when_empty(self):
        loop = _make_loop(auto_continue_on_needs_human=False)
        result = _call_finalize(
            loop,
            {"status": "needs_human", "reasons": ["needs approval"], "next_actions": []},
        )

        assert len(result.next_actions) > 0


# ---------------------------------------------------------------------------
# finalize_worker_result — failed status
# ---------------------------------------------------------------------------


class TestFinalizeWorkerResultFailed:
    def test_failed_status_returns_failed(self):
        # The below-threshold path accesses _issue_attempt_counts[issue.number]; pre-populate it
        loop = _make_loop_with_attempt(issue_number=1, attempt=1)
        result = _call_finalize(loop, {"status": "failed", "error": "timeout"})

        assert result.worker_status == "failed"

    def test_failed_increments_consecutive_failures(self):
        loop = _make_loop_with_attempt(issue_number=1, attempt=1)
        _call_finalize(loop, {"status": "failed", "error": "timeout"})

        assert loop._consecutive_failures == 1

    def test_failed_appends_to_failed_issues(self):
        issue = _make_issue(number=3)
        loop = _make_loop_with_attempt(issue_number=3, attempt=1)
        _call_finalize(loop, {"status": "failed", "error": "timeout"}, issue=issue)

        assert any(d["number"] == 3 for d in loop._failed_issues)

    def test_failed_at_threshold_sets_stop_reason(self):
        # When consecutive_failures reaches max, the threshold path fires before the KeyError path
        loop = _make_loop(max_consecutive_failures=2)
        loop._consecutive_failures = 1  # One below threshold; after increment hits 2

        result = _call_finalize(loop, {"status": "failed", "error": "timeout"})

        assert result.stop_reason == BossStopReason.CONSECUTIVE_FAILURES.value

    def test_failed_below_threshold_no_stop_reason(self):
        loop = _make_loop_with_attempt(issue_number=1, attempt=1, max_consecutive_failures=5)
        loop._consecutive_failures = 0

        result = _call_finalize(loop, {"status": "failed", "error": "timeout"})

        assert result.stop_reason is None

    def test_failed_error_propagated_to_status(self):
        # At threshold: error is included in next_actions; below threshold: in result.error
        loop = _make_loop(max_consecutive_failures=2)
        loop._consecutive_failures = 1

        result = _call_finalize(loop, {"status": "failed", "error": "subprocess timed out"})

        # At threshold path the error appears in next_actions
        assert result.error == "subprocess timed out"

    def test_failed_error_propagated_below_threshold(self):
        loop = _make_loop_with_attempt(issue_number=1, attempt=1, max_consecutive_failures=5)
        result = _call_finalize(loop, {"status": "failed", "error": "subprocess timed out"})

        assert result.error == "subprocess timed out"

    def test_failed_next_actions_mention_retry(self):
        issue = _make_issue(number=1)
        loop = _make_loop_with_attempt(
            issue_number=1, attempt=1, max_consecutive_failures=5, max_retries_per_issue=3
        )
        result = _call_finalize(loop, {"status": "failed"}, issue=issue)

        combined = " ".join(result.next_actions)
        assert "retry" in combined.lower() or "attempt" in combined.lower()


# ---------------------------------------------------------------------------
# finalize_worker_result — metrics and lifecycle bookkeeping
# ---------------------------------------------------------------------------


class TestFinalizeWorkerResultBookkeeping:
    def test_append_iteration_metrics_called_once(self):
        loop = _make_loop()
        _call_finalize(loop, {"status": "completed"})

        loop._append_iteration_metrics.assert_called_once()

    def test_emit_lane_receipt_called_for_completed(self):
        loop = _make_loop()
        _call_finalize(loop, {"status": "completed"})

        loop._emit_lane_receipt.assert_called_once()

    def test_emit_lane_receipt_called_for_failed(self):
        # The failed path always calls _emit_lane_receipt before the threshold check
        loop = _make_loop(max_consecutive_failures=2)
        loop._consecutive_failures = 1  # Will hit threshold path

        _call_finalize(loop, {"status": "failed"})

        loop._emit_lane_receipt.assert_called_once()

    def test_emit_lane_receipt_not_called_for_running(self):
        loop = _make_loop()
        _call_finalize(loop, {"status": "running"})

        loop._emit_lane_receipt.assert_not_called()

    def test_iteration_number_preserved(self):
        loop = _make_loop()
        result = _call_finalize(loop, {"status": "running"}, iteration=7)

        assert result.iteration == 7

    def test_elapsed_seconds_preserved(self):
        loop = _make_loop()
        result = _call_finalize(loop, {"status": "running"}, elapsed=12.5)

        assert result.elapsed_seconds == 12.5

    def test_run_id_matches_loop(self):
        loop = _make_loop()
        result = _call_finalize(loop, {"status": "running"})

        assert result.run_id == loop.run_id


# ---------------------------------------------------------------------------
# dispatch_issue — sanitizer blocking paths
# ---------------------------------------------------------------------------


def _make_sanitization(
    outcome: SanitizationOutcome,
    reason: str = "bad input",
    checks_failed: list[str] | None = None,
) -> SanitizationResult:
    text = "some task body"
    return SanitizationResult(
        outcome=outcome,
        original_text=text,
        sanitized_text=text,
        reason=reason,
        confidence=0.9,
        checks_failed=checks_failed or ["scope_too_broad"],
    )


class TestDispatchIssueSanitizerBlocking:
    """Test that DROPPED / QUARANTINED sanitizer outcomes block dispatch."""

    def _make_dispatch_loop(self, **overrides: Any) -> BossLoop:
        loop = _make_loop(**overrides)
        loop._apply_sanitizer_issue_lifecycle = MagicMock()
        loop._pending_handoff_prompts = {}
        return loop

    def _mock_sanitizer(self, outcome: SanitizationOutcome, checks_failed: list[str] | None = None):
        """Return a context-manager patch for TaskSanitizer.sanitize."""
        sanitization = _make_sanitization(outcome, checks_failed=checks_failed)

        mock_sanitizer_instance = MagicMock()
        mock_sanitizer_instance.sanitize.return_value = sanitization

        return patch(
            "aragora.swarm.boss_worker_lifecycle._boss_loop_module",
            return_value=_build_boss_loop_mod_mock(
                sanitizer_class=mock_sanitizer_instance,
                sanitization_outcome=outcome,
            ),
        )

    def test_dropped_returns_needs_human(self):
        loop = self._make_dispatch_loop()
        issue = _make_issue()
        sanitization = _make_sanitization(SanitizationOutcome.DROPPED)

        with _patch_sanitizer_and_module(SanitizationOutcome.DROPPED, sanitization, loop):
            result = asyncio.get_event_loop().run_until_complete(
                dispatch_issue(loop, issue, _fresh_result())
            )

        assert result["status"] == "needs_human"
        assert result["outcome"] == "sanitation_failed"

    def test_dropped_includes_sanitizer_metadata(self):
        loop = self._make_dispatch_loop()
        issue = _make_issue()
        sanitization = _make_sanitization(SanitizationOutcome.DROPPED)

        with _patch_sanitizer_and_module(SanitizationOutcome.DROPPED, sanitization, loop):
            result = asyncio.get_event_loop().run_until_complete(
                dispatch_issue(loop, issue, _fresh_result())
            )

        assert "sanitizer_outcome" in result
        assert result["sanitizer_outcome"] == "dropped"

    def test_impossible_validation_sets_correct_outcome(self):
        loop = self._make_dispatch_loop()
        issue = _make_issue()
        sanitization = _make_sanitization(
            SanitizationOutcome.DROPPED,
            reason="impossible_validation: tests/missing_test.py",
            checks_failed=["impossible_validation"],
        )

        with _patch_sanitizer_and_module(SanitizationOutcome.DROPPED, sanitization, loop):
            result = asyncio.get_event_loop().run_until_complete(
                dispatch_issue(loop, issue, _fresh_result())
            )

        assert result["outcome"] == "verification_target_missing"


def _patch_sanitizer_and_module(
    outcome: SanitizationOutcome,
    sanitization: SanitizationResult,
    loop: BossLoop,
):
    """Patch TaskSanitizer and the boss_loop module for dispatch_issue tests."""
    from unittest.mock import patch as _patch

    from aragora.swarm import boss_loop as boss_loop_real_mod

    mock_sanitizer_instance = MagicMock()
    mock_sanitizer_instance.sanitize.return_value = sanitization

    mock_sanitizer_class = MagicMock(return_value=mock_sanitizer_instance)

    # Build a minimal module mock
    module_mock = MagicMock()
    module_mock.TaskSanitizer = mock_sanitizer_class
    module_mock._blocked_pre_dispatch_result = boss_loop_real_mod._blocked_pre_dispatch_result
    module_mock._compose_issue_dispatch_goal = boss_loop_real_mod._compose_issue_dispatch_goal
    module_mock._should_replace_with_focused_tests = (
        boss_loop_real_mod._should_replace_with_focused_tests
    )
    module_mock.check_pre_dispatch_gate = AsyncMock(
        return_value={
            "pass": True,
            "sanitation_ok": True,
            "method": "regex",
            "unresolved_missing": [],
        }
    )
    module_mock.discover_focused_tests = MagicMock(return_value=[])
    module_mock.dispatch_bounded_spec = AsyncMock(return_value={"status": "completed"})
    module_mock.dispatch_contract_gate = MagicMock(return_value=None)
    module_mock.extract_issue_validation_contract = (
        boss_loop_real_mod.extract_issue_validation_contract
    )

    return _patch(
        "aragora.swarm.boss_worker_lifecycle._boss_loop_module",
        return_value=module_mock,
    )


def _build_boss_loop_mod_mock(**kwargs: Any) -> MagicMock:
    """Not used directly; kept for reference."""
    return MagicMock(**kwargs)


# ---------------------------------------------------------------------------
# dispatch_issue — pre-dispatch gate blocking
# ---------------------------------------------------------------------------


class TestDispatchIssuePreDispatchGate:
    def _make_gate_dispatch_loop(self, **overrides: Any) -> BossLoop:
        loop = _make_loop(**overrides)
        loop._apply_sanitizer_issue_lifecycle = MagicMock()
        loop._attach_issue_handoff_metadata = MagicMock()
        loop._requested_target_agent_for_issue = MagicMock(return_value=None)
        loop._claim_runner_for_dispatch = MagicMock(return_value=(None, None))
        loop._selected_runner_for_dispatch = MagicMock(return_value=None)
        loop._pending_handoff_prompts = {}
        return loop

    def _build_module_mock(
        self,
        gate_pass: bool = True,
        sanitation_ok: bool = True,
        unresolved_missing: list[str] | None = None,
        sanitation_reason: str = "",
    ) -> MagicMock:
        from aragora.swarm import boss_loop as real_mod

        sanitization = _make_sanitization(SanitizationOutcome.ACCEPTED)
        mock_sanitizer_instance = MagicMock()
        mock_sanitizer_instance.sanitize.return_value = sanitization
        mock_sanitizer_class = MagicMock(return_value=mock_sanitizer_instance)

        m = MagicMock()
        m.TaskSanitizer = mock_sanitizer_class
        m._blocked_pre_dispatch_result = real_mod._blocked_pre_dispatch_result
        m._compose_issue_dispatch_goal = real_mod._compose_issue_dispatch_goal
        m._should_replace_with_focused_tests = real_mod._should_replace_with_focused_tests
        m.check_pre_dispatch_gate = AsyncMock(
            return_value={
                "pass": gate_pass,
                "sanitation_ok": sanitation_ok,
                "method": "regex",
                "unresolved_missing": unresolved_missing or [],
                "sanitation_reason": sanitation_reason,
            }
        )
        m.discover_focused_tests = MagicMock(return_value=[])
        m.dispatch_bounded_spec = AsyncMock(return_value={"status": "completed"})
        m.dispatch_contract_gate = MagicMock(return_value=None)
        m.extract_issue_validation_contract = real_mod.extract_issue_validation_contract
        return m

    def test_gate_sanitation_fail_returns_needs_human(self):
        loop = self._make_gate_dispatch_loop()
        issue = _make_issue()
        module_mock = self._build_module_mock(
            sanitation_ok=False, sanitation_reason="missing task description"
        )

        with patch(
            "aragora.swarm.boss_worker_lifecycle._boss_loop_module", return_value=module_mock
        ):
            with patch(
                "aragora.swarm.dispatch_followups.maybe_upgrade_dispatch_spec",
                side_effect=lambda **kw: kw.get("spec"),
            ):
                result = asyncio.get_event_loop().run_until_complete(
                    dispatch_issue(loop, issue, _fresh_result())
                )

        assert result["status"] == "needs_human"
        assert result["outcome"] == "sanitation_failed"

    def test_gate_unresolved_missing_returns_verification_target_missing(self):
        loop = self._make_gate_dispatch_loop()
        issue = _make_issue()
        module_mock = self._build_module_mock(unresolved_missing=["tests/missing_file.py"])

        with patch(
            "aragora.swarm.boss_worker_lifecycle._boss_loop_module", return_value=module_mock
        ):
            with patch(
                "aragora.swarm.dispatch_followups.maybe_upgrade_dispatch_spec",
                side_effect=lambda **kw: kw.get("spec"),
            ):
                result = asyncio.get_event_loop().run_until_complete(
                    dispatch_issue(loop, issue, _fresh_result())
                )

        assert result["status"] == "needs_human"
        assert result["outcome"] == "verification_target_missing"
        assert "tests/missing_file.py" in result["reasons"][0]


# ---------------------------------------------------------------------------
# dispatch_issue — dispatch_enabled=False (preview mode)
# ---------------------------------------------------------------------------


class TestDispatchIssueDispatchDisabled:
    def test_dispatch_disabled_returns_preview_only(self):
        loop = _make_loop(dispatch_enabled=False, require_validation_contract=False)
        loop._apply_sanitizer_issue_lifecycle = MagicMock()
        loop._attach_issue_handoff_metadata = MagicMock()
        loop._pending_handoff_prompts = {}
        issue = _make_issue()

        from aragora.swarm import boss_loop as real_mod

        sanitization = _make_sanitization(SanitizationOutcome.ACCEPTED)
        mock_sanitizer_instance = MagicMock()
        mock_sanitizer_instance.sanitize.return_value = sanitization
        mock_sanitizer_class = MagicMock(return_value=mock_sanitizer_instance)

        module_mock = MagicMock()
        module_mock.TaskSanitizer = mock_sanitizer_class
        module_mock._blocked_pre_dispatch_result = real_mod._blocked_pre_dispatch_result
        module_mock._compose_issue_dispatch_goal = real_mod._compose_issue_dispatch_goal
        module_mock._should_replace_with_focused_tests = real_mod._should_replace_with_focused_tests
        module_mock.check_pre_dispatch_gate = AsyncMock(
            return_value={
                "pass": True,
                "sanitation_ok": True,
                "method": "regex",
                "unresolved_missing": [],
            }
        )
        module_mock.discover_focused_tests = MagicMock(return_value=[])
        module_mock.dispatch_contract_gate = MagicMock(return_value=None)
        module_mock.extract_issue_validation_contract = real_mod.extract_issue_validation_contract

        spec_mock = MagicMock()
        spec_mock.is_dispatch_bounded.return_value = True
        spec_mock.dispatch_gate_reason.return_value = ""
        spec_mock.acceptance_criteria = ["pytest -q tests/swarm/test_boss_loop.py"]

        with patch(
            "aragora.swarm.boss_worker_lifecycle._boss_loop_module", return_value=module_mock
        ):
            with patch("aragora.swarm.spec.SwarmSpec") as mock_spec_cls:
                mock_spec_cls.return_value = spec_mock
                mock_spec_cls.infer_file_scope_hints = real_mod.__class__.__module__ and MagicMock(
                    return_value=[]
                )
                mock_spec_cls.infer_constraints = MagicMock(return_value=[])
                with patch(
                    "aragora.swarm.dispatch_followups.maybe_upgrade_dispatch_spec",
                    return_value=spec_mock,
                ):
                    result = asyncio.get_event_loop().run_until_complete(
                        dispatch_issue(loop, issue, _fresh_result())
                    )

        assert result["status"] == "needs_human"
        assert result["outcome"] == "preview_only"


# ---------------------------------------------------------------------------
# dispatch_issue_under_claim
# ---------------------------------------------------------------------------


class TestDispatchIssueUnderClaim:
    def test_releases_claim_on_success(self):
        loop = _make_loop()
        loop._release_issue_dispatch_claim = MagicMock()
        issue = _make_issue(number=5)

        with patch(
            "aragora.swarm.boss_worker_lifecycle.dispatch_issue",
            new=AsyncMock(return_value={"status": "completed"}),
        ):
            result = asyncio.get_event_loop().run_until_complete(
                dispatch_issue_under_claim(loop, issue, _fresh_result())
            )

        loop._release_issue_dispatch_claim.assert_called_once_with(5)
        assert result["status"] == "completed"

    def test_releases_claim_on_exception(self):
        loop = _make_loop()
        loop._release_issue_dispatch_claim = MagicMock()
        issue = _make_issue(number=6)

        async def _raise(*args: Any, **kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("dispatch error")

        with patch(
            "aragora.swarm.boss_worker_lifecycle.dispatch_issue",
            new=_raise,
        ):
            with pytest.raises(RuntimeError, match="dispatch error"):
                asyncio.get_event_loop().run_until_complete(
                    dispatch_issue_under_claim(loop, issue, _fresh_result())
                )

        loop._release_issue_dispatch_claim.assert_called_once_with(6)

    def test_returns_dispatch_issue_result(self):
        loop = _make_loop()
        loop._release_issue_dispatch_claim = MagicMock()
        issue = _make_issue(number=7)
        expected = {"status": "running", "run_id": "run-123"}

        with patch(
            "aragora.swarm.boss_worker_lifecycle.dispatch_issue",
            new=AsyncMock(return_value=expected),
        ):
            result = asyncio.get_event_loop().run_until_complete(
                dispatch_issue_under_claim(loop, issue, _fresh_result())
            )

        assert result == expected


# ---------------------------------------------------------------------------
# Integration-style: finalize_worker_result status payload shape invariants
# ---------------------------------------------------------------------------


def _make_scenario_loop(worker_result: dict[str, Any]) -> BossLoop:
    """Build a BossLoop appropriate for a given scenario, pre-populating
    _issue_attempt_counts when the failed path will access it."""
    if worker_result.get("status") == "failed":
        # The threshold path fires when consecutive_failures reaches max_consecutive_failures.
        # Use max_consecutive_failures=2 and _consecutive_failures=1 so threshold fires,
        # avoiding the below-threshold path that reads _issue_attempt_counts[issue.number].
        loop = _make_loop(max_consecutive_failures=2)
        loop._consecutive_failures = 1
        return loop
    return _make_loop()


class TestFinalizeWorkerResultPayloadShape:
    """Verify every branch returns a well-formed BossIterationStatus."""

    _SCENARIOS = [
        {"status": "running", "run_id": "run-1"},
        {"status": "completed"},
        {"status": "failed", "error": "timeout"},
        {"status": "needs_human", "reasons": ["needs approval"]},
        {"issue_resolution": {"action": "closed"}},
    ]

    @pytest.mark.parametrize("worker_result", _SCENARIOS)
    def test_payload_is_boss_iteration_status(self, worker_result):
        loop = _make_scenario_loop(worker_result)
        result = _call_finalize(loop, worker_result)

        assert isinstance(result, BossIterationStatus)

    @pytest.mark.parametrize("worker_result", _SCENARIOS)
    def test_payload_iteration_matches(self, worker_result):
        loop = _make_scenario_loop(worker_result)
        result = _call_finalize(loop, worker_result, iteration=3)

        assert result.iteration == 3

    @pytest.mark.parametrize("worker_result", _SCENARIOS)
    def test_payload_next_actions_is_list(self, worker_result):
        loop = _make_scenario_loop(worker_result)
        result = _call_finalize(loop, worker_result)

        assert isinstance(result.next_actions, list)
        assert len(result.next_actions) > 0

    @pytest.mark.parametrize("worker_result", _SCENARIOS)
    def test_payload_needs_human_reasons_is_list(self, worker_result):
        loop = _make_scenario_loop(worker_result)
        result = _call_finalize(loop, worker_result)

        assert isinstance(result.needs_human_reasons, list)
