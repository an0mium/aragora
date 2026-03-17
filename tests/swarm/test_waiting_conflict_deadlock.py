"""Regression tests for waiting_conflict deadlock (issue #883).

Covers the bug where a Boss-loop run with only waiting_conflict-blocked
work orders remained active until manual kill because:
1. _derive_status treated waiting_conflict as active (not needs_human)
2. _should_stop kept the reconciler alive for waiting_conflict

Forensic evidence from the second pure live run against #873:
- run_id: e60a6aaf-f2a
- final statuses: completed(1), waiting_conflict(2), failed(1)
- no forward progress possible, killed after ~28 minutes
"""

from __future__ import annotations

import pytest

from aragora.swarm.reconciler import SwarmReconciler, SwarmReconcilerConfig
from aragora.swarm.supervisor import SupervisorRun, SwarmApprovalPolicy, SwarmSupervisor
from aragora.swarm.spec import SwarmSpec
from unittest.mock import AsyncMock, MagicMock


def _run(status: str, work_order_statuses: list[str], *, run_id: str = "run-e60a") -> SupervisorRun:
    return SupervisorRun(
        run_id=run_id,
        goal="Bump @eslint/eslintrc from 3.2.0 to 3.3.0 in /aragora/live",
        target_branch="main",
        status=status,
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy=SwarmApprovalPolicy(),
        spec=SwarmSpec(raw_goal="bump eslintrc"),
        work_orders=[
            {"work_order_id": f"wo-{idx}", "status": s}
            for idx, s in enumerate(work_order_statuses, start=1)
        ],
    )


class TestDeriveStatusDeadlock:
    """Verify _derive_status escalates dead-end runs to needs_human."""

    def test_waiting_conflict_only_escalates(self) -> None:
        work_orders = [
            {"status": "waiting_conflict"},
            {"status": "waiting_conflict"},
        ]
        assert SwarmSupervisor._derive_status(work_orders) == "needs_human"

    def test_forensic_873_shape_escalates(self) -> None:
        """Exact statuses from the failed run: completed + 2x waiting_conflict + failed."""
        work_orders = [
            {"status": "completed"},
            {"status": "waiting_conflict"},
            {"status": "waiting_conflict"},
            {"status": "failed"},
        ]
        assert SwarmSupervisor._derive_status(work_orders) == "needs_human"

    def test_waiting_resource_only_escalates(self) -> None:
        work_orders = [{"status": "waiting_resource"}]
        assert SwarmSupervisor._derive_status(work_orders) == "needs_human"

    def test_mixed_waiting_and_terminal_escalates(self) -> None:
        work_orders = [
            {"status": "completed"},
            {"status": "merged"},
            {"status": "waiting_conflict"},
        ]
        assert SwarmSupervisor._derive_status(work_orders) == "needs_human"

    def test_waiting_conflict_with_queued_stays_active(self) -> None:
        """A queued work order can still make progress — don't escalate."""
        work_orders = [
            {"status": "waiting_conflict"},
            {"status": "queued"},
        ]
        assert SwarmSupervisor._derive_status(work_orders) == "active"

    def test_waiting_conflict_with_leased_stays_active(self) -> None:
        work_orders = [
            {"status": "waiting_conflict"},
            {"status": "leased"},
        ]
        assert SwarmSupervisor._derive_status(work_orders) == "active"

    def test_waiting_conflict_with_dispatched_stays_active(self) -> None:
        work_orders = [
            {"status": "waiting_conflict"},
            {"status": "dispatched"},
        ]
        assert SwarmSupervisor._derive_status(work_orders) == "active"

    def test_all_terminal_still_completed(self) -> None:
        """Ensure terminal-only runs still resolve to completed (not needs_human)."""
        work_orders = [
            {"status": "completed"},
            {"status": "merged"},
            {"status": "failed"},
        ]
        assert SwarmSupervisor._derive_status(work_orders) == "completed"

    def test_needs_human_still_takes_precedence(self) -> None:
        work_orders = [
            {"status": "needs_human"},
            {"status": "waiting_conflict"},
        ]
        assert SwarmSupervisor._derive_status(work_orders) == "needs_human"

    def test_forensic_873_shape_campaign_outcome_is_stalled(self) -> None:
        work_orders = [
            {"status": "completed"},
            {"status": "waiting_conflict"},
            {"status": "waiting_conflict"},
            {"status": "failed"},
        ]
        outcome, blockers = SwarmSupervisor._campaign_outcome_for_work_orders(work_orders)

        assert outcome == "stalled"
        assert blockers == []

    def test_scope_violation_takes_precedence_over_stalled(self) -> None:
        work_orders = [
            {"status": "waiting_conflict"},
            {"status": "scope_violation"},
        ]
        outcome, blockers = SwarmSupervisor._campaign_outcome_for_work_orders(work_orders)

        assert outcome == "blocked"
        assert blockers == []

    def test_crash_worker_outcome_takes_precedence_over_stalled(self) -> None:
        work_orders = [
            {"status": "waiting_conflict"},
            {"status": "failed", "worker_outcome": "crash"},
        ]
        outcome, blockers = SwarmSupervisor._campaign_outcome_for_work_orders(work_orders)

        assert outcome == "crash"
        assert blockers == []


class TestReconcilerStopForensic873:
    """Verify the reconciler stops for the exact #873 failure shape."""

    def test_reconciler_stops_on_forensic_shape(self) -> None:
        run = _run("needs_human", ["completed", "waiting_conflict", "waiting_conflict", "failed"])
        assert SwarmReconciler._should_stop(run) is True

    @pytest.mark.asyncio
    async def test_watch_run_terminates_on_deadlock(self) -> None:
        """watch_run must terminate (not poll forever) when only
        waiting_conflict work orders remain."""
        # First tick: still active with a leased worker
        active = _run("active", ["leased", "waiting_conflict"])
        # Second tick: leased worker completed, only waiting_conflict left
        deadlocked = _run(
            "needs_human", ["completed", "waiting_conflict", "waiting_conflict", "failed"]
        )

        reconciler = SwarmReconciler(supervisor=MagicMock())
        reconciler.tick_run = AsyncMock(side_effect=[active, deadlocked])

        result = await reconciler.watch_run("run-e60a", interval_seconds=0.01, max_ticks=10)

        assert result.status == "needs_human"
        # Must have stopped after 2 ticks, not hit max_ticks=10
        assert reconciler.tick_run.await_count == 2

    @pytest.mark.asyncio
    async def test_watch_run_continues_with_forward_progress(self) -> None:
        """Ensure watch_run still polls when real forward progress exists."""
        tick1 = _run("active", ["leased", "waiting_conflict"])
        tick2 = _run("active", ["dispatched", "waiting_conflict"])
        tick3 = _run("completed", ["completed", "completed"])

        reconciler = SwarmReconciler(supervisor=MagicMock())
        reconciler.tick_run = AsyncMock(side_effect=[tick1, tick2, tick3])

        result = await reconciler.watch_run("run-e60a", interval_seconds=0.01, max_ticks=10)

        assert result.status == "completed"
        assert reconciler.tick_run.await_count == 3
