"""Tests for periodic swarm reconciliation."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, call

import pytest

from aragora.swarm.reconciler import SwarmReconciler, SwarmReconcilerConfig
from aragora.swarm.spec import SwarmSpec
from aragora.swarm.supervisor import SupervisorRun, SwarmApprovalPolicy


def _run(status: str, work_order_statuses: list[str], *, run_id: str = "run-123") -> SupervisorRun:
    return SupervisorRun(
        run_id=run_id,
        goal="goal",
        target_branch="main",
        status=status,
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy=SwarmApprovalPolicy(),
        spec=SwarmSpec(raw_goal="goal", refined_goal="goal"),
        work_orders=[
            {"work_order_id": f"wo-{idx}", "status": item_status}
            for idx, item_status in enumerate(work_order_statuses, start=1)
        ],
    )


@pytest.mark.asyncio
async def test_tick_run_dispatches_collects_and_syncs_queue() -> None:
    supervisor = MagicMock()
    supervisor.refresh_run.side_effect = [_run("active", ["leased"])]
    supervisor.dispatch_workers = AsyncMock(return_value=[])
    supervisor.collect_finished_results = AsyncMock(return_value=[])
    supervisor.store.sync_pending_work_queue = AsyncMock(return_value={"created": 0})
    supervisor.store.get_supervisor_run.return_value = _run("active", ["dispatched"]).to_dict()

    reconciler = SwarmReconciler(supervisor=supervisor)
    result = await reconciler.tick_run("run-123")

    assert result.run_id == "run-123"
    supervisor.refresh_run.assert_called_once_with("run-123")
    supervisor.dispatch_workers.assert_awaited_once_with("run-123")
    supervisor.collect_finished_results.assert_awaited_once_with("run-123")
    supervisor.store.sync_pending_work_queue.assert_awaited_once()


@pytest.mark.asyncio
async def test_tick_run_redispatches_after_finished_workers_free_capacity() -> None:
    supervisor = MagicMock()
    supervisor.refresh_run.side_effect = [
        _run("active", ["leased"]),
        _run("active", ["completed", "leased"]),
    ]
    supervisor.dispatch_workers = AsyncMock(side_effect=[[], []])
    supervisor.collect_finished_results = AsyncMock(return_value=[MagicMock(work_order_id="wo-1")])
    supervisor.store.sync_pending_work_queue = AsyncMock(return_value={"created": 1})
    supervisor.store.get_supervisor_run.return_value = _run(
        "active", ["completed", "dispatched"]
    ).to_dict()

    reconciler = SwarmReconciler(supervisor=supervisor)
    result = await reconciler.tick_run("run-123")

    assert result.run_id == "run-123"
    assert supervisor.refresh_run.call_args_list == [call("run-123"), call("run-123")]
    assert supervisor.dispatch_workers.await_args_list == [call("run-123"), call("run-123")]
    supervisor.collect_finished_results.assert_awaited_once_with("run-123")


@pytest.mark.asyncio
async def test_tick_run_can_skip_pending_queue_sync() -> None:
    supervisor = MagicMock()
    supervisor.refresh_run.side_effect = [_run("active", ["leased"])]
    supervisor.dispatch_workers = AsyncMock(return_value=[])
    supervisor.collect_finished_results = AsyncMock(return_value=[])
    supervisor.store.sync_pending_work_queue = AsyncMock(return_value={"created": 0})
    supervisor.store.get_supervisor_run.return_value = _run("active", ["dispatched"]).to_dict()

    reconciler = SwarmReconciler(
        supervisor=supervisor,
        config=SwarmReconcilerConfig(sync_pending_queue=False),
    )
    await reconciler.tick_run("run-123")

    supervisor.store.sync_pending_work_queue.assert_not_awaited()


@pytest.mark.asyncio
async def test_tick_open_runs_only_advances_named_runs() -> None:
    supervisor = MagicMock()
    supervisor.status_summary.return_value = {
        "runs": [
            {"run_id": "run-123"},
            {"run_id": ""},
            {"status": "active"},
            "invalid",
            {"run_id": "run-456"},
        ]
    }
    reconciler = SwarmReconciler(supervisor=supervisor)
    reconciler.tick_run = AsyncMock(
        side_effect=[
            _run("active", ["dispatched"], run_id="run-123"),
            _run("needs_human", ["blocked"], run_id="run-456"),
        ]
    )

    runs = await reconciler.tick_open_runs(limit=7)

    assert [run.run_id for run in runs] == ["run-123", "run-456"]
    supervisor.status_summary.assert_called_once_with(limit=7, refresh_scaling=False)
    assert reconciler.tick_run.await_args_list == [call("run-123"), call("run-456")]


@pytest.mark.asyncio
async def test_watch_run_stops_when_completed() -> None:
    active = _run("active", ["dispatched"])
    completed = _run("completed", ["merged"])

    reconciler = SwarmReconciler(supervisor=MagicMock())
    reconciler.tick_run = AsyncMock(side_effect=[active, completed])

    result = await reconciler.watch_run("run-123", interval_seconds=0.01, max_ticks=3)

    assert result.status == "completed"
    assert reconciler.tick_run.await_count == 2


@pytest.mark.asyncio
async def test_watch_run_uses_boundary_tick_before_hitting_max_ticks() -> None:
    active = _run("active", ["dispatched"])
    completed = _run("completed", ["merged"])

    reconciler = SwarmReconciler(supervisor=MagicMock())
    reconciler.tick_run = AsyncMock(side_effect=[active, completed])

    result = await reconciler.watch_run("run-123", interval_seconds=0.01, max_ticks=1)

    assert result.status == "completed"
    assert reconciler.tick_run.await_count == 2


def test_should_not_stop_for_waiting_resource_with_forward_progress() -> None:
    """waiting_resource alone is a dead-end, but combined with a leased
    work order the run still has forward progress."""
    run = _run("active", ["waiting_resource", "leased"])

    assert SwarmReconciler._should_stop(run) is False


def test_should_stop_when_only_terminal_work_orders_remain() -> None:
    run = _run("active", ["completed", "merged", "blocked"])

    assert SwarmReconciler._should_stop(run) is True


def test_should_stop_when_only_waiting_conflict_remains() -> None:
    """Regression for #883: waiting_conflict with no forward-progress path
    must cause the reconciler to stop, not poll indefinitely."""
    run = _run("needs_human", ["completed", "waiting_conflict", "waiting_conflict", "failed"])

    assert SwarmReconciler._should_stop(run) is True


def test_should_stop_when_only_waiting_resource_remains() -> None:
    """waiting_resource alone (no leased/dispatched/queued) is a dead-end."""
    run = _run("needs_human", ["waiting_resource"])

    assert SwarmReconciler._should_stop(run) is True


def test_should_not_stop_when_queued_work_remains() -> None:
    """queued work orders represent forward progress even if some are
    waiting_conflict."""
    run = _run("active", ["waiting_conflict", "queued"])

    assert SwarmReconciler._should_stop(run) is False


def test_should_not_stop_when_dispatched_work_remains() -> None:
    run = _run("active", ["waiting_conflict", "dispatched"])

    assert SwarmReconciler._should_stop(run) is False


@pytest.mark.asyncio
async def test_watch_run_does_not_force_collect_on_max_ticks_by_default() -> None:
    """Default watch_run behavior should stop waiting without killing workers."""
    dispatched = _run("active", ["dispatched", "dispatched"])

    supervisor = MagicMock()
    supervisor._kill_worker = AsyncMock()
    supervisor.collect_finished_results = AsyncMock(return_value=[])

    reconciler = SwarmReconciler(supervisor=supervisor)
    reconciler.tick_run = AsyncMock(return_value=dispatched)

    result = await reconciler.watch_run("run-123", interval_seconds=0.01, max_ticks=2)

    assert result.status == "active"
    supervisor._kill_worker.assert_not_awaited()
    supervisor.collect_finished_results.assert_not_awaited()


@pytest.mark.asyncio
async def test_watch_run_force_collects_on_max_ticks_exhaustion_when_enabled() -> None:
    """Bounded runs can opt in to a final salvage collect on max_ticks."""
    dispatched = _run("active", ["dispatched", "dispatched"])
    final = _run("needs_human", ["needs_human", "needs_human"])

    supervisor = MagicMock()
    supervisor.store.get_supervisor_run.side_effect = [dispatched.to_dict(), final.to_dict()]
    supervisor._kill_worker = AsyncMock()
    supervisor.collect_finished_results = AsyncMock(return_value=[])
    supervisor.refresh_run.return_value = final

    reconciler = SwarmReconciler(supervisor=supervisor)
    reconciler.tick_run = AsyncMock(return_value=dispatched)

    result = await reconciler.watch_run(
        "run-123",
        interval_seconds=0.01,
        max_ticks=2,
        force_collect_on_max_ticks=True,
    )

    assert result.status == "needs_human"
    assert supervisor._kill_worker.await_count == 2
    supervisor.collect_finished_results.assert_awaited_once_with("run-123")


@pytest.mark.asyncio
async def test_watch_run_skips_force_collect_when_naturally_stopped_even_if_enabled() -> None:
    """Natural stop should win even when bounded force-collect is enabled."""
    completed = _run("completed", ["merged"])

    supervisor = MagicMock()
    reconciler = SwarmReconciler(supervisor=supervisor)
    reconciler.tick_run = AsyncMock(return_value=completed)

    result = await reconciler.watch_run(
        "run-123",
        interval_seconds=0.01,
        max_ticks=10,
        force_collect_on_max_ticks=True,
    )

    assert result.status == "completed"
    supervisor._kill_worker.assert_not_called()


@pytest.mark.asyncio
async def test_force_collect_dispatched_kills_and_collects() -> None:
    """_force_collect_dispatched should kill all dispatched workers and
    call collect_finished_results for salvage."""
    run_dict = _run("active", ["dispatched", "completed", "dispatched"]).to_dict()

    supervisor = MagicMock()
    supervisor.store.get_supervisor_run.return_value = run_dict
    supervisor._kill_worker = AsyncMock()
    supervisor.collect_finished_results = AsyncMock(return_value=[])
    supervisor.refresh_run.return_value = _run("needs_human", ["needs_human"])

    # After force collect, return updated state
    final_dict = _run("needs_human", ["needs_human", "completed", "needs_human"]).to_dict()
    supervisor.store.get_supervisor_run.side_effect = [run_dict, final_dict]

    reconciler = SwarmReconciler(supervisor=supervisor)
    result = await reconciler._force_collect_dispatched("run-123")

    # Should kill only the 2 dispatched work orders
    assert supervisor._kill_worker.await_count == 2
    supervisor.collect_finished_results.assert_awaited_once_with("run-123")
    assert result.status == "needs_human"


@pytest.mark.asyncio
async def test_force_collect_dispatched_noop_when_none_dispatched() -> None:
    """_force_collect_dispatched should return early if no work orders are dispatched."""
    run_dict = _run("active", ["completed", "merged"]).to_dict()

    supervisor = MagicMock()
    supervisor.store.get_supervisor_run.return_value = run_dict
    supervisor._kill_worker = AsyncMock()
    supervisor.collect_finished_results = AsyncMock(return_value=[])

    reconciler = SwarmReconciler(supervisor=supervisor)
    result = await reconciler._force_collect_dispatched("run-123")

    supervisor._kill_worker.assert_not_awaited()
    supervisor.collect_finished_results.assert_not_awaited()
