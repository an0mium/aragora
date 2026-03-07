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


def test_should_not_stop_for_waiting_resource() -> None:
    run = _run("active", ["waiting_resource"])

    assert SwarmReconciler._should_stop(run) is False


def test_should_stop_when_only_terminal_work_orders_remain() -> None:
    run = _run("active", ["completed", "merged", "blocked"])

    assert SwarmReconciler._should_stop(run) is True
