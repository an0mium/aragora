"""Focused coverage for live default_target_agent supervisor behavior."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from aragora.swarm.supervisor import SwarmSupervisor


def test_default_target_agent_overrides_work_orders() -> None:
    supervisor = SwarmSupervisor.__new__(SwarmSupervisor)

    spec = MagicMock()
    spec.refined_goal = "Test goal"
    spec.raw_goal = "Test goal"
    spec.acceptance_criteria = []
    spec.constraints = []
    spec.file_scope_hints = []
    spec.work_orders = []
    spec.to_dict.return_value = {}

    mock_work_order = MagicMock()
    mock_work_order.to_dict.return_value = {
        "work_order_id": "wo-1",
        "target_agent": "codex",
        "title": "test",
    }
    supervisor._build_supervised_work_orders = MagicMock(return_value=[mock_work_order])

    store = MagicMock()
    created_record: dict[str, Any] = {}

    def capture_create(**kwargs: Any) -> dict[str, Any]:
        created_record.update(kwargs)
        created_record["run_id"] = "run-test"
        created_record["created_at"] = "2026-01-01T00:00:00"
        created_record["updated_at"] = "2026-01-01T00:00:00"
        return created_record

    store.create_supervisor_run.side_effect = capture_create
    store.get_supervisor_run.return_value = None
    supervisor.store = store
    supervisor.refresh_run = MagicMock(return_value=MagicMock())

    supervisor.start_run(
        spec=spec,
        default_target_agent="claude",
        refresh_scaling=False,
    )

    call_kwargs = store.create_supervisor_run.call_args
    work_orders = call_kwargs.kwargs.get("work_orders") or call_kwargs[1].get("work_orders", [])
    assert len(work_orders) == 1
    assert work_orders[0]["target_agent"] == "claude"


def test_no_default_target_agent_preserves_original_work_order_target() -> None:
    supervisor = SwarmSupervisor.__new__(SwarmSupervisor)

    spec = MagicMock()
    spec.refined_goal = "Test goal"
    spec.raw_goal = "Test goal"
    spec.to_dict.return_value = {}

    mock_work_order = MagicMock()
    mock_work_order.to_dict.return_value = {
        "work_order_id": "wo-1",
        "target_agent": "codex",
        "title": "test",
    }
    supervisor._build_supervised_work_orders = MagicMock(return_value=[mock_work_order])

    store = MagicMock()

    def capture_create(**kwargs: Any) -> dict[str, Any]:
        record = dict(kwargs)
        record["run_id"] = "run-test"
        record["created_at"] = "2026-01-01T00:00:00"
        record["updated_at"] = "2026-01-01T00:00:00"
        return record

    store.create_supervisor_run.side_effect = capture_create
    supervisor.store = store
    supervisor.refresh_run = MagicMock(return_value=MagicMock())

    supervisor.start_run(
        spec=spec,
        refresh_scaling=False,
    )

    call_kwargs = store.create_supervisor_run.call_args
    work_orders = call_kwargs.kwargs.get("work_orders") or call_kwargs[1].get("work_orders", [])
    assert len(work_orders) == 1
    assert work_orders[0]["target_agent"] == "codex"


def test_start_run_preserves_default_supervisor_agents_without_target_override() -> None:
    supervisor = SwarmSupervisor.__new__(SwarmSupervisor)

    spec = MagicMock()
    spec.refined_goal = "Test goal"
    spec.raw_goal = "Test goal"
    spec.to_dict.return_value = {}

    mock_work_order = MagicMock()
    mock_work_order.to_dict.return_value = {
        "work_order_id": "wo-1",
        "target_agent": "codex",
        "reviewer_agent": "claude",
        "title": "test",
    }
    supervisor._build_supervised_work_orders = MagicMock(return_value=[mock_work_order])

    store = MagicMock()

    def capture_create(**kwargs: Any) -> dict[str, Any]:
        record = dict(kwargs)
        record["run_id"] = "run-test"
        record["created_at"] = "2026-01-01T00:00:00"
        record["updated_at"] = "2026-01-01T00:00:00"
        return record

    store.create_supervisor_run.side_effect = capture_create
    supervisor.store = store
    supervisor.refresh_run = MagicMock(return_value=MagicMock())

    supervisor.start_run(
        spec=spec,
        refresh_scaling=False,
    )

    call_kwargs = store.create_supervisor_run.call_args
    supervisor_agents = call_kwargs.kwargs.get("supervisor_agents") or call_kwargs[1].get(
        "supervisor_agents", {}
    )
    assert supervisor_agents["planner"] == "codex"
    assert supervisor_agents["judge"] == "claude"

    work_orders = call_kwargs.kwargs.get("work_orders") or call_kwargs[1].get("work_orders", [])
    assert len(work_orders) == 1
    assert work_orders[0]["reviewer_agent"] == "claude"
