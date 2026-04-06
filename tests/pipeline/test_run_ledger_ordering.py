"""Tests for RunLedger stage event ordering."""

from __future__ import annotations

from pathlib import Path

import pytest

from aragora.pipeline.backbone_contracts import BackboneStage, RunLedger, RunStageEvent
from aragora.pipeline.plan_store import PlanStore


@pytest.fixture
def store(tmp_path: Path) -> PlanStore:
    return PlanStore(db_path=str(tmp_path / "test_plans.db"))


def test_list_run_stage_events_returns_chronological_order(store: PlanStore) -> None:
    run_id = "run-ordered-events"
    store.create_run(RunLedger(run_id=run_id, entrypoint="prompt_engine.run"))

    for event in (
        RunStageEvent(
            stage=BackboneStage.EXECUTION.value,
            status="started",
            event_id="evt-003",
            created_at="2026-01-01T00:00:03Z",
        ),
        RunStageEvent(
            stage=BackboneStage.INTAKE.value,
            status="completed",
            event_id="evt-001",
            created_at="2026-01-01T00:00:01Z",
        ),
        RunStageEvent(
            stage=BackboneStage.PLAN.value,
            status="completed",
            event_id="evt-002",
            created_at="2026-01-01T00:00:02Z",
        ),
    ):
        assert store.append_run_stage_event(run_id, event) is True

    persisted = store.list_run_stage_events(run_id)

    assert [event.stage for event in persisted] == ["intake", "plan", "execution"]
    assert [event.created_at for event in persisted] == [
        "2026-01-01T00:00:01Z",
        "2026-01-01T00:00:02Z",
        "2026-01-01T00:00:03Z",
    ]
