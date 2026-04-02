"""Tests for canonical DecisionPlan queue execution wiring."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from aragora.gauntlet.receipt_store import reset_receipt_store
from aragora.pipeline.backbone_contracts import BackboneStage, RunLedger
from aragora.pipeline.canonical_execution import (
    build_decision_plan_from_orchestration,
    queue_plan_execution,
)
from aragora.pipeline.execution_mode import ExecutionMode
from aragora.pipeline.plan_store import PlanStore


@pytest.fixture
def store(tmp_path: Path) -> PlanStore:
    return PlanStore(db_path=str(tmp_path / "canonical_execution.db"))


@pytest.fixture(autouse=True)
def _patch_runtime(monkeypatch: pytest.MonkeyPatch, store: PlanStore) -> None:
    monkeypatch.setattr("aragora.pipeline.plan_store.get_plan_store", lambda: store)
    monkeypatch.setattr("aragora.pipeline.executor.store_plan", lambda plan: None)
    reset_receipt_store()
    yield
    reset_receipt_store()


def _build_plan(*, source_surface: str = "canvas_pipeline") -> tuple[object, list[object]]:
    return build_decision_plan_from_orchestration(
        subject_id="pipe-123",
        subject_label="Pipeline pipe-123",
        nodes=[
            {
                "id": "task-1",
                "label": "Build cache",
                "data": {
                    "orch_type": "agent_task",
                    "files": ["src/cache.py"],
                },
            },
            {
                "id": "task-2",
                "label": "Write tests",
                "data": {
                    "orch_type": "verification",
                    "files": ["tests/test_cache.py"],
                },
            },
        ],
        edges=[{"source": "task-1", "target": "task-2"}],
        source_surface=source_surface,
        metadata={"pipeline_id": "pipe-123"},
        execution_mode="workflow",
    )


class TestQueuePlanExecution:
    def test_queue_plan_execution_creates_backbone_run(
        self,
        store: PlanStore,
    ) -> None:
        plan, _tasks = _build_plan()

        launch = queue_plan_execution(
            plan,
            auth_context=SimpleNamespace(user_id="user-1"),
            execution_mode="workflow",
        )

        stored_plan = store.get(plan.id)
        run = store.get_run(launch["run_id"])
        record = store.get_execution_record(launch["execution_id"])

        assert stored_plan is not None
        assert stored_plan.metadata["backbone_run_id"] == launch["run_id"]
        assert stored_plan.metadata["backbone_entrypoint"] == "canonical_execution.canvas_pipeline"
        assert run is not None
        assert run.plan_id == plan.id
        assert run.debate_id == plan.debate_id
        assert run.status == "execution_queued"
        assert run.execution_id == launch["execution_id"]
        assert run.receipt_envelope is not None
        assert run.metadata["source_surface"] == "canvas_pipeline"
        assert run.metadata["pipeline_id"] == "pipe-123"
        assert run.metadata["scheduled_by"] == "user-1"
        assert run.metadata["safety_mode"] == ExecutionMode.INTERACTIVE.value
        assert run.goal_refs[0]["id"] == "task-1"
        assert run.goal_refs[1]["dependencies"] == ["task-1"]
        assert any(
            event.stage == BackboneStage.INTAKE.value and event.status == "completed"
            for event in run.stage_events
        )
        assert any(
            event.stage == BackboneStage.GOALS.value and event.status == "completed"
            for event in run.stage_events
        )
        assert any(
            event.stage == BackboneStage.EXECUTION.value and event.status == "queued"
            for event in run.stage_events
        )
        assert record is not None
        assert record["metadata"]["backbone_run_id"] == launch["run_id"]
        assert record["metadata"]["safety_mode"] == ExecutionMode.INTERACTIVE.value

    def test_queue_plan_execution_reuses_existing_backbone_run(
        self,
        store: PlanStore,
    ) -> None:
        plan, _tasks = _build_plan(source_surface="pipeline_execute")
        plan.metadata["backbone_run_id"] = "run-existing"
        plan.metadata["backbone_entrypoint"] = "prompt_engine.run"
        store.create_run(
            RunLedger(
                run_id="run-existing",
                entrypoint="prompt_engine.run",
                status="plan_ready",
                plan_id=plan.id,
                debate_id=plan.debate_id,
            )
        )

        launch = queue_plan_execution(plan, execution_mode="workflow")

        runs = store.list_runs()
        run = store.get_run("run-existing")

        assert launch["run_id"] == "run-existing"
        assert len(runs) == 1
        assert run is not None
        assert run.entrypoint == "prompt_engine.run"
        assert run.status == "execution_queued"
        assert run.execution_id == launch["execution_id"]
        assert run.metadata["source_surface"] == "pipeline_execute"
        assert any(
            event.stage == BackboneStage.EXECUTION.value
            and event.artifact_ref == launch["execution_id"]
            for event in run.stage_events
        )

    def test_queue_plan_execution_reuses_existing_stored_plan(
        self,
        store: PlanStore,
    ) -> None:
        plan, _tasks = _build_plan(source_surface="decision_integrity_payload")
        store.create(plan)

        launch = queue_plan_execution(plan, execution_mode="hybrid")
        stored_plan = store.get(plan.id)
        record = store.get_execution_record(launch["execution_id"])

        assert store.count() == 1
        assert stored_plan is not None
        assert stored_plan.metadata["backbone_run_id"] == launch["run_id"]
        assert record is not None
        assert record["metadata"]["backbone_run_id"] == launch["run_id"]

    def test_queue_plan_execution_accepts_explicit_interactive_safety_mode(
        self,
        store: PlanStore,
    ) -> None:
        plan, _tasks = _build_plan(source_surface="cli_decide")

        launch = queue_plan_execution(
            plan,
            execution_mode="workflow",
            safety_mode=ExecutionMode.INTERACTIVE,
        )
        run = store.get_run(launch["run_id"])
        record = store.get_execution_record(launch["execution_id"])

        assert run is not None
        assert run.metadata["safety_mode"] == ExecutionMode.INTERACTIVE.value
        assert record is not None
        assert record["metadata"]["safety_mode"] == ExecutionMode.INTERACTIVE.value

    def test_queue_plan_execution_fail_closes_interactive_backbone_write(
        self,
        store: PlanStore,
    ) -> None:
        plan, _tasks = _build_plan(source_surface="cli_decide")

        with patch(
            "aragora.pipeline.canonical_execution.BackboneRuntime.append_stage_event",
            return_value=False,
        ):
            with pytest.raises(RuntimeError, match="backbone run"):
                queue_plan_execution(
                    plan,
                    auth_context=SimpleNamespace(user_id="user-1"),
                    execution_mode="workflow",
                )

        assert store.list_execution_records() == []
