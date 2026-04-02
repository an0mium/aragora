"""Contract tests for execution-mode backbone parity."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from aragora.pipeline.backbone_contracts import BackboneStage, RunLedger
from aragora.pipeline.decision_plan import PlanOutcome, PlanStatus
from aragora.pipeline.decision_plan.core import (
    ApprovalMode,
    ApprovalRecord,
    BudgetAllocation,
    DecisionPlan,
)
from aragora.pipeline.execution_bridge import ExecutionBridge
from aragora.pipeline.executor import PlanExecutor
from aragora.pipeline.plan_store import PlanStore

MODE_CASES = (
    pytest.param("workflow", id="workflow-passes"),
    pytest.param("hybrid", id="hybrid-passes"),
    pytest.param("fabric", id="fabric-passes"),
    pytest.param("computer_use", id="computer_use-passes"),
)


@pytest.fixture
def store(tmp_path: Path) -> PlanStore:
    return PlanStore(db_path=str(tmp_path / "execution-mode-parity.db"))


def _make_plan(*, plan_id: str, mode: str, run_id: str) -> DecisionPlan:
    return DecisionPlan(
        id=plan_id,
        debate_id=f"debate-{mode}",
        task=f"Execute {mode} contract path",
        status=PlanStatus.APPROVED,
        approval_mode=ApprovalMode.ALWAYS,
        approval_record=ApprovalRecord(approved=True, approver_id="user-42"),
        budget=BudgetAllocation(limit_usd=25.0),
        metadata={"backbone_run_id": run_id},
    )


def _successful_outcome(plan: DecisionPlan, *, mode: str) -> PlanOutcome:
    return PlanOutcome(
        plan_id=plan.id,
        debate_id=plan.debate_id,
        task=plan.task,
        success=True,
        tasks_completed=1,
        tasks_total=1,
        duration_seconds=0.25,
        receipt_id=f"receipt-{mode}",
    )


def _build_executor(*, mode: str, monkeypatch: pytest.MonkeyPatch) -> PlanExecutor:
    executor = PlanExecutor(
        continuum_memory=object(),
        knowledge_mound=object(),
        execution_mode=mode,
    )

    targeted_runner = AsyncMock(
        return_value=_successful_outcome(
            DecisionPlan(
                id="unused",
                debate_id="unused",
                task="unused",
                budget=BudgetAllocation(limit_usd=1.0),
            ),
            mode=mode,
        )
    )

    async def _run_for_mode(plan: DecisionPlan, **_: object) -> PlanOutcome:
        return _successful_outcome(plan, mode=mode)

    targeted_runner.side_effect = _run_for_mode

    unexpected = AsyncMock(side_effect=AssertionError("unexpected execution backend"))

    monkeypatch.setattr(
        executor,
        "_run_workflow",
        targeted_runner if mode == "workflow" else unexpected,
    )
    monkeypatch.setattr(
        executor,
        "_run_hybrid",
        targeted_runner if mode == "hybrid" else unexpected,
    )
    monkeypatch.setattr(
        executor,
        "_run_fabric",
        targeted_runner if mode == "fabric" else unexpected,
    )
    monkeypatch.setattr(
        executor,
        "_run_computer_use",
        targeted_runner if mode == "computer_use" else unexpected,
    )
    monkeypatch.setattr(
        executor,
        "_generate_receipt",
        AsyncMock(return_value=SimpleNamespace(receipt_id=f"receipt-{mode}")),
    )
    monkeypatch.setattr(executor, "_ingest_to_km", AsyncMock(return_value=None))
    monkeypatch.setattr(
        "aragora.pipeline.executor.record_plan_outcome",
        AsyncMock(return_value=None),
    )

    return executor


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODE_CASES)
async def test_execution_modes_emit_minimum_backbone_artifacts(
    store: PlanStore,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    run_id = f"run-{mode}"
    plan = _make_plan(plan_id=f"dp-{mode}", mode=mode, run_id=run_id)
    store.create_run(
        RunLedger(
            run_id=run_id,
            entrypoint="prompt_engine.run",
            status="plan_ready",
            plan_id=plan.id,
        )
    )
    store.create(plan)

    executor = _build_executor(mode=mode, monkeypatch=monkeypatch)
    bridge = ExecutionBridge(plan_store=store, executor=executor)

    outcome = await bridge.execute_approved_plan(plan.id, execution_mode=mode)
    run = store.get_run(run_id)

    assert outcome.success is True
    assert run is not None
    assert run.execution_id
    assert run.stage_events
    assert any(event.stage == BackboneStage.EXECUTION.value for event in run.stage_events)
    assert any(event.stage == BackboneStage.RECEIPT.value for event in run.stage_events)
    assert any(event.stage == BackboneStage.FEEDBACK.value for event in run.stage_events)
    assert run.receipt_envelope is not None or run.receipt_id
    assert run.feedback_record is not None
    assert run.feedback_record.receipt_ref
    assert run.feedback_record.next_action_recommendation

    feedback_events = [
        event
        for event in run.stage_events
        if event.stage == BackboneStage.FEEDBACK.value and event.status == "completed"
    ]
    assert feedback_events
    assert feedback_events[-1].details.get("execution_mode") == mode
