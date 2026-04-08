"""Tests for BackboneRuntime ledger side-effect orchestration."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from aragora.gauntlet.receipt_store import reset_receipt_store
from aragora.pipeline.backbone_contracts import BackboneStage, RunLedger
from aragora.pipeline.backbone_runtime import BackboneRuntime
from aragora.pipeline.decision_plan.core import (
    ApprovalMode,
    ApprovalRecord,
    DecisionPlan,
    PlanStatus,
)
from aragora.pipeline.decision_plan.memory import PlanOutcome
from aragora.pipeline.plan_store import PlanStore


@pytest.fixture
def store(tmp_path: Path) -> PlanStore:
    return PlanStore(db_path=str(tmp_path / "backbone_runtime.db"))


@pytest.fixture(autouse=True)
def _reset_receipts() -> None:
    reset_receipt_store()
    yield
    reset_receipt_store()


class TestBackboneRuntime:
    def test_sync_plan_receipt_to_run_persists_envelope_and_event(self, store: PlanStore) -> None:
        runtime = BackboneRuntime(store)
        store.create_run(
            RunLedger(run_id="run-sync", entrypoint="prompt_engine.run", status="plan_ready")
        )
        plan = DecisionPlan(
            id="dp-sync",
            debate_id="debate-sync",
            task="Sync the receipt",
            status=PlanStatus.AWAITING_APPROVAL,
            approval_mode=ApprovalMode.ALWAYS,
            metadata={"backbone_run_id": "run-sync"},
        )

        store.create(plan)
        updated_plan = store.get(plan.id)
        assert updated_plan is not None
        store.update_status(plan.id, PlanStatus.APPROVED, approved_by="approver-1")
        updated_plan = store.get(plan.id)
        assert updated_plan is not None

        synced = runtime.sync_plan_receipt_to_run(updated_plan, append_event=True)
        run = store.get_run("run-sync")

        assert synced is True
        assert run is not None
        assert run.receipt_envelope is not None
        assert run.metadata["plan_receipt_state"] == "approved"
        assert any(
            event.stage == BackboneStage.RECEIPT.value and event.status == "approved"
            for event in run.stage_events
        )

    def test_record_execution_stage_updates_run_and_event(self, store: PlanStore) -> None:
        runtime = BackboneRuntime(store)
        store.create_run(
            RunLedger(run_id="run-exec", entrypoint="prompt_engine.run", status="plan_ready")
        )

        recorded = runtime.record_execution_stage(
            "run-exec",
            status="queued",
            artifact_ref="exec-123",
            run_status="execution_queued",
            execution_id="exec-123",
            metadata={"execution_mode": "workflow"},
            details={"plan_id": "dp-sync"},
        )
        run = store.get_run("run-exec")

        assert recorded is True
        assert run is not None
        assert run.status == "execution_queued"
        assert run.execution_id == "exec-123"
        assert run.metadata["execution_mode"] == "workflow"
        assert any(
            event.stage == BackboneStage.EXECUTION.value and event.artifact_ref == "exec-123"
            for event in run.stage_events
        )

    def test_sync_plan_receipt_to_run_logs_warning_when_fetch_fails(
        self,
        store: PlanStore,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        runtime = BackboneRuntime(store)
        plan = DecisionPlan(
            id="dp-fetch-fail",
            debate_id="debate-fetch-fail",
            task="Sync the receipt",
            status=PlanStatus.APPROVED,
            approval_mode=ApprovalMode.ALWAYS,
            metadata={"backbone_run_id": "run-fetch-fail", "decision_receipt_id": "receipt-1"},
        )

        with (
            patch(
                "aragora.pipeline.receipt_store_facade.get_receipt_store_facade",
                side_effect=RuntimeError("facade unavailable"),
            ),
            caplog.at_level(logging.WARNING),
        ):
            assert runtime.sync_plan_receipt_to_run(plan) is False

        assert "Backbone receipt fetch failed for receipt-1: facade unavailable" in caplog.text

    @pytest.mark.asyncio
    async def test_attach_execution_receipt_persists_attestation(self, store: PlanStore) -> None:
        runtime = BackboneRuntime(store)
        store.create_run(
            RunLedger(run_id="run-receipt", entrypoint="prompt_engine.run", status="plan_ready")
        )
        plan = DecisionPlan(
            id="dp-receipt",
            debate_id="debate-receipt",
            task="Execute with receipt",
            status=PlanStatus.APPROVED,
            approval_mode=ApprovalMode.ALWAYS,
            approval_record=ApprovalRecord(approved=True, approver_id="user-1"),
            metadata={"backbone_run_id": "run-receipt", "execution_gate": {}},
        )
        outcome = PlanOutcome(
            plan_id=plan.id,
            debate_id=plan.debate_id,
            task=plan.task,
            success=True,
            tasks_completed=1,
            tasks_total=1,
            duration_seconds=0.1,
            receipt_id="receipt-runtime-1",
        )

        receipt_ref = await runtime.attach_execution_receipt("run-receipt", plan, outcome)
        run = store.get_run("run-receipt")

        assert receipt_ref == "receipt-runtime-1"
        assert run is not None
        assert run.receipt_envelope is not None
        assert run.receipt_envelope.receipt_id == "receipt-runtime-1"
        assert run.attestation.get("local_only") is True
        assert any(
            event.stage == BackboneStage.RECEIPT.value and event.status == "completed"
            for event in run.stage_events
        )
        assert any(
            event.stage == BackboneStage.ATTESTATION.value and event.status == "completed"
            for event in run.stage_events
        )

    @pytest.mark.asyncio
    async def test_attach_execution_receipt_logs_warning_when_attestation_fails(
        self,
        store: PlanStore,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        runtime = BackboneRuntime(store)
        store.create_run(
            RunLedger(run_id="run-attestation-fail", entrypoint="prompt_engine.run", status="ready")
        )
        plan = DecisionPlan(
            id="dp-attestation-fail",
            debate_id="debate-attestation-fail",
            task="Execute with receipt",
            status=PlanStatus.APPROVED,
            approval_mode=ApprovalMode.ALWAYS,
            approval_record=ApprovalRecord(approved=True, approver_id="user-1"),
            metadata={"backbone_run_id": "run-attestation-fail", "execution_gate": {}},
        )
        outcome = PlanOutcome(
            plan_id=plan.id,
            debate_id=plan.debate_id,
            task=plan.task,
            success=True,
            tasks_completed=1,
            tasks_total=1,
            duration_seconds=0.1,
            receipt_id="receipt-runtime-2",
        )

        with (
            patch(
                "aragora.blockchain.receipt_settlement.ReceiptSettlementService.anchor_receipt",
                side_effect=RuntimeError("anchor unavailable"),
            ),
            caplog.at_level(logging.WARNING),
        ):
            receipt_ref = await runtime.attach_execution_receipt(
                "run-attestation-fail",
                plan,
                outcome,
            )

        assert receipt_ref == "receipt-runtime-2"
        assert (
            "Backbone shadow attestation failed for receipt-runtime-2: anchor unavailable"
            in caplog.text
        )

    def test_attach_feedback_record_persists_feedback_event(self, store: PlanStore) -> None:
        runtime = BackboneRuntime(store)
        store.create_run(
            RunLedger(run_id="run-feedback", entrypoint="prompt_engine.run", status="plan_ready")
        )
        plan = DecisionPlan(
            id="dp-feedback",
            debate_id="debate-feedback",
            task="Feedback task",
            status=PlanStatus.APPROVED,
            approval_mode=ApprovalMode.ALWAYS,
            approval_record=ApprovalRecord(approved=True, approver_id="user-1"),
        )
        outcome = PlanOutcome(
            plan_id=plan.id,
            debate_id=plan.debate_id,
            task=plan.task,
            success=False,
            tasks_completed=0,
            tasks_total=1,
            duration_seconds=0.5,
            error="tests failed",
            receipt_id="receipt-feedback-1",
        )
        record = runtime.build_feedback_record(
            plan,
            outcome,
            receipt_ref="receipt-feedback-1",
            execution_mode="workflow",
        )

        attached = runtime.attach_feedback_record(
            "run-feedback",
            record,
            artifact_ref="receipt-feedback-1",
            details={"execution_mode": "workflow"},
        )
        run = store.get_run("run-feedback")

        assert attached is True
        assert run is not None
        assert run.feedback_record is not None
        assert run.feedback_record.receipt_ref == "receipt-feedback-1"
        assert any(
            event.stage == BackboneStage.FEEDBACK.value and event.status == "completed"
            for event in run.stage_events
        )
