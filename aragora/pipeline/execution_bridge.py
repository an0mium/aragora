"""Execution Bridge - connects approved DecisionPlans to the workflow engine.

When a plan is approved, the bridge:
1. Converts the DecisionPlan into a WorkflowDefinition (via plan.to_workflow_definition)
2. Hands it off to the PlanExecutor
3. Persists status transitions to PlanStore
4. Reports outcome back through the plan

Usage:
    bridge = ExecutionBridge()
    outcome = await bridge.execute_approved_plan(plan_id)

    # Or trigger execution as a fire-and-forget background task
    bridge.schedule_execution(plan_id)
"""

from __future__ import annotations

import asyncio
import logging
import threading
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
import uuid

from aragora.pipeline.backbone_contracts import (
    BackboneStage,
    OutcomeFeedbackRecord,
    ReceiptEnvelope,
    RunStageEvent,
    TaintChecker,
)
from aragora.pipeline.decision_plan import PlanOutcome, PlanStatus

if TYPE_CHECKING:
    from aragora.pipeline.executor import ExecutionMode, PlanExecutor
    from aragora.pipeline.plan_store import PlanStore
    from aragora.rbac.models import AuthorizationContext

logger = logging.getLogger(__name__)


class ExecutionBridge:
    """Bridges plan approval to workflow execution.

    Loads a plan from the persistent PlanStore, validates it, runs it through
    PlanExecutor, and records status/outcome back to PlanStore.
    """

    def __init__(
        self,
        plan_store: PlanStore | None = None,
        executor: PlanExecutor | None = None,
    ) -> None:
        self._plan_store = plan_store
        self._executor = executor

    @property
    def plan_store(self) -> PlanStore:
        if self._plan_store is None:
            from aragora.pipeline.plan_store import get_plan_store

            self._plan_store = get_plan_store()
        return self._plan_store

    @property
    def executor(self) -> PlanExecutor:
        if self._executor is None:
            from aragora.pipeline.executor import PlanExecutor

            self._executor = PlanExecutor()
        return self._executor

    @staticmethod
    def _extract_backbone_run_id(plan: Any) -> str:
        metadata = getattr(plan, "metadata", None)
        if not isinstance(metadata, dict):
            return ""
        return str(metadata.get("backbone_run_id", "") or "").strip()

    def _record_execution_stage(
        self,
        run_id: str,
        *,
        status: str,
        artifact_ref: str = "",
        details: dict[str, Any] | None = None,
    ) -> None:
        if not run_id:
            return
        self.plan_store.append_run_stage_event(
            run_id,
            RunStageEvent.create(
                BackboneStage.EXECUTION,
                status=status,
                artifact_ref=artifact_ref,
                details=details,
            ),
        )

    def _build_feedback_record(
        self,
        plan: Any,
        outcome: PlanOutcome,
        *,
        receipt_ref: str,
        execution_mode: str,
    ) -> OutcomeFeedbackRecord:
        quality_score = 1.0 if outcome.success else 0.25
        pipeline_like = SimpleNamespace(
            pipeline_id=plan.id,
            run_type="decision_plan_execution",
            domain=(
                plan.metadata.get("domain", "")
                if isinstance(getattr(plan, "metadata", None), dict)
                else ""
            ),
            overall_quality_score=quality_score,
            spec_completeness=1.0 if outcome.success else 0.5,
            execution_succeeded=outcome.success,
            tests_passed=int(getattr(outcome, "tests_passed", 0) or 0),
            tests_failed=int(getattr(outcome, "tests_failed", 0) or 0),
            files_changed=int(getattr(outcome, "files_changed", 0) or 0),
            human_interventions=0,
            rollback_triggered=bool(getattr(outcome, "rollback_triggered", False)),
            total_duration_s=outcome.duration_seconds,
        )
        next_action = ""
        if not outcome.success and int(getattr(outcome, "tests_failed", 0) or 0) > 0:
            next_action = "run_bug_fix_loop"
        return OutcomeFeedbackRecord.from_pipeline_outcome(
            pipeline_like,
            receipt_ref=receipt_ref or plan.id,
            next_action_recommendation=next_action,
        )

    async def _attach_backbone_receipt_and_feedback(
        self,
        *,
        run_id: str,
        plan: Any,
        outcome: PlanOutcome,
        execution_mode: str,
    ) -> None:
        from aragora.blockchain.receipt_settlement import ReceiptSettlementService
        from aragora.gauntlet.receipt_models import DecisionReceipt
        from aragora.pipeline.canonical_execution import build_decision_receipt_payload

        run = self.plan_store.get_run(run_id)
        receipt_payload = build_decision_receipt_payload(plan, outcome)
        receipt_ref = str(getattr(outcome, "receipt_id", "") or plan.id)

        if isinstance(receipt_payload, dict):
            taint_summary = TaintChecker.collect_taint_summary(
                intake=getattr(run, "intake_bundle", None),
                spec=getattr(run, "spec_bundle", None),
                deliberation=getattr(run, "deliberation_bundle", None),
                verification=getattr(run, "receipt_envelope", None),
            )
            metadata = getattr(plan, "metadata", None)
            gate = metadata.get("execution_gate", {}) if isinstance(metadata, dict) else {}
            envelope = ReceiptEnvelope.from_decision_receipt(
                receipt_payload,
                policy_gate_result=dict(gate or {}),
                taint_summary=taint_summary,
            )
            receipt_ref = envelope.receipt_id or receipt_ref
            self.plan_store.update_run(
                run_id,
                receipt_id=receipt_ref,
                receipt_envelope=envelope,
            )
            self.plan_store.append_run_stage_event(
                run_id,
                RunStageEvent.create(
                    BackboneStage.RECEIPT,
                    status="completed",
                    artifact_ref=receipt_ref,
                    details={"source": "decision_plan_execution"},
                ),
            )
            try:
                settled_receipt = await ReceiptSettlementService().anchor_receipt(
                    DecisionReceipt.from_dict(receipt_payload)
                )
                attestation = getattr(settled_receipt, "settlement_status", None)
                if isinstance(attestation, dict):
                    self.plan_store.update_run(run_id, attestation=attestation)
                    self.plan_store.append_run_stage_event(
                        run_id,
                        RunStageEvent.create(
                            BackboneStage.ATTESTATION,
                            status="completed",
                            artifact_ref=receipt_ref,
                            details=attestation,
                        ),
                    )
            except Exception:
                logger.debug("Backbone shadow attestation failed", exc_info=True)

        feedback_record = self._build_feedback_record(
            plan,
            outcome,
            receipt_ref=receipt_ref,
            execution_mode=execution_mode,
        )
        self.plan_store.update_run(run_id, feedback_record=feedback_record)
        self.plan_store.append_run_stage_event(
            run_id,
            RunStageEvent.create(
                BackboneStage.FEEDBACK,
                status="completed",
                artifact_ref=receipt_ref,
                details={
                    "next_action": feedback_record.next_action_recommendation,
                    "execution_mode": execution_mode,
                },
            ),
        )

    async def execute_approved_plan(
        self,
        plan_id: str,
        *,
        auth_context: AuthorizationContext | None = None,
        execution_mode: ExecutionMode | None = None,
        execution_id: str | None = None,
        correlation_id: str | None = None,
    ) -> PlanOutcome:
        """Execute an approved plan end-to-end.

        1. Load plan from store
        2. Validate it is in an executable state
        3. Transition to EXECUTING
        4. Run through PlanExecutor
        5. Record outcome and transition to COMPLETED/FAILED

        Args:
            plan_id: ID of the plan to execute.
            auth_context: Optional RBAC context for permission checks.
            execution_mode: Override default execution mode.

        Returns:
            PlanOutcome with execution results.

        Raises:
            ValueError: If plan not found or not in executable state.
        """
        store = self.plan_store
        plan = store.get(plan_id)

        if plan is None:
            raise ValueError(f"Plan not found: {plan_id}")

        if plan.status == PlanStatus.REJECTED:
            raise ValueError(f"Plan {plan_id} was rejected and cannot be executed")
        if plan.status == PlanStatus.EXECUTING:
            raise ValueError(f"Plan {plan_id} is already executing")
        if plan.status in (PlanStatus.COMPLETED, PlanStatus.FAILED):
            raise ValueError(f"Plan {plan_id} has already been executed ({plan.status.value})")
        if plan.requires_human_approval and not plan.is_approved:
            raise ValueError(f"Plan {plan_id} requires approval before execution")

        resolved_execution_id = execution_id or f"exec-{uuid.uuid4().hex[:12]}"
        resolved_correlation_id = correlation_id or f"corr-{uuid.uuid4().hex[:12]}"
        backbone_run_id = self._extract_backbone_run_id(plan)

        from aragora.pipeline.receipt_gate import evaluate_plan_execution_gate

        gate_decision = evaluate_plan_execution_gate(plan, plan_store=store)
        if not gate_decision.allow_execution:
            blocked_status = (
                "pending_approval" if gate_decision.requires_human_approval else "blocked"
            )
            if store.get_execution_record(resolved_execution_id) is None:
                store.create_execution_record(
                    execution_id=resolved_execution_id,
                    plan_id=plan.id,
                    debate_id=plan.debate_id,
                    correlation_id=resolved_correlation_id,
                    status=blocked_status,
                    metadata={
                        "execution_mode": execution_mode or "default",
                        "backbone_run_id": backbone_run_id or None,
                        "execution_gate": gate_decision.gate,
                    },
                )
            else:
                store.update_execution_record(
                    resolved_execution_id,
                    status=blocked_status,
                    metadata={"execution_gate": gate_decision.gate},
                )
            if backbone_run_id:
                store.update_run(
                    backbone_run_id,
                    status=blocked_status,
                    execution_id=resolved_execution_id,
                    metadata={"execution_gate": gate_decision.gate},
                )
                self._record_execution_stage(
                    backbone_run_id,
                    status=blocked_status,
                    artifact_ref=resolved_execution_id,
                    details={"plan_id": plan.id, "reason_codes": gate_decision.reason_codes},
                )
            raise ValueError(
                f"Plan {plan_id} blocked by execution gate ({', '.join(gate_decision.reason_codes)})"
            )

        expected_statuses = [PlanStatus.APPROVED]
        if not plan.requires_human_approval:
            expected_statuses.append(PlanStatus.CREATED)

        # Atomic claim: ensures exactly one execution worker can transition the plan.
        claimed = store.update_status_if_current(
            plan_id,
            expected_statuses=expected_statuses,
            new_status=PlanStatus.EXECUTING,
        )
        if not claimed:
            current = store.get(plan_id)
            if current is None:
                raise ValueError(f"Plan not found: {plan_id}")
            if current.status == PlanStatus.EXECUTING:
                raise ValueError(f"Plan {plan_id} is already executing")
            if current.status in (PlanStatus.COMPLETED, PlanStatus.FAILED):
                raise ValueError(
                    f"Plan {plan_id} has already been executed ({current.status.value})"
                )
            raise ValueError(
                f"Plan {plan_id} is not in an executable state ({current.status.value})"
            )

        plan.status = PlanStatus.EXECUTING
        plan.execution_started_at = datetime.now()
        if store.get_execution_record(resolved_execution_id) is None:
            store.create_execution_record(
                execution_id=resolved_execution_id,
                plan_id=plan.id,
                debate_id=plan.debate_id,
                correlation_id=resolved_correlation_id,
                status="running",
                metadata={
                    "execution_mode": execution_mode or "default",
                    "started_by": getattr(auth_context, "user_id", None),
                    "backbone_run_id": backbone_run_id or None,
                    "execution_gate": gate_decision.gate,
                },
            )
        else:
            store.update_execution_record(
                resolved_execution_id,
                status="running",
                metadata={
                    "execution_mode": execution_mode or "default",
                    "started_by": getattr(auth_context, "user_id", None),
                    "backbone_run_id": backbone_run_id or None,
                    "execution_gate": gate_decision.gate,
                },
            )
        if backbone_run_id:
            store.update_run(
                backbone_run_id,
                status="execution_running",
                execution_id=resolved_execution_id,
                metadata={
                    "execution_mode": execution_mode or "default",
                    "execution_gate": gate_decision.gate,
                },
            )
            self._record_execution_stage(
                backbone_run_id,
                status="running",
                artifact_ref=resolved_execution_id,
                details={
                    "plan_id": plan.id,
                    "execution_mode": execution_mode or "default",
                },
            )

        logger.info("Executing plan %s (debate: %s)", plan_id, plan.debate_id)

        try:
            outcome = await self.executor.execute(
                plan,
                auth_context=auth_context,
                execution_mode=execution_mode,
            )
        except Exception as exc:  # noqa: BLE001 - intentional broad catch to record failure before re-raising
            logger.error("Plan %s execution failed: %s", plan_id, exc)
            store.update_status(plan_id, PlanStatus.FAILED)
            store.update_execution_record(
                resolved_execution_id,
                status="failed",
                error={
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "at": datetime.now(timezone.utc).isoformat(),
                },
                metadata={
                    "execution_mode": execution_mode or "default",
                    "terminal_state": "failed",
                },
            )
            if backbone_run_id:
                store.update_run(
                    backbone_run_id,
                    status="execution_failed",
                    execution_id=resolved_execution_id,
                    metadata={"execution_terminal_state": "failed"},
                )
                self._record_execution_stage(
                    backbone_run_id,
                    status="failed",
                    artifact_ref=resolved_execution_id,
                    details={"error": str(exc), "plan_id": plan.id},
                )
            raise

        # Persist final status
        final_status = PlanStatus.COMPLETED if outcome.success else PlanStatus.FAILED
        store.update_status(plan_id, final_status)
        failure_error = None
        if not outcome.success:
            failure_error = {
                "type": "ExecutionFailure",
                "message": outcome.error or "Execution returned unsuccessful outcome",
                "at": datetime.now(timezone.utc).isoformat(),
            }
        store.update_execution_record(
            resolved_execution_id,
            status="succeeded" if outcome.success else "failed",
            error=failure_error,
            metadata={
                "execution_mode": execution_mode or "default",
                "duration_seconds": outcome.duration_seconds,
                "tasks_completed": outcome.tasks_completed,
                "tasks_total": outcome.tasks_total,
                "terminal_state": "succeeded" if outcome.success else "failed",
            },
        )
        if backbone_run_id:
            store.update_run(
                backbone_run_id,
                status="execution_succeeded" if outcome.success else "execution_failed",
                execution_id=resolved_execution_id,
                metadata={
                    "execution_terminal_state": "succeeded" if outcome.success else "failed",
                    "execution_duration_seconds": outcome.duration_seconds,
                },
            )
            self._record_execution_stage(
                backbone_run_id,
                status="succeeded" if outcome.success else "failed",
                artifact_ref=resolved_execution_id,
                details={
                    "plan_id": plan.id,
                    "duration_seconds": outcome.duration_seconds,
                    "tasks_completed": outcome.tasks_completed,
                    "tasks_total": outcome.tasks_total,
                },
            )
            await self._attach_backbone_receipt_and_feedback(
                run_id=backbone_run_id,
                plan=plan,
                outcome=outcome,
                execution_mode=str(execution_mode or "default"),
            )

        logger.info(
            "Plan %s execution %s (%.1fs, %d/%d tasks)",
            plan_id,
            "succeeded" if outcome.success else "failed",
            outcome.duration_seconds,
            outcome.tasks_completed,
            outcome.tasks_total,
        )

        return outcome

    def get_execution_record(self, execution_id: str) -> dict[str, Any] | None:
        """Fetch one execution record."""
        return self.plan_store.get_execution_record(execution_id)

    def list_execution_records(
        self,
        *,
        plan_id: str | None = None,
        debate_id: str | None = None,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List execution records for plan/debate lookups."""
        return self.plan_store.list_execution_records(
            plan_id=plan_id,
            debate_id=debate_id,
            status=status,
            limit=limit,
            offset=offset,
        )

    def schedule_execution(
        self,
        plan_id: str,
        *,
        auth_context: AuthorizationContext | None = None,
        execution_mode: ExecutionMode | None = None,
    ) -> None:
        """Schedule plan execution as a background asyncio task.

        Non-blocking: returns immediately. Errors are logged, not raised.
        """

        record_execution_id = f"exec-{uuid.uuid4().hex[:12]}"
        record_correlation_id = f"corr-{uuid.uuid4().hex[:12]}"
        plan = self.plan_store.get(plan_id)
        backbone_run_id = self._extract_backbone_run_id(plan) if plan is not None else ""
        if plan is not None:
            from aragora.pipeline.receipt_gate import evaluate_plan_execution_gate

            gate_decision = evaluate_plan_execution_gate(plan, plan_store=self.plan_store)
            if not gate_decision.allow_execution:
                blocked_status = (
                    "pending_approval" if gate_decision.requires_human_approval else "blocked"
                )
                self.plan_store.create_execution_record(
                    execution_id=record_execution_id,
                    plan_id=plan.id,
                    debate_id=plan.debate_id,
                    correlation_id=record_correlation_id,
                    status=blocked_status,
                    metadata={
                        "execution_mode": execution_mode or "default",
                        "scheduled_by": getattr(auth_context, "user_id", None),
                        "backbone_run_id": backbone_run_id or None,
                        "execution_gate": gate_decision.gate,
                    },
                )
                if backbone_run_id:
                    self.plan_store.update_run(
                        backbone_run_id,
                        status=blocked_status,
                        execution_id=record_execution_id,
                        metadata={"execution_gate": gate_decision.gate},
                    )
                    self._record_execution_stage(
                        backbone_run_id,
                        status=blocked_status,
                        artifact_ref=record_execution_id,
                        details={"plan_id": plan.id, "reason_codes": gate_decision.reason_codes},
                    )
                logger.warning(
                    "Execution scheduling blocked for plan %s: %s",
                    plan.id,
                    gate_decision.reason_codes,
                )
                return
            self.plan_store.create_execution_record(
                execution_id=record_execution_id,
                plan_id=plan.id,
                debate_id=plan.debate_id,
                correlation_id=record_correlation_id,
                status="queued",
                metadata={
                    "execution_mode": execution_mode or "default",
                    "scheduled_by": getattr(auth_context, "user_id", None),
                    "backbone_run_id": backbone_run_id or None,
                },
            )
            if backbone_run_id:
                self.plan_store.update_run(
                    backbone_run_id,
                    status="execution_queued",
                    execution_id=record_execution_id,
                    metadata={"execution_mode": execution_mode or "default"},
                )
                self._record_execution_stage(
                    backbone_run_id,
                    status="queued",
                    artifact_ref=record_execution_id,
                    details={
                        "plan_id": plan.id,
                        "execution_mode": execution_mode or "default",
                    },
                )

        async def _run() -> None:
            try:
                await self.execute_approved_plan(
                    plan_id,
                    auth_context=auth_context,
                    execution_mode=execution_mode,
                    execution_id=record_execution_id,
                    correlation_id=record_correlation_id,
                )
            except Exception as exc:  # noqa: BLE001 - intentional broad catch for fire-and-forget background task
                logger.error("Background execution of plan %s failed: %s", plan_id, exc)

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_run())
            logger.info("Scheduled background execution for plan %s", plan_id)
        except RuntimeError:

            def _run_in_thread() -> None:
                asyncio.run(_run())

            worker = threading.Thread(
                target=_run_in_thread,
                name=f"plan-exec-{plan_id[:8]}",
                daemon=True,
            )
            worker.start()
            logger.info(
                "Scheduled background execution for plan %s via dedicated thread",
                plan_id,
            )


# Module-level singleton
_bridge: ExecutionBridge | None = None


def get_execution_bridge() -> ExecutionBridge:
    """Return module-level ExecutionBridge singleton."""
    global _bridge
    if _bridge is None:
        _bridge = ExecutionBridge()
    return _bridge


def reset_execution_bridge() -> None:
    """Reset the singleton (for testing)."""
    global _bridge
    _bridge = None
