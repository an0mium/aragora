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
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any

from aragora.pipeline.decision_plan import DecisionPlan, PlanOutcome, PlanStatus

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

    async def execute_approved_plan(
        self,
        plan_id: str,
        *,
        auth_context: AuthorizationContext | None = None,
        execution_mode: ExecutionMode | None = None,
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
            raise ValueError(
                f"Plan {plan_id} has already been executed ({plan.status.value})"
            )
        if plan.requires_human_approval and not plan.is_approved:
            raise ValueError(f"Plan {plan_id} requires approval before execution")

        # Transition to executing in the persistent store
        store.update_status(plan_id, PlanStatus.EXECUTING)
        plan.status = PlanStatus.EXECUTING
        plan.execution_started_at = datetime.now()

        logger.info("Executing plan %s (debate: %s)", plan_id, plan.debate_id)

        try:
            outcome = await self.executor.execute(
                plan,
                auth_context=auth_context,
                execution_mode=execution_mode,
            )
        except Exception as exc:
            logger.error("Plan %s execution failed: %s", plan_id, exc)
            store.update_status(plan_id, PlanStatus.FAILED)
            raise

        # Persist final status
        final_status = PlanStatus.COMPLETED if outcome.success else PlanStatus.FAILED
        store.update_status(plan_id, final_status)

        logger.info(
            "Plan %s execution %s (%.1fs, %d/%d tasks)",
            plan_id,
            "succeeded" if outcome.success else "failed",
            outcome.duration_seconds,
            outcome.tasks_completed,
            outcome.tasks_total,
        )

        return outcome

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

        async def _run() -> None:
            try:
                await self.execute_approved_plan(
                    plan_id,
                    auth_context=auth_context,
                    execution_mode=execution_mode,
                )
            except Exception as exc:
                logger.error(
                    "Background execution of plan %s failed: %s", plan_id, exc
                )

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_run())
            logger.info("Scheduled background execution for plan %s", plan_id)
        except RuntimeError:
            logger.warning(
                "No running event loop; cannot schedule background execution for plan %s",
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
