"""Decision Plan Executor - Bridge from DecisionPlan to WorkflowEngine.

Provides the glue code to execute an approved DecisionPlan through the
WorkflowEngine, track progress, and record outcomes back to organizational
memory.

Usage:
    executor = PlanExecutor()
    outcome = await executor.execute(plan)

The executor also manages an in-memory plan store for retrieval by
plan_id, enabling the HTTP handler to look up plans across the lifecycle.

Stability: ALPHA
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any

from aragora.pipeline.decision_plan import (
    DecisionPlan,
    PlanOutcome,
    PlanStatus,
    record_plan_outcome,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-memory plan store (upgrade to persistent storage later)
# ---------------------------------------------------------------------------

_plan_store: dict[str, DecisionPlan] = {}
_plan_outcomes: dict[str, PlanOutcome] = {}

# Maximum number of plans to keep in memory
_MAX_PLANS = 1000


def store_plan(plan: DecisionPlan) -> None:
    """Store a plan in the in-memory store."""
    if len(_plan_store) >= _MAX_PLANS:
        # Evict oldest completed plan
        for pid, p in list(_plan_store.items()):
            if p.status in (PlanStatus.COMPLETED, PlanStatus.FAILED, PlanStatus.REJECTED):
                del _plan_store[pid]
                break
    _plan_store[plan.id] = plan


def get_plan(plan_id: str) -> DecisionPlan | None:
    """Retrieve a plan by ID."""
    return _plan_store.get(plan_id)


def list_plans(
    status: PlanStatus | None = None,
    limit: int = 50,
) -> list[DecisionPlan]:
    """List plans, optionally filtered by status."""
    plans = list(_plan_store.values())
    if status is not None:
        plans = [p for p in plans if p.status == status]
    plans.sort(key=lambda p: p.created_at, reverse=True)
    return plans[:limit]


def get_outcome(plan_id: str) -> PlanOutcome | None:
    """Retrieve a plan outcome by plan ID."""
    return _plan_outcomes.get(plan_id)


# ---------------------------------------------------------------------------
# Plan Executor
# ---------------------------------------------------------------------------


class PlanExecutor:
    """Executes an approved DecisionPlan through the WorkflowEngine.

    Lifecycle:
        1. Validates plan is approved
        2. Generates WorkflowDefinition from plan
        3. Executes via WorkflowEngine
        4. Records PlanOutcome to memory
        5. Updates plan status

    The executor is stateless; all state lives in the plan itself
    and the plan store.
    """

    def __init__(
        self,
        continuum_memory: Any | None = None,
        knowledge_mound: Any | None = None,
        parallel_execution: bool = False,
        max_parallel: int | None = None,
    ) -> None:
        if continuum_memory is None:
            try:
                from aragora.memory.continuum import get_continuum_memory

                continuum_memory = get_continuum_memory()
            except Exception:
                continuum_memory = None
        if knowledge_mound is None:
            try:
                from aragora.knowledge.mound import get_knowledge_mound

                knowledge_mound = get_knowledge_mound()
            except Exception:
                knowledge_mound = None

        self._continuum_memory = continuum_memory
        self._knowledge_mound = knowledge_mound
        self._parallel_execution = parallel_execution
        self._max_parallel = max_parallel

    async def execute(
        self,
        plan: DecisionPlan,
        *,
        parallel_execution: bool | None = None,
    ) -> PlanOutcome:
        """Execute a DecisionPlan and return the outcome.

        Args:
            plan: An approved DecisionPlan.

        Returns:
            PlanOutcome with execution results.

        Raises:
            ValueError: If the plan is not in an executable state.
        """
        if plan.status == PlanStatus.REJECTED:
            raise ValueError(f"Plan {plan.id} was rejected and cannot be executed")
        if plan.status == PlanStatus.EXECUTING:
            raise ValueError(f"Plan {plan.id} is already executing")
        if plan.status in (PlanStatus.COMPLETED, PlanStatus.FAILED):
            raise ValueError(f"Plan {plan.id} has already been executed ({plan.status.value})")

        if plan.requires_human_approval and not plan.is_approved:
            raise ValueError(f"Plan {plan.id} requires approval before execution")

        # Transition to executing
        plan.status = PlanStatus.EXECUTING
        plan.execution_started_at = datetime.now()
        store_plan(plan)

        start_time = time.time()
        outcome: PlanOutcome

        if parallel_execution is None:
            parallel_execution = self._parallel_execution

        try:
            outcome = await self._run_workflow(plan, parallel_execution=parallel_execution)
        except Exception as e:
            logger.error("Plan execution failed: %s: %s", plan.id, e)
            duration = time.time() - start_time
            outcome = PlanOutcome(
                plan_id=plan.id,
                debate_id=plan.debate_id,
                task=plan.task,
                success=False,
                error=str(e),
                duration_seconds=duration,
                tasks_total=len(plan.implement_plan.tasks) if plan.implement_plan else 0,
            )

        # Record outcome
        _plan_outcomes[plan.id] = outcome

        # Write back to memory (best-effort)
        try:
            await record_plan_outcome(
                plan,
                outcome,
                continuum_memory=self._continuum_memory,
                knowledge_mound=self._knowledge_mound,
            )
        except Exception as e:
            logger.warning("Failed to record plan outcome to memory: %s", e)

        # Update store with final state
        store_plan(plan)

        return outcome

    async def _run_workflow(
        self,
        plan: DecisionPlan,
        *,
        parallel_execution: bool = False,
    ) -> PlanOutcome:
        """Run the workflow engine against the plan's generated definition."""
        engine: Any  # Union of WorkflowEngine and EnhancedWorkflowEngine
        if parallel_execution:
            from aragora.workflow.engine_v2 import EnhancedWorkflowEngine, ResourceLimits

            limits = ResourceLimits(
                max_parallel_agents=self._max_parallel
                if self._max_parallel is not None
                else ResourceLimits().max_parallel_agents
            )
            engine = EnhancedWorkflowEngine(limits=limits)
            definition = plan.to_workflow_definition(parallelize=True)
        else:
            from aragora.workflow.engine import WorkflowEngine

            engine = WorkflowEngine()
            definition = plan.to_workflow_definition()

        # Execute
        start_time = time.time()
        result = await engine.execute(
            definition,
            inputs={
                "plan_id": plan.id,
                "debate_id": plan.debate_id,
                "task": plan.task,
            },
            workflow_id=plan.workflow_id,
        )

        duration = time.time() - start_time

        # Compute tasks/verification stats from workflow results
        tasks_total = len(plan.implement_plan.tasks) if plan.implement_plan else 0
        tasks_completed = 0
        verification_total = 0
        verification_passed = 0

        for step_result in getattr(result, "step_results", []):
            step_name = getattr(step_result, "step_name", "") or ""
            if step_name.startswith("Implement:"):
                if getattr(step_result, "success", False):
                    tasks_completed += 1
            elif step_name == "Run Verification":
                output = getattr(step_result, "output", {}) or {}
                verification_total = output.get("test_count", 0)
                verification_passed = output.get("passed", 0)

        # Determine overall cost
        total_cost = plan.budget.spent_usd
        for step_result in getattr(result, "step_results", []):
            cost = getattr(step_result, "cost_usd", 0.0) or 0.0
            total_cost += cost

        success = getattr(result, "success", False)
        error = getattr(result, "error", None)

        # Derive lessons from the execution
        lessons: list[str] = []
        if not success and error:
            lessons.append(f"Execution failed: {error}")
        if tasks_completed < tasks_total:
            lessons.append(f"Only {tasks_completed}/{tasks_total} tasks completed")
        if verification_total > 0 and verification_passed < verification_total:
            lessons.append(f"Verification: {verification_passed}/{verification_total} passed")

        return PlanOutcome(
            plan_id=plan.id,
            debate_id=plan.debate_id,
            task=plan.task,
            success=success,
            tasks_completed=tasks_completed,
            tasks_total=tasks_total,
            verification_passed=verification_passed,
            verification_total=verification_total,
            total_cost_usd=total_cost,
            error=error,
            duration_seconds=duration,
            lessons=lessons,
        )
