"""Decision Plan Executor - Bridge from DecisionPlan to WorkflowEngine.

Provides the glue code to execute an approved DecisionPlan through the
WorkflowEngine, track progress, and record outcomes back to organizational
memory.

Usage:
    executor = PlanExecutor()
    outcome = await executor.execute(plan)

    # Use HybridExecutor for multi-model execution (Claude + Codex)
    executor = PlanExecutor(execution_mode="hybrid", repo_path=Path.cwd())
    outcome = await executor.execute(plan)

    # Use Computer Use for browser-based implementation
    executor = PlanExecutor(execution_mode="computer_use")
    outcome = await executor.execute(plan)

The executor also manages an in-memory plan store for retrieval by
plan_id, enabling the HTTP handler to look up plans across the lifecycle.

Execution Modes:
    - "workflow": Uses WorkflowEngine with DAG-based step execution (default)
    - "hybrid": Uses HybridExecutor with Claude primary + Codex fallback
    - "fabric": Uses AgentFabric for multi-agent implementation execution
    - "computer_use": Uses ComputerUseOrchestrator for browser automation

Stability: ALPHA
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from aragora.pipeline.decision_plan import (
    DecisionPlan,
    PlanOutcome,
    PlanStatus,
    record_plan_outcome,
)

if TYPE_CHECKING:
    from aragora.rbac.models import AuthorizationContext

logger = logging.getLogger(__name__)

# Execution mode type
ExecutionMode = Literal["workflow", "hybrid", "fabric", "computer_use"]

# Environment variable to control default execution mode
DEFAULT_EXECUTION_MODE: ExecutionMode = os.environ.get("PLAN_EXECUTION_MODE", "workflow")  # type: ignore[assignment]

# Permission required to execute plans
PLAN_EXECUTE_PERMISSION = "decisions:execute"

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
    """Executes an approved DecisionPlan through WorkflowEngine or HybridExecutor.

    Lifecycle:
        1. Validates plan is approved
        2. Generates WorkflowDefinition from plan (workflow mode) or
           prepares ImplementTasks (hybrid mode)
        3. Executes via WorkflowEngine or HybridExecutor
        4. Records PlanOutcome to memory
        5. Updates plan status

    Execution Modes:
        - "workflow": DAG-based workflow engine with generic step execution
        - "hybrid": Multi-model executor with Claude + Codex (faster, model fallback)
        - "fabric": AgentFabric-managed multi-agent execution
        - "computer_use": Browser-based implementation via computer use

    The executor is stateless; all state lives in the plan itself
    and the plan store.
    """

    def __init__(
        self,
        continuum_memory: Any | None = None,
        knowledge_mound: Any | None = None,
        parallel_execution: bool = False,
        max_parallel: int | None = None,
        execution_mode: ExecutionMode | None = None,
        repo_path: Path | None = None,
    ) -> None:
        if continuum_memory is None:
            try:
                from aragora.memory.continuum import get_continuum_memory

                continuum_memory = get_continuum_memory()
            except Exception as e:
                logger.debug("Could not initialize continuum_memory: %s", e)
                continuum_memory = None
        if knowledge_mound is None:
            try:
                from aragora.knowledge.mound import get_knowledge_mound

                knowledge_mound = get_knowledge_mound()
            except Exception as e:
                logger.debug("Could not initialize knowledge_mound: %s", e)
                knowledge_mound = None

        self._continuum_memory = continuum_memory
        self._knowledge_mound = knowledge_mound
        self._parallel_execution = parallel_execution
        self._max_parallel = max_parallel
        self._execution_mode: ExecutionMode = execution_mode or DEFAULT_EXECUTION_MODE
        self._repo_path = repo_path or Path.cwd()

    @staticmethod
    def _emit_plan_event(event_type: str, data: dict[str, Any]) -> None:
        """Emit a pipeline lifecycle event (best-effort)."""
        try:
            from aragora.events.types import StreamEvent, StreamEventType

            event = StreamEvent(
                type=StreamEventType[event_type],
                data=data,
            )
            # Try to dispatch via the server event emitter if available
            try:
                from aragora.server.stream.emitter import get_global_emitter

                emitter = get_global_emitter()
                if emitter is not None:
                    emitter.emit(event)
            except (ImportError, AttributeError):
                pass
            logger.info("Pipeline event: %s plan_id=%s", event_type, data.get("plan_id", ""))
        except Exception as exc:
            logger.debug("Failed to emit pipeline event %s: %s", event_type, exc)

    @staticmethod
    def _requires_workflow(plan: DecisionPlan) -> bool:
        """Return True if plan tasks require workflow-only steps."""
        if not plan.implement_plan:
            return False
        for task in plan.implement_plan.tasks:
            task_type = str(getattr(task, "task_type", "") or "").lower()
            caps = {str(c).lower() for c in (getattr(task, "capabilities", []) or []) if c}
            if task_type in {"computer_use", "manual", "browser", "ui"}:
                return True
            if "computer_use" in caps or "manual" in caps or "browser" in caps:
                return True
        return False

    async def execute(
        self,
        plan: DecisionPlan,
        *,
        parallel_execution: bool | None = None,
        auth_context: AuthorizationContext | None = None,
        execution_mode: ExecutionMode | None = None,
        on_task_complete: Any | None = None,
    ) -> PlanOutcome:
        """Execute a DecisionPlan and return the outcome.

        Args:
            plan: An approved DecisionPlan.
            parallel_execution: Whether to execute tasks in parallel.
            auth_context: Authorization context for the requesting user.
                If provided, permission checks are enforced.
                If None, execution proceeds (for internal/system calls).
            execution_mode: Override the default execution mode for this call.
                "workflow" uses WorkflowEngine, "hybrid" uses HybridExecutor,
                "fabric" uses AgentFabric orchestration.
            on_task_complete: Optional callback invoked per task completion.

        Returns:
            PlanOutcome with execution results.

        Raises:
            ValueError: If the plan is not in an executable state.
            PermissionError: If auth_context is provided but lacks required permissions.
        """
        # Authorization check: if auth_context provided, verify permission
        if auth_context is not None:
            if not auth_context.has_permission(PLAN_EXECUTE_PERMISSION):
                logger.warning(
                    "Plan execution denied: user %s lacks permission %s for plan %s",
                    auth_context.user_id,
                    PLAN_EXECUTE_PERMISSION,
                    plan.id,
                )
                raise PermissionError(
                    f"User {auth_context.user_id} lacks permission '{PLAN_EXECUTE_PERMISSION}' "
                    f"to execute plan {plan.id}"
                )
            logger.info(
                "Plan execution authorized: user %s executing plan %s",
                auth_context.user_id,
                plan.id,
            )

        if plan.status == PlanStatus.REJECTED:
            raise ValueError(f"Plan {plan.id} was rejected and cannot be executed")
        if plan.status == PlanStatus.EXECUTING:
            raise ValueError(f"Plan {plan.id} is already executing")
        if plan.status in (PlanStatus.COMPLETED, PlanStatus.FAILED):
            raise ValueError(f"Plan {plan.id} has already been executed ({plan.status.value})")

        if plan.requires_human_approval and not plan.is_approved:
            raise ValueError(f"Plan {plan.id} requires approval before execution")

        # Capture auth context into plan metadata for downstream RBAC/memory scoping
        if auth_context is not None:
            if not isinstance(plan.metadata, dict):
                plan.metadata = {}
            plan.metadata.setdefault("owner_id", getattr(auth_context, "user_id", None))
            plan.metadata.setdefault("workspace_id", getattr(auth_context, "workspace_id", None))
            plan.metadata.setdefault("org_id", getattr(auth_context, "org_id", None))
            plan.metadata.setdefault("requested_by", getattr(auth_context, "user_id", None))

        # Transition to executing
        plan.status = PlanStatus.EXECUTING
        plan.execution_started_at = datetime.now()
        store_plan(plan)

        # Emit PLAN_EXECUTING event
        _exec_mode = execution_mode or self._execution_mode
        self._emit_plan_event("PLAN_EXECUTING", {
            "plan_id": plan.id,
            "debate_id": plan.debate_id,
            "workspace_id": (plan.metadata or {}).get("workspace_id", ""),
            "execution_mode": _exec_mode if isinstance(_exec_mode, str) else str(_exec_mode),
            "task": plan.task[:200],
        })

        start_time = time.time()
        outcome: PlanOutcome

        profile = plan.implementation_profile
        if parallel_execution is None:
            if profile and profile.parallel_execution is not None:
                parallel_execution = profile.parallel_execution
            else:
                parallel_execution = self._parallel_execution

        max_parallel = self._max_parallel
        if profile and profile.max_parallel is not None:
            max_parallel = profile.max_parallel

        # Determine execution mode
        mode = (
            execution_mode or (profile.execution_mode if profile else None) or self._execution_mode
        )
        if mode in {"hybrid", "fabric"} and self._requires_workflow(plan):
            logger.info(
                "Plan %s contains computer-use/manual tasks; switching execution mode to workflow",
                plan.id,
            )
            mode = "workflow"

        if not isinstance(plan.metadata, dict):
            plan.metadata = {}
        plan.metadata["last_execution_mode"] = mode
        if execution_mode:
            plan.metadata["requested_execution_mode"] = execution_mode

        notifier = None
        if on_task_complete is None:
            meta = plan.metadata if isinstance(plan.metadata, dict) else {}
            channel_targets = None
            thread_id = None
            thread_id_by_platform = None
            notify_origin = bool(meta.get("notify_origin", False))

            if profile:
                channel_targets = profile.channel_targets or channel_targets
                thread_id = profile.thread_id or thread_id
                thread_id_by_platform = profile.thread_id_by_platform or thread_id_by_platform

            channel_targets = (
                channel_targets or meta.get("channel_targets") or meta.get("chat_targets")
            )
            if isinstance(channel_targets, str):
                ct_str: str = channel_targets
                channel_targets = [
                    item.strip() for item in ct_str.split(",") if item.strip()
                ]
            thread_id = thread_id or meta.get("thread_id") or meta.get("origin_thread_id")
            if isinstance(meta.get("thread_id_by_platform"), dict):
                thread_id_by_platform = thread_id_by_platform or meta.get("thread_id_by_platform")

            if channel_targets or thread_id or thread_id_by_platform or notify_origin:
                try:
                    from aragora.pipeline.execution_notifier import ExecutionNotifier

                    notifier = ExecutionNotifier(
                        debate_id=plan.debate_id or plan.id,
                        plan_id=plan.id,
                        notify_channel=notify_origin,
                        notify_websocket=notify_origin,
                        channel_targets=channel_targets
                        if isinstance(channel_targets, list)
                        else None,
                        thread_id=thread_id if isinstance(thread_id, str) else None,
                        thread_id_by_platform=thread_id_by_platform
                        if isinstance(thread_id_by_platform, dict)
                        else None,
                    )
                    if plan.implement_plan is not None:
                        notifier.set_task_descriptions(plan.implement_plan.tasks)
                    on_task_complete = notifier.on_task_complete
                except Exception as exc:
                    logger.debug("Failed to initialize execution notifier: %s", exc)

        try:
            if mode == "hybrid":
                outcome = await self._run_hybrid(
                    plan,
                    parallel_execution=parallel_execution,
                    max_parallel=max_parallel,
                    on_task_complete=on_task_complete,
                )
            elif mode == "fabric":
                outcome = await self._run_fabric(
                    plan,
                    on_task_complete=on_task_complete,
                    max_parallel=max_parallel,
                )
            elif mode == "computer_use":
                outcome = await self._run_computer_use(plan)
            else:
                outcome = await self._run_workflow(
                    plan,
                    parallel_execution=parallel_execution,
                    max_parallel=max_parallel,
                )
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

        # Emit completion/failure event
        event_type = "PLAN_COMPLETED" if outcome.success else "PLAN_FAILED"
        event_data: dict[str, Any] = {
            "plan_id": plan.id,
            "debate_id": plan.debate_id,
            "workspace_id": (plan.metadata or {}).get("workspace_id", ""),
            "execution_mode": mode if isinstance(mode, str) else str(mode),
            "success": outcome.success,
            "duration_seconds": outcome.duration_seconds,
            "tasks_completed": outcome.tasks_completed,
            "tasks_total": outcome.tasks_total,
        }
        if outcome.error:
            event_data["error"] = outcome.error[:500]
        if outcome.lessons:
            event_data["lessons"] = outcome.lessons[:5]
        self._emit_plan_event(event_type, event_data)

        if notifier is not None:
            try:
                await notifier.send_completion_summary()
            except Exception as exc:
                logger.debug("Failed to send execution summary: %s", exc)

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

        # Generate receipt (best-effort)
        try:
            receipt = await self._generate_receipt(plan, outcome)
            if receipt:
                outcome.receipt_id = receipt.receipt_id
                logger.info("Generated receipt %s for plan %s", receipt.receipt_id, plan.id)
        except Exception as e:
            logger.debug("Receipt generation failed (non-critical): %s", e)

        # Ingest to Knowledge Mound via adapter (best-effort)
        try:
            await self._ingest_to_km(plan, outcome)
        except Exception as e:
            logger.debug("KM ingestion failed (non-critical): %s", e)

        # Update store with final state
        store_plan(plan)

        return outcome

    async def _run_workflow(
        self,
        plan: DecisionPlan,
        *,
        parallel_execution: bool = False,
        max_parallel: int | None = None,
    ) -> PlanOutcome:
        """Run the workflow engine against the plan's generated definition."""
        engine: Any  # Union of WorkflowEngine and EnhancedWorkflowEngine
        if parallel_execution:
            from aragora.workflow.engine_v2 import EnhancedWorkflowEngine, ResourceLimits

            limits = ResourceLimits(
                max_parallel_agents=max_parallel
                if max_parallel is not None
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
        inputs = {
            "plan_id": plan.id,
            "debate_id": plan.debate_id,
            "task": plan.task,
        }
        if isinstance(plan.metadata, dict):
            owner_id = plan.metadata.get("owner_id") or plan.metadata.get("user_id")
            workspace_id = plan.metadata.get("workspace_id") or plan.metadata.get("tenant_id")
            if owner_id:
                inputs["user_id"] = owner_id
            if workspace_id:
                inputs["workspace_id"] = workspace_id

        execution_metadata: dict[str, Any] = {}
        if isinstance(plan.metadata, dict):
            execution_metadata.update(plan.metadata)
        execution_metadata.setdefault("plan_id", plan.id)
        execution_metadata.setdefault("debate_id", plan.debate_id)
        execution_metadata.setdefault("execution_mode", "workflow")
        execution_metadata.setdefault("workflow_name", definition.name)

        result = await engine.execute(
            definition,
            inputs=inputs,
            workflow_id=plan.workflow_id,
            metadata=execution_metadata,
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

    async def _run_hybrid(
        self,
        plan: DecisionPlan,
        *,
        parallel_execution: bool = False,
        max_parallel: int | None = None,
        on_task_complete: Any | None = None,
    ) -> PlanOutcome:
        """Run plan tasks using HybridExecutor (Claude + Codex).

        HybridExecutor uses Claude for all implementation tasks with Codex
        as fallback on timeout. This mode is typically faster than the
        workflow engine for code implementation tasks.

        Args:
            plan: DecisionPlan with implement_plan containing tasks
            parallel_execution: Whether to run independent tasks in parallel

        Returns:
            PlanOutcome with execution results
        """
        from aragora.implement.executor import HybridExecutor
        from aragora.implement.types import ImplementTask  # noqa: F401

        if not plan.implement_plan or not plan.implement_plan.tasks:
            return PlanOutcome(
                plan_id=plan.id,
                debate_id=plan.debate_id,
                task=plan.task,
                success=False,
                error="No implementation tasks in plan",
                tasks_total=0,
            )

        # Create HybridExecutor (configurable via plan metadata/profile)
        impl_meta: dict[str, Any] = {}
        if isinstance(plan.metadata, dict):
            impl_meta = plan.metadata.get("implementation") or {}
        if not isinstance(impl_meta, dict):
            impl_meta = {}

        profile = plan.implementation_profile

        implementers = (
            profile.implementers
            if profile and profile.implementers is not None
            else impl_meta.get("implementers")
        )
        if isinstance(implementers, str):
            implementers = [item.strip() for item in implementers.split(",") if item.strip()]

        complexity_router = (
            profile.complexity_router
            if profile and profile.complexity_router is not None
            else impl_meta.get("complexity_router") or impl_meta.get("agent_by_complexity")
        )
        task_type_router = (
            profile.task_type_router
            if profile and profile.task_type_router is not None
            else impl_meta.get("task_type_router") or impl_meta.get("agent_by_task_type")
        )
        capability_router = (
            profile.capability_router
            if profile and profile.capability_router is not None
            else impl_meta.get("capability_router") or impl_meta.get("agent_by_capability")
        )

        executor = HybridExecutor(
            repo_path=self._repo_path,
            max_retries=2,
            strategy=profile.strategy
            if profile and profile.strategy is not None
            else impl_meta.get("strategy"),
            implementers=implementers,
            critic=profile.critic
            if profile and profile.critic is not None
            else impl_meta.get("critic"),
            reviser=profile.reviser
            if profile and profile.reviser is not None
            else impl_meta.get("reviser"),
            max_revisions=profile.max_revisions
            if profile and profile.max_revisions is not None
            else impl_meta.get("max_revisions"),
            complexity_router=complexity_router,
            task_type_router=task_type_router,
            capability_router=capability_router,
        )

        # Extract tasks from plan
        tasks = plan.implement_plan.tasks
        tasks_total = len(tasks)

        # Set parallel execution mode via environment if needed
        import os as _os

        if parallel_execution:
            _os.environ["IMPL_PARALLEL_TASKS"] = "1"
            if max_parallel:
                _os.environ["IMPL_MAX_PARALLEL"] = str(max_parallel)

        start_time = time.time()

        try:
            # Execute all tasks
            completed: set[str] = set()
            if parallel_execution:
                results = await executor.execute_plan_parallel(
                    tasks=tasks,
                    completed=completed,
                    max_parallel=max_parallel,
                    on_task_complete=on_task_complete,
                )
            else:
                results = await executor.execute_plan(
                    tasks=tasks,
                    completed=completed,
                    on_task_complete=on_task_complete,
                    stop_on_failure=False,  # Continue on failures, retry at end
                )

            duration = time.time() - start_time

            # Compute stats
            tasks_completed = sum(1 for r in results if r.success)
            total_cost = sum(getattr(r, "cost_usd", 0.0) or 0.0 for r in results)

            # Collect errors
            errors = [r.error for r in results if r.error]
            error_msg = "; ".join(errors) if errors else None

            # Derive lessons
            lessons: list[str] = []
            if tasks_completed < tasks_total:
                lessons.append(
                    f"Only {tasks_completed}/{tasks_total} tasks completed via HybridExecutor"
                )

            # Check for model fallbacks
            fallback_count = sum(1 for r in results if r.model_used and "fallback" in r.model_used)
            if fallback_count > 0:
                lessons.append(f"{fallback_count} task(s) required Codex fallback")

            success = tasks_completed == tasks_total and not error_msg

            # Update plan status
            plan.status = PlanStatus.COMPLETED if success else PlanStatus.FAILED
            plan.execution_completed_at = datetime.now()

            return PlanOutcome(
                plan_id=plan.id,
                debate_id=plan.debate_id,
                task=plan.task,
                success=success,
                tasks_completed=tasks_completed,
                tasks_total=tasks_total,
                total_cost_usd=total_cost + plan.budget.spent_usd,
                error=error_msg,
                duration_seconds=duration,
                lessons=lessons,
            )

        finally:
            # Clean up environment
            if parallel_execution:
                _os.environ.pop("IMPL_PARALLEL_TASKS", None)
                _os.environ.pop("IMPL_MAX_PARALLEL", None)

    async def _run_fabric(
        self,
        plan: DecisionPlan,
        *,
        on_task_complete: Any | None = None,
        max_parallel: int | None = None,
    ) -> PlanOutcome:
        """Run plan tasks using AgentFabric for multi-agent execution."""
        from aragora.fabric import AgentFabric
        from aragora.implement.fabric_integration import (
            FabricImplementationConfig,
            FabricImplementationRunner,
        )
        from aragora.pipeline.decision_plan.core import ImplementationProfile

        if not plan.implement_plan or not plan.implement_plan.tasks:
            return PlanOutcome(
                plan_id=plan.id,
                debate_id=plan.debate_id,
                task=plan.task,
                success=False,
                error="No implementation tasks in plan",
                tasks_total=0,
            )

        profile = plan.implementation_profile
        if profile is None and isinstance(plan.metadata, dict):
            impl_payload = plan.metadata.get("implementation_profile") or plan.metadata.get(
                "implementation"
            )
            if isinstance(impl_payload, dict):
                profile = ImplementationProfile.from_dict(impl_payload)

        models: list[str] = []
        if profile:
            if profile.fabric_models:
                models = list(profile.fabric_models)
            elif profile.implementers:
                models = list(profile.implementers)
        if not models:
            models = ["claude"]

        meta = plan.metadata if isinstance(plan.metadata, dict) else {}
        config = FabricImplementationConfig(
            pool_id=profile.fabric_pool_id if profile else None,
            models=models,
            min_agents=profile.fabric_min_agents if profile and profile.fabric_min_agents else 1,
            max_agents=profile.fabric_max_agents
            if profile and profile.fabric_max_agents is not None
            else max_parallel,
            timeout_seconds=profile.fabric_timeout_seconds
            if profile and profile.fabric_timeout_seconds
            else 1800.0,
            org_id=meta.get("org_id", "") or "",
            user_id=meta.get("owner_id", "") or meta.get("user_id", "") or "",
            workspace_id=meta.get("workspace_id", "") or meta.get("tenant_id", "") or "",
        )

        start_time = time.time()

        async with AgentFabric() as fabric:
            runner = FabricImplementationRunner(
                fabric,
                repo_path=self._repo_path,
                implementation_profile=profile,
            )
            results = await runner.run_plan(
                plan.implement_plan.tasks,
                config=config,
                on_task_complete=on_task_complete,
            )

        duration = time.time() - start_time
        tasks_total = len(plan.implement_plan.tasks)
        tasks_completed = sum(1 for r in results if r.success)
        errors = [r.error for r in results if r.error]
        error_msg = "; ".join(errors) if errors else None
        total_cost = sum(getattr(r, "cost_usd", 0.0) or 0.0 for r in results)

        lessons: list[str] = []
        if tasks_completed < tasks_total:
            lessons.append(
                f"Only {tasks_completed}/{tasks_total} tasks completed via fabric execution"
            )

        success = tasks_completed == tasks_total and not error_msg
        plan.status = PlanStatus.COMPLETED if success else PlanStatus.FAILED
        plan.execution_completed_at = datetime.now()

        return PlanOutcome(
            plan_id=plan.id,
            debate_id=plan.debate_id,
            task=plan.task,
            success=success,
            tasks_completed=tasks_completed,
            tasks_total=tasks_total,
            total_cost_usd=plan.budget.spent_usd + total_cost,
            error=error_msg,
            duration_seconds=duration,
            lessons=lessons,
        )

    async def _run_computer_use(self, plan: DecisionPlan) -> PlanOutcome:
        """Run plan using ComputerUseOrchestrator for browser-based implementation.

        Computer Use mode executes tasks through browser automation via Playwright,
        guided by Claude's computer_use tool. This is suitable for tasks that
        require interacting with web UIs or desktop applications.

        Args:
            plan: DecisionPlan with task description

        Returns:
            PlanOutcome with execution results
        """
        from aragora.computer_use.executor import ExecutorConfig, PlaywrightActionExecutor
        from aragora.computer_use.orchestrator import (
            ComputerUseConfig,
            ComputerUseOrchestrator,
            create_default_computer_policy,
        )

        start_time = time.time()

        # Use task description as the goal for computer use
        goal = plan.task
        if plan.implement_plan and plan.implement_plan.tasks:
            # Combine task descriptions for more context
            task_descriptions = [t.description for t in plan.implement_plan.tasks]
            goal = f"{plan.task}\n\nTasks:\n" + "\n".join(f"- {d}" for d in task_descriptions)

        # Configure computer use
        config = ComputerUseConfig(
            max_steps=50,  # Reasonable limit for automated tasks
            screenshot_delay_ms=500,
        )
        policy = create_default_computer_policy()

        tasks_total = len(plan.implement_plan.tasks) if plan.implement_plan else 1
        lessons: list[str] = []

        try:
            # Create executor and orchestrator
            executor_config = ExecutorConfig(
                headless=True,
                viewport_width=1920,
                viewport_height=1080,
            )

            async with PlaywrightActionExecutor(executor_config) as executor:
                orchestrator = ComputerUseOrchestrator(
                    executor=executor,
                    policy=policy,
                    config=config,
                )

                # Run the computer use task
                result = await orchestrator.run_task(
                    goal=goal,
                    max_steps=config.max_steps,
                    initial_context=f"Debate ID: {plan.debate_id}\nPlan ID: {plan.id}",
                    metadata={"debate_id": plan.debate_id, "plan_id": plan.id},
                )

            duration = time.time() - start_time

            # Extract metrics from result
            success = result.success if hasattr(result, "success") else False
            error = result.error if hasattr(result, "error") else None
            steps_completed = result.steps_completed if hasattr(result, "steps_completed") else 0

            if steps_completed > 0:
                lessons.append(f"Completed {steps_completed} browser automation steps")

            if hasattr(result, "actions") and result.actions:
                action_types = set(
                    a.action_type for a in result.actions if hasattr(a, "action_type")
                )
                lessons.append(f"Used actions: {', '.join(str(t) for t in action_types)}")

            # Update plan status
            plan.status = PlanStatus.COMPLETED if success else PlanStatus.FAILED
            plan.execution_completed_at = datetime.now()

            return PlanOutcome(
                plan_id=plan.id,
                debate_id=plan.debate_id,
                task=plan.task,
                success=success,
                tasks_completed=tasks_total if success else 0,
                tasks_total=tasks_total,
                total_cost_usd=plan.budget.spent_usd,  # Computer use cost tracking TBD
                error=error,
                duration_seconds=duration,
                lessons=lessons,
            )

        except ImportError as e:
            duration = time.time() - start_time
            logger.error("Computer use dependencies not available: %s", e)
            return PlanOutcome(
                plan_id=plan.id,
                debate_id=plan.debate_id,
                task=plan.task,
                success=False,
                error=(
                    f"Computer use dependencies not available: {e}. "
                    "Install with: pip install playwright"
                ),
                duration_seconds=duration,
                tasks_total=tasks_total,
            )

        except Exception as e:
            duration = time.time() - start_time
            logger.error("Computer use execution failed: %s", e)
            plan.status = PlanStatus.FAILED
            plan.execution_completed_at = datetime.now()
            return PlanOutcome(
                plan_id=plan.id,
                debate_id=plan.debate_id,
                task=plan.task,
                success=False,
                error=str(e),
                duration_seconds=duration,
                tasks_total=tasks_total,
                lessons=[f"Computer use failed: {e}"],
            )

    async def _generate_receipt(
        self,
        plan: DecisionPlan,
        outcome: PlanOutcome,
    ) -> Any | None:
        """Generate a cryptographic receipt for the plan execution.

        Args:
            plan: The executed DecisionPlan
            outcome: The execution outcome

        Returns:
            DecisionReceipt if successful, None otherwise
        """
        try:
            from aragora.gauntlet.receipt import DecisionReceipt

            receipt = DecisionReceipt.from_plan_outcome(outcome, plan=plan)

            # Attempt to sign the receipt
            try:
                receipt.sign()
                logger.debug("Signed receipt %s", receipt.receipt_id)
            except Exception as sign_err:
                logger.debug("Receipt signing skipped: %s", sign_err)

            # Store receipt if we have a receipt store
            if self._knowledge_mound is not None:
                try:
                    # Use KM receipt adapter if available
                    from aragora.knowledge.mound.adapters.receipt import get_receipt_adapter

                    adapter = get_receipt_adapter(self._knowledge_mound)
                    if adapter:
                        await adapter.store_receipt(receipt)
                except (ImportError, AttributeError) as e:
                    logger.debug("Receipt adapter not available: %s", e)

            return receipt
        except ImportError:
            logger.debug("Receipt generation not available (gauntlet not installed)")
            return None

    async def _ingest_to_km(
        self,
        plan: DecisionPlan,
        outcome: PlanOutcome,
    ) -> None:
        """Ingest plan outcome to Knowledge Mound via adapter.

        Args:
            plan: The executed DecisionPlan
            outcome: The execution outcome
        """
        try:
            from aragora.knowledge.mound.adapters.decision_plan_adapter import (
                get_decision_plan_adapter,
            )

            adapter = get_decision_plan_adapter(self._knowledge_mound)
            result = await adapter.ingest_plan_outcome(plan, outcome)

            if result.success:
                logger.debug(
                    "Ingested plan %s to KM: %d items, %d lessons",
                    plan.id,
                    result.items_ingested,
                    result.lessons_ingested,
                )
            elif result.errors:
                logger.debug("KM ingestion had errors: %s", result.errors)
        except ImportError:
            logger.debug("DecisionPlanAdapter not available")
