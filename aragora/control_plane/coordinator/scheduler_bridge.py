"""
Scheduler Bridge for Control Plane Coordinator.

Handles task scheduling integration, task lifecycle management,
and coordination with the TaskScheduler.
"""

from __future__ import annotations

import asyncio
import time
from typing import (
    TYPE_CHECKING,
    Any,
)

from aragora.control_plane.scheduler import Task, TaskPriority, TaskScheduler, TaskStatus

from aragora.observability import (
    get_logger,
    create_span,
    add_span_attributes,
)

from aragora.resilience.retry import (
    PROVIDER_RETRY_POLICIES,
    with_retry,
)

if TYPE_CHECKING:
    from aragora.control_plane.coordinator.state_manager import StateManager, ControlPlaneConfig
    from aragora.control_plane.coordinator.policy_enforcer import PolicyEnforcer
    from aragora.control_plane.policy import ControlPlanePolicyManager

# Optional KM integration
HAS_KM_ADAPTER = False
try:
    from aragora.knowledge.mound.adapters import control_plane_adapter as _km_adapter

    HAS_KM_ADAPTER = True
except ImportError:
    _km_adapter = None

# Optional Policy
HAS_POLICY = False
EnforcementLevelType: Any = None
try:
    from aragora.control_plane.policy import EnforcementLevel as EnforcementLevelType

    HAS_POLICY = True
except ImportError:
    pass

logger = get_logger(__name__)

# Retry configuration for control plane operations
_CP_RETRY_CONFIG = PROVIDER_RETRY_POLICIES["control_plane"]


class SchedulerBridge:
    """
    Bridge between the Coordinator and TaskScheduler.

    Handles task submission, claiming, completion, failure,
    and result waiting with proper coordination.
    """

    def __init__(
        self,
        config: ControlPlaneConfig,
        state_manager: StateManager,
        policy_enforcer: PolicyEnforcer,
        scheduler: TaskScheduler | None = None,
    ):
        """
        Initialize the scheduler bridge.

        Args:
            config: Control plane configuration
            state_manager: State manager for agent operations
            policy_enforcer: Policy enforcer for policy checks
            scheduler: Optional pre-configured TaskScheduler
        """
        self._config = config
        self._state_manager = state_manager
        self._policy_enforcer = policy_enforcer

        self._scheduler = scheduler or TaskScheduler(
            redis_url=self._config.redis_url,
            key_prefix=f"{self._config.key_prefix}tasks:",
            stream_prefix=f"{self._config.key_prefix}stream:",
            policy_manager=self._policy_enforcer.policy_manager,
        )

        self._result_waiters: dict[str, asyncio.Event] = {}

    @property
    def scheduler(self) -> TaskScheduler:
        """Get the task scheduler."""
        return self._scheduler

    def update_policy_manager(self, manager: ControlPlanePolicyManager) -> None:
        """Update the scheduler's policy manager reference."""
        self._scheduler._policy_manager = manager

    async def connect(self) -> None:
        """Connect to Redis."""
        await self._scheduler.connect()

    async def close(self) -> None:
        """Close the scheduler connection."""
        await self._scheduler.close()

    # =========================================================================
    # Task Operations
    # =========================================================================

    @with_retry(_CP_RETRY_CONFIG)
    async def submit_task(
        self,
        task_type: str,
        payload: dict[str, Any],
        required_capabilities: list[str] | None = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        timeout_seconds: float | None = None,
        metadata: dict[str, Any] | None = None,
        workspace_id: str | None = None,
    ) -> str:
        """
        Submit a task for execution.

        Args:
            task_type: Type of task
            payload: Task data
            required_capabilities: Required agent capabilities
            priority: Task priority
            timeout_seconds: Task timeout (uses config default if not specified)
            metadata: Additional metadata
            workspace_id: Optional workspace ID for policy scoping

        Returns:
            Task ID

        Raises:
            PolicyViolationError: If task violates HARD enforcement policy
        """
        with create_span(
            "scheduler_bridge.submit_task",
            {
                "task_type": task_type,
                "priority": priority.value,
                "required_capabilities": str(required_capabilities or []),
            },
        ) as span:
            start = time.monotonic()

            task_id = await self._scheduler.submit(
                task_type=task_type,
                payload=payload,
                required_capabilities=required_capabilities,
                priority=priority,
                timeout_seconds=timeout_seconds or self._config.task_timeout,
                max_retries=self._config.max_task_retries,
                metadata=metadata,
                workspace_id=workspace_id,
            )

            latency_ms = (time.monotonic() - start) * 1000
            add_span_attributes(span, {"task_id": task_id, "latency_ms": latency_ms})
            logger.info(
                "task_submitted",
                task_id=task_id,
                task_type=task_type,
                priority=priority.value,
                latency_ms=latency_ms,
            )

            # Emit task submitted notification
            try:
                from aragora.control_plane.task_events import emit_task_submitted

                await emit_task_submitted(
                    task_id=task_id,
                    task_type=task_type,
                    priority=priority.name,
                    workspace_id=workspace_id
                    or (metadata.get("workspace_id") if metadata else None),
                    metadata=metadata,
                )
            except (ImportError, AttributeError, RuntimeError) as e:
                logger.debug(f"Notification error on task submission: {e}")

            return task_id

    @with_retry(_CP_RETRY_CONFIG)
    async def claim_task(
        self,
        agent_id: str,
        capabilities: list[str],
        block_ms: int = 5000,
        agent_region: str | None = None,
        workspace_id: str | None = None,
    ) -> Task | None:
        """
        Claim a task for an agent.

        Args:
            agent_id: Agent claiming the task
            capabilities: Agent's capabilities
            block_ms: Time to block waiting
            agent_region: Region where the agent is located (for policy checks)
            workspace_id: Workspace ID for policy scoping

        Returns:
            Task if claimed, None otherwise
        """
        from aragora.control_plane.registry import AgentStatus

        task = await self._scheduler.claim(
            worker_id=agent_id,
            capabilities=capabilities,
            block_ms=block_ms,
            worker_region=agent_region,
            workspace_id=workspace_id,
        )

        if task:
            # Update agent status
            await self._state_manager.heartbeat(
                agent_id,
                status=AgentStatus.BUSY,
                current_task_id=task.id,
            )

            # Emit task claimed notification
            try:
                from aragora.control_plane.task_events import emit_task_claimed

                await emit_task_claimed(
                    task_id=task.id,
                    task_type=task.task_type,
                    agent_id=agent_id,
                    workspace_id=task.metadata.get("workspace_id") if task.metadata else None,
                )
            except (ImportError, AttributeError, RuntimeError) as e:
                logger.debug(f"Notification error on task claim: {e}")

        return task

    @with_retry(_CP_RETRY_CONFIG)
    async def complete_task(
        self,
        task_id: str,
        result: dict[str, Any] | None = None,
        agent_id: str | None = None,
        latency_ms: float | None = None,
        sla_policy_id: str | None = None,
    ) -> bool:
        """
        Mark a task as completed.

        Args:
            task_id: Task to complete
            result: Task result
            agent_id: Agent that completed the task
            latency_ms: Execution time
            sla_policy_id: Optional policy ID to check SLA compliance against

        Returns:
            True if completed, False if not found
        """
        with create_span(
            "scheduler_bridge.complete_task",
            {
                "task_id": task_id,
                "agent_id": agent_id or "unknown",
                "execution_latency_ms": latency_ms or 0.0,
            },
        ) as span:
            # Get task details before completing (for KM storage)
            task = await self._scheduler.get(task_id)
            if task:
                add_span_attributes(span, {"task_type": task.task_type})

            success = await self._scheduler.complete(task_id, result)
            add_span_attributes(span, {"success": success})

            if success and agent_id:
                # Update agent metrics
                await self._state_manager.record_task_completion(
                    agent_id,
                    success=True,
                    latency_ms=latency_ms or 0.0,
                )

                # SLA compliance check (if policy enforcer and policy_id provided)
                if sla_policy_id and task and HAS_POLICY:
                    execution_seconds = (latency_ms or 0.0) / 1000.0
                    queue_seconds = None
                    if task.assigned_at and task.created_at:
                        queue_seconds = task.assigned_at - task.created_at

                    sla_result = self._policy_enforcer.evaluate_sla_compliance(
                        policy_id=sla_policy_id,
                        execution_seconds=execution_seconds,
                        queue_seconds=queue_seconds,
                        task_id=task_id,
                        task_type=task.task_type,
                        agent_id=agent_id,
                        workspace=task.metadata.get("workspace_id") if task.metadata else None,
                    )

                    if not sla_result.allowed:
                        if HAS_POLICY and sla_result.enforcement_level == EnforcementLevelType.WARN:
                            logger.warning(
                                "sla_warning_on_complete",
                                task_id=task_id,
                                agent_id=agent_id,
                                reason=sla_result.reason,
                                policy_id=sla_policy_id,
                                execution_seconds=execution_seconds,
                            )
                        else:
                            logger.error(
                                "sla_violation_on_complete",
                                task_id=task_id,
                                agent_id=agent_id,
                                reason=sla_result.reason,
                                policy_id=sla_policy_id,
                                execution_seconds=execution_seconds,
                            )
                        add_span_attributes(
                            span,
                            {
                                "sla_compliant": False,
                                "sla_violation_reason": sla_result.reason,
                            },
                        )
                    else:
                        add_span_attributes(span, {"sla_compliant": True})

                # Store outcome in Knowledge Mound
                if task and HAS_KM_ADAPTER:
                    await self._state_manager.store_task_outcome(
                        task_id=task_id,
                        task_type=task.task_type,
                        agent_id=agent_id,
                        success=True,
                        duration_seconds=(latency_ms or 0.0) / 1000.0,
                        metadata=task.metadata or {},
                    )

                # Notify waiters
                if task_id in self._result_waiters:
                    self._result_waiters[task_id].set()

                logger.info(
                    "task_completed",
                    task_id=task_id,
                    agent_id=agent_id,
                    task_type=task.task_type if task else "unknown",
                    latency_ms=latency_ms or 0.0,
                )

                # Emit task completed notification
                try:
                    from aragora.control_plane.task_events import emit_task_completed

                    await emit_task_completed(
                        task_id=task_id,
                        task_type=task.task_type if task else "unknown",
                        agent_id=agent_id,
                        duration_seconds=(latency_ms or 0.0) / 1000.0,
                        workspace_id=(
                            task.metadata.get("workspace_id") if task and task.metadata else None
                        ),
                    )
                except (ImportError, AttributeError, RuntimeError) as e:
                    logger.debug(f"Notification error on task completion: {e}")

            return success

    async def fail_task(
        self,
        task_id: str,
        error: str,
        agent_id: str | None = None,
        latency_ms: float | None = None,
        requeue: bool = True,
    ) -> bool:
        """
        Mark a task as failed.

        Args:
            task_id: Task that failed
            error: Error message
            agent_id: Agent that failed
            latency_ms: Execution time
            requeue: Whether to requeue for retry

        Returns:
            True if processed, False if not found
        """
        with create_span(
            "scheduler_bridge.fail_task",
            {
                "task_id": task_id,
                "agent_id": agent_id or "unknown",
                "requeue": requeue,
                "error_message": error[:200],  # Truncate long errors
            },
        ) as span:
            # Get task details before failing (for KM storage)
            task = await self._scheduler.get(task_id)
            if task:
                add_span_attributes(span, {"task_type": task.task_type})

            success = await self._scheduler.fail(task_id, error, requeue)
            add_span_attributes(span, {"success": success})

            if success and agent_id:
                await self._state_manager.record_task_completion(
                    agent_id,
                    success=False,
                    latency_ms=latency_ms or 0.0,
                )

                # Store failure outcome in Knowledge Mound (only if not requeuing)
                if task and not requeue and HAS_KM_ADAPTER:
                    await self._state_manager.store_task_outcome(
                        task_id=task_id,
                        task_type=task.task_type,
                        agent_id=agent_id,
                        success=False,
                        duration_seconds=(latency_ms or 0.0) / 1000.0,
                        error_message=error,
                        metadata=task.metadata or {},
                    )

            # Notify waiters if not requeued
            task = await self._scheduler.get(task_id)
            if task and task.status in (TaskStatus.FAILED, TaskStatus.CANCELLED):
                if task_id in self._result_waiters:
                    self._result_waiters[task_id].set()

            logger.warning(
                "task_failed",
                task_id=task_id,
                agent_id=agent_id,
                task_type=task.task_type if task else "unknown",
                error=error[:200],
                requeued=requeue,
            )

            # Emit task failed notification
            try:
                from aragora.control_plane.task_events import emit_task_failed

                await emit_task_failed(
                    task_id=task_id,
                    task_type=task.task_type if task else "unknown",
                    agent_id=agent_id,
                    error=error,
                    will_retry=requeue,
                    workspace_id=(
                        task.metadata.get("workspace_id") if task and task.metadata else None
                    ),
                )
            except (ImportError, AttributeError, RuntimeError) as e:
                logger.debug(f"Notification error on task failure: {e}")

            return success

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a task."""
        success = await self._scheduler.cancel(task_id)

        if success and task_id in self._result_waiters:
            self._result_waiters[task_id].set()

        return success

    async def get_task(self, task_id: str) -> Task | None:
        """Get task by ID."""
        return await self._scheduler.get(task_id)

    async def wait_for_result(
        self,
        task_id: str,
        timeout: float | None = None,
    ) -> Task | None:
        """
        Wait for a task to complete.

        Args:
            task_id: Task to wait for
            timeout: Maximum wait time in seconds

        Returns:
            Completed task, or None if timeout/not found
        """
        task = await self._scheduler.get(task_id)
        if not task:
            return None

        # Already completed?
        if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
            return task

        # Create waiter
        if task_id not in self._result_waiters:
            self._result_waiters[task_id] = asyncio.Event()

        try:
            await asyncio.wait_for(
                self._result_waiters[task_id].wait(),
                timeout=timeout or self._config.task_timeout,
            )
            return await self._scheduler.get(task_id)
        except asyncio.TimeoutError:
            return None
        finally:
            self._result_waiters.pop(task_id, None)

    async def get_stats(self) -> dict[str, Any]:
        """Get scheduler statistics."""
        return await self._scheduler.get_stats()
