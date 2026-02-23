"""
TaskQueue Executor Adapter - Adapts TaskQueue to the WorkflowExecutor protocol.

Provides a bridge between the queue-based task execution system and the
unified WorkflowExecutor interface, enabling TaskQueue to be used
interchangeably with WorkflowEngine and EnhancedWorkflowEngine.

Usage:
    from aragora.workflow import get_workflow_executor

    # Get queue-based executor
    executor = get_workflow_executor(mode="queue")

    # Use like any other executor
    result = await executor.execute(definition, inputs)
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any

from aragora.workflow.types import (
    StepResult,
    StepStatus,
    WorkflowDefinition,
    WorkflowResult,
)

logger = logging.getLogger(__name__)


class TaskQueueExecutorAdapter:
    """
    Adapter that wraps TaskQueue to implement the WorkflowExecutor protocol.

    Converts WorkflowDefinition steps into WorkflowTask instances and
    executes them via the TaskQueue's priority and dependency system.

    This allows the queue-based execution model to be used through the
    same interface as WorkflowEngine and EnhancedWorkflowEngine.
    """

    def __init__(
        self,
        max_concurrent: int = 10,
        default_timeout: float = 300.0,
        enable_retries: bool = True,
        max_retries: int = 2,
    ):
        """
        Initialize the adapter.

        Args:
            max_concurrent: Maximum concurrent task execution
            default_timeout: Default timeout for tasks in seconds
            enable_retries: Whether to enable automatic retries
            max_retries: Maximum retry attempts per task
        """
        self._max_concurrent = max_concurrent
        self._default_timeout = default_timeout
        self._enable_retries = enable_retries
        self._max_retries = max_retries

        # Execution state
        self._results: list[StepResult] = []
        self._should_terminate: bool = False
        self._termination_reason: str | None = None

        # Queue reference (lazy-loaded to avoid circular imports)
        self._queue: Any | None = None
        self._queue_started: bool = False

    async def _ensure_queue(self) -> Any:
        """Ensure TaskQueue is initialized and started."""
        if self._queue is None:
            from aragora.workflow.queue import TaskQueue, TaskQueueConfig

            config = TaskQueueConfig(
                max_concurrent=self._max_concurrent,
                default_timeout=self._default_timeout,
                enable_retries=self._enable_retries,
                max_retries=self._max_retries,
            )
            self._queue = TaskQueue(config=config)

        if not self._queue_started:
            await self._queue.start()
            self._queue_started = True

        return self._queue

    async def execute(
        self,
        definition: WorkflowDefinition,
        inputs: dict[str, Any] | None = None,
        workflow_id: str | None = None,
    ) -> WorkflowResult:
        """
        Execute a workflow using the TaskQueue.

        Converts the workflow definition into queue tasks, enqueues them
        with appropriate dependencies, and waits for completion.

        Args:
            definition: Workflow definition to execute
            inputs: Input parameters for the workflow
            workflow_id: Optional ID (generated if not provided)

        Returns:
            WorkflowResult with step results and final output
        """
        workflow_id = workflow_id or f"wf_{uuid.uuid4().hex[:12]}"
        inputs = inputs or {}

        logger.info(
            "queue_adapter_execute_started",
            extra={
                "workflow_id": workflow_id,
                "workflow_name": definition.name,
                "step_count": len(definition.steps),
            },
        )

        # Reset execution state
        self._results = []
        self._should_terminate = False
        self._termination_reason = None

        start_time = time.time()

        try:
            queue = await self._ensure_queue()

            # Convert steps to tasks and build dependency map
            tasks = await self._convert_steps_to_tasks(definition, workflow_id, inputs)

            if not tasks:
                return WorkflowResult(
                    workflow_id=workflow_id,
                    definition_id=definition.id,
                    success=True,
                    steps=[],
                    total_duration_ms=0,
                    final_output=None,
                )

            # Enqueue all tasks

            for task in tasks:
                await queue.enqueue(task)

            # Wait for workflow completion
            timeout = definition.metadata.get("timeout_seconds", self._default_timeout)
            task_results = await queue.wait_for_workflow(workflow_id, timeout=timeout)

            # Convert task results to step results
            final_output = None
            success = True

            for step in definition.steps:
                task_id = f"{workflow_id}:{step.id}"
                task_result = task_results.get(task_id)

                if task_result:
                    step_result = StepResult(
                        step_id=step.id,
                        step_name=step.name,
                        status=(StepStatus.COMPLETED if task_result.success else StepStatus.FAILED),
                        output=task_result.output,
                        error=task_result.error,
                        duration_ms=task_result.execution_time_ms,
                    )
                    if task_result.output:
                        final_output = task_result.output
                    if not task_result.success:
                        success = False
                else:
                    step_result = StepResult(
                        step_id=step.id,
                        step_name=step.name,
                        status=StepStatus.SKIPPED,
                        error="Task not executed",
                    )
                    success = False

                self._results.append(step_result)

            total_duration_ms = (time.time() - start_time) * 1000

            logger.info(
                "queue_adapter_execute_completed",
                extra={
                    "workflow_id": workflow_id,
                    "success": success,
                    "duration_ms": total_duration_ms,
                    "steps_executed": len(self._results),
                },
            )

            return WorkflowResult(
                workflow_id=workflow_id,
                definition_id=definition.id,
                success=success,
                steps=self._results.copy(),
                total_duration_ms=total_duration_ms,
                final_output=final_output,
            )

        except asyncio.TimeoutError:
            total_duration_ms = (time.time() - start_time) * 1000
            logger.error(
                "queue_adapter_execute_timeout",
                extra={"workflow_id": workflow_id, "duration_ms": total_duration_ms},
            )
            return WorkflowResult(
                workflow_id=workflow_id,
                definition_id=definition.id,
                success=False,
                steps=self._results.copy(),
                total_duration_ms=total_duration_ms,
                error=f"Workflow timed out after {self._default_timeout}s",
            )

        except (
            RuntimeError,
            ValueError,
            TypeError,
            OSError,
            ConnectionError,
            KeyError,
            AttributeError,
            ImportError,
        ) as e:
            total_duration_ms = (time.time() - start_time) * 1000
            logger.exception(
                "queue_adapter_execute_failed",
                extra={"workflow_id": workflow_id, "error": str(e)},
            )
            return WorkflowResult(
                workflow_id=workflow_id,
                definition_id=definition.id,
                success=False,
                steps=self._results.copy(),
                total_duration_ms=total_duration_ms,
                error="Workflow execution failed",
            )

    async def _convert_steps_to_tasks(
        self,
        definition: WorkflowDefinition,
        workflow_id: str,
        inputs: dict[str, Any],
    ) -> list[Any]:
        """
        Convert workflow steps to queue tasks.

        Builds the dependency graph from step next_steps references.
        """
        from aragora.workflow.queue import WorkflowTask

        tasks: list[WorkflowTask] = []
        step_to_task_id: dict[str, str] = {}

        # First pass: create task IDs
        for step in definition.steps:
            task_id = f"{workflow_id}:{step.id}"
            step_to_task_id[step.id] = task_id

        # Second pass: create tasks with dependencies
        for step in definition.steps:
            task_id = step_to_task_id[step.id]

            # Find dependencies (steps that have this step in next_steps)
            depends_on: list[str] = []
            for other_step in definition.steps:
                if step.id in (other_step.next_steps or []):
                    depends_on.append(step_to_task_id[other_step.id])

            # Also check for explicit dependencies in step config
            explicit_deps = step.config.get("depends_on", [])
            for dep_id in explicit_deps:
                if dep_id in step_to_task_id:
                    depends_on.append(step_to_task_id[dep_id])

            task = WorkflowTask.create(
                workflow_id=workflow_id,
                step_id=step.id,
                task_type=step.step_type,
                config={
                    **step.config,
                    "inputs": inputs,
                    "step_name": step.name,
                },
                depends_on=depends_on if depends_on else None,
                priority=step.config.get("priority", 50),
                timeout_seconds=step.timeout_seconds,
                max_retries=step.retries,
            )
            tasks.append(task)

        return tasks

    def get_metrics(self) -> dict[str, Any]:
        """Get execution metrics."""
        total_duration = sum(r.duration_ms or 0 for r in self._results)
        completed = sum(1 for r in self._results if r.status == StepStatus.COMPLETED)
        failed = sum(1 for r in self._results if r.status == StepStatus.FAILED)
        skipped = sum(1 for r in self._results if r.status == StepStatus.SKIPPED)

        queue_stats = {}
        if self._queue:
            stats = self._queue.get_stats()
            queue_stats = {
                "pending_count": stats.pending_count,
                "ready_count": stats.ready_count,
                "running_count": stats.running_count,
                "completed_count": stats.completed_count,
                "failed_count": stats.failed_count,
                "avg_wait_time_ms": stats.avg_wait_time_ms,
                "avg_execution_time_ms": stats.avg_execution_time_ms,
            }

        return {
            "total_steps": len(self._results),
            "completed_steps": completed,
            "failed_steps": failed,
            "skipped_steps": skipped,
            "total_duration_ms": total_duration,
            "terminated_early": self._should_terminate,
            "termination_reason": self._termination_reason,
            "queue_stats": queue_stats,
        }

    def request_termination(self, reason: str = "Requested") -> None:
        """Request early termination of workflow execution."""
        self._should_terminate = True
        self._termination_reason = reason

        # Cancel pending tasks in the queue
        if self._queue and hasattr(self._queue, "_workflows"):
            for workflow_id in list(self._queue._workflows.keys()):
                self._queue.cancel_workflow(workflow_id)

        logger.info("Queue adapter termination requested: %s", reason)

    async def stop(self) -> None:
        """Stop the queue (cleanup)."""
        if self._queue and self._queue_started:
            await self._queue.stop(drain=True)
            self._queue_started = False


# Singleton instance for convenience
_queue_adapter_instance: TaskQueueExecutorAdapter | None = None


def get_queue_adapter(
    max_concurrent: int = 10,
    default_timeout: float = 300.0,
) -> TaskQueueExecutorAdapter:
    """
    Get or create the global TaskQueueExecutorAdapter singleton.

    Args:
        max_concurrent: Maximum concurrent task execution
        default_timeout: Default timeout for tasks in seconds

    Returns:
        TaskQueueExecutorAdapter instance
    """
    global _queue_adapter_instance

    if _queue_adapter_instance is None:
        _queue_adapter_instance = TaskQueueExecutorAdapter(
            max_concurrent=max_concurrent,
            default_timeout=default_timeout,
        )

    return _queue_adapter_instance


def reset_queue_adapter() -> None:
    """Reset the global TaskQueueExecutorAdapter singleton (for testing)."""
    global _queue_adapter_instance
    _queue_adapter_instance = None


__all__ = [
    "TaskQueueExecutorAdapter",
    "get_queue_adapter",
    "reset_queue_adapter",
]
