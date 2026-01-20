"""
Task Executor for Workflow Queue.

Provides the TaskExecutor and ExecutorPool classes for executing
workflow tasks with concurrency management and error handling.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, Optional

from aragora.workflow.queue.task import (
    TaskResult,
    WorkflowTask,
)

logger = logging.getLogger(__name__)


class ExecutorStatus(str, Enum):
    """Status of an executor."""

    IDLE = "idle"
    BUSY = "busy"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class ExecutorStats:
    """Statistics for an executor."""

    tasks_completed: int = 0
    tasks_failed: int = 0
    total_execution_time_ms: float = 0
    avg_execution_time_ms: float = 0
    last_task_at: Optional[datetime] = None


class TaskExecutor(ABC):
    """
    Abstract base class for task executors.

    Executors are responsible for actually running workflow tasks.
    Different implementations can handle different step types
    (e.g., debate steps, API calls, data processing).
    """

    def __init__(self, executor_id: str):
        self._id = executor_id
        self._status = ExecutorStatus.IDLE
        self._current_task: Optional[WorkflowTask] = None
        self._stats = ExecutorStats()

    @property
    def id(self) -> str:
        """Get executor ID."""
        return self._id

    @property
    def status(self) -> ExecutorStatus:
        """Get executor status."""
        return self._status

    @property
    def is_available(self) -> bool:
        """Check if executor can accept tasks."""
        return self._status == ExecutorStatus.IDLE

    @property
    def stats(self) -> ExecutorStats:
        """Get executor statistics."""
        return self._stats

    @abstractmethod
    async def execute(self, task: WorkflowTask) -> TaskResult:
        """
        Execute a workflow task.

        Args:
            task: Task to execute

        Returns:
            Task execution result
        """
        pass

    async def _run_with_tracking(self, task: WorkflowTask) -> TaskResult:
        """Run task with status and stats tracking."""
        self._status = ExecutorStatus.BUSY
        self._current_task = task
        start_time = datetime.now()

        try:
            result = await self.execute(task)

            # Update stats
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            self._stats.tasks_completed += 1
            self._stats.total_execution_time_ms += execution_time
            self._stats.avg_execution_time_ms = (
                self._stats.total_execution_time_ms / self._stats.tasks_completed
            )
            self._stats.last_task_at = datetime.now()

            return result

        except Exception as e:
            logger.warning(f"Task execution failed: {e}")
            self._stats.tasks_failed += 1
            raise

        finally:
            self._status = ExecutorStatus.IDLE
            self._current_task = None


class StepExecutor(TaskExecutor):
    """
    Default executor that delegates to step handlers.

    Uses a registry of step handlers to execute different step types.
    """

    def __init__(
        self,
        executor_id: str,
        handlers: Optional[Dict[str, Callable[[WorkflowTask], TaskResult]]] = None,
    ):
        super().__init__(executor_id)
        self._handlers: Dict[str, Callable[[WorkflowTask], TaskResult]] = handlers or {}

    def register_handler(
        self,
        step_type: str,
        handler: Callable[[WorkflowTask], TaskResult],
    ) -> None:
        """Register a handler for a step type."""
        self._handlers[step_type] = handler

    async def execute(self, task: WorkflowTask) -> TaskResult:
        """Execute task using registered handler."""
        step_type = task.step_config.get("type", "default")
        handler = self._handlers.get(step_type)

        if handler:
            start_time = datetime.now()

            # Handle both sync and async handlers
            if asyncio.iscoroutinefunction(handler):
                result = await handler(task)
            else:
                result = handler(task)

            execution_time = (datetime.now() - start_time).total_seconds() * 1000

            # Ensure result has execution time
            if isinstance(result, TaskResult):
                if result.execution_time_ms == 0:
                    result = TaskResult(
                        success=result.success,
                        output=result.output,
                        error=result.error,
                        execution_time_ms=execution_time,
                        retries_used=result.retries_used,
                    )
                return result

            # Wrap raw output
            return TaskResult(
                success=True,
                output=result,
                execution_time_ms=execution_time,
            )

        # No handler - return placeholder
        return TaskResult(
            success=True,
            output={"step_id": task.step_id, "executed": True},
            execution_time_ms=0,
        )


@dataclass
class PoolConfig:
    """Configuration for executor pool."""

    min_executors: int = 2
    max_executors: int = 10
    scale_up_threshold: float = 0.8  # Scale up when utilization > 80%
    scale_down_threshold: float = 0.2  # Scale down when utilization < 20%
    executor_idle_timeout: float = 60.0  # Remove idle executors after 60s


class ExecutorPool:
    """
    Pool of task executors with auto-scaling.

    Manages a pool of executors to handle workflow tasks,
    automatically scaling based on demand.
    """

    def __init__(
        self,
        config: Optional[PoolConfig] = None,
        executor_factory: Optional[Callable[[str], TaskExecutor]] = None,
    ):
        self._config = config or PoolConfig()
        self._executor_factory = executor_factory or self._default_factory

        # Executor management
        self._executors: Dict[str, TaskExecutor] = {}
        self._executor_count = 0

        # Task assignment
        self._task_assignments: Dict[str, str] = {}  # task_id -> executor_id
        self._assignment_lock = asyncio.Lock()

        # Control
        self._started = False
        self._stopping = False
        self._scale_task: Optional[asyncio.Task] = None

    def _default_factory(self, executor_id: str) -> TaskExecutor:
        """Create default step executor."""
        return StepExecutor(executor_id)

    async def start(self) -> None:
        """Start the executor pool."""
        if self._started:
            return

        # Create minimum executors
        for i in range(self._config.min_executors):
            await self._add_executor()

        self._started = True
        self._stopping = False

        # Start auto-scaling monitor
        self._scale_task = asyncio.create_task(self._scale_loop())

        logger.info(f"Executor pool started with {len(self._executors)} executors")

    async def stop(self, drain: bool = True) -> None:
        """Stop the executor pool."""
        if not self._started:
            return

        self._stopping = True

        # Stop scaling
        if self._scale_task:
            self._scale_task.cancel()
            try:
                await self._scale_task
            except asyncio.CancelledError:
                pass

        # Wait for tasks if draining
        if drain:
            while self._task_assignments:
                await asyncio.sleep(0.1)

        self._executors.clear()
        self._started = False

        logger.info("Executor pool stopped")

    async def _add_executor(self) -> TaskExecutor:
        """Add a new executor to the pool."""
        self._executor_count += 1
        executor_id = f"executor_{self._executor_count}"
        executor = self._executor_factory(executor_id)
        self._executors[executor_id] = executor
        logger.debug(f"Added executor {executor_id}")
        return executor

    async def _remove_executor(self, executor_id: str) -> None:
        """Remove an executor from the pool."""
        if executor_id in self._executors:
            executor = self._executors[executor_id]
            if executor.is_available:
                del self._executors[executor_id]
                logger.debug(f"Removed executor {executor_id}")

    async def acquire(self, task: WorkflowTask) -> Optional[TaskExecutor]:
        """
        Acquire an executor for a task.

        Args:
            task: Task to execute

        Returns:
            Available executor or None
        """
        async with self._assignment_lock:
            # Find available executor
            for executor_id, executor in self._executors.items():
                if executor.is_available:
                    self._task_assignments[task.id] = executor_id
                    return executor

            # No available executor - try to scale up
            if len(self._executors) < self._config.max_executors:
                executor = await self._add_executor()
                self._task_assignments[task.id] = executor.id
                return executor

            return None

    async def release(self, task: WorkflowTask) -> None:
        """Release executor after task completion."""
        async with self._assignment_lock:
            self._task_assignments.pop(task.id, None)

    async def execute(self, task: WorkflowTask) -> TaskResult:
        """
        Execute a task using an available executor.

        Args:
            task: Task to execute

        Returns:
            Task execution result
        """
        executor = await self.acquire(task)

        if not executor:
            # Wait for an executor
            for _ in range(100):  # Max 10 second wait
                await asyncio.sleep(0.1)
                executor = await self.acquire(task)
                if executor:
                    break

        if not executor:
            return TaskResult(
                success=False,
                error="No executor available",
            )

        try:
            return await executor._run_with_tracking(task)
        finally:
            await self.release(task)

    async def _scale_loop(self) -> None:
        """Auto-scaling monitoring loop."""
        while not self._stopping:
            try:
                await asyncio.sleep(5.0)  # Check every 5 seconds
                await self._check_scaling()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scale loop error: {e}")

    async def _check_scaling(self) -> None:
        """Check and adjust pool size based on utilization."""
        if not self._executors:
            return

        # Calculate utilization
        busy_count = sum(
            1 for e in self._executors.values()
            if e.status == ExecutorStatus.BUSY
        )
        utilization = busy_count / len(self._executors)

        # Scale up
        if utilization > self._config.scale_up_threshold:
            if len(self._executors) < self._config.max_executors:
                await self._add_executor()
                logger.debug(f"Scaled up to {len(self._executors)} executors")

        # Scale down
        elif utilization < self._config.scale_down_threshold:
            if len(self._executors) > self._config.min_executors:
                # Find idle executor to remove
                for executor_id, executor in list(self._executors.items()):
                    if executor.is_available:
                        await self._remove_executor(executor_id)
                        logger.debug(f"Scaled down to {len(self._executors)} executors")
                        break

    @property
    def executor_count(self) -> int:
        """Get number of executors in pool."""
        return len(self._executors)

    @property
    def busy_count(self) -> int:
        """Get number of busy executors."""
        return sum(
            1 for e in self._executors.values()
            if e.status == ExecutorStatus.BUSY
        )

    @property
    def utilization(self) -> float:
        """Get pool utilization (0.0 to 1.0)."""
        if not self._executors:
            return 0.0
        return self.busy_count / len(self._executors)

    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        executor_stats = {
            eid: {
                "status": e.status.value,
                "tasks_completed": e.stats.tasks_completed,
                "tasks_failed": e.stats.tasks_failed,
                "avg_execution_time_ms": e.stats.avg_execution_time_ms,
            }
            for eid, e in self._executors.items()
        }

        total_completed = sum(e.stats.tasks_completed for e in self._executors.values())
        total_failed = sum(e.stats.tasks_failed for e in self._executors.values())

        return {
            "executor_count": len(self._executors),
            "busy_count": self.busy_count,
            "utilization": self.utilization,
            "total_completed": total_completed,
            "total_failed": total_failed,
            "executors": executor_stats,
        }
