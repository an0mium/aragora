"""
Task Queue with Priority and Dependency Management.

Provides the main TaskQueue class for scheduling and managing workflow tasks.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Set

if TYPE_CHECKING:
    from aragora.workflow.queue.executor import TaskExecutor

from aragora.workflow.queue.task import (
    TaskStatus,
    TaskResult,
    WorkflowTask,
)

logger = logging.getLogger(__name__)


@dataclass
class TaskQueueConfig:
    """Configuration for the task queue."""

    max_concurrent: int = 10
    max_queue_size: int = 1000
    default_timeout: float = 300.0
    enable_retries: bool = True
    max_retries: int = 2
    retry_delay_seconds: float = 5.0
    enable_priority: bool = True
    drain_timeout_seconds: float = 60.0


@dataclass
class QueueStats:
    """Statistics about the task queue."""

    pending_count: int = 0
    ready_count: int = 0
    running_count: int = 0
    completed_count: int = 0
    failed_count: int = 0
    total_tasks: int = 0
    avg_wait_time_ms: float = 0
    avg_execution_time_ms: float = 0


class TaskQueue:
    """
    Asynchronous task queue with priority and dependency resolution.

    Features:
    - Priority-based task ordering
    - Dependency tracking and automatic resolution
    - Concurrent execution management
    - Task lifecycle events
    - Workflow-level aggregation

    Usage:
        queue = TaskQueue(config=TaskQueueConfig(max_concurrent=5))
        await queue.start()

        task = WorkflowTask.create(workflow_id="wf1", step_id="step1")
        await queue.enqueue(task)

        result = await queue.wait_for_task(task.id)
    """

    def __init__(self, config: Optional[TaskQueueConfig] = None):
        self._config = config or TaskQueueConfig()

        # Task storage
        self._tasks: Dict[str, WorkflowTask] = {}
        self._workflows: Dict[str, Set[str]] = defaultdict(set)  # workflow_id -> task_ids

        # Priority queue (heapq would be better but keeping simple)
        self._ready_queue: List[str] = []

        # Dependency tracking
        self._dependents: Dict[str, Set[str]] = defaultdict(
            set
        )  # task_id -> tasks that depend on it
        self._pending_deps: Dict[str, Set[str]] = {}  # task_id -> unresolved dependencies

        # Execution state
        self._running: Set[str] = set()
        self._semaphore: asyncio.Semaphore = asyncio.Semaphore(self._config.max_concurrent)

        # Control
        self._started = False
        self._stopping = False
        self._processor_task: Optional[asyncio.Task] = None

        # Events
        self._task_completed: Dict[str, asyncio.Event] = {}
        self._workflow_completed: Dict[str, asyncio.Event] = {}

        # Callbacks
        self._on_task_complete: Optional[Callable[[WorkflowTask], None]] = None
        self._on_task_error: Optional[Callable[[WorkflowTask, Exception], None]] = None

        # Executor reference (set by scheduler)
        self._executor: Optional["TaskExecutor"] = None

    async def start(self) -> None:
        """Start the queue processor."""
        if self._started:
            return

        self._started = True
        self._stopping = False
        self._processor_task = asyncio.create_task(self._process_loop())
        logger.info("Task queue started")

    async def stop(self, drain: bool = True) -> None:
        """
        Stop the queue processor.

        Args:
            drain: If True, wait for running tasks to complete
        """
        if not self._started:
            return

        self._stopping = True

        if drain:
            # Wait for running tasks
            if self._running:
                logger.info(f"Draining {len(self._running)} running tasks...")
                try:
                    await asyncio.wait_for(
                        self._wait_for_drain(),
                        timeout=self._config.drain_timeout_seconds,
                    )
                except asyncio.TimeoutError:
                    logger.warning("Drain timeout, some tasks still running")

        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass

        self._started = False
        logger.info("Task queue stopped")

    async def _wait_for_drain(self) -> None:
        """Wait for all running tasks to complete."""
        while self._running:
            await asyncio.sleep(0.1)

    async def enqueue(self, task: WorkflowTask) -> str:
        """
        Add a task to the queue.

        Args:
            task: Task to enqueue

        Returns:
            Task ID
        """
        if len(self._tasks) >= self._config.max_queue_size:
            raise RuntimeError("Queue is full")

        # Store task
        self._tasks[task.id] = task
        self._workflows[task.workflow_id].add(task.id)

        # Create events
        self._task_completed[task.id] = asyncio.Event()

        # Track dependencies
        if task.depends_on:
            self._pending_deps[task.id] = set(task.depends_on)
            for dep_id in task.depends_on:
                self._dependents[dep_id].add(task.id)
        else:
            # No dependencies - mark as ready
            task.mark_ready()
            self._add_to_ready_queue(task.id)

        logger.debug(f"Enqueued task {task.id} for workflow {task.workflow_id}")
        return task.id

    async def enqueue_many(self, tasks: List[WorkflowTask]) -> List[str]:
        """Enqueue multiple tasks."""
        return [await self.enqueue(task) for task in tasks]

    def _add_to_ready_queue(self, task_id: str) -> None:
        """Add task to ready queue with priority sorting."""
        task = self._tasks.get(task_id)
        if not task:
            return

        self._ready_queue.append(task_id)

        if self._config.enable_priority:
            # Sort by priority (lower = higher priority)
            self._ready_queue.sort(key=lambda tid: self._tasks[tid].priority)

    async def _process_loop(self) -> None:
        """Main processing loop."""
        while not self._stopping:
            try:
                await self._process_ready_tasks()
                await asyncio.sleep(0.01)  # Small delay to prevent busy-waiting
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in process loop: {e}")
                await asyncio.sleep(0.1)

    async def _process_ready_tasks(self) -> None:
        """Process tasks from the ready queue."""
        while self._ready_queue and not self._stopping:
            # Check if we can run more tasks
            if len(self._running) >= self._config.max_concurrent:
                return

            task_id = self._ready_queue.pop(0)
            task = self._tasks.get(task_id)

            if not task or not task.is_runnable:
                continue

            # Start execution
            self._running.add(task_id)
            asyncio.create_task(self._execute_task(task))

    async def _execute_task(self, task: WorkflowTask) -> None:
        """Execute a single task."""
        try:
            async with self._semaphore:
                task.mark_running(executor_id="queue")

                logger.debug(f"Executing task {task.id}")

                # Execute via executor if available
                if self._executor:
                    result = await self._executor.execute(task)
                else:
                    # Placeholder execution
                    result = TaskResult(
                        success=True,
                        output={"executed": True, "step_id": task.step_id},
                        execution_time_ms=0,
                    )

                task.mark_completed(result)
                logger.debug(f"Task {task.id} completed")

                if self._on_task_complete:
                    self._on_task_complete(task)

        except asyncio.TimeoutError:
            task.mark_timeout()
            if task.schedule_retry():
                self._add_to_ready_queue(task.id)
            logger.warning(f"Task {task.id} timed out")

        except Exception as e:
            task.mark_failed(str(e))
            logger.error(f"Task {task.id} failed: {e}")

            if self._on_task_error:
                self._on_task_error(task, e)

            if task.schedule_retry():
                await asyncio.sleep(self._config.retry_delay_seconds)
                self._add_to_ready_queue(task.id)

        finally:
            self._running.discard(task.id)
            self._task_completed[task.id].set()

            # Resolve dependencies
            await self._resolve_dependencies(task.id)

            # Check workflow completion
            await self._check_workflow_complete(task.workflow_id)

    async def _resolve_dependencies(self, completed_task_id: str) -> None:
        """Resolve dependencies for tasks waiting on completed task."""
        task = self._tasks.get(completed_task_id)

        # Only resolve if task completed successfully
        if not task or task.status != TaskStatus.COMPLETED:
            return

        # Find dependent tasks
        dependent_ids = self._dependents.pop(completed_task_id, set())

        for dep_id in dependent_ids:
            pending = self._pending_deps.get(dep_id)
            if pending:
                pending.discard(completed_task_id)

                # All dependencies resolved?
                if not pending:
                    del self._pending_deps[dep_id]
                    dep_task = self._tasks.get(dep_id)
                    if dep_task and dep_task.status == TaskStatus.PENDING:
                        dep_task.mark_ready()
                        self._add_to_ready_queue(dep_id)
                        logger.debug(f"Task {dep_id} dependencies resolved, now ready")

    async def _check_workflow_complete(self, workflow_id: str) -> None:
        """Check if all tasks in a workflow are complete."""
        task_ids = self._workflows.get(workflow_id, set())

        all_complete = all(self._tasks[tid].is_terminal for tid in task_ids if tid in self._tasks)

        if all_complete:
            event = self._workflow_completed.get(workflow_id)
            if event:
                event.set()

    async def wait_for_task(
        self,
        task_id: str,
        timeout: Optional[float] = None,
    ) -> Optional[TaskResult]:
        """
        Wait for a task to complete.

        Args:
            task_id: Task to wait for
            timeout: Maximum wait time in seconds

        Returns:
            Task result or None if timeout
        """
        event = self._task_completed.get(task_id)
        if not event:
            return None

        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
            task = self._tasks.get(task_id)
            return task.result if task else None
        except asyncio.TimeoutError:
            return None

    async def wait_for_workflow(
        self,
        workflow_id: str,
        timeout: Optional[float] = None,
    ) -> Dict[str, TaskResult]:
        """
        Wait for all tasks in a workflow to complete.

        Args:
            workflow_id: Workflow to wait for
            timeout: Maximum wait time in seconds

        Returns:
            Dictionary of task_id -> TaskResult
        """
        if workflow_id not in self._workflow_completed:
            self._workflow_completed[workflow_id] = asyncio.Event()

        event = self._workflow_completed[workflow_id]

        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            pass

        # Collect results
        task_ids = self._workflows.get(workflow_id, set())
        return {
            tid: self._tasks[tid].result
            for tid in task_ids
            if tid in self._tasks and self._tasks[tid].result
        }

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a task if not running."""
        task = self._tasks.get(task_id)
        if task and not task.is_terminal and task.status != TaskStatus.RUNNING:
            task.mark_cancelled()
            if task_id in self._ready_queue:
                self._ready_queue.remove(task_id)
            return True
        return False

    def cancel_workflow(self, workflow_id: str) -> int:
        """Cancel all pending tasks in a workflow."""
        cancelled = 0
        task_ids = self._workflows.get(workflow_id, set())
        for task_id in task_ids:
            if self.cancel_task(task_id):
                cancelled += 1
        return cancelled

    def get_task(self, task_id: str) -> Optional[WorkflowTask]:
        """Get a task by ID."""
        return self._tasks.get(task_id)

    def get_workflow_tasks(self, workflow_id: str) -> List[WorkflowTask]:
        """Get all tasks for a workflow."""
        task_ids = self._workflows.get(workflow_id, set())
        return [self._tasks[tid] for tid in task_ids if tid in self._tasks]

    def get_stats(self) -> QueueStats:
        """Get queue statistics."""
        tasks = list(self._tasks.values())

        pending = sum(1 for t in tasks if t.status == TaskStatus.PENDING)
        ready = len(self._ready_queue)
        running = len(self._running)
        completed = sum(1 for t in tasks if t.status == TaskStatus.COMPLETED)
        failed = sum(1 for t in tasks if t.status in (TaskStatus.FAILED, TaskStatus.TIMEOUT))

        # Calculate averages
        completed_tasks = [t for t in tasks if t.status == TaskStatus.COMPLETED]
        avg_wait = (
            sum(t.wait_time_ms for t in completed_tasks) / len(completed_tasks)
            if completed_tasks
            else 0
        )
        avg_exec = (
            sum(t.execution_time_ms for t in completed_tasks) / len(completed_tasks)
            if completed_tasks
            else 0
        )

        return QueueStats(
            pending_count=pending,
            ready_count=ready,
            running_count=running,
            completed_count=completed,
            failed_count=failed,
            total_tasks=len(tasks),
            avg_wait_time_ms=avg_wait,
            avg_execution_time_ms=avg_exec,
        )

    def set_executor(self, executor: "TaskExecutor") -> None:
        """Set the task executor."""
        self._executor = executor

    def on_task_complete(self, callback: Callable[[WorkflowTask], None]) -> None:
        """Set callback for task completion."""
        self._on_task_complete = callback

    def on_task_error(self, callback: Callable[[WorkflowTask, Exception], None]) -> None:
        """Set callback for task errors."""
        self._on_task_error = callback
