"""
Agent Scheduler - Fair task distribution across agents.

Provides task queuing, scheduling, and lifecycle management for agent workloads.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from heapq import heappop, heappush
from typing import Any
from collections.abc import Callable, Coroutine

from .models import (
    Priority,
    Task,
    TaskHandle,
    TaskStatus,
)

logger = logging.getLogger(__name__)


@dataclass(order=True)
class _PriorityTask:
    """Internal wrapper for priority queue ordering."""

    priority: int
    created_at: float
    task: Task = field(compare=False)
    agent_id: str = field(compare=False)


class AgentScheduler:
    """
    Agent task scheduler with fair queuing and priority support.

    Features:
    - Priority-based scheduling (critical, high, normal, low)
    - Per-agent task queues with configurable depth
    - Task dependencies
    - Timeout handling
    - Task cancellation
    """

    def __init__(
        self,
        max_queue_depth: int = 1000,
        default_timeout_seconds: float = 300.0,
    ) -> None:
        """
        Initialize the scheduler.

        Args:
            max_queue_depth: Maximum tasks per agent queue
            default_timeout_seconds: Default task timeout
        """
        self._max_queue_depth = max_queue_depth
        self._default_timeout = default_timeout_seconds

        # Task storage
        self._tasks: dict[str, TaskHandle] = {}
        self._task_to_task: dict[str, Task] = {}

        # Per-agent priority queues
        self._agent_queues: dict[str, list[_PriorityTask]] = defaultdict(list)
        self._agent_running: dict[str, set[str]] = defaultdict(set)

        # Dependency tracking
        self._dependencies: dict[str, set[str]] = {}  # task_id -> depends_on
        self._dependents: dict[str, set[str]] = defaultdict(set)  # task_id -> tasks waiting

        # Callbacks
        self._on_complete: dict[str, list[Callable[[TaskHandle], Coroutine[Any, Any, None]]]] = (
            defaultdict(list)
        )

        # Cancellation tokens
        self._cancel_tokens: dict[str, asyncio.Event] = {}

        # Lock for thread safety
        self._lock = asyncio.Lock()

        # Metrics
        self._tasks_scheduled = 0
        self._tasks_completed = 0
        self._tasks_failed = 0
        self._tasks_cancelled = 0

    async def schedule(
        self,
        task: Task,
        agent_id: str,
        priority: Priority = Priority.NORMAL,
        depends_on: list[str] | None = None,
        timeout_seconds: float | None = None,
    ) -> TaskHandle:
        """
        Schedule a task for execution by an agent.

        Args:
            task: Task to schedule
            agent_id: Target agent ID
            priority: Task priority
            depends_on: Task IDs this task depends on
            timeout_seconds: Task timeout override

        Returns:
            TaskHandle for tracking the task

        Raises:
            ValueError: If queue is full or invalid dependencies
        """
        async with self._lock:
            # Check queue depth
            queue = self._agent_queues[agent_id]
            if len(queue) >= self._max_queue_depth:
                raise ValueError(f"Queue full for agent {agent_id} (max {self._max_queue_depth})")

            # Validate dependencies exist
            deps = set(depends_on or [])
            for dep_id in deps:
                if dep_id not in self._tasks:
                    raise ValueError(f"Dependency task {dep_id} not found")

            # Create task handle
            handle = TaskHandle(
                task_id=task.id,
                agent_id=agent_id,
                status=TaskStatus.PENDING,
                scheduled_at=datetime.now(timezone.utc),
            )

            # Store task
            self._tasks[task.id] = handle
            self._task_to_task[task.id] = task

            # Track dependencies
            if deps:
                self._dependencies[task.id] = deps
                for dep_id in deps:
                    self._dependents[dep_id].add(task.id)

                # Check if dependencies are already complete
                pending_deps = {
                    d for d in deps if self._tasks[d].status not in (TaskStatus.COMPLETED,)
                }
                if pending_deps:
                    # Wait for dependencies
                    logger.debug("Task %s waiting on dependencies: %s", task.id, pending_deps)
                    return handle

            # Add to priority queue
            timeout = timeout_seconds or task.timeout_seconds or self._default_timeout
            task.timeout_seconds = timeout

            priority_task = _PriorityTask(
                priority=priority.value,
                created_at=task.created_at.timestamp(),
                task=task,
                agent_id=agent_id,
            )
            heappush(queue, priority_task)
            handle.status = TaskStatus.SCHEDULED

            # Create cancellation token
            self._cancel_tokens[task.id] = asyncio.Event()

            self._tasks_scheduled += 1
            logger.debug(
                "Scheduled task %s for agent %s (priority=%s)", task.id, agent_id, priority.name
            )

            return handle

    async def cancel(self, task_id: str) -> bool:
        """
        Cancel a pending or running task.

        Args:
            task_id: Task to cancel

        Returns:
            True if cancelled, False if already complete
        """
        async with self._lock:
            if task_id not in self._tasks:
                return False

            handle = self._tasks[task_id]
            if handle.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
                return False

            # Signal cancellation
            if task_id in self._cancel_tokens:
                self._cancel_tokens[task_id].set()

            handle.status = TaskStatus.CANCELLED
            handle.completed_at = datetime.now(timezone.utc)
            self._tasks_cancelled += 1

            # Notify dependents
            await self._handle_task_complete(task_id, success=False)

            logger.info("Cancelled task %s", task_id)
            return True

    async def get_status(self, task_id: str) -> TaskStatus | None:
        """Get the status of a task."""
        handle = self._tasks.get(task_id)
        return handle.status if handle else None

    async def get_handle(self, task_id: str) -> TaskHandle | None:
        """Get the full handle for a task."""
        return self._tasks.get(task_id)

    async def list_pending(self, agent_id: str) -> list[Task]:
        """List pending tasks for an agent."""
        async with self._lock:
            queue = self._agent_queues.get(agent_id, [])
            return [pt.task for pt in queue]

    async def list_running(self, agent_id: str) -> list[Task]:
        """List running tasks for an agent."""
        async with self._lock:
            running_ids = self._agent_running.get(agent_id, set())
            return [self._task_to_task[tid] for tid in running_ids if tid in self._task_to_task]

    async def pop_next(self, agent_id: str) -> Task | None:
        """
        Pop the next task for an agent to execute.

        Args:
            agent_id: Agent requesting work

        Returns:
            Next task or None if queue empty
        """
        async with self._lock:
            queue = self._agent_queues.get(agent_id)
            if not queue:
                return None

            priority_task = heappop(queue)
            task = priority_task.task
            handle = self._tasks[task.id]

            handle.status = TaskStatus.RUNNING
            handle.started_at = datetime.now(timezone.utc)
            self._agent_running[agent_id].add(task.id)

            logger.debug("Agent %s starting task %s", agent_id, task.id)
            return task

    async def complete_task(
        self,
        task_id: str,
        result: Any = None,
        error: str | None = None,
    ) -> None:
        """
        Mark a task as complete.

        Args:
            task_id: Task that completed
            result: Task result (if successful)
            error: Error message (if failed)
        """
        async with self._lock:
            if task_id not in self._tasks:
                return

            handle = self._tasks[task_id]
            handle.completed_at = datetime.now(timezone.utc)
            handle.result = result
            handle.error = error

            if error:
                handle.status = TaskStatus.FAILED
                self._tasks_failed += 1
            else:
                handle.status = TaskStatus.COMPLETED
                self._tasks_completed += 1

            # Remove from running set
            agent_id = handle.agent_id
            if agent_id in self._agent_running:
                self._agent_running[agent_id].discard(task_id)

            # Clean up cancellation token
            self._cancel_tokens.pop(task_id, None)

            # Handle dependents
            await self._handle_task_complete(task_id, success=not error)

            # Fire callbacks
            for callback in self._on_complete.get(task_id, []):
                try:
                    await callback(handle)
                except (RuntimeError, ValueError, AttributeError) as e:  # user-supplied callback
                    logger.warning("Callback error for task %s: %s", task_id, e)

            logger.debug("Task %s completed (status=%s)", task_id, handle.status.value)

    async def _handle_task_complete(self, task_id: str, success: bool) -> None:
        """Handle task completion and unblock dependents."""
        # Get tasks waiting on this one
        dependents = self._dependents.pop(task_id, set())

        for dep_task_id in dependents:
            if dep_task_id not in self._dependencies:
                continue

            # Remove this task from dependencies
            self._dependencies[dep_task_id].discard(task_id)

            # If all dependencies complete, schedule the task
            if not self._dependencies[dep_task_id]:
                del self._dependencies[dep_task_id]

                if success:
                    # Get the task and re-queue it
                    task = self._task_to_task.get(dep_task_id)
                    handle = self._tasks.get(dep_task_id)
                    if task and handle:
                        priority_task = _PriorityTask(
                            priority=Priority.NORMAL.value,
                            created_at=task.created_at.timestamp(),
                            task=task,
                            agent_id=handle.agent_id,
                        )
                        heappush(self._agent_queues[handle.agent_id], priority_task)
                        handle.status = TaskStatus.SCHEDULED
                        logger.debug("Dependency resolved, scheduled task %s", dep_task_id)
                else:
                    # Dependency failed, fail this task too
                    handle = self._tasks.get(dep_task_id)
                    if handle:
                        handle.status = TaskStatus.FAILED
                        handle.error = f"Dependency {task_id} failed"
                        handle.completed_at = datetime.now(timezone.utc)
                        logger.debug("Task %s failed due to dependency failure", dep_task_id)

    def on_complete(
        self,
        task_id: str,
        callback: Callable[[TaskHandle], Coroutine[Any, Any, None]],
    ) -> None:
        """Register a callback for task completion."""
        self._on_complete[task_id].append(callback)

    def is_cancelled(self, task_id: str) -> bool:
        """Check if a task has been cancelled."""
        token = self._cancel_tokens.get(task_id)
        return token.is_set() if token else False

    async def get_stats(self) -> dict[str, Any]:
        """Get scheduler statistics."""
        async with self._lock:
            total_pending = sum(len(q) for q in self._agent_queues.values())
            total_running = sum(len(s) for s in self._agent_running.values())

            return {
                "tasks_scheduled": self._tasks_scheduled,
                "tasks_completed": self._tasks_completed,
                "tasks_failed": self._tasks_failed,
                "tasks_cancelled": self._tasks_cancelled,
                "tasks_pending": total_pending,
                "tasks_running": total_running,
                "agents_with_work": len(self._agent_queues),
            }
