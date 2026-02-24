"""Hierarchical Task Dispatcher for multi-agent worktree coordination.

Breaks high-level goals into subtasks, assigns them to available worktrees,
and tracks completion with dependency-aware scheduling.

Usage:
    from aragora.coordination.task_dispatcher import TaskDispatcher

    dispatcher = TaskDispatcher()
    task = dispatcher.submit("Refactor auth module", priority=1, track="security")
    dispatcher.assign(task.task_id, worktree_id="abc123")
    dispatcher.complete(task.task_id)
"""

from __future__ import annotations

import heapq
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)


@dataclass(order=False)
class Task:
    """A unit of work to be dispatched to a worktree."""

    task_id: str = field(default_factory=lambda: str(uuid4())[:8])
    title: str = ""
    description: str = ""
    priority: int = 5  # 1 = highest, 10 = lowest
    track: str | None = None
    status: str = "pending"  # pending, assigned, running, completed, failed, cancelled
    worktree_id: str | None = None
    agent_id: str | None = None
    depends_on: list[str] = field(default_factory=list)
    blocked_by: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    assigned_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    retry_count: int = 0
    max_retries: int = 2
    stall_timeout_seconds: int = 600

    def __lt__(self, other: Task) -> bool:
        """Priority comparison for heap ordering."""
        return self.priority < other.priority


@dataclass
class DispatcherConfig:
    """Configuration for TaskDispatcher."""

    max_retries: int = 2
    default_stall_timeout: int = 600
    max_concurrent: int = 12


class TaskDispatcher:
    """Priority-based task dispatcher with dependency tracking.

    Maintains a priority queue of tasks, assigns them to worktrees,
    detects stalls, and reassigns failed tasks.
    """

    def __init__(self, config: DispatcherConfig | None = None):
        self.config = config or DispatcherConfig()
        self._tasks: dict[str, Task] = {}
        self._queue: list[tuple[int, str]] = []  # (priority, task_id) min-heap
        self._completed_ids: set[str] = set()

    @property
    def pending_tasks(self) -> list[Task]:
        """Tasks waiting to be assigned."""
        return [t for t in self._tasks.values() if t.status == "pending"]

    @property
    def running_tasks(self) -> list[Task]:
        """Tasks currently running."""
        return [t for t in self._tasks.values() if t.status in ("assigned", "running")]

    @property
    def completed_tasks(self) -> list[Task]:
        """Tasks that finished (completed or failed)."""
        return [t for t in self._tasks.values() if t.status in ("completed", "failed")]

    def submit(
        self,
        title: str,
        description: str = "",
        priority: int = 5,
        track: str | None = None,
        depends_on: list[str] | None = None,
        stall_timeout: int | None = None,
    ) -> Task:
        """Submit a new task to the dispatcher.

        Args:
            title: Short task title.
            description: Detailed description.
            priority: 1 (highest) to 10 (lowest).
            track: Development track for affinity routing.
            depends_on: Task IDs that must complete first.
            stall_timeout: Override default stall timeout.

        Returns:
            The created Task.
        """
        task = Task(
            title=title,
            description=description,
            priority=max(1, min(10, priority)),
            track=track,
            depends_on=depends_on or [],
            stall_timeout_seconds=stall_timeout or self.config.default_stall_timeout,
            max_retries=self.config.max_retries,
        )

        # Compute initial blocked_by from depends_on
        task.blocked_by = [
            dep_id for dep_id in task.depends_on
            if dep_id not in self._completed_ids
        ]

        self._tasks[task.task_id] = task

        if not task.blocked_by:
            heapq.heappush(self._queue, (task.priority, task.task_id))

        logger.info(
            "task_submitted id=%s title=%s priority=%d deps=%s",
            task.task_id, title, task.priority, task.depends_on,
        )
        return task

    def get_next(self, track: str | None = None) -> Task | None:
        """Get the highest-priority unblocked task.

        Args:
            track: If provided, prefer tasks matching this track.

        Returns:
            The next task, or None if no tasks available.
        """
        # Track-affinity: scan for matching track first
        if track:
            for priority, task_id in sorted(self._queue):
                task = self._tasks.get(task_id)
                if task and task.status == "pending" and task.track == track and not task.blocked_by:
                    return task

        # General: pop from priority queue
        while self._queue:
            _priority, task_id = heapq.heappop(self._queue)
            task = self._tasks.get(task_id)
            if task and task.status == "pending" and not task.blocked_by:
                return task
        return None

    def assign(self, task_id: str, worktree_id: str, agent_id: str | None = None) -> bool:
        """Assign a task to a worktree.

        Returns:
            True if assignment succeeded.
        """
        task = self._tasks.get(task_id)
        if not task or task.status != "pending":
            return False
        if task.blocked_by:
            logger.warning("task_blocked id=%s blocked_by=%s", task_id, task.blocked_by)
            return False

        task.status = "assigned"
        task.worktree_id = worktree_id
        task.agent_id = agent_id
        task.assigned_at = datetime.now(timezone.utc)

        logger.info("task_assigned id=%s worktree=%s", task_id, worktree_id)
        return True

    def start(self, task_id: str) -> bool:
        """Mark a task as running (agent has started work)."""
        task = self._tasks.get(task_id)
        if not task or task.status != "assigned":
            return False
        task.status = "running"
        task.started_at = datetime.now(timezone.utc)
        return True

    def complete(self, task_id: str, result: dict[str, Any] | None = None) -> bool:
        """Mark a task as completed and unblock dependents.

        Returns:
            True if the task was marked complete.
        """
        task = self._tasks.get(task_id)
        if not task or task.status not in ("assigned", "running"):
            return False

        task.status = "completed"
        task.completed_at = datetime.now(timezone.utc)
        task.result = result
        self._completed_ids.add(task_id)

        # Unblock dependents
        self._unblock_dependents(task_id)

        logger.info("task_completed id=%s title=%s", task_id, task.title)
        return True

    def fail(self, task_id: str, error: str = "") -> bool:
        """Mark a task as failed. May be retried if retries remain.

        Returns:
            True if marked failed (not retried).
        """
        task = self._tasks.get(task_id)
        if not task or task.status not in ("assigned", "running"):
            return False

        task.retry_count += 1
        if task.retry_count <= task.max_retries:
            # Reset for retry
            task.status = "pending"
            task.worktree_id = None
            task.agent_id = None
            task.assigned_at = None
            task.started_at = None
            task.error = error
            heapq.heappush(self._queue, (task.priority, task.task_id))
            logger.info(
                "task_retry id=%s attempt=%d/%d",
                task_id, task.retry_count, task.max_retries,
            )
            return False

        task.status = "failed"
        task.completed_at = datetime.now(timezone.utc)
        task.error = error
        logger.warning("task_failed id=%s error=%s", task_id, error)
        return True

    def cancel(self, task_id: str) -> bool:
        """Cancel a pending or assigned task."""
        task = self._tasks.get(task_id)
        if not task or task.status in ("completed", "failed", "cancelled"):
            return False
        task.status = "cancelled"
        task.completed_at = datetime.now(timezone.utc)
        return True

    def reassign(self, task_id: str, new_worktree_id: str, agent_id: str | None = None) -> bool:
        """Reassign a stalled or failed task to a different worktree."""
        task = self._tasks.get(task_id)
        if not task:
            return False
        if task.status not in ("assigned", "running", "pending"):
            return False

        old_worktree = task.worktree_id
        task.status = "assigned"
        task.worktree_id = new_worktree_id
        task.agent_id = agent_id
        task.assigned_at = datetime.now(timezone.utc)
        task.started_at = None

        logger.info(
            "task_reassigned id=%s from=%s to=%s",
            task_id, old_worktree, new_worktree_id,
        )
        return True

    def get_stalled_tasks(self) -> list[Task]:
        """Return tasks that have been running past their stall timeout."""
        now = datetime.now(timezone.utc)
        stalled = []
        for task in self._tasks.values():
            if task.status not in ("assigned", "running"):
                continue
            ref_time = task.started_at or task.assigned_at
            if ref_time is None:
                continue
            elapsed = (now - ref_time).total_seconds()
            if elapsed >= task.stall_timeout_seconds:
                stalled.append(task)
        return stalled

    def _unblock_dependents(self, completed_task_id: str) -> None:
        """Remove completed_task_id from all blocked_by lists and enqueue newly-unblocked tasks."""
        for task in self._tasks.values():
            if completed_task_id in task.blocked_by:
                task.blocked_by.remove(completed_task_id)
                if not task.blocked_by and task.status == "pending":
                    heapq.heappush(self._queue, (task.priority, task.task_id))
                    logger.info("task_unblocked id=%s", task.task_id)

    def summary(self) -> dict[str, int]:
        """Return a count of tasks by status."""
        counts: dict[str, int] = {}
        for task in self._tasks.values():
            counts[task.status] = counts.get(task.status, 0) + 1
        return counts

    def decompose(
        self,
        goal: str,
        subtasks: list[dict[str, Any]],
    ) -> list[Task]:
        """Decompose a high-level goal into subtasks.

        Args:
            goal: Top-level goal description.
            subtasks: List of dicts with keys: title, description, priority, track, depends_on.

        Returns:
            List of created Tasks.
        """
        created = []
        # First pass: create tasks and collect temporary ID mappings
        temp_to_real: dict[str, str] = {}
        for i, spec in enumerate(subtasks):
            task = self.submit(
                title=spec.get("title", f"Subtask {i + 1} of: {goal[:40]}"),
                description=spec.get("description", ""),
                priority=spec.get("priority", 5),
                track=spec.get("track"),
                depends_on=[],  # Set after all tasks created
            )
            temp_id = spec.get("id", str(i))
            temp_to_real[temp_id] = task.task_id
            created.append(task)

        # Second pass: wire up dependencies
        for i, spec in enumerate(subtasks):
            raw_deps = spec.get("depends_on", [])
            if raw_deps:
                task = created[i]
                task.depends_on = [temp_to_real.get(d, d) for d in raw_deps]
                task.blocked_by = [
                    dep_id for dep_id in task.depends_on
                    if dep_id not in self._completed_ids
                ]

        return created


__all__ = [
    "TaskDispatcher",
    "DispatcherConfig",
    "Task",
]
