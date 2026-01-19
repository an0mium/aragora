"""
Task Scheduler for the Aragora Control Plane.

Provides distributed task scheduling with:
- Priority-based queue ordering
- Capability-based agent matching
- Task timeout and retry handling
- Redis Streams backend for durability

The scheduler works with the AgentRegistry to assign tasks
to the most suitable available agents.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from aragora.server.prometheus import (
    record_control_plane_task_submitted,
    record_control_plane_task_completed,
    record_control_plane_task_retry,
    record_control_plane_claim_latency,
    record_control_plane_queue_depth,
)

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task lifecycle status."""

    PENDING = "pending"  # Task is queued, waiting for assignment
    ASSIGNED = "assigned"  # Task assigned to an agent
    RUNNING = "running"  # Task is being executed
    COMPLETED = "completed"  # Task completed successfully
    FAILED = "failed"  # Task failed after all retries
    CANCELLED = "cancelled"  # Task was cancelled
    TIMEOUT = "timeout"  # Task timed out


class TaskPriority(Enum):
    """Task priority levels."""

    LOW = 0
    NORMAL = 50
    HIGH = 75
    URGENT = 100


@dataclass
class Task:
    """
    Represents a task in the scheduling system.

    Attributes:
        id: Unique task identifier
        task_type: Type of task (e.g., "debate", "code", "analysis")
        payload: Task-specific data
        required_capabilities: Capabilities required to execute
        status: Current task status
        priority: Task priority
        created_at: Creation timestamp
        assigned_at: When task was assigned
        started_at: When execution started
        completed_at: When task completed/failed
        assigned_agent: Agent assigned to this task
        timeout_seconds: Task timeout
        max_retries: Maximum retry attempts
        retries: Current retry count
        result: Task result (if completed)
        error: Error message (if failed)
        metadata: Additional task metadata
    """

    task_type: str
    payload: Dict[str, Any]
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    required_capabilities: Set[str] = field(default_factory=set)
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.NORMAL
    created_at: float = field(default_factory=time.time)
    assigned_at: Optional[float] = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    assigned_agent: Optional[str] = None
    timeout_seconds: float = 300.0
    max_retries: int = 3
    retries: int = 0
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_timed_out(self) -> bool:
        """Check if task has timed out."""
        if self.started_at is None:
            return False
        return (time.time() - self.started_at) > self.timeout_seconds

    def should_retry(self) -> bool:
        """Check if task should be retried."""
        return self.retries < self.max_retries

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "task_type": self.task_type,
            "payload": self.payload,
            "required_capabilities": list(self.required_capabilities),
            "status": self.status.value,
            "priority": self.priority.value,
            "created_at": self.created_at,
            "assigned_at": self.assigned_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "assigned_agent": self.assigned_agent,
            "timeout_seconds": self.timeout_seconds,
            "max_retries": self.max_retries,
            "retries": self.retries,
            "result": self.result,
            "error": self.error,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        """Deserialize from dictionary."""
        return cls(
            id=data["id"],
            task_type=data["task_type"],
            payload=data["payload"],
            required_capabilities=set(data.get("required_capabilities", [])),
            status=TaskStatus(data.get("status", "pending")),
            priority=TaskPriority(data.get("priority", 50)),
            created_at=data.get("created_at", time.time()),
            assigned_at=data.get("assigned_at"),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            assigned_agent=data.get("assigned_agent"),
            timeout_seconds=data.get("timeout_seconds", 300.0),
            max_retries=data.get("max_retries", 3),
            retries=data.get("retries", 0),
            result=data.get("result"),
            error=data.get("error"),
            metadata=data.get("metadata", {}),
        )


class TaskScheduler:
    """
    Redis-backed task scheduler for distributed task distribution.

    Uses Redis Streams for durable task queuing with:
    - Priority ordering via multiple streams
    - Consumer groups for work distribution
    - Pending entry tracking for timeout detection

    Usage:
        scheduler = TaskScheduler(redis_url="redis://localhost:6379")
        await scheduler.connect()

        # Submit a task
        task_id = await scheduler.submit(
            task_type="debate",
            payload={"question": "..."},
            required_capabilities=["debate"],
            priority=TaskPriority.HIGH,
        )

        # Claim a task (from worker)
        task = await scheduler.claim(worker_id="worker-1", capabilities=["debate"])

        # Complete a task
        await scheduler.complete(task_id, result={"conclusion": "..."})

        await scheduler.close()
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        key_prefix: str = "aragora:cp:tasks:",
        stream_prefix: str = "aragora:cp:stream:",
        consumer_group: str = "aragora-workers",
        claim_timeout_ms: int = 60000,
    ):
        """
        Initialize the task scheduler.

        Args:
            redis_url: Redis connection URL
            key_prefix: Prefix for task data keys
            stream_prefix: Prefix for stream keys
            consumer_group: Consumer group name
            claim_timeout_ms: Time before unclaimed task can be reclaimed
        """
        self._redis_url = redis_url
        self._key_prefix = key_prefix
        self._stream_prefix = stream_prefix
        self._consumer_group = consumer_group
        self._claim_timeout_ms = claim_timeout_ms
        self._redis: Optional[Any] = None
        self._local_tasks: Dict[str, Task] = {}
        self._local_queue: List[Task] = []

    async def connect(self) -> None:
        """Connect to Redis."""
        try:
            import redis.asyncio as aioredis

            self._redis = aioredis.from_url(
                self._redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
            await self._redis.ping()

            # Create consumer groups for each priority stream
            for priority in TaskPriority:
                stream_key = f"{self._stream_prefix}{priority.name.lower()}"
                try:
                    await self._redis.xgroup_create(
                        stream_key,
                        self._consumer_group,
                        id="0",
                        mkstream=True,
                    )
                except Exception as e:
                    # Group may already exist (BUSYGROUP error) - this is expected
                    if "BUSYGROUP" not in str(e):
                        logger.debug(f"Consumer group creation note for {stream_key}: {e}")

            logger.info(f"TaskScheduler connected to Redis: {self._redis_url}")

        except ImportError:
            logger.warning("redis package not installed, using in-memory fallback")
            self._redis = None
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self._redis = None

    async def close(self) -> None:
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
            logger.info("TaskScheduler disconnected from Redis")

    async def submit(
        self,
        task_type: str,
        payload: Dict[str, Any],
        required_capabilities: Optional[List[str]] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        timeout_seconds: float = 300.0,
        max_retries: int = 3,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Submit a task for execution.

        Args:
            task_type: Type of task
            payload: Task data
            required_capabilities: Required agent capabilities
            priority: Task priority
            timeout_seconds: Task timeout
            max_retries: Maximum retries
            metadata: Additional metadata

        Returns:
            Task ID
        """
        task = Task(
            task_type=task_type,
            payload=payload,
            required_capabilities=set(required_capabilities or []),
            priority=priority,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            metadata=metadata or {},
        )

        await self._save_task(task)
        await self._enqueue_task(task)

        # Record metrics
        record_control_plane_task_submitted(task_type, priority.name.lower())

        logger.info(
            f"Task submitted: {task.id} (type={task_type}, priority={priority.name})"
        )

        return task.id

    async def claim(
        self,
        worker_id: str,
        capabilities: List[str],
        block_ms: int = 5000,
    ) -> Optional[Task]:
        """
        Claim a task for execution.

        Args:
            worker_id: ID of the worker claiming the task
            capabilities: Capabilities the worker provides
            block_ms: Time to block waiting for a task

        Returns:
            Task if claimed, None if no suitable task available
        """
        if self._redis:
            return await self._claim_from_redis(worker_id, capabilities, block_ms)
        else:
            return self._claim_from_local(worker_id, capabilities)

    async def complete(
        self,
        task_id: str,
        result: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Mark a task as completed.

        Args:
            task_id: Task to complete
            result: Task result

        Returns:
            True if marked complete, False if task not found
        """
        task = await self.get(task_id)
        if not task:
            return False

        task.status = TaskStatus.COMPLETED
        task.completed_at = time.time()
        task.result = result

        await self._save_task(task)
        await self._ack_task(task)

        # Record metrics
        duration = task.completed_at - (task.started_at or task.created_at)
        record_control_plane_task_completed(task.task_type, "completed", duration)

        logger.info(f"Task completed: {task_id}")
        return True

    async def fail(
        self,
        task_id: str,
        error: str,
        requeue: bool = True,
    ) -> bool:
        """
        Mark a task as failed.

        Args:
            task_id: Task that failed
            error: Error message
            requeue: Whether to requeue for retry

        Returns:
            True if processed, False if task not found
        """
        task = await self.get(task_id)
        if not task:
            return False

        task.error = error
        task.retries += 1

        if requeue and task.should_retry():
            task.status = TaskStatus.PENDING
            task.assigned_agent = None
            task.assigned_at = None
            task.started_at = None
            # Clear the message ID since we're requeuing (new stream entry)
            task.metadata.pop("_stream_message_id", None)
            await self._save_task(task)
            await self._enqueue_task(task)

            # Record retry metric
            record_control_plane_task_retry(task.task_type, "error")

            logger.warning(
                f"Task {task_id} failed (attempt {task.retries}/{task.max_retries}), "
                f"requeued: {error}"
            )
        else:
            task.status = TaskStatus.FAILED
            task.completed_at = time.time()
            await self._save_task(task)
            await self._ack_task(task)
            # Move to dead-letter queue for later analysis
            await self._move_to_dead_letter(task, error)

            # Record failed task metrics
            duration = task.completed_at - (task.started_at or task.created_at)
            record_control_plane_task_completed(task.task_type, "failed", duration)

            logger.error(f"Task {task_id} failed permanently: {error}")

        return True

    async def cancel(self, task_id: str) -> bool:
        """
        Cancel a pending or running task.

        Args:
            task_id: Task to cancel

        Returns:
            True if cancelled, False if not found or already completed
        """
        task = await self.get(task_id)
        if not task:
            return False

        if task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED):
            return False

        task.status = TaskStatus.CANCELLED
        task.completed_at = time.time()

        await self._save_task(task)
        await self._ack_task(task)

        logger.info(f"Task cancelled: {task_id}")
        return True

    async def get(self, task_id: str) -> Optional[Task]:
        """
        Get task by ID.

        Args:
            task_id: Task to retrieve

        Returns:
            Task if found, None otherwise
        """
        if self._redis:
            key = f"{self._key_prefix}{task_id}"
            data = await self._redis.get(key)
            if data:
                return Task.from_dict(json.loads(data))
            return None
        else:
            return self._local_tasks.get(task_id)

    async def list_by_status(
        self,
        status: TaskStatus,
        limit: int = 100,
    ) -> List[Task]:
        """
        List tasks by status.

        Args:
            status: Status to filter by
            limit: Maximum tasks to return

        Returns:
            List of tasks with the given status
        """
        tasks = []

        if self._redis:
            # Use SCAN to find tasks (in production, use a secondary index)
            pattern = f"{self._key_prefix}*"
            count = 0
            async for key in self._redis.scan_iter(match=pattern):
                if count >= limit:
                    break
                data = await self._redis.get(key)
                if data:
                    task = Task.from_dict(json.loads(data))
                    if task.status == status:
                        tasks.append(task)
                        count += 1
        else:
            for task in self._local_tasks.values():
                if task.status == status:
                    tasks.append(task)
                    if len(tasks) >= limit:
                        break

        return tasks

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get scheduler statistics.

        Returns:
            Dict with task counts by status, queue depths, etc.
        """
        status_counts = {s.value: 0 for s in TaskStatus}
        priority_counts = {p.name.lower(): 0 for p in TaskPriority}
        type_counts: Dict[str, int] = {}

        if self._redis:
            pattern = f"{self._key_prefix}*"
            async for key in self._redis.scan_iter(match=pattern):
                data = await self._redis.get(key)
                if data:
                    task = Task.from_dict(json.loads(data))
                    status_counts[task.status.value] += 1
                    priority_counts[task.priority.name.lower()] += 1
                    type_counts[task.task_type] = type_counts.get(task.task_type, 0) + 1
        else:
            for task in self._local_tasks.values():
                status_counts[task.status.value] += 1
                priority_counts[task.priority.name.lower()] += 1
                type_counts[task.task_type] = type_counts.get(task.task_type, 0) + 1

        return {
            "by_status": status_counts,
            "by_priority": priority_counts,
            "by_type": type_counts,
            "total": sum(status_counts.values()),
        }

    async def claim_stale_tasks(self, idle_ms: int = 60000) -> int:
        """
        Reclaim tasks from dead workers.

        Args:
            idle_ms: Minimum idle time to consider a task stale

        Returns:
            Number of tasks reclaimed
        """
        if not self._redis:
            return 0

        reclaimed = 0

        for priority in TaskPriority:
            stream_key = f"{self._stream_prefix}{priority.name.lower()}"
            try:
                # Get pending entries
                pending = await self._redis.xpending_range(
                    stream_key,
                    self._consumer_group,
                    min="-",
                    max="+",
                    count=100,
                )

                for entry in pending:
                    if entry.get("time_since_delivered", 0) > idle_ms:
                        # Claim the entry
                        messages = await self._redis.xclaim(
                            stream_key,
                            self._consumer_group,
                            "recovery-worker",
                            min_idle_time=idle_ms,
                            message_ids=[entry["message_id"]],
                        )

                        for msg_id, msg_data in messages:
                            task_id = msg_data.get("task_id")
                            if task_id:
                                task = await self.get(task_id)
                                if task and task.status == TaskStatus.RUNNING:
                                    await self.fail(
                                        task_id,
                                        "Worker timeout - task reclaimed",
                                        requeue=True,
                                    )
                                    reclaimed += 1

            except Exception as e:
                logger.error(f"Error reclaiming stale tasks for {priority.name}: {e}")

        if reclaimed > 0:
            logger.info(f"Reclaimed {reclaimed} stale tasks")

        return reclaimed

    async def _save_task(self, task: Task) -> None:
        """Save task to Redis or local storage."""
        if self._redis:
            key = f"{self._key_prefix}{task.id}"
            await self._redis.set(
                key,
                json.dumps(task.to_dict()),
                ex=86400,  # 24 hour expiry
            )
        else:
            self._local_tasks[task.id] = task

    async def _enqueue_task(self, task: Task) -> None:
        """Add task to the appropriate queue."""
        if self._redis:
            stream_key = f"{self._stream_prefix}{task.priority.name.lower()}"
            await self._redis.xadd(
                stream_key,
                {"task_id": task.id, "task_type": task.task_type},
            )
        else:
            self._local_queue.append(task)
            # Sort by priority (higher first)
            self._local_queue.sort(key=lambda t: t.priority.value, reverse=True)

    async def _ack_task(self, task: Task) -> None:
        """Acknowledge task completion in the queue.

        Performs XACK to remove the message from the pending entries list (PEL),
        then XDEL to remove the message from the stream entirely.
        """
        if not self._redis:
            return

        # Get the stream key for this task's priority
        stream_key = f"{self._stream_prefix}{task.priority.name.lower()}"

        # Get the message ID stored when we claimed the task
        message_id = task.metadata.get("_stream_message_id")
        if not message_id:
            logger.warning(f"Task {task.id} has no stream message ID for acknowledgment")
            return

        try:
            # XACK removes from pending entries list
            await self._redis.xack(stream_key, self._consumer_group, message_id)

            # XDEL removes the message from stream (keeps stream size bounded)
            await self._redis.xdel(stream_key, message_id)

            logger.debug(f"Acknowledged task {task.id} (message_id={message_id})")
        except Exception as e:
            logger.error(f"Failed to acknowledge task {task.id}: {e}")

    async def _move_to_dead_letter(self, task: Task, reason: str) -> None:
        """Move task to dead-letter queue after exhausting retries.

        Args:
            task: The failed task
            reason: Reason for moving to dead-letter queue
        """
        if not self._redis:
            return

        dlq_key = f"{self._stream_prefix}dead_letter"
        try:
            await self._redis.xadd(
                dlq_key,
                {
                    "task_id": task.id,
                    "task_type": task.task_type,
                    "reason": reason,
                    "failed_at": str(time.time()),
                    "retries": str(task.retries),
                    "original_priority": task.priority.name,
                },
            )
            logger.info(f"Task {task.id} moved to dead-letter queue: {reason}")
        except Exception as e:
            logger.error(f"Failed to move task {task.id} to dead-letter queue: {e}")

    async def _claim_from_redis(
        self,
        worker_id: str,
        capabilities: List[str],
        block_ms: int,
    ) -> Optional[Task]:
        """Claim task from Redis Streams."""
        cap_set = set(capabilities)

        # Try each priority level from highest to lowest
        for priority in sorted(TaskPriority, key=lambda p: p.value, reverse=True):
            stream_key = f"{self._stream_prefix}{priority.name.lower()}"

            try:
                # Read from consumer group
                messages = await self._redis.xreadgroup(
                    groupname=self._consumer_group,
                    consumername=worker_id,
                    streams={stream_key: ">"},
                    count=1,
                    block=block_ms if priority == TaskPriority.LOW else 0,
                )

                if not messages:
                    continue

                for stream, entries in messages:
                    for msg_id, msg_data in entries:
                        task_id = msg_data.get("task_id")
                        if not task_id:
                            continue

                        task = await self.get(task_id)
                        if not task:
                            # Task deleted, ack and skip
                            await self._redis.xack(stream_key, self._consumer_group, msg_id)
                            await self._redis.xdel(stream_key, msg_id)
                            continue

                        # Check capabilities
                        if task.required_capabilities and not task.required_capabilities.issubset(cap_set):
                            # Worker doesn't have required capabilities
                            # XACK to remove from this worker's pending list
                            await self._redis.xack(stream_key, self._consumer_group, msg_id)

                            # Track rejection in metadata for debugging
                            if "rejection_count" not in task.metadata:
                                task.metadata["rejection_count"] = 0
                            task.metadata["rejection_count"] += 1
                            task.metadata["last_rejected_by"] = worker_id
                            task.metadata["last_rejected_reason"] = (
                                f"Missing capabilities: {task.required_capabilities - cap_set}"
                            )

                            # Requeue for another worker to claim
                            # Note: This creates a new stream entry, old one is acked
                            await self._save_task(task)
                            await self._enqueue_task(task)

                            logger.debug(
                                f"Task {task_id} rejected by {worker_id}: "
                                f"needs {task.required_capabilities}, has {cap_set}"
                            )
                            continue

                        # Store message_id for later acknowledgment
                        task.metadata["_stream_message_id"] = msg_id

                        # Claim the task
                        task.status = TaskStatus.RUNNING
                        task.assigned_agent = worker_id
                        task.assigned_at = time.time()
                        task.started_at = time.time()

                        await self._save_task(task)
                        logger.debug(f"Task {task_id} claimed by {worker_id}")

                        return task

            except Exception as e:
                logger.error(f"Error claiming from {priority.name} queue: {e}")

        return None

    def _claim_from_local(
        self,
        worker_id: str,
        capabilities: List[str],
    ) -> Optional[Task]:
        """Claim task from local queue."""
        cap_set = set(capabilities)

        for i, task in enumerate(self._local_queue):
            if task.status != TaskStatus.PENDING:
                continue

            if task.required_capabilities and not task.required_capabilities.issubset(cap_set):
                continue

            # Claim the task
            task.status = TaskStatus.RUNNING
            task.assigned_agent = worker_id
            task.assigned_at = time.time()
            task.started_at = time.time()

            self._local_queue.pop(i)
            return task

        return None
