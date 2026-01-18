"""
Tests for the Control Plane TaskScheduler.

These tests verify task submission, claiming, completion, and lifecycle management.
"""

import asyncio
import time

import pytest

from aragora.control_plane.scheduler import (
    Task,
    TaskPriority,
    TaskScheduler,
    TaskStatus,
)


class TestTask:
    """Tests for Task dataclass."""

    def test_creation(self):
        """Test basic Task creation."""
        task = Task(
            task_type="debate",
            payload={"question": "What is the best approach?"},
        )

        assert task.task_type == "debate"
        assert task.payload == {"question": "What is the best approach?"}
        assert task.status == TaskStatus.PENDING
        assert task.priority == TaskPriority.NORMAL
        assert task.id is not None

    def test_creation_with_options(self):
        """Test Task creation with all options."""
        task = Task(
            task_type="code",
            payload={"task": "implement feature"},
            required_capabilities={"code", "python"},
            priority=TaskPriority.HIGH,
            timeout_seconds=600.0,
            max_retries=5,
            metadata={"project": "aragora"},
        )

        assert task.required_capabilities == {"code", "python"}
        assert task.priority == TaskPriority.HIGH
        assert task.timeout_seconds == 600.0
        assert task.max_retries == 5
        assert task.metadata == {"project": "aragora"}

    def test_is_timed_out_not_started(self):
        """Test timeout check when task not started."""
        task = Task(
            task_type="test",
            payload={},
            timeout_seconds=1.0,
        )

        assert not task.is_timed_out()

    def test_is_timed_out_within_limit(self):
        """Test timeout check within time limit."""
        task = Task(
            task_type="test",
            payload={},
            timeout_seconds=10.0,
        )
        task.started_at = time.time()

        assert not task.is_timed_out()

    def test_is_timed_out_expired(self):
        """Test timeout check when expired."""
        task = Task(
            task_type="test",
            payload={},
            timeout_seconds=1.0,
        )
        task.started_at = time.time() - 2  # Started 2 seconds ago

        assert task.is_timed_out()

    def test_should_retry_fresh(self):
        """Test retry check with fresh task."""
        task = Task(
            task_type="test",
            payload={},
            max_retries=3,
        )

        assert task.should_retry()

    def test_should_retry_exhausted(self):
        """Test retry check when retries exhausted."""
        task = Task(
            task_type="test",
            payload={},
            max_retries=3,
            retries=3,
        )

        assert not task.should_retry()

    def test_serialization(self):
        """Test to_dict and from_dict."""
        task = Task(
            task_type="debate",
            payload={"topic": "AI safety"},
            required_capabilities={"debate", "analysis"},
            priority=TaskPriority.HIGH,
            metadata={"source": "api"},
        )

        data = task.to_dict()
        restored = Task.from_dict(data)

        assert restored.id == task.id
        assert restored.task_type == task.task_type
        assert restored.payload == task.payload
        assert restored.required_capabilities == task.required_capabilities
        assert restored.priority == task.priority
        assert restored.metadata == task.metadata


class TestTaskScheduler:
    """Tests for TaskScheduler with in-memory fallback."""

    @pytest.fixture
    def scheduler(self):
        """Create a scheduler using in-memory fallback."""
        return TaskScheduler(redis_url="redis://nonexistent:6379")

    @pytest.mark.asyncio
    async def test_submit_task(self, scheduler):
        """Test task submission."""
        task_id = await scheduler.submit(
            task_type="debate",
            payload={"question": "What is the best approach?"},
            required_capabilities=["debate"],
        )

        assert task_id is not None
        assert len(task_id) > 0

    @pytest.mark.asyncio
    async def test_get_task(self, scheduler):
        """Test getting task by ID."""
        task_id = await scheduler.submit(
            task_type="test",
            payload={"data": "value"},
        )

        task = await scheduler.get(task_id)
        assert task is not None
        assert task.id == task_id
        assert task.task_type == "test"
        assert task.payload == {"data": "value"}

    @pytest.mark.asyncio
    async def test_get_nonexistent_task(self, scheduler):
        """Test getting non-existent task."""
        task = await scheduler.get("nonexistent-id")
        assert task is None

    @pytest.mark.asyncio
    async def test_claim_task(self, scheduler):
        """Test claiming a task."""
        await scheduler.submit(
            task_type="code",
            payload={"task": "implement"},
            required_capabilities=["code"],
        )

        task = await scheduler.claim(
            worker_id="worker-1",
            capabilities=["code"],
        )

        assert task is not None
        assert task.task_type == "code"
        # Task status is RUNNING when claimed (not ASSIGNED)
        assert task.status == TaskStatus.RUNNING
        assert task.assigned_agent == "worker-1"

    @pytest.mark.asyncio
    async def test_claim_no_matching_task(self, scheduler):
        """Test claiming when no matching task exists."""
        await scheduler.submit(
            task_type="code",
            payload={},
            required_capabilities=["python"],
        )

        # Try to claim with different capability
        task = await scheduler.claim(
            worker_id="worker-1",
            capabilities=["java"],
            block_ms=100,
        )

        assert task is None

    @pytest.mark.asyncio
    async def test_complete_task(self, scheduler):
        """Test task completion."""
        task_id = await scheduler.submit(
            task_type="test",
            payload={},
        )

        # Claim it first
        await scheduler.claim(worker_id="worker-1", capabilities=[])

        result = {"conclusion": "success"}
        success = await scheduler.complete(task_id, result=result)
        assert success

        task = await scheduler.get(task_id)
        assert task.status == TaskStatus.COMPLETED
        assert task.result == result
        assert task.completed_at is not None

    @pytest.mark.asyncio
    async def test_fail_task(self, scheduler):
        """Test task failure."""
        task_id = await scheduler.submit(
            task_type="test",
            payload={},
            max_retries=0,
        )

        success = await scheduler.fail(task_id, error="Something went wrong")
        assert success

        task = await scheduler.get(task_id)
        assert task.status == TaskStatus.FAILED
        assert task.error == "Something went wrong"

    @pytest.mark.asyncio
    async def test_fail_task_with_retry(self, scheduler):
        """Test task failure with requeue."""
        task_id = await scheduler.submit(
            task_type="test",
            payload={},
            max_retries=3,
        )

        success = await scheduler.fail(task_id, error="Temporary error", requeue=True)
        assert success

        task = await scheduler.get(task_id)
        # Should be requeued, not failed
        assert task.status == TaskStatus.PENDING
        assert task.retries == 1

    @pytest.mark.asyncio
    async def test_cancel_task(self, scheduler):
        """Test task cancellation."""
        task_id = await scheduler.submit(
            task_type="test",
            payload={},
        )

        success = await scheduler.cancel(task_id)
        assert success

        task = await scheduler.get(task_id)
        assert task.status == TaskStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_completed_task(self, scheduler):
        """Test that completed task cannot be cancelled."""
        task_id = await scheduler.submit(
            task_type="test",
            payload={},
        )

        await scheduler.complete(task_id)

        success = await scheduler.cancel(task_id)
        assert not success

    @pytest.mark.asyncio
    async def test_priority_ordering(self, scheduler):
        """Test that higher priority tasks are claimed first."""
        # Submit low priority first
        await scheduler.submit(
            task_type="low",
            payload={},
            priority=TaskPriority.LOW,
        )

        # Then submit high priority
        await scheduler.submit(
            task_type="high",
            payload={},
            priority=TaskPriority.HIGH,
        )

        # Should claim high priority first
        task = await scheduler.claim(worker_id="worker-1", capabilities=[])
        assert task.task_type == "high"

        # Then low priority
        task = await scheduler.claim(worker_id="worker-1", capabilities=[])
        assert task.task_type == "low"

    @pytest.mark.asyncio
    async def test_capability_matching(self, scheduler):
        """Test capability-based task matching."""
        await scheduler.submit(
            task_type="python-code",
            payload={},
            required_capabilities=["code", "python"],
        )

        await scheduler.submit(
            task_type="java-code",
            payload={},
            required_capabilities=["code", "java"],
        )

        # Claim with python capability
        task = await scheduler.claim(
            worker_id="python-dev",
            capabilities=["code", "python"],
        )
        assert task.task_type == "python-code"

    @pytest.mark.asyncio
    async def test_get_stats(self, scheduler):
        """Test statistics retrieval."""
        await scheduler.submit(task_type="t1", payload={})
        await scheduler.submit(task_type="t2", payload={})
        task_id = await scheduler.submit(task_type="t3", payload={})
        await scheduler.complete(task_id)

        stats = await scheduler.get_stats()

        assert stats["total"] >= 3
        assert "by_status" in stats
        assert "by_priority" in stats

    @pytest.mark.asyncio
    async def test_multiple_tasks_submitted(self, scheduler):
        """Test submitting multiple tasks."""
        await scheduler.submit(task_type="t1", payload={})
        await scheduler.submit(task_type="t2", payload={})
        task_id = await scheduler.submit(task_type="t3", payload={})
        await scheduler.complete(task_id)

        # Verify tasks can be claimed
        task1 = await scheduler.claim(worker_id="worker-1", capabilities=[])
        task2 = await scheduler.claim(worker_id="worker-2", capabilities=[])

        assert task1 is not None
        assert task2 is not None
        assert task1.task_type in ("t1", "t2")
        assert task2.task_type in ("t1", "t2")
