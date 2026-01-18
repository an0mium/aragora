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


class TestTaskStatusStateMachine:
    """Tests for task status state transitions."""

    @pytest.fixture
    def scheduler(self):
        """Create a scheduler using in-memory fallback."""
        return TaskScheduler(redis_url="redis://nonexistent:6379")

    @pytest.mark.asyncio
    async def test_pending_to_running(self, scheduler):
        """Test transition from PENDING to RUNNING on claim."""
        task_id = await scheduler.submit(task_type="test", payload={})

        task_before = await scheduler.get(task_id)
        assert task_before.status == TaskStatus.PENDING

        claimed = await scheduler.claim(worker_id="w1", capabilities=[])
        assert claimed is not None
        assert claimed.status == TaskStatus.RUNNING

        task_after = await scheduler.get(task_id)
        assert task_after.status == TaskStatus.RUNNING

    @pytest.mark.asyncio
    async def test_running_to_completed(self, scheduler):
        """Test transition from RUNNING to COMPLETED."""
        task_id = await scheduler.submit(task_type="test", payload={})
        await scheduler.claim(worker_id="w1", capabilities=[])

        task_before = await scheduler.get(task_id)
        assert task_before.status == TaskStatus.RUNNING

        await scheduler.complete(task_id, result={"data": "success"})

        task_after = await scheduler.get(task_id)
        assert task_after.status == TaskStatus.COMPLETED
        assert task_after.result == {"data": "success"}

    @pytest.mark.asyncio
    async def test_running_to_failed(self, scheduler):
        """Test transition from RUNNING to FAILED."""
        task_id = await scheduler.submit(task_type="test", payload={}, max_retries=0)
        await scheduler.claim(worker_id="w1", capabilities=[])

        task_before = await scheduler.get(task_id)
        assert task_before.status == TaskStatus.RUNNING

        await scheduler.fail(task_id, error="Error occurred", requeue=False)

        task_after = await scheduler.get(task_id)
        assert task_after.status == TaskStatus.FAILED
        assert task_after.error == "Error occurred"

    @pytest.mark.asyncio
    async def test_running_to_pending_on_retry(self, scheduler):
        """Test transition from RUNNING to PENDING on retry."""
        task_id = await scheduler.submit(task_type="test", payload={}, max_retries=3)
        await scheduler.claim(worker_id="w1", capabilities=[])

        await scheduler.fail(task_id, error="Temporary error", requeue=True)

        task = await scheduler.get(task_id)
        assert task.status == TaskStatus.PENDING
        assert task.retries == 1
        assert task.assigned_agent is None

    @pytest.mark.asyncio
    async def test_pending_to_cancelled(self, scheduler):
        """Test cancelling a pending task."""
        task_id = await scheduler.submit(task_type="test", payload={})

        await scheduler.cancel(task_id)

        task = await scheduler.get(task_id)
        assert task.status == TaskStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_running_to_cancelled(self, scheduler):
        """Test cancelling a running task."""
        task_id = await scheduler.submit(task_type="test", payload={})
        await scheduler.claim(worker_id="w1", capabilities=[])

        await scheduler.cancel(task_id)

        task = await scheduler.get(task_id)
        assert task.status == TaskStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_cannot_cancel_completed(self, scheduler):
        """Test that completed tasks cannot be cancelled."""
        task_id = await scheduler.submit(task_type="test", payload={})
        await scheduler.claim(worker_id="w1", capabilities=[])
        await scheduler.complete(task_id)

        result = await scheduler.cancel(task_id)
        assert not result

        task = await scheduler.get(task_id)
        assert task.status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_cannot_cancel_failed(self, scheduler):
        """Test that failed tasks cannot be cancelled."""
        task_id = await scheduler.submit(task_type="test", payload={}, max_retries=0)
        await scheduler.fail(task_id, error="Error")

        result = await scheduler.cancel(task_id)
        assert not result

        task = await scheduler.get(task_id)
        assert task.status == TaskStatus.FAILED


class TestPriorityQueueOrdering:
    """Extended tests for priority queue ordering."""

    @pytest.fixture
    def scheduler(self):
        """Create a scheduler using in-memory fallback."""
        return TaskScheduler(redis_url="redis://nonexistent:6379")

    @pytest.mark.asyncio
    async def test_urgent_priority_first(self, scheduler):
        """Test that URGENT priority tasks are claimed before all others."""
        await scheduler.submit(task_type="low", payload={}, priority=TaskPriority.LOW)
        await scheduler.submit(task_type="normal", payload={}, priority=TaskPriority.NORMAL)
        await scheduler.submit(task_type="high", payload={}, priority=TaskPriority.HIGH)
        await scheduler.submit(task_type="urgent", payload={}, priority=TaskPriority.URGENT)

        task = await scheduler.claim(worker_id="w1", capabilities=[])
        assert task.task_type == "urgent"

    @pytest.mark.asyncio
    async def test_all_priority_levels(self, scheduler):
        """Test claiming tasks in correct priority order."""
        await scheduler.submit(task_type="low", payload={}, priority=TaskPriority.LOW)
        await scheduler.submit(task_type="normal", payload={}, priority=TaskPriority.NORMAL)
        await scheduler.submit(task_type="high", payload={}, priority=TaskPriority.HIGH)
        await scheduler.submit(task_type="urgent", payload={}, priority=TaskPriority.URGENT)

        order = []
        for _ in range(4):
            task = await scheduler.claim(worker_id="w1", capabilities=[])
            if task:
                order.append(task.task_type)

        assert order == ["urgent", "high", "normal", "low"]

    @pytest.mark.asyncio
    async def test_fifo_within_same_priority(self, scheduler):
        """Test FIFO ordering within the same priority level."""
        await scheduler.submit(task_type="first", payload={}, priority=TaskPriority.NORMAL)
        await scheduler.submit(task_type="second", payload={}, priority=TaskPriority.NORMAL)
        await scheduler.submit(task_type="third", payload={}, priority=TaskPriority.NORMAL)

        task1 = await scheduler.claim(worker_id="w1", capabilities=[])
        task2 = await scheduler.claim(worker_id="w1", capabilities=[])
        task3 = await scheduler.claim(worker_id="w1", capabilities=[])

        assert task1.task_type == "first"
        assert task2.task_type == "second"
        assert task3.task_type == "third"

    @pytest.mark.asyncio
    async def test_high_priority_interleaved(self, scheduler):
        """Test high priority tasks claimed first when added after lower priority."""
        await scheduler.submit(task_type="low1", payload={}, priority=TaskPriority.LOW)
        await scheduler.submit(task_type="low2", payload={}, priority=TaskPriority.LOW)

        # Now add a high priority task
        await scheduler.submit(task_type="high", payload={}, priority=TaskPriority.HIGH)

        # High priority should come first
        task = await scheduler.claim(worker_id="w1", capabilities=[])
        assert task.task_type == "high"


class TestTimeoutAndRetry:
    """Tests for timeout detection and retry logic."""

    @pytest.fixture
    def scheduler(self):
        """Create a scheduler using in-memory fallback."""
        return TaskScheduler(redis_url="redis://nonexistent:6379")

    def test_task_timeout_detection(self):
        """Test Task.is_timed_out() method."""
        task = Task(
            task_type="test",
            payload={},
            timeout_seconds=1.0,
        )
        task.started_at = time.time() - 2  # Started 2 seconds ago

        assert task.is_timed_out()

    def test_task_not_timed_out(self):
        """Test Task.is_timed_out() returns False when within limit."""
        task = Task(
            task_type="test",
            payload={},
            timeout_seconds=60.0,
        )
        task.started_at = time.time()

        assert not task.is_timed_out()

    def test_task_timeout_not_started(self):
        """Test Task.is_timed_out() returns False when not started."""
        task = Task(
            task_type="test",
            payload={},
            timeout_seconds=1.0,
        )

        assert not task.is_timed_out()

    @pytest.mark.asyncio
    async def test_retry_increments_count(self, scheduler):
        """Test that retries increment the retry counter."""
        task_id = await scheduler.submit(
            task_type="test",
            payload={},
            max_retries=5,
        )

        # Fail and retry multiple times
        for i in range(3):
            await scheduler.claim(worker_id="w1", capabilities=[])
            await scheduler.fail(task_id, error=f"Error {i}", requeue=True)

            task = await scheduler.get(task_id)
            assert task.retries == i + 1
            assert task.status == TaskStatus.PENDING

    @pytest.mark.asyncio
    async def test_max_retries_exhausted(self, scheduler):
        """Test task fails permanently when max retries exhausted."""
        task_id = await scheduler.submit(
            task_type="test",
            payload={},
            max_retries=2,
        )

        # Exhaust retries
        for _ in range(2):
            await scheduler.claim(worker_id="w1", capabilities=[])
            await scheduler.fail(task_id, error="Error", requeue=True)

        # Now claim and fail one more time - should fail permanently
        await scheduler.claim(worker_id="w1", capabilities=[])
        await scheduler.fail(task_id, error="Final error", requeue=True)

        task = await scheduler.get(task_id)
        assert task.status == TaskStatus.FAILED

    @pytest.mark.asyncio
    async def test_requeue_clears_assignment(self, scheduler):
        """Test that requeuing clears agent assignment."""
        task_id = await scheduler.submit(
            task_type="test",
            payload={},
            max_retries=3,
        )

        task = await scheduler.claim(worker_id="worker-1", capabilities=[])
        assert task.assigned_agent == "worker-1"

        await scheduler.fail(task_id, error="Error", requeue=True)

        task = await scheduler.get(task_id)
        assert task.assigned_agent is None
        assert task.assigned_at is None
        assert task.started_at is None

    def test_should_retry_with_retries_left(self):
        """Test should_retry returns True when retries available."""
        task = Task(
            task_type="test",
            payload={},
            max_retries=3,
            retries=2,
        )
        assert task.should_retry()

    def test_should_retry_no_retries_left(self):
        """Test should_retry returns False when no retries left."""
        task = Task(
            task_type="test",
            payload={},
            max_retries=3,
            retries=3,
        )
        assert not task.should_retry()


class TestCapabilityMatching:
    """Extended tests for capability-based task matching."""

    @pytest.fixture
    def scheduler(self):
        """Create a scheduler using in-memory fallback."""
        return TaskScheduler(redis_url="redis://nonexistent:6379")

    @pytest.mark.asyncio
    async def test_exact_capability_match(self, scheduler):
        """Test task with exact capability requirement."""
        await scheduler.submit(
            task_type="code",
            payload={},
            required_capabilities=["python", "testing"],
        )

        task = await scheduler.claim(
            worker_id="w1",
            capabilities=["python", "testing"],
        )

        assert task is not None
        assert task.task_type == "code"

    @pytest.mark.asyncio
    async def test_superset_capabilities_match(self, scheduler):
        """Test worker with superset of required capabilities."""
        await scheduler.submit(
            task_type="code",
            payload={},
            required_capabilities=["python"],
        )

        task = await scheduler.claim(
            worker_id="w1",
            capabilities=["python", "java", "testing"],
        )

        assert task is not None

    @pytest.mark.asyncio
    async def test_missing_capability_no_match(self, scheduler):
        """Test worker missing required capability."""
        await scheduler.submit(
            task_type="code",
            payload={},
            required_capabilities=["python", "testing"],
        )

        task = await scheduler.claim(
            worker_id="w1",
            capabilities=["python"],  # Missing 'testing'
        )

        assert task is None

    @pytest.mark.asyncio
    async def test_no_capabilities_required(self, scheduler):
        """Test task with no capability requirements."""
        await scheduler.submit(
            task_type="simple",
            payload={},
            required_capabilities=[],
        )

        task = await scheduler.claim(
            worker_id="w1",
            capabilities=[],
        )

        assert task is not None

    @pytest.mark.asyncio
    async def test_multiple_tasks_different_capabilities(self, scheduler):
        """Test claiming correct task from multiple with different requirements."""
        await scheduler.submit(
            task_type="java-task",
            payload={},
            required_capabilities=["java"],
        )
        await scheduler.submit(
            task_type="python-task",
            payload={},
            required_capabilities=["python"],
        )

        # Worker with only python
        task = await scheduler.claim(
            worker_id="python-dev",
            capabilities=["python"],
        )

        assert task is not None
        assert task.task_type == "python-task"


class TestListByStatus:
    """Tests for listing tasks by status."""

    @pytest.fixture
    def scheduler(self):
        """Create a scheduler using in-memory fallback."""
        return TaskScheduler(redis_url="redis://nonexistent:6379")

    @pytest.mark.asyncio
    async def test_list_pending_tasks(self, scheduler):
        """Test listing pending tasks."""
        await scheduler.submit(task_type="t1", payload={})
        await scheduler.submit(task_type="t2", payload={})

        tasks = await scheduler.list_by_status(TaskStatus.PENDING)
        assert len(tasks) == 2
        assert all(t.status == TaskStatus.PENDING for t in tasks)

    @pytest.mark.asyncio
    async def test_list_running_tasks(self, scheduler):
        """Test listing running tasks."""
        await scheduler.submit(task_type="t1", payload={})
        await scheduler.submit(task_type="t2", payload={})

        await scheduler.claim(worker_id="w1", capabilities=[])

        pending = await scheduler.list_by_status(TaskStatus.PENDING)
        running = await scheduler.list_by_status(TaskStatus.RUNNING)

        assert len(pending) == 1
        assert len(running) == 1

    @pytest.mark.asyncio
    async def test_list_completed_tasks(self, scheduler):
        """Test listing completed tasks."""
        task_id1 = await scheduler.submit(task_type="t1", payload={})
        task_id2 = await scheduler.submit(task_type="t2", payload={})

        await scheduler.claim(worker_id="w1", capabilities=[])
        await scheduler.complete(task_id1)

        completed = await scheduler.list_by_status(TaskStatus.COMPLETED)
        assert len(completed) == 1
        assert completed[0].id == task_id1

    @pytest.mark.asyncio
    async def test_list_with_limit(self, scheduler):
        """Test listing with limit."""
        for i in range(10):
            await scheduler.submit(task_type=f"t{i}", payload={})

        tasks = await scheduler.list_by_status(TaskStatus.PENDING, limit=5)
        assert len(tasks) == 5

    @pytest.mark.asyncio
    async def test_list_empty_status(self, scheduler):
        """Test listing with no tasks of that status."""
        await scheduler.submit(task_type="t1", payload={})

        tasks = await scheduler.list_by_status(TaskStatus.COMPLETED)
        assert len(tasks) == 0


class TestSchedulerStats:
    """Tests for scheduler statistics."""

    @pytest.fixture
    def scheduler(self):
        """Create a scheduler using in-memory fallback."""
        return TaskScheduler(redis_url="redis://nonexistent:6379")

    @pytest.mark.asyncio
    async def test_stats_by_status(self, scheduler):
        """Test stats include correct counts by status."""
        await scheduler.submit(task_type="t1", payload={})
        await scheduler.submit(task_type="t2", payload={})
        task_id = await scheduler.submit(task_type="t3", payload={})
        await scheduler.claim(worker_id="w1", capabilities=[])
        await scheduler.complete(task_id)

        stats = await scheduler.get_stats()

        # t1 is running (claimed), t2 is pending, t3 is completed
        assert stats["by_status"]["pending"] == 1
        assert stats["by_status"]["running"] == 1
        assert stats["by_status"]["completed"] == 1

    @pytest.mark.asyncio
    async def test_stats_by_priority(self, scheduler):
        """Test stats include counts by priority."""
        await scheduler.submit(task_type="t1", payload={}, priority=TaskPriority.LOW)
        await scheduler.submit(task_type="t2", payload={}, priority=TaskPriority.HIGH)
        await scheduler.submit(task_type="t3", payload={}, priority=TaskPriority.HIGH)

        stats = await scheduler.get_stats()

        assert stats["by_priority"]["low"] == 1
        assert stats["by_priority"]["high"] == 2

    @pytest.mark.asyncio
    async def test_stats_by_type(self, scheduler):
        """Test stats include counts by task type."""
        await scheduler.submit(task_type="debate", payload={})
        await scheduler.submit(task_type="debate", payload={})
        await scheduler.submit(task_type="code", payload={})

        stats = await scheduler.get_stats()

        assert stats["by_type"]["debate"] == 2
        assert stats["by_type"]["code"] == 1

    @pytest.mark.asyncio
    async def test_stats_total(self, scheduler):
        """Test stats include correct total."""
        await scheduler.submit(task_type="t1", payload={})
        await scheduler.submit(task_type="t2", payload={})
        await scheduler.submit(task_type="t3", payload={})

        stats = await scheduler.get_stats()
        assert stats["total"] == 3


class TestTaskSerialization:
    """Tests for task serialization and deserialization."""

    def test_serialize_minimal_task(self):
        """Test serializing task with minimal fields."""
        task = Task(task_type="test", payload={"key": "value"})

        data = task.to_dict()

        assert data["task_type"] == "test"
        assert data["payload"] == {"key": "value"}
        assert data["status"] == "pending"
        assert data["priority"] == 50
        assert data["id"] is not None

    def test_serialize_full_task(self):
        """Test serializing task with all fields."""
        task = Task(
            task_type="complex",
            payload={"data": [1, 2, 3]},
            required_capabilities={"cap1", "cap2"},
            priority=TaskPriority.URGENT,
            timeout_seconds=600.0,
            max_retries=5,
            metadata={"source": "api", "version": 2},
        )
        task.status = TaskStatus.RUNNING
        task.assigned_agent = "worker-1"
        task.assigned_at = 1234567890.0
        task.started_at = 1234567891.0

        data = task.to_dict()

        assert data["task_type"] == "complex"
        assert set(data["required_capabilities"]) == {"cap1", "cap2"}
        assert data["priority"] == 100
        assert data["timeout_seconds"] == 600.0
        assert data["max_retries"] == 5
        assert data["status"] == "running"
        assert data["assigned_agent"] == "worker-1"

    def test_deserialize_task(self):
        """Test deserializing task from dict."""
        data = {
            "id": "task-123",
            "task_type": "debate",
            "payload": {"question": "test"},
            "required_capabilities": ["debate", "analysis"],
            "status": "running",
            "priority": 75,
            "created_at": 1234567890.0,
            "assigned_agent": "worker-1",
            "timeout_seconds": 300.0,
            "max_retries": 3,
            "retries": 1,
            "metadata": {"key": "value"},
        }

        task = Task.from_dict(data)

        assert task.id == "task-123"
        assert task.task_type == "debate"
        assert task.required_capabilities == {"debate", "analysis"}
        assert task.status == TaskStatus.RUNNING
        assert task.priority == TaskPriority.HIGH
        assert task.assigned_agent == "worker-1"
        assert task.retries == 1

    def test_roundtrip_serialization(self):
        """Test serialization roundtrip preserves data."""
        original = Task(
            task_type="roundtrip",
            payload={"nested": {"data": [1, 2, 3]}},
            required_capabilities={"a", "b", "c"},
            priority=TaskPriority.HIGH,
            metadata={"complex": {"nested": "value"}},
        )

        data = original.to_dict()
        restored = Task.from_dict(data)

        assert restored.id == original.id
        assert restored.task_type == original.task_type
        assert restored.payload == original.payload
        assert restored.required_capabilities == original.required_capabilities
        assert restored.priority == original.priority
        assert restored.metadata == original.metadata


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture
    def scheduler(self):
        """Create a scheduler using in-memory fallback."""
        return TaskScheduler(redis_url="redis://nonexistent:6379")

    @pytest.mark.asyncio
    async def test_complete_nonexistent_task(self, scheduler):
        """Test completing a non-existent task."""
        result = await scheduler.complete("nonexistent-id")
        assert not result

    @pytest.mark.asyncio
    async def test_fail_nonexistent_task(self, scheduler):
        """Test failing a non-existent task."""
        result = await scheduler.fail("nonexistent-id", error="Error")
        assert not result

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_task(self, scheduler):
        """Test cancelling a non-existent task."""
        result = await scheduler.cancel("nonexistent-id")
        assert not result

    @pytest.mark.asyncio
    async def test_claim_from_empty_queue(self, scheduler):
        """Test claiming from an empty queue."""
        task = await scheduler.claim(worker_id="w1", capabilities=[])
        assert task is None

    @pytest.mark.asyncio
    async def test_empty_payload(self, scheduler):
        """Test task with empty payload."""
        task_id = await scheduler.submit(task_type="empty", payload={})

        task = await scheduler.get(task_id)
        assert task.payload == {}

    @pytest.mark.asyncio
    async def test_large_payload(self, scheduler):
        """Test task with large payload."""
        large_data = {"items": list(range(1000))}
        task_id = await scheduler.submit(task_type="large", payload=large_data)

        task = await scheduler.get(task_id)
        assert task.payload == large_data

    @pytest.mark.asyncio
    async def test_zero_retries(self, scheduler):
        """Test task with zero max retries."""
        task_id = await scheduler.submit(
            task_type="no-retry",
            payload={},
            max_retries=0,
        )

        await scheduler.fail(task_id, error="First failure", requeue=True)

        task = await scheduler.get(task_id)
        assert task.status == TaskStatus.FAILED

    @pytest.mark.asyncio
    async def test_zero_timeout(self, scheduler):
        """Test task with very short timeout."""
        task = Task(
            task_type="fast",
            payload={},
            timeout_seconds=0.001,
        )
        task.started_at = time.time() - 0.01  # Started slightly in the past

        assert task.is_timed_out()

    @pytest.mark.asyncio
    async def test_claim_same_task_twice(self, scheduler):
        """Test that a task can only be claimed once."""
        await scheduler.submit(task_type="single", payload={})

        task1 = await scheduler.claim(worker_id="w1", capabilities=[])
        task2 = await scheduler.claim(worker_id="w2", capabilities=[])

        assert task1 is not None
        assert task2 is None

    @pytest.mark.asyncio
    async def test_metadata_preserved(self, scheduler):
        """Test that task metadata is preserved through lifecycle."""
        task_id = await scheduler.submit(
            task_type="meta",
            payload={},
            metadata={"custom": "data", "count": 42},
        )

        await scheduler.claim(worker_id="w1", capabilities=[])
        await scheduler.complete(task_id, result={"done": True})

        task = await scheduler.get(task_id)
        assert task.metadata["custom"] == "data"
        assert task.metadata["count"] == 42
