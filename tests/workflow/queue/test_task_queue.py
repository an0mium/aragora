"""Tests for TaskQueue and related classes."""

import asyncio
import pytest

from aragora.workflow.queue import (
    TaskQueue,
    TaskQueueConfig,
    TaskStatus,
    TaskPriority,
    WorkflowTask,
)
from aragora.workflow.queue.task import TaskResult


class TestWorkflowTask:
    """Tests for WorkflowTask dataclass."""

    def test_create_task(self):
        """Test creating a task with factory method."""
        task = WorkflowTask.create(
            workflow_id="wf_001",
            step_id="step_1",
        )
        assert task.id.startswith("task_")
        assert task.workflow_id == "wf_001"
        assert task.step_id == "step_1"
        assert task.status == TaskStatus.PENDING
        assert task.priority == TaskPriority.NORMAL

    def test_create_task_with_dependencies(self):
        """Test creating a task with dependencies."""
        task = WorkflowTask.create(
            workflow_id="wf_001",
            step_id="step_2",
            depends_on=["task_001", "task_002"],
        )
        assert task.depends_on == ["task_001", "task_002"]
        assert task.has_dependencies

    def test_create_task_with_priority(self):
        """Test creating a task with custom priority."""
        task = WorkflowTask.create(
            workflow_id="wf_001",
            step_id="step_1",
            priority=TaskPriority.HIGH,
        )
        assert task.priority == TaskPriority.HIGH

    def test_task_is_runnable(self):
        """Test is_runnable property."""
        task = WorkflowTask.create(workflow_id="wf_001", step_id="step_1")
        assert not task.is_runnable  # PENDING is not runnable

        task.mark_ready()
        assert task.is_runnable  # READY is runnable

        task.mark_running(executor_id="exec_1")
        assert not task.is_runnable  # RUNNING is not runnable

    def test_task_is_terminal(self):
        """Test is_terminal property."""
        task = WorkflowTask.create(workflow_id="wf_001", step_id="step_1")
        assert not task.is_terminal

        task.mark_ready()
        task.mark_running(executor_id="exec_1")
        task.mark_completed(TaskResult(success=True))
        assert task.is_terminal

    def test_task_lifecycle(self):
        """Test task state transitions."""
        task = WorkflowTask.create(workflow_id="wf_001", step_id="step_1")
        assert task.status == TaskStatus.PENDING

        task.mark_ready()
        assert task.status == TaskStatus.READY
        assert task.queued_at is not None

        task.mark_running(executor_id="exec_1")
        assert task.status == TaskStatus.RUNNING
        assert task.started_at is not None
        assert task.executor_id == "exec_1"

        result = TaskResult(success=True, output={"data": "test"})
        task.mark_completed(result)
        assert task.status == TaskStatus.COMPLETED
        assert task.completed_at is not None
        assert task.result == result

    def test_task_retry(self):
        """Test task retry mechanism."""
        task = WorkflowTask.create(
            workflow_id="wf_001",
            step_id="step_1",
            max_retries=2,
        )
        task.mark_ready()
        task.mark_running(executor_id="exec_1")
        task.mark_failed("Test error")

        assert task.can_retry
        assert task.schedule_retry()
        assert task.status == TaskStatus.RETRY
        assert task.retry_count == 1

        # Retry again
        task.mark_running(executor_id="exec_1")
        task.mark_failed("Test error 2")
        assert task.schedule_retry()
        assert task.retry_count == 2

        # No more retries
        task.mark_running(executor_id="exec_1")
        task.mark_failed("Test error 3")
        assert not task.can_retry
        assert not task.schedule_retry()

    def test_task_to_dict(self):
        """Test serialization to dictionary."""
        task = WorkflowTask.create(
            workflow_id="wf_001",
            step_id="step_1",
            step_config={"param": "value"},
        )
        data = task.to_dict()

        assert data["workflow_id"] == "wf_001"
        assert data["step_id"] == "step_1"
        assert data["step_config"] == {"param": "value"}
        assert data["status"] == "pending"

    def test_task_from_dict(self):
        """Test deserialization from dictionary."""
        task = WorkflowTask.create(
            workflow_id="wf_001",
            step_id="step_1",
        )
        data = task.to_dict()
        restored = WorkflowTask.from_dict(data)

        assert restored.id == task.id
        assert restored.workflow_id == task.workflow_id
        assert restored.step_id == task.step_id


class TestTaskQueue:
    """Tests for TaskQueue."""

    @pytest.fixture
    def queue(self):
        """Create a task queue."""
        config = TaskQueueConfig(max_concurrent=2)
        return TaskQueue(config=config)

    @pytest.mark.asyncio
    async def test_start_stop(self, queue):
        """Test starting and stopping the queue."""
        await queue.start()
        assert queue._started

        await queue.stop()
        assert not queue._started

    @pytest.mark.asyncio
    async def test_enqueue_task(self, queue):
        """Test enqueueing a task."""
        await queue.start()
        try:
            task = WorkflowTask.create(workflow_id="wf_001", step_id="step_1")
            task_id = await queue.enqueue(task)

            assert task_id == task.id
            assert task.status == TaskStatus.READY  # No deps, so ready
            assert task_id in queue._tasks
        finally:
            await queue.stop(drain=False)

    @pytest.mark.asyncio
    async def test_enqueue_task_with_dependencies(self, queue):
        """Test enqueueing a task with dependencies."""
        await queue.start()
        try:
            task = WorkflowTask.create(
                workflow_id="wf_001",
                step_id="step_2",
                depends_on=["task_001"],
            )
            await queue.enqueue(task)

            # Should stay pending (dependency not resolved)
            assert task.status == TaskStatus.PENDING
        finally:
            await queue.stop(drain=False)

    @pytest.mark.asyncio
    async def test_priority_ordering(self, queue):
        """Test that tasks are ordered by priority."""
        await queue.start()
        try:
            low_task = WorkflowTask.create(
                workflow_id="wf_001",
                step_id="low",
                priority=TaskPriority.LOW,
            )
            high_task = WorkflowTask.create(
                workflow_id="wf_001",
                step_id="high",
                priority=TaskPriority.HIGH,
            )

            await queue.enqueue(low_task)
            await queue.enqueue(high_task)

            # High priority should be first in queue
            assert queue._ready_queue[0] == high_task.id
        finally:
            await queue.stop(drain=False)

    @pytest.mark.asyncio
    async def test_get_task(self, queue):
        """Test getting a task by ID."""
        await queue.start()
        try:
            task = WorkflowTask.create(workflow_id="wf_001", step_id="step_1")
            await queue.enqueue(task)

            retrieved = queue.get_task(task.id)
            assert retrieved == task

            # Non-existent task
            assert queue.get_task("nonexistent") is None
        finally:
            await queue.stop(drain=False)

    @pytest.mark.asyncio
    async def test_get_workflow_tasks(self, queue):
        """Test getting all tasks for a workflow."""
        await queue.start()
        try:
            task1 = WorkflowTask.create(workflow_id="wf_001", step_id="step_1")
            task2 = WorkflowTask.create(workflow_id="wf_001", step_id="step_2")
            task3 = WorkflowTask.create(workflow_id="wf_002", step_id="step_1")

            await queue.enqueue(task1)
            await queue.enqueue(task2)
            await queue.enqueue(task3)

            wf1_tasks = queue.get_workflow_tasks("wf_001")
            assert len(wf1_tasks) == 2
            assert task1 in wf1_tasks
            assert task2 in wf1_tasks
        finally:
            await queue.stop(drain=False)

    @pytest.mark.asyncio
    async def test_cancel_task(self, queue):
        """Test cancelling a task."""
        await queue.start()
        try:
            task = WorkflowTask.create(workflow_id="wf_001", step_id="step_1")
            await queue.enqueue(task)

            assert queue.cancel_task(task.id)
            assert task.status == TaskStatus.CANCELLED
        finally:
            await queue.stop(drain=False)

    @pytest.mark.asyncio
    async def test_cancel_workflow(self, queue):
        """Test cancelling all tasks in a workflow."""
        await queue.start()
        try:
            task1 = WorkflowTask.create(workflow_id="wf_001", step_id="step_1")
            task2 = WorkflowTask.create(workflow_id="wf_001", step_id="step_2")

            await queue.enqueue(task1)
            await queue.enqueue(task2)

            cancelled = queue.cancel_workflow("wf_001")
            assert cancelled == 2
            assert task1.status == TaskStatus.CANCELLED
            assert task2.status == TaskStatus.CANCELLED
        finally:
            await queue.stop(drain=False)

    @pytest.mark.asyncio
    async def test_get_stats(self, queue):
        """Test getting queue statistics."""
        await queue.start()
        try:
            task1 = WorkflowTask.create(workflow_id="wf_001", step_id="step_1")
            task2 = WorkflowTask.create(
                workflow_id="wf_001",
                step_id="step_2",
                depends_on=[task1.id],
            )

            await queue.enqueue(task1)
            await queue.enqueue(task2)

            stats = queue.get_stats()
            assert stats.total_tasks == 2
            assert stats.ready_count == 1  # task1 is ready
            assert stats.pending_count == 1  # task2 is pending
        finally:
            await queue.stop(drain=False)

    @pytest.mark.asyncio
    async def test_queue_full(self, queue):
        """Test queue size limit."""
        queue._config.max_queue_size = 2

        await queue.start()
        try:
            task1 = WorkflowTask.create(workflow_id="wf_001", step_id="step_1")
            task2 = WorkflowTask.create(workflow_id="wf_001", step_id="step_2")
            task3 = WorkflowTask.create(workflow_id="wf_001", step_id="step_3")

            await queue.enqueue(task1)
            await queue.enqueue(task2)

            with pytest.raises(RuntimeError, match="Queue is full"):
                await queue.enqueue(task3)
        finally:
            await queue.stop(drain=False)


class TestTaskResult:
    """Tests for TaskResult."""

    def test_success_result(self):
        """Test successful result."""
        result = TaskResult(
            success=True,
            output={"data": "test"},
            execution_time_ms=100.5,
        )
        assert result.success
        assert result.output == {"data": "test"}
        assert result.error is None
        assert result.execution_time_ms == 100.5

    def test_failure_result(self):
        """Test failure result."""
        result = TaskResult(
            success=False,
            error="Something went wrong",
        )
        assert not result.success
        assert result.error == "Something went wrong"


class TestTaskPriority:
    """Tests for TaskPriority enum."""

    def test_priority_ordering(self):
        """Test priority values are ordered correctly."""
        assert TaskPriority.CRITICAL < TaskPriority.HIGH
        assert TaskPriority.HIGH < TaskPriority.NORMAL
        assert TaskPriority.NORMAL < TaskPriority.LOW
        assert TaskPriority.LOW < TaskPriority.BACKGROUND

    def test_priority_values(self):
        """Test specific priority values."""
        assert TaskPriority.CRITICAL.value == 0
        assert TaskPriority.HIGH.value == 10
        assert TaskPriority.NORMAL.value == 50
        assert TaskPriority.LOW.value == 100
        assert TaskPriority.BACKGROUND.value == 200
