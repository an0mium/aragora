"""Tests for TaskExecutor and ExecutorPool."""

import asyncio
import pytest

from aragora.workflow.queue import WorkflowTask, TaskPriority
from aragora.workflow.queue.task import TaskResult
from aragora.workflow.queue.executor import (
    ExecutorPool,
    ExecutorStatus,
    PoolConfig,
    StepExecutor,
    TaskExecutor,
)


class TestStepExecutor:
    """Tests for StepExecutor."""

    @pytest.fixture
    def executor(self):
        """Create a step executor."""
        return StepExecutor("exec_1")

    def test_executor_properties(self, executor):
        """Test executor properties."""
        assert executor.id == "exec_1"
        assert executor.status == ExecutorStatus.IDLE
        assert executor.is_available

    def test_register_handler(self, executor):
        """Test registering a step handler."""
        def handler(task):
            return TaskResult(success=True, output={"handled": True})

        executor.register_handler("test_type", handler)
        assert "test_type" in executor._handlers

    @pytest.mark.asyncio
    async def test_execute_with_handler(self, executor):
        """Test executing with a registered handler."""
        async def handler(task):
            return TaskResult(success=True, output={"step": task.step_id})

        executor.register_handler("test", handler)

        task = WorkflowTask.create(
            workflow_id="wf_001",
            step_id="step_1",
            step_config={"type": "test"},
        )

        result = await executor.execute(task)
        assert result.success
        assert result.output["step"] == "step_1"

    @pytest.mark.asyncio
    async def test_execute_without_handler(self, executor):
        """Test executing without a handler returns placeholder."""
        task = WorkflowTask.create(
            workflow_id="wf_001",
            step_id="step_1",
        )

        result = await executor.execute(task)
        assert result.success
        assert result.output["executed"]

    @pytest.mark.asyncio
    async def test_executor_stats(self, executor):
        """Test executor statistics tracking."""
        async def handler(task):
            await asyncio.sleep(0.01)
            return TaskResult(success=True)

        executor.register_handler("test", handler)

        task = WorkflowTask.create(
            workflow_id="wf_001",
            step_id="step_1",
            step_config={"type": "test"},
        )

        await executor._run_with_tracking(task)

        assert executor.stats.tasks_completed == 1
        assert executor.stats.avg_execution_time_ms > 0


class TestExecutorPool:
    """Tests for ExecutorPool."""

    @pytest.fixture
    def pool_config(self):
        """Create pool configuration."""
        return PoolConfig(
            min_executors=2,
            max_executors=4,
            scale_up_threshold=0.8,
            scale_down_threshold=0.2,
        )

    @pytest.fixture
    def pool(self, pool_config):
        """Create an executor pool."""
        return ExecutorPool(config=pool_config)

    @pytest.mark.asyncio
    async def test_start_stop(self, pool):
        """Test starting and stopping the pool."""
        await pool.start()
        assert pool._started
        assert pool.executor_count >= 2  # min_executors

        await pool.stop()
        assert not pool._started

    @pytest.mark.asyncio
    async def test_executor_count(self, pool):
        """Test initial executor count."""
        await pool.start()
        try:
            assert pool.executor_count == 2  # min_executors
        finally:
            await pool.stop(drain=False)

    @pytest.mark.asyncio
    async def test_acquire_release(self, pool):
        """Test acquiring and releasing executors."""
        await pool.start()
        try:
            task = WorkflowTask.create(workflow_id="wf_001", step_id="step_1")

            executor = await pool.acquire(task)
            assert executor is not None
            assert task.id in pool._task_assignments

            await pool.release(task)
            assert task.id not in pool._task_assignments
        finally:
            await pool.stop(drain=False)

    @pytest.mark.asyncio
    async def test_execute_task(self, pool):
        """Test executing a task through the pool."""
        await pool.start()
        try:
            task = WorkflowTask.create(
                workflow_id="wf_001",
                step_id="step_1",
            )

            result = await pool.execute(task)
            assert result.success
        finally:
            await pool.stop(drain=False)

    @pytest.mark.asyncio
    async def test_utilization(self, pool):
        """Test utilization calculation."""
        await pool.start()
        try:
            assert pool.utilization == 0.0  # No tasks running

            # Start a task
            task = WorkflowTask.create(workflow_id="wf_001", step_id="step_1")
            executor = await pool.acquire(task)

            # Manually set executor to busy for testing
            executor._status = ExecutorStatus.BUSY
            assert pool.utilization == 0.5  # 1 of 2 executors busy
        finally:
            await pool.stop(drain=False)

    @pytest.mark.asyncio
    async def test_get_stats(self, pool):
        """Test getting pool statistics."""
        await pool.start()
        try:
            stats = pool.get_stats()

            assert stats["executor_count"] == 2
            assert stats["busy_count"] == 0
            assert stats["utilization"] == 0.0
            assert "executors" in stats
        finally:
            await pool.stop(drain=False)

    @pytest.mark.asyncio
    async def test_custom_executor_factory(self):
        """Test using a custom executor factory."""
        created_count = 0

        def custom_factory(executor_id: str) -> TaskExecutor:
            nonlocal created_count
            created_count += 1
            return StepExecutor(executor_id)

        pool = ExecutorPool(
            config=PoolConfig(min_executors=3),
            executor_factory=custom_factory,
        )

        await pool.start()
        try:
            assert created_count == 3
        finally:
            await pool.stop(drain=False)


class TestExecutorStatus:
    """Tests for ExecutorStatus enum."""

    def test_status_values(self):
        """Test status enum values."""
        assert ExecutorStatus.IDLE.value == "idle"
        assert ExecutorStatus.BUSY.value == "busy"
        assert ExecutorStatus.STOPPING.value == "stopping"
        assert ExecutorStatus.STOPPED.value == "stopped"
        assert ExecutorStatus.ERROR.value == "error"
