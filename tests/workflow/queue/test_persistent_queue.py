"""
Tests for PersistentTaskQueue.

Tests SQLite-backed task persistence and recovery.
"""

import tempfile
from pathlib import Path

import pytest

from aragora.workflow.queue.persistent_queue import (
    PersistentTaskQueue,
    get_persistent_task_queue,
    reset_persistent_task_queue,
)
from aragora.workflow.queue.task import (
    TaskStatus,
    TaskPriority,
    TaskResult,
    WorkflowTask,
)
from aragora.workflow.queue.queue import TaskQueueConfig


@pytest.fixture
def temp_db_path():
    """Create a temporary database path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test_tasks.db"


@pytest.fixture
def persistent_queue(temp_db_path):
    """Create a persistent queue for testing."""
    config = TaskQueueConfig(max_concurrent=5)
    queue = PersistentTaskQueue(config=config, db_path=temp_db_path)
    return queue


@pytest.fixture
def sample_task():
    """Create a sample task."""
    return WorkflowTask.create(
        workflow_id="wf_test_123",
        step_id="step_1",
        step_config={"action": "test"},
        priority=TaskPriority.NORMAL,
    )


class TestPersistentTaskQueue:
    """Tests for PersistentTaskQueue."""

    @pytest.mark.asyncio
    async def test_enqueue_persists_task(self, persistent_queue, sample_task):
        """Enqueued tasks should be persisted to database."""
        await persistent_queue.start()
        try:
            task_id = await persistent_queue.enqueue(sample_task)

            # Verify task in memory
            memory_task = persistent_queue.get_task(task_id)
            assert memory_task is not None
            assert memory_task.workflow_id == "wf_test_123"

            # Verify task in database
            db_task = persistent_queue.get_task_from_db(task_id)
            assert db_task is not None
            assert db_task.workflow_id == "wf_test_123"
            assert db_task.step_id == "step_1"

        finally:
            await persistent_queue.stop()

    @pytest.mark.asyncio
    async def test_enqueue_many_persists_all(self, persistent_queue):
        """Batch enqueue should persist all tasks."""
        await persistent_queue.start()
        try:
            tasks = [
                WorkflowTask.create(
                    workflow_id="wf_batch",
                    step_id=f"step_{i}",
                )
                for i in range(5)
            ]

            task_ids = await persistent_queue.enqueue_many(tasks)
            assert len(task_ids) == 5

            # Verify all in database
            db_tasks, total = persistent_queue.list_tasks_from_db(workflow_id="wf_batch")
            assert len(db_tasks) == 5
            assert total == 5

        finally:
            await persistent_queue.stop()

    @pytest.mark.asyncio
    async def test_task_status_updates_persisted(self, persistent_queue, sample_task):
        """Task status changes should be persisted."""
        await persistent_queue.start()
        try:
            task_id = await persistent_queue.enqueue(sample_task)

            # Task auto-marked ready in memory (no deps)
            task = persistent_queue.get_task(task_id)
            assert task.status == TaskStatus.READY

            # Initial persist happens with PENDING (before mark_ready in enqueue)
            # The task is persisted, then mark_ready is called in memory
            db_task = persistent_queue.get_task_from_db(task_id)
            assert db_task is not None
            assert db_task.workflow_id == sample_task.workflow_id

            # Manually update and verify persistence
            task.status = TaskStatus.RUNNING
            task.executor_id = "test_executor"
            persistent_queue._update_task_status(task)

            db_task_updated = persistent_queue.get_task_from_db(task_id)
            assert db_task_updated.status == TaskStatus.RUNNING
            assert db_task_updated.executor_id == "test_executor"

        finally:
            await persistent_queue.stop()

    @pytest.mark.asyncio
    async def test_recover_tasks_after_restart(self, temp_db_path):
        """Tasks should be recoverable after queue restart."""
        config = TaskQueueConfig(max_concurrent=5)

        # Create and enqueue tasks
        queue1 = PersistentTaskQueue(config=config, db_path=temp_db_path)
        await queue1.start()

        task_ids = []
        for i in range(3):
            task = WorkflowTask.create(
                workflow_id="wf_recover",
                step_id=f"step_{i}",
            )
            task_id = await queue1.enqueue(task)
            task_ids.append(task_id)

        await queue1.stop()

        # Create new queue instance (simulates restart)
        queue2 = PersistentTaskQueue(config=config, db_path=temp_db_path)
        await queue2.start()

        try:
            # Recover tasks
            recovered = await queue2.recover_tasks()
            assert recovered == 3

            # Verify tasks are in memory
            for task_id in task_ids:
                task = queue2.get_task(task_id)
                assert task is not None
                assert task.workflow_id == "wf_recover"

        finally:
            await queue2.stop()

    @pytest.mark.asyncio
    async def test_recover_running_tasks_reset_to_ready(self, temp_db_path):
        """Running tasks should be reset to READY on recovery."""
        config = TaskQueueConfig(max_concurrent=5)

        # Create queue and add task
        queue1 = PersistentTaskQueue(config=config, db_path=temp_db_path)

        # Manually insert a "running" task into DB
        import sqlite3
        import json
        from datetime import datetime, timezone

        conn = sqlite3.connect(str(temp_db_path))
        conn.execute(
            """
            INSERT INTO task_queue (
                id, workflow_id, step_id, status, priority,
                timeout_seconds, max_retries, created_at, tenant_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                "task_running_123",
                "wf_running",
                "step_1",
                "running",
                50,
                300.0,
                2,
                datetime.now(timezone.utc).isoformat(),
                "default",
            ),
        )
        conn.commit()
        conn.close()

        # Create new queue and recover
        queue2 = PersistentTaskQueue(config=config, db_path=temp_db_path)
        await queue2.start()

        try:
            recovered = await queue2.recover_tasks()
            assert recovered == 1

            # Verify task was reset to READY
            task = queue2.get_task("task_running_123")
            assert task is not None
            assert task.status == TaskStatus.READY
            assert task.started_at is None

        finally:
            await queue2.stop()

    @pytest.mark.asyncio
    async def test_list_tasks_from_db(self, persistent_queue):
        """Should list tasks with filtering."""
        await persistent_queue.start()
        try:
            # Create tasks with different workflows and statuses
            for i in range(5):
                task = WorkflowTask.create(
                    workflow_id=f"wf_{i % 2}",  # 0 or 1
                    step_id=f"step_{i}",
                )
                await persistent_queue.enqueue(task)

            # List all
            all_tasks, total = persistent_queue.list_tasks_from_db()
            assert len(all_tasks) == 5
            assert total == 5

            # Filter by workflow
            wf0_tasks, wf0_total = persistent_queue.list_tasks_from_db(workflow_id="wf_0")
            assert len(wf0_tasks) == 3  # indices 0, 2, 4

            wf1_tasks, wf1_total = persistent_queue.list_tasks_from_db(workflow_id="wf_1")
            assert len(wf1_tasks) == 2  # indices 1, 3

        finally:
            await persistent_queue.stop()

    @pytest.mark.asyncio
    async def test_delete_completed_tasks(self, persistent_queue, sample_task):
        """Should clean up old completed tasks."""
        await persistent_queue.start()
        try:
            task_id = await persistent_queue.enqueue(sample_task)

            # Mark as completed (via direct DB update for testing)
            import sqlite3
            from datetime import datetime, timezone, timedelta

            conn = sqlite3.connect(str(persistent_queue._db_path))
            old_time = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
            conn.execute(
                """
                UPDATE task_queue SET status = 'completed', completed_at = ?
                WHERE id = ?
            """,
                (old_time, task_id),
            )
            conn.commit()
            conn.close()

            # Delete old tasks
            deleted = persistent_queue.delete_completed_tasks(older_than_hours=24)
            assert deleted == 1

            # Verify deleted from DB
            db_task = persistent_queue.get_task_from_db(task_id)
            assert db_task is None

        finally:
            await persistent_queue.stop()

    @pytest.mark.asyncio
    async def test_multi_tenant_isolation(self, persistent_queue):
        """Tasks should be isolated by tenant_id."""
        await persistent_queue.start()
        try:
            # Create tasks for different tenants
            task1 = WorkflowTask.create(
                workflow_id="wf_tenant",
                step_id="step_1",
                tenant_id="tenant_a",
            )
            task2 = WorkflowTask.create(
                workflow_id="wf_tenant",
                step_id="step_2",
                tenant_id="tenant_b",
            )

            await persistent_queue.enqueue(task1)
            await persistent_queue.enqueue(task2)

            # List by tenant
            tenant_a_tasks, _ = persistent_queue.list_tasks_from_db(tenant_id="tenant_a")
            tenant_b_tasks, _ = persistent_queue.list_tasks_from_db(tenant_id="tenant_b")

            assert len(tenant_a_tasks) == 1
            assert len(tenant_b_tasks) == 1
            assert tenant_a_tasks[0].step_id == "step_1"
            assert tenant_b_tasks[0].step_id == "step_2"

        finally:
            await persistent_queue.stop()


class TestPersistentTaskQueueGlobal:
    """Tests for global queue factory."""

    def setup_method(self):
        """Reset global queue."""
        reset_persistent_task_queue()

    def teardown_method(self):
        """Reset global queue."""
        reset_persistent_task_queue()

    def test_get_persistent_task_queue_singleton(self, temp_db_path):
        """Should return same instance on multiple calls."""
        queue1 = get_persistent_task_queue(db_path=temp_db_path)
        queue2 = get_persistent_task_queue()

        assert queue1 is queue2

    def test_reset_persistent_task_queue(self, temp_db_path):
        """Should create new instance after reset."""
        queue1 = get_persistent_task_queue(db_path=temp_db_path)
        reset_persistent_task_queue()
        queue2 = get_persistent_task_queue(db_path=temp_db_path)

        assert queue1 is not queue2


class TestTaskPersistenceRoundtrip:
    """Tests for task serialization to/from database."""

    @pytest.mark.asyncio
    async def test_task_with_result_persists(self, persistent_queue):
        """Task with result should round-trip through database."""
        await persistent_queue.start()
        try:
            task = WorkflowTask.create(
                workflow_id="wf_result",
                step_id="step_1",
            )
            task.result = TaskResult(
                success=True,
                output={"data": "test_output"},
                execution_time_ms=123.45,
            )
            task.status = TaskStatus.COMPLETED

            task_id = await persistent_queue.enqueue(task)

            # Retrieve from DB
            db_task = persistent_queue.get_task_from_db(task_id)
            assert db_task is not None
            assert db_task.result is not None
            assert db_task.result.success is True
            assert db_task.result.output == {"data": "test_output"}
            assert db_task.result.execution_time_ms == 123.45

        finally:
            await persistent_queue.stop()

    @pytest.mark.asyncio
    async def test_task_with_metadata_persists(self, persistent_queue):
        """Task metadata should persist."""
        await persistent_queue.start()
        try:
            task = WorkflowTask.create(
                workflow_id="wf_meta",
                step_id="step_1",
                metadata={"custom_field": "custom_value", "count": 42},
            )

            task_id = await persistent_queue.enqueue(task)

            db_task = persistent_queue.get_task_from_db(task_id)
            assert db_task is not None
            assert db_task.metadata == {"custom_field": "custom_value", "count": 42}

        finally:
            await persistent_queue.stop()

    @pytest.mark.asyncio
    async def test_task_with_dependencies_persists(self, persistent_queue):
        """Task dependencies should persist."""
        await persistent_queue.start()
        try:
            task = WorkflowTask.create(
                workflow_id="wf_deps",
                step_id="step_2",
                depends_on=["task_1", "task_2", "task_3"],
            )

            task_id = await persistent_queue.enqueue(task)

            db_task = persistent_queue.get_task_from_db(task_id)
            assert db_task is not None
            assert db_task.depends_on == ["task_1", "task_2", "task_3"]

        finally:
            await persistent_queue.stop()
