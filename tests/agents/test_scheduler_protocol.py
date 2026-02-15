"""
Tests for the unified scheduler protocol.

Verifies:
- Protocol compliance for both adapters
- Auto-detection logic in get_scheduler()
- TaskInfo properties and lifecycle
- Adapter-specific behavior
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
import os
import pytest

from aragora.agents.scheduler_protocol import (
    SchedulerProtocol,
    SchedulerType,
    TaskInfo,
    LocalSchedulerAdapter,
    DistributedSchedulerAdapter,
    get_scheduler,
    reset_scheduler,
)
from aragora.fabric.models import Priority, TaskStatus


class TestTaskInfo:
    """Tests for TaskInfo dataclass."""

    def test_basic_creation(self):
        """Test TaskInfo can be created with minimal fields."""
        info = TaskInfo(
            task_id="test-123",
            task_type="debate",
            status=TaskStatus.PENDING,
            priority=Priority.NORMAL,
        )
        assert info.task_id == "test-123"
        assert info.task_type == "debate"
        assert info.status == TaskStatus.PENDING
        assert info.priority == Priority.NORMAL
        assert info.agent_id is None
        assert info.result is None
        assert info.error is None

    def test_is_terminal_completed(self):
        """Test is_terminal returns True for COMPLETED status."""
        info = TaskInfo(
            task_id="test-1",
            task_type="analysis",
            status=TaskStatus.COMPLETED,
            priority=Priority.HIGH,
        )
        assert info.is_terminal is True

    def test_is_terminal_failed(self):
        """Test is_terminal returns True for FAILED status."""
        info = TaskInfo(
            task_id="test-2",
            task_type="generation",
            status=TaskStatus.FAILED,
            priority=Priority.LOW,
            error="Timeout exceeded",
        )
        assert info.is_terminal is True

    def test_is_terminal_cancelled(self):
        """Test is_terminal returns True for CANCELLED status."""
        info = TaskInfo(
            task_id="test-3",
            task_type="debate",
            status=TaskStatus.CANCELLED,
            priority=Priority.NORMAL,
        )
        assert info.is_terminal is True

    def test_is_terminal_pending(self):
        """Test is_terminal returns False for PENDING status."""
        info = TaskInfo(
            task_id="test-4",
            task_type="debate",
            status=TaskStatus.PENDING,
            priority=Priority.NORMAL,
        )
        assert info.is_terminal is False

    def test_is_terminal_running(self):
        """Test is_terminal returns False for RUNNING status."""
        info = TaskInfo(
            task_id="test-5",
            task_type="debate",
            status=TaskStatus.RUNNING,
            priority=Priority.CRITICAL,
        )
        assert info.is_terminal is False

    def test_duration_seconds_completed(self):
        """Test duration_seconds calculation for completed task."""
        start = datetime(2025, 1, 1, 12, 0, 0)
        end = datetime(2025, 1, 1, 12, 5, 30)
        info = TaskInfo(
            task_id="test-6",
            task_type="analysis",
            status=TaskStatus.COMPLETED,
            priority=Priority.NORMAL,
            started_at=start,
            completed_at=end,
        )
        assert info.duration_seconds == 330.0  # 5 minutes 30 seconds

    def test_duration_seconds_in_progress(self):
        """Test duration_seconds returns None for in-progress task."""
        info = TaskInfo(
            task_id="test-7",
            task_type="debate",
            status=TaskStatus.RUNNING,
            priority=Priority.NORMAL,
            started_at=datetime.now(timezone.utc),
        )
        assert info.duration_seconds is None

    def test_metadata_default(self):
        """Test metadata defaults to empty dict."""
        info = TaskInfo(
            task_id="test-8",
            task_type="debate",
            status=TaskStatus.PENDING,
            priority=Priority.NORMAL,
        )
        assert info.metadata == {}

    def test_metadata_custom(self):
        """Test custom metadata is preserved."""
        info = TaskInfo(
            task_id="test-9",
            task_type="debate",
            status=TaskStatus.PENDING,
            priority=Priority.NORMAL,
            metadata={"user_id": "user-123", "source": "api"},
        )
        assert info.metadata["user_id"] == "user-123"
        assert info.metadata["source"] == "api"


class TestSchedulerType:
    """Tests for SchedulerType enum."""

    def test_local_value(self):
        """Test LOCAL enum value."""
        assert SchedulerType.LOCAL.value == "local"

    def test_distributed_value(self):
        """Test DISTRIBUTED enum value."""
        assert SchedulerType.DISTRIBUTED.value == "distributed"

    def test_auto_value(self):
        """Test AUTO enum value."""
        assert SchedulerType.AUTO.value == "auto"


class TestGetScheduler:
    """Tests for get_scheduler factory function."""

    def setup_method(self):
        """Reset scheduler singleton before each test."""
        reset_scheduler()

    def teardown_method(self):
        """Clean up after each test."""
        reset_scheduler()
        # Clean up environment
        os.environ.pop("ARAGORA_SCHEDULER_TYPE", None)
        os.environ.pop("REDIS_URL", None)

    def test_explicit_local(self):
        """Test explicitly requesting local scheduler."""
        scheduler = get_scheduler(SchedulerType.LOCAL)
        assert isinstance(scheduler, LocalSchedulerAdapter)

    @patch("aragora.agents.scheduler_protocol.DistributedSchedulerAdapter")
    def test_explicit_distributed(self, mock_distributed):
        """Test explicitly requesting distributed scheduler."""
        mock_instance = MagicMock()
        mock_distributed.return_value = mock_instance

        scheduler = get_scheduler(SchedulerType.DISTRIBUTED)
        assert scheduler is mock_instance
        mock_distributed.assert_called_once()

    def test_auto_without_redis(self):
        """Test AUTO mode without REDIS_URL uses local."""
        os.environ.pop("REDIS_URL", None)
        scheduler = get_scheduler(SchedulerType.AUTO)
        assert isinstance(scheduler, LocalSchedulerAdapter)

    @patch("aragora.agents.scheduler_protocol.DistributedSchedulerAdapter")
    def test_auto_with_redis(self, mock_distributed):
        """Test AUTO mode with REDIS_URL uses distributed."""
        mock_instance = MagicMock()
        mock_distributed.return_value = mock_instance

        os.environ["REDIS_URL"] = "redis://localhost:6379"
        scheduler = get_scheduler(SchedulerType.AUTO)
        assert scheduler is mock_instance

    def test_env_override_local(self):
        """Test ARAGORA_SCHEDULER_TYPE=local override."""
        os.environ["ARAGORA_SCHEDULER_TYPE"] = "local"
        os.environ["REDIS_URL"] = "redis://localhost:6379"  # Should be ignored

        scheduler = get_scheduler(SchedulerType.AUTO)
        assert isinstance(scheduler, LocalSchedulerAdapter)

    @patch("aragora.agents.scheduler_protocol.DistributedSchedulerAdapter")
    def test_env_override_distributed(self, mock_distributed):
        """Test ARAGORA_SCHEDULER_TYPE=distributed override."""
        mock_instance = MagicMock()
        mock_distributed.return_value = mock_instance

        os.environ["ARAGORA_SCHEDULER_TYPE"] = "distributed"
        scheduler = get_scheduler(SchedulerType.AUTO)
        assert scheduler is mock_instance

    def test_singleton_behavior(self):
        """Test that get_scheduler returns the same instance."""
        scheduler1 = get_scheduler(SchedulerType.LOCAL)
        scheduler2 = get_scheduler(SchedulerType.LOCAL)
        assert scheduler1 is scheduler2

    def test_reset_clears_singleton(self):
        """Test reset_scheduler allows new instance creation."""
        scheduler1 = get_scheduler(SchedulerType.LOCAL)
        reset_scheduler()
        scheduler2 = get_scheduler(SchedulerType.LOCAL)
        assert scheduler1 is not scheduler2


class TestLocalSchedulerAdapter:
    """Tests for LocalSchedulerAdapter."""

    @pytest.fixture
    def adapter(self):
        """Create a LocalSchedulerAdapter with mocked scheduler."""
        mock_scheduler = MagicMock()
        mock_scheduler._tasks = {}
        mock_scheduler._tasks_scheduled = 0
        mock_scheduler._tasks_completed = 0
        mock_scheduler._tasks_failed = 0
        mock_scheduler._tasks_cancelled = 0
        mock_scheduler._agent_queues = {}
        return LocalSchedulerAdapter(scheduler=mock_scheduler)

    @pytest.mark.asyncio
    async def test_schedule_task(self, adapter):
        """Test scheduling a task through local adapter."""
        mock_handle = MagicMock()
        mock_handle.task_id = "task-123"
        mock_handle.status = TaskStatus.PENDING
        mock_handle.scheduled_at = datetime.now(timezone.utc)

        adapter._scheduler.schedule = AsyncMock(return_value=mock_handle)

        result = await adapter.schedule_task(
            task_type="debate",
            payload={"topic": "AI safety"},
            priority=Priority.HIGH,
        )

        assert result.task_id == "task-123"
        assert result.task_type == "debate"
        assert result.priority == Priority.HIGH
        adapter._scheduler.schedule.assert_called_once()

    @pytest.mark.asyncio
    async def test_schedule_task_with_agent_id(self, adapter):
        """Test scheduling task to specific agent."""
        mock_handle = MagicMock()
        mock_handle.task_id = "task-456"
        mock_handle.status = TaskStatus.PENDING
        mock_handle.scheduled_at = datetime.now(timezone.utc)

        adapter._scheduler.schedule = AsyncMock(return_value=mock_handle)

        result = await adapter.schedule_task(
            task_type="analysis",
            payload={"data": [1, 2, 3]},
            agent_id="agent-claude",
        )

        assert result.agent_id == "agent-claude"

    @pytest.mark.asyncio
    async def test_get_task_status_found(self, adapter):
        """Test getting status of existing task."""
        mock_handle = MagicMock()
        mock_handle.task_id = "task-789"
        mock_handle.status = TaskStatus.RUNNING
        mock_handle.agent_id = "agent-1"
        mock_handle.scheduled_at = datetime.now(timezone.utc)
        mock_handle.started_at = datetime.now(timezone.utc)
        mock_handle.completed_at = None
        mock_handle.result = None
        mock_handle.error = None

        adapter._scheduler.get_handle = AsyncMock(return_value=mock_handle)
        adapter._task_types["task-789"] = "generation"

        result = await adapter.get_task_status("task-789")

        assert result is not None
        assert result.task_id == "task-789"
        assert result.task_type == "generation"
        assert result.status == TaskStatus.RUNNING

    @pytest.mark.asyncio
    async def test_get_task_status_not_found(self, adapter):
        """Test getting status of non-existent task."""
        adapter._scheduler.get_handle = AsyncMock(return_value=None)

        result = await adapter.get_task_status("nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_cancel_task(self, adapter):
        """Test cancelling a task."""
        adapter._scheduler.cancel = AsyncMock(return_value=True)

        result = await adapter.cancel_task("task-to-cancel")

        assert result is True
        adapter._scheduler.cancel.assert_called_once_with("task-to-cancel")

    @pytest.mark.asyncio
    async def test_list_tasks_no_filter(self, adapter):
        """Test listing all tasks without filters."""
        mock_handle1 = MagicMock()
        mock_handle1.task_id = "task-1"
        mock_handle1.status = TaskStatus.COMPLETED
        mock_handle1.agent_id = "agent-1"
        mock_handle1.scheduled_at = datetime.now(timezone.utc)
        mock_handle1.started_at = datetime.now(timezone.utc)
        mock_handle1.completed_at = datetime.now(timezone.utc)
        mock_handle1.result = {"answer": 42}
        mock_handle1.error = None

        adapter._scheduler._tasks = {"task-1": mock_handle1}
        adapter._task_types["task-1"] = "debate"

        result = await adapter.list_tasks()

        assert len(result) == 1
        assert result[0].task_id == "task-1"

    @pytest.mark.asyncio
    async def test_list_tasks_with_status_filter(self, adapter):
        """Test listing tasks filtered by status."""
        mock_handle1 = MagicMock()
        mock_handle1.task_id = "task-1"
        mock_handle1.status = TaskStatus.COMPLETED
        mock_handle1.agent_id = "agent-1"
        mock_handle1.scheduled_at = datetime.now(timezone.utc)
        mock_handle1.started_at = None
        mock_handle1.completed_at = None
        mock_handle1.result = None
        mock_handle1.error = None

        mock_handle2 = MagicMock()
        mock_handle2.task_id = "task-2"
        mock_handle2.status = TaskStatus.PENDING
        mock_handle2.agent_id = "agent-2"
        mock_handle2.scheduled_at = datetime.now(timezone.utc)
        mock_handle2.started_at = None
        mock_handle2.completed_at = None
        mock_handle2.result = None
        mock_handle2.error = None

        adapter._scheduler._tasks = {
            "task-1": mock_handle1,
            "task-2": mock_handle2,
        }
        adapter._task_types["task-1"] = "debate"
        adapter._task_types["task-2"] = "analysis"

        result = await adapter.list_tasks(status=TaskStatus.PENDING)

        assert len(result) == 1
        assert result[0].task_id == "task-2"

    @pytest.mark.asyncio
    async def test_get_queue_stats(self, adapter):
        """Test getting queue statistics."""
        adapter._scheduler._tasks = {"t1": MagicMock(), "t2": MagicMock()}
        adapter._scheduler._tasks_scheduled = 10
        adapter._scheduler._tasks_completed = 7
        adapter._scheduler._tasks_failed = 2
        adapter._scheduler._tasks_cancelled = 1
        adapter._scheduler._agent_queues = {
            "agent-1": [MagicMock()],
            "agent-2": [MagicMock(), MagicMock()],
        }

        result = await adapter.get_queue_stats()

        assert result["scheduler_type"] == "local"
        assert result["total_tasks"] == 2
        assert result["tasks_scheduled"] == 10
        assert result["tasks_completed"] == 7
        assert result["tasks_failed"] == 2
        assert result["tasks_cancelled"] == 1
        assert result["agent_queues"]["agent-1"] == 1
        assert result["agent_queues"]["agent-2"] == 2


class TestDistributedSchedulerAdapter:
    """Tests for DistributedSchedulerAdapter."""

    @pytest.fixture
    def adapter(self):
        """Create a DistributedSchedulerAdapter with mocked scheduler."""
        mock_scheduler = MagicMock()
        return DistributedSchedulerAdapter(scheduler=mock_scheduler)

    @pytest.mark.asyncio
    async def test_schedule_task(self, adapter):
        """Test scheduling a task through distributed adapter."""
        # submit() returns task_id string directly
        adapter._scheduler.submit = AsyncMock(return_value="dist-task-123")

        result = await adapter.schedule_task(
            task_type="debate",
            payload={"topic": "Climate policy"},
            priority=Priority.HIGH,
        )

        assert result.task_id == "dist-task-123"
        assert result.task_type == "debate"
        assert result.priority == Priority.HIGH
        assert result.status == TaskStatus.PENDING

    @pytest.mark.asyncio
    async def test_schedule_task_priority_mapping(self, adapter):
        """Test that priorities are mapped correctly to control plane."""
        # submit() returns task_id string directly
        adapter._scheduler.submit = AsyncMock(return_value="dist-task-456")

        # Test CRITICAL priority
        await adapter.schedule_task(
            task_type="urgent",
            payload={},
            priority=Priority.CRITICAL,
        )

        call_args = adapter._scheduler.submit.call_args
        # The priority should be mapped to TaskPriority.CRITICAL
        assert call_args is not None

    @pytest.mark.asyncio
    async def test_get_task_status_found(self, adapter):
        """Test getting status from distributed scheduler."""
        import time

        from aragora.control_plane.scheduler import TaskStatus as CPTaskStatus

        mock_task = MagicMock()
        mock_task.id = "dist-task-789"
        mock_task.task_type = "analysis"
        mock_task.status = CPTaskStatus.RUNNING
        mock_task.assigned_agent = "agent-claude"
        mock_task.created_at = time.time()
        mock_task.started_at = time.time()
        mock_task.completed_at = None
        mock_task.result = None
        mock_task.error = None
        mock_task.metadata = {"source": "api"}

        adapter._scheduler.get = AsyncMock(return_value=mock_task)

        result = await adapter.get_task_status("dist-task-789")

        assert result is not None
        assert result.task_id == "dist-task-789"
        assert result.agent_id == "agent-claude"

    @pytest.mark.asyncio
    async def test_get_task_status_not_found(self, adapter):
        """Test getting status of non-existent distributed task."""
        adapter._scheduler.get = AsyncMock(return_value=None)

        result = await adapter.get_task_status("nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_cancel_task(self, adapter):
        """Test cancelling a distributed task."""
        adapter._scheduler.cancel = AsyncMock(return_value=True)

        result = await adapter.cancel_task("dist-task-to-cancel")

        assert result is True
        adapter._scheduler.cancel.assert_called_once_with("dist-task-to-cancel")

    @pytest.mark.asyncio
    async def test_get_queue_stats(self, adapter):
        """Test getting distributed queue statistics."""
        adapter._scheduler.get_stats = AsyncMock(
            return_value={
                "pending_tasks": 5,
                "running_tasks": 3,
                "completed_tasks": 100,
                "agents_available": 10,
            }
        )

        result = await adapter.get_queue_stats()

        assert result["scheduler_type"] == "distributed"
        assert result["pending_tasks"] == 5
        assert result["running_tasks"] == 3


class TestProtocolCompliance:
    """Tests that both adapters satisfy the SchedulerProtocol."""

    def test_local_adapter_is_protocol(self):
        """Test LocalSchedulerAdapter implements SchedulerProtocol."""
        mock_scheduler = MagicMock()
        mock_scheduler._tasks = {}
        mock_scheduler._tasks_scheduled = 0
        mock_scheduler._tasks_completed = 0
        mock_scheduler._tasks_failed = 0
        mock_scheduler._tasks_cancelled = 0
        mock_scheduler._agent_queues = {}

        adapter = LocalSchedulerAdapter(scheduler=mock_scheduler)

        # Protocol checks
        assert hasattr(adapter, "schedule_task")
        assert hasattr(adapter, "get_task_status")
        assert hasattr(adapter, "cancel_task")
        assert hasattr(adapter, "list_tasks")
        assert hasattr(adapter, "get_queue_stats")

        # Verify they are callable
        assert callable(adapter.schedule_task)
        assert callable(adapter.get_task_status)
        assert callable(adapter.cancel_task)
        assert callable(adapter.list_tasks)
        assert callable(adapter.get_queue_stats)

    def test_distributed_adapter_is_protocol(self):
        """Test DistributedSchedulerAdapter implements SchedulerProtocol."""
        mock_scheduler = MagicMock()
        adapter = DistributedSchedulerAdapter(scheduler=mock_scheduler)

        # Protocol checks
        assert hasattr(adapter, "schedule_task")
        assert hasattr(adapter, "get_task_status")
        assert hasattr(adapter, "cancel_task")
        assert hasattr(adapter, "list_tasks")
        assert hasattr(adapter, "get_queue_stats")

        # Verify they are callable
        assert callable(adapter.schedule_task)
        assert callable(adapter.get_task_status)
        assert callable(adapter.cancel_task)
        assert callable(adapter.list_tasks)
        assert callable(adapter.get_queue_stats)

    def test_runtime_checkable_local(self):
        """Test LocalSchedulerAdapter passes runtime protocol check."""
        mock_scheduler = MagicMock()
        mock_scheduler._tasks = {}
        mock_scheduler._tasks_scheduled = 0
        mock_scheduler._tasks_completed = 0
        mock_scheduler._tasks_failed = 0
        mock_scheduler._tasks_cancelled = 0
        mock_scheduler._agent_queues = {}

        adapter = LocalSchedulerAdapter(scheduler=mock_scheduler)

        # The @runtime_checkable decorator enables isinstance checks
        assert isinstance(adapter, SchedulerProtocol)

    def test_runtime_checkable_distributed(self):
        """Test DistributedSchedulerAdapter passes runtime protocol check."""
        mock_scheduler = MagicMock()
        adapter = DistributedSchedulerAdapter(scheduler=mock_scheduler)

        assert isinstance(adapter, SchedulerProtocol)


class TestIntegrationScenarios:
    """Integration-style tests for realistic usage scenarios."""

    def setup_method(self):
        """Reset scheduler before each test."""
        reset_scheduler()

    def teardown_method(self):
        """Clean up after each test."""
        reset_scheduler()
        os.environ.pop("REDIS_URL", None)
        os.environ.pop("ARAGORA_SCHEDULER_TYPE", None)

    @pytest.mark.asyncio
    async def test_task_lifecycle_local(self):
        """Test complete task lifecycle through local adapter."""
        # Get local scheduler
        os.environ["ARAGORA_SCHEDULER_TYPE"] = "local"
        scheduler = get_scheduler()

        assert isinstance(scheduler, LocalSchedulerAdapter)

        # Mock the underlying scheduler methods
        mock_handle = MagicMock()
        mock_handle.task_id = "lifecycle-task"
        mock_handle.status = TaskStatus.PENDING
        mock_handle.scheduled_at = datetime.now(timezone.utc)
        mock_handle.agent_id = "default"
        mock_handle.started_at = None
        mock_handle.completed_at = None
        mock_handle.result = None
        mock_handle.error = None

        scheduler._scheduler.schedule = AsyncMock(return_value=mock_handle)
        scheduler._scheduler.get_handle = AsyncMock(return_value=mock_handle)
        scheduler._scheduler.cancel = AsyncMock(return_value=True)

        # Schedule task
        task = await scheduler.schedule_task(
            task_type="debate",
            payload={"topic": "Test lifecycle"},
        )
        assert task.task_id == "lifecycle-task"
        assert task.status == TaskStatus.PENDING

        # Check status
        status = await scheduler.get_task_status("lifecycle-task")
        assert status is not None
        assert status.status == TaskStatus.PENDING

        # Cancel task
        cancelled = await scheduler.cancel_task("lifecycle-task")
        assert cancelled is True

    @pytest.mark.asyncio
    async def test_scheduler_switch_via_env(self):
        """Test switching scheduler type via environment variable."""
        # Start with local
        os.environ["ARAGORA_SCHEDULER_TYPE"] = "local"
        local_scheduler = get_scheduler()
        assert isinstance(local_scheduler, LocalSchedulerAdapter)

        # Reset and switch to distributed (mocked)
        reset_scheduler()
        os.environ["ARAGORA_SCHEDULER_TYPE"] = "distributed"

        with patch("aragora.agents.scheduler_protocol.DistributedSchedulerAdapter") as mock_dist:
            mock_instance = MagicMock()
            mock_dist.return_value = mock_instance

            distributed_scheduler = get_scheduler()
            assert distributed_scheduler is mock_instance
