"""
Tests for background task management.

Tests the BackgroundTaskManager and default tasks including:
- Memory tier cleanup with pressure monitoring
- Task registration and lifecycle
- Shared memory instance handling
"""

import os
import tempfile
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.background import (
    BackgroundTaskManager,
    get_background_manager,
    setup_default_tasks,
)


@pytest.fixture
def manager():
    """Create a fresh BackgroundTaskManager for testing."""
    return BackgroundTaskManager()


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    os.unlink(path)


class TestBackgroundTaskManager:
    """Test BackgroundTaskManager core functionality."""

    def test_register_task(self, manager):
        """Test registering a task."""
        callback = MagicMock()
        manager.register_task(
            name="test_task",
            interval_seconds=60,
            callback=callback,
            enabled=True,
        )

        assert "test_task" in manager._tasks
        assert manager._tasks["test_task"].interval_seconds == 60
        assert manager._tasks["test_task"].enabled is True

    def test_unregister_task(self, manager):
        """Test unregistering a task."""
        manager.register_task(
            name="to_remove",
            interval_seconds=60,
            callback=lambda: None,
        )
        assert "to_remove" in manager._tasks

        result = manager.unregister_task("to_remove")
        assert result is True
        assert "to_remove" not in manager._tasks

    def test_unregister_nonexistent_task(self, manager):
        """Test unregistering a non-existent task returns False."""
        result = manager.unregister_task("nonexistent")
        assert result is False

    def test_disable_enable_task(self, manager):
        """Test disabling and enabling a task."""
        manager.register_task(
            name="toggle_task",
            interval_seconds=60,
            callback=lambda: None,
            enabled=True,
        )

        manager.enable_task("toggle_task", enabled=False)
        assert manager._tasks["toggle_task"].enabled is False

        manager.enable_task("toggle_task", enabled=True)
        assert manager._tasks["toggle_task"].enabled is True

    def test_get_stats(self, manager):
        """Test getting task statistics."""
        manager.register_task(
            name="stats_task",
            interval_seconds=120,
            callback=lambda: None,
        )

        stats = manager.get_stats()
        assert "tasks" in stats
        assert "stats_task" in stats["tasks"]
        assert stats["tasks"]["stats_task"]["interval_seconds"] == 120
        assert stats["tasks"]["stats_task"]["run_count"] == 0


class TestSetupDefaultTasks:
    """Test setup_default_tasks function."""

    def test_default_tasks_registered(self):
        """Test that default tasks are registered."""
        # Reset global manager
        import aragora.server.background as bg

        bg._background_manager = None

        setup_default_tasks()
        manager = get_background_manager()

        # Check that default tasks are registered
        assert "memory_tier_cleanup" in manager._tasks
        assert "stale_debate_cleanup" in manager._tasks
        assert "circuit_breaker_cleanup" in manager._tasks

    def test_memory_cleanup_with_shared_instance(self, temp_db):
        """Test memory cleanup uses shared instance when provided."""
        from aragora.memory.continuum import ContinuumMemory, MemoryTier

        # Create a memory instance with some data
        memory = ContinuumMemory(db_path=temp_db)
        for i in range(10):
            memory.add(
                id=f"shared_test_{i}",
                content=f"Content {i}",
                tier=MemoryTier.FAST,
            )

        # Reset global manager and setup with shared instance
        import aragora.server.background as bg

        bg._background_manager = None

        setup_default_tasks(memory_instance=memory)
        manager = get_background_manager()

        # Get the cleanup task callback
        cleanup_task = manager._tasks["memory_tier_cleanup"]
        assert cleanup_task is not None

        # The callback should use the shared instance
        # We verify this by checking that pressure check works
        with patch.object(memory, "get_memory_pressure", return_value=0.5):
            # Low pressure - cleanup should be skipped (no error)
            cleanup_task.callback()

    def test_memory_cleanup_with_high_pressure(self, temp_db):
        """Test memory cleanup triggers when pressure is high."""
        from aragora.memory.continuum import ContinuumMemory, MemoryTier

        memory = ContinuumMemory(db_path=temp_db)

        # Reset global manager and setup with shared instance
        import aragora.server.background as bg

        bg._background_manager = None

        setup_default_tasks(memory_instance=memory, pressure_threshold=0.3)
        manager = get_background_manager()
        cleanup_task = manager._tasks["memory_tier_cleanup"]

        # Mock high pressure
        with patch.object(memory, "get_memory_pressure", return_value=0.9):
            with patch.object(
                memory, "cleanup_expired_memories", return_value={"archived": 5, "deleted": 2}
            ) as mock_cleanup:
                cleanup_task.callback()
                mock_cleanup.assert_called_once_with(archive=True)

    def test_memory_cleanup_skipped_below_threshold(self, temp_db):
        """Test memory cleanup is skipped when pressure is below threshold."""
        from aragora.memory.continuum import ContinuumMemory

        memory = ContinuumMemory(db_path=temp_db)

        import aragora.server.background as bg

        bg._background_manager = None

        setup_default_tasks(memory_instance=memory, pressure_threshold=0.8)
        manager = get_background_manager()
        cleanup_task = manager._tasks["memory_tier_cleanup"]

        # Mock low pressure
        with patch.object(memory, "get_memory_pressure", return_value=0.3):
            with patch.object(memory, "cleanup_expired_memories") as mock_cleanup:
                cleanup_task.callback()
                # Should NOT be called because pressure is below threshold
                mock_cleanup.assert_not_called()


class TestBackgroundManagerLifecycle:
    """Test BackgroundTaskManager start/stop lifecycle."""

    def test_start_stop(self, manager):
        """Test starting and stopping the manager."""
        call_count = {"value": 0}

        def counter():
            call_count["value"] += 1

        manager.register_task(
            name="counter_task",
            interval_seconds=0.1,
            callback=counter,
            enabled=True,
            run_on_startup=True,
        )

        manager.start()
        assert manager._running is True
        assert manager._thread is not None
        assert manager._thread.is_alive()

        # Wait a bit for the task to run
        time.sleep(0.3)

        manager.stop()
        assert manager._running is False

        # Task should have run at least once
        assert call_count["value"] >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
