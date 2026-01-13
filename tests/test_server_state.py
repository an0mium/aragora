"""
Tests for server state management (aragora/server/state.py).

Tests thread safety, debate registration, executor management,
and cleanup functionality.
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch

import pytest

from aragora.server.state import DebateState, StateManager, get_state_manager, reset_state_manager


class TestDebateState:
    """Tests for DebateState dataclass."""

    def test_debate_state_creation(self):
        """Test basic DebateState creation."""
        state = DebateState(
            debate_id="test-123",
            task="Test task",
            agents=["agent-1", "agent-2"],
            start_time=time.time(),
        )
        assert state.debate_id == "test-123"
        assert state.task == "Test task"
        assert state.status == "running"
        assert state.current_round == 0
        assert state.total_rounds == 3

    def test_debate_state_to_dict(self):
        """Test DebateState.to_dict() conversion."""
        start = time.time()
        state = DebateState(
            debate_id="test-456",
            task="Convert task",
            agents=["a", "b"],
            start_time=start,
            status="completed",
            current_round=2,
            total_rounds=5,
        )
        result = state.to_dict()

        assert result["debate_id"] == "test-456"
        assert result["task"] == "Convert task"
        assert result["agents"] == ["a", "b"]
        assert result["status"] == "completed"
        assert result["current_round"] == 2
        assert result["total_rounds"] == 5
        assert "elapsed_seconds" in result
        assert "message_count" in result
        assert "subscriber_count" in result


class TestStateManagerDebates:
    """Tests for StateManager debate operations."""

    def setup_method(self):
        """Reset state manager before each test."""
        reset_state_manager()
        self.manager = StateManager()

    def test_register_debate(self):
        """Test registering a new debate."""
        state = self.manager.register_debate(
            debate_id="debate-1",
            task="Test topic",
            agents=["agent-a", "agent-b"],
            total_rounds=5,
        )
        assert state.debate_id == "debate-1"
        assert state.task == "Test topic"
        assert state.agents == ["agent-a", "agent-b"]
        assert state.total_rounds == 5
        assert state.status == "running"

    def test_register_debate_with_metadata(self):
        """Test registering a debate with metadata."""
        state = self.manager.register_debate(
            debate_id="debate-meta",
            task="Meta task",
            agents=["a"],
            metadata={"priority": "high", "source": "api"},
        )
        assert state.metadata["priority"] == "high"
        assert state.metadata["source"] == "api"

    def test_get_debate(self):
        """Test retrieving a registered debate."""
        self.manager.register_debate("d1", "Task 1", ["a"])
        state = self.manager.get_debate("d1")
        assert state is not None
        assert state.debate_id == "d1"

    def test_get_debate_not_found(self):
        """Test retrieving non-existent debate returns None."""
        state = self.manager.get_debate("nonexistent")
        assert state is None

    def test_unregister_debate(self):
        """Test unregistering a debate."""
        self.manager.register_debate("d2", "Task 2", ["a"])
        removed = self.manager.unregister_debate("d2")
        assert removed is not None
        assert removed.debate_id == "d2"
        assert self.manager.get_debate("d2") is None

    def test_unregister_nonexistent_debate(self):
        """Test unregistering non-existent debate returns None."""
        removed = self.manager.unregister_debate("missing")
        assert removed is None

    def test_get_active_debates(self):
        """Test getting all active debates."""
        self.manager.register_debate("d1", "Task 1", ["a"])
        self.manager.register_debate("d2", "Task 2", ["b"])

        active = self.manager.get_active_debates()
        assert len(active) == 2
        assert "d1" in active
        assert "d2" in active

    def test_get_active_debates_returns_copy(self):
        """Test that get_active_debates returns a copy."""
        self.manager.register_debate("d1", "Task", ["a"])
        active = self.manager.get_active_debates()

        # Modifying the copy shouldn't affect internal state
        active["d1"].status = "modified"
        # Note: This tests dict copy, not deep copy of values

        active.pop("d1", None)
        assert self.manager.get_debate("d1") is not None

    def test_get_active_debate_count(self):
        """Test counting active debates."""
        assert self.manager.get_active_debate_count() == 0

        self.manager.register_debate("d1", "Task 1", ["a"])
        assert self.manager.get_active_debate_count() == 1

        self.manager.register_debate("d2", "Task 2", ["b"])
        assert self.manager.get_active_debate_count() == 2

        self.manager.unregister_debate("d1")
        assert self.manager.get_active_debate_count() == 1

    def test_update_debate_status(self):
        """Test updating debate status."""
        self.manager.register_debate("d1", "Task", ["a"])

        success = self.manager.update_debate_status("d1", status="completed")
        assert success
        assert self.manager.get_debate("d1").status == "completed"

    def test_update_debate_round(self):
        """Test updating debate round."""
        self.manager.register_debate("d1", "Task", ["a"])

        success = self.manager.update_debate_status("d1", current_round=3)
        assert success
        assert self.manager.get_debate("d1").current_round == 3

    def test_update_nonexistent_debate(self):
        """Test updating non-existent debate returns False."""
        success = self.manager.update_debate_status("missing", status="failed")
        assert not success


class TestStateManagerThreadSafety:
    """Tests for StateManager thread safety."""

    def setup_method(self):
        """Reset state manager before each test."""
        reset_state_manager()
        self.manager = StateManager()

    def test_concurrent_registration(self):
        """Test concurrent debate registration."""
        num_threads = 20
        results = []
        errors = []

        def register_debate(i):
            try:
                state = self.manager.register_debate(
                    f"debate-{i}",
                    f"Task {i}",
                    [f"agent-{i}"],
                )
                results.append(state.debate_id)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=register_debate, args=(i,)) for i in range(num_threads)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == num_threads
        assert self.manager.get_active_debate_count() == num_threads

    def test_concurrent_read_write(self):
        """Test concurrent reads and writes."""
        num_writers = 5
        num_readers = 10
        results = {"reads": [], "writes": [], "errors": []}

        def writer(i):
            try:
                self.manager.register_debate(f"w-{i}", f"Task {i}", ["a"])
                results["writes"].append(i)
            except Exception as e:
                results["errors"].append(f"write: {e}")

        def reader():
            try:
                for _ in range(10):
                    count = self.manager.get_active_debate_count()
                    results["reads"].append(count)
                    time.sleep(0.001)
            except Exception as e:
                results["errors"].append(f"read: {e}")

        threads = []
        threads.extend([threading.Thread(target=writer, args=(i,)) for i in range(num_writers)])
        threads.extend([threading.Thread(target=reader) for _ in range(num_readers)])

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results["errors"]) == 0
        assert len(results["writes"]) == num_writers


class TestStateManagerExecutor:
    """Tests for StateManager executor management."""

    def setup_method(self):
        """Reset state manager before each test."""
        reset_state_manager()
        self.manager = StateManager()

    def teardown_method(self):
        """Shutdown executor after each test."""
        self.manager.shutdown_executor()

    def test_get_executor_creates_executor(self):
        """Test that get_executor creates an executor."""
        executor = self.manager.get_executor()
        assert executor is not None
        assert isinstance(executor, ThreadPoolExecutor)

    def test_get_executor_reuses_executor(self):
        """Test that get_executor returns same executor."""
        executor1 = self.manager.get_executor()
        executor2 = self.manager.get_executor()
        assert executor1 is executor2

    def test_get_executor_with_max_workers(self):
        """Test get_executor respects max_workers."""
        executor = self.manager.get_executor(max_workers=8)
        assert executor is not None
        # Can't directly check max_workers, but can submit tasks
        future = executor.submit(lambda: 42)
        assert future.result() == 42

    def test_shutdown_executor(self):
        """Test executor shutdown."""
        executor = self.manager.get_executor()
        assert executor is not None

        self.manager.shutdown_executor()
        # After shutdown, getting executor should create a new one
        executor2 = self.manager.get_executor()
        assert executor2 is not executor


class TestStateManagerCleanup:
    """Tests for StateManager cleanup functionality."""

    def setup_method(self):
        """Reset state manager before each test."""
        reset_state_manager()
        self.manager = StateManager()
        self.manager._cleanup_interval = 2  # Trigger cleanup every 2 unregistrations

    def test_cleanup_stale_debates(self):
        """Test cleanup of stale debates."""
        # Register a debate and make it stale
        self.manager.register_debate("stale-1", "Task", ["a"])
        state = self.manager.get_debate("stale-1")
        # Artificially age the debate
        state.start_time = time.time() - 4000  # ~1 hour old

        # Cleanup should remove stale debates (returns count of removed)
        removed_count = self.manager._cleanup_stale_debates(max_age_seconds=3600)
        assert removed_count == 1
        assert self.manager.get_debate("stale-1") is None

    def test_cleanup_keeps_recent_debates(self):
        """Test cleanup keeps recent debates."""
        self.manager.register_debate("recent", "Task", ["a"])

        removed_count = self.manager._cleanup_stale_debates(max_age_seconds=3600)
        assert removed_count == 0
        assert self.manager.get_debate("recent") is not None


class TestStateManagerSingleton:
    """Tests for StateManager singleton pattern."""

    def test_get_state_manager_returns_singleton(self):
        """Test get_state_manager returns same instance."""
        reset_state_manager()
        manager1 = get_state_manager()
        manager2 = get_state_manager()
        assert manager1 is manager2

    def test_reset_state_manager(self):
        """Test reset_state_manager creates new instance."""
        manager1 = get_state_manager()
        manager1.register_debate("d1", "Task", ["a"])

        reset_state_manager()
        manager2 = get_state_manager()

        assert manager1 is not manager2
        assert manager2.get_active_debate_count() == 0


class TestStateManagerUptime:
    """Tests for StateManager uptime tracking."""

    def setup_method(self):
        """Reset state manager before each test."""
        reset_state_manager()
        self.manager = StateManager()

    def test_server_start_time(self):
        """Test server_start_time property."""
        now = time.time()
        assert abs(self.manager.server_start_time - now) < 1.0

    def test_uptime_seconds(self):
        """Test uptime_seconds property."""
        time.sleep(0.1)
        uptime = self.manager.uptime_seconds
        assert uptime >= 0.1
        assert uptime < 2.0  # Shouldn't take more than 2 seconds
