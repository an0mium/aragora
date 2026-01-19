"""Tests for stream state management.

Tests the BoundedDebateDict, LoopInstance, and DebateStateManager classes
used for managing streaming state in the Aragora server.
"""

import threading
import time
from datetime import datetime

import pytest


class TestBoundedDebateDict:
    """Test BoundedDebateDict class."""

    def test_basic_operations(self):
        """Test basic dict operations."""
        from aragora.server.stream.state_manager import BoundedDebateDict

        d = BoundedDebateDict(maxsize=5)

        d["a"] = {"value": 1}
        d["b"] = {"value": 2}

        assert "a" in d
        assert "b" in d
        assert d["a"]["value"] == 1
        assert len(d) == 2

    def test_maxsize_eviction(self):
        """Test that oldest entries are evicted when full."""
        from aragora.server.stream.state_manager import BoundedDebateDict

        d = BoundedDebateDict(maxsize=3)

        d["a"] = {"value": 1}
        d["b"] = {"value": 2}
        d["c"] = {"value": 3}

        assert len(d) == 3

        # Adding a new entry should evict "a"
        d["d"] = {"value": 4}

        assert len(d) == 3
        assert "a" not in d
        assert "b" in d
        assert "d" in d

    def test_update_existing_key(self):
        """Test that updating existing key doesn't evict."""
        from aragora.server.stream.state_manager import BoundedDebateDict

        d = BoundedDebateDict(maxsize=2)

        d["a"] = {"value": 1}
        d["b"] = {"value": 2}

        # Update "a" - should not cause eviction
        d["a"] = {"value": 10}

        assert len(d) == 2
        assert d["a"]["value"] == 10
        assert "b" in d

    def test_preserves_order(self):
        """Test OrderedDict ordering is preserved."""
        from aragora.server.stream.state_manager import BoundedDebateDict

        d = BoundedDebateDict(maxsize=10)

        d["c"] = 1
        d["a"] = 2
        d["b"] = 3

        keys = list(d.keys())
        assert keys == ["c", "a", "b"]


class TestLoopInstance:
    """Test LoopInstance dataclass."""

    def test_creation(self):
        """Test creating a LoopInstance."""
        from aragora.server.stream.state_manager import LoopInstance

        now = time.time()
        instance = LoopInstance(
            loop_id="loop_001",
            name="Test Loop",
            started_at=now,
            path="/path/to/loop",
        )

        assert instance.loop_id == "loop_001"
        assert instance.name == "Test Loop"
        assert instance.started_at == now
        assert instance.cycle == 0  # Default
        assert instance.phase == "starting"  # Default
        assert instance.path == "/path/to/loop"

    def test_mutable_fields(self):
        """Test that mutable fields can be updated."""
        from aragora.server.stream.state_manager import LoopInstance

        instance = LoopInstance(
            loop_id="loop_001",
            name="Test Loop",
            started_at=time.time(),
        )

        instance.cycle = 5
        instance.phase = "verification"

        assert instance.cycle == 5
        assert instance.phase == "verification"


class TestDebateStateManager:
    """Test DebateStateManager class."""

    @pytest.fixture
    def manager(self):
        """Create a fresh DebateStateManager."""
        from aragora.server.stream.state_manager import DebateStateManager

        return DebateStateManager()

    def test_register_loop(self, manager):
        """Test registering a new loop."""
        instance = manager.register_loop("loop_001", "Test Loop", "/path")

        assert instance.loop_id == "loop_001"
        assert instance.name == "Test Loop"
        assert instance.path == "/path"
        assert "loop_001" in manager.active_loops

    def test_unregister_loop(self, manager):
        """Test unregistering a loop."""
        manager.register_loop("loop_001", "Test Loop")

        assert manager.unregister_loop("loop_001") is True
        assert "loop_001" not in manager.active_loops

        # Unregistering non-existent loop returns False
        assert manager.unregister_loop("nonexistent") is False

    def test_update_loop_state(self, manager):
        """Test updating loop state."""
        manager.register_loop("loop_001", "Test Loop")

        manager.update_loop_state("loop_001", cycle=3, phase="design")

        instance = manager.active_loops["loop_001"]
        assert instance.cycle == 3
        assert instance.phase == "design"

    def test_update_loop_state_partial(self, manager):
        """Test partial update of loop state."""
        manager.register_loop("loop_001", "Test Loop")

        manager.update_loop_state("loop_001", cycle=5)

        instance = manager.active_loops["loop_001"]
        assert instance.cycle == 5
        assert instance.phase == "starting"  # Unchanged

    def test_get_loop_list(self, manager):
        """Test getting list of active loops."""
        manager.register_loop("loop_001", "Test Loop 1")
        manager.register_loop("loop_002", "Test Loop 2")

        loops = manager.get_loop_list()

        assert len(loops) == 2
        assert any(l["loop_id"] == "loop_001" for l in loops)
        assert any(l["loop_id"] == "loop_002" for l in loops)

    def test_debate_state_operations(self, manager):
        """Test debate state get/set/remove."""
        state = {"round": 3, "agents": ["claude", "gpt-4"]}

        manager.set_debate_state("loop_001", state)

        retrieved = manager.get_debate_state("loop_001")
        assert retrieved == state

        manager.remove_debate_state("loop_001")
        assert manager.get_debate_state("loop_001") is None

    def test_get_debate_state_nonexistent(self, manager):
        """Test getting state for non-existent loop."""
        assert manager.get_debate_state("nonexistent") is None

    def test_should_cleanup_counter(self, manager):
        """Test cleanup counter logic."""
        # Should not trigger cleanup until counter reaches threshold
        for _ in range(99):
            assert manager.should_cleanup() is False

        # 100th call should trigger cleanup
        assert manager.should_cleanup() is True

        # Counter should reset
        assert manager.should_cleanup() is False

    def test_loop_lru_eviction(self, manager):
        """Test LRU eviction when max loops reached."""
        # Set a low max for testing
        manager._MAX_ACTIVE_LOOPS = 3

        manager.register_loop("loop_001", "Loop 1")
        time.sleep(0.01)  # Ensure different access times
        manager.register_loop("loop_002", "Loop 2")
        time.sleep(0.01)
        manager.register_loop("loop_003", "Loop 3")

        # Adding a 4th should evict loop_001 (oldest)
        manager.register_loop("loop_004", "Loop 4")

        assert "loop_001" not in manager.active_loops
        assert "loop_002" in manager.active_loops
        assert "loop_003" in manager.active_loops
        assert "loop_004" in manager.active_loops

    def test_cleanup_stale_entries(self, manager):
        """Test cleaning up stale entries."""
        # Set very short TTL for testing
        manager._ACTIVE_LOOPS_TTL = 0.01
        manager._DEBATE_STATES_TTL = 0.01

        manager.register_loop("loop_001", "Loop 1")
        manager.set_debate_state("loop_001", {"ended": True})

        # Wait for entries to become stale
        time.sleep(0.02)

        cleaned = manager.cleanup_stale_entries()

        # Should have cleaned up both loop and debate state
        assert cleaned >= 1


class TestDebateStateManagerConcurrency:
    """Test thread-safety of DebateStateManager."""

    def test_concurrent_loop_registration(self):
        """Test concurrent loop registration is thread-safe."""
        from aragora.server.stream.state_manager import DebateStateManager

        manager = DebateStateManager()
        errors = []

        def register_loops(start_id: int):
            try:
                for i in range(10):
                    manager.register_loop(f"loop_{start_id}_{i}", f"Loop {start_id}_{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=register_loops, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(manager.active_loops) == 50

    def test_concurrent_state_access(self):
        """Test concurrent debate state access is thread-safe."""
        from aragora.server.stream.state_manager import DebateStateManager

        manager = DebateStateManager()
        errors = []

        def access_state(loop_id: str):
            try:
                for i in range(100):
                    manager.set_debate_state(loop_id, {"value": i})
                    manager.get_debate_state(loop_id)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=access_state, args=(f"loop_{i}",)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


class TestGlobalStateFunctions:
    """Test global state accessor functions."""

    def test_get_active_debates(self):
        """Test get_active_debates returns BoundedDebateDict."""
        from aragora.server.stream.state_manager import (
            BoundedDebateDict,
            get_active_debates,
        )

        debates = get_active_debates()
        assert isinstance(debates, BoundedDebateDict)

    def test_get_stream_state_manager_singleton(self):
        """Test get_stream_state_manager returns singleton."""
        from aragora.server.stream.state_manager import (
            DebateStateManager,
            get_stream_state_manager,
        )

        manager1 = get_stream_state_manager()
        manager2 = get_stream_state_manager()

        assert manager1 is manager2
        assert isinstance(manager1, DebateStateManager)

    def test_cleanup_stale_debates(self):
        """Test cleanup_stale_debates function."""
        from aragora.server.stream.state_manager import (
            cleanup_stale_debates,
            get_active_debates,
            get_active_debates_lock,
        )

        # Should not raise
        cleanup_stale_debates()

    def test_increment_cleanup_counter(self):
        """Test increment_cleanup_counter function."""
        from aragora.server.stream.state_manager import increment_cleanup_counter

        # Should return False until counter reaches 100
        for _ in range(99):
            assert increment_cleanup_counter() is False

        # 100th call should return True
        assert increment_cleanup_counter() is True
