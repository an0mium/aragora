"""
Tests for debate_utils module.

Tests:
- StateManager integration via proxy dict
- Debate status updates
- Stale debate cleanup
- Streaming wrapper functionality
- Thread safety
"""

import pytest
import threading
import time
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from aragora.server.debate_utils import (
    get_active_debates,
    get_active_debates_lock,
    update_debate_status,
    cleanup_stale_debates,
    increment_cleanup_counter,
    wrap_agent_for_streaming,
    _ActiveDebatesProxy,
    _DEBATE_TTL_SECONDS,
)
from aragora.server.state import get_state_manager, reset_state_manager


@pytest.fixture(autouse=True)
def reset_state():
    """Reset state manager before each test."""
    reset_state_manager()
    yield
    reset_state_manager()


class TestActiveDebatesProxy:
    """Tests for the _ActiveDebatesProxy backward compatibility layer."""

    def test_proxy_is_dict_like(self):
        """Test proxy behaves like a dict."""
        debates = get_active_debates()
        assert isinstance(debates, _ActiveDebatesProxy)

    def test_setitem_registers_debate(self):
        """Test setting item registers debate with StateManager."""
        debates = get_active_debates()
        debates["test-1"] = {
            "task": "Test task",
            "agents": ["agent1", "agent2"],
            "total_rounds": 3,
        }

        # Verify via StateManager
        state = get_state_manager().get_debate("test-1")
        assert state is not None
        assert state.task == "Test task"
        assert state.agents == ["agent1", "agent2"]

    def test_getitem_returns_dict(self):
        """Test getting item returns debate dict."""
        manager = get_state_manager()
        manager.register_debate("test-2", "Task 2", ["a1"])

        debates = get_active_debates()
        debate = debates["test-2"]

        assert debate["debate_id"] == "test-2"
        assert debate["task"] == "Task 2"
        assert debate["agents"] == ["a1"]

    def test_getitem_raises_keyerror(self):
        """Test getting missing item raises KeyError."""
        debates = get_active_debates()
        with pytest.raises(KeyError):
            _ = debates["nonexistent"]

    def test_delitem_unregisters_debate(self):
        """Test deleting item unregisters debate."""
        manager = get_state_manager()
        manager.register_debate("test-3", "Task 3", ["a1"])

        debates = get_active_debates()
        del debates["test-3"]

        assert manager.get_debate("test-3") is None

    def test_contains_checks_existence(self):
        """Test 'in' operator checks debate existence."""
        manager = get_state_manager()
        manager.register_debate("test-4", "Task 4", ["a1"])

        debates = get_active_debates()
        assert "test-4" in debates
        assert "nonexistent" not in debates

    def test_len_returns_count(self):
        """Test len() returns debate count."""
        manager = get_state_manager()
        debates = get_active_debates()

        assert len(debates) == 0

        manager.register_debate("d1", "T1", ["a"])
        manager.register_debate("d2", "T2", ["b"])

        assert len(debates) == 2

    def test_iter_yields_keys(self):
        """Test iteration yields debate IDs."""
        manager = get_state_manager()
        manager.register_debate("d1", "T1", ["a"])
        manager.register_debate("d2", "T2", ["b"])

        debates = get_active_debates()
        keys = list(debates)

        assert "d1" in keys
        assert "d2" in keys

    def test_get_with_default(self):
        """Test get() returns default for missing."""
        debates = get_active_debates()

        result = debates.get("missing", {"default": True})
        assert result == {"default": True}

    def test_items_returns_pairs(self):
        """Test items() returns key-value pairs."""
        manager = get_state_manager()
        manager.register_debate("d1", "T1", ["a"])

        debates = get_active_debates()
        items = list(debates.items())

        assert len(items) == 1
        assert items[0][0] == "d1"
        assert items[0][1]["task"] == "T1"

    def test_keys_returns_ids(self):
        """Test keys() returns debate IDs."""
        manager = get_state_manager()
        manager.register_debate("d1", "T1", ["a"])
        manager.register_debate("d2", "T2", ["b"])

        debates = get_active_debates()
        keys = list(debates.keys())

        assert "d1" in keys
        assert "d2" in keys

    def test_values_returns_dicts(self):
        """Test values() returns debate dicts."""
        manager = get_state_manager()
        manager.register_debate("d1", "Task One", ["a"])

        debates = get_active_debates()
        values = list(debates.values())

        assert len(values) == 1
        assert values[0]["task"] == "Task One"

    def test_pop_returns_and_removes(self):
        """Test pop() returns and removes debate."""
        manager = get_state_manager()
        manager.register_debate("d1", "T1", ["a"])

        debates = get_active_debates()
        result = debates.pop("d1")

        assert result["debate_id"] == "d1"
        assert manager.get_debate("d1") is None

    def test_pop_with_default(self):
        """Test pop() returns default for missing."""
        debates = get_active_debates()
        result = debates.pop("missing", {"default": True})
        assert result == {"default": True}

    def test_pop_raises_keyerror(self):
        """Test pop() raises KeyError without default."""
        debates = get_active_debates()
        with pytest.raises(KeyError):
            debates.pop("missing")


class TestUpdateDebateStatus:
    """Tests for update_debate_status function."""

    def test_updates_status(self):
        """Test status update works."""
        manager = get_state_manager()
        manager.register_debate("d1", "T1", ["a"])

        update_debate_status("d1", "completed")

        state = manager.get_debate("d1")
        assert state.status == "completed"

    def test_updates_current_round(self):
        """Test current_round update works."""
        manager = get_state_manager()
        manager.register_debate("d1", "T1", ["a"])

        update_debate_status("d1", "running", current_round=2)

        state = manager.get_debate("d1")
        assert state.current_round == 2

    def test_stores_metadata(self):
        """Test additional kwargs stored in metadata."""
        manager = get_state_manager()
        manager.register_debate("d1", "T1", ["a"])

        update_debate_status("d1", "running", custom_field="custom_value")

        state = manager.get_debate("d1")
        assert state.metadata.get("custom_field") == "custom_value"

    def test_records_completion_time(self):
        """Test completion time recorded for completed/error status."""
        manager = get_state_manager()
        manager.register_debate("d1", "T1", ["a"])

        before = time.time()
        update_debate_status("d1", "completed")
        after = time.time()

        state = manager.get_debate("d1")
        completed_at = state.metadata.get("completed_at")
        assert completed_at is not None
        assert before <= completed_at <= after

    def test_noop_for_missing_debate(self):
        """Test no error for missing debate."""
        # Should not raise
        update_debate_status("nonexistent", "completed")


class TestCleanupStaleDebates:
    """Tests for cleanup_stale_debates function."""

    def test_removes_old_completed_debates(self):
        """Test old completed debates are removed."""
        manager = get_state_manager()
        manager.register_debate("d1", "T1", ["a"])

        # Manually set old completion time
        state = manager.get_debate("d1")
        state.status = "completed"
        state.metadata["completed_at"] = time.time() - _DEBATE_TTL_SECONDS - 100

        cleanup_stale_debates()

        assert manager.get_debate("d1") is None

    def test_keeps_recent_completed_debates(self):
        """Test recent completed debates are kept."""
        manager = get_state_manager()
        manager.register_debate("d1", "T1", ["a"])

        state = manager.get_debate("d1")
        state.status = "completed"
        state.metadata["completed_at"] = time.time()  # Just now

        cleanup_stale_debates()

        assert manager.get_debate("d1") is not None

    def test_keeps_running_debates(self):
        """Test running debates are not cleaned up."""
        manager = get_state_manager()
        manager.register_debate("d1", "T1", ["a"])

        # Even with old start_time, running debates should stay
        state = manager.get_debate("d1")
        state.status = "running"

        cleanup_stale_debates()

        assert manager.get_debate("d1") is not None


class TestIncrementCleanupCounter:
    """Tests for increment_cleanup_counter function."""

    def test_returns_false(self):
        """Test always returns False (StateManager handles cleanup)."""
        assert increment_cleanup_counter() is False


class TestGetActiveDebatesLock:
    """Tests for get_active_debates_lock function."""

    def test_returns_lock(self):
        """Test returns a threading.Lock."""
        lock = get_active_debates_lock()
        assert isinstance(lock, type(threading.Lock()))


class TestConcurrentAccess:
    """Tests for thread-safe concurrent access."""

    def test_concurrent_registration(self):
        """Test concurrent debate registration is thread-safe."""
        manager = get_state_manager()
        results = []
        errors = []

        def register(i):
            try:
                manager.register_debate(f"debate-{i}", f"Task {i}", [f"agent-{i}"])
                results.append(i)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=register, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 20
        assert manager.get_active_debate_count() == 20

    def test_concurrent_status_updates(self):
        """Test concurrent status updates are thread-safe."""
        manager = get_state_manager()
        manager.register_debate("d1", "T1", ["a"])
        errors = []

        def update_status(round_num):
            try:
                update_debate_status("d1", "running", current_round=round_num)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=update_status, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # Final round should be one of the values set
        state = manager.get_debate("d1")
        assert state.current_round in range(10)


class TestWrapAgentForStreaming:
    """Tests for wrap_agent_for_streaming function."""

    def test_returns_unchanged_if_no_stream_method(self):
        """Test agent without generate_stream is returned unchanged."""
        agent = Mock()
        agent.name = "test-agent"
        del agent.generate_stream  # Ensure no streaming

        emitter = Mock()
        result = wrap_agent_for_streaming(agent, emitter, "debate-1")

        assert result is agent

    def test_wraps_agent_with_stream_method(self):
        """Test agent with generate_stream gets wrapped."""
        agent = Mock()
        agent.name = "test-agent"
        agent.generate_stream = AsyncMock()
        original_generate = AsyncMock(return_value="original")
        agent.generate = original_generate

        emitter = Mock()
        wrapped = wrap_agent_for_streaming(agent, emitter, "debate-1")

        # The generate method should be replaced (agent is modified in-place)
        # Check that generate is now a different function
        assert wrapped is agent  # Same object
        assert "streaming_generate" in str(wrapped.generate)  # Wrapped function

    @pytest.mark.asyncio
    async def test_streaming_emits_events(self):
        """Test streaming wrapper emits TOKEN_* events."""
        agent = Mock()
        agent.name = "test-agent"

        async def mock_stream(prompt, context=None):
            yield "Hello"
            yield " "
            yield "World"

        agent.generate_stream = mock_stream

        emitter = Mock()
        wrapped = wrap_agent_for_streaming(agent, emitter, "debate-1")

        result = await wrapped.generate("test prompt")

        assert result == "Hello World"

        # Check events emitted
        calls = emitter.emit.call_args_list
        assert len(calls) >= 3  # START + DELTAs + END

        # First should be TOKEN_START
        first_event = calls[0][0][0]
        assert first_event.type.value == "token_start"

        # Last should be TOKEN_END
        last_event = calls[-1][0][0]
        assert last_event.type.value == "token_end"


class TestIntegrationWithStateManager:
    """Integration tests verifying debate_utils works with StateManager."""

    def test_full_debate_lifecycle(self):
        """Test full debate lifecycle through debate_utils."""
        debates = get_active_debates()

        # Create debate via proxy
        debates["lifecycle-test"] = {
            "task": "Lifecycle test task",
            "agents": ["agent1", "agent2"],
            "total_rounds": 3,
        }

        # Verify creation
        assert "lifecycle-test" in debates
        assert len(debates) == 1

        # Update status
        update_debate_status("lifecycle-test", "running", current_round=1)
        state = get_state_manager().get_debate("lifecycle-test")
        assert state.status == "running"
        assert state.current_round == 1

        # Complete debate
        update_debate_status("lifecycle-test", "completed")
        state = get_state_manager().get_debate("lifecycle-test")
        assert state.status == "completed"
        assert "completed_at" in state.metadata

        # Remove debate
        del debates["lifecycle-test"]
        assert "lifecycle-test" not in debates
        assert len(debates) == 0
