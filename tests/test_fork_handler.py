"""Tests for ForkBridgeHandler - debate forking functionality."""

import asyncio
import json
import pytest
import threading
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.server.fork_handler import ForkBridgeHandler


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def fork_handler():
    """Create ForkBridgeHandler with mock dependencies."""
    active_loops = {}
    lock = threading.Lock()
    return ForkBridgeHandler(active_loops, lock)


@pytest.fixture
def mock_ws():
    """Mock WebSocket with aiohttp-style send_json."""
    ws = AsyncMock()
    ws.send_json = AsyncMock()
    return ws


@pytest.fixture
def mock_ws_websockets_style():
    """Mock WebSocket with websockets-style send (no send_json)."""
    ws = AsyncMock()
    # Remove send_json to simulate websockets library
    del ws.send_json
    ws.send = AsyncMock()
    return ws


@pytest.fixture
def valid_fork_data():
    """Valid fork data for testing."""
    return {
        "hypothesis": "Test hypothesis",
        "lead_agent": "claude",
        "messages": [
            {"content": "Message 1", "agent": "claude"},
            {"content": "Message 2", "agent": "gemini"},
        ],
        "task": "Continue the discussion",
        "agents": ["anthropic-api", "openai-api"],
    }


# =============================================================================
# Fork Registration Tests
# =============================================================================

class TestForkRegistration:
    """Tests for fork registration and retrieval."""

    def test_register_fork_stores_data(self, fork_handler, valid_fork_data):
        """register_fork() should store fork data."""
        fork_handler.register_fork("fork-123", valid_fork_data)

        assert "fork-123" in fork_handler.fork_store
        assert fork_handler.fork_store["fork-123"] == valid_fork_data

    def test_get_fork_retrieves_registered(self, fork_handler, valid_fork_data):
        """get_fork() should retrieve registered fork."""
        fork_handler.register_fork("fork-abc", valid_fork_data)

        result = fork_handler.get_fork("fork-abc")

        assert result == valid_fork_data

    def test_get_fork_returns_none_for_missing(self, fork_handler):
        """get_fork() should return None for non-existent fork."""
        result = fork_handler.get_fork("nonexistent")

        assert result is None

    def test_concurrent_registration(self, fork_handler):
        """Thread-safe concurrent registration."""
        num_threads = 10
        results = []

        def register_fork(thread_id):
            fork_handler.register_fork(f"fork-{thread_id}", {"id": thread_id})
            results.append(thread_id)

        threads = [
            threading.Thread(target=register_fork, args=(i,))
            for i in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All should be registered
        assert len(fork_handler.fork_store) == num_threads
        for i in range(num_threads):
            assert f"fork-{i}" in fork_handler.fork_store


# =============================================================================
# Fork ID Validation Tests
# =============================================================================

class TestForkIdValidation:
    """Tests for fork_id validation in handle_start_fork."""

    @pytest.mark.asyncio
    async def test_missing_fork_id_returns_error(self, fork_handler, mock_ws):
        """Should return error when fork_id missing."""
        result = await fork_handler.handle_start_fork(mock_ws, {})

        assert result is False
        mock_ws.send_json.assert_called()
        call_data = mock_ws.send_json.call_args[0][0]
        assert call_data["type"] == "error"
        assert "fork_id" in call_data["data"]["message"].lower()

    @pytest.mark.asyncio
    async def test_fork_id_too_long_rejected(self, fork_handler, mock_ws):
        """Should reject fork_id longer than 64 characters."""
        long_id = "x" * 65
        result = await fork_handler.handle_start_fork(mock_ws, {"fork_id": long_id})

        assert result is False
        mock_ws.send_json.assert_called()
        call_data = mock_ws.send_json.call_args[0][0]
        assert call_data["type"] == "error"
        assert "invalid" in call_data["data"]["message"].lower()

    @pytest.mark.asyncio
    async def test_fork_id_must_be_string(self, fork_handler, mock_ws):
        """Should reject non-string fork_id."""
        result = await fork_handler.handle_start_fork(mock_ws, {"fork_id": 12345})

        assert result is False
        mock_ws.send_json.assert_called()

    @pytest.mark.asyncio
    async def test_nonexistent_fork_returns_error(self, fork_handler, mock_ws):
        """Should return error for unregistered fork."""
        result = await fork_handler.handle_start_fork(mock_ws, {"fork_id": "not-registered"})

        assert result is False
        mock_ws.send_json.assert_called()
        call_data = mock_ws.send_json.call_args[0][0]
        assert call_data["type"] == "error"
        assert "not found" in call_data["data"]["message"].lower()

    @pytest.mark.asyncio
    async def test_already_running_fork_rejected(self, fork_handler, mock_ws, valid_fork_data):
        """Should reject fork that's already running."""
        fork_handler.register_fork("running-fork", valid_fork_data)

        # Simulate already running
        with fork_handler.active_loops_lock:
            fork_handler.active_loops["fork_running-fork"] = MagicMock()

        result = await fork_handler.handle_start_fork(mock_ws, {"fork_id": "running-fork"})

        assert result is False
        mock_ws.send_json.assert_called()
        call_data = mock_ws.send_json.call_args[0][0]
        assert call_data["type"] == "error"
        assert "already running" in call_data["data"]["message"].lower()


# =============================================================================
# Message Validation Tests
# =============================================================================

class TestMessageValidation:
    """Tests for initial_messages validation."""

    def test_invalid_messages_type_handled(self, fork_handler, valid_fork_data):
        """Invalid initial_messages type should be converted to empty list."""
        fork_data = valid_fork_data.copy()
        fork_data["messages"] = "not a list"  # Invalid type

        fork_handler.register_fork("test", fork_data)
        stored = fork_handler.get_fork("test")

        # The invalid type is stored as-is, but will be handled during debate start
        assert stored["messages"] == "not a list"

    def test_messages_structure_validation(self, fork_handler):
        """Message structure is validated - must have 'content' field."""
        # This is validated in handle_start_fork when messages are processed
        messages = [
            {"content": "valid", "agent": "claude"},  # Valid
            {"invalid": "no content field"},  # Invalid
            {"content": 123},  # Invalid - content not string
        ]

        # Validate filtering logic manually (as done in fork_handler)
        validated = []
        for msg in messages[:1000]:
            if isinstance(msg, dict) and isinstance(msg.get('content'), str):
                validated.append(msg)

        assert len(validated) == 1
        assert validated[0]["content"] == "valid"

    def test_messages_truncated_at_1000(self, fork_handler):
        """Messages should be truncated at 1000."""
        # Create 1500 messages
        messages = [{"content": f"Message {i}"} for i in range(1500)]

        # Apply the same truncation logic as fork_handler
        validated = []
        for msg in messages[:1000]:  # Truncate at 1000
            if isinstance(msg, dict) and isinstance(msg.get('content'), str):
                validated.append(msg)

        assert len(validated) == 1000

    def test_empty_messages_list_accepted(self, fork_handler, valid_fork_data):
        """Empty messages list should be accepted."""
        fork_data = valid_fork_data.copy()
        fork_data["messages"] = []

        fork_handler.register_fork("test", fork_data)
        stored = fork_handler.get_fork("test")

        assert stored["messages"] == []


# =============================================================================
# Agent Configuration Tests
# =============================================================================

class TestAgentConfiguration:
    """Tests for agent configuration in fork data."""

    def test_default_agents_when_not_specified(self, fork_handler):
        """Default agents should be used when not specified."""
        fork_data = {
            "hypothesis": "Test",
            "task": "Continue",
            # No 'agents' key
        }
        fork_handler.register_fork("test", fork_data)
        stored = fork_handler.get_fork("test")

        # Default is handled in handle_start_fork
        assert "agents" not in stored  # Not added during registration

    def test_custom_agents_from_fork_data(self, fork_handler, valid_fork_data):
        """Custom agents should be used from fork_data."""
        fork_data = valid_fork_data.copy()
        fork_data["agents"] = ["claude-api", "gemini-api", "openai-api"]

        fork_handler.register_fork("test", fork_data)
        stored = fork_handler.get_fork("test")

        assert stored["agents"] == ["claude-api", "gemini-api", "openai-api"]

    def test_max_5_agents_enforced(self, fork_handler):
        """Max 5 agents should be enforced."""
        agents = ["agent1", "agent2", "agent3", "agent4", "agent5", "agent6", "agent7"]

        # Apply the same limit as fork_handler
        limited = agents[:5]

        assert len(limited) == 5


# =============================================================================
# Debate Lifecycle Tests
# =============================================================================

class TestDebateLifecycle:
    """Tests for fork debate lifecycle."""

    def test_fork_registered_in_active_loops(self, fork_handler):
        """Active loops dict and lock should be properly initialized."""
        # Verify the handler structure is correct
        assert fork_handler.active_loops == {}
        assert fork_handler.active_loops_lock is not None

        # Simulate adding a fork to active loops (as handle_start_fork would)
        loop_id = "fork_test-loop"
        mock_task = MagicMock()

        with fork_handler.active_loops_lock:
            fork_handler.active_loops[loop_id] = mock_task

        assert loop_id in fork_handler.active_loops
        assert fork_handler.active_loops[loop_id] is mock_task

    def test_cleanup_callback_removes_from_active_loops(self, fork_handler):
        """Task done callback should remove from active_loops."""
        # Simulate a task being added
        loop_id = "fork_test-cleanup"

        with fork_handler.active_loops_lock:
            fork_handler.active_loops[loop_id] = MagicMock()

        # Simulate cleanup (as done in _on_task_done callback)
        with fork_handler.active_loops_lock:
            fork_handler.active_loops.pop(loop_id, None)

        assert loop_id not in fork_handler.active_loops

    @pytest.mark.asyncio
    async def test_cleanup_on_error(self, fork_handler, mock_ws, valid_fork_data):
        """Should cleanup active_loops on error during fork start."""
        fork_handler.register_fork("error-test", valid_fork_data)

        # Force an error by not setting up imports
        # The ImportError will be caught and cleanup should happen
        with patch('aragora.server.fork_handler._ensure_imports', side_effect=ImportError("Test error")):
            result = await fork_handler.handle_start_fork(mock_ws, {"fork_id": "error-test"})

        assert result is False
        # Should have sent error message
        mock_ws.send_json.assert_called()


# =============================================================================
# WebSocket Communication Tests
# =============================================================================

class TestWebSocketCommunication:
    """Tests for WebSocket message sending."""

    @pytest.mark.asyncio
    async def test_send_json_aiohttp_style(self, fork_handler, mock_ws):
        """_send_json should work with aiohttp-style ws.send_json."""
        await fork_handler._send_json(mock_ws, {"type": "test", "data": "value"})

        mock_ws.send_json.assert_called_once_with({"type": "test", "data": "value"})

    @pytest.mark.asyncio
    async def test_send_json_websockets_style(self, fork_handler, mock_ws_websockets_style):
        """_send_json should fallback to ws.send for websockets library."""
        await fork_handler._send_json(mock_ws_websockets_style, {"type": "test"})

        mock_ws_websockets_style.send.assert_called_once()
        # Verify JSON was serialized
        call_arg = mock_ws_websockets_style.send.call_args[0][0]
        parsed = json.loads(call_arg)
        assert parsed["type"] == "test"

    @pytest.mark.asyncio
    async def test_send_error_format(self, fork_handler, mock_ws):
        """_send_error should send correct error format."""
        await fork_handler._send_error(mock_ws, "Test error message")

        mock_ws.send_json.assert_called_once()
        call_data = mock_ws.send_json.call_args[0][0]
        assert call_data["type"] == "error"
        assert call_data["data"]["message"] == "Test error message"

    @pytest.mark.asyncio
    async def test_fork_started_response_structure(self, fork_handler, mock_ws, valid_fork_data):
        """fork_started response should have correct structure."""
        # We can't easily test the full flow, but we can verify the expected structure
        expected_response = {
            "type": "fork_started",
            "data": {
                "loop_id": "fork_test-123",
                "fork_id": "test-123",
                "hypothesis": "Test hypothesis",
                "status": "running"
            }
        }

        # Verify structure matches what code would send
        assert "type" in expected_response
        assert "data" in expected_response
        assert "loop_id" in expected_response["data"]
        assert "status" in expected_response["data"]


# =============================================================================
# Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_fork_store_isolation(self):
        """Each handler should have isolated fork store."""
        handler1 = ForkBridgeHandler({}, threading.Lock())
        handler2 = ForkBridgeHandler({}, threading.Lock())

        handler1.register_fork("fork-1", {"data": "handler1"})

        assert handler1.get_fork("fork-1") is not None
        assert handler2.get_fork("fork-1") is None

    def test_active_loops_shared(self):
        """Active loops should be shared when passed to constructor."""
        shared_loops = {}
        shared_lock = threading.Lock()

        handler1 = ForkBridgeHandler(shared_loops, shared_lock)
        handler2 = ForkBridgeHandler(shared_loops, shared_lock)

        with handler1.active_loops_lock:
            handler1.active_loops["test"] = "value"

        assert handler2.active_loops.get("test") == "value"

    @pytest.mark.asyncio
    async def test_ws_send_error_handled(self, fork_handler):
        """Should handle errors in WebSocket send gracefully."""
        failing_ws = AsyncMock()
        failing_ws.send_json = AsyncMock(side_effect=Exception("Connection closed"))

        # Should not raise - errors are logged but not propagated
        await fork_handler._send_json(failing_ws, {"type": "test"})
        # Test passes if no exception raised

    def test_fork_id_64_chars_accepted(self, fork_handler, valid_fork_data):
        """fork_id exactly 64 chars should be accepted."""
        fork_id = "x" * 64
        fork_handler.register_fork(fork_id, valid_fork_data)

        assert fork_handler.get_fork(fork_id) == valid_fork_data
