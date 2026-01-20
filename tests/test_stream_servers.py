"""
Tests for stream/servers.py - WebSocket streaming server.

Tests cover:
- Token extraction from WebSocket
- WebSocket authentication
- Audience payload validation
- Loop registration/unregistration
- Debate state management
- Event broadcasting
"""

import pytest
import asyncio
import json
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from queue import Queue


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_websocket():
    """Create a mock WebSocket connection."""
    ws = Mock()
    ws.request = Mock()
    ws.request.headers = {}
    return ws


@pytest.fixture
def stream_server():
    """Create a DebateStreamServer instance for testing."""
    from aragora.server.stream.debate_stream_server import DebateStreamServer

    with patch("aragora.server.stream.debate_stream_server.auth_config") as mock_auth:
        mock_auth.enabled = False
        server = DebateStreamServer(host="localhost", port=8765)
        yield server


@pytest.fixture
def stream_server_with_auth():
    """Create a DebateStreamServer with auth enabled."""
    from aragora.server.stream.debate_stream_server import DebateStreamServer

    with patch("aragora.server.stream.debate_stream_server.auth_config") as mock_auth:
        mock_auth.enabled = True
        mock_auth.validate_token = Mock(return_value=True)
        server = DebateStreamServer(host="localhost", port=8765)
        server._auth_config = mock_auth
        yield server, mock_auth


# =============================================================================
# Token Extraction Tests
# =============================================================================


class TestExtractWsToken:
    """Tests for _extract_ws_token method."""

    def test_extracts_bearer_token(self, stream_server, mock_websocket):
        """Should extract Bearer token from Authorization header."""
        mock_websocket.request.headers = {"Authorization": "Bearer test-token-123"}

        result = stream_server._extract_ws_token(mock_websocket)
        assert result == "test-token-123"

    def test_returns_none_for_missing_header(self, stream_server, mock_websocket):
        """Should return None when Authorization header is missing."""
        mock_websocket.request.headers = {}

        result = stream_server._extract_ws_token(mock_websocket)
        assert result is None

    def test_returns_none_for_non_bearer(self, stream_server, mock_websocket):
        """Should return None for non-Bearer auth schemes."""
        mock_websocket.request.headers = {"Authorization": "Basic dXNlcjpwYXNz"}

        result = stream_server._extract_ws_token(mock_websocket)
        assert result is None

    def test_handles_legacy_websocket_api(self, stream_server):
        """Should handle older websockets API with request_headers."""
        mock_ws = Mock(spec=[])
        mock_ws.request_headers = {"Authorization": "Bearer legacy-token"}

        result = stream_server._extract_ws_token(mock_ws)
        assert result == "legacy-token"

    def test_handles_missing_request_attribute(self, stream_server):
        """Should return None when websocket has no request attribute."""
        mock_ws = Mock(spec=[])  # No request or request_headers

        result = stream_server._extract_ws_token(mock_ws)
        assert result is None


# =============================================================================
# WebSocket Auth Tests
# =============================================================================


class TestValidateWsAuth:
    """Tests for _validate_ws_auth method."""

    def test_returns_true_when_auth_disabled(self, stream_server, mock_websocket):
        """Should return True when auth is disabled."""
        with patch("aragora.server.stream.servers.auth_config") as mock_auth:
            mock_auth.enabled = False

            result = stream_server._validate_ws_auth(mock_websocket)
            assert result is True

    def test_returns_false_when_no_token(self, mock_websocket):
        """Should return False when auth enabled but no token."""
        from aragora.server.stream.debate_stream_server import DebateStreamServer

        with patch("aragora.server.stream.debate_stream_server.auth_config") as mock_auth:
            mock_auth.enabled = True
            server = DebateStreamServer(host="localhost", port=8765)

            mock_websocket.request.headers = {}
            result = server._validate_ws_auth(mock_websocket)
            assert result is False

    def test_validates_token_with_auth_config(self, mock_websocket):
        """Should call auth_config.validate_token when token present."""
        from aragora.server.stream.debate_stream_server import DebateStreamServer

        with patch("aragora.server.stream.debate_stream_server.auth_config") as mock_auth:
            mock_auth.enabled = True
            mock_auth.validate_token = Mock(return_value=True)
            server = DebateStreamServer(host="localhost", port=8765)

            mock_websocket.request.headers = {"Authorization": "Bearer valid-token"}
            result = server._validate_ws_auth(mock_websocket, loop_id="test-loop")

            assert result is True
            mock_auth.validate_token.assert_called_once_with("valid-token", "test-loop")


# =============================================================================
# Audience Payload Validation Tests
# =============================================================================


class TestValidateAudiencePayload:
    """Tests for _validate_audience_payload method."""

    def test_valid_payload(self, stream_server):
        """Should return payload for valid input."""
        data = {"payload": {"vote": "up", "agent": "claude"}}

        payload, error = stream_server._validate_audience_payload(data)
        assert payload == {"vote": "up", "agent": "claude"}
        assert error is None

    def test_missing_payload_key(self, stream_server):
        """Should return empty dict for missing payload key."""
        data = {}

        payload, error = stream_server._validate_audience_payload(data)
        assert payload == {}
        assert error is None

    def test_non_dict_payload(self, stream_server):
        """Should return error for non-dict payload."""
        data = {"payload": "not a dict"}

        payload, error = stream_server._validate_audience_payload(data)
        assert payload is None
        assert error == "Invalid payload format"

    def test_payload_too_large(self, stream_server):
        """Should return error for payload exceeding 10KB."""
        data = {"payload": {"data": "x" * 20000}}

        payload, error = stream_server._validate_audience_payload(data)
        assert payload is None
        assert "too large" in error

    def test_non_serializable_payload(self, stream_server):
        """Should return error for non-JSON-serializable payload."""
        # Create a payload with non-serializable content
        data = {"payload": {"func": lambda x: x}}

        payload, error = stream_server._validate_audience_payload(data)
        assert payload is None
        assert error == "Invalid payload structure"


# =============================================================================
# Loop Registration Tests
# =============================================================================


class TestLoopRegistration:
    """Tests for loop registration/unregistration."""

    def test_register_loop(self, stream_server):
        """Should register a new loop as LoopInstance."""
        stream_server.register_loop("loop-123", "Test Loop", "/path/to/loop")

        assert "loop-123" in stream_server.active_loops
        # active_loops stores LoopInstance objects, not dicts
        assert stream_server.active_loops["loop-123"].name == "Test Loop"
        assert stream_server.active_loops["loop-123"].path == "/path/to/loop"

    def test_register_loop_updates_existing(self, stream_server):
        """Should update existing loop registration."""
        stream_server.register_loop("loop-123", "Old Name")
        stream_server.register_loop("loop-123", "New Name")

        # LoopInstance is replaced, not updated
        assert stream_server.active_loops["loop-123"].name == "New Name"

    def test_unregister_loop(self, stream_server):
        """Should unregister a loop."""
        stream_server.register_loop("loop-123", "Test Loop")
        stream_server.unregister_loop("loop-123")

        assert "loop-123" not in stream_server.active_loops

    def test_unregister_nonexistent_loop(self, stream_server):
        """Should handle unregistering nonexistent loop gracefully."""
        stream_server.unregister_loop("nonexistent")
        # Should not raise

    def test_update_loop_state(self, stream_server):
        """Should update loop state via LoopInstance attributes."""
        stream_server.register_loop("loop-123", "Test Loop")
        stream_server.update_loop_state("loop-123", cycle=5, phase="debate")

        # Access via LoopInstance attributes
        assert stream_server.active_loops["loop-123"].cycle == 5
        assert stream_server.active_loops["loop-123"].phase == "debate"

    def test_get_loop_list(self, stream_server):
        """Should return list of active loops as dicts."""
        stream_server.register_loop("loop-1", "Loop One")
        stream_server.register_loop("loop-2", "Loop Two")

        loops = stream_server.get_loop_list()

        assert len(loops) == 2
        # get_loop_list returns dicts with 'loop_id' key (not 'id')
        loop_ids = [loop["loop_id"] for loop in loops]
        assert "loop-1" in loop_ids
        assert "loop-2" in loop_ids


# =============================================================================
# Debate State Tests
# =============================================================================


class TestDebateState:
    """Tests for debate state management."""

    def test_update_debate_state_on_start(self, stream_server):
        """Should create new state on DEBATE_START event."""
        from aragora.server.stream import StreamEvent, StreamEventType

        event = StreamEvent(
            type=StreamEventType.DEBATE_START,
            loop_id="loop-123",
            data={"task": "Test topic", "agents": ["claude", "gpt4"]},
        )

        stream_server._update_debate_state(event)

        assert "loop-123" in stream_server.debate_states
        state = stream_server.debate_states["loop-123"]
        assert state["task"] == "Test topic"
        assert "claude" in state["agents"]

    def test_update_debate_state_on_end(self, stream_server):
        """Should mark debate as ended on DEBATE_END event."""
        from aragora.server.stream import StreamEvent, StreamEventType

        # First start the debate
        start_event = StreamEvent(
            type=StreamEventType.DEBATE_START,
            loop_id="loop-123",
            data={"task": "Test", "agents": ["claude"]},
        )
        stream_server._update_debate_state(start_event)

        # Then end it
        end_event = StreamEvent(
            type=StreamEventType.DEBATE_END,
            loop_id="loop-123",
            data={"duration": 10.5, "rounds": 3},
        )
        stream_server._update_debate_state(end_event)

        assert stream_server.debate_states["loop-123"]["ended"] is True


# =============================================================================
# Broadcast Tests
# =============================================================================


class TestBroadcast:
    """Tests for event broadcasting."""

    @pytest.mark.asyncio
    async def test_broadcast_sends_to_clients(self, stream_server):
        """Should send event to connected clients."""
        from aragora.server.stream import StreamEvent, StreamEventType

        event = StreamEvent(
            type=StreamEventType.AGENT_MESSAGE,
            loop_id="loop-123",
            data={"message": "Hello"},
        )

        # Add a mock client
        mock_client = AsyncMock()
        stream_server.clients.add(mock_client)

        await stream_server.broadcast(event)

        # Client should have received the message
        mock_client.send.assert_called_once()
        sent_message = mock_client.send.call_args[0][0]
        # Event type is serialized as lowercase
        assert "agent_message" in sent_message

    @pytest.mark.asyncio
    async def test_broadcast_batch_sends_array(self, stream_server):
        """Should send multiple events as JSON array."""
        from aragora.server.stream import StreamEvent, StreamEventType

        events = [
            StreamEvent(type=StreamEventType.AGENT_MESSAGE, loop_id="loop-1", data={"msg": "1"}),
            StreamEvent(type=StreamEventType.AGENT_MESSAGE, loop_id="loop-1", data={"msg": "2"}),
            StreamEvent(type=StreamEventType.AGENT_MESSAGE, loop_id="loop-1", data={"msg": "3"}),
        ]

        # Add a mock client
        mock_client = AsyncMock()
        stream_server.clients.add(mock_client)

        await stream_server.broadcast_batch(events)

        # Client should have received a JSON array
        mock_client.send.assert_called_once()
        sent_message = mock_client.send.call_args[0][0]
        parsed = json.loads(sent_message)
        assert isinstance(parsed, list)
        assert len(parsed) == 3

    @pytest.mark.asyncio
    async def test_broadcast_handles_disconnected_client(self, stream_server):
        """Should remove disconnected clients during broadcast."""
        from aragora.server.stream import StreamEvent, StreamEventType

        event = StreamEvent(
            type=StreamEventType.AGENT_MESSAGE,
            loop_id="loop-123",
            data={"message": "Hello"},
        )

        # Add a mock client that raises exception
        mock_client = AsyncMock()
        mock_client.send.side_effect = Exception("Connection closed")
        stream_server.clients.add(mock_client)

        await stream_server.broadcast(event)

        # Client should be removed from clients set
        assert mock_client not in stream_server.clients


# =============================================================================
# Cleanup Tests
# =============================================================================


class TestCleanup:
    """Tests for stale entry cleanup."""

    def test_cleanup_stale_entries(self, stream_server):
        """Should cleanup stale entries based on last access time and TTL."""
        import time
        from aragora.server.stream.state_manager import LoopInstance

        # Simulate old entry by directly setting timestamps
        old_loop = LoopInstance(loop_id="old-loop", name="Old", started_at=time.time() - 100000)
        stream_server.active_loops["old-loop"] = old_loop
        stream_server._active_loops_last_access["old-loop"] = time.time() - 100000  # Way past TTL

        # Add a current entry via register_loop (sets proper timestamp)
        stream_server.register_loop("current-loop", "Current")

        stream_server._cleanup_stale_entries()

        # Old loop should be cleaned up (TTL exceeded)
        assert "old-loop" not in stream_server.active_loops
        # Current loop should remain
        assert "current-loop" in stream_server.active_loops


# =============================================================================
# Server Lifecycle Tests
# =============================================================================


class TestServerLifecycle:
    """Tests for server start/stop."""

    def test_server_initialization(self, stream_server):
        """Should initialize with correct attributes."""
        assert stream_server.host == "localhost"
        assert stream_server.port == 8765
        assert stream_server.active_loops == {}
        assert stream_server.debate_states == {}

    def test_stop_server(self, stream_server):
        """Should set running flag to False on stop."""
        stream_server._running = True
        stream_server.stop()

        assert stream_server._running is False

    def test_emitter_property(self, stream_server):
        """Should return emitter instance."""
        emitter = stream_server.emitter
        assert emitter is not None


# =============================================================================
# Origin Extraction Tests
# =============================================================================


class TestExtractWsOrigin:
    """Tests for _extract_ws_origin method."""

    def test_extracts_origin_header(self, stream_server, mock_websocket):
        """Should extract Origin header from WebSocket."""
        mock_websocket.request.headers = {"Origin": "https://example.com"}

        result = stream_server._extract_ws_origin(mock_websocket)
        assert result == "https://example.com"

    def test_returns_empty_for_missing_origin(self, stream_server, mock_websocket):
        """Should return empty string for missing Origin."""
        mock_websocket.request.headers = {}

        result = stream_server._extract_ws_origin(mock_websocket)
        assert result == ""

    def test_handles_exception_gracefully(self, stream_server):
        """Should return empty string when websocket lacks required attributes."""
        # Create a mock that has neither request nor request_headers
        mock_ws = Mock(spec=[])

        result = stream_server._extract_ws_origin(mock_ws)
        assert result == ""

    def test_uses_legacy_request_headers(self, stream_server):
        """Should use request_headers for older websockets API."""
        mock_ws = Mock(spec=[])  # No .request attribute
        mock_ws.request_headers = {"Origin": "https://legacy.example.com"}

        result = stream_server._extract_ws_origin(mock_ws)
        assert result == "https://legacy.example.com"
