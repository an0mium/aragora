"""
Integration tests for Control Plane WebSocket streaming.

Tests:
- ControlPlaneStreamServer initialization and lifecycle
- Event emission and broadcasting
- Client connection handling
- Event serialization
"""

import asyncio
import json
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from aragora.server.stream.control_plane_stream import (
    ControlPlaneStreamServer,
    ControlPlaneEventType,
    ControlPlaneEvent,
)


# ===========================================================================
# Test Fixtures
# ===========================================================================


@pytest.fixture
def stream_server():
    """Create a ControlPlaneStreamServer instance for testing."""
    return ControlPlaneStreamServer(port=0, host="127.0.0.1")


@pytest.fixture
def mock_websocket():
    """Create a mock WebSocket connection."""
    ws = AsyncMock()
    ws.send = AsyncMock()
    ws.close = AsyncMock()
    return ws


# ===========================================================================
# Test ControlPlaneEvent
# ===========================================================================


class TestControlPlaneEvent:
    """Tests for ControlPlaneEvent dataclass."""

    def test_event_to_dict(self):
        """Should convert event to dictionary format."""
        event = ControlPlaneEvent(
            event_type=ControlPlaneEventType.AGENT_REGISTERED,
            data={"agent_id": "agent-1", "capabilities": ["debate"]},
        )
        result = event.to_dict()

        assert result["type"] == "agent_registered"
        assert "timestamp" in result
        assert result["data"]["agent_id"] == "agent-1"
        assert result["data"]["capabilities"] == ["debate"]

    def test_event_to_json(self):
        """Should serialize event to JSON."""
        event = ControlPlaneEvent(
            event_type=ControlPlaneEventType.TASK_COMPLETED,
            data={"task_id": "task-1", "agent_id": "agent-1"},
        )
        json_str = event.to_json()

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed["type"] == "task_completed"
        assert parsed["data"]["task_id"] == "task-1"

    def test_event_with_default_timestamp(self):
        """Should have timestamp automatically set."""
        event = ControlPlaneEvent(
            event_type=ControlPlaneEventType.CONNECTED,
            data={"message": "Hello"},
        )
        assert event.timestamp > 0
        assert isinstance(event.timestamp, float)


# ===========================================================================
# Test ControlPlaneEventType
# ===========================================================================


class TestControlPlaneEventType:
    """Tests for event type enum."""

    def test_agent_events_exist(self):
        """Should have all agent-related event types."""
        assert ControlPlaneEventType.AGENT_REGISTERED.value == "agent_registered"
        assert ControlPlaneEventType.AGENT_UNREGISTERED.value == "agent_unregistered"
        assert ControlPlaneEventType.AGENT_STATUS_CHANGED.value == "agent_status_changed"
        assert ControlPlaneEventType.AGENT_HEARTBEAT.value == "agent_heartbeat"
        assert ControlPlaneEventType.AGENT_TIMEOUT.value == "agent_timeout"

    def test_task_events_exist(self):
        """Should have all task-related event types."""
        assert ControlPlaneEventType.TASK_SUBMITTED.value == "task_submitted"
        assert ControlPlaneEventType.TASK_CLAIMED.value == "task_claimed"
        assert ControlPlaneEventType.TASK_STARTED.value == "task_started"
        assert ControlPlaneEventType.TASK_COMPLETED.value == "task_completed"
        assert ControlPlaneEventType.TASK_FAILED.value == "task_failed"
        assert ControlPlaneEventType.TASK_CANCELLED.value == "task_cancelled"
        assert ControlPlaneEventType.TASK_DEAD_LETTERED.value == "task_dead_lettered"

    def test_system_events_exist(self):
        """Should have all system-related event types."""
        assert ControlPlaneEventType.HEALTH_UPDATE.value == "health_update"
        assert ControlPlaneEventType.METRICS_UPDATE.value == "metrics_update"
        assert ControlPlaneEventType.SCHEDULER_STATS.value == "scheduler_stats"
        assert ControlPlaneEventType.ERROR.value == "error"


# ===========================================================================
# Test ControlPlaneStreamServer Initialization
# ===========================================================================


class TestControlPlaneStreamServerInit:
    """Tests for server initialization."""

    def test_init_with_defaults(self):
        """Should initialize with default port and host."""
        server = ControlPlaneStreamServer()
        assert server.port == 8766
        assert server.host == "0.0.0.0"

    def test_init_with_custom_port(self):
        """Should accept custom port."""
        server = ControlPlaneStreamServer(port=9999)
        assert server.port == 9999

    def test_init_with_custom_host(self):
        """Should accept custom host."""
        server = ControlPlaneStreamServer(host="localhost")
        assert server.host == "localhost"

    def test_initial_state(self, stream_server):
        """Should have correct initial state."""
        assert stream_server._clients == set()
        assert stream_server._running is False
        assert stream_server._server is None


# ===========================================================================
# Test Client Management
# ===========================================================================


class TestClientManagement:
    """Tests for client registration and unregistration."""

    @pytest.mark.asyncio
    async def test_register_client(self, stream_server, mock_websocket):
        """Should register a new client."""
        await stream_server._register_client(mock_websocket)
        assert mock_websocket in stream_server._clients
        assert stream_server.client_count == 1

    @pytest.mark.asyncio
    async def test_unregister_client(self, stream_server, mock_websocket):
        """Should unregister a client."""
        await stream_server._register_client(mock_websocket)
        await stream_server._unregister_client(mock_websocket)
        assert mock_websocket not in stream_server._clients
        assert stream_server.client_count == 0

    @pytest.mark.asyncio
    async def test_multiple_clients(self, stream_server):
        """Should handle multiple clients."""
        ws1 = AsyncMock()
        ws2 = AsyncMock()
        ws3 = AsyncMock()

        await stream_server._register_client(ws1)
        await stream_server._register_client(ws2)
        await stream_server._register_client(ws3)

        assert stream_server.client_count == 3

        await stream_server._unregister_client(ws2)
        assert stream_server.client_count == 2
        assert ws1 in stream_server._clients
        assert ws3 in stream_server._clients


# ===========================================================================
# Test Message Handling
# ===========================================================================


class TestMessageHandling:
    """Tests for handling incoming WebSocket messages."""

    @pytest.mark.asyncio
    async def test_handle_ping_message(self, stream_server, mock_websocket):
        """Should respond to ping with pong."""
        message = json.dumps({"type": "ping"})
        await stream_server._handle_message(mock_websocket, message)

        mock_websocket.send.assert_called_once()
        sent_data = json.loads(mock_websocket.send.call_args[0][0])
        assert sent_data["type"] == "pong"
        assert "timestamp" in sent_data

    @pytest.mark.asyncio
    async def test_handle_invalid_json(self, stream_server, mock_websocket):
        """Should handle invalid JSON gracefully."""
        # Should not raise, just log warning
        await stream_server._handle_message(mock_websocket, "not valid json {")
        mock_websocket.send.assert_not_called()


# ===========================================================================
# Test Broadcasting
# ===========================================================================


class TestBroadcasting:
    """Tests for event broadcasting."""

    @pytest.mark.asyncio
    async def test_broadcast_to_no_clients(self, stream_server):
        """Should handle broadcasting with no clients."""
        event = ControlPlaneEvent(
            event_type=ControlPlaneEventType.AGENT_REGISTERED,
            data={"agent_id": "agent-1"},
        )
        # Should not raise
        await stream_server.broadcast(event)

    @pytest.mark.asyncio
    async def test_broadcast_to_single_client(self, stream_server, mock_websocket):
        """Should broadcast to a single client."""
        await stream_server._register_client(mock_websocket)

        event = ControlPlaneEvent(
            event_type=ControlPlaneEventType.TASK_COMPLETED,
            data={"task_id": "task-1"},
        )
        await stream_server.broadcast(event)

        mock_websocket.send.assert_called_once()
        sent_data = json.loads(mock_websocket.send.call_args[0][0])
        assert sent_data["type"] == "task_completed"

    @pytest.mark.asyncio
    async def test_broadcast_to_multiple_clients(self, stream_server):
        """Should broadcast to all connected clients."""
        ws1 = AsyncMock()
        ws2 = AsyncMock()

        await stream_server._register_client(ws1)
        await stream_server._register_client(ws2)

        event = ControlPlaneEvent(
            event_type=ControlPlaneEventType.HEALTH_UPDATE,
            data={"status": "healthy"},
        )
        await stream_server.broadcast(event)

        ws1.send.assert_called_once()
        ws2.send.assert_called_once()


# ===========================================================================
# Test High-Level Event Emission
# ===========================================================================


class TestEventEmission:
    """Tests for high-level event emission methods."""

    @pytest.mark.asyncio
    async def test_emit_agent_registered(self, stream_server, mock_websocket):
        """Should emit agent registered event."""
        await stream_server._register_client(mock_websocket)

        await stream_server.emit_agent_registered(
            agent_id="agent-1",
            capabilities=["debate", "analysis"],
            model="gpt-4",
            provider="openai",
        )

        sent_data = json.loads(mock_websocket.send.call_args[0][0])
        assert sent_data["type"] == "agent_registered"
        assert sent_data["data"]["agent_id"] == "agent-1"
        assert sent_data["data"]["capabilities"] == ["debate", "analysis"]
        assert sent_data["data"]["model"] == "gpt-4"
        assert sent_data["data"]["provider"] == "openai"

    @pytest.mark.asyncio
    async def test_emit_agent_unregistered(self, stream_server, mock_websocket):
        """Should emit agent unregistered event."""
        await stream_server._register_client(mock_websocket)

        await stream_server.emit_agent_unregistered(
            agent_id="agent-1",
            reason="timeout",
        )

        sent_data = json.loads(mock_websocket.send.call_args[0][0])
        assert sent_data["type"] == "agent_unregistered"
        assert sent_data["data"]["agent_id"] == "agent-1"
        assert sent_data["data"]["reason"] == "timeout"

    @pytest.mark.asyncio
    async def test_emit_agent_status_changed(self, stream_server, mock_websocket):
        """Should emit agent status changed event."""
        await stream_server._register_client(mock_websocket)

        await stream_server.emit_agent_status_changed(
            agent_id="agent-1",
            old_status="idle",
            new_status="busy",
        )

        sent_data = json.loads(mock_websocket.send.call_args[0][0])
        assert sent_data["type"] == "agent_status_changed"
        assert sent_data["data"]["agent_id"] == "agent-1"
        assert sent_data["data"]["old_status"] == "idle"
        assert sent_data["data"]["new_status"] == "busy"

    @pytest.mark.asyncio
    async def test_emit_task_submitted(self, stream_server, mock_websocket):
        """Should emit task submitted event."""
        await stream_server._register_client(mock_websocket)

        await stream_server.emit_task_submitted(
            task_id="task-1",
            task_type="debate",
            priority="high",
            required_capabilities=["debate"],
        )

        sent_data = json.loads(mock_websocket.send.call_args[0][0])
        assert sent_data["type"] == "task_submitted"
        assert sent_data["data"]["task_id"] == "task-1"
        assert sent_data["data"]["task_type"] == "debate"
        assert sent_data["data"]["priority"] == "high"

    @pytest.mark.asyncio
    async def test_emit_task_claimed(self, stream_server, mock_websocket):
        """Should emit task claimed event."""
        await stream_server._register_client(mock_websocket)

        await stream_server.emit_task_claimed(
            task_id="task-1",
            agent_id="agent-1",
        )

        sent_data = json.loads(mock_websocket.send.call_args[0][0])
        assert sent_data["type"] == "task_claimed"
        assert sent_data["data"]["task_id"] == "task-1"
        assert sent_data["data"]["agent_id"] == "agent-1"

    @pytest.mark.asyncio
    async def test_emit_task_completed(self, stream_server, mock_websocket):
        """Should emit task completed event."""
        await stream_server._register_client(mock_websocket)

        await stream_server.emit_task_completed(
            task_id="task-1",
            agent_id="agent-1",
            result={"success": True},
        )

        sent_data = json.loads(mock_websocket.send.call_args[0][0])
        assert sent_data["type"] == "task_completed"
        assert sent_data["data"]["task_id"] == "task-1"
        assert sent_data["data"]["agent_id"] == "agent-1"
        # Result is truncated to 200 chars
        assert "success" in sent_data["data"]["result_summary"]

    @pytest.mark.asyncio
    async def test_emit_task_failed(self, stream_server, mock_websocket):
        """Should emit task failed event."""
        await stream_server._register_client(mock_websocket)

        await stream_server.emit_task_failed(
            task_id="task-1",
            agent_id="agent-1",
            error="Timeout exceeded",
            retries_left=2,
        )

        sent_data = json.loads(mock_websocket.send.call_args[0][0])
        assert sent_data["type"] == "task_failed"
        assert sent_data["data"]["task_id"] == "task-1"
        assert sent_data["data"]["error"] == "Timeout exceeded"
        assert sent_data["data"]["retries_left"] == 2

    @pytest.mark.asyncio
    async def test_emit_task_dead_lettered(self, stream_server, mock_websocket):
        """Should emit task dead lettered event."""
        await stream_server._register_client(mock_websocket)

        await stream_server.emit_task_dead_lettered(
            task_id="task-1",
            reason="Max retries exceeded",
        )

        sent_data = json.loads(mock_websocket.send.call_args[0][0])
        assert sent_data["type"] == "task_dead_lettered"
        assert sent_data["data"]["task_id"] == "task-1"
        assert sent_data["data"]["reason"] == "Max retries exceeded"

    @pytest.mark.asyncio
    async def test_emit_health_update(self, stream_server, mock_websocket):
        """Should emit health update event."""
        await stream_server._register_client(mock_websocket)

        await stream_server.emit_health_update(
            status="healthy",
            agents={"agent-1": {"status": "idle"}},
        )

        sent_data = json.loads(mock_websocket.send.call_args[0][0])
        assert sent_data["type"] == "health_update"
        assert sent_data["data"]["status"] == "healthy"
        assert "agent-1" in sent_data["data"]["agents"]

    @pytest.mark.asyncio
    async def test_emit_scheduler_stats(self, stream_server, mock_websocket):
        """Should emit scheduler stats event."""
        await stream_server._register_client(mock_websocket)

        await stream_server.emit_scheduler_stats(
            {
                "pending_tasks": 5,
                "running_tasks": 3,
                "agents_idle": 2,
            }
        )

        sent_data = json.loads(mock_websocket.send.call_args[0][0])
        assert sent_data["type"] == "scheduler_stats"
        assert sent_data["data"]["pending_tasks"] == 5

    @pytest.mark.asyncio
    async def test_emit_error(self, stream_server, mock_websocket):
        """Should emit error event."""
        await stream_server._register_client(mock_websocket)

        await stream_server.emit_error(
            error="Something went wrong",
            context={"task_id": "task-1"},
        )

        sent_data = json.loads(mock_websocket.send.call_args[0][0])
        assert sent_data["type"] == "error"
        assert sent_data["data"]["error"] == "Something went wrong"
        assert sent_data["data"]["context"]["task_id"] == "task-1"


# ===========================================================================
# Test Server Lifecycle
# ===========================================================================


class TestServerLifecycle:
    """Tests for server start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_stop_without_start(self, stream_server):
        """Should handle stop when server wasn't started."""
        # Should not raise
        await stream_server.stop()
        assert stream_server._running is False

    def test_client_count_property(self, stream_server):
        """Should return correct client count."""
        assert stream_server.client_count == 0
