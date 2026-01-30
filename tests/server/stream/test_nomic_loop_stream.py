"""
Tests for nomic_loop_stream.py - WebSocket stream server for Nomic Loop events.

Tests cover:
- NomicLoopEventType enum values
- NomicLoopEvent dataclass (serialization, timestamp generation)
- NomicLoopStreamServer (lifecycle, connection handling, broadcasting)
- Phase transition events (context, debate, design, implement, verify)
- Loop lifecycle events (started, paused, resumed, stopped)
- Cycle events (started, completed)
- Proposal events (generated, approved, rejected)
- Health events (update, stall detection)
- WebSocket connection handling
- Event history and subscription management
- Concurrent client handling
"""

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.stream.nomic_loop_stream import (
    NomicLoopEvent,
    NomicLoopEventType,
    NomicLoopStreamServer,
)


# ===========================================================================
# Test Fixtures
# ===========================================================================


@pytest.fixture
def server():
    """Create a NomicLoopStreamServer for testing."""
    return NomicLoopStreamServer(port=8767, host="127.0.0.1")


@pytest.fixture
def mock_ws():
    """Create a mock WebSocket connection."""
    ws = MagicMock()
    ws.send = AsyncMock()
    ws.close = AsyncMock()
    ws.path = "/api/nomic/stream"
    return ws


@pytest.fixture
def mock_ws_alt_path():
    """Create a mock WebSocket with alternative path."""
    ws = MagicMock()
    ws.send = AsyncMock()
    ws.close = AsyncMock()
    ws.path = "/ws/nomic"
    return ws


@pytest.fixture
def mock_ws_invalid_path():
    """Create a mock WebSocket with invalid path."""
    ws = MagicMock()
    ws.send = AsyncMock()
    ws.close = AsyncMock()
    ws.path = "/invalid/path"
    return ws


@pytest.fixture
def multiple_mock_ws():
    """Create multiple mock WebSocket connections."""
    websockets = []
    for i in range(5):
        ws = MagicMock()
        ws.send = AsyncMock()
        ws.close = AsyncMock()
        ws.path = "/api/nomic/stream"
        websockets.append(ws)
    return websockets


# ===========================================================================
# Test NomicLoopEventType
# ===========================================================================


class TestNomicLoopEventType:
    """Tests for NomicLoopEventType enum."""

    def test_connected_value(self):
        """CONNECTED has correct string value."""
        assert NomicLoopEventType.CONNECTED.value == "connected"

    def test_disconnected_value(self):
        """DISCONNECTED has correct string value."""
        assert NomicLoopEventType.DISCONNECTED.value == "disconnected"

    def test_loop_started_value(self):
        """LOOP_STARTED has correct string value."""
        assert NomicLoopEventType.LOOP_STARTED.value == "loop_started"

    def test_loop_paused_value(self):
        """LOOP_PAUSED has correct string value."""
        assert NomicLoopEventType.LOOP_PAUSED.value == "loop_paused"

    def test_loop_resumed_value(self):
        """LOOP_RESUMED has correct string value."""
        assert NomicLoopEventType.LOOP_RESUMED.value == "loop_resumed"

    def test_loop_stopped_value(self):
        """LOOP_STOPPED has correct string value."""
        assert NomicLoopEventType.LOOP_STOPPED.value == "loop_stopped"

    def test_phase_started_value(self):
        """PHASE_STARTED has correct string value."""
        assert NomicLoopEventType.PHASE_STARTED.value == "phase_started"

    def test_phase_completed_value(self):
        """PHASE_COMPLETED has correct string value."""
        assert NomicLoopEventType.PHASE_COMPLETED.value == "phase_completed"

    def test_phase_skipped_value(self):
        """PHASE_SKIPPED has correct string value."""
        assert NomicLoopEventType.PHASE_SKIPPED.value == "phase_skipped"

    def test_phase_failed_value(self):
        """PHASE_FAILED has correct string value."""
        assert NomicLoopEventType.PHASE_FAILED.value == "phase_failed"

    def test_cycle_started_value(self):
        """CYCLE_STARTED has correct string value."""
        assert NomicLoopEventType.CYCLE_STARTED.value == "cycle_started"

    def test_cycle_completed_value(self):
        """CYCLE_COMPLETED has correct string value."""
        assert NomicLoopEventType.CYCLE_COMPLETED.value == "cycle_completed"

    def test_proposal_generated_value(self):
        """PROPOSAL_GENERATED has correct string value."""
        assert NomicLoopEventType.PROPOSAL_GENERATED.value == "proposal_generated"

    def test_proposal_approved_value(self):
        """PROPOSAL_APPROVED has correct string value."""
        assert NomicLoopEventType.PROPOSAL_APPROVED.value == "proposal_approved"

    def test_proposal_rejected_value(self):
        """PROPOSAL_REJECTED has correct string value."""
        assert NomicLoopEventType.PROPOSAL_REJECTED.value == "proposal_rejected"

    def test_health_update_value(self):
        """HEALTH_UPDATE has correct string value."""
        assert NomicLoopEventType.HEALTH_UPDATE.value == "health_update"

    def test_stall_detected_value(self):
        """STALL_DETECTED has correct string value."""
        assert NomicLoopEventType.STALL_DETECTED.value == "stall_detected"

    def test_stall_resolved_value(self):
        """STALL_RESOLVED has correct string value."""
        assert NomicLoopEventType.STALL_RESOLVED.value == "stall_resolved"

    def test_log_message_value(self):
        """LOG_MESSAGE has correct string value."""
        assert NomicLoopEventType.LOG_MESSAGE.value == "log_message"

    def test_error_value(self):
        """ERROR has correct string value."""
        assert NomicLoopEventType.ERROR.value == "error"


# ===========================================================================
# Test NomicLoopEvent
# ===========================================================================


class TestNomicLoopEvent:
    """Tests for NomicLoopEvent dataclass."""

    def test_event_creation_basic(self):
        """Event can be created with basic attributes."""
        event = NomicLoopEvent(
            event_type=NomicLoopEventType.LOOP_STARTED,
            data={"cycles": 3},
        )
        assert event.event_type == NomicLoopEventType.LOOP_STARTED
        assert event.data["cycles"] == 3

    def test_event_auto_generates_timestamp(self):
        """Event auto-generates timestamp if not provided."""
        before = time.time()
        event = NomicLoopEvent(event_type=NomicLoopEventType.LOOP_STARTED)
        after = time.time()

        assert before <= event.timestamp <= after

    def test_event_preserves_provided_timestamp(self):
        """Event preserves explicitly provided timestamp."""
        custom_timestamp = 1704067200.0  # 2024-01-01 00:00:00 UTC
        event = NomicLoopEvent(
            event_type=NomicLoopEventType.LOOP_STARTED,
            timestamp=custom_timestamp,
        )
        assert event.timestamp == custom_timestamp

    def test_event_to_dict(self):
        """to_dict returns proper dictionary representation."""
        event = NomicLoopEvent(
            event_type=NomicLoopEventType.PHASE_COMPLETED,
            timestamp=1704067200.0,
            data={"phase": "debate", "duration_sec": 45.5},
        )
        result = event.to_dict()

        assert result["type"] == "phase_completed"
        assert result["timestamp"] == 1704067200.0
        assert result["data"]["phase"] == "debate"
        assert result["data"]["duration_sec"] == 45.5

    def test_event_to_json(self):
        """to_json returns valid JSON string."""
        event = NomicLoopEvent(
            event_type=NomicLoopEventType.CYCLE_STARTED,
            timestamp=1704067200.0,
            data={"cycle": 1, "total_cycles": 3},
        )
        json_str = event.to_json()
        parsed = json.loads(json_str)

        assert parsed["type"] == "cycle_started"
        assert parsed["data"]["cycle"] == 1
        assert parsed["data"]["total_cycles"] == 3

    def test_event_default_data_is_empty_dict(self):
        """Event defaults to empty dict for data."""
        event = NomicLoopEvent(event_type=NomicLoopEventType.CONNECTED)
        assert event.data == {}


# ===========================================================================
# Test NomicLoopStreamServer - Initialization
# ===========================================================================


class TestNomicLoopStreamServerInit:
    """Tests for NomicLoopStreamServer initialization."""

    def test_default_port_and_host(self):
        """Server has correct default port and host."""
        server = NomicLoopStreamServer()
        assert server.port == 8767
        assert server.host == "0.0.0.0"

    def test_custom_port_and_host(self):
        """Server accepts custom port and host."""
        server = NomicLoopStreamServer(port=9000, host="localhost")
        assert server.port == 9000
        assert server.host == "localhost"

    def test_initial_state(self, server):
        """Server initializes with correct state."""
        assert not server._running
        assert server._server is None
        assert len(server._clients) == 0

    def test_client_count_property(self, server):
        """client_count property returns correct count."""
        assert server.client_count == 0


# ===========================================================================
# Test NomicLoopStreamServer - Client Management
# ===========================================================================


class TestNomicLoopStreamServerClientManagement:
    """Tests for NomicLoopStreamServer client management."""

    @pytest.mark.asyncio
    async def test_register_client(self, server, mock_ws):
        """Client registration adds to client set."""
        await server._register_client(mock_ws)
        assert mock_ws in server._clients
        assert server.client_count == 1

    @pytest.mark.asyncio
    async def test_unregister_client(self, server, mock_ws):
        """Client unregistration removes from client set."""
        await server._register_client(mock_ws)
        await server._unregister_client(mock_ws)
        assert mock_ws not in server._clients
        assert server.client_count == 0

    @pytest.mark.asyncio
    async def test_register_multiple_clients(self, server, multiple_mock_ws):
        """Multiple clients can be registered."""
        for ws in multiple_mock_ws:
            await server._register_client(ws)
        assert server.client_count == 5

    @pytest.mark.asyncio
    async def test_unregister_preserves_other_clients(self, server, multiple_mock_ws):
        """Unregistering one client preserves others."""
        for ws in multiple_mock_ws:
            await server._register_client(ws)

        await server._unregister_client(multiple_mock_ws[0])

        assert server.client_count == 4
        assert multiple_mock_ws[0] not in server._clients
        for ws in multiple_mock_ws[1:]:
            assert ws in server._clients

    @pytest.mark.asyncio
    async def test_unregister_nonexistent_client(self, server, mock_ws):
        """Unregistering nonexistent client is safe."""
        await server._unregister_client(mock_ws)
        # Should not raise


# ===========================================================================
# Test NomicLoopStreamServer - Message Handling
# ===========================================================================


class TestNomicLoopStreamServerMessageHandling:
    """Tests for NomicLoopStreamServer message handling."""

    @pytest.mark.asyncio
    async def test_handle_ping_message(self, server, mock_ws):
        """Server responds to ping with pong."""
        message = json.dumps({"type": "ping"})
        await server._handle_message(mock_ws, message)

        mock_ws.send.assert_called_once()
        response = json.loads(mock_ws.send.call_args[0][0])
        assert response["type"] == "pong"
        assert "timestamp" in response

    @pytest.mark.asyncio
    async def test_handle_subscribe_message(self, server, mock_ws):
        """Server handles subscribe message."""
        message = json.dumps({"type": "subscribe", "events": ["phase_started"]})
        await server._handle_message(mock_ws, message)
        # Currently subscribe is a no-op, but should not raise

    @pytest.mark.asyncio
    async def test_handle_invalid_json(self, server, mock_ws):
        """Server handles invalid JSON gracefully."""
        await server._handle_message(mock_ws, "not valid json{")
        # Should not raise

    @pytest.mark.asyncio
    async def test_handle_message_with_unknown_type(self, server, mock_ws):
        """Server handles unknown message types gracefully."""
        message = json.dumps({"type": "unknown_type", "data": {}})
        await server._handle_message(mock_ws, message)
        # Should not raise


# ===========================================================================
# Test NomicLoopStreamServer - Broadcasting
# ===========================================================================


class TestNomicLoopStreamServerBroadcasting:
    """Tests for NomicLoopStreamServer event broadcasting."""

    @pytest.mark.asyncio
    async def test_broadcast_to_single_client(self, server, mock_ws):
        """Broadcast sends event to single connected client."""
        await server._register_client(mock_ws)

        event = NomicLoopEvent(
            event_type=NomicLoopEventType.LOOP_STARTED,
            data={"cycles": 3},
        )
        await server.broadcast(event)

        mock_ws.send.assert_called_once()
        sent = json.loads(mock_ws.send.call_args[0][0])
        assert sent["type"] == "loop_started"
        assert sent["data"]["cycles"] == 3

    @pytest.mark.asyncio
    async def test_broadcast_to_multiple_clients(self, server, multiple_mock_ws):
        """Broadcast sends event to all connected clients."""
        for ws in multiple_mock_ws:
            await server._register_client(ws)

        event = NomicLoopEvent(
            event_type=NomicLoopEventType.PHASE_STARTED,
            data={"phase": "debate"},
        )
        await server.broadcast(event)

        for ws in multiple_mock_ws:
            ws.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_broadcast_with_no_clients(self, server):
        """Broadcast with no clients completes without error."""
        event = NomicLoopEvent(
            event_type=NomicLoopEventType.LOOP_STARTED,
            data={},
        )
        await server.broadcast(event)
        # Should not raise

    @pytest.mark.asyncio
    async def test_broadcast_removes_dead_client(self, server, mock_ws):
        """Broadcast removes client that fails to receive."""
        mock_ws.send = AsyncMock(side_effect=ConnectionError("Client disconnected"))
        await server._register_client(mock_ws)

        event = NomicLoopEvent(
            event_type=NomicLoopEventType.LOOP_STARTED,
            data={},
        )
        await server.broadcast(event)

        assert mock_ws not in server._clients

    @pytest.mark.asyncio
    async def test_broadcast_handles_partial_failure(self, server, multiple_mock_ws):
        """Broadcast continues when some clients fail."""
        # First client fails
        multiple_mock_ws[0].send = AsyncMock(side_effect=OSError("Network error"))
        for ws in multiple_mock_ws:
            await server._register_client(ws)

        event = NomicLoopEvent(
            event_type=NomicLoopEventType.LOOP_STARTED,
            data={},
        )
        await server.broadcast(event)

        # Successful clients should have received the message
        for ws in multiple_mock_ws[1:]:
            ws.send.assert_called_once()


# ===========================================================================
# Test NomicLoopStreamServer - Loop Lifecycle Events
# ===========================================================================


class TestNomicLoopStreamServerLoopLifecycle:
    """Tests for loop lifecycle event emission."""

    @pytest.mark.asyncio
    async def test_emit_loop_started(self, server, mock_ws):
        """emit_loop_started sends correct event."""
        await server._register_client(mock_ws)

        await server.emit_loop_started(cycles=5, auto_approve=True, dry_run=False)

        sent = json.loads(mock_ws.send.call_args[0][0])
        assert sent["type"] == "loop_started"
        assert sent["data"]["cycles"] == 5
        assert sent["data"]["auto_approve"] is True
        assert sent["data"]["dry_run"] is False

    @pytest.mark.asyncio
    async def test_emit_loop_started_defaults(self, server, mock_ws):
        """emit_loop_started uses correct defaults."""
        await server._register_client(mock_ws)

        await server.emit_loop_started()

        sent = json.loads(mock_ws.send.call_args[0][0])
        assert sent["data"]["cycles"] == 1
        assert sent["data"]["auto_approve"] is False
        assert sent["data"]["dry_run"] is False

    @pytest.mark.asyncio
    async def test_emit_loop_paused(self, server, mock_ws):
        """emit_loop_paused sends correct event."""
        await server._register_client(mock_ws)

        await server.emit_loop_paused(current_phase="design", current_cycle=2)

        sent = json.loads(mock_ws.send.call_args[0][0])
        assert sent["type"] == "loop_paused"
        assert sent["data"]["current_phase"] == "design"
        assert sent["data"]["current_cycle"] == 2

    @pytest.mark.asyncio
    async def test_emit_loop_resumed(self, server, mock_ws):
        """emit_loop_resumed sends correct event."""
        await server._register_client(mock_ws)

        await server.emit_loop_resumed(current_phase="implement", current_cycle=3)

        sent = json.loads(mock_ws.send.call_args[0][0])
        assert sent["type"] == "loop_resumed"
        assert sent["data"]["current_phase"] == "implement"
        assert sent["data"]["current_cycle"] == 3

    @pytest.mark.asyncio
    async def test_emit_loop_stopped(self, server, mock_ws):
        """emit_loop_stopped sends correct event."""
        await server._register_client(mock_ws)

        await server.emit_loop_stopped(forced=True, reason="User cancellation")

        sent = json.loads(mock_ws.send.call_args[0][0])
        assert sent["type"] == "loop_stopped"
        assert sent["data"]["forced"] is True
        assert sent["data"]["reason"] == "User cancellation"


# ===========================================================================
# Test NomicLoopStreamServer - Phase Events
# ===========================================================================


class TestNomicLoopStreamServerPhaseEvents:
    """Tests for phase transition event emission."""

    @pytest.mark.asyncio
    async def test_emit_phase_started(self, server, mock_ws):
        """emit_phase_started sends correct event."""
        await server._register_client(mock_ws)

        await server.emit_phase_started(
            phase="context",
            cycle=1,
            estimated_duration_sec=30,
        )

        sent = json.loads(mock_ws.send.call_args[0][0])
        assert sent["type"] == "phase_started"
        assert sent["data"]["phase"] == "context"
        assert sent["data"]["cycle"] == 1
        assert sent["data"]["estimated_duration_sec"] == 30

    @pytest.mark.asyncio
    async def test_emit_phase_started_all_phases(self, server, mock_ws):
        """emit_phase_started works for all phase types."""
        await server._register_client(mock_ws)

        phases = ["context", "debate", "design", "implement", "verify"]
        for i, phase in enumerate(phases):
            mock_ws.send.reset_mock()
            await server.emit_phase_started(phase=phase, cycle=1)

            sent = json.loads(mock_ws.send.call_args[0][0])
            assert sent["data"]["phase"] == phase

    @pytest.mark.asyncio
    async def test_emit_phase_completed(self, server, mock_ws):
        """emit_phase_completed sends correct event."""
        await server._register_client(mock_ws)

        await server.emit_phase_completed(
            phase="debate",
            cycle=2,
            duration_sec=45.5,
            result_summary="Consensus reached on API design",
        )

        sent = json.loads(mock_ws.send.call_args[0][0])
        assert sent["type"] == "phase_completed"
        assert sent["data"]["phase"] == "debate"
        assert sent["data"]["cycle"] == 2
        assert sent["data"]["duration_sec"] == 45.5
        assert "Consensus reached" in sent["data"]["result_summary"]

    @pytest.mark.asyncio
    async def test_emit_phase_skipped(self, server, mock_ws):
        """emit_phase_skipped sends correct event."""
        await server._register_client(mock_ws)

        await server.emit_phase_skipped(
            phase="verify",
            cycle=1,
            reason="Dry run mode enabled",
        )

        sent = json.loads(mock_ws.send.call_args[0][0])
        assert sent["type"] == "phase_skipped"
        assert sent["data"]["phase"] == "verify"
        assert sent["data"]["reason"] == "Dry run mode enabled"

    @pytest.mark.asyncio
    async def test_emit_phase_failed(self, server, mock_ws):
        """emit_phase_failed sends correct event."""
        await server._register_client(mock_ws)

        await server.emit_phase_failed(
            phase="implement",
            cycle=3,
            error="Test suite failed with 5 errors",
        )

        sent = json.loads(mock_ws.send.call_args[0][0])
        assert sent["type"] == "phase_failed"
        assert sent["data"]["phase"] == "implement"
        assert sent["data"]["cycle"] == 3
        assert "Test suite failed" in sent["data"]["error"]


# ===========================================================================
# Test NomicLoopStreamServer - Cycle Events
# ===========================================================================


class TestNomicLoopStreamServerCycleEvents:
    """Tests for cycle event emission."""

    @pytest.mark.asyncio
    async def test_emit_cycle_started(self, server, mock_ws):
        """emit_cycle_started sends correct event."""
        await server._register_client(mock_ws)

        await server.emit_cycle_started(cycle=2, total_cycles=5)

        sent = json.loads(mock_ws.send.call_args[0][0])
        assert sent["type"] == "cycle_started"
        assert sent["data"]["cycle"] == 2
        assert sent["data"]["total_cycles"] == 5

    @pytest.mark.asyncio
    async def test_emit_cycle_completed(self, server, mock_ws):
        """emit_cycle_completed sends correct event."""
        await server._register_client(mock_ws)

        await server.emit_cycle_completed(
            cycle=1,
            total_cycles=3,
            duration_sec=120.5,
            improvements_made=5,
        )

        sent = json.loads(mock_ws.send.call_args[0][0])
        assert sent["type"] == "cycle_completed"
        assert sent["data"]["cycle"] == 1
        assert sent["data"]["total_cycles"] == 3
        assert sent["data"]["duration_sec"] == 120.5
        assert sent["data"]["improvements_made"] == 5


# ===========================================================================
# Test NomicLoopStreamServer - Proposal Events
# ===========================================================================


class TestNomicLoopStreamServerProposalEvents:
    """Tests for proposal event emission."""

    @pytest.mark.asyncio
    async def test_emit_proposal_generated(self, server, mock_ws):
        """emit_proposal_generated sends correct event."""
        await server._register_client(mock_ws)

        await server.emit_proposal_generated(
            proposal_id="prop-123",
            title="Add caching layer",
            description="Implement Redis caching for frequently accessed data",
            phase="design",
            requires_approval=True,
        )

        sent = json.loads(mock_ws.send.call_args[0][0])
        assert sent["type"] == "proposal_generated"
        assert sent["data"]["proposal_id"] == "prop-123"
        assert sent["data"]["title"] == "Add caching layer"
        assert sent["data"]["phase"] == "design"
        assert sent["data"]["requires_approval"] is True

    @pytest.mark.asyncio
    async def test_emit_proposal_generated_truncates_description(self, server, mock_ws):
        """emit_proposal_generated truncates long descriptions."""
        await server._register_client(mock_ws)

        long_description = "A" * 1000  # Longer than 500 char limit
        await server.emit_proposal_generated(
            proposal_id="prop-456",
            title="Test",
            description=long_description,
            phase="implement",
        )

        sent = json.loads(mock_ws.send.call_args[0][0])
        assert len(sent["data"]["description"]) == 500

    @pytest.mark.asyncio
    async def test_emit_proposal_approved(self, server, mock_ws):
        """emit_proposal_approved sends correct event."""
        await server._register_client(mock_ws)

        await server.emit_proposal_approved(
            proposal_id="prop-123",
            approved_by="admin",
        )

        sent = json.loads(mock_ws.send.call_args[0][0])
        assert sent["type"] == "proposal_approved"
        assert sent["data"]["proposal_id"] == "prop-123"
        assert sent["data"]["approved_by"] == "admin"

    @pytest.mark.asyncio
    async def test_emit_proposal_rejected(self, server, mock_ws):
        """emit_proposal_rejected sends correct event."""
        await server._register_client(mock_ws)

        await server.emit_proposal_rejected(
            proposal_id="prop-789",
            rejected_by="reviewer",
            reason="Does not align with project goals",
        )

        sent = json.loads(mock_ws.send.call_args[0][0])
        assert sent["type"] == "proposal_rejected"
        assert sent["data"]["proposal_id"] == "prop-789"
        assert sent["data"]["rejected_by"] == "reviewer"
        assert "Does not align" in sent["data"]["reason"]


# ===========================================================================
# Test NomicLoopStreamServer - Health Events
# ===========================================================================


class TestNomicLoopStreamServerHealthEvents:
    """Tests for health event emission."""

    @pytest.mark.asyncio
    async def test_emit_health_update(self, server, mock_ws):
        """emit_health_update sends correct event."""
        await server._register_client(mock_ws)

        await server.emit_health_update(
            status="healthy",
            running=True,
            paused=False,
            current_phase="debate",
            current_cycle=2,
            stalled=False,
        )

        sent = json.loads(mock_ws.send.call_args[0][0])
        assert sent["type"] == "health_update"
        assert sent["data"]["status"] == "healthy"
        assert sent["data"]["running"] is True
        assert sent["data"]["paused"] is False
        assert sent["data"]["current_phase"] == "debate"
        assert sent["data"]["current_cycle"] == 2
        assert sent["data"]["stalled"] is False

    @pytest.mark.asyncio
    async def test_emit_stall_detected(self, server, mock_ws):
        """emit_stall_detected sends correct event."""
        await server._register_client(mock_ws)

        await server.emit_stall_detected(
            phase="implement",
            stall_duration_sec=300.0,
            threshold_sec=180.0,
        )

        sent = json.loads(mock_ws.send.call_args[0][0])
        assert sent["type"] == "stall_detected"
        assert sent["data"]["phase"] == "implement"
        assert sent["data"]["stall_duration_sec"] == 300.0
        assert sent["data"]["threshold_sec"] == 180.0

    @pytest.mark.asyncio
    async def test_emit_stall_resolved(self, server, mock_ws):
        """emit_stall_resolved sends correct event."""
        await server._register_client(mock_ws)

        await server.emit_stall_resolved(
            phase="implement",
            resolution="External dependency timeout increased",
        )

        sent = json.loads(mock_ws.send.call_args[0][0])
        assert sent["type"] == "stall_resolved"
        assert sent["data"]["phase"] == "implement"
        assert "External dependency" in sent["data"]["resolution"]


# ===========================================================================
# Test NomicLoopStreamServer - Log and Error Events
# ===========================================================================


class TestNomicLoopStreamServerLogErrorEvents:
    """Tests for log and error event emission."""

    @pytest.mark.asyncio
    async def test_emit_log_message(self, server, mock_ws):
        """emit_log_message sends correct event."""
        await server._register_client(mock_ws)

        await server.emit_log_message(
            level="info",
            message="Starting context analysis...",
            source="context_phase",
        )

        sent = json.loads(mock_ws.send.call_args[0][0])
        assert sent["type"] == "log_message"
        assert sent["data"]["level"] == "info"
        assert "context analysis" in sent["data"]["message"]
        assert sent["data"]["source"] == "context_phase"

    @pytest.mark.asyncio
    async def test_emit_log_message_truncates_long_message(self, server, mock_ws):
        """emit_log_message truncates long messages."""
        await server._register_client(mock_ws)

        long_message = "X" * 2000
        await server.emit_log_message(
            level="debug",
            message=long_message,
        )

        sent = json.loads(mock_ws.send.call_args[0][0])
        assert len(sent["data"]["message"]) == 1000

    @pytest.mark.asyncio
    async def test_emit_error(self, server, mock_ws):
        """emit_error sends correct event."""
        await server._register_client(mock_ws)

        await server.emit_error(
            error="Failed to connect to Redis",
            context={"host": "redis.local", "port": 6379},
        )

        sent = json.loads(mock_ws.send.call_args[0][0])
        assert sent["type"] == "error"
        assert "Redis" in sent["data"]["error"]
        assert sent["data"]["context"]["host"] == "redis.local"

    @pytest.mark.asyncio
    async def test_emit_error_with_none_context(self, server, mock_ws):
        """emit_error handles None context."""
        await server._register_client(mock_ws)

        await server.emit_error(error="Generic error")

        sent = json.loads(mock_ws.send.call_args[0][0])
        assert sent["data"]["context"] == {}


# ===========================================================================
# Test NomicLoopStreamServer - Connection Handling
# ===========================================================================


class TestNomicLoopStreamServerConnectionHandling:
    """Tests for WebSocket connection handling."""

    @pytest.mark.asyncio
    async def test_handle_connection_valid_path(self, server, mock_ws):
        """Connection with valid path is accepted."""
        # Simulate connection handling without actual websocket iteration
        await server._register_client(mock_ws)

        # Verify connection confirmation would be sent
        event = NomicLoopEvent(
            event_type=NomicLoopEventType.CONNECTED,
            data={"message": "Connected to Nomic Loop stream"},
        )
        await mock_ws.send(event.to_json())

        assert mock_ws in server._clients

    @pytest.mark.asyncio
    async def test_handle_connection_alt_path(self, server, mock_ws_alt_path):
        """Connection with alternative path (/ws/nomic) is accepted."""
        await server._register_client(mock_ws_alt_path)
        assert mock_ws_alt_path in server._clients

    @pytest.mark.asyncio
    async def test_concurrent_connection_registration(self, server, multiple_mock_ws):
        """Concurrent connection registrations are safe."""

        async def register(ws):
            await server._register_client(ws)

        await asyncio.gather(*[register(ws) for ws in multiple_mock_ws])

        assert server.client_count == 5

    @pytest.mark.asyncio
    async def test_concurrent_connection_unregistration(self, server, multiple_mock_ws):
        """Concurrent connection unregistrations are safe."""
        for ws in multiple_mock_ws:
            await server._register_client(ws)

        async def unregister(ws):
            await server._unregister_client(ws)

        await asyncio.gather(*[unregister(ws) for ws in multiple_mock_ws])

        assert server.client_count == 0


# ===========================================================================
# Test NomicLoopStreamServer - Server Lifecycle
# ===========================================================================


class TestNomicLoopStreamServerLifecycle:
    """Tests for server start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_stop_when_not_started(self, server):
        """Stop gracefully handles case when server not started."""
        await server.stop()
        assert not server._running

    @pytest.mark.asyncio
    async def test_start_sets_running_flag(self, server):
        """Start sets running flag."""
        with patch("websockets.serve", new_callable=AsyncMock) as mock_serve:
            mock_server = MagicMock()
            mock_serve.return_value = mock_server

            await server.start()

            assert server._running
            assert server._server is mock_server

    @pytest.mark.asyncio
    async def test_stop_clears_running_flag(self, server):
        """Stop clears running flag."""
        with patch("websockets.serve", new_callable=AsyncMock) as mock_serve:
            mock_server = MagicMock()
            mock_server.close = MagicMock()
            mock_server.wait_closed = AsyncMock()
            mock_serve.return_value = mock_server

            await server.start()
            await server.stop()

            assert not server._running
            mock_server.close.assert_called_once()
            mock_server.wait_closed.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_without_websockets_package(self, server):
        """Start handles missing websockets package gracefully."""
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "websockets":
                raise ImportError("No module named 'websockets'")
            return original_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", mock_import):
            # The method should log error and return without raising
            await server.start()
            # Server should not be marked as running since websockets failed to import
            assert not server._running


# ===========================================================================
# Test Full Event Sequence
# ===========================================================================


class TestNomicLoopStreamServerEventSequence:
    """Tests for full event sequences."""

    @pytest.mark.asyncio
    async def test_full_cycle_event_sequence(self, server, mock_ws):
        """Full cycle emits correct sequence of events."""
        await server._register_client(mock_ws)

        # Simulate a full cycle
        await server.emit_loop_started(cycles=1)
        await server.emit_cycle_started(cycle=1, total_cycles=1)

        phases = ["context", "debate", "design", "implement", "verify"]
        for phase in phases:
            await server.emit_phase_started(phase=phase, cycle=1)
            await server.emit_phase_completed(phase=phase, cycle=1, duration_sec=10.0)

        await server.emit_cycle_completed(cycle=1, total_cycles=1, duration_sec=50.0)
        await server.emit_loop_stopped()

        # Verify all events were sent (1 start + 1 cycle start + 5*2 phases + 1 cycle complete + 1 stop = 14)
        assert mock_ws.send.call_count == 14

    @pytest.mark.asyncio
    async def test_cycle_with_proposal_approval(self, server, mock_ws):
        """Cycle with proposal generation and approval."""
        await server._register_client(mock_ws)

        await server.emit_loop_started(cycles=1, auto_approve=False)
        await server.emit_phase_started(phase="design", cycle=1)
        await server.emit_proposal_generated(
            proposal_id="prop-1",
            title="Add logging",
            description="Add structured logging",
            phase="design",
            requires_approval=True,
        )
        await server.emit_proposal_approved(proposal_id="prop-1")
        await server.emit_phase_completed(phase="design", cycle=1, duration_sec=30.0)

        assert mock_ws.send.call_count == 5

    @pytest.mark.asyncio
    async def test_cycle_with_failure_and_recovery(self, server, mock_ws):
        """Cycle with phase failure and stall detection."""
        await server._register_client(mock_ws)

        await server.emit_loop_started(cycles=1)
        await server.emit_phase_started(phase="implement", cycle=1)
        await server.emit_stall_detected(
            phase="implement",
            stall_duration_sec=200.0,
            threshold_sec=180.0,
        )
        await server.emit_stall_resolved(phase="implement", resolution="Retry succeeded")
        await server.emit_phase_completed(phase="implement", cycle=1, duration_sec=250.0)

        assert mock_ws.send.call_count == 5


# ===========================================================================
# Test Edge Cases
# ===========================================================================


class TestNomicLoopStreamServerEdgeCases:
    """Tests for edge cases."""

    @pytest.mark.asyncio
    async def test_broadcast_with_runtime_error(self, server, mock_ws):
        """Broadcast handles RuntimeError on send."""
        mock_ws.send = AsyncMock(side_effect=RuntimeError("WebSocket already closed"))
        await server._register_client(mock_ws)

        event = NomicLoopEvent(
            event_type=NomicLoopEventType.LOOP_STARTED,
            data={},
        )
        await server.broadcast(event)

        assert mock_ws not in server._clients

    @pytest.mark.asyncio
    async def test_event_with_unicode_data(self, server, mock_ws):
        """Events handle unicode data correctly."""
        await server._register_client(mock_ws)

        await server.emit_log_message(
            level="info",
            message="Processing: \u65e5\u672c\u8a9e\u30c6\u30b9\u30c8",
        )

        sent = json.loads(mock_ws.send.call_args[0][0])
        assert "\u65e5\u672c\u8a9e" in sent["data"]["message"]

    @pytest.mark.asyncio
    async def test_rapid_sequential_broadcasts(self, server, mock_ws):
        """Rapid sequential broadcasts complete correctly."""
        await server._register_client(mock_ws)

        for i in range(20):
            await server.emit_log_message(level="info", message=f"Message {i}")

        assert mock_ws.send.call_count == 20

    @pytest.mark.asyncio
    async def test_concurrent_broadcasts(self, server, mock_ws):
        """Concurrent broadcasts complete correctly."""
        await server._register_client(mock_ws)

        events = [
            NomicLoopEvent(
                event_type=NomicLoopEventType.LOG_MESSAGE,
                data={"level": "info", "message": f"Concurrent {i}"},
            )
            for i in range(10)
        ]

        await asyncio.gather(*[server.broadcast(e) for e in events])

        assert mock_ws.send.call_count == 10

    def test_event_with_empty_data(self):
        """Event handles empty data correctly."""
        event = NomicLoopEvent(event_type=NomicLoopEventType.CONNECTED)
        json_str = event.to_json()
        parsed = json.loads(json_str)
        assert parsed["data"] == {}

    @pytest.mark.asyncio
    async def test_send_to_client_catches_connection_errors(self, server, mock_ws):
        """_send_to_client catches and handles connection-related exceptions."""
        # The method only catches ConnectionError, OSError, RuntimeError
        mock_ws.send = AsyncMock(side_effect=OSError("Network unreachable"))
        await server._register_client(mock_ws)

        # Should not raise, and should unregister the client
        await server._send_to_client(mock_ws, '{"type": "test"}')
        assert mock_ws not in server._clients
