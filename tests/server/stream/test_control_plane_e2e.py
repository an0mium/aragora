"""
End-to-end tests for Control Plane WebSocket streaming.

Tests actual WebSocket connections:
- Server startup and shutdown
- Client connection and disconnection
- Event reception over real WebSocket
- Ping/pong protocol
- Event broadcasting to multiple clients
"""

import asyncio
import json
import pytest
from typing import List

from aragora.server.stream.control_plane_stream import (
    ControlPlaneStreamServer,
    ControlPlaneEventType,
    ControlPlaneEvent,
)


# ===========================================================================
# Test Fixtures
# ===========================================================================


@pytest.fixture
async def stream_server():
    """Start a real ControlPlaneStreamServer for testing."""
    # Use port 0 to get a random available port
    server = ControlPlaneStreamServer(port=0, host="127.0.0.1")
    await server.start()

    # Get the actual port assigned
    if server._server and server._server.sockets:
        actual_port = server._server.sockets[0].getsockname()[1]
        server.port = actual_port

    yield server

    await server.stop()


async def connect_client(port: int, path: str = "/api/control-plane/stream"):
    """Connect a WebSocket client to the server."""
    import websockets
    uri = f"ws://127.0.0.1:{port}{path}"
    return await websockets.connect(uri)


async def receive_with_timeout(ws, timeout: float = 2.0) -> dict:
    """Receive a message with timeout."""
    try:
        message = await asyncio.wait_for(ws.recv(), timeout=timeout)
        return json.loads(message)
    except asyncio.TimeoutError:
        raise TimeoutError(f"No message received within {timeout}s")


# ===========================================================================
# Test Connection Lifecycle
# ===========================================================================


class TestConnectionLifecycle:
    """Tests for WebSocket connection lifecycle."""

    @pytest.mark.asyncio
    async def test_connect_receives_connected_event(self, stream_server):
        """Should receive CONNECTED event on connection."""
        ws = await connect_client(stream_server.port)
        try:
            message = await receive_with_timeout(ws)
            assert message["type"] == "connected"
            assert "message" in message["data"]
            assert "timestamp" in message
        finally:
            await ws.close()

    @pytest.mark.asyncio
    async def test_connect_invalid_path_rejected(self, stream_server):
        """Should reject connections to invalid paths."""
        import websockets

        with pytest.raises(websockets.exceptions.ConnectionClosed):
            ws = await connect_client(stream_server.port, path="/invalid/path")
            # Try to receive - should fail because connection was closed
            await ws.recv()

    @pytest.mark.asyncio
    async def test_client_count_updates(self, stream_server):
        """Should track client connections correctly."""
        assert stream_server.client_count == 0

        ws1 = await connect_client(stream_server.port)
        await receive_with_timeout(ws1)  # Consume connected event
        await asyncio.sleep(0.1)  # Let registration complete
        assert stream_server.client_count == 1

        ws2 = await connect_client(stream_server.port)
        await receive_with_timeout(ws2)
        await asyncio.sleep(0.1)
        assert stream_server.client_count == 2

        await ws1.close()
        await asyncio.sleep(0.1)
        assert stream_server.client_count == 1

        await ws2.close()
        await asyncio.sleep(0.1)
        assert stream_server.client_count == 0


# ===========================================================================
# Test Ping/Pong Protocol
# ===========================================================================


class TestPingPong:
    """Tests for ping/pong keepalive protocol."""

    @pytest.mark.asyncio
    async def test_ping_receives_pong(self, stream_server):
        """Should respond to ping with pong."""
        ws = await connect_client(stream_server.port)
        try:
            await receive_with_timeout(ws)  # Consume connected event

            # Send ping
            await ws.send(json.dumps({"type": "ping"}))

            # Receive pong
            message = await receive_with_timeout(ws)
            assert message["type"] == "pong"
            assert "timestamp" in message
        finally:
            await ws.close()


# ===========================================================================
# Test Event Broadcasting
# ===========================================================================


class TestEventBroadcasting:
    """Tests for event broadcasting to clients."""

    @pytest.mark.asyncio
    async def test_broadcast_reaches_single_client(self, stream_server):
        """Should broadcast events to connected client."""
        ws = await connect_client(stream_server.port)
        try:
            await receive_with_timeout(ws)  # Consume connected event

            # Broadcast an event
            event = ControlPlaneEvent(
                event_type=ControlPlaneEventType.AGENT_REGISTERED,
                data={"agent_id": "test-agent", "capabilities": ["debate"]},
            )
            await stream_server.broadcast(event)

            # Receive the broadcast
            message = await receive_with_timeout(ws)
            assert message["type"] == "agent_registered"
            assert message["data"]["agent_id"] == "test-agent"
            assert message["data"]["capabilities"] == ["debate"]
        finally:
            await ws.close()

    @pytest.mark.asyncio
    async def test_broadcast_reaches_multiple_clients(self, stream_server):
        """Should broadcast events to all connected clients."""
        clients = []
        try:
            # Connect 3 clients
            for _ in range(3):
                ws = await connect_client(stream_server.port)
                await receive_with_timeout(ws)  # Consume connected event
                clients.append(ws)

            # Broadcast an event
            event = ControlPlaneEvent(
                event_type=ControlPlaneEventType.TASK_COMPLETED,
                data={"task_id": "task-123", "agent_id": "agent-1"},
            )
            await stream_server.broadcast(event)

            # All clients should receive the event
            for ws in clients:
                message = await receive_with_timeout(ws)
                assert message["type"] == "task_completed"
                assert message["data"]["task_id"] == "task-123"
        finally:
            for ws in clients:
                await ws.close()

    @pytest.mark.asyncio
    async def test_multiple_broadcasts(self, stream_server):
        """Should deliver multiple events in order."""
        ws = await connect_client(stream_server.port)
        try:
            await receive_with_timeout(ws)  # Consume connected event

            # Broadcast multiple events
            events = [
                ControlPlaneEvent(
                    event_type=ControlPlaneEventType.TASK_SUBMITTED,
                    data={"task_id": "task-1"},
                ),
                ControlPlaneEvent(
                    event_type=ControlPlaneEventType.TASK_CLAIMED,
                    data={"task_id": "task-1", "agent_id": "agent-1"},
                ),
                ControlPlaneEvent(
                    event_type=ControlPlaneEventType.TASK_COMPLETED,
                    data={"task_id": "task-1", "result": "done"},
                ),
            ]

            for event in events:
                await stream_server.broadcast(event)

            # Receive all events in order
            received = []
            for _ in range(3):
                message = await receive_with_timeout(ws)
                received.append(message["type"])

            assert received == ["task_submitted", "task_claimed", "task_completed"]
        finally:
            await ws.close()


# ===========================================================================
# Test High-Level Event Emission
# ===========================================================================


class TestEventEmission:
    """Tests for high-level event emission methods."""

    @pytest.mark.asyncio
    async def test_emit_agent_registered(self, stream_server):
        """Should emit agent_registered event over WebSocket."""
        ws = await connect_client(stream_server.port)
        try:
            await receive_with_timeout(ws)

            await stream_server.emit_agent_registered(
                agent_id="agent-1",
                capabilities=["debate", "analysis"],
                model="gpt-4",
                provider="openai",
            )

            message = await receive_with_timeout(ws)
            assert message["type"] == "agent_registered"
            assert message["data"]["agent_id"] == "agent-1"
            assert message["data"]["model"] == "gpt-4"
        finally:
            await ws.close()

    @pytest.mark.asyncio
    async def test_emit_task_completed(self, stream_server):
        """Should emit task_completed event over WebSocket."""
        ws = await connect_client(stream_server.port)
        try:
            await receive_with_timeout(ws)

            await stream_server.emit_task_completed(
                task_id="task-1",
                agent_id="agent-1",
                result={"summary": "Task completed successfully"},
            )

            message = await receive_with_timeout(ws)
            assert message["type"] == "task_completed"
            assert message["data"]["task_id"] == "task-1"
            assert message["data"]["agent_id"] == "agent-1"
        finally:
            await ws.close()

    @pytest.mark.asyncio
    async def test_emit_scheduler_stats(self, stream_server):
        """Should emit scheduler_stats event over WebSocket."""
        ws = await connect_client(stream_server.port)
        try:
            await receive_with_timeout(ws)

            await stream_server.emit_scheduler_stats({
                "pending_tasks": 5,
                "running_tasks": 3,
                "agents_idle": 2,
                "agents_busy": 3,
            })

            message = await receive_with_timeout(ws)
            assert message["type"] == "scheduler_stats"
            assert message["data"]["pending_tasks"] == 5
            assert message["data"]["running_tasks"] == 3
        finally:
            await ws.close()


# ===========================================================================
# Test Error Handling
# ===========================================================================


class TestErrorHandling:
    """Tests for error handling in E2E scenarios."""

    @pytest.mark.asyncio
    async def test_broadcast_handles_disconnected_client(self, stream_server):
        """Should handle broadcasting when client disconnects."""
        ws1 = await connect_client(stream_server.port)
        ws2 = await connect_client(stream_server.port)
        try:
            await receive_with_timeout(ws1)
            await receive_with_timeout(ws2)

            # Close ws1 abruptly
            await ws1.close()
            await asyncio.sleep(0.1)

            # Broadcast should still work for ws2
            event = ControlPlaneEvent(
                event_type=ControlPlaneEventType.HEALTH_UPDATE,
                data={"status": "healthy"},
            )
            await stream_server.broadcast(event)

            message = await receive_with_timeout(ws2)
            assert message["type"] == "health_update"
        finally:
            await ws2.close()

    @pytest.mark.asyncio
    async def test_server_handles_invalid_json(self, stream_server):
        """Should handle invalid JSON messages gracefully."""
        ws = await connect_client(stream_server.port)
        try:
            await receive_with_timeout(ws)

            # Send invalid JSON
            await ws.send("not valid json {")

            # Connection should still be alive
            await ws.send(json.dumps({"type": "ping"}))
            message = await receive_with_timeout(ws)
            assert message["type"] == "pong"
        finally:
            await ws.close()


# ===========================================================================
# Test Alternative Path
# ===========================================================================


class TestAlternativePath:
    """Tests for alternative WebSocket path."""

    @pytest.mark.asyncio
    async def test_connect_via_ws_path(self, stream_server):
        """Should accept connections on /ws/control-plane path."""
        ws = await connect_client(stream_server.port, path="/ws/control-plane")
        try:
            message = await receive_with_timeout(ws)
            assert message["type"] == "connected"
        finally:
            await ws.close()
