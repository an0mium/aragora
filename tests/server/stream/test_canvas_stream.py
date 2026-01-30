"""
Tests for CanvasStreamServer WebSocket stream module.

Tests cover:
- Canvas session creation and management
- Real-time updates broadcasting
- Connection lifecycle (join, leave, disconnect)
- Concurrent collaboration scenarios
- Session cleanup and expiry
- Event types (cursor_move, selection_change, content_update, etc.)
- User presence tracking
- Error handling
"""

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ===========================================================================
# Helper Classes
# ===========================================================================


class AsyncIteratorMock:
    """Mock for async iteration over WebSocket messages."""

    def __init__(self, messages=None, raise_on_iterate=None):
        self.messages = messages or []
        self.index = 0
        self.raise_on_iterate = raise_on_iterate

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.raise_on_iterate:
            raise self.raise_on_iterate
        if self.index >= len(self.messages):
            raise StopAsyncIteration
        msg = self.messages[self.index]
        self.index += 1
        return msg


# ===========================================================================
# Fixtures
# ===========================================================================


def create_mock_websocket(path="/ws/canvas/test-canvas-123", messages=None):
    """Create a mock WebSocket connection with async iteration support."""
    ws = MagicMock()
    ws.send = AsyncMock()
    ws.close = AsyncMock()
    ws.path = path
    # Set up async iteration
    iterator = AsyncIteratorMock(messages or [])
    ws.__aiter__ = lambda self=iterator: iterator
    ws.__anext__ = lambda self=iterator: iterator.__anext__()
    return ws


@pytest.fixture
def mock_websocket():
    """Create a mock WebSocket connection."""
    return create_mock_websocket()


@pytest.fixture
def mock_websocket_factory():
    """Factory to create multiple mock WebSocket connections."""

    def factory(path="/ws/canvas/test-canvas-123", messages=None):
        return create_mock_websocket(path, messages)

    return factory


@pytest.fixture
def mock_canvas_manager():
    """Create a mock CanvasStateManager."""
    manager = MagicMock()
    manager.get_or_create_canvas = AsyncMock()
    manager.get_state = AsyncMock(
        return_value={
            "canvas": {
                "id": "test-canvas-123",
                "name": "Test Canvas",
                "nodes": [],
                "edges": [],
            },
            "selections": {},
        }
    )
    manager.subscribe = AsyncMock(return_value=True)
    manager.unsubscribe = AsyncMock(return_value=True)
    manager.sync_state = AsyncMock()
    manager.add_node = AsyncMock()
    manager.update_node = AsyncMock()
    manager.move_node = AsyncMock()
    manager.resize_node = AsyncMock()
    manager.delete_node = AsyncMock()
    manager.select_node = AsyncMock()
    manager.add_edge = AsyncMock()
    manager.update_edge = AsyncMock()
    manager.delete_edge = AsyncMock()
    manager.execute_action = AsyncMock(return_value={"success": True})
    return manager


@pytest.fixture
def canvas_stream_server(mock_canvas_manager):
    """Create a CanvasStreamServer with mocked manager."""
    from aragora.server.stream.canvas_stream import CanvasStreamServer

    server = CanvasStreamServer(port=8767)
    server._manager = mock_canvas_manager
    return server


# ===========================================================================
# Server Initialization Tests
# ===========================================================================


class TestCanvasStreamServerInit:
    """Tests for CanvasStreamServer initialization."""

    def test_server_init_default_values(self):
        """Test server initializes with default values."""
        from aragora.server.stream.canvas_stream import CanvasStreamServer

        server = CanvasStreamServer()

        assert server.port == 8767
        assert server.host == "0.0.0.0"
        assert server._clients == {}
        assert server._user_info == {}
        assert server._running is False
        assert server._server is None
        assert server._manager is None

    def test_server_init_custom_values(self):
        """Test server initializes with custom port and host."""
        from aragora.server.stream.canvas_stream import CanvasStreamServer

        server = CanvasStreamServer(port=9000, host="127.0.0.1")

        assert server.port == 9000
        assert server.host == "127.0.0.1"

    def test_get_manager_creates_singleton(self):
        """Test _get_manager creates and caches the manager."""
        from aragora.server.stream.canvas_stream import CanvasStreamServer

        server = CanvasStreamServer()

        # First call creates manager
        manager1 = server._get_manager()
        assert manager1 is not None

        # Second call uses cached manager (same instance)
        manager2 = server._get_manager()
        assert manager2 is manager1
        assert server._manager is manager1


# ===========================================================================
# Server Start/Stop Tests
# ===========================================================================


class TestServerLifecycle:
    """Tests for server start and stop operations."""

    @pytest.mark.asyncio
    async def test_server_start_logs_error_without_websockets(self, canvas_stream_server):
        """Test server handles missing websockets package gracefully."""
        # Create a fresh server without websockets imported
        from aragora.server.stream.canvas_stream import CanvasStreamServer

        server = CanvasStreamServer()

        # Patch the import inside the start method to raise ImportError
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "websockets":
                raise ImportError("No module named 'websockets'")
            return original_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", mock_import):
            # This should not raise, just log error and return
            await server.start()
            # Server should not be running
            assert server._running is False

    @pytest.mark.asyncio
    async def test_server_stop_closes_connections(self, canvas_stream_server):
        """Test server stop closes all connections."""
        mock_server = MagicMock()
        mock_server.close = MagicMock()
        mock_server.wait_closed = AsyncMock()

        canvas_stream_server._running = True
        canvas_stream_server._server = mock_server

        await canvas_stream_server.stop()

        assert canvas_stream_server._running is False
        mock_server.close.assert_called_once()
        mock_server.wait_closed.assert_called_once()

    @pytest.mark.asyncio
    async def test_server_stop_without_server(self, canvas_stream_server):
        """Test server stop handles no active server."""
        canvas_stream_server._running = True
        canvas_stream_server._server = None

        # Should not raise
        await canvas_stream_server.stop()
        assert canvas_stream_server._running is False


# ===========================================================================
# Connection Handling Tests
# ===========================================================================


class TestConnectionHandling:
    """Tests for WebSocket connection handling."""

    @pytest.mark.asyncio
    async def test_handle_standalone_connection_valid_path(
        self, canvas_stream_server, mock_websocket_factory
    ):
        """Test handling standalone connection with valid path."""
        mock_websocket = mock_websocket_factory(path="/ws/canvas/my-canvas-id")

        with patch.object(
            canvas_stream_server, "handle_connection", new_callable=AsyncMock
        ) as mock_handle:
            await canvas_stream_server._handle_standalone_connection(mock_websocket)
            mock_handle.assert_called_once_with(mock_websocket, "my-canvas-id")

    @pytest.mark.asyncio
    async def test_handle_standalone_connection_invalid_path(
        self, canvas_stream_server, mock_websocket_factory
    ):
        """Test handling standalone connection with invalid path."""
        mock_websocket = mock_websocket_factory(path="/invalid/path")

        await canvas_stream_server._handle_standalone_connection(mock_websocket)

        mock_websocket.close.assert_called_once_with(
            1003, "Invalid path. Use /ws/canvas/{canvas_id}"
        )

    @pytest.mark.asyncio
    async def test_handle_connection_registers_client(
        self, canvas_stream_server, mock_websocket_factory, mock_canvas_manager
    ):
        """Test connection registration adds client to tracking."""
        canvas_id = "test-canvas-123"
        mock_websocket = mock_websocket_factory()

        await canvas_stream_server.handle_connection(mock_websocket, canvas_id)

        # Verify canvas was created/retrieved
        mock_canvas_manager.get_or_create_canvas.assert_called_once_with(canvas_id)

        # Verify state was sent
        assert mock_websocket.send.called

    @pytest.mark.asyncio
    async def test_handle_connection_with_user_id(
        self, canvas_stream_server, mock_websocket_factory, mock_canvas_manager
    ):
        """Test connection with explicit user ID."""
        canvas_id = "test-canvas-123"
        user_id = "user-456"
        mock_websocket = mock_websocket_factory()

        await canvas_stream_server.handle_connection(mock_websocket, canvas_id, user_id=user_id)

        # Check user info was stored (before cleanup)
        # Since the connection ended, we verify the initial state was sent
        assert mock_websocket.send.called

    @pytest.mark.asyncio
    async def test_handle_connection_error_cleanup(self, canvas_stream_server, mock_canvas_manager):
        """Test connection cleanup on error."""
        canvas_id = "test-canvas-123"

        # Create a websocket that raises an error on iteration
        ws = MagicMock()
        ws.send = AsyncMock()
        ws.close = AsyncMock()
        ws.path = "/ws/canvas/test"

        # Create iterator that raises ConnectionError
        error_iterator = AsyncIteratorMock(raise_on_iterate=ConnectionError("Lost"))
        ws.__aiter__ = lambda self=error_iterator: error_iterator

        # Should not raise, error should be caught
        await canvas_stream_server.handle_connection(ws, canvas_id)


# ===========================================================================
# Client Registration Tests
# ===========================================================================


class TestClientRegistration:
    """Tests for client registration and unregistration."""

    @pytest.mark.asyncio
    async def test_register_client_new_canvas(self, canvas_stream_server, mock_websocket):
        """Test registering first client for a canvas."""
        canvas_id = "new-canvas"

        await canvas_stream_server._register_client(canvas_id, mock_websocket)

        assert canvas_id in canvas_stream_server._clients
        assert mock_websocket in canvas_stream_server._clients[canvas_id]
        assert len(canvas_stream_server._clients[canvas_id]) == 1

    @pytest.mark.asyncio
    async def test_register_multiple_clients(self, canvas_stream_server, mock_websocket_factory):
        """Test registering multiple clients for same canvas."""
        canvas_id = "shared-canvas"
        ws1 = mock_websocket_factory()
        ws2 = mock_websocket_factory()
        ws3 = mock_websocket_factory()

        await canvas_stream_server._register_client(canvas_id, ws1)
        await canvas_stream_server._register_client(canvas_id, ws2)
        await canvas_stream_server._register_client(canvas_id, ws3)

        assert len(canvas_stream_server._clients[canvas_id]) == 3

    @pytest.mark.asyncio
    async def test_unregister_client(self, canvas_stream_server, mock_websocket):
        """Test unregistering a client."""
        canvas_id = "test-canvas"

        await canvas_stream_server._register_client(canvas_id, mock_websocket)
        await canvas_stream_server._unregister_client(canvas_id, mock_websocket)

        # Canvas should be removed when no clients remain
        assert canvas_id not in canvas_stream_server._clients

    @pytest.mark.asyncio
    async def test_unregister_partial(self, canvas_stream_server, mock_websocket_factory):
        """Test unregistering one client when multiple exist."""
        canvas_id = "shared-canvas"
        ws1 = mock_websocket_factory()
        ws2 = mock_websocket_factory()

        await canvas_stream_server._register_client(canvas_id, ws1)
        await canvas_stream_server._register_client(canvas_id, ws2)
        await canvas_stream_server._unregister_client(canvas_id, ws1)

        assert canvas_id in canvas_stream_server._clients
        assert len(canvas_stream_server._clients[canvas_id]) == 1
        assert ws2 in canvas_stream_server._clients[canvas_id]

    @pytest.mark.asyncio
    async def test_unregister_nonexistent_canvas(self, canvas_stream_server, mock_websocket):
        """Test unregistering from non-existent canvas."""
        # Should not raise
        await canvas_stream_server._unregister_client("nonexistent", mock_websocket)


# ===========================================================================
# Message Handling Tests
# ===========================================================================


class TestMessageHandling:
    """Tests for WebSocket message handling."""

    @pytest.mark.asyncio
    async def test_handle_message_invalid_json(self, canvas_stream_server, mock_websocket):
        """Test handling invalid JSON message."""
        canvas_stream_server._user_info[mock_websocket] = {
            "user_id": "user-1",
            "canvas_id": "canvas-1",
        }

        await canvas_stream_server._handle_message(mock_websocket, "not valid json")

        # Should send error
        mock_websocket.send.assert_called_once()
        call_data = json.loads(mock_websocket.send.call_args[0][0])
        assert call_data["type"] == "canvas:error"
        assert "Invalid JSON" in call_data["error"]

    @pytest.mark.asyncio
    async def test_handle_message_no_canvas(self, canvas_stream_server, mock_websocket):
        """Test handling message when not connected to canvas."""
        canvas_stream_server._user_info[mock_websocket] = {"user_id": "user-1"}

        await canvas_stream_server._handle_message(mock_websocket, json.dumps({"type": "ping"}))

        call_data = json.loads(mock_websocket.send.call_args[0][0])
        assert call_data["type"] == "canvas:error"
        assert "Not connected" in call_data["error"]

    @pytest.mark.asyncio
    async def test_handle_message_no_user_id(self, canvas_stream_server, mock_websocket):
        """Test handling message when user ID not available."""
        canvas_stream_server._user_info[mock_websocket] = {"canvas_id": "canvas-1"}

        await canvas_stream_server._handle_message(
            mock_websocket, json.dumps({"type": "canvas:node:create"})
        )

        call_data = json.loads(mock_websocket.send.call_args[0][0])
        assert call_data["type"] == "canvas:error"
        assert "User ID" in call_data["error"]

    @pytest.mark.asyncio
    async def test_handle_ping_message(
        self, canvas_stream_server, mock_websocket, mock_canvas_manager
    ):
        """Test handling ping message."""
        canvas_stream_server._user_info[mock_websocket] = {
            "user_id": "user-1",
            "canvas_id": "canvas-1",
        }

        await canvas_stream_server._handle_message(mock_websocket, json.dumps({"type": "ping"}))

        call_data = json.loads(mock_websocket.send.call_args[0][0])
        assert call_data["type"] == "pong"
        assert "timestamp" in call_data

    @pytest.mark.asyncio
    async def test_handle_sync_message(
        self, canvas_stream_server, mock_websocket, mock_canvas_manager
    ):
        """Test handling canvas sync message."""
        canvas_stream_server._user_info[mock_websocket] = {
            "user_id": "user-1",
            "canvas_id": "canvas-1",
        }

        await canvas_stream_server._handle_message(
            mock_websocket, json.dumps({"type": "canvas:sync"})
        )

        mock_canvas_manager.sync_state.assert_called_once_with("canvas-1", "user-1")

    @pytest.mark.asyncio
    async def test_handle_unknown_message_type(
        self, canvas_stream_server, mock_websocket, mock_canvas_manager
    ):
        """Test handling unknown message type."""
        canvas_stream_server._user_info[mock_websocket] = {
            "user_id": "user-1",
            "canvas_id": "canvas-1",
        }

        # Should not raise, just log
        await canvas_stream_server._handle_message(
            mock_websocket, json.dumps({"type": "unknown:type"})
        )


# ===========================================================================
# Node Operation Tests
# ===========================================================================


class TestNodeOperations:
    """Tests for node operation message handling."""

    @pytest.mark.asyncio
    async def test_handle_node_create(
        self, canvas_stream_server, mock_websocket, mock_canvas_manager
    ):
        """Test handling node creation message."""
        canvas_stream_server._user_info[mock_websocket] = {
            "user_id": "user-1",
            "canvas_id": "canvas-1",
        }

        message = json.dumps(
            {
                "type": "canvas:node:create",
                "node": {
                    "type": "text",
                    "position": {"x": 100, "y": 200},
                    "label": "Test Node",
                    "data": {"content": "Hello"},
                },
            }
        )

        await canvas_stream_server._handle_message(mock_websocket, message)

        mock_canvas_manager.add_node.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_node_update(
        self, canvas_stream_server, mock_websocket, mock_canvas_manager
    ):
        """Test handling node update message."""
        canvas_stream_server._user_info[mock_websocket] = {
            "user_id": "user-1",
            "canvas_id": "canvas-1",
        }

        message = json.dumps(
            {
                "type": "canvas:node:update",
                "node_id": "node-123",
                "updates": {"label": "Updated Label"},
            }
        )

        await canvas_stream_server._handle_message(mock_websocket, message)

        mock_canvas_manager.update_node.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_node_move(
        self, canvas_stream_server, mock_websocket, mock_canvas_manager
    ):
        """Test handling node move message."""
        canvas_stream_server._user_info[mock_websocket] = {
            "user_id": "user-1",
            "canvas_id": "canvas-1",
        }

        message = json.dumps(
            {
                "type": "canvas:node:move",
                "node_id": "node-123",
                "position": {"x": 300, "y": 400},
            }
        )

        await canvas_stream_server._handle_message(mock_websocket, message)

        mock_canvas_manager.move_node.assert_called_once_with(
            canvas_id="canvas-1",
            node_id="node-123",
            x=300,
            y=400,
            user_id="user-1",
        )

    @pytest.mark.asyncio
    async def test_handle_node_resize(
        self, canvas_stream_server, mock_websocket, mock_canvas_manager
    ):
        """Test handling node resize message."""
        canvas_stream_server._user_info[mock_websocket] = {
            "user_id": "user-1",
            "canvas_id": "canvas-1",
        }

        message = json.dumps(
            {
                "type": "canvas:node:resize",
                "node_id": "node-123",
                "size": {"width": 300, "height": 200},
            }
        )

        await canvas_stream_server._handle_message(mock_websocket, message)

        mock_canvas_manager.resize_node.assert_called_once_with(
            canvas_id="canvas-1",
            node_id="node-123",
            width=300,
            height=200,
            user_id="user-1",
        )

    @pytest.mark.asyncio
    async def test_handle_node_delete(
        self, canvas_stream_server, mock_websocket, mock_canvas_manager
    ):
        """Test handling node delete message."""
        canvas_stream_server._user_info[mock_websocket] = {
            "user_id": "user-1",
            "canvas_id": "canvas-1",
        }

        message = json.dumps({"type": "canvas:node:delete", "node_id": "node-123"})

        await canvas_stream_server._handle_message(mock_websocket, message)

        mock_canvas_manager.delete_node.assert_called_once_with(
            canvas_id="canvas-1",
            node_id="node-123",
            user_id="user-1",
        )

    @pytest.mark.asyncio
    async def test_handle_node_select(
        self, canvas_stream_server, mock_websocket, mock_canvas_manager
    ):
        """Test handling node select message."""
        canvas_stream_server._user_info[mock_websocket] = {
            "user_id": "user-1",
            "canvas_id": "canvas-1",
        }

        message = json.dumps(
            {"type": "canvas:node:select", "node_id": "node-123", "multi_select": True}
        )

        await canvas_stream_server._handle_message(mock_websocket, message)

        mock_canvas_manager.select_node.assert_called_once_with(
            canvas_id="canvas-1",
            node_id="node-123",
            user_id="user-1",
            multi_select=True,
        )

    @pytest.mark.asyncio
    async def test_handle_node_operation_no_node_id(
        self, canvas_stream_server, mock_websocket, mock_canvas_manager
    ):
        """Test node operation without node_id is ignored."""
        canvas_stream_server._user_info[mock_websocket] = {
            "user_id": "user-1",
            "canvas_id": "canvas-1",
        }

        # Missing node_id
        message = json.dumps({"type": "canvas:node:delete"})

        await canvas_stream_server._handle_message(mock_websocket, message)

        mock_canvas_manager.delete_node.assert_not_called()


# ===========================================================================
# Edge Operation Tests
# ===========================================================================


class TestEdgeOperations:
    """Tests for edge operation message handling."""

    @pytest.mark.asyncio
    async def test_handle_edge_create(
        self, canvas_stream_server, mock_websocket, mock_canvas_manager
    ):
        """Test handling edge creation message."""
        canvas_stream_server._user_info[mock_websocket] = {
            "user_id": "user-1",
            "canvas_id": "canvas-1",
        }

        message = json.dumps(
            {
                "type": "canvas:edge:create",
                "source": "node-1",
                "target": "node-2",
                "edge_type": "data_flow",
                "label": "Connection",
            }
        )

        await canvas_stream_server._handle_message(mock_websocket, message)

        mock_canvas_manager.add_edge.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_edge_create_with_source_id(
        self, canvas_stream_server, mock_websocket, mock_canvas_manager
    ):
        """Test handling edge creation with source_id/target_id format."""
        canvas_stream_server._user_info[mock_websocket] = {
            "user_id": "user-1",
            "canvas_id": "canvas-1",
        }

        message = json.dumps(
            {
                "type": "canvas:edge:create",
                "source_id": "node-1",
                "target_id": "node-2",
            }
        )

        await canvas_stream_server._handle_message(mock_websocket, message)

        mock_canvas_manager.add_edge.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_edge_update(
        self, canvas_stream_server, mock_websocket, mock_canvas_manager
    ):
        """Test handling edge update message."""
        canvas_stream_server._user_info[mock_websocket] = {
            "user_id": "user-1",
            "canvas_id": "canvas-1",
        }

        message = json.dumps(
            {
                "type": "canvas:edge:update",
                "edge_id": "edge-123",
                "updates": {"label": "Updated Connection"},
            }
        )

        await canvas_stream_server._handle_message(mock_websocket, message)

        mock_canvas_manager.update_edge.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_edge_delete(
        self, canvas_stream_server, mock_websocket, mock_canvas_manager
    ):
        """Test handling edge delete message."""
        canvas_stream_server._user_info[mock_websocket] = {
            "user_id": "user-1",
            "canvas_id": "canvas-1",
        }

        message = json.dumps({"type": "canvas:edge:delete", "edge_id": "edge-123"})

        await canvas_stream_server._handle_message(mock_websocket, message)

        mock_canvas_manager.delete_edge.assert_called_once_with(
            canvas_id="canvas-1",
            edge_id="edge-123",
            user_id="user-1",
        )

    @pytest.mark.asyncio
    async def test_handle_edge_create_missing_nodes(
        self, canvas_stream_server, mock_websocket, mock_canvas_manager
    ):
        """Test edge creation without source/target is ignored."""
        canvas_stream_server._user_info[mock_websocket] = {
            "user_id": "user-1",
            "canvas_id": "canvas-1",
        }

        message = json.dumps({"type": "canvas:edge:create", "edge_type": "data_flow"})

        await canvas_stream_server._handle_message(mock_websocket, message)

        mock_canvas_manager.add_edge.assert_not_called()


# ===========================================================================
# Action Handling Tests
# ===========================================================================


class TestActionHandling:
    """Tests for canvas action handling."""

    @pytest.mark.asyncio
    async def test_handle_action(self, canvas_stream_server, mock_websocket, mock_canvas_manager):
        """Test handling canvas action message."""
        canvas_stream_server._user_info[mock_websocket] = {
            "user_id": "user-1",
            "canvas_id": "canvas-1",
        }

        message = json.dumps(
            {
                "type": "canvas:action",
                "action": "start_debate",
                "params": {"question": "What is the best approach?"},
            }
        )

        await canvas_stream_server._handle_message(mock_websocket, message)

        mock_canvas_manager.execute_action.assert_called_once_with(
            canvas_id="canvas-1",
            action="start_debate",
            params={"question": "What is the best approach?"},
            user_id="user-1",
        )

    @pytest.mark.asyncio
    async def test_handle_action_no_action_name(
        self, canvas_stream_server, mock_websocket, mock_canvas_manager
    ):
        """Test action handling without action name is ignored."""
        canvas_stream_server._user_info[mock_websocket] = {
            "user_id": "user-1",
            "canvas_id": "canvas-1",
        }

        message = json.dumps({"type": "canvas:action", "params": {}})

        await canvas_stream_server._handle_message(mock_websocket, message)

        mock_canvas_manager.execute_action.assert_not_called()


# ===========================================================================
# Broadcasting Tests
# ===========================================================================


class TestBroadcasting:
    """Tests for message broadcasting to clients."""

    @pytest.mark.asyncio
    async def test_broadcast_to_canvas(self, canvas_stream_server, mock_websocket_factory):
        """Test broadcasting message to all canvas clients."""
        canvas_id = "broadcast-canvas"
        ws1 = mock_websocket_factory()
        ws2 = mock_websocket_factory()
        ws3 = mock_websocket_factory()

        await canvas_stream_server._register_client(canvas_id, ws1)
        await canvas_stream_server._register_client(canvas_id, ws2)
        await canvas_stream_server._register_client(canvas_id, ws3)

        message = {"type": "test:broadcast", "data": "Hello all!"}
        await canvas_stream_server.broadcast_to_canvas(canvas_id, message)

        # All clients should receive the message
        ws1.send.assert_called_once()
        ws2.send.assert_called_once()
        ws3.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_broadcast_to_empty_canvas(self, canvas_stream_server):
        """Test broadcasting to canvas with no clients."""
        # Should not raise
        await canvas_stream_server.broadcast_to_canvas("empty-canvas", {"type": "test"})

    @pytest.mark.asyncio
    async def test_broadcast_handles_send_error(self, canvas_stream_server, mock_websocket_factory):
        """Test broadcasting handles client send errors gracefully."""
        canvas_id = "error-canvas"
        ws1 = mock_websocket_factory()
        ws2 = mock_websocket_factory()

        # Make ws1 fail on send
        ws1.send = AsyncMock(side_effect=ConnectionError("Connection closed"))

        await canvas_stream_server._register_client(canvas_id, ws1)
        await canvas_stream_server._register_client(canvas_id, ws2)

        # Should not raise
        await canvas_stream_server.broadcast_to_canvas(canvas_id, {"type": "test:broadcast"})

        # ws2 should still receive the message
        ws2.send.assert_called_once()


# ===========================================================================
# Send Message Tests
# ===========================================================================


class TestSendMessage:
    """Tests for sending messages to individual clients."""

    @pytest.mark.asyncio
    async def test_send_message_success(self, canvas_stream_server, mock_websocket):
        """Test sending message to client."""
        message = {"type": "test", "data": "Hello"}

        await canvas_stream_server._send_message(mock_websocket, message)

        mock_websocket.send.assert_called_once_with(json.dumps(message))

    @pytest.mark.asyncio
    async def test_send_message_connection_error(self, canvas_stream_server, mock_websocket):
        """Test sending message handles connection error."""
        mock_websocket.send = AsyncMock(side_effect=ConnectionError("Disconnected"))

        # Should not raise
        await canvas_stream_server._send_message(mock_websocket, {"type": "test"})

    @pytest.mark.asyncio
    async def test_send_message_os_error(self, canvas_stream_server, mock_websocket):
        """Test sending message handles OS error."""
        mock_websocket.send = AsyncMock(side_effect=OSError("Socket error"))

        # Should not raise
        await canvas_stream_server._send_message(mock_websocket, {"type": "test"})

    @pytest.mark.asyncio
    async def test_send_error_message(self, canvas_stream_server, mock_websocket):
        """Test sending error message."""
        await canvas_stream_server._send_error(mock_websocket, "Test error")

        call_data = json.loads(mock_websocket.send.call_args[0][0])
        assert call_data["type"] == "canvas:error"
        assert call_data["error"] == "Test error"
        assert "timestamp" in call_data


# ===========================================================================
# User Presence Tests
# ===========================================================================


class TestUserPresence:
    """Tests for user presence tracking."""

    def test_get_connected_users_empty(self, canvas_stream_server):
        """Test getting users for canvas with no connections."""
        users = canvas_stream_server.get_connected_users("nonexistent")
        assert users == []

    def test_get_connected_users_single(self, canvas_stream_server, mock_websocket):
        """Test getting single connected user."""
        canvas_id = "presence-canvas"
        canvas_stream_server._clients[canvas_id] = {mock_websocket}
        canvas_stream_server._user_info[mock_websocket] = {
            "user_id": "user-123",
            "canvas_id": canvas_id,
            "connected_at": time.time(),
        }

        users = canvas_stream_server.get_connected_users(canvas_id)

        assert len(users) == 1
        assert users[0]["user_id"] == "user-123"
        assert "connected_at" in users[0]

    def test_get_connected_users_multiple(self, canvas_stream_server, mock_websocket_factory):
        """Test getting multiple connected users."""
        canvas_id = "presence-canvas"
        ws1 = mock_websocket_factory()
        ws2 = mock_websocket_factory()

        canvas_stream_server._clients[canvas_id] = {ws1, ws2}
        canvas_stream_server._user_info[ws1] = {
            "user_id": "user-1",
            "canvas_id": canvas_id,
            "connected_at": time.time(),
        }
        canvas_stream_server._user_info[ws2] = {
            "user_id": "user-2",
            "canvas_id": canvas_id,
            "connected_at": time.time(),
        }

        users = canvas_stream_server.get_connected_users(canvas_id)

        assert len(users) == 2
        user_ids = {u["user_id"] for u in users}
        assert user_ids == {"user-1", "user-2"}

    def test_get_connected_users_missing_info(self, canvas_stream_server, mock_websocket):
        """Test getting users when user_info is missing."""
        canvas_id = "presence-canvas"
        canvas_stream_server._clients[canvas_id] = {mock_websocket}
        # No user_info entry

        users = canvas_stream_server.get_connected_users(canvas_id)

        assert users == []


# ===========================================================================
# Concurrent Collaboration Tests
# ===========================================================================


class TestConcurrentCollaboration:
    """Tests for concurrent collaboration scenarios."""

    @pytest.mark.asyncio
    async def test_concurrent_client_registration(
        self, canvas_stream_server, mock_websocket_factory
    ):
        """Test concurrent client registrations."""
        canvas_id = "concurrent-canvas"
        websockets = [mock_websocket_factory() for _ in range(10)]

        # Register concurrently
        await asyncio.gather(
            *[canvas_stream_server._register_client(canvas_id, ws) for ws in websockets]
        )

        assert len(canvas_stream_server._clients[canvas_id]) == 10

    @pytest.mark.asyncio
    async def test_concurrent_client_unregistration(
        self, canvas_stream_server, mock_websocket_factory
    ):
        """Test concurrent client unregistrations."""
        canvas_id = "concurrent-canvas"
        websockets = [mock_websocket_factory() for _ in range(10)]

        # Register all
        for ws in websockets:
            await canvas_stream_server._register_client(canvas_id, ws)

        # Unregister concurrently
        await asyncio.gather(
            *[canvas_stream_server._unregister_client(canvas_id, ws) for ws in websockets]
        )

        assert canvas_id not in canvas_stream_server._clients

    @pytest.mark.asyncio
    async def test_concurrent_broadcasts(self, canvas_stream_server, mock_websocket_factory):
        """Test concurrent broadcast operations."""
        canvas_id = "concurrent-canvas"
        websockets = [mock_websocket_factory() for _ in range(5)]

        for ws in websockets:
            await canvas_stream_server._register_client(canvas_id, ws)

        # Broadcast multiple messages concurrently
        messages = [{"type": "test", "id": i} for i in range(10)]
        await asyncio.gather(
            *[canvas_stream_server.broadcast_to_canvas(canvas_id, msg) for msg in messages]
        )

        # Each websocket should receive all messages
        for ws in websockets:
            assert ws.send.call_count == 10


# ===========================================================================
# Event Handling Tests
# ===========================================================================


class TestEventHandling:
    """Tests for canvas event handling."""

    @pytest.mark.asyncio
    async def test_send_event_to_client(self, canvas_stream_server, mock_websocket):
        """Test sending canvas event to client."""
        from aragora.canvas.models import CanvasEvent, CanvasEventType

        event = CanvasEvent(
            event_type=CanvasEventType.NODE_CREATE,
            canvas_id="test-canvas",
            node_id="node-123",
            data={"label": "New Node"},
        )

        await canvas_stream_server._send_event_to_client(mock_websocket, event)

        mock_websocket.send.assert_called_once()
        call_data = json.loads(mock_websocket.send.call_args[0][0])
        assert call_data["type"] == "canvas:node:create"
        assert call_data["canvas_id"] == "test-canvas"
        assert call_data["node_id"] == "node-123"

    @pytest.mark.asyncio
    async def test_send_event_connection_error(self, canvas_stream_server, mock_websocket):
        """Test sending event handles connection error."""
        from aragora.canvas.models import CanvasEvent, CanvasEventType

        mock_websocket.send = AsyncMock(side_effect=ConnectionError("Lost"))

        event = CanvasEvent(
            event_type=CanvasEventType.NODE_CREATE,
            canvas_id="test-canvas",
        )

        # Should not raise
        await canvas_stream_server._send_event_to_client(mock_websocket, event)


# ===========================================================================
# Error Handling Tests
# ===========================================================================


class TestErrorHandling:
    """Tests for error handling scenarios."""

    @pytest.mark.asyncio
    async def test_handle_message_value_error(
        self, canvas_stream_server, mock_websocket, mock_canvas_manager
    ):
        """Test handling ValueError during message processing."""
        canvas_stream_server._user_info[mock_websocket] = {
            "user_id": "user-1",
            "canvas_id": "canvas-1",
        }

        mock_canvas_manager.add_node = AsyncMock(side_effect=ValueError("Invalid data"))

        message = json.dumps(
            {
                "type": "canvas:node:create",
                "node": {"type": "text", "position": {"x": 0, "y": 0}},
            }
        )

        await canvas_stream_server._handle_message(mock_websocket, message)

        # Error should be sent
        call_data = json.loads(mock_websocket.send.call_args[0][0])
        assert call_data["type"] == "canvas:error"

    @pytest.mark.asyncio
    async def test_handle_message_key_error(
        self, canvas_stream_server, mock_websocket, mock_canvas_manager
    ):
        """Test handling KeyError during message processing."""
        canvas_stream_server._user_info[mock_websocket] = {
            "user_id": "user-1",
            "canvas_id": "canvas-1",
        }

        mock_canvas_manager.update_node = AsyncMock(side_effect=KeyError("missing_key"))

        message = json.dumps({"type": "canvas:node:update", "node_id": "node-1", "updates": {}})

        await canvas_stream_server._handle_message(mock_websocket, message)

        call_data = json.loads(mock_websocket.send.call_args[0][0])
        assert call_data["type"] == "canvas:error"


# ===========================================================================
# Global Server Instance Tests
# ===========================================================================


class TestGlobalServerInstance:
    """Tests for global server instance management."""

    def test_get_canvas_stream_server_creates_singleton(self):
        """Test get_canvas_stream_server creates and caches instance."""
        import aragora.server.stream.canvas_stream as module

        # Reset global state
        original = module._canvas_server
        module._canvas_server = None

        try:
            server1 = module.get_canvas_stream_server()
            server2 = module.get_canvas_stream_server()

            assert server1 is server2
            assert isinstance(server1, module.CanvasStreamServer)
        finally:
            # Restore original state
            module._canvas_server = original


# ===========================================================================
# Standalone Connection Path Extraction Tests
# ===========================================================================


class TestPathExtraction:
    """Tests for WebSocket path extraction in standalone mode."""

    @pytest.mark.asyncio
    async def test_path_from_websocket_attribute(self, canvas_stream_server):
        """Test extracting path from websocket.path attribute."""
        ws = MagicMock()
        ws.path = "/ws/canvas/from-attribute"
        ws.__aiter__ = MagicMock(return_value=iter([]))

        with patch.object(
            canvas_stream_server, "handle_connection", new_callable=AsyncMock
        ) as mock_handle:
            await canvas_stream_server._handle_standalone_connection(ws)
            mock_handle.assert_called_once_with(ws, "from-attribute")

    @pytest.mark.asyncio
    async def test_path_from_request_object(self, canvas_stream_server):
        """Test extracting path from websocket.request.path."""
        ws = MagicMock()
        ws.send = AsyncMock()
        ws.close = AsyncMock()
        # Delete the path attribute so getattr falls back to request
        del ws.path
        ws.request = MagicMock()
        ws.request.path = "/ws/canvas/from-request"

        with patch.object(
            canvas_stream_server, "handle_connection", new_callable=AsyncMock
        ) as mock_handle:
            await canvas_stream_server._handle_standalone_connection(ws)
            mock_handle.assert_called_once_with(ws, "from-request")

    @pytest.mark.asyncio
    async def test_path_fallback_to_root(self, canvas_stream_server):
        """Test path fallback to root when not available."""
        ws = MagicMock()
        ws.path = None
        del ws.request
        ws.close = AsyncMock()

        await canvas_stream_server._handle_standalone_connection(ws)

        ws.close.assert_called_once_with(1003, "Invalid path. Use /ws/canvas/{canvas_id}")


# ===========================================================================
# Session Cleanup Tests
# ===========================================================================


class TestSessionCleanup:
    """Tests for session cleanup on disconnect."""

    @pytest.mark.asyncio
    async def test_cleanup_on_cancelled_error(self, canvas_stream_server, mock_canvas_manager):
        """Test cleanup when connection is cancelled."""
        canvas_id = "cleanup-canvas"

        # Create a websocket that raises CancelledError on iteration
        ws = MagicMock()
        ws.send = AsyncMock()
        ws.close = AsyncMock()
        ws.path = "/ws/canvas/test"

        error_iterator = AsyncIteratorMock(raise_on_iterate=asyncio.CancelledError())
        ws.__aiter__ = lambda self=error_iterator: error_iterator

        await canvas_stream_server.handle_connection(ws, canvas_id)

        # User info should be cleaned up
        assert ws not in canvas_stream_server._user_info

    @pytest.mark.asyncio
    async def test_cleanup_removes_from_clients(
        self, canvas_stream_server, mock_websocket_factory, mock_canvas_manager
    ):
        """Test cleanup removes client from tracking."""
        canvas_id = "cleanup-canvas"
        mock_websocket = mock_websocket_factory()

        await canvas_stream_server.handle_connection(mock_websocket, canvas_id)

        # Client should be cleaned up (canvas has no more clients)
        assert canvas_id not in canvas_stream_server._clients


# ===========================================================================
# Initial State Sync Tests
# ===========================================================================


class TestInitialStateSync:
    """Tests for initial state synchronization on connect."""

    @pytest.mark.asyncio
    async def test_initial_state_sent_on_connect(
        self, canvas_stream_server, mock_websocket_factory, mock_canvas_manager
    ):
        """Test initial canvas state is sent to client on connect."""
        canvas_id = "state-canvas"
        mock_canvas_manager.get_state.return_value = {
            "canvas": {
                "id": canvas_id,
                "name": "Test",
                "nodes": [{"id": "n1", "type": "text"}],
                "edges": [],
            },
            "selections": {"user-1": ["n1"]},
        }
        mock_websocket = mock_websocket_factory()

        await canvas_stream_server.handle_connection(mock_websocket, canvas_id)

        # Verify state was sent
        call_data = json.loads(mock_websocket.send.call_args[0][0])
        assert call_data["type"] == "canvas:state"
        assert call_data["canvas_id"] == canvas_id
        assert "data" in call_data
        assert "timestamp" in call_data
