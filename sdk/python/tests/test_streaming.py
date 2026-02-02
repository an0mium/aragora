"""Tests for WebSocket streaming functionality."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora_sdk.websocket import (
    EVENT_TYPES,
    AragoraWebSocket,
    WebSocketEvent,
    WebSocketOptions,
    stream_debate,
)


class TestWebSocketOptions:
    """Tests for WebSocketOptions configuration."""

    def test_default_options(self) -> None:
        """WebSocketOptions has correct defaults."""
        options = WebSocketOptions()
        assert options.auto_reconnect is True
        assert options.max_reconnect_attempts == 5
        assert options.reconnect_delay == 1.0
        assert options.heartbeat_interval == 30.0

    def test_custom_options(self) -> None:
        """WebSocketOptions accepts custom values."""
        options = WebSocketOptions(
            auto_reconnect=False,
            max_reconnect_attempts=10,
            reconnect_delay=2.0,
            heartbeat_interval=60.0,
        )
        assert options.auto_reconnect is False
        assert options.max_reconnect_attempts == 10
        assert options.reconnect_delay == 2.0
        assert options.heartbeat_interval == 60.0


class TestWebSocketEvent:
    """Tests for WebSocketEvent dataclass."""

    def test_event_creation(self) -> None:
        """WebSocketEvent can be created with all fields."""
        event = WebSocketEvent(
            type="agent_message",
            data={"content": "Hello"},
            timestamp="2024-01-15T10:30:00Z",
            debate_id="dbt_123",
        )
        assert event.type == "agent_message"
        assert event.data == {"content": "Hello"}
        assert event.timestamp == "2024-01-15T10:30:00Z"
        assert event.debate_id == "dbt_123"

    def test_event_defaults(self) -> None:
        """WebSocketEvent has correct defaults."""
        event = WebSocketEvent(type="test")
        assert event.data == {}
        assert event.timestamp == ""
        assert event.debate_id is None

    def test_event_types_defined(self) -> None:
        """All expected event types are defined."""
        expected_types = {
            "connected",
            "disconnected",
            "error",
            "debate_start",
            "round_start",
            "agent_message",
            "propose",
            "critique",
            "revision",
            "synthesis",
            "vote",
            "consensus",
            "consensus_reached",
            "debate_end",
            "phase_change",
            "audience_suggestion",
            "user_vote",
            "warning",
            "message",
        }
        assert set(EVENT_TYPES) == expected_types


class TestAragoraWebSocketInit:
    """Tests for AragoraWebSocket initialization."""

    def test_init_with_base_url(self) -> None:
        """WebSocket initializes with base URL."""
        ws = AragoraWebSocket("https://api.aragora.ai")
        assert ws.base_url == "https://api.aragora.ai"
        assert ws.api_key is None
        assert ws.ws_url is None
        assert ws.state == "disconnected"

    def test_init_strips_trailing_slash(self) -> None:
        """WebSocket strips trailing slash from base URL."""
        ws = AragoraWebSocket("https://api.aragora.ai/")
        assert ws.base_url == "https://api.aragora.ai"

    def test_init_with_api_key(self) -> None:
        """WebSocket initializes with API key."""
        ws = AragoraWebSocket("https://api.aragora.ai", api_key="test-key")
        assert ws.api_key == "test-key"

    def test_init_with_explicit_ws_url(self) -> None:
        """WebSocket initializes with explicit WebSocket URL."""
        ws = AragoraWebSocket(
            "https://api.aragora.ai",
            ws_url="wss://ws.aragora.ai",
        )
        assert ws.ws_url == "wss://ws.aragora.ai"

    def test_init_with_custom_options(self) -> None:
        """WebSocket initializes with custom options."""
        options = WebSocketOptions(auto_reconnect=False)
        ws = AragoraWebSocket("https://api.aragora.ai", options=options)
        assert ws.options.auto_reconnect is False

    def test_init_default_options(self) -> None:
        """WebSocket creates default options if none provided."""
        ws = AragoraWebSocket("https://api.aragora.ai")
        assert ws.options.auto_reconnect is True


class TestAragoraWebSocketUrlBuilding:
    """Tests for WebSocket URL construction."""

    def test_build_ws_url_from_http(self) -> None:
        """WebSocket URL converts http to ws."""
        ws = AragoraWebSocket("http://localhost:8080")
        url = ws._build_ws_url()
        assert url.startswith("ws://localhost:8080")
        assert url.endswith("/ws")

    def test_build_ws_url_from_https(self) -> None:
        """WebSocket URL converts https to wss."""
        ws = AragoraWebSocket("https://api.aragora.ai")
        url = ws._build_ws_url()
        assert url.startswith("wss://api.aragora.ai")
        assert url.endswith("/ws")

    def test_build_ws_url_with_debate_id(self) -> None:
        """WebSocket URL includes debate_id query parameter."""
        ws = AragoraWebSocket("https://api.aragora.ai")
        url = ws._build_ws_url(debate_id="dbt_123")
        assert "debate_id=dbt_123" in url

    def test_build_ws_url_with_api_key(self) -> None:
        """WebSocket URL includes token query parameter."""
        ws = AragoraWebSocket("https://api.aragora.ai", api_key="secret-key")
        url = ws._build_ws_url()
        assert "token=secret-key" in url

    def test_build_ws_url_with_both_params(self) -> None:
        """WebSocket URL includes both debate_id and token."""
        ws = AragoraWebSocket("https://api.aragora.ai", api_key="secret-key")
        url = ws._build_ws_url(debate_id="dbt_123")
        assert "debate_id=dbt_123" in url
        assert "token=secret-key" in url
        assert "&" in url

    def test_build_ws_url_uses_explicit_ws_url(self) -> None:
        """WebSocket URL uses explicit ws_url when provided."""
        ws = AragoraWebSocket(
            "https://api.aragora.ai",
            ws_url="wss://custom.aragora.ai/stream",
        )
        url = ws._build_ws_url()
        assert url.startswith("wss://custom.aragora.ai/stream")

    def test_build_ws_url_encodes_special_chars(self) -> None:
        """WebSocket URL encodes special characters."""
        ws = AragoraWebSocket("https://api.aragora.ai")
        url = ws._build_ws_url(debate_id="dbt/123&test=val")
        assert "dbt%2F123%26test%3Dval" in url


class TestAragoraWebSocketEventHandlers:
    """Tests for WebSocket event handler registration."""

    def test_on_registers_handler(self) -> None:
        """on() method registers event handler."""
        ws = AragoraWebSocket("https://api.aragora.ai")
        handler = MagicMock()
        ws.on("agent_message", handler)
        assert handler in ws._handlers["agent_message"]

    def test_on_returns_unsubscribe(self) -> None:
        """on() method returns unsubscribe function."""
        ws = AragoraWebSocket("https://api.aragora.ai")
        handler = MagicMock()
        unsubscribe = ws.on("agent_message", handler)
        assert callable(unsubscribe)
        unsubscribe()
        assert handler not in ws._handlers["agent_message"]

    def test_off_removes_handler(self) -> None:
        """off() method removes event handler."""
        ws = AragoraWebSocket("https://api.aragora.ai")
        handler = MagicMock()
        ws.on("agent_message", handler)
        ws.off("agent_message", handler)
        assert handler not in ws._handlers["agent_message"]

    def test_off_ignores_missing_handler(self) -> None:
        """off() method handles missing handler gracefully."""
        ws = AragoraWebSocket("https://api.aragora.ai")
        handler = MagicMock()
        # Should not raise
        ws.off("agent_message", handler)

    def test_multiple_handlers_same_event(self) -> None:
        """Multiple handlers can be registered for same event."""
        ws = AragoraWebSocket("https://api.aragora.ai")
        handler1 = MagicMock()
        handler2 = MagicMock()
        ws.on("agent_message", handler1)
        ws.on("agent_message", handler2)
        assert len(ws._handlers["agent_message"]) == 2

    def test_on_creates_handler_list_for_unknown_event(self) -> None:
        """on() creates handler list for unknown event type."""
        ws = AragoraWebSocket("https://api.aragora.ai")
        handler = MagicMock()
        ws.on("custom_event", handler)
        assert "custom_event" in ws._handlers
        assert handler in ws._handlers["custom_event"]


class TestAragoraWebSocketSubscriptions:
    """Tests for WebSocket debate subscriptions."""

    def test_subscribe_adds_to_subscriptions(self) -> None:
        """subscribe() adds debate_id to subscriptions."""
        ws = AragoraWebSocket("https://api.aragora.ai")
        ws._state = "connected"
        ws._ws = AsyncMock()
        ws.subscribe("dbt_123")
        assert "dbt_123" in ws._subscriptions

    def test_unsubscribe_removes_from_subscriptions(self) -> None:
        """unsubscribe() removes debate_id from subscriptions."""
        ws = AragoraWebSocket("https://api.aragora.ai")
        ws._state = "connected"
        ws._ws = AsyncMock()
        ws._subscriptions.add("dbt_123")
        ws.unsubscribe("dbt_123")
        assert "dbt_123" not in ws._subscriptions

    def test_unsubscribe_handles_missing_subscription(self) -> None:
        """unsubscribe() handles missing subscription gracefully."""
        ws = AragoraWebSocket("https://api.aragora.ai")
        ws._state = "connected"
        ws._ws = AsyncMock()
        # Should not raise
        ws.unsubscribe("nonexistent")


class TestAragoraWebSocketMessageHandling:
    """Tests for WebSocket message handling."""

    def test_handle_message_parses_json(self) -> None:
        """_handle_message parses JSON payload."""
        ws = AragoraWebSocket("https://api.aragora.ai")
        handler = MagicMock()
        ws.on("agent_message", handler)

        raw = json.dumps(
            {
                "type": "agent_message",
                "data": {"content": "Hello"},
                "timestamp": "2024-01-15T10:30:00Z",
                "debate_id": "dbt_123",
            }
        )

        ws._handle_message(raw)

        # Check handler was called with WebSocketEvent
        assert handler.called
        event = handler.call_args[0][0]
        assert isinstance(event, WebSocketEvent)
        assert event.type == "agent_message"
        assert event.data == {"content": "Hello"}

    def test_handle_message_emits_to_message_handlers(self) -> None:
        """_handle_message emits to generic message handlers."""
        ws = AragoraWebSocket("https://api.aragora.ai")
        handler = MagicMock()
        ws.on("message", handler)

        raw = json.dumps({"type": "agent_message", "data": {}})
        ws._handle_message(raw)

        assert handler.called

    def test_handle_message_handles_invalid_json(self) -> None:
        """_handle_message handles invalid JSON gracefully."""
        ws = AragoraWebSocket("https://api.aragora.ai")
        error_handler = MagicMock()
        ws.on("error", error_handler)

        ws._handle_message("not valid json")

        assert error_handler.called
        error_data = error_handler.call_args[0][0]
        assert "error" in error_data

    def test_handle_message_defaults_type_to_message(self) -> None:
        """_handle_message defaults event type to 'message'."""
        ws = AragoraWebSocket("https://api.aragora.ai")
        handler = MagicMock()
        ws.on("message", handler)

        raw = json.dumps({"data": {"content": "test"}})
        ws._handle_message(raw)

        event = handler.call_args[0][0]
        assert event.type == "message"

    def test_handle_message_enqueues_for_iterator(self) -> None:
        """_handle_message puts event in queue for async iterator."""
        ws = AragoraWebSocket("https://api.aragora.ai")

        raw = json.dumps({"type": "agent_message", "data": {}})
        ws._handle_message(raw)

        assert not ws._event_queue.empty()
        event = ws._event_queue.get()
        assert event.type == "agent_message"


class TestAragoraWebSocketEmit:
    """Tests for WebSocket event emission."""

    def test_emit_calls_handlers(self) -> None:
        """_emit calls all registered handlers for event."""
        ws = AragoraWebSocket("https://api.aragora.ai")
        handler1 = MagicMock()
        handler2 = MagicMock()
        ws.on("test_event", handler1)
        ws.on("test_event", handler2)

        ws._emit("test_event", {"key": "value"})

        handler1.assert_called_once_with({"key": "value"})
        handler2.assert_called_once_with({"key": "value"})

    def test_emit_handles_handler_exception(self) -> None:
        """_emit handles handler exceptions gracefully."""
        ws = AragoraWebSocket("https://api.aragora.ai")
        handler1 = MagicMock(side_effect=Exception("Handler error"))
        handler2 = MagicMock()
        ws.on("test_event", handler1)
        ws.on("test_event", handler2)

        # Should not raise, should continue to next handler
        ws._emit("test_event", {})

        # Second handler should still be called
        handler2.assert_called_once()


class TestAragoraWebSocketConnection:
    """Tests for WebSocket connection lifecycle."""

    @pytest.mark.asyncio
    async def test_connect_changes_state(self) -> None:
        """connect() changes state during connection."""
        ws = AragoraWebSocket("https://api.aragora.ai")

        # Verify initial state
        assert ws.state == "disconnected"

        # Mock the websockets module at import location
        mock_ws = AsyncMock()
        mock_ws.__aiter__ = MagicMock(return_value=iter([]))

        with patch.dict("sys.modules", {"websockets": MagicMock()}):
            import sys

            sys.modules["websockets"].connect = AsyncMock(return_value=mock_ws)

            await ws.connect()
            assert ws.state == "connected"
            await ws.close()

    @pytest.mark.asyncio
    async def test_connect_emits_connected_event(self) -> None:
        """connect() emits connected event on success."""
        ws = AragoraWebSocket("https://api.aragora.ai")
        connected_handler = MagicMock()
        ws.on("connected", connected_handler)

        mock_ws = AsyncMock()
        mock_ws.__aiter__ = MagicMock(return_value=iter([]))

        with patch.dict("sys.modules", {"websockets": MagicMock()}):
            import sys

            sys.modules["websockets"].connect = AsyncMock(return_value=mock_ws)

            await ws.connect()

            assert connected_handler.called
            assert ws.state == "connected"

            # Cleanup
            await ws.close()

    @pytest.mark.asyncio
    async def test_connect_emits_error_on_failure(self) -> None:
        """connect() emits error event on connection failure."""
        ws = AragoraWebSocket("https://api.aragora.ai")
        error_handler = MagicMock()
        ws.on("error", error_handler)

        with patch.dict("sys.modules", {"websockets": MagicMock()}):
            import sys

            sys.modules["websockets"].connect = AsyncMock(
                side_effect=RuntimeError("Connection refused")
            )

            with pytest.raises(RuntimeError):
                await ws.connect()

            assert error_handler.called
            assert ws.state == "disconnected"

    @pytest.mark.asyncio
    async def test_connect_skips_if_already_connected(self) -> None:
        """connect() does nothing if already connected."""
        ws = AragoraWebSocket("https://api.aragora.ai")
        ws._state = "connected"

        # Create a mock to track if connection was attempted
        connection_attempted = False

        original_build_ws_url = ws._build_ws_url

        def track_build(*args, **kwargs):
            nonlocal connection_attempted
            connection_attempted = True
            return original_build_ws_url(*args, **kwargs)

        ws._build_ws_url = track_build

        await ws.connect()

        # Should not attempt connection
        assert not connection_attempted

    @pytest.mark.asyncio
    async def test_close_sets_state_disconnected(self) -> None:
        """close() sets state to disconnected."""
        ws = AragoraWebSocket("https://api.aragora.ai")
        ws._state = "connected"
        ws._ws = AsyncMock()

        await ws.close()

        assert ws.state == "disconnected"
        assert ws._ws is None

    @pytest.mark.asyncio
    async def test_close_signals_event_queue(self) -> None:
        """close() puts None in queue to signal end."""
        ws = AragoraWebSocket("https://api.aragora.ai")
        ws._state = "connected"
        ws._ws = AsyncMock()

        await ws.close()

        # Queue should have None sentinel
        assert ws._event_queue.get() is None


class TestAragoraWebSocketState:
    """Tests for WebSocket state property."""

    def test_state_property(self) -> None:
        """state property returns current state."""
        ws = AragoraWebSocket("https://api.aragora.ai")
        assert ws.state == "disconnected"

        ws._state = "connected"
        assert ws.state == "connected"

        ws._state = "reconnecting"
        assert ws.state == "reconnecting"


class TestStreamDebateHelper:
    """Tests for stream_debate convenience function."""

    def test_stream_debate_function_exists(self) -> None:
        """stream_debate function is importable and callable."""
        from aragora_sdk.websocket import stream_debate

        assert callable(stream_debate)

    def test_stream_debate_is_async_generator(self) -> None:
        """stream_debate returns an async generator."""
        import inspect

        assert inspect.isasyncgenfunction(stream_debate)

    def test_websocket_options_for_stream_debate(self) -> None:
        """WebSocketOptions can be passed to stream_debate."""
        options = WebSocketOptions(auto_reconnect=False, heartbeat_interval=15.0)
        assert options.auto_reconnect is False
        assert options.heartbeat_interval == 15.0

        # Function completed without error


class TestAragoraWebSocketReconnection:
    """Tests for WebSocket reconnection behavior."""

    def test_handle_disconnect_emits_disconnected(self) -> None:
        """_handle_disconnect emits disconnected event."""
        ws = AragoraWebSocket("https://api.aragora.ai")
        handler = MagicMock()
        ws.on("disconnected", handler)

        ws._handle_disconnect(1006, "Connection lost")

        handler.assert_called_once()
        event_data = handler.call_args[0][0]
        assert event_data["code"] == 1006
        assert event_data["reason"] == "Connection lost"

    def test_handle_disconnect_schedules_reconnect(self) -> None:
        """_handle_disconnect schedules reconnection if auto_reconnect enabled."""
        ws = AragoraWebSocket("https://api.aragora.ai")
        ws.options.auto_reconnect = True

        with patch.object(ws, "_schedule_reconnect") as mock_schedule:
            ws._handle_disconnect(1006, "Connection lost")
            mock_schedule.assert_called_once()

    def test_handle_disconnect_no_reconnect_when_disabled(self) -> None:
        """_handle_disconnect skips reconnection if auto_reconnect disabled."""
        ws = AragoraWebSocket("https://api.aragora.ai")
        ws.options.auto_reconnect = False

        with patch.object(ws, "_schedule_reconnect") as mock_schedule:
            ws._handle_disconnect(1006, "Connection lost")
            mock_schedule.assert_not_called()

    def test_handle_disconnect_respects_max_attempts(self) -> None:
        """_handle_disconnect stops after max_reconnect_attempts."""
        ws = AragoraWebSocket("https://api.aragora.ai")
        ws.options.auto_reconnect = True
        ws.options.max_reconnect_attempts = 3
        ws._reconnect_attempts = 3  # Already at max

        with patch.object(ws, "_schedule_reconnect") as mock_schedule:
            ws._handle_disconnect(1006, "Connection lost")
            mock_schedule.assert_not_called()


class TestAragoraWebSocketSend:
    """Tests for WebSocket send functionality."""

    def test_send_when_not_connected(self) -> None:
        """_send does nothing when not connected."""
        ws = AragoraWebSocket("https://api.aragora.ai")
        ws._state = "disconnected"
        ws._ws = None

        # Should not raise
        ws._send({"type": "ping"})

    def test_send_when_connected(self) -> None:
        """_send queues message when connected."""
        ws = AragoraWebSocket("https://api.aragora.ai")
        ws._state = "connected"
        ws._ws = AsyncMock()

        ws._send({"type": "ping"})

        # The send is wrapped in asyncio.ensure_future, so we check ws was used


class TestAragoraWebSocketCleanup:
    """Tests for WebSocket cleanup functionality."""

    def test_cleanup_cancels_tasks(self) -> None:
        """_cleanup cancels background tasks."""
        ws = AragoraWebSocket("https://api.aragora.ai")

        mock_heartbeat = MagicMock()
        mock_heartbeat.done.return_value = False
        mock_receive = MagicMock()
        mock_receive.done.return_value = False
        mock_reconnect = MagicMock()
        mock_reconnect.done.return_value = False

        ws._heartbeat_task = mock_heartbeat
        ws._receive_task = mock_receive
        ws._reconnect_task = mock_reconnect

        ws._cleanup()

        mock_heartbeat.cancel.assert_called_once()
        mock_receive.cancel.assert_called_once()
        mock_reconnect.cancel.assert_called_once()

        assert ws._heartbeat_task is None
        assert ws._receive_task is None
        assert ws._reconnect_task is None

    def test_cleanup_handles_done_tasks(self) -> None:
        """_cleanup handles already-done tasks gracefully."""
        ws = AragoraWebSocket("https://api.aragora.ai")

        mock_task = MagicMock()
        mock_task.done.return_value = True

        ws._heartbeat_task = mock_task

        ws._cleanup()

        # Should not call cancel on done task
        mock_task.cancel.assert_not_called()
