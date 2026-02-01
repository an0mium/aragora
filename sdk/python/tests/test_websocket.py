"""Tests for WebSocket module."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.websocket import (
    _STATE_CONNECTED,
    _STATE_DISCONNECTED,
    EVENT_TYPES,
    AragoraWebSocket,
    WebSocketEvent,
    WebSocketOptions,
    stream_debate,
)


class TestWebSocketOptions:
    """Tests for WebSocketOptions dataclass."""

    def test_default_values(self) -> None:
        """WebSocketOptions has sensible defaults."""
        options = WebSocketOptions()

        assert options.auto_reconnect is True
        assert options.max_reconnect_attempts == 5
        assert options.reconnect_delay == 1.0
        assert options.heartbeat_interval == 30.0

    def test_custom_values(self) -> None:
        """WebSocketOptions accepts custom values."""
        options = WebSocketOptions(
            auto_reconnect=False,
            max_reconnect_attempts=10,
            reconnect_delay=2.5,
            heartbeat_interval=60.0,
        )

        assert options.auto_reconnect is False
        assert options.max_reconnect_attempts == 10
        assert options.reconnect_delay == 2.5
        assert options.heartbeat_interval == 60.0

    def test_partial_custom_values(self) -> None:
        """WebSocketOptions allows partial customization."""
        options = WebSocketOptions(max_reconnect_attempts=3)

        assert options.auto_reconnect is True
        assert options.max_reconnect_attempts == 3
        assert options.reconnect_delay == 1.0
        assert options.heartbeat_interval == 30.0


class TestWebSocketEvent:
    """Tests for WebSocketEvent dataclass."""

    def test_minimal_event(self) -> None:
        """WebSocketEvent can be created with just type."""
        event = WebSocketEvent(type="debate_start")

        assert event.type == "debate_start"
        assert event.data == {}
        assert event.timestamp == ""
        assert event.debate_id is None
        assert event.typed_data is None

    def test_full_event(self) -> None:
        """WebSocketEvent accepts all fields."""
        data = {"agent": "claude", "content": "Hello"}
        event = WebSocketEvent(
            type="agent_message",
            data=data,
            timestamp="2024-01-15T10:30:00Z",
            debate_id="deb_123",
            typed_data=MagicMock(),
        )

        assert event.type == "agent_message"
        assert event.data == data
        assert event.timestamp == "2024-01-15T10:30:00Z"
        assert event.debate_id == "deb_123"
        assert event.typed_data is not None

    def test_event_with_empty_data(self) -> None:
        """WebSocketEvent handles empty data dict."""
        event = WebSocketEvent(type="connected", data={})

        assert event.type == "connected"
        assert event.data == {}


class TestBuildWsUrl:
    """Tests for AragoraWebSocket._build_ws_url()."""

    def test_http_to_ws_conversion(self) -> None:
        """HTTP URL is converted to WS URL."""
        ws = AragoraWebSocket("http://localhost:8080")
        url = ws._build_ws_url()

        assert url == "ws://localhost:8080/ws"

    def test_https_to_wss_conversion(self) -> None:
        """HTTPS URL is converted to WSS URL."""
        ws = AragoraWebSocket("https://api.aragora.ai")
        url = ws._build_ws_url()

        assert url == "wss://api.aragora.ai/ws"

    def test_trailing_slash_handling(self) -> None:
        """Trailing slashes are handled correctly."""
        ws = AragoraWebSocket("http://localhost:8080/")
        url = ws._build_ws_url()

        assert url == "ws://localhost:8080/ws"

    def test_with_debate_id(self) -> None:
        """Debate ID is added as query parameter."""
        ws = AragoraWebSocket("http://localhost:8080")
        url = ws._build_ws_url(debate_id="deb_123")

        assert url == "ws://localhost:8080/ws?debate_id=deb_123"

    def test_with_api_key(self) -> None:
        """API key is added as token query parameter."""
        ws = AragoraWebSocket("http://localhost:8080", api_key="secret-key")
        url = ws._build_ws_url()

        assert url == "ws://localhost:8080/ws?token=secret-key"

    def test_with_debate_id_and_api_key(self) -> None:
        """Both debate_id and api_key are added as query params."""
        ws = AragoraWebSocket("http://localhost:8080", api_key="secret-key")
        url = ws._build_ws_url(debate_id="deb_123")

        assert "debate_id=deb_123" in url
        assert "token=secret-key" in url
        assert url.startswith("ws://localhost:8080/ws?")

    def test_special_characters_in_debate_id(self) -> None:
        """Special characters in debate_id are URL encoded."""
        ws = AragoraWebSocket("http://localhost:8080")
        url = ws._build_ws_url(debate_id="debate with spaces")

        assert "debate%20with%20spaces" in url

    def test_explicit_ws_url_overrides_base_url(self) -> None:
        """Explicit ws_url takes precedence over base_url."""
        ws = AragoraWebSocket("http://localhost:8080", ws_url="wss://custom.example.com/websocket")
        url = ws._build_ws_url()

        assert url == "wss://custom.example.com/websocket/ws"

    def test_explicit_ws_url_with_ws_path(self) -> None:
        """Explicit ws_url ending with /ws is not duplicated."""
        ws = AragoraWebSocket("http://localhost:8080", ws_url="wss://custom.example.com/ws")
        url = ws._build_ws_url()

        assert url == "wss://custom.example.com/ws"


class TestHandlerRegistration:
    """Tests for on() and off() handler registration."""

    def test_on_registers_handler(self) -> None:
        """on() registers a handler for an event type."""
        ws = AragoraWebSocket("http://localhost:8080")
        handler = MagicMock()

        ws.on("agent_message", handler)

        assert handler in ws._handlers["agent_message"]

    def test_on_returns_unsubscribe_function(self) -> None:
        """on() returns a function that removes the handler."""
        ws = AragoraWebSocket("http://localhost:8080")
        handler = MagicMock()

        unsubscribe = ws.on("agent_message", handler)
        assert handler in ws._handlers["agent_message"]

        unsubscribe()
        assert handler not in ws._handlers["agent_message"]

    def test_off_removes_handler(self) -> None:
        """off() removes a registered handler."""
        ws = AragoraWebSocket("http://localhost:8080")
        handler = MagicMock()

        ws.on("debate_start", handler)
        ws.off("debate_start", handler)

        assert handler not in ws._handlers["debate_start"]

    def test_off_handles_nonexistent_handler(self) -> None:
        """off() silently handles removing a non-existent handler."""
        ws = AragoraWebSocket("http://localhost:8080")
        handler = MagicMock()

        # Should not raise
        ws.off("debate_start", handler)

    def test_off_handles_nonexistent_event_type(self) -> None:
        """off() silently handles non-existent event types."""
        ws = AragoraWebSocket("http://localhost:8080")
        handler = MagicMock()

        # Should not raise
        ws.off("nonexistent_event", handler)

    def test_multiple_handlers_same_event(self) -> None:
        """Multiple handlers can be registered for same event."""
        ws = AragoraWebSocket("http://localhost:8080")
        handler1 = MagicMock()
        handler2 = MagicMock()

        ws.on("agent_message", handler1)
        ws.on("agent_message", handler2)

        assert handler1 in ws._handlers["agent_message"]
        assert handler2 in ws._handlers["agent_message"]

    def test_on_creates_handler_list_for_custom_event(self) -> None:
        """on() creates handler list for custom event types."""
        ws = AragoraWebSocket("http://localhost:8080")
        handler = MagicMock()

        ws.on("custom_event", handler)

        assert "custom_event" in ws._handlers
        assert handler in ws._handlers["custom_event"]


class TestHandleMessage:
    """Tests for _handle_message() parsing and event creation."""

    def test_parse_valid_json_message(self) -> None:
        """Valid JSON message is parsed correctly."""
        ws = AragoraWebSocket("http://localhost:8080")
        handler = MagicMock()
        ws.on("message", handler)

        message = json.dumps(
            {
                "type": "agent_message",
                "data": {"agent": "claude", "content": "Hello"},
                "timestamp": "2024-01-15T10:30:00Z",
                "debate_id": "deb_123",
            }
        )
        ws._handle_message(message)

        handler.assert_called_once()
        event = handler.call_args[0][0]
        assert isinstance(event, WebSocketEvent)
        assert event.type == "agent_message"
        assert event.data == {"agent": "claude", "content": "Hello"}
        assert event.timestamp == "2024-01-15T10:30:00Z"
        assert event.debate_id == "deb_123"

    def test_parse_invalid_json_emits_error(self) -> None:
        """Invalid JSON emits an error event."""
        ws = AragoraWebSocket("http://localhost:8080")
        error_handler = MagicMock()
        ws.on("error", error_handler)

        ws._handle_message("not valid json")

        error_handler.assert_called_once()
        error_data = error_handler.call_args[0][0]
        assert "Failed to parse message" in error_data["error"]

    def test_missing_type_defaults_to_message(self) -> None:
        """Message without type defaults to 'message' type."""
        ws = AragoraWebSocket("http://localhost:8080")
        handler = MagicMock()
        ws.on("message", handler)

        ws._handle_message(json.dumps({"data": {"content": "test"}}))

        event = handler.call_args[0][0]
        assert event.type == "message"

    def test_typed_data_parsing(self) -> None:
        """Typed data is parsed when event class exists."""
        ws = AragoraWebSocket("http://localhost:8080")
        handler = MagicMock()
        ws.on("debate_start", handler)

        message = json.dumps(
            {
                "type": "debate_start",
                "data": {
                    "debate_id": "deb_123",
                    "task": "Test task",
                    "agents": ["claude", "gpt-4"],
                    "total_rounds": 3,
                },
            }
        )
        ws._handle_message(message)

        event = handler.call_args[0][0]
        assert event.typed_data is not None
        assert event.typed_data.debate_id == "deb_123"
        assert event.typed_data.task == "Test task"
        assert event.typed_data.agents == ["claude", "gpt-4"]

    def test_event_added_to_queue(self) -> None:
        """Events are added to the event queue for async iterator."""
        ws = AragoraWebSocket("http://localhost:8080")

        ws._handle_message(json.dumps({"type": "debate_start", "data": {}}))

        assert not ws._event_queue.empty()
        event = ws._event_queue.get_nowait()
        assert event.type == "debate_start"

    def test_specific_event_handler_called(self) -> None:
        """Specific event type handlers are called."""
        ws = AragoraWebSocket("http://localhost:8080")
        debate_start_handler = MagicMock()
        ws.on("debate_start", debate_start_handler)

        ws._handle_message(json.dumps({"type": "debate_start", "data": {}}))

        debate_start_handler.assert_called_once()


class TestSubscriptions:
    """Tests for subscribe() and unsubscribe()."""

    def test_subscribe_adds_to_subscriptions(self) -> None:
        """subscribe() adds debate_id to subscriptions set."""
        ws = AragoraWebSocket("http://localhost:8080")

        ws.subscribe("deb_123")

        assert "deb_123" in ws._subscriptions

    def test_unsubscribe_removes_from_subscriptions(self) -> None:
        """unsubscribe() removes debate_id from subscriptions set."""
        ws = AragoraWebSocket("http://localhost:8080")
        ws._subscriptions.add("deb_123")

        ws.unsubscribe("deb_123")

        assert "deb_123" not in ws._subscriptions

    def test_unsubscribe_handles_nonexistent_id(self) -> None:
        """unsubscribe() handles non-existent debate_id gracefully."""
        ws = AragoraWebSocket("http://localhost:8080")

        # Should not raise
        ws.unsubscribe("nonexistent")


class TestConnectionState:
    """Tests for connection state management."""

    def test_initial_state_is_disconnected(self) -> None:
        """Initial state is disconnected."""
        ws = AragoraWebSocket("http://localhost:8080")

        assert ws.state == _STATE_DISCONNECTED

    def test_state_property(self) -> None:
        """state property returns current state."""
        ws = AragoraWebSocket("http://localhost:8080")

        assert ws.state == ws._state

    @pytest.mark.asyncio
    async def test_connect_changes_state(self) -> None:
        """connect() changes state to connected."""
        ws = AragoraWebSocket("http://localhost:8080")

        mock_connection = AsyncMock()
        mock_connection.__aiter__ = MagicMock(
            return_value=AsyncMock(__anext__=AsyncMock(side_effect=StopAsyncIteration))
        )

        mock_websockets = MagicMock()
        mock_websockets.connect = AsyncMock(return_value=mock_connection)

        with patch.dict("sys.modules", {"websockets": mock_websockets}):
            await ws.connect()

            assert ws.state == _STATE_CONNECTED

    @pytest.mark.asyncio
    async def test_connect_emits_connected_event(self) -> None:
        """connect() emits connected event on success."""
        ws = AragoraWebSocket("http://localhost:8080")
        connected_handler = MagicMock()
        ws.on("connected", connected_handler)

        mock_connection = AsyncMock()
        mock_connection.__aiter__ = MagicMock(
            return_value=AsyncMock(__anext__=AsyncMock(side_effect=StopAsyncIteration))
        )

        mock_websockets = MagicMock()
        mock_websockets.connect = AsyncMock(return_value=mock_connection)

        with patch.dict("sys.modules", {"websockets": mock_websockets}):
            await ws.connect()

            connected_handler.assert_called_once_with({})

    @pytest.mark.asyncio
    async def test_connect_emits_error_on_failure(self) -> None:
        """connect() emits error event on connection failure."""
        ws = AragoraWebSocket("http://localhost:8080")
        error_handler = MagicMock()
        ws.on("error", error_handler)

        mock_websockets = MagicMock()
        mock_websockets.connect = AsyncMock(side_effect=ConnectionError("Failed"))

        with patch.dict("sys.modules", {"websockets": mock_websockets}):
            with pytest.raises(ConnectionError):
                await ws.connect()

            error_handler.assert_called_once()
            assert "Failed" in error_handler.call_args[0][0]["error"]

    @pytest.mark.asyncio
    async def test_connect_skips_if_already_connected(self) -> None:
        """connect() returns early if already connected."""
        ws = AragoraWebSocket("http://localhost:8080")
        ws._state = _STATE_CONNECTED

        mock_websockets = MagicMock()
        mock_websockets.connect = AsyncMock()

        with patch.dict("sys.modules", {"websockets": mock_websockets}):
            await ws.connect()

            mock_websockets.connect.assert_not_called()

    @pytest.mark.asyncio
    async def test_close_changes_state_to_disconnected(self) -> None:
        """close() changes state to disconnected."""
        ws = AragoraWebSocket("http://localhost:8080")
        ws._state = _STATE_CONNECTED
        ws._ws = AsyncMock()

        await ws.close()

        assert ws.state == _STATE_DISCONNECTED

    @pytest.mark.asyncio
    async def test_close_signals_event_queue(self) -> None:
        """close() puts None in event queue to signal end."""
        ws = AragoraWebSocket("http://localhost:8080")
        ws._state = _STATE_CONNECTED
        ws._ws = AsyncMock()

        await ws.close()

        assert ws._event_queue.get_nowait() is None


class TestEmit:
    """Tests for _emit() handler invocation."""

    def test_emit_calls_all_handlers(self) -> None:
        """_emit() calls all registered handlers for event."""
        ws = AragoraWebSocket("http://localhost:8080")
        handler1 = MagicMock()
        handler2 = MagicMock()
        ws.on("test_event", handler1)
        ws.on("test_event", handler2)

        ws._emit("test_event", {"key": "value"})

        handler1.assert_called_once_with({"key": "value"})
        handler2.assert_called_once_with({"key": "value"})

    def test_emit_handles_handler_exception(self) -> None:
        """_emit() catches exceptions in handlers."""
        ws = AragoraWebSocket("http://localhost:8080")
        bad_handler = MagicMock(side_effect=ValueError("Handler error"))
        good_handler = MagicMock()
        ws.on("test_event", bad_handler)
        ws.on("test_event", good_handler)

        # Should not raise
        ws._emit("test_event", {})

        # Good handler should still be called
        good_handler.assert_called_once()


class TestEventTypes:
    """Tests for EVENT_TYPES constant."""

    def test_event_types_contains_expected_events(self) -> None:
        """EVENT_TYPES contains all expected event types."""
        expected = [
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
        ]

        for event_type in expected:
            assert event_type in EVENT_TYPES


class TestStreamDebate:
    """Tests for stream_debate convenience function."""

    @pytest.mark.asyncio
    async def test_stream_debate_creates_websocket(self) -> None:
        """stream_debate creates and connects WebSocket."""
        with patch.object(AragoraWebSocket, "connect", new_callable=AsyncMock) as mock_connect:
            with patch.object(AragoraWebSocket, "close", new_callable=AsyncMock):
                with patch.object(AragoraWebSocket, "events") as mock_events:
                    # Return an async generator that yields one event then stops
                    async def mock_event_gen():
                        yield WebSocketEvent(type="debate_end", data={})

                    mock_events.return_value = mock_event_gen()

                    events = []
                    async for event in stream_debate(
                        "http://localhost:8080", debate_id="deb_123", api_key="key"
                    ):
                        events.append(event)

                    mock_connect.assert_called_once_with(debate_id="deb_123")
                    assert len(events) == 1
                    assert events[0].type == "debate_end"

    @pytest.mark.asyncio
    async def test_stream_debate_closes_on_error(self) -> None:
        """stream_debate closes WebSocket on error event."""
        with patch.object(AragoraWebSocket, "connect", new_callable=AsyncMock):
            with patch.object(AragoraWebSocket, "close", new_callable=AsyncMock) as mock_close:
                with patch.object(AragoraWebSocket, "events") as mock_events:

                    async def mock_event_gen():
                        yield WebSocketEvent(type="error", data={"error": "Test error"})

                    mock_events.return_value = mock_event_gen()

                    async for _ in stream_debate("http://localhost:8080"):
                        pass

                    mock_close.assert_called_once()

    @pytest.mark.asyncio
    async def test_stream_debate_uses_custom_options(self) -> None:
        """stream_debate passes custom options to WebSocket."""
        options = WebSocketOptions(max_reconnect_attempts=10)

        with patch.object(AragoraWebSocket, "connect", new_callable=AsyncMock):
            with patch.object(AragoraWebSocket, "close", new_callable=AsyncMock):
                with patch.object(AragoraWebSocket, "events") as mock_events:

                    async def mock_event_gen():
                        yield WebSocketEvent(type="debate_end", data={})

                    mock_events.return_value = mock_event_gen()

                    # We need to capture the WebSocket instance
                    async for _ in stream_debate("http://localhost:8080", options=options):
                        pass

                    # The function creates WebSocket internally, hard to test options directly
                    # but we verify the function runs without error
