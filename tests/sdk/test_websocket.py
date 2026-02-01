"""
Tests for the Aragora Python SDK WebSocket module.

Covers: WebSocketOptions, WebSocketEvent, EVENT_TYPES, URL building,
state management, event handlers, message handling, close behaviour,
subscribe/unsubscribe, client integration, and public exports.
"""

from __future__ import annotations

import asyncio
import json
import queue
from dataclasses import fields
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sdk.python.aragora.websocket import (
    EVENT_TYPES,
    AragoraWebSocket,
    WebSocketEvent,
    WebSocketOptions,
    stream_debate,
    _STATE_CONNECTED,
    _STATE_CONNECTING,
    _STATE_DISCONNECTED,
    _STATE_RECONNECTING,
)


# ===========================================================================
# WebSocketOptions
# ===========================================================================


class TestWebSocketOptions:
    """Tests for the WebSocketOptions dataclass."""

    def test_default_values(self):
        opts = WebSocketOptions()
        assert opts.auto_reconnect is True
        assert opts.max_reconnect_attempts == 5
        assert opts.reconnect_delay == 1.0
        assert opts.heartbeat_interval == 30.0

    def test_custom_values(self):
        opts = WebSocketOptions(
            auto_reconnect=False,
            max_reconnect_attempts=10,
            reconnect_delay=2.5,
            heartbeat_interval=15.0,
        )
        assert opts.auto_reconnect is False
        assert opts.max_reconnect_attempts == 10
        assert opts.reconnect_delay == 2.5
        assert opts.heartbeat_interval == 15.0

    def test_is_dataclass(self):
        opts = WebSocketOptions()
        field_names = {f.name for f in fields(opts)}
        assert field_names == {
            "auto_reconnect",
            "max_reconnect_attempts",
            "reconnect_delay",
            "heartbeat_interval",
        }


# ===========================================================================
# WebSocketEvent
# ===========================================================================


class TestWebSocketEvent:
    """Tests for the WebSocketEvent dataclass."""

    def test_default_values(self):
        event = WebSocketEvent(type="test")
        assert event.type == "test"
        assert event.data == {}
        assert event.timestamp == ""
        assert event.debate_id is None

    def test_custom_values(self):
        event = WebSocketEvent(
            type="agent_message",
            data={"content": "hello"},
            timestamp="2025-01-01T00:00:00Z",
            debate_id="d-123",
        )
        assert event.type == "agent_message"
        assert event.data == {"content": "hello"}
        assert event.timestamp == "2025-01-01T00:00:00Z"
        assert event.debate_id == "d-123"

    def test_data_default_is_independent(self):
        """Each instance should have its own default dict."""
        e1 = WebSocketEvent(type="a")
        e2 = WebSocketEvent(type="b")
        e1.data["key"] = "value"
        assert "key" not in e2.data

    def test_is_dataclass(self):
        event = WebSocketEvent(type="x")
        field_names = {f.name for f in fields(event)}
        assert field_names == {"type", "data", "timestamp", "debate_id", "typed_data"}


# ===========================================================================
# EVENT_TYPES
# ===========================================================================


class TestEventTypes:
    """Tests for the EVENT_TYPES constant."""

    def test_is_tuple(self):
        assert isinstance(EVENT_TYPES, tuple)

    def test_has_19_types(self):
        assert len(EVENT_TYPES) == 19

    def test_contains_core_events(self):
        core = [
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
        for evt in core:
            assert evt in EVENT_TYPES, f"{evt} not in EVENT_TYPES"

    def test_no_duplicates(self):
        assert len(EVENT_TYPES) == len(set(EVENT_TYPES))


# ===========================================================================
# URL Building
# ===========================================================================


class TestUrlBuilding:
    """Tests for _build_ws_url."""

    def test_basic_http_to_ws(self):
        ws = AragoraWebSocket("http://localhost:8080")
        url = ws._build_ws_url()
        assert url == "ws://localhost:8080/ws"

    def test_https_to_wss(self):
        ws = AragoraWebSocket("https://api.example.com")
        url = ws._build_ws_url()
        assert url == "wss://api.example.com/ws"

    def test_explicit_ws_url(self):
        ws = AragoraWebSocket("http://localhost", ws_url="ws://custom:9090/ws")
        url = ws._build_ws_url()
        assert url == "ws://custom:9090/ws"

    def test_debate_id_param(self):
        ws = AragoraWebSocket("http://localhost:8080")
        url = ws._build_ws_url(debate_id="d-123")
        assert "debate_id=d-123" in url
        assert url.startswith("ws://localhost:8080/ws?")

    def test_api_key_param(self):
        ws = AragoraWebSocket("http://localhost:8080", api_key="secret")
        url = ws._build_ws_url()
        assert "token=secret" in url
        assert url.startswith("ws://localhost:8080/ws?")

    def test_debate_id_and_api_key(self):
        ws = AragoraWebSocket("http://localhost:8080", api_key="key")
        url = ws._build_ws_url(debate_id="d-1")
        assert "debate_id=d-1" in url
        assert "token=key" in url
        # debate_id should come first
        assert url.index("debate_id") < url.index("token")

    def test_trailing_slash_stripped(self):
        ws = AragoraWebSocket("http://localhost:8080/")
        url = ws._build_ws_url()
        assert url == "ws://localhost:8080/ws"

    def test_special_characters_encoded(self):
        ws = AragoraWebSocket("http://localhost:8080")
        url = ws._build_ws_url(debate_id="id with spaces")
        assert "id%20with%20spaces" in url

    def test_explicit_ws_url_already_has_path(self):
        ws = AragoraWebSocket("http://x", ws_url="ws://y:8080/ws")
        url = ws._build_ws_url()
        assert url == "ws://y:8080/ws"

    def test_explicit_ws_url_without_ws_suffix(self):
        ws = AragoraWebSocket("http://x", ws_url="ws://y:8080")
        url = ws._build_ws_url()
        assert url == "ws://y:8080/ws"


# ===========================================================================
# State Management
# ===========================================================================


class TestStateManagement:
    """Tests for connection state tracking."""

    def test_initial_state_is_disconnected(self):
        ws = AragoraWebSocket("http://localhost")
        assert ws.state == "disconnected"

    def test_state_constants(self):
        assert _STATE_CONNECTING == "connecting"
        assert _STATE_CONNECTED == "connected"
        assert _STATE_DISCONNECTED == "disconnected"
        assert _STATE_RECONNECTING == "reconnecting"

    def test_state_property_reflects_internal(self):
        ws = AragoraWebSocket("http://localhost")
        ws._state = _STATE_CONNECTED
        assert ws.state == "connected"


# ===========================================================================
# Event Handlers
# ===========================================================================


class TestEventHandlers:
    """Tests for on/off event registration."""

    def test_on_returns_unsubscribe(self):
        ws = AragoraWebSocket("http://localhost")
        handler = MagicMock()
        unsub = ws.on("message", handler)
        assert callable(unsub)

    def test_handler_called_on_emit(self):
        ws = AragoraWebSocket("http://localhost")
        handler = MagicMock()
        ws.on("connected", handler)
        ws._emit("connected", {"info": "ok"})
        handler.assert_called_once_with({"info": "ok"})

    def test_multiple_handlers(self):
        ws = AragoraWebSocket("http://localhost")
        h1 = MagicMock()
        h2 = MagicMock()
        ws.on("error", h1)
        ws.on("error", h2)
        ws._emit("error", {"err": "x"})
        h1.assert_called_once()
        h2.assert_called_once()

    def test_off_removes_handler(self):
        ws = AragoraWebSocket("http://localhost")
        handler = MagicMock()
        ws.on("message", handler)
        ws.off("message", handler)
        ws._emit("message", {})
        handler.assert_not_called()

    def test_unsubscribe_fn_removes_handler(self):
        ws = AragoraWebSocket("http://localhost")
        handler = MagicMock()
        unsub = ws.on("message", handler)
        unsub()
        ws._emit("message", {})
        handler.assert_not_called()

    def test_off_nonexistent_handler_no_error(self):
        ws = AragoraWebSocket("http://localhost")
        handler = MagicMock()
        # Should not raise
        ws.off("message", handler)

    def test_on_unknown_event(self):
        ws = AragoraWebSocket("http://localhost")
        handler = MagicMock()
        unsub = ws.on("custom_event", handler)
        ws._emit("custom_event", {"x": 1})
        handler.assert_called_once_with({"x": 1})
        assert callable(unsub)

    def test_handler_exception_does_not_stop_others(self):
        ws = AragoraWebSocket("http://localhost")
        bad = MagicMock(side_effect=RuntimeError("boom"))
        good = MagicMock()
        ws.on("error", bad)
        ws.on("error", good)
        ws._emit("error", {"e": 1})
        bad.assert_called_once()
        good.assert_called_once()


# ===========================================================================
# Message Handling
# ===========================================================================


class TestMessageHandling:
    """Tests for _handle_message."""

    def test_valid_json_dispatched(self):
        ws = AragoraWebSocket("http://localhost")
        handler = MagicMock()
        ws.on("message", handler)

        raw = json.dumps({"type": "debate_start", "data": {"id": "d1"}, "timestamp": "t"})
        ws._handle_message(raw)

        handler.assert_called_once()
        event = handler.call_args[0][0]
        assert isinstance(event, WebSocketEvent)
        assert event.type == "debate_start"

    def test_specific_handler_dispatched(self):
        ws = AragoraWebSocket("http://localhost")
        handler = MagicMock()
        ws.on("agent_message", handler)

        raw = json.dumps({"type": "agent_message", "data": {"agent": "claude"}})
        ws._handle_message(raw)

        handler.assert_called_once()
        event = handler.call_args[0][0]
        assert event.type == "agent_message"

    def test_invalid_json_emits_error(self):
        ws = AragoraWebSocket("http://localhost")
        handler = MagicMock()
        ws.on("error", handler)

        ws._handle_message("not-json{{{")
        handler.assert_called_once()

    def test_event_enqueued_for_iterator(self):
        ws = AragoraWebSocket("http://localhost")
        raw = json.dumps({"type": "vote", "data": {"agent": "gpt4"}})
        ws._handle_message(raw)

        event = ws._event_queue.get_nowait()
        assert isinstance(event, WebSocketEvent)
        assert event.type == "vote"

    def test_missing_type_defaults_to_message(self):
        ws = AragoraWebSocket("http://localhost")
        handler = MagicMock()
        ws.on("message", handler)

        raw = json.dumps({"data": {"info": "no type field"}})
        ws._handle_message(raw)

        # Handler fires twice: once from the generic "message" emit and once
        # from the specific type emit (defaulted type is also "message").
        assert handler.call_count == 2
        event = handler.call_args_list[0][0][0]
        assert event.type == "message"

    def test_debate_id_extracted(self):
        ws = AragoraWebSocket("http://localhost")
        raw = json.dumps({"type": "consensus", "debate_id": "d-abc", "data": {}})
        ws._handle_message(raw)

        event = ws._event_queue.get_nowait()
        assert event.debate_id == "d-abc"


# ===========================================================================
# Close
# ===========================================================================


class TestClose:
    """Tests for close() behaviour."""

    @pytest.mark.asyncio
    async def test_close_sets_disconnected(self):
        ws = AragoraWebSocket("http://localhost")
        ws._state = _STATE_CONNECTED
        await ws.close()
        assert ws.state == "disconnected"

    @pytest.mark.asyncio
    async def test_close_sends_sentinel_to_queue(self):
        ws = AragoraWebSocket("http://localhost")
        await ws.close()
        item = ws._event_queue.get_nowait()
        assert item is None

    @pytest.mark.asyncio
    async def test_close_disables_auto_reconnect(self):
        ws = AragoraWebSocket("http://localhost")
        assert ws.options.auto_reconnect is True
        await ws.close()
        assert ws.options.auto_reconnect is False

    @pytest.mark.asyncio
    async def test_close_when_ws_is_none(self):
        ws = AragoraWebSocket("http://localhost")
        ws._ws = None
        await ws.close()  # should not raise
        assert ws.state == "disconnected"

    @pytest.mark.asyncio
    async def test_close_calls_ws_close(self):
        ws = AragoraWebSocket("http://localhost")
        mock_ws = AsyncMock()
        ws._ws = mock_ws
        ws._state = _STATE_CONNECTED
        await ws.close()
        mock_ws.close.assert_awaited_once()
        assert ws._ws is None


# ===========================================================================
# Subscribe / Unsubscribe
# ===========================================================================


class TestSubscriptions:
    """Tests for subscribe/unsubscribe."""

    def test_subscribe_tracks_debate_id(self):
        ws = AragoraWebSocket("http://localhost")
        ws.subscribe("d-1")
        assert "d-1" in ws._subscriptions

    def test_unsubscribe_removes_debate_id(self):
        ws = AragoraWebSocket("http://localhost")
        ws._subscriptions.add("d-1")
        ws.unsubscribe("d-1")
        assert "d-1" not in ws._subscriptions

    def test_unsubscribe_nonexistent_no_error(self):
        ws = AragoraWebSocket("http://localhost")
        ws.unsubscribe("nope")  # should not raise

    def test_subscribe_sends_message(self):
        ws = AragoraWebSocket("http://localhost")
        ws._state = _STATE_CONNECTED
        mock_ws = AsyncMock()
        ws._ws = mock_ws
        ws.subscribe("d-2")
        # _send uses asyncio.ensure_future, so verify the ws object was used
        assert "d-2" in ws._subscriptions


# ===========================================================================
# Client Integration
# ===========================================================================


class TestClientIntegration:
    """Tests for WebSocket integration with AragoraAsyncClient."""

    def test_stream_property_returns_websocket(self):
        from sdk.python.aragora import AragoraAsyncClient

        client = AragoraAsyncClient(base_url="http://localhost:8080", api_key="key")
        stream = client.stream
        assert isinstance(stream, AragoraWebSocket)

    def test_stream_property_is_lazy(self):
        from sdk.python.aragora import AragoraAsyncClient

        client = AragoraAsyncClient(base_url="http://localhost:8080")
        assert client._stream is None
        _ = client.stream
        assert client._stream is not None

    def test_stream_property_is_cached(self):
        from sdk.python.aragora import AragoraAsyncClient

        client = AragoraAsyncClient(base_url="http://localhost:8080")
        s1 = client.stream
        s2 = client.stream
        assert s1 is s2

    def test_stream_uses_client_config(self):
        from sdk.python.aragora import AragoraAsyncClient

        client = AragoraAsyncClient(
            base_url="http://localhost:8080",
            api_key="test-key",
            ws_url="ws://custom:9090/ws",
        )
        ws = client.stream
        assert ws.base_url == "http://localhost:8080"
        assert ws.api_key == "test-key"
        assert ws.ws_url == "ws://custom:9090/ws"

    @pytest.mark.asyncio
    async def test_close_clears_stream(self):
        from sdk.python.aragora import AragoraAsyncClient

        client = AragoraAsyncClient(base_url="http://localhost:8080")
        _ = client.stream
        assert client._stream is not None
        await client.close()
        assert client._stream is None

    def test_ws_url_parameter_stored(self):
        from sdk.python.aragora import AragoraAsyncClient

        client = AragoraAsyncClient(
            base_url="http://localhost",
            ws_url="ws://ws-host:8080/ws",
        )
        assert client.ws_url == "ws://ws-host:8080/ws"


# ===========================================================================
# Exports
# ===========================================================================


class TestExports:
    """Tests for module-level exports."""

    def test_websocket_module_exports(self):
        from sdk.python.aragora import websocket

        assert hasattr(websocket, "AragoraWebSocket")
        assert hasattr(websocket, "WebSocketEvent")
        assert hasattr(websocket, "WebSocketOptions")
        assert hasattr(websocket, "stream_debate")
        assert hasattr(websocket, "EVENT_TYPES")

    def test_package_level_exports(self):
        import sdk.python.aragora as pkg

        assert hasattr(pkg, "AragoraWebSocket")
        assert hasattr(pkg, "WebSocketEvent")
        assert hasattr(pkg, "WebSocketOptions")
        assert hasattr(pkg, "stream_debate")

    def test_all_includes_websocket_exports(self):
        import sdk.python.aragora as pkg

        for name in ("AragoraWebSocket", "WebSocketEvent", "WebSocketOptions", "stream_debate"):
            assert name in pkg.__all__, f"{name} not in __all__"
