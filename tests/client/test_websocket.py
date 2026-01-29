"""
Tests for aragora.client.websocket - WebSocket client for debate streaming.

These tests verify:
- DebateEventType: enum values, string representation
- DebateEvent: construction, from_dict parsing, edge cases
- WebSocketOptions: default values, custom values
- DebateStream: connection lifecycle, callbacks, event filtering,
  reconnection, timeout, error handling, async iteration
- stream_debate / stream_debate_by_id: convenience generators
"""

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.client.websocket import (
    MAX_EVENT_QUEUE_SIZE,
    DebateEvent,
    DebateEventType,
    DebateStream,
    WebSocketOptions,
    stream_debate,
    stream_debate_by_id,
)


# ---------------------------------------------------------------------------
# DebateEventType tests
# ---------------------------------------------------------------------------


class TestDebateEventType:
    """Tests for the DebateEventType enum."""

    def test_debate_event_types_are_strings(self):
        """All event types should be usable as plain strings."""
        assert DebateEventType.DEBATE_START == "debate_start"
        assert DebateEventType.CONSENSUS == "consensus"
        assert DebateEventType.ERROR == "error"

    def test_event_type_from_value(self):
        """Creating enum members from string values should work."""
        assert DebateEventType("agent_message") is DebateEventType.AGENT_MESSAGE
        assert DebateEventType("debate_end") is DebateEventType.DEBATE_END

    def test_invalid_event_type_raises(self):
        """Unknown values should raise ValueError."""
        with pytest.raises(ValueError):
            DebateEventType("nonexistent_event")

    def test_event_type_value_attribute(self):
        """The .value attribute should return the raw string."""
        assert DebateEventType.PING.value == "ping"
        assert DebateEventType.TOKEN_DELTA.value == "token_delta"


# ---------------------------------------------------------------------------
# DebateEvent tests
# ---------------------------------------------------------------------------


class TestDebateEvent:
    """Tests for DebateEvent dataclass and from_dict parsing."""

    def test_default_construction(self):
        """Constructing with just a type should set sensible defaults."""
        event = DebateEvent(type=DebateEventType.DEBATE_START)
        assert event.type == DebateEventType.DEBATE_START
        assert event.data == {}
        assert event.debate_id == ""
        assert event.round == 0
        assert event.agent == ""
        assert isinstance(event.timestamp, float)

    def test_from_dict_basic(self):
        """from_dict should parse a well-formed dictionary."""
        d = {
            "type": "agent_message",
            "debate_id": "d-1",
            "round": 2,
            "agent": "claude",
            "data": {"text": "hello"},
            "timestamp": 1700000000.0,
            "seq": 5,
            "agent_seq": 3,
        }
        event = DebateEvent.from_dict(d)
        assert event.type == DebateEventType.AGENT_MESSAGE
        assert event.debate_id == "d-1"
        assert event.round == 2
        assert event.agent == "claude"
        assert event.data == {"text": "hello"}
        assert event.timestamp == 1700000000.0
        assert event.seq == 5
        assert event.agent_seq == 3

    def test_from_dict_unknown_type_falls_back_to_error(self):
        """Unknown event types should map to ERROR."""
        event = DebateEvent.from_dict({"type": "unknown_xyz"})
        assert event.type == DebateEventType.ERROR

    def test_from_dict_missing_type_falls_back_to_error(self):
        """Missing type key should default to ERROR."""
        event = DebateEvent.from_dict({})
        assert event.type == DebateEventType.ERROR

    def test_from_dict_non_dict_data_becomes_empty(self):
        """Non-dict data field should be replaced with empty dict."""
        event = DebateEvent.from_dict({"type": "ping", "data": "not-a-dict"})
        assert event.data == {}

    def test_from_dict_iso_timestamp(self):
        """ISO 8601 timestamp strings should be parsed."""
        event = DebateEvent.from_dict(
            {
                "type": "ping",
                "timestamp": "2024-01-01T00:00:00",
            }
        )
        assert isinstance(event.timestamp, float)
        assert event.timestamp > 0

    def test_from_dict_invalid_timestamp_falls_back(self):
        """Invalid timestamp strings should fall back to current time."""
        before = time.time()
        event = DebateEvent.from_dict(
            {
                "type": "ping",
                "timestamp": "not-a-date",
            }
        )
        after = time.time()
        assert before <= event.timestamp <= after

    def test_from_dict_none_timestamp_falls_back(self):
        """None timestamp should fall back to current time."""
        before = time.time()
        event = DebateEvent.from_dict({"type": "ping", "timestamp": None})
        after = time.time()
        assert before <= event.timestamp <= after

    def test_from_dict_loop_id_from_data(self):
        """loop_id should be extractable from inner data dict."""
        event = DebateEvent.from_dict(
            {
                "type": "ping",
                "data": {"loop_id": "loop-99"},
            }
        )
        assert event.loop_id == "loop-99"

    def test_from_dict_debate_id_falls_back_to_loop_id(self):
        """debate_id should fall back to loop_id when absent."""
        event = DebateEvent.from_dict(
            {
                "type": "ping",
                "loop_id": "loop-42",
            }
        )
        assert event.debate_id == "loop-42"
        assert event.loop_id == "loop-42"


# ---------------------------------------------------------------------------
# WebSocketOptions tests
# ---------------------------------------------------------------------------


class TestWebSocketOptions:
    """Tests for WebSocketOptions defaults and overrides."""

    def test_defaults(self):
        opts = WebSocketOptions()
        assert opts.reconnect is True
        assert opts.reconnect_interval == 1.0
        assert opts.max_reconnect_attempts == 5
        assert opts.heartbeat_interval == 30.0
        assert opts.connect_timeout == 10.0

    def test_custom_values(self):
        opts = WebSocketOptions(
            reconnect=False,
            reconnect_interval=5.0,
            max_reconnect_attempts=10,
            heartbeat_interval=60.0,
            connect_timeout=20.0,
        )
        assert opts.reconnect is False
        assert opts.reconnect_interval == 5.0
        assert opts.max_reconnect_attempts == 10
        assert opts.heartbeat_interval == 60.0
        assert opts.connect_timeout == 20.0


# ---------------------------------------------------------------------------
# DebateStream tests
# ---------------------------------------------------------------------------


class TestDebateStream:
    """Tests for the DebateStream WebSocket client."""

    def test_build_url_from_ws(self):
        """ws:// URLs should get /ws appended."""
        stream = DebateStream("ws://localhost:8765", "d-1")
        assert stream.url == "ws://localhost:8765/ws"

    def test_build_url_already_has_ws_path(self):
        """URLs already ending in /ws should not get doubled."""
        stream = DebateStream("ws://localhost:8765/ws", "d-1")
        assert stream.url == "ws://localhost:8765/ws"

    def test_build_url_from_http(self):
        """http:// URLs should be converted to ws://."""
        stream = DebateStream("http://localhost:8080", "d-1")
        assert stream.url == "ws://localhost:8080/ws"

    def test_build_url_from_https(self):
        """https:// URLs should be converted to wss://."""
        stream = DebateStream("https://example.com", "d-1")
        assert stream.url == "wss://example.com/ws"

    def test_default_options(self):
        """Stream without explicit options should get defaults."""
        stream = DebateStream("ws://localhost", "d-1")
        assert stream.options.reconnect is True

    def test_custom_options(self):
        """Custom options should be stored."""
        opts = WebSocketOptions(reconnect=False)
        stream = DebateStream("ws://localhost", "d-1", options=opts)
        assert stream.options.reconnect is False

    def test_on_registers_callback(self):
        """on() should register event callbacks and return self for chaining."""
        stream = DebateStream("ws://localhost", "d-1")
        cb = MagicMock()
        result = stream.on("agent_message", cb)
        assert result is stream
        assert cb in stream._event_callbacks["agent_message"]

    def test_off_removes_callback(self):
        """off() should remove a previously registered callback."""
        stream = DebateStream("ws://localhost", "d-1")
        cb = MagicMock()
        stream.on("agent_message", cb)
        stream.off("agent_message", cb)
        assert cb not in stream._event_callbacks.get("agent_message", [])

    def test_off_nonexistent_callback_no_error(self):
        """off() with an unregistered callback should not raise."""
        stream = DebateStream("ws://localhost", "d-1")
        cb = MagicMock()
        stream.off("agent_message", cb)  # should not raise

    def test_on_error_registers(self):
        """on_error() should register error callbacks."""
        stream = DebateStream("ws://localhost", "d-1")
        cb = MagicMock()
        result = stream.on_error(cb)
        assert result is stream
        assert cb in stream._error_callbacks

    def test_on_close_registers(self):
        """on_close() should register close callbacks."""
        stream = DebateStream("ws://localhost", "d-1")
        cb = MagicMock()
        result = stream.on_close(cb)
        assert result is stream
        assert cb in stream._close_callbacks

    def test_emit_event_calls_type_specific_and_wildcard(self):
        """_emit_event should call both type-specific and wildcard callbacks."""
        stream = DebateStream("ws://localhost", "d-1")
        specific_cb = MagicMock()
        wildcard_cb = MagicMock()
        stream.on("ping", specific_cb)
        stream.on("*", wildcard_cb)

        event = DebateEvent(type=DebateEventType.PING, debate_id="d-1")
        stream._emit_event(event)

        specific_cb.assert_called_once_with(event)
        wildcard_cb.assert_called_once_with(event)

    def test_emit_event_ignores_callback_exceptions(self):
        """Callback exceptions should be caught, not propagated."""
        stream = DebateStream("ws://localhost", "d-1")
        bad_cb = MagicMock(side_effect=RuntimeError("boom"))
        good_cb = MagicMock()
        stream.on("ping", bad_cb)
        stream.on("ping", good_cb)

        event = DebateEvent(type=DebateEventType.PING, debate_id="d-1")
        stream._emit_event(event)  # should not raise

        bad_cb.assert_called_once()
        good_cb.assert_called_once()

    def test_emit_error_calls_callbacks(self):
        """_emit_error should invoke all registered error callbacks."""
        stream = DebateStream("ws://localhost", "d-1")
        cb = MagicMock()
        stream.on_error(cb)
        err = RuntimeError("test")
        stream._emit_error(err)
        cb.assert_called_once_with(err)

    def test_emit_close_calls_callbacks(self):
        """_emit_close should invoke all registered close callbacks."""
        stream = DebateStream("ws://localhost", "d-1")
        cb = MagicMock()
        stream.on_close(cb)
        stream._emit_close(1000, "normal")
        cb.assert_called_once_with(1000, "normal")

    def test_should_emit_no_debate_id_filter(self):
        """With empty debate_id, all events should be emitted."""
        stream = DebateStream("ws://localhost", "")
        event = DebateEvent(type=DebateEventType.PING, debate_id="any-id")
        assert stream._should_emit(event) is True

    def test_should_emit_matching_debate_id(self):
        """Events with matching debate_id should be emitted."""
        stream = DebateStream("ws://localhost", "d-1")
        event = DebateEvent(type=DebateEventType.PING, debate_id="d-1")
        assert stream._should_emit(event) is True

    def test_should_emit_non_matching_debate_id(self):
        """Events with different debate_id should be filtered out."""
        stream = DebateStream("ws://localhost", "d-1")
        event = DebateEvent(type=DebateEventType.PING, debate_id="d-2")
        assert stream._should_emit(event) is False

    def test_should_emit_event_without_id(self):
        """Events with no debate_id/loop_id should be emitted (broadcast)."""
        stream = DebateStream("ws://localhost", "d-1")
        event = DebateEvent(type=DebateEventType.PING, debate_id="", loop_id="")
        assert stream._should_emit(event) is True

    def test_is_connected_false_initially(self):
        """is_connected should be False before connecting."""
        stream = DebateStream("ws://localhost", "d-1")
        assert stream.is_connected is False

    @pytest.mark.asyncio
    async def test_connect_missing_websockets_import(self):
        """connect() should raise ImportError if websockets not installed."""
        stream = DebateStream("ws://localhost", "d-1")
        with patch.dict("sys.modules", {"websockets": None}):
            with pytest.raises(ImportError, match="websockets package required"):
                await stream.connect()

    @pytest.mark.asyncio
    async def test_connect_timeout(self):
        """connect() should raise ConnectionError on timeout."""
        stream = DebateStream(
            "ws://localhost", "d-1", options=WebSocketOptions(connect_timeout=0.01)
        )

        mock_ws_module = MagicMock()
        # Make connect() hang forever to trigger timeout
        never_resolves = asyncio.Future()
        mock_ws_module.connect.return_value = never_resolves

        with patch.dict("sys.modules", {"websockets": mock_ws_module}):
            with pytest.raises(ConnectionError, match="Connection timeout"):
                await stream.connect()

    @pytest.mark.asyncio
    async def test_connect_generic_failure(self):
        """connect() should wrap generic exceptions in ConnectionError."""
        stream = DebateStream(
            "ws://localhost", "d-1", options=WebSocketOptions(connect_timeout=1.0)
        )

        mock_ws_module = MagicMock()
        fut = asyncio.Future()
        fut.set_exception(OSError("refused"))
        mock_ws_module.connect.return_value = fut

        with patch.dict("sys.modules", {"websockets": mock_ws_module}):
            with pytest.raises(ConnectionError, match="Failed to connect"):
                await stream.connect()

    @pytest.mark.asyncio
    async def test_disconnect_sets_closing_flag(self):
        """disconnect() should set _is_closing and clear ws."""
        stream = DebateStream("ws://localhost", "d-1")
        mock_ws = AsyncMock()
        mock_ws.open = True
        stream._ws = mock_ws

        await stream.disconnect()
        assert stream._is_closing is True
        assert stream._ws is None
        mock_ws.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_disconnect_cancels_tasks(self):
        """disconnect() should cancel heartbeat and reconnect tasks."""
        stream = DebateStream("ws://localhost", "d-1")
        stream._ws = AsyncMock()

        hb_task = MagicMock()
        rc_task = MagicMock()
        stream._heartbeat_task = hb_task
        stream._reconnect_task = rc_task

        await stream.disconnect()
        hb_task.cancel.assert_called_once()
        rc_task.cancel.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_when_connected(self):
        """send() should serialize to JSON and send via ws."""
        stream = DebateStream("ws://localhost", "d-1")
        mock_ws = AsyncMock()
        stream._ws = mock_ws

        await stream.send({"action": "vote", "value": 1})
        mock_ws.send.assert_awaited_once()
        sent = json.loads(mock_ws.send.call_args[0][0])
        assert sent == {"action": "vote", "value": 1}

    @pytest.mark.asyncio
    async def test_send_when_not_connected(self):
        """send() should be a no-op when ws is None."""
        stream = DebateStream("ws://localhost", "d-1")
        await stream.send({"action": "test"})  # should not raise

    @pytest.mark.asyncio
    async def test_anext_returns_ping_on_timeout_when_connected(self):
        """__anext__ should return a synthetic PING event on timeout if connected."""
        stream = DebateStream(
            "ws://localhost", "d-1", options=WebSocketOptions(heartbeat_interval=0.01)
        )
        mock_ws = MagicMock()
        mock_ws.open = True
        stream._ws = mock_ws

        event = await stream.__anext__()
        assert event.type == DebateEventType.PING
        assert event.data == {"keepalive": True}

    @pytest.mark.asyncio
    async def test_anext_raises_stop_when_closing_and_empty(self):
        """__anext__ should raise StopAsyncIteration when closing with empty queue."""
        stream = DebateStream("ws://localhost", "d-1")
        stream._is_closing = True

        with pytest.raises(StopAsyncIteration):
            await stream.__anext__()

    @pytest.mark.asyncio
    async def test_anext_returns_queued_event(self):
        """__anext__ should return events from the queue."""
        stream = DebateStream(
            "ws://localhost", "d-1", options=WebSocketOptions(heartbeat_interval=0.05)
        )
        event = DebateEvent(type=DebateEventType.DEBATE_START, debate_id="d-1")
        await stream._event_queue.put(event)

        result = await stream.__anext__()
        assert result.type == DebateEventType.DEBATE_START

    @pytest.mark.asyncio
    async def test_aiter_returns_self(self):
        """__aiter__ should return self."""
        stream = DebateStream("ws://localhost", "d-1")
        assert stream.__aiter__() is stream

    @pytest.mark.asyncio
    async def test_attempt_reconnect_max_reached(self):
        """Reconnect should emit error when max attempts exceeded."""
        stream = DebateStream(
            "ws://localhost", "d-1", options=WebSocketOptions(max_reconnect_attempts=0)
        )
        error_cb = MagicMock()
        stream.on_error(error_cb)

        await stream._attempt_reconnect()
        error_cb.assert_called_once()
        err = error_cb.call_args[0][0]
        assert "Max reconnect attempts" in str(err)

    @pytest.mark.asyncio
    async def test_attempt_reconnect_increments_count(self):
        """Each reconnect attempt should increment the counter."""
        opts = WebSocketOptions(
            reconnect_interval=0.001,
            max_reconnect_attempts=3,
        )
        stream = DebateStream("ws://localhost", "d-1", options=opts)

        with patch.object(stream, "connect", new_callable=AsyncMock) as mock_connect:
            mock_connect.return_value = None
            await stream._attempt_reconnect()
            assert stream._reconnect_attempts == 1
            mock_connect.assert_awaited_once()
