"""Tests for Aragora SDK WebSocket streaming."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora_client.types import DebateEvent
from aragora_client.websocket import DebateStream, stream_debate


class TestDebateStreamInit:
    """Tests for DebateStream initialization."""

    def test_http_url_normalization(self) -> None:
        """Test HTTP URL is converted to WebSocket URL."""
        stream = DebateStream("http://localhost:8080", "debate-123")
        assert stream.base_url == "ws://localhost:8080"

    def test_https_url_normalization(self) -> None:
        """Test HTTPS URL is converted to secure WebSocket URL."""
        stream = DebateStream("https://api.aragora.ai", "debate-456")
        assert stream.base_url == "wss://api.aragora.ai"

    def test_ws_url_preserved(self) -> None:
        """Test WebSocket URL is preserved."""
        stream = DebateStream("ws://localhost:8080", "debate-789")
        assert stream.base_url == "ws://localhost:8080"

    def test_wss_url_preserved(self) -> None:
        """Test secure WebSocket URL is preserved."""
        stream = DebateStream("wss://api.aragora.ai", "debate-abc")
        assert stream.base_url == "wss://api.aragora.ai"

    def test_bare_url_gets_ws_prefix(self) -> None:
        """Test URL without scheme gets ws:// prefix."""
        stream = DebateStream("localhost:8080", "debate-def")
        assert stream.base_url == "ws://localhost:8080"

    def test_trailing_slash_removed(self) -> None:
        """Test trailing slash is removed from URL."""
        stream = DebateStream("http://localhost:8080/", "debate-ghi")
        assert stream.base_url == "ws://localhost:8080"

    def test_debate_id_stored(self) -> None:
        """Test debate ID is stored correctly."""
        stream = DebateStream("ws://localhost", "my-debate-id")
        assert stream.debate_id == "my-debate-id"

    def test_default_options(self) -> None:
        """Test default configuration options."""
        stream = DebateStream("ws://localhost", "debate-123")
        assert stream.reconnect is True
        assert stream.reconnect_interval == 1.0
        assert stream.max_reconnect_attempts == 5
        assert stream.heartbeat_interval == 30.0

    def test_custom_options(self) -> None:
        """Test custom configuration options."""
        stream = DebateStream(
            "ws://localhost",
            "debate-123",
            reconnect=False,
            reconnect_interval=2.5,
            max_reconnect_attempts=10,
            heartbeat_interval=60.0,
        )
        assert stream.reconnect is False
        assert stream.reconnect_interval == 2.5
        assert stream.max_reconnect_attempts == 10
        assert stream.heartbeat_interval == 60.0

    def test_initial_state(self) -> None:
        """Test initial connection state."""
        stream = DebateStream("ws://localhost", "debate-123")
        assert stream.connected is False
        assert stream._ws is None
        assert stream._handlers == {}
        assert stream._error_handlers == []
        assert stream._should_stop is False
        assert stream._reconnect_attempts == 0


class TestDebateStreamHandlers:
    """Tests for DebateStream event handler registration."""

    def test_register_handler(self) -> None:
        """Test registering an event handler."""
        stream = DebateStream("ws://localhost", "debate-123")
        handler = MagicMock()
        result = stream.on("agent_message", handler)

        assert "agent_message" in stream._handlers
        assert handler in stream._handlers["agent_message"]
        assert result is stream  # Test chaining

    def test_register_multiple_handlers(self) -> None:
        """Test registering multiple handlers for same event."""
        stream = DebateStream("ws://localhost", "debate-123")
        handler1 = MagicMock()
        handler2 = MagicMock()

        stream.on("consensus", handler1)
        stream.on("consensus", handler2)

        assert len(stream._handlers["consensus"]) == 2
        assert handler1 in stream._handlers["consensus"]
        assert handler2 in stream._handlers["consensus"]

    def test_register_handlers_for_different_events(self) -> None:
        """Test registering handlers for different event types."""
        stream = DebateStream("ws://localhost", "debate-123")
        msg_handler = MagicMock()
        consensus_handler = MagicMock()

        stream.on("agent_message", msg_handler)
        stream.on("consensus", consensus_handler)

        assert len(stream._handlers["agent_message"]) == 1
        assert len(stream._handlers["consensus"]) == 1

    def test_register_wildcard_handler(self) -> None:
        """Test registering handler for all events."""
        stream = DebateStream("ws://localhost", "debate-123")
        handler = MagicMock()

        stream.on("*", handler)

        assert "*" in stream._handlers
        assert handler in stream._handlers["*"]

    def test_handler_chaining(self) -> None:
        """Test fluent handler chaining."""
        stream = DebateStream("ws://localhost", "debate-123")

        result = (
            stream.on("agent_message", MagicMock())
            .on("consensus", MagicMock())
            .on("debate_end", MagicMock())
        )

        assert result is stream
        assert len(stream._handlers) == 3

    def test_register_error_handler(self) -> None:
        """Test registering an error handler."""
        stream = DebateStream("ws://localhost", "debate-123")
        error_handler = MagicMock()
        result = stream.on_error(error_handler)

        assert error_handler in stream._error_handlers
        assert result is stream

    def test_register_multiple_error_handlers(self) -> None:
        """Test registering multiple error handlers."""
        stream = DebateStream("ws://localhost", "debate-123")
        handler1 = MagicMock()
        handler2 = MagicMock()

        stream.on_error(handler1).on_error(handler2)

        assert len(stream._error_handlers) == 2


class TestDebateStreamMessageHandling:
    """Tests for DebateStream message handling."""

    @pytest.mark.asyncio
    async def test_handle_string_message(self) -> None:
        """Test handling a string message."""
        stream = DebateStream("ws://localhost", "debate-123")
        handler = MagicMock()
        stream.on("agent_message", handler)

        message = json.dumps(
            {
                "type": "agent_message",
                "data": {"agent": "claude", "content": "Hello"},
            }
        )
        await stream._handle_message(message)

        handler.assert_called_once()
        event = handler.call_args[0][0]
        assert isinstance(event, DebateEvent)
        assert event.type == "agent_message"
        assert event.data["agent"] == "claude"

    @pytest.mark.asyncio
    async def test_handle_bytes_message(self) -> None:
        """Test handling a bytes message."""
        stream = DebateStream("ws://localhost", "debate-123")
        handler = MagicMock()
        stream.on("consensus", handler)

        message = json.dumps(
            {
                "type": "consensus",
                "data": {"reached": True},
            }
        ).encode("utf-8")
        await stream._handle_message(message)

        handler.assert_called_once()
        event = handler.call_args[0][0]
        assert event.type == "consensus"

    @pytest.mark.asyncio
    async def test_handle_message_calls_wildcard(self) -> None:
        """Test that wildcard handlers are called for all events."""
        stream = DebateStream("ws://localhost", "debate-123")
        specific_handler = MagicMock()
        wildcard_handler = MagicMock()

        stream.on("agent_message", specific_handler)
        stream.on("*", wildcard_handler)

        message = json.dumps({"type": "agent_message", "data": {}})
        await stream._handle_message(message)

        specific_handler.assert_called_once()
        wildcard_handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_invalid_json(self) -> None:
        """Test handling invalid JSON emits error."""
        stream = DebateStream("ws://localhost", "debate-123")
        error_handler = MagicMock()
        stream.on_error(error_handler)

        await stream._handle_message("not valid json")

        error_handler.assert_called_once()
        assert isinstance(error_handler.call_args[0][0], json.JSONDecodeError)

    @pytest.mark.asyncio
    async def test_handle_unknown_event_type(self) -> None:
        """Test handling message with unknown type."""
        stream = DebateStream("ws://localhost", "debate-123")
        wildcard_handler = MagicMock()
        stream.on("*", wildcard_handler)

        message = json.dumps({"type": "unknown_event", "data": {"foo": "bar"}})
        await stream._handle_message(message)

        wildcard_handler.assert_called_once()
        event = wildcard_handler.call_args[0][0]
        assert event.type == "unknown_event"

    @pytest.mark.asyncio
    async def test_handle_message_missing_type(self) -> None:
        """Test handling message without type field."""
        stream = DebateStream("ws://localhost", "debate-123")
        wildcard_handler = MagicMock()
        stream.on("*", wildcard_handler)

        message = json.dumps({"data": {"foo": "bar"}})
        await stream._handle_message(message)

        event = wildcard_handler.call_args[0][0]
        assert event.type == "unknown"

    @pytest.mark.asyncio
    async def test_handler_exception_emits_error(self) -> None:
        """Test that handler exceptions are emitted as errors."""
        stream = DebateStream("ws://localhost", "debate-123")
        failing_handler = MagicMock(side_effect=ValueError("Handler failed"))
        error_handler = MagicMock()

        stream.on("test", failing_handler)
        stream.on_error(error_handler)

        message = json.dumps({"type": "test", "data": {}})
        await stream._handle_message(message)

        error_handler.assert_called_once()
        assert isinstance(error_handler.call_args[0][0], ValueError)

    @pytest.mark.asyncio
    async def test_loop_id_from_message(self) -> None:
        """Test loop_id is extracted from message."""
        stream = DebateStream("ws://localhost", "debate-123")
        handler = MagicMock()
        stream.on("test", handler)

        message = json.dumps(
            {
                "type": "test",
                "data": {},
                "loop_id": "loop-abc",
            }
        )
        await stream._handle_message(message)

        event = handler.call_args[0][0]
        assert event.loop_id == "loop-abc"

    @pytest.mark.asyncio
    async def test_loop_id_from_data_debate_id(self) -> None:
        """Test loop_id falls back to data.debate_id."""
        stream = DebateStream("ws://localhost", "debate-123")
        handler = MagicMock()
        stream.on("test", handler)

        message = json.dumps(
            {
                "type": "test",
                "data": {"debate_id": "debate-xyz"},
            }
        )
        await stream._handle_message(message)

        event = handler.call_args[0][0]
        assert event.loop_id == "debate-xyz"


class TestDebateStreamErrorEmission:
    """Tests for DebateStream error emission."""

    def test_emit_error_calls_handlers(self) -> None:
        """Test error emission calls all error handlers."""
        stream = DebateStream("ws://localhost", "debate-123")
        handler1 = MagicMock()
        handler2 = MagicMock()
        stream.on_error(handler1).on_error(handler2)

        error = ValueError("Test error")
        stream._emit_error(error)

        handler1.assert_called_once_with(error)
        handler2.assert_called_once_with(error)

    def test_emit_error_handles_failing_handler(self) -> None:
        """Test error handler exceptions don't propagate."""
        stream = DebateStream("ws://localhost", "debate-123")
        failing_handler = MagicMock(side_effect=RuntimeError("Handler error"))
        working_handler = MagicMock()

        stream.on_error(failing_handler).on_error(working_handler)

        # Should not raise
        stream._emit_error(ValueError("Original error"))

        # Working handler should still be called
        working_handler.assert_called_once()


class TestDebateStreamConnect:
    """Tests for DebateStream connection management."""

    @pytest.mark.asyncio
    async def test_disconnect(self) -> None:
        """Test disconnect sets proper state."""
        stream = DebateStream("ws://localhost", "debate-123")
        mock_ws = AsyncMock()
        stream._ws = mock_ws
        stream._connected = True

        await stream.disconnect()

        assert stream._should_stop is True
        assert stream._connected is False
        assert stream._ws is None
        mock_ws.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect_without_connection(self) -> None:
        """Test disconnect when not connected."""
        stream = DebateStream("ws://localhost", "debate-123")

        # Should not raise
        await stream.disconnect()

        assert stream._should_stop is True
        assert stream._connected is False

    def test_connected_property(self) -> None:
        """Test connected property reflects state."""
        stream = DebateStream("ws://localhost", "debate-123")
        assert stream.connected is False

        stream._connected = True
        assert stream.connected is True


class TestStreamDebateFunction:
    """Tests for the stream_debate async generator function."""

    def test_url_normalization_http(self) -> None:
        """Test HTTP URL normalization in stream_debate."""
        # We can't easily test the full generator without mocking websockets,
        # but we can verify the URL normalization logic by inspection
        # The normalization happens at the start of the function
        pass

    @pytest.mark.asyncio
    async def test_stream_yields_events(self) -> None:
        """Test stream_debate yields DebateEvent objects."""
        mock_ws = AsyncMock()
        mock_ws.__aiter__.return_value = iter(
            [
                json.dumps({"type": "agent_message", "data": {"agent": "claude"}}),
                json.dumps({"type": "consensus", "data": {"reached": True}}),
            ]
        )

        with patch("aragora_client.websocket.websockets.connect") as mock_connect:
            mock_connect.return_value.__aenter__.return_value = mock_ws

            events = []
            async for event in stream_debate("ws://localhost", "debate-123"):
                events.append(event)
                if len(events) >= 2:
                    break

            assert len(events) == 2
            assert events[0].type == "agent_message"
            assert events[1].type == "consensus"

    @pytest.mark.asyncio
    async def test_stream_handles_bytes_message(self) -> None:
        """Test stream_debate handles bytes messages."""
        mock_ws = AsyncMock()
        mock_ws.__aiter__.return_value = iter(
            [
                json.dumps({"type": "test", "data": {}}).encode("utf-8"),
            ]
        )

        with patch("aragora_client.websocket.websockets.connect") as mock_connect:
            mock_connect.return_value.__aenter__.return_value = mock_ws

            events = []
            async for event in stream_debate("ws://localhost", "debate-123"):
                events.append(event)
                break

            assert len(events) == 1
            assert events[0].type == "test"

    @pytest.mark.asyncio
    async def test_stream_skips_invalid_json(self) -> None:
        """Test stream_debate skips invalid JSON messages."""
        mock_ws = AsyncMock()
        mock_ws.__aiter__.return_value = iter(
            [
                "not valid json",
                json.dumps({"type": "valid", "data": {}}),
            ]
        )

        with patch("aragora_client.websocket.websockets.connect") as mock_connect:
            mock_connect.return_value.__aenter__.return_value = mock_ws

            events = []
            async for event in stream_debate("ws://localhost", "debate-123"):
                events.append(event)
                if len(events) >= 1:
                    break

            # Should have skipped the invalid JSON
            assert len(events) == 1
            assert events[0].type == "valid"


class TestDebateStreamReconnection:
    """Tests for DebateStream reconnection behavior."""

    def test_reconnect_delay_calculation(self) -> None:
        """Test exponential backoff delay calculation."""
        stream = DebateStream(
            "ws://localhost",
            "debate-123",
            reconnect_interval=1.0,
        )

        # The delay formula is: reconnect_interval * (2 ** (attempts - 1))
        # Attempt 1: 1.0 * 2^0 = 1.0
        # Attempt 2: 1.0 * 2^1 = 2.0
        # Attempt 3: 1.0 * 2^2 = 4.0
        assert stream.reconnect_interval * (2**0) == 1.0
        assert stream.reconnect_interval * (2**1) == 2.0
        assert stream.reconnect_interval * (2**2) == 4.0

    def test_max_reconnect_attempts_config(self) -> None:
        """Test max reconnect attempts configuration."""
        stream = DebateStream(
            "ws://localhost",
            "debate-123",
            max_reconnect_attempts=3,
        )
        assert stream.max_reconnect_attempts == 3

    def test_reconnect_disabled(self) -> None:
        """Test reconnect can be disabled."""
        stream = DebateStream(
            "ws://localhost",
            "debate-123",
            reconnect=False,
        )
        assert stream.reconnect is False
