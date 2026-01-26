"""Tests for Python SDK streaming parity with TypeScript SDK."""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch


class TestDebateStreamOnce:
    """Tests for DebateStream.once() method."""

    def test_once_method_exists(self):
        """Test that once method exists on DebateStream."""
        from aragora.client.websocket import DebateStream

        stream = DebateStream(base_url="ws://localhost:8080", debate_id="test-123")
        assert hasattr(stream, "once")
        assert callable(stream.once)

    @pytest.mark.asyncio
    async def test_once_returns_event(self):
        """Test once() returns the first matching event."""
        from aragora.client.websocket import DebateStream, DebateEvent, DebateEventType

        stream = DebateStream(base_url="ws://localhost:8080", debate_id="test-123")

        # Simulate an event being emitted after a short delay
        async def emit_after_delay():
            await asyncio.sleep(0.01)
            event = DebateEvent(
                type=DebateEventType.DEBATE_START,
                data={"debate_id": "test-123"},
                debate_id="test-123",
            )
            stream._emit_event(event)

        task = asyncio.create_task(emit_after_delay())

        event = await stream.once("debate_start", timeout=1.0)

        assert isinstance(event, DebateEvent)
        assert event.type == DebateEventType.DEBATE_START

        await task

    @pytest.mark.asyncio
    async def test_once_times_out(self):
        """Test once() raises TimeoutError when timeout exceeded."""
        from aragora.client.websocket import DebateStream

        stream = DebateStream(base_url="ws://localhost:8080", debate_id="test-123")

        with pytest.raises(asyncio.TimeoutError):
            await stream.once("nonexistent_event", timeout=0.01)

    @pytest.mark.asyncio
    async def test_once_cleans_up_handler(self):
        """Test once() removes handler after completion."""
        from aragora.client.websocket import DebateStream, DebateEvent, DebateEventType

        stream = DebateStream(base_url="ws://localhost:8080", debate_id="test-123")

        async def emit_after_delay():
            await asyncio.sleep(0.01)
            event = DebateEvent(
                type=DebateEventType.CONSENSUS,
                data={"result": "yes"},
                debate_id="test-123",
            )
            stream._emit_event(event)

        task = asyncio.create_task(emit_after_delay())

        await stream.once("consensus", timeout=1.0)
        await task

        # Handler should be removed
        assert (
            "consensus" not in stream._event_callbacks
            or len(stream._event_callbacks.get("consensus", [])) == 0
        )

    @pytest.mark.asyncio
    async def test_once_cleans_up_on_timeout(self):
        """Test once() removes handler even on timeout."""
        from aragora.client.websocket import DebateStream

        stream = DebateStream(base_url="ws://localhost:8080", debate_id="test-123")

        with pytest.raises(asyncio.TimeoutError):
            await stream.once("timeout_event", timeout=0.01)

        # Handler should be removed
        assert (
            "timeout_event" not in stream._event_callbacks
            or len(stream._event_callbacks.get("timeout_event", [])) == 0
        )


class TestStreamDebateById:
    """Tests for stream_debate_by_id convenience function."""

    def test_stream_debate_by_id_importable(self):
        """Test stream_debate_by_id is importable from aragora.client."""
        from aragora.client import stream_debate_by_id

        assert stream_debate_by_id is not None
        assert callable(stream_debate_by_id)

    def test_stream_debate_by_id_from_websocket(self):
        """Test stream_debate_by_id is importable from websocket module."""
        from aragora.client.websocket import stream_debate_by_id

        assert stream_debate_by_id is not None

    @pytest.mark.asyncio
    async def test_stream_debate_by_id_requires_debate_id(self):
        """Test stream_debate_by_id signature requires debate_id."""
        from aragora.client.websocket import stream_debate_by_id
        import inspect

        sig = inspect.signature(stream_debate_by_id)
        params = list(sig.parameters.keys())

        assert "debate_id" in params
        # debate_id should be required (no default)
        assert sig.parameters["debate_id"].default == inspect.Parameter.empty


class TestClientStreamingMethods:
    """Tests for AragoraClient streaming convenience methods."""

    def test_client_has_stream_debate_method(self):
        """Test AragoraClient has stream_debate method."""
        from aragora.client import AragoraClient

        client = AragoraClient(base_url="http://localhost:8080")
        assert hasattr(client, "stream_debate")
        assert callable(client.stream_debate)

    def test_client_has_stream_all_debates_method(self):
        """Test AragoraClient has stream_all_debates method."""
        from aragora.client import AragoraClient

        client = AragoraClient(base_url="http://localhost:8080")
        assert hasattr(client, "stream_all_debates")
        assert callable(client.stream_all_debates)

    def test_client_has_create_debate_and_stream_method(self):
        """Test AragoraClient has create_debate_and_stream method."""
        from aragora.client import AragoraClient

        client = AragoraClient(base_url="http://localhost:8080")
        assert hasattr(client, "create_debate_and_stream")
        assert callable(client.create_debate_and_stream)

    @pytest.mark.asyncio
    async def test_stream_debate_is_async_generator(self):
        """Test stream_debate returns an async generator."""
        from aragora.client import AragoraClient
        import inspect

        client = AragoraClient(base_url="http://localhost:8080")

        # The method should be an async generator function
        assert inspect.isasyncgenfunction(client.stream_debate)

    @pytest.mark.asyncio
    async def test_stream_all_debates_is_async_generator(self):
        """Test stream_all_debates returns an async generator."""
        from aragora.client import AragoraClient
        import inspect

        client = AragoraClient(base_url="http://localhost:8080")

        assert inspect.isasyncgenfunction(client.stream_all_debates)


class TestStreamingExports:
    """Tests for streaming module exports."""

    def test_all_streaming_exports_from_client(self):
        """Test all streaming exports are available from aragora.client package."""
        from aragora.client import (
            DebateStream,
            DebateEvent,
            DebateEventType,
            WebSocketOptions,
            stream_debate,
            stream_debate_by_id,
        )

        assert DebateStream is not None
        assert DebateEvent is not None
        assert DebateEventType is not None
        assert WebSocketOptions is not None
        assert stream_debate is not None
        assert stream_debate_by_id is not None

    def test_debate_event_type_enum(self):
        """Test DebateEventType enum values."""
        from aragora.client import DebateEventType

        assert DebateEventType.DEBATE_START.value == "debate_start"
        assert DebateEventType.DEBATE_END.value == "debate_end"
        assert DebateEventType.CONSENSUS.value == "consensus"
        assert DebateEventType.AGENT_MESSAGE.value == "agent_message"

    def test_websocket_options_defaults(self):
        """Test WebSocketOptions default values."""
        from aragora.client import WebSocketOptions

        options = WebSocketOptions()
        assert options.reconnect is True
        assert options.max_reconnect_attempts == 5
        assert options.reconnect_interval == 1.0
        assert options.heartbeat_interval == 30.0


class TestDebateEventModel:
    """Tests for DebateEvent dataclass."""

    def test_debate_event_creation(self):
        """Test DebateEvent can be created."""
        from aragora.client import DebateEvent, DebateEventType

        event = DebateEvent(
            type=DebateEventType.AGENT_MESSAGE,
            data={"content": "Hello"},
            debate_id="debate-123",
        )

        assert event.type == DebateEventType.AGENT_MESSAGE
        assert event.data == {"content": "Hello"}
        assert event.debate_id == "debate-123"

    def test_debate_event_from_dict(self):
        """Test DebateEvent.from_dict factory method."""
        from aragora.client import DebateEvent, DebateEventType

        data = {
            "type": "consensus",
            "data": {"result": "Yes"},
            "debate_id": "d-456",
            "timestamp": 1704067200.0,
        }

        event = DebateEvent.from_dict(data)

        assert event.type == DebateEventType.CONSENSUS
        assert event.data == {"result": "Yes"}
        assert event.debate_id == "d-456"

    def test_debate_event_from_dict_defaults(self):
        """Test DebateEvent.from_dict handles missing fields."""
        from aragora.client import DebateEvent, DebateEventType

        event = DebateEvent.from_dict({})

        assert event.type == DebateEventType.ERROR
        assert event.data == {}
        assert event.debate_id == ""


class TestTypescriptParity:
    """Tests verifying Python SDK has parity with TypeScript SDK features."""

    def test_once_method_parity(self):
        """Test Python has once() like TypeScript."""
        from aragora.client.websocket import DebateStream

        # TypeScript: await stream.once('debate_start')
        # Python should have the same pattern
        stream = DebateStream(base_url="ws://localhost:8080", debate_id="test")
        assert hasattr(stream, "once")

    def test_client_stream_debate_parity(self):
        """Test Python client has stream_debate like TypeScript."""
        from aragora.client import AragoraClient

        # TypeScript: client.streamDebate(debateId)
        # Python: client.stream_debate(debate_id)
        client = AragoraClient()
        assert hasattr(client, "stream_debate")

    def test_client_stream_all_debates_parity(self):
        """Test Python client has stream_all_debates like TypeScript."""
        from aragora.client import AragoraClient

        # TypeScript: client.streamAllDebates()
        # Python: client.stream_all_debates()
        client = AragoraClient()
        assert hasattr(client, "stream_all_debates")

    def test_client_create_debate_and_stream_parity(self):
        """Test Python client has create_debate_and_stream like TypeScript."""
        from aragora.client import AragoraClient

        # TypeScript: client.createDebateAndStream(request)
        # Python: client.create_debate_and_stream(...)
        client = AragoraClient()
        assert hasattr(client, "create_debate_and_stream")

    def test_stream_debate_by_id_parity(self):
        """Test Python has stream_debate_by_id convenience function."""
        from aragora.client import stream_debate_by_id

        # Convenience function for streaming specific debate
        assert stream_debate_by_id is not None
