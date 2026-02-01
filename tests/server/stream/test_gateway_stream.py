"""
Tests for Gateway WebSocket Stream Server.

Tests cover:
- GatewayEvent: Event structure and serialization
- GatewayStreamServer: Subscription management, broadcasting, event methods
- Global server instance management
- Error handling and isolation
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from aragora.server.stream.gateway_stream import (
    GatewayEvent,
    GatewayEventType,
    GatewayStreamServer,
    get_gateway_stream_server,
    set_gateway_stream_server,
)


# ===========================================================================
# Test Fixtures
# ===========================================================================


@pytest.fixture
def server():
    """Create a fresh GatewayStreamServer instance."""
    return GatewayStreamServer()


@pytest.fixture(autouse=True)
def reset_global_server():
    """Reset the global server before each test."""
    import aragora.server.stream.gateway_stream as module

    original = module._gateway_stream_server
    module._gateway_stream_server = None
    yield
    module._gateway_stream_server = original


# ===========================================================================
# Test GatewayEvent
# ===========================================================================


class TestGatewayEvent:
    """Tests for GatewayEvent dataclass."""

    def test_event_to_dict(self):
        """Event converts to dictionary correctly."""
        event = GatewayEvent(
            event_type=GatewayEventType.AGENT_RESPONSE_CHUNK,
            agent_name="claude",
            data={"chunk": "Hello"},
            debate_id="debate-123",
            sequence=5,
        )
        result = event.to_dict()

        assert result["type"] == "agent_response_chunk"
        assert result["agent"] == "claude"
        assert result["data"]["chunk"] == "Hello"
        assert result["debate_id"] == "debate-123"
        assert result["sequence"] == 5
        assert "timestamp" in result

    def test_event_has_default_timestamp(self):
        """Event has timestamp automatically set."""
        event = GatewayEvent(
            event_type=GatewayEventType.AGENT_COMPLETE,
            agent_name="gpt4",
            data={},
        )
        assert event.timestamp is not None
        assert isinstance(event.timestamp, str)
        # ISO format should contain 'T'
        assert "T" in event.timestamp

    def test_event_default_sequence(self):
        """Event has default sequence of 0."""
        event = GatewayEvent(
            event_type=GatewayEventType.AGENT_ERROR,
            agent_name="test",
            data={},
        )
        assert event.sequence == 0

    def test_event_default_debate_id(self):
        """Event has default debate_id of None."""
        event = GatewayEvent(
            event_type=GatewayEventType.AGENT_ERROR,
            agent_name="test",
            data={},
        )
        assert event.debate_id is None


# ===========================================================================
# Test GatewayEventType
# ===========================================================================


class TestGatewayEventType:
    """Tests for GatewayEventType enum."""

    def test_all_event_types_exist(self):
        """All expected event types are defined."""
        assert GatewayEventType.AGENT_RESPONSE_CHUNK.value == "agent_response_chunk"
        assert GatewayEventType.AGENT_CAPABILITY_USED.value == "agent_capability_used"
        assert GatewayEventType.AGENT_ERROR.value == "agent_error"
        assert GatewayEventType.AGENT_COMPLETE.value == "agent_complete"
        assert GatewayEventType.DEBATE_PROGRESS.value == "debate_progress"
        assert GatewayEventType.VERIFICATION_UPDATE.value == "verification_update"


# ===========================================================================
# Test Subscribe/Unsubscribe
# ===========================================================================


class TestSubscription:
    """Tests for subscription management."""

    @pytest.mark.asyncio
    async def test_subscribe_returns_unsubscribe(self, server):
        """Subscribe returns an unsubscribe callable."""
        callback = MagicMock()
        unsubscribe = await server.subscribe(callback)

        assert callable(unsubscribe)

    @pytest.mark.asyncio
    async def test_unsubscribe_removes_callback(self, server):
        """Unsubscribe stops events from being received."""
        received_events = []

        async def callback(event):
            received_events.append(event)

        unsubscribe = await server.subscribe(callback)

        # Send event before unsubscribe
        await server.broadcast_agent_chunk("test", "chunk1")
        assert len(received_events) == 1

        # Unsubscribe
        unsubscribe()

        # Send event after unsubscribe
        await server.broadcast_agent_chunk("test", "chunk2")
        assert len(received_events) == 1  # Still 1, no new event

    @pytest.mark.asyncio
    async def test_subscribe_to_specific_debate(self, server):
        """Can subscribe to a specific debate ID."""
        received = []

        async def callback(event):
            received.append(event)

        await server.subscribe(callback, debate_id="debate-123")

        # Event for subscribed debate
        await server.broadcast_agent_chunk("test", "chunk", debate_id="debate-123")
        assert len(received) == 1

        # Event for different debate should not be received
        await server.broadcast_agent_chunk("test", "chunk", debate_id="debate-456")
        assert len(received) == 1  # Still 1

    @pytest.mark.asyncio
    async def test_unsubscribe_from_specific_debate(self, server):
        """Unsubscribe removes debate-specific callback."""
        callback = MagicMock()
        unsubscribe = await server.subscribe(callback, debate_id="debate-123")

        unsubscribe()

        assert server.get_subscriber_count("debate-123") == 0


# ===========================================================================
# Test Broadcasting
# ===========================================================================


class TestBroadcasting:
    """Tests for event broadcasting."""

    @pytest.mark.asyncio
    async def test_broadcast_to_global_subscribers(self, server):
        """Global subscribers receive all events."""
        received = []

        async def callback(event):
            received.append(event)

        await server.subscribe(callback)  # Global subscription

        await server.broadcast_agent_chunk("test", "chunk1", debate_id="d1")
        await server.broadcast_agent_chunk("test", "chunk2", debate_id="d2")
        await server.broadcast_agent_chunk("test", "chunk3")  # No debate ID

        assert len(received) == 3

    @pytest.mark.asyncio
    async def test_broadcast_to_debate_subscribers(self, server):
        """Debate-specific subscribers receive only their events."""
        debate1_events = []
        debate2_events = []

        async def callback1(event):
            debate1_events.append(event)

        async def callback2(event):
            debate2_events.append(event)

        await server.subscribe(callback1, debate_id="debate-1")
        await server.subscribe(callback2, debate_id="debate-2")

        await server.broadcast_agent_chunk("test", "chunk", debate_id="debate-1")

        assert len(debate1_events) == 1
        assert len(debate2_events) == 0

    @pytest.mark.asyncio
    async def test_broadcast_returns_count(self, server):
        """Broadcast returns the count of subscribers notified."""
        cb1 = AsyncMock()
        cb2 = AsyncMock()

        await server.subscribe(cb1)
        await server.subscribe(cb2)

        count = await server.broadcast_agent_chunk("test", "chunk")
        assert count == 2


# ===========================================================================
# Test Convenience Methods
# ===========================================================================


class TestConvenienceMethods:
    """Tests for convenience broadcast methods."""

    @pytest.mark.asyncio
    async def test_broadcast_agent_chunk(self, server):
        """broadcast_agent_chunk creates correct event structure."""
        received = []

        async def callback(event):
            received.append(event)

        await server.subscribe(callback)
        await server.broadcast_agent_chunk("claude", "Hello world", debate_id="d1")

        assert len(received) == 1
        event = received[0]
        assert event.event_type == GatewayEventType.AGENT_RESPONSE_CHUNK
        assert event.agent_name == "claude"
        assert event.data["chunk"] == "Hello world"
        assert event.debate_id == "d1"

    @pytest.mark.asyncio
    async def test_broadcast_capability_used(self, server):
        """broadcast_capability_used creates correct event structure."""
        received = []

        async def callback(event):
            received.append(event)

        await server.subscribe(callback)
        await server.broadcast_capability_used(
            "gpt4", "code_interpreter", {"language": "python"}, debate_id="d1"
        )

        event = received[0]
        assert event.event_type == GatewayEventType.AGENT_CAPABILITY_USED
        assert event.agent_name == "gpt4"
        assert event.data["capability"] == "code_interpreter"
        assert event.data["details"]["language"] == "python"

    @pytest.mark.asyncio
    async def test_broadcast_capability_used_default_details(self, server):
        """broadcast_capability_used uses empty dict for details if None."""
        received = []

        async def callback(event):
            received.append(event)

        await server.subscribe(callback)
        await server.broadcast_capability_used("agent", "tool")

        event = received[0]
        assert event.data["details"] == {}

    @pytest.mark.asyncio
    async def test_broadcast_agent_error(self, server):
        """broadcast_agent_error creates correct event structure."""
        received = []

        async def callback(event):
            received.append(event)

        await server.subscribe(callback)
        await server.broadcast_agent_error(
            "claude", "Connection timeout", error_code="TIMEOUT", debate_id="d1"
        )

        event = received[0]
        assert event.event_type == GatewayEventType.AGENT_ERROR
        assert event.agent_name == "claude"
        assert event.data["error"] == "Connection timeout"
        assert event.data["error_code"] == "TIMEOUT"

    @pytest.mark.asyncio
    async def test_broadcast_agent_complete(self, server):
        """broadcast_agent_complete creates correct event structure."""
        received = []

        async def callback(event):
            received.append(event)

        await server.subscribe(callback)
        result = {"answer": "42", "confidence": 0.95}
        await server.broadcast_agent_complete("gpt4", result, debate_id="d1")

        event = received[0]
        assert event.event_type == GatewayEventType.AGENT_COMPLETE
        assert event.agent_name == "gpt4"
        assert event.data["result"] == result

    @pytest.mark.asyncio
    async def test_broadcast_debate_progress(self, server):
        """broadcast_debate_progress creates correct event structure."""
        received = []

        async def callback(event):
            received.append(event)

        await server.subscribe(callback)
        await server.broadcast_debate_progress(
            debate_id="d1", phase="proposal", progress=0.5, message="Halfway done"
        )

        event = received[0]
        assert event.event_type == GatewayEventType.DEBATE_PROGRESS
        assert event.agent_name == "system"
        assert event.data["phase"] == "proposal"
        assert event.data["progress"] == 0.5
        assert event.data["message"] == "Halfway done"
        assert event.debate_id == "d1"

    @pytest.mark.asyncio
    async def test_broadcast_verification_update(self, server):
        """broadcast_verification_update creates correct event structure."""
        received = []

        async def callback(event):
            received.append(event)

        await server.subscribe(callback)
        await server.broadcast_verification_update(
            debate_id="d1",
            verifier_name="formal_verifier",
            status="completed",
            critique="Logic is sound",
        )

        event = received[0]
        assert event.event_type == GatewayEventType.VERIFICATION_UPDATE
        assert event.agent_name == "formal_verifier"
        assert event.data["status"] == "completed"
        assert event.data["critique"] == "Logic is sound"
        assert event.debate_id == "d1"


# ===========================================================================
# Test Sequence Numbers
# ===========================================================================


class TestSequenceNumbers:
    """Tests for event sequence numbering."""

    @pytest.mark.asyncio
    async def test_sequence_numbers_increment(self, server):
        """Sequence numbers increase with each broadcast."""
        received = []

        async def callback(event):
            received.append(event)

        await server.subscribe(callback)

        await server.broadcast_agent_chunk("test", "chunk1")
        await server.broadcast_agent_chunk("test", "chunk2")
        await server.broadcast_agent_chunk("test", "chunk3")

        assert received[0].sequence == 1
        assert received[1].sequence == 2
        assert received[2].sequence == 3

    @pytest.mark.asyncio
    async def test_sequence_starts_at_one(self, server):
        """First event has sequence 1."""
        received = []

        async def callback(event):
            received.append(event)

        await server.subscribe(callback)
        await server.broadcast_agent_chunk("test", "chunk")

        assert received[0].sequence == 1


# ===========================================================================
# Test Error Handling
# ===========================================================================


class TestErrorHandling:
    """Tests for error handling in broadcasting."""

    @pytest.mark.asyncio
    async def test_subscriber_error_doesnt_break_broadcast(self, server):
        """Errors in one subscriber don't prevent others from receiving events."""
        received = []

        def bad_callback(event):
            raise RuntimeError("Intentional error")

        async def good_callback(event):
            received.append(event)

        await server.subscribe(bad_callback)
        await server.subscribe(good_callback)

        # Should not raise, and good_callback should still receive the event
        count = await server.broadcast_agent_chunk("test", "chunk")

        assert len(received) == 1
        # Count includes both attempts, but one failed
        assert count == 1  # Only successful callbacks counted

    @pytest.mark.asyncio
    async def test_async_subscriber_error_handled(self, server):
        """Errors in async subscribers are handled gracefully."""
        received = []

        async def bad_callback(event):
            raise ValueError("Async error")

        async def good_callback(event):
            received.append(event)

        await server.subscribe(bad_callback)
        await server.subscribe(good_callback)

        # Should not raise
        await server.broadcast_agent_chunk("test", "chunk")
        assert len(received) == 1


# ===========================================================================
# Test Subscriber Count
# ===========================================================================


class TestSubscriberCount:
    """Tests for subscriber counting."""

    @pytest.mark.asyncio
    async def test_get_subscriber_count_global(self, server):
        """get_subscriber_count returns total count when debate_id is None."""
        cb1 = MagicMock()
        cb2 = MagicMock()
        cb3 = MagicMock()

        await server.subscribe(cb1)  # Global
        await server.subscribe(cb2, debate_id="d1")
        await server.subscribe(cb3, debate_id="d2")

        assert server.get_subscriber_count() == 3

    @pytest.mark.asyncio
    async def test_get_subscriber_count_for_debate(self, server):
        """get_subscriber_count for debate includes global + debate-specific."""
        cb_global = MagicMock()
        cb_d1 = MagicMock()
        cb_d2 = MagicMock()

        await server.subscribe(cb_global)  # Global
        await server.subscribe(cb_d1, debate_id="d1")
        await server.subscribe(cb_d2, debate_id="d2")

        # For debate-1: global + d1-specific
        assert server.get_subscriber_count("d1") == 2
        # For debate-2: global + d2-specific
        assert server.get_subscriber_count("d2") == 2
        # For non-existent debate: just global
        assert server.get_subscriber_count("d3") == 1

    def test_get_subscriber_count_empty(self, server):
        """get_subscriber_count returns 0 for empty server."""
        assert server.get_subscriber_count() == 0
        assert server.get_subscriber_count("any-debate") == 0


# ===========================================================================
# Test Global Instance
# ===========================================================================


class TestGlobalInstance:
    """Tests for global server instance management."""

    def test_get_gateway_stream_server_creates_instance(self):
        """get_gateway_stream_server creates a new instance if none exists."""
        server = get_gateway_stream_server()
        assert isinstance(server, GatewayStreamServer)

    def test_get_gateway_stream_server_returns_same_instance(self):
        """get_gateway_stream_server returns the same instance on subsequent calls."""
        server1 = get_gateway_stream_server()
        server2 = get_gateway_stream_server()
        assert server1 is server2

    def test_set_gateway_stream_server(self):
        """set_gateway_stream_server sets the global instance."""
        custom_server = GatewayStreamServer()
        set_gateway_stream_server(custom_server)

        retrieved = get_gateway_stream_server()
        assert retrieved is custom_server


# ===========================================================================
# Test Sync Callbacks
# ===========================================================================


class TestSyncCallbacks:
    """Tests for synchronous callback support."""

    @pytest.mark.asyncio
    async def test_sync_callback_works(self, server):
        """Synchronous callbacks are supported."""
        received = []

        def sync_callback(event):
            received.append(event)

        await server.subscribe(sync_callback)
        await server.broadcast_agent_chunk("test", "chunk")

        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_mixed_sync_async_callbacks(self, server):
        """Both sync and async callbacks work together."""
        sync_received = []
        async_received = []

        def sync_cb(event):
            sync_received.append(event)

        async def async_cb(event):
            async_received.append(event)

        await server.subscribe(sync_cb)
        await server.subscribe(async_cb)

        await server.broadcast_agent_chunk("test", "chunk")

        assert len(sync_received) == 1
        assert len(async_received) == 1
