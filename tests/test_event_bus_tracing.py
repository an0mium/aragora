"""
Tests for correlation ID support in aragora.debate.event_bus.

Covers:
- DebateEvent correlation ID auto-population
- Correlation ID propagation through emit()
- Serialization with correlation IDs
"""

from unittest.mock import MagicMock, patch

import pytest

from aragora.debate.event_bus import DebateEvent, EventBus


class TestDebateEventCorrelationId:
    """Tests for DebateEvent correlation ID support."""

    def test_event_auto_populates_correlation_id_from_trace_context(self):
        """Event should auto-populate correlation_id from trace context."""
        with patch("aragora.debate.event_bus.get_trace_id", return_value="trace-123"):
            with patch("aragora.debate.event_bus.get_span_id", return_value="span-456"):
                event = DebateEvent(
                    event_type="debate_start",
                    debate_id="debate-001",
                )

                assert event.correlation_id == "trace-123"
                assert event.span_id == "span-456"

    def test_event_uses_explicit_correlation_id(self):
        """Event should use explicitly provided correlation_id."""
        with patch("aragora.debate.event_bus.get_trace_id", return_value="trace-from-context"):
            event = DebateEvent(
                event_type="debate_start",
                debate_id="debate-001",
                correlation_id="explicit-correlation-id",
            )

            assert event.correlation_id == "explicit-correlation-id"

    def test_event_handles_no_trace_context(self):
        """Event should handle missing trace context gracefully."""
        with patch("aragora.debate.event_bus.get_trace_id", return_value=None):
            with patch("aragora.debate.event_bus.get_span_id", return_value=None):
                event = DebateEvent(
                    event_type="debate_start",
                    debate_id="debate-001",
                )

                assert event.correlation_id is None
                assert event.span_id is None

    def test_to_dict_includes_correlation_id(self):
        """to_dict should include correlation_id when present."""
        event = DebateEvent(
            event_type="round_start",
            debate_id="debate-001",
            correlation_id="corr-123",
            span_id="span-456",
            data={"round_number": 1},
        )

        result = event.to_dict()

        assert result["correlation_id"] == "corr-123"
        assert result["span_id"] == "span-456"
        assert result["event_type"] == "round_start"
        assert result["debate_id"] == "debate-001"
        assert result["round_number"] == 1

    def test_to_dict_excludes_none_correlation_id(self):
        """to_dict should exclude correlation_id when None."""
        event = DebateEvent(
            event_type="round_start",
            debate_id="debate-001",
            correlation_id=None,
            span_id=None,
        )

        result = event.to_dict()

        assert "correlation_id" not in result
        assert "span_id" not in result


class TestEventBusCorrelationId:
    """Tests for EventBus correlation ID propagation."""

    @pytest.fixture
    def event_bus(self):
        """Create an EventBus for testing."""
        return EventBus()

    @pytest.mark.asyncio
    async def test_emit_with_explicit_correlation_id(self, event_bus):
        """emit should use explicit correlation_id."""
        received_events = []

        async def handler(event):
            received_events.append(event)

        event_bus.subscribe("test_event", handler)

        await event_bus.emit(
            "test_event",
            debate_id="debate-001",
            correlation_id="explicit-corr-id",
            some_data="value",
        )

        assert len(received_events) == 1
        assert received_events[0].correlation_id == "explicit-corr-id"

    @pytest.mark.asyncio
    async def test_emit_auto_populates_correlation_id(self, event_bus):
        """emit should auto-populate correlation_id from trace context."""
        received_events = []

        async def handler(event):
            received_events.append(event)

        event_bus.subscribe("test_event", handler)

        with patch("aragora.debate.event_bus.get_trace_id", return_value="auto-trace-123"):
            await event_bus.emit(
                "test_event",
                debate_id="debate-001",
            )

        assert len(received_events) == 1
        assert received_events[0].correlation_id == "auto-trace-123"

    def test_emit_sync_with_correlation_id(self, event_bus):
        """emit_sync should support correlation_id."""
        received_events = []

        def handler(event):
            received_events.append(event)

        event_bus.subscribe_sync("test_event", handler)

        event_bus.emit_sync(
            "test_event",
            debate_id="debate-001",
            correlation_id="sync-corr-id",
        )

        assert len(received_events) == 1
        assert received_events[0].correlation_id == "sync-corr-id"

    @pytest.mark.asyncio
    async def test_event_bridge_receives_correlation_id(self, event_bus):
        """Event bridge should receive correlation_id in notification."""
        event_bridge = MagicMock()
        event_bus._event_bridge = event_bridge

        await event_bus.emit(
            "debate_start",
            debate_id="debate-001",
            correlation_id="bridge-corr-id",
            task="Test task",
        )

        event_bridge.notify.assert_called_once()
        call_kwargs = event_bridge.notify.call_args[1]
        assert call_kwargs["correlation_id"] == "bridge-corr-id"


class TestCorrelationIdPropagation:
    """Tests for correlation ID propagation through specialized event methods."""

    @pytest.fixture
    def event_bus(self):
        """Create an EventBus for testing."""
        return EventBus()

    @pytest.mark.asyncio
    async def test_notify_spectator_propagates_correlation_id(self, event_bus):
        """notify_spectator should propagate correlation_id."""
        received_events = []

        async def handler(event):
            received_events.append(event)

        event_bus.subscribe("agent_message", handler)

        with patch("aragora.debate.event_bus.get_trace_id", return_value="spectator-trace"):
            await event_bus.notify_spectator(
                "agent_message",
                debate_id="debate-001",
                agent="agent1",
                message="Hello",
            )

        assert len(received_events) == 1
        assert received_events[0].correlation_id == "spectator-trace"

    @pytest.mark.asyncio
    async def test_emit_moment_event_includes_correlation_id(self, event_bus):
        """emit_moment_event should include correlation_id."""
        received_events = []

        async def handler(event):
            received_events.append(event)

        event_bus.subscribe("moment", handler)

        with patch("aragora.debate.event_bus.get_trace_id", return_value="moment-trace"):
            await event_bus.emit_moment_event(
                debate_id="debate-001",
                moment_type="breakthrough",
                description="Key insight discovered",
                agent="agent1",
                significance=0.9,
            )

        assert len(received_events) == 1
        event = received_events[0]
        assert event.correlation_id == "moment-trace"
        assert event.data["moment_type"] == "breakthrough"

    @pytest.mark.asyncio
    async def test_broadcast_health_event_includes_correlation_id(self, event_bus):
        """broadcast_health_event should include correlation_id."""
        received_events = []

        async def handler(event):
            received_events.append(event)

        # Set up immune system to enable health events
        event_bus._immune_system = MagicMock()
        event_bus.subscribe("health_update", handler)

        with patch("aragora.debate.event_bus.get_trace_id", return_value="health-trace"):
            await event_bus.broadcast_health_event(
                debate_id="debate-001",
                health_status={"status": "healthy", "score": 0.95},
            )

        assert len(received_events) == 1
        assert received_events[0].correlation_id == "health-trace"
