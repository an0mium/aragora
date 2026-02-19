"""Comprehensive tests for aragora.spectate.ws_bridge module.

Tests SpectateWebSocketBridge including start/stop lifecycle,
event forwarding, subscriber management, buffer limits,
thread safety, and singleton accessor.
"""

from __future__ import annotations

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from aragora.spectate.ws_bridge import (
    SpectateEvent,
    SpectateWebSocketBridge,
    get_spectate_bridge,
    reset_spectate_bridge,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset the global bridge singleton before and after each test."""
    reset_spectate_bridge()
    yield
    reset_spectate_bridge()


@pytest.fixture
def bridge():
    """Create a fresh bridge instance."""
    return SpectateWebSocketBridge(max_buffer=100)


# ---------------------------------------------------------------------------
# SpectateEvent tests
# ---------------------------------------------------------------------------


class TestSpectateEvent:
    """Tests for the SpectateEvent dataclass."""

    def test_basic_creation(self):
        event = SpectateEvent(
            event_type="debate_start",
            timestamp="2026-02-18T10:00:00+00:00",
        )
        assert event.event_type == "debate_start"
        assert event.timestamp == "2026-02-18T10:00:00+00:00"
        assert event.data == {}
        assert event.debate_id is None
        assert event.pipeline_id is None
        assert event.agent_name is None
        assert event.round_number is None

    def test_creation_with_all_fields(self):
        event = SpectateEvent(
            event_type="proposal",
            timestamp="2026-02-18T10:00:00+00:00",
            data={"details": "Rate limiter design"},
            debate_id="d-123",
            pipeline_id="p-456",
            agent_name="claude",
            round_number=2,
        )
        assert event.agent_name == "claude"
        assert event.round_number == 2
        assert event.data["details"] == "Rate limiter design"

    def test_to_dict(self):
        event = SpectateEvent(
            event_type="vote",
            timestamp="2026-02-18T10:00:00+00:00",
            data={"metric": 0.85},
            agent_name="gpt4",
            round_number=1,
        )
        d = event.to_dict()
        assert d["event_type"] == "vote"
        assert d["timestamp"] == "2026-02-18T10:00:00+00:00"
        assert d["data"]["metric"] == 0.85
        assert d["agent_name"] == "gpt4"
        assert d["round_number"] == 1
        assert d["debate_id"] is None
        assert d["pipeline_id"] is None


# ---------------------------------------------------------------------------
# Bridge lifecycle tests
# ---------------------------------------------------------------------------


class TestBridgeLifecycle:
    """Tests for bridge start/stop behavior."""

    def test_initial_state(self, bridge: SpectateWebSocketBridge):
        assert not bridge.running
        assert bridge.subscriber_count == 0
        assert bridge.buffer_size == 0

    def test_start_sets_running(self, bridge: SpectateWebSocketBridge):
        bridge.start()
        assert bridge.running
        bridge.stop()

    def test_stop_clears_running(self, bridge: SpectateWebSocketBridge):
        bridge.start()
        bridge.stop()
        assert not bridge.running

    def test_double_start_is_noop(self, bridge: SpectateWebSocketBridge):
        bridge.start()
        # Second start should not raise
        bridge.start()
        assert bridge.running
        bridge.stop()

    def test_stop_without_start_is_safe(self, bridge: SpectateWebSocketBridge):
        # Should not raise
        bridge.stop()
        assert not bridge.running

    def test_start_restores_emit_on_stop(self, bridge: SpectateWebSocketBridge):
        from aragora.spectate.stream import SpectatorStream

        original_emit = SpectatorStream.emit
        bridge.start()
        assert SpectatorStream.emit is not original_emit
        bridge.stop()
        assert SpectatorStream.emit is original_emit


# ---------------------------------------------------------------------------
# Event forwarding tests
# ---------------------------------------------------------------------------


class TestEventForwarding:
    """Tests for forwarding SpectatorStream events through the bridge."""

    def test_emit_creates_event_in_buffer(self, bridge: SpectateWebSocketBridge):
        bridge.start()
        try:
            from aragora.spectate.stream import SpectatorStream

            stream = SpectatorStream(enabled=True, format="plain")
            stream.emit("debate_start", agent="claude", details="Starting")

            assert bridge.buffer_size == 1
            events = bridge.get_recent_events()
            assert len(events) == 1
            assert events[0].event_type == "debate_start"
            assert events[0].agent_name == "claude"
            assert events[0].data["details"] == "Starting"
        finally:
            bridge.stop()

    def test_emit_with_metric(self, bridge: SpectateWebSocketBridge):
        bridge.start()
        try:
            from aragora.spectate.stream import SpectatorStream

            stream = SpectatorStream(enabled=True, format="plain")
            stream.emit("convergence", metric=0.92, round_number=3)

            events = bridge.get_recent_events()
            assert len(events) == 1
            assert events[0].data["metric"] == 0.92
            assert events[0].round_number == 3
        finally:
            bridge.stop()

    def test_emit_without_optional_fields(self, bridge: SpectateWebSocketBridge):
        bridge.start()
        try:
            from aragora.spectate.stream import SpectatorStream

            stream = SpectatorStream(enabled=True, format="plain")
            stream.emit("system")

            events = bridge.get_recent_events()
            assert len(events) == 1
            assert events[0].agent_name is None
            assert events[0].round_number is None
            assert events[0].data == {}
        finally:
            bridge.stop()

    def test_multiple_events_ordered(self, bridge: SpectateWebSocketBridge):
        bridge.start()
        try:
            from aragora.spectate.stream import SpectatorStream

            stream = SpectatorStream(enabled=True, format="plain")
            stream.emit("debate_start")
            stream.emit("round_start", round_number=1)
            stream.emit("proposal", agent="claude")

            events = bridge.get_recent_events()
            assert len(events) == 3
            assert events[0].event_type == "debate_start"
            assert events[1].event_type == "round_start"
            assert events[2].event_type == "proposal"
        finally:
            bridge.stop()

    def test_disabled_stream_does_not_forward(self, bridge: SpectateWebSocketBridge):
        """SpectatorStream with enabled=False should not emit, so no forwarding."""
        bridge.start()
        try:
            from aragora.spectate.stream import SpectatorStream

            stream = SpectatorStream(enabled=False)
            stream.emit("debate_start")

            assert bridge.buffer_size == 0
        finally:
            bridge.stop()


# ---------------------------------------------------------------------------
# Subscriber tests
# ---------------------------------------------------------------------------


class TestSubscribers:
    """Tests for subscribe/unsubscribe behavior."""

    def test_subscribe_receives_events(self, bridge: SpectateWebSocketBridge):
        received: list[SpectateEvent] = []
        bridge.subscribe(received.append)
        bridge.start()
        try:
            from aragora.spectate.stream import SpectatorStream

            stream = SpectatorStream(enabled=True, format="plain")
            stream.emit("debate_start", agent="claude")

            assert len(received) == 1
            assert received[0].event_type == "debate_start"
        finally:
            bridge.stop()

    def test_multiple_subscribers(self, bridge: SpectateWebSocketBridge):
        received_a: list[SpectateEvent] = []
        received_b: list[SpectateEvent] = []
        bridge.subscribe(received_a.append)
        bridge.subscribe(received_b.append)
        bridge.start()
        try:
            from aragora.spectate.stream import SpectatorStream

            stream = SpectatorStream(enabled=True, format="plain")
            stream.emit("vote")

            assert len(received_a) == 1
            assert len(received_b) == 1
        finally:
            bridge.stop()

    def test_unsubscribe_stops_delivery(self, bridge: SpectateWebSocketBridge):
        received: list[SpectateEvent] = []
        bridge.subscribe(received.append)
        bridge.start()
        try:
            from aragora.spectate.stream import SpectatorStream

            stream = SpectatorStream(enabled=True, format="plain")
            stream.emit("debate_start")
            assert len(received) == 1

            bridge.unsubscribe(received.append)
            stream.emit("debate_end")
            assert len(received) == 1  # No new event
        finally:
            bridge.stop()

    def test_subscriber_count(self, bridge: SpectateWebSocketBridge):
        cb1 = lambda e: None  # noqa: E731
        cb2 = lambda e: None  # noqa: E731
        assert bridge.subscriber_count == 0
        bridge.subscribe(cb1)
        assert bridge.subscriber_count == 1
        bridge.subscribe(cb2)
        assert bridge.subscriber_count == 2
        bridge.unsubscribe(cb1)
        assert bridge.subscriber_count == 1

    def test_subscriber_error_does_not_break_others(self, bridge: SpectateWebSocketBridge):
        """A failing subscriber should not prevent other subscribers from receiving events."""
        received: list[SpectateEvent] = []

        def bad_subscriber(event: SpectateEvent) -> None:
            raise ValueError("Subscriber error")

        bridge.subscribe(bad_subscriber)
        bridge.subscribe(received.append)
        bridge.start()
        try:
            from aragora.spectate.stream import SpectatorStream

            stream = SpectatorStream(enabled=True, format="plain")
            stream.emit("debate_start")

            # The second subscriber should still receive the event
            assert len(received) == 1
        finally:
            bridge.stop()


# ---------------------------------------------------------------------------
# Buffer tests
# ---------------------------------------------------------------------------


class TestBuffer:
    """Tests for event buffer management."""

    def test_buffer_respects_max_size(self):
        bridge = SpectateWebSocketBridge(max_buffer=5)
        bridge.start()
        try:
            from aragora.spectate.stream import SpectatorStream

            stream = SpectatorStream(enabled=True, format="plain")
            for i in range(10):
                stream.emit("system", details=f"event-{i}")

            assert bridge.buffer_size == 5
            events = bridge.get_recent_events()
            # Should keep the most recent 5
            assert events[0].data["details"] == "event-5"
            assert events[4].data["details"] == "event-9"
        finally:
            bridge.stop()

    def test_get_recent_events_with_count(self, bridge: SpectateWebSocketBridge):
        bridge.start()
        try:
            from aragora.spectate.stream import SpectatorStream

            stream = SpectatorStream(enabled=True, format="plain")
            for i in range(10):
                stream.emit("system", details=f"event-{i}")

            events = bridge.get_recent_events(count=3)
            assert len(events) == 3
            assert events[0].data["details"] == "event-7"
            assert events[2].data["details"] == "event-9"
        finally:
            bridge.stop()

    def test_get_recent_events_empty_buffer(self, bridge: SpectateWebSocketBridge):
        events = bridge.get_recent_events()
        assert events == []

    def test_clear_buffer(self, bridge: SpectateWebSocketBridge):
        bridge.start()
        try:
            from aragora.spectate.stream import SpectatorStream

            stream = SpectatorStream(enabled=True, format="plain")
            stream.emit("debate_start")
            assert bridge.buffer_size == 1

            bridge.clear_buffer()
            assert bridge.buffer_size == 0
            assert bridge.get_recent_events() == []
        finally:
            bridge.stop()


# ---------------------------------------------------------------------------
# Thread safety tests
# ---------------------------------------------------------------------------


class TestThreadSafety:
    """Tests for concurrent access to the bridge."""

    def test_concurrent_emit_and_subscribe(self, bridge: SpectateWebSocketBridge):
        """Multiple threads emitting events concurrently should not cause errors."""
        bridge.start()
        received: list[SpectateEvent] = []
        bridge.subscribe(received.append)

        errors: list[Exception] = []

        def emit_events(stream, count: int) -> None:
            try:
                for i in range(count):
                    stream.emit("system", details=f"thread-event-{i}")
            except Exception as e:
                errors.append(e)

        try:
            from aragora.spectate.stream import SpectatorStream

            stream = SpectatorStream(enabled=True, format="plain")

            threads = [
                threading.Thread(target=emit_events, args=(stream, 20))
                for _ in range(5)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=5)

            assert not errors
            # 5 threads * 20 events = 100 events total
            assert bridge.buffer_size == 100
            assert len(received) == 100
        finally:
            bridge.stop()


# ---------------------------------------------------------------------------
# Singleton tests
# ---------------------------------------------------------------------------


class TestSingleton:
    """Tests for the singleton accessor."""

    def test_get_spectate_bridge_returns_same_instance(self):
        b1 = get_spectate_bridge()
        b2 = get_spectate_bridge()
        assert b1 is b2

    def test_reset_creates_new_instance(self):
        b1 = get_spectate_bridge()
        reset_spectate_bridge()
        b2 = get_spectate_bridge()
        assert b1 is not b2

    def test_reset_stops_running_bridge(self):
        b = get_spectate_bridge()
        b.start()
        assert b.running
        reset_spectate_bridge()
        assert not b.running
