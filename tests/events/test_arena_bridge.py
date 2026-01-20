"""Tests for ArenaEventBridge integration."""

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from aragora.events.arena_bridge import (
    ArenaEventBridge,
    create_arena_bridge,
    EVENT_TYPE_MAP,
)
from aragora.events.cross_subscribers import (
    CrossSubscriberManager,
    reset_cross_subscriber_manager,
)
from aragora.events.types import StreamEvent, StreamEventType
from aragora.debate.event_bus import DebateEvent, EventBus


class TestArenaEventBridge:
    """Test ArenaEventBridge functionality."""

    def setup_method(self):
        """Reset global state before each test."""
        reset_cross_subscriber_manager()

    def test_bridge_connects_to_event_bus(self):
        """Test bridge subscribes to EventBus events."""
        event_bus = EventBus()
        cross_manager = CrossSubscriberManager()

        bridge = ArenaEventBridge(event_bus, cross_manager)
        bridge.connect_to_cross_subscribers()

        assert bridge.is_connected
        # Check that sync handlers were registered
        assert len(event_bus._sync_handlers) > 0

    def test_bridge_prevents_double_connection(self):
        """Test bridge doesn't double-subscribe."""
        event_bus = EventBus()
        cross_manager = CrossSubscriberManager()

        bridge = ArenaEventBridge(event_bus, cross_manager)
        bridge.connect_to_cross_subscribers()
        initial_handlers = sum(len(h) for h in event_bus._sync_handlers.values())

        bridge.connect_to_cross_subscribers()  # Second call
        final_handlers = sum(len(h) for h in event_bus._sync_handlers.values())

        assert initial_handlers == final_handlers

    def test_debate_event_conversion(self):
        """Test DebateEvent is converted to StreamEvent correctly."""
        event_bus = EventBus()
        cross_manager = CrossSubscriberManager()
        received_events = []

        def capture_handler(event):
            received_events.append(event)

        cross_manager.register(
            "test_capture",
            StreamEventType.DEBATE_START,
            capture_handler,
        )

        bridge = ArenaEventBridge(event_bus, cross_manager)
        bridge.connect_to_cross_subscribers()

        # Emit a debate event through EventBus
        event_bus.emit_sync("debate_start", debate_id="test-123", task="Test task")

        assert len(received_events) == 1
        assert received_events[0].type == StreamEventType.DEBATE_START
        assert received_events[0].data["debate_id"] == "test-123"

    def test_memory_event_bridging(self):
        """Test memory events are bridged correctly."""
        event_bus = EventBus()
        cross_manager = CrossSubscriberManager()
        received_events = []

        def capture_handler(event):
            received_events.append(event)

        cross_manager.register(
            "test_capture",
            StreamEventType.MEMORY_STORED,
            capture_handler,
        )

        bridge = ArenaEventBridge(event_bus, cross_manager)
        bridge.connect_to_cross_subscribers()

        # Emit memory stored event
        event_bus.emit_sync(
            "memory_stored",
            debate_id="debate-001",
            tier="medium",
            importance=0.8,
        )

        assert len(received_events) == 1
        assert received_events[0].type == StreamEventType.MEMORY_STORED
        assert received_events[0].data["tier"] == "medium"

    def test_elo_event_triggers_handler(self):
        """Test ELO events trigger built-in handler."""
        event_bus = EventBus()
        cross_manager = CrossSubscriberManager()

        bridge = ArenaEventBridge(event_bus, cross_manager)
        bridge.connect_to_cross_subscribers()

        # Emit ELO update event
        event_bus.emit_sync(
            "agent_elo_updated",
            debate_id="debate-001",
            agent="claude",
            elo=1650,
            delta=100,
        )

        # Check built-in handler was called
        stats = cross_manager.get_stats()
        assert stats["elo_to_debate"]["events_processed"] == 1

    def test_knowledge_event_bridging(self):
        """Test knowledge events are bridged."""
        event_bus = EventBus()
        cross_manager = CrossSubscriberManager()
        received_events = []

        def capture_handler(event):
            received_events.append(event)

        cross_manager.register(
            "test_capture",
            StreamEventType.KNOWLEDGE_INDEXED,
            capture_handler,
        )

        bridge = ArenaEventBridge(event_bus, cross_manager)
        bridge.connect_to_cross_subscribers()

        event_bus.emit_sync(
            "knowledge_indexed",
            debate_id="debate-001",
            node_id="kn_001",
            content="Test content",
            node_type="fact",
        )

        assert len(received_events) == 1
        assert received_events[0].data["node_id"] == "kn_001"

    def test_unmapped_events_ignored(self):
        """Test unmapped event types are silently ignored."""
        event_bus = EventBus()
        cross_manager = CrossSubscriberManager()

        bridge = ArenaEventBridge(event_bus, cross_manager)
        bridge.connect_to_cross_subscribers()

        # Emit an unmapped event type
        event_bus.emit_sync(
            "some_unknown_event",
            debate_id="test",
            data="test",
        )

        # Should not raise, should not crash
        # Stats should remain unchanged
        stats = cross_manager.get_stats()
        total_processed = sum(s["events_processed"] for s in stats.values())
        assert total_processed == 0

    def test_create_arena_bridge_factory(self):
        """Test factory function creates connected bridge."""
        event_bus = EventBus()

        bridge = create_arena_bridge(event_bus)

        assert bridge.is_connected

    def test_disconnect(self):
        """Test bridge can be disconnected."""
        event_bus = EventBus()

        bridge = ArenaEventBridge(event_bus)
        bridge.connect_to_cross_subscribers()
        assert bridge.is_connected

        bridge.disconnect()
        assert not bridge.is_connected


class TestEventTypeMapping:
    """Test event type mappings are correct."""

    def test_all_mapped_types_exist(self):
        """Test all mapped types are valid StreamEventTypes."""
        for event_type_str, stream_type in EVENT_TYPE_MAP.items():
            assert isinstance(stream_type, StreamEventType)
            assert stream_type.value  # Has a value

    def test_key_event_types_mapped(self):
        """Test critical event types are mapped."""
        assert "debate_start" in EVENT_TYPE_MAP
        assert "debate_end" in EVENT_TYPE_MAP
        assert "memory_stored" in EVENT_TYPE_MAP
        assert "agent_elo_updated" in EVENT_TYPE_MAP
        assert "knowledge_indexed" in EVENT_TYPE_MAP


class TestBridgeErrorHandling:
    """Test error handling in the bridge."""

    def setup_method(self):
        """Reset global state before each test."""
        reset_cross_subscriber_manager()

    def test_handler_error_doesnt_break_bridge(self):
        """Test error in one handler doesn't stop others."""
        event_bus = EventBus()
        cross_manager = CrossSubscriberManager()
        results = []

        def good_handler(event):
            results.append("good")

        def bad_handler(event):
            raise ValueError("Intentional error")

        cross_manager.register("good", StreamEventType.DEBATE_START, good_handler)
        cross_manager.register("bad", StreamEventType.DEBATE_START, bad_handler)

        bridge = ArenaEventBridge(event_bus, cross_manager)
        bridge.connect_to_cross_subscribers()

        # This should not raise
        event_bus.emit_sync("debate_start", debate_id="test")

        # Good handler should still have been called
        assert "good" in results

        # Bad handler's error should be tracked
        stats = cross_manager.get_stats()
        assert stats["bad"]["events_failed"] == 1
