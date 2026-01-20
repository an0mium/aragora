"""Integration tests for cross-subsystem event flow.

Tests verify that:
- Subsystems emit events correctly
- CrossSubscriberManager receives and dispatches events
- Handlers are called with correct event data
- Event statistics are tracked
"""

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime

from aragora.events.types import StreamEvent, StreamEventType
from aragora.events.cross_subscribers import (
    CrossSubscriberManager,
    get_cross_subscriber_manager,
    reset_cross_subscriber_manager,
)


class TestEventFlowIntegration:
    """Test event flow from subsystems to cross-subscribers."""

    def setup_method(self):
        """Reset global state before each test."""
        reset_cross_subscriber_manager()

    def test_memory_event_flow(self):
        """Test MEMORY_STORED event triggers cross-subscriber handler."""
        manager = CrossSubscriberManager()
        received_events = []

        def capture_handler(event):
            received_events.append(event)

        manager.register("test_capture", StreamEventType.MEMORY_STORED, capture_handler)

        # Simulate memory store event
        event = StreamEvent(
            type=StreamEventType.MEMORY_STORED,
            data={
                "memory_id": "mem_001",
                "tier": "medium",
                "importance": 0.8,
                "content_length": 500,
            },
        )
        manager._dispatch_event(event)

        assert len(received_events) == 1
        assert received_events[0].data["memory_id"] == "mem_001"

    def test_elo_event_flow(self):
        """Test AGENT_ELO_UPDATED event triggers cross-subscriber handler."""
        manager = CrossSubscriberManager()
        received_events = []

        def capture_handler(event):
            received_events.append(event)

        manager.register("test_capture", StreamEventType.AGENT_ELO_UPDATED, capture_handler)

        # Simulate ELO update event
        event = StreamEvent(
            type=StreamEventType.AGENT_ELO_UPDATED,
            data={
                "agent": "claude",
                "elo": 1650,
                "delta": 75,
                "debate_id": "debate_001",
            },
        )
        manager._dispatch_event(event)

        assert len(received_events) == 1
        assert received_events[0].data["agent"] == "claude"
        assert received_events[0].data["elo"] == 1650

    def test_knowledge_event_flow(self):
        """Test KNOWLEDGE_INDEXED event triggers cross-subscriber handler."""
        manager = CrossSubscriberManager()
        received_events = []

        def capture_handler(event):
            received_events.append(event)

        manager.register("test_capture", StreamEventType.KNOWLEDGE_INDEXED, capture_handler)

        # Simulate knowledge indexed event
        event = StreamEvent(
            type=StreamEventType.KNOWLEDGE_INDEXED,
            data={
                "node_id": "kn_001",
                "content": "Test knowledge content",
                "node_type": "fact",
                "workspace_id": "workspace_001",
            },
        )
        manager._dispatch_event(event)

        assert len(received_events) == 1
        assert received_events[0].data["node_id"] == "kn_001"

    def test_builtin_handlers_receive_events(self):
        """Test built-in handlers process events and update stats."""
        manager = CrossSubscriberManager()

        # Dispatch events that trigger built-in handlers
        events = [
            StreamEvent(
                type=StreamEventType.MEMORY_RETRIEVED,
                data={"tier": "fast", "cache_hit": True},
            ),
            StreamEvent(
                type=StreamEventType.AGENT_ELO_UPDATED,
                data={"agent": "gpt", "elo": 1550, "delta": -25},
            ),
            StreamEvent(
                type=StreamEventType.KNOWLEDGE_INDEXED,
                data={"node_id": "kn_002", "content": "Test", "node_type": "claim"},
            ),
        ]

        for event in events:
            manager._dispatch_event(event)

        stats = manager.get_stats()

        # Check that built-in handlers processed events
        assert stats["memory_to_rlm"]["events_processed"] == 1
        assert stats["elo_to_debate"]["events_processed"] == 1
        assert stats["knowledge_to_memory"]["events_processed"] == 1

    def test_event_emitter_connection(self):
        """Test manager connects to and receives from event emitter."""
        manager = CrossSubscriberManager()
        received_events = []

        def capture_handler(event):
            received_events.append(event)

        manager.register("test_capture", StreamEventType.DEBATE_START, capture_handler)

        # Create mock emitter
        callback = None

        def mock_subscribe(cb):
            nonlocal callback
            callback = cb

        mock_emitter = MagicMock()
        mock_emitter.subscribe = mock_subscribe

        # Connect manager to emitter
        manager.connect(mock_emitter)
        assert manager._connected is True

        # Simulate emitter emitting event
        event = StreamEvent(
            type=StreamEventType.DEBATE_START,
            data={"debate_id": "debate_001"},
        )
        callback(event)

        assert len(received_events) == 1
        assert received_events[0].data["debate_id"] == "debate_001"


class TestMultipleHandlers:
    """Test multiple handlers for same event type."""

    def setup_method(self):
        reset_cross_subscriber_manager()

    def test_multiple_handlers_all_called(self):
        """Test all handlers for an event type are called."""
        manager = CrossSubscriberManager()
        results = []

        def handler_a(event):
            results.append("a")

        def handler_b(event):
            results.append("b")

        def handler_c(event):
            results.append("c")

        manager.register("handler_a", StreamEventType.DEBATE_END, handler_a)
        manager.register("handler_b", StreamEventType.DEBATE_END, handler_b)
        manager.register("handler_c", StreamEventType.DEBATE_END, handler_c)

        event = StreamEvent(
            type=StreamEventType.DEBATE_END,
            data={"debate_id": "test"},
        )
        manager._dispatch_event(event)

        assert "a" in results
        assert "b" in results
        assert "c" in results


class TestErrorHandling:
    """Test error handling in event dispatch."""

    def setup_method(self):
        reset_cross_subscriber_manager()

    def test_error_in_handler_doesnt_stop_others(self):
        """Test that error in one handler doesn't prevent others from running."""
        manager = CrossSubscriberManager()
        results = []

        def handler_good1(event):
            results.append("good1")

        def handler_bad(event):
            raise ValueError("Intentional error")

        def handler_good2(event):
            results.append("good2")

        manager.register("good1", StreamEventType.DEBATE_END, handler_good1)
        manager.register("bad", StreamEventType.DEBATE_END, handler_bad)
        manager.register("good2", StreamEventType.DEBATE_END, handler_good2)

        event = StreamEvent(type=StreamEventType.DEBATE_END, data={})
        manager._dispatch_event(event)

        assert "good1" in results
        assert "good2" in results

        # Check error was tracked
        stats = manager.get_stats()
        assert stats["bad"]["events_failed"] == 1


class TestSingletonBehavior:
    """Test singleton getter behavior."""

    def setup_method(self):
        reset_cross_subscriber_manager()

    def test_get_returns_singleton(self):
        """Test get_cross_subscriber_manager returns same instance."""
        manager1 = get_cross_subscriber_manager()
        manager2 = get_cross_subscriber_manager()
        assert manager1 is manager2

    def test_reset_creates_new_instance(self):
        """Test reset creates fresh instance."""
        manager1 = get_cross_subscriber_manager()
        reset_cross_subscriber_manager()
        manager2 = get_cross_subscriber_manager()
        assert manager1 is not manager2
