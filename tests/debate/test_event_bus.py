"""
Tests for aragora/debate/event_bus.py

Covers:
- DebateEvent dataclass (tracing, to_dict)
- EventBus subscription/unsubscription (async and sync)
- Event emission (async and sync)
- Event bridge integration
- Spectator integration
- Handler error isolation
- User event queue (thread-safe)
- Metrics tracking
- Cleanup and context manager
- Singleton pattern
"""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from aragora.debate.event_bus import (
    DebateEvent,
    EventBus,
    get_event_bus,
    set_event_bus,
)


# ============================================================================
# DebateEvent Tests
# ============================================================================


class TestDebateEvent:
    """Test DebateEvent dataclass."""

    @patch("aragora.debate.event_bus.get_trace_id", return_value="trace-123")
    @patch("aragora.debate.event_bus.get_span_id", return_value="span-456")
    def test_auto_populate_trace_ids(self, mock_span_id, mock_trace_id):
        """Test correlation_id and span_id auto-populated from trace context."""
        event = DebateEvent(
            event_type="debate_start",
            debate_id="debate-1",
        )
        assert event.correlation_id == "trace-123"
        assert event.span_id == "span-456"
        assert event.event_type == "debate_start"
        assert event.debate_id == "debate-1"
        assert isinstance(event.timestamp, datetime)
        assert event.data == {}

    def test_explicit_trace_ids(self):
        """Test explicit correlation_id and span_id override auto-population."""
        event = DebateEvent(
            event_type="round_start",
            debate_id="debate-2",
            correlation_id="custom-trace",
            span_id="custom-span",
        )
        assert event.correlation_id == "custom-trace"
        assert event.span_id == "custom-span"

    def test_to_dict_with_trace_ids(self):
        """Test to_dict includes trace IDs when present."""
        event = DebateEvent(
            event_type="vote",
            debate_id="debate-3",
            correlation_id="trace-789",
            span_id="span-012",
            data={"agent": "claude", "position": "support"},
        )
        result = event.to_dict()
        assert result["event_type"] == "vote"
        assert result["debate_id"] == "debate-3"
        assert result["correlation_id"] == "trace-789"
        assert result["span_id"] == "span-012"
        assert result["agent"] == "claude"
        assert result["position"] == "support"
        assert "timestamp" in result
        assert isinstance(result["timestamp"], str)  # ISO format

    def test_to_dict_without_trace_ids(self):
        """Test to_dict omits None trace IDs."""
        with (
            patch("aragora.debate.event_bus.get_trace_id", return_value=None),
            patch("aragora.debate.event_bus.get_span_id", return_value=None),
        ):
            event = DebateEvent(
                event_type="consensus",
                debate_id="debate-4",
                data={"outcome": "resolved"},
            )
            result = event.to_dict()
            assert "correlation_id" not in result
            assert "span_id" not in result
            assert result["outcome"] == "resolved"

    def test_custom_timestamp(self):
        """Test custom timestamp is preserved."""
        custom_time = datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
        event = DebateEvent(
            event_type="test",
            debate_id="debate-5",
            timestamp=custom_time,
        )
        assert event.timestamp == custom_time
        result = event.to_dict()
        assert result["timestamp"] == "2025-01-15T12:00:00+00:00"

    def test_empty_data_dict(self):
        """Test event with no additional data."""
        event = DebateEvent(
            event_type="ping",
            debate_id="debate-6",
        )
        result = event.to_dict()
        assert result["event_type"] == "ping"
        assert result["debate_id"] == "debate-6"
        # Only event_type, debate_id, timestamp, and optional trace IDs


# ============================================================================
# EventBus Subscription Tests
# ============================================================================


class TestEventBusSubscription:
    """Test subscription management."""

    def test_subscribe_async_handler(self):
        """Test subscribing async handler."""
        bus = EventBus()
        handler = AsyncMock()
        bus.subscribe("test_event", handler)
        assert "test_event" in bus._async_handlers
        assert handler in bus._async_handlers["test_event"]

    def test_subscribe_multiple_handlers(self):
        """Test multiple handlers for same event."""
        bus = EventBus()
        handler1 = AsyncMock()
        handler2 = AsyncMock()
        bus.subscribe("test_event", handler1)
        bus.subscribe("test_event", handler2)
        assert len(bus._async_handlers["test_event"]) == 2

    def test_subscribe_sync_handler(self):
        """Test subscribing sync handler."""
        bus = EventBus()
        handler = Mock()
        bus.subscribe_sync("test_event", handler)
        assert "test_event" in bus._sync_handlers
        assert handler in bus._sync_handlers["test_event"]

    def test_unsubscribe_async_handler(self):
        """Test unsubscribing async handler."""
        bus = EventBus()
        handler = AsyncMock()
        bus.subscribe("test_event", handler)
        result = bus.unsubscribe("test_event", handler)
        assert result is True
        assert handler not in bus._async_handlers.get("test_event", [])

    def test_unsubscribe_nonexistent_handler(self):
        """Test unsubscribing handler that was never added."""
        bus = EventBus()
        handler = AsyncMock()
        result = bus.unsubscribe("test_event", handler)
        assert result is False

    def test_unsubscribe_sync_handler(self):
        """Test unsubscribing sync handler."""
        bus = EventBus()
        handler = Mock()
        bus.subscribe_sync("test_event", handler)
        result = bus.unsubscribe_sync("test_event", handler)
        assert result is True
        assert handler not in bus._sync_handlers.get("test_event", [])

    def test_clear_handlers_specific_event(self):
        """Test clearing handlers for specific event type."""
        bus = EventBus()
        handler1 = AsyncMock()
        handler2 = Mock()
        bus.subscribe("event_a", handler1)
        bus.subscribe_sync("event_a", handler2)
        bus.subscribe("event_b", AsyncMock())
        removed = bus.clear_handlers("event_a")
        assert removed == 2
        assert "event_a" not in bus._async_handlers
        assert "event_a" not in bus._sync_handlers
        assert "event_b" in bus._async_handlers

    def test_clear_all_handlers(self):
        """Test clearing all handlers."""
        bus = EventBus()
        bus.subscribe("event_a", AsyncMock())
        bus.subscribe("event_b", AsyncMock())
        bus.subscribe_sync("event_c", Mock())
        removed = bus.clear_handlers()
        assert removed == 3
        assert len(bus._async_handlers) == 0
        assert len(bus._sync_handlers) == 0


# ============================================================================
# Event Emission Tests
# ============================================================================


class TestEventEmission:
    """Test event emission."""

    @pytest.mark.asyncio
    async def test_emit_calls_async_handlers(self):
        """Test emit calls all async handlers."""
        bus = EventBus()
        handler1 = AsyncMock()
        handler2 = AsyncMock()
        bus.subscribe("test_event", handler1)
        bus.subscribe("test_event", handler2)
        await bus.emit("test_event", debate_id="debate-1", key="value")
        handler1.assert_called_once()
        handler2.assert_called_once()
        event = handler1.call_args[0][0]
        assert event.event_type == "test_event"
        assert event.debate_id == "debate-1"
        assert event.data["key"] == "value"

    @pytest.mark.asyncio
    async def test_emit_calls_sync_handlers(self):
        """Test emit calls sync handlers."""
        bus = EventBus()
        handler = Mock()
        bus.subscribe_sync("test_event", handler)
        await bus.emit("test_event", debate_id="debate-2")
        handler.assert_called_once()
        event = handler.call_args[0][0]
        assert event.event_type == "test_event"
        assert event.debate_id == "debate-2"

    @pytest.mark.asyncio
    async def test_emit_updates_metrics(self):
        """Test emit updates event metrics."""
        bus = EventBus()
        await bus.emit("event_a", debate_id="debate-1")
        await bus.emit("event_a", debate_id="debate-2")
        await bus.emit("event_b", debate_id="debate-3")
        metrics = bus.get_metrics()
        assert metrics["total_events_emitted"] == 3
        assert metrics["events_by_type"]["event_a"] == 2
        assert metrics["events_by_type"]["event_b"] == 1

    @pytest.mark.asyncio
    async def test_emit_with_event_bridge(self):
        """Test emit notifies event bridge."""
        event_bridge = Mock()
        bus = EventBus(event_bridge=event_bridge)
        await bus.emit("test_event", debate_id="debate-1", key="value")
        event_bridge.notify.assert_called_once()
        args = event_bridge.notify.call_args
        assert args[0][0] == "test_event"
        assert "debate_id" in args[1]
        assert "key" in args[1]

    @pytest.mark.asyncio
    async def test_emit_with_spectator(self):
        """Test emit notifies spectator."""
        spectator = Mock()
        bus = EventBus(spectator=spectator)
        await bus.emit("test_event", debate_id="debate-1")
        spectator.emit.assert_called_once()
        args = spectator.emit.call_args
        assert args[0][0] == "test_event"
        assert isinstance(args[0][1], dict)

    @pytest.mark.asyncio
    async def test_emit_handler_error_isolation(self):
        """Test handler errors don't affect other handlers."""
        bus = EventBus()
        handler1 = AsyncMock(side_effect=ValueError("Handler 1 failed"))
        handler2 = AsyncMock()
        handler3 = Mock(side_effect=RuntimeError("Handler 3 failed"))
        bus.subscribe("test_event", handler1)
        bus.subscribe("test_event", handler2)
        bus.subscribe_sync("test_event", handler3)
        # Should not raise despite handler failures
        await bus.emit("test_event", debate_id="debate-1")
        handler1.assert_called_once()
        handler2.assert_called_once()
        handler3.assert_called_once()

    @pytest.mark.asyncio
    async def test_emit_event_bridge_failure_isolated(self):
        """Test event bridge failure doesn't stop emission."""
        event_bridge = Mock()
        event_bridge.notify.side_effect = RuntimeError("Bridge down")
        bus = EventBus(event_bridge=event_bridge)
        handler = AsyncMock()
        bus.subscribe("test_event", handler)
        # Should not raise
        await bus.emit("test_event", debate_id="debate-1")
        handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_emit_spectator_failure_isolated(self):
        """Test spectator failure doesn't stop emission."""
        spectator = Mock()
        spectator.emit.side_effect = OSError("Network error")
        bus = EventBus(spectator=spectator)
        handler = AsyncMock()
        bus.subscribe("test_event", handler)
        # Should not raise
        await bus.emit("test_event", debate_id="debate-1")
        handler.assert_called_once()

    def test_emit_sync_only_sync_handlers(self):
        """Test emit_sync only calls sync handlers."""
        bus = EventBus()
        async_handler = AsyncMock()
        sync_handler = Mock()
        bus.subscribe("test_event", async_handler)
        bus.subscribe_sync("test_event", sync_handler)
        bus.emit_sync("test_event", debate_id="debate-1")
        sync_handler.assert_called_once()
        async_handler.assert_not_called()

    def test_emit_sync_updates_metrics(self):
        """Test emit_sync updates metrics."""
        bus = EventBus()
        bus.emit_sync("event_a", debate_id="debate-1")
        bus.emit_sync("event_a", debate_id="debate-2")
        metrics = bus.get_metrics()
        assert metrics["total_events_emitted"] == 2
        assert metrics["events_by_type"]["event_a"] == 2

    def test_emit_sync_with_event_bridge(self):
        """Test emit_sync notifies event bridge."""
        event_bridge = Mock()
        bus = EventBus(event_bridge=event_bridge)
        bus.emit_sync("test_event", debate_id="debate-1", key="value")
        event_bridge.notify.assert_called_once()

    def test_emit_sync_handler_error_isolation(self):
        """Test emit_sync isolates handler errors."""
        bus = EventBus()
        handler1 = Mock(side_effect=TypeError("Handler 1 failed"))
        handler2 = Mock()
        bus.subscribe_sync("test_event", handler1)
        bus.subscribe_sync("test_event", handler2)
        # Should not raise
        bus.emit_sync("test_event", debate_id="debate-1")
        handler1.assert_called_once()
        handler2.assert_called_once()


# ============================================================================
# Specialized Event Methods Tests
# ============================================================================


class TestSpecializedMethods:
    """Test specialized event methods."""

    @pytest.mark.asyncio
    async def test_notify_spectator(self):
        """Test notify_spectator delegates to emit."""
        bus = EventBus()
        handler = AsyncMock()
        bus.subscribe("test_event", handler)
        await bus.notify_spectator("test_event", debate_id="debate-1", key="value")
        handler.assert_called_once()
        event = handler.call_args[0][0]
        assert event.event_type == "test_event"
        assert event.data["key"] == "value"

    @pytest.mark.asyncio
    async def test_emit_moment_event(self):
        """Test emit_moment_event emits with correct structure."""
        bus = EventBus()
        handler = AsyncMock()
        bus.subscribe("moment", handler)
        await bus.emit_moment_event(
            debate_id="debate-1",
            moment_type="breakthrough",
            description="Significant insight",
            agent="claude",
            round_num=3,
            significance=0.9,
            extra_key="extra_value",
        )
        handler.assert_called_once()
        event = handler.call_args[0][0]
        assert event.event_type == "moment"
        assert event.data["moment_type"] == "breakthrough"
        assert event.data["description"] == "Significant insight"
        assert event.data["agent"] == "claude"
        assert event.data["round_num"] == 3
        assert event.data["significance"] == 0.9
        assert event.data["extra_key"] == "extra_value"

    @pytest.mark.asyncio
    async def test_broadcast_health_event_with_immune_system(self):
        """Test broadcast_health_event emits when immune_system present."""
        immune_system = Mock()
        bus = EventBus(immune_system=immune_system)
        handler = AsyncMock()
        bus.subscribe("health_update", handler)
        health_status = {"status": "healthy", "score": 0.95}
        await bus.broadcast_health_event(
            debate_id="debate-1",
            health_status=health_status,
        )
        handler.assert_called_once()
        event = handler.call_args[0][0]
        assert event.event_type == "health_update"
        assert event.data["health"] == health_status
        assert "timestamp" in event.data

    @pytest.mark.asyncio
    async def test_broadcast_health_event_without_immune_system(self):
        """Test broadcast_health_event skips when no immune_system."""
        bus = EventBus()
        handler = AsyncMock()
        bus.subscribe("health_update", handler)
        await bus.broadcast_health_event(
            debate_id="debate-1",
            health_status={"status": "healthy"},
        )
        handler.assert_not_called()


# ============================================================================
# User Event Queue Tests
# ============================================================================


class TestUserEventQueue:
    """Test thread-safe user event queue."""

    def test_queue_user_event(self):
        """Test queuing user event."""
        bus = EventBus()
        event = {"type": "vote", "user_id": "user-1", "vote": "support"}
        bus.queue_user_event(event)
        assert bus._user_event_queue.qsize() == 1

    def test_queue_multiple_events(self):
        """Test queuing multiple events."""
        bus = EventBus()
        bus.queue_user_event({"type": "vote"})
        bus.queue_user_event({"type": "suggestion"})
        assert bus._user_event_queue.qsize() == 2

    @pytest.mark.asyncio
    async def test_drain_user_events_without_audience_manager(self):
        """Test drain_user_events returns events without processing."""
        bus = EventBus()
        bus.queue_user_event({"type": "vote", "user_id": "user-1"})
        bus.queue_user_event({"type": "suggestion", "user_id": "user-2"})
        events = await bus.drain_user_events(debate_id="debate-1")
        assert len(events) == 2
        assert bus._user_event_queue.qsize() == 0

    @pytest.mark.asyncio
    async def test_drain_user_events_with_audience_manager(self):
        """Test drain_user_events processes through audience manager."""
        audience_manager = AsyncMock()
        bus = EventBus(audience_manager=audience_manager)
        vote_event = {"type": "vote", "user_id": "user-1", "vote": "support"}
        suggestion_event = {
            "type": "suggestion",
            "user_id": "user-2",
            "content": "Consider X",
        }
        bus.queue_user_event(vote_event)
        bus.queue_user_event(suggestion_event)
        events = await bus.drain_user_events(debate_id="debate-1")
        assert len(events) == 2
        audience_manager.record_vote.assert_called_once_with(
            debate_id="debate-1",
            user_id="user-1",
            vote="support",
        )
        audience_manager.add_suggestion.assert_called_once_with(
            debate_id="debate-1",
            user_id="user-2",
            content="Consider X",
        )

    @pytest.mark.asyncio
    async def test_drain_user_events_handles_errors(self):
        """Test drain_user_events continues on processing errors."""
        audience_manager = AsyncMock()
        audience_manager.record_vote.side_effect = ValueError("Invalid vote")
        bus = EventBus(audience_manager=audience_manager)
        bus.queue_user_event({"type": "vote", "user_id": "user-1"})
        bus.queue_user_event({"type": "suggestion", "user_id": "user-2"})
        # Should not raise
        events = await bus.drain_user_events(debate_id="debate-1")
        assert len(events) == 2

    def test_queue_user_event_lock_timeout(self):
        """Test queue_user_event handles lock timeout."""
        bus = EventBus()
        # Simulate lock held by another thread
        bus._user_event_lock.acquire()
        try:
            # This should timeout and log warning
            with patch.object(bus, "LOCK_TIMEOUT", 0.01):
                bus.queue_user_event({"type": "vote"})
            # Event should be dropped
            assert bus._user_event_queue.qsize() == 0
        finally:
            bus._user_event_lock.release()

    @pytest.mark.asyncio
    async def test_drain_user_events_lock_timeout(self):
        """Test drain_user_events handles lock timeout."""
        bus = EventBus()
        bus.queue_user_event({"type": "vote"})
        # Simulate lock held by another thread
        bus._user_event_lock.acquire()
        try:
            with patch.object(bus, "LOCK_TIMEOUT", 0.01):
                events = await bus.drain_user_events(debate_id="debate-1")
            # Should return empty list
            assert events == []
        finally:
            bus._user_event_lock.release()

    def test_user_event_queue_thread_safety(self):
        """Test user event queue operations are thread-safe."""
        bus = EventBus()
        results = []

        def producer():
            for i in range(10):
                bus.queue_user_event({"type": "vote", "id": i})
                time.sleep(0.001)

        def consumer():
            for _ in range(5):
                try:
                    event = bus._user_event_queue.get(timeout=0.1)
                    results.append(event)
                except queue.Empty:
                    pass
                time.sleep(0.001)

        threads = [
            threading.Thread(target=producer),
            threading.Thread(target=consumer),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have some events queued and some consumed
        total = len(results) + bus._user_event_queue.qsize()
        assert total == 10


# ============================================================================
# Metrics Tests
# ============================================================================


class TestMetrics:
    """Test metrics tracking."""

    @pytest.mark.asyncio
    async def test_get_metrics_initial_state(self):
        """Test get_metrics returns initial state."""
        bus = EventBus()
        metrics = bus.get_metrics()
        assert metrics["total_events_emitted"] == 0
        assert metrics["events_by_type"] == {}
        assert metrics["async_subscribers"] == {}
        assert metrics["sync_subscribers"] == {}
        assert metrics["pending_user_events"] == 0

    @pytest.mark.asyncio
    async def test_get_metrics_after_events(self):
        """Test get_metrics after emitting events."""
        bus = EventBus()
        bus.subscribe("event_a", AsyncMock())
        bus.subscribe("event_a", AsyncMock())
        bus.subscribe_sync("event_b", Mock())
        await bus.emit("event_a", debate_id="debate-1")
        await bus.emit("event_b", debate_id="debate-2")
        bus.queue_user_event({"type": "vote"})
        metrics = bus.get_metrics()
        assert metrics["total_events_emitted"] == 2
        assert metrics["events_by_type"]["event_a"] == 1
        assert metrics["events_by_type"]["event_b"] == 1
        assert metrics["async_subscribers"]["event_a"] == 2
        assert metrics["sync_subscribers"]["event_b"] == 1
        assert metrics["pending_user_events"] == 1

    def test_reset_metrics(self):
        """Test reset_metrics clears counters."""
        bus = EventBus()
        bus.emit_sync("test_event", debate_id="debate-1")
        bus.emit_sync("test_event", debate_id="debate-2")
        bus.reset_metrics()
        metrics = bus.get_metrics()
        assert metrics["total_events_emitted"] == 0
        assert metrics["events_by_type"] == {}

    def test_metrics_isolation(self):
        """Test get_metrics returns copy of events_by_type."""
        bus = EventBus()
        bus.emit_sync("test_event", debate_id="debate-1")
        metrics = bus.get_metrics()
        # Modify returned dict
        metrics["events_by_type"]["new_event"] = 99
        # Should not affect internal state
        metrics2 = bus.get_metrics()
        assert "new_event" not in metrics2["events_by_type"]


# ============================================================================
# Cleanup and Context Manager Tests
# ============================================================================


class TestCleanupAndContext:
    """Test cleanup and context manager."""

    def test_cleanup_clears_handlers(self):
        """Test cleanup clears all handlers."""
        bus = EventBus()
        bus.subscribe("event_a", AsyncMock())
        bus.subscribe_sync("event_b", Mock())
        bus.cleanup()
        assert len(bus._async_handlers) == 0
        assert len(bus._sync_handlers) == 0

    def test_cleanup_clears_metrics(self):
        """Test cleanup clears metrics."""
        bus = EventBus()
        bus.emit_sync("test_event", debate_id="debate-1")
        bus.cleanup()
        assert bus._events_emitted == 0
        assert bus._events_by_type == {}

    def test_cleanup_drains_queue(self):
        """Test cleanup drains user event queue."""
        bus = EventBus()
        bus.queue_user_event({"type": "vote"})
        bus.queue_user_event({"type": "suggestion"})
        bus.cleanup()
        assert bus._user_event_queue.qsize() == 0

    def test_context_manager_cleanup_on_exit(self):
        """Test context manager calls cleanup on exit."""
        with EventBus() as bus:
            bus.subscribe("test_event", AsyncMock())
            bus.emit_sync("test_event", debate_id="debate-1")
            assert len(bus._async_handlers) > 0
        # After exit, should be cleaned up
        assert len(bus._async_handlers) == 0
        assert bus._events_emitted == 0

    def test_context_manager_cleanup_on_exception(self):
        """Test context manager cleans up even on exception."""
        try:
            with EventBus() as bus:
                bus.subscribe("test_event", AsyncMock())
                raise ValueError("Test error")
        except ValueError:
            pass
        # Should still be cleaned up
        assert len(bus._async_handlers) == 0


# ============================================================================
# Singleton Pattern Tests
# ============================================================================


class TestSingletonPattern:
    """Test global singleton pattern."""

    def test_get_event_bus_creates_default(self):
        """Test get_event_bus creates default instance."""
        # Reset global state
        from aragora.debate import event_bus as eb_module

        eb_module._default_bus = None
        bus1 = get_event_bus()
        bus2 = get_event_bus()
        assert bus1 is bus2
        assert isinstance(bus1, EventBus)

    def test_set_event_bus(self):
        """Test set_event_bus replaces default."""
        custom_bus = EventBus()
        set_event_bus(custom_bus)
        bus = get_event_bus()
        assert bus is custom_bus

    def test_singleton_isolation(self):
        """Test singleton state is independent per instance."""
        bus1 = EventBus()
        bus2 = EventBus()
        bus1.emit_sync("event_a", debate_id="debate-1")
        bus2.emit_sync("event_b", debate_id="debate-2")
        metrics1 = bus1.get_metrics()
        metrics2 = bus2.get_metrics()
        assert metrics1["total_events_emitted"] == 1
        assert metrics2["total_events_emitted"] == 1
        assert "event_a" in metrics1["events_by_type"]
        assert "event_b" in metrics2["events_by_type"]


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Test integration scenarios."""

    @pytest.mark.asyncio
    async def test_full_event_lifecycle(self):
        """Test complete event emission lifecycle."""
        event_bridge = Mock()
        audience_manager = AsyncMock()
        spectator = Mock()
        bus = EventBus(
            event_bridge=event_bridge,
            audience_manager=audience_manager,
            spectator=spectator,
        )

        # Subscribe handlers
        async_handler = AsyncMock()
        sync_handler = Mock()
        bus.subscribe("debate_start", async_handler)
        bus.subscribe_sync("debate_start", sync_handler)

        # Emit event
        await bus.emit(
            "debate_start",
            debate_id="debate-1",
            task="Design system",
            participants=["claude", "gpt4"],
        )

        # Verify all components notified
        event_bridge.notify.assert_called_once()
        spectator.emit.assert_called_once()
        async_handler.assert_called_once()
        sync_handler.assert_called_once()

        # Verify event structure
        event = async_handler.call_args[0][0]
        assert event.event_type == "debate_start"
        assert event.debate_id == "debate-1"
        assert event.data["task"] == "Design system"

    @pytest.mark.asyncio
    async def test_mixed_async_sync_handlers(self):
        """Test async and sync handlers can coexist."""
        bus = EventBus()
        results = []

        async def async_handler(event: DebateEvent) -> None:
            results.append(("async", event.event_type))

        def sync_handler(event: DebateEvent) -> None:
            results.append(("sync", event.event_type))

        bus.subscribe("test_event", async_handler)
        bus.subscribe_sync("test_event", sync_handler)
        await bus.emit("test_event", debate_id="debate-1")

        assert ("async", "test_event") in results
        assert ("sync", "test_event") in results

    @pytest.mark.asyncio
    async def test_concurrent_emissions(self):
        """Test concurrent event emissions."""
        bus = EventBus()
        handler = AsyncMock()
        bus.subscribe("test_event", handler)

        # Emit multiple events concurrently
        await asyncio.gather(
            bus.emit("test_event", debate_id="debate-1"),
            bus.emit("test_event", debate_id="debate-2"),
            bus.emit("test_event", debate_id="debate-3"),
        )

        assert handler.call_count == 3
        metrics = bus.get_metrics()
        assert metrics["total_events_emitted"] == 3

    @pytest.mark.asyncio
    async def test_event_bridge_strips_event_type(self):
        """Test event_bridge.notify doesn't receive duplicate event_type."""
        event_bridge = Mock()
        bus = EventBus(event_bridge=event_bridge)
        await bus.emit("test_event", debate_id="debate-1", key="value")
        args, kwargs = event_bridge.notify.call_args
        assert args[0] == "test_event"
        assert "event_type" not in kwargs
        assert "key" in kwargs

    @pytest.mark.asyncio
    async def test_audience_manager_vote_processing(self):
        """Test audience manager processes votes correctly."""
        audience_manager = AsyncMock()
        bus = EventBus(audience_manager=audience_manager)
        bus.queue_user_event(
            {
                "type": "vote",
                "user_id": "alice",
                "vote": "approve",
            }
        )
        await bus.drain_user_events(debate_id="debate-1")
        audience_manager.record_vote.assert_called_once_with(
            debate_id="debate-1",
            user_id="alice",
            vote="approve",
        )

    @pytest.mark.asyncio
    async def test_audience_manager_suggestion_processing(self):
        """Test audience manager processes suggestions correctly."""
        audience_manager = AsyncMock()
        bus = EventBus(audience_manager=audience_manager)
        bus.queue_user_event(
            {
                "type": "suggestion",
                "user_id": "bob",
                "content": "Consider security",
            }
        )
        await bus.drain_user_events(debate_id="debate-1")
        audience_manager.add_suggestion.assert_called_once_with(
            debate_id="debate-1",
            user_id="bob",
            content="Consider security",
        )

    @pytest.mark.asyncio
    async def test_unknown_user_event_type_ignored(self):
        """Test unknown user event types are ignored gracefully."""
        audience_manager = AsyncMock()
        bus = EventBus(audience_manager=audience_manager)
        bus.queue_user_event(
            {
                "type": "unknown_type",
                "user_id": "charlie",
            }
        )
        events = await bus.drain_user_events(debate_id="debate-1")
        assert len(events) == 1
        audience_manager.record_vote.assert_not_called()
        audience_manager.add_suggestion.assert_not_called()
