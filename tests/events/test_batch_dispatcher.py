"""
Tests for batched webhook dispatcher.
"""

import time
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest


class TestBatchedEvent:
    """Tests for BatchedEvent dataclass."""

    def test_creates_with_timestamp(self):
        """Test that BatchedEvent gets a timestamp."""
        from aragora.events.batch_dispatcher import BatchedEvent

        before = time.time()
        event = BatchedEvent(event_type="test", data={"key": "value"})
        after = time.time()

        assert before <= event.timestamp <= after
        assert event.event_type == "test"
        assert event.data == {"key": "value"}


class TestEventBatch:
    """Tests for EventBatch dataclass."""

    def test_add_event(self):
        """Test adding events to a batch."""
        from aragora.events.batch_dispatcher import BatchedEvent, EventBatch

        batch = EventBatch(event_type="test")
        event = BatchedEvent(event_type="test", data={"n": 1})

        batch.add(event)

        assert len(batch.events) == 1
        assert batch.events[0].data == {"n": 1}

    def test_is_full(self):
        """Test batch fullness check."""
        from aragora.events.batch_dispatcher import BatchedEvent, EventBatch

        batch = EventBatch(event_type="test")

        # Add events up to max
        for i in range(5):
            batch.add(BatchedEvent(event_type="test", data={"n": i}))

        assert not batch.is_full(max_size=10)
        assert batch.is_full(max_size=5)

    def test_is_expired(self):
        """Test batch expiration check."""
        from aragora.events.batch_dispatcher import EventBatch

        batch = EventBatch(event_type="test")

        # Just created - not expired
        assert not batch.is_expired(window=5.0)

        # Simulate old batch
        batch.created_at = time.time() - 10.0
        assert batch.is_expired(window=5.0)

    def test_to_payload(self):
        """Test converting batch to payload."""
        from aragora.events.batch_dispatcher import BatchedEvent, EventBatch

        batch = EventBatch(event_type="slo_violation")
        batch.add(BatchedEvent(event_type="slo_violation", data={"operation": "km_query", "severity": "major"}))
        batch.add(BatchedEvent(event_type="slo_violation", data={"operation": "km_query", "severity": "minor"}))

        payload = batch.to_payload()

        assert payload["event"] == "slo_violation_batch"
        assert payload["batch_size"] == 2
        assert len(payload["events"]) == 2
        assert "summary" in payload
        assert payload["summary"]["count"] == 2
        assert payload["summary"]["by_operation"]["km_query"] == 2

    def test_summary_by_severity(self):
        """Test that summary includes severity breakdown."""
        from aragora.events.batch_dispatcher import BatchedEvent, EventBatch

        batch = EventBatch(event_type="slo_violation")
        batch.add(BatchedEvent(event_type="slo_violation", data={"severity": "critical"}))
        batch.add(BatchedEvent(event_type="slo_violation", data={"severity": "critical"}))
        batch.add(BatchedEvent(event_type="slo_violation", data={"severity": "minor"}))

        payload = batch.to_payload()

        assert payload["summary"]["by_severity"]["critical"] == 2
        assert payload["summary"]["by_severity"]["minor"] == 1


class TestBatchWebhookDispatcher:
    """Tests for BatchWebhookDispatcher."""

    @pytest.fixture
    def dispatcher(self):
        """Create a test dispatcher with short window."""
        from aragora.events.batch_dispatcher import BatchWebhookDispatcher

        d = BatchWebhookDispatcher(
            batch_window=0.1,  # 100ms for fast tests
            max_batch_size=5,
            priority_events=frozenset(["priority_event"]),
        )
        yield d
        d.shutdown(flush=False)

    def test_queue_event(self, dispatcher):
        """Test queueing an event."""
        delivered = []
        dispatcher.set_delivery_callback(lambda t, p: delivered.append((t, p)))

        dispatcher.queue_event("test_event", {"data": "value"})

        # Should be batched, not delivered yet
        assert len(delivered) == 0

        stats = dispatcher.get_stats()
        assert stats["events_queued"] == 1
        assert stats["pending_events"] == 1

    def test_priority_event_bypasses_batching(self, dispatcher):
        """Test that priority events are delivered immediately."""
        delivered = []
        dispatcher.set_delivery_callback(lambda t, p: delivered.append((t, p)))

        dispatcher.queue_event("priority_event", {"urgent": True})

        # Priority event should be delivered immediately
        assert len(delivered) == 1
        assert delivered[0][0] == "priority_event"
        assert delivered[0][1]["batched"] is False

    def test_batch_flushes_on_size(self, dispatcher):
        """Test that batch flushes when full."""
        delivered = []
        dispatcher.set_delivery_callback(lambda t, p: delivered.append((t, p)))

        # Queue up to max batch size
        for i in range(5):
            dispatcher.queue_event("test_event", {"n": i})

        # Should trigger flush
        assert len(delivered) == 1
        assert delivered[0][1]["batch_size"] == 5

    def test_batch_flushes_on_time(self, dispatcher):
        """Test that batch flushes after window expires."""
        delivered = []
        dispatcher.set_delivery_callback(lambda t, p: delivered.append((t, p)))

        dispatcher.queue_event("test_event", {"data": "value"})

        # Wait for window + flush interval
        time.sleep(0.3)

        assert len(delivered) == 1
        assert delivered[0][1]["batch_size"] == 1

    def test_flush_all(self, dispatcher):
        """Test manual flush of all batches."""
        delivered = []
        dispatcher.set_delivery_callback(lambda t, p: delivered.append((t, p)))

        dispatcher.queue_event("type_a", {"a": 1})
        dispatcher.queue_event("type_b", {"b": 2})

        dispatcher.flush_all()

        assert len(delivered) == 2

    def test_get_stats(self, dispatcher):
        """Test stats reporting."""
        dispatcher.queue_event("test_event", {"data": 1})
        dispatcher.queue_event("test_event", {"data": 2})

        stats = dispatcher.get_stats()

        assert stats["events_queued"] == 2
        assert stats["pending_events"] == 2
        assert stats["pending_batches"] == 1
        assert stats["batch_window"] == 0.1

    def test_shutdown_flushes_by_default(self, dispatcher):
        """Test that shutdown flushes pending batches."""
        delivered = []
        dispatcher.set_delivery_callback(lambda t, p: delivered.append((t, p)))

        dispatcher.queue_event("test_event", {"data": "value"})

        # Manual shutdown (fixture won't flush)
        dispatcher.shutdown(flush=True)

        assert len(delivered) == 1


class TestGlobalBatchDispatcher:
    """Tests for global batch dispatcher functions."""

    @pytest.fixture(autouse=True)
    def reset_global(self):
        """Reset global dispatcher before each test."""
        from aragora.events import batch_dispatcher

        batch_dispatcher._batch_dispatcher = None
        yield
        if batch_dispatcher._batch_dispatcher:
            batch_dispatcher._batch_dispatcher.shutdown(flush=False)
            batch_dispatcher._batch_dispatcher = None

    def test_get_batch_dispatcher(self):
        """Test getting global dispatcher."""
        from aragora.events.batch_dispatcher import get_batch_dispatcher

        d1 = get_batch_dispatcher()
        d2 = get_batch_dispatcher()

        assert d1 is d2

    def test_queue_batched_event(self):
        """Test convenience function."""
        from aragora.events.batch_dispatcher import (
            get_batch_dispatcher,
            queue_batched_event,
        )

        dispatcher = get_batch_dispatcher()

        # Override callback to track
        delivered = []
        dispatcher.set_delivery_callback(lambda t, p: delivered.append((t, p)))

        queue_batched_event("test_event", {"key": "value"})
        dispatcher.flush_all()

        assert len(delivered) == 1

    def test_shutdown_batch_dispatcher(self):
        """Test shutdown function."""
        from aragora.events.batch_dispatcher import (
            get_batch_dispatcher,
            shutdown_batch_dispatcher,
        )
        from aragora.events import batch_dispatcher

        get_batch_dispatcher()
        assert batch_dispatcher._batch_dispatcher is not None

        shutdown_batch_dispatcher()
        assert batch_dispatcher._batch_dispatcher is None


class TestBatchingIntegration:
    """Integration tests for batching behavior."""

    def test_multiple_event_types_batched_separately(self):
        """Test that different event types are batched separately."""
        from aragora.events.batch_dispatcher import BatchWebhookDispatcher

        delivered: List[tuple] = []
        dispatcher = BatchWebhookDispatcher(
            batch_window=1.0,
            max_batch_size=10,
        )
        dispatcher.set_delivery_callback(lambda t, p: delivered.append((t, p)))

        try:
            dispatcher.queue_event("type_a", {"data": 1})
            dispatcher.queue_event("type_a", {"data": 2})
            dispatcher.queue_event("type_b", {"data": 3})
            dispatcher.queue_event("type_b", {"data": 4})

            dispatcher.flush_all()

            assert len(delivered) == 2
            types = {d[0] for d in delivered}
            assert types == {"type_a", "type_b"}

        finally:
            dispatcher.shutdown(flush=False)

    def test_high_volume_batching(self):
        """Test handling high volume of events."""
        from aragora.events.batch_dispatcher import BatchWebhookDispatcher

        delivered: List[tuple] = []
        # Use empty priority events so nothing bypasses batching
        dispatcher = BatchWebhookDispatcher(
            batch_window=5.0,
            max_batch_size=50,
            priority_events=frozenset(),  # No priority events
        )
        dispatcher.set_delivery_callback(lambda t, p: delivered.append((t, p)))

        try:
            # Queue 200 events (using a non-priority event type)
            for i in range(200):
                dispatcher.queue_event("bulk_notification", {"n": i})

            # Should have flushed 4 batches (200 / 50)
            assert len(delivered) == 4

            for _, payload in delivered:
                assert payload["batch_size"] == 50

        finally:
            dispatcher.shutdown(flush=False)
