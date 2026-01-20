"""
Tests for EventBatcher - batching KM events for efficient WebSocket transmission.
"""

import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock

from aragora.knowledge.mound.event_batcher import (
    EventBatcher,
    EventBatch,
    BatchedEvent,
    AdapterEventBatcher,
)


class TestBatchedEvent:
    """Tests for BatchedEvent dataclass."""

    def test_create_event(self):
        """Test creating a batched event."""
        event = BatchedEvent(
            event_type="knowledge_indexed",
            data={"source": "evidence", "id": "ev_001"},
        )

        assert event.event_type == "knowledge_indexed"
        assert event.data["source"] == "evidence"
        assert event.timestamp > 0


class TestEventBatch:
    """Tests for EventBatch dataclass."""

    def test_create_batch(self):
        """Test creating an empty batch."""
        batch = EventBatch()
        assert batch.count == 0
        assert batch.events == []

    def test_batch_count(self):
        """Test batch count property."""
        batch = EventBatch(events=[
            BatchedEvent(event_type="e1", data={}),
            BatchedEvent(event_type="e2", data={}),
        ])
        assert batch.count == 2

    def test_batch_to_dict(self):
        """Test batch serialization."""
        batch = EventBatch(events=[
            BatchedEvent(event_type="knowledge_indexed", data={"id": "1"}),
        ])

        result = batch.to_dict()

        assert result["type"] == "km_batch"
        assert result["count"] == 1
        assert len(result["events"]) == 1
        assert result["events"][0]["type"] == "knowledge_indexed"


class TestEventBatcher:
    """Tests for EventBatcher."""

    def test_create_batcher(self):
        """Test creating a batcher."""
        batcher = EventBatcher()
        assert batcher._batch_interval_ms == 100.0
        assert batcher._max_batch_size == 50

    def test_create_batcher_custom_config(self):
        """Test creating a batcher with custom config."""
        batcher = EventBatcher(
            batch_interval_ms=200.0,
            max_batch_size=100,
        )
        assert batcher._batch_interval_ms == 200.0
        assert batcher._max_batch_size == 100

    def test_queue_event(self):
        """Test queuing an event."""
        batcher = EventBatcher()
        batcher.queue_event("test_event", {"key": "value"})

        assert batcher._total_events_queued == 1
        assert len(batcher._batch.events) == 1

    def test_queue_multiple_events(self):
        """Test queuing multiple events."""
        batcher = EventBatcher()

        for i in range(5):
            batcher.queue_event(f"event_{i}", {"index": i})

        assert batcher._total_events_queued == 5
        assert len(batcher._batch.events) == 5

    def test_passthrough_event(self):
        """Test that passthrough events bypass batching."""
        callback = MagicMock()
        batcher = EventBatcher(
            callback=callback,
            passthrough_event_types=["urgent_event"],
        )

        batcher.queue_event("urgent_event", {"priority": "high"})

        # Should be emitted immediately, not batched
        callback.assert_called_once_with("urgent_event", {"priority": "high"})
        assert batcher._passthrough_events == 1
        assert len(batcher._batch.events) == 0

    def test_batch_event_type_filter(self):
        """Test that only specified event types are batched."""
        callback = MagicMock()
        batcher = EventBatcher(
            callback=callback,
            batch_event_types=["knowledge_indexed", "belief_converged"],
        )

        # This should be batched
        batcher.queue_event("knowledge_indexed", {"id": "1"})

        # This should pass through (not in batch list)
        batcher.queue_event("other_event", {"id": "2"})

        # Batched event in queue
        assert len(batcher._batch.events) == 1
        assert batcher._batch.events[0].event_type == "knowledge_indexed"

        # Other event passed through
        callback.assert_called_once_with("other_event", {"id": "2"})

    def test_flush_sync(self):
        """Test synchronous flush."""
        events_received = []

        def callback(event_type, data):
            events_received.append((event_type, data))

        batcher = EventBatcher(callback=callback)

        batcher.queue_event("event_1", {"id": "1"})
        batcher.queue_event("event_2", {"id": "2"})

        count = batcher._flush_sync()

        assert count == 2
        assert len(events_received) == 1
        assert events_received[0][0] == "km_batch"
        assert events_received[0][1]["count"] == 2

    def test_max_batch_size_trigger(self):
        """Test that reaching max batch size triggers flush."""
        events_received = []

        def callback(event_type, data):
            events_received.append((event_type, data))

        batcher = EventBatcher(callback=callback, max_batch_size=3)

        # Queue 3 events (should trigger flush at 3)
        batcher.queue_event("e1", {})
        batcher.queue_event("e2", {})
        batcher.queue_event("e3", {})

        # Give sync flush a chance
        batcher._flush_sync()

        # Should have at least one batch
        assert batcher._total_batches_emitted >= 1

    def test_get_stats(self):
        """Test getting statistics."""
        batcher = EventBatcher()

        batcher.queue_event("e1", {})
        batcher.queue_event("e2", {})

        stats = batcher.get_stats()

        assert stats["total_events_queued"] == 2
        assert stats["pending_events"] == 2
        assert stats["batch_interval_ms"] == 100.0
        assert stats["max_batch_size"] == 50

    def test_reset_stats(self):
        """Test resetting statistics."""
        batcher = EventBatcher()

        batcher.queue_event("e1", {})
        batcher._total_events_emitted = 10
        batcher._total_batches_emitted = 5

        batcher.reset_stats()

        assert batcher._total_events_queued == 0
        assert batcher._total_events_emitted == 0
        assert batcher._total_batches_emitted == 0


class TestEventBatcherAsync:
    """Async tests for EventBatcher."""

    @pytest.mark.asyncio
    async def test_async_flush(self):
        """Test async flush."""
        events_received = []

        async def callback(event_type, data):
            events_received.append((event_type, data))

        batcher = EventBatcher(callback=callback)

        batcher.queue_event("event_1", {"id": "1"})
        batcher.queue_event("event_2", {"id": "2"})

        count = await batcher.flush()

        assert count == 2
        assert len(events_received) == 1
        assert events_received[0][0] == "km_batch"

    @pytest.mark.asyncio
    async def test_start_stop(self):
        """Test starting and stopping the batcher."""
        callback = AsyncMock()
        batcher = EventBatcher(callback=callback, batch_interval_ms=50)

        loop = asyncio.get_event_loop()
        batcher.start(loop)

        assert batcher._running is True
        assert batcher._flush_task is not None

        # Queue some events
        batcher.queue_event("e1", {})
        batcher.queue_event("e2", {})

        # Wait for a flush cycle
        await asyncio.sleep(0.1)

        await batcher.stop()

        assert batcher._running is False
        # Events should have been flushed
        assert callback.called

    @pytest.mark.asyncio
    async def test_empty_flush(self):
        """Test flushing empty batch."""
        callback = AsyncMock()
        batcher = EventBatcher(callback=callback)

        count = await batcher.flush()

        assert count == 0
        callback.assert_not_called()


class TestAdapterEventBatcher:
    """Tests for AdapterEventBatcher wrapper."""

    def test_create_adapter_batcher(self):
        """Test creating adapter batcher."""
        batcher = EventBatcher()
        adapter_batcher = AdapterEventBatcher(batcher)

        assert adapter_batcher._batcher is batcher
        assert adapter_batcher._prefix == "km"

    def test_event_callback_adds_prefix(self):
        """Test that event callback adds prefix."""
        batcher = EventBatcher()
        adapter_batcher = AdapterEventBatcher(batcher, prefix="test")

        adapter_batcher.event_callback("indexed", {"id": "1"})

        # Event should have prefix
        assert batcher._batch.events[0].event_type == "test_indexed"

    def test_event_callback_no_double_prefix(self):
        """Test that callback doesn't double-prefix."""
        batcher = EventBatcher()
        adapter_batcher = AdapterEventBatcher(batcher, prefix="km")

        adapter_batcher.event_callback("km_indexed", {"id": "1"})

        # Should keep single prefix
        assert batcher._batch.events[0].event_type == "km_indexed"

    def test_stats_property(self):
        """Test stats property passthrough."""
        batcher = EventBatcher()
        adapter_batcher = AdapterEventBatcher(batcher)

        adapter_batcher.event_callback("e1", {})
        adapter_batcher.event_callback("e2", {})

        stats = adapter_batcher.stats

        assert stats["total_events_queued"] == 2

    def test_integration_with_adapter(self):
        """Test integration with a mock adapter."""
        # Setup batcher
        events_received = []

        def ws_callback(event_type, data):
            events_received.append((event_type, data))

        batcher = EventBatcher(callback=ws_callback)
        adapter_batcher = AdapterEventBatcher(batcher)

        # Simulate adapter calls
        adapter_batcher.event_callback("knowledge_indexed", {"source": "evidence"})
        adapter_batcher.event_callback("belief_converged", {"debate_id": "d1"})
        adapter_batcher.event_callback("elo_updated", {"agent": "claude"})

        # Flush
        batcher._flush_sync()

        # Should have one batch with 3 events
        assert len(events_received) == 1
        batch = events_received[0][1]
        assert batch["count"] == 3
        assert batch["events"][0]["type"] == "km_knowledge_indexed"
