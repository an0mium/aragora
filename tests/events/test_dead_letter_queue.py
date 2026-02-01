"""Tests for Internal Event Dead Letter Queue.

Tests cover:
- Enqueuing a failed event (capture)
- Dequeue/retry of a failed event
- Max retry limit enforcement via queue size bounds
- TTL expiration / cleanup of old events
- Queue size bounds and eviction
- Event metadata tracking (retry count, failure reason, timestamps)
- Concurrent enqueue operations
- Empty queue handling
- Event prioritization (filtering by handler)
- Dead letter queue statistics
- Discard events
- Global singleton management
"""

from __future__ import annotations

import os
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from aragora.events.dead_letter_queue import (
    DLQStats,
    EventDLQ,
    EventDLQPersistence,
    FailedEvent,
    FailedEventStatus,
    get_event_dlq,
    reset_event_dlq,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_error(msg: str = "something went wrong") -> Exception:
    """Create a test exception."""
    return ValueError(msg)


def _make_runtime_error(msg: str = "runtime failure") -> RuntimeError:
    return RuntimeError(msg)


@pytest.fixture()
def dlq(tmp_path):
    """Create an EventDLQ backed by a temporary SQLite database."""
    db_path = str(tmp_path / "test_dlq.db")
    persistence = EventDLQPersistence(db_path=db_path)
    return EventDLQ(persistence=persistence)


@pytest.fixture()
def small_dlq(tmp_path):
    """Create a small-capacity DLQ for testing eviction."""
    db_path = str(tmp_path / "small_dlq.db")
    persistence = EventDLQPersistence(db_path=db_path)
    return EventDLQ(max_size=3, persistence=persistence)


@pytest.fixture(autouse=True)
def _reset_global_dlq():
    """Reset the global DLQ singleton before and after each test."""
    reset_event_dlq()
    yield
    reset_event_dlq()


# ===========================================================================
# Test: Enqueue a failed event
# ===========================================================================


class TestCaptureFailedEvent:
    """Tests for capturing (enqueuing) events into the DLQ."""

    def test_capture_returns_failed_event(self, dlq: EventDLQ):
        """Capturing an event should return a FailedEvent instance."""
        event = dlq.capture(
            event_type="debate.completed",
            event_data={"debate_id": "d-1"},
            handler_name="memory_sync",
            error=_make_error("sync failed"),
        )
        assert isinstance(event, FailedEvent)
        assert event.event_type == "debate.completed"
        assert event.handler_name == "memory_sync"
        assert event.error_message == "sync failed"
        assert event.error_type == "ValueError"
        assert event.status == FailedEventStatus.PENDING

    def test_capture_assigns_unique_id(self, dlq: EventDLQ):
        """Each captured event should get a unique ID."""
        e1 = dlq.capture(
            event_type="test.event",
            event_data={},
            handler_name="h1",
            error=_make_error(),
        )
        e2 = dlq.capture(
            event_type="test.event",
            event_data={},
            handler_name="h1",
            error=_make_error(),
        )
        assert e1.id != e2.id

    def test_capture_preserves_event_data(self, dlq: EventDLQ):
        """Event data payload should be stored and retrievable."""
        payload = {"debate_id": "d-42", "round": 3, "nested": {"key": "value"}}
        event = dlq.capture(
            event_type="round.failed",
            event_data=payload,
            handler_name="round_handler",
            error=_make_error(),
        )
        retrieved = dlq.get_event(event.id)
        assert retrieved is not None
        assert retrieved.event_data == payload

    def test_capture_records_timestamps(self, dlq: EventDLQ):
        """Captured events should have created_at and updated_at timestamps."""
        before = time.time()
        event = dlq.capture(
            event_type="test.event",
            event_data={},
            handler_name="h",
            error=_make_error(),
        )
        after = time.time()

        assert before <= event.created_at <= after
        assert before <= event.updated_at <= after

    def test_capture_with_optional_fields(self, dlq: EventDLQ):
        """Capture should accept optional trace_id, correlation_id, metadata."""
        event = dlq.capture(
            event_type="test.event",
            event_data={},
            handler_name="h",
            error=_make_error(),
            retry_count=5,
            original_timestamp=1000.0,
            trace_id="trace-abc",
            correlation_id="corr-xyz",
            metadata={"source": "kafka"},
        )
        assert event.retry_count == 5
        assert event.original_timestamp == 1000.0
        assert event.trace_id == "trace-abc"
        assert event.correlation_id == "corr-xyz"
        assert event.metadata == {"source": "kafka"}


# ===========================================================================
# Test: Dequeue / retry a failed event
# ===========================================================================


class TestRetryEvent:
    """Tests for retrying failed events from the DLQ."""

    def test_retry_success_marks_recovered(self, dlq: EventDLQ):
        """Successful retry should mark the event as recovered."""
        event = dlq.capture(
            event_type="test.event",
            event_data={"key": "value"},
            handler_name="h",
            error=_make_error(),
        )
        handler = MagicMock()

        result = dlq.retry_event(event.id, handler)

        assert result is True
        handler.assert_called_once_with({"key": "value"})
        recovered = dlq.get_event(event.id)
        assert recovered is not None
        assert recovered.status == FailedEventStatus.RECOVERED

    def test_retry_failure_keeps_pending(self, dlq: EventDLQ):
        """Failed retry should keep the event pending with updated error info."""
        event = dlq.capture(
            event_type="test.event",
            event_data={},
            handler_name="h",
            error=_make_error("original error"),
        )

        def failing_handler(data):
            raise RuntimeError("retry also failed")

        result = dlq.retry_event(event.id, failing_handler)

        assert result is False
        updated = dlq.get_event(event.id)
        assert updated is not None
        assert updated.status == FailedEventStatus.PENDING
        assert updated.error_message == "retry also failed"
        assert updated.error_type == "RuntimeError"
        assert updated.retry_count == 1  # incremented from 0

    def test_retry_nonexistent_event(self, dlq: EventDLQ):
        """Retrying a nonexistent event should return False."""
        result = dlq.retry_event("nonexistent-id", MagicMock())
        assert result is False

    def test_retry_non_pending_event(self, dlq: EventDLQ):
        """Retrying a non-pending event (e.g., already recovered) should fail."""
        event = dlq.capture(
            event_type="test.event",
            event_data={},
            handler_name="h",
            error=_make_error(),
        )
        # Recover it first
        dlq.retry_event(event.id, MagicMock())

        # Try to retry again
        result = dlq.retry_event(event.id, MagicMock())
        assert result is False

    def test_retry_increments_count_on_failure(self, dlq: EventDLQ):
        """Multiple failed retries should increment the retry count."""
        event = dlq.capture(
            event_type="test.event",
            event_data={},
            handler_name="h",
            error=_make_error(),
            retry_count=3,
        )

        def failing_handler(data):
            raise RuntimeError("still failing")

        dlq.retry_event(event.id, failing_handler)
        updated = dlq.get_event(event.id)
        assert updated is not None
        assert updated.retry_count == 4  # was 3, now 4


# ===========================================================================
# Test: Max retry / queue size bounds (eviction)
# ===========================================================================


class TestQueueSizeBounds:
    """Tests for queue size enforcement via eviction of oldest events."""

    def test_eviction_when_max_size_exceeded(self, small_dlq: EventDLQ):
        """When max_size is exceeded, oldest pending events should be evicted."""
        events = []
        for i in range(4):
            e = small_dlq.capture(
                event_type="test.event",
                event_data={"idx": i},
                handler_name="h",
                error=_make_error(f"error {i}"),
            )
            events.append(e)

        # The first event should have been evicted (discarded)
        first_event = small_dlq.get_event(events[0].id)
        assert first_event is not None
        assert first_event.status == FailedEventStatus.DISCARDED

        # The last 3 should still be pending
        for e in events[1:]:
            retrieved = small_dlq.get_event(e.id)
            assert retrieved is not None
            assert retrieved.status == FailedEventStatus.PENDING

    def test_eviction_preserves_newest(self, small_dlq: EventDLQ):
        """Eviction should always preserve the newest events."""
        for i in range(5):
            small_dlq.capture(
                event_type="test.event",
                event_data={"idx": i},
                handler_name="h",
                error=_make_error(),
            )

        pending = small_dlq.get_pending_events()
        assert len(pending) == 3  # max_size is 3


# ===========================================================================
# Test: TTL expiration (cleanup of old events)
# ===========================================================================


class TestTTLCleanup:
    """Tests for cleanup of old recovered/discarded events."""

    def test_cleanup_removes_old_recovered_events(self, dlq: EventDLQ):
        """Cleanup should remove old recovered events past retention."""
        event = dlq.capture(
            event_type="test.event",
            event_data={},
            handler_name="h",
            error=_make_error(),
        )
        # Recover it
        dlq.retry_event(event.id, MagicMock())

        # Patch time to make the event appear old (8 days ago)
        old_time = time.time() - (8 * 24 * 60 * 60)
        dlq._persistence._get_conn().execute(
            "UPDATE failed_events SET created_at = ? WHERE id = ?",
            (old_time, event.id),
        )
        dlq._persistence._get_conn().commit()

        removed = dlq.cleanup(retention_days=7)
        assert removed == 1

        # Event should be gone
        assert dlq.get_event(event.id) is None

    def test_cleanup_preserves_recent_events(self, dlq: EventDLQ):
        """Cleanup should preserve recently recovered events."""
        event = dlq.capture(
            event_type="test.event",
            event_data={},
            handler_name="h",
            error=_make_error(),
        )
        # Recover it
        dlq.retry_event(event.id, MagicMock())

        removed = dlq.cleanup(retention_days=7)
        assert removed == 0
        assert dlq.get_event(event.id) is not None

    def test_cleanup_preserves_pending_events(self, dlq: EventDLQ):
        """Cleanup should never remove pending events, even if old."""
        event = dlq.capture(
            event_type="test.event",
            event_data={},
            handler_name="h",
            error=_make_error(),
        )

        # Make it appear old
        old_time = time.time() - (30 * 24 * 60 * 60)
        dlq._persistence._get_conn().execute(
            "UPDATE failed_events SET created_at = ? WHERE id = ?",
            (old_time, event.id),
        )
        dlq._persistence._get_conn().commit()

        removed = dlq.cleanup(retention_days=7)
        assert removed == 0  # Pending events are not cleaned up
        assert dlq.get_event(event.id) is not None

    def test_cleanup_removes_old_discarded_events(self, dlq: EventDLQ):
        """Cleanup should remove old discarded events."""
        event = dlq.capture(
            event_type="test.event",
            event_data={},
            handler_name="h",
            error=_make_error(),
        )
        dlq.discard_event(event.id)

        # Make it old
        old_time = time.time() - (10 * 24 * 60 * 60)
        dlq._persistence._get_conn().execute(
            "UPDATE failed_events SET created_at = ? WHERE id = ?",
            (old_time, event.id),
        )
        dlq._persistence._get_conn().commit()

        removed = dlq.cleanup(retention_days=7)
        assert removed == 1


# ===========================================================================
# Test: Event metadata tracking
# ===========================================================================


class TestEventMetadataTracking:
    """Tests for event metadata fields (retry count, failure reason, timestamps)."""

    def test_error_type_captured(self, dlq: EventDLQ):
        """The error type name should be captured from the exception."""
        event = dlq.capture(
            event_type="test.event",
            event_data={},
            handler_name="h",
            error=_make_runtime_error("timeout"),
        )
        assert event.error_type == "RuntimeError"
        assert event.error_message == "timeout"

    def test_retry_count_initial_value(self, dlq: EventDLQ):
        """Default retry count should be 0 unless specified."""
        event = dlq.capture(
            event_type="test.event",
            event_data={},
            handler_name="h",
            error=_make_error(),
        )
        assert event.retry_count == 0

    def test_retry_count_can_be_set(self, dlq: EventDLQ):
        """Retry count can be explicitly set on capture."""
        event = dlq.capture(
            event_type="test.event",
            event_data={},
            handler_name="h",
            error=_make_error(),
            retry_count=7,
        )
        assert event.retry_count == 7

    def test_updated_at_changes_on_retry(self, dlq: EventDLQ):
        """updated_at should change after a failed retry attempt."""
        event = dlq.capture(
            event_type="test.event",
            event_data={},
            handler_name="h",
            error=_make_error(),
        )
        original_updated = event.updated_at

        def failing_handler(data):
            raise RuntimeError("fail")

        time.sleep(0.01)
        dlq.retry_event(event.id, failing_handler)

        updated = dlq.get_event(event.id)
        assert updated is not None
        assert updated.updated_at >= original_updated

    def test_metadata_dict_persisted(self, dlq: EventDLQ):
        """Custom metadata dict should be persisted and retrievable."""
        meta = {"source": "kafka", "partition": 5, "offset": 12345}
        event = dlq.capture(
            event_type="test.event",
            event_data={},
            handler_name="h",
            error=_make_error(),
            metadata=meta,
        )

        retrieved = dlq.get_event(event.id)
        assert retrieved is not None
        assert retrieved.metadata == meta

    def test_original_timestamp_defaults_to_now(self, dlq: EventDLQ):
        """If not provided, original_timestamp defaults to current time."""
        before = time.time()
        event = dlq.capture(
            event_type="test.event",
            event_data={},
            handler_name="h",
            error=_make_error(),
        )
        after = time.time()

        assert before <= event.original_timestamp <= after

    def test_original_timestamp_explicitly_set(self, dlq: EventDLQ):
        """original_timestamp can be explicitly set."""
        event = dlq.capture(
            event_type="test.event",
            event_data={},
            handler_name="h",
            error=_make_error(),
            original_timestamp=1234567890.0,
        )
        assert event.original_timestamp == 1234567890.0


# ===========================================================================
# Test: Concurrent enqueue operations
# ===========================================================================


class TestConcurrentOperations:
    """Tests for thread-safety of concurrent capture operations."""

    def test_concurrent_captures(self, tmp_path):
        """Multiple threads capturing events concurrently should not lose data."""
        db_path = str(tmp_path / "concurrent_dlq.db")
        persistence = EventDLQPersistence(db_path=db_path)
        dlq = EventDLQ(max_size=10000, persistence=persistence)

        num_threads = 8
        events_per_thread = 25
        captured_ids: list[str] = []
        lock = threading.Lock()

        def capture_events(thread_idx: int):
            local_ids = []
            for i in range(events_per_thread):
                event = dlq.capture(
                    event_type=f"thread.{thread_idx}",
                    event_data={"thread": thread_idx, "idx": i},
                    handler_name=f"handler_{thread_idx}",
                    error=_make_error(f"error-{thread_idx}-{i}"),
                )
                local_ids.append(event.id)
            with lock:
                captured_ids.extend(local_ids)

        threads = [threading.Thread(target=capture_events, args=(t,)) for t in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        total_expected = num_threads * events_per_thread
        assert len(captured_ids) == total_expected

        # All IDs should be unique
        assert len(set(captured_ids)) == total_expected

    def test_concurrent_capture_and_retry(self, tmp_path):
        """Concurrent capture + retry operations should not corrupt state."""
        db_path = str(tmp_path / "concurrent_retry_dlq.db")
        persistence = EventDLQPersistence(db_path=db_path)
        dlq = EventDLQ(max_size=10000, persistence=persistence)

        # Pre-populate with events
        event_ids = []
        for i in range(20):
            e = dlq.capture(
                event_type="test.event",
                event_data={"idx": i},
                handler_name="h",
                error=_make_error(),
            )
            event_ids.append(e.id)

        results = {"retried": 0, "captured": 0}
        lock = threading.Lock()

        def retry_events():
            for eid in event_ids[:10]:
                success = dlq.retry_event(eid, lambda data: None)
                if success:
                    with lock:
                        results["retried"] += 1

        def capture_more():
            for i in range(10):
                dlq.capture(
                    event_type="new.event",
                    event_data={},
                    handler_name="h2",
                    error=_make_error(),
                )
                with lock:
                    results["captured"] += 1

        t1 = threading.Thread(target=retry_events)
        t2 = threading.Thread(target=capture_more)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert results["retried"] == 10
        assert results["captured"] == 10


# ===========================================================================
# Test: Empty queue handling
# ===========================================================================


class TestEmptyQueueHandling:
    """Tests for operations on an empty DLQ."""

    def test_get_pending_on_empty_queue(self, dlq: EventDLQ):
        """Getting pending events from empty queue returns empty list."""
        assert dlq.get_pending_events() == []

    def test_get_event_on_empty_queue(self, dlq: EventDLQ):
        """Getting a specific event from empty queue returns None."""
        assert dlq.get_event("nonexistent") is None

    def test_retry_on_empty_queue(self, dlq: EventDLQ):
        """Retrying on empty queue returns False."""
        assert dlq.retry_event("nonexistent", MagicMock()) is False

    def test_discard_on_empty_queue(self, dlq: EventDLQ):
        """Discarding on empty queue returns False."""
        assert dlq.discard_event("nonexistent") is False

    def test_stats_on_empty_queue(self, dlq: EventDLQ):
        """Stats on empty queue should show zeros."""
        stats = dlq.get_stats()
        assert stats.total_events == 0
        assert stats.pending_events == 0
        assert stats.recovered_events == 0
        assert stats.discarded_events == 0
        assert stats.events_by_handler == {}
        assert stats.events_by_type == {}
        assert stats.oldest_event_age_seconds is None

    def test_cleanup_on_empty_queue(self, dlq: EventDLQ):
        """Cleanup on empty queue removes nothing."""
        assert dlq.cleanup() == 0

    def test_pending_count_on_empty_queue(self, dlq: EventDLQ):
        """Pending count on empty queue should be 0."""
        # Force cache refresh
        dlq._cache_updated_at = 0
        assert dlq.pending_count == 0


# ===========================================================================
# Test: Event filtering (by handler name, by type)
# ===========================================================================


class TestEventFiltering:
    """Tests for querying events by handler and type."""

    def test_get_pending_filtered_by_handler(self, dlq: EventDLQ):
        """Should filter pending events by handler name."""
        dlq.capture(event_type="a", event_data={}, handler_name="handler_a", error=_make_error())
        dlq.capture(event_type="b", event_data={}, handler_name="handler_b", error=_make_error())
        dlq.capture(event_type="c", event_data={}, handler_name="handler_a", error=_make_error())

        filtered = dlq.get_pending_events(handler_name="handler_a")
        assert len(filtered) == 2
        assert all(e.handler_name == "handler_a" for e in filtered)

    def test_get_events_by_handler(self, dlq: EventDLQ):
        """Should return all events (any status) for a given handler."""
        e1 = dlq.capture(
            event_type="a", event_data={}, handler_name="my_handler", error=_make_error()
        )
        dlq.capture(
            event_type="b", event_data={}, handler_name="other_handler", error=_make_error()
        )
        e3 = dlq.capture(
            event_type="c", event_data={}, handler_name="my_handler", error=_make_error()
        )

        # Recover one of them
        dlq.retry_event(e1.id, MagicMock())

        events = dlq.get_events_by_handler("my_handler")
        assert len(events) == 2

    def test_get_pending_with_limit(self, dlq: EventDLQ):
        """get_pending_events should respect the limit parameter."""
        for i in range(10):
            dlq.capture(event_type="test", event_data={}, handler_name="h", error=_make_error())

        limited = dlq.get_pending_events(limit=3)
        assert len(limited) == 3


# ===========================================================================
# Test: Dead letter queue statistics
# ===========================================================================


class TestDLQStatistics:
    """Tests for DLQ statistics computation."""

    def test_stats_counts_by_status(self, dlq: EventDLQ):
        """Stats should correctly count events by status."""
        e1 = dlq.capture(event_type="a", event_data={}, handler_name="h1", error=_make_error())
        e2 = dlq.capture(event_type="b", event_data={}, handler_name="h2", error=_make_error())
        dlq.capture(event_type="c", event_data={}, handler_name="h1", error=_make_error())

        # Recover e1
        dlq.retry_event(e1.id, MagicMock())
        # Discard e2
        dlq.discard_event(e2.id)

        stats = dlq.get_stats()
        assert stats.total_events == 3
        assert stats.pending_events == 1
        assert stats.recovered_events == 1
        assert stats.discarded_events == 1

    def test_stats_events_by_handler(self, dlq: EventDLQ):
        """Stats should break down pending events by handler name."""
        dlq.capture(event_type="a", event_data={}, handler_name="handler_x", error=_make_error())
        dlq.capture(event_type="b", event_data={}, handler_name="handler_x", error=_make_error())
        dlq.capture(event_type="c", event_data={}, handler_name="handler_y", error=_make_error())

        stats = dlq.get_stats()
        assert stats.events_by_handler["handler_x"] == 2
        assert stats.events_by_handler["handler_y"] == 1

    def test_stats_events_by_type(self, dlq: EventDLQ):
        """Stats should break down pending events by event type."""
        dlq.capture(
            event_type="debate.completed",
            event_data={},
            handler_name="h",
            error=_make_error(),
        )
        dlq.capture(
            event_type="debate.completed",
            event_data={},
            handler_name="h",
            error=_make_error(),
        )
        dlq.capture(
            event_type="round.failed",
            event_data={},
            handler_name="h",
            error=_make_error(),
        )

        stats = dlq.get_stats()
        assert stats.events_by_type["debate.completed"] == 2
        assert stats.events_by_type["round.failed"] == 1

    def test_stats_age_tracking(self, dlq: EventDLQ):
        """Stats should track oldest and newest event ages."""
        dlq.capture(event_type="a", event_data={}, handler_name="h", error=_make_error())
        time.sleep(0.02)
        dlq.capture(event_type="b", event_data={}, handler_name="h", error=_make_error())

        stats = dlq.get_stats()
        assert stats.oldest_event_age_seconds is not None
        assert stats.newest_event_age_seconds is not None
        assert stats.oldest_event_age_seconds >= stats.newest_event_age_seconds

    def test_stats_to_dict(self, dlq: EventDLQ):
        """DLQStats.to_dict should produce a serializable dictionary."""
        dlq.capture(event_type="a", event_data={}, handler_name="h", error=_make_error())
        stats = dlq.get_stats()
        d = stats.to_dict()

        assert isinstance(d, dict)
        assert "total_events" in d
        assert "pending_events" in d
        assert "events_by_handler" in d
        assert "events_by_type" in d


# ===========================================================================
# Test: Discard event
# ===========================================================================


class TestDiscardEvent:
    """Tests for discarding events from the DLQ."""

    def test_discard_marks_event(self, dlq: EventDLQ):
        """Discarding an event should mark it as DISCARDED."""
        event = dlq.capture(
            event_type="test.event",
            event_data={},
            handler_name="h",
            error=_make_error(),
        )

        result = dlq.discard_event(event.id)
        assert result is True

        discarded = dlq.get_event(event.id)
        assert discarded is not None
        assert discarded.status == FailedEventStatus.DISCARDED

    def test_discard_reduces_pending_count(self, dlq: EventDLQ):
        """Discarding should reduce the pending count."""
        dlq.capture(
            event_type="test.event",
            event_data={},
            handler_name="h",
            error=_make_error(),
        )

        # Force cache refresh
        dlq._cache_updated_at = 0
        before = dlq.pending_count

        pending = dlq.get_pending_events()
        dlq.discard_event(pending[0].id)

        dlq._cache_updated_at = 0
        after = dlq.pending_count

        assert after == before - 1


# ===========================================================================
# Test: FailedEvent serialization
# ===========================================================================


class TestFailedEventSerialization:
    """Tests for FailedEvent to_dict / from_dict round-trip."""

    def test_to_dict_round_trip(self):
        """to_dict and from_dict should produce equivalent objects."""
        event = FailedEvent(
            id="test-id",
            event_type="test.type",
            event_data={"key": "value"},
            handler_name="handler",
            error_message="something failed",
            error_type="ValueError",
            retry_count=3,
            status=FailedEventStatus.PENDING,
            created_at=1000.0,
            updated_at=1001.0,
            original_timestamp=999.0,
            trace_id="trace-1",
            correlation_id="corr-1",
            metadata={"extra": "info"},
        )

        d = event.to_dict()
        restored = FailedEvent.from_dict(d)

        assert restored.id == event.id
        assert restored.event_type == event.event_type
        assert restored.event_data == event.event_data
        assert restored.handler_name == event.handler_name
        assert restored.error_message == event.error_message
        assert restored.error_type == event.error_type
        assert restored.retry_count == event.retry_count
        assert restored.status == event.status
        assert restored.created_at == event.created_at
        assert restored.updated_at == event.updated_at
        assert restored.original_timestamp == event.original_timestamp
        assert restored.trace_id == event.trace_id
        assert restored.correlation_id == event.correlation_id
        assert restored.metadata == event.metadata

    def test_from_dict_with_minimal_data(self):
        """from_dict should handle minimal required fields with defaults."""
        d = {
            "id": "min-id",
            "event_type": "test",
            "handler_name": "h",
            "error_message": "err",
            "created_at": 1000.0,
        }
        event = FailedEvent.from_dict(d)
        assert event.id == "min-id"
        assert event.retry_count == 0
        assert event.status == FailedEventStatus.PENDING
        assert event.error_type == "Exception"
        assert event.metadata == {}


# ===========================================================================
# Test: Global singleton management
# ===========================================================================


class TestGlobalSingleton:
    """Tests for the global DLQ singleton pattern."""

    def test_get_event_dlq_returns_same_instance(self):
        """get_event_dlq should return the same instance on repeated calls."""
        with patch("aragora.events.dead_letter_queue.EventDLQPersistence"):
            dlq1 = get_event_dlq()
            dlq2 = get_event_dlq()
            assert dlq1 is dlq2

    def test_reset_clears_singleton(self):
        """reset_event_dlq should clear the global instance."""
        with patch("aragora.events.dead_letter_queue.EventDLQPersistence"):
            dlq1 = get_event_dlq()
            reset_event_dlq()
            dlq2 = get_event_dlq()
            assert dlq1 is not dlq2


# ===========================================================================
# Test: Persistence layer directly
# ===========================================================================


class TestEventDLQPersistence:
    """Tests for the SQLite persistence layer."""

    @pytest.fixture()
    def persistence(self, tmp_path):
        db_path = str(tmp_path / "persist_test.db")
        return EventDLQPersistence(db_path=db_path)

    def test_save_and_get(self, persistence: EventDLQPersistence):
        """Save and retrieve a failed event."""
        now = time.time()
        event = FailedEvent(
            id="p-1",
            event_type="test",
            event_data={"k": "v"},
            handler_name="h",
            error_message="err",
            error_type="ValueError",
            retry_count=0,
            status=FailedEventStatus.PENDING,
            created_at=now,
            updated_at=now,
            original_timestamp=now,
        )
        persistence.save(event)

        retrieved = persistence.get("p-1")
        assert retrieved is not None
        assert retrieved.id == "p-1"
        assert retrieved.event_data == {"k": "v"}

    def test_get_nonexistent(self, persistence: EventDLQPersistence):
        """Getting a nonexistent event returns None."""
        assert persistence.get("does-not-exist") is None

    def test_delete(self, persistence: EventDLQPersistence):
        """Delete should remove an event."""
        now = time.time()
        event = FailedEvent(
            id="del-1",
            event_type="test",
            event_data={},
            handler_name="h",
            error_message="err",
            error_type="ValueError",
            retry_count=0,
            status=FailedEventStatus.PENDING,
            created_at=now,
            updated_at=now,
            original_timestamp=now,
        )
        persistence.save(event)

        assert persistence.delete("del-1") is True
        assert persistence.get("del-1") is None

    def test_delete_nonexistent(self, persistence: EventDLQPersistence):
        """Deleting a nonexistent event returns False."""
        assert persistence.delete("nope") is False

    def test_update_status(self, persistence: EventDLQPersistence):
        """update_status should change event status."""
        now = time.time()
        event = FailedEvent(
            id="st-1",
            event_type="test",
            event_data={},
            handler_name="h",
            error_message="err",
            error_type="ValueError",
            retry_count=0,
            status=FailedEventStatus.PENDING,
            created_at=now,
            updated_at=now,
            original_timestamp=now,
        )
        persistence.save(event)

        result = persistence.update_status("st-1", FailedEventStatus.RECOVERED)
        assert result is True

        updated = persistence.get("st-1")
        assert updated is not None
        assert updated.status == FailedEventStatus.RECOVERED

    def test_count_pending(self, persistence: EventDLQPersistence):
        """count_pending should return correct count."""
        now = time.time()
        for i in range(5):
            event = FailedEvent(
                id=f"cnt-{i}",
                event_type="test",
                event_data={},
                handler_name="h",
                error_message="err",
                error_type="ValueError",
                retry_count=0,
                status=FailedEventStatus.PENDING,
                created_at=now,
                updated_at=now,
                original_timestamp=now,
            )
            persistence.save(event)

        # Mark 2 as recovered
        persistence.update_status("cnt-0", FailedEventStatus.RECOVERED)
        persistence.update_status("cnt-1", FailedEventStatus.RECOVERED)

        assert persistence.count_pending() == 3
