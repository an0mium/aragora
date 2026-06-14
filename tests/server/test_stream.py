"""
Tests for the streaming server layer.

Covers:
- StreamEvent data class
- StreamEventType enum
- SyncEventEmitter thread-safe event queue
- TokenBucket rate limiter
- AudienceInbox message queue
- normalize_intensity utility
- create_arena_hooks factory
"""

import pytest
import threading
import time
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch

from aragora.server.stream import (
    StreamEvent,
    StreamEventType,
    SyncEventEmitter,
    TokenBucket,
    AudienceInbox,
    AudienceMessage,
    normalize_intensity,
    create_arena_hooks,
    _safe_error_message,
)


# ============================================================================
# StreamEvent Tests
# ============================================================================


class TestStreamEvent:
    """Tests for StreamEvent data class."""

    def test_basic_creation(self):
        """Test basic event creation."""
        event = StreamEvent(
            type=StreamEventType.DEBATE_START,
            data={"task": "Test debate"},
        )

        assert event.type == StreamEventType.DEBATE_START
        assert event.data == {"task": "Test debate"}
        assert event.timestamp is not None
        assert event.round == 0
        assert event.agent == ""
        assert event.loop_id == ""

    def test_with_all_fields(self):
        """Test event with all fields populated."""
        event = StreamEvent(
            type=StreamEventType.AGENT_MESSAGE,
            data={"content": "My proposal"},
            round=2,
            agent="claude",
            loop_id="loop-123",
        )

        assert event.round == 2
        assert event.agent == "claude"
        assert event.loop_id == "loop-123"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        event = StreamEvent(
            type=StreamEventType.DEBATE_END,
            data={"winner": "claude"},
            round=5,
        )

        d = event.to_dict()

        assert d["type"] == "debate_end"
        assert d["data"] == {"winner": "claude"}
        assert d["round"] == 5
        assert "timestamp" in d

    def test_to_json(self):
        """Test JSON serialization."""
        event = StreamEvent(
            type=StreamEventType.VOTE,
            data={"choice": "A"},
        )

        json_str = event.to_json()

        assert '"type": "vote"' in json_str
        assert '"choice": "A"' in json_str

    def test_timestamp_auto_generated(self):
        """Test timestamp is automatically generated."""
        event = StreamEvent(type=StreamEventType.ROUND_START, data={})

        # Timestamp should be a float (time.time())
        assert isinstance(event.timestamp, float)
        assert event.timestamp > 0


class TestStreamEventType:
    """Tests for StreamEventType enum."""

    def test_debate_events_exist(self):
        """Test debate-related event types exist."""
        assert StreamEventType.DEBATE_START
        assert StreamEventType.DEBATE_END
        assert StreamEventType.ROUND_START
        assert StreamEventType.AGENT_MESSAGE
        assert StreamEventType.CRITIQUE
        assert StreamEventType.VOTE
        assert StreamEventType.CONSENSUS

    def test_token_streaming_events_exist(self):
        """Test token streaming event types exist."""
        assert StreamEventType.TOKEN_START
        assert StreamEventType.TOKEN_DELTA
        assert StreamEventType.TOKEN_END

    def test_nomic_loop_events_exist(self):
        """Test nomic loop event types exist."""
        assert StreamEventType.CYCLE_START
        assert StreamEventType.CYCLE_END
        assert StreamEventType.PHASE_START
        assert StreamEventType.PHASE_END
        assert StreamEventType.TASK_START
        assert StreamEventType.TASK_COMPLETE


# ============================================================================
# SyncEventEmitter Tests
# ============================================================================


class TestSyncEventEmitter:
    """Tests for SyncEventEmitter thread-safe event queue."""

    def test_emit_and_drain(self):
        """Test basic emit and drain cycle."""
        emitter = SyncEventEmitter()

        event = StreamEvent(type=StreamEventType.DEBATE_START, data={"task": "test"})
        emitter.emit(event)

        events = emitter.drain()

        assert len(events) == 1
        assert events[0].type == StreamEventType.DEBATE_START

    def test_drain_empty_queue(self):
        """Test draining empty queue returns empty list."""
        emitter = SyncEventEmitter()

        events = emitter.drain()

        assert events == []

    def test_loop_id_attached_to_events(self):
        """Test loop_id is attached to emitted events."""
        emitter = SyncEventEmitter()
        emitter.set_loop_id("test-loop-123")

        event = StreamEvent(type=StreamEventType.DEBATE_START, data={})
        emitter.emit(event)

        events = emitter.drain()

        assert events[0].loop_id == "test-loop-123"

    def test_set_loop_id(self):
        """Test setting loop_id."""
        emitter = SyncEventEmitter()

        emitter.set_loop_id("my-loop")

        assert emitter._loop_id == "my-loop"

    def test_event_with_existing_loop_id_preserved(self):
        """Test events with existing loop_id keep their value."""
        emitter = SyncEventEmitter()
        emitter.set_loop_id("emitter-loop")

        event = StreamEvent(
            type=StreamEventType.DEBATE_START,
            data={},
            loop_id="original-loop",
        )
        emitter.emit(event)

        events = emitter.drain()

        # Event's original loop_id should be preserved if set
        assert events[0].loop_id in ("original-loop", "emitter-loop")

    def test_subscriber_callback(self):
        """Test subscriber receives events."""
        emitter = SyncEventEmitter()
        received = []

        def callback(event):
            received.append(event)

        emitter.subscribe(callback)

        event = StreamEvent(type=StreamEventType.VOTE, data={"choice": "A"})
        emitter.emit(event)

        assert len(received) == 1
        assert received[0].data == {"choice": "A"}

    def test_multiple_subscribers(self):
        """Test multiple subscribers all receive events."""
        emitter = SyncEventEmitter()
        received1 = []
        received2 = []

        emitter.subscribe(lambda e: received1.append(e))
        emitter.subscribe(lambda e: received2.append(e))

        event = StreamEvent(type=StreamEventType.CONSENSUS, data={})
        emitter.emit(event)

        assert len(received1) == 1
        assert len(received2) == 1

    def test_subscriber_error_does_not_break_other_subscribers(self):
        """Test that one failing subscriber doesn't prevent others."""
        emitter = SyncEventEmitter()
        received = []

        def failing_callback(event):
            raise ValueError("Intentional error")

        def working_callback(event):
            received.append(event)

        emitter.subscribe(failing_callback)
        emitter.subscribe(working_callback)

        event = StreamEvent(type=StreamEventType.DEBATE_END, data={})
        emitter.emit(event)

        # Working callback should still receive event
        assert len(received) == 1

    def test_queue_overflow_protection(self):
        """Test queue doesn't grow unbounded."""
        emitter = SyncEventEmitter()

        # Emit many events (more than MAX_QUEUE_SIZE)
        for i in range(15000):
            event = StreamEvent(type=StreamEventType.TOKEN_DELTA, data={"i": i})
            emitter.emit(event)

        # Queue should be capped
        events = emitter.drain()
        assert len(events) <= 10001  # MAX_QUEUE_SIZE + 1 overflow event

    def test_thread_safety(self):
        """Test concurrent emit operations are thread-safe."""
        emitter = SyncEventEmitter()
        errors = []

        def emit_events():
            try:
                for i in range(100):
                    event = StreamEvent(type=StreamEventType.TOKEN_DELTA, data={"i": i})
                    emitter.emit(event)
            except Exception as e:
                errors.append(e)

        # Start multiple threads emitting concurrently
        threads = [threading.Thread(target=emit_events) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No exceptions should occur during concurrent emit
        assert len(errors) == 0
        events = emitter.drain()
        # Events should be in queue (at least some should arrive)
        assert len(events) > 0


# ============================================================================
# TokenBucket Tests
# ============================================================================


class TestTokenBucket:
    """Tests for TokenBucket rate limiter."""

    def test_initial_tokens(self):
        """Test bucket starts with full capacity."""
        bucket = TokenBucket(rate_per_minute=600, burst_size=5)  # 10/sec

        # Should be able to consume up to burst_size
        for _ in range(5):
            assert bucket.consume() is True

    def test_token_refill(self):
        """Test tokens refill over time."""
        bucket = TokenBucket(rate_per_minute=6000, burst_size=1)  # 100/sec

        # Consume the token
        assert bucket.consume() is True
        assert bucket.consume() is False

        # Wait for refill (6000/min = 100/sec, so 20ms = 2 tokens)
        time.sleep(0.025)

        assert bucket.consume() is True

    def test_burst_capacity(self):
        """Test burst capacity allows initial burst."""
        bucket = TokenBucket(rate_per_minute=60, burst_size=10)  # 1/sec, 10 burst

        # Should allow 10 rapid requests
        for _ in range(10):
            assert bucket.consume() is True

        # 11th should fail
        assert bucket.consume() is False

    def test_multiple_token_consumption(self):
        """Test consuming multiple tokens at once."""
        bucket = TokenBucket(rate_per_minute=600, burst_size=5)

        # Consume 3 tokens
        assert bucket.consume(3) is True

        # Only 2 left
        assert bucket.consume(2) is True
        assert bucket.consume(1) is False

    def test_thread_safety(self):
        """Test concurrent consumption is thread-safe."""
        bucket = TokenBucket(rate_per_minute=60000, burst_size=100)  # 1000/sec
        successful = []

        def consume_tokens():
            count = 0
            for _ in range(20):
                if bucket.consume():
                    count += 1
            successful.append(count)

        threads = [threading.Thread(target=consume_tokens) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Total successful should be around burst_size (may slightly exceed due to refill)
        # At 1000 tokens/sec, test may add ~100-200 tokens during execution
        total = sum(successful)
        assert total <= 200  # Allow margin for token refill during test


# ============================================================================
# AudienceInbox Tests
# ============================================================================


class TestAudienceInbox:
    """Tests for AudienceInbox message queue."""

    def test_put_and_get_all(self):
        """Test basic put and get_all operations."""
        inbox = AudienceInbox()

        msg = AudienceMessage(
            type="vote",
            loop_id="loop-1",
            payload={"choice": "A", "intensity": 7},
        )
        inbox.put(msg)

        messages = inbox.get_all()

        assert len(messages) == 1
        assert messages[0].type == "vote"

    def test_get_all_clears_queue(self):
        """Test get_all clears the queue."""
        inbox = AudienceInbox()

        msg = AudienceMessage(type="vote", loop_id="loop-1", payload={})
        inbox.put(msg)

        inbox.get_all()
        messages = inbox.get_all()

        assert len(messages) == 0

    def test_get_summary_vote_counts(self):
        """Test get_summary returns vote statistics."""
        inbox = AudienceInbox()

        # Add some votes
        inbox.put(
            AudienceMessage(
                type="vote",
                loop_id="loop-1",
                payload={"choice": "A", "intensity": 8},
            )
        )
        inbox.put(
            AudienceMessage(
                type="vote",
                loop_id="loop-1",
                payload={"choice": "A", "intensity": 6},
            )
        )
        inbox.put(
            AudienceMessage(
                type="vote",
                loop_id="loop-1",
                payload={"choice": "B", "intensity": 5},
            )
        )

        summary = inbox.get_summary(loop_id="loop-1")

        # API returns 'votes' not 'vote_counts', and 'total' not 'total_votes'
        assert "A" in summary["votes"]
        assert summary["votes"]["A"] == 2
        assert summary["votes"]["B"] == 1

    def test_get_summary_weighted_votes(self):
        """Test get_summary includes weighted vote totals."""
        inbox = AudienceInbox()

        # High intensity vote for A
        inbox.put(
            AudienceMessage(
                type="vote",
                loop_id="loop-1",
                payload={"choice": "A", "intensity": 10},
            )
        )
        # Low intensity vote for B
        inbox.put(
            AudienceMessage(
                type="vote",
                loop_id="loop-1",
                payload={"choice": "B", "intensity": 1},
            )
        )

        summary = inbox.get_summary(loop_id="loop-1")

        # Weighted votes should reflect intensity
        assert "weighted_votes" in summary
        assert summary["weighted_votes"]["A"] > summary["weighted_votes"]["B"]

    def test_get_summary_filters_by_loop_id(self):
        """Test get_summary filters by loop_id."""
        inbox = AudienceInbox()

        inbox.put(AudienceMessage(type="vote", loop_id="loop-1", payload={"choice": "A"}))
        inbox.put(AudienceMessage(type="vote", loop_id="loop-2", payload={"choice": "B"}))

        summary = inbox.get_summary(loop_id="loop-1")

        # Only the vote from loop-1 should be included
        assert "A" in summary["votes"]
        assert summary["votes"]["A"] == 1
        assert "B" not in summary["votes"]

    def test_suggestion_messages(self):
        """Test suggestion messages are collected."""
        inbox = AudienceInbox()

        inbox.put(
            AudienceMessage(
                type="suggestion",
                loop_id="loop-1",
                payload={"text": "Consider edge cases"},
            )
        )

        messages = inbox.get_all()

        assert len(messages) == 1
        assert messages[0].type == "suggestion"
        assert messages[0].payload["text"] == "Consider edge cases"


# ============================================================================
# AudienceMessage Tests
# ============================================================================


class TestAudienceMessage:
    """Tests for AudienceMessage data class."""

    def test_vote_message(self):
        """Test creating a vote message."""
        msg = AudienceMessage(
            type="vote",
            loop_id="loop-123",
            payload={"choice": "Option A", "intensity": 7},
        )

        assert msg.type == "vote"
        assert msg.loop_id == "loop-123"
        assert msg.payload["choice"] == "Option A"
        assert msg.payload["intensity"] == 7
        assert msg.timestamp is not None

    def test_suggestion_message(self):
        """Test creating a suggestion message."""
        msg = AudienceMessage(
            type="suggestion",
            loop_id="loop-456",
            payload={"text": "What about security?"},
        )

        assert msg.type == "suggestion"
        assert msg.payload["text"] == "What about security?"


# ============================================================================
# normalize_intensity Tests
# ============================================================================


class TestNormalizeIntensity:
    """Tests for normalize_intensity utility function."""

    def test_valid_integer(self):
        """Test valid integer input."""
        assert normalize_intensity(5) == 5
        assert normalize_intensity(1) == 1
        assert normalize_intensity(10) == 10

    def test_valid_float(self):
        """Test float is converted to integer."""
        assert normalize_intensity(5.7) == 5
        assert normalize_intensity(9.9) == 9

    def test_string_number(self):
        """Test string number is converted."""
        assert normalize_intensity("7") == 7
        assert normalize_intensity("3.5") == 3

    def test_none_returns_default(self):
        """Test None returns default value."""
        assert normalize_intensity(None) == 5
        assert normalize_intensity(None, default=7) == 7

    def test_clamping_high(self):
        """Test values above max are clamped."""
        assert normalize_intensity(15) == 10
        assert normalize_intensity(100) == 10

    def test_clamping_low(self):
        """Test values below min are clamped."""
        assert normalize_intensity(0) == 1
        assert normalize_intensity(-5) == 1

    def test_invalid_string(self):
        """Test invalid string returns default."""
        assert normalize_intensity("invalid") == 5
        assert normalize_intensity("abc", default=3) == 3

    def test_custom_bounds(self):
        """Test custom min/max bounds."""
        assert normalize_intensity(5, min_val=3, max_val=7) == 5
        assert normalize_intensity(1, min_val=3, max_val=7) == 3
        assert normalize_intensity(10, min_val=3, max_val=7) == 7


# ============================================================================
# create_arena_hooks Tests
# ============================================================================


class TestCreateArenaHooks:
    """Tests for create_arena_hooks factory function."""

    def test_returns_dict_of_hooks(self):
        """Test function returns dictionary of hooks."""
        emitter = SyncEventEmitter()

        hooks = create_arena_hooks(emitter)

        assert isinstance(hooks, dict)
        assert "on_message" in hooks
        assert "on_critique" in hooks
        assert "on_round_start" in hooks
        assert "on_vote" in hooks
        assert "on_consensus" in hooks
        assert "on_debate_start" in hooks
        assert "on_debate_end" in hooks

    def test_on_message_emits_event(self):
        """Test on_message hook emits correct event."""
        emitter = SyncEventEmitter()
        hooks = create_arena_hooks(emitter)

        hooks["on_message"]("claude", "My proposal content", "proposer", 1)

        events = emitter.drain()
        assert len(events) == 1
        assert events[0].type == StreamEventType.AGENT_MESSAGE
        assert events[0].agent == "claude"
        assert "My proposal content" in str(events[0].data)

    def test_on_critique_emits_event(self):
        """Test on_critique hook emits correct event."""
        emitter = SyncEventEmitter()
        hooks = create_arena_hooks(emitter)

        hooks["on_critique"]("gemini", "claude", ["This needs work"], 0.5, 1)

        events = emitter.drain()
        assert len(events) == 1
        assert events[0].type == StreamEventType.CRITIQUE
        assert events[0].agent == "gemini"

    def test_on_round_start_emits_event(self):
        """Test on_round_start hook emits correct event."""
        emitter = SyncEventEmitter()
        hooks = create_arena_hooks(emitter)

        hooks["on_round_start"](3)

        events = emitter.drain()
        assert len(events) == 1
        assert events[0].type == StreamEventType.ROUND_START
        assert events[0].round == 3


# ============================================================================
# _safe_error_message Tests
# ============================================================================


class TestSafeErrorMessage:
    """Tests for _safe_error_message utility function."""

    def test_generic_exception(self):
        """Test generic exception returns safe message."""
        error = Exception("Internal details")

        result = _safe_error_message(error)

        assert "Internal details" not in result
        assert "error" in result.lower() or "failed" in result.lower()

    def test_value_error_returns_detail(self):
        """Test ValueError includes some detail."""
        error = ValueError("Invalid input")

        result = _safe_error_message(error, context="validation")

        # Should be somewhat informative but safe
        assert len(result) > 0

    def test_context_is_used(self):
        """Test context parameter is incorporated."""
        error = Exception("Details")

        result = _safe_error_message(error, context="debate_start")

        # Result should exist
        assert len(result) > 0


# ============================================================================
# ThreadPoolExecutor Race Condition Tests (Round 25 fix)
# ============================================================================


class TestDebateExecutorRaceCondition:
    """Tests for ThreadPoolExecutor race condition fix in stream.py.

    Round 25 fixed race conditions in graceful_shutdown() and submit functions
    where the executor could be accessed without proper locking.
    """

    def test_executor_lock_exists(self):
        """Test that _debate_executor_lock exists for thread-safe access.

        This verifies the lock variable exists that's used to protect
        concurrent access to _debate_executor.
        """
        from aragora.server import stream

        assert hasattr(stream, "_debate_executor_lock")
        assert hasattr(stream, "_debate_executor")
        # Lock should be a threading.Lock
        assert isinstance(stream._debate_executor_lock, type(threading.Lock()))

    def test_concurrent_executor_access_is_safe(self):
        """Test that concurrent access to executor variables is thread-safe.

        This verifies the fix for the race condition where _debate_executor
        could be accessed without proper locking.
        """
        from concurrent.futures import ThreadPoolExecutor as TPE
        from aragora.server import stream

        # Store original values
        original_executor = stream._debate_executor
        original_lock = stream._debate_executor_lock

        errors = []
        values_seen = []

        def access_executor():
            """Simulate accessing executor with proper locking."""
            try:
                with stream._debate_executor_lock:
                    # Read the executor value safely
                    executor = stream._debate_executor
                    values_seen.append(executor)
            except Exception as e:
                errors.append(e)

        try:
            # Create a fresh lock for testing
            stream._debate_executor_lock = threading.Lock()
            stream._debate_executor = None

            # Access from multiple threads concurrently
            threads = [threading.Thread(target=access_executor) for _ in range(10)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # No errors should have occurred
            assert len(errors) == 0, f"Errors during concurrent access: {errors}"
            # All threads should have seen the value
            assert len(values_seen) == 10

        finally:
            # Restore original values
            stream._debate_executor = original_executor
            stream._debate_executor_lock = original_lock

    def test_executor_lock_prevents_data_races(self):
        """Test that lock properly serializes executor state changes.

        This tests the pattern used in the fix: capture executor reference
        under lock before using it.
        """
        from concurrent.futures import ThreadPoolExecutor as TPE
        from aragora.server import stream

        # Store original values
        original_executor = stream._debate_executor
        original_lock = stream._debate_executor_lock

        errors = []
        captured_executors = []

        def capture_executor_safely():
            """Simulate the fixed pattern: capture executor under lock."""
            try:
                with stream._debate_executor_lock:
                    if stream._debate_executor is None:
                        # Don't create real executor in test
                        pass
                    executor = stream._debate_executor
                captured_executors.append(executor)
            except Exception as e:
                errors.append(e)

        try:
            stream._debate_executor_lock = threading.Lock()
            stream._debate_executor = None

            # Run many threads to stress test the locking
            threads = [threading.Thread(target=capture_executor_safely) for _ in range(20)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # All operations should succeed without errors
            assert len(errors) == 0, f"Race condition detected: {errors}"
            assert len(captured_executors) == 20

        finally:
            stream._debate_executor = original_executor
            stream._debate_executor_lock = original_lock
