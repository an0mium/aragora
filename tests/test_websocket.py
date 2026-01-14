"""
Tests for WebSocket streaming functionality.

Tests cover:
- StreamEvent serialization
- SyncEventEmitter event queuing and subscription
- TokenBucket rate limiting
- AudienceMessage handling
"""

import json
import pytest
import time
import threading
from unittest.mock import Mock, patch

from aragora.server.stream import (
    StreamEvent,
    StreamEventType,
    SyncEventEmitter,
    TokenBucket,
    AudienceInbox,
    AudienceMessage,
    normalize_intensity,
)


class TestStreamEvent:
    """Tests for StreamEvent dataclass."""

    def test_basic_event_creation(self):
        """Test creating a basic stream event."""
        event = StreamEvent(
            type=StreamEventType.DEBATE_START,
            data={"task": "Test debate", "agents": ["claude", "gemini"]},
            round=0,
            agent="",
        )

        assert event.type == StreamEventType.DEBATE_START
        assert event.data["task"] == "Test debate"
        assert event.round == 0
        assert event.timestamp > 0

    def test_event_to_dict(self):
        """Test event serialization to dictionary."""
        event = StreamEvent(
            type=StreamEventType.AGENT_MESSAGE,
            data={"content": "Hello world"},
            round=1,
            agent="claude",
            loop_id="loop-123",
        )

        result = event.to_dict()

        assert result["type"] == "agent_message"
        assert result["data"]["content"] == "Hello world"
        assert result["round"] == 1
        assert result["agent"] == "claude"
        assert result["loop_id"] == "loop-123"

    def test_event_to_json(self):
        """Test event serialization to JSON string."""
        event = StreamEvent(
            type=StreamEventType.VOTE,
            data={"choice": "option1", "weight": 0.8},
            round=2,
            agent="gemini",
        )

        json_str = event.to_json()
        parsed = json.loads(json_str)

        assert parsed["type"] == "vote"
        assert parsed["data"]["choice"] == "option1"

    def test_event_without_loop_id(self):
        """Test that loop_id is omitted when empty."""
        event = StreamEvent(
            type=StreamEventType.CONSENSUS,
            data={"reached": True},
            loop_id="",
        )

        result = event.to_dict()
        assert "loop_id" not in result

    def test_all_event_types_serializable(self):
        """Verify all event types can be serialized."""
        for event_type in StreamEventType:
            event = StreamEvent(
                type=event_type,
                data={"test": True},
            )
            # Should not raise
            json_str = event.to_json()
            parsed = json.loads(json_str)
            assert parsed["type"] == event_type.value


class TestSyncEventEmitter:
    """Tests for SyncEventEmitter."""

    def test_emit_and_drain(self):
        """Test basic event emission and draining."""
        emitter = SyncEventEmitter()

        event1 = StreamEvent(type=StreamEventType.DEBATE_START, data={"id": 1})
        event2 = StreamEvent(type=StreamEventType.ROUND_START, data={"id": 2})

        emitter.emit(event1)
        emitter.emit(event2)

        events = emitter.drain()

        assert len(events) == 2
        assert events[0].data["id"] == 1
        assert events[1].data["id"] == 2

    def test_drain_respects_batch_size(self):
        """Test that drain respects max_batch_size."""
        emitter = SyncEventEmitter()

        for i in range(10):
            emitter.emit(StreamEvent(type=StreamEventType.LOG_MESSAGE, data={"i": i}))

        batch1 = emitter.drain(max_batch_size=3)
        batch2 = emitter.drain(max_batch_size=3)

        assert len(batch1) == 3
        assert len(batch2) == 3

    def test_drain_empty_queue(self):
        """Test draining an empty queue returns empty list."""
        emitter = SyncEventEmitter()
        events = emitter.drain()
        assert events == []

    def test_loop_id_attached_to_events(self):
        """Test that loop_id is attached to emitted events."""
        emitter = SyncEventEmitter(loop_id="loop-abc")

        event = StreamEvent(type=StreamEventType.PHASE_START, data={})
        emitter.emit(event)

        events = emitter.drain()
        assert events[0].loop_id == "loop-abc"

    def test_set_loop_id(self):
        """Test changing loop_id."""
        emitter = SyncEventEmitter()
        emitter.set_loop_id("new-loop-id")

        event = StreamEvent(type=StreamEventType.CYCLE_START, data={})
        emitter.emit(event)

        events = emitter.drain()
        assert events[0].loop_id == "new-loop-id"

    def test_event_with_existing_loop_id_preserved(self):
        """Test that events with existing loop_id are not overwritten."""
        emitter = SyncEventEmitter(loop_id="default-loop")

        event = StreamEvent(
            type=StreamEventType.DEBATE_END,
            data={},
            loop_id="specific-loop",
        )
        emitter.emit(event)

        events = emitter.drain()
        assert events[0].loop_id == "specific-loop"

    def test_subscriber_callback(self):
        """Test that subscribers receive events."""
        emitter = SyncEventEmitter()
        received_events = []

        def callback(event):
            received_events.append(event)

        emitter.subscribe(callback)

        event = StreamEvent(type=StreamEventType.AGENT_MESSAGE, data={"msg": "test"})
        emitter.emit(event)

        assert len(received_events) == 1
        assert received_events[0].data["msg"] == "test"

    def test_multiple_subscribers(self):
        """Test multiple subscribers all receive events."""
        emitter = SyncEventEmitter()
        counts = {"a": 0, "b": 0}

        emitter.subscribe(lambda e: counts.__setitem__("a", counts["a"] + 1))
        emitter.subscribe(lambda e: counts.__setitem__("b", counts["b"] + 1))

        emitter.emit(StreamEvent(type=StreamEventType.VOTE, data={}))

        assert counts["a"] == 1
        assert counts["b"] == 1

    def test_subscriber_error_does_not_break_other_subscribers(self):
        """Test that one failing subscriber doesn't break others."""
        emitter = SyncEventEmitter()
        received = []

        def failing_callback(event):
            raise ValueError("Intentional error")

        def working_callback(event):
            received.append(event)

        emitter.subscribe(failing_callback)
        emitter.subscribe(working_callback)

        event = StreamEvent(type=StreamEventType.CRITIQUE, data={})
        emitter.emit(event)  # Should not raise

        assert len(received) == 1

    def test_queue_overflow_protection(self):
        """Test that queue size is bounded."""
        emitter = SyncEventEmitter()
        original_max = SyncEventEmitter.MAX_QUEUE_SIZE

        # Temporarily reduce max for testing
        SyncEventEmitter.MAX_QUEUE_SIZE = 5

        try:
            for i in range(10):
                emitter.emit(StreamEvent(type=StreamEventType.LOG_MESSAGE, data={"i": i}))

            events = emitter.drain()
            # Queue should have dropped oldest events
            assert len(events) == 5
            # Should have the newer events (5-9)
            assert events[-1].data["i"] == 9
        finally:
            SyncEventEmitter.MAX_QUEUE_SIZE = original_max

    def test_thread_safety(self):
        """Test that emitter is thread-safe."""
        emitter = SyncEventEmitter()
        errors = []

        def emit_events():
            try:
                for i in range(100):
                    emitter.emit(StreamEvent(type=StreamEventType.LOG_MESSAGE, data={"i": i}))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=emit_events) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # Drain all and verify count
        total = 0
        while True:
            batch = emitter.drain(max_batch_size=100)
            if not batch:
                break
            total += len(batch)
        assert total == 500


class TestTokenBucket:
    """Tests for TokenBucket rate limiter."""

    def test_initial_tokens(self):
        """Test bucket starts full."""
        bucket = TokenBucket(rate_per_minute=60, burst_size=10)

        # Should be able to consume burst_size tokens immediately
        for i in range(10):
            assert bucket.consume() is True

        # 11th should fail (bucket empty)
        assert bucket.consume() is False

    def test_token_refill(self):
        """Test tokens refill over time."""
        bucket = TokenBucket(rate_per_minute=6000, burst_size=1)  # 100/sec

        # Empty the bucket
        bucket.consume()
        assert bucket.consume() is False

        # Wait a bit for refill (at 100/sec, ~10ms should give 1 token)
        time.sleep(0.02)

        assert bucket.consume() is True

    def test_burst_capacity(self):
        """Test burst handling."""
        bucket = TokenBucket(rate_per_minute=60, burst_size=5)

        # Consume all burst capacity
        for _ in range(5):
            assert bucket.consume() is True

        # Bucket should be empty
        assert bucket.consume() is False

    def test_multiple_token_consumption(self):
        """Test consuming multiple tokens at once."""
        bucket = TokenBucket(rate_per_minute=60, burst_size=10)

        assert bucket.consume(5) is True
        assert bucket.consume(5) is True
        assert bucket.consume(1) is False

    def test_thread_safety(self):
        """Test token bucket is thread-safe."""
        # Use very low refill rate to avoid refills during test
        bucket = TokenBucket(rate_per_minute=1, burst_size=100)
        consumed = {"count": 0}
        lock = threading.Lock()

        def consume_tokens():
            for _ in range(50):
                if bucket.consume():
                    with lock:
                        consumed["count"] += 1

        threads = [threading.Thread(target=consume_tokens) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Should have consumed at most burst_size tokens (no more due to low refill)
        assert consumed["count"] <= 100
        # Should have consumed at least most of them
        assert consumed["count"] >= 95


class TestAudienceMessage:
    """Tests for AudienceMessage dataclass."""

    def test_vote_message(self):
        """Test creating a vote message."""
        msg = AudienceMessage(
            type="vote",
            loop_id="loop-123",
            payload={"choice": "option_a", "intensity": 8},
            user_id="user-456",
        )

        assert msg.type == "vote"
        assert msg.loop_id == "loop-123"
        assert msg.payload["choice"] == "option_a"
        assert msg.timestamp > 0

    def test_suggestion_message(self):
        """Test creating a suggestion message."""
        msg = AudienceMessage(
            type="suggestion",
            loop_id="loop-789",
            payload={"text": "Consider using async/await"},
        )

        assert msg.type == "suggestion"
        assert msg.payload["text"] == "Consider using async/await"


class TestNormalizeIntensity:
    """Tests for normalize_intensity function."""

    def test_valid_integer(self):
        """Test with valid integer input."""
        assert normalize_intensity(7) == 7

    def test_valid_float(self):
        """Test with valid float input."""
        assert normalize_intensity(7.8) == 7

    def test_string_number(self):
        """Test with string number."""
        assert normalize_intensity("8") == 8

    def test_none_returns_default(self):
        """Test None returns default."""
        assert normalize_intensity(None) == 5
        assert normalize_intensity(None, default=3) == 3

    def test_clamping_high(self):
        """Test values above max are clamped."""
        assert normalize_intensity(15) == 10
        assert normalize_intensity(100, max_val=10) == 10

    def test_clamping_low(self):
        """Test values below min are clamped."""
        assert normalize_intensity(-5) == 1
        assert normalize_intensity(0, min_val=1) == 1

    def test_invalid_string(self):
        """Test invalid string returns default."""
        assert normalize_intensity("not a number") == 5

    def test_custom_bounds(self):
        """Test custom min/max bounds."""
        assert normalize_intensity(50, min_val=0, max_val=100) == 50
        assert normalize_intensity(150, min_val=0, max_val=100) == 100


class TestEventTypeValues:
    """Verify event type values for API compatibility."""

    def test_debate_event_types(self):
        """Verify debate event type values."""
        assert StreamEventType.DEBATE_START.value == "debate_start"
        assert StreamEventType.ROUND_START.value == "round_start"
        assert StreamEventType.AGENT_MESSAGE.value == "agent_message"
        assert StreamEventType.CRITIQUE.value == "critique"
        assert StreamEventType.VOTE.value == "vote"
        assert StreamEventType.CONSENSUS.value == "consensus"
        assert StreamEventType.DEBATE_END.value == "debate_end"

    def test_token_streaming_types(self):
        """Verify token streaming event types."""
        assert StreamEventType.TOKEN_START.value == "token_start"
        assert StreamEventType.TOKEN_DELTA.value == "token_delta"
        assert StreamEventType.TOKEN_END.value == "token_end"

    def test_nomic_loop_types(self):
        """Verify nomic loop event types."""
        assert StreamEventType.CYCLE_START.value == "cycle_start"
        assert StreamEventType.PHASE_START.value == "phase_start"
        assert StreamEventType.VERIFICATION_RESULT.value == "verification_result"

    def test_audience_types(self):
        """Verify audience event types."""
        assert StreamEventType.USER_VOTE.value == "user_vote"
        assert StreamEventType.USER_SUGGESTION.value == "user_suggestion"
        assert StreamEventType.AUDIENCE_SUMMARY.value == "audience_summary"


class TestSequenceNumberOrdering:
    """Test sequence number assignment for event ordering."""

    def test_global_sequence_increments(self):
        """Test that global sequence numbers increment monotonically."""
        emitter = SyncEventEmitter(loop_id="test-loop")

        # Emit multiple events
        for i in range(5):
            event = StreamEvent(
                type=StreamEventType.AGENT_MESSAGE,
                data={"content": f"Message {i}"},
                agent="agent1",
            )
            emitter.emit(event)

        events = emitter.drain()
        assert len(events) == 5

        # Verify sequence numbers are monotonically increasing
        for i, event in enumerate(events, start=1):
            assert event.seq == i

    def test_sequence_reset(self):
        """Test that reset_sequences resets counters."""
        emitter = SyncEventEmitter(loop_id="test-loop")

        # Emit some events
        for _ in range(3):
            emitter.emit(
                StreamEvent(
                    type=StreamEventType.AGENT_MESSAGE,
                    data={"content": "test"},
                    agent="agent1",
                )
            )

        emitter.drain()  # Clear queue
        emitter.reset_sequences()

        # Emit new event - should start at seq 1
        emitter.emit(
            StreamEvent(
                type=StreamEventType.AGENT_MESSAGE,
                data={"content": "after reset"},
                agent="agent1",
            )
        )

        events = emitter.drain()
        assert len(events) == 1
        assert events[0].seq == 1

    def test_per_agent_sequence_tracking(self):
        """Test per-agent sequence numbers for token stream integrity."""
        emitter = SyncEventEmitter(loop_id="test-loop")

        # Emit interleaved events from multiple agents
        agents = ["claude", "gemini", "grok"]
        for i in range(9):
            agent = agents[i % 3]
            emitter.emit(
                StreamEvent(
                    type=StreamEventType.TOKEN_DELTA,
                    data={"token": f"token_{i}"},
                    agent=agent,
                )
            )

        events = emitter.drain()
        assert len(events) == 9

        # Group by agent and verify per-agent sequences
        agent_events: dict[str, list[StreamEvent]] = {}
        for event in events:
            if event.agent not in agent_events:
                agent_events[event.agent] = []
            agent_events[event.agent].append(event)

        # Each agent should have 3 events with seq 1, 2, 3
        for agent, agent_evts in agent_events.items():
            assert len(agent_evts) == 3
            for i, evt in enumerate(agent_evts, start=1):
                assert evt.agent_seq == i, f"{agent} event {i} has wrong agent_seq"

    def test_global_vs_agent_sequence_independence(self):
        """Test that global and agent sequences are independent."""
        emitter = SyncEventEmitter(loop_id="test-loop")

        # Emit events alternating between agents
        emitter.emit(
            StreamEvent(type=StreamEventType.AGENT_MESSAGE, data={}, agent="agent1")
        )
        emitter.emit(
            StreamEvent(type=StreamEventType.AGENT_MESSAGE, data={}, agent="agent2")
        )
        emitter.emit(
            StreamEvent(type=StreamEventType.AGENT_MESSAGE, data={}, agent="agent1")
        )

        events = emitter.drain()

        # Global sequence: 1, 2, 3
        assert [e.seq for e in events] == [1, 2, 3]

        # Agent1 sequence: 1, _, 2
        assert events[0].agent_seq == 1  # agent1 first
        assert events[2].agent_seq == 2  # agent1 second

        # Agent2 sequence: _, 1, _
        assert events[1].agent_seq == 1  # agent2 first


class TestQueueOverflowProtection:
    """Test queue overflow handling to prevent memory exhaustion."""

    def test_overflow_drops_oldest_events(self, monkeypatch):
        """Test that queue overflow drops oldest events."""
        # Use a small queue size for testing
        monkeypatch.setattr(SyncEventEmitter, "MAX_QUEUE_SIZE", 5)

        emitter = SyncEventEmitter(loop_id="test-loop")

        # Emit more events than queue can hold
        for i in range(10):
            emitter.emit(
                StreamEvent(
                    type=StreamEventType.AGENT_MESSAGE,
                    data={"index": i},
                )
            )

        events = emitter.drain()

        # Should have max 5 events (newest ones)
        assert len(events) <= 5

        # Events should have increasing sequence numbers
        seqs = [e.seq for e in events]
        assert seqs == sorted(seqs)

    def test_overflow_counter_increments(self, monkeypatch):
        """Test that overflow counter tracks dropped events."""
        monkeypatch.setattr(SyncEventEmitter, "MAX_QUEUE_SIZE", 3)

        emitter = SyncEventEmitter(loop_id="test-loop")

        # Fill queue
        for _ in range(3):
            emitter.emit(
                StreamEvent(type=StreamEventType.AGENT_MESSAGE, data={})
            )

        # This should trigger overflow
        emitter.emit(StreamEvent(type=StreamEventType.AGENT_MESSAGE, data={}))

        assert emitter._overflow_count >= 1


class TestAudienceInboxOverflow:
    """Test AudienceInbox bounded queue behavior."""

    def test_inbox_max_size_enforced(self):
        """Test that inbox respects max size limit."""
        inbox = AudienceInbox(max_messages=5)

        # Add more messages than limit
        for i in range(10):
            inbox.put(
                AudienceMessage(
                    type="vote",
                    loop_id="test",
                    payload={"choice": f"option_{i}"},
                )
            )

        messages = inbox.get_all()
        assert len(messages) == 5

        # Should have newest messages (indices 5-9)
        choices = [m.payload["choice"] for m in messages]
        assert choices == [f"option_{i}" for i in range(5, 10)]

    def test_inbox_overflow_counter(self):
        """Test that overflow counter tracks dropped messages."""
        inbox = AudienceInbox(max_messages=3)

        # Fill inbox
        for i in range(5):
            inbox.put(
                AudienceMessage(type="vote", loop_id="test", payload={"i": i})
            )

        # Should have dropped 2 messages
        assert inbox._overflow_count == 2


class TestTokenBucketRateLimiting:
    """Test TokenBucket rate limiter."""

    def test_initial_burst_allowed(self):
        """Test that initial burst is allowed up to burst_size."""
        bucket = TokenBucket(rate_per_minute=60, burst_size=5)

        # Should allow burst_size requests immediately
        for _ in range(5):
            assert bucket.consume() is True

        # 6th request should fail
        assert bucket.consume() is False

    def test_token_refill_over_time(self):
        """Test that tokens refill based on rate."""
        bucket = TokenBucket(rate_per_minute=6000, burst_size=1)  # 100/sec

        # Consume the only token
        assert bucket.consume() is True
        assert bucket.consume() is False

        # Manually set last_refill to simulate time passing
        import time

        bucket.last_refill = time.monotonic() - 0.1  # 100ms ago = 10 tokens at 6000/min

        # Should be able to consume now
        assert bucket.consume() is True
