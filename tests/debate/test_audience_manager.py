"""
Tests for AudienceManager.

Tests the Stadium Mailbox pattern for audience participation:
- Event queuing and filtering (thread-safe)
- Event draining (digest phase)
- Vote and suggestion management
- Thread safety and lock handling
- Event emitter subscription
- Edge cases and error conditions
"""

import queue
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ============================================================================
# Mock Classes
# ============================================================================


class MockStreamEventType(Enum):
    """Mock event types for testing."""

    USER_VOTE = "user_vote"
    USER_SUGGESTION = "user_suggestion"
    DEBATE_START = "debate_start"


@dataclass
class MockEvent:
    """Mock event for testing."""

    type: Any
    data: dict
    loop_id: str | None = None


class MockEventEmitter:
    """Mock event emitter for testing."""

    def __init__(self):
        self.subscribers = []

    def subscribe(self, callback):
        self.subscribers.append(callback)

    def emit(self, event):
        for sub in self.subscribers:
            sub(event)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def manager():
    """Create an AudienceManager for testing."""
    from aragora.debate.audience_manager import AudienceManager

    return AudienceManager(loop_id="test-loop")


@pytest.fixture
def strict_manager():
    """Create an AudienceManager with strict loop scoping."""
    from aragora.debate.audience_manager import AudienceManager

    return AudienceManager(loop_id="test-loop", strict_loop_scoping=True)


@pytest.fixture
def small_queue_manager():
    """Create an AudienceManager with small queue size."""
    from aragora.debate.audience_manager import AudienceManager

    return AudienceManager(loop_id="test-loop", queue_size=3)


# ============================================================================
# Initialization Tests
# ============================================================================


class TestAudienceManagerInit:
    """Tests for AudienceManager initialization."""

    def test_init_default_values(self):
        """Test initialization with default values."""
        from aragora.config import USER_EVENT_QUEUE_SIZE
        from aragora.debate.audience_manager import AudienceManager

        manager = AudienceManager()

        assert manager.loop_id is None
        assert manager.strict_loop_scoping is False
        assert manager._event_queue.maxsize == USER_EVENT_QUEUE_SIZE
        assert manager._notify_callback is None

    def test_init_with_loop_id(self):
        """Test initialization with loop_id."""
        from aragora.debate.audience_manager import AudienceManager

        manager = AudienceManager(loop_id="debate-123")

        assert manager.loop_id == "debate-123"

    def test_init_with_strict_scoping(self):
        """Test initialization with strict loop scoping."""
        from aragora.debate.audience_manager import AudienceManager

        manager = AudienceManager(strict_loop_scoping=True)

        assert manager.strict_loop_scoping is True

    def test_init_with_custom_queue_size(self):
        """Test initialization with custom queue size."""
        from aragora.debate.audience_manager import AudienceManager

        manager = AudienceManager(queue_size=500)

        assert manager._event_queue.maxsize == 500


# ============================================================================
# Callback Tests
# ============================================================================


class TestNotifyCallback:
    """Tests for notify callback functionality."""

    def test_set_notify_callback(self, manager):
        """Test setting notify callback."""
        callback = MagicMock()
        manager.set_notify_callback(callback)

        assert manager._notify_callback == callback

    def test_callback_called_on_drain(self, manager):
        """Test callback is called when events are drained."""
        from aragora.events.types import StreamEventType

        callback = MagicMock()
        manager.set_notify_callback(callback)

        # Add event directly to queue
        manager._event_queue.put_nowait((StreamEventType.USER_VOTE, {"agent": "a"}))

        drained = manager.drain_events()

        assert drained == 1
        callback.assert_called_once()
        assert callback.call_args[0][0] == "audience_drain"

    def test_callback_not_called_when_no_events(self, manager):
        """Test callback is not called when no events drained."""
        callback = MagicMock()
        manager.set_notify_callback(callback)

        drained = manager.drain_events()

        assert drained == 0
        callback.assert_not_called()


# ============================================================================
# Event Emitter Subscription Tests
# ============================================================================


class TestEventEmitterSubscription:
    """Tests for event emitter subscription."""

    def test_subscribe_to_emitter(self, manager):
        """Test subscribing to event emitter."""
        emitter = MockEventEmitter()
        manager.subscribe_to_emitter(emitter)

        assert manager.handle_event in emitter.subscribers

    def test_subscribe_to_none_emitter(self, manager):
        """Test subscribing to None emitter."""
        # Should not raise
        manager.subscribe_to_emitter(None)

    def test_events_flow_through_emitter(self, manager):
        """Test events flow through emitter to manager."""
        from aragora.events.types import StreamEventType

        emitter = MockEventEmitter()
        manager.subscribe_to_emitter(emitter)

        event = MockEvent(
            type=StreamEventType.USER_VOTE,
            data={"agent": "agent_a"},
            loop_id="test-loop",
        )
        emitter.emit(event)

        assert manager.pending_count == 1


# ============================================================================
# Handle Event Tests
# ============================================================================


class TestHandleEvent:
    """Tests for handle_event method."""

    def test_handle_user_vote_event(self, manager):
        """Test handling USER_VOTE event."""
        from aragora.events.types import StreamEventType

        event = MockEvent(
            type=StreamEventType.USER_VOTE,
            data={"agent": "agent_a", "stance": "agree"},
            loop_id="test-loop",
        )

        manager.handle_event(event)

        assert manager.pending_count == 1

    def test_handle_user_suggestion_event(self, manager):
        """Test handling USER_SUGGESTION event."""
        from aragora.events.types import StreamEventType

        event = MockEvent(
            type=StreamEventType.USER_SUGGESTION,
            data={"text": "Consider this point"},
            loop_id="test-loop",
        )

        manager.handle_event(event)

        assert manager.pending_count == 1

    def test_ignore_other_event_types(self, manager):
        """Test that non-user events are ignored."""
        from aragora.events.types import StreamEventType

        event = MockEvent(
            type=StreamEventType.DEBATE_START,
            data={"task": "Test"},
            loop_id="test-loop",
        )

        manager.handle_event(event)

        assert manager.pending_count == 0

    def test_ignore_events_from_other_loops(self, manager):
        """Test events from other loops are ignored."""
        from aragora.events.types import StreamEventType

        event = MockEvent(
            type=StreamEventType.USER_VOTE,
            data={"agent": "agent_a"},
            loop_id="other-loop",  # Different loop
        )

        manager.handle_event(event)

        assert manager.pending_count == 0

    def test_accept_events_without_loop_id_non_strict(self, manager):
        """Test events without loop_id are accepted in non-strict mode."""
        from aragora.events.types import StreamEventType

        event = MockEvent(
            type=StreamEventType.USER_VOTE,
            data={"agent": "agent_a"},
            loop_id=None,
        )

        manager.handle_event(event)

        assert manager.pending_count == 1

    def test_drop_events_without_loop_id_strict(self, strict_manager):
        """Test events without loop_id are dropped in strict mode."""
        from aragora.events.types import StreamEventType

        event = MockEvent(
            type=StreamEventType.USER_VOTE,
            data={"agent": "agent_a"},
            loop_id=None,
        )

        strict_manager.handle_event(event)

        assert strict_manager.pending_count == 0

    def test_queue_full_drops_event(self, small_queue_manager, caplog):
        """Test that events are dropped when queue is full."""
        from aragora.events.types import StreamEventType

        # Fill the queue (size is 3)
        for i in range(5):
            event = MockEvent(
                type=StreamEventType.USER_VOTE,
                data={"agent": f"agent_{i}"},
                loop_id="test-loop",
            )
            small_queue_manager.handle_event(event)

        # Only 3 should be queued
        assert small_queue_manager.pending_count == 3


# ============================================================================
# Drain Events Tests
# ============================================================================


class TestDrainEvents:
    """Tests for drain_events method."""

    def test_drain_votes(self, manager):
        """Test draining vote events."""
        from aragora.events.types import StreamEventType

        # Add votes to queue
        manager._event_queue.put_nowait((StreamEventType.USER_VOTE, {"agent": "a"}))
        manager._event_queue.put_nowait((StreamEventType.USER_VOTE, {"agent": "b"}))

        drained = manager.drain_events()

        assert drained == 2
        assert manager.votes_count == 2

    def test_drain_suggestions(self, manager):
        """Test draining suggestion events."""
        from aragora.events.types import StreamEventType

        # Add suggestions to queue
        manager._event_queue.put_nowait(
            (StreamEventType.USER_SUGGESTION, {"text": "suggestion 1"})
        )
        manager._event_queue.put_nowait(
            (StreamEventType.USER_SUGGESTION, {"text": "suggestion 2"})
        )

        drained = manager.drain_events()

        assert drained == 2
        assert manager.suggestions_count == 2

    def test_drain_mixed_events(self, manager):
        """Test draining mixed vote and suggestion events."""
        from aragora.events.types import StreamEventType

        manager._event_queue.put_nowait((StreamEventType.USER_VOTE, {"agent": "a"}))
        manager._event_queue.put_nowait(
            (StreamEventType.USER_SUGGESTION, {"text": "tip"})
        )
        manager._event_queue.put_nowait((StreamEventType.USER_VOTE, {"agent": "b"}))

        drained = manager.drain_events()

        assert drained == 3
        assert manager.votes_count == 2
        assert manager.suggestions_count == 1

    def test_drain_empty_queue(self, manager):
        """Test draining empty queue."""
        drained = manager.drain_events()

        assert drained == 0
        assert manager.votes_count == 0
        assert manager.suggestions_count == 0

    def test_drain_clears_queue(self, manager):
        """Test that drain clears the queue."""
        from aragora.events.types import StreamEventType

        manager._event_queue.put_nowait((StreamEventType.USER_VOTE, {"agent": "a"}))

        manager.drain_events()

        assert manager.pending_count == 0


# ============================================================================
# Get Votes/Suggestions Tests
# ============================================================================


class TestGetVotesAndSuggestions:
    """Tests for getting votes and suggestions."""

    def test_get_votes_empty(self, manager):
        """Test getting votes when empty."""
        votes = manager.get_votes()

        assert votes == []

    def test_get_votes_after_drain(self, manager):
        """Test getting votes after draining."""
        from aragora.events.types import StreamEventType

        manager._event_queue.put_nowait(
            (StreamEventType.USER_VOTE, {"agent": "a", "stance": "agree"})
        )
        manager._event_queue.put_nowait(
            (StreamEventType.USER_VOTE, {"agent": "b", "stance": "disagree"})
        )
        manager.drain_events()

        votes = manager.get_votes()

        assert len(votes) == 2
        assert votes[0]["agent"] == "a"
        assert votes[1]["agent"] == "b"

    def test_get_votes_returns_copy(self, manager):
        """Test that get_votes returns a copy."""
        from aragora.events.types import StreamEventType

        manager._event_queue.put_nowait((StreamEventType.USER_VOTE, {"agent": "a"}))
        manager.drain_events()

        votes = manager.get_votes()
        votes.append({"agent": "fake"})

        assert manager.votes_count == 1  # Original unchanged

    def test_get_suggestions_empty(self, manager):
        """Test getting suggestions when empty."""
        suggestions = manager.get_suggestions()

        assert suggestions == []

    def test_get_suggestions_after_drain(self, manager):
        """Test getting suggestions after draining."""
        from aragora.events.types import StreamEventType

        manager._event_queue.put_nowait(
            (StreamEventType.USER_SUGGESTION, {"text": "idea 1"})
        )
        manager.drain_events()

        suggestions = manager.get_suggestions()

        assert len(suggestions) == 1
        assert suggestions[0]["text"] == "idea 1"


# ============================================================================
# Clear Tests
# ============================================================================


class TestClear:
    """Tests for clear methods."""

    def test_clear_votes(self, manager):
        """Test clearing votes."""
        from aragora.events.types import StreamEventType

        manager._event_queue.put_nowait((StreamEventType.USER_VOTE, {"agent": "a"}))
        manager.drain_events()
        assert manager.votes_count == 1

        manager.clear_votes()

        assert manager.votes_count == 0

    def test_clear_suggestions(self, manager):
        """Test clearing suggestions."""
        from aragora.events.types import StreamEventType

        manager._event_queue.put_nowait(
            (StreamEventType.USER_SUGGESTION, {"text": "idea"})
        )
        manager.drain_events()
        assert manager.suggestions_count == 1

        manager.clear_suggestions()

        assert manager.suggestions_count == 0

    def test_clear_all(self, manager):
        """Test clearing all events."""
        from aragora.events.types import StreamEventType

        # Add events to queue and drain
        manager._event_queue.put_nowait((StreamEventType.USER_VOTE, {"agent": "a"}))
        manager._event_queue.put_nowait(
            (StreamEventType.USER_SUGGESTION, {"text": "idea"})
        )
        manager.drain_events()

        # Add more events to queue (not drained)
        manager._event_queue.put_nowait((StreamEventType.USER_VOTE, {"agent": "b"}))

        manager.clear_all()

        assert manager.votes_count == 0
        assert manager.suggestions_count == 0
        assert manager.pending_count == 0


# ============================================================================
# Property Tests
# ============================================================================


class TestProperties:
    """Tests for property methods."""

    def test_pending_count(self, manager):
        """Test pending_count property."""
        from aragora.events.types import StreamEventType

        assert manager.pending_count == 0

        manager._event_queue.put_nowait((StreamEventType.USER_VOTE, {"agent": "a"}))
        assert manager.pending_count == 1

        manager._event_queue.put_nowait((StreamEventType.USER_VOTE, {"agent": "b"}))
        assert manager.pending_count == 2

    def test_votes_count(self, manager):
        """Test votes_count property."""
        from aragora.events.types import StreamEventType

        assert manager.votes_count == 0

        manager._event_queue.put_nowait((StreamEventType.USER_VOTE, {"agent": "a"}))
        manager.drain_events()

        assert manager.votes_count == 1

    def test_suggestions_count(self, manager):
        """Test suggestions_count property."""
        from aragora.events.types import StreamEventType

        assert manager.suggestions_count == 0

        manager._event_queue.put_nowait(
            (StreamEventType.USER_SUGGESTION, {"text": "idea"})
        )
        manager.drain_events()

        assert manager.suggestions_count == 1


# ============================================================================
# Bounded Deque Tests
# ============================================================================


class TestBoundedDeques:
    """Tests for bounded deque behavior."""

    def test_votes_auto_evict_oldest(self, small_queue_manager):
        """Test that votes deque evicts oldest entries."""
        from aragora.events.types import StreamEventType

        # Queue size is 3, which also bounds the deques
        # Add 3 events
        for i in range(3):
            small_queue_manager._event_queue.put_nowait(
                (StreamEventType.USER_VOTE, {"agent": f"agent_{i}"})
            )
        small_queue_manager.drain_events()

        # Manually add more to test deque eviction
        # The deque maxlen is queue_size (3)
        small_queue_manager._votes.append({"agent": "agent_3"})

        # Should only have 3 votes (oldest evicted)
        votes = small_queue_manager.get_votes()
        assert len(votes) == 3
        # First entry should be agent_1 (agent_0 evicted)
        assert votes[0]["agent"] == "agent_1"

    def test_suggestions_auto_evict_oldest(self, small_queue_manager):
        """Test that suggestions deque evicts oldest entries."""
        from aragora.events.types import StreamEventType

        # Add 3 suggestions
        for i in range(3):
            small_queue_manager._event_queue.put_nowait(
                (StreamEventType.USER_SUGGESTION, {"text": f"suggestion_{i}"})
            )
        small_queue_manager.drain_events()

        # Add another to trigger eviction
        small_queue_manager._suggestions.append({"text": "suggestion_3"})

        suggestions = small_queue_manager.get_suggestions()
        assert len(suggestions) == 3
        # First entry should be suggestion_1 (suggestion_0 evicted)
        assert suggestions[0]["text"] == "suggestion_1"


# ============================================================================
# Thread Safety Tests
# ============================================================================


class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_handle_events(self, manager):
        """Test handling events from multiple threads."""
        from aragora.events.types import StreamEventType

        events_per_thread = 50
        num_threads = 5
        errors = []

        def add_events(thread_id: int):
            try:
                for i in range(events_per_thread):
                    event = MockEvent(
                        type=StreamEventType.USER_VOTE,
                        data={"agent": f"agent_{thread_id}_{i}"},
                        loop_id="test-loop",
                    )
                    manager.handle_event(event)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=add_events, args=(i,)) for i in range(num_threads)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # Some may be dropped if queue fills, but should not crash
        assert manager.pending_count <= events_per_thread * num_threads

    def test_concurrent_drain_and_handle(self, manager):
        """Test draining while events are being added."""
        from aragora.events.types import StreamEventType

        errors = []
        stop_flag = threading.Event()

        def add_events():
            try:
                while not stop_flag.is_set():
                    event = MockEvent(
                        type=StreamEventType.USER_VOTE,
                        data={"agent": "concurrent"},
                        loop_id="test-loop",
                    )
                    manager.handle_event(event)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def drain_events():
            try:
                for _ in range(20):
                    manager.drain_events()
                    time.sleep(0.005)
            except Exception as e:
                errors.append(e)

        producer = threading.Thread(target=add_events)
        consumer = threading.Thread(target=drain_events)

        producer.start()
        consumer.start()

        consumer.join()
        stop_flag.set()
        producer.join()

        assert len(errors) == 0

    def test_concurrent_get_votes_and_drain(self, manager):
        """Test getting votes while draining."""
        from aragora.events.types import StreamEventType

        errors = []
        stop_flag = threading.Event()

        def drain_loop():
            try:
                while not stop_flag.is_set():
                    # Add and drain
                    manager._event_queue.put_nowait(
                        (StreamEventType.USER_VOTE, {"agent": "drainer"})
                    )
                    manager.drain_events()
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        def read_loop():
            try:
                for _ in range(50):
                    manager.get_votes()
                    manager.get_suggestions()
                    time.sleep(0.002)
            except Exception as e:
                errors.append(e)

        drainer = threading.Thread(target=drain_loop)
        reader = threading.Thread(target=read_loop)

        drainer.start()
        reader.start()

        reader.join()
        stop_flag.set()
        drainer.join()

        assert len(errors) == 0


# ============================================================================
# Lock Timeout Tests
# ============================================================================


class TestLockTimeout:
    """Tests for lock timeout behavior."""

    def test_drain_handles_lock_timeout(self, manager, caplog):
        """Test that drain handles lock timeout gracefully."""
        from aragora.events.types import StreamEventType

        # Add event
        manager._event_queue.put_nowait((StreamEventType.USER_VOTE, {"agent": "a"}))

        # Simulate lock being held
        original_timeout = manager._lock_timeout
        manager._lock_timeout = 0.001  # Very short timeout

        # Hold the lock in another thread
        lock_held = threading.Event()
        release_lock = threading.Event()

        def hold_lock():
            with manager._data_lock:
                lock_held.set()
                release_lock.wait(timeout=1.0)

        holder = threading.Thread(target=hold_lock)
        holder.start()
        lock_held.wait()

        # Try to drain - should timeout
        drained = manager.drain_events()

        release_lock.set()
        holder.join()

        manager._lock_timeout = original_timeout

        # Event was skipped due to timeout, but no error raised
        assert drained == 0


# ============================================================================
# Edge Cases Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_handle_event_without_type_attribute(self, manager):
        """Test handling event without proper type attribute."""
        # Event without StreamEventType - should not crash
        event = MagicMock()
        event.type = "invalid_type"
        event.loop_id = "test-loop"

        # Should not raise
        manager.handle_event(event)
        assert manager.pending_count == 0

    def test_handle_event_without_loop_id_attribute(self, manager):
        """Test handling event without loop_id attribute."""
        from aragora.events.types import StreamEventType

        class NoLoopIdEvent:
            type = StreamEventType.USER_VOTE
            data = {"agent": "a"}
            # No loop_id attribute

        event = NoLoopIdEvent()

        # Should not crash, getattr returns None
        manager.handle_event(event)

    def test_multiple_drains_idempotent(self, manager):
        """Test that multiple drains are idempotent."""
        from aragora.events.types import StreamEventType

        manager._event_queue.put_nowait((StreamEventType.USER_VOTE, {"agent": "a"}))

        drained1 = manager.drain_events()
        drained2 = manager.drain_events()
        drained3 = manager.drain_events()

        assert drained1 == 1
        assert drained2 == 0
        assert drained3 == 0
        assert manager.votes_count == 1

    def test_clear_while_empty(self, manager):
        """Test clearing when already empty."""
        # Should not raise
        manager.clear_votes()
        manager.clear_suggestions()
        manager.clear_all()

        assert manager.votes_count == 0
        assert manager.suggestions_count == 0


# ============================================================================
# Integration Tests
# ============================================================================


class TestAudienceManagerIntegration:
    """Integration tests for AudienceManager."""

    def test_full_workflow(self, manager):
        """Test complete audience participation workflow."""
        from aragora.events.types import StreamEventType

        callback_calls = []

        def callback(event_type: str, **kwargs):
            callback_calls.append((event_type, kwargs))

        # Setup
        manager.set_notify_callback(callback)
        emitter = MockEventEmitter()
        manager.subscribe_to_emitter(emitter)

        # Simulate audience participation
        vote1 = MockEvent(
            type=StreamEventType.USER_VOTE,
            data={"agent": "claude", "stance": "agree", "conviction": 0.8},
            loop_id="test-loop",
        )
        vote2 = MockEvent(
            type=StreamEventType.USER_VOTE,
            data={"agent": "gpt", "stance": "disagree", "conviction": 0.6},
            loop_id="test-loop",
        )
        suggestion = MockEvent(
            type=StreamEventType.USER_SUGGESTION,
            data={"text": "Consider edge cases", "user_id": "user-123"},
            loop_id="test-loop",
        )

        emitter.emit(vote1)
        emitter.emit(vote2)
        emitter.emit(suggestion)

        # Verify events queued
        assert manager.pending_count == 3

        # Drain at checkpoint
        drained = manager.drain_events()
        assert drained == 3
        assert manager.pending_count == 0

        # Verify callback was called
        assert len(callback_calls) == 1
        assert callback_calls[0][0] == "audience_drain"

        # Get processed events
        votes = manager.get_votes()
        suggestions = manager.get_suggestions()

        assert len(votes) == 2
        assert len(suggestions) == 1
        assert votes[0]["agent"] == "claude"
        assert suggestions[0]["text"] == "Consider edge cases"

        # Clear for next round
        manager.clear_votes()
        assert manager.votes_count == 0
        assert manager.suggestions_count == 1  # Suggestions preserved

    def test_multi_round_workflow(self, manager):
        """Test audience participation across multiple rounds."""
        from aragora.events.types import StreamEventType

        # Round 1
        manager._event_queue.put_nowait(
            (StreamEventType.USER_VOTE, {"round": 1, "agent": "a"})
        )
        manager.drain_events()
        round1_votes = manager.get_votes()
        assert len(round1_votes) == 1

        manager.clear_all()

        # Round 2
        manager._event_queue.put_nowait(
            (StreamEventType.USER_VOTE, {"round": 2, "agent": "b"})
        )
        manager._event_queue.put_nowait(
            (StreamEventType.USER_VOTE, {"round": 2, "agent": "c"})
        )
        manager.drain_events()
        round2_votes = manager.get_votes()
        assert len(round2_votes) == 2

        # Round 1 votes are gone
        assert round1_votes[0] not in round2_votes
