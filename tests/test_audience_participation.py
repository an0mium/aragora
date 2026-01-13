"""
Tests for audience participation in debates.

Tests thread-safety of the Arena mailbox pattern, loop scoping,
and integration of user votes into consensus calculations.
"""

import queue
import threading
import pytest
from unittest.mock import Mock, MagicMock, AsyncMock
from dataclasses import dataclass

from aragora.core import Agent, Environment
from aragora.debate.orchestrator import Arena, DebateProtocol


# === Fixtures ===


@pytest.fixture
def mock_agent():
    """Create a mock agent for testing."""
    agent = Mock(spec=Agent)
    agent.name = "test_agent"
    agent.role = "proposer"
    agent.system_prompt = ""  # Required for _apply_agreement_intensity
    agent.stance = None
    agent.generate = AsyncMock(return_value="Test proposal")
    agent.critique = AsyncMock()
    agent.vote = AsyncMock()
    return agent


@pytest.fixture
def mock_environment():
    """Create a mock environment for testing."""
    env = Mock(spec=Environment)
    env.task = "Test debate task"
    env.context = ""
    return env


@pytest.fixture
def arena_with_emitter(mock_environment, mock_agent):
    """Create Arena with event emitter for testing mailbox pattern."""
    emitter = Mock()
    emitter.subscribe = Mock()

    arena = Arena(
        environment=mock_environment,
        agents=[mock_agent],
        protocol=DebateProtocol(rounds=1),
        event_emitter=emitter,
        loop_id="test-loop-123",
    )
    return arena


# === Thread Safety Tests ===


class TestMailboxThreadSafety:
    """Test thread-safe event queue handling."""

    def test_queue_initialized(self, arena_with_emitter):
        """Test that Arena initializes the event queue via AudienceManager."""
        assert hasattr(arena_with_emitter, "audience_manager")
        assert isinstance(arena_with_emitter.audience_manager._event_queue, queue.Queue)

    def test_handle_user_event_enqueues(self, arena_with_emitter):
        """Test that _handle_user_event enqueues rather than direct append."""
        from aragora.server.stream import StreamEventType

        # Create mock event
        event = Mock()
        event.type = StreamEventType.USER_VOTE
        event.data = {"choice": "option1", "user_id": "user1"}
        event.loop_id = "test-loop-123"

        # Handle event
        arena_with_emitter._handle_user_event(event)

        # Verify it was enqueued, not directly appended (via AudienceManager)
        assert arena_with_emitter.audience_manager._event_queue.qsize() == 1
        assert len(arena_with_emitter.user_votes) == 0  # Not yet drained

    def test_drain_moves_to_lists(self, arena_with_emitter):
        """Test that _drain_user_events moves events from queue to lists."""
        from aragora.server.stream import StreamEventType

        # Enqueue events directly (via AudienceManager)
        arena_with_emitter.audience_manager._event_queue.put(
            (StreamEventType.USER_VOTE, {"choice": "A"})
        )
        arena_with_emitter.audience_manager._event_queue.put(
            (StreamEventType.USER_SUGGESTION, {"suggestion": "Great idea"})
        )
        arena_with_emitter.audience_manager._event_queue.put(
            (StreamEventType.USER_VOTE, {"choice": "B"})
        )

        # Drain
        arena_with_emitter._drain_user_events()

        # Verify moved to lists
        assert len(arena_with_emitter.user_votes) == 2
        assert len(arena_with_emitter.user_suggestions) == 1
        assert arena_with_emitter.audience_manager._event_queue.empty()

    def test_concurrent_enqueue_and_drain(self, arena_with_emitter):
        """Test thread safety with concurrent enqueue and drain operations."""
        from aragora.server.stream import StreamEventType

        num_events = 100

        def producer():
            """Simulate WebSocket thread adding events."""
            for i in range(num_events):
                event = Mock()
                event.type = StreamEventType.USER_VOTE
                event.data = {"choice": f"option{i}", "user_id": f"user{i}"}
                event.loop_id = "test-loop-123"
                arena_with_emitter._handle_user_event(event)

        def consumer():
            """Simulate debate loop draining events."""
            for _ in range(10):  # Multiple drain attempts
                arena_with_emitter._drain_user_events()
                threading.Event().wait(0.01)  # Small delay
            arena_with_emitter._drain_user_events()  # Final drain

        # Run concurrently
        producer_thread = threading.Thread(target=producer)
        consumer_thread = threading.Thread(target=consumer)

        producer_thread.start()
        consumer_thread.start()

        producer_thread.join()
        consumer_thread.join()

        # Final drain to catch any stragglers
        arena_with_emitter._drain_user_events()

        # All events should be in user_votes
        assert len(arena_with_emitter.user_votes) == num_events


# === Loop Scoping Tests ===


class TestLoopScoping:
    """Test loop_id filtering for multi-tenant scenarios."""

    def test_events_filtered_by_loop_id(self, mock_environment, mock_agent):
        """Test that events from other loops are ignored."""
        from aragora.server.stream import StreamEventType

        arena = Arena(
            environment=mock_environment,
            agents=[mock_agent],
            protocol=DebateProtocol(rounds=1),
            loop_id="loop-A",
        )

        # Event from same loop
        event_same = Mock()
        event_same.type = StreamEventType.USER_VOTE
        event_same.data = {"choice": "yes"}
        event_same.loop_id = "loop-A"

        # Event from different loop
        event_different = Mock()
        event_different.type = StreamEventType.USER_VOTE
        event_different.data = {"choice": "no"}
        event_different.loop_id = "loop-B"

        arena._handle_user_event(event_same)
        arena._handle_user_event(event_different)
        arena._drain_user_events()

        # Only same-loop event should be processed
        assert len(arena.user_votes) == 1
        assert arena.user_votes[0]["choice"] == "yes"

    def test_strict_loop_scoping_drops_empty_loop_id(self, mock_environment, mock_agent):
        """Test strict_loop_scoping drops events without loop_id."""
        from aragora.server.stream import StreamEventType

        arena = Arena(
            environment=mock_environment,
            agents=[mock_agent],
            protocol=DebateProtocol(rounds=1),
            loop_id="loop-A",
            strict_loop_scoping=True,
        )

        # Event with matching loop_id
        event_with_id = Mock()
        event_with_id.type = StreamEventType.USER_VOTE
        event_with_id.data = {"choice": "yes"}
        event_with_id.loop_id = "loop-A"

        # Event without loop_id
        event_no_id = Mock()
        event_no_id.type = StreamEventType.USER_VOTE
        event_no_id.data = {"choice": "no"}
        event_no_id.loop_id = ""

        arena._handle_user_event(event_with_id)
        arena._handle_user_event(event_no_id)
        arena._drain_user_events()

        # Only event with loop_id should be processed
        assert len(arena.user_votes) == 1
        assert arena.user_votes[0]["choice"] == "yes"

    def test_non_strict_allows_empty_loop_id(self, mock_environment, mock_agent):
        """Test non-strict mode allows events without loop_id."""
        from aragora.server.stream import StreamEventType

        arena = Arena(
            environment=mock_environment,
            agents=[mock_agent],
            protocol=DebateProtocol(rounds=1),
            loop_id="loop-A",
            strict_loop_scoping=False,  # Default
        )

        # Event without loop_id
        event_no_id = Mock()
        event_no_id.type = StreamEventType.USER_VOTE
        event_no_id.data = {"choice": "maybe"}
        event_no_id.loop_id = ""

        arena._handle_user_event(event_no_id)
        arena._drain_user_events()

        # Event should be processed (non-strict mode)
        assert len(arena.user_votes) == 1

    def test_strict_scoping_with_none_loop_id(self, mock_environment, mock_agent):
        """Test strict_loop_scoping with None loop_id attribute."""
        from aragora.server.stream import StreamEventType

        arena = Arena(
            environment=mock_environment,
            agents=[mock_agent],
            protocol=DebateProtocol(rounds=1),
            loop_id="loop-A",
            strict_loop_scoping=True,
        )

        # Event without loop_id attribute at all
        event_no_attr = Mock()
        event_no_attr.type = StreamEventType.USER_VOTE
        event_no_attr.data = {"choice": "no"}
        # Don't set loop_id at all - will be None via getattr

        arena._handle_user_event(event_no_attr)
        arena._drain_user_events()

        # Should be dropped in strict mode
        assert len(arena.user_votes) == 0


# === Edge Cases ===


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_drain_empty_queue(self, arena_with_emitter):
        """Test draining an empty queue doesn't error."""
        arena_with_emitter._drain_user_events()
        assert len(arena_with_emitter.user_votes) == 0
        assert len(arena_with_emitter.user_suggestions) == 0

    def test_multiple_drains_idempotent(self, arena_with_emitter):
        """Test that multiple drain calls are safe."""
        from aragora.server.stream import StreamEventType

        arena_with_emitter.audience_manager._event_queue.put(
            (StreamEventType.USER_VOTE, {"choice": "A"})
        )

        # Multiple drains
        arena_with_emitter._drain_user_events()
        arena_with_emitter._drain_user_events()
        arena_with_emitter._drain_user_events()

        # Should only have 1 vote
        assert len(arena_with_emitter.user_votes) == 1

    def test_malformed_event_data(self, arena_with_emitter):
        """Test handling of malformed event data."""
        from aragora.server.stream import StreamEventType

        # Event with None data (via AudienceManager)
        arena_with_emitter.audience_manager._event_queue.put((StreamEventType.USER_VOTE, None))

        # Should not crash
        arena_with_emitter._drain_user_events()

        # None data is appended (downstream code handles it)
        assert len(arena_with_emitter.user_votes) == 1

    def test_unknown_event_type_ignored(self, arena_with_emitter):
        """Test that unknown event types are not enqueued."""
        # Create event with unknown type
        event = Mock()
        event.type = "UNKNOWN_TYPE"
        event.data = {"foo": "bar"}
        event.loop_id = "test-loop-123"

        arena_with_emitter._handle_user_event(event)

        # Should not be enqueued (via AudienceManager)
        assert arena_with_emitter.audience_manager._event_queue.empty()


# === Suggestion Integration Tests ===


class TestSuggestionIntegration:
    """Test audience suggestion integration into prompts."""

    def test_suggestions_accumulate(self, arena_with_emitter):
        """Test that suggestions accumulate correctly."""
        from aragora.server.stream import StreamEventType

        # Add multiple suggestions (via AudienceManager)
        for i in range(5):
            arena_with_emitter.audience_manager._event_queue.put(
                (
                    StreamEventType.USER_SUGGESTION,
                    {"suggestion": f"Idea {i}", "user_id": f"user{i}"},
                )
            )

        arena_with_emitter._drain_user_events()

        assert len(arena_with_emitter.user_suggestions) == 5

    def test_votes_and_suggestions_separate(self, arena_with_emitter):
        """Test that votes and suggestions are kept in separate lists."""
        from aragora.server.stream import StreamEventType

        arena_with_emitter.audience_manager._event_queue.put(
            (StreamEventType.USER_VOTE, {"choice": "A"})
        )
        arena_with_emitter.audience_manager._event_queue.put(
            (StreamEventType.USER_SUGGESTION, {"suggestion": "Good point"})
        )
        arena_with_emitter.audience_manager._event_queue.put(
            (StreamEventType.USER_VOTE, {"choice": "B"})
        )

        arena_with_emitter._drain_user_events()

        assert len(arena_with_emitter.user_votes) == 2
        assert len(arena_with_emitter.user_suggestions) == 1
        assert arena_with_emitter.user_votes[0]["choice"] == "A"
        assert arena_with_emitter.user_suggestions[0]["suggestion"] == "Good point"


# === Arena Parameter Tests ===


class TestArenaParameters:
    """Test Arena constructor parameters."""

    def test_strict_loop_scoping_default_false(self, mock_environment, mock_agent):
        """Test that strict_loop_scoping defaults to False."""
        arena = Arena(
            environment=mock_environment,
            agents=[mock_agent],
        )
        assert arena.strict_loop_scoping is False

    def test_strict_loop_scoping_can_be_enabled(self, mock_environment, mock_agent):
        """Test that strict_loop_scoping can be enabled."""
        arena = Arena(
            environment=mock_environment,
            agents=[mock_agent],
            strict_loop_scoping=True,
        )
        assert arena.strict_loop_scoping is True

    def test_loop_id_stored(self, mock_environment, mock_agent):
        """Test that loop_id is properly stored."""
        arena = Arena(
            environment=mock_environment,
            agents=[mock_agent],
            loop_id="my-custom-loop",
        )
        assert arena.loop_id == "my-custom-loop"
