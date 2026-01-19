"""
Tests for RLM streaming functionality.

Tests the streaming methods that yield progress events during
RLM query execution and refinement.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.rlm.bridge import AragoraRLM, DebateContextAdapter
from aragora.rlm.types import (
    AbstractionLevel,
    AbstractionNode,
    RLMConfig,
    RLMContext,
    RLMResult,
    RLMStreamEvent,
    RLMStreamEventType,
)
from aragora.rlm.compressor import clear_compression_cache


@pytest.fixture
def sample_context() -> RLMContext:
    """Create a sample RLM context for testing."""
    context = RLMContext(
        original_content="This is a test document about feature flags and deployment.",
        original_tokens=100,
        source_type="text",
    )

    # Add summary level nodes
    summary_node = AbstractionNode(
        id="summary_1",
        level=AbstractionLevel.SUMMARY,
        content="Summary: Feature flags for deployment control.",
        token_count=20,
    )
    context.levels[AbstractionLevel.SUMMARY] = [summary_node]
    context.nodes_by_id["summary_1"] = summary_node

    # Add detailed level nodes
    detailed_node = AbstractionNode(
        id="detailed_1",
        level=AbstractionLevel.DETAILED,
        content="Detailed: Feature flags enable gradual rollout and quick rollback.",
        token_count=40,
        parent_id="summary_1",
    )
    context.levels[AbstractionLevel.DETAILED] = [detailed_node]
    context.nodes_by_id["detailed_1"] = detailed_node
    summary_node.child_ids = ["detailed_1"]

    return context


class TestRLMStreamEvent:
    """Tests for RLMStreamEvent dataclass."""

    def test_event_creation(self):
        """Test creating a stream event."""
        event = RLMStreamEvent(
            event_type=RLMStreamEventType.QUERY_START,
            query="What is X?",
        )

        assert event.event_type == RLMStreamEventType.QUERY_START
        assert event.query == "What is X?"
        assert event.timestamp > 0

    def test_event_auto_timestamp(self):
        """Test that timestamp is auto-generated."""
        event1 = RLMStreamEvent(event_type=RLMStreamEventType.QUERY_START)
        event2 = RLMStreamEvent(event_type=RLMStreamEventType.QUERY_COMPLETE)

        assert event1.timestamp > 0
        assert event2.timestamp >= event1.timestamp

    def test_event_to_dict(self):
        """Test event serialization to dictionary."""
        event = RLMStreamEvent(
            event_type=RLMStreamEventType.LEVEL_ENTERED,
            query="test",
            level=AbstractionLevel.SUMMARY,
            tokens_processed=100,
        )

        data = event.to_dict()

        assert data["event_type"] == "level_entered"
        assert data["query"] == "test"
        assert data["level"] == "SUMMARY"
        assert data["tokens_processed"] == 100

    def test_event_to_dict_with_result(self):
        """Test event serialization includes result."""
        result = RLMResult(
            answer="Test answer",
            ready=True,
            confidence=0.9,
            iteration=1,
        )

        event = RLMStreamEvent(
            event_type=RLMStreamEventType.QUERY_COMPLETE,
            query="test",
            result=result,
        )

        data = event.to_dict()

        assert "result" in data
        assert data["result"]["answer"] == "Test answer"
        assert data["result"]["ready"] is True
        assert data["result"]["confidence"] == 0.9

    def test_event_to_dict_with_error(self):
        """Test event serialization includes error."""
        event = RLMStreamEvent(
            event_type=RLMStreamEventType.ERROR,
            query="test",
            error="Something went wrong",
        )

        data = event.to_dict()

        assert data["error"] == "Something went wrong"


class TestRLMStreamEventType:
    """Tests for RLMStreamEventType enum."""

    def test_all_event_types_exist(self):
        """Test that all expected event types exist."""
        expected = [
            "query_start",
            "query_complete",
            "iteration_start",
            "iteration_complete",
            "feedback_generated",
            "level_entered",
            "node_examined",
            "sub_call_start",
            "sub_call_complete",
            "partial_answer",
            "final_answer",
            "confidence_update",
            "error",
        ]

        actual = [e.value for e in RLMStreamEventType]
        for event_type in expected:
            assert event_type in actual, f"Missing event type: {event_type}"


class TestQueryStream:
    """Tests for streaming query execution."""

    @pytest.mark.asyncio
    async def test_query_stream_emits_start_event(self, sample_context):
        """Test that query_stream emits a start event."""
        rlm = AragoraRLM()
        rlm.query = AsyncMock(return_value=RLMResult(
            answer="Test answer",
            ready=True,
            confidence=0.9,
        ))

        events = []
        async for event in rlm.query_stream("What is X?", sample_context):
            events.append(event)

        # First event should be query start
        assert events[0].event_type == RLMStreamEventType.QUERY_START
        assert events[0].query == "What is X?"

    @pytest.mark.asyncio
    async def test_query_stream_emits_level_events(self, sample_context):
        """Test that query_stream emits level entered events."""
        rlm = AragoraRLM()
        rlm.query = AsyncMock(return_value=RLMResult(
            answer="Test answer",
            ready=True,
            confidence=0.9,
        ))

        events = []
        async for event in rlm.query_stream("What is X?", sample_context):
            events.append(event)

        # Should have level entered events
        level_events = [e for e in events if e.event_type == RLMStreamEventType.LEVEL_ENTERED]
        assert len(level_events) >= 1

    @pytest.mark.asyncio
    async def test_query_stream_emits_node_events(self, sample_context):
        """Test that query_stream emits node examined events."""
        rlm = AragoraRLM()
        rlm.query = AsyncMock(return_value=RLMResult(
            answer="Test answer",
            ready=True,
            confidence=0.9,
        ))

        events = []
        async for event in rlm.query_stream("What is X?", sample_context):
            events.append(event)

        # Should have node examined events
        node_events = [e for e in events if e.event_type == RLMStreamEventType.NODE_EXAMINED]
        assert len(node_events) >= 1
        assert node_events[0].node_id in ["summary_1", "detailed_1"]

    @pytest.mark.asyncio
    async def test_query_stream_emits_complete_event(self, sample_context):
        """Test that query_stream emits a complete event with result."""
        rlm = AragoraRLM()
        rlm.query = AsyncMock(return_value=RLMResult(
            answer="Test answer",
            ready=True,
            confidence=0.9,
            tokens_processed=100,
        ))

        events = []
        async for event in rlm.query_stream("What is X?", sample_context):
            events.append(event)

        # Last event should be query complete
        assert events[-1].event_type == RLMStreamEventType.QUERY_COMPLETE
        assert events[-1].result is not None
        assert events[-1].result.answer == "Test answer"

    @pytest.mark.asyncio
    async def test_query_stream_handles_error(self, sample_context):
        """Test that query_stream emits error event on failure."""
        rlm = AragoraRLM()
        rlm.query = AsyncMock(side_effect=RuntimeError("Query failed"))

        events = []
        with pytest.raises(RuntimeError, match="Query failed"):
            async for event in rlm.query_stream("What is X?", sample_context):
                events.append(event)

        # Should have error event
        error_events = [e for e in events if e.event_type == RLMStreamEventType.ERROR]
        assert len(error_events) == 1
        assert "Query failed" in error_events[0].error


class TestQueryWithRefinementStream:
    """Tests for streaming refinement execution."""

    @pytest.mark.asyncio
    async def test_refinement_stream_single_iteration(self, sample_context):
        """Test streaming with single successful iteration."""
        rlm = AragoraRLM()

        async def mock_query_iteration(query, context, strategy, iteration, feedback):
            return RLMResult(
                answer="Final answer",
                ready=True,
                confidence=0.9,
            )

        rlm._query_iteration = mock_query_iteration

        events = []
        async for event in rlm.query_with_refinement_stream(
            "What is X?",
            sample_context,
        ):
            events.append(event)

        # Should have start, iteration start, confidence update, iteration complete, final, complete
        event_types = [e.event_type for e in events]
        assert RLMStreamEventType.QUERY_START in event_types
        assert RLMStreamEventType.ITERATION_START in event_types
        assert RLMStreamEventType.ITERATION_COMPLETE in event_types
        assert RLMStreamEventType.FINAL_ANSWER in event_types
        assert RLMStreamEventType.QUERY_COMPLETE in event_types

    @pytest.mark.asyncio
    async def test_refinement_stream_multiple_iterations(self, sample_context):
        """Test streaming with multiple refinement iterations."""
        rlm = AragoraRLM()

        call_count = 0

        async def mock_query_iteration(query, context, strategy, iteration, feedback):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return RLMResult(
                    answer=f"Attempt {call_count}",
                    ready=False,
                    confidence=0.5,
                )
            return RLMResult(
                answer="Final answer",
                ready=True,
                confidence=0.9,
            )

        rlm._query_iteration = mock_query_iteration

        events = []
        async for event in rlm.query_with_refinement_stream(
            "What is X?",
            sample_context,
            max_iterations=5,
        ):
            events.append(event)

        # Should have multiple iteration events
        iteration_starts = [e for e in events if e.event_type == RLMStreamEventType.ITERATION_START]
        assert len(iteration_starts) == 3  # 3 iterations

        # Should have feedback generated events for iterations > 0
        feedback_events = [e for e in events if e.event_type == RLMStreamEventType.FEEDBACK_GENERATED]
        assert len(feedback_events) == 2  # Iterations 1 and 2

        # Should have partial answer events
        partial_events = [e for e in events if e.event_type == RLMStreamEventType.PARTIAL_ANSWER]
        assert len(partial_events) == 2  # First two attempts

    @pytest.mark.asyncio
    async def test_refinement_stream_confidence_updates(self, sample_context):
        """Test that confidence updates are streamed."""
        rlm = AragoraRLM()

        async def mock_query_iteration(query, context, strategy, iteration, feedback):
            confidence = 0.3 + (iteration * 0.2)
            return RLMResult(
                answer=f"Attempt {iteration}",
                ready=iteration >= 2,
                confidence=confidence,
            )

        rlm._query_iteration = mock_query_iteration

        events = []
        async for event in rlm.query_with_refinement_stream(
            "What is X?",
            sample_context,
            max_iterations=5,
        ):
            events.append(event)

        # Should have confidence update events
        confidence_events = [e for e in events if e.event_type == RLMStreamEventType.CONFIDENCE_UPDATE]
        assert len(confidence_events) >= 1

        # Confidence should increase
        confidences = [e.confidence for e in confidence_events]
        assert confidences[-1] > confidences[0]

    @pytest.mark.asyncio
    async def test_refinement_stream_final_answer(self, sample_context):
        """Test that final answer is emitted correctly."""
        rlm = AragoraRLM()

        async def mock_query_iteration(query, context, strategy, iteration, feedback):
            return RLMResult(
                answer="The final answer is X",
                ready=True,
                confidence=0.95,
            )

        rlm._query_iteration = mock_query_iteration

        events = []
        async for event in rlm.query_with_refinement_stream(
            "What is X?",
            sample_context,
        ):
            events.append(event)

        # Should have final answer event
        final_events = [e for e in events if e.event_type == RLMStreamEventType.FINAL_ANSWER]
        assert len(final_events) == 1
        assert final_events[0].result.answer == "The final answer is X"
        assert final_events[0].result.ready is True


class TestCompressStream:
    """Tests for streaming compression."""

    @pytest.mark.asyncio
    async def test_compress_stream_emits_events(self):
        """Test that compress_stream emits progress events."""
        clear_compression_cache()

        rlm = AragoraRLM(aragora_config=RLMConfig(cache_compressions=False))

        content = "Test content " * 100  # Create substantial content

        events = []
        async for event in rlm.compress_stream(content, "text"):
            events.append(event)

        # Should have start and complete events
        event_types = [e.event_type for e in events]
        assert RLMStreamEventType.QUERY_START in event_types
        assert RLMStreamEventType.QUERY_COMPLETE in event_types

    @pytest.mark.asyncio
    async def test_compress_stream_emits_level_events(self):
        """Test that compress_stream emits level created events."""
        clear_compression_cache()

        rlm = AragoraRLM(aragora_config=RLMConfig(cache_compressions=False))

        content = "Test content about feature flags and deployment strategies. " * 50

        events = []
        async for event in rlm.compress_stream(content, "text"):
            events.append(event)

        # Should have level entered events
        level_events = [e for e in events if e.event_type == RLMStreamEventType.LEVEL_ENTERED]
        # At least one level should be created
        assert len(level_events) >= 1


class TestStreamEventOrdering:
    """Tests for correct event ordering in streams."""

    @pytest.mark.asyncio
    async def test_events_are_chronologically_ordered(self, sample_context):
        """Test that events have increasing timestamps."""
        rlm = AragoraRLM()
        rlm.query = AsyncMock(return_value=RLMResult(
            answer="Test answer",
            ready=True,
            confidence=0.9,
        ))

        events = []
        async for event in rlm.query_stream("What is X?", sample_context):
            events.append(event)

        # All events should have timestamps
        for event in events:
            assert event.timestamp > 0

        # Timestamps should be non-decreasing
        for i in range(1, len(events)):
            assert events[i].timestamp >= events[i - 1].timestamp

    @pytest.mark.asyncio
    async def test_start_before_complete(self, sample_context):
        """Test that start events come before complete events."""
        rlm = AragoraRLM()
        rlm.query = AsyncMock(return_value=RLMResult(
            answer="Test answer",
            ready=True,
            confidence=0.9,
        ))

        events = []
        async for event in rlm.query_stream("What is X?", sample_context):
            events.append(event)

        event_types = [e.event_type for e in events]

        # Query start should be before query complete
        start_idx = event_types.index(RLMStreamEventType.QUERY_START)
        complete_idx = event_types.index(RLMStreamEventType.QUERY_COMPLETE)
        assert start_idx < complete_idx
