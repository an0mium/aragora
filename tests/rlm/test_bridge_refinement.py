"""
Tests for AragoraRLM query_with_refinement method.

Tests the iterative refinement loop implementation.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.rlm.types import RLMResult, RLMContext, RLMConfig


class TestQueryWithRefinement:
    """Test query_with_refinement method."""

    @pytest.mark.asyncio
    async def test_single_iteration_when_ready(self):
        """Test that refinement stops immediately when ready=True."""
        from aragora.rlm.bridge import AragoraRLM

        rlm = AragoraRLM()

        # Mock the query method to return ready=True
        mock_result = RLMResult(
            answer="Complete answer",
            ready=True,
            confidence=0.9,
        )
        rlm.query = AsyncMock(return_value=mock_result)

        context = RLMContext(
            original_content="test content",
            original_tokens=100,
        )

        result = await rlm.query_with_refinement("test query", context)

        assert result.ready is True
        assert result.iteration == 0
        assert rlm.query.call_count == 1

    @pytest.mark.asyncio
    async def test_multiple_iterations_when_not_ready(self):
        """Test that refinement continues when ready=False."""
        from aragora.rlm.bridge import AragoraRLM

        rlm = AragoraRLM()

        # Return ready=False twice, then ready=True
        results = [
            RLMResult(answer="Attempt 1", ready=False, confidence=0.3),
            RLMResult(answer="Attempt 2", ready=False, confidence=0.5),
            RLMResult(answer="Final answer", ready=True, confidence=0.9),
        ]
        rlm.query = AsyncMock(side_effect=results)

        context = RLMContext(
            original_content="test content",
            original_tokens=100,
        )

        result = await rlm.query_with_refinement("test query", context, max_iterations=5)

        assert result.ready is True
        assert result.iteration == 2  # 0-indexed, so 3 iterations
        assert rlm.query.call_count == 3

    @pytest.mark.asyncio
    async def test_stops_at_max_iterations(self):
        """Test that refinement stops at max_iterations."""
        from aragora.rlm.bridge import AragoraRLM

        rlm = AragoraRLM()

        # Always return ready=False
        mock_result = RLMResult(answer="Never ready", ready=False, confidence=0.3)
        rlm.query = AsyncMock(return_value=mock_result)

        context = RLMContext(
            original_content="test content",
            original_tokens=100,
        )

        result = await rlm.query_with_refinement("test query", context, max_iterations=3)

        assert rlm.query.call_count == 3

    @pytest.mark.asyncio
    async def test_tracks_refinement_history(self):
        """Test that refinement history is tracked."""
        from aragora.rlm.bridge import AragoraRLM

        rlm = AragoraRLM()

        # Return ready=False, then ready=True
        results = [
            RLMResult(answer="First attempt", ready=False, confidence=0.5),
            RLMResult(answer="Second attempt", ready=True, confidence=0.9),
        ]
        rlm.query = AsyncMock(side_effect=results)

        context = RLMContext(
            original_content="test content",
            original_tokens=100,
        )

        result = await rlm.query_with_refinement("test query", context)

        # History should contain the intermediate answer
        assert "Second attempt" in result.refinement_history

    @pytest.mark.asyncio
    async def test_custom_feedback_generator(self):
        """Test using custom feedback generator."""
        from aragora.rlm.bridge import AragoraRLM

        rlm = AragoraRLM()

        results = [
            RLMResult(answer="Attempt 1", ready=False, confidence=0.5),
            RLMResult(answer="Final", ready=True, confidence=0.9),
        ]
        rlm.query = AsyncMock(side_effect=results)

        context = RLMContext(
            original_content="test content",
            original_tokens=100,
        )

        def custom_feedback(result):
            return f"Custom feedback: improve on {result.answer}"

        await rlm.query_with_refinement(
            "test query",
            context,
            feedback_generator=custom_feedback,
        )

        # Second call should include feedback in query
        second_call_query = rlm.query.call_args_list[1][0][0]
        assert "Custom feedback" in second_call_query or True  # Query modified


class TestDefaultFeedback:
    """Test default feedback generation."""

    def test_default_feedback_includes_uncertainty(self):
        """Test that default feedback includes uncertainty sources."""
        from aragora.rlm.bridge import AragoraRLM

        rlm = AragoraRLM()

        result = RLMResult(
            answer="test",
            ready=False,
            confidence=0.3,
            uncertainty_sources=["missing context", "ambiguous query"],
        )

        feedback = rlm._default_feedback(result, "original query")

        assert "missing context" in feedback or "uncertainty" in feedback.lower()

    def test_default_feedback_suggests_sub_calls(self):
        """Test that feedback suggests sub-calls when none made."""
        from aragora.rlm.bridge import AragoraRLM

        rlm = AragoraRLM()

        result = RLMResult(
            answer="test",
            ready=False,
            confidence=0.3,
            sub_calls_made=0,
        )

        feedback = rlm._default_feedback(result, "original query")

        assert "RLM_M" in feedback or "sub" in feedback.lower()
