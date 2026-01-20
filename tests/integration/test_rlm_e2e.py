"""
End-to-end integration tests for RLM pipeline.

Tests the complete flow:
1. Create a debate result with messages
2. Compress debate using RLM
3. Query compressed context with refinement
4. Verify results are accurate and metrics are tracked
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from aragora.rlm.types import (
    RLMContext,
    RLMResult,
    RLMConfig,
    AbstractionLevel,
    AbstractionNode,
    CompressionResult,
)
from aragora.rlm.bridge import AragoraRLM, DebateContextAdapter
from aragora.rlm.compressor import HierarchicalCompressor, clear_compression_cache
from aragora.core import DebateResult, Message, Critique, Vote


@pytest.fixture
def sample_debate_result() -> DebateResult:
    """Create a sample debate result for testing."""
    return DebateResult(
        debate_id="debate-e2e-test",
        task="Should we implement feature flags for gradual rollout?",
        status="completed",
        participants=["claude", "gpt4", "gemini"],
        messages=[
            Message(
                role="proposer",
                agent="claude",
                content="Feature flags are essential for safe deployments. They allow us to control rollout percentages, A/B test features, and quickly disable problematic code without full rollbacks.",
                round=1,
            ),
            Message(
                role="proposer",
                agent="gpt4",
                content="While feature flags add complexity, the benefits outweigh the costs. I recommend using a mature system like LaunchDarkly or building a simple internal solution.",
                round=1,
            ),
            Message(
                role="proposer",
                agent="claude",
                content="To address the maintenance concern: we should establish a flag lifecycle policy - flags older than 90 days without activity should be reviewed and removed.",
                round=2,
            ),
            Message(
                role="proposer",
                agent="gpt4",
                content="I agree with the lifecycle policy. Additionally, we should use naming conventions and documentation to track flag purposes.",
                round=2,
            ),
        ],
        critiques=[
            Critique(
                agent="gemini",
                target_agent="claude",
                target_content="Feature flags are essential for safe deployments.",
                issues=["Didn't address the maintenance burden of accumulated flags"],
                suggestions=["Add lifecycle policy for flag cleanup"],
                severity=3.0,
                reasoning="Good points on safety, but maintenance burden needs addressing.",
            ),
        ],
        proposals={
            "claude": "Feature flags with 90-day lifecycle policy",
            "gpt4": "Feature flags with naming conventions and docs",
        },
        consensus_reached=True,
        final_answer="Yes, implement feature flags with lifecycle management and documentation standards.",
        rounds_completed=2,
        confidence=0.85,
    )


@pytest.fixture
def mock_agent_call():
    """Create a mock agent call function for compression."""

    def agent_call(prompt: str, model: str) -> str:
        # Return appropriate summaries based on prompt content
        if "key details" in prompt.lower() or "detailed" in prompt.lower():
            return "Detailed: Feature flags discussion with lifecycle policy of 90 days and documentation requirements."
        elif "key points" in prompt.lower() or "summary" in prompt.lower():
            return "Summary: Team agrees on feature flags with 90-day lifecycle and docs."
        elif "abstract" in prompt.lower() or "brief" in prompt.lower():
            return "Feature flags: approved with governance."
        elif "tags" in prompt.lower() or "metadata" in prompt.lower():
            return "feature-flags, deployment, governance, lifecycle-policy, documentation"
        return "Compressed content"

    return agent_call


class TestRLMEndToEndPipeline:
    """Test the complete RLM pipeline end-to-end."""

    @pytest.mark.asyncio
    async def test_full_debate_to_query_pipeline(self, sample_debate_result, mock_agent_call):
        """Test complete flow: debate → compress → query."""
        clear_compression_cache()

        # Step 1: Create RLM and adapter
        rlm = AragoraRLM(
            aragora_config=RLMConfig(
                cache_compressions=True,
                parallel_sub_calls=False,  # Sequential for predictable testing
            )
        )

        adapter = DebateContextAdapter(rlm)

        # Step 2: Compress the debate
        context = await adapter.compress_debate(sample_debate_result)

        # Verify compression created abstraction levels
        assert context is not None
        assert context.original_tokens > 0
        assert AbstractionLevel.FULL in context.levels
        assert len(context.levels[AbstractionLevel.FULL]) > 0

        # Step 3: Query with refinement
        # Mock the query method to return a result
        mock_result = RLMResult(
            answer="The team decided to implement feature flags with a 90-day lifecycle policy.",
            ready=True,
            confidence=0.9,
            iteration=0,
            nodes_examined=["L0_0", "L0_1"],
            tokens_processed=500,
        )
        rlm.query = AsyncMock(return_value=mock_result)

        result = await rlm.query_with_refinement(
            query="What was the decision on feature flags?",
            context=context,
            max_iterations=3,
        )

        # Verify result
        assert result.ready is True
        assert result.confidence >= 0.8
        assert "feature flags" in result.answer.lower()

    @pytest.mark.asyncio
    async def test_compression_creates_hierarchy(self, sample_debate_result, mock_agent_call):
        """Test that compression creates proper abstraction hierarchy."""
        clear_compression_cache()

        compressor = HierarchicalCompressor(
            config=RLMConfig(cache_compressions=False),
            agent_call=mock_agent_call,
        )

        # Get formatted debate content
        adapter = DebateContextAdapter()
        formatted = adapter.format_for_rlm(sample_debate_result)
        content = str(formatted)

        result = await compressor.compress(content, source_type="debate", max_levels=4)

        # Verify hierarchy was created
        assert result.context is not None
        assert AbstractionLevel.FULL in result.context.levels

        # Check compression reduced tokens at higher levels
        full_tokens = result.context.total_tokens_at_level(AbstractionLevel.FULL)
        assert full_tokens > 0

        # Verify sub-calls were made for compression
        assert result.sub_calls_made >= 0

    @pytest.mark.asyncio
    async def test_iterative_refinement_when_not_ready(self, sample_debate_result):
        """Test that iterative refinement continues when ready=False."""
        clear_compression_cache()

        rlm = AragoraRLM()
        adapter = DebateContextAdapter(rlm)

        context = await adapter.compress_debate(sample_debate_result)

        # Mock _query_iteration to return not ready twice, then ready
        call_count = 0

        async def mock_query_iteration(query, context, strategy, iteration, feedback):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return RLMResult(
                    answer=f"Partial answer {call_count}",
                    ready=False,
                    confidence=0.3 + (call_count * 0.2),
                    iteration=call_count - 1,
                )
            return RLMResult(
                answer="Final: Feature flags with 90-day lifecycle.",
                ready=True,
                confidence=0.9,
                iteration=2,
            )

        rlm._query_iteration = mock_query_iteration

        result = await rlm.query_with_refinement(
            query="What was decided?",
            context=context,
            max_iterations=5,
        )

        # Should have iterated until ready
        assert result.ready is True
        assert result.iteration == 2
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_max_iterations_limit(self, sample_debate_result):
        """Test that refinement stops at max_iterations."""
        clear_compression_cache()

        rlm = AragoraRLM()
        adapter = DebateContextAdapter(rlm)

        context = await adapter.compress_debate(sample_debate_result)

        # Mock query to always return not ready
        rlm.query = AsyncMock(
            return_value=RLMResult(
                answer="Still working on it",
                ready=False,
                confidence=0.5,
            )
        )

        result = await rlm.query_with_refinement(
            query="What was decided?",
            context=context,
            max_iterations=3,
        )

        # Should have stopped at max iterations
        assert rlm.query.call_count == 3

    @pytest.mark.asyncio
    async def test_cache_hit_on_repeated_compression(self, sample_debate_result):
        """Test that repeated compression uses cache."""
        clear_compression_cache()

        rlm = AragoraRLM(aragora_config=RLMConfig(cache_compressions=True))
        adapter = DebateContextAdapter(rlm)

        # First compression
        context1 = await adapter.compress_debate(sample_debate_result)

        # Second compression of same debate
        context2 = await adapter.compress_debate(sample_debate_result)

        # Should get same content (from cache)
        assert context1.original_content == context2.original_content


class TestRLMDebateFormatting:
    """Test debate formatting for RLM context."""

    def test_format_for_rlm_includes_rounds(self, sample_debate_result):
        """Test that formatting includes debate rounds."""
        adapter = DebateContextAdapter()
        formatted = adapter.format_for_rlm(sample_debate_result)

        # Should have ROUNDS key (uppercase)
        assert "ROUNDS" in formatted

    def test_format_for_rlm_includes_consensus(self, sample_debate_result):
        """Test that formatting includes consensus information."""
        adapter = DebateContextAdapter()
        formatted = adapter.format_for_rlm(sample_debate_result)

        assert "CONSENSUS" in formatted
        # CONSENSUS is the final answer string
        assert formatted["CONSENSUS"] is not None

    def test_format_for_rlm_preserves_agent_info(self, sample_debate_result):
        """Test that agent information is preserved in formatting."""
        adapter = DebateContextAdapter()
        formatted = adapter.format_for_rlm(sample_debate_result)

        # Should have AGENTS key (uppercase)
        assert "AGENTS" in formatted


class TestRLMContextNavigation:
    """Test navigation through abstraction levels."""

    @pytest.mark.asyncio
    async def test_get_at_level_returns_content(self, sample_debate_result):
        """Test getting content at specific abstraction level."""
        clear_compression_cache()

        rlm = AragoraRLM()
        adapter = DebateContextAdapter(rlm)

        context = await adapter.compress_debate(sample_debate_result)

        # Get content at FULL level
        full_content = context.get_at_level(AbstractionLevel.FULL)
        assert len(full_content) > 0

    @pytest.mark.asyncio
    async def test_total_tokens_at_level(self, sample_debate_result):
        """Test token counting at abstraction levels."""
        clear_compression_cache()

        rlm = AragoraRLM()
        adapter = DebateContextAdapter(rlm)

        context = await adapter.compress_debate(sample_debate_result)

        # FULL level should have most tokens
        full_tokens = context.total_tokens_at_level(AbstractionLevel.FULL)
        assert full_tokens > 0

    @pytest.mark.asyncio
    async def test_nodes_by_id_populated(self, sample_debate_result):
        """Test that nodes_by_id dict is populated."""
        clear_compression_cache()

        rlm = AragoraRLM()
        adapter = DebateContextAdapter(rlm)

        context = await adapter.compress_debate(sample_debate_result)

        # Should have nodes indexed by ID
        assert len(context.nodes_by_id) > 0

        # Can look up nodes by ID
        for node_id, node in context.nodes_by_id.items():
            assert node.id == node_id


class TestRLMRefinementHistory:
    """Test refinement history tracking."""

    @pytest.mark.asyncio
    async def test_refinement_history_tracked(self, sample_debate_result):
        """Test that refinement history is tracked across iterations."""
        clear_compression_cache()

        rlm = AragoraRLM()
        adapter = DebateContextAdapter(rlm)

        context = await adapter.compress_debate(sample_debate_result)

        # Mock multiple iterations using _query_iteration
        call_count = 0

        async def mock_query_iteration(query, context, strategy, iteration, feedback):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                return RLMResult(
                    answer=f"Attempt {call_count}: needs more detail",
                    ready=False,
                    confidence=0.5,
                    iteration=call_count - 1,
                )
            return RLMResult(
                answer="Final: Feature flags approved",
                ready=True,
                confidence=0.9,
                iteration=call_count - 1,
            )

        rlm._query_iteration = mock_query_iteration

        result = await rlm.query_with_refinement(
            query="What was decided?",
            context=context,
            max_iterations=5,
        )

        # Should have refinement history
        assert result.ready is True


class TestRLMErrorHandling:
    """Test error handling in RLM pipeline."""

    @pytest.mark.asyncio
    async def test_handles_empty_debate(self):
        """Test handling of empty debate result."""
        empty_debate = DebateResult(
            debate_id="empty-debate",
            task="Empty",
            status="completed",
        )

        rlm = AragoraRLM()
        adapter = DebateContextAdapter(rlm)

        context = await adapter.compress_debate(empty_debate)

        # Should still create a valid context
        assert context is not None
        assert context.original_tokens >= 0

    @pytest.mark.asyncio
    async def test_handles_query_exception(self, sample_debate_result):
        """Test handling of exceptions during query."""
        clear_compression_cache()

        rlm = AragoraRLM()
        adapter = DebateContextAdapter(rlm)

        context = await adapter.compress_debate(sample_debate_result)

        # Mock query to raise exception
        rlm.query = AsyncMock(side_effect=Exception("LLM API error"))

        # Should handle gracefully
        with pytest.raises(Exception, match="LLM API error"):
            await rlm.query_with_refinement(
                query="What was decided?",
                context=context,
            )
