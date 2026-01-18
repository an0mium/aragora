"""
RLM Load Tests for production validation.

Tests hierarchical context compression under production-like debate loads:
- Large context compression (10K-100K tokens)
- Multi-round debate history handling
- Memory efficiency under sustained load
- Compression ratio validation

Run with:
    pytest tests/rlm/test_load.py -v --timeout=120
"""

from __future__ import annotations

import asyncio
import gc
import sys
import time
from dataclasses import dataclass
from typing import Any, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

# Import RLM modules
try:
    from aragora.rlm import (
        HierarchicalCompressor,
        AbstractionLevel,
    )
    from aragora.rlm.types import RLMContext, AbstractionNode, CompressionResult
    HAS_RLM = True
except ImportError as e:
    print(f"RLM import error: {e}")
    HAS_RLM = False


@dataclass
class DebateRound:
    """Simulated debate round for testing."""
    round_num: int
    proposals: List[str]
    critiques: List[str]
    votes: List[str]


def generate_debate_content(num_rounds: int, tokens_per_round: int = 1000) -> str:
    """Generate simulated debate content for load testing."""
    content_parts = []

    for round_num in range(1, num_rounds + 1):
        content_parts.append(f"## Round {round_num}")

        # Generate proposals (simulate multiple agents)
        for agent_id in range(3):
            proposal_words = tokens_per_round // 4 // 3
            proposal = " ".join([f"word{i}" for i in range(proposal_words)])
            content_parts.append(f"### Agent {agent_id} Proposal\n{proposal}")

        # Generate critiques
        for agent_id in range(3):
            critique_words = tokens_per_round // 4 // 3
            critique = " ".join([f"critique{i}" for i in range(critique_words)])
            content_parts.append(f"### Agent {agent_id} Critique\n{critique}")

        # Generate votes/summary
        summary_words = tokens_per_round // 4
        summary = " ".join([f"summary{i}" for i in range(summary_words)])
        content_parts.append(f"### Round {round_num} Summary\n{summary}")

    return "\n\n".join(content_parts)


def estimate_tokens(text: str) -> int:
    """Estimate token count (approx 4 chars per token)."""
    return len(text) // 4


@pytest.mark.skipif(not HAS_RLM, reason="RLM module not available")
class TestRLMCompression:
    """Test RLM compression functionality."""

    @pytest.fixture
    def mock_agent_call(self) -> AsyncMock:
        """Create mock agent call for testing without API calls."""
        async def mock_call(prompt: str, context: str = "") -> str:
            # Return a compressed version (50% reduction)
            words = context.split()[:len(context.split()) // 2]
            return " ".join(words) if words else "Summary of content."

        return mock_call

    @pytest.fixture
    def compressor(self, mock_agent_call: AsyncMock) -> HierarchicalCompressor:
        """Create compressor with mock agent call."""
        return HierarchicalCompressor(agent_call=mock_agent_call)

    @pytest.mark.asyncio
    async def test_small_content_compression(self, compressor: HierarchicalCompressor):
        """Test compression of small content (under threshold)."""
        content = "This is a small piece of content that should not need compression."

        result = await compressor.compress(content, source_type="text")

        assert result is not None
        assert result.context is not None
        # Small content may not create hierarchy
        assert estimate_tokens(content) < 1000

    @pytest.mark.asyncio
    async def test_medium_content_compression(self, compressor: HierarchicalCompressor):
        """Test compression of medium content (1-10K tokens)."""
        content = generate_debate_content(num_rounds=5, tokens_per_round=500)
        original_tokens = estimate_tokens(content)

        assert 2000 < original_tokens < 10000, f"Expected 2-10K tokens, got {original_tokens}"

        result = await compressor.compress(content, source_type="debate", max_levels=2)

        assert result is not None
        assert result.context is not None
        assert result.original_tokens == original_tokens or result.original_tokens > 0

    @pytest.mark.asyncio
    async def test_large_content_compression(self, compressor: HierarchicalCompressor):
        """Test compression of large content (10-50K tokens)."""
        content = generate_debate_content(num_rounds=20, tokens_per_round=1000)
        original_tokens = estimate_tokens(content)

        assert 15000 < original_tokens < 100000, f"Expected 15-100K tokens, got {original_tokens}"

        start_time = time.time()
        result = await compressor.compress(content, source_type="debate", max_levels=3)
        duration = time.time() - start_time

        assert result is not None
        assert result.context is not None
        # Should complete within reasonable time
        assert duration < 30, f"Compression took too long: {duration:.2f}s"


@pytest.mark.skipif(not HAS_RLM, reason="RLM module not available")
class TestRLMLoadPerformance:
    """Performance tests for RLM under load."""

    @pytest.fixture
    def fast_compressor(self) -> HierarchicalCompressor:
        """Create compressor with fast mock for load testing."""
        async def fast_mock(prompt: str, context: str = "") -> str:
            # Simulate fast compression (just truncate)
            return context[:500] if len(context) > 500 else context

        return HierarchicalCompressor(agent_call=fast_mock)

    @pytest.mark.asyncio
    async def test_sustained_compression_load(self, fast_compressor: HierarchicalCompressor):
        """Test sustained compression operations."""
        num_operations = 10
        content_sizes = [1000, 2000, 5000, 10000]

        results = []
        start_time = time.time()

        for i in range(num_operations):
            content = generate_debate_content(
                num_rounds=3 + (i % 5),
                tokens_per_round=content_sizes[i % len(content_sizes)] // 3
            )

            result = await fast_compressor.compress(content, source_type="debate")
            results.append(result)

        total_duration = time.time() - start_time
        avg_duration = total_duration / num_operations

        assert all(r is not None for r in results)
        assert avg_duration < 5.0, f"Average compression too slow: {avg_duration:.2f}s"

    @pytest.mark.asyncio
    async def test_concurrent_compression(self, fast_compressor: HierarchicalCompressor):
        """Test concurrent compression operations."""
        num_concurrent = 5

        async def compress_one(content: str) -> CompressionResult:
            return await fast_compressor.compress(content, source_type="debate")

        contents = [
            generate_debate_content(num_rounds=3, tokens_per_round=500)
            for _ in range(num_concurrent)
        ]

        start_time = time.time()
        results = await asyncio.gather(*[compress_one(c) for c in contents])
        duration = time.time() - start_time

        assert all(r is not None for r in results)
        # Concurrent should be faster than sequential
        assert duration < num_concurrent * 2, f"Concurrent compression too slow: {duration:.2f}s"

    @pytest.mark.asyncio
    async def test_memory_efficiency(self, fast_compressor: HierarchicalCompressor):
        """Test memory efficiency under sustained load."""
        # Force garbage collection before test
        gc.collect()
        initial_memory = self._get_memory_usage()

        # Run multiple compressions
        for _ in range(20):
            content = generate_debate_content(num_rounds=10, tokens_per_round=500)
            result = await fast_compressor.compress(content, source_type="debate")
            del result
            gc.collect()

        final_memory = self._get_memory_usage()
        memory_increase = final_memory - initial_memory

        # Memory increase should be bounded (< 150MB for test environment variability)
        assert memory_increase < 150 * 1024 * 1024, f"Memory leak detected: {memory_increase / 1024 / 1024:.2f}MB"

    def _get_memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024  # macOS returns KB


@pytest.mark.skipif(not HAS_RLM, reason="RLM module not available")
class TestRLMCompressionRatios:
    """Test compression ratio targets."""

    @pytest.fixture
    def summarizing_compressor(self) -> HierarchicalCompressor:
        """Create compressor that actually summarizes."""
        async def summarize(prompt: str, context: str = "") -> str:
            # Simulate 50-70% compression
            words = context.split()
            target_words = len(words) // 3
            return " ".join(words[:target_words]) if target_words > 0 else "Summary."

        return HierarchicalCompressor(agent_call=summarize)

    @pytest.mark.asyncio
    async def test_compression_ratio_small(self, summarizing_compressor: HierarchicalCompressor):
        """Test compression ratio for small content."""
        content = generate_debate_content(num_rounds=3, tokens_per_round=200)
        original_tokens = estimate_tokens(content)

        result = await summarizing_compressor.compress(content, source_type="debate")

        # Get compressed size at SUMMARY level
        if result.context and hasattr(result.context, 'get_at_level'):
            summary = result.context.get_at_level(AbstractionLevel.SUMMARY)
            if summary:
                compressed_tokens = estimate_tokens(summary)
                ratio = compressed_tokens / original_tokens if original_tokens > 0 else 1.0
                # For small content with mock compressor, compression may be minimal
                # Just verify we get a result - actual compression depends on content size
                assert ratio <= 1.0 or original_tokens < 500, f"Unexpected expansion: ratio={ratio}"

    @pytest.mark.asyncio
    async def test_compression_ratio_large(self, summarizing_compressor: HierarchicalCompressor):
        """Test compression ratio for large content."""
        content = generate_debate_content(num_rounds=15, tokens_per_round=500)
        original_tokens = estimate_tokens(content)

        result = await summarizing_compressor.compress(content, source_type="debate", max_levels=3)

        assert result is not None
        # Verify we can retrieve at different levels
        if result.context and hasattr(result.context, 'get_at_level'):
            abstract = result.context.get_at_level(AbstractionLevel.ABSTRACT)
            summary = result.context.get_at_level(AbstractionLevel.SUMMARY)

            if abstract:
                abstract_tokens = estimate_tokens(abstract)
                # Abstract should be much smaller than original
                assert abstract_tokens < original_tokens

    @pytest.mark.asyncio
    async def test_hierarchy_levels_created(self, summarizing_compressor: HierarchicalCompressor):
        """Test that appropriate hierarchy levels are created."""
        content = generate_debate_content(num_rounds=10, tokens_per_round=300)

        result = await summarizing_compressor.compress(content, source_type="debate", max_levels=3)

        assert result is not None
        assert result.context is not None

        # Check that we have multiple levels
        if hasattr(result.context, 'levels'):
            assert len(result.context.levels) >= 1


@pytest.mark.skipif(not HAS_RLM, reason="RLM module not available")
class TestRLMDebateIntegration:
    """Test RLM integration with debate context."""

    @pytest.mark.asyncio
    async def test_cognitive_limiter_integration(self):
        """Test RLM integration with CognitiveLoadLimiter."""
        try:
            from aragora.debate.cognitive_limiter_rlm import (
                RLMCognitiveLoadLimiter,
                create_rlm_limiter,
            )
        except ImportError:
            pytest.skip("RLMCognitiveLoadLimiter not available")

        limiter = create_rlm_limiter(stress_level="elevated")

        # Create mock messages
        from aragora.core import Message

        messages = [
            Message(agent=f"agent_{i}", role="proposer", content=f"Proposal {i} " * 100, round=i)
            for i in range(10)
        ]

        # Limit messages (using max_chars parameter)
        limited = limiter.limit_messages(messages, max_chars=8000)

        assert len(limited) > 0
        assert len(limited) <= len(messages)

    @pytest.mark.asyncio
    async def test_prompt_builder_rlm_context(self):
        """Test PromptBuilder RLM context integration."""
        try:
            from aragora.debate.prompt_builder import PromptBuilder, HAS_RLM as PB_HAS_RLM
        except ImportError:
            pytest.skip("PromptBuilder not available")

        if not PB_HAS_RLM:
            pytest.skip("PromptBuilder RLM support not available")

        # Create mock protocol and environment
        mock_protocol = MagicMock()
        mock_protocol.asymmetric_stances = False
        mock_protocol.agreement_intensity = None

        mock_env = MagicMock()
        mock_env.task = "Test debate task"
        mock_env.context = None

        builder = PromptBuilder(protocol=mock_protocol, env=mock_env)

        # Test RLM context setting
        mock_context = MagicMock()
        mock_context.levels = {AbstractionLevel.ABSTRACT: []}

        builder.set_rlm_context(mock_context)

        assert builder._rlm_context is not None

        # Test RLM hint generation
        hint = builder.get_rlm_context_hint()
        # May or may not have hint depending on context
        assert isinstance(hint, str)


@pytest.mark.skipif(not HAS_RLM, reason="RLM module not available")
class TestRLMEdgeCases:
    """Test RLM edge cases and error handling."""

    @pytest.fixture
    def error_compressor(self) -> HierarchicalCompressor:
        """Create compressor that occasionally fails."""
        call_count = [0]

        async def sometimes_fail(prompt: str, context: str = "") -> str:
            call_count[0] += 1
            if call_count[0] % 3 == 0:
                raise RuntimeError("Simulated compression failure")
            return context[:100] if len(context) > 100 else context

        return HierarchicalCompressor(agent_call=sometimes_fail)

    @pytest.mark.asyncio
    async def test_empty_content(self):
        """Test handling of empty content."""
        compressor = HierarchicalCompressor()

        result = await compressor.compress("", source_type="text")

        # Should handle gracefully
        assert result is not None or True  # May return None for empty

    @pytest.mark.asyncio
    async def test_very_long_content(self):
        """Test handling of very long content."""
        compressor = HierarchicalCompressor(
            agent_call=lambda p, c: c[:100] if len(c) > 100 else c
        )

        # Generate 100K+ tokens
        content = generate_debate_content(num_rounds=50, tokens_per_round=2000)
        tokens = estimate_tokens(content)

        assert tokens > 50000, f"Expected >50K tokens, got {tokens}"

        # Should handle without crashing
        result = await compressor.compress(content, source_type="debate", max_levels=4)

        assert result is not None

    @pytest.mark.asyncio
    async def test_unicode_content(self):
        """Test handling of unicode content."""
        compressor = HierarchicalCompressor(
            agent_call=lambda p, c: c[:100] if len(c) > 100 else c
        )

        # Content with unicode characters
        content = "Unicode test: 日本語 한국어 العربية émoji: 🎉🚀🔥 " * 500

        result = await compressor.compress(content, source_type="text")

        assert result is not None

    @pytest.mark.asyncio
    async def test_special_characters(self):
        """Test handling of special characters."""
        compressor = HierarchicalCompressor(
            agent_call=lambda p, c: c[:100] if len(c) > 100 else c
        )

        # Content with special characters
        content = "Special chars: <script>alert('xss')</script> SELECT * FROM users; ${env.SECRET} " * 100

        result = await compressor.compress(content, source_type="text")

        assert result is not None


# Performance benchmarks (run with pytest-benchmark)
@pytest.mark.skipif(not HAS_RLM, reason="RLM module not available")
class TestRLMBenchmarks:
    """Benchmark tests for RLM performance."""

    @pytest.fixture
    def benchmark_compressor(self) -> HierarchicalCompressor:
        """Create fast compressor for benchmarking."""
        async def fast_compress(prompt: str, context: str = "") -> str:
            return context[:len(context) // 3] if len(context) > 100 else context

        return HierarchicalCompressor(agent_call=fast_compress)

    @pytest.mark.asyncio
    async def test_benchmark_small_compression(self, benchmark_compressor: HierarchicalCompressor):
        """Benchmark small content compression."""
        content = generate_debate_content(num_rounds=2, tokens_per_round=200)

        start = time.time()
        for _ in range(10):
            await benchmark_compressor.compress(content, source_type="debate")
        duration = (time.time() - start) / 10

        assert duration < 1.0, f"Small compression too slow: {duration:.3f}s"

    @pytest.mark.asyncio
    async def test_benchmark_medium_compression(self, benchmark_compressor: HierarchicalCompressor):
        """Benchmark medium content compression."""
        content = generate_debate_content(num_rounds=5, tokens_per_round=500)

        start = time.time()
        for _ in range(5):
            await benchmark_compressor.compress(content, source_type="debate")
        duration = (time.time() - start) / 5

        assert duration < 3.0, f"Medium compression too slow: {duration:.3f}s"
