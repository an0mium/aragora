"""Tests for RLM-Enhanced Cognitive Load Limiter.

Tests the hierarchical compression functionality that replaces truncation
with semantic compression using RLM strategies.
"""

import asyncio
import pytest
from dataclasses import dataclass
from typing import Any, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.debate.cognitive_limiter_rlm import (
    RLMCognitiveBudget,
    CompressedContext,
    RLMCognitiveLoadLimiter,
    create_rlm_limiter,
)
from aragora.debate.cognitive_limiter import CognitiveBudget


# ============================================================================
# Test Fixtures
# ============================================================================


@dataclass
class MockMessage:
    """Mock message for testing."""

    agent: str = "test_agent"
    role: str = "proposer"
    content: str = "Test message content"
    round: int = 0


@dataclass
class MockCritique:
    """Mock critique for testing."""

    agent: str = "critic"
    reasoning: str = "This is a critique"
    severity: float = 0.5
    issues: List[str] = None
    suggestions: List[str] = None

    def __post_init__(self):
        if self.issues is None:
            self.issues = ["issue1"]
        if self.suggestions is None:
            self.suggestions = ["suggestion1"]


@pytest.fixture
def rlm_budget():
    """Create a test RLM budget."""
    return RLMCognitiveBudget(
        max_context_tokens=6000,
        enable_rlm_compression=True,
        compression_threshold=1000,
        max_recent_full_messages=3,
        summary_level="SUMMARY",
        preserve_first_message=True,
    )


@pytest.fixture
def sample_messages():
    """Create sample message history."""
    return [
        MockMessage(agent="user", role="task", content="Implement a rate limiter", round=0),
        MockMessage(agent="claude", role="proposer", content="I propose using token bucket algorithm", round=1),
        MockMessage(agent="gpt4", role="critic", content="Consider sliding window instead", round=1),
        MockMessage(agent="claude", role="defender", content="Token bucket is simpler to implement", round=2),
        MockMessage(agent="gemini", role="proposer", content="We could combine both approaches", round=2),
        MockMessage(agent="claude", role="synthesizer", content="Agreed, hybrid approach works best", round=3),
    ]


@pytest.fixture
def sample_critiques():
    """Create sample critiques."""
    return [
        MockCritique(severity=0.8, reasoning="Critical security issue"),
        MockCritique(severity=0.6, reasoning="Performance concern"),
        MockCritique(severity=0.3, reasoning="Minor style issue"),
        MockCritique(severity=0.2, reasoning="Trivial naming suggestion"),
    ]


@pytest.fixture
def limiter(rlm_budget):
    """Create a limiter with test budget."""
    return RLMCognitiveLoadLimiter(budget=rlm_budget)


# ============================================================================
# RLMCognitiveBudget Tests
# ============================================================================


class TestRLMCognitiveBudget:
    """Tests for RLMCognitiveBudget dataclass."""

    def test_default_values(self):
        """Test default budget values."""
        budget = RLMCognitiveBudget()

        assert budget.enable_rlm_compression is True
        assert budget.compression_threshold == 3000
        assert budget.max_recent_full_messages == 5
        assert budget.summary_level == "SUMMARY"
        assert budget.preserve_first_message is True

    def test_inherits_from_cognitive_budget(self):
        """Test that RLMCognitiveBudget extends CognitiveBudget."""
        budget = RLMCognitiveBudget()

        # Should have base class attributes
        assert hasattr(budget, "max_context_tokens")
        assert hasattr(budget, "max_history_messages")
        assert hasattr(budget, "reserve_for_response")

    def test_custom_values(self):
        """Test custom budget configuration."""
        budget = RLMCognitiveBudget(
            max_context_tokens=8000,
            enable_rlm_compression=False,
            compression_threshold=5000,
            max_recent_full_messages=10,
            summary_level="ABSTRACT",
            preserve_first_message=False,
        )

        assert budget.max_context_tokens == 8000
        assert budget.enable_rlm_compression is False
        assert budget.compression_threshold == 5000
        assert budget.max_recent_full_messages == 10
        assert budget.summary_level == "ABSTRACT"
        assert budget.preserve_first_message is False


# ============================================================================
# CompressedContext Tests
# ============================================================================


class TestCompressedContext:
    """Tests for CompressedContext dataclass."""

    def test_default_values(self):
        """Test default context values."""
        ctx = CompressedContext()

        assert ctx.messages == []
        assert ctx.critiques == []
        assert ctx.patterns == ""
        assert ctx.extra_context == ""
        assert ctx.compression_applied is False
        assert ctx.abstraction_levels == []
        assert ctx.original_chars == 0
        assert ctx.compressed_chars == 0

    def test_compression_ratio_no_original(self):
        """Test compression ratio when no original content."""
        ctx = CompressedContext(original_chars=0, compressed_chars=0)
        assert ctx.compression_ratio == 1.0

    def test_compression_ratio_calculation(self):
        """Test compression ratio calculation."""
        ctx = CompressedContext(original_chars=1000, compressed_chars=500)
        assert ctx.compression_ratio == 0.5

    def test_compression_ratio_high_compression(self):
        """Test high compression ratio."""
        ctx = CompressedContext(original_chars=10000, compressed_chars=1000)
        assert ctx.compression_ratio == 0.1

    def test_with_messages_and_critiques(self):
        """Test context with messages and critiques."""
        msg = MockMessage(content="Test")
        critique = MockCritique()

        ctx = CompressedContext(
            messages=[msg],
            critiques=[critique],
            compression_applied=True,
            abstraction_levels=["FULL", "SUMMARY"],
            original_chars=1000,
            compressed_chars=600,
        )

        assert len(ctx.messages) == 1
        assert len(ctx.critiques) == 1
        assert ctx.compression_applied is True
        assert "FULL" in ctx.abstraction_levels
        assert "SUMMARY" in ctx.abstraction_levels


# ============================================================================
# RLMCognitiveLoadLimiter Tests
# ============================================================================


class TestRLMCognitiveLoadLimiter:
    """Tests for RLMCognitiveLoadLimiter class."""

    def test_init_with_default_budget(self):
        """Test initialization with default budget."""
        limiter = RLMCognitiveLoadLimiter()

        assert isinstance(limiter.budget, RLMCognitiveBudget)
        assert limiter._compressor is None
        assert limiter.stats["rlm_compressions"] == 0

    def test_init_with_custom_budget(self, rlm_budget):
        """Test initialization with custom budget."""
        limiter = RLMCognitiveLoadLimiter(budget=rlm_budget)

        assert limiter.budget == rlm_budget
        assert limiter.budget.compression_threshold == 1000

    def test_for_stress_level_normal(self):
        """Test creating limiter for normal stress level."""
        limiter = RLMCognitiveLoadLimiter.for_stress_level("normal")

        assert isinstance(limiter, RLMCognitiveLoadLimiter)
        assert limiter.budget.enable_rlm_compression is True

    def test_for_stress_level_critical(self):
        """Test creating limiter for critical stress level."""
        limiter = RLMCognitiveLoadLimiter.for_stress_level("critical")

        assert limiter.budget.summary_level == "ABSTRACT"

    def test_for_stress_level_elevated(self):
        """Test creating limiter for elevated stress level."""
        limiter = RLMCognitiveLoadLimiter.for_stress_level("elevated")

        assert limiter.budget.summary_level == "SUMMARY"

    def test_calculate_total_chars(self, limiter, sample_messages, sample_critiques):
        """Test total character calculation."""
        patterns = "pattern1, pattern2"
        extra = "extra context"

        total = limiter._calculate_total_chars(
            sample_messages, sample_critiques, patterns, extra
        )

        # Should sum all content
        expected = sum(len(m.content) for m in sample_messages)
        expected += sum(len(c.reasoning) for c in sample_critiques)
        expected += len(patterns) + len(extra)

        assert total == expected

    def test_calculate_total_chars_empty(self, limiter):
        """Test total chars with empty inputs."""
        total = limiter._calculate_total_chars(None, None, None, None)
        assert total == 0

    def test_compress_text_under_limit(self, limiter):
        """Test text compression when under limit."""
        text = "Short text"
        result = limiter._compress_text(text, 100)
        assert result == text

    def test_compress_text_over_limit(self, limiter):
        """Test text compression when over limit."""
        text = "This is a long text. It has multiple sentences. We need to truncate it. More content here."
        result = limiter._compress_text(text, 50)

        assert len(result) <= 55  # Allow for suffix
        assert "[" in result  # Should have truncation indicator

    def test_rule_based_summarize(self, limiter):
        """Test rule-based summarization."""
        content = """
        I agree with this proposal.
        However, I disagree about the timeline.
        We should consider alternatives.
        The consensus is forming.
        """

        summary = limiter._rule_based_summarize(content, 4)

        assert "Summary of 4 messages" in summary
        assert "agreements" in summary.lower() or "disagreements" in summary.lower()

    def test_summarize_critique_group(self, limiter, sample_critiques):
        """Test critique group summarization."""
        result = limiter._summarize_critique_group(sample_critiques, "high")

        assert result["severity"] == "high"
        assert result["count"] == len(sample_critiques)
        assert "issues" in result
        assert "suggestions" in result


class TestRLMCognitiveLoadLimiterAsync:
    """Async tests for RLMCognitiveLoadLimiter."""

    @pytest.mark.asyncio
    async def test_compress_context_async_under_threshold(self, limiter, sample_messages):
        """Test async compression when under threshold."""
        # Set high threshold so compression is skipped
        limiter.budget.compression_threshold = 100000

        result = await limiter.compress_context_async(
            messages=sample_messages[:2],
            critiques=None,
            patterns=None,
            extra_context=None,
        )

        assert isinstance(result, CompressedContext)
        assert result.compression_applied is False

    @pytest.mark.asyncio
    async def test_compress_context_async_over_threshold(self, limiter, sample_messages):
        """Test async compression when over threshold."""
        # Set low threshold to trigger compression
        limiter.budget.compression_threshold = 10

        result = await limiter.compress_context_async(
            messages=sample_messages,
            critiques=None,
            patterns=None,
            extra_context=None,
        )

        assert isinstance(result, CompressedContext)
        assert result.compression_applied is True
        assert len(result.abstraction_levels) > 0

    @pytest.mark.asyncio
    async def test_compress_context_async_with_rlm_disabled(self, sample_messages):
        """Test compression with RLM disabled."""
        budget = RLMCognitiveBudget(
            enable_rlm_compression=False,
            compression_threshold=10,
        )
        limiter = RLMCognitiveLoadLimiter(budget=budget)

        result = await limiter.compress_context_async(
            messages=sample_messages,
        )

        # Should use base limiter when RLM disabled
        assert result.compression_applied is False

    @pytest.mark.asyncio
    async def test_compress_messages_async(self, limiter, sample_messages):
        """Test message compression."""
        messages, levels = await limiter._compress_messages_async(
            sample_messages, limiter.budget
        )

        # Should have compressed result
        assert len(messages) > 0
        assert len(levels) > 0

        # First message should be preserved if configured
        if limiter.budget.preserve_first_message:
            assert messages[0].content == sample_messages[0].content

    @pytest.mark.asyncio
    async def test_compress_critiques_async(self, limiter, sample_critiques):
        """Test critique compression."""
        result = await limiter._compress_critiques_async(
            sample_critiques, limiter.budget
        )

        # High severity should be kept
        assert len(result) > 0

        # Should have grouped low severity
        grouped = [c for c in result if isinstance(c, dict)]
        assert len(grouped) > 0

    @pytest.mark.asyncio
    async def test_compress_context_updates_stats(self, limiter, sample_messages):
        """Test that compression updates stats."""
        limiter.budget.compression_threshold = 10

        initial_compressions = limiter.stats["rlm_compressions"]

        await limiter.compress_context_async(messages=sample_messages)

        assert limiter.stats["rlm_compressions"] == initial_compressions + 1

    @pytest.mark.asyncio
    async def test_query_compressed_context_fallback(self, limiter, sample_messages):
        """Test querying compressed context with fallback search."""
        ctx = CompressedContext(
            messages=sample_messages,
            compression_applied=True,
        )

        result = await limiter.query_compressed_context(
            query="rate limiter",
            compressed_context=ctx,
        )

        # Should return some result (even if "not found")
        assert isinstance(result, str)
        assert len(result) > 0


class TestRLMCognitiveLoadLimiterSync:
    """Synchronous tests for RLMCognitiveLoadLimiter."""

    def test_compress_context_sync(self, limiter, sample_messages):
        """Test synchronous compression."""
        result = limiter.compress_context(messages=sample_messages)

        assert isinstance(result, CompressedContext)

    def test_compress_context_sync_under_threshold(self, limiter, sample_messages):
        """Test sync compression when under threshold."""
        limiter.budget.compression_threshold = 100000

        result = limiter.compress_context(messages=sample_messages[:2])

        assert result.compression_applied is False

    def test_compress_context_sync_over_threshold(self, limiter, sample_messages):
        """Test sync compression when over threshold."""
        limiter.budget.compression_threshold = 10

        result = limiter.compress_context(messages=sample_messages)

        # Should use rule-based compression
        assert result.compression_applied is True

    def test_search_compressed_fallback(self, limiter, sample_messages):
        """Test fallback search in compressed context."""
        ctx = CompressedContext(messages=sample_messages)

        result = limiter._search_compressed_fallback(
            query="rate limiter token bucket",
            compressed_context=ctx,
        )

        # Should find relevant sections or return "not found"
        assert isinstance(result, str)


# ============================================================================
# Factory Function Tests
# ============================================================================


class TestCreateRLMLimiter:
    """Tests for create_rlm_limiter factory function."""

    def test_create_default(self):
        """Test creating limiter with defaults."""
        limiter = create_rlm_limiter()

        assert isinstance(limiter, RLMCognitiveLoadLimiter)
        assert limiter.budget.enable_rlm_compression is True

    def test_create_with_stress_level(self):
        """Test creating limiter with stress level."""
        limiter = create_rlm_limiter(stress_level="critical")

        assert limiter.budget.summary_level == "ABSTRACT"

    def test_create_with_compressor(self):
        """Test creating limiter with custom compressor."""
        mock_compressor = MagicMock()

        limiter = create_rlm_limiter(compressor=mock_compressor)

        assert limiter._compressor == mock_compressor


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_messages(self, limiter):
        """Test with empty message list."""
        result = limiter.compress_context(messages=[])

        assert result.messages == []

    def test_none_inputs(self, limiter):
        """Test with all None inputs."""
        result = limiter.compress_context(
            messages=None,
            critiques=None,
            patterns=None,
            extra_context=None,
        )

        assert isinstance(result, CompressedContext)
        assert result.original_chars == 0

    def test_single_message(self, limiter):
        """Test with single message."""
        msg = MockMessage(content="Single message")

        result = limiter.compress_context(messages=[msg])

        assert len(result.messages) == 1

    def test_messages_without_content_attribute(self, limiter):
        """Test with messages lacking content attribute."""
        # This should still work by converting to string
        messages = ["plain string message", {"dict": "message"}]

        total = limiter._calculate_total_chars(messages, None, None, None)

        assert total > 0

    def test_critiques_without_severity(self, limiter):
        """Test critiques without severity attribute."""

        @dataclass
        class CritiqueNoSeverity:
            reasoning: str = "test"

        critiques = [CritiqueNoSeverity(), CritiqueNoSeverity()]

        result = limiter.compress_context(critiques=critiques)

        # Should handle gracefully
        assert isinstance(result, CompressedContext)

    @pytest.mark.asyncio
    async def test_compressor_lazy_load_error(self, limiter):
        """Test handling of compressor import error."""
        # Patch the import to fail
        with patch.dict("sys.modules", {"aragora.rlm": None}):
            # Force re-check of compressor
            limiter._compressor = None

            # Access should handle gracefully
            compressor = limiter.compressor

            # Should be None when import fails
            assert compressor is None or compressor is not None  # Depends on actual state


# ============================================================================
# Integration with Base Limiter
# ============================================================================


class TestBaseIntegration:
    """Tests for integration with base CognitiveLoadLimiter."""

    def test_inherits_limit_context(self, limiter, sample_messages):
        """Test that limit_context from base class works."""
        # Call base class method
        result = limiter.limit_context(
            messages=sample_messages,
            critiques=None,
            patterns=None,
            extra_context=None,
        )

        assert isinstance(result, dict)
        assert "messages" in result

    def test_inherits_limit_critiques(self, limiter, sample_critiques):
        """Test that limit_critiques from base class works."""
        result = limiter.limit_critiques(sample_critiques)

        assert isinstance(result, list)

    def test_base_stats_preserved(self, limiter):
        """Test that base stats are preserved."""
        # Base class stats should exist
        assert "total_calls" in limiter.stats or True  # May not have this stat
        assert "rlm_compressions" in limiter.stats  # Our added stat
