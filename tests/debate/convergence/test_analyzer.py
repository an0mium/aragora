"""Tests for aragora.debate.convergence.analyzer — AdvancedConvergenceAnalyzer."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from aragora.debate.convergence.analyzer import AdvancedConvergenceAnalyzer
from aragora.debate.convergence.metrics import (
    AdvancedConvergenceMetrics,
    ArgumentDiversityMetric,
    EvidenceConvergenceMetric,
    StanceVolatilityMetric,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class FakeBackend:
    """Minimal SimilarityBackend stub."""

    def compute_similarity(self, text1: str, text2: str) -> float:
        # Simple word-overlap heuristic for deterministic tests
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0.0
        overlap = len(words1 & words2)
        return overlap / max(len(words1), len(words2))


@pytest.fixture
def backend():
    return FakeBackend()


@pytest.fixture
def analyzer(backend):
    return AdvancedConvergenceAnalyzer(similarity_backend=backend)


@pytest.fixture
def cached_analyzer(backend):
    return AdvancedConvergenceAnalyzer(
        similarity_backend=backend,
        debate_id="test-debate-123",
        enable_cache=True,
    )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestInit:
    def test_default_backend(self):
        a = AdvancedConvergenceAnalyzer()
        assert a.backend is not None

    def test_custom_backend(self, backend):
        a = AdvancedConvergenceAnalyzer(similarity_backend=backend)
        assert a.backend is backend

    def test_cache_enabled_with_debate_id(self, backend):
        a = AdvancedConvergenceAnalyzer(
            similarity_backend=backend,
            debate_id="d1",
            enable_cache=True,
        )
        assert a._enable_cache is True
        assert a._similarity_cache is not None

    def test_cache_disabled_without_debate_id(self, backend):
        a = AdvancedConvergenceAnalyzer(
            similarity_backend=backend,
            debate_id=None,
            enable_cache=True,
        )
        assert a._enable_cache is False

    def test_cache_disabled_explicitly(self, backend):
        a = AdvancedConvergenceAnalyzer(
            similarity_backend=backend,
            debate_id="d1",
            enable_cache=False,
        )
        assert a._enable_cache is False


# ---------------------------------------------------------------------------
# extract_arguments
# ---------------------------------------------------------------------------


class TestExtractArguments:
    def test_empty_text(self, analyzer):
        assert analyzer.extract_arguments("") == []

    def test_short_sentence_filtered(self, analyzer):
        assert analyzer.extract_arguments("Too short.") == []

    def test_substantive_sentences_extracted(self, analyzer):
        text = "This is a long argument about security. It has many important words to consider."
        args = analyzer.extract_arguments(text)
        assert len(args) >= 1
        for arg in args:
            assert len(arg.split()) > 5

    def test_multiple_sentences(self, analyzer):
        text = (
            "The first argument is that we should use encryption for all data. "
            "The second point is that rate limiting prevents abuse effectively. "
            "Ok."
        )
        args = analyzer.extract_arguments(text)
        assert len(args) == 2  # Third sentence too short


# ---------------------------------------------------------------------------
# extract_citations
# ---------------------------------------------------------------------------


class TestExtractCitations:
    def test_empty(self, analyzer):
        assert analyzer.extract_citations("") == set()

    def test_url(self, analyzer):
        cites = analyzer.extract_citations("See https://example.com/paper for details.")
        assert "https://example.com/paper" in cites

    def test_academic_citation(self, analyzer):
        cites = analyzer.extract_citations("This was shown by (Smith, 2024).")
        assert "(Smith, 2024)" in cites

    def test_numbered_citation(self, analyzer):
        cites = analyzer.extract_citations("As mentioned in [1] and [2].")
        assert "[1]" in cites
        assert "[2]" in cites

    def test_quoted_source(self, analyzer):
        cites = analyzer.extract_citations("According to John Smith, this is correct.")
        assert any("John Smith" in c for c in cites)

    def test_multiple_types(self, analyzer):
        text = "See https://example.com and (Jones, 2023) and [3]."
        cites = analyzer.extract_citations(text)
        assert len(cites) >= 3


# ---------------------------------------------------------------------------
# detect_stance
# ---------------------------------------------------------------------------


class TestDetectStance:
    def test_support(self, analyzer):
        assert analyzer.detect_stance("I agree and support this recommendation") == "support"

    def test_oppose(self, analyzer):
        assert analyzer.detect_stance("I disagree and oppose this proposal strongly") == "oppose"

    def test_neutral(self, analyzer):
        assert (
            analyzer.detect_stance("The weather is nice today and birds are singing") == "neutral"
        )

    def test_mixed(self, analyzer):
        stance = analyzer.detect_stance(
            "I agree with some points however I disagree on the other hand"
        )
        assert stance in ("mixed", "support", "oppose")  # depends on counts


# ---------------------------------------------------------------------------
# compute_argument_diversity
# ---------------------------------------------------------------------------


class TestComputeArgumentDiversity:
    def test_empty_responses(self, analyzer):
        result = analyzer.compute_argument_diversity({})
        assert result.unique_arguments == 0
        assert result.total_arguments == 0
        assert result.diversity_score == 0.0

    def test_no_substantive_arguments(self, analyzer):
        result = analyzer.compute_argument_diversity({"a": "short", "b": "tiny"})
        assert result.total_arguments == 0

    def test_diverse_arguments(self, analyzer):
        responses = {
            "claude": "We should implement encryption using AES-256 for all data at rest.",
            "gpt": "The rate limiting system needs Redis-based sliding window counters.",
        }
        result = analyzer.compute_argument_diversity(responses, use_optimized=False)
        assert result.total_arguments >= 2
        assert result.diversity_score > 0

    def test_identical_arguments_low_diversity(self, analyzer):
        same = "We should implement encryption using AES-256 for all data at rest securely."
        responses = {"a": same, "b": same, "c": same}
        result = analyzer.compute_argument_diversity(responses, use_optimized=False)
        assert result.diversity_score < 0.5

    def test_optimized_fallback(self, analyzer):
        """When optimized path raises, falls back to O(n^2)."""
        responses = {
            "a": "First argument about security protocols and encryption standards for the system.",
            "b": "Second argument about database optimization and query performance tuning strategies.",
            "c": "Third point about user interface design patterns and accessibility compliance.",
            "d": "Fourth observation on continuous integration pipelines and deployment automation.",
            "e": "Fifth claim regarding API versioning strategies and backward compatibility maintenance.",
        }
        with patch.object(
            analyzer,
            "_compute_diversity_optimized",
            side_effect=ValueError("no embeddings"),
        ):
            result = analyzer.compute_argument_diversity(responses, use_optimized=True)
            assert result.total_arguments >= 5


# ---------------------------------------------------------------------------
# compute_evidence_convergence
# ---------------------------------------------------------------------------


class TestComputeEvidenceConvergence:
    def test_no_citations(self, analyzer):
        result = analyzer.compute_evidence_convergence({"a": "no refs", "b": "none here"})
        assert result.shared_citations == 0
        assert result.total_citations == 0
        assert result.overlap_score == 0.0

    def test_shared_citations(self, analyzer):
        responses = {
            "claude": "According to (Smith, 2024), we need encryption. See https://example.com.",
            "gpt": "As (Smith, 2024) noted, encryption is critical. Also https://other.com.",
        }
        result = analyzer.compute_evidence_convergence(responses)
        assert result.shared_citations >= 1  # Smith 2024 shared
        assert result.total_citations >= 2
        assert result.overlap_score > 0

    def test_no_overlap(self, analyzer):
        responses = {
            "claude": "See https://example.com for details.",
            "gpt": "According to (Jones, 2023), this works.",
        }
        result = analyzer.compute_evidence_convergence(responses)
        assert result.shared_citations == 0

    def test_single_agent(self, analyzer):
        result = analyzer.compute_evidence_convergence({"claude": "See https://example.com."})
        assert result.overlap_score == 0.0


# ---------------------------------------------------------------------------
# compute_stance_volatility
# ---------------------------------------------------------------------------


class TestComputeStanceVolatility:
    def test_single_round(self, analyzer):
        result = analyzer.compute_stance_volatility([{"a": "I agree"}])
        assert result.stance_changes == 0
        assert result.total_responses == 0
        assert result.volatility_score == 0.0

    def test_no_changes(self, analyzer):
        history = [
            {"claude": "I agree with this approach", "gpt": "I disagree strongly"},
            {"claude": "I still agree with it", "gpt": "I still disagree firmly"},
        ]
        result = analyzer.compute_stance_volatility(history)
        assert result.volatility_score <= 0.5

    def test_high_volatility(self, analyzer):
        history = [
            {"claude": "I definitely agree and support this"},
            {"claude": "I strongly disagree and reject this"},
            {"claude": "I agree and favor this approach"},
        ]
        result = analyzer.compute_stance_volatility(history)
        assert result.stance_changes >= 1
        assert result.volatility_score > 0


# ---------------------------------------------------------------------------
# _compute_similarity_cached
# ---------------------------------------------------------------------------


class TestCachedSimilarity:
    def test_without_cache(self, analyzer):
        sim = analyzer._compute_similarity_cached("hello world", "hello world")
        assert sim > 0

    def test_with_cache_stores_and_retrieves(self, cached_analyzer):
        sim1 = cached_analyzer._compute_similarity_cached("foo bar baz", "foo bar qux")
        # Second call should hit cache
        sim2 = cached_analyzer._compute_similarity_cached("foo bar baz", "foo bar qux")
        assert sim1 == sim2

    def test_cache_stats(self, cached_analyzer):
        cached_analyzer._compute_similarity_cached("hello", "world")
        stats = cached_analyzer.get_cache_stats()
        assert stats is not None
        assert "total_entries" in stats or "size" in stats or isinstance(stats, dict)

    def test_no_cache_stats_when_disabled(self, analyzer):
        assert analyzer.get_cache_stats() is None


# ---------------------------------------------------------------------------
# analyze (comprehensive)
# ---------------------------------------------------------------------------


class TestAnalyze:
    def test_basic_analysis(self, analyzer):
        current = {
            "claude": "We should use encryption for all data at rest and in transit.",
            "gpt": "I agree we should implement encryption across the entire system.",
        }
        result = analyzer.analyze(current)
        assert isinstance(result, AdvancedConvergenceMetrics)
        assert isinstance(result.argument_diversity, ArgumentDiversityMetric)
        assert isinstance(result.evidence_convergence, EvidenceConvergenceMetric)

    def test_with_previous_responses(self, analyzer):
        prev = {
            "claude": "We need security improvements across the board for the system.",
            "gpt": "Performance should be our top priority for the application.",
        }
        curr = {
            "claude": "We need security improvements across the board for the system.",
            "gpt": "I now agree security should be our top priority for the system.",
        }
        result = analyzer.analyze(curr, previous_responses=prev)
        assert result.semantic_similarity > 0  # claude's response unchanged

    def test_without_previous_responses(self, analyzer):
        result = analyzer.analyze({"a": "Some text about encryption."})
        assert result.semantic_similarity == 0.0

    def test_with_response_history(self, analyzer):
        history = [
            {"claude": "I agree with encryption", "gpt": "I disagree"},
            {"claude": "I still agree with encryption", "gpt": "Now I also agree"},
        ]
        result = analyzer.analyze(
            history[-1],
            response_history=history,
        )
        assert result.stance_volatility is not None
        assert isinstance(result.stance_volatility, StanceVolatilityMetric)

    def test_without_history_no_stance(self, analyzer):
        result = analyzer.analyze({"a": "hello world test"})
        assert result.stance_volatility is None

    def test_domain_passthrough(self, analyzer):
        result = analyzer.analyze({"a": "test content here"}, domain="healthcare")
        assert result.domain == "healthcare"

    def test_no_common_agents(self, analyzer):
        prev = {"claude": "hello world text"}
        curr = {"gpt": "different agent text"}
        result = analyzer.analyze(curr, previous_responses=prev)
        assert result.semantic_similarity == 0.0

    def test_overall_convergence_computed(self, analyzer):
        result = analyzer.analyze(
            {"a": "Some argument about system design patterns and architecture."}
        )
        # compute_overall_score is called, sets overall_convergence
        assert hasattr(result, "overall_convergence")


# ---------------------------------------------------------------------------
# cleanup
# ---------------------------------------------------------------------------


class TestCleanup:
    def test_cleanup_with_debate_id(self, cached_analyzer):
        # Should not raise
        cached_analyzer.cleanup()

    def test_cleanup_without_debate_id(self, analyzer):
        # No debate_id, cleanup is a no-op
        analyzer.cleanup()
