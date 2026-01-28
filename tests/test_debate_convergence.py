"""
Tests for Convergence Detection module.

Tests cover:
- Similarity backends (Jaccard, TF-IDF, SentenceTransformer)
- ConvergenceResult dataclass
- Advanced convergence metrics (ArgumentDiversity, EvidenceConvergence, StanceVolatility)
- AdvancedConvergenceAnalyzer
- ConvergenceDetector main class
- get_similarity_backend factory function
"""

from __future__ import annotations

import pytest
from unittest.mock import Mock, patch, MagicMock

# Import centralized skip markers
from tests.conftest import requires_sklearn, REQUIRES_SKLEARN

from aragora.debate.convergence import (
    SimilarityBackend,
    JaccardBackend,
    TFIDFBackend,
    ConvergenceResult,
    ArgumentDiversityMetric,
    EvidenceConvergenceMetric,
    StanceVolatilityMetric,
    AdvancedConvergenceMetrics,
    AdvancedConvergenceAnalyzer,
    ConvergenceDetector,
    get_similarity_backend,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def jaccard_backend():
    """Create Jaccard backend for testing."""
    backend = JaccardBackend()
    JaccardBackend.clear_cache()
    return backend


@pytest.fixture
def sample_responses():
    """Sample agent responses for testing."""
    return {
        "agent_a": "I agree that climate change is a serious issue. We should take action.",
        "agent_b": "Climate change is indeed serious and requires immediate action.",
        "agent_c": "The issue of climate change demands urgent attention and action.",
    }


@pytest.fixture
def divergent_responses():
    """Sample divergent responses for testing."""
    return {
        "agent_a": "Climate change is serious and requires action.",
        "agent_b": "Economic growth should be prioritized over environmental concerns.",
        "agent_c": "We should focus on technology solutions rather than policy.",
    }


@pytest.fixture
def previous_responses():
    """Previous round responses for convergence testing."""
    return {
        "agent_a": "I think climate change is an important issue to address.",
        "agent_b": "Climate change requires our attention and action.",
        "agent_c": "Addressing climate change should be a priority.",
    }


# ============================================================================
# JaccardBackend Tests
# ============================================================================


class TestJaccardBackend:
    """Tests for JaccardBackend."""

    def test_identical_texts(self, jaccard_backend):
        """Test identical texts return 1.0 similarity."""
        text = "This is a test sentence"
        similarity = jaccard_backend.compute_similarity(text, text)
        assert similarity == 1.0

    def test_completely_different_texts(self, jaccard_backend):
        """Test completely different texts return 0.0 similarity."""
        text1 = "apple banana cherry"
        text2 = "dog elephant frog"
        similarity = jaccard_backend.compute_similarity(text1, text2)
        assert similarity == 0.0

    def test_partial_overlap(self, jaccard_backend):
        """Test partial overlap returns value between 0 and 1."""
        text1 = "apple banana cherry"
        text2 = "apple banana date"
        similarity = jaccard_backend.compute_similarity(text1, text2)
        # Intersection: {apple, banana} = 2, Union: {apple, banana, cherry, date} = 4
        assert similarity == pytest.approx(0.5)

    def test_empty_text_returns_zero(self, jaccard_backend):
        """Test empty text returns 0.0 similarity."""
        assert jaccard_backend.compute_similarity("", "test") == 0.0
        assert jaccard_backend.compute_similarity("test", "") == 0.0
        assert jaccard_backend.compute_similarity("", "") == 0.0

    def test_case_insensitive(self, jaccard_backend):
        """Test comparison is case insensitive."""
        text1 = "Apple Banana Cherry"
        text2 = "apple banana cherry"
        similarity = jaccard_backend.compute_similarity(text1, text2)
        assert similarity == 1.0

    def test_caching(self, jaccard_backend):
        """Test similarity results are cached."""
        text1 = "test sentence one"
        text2 = "test sentence two"

        # First call
        similarity1 = jaccard_backend.compute_similarity(text1, text2)

        # Second call (should be cached)
        similarity2 = jaccard_backend.compute_similarity(text1, text2)

        assert similarity1 == similarity2

    def test_cache_symmetric(self, jaccard_backend):
        """Test cache is symmetric for (a,b) and (b,a)."""
        text1 = "hello world"
        text2 = "world hello again"

        sim1 = jaccard_backend.compute_similarity(text1, text2)
        sim2 = jaccard_backend.compute_similarity(text2, text1)

        assert sim1 == sim2

    def test_clear_cache(self, jaccard_backend):
        """Test cache can be cleared."""
        jaccard_backend.compute_similarity("a", "b")
        assert len(JaccardBackend._similarity_cache) > 0

        JaccardBackend.clear_cache()
        assert len(JaccardBackend._similarity_cache) == 0

    def test_compute_batch_similarity(self, jaccard_backend):
        """Test batch similarity computation."""
        texts = ["hello world", "hello there", "hello everyone"]
        avg_sim = jaccard_backend.compute_batch_similarity(texts)

        # Should be between 0 and 1
        assert 0.0 <= avg_sim <= 1.0

    def test_compute_batch_similarity_single_text(self, jaccard_backend):
        """Test batch similarity with single text returns 1.0."""
        texts = ["hello world"]
        avg_sim = jaccard_backend.compute_batch_similarity(texts)
        assert avg_sim == 1.0

    def test_compute_batch_similarity_empty(self, jaccard_backend):
        """Test batch similarity with empty list returns 1.0."""
        avg_sim = jaccard_backend.compute_batch_similarity([])
        assert avg_sim == 1.0


# ============================================================================
# TFIDFBackend Tests
# ============================================================================


@pytest.mark.skipif(requires_sklearn, reason=REQUIRES_SKLEARN)
class TestTFIDFBackend:
    """Tests for TFIDFBackend."""

    @pytest.fixture
    def tfidf_backend(self):
        """Create TF-IDF backend for testing."""
        backend = TFIDFBackend()
        TFIDFBackend.clear_cache()
        return backend

    def test_identical_texts(self, tfidf_backend):
        """Test identical texts return high similarity."""
        text = "This is a test sentence for similarity"
        similarity = tfidf_backend.compute_similarity(text, text)
        assert similarity == pytest.approx(1.0, abs=0.01)

    def test_completely_different_texts(self, tfidf_backend):
        """Test completely different texts return low similarity."""
        text1 = "apple banana cherry date"
        text2 = "elephant frog giraffe hippo"
        similarity = tfidf_backend.compute_similarity(text1, text2)
        assert similarity < 0.3

    def test_partial_overlap(self, tfidf_backend):
        """Test partial overlap returns value between 0 and 1."""
        text1 = "the quick brown fox jumps"
        text2 = "the lazy brown dog sleeps"
        similarity = tfidf_backend.compute_similarity(text1, text2)
        assert 0.0 < similarity < 1.0

    def test_empty_text_returns_zero(self, tfidf_backend):
        """Test empty text returns 0.0 similarity."""
        assert tfidf_backend.compute_similarity("", "test") == 0.0
        assert tfidf_backend.compute_similarity("test", "") == 0.0

    def test_caching(self, tfidf_backend):
        """Test similarity results are cached."""
        text1 = "machine learning is fascinating"
        text2 = "deep learning is powerful"

        sim1 = tfidf_backend.compute_similarity(text1, text2)
        sim2 = tfidf_backend.compute_similarity(text1, text2)

        assert sim1 == sim2


# ============================================================================
# ConvergenceResult Tests
# ============================================================================


class TestConvergenceResult:
    """Tests for ConvergenceResult dataclass."""

    def test_create_converged_result(self):
        """Test creating a converged result."""
        result = ConvergenceResult(
            converged=True,
            status="converged",
            min_similarity=0.9,
            avg_similarity=0.92,
            per_agent_similarity={"agent_a": 0.9, "agent_b": 0.94},
            consecutive_stable_rounds=2,
        )

        assert result.converged is True
        assert result.status == "converged"
        assert result.min_similarity == 0.9
        assert result.avg_similarity == 0.92
        assert len(result.per_agent_similarity) == 2

    def test_create_refining_result(self):
        """Test creating a refining result."""
        result = ConvergenceResult(
            converged=False,
            status="refining",
            min_similarity=0.6,
            avg_similarity=0.7,
        )

        assert result.converged is False
        assert result.status == "refining"

    def test_create_diverging_result(self):
        """Test creating a diverging result."""
        result = ConvergenceResult(
            converged=False,
            status="diverging",
            min_similarity=0.3,
            avg_similarity=0.35,
        )

        assert result.status == "diverging"

    def test_default_per_agent_similarity(self):
        """Test default value for per_agent_similarity."""
        result = ConvergenceResult(
            converged=False,
            status="refining",
            min_similarity=0.5,
            avg_similarity=0.6,
        )

        assert result.per_agent_similarity == {}
        assert result.consecutive_stable_rounds == 0


# ============================================================================
# Advanced Metrics Tests
# ============================================================================


class TestArgumentDiversityMetric:
    """Tests for ArgumentDiversityMetric."""

    def test_is_converging_low_diversity(self):
        """Test is_converging returns True for low diversity."""
        metric = ArgumentDiversityMetric(
            unique_arguments=2,
            total_arguments=10,
            diversity_score=0.2,
        )
        assert metric.is_converging is True

    def test_is_converging_high_diversity(self):
        """Test is_converging returns False for high diversity."""
        metric = ArgumentDiversityMetric(
            unique_arguments=8,
            total_arguments=10,
            diversity_score=0.8,
        )
        assert metric.is_converging is False


class TestEvidenceConvergenceMetric:
    """Tests for EvidenceConvergenceMetric."""

    def test_is_converging_high_overlap(self):
        """Test is_converging returns True for high overlap."""
        metric = EvidenceConvergenceMetric(
            shared_citations=8,
            total_citations=10,
            overlap_score=0.8,
        )
        assert metric.is_converging is True

    def test_is_converging_low_overlap(self):
        """Test is_converging returns False for low overlap."""
        metric = EvidenceConvergenceMetric(
            shared_citations=2,
            total_citations=10,
            overlap_score=0.2,
        )
        assert metric.is_converging is False


class TestStanceVolatilityMetric:
    """Tests for StanceVolatilityMetric."""

    def test_is_stable_low_volatility(self):
        """Test is_stable returns True for low volatility."""
        metric = StanceVolatilityMetric(
            stance_changes=1,
            total_responses=10,
            volatility_score=0.1,
        )
        assert metric.is_stable is True

    def test_is_stable_high_volatility(self):
        """Test is_stable returns False for high volatility."""
        metric = StanceVolatilityMetric(
            stance_changes=8,
            total_responses=10,
            volatility_score=0.8,
        )
        assert metric.is_stable is False


class TestAdvancedConvergenceMetrics:
    """Tests for AdvancedConvergenceMetrics."""

    def test_compute_overall_score_semantic_only(self):
        """Test overall score with semantic similarity only."""
        metrics = AdvancedConvergenceMetrics(
            semantic_similarity=0.8,
        )
        score = metrics.compute_overall_score()

        # 0.8 * 0.4 = 0.32
        assert score == pytest.approx(0.32)

    def test_compute_overall_score_with_all_metrics(self):
        """Test overall score with all metrics."""
        metrics = AdvancedConvergenceMetrics(
            semantic_similarity=0.8,
            argument_diversity=ArgumentDiversityMetric(
                2, 10, 0.2
            ),  # Low diversity = high convergence
            evidence_convergence=EvidenceConvergenceMetric(8, 10, 0.8),
            stance_volatility=StanceVolatilityMetric(
                1, 10, 0.1
            ),  # Low volatility = high convergence
        )
        score = metrics.compute_overall_score()

        # Should be high since all indicators suggest convergence
        assert score > 0.6

    def test_to_dict(self):
        """Test to_dict conversion."""
        metrics = AdvancedConvergenceMetrics(
            semantic_similarity=0.8,
            domain="climate",
        )
        metrics.compute_overall_score()

        d = metrics.to_dict()

        assert d["semantic_similarity"] == 0.8
        assert d["domain"] == "climate"
        assert "overall_convergence" in d


# ============================================================================
# AdvancedConvergenceAnalyzer Tests
# ============================================================================


class TestAdvancedConvergenceAnalyzer:
    """Tests for AdvancedConvergenceAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer with Jaccard backend."""
        return AdvancedConvergenceAnalyzer(similarity_backend=JaccardBackend())

    def test_extract_arguments(self, analyzer):
        """Test argument extraction from text."""
        text = "This is a short one. This is a longer sentence that should be extracted as an argument. Another short. This sentence is also long enough to be considered an argument."

        arguments = analyzer.extract_arguments(text)

        # Should extract sentences with > 5 words
        assert len(arguments) == 2

    def test_extract_citations_urls(self, analyzer):
        """Test URL citation extraction."""
        text = "See https://example.com/article for more info."

        citations = analyzer.extract_citations(text)

        assert "https://example.com/article" in citations

    def test_extract_citations_academic(self, analyzer):
        """Test academic citation extraction."""
        text = "According to (Smith, 2024), this is true."

        citations = analyzer.extract_citations(text)

        assert "(Smith, 2024)" in citations

    def test_extract_citations_numbered(self, analyzer):
        """Test numbered citation extraction."""
        text = "This is supported by evidence [1] and [2]."

        citations = analyzer.extract_citations(text)

        assert "[1]" in citations
        assert "[2]" in citations

    def test_detect_stance_support(self, analyzer):
        """Test support stance detection."""
        text = "I strongly agree with this proposal and support its implementation."

        stance = analyzer.detect_stance(text)

        assert stance == "support"

    def test_detect_stance_oppose(self, analyzer):
        """Test oppose stance detection."""
        text = "I disagree with this approach and must reject the proposal."

        stance = analyzer.detect_stance(text)

        assert stance == "oppose"

    def test_detect_stance_neutral(self, analyzer):
        """Test neutral stance detection."""
        text = "There are various options to consider."

        stance = analyzer.detect_stance(text)

        assert stance == "neutral"

    def test_compute_argument_diversity(self, analyzer, sample_responses):
        """Test argument diversity computation."""
        diversity = analyzer.compute_argument_diversity(sample_responses)

        assert isinstance(diversity, ArgumentDiversityMetric)
        assert diversity.total_arguments > 0
        assert 0.0 <= diversity.diversity_score <= 1.0

    def test_compute_evidence_convergence(self, analyzer):
        """Test evidence convergence computation."""
        responses = {
            "agent_a": "According to (Smith, 2024), this is true. See [1].",
            "agent_b": "Per (Smith, 2024), I concur. Reference [1] confirms.",
        }

        convergence = analyzer.compute_evidence_convergence(responses)

        assert isinstance(convergence, EvidenceConvergenceMetric)
        # Both agents cite Smith and [1]
        assert convergence.shared_citations >= 1

    def test_compute_stance_volatility(self, analyzer):
        """Test stance volatility computation."""
        history = [
            {"agent_a": "I agree with this.", "agent_b": "I disagree."},
            {"agent_a": "I agree still.", "agent_b": "Now I agree too."},
        ]

        volatility = analyzer.compute_stance_volatility(history)

        assert isinstance(volatility, StanceVolatilityMetric)
        assert volatility.stance_changes >= 0

    def test_analyze(self, analyzer, sample_responses, previous_responses):
        """Test full analysis."""
        metrics = analyzer.analyze(
            current_responses=sample_responses,
            previous_responses=previous_responses,
            domain="climate",
        )

        assert isinstance(metrics, AdvancedConvergenceMetrics)
        assert metrics.domain == "climate"
        assert metrics.argument_diversity is not None
        assert metrics.evidence_convergence is not None


# ============================================================================
# ConvergenceDetector Tests
# ============================================================================


class TestConvergenceDetector:
    """Tests for ConvergenceDetector main class."""

    @pytest.fixture
    def detector(self):
        """Create detector with Jaccard backend."""
        detector = ConvergenceDetector(
            convergence_threshold=0.85,
            divergence_threshold=0.40,
            min_rounds_before_check=1,
            consecutive_rounds_needed=1,
        )
        # Force Jaccard backend
        detector.backend = JaccardBackend()
        return detector

    def test_init_defaults(self):
        """Test default initialization."""
        detector = ConvergenceDetector()

        assert detector.convergence_threshold == 0.85
        assert detector.divergence_threshold == 0.40
        assert detector.min_rounds_before_check == 1
        assert detector.consecutive_rounds_needed == 1

    def test_init_custom_thresholds(self):
        """Test custom threshold initialization."""
        detector = ConvergenceDetector(
            convergence_threshold=0.9,
            divergence_threshold=0.3,
            min_rounds_before_check=2,
            consecutive_rounds_needed=3,
        )

        assert detector.convergence_threshold == 0.9
        assert detector.divergence_threshold == 0.3
        assert detector.min_rounds_before_check == 2
        assert detector.consecutive_rounds_needed == 3

    def test_check_convergence_too_early(self, detector):
        """Test convergence check returns None if too early."""
        result = detector.check_convergence(
            current_responses={"a": "test"},
            previous_responses={"a": "test"},
            round_number=1,
        )

        assert result is None

    def test_check_convergence_no_common_agents(self, detector):
        """Test convergence check with no common agents."""
        result = detector.check_convergence(
            current_responses={"a": "test"},
            previous_responses={"b": "test"},
            round_number=2,
        )

        assert result is None

    def test_check_convergence_identical_responses(self, detector):
        """Test convergence with identical responses."""
        responses = {"agent_a": "This is the response", "agent_b": "Another response here"}

        result = detector.check_convergence(
            current_responses=responses,
            previous_responses=responses,
            round_number=2,
        )

        assert result is not None
        assert result.min_similarity == 1.0
        assert result.converged is True
        assert result.status == "converged"

    def test_check_convergence_diverging(self, detector, sample_responses, divergent_responses):
        """Test diverging detection."""
        result = detector.check_convergence(
            current_responses=divergent_responses,
            previous_responses=sample_responses,
            round_number=2,
        )

        assert result is not None
        # Should detect divergence due to different topics
        assert result.min_similarity < detector.convergence_threshold

    def test_check_convergence_refining(self, detector, sample_responses, previous_responses):
        """Test refining status detection."""
        result = detector.check_convergence(
            current_responses=sample_responses,
            previous_responses=previous_responses,
            round_number=2,
        )

        assert result is not None
        assert result.status in ["converged", "refining", "diverging"]

    def test_consecutive_stable_rounds(self, detector):
        """Test consecutive stable rounds tracking."""
        responses = {"a": "same response text here for testing"}

        detector.check_convergence(responses, responses, round_number=2)
        detector.check_convergence(responses, responses, round_number=3)

        assert detector.consecutive_stable_count >= 1

    def test_reset(self, detector):
        """Test reset clears consecutive count."""
        responses = {"a": "same response text here"}
        detector.check_convergence(responses, responses, round_number=2)

        detector.reset()

        assert detector.consecutive_stable_count == 0

    def test_per_agent_similarity(self, detector):
        """Test per-agent similarity is tracked."""
        responses = {
            "agent_a": "This is agent A response",
            "agent_b": "This is agent B response",
        }

        result = detector.check_convergence(
            current_responses=responses,
            previous_responses=responses,
            round_number=2,
        )

        assert "agent_a" in result.per_agent_similarity
        assert "agent_b" in result.per_agent_similarity


# ============================================================================
# get_similarity_backend Tests
# ============================================================================


class TestGetSimilarityBackend:
    """Tests for get_similarity_backend factory function."""

    def test_get_jaccard_backend(self):
        """Test getting Jaccard backend."""
        backend = get_similarity_backend("jaccard")

        assert isinstance(backend, JaccardBackend)

    def test_get_tfidf_backend(self):
        """Test getting TF-IDF backend."""
        try:
            backend = get_similarity_backend("tfidf")
            assert isinstance(backend, TFIDFBackend)
        except ImportError:
            pytest.skip("scikit-learn not available")

    def test_get_auto_backend(self):
        """Test auto backend selection."""
        backend = get_similarity_backend("auto")

        # Should return some kind of backend
        assert isinstance(backend, SimilarityBackend)

    def test_unknown_backend_uses_auto(self):
        """Test unknown backend falls back to auto."""
        # This will try auto-selection
        backend = get_similarity_backend("unknown")

        # Should still return a backend
        assert isinstance(backend, SimilarityBackend)


# ============================================================================
# Integration Tests
# ============================================================================


class TestConvergenceIntegration:
    """Integration tests for convergence detection."""

    def test_full_debate_convergence_flow(self):
        """Test full convergence detection flow across multiple rounds."""
        detector = ConvergenceDetector(
            convergence_threshold=0.8,
            divergence_threshold=0.3,
            min_rounds_before_check=1,
            consecutive_rounds_needed=2,
        )
        detector.backend = JaccardBackend()

        # Round 1: Initial positions
        round1 = {
            "agent_a": "I think we should focus on renewable energy sources",
            "agent_b": "Nuclear power is a viable clean energy option",
        }

        # Round 2: Some convergence
        round2 = {
            "agent_a": "Renewable energy and nuclear can work together",
            "agent_b": "Nuclear power combined with renewables is promising",
        }

        # Round 3: High convergence
        round3 = {
            "agent_a": "A mix of renewable and nuclear energy is the best approach",
            "agent_b": "The optimal solution combines renewable and nuclear energy",
        }

        # Check round 2 vs round 1
        result1 = detector.check_convergence(round2, round1, round_number=2)
        assert result1 is not None

        # Check round 3 vs round 2
        result2 = detector.check_convergence(round3, round2, round_number=3)
        assert result2 is not None

    def test_analyzer_full_workflow(self):
        """Test full analyzer workflow with history."""
        analyzer = AdvancedConvergenceAnalyzer(similarity_backend=JaccardBackend())

        history = [
            {"agent_a": "I disagree with this approach.", "agent_b": "I support this idea."},
            {"agent_a": "Maybe there's some merit here.", "agent_b": "I still support this."},
            {"agent_a": "I agree this could work.", "agent_b": "I agree as well."},
        ]

        current = {
            "agent_a": "I agree this is the right approach.",
            "agent_b": "I agree completely.",
        }
        previous = history[-1]

        metrics = analyzer.analyze(
            current_responses=current,
            previous_responses=previous,
            response_history=history,
            domain="policy",
        )

        assert metrics.stance_volatility is not None
        # Agent A changed stance from disagree to agree
        assert metrics.stance_volatility.stance_changes >= 1
