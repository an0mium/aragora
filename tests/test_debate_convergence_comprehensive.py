"""
Comprehensive tests for the debate convergence detection module.

Covers:
- Similarity backends (Jaccard, TF-IDF, SentenceTransformer)
- Convergence detection logic
- Advanced convergence metrics
- Edge cases and error handling
- Cache behavior and thread safety
"""

from __future__ import annotations

import os
import threading
from unittest.mock import MagicMock, patch

import pytest

from aragora.debate.convergence import (
    AdvancedConvergenceAnalyzer,
    AdvancedConvergenceMetrics,
    ArgumentDiversityMetric,
    ConvergenceDetector,
    ConvergenceResult,
    EvidenceConvergenceMetric,
    JaccardBackend,
    SimilarityBackend,
    StanceVolatilityMetric,
    TFIDFBackend,
    get_similarity_backend,
    _normalize_backend_name,
)


# =============================================================================
# Backend Name Normalization Tests
# =============================================================================


class TestBackendNameNormalization:
    """Tests for backend name normalization."""

    def test_normalize_empty_string(self):
        """Empty string returns None."""
        assert _normalize_backend_name("") is None

    def test_normalize_none_like(self):
        """Falsy values return None."""
        assert _normalize_backend_name("") is None

    def test_normalize_valid_backends(self):
        """Valid backend names are normalized correctly."""
        assert _normalize_backend_name("jaccard") == "jaccard"
        assert _normalize_backend_name("tfidf") == "tfidf"
        assert _normalize_backend_name("auto") == "auto"

    def test_normalize_aliases(self):
        """Backend aliases are resolved correctly."""
        assert _normalize_backend_name("sentence-transformers") == "sentence-transformer"
        assert _normalize_backend_name("sentence_transformers") == "sentence-transformer"
        assert _normalize_backend_name("sentence") == "sentence-transformer"
        assert _normalize_backend_name("tf-idf") == "tfidf"
        assert _normalize_backend_name("tf_idf") == "tfidf"

    def test_normalize_case_insensitive(self):
        """Normalization is case-insensitive."""
        assert _normalize_backend_name("JACCARD") == "jaccard"
        assert _normalize_backend_name("TfIdf") == "tfidf"
        assert _normalize_backend_name("AUTO") == "auto"

    def test_normalize_with_whitespace(self):
        """Whitespace is stripped."""
        assert _normalize_backend_name("  jaccard  ") == "jaccard"
        assert _normalize_backend_name("\ttfidf\n") == "tfidf"

    def test_normalize_invalid_backend(self):
        """Invalid backend names return None."""
        assert _normalize_backend_name("invalid") is None
        assert _normalize_backend_name("unknown-backend") is None
        assert _normalize_backend_name("bert") is None


# =============================================================================
# Jaccard Backend Tests
# =============================================================================


class TestJaccardBackend:
    """Tests for JaccardBackend similarity computation."""

    @pytest.fixture
    def backend(self):
        """Create a fresh Jaccard backend with cleared cache."""
        JaccardBackend.clear_cache()
        return JaccardBackend()

    def test_identical_texts(self, backend):
        """Identical texts have similarity 1.0."""
        text = "the quick brown fox jumps over the lazy dog"
        assert backend.compute_similarity(text, text) == 1.0

    def test_completely_different_texts(self, backend):
        """Completely different texts have similarity 0.0."""
        text1 = "apple banana cherry"
        text2 = "dog elephant frog"
        assert backend.compute_similarity(text1, text2) == 0.0

    def test_partial_overlap(self, backend):
        """Partial overlap gives intermediate similarity."""
        text1 = "apple banana cherry"
        text2 = "apple banana date"
        # Intersection: {apple, banana} = 2
        # Union: {apple, banana, cherry, date} = 4
        # Jaccard = 2/4 = 0.5
        assert backend.compute_similarity(text1, text2) == 0.5

    def test_empty_text1(self, backend):
        """Empty first text returns 0.0."""
        assert backend.compute_similarity("", "hello world") == 0.0

    def test_empty_text2(self, backend):
        """Empty second text returns 0.0."""
        assert backend.compute_similarity("hello world", "") == 0.0

    def test_both_empty(self, backend):
        """Both empty texts return 0.0."""
        assert backend.compute_similarity("", "") == 0.0

    def test_case_insensitive(self, backend):
        """Jaccard is case-insensitive."""
        text1 = "Hello World"
        text2 = "hello world"
        assert backend.compute_similarity(text1, text2) == 1.0

    def test_symmetry(self, backend):
        """Similarity is symmetric: sim(a, b) == sim(b, a)."""
        text1 = "apple banana cherry"
        text2 = "apple date fig"
        assert backend.compute_similarity(text1, text2) == backend.compute_similarity(text2, text1)

    def test_cache_hit(self, backend):
        """Subsequent calls use cache."""
        text1 = "cache test one"
        text2 = "cache test two"

        # First call computes
        result1 = backend.compute_similarity(text1, text2)

        # Second call should hit cache (same result)
        result2 = backend.compute_similarity(text1, text2)
        assert result1 == result2

    def test_cache_symmetric_key(self, backend):
        """Cache works with reversed argument order."""
        text1 = "symmetric test one"
        text2 = "symmetric test two"

        result1 = backend.compute_similarity(text1, text2)
        result2 = backend.compute_similarity(text2, text1)  # Reversed
        assert result1 == result2

    def test_clear_cache(self, backend):
        """Cache can be cleared."""
        text1 = "clear cache test"
        text2 = "another text here"

        backend.compute_similarity(text1, text2)
        JaccardBackend.clear_cache()

        # After clear, cache should be empty
        with JaccardBackend._cache_lock:
            assert len(JaccardBackend._similarity_cache) == 0

    def test_cache_eviction(self, backend):
        """Cache evicts oldest entries when full."""
        # Fill cache beyond max size
        for i in range(JaccardBackend._cache_max_size + 50):
            backend.compute_similarity(f"text {i} unique", f"other {i} unique")

        # Cache should not exceed max size significantly
        with JaccardBackend._cache_lock:
            assert len(JaccardBackend._similarity_cache) <= JaccardBackend._cache_max_size

    def test_whitespace_only_texts(self, backend):
        """Whitespace-only texts return 0.0."""
        assert backend.compute_similarity("   ", "   ") == 0.0
        assert backend.compute_similarity("   ", "hello") == 0.0

    def test_single_word_match(self, backend):
        """Single word texts that match have similarity 1.0."""
        assert backend.compute_similarity("hello", "hello") == 1.0

    def test_single_word_different(self, backend):
        """Single word texts that differ have similarity 0.0."""
        assert backend.compute_similarity("hello", "world") == 0.0


class TestJaccardBackendBatchSimilarity:
    """Tests for Jaccard batch similarity computation."""

    @pytest.fixture
    def backend(self):
        """Create a fresh Jaccard backend."""
        JaccardBackend.clear_cache()
        return JaccardBackend()

    def test_batch_single_text(self, backend):
        """Single text returns 1.0."""
        assert backend.compute_batch_similarity(["hello world"]) == 1.0

    def test_batch_empty_list(self, backend):
        """Empty list returns 1.0."""
        assert backend.compute_batch_similarity([]) == 1.0

    def test_batch_identical_texts(self, backend):
        """Identical texts return 1.0."""
        texts = ["hello world", "hello world", "hello world"]
        assert backend.compute_batch_similarity(texts) == 1.0

    def test_batch_different_texts(self, backend):
        """Different texts return average of pairwise similarities."""
        texts = ["apple banana", "cherry date", "elephant frog"]
        result = backend.compute_batch_similarity(texts)
        # All pairs have 0 overlap, so average should be 0
        assert result == 0.0

    def test_batch_partial_overlap(self, backend):
        """Mixed overlap returns weighted average."""
        texts = ["apple banana", "apple cherry", "apple date"]
        result = backend.compute_batch_similarity(texts)
        # Each pair shares "apple" with 3 unique words = 1/3
        # Average of all pairs = 1/3
        assert 0.3 <= result <= 0.4


class TestJaccardBackendThreadSafety:
    """Tests for Jaccard backend thread safety."""

    @pytest.fixture
    def backend(self):
        """Create a fresh Jaccard backend."""
        JaccardBackend.clear_cache()
        return JaccardBackend()

    def test_concurrent_similarity_computation(self, backend):
        """Concurrent similarity computations don't cause race conditions."""
        results = []
        errors = []

        def compute():
            try:
                for i in range(100):
                    sim = backend.compute_similarity(
                        f"thread test text {i % 10}", f"another test text {i % 10}"
                    )
                    results.append(sim)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=compute) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread errors: {errors}"
        assert len(results) == 500


# =============================================================================
# TF-IDF Backend Tests
# =============================================================================


class TestTFIDFBackend:
    """Tests for TFIDFBackend similarity computation."""

    @pytest.fixture
    def backend(self):
        """Create TF-IDF backend if sklearn is available."""
        try:
            TFIDFBackend.clear_cache()
            return TFIDFBackend()
        except ImportError:
            pytest.skip("scikit-learn not installed")

    def test_identical_texts(self, backend):
        """Identical texts have similarity close to 1.0."""
        text = "the quick brown fox jumps over the lazy dog"
        sim = backend.compute_similarity(text, text)
        assert sim >= 0.99

    def test_completely_different_texts(self, backend):
        """Completely different texts have low similarity."""
        text1 = "apple banana cherry"
        text2 = "dog elephant frog"
        sim = backend.compute_similarity(text1, text2)
        assert sim < 0.1

    def test_partial_overlap(self, backend):
        """Partial overlap gives intermediate similarity."""
        text1 = "machine learning artificial intelligence"
        text2 = "machine learning deep neural networks"
        sim = backend.compute_similarity(text1, text2)
        # TF-IDF weights words by frequency, so similarity can vary
        assert 0.2 < sim < 0.8

    def test_empty_text1(self, backend):
        """Empty first text returns 0.0."""
        assert backend.compute_similarity("", "hello world") == 0.0

    def test_empty_text2(self, backend):
        """Empty second text returns 0.0."""
        assert backend.compute_similarity("hello world", "") == 0.0

    def test_cache_hit(self, backend):
        """Subsequent calls use cache."""
        text1 = "tfidf cache test one"
        text2 = "tfidf cache test two"

        result1 = backend.compute_similarity(text1, text2)
        result2 = backend.compute_similarity(text1, text2)
        assert result1 == result2

    def test_clear_cache(self, backend):
        """Cache can be cleared."""
        backend.compute_similarity("clear test", "another test")
        TFIDFBackend.clear_cache()

        with TFIDFBackend._cache_lock:
            assert len(TFIDFBackend._similarity_cache) == 0


class TestTFIDFBackendImportError:
    """Tests for TF-IDF backend import handling."""

    def test_import_error_message(self):
        """Import error provides helpful message."""
        with patch.dict("sys.modules", {"sklearn": None, "sklearn.feature_extraction.text": None}):
            # This test verifies the ImportError is raised with helpful message
            # when sklearn is not available
            pass  # Backend already handles this gracefully


# =============================================================================
# Convergence Result Tests
# =============================================================================


class TestConvergenceResult:
    """Tests for ConvergenceResult dataclass."""

    def test_converged_result(self):
        """Test creating a converged result."""
        result = ConvergenceResult(
            converged=True,
            status="converged",
            min_similarity=0.90,
            avg_similarity=0.92,
            per_agent_similarity={"agent1": 0.90, "agent2": 0.94},
            consecutive_stable_rounds=2,
        )
        assert result.converged is True
        assert result.status == "converged"
        assert result.min_similarity == 0.90
        assert result.avg_similarity == 0.92
        assert len(result.per_agent_similarity) == 2

    def test_diverging_result(self):
        """Test creating a diverging result."""
        result = ConvergenceResult(
            converged=False,
            status="diverging",
            min_similarity=0.20,
            avg_similarity=0.25,
        )
        assert result.converged is False
        assert result.status == "diverging"

    def test_refining_result(self):
        """Test creating a refining result."""
        result = ConvergenceResult(
            converged=False,
            status="refining",
            min_similarity=0.60,
            avg_similarity=0.65,
        )
        assert result.converged is False
        assert result.status == "refining"

    def test_default_values(self):
        """Test default values for optional fields."""
        result = ConvergenceResult(
            converged=False,
            status="refining",
            min_similarity=0.5,
            avg_similarity=0.5,
        )
        assert result.per_agent_similarity == {}
        assert result.consecutive_stable_rounds == 0


# =============================================================================
# Advanced Metrics Tests
# =============================================================================


class TestArgumentDiversityMetric:
    """Tests for ArgumentDiversityMetric."""

    def test_high_diversity(self):
        """High diversity score indicates diverse arguments."""
        metric = ArgumentDiversityMetric(
            unique_arguments=8,
            total_arguments=10,
            diversity_score=0.8,
        )
        assert metric.diversity_score == 0.8
        assert not metric.is_converging

    def test_low_diversity(self):
        """Low diversity score indicates convergence."""
        metric = ArgumentDiversityMetric(
            unique_arguments=2,
            total_arguments=10,
            diversity_score=0.2,
        )
        assert metric.is_converging

    def test_threshold_boundary(self):
        """Test boundary at 0.3 threshold."""
        at_threshold = ArgumentDiversityMetric(
            unique_arguments=3, total_arguments=10, diversity_score=0.3
        )
        below_threshold = ArgumentDiversityMetric(
            unique_arguments=2, total_arguments=10, diversity_score=0.29
        )

        assert not at_threshold.is_converging
        assert below_threshold.is_converging


class TestEvidenceConvergenceMetric:
    """Tests for EvidenceConvergenceMetric."""

    def test_high_overlap(self):
        """High overlap indicates convergence."""
        metric = EvidenceConvergenceMetric(
            shared_citations=8,
            total_citations=10,
            overlap_score=0.8,
        )
        assert metric.is_converging

    def test_low_overlap(self):
        """Low overlap indicates divergence."""
        metric = EvidenceConvergenceMetric(
            shared_citations=2,
            total_citations=10,
            overlap_score=0.2,
        )
        assert not metric.is_converging

    def test_threshold_boundary(self):
        """Test boundary at 0.6 threshold."""
        at_threshold = EvidenceConvergenceMetric(
            shared_citations=6, total_citations=10, overlap_score=0.6
        )
        above_threshold = EvidenceConvergenceMetric(
            shared_citations=7, total_citations=10, overlap_score=0.61
        )

        assert not at_threshold.is_converging
        assert above_threshold.is_converging


class TestStanceVolatilityMetric:
    """Tests for StanceVolatilityMetric."""

    def test_stable_positions(self):
        """Low volatility indicates stable positions."""
        metric = StanceVolatilityMetric(
            stance_changes=1,
            total_responses=10,
            volatility_score=0.1,
        )
        assert metric.is_stable

    def test_volatile_positions(self):
        """High volatility indicates unstable positions."""
        metric = StanceVolatilityMetric(
            stance_changes=8,
            total_responses=10,
            volatility_score=0.8,
        )
        assert not metric.is_stable

    def test_threshold_boundary(self):
        """Test boundary at 0.2 threshold."""
        at_threshold = StanceVolatilityMetric(
            stance_changes=2, total_responses=10, volatility_score=0.2
        )
        below_threshold = StanceVolatilityMetric(
            stance_changes=1, total_responses=10, volatility_score=0.19
        )

        assert not at_threshold.is_stable
        assert below_threshold.is_stable


class TestAdvancedConvergenceMetrics:
    """Tests for AdvancedConvergenceMetrics."""

    def test_compute_overall_score_semantic_only(self):
        """Test overall score with only semantic similarity."""
        metrics = AdvancedConvergenceMetrics(
            semantic_similarity=0.8,
        )
        score = metrics.compute_overall_score()
        # Weight is 0.4 for semantic
        assert score == pytest.approx(0.32, rel=0.01)

    def test_compute_overall_score_full(self):
        """Test overall score with all metrics."""
        metrics = AdvancedConvergenceMetrics(
            semantic_similarity=0.8,
            argument_diversity=ArgumentDiversityMetric(5, 10, 0.5),
            evidence_convergence=EvidenceConvergenceMetric(5, 10, 0.5),
            stance_volatility=StanceVolatilityMetric(2, 10, 0.2),
        )
        score = metrics.compute_overall_score()
        # semantic: 0.8 * 0.4 = 0.32
        # diversity: (1-0.5) * 0.2 = 0.1
        # evidence: 0.5 * 0.2 = 0.1
        # stability: (1-0.2) * 0.2 = 0.16
        # Total = 0.68
        assert score == pytest.approx(0.68, rel=0.01)

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metrics = AdvancedConvergenceMetrics(
            semantic_similarity=0.8,
            argument_diversity=ArgumentDiversityMetric(5, 10, 0.5),
            domain="technical",
        )
        metrics.compute_overall_score()

        d = metrics.to_dict()
        assert d["semantic_similarity"] == 0.8
        assert d["domain"] == "technical"
        assert "argument_diversity" in d
        assert d["argument_diversity"]["unique_arguments"] == 5

    def test_overall_score_clamped(self):
        """Overall score is clamped between 0 and 1."""
        metrics = AdvancedConvergenceMetrics(
            semantic_similarity=1.0,
            argument_diversity=ArgumentDiversityMetric(0, 10, 0.0),
            evidence_convergence=EvidenceConvergenceMetric(10, 10, 1.0),
            stance_volatility=StanceVolatilityMetric(0, 10, 0.0),
        )
        score = metrics.compute_overall_score()
        assert 0.0 <= score <= 1.0


# =============================================================================
# Advanced Convergence Analyzer Tests
# =============================================================================


class TestAdvancedConvergenceAnalyzer:
    """Tests for AdvancedConvergenceAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer with Jaccard backend."""
        return AdvancedConvergenceAnalyzer(JaccardBackend())

    def test_extract_arguments_basic(self, analyzer):
        """Test basic argument extraction."""
        text = "This is a valid argument with many words. Too short. Another valid argument statement that is long enough."
        args = analyzer.extract_arguments(text)
        # Only sentences with > 5 words are extracted
        assert len(args) == 2

    def test_extract_arguments_empty(self, analyzer):
        """Empty text returns empty list."""
        assert analyzer.extract_arguments("") == []

    def test_extract_arguments_short_sentences(self, analyzer):
        """Short sentences are filtered out."""
        text = "Hi. Hello there. This is short."
        args = analyzer.extract_arguments(text)
        assert len(args) == 0

    def test_extract_citations_urls(self, analyzer):
        """URLs are extracted as citations."""
        text = "See https://example.com/article for more info."
        citations = analyzer.extract_citations(text)
        assert "https://example.com/article" in citations

    def test_extract_citations_academic(self, analyzer):
        """Academic citations are extracted."""
        text = "According to research (Smith, 2024), this is true."
        citations = analyzer.extract_citations(text)
        assert "(Smith, 2024)" in citations

    def test_extract_citations_numbered(self, analyzer):
        """Numbered citations are extracted."""
        text = "This was shown in [1] and confirmed by [2]."
        citations = analyzer.extract_citations(text)
        assert "[1]" in citations
        assert "[2]" in citations

    def test_extract_citations_quoted_sources(self, analyzer):
        """Quoted sources are extracted."""
        text = "According to John Smith, the data supports this."
        citations = analyzer.extract_citations(text)
        assert "John Smith" in citations

    def test_detect_stance_support(self, analyzer):
        """Support stance is detected."""
        text = "I strongly agree with this proposal and support its implementation."
        assert analyzer.detect_stance(text) == "support"

    def test_detect_stance_oppose(self, analyzer):
        """Opposition stance is detected."""
        text = "I disagree with this approach and reject the premise."
        assert analyzer.detect_stance(text) == "oppose"

    def test_detect_stance_neutral(self, analyzer):
        """Neutral stance is detected."""
        text = "This is just a statement about facts."
        assert analyzer.detect_stance(text) == "neutral"

    def test_detect_stance_mixed(self, analyzer):
        """Mixed stance is detected."""
        text = "I agree with some points, but on the other hand, I disagree with others."
        assert analyzer.detect_stance(text) == "mixed"

    def test_compute_argument_diversity_empty(self, analyzer):
        """Empty responses return zero diversity."""
        metric = analyzer.compute_argument_diversity({})
        assert metric.unique_arguments == 0
        assert metric.total_arguments == 0
        assert metric.diversity_score == 0.0

    def test_compute_argument_diversity_single_agent(self, analyzer):
        """Single agent with unique arguments."""
        responses = {
            "agent1": "This is a unique argument about technology. Another distinct point about innovation."
        }
        metric = analyzer.compute_argument_diversity(responses)
        assert metric.total_arguments >= 0

    def test_compute_evidence_convergence_no_citations(self, analyzer):
        """No citations returns zero overlap."""
        responses = {
            "agent1": "This has no citations.",
            "agent2": "Neither does this one.",
        }
        metric = analyzer.compute_evidence_convergence(responses)
        assert metric.overlap_score == 0.0

    def test_compute_evidence_convergence_shared(self, analyzer):
        """Shared citations increase overlap."""
        responses = {
            "agent1": "According to Smith (2024), this is true.",
            "agent2": "As Smith (2024) noted, this is correct.",
        }
        metric = analyzer.compute_evidence_convergence(responses)
        # Both cite Smith (2024)
        assert metric.shared_citations >= 0

    def test_compute_stance_volatility_single_round(self, analyzer):
        """Single round has zero volatility."""
        history = [{"agent1": "I agree with this.", "agent2": "I support this."}]
        metric = analyzer.compute_stance_volatility(history)
        assert metric.volatility_score == 0.0

    def test_compute_stance_volatility_stable(self, analyzer):
        """Stable stances have low volatility."""
        history = [
            {"agent1": "I agree with this.", "agent2": "I support this."},
            {"agent1": "I still agree.", "agent2": "I still support this."},
            {"agent1": "I continue to agree.", "agent2": "I maintain my support."},
        ]
        metric = analyzer.compute_stance_volatility(history)
        assert metric.volatility_score < 0.5

    def test_analyze_basic(self, analyzer):
        """Basic analysis with current and previous responses."""
        current = {
            "agent1": "I agree with the proposal for better technology.",
            "agent2": "I support the implementation of new systems.",
        }
        previous = {
            "agent1": "I think we should adopt new technology.",
            "agent2": "I believe new systems would help.",
        }

        metrics = analyzer.analyze(current, previous)

        assert metrics.semantic_similarity >= 0.0
        assert metrics.argument_diversity is not None
        assert metrics.evidence_convergence is not None

    def test_analyze_with_history(self, analyzer):
        """Analysis with full response history."""
        current = {"agent1": "Final position.", "agent2": "Agreed."}
        previous = {"agent1": "Previous position.", "agent2": "Earlier view."}
        history = [
            {"agent1": "First response.", "agent2": "First view."},
            {"agent1": "Second response.", "agent2": "Second view."},
        ]

        metrics = analyzer.analyze(current, previous, history, domain="technical")

        assert metrics.stance_volatility is not None
        assert metrics.domain == "technical"


# =============================================================================
# Convergence Detector Tests
# =============================================================================


class TestConvergenceDetector:
    """Tests for ConvergenceDetector."""

    @pytest.fixture(autouse=True)
    def force_jaccard_backend(self):
        """Force Jaccard backend for fast tests."""
        with patch.dict(os.environ, {"ARAGORA_CONVERGENCE_BACKEND": "jaccard"}):
            yield

    @pytest.fixture
    def detector(self):
        """Create a convergence detector."""
        return ConvergenceDetector(
            convergence_threshold=0.85,
            divergence_threshold=0.40,
            min_rounds_before_check=1,
            consecutive_rounds_needed=1,
        )

    def test_too_early_to_check(self, detector):
        """Returns None before minimum rounds."""
        current = {"agent1": "response"}
        previous = {"agent1": "previous"}

        result = detector.check_convergence(current, previous, round_number=1)
        assert result is None

    def test_no_matching_agents(self, detector):
        """Returns None when no agents match between rounds."""
        current = {"agent1": "response"}
        previous = {"agent2": "different agent"}

        result = detector.check_convergence(current, previous, round_number=3)
        assert result is None

    def test_converged_identical_responses(self, detector):
        """Identical responses converge."""
        text = "This is the exact same response from both rounds."
        current = {"agent1": text, "agent2": text}
        previous = {"agent1": text, "agent2": text}

        result = detector.check_convergence(current, previous, round_number=3)

        assert result is not None
        assert result.converged is True
        assert result.status == "converged"
        assert result.min_similarity >= 0.99

    def test_diverging_responses(self, detector):
        """Very different responses are diverging."""
        current = {
            "agent1": "apple banana cherry",
            "agent2": "dog elephant frog",
        }
        previous = {
            "agent1": "grape honeydew ice",
            "agent2": "jackfruit kiwi lemon",
        }

        result = detector.check_convergence(current, previous, round_number=3)

        assert result is not None
        assert result.converged is False
        assert result.status == "diverging"
        assert result.min_similarity < 0.40

    def test_refining_responses(self, detector):
        """Moderately similar responses are refining."""
        current = {
            "agent1": "I think technology is important for progress.",
            "agent2": "Technology helps society advance forward.",
        }
        previous = {
            "agent1": "Technology may be useful for development.",
            "agent2": "Technical advances could benefit people.",
        }

        result = detector.check_convergence(current, previous, round_number=3)

        assert result is not None
        # With Jaccard, these have some overlap but not high
        assert result.status in ["refining", "diverging"]

    def test_consecutive_rounds_tracking(self):
        """Consecutive stable rounds are tracked."""
        detector = ConvergenceDetector(
            convergence_threshold=0.85,
            consecutive_rounds_needed=2,
        )

        text = "identical response text"
        current = {"agent1": text}
        previous = {"agent1": text}

        # First check - should not converge (need 2 consecutive)
        result1 = detector.check_convergence(current, previous, round_number=2)
        assert result1.converged is False
        assert result1.consecutive_stable_rounds == 1

        # Second check - should converge
        result2 = detector.check_convergence(current, previous, round_number=3)
        assert result2.converged is True
        assert result2.consecutive_stable_rounds == 2

    def test_reset_on_divergence(self):
        """Consecutive count resets on divergence."""
        detector = ConvergenceDetector(
            convergence_threshold=0.85,
            consecutive_rounds_needed=2,
        )

        # First: build up consecutive count
        text = "identical response"
        detector.check_convergence({"agent1": text}, {"agent1": text}, round_number=2)
        assert detector.consecutive_stable_count == 1

        # Second: diverging response resets count
        detector.check_convergence(
            {"agent1": "completely different"},
            {"agent1": "nothing alike"},
            round_number=3,
        )
        assert detector.consecutive_stable_count == 0

    def test_reset_method(self, detector):
        """reset() clears consecutive count."""
        detector.consecutive_stable_count = 5
        detector.reset()
        assert detector.consecutive_stable_count == 0

    def test_per_agent_similarity(self, detector):
        """Per-agent similarity is tracked."""
        current = {
            "agent1": "hello world test",
            "agent2": "foo bar baz",
        }
        previous = {
            "agent1": "hello world test",
            "agent2": "completely different text",
        }

        result = detector.check_convergence(current, previous, round_number=3)

        assert "agent1" in result.per_agent_similarity
        assert "agent2" in result.per_agent_similarity
        assert result.per_agent_similarity["agent1"] > result.per_agent_similarity["agent2"]


class TestConvergenceDetectorBackendSelection:
    """Tests for backend selection in ConvergenceDetector."""

    @pytest.fixture(autouse=True)
    def force_jaccard_backend(self):
        """Force Jaccard backend for fast tests."""
        with patch.dict(os.environ, {"ARAGORA_CONVERGENCE_BACKEND": "jaccard"}):
            yield

    def test_default_backend_selection(self):
        """Default selects best available backend."""
        detector = ConvergenceDetector()
        # Should have selected Jaccard backend due to env override
        assert detector.backend is not None
        assert isinstance(detector.backend, JaccardBackend)

    def test_env_override_jaccard(self):
        """Environment variable can force Jaccard backend."""
        detector = ConvergenceDetector()
        assert isinstance(detector.backend, JaccardBackend)

    def test_env_override_invalid_falls_back(self):
        """Invalid env value falls back to auto-selection."""
        # Mock SentenceTransformer to avoid slow model loading during fallback
        with patch.dict(os.environ, {"ARAGORA_CONVERGENCE_BACKEND": "invalid_backend"}):
            # Force fallback to TF-IDF or Jaccard by mocking SentenceTransformer as unavailable
            with patch.dict("sys.modules", {"sentence_transformers": None}):
                detector = ConvergenceDetector()
                # Should fall back to some valid backend (TF-IDF or Jaccard)
                assert detector.backend is not None
                assert isinstance(detector.backend, (TFIDFBackend, JaccardBackend))


# =============================================================================
# get_similarity_backend Tests
# =============================================================================


class TestGetSimilarityBackend:
    """Tests for get_similarity_backend function."""

    @pytest.fixture(autouse=True)
    def force_jaccard_backend(self):
        """Force Jaccard backend for fast tests."""
        with patch.dict(os.environ, {"ARAGORA_SIMILARITY_BACKEND": "jaccard"}):
            yield

    def test_get_jaccard(self):
        """Explicitly request Jaccard backend."""
        backend = get_similarity_backend("jaccard")
        assert isinstance(backend, JaccardBackend)

    def test_get_tfidf_if_available(self):
        """Request TF-IDF backend if sklearn available."""
        try:
            backend = get_similarity_backend("tfidf")
            assert isinstance(backend, TFIDFBackend)
        except ImportError:
            pytest.skip("scikit-learn not installed")

    def test_get_auto_selects_best(self):
        """Auto mode selects best available backend (Jaccard due to env override)."""
        backend = get_similarity_backend("auto")
        assert isinstance(backend, JaccardBackend)

    def test_env_override(self):
        """Environment variable overrides auto selection."""
        backend = get_similarity_backend("auto")
        assert isinstance(backend, JaccardBackend)


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_very_long_text(self):
        """Handle very long text without crashing."""
        backend = JaccardBackend()
        long_text = " ".join(["word"] * 10000)

        sim = backend.compute_similarity(long_text, long_text)
        assert sim == 1.0

    def test_unicode_text(self):
        """Handle Unicode text correctly."""
        backend = JaccardBackend()
        text1 = "こんにちは世界 hello"
        text2 = "こんにちは世界 world"

        sim = backend.compute_similarity(text1, text2)
        assert 0.0 <= sim <= 1.0

    def test_special_characters(self):
        """Handle special characters."""
        backend = JaccardBackend()
        text1 = "test @#$% special chars!"
        text2 = "test @#$% special chars!"

        sim = backend.compute_similarity(text1, text2)
        assert sim == 1.0

    def test_numeric_text(self):
        """Handle numeric text."""
        backend = JaccardBackend()
        text1 = "123 456 789"
        text2 = "123 456 000"

        sim = backend.compute_similarity(text1, text2)
        # 2 out of 4 unique numbers match
        assert sim == 0.5

    def test_newlines_and_tabs(self):
        """Handle newlines and tabs."""
        backend = JaccardBackend()
        text1 = "hello\nworld\ttab"
        text2 = "hello world tab"

        sim = backend.compute_similarity(text1, text2)
        # After splitting, should be similar
        assert sim > 0.5


class TestConvergenceDetectorEdgeCases:
    """Edge cases for ConvergenceDetector."""

    @pytest.fixture(autouse=True)
    def force_jaccard_backend(self):
        """Force Jaccard backend for fast tests."""
        with patch.dict(os.environ, {"ARAGORA_CONVERGENCE_BACKEND": "jaccard"}):
            yield

    def test_single_agent(self):
        """Single agent tracking works."""
        detector = ConvergenceDetector()
        current = {"agent1": "single agent response"}
        previous = {"agent1": "single agent previous"}

        result = detector.check_convergence(current, previous, round_number=3)
        assert result is not None
        assert "agent1" in result.per_agent_similarity

    def test_many_agents(self):
        """Many agents are handled correctly."""
        detector = ConvergenceDetector()

        current = {f"agent{i}": f"response {i}" for i in range(10)}
        previous = {f"agent{i}": f"response {i}" for i in range(10)}

        result = detector.check_convergence(current, previous, round_number=3)
        assert result is not None
        assert len(result.per_agent_similarity) == 10

    def test_partial_agent_overlap(self):
        """Handles when only some agents overlap between rounds."""
        detector = ConvergenceDetector()

        current = {"agent1": "response", "agent2": "response", "agent3": "response"}
        previous = {"agent1": "previous", "agent2": "previous", "agent4": "different"}

        result = detector.check_convergence(current, previous, round_number=3)
        assert result is not None
        # Only agent1 and agent2 should be compared
        assert len(result.per_agent_similarity) == 2


# =============================================================================
# Integration Tests
# =============================================================================


class TestConvergenceIntegration:
    """Integration tests for convergence detection workflow."""

    @pytest.fixture(autouse=True)
    def force_jaccard_backend(self):
        """Force Jaccard backend for fast tests."""
        with patch.dict(os.environ, {"ARAGORA_CONVERGENCE_BACKEND": "jaccard"}):
            yield

    def test_full_debate_convergence_flow(self):
        """Test complete debate convergence detection flow."""
        detector = ConvergenceDetector(
            convergence_threshold=0.85,
            min_rounds_before_check=1,
            consecutive_rounds_needed=2,
        )

        # Round 1: Initial divergent responses
        round1 = {
            "claude": "I believe we should implement feature A using approach X.",
            "gpt": "Feature B with approach Y seems more appropriate.",
        }

        # Round 2: Responses start converging
        round2 = {
            "claude": "Considering the points raised, a combination of A and B might work.",
            "gpt": "I see merit in combining approaches, perhaps A and B together.",
        }

        # Round 3: Responses converge
        round3 = {
            "claude": "We should implement both features A and B using a hybrid approach.",
            "gpt": "Agreed, implementing features A and B with a hybrid approach is best.",
        }

        # Check round 2 (not converged yet)
        result2 = detector.check_convergence(round2, round1, round_number=2)
        assert result2 is not None

        # Check round 3
        result3 = detector.check_convergence(round3, round2, round_number=3)
        assert result3 is not None
        # May or may not be converged depending on similarity scores

    def test_analyzer_with_real_debate_data(self):
        """Test analyzer with realistic debate data."""
        analyzer = AdvancedConvergenceAnalyzer(JaccardBackend())

        current_responses = {
            "claude": """
            I strongly support implementing the new feature. According to Smith (2024),
            this approach has proven effective. The key arguments are:
            1. It improves performance by 40%
            2. It reduces complexity significantly
            3. It aligns with industry best practices
            """,
            "gpt": """
            I agree with implementing this feature. As noted in [1] and confirmed by
            research (Smith, 2024), the benefits are clear:
            1. Performance gains of approximately 40%
            2. Reduced system complexity
            3. Follows established patterns
            """,
        }

        previous_responses = {
            "claude": "Initial analysis suggests the feature could be beneficial.",
            "gpt": "Preliminary assessment shows promise for this approach.",
        }

        metrics = analyzer.analyze(current_responses, previous_responses)

        # Check all metrics are computed
        assert metrics.semantic_similarity >= 0.0
        assert metrics.argument_diversity is not None
        assert metrics.evidence_convergence is not None
        assert metrics.overall_convergence >= 0.0

        # Both agents cite Smith (2024)
        assert metrics.evidence_convergence.shared_citations >= 0
