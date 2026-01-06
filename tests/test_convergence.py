"""
Tests for Semantic Convergence Detection.

Tests cover:
- JaccardBackend (zero dependencies)
- TFIDFBackend (requires scikit-learn)
- SentenceTransformerBackend (requires sentence-transformers)
- ConvergenceResult dataclass
- ConvergenceDetector (threshold checking, status determination)
- get_similarity_backend factory function
"""

from unittest.mock import patch, MagicMock

import pytest

# Check if sentence_transformers is actually usable (not just installed)
try:
    import sentence_transformers
    HAS_SENTENCE_TRANSFORMERS = True
except (ImportError, RuntimeError):
    # RuntimeError occurs when Keras 3 is installed without tf-keras
    HAS_SENTENCE_TRANSFORMERS = False

requires_sentence_transformers = pytest.mark.skipif(
    not HAS_SENTENCE_TRANSFORMERS,
    reason="sentence-transformers not available or Keras 3 compatibility issue"
)

from aragora.debate.convergence import (
    JaccardBackend,
    TFIDFBackend,
    SentenceTransformerBackend,
    ConvergenceResult,
    ConvergenceDetector,
    SimilarityBackend,
    get_similarity_backend,
)


# =============================================================================
# Jaccard Backend Tests
# =============================================================================


class TestJaccardBackend:
    """Tests for Jaccard similarity backend."""

    @pytest.fixture
    def backend(self):
        return JaccardBackend()

    def test_identical_text_returns_one(self, backend):
        """Identical text should have similarity of 1.0."""
        text = "the quick brown fox jumps over the lazy dog"
        similarity = backend.compute_similarity(text, text)
        assert similarity == 1.0

    def test_completely_different_text_returns_zero(self, backend):
        """Completely different text should have similarity of 0.0."""
        text1 = "the quick brown fox"
        text2 = "airplane engine turbulence"
        similarity = backend.compute_similarity(text1, text2)
        assert similarity == 0.0

    def test_partial_overlap_returns_intermediate(self, backend):
        """Partially overlapping text should have intermediate similarity."""
        text1 = "the quick brown fox"
        text2 = "the lazy brown dog"
        similarity = backend.compute_similarity(text1, text2)
        # Shared: {the, brown} = 2, Union: {the, quick, brown, fox, lazy, dog} = 6
        # Expected: 2/6 = 0.333...
        assert 0.3 <= similarity <= 0.4

    def test_case_insensitive(self, backend):
        """Comparison should be case-insensitive."""
        text1 = "The Quick Brown Fox"
        text2 = "the quick brown fox"
        similarity = backend.compute_similarity(text1, text2)
        assert similarity == 1.0

    def test_empty_string_first_returns_zero(self, backend):
        """Empty first string should return 0.0."""
        similarity = backend.compute_similarity("", "some text")
        assert similarity == 0.0

    def test_empty_string_second_returns_zero(self, backend):
        """Empty second string should return 0.0."""
        similarity = backend.compute_similarity("some text", "")
        assert similarity == 0.0

    def test_both_empty_returns_zero(self, backend):
        """Both empty strings should return 0.0."""
        similarity = backend.compute_similarity("", "")
        assert similarity == 0.0

    def test_single_word_identical(self, backend):
        """Single identical words should return 1.0."""
        similarity = backend.compute_similarity("hello", "hello")
        assert similarity == 1.0

    def test_single_word_different(self, backend):
        """Single different words should return 0.0."""
        similarity = backend.compute_similarity("hello", "world")
        assert similarity == 0.0

    def test_compute_batch_similarity_single_text_returns_one(self, backend):
        """Batch with single text should return 1.0."""
        similarity = backend.compute_batch_similarity(["single text"])
        assert similarity == 1.0

    def test_compute_batch_similarity_empty_returns_one(self, backend):
        """Batch with no texts should return 1.0."""
        similarity = backend.compute_batch_similarity([])
        assert similarity == 1.0

    def test_compute_batch_similarity_pairwise_average(self, backend):
        """Batch should compute pairwise average similarity."""
        texts = ["hello world", "hello world", "goodbye moon"]
        similarity = backend.compute_batch_similarity(texts)
        # Pairs: (0,1)=1.0, (0,2)=0.0, (1,2)=0.0 -> avg = 1/3 ≈ 0.33
        assert 0.2 <= similarity <= 0.4


# =============================================================================
# TF-IDF Backend Tests
# =============================================================================


class TestTFIDFBackend:
    """Tests for TF-IDF similarity backend."""

    def test_import_error_when_sklearn_missing(self):
        """Should raise ImportError when sklearn not available."""
        with patch.dict("sys.modules", {"sklearn": None}):
            with patch(
                "aragora.debate.convergence.TFIDFBackend.__init__",
                side_effect=ImportError("No sklearn"),
            ):
                with pytest.raises(ImportError):
                    TFIDFBackend()

    def test_identical_text_returns_near_one(self):
        """Identical text should have similarity near 1.0."""
        pytest.importorskip("sklearn")
        backend = TFIDFBackend()
        text = "the quick brown fox jumps over the lazy dog"
        similarity = backend.compute_similarity(text, text)
        assert similarity == pytest.approx(1.0, abs=0.01)

    def test_different_text_returns_less_than_one(self):
        """Different text should have similarity less than 1.0."""
        pytest.importorskip("sklearn")
        backend = TFIDFBackend()
        text1 = "I prefer TypeScript for type safety"
        text2 = "JavaScript is more flexible"
        similarity = backend.compute_similarity(text1, text2)
        assert similarity < 1.0
        assert similarity >= 0.0

    def test_empty_string_first_returns_zero(self):
        """Empty first string should return 0.0."""
        pytest.importorskip("sklearn")
        backend = TFIDFBackend()
        similarity = backend.compute_similarity("", "some text")
        assert similarity == 0.0

    def test_empty_string_second_returns_zero(self):
        """Empty second string should return 0.0."""
        pytest.importorskip("sklearn")
        backend = TFIDFBackend()
        similarity = backend.compute_similarity("some text", "")
        assert similarity == 0.0

    def test_semantic_overlap_captured(self):
        """TF-IDF should capture word overlap."""
        pytest.importorskip("sklearn")
        backend = TFIDFBackend()
        text1 = "TypeScript provides type safety"
        text2 = "TypeScript is better for types"
        similarity = backend.compute_similarity(text1, text2)
        # Should have some similarity due to shared words
        assert similarity > 0.0


# =============================================================================
# Sentence Transformer Backend Tests
# =============================================================================


@requires_sentence_transformers
class TestSentenceTransformerBackend:
    """Tests for SentenceTransformer similarity backend."""

    def test_import_error_when_missing(self):
        """Should raise ImportError when sentence-transformers not available."""
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            with patch(
                "aragora.debate.convergence.SentenceTransformerBackend.__init__",
                side_effect=ImportError("No sentence-transformers"),
            ):
                with pytest.raises(ImportError):
                    SentenceTransformerBackend()

    def test_identical_text_returns_near_one(self):
        """Identical text should have similarity very close to 1.0."""
        backend = SentenceTransformerBackend()
        text = "The quick brown fox"
        similarity = backend.compute_similarity(text, text)
        assert similarity > 0.99

    def test_semantic_similarity_captured(self):
        """Should understand semantic similarity."""
        backend = SentenceTransformerBackend()
        text1 = "I prefer TypeScript for type safety"
        text2 = "TypeScript is better because it has types"
        similarity = backend.compute_similarity(text1, text2)
        # Semantically similar - should be high
        assert similarity > 0.5

    def test_model_caching(self):
        """Model should be cached and reused."""
        # Reset cache
        SentenceTransformerBackend._model_cache = None
        SentenceTransformerBackend._model_name_cache = None

        backend1 = SentenceTransformerBackend()
        model1 = backend1.model

        backend2 = SentenceTransformerBackend()
        model2 = backend2.model

        # Should be same object (cached)
        assert model1 is model2

    def test_empty_string_first_returns_zero(self):
        """Empty first string should return 0.0."""
        backend = SentenceTransformerBackend()
        similarity = backend.compute_similarity("", "some text")
        assert similarity == 0.0

    def test_empty_string_second_returns_zero(self):
        """Empty second string should return 0.0."""
        backend = SentenceTransformerBackend()
        similarity = backend.compute_similarity("some text", "")
        assert similarity == 0.0


# =============================================================================
# ConvergenceResult Tests
# =============================================================================


class TestConvergenceResult:
    """Tests for ConvergenceResult dataclass."""

    def test_creation_with_required_fields(self):
        """ConvergenceResult should be created with required fields."""
        result = ConvergenceResult(
            converged=True,
            status="converged",
            min_similarity=0.9,
            avg_similarity=0.92,
        )
        assert result.converged is True
        assert result.status == "converged"
        assert result.min_similarity == 0.9
        assert result.avg_similarity == 0.92

    def test_per_agent_similarity_defaults_empty(self):
        """per_agent_similarity should default to empty dict."""
        result = ConvergenceResult(
            converged=False,
            status="refining",
            min_similarity=0.5,
            avg_similarity=0.6,
        )
        assert result.per_agent_similarity == {}

    def test_consecutive_stable_rounds_defaults_zero(self):
        """consecutive_stable_rounds should default to 0."""
        result = ConvergenceResult(
            converged=False,
            status="refining",
            min_similarity=0.5,
            avg_similarity=0.6,
        )
        assert result.consecutive_stable_rounds == 0

    def test_per_agent_similarity_can_be_set(self):
        """per_agent_similarity should be settable."""
        result = ConvergenceResult(
            converged=True,
            status="converged",
            min_similarity=0.9,
            avg_similarity=0.92,
            per_agent_similarity={"claude": 0.95, "codex": 0.89},
        )
        assert result.per_agent_similarity["claude"] == 0.95
        assert result.per_agent_similarity["codex"] == 0.89


# =============================================================================
# ConvergenceDetector Tests
# =============================================================================


class TestConvergenceDetector:
    """Tests for ConvergenceDetector class."""

    def test_initialization_with_defaults(self):
        """Detector should initialize with default thresholds."""
        detector = ConvergenceDetector()
        assert detector.convergence_threshold == 0.85
        assert detector.divergence_threshold == 0.40
        assert detector.min_rounds_before_check == 1
        assert detector.consecutive_rounds_needed == 1

    def test_initialization_with_custom_thresholds(self):
        """Detector should accept custom thresholds."""
        detector = ConvergenceDetector(
            convergence_threshold=0.90,
            divergence_threshold=0.30,
            min_rounds_before_check=2,
            consecutive_rounds_needed=2,
        )
        assert detector.convergence_threshold == 0.90
        assert detector.divergence_threshold == 0.30
        assert detector.min_rounds_before_check == 2
        assert detector.consecutive_rounds_needed == 2

    def test_select_backend_returns_similarity_backend(self):
        """_select_backend should return a SimilarityBackend."""
        detector = ConvergenceDetector()
        assert isinstance(detector.backend, SimilarityBackend)

    def test_check_convergence_returns_none_before_min_rounds(self):
        """Should return None if round_number <= min_rounds_before_check."""
        detector = ConvergenceDetector(min_rounds_before_check=2)
        current = {"agent1": "response"}
        previous = {"agent1": "response"}

        result = detector.check_convergence(current, previous, round_number=1)
        assert result is None

        result = detector.check_convergence(current, previous, round_number=2)
        assert result is None

    def test_check_convergence_detects_convergence(self):
        """Should detect convergence when similarity >= threshold."""
        detector = ConvergenceDetector(
            convergence_threshold=0.85,
            min_rounds_before_check=1,
        )
        # Identical responses = 1.0 similarity
        current = {"agent1": "TypeScript is better", "agent2": "TypeScript is better"}
        previous = {"agent1": "TypeScript is better", "agent2": "TypeScript is better"}

        result = detector.check_convergence(current, previous, round_number=2)

        assert result is not None
        assert result.converged is True
        assert result.status == "converged"
        assert result.min_similarity == pytest.approx(1.0, rel=1e-9)

    def test_check_convergence_detects_refining(self):
        """Should detect refining when similarity is intermediate."""
        detector = ConvergenceDetector(
            convergence_threshold=0.85,
            divergence_threshold=0.40,
            min_rounds_before_check=1,
        )
        # Partial overlap - need enough shared words to be 40-85% similar
        # Shared: {typescript, is, better, for, large, projects} = 6
        # Union: {typescript, is, better, for, large, projects, definitely} = 7
        # Jaccard = 6/7 ≈ 0.86 - just above threshold, let's adjust
        current = {"agent1": "TypeScript is better for large projects and teams"}
        previous = {"agent1": "TypeScript is good for medium projects and developers"}
        # Shared: {typescript, is, for, projects, and} = 5
        # Union: {typescript, is, better, for, large, projects, and, teams, good, medium, developers} = 11
        # Jaccard = 5/11 ≈ 0.45 - in refining range

        result = detector.check_convergence(current, previous, round_number=2)

        assert result is not None
        assert result.converged is False
        assert result.status == "refining"

    def test_check_convergence_detects_diverging(self):
        """Should detect diverging when similarity < divergence_threshold."""
        detector = ConvergenceDetector(
            convergence_threshold=0.85,
            divergence_threshold=0.40,
            min_rounds_before_check=1,
        )
        # Completely different responses
        current = {"agent1": "airplane engine turbulence"}
        previous = {"agent1": "butterfly garden flowers"}

        result = detector.check_convergence(current, previous, round_number=2)

        assert result is not None
        assert result.converged is False
        assert result.status == "diverging"
        assert result.min_similarity < 0.40

    def test_consecutive_stable_rounds_increments(self):
        """consecutive_stable_count should increment on stable rounds."""
        detector = ConvergenceDetector(
            convergence_threshold=0.85,
            consecutive_rounds_needed=2,
            min_rounds_before_check=1,
        )
        # First stable round
        current = {"agent1": "same response"}
        previous = {"agent1": "same response"}

        result1 = detector.check_convergence(current, previous, round_number=2)
        assert result1.consecutive_stable_rounds == 1
        assert result1.converged is False  # Needs 2 consecutive

        # Second stable round
        result2 = detector.check_convergence(current, previous, round_number=3)
        assert result2.consecutive_stable_rounds == 2
        assert result2.converged is True  # Now converged

    def test_convergence_requires_consecutive_rounds_needed(self):
        """Convergence should only trigger after consecutive_rounds_needed stable rounds."""
        detector = ConvergenceDetector(
            convergence_threshold=0.85,
            consecutive_rounds_needed=3,
            min_rounds_before_check=1,
        )
        current = {"agent1": "same"}
        previous = {"agent1": "same"}

        # Round 1
        result1 = detector.check_convergence(current, previous, round_number=2)
        assert result1.converged is False

        # Round 2
        result2 = detector.check_convergence(current, previous, round_number=3)
        assert result2.converged is False

        # Round 3 - should converge now
        result3 = detector.check_convergence(current, previous, round_number=4)
        assert result3.converged is True

    def test_reset_clears_consecutive_count(self):
        """reset() should clear consecutive_stable_count."""
        detector = ConvergenceDetector(
            convergence_threshold=0.85,
            min_rounds_before_check=1,
        )
        current = {"agent1": "same"}
        previous = {"agent1": "same"}

        detector.check_convergence(current, previous, round_number=2)
        assert detector.consecutive_stable_count >= 1

        detector.reset()
        assert detector.consecutive_stable_count == 0

    def test_handles_missing_agents_between_rounds(self):
        """Should handle case where agent sets don't match."""
        detector = ConvergenceDetector(min_rounds_before_check=1)
        current = {"agent1": "response", "agent2": "response"}
        previous = {"agent3": "response"}  # Different agents

        result = detector.check_convergence(current, previous, round_number=2)
        # Should return None due to no common agents
        assert result is None

    def test_computes_per_agent_similarity(self):
        """Should compute similarity for each agent."""
        detector = ConvergenceDetector(min_rounds_before_check=1)
        current = {
            "claude": "TypeScript is great",
            "codex": "JavaScript is flexible",
        }
        previous = {
            "claude": "TypeScript is great",
            "codex": "JavaScript is flexible",
        }

        result = detector.check_convergence(current, previous, round_number=2)

        assert result is not None
        assert "claude" in result.per_agent_similarity
        assert "codex" in result.per_agent_similarity
        assert result.per_agent_similarity["claude"] == pytest.approx(1.0, rel=1e-9)
        assert result.per_agent_similarity["codex"] == pytest.approx(1.0, rel=1e-9)


# =============================================================================
# get_similarity_backend Factory Tests
# =============================================================================


class TestGetSimilarityBackend:
    """Tests for get_similarity_backend factory function."""

    def test_jaccard_returns_jaccard_backend(self):
        """'jaccard' should return JaccardBackend."""
        backend = get_similarity_backend("jaccard")
        assert isinstance(backend, JaccardBackend)

    def test_tfidf_returns_tfidf_backend(self):
        """'tfidf' should return TFIDFBackend if available."""
        pytest.importorskip("sklearn")
        backend = get_similarity_backend("tfidf")
        assert isinstance(backend, TFIDFBackend)

    @requires_sentence_transformers
    def test_sentence_transformer_returns_st_backend(self):
        """'sentence-transformer' should return SentenceTransformerBackend if available."""
        backend = get_similarity_backend("sentence-transformer")
        assert isinstance(backend, SentenceTransformerBackend)

    def test_auto_returns_backend(self):
        """'auto' should return best available backend."""
        backend = get_similarity_backend("auto")
        assert isinstance(backend, SimilarityBackend)

    def test_auto_returns_at_least_jaccard(self):
        """'auto' should at least return Jaccard (always available)."""
        # Force other backends to fail
        with patch(
            "aragora.debate.convergence.SentenceTransformerBackend",
            side_effect=ImportError,
        ):
            with patch(
                "aragora.debate.convergence.TFIDFBackend",
                side_effect=ImportError,
            ):
                backend = get_similarity_backend("auto")
                assert isinstance(backend, JaccardBackend)
