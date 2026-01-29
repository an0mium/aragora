"""
Tests for aragora.ml.quality_scorer module.

Tests cover:
- QualityScore dataclass and properties
- QualityScorerConfig defaults and custom values
- QualityScorer feature extraction
- Component scoring (coherence, completeness, clarity, relevance)
- Overall scoring and batch scoring
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aragora.ml.quality_scorer import (
    QualityScore,
    QualityScorer,
    QualityScorerConfig,
)


# =============================================================================
# TestQualityScore - Dataclass Tests
# =============================================================================


class TestQualityScoreInit:
    """Tests for QualityScore initialization."""

    def test_creates_with_required_fields(self):
        """Should create with all required fields."""
        score = QualityScore(
            overall=0.8,
            coherence=0.9,
            completeness=0.7,
            relevance=0.8,
            clarity=0.85,
            confidence=0.95,
        )
        assert score.overall == 0.8
        assert score.coherence == 0.9
        assert score.completeness == 0.7
        assert score.relevance == 0.8
        assert score.clarity == 0.85
        assert score.confidence == 0.95

    def test_features_defaults_to_empty_dict(self):
        """Features should default to empty dict."""
        score = QualityScore(
            overall=0.5,
            coherence=0.5,
            completeness=0.5,
            relevance=0.5,
            clarity=0.5,
            confidence=0.5,
        )
        assert score.features == {}

    def test_features_can_be_provided(self):
        """Features can be provided explicitly."""
        features = {"word_count": 100.0, "sentence_count": 5.0}
        score = QualityScore(
            overall=0.5,
            coherence=0.5,
            completeness=0.5,
            relevance=0.5,
            clarity=0.5,
            confidence=0.5,
            features=features,
        )
        assert score.features == features


class TestQualityScoreToDict:
    """Tests for QualityScore.to_dict()."""

    def test_returns_dict_with_rounded_values(self):
        """to_dict should return dict with 3 decimal precision."""
        score = QualityScore(
            overall=0.12345,
            coherence=0.23456,
            completeness=0.34567,
            relevance=0.45678,
            clarity=0.56789,
            confidence=0.67890,
        )
        result = score.to_dict()

        assert result["overall"] == 0.123
        assert result["coherence"] == 0.235
        assert result["completeness"] == 0.346
        assert result["relevance"] == 0.457
        assert result["clarity"] == 0.568
        assert result["confidence"] == 0.679

    def test_excludes_features(self):
        """to_dict should not include features."""
        score = QualityScore(
            overall=0.5,
            coherence=0.5,
            completeness=0.5,
            relevance=0.5,
            clarity=0.5,
            confidence=0.5,
            features={"test": 1.0},
        )
        result = score.to_dict()
        assert "features" not in result


class TestQualityScoreProperties:
    """Tests for QualityScore computed properties."""

    def test_is_high_quality_true(self):
        """is_high_quality should be True for good scores."""
        score = QualityScore(
            overall=0.8,
            coherence=0.9,
            completeness=0.7,
            relevance=0.8,
            clarity=0.85,
            confidence=0.6,
        )
        assert score.is_high_quality is True

    def test_is_high_quality_false_low_overall(self):
        """is_high_quality should be False for low overall."""
        score = QualityScore(
            overall=0.6,  # Below 0.7 threshold
            coherence=0.9,
            completeness=0.7,
            relevance=0.8,
            clarity=0.85,
            confidence=0.9,
        )
        assert score.is_high_quality is False

    def test_is_high_quality_false_low_confidence(self):
        """is_high_quality should be False for low confidence."""
        score = QualityScore(
            overall=0.8,
            coherence=0.9,
            completeness=0.7,
            relevance=0.8,
            clarity=0.85,
            confidence=0.4,  # Below 0.5 threshold
        )
        assert score.is_high_quality is False

    def test_needs_review_true_low_overall(self):
        """needs_review should be True for low overall."""
        score = QualityScore(
            overall=0.4,  # Below 0.5 threshold
            coherence=0.9,
            completeness=0.7,
            relevance=0.8,
            clarity=0.85,
            confidence=0.9,
        )
        assert score.needs_review is True

    def test_needs_review_true_low_confidence(self):
        """needs_review should be True for low confidence."""
        score = QualityScore(
            overall=0.8,
            coherence=0.9,
            completeness=0.7,
            relevance=0.8,
            clarity=0.85,
            confidence=0.2,  # Below 0.3 threshold
        )
        assert score.needs_review is True

    def test_needs_review_false_good_scores(self):
        """needs_review should be False for good scores."""
        score = QualityScore(
            overall=0.8,
            coherence=0.9,
            completeness=0.7,
            relevance=0.8,
            clarity=0.85,
            confidence=0.9,
        )
        assert score.needs_review is False


# =============================================================================
# TestQualityScorerConfig - Configuration Tests
# =============================================================================


class TestQualityScorerConfig:
    """Tests for QualityScorerConfig."""

    def test_default_weights(self):
        """Default weights should sum to ~1.0."""
        config = QualityScorerConfig()
        total = (
            config.weight_coherence
            + config.weight_completeness
            + config.weight_relevance
            + config.weight_clarity
        )
        assert abs(total - 1.0) < 0.01

    def test_default_length_thresholds(self):
        """Default length thresholds should be reasonable."""
        config = QualityScorerConfig()
        assert config.min_response_length < config.optimal_response_length
        assert config.optimal_response_length < config.max_response_length

    def test_custom_config(self):
        """Should accept custom configuration."""
        config = QualityScorerConfig(
            weight_coherence=0.3,
            weight_completeness=0.3,
            weight_relevance=0.2,
            weight_clarity=0.2,
            min_response_length=100,
            use_embeddings=False,
        )
        assert config.weight_coherence == 0.3
        assert config.min_response_length == 100
        assert config.use_embeddings is False


# =============================================================================
# TestQualityScorerInit - Initialization Tests
# =============================================================================


class TestQualityScorerInit:
    """Tests for QualityScorer initialization."""

    def test_creates_with_default_config(self):
        """Should create with default config."""
        scorer = QualityScorer()
        assert scorer.config is not None
        assert isinstance(scorer.config, QualityScorerConfig)

    def test_creates_with_custom_config(self):
        """Should accept custom config."""
        config = QualityScorerConfig(use_embeddings=False)
        scorer = QualityScorer(config=config)
        assert scorer.config.use_embeddings is False


# =============================================================================
# TestQualityScorerExtractFeatures - Feature Extraction Tests
# =============================================================================


class TestQualityScorerExtractFeatures:
    """Tests for QualityScorer._extract_features()."""

    @pytest.fixture
    def scorer(self):
        """Create scorer with embeddings disabled."""
        return QualityScorer(QualityScorerConfig(use_embeddings=False))

    def test_extracts_length_features(self, scorer):
        """Should extract char, word, and sentence counts."""
        text = "This is a test. It has two sentences."
        features = scorer._extract_features(text)

        assert features["char_count"] == len(text)
        assert features["word_count"] == 8
        assert features["sentence_count"] >= 2

    def test_extracts_structural_features(self, scorer):
        """Should detect paragraphs, lists, code, headers."""
        text = """# Header

This is a paragraph.

- List item 1
- List item 2

```python
code block
```
"""
        features = scorer._extract_features(text)

        assert features["has_paragraphs"] == 1.0
        assert features["has_lists"] == 1.0
        assert features["has_code"] == 1.0
        assert features["has_headers"] == 1.0

    def test_extracts_vocabulary_features(self, scorer):
        """Should calculate vocabulary richness."""
        # High richness - all unique words
        features1 = scorer._extract_features("apple banana cherry date elderberry")
        assert features1["vocabulary_richness"] == 1.0

        # Low richness - repeated words
        features2 = scorer._extract_features("the the the the the")
        assert features2["vocabulary_richness"] == 0.2

    def test_extracts_filler_ratio(self, scorer):
        """Should detect filler words."""
        # Text with fillers
        features1 = scorer._extract_features("um like basically just really very")
        assert features1["filler_ratio"] > 0.5

        # Text without fillers
        features2 = scorer._extract_features("The function processes data efficiently")
        assert features2["filler_ratio"] < 0.1

    def test_extracts_technical_ratio(self, scorer):
        """Should detect technical words."""
        # Technical text
        features1 = scorer._extract_features(
            "The function implements the algorithm using database parameters"
        )
        assert features1["technical_ratio"] > 0.2

        # Non-technical text
        features2 = scorer._extract_features("The dog ran through the park happily")
        assert features2["technical_ratio"] < 0.1

    def test_detects_explanation_phrases(self, scorer):
        """Should detect explanation phrases."""
        features1 = scorer._extract_features("This is because the system processes data")
        assert features1["has_explanation"] == 1.0

        features2 = scorer._extract_features("Hello world")
        assert features2["has_explanation"] == 0.0


# =============================================================================
# TestQualityScorerComponentScores - Component Scoring Tests
# =============================================================================


class TestQualityScorerCoherence:
    """Tests for QualityScorer._score_coherence()."""

    @pytest.fixture
    def scorer(self):
        return QualityScorer(QualityScorerConfig(use_embeddings=False))

    def test_high_coherence_good_structure(self, scorer):
        """Good structure should score high coherence."""
        text = """This is a well-structured response. It contains multiple sentences.

The explanation is clear because it uses proper paragraph breaks.
Furthermore, the content flows logically from one point to another."""
        features = scorer._extract_features(text)
        score = scorer._score_coherence(features)
        assert score >= 0.6

    def test_low_coherence_short_text(self, scorer):
        """Very short text should score lower coherence."""
        text = "OK."
        features = scorer._extract_features(text)
        score = scorer._score_coherence(features)
        assert score <= 0.5  # Short text gets penalized


class TestQualityScorerCompleteness:
    """Tests for QualityScorer._score_completeness()."""

    @pytest.fixture
    def scorer(self):
        return QualityScorer(QualityScorerConfig(use_embeddings=False))

    def test_high_completeness_optimal_length(self, scorer):
        """Optimal length text should score high completeness."""
        # Generate text around optimal length (500 words)
        text = " ".join(["word"] * 400)
        features = scorer._extract_features(text)
        score = scorer._score_completeness(features)
        assert score >= 0.65  # Close to optimal gets good score

    def test_low_completeness_short_text(self, scorer):
        """Very short text should score low completeness."""
        text = "Yes."
        features = scorer._extract_features(text)
        score = scorer._score_completeness(features)
        assert score < 0.3


class TestQualityScorerClarity:
    """Tests for QualityScorer._score_clarity()."""

    @pytest.fixture
    def scorer(self):
        return QualityScorer(QualityScorerConfig(use_embeddings=False))

    def test_high_clarity_good_vocabulary(self, scorer):
        """Good vocabulary diversity should score high clarity."""
        text = "The implementation provides efficient data processing capabilities through optimized algorithms and careful resource management."
        features = scorer._extract_features(text)
        score = scorer._score_clarity(features)
        assert score >= 0.5

    def test_low_clarity_filler_words(self, scorer):
        """Text with filler words should score lower clarity."""
        text = "Um like basically just really very um like basically"
        features = scorer._extract_features(text)
        score = scorer._score_clarity(features)
        # Filler words reduce clarity but don't make it terrible
        assert score <= 0.7


class TestQualityScorerRelevance:
    """Tests for QualityScorer._score_relevance()."""

    @pytest.fixture
    def scorer(self):
        return QualityScorer(QualityScorerConfig(use_embeddings=False))

    def test_relevance_without_context(self, scorer):
        """Without context, should use heuristics."""
        text = "The function implements the algorithm efficiently."
        features = scorer._extract_features(text)
        score = scorer._score_relevance(text, None, features)
        assert 0.0 <= score <= 1.0

    def test_relevance_with_context_keyword_overlap(self, scorer):
        """With context, should use keyword overlap."""
        context = "Implement a sorting algorithm"
        text = "The sorting algorithm is implemented using quicksort for efficiency."
        features = scorer._extract_features(text)
        score = scorer._score_relevance(text, context, features)
        assert score > 0.3  # Should have keyword overlap


# =============================================================================
# TestQualityScorerScore - Main Scoring Tests
# =============================================================================


class TestQualityScorerScore:
    """Tests for QualityScorer.score()."""

    @pytest.fixture
    def scorer(self):
        return QualityScorer(QualityScorerConfig(use_embeddings=False))

    def test_empty_text_returns_zero_score(self, scorer):
        """Empty text should return zero overall score."""
        score = scorer.score("")
        assert score.overall == 0.0
        assert score.confidence == 1.0

    def test_whitespace_only_returns_zero_score(self, scorer):
        """Whitespace-only text should return zero score."""
        score = scorer.score("   \n\t  ")
        assert score.overall == 0.0

    def test_returns_quality_score_object(self, scorer):
        """Should return QualityScore instance."""
        score = scorer.score("This is a test response.")
        assert isinstance(score, QualityScore)

    def test_includes_all_component_scores(self, scorer):
        """Should include all component scores."""
        score = scorer.score("This is a comprehensive test response with multiple sentences.")
        assert 0.0 <= score.coherence <= 1.0
        assert 0.0 <= score.completeness <= 1.0
        assert 0.0 <= score.relevance <= 1.0
        assert 0.0 <= score.clarity <= 1.0
        assert 0.0 <= score.overall <= 1.0
        assert 0.0 <= score.confidence <= 1.0

    def test_high_quality_response(self, scorer):
        """High quality response should score well."""
        text = """
        The implementation uses a well-designed architecture that separates concerns effectively.

        First, the data layer handles all database operations through a repository pattern.
        This ensures that the business logic remains decoupled from storage details.

        Second, the service layer coordinates complex operations and enforces business rules.
        For example, validation is performed before any state changes occur.

        Finally, the presentation layer transforms data for API consumers.
        This separation allows each layer to evolve independently.
        """
        score = scorer.score(text)
        assert score.overall >= 0.5

    def test_low_quality_response(self, scorer):
        """Low quality response should score poorly."""
        score = scorer.score("ok")
        assert score.overall < 0.5

    def test_includes_features(self, scorer):
        """Score should include extracted features."""
        score = scorer.score("This is a test response with several words.")
        assert "word_count" in score.features
        assert "sentence_count" in score.features

    def test_with_context(self, scorer):
        """Should use context for relevance scoring."""
        context = "Explain how to implement a binary search algorithm"
        text = "Binary search works by repeatedly dividing the search interval in half."
        score = scorer.score(text, context=context)
        assert score.relevance > 0.0


class TestQualityScorerScoreBatch:
    """Tests for QualityScorer.score_batch()."""

    @pytest.fixture
    def scorer(self):
        return QualityScorer(QualityScorerConfig(use_embeddings=False))

    def test_returns_list_of_scores(self, scorer):
        """Should return list of QualityScore objects."""
        texts = ["First response.", "Second response.", "Third response."]
        scores = scorer.score_batch(texts)

        assert len(scores) == 3
        assert all(isinstance(s, QualityScore) for s in scores)

    def test_handles_empty_list(self, scorer):
        """Should handle empty input list."""
        scores = scorer.score_batch([])
        assert scores == []

    def test_with_contexts(self, scorer):
        """Should use provided contexts."""
        texts = ["Answer about sorting", "Answer about searching"]
        contexts = ["Explain sorting algorithms", "Explain search algorithms"]
        scores = scorer.score_batch(texts, contexts=contexts)

        assert len(scores) == 2

    def test_with_none_contexts(self, scorer):
        """Should work without contexts."""
        texts = ["First response.", "Second response."]
        scores = scorer.score_batch(texts, contexts=None)

        assert len(scores) == 2


# =============================================================================
# TestQualityScorerEmbeddings - Embedding Service Tests
# =============================================================================


class TestQualityScorerEmbeddings:
    """Tests for embedding service integration."""

    def test_lazy_loads_embedding_service(self):
        """Should lazy load embedding service."""
        scorer = QualityScorer(QualityScorerConfig(use_embeddings=True))
        assert scorer._embedding_service is None

        # Trigger lazy load (will fail without actual service)
        with patch("aragora.ml.quality_scorer.logger"):
            scorer._get_embedding_service()

    def test_disables_embeddings_on_error(self):
        """Should disable embeddings if service fails to load."""
        config = QualityScorerConfig(use_embeddings=True)
        scorer = QualityScorer(config)

        with patch(
            "aragora.ml.embeddings.get_embedding_service",
            side_effect=ImportError("No embedding service"),
        ):
            scorer._get_embedding_service()

        assert scorer.config.use_embeddings is False


# =============================================================================
# Integration Tests
# =============================================================================


class TestQualityScorerIntegration:
    """Integration tests for quality scoring workflow."""

    def test_filter_low_quality_responses(self):
        """Should filter low quality responses effectively."""
        scorer = QualityScorer(QualityScorerConfig(use_embeddings=False))

        responses = [
            "ok",  # Low quality
            "yes",  # Low quality
            """
            The implementation follows best practices by separating concerns.
            The data layer handles persistence through repositories.
            The service layer coordinates business operations.
            This architecture enables independent testing and evolution.
            """,  # High quality
        ]

        scores = scorer.score_batch(responses)

        # High quality response should score higher
        assert scores[2].overall > scores[0].overall
        assert scores[2].overall > scores[1].overall

    def test_consistency_across_calls(self):
        """Same text should produce same scores."""
        scorer = QualityScorer(QualityScorerConfig(use_embeddings=False))
        text = "This is a test response for consistency checking."

        score1 = scorer.score(text)
        score2 = scorer.score(text)

        assert score1.overall == score2.overall
        assert score1.coherence == score2.coherence
        assert score1.completeness == score2.completeness
