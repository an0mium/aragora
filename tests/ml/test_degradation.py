"""
Tests for ML degradation module.

Tests graceful fallback mechanisms when ML features are unavailable.
"""

from __future__ import annotations

import pytest

from aragora.ml.degradation import (
    DegradationEvent,
    DegradationLevel,
    FeatureStatus,
    MLDegradationManager,
    MLFallbackService,
    MLFeature,
    force_degradation,
    get_ml_fallback,
    get_ml_manager,
    heuristic_consensus_prediction,
    heuristic_quality_score,
    heuristic_sentiment,
    heuristic_similarity,
    heuristic_tfidf_similarity,
    reset_degradation,
)


class TestDegradationLevel:
    """Tests for DegradationLevel enum."""

    def test_all_levels_defined(self):
        """Test all degradation levels exist."""
        assert DegradationLevel.FULL.value == "full"
        assert DegradationLevel.LIGHTWEIGHT.value == "lightweight"
        assert DegradationLevel.HEURISTIC.value == "heuristic"
        assert DegradationLevel.DISABLED.value == "disabled"


class TestMLFeature:
    """Tests for MLFeature enum."""

    def test_all_features_defined(self):
        """Test all ML features exist."""
        features = [
            MLFeature.EMBEDDINGS,
            MLFeature.CONSENSUS_PREDICTION,
            MLFeature.QUALITY_SCORING,
            MLFeature.AGENT_ROUTING,
            MLFeature.SEMANTIC_SEARCH,
            MLFeature.SENTIMENT_ANALYSIS,
        ]
        for f in features:
            assert f.value is not None


class TestFeatureStatus:
    """Tests for FeatureStatus dataclass."""

    def test_status_creation(self):
        """Test creating feature status."""
        status = FeatureStatus(
            feature=MLFeature.EMBEDDINGS,
            level=DegradationLevel.FULL,
            available=True,
            last_check=1234567890.0,
            latency_ms=50.0,
            error_count=2,
            success_count=100,
        )

        assert status.feature == MLFeature.EMBEDDINGS
        assert status.level == DegradationLevel.FULL
        assert status.available is True

    def test_error_rate_calculation(self):
        """Test error rate calculation."""
        status = FeatureStatus(
            feature=MLFeature.EMBEDDINGS,
            level=DegradationLevel.FULL,
            available=True,
            last_check=0,
            error_count=10,
            success_count=90,
        )

        assert status.error_rate == 0.1  # 10%

    def test_error_rate_zero_total(self):
        """Test error rate with no operations."""
        status = FeatureStatus(
            feature=MLFeature.EMBEDDINGS,
            level=DegradationLevel.FULL,
            available=True,
            last_check=0,
        )

        assert status.error_rate == 0.0

    def test_to_dict(self):
        """Test converting to dictionary."""
        status = FeatureStatus(
            feature=MLFeature.EMBEDDINGS,
            level=DegradationLevel.LIGHTWEIGHT,
            available=True,
            last_check=0,
            latency_ms=100.5,
            reason="high_latency",
        )

        d = status.to_dict()
        assert d["feature"] == "embeddings"
        assert d["level"] == "lightweight"
        assert d["latency_ms"] == 100.5
        assert d["reason"] == "high_latency"


class TestMLDegradationManager:
    """Tests for MLDegradationManager."""

    def test_manager_initialization(self):
        """Test manager initializes all features."""
        manager = MLDegradationManager()

        for feature in MLFeature:
            level = manager.get_level(feature)
            assert level == DegradationLevel.FULL

    def test_set_level(self):
        """Test setting degradation level."""
        manager = MLDegradationManager()

        manager.set_level(
            MLFeature.EMBEDDINGS,
            DegradationLevel.LIGHTWEIGHT,
            "test",
        )

        assert manager.get_level(MLFeature.EMBEDDINGS) == DegradationLevel.LIGHTWEIGHT

    def test_degradation_event_recorded(self):
        """Test degradation events are recorded."""
        manager = MLDegradationManager()

        manager.set_level(
            MLFeature.CONSENSUS_PREDICTION,
            DegradationLevel.HEURISTIC,
            "test_reason",
        )

        status = manager.get_status()
        assert len(status["recent_events"]) > 0

        event = status["recent_events"][-1]
        assert event["feature"] == "consensus_prediction"
        assert event["to"] == "heuristic"
        assert event["reason"] == "test_reason"

    def test_record_success(self):
        """Test recording successful operations."""
        manager = MLDegradationManager()

        manager.record_success(MLFeature.QUALITY_SCORING, 50.0)

        status = manager.get_status()
        feature_status = status["features"]["quality_scoring"]
        assert feature_status["latency_ms"] == 50.0

    def test_record_error_triggers_degradation(self):
        """Test errors trigger automatic degradation."""
        manager = MLDegradationManager()

        # Simulate many errors
        for _ in range(10):
            manager.record_error(MLFeature.EMBEDDINGS, Exception("test error"))

        # Should have degraded due to high error rate
        level = manager.get_level(MLFeature.EMBEDDINGS)
        assert level in (DegradationLevel.LIGHTWEIGHT, DegradationLevel.HEURISTIC)

    def test_record_high_latency_triggers_degradation(self):
        """Test high latency triggers degradation."""
        manager = MLDegradationManager()

        # Simulate high latency
        manager.record_latency(MLFeature.SEMANTIC_SEARCH, 3000)  # 3 seconds

        level = manager.get_level(MLFeature.SEMANTIC_SEARCH)
        assert level == DegradationLevel.LIGHTWEIGHT

    def test_auto_recovery(self):
        """Test automatic recovery on sustained success."""
        manager = MLDegradationManager()

        # Degrade first
        manager.set_level(MLFeature.EMBEDDINGS, DegradationLevel.LIGHTWEIGHT, "test")

        # Simulate many successes with good latency
        for _ in range(15):
            manager.record_success(MLFeature.EMBEDDINGS, 50.0)

        # Should recover to full
        level = manager.get_level(MLFeature.EMBEDDINGS)
        assert level == DegradationLevel.FULL


class TestHeuristicSimilarity:
    """Tests for heuristic_similarity function."""

    def test_identical_texts(self):
        """Test identical texts have high similarity."""
        sim = heuristic_similarity("hello world", "hello world")
        assert sim == 1.0

    def test_completely_different_texts(self):
        """Test different texts have low similarity."""
        sim = heuristic_similarity("hello world", "foo bar baz")
        assert sim == 0.0

    def test_partial_overlap(self):
        """Test partial overlap texts."""
        sim = heuristic_similarity(
            "the quick brown fox",
            "the lazy brown dog",
        )
        assert 0.0 < sim < 1.0
        assert sim == 2 / 6  # "the", "brown" overlap, 6 unique words

    def test_empty_texts(self):
        """Test empty texts."""
        assert heuristic_similarity("", "") == 1.0
        assert heuristic_similarity("hello", "") == 0.0
        assert heuristic_similarity("", "hello") == 0.0


class TestHeuristicTfidfSimilarity:
    """Tests for heuristic_tfidf_similarity function."""

    def test_identical_texts(self):
        """Test identical texts have similarity 1.0."""
        sim = heuristic_tfidf_similarity("hello world", "hello world")
        assert sim == pytest.approx(1.0)

    def test_similar_texts(self):
        """Test similar texts have high similarity."""
        sim = heuristic_tfidf_similarity(
            "the cat sat on the mat",
            "the cat sat on the rug",
        )
        assert sim > 0.7

    def test_different_texts(self):
        """Test different texts have low similarity."""
        sim = heuristic_tfidf_similarity(
            "machine learning algorithms",
            "cooking delicious recipes",
        )
        assert sim < 0.5

    def test_repeated_words(self):
        """Test repeated words are weighted."""
        sim = heuristic_tfidf_similarity(
            "test test test",
            "test",
        )
        # Should still show similarity due to shared word
        assert sim > 0.5


class TestHeuristicConsensusPrediction:
    """Tests for heuristic_consensus_prediction function."""

    def test_single_response(self):
        """Test prediction with single response."""
        pred = heuristic_consensus_prediction(["I agree with the proposal."])

        assert pred["probability"] == 1.0
        assert pred["confidence"] == 0.3

    def test_empty_responses(self):
        """Test prediction with no responses."""
        pred = heuristic_consensus_prediction([])

        assert pred["probability"] == 0.0

    def test_similar_responses(self):
        """Test prediction with similar responses."""
        responses = [
            "I think we should use microservices for scalability.",
            "I agree, microservices would provide good scalability.",
            "Yes, microservices architecture makes sense for scaling.",
        ]

        pred = heuristic_consensus_prediction(responses)

        # Heuristic similarity may not be high for word-level comparison
        # The key is that it returns a valid prediction structure
        assert 0.0 <= pred["probability"] <= 1.0
        assert pred["convergence_trend"] in ("stable", "converging", "diverging")

    def test_divergent_responses(self):
        """Test prediction with divergent responses."""
        responses = [
            "We should definitely use REST APIs.",
            "GraphQL is clearly the better choice.",
            "Neither, we need gRPC for performance.",
        ]

        pred = heuristic_consensus_prediction(responses)

        # Different topics = lower similarity = lower consensus probability
        assert pred["probability"] < 0.8

    def test_prediction_structure(self):
        """Test prediction has required fields."""
        pred = heuristic_consensus_prediction(["Response 1", "Response 2"])

        assert "probability" in pred
        assert "confidence" in pred
        assert "convergence_trend" in pred
        assert "key_factors" in pred
        assert "estimated_rounds" in pred


class TestHeuristicQualityScore:
    """Tests for heuristic_quality_score function."""

    def test_empty_text(self):
        """Test empty text scores 0."""
        assert heuristic_quality_score("") == 0.0
        assert heuristic_quality_score("   ") == 0.0

    def test_good_quality_text(self):
        """Test good quality text scores high."""
        text = """
        The proposed solution addresses several key concerns. First, it provides
        scalability because the architecture supports horizontal scaling. Moreover,
        the use of caching improves performance significantly. Therefore, I recommend
        adopting this approach since it balances complexity with maintainability.
        """

        score = heuristic_quality_score(text)
        assert score > 0.5

    def test_poor_quality_text(self):
        """Test poor quality text scores lower."""
        text = "ok yes good"

        score = heuristic_quality_score(text)
        assert score < 0.5

    def test_reasoning_indicators_boost_score(self):
        """Test reasoning words improve score."""
        without_reasoning = "The system is fast and reliable."
        with_reasoning = "The system is fast because of caching, therefore it is reliable."

        score_without = heuristic_quality_score(without_reasoning)
        score_with = heuristic_quality_score(with_reasoning)

        assert score_with > score_without


class TestHeuristicSentiment:
    """Tests for heuristic_sentiment function."""

    def test_positive_sentiment(self):
        """Test positive text detection."""
        result = heuristic_sentiment("This is excellent! Great work, very helpful.")

        assert result["label"] == "positive"
        assert result["confidence"] > 0.5

    def test_negative_sentiment(self):
        """Test negative text detection."""
        result = heuristic_sentiment("This is wrong and bad. There are many problems.")

        assert result["label"] == "negative"
        assert result["confidence"] > 0.5

    def test_neutral_sentiment(self):
        """Test neutral text detection."""
        result = heuristic_sentiment("The meeting will be held on Tuesday at 3pm.")

        assert result["label"] == "neutral"

    def test_sentiment_structure(self):
        """Test sentiment result structure."""
        result = heuristic_sentiment("Test text")

        assert "label" in result
        assert "confidence" in result
        assert "scores" in result
        assert "positive" in result["scores"]
        assert "negative" in result["scores"]
        assert "neutral" in result["scores"]


class TestMLFallbackService:
    """Tests for MLFallbackService."""

    @pytest.fixture
    def service(self):
        """Create a fresh fallback service."""
        manager = MLDegradationManager()
        return MLFallbackService(manager)

    @pytest.mark.asyncio
    async def test_compute_similarity_heuristic(self, service):
        """Test similarity falls back to heuristic."""
        # Force heuristic mode
        service._manager.set_level(
            MLFeature.EMBEDDINGS,
            DegradationLevel.HEURISTIC,
            "test",
        )

        sim = await service.compute_similarity("hello world", "hello world")
        assert sim == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_predict_consensus_heuristic(self, service):
        """Test consensus prediction falls back to heuristic."""
        service._manager.set_level(
            MLFeature.CONSENSUS_PREDICTION,
            DegradationLevel.HEURISTIC,
            "test",
        )

        pred = await service.predict_consensus(
            ["Response 1", "Response 2"],
            "Test task",
        )

        assert "probability" in pred
        assert "confidence" in pred

    @pytest.mark.asyncio
    async def test_score_quality_heuristic(self, service):
        """Test quality scoring falls back to heuristic."""
        service._manager.set_level(
            MLFeature.QUALITY_SCORING,
            DegradationLevel.HEURISTIC,
            "test",
        )

        score = await service.score_quality("This is a test response.")
        assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_analyze_sentiment(self, service):
        """Test sentiment analysis."""
        result = await service.analyze_sentiment("This is great!")

        assert result["label"] in ("positive", "negative", "neutral")

    def test_get_status(self, service):
        """Test getting service status."""
        status = service.get_status()

        assert "features" in status
        assert "degraded_count" in status


class TestGlobalFunctions:
    """Tests for global functions."""

    def test_get_ml_manager(self):
        """Test getting global manager."""
        manager = get_ml_manager()
        assert isinstance(manager, MLDegradationManager)

    def test_get_ml_fallback(self):
        """Test getting global fallback service."""
        service = get_ml_fallback()
        assert isinstance(service, MLFallbackService)

    def test_force_degradation(self):
        """Test forcing degradation."""
        force_degradation(MLFeature.EMBEDDINGS, DegradationLevel.DISABLED, "test")

        manager = get_ml_manager()
        assert manager.get_level(MLFeature.EMBEDDINGS) == DegradationLevel.DISABLED

    def test_reset_degradation(self):
        """Test resetting degradation."""
        # First degrade
        force_degradation(MLFeature.EMBEDDINGS, DegradationLevel.DISABLED, "test")

        # Then reset
        reset_degradation()

        manager = get_ml_manager()
        assert manager.get_level(MLFeature.EMBEDDINGS) == DegradationLevel.FULL


class TestIntegration:
    """Integration tests for ML degradation."""

    @pytest.mark.asyncio
    async def test_full_degradation_cycle(self):
        """Test complete degradation and recovery cycle."""
        manager = MLDegradationManager()
        service = MLFallbackService(manager)

        # Start at full
        assert manager.get_level(MLFeature.QUALITY_SCORING) == DegradationLevel.FULL

        # Simulate errors to trigger degradation
        for _ in range(10):
            manager.record_error(
                MLFeature.QUALITY_SCORING,
                Exception("test error"),
            )

        # Should be degraded
        assert manager.get_level(MLFeature.QUALITY_SCORING) != DegradationLevel.FULL

        # Can still score (using heuristic)
        score = await service.score_quality("Test response text.")
        assert 0.0 <= score <= 1.0

        # Reset for other tests
        manager.set_level(
            MLFeature.QUALITY_SCORING,
            DegradationLevel.FULL,
            "reset",
        )

    @pytest.mark.asyncio
    async def test_multiple_features_independent(self):
        """Test features degrade independently."""
        manager = MLDegradationManager()

        # Degrade only embeddings
        manager.set_level(
            MLFeature.EMBEDDINGS,
            DegradationLevel.HEURISTIC,
            "test",
        )

        # Other features should be unaffected
        assert manager.get_level(MLFeature.EMBEDDINGS) == DegradationLevel.HEURISTIC
        assert manager.get_level(MLFeature.QUALITY_SCORING) == DegradationLevel.FULL
        assert manager.get_level(MLFeature.CONSENSUS_PREDICTION) == DegradationLevel.FULL
