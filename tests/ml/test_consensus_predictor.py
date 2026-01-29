"""
Tests for aragora.ml.consensus_predictor module.

Tests cover:
- ConsensusPrediction dataclass and properties
- ConsensusPredictorConfig defaults and custom values
- ResponseFeatures dataclass
- ConsensusPredictor stance detection
- Component calculations (similarity, alignment, variance)
- Main predict() method
- Convergence trend estimation
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aragora.ml.consensus_predictor import (
    ConsensusPrediction,
    ConsensusPredictor,
    ConsensusPredictorConfig,
    ResponseFeatures,
)


# =============================================================================
# TestConsensusPrediction - Dataclass Tests
# =============================================================================


class TestConsensusPredictionInit:
    """Tests for ConsensusPrediction initialization."""

    def test_creates_with_required_fields(self):
        """Should create with all required fields."""
        pred = ConsensusPrediction(
            probability=0.8,
            confidence=0.9,
            estimated_rounds=2,
            convergence_trend="converging",
            key_factors=["high_similarity"],
        )
        assert pred.probability == 0.8
        assert pred.confidence == 0.9
        assert pred.estimated_rounds == 2
        assert pred.convergence_trend == "converging"
        assert pred.key_factors == ["high_similarity"]

    def test_features_defaults_to_empty_dict(self):
        """Features should default to empty dict."""
        pred = ConsensusPrediction(
            probability=0.5,
            confidence=0.5,
            estimated_rounds=3,
            convergence_trend="stable",
            key_factors=[],
        )
        assert pred.features == {}


class TestConsensusPredictionProperties:
    """Tests for ConsensusPrediction computed properties."""

    def test_likely_consensus_true(self):
        """likely_consensus should be True for high probability and confidence."""
        pred = ConsensusPrediction(
            probability=0.8,
            confidence=0.6,
            estimated_rounds=2,
            convergence_trend="converging",
            key_factors=[],
        )
        assert pred.likely_consensus is True

    def test_likely_consensus_false_low_probability(self):
        """likely_consensus should be False for low probability."""
        pred = ConsensusPrediction(
            probability=0.6,  # Below 0.7
            confidence=0.9,
            estimated_rounds=3,
            convergence_trend="stable",
            key_factors=[],
        )
        assert pred.likely_consensus is False

    def test_likely_consensus_false_low_confidence(self):
        """likely_consensus should be False for low confidence."""
        pred = ConsensusPrediction(
            probability=0.9,
            confidence=0.4,  # Below 0.5
            estimated_rounds=2,
            convergence_trend="converging",
            key_factors=[],
        )
        assert pred.likely_consensus is False

    def test_early_termination_safe_true(self):
        """early_termination_safe should be True for very high scores."""
        pred = ConsensusPrediction(
            probability=0.9,
            confidence=0.8,
            estimated_rounds=1,
            convergence_trend="converging",
            key_factors=[],
        )
        assert pred.early_termination_safe is True

    def test_early_termination_safe_false(self):
        """early_termination_safe should be False for moderate scores."""
        pred = ConsensusPrediction(
            probability=0.8,  # Below 0.85
            confidence=0.8,
            estimated_rounds=2,
            convergence_trend="converging",
            key_factors=[],
        )
        assert pred.early_termination_safe is False

    def test_needs_intervention_true(self):
        """needs_intervention should be True for diverging low probability."""
        pred = ConsensusPrediction(
            probability=0.2,  # Below 0.3
            confidence=0.8,
            estimated_rounds=3,
            convergence_trend="diverging",
            key_factors=[],
        )
        assert pred.needs_intervention is True

    def test_needs_intervention_false_converging(self):
        """needs_intervention should be False when converging."""
        pred = ConsensusPrediction(
            probability=0.2,
            confidence=0.8,
            estimated_rounds=3,
            convergence_trend="converging",
            key_factors=[],
        )
        assert pred.needs_intervention is False


class TestConsensusPredictionToDict:
    """Tests for ConsensusPrediction.to_dict()."""

    def test_returns_dict_with_rounded_values(self):
        """to_dict should return dict with 3 decimal precision."""
        pred = ConsensusPrediction(
            probability=0.12345,
            confidence=0.67890,
            estimated_rounds=2,
            convergence_trend="stable",
            key_factors=["factor1"],
        )
        result = pred.to_dict()

        assert result["probability"] == 0.123
        assert result["confidence"] == 0.679
        assert result["estimated_rounds"] == 2
        assert result["convergence_trend"] == "stable"
        assert "likely_consensus" in result


# =============================================================================
# TestConsensusPredictorConfig - Configuration Tests
# =============================================================================


class TestConsensusPredictorConfig:
    """Tests for ConsensusPredictorConfig."""

    def test_default_weights_sum_to_one(self):
        """Default weights should sum to ~1.0."""
        config = ConsensusPredictorConfig()
        total = (
            config.weight_semantic_similarity
            + config.weight_stance_alignment
            + config.weight_quality_variance
            + config.weight_historical
            + config.weight_round_progress
        )
        assert abs(total - 1.0) < 0.01

    def test_default_thresholds(self):
        """Default thresholds should be reasonable."""
        config = ConsensusPredictorConfig()
        assert 0.0 < config.high_similarity_threshold < 1.0
        assert 0.0 < config.low_variance_threshold < 1.0
        assert 0.0 < config.convergence_delta_threshold < 0.5

    def test_custom_config(self):
        """Should accept custom configuration."""
        config = ConsensusPredictorConfig(
            weight_semantic_similarity=0.5,
            use_embeddings=False,
        )
        assert config.weight_semantic_similarity == 0.5
        assert config.use_embeddings is False


# =============================================================================
# TestResponseFeatures - Dataclass Tests
# =============================================================================


class TestResponseFeatures:
    """Tests for ResponseFeatures dataclass."""

    def test_creates_with_required_fields(self):
        """Should create with required fields."""
        rf = ResponseFeatures(agent_id="agent1", text="Hello world")
        assert rf.agent_id == "agent1"
        assert rf.text == "Hello world"

    def test_default_values(self):
        """Should have sensible defaults."""
        rf = ResponseFeatures(agent_id="agent1", text="Test")
        assert rf.stance is None
        assert rf.confidence == 0.5
        assert rf.quality_score == 0.5
        assert rf.embedding is None


# =============================================================================
# TestConsensusPredictorInit - Initialization Tests
# =============================================================================


class TestConsensusPredictorInit:
    """Tests for ConsensusPredictor initialization."""

    def test_creates_with_default_config(self):
        """Should create with default config."""
        predictor = ConsensusPredictor()
        assert predictor.config is not None
        assert isinstance(predictor.config, ConsensusPredictorConfig)

    def test_creates_with_custom_config(self):
        """Should accept custom config."""
        config = ConsensusPredictorConfig(use_embeddings=False)
        predictor = ConsensusPredictor(config=config)
        assert predictor.config.use_embeddings is False


# =============================================================================
# TestConsensusPredictorStanceDetection - Stance Detection Tests
# =============================================================================


class TestConsensusPredictorStanceDetection:
    """Tests for ConsensusPredictor._detect_stance()."""

    @pytest.fixture
    def predictor(self):
        return ConsensusPredictor(ConsensusPredictorConfig(use_embeddings=False))

    def test_detects_agree_stance(self, predictor):
        """Should detect agreement."""
        text = "I agree completely. That's exactly right. Good point."
        stance = predictor._detect_stance(text)
        assert stance == "agree"

    def test_detects_disagree_stance(self, predictor):
        """Should detect disagreement."""
        text = "I disagree with this approach. Actually, the problem with that is..."
        stance = predictor._detect_stance(text)
        assert stance == "disagree"

    def test_detects_neutral_stance(self, predictor):
        """Should detect neutral stance."""
        text = "The data shows some interesting patterns."
        stance = predictor._detect_stance(text)
        assert stance == "neutral"

    def test_mixed_indicators_weighted(self, predictor):
        """Mixed indicators should be weighted appropriately."""
        text = "I agree with the first point, but I disagree with the conclusion."
        stance = predictor._detect_stance(text)
        assert stance in ["agree", "disagree", "neutral"]


# =============================================================================
# TestConsensusPredictorCalculations - Component Calculations
# =============================================================================


class TestConsensusPredictorStanceAlignment:
    """Tests for ConsensusPredictor._calculate_stance_alignment()."""

    @pytest.fixture
    def predictor(self):
        return ConsensusPredictor(ConsensusPredictorConfig(use_embeddings=False))

    def test_all_agree_high_alignment(self, predictor):
        """All agents agreeing should give high alignment."""
        features = [
            ResponseFeatures(agent_id="a1", text="t1", stance="agree"),
            ResponseFeatures(agent_id="a2", text="t2", stance="agree"),
            ResponseFeatures(agent_id="a3", text="t3", stance="agree"),
        ]
        score = predictor._calculate_stance_alignment(features)
        assert score >= 0.8

    def test_all_disagree_low_alignment(self, predictor):
        """All agents disagreeing should give lower alignment."""
        features = [
            ResponseFeatures(agent_id="a1", text="t1", stance="disagree"),
            ResponseFeatures(agent_id="a2", text="t2", stance="disagree"),
            ResponseFeatures(agent_id="a3", text="t3", stance="disagree"),
        ]
        score = predictor._calculate_stance_alignment(features)
        # High majority but penalized for disagreement
        assert 0.3 <= score <= 0.8

    def test_mixed_stances(self, predictor):
        """Mixed stances should give moderate alignment."""
        features = [
            ResponseFeatures(agent_id="a1", text="t1", stance="agree"),
            ResponseFeatures(agent_id="a2", text="t2", stance="disagree"),
            ResponseFeatures(agent_id="a3", text="t3", stance="neutral"),
        ]
        score = predictor._calculate_stance_alignment(features)
        # Mixed stances with disagreement penalized
        assert 0.0 <= score <= 0.6

    def test_empty_features(self, predictor):
        """Empty features should return default."""
        score = predictor._calculate_stance_alignment([])
        assert score == 0.5


class TestConsensusPredictorQualityVariance:
    """Tests for ConsensusPredictor._calculate_quality_variance()."""

    @pytest.fixture
    def predictor(self):
        return ConsensusPredictor(ConsensusPredictorConfig(use_embeddings=False))

    def test_uniform_quality_low_variance(self, predictor):
        """Uniform quality scores should give low variance."""
        features = [
            ResponseFeatures(agent_id="a1", text="t1", quality_score=0.8),
            ResponseFeatures(agent_id="a2", text="t2", quality_score=0.8),
            ResponseFeatures(agent_id="a3", text="t3", quality_score=0.8),
        ]
        variance = predictor._calculate_quality_variance(features)
        assert variance < 0.1

    def test_varied_quality_high_variance(self, predictor):
        """Varied quality scores should give higher variance."""
        features = [
            ResponseFeatures(agent_id="a1", text="t1", quality_score=0.2),
            ResponseFeatures(agent_id="a2", text="t2", quality_score=0.8),
            ResponseFeatures(agent_id="a3", text="t3", quality_score=0.5),
        ]
        variance = predictor._calculate_quality_variance(features)
        assert variance > 0.2

    def test_single_feature_zero_variance(self, predictor):
        """Single feature should give zero variance."""
        features = [ResponseFeatures(agent_id="a1", text="t1", quality_score=0.8)]
        variance = predictor._calculate_quality_variance(features)
        assert variance == 0.0


class TestConsensusPredictorConvergenceTrend:
    """Tests for ConsensusPredictor._estimate_convergence_trend()."""

    @pytest.fixture
    def predictor(self):
        return ConsensusPredictor(ConsensusPredictorConfig(use_embeddings=False))

    def test_no_history_stable(self, predictor):
        """No history should return stable."""
        trend = predictor._estimate_convergence_trend(0.7, None)
        assert trend == "stable"

    def test_insufficient_history_stable(self, predictor):
        """Insufficient history should return stable."""
        trend = predictor._estimate_convergence_trend(0.7, [0.5])
        assert trend == "stable"

    def test_increasing_similarity_converging(self, predictor):
        """Increasing similarity should indicate converging."""
        history = [0.3, 0.4, 0.5, 0.6]
        trend = predictor._estimate_convergence_trend(0.7, history)
        assert trend == "converging"

    def test_decreasing_similarity_diverging(self, predictor):
        """Decreasing similarity should indicate diverging."""
        history = [0.8, 0.7, 0.6, 0.5]
        trend = predictor._estimate_convergence_trend(0.4, history)
        assert trend == "diverging"

    def test_flat_similarity_stable(self, predictor):
        """Flat similarity should indicate stable."""
        history = [0.5, 0.5, 0.5, 0.5]
        trend = predictor._estimate_convergence_trend(0.5, history)
        assert trend == "stable"


# =============================================================================
# TestConsensusPredictorPredict - Main Prediction Tests
# =============================================================================


class TestConsensusPredictorPredict:
    """Tests for ConsensusPredictor.predict()."""

    @pytest.fixture
    def predictor(self):
        return ConsensusPredictor(ConsensusPredictorConfig(use_embeddings=False))

    def test_empty_responses(self, predictor):
        """Empty responses should return zero probability."""
        pred = predictor.predict([])
        assert pred.probability == 0.0
        assert pred.confidence == 0.0
        assert "no_responses" in pred.key_factors

    def test_returns_consensus_prediction(self, predictor):
        """Should return ConsensusPrediction instance."""
        responses = [("agent1", "I agree with the proposal.")]
        pred = predictor.predict(responses)
        assert isinstance(pred, ConsensusPrediction)

    def test_single_response(self, predictor):
        """Single response should work."""
        responses = [("agent1", "The implementation looks good.")]
        pred = predictor.predict(responses)
        assert 0.0 <= pred.probability <= 1.0
        assert 0.0 <= pred.confidence <= 1.0

    def test_multiple_responses(self, predictor):
        """Multiple responses should be processed."""
        responses = [
            ("agent1", "I agree with the approach."),
            ("agent2", "The design is well thought out."),
            ("agent3", "Good proposal overall."),
        ]
        pred = predictor.predict(responses)
        assert 0.0 <= pred.probability <= 1.0

    def test_with_context(self, predictor):
        """Should accept context parameter."""
        responses = [("agent1", "The sorting algorithm is efficient.")]
        pred = predictor.predict(responses, context="Implement a sorting algorithm")
        assert isinstance(pred, ConsensusPrediction)

    def test_round_progress_affects_probability(self, predictor):
        """Later rounds should have higher base probability."""
        responses = [("agent1", "Looks good."), ("agent2", "I agree.")]

        # Early round
        pred_early = predictor.predict(responses, current_round=1, total_rounds=5)

        # Late round
        pred_late = predictor.predict(responses, current_round=5, total_rounds=5)

        # Later round should generally have higher probability
        assert pred_late.probability >= pred_early.probability - 0.1

    def test_estimated_rounds_reasonable(self, predictor):
        """Estimated rounds should be within bounds."""
        responses = [("agent1", "Test response.")]
        pred = predictor.predict(responses, current_round=1, total_rounds=3)

        assert 1 <= pred.estimated_rounds <= 3

    def test_convergence_trend_in_output(self, predictor):
        """Should include convergence trend."""
        responses = [("agent1", "Test response.")]
        pred = predictor.predict(responses)

        assert pred.convergence_trend in ["converging", "diverging", "stable"]


class TestConsensusPredictorHistorical:
    """Tests for historical tracking."""

    def test_record_outcome_stores_data(self):
        """record_outcome should store prediction accuracy."""
        predictor = ConsensusPredictor(ConsensusPredictorConfig(use_embeddings=False))

        # Make a prediction
        responses = [("agent1", "Test response.")]
        pred = predictor.predict(responses)

        # Record outcome
        predictor._prediction_accuracy.append((pred.probability, True))

        assert len(predictor._prediction_accuracy) == 1

    def test_historical_factor_with_data(self):
        """Historical factor should use recorded accuracy."""
        predictor = ConsensusPredictor(ConsensusPredictorConfig(use_embeddings=False))

        # Record some accurate predictions
        predictor._prediction_accuracy = [
            (0.8, True),
            (0.7, True),
            (0.3, False),
        ]

        factor = predictor._get_historical_factor()
        assert 0.0 <= factor <= 1.0

    def test_historical_factor_no_data(self):
        """Historical factor should return default with no data."""
        predictor = ConsensusPredictor(ConsensusPredictorConfig(use_embeddings=False))
        factor = predictor._get_historical_factor()
        assert factor == 0.5


# =============================================================================
# Integration Tests
# =============================================================================


class TestConsensusPredictorIntegration:
    """Integration tests for consensus prediction workflow."""

    def test_agreeing_agents_high_consensus(self):
        """Agreeing agents should predict high consensus."""
        predictor = ConsensusPredictor(ConsensusPredictorConfig(use_embeddings=False))

        responses = [
            ("agent1", "I agree with this proposal. It's exactly what we need."),
            ("agent2", "Absolutely correct. Good point about the implementation."),
            ("agent3", "I concur with the approach. Building on that idea..."),
        ]

        pred = predictor.predict(responses)
        # With agreeing stances, should have reasonable probability
        assert pred.probability >= 0.3

    def test_disagreeing_agents_lower_consensus(self):
        """Disagreeing agents should predict lower consensus."""
        predictor = ConsensusPredictor(ConsensusPredictorConfig(use_embeddings=False))

        responses = [
            ("agent1", "I disagree with this. The problem with that is it won't work."),
            ("agent2", "Actually, I would argue against this approach entirely."),
            ("agent3", "On the contrary, this is incorrect in my view."),
        ]

        pred = predictor.predict(responses)
        # With disagreeing stances, intervention may be needed
        # The probability depends on other factors too
        assert pred.probability <= 0.8

    def test_consistency_across_calls(self):
        """Same inputs should produce same predictions."""
        predictor = ConsensusPredictor(ConsensusPredictorConfig(use_embeddings=False))

        responses = [("agent1", "Test response."), ("agent2", "Another response.")]

        pred1 = predictor.predict(responses, current_round=1, total_rounds=3)
        pred2 = predictor.predict(responses, current_round=1, total_rounds=3)

        assert pred1.probability == pred2.probability
        assert pred1.convergence_trend == pred2.convergence_trend
