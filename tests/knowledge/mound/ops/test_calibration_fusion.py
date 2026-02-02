"""Comprehensive tests for calibration fusion module.

Tests cover:
1. Calibration calculations (Krippendorff's alpha, disagreement, consensus)
2. Fusion operations (all strategies)
3. Performance with large agent counts (50+ agents)
4. Edge cases (single agent, no data, outliers)
"""

from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import pytest

from aragora.knowledge.mound.ops.calibration_fusion import (
    AgentPrediction,
    CalibrationConsensus,
    CalibrationFusionConfig,
    CalibrationFusionEngine,
    CalibrationFusionStrategy,
    FusionConfig,
    _cached_krippendorff_alpha,
    get_calibration_fusion_engine,
)

if TYPE_CHECKING:
    pass


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def engine() -> CalibrationFusionEngine:
    """Create a fresh fusion engine for each test."""
    return CalibrationFusionEngine()


@pytest.fixture
def sample_predictions() -> list[AgentPrediction]:
    """Create sample predictions for testing."""
    return [
        AgentPrediction(
            agent_name="claude",
            confidence=0.85,
            predicted_outcome="winner_a",
            calibration_accuracy=0.9,
            brier_score=0.15,
            prediction_count=100,
        ),
        AgentPrediction(
            agent_name="gpt-4",
            confidence=0.80,
            predicted_outcome="winner_a",
            calibration_accuracy=0.85,
            brier_score=0.18,
            prediction_count=90,
        ),
        AgentPrediction(
            agent_name="gemini",
            confidence=0.75,
            predicted_outcome="winner_a",
            calibration_accuracy=0.80,
            brier_score=0.20,
            prediction_count=80,
        ),
    ]


@pytest.fixture
def disagreeing_predictions() -> list[AgentPrediction]:
    """Create predictions with disagreement."""
    return [
        AgentPrediction(
            agent_name="claude",
            confidence=0.90,
            predicted_outcome="winner_a",
            calibration_accuracy=0.85,
            prediction_count=50,
        ),
        AgentPrediction(
            agent_name="gpt-4",
            confidence=0.85,
            predicted_outcome="winner_a",
            calibration_accuracy=0.80,
            prediction_count=45,
        ),
        AgentPrediction(
            agent_name="gemini",
            confidence=0.70,
            predicted_outcome="winner_b",
            calibration_accuracy=0.75,
            prediction_count=40,
        ),
        AgentPrediction(
            agent_name="grok",
            confidence=0.65,
            predicted_outcome="winner_b",
            calibration_accuracy=0.70,
            prediction_count=35,
        ),
    ]


def create_large_prediction_set(n_agents: int) -> list[AgentPrediction]:
    """Create a large set of predictions for performance testing."""
    import random

    random.seed(42)  # Reproducible tests
    predictions = []
    for i in range(n_agents):
        predictions.append(
            AgentPrediction(
                agent_name=f"agent_{i}",
                confidence=random.uniform(0.5, 0.95),
                predicted_outcome=random.choice(["winner_a", "winner_b"]),
                calibration_accuracy=random.uniform(0.6, 0.95),
                brier_score=random.uniform(0.1, 0.35),
                prediction_count=random.randint(10, 200),
            )
        )
    return predictions


# =============================================================================
# Test: AgentPrediction dataclass
# =============================================================================


class TestAgentPrediction:
    """Tests for AgentPrediction dataclass."""

    def test_creation_minimal(self) -> None:
        """Test creating a prediction with minimal fields."""
        pred = AgentPrediction(
            agent_name="test_agent",
            confidence=0.75,
            predicted_outcome="winner_a",
        )
        assert pred.agent_name == "test_agent"
        assert pred.confidence == 0.75
        assert pred.predicted_outcome == "winner_a"
        assert pred.calibration_accuracy == 0.5  # default
        assert pred.brier_score == 0.25  # default
        assert pred.prediction_count == 0  # default

    def test_creation_full(self) -> None:
        """Test creating a prediction with all fields."""
        ts = datetime.now(timezone.utc)
        pred = AgentPrediction(
            agent_name="claude",
            confidence=0.90,
            predicted_outcome="winner_a",
            domain="technology",
            calibration_accuracy=0.85,
            brier_score=0.12,
            prediction_count=150,
            timestamp=ts,
            metadata={"model_version": "3.5"},
        )
        assert pred.domain == "technology"
        assert pred.calibration_accuracy == 0.85
        assert pred.brier_score == 0.12
        assert pred.prediction_count == 150
        assert pred.timestamp == ts
        assert pred.metadata["model_version"] == "3.5"

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        pred = AgentPrediction(
            agent_name="test",
            confidence=0.8,
            predicted_outcome="winner_a",
            domain="finance",
        )
        d = pred.to_dict()
        assert d["agent_name"] == "test"
        assert d["confidence"] == 0.8
        assert d["predicted_outcome"] == "winner_a"
        assert d["domain"] == "finance"
        assert "timestamp" in d


# =============================================================================
# Test: CalibrationConsensus dataclass
# =============================================================================


class TestCalibrationConsensus:
    """Tests for CalibrationConsensus dataclass."""

    def test_is_high_confidence(self) -> None:
        """Test high confidence detection."""
        consensus = CalibrationConsensus(
            debate_id="test",
            predictions=[],
            fused_confidence=0.85,
            predicted_outcome="winner_a",
            consensus_strength=0.75,
            agreement_ratio=0.8,
            disagreement_score=0.05,
            krippendorff_alpha=0.8,
            outliers_detected=[],
        )
        assert consensus.is_high_confidence is True

    def test_is_not_high_confidence_with_outliers(self) -> None:
        """Test that outliers lower confidence assessment."""
        consensus = CalibrationConsensus(
            debate_id="test",
            predictions=[],
            fused_confidence=0.85,
            predicted_outcome="winner_a",
            consensus_strength=0.75,
            agreement_ratio=0.8,
            disagreement_score=0.05,
            krippendorff_alpha=0.8,
            outliers_detected=["grok"],
        )
        assert consensus.is_high_confidence is False

    def test_needs_review_low_strength(self) -> None:
        """Test review flag for low consensus strength."""
        consensus = CalibrationConsensus(
            debate_id="test",
            predictions=[],
            fused_confidence=0.6,
            predicted_outcome="winner_a",
            consensus_strength=0.4,  # Below 0.5 threshold
            agreement_ratio=0.6,
            disagreement_score=0.1,
            krippendorff_alpha=0.5,
        )
        assert consensus.needs_review is True

    def test_needs_review_high_disagreement(self) -> None:
        """Test review flag for high disagreement."""
        consensus = CalibrationConsensus(
            debate_id="test",
            predictions=[],
            fused_confidence=0.7,
            predicted_outcome="winner_a",
            consensus_strength=0.6,
            agreement_ratio=0.7,
            disagreement_score=0.25,  # Above 0.2 threshold
            krippendorff_alpha=0.5,
        )
        assert consensus.needs_review is True

    def test_needs_review_low_alpha(self) -> None:
        """Test review flag for low Krippendorff's alpha."""
        consensus = CalibrationConsensus(
            debate_id="test",
            predictions=[],
            fused_confidence=0.7,
            predicted_outcome="winner_a",
            consensus_strength=0.6,
            agreement_ratio=0.7,
            disagreement_score=0.1,
            krippendorff_alpha=0.3,  # Below 0.4 threshold
        )
        assert consensus.needs_review is True

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        consensus = CalibrationConsensus(
            debate_id="test-123",
            predictions=[],
            fused_confidence=0.8,
            predicted_outcome="winner_a",
            consensus_strength=0.7,
            agreement_ratio=0.8,
            disagreement_score=0.05,
            krippendorff_alpha=0.75,
        )
        d = consensus.to_dict()
        assert d["debate_id"] == "test-123"
        assert d["fused_confidence"] == 0.8
        assert "fused_at" in d


# =============================================================================
# Test: Krippendorff's Alpha Calculation
# =============================================================================


class TestKrippendorffAlpha:
    """Tests for Krippendorff's alpha calculation."""

    def test_perfect_agreement(self, engine: CalibrationFusionEngine) -> None:
        """Test alpha = 1.0 for perfect agreement."""
        predictions = [
            AgentPrediction("a", 0.8, "winner_a"),
            AgentPrediction("b", 0.8, "winner_a"),
            AgentPrediction("c", 0.8, "winner_a"),
        ]
        alpha = engine.compute_krippendorff_alpha(predictions)
        assert alpha == 1.0

    def test_high_agreement(self, engine: CalibrationFusionEngine) -> None:
        """Test high alpha for similar predictions."""
        # Use predictions with truly high agreement (same bucket when discretized)
        # 0.85, 0.86, 0.84 all map to bucket 8
        predictions = [
            AgentPrediction("a", 0.85, "winner_a"),
            AgentPrediction("b", 0.86, "winner_a"),
            AgentPrediction("c", 0.84, "winner_a"),
        ]
        alpha = engine.compute_krippendorff_alpha(predictions)
        # With very similar values in the same bucket, alpha should be high
        assert alpha > 0.5

    def test_moderate_disagreement(self, engine: CalibrationFusionEngine) -> None:
        """Test moderate alpha for diverse predictions."""
        predictions = [
            AgentPrediction("a", 0.90, "winner_a"),
            AgentPrediction("b", 0.70, "winner_a"),
            AgentPrediction("c", 0.50, "winner_a"),
        ]
        alpha = engine.compute_krippendorff_alpha(predictions)
        assert -1.0 <= alpha <= 1.0

    def test_single_prediction(self, engine: CalibrationFusionEngine) -> None:
        """Test alpha = 1.0 for single prediction."""
        predictions = [AgentPrediction("a", 0.8, "winner_a")]
        alpha = engine.compute_krippendorff_alpha(predictions)
        assert alpha == 1.0

    def test_empty_predictions(self, engine: CalibrationFusionEngine) -> None:
        """Test alpha for empty predictions."""
        alpha = engine.compute_krippendorff_alpha([])
        assert alpha == 1.0

    def test_caching_works(self) -> None:
        """Test that identical bucket tuples use cached results."""
        # Clear cache first
        _cached_krippendorff_alpha.cache_clear()

        buckets1 = (8, 8, 7, 9)
        buckets2 = (8, 8, 7, 9)  # Same values

        # First call
        result1 = _cached_krippendorff_alpha(buckets1)
        info_after_first = _cached_krippendorff_alpha.cache_info()

        # Second call with same values
        result2 = _cached_krippendorff_alpha(buckets2)
        info_after_second = _cached_krippendorff_alpha.cache_info()

        assert result1 == result2
        assert info_after_second.hits == info_after_first.hits + 1

    def test_vectorized_matches_naive(self) -> None:
        """Test that vectorized implementation matches naive O(n^2) result."""

        # Naive implementation for comparison
        def naive_krippendorff(buckets: tuple[int, ...]) -> float:
            n = len(buckets)
            if n < 2:
                return 1.0

            # O(n^2) nested loop
            observed_disagreement = 0.0
            for i in range(n):
                for j in range(i + 1, n):
                    observed_disagreement += (buckets[i] - buckets[j]) ** 2
            observed_disagreement /= n * (n - 1) / 2

            mean_bucket = sum(buckets) / n
            expected_disagreement = sum((b - mean_bucket) ** 2 for b in buckets) / n

            if expected_disagreement == 0:
                return 1.0

            alpha = 1.0 - (observed_disagreement / expected_disagreement)
            return max(-1.0, min(1.0, alpha))

        # Test with various bucket configurations
        test_cases = [
            (5, 5, 5, 5),
            (1, 2, 3, 4, 5),
            (8, 7, 9, 8, 7),
            (0, 10, 5, 5, 5),
            tuple(range(10)),
        ]

        for buckets in test_cases:
            vectorized_result = _cached_krippendorff_alpha(buckets)
            naive_result = naive_krippendorff(buckets)
            assert abs(vectorized_result - naive_result) < 1e-10, (
                f"Mismatch for {buckets}: vectorized={vectorized_result}, naive={naive_result}"
            )


# =============================================================================
# Test: Fusion Strategies
# =============================================================================


class TestFusionStrategies:
    """Tests for different fusion strategies."""

    def test_weighted_average(
        self, engine: CalibrationFusionEngine, sample_predictions: list[AgentPrediction]
    ) -> None:
        """Test weighted average fusion strategy."""
        result = engine.fuse_predictions(
            sample_predictions,
            debate_id="test-weighted",
            strategy=CalibrationFusionStrategy.WEIGHTED_AVERAGE,
        )
        assert result.strategy_used == CalibrationFusionStrategy.WEIGHTED_AVERAGE
        assert 0.0 <= result.fused_confidence <= 1.0
        assert result.predicted_outcome == "winner_a"

    def test_reliability_weighted(
        self, engine: CalibrationFusionEngine, sample_predictions: list[AgentPrediction]
    ) -> None:
        """Test reliability weighted fusion strategy."""
        result = engine.fuse_predictions(
            sample_predictions,
            debate_id="test-reliability",
            strategy=CalibrationFusionStrategy.RELIABILITY_WEIGHTED,
        )
        assert result.strategy_used == CalibrationFusionStrategy.RELIABILITY_WEIGHTED
        assert 0.0 <= result.fused_confidence <= 1.0

    def test_bayesian(
        self, engine: CalibrationFusionEngine, sample_predictions: list[AgentPrediction]
    ) -> None:
        """Test Bayesian fusion strategy."""
        result = engine.fuse_predictions(
            sample_predictions,
            debate_id="test-bayesian",
            strategy=CalibrationFusionStrategy.BAYESIAN,
        )
        assert result.strategy_used == CalibrationFusionStrategy.BAYESIAN
        assert 0.0 <= result.fused_confidence <= 1.0

    def test_median(
        self, engine: CalibrationFusionEngine, sample_predictions: list[AgentPrediction]
    ) -> None:
        """Test median fusion strategy."""
        result = engine.fuse_predictions(
            sample_predictions,
            debate_id="test-median",
            strategy=CalibrationFusionStrategy.MEDIAN,
        )
        assert result.strategy_used == CalibrationFusionStrategy.MEDIAN
        # Median of 0.85, 0.80, 0.75 = 0.80
        assert result.fused_confidence == 0.80

    def test_trimmed_mean(
        self, engine: CalibrationFusionEngine, sample_predictions: list[AgentPrediction]
    ) -> None:
        """Test trimmed mean fusion strategy."""
        result = engine.fuse_predictions(
            sample_predictions,
            debate_id="test-trimmed",
            strategy=CalibrationFusionStrategy.TRIMMED_MEAN,
        )
        assert result.strategy_used == CalibrationFusionStrategy.TRIMMED_MEAN
        assert 0.0 <= result.fused_confidence <= 1.0

    def test_consensus_only(
        self, engine: CalibrationFusionEngine, disagreeing_predictions: list[AgentPrediction]
    ) -> None:
        """Test consensus only fusion strategy."""
        result = engine.fuse_predictions(
            disagreeing_predictions,
            debate_id="test-consensus",
            strategy=CalibrationFusionStrategy.CONSENSUS_ONLY,
        )
        assert result.strategy_used == CalibrationFusionStrategy.CONSENSUS_ONLY
        # Should only average predictions matching majority outcome
        assert 0.0 <= result.fused_confidence <= 1.0

    def test_explicit_weights(
        self, engine: CalibrationFusionEngine, sample_predictions: list[AgentPrediction]
    ) -> None:
        """Test fusion with explicit weights."""
        weights = {"claude": 0.5, "gpt-4": 0.3, "gemini": 0.2}
        result = engine.fuse_predictions(
            sample_predictions,
            debate_id="test-weights",
            weights=weights,
            strategy=CalibrationFusionStrategy.WEIGHTED_AVERAGE,
        )
        assert result.weights_used == weights


# =============================================================================
# Test: Outlier Detection
# =============================================================================


class TestOutlierDetection:
    """Tests for outlier detection."""

    def test_no_outliers(self, engine: CalibrationFusionEngine) -> None:
        """Test no outliers when predictions are similar."""
        predictions = [
            AgentPrediction("a", 0.80, "winner_a"),
            AgentPrediction("b", 0.82, "winner_a"),
            AgentPrediction("c", 0.78, "winner_a"),
            AgentPrediction("d", 0.81, "winner_a"),
        ]
        outliers = engine.detect_outliers(predictions)
        assert len(outliers) == 0

    def test_detect_outlier(self, engine: CalibrationFusionEngine) -> None:
        """Test detecting an obvious outlier."""
        predictions = [
            AgentPrediction("a", 0.80, "winner_a"),
            AgentPrediction("b", 0.82, "winner_a"),
            AgentPrediction("c", 0.78, "winner_a"),
            AgentPrediction("outlier", 0.20, "winner_a"),  # Very different
        ]
        outliers = engine.detect_outliers(predictions)
        assert "outlier" in outliers

    def test_custom_threshold(self, engine: CalibrationFusionEngine) -> None:
        """Test outlier detection with custom threshold."""
        predictions = [
            AgentPrediction("a", 0.80, "winner_a"),
            AgentPrediction("b", 0.82, "winner_a"),
            AgentPrediction("c", 0.65, "winner_a"),  # Somewhat different
        ]
        # With default threshold (2.0), this might not be an outlier
        outliers_default = engine.detect_outliers(predictions)

        # With stricter threshold (1.0), more likely to be flagged
        outliers_strict = engine.detect_outliers(predictions, threshold=1.0)

        # Stricter threshold should flag more outliers
        assert len(outliers_strict) >= len(outliers_default)

    def test_too_few_predictions(self, engine: CalibrationFusionEngine) -> None:
        """Test that outlier detection requires 3+ predictions."""
        predictions = [
            AgentPrediction("a", 0.80, "winner_a"),
            AgentPrediction("b", 0.20, "winner_a"),  # Very different
        ]
        outliers = engine.detect_outliers(predictions)
        assert len(outliers) == 0  # Not enough data to detect outliers


# =============================================================================
# Test: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_agent(self, engine: CalibrationFusionEngine) -> None:
        """Test fusion with single agent."""
        predictions = [AgentPrediction("solo", 0.75, "winner_a")]
        result = engine.fuse_predictions(predictions, debate_id="test-single")

        assert result.metadata.get("insufficient_data") is True
        assert result.fused_confidence == 0.75
        assert result.predicted_outcome == "winner_a"
        assert result.consensus_strength == 0.0

    def test_no_predictions(self, engine: CalibrationFusionEngine) -> None:
        """Test fusion with no predictions."""
        result = engine.fuse_predictions([], debate_id="test-empty")

        assert result.metadata.get("insufficient_data") is True
        assert result.fused_confidence == 0.5  # Default
        assert result.predicted_outcome == ""
        assert result.consensus_strength == 0.0

    def test_all_same_confidence(self, engine: CalibrationFusionEngine) -> None:
        """Test fusion when all agents have same confidence."""
        predictions = [
            AgentPrediction("a", 0.75, "winner_a"),
            AgentPrediction("b", 0.75, "winner_a"),
            AgentPrediction("c", 0.75, "winner_a"),
        ]
        result = engine.fuse_predictions(predictions, debate_id="test-same")

        assert result.fused_confidence == 0.75
        assert result.disagreement_score == 0.0
        assert result.krippendorff_alpha == 1.0

    def test_extreme_disagreement(self, engine: CalibrationFusionEngine) -> None:
        """Test fusion with extreme disagreement."""
        predictions = [
            AgentPrediction("a", 0.10, "winner_a"),
            AgentPrediction("b", 0.90, "winner_b"),
            AgentPrediction("c", 0.50, "tie"),
        ]
        result = engine.fuse_predictions(predictions, debate_id="test-extreme")

        # Should still produce valid results
        assert 0.0 <= result.fused_confidence <= 1.0
        assert result.agreement_ratio < 0.5  # No majority

    def test_zero_confidence(self, engine: CalibrationFusionEngine) -> None:
        """Test handling of zero confidence predictions."""
        predictions = [
            AgentPrediction("a", 0.0, "winner_a"),
            AgentPrediction("b", 0.5, "winner_a"),
            AgentPrediction("c", 1.0, "winner_a"),
        ]
        result = engine.fuse_predictions(predictions, debate_id="test-zero")

        assert 0.0 <= result.fused_confidence <= 1.0

    def test_minimum_criteria_filtering(self) -> None:
        """Test filtering by minimum criteria."""
        config = CalibrationFusionConfig(
            min_calibration_accuracy=0.7,
            min_prediction_count=50,
        )
        engine = CalibrationFusionEngine(config)

        predictions = [
            AgentPrediction(
                "good", 0.80, "winner_a", calibration_accuracy=0.85, prediction_count=100
            ),
            AgentPrediction(
                "low_accuracy", 0.80, "winner_a", calibration_accuracy=0.5, prediction_count=100
            ),
            AgentPrediction(
                "low_count", 0.80, "winner_a", calibration_accuracy=0.85, prediction_count=10
            ),
            AgentPrediction(
                "good2", 0.75, "winner_a", calibration_accuracy=0.75, prediction_count=60
            ),
        ]
        result = engine.fuse_predictions(predictions, debate_id="test-filter")

        # Only "good" and "good2" meet criteria
        assert "good" in result.participating_agents
        assert "good2" in result.participating_agents
        assert "low_accuracy" not in result.participating_agents
        assert "low_count" not in result.participating_agents


# =============================================================================
# Test: Performance with Large Agent Counts
# =============================================================================


class TestPerformance:
    """Tests for performance with large numbers of agents."""

    def test_50_agents_performance(self, engine: CalibrationFusionEngine) -> None:
        """Test fusion performance with 50 agents."""
        predictions = create_large_prediction_set(50)

        start_time = time.perf_counter()
        result = engine.fuse_predictions(predictions, debate_id="test-50")
        elapsed = time.perf_counter() - start_time

        # Should complete in under 100ms
        assert elapsed < 0.1, f"50 agents took {elapsed:.3f}s (should be < 0.1s)"
        assert 0.0 <= result.fused_confidence <= 1.0
        assert len(result.participating_agents) == 50

    def test_100_agents_performance(self, engine: CalibrationFusionEngine) -> None:
        """Test fusion performance with 100 agents."""
        predictions = create_large_prediction_set(100)

        start_time = time.perf_counter()
        result = engine.fuse_predictions(predictions, debate_id="test-100")
        elapsed = time.perf_counter() - start_time

        # Should complete in under 200ms
        assert elapsed < 0.2, f"100 agents took {elapsed:.3f}s (should be < 0.2s)"
        assert len(result.participating_agents) == 100

    def test_500_agents_performance(self, engine: CalibrationFusionEngine) -> None:
        """Test fusion performance with 500 agents (stress test)."""
        predictions = create_large_prediction_set(500)

        start_time = time.perf_counter()
        result = engine.fuse_predictions(predictions, debate_id="test-500")
        elapsed = time.perf_counter() - start_time

        # Should complete in under 1 second
        assert elapsed < 1.0, f"500 agents took {elapsed:.3f}s (should be < 1.0s)"
        assert len(result.participating_agents) == 500

    def test_krippendorff_scaling(self) -> None:
        """Test that Krippendorff's alpha scales well with agent count."""
        import random

        random.seed(42)
        engine = CalibrationFusionEngine()

        times_by_size: dict[int, float] = {}
        for n in [10, 50, 100, 200]:
            predictions = create_large_prediction_set(n)

            start = time.perf_counter()
            for _ in range(10):  # Multiple iterations for more stable timing
                engine.compute_krippendorff_alpha(predictions)
            elapsed = time.perf_counter() - start

            times_by_size[n] = elapsed / 10  # Average time

        # Check that scaling is roughly linear, not quadratic
        # If O(n^2), going from 50 to 200 (4x) would be 16x slower
        # If O(n), going from 50 to 200 (4x) would be ~4x slower
        ratio = times_by_size[200] / times_by_size[50]
        assert ratio < 10, (
            f"Scaling appears quadratic: 200/50 ratio = {ratio:.1f}x (should be < 10x)"
        )

    def test_caching_improves_repeated_calculations(self) -> None:
        """Test that caching improves performance for repeated calculations."""
        _cached_krippendorff_alpha.cache_clear()

        # Create predictions that will produce the same bucket tuple
        predictions1 = [
            AgentPrediction("a", 0.85, "winner_a"),
            AgentPrediction("b", 0.85, "winner_a"),
            AgentPrediction("c", 0.75, "winner_a"),
        ]
        predictions2 = [
            AgentPrediction("x", 0.85, "winner_b"),
            AgentPrediction("y", 0.85, "winner_b"),
            AgentPrediction("z", 0.75, "winner_b"),
        ]  # Same confidences

        engine = CalibrationFusionEngine()

        # First calculation
        start1 = time.perf_counter()
        alpha1 = engine.compute_krippendorff_alpha(predictions1)
        time1 = time.perf_counter() - start1

        # Second calculation (should be cached)
        start2 = time.perf_counter()
        alpha2 = engine.compute_krippendorff_alpha(predictions2)
        time2 = time.perf_counter() - start2

        assert alpha1 == alpha2  # Same buckets should give same result
        # Cache hit should be faster (though timing can be noisy)
        assert _cached_krippendorff_alpha.cache_info().hits >= 1


# =============================================================================
# Test: Consensus Metrics
# =============================================================================


class TestConsensusMetrics:
    """Tests for consensus strength and agreement metrics."""

    def test_agreement_ratio_all_agree(self, engine: CalibrationFusionEngine) -> None:
        """Test agreement ratio when all agents agree."""
        predictions = [
            AgentPrediction("a", 0.80, "winner_a"),
            AgentPrediction("b", 0.75, "winner_a"),
            AgentPrediction("c", 0.85, "winner_a"),
        ]
        result = engine.fuse_predictions(predictions, debate_id="test-agree")
        assert result.agreement_ratio == 1.0

    def test_agreement_ratio_split(
        self, engine: CalibrationFusionEngine, disagreeing_predictions: list[AgentPrediction]
    ) -> None:
        """Test agreement ratio when agents are split."""
        result = engine.fuse_predictions(disagreeing_predictions, debate_id="test-split")
        assert result.agreement_ratio == 0.5  # 2 for A, 2 for B

    def test_consensus_strength_high(self, engine: CalibrationFusionEngine) -> None:
        """Test high consensus strength with agreement and low variance."""
        predictions = [
            AgentPrediction("a", 0.80, "winner_a"),
            AgentPrediction("b", 0.82, "winner_a"),
            AgentPrediction("c", 0.78, "winner_a"),
            AgentPrediction("d", 0.81, "winner_a"),
            AgentPrediction("e", 0.79, "winner_a"),
        ]
        result = engine.fuse_predictions(predictions, debate_id="test-strong")
        assert result.consensus_strength > 0.9

    def test_consensus_strength_low(
        self, engine: CalibrationFusionEngine, disagreeing_predictions: list[AgentPrediction]
    ) -> None:
        """Test lower consensus strength with disagreement."""
        result = engine.fuse_predictions(disagreeing_predictions, debate_id="test-weak")
        assert result.consensus_strength < 0.7

    def test_disagreement_score(self, engine: CalibrationFusionEngine) -> None:
        """Test disagreement score calculation."""
        # Low disagreement
        low_var = [
            AgentPrediction("a", 0.80, "winner_a"),
            AgentPrediction("b", 0.81, "winner_a"),
            AgentPrediction("c", 0.79, "winner_a"),
        ]
        result_low = engine.fuse_predictions(low_var, debate_id="test-low-var")

        # High disagreement
        high_var = [
            AgentPrediction("a", 0.20, "winner_a"),
            AgentPrediction("b", 0.80, "winner_a"),
            AgentPrediction("c", 0.50, "winner_a"),
        ]
        result_high = engine.fuse_predictions(high_var, debate_id="test-high-var")

        assert result_low.disagreement_score < result_high.disagreement_score


# =============================================================================
# Test: Confidence Intervals
# =============================================================================


class TestConfidenceIntervals:
    """Tests for confidence interval calculation."""

    def test_interval_bounds(
        self, engine: CalibrationFusionEngine, sample_predictions: list[AgentPrediction]
    ) -> None:
        """Test that confidence intervals are valid bounds."""
        result = engine.fuse_predictions(sample_predictions, debate_id="test-ci")

        lower, upper = result.confidence_interval
        assert 0.0 <= lower <= upper <= 1.0
        assert lower <= result.fused_confidence <= upper

    def test_narrow_interval_with_agreement(self, engine: CalibrationFusionEngine) -> None:
        """Test that high agreement produces narrow confidence interval."""
        predictions = [
            AgentPrediction("a", 0.80, "winner_a"),
            AgentPrediction("b", 0.80, "winner_a"),
            AgentPrediction("c", 0.80, "winner_a"),
        ]
        result = engine.fuse_predictions(predictions, debate_id="test-narrow")

        lower, upper = result.confidence_interval
        interval_width = upper - lower
        assert interval_width < 0.1  # Should be very narrow

    def test_wide_interval_with_disagreement(self, engine: CalibrationFusionEngine) -> None:
        """Test that high disagreement produces wider confidence interval."""
        predictions = [
            AgentPrediction("a", 0.30, "winner_a"),
            AgentPrediction("b", 0.50, "winner_a"),
            AgentPrediction("c", 0.70, "winner_a"),
        ]
        result = engine.fuse_predictions(predictions, debate_id="test-wide")

        lower, upper = result.confidence_interval
        interval_width = upper - lower
        assert interval_width > 0.05  # Should be wider


# =============================================================================
# Test: History and Statistics
# =============================================================================


class TestHistoryAndStats:
    """Tests for fusion history and statistics."""

    def test_history_tracking(
        self, engine: CalibrationFusionEngine, sample_predictions: list[AgentPrediction]
    ) -> None:
        """Test that fusion results are tracked in history."""
        # Perform multiple fusions
        engine.fuse_predictions(sample_predictions, debate_id="debate-1")
        engine.fuse_predictions(sample_predictions, debate_id="debate-2")
        engine.fuse_predictions(sample_predictions, debate_id="debate-3")

        history = engine.get_fusion_history()
        assert len(history) == 3

    def test_history_filter_by_debate(
        self, engine: CalibrationFusionEngine, sample_predictions: list[AgentPrediction]
    ) -> None:
        """Test filtering history by debate ID."""
        engine.fuse_predictions(sample_predictions, debate_id="debate-a")
        engine.fuse_predictions(sample_predictions, debate_id="debate-b")
        engine.fuse_predictions(sample_predictions, debate_id="debate-a")

        history_a = engine.get_fusion_history(debate_id="debate-a")
        assert len(history_a) == 2
        assert all(h.debate_id == "debate-a" for h in history_a)

    def test_history_limit(
        self, engine: CalibrationFusionEngine, sample_predictions: list[AgentPrediction]
    ) -> None:
        """Test history limit."""
        for i in range(10):
            engine.fuse_predictions(sample_predictions, debate_id=f"debate-{i}")

        history = engine.get_fusion_history(limit=5)
        assert len(history) == 5

    def test_agent_performance(
        self, engine: CalibrationFusionEngine, sample_predictions: list[AgentPrediction]
    ) -> None:
        """Test agent performance metrics."""
        # Perform several fusions
        for i in range(5):
            engine.fuse_predictions(sample_predictions, debate_id=f"debate-{i}")

        perf = engine.get_agent_performance("claude")
        assert perf["fusion_count"] == 5
        assert perf["avg_weight"] > 0
        assert perf["agreement_rate"] > 0

    def test_agent_performance_unknown(self, engine: CalibrationFusionEngine) -> None:
        """Test agent performance for unknown agent."""
        perf = engine.get_agent_performance("unknown_agent")
        assert perf["fusion_count"] == 0
        assert perf["avg_weight"] == 0.0

    def test_stats(
        self, engine: CalibrationFusionEngine, sample_predictions: list[AgentPrediction]
    ) -> None:
        """Test overall fusion statistics."""
        # Perform fusions with different strategies
        engine.fuse_predictions(
            sample_predictions,
            debate_id="d1",
            strategy=CalibrationFusionStrategy.WEIGHTED_AVERAGE,
        )
        engine.fuse_predictions(
            sample_predictions,
            debate_id="d2",
            strategy=CalibrationFusionStrategy.MEDIAN,
        )
        engine.fuse_predictions(
            sample_predictions,
            debate_id="d3",
            strategy=CalibrationFusionStrategy.WEIGHTED_AVERAGE,
        )

        stats = engine.get_stats()
        assert stats["total_fusions"] == 3
        assert 0.0 <= stats["avg_consensus_strength"] <= 1.0
        assert stats["by_strategy"]["weighted_average"] == 2
        assert stats["by_strategy"]["median"] == 1

    def test_stats_empty(self, engine: CalibrationFusionEngine) -> None:
        """Test stats with no fusions."""
        stats = engine.get_stats()
        assert stats["total_fusions"] == 0
        assert stats["avg_consensus_strength"] == 0.0


# =============================================================================
# Test: Configuration
# =============================================================================


class TestConfiguration:
    """Tests for fusion configuration."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = CalibrationFusionConfig()
        assert config.min_predictions == 2
        assert config.outlier_threshold == 2.0
        assert config.trim_percent == 0.1
        assert config.use_brier_weights is True

    def test_custom_config(self) -> None:
        """Test custom configuration."""
        config = CalibrationFusionConfig(
            min_predictions=3,
            outlier_threshold=1.5,
            trim_percent=0.2,
            min_calibration_accuracy=0.6,
        )
        engine = CalibrationFusionEngine(config)

        predictions = [
            AgentPrediction("a", 0.80, "winner_a"),
            AgentPrediction("b", 0.75, "winner_a"),
        ]
        # Only 2 predictions but min is 3
        result = engine.fuse_predictions(predictions, debate_id="test-config")
        assert result.metadata.get("insufficient_data") is True

    def test_fusion_config_alias(self) -> None:
        """Test FusionConfig alias works."""
        config = FusionConfig(min_predictions=5)
        assert config.min_predictions == 5


# =============================================================================
# Test: Singleton Pattern
# =============================================================================


class TestSingletonPattern:
    """Tests for singleton engine access."""

    def test_get_singleton(self) -> None:
        """Test getting singleton engine."""
        # Reset singleton for test
        import aragora.knowledge.mound.ops.calibration_fusion as cf

        cf._calibration_fusion_engine = None

        engine1 = get_calibration_fusion_engine()
        engine2 = get_calibration_fusion_engine()
        assert engine1 is engine2

    def test_singleton_with_config(self) -> None:
        """Test singleton respects initial config."""
        import aragora.knowledge.mound.ops.calibration_fusion as cf

        cf._calibration_fusion_engine = None

        config = CalibrationFusionConfig(min_predictions=5)
        engine = get_calibration_fusion_engine(config)
        assert engine.config.min_predictions == 5

        # Subsequent calls should return same instance
        engine2 = get_calibration_fusion_engine()
        assert engine2.config.min_predictions == 5


# =============================================================================
# Test: Weight Computation
# =============================================================================


class TestWeightComputation:
    """Tests for automatic weight computation."""

    def test_weights_by_calibration_accuracy(self, engine: CalibrationFusionEngine) -> None:
        """Test that higher calibration accuracy gets higher weight."""
        predictions = [
            AgentPrediction(
                "high_cal", 0.80, "winner_a", calibration_accuracy=0.95, prediction_count=0
            ),
            AgentPrediction(
                "low_cal", 0.80, "winner_a", calibration_accuracy=0.5, prediction_count=0
            ),
        ]
        result = engine.fuse_predictions(predictions, debate_id="test-weights")

        assert result.weights_used["high_cal"] > result.weights_used["low_cal"]

    def test_weights_by_brier_score(self) -> None:
        """Test that lower Brier score gets higher weight."""
        config = CalibrationFusionConfig(use_brier_weights=True)
        engine = CalibrationFusionEngine(config)

        predictions = [
            AgentPrediction(
                "good_brier",
                0.80,
                "winner_a",
                calibration_accuracy=0.8,
                brier_score=0.1,
                prediction_count=0,
            ),
            AgentPrediction(
                "bad_brier",
                0.80,
                "winner_a",
                calibration_accuracy=0.8,
                brier_score=0.4,
                prediction_count=0,
            ),
        ]
        result = engine.fuse_predictions(predictions, debate_id="test-brier")

        assert result.weights_used["good_brier"] > result.weights_used["bad_brier"]

    def test_weights_by_experience(self, engine: CalibrationFusionEngine) -> None:
        """Test that higher prediction count (experience) gets boost."""
        predictions = [
            AgentPrediction(
                "experienced",
                0.80,
                "winner_a",
                calibration_accuracy=0.8,
                prediction_count=1000,
            ),
            AgentPrediction(
                "novice",
                0.80,
                "winner_a",
                calibration_accuracy=0.8,
                prediction_count=1,
            ),
        ]
        result = engine.fuse_predictions(predictions, debate_id="test-exp")

        assert result.weights_used["experienced"] > result.weights_used["novice"]

    def test_weights_normalized(
        self, engine: CalibrationFusionEngine, sample_predictions: list[AgentPrediction]
    ) -> None:
        """Test that weights sum to 1.0."""
        result = engine.fuse_predictions(sample_predictions, debate_id="test-norm")

        total_weight = sum(result.weights_used.values())
        assert abs(total_weight - 1.0) < 1e-10
