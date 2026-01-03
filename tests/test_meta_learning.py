"""
Tests for MetaLearner self-tuning hyperparameters.

Tests the outer optimization loop that:
- Evaluates learning efficiency
- Adjusts hyperparameters based on metrics
- Tracks adjustment history
"""

import os
import tempfile
import pytest

from aragora.memory.continuum import ContinuumMemory, MemoryTier
from aragora.learning.meta import (
    MetaLearner,
    LearningMetrics,
    HyperparameterState,
)


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    os.unlink(path)


@pytest.fixture
def cms(temp_db):
    """Create a ContinuumMemory instance."""
    return ContinuumMemory(db_path=temp_db)


@pytest.fixture
def meta(temp_db):
    """Create a MetaLearner instance."""
    return MetaLearner(db_path=temp_db)


class TestHyperparameterState:
    """Test hyperparameter state management."""

    def test_default_state(self):
        """Test default hyperparameter values."""
        state = HyperparameterState()

        assert state.meta_learning_rate == 0.01
        assert state.surprise_weight_success == 0.3
        assert state.consolidation_threshold == 100

    def test_to_dict(self):
        """Test serialization to dict."""
        state = HyperparameterState()
        d = state.to_dict()

        assert "surprise_weight_success" in d
        assert "meta_learning_rate" in d
        assert d["fast_half_life_hours"] == 1.0

    def test_from_dict(self):
        """Test deserialization from dict."""
        data = {
            "surprise_weight_success": 0.4,
            "meta_learning_rate": 0.02,
            "consolidation_threshold": 50,
        }
        state = HyperparameterState.from_dict(data)

        assert state.surprise_weight_success == 0.4
        assert state.meta_learning_rate == 0.02
        assert state.consolidation_threshold == 50


class TestLearningMetrics:
    """Test learning metrics calculations."""

    def test_metrics_to_dict(self):
        """Test metrics serialization."""
        metrics = LearningMetrics(
            cycles_evaluated=10,
            pattern_retention_rate=0.8,
            forgetting_rate=0.1,
        )
        d = metrics.to_dict()

        assert d["cycles_evaluated"] == 10
        assert d["pattern_retention_rate"] == 0.8
        assert "tier_efficiency" in d


class TestMetaLearnerBasics:
    """Test basic MetaLearner operations."""

    def test_initialization(self, meta):
        """Test MetaLearner initializes with default state."""
        assert meta.state is not None
        assert meta.state.meta_learning_rate == 0.01

    def test_get_current_hyperparams(self, meta):
        """Test getting current hyperparameters for CMS."""
        params = meta.get_current_hyperparams()

        assert "surprise_weight_success" in params
        assert "consolidation_threshold" in params
        assert "promotion_cooldown_hours" in params

    def test_reset_to_defaults(self, meta):
        """Test resetting hyperparameters to defaults."""
        # Modify state
        meta.state.meta_learning_rate = 0.5

        # Reset
        meta.reset_to_defaults()

        assert meta.state.meta_learning_rate == 0.01


class TestLearningEfficiencyEvaluation:
    """Test learning efficiency evaluation."""

    def test_evaluate_empty_cms(self, cms, meta):
        """Test evaluation with empty CMS returns default metrics."""
        cycle_results = {"cycle": 1, "consensus_rate": 0.5}
        metrics = meta.evaluate_learning_efficiency(cms, cycle_results)

        # Empty CMS returns default metrics (early return)
        assert metrics.pattern_retention_rate == 0.0
        assert metrics.forgetting_rate == 0.0

    def test_evaluate_with_patterns(self, cms, meta):
        """Test evaluation with some patterns."""
        # Add patterns with varying success
        cms.add(id="good_1", content="Good pattern", tier=MemoryTier.SLOW)
        cms.update_outcome("good_1", success=True)
        cms.update_outcome("good_1", success=True)

        cms.add(id="bad_1", content="Bad pattern", tier=MemoryTier.SLOW)
        cms.update_outcome("bad_1", success=False)
        cms.update_outcome("bad_1", success=False)

        cycle_results = {"cycle": 5, "consensus_rate": 0.8, "avg_calibration": 0.6}
        metrics = meta.evaluate_learning_efficiency(cms, cycle_results)

        assert metrics.cycles_evaluated == 5
        assert metrics.consensus_rate == 0.8
        assert metrics.prediction_accuracy == 0.6

    def test_metrics_stored_in_history(self, cms, meta):
        """Test that metrics are stored in history when CMS has data."""
        # Add some data so evaluation doesn't early return
        cms.add(id="history_test", content="Test pattern", tier=MemoryTier.SLOW)

        cycle_results = {"cycle": 1}
        meta.evaluate_learning_efficiency(cms, cycle_results)

        assert len(meta.metrics_history) == 1


class TestHyperparameterAdjustment:
    """Test hyperparameter self-modification."""

    def test_low_retention_increases_half_lives(self, meta):
        """Test that low retention increases decay half-lives."""
        original_half_life = meta.state.slow_half_life_hours

        metrics = LearningMetrics(pattern_retention_rate=0.4)  # Low retention
        meta.adjust_hyperparameters(metrics)

        assert meta.state.slow_half_life_hours > original_half_life

    def test_high_forgetting_lowers_thresholds(self, meta):
        """Test that high forgetting lowers promotion thresholds."""
        original_threshold = meta.state.medium_promotion_threshold

        metrics = LearningMetrics(forgetting_rate=0.5)  # High forgetting
        meta.adjust_hyperparameters(metrics)

        assert meta.state.medium_promotion_threshold < original_threshold

    def test_poor_calibration_reduces_agent_weight(self, meta):
        """Test that poor agent calibration reduces agent weight."""
        original_weight = meta.state.surprise_weight_agent

        metrics = LearningMetrics(prediction_accuracy=0.3)  # Poor calibration
        meta.adjust_hyperparameters(metrics)

        assert meta.state.surprise_weight_agent < original_weight

    def test_good_calibration_increases_agent_weight(self, meta):
        """Test that good agent calibration increases agent weight."""
        original_weight = meta.state.surprise_weight_agent

        metrics = LearningMetrics(prediction_accuracy=0.8)  # Good calibration
        meta.adjust_hyperparameters(metrics)

        assert meta.state.surprise_weight_agent > original_weight

    def test_weights_normalized_to_one(self, meta):
        """Test that surprise weights are normalized to sum to 1."""
        metrics = LearningMetrics(prediction_accuracy=0.8)
        meta.adjust_hyperparameters(metrics)

        total = (
            meta.state.surprise_weight_success +
            meta.state.surprise_weight_semantic +
            meta.state.surprise_weight_temporal +
            meta.state.surprise_weight_agent
        )
        assert abs(total - 1.0) < 0.01  # Allow small floating point error


class TestHyperparameterClamping:
    """Test that hyperparameters stay in valid ranges."""

    def test_threshold_clamping(self, meta):
        """Test that thresholds are clamped to valid range."""
        meta.state.fast_promotion_threshold = 2.0  # Invalid high
        meta._clamp_hyperparameters()
        assert meta.state.fast_promotion_threshold <= 0.9

        meta.state.medium_demotion_threshold = -0.5  # Invalid low
        meta._clamp_hyperparameters()
        assert meta.state.medium_demotion_threshold >= 0.1

    def test_half_life_clamping(self, meta):
        """Test that half-lives are clamped to valid range."""
        meta.state.fast_half_life_hours = 0.1  # Too short
        meta._clamp_hyperparameters()
        assert meta.state.fast_half_life_hours >= 0.5

        meta.state.glacial_half_life_hours = 5000  # Too long
        meta._clamp_hyperparameters()
        assert meta.state.glacial_half_life_hours <= 2000

    def test_weight_clamping(self, meta):
        """Test that weights are clamped to valid range."""
        meta.state.surprise_weight_success = 1.5  # Too high
        meta._clamp_hyperparameters()
        assert meta.state.surprise_weight_success <= 0.6


class TestAdjustmentHistory:
    """Test adjustment history tracking."""

    def test_adjustments_saved(self, meta):
        """Test that adjustments are saved to database."""
        metrics = LearningMetrics(pattern_retention_rate=0.4)
        meta.adjust_hyperparameters(metrics)

        history = meta.get_adjustment_history()
        assert len(history) >= 1

    def test_history_contains_reason(self, meta):
        """Test that history contains adjustment reason."""
        metrics = LearningMetrics(pattern_retention_rate=0.4)
        meta.adjust_hyperparameters(metrics)

        history = meta.get_adjustment_history()
        if history:
            assert "reason" in history[0]


class TestLearningSummary:
    """Test learning summary generation."""

    def test_summary_no_data(self, meta):
        """Test summary with no evaluation data."""
        summary = meta.get_learning_summary()
        assert summary["status"] == "no data"

    def test_summary_with_data(self, meta, cms):
        """Test summary with evaluation data."""
        # Add data so evaluations don't early return
        cms.add(id="summary_test", content="Test pattern", tier=MemoryTier.SLOW)

        for i in range(5):
            cycle_results = {"cycle": i + 1}
            meta.evaluate_learning_efficiency(cms, cycle_results)

        summary = meta.get_learning_summary()

        assert "evaluations" in summary
        assert summary["evaluations"] == 5
        assert "current_hyperparams" in summary
        assert "trend" in summary


class TestTrend:
    """Test learning trend computation."""

    def test_improving_trend(self, meta):
        """Test detection of improving trend."""
        # Add metrics with improving retention
        for i in range(10):
            meta.metrics_history.append(
                LearningMetrics(pattern_retention_rate=0.5 + i * 0.05)
            )

        trend = meta._compute_trend(meta.metrics_history)
        assert trend == "improving"

    def test_declining_trend(self, meta):
        """Test detection of declining trend."""
        # Add metrics with declining retention
        for i in range(10):
            meta.metrics_history.append(
                LearningMetrics(pattern_retention_rate=0.9 - i * 0.05)
            )

        trend = meta._compute_trend(meta.metrics_history)
        assert trend == "declining"

    def test_stable_trend(self, meta):
        """Test detection of stable trend."""
        # Add metrics with stable retention
        for i in range(10):
            meta.metrics_history.append(
                LearningMetrics(pattern_retention_rate=0.7)
            )

        trend = meta._compute_trend(meta.metrics_history)
        assert trend == "stable"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
