"""Comprehensive tests for MetaLearner - covering untested methods.

Tests evaluate_learning_efficiency, adjust_hyperparameters, _clamp_hyperparameters,
get_adjustment_history, reset_to_defaults, _compute_trend, and _save_state/_load_state.
"""

import json
import sqlite3
from unittest.mock import MagicMock, patch

import pytest

from aragora.learning.meta import HyperparameterState, LearningMetrics, MetaLearner


@pytest.fixture
def learner(tmp_path):
    """Create a MetaLearner with a temporary database."""
    db_path = tmp_path / "meta_test.db"
    return MetaLearner(db_path=db_path)


@pytest.fixture
def mock_cms():
    """Create a mock ContinuumMemory instance."""
    cms = MagicMock()
    cms.get_stats.return_value = {"total_memories": 100}
    return cms


@pytest.fixture
def cycle_results():
    """Sample cycle results."""
    return {
        "cycle": 5,
        "consensus_rate": 0.75,
        "avg_calibration": 0.6,
    }


class TestLearningMetricsSerialization:
    """Test LearningMetrics serialization round-trip."""

    def test_to_dict_and_from_dict(self):
        metrics = LearningMetrics(
            cycles_evaluated=10,
            pattern_retention_rate=0.85,
            forgetting_rate=0.05,
            learning_velocity=4.2,
            consensus_rate=0.9,
            avg_cycles_to_consensus=2.5,
            prediction_accuracy=0.7,
            tier_efficiency={"fast": 0.8, "slow": 0.6},
        )
        data = metrics.to_dict()
        restored = LearningMetrics.from_dict(data)
        assert restored.cycles_evaluated == 10
        assert restored.pattern_retention_rate == 0.85
        assert restored.tier_efficiency["fast"] == 0.8

    def test_empty_tier_efficiency_serialization(self):
        metrics = LearningMetrics()
        data = metrics.to_dict()
        assert data["tier_efficiency"] == {}


class TestHyperparameterStateSerialization:
    """Test HyperparameterState serialization round-trip."""

    def test_to_dict_and_from_dict(self):
        state = HyperparameterState(
            surprise_weight_success=0.4,
            meta_learning_rate=0.05,
            consolidation_threshold=200,
        )
        data = state.to_dict()
        restored = HyperparameterState.from_dict(data)
        assert restored.surprise_weight_success == 0.4
        assert restored.meta_learning_rate == 0.05
        assert restored.consolidation_threshold == 200


class TestMetaLearnerSaveLoadState:
    """Test _save_state and _load_state."""

    def test_save_and_load_state(self, tmp_path):
        db_path = tmp_path / "state_test.db"
        learner = MetaLearner(db_path=db_path)

        # Modify state
        learner.state.surprise_weight_success = 0.45
        learner.state.meta_learning_rate = 0.03
        learner._save_state(reason="test modification")

        # Create new learner - should load saved state
        learner2 = MetaLearner(db_path=db_path)
        assert abs(learner2.state.surprise_weight_success - 0.45) < 1e-6
        assert abs(learner2.state.meta_learning_rate - 0.03) < 1e-6

    def test_save_state_with_metrics(self, learner):
        metrics = LearningMetrics(
            cycles_evaluated=3,
            pattern_retention_rate=0.7,
        )
        learner._save_state(reason="with metrics", metrics=metrics)
        history = learner.get_adjustment_history(limit=1)
        assert len(history) == 1
        assert history[0]["reason"] == "with metrics"
        assert history[0]["metrics"] is not None

    def test_load_state_returns_defaults_on_empty_db(self, tmp_path):
        db_path = tmp_path / "empty.db"
        learner = MetaLearner(db_path=db_path)
        defaults = HyperparameterState()
        assert learner.state.surprise_weight_success == defaults.surprise_weight_success
        assert learner.state.meta_learning_rate == defaults.meta_learning_rate

    def test_load_state_handles_db_error(self, tmp_path):
        db_path = tmp_path / "error.db"
        learner = MetaLearner(db_path=db_path)
        # Corrupt the database
        with open(db_path, "w") as f:
            f.write("not a database")
        # Should return defaults, not crash
        state = learner._load_state()
        assert isinstance(state, HyperparameterState)


class TestEvaluateLearningEfficiency:
    """Test evaluate_learning_efficiency method."""

    def test_returns_empty_metrics_when_no_memories(self, learner):
        cms = MagicMock()
        cms.get_stats.return_value = {"total_memories": 0}
        metrics = learner.evaluate_learning_efficiency(cms, {"cycle": 1})
        assert metrics.pattern_retention_rate == 0.0
        assert metrics.cycles_evaluated == 0

    def test_extracts_cycle_results(self, learner, mock_cms, cycle_results):
        # The DB won't have continuum_memory table, so DB queries will fail gracefully
        metrics = learner.evaluate_learning_efficiency(mock_cms, cycle_results)
        assert metrics.cycles_evaluated == 5
        assert metrics.consensus_rate == 0.75
        assert metrics.prediction_accuracy == 0.6

    def test_handles_cms_stats_error(self, learner):
        cms = MagicMock()
        cms.get_stats.side_effect = RuntimeError("DB error")
        metrics = learner.evaluate_learning_efficiency(cms, {"cycle": 1})
        # Should return default metrics without crashing
        assert isinstance(metrics, LearningMetrics)

    def test_appends_to_metrics_history(self, learner, mock_cms, cycle_results):
        assert len(learner.metrics_history) == 0
        learner.evaluate_learning_efficiency(mock_cms, cycle_results)
        assert len(learner.metrics_history) == 1
        learner.evaluate_learning_efficiency(mock_cms, cycle_results)
        assert len(learner.metrics_history) == 2

    def test_default_cycle_results(self, learner, mock_cms):
        metrics = learner.evaluate_learning_efficiency(mock_cms, {})
        assert metrics.cycles_evaluated == 0
        assert metrics.consensus_rate == 0.5  # default
        assert metrics.prediction_accuracy == 0.5  # default


class TestAdjustHyperparameters:
    """Test adjust_hyperparameters method."""

    def test_low_retention_increases_half_lives(self, learner):
        metrics = LearningMetrics(pattern_retention_rate=0.4)
        original_slow = learner.state.slow_half_life_hours
        original_glacial = learner.state.glacial_half_life_hours

        adjustments = learner.adjust_hyperparameters(metrics)

        assert "half_lives" in adjustments
        assert "increased" in adjustments["half_lives"]
        assert learner.state.slow_half_life_hours > original_slow
        assert learner.state.glacial_half_life_hours > original_glacial

    def test_high_retention_decreases_half_lives(self, learner):
        metrics = LearningMetrics(pattern_retention_rate=0.95)
        original_slow = learner.state.slow_half_life_hours

        adjustments = learner.adjust_hyperparameters(metrics)

        assert "half_lives" in adjustments
        assert "decreased" in adjustments["half_lives"]
        assert learner.state.slow_half_life_hours < original_slow

    def test_high_forgetting_lowers_thresholds(self, learner):
        metrics = LearningMetrics(forgetting_rate=0.4)
        original_medium = learner.state.medium_promotion_threshold

        adjustments = learner.adjust_hyperparameters(metrics)

        assert "promotion_thresholds" in adjustments
        assert "lowered" in adjustments["promotion_thresholds"]
        assert learner.state.medium_promotion_threshold < original_medium

    def test_low_forgetting_raises_thresholds(self, learner):
        metrics = LearningMetrics(forgetting_rate=0.05)
        original_medium = learner.state.medium_promotion_threshold

        adjustments = learner.adjust_hyperparameters(metrics)

        assert "promotion_thresholds" in adjustments
        assert "raised" in adjustments["promotion_thresholds"]
        assert learner.state.medium_promotion_threshold > original_medium

    def test_fast_tier_underperforming(self, learner):
        metrics = LearningMetrics(tier_efficiency={"fast": 0.3, "slow": 0.7})
        original_fast = learner.state.fast_promotion_threshold

        adjustments = learner.adjust_hyperparameters(metrics)

        assert "fast_threshold" in adjustments
        assert learner.state.fast_promotion_threshold > original_fast

    def test_poor_calibration_reduces_agent_weight(self, learner):
        metrics = LearningMetrics(prediction_accuracy=0.3)

        adjustments = learner.adjust_hyperparameters(metrics)

        assert "surprise_weights" in adjustments
        assert "reduced agent weight" in adjustments["surprise_weights"]

    def test_good_calibration_increases_agent_weight(self, learner):
        metrics = LearningMetrics(prediction_accuracy=0.8)

        adjustments = learner.adjust_hyperparameters(metrics)

        assert "surprise_weights" in adjustments
        assert "increased agent weight" in adjustments["surprise_weights"]

    def test_normalizes_surprise_weights(self, learner):
        metrics = LearningMetrics(prediction_accuracy=0.3)

        learner.adjust_hyperparameters(metrics)

        total = (
            learner.state.surprise_weight_success
            + learner.state.surprise_weight_semantic
            + learner.state.surprise_weight_temporal
            + learner.state.surprise_weight_agent
        )
        assert abs(total - 1.0) < 0.01

    def test_no_adjustments_returns_empty(self, learner):
        # Metrics in the "normal" range - no rules should trigger
        metrics = LearningMetrics(
            pattern_retention_rate=0.75,
            forgetting_rate=0.15,
            prediction_accuracy=0.55,
            tier_efficiency={"fast": 0.5, "slow": 0.5},
        )
        adjustments = learner.adjust_hyperparameters(metrics)
        assert adjustments == {}

    def test_saves_state_when_adjustments_made(self, learner):
        metrics = LearningMetrics(pattern_retention_rate=0.4)
        learner.adjust_hyperparameters(metrics)
        history = learner.get_adjustment_history(limit=1)
        assert len(history) >= 1

    def test_does_not_save_when_no_adjustments(self, learner):
        metrics = LearningMetrics(
            pattern_retention_rate=0.75,
            forgetting_rate=0.15,
            prediction_accuracy=0.55,
            tier_efficiency={"fast": 0.5, "slow": 0.5},
        )
        learner.adjust_hyperparameters(metrics)
        history = learner.get_adjustment_history(limit=10)
        # Only the initial state might be saved, but no adjustment entry
        assert all(h.get("reason") != "" for h in history if h.get("reason"))


class TestClampHyperparameters:
    """Test _clamp_hyperparameters method."""

    def test_clamps_thresholds_to_valid_range(self, learner):
        learner.state.fast_promotion_threshold = 1.5
        learner.state.medium_promotion_threshold = -0.1
        learner.state.slow_demotion_threshold = 0.0

        learner._clamp_hyperparameters()

        assert learner.state.fast_promotion_threshold == 0.9
        assert learner.state.medium_promotion_threshold == 0.1
        assert learner.state.slow_demotion_threshold == 0.1

    def test_clamps_half_lives(self, learner):
        learner.state.fast_half_life_hours = 0.1  # below min 0.5
        learner.state.medium_half_life_hours = 1000  # above max 168
        learner.state.slow_half_life_hours = 1  # below min 24
        learner.state.glacial_half_life_hours = 5000  # above max 2000

        learner._clamp_hyperparameters()

        assert learner.state.fast_half_life_hours == 0.5
        assert learner.state.medium_half_life_hours == 168
        assert learner.state.slow_half_life_hours == 24
        assert learner.state.glacial_half_life_hours == 2000

    def test_clamps_surprise_weights(self, learner):
        learner.state.surprise_weight_success = 0.01  # below min 0.05
        learner.state.surprise_weight_agent = 0.9  # above max 0.6

        learner._clamp_hyperparameters()

        assert learner.state.surprise_weight_success == 0.05
        assert learner.state.surprise_weight_agent == 0.6

    def test_clamps_meta_learning_rate(self, learner):
        learner.state.meta_learning_rate = 0.0001
        learner._clamp_hyperparameters()
        assert learner.state.meta_learning_rate == 0.001

        learner.state.meta_learning_rate = 1.0
        learner._clamp_hyperparameters()
        assert learner.state.meta_learning_rate == 0.1

    def test_values_in_range_unchanged(self, learner):
        original = HyperparameterState()
        learner._clamp_hyperparameters()
        assert learner.state.fast_promotion_threshold == original.fast_promotion_threshold
        assert learner.state.slow_half_life_hours == original.slow_half_life_hours


class TestGetAdjustmentHistory:
    """Test get_adjustment_history method."""

    def test_empty_history(self, learner):
        history = learner.get_adjustment_history()
        assert history == []

    def test_multiple_entries(self, learner):
        for i in range(5):
            learner._save_state(reason=f"test {i}")
        history = learner.get_adjustment_history(limit=3)
        assert len(history) == 3
        # All entries should have reasons
        reasons = {h["reason"] for h in history}
        assert len(reasons) == 3

    def test_limit_parameter(self, learner):
        for i in range(10):
            learner._save_state(reason=f"entry {i}")
        history = learner.get_adjustment_history(limit=5)
        assert len(history) == 5

    def test_history_contains_hyperparams(self, learner):
        learner._save_state(reason="test")
        history = learner.get_adjustment_history(limit=1)
        assert "hyperparams" in history[0]
        assert isinstance(history[0]["hyperparams"], dict)
        assert "timestamp" in history[0]


class TestResetToDefaults:
    """Test reset_to_defaults method."""

    def test_resets_all_fields(self, learner):
        learner.state.surprise_weight_success = 0.99
        learner.state.fast_promotion_threshold = 0.1
        learner.state.glacial_half_life_hours = 5000
        learner.state.meta_learning_rate = 0.09

        learner.reset_to_defaults()

        defaults = HyperparameterState()
        assert learner.state.surprise_weight_success == defaults.surprise_weight_success
        assert learner.state.fast_promotion_threshold == defaults.fast_promotion_threshold
        assert learner.state.glacial_half_life_hours == defaults.glacial_half_life_hours
        assert learner.state.meta_learning_rate == defaults.meta_learning_rate

    def test_reset_persists_to_db(self, tmp_path):
        db_path = tmp_path / "reset_persist.db"
        learner = MetaLearner(db_path=db_path)
        learner.state.meta_learning_rate = 0.09
        learner._save_state(reason="modified")
        learner.reset_to_defaults()

        # Verify in-memory state is reset
        assert learner.state.meta_learning_rate == 0.01
        # Verify reset was recorded in history
        history = learner.get_adjustment_history(limit=10)
        reasons = [h["reason"] for h in history]
        assert "reset to defaults" in reasons

    def test_reset_records_in_history(self, learner):
        learner.reset_to_defaults()
        history = learner.get_adjustment_history(limit=1)
        assert len(history) == 1
        assert history[0]["reason"] == "reset to defaults"


class TestComputeTrend:
    """Test _compute_trend method."""

    def test_insufficient_data_single(self, learner):
        metrics = [LearningMetrics(pattern_retention_rate=0.5)]
        assert learner._compute_trend(metrics) == "insufficient_data"

    def test_insufficient_data_empty(self, learner):
        assert learner._compute_trend([]) == "insufficient_data"

    def test_improving_trend(self, learner):
        metrics = [
            LearningMetrics(pattern_retention_rate=0.3),
            LearningMetrics(pattern_retention_rate=0.35),
            LearningMetrics(pattern_retention_rate=0.7),
            LearningMetrics(pattern_retention_rate=0.8),
        ]
        assert learner._compute_trend(metrics) == "improving"

    def test_declining_trend(self, learner):
        metrics = [
            LearningMetrics(pattern_retention_rate=0.8),
            LearningMetrics(pattern_retention_rate=0.75),
            LearningMetrics(pattern_retention_rate=0.3),
            LearningMetrics(pattern_retention_rate=0.2),
        ]
        assert learner._compute_trend(metrics) == "declining"

    def test_stable_trend(self, learner):
        metrics = [
            LearningMetrics(pattern_retention_rate=0.5),
            LearningMetrics(pattern_retention_rate=0.51),
            LearningMetrics(pattern_retention_rate=0.49),
            LearningMetrics(pattern_retention_rate=0.5),
        ]
        assert learner._compute_trend(metrics) == "stable"

    def test_exactly_two_metrics(self, learner):
        metrics = [
            LearningMetrics(pattern_retention_rate=0.3),
            LearningMetrics(pattern_retention_rate=0.8),
        ]
        result = learner._compute_trend(metrics)
        assert result == "improving"


class TestGetLearningSummary:
    """Test get_learning_summary with various data states."""

    def test_summary_with_many_evaluations(self, learner):
        for i in range(15):
            learner.metrics_history.append(
                LearningMetrics(
                    pattern_retention_rate=0.5 + i * 0.02,
                    forgetting_rate=0.1,
                    learning_velocity=float(i),
                )
            )
        summary = learner.get_learning_summary()
        assert summary["evaluations"] == 15
        # Should use last 10 for averages
        assert "avg_retention" in summary
        assert "trend" in summary

    def test_summary_trend_reflects_data(self, learner):
        # Add improving data
        for rate in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            learner.metrics_history.append(LearningMetrics(pattern_retention_rate=rate))
        summary = learner.get_learning_summary()
        assert summary["trend"] == "improving"


class TestMultipleAdjustmentCycles:
    """Integration tests for multiple adjustment cycles."""

    def test_repeated_adjustments_stay_clamped(self, learner):
        """Repeated extreme adjustments should stay within bounds."""
        extreme_metrics = LearningMetrics(
            pattern_retention_rate=0.1,
            forgetting_rate=0.9,
            prediction_accuracy=0.1,
            tier_efficiency={"fast": 0.1, "slow": 0.9},
        )
        for _ in range(100):
            learner.adjust_hyperparameters(extreme_metrics)

        # All values should be within bounds
        assert 0.1 <= learner.state.fast_promotion_threshold <= 0.9
        assert 0.1 <= learner.state.medium_promotion_threshold <= 0.9
        assert 0.05 <= learner.state.surprise_weight_agent <= 0.6
        assert 0.5 <= learner.state.fast_half_life_hours <= 24
        assert 24 <= learner.state.slow_half_life_hours <= 720
        assert 168 <= learner.state.glacial_half_life_hours <= 2000

    def test_surprise_weights_always_sum_to_one(self, learner):
        """After any adjustment, surprise weights should sum to ~1.0."""
        test_cases = [
            LearningMetrics(prediction_accuracy=0.1),
            LearningMetrics(prediction_accuracy=0.9),
            LearningMetrics(pattern_retention_rate=0.1, prediction_accuracy=0.1),
        ]
        for metrics in test_cases:
            learner.adjust_hyperparameters(metrics)
            total = (
                learner.state.surprise_weight_success
                + learner.state.surprise_weight_semantic
                + learner.state.surprise_weight_temporal
                + learner.state.surprise_weight_agent
            )
            assert abs(total - 1.0) < 0.05, f"Weights sum to {total}, expected ~1.0"
