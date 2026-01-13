"""
Tests for aragora.learning.meta module.

Tests MetaLearner, HyperparameterState, LearningMetrics, and self-tuning behavior.
"""

import json
import pytest
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from aragora.learning.meta import (
    LearningMetrics,
    HyperparameterState,
    MetaLearner,
)


# ============================================================================
# LearningMetrics Tests
# ============================================================================


class TestLearningMetrics:
    """Tests for LearningMetrics dataclass."""

    def test_defaults(self):
        """Should have sensible defaults."""
        metrics = LearningMetrics()

        assert metrics.cycles_evaluated == 0
        assert metrics.pattern_retention_rate == 0.0
        assert metrics.forgetting_rate == 0.0
        assert metrics.learning_velocity == 0.0
        assert metrics.tier_efficiency == {}

    def test_to_dict(self):
        """Should serialize all fields."""
        metrics = LearningMetrics(
            cycles_evaluated=10,
            pattern_retention_rate=0.75,
            forgetting_rate=0.15,
            learning_velocity=5.0,
            consensus_rate=0.8,
            avg_cycles_to_consensus=2.5,
            prediction_accuracy=0.7,
            tier_efficiency={"fast": 0.6, "slow": 0.8},
        )
        d = metrics.to_dict()

        assert d["cycles_evaluated"] == 10
        assert d["pattern_retention_rate"] == 0.75
        assert d["forgetting_rate"] == 0.15
        assert d["tier_efficiency"]["fast"] == 0.6


# ============================================================================
# HyperparameterState Tests
# ============================================================================


class TestHyperparameterState:
    """Tests for HyperparameterState dataclass."""

    def test_defaults(self):
        """Should have sensible default values."""
        state = HyperparameterState()

        # Surprise weights should sum to 1.0
        total = (
            state.surprise_weight_success
            + state.surprise_weight_semantic
            + state.surprise_weight_temporal
            + state.surprise_weight_agent
        )
        assert abs(total - 1.0) < 0.01

        # Tier thresholds should be in valid range
        assert 0 < state.fast_promotion_threshold < 1
        assert 0 < state.medium_promotion_threshold < 1

    def test_to_dict(self):
        """Should serialize all fields."""
        state = HyperparameterState()
        d = state.to_dict()

        assert "surprise_weight_success" in d
        assert "fast_promotion_threshold" in d
        assert "fast_half_life_hours" in d
        assert "meta_learning_rate" in d

    def test_from_dict(self):
        """Should deserialize from dict."""
        original = HyperparameterState(
            surprise_weight_success=0.4,
            fast_promotion_threshold=0.8,
        )
        d = original.to_dict()
        restored = HyperparameterState.from_dict(d)

        assert restored.surprise_weight_success == 0.4
        assert restored.fast_promotion_threshold == 0.8

    def test_from_dict_ignores_unknown(self):
        """Should ignore unknown keys."""
        d = {"unknown_key": "value", "surprise_weight_success": 0.5}
        state = HyperparameterState.from_dict(d)

        assert state.surprise_weight_success == 0.5


# ============================================================================
# MetaLearner Tests
# ============================================================================


class TestMetaLearner:
    """Tests for MetaLearner class."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            yield f.name
        Path(f.name).unlink(missing_ok=True)

    @pytest.fixture
    def meta_learner(self, temp_db):
        """Create a MetaLearner with temp database."""
        return MetaLearner(db_path=temp_db)

    def test_init_creates_tables(self, meta_learner, temp_db):
        """Should create required tables on init."""
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='meta_hyperparams'"
        )
        assert cursor.fetchone() is not None

        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='meta_efficiency_log'"
        )
        assert cursor.fetchone() is not None

        conn.close()

    def test_init_loads_default_state(self, meta_learner):
        """Should start with default hyperparameters."""
        state = meta_learner.state

        assert isinstance(state, HyperparameterState)
        assert state.meta_learning_rate == 0.01

    def test_get_current_hyperparams(self, meta_learner):
        """Should return hyperparameters dict."""
        params = meta_learner.get_current_hyperparams()

        assert "surprise_weight_success" in params
        assert "consolidation_threshold" in params
        assert "promotion_cooldown_hours" in params

    def test_save_and_load_state(self, temp_db):
        """Should persist and restore state."""
        # Create learner and modify state
        learner1 = MetaLearner(db_path=temp_db)
        learner1.state.fast_promotion_threshold = 0.85
        learner1._save_state(reason="test")

        # Create new learner, should load saved state
        learner2 = MetaLearner(db_path=temp_db)

        assert learner2.state.fast_promotion_threshold == 0.85


class TestMetaLearnerAdjustments:
    """Tests for MetaLearner hyperparameter adjustments."""

    @pytest.fixture
    def temp_db(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            yield f.name
        Path(f.name).unlink(missing_ok=True)

    @pytest.fixture
    def meta_learner(self, temp_db):
        return MetaLearner(db_path=temp_db)

    def test_adjust_on_low_retention(self, meta_learner):
        """Should increase half-lives when retention is low."""
        original_slow = meta_learner.state.slow_half_life_hours
        original_glacial = meta_learner.state.glacial_half_life_hours

        metrics = LearningMetrics(pattern_retention_rate=0.4)
        adjustments = meta_learner.adjust_hyperparameters(metrics)

        assert meta_learner.state.slow_half_life_hours > original_slow
        assert meta_learner.state.glacial_half_life_hours > original_glacial
        assert "half_lives" in adjustments

    def test_adjust_on_high_retention(self, meta_learner):
        """Should decrease half-lives when retention is very high."""
        original_slow = meta_learner.state.slow_half_life_hours

        metrics = LearningMetrics(pattern_retention_rate=0.95)
        adjustments = meta_learner.adjust_hyperparameters(metrics)

        assert meta_learner.state.slow_half_life_hours < original_slow
        assert "half_lives" in adjustments

    def test_adjust_on_high_forgetting(self, meta_learner):
        """Should lower promotion thresholds on high forgetting."""
        original_medium = meta_learner.state.medium_promotion_threshold

        metrics = LearningMetrics(forgetting_rate=0.4)
        adjustments = meta_learner.adjust_hyperparameters(metrics)

        assert meta_learner.state.medium_promotion_threshold < original_medium
        assert "promotion_thresholds" in adjustments

    def test_adjust_on_low_forgetting(self, meta_learner):
        """Should raise promotion thresholds on low forgetting."""
        original_medium = meta_learner.state.medium_promotion_threshold

        metrics = LearningMetrics(forgetting_rate=0.05)
        adjustments = meta_learner.adjust_hyperparameters(metrics)

        assert meta_learner.state.medium_promotion_threshold > original_medium
        assert "promotion_thresholds" in adjustments

    def test_adjust_tier_imbalance(self, meta_learner):
        """Should adjust when fast tier underperforms slow."""
        metrics = LearningMetrics(tier_efficiency={"fast": 0.4, "slow": 0.7})
        original_fast = meta_learner.state.fast_promotion_threshold

        adjustments = meta_learner.adjust_hyperparameters(metrics)

        assert meta_learner.state.fast_promotion_threshold > original_fast
        assert "fast_threshold" in adjustments

    def test_adjust_on_poor_calibration(self, meta_learner):
        """Should reduce agent weight on poor prediction accuracy."""
        original_agent_weight = meta_learner.state.surprise_weight_agent

        metrics = LearningMetrics(prediction_accuracy=0.3)
        adjustments = meta_learner.adjust_hyperparameters(metrics)

        assert meta_learner.state.surprise_weight_agent < original_agent_weight
        assert "surprise_weights" in adjustments

    def test_adjust_on_good_calibration(self, meta_learner):
        """Should increase agent weight on good prediction accuracy."""
        original_agent_weight = meta_learner.state.surprise_weight_agent

        metrics = LearningMetrics(prediction_accuracy=0.8)
        adjustments = meta_learner.adjust_hyperparameters(metrics)

        assert meta_learner.state.surprise_weight_agent > original_agent_weight

    def test_weights_stay_normalized(self, meta_learner):
        """Surprise weights should sum to ~1.0 after adjustments."""
        metrics = LearningMetrics(prediction_accuracy=0.3)
        meta_learner.adjust_hyperparameters(metrics)

        total = (
            meta_learner.state.surprise_weight_success
            + meta_learner.state.surprise_weight_semantic
            + meta_learner.state.surprise_weight_temporal
            + meta_learner.state.surprise_weight_agent
        )

        assert abs(total - 1.0) < 0.01

    def test_clamp_prevents_invalid_values(self, meta_learner):
        """Should clamp hyperparameters to valid ranges."""
        # Set extreme values
        meta_learner.state.fast_promotion_threshold = 2.0
        meta_learner.state.fast_half_life_hours = 0.0
        meta_learner.state.surprise_weight_success = 1.5

        meta_learner._clamp_hyperparameters()

        assert 0.1 <= meta_learner.state.fast_promotion_threshold <= 0.9
        assert meta_learner.state.fast_half_life_hours >= 0.5
        assert meta_learner.state.surprise_weight_success <= 0.6


class TestMetaLearnerEvaluation:
    """Tests for MetaLearner learning efficiency evaluation."""

    @pytest.fixture
    def temp_db(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            yield f.name
        Path(f.name).unlink(missing_ok=True)

    @pytest.fixture
    def meta_learner(self, temp_db):
        return MetaLearner(db_path=temp_db)

    def test_evaluate_empty_cms(self, meta_learner):
        """Should handle CMS with no memories."""
        mock_cms = MagicMock()
        mock_cms.get_stats.return_value = {"total_memories": 0}

        metrics = meta_learner.evaluate_learning_efficiency(
            mock_cms,
            {"cycle": 1},
        )

        # With 0 total memories, function returns early with default metrics
        assert metrics.cycles_evaluated == 0
        assert metrics.pattern_retention_rate == 0.0

    def test_evaluate_stores_in_history(self, meta_learner, temp_db):
        """Should store metrics in history when there are memories."""
        # Create continuum_memory table needed by evaluate_learning_efficiency
        conn = sqlite3.connect(temp_db)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS continuum_memory (
                id INTEGER PRIMARY KEY,
                success_count INTEGER DEFAULT 0,
                failure_count INTEGER DEFAULT 0,
                update_count INTEGER DEFAULT 0,
                tier TEXT DEFAULT 'fast',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        conn.commit()
        conn.close()

        mock_cms = MagicMock()
        mock_cms.get_stats.return_value = {"total_memories": 10}

        meta_learner.evaluate_learning_efficiency(mock_cms, {"cycle": 1})

        assert len(meta_learner.metrics_history) == 1

    def test_evaluate_logs_to_database(self, meta_learner, temp_db):
        """Should log metrics to database when there are memories."""
        # Create continuum_memory table needed by evaluate_learning_efficiency
        conn = sqlite3.connect(temp_db)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS continuum_memory (
                id INTEGER PRIMARY KEY,
                success_count INTEGER DEFAULT 0,
                failure_count INTEGER DEFAULT 0,
                update_count INTEGER DEFAULT 0,
                tier TEXT DEFAULT 'fast',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        conn.commit()
        conn.close()

        mock_cms = MagicMock()
        mock_cms.get_stats.return_value = {"total_memories": 10}

        meta_learner.evaluate_learning_efficiency(mock_cms, {"cycle": 5})

        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute("SELECT cycle_number, metrics FROM meta_efficiency_log")
        row = cursor.fetchone()
        conn.close()

        assert row is not None
        assert row[0] == 5


class TestMetaLearnerHistory:
    """Tests for MetaLearner history and summary functions."""

    @pytest.fixture
    def temp_db(self):
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            yield f.name
        Path(f.name).unlink(missing_ok=True)

    @pytest.fixture
    def meta_learner(self, temp_db):
        return MetaLearner(db_path=temp_db)

    def test_get_adjustment_history_empty(self, meta_learner):
        """Should return empty list when no history."""
        history = meta_learner.get_adjustment_history()
        assert history == []

    def test_get_adjustment_history_with_data(self, meta_learner):
        """Should return adjustment history."""
        meta_learner._save_state(reason="test adjustment 1")
        meta_learner._save_state(reason="test adjustment 2")

        history = meta_learner.get_adjustment_history(limit=10)

        assert len(history) == 2
        # Both entries should be present
        reasons = [h["reason"] for h in history]
        assert "test adjustment 1" in reasons
        assert "test adjustment 2" in reasons
        assert "hyperparams" in history[0]

    def test_get_adjustment_history_respects_limit(self, meta_learner):
        """Should respect limit parameter."""
        for i in range(5):
            meta_learner._save_state(reason=f"adjustment {i}")

        history = meta_learner.get_adjustment_history(limit=3)

        assert len(history) == 3

    def test_reset_to_defaults(self, meta_learner):
        """Should reset hyperparameters to defaults."""
        meta_learner.state.fast_promotion_threshold = 0.99
        meta_learner.reset_to_defaults()

        default = HyperparameterState()
        assert meta_learner.state.fast_promotion_threshold == default.fast_promotion_threshold

    def test_get_learning_summary_no_data(self, meta_learner):
        """Should indicate no data when empty."""
        summary = meta_learner.get_learning_summary()

        assert summary["status"] == "no data"

    def test_get_learning_summary_with_data(self, meta_learner):
        """Should compute summary from history."""
        # Add some metrics
        for i in range(5):
            meta_learner.metrics_history.append(
                LearningMetrics(
                    pattern_retention_rate=0.6 + i * 0.05,
                    forgetting_rate=0.2 - i * 0.02,
                    learning_velocity=i + 1,
                )
            )

        summary = meta_learner.get_learning_summary()

        assert summary["evaluations"] == 5
        assert summary["avg_retention"] > 0
        assert summary["avg_forgetting"] > 0
        assert "current_hyperparams" in summary

    def test_compute_trend_improving(self, meta_learner):
        """Should detect improving trend."""
        # First half: low retention
        for _ in range(3):
            meta_learner.metrics_history.append(LearningMetrics(pattern_retention_rate=0.5))
        # Second half: high retention
        for _ in range(3):
            meta_learner.metrics_history.append(LearningMetrics(pattern_retention_rate=0.8))

        trend = meta_learner._compute_trend(meta_learner.metrics_history)

        assert trend == "improving"

    def test_compute_trend_declining(self, meta_learner):
        """Should detect declining trend."""
        # First half: high retention
        for _ in range(3):
            meta_learner.metrics_history.append(LearningMetrics(pattern_retention_rate=0.8))
        # Second half: low retention
        for _ in range(3):
            meta_learner.metrics_history.append(LearningMetrics(pattern_retention_rate=0.5))

        trend = meta_learner._compute_trend(meta_learner.metrics_history)

        assert trend == "declining"

    def test_compute_trend_stable(self, meta_learner):
        """Should detect stable trend."""
        for _ in range(6):
            meta_learner.metrics_history.append(LearningMetrics(pattern_retention_rate=0.7))

        trend = meta_learner._compute_trend(meta_learner.metrics_history)

        assert trend == "stable"

    def test_compute_trend_insufficient_data(self, meta_learner):
        """Should indicate insufficient data."""
        meta_learner.metrics_history.append(LearningMetrics(pattern_retention_rate=0.7))

        trend = meta_learner._compute_trend(meta_learner.metrics_history)

        assert trend == "insufficient_data"
