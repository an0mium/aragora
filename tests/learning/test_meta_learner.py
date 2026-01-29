"""Tests for the MetaLearner, LearningMetrics, and HyperparameterState classes."""

import pytest

from aragora.learning.meta import HyperparameterState, LearningMetrics, MetaLearner


class TestImports:
    """Verify that key classes are importable."""

    def test_import_meta_learner(self):
        assert MetaLearner is not None

    def test_import_learning_metrics(self):
        assert LearningMetrics is not None

    def test_import_hyperparameter_state(self):
        assert HyperparameterState is not None


class TestLearningMetrics:
    """Tests for LearningMetrics dataclass."""

    def test_default_instantiation(self):
        metrics = LearningMetrics()
        assert metrics.cycles_evaluated == 0
        assert metrics.pattern_retention_rate == 0.0
        assert metrics.forgetting_rate == 0.0
        assert metrics.learning_velocity == 0.0
        assert metrics.consensus_rate == 0.0
        assert metrics.avg_cycles_to_consensus == 0.0
        assert metrics.prediction_accuracy == 0.0
        assert metrics.tier_efficiency == {}

    def test_custom_instantiation(self):
        metrics = LearningMetrics(
            cycles_evaluated=5,
            pattern_retention_rate=0.8,
            forgetting_rate=0.1,
            learning_velocity=3.0,
            consensus_rate=0.75,
            prediction_accuracy=0.6,
            tier_efficiency={"fast": 0.9, "slow": 0.5},
        )
        assert metrics.cycles_evaluated == 5
        assert metrics.pattern_retention_rate == 0.8
        assert metrics.tier_efficiency["fast"] == 0.9


class TestHyperparameterState:
    """Tests for HyperparameterState dataclass."""

    def test_default_instantiation(self):
        state = HyperparameterState()
        assert state.surprise_weight_success == 0.3
        assert state.surprise_weight_semantic == 0.3
        assert state.surprise_weight_temporal == 0.2
        assert state.surprise_weight_agent == 0.2
        assert state.fast_promotion_threshold == 0.7
        assert state.consolidation_threshold == 100
        assert state.meta_learning_rate == 0.01

    def test_custom_instantiation(self):
        state = HyperparameterState(
            surprise_weight_success=0.5,
            meta_learning_rate=0.05,
        )
        assert state.surprise_weight_success == 0.5
        assert state.meta_learning_rate == 0.05


class TestMetaLearner:
    """Tests for MetaLearner."""

    def test_instantiation(self, tmp_path):
        db_path = tmp_path / "meta.db"
        learner = MetaLearner(db_path=db_path)
        assert learner is not None
        assert isinstance(learner.state, HyperparameterState)

    def test_get_current_hyperparams(self, tmp_path):
        db_path = tmp_path / "meta.db"
        learner = MetaLearner(db_path=db_path)
        params = learner.get_current_hyperparams()
        assert isinstance(params, dict)
        assert "surprise_weight_success" in params
        assert "surprise_weight_semantic" in params
        assert "surprise_weight_temporal" in params
        assert "surprise_weight_agent" in params
        assert "consolidation_threshold" in params
        assert "promotion_cooldown_hours" in params

    def test_get_learning_summary_no_data(self, tmp_path):
        db_path = tmp_path / "meta.db"
        learner = MetaLearner(db_path=db_path)
        summary = learner.get_learning_summary()
        assert summary == {"status": "no data"}

    def test_get_learning_summary_with_data(self, tmp_path):
        db_path = tmp_path / "meta.db"
        learner = MetaLearner(db_path=db_path)
        learner.metrics_history = [
            LearningMetrics(pattern_retention_rate=0.7, forgetting_rate=0.1, learning_velocity=2.0),
            LearningMetrics(
                pattern_retention_rate=0.8, forgetting_rate=0.05, learning_velocity=3.0
            ),
        ]
        summary = learner.get_learning_summary()
        assert "evaluations" in summary
        assert summary["evaluations"] == 2
        assert "avg_retention" in summary
        assert "avg_forgetting" in summary
        assert "avg_learning_velocity" in summary
        assert "current_hyperparams" in summary
        assert "trend" in summary

    def test_reset_to_defaults(self, tmp_path):
        db_path = tmp_path / "meta.db"
        learner = MetaLearner(db_path=db_path)
        # Modify state
        learner.state.meta_learning_rate = 0.09
        learner.state.surprise_weight_success = 0.5
        # Reset
        learner.reset_to_defaults()
        assert learner.state.meta_learning_rate == 0.01
        assert learner.state.surprise_weight_success == 0.3

    def test_get_adjustment_history(self, tmp_path):
        db_path = tmp_path / "meta.db"
        learner = MetaLearner(db_path=db_path)
        history = learner.get_adjustment_history()
        assert isinstance(history, list)

    def test_get_adjustment_history_after_save(self, tmp_path):
        db_path = tmp_path / "meta.db"
        learner = MetaLearner(db_path=db_path)
        learner._save_state(reason="test save")
        history = learner.get_adjustment_history()
        assert len(history) >= 1
        assert history[0]["reason"] == "test save"
