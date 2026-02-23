"""Tests for MetaLearner debate-level tuning."""

import pytest
from aragora.learning.meta import HyperparameterState, LearningMetrics, MetaLearner


class TestHyperparameterStateDebateFields:
    """Verify new debate-level fields on HyperparameterState."""

    def test_debate_round_adjustment_default(self):
        state = HyperparameterState()
        assert state.debate_round_adjustment == 0.0

    def test_consensus_threshold_adjustment_default(self):
        state = HyperparameterState()
        assert state.consensus_threshold_adjustment == 0.0

    def test_agent_diversity_weight_default(self):
        state = HyperparameterState()
        assert state.agent_diversity_weight == 0.5

    def test_backward_compat_serialization(self):
        """Old serialized state without new fields should still deserialize."""
        old_data = {"surprise_weight_success": 0.3}
        state = HyperparameterState.from_dict(old_data)
        assert state.debate_round_adjustment == 0.0
        assert state.agent_diversity_weight == 0.5


class TestMetaLearnerDebateTuning:
    """Test debate-level tuning in MetaLearner."""

    @pytest.fixture
    def learner(self, tmp_path):
        return MetaLearner(db_path=tmp_path / "test_meta.db")

    def test_get_debate_tuning_returns_dict(self, learner):
        tuning = learner.get_debate_tuning()
        assert "round_adjustment" in tuning
        assert "consensus_threshold_adjustment" in tuning
        assert "diversity_weight" in tuning

    def test_get_debate_tuning_defaults(self, learner):
        tuning = learner.get_debate_tuning()
        assert tuning["round_adjustment"] == 0.0
        assert tuning["consensus_threshold_adjustment"] == 0.0
        assert tuning["diversity_weight"] == 0.5

    def test_low_efficiency_increases_round_adjustment(self, learner):
        """Low debate efficiency should increase round adjustment."""
        metrics = LearningMetrics(consensus_rate=0.3, prediction_accuracy=0.5)
        learner._last_cycle_results = {"debate_efficiency": 0.3, "avg_confidence": 0.5}
        learner.adjust_hyperparameters(metrics)
        assert learner.state.debate_round_adjustment > 0.0

    def test_high_efficiency_decreases_round_adjustment(self, learner):
        """High debate efficiency should decrease round adjustment."""
        # First make it positive
        learner.state.debate_round_adjustment = 0.1
        metrics = LearningMetrics(consensus_rate=0.8, prediction_accuracy=0.5)
        learner._last_cycle_results = {"debate_efficiency": 0.9, "avg_confidence": 0.5}
        learner.adjust_hyperparameters(metrics)
        assert learner.state.debate_round_adjustment < 0.1

    def test_overconfidence_tightens_threshold(self, learner):
        """High confidence + low consensus should tighten threshold."""
        metrics = LearningMetrics(consensus_rate=0.3, prediction_accuracy=0.5)
        learner._last_cycle_results = {"debate_efficiency": 0.5, "avg_confidence": 0.9}
        learner.adjust_hyperparameters(metrics)
        assert learner.state.consensus_threshold_adjustment > 0.0

    def test_no_debate_metrics_no_change(self, learner):
        """Without debate metrics, debate tuning params should stay at defaults."""
        initial_round_adj = learner.state.debate_round_adjustment
        metrics = LearningMetrics(consensus_rate=0.5, prediction_accuracy=0.5)
        learner._last_cycle_results = {}
        learner.adjust_hyperparameters(metrics)
        assert learner.state.debate_round_adjustment == initial_round_adj
