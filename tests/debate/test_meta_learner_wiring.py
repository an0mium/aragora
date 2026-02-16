"""Tests for MetaLearner wiring between Arena and FeedbackPhase."""
import pytest
from unittest.mock import MagicMock, patch


class TestMetaLearnerWiring:
    """Verify MetaLearner is properly passed from Arena to FeedbackPhase."""

    def test_feedback_phase_receives_meta_learner(self):
        """FeedbackPhase should receive meta_learner from arena_phases."""
        from aragora.debate.phases.feedback_phase import FeedbackPhase

        mock_learner = MagicMock()
        phase = FeedbackPhase(meta_learner=mock_learner)
        assert phase.meta_learner is mock_learner

    def test_feedback_phase_meta_learner_none_by_default(self):
        """FeedbackPhase should have None meta_learner by default."""
        from aragora.debate.phases.feedback_phase import FeedbackPhase

        phase = FeedbackPhase()
        assert phase.meta_learner is None

    def test_arena_phases_passes_meta_learner(self):
        """_create_feedback_phase should pass meta_learner from arena."""
        from aragora.debate.arena_phases import init_phases

        mock_arena = MagicMock()
        mock_arena.meta_learner = MagicMock()
        mock_arena.extensions.broadcast_pipeline = None
        mock_arena.extensions.auto_broadcast = False
        mock_arena.extensions.broadcast_min_confidence = 0.8
        mock_arena.extensions.training_exporter = None
        mock_arena.extensions.cost_tracker = None
        mock_arena.auto_evolve = False
        mock_arena.population_manager = None

        init_phases(mock_arena)
        assert mock_arena.feedback_phase.meta_learner is mock_arena.meta_learner

    def test_arena_phases_no_meta_learner_attr(self):
        """init_phases handles arena without meta_learner gracefully."""
        from aragora.debate.arena_phases import init_phases

        mock_arena = MagicMock()
        # Remove meta_learner so getattr falls back to None
        del mock_arena.meta_learner
        mock_arena.extensions.broadcast_pipeline = None
        mock_arena.extensions.auto_broadcast = False
        mock_arena.extensions.broadcast_min_confidence = 0.8
        mock_arena.extensions.training_exporter = None
        mock_arena.extensions.cost_tracker = None
        mock_arena.auto_evolve = False
        mock_arena.population_manager = None

        init_phases(mock_arena)
        assert mock_arena.feedback_phase.meta_learner is None

    def test_meta_learner_evaluate_called_in_feedback(self):
        """_evaluate_meta_learning should call meta_learner methods."""
        from aragora.debate.phases.feedback_phase import FeedbackPhase

        mock_learner = MagicMock()
        mock_learner.evaluate_learning_efficiency.return_value = MagicMock()
        mock_learner.adjust_hyperparameters.return_value = {"test": "value"}

        phase = FeedbackPhase(meta_learner=mock_learner, enable_meta_learning=True)

        mock_ctx = MagicMock()
        mock_ctx.result = MagicMock()
        mock_ctx.result.consensus_reached = True
        mock_ctx.result.confidence = 0.8
        mock_ctx.result.rounds_completed = 2
        mock_ctx.protocol = MagicMock()
        mock_ctx.protocol.rounds = 3
        mock_ctx.agents = []

        phase._evaluate_meta_learning(mock_ctx)
        mock_learner.evaluate_learning_efficiency.assert_called_once()

    def test_meta_learning_disabled_skips(self):
        """When enable_meta_learning=False, skip evaluation."""
        from aragora.debate.phases.feedback_phase import FeedbackPhase

        mock_learner = MagicMock()
        phase = FeedbackPhase(meta_learner=mock_learner, enable_meta_learning=False)

        mock_ctx = MagicMock()
        mock_ctx.result = MagicMock()
        phase._evaluate_meta_learning(mock_ctx)
        mock_learner.evaluate_learning_efficiency.assert_not_called()
