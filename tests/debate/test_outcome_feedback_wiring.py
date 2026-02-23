"""Tests for OutcomeFeedbackBridge wiring into PostDebateCoordinator."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aragora.debate.post_debate_coordinator import (
    PostDebateConfig,
    PostDebateCoordinator,
    PostDebateResult,
)


class TestOutcomeFeedbackConfig:
    """Tests for auto_outcome_feedback config flag."""

    def test_default_enabled(self):
        config = PostDebateConfig()
        assert config.auto_outcome_feedback is True

    def test_can_enable(self):
        config = PostDebateConfig(auto_outcome_feedback=True)
        assert config.auto_outcome_feedback is True


class TestOutcomeFeedbackResult:
    """Tests for outcome_feedback field on PostDebateResult."""

    def test_default_none(self):
        result = PostDebateResult()
        assert result.outcome_feedback is None

    def test_can_store_result(self):
        result = PostDebateResult()
        result.outcome_feedback = {
            "goals_generated": 2,
            "suggestions_queued": 1,
            "trickster_adjustment": 0.95,
            "domains_flagged": ["security"],
            "agents_flagged": ["claude"],
        }
        assert result.outcome_feedback["goals_generated"] == 2
        assert result.outcome_feedback["domains_flagged"] == ["security"]


class TestOutcomeFeedbackStep:
    """Tests for _step_outcome_feedback method."""

    def _make_coordinator(self, **config_kwargs):
        defaults = {
            "auto_outcome_feedback": True,
            "auto_explain": False,
            "auto_create_plan": False,
            "auto_notify": False,
            "auto_persist_receipt": False,
            "auto_gauntlet_validate": False,
            "auto_execution_bridge": False,
            "auto_push_calibration": False,
        }
        defaults.update(config_kwargs)
        config = PostDebateConfig(**defaults)
        return PostDebateCoordinator(config=config)

    def _make_debate_result(self):
        result = MagicMock()
        result.messages = []
        result.final_answer = "Use modular monolith"
        result.confidence = 0.9
        return result

    def test_step_runs_when_enabled(self):
        """When auto_outcome_feedback is True, the step executes."""
        coordinator = self._make_coordinator()
        mock_result = self._make_debate_result()

        cycle_result = {
            "goals_generated": 3,
            "suggestions_queued": 2,
            "trickster_adjustment": 0.9,
            "domains_flagged": ["security", "legal"],
            "agents_flagged": ["claude", "gpt4"],
        }

        with patch(
            "aragora.debate.post_debate_coordinator.PostDebateCoordinator._step_outcome_feedback",
            return_value=cycle_result,
        ):
            result = coordinator.run("d1", mock_result, confidence=0.9, task="test")

        assert result.outcome_feedback is not None
        assert result.outcome_feedback["goals_generated"] == 3
        assert result.outcome_feedback["suggestions_queued"] == 2

    def test_step_skipped_when_disabled(self):
        """When auto_outcome_feedback is False, the step is skipped."""
        config = PostDebateConfig(
            auto_outcome_feedback=False,
            auto_explain=False,
            auto_create_plan=False,
            auto_notify=False,
            auto_persist_receipt=False,
            auto_gauntlet_validate=False,
            auto_execution_bridge=False,
            auto_push_calibration=False,
        )
        coordinator = PostDebateCoordinator(config=config)
        mock_result = self._make_debate_result()

        with patch.object(coordinator, "_step_outcome_feedback") as mock_step:
            result = coordinator.run("d1", mock_result, confidence=0.9, task="test")

        mock_step.assert_not_called()
        assert result.outcome_feedback is None

    def test_import_error_returns_none(self):
        """When OutcomeFeedbackBridge can't be imported, step returns None."""
        coordinator = self._make_coordinator()

        original_import = __import__

        def selective_import(name, *args, **kwargs):
            if "outcome_feedback" in name:
                raise ImportError("outcome_feedback not installed")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=selective_import):
            result = coordinator._step_outcome_feedback("d1")

        assert result is None

    def test_runtime_error_returns_none(self):
        """When bridge raises RuntimeError, step returns None."""
        coordinator = self._make_coordinator()

        mock_bridge_cls = MagicMock()
        mock_bridge = MagicMock()
        mock_bridge.run_feedback_cycle.side_effect = RuntimeError("verifier broken")
        mock_bridge_cls.return_value = mock_bridge

        with patch(
            "aragora.nomic.outcome_feedback.OutcomeFeedbackBridge",
            mock_bridge_cls,
        ):
            result = coordinator._step_outcome_feedback("d1")

        assert result is None

    def test_result_attached_to_pipeline_output(self):
        """Feedback result is stored on PostDebateResult.outcome_feedback."""
        result = PostDebateResult()
        assert result.outcome_feedback is None

        feedback_data = {
            "goals_generated": 1,
            "suggestions_queued": 1,
            "trickster_adjustment": 1.0,
            "domains_flagged": ["finance"],
            "agents_flagged": ["gemini"],
        }
        result.outcome_feedback = feedback_data
        assert result.outcome_feedback["goals_generated"] == 1
        assert result.outcome_feedback["agents_flagged"] == ["gemini"]

    def test_feedback_failure_doesnt_cascade(self):
        """When feedback returns None, other steps still execute."""
        coordinator = self._make_coordinator(
            auto_explain=True,
            auto_persist_receipt=True,
        )
        mock_result = self._make_debate_result()

        with (
            patch.object(coordinator, "_step_explain", return_value={"explanation": "test"}),
            patch.object(coordinator, "_step_outcome_feedback", return_value=None),
            patch.object(coordinator, "_step_persist_receipt", return_value=True),
        ):
            result = coordinator.run("d1", mock_result, confidence=0.9, task="test")

        assert result.outcome_feedback is None
        assert result.explanation == {"explanation": "test"}
        assert result.receipt_persisted is True

    def test_step_with_mock_bridge_returns_dict(self):
        """Full integration: mock the bridge to return a cycle result."""
        coordinator = self._make_coordinator()

        mock_cycle_result = {
            "goals_generated": 2,
            "suggestions_queued": 2,
            "trickster_adjustment": 0.85,
            "domains_flagged": ["healthcare", "security"],
            "agents_flagged": ["claude"],
        }

        mock_bridge_cls = MagicMock()
        mock_bridge = MagicMock()
        mock_bridge.run_feedback_cycle.return_value = mock_cycle_result
        mock_bridge_cls.return_value = mock_bridge

        with patch(
            "aragora.nomic.outcome_feedback.OutcomeFeedbackBridge",
            mock_bridge_cls,
        ):
            result = coordinator._step_outcome_feedback("d1")

        assert result is not None
        assert result["goals_generated"] == 2
        assert result["trickster_adjustment"] == 0.85
        assert "healthcare" in result["domains_flagged"]

    def test_step_ordering_after_calibration_before_bridge(self):
        """Outcome feedback runs after calibration push and before execution bridge."""
        call_order = []

        coordinator = self._make_coordinator(
            auto_push_calibration=True,
            auto_outcome_feedback=True,
            auto_execution_bridge=True,
            auto_explain=False,
            auto_persist_receipt=False,
        )
        mock_result = self._make_debate_result()

        def track_calibration(*args, **kwargs):
            call_order.append("calibration")
            return False

        def track_feedback(*args, **kwargs):
            call_order.append("outcome_feedback")
            return None

        def track_bridge(*args, **kwargs):
            call_order.append("execution_bridge")
            return []

        with (
            patch.object(coordinator, "_step_push_calibration", side_effect=track_calibration),
            patch.object(coordinator, "_step_outcome_feedback", side_effect=track_feedback),
            patch.object(coordinator, "_step_execution_bridge", side_effect=track_bridge),
        ):
            coordinator.run("d1", mock_result, confidence=0.95, task="test")

        assert call_order == ["calibration", "outcome_feedback", "execution_bridge"]

    def test_zero_goals_returns_result(self):
        """When no systematic errors, bridge still returns a valid result."""
        coordinator = self._make_coordinator()

        mock_cycle_result = {
            "goals_generated": 0,
            "suggestions_queued": 0,
            "trickster_adjustment": 1.0,
            "domains_flagged": [],
            "agents_flagged": [],
        }

        mock_bridge_cls = MagicMock()
        mock_bridge = MagicMock()
        mock_bridge.run_feedback_cycle.return_value = mock_cycle_result
        mock_bridge_cls.return_value = mock_bridge

        with patch(
            "aragora.nomic.outcome_feedback.OutcomeFeedbackBridge",
            mock_bridge_cls,
        ):
            result = coordinator._step_outcome_feedback("d1")

        assert result is not None
        assert result["goals_generated"] == 0
        assert result["domains_flagged"] == []
