"""Tests for OutcomeFeedbackBridge — Nomic Loop outcome integration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from aragora.nomic.outcome_feedback import (
    FeedbackGoal,
    OutcomeFeedbackBridge,
)


@pytest.fixture
def bridge():
    return OutcomeFeedbackBridge()


class TestFeedbackGoal:
    """Tests for FeedbackGoal dataclass."""

    def test_priority_from_severity(self):
        goal = FeedbackGoal(
            domain="security",
            agent="claude",
            goal_type="reduce_overconfidence",
            severity=0.5,
            description="Test",
        )
        assert goal.priority == 5

    def test_priority_clamped_high(self):
        goal = FeedbackGoal(
            domain="d", agent="a", goal_type="t", severity=2.0, description="T"
        )
        assert goal.priority == 10

    def test_priority_clamped_low(self):
        goal = FeedbackGoal(
            domain="d", agent="a", goal_type="t", severity=0.0, description="T"
        )
        assert goal.priority == 1


class TestErrorToGoals:
    """Tests for converting error patterns to goals."""

    def test_overconfidence_goal(self, bridge):
        error = {
            "domain": "security",
            "agent": "claude",
            "overconfidence": 0.2,
            "success_rate": 0.7,
            "avg_confidence": 0.9,
            "avg_brier_score": 0.15,
            "total_verifications": 10,
        }
        goals = bridge._error_to_goals(error)
        overconf_goals = [g for g in goals if g.goal_type == "reduce_overconfidence"]
        assert len(overconf_goals) == 1
        assert overconf_goals[0].domain == "security"
        assert overconf_goals[0].agent == "claude"
        assert overconf_goals[0].severity > 0

    def test_low_accuracy_goal(self, bridge):
        error = {
            "domain": "medical",
            "agent": "gpt4",
            "overconfidence": 0.3,
            "success_rate": 0.4,
            "avg_confidence": 0.7,
            "avg_brier_score": 0.35,
            "total_verifications": 10,
        }
        goals = bridge._error_to_goals(error)
        accuracy_goals = [g for g in goals if g.goal_type == "increase_accuracy"]
        assert len(accuracy_goals) == 1
        assert "low accuracy" in accuracy_goals[0].description

    def test_domain_training_goal(self, bridge):
        error = {
            "domain": "finance",
            "agent": "gemini",
            "overconfidence": 0.15,
            "success_rate": 0.65,
            "avg_confidence": 0.8,
            "avg_brier_score": 0.45,
            "total_verifications": 15,  # >= min_verifications * 2
        }
        goals = bridge._error_to_goals(error)
        training_goals = [g for g in goals if g.goal_type == "domain_training"]
        assert len(training_goals) == 1
        assert "temperature scaling" in training_goals[0].description

    def test_no_goals_for_well_calibrated(self, bridge):
        error = {
            "domain": "general",
            "agent": "claude",
            "overconfidence": 0.02,
            "success_rate": 0.85,
            "avg_confidence": 0.87,
            "avg_brier_score": 0.05,
            "total_verifications": 20,
        }
        goals = bridge._error_to_goals(error)
        assert len(goals) == 0

    def test_multiple_goals_from_bad_error(self, bridge):
        error = {
            "domain": "security",
            "agent": "claude",
            "overconfidence": 0.3,
            "success_rate": 0.4,
            "avg_confidence": 0.7,
            "avg_brier_score": 0.5,
            "total_verifications": 20,
        }
        goals = bridge._error_to_goals(error)
        types = {g.goal_type for g in goals}
        assert "reduce_overconfidence" in types
        assert "increase_accuracy" in types
        assert "domain_training" in types


class TestGenerateGoals:
    """Tests for generate_improvement_goals."""

    @patch("aragora.nomic.outcome_feedback.OutcomeVerifier")
    def test_generates_from_verifier(self, MockVerifier, bridge):
        mock_instance = MagicMock()
        MockVerifier.return_value = mock_instance
        mock_instance.get_systematic_errors.return_value = [
            {
                "domain": "security",
                "agent": "claude",
                "overconfidence": 0.25,
                "success_rate": 0.6,
                "avg_confidence": 0.85,
                "avg_brier_score": 0.2,
                "total_verifications": 10,
            }
        ]

        goals = bridge.generate_improvement_goals()
        assert len(goals) > 0
        assert goals[0].severity >= goals[-1].severity  # sorted

    @patch("aragora.nomic.outcome_feedback.OutcomeVerifier", side_effect=ImportError)
    def test_handles_missing_verifier(self, MockVerifier, bridge):
        goals = bridge.generate_improvement_goals()
        assert goals == []

    @patch("aragora.nomic.outcome_feedback.OutcomeVerifier")
    def test_empty_errors_returns_empty(self, MockVerifier, bridge):
        MockVerifier.return_value.get_systematic_errors.return_value = []
        goals = bridge.generate_improvement_goals()
        assert goals == []


class TestGoalToAction:
    """Tests for converting goals to actionable suggestions."""

    def test_overconfidence_action(self, bridge):
        goal = FeedbackGoal(
            domain="security",
            agent="claude",
            goal_type="reduce_overconfidence",
            severity=0.5,
            description="Test",
            metrics={"overconfidence": 0.2},
        )
        action = bridge._goal_to_action(goal)
        assert "temperature scaling" in action.lower()
        assert "claude" in action

    def test_accuracy_action(self, bridge):
        goal = FeedbackGoal(
            domain="medical",
            agent="gpt4",
            goal_type="increase_accuracy",
            severity=0.5,
            description="Test",
            metrics={"success_rate": 0.4},
        )
        action = bridge._goal_to_action(goal)
        assert "selection weight" in action.lower()

    def test_training_action(self, bridge):
        goal = FeedbackGoal(
            domain="finance",
            agent="gemini",
            goal_type="domain_training",
            severity=0.5,
            description="Test",
            metrics={},
        )
        action = bridge._goal_to_action(goal)
        assert "auto_tune_agent" in action


class TestQueueSuggestions:
    """Tests for queuing suggestions."""

    @patch("aragora.nomic.outcome_feedback.OutcomeVerifier")
    @patch("aragora.nomic.outcome_feedback.get_improvement_queue")
    def test_queues_suggestions(self, mock_get_queue, MockVerifier, bridge):
        # Setup verifier
        MockVerifier.return_value.get_systematic_errors.return_value = [
            {
                "domain": "security",
                "agent": "claude",
                "overconfidence": 0.25,
                "success_rate": 0.5,
                "avg_confidence": 0.75,
                "avg_brier_score": 0.2,
                "total_verifications": 10,
            }
        ]

        # Setup queue
        mock_queue = MagicMock()
        mock_get_queue.return_value = mock_queue

        queued = bridge.queue_improvement_suggestions()
        assert queued > 0
        mock_queue.enqueue.assert_called()


class TestRunFeedbackCycle:
    """Tests for full feedback cycle."""

    @patch("aragora.nomic.outcome_feedback.OutcomeVerifier")
    @patch("aragora.nomic.outcome_feedback.OutcomeTracker")
    @patch("aragora.nomic.outcome_feedback.get_improvement_queue")
    def test_full_cycle(self, mock_get_queue, MockTracker, MockVerifier, bridge):
        MockVerifier.return_value.get_systematic_errors.return_value = [
            {
                "domain": "test",
                "agent": "a",
                "overconfidence": 0.2,
                "success_rate": 0.7,
                "avg_confidence": 0.9,
                "avg_brier_score": 0.1,
                "total_verifications": 10,
            }
        ]
        MockTracker.return_value.get_calibration_adjustment.return_value = 0.85
        mock_get_queue.return_value = MagicMock()

        result = bridge.run_feedback_cycle()
        assert result["goals_generated"] > 0
        assert result["trickster_adjustment"] == 0.85
        assert "test" in result["domains_flagged"]
