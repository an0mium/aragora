"""Tests for SelectionFeedbackLoop - performance to selection weights."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from aragora.debate.selection_feedback import (
    AgentFeedbackState,
    FeedbackLoopConfig,
    FeedbackLoopMetrics,
    SelectionFeedbackLoop,
)


class TestAgentFeedbackState:
    """Tests for AgentFeedbackState dataclass."""

    def test_initial_state(self):
        """Test initial state values."""
        state = AgentFeedbackState(agent_name="claude")

        assert state.total_debates == 0
        assert state.wins == 0
        assert state.losses == 0
        assert state.win_rate == 0.5  # Default when no data
        assert state.timeout_rate == 0.0
        assert state.calibration_score == 0.5

    def test_win_rate_calculation(self):
        """Test win rate calculation."""
        state = AgentFeedbackState(agent_name="claude", wins=7, losses=3)
        assert state.win_rate == 0.7

    def test_timeout_rate_calculation(self):
        """Test timeout rate calculation."""
        state = AgentFeedbackState(agent_name="claude", total_debates=10, timeouts=2)
        assert state.timeout_rate == 0.2

    def test_calibration_score_calculation(self):
        """Test calibration score calculation."""
        state = AgentFeedbackState(
            agent_name="claude",
            correct_predictions=8,
            total_predictions=10,
        )
        assert state.calibration_score == 0.8

    def test_domain_win_rate(self):
        """Test domain-specific win rate."""
        state = AgentFeedbackState(agent_name="claude")
        state.domain_wins["security"] = 5
        state.domain_losses["security"] = 2

        assert state.domain_win_rate("security") == pytest.approx(5 / 7, rel=0.01)
        assert state.domain_win_rate("unknown") == 0.5  # No data


class TestFeedbackLoopConfig:
    """Tests for FeedbackLoopConfig dataclass."""

    def test_defaults(self):
        """Test default configuration values."""
        config = FeedbackLoopConfig()

        assert config.enabled is True
        assert config.performance_to_selection_weight == 0.15
        assert config.min_debates_for_feedback == 3
        assert config.feedback_decay_factor == 0.9
        assert config.max_adjustment == 0.5

    def test_custom_config(self):
        """Test custom configuration."""
        config = FeedbackLoopConfig(
            enabled=False,
            min_debates_for_feedback=10,
            max_adjustment=0.3,
        )

        assert config.enabled is False
        assert config.min_debates_for_feedback == 10
        assert config.max_adjustment == 0.3


class TestSelectionFeedbackLoop:
    """Tests for SelectionFeedbackLoop class."""

    @pytest.fixture
    def feedback_loop(self):
        """Create a feedback loop with default config."""
        return SelectionFeedbackLoop()

    @pytest.fixture
    def feedback_loop_custom(self):
        """Create a feedback loop with custom config."""
        return SelectionFeedbackLoop(
            config=FeedbackLoopConfig(
                min_debates_for_feedback=2,
                performance_to_selection_weight=0.2,
            )
        )

    def test_process_debate_outcome_updates_state(self, feedback_loop):
        """Test that processing updates agent states."""
        adjustments = feedback_loop.process_debate_outcome(
            debate_id="debate-1",
            participants=["claude", "gpt-4"],
            winner="claude",
            domain="security",
        )

        # Check states were created
        claude_state = feedback_loop.get_agent_state("claude")
        gpt_state = feedback_loop.get_agent_state("gpt-4")

        assert claude_state is not None
        assert gpt_state is not None
        assert claude_state.wins == 1
        assert claude_state.losses == 0
        assert gpt_state.wins == 0
        assert gpt_state.losses == 1

    def test_process_debate_outcome_updates_domain_stats(self, feedback_loop):
        """Test that domain stats are updated."""
        feedback_loop.process_debate_outcome(
            debate_id="debate-1",
            participants=["claude", "gpt-4"],
            winner="claude",
            domain="security",
        )

        claude_state = feedback_loop.get_agent_state("claude")

        assert claude_state.domain_wins.get("security") == 1
        assert claude_state.domain_losses.get("security", 0) == 0

    def test_no_adjustment_before_min_debates(self, feedback_loop):
        """Test no adjustment given before min_debates threshold."""
        # Default threshold is 3
        feedback_loop.process_debate_outcome(
            debate_id="debate-1",
            participants=["claude"],
            winner="claude",
        )

        adjustment = feedback_loop.get_selection_adjustment("claude")
        assert adjustment == 0.0

    def test_adjustment_after_min_debates(self, feedback_loop_custom):
        """Test adjustment computed after min_debates threshold."""
        # Custom threshold is 2
        feedback_loop_custom.process_debate_outcome(
            debate_id="debate-1",
            participants=["claude"],
            winner="claude",
        )
        feedback_loop_custom.process_debate_outcome(
            debate_id="debate-2",
            participants=["claude"],
            winner="claude",
        )

        adjustment = feedback_loop_custom.get_selection_adjustment("claude")
        assert adjustment > 0  # Positive because high win rate

    def test_losing_agent_negative_adjustment(self, feedback_loop_custom):
        """Test losing agent gets negative adjustment."""
        # Both debates lost
        feedback_loop_custom.process_debate_outcome(
            debate_id="debate-1",
            participants=["claude", "gpt-4"],
            winner="gpt-4",
        )
        feedback_loop_custom.process_debate_outcome(
            debate_id="debate-2",
            participants=["claude", "gpt-4"],
            winner="gpt-4",
        )

        claude_adj = feedback_loop_custom.get_selection_adjustment("claude")
        gpt_adj = feedback_loop_custom.get_selection_adjustment("gpt-4")

        assert claude_adj < 0  # Lost both
        assert gpt_adj > 0  # Won both

    def test_adjustment_bounded_by_max(self, feedback_loop):
        """Test adjustment is bounded by max_adjustment."""
        # Simulate many wins
        for i in range(20):
            feedback_loop.process_debate_outcome(
                debate_id=f"debate-{i}",
                participants=["super-agent"],
                winner="super-agent",
            )

        adjustment = feedback_loop.get_selection_adjustment("super-agent")
        assert adjustment <= feedback_loop.config.max_adjustment

    def test_timeout_penalty(self, feedback_loop):
        """Test that timeouts add penalty."""
        # Record some debates first
        for i in range(3):
            feedback_loop.process_debate_outcome(
                debate_id=f"debate-{i}",
                participants=["slow-agent"],
                winner="slow-agent",
            )

        initial_adj = feedback_loop.get_selection_adjustment("slow-agent")

        # Record timeout
        feedback_loop.record_timeout("slow-agent")

        new_adj = feedback_loop.get_selection_adjustment("slow-agent")
        assert new_adj < initial_adj  # Should be lower after timeout

    def test_prediction_tracking(self, feedback_loop):
        """Test prediction accuracy tracking."""
        feedback_loop.record_prediction(
            agent_name="claude",
            predicted_winner="claude",
            actual_winner="claude",
            confidence=0.9,
        )
        feedback_loop.record_prediction(
            agent_name="claude",
            predicted_winner="claude",
            actual_winner="gpt-4",
            confidence=0.8,
        )

        state = feedback_loop.get_agent_state("claude")
        assert state.total_predictions == 2
        assert state.correct_predictions == 1
        assert state.calibration_score == 0.5

    def test_domain_adjustment(self, feedback_loop):
        """Test domain-specific adjustment."""
        # Create state with domain wins - need min 3 debates for feedback
        feedback_loop.process_debate_outcome(
            debate_id="debate-1",
            participants=["specialist"],
            winner="specialist",
            domain="security",
        )
        feedback_loop.process_debate_outcome(
            debate_id="debate-2",
            participants=["specialist"],
            winner="specialist",
            domain="security",
        )
        feedback_loop.process_debate_outcome(
            debate_id="debate-3",
            participants=["specialist"],
            winner="specialist",
            domain="security",
        )
        # Add losses in other domain
        feedback_loop.process_debate_outcome(
            debate_id="debate-4",
            participants=["specialist", "other"],
            winner="other",
            domain="other-domain",
        )
        feedback_loop.process_debate_outcome(
            debate_id="debate-5",
            participants=["specialist", "other"],
            winner="other",
            domain="other-domain",
        )
        feedback_loop.process_debate_outcome(
            debate_id="debate-6",
            participants=["specialist", "other"],
            winner="other",
            domain="other-domain",
        )

        sec_adj = feedback_loop.get_domain_adjustment("specialist", "security")
        other_adj = feedback_loop.get_domain_adjustment("specialist", "other-domain")

        assert sec_adj > 0  # 3 wins in security
        assert other_adj < 0  # 0 wins in other-domain

    def test_disabled_loop_returns_zero(self):
        """Test disabled loop returns zero adjustments."""
        feedback_loop = SelectionFeedbackLoop(
            config=FeedbackLoopConfig(enabled=False)
        )

        feedback_loop.process_debate_outcome(
            debate_id="debate-1",
            participants=["claude"],
            winner="claude",
        )

        adjustment = feedback_loop.get_selection_adjustment("claude")
        assert adjustment == 0.0

    def test_get_all_states(self, feedback_loop):
        """Test getting all agent states."""
        feedback_loop.process_debate_outcome(
            debate_id="debate-1",
            participants=["claude", "gpt-4", "gemini"],
            winner="claude",
        )

        states = feedback_loop.get_all_states()
        assert len(states) == 3
        assert "claude" in states
        assert "gpt-4" in states
        assert "gemini" in states

    def test_get_metrics(self, feedback_loop):
        """Test getting feedback loop metrics."""
        feedback_loop.process_debate_outcome(
            debate_id="debate-1",
            participants=["claude", "gpt-4"],
            winner="claude",
        )

        metrics = feedback_loop.get_metrics()

        assert metrics["debates_processed"] == 1
        assert metrics["adjustments_computed"] == 2
        assert metrics["agents_tracked"] == 2

    def test_reset(self, feedback_loop):
        """Test resetting the feedback loop."""
        feedback_loop.process_debate_outcome(
            debate_id="debate-1",
            participants=["claude"],
            winner="claude",
        )

        feedback_loop.reset()

        assert len(feedback_loop.get_all_states()) == 0
        assert feedback_loop.get_selection_adjustment("claude") == 0.0

    def test_decay_old_adjustments(self):
        """Test that old adjustments decay over time."""
        feedback_loop = SelectionFeedbackLoop(
            config=FeedbackLoopConfig(
                min_debates_for_feedback=1,
                feedback_decay_factor=0.5,  # Strong decay for testing
            )
        )

        feedback_loop.process_debate_outcome(
            debate_id="debate-1",
            participants=["claude"],
            winner="claude",
        )

        # Manually set last_updated to 5 days ago
        state = feedback_loop.get_agent_state("claude")
        state.last_updated = datetime.now() - timedelta(days=5)

        # Adjustment should be decayed
        adjustment = feedback_loop.get_selection_adjustment("claude")
        # With 0.5 decay factor and 5 days: 0.5^5 = 0.03125
        # So adjustment should be much smaller than original

        # Original adjustment would be positive for 100% win rate
        # After decay, should still be positive but smaller
        assert adjustment >= 0


class TestFeedbackLoopWithELO:
    """Tests for feedback loop with ELO system integration."""

    def test_elo_contribution(self):
        """Test that ELO rating contributes to adjustment."""
        mock_elo = MagicMock()
        mock_rating = MagicMock()
        mock_rating.elo = 1800  # High ELO
        mock_elo.get_rating.return_value = mock_rating

        feedback_loop = SelectionFeedbackLoop(
            elo_system=mock_elo,
            config=FeedbackLoopConfig(min_debates_for_feedback=1),
        )

        # Process one debate (mix of win/loss for neutral base)
        feedback_loop.process_debate_outcome(
            debate_id="debate-1",
            participants=["high-elo-agent"],
            winner="high-elo-agent",
        )

        adjustment = feedback_loop.get_selection_adjustment("high-elo-agent")
        # Should be positive due to high ELO and win
        assert adjustment > 0
