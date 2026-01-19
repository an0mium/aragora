"""
Tests for the JudgmentPhase class.

Tests judge selection strategies and termination logic.
"""

from dataclasses import dataclass
from typing import Optional
from unittest.mock import MagicMock

import pytest


@dataclass
class MockAgent:
    """Mock Agent for testing."""

    name: str
    role: str = "debater"


@dataclass
class MockRating:
    """Mock agent rating for testing."""

    elo: float = 1200.0
    calibration_score: float = 0.7


@dataclass
class MockProtocol:
    """Mock DebateProtocol for testing."""

    judge_selection: str = "last"
    judge_termination: bool = False
    min_rounds_before_judge_check: int = 2


class TestJudgeSelection:
    """Tests for judge selection strategies."""

    def test_last_selection_returns_synthesizer(self):
        """'last' selection should prefer synthesizer role."""
        from aragora.debate.phases.judgment import JudgmentPhase

        protocol = MockProtocol(judge_selection="last")
        agents = [
            MockAgent("alice", role="debater"),
            MockAgent("bob", role="synthesizer"),
            MockAgent("charlie", role="debater"),
        ]
        phase = JudgmentPhase(protocol, agents)

        judge = phase.select_judge({}, [])

        assert judge.name == "bob"
        assert judge.role == "synthesizer"

    def test_last_selection_falls_back_to_last_agent(self):
        """'last' selection should use last agent if no synthesizer."""
        from aragora.debate.phases.judgment import JudgmentPhase

        protocol = MockProtocol(judge_selection="last")
        agents = [
            MockAgent("alice"),
            MockAgent("bob"),
            MockAgent("charlie"),
        ]
        phase = JudgmentPhase(protocol, agents)

        judge = phase.select_judge({}, [])

        assert judge.name == "charlie"

    def test_random_selection_returns_valid_agent(self):
        """'random' selection should return one of the agents."""
        from aragora.debate.phases.judgment import JudgmentPhase

        protocol = MockProtocol(judge_selection="random")
        agents = [MockAgent("alice"), MockAgent("bob"), MockAgent("charlie")]
        phase = JudgmentPhase(protocol, agents)

        judge = phase.select_judge({}, [])

        assert judge.name in ["alice", "bob", "charlie"]

    def test_voted_falls_back_to_random_without_async(self):
        """'voted' selection should fall back to random in sync context."""
        from aragora.debate.phases.judgment import JudgmentPhase

        protocol = MockProtocol(judge_selection="voted")
        agents = [MockAgent("alice"), MockAgent("bob")]
        phase = JudgmentPhase(protocol, agents)

        judge = phase.select_judge({}, [])

        assert judge.name in ["alice", "bob"]

    def test_elo_ranked_selects_highest_rated(self):
        """'elo_ranked' should select the highest ELO-rated agent."""
        from aragora.debate.phases.judgment import JudgmentPhase

        mock_elo = MagicMock()
        mock_elo.get_leaderboard.return_value = [
            MagicMock(agent_name="bob", elo=1500),
            MagicMock(agent_name="alice", elo=1400),
            MagicMock(agent_name="charlie", elo=1300),
        ]

        protocol = MockProtocol(judge_selection="elo_ranked")
        agents = [MockAgent("alice"), MockAgent("bob"), MockAgent("charlie")]
        phase = JudgmentPhase(protocol, agents, elo_system=mock_elo)

        judge = phase.select_judge({}, [])

        assert judge.name == "bob"

    def test_elo_ranked_without_elo_system_falls_back(self):
        """'elo_ranked' without ELO system should fall back to random."""
        from aragora.debate.phases.judgment import JudgmentPhase

        protocol = MockProtocol(judge_selection="elo_ranked")
        agents = [MockAgent("alice"), MockAgent("bob")]
        phase = JudgmentPhase(protocol, agents, elo_system=None)

        judge = phase.select_judge({}, [])

        assert judge.name in ["alice", "bob"]

    def test_calibrated_uses_composite_score(self):
        """'calibrated' should use composite score for selection."""
        from aragora.debate.phases.judgment import JudgmentPhase

        mock_elo = MagicMock()

        def mock_composite(agent_name: str) -> float:
            scores = {"alice": 0.8, "bob": 0.9, "charlie": 0.7}
            return scores.get(agent_name, 0.0)

        protocol = MockProtocol(judge_selection="calibrated")
        agents = [MockAgent("alice"), MockAgent("bob"), MockAgent("charlie")]
        phase = JudgmentPhase(
            protocol, agents, elo_system=mock_elo, composite_score_fn=mock_composite
        )

        judge = phase.select_judge({}, [])

        assert judge.name == "bob"  # Highest composite score

    def test_calibrated_without_score_fn_falls_back_to_elo(self):
        """'calibrated' without composite_score_fn should fall back to elo_ranked."""
        from aragora.debate.phases.judgment import JudgmentPhase

        mock_elo = MagicMock()
        mock_elo.get_leaderboard.return_value = [
            MagicMock(agent_name="alice", elo=1500),
        ]

        protocol = MockProtocol(judge_selection="calibrated")
        agents = [MockAgent("alice"), MockAgent("bob")]
        phase = JudgmentPhase(protocol, agents, elo_system=mock_elo)

        judge = phase.select_judge({}, [])

        assert judge.name == "alice"

    def test_unknown_selection_defaults_to_random(self):
        """Unknown selection method should default to random."""
        from aragora.debate.phases.judgment import JudgmentPhase

        protocol = MockProtocol(judge_selection="unknown-method")
        agents = [MockAgent("alice"), MockAgent("bob")]
        phase = JudgmentPhase(protocol, agents)

        judge = phase.select_judge({}, [])

        assert judge.name in ["alice", "bob"]

    def test_empty_agents_raises_error(self):
        """Empty agents list should raise ValueError."""
        from aragora.debate.phases.judgment import JudgmentPhase

        protocol = MockProtocol(judge_selection="random")
        phase = JudgmentPhase(protocol, [])

        with pytest.raises(ValueError, match="No agents available"):
            phase.select_judge({}, [])


class TestJudgeTermination:
    """Tests for judge-based termination logic."""

    def test_no_termination_when_disabled(self):
        """Should continue when judge_termination is disabled."""
        from aragora.debate.phases.judgment import JudgmentPhase

        protocol = MockProtocol(judge_termination=False)
        phase = JudgmentPhase(protocol, [])

        should_continue, reason = phase.should_terminate(5, {}, "Conclusive: yes")

        assert should_continue is True

    def test_no_termination_before_min_rounds(self):
        """Should continue before min_rounds_before_judge_check."""
        from aragora.debate.phases.judgment import JudgmentPhase

        protocol = MockProtocol(
            judge_termination=True, min_rounds_before_judge_check=3
        )
        phase = JudgmentPhase(protocol, [])

        should_continue, reason = phase.should_terminate(2, {}, "Conclusive: yes")

        assert should_continue is True

    def test_no_termination_without_response(self):
        """Should continue when judge_response is None."""
        from aragora.debate.phases.judgment import JudgmentPhase

        protocol = MockProtocol(
            judge_termination=True, min_rounds_before_judge_check=1
        )
        phase = JudgmentPhase(protocol, [])

        should_continue, reason = phase.should_terminate(5, {}, None)

        assert should_continue is True

    def test_terminates_on_conclusive_yes(self):
        """Should terminate when judge says 'Conclusive: yes'."""
        from aragora.debate.phases.judgment import JudgmentPhase

        protocol = MockProtocol(
            judge_termination=True, min_rounds_before_judge_check=1
        )
        phase = JudgmentPhase(protocol, [])

        should_continue, reason = phase.should_terminate(
            5, {}, "Conclusive: yes\nReason: Agreement reached"
        )

        assert should_continue is False
        assert "Agreement reached" in reason

    def test_continues_on_conclusive_no(self):
        """Should continue when judge says 'Conclusive: no'."""
        from aragora.debate.phases.judgment import JudgmentPhase

        protocol = MockProtocol(
            judge_termination=True, min_rounds_before_judge_check=1
        )
        phase = JudgmentPhase(protocol, [])

        should_continue, reason = phase.should_terminate(
            5, {}, "Conclusive: no\nReason: More discussion needed"
        )

        assert should_continue is True

    def test_continues_on_missing_conclusive(self):
        """Should continue when response lacks 'Conclusive:' marker."""
        from aragora.debate.phases.judgment import JudgmentPhase

        protocol = MockProtocol(
            judge_termination=True, min_rounds_before_judge_check=1
        )
        phase = JudgmentPhase(protocol, [])

        should_continue, reason = phase.should_terminate(
            5, {}, "This is a general response without markers"
        )

        assert should_continue is True


class TestJudgeStats:
    """Tests for judge statistics."""

    def test_basic_stats(self):
        """Should return basic judge information."""
        from aragora.debate.phases.judgment import JudgmentPhase

        protocol = MockProtocol(judge_selection="random")
        agents = [MockAgent("alice", role="judge")]
        phase = JudgmentPhase(protocol, agents)

        judge = agents[0]
        stats = phase.get_judge_stats(judge)

        assert stats["name"] == "alice"
        assert stats["role"] == "judge"
        assert stats["selection_method"] == "random"

    def test_stats_includes_elo_when_available(self):
        """Should include ELO rating when available."""
        from aragora.debate.phases.judgment import JudgmentPhase

        mock_elo = MagicMock()
        mock_elo.get_rating.return_value = MockRating(elo=1350)

        protocol = MockProtocol()
        agents = [MockAgent("alice")]
        phase = JudgmentPhase(protocol, agents, elo_system=mock_elo)

        stats = phase.get_judge_stats(agents[0])

        assert stats["elo"] == 1350

    def test_stats_includes_calibration_when_available(self):
        """Should include calibration weight when available."""
        from aragora.debate.phases.judgment import JudgmentPhase

        def mock_calibration(name: str) -> float:
            return 1.2

        protocol = MockProtocol()
        agents = [MockAgent("alice")]
        phase = JudgmentPhase(
            protocol, agents, calibration_weight_fn=mock_calibration
        )

        stats = phase.get_judge_stats(agents[0])

        assert stats["calibration_weight"] == 1.2

    def test_stats_handles_missing_elo_gracefully(self):
        """Should handle ELO lookup failure gracefully."""
        from aragora.debate.phases.judgment import JudgmentPhase

        mock_elo = MagicMock()
        mock_elo.get_rating.side_effect = KeyError("Unknown agent")

        protocol = MockProtocol()
        agents = [MockAgent("alice")]
        phase = JudgmentPhase(protocol, agents, elo_system=mock_elo)

        stats = phase.get_judge_stats(agents[0])

        assert "elo" not in stats
        assert stats["name"] == "alice"
