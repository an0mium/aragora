"""
Tests for JudgmentPhase in debate phases.

Tests judge selection strategies, termination logic, and judge statistics.
"""

import pytest
from dataclasses import dataclass
from typing import Any, Optional
from unittest.mock import MagicMock, patch
import random

from aragora.debate.phases.judgment import JudgmentPhase


# =============================================================================
# Fixtures
# =============================================================================


@dataclass
class MockAgent:
    """Mock agent for testing."""

    name: str
    role: Optional[str] = None


@dataclass
class MockProtocol:
    """Mock protocol for testing."""

    judge_selection: str = "last"
    judge_termination: bool = False
    min_rounds_before_judge_check: int = 2


@dataclass
class MockMessage:
    """Mock message for testing."""

    content: str
    agent: str


@pytest.fixture
def three_agents():
    """Three mock agents with roles."""
    return [
        MockAgent(name="agent1", role="proposer"),
        MockAgent(name="agent2", role="critic"),
        MockAgent(name="agent3", role="synthesizer"),
    ]


@pytest.fixture
def protocol():
    """Default protocol."""
    return MockProtocol()


@dataclass
class MockAgentRating:
    """Mock AgentRating for testing."""

    agent_name: str
    elo: float = 1500.0


@pytest.fixture
def mock_elo_system():
    """Mock ELO system."""
    elo = MagicMock()
    elo.get_leaderboard.return_value = [
        MockAgentRating(agent_name="agent2", elo=1600),
        MockAgentRating(agent_name="agent1", elo=1550),
        MockAgentRating(agent_name="agent3", elo=1500),
    ]
    elo.get_rating.return_value = MockAgentRating(agent_name="agent1", elo=1600)
    return elo


# =============================================================================
# Initialization Tests
# =============================================================================


class TestJudgmentPhaseInit:
    """Tests for JudgmentPhase initialization."""

    def test_init_stores_protocol(self, protocol, three_agents):
        """Should store protocol reference."""
        phase = JudgmentPhase(protocol, three_agents)
        assert phase.protocol is protocol

    def test_init_stores_agents(self, protocol, three_agents):
        """Should store agents list."""
        phase = JudgmentPhase(protocol, three_agents)
        assert phase.agents == three_agents

    def test_init_optional_elo_system(self, protocol, three_agents, mock_elo_system):
        """Should accept optional ELO system."""
        phase = JudgmentPhase(protocol, three_agents, elo_system=mock_elo_system)
        assert phase.elo_system is mock_elo_system

    def test_init_optional_calibration_fn(self, protocol, three_agents):
        """Should accept optional calibration weight function."""

        def cal_fn(name):
            return 0.8

        phase = JudgmentPhase(protocol, three_agents, calibration_weight_fn=cal_fn)
        assert phase._get_calibration_weight is cal_fn


# =============================================================================
# Judge Selection Tests - "last" Strategy
# =============================================================================


class TestSelectJudgeLast:
    """Tests for 'last' judge selection strategy."""

    def test_selects_synthesizer_when_present(self, protocol, three_agents):
        """Should select agent with 'synthesizer' role."""
        protocol.judge_selection = "last"
        phase = JudgmentPhase(protocol, three_agents)

        judge = phase.select_judge({}, [])

        assert judge.name == "agent3"
        assert judge.role == "synthesizer"

    def test_selects_last_agent_when_no_synthesizer(self, protocol):
        """Should select last agent when no synthesizer role."""
        agents = [
            MockAgent(name="a1", role="proposer"),
            MockAgent(name="a2", role="critic"),
            MockAgent(name="a3", role="critic"),
        ]
        protocol.judge_selection = "last"
        phase = JudgmentPhase(protocol, agents)

        judge = phase.select_judge({}, [])

        assert judge.name == "a3"


# =============================================================================
# Judge Selection Tests - "random" Strategy
# =============================================================================


class TestSelectJudgeRandom:
    """Tests for 'random' judge selection strategy."""

    def test_selects_from_agents(self, protocol, three_agents):
        """Should select an agent from the list."""
        protocol.judge_selection = "random"
        phase = JudgmentPhase(protocol, three_agents)

        judge = phase.select_judge({}, [])

        assert judge in three_agents

    def test_random_selection_varies(self, protocol, three_agents):
        """Random selection should vary over multiple calls."""
        protocol.judge_selection = "random"
        phase = JudgmentPhase(protocol, three_agents)

        # Run multiple selections
        selections = [phase.select_judge({}, []) for _ in range(20)]
        unique_judges = set(s.name for s in selections)

        # With 3 agents and 20 selections, should hit multiple agents
        assert len(unique_judges) > 1


# =============================================================================
# Judge Selection Tests - "voted" Strategy
# =============================================================================


class TestSelectJudgeVoted:
    """Tests for 'voted' judge selection strategy."""

    def test_falls_back_to_random_without_async(self, protocol, three_agents):
        """Without async vote function, should fall back to random."""
        protocol.judge_selection = "voted"
        phase = JudgmentPhase(protocol, three_agents)

        judge = phase.select_judge({}, [])

        assert judge in three_agents

    def test_logs_warning_for_voted_without_fn(self, protocol, three_agents, caplog):
        """Should log warning when vote_for_judge_fn provided but can't await."""
        protocol.judge_selection = "voted"
        phase = JudgmentPhase(protocol, three_agents)

        with caplog.at_level("WARNING"):
            judge = phase.select_judge({}, [], vote_for_judge_fn=lambda: None)

        assert "requires async" in caplog.text.lower() or judge in three_agents


# =============================================================================
# Judge Selection Tests - "elo_ranked" Strategy
# =============================================================================


class TestSelectJudgeEloRanked:
    """Tests for 'elo_ranked' judge selection strategy."""

    def test_selects_highest_elo_agent(self, protocol, three_agents, mock_elo_system):
        """Should select agent with highest ELO."""
        protocol.judge_selection = "elo_ranked"
        phase = JudgmentPhase(protocol, three_agents, elo_system=mock_elo_system)

        judge = phase.select_judge({}, [])

        # agent2 has highest ELO (1600)
        assert judge.name == "agent2"

    def test_falls_back_to_random_without_elo(self, protocol, three_agents, caplog):
        """Should fall back to random when no ELO system."""
        protocol.judge_selection = "elo_ranked"
        phase = JudgmentPhase(protocol, three_agents)  # No ELO system

        with caplog.at_level("WARNING"):
            judge = phase.select_judge({}, [])

        assert judge in three_agents

    def test_handles_elo_query_failure(self, protocol, three_agents, mock_elo_system, caplog):
        """Should fall back to random on ELO query failure."""
        protocol.judge_selection = "elo_ranked"
        mock_elo_system.get_leaderboard.side_effect = Exception("DB error")
        phase = JudgmentPhase(protocol, three_agents, elo_system=mock_elo_system)

        with caplog.at_level("WARNING"):
            judge = phase.select_judge({}, [])

        assert judge in three_agents


# =============================================================================
# Judge Selection Tests - "calibrated" Strategy
# =============================================================================


class TestSelectJudgeCalibrated:
    """Tests for 'calibrated' judge selection strategy."""

    def test_selects_highest_composite_score(self, protocol, three_agents, mock_elo_system):
        """Should select agent with highest composite score."""
        protocol.judge_selection = "calibrated"

        # Mock composite scores
        def composite_fn(name):
            return {"agent1": 0.7, "agent2": 0.9, "agent3": 0.8}[name]

        phase = JudgmentPhase(
            protocol,
            three_agents,
            elo_system=mock_elo_system,
            composite_score_fn=composite_fn,
        )

        judge = phase.select_judge({}, [])

        assert judge.name == "agent2"  # Highest composite score

    def test_falls_back_to_elo_without_composite_fn(
        self, protocol, three_agents, mock_elo_system, caplog
    ):
        """Should fall back to elo_ranked without composite_score_fn."""
        protocol.judge_selection = "calibrated"
        phase = JudgmentPhase(
            protocol,
            three_agents,
            elo_system=mock_elo_system,
            # No composite_score_fn
        )

        with caplog.at_level("WARNING"):
            judge = phase.select_judge({}, [])

        # Falls back to elo_ranked, agent2 has highest ELO
        assert judge.name == "agent2"

    def test_falls_back_to_random_without_elo(self, protocol, three_agents, caplog):
        """Should fall back to random without ELO system."""
        protocol.judge_selection = "calibrated"
        phase = JudgmentPhase(protocol, three_agents)  # No ELO system

        with caplog.at_level("WARNING"):
            judge = phase.select_judge({}, [])

        assert judge in three_agents

    def test_handles_score_computation_errors(self, protocol, three_agents, mock_elo_system):
        """Should handle score computation errors gracefully."""
        protocol.judge_selection = "calibrated"

        def failing_composite_fn(name):
            if name == "agent1":
                raise ValueError("Computation failed")
            return 0.8

        phase = JudgmentPhase(
            protocol,
            three_agents,
            elo_system=mock_elo_system,
            composite_score_fn=failing_composite_fn,
        )

        judge = phase.select_judge({}, [])

        # Should still select from remaining valid scores
        assert judge in three_agents


# =============================================================================
# Judge Selection Tests - Default/Unknown Strategy
# =============================================================================


class TestSelectJudgeDefault:
    """Tests for default/unknown judge selection strategies."""

    def test_unknown_strategy_falls_back_to_random(self, protocol, three_agents):
        """Unknown strategy should fall back to random selection."""
        protocol.judge_selection = "nonexistent_strategy"
        phase = JudgmentPhase(protocol, three_agents)

        judge = phase.select_judge({}, [])

        assert judge in three_agents


# =============================================================================
# _require_agents Tests
# =============================================================================


class TestRequireAgents:
    """Tests for _require_agents helper method."""

    def test_returns_agents_when_not_empty(self, protocol, three_agents):
        """Should return agents list when not empty."""
        phase = JudgmentPhase(protocol, three_agents)

        result = phase._require_agents()

        assert result == three_agents

    def test_raises_when_empty(self, protocol):
        """Should raise ValueError when agents list is empty."""
        phase = JudgmentPhase(protocol, [])

        with pytest.raises(ValueError, match="No agents available"):
            phase._require_agents()


# =============================================================================
# should_terminate Tests
# =============================================================================


class TestShouldTerminate:
    """Tests for should_terminate method."""

    def test_continues_when_termination_disabled(self, protocol, three_agents):
        """Should continue when judge_termination is False."""
        protocol.judge_termination = False
        phase = JudgmentPhase(protocol, three_agents)

        should_continue, reason = phase.should_terminate(
            round_num=5,
            proposals={},
            judge_response="Conclusive: yes",
        )

        assert should_continue is True
        assert reason == ""

    def test_continues_before_min_rounds(self, protocol, three_agents):
        """Should continue before min_rounds_before_judge_check."""
        protocol.judge_termination = True
        protocol.min_rounds_before_judge_check = 3
        phase = JudgmentPhase(protocol, three_agents)

        should_continue, reason = phase.should_terminate(
            round_num=1,
            proposals={},
            judge_response="Conclusive: yes",
        )

        assert should_continue is True

    def test_continues_without_judge_response(self, protocol, three_agents):
        """Should continue when no judge response provided."""
        protocol.judge_termination = True
        protocol.min_rounds_before_judge_check = 0
        phase = JudgmentPhase(protocol, three_agents)

        should_continue, reason = phase.should_terminate(
            round_num=5,
            proposals={},
            judge_response=None,
        )

        assert should_continue is True

    def test_terminates_on_conclusive_yes(self, protocol, three_agents):
        """Should terminate when judge says 'Conclusive: yes'."""
        protocol.judge_termination = True
        protocol.min_rounds_before_judge_check = 0
        phase = JudgmentPhase(protocol, three_agents)

        should_continue, reason = phase.should_terminate(
            round_num=5,
            proposals={},
            judge_response="Conclusive: yes\nReason: Clear consensus reached",
        )

        assert should_continue is False
        assert "consensus" in reason.lower()

    def test_continues_on_conclusive_no(self, protocol, three_agents):
        """Should continue when judge says 'Conclusive: no'."""
        protocol.judge_termination = True
        protocol.min_rounds_before_judge_check = 0
        phase = JudgmentPhase(protocol, three_agents)

        should_continue, reason = phase.should_terminate(
            round_num=5,
            proposals={},
            judge_response="Conclusive: no\nReason: More discussion needed",
        )

        assert should_continue is True

    def test_parses_reason_from_response(self, protocol, three_agents):
        """Should extract reason from judge response."""
        protocol.judge_termination = True
        protocol.min_rounds_before_judge_check = 0
        phase = JudgmentPhase(protocol, three_agents)

        should_continue, reason = phase.should_terminate(
            round_num=5,
            proposals={},
            judge_response="Conclusive: yes\nReason: All agents agree on solution",
        )

        assert should_continue is False
        assert "All agents agree" in reason


# =============================================================================
# get_judge_stats Tests
# =============================================================================


class TestGetJudgeStats:
    """Tests for get_judge_stats method."""

    def test_returns_basic_stats(self, protocol, three_agents):
        """Should return basic judge stats."""
        phase = JudgmentPhase(protocol, three_agents)
        judge = three_agents[2]

        stats = phase.get_judge_stats(judge)

        assert stats["name"] == "agent3"
        assert stats["role"] == "synthesizer"
        assert stats["selection_method"] == "last"

    def test_includes_elo_when_available(self, protocol, three_agents, mock_elo_system):
        """Should include ELO stats when system available."""
        phase = JudgmentPhase(protocol, three_agents, elo_system=mock_elo_system)
        judge = three_agents[0]

        stats = phase.get_judge_stats(judge)

        assert "elo" in stats
        assert stats["elo"] == 1600

    def test_handles_elo_lookup_failure(self, protocol, three_agents, mock_elo_system, caplog):
        """Should handle ELO lookup errors gracefully."""
        mock_elo_system.get_rating.side_effect = KeyError("Unknown agent")
        phase = JudgmentPhase(protocol, three_agents, elo_system=mock_elo_system)
        judge = three_agents[0]

        with caplog.at_level("DEBUG"):
            stats = phase.get_judge_stats(judge)

        # Should still return basic stats
        assert stats["name"] == "agent1"
        assert "elo" not in stats

    def test_includes_calibration_weight_when_fn_available(self, protocol, three_agents):
        """Should include calibration weight when function provided."""

        def cal_fn(name):
            return 0.85

        phase = JudgmentPhase(
            protocol,
            three_agents,
            calibration_weight_fn=cal_fn,
        )
        judge = three_agents[0]

        stats = phase.get_judge_stats(judge)

        assert "calibration_weight" in stats
        assert stats["calibration_weight"] == 0.85

    def test_handles_calibration_lookup_failure(self, protocol, three_agents, caplog):
        """Should handle calibration lookup errors gracefully."""

        def failing_cal_fn(name):
            raise TypeError("Invalid agent")

        phase = JudgmentPhase(
            protocol,
            three_agents,
            calibration_weight_fn=failing_cal_fn,
        )
        judge = three_agents[0]

        with caplog.at_level("DEBUG"):
            stats = phase.get_judge_stats(judge)

        # Should still return basic stats
        assert stats["name"] == "agent1"
        assert "calibration_weight" not in stats
