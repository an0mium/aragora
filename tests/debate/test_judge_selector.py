"""
Tests for judge selector module.

Tests cover:
- JudgeScore dataclass
- JudgeScoringMixin: calibration weights, composite scores, batch scoring
- JudgeSelectionStrategy subclasses: Last, Random, EloRanked, Calibrated, CruxAware, Voted
- JudgeSelector: initialization, select_judge, get_judge_candidates, from_protocol, circuit breaker
- JudgePanel: record_vote, get_result, reset, all strategies (MAJORITY, SUPERMAJORITY, UNANIMOUS, WEIGHTED)
- JudgePanel.deliberate_and_vote async deliberation flow
- JudgingResult.to_dict serialization
- create_judge_panel convenience function
- Edge cases: empty agents, missing ELO, unknown strategy, all unavailable
"""

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.debate.judge_selector import (
    CalibratedStrategy,
    CruxAwareStrategy,
    EloRankedStrategy,
    JudgePanel,
    JudgeScore,
    JudgeSelector,
    JudgeScoringMixin,
    JudgeVote,
    JudgeVoteRecord,
    JudgingResult,
    JudgingStrategy,
    LastAgentStrategy,
    RandomStrategy,
    VotedStrategy,
    create_judge_panel,
)


# =============================================================================
# Mock Objects
# =============================================================================


@dataclass
class MockAgent:
    """Mock agent for testing."""

    name: str = "test-agent"
    role: str = ""


@dataclass
class MockRating:
    """Mock ELO rating object."""

    agent_name: str = "test-agent"
    elo: float = 1200.0
    calibration_score: float = 0.7


@dataclass
class MockDebateRecord:
    """Mock historical debate record."""

    dissenting_agents: list = field(default_factory=list)
    consensus: Any = None


class MockConsensus:
    """Mock consensus with dissenting_agents."""

    def __init__(self, dissenting_agents: list | None = None):
        self.dissenting_agents = dissenting_agents or []


# =============================================================================
# JudgeScore Tests
# =============================================================================


class TestJudgeScore:
    """Tests for JudgeScore dataclass."""

    def test_stores_all_fields(self):
        """JudgeScore stores all fields correctly."""
        score = JudgeScore(
            agent_name="claude",
            elo_score=0.4,
            calibration_score=0.8,
            composite_score=0.52,
        )
        assert score.agent_name == "claude"
        assert score.elo_score == 0.4
        assert score.calibration_score == 0.8
        assert score.composite_score == 0.52


# =============================================================================
# JudgeScoringMixin Tests
# =============================================================================


class TestJudgeScoringMixin:
    """Tests for JudgeScoringMixin scoring utilities."""

    def test_calibration_weight_no_elo_system(self):
        """Returns 1.0 when no ELO system is configured."""
        mixin = JudgeScoringMixin(elo_system=None)
        assert mixin.get_calibration_weight("agent") == 1.0

    def test_calibration_weight_with_elo(self):
        """Returns weight = 0.5 + calibration_score."""
        elo = MagicMock()
        elo.get_rating.return_value = MockRating(calibration_score=0.7)
        mixin = JudgeScoringMixin(elo_system=elo)
        assert mixin.get_calibration_weight("agent") == pytest.approx(1.2)

    def test_calibration_weight_on_error(self):
        """Returns 1.0 when ELO lookup raises an exception."""
        elo = MagicMock()
        elo.get_rating.side_effect = RuntimeError("not found")
        mixin = JudgeScoringMixin(elo_system=elo)
        assert mixin.get_calibration_weight("agent") == 1.0

    def test_composite_score_no_elo_system(self):
        """Returns 0.0 when no ELO system is configured."""
        mixin = JudgeScoringMixin(elo_system=None)
        assert mixin.compute_composite_score("agent") == 0.0

    def test_composite_score_calculation(self):
        """Computes 70% ELO + 30% calibration."""
        elo = MagicMock()
        # ELO=1500 -> normalized=(1500-1000)/500=1.0
        elo.get_rating.return_value = MockRating(elo=1500, calibration_score=0.8)
        mixin = JudgeScoringMixin(elo_system=elo)
        expected = (1.0 * 0.7) + (0.8 * 0.3)
        assert mixin.compute_composite_score("agent") == pytest.approx(expected)

    def test_composite_score_clamps_low_elo(self):
        """ELO below 1000 is clamped to 0 in normalized score."""
        elo = MagicMock()
        elo.get_rating.return_value = MockRating(elo=800, calibration_score=0.5)
        mixin = JudgeScoringMixin(elo_system=elo)
        expected = (0.0 * 0.7) + (0.5 * 0.3)
        assert mixin.compute_composite_score("agent") == pytest.approx(expected)

    def test_get_all_scores_sorted_descending(self):
        """get_all_scores returns scores sorted by composite descending."""
        elo = MagicMock()
        elo.get_ratings_batch.return_value = {
            "low": MockRating(agent_name="low", elo=1100, calibration_score=0.3),
            "high": MockRating(agent_name="high", elo=1500, calibration_score=0.9),
        }
        agents = [MockAgent(name="low"), MockAgent(name="high")]
        mixin = JudgeScoringMixin(elo_system=elo)
        scores = mixin.get_all_scores(agents)
        assert len(scores) == 2
        assert scores[0].agent_name == "high"
        assert scores[0].composite_score > scores[1].composite_score

    def test_get_all_scores_no_elo(self):
        """get_all_scores returns zero scores when no ELO system."""
        mixin = JudgeScoringMixin(elo_system=None)
        agents = [MockAgent(name="a"), MockAgent(name="b")]
        scores = mixin.get_all_scores(agents)
        assert len(scores) == 2
        for s in scores:
            assert s.composite_score == 0.0


# =============================================================================
# LastAgentStrategy Tests
# =============================================================================


class TestLastAgentStrategy:
    """Tests for LastAgentStrategy."""

    @pytest.mark.asyncio
    async def test_selects_synthesizer_if_present(self):
        """Selects agent with role='synthesizer' when available."""
        agents = [
            MockAgent(name="a1"),
            MockAgent(name="synth", role="synthesizer"),
            MockAgent(name="a3"),
        ]
        strategy = LastAgentStrategy()
        judge = await strategy.select(agents, {}, [])
        assert judge.name == "synth"

    @pytest.mark.asyncio
    async def test_selects_last_agent_when_no_synthesizer(self):
        """Selects last agent when no synthesizer is present."""
        agents = [MockAgent(name="a"), MockAgent(name="b"), MockAgent(name="c")]
        strategy = LastAgentStrategy()
        judge = await strategy.select(agents, {}, [])
        assert judge.name == "c"

    @pytest.mark.asyncio
    async def test_returns_none_for_empty_agents(self):
        """Returns None when agents list is empty."""
        strategy = LastAgentStrategy()
        assert await strategy.select([], {}, []) is None


# =============================================================================
# RandomStrategy Tests
# =============================================================================


class TestRandomStrategy:
    """Tests for RandomStrategy."""

    @pytest.mark.asyncio
    async def test_selects_from_agents(self):
        """Selects an agent from the provided list."""
        agents = [MockAgent(name="a"), MockAgent(name="b")]
        strategy = RandomStrategy()
        judge = await strategy.select(agents, {}, [])
        assert judge.name in ("a", "b")

    @pytest.mark.asyncio
    async def test_returns_none_for_empty(self):
        """Returns None for empty agents."""
        strategy = RandomStrategy()
        assert await strategy.select([], {}, []) is None


# =============================================================================
# EloRankedStrategy Tests
# =============================================================================


class TestEloRankedStrategy:
    """Tests for EloRankedStrategy."""

    @pytest.mark.asyncio
    async def test_selects_highest_elo(self):
        """Selects agent with highest ELO from leaderboard."""
        elo = MagicMock()
        elo.get_leaderboard.return_value = [
            MockRating(agent_name="top", elo=1500),
            MockRating(agent_name="mid", elo=1200),
        ]
        agents = [MockAgent(name="mid"), MockAgent(name="top")]
        strategy = EloRankedStrategy(elo_system=elo)
        judge = await strategy.select(agents, {}, [])
        assert judge.name == "top"

    @pytest.mark.asyncio
    async def test_falls_back_to_random_without_elo(self):
        """Falls back to random when no ELO system."""
        agents = [MockAgent(name="a"), MockAgent(name="b")]
        strategy = EloRankedStrategy(elo_system=None)
        judge = await strategy.select(agents, {}, [])
        assert judge is not None
        assert judge.name in ("a", "b")

    @pytest.mark.asyncio
    async def test_falls_back_on_elo_error(self):
        """Falls back to random when ELO query raises exception."""
        elo = MagicMock()
        elo.get_leaderboard.side_effect = RuntimeError("db error")
        agents = [MockAgent(name="a")]
        strategy = EloRankedStrategy(elo_system=elo)
        judge = await strategy.select(agents, {}, [])
        assert judge is not None


# =============================================================================
# CalibratedStrategy Tests
# =============================================================================


class TestCalibratedStrategy:
    """Tests for CalibratedStrategy."""

    @pytest.mark.asyncio
    async def test_selects_best_composite(self):
        """Selects agent with best composite score."""
        elo = MagicMock()
        elo.get_ratings_batch.return_value = {
            "claude": MockRating(agent_name="claude", elo=1500, calibration_score=0.9),
            "gpt": MockRating(agent_name="gpt", elo=1200, calibration_score=0.5),
        }
        agents = [MockAgent(name="gpt"), MockAgent(name="claude")]
        strategy = CalibratedStrategy(elo_system=elo)
        judge = await strategy.select(agents, {}, [])
        assert judge.name == "claude"

    @pytest.mark.asyncio
    async def test_falls_back_without_elo(self):
        """Falls back to random without ELO system."""
        agents = [MockAgent(name="a")]
        strategy = CalibratedStrategy(elo_system=None)
        judge = await strategy.select(agents, {}, [])
        assert judge is not None


# =============================================================================
# CruxAwareStrategy Tests
# =============================================================================


class TestCruxAwareStrategy:
    """Tests for CruxAwareStrategy."""

    @pytest.mark.asyncio
    async def test_selects_historical_dissenter(self):
        """Selects agent who historically dissented on similar cruxes."""
        elo = MagicMock()
        elo.get_ratings_batch.return_value = {
            "contrarian": MockRating(agent_name="contrarian", elo=1300),
        }
        consensus_memory = MagicMock()
        debate_record = MockDebateRecord()
        debate_record.consensus = MockConsensus(dissenting_agents=["contrarian"])
        consensus_memory.find_similar_debates.return_value = [debate_record]

        agents = [MockAgent(name="mainstream"), MockAgent(name="contrarian")]
        strategy = CruxAwareStrategy(elo_system=elo, consensus_memory=consensus_memory)
        judge = await strategy.select(agents, {}, [], cruxes=[{"claim": "AI safety is paramount"}])
        assert judge.name == "contrarian"

    @pytest.mark.asyncio
    async def test_falls_back_without_cruxes(self):
        """Falls back to calibrated when no cruxes provided."""
        elo = MagicMock()
        elo.get_ratings_batch.return_value = {
            "best": MockRating(agent_name="best", elo=1500, calibration_score=0.9),
            "ok": MockRating(agent_name="ok", elo=1100, calibration_score=0.4),
        }
        agents = [MockAgent(name="ok"), MockAgent(name="best")]
        strategy = CruxAwareStrategy(elo_system=elo, consensus_memory=None)
        judge = await strategy.select(agents, {}, [], cruxes=None)
        assert judge.name == "best"

    @pytest.mark.asyncio
    async def test_returns_none_for_empty_agents(self):
        """Returns None for empty agents list."""
        strategy = CruxAwareStrategy()
        assert await strategy.select([], {}, []) is None

    @pytest.mark.asyncio
    async def test_ultimate_fallback_to_random(self):
        """Falls back to random when no ELO and no consensus memory."""
        agents = [MockAgent(name="a"), MockAgent(name="b")]
        strategy = CruxAwareStrategy(elo_system=None, consensus_memory=None)
        judge = await strategy.select(agents, {}, [], cruxes=None)
        assert judge is not None

    def test_find_historical_dissenters_no_memory(self):
        """Returns empty list when no consensus memory."""
        strategy = CruxAwareStrategy(consensus_memory=None)
        result = strategy._find_historical_dissenters([{"claim": "test"}], [])
        assert result == []

    def test_rank_by_elo_no_system(self):
        """Returns agents in original order when no ELO system."""
        agents = [MockAgent(name="b"), MockAgent(name="a")]
        strategy = CruxAwareStrategy(elo_system=None)
        ranked = strategy._rank_by_elo(agents)
        assert [a.name for a in ranked] == ["b", "a"]


# =============================================================================
# VotedStrategy Tests
# =============================================================================


class TestVotedStrategy:
    """Tests for VotedStrategy."""

    @pytest.mark.asyncio
    async def test_agents_vote_for_judge(self):
        """Agents vote and majority winner is selected."""
        agents = [MockAgent(name="alice"), MockAgent(name="bob"), MockAgent(name="carol")]

        async def mock_generate(agent, prompt, context):
            if agent.name in ("alice", "carol"):
                return "I vote for bob"
            return "I vote for carol"

        def mock_build_prompt(other_agents, proposals):
            return "Vote for the best judge."

        strategy = VotedStrategy(
            generate_fn=mock_generate,
            build_vote_prompt_fn=mock_build_prompt,
        )
        judge = await strategy.select(agents, {"alice": "p1", "bob": "p2"}, [])
        assert judge.name == "bob"

    @pytest.mark.asyncio
    async def test_falls_back_on_all_errors(self):
        """Falls back to random when all vote generations fail."""

        async def mock_generate(agent, prompt, context):
            raise RuntimeError("fail")

        def mock_build_prompt(other_agents, proposals):
            return "Vote"

        agents = [MockAgent(name="a"), MockAgent(name="b")]
        strategy = VotedStrategy(
            generate_fn=mock_generate,
            build_vote_prompt_fn=mock_build_prompt,
        )
        judge = await strategy.select(agents, {}, [])
        assert judge is not None

    @pytest.mark.asyncio
    async def test_returns_none_for_empty_agents(self):
        """Returns None for empty agents."""
        strategy = VotedStrategy(
            generate_fn=AsyncMock(),
            build_vote_prompt_fn=lambda a, p: "",
        )
        assert await strategy.select([], {}, []) is None

    @pytest.mark.asyncio
    async def test_sanitize_fn_applied(self):
        """Sanitize function is applied to raw responses."""
        agents = [MockAgent(name="alice"), MockAgent(name="bob")]

        async def mock_generate(agent, prompt, context):
            return "VOTE: bob!!!"

        def mock_build_prompt(other_agents, proposals):
            return "Vote"

        def mock_sanitize(response, agent_name):
            return response.lower()

        strategy = VotedStrategy(
            generate_fn=mock_generate,
            build_vote_prompt_fn=mock_build_prompt,
            sanitize_fn=mock_sanitize,
        )
        judge = await strategy.select(agents, {}, [])
        assert judge is not None


# =============================================================================
# JudgeSelector Initialization Tests
# =============================================================================


class TestJudgeSelectorInit:
    """Tests for JudgeSelector initialization."""

    def test_default_strategy_is_random(self):
        """Default selection strategy is random."""
        selector = JudgeSelector(agents=[MockAgent(name="a")])
        assert selector._judge_selection == "random"

    def test_stores_agents_and_elo(self):
        """Stores agents and ELO system references."""
        elo = MagicMock()
        selector = JudgeSelector(agents=[MockAgent(name="a")], elo_system=elo)
        assert selector._elo_system is elo
        assert len(selector._agents) == 1

    def test_registers_voted_when_fns_provided(self):
        """Registers voted strategy when generate and prompt fns are provided."""
        selector = JudgeSelector(
            agents=[MockAgent(name="a")],
            generate_fn=AsyncMock(),
            build_vote_prompt_fn=lambda a, p: "",
        )
        assert "voted" in selector._strategies

    def test_no_voted_without_fns(self):
        """Does not register voted strategy without generate_fn."""
        selector = JudgeSelector(agents=[MockAgent(name="a")])
        assert "voted" not in selector._strategies

    def test_all_builtin_strategies_registered(self):
        """Registers last, random, elo_ranked, calibrated, crux_aware."""
        selector = JudgeSelector(agents=[MockAgent(name="a")])
        for name in ("last", "random", "elo_ranked", "calibrated", "crux_aware"):
            assert name in selector._strategies


# =============================================================================
# JudgeSelector.select_judge Tests
# =============================================================================


class TestJudgeSelectorSelectJudge:
    """Tests for JudgeSelector.select_judge method."""

    @pytest.mark.asyncio
    async def test_selects_with_last_strategy(self):
        """Uses the 'last' strategy to select last agent."""
        agents = [MockAgent(name="a"), MockAgent(name="b"), MockAgent(name="c")]
        selector = JudgeSelector(agents=agents, judge_selection="last")
        judge = await selector.select_judge({}, [])
        assert judge.name == "c"

    @pytest.mark.asyncio
    async def test_falls_back_for_unknown_strategy(self):
        """Falls back to random for unknown strategy name."""
        agents = [MockAgent(name="x")]
        selector = JudgeSelector(agents=agents, judge_selection="nonexistent")
        judge = await selector.select_judge({}, [])
        assert judge is not None

    @pytest.mark.asyncio
    async def test_circuit_breaker_filters_agents(self):
        """Circuit breaker filters unavailable agents."""
        agents = [MockAgent(name="up"), MockAgent(name="down")]
        cb = MagicMock()
        cb.is_available.side_effect = lambda name: name == "up"
        selector = JudgeSelector(agents=agents, judge_selection="last", circuit_breaker=cb)
        judge = await selector.select_judge({}, [])
        assert judge.name == "up"

    @pytest.mark.asyncio
    async def test_all_unavailable_uses_all(self):
        """Uses all agents when circuit breaker marks all as unavailable."""
        agents = [MockAgent(name="a"), MockAgent(name="b")]
        cb = MagicMock()
        cb.is_available.return_value = False
        selector = JudgeSelector(agents=agents, judge_selection="random", circuit_breaker=cb)
        judge = await selector.select_judge({}, [])
        assert judge is not None

    @pytest.mark.asyncio
    async def test_returns_none_for_empty_agents(self):
        """Returns None when no agents are available."""
        selector = JudgeSelector(agents=[], judge_selection="random")
        judge = await selector.select_judge({}, [])
        assert judge is None


# =============================================================================
# JudgeSelector.get_judge_candidates Tests
# =============================================================================


class TestGetJudgeCandidates:
    """Tests for JudgeSelector.get_judge_candidates."""

    @pytest.mark.asyncio
    async def test_returns_ordered_candidates(self):
        """Returns candidates ordered by composite score."""
        elo = MagicMock()
        elo.get_ratings_batch.return_value = {
            "best": MockRating(agent_name="best", elo=1500, calibration_score=0.9),
            "mid": MockRating(agent_name="mid", elo=1200, calibration_score=0.5),
            "low": MockRating(agent_name="low", elo=1000, calibration_score=0.2),
        }
        agents = [MockAgent(name="low"), MockAgent(name="mid"), MockAgent(name="best")]
        selector = JudgeSelector(agents=agents, elo_system=elo)
        candidates = await selector.get_judge_candidates({}, [], max_candidates=2)
        assert len(candidates) == 2
        assert candidates[0].name == "best"

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_agents(self):
        """Returns empty list when no agents are available."""
        selector = JudgeSelector(agents=[])
        candidates = await selector.get_judge_candidates({}, [])
        assert candidates == []

    @pytest.mark.asyncio
    async def test_works_without_elo(self):
        """Returns candidates (shuffled) when no ELO system."""
        agents = [MockAgent(name="a"), MockAgent(name="b")]
        selector = JudgeSelector(agents=agents, elo_system=None)
        candidates = await selector.get_judge_candidates({}, [], max_candidates=2)
        assert len(candidates) == 2


# =============================================================================
# JudgeSelector.from_protocol Tests
# =============================================================================


class TestFromProtocol:
    """Tests for JudgeSelector.from_protocol classmethod."""

    def test_creates_selector_from_protocol(self):
        """Creates selector with strategy from protocol object."""
        protocol = MagicMock()
        protocol.judge_selection = "calibrated"
        agents = [MockAgent(name="a")]
        selector = JudgeSelector.from_protocol(protocol, agents=agents)
        assert selector._judge_selection == "calibrated"


# =============================================================================
# JudgePanel Tests
# =============================================================================


class TestJudgePanel:
    """Tests for JudgePanel class."""

    def test_init_defaults(self):
        """Panel initializes with default strategy and empty votes."""
        panel = JudgePanel(judges=[MockAgent(name="j1")])
        assert panel.strategy == JudgingStrategy.MAJORITY
        assert panel.votes == []
        assert panel.judge_weights == {}

    def test_record_vote_with_custom_weight(self):
        """Records a vote and applies judge weight."""
        panel = JudgePanel(judges=[MockAgent(name="j1")], judge_weights={"j1": 1.5})
        record = panel.record_vote("j1", JudgeVote.APPROVE, 0.9, "Good reasoning")
        assert record.weight == 1.5
        assert len(panel.votes) == 1

    def test_reset_clears_votes(self):
        """Reset clears all recorded votes."""
        panel = JudgePanel(judges=[MockAgent(name="j1")])
        panel.record_vote("j1", JudgeVote.APPROVE, 0.8, "ok")
        panel.reset()
        assert len(panel.votes) == 0

    def test_no_votes_result(self):
        """Returns not-approved result when no votes recorded."""
        panel = JudgePanel(judges=[])
        result = panel.get_result()
        assert result.approved is False
        assert result.reasoning == "No votes recorded"

    def test_majority_approves(self):
        """MAJORITY strategy approves when >50% approve."""
        panel = JudgePanel(
            judges=[MockAgent(name=f"j{i}") for i in range(3)],
            strategy=JudgingStrategy.MAJORITY,
        )
        panel.record_vote("j0", JudgeVote.APPROVE, 0.9, "Good")
        panel.record_vote("j1", JudgeVote.APPROVE, 0.8, "Good")
        panel.record_vote("j2", JudgeVote.REJECT, 0.7, "Bad")
        result = panel.get_result()
        assert result.approved is True
        assert result.approval_ratio == pytest.approx(2 / 3)

    def test_majority_rejects(self):
        """MAJORITY strategy rejects when <=50% approve."""
        panel = JudgePanel(
            judges=[MockAgent(name=f"j{i}") for i in range(2)],
            strategy=JudgingStrategy.MAJORITY,
        )
        panel.record_vote("j0", JudgeVote.APPROVE, 0.5, "ok")
        panel.record_vote("j1", JudgeVote.REJECT, 0.5, "no")
        result = panel.get_result()
        assert result.approved is False

    def test_supermajority_threshold(self):
        """SUPERMAJORITY requires >=2/3 approval."""
        panel = JudgePanel(
            judges=[MockAgent(name=f"j{i}") for i in range(3)],
            strategy=JudgingStrategy.SUPERMAJORITY,
        )
        panel.record_vote("j0", JudgeVote.APPROVE, 0.9, "yes")
        panel.record_vote("j1", JudgeVote.APPROVE, 0.8, "yes")
        panel.record_vote("j2", JudgeVote.REJECT, 0.7, "no")
        result = panel.get_result()
        assert result.approved is True  # 2/3 meets threshold

    def test_unanimous_requires_all(self):
        """UNANIMOUS rejects if any judge rejects."""
        panel = JudgePanel(
            judges=[MockAgent(name=f"j{i}") for i in range(2)],
            strategy=JudgingStrategy.UNANIMOUS,
        )
        panel.record_vote("j0", JudgeVote.APPROVE, 0.9, "yes")
        panel.record_vote("j1", JudgeVote.REJECT, 0.5, "no")
        result = panel.get_result()
        assert result.approved is False

    def test_unanimous_all_approve(self):
        """UNANIMOUS approves when all judges approve."""
        panel = JudgePanel(
            judges=[MockAgent(name=f"j{i}") for i in range(2)],
            strategy=JudgingStrategy.UNANIMOUS,
        )
        panel.record_vote("j0", JudgeVote.APPROVE, 0.9, "yes")
        panel.record_vote("j1", JudgeVote.APPROVE, 0.8, "yes")
        result = panel.get_result()
        assert result.approved is True

    def test_weighted_strategy(self):
        """WEIGHTED strategy uses weights for approval calculation."""
        panel = JudgePanel(
            judges=[MockAgent(name="expert"), MockAgent(name="novice")],
            strategy=JudgingStrategy.WEIGHTED,
            judge_weights={"expert": 2.0, "novice": 0.5},
        )
        panel.record_vote("expert", JudgeVote.REJECT, 0.9, "No")
        panel.record_vote("novice", JudgeVote.APPROVE, 0.5, "Yes")
        result = panel.get_result()
        # weighted_approval = 0.5 / (2.0+0.5) = 0.2 -> not > 0.5
        assert result.approved is False

    def test_abstentions_excluded(self):
        """Abstentions are excluded from approval ratio."""
        panel = JudgePanel(
            judges=[MockAgent(name=f"j{i}") for i in range(3)],
            strategy=JudgingStrategy.MAJORITY,
        )
        panel.record_vote("j0", JudgeVote.APPROVE, 0.9, "yes")
        panel.record_vote("j1", JudgeVote.ABSTAIN, 0.3, "unsure")
        panel.record_vote("j2", JudgeVote.REJECT, 0.6, "no")
        result = panel.get_result()
        assert result.approval_ratio == pytest.approx(0.5)
        assert result.abstaining_judges == ["j1"]

    def test_dissenting_judges_tracked(self):
        """Dissenting judges are tracked in result."""
        panel = JudgePanel(judges=[MockAgent(name="j1"), MockAgent(name="j2")])
        panel.record_vote("j1", JudgeVote.APPROVE, 0.9, "yes")
        panel.record_vote("j2", JudgeVote.REJECT, 0.7, "no")
        result = panel.get_result()
        assert "j2" in result.dissenting_judges


# =============================================================================
# JudgingResult.to_dict Tests
# =============================================================================


class TestJudgingResultToDict:
    """Tests for JudgingResult.to_dict serialization."""

    def test_serializes_all_fields(self):
        """to_dict includes all required fields."""
        vote_record = JudgeVoteRecord(
            judge_name="claude",
            vote=JudgeVote.APPROVE,
            confidence=0.9,
            reasoning="Solid argument",
            weight=1.2,
        )
        result = JudgingResult(
            approved=True,
            strategy=JudgingStrategy.MAJORITY,
            votes=[vote_record],
            approval_ratio=1.0,
            weighted_approval=1.0,
            confidence=0.9,
            reasoning="Approved",
            dissenting_judges=[],
            abstaining_judges=[],
        )
        d = result.to_dict()
        assert d["approved"] is True
        assert d["strategy"] == "majority"
        assert len(d["votes"]) == 1
        assert d["votes"][0]["judge"] == "claude"
        assert d["votes"][0]["vote"] == "approve"
        assert d["confidence"] == 0.9
        assert d["dissenting_judges"] == []
        assert d["abstaining_judges"] == []


# =============================================================================
# JudgePanel.deliberate_and_vote Tests
# =============================================================================


class TestDeliberateAndVote:
    """Tests for JudgePanel.deliberate_and_vote async method."""

    @pytest.mark.asyncio
    async def test_produces_result(self):
        """Deliberation collects assessments and produces a result."""
        judges = [MockAgent(name="j1"), MockAgent(name="j2")]
        panel = JudgePanel(judges=judges, strategy=JudgingStrategy.MAJORITY)

        async def mock_generate(judge, prompt, context):
            return "I approve the consensus."

        result = await panel.deliberate_and_vote(
            proposals={"a": "text"},
            task="Evaluate",
            context=[],
            generate_fn=mock_generate,
            deliberation_rounds=1,
        )
        assert isinstance(result, JudgingResult)
        assert len(result.votes) > 0

    @pytest.mark.asyncio
    async def test_no_assessments_returns_empty(self):
        """Returns empty result when all assessments fail."""
        judges = [MockAgent(name="j1")]
        panel = JudgePanel(judges=judges)

        async def failing_generate(judge, prompt, context):
            raise RuntimeError("fail")

        result = await panel.deliberate_and_vote(
            proposals={"a": "text"},
            task="task",
            context=[],
            generate_fn=failing_generate,
        )
        assert result.approved is False
        assert "No assessments" in result.reasoning

    @pytest.mark.asyncio
    async def test_custom_assessment_prompt(self):
        """Uses custom assessment prompt when provided."""
        judges = [MockAgent(name="j1")]
        panel = JudgePanel(judges=judges)
        called = []

        def custom_prompt(proposals, task):
            called.append(True)
            return f"Custom assess: {task}"

        async def mock_generate(judge, prompt, context):
            return "I approve this."

        await panel.deliberate_and_vote(
            proposals={"a": "text"},
            task="task",
            context=[],
            generate_fn=mock_generate,
            build_assessment_prompt=custom_prompt,
        )
        assert len(called) > 0


# =============================================================================
# create_judge_panel Tests
# =============================================================================


class TestCreateJudgePanel:
    """Tests for create_judge_panel convenience function."""

    def test_creates_panel_with_defaults(self):
        """Creates panel from candidates with default settings."""
        candidates = [MockAgent(name=f"c{i}") for i in range(5)]
        panel = create_judge_panel(candidates=candidates, count=3)
        assert isinstance(panel, JudgePanel)
        assert len(panel.judges) == 3

    def test_excludes_participants(self):
        """Excludes debate participants from judging pool."""
        candidates = [MockAgent(name="p1"), MockAgent(name="p2"), MockAgent(name="j1")]
        participants = [MockAgent(name="p1"), MockAgent(name="p2")]
        panel = create_judge_panel(
            candidates=candidates,
            participants=participants,
            count=3,
            exclude_participants=True,
        )
        judge_names = [j.name for j in panel.judges]
        assert "p1" not in judge_names
        assert "j1" in judge_names

    def test_falls_back_when_all_excluded(self):
        """Falls back to all candidates when filtering removes everyone."""
        candidates = [MockAgent(name="p1")]
        participants = [MockAgent(name="p1")]
        panel = create_judge_panel(candidates=candidates, participants=participants, count=1)
        assert len(panel.judges) >= 1

    def test_uses_elo_for_weights(self):
        """Uses ELO system for scoring and setting judge weights."""
        elo = MagicMock()
        elo.get_ratings_batch.return_value = {
            "best": MockRating(agent_name="best", elo=1500, calibration_score=0.9),
            "ok": MockRating(agent_name="ok", elo=1100, calibration_score=0.4),
        }
        candidates = [MockAgent(name="ok"), MockAgent(name="best")]
        panel = create_judge_panel(candidates=candidates, count=2, elo_system=elo)
        assert len(panel.judge_weights) > 0

    def test_custom_strategy(self):
        """Creates panel with specified strategy."""
        candidates = [MockAgent(name=f"c{i}") for i in range(3)]
        panel = create_judge_panel(
            candidates=candidates,
            strategy=JudgingStrategy.SUPERMAJORITY,
            count=3,
        )
        assert panel.strategy == JudgingStrategy.SUPERMAJORITY
