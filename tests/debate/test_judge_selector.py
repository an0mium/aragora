"""
Tests for JudgeSelector - judge selection strategies for debates.

Tests cover:
- JudgeSelector initialization and configuration
- Selection strategies (Last, Random, EloRanked, Calibrated, CruxAware, Voted)
- JudgePanel multi-judge coordination
- JudgingStrategy voting (Majority, Supermajority, Unanimous, Weighted)
- Circuit breaker awareness
- Fallback selection
- Edge cases and error handling
"""

from __future__ import annotations

from datetime import datetime
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


class MockAgent:
    """Mock agent for testing."""

    def __init__(self, name: str, role: str = "proposer"):
        self.name = name
        self.role = role


@pytest.fixture
def mock_agents():
    """Create list of mock agents."""
    return [
        MockAgent("claude"),
        MockAgent("gpt4"),
        MockAgent("gemini"),
    ]


@pytest.fixture
def mock_elo_system():
    """Create mock ELO system."""
    elo = MagicMock()

    def make_rating(name):
        rating = MagicMock()
        rating.elo = 1500 + hash(name) % 200
        rating.calibration_score = 0.8
        rating.wins = 10
        rating.losses = 5
        return rating

    elo.get_rating = MagicMock(side_effect=make_rating)
    elo.get_ratings_batch = MagicMock(
        side_effect=lambda names: {name: make_rating(name) for name in names}
    )
    elo.get_leaderboard = MagicMock(
        return_value=[
            MagicMock(agent="claude", elo=1600),
            MagicMock(agent="gpt4", elo=1550),
            MagicMock(agent="gemini", elo=1500),
        ]
    )
    return elo


class TestJudgeScore:
    """Tests for JudgeScore dataclass."""

    def test_judge_score_creation(self):
        """Test creating JudgeScore."""
        score = JudgeScore(
            agent_name="claude",
            elo_score=1.0,
            calibration_score=0.85,
            composite_score=0.75,
        )

        assert score.agent_name == "claude"
        assert score.elo_score == 1.0
        assert score.calibration_score == 0.85
        assert score.composite_score == 0.75


class TestJudgeScoringMixin:
    """Tests for JudgeScoringMixin."""

    def test_get_calibration_weight_no_elo(self):
        """Test calibration weight without ELO system."""
        mixin = JudgeScoringMixin()
        weight = mixin.get_calibration_weight("claude")

        assert weight == 1.0

    def test_get_calibration_weight_with_elo(self, mock_elo_system):
        """Test calibration weight with ELO system."""
        mixin = JudgeScoringMixin(elo_system=mock_elo_system)
        weight = mixin.get_calibration_weight("claude")

        # 0.5 + calibration_score (0.8) = 1.3
        assert 0.5 <= weight <= 1.5

    def test_compute_composite_score_no_elo(self):
        """Test composite score without ELO system."""
        mixin = JudgeScoringMixin()
        score = mixin.compute_composite_score("claude")

        assert score == 0.0

    def test_compute_composite_score_with_elo(self, mock_elo_system):
        """Test composite score with ELO system."""
        mixin = JudgeScoringMixin(elo_system=mock_elo_system)
        score = mixin.compute_composite_score("claude")

        assert isinstance(score, float)

    def test_get_all_scores(self, mock_agents, mock_elo_system):
        """Test getting scores for all agents."""
        mixin = JudgeScoringMixin(elo_system=mock_elo_system)
        scores = mixin.get_all_scores(mock_agents)

        assert len(scores) == len(mock_agents)
        assert all(isinstance(s, JudgeScore) for s in scores)
        # Should be sorted by composite score descending
        assert scores[0].composite_score >= scores[-1].composite_score


class TestLastAgentStrategy:
    """Tests for LastAgentStrategy."""

    @pytest.mark.asyncio
    async def test_select_last_agent(self, mock_agents):
        """Test selecting last agent."""
        strategy = LastAgentStrategy()
        result = await strategy.select(agents=mock_agents, proposals={}, context=[])

        assert result == mock_agents[-1]

    @pytest.mark.asyncio
    async def test_select_synthesizer_if_available(self, mock_agents):
        """Test selecting synthesizer if available."""
        # Add a synthesizer
        synth = MockAgent("synth", role="synthesizer")
        agents = mock_agents + [synth]

        strategy = LastAgentStrategy()
        result = await strategy.select(agents=agents, proposals={}, context=[])

        assert result == synth

    @pytest.mark.asyncio
    async def test_select_empty_agents(self):
        """Test selecting from empty list."""
        strategy = LastAgentStrategy()
        result = await strategy.select(agents=[], proposals={}, context=[])

        assert result is None


class TestRandomStrategy:
    """Tests for RandomStrategy."""

    @pytest.mark.asyncio
    async def test_select_random_agent(self, mock_agents):
        """Test random agent selection."""
        strategy = RandomStrategy()
        result = await strategy.select(agents=mock_agents, proposals={}, context=[])

        assert result in mock_agents

    @pytest.mark.asyncio
    async def test_select_random_distribution(self, mock_agents):
        """Test random selection has some distribution."""
        strategy = RandomStrategy()
        selections = []

        for _ in range(100):
            result = await strategy.select(agents=mock_agents, proposals={}, context=[])
            selections.append(result.name)

        # Should have some variety in selections
        unique_selections = set(selections)
        assert len(unique_selections) > 1


class TestEloRankedStrategy:
    """Tests for EloRankedStrategy."""

    @pytest.mark.asyncio
    async def test_select_highest_elo(self, mock_agents, mock_elo_system):
        """Test selecting highest ELO agent."""
        strategy = EloRankedStrategy(elo_system=mock_elo_system)
        result = await strategy.select(agents=mock_agents, proposals={}, context=[])

        assert result in mock_agents
        # Should select claude which is at top of leaderboard
        assert result.name == "claude"

    @pytest.mark.asyncio
    async def test_select_no_elo_system(self, mock_agents):
        """Test selection without ELO system (fallback to random)."""
        strategy = EloRankedStrategy(elo_system=None)
        result = await strategy.select(agents=mock_agents, proposals={}, context=[])

        assert result in mock_agents


class TestCalibratedStrategy:
    """Tests for CalibratedStrategy."""

    @pytest.mark.asyncio
    async def test_select_best_calibrated(self, mock_agents, mock_elo_system):
        """Test selecting best calibrated agent."""
        strategy = CalibratedStrategy(elo_system=mock_elo_system)
        result = await strategy.select(agents=mock_agents, proposals={}, context=[])

        assert result in mock_agents

    @pytest.mark.asyncio
    async def test_select_uses_composite_score(self, mock_agents, mock_elo_system):
        """Test that selection uses composite score."""
        strategy = CalibratedStrategy(elo_system=mock_elo_system)

        # Get all scores
        scores = strategy.get_all_scores(mock_agents)

        result = await strategy.select(agents=mock_agents, proposals={}, context=[])

        # Result should be agent with highest composite score
        best_score = max(scores, key=lambda s: s.composite_score)
        assert result.name == best_score.agent_name


class TestCruxAwareStrategy:
    """Tests for CruxAwareStrategy."""

    @pytest.mark.asyncio
    async def test_select_crux_aware(self, mock_agents, mock_elo_system):
        """Test crux-aware selection."""
        mock_consensus_memory = MagicMock()
        mock_consensus_memory.find_similar_debates = MagicMock(return_value=[])

        strategy = CruxAwareStrategy(
            elo_system=mock_elo_system, consensus_memory=mock_consensus_memory
        )
        result = await strategy.select(agents=mock_agents, proposals={}, context=[])

        assert result in mock_agents

    @pytest.mark.asyncio
    async def test_fallback_to_calibrated(self, mock_agents, mock_elo_system):
        """Test fallback to calibrated when no crux history."""
        strategy = CruxAwareStrategy(elo_system=mock_elo_system, consensus_memory=None)
        result = await strategy.select(agents=mock_agents, proposals={}, context=[])

        assert result in mock_agents


class TestVotedStrategy:
    """Tests for VotedStrategy."""

    @pytest.mark.asyncio
    async def test_voted_selection(self, mock_agents):
        """Test voted selection strategy."""

        async def mock_generate(agent, prompt, context):
            # Each agent votes for a different agent
            if agent.name == "claude":
                return "I vote for gpt4"
            elif agent.name == "gpt4":
                return "I vote for gemini"
            else:
                return "I vote for claude"

        def build_prompt(candidates, proposals):
            return "Vote for the best judge."

        strategy = VotedStrategy(
            generate_fn=mock_generate,
            build_vote_prompt_fn=build_prompt,
        )
        result = await strategy.select(
            agents=mock_agents,
            proposals={"claude": "prop1", "gpt4": "prop2"},
            context=[],
        )

        # Each agent should get one vote, random tiebreaker
        assert result in mock_agents


class TestJudgeSelector:
    """Tests for JudgeSelector main class."""

    def test_init_basic(self, mock_agents):
        """Test basic initialization."""
        selector = JudgeSelector(agents=mock_agents, judge_selection="random")

        assert selector._agents == list(mock_agents)
        assert selector._judge_selection == "random"

    def test_init_with_elo(self, mock_agents, mock_elo_system):
        """Test initialization with ELO system."""
        selector = JudgeSelector(
            agents=mock_agents,
            elo_system=mock_elo_system,
            judge_selection="elo_ranked",
        )

        assert selector._elo_system is mock_elo_system

    def test_init_with_circuit_breaker(self, mock_agents):
        """Test initialization with circuit breaker."""
        mock_breaker = MagicMock()
        selector = JudgeSelector(
            agents=mock_agents,
            judge_selection="random",
            circuit_breaker=mock_breaker,
        )

        assert selector._circuit_breaker is mock_breaker

    @pytest.mark.asyncio
    async def test_select_judge(self, mock_agents, mock_elo_system):
        """Test judge selection."""
        selector = JudgeSelector(
            agents=mock_agents,
            elo_system=mock_elo_system,
            judge_selection="elo_ranked",
        )

        result = await selector.select_judge(proposals={"claude": "Proposal"}, context=[])

        assert result in mock_agents

    @pytest.mark.asyncio
    async def test_get_judge_candidates(self, mock_agents, mock_elo_system):
        """Test getting ordered judge candidates."""
        selector = JudgeSelector(
            agents=mock_agents,
            elo_system=mock_elo_system,
            judge_selection="calibrated",
        )

        candidates = await selector.get_judge_candidates(proposals={}, context=[])

        assert len(candidates) <= len(mock_agents)
        assert all(c in mock_agents for c in candidates)

    def test_filter_available_agents_no_breaker(self, mock_agents):
        """Test filtering without circuit breaker returns all agents."""
        selector = JudgeSelector(agents=mock_agents, judge_selection="random")

        available = selector._filter_available_agents(mock_agents)

        assert len(available) == len(mock_agents)

    def test_filter_available_agents_with_breaker(self, mock_agents):
        """Test filtering with circuit breaker."""
        mock_breaker = MagicMock()
        # gpt4 is unavailable
        mock_breaker.is_available = MagicMock(side_effect=lambda name: name != "gpt4")

        selector = JudgeSelector(
            agents=mock_agents,
            judge_selection="random",
            circuit_breaker=mock_breaker,
        )

        available = selector._filter_available_agents(mock_agents)

        # gpt4 should be filtered out
        assert len(available) == 2
        assert all(a.name != "gpt4" for a in available)

    def test_from_protocol(self, mock_agents, mock_elo_system):
        """Test creating from protocol."""
        mock_protocol = MagicMock()
        mock_protocol.judge_selection = "calibrated"

        selector = JudgeSelector.from_protocol(
            protocol=mock_protocol,
            agents=mock_agents,
            elo_system=mock_elo_system,
        )

        assert isinstance(selector, JudgeSelector)
        assert selector._judge_selection == "calibrated"


class TestJudgingStrategy:
    """Tests for JudgingStrategy enum."""

    def test_strategy_values(self):
        """Test strategy enum values."""
        assert JudgingStrategy.MAJORITY.value == "majority"
        assert JudgingStrategy.SUPERMAJORITY.value == "supermajority"
        assert JudgingStrategy.UNANIMOUS.value == "unanimous"
        assert JudgingStrategy.WEIGHTED.value == "weighted"


class TestJudgeVote:
    """Tests for JudgeVote enum."""

    def test_vote_values(self):
        """Test vote enum values."""
        assert JudgeVote.APPROVE.value == "approve"
        assert JudgeVote.REJECT.value == "reject"
        assert JudgeVote.ABSTAIN.value == "abstain"


class TestJudgeVoteRecord:
    """Tests for JudgeVoteRecord dataclass."""

    def test_vote_record_creation(self):
        """Test creating vote record."""
        record = JudgeVoteRecord(
            judge_name="claude",
            vote=JudgeVote.APPROVE,
            confidence=0.9,
            reasoning="Well-reasoned argument",
            weight=1.2,
        )

        assert record.judge_name == "claude"
        assert record.vote == JudgeVote.APPROVE
        assert record.confidence == 0.9
        assert "Well-reasoned" in record.reasoning
        assert record.weight == 1.2

    def test_vote_record_default_weight(self):
        """Test vote record has default weight."""
        record = JudgeVoteRecord(
            judge_name="claude",
            vote=JudgeVote.APPROVE,
            confidence=0.9,
            reasoning="Good",
        )

        assert record.weight == 1.0


class TestJudgingResult:
    """Tests for JudgingResult dataclass."""

    def test_result_creation(self):
        """Test creating judging result."""
        result = JudgingResult(
            approved=True,
            strategy=JudgingStrategy.MAJORITY,
            votes=[],
            approval_ratio=0.67,
            weighted_approval=0.7,
            confidence=0.85,
            reasoning="Majority approved",
        )

        assert result.approved is True
        assert result.strategy == JudgingStrategy.MAJORITY
        assert result.approval_ratio == 0.67
        assert result.confidence == 0.85

    def test_result_to_dict(self):
        """Test result serialization."""
        result = JudgingResult(
            approved=True,
            strategy=JudgingStrategy.MAJORITY,
            votes=[],
            approval_ratio=0.67,
            weighted_approval=0.7,
            confidence=0.85,
            reasoning="Majority approved",
        )

        data = result.to_dict()

        assert data["approved"] is True
        assert data["strategy"] == "majority"
        assert data["approval_ratio"] == 0.67


class TestJudgePanel:
    """Tests for JudgePanel multi-judge coordination."""

    def test_panel_creation(self, mock_agents):
        """Test creating judge panel."""
        panel = JudgePanel(judges=mock_agents[:2], strategy=JudgingStrategy.MAJORITY)

        assert len(panel.judges) == 2
        assert panel.strategy == JudgingStrategy.MAJORITY

    def test_record_vote(self, mock_agents):
        """Test recording a vote."""
        panel = JudgePanel(judges=mock_agents[:2], strategy=JudgingStrategy.MAJORITY)

        record = panel.record_vote(
            judge_name="claude",
            vote=JudgeVote.APPROVE,
            confidence=0.9,
            reasoning="Good proposal",
        )

        assert len(panel.votes) == 1
        assert record.judge_name == "claude"
        assert record.vote == JudgeVote.APPROVE

    def test_get_result_majority_approved(self, mock_agents):
        """Test getting result with majority approval."""
        panel = JudgePanel(judges=mock_agents, strategy=JudgingStrategy.MAJORITY)

        panel.record_vote("claude", JudgeVote.APPROVE, 0.9, "Good")
        panel.record_vote("gpt4", JudgeVote.APPROVE, 0.8, "Agree")
        panel.record_vote("gemini", JudgeVote.REJECT, 0.7, "Disagree")

        result = panel.get_result()

        assert result.approved is True
        assert result.approval_ratio == 2 / 3  # 66.7%

    def test_get_result_majority_rejected(self, mock_agents):
        """Test getting result with majority rejection."""
        panel = JudgePanel(judges=mock_agents, strategy=JudgingStrategy.MAJORITY)

        panel.record_vote("claude", JudgeVote.REJECT, 0.9, "Bad")
        panel.record_vote("gpt4", JudgeVote.REJECT, 0.8, "Disagree")
        panel.record_vote("gemini", JudgeVote.APPROVE, 0.7, "Good")

        result = panel.get_result()

        assert result.approved is False
        assert result.approval_ratio == 1 / 3

    def test_get_result_supermajority_approved(self, mock_agents):
        """Test getting result with supermajority approval."""
        panel = JudgePanel(judges=mock_agents, strategy=JudgingStrategy.SUPERMAJORITY)

        panel.record_vote("claude", JudgeVote.APPROVE, 0.9, "Good")
        panel.record_vote("gpt4", JudgeVote.APPROVE, 0.8, "Agree")
        panel.record_vote("gemini", JudgeVote.REJECT, 0.7, "Disagree")

        result = panel.get_result()

        # 2/3 = 66.7% which meets supermajority threshold
        assert result.approved is True

    def test_get_result_supermajority_rejected(self, mock_agents):
        """Test supermajority rejection when threshold not met."""
        # Need 4 agents for this test
        agents = mock_agents + [MockAgent("llama")]
        panel = JudgePanel(judges=agents, strategy=JudgingStrategy.SUPERMAJORITY)

        panel.record_vote("claude", JudgeVote.APPROVE, 0.9, "Good")
        panel.record_vote("gpt4", JudgeVote.APPROVE, 0.8, "Agree")
        panel.record_vote("gemini", JudgeVote.REJECT, 0.7, "Disagree")
        panel.record_vote("llama", JudgeVote.REJECT, 0.7, "Disagree")

        result = panel.get_result()

        # 2/4 = 50% which is below supermajority (66.7%)
        assert result.approved is False

    def test_get_result_unanimous_approved(self, mock_agents):
        """Test getting result with unanimous approval."""
        panel = JudgePanel(judges=mock_agents, strategy=JudgingStrategy.UNANIMOUS)

        panel.record_vote("claude", JudgeVote.APPROVE, 0.9, "Good")
        panel.record_vote("gpt4", JudgeVote.APPROVE, 0.8, "Agree")
        panel.record_vote("gemini", JudgeVote.APPROVE, 0.7, "Agree")

        result = panel.get_result()

        assert result.approved is True

    def test_get_result_unanimous_rejected(self, mock_agents):
        """Test unanimous rejection when any judge rejects."""
        panel = JudgePanel(judges=mock_agents, strategy=JudgingStrategy.UNANIMOUS)

        panel.record_vote("claude", JudgeVote.APPROVE, 0.9, "Good")
        panel.record_vote("gpt4", JudgeVote.APPROVE, 0.8, "Agree")
        panel.record_vote("gemini", JudgeVote.REJECT, 0.7, "Disagree")

        result = panel.get_result()

        # One rejection breaks unanimous
        assert result.approved is False
        assert "gemini" in result.dissenting_judges

    def test_get_result_weighted(self, mock_agents):
        """Test getting result with weighted strategy."""
        panel = JudgePanel(
            judges=mock_agents,
            strategy=JudgingStrategy.WEIGHTED,
            judge_weights={"claude": 1.5, "gpt4": 1.0, "gemini": 0.8},
        )

        # Claude approves with high weight, others reject
        panel.record_vote("claude", JudgeVote.APPROVE, 0.9, "Good")
        panel.record_vote("gpt4", JudgeVote.REJECT, 0.8, "Disagree")
        panel.record_vote("gemini", JudgeVote.REJECT, 0.7, "Disagree")

        result = panel.get_result()

        # Claude: 1.5 approve, gpt4+gemini: 1.8 reject
        # weighted_approval = 1.5 / (1.5 + 1.0 + 0.8) = 0.45
        assert result.approved is False

    def test_reset_panel(self, mock_agents):
        """Test resetting panel votes."""
        panel = JudgePanel(judges=mock_agents[:2], strategy=JudgingStrategy.MAJORITY)

        panel.record_vote("claude", JudgeVote.APPROVE, 0.9, "Good")
        assert len(panel.votes) == 1

        panel.reset()
        assert len(panel.votes) == 0

    def test_panel_no_votes(self, mock_agents):
        """Test panel result with no votes."""
        panel = JudgePanel(judges=mock_agents[:2], strategy=JudgingStrategy.MAJORITY)

        result = panel.get_result()

        assert result.approved is False
        assert result.reasoning == "No votes recorded"

    def test_panel_abstentions_excluded(self, mock_agents):
        """Test abstentions are excluded from ratio calculation."""
        panel = JudgePanel(judges=mock_agents, strategy=JudgingStrategy.MAJORITY)

        panel.record_vote("claude", JudgeVote.APPROVE, 0.9, "Good")
        panel.record_vote("gpt4", JudgeVote.REJECT, 0.8, "Bad")
        panel.record_vote("gemini", JudgeVote.ABSTAIN, 0.5, "Uncertain")

        result = panel.get_result()

        # 1 approve, 1 reject, 1 abstain
        # approval_ratio = 1 / 2 = 0.5 (abstentions excluded)
        assert result.approval_ratio == 0.5
        assert "gemini" in result.abstaining_judges


class TestCreateJudgePanel:
    """Tests for create_judge_panel convenience function."""

    def test_create_panel_basic(self, mock_agents):
        """Test creating panel with defaults."""
        panel = create_judge_panel(
            candidates=mock_agents,
            count=2,
        )

        assert len(panel.judges) == 2

    def test_create_panel_excludes_participants(self, mock_agents):
        """Test panel excludes debate participants."""
        participants = [mock_agents[0]]  # claude

        panel = create_judge_panel(
            candidates=mock_agents,
            participants=participants,
            count=3,
            exclude_participants=True,
        )

        # Should exclude claude
        assert all(j.name != "claude" for j in panel.judges)

    def test_create_panel_with_strategy(self, mock_agents):
        """Test creating panel with specific strategy."""
        panel = create_judge_panel(
            candidates=mock_agents,
            strategy=JudgingStrategy.SUPERMAJORITY,
            count=3,
        )

        assert panel.strategy == JudgingStrategy.SUPERMAJORITY

    def test_create_panel_with_elo_weights(self, mock_agents, mock_elo_system):
        """Test panel uses ELO-based weights."""
        panel = create_judge_panel(
            candidates=mock_agents,
            elo_system=mock_elo_system,
            count=3,
        )

        # Should have weights based on calibration scores
        assert len(panel.judge_weights) > 0


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_select_single_agent(self, mock_elo_system):
        """Test selection with single agent."""
        single_agent = MockAgent("solo")

        selector = JudgeSelector(
            agents=[single_agent],
            elo_system=mock_elo_system,
            judge_selection="elo_ranked",
        )

        result = await selector.select_judge(proposals={}, context=[])

        assert result == single_agent

    @pytest.mark.asyncio
    async def test_select_all_agents_unavailable(self, mock_agents):
        """Test selection when all agents are unavailable via circuit breaker."""
        mock_breaker = MagicMock()
        mock_breaker.is_available = MagicMock(return_value=False)

        selector = JudgeSelector(
            agents=mock_agents,
            judge_selection="random",
            circuit_breaker=mock_breaker,
        )

        result = await selector.select_judge(proposals={}, context=[])

        # Should fall back to selecting from all agents
        assert result in mock_agents

    @pytest.mark.asyncio
    async def test_selector_with_invalid_strategy(self, mock_agents):
        """Test selector with invalid strategy name."""
        selector = JudgeSelector(
            agents=mock_agents,
            judge_selection="invalid_strategy",
        )

        # Should fall back to random
        result = await selector.select_judge(proposals={}, context=[])

        assert result in mock_agents

    def test_panel_all_abstain(self, mock_agents):
        """Test panel result when all abstain."""
        panel = JudgePanel(judges=mock_agents, strategy=JudgingStrategy.MAJORITY)

        panel.record_vote("claude", JudgeVote.ABSTAIN, 0.5, "Uncertain")
        panel.record_vote("gpt4", JudgeVote.ABSTAIN, 0.5, "Uncertain")
        panel.record_vote("gemini", JudgeVote.ABSTAIN, 0.5, "Uncertain")

        result = panel.get_result()

        # All abstain means 0 non-abstaining votes
        assert result.approval_ratio == 0.0
        assert len(result.abstaining_judges) == 3

    @pytest.mark.asyncio
    async def test_empty_agents_list(self):
        """Test handling empty agents list."""
        selector = JudgeSelector(
            agents=[],
            judge_selection="random",
        )

        result = await selector.select_judge(proposals={}, context=[])

        assert result is None
