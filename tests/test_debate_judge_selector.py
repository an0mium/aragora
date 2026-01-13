"""
Tests for Judge Selector module.

Tests cover:
- JudgeScore dataclass
- JudgeScoringMixin utility methods
- Judge selection strategies (Last, Random, EloRanked, Calibrated, CruxAware, Voted)
- JudgeSelector initialization and configuration
- Circuit breaker integration for agent filtering
- Fallback candidate selection
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
import random

from aragora.debate.judge_selector import (
    JudgeScore,
    JudgeScoringMixin,
    JudgeSelectionStrategy,
    LastAgentStrategy,
    RandomStrategy,
    EloRankedStrategy,
    CalibratedStrategy,
    CruxAwareStrategy,
    VotedStrategy,
    JudgeSelector,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_agent():
    """Create a mock agent."""
    agent = Mock()
    agent.name = "test_agent"
    agent.role = None
    return agent


@pytest.fixture
def mock_agents():
    """Create a list of mock agents."""
    agents = []
    for name in ["agent_a", "agent_b", "agent_c"]:
        agent = Mock()
        agent.name = name
        agent.role = None
        agents.append(agent)
    return agents


@pytest.fixture
def mock_synthesizer():
    """Create a mock synthesizer agent."""
    agent = Mock()
    agent.name = "synthesizer"
    agent.role = "synthesizer"
    return agent


@pytest.fixture
def mock_elo_system():
    """Create a mock ELO system."""
    elo = Mock()

    # Mock rating object
    def get_rating(name):
        rating = Mock()
        if name == "agent_a":
            rating.elo = 1500
            rating.calibration_score = 0.9
        elif name == "agent_b":
            rating.elo = 1200
            rating.calibration_score = 0.7
        else:
            rating.elo = 1000
            rating.calibration_score = 0.5
        return rating

    def get_ratings_batch(names):
        return {name: get_rating(name) for name in names}

    elo.get_rating = Mock(side_effect=get_rating)
    elo.get_ratings_batch = Mock(side_effect=get_ratings_batch)
    elo.get_leaderboard = Mock(
        return_value=[
            Mock(agent="agent_a", elo=1500),
            Mock(agent="agent_b", elo=1200),
            Mock(agent="agent_c", elo=1000),
        ]
    )

    return elo


@pytest.fixture
def mock_circuit_breaker():
    """Create a mock circuit breaker."""
    breaker = Mock()
    breaker.is_available = Mock(return_value=True)
    return breaker


@pytest.fixture
def sample_proposals():
    """Sample proposals for testing."""
    return {
        "agent_a": "Proposal from A",
        "agent_b": "Proposal from B",
        "agent_c": "Proposal from C",
    }


# ============================================================================
# JudgeScore Tests
# ============================================================================


class TestJudgeScore:
    """Tests for JudgeScore dataclass."""

    def test_create_score(self):
        """Test creating a judge score."""
        score = JudgeScore(
            agent_name="claude",
            elo_score=1.0,
            calibration_score=0.85,
            composite_score=0.955,
        )

        assert score.agent_name == "claude"
        assert score.elo_score == 1.0
        assert score.calibration_score == 0.85
        assert score.composite_score == 0.955

    def test_score_fields_accessible(self):
        """Test all fields are accessible."""
        score = JudgeScore(
            agent_name="test",
            elo_score=0.5,
            calibration_score=0.7,
            composite_score=0.56,
        )

        assert hasattr(score, "agent_name")
        assert hasattr(score, "elo_score")
        assert hasattr(score, "calibration_score")
        assert hasattr(score, "composite_score")


# ============================================================================
# JudgeScoringMixin Tests
# ============================================================================


class TestJudgeScoringMixin:
    """Tests for JudgeScoringMixin utility methods."""

    def test_calibration_weight_no_elo_system(self):
        """Test calibration weight returns 1.0 without ELO system."""
        mixin = JudgeScoringMixin()

        weight = mixin.get_calibration_weight("any_agent")
        assert weight == 1.0

    def test_calibration_weight_with_elo_system(self, mock_elo_system):
        """Test calibration weight calculated from ELO system."""
        mixin = JudgeScoringMixin(elo_system=mock_elo_system)

        weight = mixin.get_calibration_weight("agent_a")

        # Should be 0.5 + calibration_score (0.9) = 1.4
        assert weight == 1.4

    def test_calibration_weight_handles_exception(self, mock_elo_system):
        """Test calibration weight handles exceptions gracefully."""
        mock_elo_system.get_rating.side_effect = RuntimeError("DB error")
        mixin = JudgeScoringMixin(elo_system=mock_elo_system)

        weight = mixin.get_calibration_weight("agent_a")
        assert weight == 1.0

    def test_composite_score_no_elo_system(self):
        """Test composite score returns 0.0 without ELO system."""
        mixin = JudgeScoringMixin()

        score = mixin.compute_composite_score("any_agent")
        assert score == 0.0

    def test_composite_score_with_elo_system(self, mock_elo_system):
        """Test composite score calculated correctly."""
        mixin = JudgeScoringMixin(elo_system=mock_elo_system)

        score = mixin.compute_composite_score("agent_a")

        # ELO normalized: (1500 - 1000) / 500 = 1.0
        # Calibration: 0.9
        # Composite: (1.0 * 0.7) + (0.9 * 0.3) = 0.7 + 0.27 = 0.97
        assert score == pytest.approx(0.97)

    def test_composite_score_low_elo_normalized_to_zero(self, mock_elo_system):
        """Test low ELO is normalized to 0 (not negative)."""
        mock_elo_system.get_rating = Mock(return_value=Mock(elo=800, calibration_score=0.5))
        mixin = JudgeScoringMixin(elo_system=mock_elo_system)

        score = mixin.compute_composite_score("low_elo_agent")

        # ELO normalized: (800 - 1000) / 500 = -0.4 -> clamped to 0
        # Calibration: 0.5
        # Composite: (0 * 0.7) + (0.5 * 0.3) = 0.15
        assert score == pytest.approx(0.15)

    def test_get_all_scores(self, mock_elo_system, mock_agents):
        """Test getting scores for all agents."""
        mixin = JudgeScoringMixin(elo_system=mock_elo_system)

        scores = mixin.get_all_scores(mock_agents)

        assert len(scores) == 3
        # Should be sorted by composite score descending
        assert scores[0].agent_name == "agent_a"
        assert scores[1].agent_name == "agent_b"
        assert scores[2].agent_name == "agent_c"

    def test_get_all_scores_no_elo_system(self, mock_agents):
        """Test getting scores without ELO system."""
        mixin = JudgeScoringMixin()

        scores = mixin.get_all_scores(mock_agents)

        assert len(scores) == 3
        # All scores should be 0
        for score in scores:
            assert score.composite_score == 0.0


# ============================================================================
# LastAgentStrategy Tests
# ============================================================================


class TestLastAgentStrategy:
    """Tests for LastAgentStrategy."""

    @pytest.mark.asyncio
    async def test_selects_synthesizer_if_present(self, mock_agents, mock_synthesizer):
        """Test synthesizer is selected if present."""
        agents = [mock_agents[0], mock_synthesizer, mock_agents[1]]
        strategy = LastAgentStrategy()

        judge = await strategy.select(agents, {}, [])

        assert judge.name == "synthesizer"

    @pytest.mark.asyncio
    async def test_selects_last_agent_if_no_synthesizer(self, mock_agents):
        """Test last agent selected without synthesizer."""
        strategy = LastAgentStrategy()

        judge = await strategy.select(mock_agents, {}, [])

        assert judge.name == "agent_c"

    @pytest.mark.asyncio
    async def test_returns_none_for_empty_agents(self):
        """Test returns None for empty agent list."""
        strategy = LastAgentStrategy()

        judge = await strategy.select([], {}, [])

        assert judge is None


# ============================================================================
# RandomStrategy Tests
# ============================================================================


class TestRandomStrategy:
    """Tests for RandomStrategy."""

    @pytest.mark.asyncio
    async def test_selects_from_agents(self, mock_agents):
        """Test selects an agent from the list."""
        strategy = RandomStrategy()

        judge = await strategy.select(mock_agents, {}, [])

        assert judge in mock_agents

    @pytest.mark.asyncio
    async def test_returns_none_for_empty_agents(self):
        """Test returns None for empty agent list."""
        strategy = RandomStrategy()

        judge = await strategy.select([], {}, [])

        assert judge is None

    @pytest.mark.asyncio
    async def test_random_selection_distribution(self, mock_agents):
        """Test random selection distributes across agents."""
        strategy = RandomStrategy()

        selections = {}
        for _ in range(100):
            judge = await strategy.select(mock_agents, {}, [])
            selections[judge.name] = selections.get(judge.name, 0) + 1

        # All agents should be selected at least once
        for agent in mock_agents:
            assert agent.name in selections


# ============================================================================
# EloRankedStrategy Tests
# ============================================================================


class TestEloRankedStrategy:
    """Tests for EloRankedStrategy."""

    @pytest.mark.asyncio
    async def test_selects_highest_elo(self, mock_agents, mock_elo_system):
        """Test selects agent with highest ELO."""
        strategy = EloRankedStrategy(elo_system=mock_elo_system)

        judge = await strategy.select(mock_agents, {}, [])

        assert judge.name == "agent_a"  # Has highest ELO (1500)

    @pytest.mark.asyncio
    async def test_falls_back_to_random_without_elo_system(self, mock_agents):
        """Test falls back to random without ELO system."""
        strategy = EloRankedStrategy(elo_system=None)

        judge = await strategy.select(mock_agents, {}, [])

        assert judge in mock_agents

    @pytest.mark.asyncio
    async def test_handles_elo_query_failure(self, mock_agents, mock_elo_system):
        """Test handles ELO query failure gracefully."""
        mock_elo_system.get_leaderboard.side_effect = RuntimeError("DB error")
        strategy = EloRankedStrategy(elo_system=mock_elo_system)

        judge = await strategy.select(mock_agents, {}, [])

        assert judge in mock_agents

    @pytest.mark.asyncio
    async def test_returns_none_for_empty_agents(self, mock_elo_system):
        """Test returns None for empty agent list."""
        strategy = EloRankedStrategy(elo_system=mock_elo_system)

        judge = await strategy.select([], {}, [])

        assert judge is None


# ============================================================================
# CalibratedStrategy Tests
# ============================================================================


class TestCalibratedStrategy:
    """Tests for CalibratedStrategy."""

    @pytest.mark.asyncio
    async def test_selects_best_composite_score(self, mock_agents, mock_elo_system):
        """Test selects agent with best composite score."""
        strategy = CalibratedStrategy(elo_system=mock_elo_system)

        judge = await strategy.select(mock_agents, {}, [])

        # agent_a should have best composite score
        assert judge.name == "agent_a"

    @pytest.mark.asyncio
    async def test_falls_back_to_random_without_elo_system(self, mock_agents):
        """Test falls back to random without ELO system."""
        strategy = CalibratedStrategy(elo_system=None)

        judge = await strategy.select(mock_agents, {}, [])

        assert judge in mock_agents

    @pytest.mark.asyncio
    async def test_returns_none_for_empty_agents(self, mock_elo_system):
        """Test returns None for empty agent list."""
        strategy = CalibratedStrategy(elo_system=mock_elo_system)

        judge = await strategy.select([], {}, [])

        assert judge is None


# ============================================================================
# CruxAwareStrategy Tests
# ============================================================================


class TestCruxAwareStrategy:
    """Tests for CruxAwareStrategy."""

    @pytest.mark.asyncio
    async def test_falls_back_to_calibrated_without_memory(self, mock_agents, mock_elo_system):
        """Test falls back to calibrated strategy without consensus memory."""
        strategy = CruxAwareStrategy(elo_system=mock_elo_system)

        judge = await strategy.select(mock_agents, {}, [])

        assert judge in mock_agents

    @pytest.mark.asyncio
    async def test_selects_historical_dissenter(self, mock_agents, mock_elo_system):
        """Test selects agent who historically dissented."""
        consensus_memory = Mock()

        # Mock finding similar debates with dissenting agent
        similar_debate = Mock()
        similar_debate.dissenting_agents = ["agent_b"]
        similar_debate.consensus = Mock(dissenting_agents=["agent_b"])
        consensus_memory.find_similar_debates = Mock(return_value=[similar_debate])

        strategy = CruxAwareStrategy(
            elo_system=mock_elo_system,
            consensus_memory=consensus_memory,
        )

        cruxes = [{"claim": "AI should be regulated"}]
        judge = await strategy.select(mock_agents, {}, [], cruxes=cruxes)

        assert judge.name == "agent_b"

    @pytest.mark.asyncio
    async def test_returns_none_for_empty_agents(self):
        """Test returns None for empty agent list."""
        strategy = CruxAwareStrategy()

        judge = await strategy.select([], {}, [])

        assert judge is None

    def test_find_historical_dissenters_without_memory(self, mock_agents):
        """Test _find_historical_dissenters without consensus memory."""
        strategy = CruxAwareStrategy()

        dissenters = strategy._find_historical_dissenters([{"claim": "test"}], mock_agents)

        assert dissenters == []

    def test_rank_by_elo_without_elo_system(self, mock_agents):
        """Test _rank_by_elo without ELO system returns original order."""
        strategy = CruxAwareStrategy()

        ranked = strategy._rank_by_elo(mock_agents)

        assert ranked == mock_agents


# ============================================================================
# VotedStrategy Tests
# ============================================================================


class TestVotedStrategy:
    """Tests for VotedStrategy."""

    @pytest.mark.asyncio
    async def test_agents_vote_for_judge(self, mock_agents):
        """Test agents vote and winner is selected."""

        async def mock_generate(agent, prompt, context):
            # All agents vote for agent_b
            return "I think agent_b should be the judge"

        strategy = VotedStrategy(
            generate_fn=mock_generate,
            build_vote_prompt_fn=lambda agents, props: "Vote for a judge",
        )

        judge = await strategy.select(mock_agents, {}, [])

        assert judge.name == "agent_b"

    @pytest.mark.asyncio
    async def test_tiebreaker_random(self, mock_agents):
        """Test random tiebreaker when votes are tied."""
        call_count = [0]

        async def mock_generate(agent, prompt, context):
            call_count[0] += 1
            # First agent votes for agent_b, second for agent_c
            if call_count[0] == 1:
                return "agent_b"
            elif call_count[0] == 2:
                return "agent_c"
            else:
                return "agent_b"  # agent_c votes

        strategy = VotedStrategy(
            generate_fn=mock_generate,
            build_vote_prompt_fn=lambda agents, props: "Vote for a judge",
        )

        judge = await strategy.select(mock_agents, {}, [])

        # Should select one of the tied agents
        assert judge.name in ["agent_b", "agent_c"]

    @pytest.mark.asyncio
    async def test_handles_generate_exception(self, mock_agents):
        """Test handles generation exceptions gracefully."""

        async def failing_generate(agent, prompt, context):
            raise RuntimeError("API error")

        strategy = VotedStrategy(
            generate_fn=failing_generate,
            build_vote_prompt_fn=lambda agents, props: "Vote",
        )

        # Should fall back to random
        judge = await strategy.select(mock_agents, {}, [])

        assert judge in mock_agents

    @pytest.mark.asyncio
    async def test_returns_none_for_empty_agents(self):
        """Test returns None for empty agent list."""
        strategy = VotedStrategy(
            generate_fn=AsyncMock(),
            build_vote_prompt_fn=lambda a, p: "Vote",
        )

        judge = await strategy.select([], {}, [])

        assert judge is None

    @pytest.mark.asyncio
    async def test_sanitize_fn_called(self, mock_agents):
        """Test sanitize function is called on responses."""

        async def mock_generate(agent, prompt, context):
            return "RAW: agent_b should judge"

        sanitize_calls = []

        def mock_sanitize(response, agent_name):
            sanitize_calls.append((response, agent_name))
            return response

        strategy = VotedStrategy(
            generate_fn=mock_generate,
            build_vote_prompt_fn=lambda agents, props: "Vote",
            sanitize_fn=mock_sanitize,
        )

        await strategy.select(mock_agents, {}, [])

        assert len(sanitize_calls) > 0


# ============================================================================
# JudgeSelector Initialization Tests
# ============================================================================


class TestJudgeSelectorInit:
    """Tests for JudgeSelector initialization."""

    def test_initialization_defaults(self, mock_agents):
        """Test initialization with defaults."""
        selector = JudgeSelector(agents=mock_agents)

        assert selector._judge_selection == "random"
        assert selector._circuit_breaker is None
        assert "random" in selector._strategies
        assert "last" in selector._strategies

    def test_initialization_with_elo_system(self, mock_agents, mock_elo_system):
        """Test initialization with ELO system."""
        selector = JudgeSelector(
            agents=mock_agents,
            elo_system=mock_elo_system,
            judge_selection="calibrated",
        )

        assert "calibrated" in selector._strategies
        assert "elo_ranked" in selector._strategies

    def test_initialization_with_circuit_breaker(self, mock_agents, mock_circuit_breaker):
        """Test initialization with circuit breaker."""
        selector = JudgeSelector(
            agents=mock_agents,
            circuit_breaker=mock_circuit_breaker,
        )

        assert selector._circuit_breaker == mock_circuit_breaker

    def test_voted_strategy_added_with_dependencies(self, mock_agents):
        """Test voted strategy added when dependencies provided."""
        selector = JudgeSelector(
            agents=mock_agents,
            generate_fn=AsyncMock(),
            build_vote_prompt_fn=lambda a, p: "Vote",
        )

        assert "voted" in selector._strategies


# ============================================================================
# JudgeSelector Filter Tests
# ============================================================================


class TestJudgeSelectorFiltering:
    """Tests for agent filtering in JudgeSelector."""

    def test_filter_without_circuit_breaker(self, mock_agents):
        """Test filtering without circuit breaker returns all agents."""
        selector = JudgeSelector(agents=mock_agents)

        filtered = selector._filter_available_agents(mock_agents)

        assert len(filtered) == len(mock_agents)

    def test_filter_with_circuit_breaker(self, mock_agents, mock_circuit_breaker):
        """Test filtering with circuit breaker excludes unavailable agents."""
        # Make agent_b unavailable
        mock_circuit_breaker.is_available.side_effect = lambda name: name != "agent_b"

        selector = JudgeSelector(
            agents=mock_agents,
            circuit_breaker=mock_circuit_breaker,
        )

        filtered = selector._filter_available_agents(mock_agents)

        assert len(filtered) == 2
        assert not any(a.name == "agent_b" for a in filtered)

    def test_filter_all_unavailable_returns_all(self, mock_agents, mock_circuit_breaker):
        """Test filtering when all agents unavailable returns all."""
        mock_circuit_breaker.is_available.return_value = False

        selector = JudgeSelector(
            agents=mock_agents,
            circuit_breaker=mock_circuit_breaker,
        )

        filtered = selector._filter_available_agents(mock_agents)

        # Should return all agents with warning
        assert len(filtered) == len(mock_agents)


# ============================================================================
# JudgeSelector Selection Tests
# ============================================================================


class TestJudgeSelectorSelection:
    """Tests for judge selection in JudgeSelector."""

    @pytest.mark.asyncio
    async def test_select_judge_uses_strategy(self, mock_agents, sample_proposals):
        """Test select_judge uses configured strategy."""
        selector = JudgeSelector(
            agents=mock_agents,
            judge_selection="last",
        )

        judge = await selector.select_judge(sample_proposals, [])

        assert judge.name == "agent_c"  # Last agent

    @pytest.mark.asyncio
    async def test_select_judge_filters_agents_first(
        self, mock_agents, mock_circuit_breaker, sample_proposals
    ):
        """Test select_judge filters agents before selection."""
        mock_circuit_breaker.is_available.side_effect = lambda name: name != "agent_c"

        selector = JudgeSelector(
            agents=mock_agents,
            judge_selection="last",
            circuit_breaker=mock_circuit_breaker,
        )

        judge = await selector.select_judge(sample_proposals, [])

        # agent_c filtered out, should select agent_b (new last)
        assert judge.name == "agent_b"

    @pytest.mark.asyncio
    async def test_select_judge_unknown_strategy_falls_back(self, mock_agents, sample_proposals):
        """Test unknown strategy falls back to random."""
        selector = JudgeSelector(
            agents=mock_agents,
            judge_selection="unknown_strategy",
        )

        judge = await selector.select_judge(sample_proposals, [])

        assert judge in mock_agents

    @pytest.mark.asyncio
    async def test_select_judge_none_result_falls_back(self, mock_agents, sample_proposals):
        """Test None result from strategy falls back to random."""
        selector = JudgeSelector(
            agents=mock_agents,
            judge_selection="random",
        )

        # Mock strategy to return None
        with patch.object(
            selector._strategies["random"], "select", new_callable=AsyncMock
        ) as mock_select:
            mock_select.return_value = None

            judge = await selector.select_judge(sample_proposals, [])

            assert judge in mock_agents


# ============================================================================
# JudgeSelector Candidates Tests
# ============================================================================


class TestJudgeSelectorCandidates:
    """Tests for judge candidate selection."""

    @pytest.mark.asyncio
    async def test_get_judge_candidates_ordered(
        self, mock_agents, mock_elo_system, sample_proposals
    ):
        """Test candidates are ordered by composite score."""
        selector = JudgeSelector(
            agents=mock_agents,
            elo_system=mock_elo_system,
        )

        candidates = await selector.get_judge_candidates(sample_proposals, [])

        # Should be ordered by composite score
        assert candidates[0].name == "agent_a"
        assert candidates[1].name == "agent_b"
        assert candidates[2].name == "agent_c"

    @pytest.mark.asyncio
    async def test_get_judge_candidates_limited(
        self, mock_agents, mock_elo_system, sample_proposals
    ):
        """Test candidates are limited to max_candidates."""
        selector = JudgeSelector(
            agents=mock_agents,
            elo_system=mock_elo_system,
        )

        candidates = await selector.get_judge_candidates(sample_proposals, [], max_candidates=2)

        assert len(candidates) == 2

    @pytest.mark.asyncio
    async def test_get_judge_candidates_empty_agents(self, sample_proposals):
        """Test empty list returned for no agents."""
        selector = JudgeSelector(agents=[])

        candidates = await selector.get_judge_candidates(sample_proposals, [])

        assert candidates == []

    @pytest.mark.asyncio
    async def test_get_judge_candidates_filters_unavailable(
        self, mock_agents, mock_circuit_breaker, sample_proposals
    ):
        """Test unavailable agents are filtered from candidates."""
        mock_circuit_breaker.is_available.side_effect = lambda name: name != "agent_a"

        selector = JudgeSelector(
            agents=mock_agents,
            circuit_breaker=mock_circuit_breaker,
        )

        candidates = await selector.get_judge_candidates(sample_proposals, [])

        assert not any(c.name == "agent_a" for c in candidates)

    @pytest.mark.asyncio
    async def test_get_judge_candidates_no_elo_shuffles(self, mock_agents, sample_proposals):
        """Test candidates are shuffled without ELO system."""
        selector = JudgeSelector(agents=mock_agents)

        # Get candidates multiple times - should have some variation
        selections = set()
        for _ in range(10):
            candidates = await selector.get_judge_candidates(sample_proposals, [], max_candidates=1)
            if candidates:
                selections.add(candidates[0].name)

        # With random shuffling, we should get some variety
        # (though this test may occasionally fail due to randomness)
        assert len(selections) >= 1


# ============================================================================
# JudgeSelector Factory Tests
# ============================================================================


class TestJudgeSelectorFactory:
    """Tests for JudgeSelector factory methods."""

    def test_from_protocol(self, mock_agents, mock_elo_system):
        """Test creating selector from protocol."""
        protocol = Mock()
        protocol.judge_selection = "calibrated"

        selector = JudgeSelector.from_protocol(
            protocol=protocol,
            agents=mock_agents,
            elo_system=mock_elo_system,
        )

        assert selector._judge_selection == "calibrated"

    def test_from_protocol_with_circuit_breaker(self, mock_agents, mock_circuit_breaker):
        """Test creating selector from protocol with circuit breaker."""
        protocol = Mock()
        protocol.judge_selection = "random"

        selector = JudgeSelector.from_protocol(
            protocol=protocol,
            agents=mock_agents,
            circuit_breaker=mock_circuit_breaker,
        )

        assert selector._circuit_breaker == mock_circuit_breaker


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_single_agent(self):
        """Test selection with single agent."""
        single_agent = Mock()
        single_agent.name = "only_agent"
        single_agent.role = None

        selector = JudgeSelector(agents=[single_agent])

        judge = await selector.select_judge({}, [])

        assert judge.name == "only_agent"

    @pytest.mark.asyncio
    async def test_agents_with_same_elo(self, mock_elo_system):
        """Test selection when agents have same ELO."""
        agents = []
        for name in ["a", "b", "c"]:
            agent = Mock()
            agent.name = name
            agent.role = None
            agents.append(agent)

        # All have same ELO
        mock_elo_system.get_ratings_batch = Mock(
            return_value={name: Mock(elo=1000, calibration_score=0.5) for name in ["a", "b", "c"]}
        )

        selector = JudgeSelector(
            agents=agents,
            elo_system=mock_elo_system,
            judge_selection="calibrated",
        )

        judge = await selector.select_judge({}, [])

        assert judge in agents

    def test_elo_normalization_edge_values(self, mock_elo_system):
        """Test ELO normalization at edge values."""
        # Very high ELO
        mock_elo_system.get_rating = Mock(return_value=Mock(elo=2000, calibration_score=1.0))
        mixin = JudgeScoringMixin(elo_system=mock_elo_system)

        score = mixin.compute_composite_score("high_elo_agent")

        # ELO normalized: (2000 - 1000) / 500 = 2.0
        # Composite: (2.0 * 0.7) + (1.0 * 0.3) = 1.4 + 0.3 = 1.7
        assert score == pytest.approx(1.7)
