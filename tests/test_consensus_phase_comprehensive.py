"""
Consensus Phase Comprehensive Tests.

Tests for the consensus resolution phase of debate orchestration.
Covers:
- Consensus modes (none, majority, unanimous, judge)
- Vote collection and weighting
- Calibration adjustments
- Fallback mechanisms
- Error handling and timeouts
"""

from __future__ import annotations

import asyncio
import pytest
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch


# =============================================================================
# Mock Classes
# =============================================================================


@dataclass
class MockVote:
    """Mock vote for testing."""

    agent: str
    choice: str
    confidence: float = 0.8
    reasoning: str = "Test reasoning"
    continue_debate: bool = False


@dataclass
class MockAgent:
    """Mock agent for testing."""

    name: str


@dataclass
class MockProtocol:
    """Mock debate protocol."""

    consensus: str = "majority"
    consensus_threshold: float = 0.5
    consensus_timeout: float = 120.0
    rounds: int = 3
    judge_selection: str = "random"
    user_vote_weight: float = 0.5
    verify_claims_during_consensus: bool = False
    verification_weight_bonus: float = 0.2
    verification_timeout_seconds: float = 5.0


@dataclass
class MockEnvironment:
    """Mock environment."""

    task: str = "Test debate task"


@dataclass
class MockDebateResult:
    """Mock debate result."""

    id: str = "test-debate-123"
    final_answer: str = ""
    consensus_reached: bool = False
    confidence: float = 0.0
    winner: Optional[str] = None
    consensus_strength: str = ""
    consensus_variance: float = 0.0
    votes: list = field(default_factory=list)
    critiques: list = field(default_factory=list)
    messages: list = field(default_factory=list)
    dissenting_views: list = field(default_factory=list)
    rounds_used: int = 3
    verification_results: dict = field(default_factory=dict)
    verification_bonuses: dict = field(default_factory=dict)
    debate_cruxes: list = field(default_factory=list)
    evidence_suggestions: list = field(default_factory=list)


@dataclass
class MockDebateContext:
    """Mock debate context."""

    result: MockDebateResult = field(default_factory=MockDebateResult)
    proposals: dict = field(default_factory=dict)
    agents: list = field(default_factory=list)
    context_messages: list = field(default_factory=list)
    env: MockEnvironment = field(default_factory=MockEnvironment)
    vote_tally: dict = field(default_factory=dict)
    winner_agent: Optional[str] = None
    loop_id: str = "test-loop"
    debate_id: str = "test-debate"
    event_emitter: Any = None


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_agents():
    """Create mock agents."""
    return [
        MockAgent("claude"),
        MockAgent("gpt4"),
        MockAgent("gemini"),
    ]


@pytest.fixture
def mock_proposals():
    """Create mock proposals."""
    return {
        "claude": "Claude's proposal for the solution.",
        "gpt4": "GPT-4's proposal for the solution.",
        "gemini": "Gemini's proposal for the solution.",
    }


@pytest.fixture
def mock_context(mock_agents, mock_proposals):
    """Create mock debate context."""
    return MockDebateContext(
        proposals=mock_proposals,
        agents=mock_agents,
    )


@pytest.fixture
def mock_protocol():
    """Create mock protocol."""
    return MockProtocol()


@pytest.fixture
def consensus_phase(mock_protocol):
    """Create consensus phase with minimal dependencies."""
    from aragora.debate.phases.consensus_phase import (
        ConsensusPhase,
        ConsensusDependencies,
    )

    deps = ConsensusDependencies(
        protocol=mock_protocol,
    )
    return ConsensusPhase(deps=deps)


# =============================================================================
# ConsensusDependencies Tests
# =============================================================================


class TestConsensusDependencies:
    """Tests for ConsensusDependencies dataclass."""

    def test_dependencies_default_values(self):
        """Test ConsensusDependencies has correct defaults."""
        from aragora.debate.phases.consensus_phase import ConsensusDependencies

        deps = ConsensusDependencies()
        assert deps.protocol is None
        assert deps.elo_system is None
        assert deps.memory is None
        assert deps.agent_weights == {}
        assert deps.hooks == {}
        assert deps.user_votes == []

    def test_dependencies_with_values(self):
        """Test ConsensusDependencies accepts custom values."""
        from aragora.debate.phases.consensus_phase import ConsensusDependencies

        mock_protocol = MockProtocol()
        mock_weights = {"claude": 1.2, "gpt4": 0.8}

        deps = ConsensusDependencies(
            protocol=mock_protocol,
            agent_weights=mock_weights,
            user_votes=[{"choice": "claude"}],
        )

        assert deps.protocol is mock_protocol
        assert deps.agent_weights == mock_weights
        assert len(deps.user_votes) == 1


class TestConsensusCallbacks:
    """Tests for ConsensusCallbacks dataclass."""

    def test_callbacks_default_values(self):
        """Test ConsensusCallbacks has all None defaults."""
        from aragora.debate.phases.consensus_phase import ConsensusCallbacks

        callbacks = ConsensusCallbacks()
        assert callbacks.vote_with_agent is None
        assert callbacks.with_timeout is None
        assert callbacks.select_judge is None
        assert callbacks.generate_with_agent is None
        assert callbacks.group_similar_votes is None

    def test_callbacks_with_functions(self):
        """Test ConsensusCallbacks accepts functions."""
        from aragora.debate.phases.consensus_phase import ConsensusCallbacks

        async def mock_vote_fn():
            pass

        callbacks = ConsensusCallbacks(
            vote_with_agent=mock_vote_fn,
        )
        assert callbacks.vote_with_agent is mock_vote_fn


# =============================================================================
# Initialization Tests
# =============================================================================


class TestConsensusPhaseInit:
    """Tests for ConsensusPhase initialization."""

    def test_init_with_dependencies_dataclass(self):
        """Test initialization with ConsensusDependencies dataclass."""
        from aragora.debate.phases.consensus_phase import (
            ConsensusPhase,
            ConsensusDependencies,
        )

        protocol = MockProtocol()
        deps = ConsensusDependencies(
            protocol=protocol,
            agent_weights={"claude": 1.0},
        )

        phase = ConsensusPhase(deps=deps)
        assert phase.protocol is protocol
        assert phase.agent_weights == {"claude": 1.0}

    def test_init_with_legacy_parameters(self):
        """Test backward-compatible initialization with keyword args."""
        from aragora.debate.phases.consensus_phase import ConsensusPhase

        protocol = MockProtocol()
        phase = ConsensusPhase(
            protocol=protocol,
            agent_weights={"claude": 1.0},
        )

        assert phase.protocol is protocol
        assert phase.agent_weights == {"claude": 1.0}

    def test_init_with_callbacks_dataclass(self):
        """Test initialization with ConsensusCallbacks dataclass."""
        from aragora.debate.phases.consensus_phase import (
            ConsensusPhase,
            ConsensusDependencies,
            ConsensusCallbacks,
        )

        async def mock_vote():
            pass

        deps = ConsensusDependencies()
        callbacks = ConsensusCallbacks(vote_with_agent=mock_vote)

        phase = ConsensusPhase(deps=deps, callbacks=callbacks)
        assert phase._vote_with_agent is mock_vote


# =============================================================================
# None Consensus Tests
# =============================================================================


class TestNoneConsensus:
    """Tests for 'none' consensus mode."""

    @pytest.mark.asyncio
    async def test_none_consensus_combines_proposals(self, consensus_phase, mock_context):
        """Test none mode combines all proposals."""
        await consensus_phase._handle_none_consensus(mock_context)

        assert "[claude]:" in mock_context.result.final_answer.lower()
        assert "[gpt4]:" in mock_context.result.final_answer.lower()
        assert "[gemini]:" in mock_context.result.final_answer.lower()
        assert mock_context.result.consensus_reached is False
        assert mock_context.result.confidence == 0.5

    @pytest.mark.asyncio
    async def test_none_consensus_empty_proposals(self, consensus_phase):
        """Test none mode with empty proposals."""
        ctx = MockDebateContext(proposals={})
        await consensus_phase._handle_none_consensus(ctx)

        assert ctx.result.final_answer == ""
        assert ctx.result.consensus_reached is False

    @pytest.mark.asyncio
    async def test_none_consensus_single_proposal(self, consensus_phase):
        """Test none mode with single proposal."""
        ctx = MockDebateContext(proposals={"claude": "Only proposal"})
        await consensus_phase._handle_none_consensus(ctx)

        assert "Only proposal" in ctx.result.final_answer
        assert "[claude]:" in ctx.result.final_answer.lower()


# =============================================================================
# Majority Consensus Tests
# =============================================================================


class TestMajorityConsensus:
    """Tests for 'majority' consensus mode."""

    @pytest.mark.asyncio
    async def test_majority_consensus_with_clear_winner(self, mock_context, mock_agents):
        """Test majority consensus with clear winner."""
        from aragora.debate.phases.consensus_phase import (
            ConsensusPhase,
            ConsensusDependencies,
            ConsensusCallbacks,
        )

        protocol = MockProtocol(consensus="majority")

        # Mock vote collection to return votes for claude
        async def mock_vote(agent, proposals, task):
            return MockVote(agent=agent.name, choice="claude", confidence=0.9)

        deps = ConsensusDependencies(protocol=protocol)
        callbacks = ConsensusCallbacks(vote_with_agent=mock_vote)
        phase = ConsensusPhase(deps=deps, callbacks=callbacks)

        mock_context.agents = mock_agents
        await phase._handle_majority_consensus(mock_context)

        assert mock_context.result.winner == "claude"
        assert mock_context.result.consensus_reached is True
        assert mock_context.result.confidence >= 0.5

    @pytest.mark.asyncio
    async def test_majority_consensus_no_votes(self, mock_context):
        """Test majority consensus when no votes collected."""
        from aragora.debate.phases.consensus_phase import (
            ConsensusPhase,
            ConsensusDependencies,
        )

        protocol = MockProtocol(consensus="majority")
        deps = ConsensusDependencies(protocol=protocol)
        # No vote callback means no votes
        phase = ConsensusPhase(deps=deps)

        await phase._handle_majority_consensus(mock_context)

        # Should fall back to first proposal
        assert mock_context.result.consensus_reached is False


# =============================================================================
# Unanimous Consensus Tests
# =============================================================================


class TestUnanimousConsensus:
    """Tests for 'unanimous' consensus mode."""

    @pytest.mark.asyncio
    async def test_unanimous_all_agree(self, mock_context, mock_agents):
        """Test unanimous consensus when all agents agree."""
        from aragora.debate.phases.consensus_phase import (
            ConsensusPhase,
            ConsensusDependencies,
            ConsensusCallbacks,
        )

        protocol = MockProtocol(consensus="unanimous")

        async def mock_vote(agent, proposals, task):
            return MockVote(agent=agent.name, choice="claude", confidence=0.95)

        deps = ConsensusDependencies(protocol=protocol)
        callbacks = ConsensusCallbacks(vote_with_agent=mock_vote)
        phase = ConsensusPhase(deps=deps, callbacks=callbacks)

        mock_context.agents = mock_agents
        await phase._handle_unanimous_consensus(mock_context)

        assert mock_context.result.consensus_reached is True
        assert mock_context.result.confidence >= 1.0
        assert mock_context.result.consensus_strength == "unanimous"

    @pytest.mark.asyncio
    async def test_unanimous_not_reached(self, mock_context, mock_agents):
        """Test unanimous consensus when agents disagree."""
        from aragora.debate.phases.consensus_phase import (
            ConsensusPhase,
            ConsensusDependencies,
            ConsensusCallbacks,
        )

        protocol = MockProtocol(consensus="unanimous")

        vote_choices = ["claude", "gpt4", "claude"]

        async def mock_vote(agent, proposals, task):
            idx = mock_agents.index(agent) if agent in mock_agents else 0
            return MockVote(agent=agent.name, choice=vote_choices[idx], confidence=0.8)

        deps = ConsensusDependencies(protocol=protocol)
        callbacks = ConsensusCallbacks(vote_with_agent=mock_vote)
        phase = ConsensusPhase(deps=deps, callbacks=callbacks)

        mock_context.agents = mock_agents
        await phase._handle_unanimous_consensus(mock_context)

        assert mock_context.result.consensus_reached is False
        assert mock_context.result.consensus_strength == "none"
        assert "No unanimous consensus" in mock_context.result.final_answer


# =============================================================================
# Judge Consensus Tests
# =============================================================================


class TestJudgeConsensus:
    """Tests for 'judge' consensus mode."""

    @pytest.mark.asyncio
    async def test_judge_consensus_synthesis(self, mock_context, mock_agents):
        """Test judge consensus with successful synthesis."""
        from aragora.debate.phases.consensus_phase import (
            ConsensusPhase,
            ConsensusDependencies,
            ConsensusCallbacks,
        )

        protocol = MockProtocol(consensus="judge")

        async def mock_select_judge(proposals, context_messages):
            return mock_agents[0]  # claude as judge

        async def mock_generate(agent, prompt, context):
            return "Judge's synthesized answer based on all proposals."

        deps = ConsensusDependencies(protocol=protocol)
        callbacks = ConsensusCallbacks(
            select_judge=mock_select_judge,
            generate_with_agent=mock_generate,
        )
        phase = ConsensusPhase(deps=deps, callbacks=callbacks)

        mock_context.agents = mock_agents
        await phase._handle_judge_consensus(mock_context)

        assert mock_context.result.consensus_reached is True
        assert mock_context.result.confidence == 0.8
        assert "synthesized" in mock_context.result.final_answer.lower()
        assert mock_context.result.winner == "claude"

    @pytest.mark.asyncio
    async def test_judge_consensus_missing_callbacks(self, mock_context):
        """Test judge consensus fails gracefully without required callbacks."""
        from aragora.debate.phases.consensus_phase import (
            ConsensusPhase,
            ConsensusDependencies,
        )

        protocol = MockProtocol(consensus="judge")
        deps = ConsensusDependencies(protocol=protocol)
        phase = ConsensusPhase(deps=deps)

        mock_context.proposals = {"claude": "Test proposal"}
        await phase._handle_judge_consensus(mock_context)

        # Should fall back to first proposal
        assert mock_context.result.consensus_reached is False


# =============================================================================
# Fallback Consensus Tests
# =============================================================================


class TestFallbackConsensus:
    """Tests for fallback consensus mechanisms."""

    @pytest.mark.asyncio
    async def test_fallback_from_votes(self, mock_context):
        """Test fallback uses existing votes to determine winner."""
        from aragora.debate.phases.consensus_phase import (
            ConsensusPhase,
            ConsensusDependencies,
        )

        deps = ConsensusDependencies()
        phase = ConsensusPhase(deps=deps)

        # Add some votes to result
        mock_context.result.votes = [
            MockVote("agent1", "claude", 0.8),
            MockVote("agent2", "claude", 0.9),
            MockVote("agent3", "gpt4", 0.7),
        ]

        await phase._handle_fallback_consensus(mock_context, reason="timeout")

        assert mock_context.result.winner == "claude"
        assert mock_context.result.consensus_reached is True
        assert mock_context.result.consensus_strength == "fallback"

    @pytest.mark.asyncio
    async def test_fallback_from_vote_tally(self, mock_context):
        """Test fallback uses vote_tally when available."""
        from aragora.debate.phases.consensus_phase import (
            ConsensusPhase,
            ConsensusDependencies,
        )

        deps = ConsensusDependencies()
        phase = ConsensusPhase(deps=deps)

        mock_context.vote_tally = {"claude": 3, "gpt4": 1}

        await phase._handle_fallback_consensus(mock_context, reason="error")

        assert mock_context.result.winner == "claude"
        assert mock_context.result.consensus_strength == "fallback"

    @pytest.mark.asyncio
    async def test_fallback_no_votes(self, mock_context):
        """Test fallback combines proposals when no votes available."""
        from aragora.debate.phases.consensus_phase import (
            ConsensusPhase,
            ConsensusDependencies,
        )

        deps = ConsensusDependencies()
        phase = ConsensusPhase(deps=deps)

        await phase._handle_fallback_consensus(mock_context, reason="timeout")

        assert mock_context.result.consensus_reached is False
        assert "fallback" in mock_context.result.final_answer.lower()
        assert mock_context.result.confidence == 0.5


# =============================================================================
# Vote Collection Tests
# =============================================================================


class TestVoteCollection:
    """Tests for vote collection logic."""

    @pytest.mark.asyncio
    async def test_collect_votes_success(self, mock_context, mock_agents):
        """Test successful vote collection from all agents."""
        from aragora.debate.phases.consensus_phase import (
            ConsensusPhase,
            ConsensusDependencies,
            ConsensusCallbacks,
        )

        async def mock_vote(agent, proposals, task):
            return MockVote(agent=agent.name, choice="claude", confidence=0.8)

        deps = ConsensusDependencies()
        callbacks = ConsensusCallbacks(vote_with_agent=mock_vote)
        phase = ConsensusPhase(deps=deps, callbacks=callbacks)

        mock_context.agents = mock_agents
        votes = await phase._collect_votes(mock_context)

        assert len(votes) == 3

    @pytest.mark.asyncio
    async def test_collect_votes_partial_failure(self, mock_context, mock_agents):
        """Test vote collection with some agent failures."""
        from aragora.debate.phases.consensus_phase import (
            ConsensusPhase,
            ConsensusDependencies,
            ConsensusCallbacks,
        )

        call_count = 0

        async def mock_vote(agent, proposals, task):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise Exception("Agent failed")
            return MockVote(agent=agent.name, choice="claude", confidence=0.8)

        deps = ConsensusDependencies()
        callbacks = ConsensusCallbacks(vote_with_agent=mock_vote)
        phase = ConsensusPhase(deps=deps, callbacks=callbacks)

        mock_context.agents = mock_agents
        votes = await phase._collect_votes(mock_context)

        # Should have 2 votes (one failed)
        assert len(votes) == 2

    @pytest.mark.asyncio
    async def test_collect_votes_no_callback(self, mock_context, mock_agents):
        """Test vote collection returns empty list without callback."""
        from aragora.debate.phases.consensus_phase import (
            ConsensusPhase,
            ConsensusDependencies,
        )

        deps = ConsensusDependencies()
        phase = ConsensusPhase(deps=deps)

        mock_context.agents = mock_agents
        votes = await phase._collect_votes(mock_context)

        assert len(votes) == 0


class TestVoteCollectionWithErrors:
    """Tests for vote collection with error tracking."""

    @pytest.mark.asyncio
    async def test_collect_votes_with_errors_tracking(self, mock_context, mock_agents):
        """Test vote collection tracks errors for unanimity mode."""
        from aragora.debate.phases.consensus_phase import (
            ConsensusPhase,
            ConsensusDependencies,
            ConsensusCallbacks,
        )

        call_count = 0

        async def mock_vote(agent, proposals, task):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise Exception("Agent failed")
            return MockVote(agent=agent.name, choice="claude", confidence=0.8)

        deps = ConsensusDependencies()
        callbacks = ConsensusCallbacks(vote_with_agent=mock_vote)
        phase = ConsensusPhase(deps=deps, callbacks=callbacks)

        mock_context.agents = mock_agents
        votes, errors = await phase._collect_votes_with_errors(mock_context)

        assert len(votes) == 2
        assert errors == 1


# =============================================================================
# Vote Grouping Tests
# =============================================================================


class TestVoteGrouping:
    """Tests for vote grouping logic."""

    def test_compute_vote_groups_no_grouping(self, consensus_phase):
        """Test vote groups without grouping callback."""
        votes = [
            MockVote("agent1", "claude"),
            MockVote("agent2", "gpt4"),
        ]

        groups, mapping = consensus_phase._compute_vote_groups(votes)

        assert "claude" in groups
        assert "gpt4" in groups
        assert mapping["claude"] == "claude"
        assert mapping["gpt4"] == "gpt4"

    def test_compute_vote_groups_with_grouping(self):
        """Test vote groups with custom grouping callback."""
        from aragora.debate.phases.consensus_phase import (
            ConsensusPhase,
            ConsensusDependencies,
            ConsensusCallbacks,
        )

        def mock_group(votes):
            # Group "Claude" and "claude-visionary" together
            return {"claude": ["claude", "claude-visionary"]}

        deps = ConsensusDependencies()
        callbacks = ConsensusCallbacks(group_similar_votes=mock_group)
        phase = ConsensusPhase(deps=deps, callbacks=callbacks)

        votes = [
            MockVote("agent1", "claude"),
            MockVote("agent2", "claude-visionary"),
        ]

        groups, mapping = phase._compute_vote_groups(votes)

        assert "claude" in groups
        assert mapping.get("claude-visionary") == "claude"


# =============================================================================
# Vote Weighting Tests
# =============================================================================


class TestVoteWeighting:
    """Tests for vote weight computation."""

    def test_compute_vote_weights_default(self, consensus_phase, mock_context, mock_agents):
        """Test default vote weights are 1.0."""
        mock_context.agents = mock_agents
        weights = consensus_phase._compute_vote_weights(mock_context)

        for agent in mock_agents:
            assert weights.get(agent.name, 1.0) == 1.0

    def test_compute_vote_weights_with_agent_weights(self, mock_context, mock_agents):
        """Test vote weights use agent_weights."""
        from aragora.debate.phases.consensus_phase import (
            ConsensusPhase,
            ConsensusDependencies,
        )

        deps = ConsensusDependencies(agent_weights={"claude": 1.5, "gpt4": 0.8})
        phase = ConsensusPhase(deps=deps)

        mock_context.agents = mock_agents
        weights = phase._compute_vote_weights(mock_context)

        assert weights["claude"] == 1.5
        assert weights["gpt4"] == 0.8
        assert weights["gemini"] == 1.0  # Default

    def test_count_weighted_votes(self, consensus_phase):
        """Test weighted vote counting."""
        votes = [
            MockVote("claude", "proposal_a"),
            MockVote("gpt4", "proposal_a"),
            MockVote("gemini", "proposal_b"),
        ]
        choice_mapping = {"proposal_a": "proposal_a", "proposal_b": "proposal_b"}
        weight_cache = {"claude": 1.5, "gpt4": 1.0, "gemini": 0.5}

        counts, total = consensus_phase._count_weighted_votes(votes, choice_mapping, weight_cache)

        assert counts["proposal_a"] == 2.5  # 1.5 + 1.0
        assert counts["proposal_b"] == 0.5
        assert total == 3.0


# =============================================================================
# User Votes Tests
# =============================================================================


class TestUserVotes:
    """Tests for user vote handling."""

    def test_add_user_votes(self):
        """Test adding user votes to counts."""
        from aragora.debate.phases.consensus_phase import (
            ConsensusPhase,
            ConsensusDependencies,
        )

        protocol = MockProtocol(user_vote_weight=0.5)
        deps = ConsensusDependencies(
            protocol=protocol,
            user_votes=[
                {"choice": "claude", "intensity": 5, "user_id": "user1"},
                {"choice": "gpt4", "intensity": 10, "user_id": "user2"},
            ],
        )
        phase = ConsensusPhase(deps=deps)

        vote_counts = Counter({"claude": 2.0, "gpt4": 1.0})
        total_weighted = 3.0
        choice_mapping = {"claude": "claude", "gpt4": "gpt4"}

        new_counts, new_total = phase._add_user_votes(vote_counts, total_weighted, choice_mapping)

        # User votes should be added
        assert new_counts["claude"] > 2.0
        assert new_total > 3.0


# =============================================================================
# Choice Normalization Tests
# =============================================================================


class TestChoiceNormalization:
    """Tests for vote choice normalization."""

    def test_normalize_exact_match(self, consensus_phase, mock_agents, mock_proposals):
        """Test normalization with exact match."""
        result = consensus_phase._normalize_choice_to_agent("claude", mock_agents, mock_proposals)
        assert result == "claude"

    def test_normalize_case_insensitive(self, consensus_phase, mock_agents, mock_proposals):
        """Test normalization is case-insensitive."""
        result = consensus_phase._normalize_choice_to_agent("CLAUDE", mock_agents, mock_proposals)
        assert result == "claude"

    def test_normalize_prefix_match(self, consensus_phase):
        """Test normalization with prefix matching."""
        agents = [MockAgent("claude-visionary")]
        # Use proposals without "claude" key to test agent prefix matching
        proposals = {"gemini": "Gemini proposal", "gpt4": "GPT-4 proposal"}
        result = consensus_phase._normalize_choice_to_agent("claude", agents, proposals)
        assert result == "claude-visionary"

    def test_normalize_no_match(self, consensus_phase, mock_agents, mock_proposals):
        """Test normalization returns original when no match."""
        result = consensus_phase._normalize_choice_to_agent(
            "unknown-agent", mock_agents, mock_proposals
        )
        assert result == "unknown-agent"

    def test_normalize_empty_choice(self, consensus_phase, mock_agents, mock_proposals):
        """Test normalization with empty choice."""
        result = consensus_phase._normalize_choice_to_agent("", mock_agents, mock_proposals)
        assert result == ""


# =============================================================================
# Winner Determination Tests
# =============================================================================


class TestWinnerDetermination:
    """Tests for winner determination logic."""

    def test_determine_majority_winner(self, mock_context, mock_agents):
        """Test majority winner determination."""
        from aragora.debate.phases.consensus_phase import (
            ConsensusPhase,
            ConsensusDependencies,
        )

        protocol = MockProtocol(consensus_threshold=0.5)
        deps = ConsensusDependencies(protocol=protocol)
        phase = ConsensusPhase(deps=deps)

        mock_context.agents = mock_agents
        vote_counts = Counter({"claude": 2.0, "gpt4": 1.0})
        total_votes = 3.0
        choice_mapping = {"claude": "claude", "gpt4": "gpt4"}

        phase._determine_majority_winner(mock_context, vote_counts, total_votes, choice_mapping)

        assert mock_context.result.winner == "claude"
        assert mock_context.result.consensus_reached is True
        assert mock_context.result.confidence == pytest.approx(2 / 3, rel=0.01)

    def test_determine_majority_winner_no_votes(self, mock_context, mock_agents):
        """Test winner determination with empty vote counts."""
        from aragora.debate.phases.consensus_phase import (
            ConsensusPhase,
            ConsensusDependencies,
        )

        deps = ConsensusDependencies()
        phase = ConsensusPhase(deps=deps)

        mock_context.agents = mock_agents
        vote_counts = Counter()
        total_votes = 0.0
        choice_mapping = {}

        phase._determine_majority_winner(mock_context, vote_counts, total_votes, choice_mapping)

        assert mock_context.result.consensus_reached is False
        assert mock_context.result.confidence == 0.5


# =============================================================================
# Unanimous Winner Tests
# =============================================================================


class TestUnanimousWinner:
    """Tests for unanimous winner setting."""

    def test_set_unanimous_winner(self, mock_context):
        """Test setting unanimous winner."""
        from aragora.debate.phases.consensus_phase import (
            ConsensusPhase,
            ConsensusDependencies,
        )

        deps = ConsensusDependencies()
        phase = ConsensusPhase(deps=deps)

        phase._set_unanimous_winner(
            mock_context,
            winner="claude",
            unanimity_ratio=1.0,
            total_voters=3,
            count=3,
        )

        assert mock_context.result.winner == "claude"
        assert mock_context.result.consensus_reached is True
        assert mock_context.result.consensus_strength == "unanimous"
        assert mock_context.result.consensus_variance == 0.0

    def test_set_no_unanimity(self, mock_context):
        """Test setting result when unanimity not reached."""
        from aragora.debate.phases.consensus_phase import (
            ConsensusPhase,
            ConsensusDependencies,
        )

        deps = ConsensusDependencies()
        phase = ConsensusPhase(deps=deps)

        phase._set_no_unanimity(
            mock_context,
            winner="claude",
            unanimity_ratio=0.67,
            total_voters=3,
            count=2,
            choice_mapping={},
        )

        assert mock_context.result.consensus_reached is False
        assert mock_context.result.consensus_strength == "none"
        assert "No unanimous consensus" in mock_context.result.final_answer


# =============================================================================
# Execute Tests
# =============================================================================


class TestConsensusExecute:
    """Tests for main execute method."""

    @pytest.mark.asyncio
    async def test_execute_none_mode(self, mock_context):
        """Test execute routes to none mode."""
        from aragora.debate.phases.consensus_phase import (
            ConsensusPhase,
            ConsensusDependencies,
        )

        protocol = MockProtocol(consensus="none")
        deps = ConsensusDependencies(protocol=protocol)
        phase = ConsensusPhase(deps=deps)

        await phase.execute(mock_context)

        assert mock_context.result.consensus_reached is False

    @pytest.mark.asyncio
    async def test_execute_unknown_mode_fallback(self, mock_context):
        """Test execute handles unknown mode."""
        from aragora.debate.phases.consensus_phase import (
            ConsensusPhase,
            ConsensusDependencies,
        )

        protocol = MockProtocol(consensus="unknown_mode")
        deps = ConsensusDependencies(protocol=protocol)
        phase = ConsensusPhase(deps=deps)

        await phase.execute(mock_context)

        # Should fall back to none mode
        assert mock_context.result.consensus_reached is False

    @pytest.mark.asyncio
    async def test_execute_timeout_fallback(self, mock_context):
        """Test execute handles timeout with fallback."""
        from aragora.debate.phases.consensus_phase import (
            ConsensusPhase,
            ConsensusDependencies,
        )

        protocol = MockProtocol(consensus="majority", consensus_timeout=0.001)
        deps = ConsensusDependencies(protocol=protocol)
        phase = ConsensusPhase(deps=deps)

        # Make _execute_consensus take too long
        original_execute = phase._execute_consensus

        async def slow_execute(ctx, mode):
            await asyncio.sleep(1)
            await original_execute(ctx, mode)

        phase._execute_consensus = slow_execute

        await phase.execute(mock_context)

        # Should have fallback result
        assert mock_context.result.consensus_strength == "fallback"


# =============================================================================
# Calibration Tests
# =============================================================================


class TestCalibrationAdjustment:
    """Tests for calibration-based confidence adjustment."""

    def test_apply_calibration_no_tracker(self, consensus_phase):
        """Test calibration does nothing without tracker."""
        votes = [MockVote("agent1", "claude", 0.8)]
        result = consensus_phase._apply_calibration_to_votes(votes, MockDebateContext())
        assert result == votes

    def test_apply_calibration_exception_handling(self):
        """Test calibration handles errors gracefully."""
        from aragora.debate.phases.consensus_phase import (
            ConsensusPhase,
            ConsensusDependencies,
        )

        mock_tracker = MagicMock()
        mock_tracker.get_calibration_summary.side_effect = Exception("Tracker error")

        deps = ConsensusDependencies(calibration_tracker=mock_tracker)
        phase = ConsensusPhase(deps=deps)

        votes = [MockVote("agent1", "claude", 0.8)]
        result = phase._apply_calibration_to_votes(votes, MockDebateContext())

        # Should return original votes on error
        assert len(result) == 1


# =============================================================================
# Vote Success Handling Tests
# =============================================================================


class TestVoteSuccessHandling:
    """Tests for vote success handling."""

    def test_handle_vote_success_notifications(self, mock_context):
        """Test vote success triggers notifications."""
        from aragora.debate.phases.consensus_phase import (
            ConsensusPhase,
            ConsensusDependencies,
            ConsensusCallbacks,
        )

        notify_calls = []

        def mock_notify(event_type, **kwargs):
            notify_calls.append((event_type, kwargs))

        deps = ConsensusDependencies()
        callbacks = ConsensusCallbacks(notify_spectator=mock_notify)
        phase = ConsensusPhase(deps=deps, callbacks=callbacks)

        agent = MockAgent("claude")
        vote = MockVote("claude", "proposal_a", 0.9)

        phase._handle_vote_success(mock_context, agent, vote)

        assert len(notify_calls) == 1
        assert notify_calls[0][0] == "vote"

    def test_handle_vote_success_hooks(self, mock_context):
        """Test vote success triggers hooks."""
        from aragora.debate.phases.consensus_phase import (
            ConsensusPhase,
            ConsensusDependencies,
        )

        hook_calls = []

        def on_vote(agent_name, choice, confidence):
            hook_calls.append((agent_name, choice, confidence))

        deps = ConsensusDependencies(hooks={"on_vote": on_vote})
        phase = ConsensusPhase(deps=deps)

        agent = MockAgent("claude")
        vote = MockVote("claude", "proposal_a", 0.9)

        phase._handle_vote_success(mock_context, agent, vote)

        assert len(hook_calls) == 1
        assert hook_calls[0] == ("claude", "proposal_a", 0.9)


# =============================================================================
# Verification Bonus Tests
# =============================================================================


class TestVerificationBonus:
    """Tests for verification bonus application."""

    @pytest.mark.asyncio
    async def test_verification_bonus_disabled(self, mock_context):
        """Test verification bonus skipped when disabled."""
        from aragora.debate.phases.consensus_phase import (
            ConsensusPhase,
            ConsensusDependencies,
        )

        protocol = MockProtocol(verify_claims_during_consensus=False)
        deps = ConsensusDependencies(protocol=protocol)
        phase = ConsensusPhase(deps=deps)

        vote_counts = Counter({"claude": 2.0})
        proposals = {"claude": "Test proposal"}
        choice_mapping = {"claude": "claude"}

        result = await phase._apply_verification_bonuses(
            mock_context, vote_counts, proposals, choice_mapping
        )

        assert result["claude"] == 2.0  # Unchanged

    @pytest.mark.asyncio
    async def test_verification_bonus_applied(self, mock_context):
        """Test verification bonus is applied when enabled."""
        from aragora.debate.phases.consensus_phase import (
            ConsensusPhase,
            ConsensusDependencies,
            ConsensusCallbacks,
        )

        protocol = MockProtocol(
            verify_claims_during_consensus=True,
            verification_weight_bonus=0.5,
        )

        async def mock_verify(proposal_text, limit=2):
            return {"verified": 1, "disproven": 0}

        deps = ConsensusDependencies(protocol=protocol)
        callbacks = ConsensusCallbacks(verify_claims=mock_verify)
        phase = ConsensusPhase(deps=deps, callbacks=callbacks)

        vote_counts = Counter({"claude": 2.0})
        proposals = {"claude": "Test proposal with claims"}
        choice_mapping = {"claude": "claude"}

        result = await phase._apply_verification_bonuses(
            mock_context, vote_counts, proposals, choice_mapping
        )

        # Should have bonus applied (2.0 * 0.5 * 1 = 1.0 bonus)
        assert result["claude"] == 3.0


# =============================================================================
# Belief Network Analysis Tests
# =============================================================================


class TestBeliefNetworkAnalysis:
    """Tests for belief network analysis."""

    def test_analyze_belief_network_no_analyzer(self, consensus_phase, mock_context):
        """Test belief analysis skipped without analyzer."""
        # Should not raise
        consensus_phase._analyze_belief_network(mock_context)
        assert mock_context.result.debate_cruxes == []

    def test_analyze_belief_network_no_messages(self, mock_context):
        """Test belief analysis skipped without messages."""
        from aragora.debate.phases.consensus_phase import (
            ConsensusPhase,
            ConsensusDependencies,
            ConsensusCallbacks,
        )

        def mock_get_analyzer():
            return (MagicMock(), MagicMock())

        deps = ConsensusDependencies()
        callbacks = ConsensusCallbacks(get_belief_analyzer=mock_get_analyzer)
        phase = ConsensusPhase(deps=deps, callbacks=callbacks)

        mock_context.result.messages = []
        phase._analyze_belief_network(mock_context)

        assert mock_context.result.debate_cruxes == []


# =============================================================================
# Consensus Strength Tests
# =============================================================================


class TestConsensusStrength:
    """Tests for consensus strength calculation."""

    def test_consensus_strength_strong(self, mock_context, mock_agents):
        """Test strong consensus strength with low variance."""
        from aragora.debate.phases.consensus_phase import (
            ConsensusPhase,
            ConsensusDependencies,
        )

        protocol = MockProtocol()
        deps = ConsensusDependencies(protocol=protocol)
        phase = ConsensusPhase(deps=deps)

        mock_context.agents = mock_agents
        # High agreement: 5 votes for winner, 5 for second
        vote_counts = Counter({"claude": 5.0, "gpt4": 4.5})
        total_votes = 9.5
        choice_mapping = {"claude": "claude", "gpt4": "gpt4"}

        phase._determine_majority_winner(mock_context, vote_counts, total_votes, choice_mapping)

        # Variance should be low
        assert mock_context.result.consensus_strength in ["strong", "medium"]

    def test_consensus_strength_unanimous_single_choice(self, mock_context, mock_agents):
        """Test unanimous strength when only one choice."""
        from aragora.debate.phases.consensus_phase import (
            ConsensusPhase,
            ConsensusDependencies,
        )

        protocol = MockProtocol()
        deps = ConsensusDependencies(protocol=protocol)
        phase = ConsensusPhase(deps=deps)

        mock_context.agents = mock_agents
        vote_counts = Counter({"claude": 3.0})
        total_votes = 3.0
        choice_mapping = {"claude": "claude"}

        phase._determine_majority_winner(mock_context, vote_counts, total_votes, choice_mapping)

        assert mock_context.result.consensus_strength == "unanimous"
        assert mock_context.result.consensus_variance == 0.0


# =============================================================================
# Dissenting Views Tests
# =============================================================================


class TestDissentingViews:
    """Tests for dissenting views tracking."""

    def test_dissenting_views_tracked(self, mock_context, mock_agents):
        """Test dissenting views are recorded."""
        from aragora.debate.phases.consensus_phase import (
            ConsensusPhase,
            ConsensusDependencies,
        )

        protocol = MockProtocol()
        deps = ConsensusDependencies(protocol=protocol)
        phase = ConsensusPhase(deps=deps)

        mock_context.agents = mock_agents
        vote_counts = Counter({"claude": 2.0, "gpt4": 1.0})
        total_votes = 3.0
        choice_mapping = {"claude": "claude", "gpt4": "gpt4"}

        phase._determine_majority_winner(mock_context, vote_counts, total_votes, choice_mapping)

        # gpt4 and gemini should be in dissenting views
        assert len(mock_context.result.dissenting_views) == 2


# =============================================================================
# ELO Update from Verification Tests
# =============================================================================


class TestEloUpdateFromVerification:
    """Tests for ELO updates based on verification."""

    @pytest.mark.asyncio
    async def test_elo_update_no_system(self, mock_context):
        """Test ELO update skipped without ELO system."""
        from aragora.debate.phases.consensus_phase import (
            ConsensusPhase,
            ConsensusDependencies,
        )

        deps = ConsensusDependencies()
        phase = ConsensusPhase(deps=deps)

        mock_context.result.verification_results = {"claude": {"verified": 2}}

        # Should not raise
        await phase._update_elo_from_verification(mock_context)

    @pytest.mark.asyncio
    async def test_elo_update_with_results(self, mock_context):
        """Test ELO update called with verification results."""
        from aragora.debate.phases.consensus_phase import (
            ConsensusPhase,
            ConsensusDependencies,
        )

        mock_elo = MagicMock()
        mock_elo.update_from_verification.return_value = 10.0

        deps = ConsensusDependencies(elo_system=mock_elo)
        phase = ConsensusPhase(deps=deps)

        mock_context.result.verification_results = {"claude": {"verified": 2, "disproven": 0}}

        await phase._update_elo_from_verification(mock_context)

        mock_elo.update_from_verification.assert_called_once()


# =============================================================================
# Timeout Constants Tests
# =============================================================================


class TestTimeoutConstants:
    """Tests for timeout constant values."""

    def test_default_consensus_timeout(self):
        """Test default consensus timeout is reasonable."""
        from aragora.debate.phases.consensus_phase import ConsensusPhase

        # Should be agent timeout + margin
        assert ConsensusPhase.DEFAULT_CONSENSUS_TIMEOUT > 60

    def test_judge_timeout_less_than_agent(self):
        """Test judge per-attempt timeout is less than full agent timeout."""
        from aragora.debate.phases.consensus_phase import ConsensusPhase
        from aragora.config import AGENT_TIMEOUT_SECONDS

        assert ConsensusPhase.JUDGE_TIMEOUT_PER_ATTEMPT < AGENT_TIMEOUT_SECONDS

    def test_vote_collection_timeout(self):
        """Test vote collection has outer timeout."""
        from aragora.debate.phases.consensus_phase import ConsensusPhase

        assert ConsensusPhase.VOTE_COLLECTION_TIMEOUT > 0


# =============================================================================
# Integration-Style Tests
# =============================================================================


class TestConsensusIntegration:
    """Integration-style tests for consensus phase."""

    @pytest.mark.asyncio
    async def test_full_majority_flow(self, mock_agents, mock_proposals):
        """Test complete majority consensus flow."""
        from aragora.debate.phases.consensus_phase import (
            ConsensusPhase,
            ConsensusDependencies,
            ConsensusCallbacks,
        )

        protocol = MockProtocol(consensus="majority", consensus_threshold=0.5)

        votes_cast = []

        async def mock_vote(agent, proposals, task):
            votes_cast.append(agent.name)
            # All vote for claude
            return MockVote(agent=agent.name, choice="claude", confidence=0.9)

        deps = ConsensusDependencies(protocol=protocol)
        callbacks = ConsensusCallbacks(vote_with_agent=mock_vote)
        phase = ConsensusPhase(deps=deps, callbacks=callbacks)

        ctx = MockDebateContext(
            proposals=mock_proposals,
            agents=mock_agents,
        )

        await phase.execute(ctx)

        assert len(votes_cast) == 3
        assert ctx.result.winner == "claude"
        assert ctx.result.consensus_reached is True
        assert ctx.result.confidence == 1.0

    @pytest.mark.asyncio
    async def test_majority_to_fallback_on_error(self, mock_agents, mock_proposals):
        """Test majority consensus falls back on error."""
        from aragora.debate.phases.consensus_phase import (
            ConsensusPhase,
            ConsensusDependencies,
            ConsensusCallbacks,
        )

        protocol = MockProtocol(consensus="majority")

        async def failing_vote(agent, proposals, task):
            raise Exception("All votes fail")

        deps = ConsensusDependencies(protocol=protocol)
        callbacks = ConsensusCallbacks(vote_with_agent=failing_vote)
        phase = ConsensusPhase(deps=deps, callbacks=callbacks)

        ctx = MockDebateContext(
            proposals=mock_proposals,
            agents=mock_agents,
        )

        await phase.execute(ctx)

        # Should have completed without crashing
        # Result depends on fallback behavior
        assert ctx.result is not None


__all__ = [
    "TestConsensusDependencies",
    "TestConsensusCallbacks",
    "TestConsensusPhaseInit",
    "TestNoneConsensus",
    "TestMajorityConsensus",
    "TestUnanimousConsensus",
    "TestJudgeConsensus",
    "TestFallbackConsensus",
    "TestVoteCollection",
    "TestVoteCollectionWithErrors",
    "TestVoteGrouping",
    "TestVoteWeighting",
    "TestUserVotes",
    "TestChoiceNormalization",
    "TestWinnerDetermination",
    "TestUnanimousWinner",
    "TestConsensusExecute",
    "TestCalibrationAdjustment",
    "TestVoteSuccessHandling",
    "TestVerificationBonus",
    "TestBeliefNetworkAnalysis",
    "TestConsensusStrength",
    "TestDissentingViews",
    "TestEloUpdateFromVerification",
    "TestTimeoutConstants",
    "TestConsensusIntegration",
]
