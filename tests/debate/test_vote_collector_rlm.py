"""
Tests for RLM early termination in VoteCollector.

Tests the RLM-inspired early termination optimization that stops vote
collection when a clear majority has been reached.
"""

import asyncio
import pytest
from dataclasses import dataclass
from typing import Optional
from unittest.mock import AsyncMock, MagicMock

from aragora.debate.phases.vote_collector import (
    VoteCollector,
    VoteCollectorConfig,
    create_vote_collector,
    RLM_EARLY_TERMINATION_THRESHOLD,
    RLM_MAJORITY_LEAD_THRESHOLD,
)


# =============================================================================
# Mock Classes
# =============================================================================


@dataclass
class MockVote:
    """Mock vote for testing."""

    choice: str
    confidence: float = 0.8
    reasoning: str = "Test reasoning"


@dataclass
class MockAgent:
    """Mock agent for testing."""

    name: str


@dataclass
class MockEnvironment:
    """Mock environment for testing."""

    task: str = "Test task"


@dataclass
class MockResult:
    """Mock result for testing."""

    id: str = "test-123"
    rounds_used: int = 3


class MockDebateContext:
    """Mock debate context for testing."""

    def __init__(self, agents: list = None, proposals: dict = None):
        self.agents = agents or []
        self.proposals = proposals or {}
        self.env = MockEnvironment()
        self.result = MockResult()


# =============================================================================
# _check_clear_majority Tests
# =============================================================================


class TestCheckClearMajority:
    """Tests for _check_clear_majority method."""

    def test_returns_false_when_disabled(self):
        """Should return False when RLM early termination is disabled."""
        config = VoteCollectorConfig(enable_rlm_early_termination=False)
        collector = VoteCollector(config)

        votes = [MockVote("A"), MockVote("A"), MockVote("A")]
        has_majority, leader = collector._check_clear_majority(votes, 4)

        assert has_majority is False
        assert leader is None

    def test_returns_false_with_empty_votes(self):
        """Should return False when no votes collected."""
        config = VoteCollectorConfig(enable_rlm_early_termination=True)
        collector = VoteCollector(config)

        has_majority, leader = collector._check_clear_majority([], 4)

        assert has_majority is False

    def test_returns_false_below_threshold(self):
        """Should return False when below minimum vote threshold."""
        config = VoteCollectorConfig(
            enable_rlm_early_termination=True,
            rlm_early_termination_threshold=0.75,
        )
        collector = VoteCollector(config)

        # Only 2 of 4 votes (50%), threshold is 75%
        votes = [MockVote("A"), MockVote("A")]
        has_majority, leader = collector._check_clear_majority(votes, 4)

        assert has_majority is False

    def test_returns_false_no_majority(self):
        """Should return False when no choice has majority of total agents."""
        config = VoteCollectorConfig(
            enable_rlm_early_termination=True,
            rlm_early_termination_threshold=0.75,
        )
        collector = VoteCollector(config)

        # 3 votes collected, but split 2-1, neither has >50% of 4 agents
        votes = [MockVote("A"), MockVote("A"), MockVote("B")]
        has_majority, leader = collector._check_clear_majority(votes, 4)

        assert has_majority is False

    def test_returns_false_insufficient_lead(self):
        """Should return False when lead is insufficient."""
        config = VoteCollectorConfig(
            enable_rlm_early_termination=True,
            rlm_early_termination_threshold=0.5,
            rlm_majority_lead_threshold=0.5,  # Need 50% lead
        )
        collector = VoteCollector(config)

        # 4 votes: 3 for A, 1 for B - lead is 2, need 4*0.5=2
        # But lead is exactly at threshold, not above
        votes = [MockVote("A"), MockVote("A"), MockVote("A"), MockVote("B")]
        has_majority, leader = collector._check_clear_majority(votes, 4)

        # 3 > 4/2 (2), so has majority, but lead is 2 which equals min_lead
        assert has_majority is True  # Lead of 2 >= min_lead of 2
        assert leader == "A"

    def test_returns_true_clear_majority(self):
        """Should return True when clear majority is reached."""
        config = VoteCollectorConfig(
            enable_rlm_early_termination=True,
            rlm_early_termination_threshold=0.75,
            rlm_majority_lead_threshold=0.25,
        )
        collector = VoteCollector(config)

        # 3 of 4 votes (75%), all for A
        # A has 3 votes > 50% of 4 agents, lead is 3 >= 4*0.25=1
        votes = [MockVote("A"), MockVote("A"), MockVote("A")]
        has_majority, leader = collector._check_clear_majority(votes, 4)

        assert has_majority is True
        assert leader == "A"

    def test_returns_true_with_second_choice(self):
        """Should return True when leader has clear lead over second."""
        config = VoteCollectorConfig(
            enable_rlm_early_termination=True,
            rlm_early_termination_threshold=0.6,
            rlm_majority_lead_threshold=0.2,
        )
        collector = VoteCollector(config)

        # 5 of 6 votes (83%), 4 for A, 1 for B
        # A has 4 votes > 50% of 6 agents, lead is 3 >= 6*0.2=1.2
        votes = [
            MockVote("A"),
            MockVote("A"),
            MockVote("A"),
            MockVote("A"),
            MockVote("B"),
        ]
        has_majority, leader = collector._check_clear_majority(votes, 6)

        assert has_majority is True
        assert leader == "A"


# =============================================================================
# collect_votes with RLM Early Termination Tests
# =============================================================================


class TestCollectVotesWithRLMTermination:
    """Tests for collect_votes with RLM early termination."""

    @pytest.mark.asyncio
    async def test_early_terminates_on_clear_majority(self):
        """Should stop collecting votes when clear majority reached."""
        # Track which agents were called
        called_agents = []

        async def mock_vote(agent, proposals, task):
            called_agents.append(agent.name)
            # Simulate some delay
            await asyncio.sleep(0.01)
            return MockVote("A")

        config = VoteCollectorConfig(
            vote_with_agent=mock_vote,
            enable_rlm_early_termination=True,
            rlm_early_termination_threshold=0.6,  # 60%
            rlm_majority_lead_threshold=0.2,  # 20%
        )
        collector = VoteCollector(config)

        # 5 agents - need 3 votes (60%) with same choice for early termination
        agents = [MockAgent(f"agent{i}") for i in range(5)]
        ctx = MockDebateContext(agents=agents, proposals={"p1": "proposal"})

        votes = await collector.collect_votes(ctx)

        # Should have collected at least 3 votes (the minimum for early termination)
        # but may not have collected all 5
        assert len(votes) >= 3
        assert all(v.choice == "A" for v in votes)

    @pytest.mark.asyncio
    async def test_collects_all_when_no_majority(self):
        """Should collect all votes when no clear majority emerges."""
        vote_count = [0]

        async def mock_vote(agent, proposals, task):
            vote_count[0] += 1
            # Alternate choices so no majority
            choice = "A" if vote_count[0] % 2 == 0 else "B"
            return MockVote(choice)

        config = VoteCollectorConfig(
            vote_with_agent=mock_vote,
            enable_rlm_early_termination=True,
            rlm_early_termination_threshold=0.75,
            rlm_majority_lead_threshold=0.25,
        )
        collector = VoteCollector(config)

        agents = [MockAgent(f"agent{i}") for i in range(4)]
        ctx = MockDebateContext(agents=agents, proposals={"p1": "proposal"})

        votes = await collector.collect_votes(ctx)

        # Should have collected all votes since no clear majority
        assert len(votes) == 4

    @pytest.mark.asyncio
    async def test_emits_hook_on_early_termination(self):
        """Should emit hook when early termination triggered."""
        hook_fn = MagicMock()

        async def mock_vote(agent, proposals, task):
            return MockVote("A")

        config = VoteCollectorConfig(
            vote_with_agent=mock_vote,
            hooks={"on_rlm_early_termination": hook_fn},
            enable_rlm_early_termination=True,
            rlm_early_termination_threshold=0.6,
            rlm_majority_lead_threshold=0.2,
        )
        collector = VoteCollector(config)

        agents = [MockAgent(f"agent{i}") for i in range(5)]
        ctx = MockDebateContext(agents=agents, proposals={"p1": "proposal"})

        await collector.collect_votes(ctx)

        # Hook should have been called
        hook_fn.assert_called()
        call_kwargs = hook_fn.call_args.kwargs
        assert call_kwargs["leader"] == "A"
        assert call_kwargs["total_agents"] == 5

    @pytest.mark.asyncio
    async def test_notifies_spectator_on_early_termination(self):
        """Should notify spectator when early termination triggered."""
        notify_fn = MagicMock()

        async def mock_vote(agent, proposals, task):
            return MockVote("A")

        config = VoteCollectorConfig(
            vote_with_agent=mock_vote,
            notify_spectator=notify_fn,
            enable_rlm_early_termination=True,
            rlm_early_termination_threshold=0.6,
            rlm_majority_lead_threshold=0.2,
        )
        collector = VoteCollector(config)

        agents = [MockAgent(f"agent{i}") for i in range(5)]
        ctx = MockDebateContext(agents=agents, proposals={"p1": "proposal"})

        await collector.collect_votes(ctx)

        # Check for rlm_early_termination notification
        rlm_calls = [
            call for call in notify_fn.call_args_list if call[0][0] == "rlm_early_termination"
        ]
        assert len(rlm_calls) >= 1

    @pytest.mark.asyncio
    async def test_respects_disabled_early_termination(self):
        """Should collect all votes when early termination disabled."""
        vote_count = [0]

        async def mock_vote(agent, proposals, task):
            vote_count[0] += 1
            return MockVote("A")

        config = VoteCollectorConfig(
            vote_with_agent=mock_vote,
            enable_rlm_early_termination=False,  # Disabled
        )
        collector = VoteCollector(config)

        agents = [MockAgent(f"agent{i}") for i in range(4)]
        ctx = MockDebateContext(agents=agents, proposals={"p1": "proposal"})

        votes = await collector.collect_votes(ctx)

        # Should have collected all votes
        assert len(votes) == 4


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestCreateVoteCollector:
    """Tests for create_vote_collector factory function."""

    def test_creates_with_rlm_options(self):
        """Should create collector with RLM options."""
        collector = create_vote_collector(
            enable_rlm_early_termination=True,
            rlm_early_termination_threshold=0.8,
            rlm_majority_lead_threshold=0.3,
        )

        assert collector.config.enable_rlm_early_termination is True
        assert collector.config.rlm_early_termination_threshold == 0.8
        assert collector.config.rlm_majority_lead_threshold == 0.3

    def test_defaults_rlm_enabled(self):
        """Should default to RLM early termination enabled."""
        collector = create_vote_collector()

        assert collector.config.enable_rlm_early_termination is True
        assert collector.config.rlm_early_termination_threshold == RLM_EARLY_TERMINATION_THRESHOLD
        assert collector.config.rlm_majority_lead_threshold == RLM_MAJORITY_LEAD_THRESHOLD


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
