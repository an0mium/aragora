"""Tests for unanimous consensus mode in debate orchestrator."""

import pytest
from unittest.mock import AsyncMock, Mock, patch
from collections import Counter

from aragora.debate.orchestrator import Arena, DebateProtocol
from aragora.core import Agent, Environment, Vote


class MockAgent:
    """Mock agent for testing."""

    def __init__(self, name: str, vote_choice: str = None, should_fail: bool = False):
        self.name = name
        self.role = "proposer"
        self.vote_choice = vote_choice
        self.should_fail = should_fail
        self.stance = None
        self.system_prompt = None  # Required by Arena._apply_agreement_intensity

    async def generate(self, prompt: str, context: list = None) -> str:
        return f"Proposal from {self.name}"

    async def critique(self, proposal: str, task: str, context: list = None):
        return Mock(issues=[], severity=0.1, suggestions=[], target_agent="proposal", to_prompt=lambda: "")

    async def vote(self, proposals: dict, task: str) -> Vote:
        if self.should_fail:
            raise RuntimeError(f"Voting failed for {self.name}")
        return Vote(
            agent=self.name,
            choice=self.vote_choice or self.name,
            confidence=0.9,
            reasoning="Test vote"
        )


class TestUnanimousThreshold:
    """Test that unanimous mode requires 100% agreement."""

    def test_threshold_is_100_percent(self):
        """Verify unanimous threshold is set to 1.0, not 0.95."""
        # The fix changed line 815 from max(threshold, 0.95) to 1.0
        # We verify this by checking the behavior
        protocol = DebateProtocol(consensus="unanimous", consensus_threshold=0.5)

        # Even with a low consensus_threshold, unanimous mode should require 100%
        # This is verified through the integration tests below
        assert protocol.consensus == "unanimous"

    @pytest.mark.asyncio
    async def test_100_percent_agreement_reaches_consensus(self):
        """Test that 100% agreement reaches consensus."""
        agents = [
            MockAgent("alice", vote_choice="proposal_a"),
            MockAgent("bob", vote_choice="proposal_a"),
            MockAgent("charlie", vote_choice="proposal_a"),
        ]

        protocol = DebateProtocol(
            consensus="unanimous",
            rounds=1,
            convergence_detection=False,
            agreement_intensity=None,
        )

        env = Environment(task="Test task")
        arena = Arena(env, agents, protocol)

        result = await arena.run()

        assert result.consensus_reached is True
        assert result.consensus_strength == "unanimous"
        assert result.confidence == 1.0

    @pytest.mark.asyncio
    async def test_95_percent_agreement_no_consensus(self):
        """Test that 95% agreement does NOT reach consensus (verifies the fix)."""
        # Create 20 agents, 19 agree and 1 dissents = 95%
        agents = [MockAgent(f"agent_{i}", vote_choice="proposal_a") for i in range(19)]
        agents.append(MockAgent("dissenter", vote_choice="proposal_b"))

        protocol = DebateProtocol(
            consensus="unanimous",
            rounds=1,
            convergence_detection=False,
            agreement_intensity=None,
        )

        env = Environment(task="Test task")
        arena = Arena(env, agents, protocol)

        result = await arena.run()

        # With the fix, 95% should NOT be unanimous
        assert result.consensus_reached is False
        assert result.consensus_strength == "none"
        assert result.confidence == 0.95  # 19/20

    @pytest.mark.asyncio
    async def test_one_dissent_breaks_unanimity(self):
        """Test that a single dissent prevents unanimous consensus."""
        agents = [
            MockAgent("alice", vote_choice="proposal_a"),
            MockAgent("bob", vote_choice="proposal_a"),
            MockAgent("charlie", vote_choice="proposal_b"),  # Dissenter
        ]

        protocol = DebateProtocol(
            consensus="unanimous",
            rounds=1,
            convergence_detection=False,
        )

        env = Environment(task="Test task")
        arena = Arena(env, agents, protocol)

        result = await arena.run()

        assert result.consensus_reached is False
        assert "[No unanimous consensus reached]" in result.final_answer


class TestVotingErrors:
    """Test that voting errors count as dissent in unanimous mode."""

    @pytest.mark.asyncio
    async def test_voting_error_breaks_unanimity(self):
        """Test that a voting error prevents unanimous consensus."""
        agents = [
            MockAgent("alice", vote_choice="proposal_a"),
            MockAgent("bob", vote_choice="proposal_a"),
            MockAgent("charlie", should_fail=True),  # Will throw exception
        ]

        protocol = DebateProtocol(
            consensus="unanimous",
            rounds=1,
            convergence_detection=False,
        )

        env = Environment(task="Test task")
        arena = Arena(env, agents, protocol)

        result = await arena.run()

        # With the fix, voting errors count as dissent
        assert result.consensus_reached is False
        # 2 successful votes out of 3 total voters (including error)
        assert result.confidence == pytest.approx(2/3, rel=0.01)

    @pytest.mark.asyncio
    async def test_all_voting_errors_no_consensus(self):
        """Test that all voting errors results in no consensus."""
        agents = [
            MockAgent("alice", should_fail=True),
            MockAgent("bob", should_fail=True),
            MockAgent("charlie", should_fail=True),
        ]

        protocol = DebateProtocol(
            consensus="unanimous",
            rounds=1,
            convergence_detection=False,
        )

        env = Environment(task="Test task")
        arena = Arena(env, agents, protocol)

        result = await arena.run()

        assert result.consensus_reached is False


class TestVoteGrouping:
    """Test that vote grouping still works in unanimous mode."""

    @pytest.mark.asyncio
    async def test_similar_votes_grouped_for_unanimity(self):
        """Test that semantically similar votes are grouped."""
        # This tests that agents voting for slightly different wordings
        # can still achieve unanimity if the votes are semantically similar
        agents = [
            MockAgent("alice", vote_choice="alice"),  # Vote for own proposal
            MockAgent("bob", vote_choice="alice"),
            MockAgent("charlie", vote_choice="alice"),
        ]

        protocol = DebateProtocol(
            consensus="unanimous",
            rounds=1,
            vote_grouping=True,
            convergence_detection=False,
        )

        env = Environment(task="Test task")
        arena = Arena(env, agents, protocol)

        result = await arena.run()

        assert result.consensus_reached is True
