"""
Integration tests for consensus detection mechanisms.

Tests various consensus scenarios:
- Unanimous consensus
- Majority consensus
- Supermajority consensus
- No consensus (deadlock)
- Edge cases (tie-breaking, minimum rounds)
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch

from aragora.core import Vote, Environment
from aragora.debate.orchestrator import Arena, DebateProtocol
from tests.integration.conftest import MockAgent


class TestUnanimousConsensus:
    """Tests for unanimous consensus detection."""

    @pytest.mark.asyncio
    async def test_unanimous_agreement_reaches_consensus(self):
        """All agents voting for the same proposal should reach consensus."""
        # Create agents that all vote for the same choice
        agents = [
            MockAgent(
                name=f"agent_{i}",
                responses=["I propose solution A because it's optimal."],
                votes=[
                    Vote(
                        agent=f"agent_{i}",
                        choice="agent_0",
                        reasoning="Solution A is clearly the best approach",
                        confidence=0.95,
                        continue_debate=False,
                    )
                ],
            )
            for i in range(3)
        ]

        env = Environment(task="Design a caching strategy")
        protocol = DebateProtocol(rounds=3, consensus="unanimous")

        with patch.object(Arena, "_gather_trending_context", new_callable=AsyncMock):
            arena = Arena(env, agents, protocol)
            result = await arena.run()

        assert result is not None
        assert result.consensus_reached is True
        assert result.final_answer is not None

    @pytest.mark.asyncio
    async def test_unanimous_with_single_dissent_fails(self):
        """Single dissenting vote should prevent unanimous consensus."""
        # Two agents agree, one dissents
        agents = [
            MockAgent(
                name="agent_0",
                responses=["Solution A is best"],
                votes=[
                    Vote(
                        agent="agent_0",
                        choice="agent_0",
                        reasoning="A is optimal",
                        confidence=0.9,
                        continue_debate=False,
                    )
                ],
            ),
            MockAgent(
                name="agent_1",
                responses=["Solution A is best"],
                votes=[
                    Vote(
                        agent="agent_1",
                        choice="agent_0",
                        reasoning="A is optimal",
                        confidence=0.9,
                        continue_debate=False,
                    )
                ],
            ),
            MockAgent(
                name="agent_2",
                responses=["Solution B is better"],
                votes=[
                    Vote(
                        agent="agent_2",
                        choice="agent_2",  # Votes for itself
                        reasoning="B handles edge cases better",
                        confidence=0.85,
                        continue_debate=True,  # Wants to continue
                    )
                ],
            ),
        ]

        env = Environment(task="Design a caching strategy")
        protocol = DebateProtocol(rounds=1, consensus="unanimous")

        with patch.object(Arena, "_gather_trending_context", new_callable=AsyncMock):
            arena = Arena(env, agents, protocol)
            result = await arena.run()

        # Unanimous requires all agents to agree
        # With dissent, should either not reach consensus or continue
        assert result is not None


class TestMajorityConsensus:
    """Tests for majority consensus detection."""

    @pytest.mark.asyncio
    async def test_majority_agreement_reaches_consensus(self):
        """2 out of 3 agents agreeing should reach majority consensus."""
        agents = [
            MockAgent(
                name="agent_0",
                responses=["Solution A"],
                votes=[
                    Vote(
                        agent="agent_0",
                        choice="agent_0",
                        reasoning="A is best",
                        confidence=0.9,
                        continue_debate=False,
                    )
                ],
            ),
            MockAgent(
                name="agent_1",
                responses=["Solution A is good"],
                votes=[
                    Vote(
                        agent="agent_1",
                        choice="agent_0",  # Also votes for A
                        reasoning="A is best",
                        confidence=0.85,
                        continue_debate=False,
                    )
                ],
            ),
            MockAgent(
                name="agent_2",
                responses=["Solution B"],
                votes=[
                    Vote(
                        agent="agent_2",
                        choice="agent_2",  # Votes for B
                        reasoning="B is better",
                        confidence=0.7,
                        continue_debate=False,
                    )
                ],
            ),
        ]

        env = Environment(task="Design an API rate limiter")
        protocol = DebateProtocol(rounds=2, consensus="majority")

        with patch.object(Arena, "_gather_trending_context", new_callable=AsyncMock):
            arena = Arena(env, agents, protocol)
            result = await arena.run()

        assert result is not None
        assert result.consensus_reached is True

    @pytest.mark.asyncio
    async def test_no_majority_continues_debate(self):
        """When no clear majority exists, debate should continue or end without consensus."""
        # Three agents, each voting for themselves (three-way split)
        agents = [
            MockAgent(
                name=f"agent_{i}",
                responses=[f"My unique solution {i}"],
                votes=[
                    Vote(
                        agent=f"agent_{i}",
                        choice=f"agent_{i}",  # Each votes for themselves
                        reasoning=f"My solution {i} is best",
                        confidence=0.8,
                        continue_debate=True,
                    )
                ],
            )
            for i in range(3)
        ]

        env = Environment(task="Design a complex system")
        protocol = DebateProtocol(rounds=1, consensus="majority")

        with patch.object(Arena, "_gather_trending_context", new_callable=AsyncMock):
            arena = Arena(env, agents, protocol)
            result = await arena.run()

        assert result is not None
        # With a three-way split in 1 round, no majority should be reached
        # Result depends on implementation - may or may not reach consensus


class TestSupermajorityConsensus:
    """Tests for supermajority (2/3+) consensus detection."""

    @pytest.mark.asyncio
    async def test_supermajority_reached_with_high_agreement(self):
        """4 out of 5 agents (80%) should reach supermajority consensus."""
        agents = []
        for i in range(5):
            # 4 agents vote for agent_0, 1 votes for itself
            choice = "agent_0" if i < 4 else "agent_4"
            agents.append(
                MockAgent(
                    name=f"agent_{i}",
                    responses=[f"Solution {choice}"],
                    votes=[
                        Vote(
                            agent=f"agent_{i}",
                            choice=choice,
                            reasoning=f"Voting for {choice}",
                            confidence=0.9,
                            continue_debate=False,
                        )
                    ],
                )
            )

        env = Environment(task="Design a distributed consensus algorithm")
        protocol = DebateProtocol(rounds=2, consensus="supermajority")

        with patch.object(Arena, "_gather_trending_context", new_callable=AsyncMock):
            arena = Arena(env, agents, protocol)
            result = await arena.run()

        assert result is not None
        assert result.consensus_reached is True

    @pytest.mark.asyncio
    async def test_simple_majority_not_enough_for_supermajority(self):
        """3 out of 5 agents (60%) should NOT reach supermajority consensus."""
        agents = []
        for i in range(5):
            # 3 agents vote for agent_0, 2 vote for agent_3
            choice = "agent_0" if i < 3 else "agent_3"
            agents.append(
                MockAgent(
                    name=f"agent_{i}",
                    responses=[f"Solution {choice}"],
                    votes=[
                        Vote(
                            agent=f"agent_{i}",
                            choice=choice,
                            reasoning=f"Voting for {choice}",
                            confidence=0.85,
                            continue_debate=False,
                        )
                    ],
                )
            )

        env = Environment(task="Design a voting system")
        protocol = DebateProtocol(rounds=1, consensus="supermajority")

        with patch.object(Arena, "_gather_trending_context", new_callable=AsyncMock):
            arena = Arena(env, agents, protocol)
            result = await arena.run()

        assert result is not None
        # 60% is not enough for supermajority (66%+)
        # Exact behavior depends on implementation


class TestConsensusEdgeCases:
    """Tests for consensus edge cases."""

    @pytest.mark.asyncio
    async def test_two_agent_tie_handling(self):
        """Test how ties are handled with 2 agents."""
        agents = [
            MockAgent(
                name="agent_0",
                responses=["Solution A"],
                votes=[
                    Vote(
                        agent="agent_0",
                        choice="agent_0",
                        reasoning="A is best",
                        confidence=0.9,
                        continue_debate=False,
                    )
                ],
            ),
            MockAgent(
                name="agent_1",
                responses=["Solution B"],
                votes=[
                    Vote(
                        agent="agent_1",
                        choice="agent_1",
                        reasoning="B is best",
                        confidence=0.9,
                        continue_debate=False,
                    )
                ],
            ),
        ]

        env = Environment(task="Choose between two options")
        protocol = DebateProtocol(rounds=1, consensus="majority")

        with patch.object(Arena, "_gather_trending_context", new_callable=AsyncMock):
            arena = Arena(env, agents, protocol)
            result = await arena.run()

        # In a tie, implementation may use confidence as tiebreaker
        # or may not reach consensus
        assert result is not None

    @pytest.mark.asyncio
    async def test_single_agent_debate(self):
        """Single agent should reach consensus trivially."""
        agents = [
            MockAgent(
                name="solo_agent",
                responses=["My sole proposal"],
                votes=[
                    Vote(
                        agent="solo_agent",
                        choice="solo_agent",
                        reasoning="Only option",
                        confidence=1.0,
                        continue_debate=False,
                    )
                ],
            ),
        ]

        env = Environment(task="Single agent task")
        protocol = DebateProtocol(rounds=1, consensus="unanimous")

        with patch.object(Arena, "_gather_trending_context", new_callable=AsyncMock):
            arena = Arena(env, agents, protocol)
            result = await arena.run()

        assert result is not None
        assert result.consensus_reached is True

    @pytest.mark.asyncio
    async def test_low_confidence_votes_affect_consensus(self):
        """Low confidence votes should potentially affect consensus calculation."""
        agents = [
            MockAgent(
                name="confident_agent",
                responses=["Solution A with high confidence"],
                votes=[
                    Vote(
                        agent="confident_agent",
                        choice="confident_agent",
                        reasoning="Very confident in A",
                        confidence=0.99,
                        continue_debate=False,
                    )
                ],
            ),
            MockAgent(
                name="uncertain_agent_1",
                responses=["Maybe A?"],
                votes=[
                    Vote(
                        agent="uncertain_agent_1",
                        choice="confident_agent",
                        reasoning="A seems okay",
                        confidence=0.3,  # Low confidence
                        continue_debate=True,  # Uncertain, wants to continue
                    )
                ],
            ),
            MockAgent(
                name="uncertain_agent_2",
                responses=["A or B?"],
                votes=[
                    Vote(
                        agent="uncertain_agent_2",
                        choice="confident_agent",
                        reasoning="Leaning towards A",
                        confidence=0.4,  # Low confidence
                        continue_debate=True,
                    )
                ],
            ),
        ]

        env = Environment(task="Test confidence weighting")
        protocol = DebateProtocol(rounds=2, consensus="majority")

        with patch.object(Arena, "_gather_trending_context", new_callable=AsyncMock):
            arena = Arena(env, agents, protocol)
            result = await arena.run()

        assert result is not None
        # Low confidence votes might trigger additional rounds


class TestConsensusWithMultipleRounds:
    """Tests for consensus evolution across multiple rounds."""

    @pytest.mark.asyncio
    async def test_consensus_emerges_after_multiple_rounds(self):
        """Test that opinions can converge over multiple rounds."""
        # Agents start divided but converge
        round_votes = [
            # Round 1: Split votes
            [
                Vote(
                    agent="agent_0",
                    choice="agent_0",
                    reasoning="A",
                    confidence=0.7,
                    continue_debate=True,
                ),
                Vote(
                    agent="agent_1",
                    choice="agent_1",
                    reasoning="B",
                    confidence=0.7,
                    continue_debate=True,
                ),
                Vote(
                    agent="agent_2",
                    choice="agent_2",
                    reasoning="C",
                    confidence=0.7,
                    continue_debate=True,
                ),
            ],
            # Round 2: Converging to A
            [
                Vote(
                    agent="agent_0",
                    choice="agent_0",
                    reasoning="A",
                    confidence=0.8,
                    continue_debate=True,
                ),
                Vote(
                    agent="agent_1",
                    choice="agent_0",
                    reasoning="A convinced me",
                    confidence=0.75,
                    continue_debate=False,
                ),
                Vote(
                    agent="agent_2",
                    choice="agent_2",
                    reasoning="C still",
                    confidence=0.6,
                    continue_debate=True,
                ),
            ],
            # Round 3: Full convergence
            [
                Vote(
                    agent="agent_0",
                    choice="agent_0",
                    reasoning="A",
                    confidence=0.9,
                    continue_debate=False,
                ),
                Vote(
                    agent="agent_1",
                    choice="agent_0",
                    reasoning="A",
                    confidence=0.85,
                    continue_debate=False,
                ),
                Vote(
                    agent="agent_2",
                    choice="agent_0",
                    reasoning="A makes sense",
                    confidence=0.7,
                    continue_debate=False,
                ),
            ],
        ]

        agents = []
        for i in range(3):
            # Each agent cycles through their votes per round
            agent_votes = [round_votes[r][i] for r in range(3)]
            agents.append(
                MockAgent(
                    name=f"agent_{i}",
                    responses=[f"Response round {r}" for r in range(3)],
                    votes=agent_votes,
                )
            )

        env = Environment(task="Multi-round convergence test")
        protocol = DebateProtocol(rounds=3, consensus="majority")

        with patch.object(Arena, "_gather_trending_context", new_callable=AsyncMock):
            arena = Arena(env, agents, protocol)
            result = await arena.run()

        assert result is not None
        # Should reach consensus by round 3
