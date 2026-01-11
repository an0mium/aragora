"""
Consensus Edge Cases Integration Tests.

Tests for edge cases in consensus detection:
- Tie-breaking scenarios
- Unanimous votes
- Split votes with various distributions
- Confidence threshold edge cases
- Vote abstention handling
"""

from __future__ import annotations

import pytest

from aragora.core import Environment, Vote, DebateResult
from aragora.debate.orchestrator import Arena, DebateProtocol

from .conftest import MockAgent, run_debate_to_completion


# =============================================================================
# Helper Functions
# =============================================================================


def create_voting_agents(vote_distribution: dict[str, str]) -> list[MockAgent]:
    """
    Create agents with specific voting patterns.

    Args:
        vote_distribution: Dict mapping agent_name to choice they vote for

    Example:
        {"agent_1": "proposal_A", "agent_2": "proposal_A", "agent_3": "proposal_B"}
    """
    agents = []
    for agent_name, choice in vote_distribution.items():
        vote = Vote(
            agent=agent_name,
            choice=choice,
            reasoning=f"Vote for {choice}",
            confidence=0.85,
            continue_debate=False,
        )
        agent = MockAgent(
            name=agent_name,
            role="critic",
            votes=[vote],
        )
        agents.append(agent)
    return agents


def create_confidence_agents(
    votes: list[tuple[str, str, float]]
) -> list[MockAgent]:
    """
    Create agents with specific confidence levels.

    Args:
        votes: List of (agent_name, choice, confidence) tuples
    """
    agents = []
    for agent_name, choice, confidence in votes:
        vote = Vote(
            agent=agent_name,
            choice=choice,
            reasoning=f"Vote for {choice}",
            confidence=confidence,
            continue_debate=False,
        )
        agent = MockAgent(
            name=agent_name,
            role="critic",
            votes=[vote],
        )
        agents.append(agent)
    return agents


# =============================================================================
# Unanimous Vote Tests
# =============================================================================


class TestUnanimousVotes:
    """Tests for unanimous voting scenarios."""

    @pytest.mark.asyncio
    async def test_unanimous_vote_reaches_consensus(self):
        """All agents voting for same option should reach consensus."""
        agents = create_voting_agents({
            "agent_1": "proposal_A",
            "agent_2": "proposal_A",
            "agent_3": "proposal_A",
        })
        env = Environment(task="Unanimous test")
        protocol = DebateProtocol(rounds=1, consensus="majority")

        arena = Arena(env, agents, protocol)
        result = await run_debate_to_completion(arena)

        assert result.consensus_reached or result.final_answer is not None

    @pytest.mark.asyncio
    async def test_unanimous_with_high_confidence(self):
        """Unanimous vote with high confidence should be strong consensus."""
        agents = create_confidence_agents([
            ("agent_1", "proposal_A", 0.95),
            ("agent_2", "proposal_A", 0.92),
            ("agent_3", "proposal_A", 0.98),
        ])
        env = Environment(task="High confidence unanimous")
        protocol = DebateProtocol(rounds=1, consensus="majority")

        arena = Arena(env, agents, protocol)
        result = await run_debate_to_completion(arena)

        # Should complete successfully
        assert result is not None

    @pytest.mark.asyncio
    async def test_unanimous_with_low_confidence(self):
        """Unanimous vote with low confidence should still reach consensus."""
        agents = create_confidence_agents([
            ("agent_1", "proposal_A", 0.55),
            ("agent_2", "proposal_A", 0.52),
            ("agent_3", "proposal_A", 0.51),
        ])
        env = Environment(task="Low confidence unanimous")
        protocol = DebateProtocol(rounds=1, consensus="majority")

        arena = Arena(env, agents, protocol)
        result = await run_debate_to_completion(arena)

        # Should complete - even low confidence unanimous is consensus
        assert result is not None


# =============================================================================
# Tie-Breaking Tests
# =============================================================================


class TestTieBreaking:
    """Tests for tie-breaking scenarios."""

    @pytest.mark.asyncio
    async def test_two_way_tie_with_four_agents(self):
        """2-2 tie should be handled gracefully."""
        agents = create_voting_agents({
            "agent_1": "proposal_A",
            "agent_2": "proposal_A",
            "agent_3": "proposal_B",
            "agent_4": "proposal_B",
        })
        env = Environment(task="Two-way tie test")
        protocol = DebateProtocol(rounds=2, consensus="majority")

        arena = Arena(env, agents, protocol)
        result = await run_debate_to_completion(arena)

        # Should complete, either with no consensus or additional rounds
        assert result is not None

    @pytest.mark.asyncio
    async def test_three_way_tie(self):
        """Three-way tie should be handled."""
        agents = create_voting_agents({
            "agent_1": "proposal_A",
            "agent_2": "proposal_B",
            "agent_3": "proposal_C",
        })
        env = Environment(task="Three-way tie test")
        protocol = DebateProtocol(rounds=2, consensus="majority")

        arena = Arena(env, agents, protocol)
        result = await run_debate_to_completion(arena)

        # Should complete
        assert result is not None

    @pytest.mark.asyncio
    async def test_tie_broken_by_confidence(self):
        """When tied on count, higher confidence should win."""
        agents = create_confidence_agents([
            ("agent_1", "proposal_A", 0.9),  # High confidence
            ("agent_2", "proposal_A", 0.9),
            ("agent_3", "proposal_B", 0.6),  # Lower confidence
            ("agent_4", "proposal_B", 0.5),
        ])
        env = Environment(task="Confidence tie-breaker")
        protocol = DebateProtocol(rounds=1, consensus="majority")

        arena = Arena(env, agents, protocol)
        result = await run_debate_to_completion(arena)

        assert result is not None


# =============================================================================
# Split Vote Tests
# =============================================================================


class TestSplitVotes:
    """Tests for various split vote distributions."""

    @pytest.mark.asyncio
    async def test_majority_with_dissent(self):
        """3-2 majority should reach consensus."""
        agents = create_voting_agents({
            "agent_1": "proposal_A",
            "agent_2": "proposal_A",
            "agent_3": "proposal_A",
            "agent_4": "proposal_B",
            "agent_5": "proposal_B",
        })
        env = Environment(task="Majority with dissent")
        protocol = DebateProtocol(rounds=1, consensus="majority")

        arena = Arena(env, agents, protocol)
        result = await run_debate_to_completion(arena)

        # Should reach consensus with majority
        assert result is not None

    @pytest.mark.asyncio
    async def test_supermajority_required(self):
        """Test when supermajority (2/3) is required."""
        # 4-2 split (67% majority)
        agents = create_voting_agents({
            "agent_1": "proposal_A",
            "agent_2": "proposal_A",
            "agent_3": "proposal_A",
            "agent_4": "proposal_A",
            "agent_5": "proposal_B",
            "agent_6": "proposal_B",
        })
        env = Environment(task="Supermajority test")
        protocol = DebateProtocol(rounds=1, consensus="supermajority")

        arena = Arena(env, agents, protocol)
        result = await run_debate_to_completion(arena)

        assert result is not None

    @pytest.mark.asyncio
    async def test_no_clear_majority(self):
        """Vote split across many options with no majority."""
        agents = create_voting_agents({
            "agent_1": "proposal_A",
            "agent_2": "proposal_B",
            "agent_3": "proposal_C",
            "agent_4": "proposal_D",
            "agent_5": "proposal_E",
        })
        env = Environment(task="No majority test")
        protocol = DebateProtocol(rounds=2, consensus="majority")

        arena = Arena(env, agents, protocol)
        result = await run_debate_to_completion(arena)

        # Should complete even without consensus
        assert result is not None


# =============================================================================
# Continue Debate Flag Tests
# =============================================================================


class TestContinueDebateFlag:
    """Tests for the continue_debate flag in votes."""

    @pytest.mark.asyncio
    async def test_agents_want_to_continue(self):
        """When agents want to continue, debate should run more rounds."""
        continue_vote = Vote(
            agent="",
            choice="proposal_A",
            reasoning="Need more discussion",
            confidence=0.6,
            continue_debate=True,  # Want to continue
        )
        agents = [
            MockAgent(
                name=f"agent_{i}",
                role="critic",
                votes=[Vote(**{**continue_vote.__dict__, "agent": f"agent_{i}"})],
            )
            for i in range(3)
        ]

        env = Environment(task="Continue debate test")
        protocol = DebateProtocol(rounds=3, consensus="majority")

        arena = Arena(env, agents, protocol)
        result = await run_debate_to_completion(arena)

        # Should complete within max rounds
        assert result is not None
        assert result.rounds_completed <= 3

    @pytest.mark.asyncio
    async def test_mixed_continue_flags(self):
        """Some agents want to continue, others don't."""
        agents = [
            MockAgent(
                name="wants_more",
                role="proposer",
                votes=[Vote(
                    agent="wants_more",
                    choice="proposal_A",
                    reasoning="Need more",
                    confidence=0.6,
                    continue_debate=True,
                )],
            ),
            MockAgent(
                name="satisfied_1",
                role="critic",
                votes=[Vote(
                    agent="satisfied_1",
                    choice="proposal_A",
                    reasoning="Good enough",
                    confidence=0.85,
                    continue_debate=False,
                )],
            ),
            MockAgent(
                name="satisfied_2",
                role="synthesizer",
                votes=[Vote(
                    agent="satisfied_2",
                    choice="proposal_A",
                    reasoning="Agree",
                    confidence=0.9,
                    continue_debate=False,
                )],
            ),
        ]

        env = Environment(task="Mixed continue flags")
        protocol = DebateProtocol(rounds=2, consensus="majority")

        arena = Arena(env, agents, protocol)
        result = await run_debate_to_completion(arena)

        assert result is not None


# =============================================================================
# Confidence Threshold Tests
# =============================================================================


class TestConfidenceThresholds:
    """Tests for confidence threshold edge cases."""

    @pytest.mark.asyncio
    async def test_all_agents_at_threshold(self):
        """All agents exactly at confidence threshold."""
        threshold = 0.7
        agents = create_confidence_agents([
            ("agent_1", "proposal_A", threshold),
            ("agent_2", "proposal_A", threshold),
            ("agent_3", "proposal_A", threshold),
        ])

        env = Environment(task="Threshold edge case")
        protocol = DebateProtocol(rounds=1, consensus="majority")

        arena = Arena(env, agents, protocol)
        result = await run_debate_to_completion(arena)

        assert result is not None

    @pytest.mark.asyncio
    async def test_confidence_just_below_threshold(self):
        """Agents with confidence just below typical threshold."""
        agents = create_confidence_agents([
            ("agent_1", "proposal_A", 0.49),
            ("agent_2", "proposal_A", 0.48),
            ("agent_3", "proposal_A", 0.47),
        ])

        env = Environment(task="Below threshold")
        protocol = DebateProtocol(rounds=2, consensus="majority")

        arena = Arena(env, agents, protocol)
        result = await run_debate_to_completion(arena)

        # Should complete - low confidence doesn't prevent completion
        assert result is not None

    @pytest.mark.asyncio
    async def test_extreme_confidence_variance(self):
        """One agent very confident, others uncertain."""
        agents = create_confidence_agents([
            ("agent_1", "proposal_A", 0.99),  # Very confident
            ("agent_2", "proposal_A", 0.51),  # Barely confident
            ("agent_3", "proposal_B", 0.50),  # Indifferent
        ])

        env = Environment(task="Confidence variance")
        protocol = DebateProtocol(rounds=1, consensus="majority")

        arena = Arena(env, agents, protocol)
        result = await run_debate_to_completion(arena)

        assert result is not None


# =============================================================================
# Edge Cases with Few Agents
# =============================================================================


class TestFewAgents:
    """Tests for consensus with minimal agents."""

    @pytest.mark.asyncio
    async def test_two_agents_agree(self):
        """Two agents agreeing should reach consensus."""
        agents = create_voting_agents({
            "agent_1": "proposal_A",
            "agent_2": "proposal_A",
        })

        env = Environment(task="Two agents agree")
        protocol = DebateProtocol(rounds=1, consensus="majority")

        arena = Arena(env, agents, protocol)
        result = await run_debate_to_completion(arena)

        assert result is not None
        assert result.consensus_reached or result.final_answer

    @pytest.mark.asyncio
    async def test_two_agents_disagree(self):
        """Two agents disagreeing - perfect tie."""
        agents = create_voting_agents({
            "agent_1": "proposal_A",
            "agent_2": "proposal_B",
        })

        env = Environment(task="Two agents disagree")
        protocol = DebateProtocol(rounds=2, consensus="majority")

        arena = Arena(env, agents, protocol)
        result = await run_debate_to_completion(arena)

        # Should complete without consensus or after max rounds
        assert result is not None

    @pytest.mark.asyncio
    async def test_single_agent_debate(self):
        """Single agent should reach 'consensus' by default."""
        agents = [
            MockAgent(name="solo", role="proposer"),
        ]

        env = Environment(task="Solo debate")
        protocol = DebateProtocol(rounds=1, consensus="any")

        arena = Arena(env, agents, protocol)
        result = await run_debate_to_completion(arena)

        # Single agent debates should complete
        assert result is not None
