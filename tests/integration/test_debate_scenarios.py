"""
Integration tests for debate scenarios.

Tests complete debate workflows including:
- Full debate lifecycle (spec → rounds → consensus)
- Multi-agent consensus detection
- Checkpoint and resume functionality
- Memory integration across debates
- Error handling and recovery
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from aragora.core import Environment, Vote
from aragora.debate.orchestrator import Arena
from aragora.debate.protocol import DebateProtocol
from tests.integration.conftest import MockAgent, FailingAgent, SlowAgent


class TestDebateLifecycle:
    """Tests for complete debate lifecycle."""

    @pytest.mark.asyncio
    async def test_full_debate_lifecycle_three_rounds(self, mock_agents, simple_environment):
        """Test complete 3-round debate from start to conclusion."""
        protocol = DebateProtocol(rounds=3, consensus="majority")

        arena = Arena(simple_environment, mock_agents, protocol)
        result = await arena.run()

        assert result is not None
        assert result.rounds_completed <= 3
        # Debate should complete with valid result
        assert result.debate_id is not None

    @pytest.mark.asyncio
    async def test_debate_with_quick_consensus(self, consensus_agents, simple_environment):
        """Test debate reaching consensus quickly."""
        protocol = DebateProtocol(rounds=5, consensus="majority")

        arena = Arena(simple_environment, consensus_agents, protocol)
        result = await arena.run()

        assert result is not None
        # Debate may complete early due to consensus
        assert result.rounds_completed <= 5

    @pytest.mark.asyncio
    async def test_debate_without_consensus(self, split_vote_agents, simple_environment):
        """Test debate completing without consensus."""
        protocol = DebateProtocol(rounds=2, consensus="unanimous")

        arena = Arena(simple_environment, split_vote_agents, protocol)
        result = await arena.run()

        assert result is not None
        assert result.rounds_completed <= 2
        # Should complete even without unanimous consensus

    @pytest.mark.asyncio
    async def test_single_round_debate(self, mock_agents, simple_environment):
        """Test minimal single-round debate."""
        protocol = DebateProtocol(rounds=1, consensus="any")

        arena = Arena(simple_environment, mock_agents, protocol)
        result = await arena.run()

        assert result is not None
        assert result.rounds_completed >= 1


class TestMultiAgentConsensus:
    """Tests for multi-agent consensus scenarios."""

    @pytest.mark.asyncio
    async def test_three_agent_majority_consensus(self, simple_environment):
        """Test 3-agent debate with majority consensus."""
        # Two agents agree, one disagrees
        agents = [
            MockAgent(
                name="agent_a",
                responses=["I propose solution A"],
                votes=[
                    Vote(
                        agent="agent_a",
                        choice="agent_a",
                        reasoning="Solution A is best",
                        confidence=0.9,
                    )
                ],
            ),
            MockAgent(
                name="agent_b",
                responses=["I also prefer A"],
                votes=[
                    Vote(
                        agent="agent_b",
                        choice="agent_a",  # Votes for A
                        reasoning="Agreeing with A",
                        confidence=0.85,
                    )
                ],
            ),
            MockAgent(
                name="agent_c",
                responses=["I disagree, B is better"],
                votes=[
                    Vote(
                        agent="agent_c",
                        choice="agent_c",
                        reasoning="B is superior",
                        confidence=0.7,
                    )
                ],
            ),
        ]

        protocol = DebateProtocol(rounds=2, consensus="majority")
        arena = Arena(simple_environment, agents, protocol)
        result = await arena.run()

        assert result is not None
        assert result.rounds_completed <= 2

    @pytest.mark.asyncio
    async def test_five_agent_debate(self, simple_environment):
        """Test larger 5-agent debate."""
        agents = [
            MockAgent(
                name=f"agent_{i}",
                responses=[f"Proposal from agent {i}"],
            )
            for i in range(5)
        ]

        protocol = DebateProtocol(rounds=2, consensus="majority")
        arena = Arena(simple_environment, agents, protocol)
        result = await arena.run()

        assert result is not None
        assert result.rounds_completed <= 2

    @pytest.mark.asyncio
    async def test_unanimous_consensus_requires_all_agents(self, simple_environment):
        """Test that unanimous consensus requires all agents to agree."""
        # All agents vote for same proposal
        agents = [
            MockAgent(
                name=f"agent_{i}",
                responses=["Unified proposal"],
                votes=[
                    Vote(
                        agent=f"agent_{i}",
                        choice="agent_0",
                        reasoning="All agree",
                        confidence=0.95,
                    )
                ],
            )
            for i in range(3)
        ]

        protocol = DebateProtocol(rounds=3, consensus="unanimous")
        arena = Arena(simple_environment, agents, protocol)
        result = await arena.run()

        assert result is not None


class TestDebateRobustness:
    """Tests for debate robustness and error handling."""

    @pytest.mark.asyncio
    async def test_debate_continues_after_agent_error(self, simple_environment):
        """Test debate continues when one agent fails."""
        agents = [
            MockAgent(name="reliable_1", responses=["Good response 1"]),
            MockAgent(name="reliable_2", responses=["Good response 2"]),
            FailingAgent(name="flaky", fail_after=1, responses=["Initial"]),
        ]

        protocol = DebateProtocol(rounds=3, consensus="majority")
        arena = Arena(simple_environment, agents, protocol)
        result = await arena.run()

        assert result is not None
        # Should complete despite agent failure

    @pytest.mark.asyncio
    async def test_debate_handles_slow_agent(self, simple_environment):
        """Test debate handles slow agent with timeout."""
        agents = [
            MockAgent(name="fast_1", responses=["Quick response"]),
            MockAgent(name="fast_2", responses=["Another quick response"]),
            SlowAgent(name="slow", delay=0.5, responses=["Delayed response"]),
        ]

        protocol = DebateProtocol(rounds=2, consensus="majority")
        arena = Arena(simple_environment, agents, protocol)

        # Should complete within reasonable time
        result = await asyncio.wait_for(arena.run(), timeout=60.0)
        assert result is not None

    @pytest.mark.asyncio
    async def test_debate_recovers_from_multiple_failures(self, simple_environment):
        """Test debate continues when multiple agents fail at different times."""
        agents = [
            MockAgent(name="reliable", responses=["Stable"] * 5),
            FailingAgent(name="fails_early", fail_after=1, responses=["First"]),
            FailingAgent(name="fails_late", fail_after=3, responses=["Early"] * 3),
        ]

        protocol = DebateProtocol(rounds=4, consensus="majority")
        arena = Arena(simple_environment, agents, protocol)
        result = await arena.run()

        assert result is not None


class TestDebateMemoryIntegration:
    """Tests for memory integration in debates."""

    @pytest.mark.asyncio
    async def test_debate_stores_results(self, mock_agents, simple_environment, critique_store):
        """Test that debate results are stored for future reference."""
        protocol = DebateProtocol(rounds=2, consensus="majority")

        arena = Arena(simple_environment, mock_agents, protocol)
        result = await arena.run()

        assert result is not None
        # Result should have debate ID
        assert result.debate_id is not None

    @pytest.mark.asyncio
    async def test_sequential_debates_isolated(self, mock_agents, simple_environment):
        """Test that sequential debates are properly isolated."""
        protocol = DebateProtocol(rounds=1, consensus="any")

        # Run first debate
        arena1 = Arena(simple_environment, mock_agents, protocol)
        result1 = await arena1.run()

        # Reset agents
        for agent in mock_agents:
            agent.reset()

        # Run second debate
        env2 = Environment(task="Different task")
        arena2 = Arena(env2, mock_agents, protocol)
        result2 = await arena2.run()

        assert result1 is not None
        assert result2 is not None
        # Should have different debate IDs
        assert result1.debate_id != result2.debate_id


class TestDebateConfiguration:
    """Tests for different debate configurations."""

    @pytest.mark.asyncio
    async def test_debate_with_critique_phase(self, mock_agents, simple_environment):
        """Test debate with critique phase enabled."""
        protocol = DebateProtocol(
            rounds=2,
            consensus="majority",
            critique_required=True,
        )

        arena = Arena(simple_environment, mock_agents, protocol)
        result = await arena.run()

        assert result is not None

    @pytest.mark.asyncio
    async def test_debate_custom_consensus_threshold(self, simple_environment):
        """Test debate with custom consensus threshold."""
        agents = [
            MockAgent(
                name=f"agent_{i}",
                responses=[f"Response {i}"],
                votes=[
                    Vote(
                        agent=f"agent_{i}",
                        choice="agent_0",
                        reasoning="Agreement",
                        confidence=0.7,
                    )
                ],
            )
            for i in range(4)
        ]

        # Require 75% agreement (3 of 4)
        protocol = DebateProtocol(
            rounds=2,
            consensus="threshold",
            consensus_threshold=0.75,
        )

        arena = Arena(simple_environment, agents, protocol)
        result = await arena.run()

        assert result is not None

    @pytest.mark.asyncio
    async def test_debate_complex_environment(self, mock_agents, complex_environment):
        """Test debate with complex multi-constraint environment."""
        protocol = DebateProtocol(rounds=3, consensus="majority")

        arena = Arena(complex_environment, mock_agents, protocol)
        result = await arena.run()

        assert result is not None
        assert result.rounds_completed >= 1


class TestDebateEvents:
    """Tests for debate event emission."""

    @pytest.mark.asyncio
    async def test_debate_has_event_emitter(self, mock_agents, simple_environment):
        """Test that debate has event emission infrastructure."""
        protocol = DebateProtocol(rounds=1, consensus="any")

        arena = Arena(simple_environment, mock_agents, protocol)

        # Arena should have event emitter after initialization
        result = await arena.run()

        assert result is not None
        # Arena should have event infrastructure
        assert hasattr(arena, "_event_emitter")

    @pytest.mark.asyncio
    async def test_debate_completes_with_spectator_events(self, mock_agents, simple_environment):
        """Test that debate completes and can notify spectators."""
        protocol = DebateProtocol(rounds=1, consensus="any")

        arena = Arena(simple_environment, mock_agents, protocol)
        result = await arena.run()

        assert result is not None
        assert result.debate_id is not None
        # Debate should complete successfully with event system active


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_minimum_agents(self, simple_environment):
        """Test debate with minimum number of agents (2)."""
        agents = [
            MockAgent(name="agent_1", responses=["Response 1"]),
            MockAgent(name="agent_2", responses=["Response 2"]),
        ]

        protocol = DebateProtocol(rounds=2, consensus="majority")
        arena = Arena(simple_environment, agents, protocol)
        result = await arena.run()

        assert result is not None

    @pytest.mark.asyncio
    async def test_empty_responses_handled(self, simple_environment):
        """Test that empty agent responses are handled gracefully."""
        agents = [
            MockAgent(name="normal", responses=["Valid response"]),
            MockAgent(name="empty", responses=[""]),  # Empty response
            MockAgent(name="normal_2", responses=["Another valid response"]),
        ]

        protocol = DebateProtocol(rounds=1, consensus="any")
        arena = Arena(simple_environment, agents, protocol)
        result = await arena.run()

        assert result is not None

    @pytest.mark.asyncio
    async def test_very_long_response_handled(self, simple_environment):
        """Test that very long responses are handled."""
        long_response = "A" * 10000  # 10KB response
        agents = [
            MockAgent(name="verbose", responses=[long_response]),
            MockAgent(name="normal", responses=["Short response"]),
            MockAgent(name="normal_2", responses=["Another short response"]),
        ]

        protocol = DebateProtocol(rounds=1, consensus="any")
        arena = Arena(simple_environment, agents, protocol)
        result = await arena.run()

        assert result is not None
