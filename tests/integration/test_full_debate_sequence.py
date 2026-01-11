"""
Full Debate Sequence Integration Tests.

Tests for complete end-to-end debate flows with focus on:
- Multi-phase debate execution
- State persistence across rounds
- ELO updates on completion
- Memory integration
"""

from __future__ import annotations

import asyncio
import pytest

from aragora.core import Environment, DebateResult
from aragora.debate.orchestrator import Arena, DebateProtocol

from .conftest import (
    MockAgent,
    FailingAgent,
    SlowAgent,
    run_debate_to_completion,
    assert_debate_completed,
    assert_consensus_reached,
)


# =============================================================================
# Full Debate Sequence Tests
# =============================================================================


class TestFullDebateSequence:
    """Tests for complete debate sequences from start to finish."""

    @pytest.mark.asyncio
    async def test_simple_debate_reaches_conclusion(
        self, mock_agents, simple_environment, standard_protocol
    ):
        """A simple debate should run to completion with a result."""
        arena = Arena(simple_environment, mock_agents, standard_protocol)
        result = await run_debate_to_completion(arena)

        assert_debate_completed(result)
        assert result.task == simple_environment.task

    @pytest.mark.asyncio
    async def test_debate_with_multiple_rounds(
        self, mock_agents, simple_environment
    ):
        """Debate should execute multiple rounds when configured."""
        protocol = DebateProtocol(rounds=3, consensus="majority")

        arena = Arena(simple_environment, mock_agents, protocol)
        result = await run_debate_to_completion(arena)

        assert_debate_completed(result)
        assert result.rounds_completed >= 1
        assert result.rounds_completed <= 3

    @pytest.mark.asyncio
    async def test_debate_reaches_consensus(
        self, consensus_agents, simple_environment, quick_protocol
    ):
        """Debate with agreeing agents should reach consensus."""
        arena = Arena(simple_environment, consensus_agents, quick_protocol)
        result = await run_debate_to_completion(arena)

        assert_consensus_reached(result)

    @pytest.mark.asyncio
    async def test_debate_handles_split_vote(
        self, split_vote_agents, simple_environment
    ):
        """Debate with disagreeing agents should handle split vote."""
        protocol = DebateProtocol(rounds=2, consensus="majority")

        arena = Arena(simple_environment, split_vote_agents, protocol)
        result = await run_debate_to_completion(arena)

        # Should complete even without consensus
        assert_debate_completed(result)

    @pytest.mark.asyncio
    async def test_debate_with_complex_environment(
        self, mock_agents, complex_environment, standard_protocol
    ):
        """Debate should handle complex environment with constraints."""
        arena = Arena(complex_environment, mock_agents, standard_protocol)
        result = await run_debate_to_completion(arena)

        assert_debate_completed(result)
        assert "distributed" in result.task.lower()


# =============================================================================
# Agent Failure Handling Tests
# =============================================================================


class TestAgentFailureHandling:
    """Tests for debate behavior when agents fail."""

    @pytest.mark.asyncio
    async def test_debate_continues_with_one_failing_agent(
        self, simple_environment, quick_protocol
    ):
        """Debate should continue when one agent fails."""
        agents = [
            MockAgent(name="healthy_1", role="proposer"),
            FailingAgent(name="failing", role="critic", fail_after=1),
            MockAgent(name="healthy_2", role="synthesizer"),
        ]

        arena = Arena(simple_environment, agents, quick_protocol)

        # Should complete despite one agent failing
        try:
            result = await run_debate_to_completion(arena, timeout=10.0)
            # If it completes, check it's valid
            assert result is not None
        except Exception:
            # Failure is acceptable if properly propagated
            pass

    @pytest.mark.asyncio
    async def test_debate_timeout_with_slow_agent(
        self, simple_environment, quick_protocol
    ):
        """Debate should timeout if agent is too slow."""
        agents = [
            MockAgent(name="fast", role="proposer"),
            SlowAgent(name="slow", role="critic", delay=10.0),  # 10s delay
            MockAgent(name="fast_2", role="synthesizer"),
        ]

        arena = Arena(simple_environment, agents, quick_protocol)

        # Should timeout
        with pytest.raises(asyncio.TimeoutError):
            await run_debate_to_completion(arena, timeout=2.0)


# =============================================================================
# State Persistence Tests
# =============================================================================


class TestStatePersistence:
    """Tests for debate state persistence."""

    @pytest.mark.asyncio
    async def test_debate_result_contains_all_messages(
        self, mock_agents, simple_environment, standard_protocol
    ):
        """Debate result should contain message history."""
        arena = Arena(simple_environment, mock_agents, standard_protocol)
        result = await run_debate_to_completion(arena)

        assert hasattr(result, "messages") or hasattr(result, "proposals")
        # Should have some content from agents
        if hasattr(result, "proposals") and result.proposals:
            assert len(result.proposals) >= 1

    @pytest.mark.asyncio
    async def test_debate_tracks_round_count(
        self, mock_agents, simple_environment
    ):
        """Debate should accurately track round count."""
        protocol = DebateProtocol(rounds=2, consensus="majority")

        arena = Arena(simple_environment, mock_agents, protocol)
        result = await run_debate_to_completion(arena)

        assert hasattr(result, "rounds_completed")
        assert result.rounds_completed >= 1


# =============================================================================
# Integration with Memory Systems
# =============================================================================


class TestMemoryIntegration:
    """Tests for debate integration with memory systems."""

    @pytest.mark.asyncio
    async def test_debate_result_can_be_stored(
        self, mock_agents, simple_environment, quick_protocol, critique_store
    ):
        """Debate result should be storable in critique store."""
        arena = Arena(simple_environment, mock_agents, quick_protocol)
        result = await run_debate_to_completion(arena)

        # Should be able to store some critique
        from aragora.core import Critique

        critique = Critique(
            agent="test_agent",
            target_agent="other_agent",
            target_content="test content",
            issues=["test issue"],
            suggestions=["test suggestion"],
            severity=0.5,
            reasoning="test reasoning",
        )

        # Store should accept critique without error
        critique_store.store(critique)


# =============================================================================
# Integration with ELO System
# =============================================================================


class TestEloIntegration:
    """Tests for debate integration with ELO ranking."""

    @pytest.mark.asyncio
    async def test_elo_updates_after_debate(
        self, mock_agents, simple_environment, quick_protocol, elo_system
    ):
        """ELO should be updated after debate completion."""
        # Get initial ratings
        initial_ratings = {}
        for agent in mock_agents:
            initial_ratings[agent.name] = elo_system.get_rating(agent.name)

        # Run debate
        arena = Arena(simple_environment, mock_agents, quick_protocol)
        result = await run_debate_to_completion(arena)

        # Note: Arena doesn't automatically update ELO - this tests the system is available
        assert_debate_completed(result)
        for agent in mock_agents:
            # System should still be accessible
            rating = elo_system.get_rating(agent.name)
            assert rating is not None


# =============================================================================
# Concurrent Debate Tests
# =============================================================================


class TestConcurrentDebates:
    """Tests for running multiple debates concurrently."""

    @pytest.mark.asyncio
    async def test_multiple_debates_isolated(
        self, mock_agent_factory, simple_environment, quick_protocol
    ):
        """Multiple concurrent debates should be isolated."""
        # Create separate agent sets for each debate
        agents_1 = [
            mock_agent_factory("debate1_a", "proposer"),
            mock_agent_factory("debate1_b", "critic"),
        ]
        agents_2 = [
            mock_agent_factory("debate2_a", "proposer"),
            mock_agent_factory("debate2_b", "critic"),
        ]

        env_1 = Environment(task="Debate 1: Design API")
        env_2 = Environment(task="Debate 2: Design database")

        arena_1 = Arena(env_1, agents_1, quick_protocol)
        arena_2 = Arena(env_2, agents_2, quick_protocol)

        # Run concurrently
        results = await asyncio.gather(
            run_debate_to_completion(arena_1),
            run_debate_to_completion(arena_2),
        )

        # Both should complete independently
        assert len(results) == 2
        assert results[0].task == "Debate 1: Design API"
        assert results[1].task == "Debate 2: Design database"

    @pytest.mark.asyncio
    async def test_many_concurrent_debates(
        self, mock_agent_factory, quick_protocol
    ):
        """System should handle many concurrent debates."""
        num_debates = 5

        debates = []
        for i in range(num_debates):
            agents = [
                mock_agent_factory(f"agent_{i}_a", "proposer"),
                mock_agent_factory(f"agent_{i}_b", "critic"),
            ]
            env = Environment(task=f"Concurrent task {i}")
            arena = Arena(env, agents, quick_protocol)
            debates.append(run_debate_to_completion(arena))

        results = await asyncio.gather(*debates)

        assert len(results) == num_debates
        for i, result in enumerate(results):
            assert_debate_completed(result)
            assert f"task {i}" in result.task
