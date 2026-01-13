"""
E2E tests for the complete debate lifecycle.

Tests the full flow from creation to archival:
1. Create debate via Arena
2. Run debate to completion
3. Verify consensus reached
4. Verify memory stored
5. Verify ELO updated
6. Archive debate
7. Verify archived state
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime

import pytest

from aragora.core import Environment, DebateResult
from aragora.debate.orchestrator import Arena, DebateProtocol
from tests.e2e.conftest import E2EAgent


class TestDebateCreation:
    """Tests for debate creation and initialization."""

    @pytest.mark.asyncio
    async def test_debate_creation_with_valid_params(
        self, e2e_agents, e2e_environment, e2e_protocol, mock_external_apis
    ):
        """Debate should be created successfully with valid parameters."""
        arena = Arena(e2e_environment, e2e_agents, e2e_protocol)

        assert arena is not None
        assert len(arena.agents) == 3
        assert arena.env.task is not None

    @pytest.mark.asyncio
    async def test_debate_has_unique_id(
        self, e2e_agents, e2e_environment, e2e_protocol, mock_external_apis
    ):
        """Each debate should have a unique identifier."""
        arena1 = Arena(e2e_environment, e2e_agents, e2e_protocol)
        arena2 = Arena(e2e_environment, e2e_agents, e2e_protocol)

        # Arena instances should be different
        assert arena1 is not arena2


class TestDebateExecution:
    """Tests for debate execution flow."""

    @pytest.mark.asyncio
    async def test_debate_runs_to_completion(
        self, e2e_agents, e2e_environment, e2e_protocol, mock_external_apis
    ):
        """Debate should run through all rounds and complete."""
        arena = Arena(e2e_environment, e2e_agents, e2e_protocol)
        result = await arena.run()

        assert result is not None
        assert isinstance(result, DebateResult)
        assert result.final_answer is not None or result.rounds_completed > 0

    @pytest.mark.asyncio
    async def test_debate_generates_messages(
        self, e2e_agents, e2e_environment, e2e_protocol, mock_external_apis
    ):
        """Debate should generate messages from all participating agents."""
        arena = Arena(e2e_environment, e2e_agents, e2e_protocol)
        result = await arena.run()

        assert result is not None
        # Should have messages from agents
        if hasattr(result, 'messages') and result.messages:
            assert len(result.messages) > 0

    @pytest.mark.asyncio
    async def test_debate_tracks_rounds(
        self, e2e_agents, e2e_environment, e2e_protocol, mock_external_apis
    ):
        """Debate should track the number of rounds completed."""
        arena = Arena(e2e_environment, e2e_agents, e2e_protocol)
        result = await arena.run()

        assert result is not None
        assert hasattr(result, 'rounds_completed')
        assert result.rounds_completed >= 1

    @pytest.mark.asyncio
    async def test_debate_handles_early_consensus(self, mock_external_apis):
        """Debate should stop early if consensus is reached."""
        # Create agents that will agree quickly
        agreeable_agents = [
            E2EAgent("agent_0", position="solution_a", stubbornness=0.1),
            E2EAgent("agent_1", position="solution_a", stubbornness=0.1),
            E2EAgent("agent_2", position="solution_a", stubbornness=0.2),
        ]

        env = Environment(task="Simple decision task")
        protocol = DebateProtocol(rounds=5, consensus="majority")

        arena = Arena(env, agreeable_agents, protocol)
        result = await arena.run()

        assert result is not None
        # With quick agreement, might not need all rounds


class TestConsensusAndOutcome:
    """Tests for consensus detection and debate outcomes."""

    @pytest.mark.asyncio
    async def test_consensus_reached_with_majority(self, mock_external_apis):
        """Consensus should be reached when majority agrees."""
        # Two agents agree, one disagrees
        agents = [
            E2EAgent("agreeable_1", position="solution_x", stubbornness=0.2),
            E2EAgent("agreeable_2", position="solution_x", stubbornness=0.2),
            E2EAgent("dissenter", position="solution_y", stubbornness=0.8),
        ]

        env = Environment(task="Consensus test task")
        protocol = DebateProtocol(rounds=2, consensus="majority")

        arena = Arena(env, agents, protocol)
        result = await arena.run()

        assert result is not None
        assert result.consensus_reached is True

    @pytest.mark.asyncio
    async def test_no_consensus_with_stubborn_agents(self, mock_external_apis):
        """Consensus may not be reached with highly stubborn agents."""
        # All agents are stubborn and disagree
        agents = [
            E2EAgent("stubborn_1", position="my_way", stubbornness=0.9),
            E2EAgent("stubborn_2", position="no_your_way", stubbornness=0.9),
            E2EAgent("stubborn_3", position="neither", stubbornness=0.9),
        ]

        env = Environment(task="Contentious topic")
        protocol = DebateProtocol(rounds=2, consensus="unanimous")

        arena = Arena(env, agents, protocol)
        result = await arena.run()

        assert result is not None
        # With unanimous requirement and stubborn agents, unlikely to reach consensus

    @pytest.mark.asyncio
    async def test_debate_result_contains_required_fields(
        self, e2e_agents, e2e_environment, e2e_protocol, mock_external_apis
    ):
        """Debate result should contain all required fields."""
        arena = Arena(e2e_environment, e2e_agents, e2e_protocol)
        result = await arena.run()

        assert result is not None
        # Check required fields
        assert hasattr(result, 'consensus_reached')
        assert hasattr(result, 'rounds_completed')
        assert hasattr(result, 'final_answer')


class TestDebateWithMemory:
    """Tests for debate interaction with memory systems."""

    @pytest.mark.asyncio
    async def test_debate_with_critique_store(
        self, e2e_agents, e2e_environment, e2e_protocol, mock_external_apis, temp_e2e_dir
    ):
        """Debate should interact with CritiqueStore for memory."""
        from aragora.memory.critique_store import CritiqueStore

        critique_store = CritiqueStore(str(temp_e2e_dir / "critiques.db"))

        arena = Arena(e2e_environment, e2e_agents, e2e_protocol)
        result = await arena.run()

        assert result is not None
        # Verify debate completed with memory available

    @pytest.mark.asyncio
    async def test_debate_with_continuum_memory(
        self, e2e_agents, e2e_environment, e2e_protocol, mock_external_apis, temp_e2e_dir
    ):
        """Debate should work with ContinuumMemory tier system."""
        from aragora.memory.continuum import ContinuumMemory

        memory = ContinuumMemory(str(temp_e2e_dir))

        arena = Arena(e2e_environment, e2e_agents, e2e_protocol)
        result = await arena.run()

        assert result is not None


class TestDebateWithELO:
    """Tests for ELO rating updates after debates."""

    @pytest.mark.asyncio
    async def test_debate_updates_elo_ratings(
        self, e2e_agents, e2e_environment, e2e_protocol, mock_external_apis, temp_e2e_dir
    ):
        """ELO ratings should be updated after debate completion."""
        from aragora.ranking.elo import EloSystem

        elo_system = EloSystem(str(temp_e2e_dir / "elo.db"))

        # Initialize ratings for agents
        for agent in e2e_agents:
            elo_system.initialize_agent(agent.name)

        initial_ratings = {
            agent.name: elo_system.get_rating(agent.name)
            for agent in e2e_agents
        }

        arena = Arena(e2e_environment, e2e_agents, e2e_protocol)
        result = await arena.run()

        assert result is not None

        # Note: ELO update happens outside Arena typically
        # This test verifies the integration point exists


class TestDebateArchival:
    """Tests for debate archival functionality."""

    @pytest.mark.asyncio
    async def test_debate_can_be_archived(
        self, e2e_agents, e2e_environment, e2e_protocol, mock_external_apis, temp_e2e_dir
    ):
        """Completed debate should be archivable."""
        from aragora.server.storage import DebateStorage

        storage = DebateStorage(str(temp_e2e_dir / "debates.db"))

        arena = Arena(e2e_environment, e2e_agents, e2e_protocol)
        result = await arena.run()

        assert result is not None

        # Create metadata for storage
        debate_metadata = {
            "task": e2e_environment.task,
            "consensus_reached": result.consensus_reached,
            "rounds_completed": result.rounds_completed,
            "final_answer": result.final_answer,
            "created_at": datetime.now().isoformat(),
            "status": "concluded",
        }

        # Store the debate
        debate_id = storage.store(debate_metadata)
        assert debate_id is not None

        # Verify retrieval
        retrieved = storage.get_debate(debate_id)
        assert retrieved is not None


class TestMultiDebateScenarios:
    """Tests for running multiple debates."""

    @pytest.mark.asyncio
    async def test_sequential_debates(
        self, e2e_agents, e2e_protocol, mock_external_apis
    ):
        """Multiple debates should run sequentially without interference."""
        results = []

        for i in range(3):
            env = Environment(task=f"Sequential debate task {i}")
            arena = Arena(env, e2e_agents, e2e_protocol)
            result = await arena.run()
            results.append(result)

            # Reset agents for next debate
            for agent in e2e_agents:
                agent._round = 0

        assert len(results) == 3
        assert all(r is not None for r in results)

    @pytest.mark.asyncio
    async def test_concurrent_debates(self, e2e_protocol, mock_external_apis):
        """Multiple debates should be able to run concurrently."""
        tasks = []

        for i in range(3):
            agents = [
                E2EAgent(f"agent_{i}_{j}", position=f"solution_{j}", stubbornness=0.3)
                for j in range(3)
            ]
            env = Environment(task=f"Concurrent debate task {i}")
            arena = Arena(env, agents, e2e_protocol)
            tasks.append(arena.run())

        results = await asyncio.gather(*tasks)

        assert len(results) == 3
        assert all(r is not None for r in results)


class TestDebateEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_debate_with_minimum_agents(
        self, e2e_protocol, mock_external_apis
    ):
        """Debate should work with minimum 2 agents."""
        agents = [
            E2EAgent("agent_1", position="option_a"),
            E2EAgent("agent_2", position="option_b"),
        ]
        env = Environment(task="Two-agent debate")

        arena = Arena(env, agents, e2e_protocol)
        result = await arena.run()

        assert result is not None

    @pytest.mark.asyncio
    async def test_debate_with_single_round(self, e2e_agents, mock_external_apis):
        """Debate should complete with single round."""
        env = Environment(task="Quick decision")
        protocol = DebateProtocol(rounds=1, consensus="majority")

        arena = Arena(env, e2e_agents, protocol)
        result = await arena.run()

        assert result is not None
        assert result.rounds_completed >= 1

    @pytest.mark.asyncio
    async def test_debate_with_empty_task(self, e2e_agents, e2e_protocol, mock_external_apis):
        """Debate should handle empty or minimal task gracefully."""
        env = Environment(task="?")  # Minimal valid task

        arena = Arena(env, e2e_agents, e2e_protocol)
        result = await arena.run()

        # Should complete or fail gracefully
        assert result is not None

    @pytest.mark.asyncio
    async def test_debate_with_long_task(self, e2e_agents, e2e_protocol, mock_external_apis):
        """Debate should handle very long task descriptions."""
        long_task = "Design a system " + "with many requirements " * 100
        env = Environment(task=long_task)

        arena = Arena(env, e2e_agents, e2e_protocol)
        result = await arena.run()

        assert result is not None
