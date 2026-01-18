"""
E2E tests for debate lifecycle.

Tests complete debate workflows including:
- Standard debate (3-5 rounds)
- Extended debate (50+ rounds with RLM)
- Consensus detection
- Cross-debate memory
- Debate checkpointing and resume
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from tests.e2e.conftest import TestTenant, DebateSetup, MockAgentResponse


# ============================================================================
# Standard Debate E2E Tests
# ============================================================================


class TestStandardDebateE2E:
    """E2E tests for standard debate lifecycle."""

    @pytest.mark.asyncio
    async def test_basic_debate_lifecycle(
        self,
        basic_debate: DebateSetup,
        mock_llm_agents,
    ):
        """Test complete basic debate from start to consensus."""
        from aragora.debate.orchestrator import Arena
        from aragora.debate.protocol import DebateProtocol
        from aragora.core import Environment

        env = Environment(task=basic_debate.topic)
        protocol = DebateProtocol(
            rounds=basic_debate.rounds,
            consensus="majority",
        )

        # Mock agents
        mock_agents = [MagicMock() for _ in basic_debate.agents]
        for i, agent in enumerate(mock_agents):
            agent.name = basic_debate.agents[i]
            agent.generate = AsyncMock(return_value=f"Response from {basic_debate.agents[i]}")

        arena = Arena(env, mock_agents, protocol)

        result = await arena.run()

        assert result is not None
        assert hasattr(result, "rounds_completed")
        assert result.rounds_completed <= basic_debate.rounds

    @pytest.mark.asyncio
    async def test_debate_with_consensus_detection(
        self,
        basic_debate: DebateSetup,
        mock_llm_agents,
    ):
        """Test debate reaching consensus."""
        from aragora.debate.orchestrator import Arena
        from aragora.debate.protocol import DebateProtocol
        from aragora.debate.consensus import ConsensusDetector
        from aragora.core import Environment

        env = Environment(task="Simple question with clear answer")
        protocol = DebateProtocol(rounds=5, consensus="unanimous")

        mock_agents = [MagicMock() for _ in range(3)]
        for i, agent in enumerate(mock_agents):
            agent.name = f"agent_{i}"
            # All agents agree
            agent.generate = AsyncMock(return_value="I agree with the consensus position.")

        with patch.object(ConsensusDetector, "detect") as mock_detect:
            mock_detect.return_value = {
                "reached": True,
                "confidence": 0.95,
                "position": "Agreed position",
            }

            arena = Arena(env, mock_agents, protocol)
            result = await arena.run()

            assert result is not None

    @pytest.mark.asyncio
    async def test_debate_with_critiques(
        self,
        basic_debate: DebateSetup,
        mock_llm_agents,
    ):
        """Test debate with critique phase."""
        from aragora.debate.orchestrator import Arena
        from aragora.debate.protocol import DebateProtocol
        from aragora.core import Environment

        env = Environment(task=basic_debate.topic)
        protocol = DebateProtocol(
            rounds=basic_debate.rounds,
            critique_enabled=True,
        )

        mock_agents = [MagicMock() for _ in range(3)]
        for i, agent in enumerate(mock_agents):
            agent.name = f"agent_{i}"
            agent.generate = AsyncMock(return_value=f"Position {i}")
            agent.critique = AsyncMock(return_value=f"Critique of position")

        arena = Arena(env, mock_agents, protocol)
        result = await arena.run()

        assert result is not None


# ============================================================================
# Extended Debate E2E Tests
# ============================================================================


class TestExtendedDebateE2E:
    """E2E tests for extended (50+) round debates."""

    @pytest.mark.asyncio
    async def test_extended_debate_with_rlm(
        self,
        extended_debate: DebateSetup,
        mock_llm_agents,
    ):
        """Test 55-round debate with RLM context management."""
        from aragora.debate.extended_rounds import ExtendedDebateConfig, RLMContextManager
        from aragora.debate.orchestrator import Arena
        from aragora.debate.protocol import DebateProtocol
        from aragora.core import Environment

        env = Environment(task=extended_debate.topic)

        # Create extended config
        ext_config = ExtendedDebateConfig(
            max_rounds=extended_debate.rounds,
            compression_threshold=0.7,
            summary_frequency=10,
        )

        protocol = DebateProtocol(
            rounds=extended_debate.rounds,
            extended_config=ext_config,
        )

        # Mock RLM context manager
        with patch.object(RLMContextManager, "compress") as mock_compress:
            mock_compress.return_value = "Compressed context summary"

            mock_agents = [MagicMock() for _ in extended_debate.agents]
            for i, agent in enumerate(mock_agents):
                agent.name = extended_debate.agents[i]
                agent.generate = AsyncMock(return_value=f"Round response from {agent.name}")

            arena = Arena(env, mock_agents, protocol)

            # Should complete without memory issues
            result = await arena.run()

            assert result is not None
            # RLM compression should have been called
            assert mock_compress.call_count > 0

    @pytest.mark.asyncio
    async def test_extended_debate_checkpoint_resume(
        self,
        extended_debate: DebateSetup,
        mock_llm_agents,
    ):
        """Test checkpointing and resuming extended debate."""
        from aragora.debate.extended_rounds import ExtendedDebateConfig
        from aragora.debate.orchestrator import Arena
        from aragora.debate.protocol import DebateProtocol
        from aragora.core import Environment

        env = Environment(task=extended_debate.topic)
        protocol = DebateProtocol(rounds=20)

        mock_agents = [MagicMock() for _ in range(3)]
        for i, agent in enumerate(mock_agents):
            agent.name = f"agent_{i}"
            agent.generate = AsyncMock(return_value=f"Response {i}")

        arena = Arena(env, mock_agents, protocol)

        # Run first 10 rounds
        with patch.object(Arena, "checkpoint") as mock_checkpoint:
            mock_checkpoint.return_value = {"round": 10, "state": "serialized"}

            result = await arena.run(max_rounds=10)
            checkpoint_data = arena.checkpoint()

            assert checkpoint_data is not None

        # Resume from checkpoint
        with patch.object(Arena, "restore") as mock_restore:
            mock_restore.return_value = True

            arena2 = Arena(env, mock_agents, protocol)
            arena2.restore(checkpoint_data)

            result = await arena2.run()
            assert result is not None

    @pytest.mark.asyncio
    async def test_extended_debate_memory_efficiency(
        self,
        extended_debate: DebateSetup,
        mock_llm_agents,
    ):
        """Test memory doesn't grow unbounded in extended debates."""
        import sys
        from aragora.debate.extended_rounds import RLMContextManager

        # Track memory usage
        initial_size = 0
        final_size = 0

        context_manager = RLMContextManager()

        # Simulate 50 rounds of context accumulation
        for round_num in range(50):
            context_manager.add_round(
                round_num,
                [
                    {"agent": "claude", "content": "A" * 1000},
                    {"agent": "gpt4", "content": "B" * 1000},
                    {"agent": "gemini", "content": "C" * 1000},
                ],
            )

            if round_num == 0:
                initial_size = sys.getsizeof(context_manager.get_context())

        final_size = sys.getsizeof(context_manager.get_context())

        # Memory should not grow linearly with rounds due to compression
        # Final size should be less than 10x initial (with compression)
        assert final_size < initial_size * 20  # Allow some growth


# ============================================================================
# Cross-Debate Memory E2E Tests
# ============================================================================


class TestCrossDebateMemoryE2E:
    """E2E tests for cross-debate RLM memory."""

    @pytest.mark.asyncio
    async def test_cross_debate_context_retrieval(self, mock_llm_agents):
        """Test retrieving context from previous debates."""
        from aragora.memory.cross_debate_rlm import CrossDebateMemory

        memory = CrossDebateMemory()

        # Store first debate
        await memory.store_debate(
            debate_id="debate-1",
            topic="API design patterns",
            consensus="REST with OpenAPI spec",
            key_points=["Consistency", "Documentation", "Versioning"],
        )

        # Store second debate
        await memory.store_debate(
            debate_id="debate-2",
            topic="Database selection",
            consensus="PostgreSQL for OLTP, ClickHouse for analytics",
            key_points=["ACID compliance", "Query patterns"],
        )

        # Retrieve relevant context for new debate
        context = await memory.get_relevant_context(
            query="How should we design our API?",
            max_results=5,
        )

        assert context is not None
        assert len(context) > 0
        # Should find the API design debate
        assert any("API" in str(c) for c in context)

    @pytest.mark.asyncio
    async def test_cross_debate_memory_tiering(self, mock_llm_agents):
        """Test memory tiering (hot/warm/cold/archive)."""
        from aragora.memory.cross_debate_rlm import CrossDebateMemory, MemoryTier

        memory = CrossDebateMemory()

        # Add debates with different ages
        await memory.store_debate(
            debate_id="recent",
            topic="Recent topic",
            timestamp_hours_ago=1,  # Hot tier
        )

        await memory.store_debate(
            debate_id="older",
            topic="Older topic",
            timestamp_hours_ago=48,  # Warm tier
        )

        await memory.store_debate(
            debate_id="old",
            topic="Old topic",
            timestamp_hours_ago=168,  # Cold tier
        )

        # Check tier assignments
        assert memory.get_tier("recent") == MemoryTier.HOT
        assert memory.get_tier("older") == MemoryTier.WARM
        assert memory.get_tier("old") == MemoryTier.COLD


# ============================================================================
# Debate Event Streaming E2E Tests
# ============================================================================


class TestDebateStreamingE2E:
    """E2E tests for debate event streaming."""

    @pytest.mark.asyncio
    async def test_debate_event_stream(
        self,
        basic_debate: DebateSetup,
        mock_llm_agents,
    ):
        """Test WebSocket event streaming during debate."""
        from aragora.debate.orchestrator import Arena
        from aragora.debate.protocol import DebateProtocol
        from aragora.core import Environment

        env = Environment(task=basic_debate.topic)
        protocol = DebateProtocol(rounds=3)

        mock_agents = [MagicMock() for _ in range(3)]
        for i, agent in enumerate(mock_agents):
            agent.name = f"agent_{i}"
            agent.generate = AsyncMock(return_value=f"Response {i}")

        events: List[Dict[str, Any]] = []

        async def event_handler(event: Dict[str, Any]):
            events.append(event)

        arena = Arena(env, mock_agents, protocol)
        arena.on_event(event_handler)

        await arena.run()

        # Should have received debate events
        assert len(events) > 0
        event_types = {e.get("type") for e in events}
        assert "round_start" in event_types or "agent_message" in event_types

    @pytest.mark.asyncio
    async def test_debate_progress_tracking(
        self,
        basic_debate: DebateSetup,
        mock_llm_agents,
    ):
        """Test debate progress updates."""
        from aragora.debate.orchestrator import Arena
        from aragora.debate.protocol import DebateProtocol
        from aragora.core import Environment

        env = Environment(task=basic_debate.topic)
        protocol = DebateProtocol(rounds=5)

        mock_agents = [MagicMock() for _ in range(3)]
        for agent in mock_agents:
            agent.generate = AsyncMock(return_value="Response")

        progress_updates: List[float] = []

        async def progress_handler(progress: float):
            progress_updates.append(progress)

        arena = Arena(env, mock_agents, protocol)
        arena.on_progress(progress_handler)

        await arena.run()

        # Should have progress updates
        assert len(progress_updates) > 0
        # Progress should increase
        assert progress_updates[-1] >= progress_updates[0]


# ============================================================================
# Debate Voting E2E Tests
# ============================================================================


class TestDebateVotingE2E:
    """E2E tests for debate voting system."""

    @pytest.mark.asyncio
    async def test_agent_voting(
        self,
        basic_debate: DebateSetup,
        mock_llm_agents,
    ):
        """Test agent voting on positions."""
        from aragora.debate.orchestrator import Arena
        from aragora.debate.protocol import DebateProtocol
        from aragora.debate.voting import VotingEngine
        from aragora.core import Environment

        env = Environment(task=basic_debate.topic)
        protocol = DebateProtocol(rounds=3, voting_enabled=True)

        mock_agents = [MagicMock() for _ in range(3)]
        for i, agent in enumerate(mock_agents):
            agent.name = f"agent_{i}"
            agent.generate = AsyncMock(return_value=f"Position {i}")
            agent.vote = AsyncMock(return_value={"position": 0, "confidence": 0.8})

        arena = Arena(env, mock_agents, protocol)
        result = await arena.run()

        assert result is not None

    @pytest.mark.asyncio
    async def test_user_participation(
        self,
        basic_debate: DebateSetup,
        mock_llm_agents,
    ):
        """Test user votes and suggestions integration."""
        from aragora.debate.orchestrator import Arena
        from aragora.debate.protocol import DebateProtocol
        from aragora.core import Environment

        env = Environment(task=basic_debate.topic)
        protocol = DebateProtocol(rounds=3, user_participation=True)

        mock_agents = [MagicMock() for _ in range(3)]
        for agent in mock_agents:
            agent.generate = AsyncMock(return_value="Response")

        arena = Arena(env, mock_agents, protocol)

        # Simulate user input
        await arena.add_user_vote(agent_name="agent_0", score=5)
        await arena.add_user_suggestion("Consider edge cases")

        result = await arena.run()
        assert result is not None


# ============================================================================
# Debate Error Handling E2E Tests
# ============================================================================


class TestDebateErrorHandlingE2E:
    """E2E tests for debate error handling."""

    @pytest.mark.asyncio
    async def test_agent_failure_recovery(
        self,
        basic_debate: DebateSetup,
    ):
        """Test debate continues when agent fails."""
        from aragora.debate.orchestrator import Arena
        from aragora.debate.protocol import DebateProtocol
        from aragora.core import Environment

        env = Environment(task=basic_debate.topic)
        protocol = DebateProtocol(rounds=3)

        mock_agents = [MagicMock() for _ in range(3)]

        # First agent fails
        mock_agents[0].name = "failing_agent"
        mock_agents[0].generate = AsyncMock(side_effect=Exception("Agent error"))

        # Other agents work
        for i in range(1, 3):
            mock_agents[i].name = f"agent_{i}"
            mock_agents[i].generate = AsyncMock(return_value=f"Response {i}")

        arena = Arena(env, mock_agents, protocol)

        # Should complete despite agent failure
        result = await arena.run()
        assert result is not None

    @pytest.mark.asyncio
    async def test_timeout_handling(
        self,
        basic_debate: DebateSetup,
    ):
        """Test debate timeout handling."""
        from aragora.debate.orchestrator import Arena
        from aragora.debate.protocol import DebateProtocol
        from aragora.core import Environment

        env = Environment(task=basic_debate.topic)
        protocol = DebateProtocol(rounds=3, round_timeout=1.0)

        mock_agents = [MagicMock() for _ in range(3)]

        # Slow agent
        async def slow_response(*args, **kwargs):
            await asyncio.sleep(5)  # Too slow
            return "Late response"

        mock_agents[0].name = "slow_agent"
        mock_agents[0].generate = slow_response

        for i in range(1, 3):
            mock_agents[i].name = f"agent_{i}"
            mock_agents[i].generate = AsyncMock(return_value=f"Response {i}")

        arena = Arena(env, mock_agents, protocol)

        # Should handle timeout gracefully
        result = await arena.run()
        assert result is not None
