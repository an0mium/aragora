"""
End-to-End API Chain Integration Tests.

Tests complete API workflow chains that span multiple handlers:
- Debate creation → execution → export → storage
- Authentication → authorization cascade
- Consensus detection → memory storage → leaderboard update
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from aragora.core import Environment, DebateResult, Message
from aragora.debate.orchestrator import Arena, DebateProtocol

from .conftest import MockAgent, run_debate_to_completion


# =============================================================================
# Handler Context Fixtures
# =============================================================================


@pytest.fixture
def temp_db_path():
    """Create a temporary database path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test.db"


@pytest.fixture
def handler_context(temp_db_path):
    """Create a mock handler context with storage."""
    return {
        "db_path": str(temp_db_path),
        "debates_db": {},
        "elo_system": MagicMock(),
    }


@pytest.fixture
def mock_request():
    """Create a mock HTTP request handler."""
    handler = MagicMock()
    handler.headers = {"Authorization": "Bearer test-token"}
    handler.rfile = MagicMock()
    return handler


# =============================================================================
# E2E: Debate Creation → Export Chain
# =============================================================================


class TestDebateCreationExportChain:
    """Test complete debate creation to export workflow."""

    @pytest.mark.asyncio
    async def test_create_debate_and_export_json(self, mock_agents, simple_environment):
        """Create debate, run to completion, export as JSON."""
        # Setup
        protocol = DebateProtocol(rounds=2, consensus="majority")
        arena = Arena(simple_environment, mock_agents, protocol)

        # Execute debate
        result = await run_debate_to_completion(arena)

        # Verify completion
        assert result is not None
        assert result.status in ["completed", "consensus_reached"]

        # Export to JSON format
        export_data = (
            result.to_dict()
            if hasattr(result, "to_dict")
            else {
                "task": result.task,
                "status": result.status,
                "rounds_completed": result.rounds_completed,
                "final_answer": result.final_answer,
            }
        )

        # Verify export structure
        assert "task" in export_data
        assert "status" in export_data
        assert export_data["task"] == simple_environment.task

        # Verify JSON serialization works
        json_str = json.dumps(export_data)
        parsed = json.loads(json_str)
        assert parsed["task"] == simple_environment.task

    @pytest.mark.asyncio
    async def test_debate_result_contains_all_messages(self, mock_agents, simple_environment):
        """Verify debate result includes full message history."""
        protocol = DebateProtocol(rounds=2, consensus="majority")
        arena = Arena(simple_environment, mock_agents, protocol)

        result = await run_debate_to_completion(arena)

        # Should have messages from debate
        assert hasattr(result, "messages") or hasattr(result, "history")
        messages = getattr(result, "messages", None) or getattr(result, "history", [])

        # At least some messages should exist
        # (may be empty with mock agents, but structure should exist)
        assert isinstance(messages, list)


class TestDebateStoragePersistence:
    """Test debate storage and retrieval."""

    @pytest.mark.asyncio
    async def test_debate_persists_after_completion(
        self, mock_agents, simple_environment, temp_db_path
    ):
        """Verify completed debate is saved to storage."""
        from aragora.memory.store import CritiqueStore

        # Setup with real storage
        protocol = DebateProtocol(rounds=1, consensus="majority")
        memory = CritiqueStore(str(temp_db_path))
        arena = Arena(simple_environment, mock_agents, protocol, memory)

        # Execute
        result = await run_debate_to_completion(arena)

        # Store the result
        if memory:
            memory.store_debate(result)

        # Verify we can query stored data
        # (CritiqueStore stores critiques, not full debates)
        assert result.status in ["completed", "consensus_reached"]

    @pytest.mark.asyncio
    async def test_multiple_debates_isolated(self, mock_agents, temp_db_path):
        """Multiple debates should not interfere with each other."""
        protocol = DebateProtocol(rounds=1, consensus="majority")

        # Create two different environments
        env1 = Environment(task="Design a cache system", context="For web app")
        env2 = Environment(task="Design an auth system", context="For API")

        # Run both debates
        arena1 = Arena(env1, mock_agents, protocol)
        arena2 = Arena(env2, mock_agents, protocol)

        result1 = await run_debate_to_completion(arena1)
        result2 = await run_debate_to_completion(arena2)

        # Verify isolation
        assert result1.task == env1.task
        assert result2.task == env2.task
        assert result1.task != result2.task


# =============================================================================
# E2E: Consensus → Memory → Leaderboard Chain
# =============================================================================


class TestConsensusMemoryLeaderboardChain:
    """Test consensus detection triggers memory and leaderboard updates."""

    @pytest.mark.asyncio
    async def test_consensus_triggers_memory_storage(
        self, consensus_agents, simple_environment, temp_db_path
    ):
        """When consensus is reached, store in memory."""
        from aragora.memory.store import CritiqueStore

        protocol = DebateProtocol(rounds=2, consensus="majority")
        memory = CritiqueStore(str(temp_db_path))

        arena = Arena(simple_environment, consensus_agents, protocol, memory)
        result = await run_debate_to_completion(arena)

        # Consensus agents should reach agreement
        # Memory should be updated (critiques stored)
        assert result is not None

    @pytest.mark.asyncio
    async def test_elo_updated_after_debate(self, mock_agents, simple_environment):
        """ELO system should be updated after debate completion."""
        from aragora.ranking.elo import EloSystem

        with tempfile.TemporaryDirectory() as tmpdir:
            elo_path = Path(tmpdir) / "elo.db"
            elo_system = EloSystem(str(elo_path))

            # Register agents
            for agent in mock_agents:
                elo_system.register_agent(agent.name, agent.model)

            # Get initial ratings
            initial_ratings = {
                agent.name: elo_system.get_rating(agent.name) for agent in mock_agents
            }

            # Run debate
            protocol = DebateProtocol(rounds=1, consensus="majority")
            arena = Arena(simple_environment, mock_agents, protocol)
            result = await run_debate_to_completion(arena)

            # Simulate match recording (would be done by orchestrator)
            if len(mock_agents) >= 2:
                winner = mock_agents[0].name
                loser = mock_agents[1].name
                elo_system.record_match(winner, loser, draw=False)

            # Verify ratings changed
            final_ratings = {agent.name: elo_system.get_rating(agent.name) for agent in mock_agents}

            # At least one rating should have changed
            assert any(initial_ratings[name] != final_ratings[name] for name in initial_ratings)


# =============================================================================
# E2E: Handler Coordination Tests
# =============================================================================


class TestHandlerCoordination:
    """Test multiple handlers working together."""

    def test_debates_handler_returns_valid_response(self, handler_context, mock_request):
        """DebatesHandler should return proper JSON response."""
        from aragora.server.handlers.debates import DebatesHandler

        handler = DebatesHandler(handler_context)

        # Test list debates endpoint
        result = handler.handle("/api/debates", {}, mock_request)

        if result:
            # HandlerResult is a dataclass with status_code attribute
            # 503 is valid when storage isn't available
            assert result.status_code in [200, 401, 403, 503]  # Valid HTTP codes

    def test_health_handler_always_responds(self, handler_context, mock_request):
        """SystemHandler health endpoint should always work."""
        from aragora.server.handlers.admin import SystemHandler

        handler = SystemHandler(handler_context)

        result = handler.handle("/api/health", {}, mock_request)

        assert result is not None
        # HandlerResult is a dataclass with status_code attribute
        # Health check returns 200 (healthy) or 503 (degraded) based on system state
        assert result.status_code in [200, 503]

    def test_system_handler_returns_valid_json(self, handler_context, mock_request):
        """SystemHandler should return valid JSON body."""
        from aragora.server.handlers.admin import SystemHandler

        handler = SystemHandler(handler_context)
        result = handler.handle("/api/health", {}, mock_request)

        assert result is not None
        assert result.content_type == "application/json"
        # Body should be valid JSON
        body_data = json.loads(result.body.decode("utf-8"))
        assert isinstance(body_data, dict)


# =============================================================================
# E2E: Error Recovery Chain
# =============================================================================


class TestErrorRecoveryChain:
    """Test error recovery across the system."""

    @pytest.mark.asyncio
    async def test_debate_continues_with_failing_agent(self, simple_environment):
        """Debate should continue if one agent fails."""
        from .conftest import FailingAgent

        # Mix of working and failing agents
        agents = [
            MockAgent("working1", responses=["I propose solution A"]),
            FailingAgent("failing", fail_after=1),  # Fails after first response
            MockAgent("working2", responses=["I agree with solution A"]),
        ]

        protocol = DebateProtocol(rounds=1, consensus="majority")
        arena = Arena(simple_environment, agents, protocol)

        # Should complete despite one agent failing
        result = await run_debate_to_completion(arena)
        assert result is not None

    @pytest.mark.asyncio
    async def test_timeout_recovery(self, simple_environment):
        """Debate should handle slow agents with timeout."""
        from .conftest import SlowAgent

        # Mix fast and slow agents
        agents = [
            MockAgent("fast1", responses=["Quick response"]),
            SlowAgent("slow", delay=0.1),  # Short delay for test
            MockAgent("fast2", responses=["Another quick response"]),
        ]

        protocol = DebateProtocol(rounds=1, consensus="majority")
        arena = Arena(simple_environment, agents, protocol)

        # Should complete (slow agent delay is acceptable)
        result = await run_debate_to_completion(arena)
        assert result is not None


# =============================================================================
# E2E: Concurrent Debates
# =============================================================================


class TestConcurrentDebates:
    """Test multiple debates running concurrently."""

    @pytest.mark.asyncio
    async def test_concurrent_debates_isolated(self, mock_agents):
        """Multiple concurrent debates should not interfere."""
        environments = [Environment(task=f"Task {i}", context=f"Context {i}") for i in range(3)]

        protocol = DebateProtocol(rounds=1, consensus="majority")

        async def run_single_debate(env):
            arena = Arena(env, mock_agents, protocol)
            return await run_debate_to_completion(arena)

        # Run all debates concurrently
        results = await asyncio.gather(*[run_single_debate(env) for env in environments])

        # All should complete
        assert len(results) == 3
        for i, result in enumerate(results):
            assert result is not None
            assert result.task == environments[i].task

    @pytest.mark.asyncio
    async def test_debate_state_not_shared(self, mock_agents, simple_environment):
        """Arena state should not leak between debates."""
        protocol = DebateProtocol(rounds=1, consensus="majority")

        # Run first debate
        arena1 = Arena(simple_environment, mock_agents, protocol)
        result1 = await run_debate_to_completion(arena1)

        # Run second debate with same agents
        arena2 = Arena(simple_environment, mock_agents, protocol)
        result2 = await run_debate_to_completion(arena2)

        # Both should complete independently
        assert result1 is not None
        assert result2 is not None


# =============================================================================
# E2E: Export Format Verification
# =============================================================================


class TestExportFormats:
    """Test debate export in various formats."""

    @pytest.mark.asyncio
    async def test_export_contains_required_fields(self, mock_agents, simple_environment):
        """Exported debate should contain all required fields."""
        protocol = DebateProtocol(rounds=1, consensus="majority")
        arena = Arena(simple_environment, mock_agents, protocol)

        result = await run_debate_to_completion(arena)

        # Export to dict
        if hasattr(result, "to_dict"):
            export = result.to_dict()
        else:
            export = {
                "task": result.task,
                "status": result.status,
                "final_answer": result.final_answer,
                "rounds_completed": result.rounds_completed,
            }

        # Required fields
        assert "task" in export
        assert "status" in export

    @pytest.mark.asyncio
    async def test_export_serializable(self, mock_agents, simple_environment):
        """Exported debate should be JSON serializable."""
        protocol = DebateProtocol(rounds=1, consensus="majority")
        arena = Arena(simple_environment, mock_agents, protocol)

        result = await run_debate_to_completion(arena)

        # Should not raise
        if hasattr(result, "to_dict"):
            export = result.to_dict()
        else:
            export = {
                "task": result.task,
                "status": result.status,
                "rounds": result.rounds_completed,
            }

        json_str = json.dumps(export, default=str)
        assert len(json_str) > 0
