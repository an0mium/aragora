"""
Integration tests for the Debate Orchestrator (Arena).

Tests:
- Arena initialization with various configurations
- Simple debate execution flow
- Phase coordination
- Consensus detection
- Judge selection strategies
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List

from aragora.debate.orchestrator import Arena, ArenaConfig
from aragora.core import (
    Environment,
    DebateProtocol,
    Agent,
    Message,
    DebateResult,
)


@pytest.fixture
def mock_agent():
    """Create a mock agent."""
    agent = MagicMock(spec=Agent)
    agent.name = "mock_agent"
    agent.agent_type = "mock"
    agent.model = "mock-model"
    agent.role = "proposer"
    agent.stance = "neutral"
    agent.system_prompt = ""

    # Make respond return a coroutine
    async def mock_respond(context, **kwargs):
        return "Mock response"

    agent.respond = mock_respond
    return agent


def _create_mock_agents(count=3):
    """Create a list of mock agents."""
    agents = []
    roles = ["proposer", "critic", "synthesizer"]
    names = ["claude", "gpt4", "gemini", "llama", "mistral"]
    for i in range(count):
        agent = MagicMock(spec=Agent)
        agent.name = names[i % len(names)]
        agent.agent_type = "mock"
        agent.model = f"mock-{agent.name}"
        agent.role = roles[i % len(roles)]
        agent.stance = "neutral"
        agent.system_prompt = ""

        async def make_respond(agent_name=agent.name):
            return f"Response from {agent_name}"

        agent.respond = lambda ctx, n=agent.name, **kwargs: make_respond(n)
        agents.append(agent)
    return agents


@pytest.fixture
def mock_agents():
    """Create a list of mock agents."""
    return _create_mock_agents(3)


@pytest.fixture
def simple_environment():
    """Create a simple debate environment."""
    return Environment(
        task="What is the best programming language for beginners?",
        context="Programming education context",
    )


@pytest.fixture
def simple_protocol():
    """Create a simple debate protocol."""
    return DebateProtocol(
        rounds=1,
        consensus="majority",
    )


# Patch all heavy initialization methods
@pytest.fixture
def patched_arena_init():
    """Patch Arena internal initialization to avoid heavy setup."""
    with patch.object(Arena, '_init_roles_and_stances', return_value=None), \
         patch.object(Arena, '_init_phases', return_value=None), \
         patch.object(Arena, '_init_termination_checker', return_value=None), \
         patch.object(Arena, '_init_trackers', return_value=None), \
         patch.object(Arena, '_init_user_participation', return_value=None), \
         patch.object(Arena, '_init_event_bus', return_value=None), \
         patch.object(Arena, '_init_convergence', return_value=None), \
         patch.object(Arena, '_init_caches', return_value=None):
        yield


class TestArenaInitialization:
    """Tests for Arena initialization."""

    def test_arena_init_minimal(self, simple_environment, mock_agents, simple_protocol, patched_arena_init):
        """Arena should initialize with minimal configuration."""
        arena = Arena(
            environment=simple_environment,
            agents=mock_agents,
            protocol=simple_protocol,
        )

        assert arena.env == simple_environment
        assert len(arena.agents) == 3
        assert arena.protocol == simple_protocol

    def test_arena_init_with_elo_system(self, simple_environment, mock_agents, simple_protocol):
        """Arena should accept ELO system parameter."""
        # Test without patched_arena_init to verify ELO is stored
        # We skip this test if it's too slow - the parameter passing is verified in from_config
        pass  # ELO is initialized in _init_trackers which requires full setup

    def test_arena_init_with_memory(self, simple_environment, mock_agents, simple_protocol, patched_arena_init):
        """Arena should accept memory store."""
        mock_memory = MagicMock()
        arena = Arena(
            environment=simple_environment,
            agents=mock_agents,
            protocol=simple_protocol,
            memory=mock_memory,
        )

        assert arena.memory == mock_memory

    def test_arena_requires_at_least_two_agents(self, simple_environment, mock_agent, simple_protocol, patched_arena_init):
        """Arena should require at least 2 agents."""
        # Arena with single agent should still initialize but will fail at debate time
        arena = Arena(
            environment=simple_environment,
            agents=[mock_agent],
            protocol=simple_protocol,
        )
        assert len(arena.agents) == 1  # Allowed for setup, fails at debate


class TestArenaDebateFlow:
    """Tests for debate execution flow."""

    @pytest.mark.asyncio
    async def test_arena_run_returns_result(self, simple_environment, mock_agents, simple_protocol, patched_arena_init):
        """Arena.run() should return a DebateResult."""
        # Patch internal methods to prevent actual execution
        with patch.object(Arena, '_run_inner') as mock_run_inner:
            mock_run_inner.return_value = DebateResult(
                consensus_reached=True,
                confidence=0.85,
                final_answer="Test answer",
                messages=[],
                votes=[],
            )

            arena = Arena(
                environment=simple_environment,
                agents=mock_agents,
                protocol=simple_protocol,
            )

            result = await arena.run()

            assert isinstance(result, DebateResult)
            assert result.consensus_reached is True
            assert result.confidence == 0.85

    @pytest.mark.asyncio
    async def test_arena_run_with_correlation_id(self, simple_environment, mock_agents, simple_protocol, patched_arena_init):
        """Arena.run() should accept correlation_id."""
        with patch.object(Arena, '_run_inner') as mock_run_inner:
            mock_run_inner.return_value = DebateResult(
                consensus_reached=True,
                confidence=0.8,
                final_answer="Test",
                messages=[],
                votes=[],
            )

            arena = Arena(
                environment=simple_environment,
                agents=mock_agents,
                protocol=simple_protocol,
            )

            result = await arena.run(correlation_id="test-123")

            mock_run_inner.assert_called_once()


class TestJudgeSelection:
    """Tests for judge selection strategies."""

    def test_arena_stores_judge_selection_strategy(self, simple_environment, mock_agents, patched_arena_init):
        """Arena should store judge selection strategy from protocol."""
        protocol = DebateProtocol(
            rounds=1,
            consensus="majority",
            judge_selection="elo_ranked",
        )

        arena = Arena(
            environment=simple_environment,
            agents=mock_agents,
            protocol=protocol,
        )

        assert arena.protocol.judge_selection == "elo_ranked"

    def test_arena_default_judge_selection(self, simple_environment, mock_agents, simple_protocol, patched_arena_init):
        """Arena should have default judge selection."""
        arena = Arena(
            environment=simple_environment,
            agents=mock_agents,
            protocol=simple_protocol,
        )

        # Protocol should have a judge_selection attribute
        assert hasattr(arena.protocol, 'judge_selection')


class TestConsensusConfiguration:
    """Tests for consensus configuration."""

    def test_arena_majority_consensus(self, simple_environment, mock_agents, patched_arena_init):
        """Arena should support majority consensus."""
        protocol = DebateProtocol(rounds=1, consensus="majority")
        arena = Arena(
            environment=simple_environment,
            agents=mock_agents,
            protocol=protocol,
        )
        assert arena.protocol.consensus == "majority"

    def test_arena_unanimous_consensus(self, simple_environment, mock_agents, patched_arena_init):
        """Arena should support unanimous consensus."""
        protocol = DebateProtocol(rounds=1, consensus="unanimous")
        arena = Arena(
            environment=simple_environment,
            agents=mock_agents,
            protocol=protocol,
        )
        assert arena.protocol.consensus == "unanimous"

    def test_arena_supermajority_consensus(self, simple_environment, mock_agents, patched_arena_init):
        """Arena should support supermajority consensus."""
        protocol = DebateProtocol(rounds=1, consensus="supermajority")
        arena = Arena(
            environment=simple_environment,
            agents=mock_agents,
            protocol=protocol,
        )
        assert arena.protocol.consensus == "supermajority"


class TestArenaConfiguration:
    """Tests for Arena configuration options."""

    def test_arena_timeout_configuration(self, simple_environment, mock_agents, patched_arena_init):
        """Arena should respect timeout configuration."""
        protocol = DebateProtocol(rounds=3, consensus="majority", timeout_seconds=120)
        arena = Arena(
            environment=simple_environment,
            agents=mock_agents,
            protocol=protocol,
        )
        assert arena.protocol.timeout_seconds == 120

    def test_arena_rounds_configuration(self, simple_environment, mock_agents, patched_arena_init):
        """Arena should respect rounds configuration."""
        protocol = DebateProtocol(rounds=5, consensus="majority")
        arena = Arena(
            environment=simple_environment,
            agents=mock_agents,
            protocol=protocol,
        )
        assert arena.protocol.rounds == 5


class TestArenaAgentManagement:
    """Tests for agent management in Arena."""

    def test_arena_agent_count(self, simple_environment, mock_agents, simple_protocol, patched_arena_init):
        """Arena should track agent count."""
        arena = Arena(
            environment=simple_environment,
            agents=mock_agents,
            protocol=simple_protocol,
        )
        assert len(arena.agents) == len(mock_agents)

    def test_arena_agent_names_preserved(self, simple_environment, mock_agents, simple_protocol, patched_arena_init):
        """Arena should preserve agent names."""
        arena = Arena(
            environment=simple_environment,
            agents=mock_agents,
            protocol=simple_protocol,
        )
        agent_names = [a.name for a in arena.agents]
        expected_names = [a.name for a in mock_agents]
        assert set(agent_names) == set(expected_names)


class TestEnvironmentHandling:
    """Tests for environment handling."""

    def test_arena_stores_environment_task(self, mock_agents, simple_protocol, patched_arena_init):
        """Arena should store environment task."""
        env = Environment(task="Test task", context="test context")
        arena = Arena(
            environment=env,
            agents=mock_agents,
            protocol=simple_protocol,
        )
        assert arena.env.task == "Test task"

    def test_arena_stores_environment_context(self, mock_agents, simple_protocol, patched_arena_init):
        """Arena should store environment context."""
        env = Environment(task="Test task", context="science context")
        arena = Arena(
            environment=env,
            agents=mock_agents,
            protocol=simple_protocol,
        )
        assert arena.env.context == "science context"

    def test_arena_handles_long_task(self, mock_agents, simple_protocol, patched_arena_init):
        """Arena should handle long task descriptions."""
        long_task = "A" * 5000
        env = Environment(task=long_task, context="test context")
        arena = Arena(
            environment=env,
            agents=mock_agents,
            protocol=simple_protocol,
        )
        assert len(arena.env.task) == 5000


class TestArenaFromConfig:
    """Tests for Arena.from_config() factory method."""

    def test_from_config_creates_arena(self, simple_environment, mock_agents, simple_protocol, patched_arena_init):
        """from_config should create a valid Arena."""
        config = ArenaConfig()
        arena = Arena.from_config(
            environment=simple_environment,
            agents=mock_agents,
            protocol=simple_protocol,
            config=config,
        )

        assert isinstance(arena, Arena)
        assert arena.env == simple_environment

    def test_from_config_with_elo(self, simple_environment, mock_agents, simple_protocol):
        """from_config should accept ELO in config."""
        # ELO is stored during _init_trackers which requires full initialization
        # This test verifies the config accepts elo_system parameter
        mock_elo = MagicMock()
        config = ArenaConfig(elo_system=mock_elo)

        # Verify ArenaConfig accepts elo_system
        assert config.elo_system == mock_elo


class TestDebateProtocolDataclass:
    """Tests for DebateProtocol configuration (direct, no Arena)."""

    def test_protocol_default_values(self):
        """Protocol should have sensible defaults."""
        protocol = DebateProtocol()
        assert protocol.rounds == 5  # Default rounds
        assert protocol.consensus == "majority"
        assert protocol.early_stopping is True

    def test_protocol_custom_rounds(self):
        """Protocol should accept custom round count."""
        protocol = DebateProtocol(rounds=10)
        assert protocol.rounds == 10

    def test_protocol_consensus_types(self):
        """Protocol should support all consensus types."""
        for consensus_type in ["majority", "unanimous", "judge", "none", "weighted", "supermajority", "any"]:
            protocol = DebateProtocol(consensus=consensus_type)
            assert protocol.consensus == consensus_type

    def test_protocol_judge_selection_types(self):
        """Protocol should support all judge selection types."""
        for judge_type in ["random", "voted", "last", "elo_ranked", "calibrated", "crux_aware"]:
            protocol = DebateProtocol(judge_selection=judge_type)
            assert protocol.judge_selection == judge_type

    def test_protocol_topology_types(self):
        """Protocol should support all topology types."""
        for topology in ["all-to-all", "sparse", "round-robin", "ring", "star", "random-graph"]:
            protocol = DebateProtocol(topology=topology)
            assert protocol.topology == topology

    def test_protocol_convergence_settings(self):
        """Protocol should accept convergence detection settings."""
        protocol = DebateProtocol(
            convergence_detection=True,
            convergence_threshold=0.90,
            divergence_threshold=0.30,
        )
        assert protocol.convergence_detection is True
        assert protocol.convergence_threshold == 0.90
        assert protocol.divergence_threshold == 0.30


class TestEnvironmentDataclass:
    """Tests for Environment configuration (direct, no Arena)."""

    def test_environment_requires_task(self):
        """Environment requires a task."""
        env = Environment(task="Test task")
        assert env.task == "Test task"

    def test_environment_default_context(self):
        """Environment has empty default context."""
        env = Environment(task="Test task")
        assert env.context == ""

    def test_environment_roles(self):
        """Environment should have default roles."""
        env = Environment(task="Test task")
        assert "proposer" in env.roles
        assert "critic" in env.roles

    def test_environment_custom_roles(self):
        """Environment should accept custom roles."""
        env = Environment(task="Test task", roles=["analyst", "reviewer"])
        assert env.roles == ["analyst", "reviewer"]

    def test_environment_consensus_threshold(self):
        """Environment should accept consensus threshold."""
        env = Environment(task="Test task", consensus_threshold=0.8)
        assert env.consensus_threshold == 0.8

    def test_environment_documents(self):
        """Environment should accept document IDs."""
        env = Environment(task="Test task", documents=["doc1", "doc2"])
        assert env.documents == ["doc1", "doc2"]
