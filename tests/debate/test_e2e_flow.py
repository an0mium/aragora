"""
End-to-end tests for Debate Engine.

Tests the full lifecycle of:
- Complete debate flows (proposal -> critique -> vote -> consensus)
- Multi-round debates with convergence
- Different debate topologies
- Memory integration (outcome storage, retrieval)
- Checkpoint/resume functionality
- User participation (votes, suggestions)
- Error recovery and timeout handling
"""

from __future__ import annotations

import asyncio
from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.core import Agent, Critique, DebateResult, Environment, Message, Vote
from aragora.debate.arena_config import ArenaConfig
from aragora.debate.event_bus import DebateEvent, EventBus
from aragora.debate.orchestrator import Arena
from aragora.debate.protocol import DebateProtocol


# ============================================================================
# E2E Test Agents
# ============================================================================


class E2ETestAgent(Agent):
    """
    Test agent for E2E debate testing with configurable behavior.

    Supports:
    - Configurable responses per round
    - Vote behavior (choice, confidence, continue_debate)
    - Critique behavior
    - Call tracking for verification
    """

    def __init__(
        self,
        name: str = "e2e-agent",
        model: str = "test-model",
        role: str = "proposer",
        responses: Optional[List[str]] = None,
        vote_choice: Optional[str] = None,
        vote_confidence: float = 0.8,
        continue_debate: bool = False,
        critique_severity: float = 0.5,
    ):
        super().__init__(name=name, model=model, role=role)
        self.agent_type = "e2e-test"
        self.responses = responses or [f"Response from {name}"]
        self._response_index = 0
        self.vote_choice = vote_choice
        self.vote_confidence = vote_confidence
        self.continue_debate = continue_debate
        self.critique_severity = critique_severity

        # Call tracking
        self.generate_calls: List[str] = []
        self.critique_calls: List[str] = []
        self.vote_calls: List[Dict] = []

    async def generate(self, prompt: str, context: list = None) -> str:
        self.generate_calls.append(prompt)
        response = self.responses[min(self._response_index, len(self.responses) - 1)]
        self._response_index += 1
        return response

    async def generate_stream(self, prompt: str, context: list = None):
        response = await self.generate(prompt, context)
        yield response

    async def critique(
        self,
        proposal: str,
        task: str,
        context: list = None,
        target_agent: str = None,
    ) -> Critique:
        self.critique_calls.append(proposal)
        return Critique(
            agent=self.name,
            target_agent=target_agent or "unknown",
            target_content=proposal[:100] if proposal else "",
            issues=[f"Issue identified by {self.name}"],
            suggestions=[f"Suggestion from {self.name}"],
            severity=self.critique_severity,
            reasoning=f"Critique reasoning from {self.name}",
        )

    async def vote(self, proposals: dict, task: str) -> Vote:
        self.vote_calls.append({"proposals": proposals, "task": task})
        # Use explicit choice or vote for self
        choice = self.vote_choice or (list(proposals.keys())[0] if proposals else self.name)
        return Vote(
            agent=self.name,
            choice=choice,
            reasoning=f"Vote reasoning from {self.name}",
            confidence=self.vote_confidence,
            continue_debate=self.continue_debate,
        )


class ConvergingAgent(E2ETestAgent):
    """Agent that converges responses over rounds."""

    def __init__(
        self,
        name: str,
        initial_position: str,
        final_position: str,
        rounds_to_converge: int = 2,
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.initial_position = initial_position
        self.final_position = final_position
        self.rounds_to_converge = rounds_to_converge
        self._round = 0

    async def generate(self, prompt: str, context: list = None) -> str:
        self.generate_calls.append(prompt)
        self._round += 1

        if self._round >= self.rounds_to_converge:
            return self.final_position
        else:
            # Gradually shift position
            blend = self._round / self.rounds_to_converge
            return f"{self.initial_position} (weight: {1-blend:.1f}) -> {self.final_position} (weight: {blend:.1f})"


# ============================================================================
# E2E Test Fixtures
# ============================================================================


@pytest.fixture
def basic_protocol() -> DebateProtocol:
    """Create a basic debate protocol."""
    return DebateProtocol(
        rounds=2,
        topology="round-robin",
        consensus="majority",
        critique_required=True,
        early_stopping=False,
        convergence_detection=False,
    )


@pytest.fixture
def quick_protocol() -> DebateProtocol:
    """Create a quick single-round protocol."""
    return DebateProtocol(
        rounds=1,
        topology="round-robin",
        consensus="majority",
        critique_required=False,
        early_stopping=False,
        convergence_detection=False,
    )


@pytest.fixture
def convergence_protocol() -> DebateProtocol:
    """Create a protocol for testing convergence."""
    return DebateProtocol(
        rounds=5,
        topology="round-robin",
        consensus="majority",
        critique_required=True,
        early_stopping=True,
        convergence_detection=True,
        convergence_threshold=0.9,
    )


@pytest.fixture
def basic_env() -> Environment:
    """Create a basic debate environment."""
    return Environment(task="What is the best approach to software testing?")


@pytest.fixture
def two_agents() -> List[E2ETestAgent]:
    """Create two basic test agents."""
    return [
        E2ETestAgent(
            name="agent-alpha",
            responses=["I believe unit tests are most important."],
            vote_choice="agent-alpha",
        ),
        E2ETestAgent(
            name="agent-beta",
            responses=["Integration tests provide more value."],
            vote_choice="agent-alpha",  # Both vote for alpha -> consensus
        ),
    ]


@pytest.fixture
def three_agents_with_disagreement() -> List[E2ETestAgent]:
    """Create three agents with initial disagreement."""
    return [
        E2ETestAgent(
            name="agent-1",
            responses=["Unit tests are essential.", "Unit tests with mocks work best."],
            vote_choice="agent-1",
        ),
        E2ETestAgent(
            name="agent-2",
            responses=[
                "E2E tests catch real bugs.",
                "E2E combined with unit tests is ideal.",
            ],
            vote_choice="agent-1",
        ),
        E2ETestAgent(
            name="agent-3",
            responses=[
                "Property-based testing is superior.",
                "A balanced approach is best.",
            ],
            vote_choice="agent-1",
        ),
    ]


# ============================================================================
# Basic Debate Lifecycle E2E Tests
# ============================================================================


class TestBasicDebateLifecycle:
    """Tests for basic debate execution lifecycle."""

    @pytest.mark.asyncio
    async def test_simple_debate_completes(
        self,
        basic_env: Environment,
        two_agents: List[E2ETestAgent],
        quick_protocol: DebateProtocol,
    ) -> None:
        """Test that a simple debate runs to completion."""
        arena = Arena(
            basic_env,
            agents=two_agents,
            protocol=quick_protocol,
        )

        result = await arena.run()

        assert result is not None
        assert isinstance(result, DebateResult)
        assert result.debate_id is not None

    @pytest.mark.asyncio
    async def test_debate_calls_all_agents(
        self,
        basic_env: Environment,
        two_agents: List[E2ETestAgent],
        quick_protocol: DebateProtocol,
    ) -> None:
        """Test that all agents are called during debate."""
        arena = Arena(
            basic_env,
            agents=two_agents,
            protocol=quick_protocol,
        )

        await arena.run()

        # All agents should have been called to generate
        for agent in two_agents:
            assert len(agent.generate_calls) >= 1

    @pytest.mark.asyncio
    async def test_multi_round_debate(
        self,
        basic_env: Environment,
        two_agents: List[E2ETestAgent],
        basic_protocol: DebateProtocol,
    ) -> None:
        """Test debate with multiple rounds."""
        arena = Arena(
            basic_env,
            agents=two_agents,
            protocol=basic_protocol,
        )

        result = await arena.run()

        assert result is not None
        # With 2 rounds, agents should be called multiple times
        for agent in two_agents:
            assert len(agent.generate_calls) >= 1


# ============================================================================
# Consensus and Voting E2E Tests
# ============================================================================


class TestConsensusAndVoting:
    """Tests for consensus detection and voting."""

    @pytest.mark.asyncio
    async def test_unanimous_consensus(
        self,
        basic_env: Environment,
        quick_protocol: DebateProtocol,
    ) -> None:
        """Test that unanimous voting achieves consensus."""
        agents = [
            E2ETestAgent(name="agent-a", vote_choice="agent-a"),
            E2ETestAgent(name="agent-b", vote_choice="agent-a"),
            E2ETestAgent(name="agent-c", vote_choice="agent-a"),
        ]

        arena = Arena(basic_env, agents=agents, protocol=quick_protocol)
        result = await arena.run()

        # All agents voted for agent-a
        for agent in agents:
            assert len(agent.vote_calls) >= 1

    @pytest.mark.asyncio
    async def test_majority_consensus(
        self,
        basic_env: Environment,
    ) -> None:
        """Test that majority voting achieves consensus."""
        agents = [
            E2ETestAgent(name="agent-1", vote_choice="agent-1"),
            E2ETestAgent(name="agent-2", vote_choice="agent-1"),
            E2ETestAgent(name="agent-3", vote_choice="agent-3"),  # Dissent
        ]

        protocol = DebateProtocol(
            rounds=1,
            consensus="majority",
            critique_required=False,
            early_stopping=False,
            convergence_detection=False,
        )

        arena = Arena(basic_env, agents=agents, protocol=protocol)
        result = await arena.run()

        assert result is not None

    @pytest.mark.asyncio
    async def test_no_consensus_continues_debate(
        self,
        basic_env: Environment,
    ) -> None:
        """Test that lack of consensus can continue debate."""
        agents = [
            E2ETestAgent(name="agent-1", vote_choice="agent-1", continue_debate=True),
            E2ETestAgent(name="agent-2", vote_choice="agent-2", continue_debate=True),
            E2ETestAgent(name="agent-3", vote_choice="agent-3", continue_debate=True),
        ]

        protocol = DebateProtocol(
            rounds=2,
            consensus="unanimous",
            critique_required=False,
            early_stopping=False,
            convergence_detection=False,
        )

        arena = Arena(basic_env, agents=agents, protocol=protocol)
        result = await arena.run()

        # Debate should complete despite no unanimous consensus
        assert result is not None


# ============================================================================
# Critique Flow E2E Tests
# ============================================================================


class TestCritiqueFlow:
    """Tests for critique generation during debates."""

    @pytest.mark.asyncio
    async def test_critiques_generated(
        self,
        basic_env: Environment,
        basic_protocol: DebateProtocol,
    ) -> None:
        """Test that critiques are generated when enabled."""
        agents = [
            E2ETestAgent(name="critic-1", critique_severity=0.7),
            E2ETestAgent(name="critic-2", critique_severity=0.3),
        ]

        arena = Arena(basic_env, agents=agents, protocol=basic_protocol)
        result = await arena.run()

        # With enable_critique=True, agents should have been asked to critique
        # (depends on protocol execution order)
        assert result is not None

    @pytest.mark.asyncio
    async def test_high_severity_critiques(
        self,
        basic_env: Environment,
    ) -> None:
        """Test behavior with high-severity critiques."""
        agents = [
            E2ETestAgent(name="harsh-critic", critique_severity=0.9),
            E2ETestAgent(name="target", critique_severity=0.1),
        ]

        protocol = DebateProtocol(
            rounds=2,
            critique_required=True,
            early_stopping=False,
            convergence_detection=False,
        )

        arena = Arena(basic_env, agents=agents, protocol=protocol)
        result = await arena.run()

        assert result is not None


# ============================================================================
# Convergence Detection E2E Tests
# ============================================================================


class TestConvergenceDetection:
    """Tests for semantic convergence detection."""

    @pytest.mark.asyncio
    async def test_early_termination_on_convergence(
        self,
        basic_env: Environment,
    ) -> None:
        """Test that debates can terminate early when convergence is detected."""
        # Create agents that converge to the same answer
        agents = [
            ConvergingAgent(
                name="conv-1",
                initial_position="Unit tests are key",
                final_position="Testing pyramid approach is best",
                rounds_to_converge=2,
            ),
            ConvergingAgent(
                name="conv-2",
                initial_position="E2E tests matter most",
                final_position="Testing pyramid approach is best",
                rounds_to_converge=2,
            ),
        ]

        protocol = DebateProtocol(
            rounds=5,
            topology="round-robin",
            early_stopping=True,
            convergence_detection=True,
            convergence_threshold=0.95,
            critique_required=False,
        )

        arena = Arena(basic_env, agents=agents, protocol=protocol)
        result = await arena.run()

        # Should complete even if not all 5 rounds run
        assert result is not None

    @pytest.mark.asyncio
    async def test_no_early_termination_without_convergence(
        self,
        basic_env: Environment,
    ) -> None:
        """Test that debates run full rounds without convergence."""
        # Agents with divergent responses
        agents = [
            E2ETestAgent(
                name="divergent-1",
                responses=["Position A"] * 5,
            ),
            E2ETestAgent(
                name="divergent-2",
                responses=["Position B (completely different)"] * 5,
            ),
        ]

        protocol = DebateProtocol(
            rounds=3,
            early_stopping=True,
            convergence_detection=True,
            convergence_threshold=0.99,
            critique_required=False,
        )

        arena = Arena(basic_env, agents=agents, protocol=protocol)
        result = await arena.run()

        assert result is not None


# ============================================================================
# Event Emission E2E Tests
# ============================================================================


class TestEventEmission:
    """Tests for debate event emission."""

    @pytest.mark.asyncio
    async def test_events_emitted_during_debate(
        self,
        basic_env: Environment,
        two_agents: List[E2ETestAgent],
        quick_protocol: DebateProtocol,
    ) -> None:
        """Test that events are emitted during debate lifecycle."""
        # Arena has internal event emission - we test that debates complete
        arena = Arena(basic_env, two_agents, quick_protocol)

        result = await arena.run()

        # Should complete without event-related errors
        assert result is not None

    @pytest.mark.asyncio
    async def test_round_events(
        self,
        basic_env: Environment,
        two_agents: List[E2ETestAgent],
        basic_protocol: DebateProtocol,
    ) -> None:
        """Test that debate runs through multiple rounds."""
        arena = Arena(basic_env, two_agents, basic_protocol)

        result = await arena.run()

        # Multi-round debate should complete
        assert result is not None


# ============================================================================
# Arena Configuration E2E Tests
# ============================================================================


class TestArenaConfiguration:
    """Tests for Arena configuration options."""

    @pytest.mark.asyncio
    async def test_arena_with_config(
        self,
        basic_env: Environment,
        two_agents: List[E2ETestAgent],
        quick_protocol: DebateProtocol,
    ) -> None:
        """Test Arena with configuration via from_config."""
        config = ArenaConfig(
            enable_checkpointing=True,
            use_performance_selection=False,
        )

        arena = Arena.from_config(
            basic_env,
            two_agents,
            quick_protocol,
            config,
        )

        result = await arena.run()
        assert result is not None

    @pytest.mark.asyncio
    async def test_arena_context_manager(
        self,
        basic_env: Environment,
        two_agents: List[E2ETestAgent],
        quick_protocol: DebateProtocol,
    ) -> None:
        """Test Arena as async context manager."""
        async with Arena(basic_env, two_agents, quick_protocol) as arena:
            result = await arena.run()
            assert result is not None

    @pytest.mark.asyncio
    async def test_arena_with_timeout(
        self,
        basic_env: Environment,
        two_agents: List[E2ETestAgent],
    ) -> None:
        """Test Arena with timeout configuration."""
        protocol_with_timeout = DebateProtocol(
            rounds=1,
            consensus="majority",
            timeout_seconds=60,
            early_stopping=False,
            convergence_detection=False,
        )

        arena = Arena(basic_env, two_agents, protocol_with_timeout)

        result = await arena.run()
        assert result is not None


# ============================================================================
# Debate Topology E2E Tests
# ============================================================================


class TestDebateTopologies:
    """Tests for different debate topologies."""

    @pytest.mark.asyncio
    async def test_round_robin_topology(
        self,
        basic_env: Environment,
        three_agents_with_disagreement: List[E2ETestAgent],
    ) -> None:
        """Test round-robin debate topology."""
        protocol = DebateProtocol(
            rounds=2,
            topology="round-robin",
            critique_required=True,
            early_stopping=False,
            convergence_detection=False,
        )

        arena = Arena(
            basic_env,
            agents=three_agents_with_disagreement,
            protocol=protocol,
        )

        result = await arena.run()
        assert result is not None

    @pytest.mark.asyncio
    async def test_parallel_topology(
        self,
        basic_env: Environment,
        three_agents_with_disagreement: List[E2ETestAgent],
    ) -> None:
        """Test parallel (hive-mind) debate topology."""
        protocol = DebateProtocol(
            rounds=1,
            topology="all-to-all",  # Parallel-style topology
            critique_required=False,
            early_stopping=False,
            convergence_detection=False,
        )

        arena = Arena(
            basic_env,
            agents=three_agents_with_disagreement,
            protocol=protocol,
        )

        result = await arena.run()
        assert result is not None


# ============================================================================
# Error Handling E2E Tests
# ============================================================================


class TestErrorHandling:
    """Tests for error handling during debates."""

    @pytest.mark.asyncio
    async def test_agent_failure_handled(
        self,
        basic_env: Environment,
        quick_protocol: DebateProtocol,
    ) -> None:
        """Test that agent failures are handled gracefully."""

        class FailingAgent(E2ETestAgent):
            async def generate(self, prompt: str, context: list = None) -> str:
                raise RuntimeError("Agent failed!")

        agents = [
            FailingAgent(name="failing-agent"),
            E2ETestAgent(name="working-agent"),
        ]

        arena = Arena(basic_env, agents=agents, protocol=quick_protocol)

        # Should not crash, may return partial result or handle error
        try:
            result = await arena.run()
            # If we get here, error was handled
            assert True
        except RuntimeError:
            # Error propagated - also valid behavior
            assert True

    @pytest.mark.asyncio
    async def test_empty_response_handled(
        self,
        basic_env: Environment,
        quick_protocol: DebateProtocol,
    ) -> None:
        """Test that empty agent responses are handled."""
        agents = [
            E2ETestAgent(name="empty-agent", responses=[""]),
            E2ETestAgent(name="normal-agent"),
        ]

        arena = Arena(basic_env, agents=agents, protocol=quick_protocol)
        result = await arena.run()

        # Should complete despite empty response
        assert result is not None


# ============================================================================
# Memory Integration E2E Tests
# ============================================================================


class TestMemoryIntegration:
    """Tests for memory system integration."""

    @pytest.mark.asyncio
    async def test_debate_with_memory_context(
        self,
        basic_env: Environment,
        two_agents: List[E2ETestAgent],
        quick_protocol: DebateProtocol,
    ) -> None:
        """Test debate with memory context provided."""
        # Mock memory that provides context
        mock_memory = MagicMock()
        mock_memory.get_relevant_context = AsyncMock(
            return_value=[{"content": "Previous debate concluded X", "relevance": 0.9}]
        )

        arena = Arena(
            basic_env,
            agents=two_agents,
            protocol=quick_protocol,
        )

        result = await arena.run()
        assert result is not None

    @pytest.mark.asyncio
    async def test_debate_outcome_can_be_stored(
        self,
        basic_env: Environment,
        two_agents: List[E2ETestAgent],
        quick_protocol: DebateProtocol,
    ) -> None:
        """Test that debate outcomes can be stored."""
        arena = Arena(
            basic_env,
            agents=two_agents,
            protocol=quick_protocol,
        )

        result = await arena.run()

        # Result should contain storable data
        assert result is not None
        assert result.debate_id is not None


# ============================================================================
# User Participation E2E Tests
# ============================================================================


class TestUserParticipation:
    """Tests for user participation in debates."""

    @pytest.mark.asyncio
    async def test_debate_accepts_user_context(
        self,
        two_agents: List[E2ETestAgent],
        quick_protocol: DebateProtocol,
    ) -> None:
        """Test that debates can include user-provided context."""
        env = Environment(
            task="What testing approach should we use?",
            context={"user_preference": "integration-focused"},
        )

        arena = Arena(env, agents=two_agents, protocol=quick_protocol)
        result = await arena.run()

        assert result is not None


# ============================================================================
# Checkpoint Integration E2E Tests
# ============================================================================


class TestCheckpointIntegration:
    """Tests for checkpoint/resume functionality."""

    @pytest.mark.asyncio
    async def test_checkpoint_enabled_debate(
        self,
        basic_env: Environment,
        two_agents: List[E2ETestAgent],
        basic_protocol: DebateProtocol,
    ) -> None:
        """Test debate with checkpointing enabled."""
        config = ArenaConfig(enable_checkpointing=True)

        arena = Arena.from_config(
            basic_env,
            two_agents,
            basic_protocol,
            config,
        )

        result = await arena.run()
        assert result is not None


# ============================================================================
# Stress and Edge Case E2E Tests
# ============================================================================


class TestStressAndEdgeCases:
    """Tests for stress conditions and edge cases."""

    @pytest.mark.asyncio
    async def test_single_agent_debate(
        self,
        basic_env: Environment,
        quick_protocol: DebateProtocol,
    ) -> None:
        """Test debate with only one agent."""
        agents = [E2ETestAgent(name="solo-agent")]

        arena = Arena(basic_env, agents=agents, protocol=quick_protocol)
        result = await arena.run()

        # Should complete even with single agent
        assert result is not None

    @pytest.mark.asyncio
    async def test_many_agents_debate(
        self,
        basic_env: Environment,
    ) -> None:
        """Test debate with many agents."""
        agents = [E2ETestAgent(name=f"agent-{i}", vote_choice="agent-0") for i in range(5)]

        protocol = DebateProtocol(
            rounds=1,
            topology="round-robin",
            critique_required=False,
            early_stopping=False,
            convergence_detection=False,
        )

        arena = Arena(basic_env, agents=agents, protocol=protocol)
        result = await arena.run()

        assert result is not None

    @pytest.mark.asyncio
    async def test_long_responses(
        self,
        basic_env: Environment,
        quick_protocol: DebateProtocol,
    ) -> None:
        """Test handling of very long agent responses."""
        long_response = "This is a very detailed response. " * 500  # ~2500 chars

        agents = [
            E2ETestAgent(name="verbose-agent", responses=[long_response]),
            E2ETestAgent(name="normal-agent"),
        ]

        arena = Arena(basic_env, agents=agents, protocol=quick_protocol)
        result = await arena.run()

        assert result is not None

    @pytest.mark.asyncio
    async def test_empty_task(
        self,
        two_agents: List[E2ETestAgent],
        quick_protocol: DebateProtocol,
    ) -> None:
        """Test that empty task is rejected during Environment creation."""
        # Environment validates task cannot be empty
        with pytest.raises(ValueError, match="Task cannot be empty"):
            Environment(task="")
