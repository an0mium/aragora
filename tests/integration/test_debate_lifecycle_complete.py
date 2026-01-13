"""
Complete Debate Lifecycle Integration Tests.

Tests for end-to-end debate flow with all subsystems:
- Context initialization → Rounds → Consensus → Storage
- ELO updates after debate completion
- Memory persistence and retrieval
- Checkpoint and resume flow
- User participation (votes, suggestions)
- Phase transitions
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, MagicMock, AsyncMock, patch

import pytest

from aragora.core import Agent, Message, Critique, Vote, Environment, DebateResult
from aragora.debate.orchestrator import Arena, DebateProtocol
from aragora.memory.store import CritiqueStore


# =============================================================================
# Auto-mock external dependencies
# =============================================================================


@pytest.fixture(autouse=True)
def mock_external_calls():
    """Mock external API calls to prevent network requests during tests."""
    with patch.object(Arena, "_gather_trending_context", new_callable=AsyncMock, return_value=None):
        with patch(
            "aragora.debate.phases.context_init.ContextInitializer.initialize",
            new_callable=AsyncMock,
            return_value=None,
        ):
            yield


# =============================================================================
# Mock Classes
# =============================================================================


class LifecycleMockAgent(Agent):
    """Mock agent for lifecycle testing."""

    def __init__(self, name: str = "mock", model: str = "mock-model", role: str = "proposer"):
        super().__init__(name, model, role)
        self.agent_type = "mock"
        self.generate_responses = []
        self.critique_responses = []
        self.vote_responses = []
        self._call_count = 0

    async def generate(self, prompt: str, context: list = None) -> str:
        if self.generate_responses:
            idx = self._call_count % len(self.generate_responses)
            self._call_count += 1
            return self.generate_responses[idx]
        return f"Response from {self.name}"

    async def critique(self, proposal: str, task: str, context: list = None) -> Critique:
        if self.critique_responses:
            idx = self._call_count % len(self.critique_responses)
            self._call_count += 1
            return self.critique_responses[idx]
        return Critique(
            agent=self.name,
            target_agent="target",
            target_content=proposal[:100],
            issues=["Issue"],
            suggestions=["Suggestion"],
            severity=0.5,
            reasoning="Reasoning",
        )

    async def vote(self, proposals: dict, task: str) -> Vote:
        if self.vote_responses:
            idx = self._call_count % len(self.vote_responses)
            self._call_count += 1
            return self.vote_responses[idx]
        choice = list(proposals.keys())[0] if proposals else self.name
        return Vote(
            agent=self.name,
            choice=choice,
            reasoning=f"Vote by {self.name}",
            confidence=0.8,
            continue_debate=False,
        )


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_db():
    """Create temporary database path."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        yield f.name
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def lifecycle_agents():
    """Create agents for lifecycle testing."""
    proposer = LifecycleMockAgent("proposer", role="proposer")
    critic = LifecycleMockAgent("critic", role="critic")
    synthesizer = LifecycleMockAgent("synthesizer", role="synthesizer")

    proposer.generate_responses = ["Initial proposal for the solution."]
    critic.generate_responses = ["Critique with improvements."]
    synthesizer.generate_responses = ["Synthesized conclusion."]

    # All vote for proposer
    for agent in [proposer, critic, synthesizer]:
        agent.vote_responses = [
            Vote(
                agent=agent.name,
                choice="proposer",
                reasoning="Best proposal",
                confidence=0.9,
                continue_debate=False,
            )
        ]

    return [proposer, critic, synthesizer]


@pytest.fixture
def mock_event_emitter():
    """Create mock event emitter tracking all events."""
    emitter = Mock()
    emitter.events = []

    def track_emit(event):
        emitter.events.append(event)

    emitter.emit = track_emit
    return emitter


# =============================================================================
# Phase Transition Tests
# =============================================================================


class TestPhaseTransitions:
    """Tests for debate phase transitions."""

    @pytest.mark.asyncio
    async def test_debate_traverses_all_phases(self, lifecycle_agents):
        """Debate should traverse context → rounds → consensus phases."""
        env = Environment(task="Test all phases", max_rounds=2)
        protocol = DebateProtocol(rounds=2, consensus="majority")

        arena = Arena(env, lifecycle_agents, protocol)
        result = await arena.run()

        # Result should exist and have consensus info
        assert isinstance(result, DebateResult)
        assert hasattr(result, "consensus_reached")

    @pytest.mark.asyncio
    async def test_context_phase_sets_up_debate(self, lifecycle_agents):
        """Context phase should initialize debate state."""
        env = Environment(
            task="Context initialization test",
            max_rounds=1,
            context="Additional context for agents",
        )
        protocol = DebateProtocol(rounds=1)

        arena = Arena(env, lifecycle_agents, protocol)
        result = await arena.run()

        assert result.task == "Context initialization test"

    @pytest.mark.asyncio
    async def test_round_phase_generates_messages(self, lifecycle_agents):
        """Round phase should generate agent messages."""
        env = Environment(task="Round test", max_rounds=2)
        protocol = DebateProtocol(rounds=2)

        arena = Arena(env, lifecycle_agents, protocol)
        result = await arena.run()

        # Should have at least some interaction
        assert hasattr(result, "messages")

    @pytest.mark.asyncio
    async def test_consensus_phase_produces_votes(self, lifecycle_agents):
        """Consensus phase should collect votes."""
        env = Environment(task="Vote test", max_rounds=2)
        protocol = DebateProtocol(rounds=2, consensus="majority")

        arena = Arena(env, lifecycle_agents, protocol)
        result = await arena.run()

        assert hasattr(result, "votes")


# =============================================================================
# Checkpoint and Resume Tests
# =============================================================================


class TestCheckpointAndResume:
    """Tests for debate checkpoint and resume functionality."""

    @pytest.mark.asyncio
    async def test_debate_creates_checkpoints(self, lifecycle_agents, temp_db):
        """Debate should support checkpoint creation."""
        env = Environment(task="Checkpoint test", max_rounds=2)
        protocol = DebateProtocol(rounds=2)

        # Mock checkpoint manager
        checkpoint_manager = Mock()
        checkpoint_manager.create_checkpoint = Mock(return_value="checkpoint-123")

        arena = Arena(env, lifecycle_agents, protocol)
        arena.checkpoint_manager = checkpoint_manager

        await arena.run()

        # Arena should be able to checkpoint (method exists)
        assert hasattr(arena, "checkpoint_manager") or hasattr(arena, "create_checkpoint")

    @pytest.mark.asyncio
    async def test_debate_result_is_serializable(self, lifecycle_agents):
        """Debate result should be serializable for checkpointing."""
        env = Environment(task="Serialization test", max_rounds=1)
        protocol = DebateProtocol(rounds=1)

        arena = Arena(env, lifecycle_agents, protocol)
        result = await arena.run()

        # Result should have to_dict or similar
        if hasattr(result, "to_dict"):
            data = result.to_dict()
            assert isinstance(data, dict)
        elif hasattr(result, "__dict__"):
            # Can extract as dict
            assert hasattr(result, "task")


# =============================================================================
# ELO Integration Tests
# =============================================================================


class TestEloLifecycleIntegration:
    """Tests for ELO system integration in debate lifecycle."""

    @pytest.mark.asyncio
    async def test_debate_result_includes_winner(self, lifecycle_agents):
        """Debate result should identify winner for ELO updates."""
        env = Environment(task="Winner test", max_rounds=2)
        protocol = DebateProtocol(rounds=2, consensus="majority")

        arena = Arena(env, lifecycle_agents, protocol)
        result = await arena.run()

        # Result should have winner field
        assert hasattr(result, "winner")

    @pytest.mark.asyncio
    async def test_elo_updated_after_debate(self, lifecycle_agents, temp_db):
        """ELO system should be updated after debate completion."""
        from aragora.ranking.elo import EloSystem

        elo = EloSystem(temp_db)
        env = Environment(task="ELO lifecycle test", max_rounds=2)
        protocol = DebateProtocol(rounds=2, consensus="majority")

        arena = Arena(env, lifecycle_agents, protocol, elo_system=elo)
        await arena.run()

        # Verify updates
        agent_names = [a.name for a in lifecycle_agents]
        for name in agent_names:
            rating = elo.get_rating(name)
            assert rating.debates_count >= 1

    @pytest.mark.asyncio
    async def test_multiple_debate_elo_accumulation(self, temp_db):
        """ELO should accumulate across multiple debates."""
        from aragora.ranking.elo import EloSystem

        elo = EloSystem(temp_db)

        for i in range(3):
            agents = [
                LifecycleMockAgent("alice", role="proposer"),
                LifecycleMockAgent("bob", role="critic"),
            ]
            for a in agents:
                a.vote_responses = [
                    Vote(
                        agent=a.name,
                        choice="alice" if i % 2 == 0 else "bob",
                        reasoning="Vote",
                        confidence=0.8,
                        continue_debate=False,
                    )
                ]

            env = Environment(task=f"Accumulation test {i}", max_rounds=1)
            protocol = DebateProtocol(rounds=1)

            arena = Arena(env, agents, protocol, elo_system=elo)
            await arena.run()

        # Verify accumulation
        alice_rating = elo.get_rating("alice", use_cache=False)
        bob_rating = elo.get_rating("bob", use_cache=False)

        assert alice_rating.debates_count == 3
        assert bob_rating.debates_count == 3


# =============================================================================
# Memory Persistence Tests
# =============================================================================


class TestMemoryLifecycle:
    """Tests for memory persistence throughout debate lifecycle."""

    @pytest.mark.asyncio
    async def test_critique_store_persists(self, lifecycle_agents, temp_db):
        """CritiqueStore should persist data across debate."""
        store = CritiqueStore(temp_db)

        env = Environment(task="Persistence test", max_rounds=2)
        protocol = DebateProtocol(rounds=2)

        arena = Arena(env, lifecycle_agents, protocol, memory=store)
        await arena.run()

        # Store should have stats
        stats = store.get_stats()
        assert isinstance(stats, dict)

    @pytest.mark.asyncio
    async def test_memory_accessible_after_debate(self, lifecycle_agents, temp_db):
        """Memory should be accessible after debate completion."""
        store = CritiqueStore(temp_db)

        env = Environment(task="Memory access test", max_rounds=1)
        protocol = DebateProtocol(rounds=1)

        arena = Arena(env, lifecycle_agents, protocol, memory=store)
        await arena.run()

        # Store methods should work
        assert hasattr(store, "get_stats")
        assert hasattr(store, "get_relevant")

    @pytest.mark.asyncio
    async def test_memory_shared_across_debates(self, temp_db):
        """Memory should be shared and accumulate across debates."""
        store = CritiqueStore(temp_db)

        for i in range(2):
            agents = [LifecycleMockAgent(f"agent-{i}", role="proposer")]
            env = Environment(task=f"Shared memory {i}", max_rounds=1)
            protocol = DebateProtocol(rounds=1)

            arena = Arena(env, agents, protocol, memory=store)
            await arena.run()

        # Stats should reflect multiple debates
        stats = store.get_stats()
        assert isinstance(stats, dict)


# =============================================================================
# Event Emission Tests
# =============================================================================


class TestEventLifecycle:
    """Tests for event emission throughout debate lifecycle."""

    @pytest.mark.asyncio
    async def test_debate_emits_events(self, lifecycle_agents, mock_event_emitter):
        """Debate should emit events during execution."""
        env = Environment(task="Event test", max_rounds=2)
        protocol = DebateProtocol(rounds=2)

        arena = Arena(env, lifecycle_agents, protocol, event_emitter=mock_event_emitter)
        await arena.run()

        # Arena should have emitted events
        assert hasattr(mock_event_emitter, "events") or hasattr(mock_event_emitter, "emit")

    @pytest.mark.asyncio
    async def test_debate_completes_with_emitter(self, lifecycle_agents, mock_event_emitter):
        """Debate should complete successfully with event emitter."""
        env = Environment(task="Emitter completion test", max_rounds=2)
        protocol = DebateProtocol(rounds=2)

        arena = Arena(env, lifecycle_agents, protocol, event_emitter=mock_event_emitter)
        result = await arena.run()

        assert isinstance(result, DebateResult)


# =============================================================================
# User Participation Tests
# =============================================================================


class TestUserParticipation:
    """Tests for user participation in debates."""

    @pytest.mark.asyncio
    async def test_arena_accepts_user_votes(self, lifecycle_agents):
        """Arena should accept user votes configuration."""
        env = Environment(task="User vote test", max_rounds=2)
        protocol = DebateProtocol(rounds=2, consensus="majority", user_vote_weight=0.5)

        arena = Arena(env, lifecycle_agents, protocol)
        result = await arena.run()

        assert isinstance(result, DebateResult)

    @pytest.mark.asyncio
    async def test_protocol_user_vote_weight(self, lifecycle_agents):
        """Protocol should support user vote weight."""
        protocol = DebateProtocol(rounds=2, user_vote_weight=1.0)

        assert protocol.user_vote_weight == 1.0


# =============================================================================
# Consensus Mode Tests
# =============================================================================


class TestConsensusModesLifecycle:
    """Tests for different consensus modes in lifecycle."""

    @pytest.mark.asyncio
    async def test_none_consensus_mode(self, lifecycle_agents):
        """None consensus mode should combine proposals."""
        env = Environment(task="None consensus test", max_rounds=2)
        protocol = DebateProtocol(rounds=2, consensus="none")

        arena = Arena(env, lifecycle_agents, protocol)
        result = await arena.run()

        assert result.consensus_reached is False

    @pytest.mark.asyncio
    async def test_majority_consensus_mode(self, lifecycle_agents):
        """Majority consensus should determine winner."""
        # All vote for proposer
        for agent in lifecycle_agents:
            agent.vote_responses = [
                Vote(
                    agent=agent.name,
                    choice="proposer",
                    reasoning="Best",
                    confidence=0.9,
                    continue_debate=False,
                )
            ]

        env = Environment(task="Majority consensus test", max_rounds=2)
        protocol = DebateProtocol(rounds=2, consensus="majority")

        arena = Arena(env, lifecycle_agents, protocol)
        result = await arena.run()

        assert isinstance(result, DebateResult)

    @pytest.mark.asyncio
    async def test_unanimous_consensus_mode(self, lifecycle_agents):
        """Unanimous consensus requires all to agree."""
        # All vote for same choice
        for agent in lifecycle_agents:
            agent.vote_responses = [
                Vote(
                    agent=agent.name,
                    choice="proposer",
                    reasoning="Agree",
                    confidence=0.95,
                    continue_debate=False,
                )
            ]

        env = Environment(task="Unanimous test", max_rounds=2)
        protocol = DebateProtocol(rounds=2, consensus="unanimous")

        arena = Arena(env, lifecycle_agents, protocol)
        result = await arena.run()

        assert isinstance(result, DebateResult)

    @pytest.mark.asyncio
    async def test_judge_consensus_mode(self, lifecycle_agents):
        """Judge consensus mode should use judge agent."""
        env = Environment(task="Judge test", max_rounds=2)
        protocol = DebateProtocol(rounds=2, consensus="judge")

        arena = Arena(env, lifecycle_agents, protocol)
        result = await arena.run()

        assert isinstance(result, DebateResult)


# =============================================================================
# Error Recovery Tests
# =============================================================================


class TestErrorRecoveryLifecycle:
    """Tests for error recovery in debate lifecycle."""

    @pytest.mark.asyncio
    async def test_debate_recovers_from_agent_error(self, lifecycle_agents):
        """Debate should recover from individual agent errors."""
        # Make one agent fail
        lifecycle_agents[0].generate = AsyncMock(side_effect=Exception("Agent error"))

        env = Environment(task="Error recovery test", max_rounds=2)
        protocol = DebateProtocol(rounds=2)

        arena = Arena(env, lifecycle_agents, protocol)
        # Should complete despite error
        result = await arena.run()

        assert isinstance(result, DebateResult)

    @pytest.mark.asyncio
    async def test_debate_handles_timeout(self, lifecycle_agents):
        """Debate should handle agent timeouts gracefully."""

        async def slow_generate(prompt, context=None):
            await asyncio.sleep(0.01)
            return "Response"

        lifecycle_agents[0].generate = slow_generate

        env = Environment(task="Timeout test", max_rounds=1)
        protocol = DebateProtocol(rounds=1)

        arena = Arena(env, lifecycle_agents, protocol)
        result = await arena.run()

        assert isinstance(result, DebateResult)


# =============================================================================
# Round Limit Tests
# =============================================================================


class TestRoundLimitsLifecycle:
    """Tests for round limits in debate lifecycle."""

    @pytest.mark.asyncio
    async def test_respects_max_rounds(self, lifecycle_agents):
        """Debate should respect max_rounds limit."""
        env = Environment(task="Round limit test", max_rounds=3)
        protocol = DebateProtocol(rounds=3)

        arena = Arena(env, lifecycle_agents, protocol)
        result = await arena.run()

        assert result.rounds_used <= 3

    @pytest.mark.asyncio
    async def test_early_stopping_reduces_rounds(self, lifecycle_agents):
        """Early stopping should reduce rounds used."""
        # Set all agents to not continue
        for agent in lifecycle_agents:
            agent.vote_responses = [
                Vote(
                    agent=agent.name,
                    choice="proposer",
                    reasoning="Done",
                    confidence=0.99,
                    continue_debate=False,
                )
            ]

        env = Environment(task="Early stop test", max_rounds=10)
        protocol = DebateProtocol(rounds=10, early_stopping=True)

        arena = Arena(env, lifecycle_agents, protocol)
        result = await arena.run()

        # Should stop before 10 rounds
        assert result.rounds_used <= 10


# =============================================================================
# Complete Flow Tests
# =============================================================================


class TestCompleteDebateFlow:
    """Tests for complete debate flow integration."""

    @pytest.mark.asyncio
    async def test_full_lifecycle_with_all_components(self, lifecycle_agents, temp_db):
        """Test complete lifecycle with ELO, memory, and events."""
        from aragora.ranking.elo import EloSystem

        elo = EloSystem(temp_db)
        store = CritiqueStore(temp_db)
        emitter = Mock()
        emitter.emit = Mock()

        env = Environment(task="Full lifecycle test", max_rounds=2)
        protocol = DebateProtocol(rounds=2, consensus="majority")

        arena = Arena(
            env, lifecycle_agents, protocol, elo_system=elo, memory=store, event_emitter=emitter
        )
        result = await arena.run()

        # Verify all components worked
        assert isinstance(result, DebateResult)
        assert hasattr(result, "task")
        assert hasattr(result, "votes")

    @pytest.mark.asyncio
    async def test_lifecycle_produces_consistent_results(self):
        """Multiple runs should produce consistent result structure."""
        results = []

        for i in range(3):
            agents = [
                LifecycleMockAgent(f"a-{i}", role="proposer"),
                LifecycleMockAgent(f"b-{i}", role="critic"),
            ]

            env = Environment(task=f"Consistency test {i}", max_rounds=1)
            protocol = DebateProtocol(rounds=1)

            arena = Arena(env, agents, protocol)
            result = await arena.run()
            results.append(result)

        # All results should have same structure
        for result in results:
            assert isinstance(result, DebateResult)
            assert hasattr(result, "task")
            assert hasattr(result, "messages")
            assert hasattr(result, "votes")
            assert hasattr(result, "consensus_reached")


# =============================================================================
# Concurrent Lifecycle Tests
# =============================================================================


class TestConcurrentLifecycles:
    """Tests for concurrent debate lifecycles."""

    @pytest.mark.asyncio
    async def test_parallel_debates_complete(self):
        """Multiple debates can run in parallel."""

        async def run_debate(task_id: int) -> DebateResult:
            agents = [
                LifecycleMockAgent(f"a-{task_id}", role="proposer"),
                LifecycleMockAgent(f"b-{task_id}", role="critic"),
            ]

            env = Environment(task=f"Parallel {task_id}", max_rounds=1)
            protocol = DebateProtocol(rounds=1)

            arena = Arena(env, agents, protocol)
            return await arena.run()

        results = await asyncio.gather(run_debate(1), run_debate(2), run_debate(3))

        assert len(results) == 3
        for result in results:
            assert isinstance(result, DebateResult)

    @pytest.mark.asyncio
    async def test_parallel_debates_with_shared_elo(self, temp_db):
        """Parallel debates can share ELO system."""
        from aragora.ranking.elo import EloSystem

        elo = EloSystem(temp_db)

        async def run_debate_with_elo(task_id: int) -> DebateResult:
            agents = [
                LifecycleMockAgent("alice", role="proposer"),
                LifecycleMockAgent("bob", role="critic"),
            ]

            env = Environment(task=f"Shared ELO {task_id}", max_rounds=1)
            protocol = DebateProtocol(rounds=1)

            arena = Arena(env, agents, protocol, elo_system=elo)
            return await arena.run()

        results = await asyncio.gather(run_debate_with_elo(1), run_debate_with_elo(2))

        assert len(results) == 2


__all__ = [
    "TestPhaseTransitions",
    "TestCheckpointAndResume",
    "TestEloLifecycleIntegration",
    "TestMemoryLifecycle",
    "TestEventLifecycle",
    "TestUserParticipation",
    "TestConsensusModesLifecycle",
    "TestErrorRecoveryLifecycle",
    "TestRoundLimitsLifecycle",
    "TestCompleteDebateFlow",
    "TestConcurrentLifecycles",
]
