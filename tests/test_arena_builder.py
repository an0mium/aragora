"""
Tests for the Arena builder pattern.

Tests the ArenaBuilder class that provides a fluent interface
for constructing Arena instances with many optional components.
"""

from unittest.mock import MagicMock, patch

import pytest

from aragora.core import Agent, Environment
from aragora.debate.arena_builder import ArenaBuilder, create_arena
from aragora.debate.protocol import CircuitBreaker, DebateProtocol


class MockAgent(Agent):
    """Minimal mock agent for testing."""

    def __init__(self, name: str = "mock-agent"):
        super().__init__(name, model="mock-model", role="proposer")

    async def generate(self, prompt: str, context=None) -> str:
        return "Mock response"

    async def critique(self, proposal: str, task: str, context=None):
        from aragora.core import Critique

        return Critique(
            author=self.name,
            target="test",
            content=proposal,
            issues=["issue"],
            suggestions=["suggestion"],
            severity=0.5,
        )


class TestArenaBuilderBasics:
    """Test basic ArenaBuilder functionality."""

    def test_builder_creates_arena_with_minimal_config(self) -> None:
        """Builder creates Arena with just required parameters."""
        env = Environment(task="Test task")
        agents = [MockAgent("agent-1"), MockAgent("agent-2")]

        arena = ArenaBuilder(env, agents).build()

        assert arena.env == env
        assert arena.agents == agents
        assert arena.protocol is not None  # Default protocol created

    def test_builder_with_protocol(self) -> None:
        """Builder correctly sets protocol."""
        env = Environment(task="Test task")
        agents = [MockAgent()]
        protocol = DebateProtocol(rounds=5, consensus="majority")

        arena = ArenaBuilder(env, agents).with_protocol(protocol).build()

        assert arena.protocol == protocol
        assert arena.protocol.rounds == 5

    def test_builder_with_rounds_creates_protocol(self) -> None:
        """with_rounds creates protocol if not set."""
        env = Environment(task="Test task")
        agents = [MockAgent()]

        arena = ArenaBuilder(env, agents).with_rounds(3).build()

        assert arena.protocol.rounds == 3


class TestArenaBuilderMemory:
    """Test memory-related builder methods."""

    def test_builder_with_memory(self) -> None:
        """Builder correctly sets critique store."""
        env = Environment(task="Test task")
        agents = [MockAgent()]
        memory = MagicMock()

        arena = ArenaBuilder(env, agents).with_memory(memory).build()

        assert arena.memory == memory

    def test_builder_with_debate_embeddings(self) -> None:
        """Builder correctly sets debate embeddings."""
        env = Environment(task="Test task")
        agents = [MockAgent()]
        embeddings = MagicMock()

        arena = ArenaBuilder(env, agents).with_debate_embeddings(embeddings).build()

        assert arena.debate_embeddings == embeddings

    def test_builder_with_continuum_memory(self) -> None:
        """Builder correctly sets continuum memory."""
        env = Environment(task="Test task")
        agents = [MockAgent()]
        continuum = MagicMock()

        arena = ArenaBuilder(env, agents).with_continuum_memory(continuum).build()

        assert arena.continuum_memory == continuum

    def test_builder_with_full_memory(self) -> None:
        """with_full_memory sets all memory components."""
        env = Environment(task="Test task")
        agents = [MockAgent()]
        memory = MagicMock()
        embeddings = MagicMock()
        continuum = MagicMock()

        arena = (
            ArenaBuilder(env, agents)
            .with_full_memory(
                memory=memory,
                debate_embeddings=embeddings,
                continuum_memory=continuum,
            )
            .build()
        )

        assert arena.memory == memory
        assert arena.debate_embeddings == embeddings
        assert arena.continuum_memory == continuum


class TestArenaBuilderEvents:
    """Test event-related builder methods."""

    def test_builder_with_event_hooks(self) -> None:
        """Builder correctly sets event hooks."""
        env = Environment(task="Test task")
        agents = [MockAgent()]
        hooks = {"on_proposal": lambda x: x}

        arena = ArenaBuilder(env, agents).with_event_hooks(hooks).build()

        # Check that provided hooks are present (arena may add default hooks)
        assert "on_proposal" in arena.hooks
        assert arena.hooks["on_proposal"] == hooks["on_proposal"]

    def test_builder_with_spectator(self) -> None:
        """Builder correctly sets spectator stream."""
        env = Environment(task="Test task")
        agents = [MockAgent()]
        spectator = MagicMock()

        arena = ArenaBuilder(env, agents).with_spectator(spectator).build()

        assert arena.spectator == spectator

    def test_builder_with_recorder(self) -> None:
        """Builder correctly sets replay recorder."""
        env = Environment(task="Test task")
        agents = [MockAgent()]
        recorder = MagicMock()

        arena = ArenaBuilder(env, agents).with_recorder(recorder).build()

        assert arena.recorder == recorder


class TestArenaBuilderTracking:
    """Test tracking-related builder methods."""

    def test_builder_with_elo_system(self) -> None:
        """Builder correctly sets ELO system."""
        env = Environment(task="Test task")
        agents = [MockAgent()]
        elo = MagicMock()

        arena = ArenaBuilder(env, agents).with_elo_system(elo).build()

        assert arena.elo_system == elo

    def test_builder_with_persona_manager(self) -> None:
        """Builder correctly sets persona manager."""
        env = Environment(task="Test task")
        agents = [MockAgent()]
        manager = MagicMock()

        arena = ArenaBuilder(env, agents).with_persona_manager(manager).build()

        assert arena.persona_manager == manager

    def test_builder_with_agent_weights(self) -> None:
        """Builder correctly sets agent weights."""
        env = Environment(task="Test task")
        agents = [MockAgent("agent-1")]
        weights = {"agent-1": 1.5}

        arena = ArenaBuilder(env, agents).with_agent_weights(weights).build()

        assert arena.agent_weights == weights

    def test_builder_with_full_tracking(self) -> None:
        """with_full_tracking sets all tracking components."""
        env = Environment(task="Test task")
        agents = [MockAgent()]

        # Configure ELO mock with get_rating method
        elo = MagicMock()
        elo.get_rating.return_value = 1000.0

        persona = MagicMock()

        # Configure calibration mock to return proper numeric brier_score
        calibration = MagicMock()
        calibration_result = MagicMock()
        calibration_result.brier_score = 0.25
        calibration.get_calibration.return_value = calibration_result

        arena = (
            ArenaBuilder(env, agents)
            .with_full_tracking(
                elo_system=elo,
                persona_manager=persona,
                calibration_tracker=calibration,
            )
            .build()
        )

        # Use 'is' for identity comparison with mocks
        assert arena.elo_system is elo
        assert arena.persona_manager is persona
        assert arena.calibration_tracker is calibration


class TestArenaBuilderPosition:
    """Test position-related builder methods."""

    def test_builder_with_position_tracker(self) -> None:
        """Builder correctly sets position tracker."""
        env = Environment(task="Test task")
        agents = [MockAgent()]
        tracker = MagicMock()

        arena = ArenaBuilder(env, agents).with_position_tracker(tracker).build()

        assert arena.position_tracker == tracker

    def test_builder_with_position_ledger(self) -> None:
        """Builder correctly sets position ledger."""
        env = Environment(task="Test task")
        agents = [MockAgent()]
        ledger = MagicMock()

        arena = ArenaBuilder(env, agents).with_position_ledger(ledger).build()

        assert arena.position_ledger == ledger

    def test_builder_with_flip_detector(self) -> None:
        """Builder correctly sets flip detector."""
        env = Environment(task="Test task")
        agents = [MockAgent()]
        detector = MagicMock()

        arena = ArenaBuilder(env, agents).with_flip_detector(detector).build()

        assert arena.flip_detector == detector


class TestArenaBuilderLoop:
    """Test loop-related builder methods."""

    def test_builder_with_loop_id(self) -> None:
        """Builder correctly sets loop ID."""
        env = Environment(task="Test task")
        agents = [MockAgent()]

        arena = ArenaBuilder(env, agents).with_loop_id("test-loop-123").build()

        assert arena.loop_id == "test-loop-123"

    def test_builder_with_strict_loop_scoping(self) -> None:
        """Builder correctly sets strict loop scoping."""
        env = Environment(task="Test task")
        agents = [MockAgent()]

        arena = ArenaBuilder(env, agents).with_strict_loop_scoping(True).build()

        assert arena.strict_loop_scoping is True

    def test_builder_with_circuit_breaker(self) -> None:
        """Builder correctly sets circuit breaker."""
        env = Environment(task="Test task")
        agents = [MockAgent()]
        breaker = CircuitBreaker(failure_threshold=5)

        arena = ArenaBuilder(env, agents).with_circuit_breaker(breaker).build()

        assert arena.circuit_breaker == breaker

    def test_builder_with_initial_messages(self) -> None:
        """Builder correctly sets initial messages."""
        env = Environment(task="Test task")
        agents = [MockAgent()]
        messages = [{"role": "user", "content": "Hello"}]

        arena = ArenaBuilder(env, agents).with_initial_messages(messages).build()

        assert arena.initial_messages == messages


class TestArenaBuilderFluent:
    """Test fluent interface chaining."""

    def test_builder_chaining(self) -> None:
        """Builder methods can be chained."""
        env = Environment(task="Test task")
        agents = [MockAgent()]
        memory = MagicMock()
        elo = MagicMock()
        protocol = DebateProtocol(rounds=3)

        arena = (
            ArenaBuilder(env, agents)
            .with_protocol(protocol)
            .with_memory(memory)
            .with_elo_system(elo)
            .with_loop_id("test-loop")
            .build()
        )

        assert arena.protocol == protocol
        assert arena.memory == memory
        assert arena.elo_system == elo
        assert arena.loop_id == "test-loop"

    def test_builder_methods_return_self(self) -> None:
        """All builder methods return self for chaining."""
        env = Environment(task="Test task")
        agents = [MockAgent()]
        builder = ArenaBuilder(env, agents)

        # Each method should return the builder
        assert builder.with_protocol(DebateProtocol()) is builder
        assert builder.with_rounds(3) is builder
        assert builder.with_memory(MagicMock()) is builder
        assert builder.with_elo_system(MagicMock()) is builder
        assert builder.with_loop_id("test") is builder
        assert builder.with_strict_loop_scoping(True) is builder


class TestCreateArenaConvenience:
    """Test the create_arena convenience function."""

    def test_create_arena_minimal(self) -> None:
        """create_arena works with minimal parameters."""
        env = Environment(task="Test task")
        agents = [MockAgent()]

        arena = create_arena(env, agents)

        assert arena.env == env
        assert arena.agents == agents

    def test_create_arena_with_optional_params(self) -> None:
        """create_arena accepts common optional parameters."""
        env = Environment(task="Test task")
        agents = [MockAgent()]
        protocol = DebateProtocol(rounds=5)
        memory = MagicMock()
        elo = MagicMock()

        arena = create_arena(
            env,
            agents,
            protocol=protocol,
            memory=memory,
            elo_system=elo,
        )

        assert arena.protocol == protocol
        assert arena.memory == memory
        assert arena.elo_system == elo


class TestArenaBuilderContext:
    """Test context-related builder methods."""

    def test_builder_with_dissent_retriever(self) -> None:
        """Builder correctly sets dissent retriever."""
        env = Environment(task="Test task")
        agents = [MockAgent()]
        retriever = MagicMock()

        arena = ArenaBuilder(env, agents).with_dissent_retriever(retriever).build()

        assert arena.dissent_retriever == retriever

    def test_builder_with_evidence_collector(self) -> None:
        """Builder correctly sets evidence collector."""
        env = Environment(task="Test task")
        agents = [MockAgent()]
        collector = MagicMock()

        arena = ArenaBuilder(env, agents).with_evidence_collector(collector).build()

        assert arena.evidence_collector == collector

    def test_builder_with_trending_topic(self) -> None:
        """Builder correctly sets trending topic."""
        env = Environment(task="Test task")
        agents = [MockAgent()]
        topic = MagicMock()

        arena = ArenaBuilder(env, agents).with_trending_topic(topic).build()

        assert arena.trending_topic == topic
