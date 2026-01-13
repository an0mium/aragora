"""
Tests for Arena Builder pattern.

Tests cover:
- ArenaBuilder initialization
- Fluent method chaining
- Protocol configuration
- Memory and persistence options
- Event handling configuration
- Agent tracking options
- Position and truth grounding
- Loop configuration
- Composite configuration methods
- Build method and Arena creation
- create_arena convenience function
"""

from __future__ import annotations

import pytest
from dataclasses import dataclass
from unittest.mock import Mock, MagicMock, patch

from aragora.core import Agent, Environment
from aragora.debate.arena_builder import ArenaBuilder, create_arena
from aragora.debate.protocol import DebateProtocol, CircuitBreaker
from aragora.spectate.stream import SpectatorStream


# ============================================================================
# Test Fixtures
# ============================================================================


@dataclass
class MockAgent:
    """Mock agent for testing."""

    name: str = "mock-agent"

    async def generate(self, prompt, context=None):
        return f"Response from {self.name}"


@pytest.fixture
def environment():
    """Create a test environment."""
    return Environment(task="Test debate task", context="Test context")


@pytest.fixture
def agents():
    """Create test agents."""
    return [
        MockAgent(name="agent1"),
        MockAgent(name="agent2"),
        MockAgent(name="agent3"),
    ]


@pytest.fixture
def protocol():
    """Create a test protocol."""
    return DebateProtocol(rounds=3, consensus="majority")


@pytest.fixture
def builder(environment, agents):
    """Create a builder for testing."""
    return ArenaBuilder(environment, agents)


# ============================================================================
# Initialization Tests
# ============================================================================


class TestArenaBuilderInit:
    """Tests for ArenaBuilder initialization."""

    def test_init_stores_environment(self, environment, agents):
        """Test environment is stored."""
        builder = ArenaBuilder(environment, agents)
        assert builder._environment is environment

    def test_init_stores_agents(self, environment, agents):
        """Test agents are stored."""
        builder = ArenaBuilder(environment, agents)
        assert builder._agents is agents

    def test_init_defaults_protocol_none(self, builder):
        """Test protocol defaults to None."""
        assert builder._protocol is None

    def test_init_defaults_memory_none(self, builder):
        """Test memory defaults to None."""
        assert builder._memory is None

    def test_init_defaults_spectator_none(self, builder):
        """Test spectator defaults to None."""
        assert builder._spectator is None

    def test_init_defaults_elo_system_none(self, builder):
        """Test ELO system defaults to None."""
        assert builder._elo_system is None

    def test_init_defaults_event_hooks_empty(self, builder):
        """Test event hooks default to empty dict."""
        assert builder._event_hooks == {}

    def test_init_defaults_agent_weights_empty(self, builder):
        """Test agent weights default to empty dict."""
        assert builder._agent_weights == {}

    def test_init_defaults_loop_id_empty(self, builder):
        """Test loop ID defaults to empty string."""
        assert builder._loop_id == ""

    def test_init_defaults_strict_loop_scoping_false(self, builder):
        """Test strict loop scoping defaults to False."""
        assert builder._strict_loop_scoping is False


# ============================================================================
# Protocol Configuration Tests
# ============================================================================


class TestProtocolConfiguration:
    """Tests for protocol configuration methods."""

    def test_with_protocol_stores_protocol(self, builder, protocol):
        """Test with_protocol stores the protocol."""
        builder.with_protocol(protocol)
        assert builder._protocol is protocol

    def test_with_protocol_returns_self(self, builder, protocol):
        """Test with_protocol returns self for chaining."""
        result = builder.with_protocol(protocol)
        assert result is builder

    def test_with_rounds_creates_protocol(self, builder):
        """Test with_rounds creates protocol if none exists."""
        builder.with_rounds(5)
        assert builder._protocol is not None
        assert builder._protocol.rounds == 5

    def test_with_rounds_updates_existing_protocol(self, builder, protocol):
        """Test with_rounds updates existing protocol."""
        builder.with_protocol(protocol)
        builder.with_rounds(10)
        assert builder._protocol.rounds == 10

    def test_with_rounds_returns_self(self, builder):
        """Test with_rounds returns self for chaining."""
        result = builder.with_rounds(3)
        assert result is builder

    def test_with_template_creates_protocol(self, builder):
        """Test with_template creates protocol from template."""
        mock_template = Mock()
        mock_template.name = "test_template"

        with patch("aragora.templates.template_to_protocol") as mock_convert:
            mock_convert.return_value = DebateProtocol(rounds=5)
            builder.with_template(mock_template)

            mock_convert.assert_called_once()
            assert builder._protocol is not None
            assert builder._template is mock_template


# ============================================================================
# Memory Configuration Tests
# ============================================================================


class TestMemoryConfiguration:
    """Tests for memory configuration methods."""

    def test_with_memory_stores_memory(self, builder):
        """Test with_memory stores the critique store."""
        mock_memory = Mock()
        builder.with_memory(mock_memory)
        assert builder._memory is mock_memory

    def test_with_memory_returns_self(self, builder):
        """Test with_memory returns self for chaining."""
        result = builder.with_memory(Mock())
        assert result is builder

    def test_with_debate_embeddings_stores_embeddings(self, builder):
        """Test with_debate_embeddings stores embeddings."""
        mock_embeddings = Mock()
        builder.with_debate_embeddings(mock_embeddings)
        assert builder._debate_embeddings is mock_embeddings

    def test_with_insight_store_stores_store(self, builder):
        """Test with_insight_store stores the store."""
        mock_store = Mock()
        builder.with_insight_store(mock_store)
        assert builder._insight_store is mock_store

    def test_with_continuum_memory_stores_memory(self, builder):
        """Test with_continuum_memory stores memory."""
        mock_memory = Mock()
        builder.with_continuum_memory(mock_memory)
        assert builder._continuum_memory is mock_memory


# ============================================================================
# Event Handling Tests
# ============================================================================


class TestEventHandling:
    """Tests for event handling configuration."""

    def test_with_event_hooks_stores_hooks(self, builder):
        """Test with_event_hooks stores hooks."""
        hooks = {"on_message": lambda x: x}
        builder.with_event_hooks(hooks)
        assert builder._event_hooks == hooks

    def test_with_event_emitter_stores_emitter(self, builder):
        """Test with_event_emitter stores emitter."""
        mock_emitter = Mock()
        builder.with_event_emitter(mock_emitter)
        assert builder._event_emitter is mock_emitter

    def test_with_spectator_stores_spectator(self, builder):
        """Test with_spectator stores spectator."""
        mock_spectator = Mock(spec=SpectatorStream)
        builder.with_spectator(mock_spectator)
        assert builder._spectator is mock_spectator

    def test_with_recorder_stores_recorder(self, builder):
        """Test with_recorder stores recorder."""
        mock_recorder = Mock()
        builder.with_recorder(mock_recorder)
        assert builder._recorder is mock_recorder


# ============================================================================
# Agent Tracking Tests
# ============================================================================


class TestAgentTracking:
    """Tests for agent tracking configuration."""

    def test_with_agent_weights_stores_weights(self, builder):
        """Test with_agent_weights stores weights."""
        weights = {"agent1": 0.9, "agent2": 0.8}
        builder.with_agent_weights(weights)
        assert builder._agent_weights == weights

    def test_with_elo_system_stores_system(self, builder):
        """Test with_elo_system stores ELO system."""
        mock_elo = Mock()
        builder.with_elo_system(mock_elo)
        assert builder._elo_system is mock_elo

    def test_with_persona_manager_stores_manager(self, builder):
        """Test with_persona_manager stores manager."""
        mock_manager = Mock()
        builder.with_persona_manager(mock_manager)
        assert builder._persona_manager is mock_manager

    def test_with_calibration_tracker_stores_tracker(self, builder):
        """Test with_calibration_tracker stores tracker."""
        mock_tracker = Mock()
        builder.with_calibration_tracker(mock_tracker)
        assert builder._calibration_tracker is mock_tracker

    def test_with_relationship_tracker_stores_tracker(self, builder):
        """Test with_relationship_tracker stores tracker."""
        mock_tracker = Mock()
        builder.with_relationship_tracker(mock_tracker)
        assert builder._relationship_tracker is mock_tracker


# ============================================================================
# Position and Truth Grounding Tests
# ============================================================================


class TestPositionGrounding:
    """Tests for position and truth grounding configuration."""

    def test_with_position_tracker_stores_tracker(self, builder):
        """Test with_position_tracker stores tracker."""
        mock_tracker = Mock()
        builder.with_position_tracker(mock_tracker)
        assert builder._position_tracker is mock_tracker

    def test_with_position_ledger_stores_ledger(self, builder):
        """Test with_position_ledger stores ledger."""
        mock_ledger = Mock()
        builder.with_position_ledger(mock_ledger)
        assert builder._position_ledger is mock_ledger

    def test_with_flip_detector_stores_detector(self, builder):
        """Test with_flip_detector stores detector."""
        mock_detector = Mock()
        builder.with_flip_detector(mock_detector)
        assert builder._flip_detector is mock_detector

    def test_with_moment_detector_stores_detector(self, builder):
        """Test with_moment_detector stores detector."""
        mock_detector = Mock()
        builder.with_moment_detector(mock_detector)
        assert builder._moment_detector is mock_detector


# ============================================================================
# Historical Context Tests
# ============================================================================


class TestHistoricalContext:
    """Tests for historical context configuration."""

    def test_with_dissent_retriever_stores_retriever(self, builder):
        """Test with_dissent_retriever stores retriever."""
        mock_retriever = Mock()
        builder.with_dissent_retriever(mock_retriever)
        assert builder._dissent_retriever is mock_retriever

    def test_with_evidence_collector_stores_collector(self, builder):
        """Test with_evidence_collector stores collector."""
        mock_collector = Mock()
        builder.with_evidence_collector(mock_collector)
        assert builder._evidence_collector is mock_collector

    def test_with_trending_topic_stores_topic(self, builder):
        """Test with_trending_topic stores topic."""
        mock_topic = Mock()
        builder.with_trending_topic(mock_topic)
        assert builder._trending_topic is mock_topic


# ============================================================================
# Loop Configuration Tests
# ============================================================================


class TestLoopConfiguration:
    """Tests for loop configuration methods."""

    def test_with_loop_id_stores_id(self, builder):
        """Test with_loop_id stores loop ID."""
        builder.with_loop_id("loop-123")
        assert builder._loop_id == "loop-123"

    def test_with_strict_loop_scoping_enables(self, builder):
        """Test with_strict_loop_scoping enables strict mode."""
        builder.with_strict_loop_scoping(True)
        assert builder._strict_loop_scoping is True

    def test_with_strict_loop_scoping_default_true(self, builder):
        """Test with_strict_loop_scoping defaults to True."""
        builder.with_strict_loop_scoping()
        assert builder._strict_loop_scoping is True

    def test_with_circuit_breaker_stores_breaker(self, builder):
        """Test with_circuit_breaker stores breaker."""
        mock_breaker = Mock(spec=CircuitBreaker)
        builder.with_circuit_breaker(mock_breaker)
        assert builder._circuit_breaker is mock_breaker

    def test_with_initial_messages_stores_messages(self, builder):
        """Test with_initial_messages stores messages."""
        messages = [{"role": "user", "content": "Hello"}]
        builder.with_initial_messages(messages)
        assert builder._initial_messages == messages


# ============================================================================
# Composite Configuration Tests
# ============================================================================


class TestCompositeConfiguration:
    """Tests for composite configuration methods."""

    def test_with_full_tracking_sets_elo(self, builder):
        """Test with_full_tracking sets ELO system."""
        mock_elo = Mock()
        builder.with_full_tracking(elo_system=mock_elo)
        assert builder._elo_system is mock_elo

    def test_with_full_tracking_sets_all_optional(self, builder):
        """Test with_full_tracking sets all optional trackers."""
        mock_elo = Mock()
        mock_persona = Mock()
        mock_calibration = Mock()
        mock_relationship = Mock()

        builder.with_full_tracking(
            elo_system=mock_elo,
            persona_manager=mock_persona,
            calibration_tracker=mock_calibration,
            relationship_tracker=mock_relationship,
        )

        assert builder._elo_system is mock_elo
        assert builder._persona_manager is mock_persona
        assert builder._calibration_tracker is mock_calibration
        assert builder._relationship_tracker is mock_relationship

    def test_with_full_tracking_returns_self(self, builder):
        """Test with_full_tracking returns self for chaining."""
        result = builder.with_full_tracking(elo_system=Mock())
        assert result is builder

    def test_with_full_memory_sets_memory(self, builder):
        """Test with_full_memory sets critique store."""
        mock_memory = Mock()
        builder.with_full_memory(memory=mock_memory)
        assert builder._memory is mock_memory

    def test_with_full_memory_sets_all_optional(self, builder):
        """Test with_full_memory sets all optional stores."""
        mock_memory = Mock()
        mock_embeddings = Mock()
        mock_continuum = Mock()
        mock_insight = Mock()

        builder.with_full_memory(
            memory=mock_memory,
            debate_embeddings=mock_embeddings,
            continuum_memory=mock_continuum,
            insight_store=mock_insight,
        )

        assert builder._memory is mock_memory
        assert builder._debate_embeddings is mock_embeddings
        assert builder._continuum_memory is mock_continuum
        assert builder._insight_store is mock_insight


# ============================================================================
# Method Chaining Tests
# ============================================================================


class TestMethodChaining:
    """Tests for fluent method chaining."""

    def test_chain_multiple_methods(self, builder, protocol):
        """Test chaining multiple configuration methods."""
        mock_memory = Mock()
        mock_elo = Mock()
        mock_spectator = Mock()

        result = (
            builder.with_protocol(protocol)
            .with_memory(mock_memory)
            .with_elo_system(mock_elo)
            .with_spectator(mock_spectator)
            .with_loop_id("chain-test")
        )

        assert result is builder
        assert builder._protocol is protocol
        assert builder._memory is mock_memory
        assert builder._elo_system is mock_elo
        assert builder._spectator is mock_spectator
        assert builder._loop_id == "chain-test"

    def test_chain_all_with_methods(self, builder):
        """Test chaining all major with_ methods."""
        # This test ensures all with_ methods return self
        result = (
            builder.with_rounds(5)
            .with_memory(Mock())
            .with_debate_embeddings(Mock())
            .with_insight_store(Mock())
            .with_continuum_memory(Mock())
            .with_event_hooks({})
            .with_event_emitter(Mock())
            .with_spectator(Mock())
            .with_recorder(Mock())
            .with_agent_weights({})
            .with_elo_system(Mock())
            .with_persona_manager(Mock())
            .with_calibration_tracker(Mock())
            .with_relationship_tracker(Mock())
            .with_position_tracker(Mock())
            .with_position_ledger(Mock())
            .with_flip_detector(Mock())
            .with_moment_detector(Mock())
            .with_dissent_retriever(Mock())
            .with_evidence_collector(Mock())
            .with_trending_topic(Mock())
            .with_loop_id("test")
            .with_strict_loop_scoping(True)
            .with_circuit_breaker(Mock())
            .with_initial_messages([])
        )

        assert result is builder


# ============================================================================
# Build Tests
# ============================================================================


class TestBuild:
    """Tests for build method."""

    def test_build_creates_arena(self, builder):
        """Test build creates an Arena instance."""
        with patch("aragora.debate.orchestrator.Arena") as MockArena:
            MockArena.return_value = Mock()
            arena = builder.build()

            MockArena.assert_called_once()
            assert arena is not None

    def test_build_passes_environment(self, environment, agents):
        """Test build passes environment to Arena."""
        builder = ArenaBuilder(environment, agents)

        with patch("aragora.debate.orchestrator.Arena") as MockArena:
            MockArena.return_value = Mock()
            builder.build()

            call_kwargs = MockArena.call_args.kwargs
            assert call_kwargs["environment"] is environment

    def test_build_passes_agents(self, environment, agents):
        """Test build passes agents to Arena."""
        builder = ArenaBuilder(environment, agents)

        with patch("aragora.debate.orchestrator.Arena") as MockArena:
            MockArena.return_value = Mock()
            builder.build()

            call_kwargs = MockArena.call_args.kwargs
            assert call_kwargs["agents"] is agents

    def test_build_passes_configured_options(self, builder, protocol):
        """Test build passes all configured options."""
        mock_memory = Mock()
        mock_elo = Mock()

        builder.with_protocol(protocol)
        builder.with_memory(mock_memory)
        builder.with_elo_system(mock_elo)
        builder.with_loop_id("build-test")

        with patch("aragora.debate.orchestrator.Arena") as MockArena:
            MockArena.return_value = Mock()
            builder.build()

            call_kwargs = MockArena.call_args.kwargs
            assert call_kwargs["protocol"] is protocol
            assert call_kwargs["memory"] is mock_memory
            assert call_kwargs["elo_system"] is mock_elo
            assert call_kwargs["loop_id"] == "build-test"

    def test_build_passes_none_for_unconfigured(self, builder):
        """Test build passes None for unconfigured options."""
        with patch("aragora.debate.orchestrator.Arena") as MockArena:
            MockArena.return_value = Mock()
            builder.build()

            call_kwargs = MockArena.call_args.kwargs
            assert call_kwargs["protocol"] is None
            assert call_kwargs["memory"] is None
            assert call_kwargs["elo_system"] is None


# ============================================================================
# create_arena Convenience Function Tests
# ============================================================================


class TestCreateArena:
    """Tests for create_arena convenience function."""

    def test_create_arena_minimal(self, environment, agents):
        """Test create_arena with minimal arguments."""
        with patch("aragora.debate.orchestrator.Arena") as MockArena:
            MockArena.return_value = Mock()
            arena = create_arena(environment, agents)

            MockArena.assert_called_once()
            assert arena is not None

    def test_create_arena_with_protocol(self, environment, agents, protocol):
        """Test create_arena with protocol."""
        with patch("aragora.debate.orchestrator.Arena") as MockArena:
            MockArena.return_value = Mock()
            create_arena(environment, agents, protocol=protocol)

            call_kwargs = MockArena.call_args.kwargs
            assert call_kwargs["protocol"] is protocol

    def test_create_arena_with_memory(self, environment, agents):
        """Test create_arena with memory."""
        mock_memory = Mock()

        with patch("aragora.debate.orchestrator.Arena") as MockArena:
            MockArena.return_value = Mock()
            create_arena(environment, agents, memory=mock_memory)

            call_kwargs = MockArena.call_args.kwargs
            assert call_kwargs["memory"] is mock_memory

    def test_create_arena_with_elo_system(self, environment, agents):
        """Test create_arena with ELO system."""
        mock_elo = Mock()

        with patch("aragora.debate.orchestrator.Arena") as MockArena:
            MockArena.return_value = Mock()
            create_arena(environment, agents, elo_system=mock_elo)

            call_kwargs = MockArena.call_args.kwargs
            assert call_kwargs["elo_system"] is mock_elo

    def test_create_arena_with_all_options(self, environment, agents, protocol):
        """Test create_arena with all common options."""
        mock_memory = Mock()
        mock_elo = Mock()

        with patch("aragora.debate.orchestrator.Arena") as MockArena:
            MockArena.return_value = Mock()
            create_arena(
                environment,
                agents,
                protocol=protocol,
                memory=mock_memory,
                elo_system=mock_elo,
            )

            call_kwargs = MockArena.call_args.kwargs
            assert call_kwargs["protocol"] is protocol
            assert call_kwargs["memory"] is mock_memory
            assert call_kwargs["elo_system"] is mock_elo


# ============================================================================
# Integration Tests
# ============================================================================


class TestArenaBuilderIntegration:
    """Integration tests for ArenaBuilder."""

    def test_full_configuration_workflow(self, environment, agents, protocol):
        """Test complete configuration workflow."""
        mock_memory = Mock()
        mock_elo = Mock()
        mock_spectator = Mock()
        mock_recorder = Mock()
        mock_position_tracker = Mock()

        builder = ArenaBuilder(environment, agents)
        configured = (
            builder.with_protocol(protocol)
            .with_memory(mock_memory)
            .with_elo_system(mock_elo)
            .with_spectator(mock_spectator)
            .with_recorder(mock_recorder)
            .with_position_tracker(mock_position_tracker)
            .with_loop_id("integration-test")
            .with_strict_loop_scoping(True)
            .with_agent_weights({"agent1": 0.9})
        )

        assert configured is builder
        assert builder._protocol is protocol
        assert builder._memory is mock_memory
        assert builder._elo_system is mock_elo
        assert builder._spectator is mock_spectator
        assert builder._recorder is mock_recorder
        assert builder._position_tracker is mock_position_tracker
        assert builder._loop_id == "integration-test"
        assert builder._strict_loop_scoping is True
        assert builder._agent_weights == {"agent1": 0.9}

    def test_builder_reusable_for_multiple_arenas(self, environment, agents):
        """Test builder can be reused to create multiple arenas."""
        builder = ArenaBuilder(environment, agents)

        with patch("aragora.debate.orchestrator.Arena") as MockArena:
            MockArena.return_value = Mock()

            arena1 = builder.with_loop_id("arena-1").build()
            # Modify and build again
            arena2 = builder.with_loop_id("arena-2").build()

            assert MockArena.call_count == 2
            # Last call should have loop_id "arena-2"
            last_call_kwargs = MockArena.call_args.kwargs
            assert last_call_kwargs["loop_id"] == "arena-2"
