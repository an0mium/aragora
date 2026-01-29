"""
Tests for ArenaBuilder - fluent Arena construction.

Covers:
- Builder pattern chaining (all methods return self)
- Component storage and defaults
- InitPhase dependency graph
- Arena build with mocked components
- create_arena() convenience function
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.core import Environment
from aragora.debate.arena_builder import ArenaBuilder, InitPhase, create_arena
from aragora.debate.protocol import DebateProtocol


# =============================================================================
# Fixtures
# =============================================================================


def _make_mock_agent(name: str) -> MagicMock:
    """Create a mock agent with required attributes."""
    agent = MagicMock()
    agent.name = name
    agent.generate = AsyncMock(return_value="mock response")
    agent.critique = AsyncMock(return_value="mock critique")
    return agent


@pytest.fixture
def environment():
    return Environment(task="Test debate topic")


@pytest.fixture
def agents():
    return [_make_mock_agent("claude"), _make_mock_agent("gemini")]


@pytest.fixture
def protocol():
    return DebateProtocol(rounds=3, consensus="majority")


@pytest.fixture
def builder(environment, agents):
    return ArenaBuilder(environment=environment, agents=agents)


# =============================================================================
# InitPhase Tests
# =============================================================================


class TestInitPhase:
    """Tests for the initialization phase enum."""

    def test_all_phases_defined(self):
        phases = list(InitPhase)
        assert InitPhase.CORE in phases
        assert InitPhase.EXTENSIONS in phases
        assert InitPhase.TRACKERS in phases
        assert InitPhase.USER_PARTICIPATION in phases
        assert InitPhase.ROLES in phases
        assert InitPhase.CONVERGENCE in phases
        assert InitPhase.PHASES in phases

    def test_dependency_graph(self):
        deps = InitPhase.dependencies()
        assert isinstance(deps, dict)
        # CORE should have no dependencies
        assert InitPhase.CORE in deps
        assert deps[InitPhase.CORE] == []

    def test_phases_ordered(self):
        """Phases should be defined in dependency order."""
        phases = list(InitPhase)
        assert phases[0] == InitPhase.CORE
        # PHASES should come after CORE
        core_idx = phases.index(InitPhase.CORE)
        phases_idx = phases.index(InitPhase.PHASES)
        assert phases_idx > core_idx


# =============================================================================
# Builder Construction Tests
# =============================================================================


class TestBuilderConstruction:
    """Tests for ArenaBuilder initialization."""

    def test_basic_construction(self, environment, agents):
        builder = ArenaBuilder(environment=environment, agents=agents)
        assert builder is not None

    def test_stores_environment(self, builder, environment):
        assert builder._environment is environment

    def test_stores_agents(self, builder, agents):
        assert builder._agents is agents


# =============================================================================
# Builder Chaining Tests
# =============================================================================


class TestBuilderChaining:
    """Tests that all builder methods return self for chaining."""

    def test_with_protocol(self, builder, protocol):
        result = builder.with_protocol(protocol)
        assert result is builder

    def test_with_rounds(self, builder):
        result = builder.with_rounds(5)
        assert result is builder

    def test_with_memory(self, builder):
        result = builder.with_memory(MagicMock())
        assert result is builder

    def test_with_debate_embeddings(self, builder):
        result = builder.with_debate_embeddings(MagicMock())
        assert result is builder

    def test_with_insight_store(self, builder):
        result = builder.with_insight_store(MagicMock())
        assert result is builder

    def test_with_continuum_memory(self, builder):
        result = builder.with_continuum_memory(MagicMock())
        assert result is builder

    def test_with_event_hooks(self, builder):
        result = builder.with_event_hooks({"on_start": lambda: None})
        assert result is builder

    def test_with_event_emitter(self, builder):
        result = builder.with_event_emitter(MagicMock())
        assert result is builder

    def test_with_spectator(self, builder):
        result = builder.with_spectator(MagicMock())
        assert result is builder

    def test_with_recorder(self, builder):
        result = builder.with_recorder(MagicMock())
        assert result is builder

    def test_with_agent_weights(self, builder):
        result = builder.with_agent_weights({"claude": 1.2, "gemini": 0.8})
        assert result is builder

    def test_with_elo_system(self, builder):
        result = builder.with_elo_system(MagicMock())
        assert result is builder

    def test_with_persona_manager(self, builder):
        result = builder.with_persona_manager(MagicMock())
        assert result is builder

    def test_with_calibration_tracker(self, builder):
        result = builder.with_calibration_tracker(MagicMock())
        assert result is builder

    def test_with_relationship_tracker(self, builder):
        result = builder.with_relationship_tracker(MagicMock())
        assert result is builder

    def test_with_position_tracker(self, builder):
        result = builder.with_position_tracker(MagicMock())
        assert result is builder

    def test_with_position_ledger(self, builder):
        result = builder.with_position_ledger(MagicMock())
        assert result is builder

    def test_with_flip_detector(self, builder):
        result = builder.with_flip_detector(MagicMock())
        assert result is builder

    def test_with_moment_detector(self, builder):
        result = builder.with_moment_detector(MagicMock())
        assert result is builder

    def test_with_dissent_retriever(self, builder):
        result = builder.with_dissent_retriever(MagicMock())
        assert result is builder

    def test_with_evidence_collector(self, builder):
        result = builder.with_evidence_collector(MagicMock())
        assert result is builder

    def test_with_trending_topic(self, builder):
        result = builder.with_trending_topic(MagicMock())
        assert result is builder

    def test_with_loop_id(self, builder):
        result = builder.with_loop_id("test-loop-123")
        assert result is builder

    def test_with_strict_loop_scoping(self, builder):
        result = builder.with_strict_loop_scoping(True)
        assert result is builder

    def test_with_circuit_breaker(self, builder):
        result = builder.with_circuit_breaker(MagicMock())
        assert result is builder

    def test_with_initial_messages(self, builder):
        result = builder.with_initial_messages([{"role": "user", "content": "test"}])
        assert result is builder

    def test_with_airlock(self, builder):
        result = builder.with_airlock(enabled=True)
        assert result is builder

    def test_with_telemetry(self, builder):
        result = builder.with_telemetry(enable_telemetry=True)
        assert result is builder

    def test_with_evolution(self, builder):
        result = builder.with_evolution(auto_evolve=True)
        assert result is builder

    def test_with_checkpointing(self, builder):
        result = builder.with_checkpointing(enable_checkpointing=True)
        assert result is builder

    def test_with_agent_selection(self, builder):
        result = builder.with_agent_selection(use_performance_selection=True)
        assert result is builder

    def test_with_pulse(self, builder):
        result = builder.with_pulse(auto_fetch_trending=True)
        assert result is builder

    def test_with_billing(self, builder):
        result = builder.with_billing(org_id="org-123", user_id="user-456")
        assert result is builder

    def test_with_broadcast(self, builder):
        result = builder.with_broadcast(auto_broadcast=True)
        assert result is builder

    def test_with_training_export(self, builder):
        result = builder.with_training_export(auto_export=True)
        assert result is builder

    def test_with_consensus_memory(self, builder):
        result = builder.with_consensus_memory(MagicMock())
        assert result is builder

    def test_with_rlm_training(self, builder):
        result = builder.with_rlm_training(enabled=True)
        assert result is builder

    def test_with_session(self, builder):
        result = builder.with_session("session-abc")
        assert result is builder


# =============================================================================
# Component Storage Tests
# =============================================================================


class TestComponentStorage:
    """Tests that builder stores components correctly."""

    def test_protocol_storage(self, builder, protocol):
        builder.with_protocol(protocol)
        assert builder._protocol is protocol

    def test_rounds_override(self, builder):
        builder.with_rounds(7)
        assert builder._protocol.rounds == 7

    def test_memory_storage(self, builder):
        mem = MagicMock()
        builder.with_memory(mem)
        assert builder._memory is mem

    def test_embeddings_storage(self, builder):
        emb = MagicMock()
        builder.with_debate_embeddings(emb)
        assert builder._debate_embeddings is emb

    def test_elo_storage(self, builder):
        elo = MagicMock()
        builder.with_elo_system(elo)
        assert builder._elo_system is elo

    def test_loop_id_storage(self, builder):
        builder.with_loop_id("loop-42")
        assert builder._loop_id == "loop-42"

    def test_agent_weights_storage(self, builder):
        weights = {"claude": 1.5, "gemini": 0.9}
        builder.with_agent_weights(weights)
        assert builder._agent_weights == weights

    def test_airlock_storage(self, builder):
        builder.with_airlock(enabled=True)
        assert builder._use_airlock is True

    def test_airlock_disabled(self, builder):
        builder.with_airlock(enabled=False)
        assert builder._use_airlock is False

    def test_billing_storage(self, builder):
        builder.with_billing(org_id="org-1", user_id="user-1")
        assert builder._org_id == "org-1"
        assert builder._user_id == "user-1"

    def test_session_storage(self, builder):
        builder.with_session("sess-123")
        assert builder._session_id == "sess-123"

    def test_event_hooks_storage(self, builder):
        hooks = {"on_round_start": lambda r: None}
        builder.with_event_hooks(hooks)
        assert builder._event_hooks == hooks

    def test_strict_scoping_storage(self, builder):
        builder.with_strict_loop_scoping(True)
        assert builder._strict_loop_scoping is True


# =============================================================================
# Fluent Chaining Integration
# =============================================================================


class TestFluentChaining:
    """Tests that multiple builder calls can be chained."""

    def test_chain_multiple_calls(self, builder, protocol):
        result = (
            builder.with_protocol(protocol)
            .with_loop_id("chain-test")
            .with_airlock(enabled=True)
            .with_billing(org_id="org-1")
            .with_telemetry(enable_telemetry=True)
        )
        assert result is builder
        assert builder._loop_id == "chain-test"
        assert builder._use_airlock is True
        assert builder._org_id == "org-1"

    def test_chain_with_mocked_components(self, builder):
        elo = MagicMock()
        memory = MagicMock()
        result = (
            builder.with_elo_system(elo)
            .with_memory(memory)
            .with_rounds(5)
        )
        assert result is builder
        assert builder._elo_system is elo
        assert builder._memory is memory


# =============================================================================
# Composite Method Tests
# =============================================================================


class TestCompositeMethods:
    """Tests for composite configuration methods."""

    def test_with_full_tracking(self, builder):
        elo = MagicMock()
        persona = MagicMock()
        calibration = MagicMock()
        relationship = MagicMock()

        result = builder.with_full_tracking(
            elo_system=elo,
            persona_manager=persona,
            calibration_tracker=calibration,
            relationship_tracker=relationship,
        )
        assert result is builder
        assert builder._elo_system is elo
        assert builder._persona_manager is persona
        assert builder._calibration_tracker is calibration
        assert builder._relationship_tracker is relationship

    def test_with_full_tracking_minimal(self, builder):
        elo = MagicMock()
        result = builder.with_full_tracking(elo_system=elo)
        assert result is builder
        assert builder._elo_system is elo

    def test_with_full_memory(self, builder):
        memory = MagicMock()
        embeddings = MagicMock()
        continuum = MagicMock()
        insight = MagicMock()

        result = builder.with_full_memory(
            memory=memory,
            debate_embeddings=embeddings,
            continuum_memory=continuum,
            insight_store=insight,
        )
        assert result is builder
        assert builder._memory is memory
        assert builder._debate_embeddings is embeddings
        assert builder._continuum_memory is continuum
        assert builder._insight_store is insight

    def test_with_full_memory_minimal(self, builder):
        memory = MagicMock()
        result = builder.with_full_memory(memory=memory)
        assert result is builder
        assert builder._memory is memory


# =============================================================================
# Default Values Tests
# =============================================================================


class TestDefaults:
    """Tests for default builder values."""

    def test_default_protocol_none(self, builder):
        # Protocol starts as None until explicitly set
        assert builder._protocol is None

    def test_default_no_memory(self, builder):
        assert builder._memory is None

    def test_default_no_elo(self, builder):
        assert builder._elo_system is None

    def test_default_no_airlock(self, builder):
        assert builder._use_airlock is False

    def test_default_loop_id_empty(self, builder):
        assert not builder._loop_id  # None or empty string

    def test_default_no_billing(self, builder):
        assert builder._org_id == ""
        assert builder._user_id == ""


# =============================================================================
# Template Tests
# =============================================================================


class TestTemplate:
    """Tests for template application."""

    def test_with_template(self, builder):
        mock_template = MagicMock()
        mock_template.name = "test_template"

        with patch(
            "aragora.templates.template_to_protocol",
            return_value=DebateProtocol(rounds=5, consensus="unanimous"),
        ):
            result = builder.with_template(mock_template)
            assert result is builder
            assert builder._protocol.rounds == 5

    def test_with_template_overrides(self, builder):
        mock_template = MagicMock()
        mock_template.name = "test_template"

        with patch(
            "aragora.templates.template_to_protocol",
            return_value=DebateProtocol(rounds=5, consensus="unanimous"),
        ):
            result = builder.with_template(mock_template, overrides={"rounds": 10})
            assert result is builder


# =============================================================================
# create_arena Convenience Function
# =============================================================================


class TestCreateArena:
    """Tests for the create_arena convenience function."""

    def test_create_arena_basic(self, environment, agents):
        with patch.object(ArenaBuilder, "build", return_value=MagicMock()):
            arena = create_arena(environment=environment, agents=agents)
            assert arena is not None

    def test_create_arena_with_protocol(self, environment, agents, protocol):
        with patch.object(ArenaBuilder, "build", return_value=MagicMock()):
            arena = create_arena(
                environment=environment,
                agents=agents,
                protocol=protocol,
            )
            assert arena is not None

    def test_create_arena_with_memory(self, environment, agents):
        memory = MagicMock()
        with patch.object(ArenaBuilder, "build", return_value=MagicMock()):
            arena = create_arena(
                environment=environment,
                agents=agents,
                memory=memory,
            )
            assert arena is not None

    def test_create_arena_with_elo(self, environment, agents):
        elo = MagicMock()
        with patch.object(ArenaBuilder, "build", return_value=MagicMock()):
            arena = create_arena(
                environment=environment,
                agents=agents,
                elo_system=elo,
            )
            assert arena is not None


# =============================================================================
# Multilingual Configuration
# =============================================================================


class TestMultilingual:
    """Tests for multilingual configuration."""

    def test_with_multilingual_defaults(self, builder):
        result = builder.with_multilingual()
        assert result is builder

    def test_with_multilingual_custom(self, builder):
        manager = MagicMock()
        result = builder.with_multilingual(
            manager=manager,
            default_language="fr",
            auto_translate=False,
        )
        assert result is builder

    def test_with_multilingual_language(self, builder):
        builder.with_multilingual(default_language="es")
        assert builder._default_language == "es"


# =============================================================================
# Phase 4 Features
# =============================================================================


class TestPhase4Features:
    """Tests for Phase 4 orchestration features."""

    def test_with_hook_manager(self, builder):
        manager = MagicMock()
        result = builder.with_hook_manager(manager)
        assert result is builder

    def test_with_delegation(self, builder):
        strategy = MagicMock()
        result = builder.with_delegation(strategy)
        assert result is builder

    def test_with_cancellation(self, builder):
        token = MagicMock()
        result = builder.with_cancellation(token)
        assert result is builder

    def test_with_stream_chaining(self, builder):
        result = builder.with_stream_chaining(enabled=True)
        assert result is builder

    def test_with_byzantine_consensus(self, builder):
        config = MagicMock()
        result = builder.with_byzantine_consensus(config)
        assert result is builder
