"""
Tests for aragora.debate.arena_config module.

Tests ArenaConfig dataclass which groups optional dependencies and settings
for Arena debate orchestration.
"""

import pytest
from unittest.mock import MagicMock

from aragora.debate.arena_config import ArenaConfig


# ============================================================================
# Default Values Tests
# ============================================================================


class TestArenaConfigDefaults:
    """Tests for default values of ArenaConfig fields."""

    def test_default_loop_id(self):
        """Test default loop_id is empty string."""
        config = ArenaConfig()
        assert config.loop_id == ""

    def test_default_strict_loop_scoping(self):
        """Test default strict_loop_scoping is False."""
        config = ArenaConfig()
        assert config.strict_loop_scoping is False

    def test_default_memory_none(self):
        """Test default memory is None."""
        config = ArenaConfig()
        assert config.memory is None

    def test_default_event_hooks_none(self):
        """Test default event_hooks is None."""
        config = ArenaConfig()
        assert config.event_hooks is None

    def test_default_event_emitter_none(self):
        """Test default event_emitter is None."""
        config = ArenaConfig()
        assert config.event_emitter is None

    def test_default_spectator_none(self):
        """Test default spectator is None."""
        config = ArenaConfig()
        assert config.spectator is None

    def test_default_debate_embeddings_none(self):
        """Test default debate_embeddings is None."""
        config = ArenaConfig()
        assert config.debate_embeddings is None

    def test_default_insight_store_none(self):
        """Test default insight_store is None."""
        config = ArenaConfig()
        assert config.insight_store is None

    def test_default_recorder_none(self):
        """Test default recorder is None."""
        config = ArenaConfig()
        assert config.recorder is None

    def test_default_circuit_breaker_none(self):
        """Test default circuit_breaker is None."""
        config = ArenaConfig()
        assert config.circuit_breaker is None

    def test_default_evidence_collector_none(self):
        """Test default evidence_collector is None."""
        config = ArenaConfig()
        assert config.evidence_collector is None

    def test_default_agent_weights_none(self):
        """Test default agent_weights is None."""
        config = ArenaConfig()
        assert config.agent_weights is None

    def test_default_position_tracker_none(self):
        """Test default position_tracker is None."""
        config = ArenaConfig()
        assert config.position_tracker is None

    def test_default_position_ledger_none(self):
        """Test default position_ledger is None."""
        config = ArenaConfig()
        assert config.position_ledger is None

    def test_default_enable_position_ledger(self):
        """Test default enable_position_ledger is False."""
        config = ArenaConfig()
        assert config.enable_position_ledger is False

    def test_default_elo_system_none(self):
        """Test default elo_system is None."""
        config = ArenaConfig()
        assert config.elo_system is None

    def test_default_persona_manager_none(self):
        """Test default persona_manager is None."""
        config = ArenaConfig()
        assert config.persona_manager is None

    def test_default_dissent_retriever_none(self):
        """Test default dissent_retriever is None."""
        config = ArenaConfig()
        assert config.dissent_retriever is None

    def test_default_consensus_memory_none(self):
        """Test default consensus_memory is None."""
        config = ArenaConfig()
        assert config.consensus_memory is None

    def test_default_flip_detector_none(self):
        """Test default flip_detector is None."""
        config = ArenaConfig()
        assert config.flip_detector is None

    def test_default_calibration_tracker_none(self):
        """Test default calibration_tracker is None."""
        config = ArenaConfig()
        assert config.calibration_tracker is None

    def test_default_continuum_memory_none(self):
        """Test default continuum_memory is None."""
        config = ArenaConfig()
        assert config.continuum_memory is None

    def test_default_relationship_tracker_none(self):
        """Test default relationship_tracker is None."""
        config = ArenaConfig()
        assert config.relationship_tracker is None

    def test_default_moment_detector_none(self):
        """Test default moment_detector is None."""
        config = ArenaConfig()
        assert config.moment_detector is None

    def test_default_tier_analytics_tracker_none(self):
        """Test default tier_analytics_tracker is None."""
        config = ArenaConfig()
        assert config.tier_analytics_tracker is None

    def test_default_population_manager_none(self):
        """Test default population_manager is None."""
        config = ArenaConfig()
        assert config.population_manager is None

    def test_default_auto_evolve(self):
        """Test default auto_evolve is False."""
        config = ArenaConfig()
        assert config.auto_evolve is False

    def test_default_breeding_threshold(self):
        """Test default breeding_threshold is 0.8."""
        config = ArenaConfig()
        assert config.breeding_threshold == 0.8

    def test_default_initial_messages_none(self):
        """Test default initial_messages is None."""
        config = ArenaConfig()
        assert config.initial_messages is None

    def test_default_trending_topic_none(self):
        """Test default trending_topic is None."""
        config = ArenaConfig()
        assert config.trending_topic is None

    def test_default_pulse_manager_none(self):
        """Test default pulse_manager is None."""
        config = ArenaConfig()
        assert config.pulse_manager is None

    def test_default_auto_fetch_trending(self):
        """Test default auto_fetch_trending is False."""
        config = ArenaConfig()
        assert config.auto_fetch_trending is False

    def test_default_breakpoint_manager_none(self):
        """Test default breakpoint_manager is None."""
        config = ArenaConfig()
        assert config.breakpoint_manager is None

    def test_default_checkpoint_manager_none(self):
        """Test default checkpoint_manager is None."""
        config = ArenaConfig()
        assert config.checkpoint_manager is None

    def test_default_enable_checkpointing(self):
        """Test default enable_checkpointing is True."""
        config = ArenaConfig()
        assert config.enable_checkpointing is True

    def test_default_performance_monitor_none(self):
        """Test default performance_monitor is None."""
        config = ArenaConfig()
        assert config.performance_monitor is None

    def test_default_enable_performance_monitor(self):
        """Test default enable_performance_monitor is False."""
        config = ArenaConfig()
        assert config.enable_performance_monitor is False

    def test_default_enable_telemetry(self):
        """Test default enable_telemetry is False."""
        config = ArenaConfig()
        assert config.enable_telemetry is False

    def test_default_agent_selector_none(self):
        """Test default agent_selector is None."""
        config = ArenaConfig()
        assert config.agent_selector is None

    def test_default_use_performance_selection(self):
        """Test default use_performance_selection is False."""
        config = ArenaConfig()
        assert config.use_performance_selection is False

    def test_default_use_airlock(self):
        """Test default use_airlock is False."""
        config = ArenaConfig()
        assert config.use_airlock is False

    def test_default_airlock_config_none(self):
        """Test default airlock_config is None."""
        config = ArenaConfig()
        assert config.airlock_config is None

    def test_default_prompt_evolver_none(self):
        """Test default prompt_evolver is None."""
        config = ArenaConfig()
        assert config.prompt_evolver is None

    def test_default_enable_prompt_evolution(self):
        """Test default enable_prompt_evolution is False."""
        config = ArenaConfig()
        assert config.enable_prompt_evolution is False

    def test_default_org_id(self):
        """Test default org_id is empty string."""
        config = ArenaConfig()
        assert config.org_id == ""

    def test_default_user_id(self):
        """Test default user_id is empty string."""
        config = ArenaConfig()
        assert config.user_id == ""

    def test_default_usage_tracker_none(self):
        """Test default usage_tracker is None."""
        config = ArenaConfig()
        assert config.usage_tracker is None


# ============================================================================
# Single Field Tests
# ============================================================================


class TestArenaConfigSingleFields:
    """Tests for setting individual ArenaConfig fields."""

    def test_set_loop_id(self):
        """Test setting loop_id."""
        config = ArenaConfig(loop_id="test-loop-123")
        assert config.loop_id == "test-loop-123"

    def test_set_strict_loop_scoping(self):
        """Test setting strict_loop_scoping."""
        config = ArenaConfig(strict_loop_scoping=True)
        assert config.strict_loop_scoping is True

    def test_set_memory(self):
        """Test setting memory."""
        mock_memory = MagicMock()
        config = ArenaConfig(memory=mock_memory)
        assert config.memory == mock_memory

    def test_set_event_hooks(self):
        """Test setting event_hooks."""
        hooks = {"on_start": MagicMock()}
        config = ArenaConfig(event_hooks=hooks)
        assert config.event_hooks == hooks

    def test_set_event_emitter(self):
        """Test setting event_emitter."""
        mock_emitter = MagicMock()
        config = ArenaConfig(event_emitter=mock_emitter)
        assert config.event_emitter == mock_emitter

    def test_set_spectator(self):
        """Test setting spectator."""
        mock_spectator = MagicMock()
        config = ArenaConfig(spectator=mock_spectator)
        assert config.spectator == mock_spectator

    def test_set_elo_system(self):
        """Test setting elo_system."""
        mock_elo = MagicMock()
        config = ArenaConfig(elo_system=mock_elo)
        assert config.elo_system == mock_elo

    def test_set_auto_evolve(self):
        """Test setting auto_evolve."""
        config = ArenaConfig(auto_evolve=True)
        assert config.auto_evolve is True

    def test_set_breeding_threshold(self):
        """Test setting breeding_threshold."""
        config = ArenaConfig(breeding_threshold=0.95)
        assert config.breeding_threshold == 0.95

    def test_set_enable_checkpointing(self):
        """Test setting enable_checkpointing."""
        config = ArenaConfig(enable_checkpointing=True)
        assert config.enable_checkpointing is True

    def test_set_enable_telemetry(self):
        """Test setting enable_telemetry."""
        config = ArenaConfig(enable_telemetry=True)
        assert config.enable_telemetry is True

    def test_set_use_airlock(self):
        """Test setting use_airlock."""
        config = ArenaConfig(use_airlock=True)
        assert config.use_airlock is True

    def test_set_org_id(self):
        """Test setting org_id."""
        config = ArenaConfig(org_id="org-123")
        assert config.org_id == "org-123"

    def test_set_user_id(self):
        """Test setting user_id."""
        config = ArenaConfig(user_id="user-456")
        assert config.user_id == "user-456"


# ============================================================================
# Multi-Tenancy Tests
# ============================================================================


class TestArenaConfigMultiTenancy:
    """Tests for multi-tenancy configuration."""

    def test_org_and_user_id_together(self):
        """Test setting both org_id and user_id."""
        config = ArenaConfig(
            org_id="org-acme",
            user_id="user-john",
        )
        assert config.org_id == "org-acme"
        assert config.user_id == "user-john"

    def test_org_with_usage_tracker(self):
        """Test org_id with usage_tracker for billing."""
        mock_tracker = MagicMock()
        config = ArenaConfig(
            org_id="org-123",
            usage_tracker=mock_tracker,
        )
        assert config.org_id == "org-123"
        assert config.usage_tracker == mock_tracker


# ============================================================================
# Evolution/Genesis Tests
# ============================================================================


class TestArenaConfigEvolution:
    """Tests for evolution/genesis configuration."""

    def test_evolution_with_population_manager(self):
        """Test evolution with population_manager."""
        mock_population = MagicMock()
        config = ArenaConfig(
            population_manager=mock_population,
            auto_evolve=True,
            breeding_threshold=0.9,
        )
        assert config.population_manager == mock_population
        assert config.auto_evolve is True
        assert config.breeding_threshold == 0.9

    def test_evolution_disabled_by_default(self):
        """Test evolution is disabled by default even with population_manager."""
        mock_population = MagicMock()
        config = ArenaConfig(population_manager=mock_population)
        assert config.auto_evolve is False


# ============================================================================
# Airlock Tests
# ============================================================================


class TestArenaConfigAirlock:
    """Tests for airlock resilience configuration."""

    def test_airlock_with_config(self):
        """Test airlock with custom configuration."""
        mock_config = MagicMock()
        config = ArenaConfig(
            use_airlock=True,
            airlock_config=mock_config,
        )
        assert config.use_airlock is True
        assert config.airlock_config == mock_config

    def test_airlock_without_config(self):
        """Test airlock without custom config uses defaults."""
        config = ArenaConfig(use_airlock=True)
        assert config.use_airlock is True
        assert config.airlock_config is None


# ============================================================================
# Tracking Subsystems Tests
# ============================================================================


class TestArenaConfigTracking:
    """Tests for tracking subsystem configuration."""

    def test_position_tracking(self):
        """Test position tracking configuration."""
        mock_tracker = MagicMock()
        mock_ledger = MagicMock()
        config = ArenaConfig(
            position_tracker=mock_tracker,
            position_ledger=mock_ledger,
            enable_position_ledger=True,
        )
        assert config.position_tracker == mock_tracker
        assert config.position_ledger == mock_ledger
        assert config.enable_position_ledger is True

    def test_elo_with_calibration(self):
        """Test ELO with calibration tracking."""
        mock_elo = MagicMock()
        mock_calibration = MagicMock()
        config = ArenaConfig(
            elo_system=mock_elo,
            calibration_tracker=mock_calibration,
        )
        assert config.elo_system == mock_elo
        assert config.calibration_tracker == mock_calibration

    def test_relationship_and_flip_detection(self):
        """Test relationship tracker with flip detection."""
        mock_relationship = MagicMock()
        mock_flip = MagicMock()
        config = ArenaConfig(
            relationship_tracker=mock_relationship,
            flip_detector=mock_flip,
        )
        assert config.relationship_tracker == mock_relationship
        assert config.flip_detector == mock_flip


# ============================================================================
# Memory Systems Tests
# ============================================================================


class TestArenaConfigMemory:
    """Tests for memory system configuration."""

    def test_multiple_memory_systems(self):
        """Test configuring multiple memory systems."""
        mock_critique = MagicMock()
        mock_continuum = MagicMock()
        mock_embeddings = MagicMock()
        mock_consensus = MagicMock()
        config = ArenaConfig(
            memory=mock_critique,
            continuum_memory=mock_continuum,
            debate_embeddings=mock_embeddings,
            consensus_memory=mock_consensus,
        )
        assert config.memory == mock_critique
        assert config.continuum_memory == mock_continuum
        assert config.debate_embeddings == mock_embeddings
        assert config.consensus_memory == mock_consensus

    def test_insight_store(self):
        """Test insight store configuration."""
        mock_store = MagicMock()
        config = ArenaConfig(insight_store=mock_store)
        assert config.insight_store == mock_store


# ============================================================================
# Recording Tests
# ============================================================================


class TestArenaConfigRecording:
    """Tests for recording and evidence configuration."""

    def test_recorder_with_evidence(self):
        """Test recorder with evidence collector."""
        mock_recorder = MagicMock()
        mock_evidence = MagicMock()
        config = ArenaConfig(
            recorder=mock_recorder,
            evidence_collector=mock_evidence,
        )
        assert config.recorder == mock_recorder
        assert config.evidence_collector == mock_evidence


# ============================================================================
# Checkpointing Tests
# ============================================================================


class TestArenaConfigCheckpointing:
    """Tests for checkpointing configuration."""

    def test_checkpoint_manager(self):
        """Test checkpoint manager configuration."""
        mock_checkpoint = MagicMock()
        config = ArenaConfig(
            checkpoint_manager=mock_checkpoint,
            enable_checkpointing=True,
        )
        assert config.checkpoint_manager == mock_checkpoint
        assert config.enable_checkpointing is True

    def test_auto_create_checkpointing(self):
        """Test enable_checkpointing for auto-creation."""
        config = ArenaConfig(enable_checkpointing=True)
        assert config.enable_checkpointing is True
        assert config.checkpoint_manager is None


# ============================================================================
# Performance Monitoring Tests
# ============================================================================


class TestArenaConfigPerformance:
    """Tests for performance monitoring configuration."""

    def test_performance_monitor(self):
        """Test performance monitor configuration."""
        mock_monitor = MagicMock()
        config = ArenaConfig(
            performance_monitor=mock_monitor,
            enable_performance_monitor=True,
        )
        assert config.performance_monitor == mock_monitor
        assert config.enable_performance_monitor is True

    def test_telemetry_enabled(self):
        """Test telemetry enabled configuration."""
        config = ArenaConfig(
            enable_telemetry=True,
            enable_performance_monitor=True,
        )
        assert config.enable_telemetry is True
        assert config.enable_performance_monitor is True


# ============================================================================
# Agent Selection Tests
# ============================================================================


class TestArenaConfigAgentSelection:
    """Tests for agent selection configuration."""

    def test_agent_selector(self):
        """Test agent selector configuration."""
        mock_selector = MagicMock()
        config = ArenaConfig(
            agent_selector=mock_selector,
            use_performance_selection=True,
        )
        assert config.agent_selector == mock_selector
        assert config.use_performance_selection is True

    def test_agent_weights(self):
        """Test agent weights configuration."""
        weights = {"agent-1": 1.0, "agent-2": 0.8}
        config = ArenaConfig(agent_weights=weights)
        assert config.agent_weights == weights


# ============================================================================
# Trending/Pulse Tests
# ============================================================================


class TestArenaConfigTrending:
    """Tests for trending topic and pulse configuration."""

    def test_trending_topic(self):
        """Test trending topic configuration."""
        mock_topic = MagicMock()
        config = ArenaConfig(trending_topic=mock_topic)
        assert config.trending_topic == mock_topic

    def test_pulse_manager_with_auto_fetch(self):
        """Test pulse manager with auto-fetch."""
        mock_pulse = MagicMock()
        config = ArenaConfig(
            pulse_manager=mock_pulse,
            auto_fetch_trending=True,
        )
        assert config.pulse_manager == mock_pulse
        assert config.auto_fetch_trending is True


# ============================================================================
# Prompt Evolution Tests
# ============================================================================


class TestArenaConfigPromptEvolution:
    """Tests for prompt evolution configuration."""

    def test_prompt_evolver(self):
        """Test prompt evolver configuration."""
        mock_evolver = MagicMock()
        config = ArenaConfig(
            prompt_evolver=mock_evolver,
            enable_prompt_evolution=True,
        )
        assert config.prompt_evolver == mock_evolver
        assert config.enable_prompt_evolution is True


# ============================================================================
# Full Configuration Tests
# ============================================================================


class TestArenaConfigFullSetup:
    """Tests for full production-like configuration."""

    def test_minimal_production_config(self):
        """Test minimal production configuration."""
        config = ArenaConfig(
            loop_id="debate-123",
            strict_loop_scoping=True,
            enable_telemetry=True,
        )
        assert config.loop_id == "debate-123"
        assert config.strict_loop_scoping is True
        assert config.enable_telemetry is True

    def test_full_production_config(self):
        """Test full production configuration with all subsystems."""
        mock_memory = MagicMock()
        mock_elo = MagicMock()
        mock_spectator = MagicMock()
        mock_continuum = MagicMock()
        mock_usage = MagicMock()

        config = ArenaConfig(
            loop_id="debate-456",
            strict_loop_scoping=True,
            memory=mock_memory,
            elo_system=mock_elo,
            spectator=mock_spectator,
            continuum_memory=mock_continuum,
            enable_telemetry=True,
            enable_performance_monitor=True,
            use_airlock=True,
            org_id="org-production",
            user_id="user-admin",
            usage_tracker=mock_usage,
        )

        assert config.loop_id == "debate-456"
        assert config.strict_loop_scoping is True
        assert config.memory == mock_memory
        assert config.elo_system == mock_elo
        assert config.spectator == mock_spectator
        assert config.continuum_memory == mock_continuum
        assert config.enable_telemetry is True
        assert config.enable_performance_monitor is True
        assert config.use_airlock is True
        assert config.org_id == "org-production"
        assert config.user_id == "user-admin"
        assert config.usage_tracker == mock_usage


# ============================================================================
# Dataclass Behavior Tests
# ============================================================================


class TestArenaConfigDataclass:
    """Tests for dataclass behavior."""

    def test_is_dataclass(self):
        """Test ArenaConfig is a proper dataclass."""
        from dataclasses import is_dataclass

        assert is_dataclass(ArenaConfig)

    def test_equality(self):
        """Test ArenaConfig equality comparison."""
        config1 = ArenaConfig(loop_id="test")
        config2 = ArenaConfig(loop_id="test")
        config3 = ArenaConfig(loop_id="other")

        assert config1 == config2
        assert config1 != config3

    def test_repr(self):
        """Test ArenaConfig has useful repr."""
        config = ArenaConfig(loop_id="test-123")
        repr_str = repr(config)
        assert "ArenaConfig" in repr_str
        assert "loop_id" in repr_str

    def test_field_access(self):
        """Test field access via attribute."""
        config = ArenaConfig(loop_id="test")
        assert hasattr(config, "loop_id")
        assert config.loop_id == "test"

    def test_field_mutation(self):
        """Test fields can be mutated after creation."""
        config = ArenaConfig()
        config.loop_id = "mutated"
        assert config.loop_id == "mutated"


# ============================================================================
# Edge Cases Tests
# ============================================================================


class TestArenaConfigEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_loop_id(self):
        """Test empty loop_id is valid."""
        config = ArenaConfig(loop_id="")
        assert config.loop_id == ""

    def test_breeding_threshold_zero(self):
        """Test breeding_threshold can be zero."""
        config = ArenaConfig(breeding_threshold=0.0)
        assert config.breeding_threshold == 0.0

    def test_breeding_threshold_one(self):
        """Test breeding_threshold can be one."""
        config = ArenaConfig(breeding_threshold=1.0)
        assert config.breeding_threshold == 1.0

    def test_empty_event_hooks(self):
        """Test empty event_hooks dict is valid."""
        config = ArenaConfig(event_hooks={})
        assert config.event_hooks == {}

    def test_empty_agent_weights(self):
        """Test empty agent_weights dict is valid."""
        config = ArenaConfig(agent_weights={})
        assert config.agent_weights == {}

    def test_initial_messages_empty_list(self):
        """Test empty initial_messages list is valid."""
        config = ArenaConfig(initial_messages=[])
        assert config.initial_messages == []


# ============================================================================
# Module Exports Tests
# ============================================================================


class TestArenaConfigExports:
    """Tests for module exports."""

    def test_all_exports(self):
        """Test __all__ exports ArenaConfig."""
        from aragora.debate import arena_config

        assert "ArenaConfig" in arena_config.__all__

    def test_import_from_module(self):
        """Test ArenaConfig can be imported from module."""
        from aragora.debate.arena_config import ArenaConfig as AC

        assert AC is ArenaConfig
