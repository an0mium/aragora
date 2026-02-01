"""
Tests for Arena config object decomposition.

Covers all 10 config dataclasses, their defaults, and
backward-compatible merging with individual Arena parameters.
"""

from __future__ import annotations

from dataclasses import fields
from unittest.mock import MagicMock

import pytest

from aragora.config import DEFAULT_ROUNDS
from aragora.debate.arena_config import (
    ALL_CONFIG_CLASSES,
    PRIMARY_CONFIG_CLASSES,
    LEGACY_CONFIG_CLASSES,
    # Primary config classes (new pattern)
    DebateConfig,
    AgentConfig,
    StreamingConfig,
    ObservabilityConfig,
    # Legacy config classes
    BillingConfig,
    BroadcastConfig,
    EvolutionConfig,
    KnowledgeConfig,
    MemoryConfig,
    MLConfig,
    PersonaConfig,
    ResilienceConfig,
    RLMConfig,
    TelemetryConfig,
    TranslationConfig,
)


# ===========================================================================
# Primary Config Classes (New Pattern)
# ===========================================================================


class TestDebateConfig:
    """Tests for DebateConfig - protocol settings configuration."""

    def test_defaults(self):
        c = DebateConfig()
        assert c.rounds == DEFAULT_ROUNDS
        assert c.consensus_threshold == 0.6  # Matches DEBATE_DEFAULTS.consensus_threshold
        assert c.convergence_detection is True
        assert c.convergence_threshold == 0.85
        assert c.divergence_threshold == 0.3
        assert c.timeout_seconds == 0
        assert c.judge_selection == "elo_ranked"
        assert c.enable_adaptive_rounds is False
        assert c.debate_strategy is None
        assert c.enable_judge_termination is False
        assert c.enable_early_stopping is False
        assert c.enable_agent_hierarchy is True
        assert c.hierarchy_config is None

    def test_custom_values(self):
        c = DebateConfig(
            rounds=7,
            consensus_threshold=0.9,
            enable_adaptive_rounds=True,
            enable_agent_hierarchy=False,
        )
        assert c.rounds == 7
        assert c.consensus_threshold == 0.9
        assert c.enable_adaptive_rounds is True
        assert c.enable_agent_hierarchy is False

    def test_apply_to_protocol(self):
        from aragora.debate.protocol import DebateProtocol

        protocol = DebateProtocol()
        config = DebateConfig(rounds=5, consensus_threshold=0.85)
        result = config.apply_to_protocol(protocol)

        assert result is protocol  # Returns same object
        assert protocol.rounds == 5
        assert protocol.consensus_threshold == 0.85


class TestAgentConfig:
    """Tests for AgentConfig - agent management configuration."""

    def test_defaults(self):
        c = AgentConfig()
        assert c.agent_weights is None
        assert c.agent_selector is None
        assert c.use_performance_selection is True
        assert c.circuit_breaker is None
        assert c.use_airlock is False
        assert c.airlock_config is None
        assert c.position_tracker is None
        assert c.position_ledger is None
        assert c.enable_position_ledger is False
        assert c.elo_system is None
        assert c.calibration_tracker is None
        assert c.relationship_tracker is None
        assert c.persona_manager is None
        assert c.vertical is None
        assert c.vertical_persona_manager is None
        assert c.auto_detect_vertical is True
        assert c.fabric is None
        assert c.fabric_config is None

    def test_custom_values(self):
        cb = MagicMock()
        c = AgentConfig(
            use_airlock=True,
            circuit_breaker=cb,
            use_performance_selection=False,
            vertical="healthcare",
        )
        assert c.use_airlock is True
        assert c.circuit_breaker is cb
        assert c.use_performance_selection is False
        assert c.vertical == "healthcare"


class TestStreamingConfig:
    """Tests for StreamingConfig - WebSocket/event configuration."""

    def test_defaults(self):
        c = StreamingConfig()
        assert c.event_hooks is None
        assert c.hook_manager is None
        assert c.event_emitter is None
        assert c.spectator is None
        assert c.recorder is None
        assert c.loop_id == ""
        assert c.strict_loop_scoping is False
        assert c.skill_registry is None
        assert c.enable_skills is False
        assert c.propulsion_engine is None
        assert c.enable_propulsion is False

    def test_custom_values(self):
        emitter = MagicMock()
        c = StreamingConfig(
            event_emitter=emitter,
            loop_id="test-loop-123",
            strict_loop_scoping=True,
            enable_propulsion=True,
        )
        assert c.event_emitter is emitter
        assert c.loop_id == "test-loop-123"
        assert c.strict_loop_scoping is True
        assert c.enable_propulsion is True


class TestObservabilityConfig:
    """Tests for ObservabilityConfig - telemetry/monitoring configuration."""

    def test_defaults(self):
        c = ObservabilityConfig()
        assert c.performance_monitor is None
        assert c.enable_performance_monitor is True
        assert c.enable_telemetry is False
        assert c.prompt_evolver is None
        assert c.enable_prompt_evolution is False
        assert c.breakpoint_manager is None
        assert c.trending_topic is None
        assert c.pulse_manager is None
        assert c.auto_fetch_trending is False
        assert c.population_manager is None
        assert c.auto_evolve is False
        assert c.breeding_threshold == 0.8
        assert c.evidence_collector is None
        assert c.org_id == ""
        assert c.user_id == ""
        assert c.usage_tracker is None
        assert c.broadcast_pipeline is None
        assert c.auto_broadcast is False
        assert c.broadcast_min_confidence == 0.8
        assert c.training_exporter is None
        assert c.auto_export_training is False
        assert c.training_export_min_confidence == 0.75
        assert c.enable_ml_delegation is True
        assert c.ml_delegation_strategy is None
        assert c.ml_delegation_weight == 0.3
        assert c.enable_quality_gates is True
        assert c.quality_gate_threshold == 0.6
        assert c.enable_consensus_estimation is True
        assert c.consensus_early_termination_threshold == 0.85
        assert c.post_debate_workflow is None
        assert c.enable_post_debate_workflow is False
        assert c.post_debate_workflow_threshold == 0.7
        assert c.initial_messages is None

    def test_custom_values(self):
        c = ObservabilityConfig(
            enable_telemetry=True,
            org_id="test-org",
            user_id="test-user",
            enable_ml_delegation=True,
            ml_delegation_weight=0.5,
        )
        assert c.enable_telemetry is True
        assert c.org_id == "test-org"
        assert c.user_id == "test-user"
        assert c.enable_ml_delegation is True
        assert c.ml_delegation_weight == 0.5


class TestPrimaryConfigClasses:
    """Tests for PRIMARY_CONFIG_CLASSES tuple."""

    def test_contains_all_primary_configs(self):
        assert DebateConfig in PRIMARY_CONFIG_CLASSES
        assert AgentConfig in PRIMARY_CONFIG_CLASSES
        assert StreamingConfig in PRIMARY_CONFIG_CLASSES
        assert ObservabilityConfig in PRIMARY_CONFIG_CLASSES

    def test_primary_configs_count(self):
        assert len(PRIMARY_CONFIG_CLASSES) == 4


# ===========================================================================
# Legacy Config Classes (Backward Compatible)
# ===========================================================================


class TestMemoryConfig:
    def test_defaults(self):
        c = MemoryConfig()
        assert c.memory is None
        assert c.continuum_memory is None
        assert c.consensus_memory is None
        assert c.cross_debate_memory is None
        assert c.enable_cross_debate_memory is True
        assert c.debate_embeddings is None
        assert c.insight_store is None

    def test_custom_values(self):
        mem = MagicMock()
        c = MemoryConfig(memory=mem, enable_cross_debate_memory=False)
        assert c.memory is mem
        assert c.enable_cross_debate_memory is False


class TestKnowledgeConfig:
    def test_defaults(self):
        c = KnowledgeConfig()
        assert c.knowledge_mound is None
        assert c.auto_create_knowledge_mound is True
        assert c.enable_knowledge_retrieval is True
        assert c.enable_knowledge_ingestion is True
        assert c.enable_knowledge_extraction is False
        assert c.extraction_min_confidence == 0.3
        assert c.enable_belief_guidance is False

    def test_extraction_enabled(self):
        c = KnowledgeConfig(enable_knowledge_extraction=True, extraction_min_confidence=0.5)
        assert c.enable_knowledge_extraction is True
        assert c.extraction_min_confidence == 0.5


class TestMLConfig:
    def test_defaults(self):
        c = MLConfig()
        assert c.enable_ml_delegation is True
        assert c.ml_delegation_weight == 0.3
        assert c.enable_quality_gates is True
        assert c.quality_gate_threshold == 0.6
        assert c.enable_consensus_estimation is True
        assert c.consensus_early_termination_threshold == 0.85

    def test_custom_weight(self):
        c = MLConfig(enable_ml_delegation=True, ml_delegation_weight=0.7)
        assert c.enable_ml_delegation is True
        assert c.ml_delegation_weight == 0.7

    def test_can_be_disabled(self):
        c = MLConfig(
            enable_ml_delegation=False,
            enable_quality_gates=False,
            enable_consensus_estimation=False,
        )
        assert c.enable_ml_delegation is False
        assert c.enable_quality_gates is False
        assert c.enable_consensus_estimation is False


class TestRLMConfig:
    def test_defaults(self):
        c = RLMConfig()
        assert c.use_rlm_limiter is True
        assert c.rlm_compression_threshold == 3000
        assert c.rlm_max_recent_messages == 5
        assert c.rlm_summary_level == "SUMMARY"
        assert c.rlm_compression_round_threshold == 3


class TestTelemetryConfig:
    def test_defaults(self):
        c = TelemetryConfig()
        assert c.performance_monitor is None
        assert c.enable_performance_monitor is True
        assert c.enable_telemetry is False


class TestPersonaConfig:
    def test_defaults(self):
        c = PersonaConfig()
        assert c.persona_manager is None
        assert c.vertical is None
        assert c.auto_detect_vertical is True


class TestResilienceConfig:
    def test_defaults(self):
        c = ResilienceConfig()
        assert c.circuit_breaker is None
        assert c.use_airlock is False
        assert c.airlock_config is None


class TestEvolutionConfig:
    def test_defaults(self):
        c = EvolutionConfig()
        assert c.population_manager is None
        assert c.auto_evolve is False
        assert c.breeding_threshold == 0.8
        assert c.prompt_evolver is None
        assert c.enable_prompt_evolution is False


class TestBillingConfig:
    def test_defaults(self):
        c = BillingConfig()
        assert c.org_id == ""
        assert c.user_id == ""
        assert c.usage_tracker is None

    def test_custom(self):
        c = BillingConfig(org_id="org-1", user_id="user-1")
        assert c.org_id == "org-1"
        assert c.user_id == "user-1"


class TestBroadcastConfig:
    def test_defaults(self):
        c = BroadcastConfig()
        assert c.broadcast_pipeline is None
        assert c.auto_broadcast is False
        assert c.broadcast_min_confidence == 0.8
        assert c.training_exporter is None
        assert c.auto_export_training is False
        assert c.training_export_min_confidence == 0.75


class TestTranslationConfig:
    def test_defaults(self):
        c = TranslationConfig()
        assert c.translation_service is None
        assert c.multilingual_manager is None
        assert c.enable_translation is False
        assert c.default_language == "en"
        assert c.target_languages is None
        assert c.auto_detect_language is True
        assert c.translate_conclusions is True
        assert c.translation_cache_ttl_seconds == 3600
        assert c.translation_cache_max_entries == 10000

    def test_custom(self):
        c = TranslationConfig(
            enable_translation=True,
            default_language="es",
            target_languages=["en", "fr", "de"],
            translation_cache_ttl_seconds=7200,
        )
        assert c.enable_translation is True
        assert c.default_language == "es"
        assert c.target_languages == ["en", "fr", "de"]
        assert c.translation_cache_ttl_seconds == 7200


class TestAllConfigClasses:
    def test_all_classes_present(self):
        # TranslationConfig was added for multi-language support
        # The exact count may change as more config classes are added
        assert len(ALL_CONFIG_CLASSES) >= 11  # Minimum expected count

    def test_all_are_dataclasses(self):
        for cls in ALL_CONFIG_CLASSES:
            assert len(fields(cls)) > 0

    def test_all_instantiate_with_defaults(self):
        for cls in ALL_CONFIG_CLASSES:
            instance = cls()
            assert instance is not None


class TestArenaConfigToKwargs:
    """Tests for ArenaConfig.to_arena_kwargs() conversion."""

    def test_to_arena_kwargs_returns_dict(self):
        from aragora.debate.arena_config import ArenaConfig

        config = ArenaConfig()
        kwargs = config.to_arena_kwargs()
        assert isinstance(kwargs, dict)
        assert "memory" in kwargs
        assert "org_id" in kwargs
        assert "enable_ml_delegation" in kwargs

    def test_to_arena_kwargs_includes_custom_values(self):
        from aragora.debate.arena_config import ArenaConfig

        config = ArenaConfig(org_id="org-test", user_id="user-test")
        kwargs = config.to_arena_kwargs()
        assert kwargs["org_id"] == "org-test"
        assert kwargs["user_id"] == "user-test"

    def test_arena_config_post_init_sets_platforms(self):
        from aragora.debate.arena_config import ArenaConfig

        config = ArenaConfig()
        assert config.broadcast_platforms == ["rss"]
