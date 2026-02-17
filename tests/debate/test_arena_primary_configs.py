"""Tests for primary and legacy Arena configuration dataclasses.

Covers DebateConfig, AgentConfig, StreamingConfig, ObservabilityConfig,
MemoryConfig, SupermemoryConfig, KnowledgeConfig, MLConfig, RLMConfig,
TelemetryConfig, PersonaConfig, ResilienceConfig, EvolutionConfig,
BillingConfig, BroadcastConfig, TranslationConfig, and module-level
constants.
"""

from __future__ import annotations

from dataclasses import fields
from unittest.mock import MagicMock

import pytest

from aragora.debate.arena_primary_configs import (
    ALL_CONFIG_CLASSES,
    LEGACY_CONFIG_CLASSES,
    PRIMARY_CONFIG_CLASSES,
    AgentConfig,
    BillingConfig,
    BroadcastConfig,
    DebateConfig,
    EvolutionConfig,
    KnowledgeConfig,
    MLConfig,
    MemoryConfig,
    ObservabilityConfig,
    PersonaConfig,
    ResilienceConfig,
    RLMConfig,
    StreamingConfig,
    SupermemoryConfig,
    TelemetryConfig,
    TranslationConfig,
)


# ---------------------------------------------------------------------------
# DebateConfig
# ---------------------------------------------------------------------------


class TestDebateConfig:
    def test_defaults(self):
        cfg = DebateConfig()
        assert cfg.consensus_threshold == 0.6
        assert cfg.convergence_detection is True
        assert cfg.convergence_threshold == 0.85
        assert cfg.divergence_threshold == 0.3
        assert cfg.timeout_seconds == 0
        assert cfg.judge_selection == "elo_ranked"
        assert cfg.enable_adaptive_rounds is False
        assert cfg.debate_strategy is None
        assert cfg.enable_judge_termination is False
        assert cfg.enable_early_stopping is False
        assert cfg.enable_agent_hierarchy is True

    def test_apply_to_protocol(self):
        cfg = DebateConfig(
            rounds=5,
            consensus_threshold=0.8,
            convergence_threshold=0.9,
        )
        protocol = MagicMock()
        result = cfg.apply_to_protocol(protocol)
        assert result.rounds == 5
        assert result.consensus_threshold == 0.8
        assert result.convergence_threshold == 0.9
        assert result is protocol


# ---------------------------------------------------------------------------
# AgentConfig
# ---------------------------------------------------------------------------


class TestAgentConfig:
    def test_defaults(self):
        cfg = AgentConfig()
        assert cfg.agent_weights is None
        assert cfg.agent_selector is None
        assert cfg.use_performance_selection is True
        assert cfg.circuit_breaker is None
        assert cfg.use_airlock is False
        assert cfg.airlock_config is None
        assert cfg.position_tracker is None
        assert cfg.elo_system is None
        assert cfg.calibration_tracker is None
        assert cfg.persona_manager is None
        assert cfg.vertical is None
        assert cfg.auto_detect_vertical is True

    def test_custom_values(self):
        elo = MagicMock()
        cfg = AgentConfig(
            agent_weights={"claude": 1.5},
            use_airlock=True,
            elo_system=elo,
            vertical="healthcare",
        )
        assert cfg.agent_weights == {"claude": 1.5}
        assert cfg.use_airlock is True
        assert cfg.elo_system is elo
        assert cfg.vertical == "healthcare"


# ---------------------------------------------------------------------------
# StreamingConfig
# ---------------------------------------------------------------------------


class TestStreamingConfig:
    def test_defaults(self):
        cfg = StreamingConfig()
        assert cfg.event_hooks is None
        assert cfg.hook_manager is None
        assert cfg.event_emitter is None
        assert cfg.spectator is None
        assert cfg.recorder is None
        assert cfg.loop_id == ""
        assert cfg.strict_loop_scoping is False
        assert cfg.skill_registry is None
        assert cfg.enable_skills is False
        assert cfg.enable_propulsion is False

    def test_with_spectator(self):
        spec = MagicMock()
        cfg = StreamingConfig(spectator=spec, loop_id="loop-1")
        assert cfg.spectator is spec
        assert cfg.loop_id == "loop-1"


# ---------------------------------------------------------------------------
# ObservabilityConfig
# ---------------------------------------------------------------------------


class TestObservabilityConfig:
    def test_defaults(self):
        cfg = ObservabilityConfig()
        assert cfg.enable_performance_monitor is True
        assert cfg.enable_telemetry is False
        assert cfg.enable_prompt_evolution is False
        assert cfg.auto_fetch_trending is False
        assert cfg.auto_evolve is False
        assert cfg.breeding_threshold == 0.8
        assert cfg.org_id == ""
        assert cfg.user_id == ""
        assert cfg.enable_ml_delegation is True
        assert cfg.ml_delegation_weight == 0.3
        assert cfg.enable_quality_gates is True
        assert cfg.quality_gate_threshold == 0.6
        assert cfg.enable_consensus_estimation is True
        assert cfg.enable_post_debate_workflow is False

    def test_ml_settings(self):
        cfg = ObservabilityConfig(
            enable_ml_delegation=False,
            quality_gate_threshold=0.8,
            consensus_early_termination_threshold=0.9,
        )
        assert cfg.enable_ml_delegation is False
        assert cfg.quality_gate_threshold == 0.8
        assert cfg.consensus_early_termination_threshold == 0.9


# ---------------------------------------------------------------------------
# MemoryConfig
# ---------------------------------------------------------------------------


class TestMemoryConfig:
    def test_defaults(self):
        cfg = MemoryConfig()
        assert cfg.memory is None
        assert cfg.continuum_memory is None
        assert cfg.consensus_memory is None
        assert cfg.enable_cross_debate_memory is True
        assert cfg.auto_create_knowledge_mound is True
        assert cfg.enable_knowledge_retrieval is True
        assert cfg.enable_knowledge_ingestion is True
        assert cfg.enable_knowledge_extraction is False
        assert cfg.extraction_min_confidence == 0.3
        assert cfg.enable_belief_guidance is True
        assert cfg.use_rlm_limiter is True
        assert cfg.rlm_compression_threshold == 3000
        assert cfg.enable_supermemory is False
        assert cfg.enable_checkpointing is True

    def test_codebase_grounding(self):
        cfg = MemoryConfig(
            codebase_path="/tmp/repo",
            enable_codebase_grounding=True,
            codebase_persist_to_km=True,
        )
        assert cfg.codebase_path == "/tmp/repo"
        assert cfg.enable_codebase_grounding is True
        assert cfg.codebase_persist_to_km is True

    def test_supermemory_settings(self):
        adapter = MagicMock()
        cfg = MemoryConfig(
            enable_supermemory=True,
            supermemory_adapter=adapter,
            supermemory_max_context_items=20,
        )
        assert cfg.enable_supermemory is True
        assert cfg.supermemory_adapter is adapter
        assert cfg.supermemory_max_context_items == 20


# ---------------------------------------------------------------------------
# SupermemoryConfig
# ---------------------------------------------------------------------------


class TestSupermemoryConfig:
    def test_defaults(self):
        cfg = SupermemoryConfig()
        assert cfg.enable_supermemory is False
        assert cfg.supermemory_inject_on_start is True
        assert cfg.supermemory_max_context_items == 10
        assert cfg.supermemory_sync_on_conclusion is True
        assert cfg.supermemory_min_confidence_for_sync == 0.7
        assert cfg.supermemory_enable_privacy_filter is True
        assert cfg.supermemory_enable_resilience is True
        assert cfg.supermemory_enable_km_adapter is False


# ---------------------------------------------------------------------------
# KnowledgeConfig
# ---------------------------------------------------------------------------


class TestKnowledgeConfig:
    def test_defaults(self):
        cfg = KnowledgeConfig()
        assert cfg.knowledge_mound is None
        assert cfg.auto_create_knowledge_mound is True
        assert cfg.enable_knowledge_retrieval is True
        assert cfg.enable_knowledge_ingestion is True
        assert cfg.enable_knowledge_extraction is False
        assert cfg.extraction_min_confidence == 0.3
        assert cfg.enable_auto_revalidation is False
        assert cfg.revalidation_staleness_threshold == 0.7
        assert cfg.revalidation_check_interval_seconds == 3600
        assert cfg.enable_belief_guidance is True


# ---------------------------------------------------------------------------
# MLConfig
# ---------------------------------------------------------------------------


class TestMLConfig:
    def test_defaults(self):
        cfg = MLConfig()
        assert cfg.enable_ml_delegation is True
        assert cfg.ml_delegation_weight == 0.3
        assert cfg.enable_quality_gates is True
        assert cfg.quality_gate_threshold == 0.6
        assert cfg.enable_consensus_estimation is True
        assert cfg.consensus_early_termination_threshold == 0.85


# ---------------------------------------------------------------------------
# RLMConfig
# ---------------------------------------------------------------------------


class TestRLMConfig:
    def test_defaults(self):
        cfg = RLMConfig()
        assert cfg.use_rlm_limiter is True
        assert cfg.rlm_limiter is None
        assert cfg.rlm_compression_threshold == 3000
        assert cfg.rlm_max_recent_messages == 5
        assert cfg.rlm_summary_level == "SUMMARY"
        assert cfg.rlm_compression_round_threshold == 3


# ---------------------------------------------------------------------------
# TelemetryConfig
# ---------------------------------------------------------------------------


class TestTelemetryConfig:
    def test_defaults(self):
        cfg = TelemetryConfig()
        assert cfg.performance_monitor is None
        assert cfg.enable_performance_monitor is True
        assert cfg.enable_telemetry is False


# ---------------------------------------------------------------------------
# PersonaConfig
# ---------------------------------------------------------------------------


class TestPersonaConfig:
    def test_defaults(self):
        cfg = PersonaConfig()
        assert cfg.persona_manager is None
        assert cfg.vertical is None
        assert cfg.auto_detect_vertical is True
        assert cfg.vertical_persona_manager is None


# ---------------------------------------------------------------------------
# ResilienceConfig
# ---------------------------------------------------------------------------


class TestResilienceConfig:
    def test_defaults(self):
        cfg = ResilienceConfig()
        assert cfg.circuit_breaker is None
        assert cfg.use_airlock is False
        assert cfg.airlock_config is None


# ---------------------------------------------------------------------------
# EvolutionConfig
# ---------------------------------------------------------------------------


class TestEvolutionConfig:
    def test_defaults(self):
        cfg = EvolutionConfig()
        assert cfg.population_manager is None
        assert cfg.auto_evolve is False
        assert cfg.breeding_threshold == 0.8
        assert cfg.prompt_evolver is None
        assert cfg.enable_prompt_evolution is False


# ---------------------------------------------------------------------------
# BillingConfig
# ---------------------------------------------------------------------------


class TestBillingConfig:
    def test_defaults(self):
        cfg = BillingConfig()
        assert cfg.org_id == ""
        assert cfg.user_id == ""
        assert cfg.usage_tracker is None


# ---------------------------------------------------------------------------
# BroadcastConfig
# ---------------------------------------------------------------------------


class TestBroadcastConfig:
    def test_defaults(self):
        cfg = BroadcastConfig()
        assert cfg.broadcast_pipeline is None
        assert cfg.auto_broadcast is False
        assert cfg.broadcast_min_confidence == 0.8
        assert cfg.broadcast_platforms is None
        assert cfg.training_exporter is None
        assert cfg.auto_export_training is False
        assert cfg.training_export_min_confidence == 0.75


# ---------------------------------------------------------------------------
# TranslationConfig
# ---------------------------------------------------------------------------


class TestTranslationConfig:
    def test_defaults(self):
        cfg = TranslationConfig()
        assert cfg.translation_service is None
        assert cfg.enable_translation is False
        assert cfg.default_language == "en"
        assert cfg.target_languages is None
        assert cfg.auto_detect_language is True
        assert cfg.translate_conclusions is True
        assert cfg.translation_cache_ttl_seconds == 3600
        assert cfg.translation_cache_max_entries == 10000

    def test_multi_language(self):
        cfg = TranslationConfig(
            enable_translation=True,
            target_languages=["en", "fr", "de"],
        )
        assert cfg.target_languages == ["en", "fr", "de"]


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


class TestModuleConstants:
    def test_primary_config_classes(self):
        assert DebateConfig in PRIMARY_CONFIG_CLASSES
        assert AgentConfig in PRIMARY_CONFIG_CLASSES
        assert StreamingConfig in PRIMARY_CONFIG_CLASSES
        assert ObservabilityConfig in PRIMARY_CONFIG_CLASSES
        assert len(PRIMARY_CONFIG_CLASSES) == 4

    def test_legacy_config_classes(self):
        assert KnowledgeConfig in LEGACY_CONFIG_CLASSES
        assert SupermemoryConfig in LEGACY_CONFIG_CLASSES
        assert MLConfig in LEGACY_CONFIG_CLASSES
        assert TranslationConfig in LEGACY_CONFIG_CLASSES

    def test_all_config_classes_includes_memory(self):
        assert MemoryConfig in ALL_CONFIG_CLASSES

    def test_all_config_classes_superset(self):
        for cls in PRIMARY_CONFIG_CLASSES:
            assert cls in ALL_CONFIG_CLASSES
        for cls in LEGACY_CONFIG_CLASSES:
            assert cls in ALL_CONFIG_CLASSES


# ---------------------------------------------------------------------------
# Cross-cutting tests
# ---------------------------------------------------------------------------


class TestAllConfigs:
    ALL_CONFIGS = [
        DebateConfig,
        AgentConfig,
        StreamingConfig,
        ObservabilityConfig,
        MemoryConfig,
        SupermemoryConfig,
        KnowledgeConfig,
        MLConfig,
        RLMConfig,
        TelemetryConfig,
        PersonaConfig,
        ResilienceConfig,
        EvolutionConfig,
        BillingConfig,
        BroadcastConfig,
        TranslationConfig,
    ]

    @pytest.mark.parametrize("config_cls", ALL_CONFIGS, ids=lambda c: c.__name__)
    def test_default_instantiation(self, config_cls):
        cfg = config_cls()
        assert cfg is not None

    @pytest.mark.parametrize("config_cls", ALL_CONFIGS, ids=lambda c: c.__name__)
    def test_is_dataclass(self, config_cls):
        assert hasattr(config_cls, "__dataclass_fields__")

    @pytest.mark.parametrize("config_cls", ALL_CONFIGS, ids=lambda c: c.__name__)
    def test_has_docstring(self, config_cls):
        assert config_cls.__doc__ is not None
        assert len(config_cls.__doc__) > 10
