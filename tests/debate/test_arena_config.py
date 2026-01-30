"""
Tests for Arena config object decomposition.

Covers all 10 config dataclasses, their defaults, and
backward-compatible merging with individual Arena parameters.
"""

from __future__ import annotations

from dataclasses import fields
from unittest.mock import MagicMock

import pytest

from aragora.debate.arena_config import (
    ALL_CONFIG_CLASSES,
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
)


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
        assert c.enable_ml_delegation is False
        assert c.ml_delegation_weight == 0.3
        assert c.enable_quality_gates is False
        assert c.quality_gate_threshold == 0.6
        assert c.enable_consensus_estimation is False
        assert c.consensus_early_termination_threshold == 0.85

    def test_enabled(self):
        c = MLConfig(enable_ml_delegation=True, ml_delegation_weight=0.7)
        assert c.enable_ml_delegation is True
        assert c.ml_delegation_weight == 0.7


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


class TestAllConfigClasses:
    def test_all_classes_present(self):
        assert len(ALL_CONFIG_CLASSES) == 10

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
