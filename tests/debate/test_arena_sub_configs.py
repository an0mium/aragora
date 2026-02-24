"""Tests for ArenaConfig sub-configuration dataclasses.

Covers all 17 sub-config dataclasses in arena_sub_configs.py:
HookConfig, TrackingConfig, KnowledgeMoundConfig, MemoryCoordinationConfig,
PerformanceFeedbackConfig, AuditTrailConfig, MLIntegrationConfig,
RLMCognitiveConfig, CheckpointMemoryConfig, CrossPollinationConfig,
KMBidirectionalConfig, TranslationSubConfig, SupermemorySubConfig,
BudgetSubConfig, PowerSamplingConfig, AutoExecutionConfig, UnifiedMemorySubConfig.
"""

from __future__ import annotations

from dataclasses import fields
from unittest.mock import MagicMock

import pytest

from aragora.debate.arena_sub_configs import (
    AuditTrailConfig,
    AutoExecutionConfig,
    BudgetSubConfig,
    CheckpointMemoryConfig,
    CrossPollinationConfig,
    HookConfig,
    KMBidirectionalConfig,
    KnowledgeMoundConfig,
    MLIntegrationConfig,
    MemoryCoordinationConfig,
    PerformanceFeedbackConfig,
    PowerSamplingConfig,
    RLMCognitiveConfig,
    SupermemorySubConfig,
    TrackingConfig,
    TranslationSubConfig,
    UnifiedMemorySubConfig,
)


# ---------------------------------------------------------------------------
# HookConfig
# ---------------------------------------------------------------------------


class TestHookConfig:
    def test_defaults(self):
        cfg = HookConfig()
        assert cfg.event_hooks is None
        assert cfg.hook_manager is None
        assert cfg.yaml_hooks_dir == "hooks"
        assert cfg.enable_yaml_hooks is True
        assert cfg.yaml_hooks_recursive is True
        assert cfg.enable_hook_handlers is True
        assert cfg.hook_handler_registry is None

    def test_custom_values(self):
        hooks = {"on_start": lambda: None}
        cfg = HookConfig(
            event_hooks=hooks,
            yaml_hooks_dir="custom",
            enable_yaml_hooks=False,
        )
        assert cfg.event_hooks is hooks
        assert cfg.yaml_hooks_dir == "custom"
        assert cfg.enable_yaml_hooks is False


# ---------------------------------------------------------------------------
# TrackingConfig
# ---------------------------------------------------------------------------


class TestTrackingConfig:
    def test_defaults(self):
        cfg = TrackingConfig()
        assert cfg.position_tracker is None
        assert cfg.position_ledger is None
        assert cfg.enable_position_ledger is False
        assert cfg.elo_system is None
        assert cfg.persona_manager is None
        assert cfg.dissent_retriever is None
        assert cfg.consensus_memory is None
        assert cfg.flip_detector is None
        assert cfg.calibration_tracker is None
        assert cfg.continuum_memory is None
        assert cfg.relationship_tracker is None
        assert cfg.moment_detector is None
        assert cfg.tier_analytics_tracker is None

    def test_with_elo(self):
        elo = MagicMock()
        cfg = TrackingConfig(elo_system=elo, enable_position_ledger=True)
        assert cfg.elo_system is elo
        assert cfg.enable_position_ledger is True


# ---------------------------------------------------------------------------
# KnowledgeMoundConfig
# ---------------------------------------------------------------------------


class TestKnowledgeMoundConfig:
    def test_defaults(self):
        cfg = KnowledgeMoundConfig()
        assert cfg.knowledge_mound is None
        assert cfg.enable_knowledge_retrieval is True
        assert cfg.enable_knowledge_ingestion is True
        assert cfg.enable_knowledge_extraction is False
        assert cfg.extraction_min_confidence == 0.3
        assert cfg.enable_auto_revalidation is False
        assert cfg.revalidation_staleness_threshold == 0.7
        assert cfg.revalidation_check_interval_seconds == 3600
        assert cfg.enable_belief_guidance is True
        assert cfg.enable_cross_debate_memory is True

    def test_extraction_enabled(self):
        cfg = KnowledgeMoundConfig(
            enable_knowledge_extraction=True,
            extraction_min_confidence=0.5,
        )
        assert cfg.enable_knowledge_extraction is True
        assert cfg.extraction_min_confidence == 0.5


# ---------------------------------------------------------------------------
# MemoryCoordinationConfig
# ---------------------------------------------------------------------------


class TestMemoryCoordinationConfig:
    def test_defaults(self):
        cfg = MemoryCoordinationConfig()
        assert cfg.enable_coordinated_writes is True
        assert cfg.memory_coordinator is None
        assert cfg.coordinator_parallel_writes is False
        assert cfg.coordinator_rollback_on_failure is True
        assert cfg.coordinator_min_confidence_for_mound == 0.7

    def test_parallel_writes(self):
        cfg = MemoryCoordinationConfig(coordinator_parallel_writes=True)
        assert cfg.coordinator_parallel_writes is True


# ---------------------------------------------------------------------------
# PerformanceFeedbackConfig
# ---------------------------------------------------------------------------


class TestPerformanceFeedbackConfig:
    def test_defaults(self):
        cfg = PerformanceFeedbackConfig()
        assert cfg.enable_performance_feedback is True
        assert cfg.feedback_loop_weight == 0.25
        assert cfg.feedback_loop_decay == 0.9
        assert cfg.feedback_loop_min_debates == 2
        assert cfg.enable_performance_elo is True
        assert cfg.enable_outcome_memory is True
        assert cfg.outcome_memory_success_threshold == 0.7
        assert cfg.outcome_memory_usage_threshold == 3
        assert cfg.enable_trickster_calibration is True
        assert cfg.trickster_calibration_min_samples == 20
        assert cfg.trickster_calibration_interval == 50

    def test_domain_configs(self):
        domains = {"healthcare": {"threshold": 0.9}}
        cfg = PerformanceFeedbackConfig(trickster_domain_configs=domains)
        assert cfg.trickster_domain_configs == domains


# ---------------------------------------------------------------------------
# AuditTrailConfig
# ---------------------------------------------------------------------------


class TestAuditTrailConfig:
    def test_defaults(self):
        cfg = AuditTrailConfig()
        assert cfg.enable_receipt_generation is False
        assert cfg.receipt_min_confidence == 0.6
        assert cfg.receipt_auto_sign is False
        assert cfg.enable_provenance is False
        assert cfg.provenance_auto_persist is True
        assert cfg.enable_bead_tracking is False
        assert cfg.bead_min_confidence == 0.5
        assert cfg.bead_auto_commit is False
        assert cfg.enable_compliance_artifacts is False
        assert cfg.compliance_frameworks is None

    def test_receipt_generation(self):
        cfg = AuditTrailConfig(
            enable_receipt_generation=True,
            receipt_auto_sign=True,
            receipt_min_confidence=0.8,
        )
        assert cfg.enable_receipt_generation is True
        assert cfg.receipt_auto_sign is True
        assert cfg.receipt_min_confidence == 0.8


# ---------------------------------------------------------------------------
# MLIntegrationConfig
# ---------------------------------------------------------------------------


class TestMLIntegrationConfig:
    def test_defaults(self):
        cfg = MLIntegrationConfig()
        assert cfg.enable_ml_delegation is True
        assert cfg.ml_delegation_weight == 0.3
        assert cfg.enable_quality_gates is True
        assert cfg.quality_gate_threshold == 0.6
        assert cfg.enable_consensus_estimation is True
        assert cfg.consensus_early_termination_threshold == 0.85
        assert cfg.enable_stability_detection is False
        assert cfg.stability_threshold == 0.85
        assert cfg.stability_min_rounds == 2

    def test_stability_detection(self):
        cfg = MLIntegrationConfig(
            enable_stability_detection=True,
            stability_threshold=0.9,
            stability_min_rounds=3,
        )
        assert cfg.enable_stability_detection is True
        assert cfg.stability_threshold == 0.9


# ---------------------------------------------------------------------------
# RLMCognitiveConfig
# ---------------------------------------------------------------------------


class TestRLMCognitiveConfig:
    def test_defaults(self):
        cfg = RLMCognitiveConfig()
        assert cfg.use_rlm_limiter is True
        assert cfg.rlm_limiter is None
        assert cfg.rlm_compression_threshold == 3000
        assert cfg.rlm_max_recent_messages == 5
        assert cfg.rlm_summary_level == "SUMMARY"
        assert cfg.rlm_compression_round_threshold == 3

    def test_custom_summary(self):
        cfg = RLMCognitiveConfig(rlm_summary_level="ABSTRACT")
        assert cfg.rlm_summary_level == "ABSTRACT"


# ---------------------------------------------------------------------------
# CheckpointMemoryConfig
# ---------------------------------------------------------------------------


class TestCheckpointMemoryConfig:
    def test_defaults(self):
        cfg = CheckpointMemoryConfig()
        assert cfg.checkpoint_manager is None
        assert cfg.enable_checkpointing is True
        assert cfg.checkpoint_include_memory is True
        assert cfg.checkpoint_memory_max_entries == 100
        assert cfg.checkpoint_memory_restore_mode == "replace"

    def test_merge_mode(self):
        cfg = CheckpointMemoryConfig(checkpoint_memory_restore_mode="merge")
        assert cfg.checkpoint_memory_restore_mode == "merge"


# ---------------------------------------------------------------------------
# CrossPollinationConfig
# ---------------------------------------------------------------------------


class TestCrossPollinationConfig:
    def test_defaults(self):
        cfg = CrossPollinationConfig()
        assert cfg.enable_performance_router is True
        assert cfg.enable_outcome_complexity is True
        assert cfg.enable_analytics_selection is True
        assert cfg.enable_novelty_selection is True
        assert cfg.enable_relationship_bias is True
        assert cfg.enable_rlm_selection is True
        assert cfg.enable_calibration_cost is True

    def test_weight_fields(self):
        cfg = CrossPollinationConfig()
        assert cfg.performance_router_latency_weight == 0.3
        assert cfg.performance_router_quality_weight == 0.4
        assert cfg.performance_router_consistency_weight == 0.3
        assert cfg.outcome_complexity_high_success_boost == 0.1
        assert cfg.novelty_selection_low_penalty == 0.15
        assert cfg.relationship_bias_alliance_threshold == 0.7
        assert cfg.rlm_selection_compression_weight == 0.15
        assert cfg.calibration_cost_ece_threshold == 0.1

    def test_bridge_injection(self):
        bridge = MagicMock()
        cfg = CrossPollinationConfig(performance_router_bridge=bridge)
        assert cfg.performance_router_bridge is bridge

    def test_field_count(self):
        """Verify all 7 bridge groups have enable + bridge + params."""
        cfg = CrossPollinationConfig()
        field_names = {f.name for f in fields(cfg)}
        # Each bridge should have at least enable_ and _bridge fields
        for prefix in [
            "performance_router",
            "outcome_complexity",
            "analytics_selection",
            "novelty_selection",
            "relationship_bias",
            "rlm_selection",
            "calibration_cost",
        ]:
            assert f"enable_{prefix}" in field_names
            assert f"{prefix}_bridge" in field_names


# ---------------------------------------------------------------------------
# KMBidirectionalConfig
# ---------------------------------------------------------------------------


class TestKMBidirectionalConfig:
    def test_defaults(self):
        cfg = KMBidirectionalConfig()
        assert cfg.enable_km_bidirectional is True
        assert cfg.enable_km_continuum_sync is True
        assert cfg.enable_km_elo_sync is True
        assert cfg.enable_km_outcome_validation is True
        assert cfg.enable_km_belief_sync is True
        assert cfg.enable_km_flip_sync is True
        assert cfg.enable_km_critique_sync is True
        assert cfg.enable_km_pulse_sync is True
        assert cfg.enable_km_coordinator is True
        assert cfg.km_parallel_sync is True

    def test_confidence_thresholds(self):
        cfg = KMBidirectionalConfig()
        assert cfg.km_continuum_min_confidence == 0.7
        assert cfg.km_continuum_promotion_threshold == 0.8
        assert cfg.km_continuum_demotion_threshold == 0.3
        assert cfg.km_min_confidence_for_reverse == 0.7

    def test_sync_interval(self):
        cfg = KMBidirectionalConfig(km_sync_interval_seconds=600)
        assert cfg.km_sync_interval_seconds == 600


# ---------------------------------------------------------------------------
# TranslationSubConfig
# ---------------------------------------------------------------------------


class TestTranslationSubConfig:
    def test_defaults(self):
        cfg = TranslationSubConfig()
        assert cfg.translation_service is None
        assert cfg.enable_translation is False
        assert cfg.default_language == "en"
        assert cfg.target_languages is None
        assert cfg.auto_detect_language is True
        assert cfg.translate_conclusions is True
        assert cfg.translation_cache_ttl_seconds == 3600
        assert cfg.translation_cache_max_entries == 10000

    def test_multi_language(self):
        cfg = TranslationSubConfig(
            enable_translation=True,
            target_languages=["en", "fr", "de"],
        )
        assert cfg.target_languages == ["en", "fr", "de"]


# ---------------------------------------------------------------------------
# SupermemorySubConfig
# ---------------------------------------------------------------------------


class TestSupermemorySubConfig:
    def test_defaults(self):
        cfg = SupermemorySubConfig()
        assert cfg.enable_supermemory is True
        assert cfg.supermemory_enable_km_adapter is False
        assert cfg.supermemory_adapter is None
        assert cfg.supermemory_inject_on_start is True
        assert cfg.supermemory_max_context_items == 10
        assert cfg.supermemory_sync_on_conclusion is True
        assert cfg.supermemory_min_confidence_for_sync == 0.7
        assert cfg.supermemory_enable_privacy_filter is True
        assert cfg.supermemory_enable_resilience is True

    def test_enabled_with_adapter(self):
        adapter = MagicMock()
        cfg = SupermemorySubConfig(
            enable_supermemory=True,
            supermemory_adapter=adapter,
        )
        assert cfg.supermemory_adapter is adapter


# ---------------------------------------------------------------------------
# BudgetSubConfig
# ---------------------------------------------------------------------------


class TestBudgetSubConfig:
    def test_defaults(self):
        cfg = BudgetSubConfig()
        assert cfg.budget_limit_usd is None
        assert cfg.budget_alert_threshold == 0.75
        assert cfg.budget_hard_stop is False
        assert cfg.budget_downgrade_models is False
        assert cfg.budget_per_round_usd is None

    def test_with_limit(self):
        cfg = BudgetSubConfig(
            budget_limit_usd=5.00,
            budget_hard_stop=True,
            budget_per_round_usd=1.00,
        )
        assert cfg.budget_limit_usd == 5.00
        assert cfg.budget_hard_stop is True
        assert cfg.budget_per_round_usd == 1.00


# ---------------------------------------------------------------------------
# PowerSamplingConfig
# ---------------------------------------------------------------------------


class TestPowerSamplingConfig:
    def test_defaults(self):
        cfg = PowerSamplingConfig()
        assert cfg.enable_power_sampling is False
        assert cfg.n_samples == 8
        assert cfg.alpha == 2.0
        assert cfg.k_diverse == 3
        assert cfg.sampling_temperature == 1.0
        assert cfg.min_quality_threshold == 0.3
        assert cfg.enable_for_critiques is False
        assert cfg.custom_scorer is None
        assert cfg.sample_timeout == 30.0

    def test_enabled(self):
        cfg = PowerSamplingConfig(
            enable_power_sampling=True,
            n_samples=16,
            alpha=3.0,
        )
        assert cfg.n_samples == 16
        assert cfg.alpha == 3.0


# ---------------------------------------------------------------------------
# AutoExecutionConfig
# ---------------------------------------------------------------------------


class TestAutoExecutionConfig:
    def test_defaults(self):
        cfg = AutoExecutionConfig()
        assert cfg.enable_auto_execution is False
        assert cfg.auto_execution_mode == "workflow"
        assert cfg.auto_approval_mode == "risk_based"
        assert cfg.auto_max_risk == "low"

    def test_custom(self):
        cfg = AutoExecutionConfig(
            enable_auto_execution=True,
            auto_execution_mode="hybrid",
            auto_approval_mode="always",
            auto_max_risk="medium",
        )
        assert cfg.auto_execution_mode == "hybrid"
        assert cfg.auto_approval_mode == "always"
        assert cfg.auto_max_risk == "medium"


# ---------------------------------------------------------------------------
# UnifiedMemorySubConfig
# ---------------------------------------------------------------------------


class TestUnifiedMemorySubConfig:
    def test_defaults(self):
        cfg = UnifiedMemorySubConfig()
        assert cfg.enable_unified_memory is False
        assert cfg.enable_retention_gate is False
        assert cfg.query_timeout_seconds == 15.0
        assert cfg.dedup_threshold == 0.95
        assert cfg.default_sources is None
        assert cfg.parallel_queries is True

    def test_enabled(self):
        cfg = UnifiedMemorySubConfig(
            enable_unified_memory=True,
            enable_retention_gate=True,
            default_sources=["continuum", "km"],
        )
        assert cfg.default_sources == ["continuum", "km"]


# ---------------------------------------------------------------------------
# Cross-cutting tests
# ---------------------------------------------------------------------------


class TestAllConfigs:
    """Tests that apply to all sub-config dataclasses."""

    ALL_CONFIGS = [
        HookConfig,
        TrackingConfig,
        KnowledgeMoundConfig,
        MemoryCoordinationConfig,
        PerformanceFeedbackConfig,
        AuditTrailConfig,
        MLIntegrationConfig,
        RLMCognitiveConfig,
        CheckpointMemoryConfig,
        CrossPollinationConfig,
        KMBidirectionalConfig,
        TranslationSubConfig,
        SupermemorySubConfig,
        BudgetSubConfig,
        PowerSamplingConfig,
        AutoExecutionConfig,
        UnifiedMemorySubConfig,
    ]

    @pytest.mark.parametrize("config_cls", ALL_CONFIGS, ids=lambda c: c.__name__)
    def test_default_instantiation(self, config_cls):
        """All configs should be instantiable with no arguments."""
        cfg = config_cls()
        assert cfg is not None

    @pytest.mark.parametrize("config_cls", ALL_CONFIGS, ids=lambda c: c.__name__)
    def test_has_docstring(self, config_cls):
        """All configs should have docstrings."""
        assert config_cls.__doc__ is not None
        assert len(config_cls.__doc__) > 10

    @pytest.mark.parametrize("config_cls", ALL_CONFIGS, ids=lambda c: c.__name__)
    def test_is_dataclass(self, config_cls):
        """All configs should be proper dataclasses."""
        assert hasattr(config_cls, "__dataclass_fields__")
