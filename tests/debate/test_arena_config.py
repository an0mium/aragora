"""
Comprehensive tests for Arena configuration module.

Covers:
1. ArenaConfig dataclass - all fields, defaults, validation, sub-config delegation
2. ArenaConfigBuilder - fluent builder pattern, method chaining, build() validation
3. Sub-config dataclasses - HookConfig, TrackingConfig, etc.
4. Primary/Legacy config classes - DebateConfig, AgentConfig, etc.
5. Edge cases - invalid configs, unknown kwargs, conflicting options
6. Backward compatibility - __getattr__/__setattr__ delegation
7. Integration - to_arena_kwargs(), equality, repr
"""

from __future__ import annotations

from dataclasses import fields
from unittest.mock import MagicMock

import pytest

from aragora.config import DEFAULT_ROUNDS
from aragora.debate.arena_config import (
    # Main classes
    ArenaConfig,
    ArenaConfigBuilder,
    # Sub-config classes (new strategy/builder pattern)
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
    SUB_CONFIG_CLASSES,
    # Primary config classes
    DebateConfig,
    AgentConfig,
    StreamingConfig,
    ObservabilityConfig,
    PRIMARY_CONFIG_CLASSES,
    # Legacy config classes
    MemoryConfig,
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
    LEGACY_CONFIG_CLASSES,
    ALL_CONFIG_CLASSES,
)


# ===========================================================================
# Sub-Config Dataclasses (New Pattern)
# ===========================================================================


class TestHookConfig:
    """Tests for HookConfig dataclass."""

    def test_defaults(self):
        c = HookConfig()
        assert c.event_hooks is None
        assert c.hook_manager is None
        assert c.yaml_hooks_dir == "hooks"
        assert c.enable_yaml_hooks is True
        assert c.yaml_hooks_recursive is True
        assert c.enable_hook_handlers is True
        assert c.hook_handler_registry is None

    def test_custom_values(self):
        hooks = {"on_start": lambda: None}
        manager = MagicMock()
        registry = MagicMock()
        c = HookConfig(
            event_hooks=hooks,
            hook_manager=manager,
            yaml_hooks_dir="custom_hooks",
            enable_yaml_hooks=False,
            yaml_hooks_recursive=False,
            enable_hook_handlers=False,
            hook_handler_registry=registry,
        )
        assert c.event_hooks is hooks
        assert c.hook_manager is manager
        assert c.yaml_hooks_dir == "custom_hooks"
        assert c.enable_yaml_hooks is False
        assert c.yaml_hooks_recursive is False
        assert c.enable_hook_handlers is False
        assert c.hook_handler_registry is registry


class TestTrackingConfig:
    """Tests for TrackingConfig dataclass."""

    def test_defaults(self):
        c = TrackingConfig()
        assert c.position_tracker is None
        assert c.position_ledger is None
        assert c.enable_position_ledger is False
        assert c.elo_system is None
        assert c.persona_manager is None
        assert c.dissent_retriever is None
        assert c.consensus_memory is None
        assert c.flip_detector is None
        assert c.calibration_tracker is None
        assert c.continuum_memory is None
        assert c.relationship_tracker is None
        assert c.moment_detector is None
        assert c.tier_analytics_tracker is None

    def test_custom_values(self):
        elo = MagicMock()
        tracker = MagicMock()
        c = TrackingConfig(
            enable_position_ledger=True,
            elo_system=elo,
            relationship_tracker=tracker,
        )
        assert c.enable_position_ledger is True
        assert c.elo_system is elo
        assert c.relationship_tracker is tracker


class TestKnowledgeMoundConfig:
    """Tests for KnowledgeMoundConfig dataclass."""

    def test_defaults(self):
        c = KnowledgeMoundConfig()
        assert c.knowledge_mound is None
        assert c.enable_knowledge_retrieval is True
        assert c.enable_knowledge_ingestion is True
        assert c.enable_knowledge_extraction is False
        assert c.extraction_min_confidence == 0.3
        assert c.enable_auto_revalidation is False
        assert c.revalidation_staleness_threshold == 0.7
        assert c.revalidation_check_interval_seconds == 3600
        assert c.revalidation_scheduler is None
        assert c.enable_belief_guidance is True
        assert c.cross_debate_memory is None
        assert c.enable_cross_debate_memory is True

    def test_custom_values(self):
        km = MagicMock()
        scheduler = MagicMock()
        c = KnowledgeMoundConfig(
            knowledge_mound=km,
            enable_knowledge_extraction=True,
            extraction_min_confidence=0.5,
            enable_auto_revalidation=True,
            revalidation_staleness_threshold=0.8,
            revalidation_scheduler=scheduler,
        )
        assert c.knowledge_mound is km
        assert c.enable_knowledge_extraction is True
        assert c.extraction_min_confidence == 0.5
        assert c.enable_auto_revalidation is True
        assert c.revalidation_staleness_threshold == 0.8
        assert c.revalidation_scheduler is scheduler


class TestMemoryCoordinationConfig:
    """Tests for MemoryCoordinationConfig dataclass."""

    def test_defaults(self):
        c = MemoryCoordinationConfig()
        assert c.enable_coordinated_writes is True
        assert c.memory_coordinator is None
        assert c.coordinator_parallel_writes is False
        assert c.coordinator_rollback_on_failure is True
        assert c.coordinator_min_confidence_for_mound == 0.7

    def test_custom_values(self):
        coordinator = MagicMock()
        c = MemoryCoordinationConfig(
            enable_coordinated_writes=False,
            memory_coordinator=coordinator,
            coordinator_parallel_writes=True,
            coordinator_rollback_on_failure=False,
            coordinator_min_confidence_for_mound=0.9,
        )
        assert c.enable_coordinated_writes is False
        assert c.memory_coordinator is coordinator
        assert c.coordinator_parallel_writes is True
        assert c.coordinator_rollback_on_failure is False
        assert c.coordinator_min_confidence_for_mound == 0.9


class TestPerformanceFeedbackConfig:
    """Tests for PerformanceFeedbackConfig dataclass."""

    def test_defaults(self):
        c = PerformanceFeedbackConfig()
        assert c.enable_performance_feedback is True
        assert c.selection_feedback_loop is None
        assert c.feedback_loop_weight == 0.15
        assert c.feedback_loop_decay == 0.9
        assert c.feedback_loop_min_debates == 3
        assert c.enable_performance_elo is True
        assert c.performance_elo_integrator is None
        assert c.enable_outcome_memory is True
        assert c.outcome_memory_bridge is None
        assert c.outcome_memory_success_threshold == 0.7
        assert c.outcome_memory_usage_threshold == 3
        assert c.enable_trickster_calibration is True
        assert c.trickster_calibrator is None
        assert c.trickster_calibration_min_samples == 20
        assert c.trickster_calibration_interval == 50

    def test_custom_values(self):
        loop = MagicMock()
        c = PerformanceFeedbackConfig(
            enable_performance_feedback=False,
            selection_feedback_loop=loop,
            feedback_loop_weight=0.25,
            trickster_calibration_min_samples=50,
        )
        assert c.enable_performance_feedback is False
        assert c.selection_feedback_loop is loop
        assert c.feedback_loop_weight == 0.25
        assert c.trickster_calibration_min_samples == 50


class TestAuditTrailConfig:
    """Tests for AuditTrailConfig dataclass."""

    def test_defaults(self):
        c = AuditTrailConfig()
        assert c.enable_receipt_generation is False
        assert c.receipt_min_confidence == 0.6
        assert c.receipt_auto_sign is False
        assert c.receipt_store is None
        assert c.enable_provenance is False
        assert c.provenance_manager is None
        assert c.provenance_store is None
        assert c.provenance_auto_persist is True
        assert c.enable_bead_tracking is False
        assert c.bead_store is None
        assert c.bead_min_confidence == 0.5
        assert c.bead_auto_commit is False

    def test_custom_values(self):
        store = MagicMock()
        manager = MagicMock()
        c = AuditTrailConfig(
            enable_receipt_generation=True,
            receipt_min_confidence=0.8,
            receipt_auto_sign=True,
            receipt_store=store,
            enable_provenance=True,
            provenance_manager=manager,
        )
        assert c.enable_receipt_generation is True
        assert c.receipt_min_confidence == 0.8
        assert c.receipt_auto_sign is True
        assert c.receipt_store is store
        assert c.enable_provenance is True
        assert c.provenance_manager is manager


class TestMLIntegrationConfig:
    """Tests for MLIntegrationConfig dataclass."""

    def test_defaults(self):
        c = MLIntegrationConfig()
        assert c.enable_ml_delegation is True
        assert c.ml_delegation_strategy is None
        assert c.ml_delegation_weight == 0.3
        assert c.enable_quality_gates is True
        assert c.quality_gate_threshold == 0.6
        assert c.enable_consensus_estimation is True
        assert c.consensus_early_termination_threshold == 0.85

    def test_custom_values(self):
        strategy = MagicMock()
        c = MLIntegrationConfig(
            enable_ml_delegation=False,
            ml_delegation_strategy=strategy,
            ml_delegation_weight=0.5,
            enable_quality_gates=False,
            quality_gate_threshold=0.8,
        )
        assert c.enable_ml_delegation is False
        assert c.ml_delegation_strategy is strategy
        assert c.ml_delegation_weight == 0.5
        assert c.enable_quality_gates is False
        assert c.quality_gate_threshold == 0.8


class TestRLMCognitiveConfig:
    """Tests for RLMCognitiveConfig dataclass."""

    def test_defaults(self):
        c = RLMCognitiveConfig()
        assert c.use_rlm_limiter is True
        assert c.rlm_limiter is None
        assert c.rlm_compression_threshold == 3000
        assert c.rlm_max_recent_messages == 5
        assert c.rlm_summary_level == "SUMMARY"
        assert c.rlm_compression_round_threshold == 3

    def test_custom_values(self):
        limiter = MagicMock()
        c = RLMCognitiveConfig(
            use_rlm_limiter=False,
            rlm_limiter=limiter,
            rlm_compression_threshold=5000,
            rlm_max_recent_messages=10,
            rlm_summary_level="DETAILED",
            rlm_compression_round_threshold=5,
        )
        assert c.use_rlm_limiter is False
        assert c.rlm_limiter is limiter
        assert c.rlm_compression_threshold == 5000
        assert c.rlm_max_recent_messages == 10
        assert c.rlm_summary_level == "DETAILED"
        assert c.rlm_compression_round_threshold == 5


class TestCheckpointMemoryConfig:
    """Tests for CheckpointMemoryConfig dataclass."""

    def test_defaults(self):
        c = CheckpointMemoryConfig()
        assert c.checkpoint_manager is None
        assert c.enable_checkpointing is True
        assert c.checkpoint_include_memory is True
        assert c.checkpoint_memory_max_entries == 100
        assert c.checkpoint_memory_restore_mode == "replace"

    def test_custom_values(self):
        manager = MagicMock()
        c = CheckpointMemoryConfig(
            checkpoint_manager=manager,
            enable_checkpointing=False,
            checkpoint_include_memory=False,
            checkpoint_memory_max_entries=50,
            checkpoint_memory_restore_mode="merge",
        )
        assert c.checkpoint_manager is manager
        assert c.enable_checkpointing is False
        assert c.checkpoint_include_memory is False
        assert c.checkpoint_memory_max_entries == 50
        assert c.checkpoint_memory_restore_mode == "merge"


class TestCrossPollinationConfig:
    """Tests for CrossPollinationConfig dataclass."""

    def test_defaults(self):
        c = CrossPollinationConfig()
        assert c.enable_performance_router is True
        assert c.performance_router_bridge is None
        assert c.performance_router_latency_weight == 0.3
        assert c.performance_router_quality_weight == 0.4
        assert c.performance_router_consistency_weight == 0.3
        assert c.enable_outcome_complexity is True
        assert c.enable_analytics_selection is True
        assert c.enable_novelty_selection is True
        assert c.enable_relationship_bias is True
        assert c.enable_rlm_selection is True
        assert c.enable_calibration_cost is True

    def test_custom_values(self):
        bridge = MagicMock()
        c = CrossPollinationConfig(
            enable_performance_router=False,
            performance_router_bridge=bridge,
            performance_router_latency_weight=0.5,
            enable_novelty_selection=False,
            novelty_selection_low_penalty=0.2,
        )
        assert c.enable_performance_router is False
        assert c.performance_router_bridge is bridge
        assert c.performance_router_latency_weight == 0.5
        assert c.enable_novelty_selection is False
        assert c.novelty_selection_low_penalty == 0.2


class TestKMBidirectionalConfig:
    """Tests for KMBidirectionalConfig dataclass."""

    def test_defaults(self):
        c = KMBidirectionalConfig()
        assert c.enable_km_bidirectional is True
        assert c.enable_km_continuum_sync is True
        assert c.km_continuum_adapter is None
        assert c.km_continuum_min_confidence == 0.7
        assert c.km_continuum_promotion_threshold == 0.8
        assert c.km_continuum_demotion_threshold == 0.3
        assert c.enable_km_elo_sync is True
        assert c.enable_km_outcome_validation is True
        assert c.enable_km_belief_sync is True
        assert c.km_belief_crux_sensitivity_range == (0.2, 0.8)
        assert c.enable_km_flip_sync is True
        assert c.enable_km_critique_sync is True
        assert c.enable_km_pulse_sync is True
        assert c.enable_km_coordinator is True
        assert c.km_sync_interval_seconds == 300
        assert c.km_parallel_sync is True

    def test_custom_values(self):
        adapter = MagicMock()
        c = KMBidirectionalConfig(
            enable_km_bidirectional=False,
            km_continuum_adapter=adapter,
            km_continuum_min_confidence=0.9,
            km_belief_crux_sensitivity_range=(0.1, 0.9),
            km_sync_interval_seconds=600,
        )
        assert c.enable_km_bidirectional is False
        assert c.km_continuum_adapter is adapter
        assert c.km_continuum_min_confidence == 0.9
        assert c.km_belief_crux_sensitivity_range == (0.1, 0.9)
        assert c.km_sync_interval_seconds == 600


class TestTranslationSubConfig:
    """Tests for TranslationSubConfig dataclass."""

    def test_defaults(self):
        c = TranslationSubConfig()
        assert c.translation_service is None
        assert c.multilingual_manager is None
        assert c.enable_translation is False
        assert c.default_language == "en"
        assert c.target_languages is None
        assert c.auto_detect_language is True
        assert c.translate_conclusions is True
        assert c.translation_cache_ttl_seconds == 3600
        assert c.translation_cache_max_entries == 10000

    def test_custom_values(self):
        service = MagicMock()
        c = TranslationSubConfig(
            translation_service=service,
            enable_translation=True,
            default_language="es",
            target_languages=["en", "fr", "de"],
            translation_cache_ttl_seconds=7200,
        )
        assert c.translation_service is service
        assert c.enable_translation is True
        assert c.default_language == "es"
        assert c.target_languages == ["en", "fr", "de"]
        assert c.translation_cache_ttl_seconds == 7200


class TestSupermemorySubConfig:
    """Tests for SupermemorySubConfig dataclass."""

    def test_defaults(self):
        c = SupermemorySubConfig()
        assert c.enable_supermemory is False
        assert c.supermemory_enable_km_adapter is False
        assert c.supermemory_adapter is None
        assert c.supermemory_inject_on_start is True
        assert c.supermemory_max_context_items == 10
        assert c.supermemory_context_container_tag is None
        assert c.supermemory_sync_on_conclusion is True
        assert c.supermemory_min_confidence_for_sync == 0.7
        assert c.supermemory_outcome_container_tag is None
        assert c.supermemory_enable_privacy_filter is True
        assert c.supermemory_enable_resilience is True

    def test_custom_values(self):
        adapter = MagicMock()
        c = SupermemorySubConfig(
            enable_supermemory=True,
            supermemory_adapter=adapter,
            supermemory_max_context_items=20,
            supermemory_context_container_tag="test-container",
            supermemory_enable_privacy_filter=False,
        )
        assert c.enable_supermemory is True
        assert c.supermemory_adapter is adapter
        assert c.supermemory_max_context_items == 20
        assert c.supermemory_context_container_tag == "test-container"
        assert c.supermemory_enable_privacy_filter is False


class TestSubConfigClasses:
    """Tests for SUB_CONFIG_CLASSES tuple."""

    def test_contains_all_sub_configs(self):
        assert HookConfig in SUB_CONFIG_CLASSES
        assert TrackingConfig in SUB_CONFIG_CLASSES
        assert KnowledgeMoundConfig in SUB_CONFIG_CLASSES
        assert MemoryCoordinationConfig in SUB_CONFIG_CLASSES
        assert PerformanceFeedbackConfig in SUB_CONFIG_CLASSES
        assert AuditTrailConfig in SUB_CONFIG_CLASSES
        assert MLIntegrationConfig in SUB_CONFIG_CLASSES
        assert RLMCognitiveConfig in SUB_CONFIG_CLASSES
        assert CheckpointMemoryConfig in SUB_CONFIG_CLASSES
        assert CrossPollinationConfig in SUB_CONFIG_CLASSES
        assert KMBidirectionalConfig in SUB_CONFIG_CLASSES
        assert TranslationSubConfig in SUB_CONFIG_CLASSES
        assert SupermemorySubConfig in SUB_CONFIG_CLASSES

    def test_sub_config_count(self):
        assert len(SUB_CONFIG_CLASSES) == 14  # Includes BudgetSubConfig

    def test_all_are_dataclasses(self):
        for cls in SUB_CONFIG_CLASSES:
            assert len(fields(cls)) > 0

    def test_all_instantiate_with_defaults(self):
        for cls in SUB_CONFIG_CLASSES:
            instance = cls()
            assert instance is not None


# ===========================================================================
# ArenaConfig Main Class Tests
# ===========================================================================


class TestArenaConfigDefaults:
    """Tests for ArenaConfig default values."""

    def test_identification_defaults(self):
        c = ArenaConfig()
        assert c.loop_id == ""
        assert c.strict_loop_scoping is False

    def test_core_subsystem_defaults(self):
        c = ArenaConfig()
        assert c.memory is None
        assert c.event_emitter is None
        assert c.spectator is None
        assert c.debate_embeddings is None
        assert c.insight_store is None
        assert c.recorder is None
        assert c.circuit_breaker is None
        assert c.evidence_collector is None

    def test_skills_defaults(self):
        c = ArenaConfig()
        assert c.skill_registry is None
        assert c.enable_skills is False

    def test_propulsion_defaults(self):
        c = ArenaConfig()
        assert c.propulsion_engine is None
        assert c.enable_propulsion is False

    def test_agent_config_defaults(self):
        c = ArenaConfig()
        assert c.agent_weights is None

    def test_vertical_defaults(self):
        c = ArenaConfig()
        assert c.vertical is None
        assert c.vertical_persona_manager is None
        assert c.auto_detect_vertical is True

    def test_performance_defaults(self):
        c = ArenaConfig()
        assert c.performance_monitor is None
        assert c.enable_performance_monitor is True
        assert c.enable_telemetry is False

    def test_selection_defaults(self):
        c = ArenaConfig()
        assert c.agent_selector is None
        assert c.use_performance_selection is True

    def test_airlock_defaults(self):
        c = ArenaConfig()
        assert c.use_airlock is False
        assert c.airlock_config is None

    def test_prompt_evolution_defaults(self):
        c = ArenaConfig()
        assert c.prompt_evolver is None
        assert c.enable_prompt_evolution is False

    def test_billing_defaults(self):
        c = ArenaConfig()
        assert c.org_id == ""
        assert c.user_id == ""
        assert c.usage_tracker is None

    def test_broadcast_defaults(self):
        c = ArenaConfig()
        assert c.broadcast_pipeline is None
        assert c.auto_broadcast is False
        assert c.broadcast_min_confidence == 0.8
        assert c.broadcast_platforms == ["rss"]  # Set in post-init

    def test_training_export_defaults(self):
        c = ArenaConfig()
        assert c.training_exporter is None
        assert c.auto_export_training is False
        assert c.training_export_min_confidence == 0.75
        assert c.training_export_path == ""

    def test_evolution_defaults(self):
        c = ArenaConfig()
        assert c.population_manager is None
        assert c.auto_evolve is False
        assert c.breeding_threshold == 0.8

    def test_fork_defaults(self):
        c = ArenaConfig()
        assert c.initial_messages is None
        assert c.trending_topic is None
        assert c.pulse_manager is None
        assert c.auto_fetch_trending is False

    def test_breakpoint_defaults(self):
        c = ArenaConfig()
        assert c.breakpoint_manager is None

    def test_post_debate_workflow_defaults(self):
        c = ArenaConfig()
        assert c.post_debate_workflow is None
        assert c.enable_post_debate_workflow is False
        assert c.post_debate_workflow_threshold == 0.7

    def test_n1_detection_defaults(self):
        c = ArenaConfig()
        assert c.enable_n1_detection is False
        assert c.n1_detection_mode == "warn"
        assert c.n1_detection_threshold == 5


class TestArenaConfigSubConfigs:
    """Tests for ArenaConfig sub-config initialization."""

    def test_sub_configs_created_by_default(self):
        c = ArenaConfig()
        assert isinstance(c.hook_config, HookConfig)
        assert isinstance(c.tracking_config, TrackingConfig)
        assert isinstance(c.knowledge_config, KnowledgeMoundConfig)
        assert isinstance(c.memory_coordination_config, MemoryCoordinationConfig)
        assert isinstance(c.performance_feedback_config, PerformanceFeedbackConfig)
        assert isinstance(c.audit_trail_config, AuditTrailConfig)
        assert isinstance(c.ml_integration_config, MLIntegrationConfig)
        assert isinstance(c.rlm_cognitive_config, RLMCognitiveConfig)
        assert isinstance(c.checkpoint_memory_config, CheckpointMemoryConfig)
        assert isinstance(c.cross_pollination_config, CrossPollinationConfig)
        assert isinstance(c.km_bidirectional_config, KMBidirectionalConfig)
        assert isinstance(c.translation_sub_config, TranslationSubConfig)
        assert isinstance(c.supermemory_sub_config, SupermemorySubConfig)

    def test_explicit_sub_config_passed(self):
        hook_cfg = HookConfig(yaml_hooks_dir="custom")
        c = ArenaConfig(hook_config=hook_cfg)
        assert c.hook_config.yaml_hooks_dir == "custom"

    def test_explicit_sub_config_merged_with_flat_kwargs(self):
        """Flat kwargs should override explicit sub-config values."""
        hook_cfg = HookConfig(yaml_hooks_dir="original", enable_yaml_hooks=True)
        c = ArenaConfig(hook_config=hook_cfg, yaml_hooks_dir="overridden")
        assert c.hook_config.yaml_hooks_dir == "overridden"
        assert c.hook_config.enable_yaml_hooks is True  # Preserved from explicit


class TestArenaConfigBackwardCompatibility:
    """Tests for backward-compatible attribute access via __getattr__/__setattr__."""

    def test_getattr_delegates_to_sub_config(self):
        c = ArenaConfig()
        # These fields are in sub-configs, accessed via __getattr__
        assert c.enable_receipt_generation is False  # AuditTrailConfig
        assert c.use_rlm_limiter is True  # RLMCognitiveConfig
        assert c.enable_ml_delegation is True  # MLIntegrationConfig
        assert c.enable_knowledge_retrieval is True  # KnowledgeMoundConfig

    def test_setattr_delegates_to_sub_config(self):
        c = ArenaConfig()
        # Set via transparent delegation
        c.enable_receipt_generation = True
        assert c.audit_trail_config.enable_receipt_generation is True
        assert c.enable_receipt_generation is True

    def test_getattr_raises_for_unknown(self):
        c = ArenaConfig()
        with pytest.raises(AttributeError, match="no attribute"):
            _ = c.completely_unknown_field

    def test_flat_kwargs_populate_sub_configs(self):
        """Flat kwargs should populate the appropriate sub-config."""
        c = ArenaConfig(
            enable_receipt_generation=True,
            receipt_min_confidence=0.9,
            use_rlm_limiter=False,
            rlm_compression_threshold=5000,
        )
        # Check via sub-config
        assert c.audit_trail_config.enable_receipt_generation is True
        assert c.audit_trail_config.receipt_min_confidence == 0.9
        assert c.rlm_cognitive_config.use_rlm_limiter is False
        assert c.rlm_cognitive_config.rlm_compression_threshold == 5000
        # Check via direct access (backward compat)
        assert c.enable_receipt_generation is True
        assert c.receipt_min_confidence == 0.9
        assert c.use_rlm_limiter is False
        assert c.rlm_compression_threshold == 5000


class TestArenaConfigUnknownKwargs:
    """Tests for handling of unknown kwargs."""

    def test_unknown_kwarg_raises_type_error(self):
        with pytest.raises(TypeError, match="unknown keyword arguments"):
            ArenaConfig(completely_invalid_field=True)

    def test_multiple_unknown_kwargs_listed(self):
        with pytest.raises(TypeError, match="invalid_a") as exc_info:
            ArenaConfig(invalid_a=1, invalid_b=2)
        assert "invalid_b" in str(exc_info.value)

    def test_valid_kwarg_does_not_raise(self):
        # Should not raise
        c = ArenaConfig(loop_id="test", org_id="org-1")
        assert c.loop_id == "test"
        assert c.org_id == "org-1"


class TestArenaConfigEquality:
    """Tests for ArenaConfig equality."""

    def test_equal_configs(self):
        c1 = ArenaConfig(loop_id="test")
        c2 = ArenaConfig(loop_id="test")
        assert c1 == c2

    def test_unequal_configs(self):
        c1 = ArenaConfig(loop_id="test1")
        c2 = ArenaConfig(loop_id="test2")
        assert c1 != c2

    def test_default_configs_equal(self):
        c1 = ArenaConfig()
        c2 = ArenaConfig()
        assert c1 == c2

    def test_not_equal_to_non_config(self):
        c = ArenaConfig()
        assert c != "not a config"
        assert c != 42
        assert c is not None


class TestArenaConfigRepr:
    """Tests for ArenaConfig repr."""

    def test_repr_contains_class_name(self):
        c = ArenaConfig()
        r = repr(c)
        assert "ArenaConfig(" in r

    def test_repr_contains_custom_values(self):
        c = ArenaConfig(loop_id="my-loop", org_id="my-org")
        r = repr(c)
        assert "loop_id='my-loop'" in r
        assert "org_id='my-org'" in r

    def test_repr_omits_default_sub_configs(self):
        c = ArenaConfig()
        r = repr(c)
        # Default sub-configs should be omitted for readability
        # Only non-default values should appear
        assert "HookConfig()" not in r  # Default values omitted


# ===========================================================================
# ArenaConfigBuilder Tests
# ===========================================================================


class TestArenaConfigBuilderBasic:
    """Tests for ArenaConfigBuilder basic functionality."""

    def test_builder_instantiation(self):
        builder = ArenaConfigBuilder()
        assert builder is not None

    def test_builder_from_class_method(self):
        builder = ArenaConfig.builder()
        assert isinstance(builder, ArenaConfigBuilder)

    def test_build_returns_arena_config(self):
        config = ArenaConfig.builder().build()
        assert isinstance(config, ArenaConfig)


class TestArenaConfigBuilderChaining:
    """Tests for ArenaConfigBuilder method chaining."""

    def test_with_identity_returns_self(self):
        builder = ArenaConfigBuilder()
        result = builder.with_identity(loop_id="test")
        assert result is builder

    def test_with_core_returns_self(self):
        builder = ArenaConfigBuilder()
        result = builder.with_core(memory=MagicMock())
        assert result is builder

    def test_with_hooks_returns_self(self):
        builder = ArenaConfigBuilder()
        result = builder.with_hooks(enable_yaml_hooks=True)
        assert result is builder

    def test_with_tracking_returns_self(self):
        builder = ArenaConfigBuilder()
        result = builder.with_tracking(enable_position_ledger=True)
        assert result is builder

    def test_with_knowledge_returns_self(self):
        builder = ArenaConfigBuilder()
        result = builder.with_knowledge(enable_knowledge_retrieval=True)
        assert result is builder

    def test_with_memory_coordination_returns_self(self):
        builder = ArenaConfigBuilder()
        result = builder.with_memory_coordination(enable_coordinated_writes=True)
        assert result is builder

    def test_with_performance_feedback_returns_self(self):
        builder = ArenaConfigBuilder()
        result = builder.with_performance_feedback(enable_performance_feedback=True)
        assert result is builder

    def test_with_audit_trail_returns_self(self):
        builder = ArenaConfigBuilder()
        result = builder.with_audit_trail(enable_receipt_generation=True)
        assert result is builder

    def test_with_ml_returns_self(self):
        builder = ArenaConfigBuilder()
        result = builder.with_ml(enable_ml_delegation=True)
        assert result is builder

    def test_with_rlm_returns_self(self):
        builder = ArenaConfigBuilder()
        result = builder.with_rlm(use_rlm_limiter=True)
        assert result is builder

    def test_with_checkpoint_returns_self(self):
        builder = ArenaConfigBuilder()
        result = builder.with_checkpoint(enable_checkpointing=True)
        assert result is builder

    def test_with_cross_pollination_returns_self(self):
        builder = ArenaConfigBuilder()
        result = builder.with_cross_pollination(enable_performance_router=True)
        assert result is builder

    def test_with_km_bidirectional_returns_self(self):
        builder = ArenaConfigBuilder()
        result = builder.with_km_bidirectional(enable_km_bidirectional=True)
        assert result is builder

    def test_with_translation_returns_self(self):
        builder = ArenaConfigBuilder()
        result = builder.with_translation(enable_translation=True)
        assert result is builder

    def test_with_supermemory_returns_self(self):
        builder = ArenaConfigBuilder()
        result = builder.with_supermemory(enable_supermemory=True)
        assert result is builder


class TestArenaConfigBuilderValues:
    """Tests for ArenaConfigBuilder value accumulation."""

    def test_with_identity_sets_values(self):
        config = (
            ArenaConfig.builder()
            .with_identity(loop_id="test-loop", strict_loop_scoping=True)
            .build()
        )
        assert config.loop_id == "test-loop"
        assert config.strict_loop_scoping is True

    def test_with_hooks_sets_values(self):
        config = (
            ArenaConfig.builder()
            .with_hooks(yaml_hooks_dir="custom", enable_yaml_hooks=False)
            .build()
        )
        assert config.yaml_hooks_dir == "custom"
        assert config.enable_yaml_hooks is False

    def test_with_knowledge_sets_values(self):
        config = (
            ArenaConfig.builder()
            .with_knowledge(enable_knowledge_retrieval=False, extraction_min_confidence=0.6)
            .build()
        )
        assert config.enable_knowledge_retrieval is False
        assert config.extraction_min_confidence == 0.6

    def test_with_audit_trail_sets_values(self):
        config = (
            ArenaConfig.builder()
            .with_audit_trail(enable_receipt_generation=True, receipt_min_confidence=0.9)
            .build()
        )
        assert config.enable_receipt_generation is True
        assert config.receipt_min_confidence == 0.9

    def test_with_ml_sets_values(self):
        config = (
            ArenaConfig.builder()
            .with_ml(enable_ml_delegation=False, ml_delegation_weight=0.7)
            .build()
        )
        assert config.enable_ml_delegation is False
        assert config.ml_delegation_weight == 0.7


class TestArenaConfigBuilderFullChain:
    """Tests for full builder chain usage."""

    def test_full_builder_chain(self):
        config = (
            ArenaConfig.builder()
            .with_identity(loop_id="full-test")
            .with_hooks(enable_yaml_hooks=True, yaml_hooks_dir="hooks")
            .with_tracking(enable_position_ledger=True)
            .with_knowledge(enable_knowledge_retrieval=True, enable_knowledge_extraction=True)
            .with_audit_trail(enable_receipt_generation=True)
            .with_ml(enable_ml_delegation=True, ml_delegation_weight=0.5)
            .with_rlm(rlm_compression_threshold=4000)
            .with_checkpoint(enable_checkpointing=True)
            .build()
        )
        assert config.loop_id == "full-test"
        assert config.enable_yaml_hooks is True
        assert config.enable_position_ledger is True
        assert config.enable_knowledge_retrieval is True
        assert config.enable_knowledge_extraction is True
        assert config.enable_receipt_generation is True
        assert config.enable_ml_delegation is True
        assert config.ml_delegation_weight == 0.5
        assert config.rlm_compression_threshold == 4000
        assert config.enable_checkpointing is True

    def test_builder_later_calls_override_earlier(self):
        config = (
            ArenaConfig.builder()
            .with_ml(ml_delegation_weight=0.3)
            .with_ml(ml_delegation_weight=0.7)  # Should override
            .build()
        )
        assert config.ml_delegation_weight == 0.7


# ===========================================================================
# Primary Config Classes (New Pattern)
# ===========================================================================


class TestDebateConfig:
    """Tests for DebateConfig - protocol settings configuration."""

    def test_defaults(self):
        c = DebateConfig()
        assert c.rounds == DEFAULT_ROUNDS
        assert c.consensus_threshold == 0.6
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

        assert result is protocol
        assert protocol.rounds == 5
        assert protocol.consensus_threshold == 0.85

    def test_all_judge_selection_modes(self):
        """Verify all judge selection modes are valid."""
        modes = ["random", "voted", "last", "elo_ranked", "calibrated", "crux_aware"]
        for mode in modes:
            c = DebateConfig(judge_selection=mode)
            assert c.judge_selection == mode


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
    """Tests for MemoryConfig dataclass."""

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
    """Tests for KnowledgeConfig dataclass."""

    def test_defaults(self):
        c = KnowledgeConfig()
        assert c.knowledge_mound is None
        assert c.auto_create_knowledge_mound is True
        assert c.enable_knowledge_retrieval is True
        assert c.enable_knowledge_ingestion is True
        assert c.enable_knowledge_extraction is False
        assert c.extraction_min_confidence == 0.3
        assert c.enable_belief_guidance is True

    def test_extraction_enabled(self):
        c = KnowledgeConfig(enable_knowledge_extraction=True, extraction_min_confidence=0.5)
        assert c.enable_knowledge_extraction is True
        assert c.extraction_min_confidence == 0.5


class TestMLConfig:
    """Tests for MLConfig dataclass."""

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
    """Tests for RLMConfig dataclass."""

    def test_defaults(self):
        c = RLMConfig()
        assert c.use_rlm_limiter is True
        assert c.rlm_compression_threshold == 3000
        assert c.rlm_max_recent_messages == 5
        assert c.rlm_summary_level == "SUMMARY"
        assert c.rlm_compression_round_threshold == 3


class TestTelemetryConfig:
    """Tests for TelemetryConfig dataclass."""

    def test_defaults(self):
        c = TelemetryConfig()
        assert c.performance_monitor is None
        assert c.enable_performance_monitor is True
        assert c.enable_telemetry is False


class TestPersonaConfig:
    """Tests for PersonaConfig dataclass."""

    def test_defaults(self):
        c = PersonaConfig()
        assert c.persona_manager is None
        assert c.vertical is None
        assert c.auto_detect_vertical is True


class TestResilienceConfig:
    """Tests for ResilienceConfig dataclass."""

    def test_defaults(self):
        c = ResilienceConfig()
        assert c.circuit_breaker is None
        assert c.use_airlock is False
        assert c.airlock_config is None


class TestEvolutionConfig:
    """Tests for EvolutionConfig dataclass."""

    def test_defaults(self):
        c = EvolutionConfig()
        assert c.population_manager is None
        assert c.auto_evolve is False
        assert c.breeding_threshold == 0.8
        assert c.prompt_evolver is None
        assert c.enable_prompt_evolution is False


class TestBillingConfig:
    """Tests for BillingConfig dataclass."""

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
    """Tests for BroadcastConfig dataclass."""

    def test_defaults(self):
        c = BroadcastConfig()
        assert c.broadcast_pipeline is None
        assert c.auto_broadcast is False
        assert c.broadcast_min_confidence == 0.8
        assert c.training_exporter is None
        assert c.auto_export_training is False
        assert c.training_export_min_confidence == 0.75


class TestTranslationConfig:
    """Tests for TranslationConfig dataclass."""

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
    """Tests for ALL_CONFIG_CLASSES tuple."""

    def test_all_classes_present(self):
        assert len(ALL_CONFIG_CLASSES) >= 11

    def test_all_are_dataclasses(self):
        for cls in ALL_CONFIG_CLASSES:
            assert len(fields(cls)) > 0

    def test_all_instantiate_with_defaults(self):
        for cls in ALL_CONFIG_CLASSES:
            instance = cls()
            assert instance is not None


class TestLegacyConfigClasses:
    """Tests for LEGACY_CONFIG_CLASSES tuple."""

    def test_contains_legacy_configs(self):
        assert KnowledgeConfig in LEGACY_CONFIG_CLASSES
        assert MLConfig in LEGACY_CONFIG_CLASSES
        assert RLMConfig in LEGACY_CONFIG_CLASSES
        assert TelemetryConfig in LEGACY_CONFIG_CLASSES
        assert PersonaConfig in LEGACY_CONFIG_CLASSES
        assert ResilienceConfig in LEGACY_CONFIG_CLASSES
        assert EvolutionConfig in LEGACY_CONFIG_CLASSES
        assert BillingConfig in LEGACY_CONFIG_CLASSES
        assert BroadcastConfig in LEGACY_CONFIG_CLASSES
        assert TranslationConfig in LEGACY_CONFIG_CLASSES


# ===========================================================================
# ArenaConfig.to_arena_kwargs() Tests
# ===========================================================================


class TestArenaConfigToKwargs:
    """Tests for ArenaConfig.to_arena_kwargs() conversion."""

    def test_to_arena_kwargs_returns_dict(self):
        config = ArenaConfig()
        kwargs = config.to_arena_kwargs()
        assert isinstance(kwargs, dict)
        assert "memory" in kwargs
        assert "org_id" in kwargs
        assert "enable_ml_delegation" in kwargs

    def test_to_arena_kwargs_includes_custom_values(self):
        config = ArenaConfig(org_id="org-test", user_id="user-test")
        kwargs = config.to_arena_kwargs()
        assert kwargs["org_id"] == "org-test"
        assert kwargs["user_id"] == "user-test"

    def test_to_arena_kwargs_includes_sub_config_values(self):
        config = ArenaConfig(
            enable_receipt_generation=True,
            enable_ml_delegation=False,
            use_rlm_limiter=False,
        )
        kwargs = config.to_arena_kwargs()
        # These are from sub-configs but should appear in kwargs
        assert kwargs["enable_ml_delegation"] is False
        assert kwargs["use_rlm_limiter"] is False

    def test_to_arena_kwargs_has_expected_keys(self):
        config = ArenaConfig()
        kwargs = config.to_arena_kwargs()
        # Check a sample of expected keys
        expected_keys = [
            "memory",
            "event_hooks",
            "event_emitter",
            "spectator",
            "debate_embeddings",
            "loop_id",
            "circuit_breaker",
            "enable_knowledge_retrieval",
            "enable_ml_delegation",
            "use_rlm_limiter",
            "org_id",
            "user_id",
        ]
        for key in expected_keys:
            assert key in kwargs

    def test_arena_config_post_init_sets_platforms(self):
        config = ArenaConfig()
        assert config.broadcast_platforms == ["rss"]

    def test_custom_broadcast_platforms_preserved(self):
        config = ArenaConfig(broadcast_platforms=["slack", "teams"])
        assert config.broadcast_platforms == ["slack", "teams"]


# ===========================================================================
# Edge Cases and Validation
# ===========================================================================


class TestArenaConfigEdgeCases:
    """Tests for edge cases and validation."""

    def test_empty_config(self):
        config = ArenaConfig()
        assert config is not None

    def test_all_sub_configs_at_default(self):
        config = ArenaConfig()
        # Verify all sub-configs are at their defaults
        assert config.hook_config == HookConfig()
        assert config.tracking_config == TrackingConfig()
        assert config.knowledge_config == KnowledgeMoundConfig()
        assert config.memory_coordination_config == MemoryCoordinationConfig()
        assert config.performance_feedback_config == PerformanceFeedbackConfig()
        assert config.audit_trail_config == AuditTrailConfig()
        assert config.ml_integration_config == MLIntegrationConfig()
        assert config.rlm_cognitive_config == RLMCognitiveConfig()
        assert config.checkpoint_memory_config == CheckpointMemoryConfig()
        assert config.cross_pollination_config == CrossPollinationConfig()
        assert config.km_bidirectional_config == KMBidirectionalConfig()
        assert config.translation_sub_config == TranslationSubConfig()
        assert config.supermemory_sub_config == SupermemorySubConfig()

    def test_none_values_accepted(self):
        config = ArenaConfig(
            memory=None,
            event_emitter=None,
            spectator=None,
        )
        assert config.memory is None
        assert config.event_emitter is None
        assert config.spectator is None

    def test_mock_objects_accepted(self):
        mock_memory = MagicMock()
        mock_emitter = MagicMock()
        config = ArenaConfig(
            memory=mock_memory,
            event_emitter=mock_emitter,
        )
        assert config.memory is mock_memory
        assert config.event_emitter is mock_emitter

    def test_float_boundaries(self):
        """Test float fields at boundary values."""
        config = ArenaConfig(
            broadcast_min_confidence=0.0,
            breeding_threshold=1.0,
        )
        assert config.broadcast_min_confidence == 0.0
        assert config.breeding_threshold == 1.0

    def test_negative_threshold_accepted(self):
        """Negative values should be accepted (validation is caller's responsibility)."""
        config = ArenaConfig(breeding_threshold=-0.5)
        assert config.breeding_threshold == -0.5

    def test_large_integer_values(self):
        config = ArenaConfig(
            n1_detection_threshold=1000000,
        )
        assert config.n1_detection_threshold == 1000000


class TestArenaConfigComplexScenarios:
    """Tests for complex configuration scenarios."""

    def test_mixed_flat_and_subconfig_kwargs(self):
        """Mix of top-level, sub-config flat kwargs, and explicit sub-config objects."""
        hook_cfg = HookConfig(yaml_hooks_dir="base")
        config = ArenaConfig(
            loop_id="complex-test",
            hook_config=hook_cfg,
            yaml_hooks_dir="override",  # Should override the explicit sub-config
            enable_receipt_generation=True,  # Audit trail sub-config
            enable_ml_delegation=False,  # ML integration sub-config
        )
        assert config.loop_id == "complex-test"
        assert config.yaml_hooks_dir == "override"
        assert config.enable_receipt_generation is True
        assert config.enable_ml_delegation is False

    def test_builder_with_explicit_subconfig(self):
        """Builder should work with flat kwargs only."""
        config = (
            ArenaConfig.builder()
            .with_identity(loop_id="builder-test")
            .with_audit_trail(enable_receipt_generation=True)
            .with_hooks(yaml_hooks_dir="builder-hooks")
            .build()
        )
        assert config.loop_id == "builder-test"
        assert config.enable_receipt_generation is True
        assert config.yaml_hooks_dir == "builder-hooks"

    def test_copy_config_pattern(self):
        """Test creating a modified copy of a config."""
        original = ArenaConfig(loop_id="original", org_id="org-1")
        # Create a new config with one field changed
        modified = ArenaConfig(
            loop_id="modified",
            org_id=original.org_id,  # Copy from original
        )
        assert original.loop_id == "original"
        assert modified.loop_id == "modified"
        assert modified.org_id == "org-1"


class TestArenaConfigIntegrationReadiness:
    """Tests to ensure ArenaConfig is ready for Arena integration."""

    def test_to_arena_kwargs_complete(self):
        """Verify to_arena_kwargs returns all necessary keys."""
        config = ArenaConfig(
            loop_id="integration-test",
            org_id="test-org",
            enable_ml_delegation=True,
            enable_knowledge_retrieval=True,
        )
        kwargs = config.to_arena_kwargs()

        # Verify key categories are present
        assert "loop_id" in kwargs
        assert "org_id" in kwargs
        assert "enable_ml_delegation" in kwargs
        assert "enable_knowledge_retrieval" in kwargs
        assert "memory" in kwargs
        assert "event_emitter" in kwargs

    def test_subconfig_values_accessible_for_arena(self):
        """Ensure all sub-config values are accessible for Arena initialization."""
        config = ArenaConfig(
            enable_receipt_generation=True,
            enable_ml_delegation=True,
            use_rlm_limiter=True,
        )
        # These should be accessible both directly and via sub-configs
        assert config.enable_receipt_generation is True
        assert config.audit_trail_config.enable_receipt_generation is True
        assert config.enable_ml_delegation is True
        assert config.ml_integration_config.enable_ml_delegation is True
        assert config.use_rlm_limiter is True
        assert config.rlm_cognitive_config.use_rlm_limiter is True
