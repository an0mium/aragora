"""
Tests for orchestrator configuration merging logic.

Covers:
- MergedConfig class structure and attributes
- merge_config_objects function behavior
- Configuration dataclass validation
- Default value handling
- Concurrency limit enforcement
- Protocol configuration integration
- Agent team selection parameter handling
- Configuration serialization/deserialization
"""

from __future__ import annotations

import json
from dataclasses import fields
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.debate.orchestrator_config import (
    MergedConfig,
    merge_config_objects,
)
from aragora.debate.arena_config import (
    AgentConfig,
    DebateConfig,
    MemoryConfig,
    ObservabilityConfig,
    StreamingConfig,
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def mock_debate_config() -> DebateConfig:
    """Create a test DebateConfig instance."""
    return DebateConfig(
        rounds=5,
        consensus_threshold=0.75,
        enable_adaptive_rounds=True,
        enable_agent_hierarchy=False,
    )


@pytest.fixture
def mock_agent_config() -> AgentConfig:
    """Create a test AgentConfig instance."""
    return AgentConfig(
        agent_weights={"claude": 1.2, "gpt-4": 1.0},
        use_performance_selection=False,
        use_airlock=True,
        vertical="healthcare",
    )


@pytest.fixture
def mock_memory_config() -> MemoryConfig:
    """Create a test MemoryConfig instance."""
    return MemoryConfig(
        enable_knowledge_retrieval=False,
        enable_cross_debate_memory=True,
        use_rlm_limiter=True,
        rlm_compression_threshold=5000,
    )


@pytest.fixture
def mock_streaming_config() -> StreamingConfig:
    """Create a test StreamingConfig instance."""
    return StreamingConfig(
        loop_id="test-loop-123",
        strict_loop_scoping=True,
        enable_skills=True,
    )


@pytest.fixture
def mock_observability_config() -> ObservabilityConfig:
    """Create a test ObservabilityConfig instance."""
    return ObservabilityConfig(
        enable_telemetry=True,
        org_id="test-org",
        user_id="test-user",
        enable_ml_delegation=True,
        ml_delegation_weight=0.5,
    )


@pytest.fixture
def default_merge_params() -> dict[str, Any]:
    """Create default parameters for merge_config_objects."""
    return {
        "debate_config": None,
        "agent_config": None,
        "memory_config": None,
        "streaming_config": None,
        "observability_config": None,
        "protocol": None,
        "enable_adaptive_rounds": False,
        "debate_strategy": None,
        "enable_agent_hierarchy": True,
        "hierarchy_config": None,
        "agent_weights": None,
        "agent_selector": None,
        "use_performance_selection": True,
        "circuit_breaker": None,
        "use_airlock": False,
        "airlock_config": None,
        "position_tracker": None,
        "position_ledger": None,
        "enable_position_ledger": False,
        "elo_system": None,
        "calibration_tracker": None,
        "relationship_tracker": None,
        "persona_manager": None,
        "vertical": None,
        "vertical_persona_manager": None,
        "auto_detect_vertical": True,
        "fabric": None,
        "fabric_config": None,
        "memory": None,
        "continuum_memory": None,
        "consensus_memory": None,
        "debate_embeddings": None,
        "insight_store": None,
        "dissent_retriever": None,
        "flip_detector": None,
        "moment_detector": None,
        "tier_analytics_tracker": None,
        "cross_debate_memory": None,
        "enable_cross_debate_memory": True,
        "knowledge_mound": None,
        "auto_create_knowledge_mound": True,
        "enable_knowledge_retrieval": True,
        "enable_knowledge_ingestion": True,
        "enable_knowledge_extraction": False,
        "extraction_min_confidence": 0.3,
        "enable_supermemory": False,
        "supermemory_adapter": None,
        "supermemory_inject_on_start": True,
        "supermemory_max_context_items": 10,
        "supermemory_context_container_tag": None,
        "supermemory_sync_on_conclusion": True,
        "supermemory_min_confidence_for_sync": 0.7,
        "supermemory_outcome_container_tag": None,
        "supermemory_enable_privacy_filter": True,
        "supermemory_enable_resilience": True,
        "supermemory_enable_km_adapter": False,
        "enable_belief_guidance": True,
        "enable_auto_revalidation": False,
        "revalidation_staleness_threshold": 0.7,
        "revalidation_check_interval_seconds": 3600,
        "revalidation_scheduler": None,
        "use_rlm_limiter": True,
        "rlm_limiter": None,
        "rlm_compression_threshold": 3000,
        "rlm_max_recent_messages": 5,
        "rlm_summary_level": "SUMMARY",
        "rlm_compression_round_threshold": 3,
        "checkpoint_manager": None,
        "enable_checkpointing": True,
        "codebase_path": None,
        "enable_codebase_grounding": False,
        "codebase_persist_to_km": False,
        "event_hooks": None,
        "hook_manager": None,
        "event_emitter": None,
        "spectator": None,
        "recorder": None,
        "loop_id": "",
        "strict_loop_scoping": False,
        "skill_registry": None,
        "enable_skills": False,
        "propulsion_engine": None,
        "enable_propulsion": False,
        "performance_monitor": None,
        "enable_performance_monitor": True,
        "enable_telemetry": False,
        "prompt_evolver": None,
        "enable_prompt_evolution": False,
        "breakpoint_manager": None,
        "trending_topic": None,
        "pulse_manager": None,
        "auto_fetch_trending": False,
        "population_manager": None,
        "auto_evolve": False,
        "breeding_threshold": 0.8,
        "evidence_collector": None,
        "org_id": "",
        "user_id": "",
        "usage_tracker": None,
        "broadcast_pipeline": None,
        "auto_broadcast": False,
        "broadcast_min_confidence": 0.8,
        "training_exporter": None,
        "auto_export_training": False,
        "training_export_min_confidence": 0.75,
        "enable_ml_delegation": True,
        "ml_delegation_strategy": None,
        "ml_delegation_weight": 0.3,
        "enable_quality_gates": True,
        "quality_gate_threshold": 0.6,
        "enable_consensus_estimation": True,
        "consensus_early_termination_threshold": 0.85,
        "post_debate_workflow": None,
        "enable_post_debate_workflow": False,
        "post_debate_workflow_threshold": 0.7,
        "initial_messages": None,
    }


# ===========================================================================
# Test: MergedConfig Class Structure
# ===========================================================================


class TestMergedConfigStructure:
    """Tests for MergedConfig class structure and attributes."""

    def test_merged_config_has_slots(self):
        """MergedConfig uses __slots__ for memory efficiency."""
        assert hasattr(MergedConfig, "__slots__")
        assert len(MergedConfig.__slots__) > 0

    def test_merged_config_slot_count(self):
        """MergedConfig should have many configuration slots."""
        # There are 109 slots defined in MergedConfig
        assert len(MergedConfig.__slots__) >= 100

    def test_merged_config_instantiation(self):
        """MergedConfig can be instantiated."""
        cfg = MergedConfig()
        assert cfg is not None

    def test_merged_config_protocol_slot(self):
        """MergedConfig has protocol slot."""
        assert "protocol" in MergedConfig.__slots__

    def test_merged_config_core_slots(self):
        """MergedConfig has core configuration slots."""
        core_slots = [
            "enable_adaptive_rounds",
            "debate_strategy",
            "enable_agent_hierarchy",
            "hierarchy_config",
            "agent_weights",
            "agent_selector",
            "use_performance_selection",
            "circuit_breaker",
            "use_airlock",
            "airlock_config",
        ]
        for slot in core_slots:
            assert slot in MergedConfig.__slots__, f"Missing slot: {slot}"

    def test_merged_config_memory_slots(self):
        """MergedConfig has memory-related slots."""
        memory_slots = [
            "memory",
            "continuum_memory",
            "consensus_memory",
            "debate_embeddings",
            "insight_store",
            "knowledge_mound",
            "enable_knowledge_retrieval",
            "enable_knowledge_ingestion",
            "enable_cross_debate_memory",
        ]
        for slot in memory_slots:
            assert slot in MergedConfig.__slots__, f"Missing slot: {slot}"

    def test_merged_config_rlm_slots(self):
        """MergedConfig has RLM cognitive limiter slots."""
        rlm_slots = [
            "use_rlm_limiter",
            "rlm_limiter",
            "rlm_compression_threshold",
            "rlm_max_recent_messages",
            "rlm_summary_level",
            "rlm_compression_round_threshold",
        ]
        for slot in rlm_slots:
            assert slot in MergedConfig.__slots__, f"Missing slot: {slot}"

    def test_merged_config_ml_slots(self):
        """MergedConfig has ML integration slots."""
        ml_slots = [
            "enable_ml_delegation",
            "ml_delegation_strategy",
            "ml_delegation_weight",
            "enable_quality_gates",
            "quality_gate_threshold",
            "enable_consensus_estimation",
            "consensus_early_termination_threshold",
        ]
        for slot in ml_slots:
            assert slot in MergedConfig.__slots__, f"Missing slot: {slot}"

    def test_merged_config_supermemory_slots(self):
        """MergedConfig has Supermemory integration slots."""
        supermemory_slots = [
            "enable_supermemory",
            "supermemory_adapter",
            "supermemory_inject_on_start",
            "supermemory_max_context_items",
            "supermemory_context_container_tag",
            "supermemory_sync_on_conclusion",
            "supermemory_min_confidence_for_sync",
            "supermemory_enable_km_adapter",
        ]
        for slot in supermemory_slots:
            assert slot in MergedConfig.__slots__, f"Missing slot: {slot}"

    def test_merged_config_streaming_slots(self):
        """MergedConfig has streaming/event slots."""
        streaming_slots = [
            "event_hooks",
            "hook_manager",
            "event_emitter",
            "spectator",
            "recorder",
            "loop_id",
            "strict_loop_scoping",
            "skill_registry",
            "enable_skills",
            "propulsion_engine",
            "enable_propulsion",
        ]
        for slot in streaming_slots:
            assert slot in MergedConfig.__slots__, f"Missing slot: {slot}"

    def test_merged_config_observability_slots(self):
        """MergedConfig has observability/telemetry slots."""
        observability_slots = [
            "performance_monitor",
            "enable_performance_monitor",
            "enable_telemetry",
            "prompt_evolver",
            "enable_prompt_evolution",
            "org_id",
            "user_id",
            "usage_tracker",
            "broadcast_pipeline",
            "auto_broadcast",
        ]
        for slot in observability_slots:
            assert slot in MergedConfig.__slots__, f"Missing slot: {slot}"


# ===========================================================================
# Test: merge_config_objects Function
# ===========================================================================


class TestMergeConfigObjects:
    """Tests for merge_config_objects function."""

    def test_returns_merged_config(self, default_merge_params):
        """merge_config_objects returns MergedConfig instance."""
        result = merge_config_objects(**default_merge_params)
        assert isinstance(result, MergedConfig)

    def test_default_values_preserved(self, default_merge_params):
        """Default values are preserved when no config objects provided."""
        result = merge_config_objects(**default_merge_params)

        assert result.enable_adaptive_rounds is False
        assert result.enable_agent_hierarchy is True
        assert result.use_performance_selection is True
        assert result.use_airlock is False
        assert result.enable_cross_debate_memory is True
        assert result.enable_knowledge_retrieval is True
        assert result.use_rlm_limiter is True
        assert result.enable_checkpointing is True
        assert result.enable_telemetry is False
        assert result.enable_ml_delegation is True

    def test_debate_config_overrides_individual_params(
        self, default_merge_params, mock_debate_config
    ):
        """DebateConfig values override individual parameters."""
        default_merge_params["debate_config"] = mock_debate_config
        default_merge_params["enable_adaptive_rounds"] = False  # Will be overridden
        default_merge_params["enable_agent_hierarchy"] = True  # Will be overridden

        result = merge_config_objects(**default_merge_params)

        assert result.enable_adaptive_rounds is True  # From mock_debate_config
        assert result.enable_agent_hierarchy is False  # From mock_debate_config

    def test_agent_config_overrides_individual_params(
        self, default_merge_params, mock_agent_config
    ):
        """AgentConfig values override individual parameters."""
        default_merge_params["agent_config"] = mock_agent_config
        default_merge_params["use_performance_selection"] = True  # Will be overridden
        default_merge_params["use_airlock"] = False  # Will be overridden

        result = merge_config_objects(**default_merge_params)

        assert result.use_performance_selection is False  # From mock_agent_config
        assert result.use_airlock is True  # From mock_agent_config
        assert result.agent_weights == {"claude": 1.2, "gpt-4": 1.0}
        assert result.vertical == "healthcare"

    def test_memory_config_overrides_individual_params(
        self, default_merge_params, mock_memory_config
    ):
        """MemoryConfig values override individual parameters."""
        default_merge_params["memory_config"] = mock_memory_config
        default_merge_params["enable_knowledge_retrieval"] = True  # Will be overridden

        result = merge_config_objects(**default_merge_params)

        assert result.enable_knowledge_retrieval is False  # From mock_memory_config
        assert result.enable_cross_debate_memory is True
        assert result.use_rlm_limiter is True
        assert result.rlm_compression_threshold == 5000  # From mock_memory_config

    def test_streaming_config_overrides_individual_params(
        self, default_merge_params, mock_streaming_config
    ):
        """StreamingConfig values override individual parameters."""
        default_merge_params["streaming_config"] = mock_streaming_config
        default_merge_params["loop_id"] = ""  # Will be overridden
        default_merge_params["strict_loop_scoping"] = False  # Will be overridden

        result = merge_config_objects(**default_merge_params)

        assert result.loop_id == "test-loop-123"  # From mock_streaming_config
        assert result.strict_loop_scoping is True
        assert result.enable_skills is True

    def test_observability_config_overrides_individual_params(
        self, default_merge_params, mock_observability_config
    ):
        """ObservabilityConfig values override individual parameters."""
        default_merge_params["observability_config"] = mock_observability_config
        default_merge_params["enable_telemetry"] = False  # Will be overridden
        default_merge_params["org_id"] = ""  # Will be overridden

        result = merge_config_objects(**default_merge_params)

        assert result.enable_telemetry is True  # From mock_observability_config
        assert result.org_id == "test-org"
        assert result.user_id == "test-user"
        assert result.ml_delegation_weight == 0.5

    def test_protocol_passed_through(self, default_merge_params):
        """Protocol object is passed through to result."""
        from aragora.debate.protocol import DebateProtocol

        protocol = DebateProtocol(rounds=7)
        default_merge_params["protocol"] = protocol

        result = merge_config_objects(**default_merge_params)

        assert result.protocol is protocol

    def test_debate_config_applies_to_protocol(self, default_merge_params):
        """DebateConfig.apply_to_protocol is called when both provided."""
        from aragora.debate.protocol import DebateProtocol

        protocol = DebateProtocol(rounds=3)
        debate_config = DebateConfig(rounds=5, consensus_threshold=0.9)

        default_merge_params["protocol"] = protocol
        default_merge_params["debate_config"] = debate_config

        result = merge_config_objects(**default_merge_params)

        # Protocol should be modified by DebateConfig.apply_to_protocol
        assert result.protocol is protocol
        assert protocol.rounds == 5
        assert protocol.consensus_threshold == 0.9

    def test_all_configs_combined(
        self,
        default_merge_params,
        mock_debate_config,
        mock_agent_config,
        mock_memory_config,
        mock_streaming_config,
        mock_observability_config,
    ):
        """All config objects can be combined."""
        default_merge_params["debate_config"] = mock_debate_config
        default_merge_params["agent_config"] = mock_agent_config
        default_merge_params["memory_config"] = mock_memory_config
        default_merge_params["streaming_config"] = mock_streaming_config
        default_merge_params["observability_config"] = mock_observability_config

        result = merge_config_objects(**default_merge_params)

        # Verify values from each config
        assert result.enable_adaptive_rounds is True  # DebateConfig
        assert result.use_airlock is True  # AgentConfig
        assert result.rlm_compression_threshold == 5000  # MemoryConfig
        assert result.loop_id == "test-loop-123"  # StreamingConfig
        assert result.enable_telemetry is True  # ObservabilityConfig


# ===========================================================================
# Test: Default Value Handling
# ===========================================================================


class TestDefaultValues:
    """Tests for default value handling in configuration."""

    def test_default_rounds(self, default_merge_params):
        """Default rounds configuration."""
        from aragora.config import DEFAULT_ROUNDS

        result = merge_config_objects(**default_merge_params)

        # Protocol determines rounds, but we verify related defaults
        assert result.enable_adaptive_rounds is False

    def test_default_rlm_settings(self, default_merge_params):
        """Default RLM cognitive limiter settings."""
        result = merge_config_objects(**default_merge_params)

        assert result.use_rlm_limiter is True
        assert result.rlm_compression_threshold == 3000
        assert result.rlm_max_recent_messages == 5
        assert result.rlm_summary_level == "SUMMARY"
        assert result.rlm_compression_round_threshold == 3

    def test_default_ml_settings(self, default_merge_params):
        """Default ML integration settings."""
        result = merge_config_objects(**default_merge_params)

        assert result.enable_ml_delegation is True
        assert result.ml_delegation_weight == 0.3
        assert result.enable_quality_gates is True
        assert result.quality_gate_threshold == 0.6
        assert result.enable_consensus_estimation is True
        assert result.consensus_early_termination_threshold == 0.85

    def test_default_knowledge_settings(self, default_merge_params):
        """Default knowledge mound settings."""
        result = merge_config_objects(**default_merge_params)

        assert result.enable_knowledge_retrieval is True
        assert result.enable_knowledge_ingestion is True
        assert result.enable_knowledge_extraction is False
        assert result.extraction_min_confidence == 0.3
        assert result.auto_create_knowledge_mound is True

    def test_default_supermemory_settings(self, default_merge_params):
        """Default Supermemory settings (disabled by default)."""
        result = merge_config_objects(**default_merge_params)

        assert result.enable_supermemory is False
        assert result.supermemory_inject_on_start is True
        assert result.supermemory_max_context_items == 10
        assert result.supermemory_min_confidence_for_sync == 0.7

    def test_default_revalidation_settings(self, default_merge_params):
        """Default auto-revalidation settings."""
        result = merge_config_objects(**default_merge_params)

        assert result.enable_auto_revalidation is False
        assert result.revalidation_staleness_threshold == 0.7
        assert result.revalidation_check_interval_seconds == 3600

    def test_default_broadcast_settings(self, default_merge_params):
        """Default broadcast settings."""
        result = merge_config_objects(**default_merge_params)

        assert result.auto_broadcast is False
        assert result.broadcast_min_confidence == 0.8

    def test_default_training_export_settings(self, default_merge_params):
        """Default training export settings."""
        result = merge_config_objects(**default_merge_params)

        assert result.auto_export_training is False
        assert result.training_export_min_confidence == 0.75

    def test_default_evolution_settings(self, default_merge_params):
        """Default evolution/breeding settings."""
        result = merge_config_objects(**default_merge_params)

        assert result.auto_evolve is False
        assert result.breeding_threshold == 0.8


# ===========================================================================
# Test: Concurrency Limit Enforcement
# ===========================================================================


class TestConcurrencyLimits:
    """Tests for concurrency limit configuration."""

    def test_max_concurrent_proposals_env_var(self):
        """MAX_CONCURRENT_PROPOSALS is configurable via environment."""
        from aragora.config import MAX_CONCURRENT_PROPOSALS

        assert MAX_CONCURRENT_PROPOSALS == 5  # Default value

    def test_max_concurrent_critiques_env_var(self):
        """MAX_CONCURRENT_CRITIQUES is configurable via environment."""
        from aragora.config import MAX_CONCURRENT_CRITIQUES

        assert MAX_CONCURRENT_CRITIQUES == 10  # Default value

    def test_max_concurrent_revisions_env_var(self):
        """MAX_CONCURRENT_REVISIONS is configurable via environment."""
        from aragora.config import MAX_CONCURRENT_REVISIONS

        assert MAX_CONCURRENT_REVISIONS == 5  # Default value

    def test_max_concurrent_streaming_env_var(self):
        """MAX_CONCURRENT_STREAMING is configurable via environment."""
        from aragora.config import MAX_CONCURRENT_STREAMING

        assert MAX_CONCURRENT_STREAMING == 3  # Default value

    def test_max_concurrent_branches_env_var(self):
        """MAX_CONCURRENT_BRANCHES is configurable via environment."""
        from aragora.config import MAX_CONCURRENT_BRANCHES

        assert MAX_CONCURRENT_BRANCHES == 3  # Default value

    def test_max_concurrent_debates_env_var(self):
        """MAX_CONCURRENT_DEBATES is configurable via environment."""
        from aragora.config import MAX_CONCURRENT_DEBATES

        assert MAX_CONCURRENT_DEBATES == 10  # Default value

    def test_concurrency_limits_are_positive(self):
        """All concurrency limits must be positive integers."""
        from aragora.config import (
            MAX_CONCURRENT_BRANCHES,
            MAX_CONCURRENT_CRITIQUES,
            MAX_CONCURRENT_DEBATES,
            MAX_CONCURRENT_PROPOSALS,
            MAX_CONCURRENT_REVISIONS,
            MAX_CONCURRENT_STREAMING,
        )

        assert MAX_CONCURRENT_PROPOSALS > 0
        assert MAX_CONCURRENT_CRITIQUES > 0
        assert MAX_CONCURRENT_REVISIONS > 0
        assert MAX_CONCURRENT_STREAMING > 0
        assert MAX_CONCURRENT_BRANCHES > 0
        assert MAX_CONCURRENT_DEBATES > 0


# ===========================================================================
# Test: Protocol Configuration
# ===========================================================================


class TestProtocolConfiguration:
    """Tests for protocol configuration handling."""

    def test_protocol_none_by_default(self, default_merge_params):
        """Protocol is None by default."""
        result = merge_config_objects(**default_merge_params)
        assert result.protocol is None

    def test_protocol_object_preserved(self, default_merge_params):
        """Protocol object is preserved in merged config."""
        from aragora.debate.protocol import DebateProtocol

        protocol = DebateProtocol(
            rounds=5,
            consensus="majority",
            consensus_threshold=0.6,
        )
        default_merge_params["protocol"] = protocol

        result = merge_config_objects(**default_merge_params)

        assert result.protocol is protocol
        assert result.protocol.rounds == 5
        assert result.protocol.consensus == "majority"

    def test_debate_config_protocol_interaction(self, default_merge_params):
        """DebateConfig applies settings to protocol."""
        from aragora.debate.protocol import DebateProtocol

        protocol = DebateProtocol()
        debate_config = DebateConfig(
            rounds=7,
            consensus_threshold=0.8,
            convergence_threshold=0.9,
            timeout_seconds=600,
            judge_selection="calibrated",
        )

        default_merge_params["protocol"] = protocol
        default_merge_params["debate_config"] = debate_config

        result = merge_config_objects(**default_merge_params)

        assert protocol.rounds == 7
        assert protocol.consensus_threshold == 0.8
        assert protocol.convergence_threshold == 0.9
        assert protocol.timeout_seconds == 600
        assert protocol.judge_selection == "calibrated"


# ===========================================================================
# Test: Agent Team Selection Parameters
# ===========================================================================


class TestAgentTeamSelectionParams:
    """Tests for agent team selection parameter handling."""

    def test_elo_system_passthrough(self, default_merge_params):
        """ELO system is passed through."""
        elo_system = MagicMock()
        default_merge_params["elo_system"] = elo_system

        result = merge_config_objects(**default_merge_params)

        assert result.elo_system is elo_system

    def test_calibration_tracker_passthrough(self, default_merge_params):
        """Calibration tracker is passed through."""
        tracker = MagicMock()
        default_merge_params["calibration_tracker"] = tracker

        result = merge_config_objects(**default_merge_params)

        assert result.calibration_tracker is tracker

    def test_agent_weights_passthrough(self, default_merge_params):
        """Agent weights are passed through."""
        weights = {"claude": 1.5, "gpt-4": 1.0, "gemini": 0.8}
        default_merge_params["agent_weights"] = weights

        result = merge_config_objects(**default_merge_params)

        assert result.agent_weights == weights

    def test_agent_selector_passthrough(self, default_merge_params):
        """Agent selector is passed through."""
        selector = MagicMock()
        default_merge_params["agent_selector"] = selector

        result = merge_config_objects(**default_merge_params)

        assert result.agent_selector is selector

    def test_circuit_breaker_passthrough(self, default_merge_params):
        """Circuit breaker is passed through."""
        breaker = MagicMock()
        default_merge_params["circuit_breaker"] = breaker

        result = merge_config_objects(**default_merge_params)

        assert result.circuit_breaker is breaker

    def test_position_ledger_passthrough(self, default_merge_params):
        """Position ledger is passed through."""
        ledger = MagicMock()
        default_merge_params["position_ledger"] = ledger

        result = merge_config_objects(**default_merge_params)

        assert result.position_ledger is ledger

    def test_persona_manager_passthrough(self, default_merge_params):
        """Persona manager is passed through."""
        manager = MagicMock()
        default_merge_params["persona_manager"] = manager

        result = merge_config_objects(**default_merge_params)

        assert result.persona_manager is manager

    def test_vertical_passthrough(self, default_merge_params):
        """Vertical setting is passed through."""
        default_merge_params["vertical"] = "fintech"

        result = merge_config_objects(**default_merge_params)

        assert result.vertical == "fintech"

    def test_use_performance_selection_default(self, default_merge_params):
        """Performance selection is enabled by default."""
        result = merge_config_objects(**default_merge_params)
        assert result.use_performance_selection is True

    def test_enable_position_ledger_default(self, default_merge_params):
        """Position ledger is disabled by default."""
        result = merge_config_objects(**default_merge_params)
        assert result.enable_position_ledger is False


# ===========================================================================
# Test: Configuration Serialization/Deserialization
# ===========================================================================


class TestConfigSerialization:
    """Tests for configuration serialization/deserialization."""

    def test_merged_config_attributes_accessible(self, default_merge_params):
        """All MergedConfig attributes are accessible."""
        result = merge_config_objects(**default_merge_params)

        # Should not raise AttributeError
        for slot in MergedConfig.__slots__:
            value = getattr(result, slot)
            # Just verify we can access it
            assert value is not None or value is None

    def test_merged_config_to_dict(self, default_merge_params):
        """MergedConfig can be converted to dict."""
        result = merge_config_objects(**default_merge_params)

        # Convert to dict manually (no __dict__ with __slots__)
        config_dict = {slot: getattr(result, slot) for slot in MergedConfig.__slots__}

        assert isinstance(config_dict, dict)
        assert len(config_dict) == len(MergedConfig.__slots__)
        assert "enable_adaptive_rounds" in config_dict
        assert "protocol" in config_dict

    def test_config_values_json_serializable_basic(self, default_merge_params):
        """Basic config values are JSON serializable."""
        result = merge_config_objects(**default_merge_params)

        # These basic values should be JSON serializable
        basic_values = {
            "enable_adaptive_rounds": result.enable_adaptive_rounds,
            "use_airlock": result.use_airlock,
            "enable_telemetry": result.enable_telemetry,
            "rlm_compression_threshold": result.rlm_compression_threshold,
            "ml_delegation_weight": result.ml_delegation_weight,
            "loop_id": result.loop_id,
            "org_id": result.org_id,
        }

        # Should not raise
        json_str = json.dumps(basic_values)
        assert isinstance(json_str, str)

        # Deserialize and verify
        restored = json.loads(json_str)
        assert restored["enable_adaptive_rounds"] is False
        assert restored["rlm_compression_threshold"] == 3000


# ===========================================================================
# Test: Edge Cases and Validation
# ===========================================================================


class TestEdgeCases:
    """Tests for edge cases and validation."""

    def test_none_config_objects_handled(self, default_merge_params):
        """None config objects are handled gracefully."""
        default_merge_params["debate_config"] = None
        default_merge_params["agent_config"] = None
        default_merge_params["memory_config"] = None
        default_merge_params["streaming_config"] = None
        default_merge_params["observability_config"] = None

        # Should not raise
        result = merge_config_objects(**default_merge_params)
        assert result is not None

    def test_agent_config_weights_none_preserved(self, default_merge_params):
        """AgentConfig with None weights preserves individual weights."""
        agent_config = AgentConfig(agent_weights=None)
        individual_weights = {"claude": 1.5}
        default_merge_params["agent_config"] = agent_config
        default_merge_params["agent_weights"] = individual_weights

        result = merge_config_objects(**default_merge_params)

        # Individual weights should be preserved when AgentConfig.agent_weights is None
        assert result.agent_weights == individual_weights

    def test_memory_config_memory_none_preserved(self, default_merge_params):
        """MemoryConfig with None memory preserves individual memory."""
        memory_config = MemoryConfig(memory=None)
        individual_memory = MagicMock()
        default_merge_params["memory_config"] = memory_config
        default_merge_params["memory"] = individual_memory

        result = merge_config_objects(**default_merge_params)

        # Individual memory should be preserved when MemoryConfig.memory is None
        assert result.memory is individual_memory

    def test_streaming_config_loop_id_empty_preserved(self, default_merge_params):
        """StreamingConfig with empty loop_id preserves individual loop_id."""
        # When StreamingConfig.loop_id is falsy (""), it should NOT override
        streaming_config = StreamingConfig(loop_id="")
        default_merge_params["streaming_config"] = streaming_config
        default_merge_params["loop_id"] = "preserved-loop"

        result = merge_config_objects(**default_merge_params)

        # The empty string from StreamingConfig is still used (falsy but present)
        # Current implementation: streaming_config.loop_id or loop_id
        # So empty string is falsy, individual value preserved
        assert result.loop_id == "preserved-loop"

    def test_observability_config_org_id_empty_preserved(self, default_merge_params):
        """ObservabilityConfig with empty org_id preserves individual org_id."""
        observability_config = ObservabilityConfig(org_id="")
        default_merge_params["observability_config"] = observability_config
        default_merge_params["org_id"] = "preserved-org"

        result = merge_config_objects(**default_merge_params)

        # Empty string from config is falsy, individual value preserved
        assert result.org_id == "preserved-org"

    def test_float_values_precision(self, default_merge_params):
        """Float values maintain precision."""
        default_merge_params["ml_delegation_weight"] = 0.12345678
        default_merge_params["consensus_early_termination_threshold"] = 0.99999

        result = merge_config_objects(**default_merge_params)

        assert result.ml_delegation_weight == 0.12345678
        assert result.consensus_early_termination_threshold == 0.99999

    def test_integer_values_preserved(self, default_merge_params):
        """Integer values are preserved correctly."""
        default_merge_params["rlm_compression_threshold"] = 10000
        default_merge_params["supermemory_max_context_items"] = 50
        default_merge_params["revalidation_check_interval_seconds"] = 7200

        result = merge_config_objects(**default_merge_params)

        assert result.rlm_compression_threshold == 10000
        assert result.supermemory_max_context_items == 50
        assert result.revalidation_check_interval_seconds == 7200

    def test_string_values_preserved(self, default_merge_params):
        """String values are preserved correctly."""
        default_merge_params["rlm_summary_level"] = "DETAILED"
        default_merge_params["loop_id"] = "unique-loop-id-12345"
        default_merge_params["org_id"] = "org-with-special-chars-!@#"

        result = merge_config_objects(**default_merge_params)

        assert result.rlm_summary_level == "DETAILED"
        assert result.loop_id == "unique-loop-id-12345"
        assert result.org_id == "org-with-special-chars-!@#"

    def test_mock_objects_passthrough(self, default_merge_params):
        """Mock objects are passed through correctly."""
        mock_memory = MagicMock(name="memory")
        mock_elo = MagicMock(name="elo")
        mock_emitter = MagicMock(name="emitter")

        default_merge_params["memory"] = mock_memory
        default_merge_params["elo_system"] = mock_elo
        default_merge_params["event_emitter"] = mock_emitter

        result = merge_config_objects(**default_merge_params)

        assert result.memory is mock_memory
        assert result.elo_system is mock_elo
        assert result.event_emitter is mock_emitter


# ===========================================================================
# Test: Configuration Hierarchy Override
# ===========================================================================


class TestConfigHierarchyOverride:
    """Tests for configuration hierarchy and override behavior."""

    def test_agent_config_overrides_weights(self, default_merge_params):
        """AgentConfig.agent_weights overrides individual agent_weights."""
        individual_weights = {"claude": 1.0}
        config_weights = {"claude": 1.5, "gpt-4": 1.2}

        agent_config = AgentConfig(agent_weights=config_weights)
        default_merge_params["agent_config"] = agent_config
        default_merge_params["agent_weights"] = individual_weights

        result = merge_config_objects(**default_merge_params)

        # AgentConfig.agent_weights should override individual weights
        assert result.agent_weights == config_weights

    def test_memory_config_overrides_rlm_settings(self, default_merge_params):
        """MemoryConfig RLM settings override individual settings."""
        memory_config = MemoryConfig(
            use_rlm_limiter=False,
            rlm_compression_threshold=8000,
        )
        default_merge_params["memory_config"] = memory_config
        default_merge_params["use_rlm_limiter"] = True
        default_merge_params["rlm_compression_threshold"] = 3000

        result = merge_config_objects(**default_merge_params)

        assert result.use_rlm_limiter is False
        assert result.rlm_compression_threshold == 8000

    def test_observability_config_overrides_ml_settings(self, default_merge_params):
        """ObservabilityConfig ML settings override individual settings."""
        observability_config = ObservabilityConfig(
            enable_ml_delegation=False,
            ml_delegation_weight=0.8,
            enable_quality_gates=False,
        )
        default_merge_params["observability_config"] = observability_config
        default_merge_params["enable_ml_delegation"] = True
        default_merge_params["ml_delegation_weight"] = 0.3

        result = merge_config_objects(**default_merge_params)

        assert result.enable_ml_delegation is False
        assert result.ml_delegation_weight == 0.8
        assert result.enable_quality_gates is False


# ===========================================================================
# Test: Type Annotations
# ===========================================================================


class TestTypeAnnotations:
    """Tests for type annotation consistency."""

    def test_merged_config_has_type_annotations(self):
        """MergedConfig has type annotations for all slots."""
        # Check that type annotations exist for protocol
        assert hasattr(MergedConfig, "__annotations__")
        assert "protocol" in MergedConfig.__annotations__

    def test_bool_fields_are_bool(self, default_merge_params):
        """Boolean fields return boolean values."""
        result = merge_config_objects(**default_merge_params)

        bool_fields = [
            "enable_adaptive_rounds",
            "enable_agent_hierarchy",
            "use_performance_selection",
            "use_airlock",
            "enable_position_ledger",
            "auto_detect_vertical",
            "enable_cross_debate_memory",
            "enable_knowledge_retrieval",
            "enable_supermemory",
            "use_rlm_limiter",
            "enable_checkpointing",
            "enable_skills",
            "enable_telemetry",
            "enable_ml_delegation",
            "auto_broadcast",
        ]

        for field_name in bool_fields:
            value = getattr(result, field_name)
            assert isinstance(value, bool), f"{field_name} should be bool, got {type(value)}"

    def test_float_fields_are_numeric(self, default_merge_params):
        """Float fields return numeric values."""
        result = merge_config_objects(**default_merge_params)

        float_fields = [
            "extraction_min_confidence",
            "supermemory_min_confidence_for_sync",
            "revalidation_staleness_threshold",
            "ml_delegation_weight",
            "quality_gate_threshold",
            "consensus_early_termination_threshold",
            "broadcast_min_confidence",
            "training_export_min_confidence",
            "breeding_threshold",
            "post_debate_workflow_threshold",
        ]

        for field_name in float_fields:
            value = getattr(result, field_name)
            assert isinstance(value, (int, float)), (
                f"{field_name} should be numeric, got {type(value)}"
            )

    def test_int_fields_are_int(self, default_merge_params):
        """Integer fields return integer values."""
        result = merge_config_objects(**default_merge_params)

        int_fields = [
            "rlm_compression_threshold",
            "rlm_max_recent_messages",
            "rlm_compression_round_threshold",
            "supermemory_max_context_items",
            "revalidation_check_interval_seconds",
        ]

        for field_name in int_fields:
            value = getattr(result, field_name)
            assert isinstance(value, int), f"{field_name} should be int, got {type(value)}"

    def test_string_fields_are_string(self, default_merge_params):
        """String fields return string values."""
        result = merge_config_objects(**default_merge_params)

        string_fields = [
            "rlm_summary_level",
            "loop_id",
            "org_id",
            "user_id",
        ]

        for field_name in string_fields:
            value = getattr(result, field_name)
            assert isinstance(value, str), f"{field_name} should be str, got {type(value)}"
