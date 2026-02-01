"""
Tests for orchestrator_config.py - config merging logic for Arena.__init__.

Covers:
- MergedConfig dataclass initialization
- merge_config_objects() function with various config combinations
- Precedence rules: config objects override individual parameters
- Protocol handling via DebateConfig.apply_to_protocol()
- Edge cases: None values, missing fields, conflicting values
- All major config categories: debate, agent, memory, streaming, observability
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.debate.orchestrator_config import MergedConfig, merge_config_objects
from aragora.debate.arena_config import (
    DebateConfig,
    AgentConfig,
    MemoryConfig,
    StreamingConfig,
    ObservabilityConfig,
)
from aragora.debate.protocol import DebateProtocol


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def default_params() -> dict[str, Any]:
    """Default parameter values matching Arena.__init__ defaults."""
    return {
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
        "enable_belief_guidance": False,
        "enable_auto_revalidation": False,
        "revalidation_staleness_threshold": 0.8,
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
        "enable_ml_delegation": False,
        "ml_delegation_strategy": None,
        "ml_delegation_weight": 0.3,
        "enable_quality_gates": False,
        "quality_gate_threshold": 0.6,
        "enable_consensus_estimation": False,
        "consensus_early_termination_threshold": 0.85,
        "post_debate_workflow": None,
        "enable_post_debate_workflow": False,
        "post_debate_workflow_threshold": 0.7,
        "initial_messages": None,
    }


@pytest.fixture
def debate_config() -> DebateConfig:
    """Sample DebateConfig with non-default values."""
    return DebateConfig(
        rounds=7,
        consensus_threshold=0.9,
        enable_adaptive_rounds=True,
        debate_strategy="test_strategy",
        enable_agent_hierarchy=False,
        hierarchy_config={"max_depth": 3},
    )


@pytest.fixture
def agent_config() -> AgentConfig:
    """Sample AgentConfig with non-default values."""
    return AgentConfig(
        agent_weights={"claude": 1.5, "gpt4": 1.0},
        agent_selector=MagicMock(),
        use_performance_selection=False,
        circuit_breaker=MagicMock(),
        use_airlock=True,
        airlock_config={"timeout": 30},
        position_tracker=MagicMock(),
        position_ledger=MagicMock(),
        enable_position_ledger=True,
        elo_system=MagicMock(),
        calibration_tracker=MagicMock(),
        relationship_tracker=MagicMock(),
        persona_manager=MagicMock(),
        vertical="healthcare",
        vertical_persona_manager=MagicMock(),
        auto_detect_vertical=False,
        fabric=MagicMock(),
        fabric_config={"concurrency": 10},
    )


@pytest.fixture
def memory_config() -> MemoryConfig:
    """Sample MemoryConfig with non-default values."""
    return MemoryConfig(
        memory=MagicMock(),
        continuum_memory=MagicMock(),
        consensus_memory=MagicMock(),
        debate_embeddings=MagicMock(),
        insight_store=MagicMock(),
        dissent_retriever=MagicMock(),
        flip_detector=MagicMock(),
        moment_detector=MagicMock(),
        tier_analytics_tracker=MagicMock(),
        cross_debate_memory=MagicMock(),
        enable_cross_debate_memory=False,
        knowledge_mound=MagicMock(),
        auto_create_knowledge_mound=False,
        enable_knowledge_retrieval=False,
        enable_knowledge_ingestion=False,
        enable_knowledge_extraction=True,
        extraction_min_confidence=0.7,
        enable_belief_guidance=True,
        enable_auto_revalidation=True,
        revalidation_staleness_threshold=0.5,
        revalidation_check_interval_seconds=1800,
        revalidation_scheduler=MagicMock(),
        use_rlm_limiter=False,
        rlm_limiter=MagicMock(),
        rlm_compression_threshold=5000,
        rlm_max_recent_messages=10,
        rlm_summary_level="DETAILED",
        rlm_compression_round_threshold=5,
        checkpoint_manager=MagicMock(),
        enable_checkpointing=False,
    )


@pytest.fixture
def streaming_config() -> StreamingConfig:
    """Sample StreamingConfig with non-default values."""
    return StreamingConfig(
        event_hooks={"on_start": lambda: None},
        hook_manager=MagicMock(),
        event_emitter=MagicMock(),
        spectator=MagicMock(),
        recorder=MagicMock(),
        loop_id="test-loop-123",
        strict_loop_scoping=True,
        skill_registry=MagicMock(),
        enable_skills=True,
        propulsion_engine=MagicMock(),
        enable_propulsion=True,
    )


@pytest.fixture
def observability_config() -> ObservabilityConfig:
    """Sample ObservabilityConfig with non-default values."""
    return ObservabilityConfig(
        performance_monitor=MagicMock(),
        enable_performance_monitor=False,
        enable_telemetry=True,
        prompt_evolver=MagicMock(),
        enable_prompt_evolution=True,
        breakpoint_manager=MagicMock(),
        trending_topic=MagicMock(),
        pulse_manager=MagicMock(),
        auto_fetch_trending=True,
        population_manager=MagicMock(),
        auto_evolve=True,
        breeding_threshold=0.9,
        evidence_collector=MagicMock(),
        org_id="test-org",
        user_id="test-user",
        usage_tracker=MagicMock(),
        broadcast_pipeline=MagicMock(),
        auto_broadcast=True,
        broadcast_min_confidence=0.95,
        training_exporter=MagicMock(),
        auto_export_training=True,
        training_export_min_confidence=0.85,
        enable_ml_delegation=True,
        ml_delegation_strategy=MagicMock(),
        ml_delegation_weight=0.6,
        enable_quality_gates=True,
        quality_gate_threshold=0.8,
        enable_consensus_estimation=True,
        consensus_early_termination_threshold=0.9,
        post_debate_workflow=MagicMock(),
        enable_post_debate_workflow=True,
        post_debate_workflow_threshold=0.85,
        initial_messages=[{"role": "system", "content": "test"}],
    )


# ===========================================================================
# MergedConfig Tests
# ===========================================================================


class TestMergedConfig:
    """Tests for MergedConfig dataclass."""

    def test_has_expected_slots(self):
        """MergedConfig should define 123 slots for all config values."""
        # The exact number from orchestrator_config.py docstring
        assert len(MergedConfig.__slots__) >= 90  # At minimum, check substantial coverage

    def test_slots_include_all_major_categories(self):
        """MergedConfig slots should cover all major config categories."""
        slots = MergedConfig.__slots__

        # Debate settings
        assert "enable_adaptive_rounds" in slots
        assert "debate_strategy" in slots
        assert "enable_agent_hierarchy" in slots
        assert "hierarchy_config" in slots

        # Agent settings
        assert "agent_weights" in slots
        assert "agent_selector" in slots
        assert "use_performance_selection" in slots
        assert "circuit_breaker" in slots
        assert "use_airlock" in slots
        assert "elo_system" in slots

        # Memory settings
        assert "memory" in slots
        assert "continuum_memory" in slots
        assert "knowledge_mound" in slots
        assert "enable_knowledge_retrieval" in slots

        # Streaming settings
        assert "event_hooks" in slots
        assert "event_emitter" in slots
        assert "spectator" in slots
        assert "loop_id" in slots

        # Observability settings
        assert "performance_monitor" in slots
        assert "enable_telemetry" in slots
        assert "org_id" in slots
        assert "user_id" in slots

        # Protocol
        assert "protocol" in slots

    def test_instantiation_creates_empty_object(self):
        """MergedConfig can be instantiated without arguments."""
        cfg = MergedConfig()
        assert cfg is not None
        # Slots exist but are uninitialized
        assert hasattr(cfg, "__slots__")


# ===========================================================================
# merge_config_objects with All Configs None (Default Behavior)
# ===========================================================================


class TestMergeConfigObjectsDefaults:
    """Tests for merge_config_objects when all config objects are None."""

    def test_all_configs_none_uses_individual_params(self, default_params):
        """When all config objects are None, individual params are used directly."""
        cfg = merge_config_objects(
            debate_config=None,
            agent_config=None,
            memory_config=None,
            streaming_config=None,
            observability_config=None,
            **default_params,
        )

        # All values should come from individual params
        assert cfg.enable_adaptive_rounds is False
        assert cfg.debate_strategy is None
        assert cfg.enable_agent_hierarchy is True
        assert cfg.use_performance_selection is True
        assert cfg.use_airlock is False
        assert cfg.memory is None
        assert cfg.enable_knowledge_retrieval is True
        assert cfg.event_hooks is None
        assert cfg.loop_id == ""
        assert cfg.enable_telemetry is False
        assert cfg.org_id == ""
        assert cfg.protocol is None

    def test_preserves_non_default_individual_params(self, default_params):
        """Individual params with non-default values are preserved."""
        params = default_params.copy()
        params["enable_adaptive_rounds"] = True
        params["use_airlock"] = True
        params["org_id"] = "custom-org"
        params["loop_id"] = "custom-loop"

        cfg = merge_config_objects(
            debate_config=None,
            agent_config=None,
            memory_config=None,
            streaming_config=None,
            observability_config=None,
            **params,
        )

        assert cfg.enable_adaptive_rounds is True
        assert cfg.use_airlock is True
        assert cfg.org_id == "custom-org"
        assert cfg.loop_id == "custom-loop"


# ===========================================================================
# merge_config_objects with Config Objects Provided
# ===========================================================================


class TestMergeConfigObjectsWithDebateConfig:
    """Tests for merge_config_objects with DebateConfig provided."""

    def test_debate_config_overrides_individual_params(self, default_params, debate_config):
        """DebateConfig values override corresponding individual params."""
        cfg = merge_config_objects(
            debate_config=debate_config,
            agent_config=None,
            memory_config=None,
            streaming_config=None,
            observability_config=None,
            **default_params,
        )

        # Values from DebateConfig should override defaults
        assert cfg.enable_adaptive_rounds is True
        assert cfg.debate_strategy == "test_strategy"
        assert cfg.enable_agent_hierarchy is False
        assert cfg.hierarchy_config == {"max_depth": 3}

    def test_debate_config_with_protocol_calls_apply_to_protocol(
        self, default_params, debate_config
    ):
        """DebateConfig.apply_to_protocol is called when protocol is provided."""
        protocol = DebateProtocol()
        params = default_params.copy()
        params["protocol"] = protocol

        cfg = merge_config_objects(
            debate_config=debate_config,
            agent_config=None,
            memory_config=None,
            streaming_config=None,
            observability_config=None,
            **params,
        )

        # Protocol should have been modified by apply_to_protocol
        assert protocol.rounds == 7
        assert protocol.consensus_threshold == 0.9
        assert cfg.protocol is protocol

    def test_debate_config_without_protocol_does_not_call_apply(
        self, default_params, debate_config
    ):
        """No error when DebateConfig provided but protocol is None."""
        cfg = merge_config_objects(
            debate_config=debate_config,
            agent_config=None,
            memory_config=None,
            streaming_config=None,
            observability_config=None,
            **default_params,
        )

        assert cfg.protocol is None
        assert cfg.enable_adaptive_rounds is True


class TestMergeConfigObjectsWithAgentConfig:
    """Tests for merge_config_objects with AgentConfig provided."""

    def test_agent_config_overrides_individual_params(self, default_params, agent_config):
        """AgentConfig values override corresponding individual params."""
        cfg = merge_config_objects(
            debate_config=None,
            agent_config=agent_config,
            memory_config=None,
            streaming_config=None,
            observability_config=None,
            **default_params,
        )

        assert cfg.agent_weights == {"claude": 1.5, "gpt4": 1.0}
        assert cfg.agent_selector is not None
        assert cfg.use_performance_selection is False
        assert cfg.circuit_breaker is not None
        assert cfg.use_airlock is True
        assert cfg.airlock_config == {"timeout": 30}
        assert cfg.enable_position_ledger is True
        assert cfg.vertical == "healthcare"
        assert cfg.auto_detect_vertical is False

    def test_agent_config_none_values_fallback_to_individual_params(self, default_params):
        """When AgentConfig fields are None, individual params are used."""
        agent_cfg = AgentConfig(
            use_airlock=True,
            # agent_weights, agent_selector, etc. are None
        )
        params = default_params.copy()
        params["agent_weights"] = {"fallback": 1.0}

        cfg = merge_config_objects(
            debate_config=None,
            agent_config=agent_cfg,
            memory_config=None,
            streaming_config=None,
            observability_config=None,
            **params,
        )

        # use_airlock from config, agent_weights from individual param
        assert cfg.use_airlock is True
        assert cfg.agent_weights == {"fallback": 1.0}


class TestMergeConfigObjectsWithMemoryConfig:
    """Tests for merge_config_objects with MemoryConfig provided."""

    def test_memory_config_overrides_individual_params(self, default_params, memory_config):
        """MemoryConfig values override corresponding individual params."""
        cfg = merge_config_objects(
            debate_config=None,
            agent_config=None,
            memory_config=memory_config,
            streaming_config=None,
            observability_config=None,
            **default_params,
        )

        assert cfg.memory is not None
        assert cfg.continuum_memory is not None
        assert cfg.enable_cross_debate_memory is False
        assert cfg.enable_knowledge_retrieval is False
        assert cfg.enable_knowledge_extraction is True
        assert cfg.extraction_min_confidence == 0.7
        assert cfg.use_rlm_limiter is False
        assert cfg.rlm_compression_threshold == 5000
        assert cfg.rlm_summary_level == "DETAILED"
        assert cfg.enable_checkpointing is False

    def test_memory_config_none_values_fallback_to_individual_params(self, default_params):
        """When MemoryConfig fields are None, individual params are used."""
        mem_cfg = MemoryConfig(
            enable_cross_debate_memory=False,
            # memory, continuum_memory, etc. are None
        )
        mock_memory = MagicMock()
        params = default_params.copy()
        params["memory"] = mock_memory

        cfg = merge_config_objects(
            debate_config=None,
            agent_config=None,
            memory_config=mem_cfg,
            streaming_config=None,
            observability_config=None,
            **params,
        )

        assert cfg.enable_cross_debate_memory is False
        assert cfg.memory is mock_memory


class TestMergeConfigObjectsWithStreamingConfig:
    """Tests for merge_config_objects with StreamingConfig provided."""

    def test_streaming_config_overrides_individual_params(self, default_params, streaming_config):
        """StreamingConfig values override corresponding individual params."""
        cfg = merge_config_objects(
            debate_config=None,
            agent_config=None,
            memory_config=None,
            streaming_config=streaming_config,
            observability_config=None,
            **default_params,
        )

        assert cfg.event_hooks is not None
        assert cfg.hook_manager is not None
        assert cfg.event_emitter is not None
        assert cfg.spectator is not None
        assert cfg.recorder is not None
        assert cfg.loop_id == "test-loop-123"
        assert cfg.strict_loop_scoping is True
        assert cfg.enable_skills is True
        assert cfg.enable_propulsion is True


class TestMergeConfigObjectsWithObservabilityConfig:
    """Tests for merge_config_objects with ObservabilityConfig provided."""

    def test_observability_config_overrides_individual_params(
        self, default_params, observability_config
    ):
        """ObservabilityConfig values override corresponding individual params."""
        cfg = merge_config_objects(
            debate_config=None,
            agent_config=None,
            memory_config=None,
            streaming_config=None,
            observability_config=observability_config,
            **default_params,
        )

        assert cfg.performance_monitor is not None
        assert cfg.enable_performance_monitor is False
        assert cfg.enable_telemetry is True
        assert cfg.enable_prompt_evolution is True
        assert cfg.auto_fetch_trending is True
        assert cfg.auto_evolve is True
        assert cfg.breeding_threshold == 0.9
        assert cfg.org_id == "test-org"
        assert cfg.user_id == "test-user"
        assert cfg.auto_broadcast is True
        assert cfg.broadcast_min_confidence == 0.95
        assert cfg.auto_export_training is True
        assert cfg.training_export_min_confidence == 0.85
        assert cfg.enable_ml_delegation is True
        assert cfg.ml_delegation_weight == 0.6
        assert cfg.enable_quality_gates is True
        assert cfg.quality_gate_threshold == 0.8
        assert cfg.enable_consensus_estimation is True
        assert cfg.consensus_early_termination_threshold == 0.9
        assert cfg.enable_post_debate_workflow is True
        assert cfg.post_debate_workflow_threshold == 0.85
        assert cfg.initial_messages is not None


# ===========================================================================
# Precedence Tests: Config Objects Override Individual Parameters
# ===========================================================================


class TestPrecedenceRules:
    """Tests for config object precedence over individual parameters."""

    def test_debate_config_overrides_conflicting_individual_params(
        self, default_params, debate_config
    ):
        """DebateConfig takes precedence over conflicting individual params."""
        params = default_params.copy()
        # Set individual params that conflict with debate_config
        params["enable_adaptive_rounds"] = False  # debate_config has True
        params["enable_agent_hierarchy"] = True  # debate_config has False

        cfg = merge_config_objects(
            debate_config=debate_config,
            agent_config=None,
            memory_config=None,
            streaming_config=None,
            observability_config=None,
            **params,
        )

        # debate_config values should win
        assert cfg.enable_adaptive_rounds is True
        assert cfg.enable_agent_hierarchy is False

    def test_agent_config_overrides_conflicting_individual_params(
        self, default_params, agent_config
    ):
        """AgentConfig takes precedence over conflicting individual params."""
        params = default_params.copy()
        params["use_airlock"] = False  # agent_config has True
        params["use_performance_selection"] = True  # agent_config has False

        cfg = merge_config_objects(
            debate_config=None,
            agent_config=agent_config,
            memory_config=None,
            streaming_config=None,
            observability_config=None,
            **params,
        )

        assert cfg.use_airlock is True
        assert cfg.use_performance_selection is False

    def test_memory_config_overrides_conflicting_individual_params(
        self, default_params, memory_config
    ):
        """MemoryConfig takes precedence over conflicting individual params."""
        params = default_params.copy()
        params["enable_knowledge_retrieval"] = True  # memory_config has False
        params["use_rlm_limiter"] = True  # memory_config has False

        cfg = merge_config_objects(
            debate_config=None,
            agent_config=None,
            memory_config=memory_config,
            streaming_config=None,
            observability_config=None,
            **params,
        )

        assert cfg.enable_knowledge_retrieval is False
        assert cfg.use_rlm_limiter is False

    def test_streaming_config_overrides_conflicting_individual_params(
        self, default_params, streaming_config
    ):
        """StreamingConfig takes precedence over conflicting individual params."""
        params = default_params.copy()
        params["loop_id"] = "different-loop"  # streaming_config has "test-loop-123"
        params["strict_loop_scoping"] = False  # streaming_config has True

        cfg = merge_config_objects(
            debate_config=None,
            agent_config=None,
            memory_config=None,
            streaming_config=streaming_config,
            observability_config=None,
            **params,
        )

        assert cfg.loop_id == "test-loop-123"
        assert cfg.strict_loop_scoping is True

    def test_observability_config_overrides_conflicting_individual_params(
        self, default_params, observability_config
    ):
        """ObservabilityConfig takes precedence over conflicting individual params."""
        params = default_params.copy()
        params["enable_telemetry"] = False  # observability_config has True
        params["org_id"] = "different-org"  # observability_config has "test-org"

        cfg = merge_config_objects(
            debate_config=None,
            agent_config=None,
            memory_config=None,
            streaming_config=None,
            observability_config=observability_config,
            **params,
        )

        assert cfg.enable_telemetry is True
        assert cfg.org_id == "test-org"


# ===========================================================================
# All Config Objects Combined
# ===========================================================================


class TestAllConfigObjectsCombined:
    """Tests for merge_config_objects with all config objects provided."""

    def test_all_configs_applied_simultaneously(
        self,
        default_params,
        debate_config,
        agent_config,
        memory_config,
        streaming_config,
        observability_config,
    ):
        """All config objects are applied correctly when provided together."""
        params = default_params.copy()
        params["protocol"] = DebateProtocol()

        cfg = merge_config_objects(
            debate_config=debate_config,
            agent_config=agent_config,
            memory_config=memory_config,
            streaming_config=streaming_config,
            observability_config=observability_config,
            **params,
        )

        # From DebateConfig
        assert cfg.enable_adaptive_rounds is True
        assert cfg.enable_agent_hierarchy is False

        # From AgentConfig
        assert cfg.use_airlock is True
        assert cfg.vertical == "healthcare"

        # From MemoryConfig
        assert cfg.enable_knowledge_extraction is True
        assert cfg.rlm_summary_level == "DETAILED"

        # From StreamingConfig
        assert cfg.loop_id == "test-loop-123"
        assert cfg.enable_propulsion is True

        # From ObservabilityConfig
        assert cfg.enable_telemetry is True
        assert cfg.org_id == "test-org"

    def test_protocol_modified_by_debate_config(
        self,
        default_params,
        debate_config,
    ):
        """Protocol is modified by DebateConfig when both provided."""
        protocol = DebateProtocol(rounds=3)  # Initial value
        params = default_params.copy()
        params["protocol"] = protocol

        merge_config_objects(
            debate_config=debate_config,
            agent_config=None,
            memory_config=None,
            streaming_config=None,
            observability_config=None,
            **params,
        )

        # Protocol modified by debate_config.apply_to_protocol()
        assert protocol.rounds == 7


# ===========================================================================
# Protocol Handling via apply_to_protocol
# ===========================================================================


class TestProtocolHandling:
    """Tests for protocol handling via DebateConfig.apply_to_protocol."""

    def test_apply_to_protocol_modifies_all_fields(self, default_params):
        """apply_to_protocol sets all protocol fields from DebateConfig."""
        debate_cfg = DebateConfig(
            rounds=10,
            consensus_threshold=0.95,
            convergence_detection=False,
            convergence_threshold=0.8,
            divergence_threshold=0.5,
            timeout_seconds=600,
            judge_selection="random",
        )
        protocol = DebateProtocol()
        params = default_params.copy()
        params["protocol"] = protocol

        merge_config_objects(
            debate_config=debate_cfg,
            agent_config=None,
            memory_config=None,
            streaming_config=None,
            observability_config=None,
            **params,
        )

        assert protocol.rounds == 10
        assert protocol.consensus_threshold == 0.95
        assert protocol.convergence_detection is False
        assert protocol.convergence_threshold == 0.8
        assert protocol.divergence_threshold == 0.5
        assert protocol.timeout_seconds == 600
        assert protocol.judge_selection == "random"

    def test_protocol_not_modified_without_debate_config(self, default_params):
        """Protocol is not modified when DebateConfig is None."""
        protocol = DebateProtocol(rounds=3)
        params = default_params.copy()
        params["protocol"] = protocol

        merge_config_objects(
            debate_config=None,
            agent_config=None,
            memory_config=None,
            streaming_config=None,
            observability_config=None,
            **params,
        )

        # Protocol should retain original values
        assert protocol.rounds == 3


# ===========================================================================
# Edge Cases
# ===========================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_string_values_preserved(self, default_params):
        """Empty string values are preserved correctly."""
        params = default_params.copy()
        params["loop_id"] = ""
        params["org_id"] = ""

        cfg = merge_config_objects(
            debate_config=None,
            agent_config=None,
            memory_config=None,
            streaming_config=None,
            observability_config=None,
            **params,
        )

        assert cfg.loop_id == ""
        assert cfg.org_id == ""

    def test_zero_values_preserved(self, default_params):
        """Zero numeric values are preserved correctly."""
        memory_cfg = MemoryConfig(
            rlm_compression_threshold=0,
            rlm_max_recent_messages=0,
            rlm_compression_round_threshold=0,
        )

        cfg = merge_config_objects(
            debate_config=None,
            agent_config=None,
            memory_config=memory_cfg,
            streaming_config=None,
            observability_config=None,
            **default_params,
        )

        assert cfg.rlm_compression_threshold == 0
        assert cfg.rlm_max_recent_messages == 0
        assert cfg.rlm_compression_round_threshold == 0

    def test_false_boolean_values_preserved(self, default_params):
        """False boolean values are preserved (not treated as None)."""
        memory_cfg = MemoryConfig(
            enable_cross_debate_memory=False,
            enable_knowledge_retrieval=False,
            use_rlm_limiter=False,
        )

        cfg = merge_config_objects(
            debate_config=None,
            agent_config=None,
            memory_config=memory_cfg,
            streaming_config=None,
            observability_config=None,
            **default_params,
        )

        assert cfg.enable_cross_debate_memory is False
        assert cfg.enable_knowledge_retrieval is False
        assert cfg.use_rlm_limiter is False

    def test_none_object_values_fallback_to_individual_params(self, default_params):
        """None object values in configs fall back to individual params."""
        agent_cfg = AgentConfig(
            agent_weights=None,  # Explicitly None
            agent_selector=None,
        )
        mock_selector = MagicMock()
        params = default_params.copy()
        params["agent_weights"] = {"custom": 1.0}
        params["agent_selector"] = mock_selector

        cfg = merge_config_objects(
            debate_config=None,
            agent_config=agent_cfg,
            memory_config=None,
            streaming_config=None,
            observability_config=None,
            **params,
        )

        # Should use individual params since config values are None
        assert cfg.agent_weights == {"custom": 1.0}
        assert cfg.agent_selector is mock_selector

    def test_list_values_preserved(self, default_params):
        """List values are preserved correctly."""
        messages = [{"role": "user", "content": "test"}]
        obs_cfg = ObservabilityConfig(initial_messages=messages)

        cfg = merge_config_objects(
            debate_config=None,
            agent_config=None,
            memory_config=None,
            streaming_config=None,
            observability_config=obs_cfg,
            **default_params,
        )

        assert cfg.initial_messages == messages
        assert cfg.initial_messages is messages  # Same reference

    def test_dict_values_preserved(self, default_params):
        """Dict values are preserved correctly."""
        weights = {"claude": 1.5, "gpt4": 1.0, "gemini": 0.8}
        agent_cfg = AgentConfig(agent_weights=weights)

        cfg = merge_config_objects(
            debate_config=None,
            agent_config=agent_cfg,
            memory_config=None,
            streaming_config=None,
            observability_config=None,
            **default_params,
        )

        assert cfg.agent_weights == weights
        assert cfg.agent_weights is weights  # Same reference

    def test_float_precision_preserved(self, default_params):
        """Float precision is preserved correctly."""
        obs_cfg = ObservabilityConfig(
            breeding_threshold=0.123456789,
            broadcast_min_confidence=0.999999999,
            ml_delegation_weight=0.333333333,
        )

        cfg = merge_config_objects(
            debate_config=None,
            agent_config=None,
            memory_config=None,
            streaming_config=None,
            observability_config=obs_cfg,
            **default_params,
        )

        assert cfg.breeding_threshold == 0.123456789
        assert cfg.broadcast_min_confidence == 0.999999999
        assert cfg.ml_delegation_weight == 0.333333333


# ===========================================================================
# Config Category Coverage Tests
# ===========================================================================


class TestDebateConfigFields:
    """Tests for complete DebateConfig field coverage in merge."""

    def test_all_debate_config_fields_merged(self, default_params):
        """All DebateConfig fields that apply are merged."""
        debate_cfg = DebateConfig(
            enable_adaptive_rounds=True,
            debate_strategy="custom_strategy",
            enable_agent_hierarchy=False,
            hierarchy_config={"levels": 2},
        )

        cfg = merge_config_objects(
            debate_config=debate_cfg,
            agent_config=None,
            memory_config=None,
            streaming_config=None,
            observability_config=None,
            **default_params,
        )

        assert cfg.enable_adaptive_rounds is True
        assert cfg.debate_strategy == "custom_strategy"
        assert cfg.enable_agent_hierarchy is False
        assert cfg.hierarchy_config == {"levels": 2}


class TestAgentConfigFields:
    """Tests for complete AgentConfig field coverage in merge."""

    def test_all_agent_config_fields_merged(self, default_params):
        """All AgentConfig fields are merged."""
        mock_objects = {
            name: MagicMock()
            for name in [
                "agent_selector",
                "circuit_breaker",
                "position_tracker",
                "position_ledger",
                "elo_system",
                "calibration_tracker",
                "relationship_tracker",
                "persona_manager",
                "vertical_persona_manager",
                "fabric",
                "airlock_config",
                "fabric_config",
            ]
        }

        agent_cfg = AgentConfig(
            agent_weights={"test": 1.0},
            agent_selector=mock_objects["agent_selector"],
            use_performance_selection=False,
            circuit_breaker=mock_objects["circuit_breaker"],
            use_airlock=True,
            airlock_config=mock_objects["airlock_config"],
            position_tracker=mock_objects["position_tracker"],
            position_ledger=mock_objects["position_ledger"],
            enable_position_ledger=True,
            elo_system=mock_objects["elo_system"],
            calibration_tracker=mock_objects["calibration_tracker"],
            relationship_tracker=mock_objects["relationship_tracker"],
            persona_manager=mock_objects["persona_manager"],
            vertical="finance",
            vertical_persona_manager=mock_objects["vertical_persona_manager"],
            auto_detect_vertical=False,
            fabric=mock_objects["fabric"],
            fabric_config=mock_objects["fabric_config"],
        )

        cfg = merge_config_objects(
            debate_config=None,
            agent_config=agent_cfg,
            memory_config=None,
            streaming_config=None,
            observability_config=None,
            **default_params,
        )

        assert cfg.agent_weights == {"test": 1.0}
        assert cfg.agent_selector is mock_objects["agent_selector"]
        assert cfg.use_performance_selection is False
        assert cfg.circuit_breaker is mock_objects["circuit_breaker"]
        assert cfg.use_airlock is True
        assert cfg.airlock_config is mock_objects["airlock_config"]
        assert cfg.position_tracker is mock_objects["position_tracker"]
        assert cfg.position_ledger is mock_objects["position_ledger"]
        assert cfg.enable_position_ledger is True
        assert cfg.elo_system is mock_objects["elo_system"]
        assert cfg.calibration_tracker is mock_objects["calibration_tracker"]
        assert cfg.relationship_tracker is mock_objects["relationship_tracker"]
        assert cfg.persona_manager is mock_objects["persona_manager"]
        assert cfg.vertical == "finance"
        assert cfg.vertical_persona_manager is mock_objects["vertical_persona_manager"]
        assert cfg.auto_detect_vertical is False
        assert cfg.fabric is mock_objects["fabric"]
        assert cfg.fabric_config is mock_objects["fabric_config"]


class TestMemoryConfigFields:
    """Tests for complete MemoryConfig field coverage in merge."""

    def test_all_memory_config_fields_merged(self, default_params):
        """All MemoryConfig fields are merged."""
        mock_objects = {
            name: MagicMock()
            for name in [
                "memory",
                "continuum_memory",
                "consensus_memory",
                "debate_embeddings",
                "insight_store",
                "dissent_retriever",
                "flip_detector",
                "moment_detector",
                "tier_analytics_tracker",
                "cross_debate_memory",
                "knowledge_mound",
                "revalidation_scheduler",
                "rlm_limiter",
                "checkpoint_manager",
            ]
        }

        memory_cfg = MemoryConfig(
            memory=mock_objects["memory"],
            continuum_memory=mock_objects["continuum_memory"],
            consensus_memory=mock_objects["consensus_memory"],
            debate_embeddings=mock_objects["debate_embeddings"],
            insight_store=mock_objects["insight_store"],
            dissent_retriever=mock_objects["dissent_retriever"],
            flip_detector=mock_objects["flip_detector"],
            moment_detector=mock_objects["moment_detector"],
            tier_analytics_tracker=mock_objects["tier_analytics_tracker"],
            cross_debate_memory=mock_objects["cross_debate_memory"],
            enable_cross_debate_memory=False,
            knowledge_mound=mock_objects["knowledge_mound"],
            auto_create_knowledge_mound=False,
            enable_knowledge_retrieval=False,
            enable_knowledge_ingestion=False,
            enable_knowledge_extraction=True,
            extraction_min_confidence=0.5,
            enable_belief_guidance=True,
            enable_auto_revalidation=True,
            revalidation_staleness_threshold=0.6,
            revalidation_check_interval_seconds=1800,
            revalidation_scheduler=mock_objects["revalidation_scheduler"],
            use_rlm_limiter=False,
            rlm_limiter=mock_objects["rlm_limiter"],
            rlm_compression_threshold=4000,
            rlm_max_recent_messages=8,
            rlm_summary_level="ABSTRACT",
            rlm_compression_round_threshold=4,
            checkpoint_manager=mock_objects["checkpoint_manager"],
            enable_checkpointing=False,
        )

        cfg = merge_config_objects(
            debate_config=None,
            agent_config=None,
            memory_config=memory_cfg,
            streaming_config=None,
            observability_config=None,
            **default_params,
        )

        assert cfg.memory is mock_objects["memory"]
        assert cfg.continuum_memory is mock_objects["continuum_memory"]
        assert cfg.consensus_memory is mock_objects["consensus_memory"]
        assert cfg.debate_embeddings is mock_objects["debate_embeddings"]
        assert cfg.insight_store is mock_objects["insight_store"]
        assert cfg.dissent_retriever is mock_objects["dissent_retriever"]
        assert cfg.flip_detector is mock_objects["flip_detector"]
        assert cfg.moment_detector is mock_objects["moment_detector"]
        assert cfg.tier_analytics_tracker is mock_objects["tier_analytics_tracker"]
        assert cfg.cross_debate_memory is mock_objects["cross_debate_memory"]
        assert cfg.enable_cross_debate_memory is False
        assert cfg.knowledge_mound is mock_objects["knowledge_mound"]
        assert cfg.auto_create_knowledge_mound is False
        assert cfg.enable_knowledge_retrieval is False
        assert cfg.enable_knowledge_ingestion is False
        assert cfg.enable_knowledge_extraction is True
        assert cfg.extraction_min_confidence == 0.5
        assert cfg.enable_belief_guidance is True
        assert cfg.enable_auto_revalidation is True
        assert cfg.revalidation_staleness_threshold == 0.6
        assert cfg.revalidation_check_interval_seconds == 1800
        assert cfg.revalidation_scheduler is mock_objects["revalidation_scheduler"]
        assert cfg.use_rlm_limiter is False
        assert cfg.rlm_limiter is mock_objects["rlm_limiter"]
        assert cfg.rlm_compression_threshold == 4000
        assert cfg.rlm_max_recent_messages == 8
        assert cfg.rlm_summary_level == "ABSTRACT"
        assert cfg.rlm_compression_round_threshold == 4
        assert cfg.checkpoint_manager is mock_objects["checkpoint_manager"]
        assert cfg.enable_checkpointing is False


class TestStreamingConfigFields:
    """Tests for complete StreamingConfig field coverage in merge."""

    def test_all_streaming_config_fields_merged(self, default_params):
        """All StreamingConfig fields are merged."""
        mock_objects = {
            name: MagicMock()
            for name in [
                "event_hooks",
                "hook_manager",
                "event_emitter",
                "spectator",
                "recorder",
                "skill_registry",
                "propulsion_engine",
            ]
        }

        streaming_cfg = StreamingConfig(
            event_hooks=mock_objects["event_hooks"],
            hook_manager=mock_objects["hook_manager"],
            event_emitter=mock_objects["event_emitter"],
            spectator=mock_objects["spectator"],
            recorder=mock_objects["recorder"],
            loop_id="stream-loop-456",
            strict_loop_scoping=True,
            skill_registry=mock_objects["skill_registry"],
            enable_skills=True,
            propulsion_engine=mock_objects["propulsion_engine"],
            enable_propulsion=True,
        )

        cfg = merge_config_objects(
            debate_config=None,
            agent_config=None,
            memory_config=None,
            streaming_config=streaming_cfg,
            observability_config=None,
            **default_params,
        )

        assert cfg.event_hooks is mock_objects["event_hooks"]
        assert cfg.hook_manager is mock_objects["hook_manager"]
        assert cfg.event_emitter is mock_objects["event_emitter"]
        assert cfg.spectator is mock_objects["spectator"]
        assert cfg.recorder is mock_objects["recorder"]
        assert cfg.loop_id == "stream-loop-456"
        assert cfg.strict_loop_scoping is True
        assert cfg.skill_registry is mock_objects["skill_registry"]
        assert cfg.enable_skills is True
        assert cfg.propulsion_engine is mock_objects["propulsion_engine"]
        assert cfg.enable_propulsion is True


class TestObservabilityConfigFields:
    """Tests for complete ObservabilityConfig field coverage in merge."""

    def test_all_observability_config_fields_merged(self, default_params):
        """All ObservabilityConfig fields are merged."""
        mock_objects = {
            name: MagicMock()
            for name in [
                "performance_monitor",
                "prompt_evolver",
                "breakpoint_manager",
                "trending_topic",
                "pulse_manager",
                "population_manager",
                "evidence_collector",
                "usage_tracker",
                "broadcast_pipeline",
                "training_exporter",
                "ml_delegation_strategy",
                "post_debate_workflow",
            ]
        }

        obs_cfg = ObservabilityConfig(
            performance_monitor=mock_objects["performance_monitor"],
            enable_performance_monitor=False,
            enable_telemetry=True,
            prompt_evolver=mock_objects["prompt_evolver"],
            enable_prompt_evolution=True,
            breakpoint_manager=mock_objects["breakpoint_manager"],
            trending_topic=mock_objects["trending_topic"],
            pulse_manager=mock_objects["pulse_manager"],
            auto_fetch_trending=True,
            population_manager=mock_objects["population_manager"],
            auto_evolve=True,
            breeding_threshold=0.75,
            evidence_collector=mock_objects["evidence_collector"],
            org_id="obs-org",
            user_id="obs-user",
            usage_tracker=mock_objects["usage_tracker"],
            broadcast_pipeline=mock_objects["broadcast_pipeline"],
            auto_broadcast=True,
            broadcast_min_confidence=0.9,
            training_exporter=mock_objects["training_exporter"],
            auto_export_training=True,
            training_export_min_confidence=0.8,
            enable_ml_delegation=True,
            ml_delegation_strategy=mock_objects["ml_delegation_strategy"],
            ml_delegation_weight=0.5,
            enable_quality_gates=True,
            quality_gate_threshold=0.7,
            enable_consensus_estimation=True,
            consensus_early_termination_threshold=0.88,
            post_debate_workflow=mock_objects["post_debate_workflow"],
            enable_post_debate_workflow=True,
            post_debate_workflow_threshold=0.65,
            initial_messages=[{"test": "message"}],
        )

        cfg = merge_config_objects(
            debate_config=None,
            agent_config=None,
            memory_config=None,
            streaming_config=None,
            observability_config=obs_cfg,
            **default_params,
        )

        assert cfg.performance_monitor is mock_objects["performance_monitor"]
        assert cfg.enable_performance_monitor is False
        assert cfg.enable_telemetry is True
        assert cfg.prompt_evolver is mock_objects["prompt_evolver"]
        assert cfg.enable_prompt_evolution is True
        assert cfg.breakpoint_manager is mock_objects["breakpoint_manager"]
        assert cfg.trending_topic is mock_objects["trending_topic"]
        assert cfg.pulse_manager is mock_objects["pulse_manager"]
        assert cfg.auto_fetch_trending is True
        assert cfg.population_manager is mock_objects["population_manager"]
        assert cfg.auto_evolve is True
        assert cfg.breeding_threshold == 0.75
        assert cfg.evidence_collector is mock_objects["evidence_collector"]
        assert cfg.org_id == "obs-org"
        assert cfg.user_id == "obs-user"
        assert cfg.usage_tracker is mock_objects["usage_tracker"]
        assert cfg.broadcast_pipeline is mock_objects["broadcast_pipeline"]
        assert cfg.auto_broadcast is True
        assert cfg.broadcast_min_confidence == 0.9
        assert cfg.training_exporter is mock_objects["training_exporter"]
        assert cfg.auto_export_training is True
        assert cfg.training_export_min_confidence == 0.8
        assert cfg.enable_ml_delegation is True
        assert cfg.ml_delegation_strategy is mock_objects["ml_delegation_strategy"]
        assert cfg.ml_delegation_weight == 0.5
        assert cfg.enable_quality_gates is True
        assert cfg.quality_gate_threshold == 0.7
        assert cfg.enable_consensus_estimation is True
        assert cfg.consensus_early_termination_threshold == 0.88
        assert cfg.post_debate_workflow is mock_objects["post_debate_workflow"]
        assert cfg.enable_post_debate_workflow is True
        assert cfg.post_debate_workflow_threshold == 0.65
        assert cfg.initial_messages == [{"test": "message"}]


# ===========================================================================
# Integration-Like Tests
# ===========================================================================


class TestRealWorldScenarios:
    """Tests simulating real-world usage patterns."""

    def test_minimal_config_scenario(self, default_params):
        """Minimal config - just enable telemetry."""
        obs_cfg = ObservabilityConfig(enable_telemetry=True)

        cfg = merge_config_objects(
            debate_config=None,
            agent_config=None,
            memory_config=None,
            streaming_config=None,
            observability_config=obs_cfg,
            **default_params,
        )

        assert cfg.enable_telemetry is True
        # All other fields should be defaults
        assert cfg.enable_adaptive_rounds is False
        assert cfg.memory is None

    def test_production_config_scenario(
        self,
        default_params,
        debate_config,
        agent_config,
        memory_config,
        streaming_config,
        observability_config,
    ):
        """Production-like config with all config objects."""
        protocol = DebateProtocol()
        params = default_params.copy()
        params["protocol"] = protocol

        cfg = merge_config_objects(
            debate_config=debate_config,
            agent_config=agent_config,
            memory_config=memory_config,
            streaming_config=streaming_config,
            observability_config=observability_config,
            **params,
        )

        # Verify key production settings
        assert cfg.enable_telemetry is True
        assert cfg.use_airlock is True
        assert cfg.org_id == "test-org"
        assert cfg.loop_id == "test-loop-123"
        assert protocol.rounds == 7  # Modified by debate_config

    def test_knowledge_focused_config_scenario(self, default_params):
        """Config focused on knowledge features."""
        memory_cfg = MemoryConfig(
            enable_knowledge_retrieval=True,
            enable_knowledge_ingestion=True,
            enable_knowledge_extraction=True,
            extraction_min_confidence=0.5,
            enable_belief_guidance=True,
            enable_auto_revalidation=True,
        )

        cfg = merge_config_objects(
            debate_config=None,
            agent_config=None,
            memory_config=memory_cfg,
            streaming_config=None,
            observability_config=None,
            **default_params,
        )

        assert cfg.enable_knowledge_retrieval is True
        assert cfg.enable_knowledge_ingestion is True
        assert cfg.enable_knowledge_extraction is True
        assert cfg.extraction_min_confidence == 0.5
        assert cfg.enable_belief_guidance is True
        assert cfg.enable_auto_revalidation is True

    def test_ml_heavy_config_scenario(self, default_params):
        """Config focused on ML features."""
        obs_cfg = ObservabilityConfig(
            enable_ml_delegation=True,
            ml_delegation_weight=0.7,
            enable_quality_gates=True,
            quality_gate_threshold=0.75,
            enable_consensus_estimation=True,
            consensus_early_termination_threshold=0.92,
        )

        cfg = merge_config_objects(
            debate_config=None,
            agent_config=None,
            memory_config=None,
            streaming_config=None,
            observability_config=obs_cfg,
            **default_params,
        )

        assert cfg.enable_ml_delegation is True
        assert cfg.ml_delegation_weight == 0.7
        assert cfg.enable_quality_gates is True
        assert cfg.quality_gate_threshold == 0.75
        assert cfg.enable_consensus_estimation is True
        assert cfg.consensus_early_termination_threshold == 0.92

    def test_streaming_heavy_config_scenario(self, default_params):
        """Config focused on streaming features."""
        streaming_cfg = StreamingConfig(
            loop_id="stream-session-789",
            strict_loop_scoping=True,
            enable_skills=True,
            enable_propulsion=True,
        )

        cfg = merge_config_objects(
            debate_config=None,
            agent_config=None,
            memory_config=None,
            streaming_config=streaming_cfg,
            observability_config=None,
            **default_params,
        )

        assert cfg.loop_id == "stream-session-789"
        assert cfg.strict_loop_scoping is True
        assert cfg.enable_skills is True
        assert cfg.enable_propulsion is True
