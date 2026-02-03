"""Tests for orchestrator_init module.

Verifies that apply_core_components, apply_tracker_components,
store_post_tracker_config, and run_init_subsystems correctly
unpack dataclass fields to Arena instance attributes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.debate.orchestrator_init import (
    apply_core_components,
    apply_tracker_components,
    run_init_subsystems,
    store_post_tracker_config,
)


@dataclass
class FakeCore:
    """Minimal CoreComponents stand-in."""

    env: Any = None
    agents: list = field(default_factory=list)
    protocol: Any = None
    memory: Any = None
    hooks: dict = field(default_factory=dict)
    hook_manager: Any = None
    event_emitter: Any = None
    spectator: Any = None
    debate_embeddings: Any = None
    insight_store: Any = None
    recorder: Any = None
    agent_weights: dict = field(default_factory=dict)
    loop_id: str = "test-loop"
    strict_loop_scoping: bool = False
    circuit_breaker: Any = None
    agent_pool: Any = None
    immune_system: Any = None
    chaos_director: Any = None
    performance_monitor: Any = None
    prompt_evolver: Any = None
    autonomic: Any = None
    initial_messages: list = field(default_factory=list)
    trending_topic: Any = None
    pulse_manager: Any = None
    auto_fetch_trending: bool = False
    population_manager: Any = None
    auto_evolve: bool = False
    breeding_threshold: float = 0.8
    evidence_collector: Any = None
    breakpoint_manager: Any = None
    agent_selector: Any = None
    use_performance_selection: bool = False
    checkpoint_manager: Any = None
    org_id: str = "org-test"
    user_id: str = "user-test"
    extensions: Any = None
    cartographer: Any = None
    event_bridge: Any = None
    enable_ml_delegation: bool = False
    ml_delegation_weight: float = 0.3
    enable_quality_gates: bool = False
    quality_gate_threshold: float = 0.6
    enable_consensus_estimation: bool = False
    consensus_early_termination_threshold: float = 0.85
    ml_delegation_strategy: Any = None
    ml_quality_gate: Any = None
    ml_consensus_estimator: Any = None


@dataclass
class FakeTrackers:
    """Minimal TrackerComponents stand-in."""

    position_tracker: Any = None
    position_ledger: Any = None
    elo_system: Any = None
    persona_manager: Any = None
    dissent_retriever: Any = None
    consensus_memory: Any = None
    flip_detector: Any = None
    calibration_tracker: Any = None
    continuum_memory: Any = None
    relationship_tracker: Any = None
    moment_detector: Any = None
    tier_analytics_tracker: Any = None
    knowledge_mound: Any = None
    enable_knowledge_retrieval: bool = True
    enable_knowledge_ingestion: bool = True
    enable_knowledge_extraction: bool = False
    extraction_min_confidence: float = 0.3
    enable_belief_guidance: bool = False
    coordinator: Any = None
    vertical: str = "general"
    vertical_persona_manager: Any = None


class TestApplyCoreComponents:
    def test_unpacks_core_fields(self):
        arena = MagicMock()
        core = FakeCore(loop_id="debate-42", org_id="org-x", user_id="user-y")
        with patch("aragora.debate.orchestrator_init.try_resolve", return_value=None):
            apply_core_components(arena, core)

        assert arena.loop_id == "debate-42"
        assert arena.org_id == "org-x"
        assert arena.user_id == "user-y"

    def test_sets_budget_coordinator_from_di(self):
        arena = MagicMock()
        mock_coordinator = MagicMock()
        with patch("aragora.debate.orchestrator_init.try_resolve", return_value=mock_coordinator):
            apply_core_components(arena, FakeCore(org_id="org-1", user_id="usr-1"))

        assert arena._budget_coordinator is mock_coordinator
        assert arena._budget_coordinator.org_id == "org-1"
        assert arena._budget_coordinator.user_id == "usr-1"

    def test_creates_budget_coordinator_when_di_unavailable(self):
        arena = MagicMock()
        with patch("aragora.debate.orchestrator_init.try_resolve", return_value=None):
            apply_core_components(arena, FakeCore(org_id="org-2"))

        assert arena._budget_coordinator is not None
        # Should have been assigned (not the MagicMock default)
        assert hasattr(arena._budget_coordinator, "org_id")

    def test_sets_ml_fields(self):
        arena = MagicMock()
        core = FakeCore(enable_ml_delegation=True, ml_delegation_weight=0.5)
        with patch("aragora.debate.orchestrator_init.try_resolve", return_value=None):
            apply_core_components(arena, core)

        assert arena.enable_ml_delegation is True
        assert arena.ml_delegation_weight == 0.5

    def test_initializes_event_bus_to_none(self):
        arena = MagicMock()
        with patch("aragora.debate.orchestrator_init.try_resolve", return_value=None):
            apply_core_components(arena, FakeCore())

        assert arena.event_bus is None


class TestApplyTrackerComponents:
    def test_unpacks_tracker_fields(self):
        arena = MagicMock()
        trackers = FakeTrackers(vertical="finance", enable_knowledge_retrieval=False)
        apply_tracker_components(arena, trackers)

        assert arena.vertical == "finance"
        assert arena.enable_knowledge_retrieval is False

    def test_sets_coordinator(self):
        arena = MagicMock()
        mock_coord = MagicMock()
        trackers = FakeTrackers(coordinator=mock_coord)
        apply_tracker_components(arena, trackers)

        assert arena._trackers is mock_coord


class TestStorePostTrackerConfig:
    def test_stores_supermemory_config(self):
        arena = MagicMock()
        cfg = MagicMock()
        cfg.enable_supermemory = True
        cfg.supermemory_max_context_items = 20
        cfg.enable_auto_revalidation = False
        cfg.cross_debate_memory = None
        cfg.enable_cross_debate_memory = True

        store_post_tracker_config(arena, cfg, document_store="doc-store", evidence_store="ev-store")

        assert arena.enable_supermemory is True
        assert arena.supermemory_max_context_items == 20
        assert arena.document_store == "doc-store"
        assert arena.evidence_store == "ev-store"


class TestRunInitSubsystems:
    def test_calls_all_init_methods(self):
        arena = MagicMock()
        run_init_subsystems(arena)

        arena._init_user_participation.assert_called_once()
        arena._init_event_bus.assert_called_once()
        arena._init_roles_and_stances.assert_called_once()
        arena._init_convergence.assert_called_once()
        arena._init_caches.assert_called_once()
        arena._init_lifecycle_manager.assert_called_once()
        arena._init_event_emitter.assert_called_once()
        arena._init_checkpoint_ops.assert_called_once()
        arena._init_checkpoint_bridge.assert_called_once()
        arena._init_grounded_operations.assert_called_once()
        arena._init_knowledge_ops.assert_called_once()
        arena._init_phases.assert_called_once()
        arena._init_prompt_context_builder.assert_called_once()
        arena._init_context_delegator.assert_called_once()
        arena._init_termination_checker.assert_called_once()
        arena._init_cross_subscriber_bridge.assert_called_once()
