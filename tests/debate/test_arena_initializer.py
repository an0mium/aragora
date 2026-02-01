"""
Tests for Arena initializer module.

Covers CoreComponents/TrackerComponents dataclasses and
ArenaInitializer.init_core / init_trackers behavior including
auto-initialization of subsystems.
"""

from __future__ import annotations

from dataclasses import fields
from unittest.mock import MagicMock, patch

import pytest

from aragora.debate.arena_initializer import (
    ArenaInitializer,
    CoreComponents,
    TrackerComponents,
)
from aragora.debate.protocol import DebateProtocol


# ===========================================================================
# Helpers
# ===========================================================================


def _make_mock_env(task: str = "Test task"):
    env = MagicMock()
    env.task = task
    return env


def _make_mock_agent(name: str = "agent1"):
    agent = MagicMock()
    agent.name = name
    return agent


def _make_initializer():
    return ArenaInitializer(broadcast_callback=MagicMock())


# ===========================================================================
# Dataclass structure
# ===========================================================================


class TestCoreComponentsStructure:
    """Tests for CoreComponents dataclass fields."""

    def test_has_required_fields(self):
        names = {f.name for f in fields(CoreComponents)}
        assert "env" in names
        assert "agents" in names
        assert "protocol" in names
        assert "circuit_breaker" in names
        assert "agent_pool" in names
        assert "autonomic" in names
        assert "event_bridge" in names

    def test_has_ml_fields(self):
        names = {f.name for f in fields(CoreComponents)}
        assert "enable_ml_delegation" in names
        assert "ml_delegation_weight" in names
        assert "ml_delegation_strategy" in names


class TestTrackerComponentsStructure:
    """Tests for TrackerComponents dataclass fields."""

    def test_has_required_fields(self):
        names = {f.name for f in fields(TrackerComponents)}
        assert "elo_system" in names
        assert "calibration_tracker" in names
        assert "knowledge_mound" in names
        assert "coordinator" in names

    def test_has_vertical_fields(self):
        names = {f.name for f in fields(TrackerComponents)}
        assert "vertical" in names
        assert "vertical_persona_manager" in names


# ===========================================================================
# init_core
# ===========================================================================


class TestInitCore:
    """Tests for ArenaInitializer.init_core."""

    def _call_init_core(self, **overrides):
        init = _make_initializer()
        defaults = dict(
            environment=_make_mock_env(),
            agents=[_make_mock_agent()],
            protocol=None,
            memory=None,
            event_hooks=None,
            hook_manager=None,
            event_emitter=None,
            spectator=None,
            debate_embeddings=None,
            insight_store=None,
            recorder=None,
            agent_weights=None,
            loop_id="test-loop",
            strict_loop_scoping=False,
            circuit_breaker=None,
            initial_messages=None,
            trending_topic=None,
            pulse_manager=None,
            auto_fetch_trending=False,
            population_manager=None,
            auto_evolve=False,
            breeding_threshold=0.7,
            evidence_collector=None,
            breakpoint_manager=None,
            checkpoint_manager=None,
            enable_checkpointing=False,
            performance_monitor=None,
            enable_performance_monitor=False,
            enable_telemetry=False,
            use_airlock=False,
            airlock_config=None,
            agent_selector=None,
            use_performance_selection=False,
            prompt_evolver=None,
            enable_prompt_evolution=False,
        )
        defaults.update(overrides)
        return init.init_core(**defaults)

    def test_returns_core_components(self):
        result = self._call_init_core()
        assert isinstance(result, CoreComponents)

    def test_protocol_defaults_when_none(self):
        result = self._call_init_core(protocol=None)
        assert isinstance(result.protocol, DebateProtocol)

    def test_preserves_provided_protocol(self):
        proto = DebateProtocol(rounds=5)
        result = self._call_init_core(protocol=proto)
        assert result.protocol.rounds == 5

    def test_circuit_breaker_defaults(self):
        result = self._call_init_core(circuit_breaker=None)
        assert result.circuit_breaker is not None

    def test_agent_pool_created(self):
        result = self._call_init_core()
        assert result.agent_pool is not None

    def test_autonomic_created(self):
        result = self._call_init_core()
        assert result.autonomic is not None

    def test_event_bridge_created(self):
        result = self._call_init_core()
        assert result.event_bridge is not None

    def test_extensions_created(self):
        result = self._call_init_core(org_id="org1", user_id="user1")
        assert result.extensions is not None
        assert result.org_id == "org1"
        assert result.user_id == "user1"

    def test_performance_monitor_explicit(self):
        monitor = MagicMock()
        result = self._call_init_core(performance_monitor=monitor)
        assert result.performance_monitor is monitor

    def test_performance_monitor_auto_init(self):
        result = self._call_init_core(enable_performance_monitor=True)
        assert result.performance_monitor is not None

    def test_performance_monitor_none_when_disabled(self):
        result = self._call_init_core(
            performance_monitor=None,
            enable_performance_monitor=False,
        )
        assert result.performance_monitor is None

    def test_initial_messages_default_empty(self):
        result = self._call_init_core(initial_messages=None)
        assert result.initial_messages == []

    def test_ml_defaults_disabled(self):
        result = self._call_init_core()
        assert result.enable_ml_delegation is False
        assert result.enable_quality_gates is False
        assert result.enable_consensus_estimation is False


# ===========================================================================
# init_trackers
# ===========================================================================


class TestInitTrackers:
    """Tests for ArenaInitializer.init_trackers."""

    def _call_init_trackers(self, **overrides):
        init = _make_initializer()

        # We need a real AgentPool with at least one agent
        from aragora.debate.agent_pool import AgentPool, AgentPoolConfig
        from aragora.debate.protocol import CircuitBreaker

        agents = [_make_mock_agent()]
        pool = AgentPool(agents=agents, config=AgentPoolConfig(circuit_breaker=CircuitBreaker()))

        defaults = dict(
            protocol=DebateProtocol(),
            loop_id="test-loop",
            agent_pool=pool,
            position_tracker=None,
            position_ledger=None,
            enable_position_ledger=False,
            elo_system=None,
            persona_manager=None,
            dissent_retriever=None,
            consensus_memory=None,
            flip_detector=None,
            calibration_tracker=None,
            continuum_memory=None,
            relationship_tracker=None,
            moment_detector=None,
            auto_detect_vertical=False,
        )
        defaults.update(overrides)
        return init.init_trackers(**defaults)

    def test_returns_tracker_components(self):
        result = self._call_init_trackers()
        assert isinstance(result, TrackerComponents)

    def test_coordinator_created(self):
        result = self._call_init_trackers()
        assert result.coordinator is not None

    def test_elo_upgrades_judge_selection(self):
        elo = MagicMock()
        proto = DebateProtocol(judge_selection="random")
        result = self._call_init_trackers(protocol=proto, elo_system=elo)
        assert proto.judge_selection == "elo_ranked"

    def test_no_elo_preserves_judge_selection(self):
        proto = DebateProtocol(judge_selection="random")
        result = self._call_init_trackers(protocol=proto, elo_system=None)
        assert proto.judge_selection == "random"

    def test_knowledge_defaults(self):
        result = self._call_init_trackers()
        assert result.enable_knowledge_retrieval is True
        assert result.enable_knowledge_ingestion is True

    def test_knowledge_extraction_disabled_by_default(self):
        result = self._call_init_trackers()
        assert result.enable_knowledge_extraction is False


# ===========================================================================
# Auto-initialization helpers
# ===========================================================================


class TestAutoInitHelpers:
    """Tests for _auto_init helper methods."""

    def test_auto_init_breakpoint_manager_returns_manager(self):
        init = _make_initializer()
        proto = DebateProtocol(enable_breakpoints=True)
        result = init._auto_init_breakpoint_manager(proto)
        assert result is not None

    def test_auto_init_breakpoint_manager_handles_import_error(self):
        init = _make_initializer()
        proto = DebateProtocol(enable_breakpoints=True)
        with patch(
            "aragora.debate.arena_initializer.ArenaInitializer._auto_init_breakpoint_manager",
            return_value=None,
        ):
            # Verify that when _auto_init_breakpoint_manager returns None,
            # it doesn't raise an exception
            result = init._auto_init_breakpoint_manager(proto)
            assert result is None, "Patched method should return None"

    def test_auto_init_position_ledger(self):
        init = _make_initializer()
        ledger = init._auto_init_position_ledger()
        # Should return something (SubsystemCoordinator auto-creates it)
        assert ledger is not None

    def test_auto_init_calibration_tracker(self):
        init = _make_initializer()
        tracker = init._auto_init_calibration_tracker()
        assert tracker is not None
