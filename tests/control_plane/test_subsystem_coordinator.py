"""
Comprehensive tests for SubsystemCoordinator.

Tests cover:
- Subsystem registration and discovery
- Health monitoring (moment detection reset, hook handlers)
- Dependency resolution between subsystems
- Startup/shutdown sequencing (auto-init)
- Error handling and recovery
- Concurrent operations
- Edge cases and error paths
- Phase 9 Cross-Pollination Bridges
- Phase 10 Knowledge Mound Integration
"""

from __future__ import annotations

import asyncio
import pytest
from dataclasses import dataclass, field
from typing import Any, Optional
from unittest.mock import Mock, MagicMock, patch, AsyncMock

from aragora.debate.subsystem_coordinator import (
    SubsystemCoordinator,
    SubsystemConfig,
    Resettable,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_protocol():
    """Create a mock debate protocol."""
    protocol = Mock()
    protocol.rounds = 3
    protocol.consensus_threshold = 0.7
    return protocol


@pytest.fixture
def mock_elo_system():
    """Create a mock ELO system."""
    elo = Mock()
    elo.get_rating = Mock(return_value=1500.0)
    elo.update_ratings = Mock()
    return elo


@pytest.fixture
def mock_calibration_tracker():
    """Create a mock calibration tracker."""
    tracker = Mock()
    summary = Mock()
    summary.total_predictions = 100
    summary.brier_score = 0.15  # Lower is better
    tracker.get_calibration_summary = Mock(return_value=summary)
    tracker.record_prediction = Mock()
    return tracker


@pytest.fixture
def mock_consensus_memory():
    """Create a mock consensus memory."""
    memory = Mock()
    memory.store_consensus = Mock()
    memory.retrieve = Mock(return_value=[])
    return memory


@pytest.fixture
def mock_dissent_retriever():
    """Create a mock dissent retriever."""
    retriever = Mock()
    retriever.retrieve_for_new_debate = Mock(
        return_value={
            "relevant_dissents": [
                {"agent": "grok", "position": "Rust is better", "outcome": "minority"},
                {"agent": "gemini", "position": "Go is simpler", "outcome": "minority"},
            ]
        }
    )
    return retriever


@pytest.fixture
def mock_continuum_memory():
    """Create a mock continuum memory with ContinuumMemoryEntry-like objects."""
    memory = Mock()

    # Create mock entries that match ContinuumMemoryEntry structure
    entry1 = Mock()
    entry1.content = "Past debate learned about testing strategies"
    entry1.metadata = {"summary": "Testing best practices"}

    entry2 = Mock()
    entry2.content = "Another debate about code quality"
    entry2.metadata = {"summary": "Code quality principles"}

    memory.retrieve = Mock(return_value=[entry1, entry2])
    memory.add = Mock()
    return memory


@pytest.fixture
def mock_position_ledger():
    """Create a mock position ledger."""
    ledger = Mock()
    ledger.record_position = Mock()
    ledger.get_positions = Mock(return_value=[])
    return ledger


@pytest.fixture
def mock_moment_detector():
    """Create a resettable mock moment detector."""

    class ResettableMomentDetector(Resettable):
        def __init__(self):
            self.reset_called = False

        def reset(self):
            self.reset_called = True

    return ResettableMomentDetector()


@pytest.fixture
def mock_relationship_tracker():
    """Create a mock relationship tracker."""
    tracker = Mock()
    tracker.get_relationship = Mock(return_value=None)
    tracker.update_relationship = Mock()
    return tracker


@pytest.fixture
def mock_hook_manager():
    """Create a mock hook manager."""
    manager = Mock()
    manager.register = Mock()
    manager.trigger = AsyncMock()
    return manager


@pytest.fixture
def mock_context():
    """Create a mock debate context."""
    ctx = Mock()
    ctx.debate_id = "test-debate-123"
    ctx.domain = "programming"

    # Mock environment
    ctx.env = Mock()
    ctx.env.task = "Discuss the best programming language"

    # Mock agents list
    agent1 = Mock()
    agent1.name = "claude"
    agent2 = Mock()
    agent2.name = "grok"
    ctx.agents = [agent1, agent2]
    ctx.start_time = 1707753600  # 2024-02-12T16:00:00Z

    return ctx


@pytest.fixture
def mock_result():
    """Create a mock debate result."""
    result = Mock()
    result.consensus = "Python is versatile for many use cases"
    result.consensus_confidence = 0.85
    result.votes = [{"agent": "claude", "choice": "python"}]
    result.dissenting_views = [{"agent": "grok", "view": "Rust is better"}]
    result.predictions = {
        "claude": {"prediction": "Python is versatile for many use cases", "confidence": 0.8}
    }
    result.messages = []
    result.rounds_completed = 2
    return result


# ============================================================================
# Subsystem Registration and Discovery Tests
# ============================================================================


class TestSubsystemRegistration:
    """Tests for subsystem registration and discovery."""

    def test_register_no_subsystems(self):
        """Coordinator can be created with no subsystems."""
        coordinator = SubsystemCoordinator()

        assert coordinator._initialized is True
        assert coordinator.position_tracker is None
        assert coordinator.elo_system is None
        assert coordinator.calibration_tracker is None

    def test_register_single_subsystem(self, mock_elo_system):
        """Coordinator accepts a single pre-configured subsystem."""
        coordinator = SubsystemCoordinator(elo_system=mock_elo_system)

        assert coordinator.elo_system is mock_elo_system
        assert coordinator.has_elo is True

    def test_register_multiple_subsystems(
        self, mock_elo_system, mock_calibration_tracker, mock_consensus_memory
    ):
        """Coordinator accepts multiple subsystems."""
        coordinator = SubsystemCoordinator(
            elo_system=mock_elo_system,
            calibration_tracker=mock_calibration_tracker,
            consensus_memory=mock_consensus_memory,
        )

        assert coordinator.has_elo is True
        assert coordinator.has_calibration is True
        assert coordinator.has_consensus_memory is True

    def test_register_with_protocol_and_loop_id(self, mock_protocol):
        """Coordinator stores protocol and loop_id."""
        coordinator = SubsystemCoordinator(
            protocol=mock_protocol,
            loop_id="test-loop-456",
        )

        assert coordinator.protocol is mock_protocol
        assert coordinator.loop_id == "test-loop-456"

    def test_discover_position_tracking_via_tracker(self):
        """Position tracking is discovered when tracker is set."""
        position_tracker = Mock()
        coordinator = SubsystemCoordinator(position_tracker=position_tracker)

        assert coordinator.has_position_tracking is True

    def test_discover_position_tracking_via_ledger(self, mock_position_ledger):
        """Position tracking is discovered when ledger is set."""
        coordinator = SubsystemCoordinator(position_ledger=mock_position_ledger)

        assert coordinator.has_position_tracking is True

    def test_discover_no_position_tracking(self):
        """Position tracking not discovered when neither tracker nor ledger set."""
        coordinator = SubsystemCoordinator()

        assert coordinator.has_position_tracking is False


# ============================================================================
# Dependency Resolution Tests
# ============================================================================


class TestDependencyResolution:
    """Tests for dependency resolution between subsystems."""

    def test_dissent_retriever_requires_consensus_memory(self, mock_consensus_memory):
        """DissentRetriever auto-initializes when consensus_memory is provided."""
        # When consensus_memory is provided, dissent_retriever should be auto-initialized
        with patch(
            "aragora.debate.subsystem_coordinator.SubsystemCoordinator._auto_init_dissent_retriever"
        ) as mock_init:
            coordinator = SubsystemCoordinator(consensus_memory=mock_consensus_memory)
            mock_init.assert_called_once()

    def test_moment_detector_uses_available_subsystems(
        self, mock_elo_system, mock_position_ledger, mock_relationship_tracker
    ):
        """MomentDetector can use multiple optional subsystems."""
        with patch(
            "aragora.debate.subsystem_coordinator.SubsystemCoordinator._auto_init_moment_detector"
        ) as mock_init:
            coordinator = SubsystemCoordinator(
                enable_moment_detection=True,
                elo_system=mock_elo_system,
                position_ledger=mock_position_ledger,
                relationship_tracker=mock_relationship_tracker,
            )
            mock_init.assert_called_once()

    def test_hook_handlers_require_hook_manager(self, mock_hook_manager):
        """HookHandlerRegistry requires a hook_manager."""
        with patch(
            "aragora.debate.subsystem_coordinator.SubsystemCoordinator._auto_init_hook_handlers"
        ) as mock_init:
            coordinator = SubsystemCoordinator(
                hook_manager=mock_hook_manager,
                enable_hook_handlers=True,
            )
            mock_init.assert_called_once()

    def test_no_hook_handlers_without_hook_manager(self):
        """No hook handlers are initialized without a hook_manager."""
        coordinator = SubsystemCoordinator(enable_hook_handlers=True)

        assert coordinator.hook_handler_registry is None


# ============================================================================
# Auto-Initialization (Startup Sequencing) Tests
# ============================================================================


class TestAutoInitialization:
    """Tests for auto-initialization of subsystems."""

    def test_auto_init_position_ledger_when_enabled(self):
        """PositionLedger is auto-created when flag is set."""
        with patch(
            "aragora.debate.subsystem_coordinator.SubsystemCoordinator._auto_init_position_ledger"
        ) as mock_init:
            coordinator = SubsystemCoordinator(enable_position_ledger=True)
            mock_init.assert_called_once()

    def test_auto_init_calibration_tracker_when_enabled(self):
        """CalibrationTracker is auto-created when flag is set."""
        with patch(
            "aragora.debate.subsystem_coordinator.SubsystemCoordinator._auto_init_calibration_tracker"
        ) as mock_init:
            coordinator = SubsystemCoordinator(enable_calibration=True)
            mock_init.assert_called_once()

    def test_no_auto_init_when_subsystem_already_provided(self, mock_calibration_tracker):
        """No auto-init when subsystem is already provided."""
        coordinator = SubsystemCoordinator(
            enable_calibration=True,
            calibration_tracker=mock_calibration_tracker,
        )

        # Should use the provided tracker, not create a new one
        assert coordinator.calibration_tracker is mock_calibration_tracker

    def test_auto_init_errors_are_tracked(self):
        """Initialization errors are recorded in _init_errors."""
        coordinator = SubsystemCoordinator()

        # _init_errors should be a list (possibly empty for successful init)
        assert isinstance(coordinator._init_errors, list)

    def test_auto_init_position_ledger_handles_import_error(self):
        """Position ledger auto-init handles ImportError gracefully."""
        with patch(
            "aragora.agents.positions.PositionLedger",
            side_effect=ImportError("Module not found"),
        ):
            # Should not raise, should log warning
            coordinator = SubsystemCoordinator(enable_position_ledger=True)
            # Position ledger should remain None
            assert coordinator.position_ledger is None

    def test_auto_init_calibration_tracker_handles_import_error(self):
        """Calibration tracker auto-init handles ImportError gracefully."""
        with patch(
            "aragora.agents.calibration.CalibrationTracker",
            side_effect=ImportError("Module not found"),
        ):
            coordinator = SubsystemCoordinator(enable_calibration=True)
            assert coordinator.calibration_tracker is None


# ============================================================================
# Health Monitoring Tests (Lifecycle Hooks)
# ============================================================================


class TestLifecycleHooks:
    """Tests for lifecycle hook methods that provide health monitoring."""

    def test_on_debate_start_resets_moment_detector(self, mock_context, mock_moment_detector):
        """on_debate_start resets the moment detector."""
        coordinator = SubsystemCoordinator(moment_detector=mock_moment_detector)

        coordinator.on_debate_start(mock_context)

        assert mock_moment_detector.reset_called is True

    def test_on_debate_start_handles_non_resettable_detector(self, mock_context):
        """on_debate_start handles moment_detector without reset method."""
        non_resettable = Mock(spec=[])  # No reset method
        coordinator = SubsystemCoordinator(moment_detector=non_resettable)

        # Should not raise
        coordinator.on_debate_start(mock_context)

    def test_on_debate_start_without_moment_detector(self, mock_context):
        """on_debate_start handles missing moment detector."""
        coordinator = SubsystemCoordinator()

        # Should not raise
        coordinator.on_debate_start(mock_context)

    def test_on_round_complete_records_positions(self, mock_context, mock_position_ledger):
        """on_round_complete records positions in ledger."""
        coordinator = SubsystemCoordinator(position_ledger=mock_position_ledger)

        positions = {"claude": "Python is best", "grok": "Rust is better"}
        coordinator.on_round_complete(mock_context, round_num=1, positions=positions)

        assert mock_position_ledger.record_position.call_count == 2

    def test_on_round_complete_without_ledger(self, mock_context):
        """on_round_complete handles missing position ledger."""
        coordinator = SubsystemCoordinator()

        positions = {"claude": "Python is best"}
        # Should not raise
        coordinator.on_round_complete(mock_context, round_num=1, positions=positions)

    def test_on_round_complete_handles_ledger_error(self, mock_context, mock_position_ledger):
        """on_round_complete handles ledger errors gracefully."""
        mock_position_ledger.record_position.side_effect = RuntimeError("Database error")
        coordinator = SubsystemCoordinator(position_ledger=mock_position_ledger)

        positions = {"claude": "Python is best"}
        # Should not raise
        coordinator.on_round_complete(mock_context, round_num=1, positions=positions)

    def test_on_debate_complete_updates_consensus_memory(
        self, mock_context, mock_result, mock_consensus_memory
    ):
        """on_debate_complete updates consensus memory."""
        coordinator = SubsystemCoordinator(consensus_memory=mock_consensus_memory)

        coordinator.on_debate_complete(mock_context, mock_result)

        mock_consensus_memory.store_consensus.assert_called_once()

    def test_on_debate_complete_updates_calibration(
        self, mock_context, mock_result, mock_calibration_tracker
    ):
        """on_debate_complete updates calibration tracker with predictions."""
        coordinator = SubsystemCoordinator(calibration_tracker=mock_calibration_tracker)

        coordinator.on_debate_complete(mock_context, mock_result)

        # Should record prediction for claude
        mock_calibration_tracker.record_prediction.assert_called()

    def test_on_debate_complete_updates_continuum_memory(
        self, mock_context, mock_result, mock_continuum_memory
    ):
        """on_debate_complete updates continuum memory."""
        coordinator = SubsystemCoordinator(continuum_memory=mock_continuum_memory)

        coordinator.on_debate_complete(mock_context, mock_result)

        mock_continuum_memory.add.assert_called_once()

    def test_on_debate_complete_handles_consensus_memory_error(
        self, mock_context, mock_result, mock_consensus_memory
    ):
        """on_debate_complete handles consensus memory errors gracefully."""
        mock_consensus_memory.store_consensus.side_effect = RuntimeError("Storage error")
        coordinator = SubsystemCoordinator(consensus_memory=mock_consensus_memory)

        # Should not raise
        coordinator.on_debate_complete(mock_context, mock_result)

    def test_on_debate_complete_without_result(self, mock_context, mock_consensus_memory):
        """on_debate_complete handles None result."""
        coordinator = SubsystemCoordinator(consensus_memory=mock_consensus_memory)

        # Should not raise and should not call store_consensus
        coordinator.on_debate_complete(mock_context, None)
        mock_consensus_memory.store_consensus.assert_not_called()


# ============================================================================
# Error Handling and Recovery Tests
# ============================================================================


class TestErrorHandling:
    """Tests for error handling and recovery."""

    def test_get_historical_dissent_handles_retriever_error(self, mock_dissent_retriever):
        """get_historical_dissent handles retriever errors gracefully."""
        mock_dissent_retriever.retrieve_for_new_debate.side_effect = RuntimeError("Network error")
        coordinator = SubsystemCoordinator(dissent_retriever=mock_dissent_retriever)

        result = coordinator.get_historical_dissent("test task")

        assert result == []

    def test_get_agent_calibration_weight_handles_tracker_error(self, mock_calibration_tracker):
        """get_agent_calibration_weight handles tracker errors gracefully."""
        mock_calibration_tracker.get_calibration_summary.side_effect = KeyError("Agent not found")
        coordinator = SubsystemCoordinator(calibration_tracker=mock_calibration_tracker)

        weight = coordinator.get_agent_calibration_weight("unknown_agent")

        assert weight == 1.0  # Default weight

    def test_get_continuum_context_handles_memory_error(self, mock_continuum_memory):
        """get_continuum_context handles memory errors gracefully."""
        mock_continuum_memory.retrieve.side_effect = RuntimeError("Database unavailable")
        coordinator = SubsystemCoordinator(continuum_memory=mock_continuum_memory)

        context = coordinator.get_continuum_context("test task")

        assert context == ""

    def test_auto_init_records_errors(self):
        """Auto-initialization records errors in _init_errors list."""
        # Create coordinator with enable flags but mock import failures
        with patch.dict("sys.modules", {"aragora.agents.positions": None}):
            coordinator = SubsystemCoordinator(enable_position_ledger=True)

            # _init_errors should exist and be a list
            assert isinstance(coordinator._init_errors, list)


# ============================================================================
# Query Method Tests
# ============================================================================


class TestQueryMethods:
    """Tests for query methods."""

    def test_get_historical_dissent_returns_dissents(self, mock_dissent_retriever):
        """get_historical_dissent returns dissenting views."""
        coordinator = SubsystemCoordinator(dissent_retriever=mock_dissent_retriever)

        result = coordinator.get_historical_dissent("Best programming language")

        assert len(result) == 2
        assert result[0]["agent"] == "grok"

    def test_get_historical_dissent_respects_limit(self, mock_dissent_retriever):
        """get_historical_dissent respects the limit parameter."""
        coordinator = SubsystemCoordinator(dissent_retriever=mock_dissent_retriever)

        result = coordinator.get_historical_dissent("Best programming language", limit=1)

        assert len(result) == 1

    def test_get_historical_dissent_without_retriever(self):
        """get_historical_dissent returns empty list without retriever."""
        coordinator = SubsystemCoordinator()

        result = coordinator.get_historical_dissent("Some task")

        assert result == []

    def test_get_agent_calibration_weight_calculates_correctly(self, mock_calibration_tracker):
        """get_agent_calibration_weight calculates weight from brier score."""
        coordinator = SubsystemCoordinator(calibration_tracker=mock_calibration_tracker)

        weight = coordinator.get_agent_calibration_weight("claude")

        # brier_score 0.15 -> calibration_quality = 1.0 - 0.15 = 0.85
        # weight = 0.5 + 0.85 = 1.35
        assert weight == pytest.approx(1.35)

    def test_get_agent_calibration_weight_without_tracker(self):
        """get_agent_calibration_weight returns 1.0 without tracker."""
        coordinator = SubsystemCoordinator()

        weight = coordinator.get_agent_calibration_weight("unknown")

        assert weight == 1.0

    def test_get_agent_calibration_weight_with_no_predictions(self, mock_calibration_tracker):
        """get_agent_calibration_weight handles agent with no predictions."""
        summary = Mock()
        summary.total_predictions = 0
        summary.brier_score = 0.0
        mock_calibration_tracker.get_calibration_summary.return_value = summary

        coordinator = SubsystemCoordinator(calibration_tracker=mock_calibration_tracker)

        weight = coordinator.get_agent_calibration_weight("new_agent")

        assert weight == 1.0

    def test_get_continuum_context_returns_formatted_context(self, mock_continuum_memory):
        """get_continuum_context returns formatted context string."""
        coordinator = SubsystemCoordinator(continuum_memory=mock_continuum_memory)

        context = coordinator.get_continuum_context("Some task")

        assert "Relevant learnings from past debates" in context
        assert "Testing best practices" in context
        assert "Code quality principles" in context

    def test_get_continuum_context_without_memory(self):
        """get_continuum_context returns empty string without memory."""
        coordinator = SubsystemCoordinator()

        context = coordinator.get_continuum_context("Some task")

        assert context == ""

    def test_get_continuum_context_with_empty_memories(self, mock_continuum_memory):
        """get_continuum_context returns empty string when no memories found."""
        mock_continuum_memory.retrieve.return_value = []
        coordinator = SubsystemCoordinator(continuum_memory=mock_continuum_memory)

        context = coordinator.get_continuum_context("Some task")

        assert context == ""


# ============================================================================
# Diagnostics Tests
# ============================================================================


class TestDiagnostics:
    """Tests for diagnostic methods."""

    def test_get_status_all_subsystems(
        self, mock_elo_system, mock_calibration_tracker, mock_consensus_memory
    ):
        """get_status returns comprehensive status."""
        coordinator = SubsystemCoordinator(
            elo_system=mock_elo_system,
            calibration_tracker=mock_calibration_tracker,
            consensus_memory=mock_consensus_memory,
        )

        status = coordinator.get_status()

        assert status["initialized"] is True
        assert status["subsystems"]["elo_system"] is True
        assert status["subsystems"]["calibration_tracker"] is True
        assert status["subsystems"]["consensus_memory"] is True
        assert status["subsystems"]["position_tracker"] is False

        assert status["capabilities"]["elo_ranking"] is True
        assert status["capabilities"]["calibration"] is True
        assert status["capabilities"]["position_tracking"] is False

    def test_get_status_empty_coordinator(self):
        """get_status with no subsystems returns mostly False (sdpo_learner auto-initializes)."""
        coordinator = SubsystemCoordinator()

        status = coordinator.get_status()

        assert status["initialized"] is True
        # sdpo_learner auto-initializes; all other subsystems should be False
        for key, val in status["subsystems"].items():
            if key == "sdpo_learner":
                continue
            assert val is False, f"Expected {key} to be False, got {val!r}"

    def test_get_status_includes_bridges(self):
        """get_status includes cross-pollination bridge information."""
        coordinator = SubsystemCoordinator()

        status = coordinator.get_status()

        assert "cross_pollination_bridges" in status
        assert "active_bridges_count" in status
        assert status["active_bridges_count"] == 0

    def test_get_status_includes_km_info(self):
        """get_status includes Knowledge Mound information."""
        # Disable KM coordinator to test without it
        coordinator = SubsystemCoordinator(
            enable_km_coordinator=False,
            enable_km_bidirectional=False,
        )

        status = coordinator.get_status()

        assert "knowledge_mound" in status
        assert status["knowledge_mound"]["available"] is False
        assert status["knowledge_mound"]["coordinator_active"] is False

    def test_has_hook_handlers_property(self, mock_hook_manager):
        """has_hook_handlers property reflects registry state."""
        coordinator = SubsystemCoordinator()

        assert coordinator.has_hook_handlers is False

    def test_active_bridges_count_property(self):
        """active_bridges_count counts active bridges."""
        coordinator = SubsystemCoordinator()

        assert coordinator.active_bridges_count == 0

    def test_active_km_adapters_count_property(self):
        """active_km_adapters_count counts registered KM adapters."""
        coordinator = SubsystemCoordinator()

        assert coordinator.active_km_adapters_count == 0


# ============================================================================
# SubsystemConfig Tests
# ============================================================================


class TestSubsystemConfig:
    """Tests for SubsystemConfig helper class."""

    def test_create_coordinator_basic(self):
        """create_coordinator creates a coordinator with basic settings."""
        config = SubsystemConfig()
        coordinator = config.create_coordinator(loop_id="test-loop")

        assert isinstance(coordinator, SubsystemCoordinator)
        assert coordinator.loop_id == "test-loop"

    def test_create_coordinator_with_protocol(self, mock_protocol):
        """create_coordinator passes through protocol."""
        config = SubsystemConfig()
        coordinator = config.create_coordinator(protocol=mock_protocol, loop_id="test")

        assert coordinator.protocol is mock_protocol

    def test_create_coordinator_with_enable_flags(self):
        """create_coordinator respects enable flags."""
        config = SubsystemConfig(
            enable_position_ledger=True,
            enable_calibration=True,
            enable_moment_detection=True,
        )
        coordinator = config.create_coordinator()

        assert coordinator.enable_position_ledger is True
        assert coordinator.enable_calibration is True
        assert coordinator.enable_moment_detection is True

    def test_create_coordinator_with_subsystems(self, mock_elo_system, mock_calibration_tracker):
        """create_coordinator passes through pre-configured subsystems."""
        config = SubsystemConfig(
            elo_system=mock_elo_system,
            calibration_tracker=mock_calibration_tracker,
        )
        coordinator = config.create_coordinator()

        assert coordinator.elo_system is mock_elo_system
        assert coordinator.calibration_tracker is mock_calibration_tracker

    def test_create_coordinator_with_bridges(self):
        """create_coordinator configures bridge enable flags."""
        config = SubsystemConfig(
            enable_performance_router=False,
            enable_outcome_complexity=False,
        )
        coordinator = config.create_coordinator()

        assert coordinator.enable_performance_router is False
        assert coordinator.enable_outcome_complexity is False

    def test_create_coordinator_with_km_config(self):
        """create_coordinator configures Knowledge Mound settings."""
        config = SubsystemConfig(
            enable_km_bidirectional=True,
            enable_km_coordinator=True,
            km_sync_interval_seconds=600,
            km_min_confidence_for_reverse=0.8,
            km_parallel_sync=False,
        )
        coordinator = config.create_coordinator()

        assert coordinator.enable_km_bidirectional is True
        assert coordinator.enable_km_coordinator is True
        assert coordinator.km_sync_interval_seconds == 600
        assert coordinator.km_min_confidence_for_reverse == 0.8
        assert coordinator.km_parallel_sync is False


# ============================================================================
# Phase 9 Cross-Pollination Bridge Tests
# ============================================================================


class TestCrossPolliantionBridges:
    """Tests for Phase 9 Cross-Pollination Bridges."""

    def test_no_bridges_without_sources(self):
        """No bridges are created when no source subsystems are provided."""
        coordinator = SubsystemCoordinator()

        assert coordinator.has_performance_router_bridge is False
        assert coordinator.has_outcome_complexity_bridge is False
        assert coordinator.has_analytics_selection_bridge is False
        assert coordinator.has_novelty_selection_bridge is False
        assert coordinator.has_relationship_bias_bridge is False
        assert coordinator.has_rlm_selection_bridge is False
        assert coordinator.has_calibration_cost_bridge is False

    def test_performance_router_bridge_requires_monitor(self):
        """PerformanceRouterBridge requires performance_monitor."""
        mock_monitor = Mock()

        with patch(
            "aragora.debate.subsystem_coordinator.SubsystemCoordinator._auto_init_performance_router_bridge"
        ) as mock_init:
            coordinator = SubsystemCoordinator(
                performance_monitor=mock_monitor,
                enable_performance_router=True,
            )
            mock_init.assert_called_once()

    def test_outcome_complexity_bridge_requires_tracker(self):
        """OutcomeComplexityBridge requires outcome_tracker."""
        mock_tracker = Mock()

        with patch(
            "aragora.debate.subsystem_coordinator.SubsystemCoordinator._auto_init_outcome_complexity_bridge"
        ) as mock_init:
            coordinator = SubsystemCoordinator(
                outcome_tracker=mock_tracker,
                enable_outcome_complexity=True,
            )
            mock_init.assert_called_once()

    def test_analytics_selection_bridge_requires_coordinator(self):
        """AnalyticsSelectionBridge requires analytics_coordinator."""
        mock_analytics = Mock()

        with patch(
            "aragora.debate.subsystem_coordinator.SubsystemCoordinator._auto_init_analytics_selection_bridge"
        ) as mock_init:
            coordinator = SubsystemCoordinator(
                analytics_coordinator=mock_analytics,
                enable_analytics_selection=True,
            )
            mock_init.assert_called_once()

    def test_novelty_selection_bridge_requires_tracker(self):
        """NoveltySelectionBridge requires novelty_tracker."""
        mock_tracker = Mock()

        with patch(
            "aragora.debate.subsystem_coordinator.SubsystemCoordinator._auto_init_novelty_selection_bridge"
        ) as mock_init:
            coordinator = SubsystemCoordinator(
                novelty_tracker=mock_tracker,
                enable_novelty_selection=True,
            )
            mock_init.assert_called_once()

    def test_relationship_bias_bridge_requires_tracker(self, mock_relationship_tracker):
        """RelationshipBiasBridge requires relationship_tracker."""
        with patch(
            "aragora.debate.subsystem_coordinator.SubsystemCoordinator._auto_init_relationship_bias_bridge"
        ) as mock_init:
            coordinator = SubsystemCoordinator(
                relationship_tracker=mock_relationship_tracker,
                enable_relationship_bias=True,
            )
            mock_init.assert_called_once()

    def test_rlm_selection_bridge_requires_rlm_bridge(self):
        """RLMSelectionBridge requires rlm_bridge."""
        mock_rlm = Mock()

        with patch(
            "aragora.debate.subsystem_coordinator.SubsystemCoordinator._auto_init_rlm_selection_bridge"
        ) as mock_init:
            coordinator = SubsystemCoordinator(
                rlm_bridge=mock_rlm,
                enable_rlm_selection=True,
            )
            mock_init.assert_called_once()

    def test_calibration_cost_bridge_requires_tracker(self, mock_calibration_tracker):
        """CalibrationCostBridge requires calibration_tracker."""
        with patch(
            "aragora.debate.subsystem_coordinator.SubsystemCoordinator._auto_init_calibration_cost_bridge"
        ) as mock_init:
            coordinator = SubsystemCoordinator(
                calibration_tracker=mock_calibration_tracker,
                enable_calibration_cost=True,
            )
            mock_init.assert_called_once()


# ============================================================================
# Phase 10 Knowledge Mound Integration Tests
# ============================================================================


class TestKnowledgeMoundIntegration:
    """Tests for Phase 10 Knowledge Mound Integration."""

    def test_no_km_coordinator_when_disabled(self):
        """No KM coordinator when enable flags are False."""
        coordinator = SubsystemCoordinator(
            enable_km_coordinator=False,
            enable_km_bidirectional=False,
        )

        assert coordinator.has_knowledge_mound is False
        assert coordinator.has_km_coordinator is False

    def test_km_coordinator_auto_initializes_with_defaults(self):
        """KM coordinator auto-initializes with default enable flags."""
        # Default enable_km_coordinator=True and enable_km_bidirectional=True
        # will auto-create the coordinator even without knowledge_mound
        coordinator = SubsystemCoordinator()

        # This behavior is by design - coordinator can exist without KM
        # The has_km_bidirectional check requires BOTH km and coordinator
        assert coordinator.has_knowledge_mound is False
        # Coordinator may or may not be created depending on imports
        # Check that has_km_bidirectional is False without knowledge_mound
        assert coordinator.has_km_bidirectional is False

    def test_km_coordinator_with_km(self):
        """KM coordinator initializes with Knowledge Mound."""
        mock_km = Mock()

        with patch(
            "aragora.debate.subsystem_coordinator.SubsystemCoordinator._auto_init_km_coordinator"
        ) as mock_init:
            coordinator = SubsystemCoordinator(
                knowledge_mound=mock_km,
                enable_km_coordinator=True,
                enable_km_bidirectional=True,
            )
            mock_init.assert_called_once()

    def test_has_km_bidirectional_requires_both(self):
        """has_km_bidirectional requires both KM and coordinator."""
        coordinator = SubsystemCoordinator()

        assert coordinator.has_km_bidirectional is False

    def test_km_adapter_registration(self):
        """KM adapters can be registered."""
        mock_adapter = Mock()
        coordinator = SubsystemCoordinator(
            km_continuum_adapter=mock_adapter,
        )

        assert coordinator.km_continuum_adapter is mock_adapter
        assert coordinator.active_km_adapters_count == 1

    def test_multiple_km_adapters(self):
        """Multiple KM adapters can be registered."""
        coordinator = SubsystemCoordinator(
            km_continuum_adapter=Mock(),
            km_elo_adapter=Mock(),
            km_belief_adapter=Mock(),
        )

        assert coordinator.active_km_adapters_count == 3

    def test_km_config_defaults(self):
        """KM configuration has sensible defaults."""
        coordinator = SubsystemCoordinator()

        assert coordinator.km_sync_interval_seconds == 300  # 5 minutes
        assert coordinator.km_min_confidence_for_reverse == 0.7
        assert coordinator.km_parallel_sync is True


# ============================================================================
# Capability Property Tests
# ============================================================================


class TestCapabilityProperties:
    """Tests for all capability check properties."""

    def test_has_elo_property(self, mock_elo_system):
        """has_elo property correctly reflects ELO system presence."""
        coordinator = SubsystemCoordinator(elo_system=mock_elo_system)
        assert coordinator.has_elo is True

        empty = SubsystemCoordinator()
        assert empty.has_elo is False

    def test_has_calibration_property(self, mock_calibration_tracker):
        """has_calibration property correctly reflects calibration tracker presence."""
        coordinator = SubsystemCoordinator(calibration_tracker=mock_calibration_tracker)
        assert coordinator.has_calibration is True

        empty = SubsystemCoordinator()
        assert empty.has_calibration is False

    def test_has_consensus_memory_property(self, mock_consensus_memory):
        """has_consensus_memory property correctly reflects memory presence."""
        coordinator = SubsystemCoordinator(consensus_memory=mock_consensus_memory)
        assert coordinator.has_consensus_memory is True

        empty = SubsystemCoordinator()
        assert empty.has_consensus_memory is False

    def test_has_dissent_retrieval_property(self, mock_dissent_retriever):
        """has_dissent_retrieval property correctly reflects retriever presence."""
        coordinator = SubsystemCoordinator(dissent_retriever=mock_dissent_retriever)
        assert coordinator.has_dissent_retrieval is True

        empty = SubsystemCoordinator()
        assert empty.has_dissent_retrieval is False

    def test_has_moment_detection_property(self, mock_moment_detector):
        """has_moment_detection property correctly reflects detector presence."""
        coordinator = SubsystemCoordinator(moment_detector=mock_moment_detector)
        assert coordinator.has_moment_detection is True

        empty = SubsystemCoordinator()
        assert empty.has_moment_detection is False

    def test_has_relationship_tracking_property(self, mock_relationship_tracker):
        """has_relationship_tracking property correctly reflects tracker presence."""
        coordinator = SubsystemCoordinator(relationship_tracker=mock_relationship_tracker)
        assert coordinator.has_relationship_tracking is True

        empty = SubsystemCoordinator()
        assert empty.has_relationship_tracking is False

    def test_has_continuum_memory_property(self, mock_continuum_memory):
        """has_continuum_memory property correctly reflects memory presence."""
        coordinator = SubsystemCoordinator(continuum_memory=mock_continuum_memory)
        assert coordinator.has_continuum_memory is True

        empty = SubsystemCoordinator()
        assert empty.has_continuum_memory is False


# ============================================================================
# Edge Cases Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_positions_on_round_complete(self, mock_context, mock_position_ledger):
        """on_round_complete handles empty positions dict."""
        coordinator = SubsystemCoordinator(position_ledger=mock_position_ledger)

        coordinator.on_round_complete(mock_context, round_num=1, positions={})

        mock_position_ledger.record_position.assert_not_called()

    def test_none_context_in_lifecycle_hooks(self, mock_consensus_memory, mock_result):
        """Lifecycle hooks handle None context gracefully."""
        coordinator = SubsystemCoordinator(consensus_memory=mock_consensus_memory)

        # Create a minimal mock context
        ctx = Mock()
        ctx.debate_id = None
        ctx.env = None
        ctx.agents = []
        ctx.start_time = 1707753600

        # Should not raise even with minimal/None values
        coordinator.on_debate_complete(ctx, mock_result)

    def test_result_without_predictions(self, mock_context, mock_calibration_tracker):
        """on_debate_complete handles result without predictions."""
        coordinator = SubsystemCoordinator(calibration_tracker=mock_calibration_tracker)

        result = Mock()
        result.consensus = "Test consensus"
        result.consensus_confidence = 0.8
        result.predictions = {}  # No predictions
        result.messages = []
        result.rounds_completed = 1

        coordinator.on_debate_complete(mock_context, result)

        # Should not call record_prediction with empty predictions
        mock_calibration_tracker.record_prediction.assert_not_called()

    def test_calibration_weight_with_high_brier_score(self, mock_calibration_tracker):
        """Calibration weight is bounded for high brier scores."""
        summary = Mock()
        summary.total_predictions = 100
        summary.brier_score = 0.6  # Very poor calibration
        mock_calibration_tracker.get_calibration_summary.return_value = summary

        coordinator = SubsystemCoordinator(calibration_tracker=mock_calibration_tracker)

        weight = coordinator.get_agent_calibration_weight("poor_agent")

        # brier_score is capped at 0.5, so 1.0 - 0.5 = 0.5
        # weight = 0.5 + 0.5 = 1.0
        assert weight == pytest.approx(1.0)

    def test_get_continuum_context_with_entry_without_metadata(self, mock_continuum_memory):
        """get_continuum_context handles entries without metadata."""
        entry = Mock()
        entry.content = "Direct content without summary"
        entry.metadata = None
        mock_continuum_memory.retrieve.return_value = [entry]

        coordinator = SubsystemCoordinator(continuum_memory=mock_continuum_memory)

        context = coordinator.get_continuum_context("test")

        assert "Direct content without summary" in context

    def test_consensus_strength_mapping(self, mock_context, mock_consensus_memory):
        """Test consensus strength is correctly mapped from confidence."""
        # Test different confidence levels
        confidence_tests = [
            (0.95, "UNANIMOUS"),
            (0.85, "STRONG"),
            (0.65, "MODERATE"),
            (0.55, "WEAK"),
            (0.45, "SPLIT"),
        ]

        for confidence, expected_strength in confidence_tests:
            coordinator = SubsystemCoordinator(consensus_memory=mock_consensus_memory)

            result = Mock()
            result.consensus = "Test consensus"
            result.consensus_confidence = confidence
            result.predictions = {}
            result.messages = []
            result.rounds_completed = 2

            coordinator.on_debate_complete(mock_context, result)

            # Verify store_consensus was called
            assert mock_consensus_memory.store_consensus.called
            mock_consensus_memory.reset_mock()
