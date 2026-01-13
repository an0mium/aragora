"""
Tests for SubsystemCoordinator.

Tests cover:
- SubsystemCoordinator initialization
- Auto-initialization of subsystems
- Capability property accessors
- Lifecycle hooks (on_debate_start, on_round_complete, on_debate_complete)
- Query methods (get_historical_dissent, get_agent_calibration_weight)
- Status/diagnostics
"""

from __future__ import annotations

import pytest
from dataclasses import dataclass
from unittest.mock import Mock, MagicMock, patch

from aragora.debate.subsystem_coordinator import (
    SubsystemCoordinator,
    SubsystemConfig,
)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_elo_system():
    """Create a mock ELO system."""
    elo = Mock()
    elo.get_rating.return_value = 1500.0
    return elo


@pytest.fixture
def mock_calibration_tracker():
    """Create a mock calibration tracker."""
    tracker = Mock()
    tracker.get_agent_stats.return_value = {"calibration_score": 0.85}
    return tracker


@pytest.fixture
def mock_consensus_memory():
    """Create a mock consensus memory."""
    memory = Mock()
    memory.store_outcome = Mock()
    return memory


@pytest.fixture
def mock_continuum_memory():
    """Create a mock continuum memory."""
    memory = Mock()
    memory.retrieve.return_value = [
        {"summary": "Past debate learned X"},
        {"summary": "Another debate learned Y"},
    ]
    memory.store_debate_outcome = Mock()
    return memory


@pytest.fixture
def mock_context():
    """Create a mock debate context."""
    ctx = Mock()
    ctx.debate_id = "test-debate-123"
    ctx.task = "Discuss the best programming language"
    return ctx


@pytest.fixture
def mock_result():
    """Create a mock debate result."""
    result = Mock()
    result.consensus = "Python is versatile"
    result.consensus_confidence = 0.85
    result.votes = [{"agent": "claude", "choice": "python"}]
    result.dissenting_views = [{"agent": "grok", "view": "Rust is better"}]
    result.predictions = {}
    return result


# ============================================================================
# Initialization Tests
# ============================================================================


class TestSubsystemCoordinatorInit:
    """Tests for SubsystemCoordinator initialization."""

    def test_empty_init(self):
        """Test coordinator can be created with no subsystems."""
        coordinator = SubsystemCoordinator()

        assert coordinator._initialized is True
        assert coordinator.position_tracker is None
        assert coordinator.elo_system is None

    def test_init_with_subsystems(self, mock_elo_system, mock_calibration_tracker):
        """Test coordinator accepts pre-configured subsystems."""
        coordinator = SubsystemCoordinator(
            elo_system=mock_elo_system,
            calibration_tracker=mock_calibration_tracker,
        )

        assert coordinator.elo_system is mock_elo_system
        assert coordinator.calibration_tracker is mock_calibration_tracker

    def test_init_with_loop_id(self):
        """Test coordinator stores loop_id."""
        coordinator = SubsystemCoordinator(loop_id="test-loop-456")

        assert coordinator.loop_id == "test-loop-456"


# ============================================================================
# Auto-Initialization Tests
# ============================================================================


class TestAutoInitialization:
    """Tests for auto-initialization of subsystems."""

    def test_auto_init_position_ledger_when_enabled(self):
        """Test PositionLedger is auto-created when flag is set."""
        with patch(
            "aragora.debate.subsystem_coordinator.SubsystemCoordinator._auto_init_position_ledger"
        ) as mock_init:
            coordinator = SubsystemCoordinator(enable_position_ledger=True)
            mock_init.assert_called_once()

    def test_auto_init_calibration_tracker_when_enabled(self):
        """Test CalibrationTracker is auto-created when flag is set."""
        with patch(
            "aragora.debate.subsystem_coordinator.SubsystemCoordinator._auto_init_calibration_tracker"
        ) as mock_init:
            coordinator = SubsystemCoordinator(enable_calibration=True)
            mock_init.assert_called_once()

    def test_auto_init_dissent_retriever_with_consensus_memory(self, mock_consensus_memory):
        """Test DissentRetriever is auto-created when consensus_memory exists."""
        with patch(
            "aragora.debate.subsystem_coordinator.SubsystemCoordinator._auto_init_dissent_retriever"
        ) as mock_init:
            coordinator = SubsystemCoordinator(consensus_memory=mock_consensus_memory)
            mock_init.assert_called_once()

    def test_no_auto_init_when_subsystem_provided(self, mock_calibration_tracker):
        """Test no auto-init when subsystem is already provided."""
        coordinator = SubsystemCoordinator(
            enable_calibration=True,
            calibration_tracker=mock_calibration_tracker,  # Already provided
        )

        # Should use the provided tracker, not create a new one
        assert coordinator.calibration_tracker is mock_calibration_tracker

    def test_init_errors_are_tracked(self):
        """Test initialization errors are recorded."""
        coordinator = SubsystemCoordinator()
        # init_errors should be empty for successful init
        assert isinstance(coordinator._init_errors, list)


# ============================================================================
# Capability Property Tests
# ============================================================================


class TestCapabilityProperties:
    """Tests for capability check properties."""

    def test_has_position_tracking_with_tracker(self):
        """Test has_position_tracking when tracker is set."""
        coordinator = SubsystemCoordinator(position_tracker=Mock())
        assert coordinator.has_position_tracking is True

    def test_has_position_tracking_with_ledger(self):
        """Test has_position_tracking when ledger is set."""
        coordinator = SubsystemCoordinator(position_ledger=Mock())
        assert coordinator.has_position_tracking is True

    def test_has_position_tracking_false(self):
        """Test has_position_tracking when neither is set."""
        coordinator = SubsystemCoordinator()
        assert coordinator.has_position_tracking is False

    def test_has_elo(self, mock_elo_system):
        """Test has_elo property."""
        coordinator = SubsystemCoordinator(elo_system=mock_elo_system)
        assert coordinator.has_elo is True

        empty_coordinator = SubsystemCoordinator()
        assert empty_coordinator.has_elo is False

    def test_has_calibration(self, mock_calibration_tracker):
        """Test has_calibration property."""
        coordinator = SubsystemCoordinator(calibration_tracker=mock_calibration_tracker)
        assert coordinator.has_calibration is True

    def test_has_consensus_memory(self, mock_consensus_memory):
        """Test has_consensus_memory property."""
        coordinator = SubsystemCoordinator(consensus_memory=mock_consensus_memory)
        assert coordinator.has_consensus_memory is True

    def test_has_relationship_tracking(self):
        """Test has_relationship_tracking property."""
        coordinator = SubsystemCoordinator(relationship_tracker=Mock())
        assert coordinator.has_relationship_tracking is True


# ============================================================================
# Lifecycle Hook Tests
# ============================================================================


class TestLifecycleHooks:
    """Tests for lifecycle hook methods."""

    def test_on_debate_start_resets_moment_detector(self, mock_context):
        """Test on_debate_start resets moment detector."""
        moment_detector = Mock()
        coordinator = SubsystemCoordinator(moment_detector=moment_detector)

        coordinator.on_debate_start(mock_context)

        moment_detector.reset.assert_called_once()

    def test_on_debate_start_without_moment_detector(self, mock_context):
        """Test on_debate_start handles missing moment detector."""
        coordinator = SubsystemCoordinator()
        # Should not raise
        coordinator.on_debate_start(mock_context)

    def test_on_round_complete_records_positions(self, mock_context):
        """Test on_round_complete records positions in ledger."""
        position_ledger = Mock()
        coordinator = SubsystemCoordinator(position_ledger=position_ledger)

        positions = {"claude": "Python is best", "grok": "Rust is better"}
        coordinator.on_round_complete(mock_context, round_num=1, positions=positions)

        assert position_ledger.record_position.call_count == 2

    def test_on_debate_complete_updates_consensus_memory(
        self, mock_context, mock_result, mock_consensus_memory
    ):
        """Test on_debate_complete updates consensus memory."""
        coordinator = SubsystemCoordinator(consensus_memory=mock_consensus_memory)

        coordinator.on_debate_complete(mock_context, mock_result)

        mock_consensus_memory.store_outcome.assert_called_once()
        call_args = mock_consensus_memory.store_outcome.call_args
        assert call_args.kwargs["debate_id"] == mock_context.debate_id
        assert call_args.kwargs["consensus"] == mock_result.consensus

    def test_on_debate_complete_updates_continuum_memory(
        self, mock_context, mock_result, mock_continuum_memory
    ):
        """Test on_debate_complete updates continuum memory."""
        coordinator = SubsystemCoordinator(continuum_memory=mock_continuum_memory)

        coordinator.on_debate_complete(mock_context, mock_result)

        mock_continuum_memory.store_debate_outcome.assert_called_once()

    def test_on_debate_complete_handles_errors_gracefully(
        self, mock_context, mock_result, mock_consensus_memory
    ):
        """Test on_debate_complete handles subsystem errors gracefully."""
        mock_consensus_memory.store_outcome.side_effect = Exception("Database error")
        coordinator = SubsystemCoordinator(consensus_memory=mock_consensus_memory)

        # Should not raise
        coordinator.on_debate_complete(mock_context, mock_result)


# ============================================================================
# Query Method Tests
# ============================================================================


class TestQueryMethods:
    """Tests for query methods."""

    def test_get_historical_dissent(self, mock_consensus_memory):
        """Test get_historical_dissent returns dissenting views."""
        dissent_retriever = Mock()
        dissent_retriever.retrieve.return_value = [
            {"agent": "grok", "position": "Rust is better", "outcome": "minority"},
        ]
        coordinator = SubsystemCoordinator(dissent_retriever=dissent_retriever)

        result = coordinator.get_historical_dissent("Best programming language")

        assert len(result) == 1
        dissent_retriever.retrieve.assert_called_once_with("Best programming language", limit=3)

    def test_get_historical_dissent_without_retriever(self):
        """Test get_historical_dissent returns empty when no retriever."""
        coordinator = SubsystemCoordinator()

        result = coordinator.get_historical_dissent("Some task")

        assert result == []

    def test_get_agent_calibration_weight(self, mock_calibration_tracker):
        """Test get_agent_calibration_weight returns correct weight."""
        coordinator = SubsystemCoordinator(calibration_tracker=mock_calibration_tracker)

        weight = coordinator.get_agent_calibration_weight("claude")

        # calibration_score 0.85 -> weight 0.5 + 0.85 = 1.35
        assert weight == 1.35

    def test_get_agent_calibration_weight_default(self):
        """Test get_agent_calibration_weight returns 1.0 when no tracker."""
        coordinator = SubsystemCoordinator()

        weight = coordinator.get_agent_calibration_weight("unknown")

        assert weight == 1.0

    def test_get_continuum_context(self, mock_continuum_memory):
        """Test get_continuum_context returns formatted context."""
        coordinator = SubsystemCoordinator(continuum_memory=mock_continuum_memory)

        context = coordinator.get_continuum_context("Some task")

        assert "Past debate learned X" in context
        assert "Another debate learned Y" in context

    def test_get_continuum_context_without_memory(self):
        """Test get_continuum_context returns empty when no memory."""
        coordinator = SubsystemCoordinator()

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
        """Test get_status returns comprehensive status."""
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

    def test_get_status_empty(self):
        """Test get_status with no subsystems."""
        coordinator = SubsystemCoordinator()

        status = coordinator.get_status()

        assert status["initialized"] is True
        assert all(v is False for v in status["subsystems"].values())


# ============================================================================
# SubsystemConfig Tests
# ============================================================================


class TestSubsystemConfig:
    """Tests for SubsystemConfig helper class."""

    def test_create_coordinator_basic(self):
        """Test create_coordinator creates coordinator."""
        config = SubsystemConfig()
        coordinator = config.create_coordinator(loop_id="test-loop")

        assert isinstance(coordinator, SubsystemCoordinator)
        assert coordinator.loop_id == "test-loop"

    def test_create_coordinator_with_flags(self):
        """Test create_coordinator respects enable flags."""
        config = SubsystemConfig(
            enable_position_ledger=True,
            enable_calibration=True,
        )
        coordinator = config.create_coordinator()

        assert coordinator.enable_position_ledger is True
        assert coordinator.enable_calibration is True

    def test_create_coordinator_with_subsystems(self, mock_elo_system):
        """Test create_coordinator passes through subsystems."""
        config = SubsystemConfig(
            elo_system=mock_elo_system,
        )
        coordinator = config.create_coordinator()

        assert coordinator.elo_system is mock_elo_system
