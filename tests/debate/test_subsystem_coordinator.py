"""
Tests for Subsystem Coordinator module.

Tests cover:
- Resettable protocol
- SubsystemCoordinator dataclass and properties
- Auto-initialization of subsystems
- Lifecycle hooks (on_debate_start, on_round_complete, on_debate_complete)
- Query methods (get_historical_dissent, get_agent_calibration_weight, etc.)
- Cross-pollination bridges
- Knowledge Mound integration
- SubsystemConfig factory method
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from aragora.debate.subsystem_coordinator import (
    Resettable,
    SubsystemConfig,
    SubsystemCoordinator,
)


# =============================================================================
# Resettable Protocol Tests
# =============================================================================


class TestResettable:
    """Tests for Resettable protocol."""

    def test_resettable_protocol(self):
        """Test Resettable protocol compliance."""

        class MyResettable:
            def reset(self) -> None:
                pass

        obj = MyResettable()
        assert isinstance(obj, Resettable)

    def test_non_resettable(self):
        """Test non-Resettable objects."""

        class NotResettable:
            pass

        obj = NotResettable()
        assert not isinstance(obj, Resettable)


# =============================================================================
# SubsystemCoordinator Property Tests
# =============================================================================


class TestSubsystemCoordinatorProperties:
    """Tests for SubsystemCoordinator property accessors."""

    def test_has_position_tracking_with_tracker(self):
        """Test has_position_tracking with position_tracker."""
        coord = SubsystemCoordinator(position_tracker=Mock())
        assert coord.has_position_tracking is True

    def test_has_position_tracking_with_ledger(self):
        """Test has_position_tracking with position_ledger."""
        coord = SubsystemCoordinator(position_ledger=Mock())
        assert coord.has_position_tracking is True

    def test_has_position_tracking_false(self):
        """Test has_position_tracking is False when neither present."""
        coord = SubsystemCoordinator()
        assert coord.has_position_tracking is False

    def test_has_elo(self):
        """Test has_elo property."""
        coord = SubsystemCoordinator(elo_system=Mock())
        assert coord.has_elo is True

        coord_no_elo = SubsystemCoordinator()
        assert coord_no_elo.has_elo is False

    def test_has_calibration(self):
        """Test has_calibration property."""
        coord = SubsystemCoordinator(calibration_tracker=Mock())
        assert coord.has_calibration is True

        coord_no_cal = SubsystemCoordinator()
        assert coord_no_cal.has_calibration is False

    def test_has_consensus_memory(self):
        """Test has_consensus_memory property."""
        coord = SubsystemCoordinator(consensus_memory=Mock())
        assert coord.has_consensus_memory is True

    def test_has_dissent_retrieval(self):
        """Test has_dissent_retrieval property."""
        coord = SubsystemCoordinator(dissent_retriever=Mock())
        assert coord.has_dissent_retrieval is True

    def test_has_moment_detection(self):
        """Test has_moment_detection property."""
        coord = SubsystemCoordinator(moment_detector=Mock())
        assert coord.has_moment_detection is True

    def test_has_relationship_tracking(self):
        """Test has_relationship_tracking property."""
        coord = SubsystemCoordinator(relationship_tracker=Mock())
        assert coord.has_relationship_tracking is True

    def test_has_continuum_memory(self):
        """Test has_continuum_memory property."""
        coord = SubsystemCoordinator(continuum_memory=Mock())
        assert coord.has_continuum_memory is True


# =============================================================================
# SubsystemCoordinator Bridge Properties Tests
# =============================================================================


class TestSubsystemCoordinatorBridgeProperties:
    """Tests for cross-pollination bridge properties."""

    def test_has_performance_router_bridge(self):
        """Test has_performance_router_bridge property."""
        coord = SubsystemCoordinator(performance_router_bridge=Mock())
        assert coord.has_performance_router_bridge is True

        coord_no = SubsystemCoordinator()
        assert coord_no.has_performance_router_bridge is False

    def test_has_outcome_complexity_bridge(self):
        """Test has_outcome_complexity_bridge property."""
        coord = SubsystemCoordinator(outcome_complexity_bridge=Mock())
        assert coord.has_outcome_complexity_bridge is True

    def test_has_analytics_selection_bridge(self):
        """Test has_analytics_selection_bridge property."""
        coord = SubsystemCoordinator(analytics_selection_bridge=Mock())
        assert coord.has_analytics_selection_bridge is True

    def test_has_novelty_selection_bridge(self):
        """Test has_novelty_selection_bridge property."""
        coord = SubsystemCoordinator(novelty_selection_bridge=Mock())
        assert coord.has_novelty_selection_bridge is True

    def test_has_relationship_bias_bridge(self):
        """Test has_relationship_bias_bridge property."""
        coord = SubsystemCoordinator(relationship_bias_bridge=Mock())
        assert coord.has_relationship_bias_bridge is True

    def test_has_rlm_selection_bridge(self):
        """Test has_rlm_selection_bridge property."""
        coord = SubsystemCoordinator(rlm_selection_bridge=Mock())
        assert coord.has_rlm_selection_bridge is True

    def test_has_calibration_cost_bridge(self):
        """Test has_calibration_cost_bridge property."""
        coord = SubsystemCoordinator(calibration_cost_bridge=Mock())
        assert coord.has_calibration_cost_bridge is True

    def test_active_bridges_count(self):
        """Test active_bridges_count property."""
        coord = SubsystemCoordinator(
            performance_router_bridge=Mock(),
            outcome_complexity_bridge=Mock(),
            analytics_selection_bridge=Mock(),
        )
        assert coord.active_bridges_count == 3


# =============================================================================
# Knowledge Mound Properties Tests
# =============================================================================


class TestSubsystemCoordinatorKMProperties:
    """Tests for Knowledge Mound related properties."""

    def test_has_knowledge_mound(self):
        """Test has_knowledge_mound property."""
        coord = SubsystemCoordinator(knowledge_mound=Mock())
        assert coord.has_knowledge_mound is True

    def test_has_km_coordinator(self):
        """Test has_km_coordinator property."""
        coord = SubsystemCoordinator(km_coordinator=Mock())
        assert coord.has_km_coordinator is True

    def test_has_km_bidirectional(self):
        """Test has_km_bidirectional property."""
        # Both required
        coord = SubsystemCoordinator(
            knowledge_mound=Mock(),
            km_coordinator=Mock(),
        )
        assert coord.has_km_bidirectional is True

        # Only one (disable auto-init of coordinator)
        coord_partial = SubsystemCoordinator(
            knowledge_mound=Mock(), enable_km_coordinator=False
        )
        assert coord_partial.has_km_bidirectional is False

    def test_active_km_adapters_count(self):
        """Test active_km_adapters_count property."""
        coord = SubsystemCoordinator(
            km_continuum_adapter=Mock(),
            km_elo_adapter=Mock(),
            km_belief_adapter=Mock(),
        )
        assert coord.active_km_adapters_count == 3


# =============================================================================
# Auto-Initialization Tests
# =============================================================================


class TestSubsystemCoordinatorAutoInit:
    """Tests for auto-initialization of subsystems."""

    def test_auto_init_position_ledger(self):
        """Test auto-initialization of PositionLedger."""
        with patch(
            "aragora.debate.subsystem_coordinator.SubsystemCoordinator._auto_init_position_ledger"
        ) as mock_init:
            coord = SubsystemCoordinator(enable_position_ledger=True)
            mock_init.assert_called_once()

    def test_auto_init_calibration_tracker(self):
        """Test auto-initialization of CalibrationTracker."""
        with patch(
            "aragora.debate.subsystem_coordinator.SubsystemCoordinator._auto_init_calibration_tracker"
        ) as mock_init:
            coord = SubsystemCoordinator(enable_calibration=True)
            mock_init.assert_called_once()

    def test_auto_init_dissent_retriever(self):
        """Test auto-initialization of DissentRetriever when consensus_memory present."""
        mock_consensus = Mock()

        with patch(
            "aragora.debate.subsystem_coordinator.SubsystemCoordinator._auto_init_dissent_retriever"
        ) as mock_init:
            coord = SubsystemCoordinator(consensus_memory=mock_consensus)
            mock_init.assert_called_once()

    def test_auto_init_moment_detector(self):
        """Test auto-initialization of MomentDetector."""
        with patch(
            "aragora.debate.subsystem_coordinator.SubsystemCoordinator._auto_init_moment_detector"
        ) as mock_init:
            coord = SubsystemCoordinator(enable_moment_detection=True)
            mock_init.assert_called_once()

    def test_auto_init_hook_handlers(self):
        """Test auto-initialization of HookHandlerRegistry."""
        mock_hook_manager = Mock()

        with patch(
            "aragora.debate.subsystem_coordinator.SubsystemCoordinator._auto_init_hook_handlers"
        ) as mock_init:
            coord = SubsystemCoordinator(
                hook_manager=mock_hook_manager,
                enable_hook_handlers=True,
            )
            mock_init.assert_called_once()

    def test_auto_init_bridges_with_sources(self):
        """Test auto-initialization of bridges when sources available."""
        with patch(
            "aragora.debate.subsystem_coordinator.SubsystemCoordinator._auto_init_performance_router_bridge"
        ) as mock_perf:
            with patch(
                "aragora.debate.subsystem_coordinator.SubsystemCoordinator._auto_init_outcome_complexity_bridge"
            ) as mock_out:
                with patch(
                    "aragora.debate.subsystem_coordinator.SubsystemCoordinator._auto_init_calibration_cost_bridge"
                ) as mock_cal:
                    coord = SubsystemCoordinator(
                        performance_monitor=Mock(),
                        outcome_tracker=Mock(),
                        calibration_tracker=Mock(),
                    )
                    mock_perf.assert_called_once()
                    mock_out.assert_called_once()
                    mock_cal.assert_called_once()


# =============================================================================
# Lifecycle Hook Tests
# =============================================================================


class TestSubsystemCoordinatorLifecycle:
    """Tests for lifecycle hooks."""

    def _create_mock_context(self):
        """Helper to create mock debate context."""
        ctx = Mock()
        ctx.debate_id = "debate-123"
        ctx.domain = "technology"
        ctx.env = Mock()
        ctx.env.task = "Design a cache system"
        ctx.agents = [Mock(name="claude"), Mock(name="gpt4")]
        ctx.start_time = 1704067200.0
        return ctx

    def test_on_debate_start_resets_moment_detector(self):
        """Test on_debate_start resets moment detector if resettable."""

        class ResettableMomentDetector:
            def __init__(self):
                self.reset_called = False

            def reset(self):
                self.reset_called = True

        detector = ResettableMomentDetector()
        coord = SubsystemCoordinator(moment_detector=detector)

        ctx = self._create_mock_context()
        coord.on_debate_start(ctx)

        assert detector.reset_called is True

    def test_on_debate_start_handles_non_resettable(self):
        """Test on_debate_start handles non-resettable moment detector."""
        detector = Mock(spec=[])  # No reset method

        coord = SubsystemCoordinator(moment_detector=detector)

        ctx = self._create_mock_context()
        # Should not raise
        coord.on_debate_start(ctx)

    def test_on_round_complete_records_positions(self):
        """Test on_round_complete records positions in ledger."""
        mock_ledger = Mock()
        coord = SubsystemCoordinator(position_ledger=mock_ledger)

        ctx = self._create_mock_context()
        positions = {
            "claude": "Use Redis for caching",
            "gpt4": "Consider Memcached",
        }

        coord.on_round_complete(ctx, round_num=2, positions=positions)

        assert mock_ledger.record_position.call_count == 2

    def test_on_round_complete_handles_ledger_error(self):
        """Test on_round_complete handles ledger errors gracefully."""
        mock_ledger = Mock()
        mock_ledger.record_position.side_effect = RuntimeError("Database error")

        coord = SubsystemCoordinator(position_ledger=mock_ledger)

        ctx = self._create_mock_context()
        positions = {"claude": "Test position"}

        # Should not raise
        coord.on_round_complete(ctx, round_num=1, positions=positions)

    def test_on_debate_complete_updates_consensus_memory(self):
        """Test on_debate_complete updates consensus memory."""
        mock_consensus = Mock()
        coord = SubsystemCoordinator(consensus_memory=mock_consensus)

        ctx = self._create_mock_context()
        result = Mock()
        result.consensus = "Use Redis for caching"
        result.consensus_confidence = 0.85
        result.messages = []

        with patch("aragora.memory.consensus.ConsensusStrength") as mock_strength:
            mock_strength.STRONG = "strong"

            coord.on_debate_complete(ctx, result)

            mock_consensus.store_consensus.assert_called_once()

    def test_on_debate_complete_updates_calibration(self):
        """Test on_debate_complete updates calibration tracker."""
        mock_calibration = Mock()
        coord = SubsystemCoordinator(calibration_tracker=mock_calibration)

        ctx = self._create_mock_context()
        result = Mock()
        result.consensus = "Redis"
        result.messages = []
        result.predictions = {
            "claude": {"prediction": "Redis", "confidence": 0.9},
            "gpt4": {"prediction": "Memcached", "confidence": 0.7},
        }

        coord.on_debate_complete(ctx, result)

        assert mock_calibration.record_prediction.call_count == 2

    def test_on_debate_complete_updates_continuum_memory(self):
        """Test on_debate_complete updates continuum memory."""
        mock_continuum = Mock()
        coord = SubsystemCoordinator(continuum_memory=mock_continuum)

        ctx = self._create_mock_context()
        result = Mock()
        result.consensus = "Use Redis"
        result.consensus_confidence = 0.8
        result.messages = []

        with patch("aragora.memory.continuum.MemoryTier") as mock_tier:
            mock_tier.MEDIUM = "medium"

            coord.on_debate_complete(ctx, result)

            mock_continuum.add.assert_called_once()


# =============================================================================
# Query Method Tests
# =============================================================================


class TestSubsystemCoordinatorQueryMethods:
    """Tests for query methods."""

    def test_get_historical_dissent_no_retriever(self):
        """Test get_historical_dissent returns empty when no retriever."""
        coord = SubsystemCoordinator()

        result = coord.get_historical_dissent("Test task")

        assert result == []

    def test_get_historical_dissent_with_retriever(self):
        """Test get_historical_dissent with retriever."""
        mock_retriever = Mock()
        mock_retriever.retrieve_for_new_debate.return_value = {
            "relevant_dissents": [
                {"position": "Alternative view 1"},
                {"position": "Alternative view 2"},
                {"position": "Alternative view 3"},
                {"position": "Alternative view 4"},
            ]
        }

        coord = SubsystemCoordinator(dissent_retriever=mock_retriever)

        result = coord.get_historical_dissent("Test task", limit=2)

        assert len(result) == 2

    def test_get_historical_dissent_handles_error(self):
        """Test get_historical_dissent handles retriever errors."""
        mock_retriever = Mock()
        mock_retriever.retrieve_for_new_debate.side_effect = RuntimeError("Error")

        coord = SubsystemCoordinator(dissent_retriever=mock_retriever)

        result = coord.get_historical_dissent("Test task")

        assert result == []

    def test_get_agent_calibration_weight_no_tracker(self):
        """Test get_agent_calibration_weight returns 1.0 without tracker."""
        coord = SubsystemCoordinator()

        weight = coord.get_agent_calibration_weight("claude")

        assert weight == 1.0

    def test_get_agent_calibration_weight_with_tracker(self):
        """Test get_agent_calibration_weight with tracker."""
        mock_tracker = Mock()
        mock_summary = Mock()
        mock_summary.total_predictions = 10
        mock_summary.brier_score = 0.1  # Good calibration
        mock_tracker.get_calibration_summary.return_value = mock_summary

        coord = SubsystemCoordinator(calibration_tracker=mock_tracker)

        weight = coord.get_agent_calibration_weight("claude")

        # 0.5 + (1.0 - min(0.1, 0.5)) = 0.5 + 0.9 = 1.4
        assert 0.5 <= weight <= 1.5

    def test_get_agent_calibration_weight_poor_calibration(self):
        """Test get_agent_calibration_weight with poor calibration."""
        mock_tracker = Mock()
        mock_summary = Mock()
        mock_summary.total_predictions = 10
        mock_summary.brier_score = 0.5  # Poor calibration
        mock_tracker.get_calibration_summary.return_value = mock_summary

        coord = SubsystemCoordinator(calibration_tracker=mock_tracker)

        weight = coord.get_agent_calibration_weight("claude")

        # Should be lower for poor calibration
        assert weight == 1.0  # 0.5 + (1.0 - 0.5) = 1.0

    def test_get_agent_calibration_weight_no_predictions(self):
        """Test get_agent_calibration_weight with no predictions."""
        mock_tracker = Mock()
        mock_summary = Mock()
        mock_summary.total_predictions = 0
        mock_tracker.get_calibration_summary.return_value = mock_summary

        coord = SubsystemCoordinator(calibration_tracker=mock_tracker)

        weight = coord.get_agent_calibration_weight("claude")

        assert weight == 1.0

    def test_get_continuum_context_no_memory(self):
        """Test get_continuum_context returns empty without memory."""
        coord = SubsystemCoordinator()

        context = coord.get_continuum_context("Test task")

        assert context == ""

    def test_get_continuum_context_with_memory(self):
        """Test get_continuum_context with memory."""
        mock_continuum = Mock()
        mock_mem1 = Mock()
        mock_mem1.content = "Previous decision about Redis"
        mock_mem1.metadata = {"summary": "Redis is fast"}

        mock_mem2 = Mock()
        mock_mem2.content = "Previous decision about caching"
        mock_mem2.metadata = {}

        mock_continuum.retrieve.return_value = [mock_mem1, mock_mem2]

        coord = SubsystemCoordinator(continuum_memory=mock_continuum)

        context = coord.get_continuum_context("Cache design", limit=5)

        assert "Relevant learnings" in context
        assert "Redis is fast" in context
        assert "caching" in context

    def test_get_continuum_context_no_results(self):
        """Test get_continuum_context with no matching memories."""
        mock_continuum = Mock()
        mock_continuum.retrieve.return_value = []

        coord = SubsystemCoordinator(continuum_memory=mock_continuum)

        context = coord.get_continuum_context("Unrelated task")

        assert context == ""


# =============================================================================
# Hook Handlers Tests
# =============================================================================


class TestSubsystemCoordinatorHookHandlers:
    """Tests for hook handler integration."""

    def test_has_hook_handlers_false_no_registry(self):
        """Test has_hook_handlers is False without registry."""
        coord = SubsystemCoordinator()

        assert coord.has_hook_handlers is False

    def test_has_hook_handlers_false_not_registered(self):
        """Test has_hook_handlers is False when not registered."""
        mock_registry = Mock()
        mock_registry.is_registered = False

        coord = SubsystemCoordinator(hook_handler_registry=mock_registry)

        assert coord.has_hook_handlers is False

    def test_has_hook_handlers_true(self):
        """Test has_hook_handlers is True when registered."""
        mock_registry = Mock()
        mock_registry.is_registered = True

        coord = SubsystemCoordinator(hook_handler_registry=mock_registry)

        assert coord.has_hook_handlers is True


# =============================================================================
# Get Status Tests
# =============================================================================


class TestSubsystemCoordinatorGetStatus:
    """Tests for get_status method."""

    def test_get_status_empty(self):
        """Test get_status with no subsystems."""
        coord = SubsystemCoordinator()

        status = coord.get_status()

        assert "subsystems" in status
        assert "capabilities" in status
        assert "cross_pollination_bridges" in status
        assert "knowledge_mound" in status
        assert status["initialized"] is True

    def test_get_status_with_subsystems(self):
        """Test get_status with subsystems."""
        coord = SubsystemCoordinator(
            position_tracker=Mock(),
            elo_system=Mock(),
            consensus_memory=Mock(),
        )

        status = coord.get_status()

        assert status["subsystems"]["position_tracker"] is True
        assert status["subsystems"]["elo_system"] is True
        assert status["subsystems"]["consensus_memory"] is True
        assert status["capabilities"]["position_tracking"] is True
        assert status["capabilities"]["elo_ranking"] is True

    def test_get_status_with_bridges(self):
        """Test get_status with bridges."""
        coord = SubsystemCoordinator(
            performance_router_bridge=Mock(),
            analytics_selection_bridge=Mock(),
        )

        status = coord.get_status()

        assert status["cross_pollination_bridges"]["performance_router"] is True
        assert status["cross_pollination_bridges"]["analytics_selection"] is True
        assert status["active_bridges_count"] == 2

    def test_get_status_with_km(self):
        """Test get_status with Knowledge Mound."""
        coord = SubsystemCoordinator(
            knowledge_mound=Mock(),
            km_coordinator=Mock(),
            km_continuum_adapter=Mock(),
            km_elo_adapter=Mock(),
        )

        status = coord.get_status()

        assert status["knowledge_mound"]["available"] is True
        assert status["knowledge_mound"]["coordinator_active"] is True
        assert status["knowledge_mound"]["bidirectional_enabled"] is True
        assert status["knowledge_mound"]["adapters"]["continuum"] is True
        assert status["knowledge_mound"]["adapters"]["elo"] is True
        assert status["knowledge_mound"]["active_adapters_count"] == 2

    def test_get_status_includes_init_errors(self):
        """Test get_status includes initialization errors."""
        coord = SubsystemCoordinator()
        coord._init_errors.append("Test error")

        status = coord.get_status()

        assert "Test error" in status["init_errors"]


# =============================================================================
# SubsystemConfig Tests
# =============================================================================


class TestSubsystemConfig:
    """Tests for SubsystemConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = SubsystemConfig()

        assert config.enable_position_ledger is False
        assert config.enable_calibration is False
        assert config.enable_moment_detection is False
        assert config.enable_hook_handlers is True
        assert config.enable_performance_router is True
        assert config.enable_km_bidirectional is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = SubsystemConfig(
            enable_position_ledger=True,
            enable_calibration=True,
            enable_moment_detection=True,
            km_sync_interval_seconds=600,
        )

        assert config.enable_position_ledger is True
        assert config.enable_calibration is True
        assert config.km_sync_interval_seconds == 600

    def test_create_coordinator(self):
        """Test create_coordinator factory method."""
        mock_protocol = Mock()
        mock_elo = Mock()
        mock_km = Mock()

        config = SubsystemConfig(
            elo_system=mock_elo,
            knowledge_mound=mock_km,
            enable_calibration=True,
        )

        coord = config.create_coordinator(
            protocol=mock_protocol,
            loop_id="loop-123",
        )

        assert coord.protocol is mock_protocol
        assert coord.loop_id == "loop-123"
        assert coord.elo_system is mock_elo
        assert coord.knowledge_mound is mock_km
        assert coord.enable_calibration is True

    def test_create_coordinator_with_bridges(self):
        """Test create_coordinator with bridge sources."""
        mock_performance_monitor = Mock()
        mock_outcome_tracker = Mock()

        config = SubsystemConfig(
            performance_monitor=mock_performance_monitor,
            outcome_tracker=mock_outcome_tracker,
        )

        coord = config.create_coordinator()

        assert coord.performance_monitor is mock_performance_monitor
        assert coord.outcome_tracker is mock_outcome_tracker


# =============================================================================
# Integration Tests
# =============================================================================


class TestSubsystemCoordinatorIntegration:
    """Integration tests for SubsystemCoordinator."""

    def _create_mock_context(self):
        """Helper to create mock debate context."""
        ctx = Mock()
        ctx.debate_id = "debate-123"
        ctx.domain = "technology"
        ctx.env = Mock()
        ctx.env.task = "Design a cache system"
        ctx.agents = [Mock(name="claude"), Mock(name="gpt4")]
        ctx.start_time = 1704067200.0
        return ctx

    def test_full_debate_lifecycle(self):
        """Test complete debate lifecycle."""
        mock_ledger = Mock()
        mock_consensus = Mock()

        coord = SubsystemCoordinator(
            position_ledger=mock_ledger,
            consensus_memory=mock_consensus,
        )

        ctx = self._create_mock_context()

        # Start debate
        coord.on_debate_start(ctx)

        # Complete rounds
        coord.on_round_complete(ctx, 1, {"claude": "Proposal 1"})
        coord.on_round_complete(ctx, 2, {"claude": "Proposal 2", "gpt4": "Counter"})

        # Complete debate
        result = Mock()
        result.consensus = "Final decision"
        result.consensus_confidence = 0.9
        result.predictions = {}
        result.messages = []

        with patch("aragora.memory.consensus.ConsensusStrength") as mock_strength:
            mock_strength.UNANIMOUS = "unanimous"
            coord.on_debate_complete(ctx, result)

        # Verify interactions
        assert mock_ledger.record_position.call_count >= 2
        mock_consensus.store_consensus.assert_called_once()

    def test_coordinator_with_all_subsystems(self):
        """Test coordinator with all subsystems configured."""
        # Create mock subsystems
        mock_position_tracker = Mock()
        mock_position_ledger = Mock()
        mock_elo = Mock()
        mock_calibration = Mock()
        mock_consensus = Mock()
        mock_dissent = Mock()
        mock_continuum = Mock()
        mock_flip_detector = Mock()
        mock_moment_detector = Mock()
        mock_relationship = Mock()
        mock_tier_analytics = Mock()
        mock_persona_manager = Mock()
        mock_hook_manager = Mock()
        mock_hook_handler_registry = Mock()
        mock_hook_handler_registry.registered_count = 5

        coord = SubsystemCoordinator(
            position_tracker=mock_position_tracker,
            position_ledger=mock_position_ledger,
            elo_system=mock_elo,
            calibration_tracker=mock_calibration,
            consensus_memory=mock_consensus,
            dissent_retriever=mock_dissent,
            continuum_memory=mock_continuum,
            flip_detector=mock_flip_detector,
            moment_detector=mock_moment_detector,
            relationship_tracker=mock_relationship,
            tier_analytics_tracker=mock_tier_analytics,
            persona_manager=mock_persona_manager,
            hook_manager=mock_hook_manager,
            hook_handler_registry=mock_hook_handler_registry,
        )

        status = coord.get_status()

        assert all(status["subsystems"].values())
        assert all(status["capabilities"].values())

    def test_coordinator_error_resilience(self):
        """Test coordinator handles subsystem errors gracefully."""
        mock_ledger = Mock()
        mock_ledger.record_position.side_effect = RuntimeError("Database error")

        mock_continuum = Mock()
        mock_continuum.add.side_effect = RuntimeError("Storage error")

        coord = SubsystemCoordinator(
            position_ledger=mock_ledger,
            continuum_memory=mock_continuum,
        )

        ctx = Mock()
        ctx.debate_id = "debate-123"
        ctx.env = Mock()
        ctx.env.task = "Test"
        ctx.agents = []
        ctx.start_time = 1704067200.0

        # Should not raise
        coord.on_round_complete(ctx, 1, {"agent": "position"})

        result = Mock()
        result.consensus = "Test"
        result.consensus_confidence = 0.5
        result.predictions = {}
        result.messages = []

        # Should not raise
        coord.on_debate_complete(ctx, result)


# =============================================================================
# Edge Cases Tests
# =============================================================================


class TestSubsystemCoordinatorEdgeCases:
    """Tests for edge cases."""

    def test_empty_positions_on_round_complete(self):
        """Test on_round_complete with empty positions."""
        mock_ledger = Mock()
        coord = SubsystemCoordinator(position_ledger=mock_ledger)

        ctx = Mock()
        ctx.debate_id = "debate-123"
        ctx.env = Mock()

        coord.on_round_complete(ctx, 1, {})

        mock_ledger.record_position.assert_not_called()

    def test_none_result_on_debate_complete(self):
        """Test on_debate_complete with None result."""
        mock_consensus = Mock()
        coord = SubsystemCoordinator(consensus_memory=mock_consensus)

        ctx = Mock()

        # Should not raise
        coord.on_debate_complete(ctx, None)

        mock_consensus.store_consensus.assert_not_called()

    def test_result_without_consensus_attribute(self):
        """Test on_debate_complete with result missing attributes."""
        mock_consensus = Mock()
        coord = SubsystemCoordinator(consensus_memory=mock_consensus)

        ctx = Mock()
        ctx.debate_id = "debate-123"
        ctx.env = Mock()
        ctx.env.task = "Test"
        ctx.agents = []
        ctx.start_time = 1704067200.0

        result = Mock(spec=[])  # No consensus attribute

        # Should handle gracefully
        coord.on_debate_complete(ctx, result)

    def test_calibration_with_invalid_prediction_format(self):
        """Test calibration update with invalid prediction format."""
        mock_calibration = Mock()
        coord = SubsystemCoordinator(calibration_tracker=mock_calibration)

        ctx = Mock()
        ctx.debate_id = "debate-123"
        ctx.domain = "test"
        ctx.start_time = 1704067200.0

        result = Mock()
        result.consensus = "Result"
        result.predictions = {
            "agent1": "string_prediction",  # Not a dict
        }
        result.messages = []

        # Should handle gracefully
        coord.on_debate_complete(ctx, result)

        # Should still try to record
        mock_calibration.record_prediction.assert_called_once()

    def test_continuum_memory_none_metadata(self):
        """Test get_continuum_context handles None metadata."""
        mock_continuum = Mock()
        mock_mem = Mock()
        mock_mem.content = "Test content"
        mock_mem.metadata = None

        mock_continuum.retrieve.return_value = [mock_mem]

        coord = SubsystemCoordinator(continuum_memory=mock_continuum)

        context = coord.get_continuum_context("Test")

        assert "Test content" in context

    def test_protocol_reference(self):
        """Test protocol reference is stored."""
        mock_protocol = Mock()
        coord = SubsystemCoordinator(protocol=mock_protocol)

        assert coord.protocol is mock_protocol

    def test_loop_id(self):
        """Test loop_id is stored."""
        coord = SubsystemCoordinator(loop_id="my-loop-123")

        assert coord.loop_id == "my-loop-123"

    def test_persona_manager(self):
        """Test persona_manager is stored."""
        mock_persona = Mock()
        coord = SubsystemCoordinator(persona_manager=mock_persona)

        assert coord.persona_manager is mock_persona


# =============================================================================
# KM Coordinator Auto-Init Tests
# =============================================================================


class TestKMCoordinatorAutoInit:
    """Tests for KM coordinator auto-initialization."""

    def test_auto_init_km_coordinator_called(self):
        """Test _auto_init_km_coordinator is called when enabled."""
        with patch(
            "aragora.debate.subsystem_coordinator.SubsystemCoordinator._auto_init_km_coordinator"
        ) as mock_init:
            coord = SubsystemCoordinator(
                enable_km_coordinator=True,
                enable_km_bidirectional=True,
            )
            mock_init.assert_called_once()

    def test_register_km_adapters_with_coordinator(self):
        """Test _register_km_adapters is called after coordinator init."""
        mock_km = Mock()
        mock_km_coord = Mock()
        mock_km_coord.adapter_count = 0

        coord = SubsystemCoordinator(
            knowledge_mound=mock_km,
            km_coordinator=mock_km_coord,
            km_continuum_adapter=Mock(),
        )

        # Adapter registration should have been attempted
        # (actual registration depends on coordinator API)

    def test_km_config_values(self):
        """Test KM configuration values are passed through."""
        coord = SubsystemCoordinator(
            km_sync_interval_seconds=600,
            km_min_confidence_for_reverse=0.8,
            km_parallel_sync=False,
        )

        assert coord.km_sync_interval_seconds == 600
        assert coord.km_min_confidence_for_reverse == 0.8
        assert coord.km_parallel_sync is False


# =============================================================================
# Additional Coverage: Consensus confidence -> ConsensusStrength mapping
# =============================================================================


class TestConsensusStrengthMapping:
    """Tests for on_debate_complete confidence-to-strength mapping."""

    def _create_mock_context(self, agents=None):
        ctx = Mock()
        ctx.debate_id = "debate-strength"
        ctx.domain = "technology"
        ctx.env = Mock()
        ctx.env.task = "Test task"
        ctx.agents = agents or [Mock(name="claude")]
        ctx.start_time = 1704067200.0
        return ctx

    def test_unanimous_strength_at_090(self):
        """Test confidence >= 0.9 maps to UNANIMOUS strength."""
        mock_consensus = Mock()
        coord = SubsystemCoordinator(consensus_memory=mock_consensus)

        ctx = self._create_mock_context()
        result = Mock()
        result.consensus = "Unanimous decision"
        result.consensus_confidence = 0.95
        result.predictions = {}
        result.messages = []

        with patch("aragora.memory.consensus.ConsensusStrength") as MockStrength:
            MockStrength.UNANIMOUS = "unanimous"
            coord.on_debate_complete(ctx, result)

            call_kwargs = mock_consensus.store_consensus.call_args[1]
            assert call_kwargs["strength"] == "unanimous"

    def test_strong_strength_at_080(self):
        """Test confidence >= 0.8 (but < 0.9) maps to STRONG strength."""
        mock_consensus = Mock()
        coord = SubsystemCoordinator(consensus_memory=mock_consensus)

        ctx = self._create_mock_context()
        result = Mock()
        result.consensus = "Strong decision"
        result.consensus_confidence = 0.85
        result.predictions = {}
        result.messages = []

        with patch("aragora.memory.consensus.ConsensusStrength") as MockStrength:
            MockStrength.STRONG = "strong"
            coord.on_debate_complete(ctx, result)

            call_kwargs = mock_consensus.store_consensus.call_args[1]
            assert call_kwargs["strength"] == "strong"

    def test_moderate_strength_at_060(self):
        """Test confidence >= 0.6 (but < 0.8) maps to MODERATE strength."""
        mock_consensus = Mock()
        coord = SubsystemCoordinator(consensus_memory=mock_consensus)

        ctx = self._create_mock_context()
        result = Mock()
        result.consensus = "Moderate decision"
        result.consensus_confidence = 0.65
        result.predictions = {}
        result.messages = []

        with patch("aragora.memory.consensus.ConsensusStrength") as MockStrength:
            MockStrength.MODERATE = "moderate"
            coord.on_debate_complete(ctx, result)

            call_kwargs = mock_consensus.store_consensus.call_args[1]
            assert call_kwargs["strength"] == "moderate"

    def test_weak_strength_at_050(self):
        """Test confidence >= 0.5 (but < 0.6) maps to WEAK strength."""
        mock_consensus = Mock()
        coord = SubsystemCoordinator(consensus_memory=mock_consensus)

        ctx = self._create_mock_context()
        result = Mock()
        result.consensus = "Weak decision"
        result.consensus_confidence = 0.55
        result.predictions = {}
        result.messages = []

        with patch("aragora.memory.consensus.ConsensusStrength") as MockStrength:
            MockStrength.WEAK = "weak"
            coord.on_debate_complete(ctx, result)

            call_kwargs = mock_consensus.store_consensus.call_args[1]
            assert call_kwargs["strength"] == "weak"

    def test_split_strength_below_050(self):
        """Test confidence < 0.5 maps to SPLIT strength."""
        mock_consensus = Mock()
        coord = SubsystemCoordinator(consensus_memory=mock_consensus)

        ctx = self._create_mock_context()
        result = Mock()
        result.consensus = "Split decision"
        result.consensus_confidence = 0.35
        result.predictions = {}
        result.messages = []

        with patch("aragora.memory.consensus.ConsensusStrength") as MockStrength:
            MockStrength.SPLIT = "split"
            coord.on_debate_complete(ctx, result)

            call_kwargs = mock_consensus.store_consensus.call_args[1]
            assert call_kwargs["strength"] == "split"


# =============================================================================
# Additional Coverage: _register_km_adapters with real adapters
# =============================================================================


class TestRegisterKMAdaptersDeep:
    """Tests for _register_km_adapters with various adapter combinations."""

    def test_register_all_six_adapters(self):
        """Test _register_km_adapters registers all 6 adapter types when called manually."""
        mock_km_coord = Mock()

        # Disable all auto-inits to isolate the test from module-level state
        # that prior tests may have modified (SDPO imports, bridge factories, etc.)
        coord = SubsystemCoordinator(
            km_coordinator=mock_km_coord,
            enable_km_coordinator=False,
            enable_km_bidirectional=False,
            enable_sdpo=False,
            km_continuum_adapter=Mock(),
            km_elo_adapter=Mock(),
            km_belief_adapter=Mock(),
            km_insights_adapter=Mock(),
            km_critique_adapter=Mock(),
            km_pulse_adapter=Mock(),
        )

        # Reset mock to clear any calls from __post_init__ auto-init side effects
        mock_km_coord.reset_mock()

        coord._register_km_adapters()

        assert mock_km_coord.register_adapter.call_count == 6

        # Check priority ordering
        calls = mock_km_coord.register_adapter.call_args_list
        priorities = [c[1]["priority"] for c in calls]
        assert priorities == [1, 2, 3, 4, 5, 6]

        # Check adapter names
        names = [c[1]["name"] for c in calls]
        assert names == ["continuum", "elo", "belief", "insights", "critique", "pulse"]

    def test_register_partial_adapters(self):
        """Test _register_km_adapters registers only available adapters."""
        mock_km_coord = Mock()

        coord = SubsystemCoordinator(
            km_coordinator=mock_km_coord,
            enable_km_coordinator=False,
            enable_km_bidirectional=False,
            enable_sdpo=False,
            km_continuum_adapter=Mock(),
            km_elo_adapter=Mock(),
        )

        # Reset mock to clear any calls from __post_init__ auto-init side effects
        mock_km_coord.reset_mock()

        coord._register_km_adapters()

        assert mock_km_coord.register_adapter.call_count == 2

    def test_register_adapters_handles_registration_error(self):
        """Test _register_km_adapters handles adapter registration failure."""
        mock_km_coord = Mock()

        # Disable all auto-inits to isolate the test from module-level state
        # that prior tests may have modified (SDPO imports, bridge factories, etc.)
        coord = SubsystemCoordinator(
            km_coordinator=mock_km_coord,
            enable_km_coordinator=False,
            enable_km_bidirectional=False,
            enable_sdpo=False,
            km_continuum_adapter=Mock(),
        )

        # Reset mock to clear any calls from __post_init__ auto-init side effects
        mock_km_coord.reset_mock()

        # Now set up the error side_effect and call manually
        mock_km_coord.register_adapter.side_effect = RuntimeError("Registration failed")

        # Should not raise
        coord._register_km_adapters()

        # Registration was attempted but failed
        mock_km_coord.register_adapter.assert_called_once()

    def test_register_adapters_no_coordinator(self):
        """Test _register_km_adapters is no-op without coordinator."""
        coord = SubsystemCoordinator(
            km_coordinator=None,
            enable_km_coordinator=False,
            enable_km_bidirectional=False,
            enable_sdpo=False,
            km_continuum_adapter=Mock(),
        )

        # km_coordinator should remain None since auto-init is disabled
        assert coord.km_coordinator is None

        # Calling _register_km_adapters should be a no-op (early return)
        coord._register_km_adapters()

        # No error and adapter field is present but not registered via coordinator
        assert coord.active_km_adapters_count == 1


# =============================================================================
# Additional Coverage: on_debate_start with reset() failure
# =============================================================================


class TestOnDebateStartDeep:
    """Tests for on_debate_start edge cases."""

    def _create_mock_context(self):
        ctx = Mock()
        ctx.debate_id = "debate-start"
        ctx.env = Mock()
        ctx.env.task = "Test"
        ctx.agents = []
        return ctx

    def test_on_debate_start_reset_raises(self):
        """Test on_debate_start handles reset() exception."""

        class FaultyResettable:
            def reset(self):
                raise RuntimeError("Reset failed")

        detector = FaultyResettable()
        coord = SubsystemCoordinator(moment_detector=detector)

        ctx = self._create_mock_context()
        # Should not raise despite reset failure
        coord.on_debate_start(ctx)

    def test_on_debate_start_no_subsystems(self):
        """Test on_debate_start with no subsystems is a no-op."""
        coord = SubsystemCoordinator()

        ctx = self._create_mock_context()
        # Should not raise
        coord.on_debate_start(ctx)

    def test_on_debate_start_non_resettable_moment_detector(self):
        """Test on_debate_start skips reset for non-Resettable detector."""
        detector = Mock(spec=[])  # No reset method
        coord = SubsystemCoordinator(moment_detector=detector)

        ctx = self._create_mock_context()
        coord.on_debate_start(ctx)

        # Should not have called reset


# =============================================================================
# Additional Coverage: get_agent_calibration_weight error handling
# =============================================================================


class TestCalibrationWeightDeep:
    """Tests for get_agent_calibration_weight error cases."""

    def test_calibration_weight_summary_none(self):
        """Test get_agent_calibration_weight when summary is None."""
        mock_tracker = Mock()
        mock_tracker.get_calibration_summary.return_value = None

        coord = SubsystemCoordinator(calibration_tracker=mock_tracker)

        weight = coord.get_agent_calibration_weight("claude")

        assert weight == 1.0

    def test_calibration_weight_key_error(self):
        """Test get_agent_calibration_weight handles KeyError."""
        mock_tracker = Mock()
        mock_tracker.get_calibration_summary.side_effect = KeyError("Unknown agent")

        coord = SubsystemCoordinator(calibration_tracker=mock_tracker)

        weight = coord.get_agent_calibration_weight("unknown_agent")

        assert weight == 1.0

    def test_calibration_weight_attribute_error(self):
        """Test get_agent_calibration_weight handles AttributeError."""
        mock_tracker = Mock()
        mock_tracker.get_calibration_summary.side_effect = AttributeError("Missing attr")

        coord = SubsystemCoordinator(calibration_tracker=mock_tracker)

        weight = coord.get_agent_calibration_weight("claude")

        assert weight == 1.0

    def test_calibration_weight_exact_calculation(self):
        """Test get_agent_calibration_weight exact formula: 0.5 + (1 - min(brier, 0.5))."""
        mock_tracker = Mock()
        mock_summary = Mock()
        mock_summary.total_predictions = 20
        mock_summary.brier_score = 0.2
        mock_tracker.get_calibration_summary.return_value = mock_summary

        coord = SubsystemCoordinator(calibration_tracker=mock_tracker)

        weight = coord.get_agent_calibration_weight("claude")

        # 0.5 + (1.0 - min(0.2, 0.5)) = 0.5 + 0.8 = 1.3
        assert weight == pytest.approx(1.3, abs=0.01)

    def test_calibration_weight_brier_above_half(self):
        """Test get_agent_calibration_weight with brier_score > 0.5 caps at 0.5."""
        mock_tracker = Mock()
        mock_summary = Mock()
        mock_summary.total_predictions = 10
        mock_summary.brier_score = 0.8  # Very poor, capped to 0.5
        mock_tracker.get_calibration_summary.return_value = mock_summary

        coord = SubsystemCoordinator(calibration_tracker=mock_tracker)

        weight = coord.get_agent_calibration_weight("claude")

        # 0.5 + (1.0 - min(0.8, 0.5)) = 0.5 + (1.0 - 0.5) = 1.0
        assert weight == pytest.approx(1.0, abs=0.01)


# =============================================================================
# Additional Coverage: get_status with hook_handler_registry
# =============================================================================


class TestGetStatusDeep:
    """Tests for get_status method deep coverage."""

    def test_get_status_hook_handlers_count(self):
        """Test get_status reports hook_handlers_registered count."""
        mock_registry = Mock()
        mock_registry.is_registered = True
        mock_registry.registered_count = 7

        coord = SubsystemCoordinator(hook_handler_registry=mock_registry)

        status = coord.get_status()

        assert status["hook_handlers_registered"] == 7

    def test_get_status_hook_handlers_count_zero(self):
        """Test get_status reports 0 when no registry."""
        coord = SubsystemCoordinator()

        status = coord.get_status()

        assert status["hook_handlers_registered"] == 0

    def test_get_status_km_config(self):
        """Test get_status includes KM config values."""
        coord = SubsystemCoordinator(
            km_sync_interval_seconds=120,
            km_min_confidence_for_reverse=0.9,
            km_parallel_sync=False,
        )

        status = coord.get_status()

        assert status["knowledge_mound"]["config"]["sync_interval_seconds"] == 120
        assert status["knowledge_mound"]["config"]["min_confidence_for_reverse"] == 0.9
        assert status["knowledge_mound"]["config"]["parallel_sync"] is False

    def test_get_status_all_bridges_active(self):
        """Test get_status with all 7 bridges active."""
        coord = SubsystemCoordinator(
            performance_router_bridge=Mock(),
            outcome_complexity_bridge=Mock(),
            analytics_selection_bridge=Mock(),
            novelty_selection_bridge=Mock(),
            relationship_bias_bridge=Mock(),
            rlm_selection_bridge=Mock(),
            calibration_cost_bridge=Mock(),
        )

        status = coord.get_status()

        assert status["active_bridges_count"] == 7
        assert all(status["cross_pollination_bridges"].values())

    def test_get_status_all_adapters_active(self):
        """Test get_status with all KM adapters active."""
        coord = SubsystemCoordinator(
            knowledge_mound=Mock(),
            km_coordinator=Mock(),
            km_continuum_adapter=Mock(),
            km_elo_adapter=Mock(),
            km_belief_adapter=Mock(),
            km_insights_adapter=Mock(),
            km_critique_adapter=Mock(),
            km_pulse_adapter=Mock(),
            km_obsidian_adapter=Mock(),
        )

        status = coord.get_status()

        assert status["knowledge_mound"]["active_adapters_count"] == 7
        assert all(status["knowledge_mound"]["adapters"].values())

    def test_get_status_subsystems_all_false_when_empty(self):
        """Test get_status subsystems are all False when coordinator is empty."""
        coord = SubsystemCoordinator(enable_sdpo=False)

        status = coord.get_status()

        for key, val in status["subsystems"].items():
            assert val is False, f"Expected {key} to be False"

        for key, val in status["capabilities"].items():
            assert val is False, f"Expected capability {key} to be False"


# =============================================================================
# Additional Coverage: on_debate_complete with empty env/agents
# =============================================================================


class TestOnDebateCompleteDeep:
    """Tests for on_debate_complete edge cases."""

    def test_on_debate_complete_no_env(self):
        """Test on_debate_complete handles ctx with no env."""
        mock_consensus = Mock()
        coord = SubsystemCoordinator(consensus_memory=mock_consensus)

        ctx = Mock()
        ctx.debate_id = "debate-123"
        ctx.env = None
        ctx.agents = []
        ctx.start_time = 1704067200.0

        result = Mock()
        result.consensus = "Decision"
        result.consensus_confidence = 0.75
        result.predictions = {}
        result.messages = []

        with patch("aragora.memory.consensus.ConsensusStrength") as MockStrength:
            MockStrength.MODERATE = "moderate"
            coord.on_debate_complete(ctx, result)

            call_kwargs = mock_consensus.store_consensus.call_args[1]
            assert call_kwargs["topic"] == ""  # Empty task when no env

    def test_on_debate_complete_consensus_memory_error(self):
        """Test on_debate_complete handles consensus memory store error."""
        mock_consensus = Mock()
        mock_consensus.store_consensus.side_effect = RuntimeError("Store failed")

        coord = SubsystemCoordinator(consensus_memory=mock_consensus)

        ctx = Mock()
        ctx.debate_id = "debate-123"
        ctx.env = Mock()
        ctx.env.task = "Test"
        ctx.agents = []
        ctx.start_time = 1704067200.0

        result = Mock()
        result.consensus = "Decision"
        result.consensus_confidence = 0.8
        result.predictions = {}
        result.messages = []

        # Should not raise
        with patch("aragora.memory.consensus.ConsensusStrength") as MockStrength:
            MockStrength.STRONG = "strong"
            coord.on_debate_complete(ctx, result)

    def test_on_debate_complete_all_subsystems(self):
        """Test on_debate_complete updates all three subsystems."""
        mock_consensus = Mock()
        mock_calibration = Mock()
        mock_continuum = Mock()

        coord = SubsystemCoordinator(
            consensus_memory=mock_consensus,
            calibration_tracker=mock_calibration,
            continuum_memory=mock_continuum,
        )

        ctx = Mock()
        ctx.debate_id = "debate-all"
        ctx.domain = "technology"
        ctx.env = Mock()
        ctx.env.task = "Design cache"
        ctx.agents = [Mock(name="claude"), Mock(name="gpt4")]
        ctx.start_time = 1704067200.0

        result = Mock()
        result.consensus = "Use Redis"
        result.consensus_confidence = 0.9
        result.predictions = {
            "claude": {"prediction": "Redis", "confidence": 0.9},
        }
        result.messages = []

        with patch("aragora.memory.consensus.ConsensusStrength") as MockStrength:
            MockStrength.UNANIMOUS = "unanimous"
            with patch("aragora.memory.continuum.MemoryTier") as MockTier:
                MockTier.MEDIUM = "medium"
                coord.on_debate_complete(ctx, result)

        mock_consensus.store_consensus.assert_called_once()
        mock_calibration.record_prediction.assert_called_once()
        mock_continuum.add.assert_called_once()

    def test_on_debate_complete_dict_prediction(self):
        """Test on_debate_complete handles dict prediction format."""
        mock_calibration = Mock()
        coord = SubsystemCoordinator(calibration_tracker=mock_calibration)

        ctx = Mock()
        ctx.debate_id = "debate-dict"
        ctx.domain = "tech"
        ctx.start_time = 1704067200.0

        result = Mock()
        result.consensus = "Redis"
        result.predictions = {
            "claude": {"prediction": "Redis", "confidence": 0.9},
        }
        result.messages = []

        coord.on_debate_complete(ctx, result)

        call_kwargs = mock_calibration.record_prediction.call_args[1]
        assert call_kwargs["confidence"] == 0.9
        assert call_kwargs["correct"] is True  # prediction matches consensus

    def test_on_debate_complete_string_prediction(self):
        """Test on_debate_complete handles string prediction format."""
        mock_calibration = Mock()
        coord = SubsystemCoordinator(calibration_tracker=mock_calibration)

        ctx = Mock()
        ctx.debate_id = "debate-str"
        ctx.domain = "tech"
        ctx.start_time = 1704067200.0

        result = Mock()
        result.consensus = "Memcached"
        result.predictions = {
            "gpt4": "Redis",  # String prediction (not dict)
        }
        result.messages = []

        coord.on_debate_complete(ctx, result)

        call_kwargs = mock_calibration.record_prediction.call_args[1]
        assert call_kwargs["confidence"] == 0.5  # Default for non-dict
        assert call_kwargs["correct"] is False  # "Redis" != "Memcached"


# =============================================================================
# Additional Coverage: SubsystemConfig create_coordinator deep tests
# =============================================================================


class TestSubsystemConfigDeep:
    """Additional tests for SubsystemConfig."""

    def test_config_passes_all_km_adapters(self):
        """Test create_coordinator passes all KM adapters."""
        adapters = {
            "km_continuum_adapter": Mock(),
            "km_elo_adapter": Mock(),
            "km_belief_adapter": Mock(),
            "km_insights_adapter": Mock(),
            "km_critique_adapter": Mock(),
            "km_pulse_adapter": Mock(),
        }

        config = SubsystemConfig(**adapters)
        coord = config.create_coordinator()

        for name, mock_adapter in adapters.items():
            assert getattr(coord, name) is mock_adapter

    def test_config_passes_all_bridge_flags(self):
        """Test create_coordinator passes all bridge enable flags."""
        config = SubsystemConfig(
            enable_performance_router=False,
            enable_outcome_complexity=False,
            enable_analytics_selection=False,
            enable_novelty_selection=False,
            enable_relationship_bias=False,
            enable_rlm_selection=False,
            enable_calibration_cost=False,
        )

        coord = config.create_coordinator()

        assert coord.enable_performance_router is False
        assert coord.enable_outcome_complexity is False
        assert coord.enable_analytics_selection is False
        assert coord.enable_novelty_selection is False
        assert coord.enable_relationship_bias is False
        assert coord.enable_rlm_selection is False
        assert coord.enable_calibration_cost is False

    def test_config_passes_km_config_values(self):
        """Test create_coordinator passes KM configuration."""
        config = SubsystemConfig(
            km_sync_interval_seconds=120,
            km_min_confidence_for_reverse=0.9,
            km_parallel_sync=False,
        )

        coord = config.create_coordinator()

        assert coord.km_sync_interval_seconds == 120
        assert coord.km_min_confidence_for_reverse == 0.9
        assert coord.km_parallel_sync is False
