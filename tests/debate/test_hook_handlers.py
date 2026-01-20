"""Tests for HookHandlerRegistry hook wiring."""

from __future__ import annotations

import pytest
from dataclasses import dataclass
from typing import Any, List
from unittest.mock import Mock, AsyncMock

from aragora.debate.hooks import HookManager, HookType, HookPriority, create_hook_manager
from aragora.debate.hook_handlers import HookHandlerRegistry, create_hook_handler_registry


class TestHookHandlerRegistry:
    """Test HookHandlerRegistry basic functionality."""

    def test_init_empty_subsystems(self) -> None:
        """Registry can be created with no subsystems."""
        manager = create_hook_manager()
        registry = HookHandlerRegistry(hook_manager=manager, subsystems={})
        assert registry.registered_count == 0
        assert not registry.is_registered

    def test_register_all_empty(self) -> None:
        """register_all with no subsystems registers nothing."""
        manager = create_hook_manager()
        registry = HookHandlerRegistry(hook_manager=manager, subsystems={})
        count = registry.register_all()
        assert count == 0
        assert registry.is_registered

    def test_register_idempotent(self) -> None:
        """Calling register_all twice only registers once."""
        manager = create_hook_manager()
        registry = HookHandlerRegistry(hook_manager=manager, subsystems={})
        count1 = registry.register_all()
        count2 = registry.register_all()
        assert count1 == 0
        assert count2 == 0
        assert registry.is_registered

    def test_unregister_all(self) -> None:
        """unregister_all removes all handlers."""
        manager = create_hook_manager()

        # Create mock with a method
        mock_analytics = Mock()
        mock_analytics.on_round_complete = Mock()

        registry = HookHandlerRegistry(
            hook_manager=manager,
            subsystems={"analytics": mock_analytics},
        )

        # Register
        count = registry.register_all()
        assert count >= 1
        assert registry.is_registered

        # Unregister
        unregistered = registry.unregister_all()
        assert unregistered == count
        assert not registry.is_registered


class TestAnalyticsHandlers:
    """Test analytics subsystem hook wiring."""

    @pytest.mark.asyncio
    async def test_round_complete_handler(self) -> None:
        """POST_ROUND hook calls analytics.on_round_complete."""
        manager = create_hook_manager()
        mock_analytics = Mock()
        mock_analytics.on_round_complete = Mock()

        registry = create_hook_handler_registry(
            manager,
            analytics=mock_analytics,
        )

        # Trigger hook
        await manager.trigger(HookType.POST_ROUND, ctx=None, round_num=3)

        # Verify call
        mock_analytics.on_round_complete.assert_called_once_with(None, 3)

    @pytest.mark.asyncio
    async def test_agent_response_handler(self) -> None:
        """POST_GENERATE hook calls analytics.on_agent_response."""
        manager = create_hook_manager()
        mock_analytics = Mock()
        mock_analytics.on_agent_response = Mock()

        registry = create_hook_handler_registry(
            manager,
            analytics=mock_analytics,
        )

        mock_agent = Mock()
        await manager.trigger(HookType.POST_GENERATE, agent=mock_agent, response="test")

        mock_analytics.on_agent_response.assert_called_once_with(mock_agent, "test")

    @pytest.mark.asyncio
    async def test_debate_complete_handler(self) -> None:
        """POST_DEBATE hook calls analytics.on_debate_complete."""
        manager = create_hook_manager()
        mock_analytics = Mock()
        mock_analytics.on_debate_complete = Mock()

        registry = create_hook_handler_registry(
            manager,
            analytics=mock_analytics,
        )

        mock_result = Mock()
        await manager.trigger(HookType.POST_DEBATE, ctx=None, result=mock_result)

        mock_analytics.on_debate_complete.assert_called_once_with(None, mock_result)


class TestMemoryHandlers:
    """Test memory subsystem hook wiring."""

    @pytest.mark.asyncio
    async def test_continuum_debate_end(self) -> None:
        """POST_DEBATE hook calls continuum_memory.on_debate_end."""
        manager = create_hook_manager()
        mock_continuum = Mock()
        mock_continuum.on_debate_end = Mock()

        registry = create_hook_handler_registry(
            manager,
            continuum_memory=mock_continuum,
        )

        mock_result = Mock()
        await manager.trigger(HookType.POST_DEBATE, ctx=None, result=mock_result)

        mock_continuum.on_debate_end.assert_called_once_with(None, mock_result)

    @pytest.mark.asyncio
    async def test_consensus_memory_on_consensus(self) -> None:
        """POST_CONSENSUS hook calls consensus_memory.on_consensus_reached."""
        manager = create_hook_manager()
        mock_consensus = Mock()
        mock_consensus.on_consensus_reached = Mock()

        registry = create_hook_handler_registry(
            manager,
            consensus_memory=mock_consensus,
        )

        await manager.trigger(
            HookType.POST_CONSENSUS,
            ctx=None,
            consensus_text="Agreement reached",
            confidence=0.85,
        )

        mock_consensus.on_consensus_reached.assert_called_once_with(
            None, "Agreement reached", 0.85
        )


class TestCalibrationHandlers:
    """Test calibration subsystem hook wiring."""

    @pytest.mark.asyncio
    async def test_vote_handler(self) -> None:
        """POST_VOTE hook calls calibration_tracker.on_vote."""
        manager = create_hook_manager()
        mock_calibration = Mock()
        mock_calibration.on_vote = Mock()

        registry = create_hook_handler_registry(
            manager,
            calibration_tracker=mock_calibration,
        )

        mock_vote = Mock()
        await manager.trigger(HookType.POST_VOTE, ctx=None, vote=mock_vote)

        mock_calibration.on_vote.assert_called_once_with(None, mock_vote)

    @pytest.mark.asyncio
    async def test_debate_outcome_handler(self) -> None:
        """POST_DEBATE hook calls calibration_tracker.on_debate_outcome."""
        manager = create_hook_manager()
        mock_calibration = Mock()
        mock_calibration.on_debate_outcome = Mock()

        registry = create_hook_handler_registry(
            manager,
            calibration_tracker=mock_calibration,
        )

        mock_result = Mock()
        await manager.trigger(HookType.POST_DEBATE, ctx=None, result=mock_result)

        mock_calibration.on_debate_outcome.assert_called_once_with(None, mock_result)


class TestOutcomeHandlers:
    """Test outcome tracker hook wiring."""

    @pytest.mark.asyncio
    async def test_record_outcome(self) -> None:
        """POST_DEBATE hook calls outcome_tracker.record_outcome."""
        manager = create_hook_manager()
        mock_outcome = Mock()
        mock_outcome.record_outcome = Mock()

        registry = create_hook_handler_registry(
            manager,
            outcome_tracker=mock_outcome,
        )

        mock_result = Mock()
        await manager.trigger(HookType.POST_DEBATE, ctx=None, result=mock_result)

        mock_outcome.record_outcome.assert_called_once_with(None, mock_result)

    @pytest.mark.asyncio
    async def test_convergence_handler(self) -> None:
        """ON_CONVERGENCE hook calls outcome_tracker.on_convergence."""
        manager = create_hook_manager()
        mock_outcome = Mock()
        mock_outcome.on_convergence = Mock()

        registry = create_hook_handler_registry(
            manager,
            outcome_tracker=mock_outcome,
        )

        await manager.trigger(HookType.ON_CONVERGENCE, ctx=None)

        mock_outcome.on_convergence.assert_called_once_with(None)


class TestPerformanceHandlers:
    """Test performance monitoring hook wiring."""

    @pytest.mark.asyncio
    async def test_record_response(self) -> None:
        """POST_GENERATE hook calls performance_monitor.record_response."""
        manager = create_hook_manager()
        mock_perf = Mock()
        mock_perf.record_response = Mock()

        registry = create_hook_handler_registry(
            manager,
            performance_monitor=mock_perf,
        )

        mock_agent = Mock()
        await manager.trigger(
            HookType.POST_GENERATE,
            agent=mock_agent,
            response="test",
            latency_ms=150.5,
        )

        mock_perf.record_response.assert_called_once_with(mock_agent, "test", 150.5)

    @pytest.mark.asyncio
    async def test_record_round(self) -> None:
        """POST_ROUND hook calls performance_monitor.record_round."""
        manager = create_hook_manager()
        mock_perf = Mock()
        mock_perf.record_round = Mock()

        registry = create_hook_handler_registry(
            manager,
            performance_monitor=mock_perf,
        )

        await manager.trigger(
            HookType.POST_ROUND,
            ctx=None,
            round_num=2,
            duration_ms=5000.0,
        )

        mock_perf.record_round.assert_called_once_with(None, 2, 5000.0)


class TestSelectionFeedbackHandlers:
    """Test selection feedback loop hook wiring."""

    @pytest.mark.asyncio
    async def test_selection_feedback_outcome(self) -> None:
        """POST_DEBATE hook calls selection_feedback.record_debate_outcome."""
        manager = create_hook_manager()
        mock_feedback = Mock()
        mock_feedback.record_debate_outcome = Mock()

        registry = create_hook_handler_registry(
            manager,
            selection_feedback=mock_feedback,
        )

        mock_result = Mock()
        await manager.trigger(HookType.POST_DEBATE, ctx=None, result=mock_result)

        mock_feedback.record_debate_outcome.assert_called_once_with(None, mock_result)


class TestDetectionHandlers:
    """Test detection subsystem hook wiring."""

    @pytest.mark.asyncio
    async def test_trickster_consensus_check(self) -> None:
        """PRE_CONSENSUS hook calls trickster.check_consensus."""
        manager = create_hook_manager()
        mock_trickster = Mock()
        mock_trickster.check_consensus = Mock()

        registry = create_hook_handler_registry(
            manager,
            trickster=mock_trickster,
        )

        mock_votes = [Mock(), Mock()]
        await manager.trigger(HookType.PRE_CONSENSUS, ctx=None, votes=mock_votes)

        mock_trickster.check_consensus.assert_called_once_with(None, mock_votes)

    @pytest.mark.asyncio
    async def test_flip_detector_check(self) -> None:
        """POST_ROUND hook calls flip_detector.check_positions."""
        manager = create_hook_manager()
        mock_flip = Mock()
        mock_flip.check_positions = Mock()

        registry = create_hook_handler_registry(
            manager,
            flip_detector=mock_flip,
        )

        positions = {"agent1": "pro", "agent2": "con"}
        await manager.trigger(
            HookType.POST_ROUND,
            ctx=None,
            round_num=2,
            positions=positions,
        )

        mock_flip.check_positions.assert_called_once_with(None, 2, positions)


class TestErrorIsolation:
    """Test that handler errors don't cascade."""

    @pytest.mark.asyncio
    async def test_handler_exception_isolation(self) -> None:
        """Handler exception doesn't prevent other handlers."""
        manager = create_hook_manager()

        # First handler throws
        mock_analytics = Mock()
        mock_analytics.on_round_complete = Mock(side_effect=ValueError("test error"))

        # Second handler should still run
        mock_perf = Mock()
        mock_perf.record_round = Mock()

        registry = create_hook_handler_registry(
            manager,
            analytics=mock_analytics,
            performance_monitor=mock_perf,
        )

        # Trigger - should not raise
        await manager.trigger(HookType.POST_ROUND, ctx=None, round_num=1, duration_ms=100.0)

        # Both were attempted
        mock_analytics.on_round_complete.assert_called_once()
        mock_perf.record_round.assert_called_once()

    @pytest.mark.asyncio
    async def test_missing_method_graceful(self) -> None:
        """Missing subsystem method doesn't crash registration."""
        manager = create_hook_manager()

        # Mock without expected methods
        mock_incomplete = Mock(spec=[])  # No methods

        registry = create_hook_handler_registry(
            manager,
            analytics=mock_incomplete,
        )

        # Should register 0 handlers (no methods available)
        assert registry.registered_count == 0


class TestMultipleSubsystems:
    """Test wiring multiple subsystems together."""

    @pytest.mark.asyncio
    async def test_multiple_post_debate_handlers(self) -> None:
        """Multiple subsystems can handle POST_DEBATE."""
        manager = create_hook_manager()

        mock_analytics = Mock()
        mock_analytics.on_debate_complete = Mock()

        mock_continuum = Mock()
        mock_continuum.on_debate_end = Mock()

        mock_calibration = Mock()
        mock_calibration.on_debate_outcome = Mock()

        mock_outcome = Mock()
        mock_outcome.record_outcome = Mock()

        registry = create_hook_handler_registry(
            manager,
            analytics=mock_analytics,
            continuum_memory=mock_continuum,
            calibration_tracker=mock_calibration,
            outcome_tracker=mock_outcome,
        )

        mock_result = Mock()
        await manager.trigger(HookType.POST_DEBATE, ctx=None, result=mock_result)

        # All handlers called
        mock_analytics.on_debate_complete.assert_called_once()
        mock_continuum.on_debate_end.assert_called_once()
        mock_calibration.on_debate_outcome.assert_called_once()
        mock_outcome.record_outcome.assert_called_once()


class TestCreateHelper:
    """Test create_hook_handler_registry helper."""

    def test_auto_register_true(self) -> None:
        """auto_register=True registers handlers immediately."""
        manager = create_hook_manager()
        mock_analytics = Mock()
        mock_analytics.on_round_complete = Mock()

        registry = create_hook_handler_registry(
            manager,
            analytics=mock_analytics,
            auto_register=True,
        )

        assert registry.is_registered

    def test_auto_register_false(self) -> None:
        """auto_register=False doesn't register handlers."""
        manager = create_hook_manager()
        mock_analytics = Mock()
        mock_analytics.on_round_complete = Mock()

        registry = create_hook_handler_registry(
            manager,
            analytics=mock_analytics,
            auto_register=False,
        )

        assert not registry.is_registered


class TestSubsystemCoordinatorIntegration:
    """Test integration with SubsystemCoordinator."""

    def test_coordinator_auto_init_hook_handlers(self) -> None:
        """SubsystemCoordinator auto-initializes HookHandlerRegistry."""
        from aragora.debate.subsystem_coordinator import SubsystemCoordinator
        from aragora.debate.hooks import create_hook_manager

        manager = create_hook_manager()

        coordinator = SubsystemCoordinator(
            hook_manager=manager,
            enable_hook_handlers=True,
        )

        assert coordinator.hook_handler_registry is not None
        assert coordinator.has_hook_handlers

    def test_coordinator_respects_disable_flag(self) -> None:
        """SubsystemCoordinator respects enable_hook_handlers=False."""
        from aragora.debate.subsystem_coordinator import SubsystemCoordinator
        from aragora.debate.hooks import create_hook_manager

        manager = create_hook_manager()

        coordinator = SubsystemCoordinator(
            hook_manager=manager,
            enable_hook_handlers=False,
        )

        assert coordinator.hook_handler_registry is None
        assert not coordinator.has_hook_handlers

    def test_coordinator_status_includes_hooks(self) -> None:
        """SubsystemCoordinator.get_status includes hook info."""
        from aragora.debate.subsystem_coordinator import SubsystemCoordinator
        from aragora.debate.hooks import create_hook_manager

        manager = create_hook_manager()

        coordinator = SubsystemCoordinator(
            hook_manager=manager,
            enable_hook_handlers=True,
        )

        status = coordinator.get_status()

        assert "hook_manager" in status["subsystems"]
        assert "hook_handler_registry" in status["subsystems"]
        assert "hook_handlers" in status["capabilities"]
        assert "hook_handlers_registered" in status
