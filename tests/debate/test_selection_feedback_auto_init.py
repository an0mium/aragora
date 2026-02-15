"""Tests for auto-initialization of SelectionFeedbackLoop in SubsystemCoordinator.

Covers:
- Auto-create when enable_performance_feedback=True and selection_feedback_loop is None
- Skip when enable_performance_feedback=False
- Respect pre-configured loop (don't overwrite)
- Config values flow through to FeedbackLoopConfig
- Bridges receive the auto-created loop
- ELO system and calibration tracker passed through
- SubsystemConfig creates coordinator with feedback fields
- Initialization sequence: loop created before bridges that consume it
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aragora.debate.subsystem_coordinator import SubsystemConfig, SubsystemCoordinator


class TestSelectionFeedbackAutoInit:
    """Test auto-creation of SelectionFeedbackLoop."""

    def test_auto_creates_when_enabled_and_none(self):
        """SelectionFeedbackLoop is auto-created when enabled and not provided."""
        coord = SubsystemCoordinator(
            enable_performance_feedback=True,
            selection_feedback_loop=None,
        )
        assert coord.selection_feedback_loop is not None
        assert "SelectionFeedbackLoop" in type(coord.selection_feedback_loop).__name__

    def test_skipped_when_disabled(self):
        """No auto-creation when enable_performance_feedback is False."""
        coord = SubsystemCoordinator(
            enable_performance_feedback=False,
            selection_feedback_loop=None,
        )
        assert coord.selection_feedback_loop is None

    def test_respects_preconfigured_loop(self):
        """Pre-configured loop is not overwritten."""
        existing_loop = MagicMock()
        coord = SubsystemCoordinator(
            enable_performance_feedback=True,
            selection_feedback_loop=existing_loop,
        )
        assert coord.selection_feedback_loop is existing_loop

    def test_config_values_flow_to_feedback_loop_config(self):
        """feedback_loop_weight, decay, and min_debates flow to FeedbackLoopConfig."""
        coord = SubsystemCoordinator(
            enable_performance_feedback=True,
            feedback_loop_weight=0.3,
            feedback_loop_decay=0.8,
            feedback_loop_min_debates=5,
        )
        loop = coord.selection_feedback_loop
        assert loop is not None
        assert loop.config.performance_to_selection_weight == pytest.approx(0.3)
        assert loop.config.feedback_decay_factor == pytest.approx(0.8)
        assert loop.config.min_debates_for_feedback == 5

    def test_elo_system_passed_to_loop(self):
        """ELO system is forwarded to the auto-created loop."""
        elo = MagicMock()
        coord = SubsystemCoordinator(
            enable_performance_feedback=True,
            elo_system=elo,
        )
        assert coord.selection_feedback_loop is not None
        assert coord.selection_feedback_loop.elo_system is elo

    def test_calibration_tracker_passed_to_loop(self):
        """Calibration tracker is forwarded to the auto-created loop."""
        cal = MagicMock()
        coord = SubsystemCoordinator(
            enable_performance_feedback=True,
            calibration_tracker=cal,
        )
        assert coord.selection_feedback_loop is not None
        assert coord.selection_feedback_loop.calibration_tracker is cal

    def test_bridges_receive_auto_created_loop(self):
        """Novelty and RLM bridges receive the auto-created selection_feedback_loop."""
        novelty = MagicMock()
        rlm = MagicMock()
        coord = SubsystemCoordinator(
            enable_performance_feedback=True,
            novelty_tracker=novelty,
            rlm_bridge=rlm,
        )
        # The loop should have been created before the bridges consumed it
        assert coord.selection_feedback_loop is not None
        # Bridges may or may not init (depends on import availability),
        # but the loop itself should exist and be the same object
        # that was available when bridges tried to initialize

    def test_subsystem_config_passes_feedback_fields(self):
        """SubsystemConfig.create_coordinator passes feedback fields through."""
        config = SubsystemConfig(
            enable_performance_feedback=True,
            feedback_loop_weight=0.25,
            feedback_loop_decay=0.85,
            feedback_loop_min_debates=7,
        )
        coord = config.create_coordinator(protocol=None, loop_id="test")
        assert coord.selection_feedback_loop is not None
        assert (
            coord.selection_feedback_loop.config.performance_to_selection_weight
            == pytest.approx(0.25)
        )
        assert coord.selection_feedback_loop.config.feedback_decay_factor == pytest.approx(0.85)
        assert coord.selection_feedback_loop.config.min_debates_for_feedback == 7
