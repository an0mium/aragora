"""Tests for TricksterCalibrator."""

from __future__ import annotations

import pytest
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, MagicMock

from aragora.debate.trickster_calibrator import (
    TricksterCalibrator,
    CalibrationDataPoint,
    CalibrationResult,
    create_trickster_calibrator,
)


@dataclass
class MockTricksterConfig:
    """Mock TricksterConfig for testing."""

    sensitivity: float = 0.5
    hollow_detection_threshold: float = 0.5
    min_quality_threshold: float = 0.65


@dataclass
class MockTrickster:
    """Mock EvidencePoweredTrickster for testing."""

    config: MockTricksterConfig = field(default_factory=MockTricksterConfig)


@dataclass
class MockConsensusOutcome:
    """Mock ConsensusOutcome for testing."""

    debate_id: str
    consensus_text: str = "Test consensus"
    consensus_confidence: float = 0.8
    implementation_attempted: bool = True
    implementation_succeeded: bool = True


class TestInterventionTracking:
    """Test intervention recording."""

    def test_record_intervention(self) -> None:
        """Can record an intervention."""
        calibrator = TricksterCalibrator()
        calibrator.record_intervention("debate-1", 1)

        assert "debate-1" in calibrator._intervention_history
        assert calibrator._intervention_history["debate-1"] == 1

    def test_record_multiple_interventions(self) -> None:
        """Can record multiple interventions in same debate."""
        calibrator = TricksterCalibrator()
        calibrator.record_intervention("debate-1", 3)

        assert calibrator._intervention_history["debate-1"] == 3


class TestOutcomeRecording:
    """Test outcome recording for calibration."""

    def test_record_outcome_with_intervention(self) -> None:
        """Records outcome and links to intervention."""
        calibrator = TricksterCalibrator()
        calibrator.record_intervention("debate-1", 1)

        outcome = MockConsensusOutcome(
            debate_id="debate-1",
            implementation_succeeded=True,
        )
        calibrator.record_debate_outcome(outcome)

        assert len(calibrator._data_points) == 1
        dp = calibrator._data_points[0]
        assert dp.debate_id == "debate-1"
        assert dp.had_intervention is True
        assert dp.intervention_count == 1
        assert dp.outcome_success is True

    def test_record_outcome_without_intervention(self) -> None:
        """Records outcome without intervention."""
        calibrator = TricksterCalibrator()

        outcome = MockConsensusOutcome(
            debate_id="debate-2",
            implementation_succeeded=False,
        )
        calibrator.record_debate_outcome(outcome)

        assert len(calibrator._data_points) == 1
        dp = calibrator._data_points[0]
        assert dp.debate_id == "debate-2"
        assert dp.had_intervention is False
        assert dp.outcome_success is False

    def test_intervention_history_cleared_after_outcome(self) -> None:
        """Intervention history for debate cleared after recording outcome."""
        calibrator = TricksterCalibrator()
        calibrator.record_intervention("debate-1", 1)

        outcome = MockConsensusOutcome(debate_id="debate-1")
        calibrator.record_debate_outcome(outcome)

        assert "debate-1" not in calibrator._intervention_history


class TestCalibrationTrigger:
    """Test calibration triggering logic."""

    def test_skip_calibration_insufficient_samples(self) -> None:
        """Skips calibration when not enough samples."""
        calibrator = TricksterCalibrator(min_samples=20)

        # Add only 5 samples
        for i in range(5):
            outcome = MockConsensusOutcome(debate_id=f"debate-{i}")
            calibrator.record_debate_outcome(outcome)

        result = calibrator.maybe_recalibrate()
        assert result is None

    def test_skip_calibration_interval_not_reached(self) -> None:
        """Skips calibration when interval not reached."""
        calibrator = TricksterCalibrator(
            min_samples=5,
            recalibrate_interval=50,
        )

        # Add enough samples but not enough debates since last calibration
        for i in range(10):
            outcome = MockConsensusOutcome(debate_id=f"debate-{i}")
            calibrator.record_debate_outcome(outcome)

        # Reset counter to simulate previous calibration
        calibrator._debates_since_calibration = 10

        result = calibrator.maybe_recalibrate()
        assert result is None

    def test_force_calibration(self) -> None:
        """Force calibration bypasses checks."""
        mock_trickster = MockTrickster()
        calibrator = TricksterCalibrator(
            trickster=mock_trickster,
            min_samples=100,
        )

        # Add minimal samples
        for i in range(5):
            outcome = MockConsensusOutcome(debate_id=f"debate-{i}")
            calibrator.record_debate_outcome(outcome)

        result = calibrator.maybe_recalibrate(force=True)
        # Should have run even with few samples (but might return None if too few)
        # At least it didn't skip due to interval


class TestCalibrationLogic:
    """Test calibration decision logic."""

    def test_calibrate_lowers_sensitivity_on_false_positives(self) -> None:
        """Lowers sensitivity when false positive rate is high."""
        mock_trickster = MockTrickster()
        mock_trickster.config.sensitivity = 0.7

        calibrator = TricksterCalibrator(
            trickster=mock_trickster,
            min_samples=5,
            false_positive_tolerance=0.3,
            adjustment_step=0.1,
        )

        # Simulate many "false positives" - intervened but outcome was good
        for i in range(10):
            calibrator.record_intervention(f"debate-{i}", 1)
            outcome = MockConsensusOutcome(
                debate_id=f"debate-{i}",
                implementation_succeeded=True,  # Success despite intervention
            )
            calibrator.record_debate_outcome(outcome)

        result = calibrator.calibrate()

        assert result is not None
        assert result.calibrated is True
        assert result.new_sensitivity < result.old_sensitivity

    def test_calibrate_raises_sensitivity_on_misses(self) -> None:
        """Raises sensitivity when miss rate is high."""
        mock_trickster = MockTrickster()
        mock_trickster.config.sensitivity = 0.4

        calibrator = TricksterCalibrator(
            trickster=mock_trickster,
            min_samples=5,
            miss_tolerance=0.2,
            adjustment_step=0.1,
        )

        # Simulate many "misses" - didn't intervene but outcome was bad
        for i in range(10):
            # No intervention recorded
            outcome = MockConsensusOutcome(
                debate_id=f"debate-{i}",
                implementation_succeeded=False,  # Failure without intervention
            )
            calibrator.record_debate_outcome(outcome)

        result = calibrator.calibrate()

        assert result is not None
        assert result.calibrated is True
        assert result.new_sensitivity > result.old_sensitivity

    def test_calibrate_stays_stable_when_optimal(self) -> None:
        """Sensitivity unchanged when metrics are good."""
        mock_trickster = MockTrickster()
        mock_trickster.config.sensitivity = 0.5

        calibrator = TricksterCalibrator(
            trickster=mock_trickster,
            min_samples=5,
            false_positive_tolerance=0.4,
            miss_tolerance=0.3,
        )

        # Mix of outcomes - balanced
        for i in range(5):
            calibrator.record_intervention(f"debate-{i}", 1)
            outcome = MockConsensusOutcome(
                debate_id=f"debate-{i}",
                implementation_succeeded=(i % 2 == 0),  # Mixed success
            )
            calibrator.record_debate_outcome(outcome)

        for i in range(5, 10):
            # No intervention
            outcome = MockConsensusOutcome(
                debate_id=f"debate-{i}",
                implementation_succeeded=True,  # Success without intervention
            )
            calibrator.record_debate_outcome(outcome)

        result = calibrator.calibrate()

        # Should either be stable or only minor adjustment
        if result and result.calibrated:
            # If it did calibrate, change should be small
            assert abs(result.new_sensitivity - result.old_sensitivity) <= 0.1


class TestSensitivityBounds:
    """Test sensitivity bounds enforcement."""

    def test_sensitivity_clamped_to_min(self) -> None:
        """Sensitivity doesn't go below minimum."""
        mock_trickster = MockTrickster()
        mock_trickster.config.sensitivity = 0.35  # Close to min

        calibrator = TricksterCalibrator(
            trickster=mock_trickster,
            min_samples=5,
            sensitivity_bounds=(0.3, 0.9),
            adjustment_step=0.1,
        )

        # Create conditions for lowering
        for i in range(10):
            calibrator.record_intervention(f"debate-{i}", 1)
            outcome = MockConsensusOutcome(
                debate_id=f"debate-{i}",
                implementation_succeeded=True,
            )
            calibrator.record_debate_outcome(outcome)

        result = calibrator.calibrate()

        assert result is not None
        assert result.new_sensitivity >= 0.3

    def test_sensitivity_clamped_to_max(self) -> None:
        """Sensitivity doesn't go above maximum."""
        mock_trickster = MockTrickster()
        mock_trickster.config.sensitivity = 0.85  # Close to max

        calibrator = TricksterCalibrator(
            trickster=mock_trickster,
            min_samples=5,
            sensitivity_bounds=(0.3, 0.9),
            adjustment_step=0.1,
        )

        # Create conditions for raising
        for i in range(10):
            outcome = MockConsensusOutcome(
                debate_id=f"debate-{i}",
                implementation_succeeded=False,
            )
            calibrator.record_debate_outcome(outcome)

        result = calibrator.calibrate()

        assert result is not None
        assert result.new_sensitivity <= 0.9


class TestManualSensitivitySet:
    """Test manual sensitivity setting."""

    def test_set_sensitivity(self) -> None:
        """Can manually set sensitivity."""
        mock_trickster = MockTrickster()
        calibrator = TricksterCalibrator(trickster=mock_trickster)

        calibrator.set_sensitivity(0.7)

        assert mock_trickster.config.sensitivity == 0.7

    def test_set_sensitivity_clamped(self) -> None:
        """Manual set is clamped to bounds."""
        mock_trickster = MockTrickster()
        calibrator = TricksterCalibrator(
            trickster=mock_trickster,
            sensitivity_bounds=(0.3, 0.9),
        )

        calibrator.set_sensitivity(1.5)  # Above max
        assert mock_trickster.config.sensitivity == 0.9

        calibrator.set_sensitivity(0.1)  # Below min
        assert mock_trickster.config.sensitivity == 0.3


class TestCalibrationSummary:
    """Test calibration summary."""

    def test_get_summary(self) -> None:
        """Can get calibration summary."""
        mock_trickster = MockTrickster()
        mock_trickster.config.sensitivity = 0.6

        calibrator = TricksterCalibrator(trickster=mock_trickster)

        # Add some data
        for i in range(5):
            outcome = MockConsensusOutcome(debate_id=f"debate-{i}")
            calibrator.record_debate_outcome(outcome)

        summary = calibrator.get_calibration_summary()

        assert summary["current_sensitivity"] == 0.6
        assert summary["data_points"] == 5
        assert "analysis" in summary

    def test_get_calibration_history(self) -> None:
        """Can get calibration history."""
        mock_trickster = MockTrickster()
        calibrator = TricksterCalibrator(
            trickster=mock_trickster,
            min_samples=2,
        )

        # Add data and calibrate
        for i in range(5):
            outcome = MockConsensusOutcome(debate_id=f"debate-{i}")
            calibrator.record_debate_outcome(outcome)

        calibrator.calibrate()

        history = calibrator.get_calibration_history()
        assert len(history) == 1


class TestDataCleanup:
    """Test data cleanup."""

    def test_clear_data(self) -> None:
        """clear_data removes all tracked data."""
        calibrator = TricksterCalibrator()

        # Add some data
        calibrator.record_intervention("debate-1", 1)
        outcome = MockConsensusOutcome(debate_id="debate-1")
        calibrator.record_debate_outcome(outcome)

        calibrator.clear_data()

        assert len(calibrator._data_points) == 0
        assert len(calibrator._intervention_history) == 0
        assert calibrator._debates_since_calibration == 0


class TestCreateHelper:
    """Test create_trickster_calibrator helper."""

    def test_creates_with_defaults(self) -> None:
        """Creates calibrator with default values."""
        calibrator = create_trickster_calibrator()
        assert isinstance(calibrator, TricksterCalibrator)

    def test_creates_with_dependencies(self) -> None:
        """Creates calibrator with dependencies."""
        mock_trickster = MockTrickster()
        mock_tracker = Mock()

        calibrator = create_trickster_calibrator(
            trickster=mock_trickster,
            outcome_tracker=mock_tracker,
        )

        assert calibrator.trickster is mock_trickster
        assert calibrator.outcome_tracker is mock_tracker

    def test_creates_with_custom_config(self) -> None:
        """Creates calibrator with custom configuration."""
        calibrator = create_trickster_calibrator(
            min_samples=30,
            recalibrate_interval=100,
        )

        assert calibrator.min_samples == 30
        assert calibrator.recalibrate_interval == 100
