"""
Trickster Auto-Calibration for self-tuning hollow consensus detection.

This module connects OutcomeTracker outcomes to Trickster sensitivity,
enabling automatic adjustment of detection thresholds based on actual
debate outcomes.

The calibration logic:
1. Track debates where Trickster intervened vs didn't intervene
2. Compare outcomes: Did intervention actually help?
3. Adjust sensitivity:
   - Too many false positives (intervened but outcome was good) → lower sensitivity
   - Too many misses (didn't intervene but outcome was bad) → raise sensitivity

Usage:
    from aragora.debate.trickster_calibrator import TricksterCalibrator

    calibrator = TricksterCalibrator(trickster, outcome_tracker)

    # After each debate batch
    result = calibrator.maybe_recalibrate()
    if result:
        print(f"Adjusted sensitivity: {result.old_sensitivity} -> {result.new_sensitivity}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from aragora.debate.outcome_tracker import ConsensusOutcome, OutcomeTracker
    from aragora.debate.trickster import EvidencePoweredTrickster, TricksterConfig

logger = logging.getLogger(__name__)


@dataclass
class CalibrationDataPoint:
    """A single data point for calibration analysis."""

    debate_id: str
    had_intervention: bool
    intervention_count: int
    outcome_success: bool
    outcome_confidence: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class CalibrationResult:
    """Result of a calibration run."""

    calibrated: bool
    old_sensitivity: float
    new_sensitivity: float
    reason: str
    sample_size: int
    false_positive_rate: float  # Intervened when not needed
    miss_rate: float  # Didn't intervene when needed
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class TricksterCalibrator:
    """Auto-calibrates Trickster sensitivity based on outcome history.

    This class analyzes the correlation between Trickster interventions
    and debate outcomes to find the optimal sensitivity setting.

    The core insight is that optimal sensitivity balances:
    1. Not missing real hollow consensus (that leads to bad outcomes)
    2. Not over-intervening (which disrupts good debates)

    Attributes:
        trickster: The Trickster instance to calibrate
        outcome_tracker: OutcomeTracker for debate outcomes
        min_samples: Minimum outcomes before calibrating
        recalibrate_interval: Debates between recalibrations
        sensitivity_bounds: (min, max) allowed sensitivity
        adjustment_step: How much to adjust per calibration
    """

    trickster: Optional["EvidencePoweredTrickster"] = None
    outcome_tracker: Optional["OutcomeTracker"] = None

    # Calibration settings
    min_samples: int = 20  # Minimum outcomes before calibrating
    recalibrate_interval: int = 50  # Recalibrate every N debates
    sensitivity_bounds: Tuple[float, float] = (0.3, 0.9)  # Min/max sensitivity
    adjustment_step: float = 0.05  # How much to adjust per calibration

    # Thresholds for decision
    false_positive_tolerance: float = 0.3  # Max false positive rate before lowering
    miss_tolerance: float = 0.2  # Max miss rate before raising

    # State tracking
    _data_points: List[CalibrationDataPoint] = field(default_factory=list, repr=False)
    _intervention_history: Dict[str, int] = field(default_factory=dict, repr=False)
    _debates_since_calibration: int = field(default=0, repr=False)
    _calibration_history: List[CalibrationResult] = field(default_factory=list, repr=False)

    def record_intervention(self, debate_id: str, intervention_count: int = 1) -> None:
        """Record that Trickster intervened in a debate.

        Call this after each debate to track interventions.

        Args:
            debate_id: ID of the debate
            intervention_count: Number of interventions in this debate
        """
        self._intervention_history[debate_id] = intervention_count
        logger.debug(
            "trickster_intervention_recorded debate_id=%s count=%d",
            debate_id,
            intervention_count,
        )

    def record_debate_outcome(self, outcome: "ConsensusOutcome") -> None:
        """Record a debate outcome for calibration analysis.

        Call this after recording an outcome to OutcomeTracker.

        Args:
            outcome: The debate outcome
        """
        debate_id = outcome.debate_id
        had_intervention = debate_id in self._intervention_history
        intervention_count = self._intervention_history.get(debate_id, 0)

        data_point = CalibrationDataPoint(
            debate_id=debate_id,
            had_intervention=had_intervention,
            intervention_count=intervention_count,
            outcome_success=outcome.implementation_succeeded,
            outcome_confidence=outcome.consensus_confidence,
        )

        self._data_points.append(data_point)
        self._debates_since_calibration += 1

        # Clean up intervention history
        if debate_id in self._intervention_history:
            del self._intervention_history[debate_id]

        logger.debug(
            "calibration_data_point debate_id=%s intervention=%s success=%s",
            debate_id,
            had_intervention,
            outcome.implementation_succeeded,
        )

    def maybe_recalibrate(self, force: bool = False) -> Optional[CalibrationResult]:
        """Check if recalibration is needed and run if so.

        Args:
            force: Force recalibration even if interval not reached

        Returns:
            CalibrationResult if calibration occurred, None otherwise
        """
        if not force:
            if len(self._data_points) < self.min_samples:
                logger.debug(
                    "calibration_skipped reason=insufficient_data count=%d min=%d",
                    len(self._data_points),
                    self.min_samples,
                )
                return None

            if self._debates_since_calibration < self.recalibrate_interval:
                logger.debug(
                    "calibration_skipped reason=interval debates=%d interval=%d",
                    self._debates_since_calibration,
                    self.recalibrate_interval,
                )
                return None

        return self.calibrate()

    def calibrate(self) -> Optional[CalibrationResult]:
        """Run calibration analysis and adjust Trickster sensitivity.

        Returns:
            CalibrationResult with adjustment details
        """
        if self.trickster is None:
            logger.debug("calibration_skipped reason=no_trickster")
            return None

        if len(self._data_points) < self.min_samples // 2:
            logger.debug(
                "calibration_skipped reason=minimum_samples count=%d",
                len(self._data_points),
            )
            return None

        # Analyze outcomes
        analysis = self._analyze_outcomes()
        false_positive_rate = analysis["false_positive_rate"]
        miss_rate = analysis["miss_rate"]

        # Get current sensitivity
        current_sensitivity = self.trickster.config.sensitivity
        new_sensitivity = current_sensitivity

        # Determine adjustment direction
        reason = "optimal"
        if false_positive_rate > self.false_positive_tolerance:
            # Too many false positives - lower sensitivity
            new_sensitivity = max(
                self.sensitivity_bounds[0],
                current_sensitivity - self.adjustment_step,
            )
            reason = f"false_positive_rate={false_positive_rate:.2f} > {self.false_positive_tolerance}"

        elif miss_rate > self.miss_tolerance:
            # Too many misses - raise sensitivity
            new_sensitivity = min(
                self.sensitivity_bounds[1],
                current_sensitivity + self.adjustment_step,
            )
            reason = f"miss_rate={miss_rate:.2f} > {self.miss_tolerance}"

        # Check if we actually changed
        calibrated = abs(new_sensitivity - current_sensitivity) > 0.001

        result = CalibrationResult(
            calibrated=calibrated,
            old_sensitivity=current_sensitivity,
            new_sensitivity=new_sensitivity,
            reason=reason,
            sample_size=len(self._data_points),
            false_positive_rate=false_positive_rate,
            miss_rate=miss_rate,
        )

        # Apply new sensitivity if changed
        if calibrated:
            self._apply_sensitivity(new_sensitivity)
            logger.info(
                "trickster_calibrated old=%.2f new=%.2f reason=%s samples=%d",
                current_sensitivity,
                new_sensitivity,
                reason,
                len(self._data_points),
            )
        else:
            logger.debug(
                "trickster_calibration_stable sensitivity=%.2f fp_rate=%.2f miss_rate=%.2f",
                current_sensitivity,
                false_positive_rate,
                miss_rate,
            )

        # Track calibration history
        self._calibration_history.append(result)
        self._debates_since_calibration = 0

        return result

    def _analyze_outcomes(self) -> Dict[str, float]:
        """Analyze outcomes to compute calibration metrics.

        Returns:
            Dict with false_positive_rate and miss_rate
        """
        # Only use recent data points
        recent_points = self._data_points[-100:]  # Last 100 debates

        # Count outcomes by category
        intervention_success = 0  # Intervened AND success (maybe unnecessary)
        intervention_failure = 0  # Intervened AND failure (intervention didn't help)
        no_intervention_success = 0  # Didn't intervene AND success (correctly skipped)
        no_intervention_failure = 0  # Didn't intervene AND failure (missed opportunity)

        for dp in recent_points:
            if dp.had_intervention:
                if dp.outcome_success:
                    intervention_success += 1
                else:
                    intervention_failure += 1
            else:
                if dp.outcome_success:
                    no_intervention_success += 1
                else:
                    no_intervention_failure += 1

        # False positive rate: Intervened when outcome was good anyway
        # (Suggests intervention wasn't needed)
        total_interventions = intervention_success + intervention_failure
        if total_interventions > 0:
            # If outcome was good despite intervention, it might have been unnecessary
            # But we can't know for sure, so we use a heuristic:
            # High success rate with interventions suggests over-intervention
            false_positive_rate = intervention_success / total_interventions
        else:
            false_positive_rate = 0.0

        # Miss rate: Didn't intervene when outcome was bad
        # (Suggests we should have intervened)
        total_no_intervention = no_intervention_success + no_intervention_failure
        if total_no_intervention > 0:
            miss_rate = no_intervention_failure / total_no_intervention
        else:
            miss_rate = 0.0

        return {
            "false_positive_rate": false_positive_rate,
            "miss_rate": miss_rate,
            "intervention_success": intervention_success,
            "intervention_failure": intervention_failure,
            "no_intervention_success": no_intervention_success,
            "no_intervention_failure": no_intervention_failure,
            "total_samples": len(recent_points),
        }

    def _apply_sensitivity(self, new_sensitivity: float) -> None:
        """Apply new sensitivity to Trickster.

        Args:
            new_sensitivity: New sensitivity value (0-1)
        """
        if self.trickster is None:
            return

        # Update sensitivity
        self.trickster.config.sensitivity = new_sensitivity

        # Recalculate hollow_detection_threshold based on new sensitivity
        # Higher sensitivity = lower threshold (more sensitive to hollow consensus)
        self.trickster.config.hollow_detection_threshold = 0.8 - (new_sensitivity * 0.6)

        # Update detector threshold if accessible
        if hasattr(self.trickster, "_detector"):
            self.trickster._detector.min_quality_threshold = self.trickster.config.min_quality_threshold

        logger.debug(
            "trickster_sensitivity_applied sensitivity=%.2f threshold=%.2f",
            new_sensitivity,
            self.trickster.config.hollow_detection_threshold,
        )

    def set_sensitivity(self, sensitivity: float) -> None:
        """Manually set Trickster sensitivity.

        Args:
            sensitivity: New sensitivity value (0-1), clamped to bounds
        """
        if self.trickster is None:
            return

        # Clamp to bounds
        sensitivity = max(self.sensitivity_bounds[0], min(self.sensitivity_bounds[1], sensitivity))
        self._apply_sensitivity(sensitivity)

    def get_calibration_summary(self) -> Dict[str, Any]:
        """Get summary of calibration state and history.

        Returns:
            Dict with calibration metrics and history
        """
        analysis = (
            self._analyze_outcomes() if len(self._data_points) >= 5 else {}
        )

        current_sensitivity = 0.5
        if self.trickster is not None:
            current_sensitivity = self.trickster.config.sensitivity

        return {
            "current_sensitivity": current_sensitivity,
            "data_points": len(self._data_points),
            "debates_since_calibration": self._debates_since_calibration,
            "calibration_count": len(self._calibration_history),
            "analysis": analysis,
            "last_calibration": (
                self._calibration_history[-1].timestamp
                if self._calibration_history
                else None
            ),
            "pending_interventions": len(self._intervention_history),
        }

    def get_calibration_history(self, limit: int = 10) -> List[CalibrationResult]:
        """Get recent calibration history.

        Args:
            limit: Maximum results to return

        Returns:
            List of CalibrationResult objects
        """
        return self._calibration_history[-limit:]

    def clear_data(self) -> None:
        """Clear all calibration data."""
        self._data_points.clear()
        self._intervention_history.clear()
        self._debates_since_calibration = 0
        logger.debug("trickster_calibrator data cleared")


def create_trickster_calibrator(
    trickster: Optional["EvidencePoweredTrickster"] = None,
    outcome_tracker: Optional["OutcomeTracker"] = None,
    **kwargs: Any,
) -> TricksterCalibrator:
    """Create a TricksterCalibrator with optional configuration.

    Args:
        trickster: The Trickster instance to calibrate
        outcome_tracker: OutcomeTracker for debate outcomes
        **kwargs: Additional configuration

    Returns:
        Configured TricksterCalibrator instance
    """
    return TricksterCalibrator(
        trickster=trickster,
        outcome_tracker=outcome_tracker,
        **kwargs,
    )


__all__ = [
    "TricksterCalibrator",
    "CalibrationDataPoint",
    "CalibrationResult",
    "create_trickster_calibrator",
]
