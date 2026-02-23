"""
SDPO-CalibrationTracker Integration.

Bridges Self-Distillation Policy Optimization learning with the CalibrationTracker
to provide enhanced confidence calibration based on trajectory outcomes.

Key features:
- Uses SDPO trajectory outcomes to inform calibration
- Provides per-action-type calibration factors
- Enables continuous learning from debate outcomes
- Feeds SDPO insights back into temperature scaling

Usage:
    from aragora.agents.learning.sdpo_calibration import (
        SDPOCalibrationBridge,
        integrate_sdpo_with_calibration,
    )

    # Create bridge
    bridge = SDPOCalibrationBridge(
        sdpo_learner=learner,
        calibration_tracker=tracker,
    )

    # Record outcome and update both systems
    await bridge.record_debate_outcome(
        agent_id="claude",
        debate_id="debate_123",
        confidence=0.85,
        correct=True,
        domain="security",
        trajectory_id=trajectory.id,
    )

    # Get SDPO-enhanced confidence adjustment
    adjusted = bridge.adjust_confidence("claude", 0.8, domain="security")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aragora.agents.calibration import CalibrationTracker
    from aragora.agents.learning.sdpo import (
        AgentCalibration,
        SDPOLearner,
        TrajectoryRecord,
    )

logger = logging.getLogger(__name__)


@dataclass
class SDPOCalibrationConfig:
    """Configuration for SDPO-Calibration integration.

    Attributes:
        enable_action_type_factors: Use per-action-type calibration
        sdpo_weight: Weight of SDPO adjustment vs base calibration (0-1)
        min_trajectories_for_adjustment: Min trajectories before using SDPO
        sync_on_trajectory_complete: Auto-sync to calibration on trajectory end
        enable_bidirectional_learning: Feed calibration back to SDPO
    """

    enable_action_type_factors: bool = True
    sdpo_weight: float = 0.3
    min_trajectories_for_adjustment: int = 5
    sync_on_trajectory_complete: bool = True
    enable_bidirectional_learning: bool = True


class SDPOCalibrationBridge:
    """Bridges SDPO learning with CalibrationTracker.

    Provides enhanced confidence calibration by:
    1. Using SDPO trajectory outcomes as calibration data
    2. Applying per-action-type calibration factors
    3. Blending SDPO insights with temperature scaling
    """

    def __init__(
        self,
        sdpo_learner: SDPOLearner | None = None,
        calibration_tracker: CalibrationTracker | None = None,
        config: SDPOCalibrationConfig | None = None,
    ):
        """Initialize the bridge.

        Args:
            sdpo_learner: SDPO learner instance
            calibration_tracker: Calibration tracker instance
            config: Integration configuration
        """
        self._sdpo = sdpo_learner
        self._tracker = calibration_tracker
        self.config = config or SDPOCalibrationConfig()

        # Cache for agent calibrations
        self._agent_calibrations: dict[str, AgentCalibration] = {}
        self._synced_trajectories: set[str] = set()

    @property
    def sdpo_learner(self) -> SDPOLearner | None:
        """Get the SDPO learner."""
        return self._sdpo

    @sdpo_learner.setter
    def sdpo_learner(self, learner: SDPOLearner) -> None:
        """Set the SDPO learner."""
        self._sdpo = learner

    @property
    def calibration_tracker(self) -> CalibrationTracker | None:
        """Get the calibration tracker."""
        return self._tracker

    @calibration_tracker.setter
    def calibration_tracker(self, tracker: CalibrationTracker) -> None:
        """Set the calibration tracker."""
        self._tracker = tracker

    async def record_debate_outcome(
        self,
        agent_id: str,
        debate_id: str,
        confidence: float,
        correct: bool,
        domain: str = "general",
        trajectory_id: str | None = None,
        trajectory: TrajectoryRecord | None = None,
        action_type: str | None = None,
    ) -> None:
        """Record a debate outcome in both SDPO and CalibrationTracker.

        Args:
            agent_id: Agent identifier
            debate_id: Debate identifier
            confidence: Agent's stated confidence
            correct: Whether the agent was correct
            domain: Problem domain
            trajectory_id: Optional trajectory for SDPO learning
            action_type: Type of action (propose, critique, etc.)
        """
        # Record in CalibrationTracker
        if self._tracker is not None:
            self._tracker.record_prediction(
                agent=agent_id,
                confidence=confidence,
                correct=correct,
                domain=domain,
                debate_id=debate_id,
            )

        # Record trajectory outcome if we have a trajectory
        if self._sdpo is not None:
            resolved = trajectory
            if resolved is None and trajectory_id:
                resolved = self._sdpo.get_trajectory(trajectory_id)

            if resolved is not None:
                insights = await self._sdpo.evaluate_trajectory(resolved)
                if insights:
                    self._sdpo.update_calibration(insights)
                    calibration = self._sdpo.calibrations.get(agent_id)
                    if calibration is not None:
                        self._agent_calibrations[agent_id] = calibration

    async def sync_trajectory_to_calibration(
        self,
        trajectory: TrajectoryRecord,
    ) -> int:
        """Sync a completed trajectory to CalibrationTracker.

        Extracts prediction-like data from trajectory steps and records
        them as calibration data.

        Args:
            trajectory: Completed trajectory to sync

        Returns:
            Number of predictions synced
        """
        if self._tracker is None:
            logger.warning("No CalibrationTracker configured for sync")
            return 0

        if trajectory.id in self._synced_trajectories:
            logger.debug("Trajectory %s already synced", trajectory.id)
            return 0

        # Check if trajectory is complete (has an outcome)
        if trajectory.outcome is None:
            logger.warning("Trajectory %s is not complete", trajectory.id)
            return 0

        synced = 0
        outcome = trajectory.outcome

        # Extract calibration data from each step
        for step in trajectory.steps:
            # Skip steps without confidence information
            if step.confidence is None:
                continue

            # Determine correctness from outcome quality and step contribution
            step_correct = outcome.success and outcome.quality_score >= 0.6

            # Get agent from step (each step tracks its agent)
            agent_name = step.agent_name

            # Record as calibration prediction
            self._tracker.record_prediction(
                agent=agent_name,
                confidence=step.confidence,
                correct=step_correct,
                domain="general",  # Could be enhanced to get from trajectory metadata
            )
            synced += 1

        self._synced_trajectories.add(trajectory.id)
        logger.info("Synced %s predictions from trajectory %s", synced, trajectory.id)

        return synced

    def adjust_confidence(
        self,
        agent_id: str,
        raw_confidence: float,
        domain: str | None = None,
        action_type: str | None = None,
    ) -> float:
        """Adjust confidence using both SDPO and CalibrationTracker.

        Blends SDPO action-type factors with temperature scaling.

        Args:
            agent_id: Agent identifier
            raw_confidence: Raw confidence value
            domain: Optional domain
            action_type: Optional action type for SDPO adjustment

        Returns:
            Adjusted confidence value
        """
        adjusted = raw_confidence

        # Apply CalibrationTracker adjustment
        if self._tracker is not None:
            summary = self._tracker.get_summary(agent_id, domain=domain)  # type: ignore[attr-defined]
            if summary:
                tracker_adjusted = summary.adjust_confidence(raw_confidence, domain=domain)
            else:
                tracker_adjusted = raw_confidence
        else:
            tracker_adjusted = raw_confidence

        # Apply SDPO adjustment if available
        if self.config.enable_action_type_factors and agent_id in self._agent_calibrations:
            from aragora.agents.learning.sdpo import ActionType

            calibration = self._agent_calibrations[agent_id]
            # Convert action_type string to ActionType enum if needed
            if action_type:
                try:
                    at = ActionType(action_type)
                except ValueError:
                    at = ActionType.OTHER
            else:
                at = ActionType.OTHER

            factor = calibration.get_adjustment_factor(at)
            sdpo_adjusted = raw_confidence * factor
        else:
            sdpo_adjusted = raw_confidence

        # Blend adjustments
        sdpo_weight = self.config.sdpo_weight
        adjusted = (1 - sdpo_weight) * tracker_adjusted + sdpo_weight * sdpo_adjusted

        return max(0.05, min(0.95, adjusted))

    async def update_from_sdpo(self, agent_name: str) -> AgentCalibration | None:
        """Update calibration from SDPO learning.

        Args:
            agent_name: Agent name to update

        Returns:
            Updated calibration if available, None if no data
        """
        if self._sdpo is None:
            return None

        summary = self._sdpo.get_agent_summary(agent_name)
        # Check if SDPO has actual data for this agent
        if summary.get("status") == "no_data":
            return None

        # Get or create calibration
        if agent_name in self._agent_calibrations:
            calibration = self._agent_calibrations[agent_name]
        else:
            from aragora.agents.learning.sdpo import AgentCalibration

            calibration = AgentCalibration(agent_name=agent_name)

        # Update with SDPO summary data
        calibration.calibration_error = summary.get("calibration_error", 0.0)
        calibration.overconfidence_bias = summary.get("overconfidence_bias", 0.0)
        calibration.mean_confidence = summary.get("mean_confidence", 0.5)
        calibration.mean_quality = summary.get("mean_quality", 0.5)
        calibration.total_actions = summary.get("total_actions", 0)

        self._agent_calibrations[agent_name] = calibration
        return calibration

    def get_combined_summary(
        self,
        agent_id: str,
        domain: str | None = None,
    ) -> dict[str, Any]:
        """Get combined calibration summary from both systems.

        Args:
            agent_id: Agent identifier
            domain: Optional domain filter

        Returns:
            Combined summary with both tracker and SDPO data
        """
        summary: dict[str, Any] = {
            "agent_id": agent_id,
            "domain": domain,
            "tracker_data": None,
            "sdpo_data": None,
        }

        # Get CalibrationTracker data
        if self._tracker is not None:
            tracker_summary = self._tracker.get_summary(agent_id, domain=domain)  # type: ignore[attr-defined]
            if tracker_summary:
                summary["tracker_data"] = {
                    "brier_score": tracker_summary.brier_score,
                    "expected_calibration_error": tracker_summary.expected_calibration_error,
                    "total_predictions": tracker_summary.total_predictions,
                    "adjustment_factor": tracker_summary.get_confidence_adjustment(),
                }

        # Get SDPO data
        if agent_id in self._agent_calibrations:
            calibration = self._agent_calibrations[agent_id]
            # Convert action_type_factors keys to strings for JSON serialization
            action_factors = {
                k.value if hasattr(k, "value") else str(k): v
                for k, v in calibration.action_type_factors.items()
            }
            summary["sdpo_data"] = {
                "calibration_error": calibration.calibration_error,
                "overconfidence_bias": calibration.overconfidence_bias,
                "mean_confidence": calibration.mean_confidence,
                "mean_quality": calibration.mean_quality,
                "action_type_factors": action_factors,
            }

        return summary


def integrate_sdpo_with_calibration(
    sdpo_learner: SDPOLearner,
    calibration_tracker: CalibrationTracker,
    config: SDPOCalibrationConfig | None = None,
) -> SDPOCalibrationBridge:
    """Create and configure an SDPO-Calibration bridge.

    Args:
        sdpo_learner: SDPO learner instance
        calibration_tracker: Calibration tracker instance
        config: Optional configuration

    Returns:
        Configured bridge instance
    """
    bridge = SDPOCalibrationBridge(
        sdpo_learner=sdpo_learner,
        calibration_tracker=calibration_tracker,
        config=config,
    )

    logger.info(
        "Integrated SDPO with CalibrationTracker (sdpo_weight=%s)", bridge.config.sdpo_weight
    )

    return bridge


__all__ = [
    "SDPOCalibrationConfig",
    "SDPOCalibrationBridge",
    "integrate_sdpo_with_calibration",
]
