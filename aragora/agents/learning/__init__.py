"""
Agent Learning - Retrospective learning and calibration improvement.

This module implements learning strategies for agents:
- SDPO (Self-Distillation Policy Optimization): Learn from retrospective evaluation
- Experience Replay: Store and learn from past interactions
- Calibration Refinement: Improve confidence estimation over time
- SDPO-Calibration Bridge: Integration with CalibrationTracker
"""

from aragora.agents.learning.sdpo import (
    ActionType,
    AgentCalibration,
    ExperienceBuffer,
    RetrospectiveEvaluator,
    SDPOConfig,
    SDPOLearner,
    TrajectoryOutcome,
    TrajectoryRecord,
    TrajectoryStep,
)
from aragora.agents.learning.sdpo_calibration import (
    SDPOCalibrationBridge,
    SDPOCalibrationConfig,
    integrate_sdpo_with_calibration,
)

__all__ = [
    # Core SDPO types
    "ActionType",
    "AgentCalibration",
    "ExperienceBuffer",
    "RetrospectiveEvaluator",
    "SDPOConfig",
    "SDPOLearner",
    "TrajectoryOutcome",
    "TrajectoryRecord",
    "TrajectoryStep",
    # Calibration integration
    "SDPOCalibrationBridge",
    "SDPOCalibrationConfig",
    "integrate_sdpo_with_calibration",
]
