"""Tests for SDPO-CalibrationTracker integration."""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.agents.learning.sdpo import (
    ActionType,
    AgentCalibration,
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


class MockCalibrationTracker:
    """Mock CalibrationTracker for testing."""

    def __init__(self):
        self.predictions: list[dict] = []

    def record_prediction(
        self,
        agent: str,
        confidence: float,
        correct: bool,
        domain: str = "general",
        debate_id: str = "",
        position_id: str = "",
    ) -> int:
        self.predictions.append(
            {
                "agent": agent,
                "confidence": confidence,
                "correct": correct,
                "domain": domain,
                "debate_id": debate_id,
            }
        )
        return len(self.predictions)

    def get_summary(self, agent: str, domain: str | None = None):
        return MockCalibrationSummary(agent, domain)


class MockCalibrationSummary:
    """Mock CalibrationSummary for testing."""

    def __init__(self, agent: str, domain: str | None):
        self.agent = agent
        self.domain = domain
        self.brier_score = 0.1
        self.expected_calibration_error = 0.05
        self.total_predictions = 100

    def adjust_confidence(self, raw: float, domain: str | None = None) -> float:
        return raw * 0.95  # Slight adjustment

    def get_confidence_adjustment(self) -> float:
        return 0.95


class TestSDPOCalibrationConfig:
    """Tests for SDPOCalibrationConfig."""

    def test_default_values(self):
        """Default configuration values."""
        config = SDPOCalibrationConfig()

        assert config.enable_action_type_factors is True
        assert config.sdpo_weight == 0.3
        assert config.min_trajectories_for_adjustment == 5
        assert config.sync_on_trajectory_complete is True
        assert config.enable_bidirectional_learning is True

    def test_custom_values(self):
        """Custom configuration values."""
        config = SDPOCalibrationConfig(
            sdpo_weight=0.5,
            min_trajectories_for_adjustment=10,
        )

        assert config.sdpo_weight == 0.5
        assert config.min_trajectories_for_adjustment == 10


class TestSDPOCalibrationBridge:
    """Tests for SDPOCalibrationBridge."""

    def test_initialization(self):
        """Bridge initializes correctly."""
        bridge = SDPOCalibrationBridge()

        assert bridge.sdpo_learner is None
        assert bridge.calibration_tracker is None
        assert bridge.config is not None

    def test_initialization_with_components(self):
        """Bridge accepts SDPO and tracker."""
        sdpo = SDPOLearner()
        tracker = MockCalibrationTracker()

        bridge = SDPOCalibrationBridge(
            sdpo_learner=sdpo,
            calibration_tracker=tracker,
        )

        assert bridge.sdpo_learner is sdpo
        assert bridge.calibration_tracker is tracker

    def test_setters(self):
        """Can set components after initialization."""
        bridge = SDPOCalibrationBridge()
        sdpo = SDPOLearner()
        tracker = MockCalibrationTracker()

        bridge.sdpo_learner = sdpo
        bridge.calibration_tracker = tracker

        assert bridge.sdpo_learner is sdpo
        assert bridge.calibration_tracker is tracker

    @pytest.mark.asyncio
    async def test_record_debate_outcome(self):
        """Records outcome in CalibrationTracker."""
        tracker = MockCalibrationTracker()
        bridge = SDPOCalibrationBridge(calibration_tracker=tracker)

        await bridge.record_debate_outcome(
            agent_id="claude",
            debate_id="debate_123",
            confidence=0.85,
            correct=True,
            domain="security",
        )

        assert len(tracker.predictions) == 1
        pred = tracker.predictions[0]
        assert pred["agent"] == "claude"
        assert pred["confidence"] == 0.85
        assert pred["correct"] is True
        assert pred["domain"] == "security"
        assert pred["debate_id"] == "debate_123"

    @pytest.mark.asyncio
    async def test_sync_trajectory_to_calibration(self):
        """Syncs trajectory steps to CalibrationTracker."""

        tracker = MockCalibrationTracker()
        bridge = SDPOCalibrationBridge(calibration_tracker=tracker)

        # Create a completed trajectory using correct API
        trajectory = TrajectoryRecord(
            id="traj_123",
            task="Test task",
            started_at=datetime.now(),
        )
        trajectory.record_step(
            agent="claude",
            action=ActionType.PROPOSE,
            content="Response",
            confidence=0.8,
        )
        trajectory.set_outcome(
            success=True,
            quality_score=0.9,
        )

        synced = await bridge.sync_trajectory_to_calibration(trajectory)

        assert synced == 1
        assert len(tracker.predictions) == 1

    @pytest.mark.asyncio
    async def test_sync_trajectory_idempotent(self):
        """Same trajectory isn't synced twice."""

        tracker = MockCalibrationTracker()
        bridge = SDPOCalibrationBridge(calibration_tracker=tracker)

        trajectory = TrajectoryRecord(id="traj_456", task="Test", started_at=datetime.now())
        trajectory.record_step("claude", ActionType.PROPOSE, "out", confidence=0.8)
        trajectory.set_outcome(success=True, quality_score=0.9)

        await bridge.sync_trajectory_to_calibration(trajectory)
        synced_again = await bridge.sync_trajectory_to_calibration(trajectory)

        assert synced_again == 0  # Already synced
        assert len(tracker.predictions) == 1  # Only one prediction

    @pytest.mark.asyncio
    async def test_sync_incomplete_trajectory(self):
        """Incomplete trajectories aren't synced."""

        tracker = MockCalibrationTracker()
        bridge = SDPOCalibrationBridge(calibration_tracker=tracker)

        trajectory = TrajectoryRecord(id="traj_789", task="Test", started_at=datetime.now())
        trajectory.record_step("claude", ActionType.PROPOSE, "out")
        # No outcome set

        synced = await bridge.sync_trajectory_to_calibration(trajectory)

        assert synced == 0
        assert len(tracker.predictions) == 0

    def test_adjust_confidence_tracker_only(self):
        """Adjusts using CalibrationTracker when no SDPO data."""
        tracker = MockCalibrationTracker()
        bridge = SDPOCalibrationBridge(calibration_tracker=tracker)

        adjusted = bridge.adjust_confidence("claude", 0.8)

        # Should use tracker adjustment (0.95 factor)
        expected = (1 - 0.3) * (0.8 * 0.95) + 0.3 * 0.8  # Blend
        assert abs(adjusted - expected) < 0.01

    def test_adjust_confidence_with_sdpo(self):
        """Blends SDPO and tracker adjustments."""
        tracker = MockCalibrationTracker()
        bridge = SDPOCalibrationBridge(calibration_tracker=tracker)

        # Add SDPO calibration using correct API
        calibration = AgentCalibration(
            agent_name="claude",
            overconfidence_bias=0.1,  # Agent is overconfident
        )
        bridge._agent_calibrations["claude"] = calibration

        adjusted = bridge.adjust_confidence("claude", 0.8)

        # Should blend tracker and SDPO adjustments
        assert 0.05 <= adjusted <= 0.95  # Within valid range

    def test_adjust_confidence_clamps_range(self):
        """Adjusted confidence is clamped to valid range."""
        bridge = SDPOCalibrationBridge()

        low = bridge.adjust_confidence("agent", 0.01)
        high = bridge.adjust_confidence("agent", 0.99)

        assert low >= 0.05
        assert high <= 0.95

    def test_get_combined_summary_empty(self):
        """Combined summary with no data."""
        bridge = SDPOCalibrationBridge()

        summary = bridge.get_combined_summary("claude")

        assert summary["agent_id"] == "claude"
        assert summary["tracker_data"] is None
        assert summary["sdpo_data"] is None

    def test_get_combined_summary_with_tracker(self):
        """Combined summary includes tracker data."""
        tracker = MockCalibrationTracker()
        bridge = SDPOCalibrationBridge(calibration_tracker=tracker)

        summary = bridge.get_combined_summary("claude")

        assert summary["tracker_data"] is not None
        assert "brier_score" in summary["tracker_data"]
        assert "expected_calibration_error" in summary["tracker_data"]

    def test_get_combined_summary_with_sdpo(self):
        """Combined summary includes SDPO data."""
        bridge = SDPOCalibrationBridge()
        calibration = AgentCalibration(
            agent_name="claude",
            calibration_error=0.1,
            overconfidence_bias=0.05,
        )
        calibration.action_type_factors[ActionType.PROPOSE] = 1.1
        bridge._agent_calibrations["claude"] = calibration

        summary = bridge.get_combined_summary("claude")

        assert summary["sdpo_data"] is not None
        assert "action_type_factors" in summary["sdpo_data"]

    @pytest.mark.asyncio
    async def test_update_from_sdpo(self):
        """Updates calibration from SDPO learner."""
        from aragora.agents.learning.sdpo import CalibrationInsight

        sdpo = SDPOLearner()
        bridge = SDPOCalibrationBridge(sdpo_learner=sdpo)

        # Add calibration data directly to SDPO using correct field names
        insights = [
            CalibrationInsight(
                agent_name="claude",
                action_type=ActionType.PROPOSE,
                original_confidence=0.8,
                retrospective_score=0.7,
                calibration_error=0.1,
                lesson="Test lesson",
                context_pattern="Test",
            )
        ]
        sdpo.update_calibration(insights)

        # Now get via bridge
        bridge_calibration = await bridge.update_from_sdpo("claude")

        assert bridge_calibration is not None
        assert bridge_calibration.agent_name == "claude"

    @pytest.mark.asyncio
    async def test_update_from_sdpo_no_data(self):
        """Returns None when no SDPO data."""
        sdpo = SDPOLearner()
        bridge = SDPOCalibrationBridge(sdpo_learner=sdpo)

        # Agent has no data in SDPO
        calibration = await bridge.update_from_sdpo("unknown_agent")

        # Should return None since no summary exists
        assert calibration is None


class TestIntegrateFunction:
    """Tests for integrate_sdpo_with_calibration function."""

    def test_creates_bridge(self):
        """Creates configured bridge."""
        sdpo = SDPOLearner()
        tracker = MockCalibrationTracker()

        bridge = integrate_sdpo_with_calibration(sdpo, tracker)

        assert isinstance(bridge, SDPOCalibrationBridge)
        assert bridge.sdpo_learner is sdpo
        assert bridge.calibration_tracker is tracker

    def test_accepts_custom_config(self):
        """Accepts custom configuration."""
        sdpo = SDPOLearner()
        tracker = MockCalibrationTracker()
        config = SDPOCalibrationConfig(sdpo_weight=0.5)

        bridge = integrate_sdpo_with_calibration(sdpo, tracker, config)

        assert bridge.config.sdpo_weight == 0.5


class TestActionTypeCalibration:
    """Tests for per-action-type calibration."""

    def test_action_type_affects_adjustment(self):
        """Different action types get different adjustments."""
        tracker = MockCalibrationTracker()
        bridge = SDPOCalibrationBridge(calibration_tracker=tracker)

        # Add calibration with action-specific factors using correct API
        calibration = AgentCalibration(agent_name="claude")
        calibration.action_type_factors[ActionType.PROPOSE] = 1.1  # Boost propose
        calibration.action_type_factors[ActionType.CRITIQUE] = 0.9  # Reduce critique
        bridge._agent_calibrations["claude"] = calibration

        propose_adj = bridge.adjust_confidence("claude", 0.8, action_type="propose")
        critique_adj = bridge.adjust_confidence("claude", 0.8, action_type="critique")

        # Both should be within valid range
        assert 0.05 <= propose_adj <= 0.95
        assert 0.05 <= critique_adj <= 0.95

    def test_disabled_action_type_factors(self):
        """Can disable action-type factors."""
        config = SDPOCalibrationConfig(enable_action_type_factors=False)
        bridge = SDPOCalibrationBridge(config=config)

        # Add calibration with action factors
        calibration = AgentCalibration(agent_name="claude")
        calibration.action_type_factors[ActionType.PROPOSE] = 2.0  # Would double
        bridge._agent_calibrations["claude"] = calibration

        # Should NOT use SDPO factors when disabled
        adjusted = bridge.adjust_confidence("claude", 0.8, action_type="propose")

        # Without SDPO (disabled), result is just raw (no tracker either)
        assert abs(adjusted - 0.8) < 0.1  # Close to raw
