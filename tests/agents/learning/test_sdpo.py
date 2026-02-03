"""
Tests for SDPO (Self-Distillation Policy Optimization) module.

Tests cover:
- Trajectory recording
- Retrospective evaluation
- Calibration updates
- Experience buffer
- Persistence
"""

import asyncio
import pytest
import tempfile
from datetime import datetime
from pathlib import Path

from aragora.agents.learning.sdpo import (
    ActionType,
    TrajectoryStep,
    TrajectoryOutcome,
    TrajectoryRecord,
    CalibrationInsight,
    AgentCalibration,
    SDPOConfig,
    ExperienceBuffer,
    DefaultRetrospectiveEvaluator,
    SDPOLearner,
)


class TestActionType:
    """Tests for ActionType enum."""

    def test_all_actions_exist(self):
        """All expected actions are defined."""
        expected = [
            "propose",
            "critique",
            "synthesize",
            "judge",
            "search",
            "reason",
            "other",
        ]
        actual = [a.value for a in ActionType]
        assert set(expected) == set(actual)


class TestTrajectoryStep:
    """Tests for TrajectoryStep dataclass."""

    def test_creation(self):
        """Step can be created with required fields."""
        step = TrajectoryStep(
            id="step_1",
            timestamp=datetime.now(),
            agent_name="claude",
            action_type=ActionType.PROPOSE,
            content="We should use REST API",
            confidence=0.8,
        )
        assert step.agent_name == "claude"
        assert step.action_type == ActionType.PROPOSE
        assert step.confidence == 0.8

    def test_retrospective_fields_none_initially(self):
        """Retrospective fields are None before evaluation."""
        step = TrajectoryStep(
            id="step_1",
            timestamp=datetime.now(),
            agent_name="agent",
            action_type=ActionType.CRITIQUE,
            content="content",
        )
        assert step.retrospective_score is None
        assert step.contribution_to_outcome is None


class TestTrajectoryOutcome:
    """Tests for TrajectoryOutcome dataclass."""

    def test_creation(self):
        """Outcome can be created."""
        outcome = TrajectoryOutcome(
            success=True,
            quality_score=0.9,
            feedback="Good decision",
        )
        assert outcome.success is True
        assert outcome.quality_score == 0.9


class TestTrajectoryRecord:
    """Tests for TrajectoryRecord dataclass."""

    def test_record_step(self):
        """Steps can be recorded."""
        record = TrajectoryRecord(
            id="traj_1",
            task="Design API",
            started_at=datetime.now(),
        )

        step = record.record_step(
            agent="claude",
            action=ActionType.PROPOSE,
            content="Use GraphQL",
            confidence=0.7,
        )

        assert len(record.steps) == 1
        assert step.agent_name == "claude"
        assert step.confidence == 0.7

    def test_record_step_string_action(self):
        """String actions are converted to ActionType."""
        record = TrajectoryRecord(
            id="traj_1",
            task="task",
            started_at=datetime.now(),
        )

        step = record.record_step(
            agent="agent",
            action="propose",  # String instead of enum
            content="content",
        )

        assert step.action_type == ActionType.PROPOSE

    def test_record_step_unknown_action(self):
        """Unknown actions become OTHER."""
        record = TrajectoryRecord(
            id="traj_1",
            task="task",
            started_at=datetime.now(),
        )

        step = record.record_step(
            agent="agent",
            action="unknown_action",
            content="content",
        )

        assert step.action_type == ActionType.OTHER

    def test_set_outcome(self):
        """Outcome can be set."""
        record = TrajectoryRecord(
            id="traj_1",
            task="task",
            started_at=datetime.now(),
        )

        record.set_outcome(
            success=True,
            quality_score=0.85,
            feedback="Well done",
        )

        assert record.outcome is not None
        assert record.outcome.success is True
        assert record.completed_at is not None

    def test_is_complete(self):
        """is_complete reflects outcome presence."""
        record = TrajectoryRecord(
            id="traj_1",
            task="task",
            started_at=datetime.now(),
        )

        assert not record.is_complete

        record.set_outcome(True, 0.8)
        assert record.is_complete

    def test_duration_seconds(self):
        """Duration is calculated when complete."""
        record = TrajectoryRecord(
            id="traj_1",
            task="task",
            started_at=datetime.now(),
        )

        assert record.duration_seconds is None

        record.set_outcome(True, 0.8)
        assert record.duration_seconds is not None
        assert record.duration_seconds >= 0


class TestCalibrationInsight:
    """Tests for CalibrationInsight dataclass."""

    def test_creation(self):
        """Insight can be created."""
        insight = CalibrationInsight(
            agent_name="claude",
            action_type=ActionType.PROPOSE,
            original_confidence=0.9,
            retrospective_score=0.7,
            calibration_error=0.2,
            lesson="Agent was overconfident",
        )
        assert insight.calibration_error == 0.2


class TestAgentCalibration:
    """Tests for AgentCalibration dataclass."""

    def test_default_adjustment_factor(self):
        """Default adjustment factor is close to 1.0."""
        cal = AgentCalibration(agent_name="test")
        factor = cal.get_adjustment_factor(ActionType.PROPOSE)
        assert 0.9 <= factor <= 1.1

    def test_adjustment_with_overconfidence(self):
        """Overconfidence reduces adjustment factor."""
        cal = AgentCalibration(
            agent_name="test",
            overconfidence_bias=0.3,  # 30% overconfident
        )
        factor = cal.get_adjustment_factor(ActionType.PROPOSE)
        assert factor < 1.0

    def test_action_type_factor(self):
        """Action type factors are applied."""
        cal = AgentCalibration(agent_name="test")
        cal.action_type_factors[ActionType.CRITIQUE] = 1.2

        factor = cal.get_adjustment_factor(ActionType.CRITIQUE)
        assert factor > cal.get_adjustment_factor(ActionType.PROPOSE)


class TestSDPOConfig:
    """Tests for SDPOConfig dataclass."""

    def test_default_values(self):
        """Defaults are sensible."""
        config = SDPOConfig()
        assert config.buffer_size == 100
        assert config.learning_rate == 0.1
        assert config.min_trajectories_for_update == 5

    def test_custom_values(self):
        """Custom values are set."""
        config = SDPOConfig(
            buffer_size=50,
            learning_rate=0.2,
        )
        assert config.buffer_size == 50
        assert config.learning_rate == 0.2


class TestExperienceBuffer:
    """Tests for ExperienceBuffer class."""

    def test_add_and_get_recent(self):
        """Buffer stores and retrieves trajectories."""
        buffer = ExperienceBuffer(max_size=10)

        for i in range(5):
            record = TrajectoryRecord(
                id=f"traj_{i}",
                task=f"task {i}",
                started_at=datetime.now(),
            )
            record.set_outcome(True, 0.8)
            buffer.add(record)

        recent = buffer.get_recent(3)
        assert len(recent) == 3

    def test_max_size_respected(self):
        """Buffer respects max size."""
        buffer = ExperienceBuffer(max_size=3)

        for i in range(5):
            record = TrajectoryRecord(
                id=f"traj_{i}",
                task=f"task {i}",
                started_at=datetime.now(),
            )
            record.set_outcome(True, 0.8)
            buffer.add(record)

        assert len(buffer) == 3

    def test_incomplete_trajectory_not_added(self):
        """Incomplete trajectories are not added."""
        buffer = ExperienceBuffer()
        record = TrajectoryRecord(
            id="incomplete",
            task="task",
            started_at=datetime.now(),
        )
        # No outcome set

        buffer.add(record)
        assert len(buffer) == 0

    def test_get_by_task_pattern(self):
        """Can filter by task pattern."""
        buffer = ExperienceBuffer()

        for task in ["API design", "Database setup", "API testing"]:
            record = TrajectoryRecord(
                id=f"traj_{task}",
                task=task,
                started_at=datetime.now(),
            )
            record.set_outcome(True, 0.8)
            buffer.add(record)

        api_trajectories = buffer.get_by_task_pattern("API")
        assert len(api_trajectories) == 2

    def test_get_by_agent(self):
        """Can filter by agent."""
        buffer = ExperienceBuffer()

        record = TrajectoryRecord(
            id="traj_1",
            task="task",
            started_at=datetime.now(),
        )
        record.record_step("claude", ActionType.PROPOSE, "content")
        record.set_outcome(True, 0.8)
        buffer.add(record)

        claude_trajectories = buffer.get_by_agent("claude")
        assert len(claude_trajectories) == 1

        other_trajectories = buffer.get_by_agent("other_agent")
        assert len(other_trajectories) == 0


class TestDefaultRetrospectiveEvaluator:
    """Tests for DefaultRetrospectiveEvaluator class."""

    @pytest.mark.asyncio
    async def test_evaluate_step(self):
        """Evaluator produces score and explanation."""
        evaluator = DefaultRetrospectiveEvaluator()

        trajectory = TrajectoryRecord(
            id="traj_1",
            task="Design API",
            started_at=datetime.now(),
        )
        step = trajectory.record_step(
            agent="claude",
            action=ActionType.PROPOSE,
            content="We should use REST because it's simpler and more widely understood.",
            confidence=0.8,
        )
        trajectory.set_outcome(True, 0.9)

        score, explanation = await evaluator.evaluate_step(step, trajectory.outcome, trajectory)

        assert 0.0 <= score <= 1.0
        assert len(explanation) > 0
        assert "claude" in explanation

    @pytest.mark.asyncio
    async def test_outcome_quality_affects_score(self):
        """Better outcomes lead to higher contribution scores."""
        evaluator = DefaultRetrospectiveEvaluator()

        # Create two similar trajectories with different outcomes
        good_trajectory = TrajectoryRecord(
            id="good",
            task="task",
            started_at=datetime.now(),
        )
        step_good = good_trajectory.record_step("agent", ActionType.PROPOSE, "content")
        good_trajectory.set_outcome(True, 0.95)

        bad_trajectory = TrajectoryRecord(
            id="bad",
            task="task",
            started_at=datetime.now(),
        )
        step_bad = bad_trajectory.record_step("agent", ActionType.PROPOSE, "content")
        bad_trajectory.set_outcome(False, 0.2)

        good_score, _ = await evaluator.evaluate_step(
            step_good, good_trajectory.outcome, good_trajectory
        )
        bad_score, _ = await evaluator.evaluate_step(
            step_bad, bad_trajectory.outcome, bad_trajectory
        )

        assert good_score > bad_score


class TestSDPOLearner:
    """Tests for SDPOLearner class."""

    def test_start_trajectory(self):
        """Learner can start tracking a trajectory."""
        learner = SDPOLearner()
        trajectory = learner.start_trajectory("Design caching system")

        assert trajectory.task == "Design caching system"
        assert trajectory.id in learner._active_trajectories

    def test_complete_trajectory(self):
        """Learner can complete a trajectory."""
        learner = SDPOLearner()
        trajectory = learner.start_trajectory("task")
        trajectory.record_step("agent", ActionType.PROPOSE, "content")

        completed = learner.complete_trajectory(
            trajectory.id,
            success=True,
            quality_score=0.85,
            feedback="Good work",
        )

        assert completed.is_complete
        assert completed.outcome.quality_score == 0.85
        assert trajectory.id not in learner._active_trajectories
        assert len(learner.buffer) == 1

    def test_complete_nonexistent_trajectory(self):
        """Completing nonexistent trajectory returns None."""
        learner = SDPOLearner()
        result = learner.complete_trajectory("fake_id", True, 0.8)
        assert result is None

    @pytest.mark.asyncio
    async def test_evaluate_trajectory(self):
        """Learner can evaluate a complete trajectory."""
        learner = SDPOLearner()
        trajectory = learner.start_trajectory("API design")
        trajectory.record_step("claude", ActionType.PROPOSE, "Use REST", confidence=0.9)
        trajectory.record_step("gpt4", ActionType.CRITIQUE, "Consider GraphQL", confidence=0.7)
        trajectory.set_outcome(True, 0.85)

        insights = await learner.evaluate_trajectory(trajectory)

        assert len(insights) == 2
        for insight in insights:
            assert insight.agent_name in ["claude", "gpt4"]
            assert 0.0 <= insight.retrospective_score <= 1.0

    @pytest.mark.asyncio
    async def test_evaluate_incomplete_trajectory(self):
        """Evaluating incomplete trajectory returns empty list."""
        learner = SDPOLearner()
        trajectory = TrajectoryRecord(
            id="incomplete",
            task="task",
            started_at=datetime.now(),
        )

        insights = await learner.evaluate_trajectory(trajectory)
        assert insights == []

    def test_update_calibration(self):
        """Calibration is updated from insights."""
        learner = SDPOLearner()

        insights = [
            CalibrationInsight(
                agent_name="claude",
                action_type=ActionType.PROPOSE,
                original_confidence=0.9,
                retrospective_score=0.7,
                calibration_error=0.2,
                lesson="lesson",
            ),
            CalibrationInsight(
                agent_name="claude",
                action_type=ActionType.PROPOSE,
                original_confidence=0.8,
                retrospective_score=0.6,
                calibration_error=0.2,
                lesson="lesson",
            ),
        ]

        learner.update_calibration(insights)

        assert "claude" in learner.calibrations
        cal = learner.calibrations["claude"]
        assert cal.total_actions == 2
        assert cal.overconfidence_bias > 0  # Was overconfident

    def test_calibrate_confidence(self):
        """Confidence can be calibrated."""
        learner = SDPOLearner()

        # No calibration data yet
        raw = learner.calibrate_confidence("unknown", ActionType.PROPOSE, 0.8)
        assert raw == 0.8  # Unchanged

        # Add calibration showing overconfidence
        learner.calibrations["claude"] = AgentCalibration(
            agent_name="claude",
            overconfidence_bias=0.2,
        )

        calibrated = learner.calibrate_confidence("claude", ActionType.PROPOSE, 0.8)
        assert calibrated < 0.8  # Reduced due to overconfidence

    def test_get_agent_summary(self):
        """Agent summary is returned correctly."""
        learner = SDPOLearner()

        # No data
        summary = learner.get_agent_summary("unknown")
        assert summary["status"] == "no_data"

        # With data
        learner.calibrations["claude"] = AgentCalibration(
            agent_name="claude",
            total_actions=10,
            mean_confidence=0.8,
            mean_quality=0.7,
            overconfidence_bias=0.1,
        )

        summary = learner.get_agent_summary("claude")
        assert summary["total_actions"] == 10
        assert summary["is_overconfident"] is False  # 0.1 < threshold of 0.1

    @pytest.mark.asyncio
    async def test_batch_update_insufficient(self):
        """Batch update requires minimum trajectories."""
        learner = SDPOLearner(config=SDPOConfig(min_trajectories_for_update=5))

        # Add only 2 trajectories
        for i in range(2):
            traj = learner.start_trajectory(f"task {i}")
            traj.record_step("agent", ActionType.PROPOSE, "content")
            learner.complete_trajectory(traj.id, True, 0.8)

        count = await learner.batch_update()
        assert count == 0  # Not enough trajectories

    @pytest.mark.asyncio
    async def test_batch_update_sufficient(self):
        """Batch update processes trajectories when threshold met."""
        learner = SDPOLearner(config=SDPOConfig(min_trajectories_for_update=2))

        # Add 3 trajectories
        for i in range(3):
            traj = learner.start_trajectory(f"task {i}")
            traj.record_step("agent", ActionType.PROPOSE, f"content {i}", confidence=0.8)
            learner.complete_trajectory(traj.id, True, 0.7)

        count = await learner.batch_update()
        assert count == 2  # Processed min_trajectories_for_update

    def test_save_and_load(self):
        """Learner state can be saved and loaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "learner"

            # Create and save
            learner1 = SDPOLearner()
            learner1.calibrations["claude"] = AgentCalibration(
                agent_name="claude",
                total_actions=50,
                mean_confidence=0.75,
                mean_quality=0.8,
                calibration_error=0.1,
                overconfidence_bias=-0.05,
            )
            learner1.calibrations["claude"].action_type_factors[ActionType.PROPOSE] = 1.1
            learner1.save(path)

            # Load into new instance
            learner2 = SDPOLearner()
            learner2.load(path)

            assert "claude" in learner2.calibrations
            cal = learner2.calibrations["claude"]
            assert cal.total_actions == 50
            assert cal.mean_confidence == 0.75
            assert ActionType.PROPOSE in cal.action_type_factors


class TestEndToEndWorkflow:
    """End-to-end workflow tests."""

    @pytest.mark.asyncio
    async def test_full_learning_cycle(self):
        """Test complete learning cycle from start to calibration."""
        learner = SDPOLearner(config=SDPOConfig(learning_rate=0.5))

        # Simulate multiple debates
        for i in range(3):
            trajectory = learner.start_trajectory(f"Decision {i}")

            # Agent proposes with high confidence
            trajectory.record_step(
                agent="claude",
                action=ActionType.PROPOSE,
                content="Here is my proposal with detailed reasoning",
                confidence=0.9,  # High confidence
            )

            # Outcome is only moderately good
            learner.complete_trajectory(
                trajectory.id,
                success=True,
                quality_score=0.6,  # Lower than confidence
            )

        # Evaluate trajectories
        for traj in learner.buffer.get_recent(3):
            insights = await learner.evaluate_trajectory(traj)
            learner.update_calibration(insights)

        # Check that calibration detected overconfidence
        assert "claude" in learner.calibrations
        cal = learner.calibrations["claude"]
        assert cal.overconfidence_bias > 0  # Detected overconfidence

        # Future confidence should be adjusted down
        raw = 0.9
        calibrated = learner.calibrate_confidence("claude", ActionType.PROPOSE, raw)
        assert calibrated < raw
