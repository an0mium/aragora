"""Tests for the pipeline step verifier module."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from aragora.pipeline.step_verifier import (
    PipelineStepVerifier,
    PipelineVerificationResult,
    PipelineVerificationStep,
)


class TestPipelineVerificationResult:
    """Tests for PipelineVerificationResult dataclass."""

    def test_defaults(self):
        result = PipelineVerificationResult(
            pipeline_id="test-123",
            stage="ideas_to_goals",
        )
        assert result.total_steps == 0
        assert result.verified_steps == 0
        assert result.flagged_steps == 0
        assert result.overall_score == 0.0
        assert result.step_results == []

    def test_to_dict(self):
        result = PipelineVerificationResult(
            pipeline_id="test-123",
            stage="goals_to_actions",
            total_steps=5,
            verified_steps=4,
            flagged_steps=1,
            overall_score=0.8,
        )
        d = result.to_dict()
        assert d["pipeline_id"] == "test-123"
        assert d["stage"] == "goals_to_actions"
        assert d["total_steps"] == 5
        assert d["verified_steps"] == 4
        assert d["overall_score"] == 0.8


class TestPipelineVerificationStep:
    """Tests for PipelineVerificationStep dataclass."""

    def test_defaults(self):
        step = PipelineVerificationStep(
            step_id="step-1",
            stage="ideas_to_goals",
            content="Goal: improve UX",
        )
        assert step.agent_id == ""
        assert step.dependencies == []

    def test_with_dependencies(self):
        step = PipelineVerificationStep(
            step_id="step-1",
            stage="goals_to_actions",
            content="Action: research patterns",
            dependencies=["Goal: redesign nav"],
        )
        assert len(step.dependencies) == 1


class TestStepExtraction:
    """Tests for extracting verifiable steps from pipeline data."""

    @pytest.fixture
    def verifier(self):
        return PipelineStepVerifier()

    def test_extract_goal_steps(self, verifier):
        steps = verifier._extract_goal_steps(
            ideas=["improve UX", "add caching"],
            goals=[
                {"title": "Redesign navigation", "priority": "high"},
                {"title": "Add Redis cache", "priority": "medium"},
            ],
        )
        assert len(steps) == 2
        assert steps[0].step_id == "goal-0"
        assert steps[0].stage == "ideas_to_goals"
        assert "Redesign navigation" in steps[0].content
        assert len(steps[0].dependencies) == 2

    def test_extract_action_steps(self, verifier):
        steps = verifier._extract_action_steps(
            goals=[{"title": "Redesign nav"}],
            actions=[
                {"name": "Research UX patterns", "step_type": "research"},
                {"name": "Implement new nav", "step_type": "task"},
            ],
        )
        assert len(steps) == 2
        assert steps[0].step_id == "action-0"
        assert "Research UX patterns" in steps[0].content

    def test_extract_assignment_steps(self, verifier):
        steps = verifier._extract_assignment_steps(
            actions=[{"name": "Research UX"}],
            assignments=[
                {"name": "Research UX", "agent_id": "agent-researcher"},
                {"name": "Implement nav", "agent_id": "agent-implementer"},
            ],
        )
        assert len(steps) == 2
        assert steps[0].agent_id == "agent-researcher"
        assert "agent-researcher" in steps[0].content

    def test_extract_empty_goals(self, verifier):
        steps = verifier._extract_goal_steps(ideas=[], goals=[])
        assert steps == []


class TestStructuralVerification:
    """Tests for fallback structural verification."""

    @pytest.fixture
    def verifier(self):
        return PipelineStepVerifier()

    def test_correct_with_content_and_deps(self, verifier):
        steps = [
            PipelineVerificationStep(
                step_id="step-1",
                stage="test",
                content="Some content",
                dependencies=["dep1"],
            )
        ]
        results = verifier._structural_verify(steps, "test")
        assert len(results) == 1
        assert results[0]["verdict"] == "correct"
        assert results[0]["confidence"] == 0.6

    def test_uncertain_without_deps(self, verifier):
        steps = [
            PipelineVerificationStep(
                step_id="step-1",
                stage="test",
                content="Some content",
                dependencies=[],
            )
        ]
        results = verifier._structural_verify(steps, "test")
        assert results[0]["verdict"] == "uncertain"

    def test_incorrect_without_content(self, verifier):
        steps = [
            PipelineVerificationStep(
                step_id="step-1",
                stage="test",
                content="",
                dependencies=["dep1"],
            )
        ]
        results = verifier._structural_verify(steps, "test")
        assert results[0]["verdict"] == "incorrect"


class TestGoalExtractionVerification:
    """Tests for verify_goal_extraction."""

    @pytest.mark.asyncio
    async def test_verify_with_structural_fallback(self):
        verifier = PipelineStepVerifier()

        # This will use structural fallback since ThinkPRM needs API
        result = await verifier.verify_goal_extraction(
            ideas=["improve UX", "add tests"],
            goals=[
                {"title": "Redesign navigation", "priority": "high"},
                {"title": "Add unit tests", "priority": "medium"},
            ],
            pipeline_id="test-pipe",
        )

        assert result.pipeline_id == "test-pipe"
        assert result.stage == "ideas_to_goals"
        assert result.total_steps == 2
        assert result.duration_ms > 0
        # With deps from ideas, structural verify gives "correct"
        assert result.overall_score > 0

    @pytest.mark.asyncio
    async def test_verify_empty_goals(self):
        verifier = PipelineStepVerifier()

        result = await verifier.verify_goal_extraction(
            ideas=["improve UX"],
            goals=[],
            pipeline_id="test-pipe",
        )

        assert result.total_steps == 0
        assert result.overall_score == 1.0  # No steps = perfect


class TestActionDecompositionVerification:
    """Tests for verify_action_decomposition."""

    @pytest.mark.asyncio
    async def test_verify_action_decomposition(self):
        verifier = PipelineStepVerifier()

        result = await verifier.verify_action_decomposition(
            goals=[
                {"title": "Redesign navigation", "priority": "high"},
            ],
            actions=[
                {"name": "Research UX patterns", "step_type": "research"},
                {"name": "Implement new nav", "step_type": "task"},
            ],
            pipeline_id="test-pipe",
        )

        assert result.stage == "goals_to_actions"
        assert result.total_steps == 2
        assert result.overall_score > 0


class TestAgentAssignmentVerification:
    """Tests for verify_agent_assignments."""

    @pytest.mark.asyncio
    async def test_verify_assignments(self):
        verifier = PipelineStepVerifier()

        result = await verifier.verify_agent_assignments(
            actions=[{"name": "Research UX"}],
            assignments=[
                {"name": "Research UX", "agent_id": "agent-researcher"},
            ],
            pipeline_id="test-pipe",
        )

        assert result.stage == "actions_to_orchestration"
        assert result.total_steps == 1


class TestCalibrationIntegration:
    """Tests for calibration-weighted score adjustment."""

    @pytest.mark.asyncio
    async def test_calibration_unavailable_returns_raw(self):
        verifier = PipelineStepVerifier(calibration_weight=0.3)
        raw_score = 0.8

        adjusted = await verifier._apply_calibration(raw_score, [])
        assert adjusted == raw_score  # No agent IDs → no adjustment

    @pytest.mark.asyncio
    async def test_calibration_adjusts_score(self):
        verifier = PipelineStepVerifier(calibration_weight=0.3)

        mock_tracker = MagicMock()
        mock_tracker.get_brier_score.return_value = 0.1  # Good calibration

        # CalibrationTracker is imported inline, patch at source module
        import aragora.agents.calibration as cal_mod

        original = getattr(cal_mod, "CalibrationTracker", None)
        cal_mod.CalibrationTracker = MagicMock(return_value=mock_tracker)
        try:
            adjusted = await verifier._apply_calibration(
                0.8,
                [{"agent_id": "agent-1"}],
            )
            # 0.8 * 0.7 + 0.9 * 0.3 = 0.56 + 0.27 = 0.83
            assert 0.82 < adjusted < 0.84
        finally:
            if original is not None:
                cal_mod.CalibrationTracker = original


class TestPipelineConfigIntegration:
    """Tests for PipelineConfig integration."""

    def test_config_has_step_verification_flag(self):
        from aragora.pipeline.idea_to_execution import PipelineConfig

        config = PipelineConfig()
        assert config.enable_step_verification is False

    def test_config_enables_step_verification(self):
        from aragora.pipeline.idea_to_execution import PipelineConfig

        config = PipelineConfig(enable_step_verification=True)
        assert config.enable_step_verification is True
