"""Tests for the pipeline intake module."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, AsyncMock

from aragora.pipeline.intake import (
    AutonomyLevel,
    IntakeRequest,
    IntakeResult,
    PipelineIntake,
)


class TestAutonomyLevel:
    """Tests for AutonomyLevel enum."""

    def test_values(self):
        assert AutonomyLevel.PROPOSE_AND_EXPLAIN == 1
        assert AutonomyLevel.PROPOSE_AND_APPROVE == 2
        assert AutonomyLevel.EXECUTE_AND_REPORT == 3
        assert AutonomyLevel.FULLY_AUTONOMOUS == 4
        assert AutonomyLevel.CONTINUOUS == 5

    def test_ordering(self):
        assert AutonomyLevel.PROPOSE_AND_EXPLAIN < AutonomyLevel.FULLY_AUTONOMOUS

    def test_from_int(self):
        level = AutonomyLevel(3)
        assert level == AutonomyLevel.EXECUTE_AND_REPORT


class TestIntakeRequest:
    """Tests for IntakeRequest dataclass."""

    def test_defaults(self):
        req = IntakeRequest(prompt="test prompt")
        assert req.prompt == "test prompt"
        assert req.autonomy_level == AutonomyLevel.PROPOSE_AND_APPROVE
        assert req.skip_interrogation is False
        assert req.max_interrogation_turns == 5
        assert req.workspace_id == "default"
        assert req.pipeline_id  # should be auto-generated UUID

    def test_custom_values(self):
        req = IntakeRequest(
            prompt="test",
            autonomy_level=AutonomyLevel.FULLY_AUTONOMOUS,
            skip_interrogation=True,
            pipeline_id="custom-id",
            user_id="user-1",
        )
        assert req.autonomy_level == AutonomyLevel.FULLY_AUTONOMOUS
        assert req.skip_interrogation is True
        assert req.pipeline_id == "custom-id"
        assert req.user_id == "user-1"


class TestIntakeResult:
    """Tests for IntakeResult dataclass."""

    def test_defaults(self):
        result = IntakeResult(pipeline_id="test-123")
        assert result.pipeline_id == "test-123"
        assert result.ideas == []
        assert result.themes == []
        assert result.ready_for_pipeline is False
        assert result.error is None

    def test_to_dict(self):
        result = IntakeResult(
            pipeline_id="test-123",
            ideas=["idea1", "idea2"],
            themes=["ux"],
            refined_goal="improve UX",
            acceptance_criteria=["tests pass"],
            ready_for_pipeline=True,
        )
        d = result.to_dict()
        assert d["pipeline_id"] == "test-123"
        assert d["ideas"] == ["idea1", "idea2"]
        assert d["themes"] == ["ux"]
        assert d["refined_goal"] == "improve UX"
        assert d["ready_for_pipeline"] is True


class TestPipelineIntake:
    """Tests for PipelineIntake class."""

    @pytest.fixture
    def intake(self):
        return PipelineIntake()

    @pytest.mark.asyncio
    async def test_process_parses_prompt(self, intake):
        """Test that process parses the prompt into ideas."""
        mock_enriched = MagicMock()
        mock_enriched.ideas = ["idea one", "idea two"]
        mock_enriched.detected_themes = ["security"]
        mock_enriched.urgency_signals = []

        mock_parser = MagicMock()
        mock_parser.parse_enriched.return_value = mock_enriched

        intake._parser = mock_parser

        request = IntakeRequest(
            prompt="improve security and add tests",
            skip_interrogation=True,
        )
        result = await intake.process(request)

        assert result.ideas == ["idea one", "idea two"]
        assert result.themes == ["security"]
        assert result.ready_for_pipeline is True
        mock_parser.parse_enriched.assert_called_once_with("improve security and add tests")

    @pytest.mark.asyncio
    async def test_process_fallback_on_parse_error(self, intake):
        """Test that process falls back to raw prompt when parser fails."""
        mock_parser = MagicMock()
        mock_parser.parse_enriched.side_effect = RuntimeError("parse failed")

        intake._parser = mock_parser

        request = IntakeRequest(
            prompt="just a simple prompt",
            skip_interrogation=True,
        )
        result = await intake.process(request)

        assert result.ideas == ["just a simple prompt"]
        assert result.ready_for_pipeline is True

    @pytest.mark.asyncio
    async def test_process_skips_interrogation_at_high_autonomy(self, intake):
        """Test that high autonomy levels skip interrogation."""
        mock_enriched = MagicMock()
        mock_enriched.ideas = ["idea"]
        mock_enriched.detected_themes = []
        mock_enriched.urgency_signals = []

        mock_parser = MagicMock()
        mock_parser.parse_enriched.return_value = mock_enriched
        intake._parser = mock_parser

        request = IntakeRequest(
            prompt="test prompt",
            autonomy_level=AutonomyLevel.FULLY_AUTONOMOUS,
        )
        result = await intake.process(request)

        # Should not attempt interrogation
        assert result.refined_goal == "test prompt"
        assert result.interrogation_summary == ""

    @pytest.mark.asyncio
    async def test_process_runs_interrogation_at_low_autonomy(self, intake):
        """Test that low autonomy levels attempt interrogation."""
        mock_enriched = MagicMock()
        mock_enriched.ideas = ["idea"]
        mock_enriched.detected_themes = []
        mock_enriched.urgency_signals = []

        mock_parser = MagicMock()
        mock_parser.parse_enriched.return_value = mock_enriched
        intake._parser = mock_parser

        mock_spec = MagicMock()
        mock_spec.refined_goal = "refined: improve UX"
        mock_spec.acceptance_criteria = ["users love it"]
        mock_spec.summary.return_value = "Summary of interrogation"

        mock_interrogator = AsyncMock()
        mock_interrogator.interrogate.return_value = mock_spec

        MockClass = MagicMock(return_value=mock_interrogator)

        # Patch at the source module where inline import resolves
        import aragora.pipeline.interrogator as interrog_mod

        original = interrog_mod.PipelineInterrogator
        interrog_mod.PipelineInterrogator = MockClass
        try:
            request = IntakeRequest(
                prompt="improve UX",
                autonomy_level=AutonomyLevel.PROPOSE_AND_EXPLAIN,
            )
            result = await intake.process(request)

            assert result.refined_goal == "refined: improve UX"
            assert result.acceptance_criteria == ["users love it"]
            assert result.interrogation_summary == "Summary of interrogation"
        finally:
            interrog_mod.PipelineInterrogator = original

    @pytest.mark.asyncio
    async def test_execute_calls_pipeline(self, intake):
        """Test that execute calls IdeaToExecutionPipeline.from_ideas."""
        mock_pipeline_result = MagicMock()
        mock_pipeline_result.stage_status = {"ideation": "complete"}

        mock_pipeline = MagicMock()
        mock_pipeline.from_ideas.return_value = mock_pipeline_result
        intake._pipeline = mock_pipeline

        intake_result = IntakeResult(
            pipeline_id="test-123",
            ideas=["idea1", "idea2"],
            ready_for_pipeline=True,
        )

        request = IntakeRequest(
            prompt="test",
            autonomy_level=AutonomyLevel.EXECUTE_AND_REPORT,
        )

        # execute() does an inline import of PipelineConfig — patch at source
        import aragora.pipeline.idea_to_execution as pipe_mod

        original_config = pipe_mod.PipelineConfig
        pipe_mod.PipelineConfig = MagicMock()
        try:
            result = await intake.execute(intake_result, request)
        finally:
            pipe_mod.PipelineConfig = original_config

        mock_pipeline.from_ideas.assert_called_once()
        call_kwargs = mock_pipeline.from_ideas.call_args
        assert call_kwargs.kwargs.get("ideas") == ["idea1", "idea2"] or call_kwargs[1].get(
            "ideas"
        ) == ["idea1", "idea2"]

    @pytest.mark.asyncio
    async def test_execute_sets_auto_advance_by_autonomy(self, intake):
        """Test that autonomy level controls auto_advance."""
        mock_pipeline = MagicMock()
        mock_pipeline.from_ideas.return_value = MagicMock(stage_status={})
        intake._pipeline = mock_pipeline

        intake_result = IntakeResult(
            pipeline_id="test-123",
            ideas=["idea"],
            ready_for_pipeline=True,
        )

        # Low autonomy → auto_advance=False
        request_low = IntakeRequest(
            prompt="test",
            autonomy_level=AutonomyLevel.PROPOSE_AND_APPROVE,
        )

        import aragora.pipeline.idea_to_execution as pipe_mod

        original_config = pipe_mod.PipelineConfig
        pipe_mod.PipelineConfig = MagicMock()
        try:
            await intake.execute(intake_result, request_low)
            call_args = mock_pipeline.from_ideas.call_args
            assert call_args.kwargs.get("auto_advance") is False

            mock_pipeline.from_ideas.reset_mock()

            # High autonomy → auto_advance=True
            request_high = IntakeRequest(
                prompt="test",
                autonomy_level=AutonomyLevel.EXECUTE_AND_REPORT,
            )
            await intake.execute(intake_result, request_high)
            call_args = mock_pipeline.from_ideas.call_args
            assert call_args.kwargs.get("auto_advance") is True
        finally:
            pipe_mod.PipelineConfig = original_config

    @pytest.mark.asyncio
    async def test_process_handles_interrogation_failure(self, intake):
        """Test graceful fallback when interrogation fails."""
        mock_enriched = MagicMock()
        mock_enriched.ideas = ["idea"]
        mock_enriched.detected_themes = []
        mock_enriched.urgency_signals = []

        mock_parser = MagicMock()
        mock_parser.parse_enriched.return_value = mock_enriched
        intake._parser = mock_parser

        MockClass = MagicMock(side_effect=RuntimeError("interrogator broken"))

        import aragora.pipeline.interrogator as interrog_mod

        original = interrog_mod.PipelineInterrogator
        interrog_mod.PipelineInterrogator = MockClass
        try:
            request = IntakeRequest(
                prompt="test prompt",
                autonomy_level=AutonomyLevel.PROPOSE_AND_EXPLAIN,
            )
            result = await intake.process(request)

            # Should fall back gracefully
            assert result.refined_goal == "test prompt"
            assert result.ready_for_pipeline is True
        finally:
            interrog_mod.PipelineInterrogator = original
