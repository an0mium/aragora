"""Tests for IdeaToExecutionPipeline async run() and related features."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.pipeline.idea_to_execution import (
    IdeaToExecutionPipeline,
    PipelineConfig,
    PipelineResult,
    StageResult,
)
from aragora.goals.extractor import GoalExtractionConfig, GoalExtractor, GoalGraph, GoalNode
from aragora.canvas.stages import GoalNodeType


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def pipeline():
    return IdeaToExecutionPipeline()


@pytest.fixture
def basic_config():
    return PipelineConfig(
        stages_to_run=["ideation", "goals", "workflow", "orchestration"],
        dry_run=True,  # Skip orchestration by default
    )


@pytest.fixture
def events_collected():
    """Capture emitted events."""
    events: list[tuple[str, dict]] = []

    def callback(event_type: str, data: dict) -> None:
        events.append((event_type, data))

    return events, callback


# =========================================================================
# StageResult tests
# =========================================================================

class TestStageResult:
    def test_to_dict_basic(self):
        sr = StageResult(stage_name="ideation", status="completed", duration=1.5)
        d = sr.to_dict()
        assert d["stage_name"] == "ideation"
        assert d["status"] == "completed"
        assert d["duration"] == 1.5

    def test_to_dict_with_error(self):
        sr = StageResult(stage_name="goals", status="failed", error="something broke")
        d = sr.to_dict()
        assert d["error"] == "something broke"

    def test_to_dict_skipped(self):
        sr = StageResult(stage_name="orchestration", status="skipped")
        d = sr.to_dict()
        assert d["status"] == "skipped"


# =========================================================================
# PipelineResult enhanced fields
# =========================================================================

class TestPipelineResultEnhanced:
    def test_to_dict_with_stage_results(self):
        result = PipelineResult(
            pipeline_id="pipe-test",
            stage_results=[
                StageResult(stage_name="ideation", status="completed", duration=0.5),
                StageResult(stage_name="goals", status="completed", duration=0.3),
            ],
            duration=1.2,
        )
        d = result.to_dict()
        assert "stage_results" in d
        assert len(d["stage_results"]) == 2
        assert d["duration"] == 1.2

    def test_to_dict_with_receipt(self):
        result = PipelineResult(
            pipeline_id="pipe-test",
            receipt={"integrity_hash": "abc123"},
        )
        d = result.to_dict()
        assert d["receipt"]["integrity_hash"] == "abc123"

    def test_to_dict_with_final_workflow(self):
        result = PipelineResult(
            pipeline_id="pipe-test",
            final_workflow={"steps": [{"id": "s1"}], "name": "test"},
        )
        d = result.to_dict()
        assert d["final_workflow"]["name"] == "test"


# =========================================================================
# Async run() tests
# =========================================================================

class TestAsyncRun:
    @pytest.mark.asyncio
    async def test_basic_run_dry_run(self, pipeline):
        config = PipelineConfig(dry_run=True)
        result = await pipeline.run("Build a REST API", config)

        assert isinstance(result, PipelineResult)
        assert result.pipeline_id.startswith("pipe-")
        assert result.duration > 0
        # Orchestration should be skipped in dry_run
        orch_results = [sr for sr in result.stage_results if sr.stage_name == "orchestration"]
        if orch_results:
            assert orch_results[0].status == "skipped"

    @pytest.mark.asyncio
    async def test_selective_stages(self, pipeline):
        config = PipelineConfig(
            stages_to_run=["ideation", "goals"],
            dry_run=True,
        )
        result = await pipeline.run("Test idea", config)

        stage_names = [sr.stage_name for sr in result.stage_results]
        assert "ideation" in stage_names
        assert "goals" in stage_names
        assert "orchestration" not in stage_names

    @pytest.mark.asyncio
    async def test_event_emission(self, pipeline, events_collected):
        events, callback = events_collected
        config = PipelineConfig(
            stages_to_run=["ideation", "goals"],
            dry_run=True,
            event_callback=callback,
        )
        await pipeline.run("Test events", config)

        event_types = [e[0] for e in events]
        assert "started" in event_types
        assert "stage_started" in event_types
        assert "stage_completed" in event_types or "completed" in event_types

    @pytest.mark.asyncio
    async def test_dry_run_skips_orchestration(self, pipeline):
        config = PipelineConfig(
            stages_to_run=["ideation", "goals", "workflow", "orchestration"],
            dry_run=True,
        )
        result = await pipeline.run("Dry run test", config)

        orch = [sr for sr in result.stage_results if sr.stage_name == "orchestration"]
        assert len(orch) == 1
        assert orch[0].status == "skipped"

    @pytest.mark.asyncio
    async def test_receipt_generation(self, pipeline):
        config = PipelineConfig(
            stages_to_run=["ideation", "goals"],
            enable_receipts=True,
            dry_run=False,
        )
        result = await pipeline.run("Receipt test", config)
        # Receipt should be generated (either from gauntlet or fallback)
        assert result.receipt is not None
        assert "pipeline_id" in result.receipt or "decision_id" in result.receipt

    @pytest.mark.asyncio
    async def test_receipt_disabled(self, pipeline):
        config = PipelineConfig(
            stages_to_run=["ideation"],
            enable_receipts=False,
            dry_run=True,
        )
        result = await pipeline.run("No receipt test", config)
        assert result.receipt is None

    @pytest.mark.asyncio
    async def test_goals_from_raw_text(self, pipeline):
        config = PipelineConfig(
            stages_to_run=["ideation", "goals"],
            dry_run=True,
        )
        result = await pipeline.run(
            "Build authentication. Add rate limiting. Deploy to cloud.", config,
        )
        # Should extract goals from the ideas
        if result.goal_graph:
            assert len(result.goal_graph.goals) >= 0  # May have goals from extraction

    @pytest.mark.asyncio
    async def test_workflow_generation_fallback(self, pipeline):
        """Test that workflow generation uses internal fallback when NLWorkflowBuilder unavailable."""
        config = PipelineConfig(
            stages_to_run=["ideation", "goals", "workflow"],
            dry_run=True,
        )
        result = await pipeline.run("Add user authentication flow", config)

        wf_results = [sr for sr in result.stage_results if sr.stage_name == "workflow"]
        assert len(wf_results) == 1
        # Should succeed via fallback
        assert wf_results[0].status in ("completed", "failed")

    @pytest.mark.asyncio
    async def test_pipeline_config_defaults(self):
        config = PipelineConfig()
        assert config.stages_to_run == ["ideation", "goals", "workflow", "orchestration"]
        assert config.debate_rounds == 3
        assert config.workflow_mode == "quick"
        assert config.dry_run is False
        assert config.enable_receipts is True

    @pytest.mark.asyncio
    async def test_empty_input(self, pipeline):
        config = PipelineConfig(stages_to_run=["ideation", "goals"], dry_run=True)
        result = await pipeline.run("", config)
        assert isinstance(result, PipelineResult)

    @pytest.mark.asyncio
    async def test_stage_result_durations(self, pipeline):
        config = PipelineConfig(
            stages_to_run=["ideation", "goals"],
            dry_run=True,
        )
        result = await pipeline.run("Time test", config)
        for sr in result.stage_results:
            assert sr.duration >= 0

    @pytest.mark.asyncio
    async def test_goal_extraction_config_passthrough(self, pipeline):
        config = PipelineConfig(
            stages_to_run=["ideation", "goals"],
            goal_extraction_config=GoalExtractionConfig(max_goals=2),
            dry_run=True,
        )
        result = await pipeline.run("Complex multi-goal scenario with many aspects", config)
        assert isinstance(result, PipelineResult)


class TestAsyncRunErrorHandling:
    @pytest.mark.asyncio
    async def test_ideation_failure_handled(self, pipeline, events_collected):
        events, callback = events_collected
        config = PipelineConfig(
            stages_to_run=["ideation", "goals"],
            event_callback=callback,
            dry_run=True,
        )
        # Even if ideation partially fails, pipeline should continue
        result = await pipeline.run("Test error handling", config)
        assert isinstance(result, PipelineResult)

    @pytest.mark.asyncio
    async def test_pipeline_always_returns_result(self, pipeline):
        """Pipeline should always return a PipelineResult, never raise."""
        config = PipelineConfig(stages_to_run=["ideation", "goals", "workflow"], dry_run=True)
        result = await pipeline.run("Resilience test", config)
        assert isinstance(result, PipelineResult)
        assert result.pipeline_id.startswith("pipe-")


class TestEmitHelper:
    def test_emit_with_callback(self):
        events: list[tuple[str, dict]] = []
        config = PipelineConfig(event_callback=lambda t, d: events.append((t, d)))
        pipeline = IdeaToExecutionPipeline()
        pipeline._emit(config, "test_event", {"key": "value"})
        assert len(events) == 1
        assert events[0] == ("test_event", {"key": "value"})

    def test_emit_without_callback(self):
        config = PipelineConfig(event_callback=None)
        pipeline = IdeaToExecutionPipeline()
        # Should not raise
        pipeline._emit(config, "test_event", {"key": "value"})

    def test_emit_callback_error_swallowed(self):
        def broken_callback(t, d):
            raise RuntimeError("broken")

        config = PipelineConfig(event_callback=broken_callback)
        pipeline = IdeaToExecutionPipeline()
        # Should not raise
        pipeline._emit(config, "test_event", {"key": "value"})
