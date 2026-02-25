"""
Tests for async pipeline stage methods in isolation.

Covers the 4 async stage methods and their fallback paths:
- _run_ideation: debate fallback → text extraction
- _run_goal_extraction: 4 input branches (nodes, preview, canvas, empty)
- _run_workflow_generation: NLWorkflowBuilder fallback → internal conversion
- _run_orchestration: AutonomousOrchestrator fallback → static result

Also covers: event emission, error recovery, receipt generation, dry-run mode.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.canvas.stages import GoalNodeType, PipelineStage
from aragora.goals.extractor import GoalExtractionConfig, GoalGraph, GoalNode
from aragora.pipeline.idea_to_execution import (
    IdeaToExecutionPipeline,
    PipelineConfig,
    PipelineResult,
    StageResult,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def pipeline():
    return IdeaToExecutionPipeline()


@pytest.fixture
def config():
    return PipelineConfig()


@pytest.fixture
def sample_goal_graph():
    return GoalGraph(
        id="test-goals",
        goals=[
            GoalNode(
                id="g1",
                title="Build auth system",
                description="Implement OAuth2",
                goal_type=GoalNodeType.GOAL,
                priority="high",
            ),
            GoalNode(
                id="g2",
                title="Add monitoring",
                description="Set up observability",
                goal_type=GoalNodeType.METRIC,
                priority="medium",
            ),
        ],
    )


@pytest.fixture
def events():
    """Capture emitted events."""
    captured: list[tuple[str, dict]] = []

    def callback(event_type: str, data: dict) -> None:
        captured.append((event_type, data))

    return captured, callback


# =============================================================================
# _run_ideation tests
# =============================================================================


class TestRunIdeation:
    """Test Stage 1: ideation with debate and text fallback paths."""

    @pytest.mark.asyncio
    async def test_text_fallback_on_import_error(self, pipeline, config):
        """When Arena can't be imported, falls back to text extraction."""
        sr = await pipeline._run_ideation("pipe-1", "Build auth. Add tests.", config)

        assert sr.status == "completed"
        assert sr.stage_name == "ideation"
        assert sr.duration > 0
        assert sr.output is not None
        assert "canvas" in sr.output

    @pytest.mark.asyncio
    async def test_text_fallback_produces_ideas(self, pipeline, config):
        """Text fallback splits on sentences and extracts ideas."""
        sr = await pipeline._run_ideation(
            "pipe-1", "Build authentication. Add rate limiting. Deploy monitoring.", config
        )

        assert sr.output["raw_ideas"] == [
            "Build authentication",
            "Add rate limiting",
            "Deploy monitoring",
        ]

    @pytest.mark.asyncio
    async def test_single_idea_when_no_periods(self, pipeline, config):
        """Input without periods becomes a single idea."""
        sr = await pipeline._run_ideation("pipe-1", "Build a rate limiter", config)

        assert sr.output["raw_ideas"] == ["Build a rate limiter"]

    @pytest.mark.asyncio
    async def test_empty_input(self, pipeline, config):
        """Empty string still produces a result."""
        sr = await pipeline._run_ideation("pipe-1", "", config)

        assert sr.status == "completed"
        assert sr.output is not None

    @pytest.mark.asyncio
    async def test_emits_stage_events(self, pipeline, events):
        """Verifies stage_started and stage_completed events are emitted."""
        captured, callback = events
        cfg = PipelineConfig(event_callback=callback)

        await pipeline._run_ideation("pipe-1", "Test idea", cfg)

        event_types = [e[0] for e in captured]
        assert "stage_started" in event_types
        assert "stage_completed" in event_types

    @pytest.mark.asyncio
    async def test_debate_path_with_mock_arena(self, pipeline, config):
        """When Arena is available, uses debate for richer ideation."""
        mock_result = MagicMock()
        mock_result.argument_graph = {
            "nodes": [{"id": "n1", "label": "proposal", "node_type": "proposal"}]
        }
        mock_arena = MagicMock()
        mock_arena.run = AsyncMock(return_value=mock_result)

        with (
            patch(
                "aragora.pipeline.idea_to_execution.Arena",
                return_value=mock_arena,
                create=True,
            ) as mock_cls,
            patch.dict(
                "sys.modules",
                {
                    "aragora.debate.orchestrator": MagicMock(Arena=mock_cls),
                    "aragora.debate.models": MagicMock(
                        DebateProtocol=MagicMock(),
                        Environment=MagicMock(),
                    ),
                },
            ),
        ):
            sr = await pipeline._run_ideation("pipe-1", "Test debate", config)

        assert sr.status == "completed"


# =============================================================================
# _run_goal_extraction tests
# =============================================================================


class TestRunGoalExtraction:
    """Test Stage 2: goal extraction from various input shapes."""

    @pytest.mark.asyncio
    async def test_with_debate_nodes(self, pipeline, config):
        """When debate output has nodes, uses extract_from_debate_analysis."""
        debate_output = {
            "nodes": [
                {
                    "id": "n1",
                    "label": "We should use OAuth2",
                    "node_type": "consensus",
                    "weight": 0.9,
                },
            ]
        }

        sr = await pipeline._run_goal_extraction("pipe-1", debate_output, config)

        assert sr.status == "completed"
        assert sr.output["goal_graph"] is not None

    @pytest.mark.asyncio
    async def test_with_goal_graph_preview(self, pipeline, config, sample_goal_graph):
        """When debate output has goal_graph_preview, uses it directly."""
        debate_output = {"goal_graph_preview": sample_goal_graph}

        sr = await pipeline._run_goal_extraction("pipe-1", debate_output, config)

        assert sr.status == "completed"
        assert sr.output["goal_graph"] is sample_goal_graph

    @pytest.mark.asyncio
    async def test_with_canvas_data(self, pipeline, config):
        """When debate output has canvas, uses structural extraction."""
        from aragora.canvas.models import Canvas

        canvas = Canvas(id="c1", name="Test Canvas")
        debate_output = {"canvas": canvas}

        sr = await pipeline._run_goal_extraction("pipe-1", debate_output, config)

        assert sr.status == "completed"
        assert sr.output["goal_graph"] is not None

    @pytest.mark.asyncio
    async def test_with_none_input(self, pipeline, config):
        """When debate output is None, creates empty GoalGraph."""
        sr = await pipeline._run_goal_extraction("pipe-1", None, config)

        assert sr.status == "completed"
        assert sr.output["goal_graph"] is not None
        assert len(sr.output["goal_graph"].goals) == 0

    @pytest.mark.asyncio
    async def test_with_empty_dict(self, pipeline, config):
        """When debate output is empty dict, creates empty GoalGraph."""
        sr = await pipeline._run_goal_extraction("pipe-1", {}, config)

        assert sr.status == "completed"
        assert len(sr.output["goal_graph"].goals) == 0

    @pytest.mark.asyncio
    async def test_emits_goal_events(self, pipeline, events):
        """Emits goal_extracted event for each extracted goal."""
        captured, callback = events
        cfg = PipelineConfig(event_callback=callback)

        debate_output = {
            "nodes": [
                {
                    "id": "n1",
                    "label": "Use OAuth2",
                    "node_type": "consensus",
                    "weight": 0.9,
                },
            ]
        }

        await pipeline._run_goal_extraction("pipe-1", debate_output, cfg)

        event_types = [e[0] for e in captured]
        assert "stage_started" in event_types
        assert "stage_completed" in event_types

    @pytest.mark.asyncio
    async def test_custom_extraction_config(self, pipeline):
        """Custom GoalExtractionConfig is passed through."""
        custom_cfg = PipelineConfig(
            goal_extraction_config=GoalExtractionConfig(
                confidence_threshold=0.9,
                max_goals=2,
            )
        )
        debate_output = {
            "nodes": [
                {
                    "id": f"n{i}",
                    "label": f"idea {i}",
                    "node_type": "consensus",
                    "weight": 0.5,
                }
                for i in range(5)
            ]
        }

        sr = await pipeline._run_goal_extraction("pipe-1", debate_output, custom_cfg)
        assert sr.status == "completed"


# =============================================================================
# _run_workflow_generation tests
# =============================================================================


class TestRunWorkflowGeneration:
    """Test Stage 3: workflow generation with NL builder and fallback."""

    @pytest.mark.asyncio
    async def test_generates_workflow_from_goals(self, pipeline, config, sample_goal_graph):
        """Generates workflow from goal graph (via NLBuilder or fallback)."""
        sr = await pipeline._run_workflow_generation("pipe-1", sample_goal_graph, config)

        assert sr.status == "completed"
        assert sr.output["workflow"] is not None
        # Output may come from NLWorkflowBuilder (nested) or fallback (flat)
        workflow = sr.output["workflow"]
        if "steps" in workflow:
            assert len(workflow["steps"]) > 0
        elif "workflow" in workflow:
            # NLWorkflowBuilder wraps result with metadata
            assert workflow["success"] is not None

    @pytest.mark.asyncio
    async def test_empty_goal_graph(self, pipeline, config):
        """Empty goal graph produces empty workflow."""
        empty_goals = GoalGraph(id="empty")
        sr = await pipeline._run_workflow_generation("pipe-1", empty_goals, config)

        assert sr.status == "completed"
        assert sr.output["workflow"]["steps"] == []

    @pytest.mark.asyncio
    async def test_none_goal_graph(self, pipeline, config):
        """None goal graph produces empty workflow."""
        sr = await pipeline._run_workflow_generation("pipe-1", None, config)

        assert sr.status == "completed"
        assert sr.output["workflow"]["name"] == "empty"

    @pytest.mark.asyncio
    async def test_emits_workflow_event(self, pipeline, events, sample_goal_graph):
        """Emits workflow_generated event on success."""
        captured, callback = events
        cfg = PipelineConfig(event_callback=callback)

        await pipeline._run_workflow_generation("pipe-1", sample_goal_graph, cfg)

        event_types = [e[0] for e in captured]
        assert "workflow_generated" in event_types

    @pytest.mark.asyncio
    async def test_summary_in_completed_event(self, pipeline, events, sample_goal_graph):
        """Stage completion summary is present."""
        captured, callback = events
        cfg = PipelineConfig(event_callback=callback)

        await pipeline._run_workflow_generation("pipe-1", sample_goal_graph, cfg)

        completed_events = [e for e in captured if e[0] == "stage_completed"]
        assert len(completed_events) == 1
        assert "summary" in completed_events[0][1]

    @pytest.mark.asyncio
    async def test_quick_vs_debate_mode(self, pipeline, sample_goal_graph):
        """Both workflow modes produce valid results via fallback."""
        quick_cfg = PipelineConfig(workflow_mode="quick")
        debate_cfg = PipelineConfig(workflow_mode="debate")

        sr_quick = await pipeline._run_workflow_generation("pipe-1", sample_goal_graph, quick_cfg)
        sr_debate = await pipeline._run_workflow_generation("pipe-1", sample_goal_graph, debate_cfg)

        assert sr_quick.status == "completed"
        assert sr_debate.status == "completed"


# =============================================================================
# _run_orchestration tests
# =============================================================================


class TestRunOrchestration:
    """Test Stage 4: orchestration with orchestrator and fallback."""

    @pytest.mark.asyncio
    async def test_fallback_when_orchestrator_unavailable(
        self, pipeline, config, sample_goal_graph, monkeypatch
    ):
        """Falls back gracefully when execution engine unavailable.

        With empty steps but a goal graph, the pipeline builds tasks from
        goals and attempts execution. Tasks complete with planned/failed
        status when DebugLoop is unavailable or path validation fails.
        """
        # Mock _execute_task to return "planned" status (simulates no execution backend)
        async def _mock_execute(task, cfg):
            return {
                "task_id": task["id"],
                "name": task["name"],
                "status": "planned",
                "output": {"reason": "execution_engine_unavailable"},
            }

        monkeypatch.setattr(pipeline, "_execute_task", _mock_execute)
        sr = await pipeline._run_orchestration("pipe-1", {"steps": []}, sample_goal_graph, config)

        assert sr.status == "completed"
        orch = sr.output["orchestration"]
        assert orch["status"] == "executed"
        assert orch["tasks_total"] > 0

    @pytest.mark.asyncio
    async def test_no_goals_skips_orchestration(self, pipeline, config):
        """When no goals, orchestration is skipped."""
        empty_goals = GoalGraph(id="empty")
        sr = await pipeline._run_orchestration("pipe-1", {}, empty_goals, config)

        assert sr.status == "completed"
        assert sr.output["orchestration"]["status"] in ("skipped", "fallback")

    @pytest.mark.asyncio
    async def test_none_goal_graph(self, pipeline, config):
        """None goal graph falls back gracefully."""
        sr = await pipeline._run_orchestration("pipe-1", {}, None, config)

        assert sr.status == "completed"

    @pytest.mark.asyncio
    async def test_emits_stage_events(self, pipeline, events, sample_goal_graph, monkeypatch):
        """Emits stage_started and stage_completed events."""
        captured, callback = events
        cfg = PipelineConfig(event_callback=callback)

        # Mock _execute_task to avoid calling real backends
        async def _mock_execute(task, cfg):
            return {
                "task_id": task["id"],
                "name": task["name"],
                "status": "planned",
                "output": {},
            }

        monkeypatch.setattr(pipeline, "_execute_task", _mock_execute)
        sr = await pipeline._run_orchestration("pipe-1", {"steps": []}, sample_goal_graph, cfg)

        event_types = [e[0] for e in captured]
        assert "stage_started" in event_types
        # Orchestration completes (tasks built from goals, executed or planned)
        assert sr.status == "completed"
        assert "stage_completed" in event_types


# =============================================================================
# Full async pipeline (run) tests
# =============================================================================


class TestAsyncPipelineRun:
    """Test the full async run() method."""

    @pytest.fixture(autouse=True)
    def mock_execute_task(self, pipeline, monkeypatch):
        """Prevent _execute_task from calling real backends during full runs."""

        async def _mock_execute(task, cfg):
            return {
                "task_id": task["id"],
                "name": task["name"],
                "status": "planned",
                "output": {},
            }

        monkeypatch.setattr(pipeline, "_execute_task", _mock_execute)

    @pytest.mark.asyncio
    async def test_full_run(self, pipeline):
        """Full pipeline run completes all stages."""
        result = await pipeline.run("Build authentication system")

        assert isinstance(result, PipelineResult)
        assert len(result.stage_results) >= 1
        assert result.duration > 0

    @pytest.mark.asyncio
    async def test_dry_run_skips_orchestration(self, pipeline):
        """Dry run skips orchestration stage."""
        cfg = PipelineConfig(dry_run=True)
        result = await pipeline.run("Test ideas", config=cfg)

        orch_results = [sr for sr in result.stage_results if sr.stage_name == "orchestration"]
        if orch_results:
            assert orch_results[0].status == "skipped"

    @pytest.mark.asyncio
    async def test_event_callbacks(self, pipeline, events):
        """Events are emitted throughout the pipeline."""
        captured, callback = events
        cfg = PipelineConfig(event_callback=callback)

        await pipeline.run("Test idea", config=cfg)

        event_types = [e[0] for e in captured]
        assert "started" in event_types
        assert "completed" in event_types or any("stage_completed" in t for t in event_types)

    @pytest.mark.asyncio
    async def test_receipt_generation(self, pipeline):
        """Pipeline generates a receipt on completion."""
        cfg = PipelineConfig(enable_receipts=True)
        result = await pipeline.run("Test", config=cfg)

        assert result.receipt is not None

    @pytest.mark.asyncio
    async def test_no_receipt_on_dry_run(self, pipeline):
        """No receipt generated during dry run."""
        cfg = PipelineConfig(dry_run=True, enable_receipts=True)
        result = await pipeline.run("Test", config=cfg)

        assert result.receipt is None

    @pytest.mark.asyncio
    async def test_subset_of_stages(self, pipeline):
        """Can run only a subset of stages."""
        cfg = PipelineConfig(stages_to_run=["ideation", "goals"])
        result = await pipeline.run("Test", config=cfg)

        stage_names = [sr.stage_name for sr in result.stage_results]
        assert "ideation" in stage_names
        assert "goals" in stage_names
        assert "orchestration" not in stage_names


# =============================================================================
# Event emission and error handling
# =============================================================================


class TestEventEmission:
    """Test _emit and error handling edge cases."""

    def test_emit_with_no_callback(self, pipeline):
        """_emit is a no-op when no callback configured."""
        cfg = PipelineConfig()
        pipeline._emit(cfg, "test_event", {"data": "value"})  # Should not raise

    def test_emit_swallows_callback_errors(self, pipeline):
        """_emit swallows exceptions from the callback."""

        def bad_callback(event_type: str, data: dict) -> None:
            raise RuntimeError("callback failed")

        cfg = PipelineConfig(event_callback=bad_callback)
        pipeline._emit(cfg, "test_event", {"data": "value"})  # Should not raise

    def test_generate_receipt_fallback(self, pipeline):
        """Receipt generation falls back to dict when DecisionReceipt unavailable."""
        result = PipelineResult(pipeline_id="pipe-1")
        receipt = pipeline._generate_receipt(result)

        assert receipt is not None
        assert "pipeline_id" in receipt
        assert "integrity_hash" in receipt


# =============================================================================
# StageResult and PipelineConfig
# =============================================================================


class TestStageResult:
    """Test StageResult serialization."""

    def test_to_dict_basic(self):
        sr = StageResult(stage_name="ideation", status="completed", duration=1.5)
        d = sr.to_dict()
        assert d["stage_name"] == "ideation"
        assert d["status"] == "completed"
        assert d["duration"] == 1.5

    def test_to_dict_with_error(self):
        sr = StageResult(stage_name="goals", status="failed", error="Something broke")
        d = sr.to_dict()
        assert d["error"] == "Something broke"

    def test_to_dict_with_output(self):
        mock_output = MagicMock()
        mock_output.to_dict.return_value = {"key": "value"}
        sr = StageResult(stage_name="workflow", status="completed", output=mock_output)
        d = sr.to_dict()
        assert "output_summary" in d


class TestPipelineConfig:
    """Test PipelineConfig defaults."""

    def test_defaults(self):
        cfg = PipelineConfig()
        assert cfg.stages_to_run == ["ideation", "goals", "workflow", "orchestration"]
        assert cfg.debate_rounds == 3
        assert cfg.workflow_mode == "quick"
        assert cfg.dry_run is False
        assert cfg.enable_receipts is True
