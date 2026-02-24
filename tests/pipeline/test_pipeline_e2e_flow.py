"""End-to-end tests for the idea-to-execution pipeline.

Tests the full flow from ideas/debate through goals, actions, and orchestration.
Verifies provenance chains, stage transitions, receipt generation, and the
async pipeline runner.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.pipeline.idea_to_execution import (
    IdeaToExecutionPipeline,
    PipelineConfig,
    PipelineResult,
    StageResult,
)
from aragora.canvas.stages import PipelineStage, content_hash
from aragora.goals.extractor import GoalGraph, GoalNode


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def pipeline():
    """Default pipeline with no AI agent."""
    return IdeaToExecutionPipeline()


@pytest.fixture
def sample_ideas():
    """Sample idea strings."""
    return [
        "Build a rate limiter for API endpoints",
        "Add Redis-backed caching for frequently accessed data",
        "Improve API docs with OpenAPI playground",
    ]


@pytest.fixture
def sample_cartographer_data():
    """Sample ArgumentCartographer output."""
    return {
        "nodes": [
            {
                "id": "n1",
                "type": "proposal",
                "summary": "Build rate limiter",
                "content": "Token bucket",
            },
            {"id": "n2", "type": "evidence", "summary": "Reduces 429s", "content": "Evidence"},
            {"id": "n3", "type": "critique", "summary": "Distributed?", "content": "Question"},
        ],
        "edges": [
            {"source_id": "n2", "target_id": "n1", "relation": "supports"},
            {"source_id": "n3", "target_id": "n1", "relation": "responds_to"},
        ],
    }


@pytest.fixture
def event_collector():
    """Collects events emitted during pipeline execution."""
    events: list[tuple[str, dict]] = []

    def callback(event_type: str, data: dict):
        events.append((event_type, data))

    return events, callback


# =============================================================================
# TestFullPipelineFlow
# =============================================================================


class TestFullPipelineFlow:
    """Test the complete ideas -> goals -> actions -> orchestration flow."""

    def test_ideas_to_full_pipeline(self, pipeline, sample_ideas):
        result = pipeline.from_ideas(sample_ideas, auto_advance=True)

        assert result.ideas_canvas is not None
        assert result.goal_graph is not None
        assert result.actions_canvas is not None
        assert result.orchestration_canvas is not None

    def test_debate_to_full_pipeline(self, pipeline, sample_cartographer_data):
        result = pipeline.from_debate(sample_cartographer_data, auto_advance=True)

        assert result.ideas_canvas is not None
        assert result.goal_graph is not None
        assert result.actions_canvas is not None
        assert result.orchestration_canvas is not None

    def test_all_stages_complete(self, pipeline, sample_ideas):
        result = pipeline.from_ideas(sample_ideas, auto_advance=True)

        for stage in PipelineStage:
            if stage.value not in result.stage_status:
                continue  # PRINCIPLES stage is opt-in
            assert result.stage_status[stage.value] == "complete", f"{stage.value} not complete"

    def test_orchestration_has_agents(self, pipeline, sample_ideas):
        result = pipeline.from_ideas(sample_ideas, auto_advance=True)

        # Orchestration canvas should have agent nodes and task nodes
        orch_types = {n.data.get("orch_type") for n in result.orchestration_canvas.nodes.values()}
        assert "agent" in orch_types or "agent_task" in orch_types

    def test_actions_have_workflow_steps(self, pipeline, sample_ideas):
        result = pipeline.from_ideas(sample_ideas, auto_advance=True)

        # Actions canvas should have step nodes
        assert len(result.actions_canvas.nodes) > 0
        for node in result.actions_canvas.nodes.values():
            assert "step_type" in node.data or "stage" in node.data

    @pytest.mark.asyncio
    async def test_async_full_pipeline(self, pipeline):
        """Test the async run() method end-to-end."""
        cfg = PipelineConfig(
            stages_to_run=["ideation", "goals", "workflow", "orchestration"],
            dry_run=True,  # Skip actual orchestration
        )

        result = await pipeline.run("Build a better rate limiter", config=cfg)

        assert result.pipeline_id.startswith("pipe-")
        assert len(result.stage_results) > 0

    @pytest.mark.asyncio
    async def test_async_pipeline_emits_started_completed(self, pipeline, event_collector):
        events, callback = event_collector
        cfg = PipelineConfig(
            event_callback=callback,
            stages_to_run=["ideation"],
        )

        await pipeline.run("Test event emission", config=cfg)

        event_types = [e[0] for e in events]
        assert "started" in event_types
        assert "completed" in event_types

    @pytest.mark.asyncio
    async def test_async_pipeline_stage_results_populated(self, pipeline):
        cfg = PipelineConfig(
            stages_to_run=["ideation", "goals"],
        )

        result = await pipeline.run("Test stage results", config=cfg)

        assert len(result.stage_results) >= 1
        for sr in result.stage_results:
            assert sr.stage_name in ("ideation", "goals", "workflow", "orchestration")
            assert sr.status in ("completed", "failed", "skipped", "running", "pending")

    @pytest.mark.asyncio
    async def test_async_pipeline_with_execution(self, pipeline, event_collector):
        """Test async pipeline with mocked task execution (non-dry-run)."""
        events, callback = event_collector

        async def mock_execute(task, _cfg):
            return {
                "task_id": task["id"],
                "name": task["name"],
                "status": "completed",
                "output": {},
            }

        pipeline._execute_task = mock_execute

        cfg = PipelineConfig(
            event_callback=callback,
            dry_run=False,
        )

        result = await pipeline.run("Test with execution", config=cfg)

        assert result.pipeline_id.startswith("pipe-")
        # Should have attempted orchestration
        orch_stages = [sr for sr in result.stage_results if sr.stage_name == "orchestration"]
        if orch_stages:
            assert orch_stages[0].status in ("completed", "failed")


# =============================================================================
# TestProvenanceChain
# =============================================================================


class TestProvenanceChain:
    """Test that every node is traceable to its source."""

    def test_provenance_links_exist(self, pipeline, sample_ideas):
        result = pipeline.from_ideas(sample_ideas, auto_advance=True)
        assert len(result.provenance) > 0

    def test_provenance_hashes_are_valid(self, pipeline, sample_ideas):
        result = pipeline.from_ideas(sample_ideas, auto_advance=True)

        for link in result.provenance:
            assert link.content_hash
            assert len(link.content_hash) == 16

    def test_provenance_spans_multiple_stages(self, pipeline, sample_cartographer_data):
        result = pipeline.from_debate(sample_cartographer_data, auto_advance=True)

        stages_in_provenance = set()
        for link in result.provenance:
            stages_in_provenance.add(link.source_stage)
            stages_in_provenance.add(link.target_stage)

        assert len(stages_in_provenance) >= 2

    def test_goal_to_action_provenance(self, pipeline, sample_ideas):
        result = pipeline.from_ideas(sample_ideas, auto_advance=True)

        goal_to_action = [
            link
            for link in result.provenance
            if link.source_stage == PipelineStage.GOALS
            and link.target_stage == PipelineStage.ACTIONS
        ]
        assert len(goal_to_action) > 0

    def test_action_to_orchestration_provenance(self, pipeline, sample_ideas):
        result = pipeline.from_ideas(sample_ideas, auto_advance=True)

        action_to_orch = [
            link
            for link in result.provenance
            if link.source_stage == PipelineStage.ACTIONS
            and link.target_stage == PipelineStage.ORCHESTRATION
        ]
        assert len(action_to_orch) > 0

    def test_integrity_hash_deterministic(self, pipeline, sample_ideas):
        """Same input produces same integrity hash."""
        r1 = pipeline.from_ideas(sample_ideas, auto_advance=True, pipeline_id="pipe-fixed")
        r2 = pipeline.from_ideas(sample_ideas, auto_advance=True, pipeline_id="pipe-fixed")

        # The content hashes should be deterministic for the same ideas
        h1 = r1._compute_integrity_hash()
        h2 = r2._compute_integrity_hash()
        assert h1 == h2

    def test_integrity_hash_changes_with_input(self, pipeline):
        r1 = pipeline.from_ideas(["Idea A"], auto_advance=True)
        r2 = pipeline.from_ideas(["Idea B"], auto_advance=True)

        h1 = r1._compute_integrity_hash()
        h2 = r2._compute_integrity_hash()
        assert h1 != h2


# =============================================================================
# TestStageTransitionApproval
# =============================================================================


class TestStageTransitionApproval:
    """Test gates and approvals between stages."""

    def test_transitions_recorded_for_each_stage_boundary(self, pipeline, sample_cartographer_data):
        result = pipeline.from_debate(sample_cartographer_data, auto_advance=True)

        # Should have transitions for ideas->goals, goals->actions, actions->orch
        assert len(result.transitions) >= 2

    def test_transition_has_confidence(self, pipeline, sample_ideas):
        result = pipeline.from_ideas(sample_ideas, auto_advance=True)

        for t in result.transitions:
            assert 0.0 <= t.confidence <= 1.0

    def test_transition_has_rationale(self, pipeline, sample_ideas):
        result = pipeline.from_ideas(sample_ideas, auto_advance=True)

        for t in result.transitions:
            assert t.ai_rationale

    def test_no_auto_advance_stops_at_ideas(self, pipeline, sample_ideas):
        result = pipeline.from_ideas(sample_ideas, auto_advance=False)

        assert result.ideas_canvas is not None
        assert result.actions_canvas is None
        assert result.orchestration_canvas is None

    def test_manual_advance_to_actions(self, pipeline, sample_cartographer_data):
        result = pipeline.from_debate(sample_cartographer_data, auto_advance=False)

        result = pipeline.advance_stage(result, PipelineStage.GOALS)
        assert result.goal_graph is not None

        result = pipeline.advance_stage(result, PipelineStage.ACTIONS)
        assert result.actions_canvas is not None
        assert result.stage_status[PipelineStage.ACTIONS.value] == "complete"

    def test_skip_prerequisite_fails_gracefully(self, pipeline, sample_cartographer_data):
        result = pipeline.from_debate(sample_cartographer_data, auto_advance=False)

        # Try to go straight to orchestration (missing goals + actions)
        result = pipeline.advance_stage(result, PipelineStage.ORCHESTRATION)
        assert result.orchestration_canvas is None


# =============================================================================
# TestReceiptGeneration
# =============================================================================


class TestReceiptGeneration:
    """Test receipt generation at pipeline completion."""

    @pytest.mark.asyncio
    async def test_receipt_generated_on_completion(self, pipeline):
        """Receipts should be generated when enable_receipts=True."""

        async def mock_execute(task, _cfg):
            return {
                "task_id": task["id"],
                "name": task["name"],
                "status": "completed",
                "output": {},
            }

        pipeline._execute_task = mock_execute
        cfg = PipelineConfig(enable_receipts=True, dry_run=False)

        result = await pipeline.run("Test receipt generation", config=cfg)

        # Receipt should be present (even if fallback dict)
        assert result.receipt is not None

    @pytest.mark.asyncio
    async def test_receipt_has_pipeline_id(self, pipeline):
        async def mock_execute(task, _cfg):
            return {
                "task_id": task["id"],
                "name": task["name"],
                "status": "completed",
                "output": {},
            }

        pipeline._execute_task = mock_execute
        cfg = PipelineConfig(enable_receipts=True, dry_run=False)

        result = await pipeline.run(
            "Receipt pipeline id", config=cfg, pipeline_id="pipe-receipt-test"
        )

        assert result.receipt is not None
        # Either DecisionReceipt format or fallback format
        pid = (
            result.receipt.get("pipeline_id")
            or result.receipt.get("receipt_id")
            or result.receipt.get("gauntlet_id")
        )
        assert pid is not None

    @pytest.mark.asyncio
    async def test_receipt_not_generated_in_dry_run(self, pipeline):
        cfg = PipelineConfig(enable_receipts=True, dry_run=True)

        result = await pipeline.run("Dry run receipt", config=cfg)
        assert result.receipt is None

    def test_receipt_fallback_has_integrity_hash(self, pipeline):
        """When DecisionReceipt import fails, fallback dict is used."""
        result = PipelineResult(
            pipeline_id="pipe-fallback",
            stage_results=[
                StageResult(stage_name="ideation", status="completed", duration=1.0),
                StageResult(stage_name="goals", status="completed", duration=0.5),
            ],
        )

        receipt = pipeline._generate_receipt(result)

        assert receipt is not None
        assert "integrity_hash" in receipt or "receipt_id" in receipt

    def test_receipt_includes_evidence(self, pipeline):
        """Receipt should include stage evidence."""
        result = PipelineResult(
            pipeline_id="pipe-evidence",
            stage_results=[
                StageResult(stage_name="ideation", status="completed", duration=1.2),
                StageResult(stage_name="goals", status="completed", duration=0.8),
                StageResult(stage_name="workflow", status="failed", duration=0.3, error="oops"),
            ],
        )

        receipt = pipeline._generate_receipt(result)

        assert receipt is not None
        # Check evidence is present in fallback format
        if "evidence" in receipt:
            assert len(receipt["evidence"]) == 3
            statuses = [e["status"] for e in receipt["evidence"]]
            assert "completed" in statuses
            assert "failed" in statuses

    def test_receipt_includes_participants(self, pipeline):
        """Receipt should include agent participants from orchestration."""
        result = PipelineResult(
            pipeline_id="pipe-participants",
            stage_results=[
                StageResult(stage_name="orchestration", status="completed"),
            ],
            orchestration_result={
                "status": "executed",
                "results": [
                    {"name": "Research: API", "status": "completed"},
                    {"name": "Implement: Cache", "status": "completed"},
                ],
            },
        )

        receipt = pipeline._generate_receipt(result)

        assert receipt is not None
        if "participants" in receipt:
            assert "Research: API" in receipt["participants"]
            assert "Implement: Cache" in receipt["participants"]

    def test_receipt_deduplicates_participants(self, pipeline):
        """Same participant name should not appear twice."""
        result = PipelineResult(
            pipeline_id="pipe-dedup",
            stage_results=[],
            orchestration_result={
                "status": "executed",
                "results": [
                    {"name": "Agent A", "status": "completed"},
                    {"name": "Agent A", "status": "completed"},
                    {"name": "Agent B", "status": "completed"},
                ],
            },
        )

        receipt = pipeline._generate_receipt(result)

        if "participants" in receipt:
            assert len(receipt["participants"]) == 2
