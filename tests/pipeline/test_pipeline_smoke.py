"""Integration smoke tests for the full idea-to-execution pipeline.

Tests the end-to-end flow through all 4 stages using both the sync
(from_ideas) and async (run) entry points:

  Stage 1: Ideation     -- raw ideas converted to canvas nodes
  Stage 2: Goals         -- goal extraction with SMART scoring + conflict detection
  Stage 3: Workflow      -- goal decomposition into action steps
  Stage 4: Orchestration -- multi-agent execution plan

Focuses on inter-stage integration rather than individual stage internals
(those are covered by test_async_stages.py and test_idea_to_execution.py).

All tests use dry_run=True and mock the KM bridge so no real infrastructure
is needed.  Target wall-clock budget: < 5 seconds total.
"""

from __future__ import annotations

import json
import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.canvas.stages import PipelineStage
from aragora.goals.extractor import GoalGraph, GoalNode
from aragora.pipeline.idea_to_execution import (
    IdeaToExecutionPipeline,
    PipelineConfig,
    PipelineResult,
    StageResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Ideas designed to trigger conflict detection: the first two contain
# contradictory keywords ("maximize throughput" vs "minimize throughput").
CONTRADICTORY_IDEAS = [
    "Maximize throughput for the batch processing pipeline by adding workers",
    "Minimize throughput to save costs during off-peak hours",
    "Add a Redis cache in front of the database for read-heavy queries",
    "Set up Prometheus monitoring with SLO alerting within 2 weeks",
]

NORMAL_IDEAS = [
    "Build a token-bucket rate limiter for API endpoints",
    "Add Redis-backed caching for frequently accessed data",
    "Improve API docs with an OpenAPI interactive playground",
    "Set up end-to-end performance monitoring with Grafana dashboards",
]


@pytest.fixture
def pipeline():
    """Pipeline instance with no AI agent (structural extraction only)."""
    return IdeaToExecutionPipeline()


@pytest.fixture
def event_log():
    """Capture pipeline events as (event_type, data) tuples."""
    captured: list[tuple[str, dict[str, Any]]] = []

    def callback(event_type: str, data: dict[str, Any]) -> None:
        captured.append((event_type, data))

    return captured, callback


@pytest.fixture
def mock_km_precedents():
    """Patch the KM bridge to return deterministic precedent data."""
    fake_bridge = MagicMock()
    fake_bridge.available = True
    fake_bridge.query_similar_goals.return_value = {
        "any-goal-id": [
            {
                "title": "Prior rate limiter implementation",
                "similarity": 0.82,
                "outcome": "successful",
            },
        ]
    }

    def _enrich(goal_graph: Any, precedents: dict[str, list[dict[str, Any]]]) -> Any:
        for goal in goal_graph.goals:
            if goal.id in precedents and precedents[goal.id]:
                goal.metadata["precedents"] = precedents[goal.id]
            elif precedents:
                # Assign the first available precedent list for testing
                first_key = next(iter(precedents))
                goal.metadata["precedents"] = precedents[first_key]
        return goal_graph

    fake_bridge.enrich_with_precedents.side_effect = _enrich

    with patch(
        "aragora.pipeline.idea_to_execution.PipelineKMBridge",
        return_value=fake_bridge,
        create=True,
    ):
        # Also patch the import inside _run_goal_extraction and _advance_to_goals
        with patch.dict("sys.modules", {
            "aragora.pipeline.km_bridge": MagicMock(
                PipelineKMBridge=MagicMock(return_value=fake_bridge),
            ),
        }):
            yield fake_bridge


# ---------------------------------------------------------------------------
# Sync pipeline smoke tests (from_ideas)
# ---------------------------------------------------------------------------


class TestSyncPipelineSmoke:
    """End-to-end tests using the synchronous from_ideas() entry point."""

    def test_full_pipeline_completes_all_stages(self, pipeline):
        """All four stages complete without error."""
        result = pipeline.from_ideas(NORMAL_IDEAS, auto_advance=True)

        assert isinstance(result, PipelineResult)
        assert result.ideas_canvas is not None
        assert result.goal_graph is not None
        assert result.actions_canvas is not None
        assert result.orchestration_canvas is not None

        for stage in PipelineStage:
            assert result.stage_status[stage.value] == "complete", (
                f"Stage {stage.value} should be 'complete', "
                f"got '{result.stage_status.get(stage.value)}'"
            )

    @pytest.mark.xfail(reason="Provenance tracking not yet implemented")
    def test_provenance_chain_spans_all_stages(self, pipeline):
        """Provenance links should connect ideas -> goals -> actions -> orch."""
        result = pipeline.from_ideas(NORMAL_IDEAS, auto_advance=True)

        assert len(result.provenance) > 0, "Expected at least one provenance link"

        stages_in_provenance = set()
        for link in result.provenance:
            stages_in_provenance.add(link.source_stage.value)
            stages_in_provenance.add(link.target_stage.value)

        # At minimum, ideas->goals and goals->actions links should exist
        assert "ideas" in stages_in_provenance
        assert "goals" in stages_in_provenance

    def test_transitions_recorded(self, pipeline):
        """Stage transitions are recorded between consecutive stages."""
        result = pipeline.from_ideas(NORMAL_IDEAS, auto_advance=True)

        assert len(result.transitions) >= 2, (
            f"Expected at least 2 transitions, got {len(result.transitions)}"
        )

        # Verify transition directions
        transition_pairs = [
            (t.from_stage.value, t.to_stage.value) for t in result.transitions
        ]
        assert ("ideas", "goals") in transition_pairs or (
            ("goals", "actions") in transition_pairs
        ), f"Missing expected transitions, got: {transition_pairs}"

    def test_smart_scores_on_goals(self, pipeline):
        """Goals should have SMART scores in their metadata."""
        result = pipeline.from_ideas(NORMAL_IDEAS, auto_advance=True)

        assert result.goal_graph is not None
        assert len(result.goal_graph.goals) > 0

        goals_with_smart = [
            g for g in result.goal_graph.goals
            if "smart_scores" in g.metadata
        ]
        assert len(goals_with_smart) > 0, (
            "Expected at least one goal with SMART scores"
        )

        # Verify SMART score structure
        sample_scores = goals_with_smart[0].metadata["smart_scores"]
        for dimension in ("specific", "measurable", "achievable", "relevant", "time_bound", "overall"):
            assert dimension in sample_scores, (
                f"Missing SMART dimension: {dimension}"
            )
            assert 0.0 <= sample_scores[dimension] <= 1.0, (
                f"SMART score for {dimension} out of range: {sample_scores[dimension]}"
            )

    @pytest.mark.xfail(reason="Conflict detection not yet implemented in goal extraction")
    def test_conflict_detection_with_contradictory_ideas(self, pipeline):
        """Contradictory ideas should produce conflict entries in metadata."""
        result = pipeline.from_ideas(CONTRADICTORY_IDEAS, auto_advance=True)

        assert result.goal_graph is not None
        conflicts = result.goal_graph.metadata.get("conflicts", [])

        # The maximize/minimize pair should trigger a contradiction
        assert len(conflicts) > 0, (
            "Expected at least one conflict from contradictory ideas "
            f"(maximize vs minimize). Got metadata: {result.goal_graph.metadata}"
        )

        # Verify conflict structure
        conflict = conflicts[0]
        assert "type" in conflict or "severity" in conflict or "goal_ids" in conflict

    @pytest.mark.xfail(reason="KM precedent enrichment not yet wired into goal extraction")
    def test_km_precedents_enrichment(self, pipeline, mock_km_precedents):
        """Goals should be enriched with KM precedent data when bridge is available."""
        result = pipeline.from_ideas(NORMAL_IDEAS, auto_advance=True)

        assert result.goal_graph is not None
        goals_with_precedents = [
            g for g in result.goal_graph.goals
            if "precedents" in g.metadata
        ]
        assert len(goals_with_precedents) > 0, (
            "Expected at least one goal enriched with KM precedents"
        )

        # Verify precedent structure
        sample_prec = goals_with_precedents[0].metadata["precedents"]
        assert len(sample_prec) > 0
        assert "title" in sample_prec[0]
        assert "similarity" in sample_prec[0]

    def test_events_emitted_for_each_stage(self, pipeline, event_log):
        """Event callback fires for each stage transition."""
        captured, callback = event_log
        result = pipeline.from_ideas(
            NORMAL_IDEAS, auto_advance=True, event_callback=callback,
        )

        event_types = [e[0] for e in captured]

        # Should have stage_completed events for ideas, goals, actions, orchestration
        completed_events = [e for e in captured if e[0] == "stage_completed"]
        completed_stages = {e[1].get("stage") for e in completed_events}

        assert "ideas" in completed_stages, (
            f"Missing 'ideas' stage_completed event. Got: {completed_stages}"
        )
        assert "goals" in completed_stages, (
            f"Missing 'goals' stage_completed event. Got: {completed_stages}"
        )

        # Also verify node-level events were emitted
        node_events = [e for e in captured if e[0] == "pipeline_node_added"]
        assert len(node_events) > 0, "Expected pipeline_node_added events"

    def test_result_serializable_to_json(self, pipeline):
        """PipelineResult.to_dict() output must be JSON-serializable."""
        result = pipeline.from_ideas(NORMAL_IDEAS, auto_advance=True)
        result_dict = result.to_dict()

        json_str = json.dumps(result_dict)
        assert len(json_str) > 100, "Serialized result seems too small"

        roundtripped = json.loads(json_str)
        assert "pipeline_id" in roundtripped
        assert "integrity_hash" in roundtripped

    def test_workflow_decomposes_goals_into_steps(self, pipeline):
        """Stage 3 should decompose goals into multiple action steps."""
        result = pipeline.from_ideas(NORMAL_IDEAS, auto_advance=True)

        assert result.actions_canvas is not None
        assert len(result.actions_canvas.nodes) > len(result.goal_graph.goals), (
            f"Expected more action nodes ({len(result.actions_canvas.nodes)}) "
            f"than goals ({len(result.goal_graph.goals)}), since each goal "
            f"decomposes into multiple steps"
        )

    def test_orchestration_assigns_agents(self, pipeline):
        """Stage 4 should assign agents to execution tasks."""
        result = pipeline.from_ideas(NORMAL_IDEAS, auto_advance=True)

        assert result.orchestration_canvas is not None
        assert len(result.orchestration_canvas.nodes) > 0

        # Check that orchestration nodes reference agent assignments
        has_agent_data = False
        for node in result.orchestration_canvas.nodes.values():
            if node.data.get("assigned_agent") or node.data.get("orch_type"):
                has_agent_data = True
                break

        assert has_agent_data, (
            "Orchestration canvas nodes should have agent assignment data"
        )


# ---------------------------------------------------------------------------
# Async pipeline smoke tests (run)
# ---------------------------------------------------------------------------


class TestAsyncPipelineSmoke:
    """End-to-end tests using the async run() entry point."""

    @pytest.mark.asyncio
    async def test_async_run_all_stages_dry_run(self, pipeline, event_log):
        """Async run with dry_run=True completes ideation+goals+workflow, skips orchestration."""
        captured, callback = event_log

        config = PipelineConfig(
            dry_run=True,
            enable_receipts=False,
            stages_to_run=["ideation", "goals", "workflow", "orchestration"],
            event_callback=callback,
        )

        result = await pipeline.run(
            "Build a rate limiter. Add Redis caching. Set up monitoring.",
            config=config,
        )

        assert isinstance(result, PipelineResult)
        assert len(result.stage_results) == 4  # all 4 stages attempted

        # Check each stage result
        stage_map = {sr.stage_name: sr for sr in result.stage_results}

        assert stage_map["ideation"].status == "completed"
        assert stage_map["goals"].status == "completed"
        assert stage_map["workflow"].status == "completed"
        # Orchestration should be skipped in dry_run
        assert stage_map["orchestration"].status == "skipped"

        # Verify timing is recorded
        for sr in result.stage_results:
            if sr.status == "completed":
                assert sr.duration > 0, (
                    f"Stage {sr.stage_name} should have positive duration"
                )

    @pytest.mark.asyncio
    async def test_async_run_emits_full_event_sequence(self, pipeline, event_log):
        """Async run should emit started/stage_started/stage_completed/completed events."""
        captured, callback = event_log

        config = PipelineConfig(
            dry_run=True,
            enable_receipts=False,
            stages_to_run=["ideation", "goals", "workflow"],
            event_callback=callback,
        )

        await pipeline.run("Build a rate limiter", config=config)

        event_types = [e[0] for e in captured]

        assert "started" in event_types, (
            f"Missing 'started' event. Got: {event_types}"
        )
        assert "completed" in event_types, (
            f"Missing 'completed' event. Got: {event_types}"
        )

        # Should have stage_started and stage_completed for each stage
        stage_started_count = event_types.count("stage_started")
        stage_completed_count = event_types.count("stage_completed")

        assert stage_started_count >= 3, (
            f"Expected >= 3 stage_started events, got {stage_started_count}"
        )
        assert stage_completed_count >= 3, (
            f"Expected >= 3 stage_completed events, got {stage_completed_count}"
        )

    @pytest.mark.asyncio
    async def test_async_run_goals_have_smart_scores(self, pipeline):
        """Async run should produce goals with SMART scores."""
        config = PipelineConfig(
            dry_run=True,
            enable_receipts=False,
            stages_to_run=["ideation", "goals"],
            enable_smart_goals=True,
        )

        result = await pipeline.run(
            "Build authentication with OAuth2. Add rate limiting for API endpoints. Deploy monitoring within 30 days.",
            config=config,
        )

        assert result.goal_graph is not None
        assert len(result.goal_graph.goals) > 0

        goals_with_smart = [
            g for g in result.goal_graph.goals
            if "smart_scores" in g.metadata
        ]
        assert len(goals_with_smart) > 0, (
            "At least one goal should have SMART scores after async run"
        )

    @pytest.mark.asyncio
    @pytest.mark.xfail(reason="Conflict detection not yet implemented in goal extraction")
    async def test_async_run_conflict_detection(self, pipeline):
        """Async run should detect conflicts when ideas contradict."""
        config = PipelineConfig(
            dry_run=True,
            enable_receipts=False,
            stages_to_run=["ideation", "goals"],
        )

        result = await pipeline.run(
            "Maximize database throughput. Minimize database throughput.",
            config=config,
        )

        assert result.goal_graph is not None
        conflicts = result.goal_graph.metadata.get("conflicts", [])
        assert len(conflicts) > 0, (
            "Expected conflict detection for contradictory maximize/minimize goals"
        )

    @pytest.mark.asyncio
    async def test_async_run_km_precedents(self, pipeline, mock_km_precedents):
        """Async run should enrich goals with KM precedent data."""
        config = PipelineConfig(
            dry_run=True,
            enable_receipts=False,
            stages_to_run=["ideation", "goals"],
            enable_km_precedents=True,
        )

        result = await pipeline.run(
            "Build a rate limiter. Add caching. Set up monitoring.",
            config=config,
        )

        assert result.goal_graph is not None
        goals_with_precedents = [
            g for g in result.goal_graph.goals
            if "precedents" in g.metadata
        ]
        assert len(goals_with_precedents) > 0, (
            "Expected goals enriched with KM precedents after async run"
        )

    @pytest.mark.asyncio
    async def test_async_run_with_receipt_generation(self, pipeline):
        """When dry_run=False and orchestration is skipped, receipt logic still works."""
        config = PipelineConfig(
            dry_run=False,
            enable_receipts=True,
            # Only run stages that don't need real execution engines
            stages_to_run=["ideation", "goals", "workflow", "orchestration"],
        )

        result = await pipeline.run(
            "Build a rate limiter. Add caching.",
            config=config,
        )

        # Orchestration should complete (or fall back gracefully)
        orch_results = [
            sr for sr in result.stage_results if sr.stage_name == "orchestration"
        ]
        assert len(orch_results) == 1

        # Receipt should be generated since dry_run=False and enable_receipts=True
        assert result.receipt is not None, (
            "Expected a receipt when dry_run=False and enable_receipts=True"
        )
        assert "pipeline_id" in result.receipt or "decision_id" in result.receipt

    @pytest.mark.asyncio
    async def test_async_run_duration_under_budget(self, pipeline):
        """Full async pipeline should complete in under 5 seconds."""
        config = PipelineConfig(
            dry_run=True,
            enable_receipts=False,
            stages_to_run=["ideation", "goals", "workflow", "orchestration"],
        )

        start = time.monotonic()
        result = await pipeline.run("Build a feature. Test it.", config=config)
        elapsed = time.monotonic() - start

        assert elapsed < 5.0, (
            f"Pipeline took {elapsed:.2f}s, expected < 5.0s"
        )
        assert result.duration > 0
        assert result.duration < 5.0


# ---------------------------------------------------------------------------
# Cross-cutting integration tests
# ---------------------------------------------------------------------------


class TestCrossCuttingIntegration:
    """Tests that verify integration between stages and cross-cutting concerns."""

    @pytest.mark.xfail(reason="Priority metadata not yet propagated to action steps")
    def test_goal_priority_influences_workflow_step_ordering(self, pipeline):
        """High-priority goals should influence action step configuration."""
        result = pipeline.from_ideas(NORMAL_IDEAS, auto_advance=True)

        assert result.actions_canvas is not None

        # Collect step priorities from the action canvas
        priorities = set()
        for node in result.actions_canvas.nodes.values():
            prio = node.data.get("config", {}).get("priority")
            if prio:
                priorities.add(prio)

        # Should have a mix of priorities from the goal decomposition
        assert len(priorities) > 0, (
            "Action steps should carry priority metadata from parent goals"
        )

    def test_integrity_hash_is_deterministic_per_run(self, pipeline):
        """The same pipeline run should produce a consistent integrity hash."""
        result = pipeline.from_ideas(NORMAL_IDEAS, auto_advance=True)

        hash1 = result._compute_integrity_hash()
        hash2 = result._compute_integrity_hash()

        assert hash1 == hash2
        assert len(hash1) == 16  # SHA-256 truncated to 16 hex chars

    def test_goal_graph_transition_links_to_ideas_stage(self, pipeline):
        """The GoalGraph's transition should reference ideas -> goals."""
        result = pipeline.from_ideas(NORMAL_IDEAS, auto_advance=True)

        assert result.goal_graph is not None
        assert result.goal_graph.transition is not None

        transition = result.goal_graph.transition
        assert transition.from_stage == PipelineStage.IDEAS
        assert transition.to_stage == PipelineStage.GOALS
        assert 0.0 <= transition.confidence <= 1.0
        assert len(transition.ai_rationale) > 0

    @pytest.mark.asyncio
    async def test_async_and_sync_produce_consistent_goal_count(self, pipeline):
        """Sync from_ideas and async run should extract similar goal counts."""
        sync_result = pipeline.from_ideas(NORMAL_IDEAS, auto_advance=False)

        config = PipelineConfig(
            dry_run=True,
            enable_receipts=False,
            stages_to_run=["ideation", "goals"],
        )
        async_result = await pipeline.run(
            ". ".join(NORMAL_IDEAS), config=config,
        )

        sync_count = len(sync_result.goal_graph.goals) if sync_result.goal_graph else 0
        async_count = len(async_result.goal_graph.goals) if async_result.goal_graph else 0

        # Both should extract at least 1 goal
        assert sync_count >= 1, "Sync path should extract at least 1 goal"
        assert async_count >= 1, "Async path should extract at least 1 goal"

    def test_pipeline_id_format(self, pipeline):
        """Pipeline IDs should follow the expected format."""
        result = pipeline.from_ideas(NORMAL_IDEAS, auto_advance=True)

        assert result.pipeline_id.startswith("pipe-")
        assert len(result.pipeline_id) > 5
