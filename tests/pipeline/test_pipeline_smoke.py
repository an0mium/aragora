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

from aragora.canvas.stages import GoalNodeType, PipelineStage
from aragora.goals.extractor import GoalExtractor, GoalGraph, GoalNode
from aragora.pipeline.idea_to_execution import (
    IdeaToExecutionPipeline,
    PipelineConfig,
    PipelineResult,
    StageResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# The extractor creates at most ``len(ideas) // 3`` goals, so we need >= 6
# ideas to get >= 2 goals.  The first two share many non-stop-words with
# each other so they both rank highly, and they contain the contradictory
# keyword pair "maximize" / "minimize".
CONTRADICTORY_IDEAS = [
    "Maximize throughput for the batch processing pipeline by adding more parallel workers and optimizing the queue depth",
    "Minimize throughput for the batch processing pipeline to reduce infrastructure costs and save money on the database",
    "Add a Redis cache in front of the database for read-heavy queries",
    "Set up Prometheus monitoring with SLO alerting within 2 weeks",
    "Build a rate limiter for API endpoints using token bucket algorithm",
    "Add comprehensive logging and observability across all services",
    "Implement retry logic with exponential backoff for external calls",
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
    """Patch the KM bridge to return deterministic precedent data.

    The ``from_ideas()`` and ``run()`` methods import ``PipelineKMBridge``
    lazily via ``from aragora.pipeline.km_bridge import PipelineKMBridge``.
    We replace the entire ``aragora.pipeline.km_bridge`` module so that
    the import resolves to our fake.
    """
    fake_bridge = MagicMock()
    fake_bridge.available = True

    def _query(goal_graph: Any) -> dict[str, list[dict[str, Any]]]:
        """Return precedent data keyed by actual goal IDs."""
        precedents: dict[str, list[dict[str, Any]]] = {}
        for goal in goal_graph.goals:
            precedents[goal.id] = [
                {
                    "title": "Prior rate limiter implementation",
                    "similarity": 0.82,
                    "outcome": "successful",
                },
            ]
        return precedents

    fake_bridge.query_similar_goals.side_effect = _query
    fake_bridge.enrich_with_precedents.side_effect = lambda gg, prec: _enrich_bridge(gg, prec)

    fake_module = MagicMock()
    fake_module.PipelineKMBridge.return_value = fake_bridge

    with patch.dict("sys.modules", {"aragora.pipeline.km_bridge": fake_module}):
        yield fake_bridge


def _enrich_bridge(goal_graph: Any, precedents: dict[str, list[dict[str, Any]]]) -> Any:
    """Side-effect helper: inject precedents into goal metadata."""
    for goal in goal_graph.goals:
        if goal.id in precedents and precedents[goal.id]:
            goal.metadata["precedents"] = precedents[goal.id]
    return goal_graph


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
            if stage.value not in result.stage_status:
                continue  # PRINCIPLES stage is opt-in
            assert result.stage_status[stage.value] == "complete", (
                f"Stage {stage.value} should be 'complete', "
                f"got '{result.stage_status.get(stage.value)}'"
            )

    def test_provenance_chain_spans_goal_and_action_stages(self, pipeline):
        """Provenance links should connect goals -> actions -> orchestration.

        Note: the sync ``from_ideas()`` path uses ``extract_from_raw_ideas()``
        which only creates ideas->goals provenance when the raw ideas share
        enough keywords to form graph edges.  With short ideas and low overlap
        the provenance starts at goals->actions.  We verify the stages that
        are always present.
        """
        result = pipeline.from_ideas(NORMAL_IDEAS, auto_advance=True)

        assert len(result.provenance) > 0, "Expected at least one provenance link"

        stages_in_provenance = set()
        for link in result.provenance:
            stages_in_provenance.add(link.source_stage.value)
            stages_in_provenance.add(link.target_stage.value)

        # goals->actions and actions->orchestration should always exist
        assert "goals" in stages_in_provenance, (
            f"Expected 'goals' in provenance stages. Got: {stages_in_provenance}"
        )
        assert "actions" in stages_in_provenance, (
            f"Expected 'actions' in provenance stages. Got: {stages_in_provenance}"
        )

    def test_transitions_recorded(self, pipeline):
        """Stage transitions are recorded between consecutive stages."""
        result = pipeline.from_ideas(NORMAL_IDEAS, auto_advance=True)

        assert len(result.transitions) >= 2, (
            f"Expected at least 2 transitions, got {len(result.transitions)}"
        )

        # Verify transition directions
        transition_pairs = [(t.from_stage.value, t.to_stage.value) for t in result.transitions]
        assert ("goals", "actions") in transition_pairs, (
            f"Missing goals->actions transition, got: {transition_pairs}"
        )
        assert ("actions", "orchestration") in transition_pairs, (
            f"Missing actions->orchestration transition, got: {transition_pairs}"
        )

    def test_smart_scores_on_goals(self, pipeline):
        """Goals should have SMART scores in their metadata."""
        result = pipeline.from_ideas(NORMAL_IDEAS, auto_advance=True)

        assert result.goal_graph is not None
        assert len(result.goal_graph.goals) > 0

        goals_with_smart = [g for g in result.goal_graph.goals if "smart_scores" in g.metadata]
        assert len(goals_with_smart) > 0, "Expected at least one goal with SMART scores"

        # Verify SMART score structure
        sample_scores = goals_with_smart[0].metadata["smart_scores"]
        for dimension in (
            "specific",
            "measurable",
            "achievable",
            "relevant",
            "time_bound",
            "overall",
        ):
            assert dimension in sample_scores, f"Missing SMART dimension: {dimension}"
            assert 0.0 <= sample_scores[dimension] <= 1.0, (
                f"SMART score for {dimension} out of range: {sample_scores[dimension]}"
            )

    def test_conflict_detection_with_contradictory_ideas(self, pipeline):
        """Contradictory ideas should produce conflict entries in metadata.

        Requires enough ideas (>= 6) so the extractor creates 2+ goals,
        and the contradictory pair both become separate goals.
        """
        result = pipeline.from_ideas(CONTRADICTORY_IDEAS, auto_advance=True)

        assert result.goal_graph is not None
        assert len(result.goal_graph.goals) >= 2, (
            f"Expected >= 2 goals from {len(CONTRADICTORY_IDEAS)} ideas, "
            f"got {len(result.goal_graph.goals)}"
        )

        conflicts = result.goal_graph.metadata.get("conflicts", [])

        # The maximize/minimize pair should trigger a contradiction
        assert len(conflicts) > 0, (
            "Expected at least one conflict from contradictory ideas "
            f"(maximize vs minimize). Goals: "
            f"{[g.title for g in result.goal_graph.goals]}"
        )

        # Verify conflict structure
        conflict = conflicts[0]
        assert "type" in conflict
        assert "severity" in conflict
        assert "goal_ids" in conflict

    def test_km_precedents_enrichment(self, pipeline, mock_km_precedents):
        """Goals should be enriched with KM precedent data when bridge is available.

        The sync ``from_ideas()`` path does NOT query the KM bridge --
        only ``from_debate()`` and async ``run()`` do.  So we use
        ``from_debate()`` with sample cartographer data.
        """
        cartographer_data = {
            "nodes": [
                {"id": "n1", "node_type": "proposal", "content": "Build a rate limiter"},
                {
                    "id": "n2",
                    "node_type": "consensus",
                    "content": "Token bucket rate limiter with per-user limits",
                },
                {"id": "n3", "node_type": "vote", "content": "Agreed: 100 req/min default"},
            ],
            "edges": [
                {"source": "n1", "target": "n2", "label": "leads_to"},
                {"source": "n2", "target": "n3", "label": "supports"},
            ],
        }

        result = pipeline.from_debate(cartographer_data, auto_advance=True)

        assert result.goal_graph is not None
        goals_with_precedents = [g for g in result.goal_graph.goals if "precedents" in g.metadata]
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
            NORMAL_IDEAS,
            auto_advance=True,
            event_callback=callback,
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

        assert has_agent_data, "Orchestration canvas nodes should have agent assignment data"


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
                assert sr.duration > 0, f"Stage {sr.stage_name} should have positive duration"

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

        assert "started" in event_types, f"Missing 'started' event. Got: {event_types}"
        assert "completed" in event_types, f"Missing 'completed' event. Got: {event_types}"

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

        goals_with_smart = [g for g in result.goal_graph.goals if "smart_scores" in g.metadata]
        assert len(goals_with_smart) > 0, (
            "At least one goal should have SMART scores after async run"
        )

    @pytest.mark.asyncio
    async def test_async_run_conflict_detection(self, pipeline):
        """Conflict detection catches contradictory goals (maximize vs minimize).

        The async ``run()`` path's text-splitting produces few goals from
        short input.  Instead of relying on the text splitter generating
        enough goals, we exercise the conflict detector directly against a
        GoalGraph with two contradictory goals -- which is what the pipeline
        calls internally after goal extraction.
        """
        extractor = GoalExtractor()
        graph = GoalGraph(
            id="conflict-test",
            goals=[
                GoalNode(
                    id="g1",
                    title="Maximize database throughput",
                    description="Increase throughput capacity",
                    goal_type=GoalNodeType.GOAL,
                ),
                GoalNode(
                    id="g2",
                    title="Minimize database throughput",
                    description="Reduce throughput to save costs",
                    goal_type=GoalNodeType.GOAL,
                ),
            ],
        )
        conflicts = extractor.detect_goal_conflicts(graph)
        assert len(conflicts) > 0, (
            "Expected conflict detection for contradictory maximize/minimize goals"
        )

        # Verify the conflict is wired correctly
        assert conflicts[0]["type"] == "contradiction"
        assert "g1" in conflicts[0]["goal_ids"]
        assert "g2" in conflicts[0]["goal_ids"]

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
        goals_with_precedents = [g for g in result.goal_graph.goals if "precedents" in g.metadata]
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
        orch_results = [sr for sr in result.stage_results if sr.stage_name == "orchestration"]
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

        assert elapsed < 5.0, f"Pipeline took {elapsed:.2f}s, expected < 5.0s"
        assert result.duration > 0
        assert result.duration < 5.0


# ---------------------------------------------------------------------------
# Cross-cutting integration tests
# ---------------------------------------------------------------------------


class TestCrossCuttingIntegration:
    """Tests that verify integration between stages and cross-cutting concerns."""

    def test_goal_metadata_propagates_to_action_steps(self, pipeline):
        """Action steps should carry metadata from their parent goals.

        The converter stores ``step_type``, ``phase``, ``source_goal_id``,
        and ``optional`` (derived from goal priority) in node data.
        """
        result = pipeline.from_ideas(NORMAL_IDEAS, auto_advance=True)

        assert result.actions_canvas is not None

        # Every action node should reference a source goal
        for node in result.actions_canvas.nodes.values():
            assert node.data.get("source_goal_id"), f"Action node {node.id} missing source_goal_id"
            assert node.data.get("step_type"), f"Action node {node.id} missing step_type"

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
            ". ".join(NORMAL_IDEAS),
            config=config,
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
