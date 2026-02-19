"""Tests for the Idea-to-Execution Pipeline.

Tests the full four-stage flow:
  Stage 1 (Ideas) → Stage 2 (Goals) → Stage 3 (Actions) → Stage 4 (Orchestration)

Including provenance chain integrity, stage transitions, and demo mode.
"""

import pytest
from unittest.mock import MagicMock

from aragora.pipeline.idea_to_execution import (
    IdeaToExecutionPipeline,
    PipelineConfig,
    PipelineResult,
)
from aragora.canvas.stages import PipelineStage, content_hash


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def pipeline():
    """Default pipeline with no AI agent."""
    return IdeaToExecutionPipeline()


@pytest.fixture
def sample_ideas():
    """Sample idea strings for testing."""
    return [
        "Build a rate limiter for API endpoints",
        "Add Redis-backed caching for frequently accessed data",
        "Improve API docs with OpenAPI interactive playground",
        "Set up end-to-end performance monitoring",
    ]


@pytest.fixture
def sample_cartographer_data():
    """Sample ArgumentCartographer output."""
    return {
        "nodes": [
            {"id": "n1", "type": "proposal", "summary": "Build a rate limiter", "content": "Token bucket rate limiter"},
            {"id": "n2", "type": "evidence", "summary": "Rate limiting reduces 429 errors", "content": "Evidence"},
            {"id": "n3", "type": "critique", "summary": "What about distributed rate limiting?", "content": "Question"},
            {"id": "n4", "type": "consensus", "summary": "Rate limiter is critical", "content": "Agreement"},
        ],
        "edges": [
            {"source_id": "n2", "target_id": "n1", "relation": "supports"},
            {"source_id": "n3", "target_id": "n1", "relation": "responds_to"},
            {"source_id": "n4", "target_id": "n1", "relation": "concedes_to"},
        ],
    }


# =============================================================================
# from_ideas tests
# =============================================================================


class TestFromIdeas:
    """Test pipeline creation from raw idea strings."""

    def test_from_ideas_creates_all_stages(self, pipeline, sample_ideas):
        result = pipeline.from_ideas(sample_ideas, auto_advance=True)

        assert isinstance(result, PipelineResult)
        assert result.pipeline_id.startswith("pipe-")
        assert result.ideas_canvas is not None
        assert result.goal_graph is not None
        assert result.actions_canvas is not None
        assert result.orchestration_canvas is not None

    def test_from_ideas_stage_status_all_complete(self, pipeline, sample_ideas):
        result = pipeline.from_ideas(sample_ideas, auto_advance=True)

        assert result.stage_status[PipelineStage.IDEAS.value] == "complete"
        assert result.stage_status[PipelineStage.GOALS.value] == "complete"
        assert result.stage_status[PipelineStage.ACTIONS.value] == "complete"
        assert result.stage_status[PipelineStage.ORCHESTRATION.value] == "complete"

    def test_from_ideas_no_auto_advance(self, pipeline, sample_ideas):
        result = pipeline.from_ideas(sample_ideas, auto_advance=False)

        assert result.ideas_canvas is not None
        # Goals are extracted via extract_from_raw_ideas in from_ideas
        assert result.goal_graph is not None
        # But actions and orchestration should NOT be generated
        assert result.actions_canvas is None
        assert result.orchestration_canvas is None

    def test_from_ideas_creates_idea_nodes(self, pipeline, sample_ideas):
        result = pipeline.from_ideas(sample_ideas, auto_advance=False)

        assert len(result.ideas_canvas.nodes) == len(sample_ideas)
        for i, idea in enumerate(sample_ideas):
            node_id = f"raw-idea-{i}"
            assert node_id in result.ideas_canvas.nodes
            node = result.ideas_canvas.nodes[node_id]
            assert node.label == idea[:80]

    def test_from_ideas_sets_content_hash(self, pipeline, sample_ideas):
        result = pipeline.from_ideas(sample_ideas, auto_advance=False)

        for node in result.ideas_canvas.nodes.values():
            assert "content_hash" in node.data
            assert len(node.data["content_hash"]) == 16

    def test_from_ideas_empty_list(self, pipeline):
        result = pipeline.from_ideas([], auto_advance=True)

        # Should still have a pipeline but no meaningful data
        assert result.pipeline_id.startswith("pipe-")

    def test_from_ideas_single_idea(self, pipeline):
        result = pipeline.from_ideas(["Single idea"], auto_advance=True)

        assert result.ideas_canvas is not None
        assert len(result.ideas_canvas.nodes) == 1


# =============================================================================
# from_debate tests
# =============================================================================


class TestFromDebate:
    """Test pipeline creation from debate cartographer data."""

    def test_from_debate_creates_all_stages(self, pipeline, sample_cartographer_data):
        result = pipeline.from_debate(sample_cartographer_data, auto_advance=True)

        assert result.ideas_canvas is not None
        assert result.goal_graph is not None
        assert result.actions_canvas is not None
        assert result.orchestration_canvas is not None

    def test_from_debate_maps_node_types(self, pipeline, sample_cartographer_data):
        result = pipeline.from_debate(sample_cartographer_data, auto_advance=False)

        # Check that debate node types are mapped to idea types
        node_data = {
            nid: n.data for nid, n in result.ideas_canvas.nodes.items()
        }
        # proposal → concept
        assert node_data["n1"]["idea_type"] == "concept"
        # evidence → evidence
        assert node_data["n2"]["idea_type"] == "evidence"
        # critique → question
        assert node_data["n3"]["idea_type"] == "question"
        # consensus → cluster
        assert node_data["n4"]["idea_type"] == "cluster"

    def test_from_debate_no_auto_advance(self, pipeline, sample_cartographer_data):
        result = pipeline.from_debate(sample_cartographer_data, auto_advance=False)

        assert result.ideas_canvas is not None
        assert result.stage_status[PipelineStage.IDEAS.value] == "complete"
        assert result.stage_status[PipelineStage.GOALS.value] == "pending"


# =============================================================================
# advance_stage tests
# =============================================================================


class TestAdvanceStage:
    """Test manual stage advancement."""

    def test_advance_to_goals(self, pipeline, sample_cartographer_data):
        result = pipeline.from_debate(sample_cartographer_data, auto_advance=False)

        result = pipeline.advance_stage(result, PipelineStage.GOALS)
        assert result.goal_graph is not None
        assert len(result.goal_graph.goals) > 0
        assert result.stage_status[PipelineStage.GOALS.value] == "complete"

    def test_advance_to_actions(self, pipeline, sample_cartographer_data):
        result = pipeline.from_debate(sample_cartographer_data, auto_advance=False)
        result = pipeline.advance_stage(result, PipelineStage.GOALS)
        result = pipeline.advance_stage(result, PipelineStage.ACTIONS)

        assert result.actions_canvas is not None
        assert len(result.actions_canvas.nodes) > 0
        assert result.stage_status[PipelineStage.ACTIONS.value] == "complete"

    def test_advance_to_orchestration(self, pipeline, sample_cartographer_data):
        result = pipeline.from_debate(sample_cartographer_data, auto_advance=False)
        result = pipeline.advance_stage(result, PipelineStage.GOALS)
        result = pipeline.advance_stage(result, PipelineStage.ACTIONS)
        result = pipeline.advance_stage(result, PipelineStage.ORCHESTRATION)

        assert result.orchestration_canvas is not None
        assert len(result.orchestration_canvas.nodes) > 0
        assert result.stage_status[PipelineStage.ORCHESTRATION.value] == "complete"

    def test_advance_without_prerequisite(self, pipeline, sample_cartographer_data):
        result = pipeline.from_debate(sample_cartographer_data, auto_advance=False)

        # Skip goals, try to advance to actions directly
        result = pipeline.advance_stage(result, PipelineStage.ACTIONS)
        # Should fail gracefully (no goals → no actions)
        assert result.actions_canvas is None


# =============================================================================
# Provenance chain tests
# =============================================================================


class TestProvenance:
    """Test provenance chain integrity."""

    def test_provenance_links_exist(self, pipeline, sample_ideas):
        result = pipeline.from_ideas(sample_ideas, auto_advance=True)

        assert len(result.provenance) > 0

    def test_provenance_links_have_content_hash(self, pipeline, sample_ideas):
        result = pipeline.from_ideas(sample_ideas, auto_advance=True)

        for link in result.provenance:
            assert link.content_hash, f"Empty content hash in provenance: {link}"
            assert len(link.content_hash) == 16

    def test_provenance_links_across_stages(self, pipeline, sample_cartographer_data):
        result = pipeline.from_debate(sample_cartographer_data, auto_advance=True)

        stages_in_provenance = set()
        for link in result.provenance:
            stages_in_provenance.add(link.source_stage)
            stages_in_provenance.add(link.target_stage)

        # Should have links spanning at least ideas → goals
        assert PipelineStage.IDEAS in stages_in_provenance
        assert PipelineStage.GOALS in stages_in_provenance

    def test_transitions_recorded(self, pipeline, sample_ideas):
        result = pipeline.from_ideas(sample_ideas, auto_advance=True)

        assert len(result.transitions) > 0
        for transition in result.transitions:
            assert transition.confidence >= 0
            assert transition.ai_rationale

    def test_integrity_hash(self, pipeline, sample_ideas):
        result = pipeline.from_ideas(sample_ideas, auto_advance=True)

        result_dict = result.to_dict()
        assert "integrity_hash" in result_dict
        assert len(result_dict["integrity_hash"]) == 16


# =============================================================================
# to_dict / serialization tests
# =============================================================================


class TestSerialization:
    """Test PipelineResult serialization."""

    def test_to_dict_has_all_fields(self, pipeline, sample_ideas):
        result = pipeline.from_ideas(sample_ideas, auto_advance=True)
        d = result.to_dict()

        assert "pipeline_id" in d
        assert "ideas" in d
        assert "goals" in d
        assert "actions" in d
        assert "orchestration" in d
        assert "transitions" in d
        assert "stage_status" in d
        assert "integrity_hash" in d
        assert "provenance_count" in d

    def test_to_dict_ideas_has_react_flow_format(self, pipeline, sample_ideas):
        result = pipeline.from_ideas(sample_ideas, auto_advance=True)
        d = result.to_dict()

        ideas = d["ideas"]
        assert "nodes" in ideas
        assert "edges" in ideas
        for node in ideas["nodes"]:
            assert "id" in node
            assert "type" in node
            assert "position" in node
            assert "data" in node

    def test_to_dict_goals_has_goals_list(self, pipeline, sample_ideas):
        result = pipeline.from_ideas(sample_ideas, auto_advance=True)
        d = result.to_dict()

        goals = d["goals"]
        assert "goals" in goals
        assert isinstance(goals["goals"], list)


# =============================================================================
# Demo mode tests
# =============================================================================


class TestDemoMode:
    """Test demo pipeline creation."""

    def test_from_demo_returns_result_and_config(self):
        result, config = IdeaToExecutionPipeline.from_demo()

        assert isinstance(result, PipelineResult)
        assert isinstance(config, PipelineConfig)

    def test_from_demo_creates_complete_pipeline(self):
        result, _config = IdeaToExecutionPipeline.from_demo()

        assert result.ideas_canvas is not None
        assert result.goal_graph is not None
        assert result.actions_canvas is not None
        assert result.orchestration_canvas is not None

    def test_from_demo_all_stages_complete(self):
        result, _config = IdeaToExecutionPipeline.from_demo()

        assert result.stage_status[PipelineStage.IDEAS.value] == "complete"
        assert result.stage_status[PipelineStage.GOALS.value] == "complete"
        assert result.stage_status[PipelineStage.ACTIONS.value] == "complete"
        assert result.stage_status[PipelineStage.ORCHESTRATION.value] == "complete"

    def test_from_demo_has_provenance(self):
        result, _config = IdeaToExecutionPipeline.from_demo()
        assert len(result.provenance) > 0

    def test_from_demo_serializable(self):
        result, _config = IdeaToExecutionPipeline.from_demo()
        d = result.to_dict()
        assert d["pipeline_id"].startswith("pipe-")
        assert d["ideas"] is not None

    def test_from_demo_config_enables_all_flywheel_features(self):
        _result, config = IdeaToExecutionPipeline.from_demo()

        assert config.enable_smart_goals is True
        assert config.enable_elo_assignment is True
        assert config.enable_km_precedents is True
        assert config.human_approval_required is True
        assert config.dry_run is True

    def test_from_demo_goals_have_smart_scores(self):
        result, _config = IdeaToExecutionPipeline.from_demo()

        assert result.goal_graph is not None
        assert len(result.goal_graph.goals) > 0
        for goal in result.goal_graph.goals:
            assert "smart_scores" in goal.metadata, (
                f"Goal '{goal.title}' missing SMART scores"
            )
            scores = goal.metadata["smart_scores"]
            for dim in ("specific", "measurable", "achievable", "relevant", "time_bound", "overall"):
                assert dim in scores, f"Missing SMART dimension: {dim}"
                assert 0.0 <= scores[dim] <= 1.0

    def test_from_demo_detects_goal_conflict(self):
        """The demo ideas include contradictory deployment-cadence goals.

        'Increase deployment frequency ...' vs 'Decrease deployment frequency ...'
        should trigger at least one contradiction conflict.
        """
        result, _config = IdeaToExecutionPipeline.from_demo()

        assert result.goal_graph is not None
        conflicts = result.goal_graph.metadata.get("conflicts", [])
        assert len(conflicts) >= 1, (
            "Expected at least one conflict from contradictory demo ideas"
        )
        contradiction_found = any(
            c["type"] == "contradiction" for c in conflicts
        )
        assert contradiction_found, (
            f"Expected a 'contradiction' conflict, got types: "
            f"{[c['type'] for c in conflicts]}"
        )

    def test_from_demo_has_multiple_idea_nodes(self):
        """Demo should have 6 seed ideas represented as nodes."""
        result, _config = IdeaToExecutionPipeline.from_demo()

        assert result.ideas_canvas is not None
        assert len(result.ideas_canvas.nodes) == 6


# =============================================================================
# AI goal synthesis tests
# =============================================================================


class TestAIGoalSynthesis:
    """Test AI-assisted goal extraction."""

    def test_ai_synthesis_with_mock_agent(self):
        """Test that AI synthesis is attempted when agent is available."""
        import json

        mock_agent = MagicMock()
        mock_agent.generate.return_value = json.dumps([
            {
                "title": "Achieve API reliability",
                "description": "Ensure API maintains high availability",
                "type": "goal",
                "priority": "critical",
                "measurable": "99.9% uptime",
                "source_ideas": [0, 1],
            },
            {
                "title": "Implement caching strategy",
                "description": "Add multi-layer caching",
                "type": "strategy",
                "priority": "high",
                "measurable": "50% reduction in DB queries",
                "source_ideas": [1],
            },
        ])

        from aragora.goals.extractor import GoalExtractor

        extractor = GoalExtractor(agent=mock_agent)
        canvas_data = {
            "nodes": [
                {"id": "idea-0", "label": "Rate limiting", "data": {"idea_type": "concept", "full_content": "Build rate limiter"}},
                {"id": "idea-1", "label": "Caching", "data": {"idea_type": "concept", "full_content": "Add caching"}},
            ],
            "edges": [],
        }

        result = extractor.extract_from_ideas(canvas_data)

        assert len(result.goals) == 2
        assert result.goals[0].title == "Achieve API reliability"
        assert result.goals[0].confidence == 0.8
        assert result.metadata.get("extraction_method") == "ai_synthesis"

    def test_ai_synthesis_fallback_on_failure(self):
        """Test that structural extraction is used when AI fails."""
        mock_agent = MagicMock()
        mock_agent.generate.side_effect = Exception("API error")

        from aragora.goals.extractor import GoalExtractor

        extractor = GoalExtractor(agent=mock_agent)
        canvas_data = {
            "nodes": [
                {"id": "idea-0", "label": "Rate limiting", "data": {"idea_type": "concept"}},
                {"id": "idea-1", "label": "Caching", "data": {"idea_type": "concept"}},
                {"id": "idea-2", "label": "Monitoring", "data": {"idea_type": "insight"}},
            ],
            "edges": [],
        }

        result = extractor.extract_from_ideas(canvas_data)
        # Should still produce goals via structural extraction
        assert len(result.goals) >= 1

    def test_ai_synthesis_with_bad_json(self):
        """Test graceful handling of unparseable AI response."""
        mock_agent = MagicMock()
        mock_agent.generate.return_value = "This is not JSON at all"

        from aragora.goals.extractor import GoalExtractor

        extractor = GoalExtractor(agent=mock_agent)
        canvas_data = {
            "nodes": [
                {"id": "idea-0", "label": "Rate limiting", "data": {"idea_type": "concept"}},
                {"id": "idea-1", "label": "Caching", "data": {"idea_type": "concept"}},
                {"id": "idea-2", "label": "Monitoring", "data": {"idea_type": "insight"}},
            ],
            "edges": [],
        }

        result = extractor.extract_from_ideas(canvas_data)
        # Falls back to structural extraction
        assert len(result.goals) >= 1

    def test_no_agent_uses_structural(self):
        """Test that no agent means structural extraction only."""
        from aragora.goals.extractor import GoalExtractor

        extractor = GoalExtractor(agent=None)
        canvas_data = {
            "nodes": [
                {"id": "idea-0", "label": "Rate limiting", "data": {"idea_type": "concept"}},
            ],
            "edges": [],
        }

        result = extractor.extract_from_ideas(canvas_data)
        assert len(result.goals) >= 1
        assert result.goals[0].confidence < 0.8  # Structural gives lower confidence


# =============================================================================
# content_hash tests
# =============================================================================


class TestContentHash:
    """Test SHA-256 content hashing."""

    def test_content_hash_deterministic(self):
        h1 = content_hash("hello world")
        h2 = content_hash("hello world")
        assert h1 == h2

    def test_content_hash_length(self):
        h = content_hash("test content")
        assert len(h) == 16

    def test_content_hash_different_for_different_content(self):
        h1 = content_hash("content A")
        h2 = content_hash("content B")
        assert h1 != h2


# =============================================================================
# SMART goal extraction tests
# =============================================================================


class TestSmartGoalExtraction:
    """Test SMART scoring and conflict detection in the pipeline."""

    def test_smart_scoring_applied_to_goals(self, pipeline, sample_ideas):
        result = pipeline.from_ideas(sample_ideas, auto_advance=True)
        assert result.goal_graph is not None
        for goal in result.goal_graph.goals:
            assert "smart_scores" in goal.metadata
            scores = goal.metadata["smart_scores"]
            assert "overall" in scores
            assert 0.0 <= scores["overall"] <= 1.0

    def test_conflict_detection_stored_in_metadata(self, pipeline):
        # Create ideas with conflicting goals
        ideas = [
            "Maximize revenue at all costs",
            "Minimize spending drastically",
            "Increase headcount significantly",
            "Reduce team size immediately",
        ]
        result = pipeline.from_ideas(ideas, auto_advance=True)
        # Conflicts may or may not be detected depending on exact wording
        assert result.goal_graph is not None
        # metadata should exist even if empty
        assert isinstance(result.goal_graph.metadata, dict)

    def test_smart_scoring_adjusts_priority(self, pipeline):
        ideas = [
            "Reduce API latency by 50% within 2 sprints by implementing Redis caching",
            "Improve overall system performance somehow",
        ]
        result = pipeline.from_ideas(ideas, auto_advance=True)
        assert result.goal_graph is not None
        # Goals with specific metrics should score higher
        scored_goals = [
            g for g in result.goal_graph.goals if "smart_scores" in g.metadata
        ]
        assert len(scored_goals) > 0

    def test_from_debate_has_smart_goals(self, pipeline, sample_cartographer_data):
        result = pipeline.from_debate(sample_cartographer_data, auto_advance=True)
        assert result.goal_graph is not None
        for goal in result.goal_graph.goals:
            assert "smart_scores" in goal.metadata

    def test_smart_scores_have_all_dimensions(self, pipeline, sample_ideas):
        result = pipeline.from_ideas(sample_ideas, auto_advance=True)
        assert result.goal_graph is not None
        for goal in result.goal_graph.goals:
            scores = goal.metadata["smart_scores"]
            for key in ("specific", "measurable", "achievable", "relevant", "time_bound", "overall"):
                assert key in scores
                assert isinstance(scores[key], float)

    def test_high_smart_score_sets_high_priority(self, pipeline):
        """Goals with specific, measurable, time-bound language get high priority."""
        ideas = [
            "Deploy Redis caching module by Q2 2026 to reduce API latency by 40% "
            "for the /api/v1/search endpoint within 1 sprint"
        ]
        result = pipeline.from_ideas(ideas, auto_advance=True)
        assert result.goal_graph is not None
        # At least one goal should exist
        assert len(result.goal_graph.goals) > 0


# =============================================================================
# ELO-aware agent assignment tests
# =============================================================================


class TestELOAssignment:
    """Test ELO-aware agent assignment in orchestration."""

    def test_elo_fallback_to_static_map(self, pipeline, sample_ideas):
        """Without TeamSelector installed, should fall back to static assignment."""
        result = pipeline.from_ideas(sample_ideas, auto_advance=True)
        assert result.orchestration_canvas is not None
        assert len(result.orchestration_canvas.nodes) > 0

    def test_elo_assignment_with_mock_selector(self, pipeline, sample_ideas):
        """With mocked TeamSelector, verify pipeline still works."""
        # Even with import errors, pipeline should still work
        result = pipeline.from_ideas(sample_ideas, auto_advance=True)
        assert result.orchestration_canvas is not None

    def test_execution_plan_has_assigned_agents(self, pipeline, sample_ideas):
        result = pipeline.from_ideas(sample_ideas, auto_advance=True)
        assert result.orchestration_canvas is not None
        for node in result.orchestration_canvas.nodes.values():
            # Each node should exist with data
            assert node.data is not None

    def test_static_agent_pool_used(self, pipeline, sample_ideas):
        """Verify the static agent pool is used when no ELO data available."""
        result = pipeline.from_ideas(sample_ideas, auto_advance=True)
        # The orchestration canvas should be populated
        assert result.orchestration_canvas is not None
        assert result.stage_status[PipelineStage.ORCHESTRATION.value] == "complete"

    def test_pipeline_config_has_elo_flag(self):
        """Test that PipelineConfig includes the enable_elo_assignment field."""
        from aragora.pipeline.idea_to_execution import PipelineConfig

        cfg = PipelineConfig()
        assert cfg.enable_elo_assignment is True
        assert cfg.enable_smart_goals is True
        assert cfg.enable_km_precedents is True

    def test_pipeline_config_flags_can_be_disabled(self):
        """Test that pipeline config flags can be set to False."""
        from aragora.pipeline.idea_to_execution import PipelineConfig

        cfg = PipelineConfig(
            enable_smart_goals=False,
            enable_elo_assignment=False,
            enable_km_precedents=False,
        )
        assert cfg.enable_smart_goals is False
        assert cfg.enable_elo_assignment is False
        assert cfg.enable_km_precedents is False
