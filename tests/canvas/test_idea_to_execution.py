"""
Tests for the Idea-to-Execution Pipeline.

Covers all four stages and the transitions between them:
- Stage types and provenance
- Converters (debate → ideas, goals → canvas, workflow → actions, execution → orch)
- Goal extraction from idea graphs
- Full pipeline flow (from_debate, from_ideas)
- React Flow JSON export format
- Handler endpoints
"""

from __future__ import annotations

import json

import pytest

from aragora.canvas.converters import (
    _hierarchical_layout,
    _radial_layout,
    debate_to_ideas_canvas,
    execution_to_orchestration_canvas,
    goals_to_canvas,
    to_react_flow,
    workflow_to_actions_canvas,
)
from aragora.canvas.models import Canvas, CanvasNode, CanvasNodeType, Position
from aragora.canvas.stages import (
    NODE_TYPE_COLORS,
    STAGE_COLORS,
    GoalNodeType,
    IdeaNodeType,
    PipelineStage,
    ProvenanceLink,
    StageTransition,
    content_hash,
)
from aragora.goals.extractor import GoalExtractor, GoalGraph, GoalNode
from aragora.pipeline.idea_to_execution import IdeaToExecutionPipeline, PipelineResult


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_debate_data():
    """Sample ArgumentCartographer.to_dict() output."""
    return {
        "nodes": [
            {
                "id": "n1",
                "type": "proposal",
                "summary": "Use microservices for scalability",
                "full_content": "We should use microservices to scale the system",
                "agent": "claude",
                "round_num": 1,
            },
            {
                "id": "n2",
                "type": "critique",
                "summary": "Microservices add operational complexity",
                "full_content": "Microservices add significant operational overhead",
                "agent": "gpt-4",
                "round_num": 1,
            },
            {
                "id": "n3",
                "type": "evidence",
                "summary": "Netflix scaled with microservices",
                "full_content": "Netflix successfully scaled to 200M+ users with microservices",
                "agent": "claude",
                "round_num": 2,
            },
            {
                "id": "n4",
                "type": "concession",
                "summary": "Complexity is manageable with Kubernetes",
                "full_content": "Complexity is manageable with proper tooling like K8s",
                "agent": "gpt-4",
                "round_num": 2,
            },
            {
                "id": "n5",
                "type": "consensus",
                "summary": "Adopt microservices with K8s for orchestration",
                "full_content": "Team agrees to adopt microservices with K8s",
                "agent": "system",
                "round_num": 3,
            },
        ],
        "edges": [
            {"source_id": "n2", "target_id": "n1", "relation": "refutes"},
            {"source_id": "n3", "target_id": "n1", "relation": "supports"},
            {"source_id": "n4", "target_id": "n2", "relation": "concedes_to"},
            {"source_id": "n5", "target_id": "n1", "relation": "supports"},
        ],
    }


@pytest.fixture
def sample_goals():
    """Sample goal list for Stage 2."""
    return [
        {
            "id": "g1",
            "type": "goal",
            "title": "Implement microservices architecture",
            "description": "Break monolith into domain services",
            "priority": "high",
            "dependencies": [],
        },
        {
            "id": "g2",
            "type": "milestone",
            "title": "Deploy K8s cluster",
            "description": "Set up production Kubernetes cluster",
            "priority": "high",
            "dependencies": [],
        },
        {
            "id": "g3",
            "type": "goal",
            "title": "Migrate auth service first",
            "description": "Extract authentication into standalone service",
            "priority": "medium",
            "dependencies": ["g1", "g2"],
        },
    ]


@pytest.fixture
def sample_workflow():
    """Sample WorkflowDefinition.to_dict() output."""
    return {
        "id": "wf-1",
        "name": "Migration Workflow",
        "steps": [
            {
                "id": "s1",
                "name": "Setup K8s",
                "step_type": "task",
                "description": "Provision cluster",
                "visual": {"position": {"x": 0, "y": 0}},
            },
            {
                "id": "s2",
                "name": "Deploy Auth Service",
                "step_type": "implementation",
                "description": "Deploy auth microservice",
                "visual": {"position": {"x": 300, "y": 0}},
            },
            {
                "id": "s3",
                "name": "Verify Auth",
                "step_type": "human_checkpoint",
                "description": "Manual verification",
                "visual": {"position": {"x": 600, "y": 0}},
            },
        ],
        "transitions": [
            {"id": "t1", "from_step": "s1", "to_step": "s2", "label": "then"},
            {"id": "t2", "from_step": "s2", "to_step": "s3", "label": "verify"},
        ],
    }


@pytest.fixture
def sample_execution_plan():
    """Sample execution plan for Stage 4."""
    return {
        "agents": [
            {"id": "a1", "name": "Analyst", "type": "claude", "capabilities": ["research"]},
            {"id": "a2", "name": "Developer", "type": "codex", "capabilities": ["code"]},
        ],
        "tasks": [
            {"id": "t1", "name": "Research best practices", "type": "agent_task",
             "assigned_agent": "a1", "depends_on": []},
            {"id": "t2", "name": "Implement service", "type": "agent_task",
             "assigned_agent": "a2", "depends_on": ["t1"]},
            {"id": "t3", "name": "Review code", "type": "verification",
             "assigned_agent": "a1", "depends_on": ["t2"]},
        ],
    }


# =============================================================================
# Stage Types Tests
# =============================================================================


class TestPipelineStages:
    """Test pipeline stage types and provenance."""

    def test_pipeline_stage_values(self):
        assert PipelineStage.IDEAS.value == "ideas"
        assert PipelineStage.GOALS.value == "goals"
        assert PipelineStage.ACTIONS.value == "actions"
        assert PipelineStage.ORCHESTRATION.value == "orchestration"

    def test_all_stages_have_colors(self):
        for stage in PipelineStage:
            assert stage in STAGE_COLORS
            colors = STAGE_COLORS[stage]
            assert "primary" in colors
            assert "secondary" in colors
            assert "accent" in colors

    def test_idea_node_types(self):
        assert len(IdeaNodeType) == 9
        assert IdeaNodeType.CONCEPT.value == "concept"
        assert IdeaNodeType.CLUSTER.value == "cluster"

    def test_goal_node_types(self):
        assert len(GoalNodeType) == 6
        assert GoalNodeType.GOAL.value == "goal"
        assert GoalNodeType.PRINCIPLE.value == "principle"

    def test_content_hash_deterministic(self):
        h1 = content_hash("test content")
        h2 = content_hash("test content")
        assert h1 == h2
        assert len(h1) == 16

    def test_content_hash_different_inputs(self):
        h1 = content_hash("hello")
        h2 = content_hash("world")
        assert h1 != h2

    def test_provenance_link_serialization(self):
        link = ProvenanceLink(
            source_node_id="n1",
            source_stage=PipelineStage.IDEAS,
            target_node_id="g1",
            target_stage=PipelineStage.GOALS,
            content_hash="abc123",
            method="structural_extraction",
        )
        d = link.to_dict()
        assert d["source_node_id"] == "n1"
        assert d["source_stage"] == "ideas"
        assert d["target_stage"] == "goals"
        assert d["method"] == "structural_extraction"

    def test_stage_transition_serialization(self):
        trans = StageTransition(
            id="t1",
            from_stage=PipelineStage.IDEAS,
            to_stage=PipelineStage.GOALS,
            status="approved",
            confidence=0.85,
            ai_rationale="Extracted 3 goals from 5 ideas",
        )
        d = trans.to_dict()
        assert d["from_stage"] == "ideas"
        assert d["to_stage"] == "goals"
        assert d["confidence"] == 0.85

    def test_node_type_colors_coverage(self):
        """Every node type should have a color."""
        for idea_type in IdeaNodeType:
            assert idea_type.value in NODE_TYPE_COLORS
        for goal_type in GoalNodeType:
            assert goal_type.value in NODE_TYPE_COLORS


# =============================================================================
# Converter Tests
# =============================================================================


class TestDebateToIdeasConverter:
    """Test converting ArgumentCartographer output to Ideas canvas."""

    def test_basic_conversion(self, sample_debate_data):
        canvas = debate_to_ideas_canvas(sample_debate_data)
        assert isinstance(canvas, Canvas)
        assert len(canvas.nodes) == 5
        assert len(canvas.edges) == 4
        assert canvas.metadata["stage"] == "ideas"

    def test_node_types_mapped(self, sample_debate_data):
        canvas = debate_to_ideas_canvas(sample_debate_data)
        node_types = {
            n.data["idea_type"] for n in canvas.nodes.values()
        }
        assert "concept" in node_types  # from proposal
        assert "question" in node_types  # from critique
        assert "evidence" in node_types  # from evidence
        assert "cluster" in node_types  # from consensus

    def test_edge_types_mapped(self, sample_debate_data):
        canvas = debate_to_ideas_canvas(sample_debate_data)
        edge_labels = {e.label for e in canvas.edges.values()}
        assert "refutes" in edge_labels
        assert "supports" in edge_labels

    def test_nodes_have_positions(self, sample_debate_data):
        canvas = debate_to_ideas_canvas(sample_debate_data)
        for node in canvas.nodes.values():
            assert isinstance(node.position, Position)

    def test_nodes_have_content_hash(self, sample_debate_data):
        canvas = debate_to_ideas_canvas(sample_debate_data)
        for node in canvas.nodes.values():
            assert "content_hash" in node.data
            assert len(node.data["content_hash"]) == 16

    def test_custom_canvas_name(self, sample_debate_data):
        canvas = debate_to_ideas_canvas(sample_debate_data, canvas_name="My Ideas")
        assert canvas.name == "My Ideas"

    def test_empty_debate(self):
        canvas = debate_to_ideas_canvas({"nodes": [], "edges": []})
        assert len(canvas.nodes) == 0
        assert len(canvas.edges) == 0


class TestGoalsToCanvas:
    """Test converting goals to Canvas."""

    def test_basic_conversion(self, sample_goals):
        canvas = goals_to_canvas(sample_goals)
        assert len(canvas.nodes) == 3
        assert canvas.metadata["stage"] == "goals"

    def test_dependency_edges(self, sample_goals):
        canvas = goals_to_canvas(sample_goals)
        dep_edges = [
            e for e in canvas.edges.values()
            if e.label == "requires"
        ]
        assert len(dep_edges) == 2  # g3 depends on g1 and g2

    def test_provenance_edges(self, sample_goals):
        prov = [
            ProvenanceLink(
                source_node_id="n1",
                source_stage=PipelineStage.IDEAS,
                target_node_id="g1",
                target_stage=PipelineStage.GOALS,
                content_hash="abc",
                method="test",
            )
        ]
        canvas = goals_to_canvas(sample_goals, provenance=prov)
        prov_edges = [
            e for e in canvas.edges.values()
            if e.data.get("provenance")
        ]
        assert len(prov_edges) == 1


class TestWorkflowToActionsCanvas:
    """Test converting WorkflowDefinition to Actions canvas."""

    def test_basic_conversion(self, sample_workflow):
        canvas = workflow_to_actions_canvas(sample_workflow)
        assert len(canvas.nodes) == 3
        assert canvas.metadata["stage"] == "actions"

    def test_transition_edges(self, sample_workflow):
        canvas = workflow_to_actions_canvas(sample_workflow)
        assert len(canvas.edges) == 2

    def test_visual_positions_preserved(self, sample_workflow):
        canvas = workflow_to_actions_canvas(sample_workflow)
        s1 = canvas.nodes.get("s1")
        assert s1 is not None
        assert s1.position.x == 0
        assert s1.position.y == 0


class TestExecutionToOrchestrationCanvas:
    """Test converting execution plans to Orchestration canvas."""

    def test_basic_conversion(self, sample_execution_plan):
        canvas = execution_to_orchestration_canvas(sample_execution_plan)
        # 2 agents + 3 tasks = 5 nodes
        assert len(canvas.nodes) == 5
        assert canvas.metadata["stage"] == "orchestration"

    def test_agent_nodes(self, sample_execution_plan):
        canvas = execution_to_orchestration_canvas(sample_execution_plan)
        agent_nodes = [
            n for n in canvas.nodes.values()
            if n.data.get("orch_type") == "agent"
        ]
        assert len(agent_nodes) == 2

    def test_dependency_edges(self, sample_execution_plan):
        canvas = execution_to_orchestration_canvas(sample_execution_plan)
        dep_edges = [
            e for e in canvas.edges.values()
            if e.label == "blocks"
        ]
        assert len(dep_edges) == 2  # t2 depends on t1, t3 depends on t2


# =============================================================================
# React Flow Export Tests
# =============================================================================


class TestReactFlowExport:
    """Test React Flow JSON format compliance."""

    def test_format_structure(self, sample_debate_data):
        canvas = debate_to_ideas_canvas(sample_debate_data)
        rf = to_react_flow(canvas)
        assert "nodes" in rf
        assert "edges" in rf
        assert "metadata" in rf

    def test_node_format(self, sample_debate_data):
        canvas = debate_to_ideas_canvas(sample_debate_data)
        rf = to_react_flow(canvas)
        node = rf["nodes"][0]
        assert "id" in node
        assert "type" in node
        assert "position" in node
        assert "x" in node["position"]
        assert "y" in node["position"]
        assert "data" in node
        assert "label" in node["data"]

    def test_edge_format(self, sample_debate_data):
        canvas = debate_to_ideas_canvas(sample_debate_data)
        rf = to_react_flow(canvas)
        edge = rf["edges"][0]
        assert "id" in edge
        assert "source" in edge
        assert "target" in edge
        assert "type" in edge

    def test_metadata_includes_stage(self, sample_debate_data):
        canvas = debate_to_ideas_canvas(sample_debate_data)
        rf = to_react_flow(canvas)
        assert rf["metadata"]["stage"] == "ideas"


# =============================================================================
# Layout Tests
# =============================================================================


class TestLayoutHelpers:
    """Test radial and hierarchical layout algorithms."""

    def test_radial_layout_single(self):
        positions = _radial_layout(1)
        assert len(positions) == 1

    def test_radial_layout_multiple(self):
        positions = _radial_layout(6)
        assert len(positions) == 6
        # All positions should be different
        coords = {(p.x, p.y) for p in positions}
        assert len(coords) == 6

    def test_radial_layout_empty(self):
        assert _radial_layout(0) == []

    def test_hierarchical_layout_with_deps(self):
        items = [
            {"id": "a", "dependencies": []},
            {"id": "b", "dependencies": ["a"]},
            {"id": "c", "dependencies": ["b"]},
        ]
        positions = _hierarchical_layout(items)
        assert positions["a"].y < positions["b"].y
        assert positions["b"].y < positions["c"].y

    def test_hierarchical_layout_no_deps(self):
        items = [
            {"id": "a", "dependencies": []},
            {"id": "b", "dependencies": []},
        ]
        positions = _hierarchical_layout(items)
        # Both at depth 0
        assert positions["a"].y == positions["b"].y


# =============================================================================
# Goal Extractor Tests
# =============================================================================


class TestGoalExtractor:
    """Test structural goal extraction from idea graphs."""

    def test_extract_from_debate_canvas(self, sample_debate_data):
        canvas = debate_to_ideas_canvas(sample_debate_data)
        extractor = GoalExtractor()
        goal_graph = extractor.extract_from_ideas(canvas.to_dict())
        assert isinstance(goal_graph, GoalGraph)
        assert len(goal_graph.goals) > 0

    def test_goals_have_titles(self, sample_debate_data):
        canvas = debate_to_ideas_canvas(sample_debate_data)
        extractor = GoalExtractor()
        goal_graph = extractor.extract_from_ideas(canvas.to_dict())
        for goal in goal_graph.goals:
            assert goal.title
            assert len(goal.title) > 5

    def test_goals_have_provenance(self, sample_debate_data):
        canvas = debate_to_ideas_canvas(sample_debate_data)
        extractor = GoalExtractor()
        goal_graph = extractor.extract_from_ideas(canvas.to_dict())
        assert len(goal_graph.provenance) > 0
        for link in goal_graph.provenance:
            assert link.source_stage == PipelineStage.IDEAS
            assert link.target_stage == PipelineStage.GOALS

    def test_goals_have_transition(self, sample_debate_data):
        canvas = debate_to_ideas_canvas(sample_debate_data)
        extractor = GoalExtractor()
        goal_graph = extractor.extract_from_ideas(canvas.to_dict())
        assert goal_graph.transition is not None
        assert goal_graph.transition.from_stage == PipelineStage.IDEAS
        assert goal_graph.transition.to_stage == PipelineStage.GOALS

    def test_priority_from_support(self):
        """Nodes with more support should get higher priority goals."""
        canvas_data = {
            "nodes": [
                {"id": "well-supported", "label": "Popular idea", "data": {"idea_type": "concept"}},
                {"id": "s1", "label": "Support 1", "data": {"idea_type": "evidence"}},
                {"id": "s2", "label": "Support 2", "data": {"idea_type": "evidence"}},
                {"id": "s3", "label": "Support 3", "data": {"idea_type": "evidence"}},
                {"id": "lonely", "label": "Unsupported idea", "data": {"idea_type": "concept"}},
            ],
            "edges": [
                {"source": "s1", "target": "well-supported", "type": "supports"},
                {"source": "s2", "target": "well-supported", "type": "supports"},
                {"source": "s3", "target": "well-supported", "type": "supports"},
            ],
        }
        extractor = GoalExtractor()
        goal_graph = extractor.extract_from_ideas(canvas_data)
        # The well-supported idea should produce a higher-priority goal
        assert len(goal_graph.goals) >= 1

    def test_extract_from_raw_ideas(self):
        extractor = GoalExtractor()
        goal_graph = extractor.extract_from_raw_ideas([
            "We need better user authentication",
            "The authentication system should support OAuth2",
            "Users want single sign-on capabilities",
            "Performance monitoring is important",
            "We should add metrics to all endpoints",
        ])
        assert len(goal_graph.goals) >= 1

    def test_empty_ideas(self):
        extractor = GoalExtractor()
        goal_graph = extractor.extract_from_ideas({"nodes": [], "edges": []})
        assert len(goal_graph.goals) == 0

    def test_goal_serialization(self):
        goal = GoalNode(
            id="g1",
            title="Test Goal",
            description="A test goal",
            goal_type=GoalNodeType.GOAL,
            priority="high",
        )
        d = goal.to_dict()
        assert d["id"] == "g1"
        assert d["type"] == "goal"
        assert d["priority"] == "high"

    def test_goal_graph_serialization(self):
        graph = GoalGraph(
            id="gg1",
            goals=[GoalNode(id="g1", title="Test", description="Desc")],
        )
        d = graph.to_dict()
        assert d["id"] == "gg1"
        assert len(d["goals"]) == 1


# =============================================================================
# Full Pipeline Tests
# =============================================================================


class TestIdeaToExecutionPipeline:
    """Test the full four-stage pipeline."""

    def test_from_debate_full_pipeline(self, sample_debate_data):
        pipeline = IdeaToExecutionPipeline()
        result = pipeline.from_debate(sample_debate_data)

        assert isinstance(result, PipelineResult)
        assert result.ideas_canvas is not None
        assert result.goal_graph is not None
        assert result.actions_canvas is not None
        assert result.orchestration_canvas is not None

    def test_all_stages_complete(self, sample_debate_data):
        pipeline = IdeaToExecutionPipeline()
        result = pipeline.from_debate(sample_debate_data)

        for stage in PipelineStage:
            assert result.stage_status[stage.value] == "complete"

    def test_provenance_chain(self, sample_debate_data):
        pipeline = IdeaToExecutionPipeline()
        result = pipeline.from_debate(sample_debate_data)

        assert len(result.provenance) > 0
        # Provenance should span stages
        stages = {link.source_stage for link in result.provenance}
        assert PipelineStage.IDEAS in stages

    def test_transitions_recorded(self, sample_debate_data):
        pipeline = IdeaToExecutionPipeline()
        result = pipeline.from_debate(sample_debate_data)

        assert len(result.transitions) >= 2  # ideas→goals, goals→actions, actions→orch

    def test_integrity_hash(self, sample_debate_data):
        pipeline = IdeaToExecutionPipeline()
        result = pipeline.from_debate(sample_debate_data)

        result_dict = result.to_dict()
        assert "integrity_hash" in result_dict
        assert len(result_dict["integrity_hash"]) == 16

    def test_from_debate_no_auto_advance(self, sample_debate_data):
        pipeline = IdeaToExecutionPipeline()
        result = pipeline.from_debate(sample_debate_data, auto_advance=False)

        assert result.ideas_canvas is not None
        assert result.goal_graph is None
        assert result.stage_status[PipelineStage.IDEAS.value] == "complete"
        assert result.stage_status[PipelineStage.GOALS.value] == "pending"

    def test_manual_stage_advance(self, sample_debate_data):
        pipeline = IdeaToExecutionPipeline()
        result = pipeline.from_debate(sample_debate_data, auto_advance=False)

        # Manually advance to goals
        result = pipeline.advance_stage(result, PipelineStage.GOALS)
        assert result.goal_graph is not None
        assert result.stage_status[PipelineStage.GOALS.value] == "complete"

    def test_from_ideas_list(self):
        pipeline = IdeaToExecutionPipeline()
        result = pipeline.from_ideas([
            "Build a REST API for user management",
            "Add JWT authentication",
            "Create admin dashboard",
            "Set up CI/CD pipeline",
            "Write integration tests",
            "Deploy to production",
        ])

        assert result.ideas_canvas is not None
        assert result.goal_graph is not None
        assert len(result.goal_graph.goals) >= 1

    def test_serialization_roundtrip(self, sample_debate_data):
        pipeline = IdeaToExecutionPipeline()
        result = pipeline.from_debate(sample_debate_data)

        result_dict = result.to_dict()
        assert result_dict["pipeline_id"].startswith("pipe-")
        assert result_dict["ideas"] is not None
        assert "nodes" in result_dict["ideas"]
        assert "edges" in result_dict["ideas"]

    def test_orchestration_has_agents(self, sample_debate_data):
        pipeline = IdeaToExecutionPipeline()
        result = pipeline.from_debate(sample_debate_data)

        orch_rf = to_react_flow(result.orchestration_canvas)
        agent_nodes = [
            n for n in orch_rf["nodes"]
            if n["data"].get("orch_type") == "agent"
        ]
        assert len(agent_nodes) >= 1


# =============================================================================
# Handler Tests
# =============================================================================


def _parse(hr) -> dict:
    """Parse HandlerResult body as JSON dict."""
    body = hr.body if isinstance(hr.body, bytes) else str(hr.body).encode()
    return json.loads(body)


class TestCanvasPipelineHandler:
    """Test the REST API handler."""

    @pytest.fixture
    def handler(self):
        return CanvasPipelineHandler(ctx={})

    @pytest.mark.asyncio
    async def test_from_debate_endpoint(self, handler, sample_debate_data):
        hr = await handler.handle_from_debate({
            "cartographer_data": sample_debate_data,
        })
        result = _parse(hr)
        assert "pipeline_id" in result
        assert result["stages_completed"] == 4

    @pytest.mark.asyncio
    async def test_from_debate_missing_data(self, handler):
        result = _parse(await handler.handle_from_debate({}))
        assert "error" in result

    @pytest.mark.asyncio
    async def test_from_ideas_endpoint(self, handler):
        result = _parse(await handler.handle_from_ideas({
            "ideas": ["idea one", "idea two", "idea three"],
        }))
        assert "pipeline_id" in result
        assert result["goals_count"] >= 1

    @pytest.mark.asyncio
    async def test_from_ideas_missing_data(self, handler):
        result = _parse(await handler.handle_from_ideas({}))
        assert "error" in result

    @pytest.mark.asyncio
    async def test_convert_debate_endpoint(self, handler, sample_debate_data):
        result = _parse(await handler.handle_convert_debate({
            "cartographer_data": sample_debate_data,
        }))
        assert "nodes" in result
        assert "edges" in result

    @pytest.mark.asyncio
    async def test_convert_workflow_endpoint(self, handler, sample_workflow):
        result = _parse(await handler.handle_convert_workflow({
            "workflow_data": sample_workflow,
        }))
        assert "nodes" in result
        assert len(result["nodes"]) == 3

    @pytest.mark.asyncio
    async def test_handler_can_handle(self, handler):
        assert handler.can_handle("/api/v1/canvas/pipeline/from-debate")
        assert handler.can_handle("/api/canvas/convert/debate")
        assert not handler.can_handle("/api/v1/debates/")

    @pytest.mark.asyncio
    async def test_get_pipeline_not_found(self, handler):
        result = _parse(await handler.handle_get_pipeline("nonexistent"))
        assert "error" in result

    @pytest.mark.asyncio
    async def test_get_stage_not_found(self, handler):
        result = _parse(await handler.handle_get_stage("nonexistent", "ideas"))
        assert "error" in result

    # =========================================================================
    # Advance endpoint tests
    # =========================================================================

    @pytest.mark.asyncio
    async def test_advance_to_goals(self, handler, sample_debate_data):
        """Create pipeline with auto_advance=False, then advance to goals."""
        create_result = _parse(await handler.handle_from_debate({
            "cartographer_data": sample_debate_data,
            "auto_advance": False,
        }))
        pid = create_result["pipeline_id"]

        result = _parse(await handler.handle_advance({
            "pipeline_id": pid,
            "target_stage": "goals",
        }))
        assert result["pipeline_id"] == pid
        assert result["advanced_to"] == "goals"
        assert result["stage_status"]["goals"] == "complete"

    @pytest.mark.asyncio
    async def test_advance_to_actions(self, handler, sample_debate_data):
        """Advance from goals to actions."""
        create_result = _parse(await handler.handle_from_debate({
            "cartographer_data": sample_debate_data,
            "auto_advance": False,
        }))
        pid = create_result["pipeline_id"]

        # First advance to goals
        await handler.handle_advance({
            "pipeline_id": pid,
            "target_stage": "goals",
        })
        # Then advance to actions
        result = _parse(await handler.handle_advance({
            "pipeline_id": pid,
            "target_stage": "actions",
        }))
        assert result["advanced_to"] == "actions"
        assert result["stage_status"]["actions"] == "complete"

    @pytest.mark.asyncio
    async def test_advance_to_orchestration(self, handler, sample_debate_data):
        """Full stage-by-stage advance to orchestration."""
        create_result = _parse(await handler.handle_from_debate({
            "cartographer_data": sample_debate_data,
            "auto_advance": False,
        }))
        pid = create_result["pipeline_id"]

        for stage in ["goals", "actions", "orchestration"]:
            await handler.handle_advance({
                "pipeline_id": pid,
                "target_stage": stage,
            })

        result = _parse(await handler.handle_get_pipeline(pid))
        assert result["stage_status"]["orchestration"] == "complete"

    @pytest.mark.asyncio
    async def test_advance_missing_pipeline_id(self, handler):
        result = _parse(await handler.handle_advance({"target_stage": "goals"}))
        assert "error" in result
        assert "pipeline_id" in result["error"]

    @pytest.mark.asyncio
    async def test_advance_invalid_stage(self, handler, sample_debate_data):
        create_result = _parse(await handler.handle_from_debate({
            "cartographer_data": sample_debate_data,
            "auto_advance": False,
        }))
        pid = create_result["pipeline_id"]
        result = _parse(await handler.handle_advance({
            "pipeline_id": pid,
            "target_stage": "nonexistent_stage",
        }))
        assert "error" in result

    @pytest.mark.asyncio
    async def test_advance_not_found(self, handler):
        result = _parse(await handler.handle_advance({
            "pipeline_id": "pipe-doesnotexist",
            "target_stage": "goals",
        }))
        assert "error" in result
        assert "not found" in result["error"]

    @pytest.mark.asyncio
    async def test_advance_preserves_provenance(self, handler, sample_debate_data):
        """Advancing should maintain provenance chain integrity."""
        create_result = _parse(await handler.handle_from_debate({
            "cartographer_data": sample_debate_data,
            "auto_advance": False,
        }))
        pid = create_result["pipeline_id"]

        result = _parse(await handler.handle_advance({
            "pipeline_id": pid,
            "target_stage": "goals",
        }))
        pipeline_data = result["result"]
        assert pipeline_data["provenance_count"] > 0

    @pytest.mark.asyncio
    async def test_advance_full_sequence(self, handler, sample_debate_data):
        """Advance through all 4 stages sequentially and verify final state."""
        create_result = _parse(await handler.handle_from_debate({
            "cartographer_data": sample_debate_data,
            "auto_advance": False,
        }))
        pid = create_result["pipeline_id"]
        assert create_result["stage_status"]["ideas"] == "complete"
        assert create_result["stage_status"]["goals"] == "pending"

        for stage in ["goals", "actions", "orchestration"]:
            result = _parse(await handler.handle_advance({
                "pipeline_id": pid,
                "target_stage": stage,
            }))
            assert result["advanced_to"] == stage
            assert result["stage_status"][stage] == "complete"

        final = _parse(await handler.handle_get_pipeline(pid))
        for stage_val in ["ideas", "goals", "actions", "orchestration"]:
            assert final["stage_status"][stage_val] == "complete"
        assert final["integrity_hash"]
        assert len(final["integrity_hash"]) == 16


# Import handler for test
from aragora.server.handlers.canvas_pipeline import CanvasPipelineHandler
