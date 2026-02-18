"""Tests for bidirectional adapters between existing models and UniversalNode."""

from __future__ import annotations

import uuid

import pytest

from aragora.canvas.models import (
    Canvas,
    CanvasEdge,
    CanvasNode,
    CanvasNodeType,
    EdgeType,
    Position,
    Size,
)
from aragora.canvas.stages import (
    GoalNodeType,
    PipelineStage,
    StageEdgeType,
    content_hash,
)
from aragora.goals.extractor import GoalNode
from aragora.pipeline.adapters import (
    canvas_to_universal_graph,
    from_argument_node,
    from_canvas_node,
    from_goal_node,
    to_canvas_node,
    to_goal_node,
    universal_graph_to_canvas,
)
from aragora.pipeline.universal_node import (
    UniversalEdge,
    UniversalGraph,
    UniversalNode,
)


class TestCanvasNodeAdapter:
    def test_from_canvas_node(self):
        cn = CanvasNode(
            id="cn-1",
            node_type=CanvasNodeType.KNOWLEDGE,
            position=Position(x=100, y=200),
            size=Size(width=250, height=150),
            label="Test Concept",
            data={"idea_type": "insight", "full_content": "A deep insight", "confidence": 0.85},
            style={"borderRadius": "8px"},
        )
        un = from_canvas_node(cn, PipelineStage.IDEAS)
        assert un.id == "cn-1"
        assert un.stage == PipelineStage.IDEAS
        assert un.node_subtype == "insight"
        assert un.label == "Test Concept"
        assert un.description == "A deep insight"
        assert un.position_x == 100
        assert un.position_y == 200
        assert un.width == 250
        assert un.height == 150
        assert un.confidence == 0.85
        assert un.style == {"borderRadius": "8px"}

    def test_to_canvas_node(self):
        un = UniversalNode(
            id="un-1", stage=PipelineStage.IDEAS,
            node_subtype="concept", label="Restored",
            description="Full description",
            position_x=50, position_y=75,
            width=300, height=120,
            confidence=0.6,
            data={"extra": "data"},
            style={"border": "1px"},
            metadata={"canvas_node_type": "knowledge"},
        )
        cn = to_canvas_node(un)
        assert cn.id == "un-1"
        assert cn.node_type == CanvasNodeType.KNOWLEDGE
        assert cn.label == "Restored"
        assert cn.position.x == 50
        assert cn.position.y == 75
        assert cn.size.width == 300
        assert cn.data["idea_type"] == "concept"
        assert cn.data["full_content"] == "Full description"

    def test_canvas_node_roundtrip(self):
        cn = CanvasNode(
            id="rt-1",
            node_type=CanvasNodeType.EVIDENCE,
            position=Position(x=10, y=20),
            label="Evidence Item",
            data={"idea_type": "evidence", "full_content": "Supporting data"},
        )
        un = from_canvas_node(cn, PipelineStage.IDEAS)
        cn2 = to_canvas_node(un)
        assert cn2.id == cn.id
        assert cn2.label == cn.label
        assert cn2.position.x == cn.position.x
        assert cn2.position.y == cn.position.y

    def test_default_subtype(self):
        cn = CanvasNode(
            id="d1", node_type=CanvasNodeType.TEXT,
            label="Plain text",
        )
        un = from_canvas_node(cn, PipelineStage.IDEAS)
        assert un.node_subtype == "concept"  # default


class TestGoalNodeAdapter:
    def test_from_goal_node(self):
        gn = GoalNode(
            id="g-1",
            title="Build REST API",
            description="Create a full REST API",
            goal_type=GoalNodeType.GOAL,
            priority="high",
            measurable="100% endpoint coverage",
            source_idea_ids=["idea-1", "idea-2"],
            confidence=0.85,
            dependencies=["g-0"],
            metadata={"rank": 1},
        )
        un = from_goal_node(gn)
        assert un.id == "g-1"
        assert un.stage == PipelineStage.GOALS
        assert un.node_subtype == "goal"
        assert un.label == "Build REST API"
        assert un.description == "Create a full REST API"
        assert un.parent_ids == ["idea-1", "idea-2"]
        assert un.source_stage == PipelineStage.IDEAS
        assert un.confidence == 0.85
        assert un.data["priority"] == "high"
        assert un.data["measurable"] == "100% endpoint coverage"
        assert un.metadata["dependencies"] == ["g-0"]

    def test_to_goal_node(self):
        un = UniversalNode(
            id="ug-1", stage=PipelineStage.GOALS,
            node_subtype="strategy", label="My Strategy",
            description="Strategy desc",
            parent_ids=["src-1"],
            confidence=0.7,
            data={"priority": "medium", "measurable": "KPI > 90"},
            metadata={"dependencies": ["dep-1"], "rank": 2},
        )
        gn = to_goal_node(un)
        assert gn.id == "ug-1"
        assert gn.title == "My Strategy"
        assert gn.goal_type == GoalNodeType.STRATEGY
        assert gn.priority == "medium"
        assert gn.measurable == "KPI > 90"
        assert gn.source_idea_ids == ["src-1"]
        assert gn.confidence == 0.7
        assert gn.dependencies == ["dep-1"]

    def test_goal_node_roundtrip(self):
        gn = GoalNode(
            id="rt-g", title="Test Goal",
            description="Test desc",
            goal_type=GoalNodeType.MILESTONE,
            priority="critical",
            source_idea_ids=["i1", "i2"],
            confidence=0.9,
        )
        un = from_goal_node(gn)
        gn2 = to_goal_node(un)
        assert gn2.id == gn.id
        assert gn2.title == gn.title
        assert gn2.description == gn.description
        assert gn2.goal_type == gn.goal_type
        assert gn2.priority == gn.priority
        assert gn2.source_idea_ids == gn.source_idea_ids
        assert gn2.confidence == gn.confidence

    def test_invalid_subtype_defaults_to_goal(self):
        un = UniversalNode(
            id="bad-type", stage=PipelineStage.GOALS,
            node_subtype="unknown_type", label="Test",
        )
        gn = to_goal_node(un)
        assert gn.goal_type == GoalNodeType.GOAL

    def test_no_source_ideas_no_source_stage(self):
        gn = GoalNode(
            id="no-src", title="Orphan Goal",
            description="No sources",
            source_idea_ids=[],
        )
        un = from_goal_node(gn)
        assert un.source_stage is None
        assert un.parent_ids == []


class TestArgumentNodeAdapter:
    def test_from_argument_node(self):
        arg = {
            "id": "arg-1",
            "label": "Key Argument",
            "node_type": "insight",
            "description": "An important point",
            "weight": 0.75,
            "extra_field": "preserved",
        }
        un = from_argument_node(arg)
        assert un.id == "arg-1"
        assert un.stage == PipelineStage.IDEAS
        assert un.node_subtype == "insight"
        assert un.label == "Key Argument"
        assert un.confidence == 0.75
        assert un.data.get("extra_field") == "preserved"

    def test_from_argument_node_custom_stage(self):
        arg = {"id": "a1", "content": "Test content", "type": "concept"}
        un = from_argument_node(arg, stage=PipelineStage.GOALS)
        assert un.stage == PipelineStage.GOALS
        assert un.label == "Test content"

    def test_unknown_type_defaults_concept(self):
        arg = {"id": "a1", "label": "X", "node_type": "weird_type"}
        un = from_argument_node(arg)
        assert un.node_subtype == "concept"


class TestBulkConversions:
    def test_canvas_to_universal_graph(self):
        canvas = Canvas(id="c1", name="Test Canvas", owner_id="user-1")
        n1 = CanvasNode(
            id="cn1", node_type=CanvasNodeType.KNOWLEDGE,
            position=Position(x=0, y=0), label="Node 1",
            data={"idea_type": "concept"},
        )
        n2 = CanvasNode(
            id="cn2", node_type=CanvasNodeType.KNOWLEDGE,
            position=Position(x=100, y=100), label="Node 2",
            data={"idea_type": "insight"},
        )
        canvas.nodes["cn1"] = n1
        canvas.nodes["cn2"] = n2
        edge = CanvasEdge(
            id="ce1", source_id="cn1", target_id="cn2",
            edge_type=EdgeType.SUPPORT, label="supports",
        )
        canvas.edges["ce1"] = edge

        ug = canvas_to_universal_graph(canvas, PipelineStage.IDEAS)
        assert ug.id == "c1"
        assert ug.name == "Test Canvas"
        assert len(ug.nodes) == 2
        assert len(ug.edges) == 1
        assert ug.nodes["cn1"].stage == PipelineStage.IDEAS
        assert ug.nodes["cn2"].node_subtype == "insight"
        assert ug.edges["ce1"].edge_type == StageEdgeType.SUPPORTS

    def test_universal_graph_to_canvas(self):
        graph = UniversalGraph(id="ug1", name="UG Test")
        n1 = UniversalNode(
            id="un1", stage=PipelineStage.IDEAS,
            node_subtype="concept", label="Node A",
            position_x=10, position_y=20,
        )
        n2 = UniversalNode(
            id="un2", stage=PipelineStage.IDEAS,
            node_subtype="evidence", label="Node B",
        )
        n3 = UniversalNode(
            id="un3", stage=PipelineStage.GOALS,
            node_subtype="goal", label="Goal C",
        )
        graph.add_node(n1)
        graph.add_node(n2)
        graph.add_node(n3)
        edge = UniversalEdge(
            id="ue1", source_id="un1", target_id="un2",
            edge_type=StageEdgeType.SUPPORTS,
        )
        graph.add_edge(edge)

        canvas = universal_graph_to_canvas(graph, PipelineStage.IDEAS)
        assert len(canvas.nodes) == 2  # only IDEAS stage
        assert "un1" in canvas.nodes
        assert "un2" in canvas.nodes
        assert "un3" not in canvas.nodes  # GOALS filtered out
        assert len(canvas.edges) == 1

    def test_canvas_roundtrip(self):
        canvas = Canvas(id="rt-c", name="Roundtrip")
        n = CanvasNode(
            id="rn1", node_type=CanvasNodeType.KNOWLEDGE,
            position=Position(x=50, y=75),
            label="Roundtrip Node",
            data={"idea_type": "concept", "extra": "data"},
        )
        canvas.nodes["rn1"] = n

        ug = canvas_to_universal_graph(canvas, PipelineStage.IDEAS)
        canvas2 = universal_graph_to_canvas(ug, PipelineStage.IDEAS)

        assert "rn1" in canvas2.nodes
        assert canvas2.nodes["rn1"].label == "Roundtrip Node"
        assert canvas2.nodes["rn1"].position.x == 50

    def test_canvas_conversion_preserves_edge_types(self):
        canvas = Canvas(id="et-c")
        n1 = CanvasNode(id="n1", node_type=CanvasNodeType.TEXT, label="A")
        n2 = CanvasNode(id="n2", node_type=CanvasNodeType.TEXT, label="B")
        canvas.nodes["n1"] = n1
        canvas.nodes["n2"] = n2

        for etype in [EdgeType.DEPENDENCY, EdgeType.CRITIQUE, EdgeType.SUPPORT]:
            edge = CanvasEdge(
                id=f"e-{etype.value}", source_id="n1", target_id="n2",
                edge_type=etype,
            )
            canvas.edges[edge.id] = edge

        ug = canvas_to_universal_graph(canvas, PipelineStage.IDEAS)
        assert len(ug.edges) == 3

    def test_empty_canvas_conversion(self):
        canvas = Canvas(id="empty")
        ug = canvas_to_universal_graph(canvas, PipelineStage.IDEAS)
        assert len(ug.nodes) == 0
        assert len(ug.edges) == 0

        canvas2 = universal_graph_to_canvas(ug, PipelineStage.IDEAS)
        assert len(canvas2.nodes) == 0
