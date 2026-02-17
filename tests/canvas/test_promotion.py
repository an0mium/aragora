"""Tests for idea → goal promotion pipeline bridge."""

from __future__ import annotations

import pytest

from aragora.canvas.models import Canvas, CanvasNode, CanvasNodeType, Position, Size
from aragora.canvas.promotion import (
    IDEA_TO_GOAL_TYPE,
    promote_ideas_to_goals,
)
from aragora.canvas.stages import PipelineStage


def _make_idea_node(node_id: str, idea_type: str, label: str = "Test") -> CanvasNode:
    """Helper to create an idea canvas node."""
    return CanvasNode(
        id=node_id,
        node_type=CanvasNodeType.KNOWLEDGE,
        position=Position(x=0, y=0),
        size=Size(220, 80),
        label=label,
        data={
            "idea_type": idea_type,
            "stage": "ideas",
            "body": f"Body of {label}",
        },
    )


def _make_canvas(nodes: list[CanvasNode]) -> Canvas:
    """Create a canvas with the given nodes."""
    c = Canvas(id="test-canvas", name="Test Ideas")
    for n in nodes:
        c.nodes[n.id] = n
    return c


class TestIdeaToGoalMapping:
    """IDEA_TO_GOAL_TYPE coverage."""

    def test_concept_maps_to_goal(self):
        assert IDEA_TO_GOAL_TYPE["concept"] == "goal"

    def test_hypothesis_maps_to_strategy(self):
        assert IDEA_TO_GOAL_TYPE["hypothesis"] == "strategy"

    def test_question_maps_to_risk(self):
        assert IDEA_TO_GOAL_TYPE["question"] == "risk"

    def test_observation_maps_to_milestone(self):
        assert IDEA_TO_GOAL_TYPE["observation"] == "milestone"

    def test_insight_maps_to_metric(self):
        assert IDEA_TO_GOAL_TYPE["insight"] == "metric"

    def test_all_idea_types_covered(self):
        from aragora.canvas.stages import IdeaNodeType
        for t in IdeaNodeType:
            assert t.value in IDEA_TO_GOAL_TYPE, f"{t.value} not in mapping"


class TestPromoteIdeasToGoals:
    """promote_ideas_to_goals() function tests."""

    def test_basic_promotion(self):
        node = _make_idea_node("n1", "concept", "Build widget")
        canvas = _make_canvas([node])

        goals_canvas, provenance = promote_ideas_to_goals(canvas, ["n1"], "user-1")

        assert len(goals_canvas.nodes) == 1
        assert len(provenance) == 1

    def test_provenance_links(self):
        node = _make_idea_node("n1", "concept", "Test idea")
        canvas = _make_canvas([node])

        _, provenance = promote_ideas_to_goals(canvas, ["n1"], "user-1")
        link = provenance[0]

        assert link.source_node_id == "n1"
        assert link.source_stage == PipelineStage.IDEAS
        assert link.target_stage == PipelineStage.GOALS
        assert link.method == "manual_promotion"
        assert len(link.content_hash) == 16

    def test_goal_node_metadata(self):
        node = _make_idea_node("n1", "hypothesis", "Users prefer dark mode")
        canvas = _make_canvas([node])

        goals, _ = promote_ideas_to_goals(canvas, ["n1"], "user-1")
        goal_node = list(goals.nodes.values())[0]

        assert goal_node.data["goal_type"] == "strategy"
        assert goal_node.data["source_idea_type"] == "hypothesis"
        assert goal_node.data["source_node_id"] == "n1"

    def test_source_node_marked(self):
        node = _make_idea_node("n1", "concept", "Test")
        canvas = _make_canvas([node])

        goals, _ = promote_ideas_to_goals(canvas, ["n1"], "user-1")
        goal_id = list(goals.nodes.keys())[0]

        assert canvas.nodes["n1"].data["promoted_to_goal_id"] == goal_id

    def test_multiple_nodes(self):
        nodes = [
            _make_idea_node("n1", "concept", "A"),
            _make_idea_node("n2", "observation", "B"),
            _make_idea_node("n3", "question", "C"),
        ]
        canvas = _make_canvas(nodes)

        goals, provenance = promote_ideas_to_goals(canvas, ["n1", "n2", "n3"], "user-1")

        assert len(goals.nodes) == 3
        assert len(provenance) == 3

    def test_missing_node_skipped(self):
        node = _make_idea_node("n1", "concept", "Test")
        canvas = _make_canvas([node])

        goals, provenance = promote_ideas_to_goals(canvas, ["n1", "missing"], "user-1")

        assert len(goals.nodes) == 1
        assert len(provenance) == 1

    def test_canvas_metadata(self):
        node = _make_idea_node("n1", "concept", "Test")
        canvas = _make_canvas([node])

        goals, _ = promote_ideas_to_goals(canvas, ["n1"], "user-1")

        assert goals.metadata["stage"] == "goals"
        assert goals.metadata["source_canvas_id"] == "test-canvas"
        assert goals.metadata["promoted_by"] == "user-1"

    def test_empty_node_ids(self):
        canvas = _make_canvas([_make_idea_node("n1", "concept", "Test")])
        goals, provenance = promote_ideas_to_goals(canvas, [], "user-1")
        assert len(goals.nodes) == 0
        assert len(provenance) == 0
