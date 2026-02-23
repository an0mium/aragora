"""Tests for the demo pipeline flow.

Verifies that demo data progresses correctly through all 4 stages.
"""

from __future__ import annotations

import pytest

from aragora.canvas.stages import PipelineStage
from aragora.pipeline.stage_transitions import (
    actions_to_orchestration,
    goals_to_actions,
    ideas_to_goals,
)
from aragora.pipeline.universal_node import UniversalGraph, UniversalNode


@pytest.fixture
def demo_graph():
    graph = UniversalGraph(id="demo-pipeline", name="Demo Pipeline")
    ideas = [
        ("idea-1", "Build a rate limiter", "concept"),
        ("idea-2", "Add caching layer", "concept"),
        ("idea-3", "Improve API documentation", "insight"),
        ("idea-4", "Performance monitoring", "question"),
    ]
    for nid, label, subtype in ideas:
        node = UniversalNode(
            id=nid,
            stage=PipelineStage.IDEAS,
            node_subtype=subtype,
            label=label,
            description=f"Full description of: {label}",
        )
        graph.add_node(node)
    return graph


class TestDemoPipelineFlow:
    def test_demo_graph_has_all_ideas(self, demo_graph):
        idea_nodes = [n for n in demo_graph.nodes.values() if n.stage == PipelineStage.IDEAS]
        assert len(idea_nodes) == 4

    def test_ideas_to_goals_structural(self, demo_graph):
        idea_ids = list(demo_graph.nodes.keys())
        goals = ideas_to_goals(demo_graph, idea_ids)
        assert len(goals) >= 1
        for g in goals:
            assert g.stage == PipelineStage.GOALS
            assert g.parent_ids

    def test_goals_to_actions(self, demo_graph):
        idea_ids = list(demo_graph.nodes.keys())
        goals = ideas_to_goals(demo_graph, idea_ids)
        goal_ids = [g.id for g in goals]
        actions = goals_to_actions(demo_graph, goal_ids)
        assert len(actions) >= 1
        for a in actions:
            assert a.stage == PipelineStage.ACTIONS

    def test_actions_to_orchestration(self, demo_graph):
        idea_ids = list(demo_graph.nodes.keys())
        goals = ideas_to_goals(demo_graph, idea_ids)
        goal_ids = [g.id for g in goals]
        actions = goals_to_actions(demo_graph, goal_ids)
        action_ids = [a.id for a in actions]
        orch = actions_to_orchestration(demo_graph, action_ids)
        assert len(orch) >= 1
        for o in orch:
            assert o.stage == PipelineStage.ORCHESTRATION

    def test_full_pipeline_creates_transitions(self, demo_graph):
        idea_ids = list(demo_graph.nodes.keys())
        goals = ideas_to_goals(demo_graph, idea_ids)
        goal_ids = [g.id for g in goals]
        actions = goals_to_actions(demo_graph, goal_ids)
        action_ids = [a.id for a in actions]
        actions_to_orchestration(demo_graph, action_ids)
        assert len(demo_graph.transitions) == 3
        stages = [(t.from_stage, t.to_stage) for t in demo_graph.transitions]
        assert (PipelineStage.IDEAS, PipelineStage.GOALS) in stages
        assert (PipelineStage.GOALS, PipelineStage.ACTIONS) in stages
        assert (PipelineStage.ACTIONS, PipelineStage.ORCHESTRATION) in stages

    def test_provenance_chain_links_all_stages(self, demo_graph):
        idea_ids = list(demo_graph.nodes.keys())
        goals = ideas_to_goals(demo_graph, idea_ids)
        goal_ids = [g.id for g in goals]
        actions = goals_to_actions(demo_graph, goal_ids)
        action_ids = [a.id for a in actions]
        actions_to_orchestration(demo_graph, action_ids)
        for node in demo_graph.nodes.values():
            if node.stage != PipelineStage.IDEAS:
                assert node.parent_ids, f"Node {node.id} ({node.stage}) has no parent_ids"

    def test_stage_status_progression(self):
        """Simulate frontend demo stage progression logic."""
        STAGE_ORDER = ["ideas", "goals", "actions", "orchestration"]
        stage_status = dict.fromkeys(STAGE_ORDER, "pending")
        stage_status["ideas"] = "complete"

        # Advance to goals
        target_idx = STAGE_ORDER.index("goals")
        for i in range(target_idx + 1):
            stage_status[STAGE_ORDER[i]] = "complete"
        assert stage_status["goals"] == "complete"
        assert stage_status["actions"] == "pending"

        # Advance to orchestration
        target_idx = STAGE_ORDER.index("orchestration")
        for i in range(target_idx + 1):
            stage_status[STAGE_ORDER[i]] = "complete"
        assert all(s == "complete" for s in stage_status.values())
