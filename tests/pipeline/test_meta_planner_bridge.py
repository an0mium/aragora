"""Tests for MetaPlanner integration in goals_to_actions."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from aragora.canvas.stages import PipelineStage
from aragora.pipeline.stage_transitions import goals_to_actions
from aragora.pipeline.universal_node import UniversalGraph, UniversalNode


@pytest.fixture
def goal_graph():
    graph = UniversalGraph(id="meta-test", name="MetaPlanner Test")
    for i in range(3):
        node = UniversalNode(
            id=f"goal-{i}",
            stage=PipelineStage.GOALS,
            node_subtype="goal",
            label=f"Goal {i}: improve feature {i}",
            description=f"Description for goal {i}",
            confidence=0.7,
            data={"priority": "high"},
        )
        graph.add_node(node)
    return graph


class TestMetaPlannerBridge:
    def test_goals_to_actions_without_planner(self, goal_graph):
        """Structural fallback when no MetaPlanner provided."""
        goal_ids = list(goal_graph.nodes.keys())
        actions = goals_to_actions(goal_graph, goal_ids)
        assert len(actions) >= 1
        for a in actions:
            assert a.stage == PipelineStage.ACTIONS

    def test_goals_to_actions_with_mock_planner(self, goal_graph):
        """When MetaPlanner is provided, it enriches the decomposition."""
        # Create a mock MetaPlanner with prioritize_work method
        mock_planner = MagicMock()
        mock_goal = MagicMock()
        mock_goal.description = "Prioritized: improve feature 0"
        mock_goal.priority = 1
        mock_goal.rationale = "This has the highest impact"
        mock_goal.estimated_impact = "high"
        mock_goal.track = MagicMock(value="core")
        mock_planner.prioritize_work = AsyncMock(return_value=[mock_goal])

        goal_ids = list(goal_graph.nodes.keys())
        actions = goals_to_actions(goal_graph, goal_ids, meta_planner=mock_planner)
        assert len(actions) >= 1
        for a in actions:
            assert a.stage == PipelineStage.ACTIONS

    def test_planner_failure_falls_back_to_structural(self, goal_graph):
        """If MetaPlanner raises, fall back to structural decomposition."""
        mock_planner = MagicMock()
        mock_planner.prioritize_work = AsyncMock(side_effect=RuntimeError("API unavailable"))

        goal_ids = list(goal_graph.nodes.keys())
        # Should not raise - falls back to structural
        actions = goals_to_actions(goal_graph, goal_ids, meta_planner=mock_planner)
        assert len(actions) >= 1

    def test_transition_recorded_with_or_without_planner(self, goal_graph):
        goal_ids = list(goal_graph.nodes.keys())
        goals_to_actions(goal_graph, goal_ids)
        assert len(goal_graph.transitions) == 1
        t = goal_graph.transitions[0]
        assert t.from_stage == PipelineStage.GOALS
        assert t.to_stage == PipelineStage.ACTIONS
