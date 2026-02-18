"""Tests for stage transition functions."""

from __future__ import annotations

import pytest

from aragora.canvas.stages import PipelineStage, StageEdgeType
from aragora.pipeline.stage_transitions import (
    actions_to_orchestration,
    goals_to_actions,
    ideas_to_goals,
    promote_node,
)
from aragora.pipeline.universal_node import (
    UniversalEdge,
    UniversalGraph,
    UniversalNode,
)


@pytest.fixture
def idea_graph():
    graph = UniversalGraph(id="g1", name="Test")
    for i in range(3):
        node = UniversalNode(
            id=f"idea-{i}",
            stage=PipelineStage.IDEAS,
            node_subtype="concept",
            label=f"Idea {i}",
            description=f"Description for idea {i}",
            confidence=0.8,
        )
        graph.add_node(node)
    return graph


@pytest.fixture
def goal_graph():
    graph = UniversalGraph(id="g2", name="Goals Test")
    for i in range(2):
        node = UniversalNode(
            id=f"goal-{i}",
            stage=PipelineStage.GOALS,
            node_subtype="goal",
            label=f"Goal {i}",
            description=f"Goal description {i}",
            confidence=0.7,
            data={"priority": "high"},
        )
        graph.add_node(node)
    return graph


@pytest.fixture
def action_graph():
    graph = UniversalGraph(id="g3", name="Actions Test")
    for i in range(2):
        node = UniversalNode(
            id=f"action-{i}",
            stage=PipelineStage.ACTIONS,
            node_subtype="task",
            label=f"Implement feature {i}",
            description=f"Action description {i}",
            confidence=0.6,
            data={"priority": "medium"},
        )
        graph.add_node(node)
    return graph


class TestPromoteNode:
    def test_promote_idea_to_goal(self, idea_graph):
        result = promote_node(
            idea_graph, "idea-0", PipelineStage.GOALS, "goal"
        )
        assert result.stage == PipelineStage.GOALS
        assert result.node_subtype == "goal"
        assert result.parent_ids == ["idea-0"]
        assert result.source_stage == PipelineStage.IDEAS
        assert result.id in idea_graph.nodes

    def test_promote_creates_cross_stage_edge(self, idea_graph):
        result = promote_node(
            idea_graph, "idea-0", PipelineStage.GOALS, "goal"
        )
        cross_edges = idea_graph.get_cross_stage_edges()
        assert len(cross_edges) == 1
        assert cross_edges[0].source_id == "idea-0"
        assert cross_edges[0].target_id == result.id

    def test_promote_preserves_content(self, idea_graph):
        result = promote_node(
            idea_graph, "idea-0", PipelineStage.GOALS, "goal"
        )
        source = idea_graph.nodes["idea-0"]
        assert result.description == source.description
        assert result.previous_hash == source.content_hash

    def test_promote_with_custom_label(self, idea_graph):
        result = promote_node(
            idea_graph, "idea-0", PipelineStage.GOALS, "goal",
            new_label="Custom Goal",
        )
        assert result.label == "Custom Goal"

    def test_promote_nonexistent_raises(self, idea_graph):
        with pytest.raises(ValueError, match="not found"):
            promote_node(idea_graph, "no-node", PipelineStage.GOALS, "goal")


class TestIdeasToGoals:
    def test_promotes_all_ideas(self, idea_graph):
        ids = ["idea-0", "idea-1", "idea-2"]
        goals = ideas_to_goals(idea_graph, ids)
        assert len(goals) == 3
        for g in goals:
            assert g.stage == PipelineStage.GOALS
            assert g.source_stage == PipelineStage.IDEAS

    def test_creates_provenance(self, idea_graph):
        goals = ideas_to_goals(idea_graph, ["idea-0"])
        assert len(goals) == 1
        goal = goals[0]
        assert goal.parent_ids == ["idea-0"]
        assert goal.previous_hash == idea_graph.nodes["idea-0"].content_hash

    def test_records_transition(self, idea_graph):
        ideas_to_goals(idea_graph, ["idea-0", "idea-1"])
        assert len(idea_graph.transitions) == 1
        t = idea_graph.transitions[0]
        assert t.from_stage == PipelineStage.IDEAS
        assert t.to_stage == PipelineStage.GOALS

    def test_creates_cross_stage_edges(self, idea_graph):
        ideas_to_goals(idea_graph, ["idea-0"])
        cross = idea_graph.get_cross_stage_edges()
        assert len(cross) == 1
        assert cross[0].edge_type == StageEdgeType.DERIVED_FROM

    def test_skips_non_idea_nodes(self, idea_graph):
        # Add a goal node and try to promote it as an idea
        goal = UniversalNode(
            id="goal-x", stage=PipelineStage.GOALS,
            node_subtype="goal", label="Already a goal",
        )
        idea_graph.add_node(goal)
        result = ideas_to_goals(idea_graph, ["goal-x"])
        assert len(result) == 0

    def test_skips_nonexistent(self, idea_graph):
        result = ideas_to_goals(idea_graph, ["no-such-id"])
        assert len(result) == 0

    def test_empty_ids(self, idea_graph):
        result = ideas_to_goals(idea_graph, [])
        assert result == []
        assert len(idea_graph.transitions) == 0

    def test_subtype_mapping(self, idea_graph):
        # Add different idea types
        question = UniversalNode(
            id="q1", stage=PipelineStage.IDEAS,
            node_subtype="question", label="What?",
        )
        constraint = UniversalNode(
            id="c1", stage=PipelineStage.IDEAS,
            node_subtype="constraint", label="Must be fast",
        )
        idea_graph.add_node(question)
        idea_graph.add_node(constraint)
        goals = ideas_to_goals(idea_graph, ["q1", "c1"])
        subtypes = {g.node_subtype for g in goals}
        assert "milestone" in subtypes  # question → milestone
        assert "principle" in subtypes  # constraint → principle


class TestGoalsToActions:
    def test_promotes_goals(self, goal_graph):
        ids = ["goal-0", "goal-1"]
        actions = goals_to_actions(goal_graph, ids)
        assert len(actions) == 2
        for a in actions:
            assert a.stage == PipelineStage.ACTIONS
            assert a.source_stage == PipelineStage.GOALS

    def test_action_subtype_mapping(self, goal_graph):
        # Add milestone goal
        ms = UniversalNode(
            id="ms-1", stage=PipelineStage.GOALS,
            node_subtype="milestone", label="Reach milestone",
        )
        goal_graph.add_node(ms)
        actions = goals_to_actions(goal_graph, ["ms-1"])
        assert actions[0].node_subtype == "checkpoint"

    def test_creates_implements_edges(self, goal_graph):
        goals_to_actions(goal_graph, ["goal-0"])
        cross = goal_graph.get_cross_stage_edges()
        assert len(cross) == 1
        assert cross[0].edge_type == StageEdgeType.IMPLEMENTS

    def test_records_transition(self, goal_graph):
        goals_to_actions(goal_graph, ["goal-0"])
        assert len(goal_graph.transitions) == 1
        t = goal_graph.transitions[0]
        assert t.from_stage == PipelineStage.GOALS
        assert t.to_stage == PipelineStage.ACTIONS

    def test_preserves_priority(self, goal_graph):
        actions = goals_to_actions(goal_graph, ["goal-0"])
        assert actions[0].data.get("priority") == "high"

    def test_confidence_decay(self, goal_graph):
        actions = goals_to_actions(goal_graph, ["goal-0"])
        source = goal_graph.nodes["goal-0"]
        assert actions[0].confidence == pytest.approx(source.confidence * 0.9)


class TestActionsToOrchestration:
    def test_promotes_actions(self, action_graph):
        ids = ["action-0", "action-1"]
        orch = actions_to_orchestration(action_graph, ids)
        assert len(orch) == 2
        for o in orch:
            assert o.stage == PipelineStage.ORCHESTRATION
            assert o.source_stage == PipelineStage.ACTIONS

    def test_assigns_agent_type(self, action_graph):
        orch = actions_to_orchestration(action_graph, ["action-0"])
        assert "agent_type" in orch[0].data

    def test_creates_executes_edges(self, action_graph):
        actions_to_orchestration(action_graph, ["action-0"])
        cross = action_graph.get_cross_stage_edges()
        assert len(cross) == 1
        assert cross[0].edge_type == StageEdgeType.EXECUTES

    def test_records_transition(self, action_graph):
        actions_to_orchestration(action_graph, ["action-0"])
        assert len(action_graph.transitions) == 1
        t = action_graph.transitions[0]
        assert t.from_stage == PipelineStage.ACTIONS
        assert t.to_stage == PipelineStage.ORCHESTRATION

    def test_checkpoint_becomes_human_gate(self, action_graph):
        cp = UniversalNode(
            id="cp-1", stage=PipelineStage.ACTIONS,
            node_subtype="checkpoint", label="Review checkpoint",
        )
        action_graph.add_node(cp)
        orch = actions_to_orchestration(action_graph, ["cp-1"])
        assert orch[0].node_subtype == "human_gate"


class TestFullPipelinePromotion:
    def test_three_stage_pipeline(self):
        """Test promoting through all three transitions: ideas→goals→actions→orchestration."""
        graph = UniversalGraph(id="full-pipe", name="Full Pipeline")
        idea = UniversalNode(
            id="idea-root", stage=PipelineStage.IDEAS,
            node_subtype="concept", label="Build a rate limiter",
            description="Implement token bucket rate limiting",
            confidence=0.9,
        )
        graph.add_node(idea)

        # Stage 1 → 2
        goals = ideas_to_goals(graph, ["idea-root"])
        assert len(goals) == 1

        # Stage 2 → 3
        goal_ids = [g.id for g in goals]
        actions = goals_to_actions(graph, goal_ids)
        assert len(actions) == 1

        # Stage 3 → 4
        action_ids = [a.id for a in actions]
        orch = actions_to_orchestration(graph, action_ids)
        assert len(orch) == 1

        # Verify full provenance chain
        chain = graph.get_provenance_chain(orch[0].id)
        assert len(chain) == 4  # orch → action → goal → idea
        stages_in_chain = [n.stage for n in chain]
        assert PipelineStage.ORCHESTRATION in stages_in_chain
        assert PipelineStage.ACTIONS in stages_in_chain
        assert PipelineStage.GOALS in stages_in_chain
        assert PipelineStage.IDEAS in stages_in_chain

        # Verify transitions recorded
        assert len(graph.transitions) == 3

        # Verify integrity hash is stable
        h1 = graph.integrity_hash()
        h2 = graph.integrity_hash()
        assert h1 == h2

    def test_sha256_chain(self):
        """Verify content hash chains through promotions."""
        graph = UniversalGraph(id="hash-chain")
        idea = UniversalNode(
            id="i1", stage=PipelineStage.IDEAS,
            node_subtype="concept", label="Test chain",
        )
        graph.add_node(idea)

        goals = ideas_to_goals(graph, ["i1"])
        goal = goals[0]
        assert goal.previous_hash == idea.content_hash
        assert goal.content_hash != idea.content_hash

        actions = goals_to_actions(graph, [goal.id])
        action = actions[0]
        assert action.previous_hash == goal.content_hash
