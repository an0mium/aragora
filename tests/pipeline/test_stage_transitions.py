"""Tests for stage transition functions."""

from __future__ import annotations

import pytest

from aragora.canvas.stages import PipelineStage, StageEdgeType
from aragora.pipeline.stage_transitions import (
    actions_to_orchestration,
    approve_generated_node,
    approve_transition,
    generate_goals_to_actions_questions,
    generate_ideas_to_goals_questions,
    goals_to_actions,
    ideas_to_goals,
    interactive_goals_to_actions,
    interactive_ideas_to_goals,
    merge_generated_nodes,
    promote_node,
    reject_generated_node,
    revise_generated_node,
    split_generated_node,
    submit_approved_actions_to_swarm,
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
        result = promote_node(idea_graph, "idea-0", PipelineStage.GOALS, "goal")
        assert result.stage == PipelineStage.GOALS
        assert result.node_subtype == "goal"
        assert result.parent_ids == ["idea-0"]
        assert result.source_stage == PipelineStage.IDEAS
        assert result.id in idea_graph.nodes

    def test_promote_creates_cross_stage_edge(self, idea_graph):
        result = promote_node(idea_graph, "idea-0", PipelineStage.GOALS, "goal")
        cross_edges = idea_graph.get_cross_stage_edges()
        assert len(cross_edges) == 1
        assert cross_edges[0].source_id == "idea-0"
        assert cross_edges[0].target_id == result.id

    def test_promote_preserves_content(self, idea_graph):
        result = promote_node(idea_graph, "idea-0", PipelineStage.GOALS, "goal")
        source = idea_graph.nodes["idea-0"]
        assert result.description == source.description
        assert result.previous_hash == source.content_hash

    def test_promote_with_custom_label(self, idea_graph):
        result = promote_node(
            idea_graph,
            "idea-0",
            PipelineStage.GOALS,
            "goal",
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

    def test_transition_keeps_review_defaults_and_provenance_bundle(self, idea_graph):
        goals = ideas_to_goals(idea_graph, ["idea-0"])
        transition = idea_graph.transitions[0]

        assert transition.status == "pending"
        assert transition.human_notes == ""
        assert transition.reviewed_at is None
        assert len(transition.provenance) == 1
        assert transition.provenance[0].source_node_id == "idea-0"
        assert transition.provenance[0].target_node_id == goals[0].id

    def test_creates_cross_stage_edges(self, idea_graph):
        ideas_to_goals(idea_graph, ["idea-0"])
        cross = idea_graph.get_cross_stage_edges()
        assert len(cross) == 1
        assert cross[0].edge_type == StageEdgeType.DERIVED_FROM

    def test_skips_non_idea_nodes(self, idea_graph):
        # Add a goal node and try to promote it as an idea
        goal = UniversalNode(
            id="goal-x",
            stage=PipelineStage.GOALS,
            node_subtype="goal",
            label="Already a goal",
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
            id="q1",
            stage=PipelineStage.IDEAS,
            node_subtype="question",
            label="What?",
        )
        constraint = UniversalNode(
            id="c1",
            stage=PipelineStage.IDEAS,
            node_subtype="constraint",
            label="Must be fast",
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
            id="ms-1",
            stage=PipelineStage.GOALS,
            node_subtype="milestone",
            label="Reach milestone",
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
            id="cp-1",
            stage=PipelineStage.ACTIONS,
            node_subtype="checkpoint",
            label="Review checkpoint",
        )
        action_graph.add_node(cp)
        orch = actions_to_orchestration(action_graph, ["cp-1"])
        assert orch[0].node_subtype == "human_gate"


class TestFullPipelinePromotion:
    def test_three_stage_pipeline(self):
        """Test promoting through all three transitions: ideas→goals→actions→orchestration."""
        graph = UniversalGraph(id="full-pipe", name="Full Pipeline")
        idea = UniversalNode(
            id="idea-root",
            stage=PipelineStage.IDEAS,
            node_subtype="concept",
            label="Build a rate limiter",
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
            id="i1",
            stage=PipelineStage.IDEAS,
            node_subtype="concept",
            label="Test chain",
        )
        graph.add_node(idea)

        goals = ideas_to_goals(graph, ["i1"])
        goal = goals[0]
        assert goal.previous_hash == idea.content_hash
        assert goal.content_hash != idea.content_hash

        actions = goals_to_actions(graph, [goal.id])
        action = actions[0]
        assert action.previous_hash == goal.content_hash


class TestInteractiveTransitions:
    def test_generates_targeted_questions_from_idea_gaps(self):
        graph = UniversalGraph(id="interactive-ideas", name="Interactive Ideas")
        graph.add_node(
            UniversalNode(
                id="idea-a",
                stage=PipelineStage.IDEAS,
                node_subtype="concept",
                label="Improve onboarding",
                description="",
                confidence=0.6,
            )
        )
        graph.add_node(
            UniversalNode(
                id="idea-b",
                stage=PipelineStage.IDEAS,
                node_subtype="question",
                label="Which user segment first?",
                description="",
                confidence=0.5,
            )
        )

        questions = generate_ideas_to_goals_questions(graph, ["idea-a", "idea-b"], max_questions=3)

        question_ids = {question.id for question in questions}
        assert "primary_outcome" in question_ids
        assert "non_negotiable_constraints" in question_ids
        assert "success_signal" in question_ids

    def test_interactive_ideas_to_goals_creates_editable_semantic_nodes(self):
        graph = UniversalGraph(id="interactive-goals", name="Interactive Goals")
        graph.add_node(
            UniversalNode(
                id="idea-1",
                stage=PipelineStage.IDEAS,
                node_subtype="concept",
                label="Interactive stage transitions",
                description="Turn vague work into executable graphs",
                confidence=0.7,
            )
        )
        graph.add_node(
            UniversalNode(
                id="idea-2",
                stage=PipelineStage.IDEAS,
                node_subtype="constraint",
                label="Stay inside the workbench",
                description="No context switching to a separate tool",
                confidence=0.8,
            )
        )

        result = interactive_ideas_to_goals(
            graph,
            ["idea-1", "idea-2"],
            answers={
                "primary_outcome": "Produce approved goal and principle DAGs",
                "success_signal": "A reviewer can approve nodes without rewriting them",
            },
        )

        assert len(result.generated_nodes) == 2
        semantic_types = {node.data["semantic_type"] for node in result.generated_nodes}
        assert semantic_types == {"goal", "constraint"}
        for node in result.generated_nodes:
            assert node.approval_status == "pending"
            assert node.data["editable"] is True
            assert node.metadata["generated_by_transition_id"] == result.transition.id

        assert result.transition.generated_node_ids == [node.id for node in result.generated_nodes]
        assert (
            result.transition.answers["primary_outcome"]
            == "Produce approved goal and principle DAGs"
        )
        assert any(edge.edge_type == StageEdgeType.CONSTRAINS for edge in graph.edges.values())

    def test_interactive_goals_to_actions_creates_tasks_specs_and_dependencies(self):
        graph = UniversalGraph(id="interactive-actions", name="Interactive Actions")
        goal = UniversalNode(
            id="goal-1",
            stage=PipelineStage.GOALS,
            node_subtype="goal",
            label="Goal: Ship stage transitions",
            description="Deliver the interactive transition flow",
            confidence=0.75,
            data={"semantic_type": "goal", "priority": "high"},
        )
        constraint = UniversalNode(
            id="goal-2",
            stage=PipelineStage.GOALS,
            node_subtype="principle",
            label="Constraint: Preserve provenance",
            description="Every edit and approval must keep upstream lineage visible",
            confidence=0.8,
            data={"semantic_type": "constraint", "priority": "high"},
        )
        graph.add_node(goal)
        graph.add_node(constraint)
        graph.add_edge(
            UniversalEdge(
                id="goal-edge",
                source_id="goal-2",
                target_id="goal-1",
                edge_type=StageEdgeType.CONSTRAINS,
                label="constrains",
            )
        )

        questions = generate_goals_to_actions_questions(graph, ["goal-1", "goal-2"])
        assert {question.id for question in questions} == {
            "done_definition",
            "execution_order",
            "delivery_constraints",
        }

        result = interactive_goals_to_actions(
            graph,
            ["goal-1", "goal-2"],
            answers={
                "done_definition": "Tasks have explicit acceptance criteria and can be submitted to swarm",
                "delivery_constraints": ["Keep verification fast"],
            },
        )

        assert len(result.generated_nodes) == 2
        semantic_types = {node.data["semantic_type"] for node in result.generated_nodes}
        assert semantic_types == {"task", "spec"}
        task_node = next(
            node for node in result.generated_nodes if node.data["semantic_type"] == "task"
        )
        spec_node = next(
            node for node in result.generated_nodes if node.data["semantic_type"] == "spec"
        )
        assert task_node.data["acceptance_criteria"]
        assert "Keep verification fast" in task_node.data["constraints"]
        assert spec_node.id in task_node.data["dependency_ids"]
        assert result.transition.questions
        assert result.transition.generated_node_ids == [node.id for node in result.generated_nodes]

    def test_inline_revision_split_merge_and_reject_preserve_lineage(self):
        graph = UniversalGraph(id="interactive-edits", name="Interactive Edits")
        graph.add_node(
            UniversalNode(
                id="idea-1",
                stage=PipelineStage.IDEAS,
                node_subtype="concept",
                label="Upgrade vague ideas",
                description="",
                confidence=0.6,
            )
        )
        transition_result = interactive_ideas_to_goals(graph, ["idea-1"])
        generated = transition_result.generated_nodes[0]

        revised = revise_generated_node(
            graph,
            generated.id,
            label="Goal: Upgrade vague ideas into editable DAGs",
            description="Keep questioning and editing in the same workbench.",
            data_updates={"acceptance_signal": "Nodes are editable inline"},
        )
        assert revised.approval_status == "revised"
        assert revised.previous_hash is not None
        assert transition_result.transition.status == "revised"

        split_nodes = split_generated_node(
            graph,
            revised.id,
            splits=[
                {"label": "Goal: Ask clarifying questions"},
                {"label": "Goal: Generate editable nodes"},
            ],
        )
        assert len(split_nodes) == 2
        assert revised.status == "archived"
        assert all(revised.id in node.parent_ids for node in split_nodes)

        merged = merge_generated_nodes(
            graph,
            [node.id for node in split_nodes],
            label="Goal: Interactive transition workbench",
            description="Merged workstream for question asking and node editing",
        )
        assert set(merged.metadata["merged_from"]) == {node.id for node in split_nodes}
        rejected = reject_generated_node(graph, merged.id, reviewer_id="user-7", reason="Too broad")
        assert rejected.approval_status == "rejected"
        assert rejected.metadata["rejection"]["reason"] == "Too broad"

    def test_approvals_and_swarm_submission_update_transition_and_nodes(self, monkeypatch):
        graph = UniversalGraph(id="interactive-submit", name="Interactive Submit")
        goal = UniversalNode(
            id="goal-ship",
            stage=PipelineStage.GOALS,
            node_subtype="goal",
            label="Goal: Ship the workbench",
            description="Land the interactive transition system",
            confidence=0.82,
            data={
                "semantic_type": "goal",
                "allowed_write_scope": ["aragora/pipeline/**"],
                "verification_commands": [
                    "python3 -m pytest tests/pipeline/test_stage_transitions.py -q"
                ],
            },
        )
        principle = UniversalNode(
            id="goal-principle",
            stage=PipelineStage.GOALS,
            node_subtype="principle",
            label="Principle: Keep provenance intact",
            description="Every task should carry its lineage and review state",
            confidence=0.78,
            data={"semantic_type": "principle"},
        )
        graph.add_node(goal)
        graph.add_node(principle)

        action_result = interactive_goals_to_actions(
            graph,
            ["goal-ship", "goal-principle"],
            answers={"done_definition": "Approved nodes can be handed directly to swarm"},
        )

        for node in action_result.generated_nodes:
            approve_generated_node(graph, node.id, approver_id="reviewer-1", notes="Looks good")

        assert action_result.transition.status == "approved"

        approve_transition(
            graph,
            action_result.transition.id,
            approver_id="reviewer-1",
            notes="Ready for execution",
        )
        assert action_result.transition.reviewed_at is not None

        captured: dict[str, object] = {}

        def fake_submit(bundle, **kwargs):
            captured["bundle"] = bundle
            captured["kwargs"] = kwargs
            return {"manifest_id": "tranche-demo", "submission_status": "ready_to_prepare"}

        monkeypatch.setattr("aragora.swarm.tranche_submit.submit_intake_bundle", fake_submit)

        result = submit_approved_actions_to_swarm(graph, repo_root=".")

        assert result["submitted_node_ids"]
        assert captured["bundle"]["candidate_lanes"]
        for node in action_result.generated_nodes:
            assert node.execution_status == "submitted"
            assert node.metadata["swarm_submission"]["manifest_id"] == "tranche-demo"
        assert action_result.transition.submission["manifest_id"] == "tranche-demo"
