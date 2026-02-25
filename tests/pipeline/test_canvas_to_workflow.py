"""Tests for canvas_to_workflow() — converting PipelineResult goal graphs into WorkflowDefinitions.

Validates that:
- Goal nodes correctly convert to workflow task steps
- Dependencies are preserved as transitions and next_steps
- Empty canvas produces empty workflow
- Canvas with no goals produces empty workflow
- Entry step selection favours root goals (no dependencies)
"""

import pytest

from aragora.canvas.stages import GoalNodeType
from aragora.goals.extractor import GoalGraph, GoalNode
from aragora.pipeline.idea_to_execution import PipelineResult, canvas_to_workflow
from aragora.workflow.types import WorkflowDefinition


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def empty_result() -> PipelineResult:
    """A PipelineResult with no goal graph."""
    return PipelineResult(pipeline_id="pipe-empty")


@pytest.fixture
def result_with_no_goals() -> PipelineResult:
    """A PipelineResult with an empty goal graph."""
    return PipelineResult(
        pipeline_id="pipe-no-goals",
        goal_graph=GoalGraph(id="gg-empty", goals=[]),
    )


@pytest.fixture
def single_goal_result() -> PipelineResult:
    """A PipelineResult with one goal and no dependencies."""
    goal = GoalNode(
        id="goal-1",
        title="Build rate limiter",
        description="Implement token-bucket rate limiting for API endpoints",
        goal_type=GoalNodeType.GOAL,
        priority="high",
        measurable="Reduce 429 errors by 90%",
        dependencies=[],
        source_idea_ids=["idea-1"],
        confidence=0.85,
    )
    return PipelineResult(
        pipeline_id="pipe-single",
        goal_graph=GoalGraph(id="gg-single", goals=[goal]),
    )


@pytest.fixture
def multi_goal_result() -> PipelineResult:
    """A PipelineResult with multiple goals and dependency edges."""
    goals = [
        GoalNode(
            id="goal-a",
            title="Design API schema",
            description="Define OpenAPI schema for all endpoints",
            goal_type=GoalNodeType.GOAL,
            priority="high",
            dependencies=[],
            source_idea_ids=["idea-1"],
        ),
        GoalNode(
            id="goal-b",
            title="Implement caching layer",
            description="Add Redis caching for frequently accessed data",
            goal_type=GoalNodeType.GOAL,
            priority="medium",
            dependencies=["goal-a"],
            source_idea_ids=["idea-2"],
        ),
        GoalNode(
            id="goal-c",
            title="Add monitoring",
            description="Set up performance monitoring dashboards",
            goal_type=GoalNodeType.GOAL,
            priority="medium",
            dependencies=["goal-a"],
            source_idea_ids=["idea-3"],
        ),
        GoalNode(
            id="goal-d",
            title="Load test",
            description="Run load tests against cached endpoints",
            goal_type=GoalNodeType.GOAL,
            priority="low",
            dependencies=["goal-b", "goal-c"],
            source_idea_ids=["idea-4"],
        ),
    ]
    return PipelineResult(
        pipeline_id="pipe-multi",
        goal_graph=GoalGraph(id="gg-multi", goals=goals),
    )


# =============================================================================
# Tests
# =============================================================================


class TestCanvasToWorkflowBasic:
    """Basic conversion tests."""

    def test_empty_canvas_produces_empty_workflow(self, empty_result: PipelineResult) -> None:
        wf = canvas_to_workflow(empty_result)
        assert isinstance(wf, WorkflowDefinition)
        assert wf.steps == []
        assert wf.transitions == []
        assert wf.id == "wf-pipe-empty"
        assert wf.metadata["source_pipeline_id"] == "pipe-empty"

    def test_no_goals_produces_empty_workflow(self, result_with_no_goals: PipelineResult) -> None:
        wf = canvas_to_workflow(result_with_no_goals)
        assert isinstance(wf, WorkflowDefinition)
        assert wf.steps == []
        assert wf.transitions == []

    def test_single_goal_converts_to_single_step(self, single_goal_result: PipelineResult) -> None:
        wf = canvas_to_workflow(single_goal_result)
        assert len(wf.steps) == 1

        step = wf.steps[0]
        assert step.id == "goal-1"
        assert step.name == "Build rate limiter"
        assert step.step_type == "task"
        assert step.description == "Implement token-bucket rate limiting for API endpoints"
        assert step.config["priority"] == "high"
        assert step.config["measurable"] == "Reduce 429 errors by 90%"
        assert step.config["goal_type"] == "goal"
        assert step.config["source_idea_ids"] == ["idea-1"]

    def test_single_goal_has_no_transitions(self, single_goal_result: PipelineResult) -> None:
        wf = canvas_to_workflow(single_goal_result)
        assert wf.transitions == []

    def test_single_goal_is_entry_step(self, single_goal_result: PipelineResult) -> None:
        wf = canvas_to_workflow(single_goal_result)
        assert wf.entry_step == "goal-1"


class TestCanvasToWorkflowDependencies:
    """Tests for dependency preservation."""

    def test_multi_goal_step_count(self, multi_goal_result: PipelineResult) -> None:
        wf = canvas_to_workflow(multi_goal_result)
        assert len(wf.steps) == 4

    def test_dependencies_become_transitions(self, multi_goal_result: PipelineResult) -> None:
        wf = canvas_to_workflow(multi_goal_result)
        # goal-b depends on goal-a => transition a -> b
        # goal-c depends on goal-a => transition a -> c
        # goal-d depends on goal-b and goal-c => transitions b -> d, c -> d
        assert len(wf.transitions) == 4

        tr_pairs = {(t.from_step, t.to_step) for t in wf.transitions}
        assert ("goal-a", "goal-b") in tr_pairs
        assert ("goal-a", "goal-c") in tr_pairs
        assert ("goal-b", "goal-d") in tr_pairs
        assert ("goal-c", "goal-d") in tr_pairs

    def test_next_steps_wired_on_source(self, multi_goal_result: PipelineResult) -> None:
        wf = canvas_to_workflow(multi_goal_result)
        step_map = {s.id: s for s in wf.steps}

        assert "goal-b" in step_map["goal-a"].next_steps
        assert "goal-c" in step_map["goal-a"].next_steps
        assert "goal-d" in step_map["goal-b"].next_steps
        assert "goal-d" in step_map["goal-c"].next_steps
        # goal-d is a leaf — no next_steps
        assert step_map["goal-d"].next_steps == []

    def test_root_goal_is_entry_step(self, multi_goal_result: PipelineResult) -> None:
        wf = canvas_to_workflow(multi_goal_result)
        # goal-a has no dependencies, so it is the entry step
        assert wf.entry_step == "goal-a"

    def test_transition_conditions_are_true(self, multi_goal_result: PipelineResult) -> None:
        wf = canvas_to_workflow(multi_goal_result)
        for tr in wf.transitions:
            assert tr.condition == "True"


class TestCanvasToWorkflowEdgeCases:
    """Edge case tests."""

    def test_dangling_dependency_is_filtered(self) -> None:
        """Dependencies referencing non-existent goals are ignored."""
        goal = GoalNode(
            id="goal-x",
            title="Standalone goal",
            description="Has a dep that does not exist",
            dependencies=["goal-nonexistent"],
        )
        result = PipelineResult(
            pipeline_id="pipe-dangling",
            goal_graph=GoalGraph(id="gg-dangling", goals=[goal]),
        )
        wf = canvas_to_workflow(result)
        assert len(wf.steps) == 1
        assert wf.transitions == []
        assert wf.steps[0].next_steps == []

    def test_workflow_id_includes_pipeline_id(self, single_goal_result: PipelineResult) -> None:
        wf = canvas_to_workflow(single_goal_result)
        assert "pipe-single" in wf.id

    def test_workflow_metadata_has_source_pipeline(self, multi_goal_result: PipelineResult) -> None:
        wf = canvas_to_workflow(multi_goal_result)
        assert wf.metadata["source_pipeline_id"] == "pipe-multi"

    def test_workflow_to_dict_roundtrip(self, multi_goal_result: PipelineResult) -> None:
        """WorkflowDefinition produced can serialize to dict."""
        wf = canvas_to_workflow(multi_goal_result)
        d = wf.to_dict()
        assert d["id"] == wf.id
        assert len(d["steps"]) == 4
        assert len(d["transitions"]) == 4
        assert d["entry_step"] == "goal-a"

    def test_multiple_roots_picks_first(self) -> None:
        """When multiple goals have no dependencies, entry_step is the first."""
        goals = [
            GoalNode(id="root-1", title="First root", description="", dependencies=[]),
            GoalNode(id="root-2", title="Second root", description="", dependencies=[]),
        ]
        result = PipelineResult(
            pipeline_id="pipe-multi-root",
            goal_graph=GoalGraph(id="gg-roots", goals=goals),
        )
        wf = canvas_to_workflow(result)
        assert wf.entry_step == "root-1"
