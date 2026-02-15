"""Tests for Planner/Worker/Judge hierarchy in AutonomousOrchestrator.

Verifies that:
- HierarchyConfig controls plan approval gates and judge reviews
- Workflow steps include plan_approval and judge_review when enabled
- Agent types are correctly assigned per hierarchy role
- Hierarchy disabled produces standard 3-step workflow
"""

from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from aragora.nomic.autonomous_orchestrator import (
    AutonomousOrchestrator,
    AgentAssignment,
    AgentRouter,
    HierarchyConfig,
    Track,
    TrackConfig,
    DEFAULT_TRACK_CONFIGS,
)
from aragora.nomic.task_decomposer import SubTask


def _make_subtask(
    subtask_id: str = "st-1",
    title: str = "Implement feature",
    description: str = "Add a new feature",
    file_scope: list[str] | None = None,
    complexity: str = "medium",
) -> SubTask:
    return SubTask(
        id=subtask_id,
        title=title,
        description=description,
        file_scope=file_scope or ["aragora/server/handler.py"],
        estimated_complexity=complexity,
    )


def _make_assignment(
    subtask: SubTask | None = None,
    track: Track = Track.DEVELOPER,
    agent_type: str = "claude",
) -> AgentAssignment:
    return AgentAssignment(
        subtask=subtask or _make_subtask(),
        track=track,
        agent_type=agent_type,
    )


class TestHierarchyConfig:
    """Tests for HierarchyConfig dataclass."""

    def test_default_disabled(self):
        """Hierarchy should be disabled by default."""
        config = HierarchyConfig()
        assert config.enabled is False

    def test_defaults(self):
        config = HierarchyConfig()
        assert config.planner_agent == "claude"
        assert config.worker_agents == ["claude", "codex"]
        assert config.judge_agent == "claude"
        assert config.plan_gate_blocking is True
        assert config.final_review_blocking is True
        assert config.max_plan_revisions == 2

    def test_custom_agents(self):
        config = HierarchyConfig(
            enabled=True,
            planner_agent="gemini",
            worker_agents=["codex"],
            judge_agent="claude",
        )
        assert config.planner_agent == "gemini"
        assert config.worker_agents == ["codex"]


class TestWorkflowWithoutHierarchy:
    """Standard workflow when hierarchy is disabled."""

    def test_three_step_workflow(self):
        """Standard workflow has design -> implement -> verify."""
        orchestrator = AutonomousOrchestrator(
            hierarchy=HierarchyConfig(enabled=False),
        )
        assignment = _make_assignment()

        workflow = orchestrator._build_subtask_workflow(assignment)

        step_ids = [s.id for s in workflow.steps]
        assert step_ids == ["design", "implement", "verify"]

    def test_design_uses_assignment_agent(self):
        """Without hierarchy, design uses the assigned agent type."""
        orchestrator = AutonomousOrchestrator(
            hierarchy=HierarchyConfig(enabled=False),
        )
        assignment = _make_assignment(agent_type="gemini")

        workflow = orchestrator._build_subtask_workflow(assignment)

        design_step = workflow.steps[0]
        assert design_step.config["agent_type"] == "gemini"

    def test_verify_has_no_next(self):
        """Verify step should have no next steps."""
        orchestrator = AutonomousOrchestrator(
            hierarchy=HierarchyConfig(enabled=False),
        )
        assignment = _make_assignment()

        workflow = orchestrator._build_subtask_workflow(assignment)

        verify_step = next(s for s in workflow.steps if s.id == "verify")
        assert verify_step.next_steps == []


class TestWorkflowWithHierarchy:
    """Workflow with Planner/Worker/Judge hierarchy enabled."""

    def test_five_step_workflow(self):
        """Hierarchy adds plan_approval and judge_review steps."""
        orchestrator = AutonomousOrchestrator(
            hierarchy=HierarchyConfig(enabled=True),
        )
        assignment = _make_assignment()

        workflow = orchestrator._build_subtask_workflow(assignment)

        step_ids = [s.id for s in workflow.steps]
        assert step_ids == ["design", "plan_approval", "implement", "verify", "judge_review"]

    def test_design_uses_planner_agent(self):
        """Design step should use the planner agent."""
        orchestrator = AutonomousOrchestrator(
            hierarchy=HierarchyConfig(enabled=True, planner_agent="gemini"),
        )
        assignment = _make_assignment()

        workflow = orchestrator._build_subtask_workflow(assignment)

        design_step = workflow.steps[0]
        assert design_step.config["agent_type"] == "gemini"

    def test_plan_approval_uses_judge_agent(self):
        """Plan approval gate should use the judge agent."""
        orchestrator = AutonomousOrchestrator(
            hierarchy=HierarchyConfig(enabled=True, judge_agent="claude"),
        )
        assignment = _make_assignment()

        workflow = orchestrator._build_subtask_workflow(assignment)

        approval_step = next(s for s in workflow.steps if s.id == "plan_approval")
        assert approval_step.config["agent_type"] == "claude"

    def test_plan_approval_is_gate(self):
        """Plan approval should be configured as a gate."""
        orchestrator = AutonomousOrchestrator(
            hierarchy=HierarchyConfig(enabled=True),
        )
        assignment = _make_assignment()

        workflow = orchestrator._build_subtask_workflow(assignment)

        approval_step = next(s for s in workflow.steps if s.id == "plan_approval")
        assert approval_step.config["gate"] is True

    def test_plan_approval_blocking_flag(self):
        """Plan approval should respect blocking configuration."""
        for blocking in [True, False]:
            orchestrator = AutonomousOrchestrator(
                hierarchy=HierarchyConfig(enabled=True, plan_gate_blocking=blocking),
            )
            assignment = _make_assignment()

            workflow = orchestrator._build_subtask_workflow(assignment)

            approval_step = next(s for s in workflow.steps if s.id == "plan_approval")
            assert approval_step.config["blocking"] is blocking

    def test_plan_approval_max_revisions(self):
        """Plan approval should include max revisions from config."""
        orchestrator = AutonomousOrchestrator(
            hierarchy=HierarchyConfig(enabled=True, max_plan_revisions=5),
        )
        assignment = _make_assignment()

        workflow = orchestrator._build_subtask_workflow(assignment)

        approval_step = next(s for s in workflow.steps if s.id == "plan_approval")
        assert approval_step.config["max_revisions"] == 5

    def test_implement_uses_worker_agent(self):
        """Implement step should use a worker agent."""
        orchestrator = AutonomousOrchestrator(
            hierarchy=HierarchyConfig(
                enabled=True,
                worker_agents=["codex", "claude"],
            ),
        )
        assignment = _make_assignment()

        workflow = orchestrator._build_subtask_workflow(assignment)

        implement_step = next(s for s in workflow.steps if s.id == "implement")
        assert implement_step.config["agent_type"] in ["codex", "claude"]

    def test_judge_review_uses_judge_agent(self):
        """Judge review should use the judge agent."""
        orchestrator = AutonomousOrchestrator(
            hierarchy=HierarchyConfig(enabled=True, judge_agent="claude"),
        )
        assignment = _make_assignment()

        workflow = orchestrator._build_subtask_workflow(assignment)

        review_step = next(s for s in workflow.steps if s.id == "judge_review")
        assert review_step.config["agent_type"] == "claude"

    def test_judge_review_is_gate(self):
        """Judge review should be configured as a gate."""
        orchestrator = AutonomousOrchestrator(
            hierarchy=HierarchyConfig(enabled=True),
        )
        assignment = _make_assignment()

        workflow = orchestrator._build_subtask_workflow(assignment)

        review_step = next(s for s in workflow.steps if s.id == "judge_review")
        assert review_step.config["gate"] is True

    def test_judge_review_final_step(self):
        """Judge review should be the terminal step."""
        orchestrator = AutonomousOrchestrator(
            hierarchy=HierarchyConfig(enabled=True),
        )
        assignment = _make_assignment()

        workflow = orchestrator._build_subtask_workflow(assignment)

        review_step = next(s for s in workflow.steps if s.id == "judge_review")
        assert review_step.next_steps == []

    def test_verify_chains_to_judge(self):
        """Verify should chain to judge_review when hierarchy is enabled."""
        orchestrator = AutonomousOrchestrator(
            hierarchy=HierarchyConfig(enabled=True),
        )
        assignment = _make_assignment()

        workflow = orchestrator._build_subtask_workflow(assignment)

        verify_step = next(s for s in workflow.steps if s.id == "verify")
        assert verify_step.next_steps == ["judge_review"]

    def test_design_chains_to_plan_approval(self):
        """Design should chain to plan_approval when hierarchy is enabled."""
        orchestrator = AutonomousOrchestrator(
            hierarchy=HierarchyConfig(enabled=True),
        )
        assignment = _make_assignment()

        workflow = orchestrator._build_subtask_workflow(assignment)

        design_step = workflow.steps[0]
        assert design_step.next_steps == ["plan_approval"]

    def test_final_review_blocking_flag(self):
        """Judge review should respect final_review_blocking."""
        for blocking in [True, False]:
            orchestrator = AutonomousOrchestrator(
                hierarchy=HierarchyConfig(enabled=True, final_review_blocking=blocking),
            )
            assignment = _make_assignment()

            workflow = orchestrator._build_subtask_workflow(assignment)

            review_step = next(s for s in workflow.steps if s.id == "judge_review")
            assert review_step.config["blocking"] is blocking


class TestHierarchyOrchestratorInit:
    """Tests for orchestrator initialization with hierarchy."""

    def test_default_hierarchy_disabled(self):
        """Orchestrator should have hierarchy disabled by default."""
        orchestrator = AutonomousOrchestrator()
        assert orchestrator.hierarchy.enabled is False

    def test_accepts_hierarchy_config(self):
        """Orchestrator should accept and store hierarchy config."""
        config = HierarchyConfig(enabled=True, planner_agent="gemini")
        orchestrator = AutonomousOrchestrator(hierarchy=config)
        assert orchestrator.hierarchy is config
        assert orchestrator.hierarchy.planner_agent == "gemini"

    def test_worker_agent_selection_prefers_track_match(self):
        """Worker agent should prefer agents matching the track config."""
        orchestrator = AutonomousOrchestrator(
            hierarchy=HierarchyConfig(
                enabled=True,
                worker_agents=["codex", "gemini", "claude"],
            ),
        )
        # CORE track prefers "claude"
        assignment = _make_assignment(track=Track.CORE)

        workflow = orchestrator._build_subtask_workflow(assignment)

        implement_step = next(s for s in workflow.steps if s.id == "implement")
        # claude is in CORE track's agent_types and also in worker_agents
        assert implement_step.config["agent_type"] == "claude"


class TestPlanApprovalPrompt:
    """Tests for plan approval prompt content."""

    def test_prompt_contains_task_description(self):
        """Plan approval prompt should include the subtask description."""
        orchestrator = AutonomousOrchestrator(
            hierarchy=HierarchyConfig(enabled=True),
        )
        subtask = _make_subtask(description="Refactor the auth module")
        assignment = _make_assignment(subtask=subtask)

        workflow = orchestrator._build_subtask_workflow(assignment)

        approval_step = next(s for s in workflow.steps if s.id == "plan_approval")
        assert "Refactor the auth module" in approval_step.config["task"]

    def test_prompt_requests_approve_or_reject(self):
        """Plan approval prompt should ask for APPROVE or REJECT."""
        orchestrator = AutonomousOrchestrator(
            hierarchy=HierarchyConfig(enabled=True),
        )
        assignment = _make_assignment()

        workflow = orchestrator._build_subtask_workflow(assignment)

        approval_step = next(s for s in workflow.steps if s.id == "plan_approval")
        assert "APPROVE" in approval_step.config["task"]
        assert "REJECT" in approval_step.config["task"]

    def test_judge_review_prompt_mentions_plan(self):
        """Judge review prompt should mention checking implementation vs plan."""
        orchestrator = AutonomousOrchestrator(
            hierarchy=HierarchyConfig(enabled=True),
        )
        assignment = _make_assignment()

        workflow = orchestrator._build_subtask_workflow(assignment)

        review_step = next(s for s in workflow.steps if s.id == "judge_review")
        assert "approved plan" in review_step.config["task"]
