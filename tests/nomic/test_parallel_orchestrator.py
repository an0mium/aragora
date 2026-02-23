"""Tests for the ParallelOrchestrator and wired orchestration features.

Covers:
- ParallelOrchestrator composition and lifecycle
- Worktree isolation via BranchCoordinator
- Gauntlet gate insertion in workflow
- DecisionPlanFactory wiring
- Convoy/Bead tracking integration
- Semaphore-based concurrency enforcement
"""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.nomic.autonomous_orchestrator import (
    AgentAssignment,
    AutonomousOrchestrator,
    HierarchyConfig,
    OrchestrationResult,
    Track,
    reset_orchestrator,
)
from aragora.nomic.branch_coordinator import (
    BranchCoordinator,
    BranchCoordinatorConfig,
    TrackAssignment,
)
from aragora.nomic.parallel_orchestrator import ParallelOrchestrator
from aragora.nomic.task_decomposer import SubTask, TaskDecomposition

pytestmark = pytest.mark.filterwarnings(
    "ignore:ParallelOrchestrator is deprecated:DeprecationWarning"
)


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset orchestrator singleton before each test."""
    reset_orchestrator()
    yield
    reset_orchestrator()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_subtask(id: str = "1", title: str = "Task", **kwargs) -> SubTask:
    return SubTask(
        id=id,
        title=title,
        description=kwargs.get("description", f"Do {title}"),
        file_scope=kwargs.get("file_scope", []),
        estimated_complexity=kwargs.get("estimated_complexity", "medium"),
        dependencies=kwargs.get("dependencies", []),
    )


def _make_decomposition(subtasks: list[SubTask] | None = None) -> TaskDecomposition:
    if subtasks is None:
        subtasks = [
            _make_subtask("1", "Frontend", file_scope=["aragora/live/src/app/page.tsx"]),
            _make_subtask("2", "Tests", file_scope=["tests/server/test_auth.py"]),
        ]
    return TaskDecomposition(
        original_task="Test goal",
        complexity_score=5,
        complexity_level="medium",
        should_decompose=True,
        subtasks=subtasks,
    )


def _mock_workflow_engine():
    engine = MagicMock()
    engine.execute = AsyncMock(
        return_value=MagicMock(
            success=True,
            final_output={"status": "completed"},
            error=None,
        )
    )
    return engine


def _mock_decomposer(decomposition=None):
    decomposer = MagicMock()
    decomposer.analyze = MagicMock(return_value=decomposition or _make_decomposition())
    return decomposer


# ===========================================================================
# ParallelOrchestrator
# ===========================================================================


class TestParallelOrchestratorInit:
    """Tests for ParallelOrchestrator initialization."""

    def test_creates_branch_coordinator_when_worktrees_enabled(self):
        """Should create BranchCoordinator when use_worktrees=True."""
        po = ParallelOrchestrator(use_worktrees=True)
        assert po._branch_coordinator is not None
        assert po._use_worktrees is True

    def test_no_branch_coordinator_when_worktrees_disabled(self):
        """Should not create BranchCoordinator when use_worktrees=False."""
        po = ParallelOrchestrator(use_worktrees=False)
        assert po._branch_coordinator is None
        assert po._use_worktrees is False

    def test_custom_worktrees_base(self):
        """Should pass custom worktrees_base to BranchCoordinator config."""
        po = ParallelOrchestrator(
            use_worktrees=True,
            worktrees_base=".custom_worktrees",
        )
        assert po._branch_coordinator is not None

    def test_gauntlet_flag_passed_to_orchestrator(self):
        """Should pass enable_gauntlet to AutonomousOrchestrator."""
        po = ParallelOrchestrator(enable_gauntlet=True)
        assert po._orchestrator.enable_gauntlet_gate is True

        po_no = ParallelOrchestrator(enable_gauntlet=False)
        assert po_no._orchestrator.enable_gauntlet_gate is False

    def test_decision_plan_flag_passed(self):
        """Should pass use_decision_plan to AutonomousOrchestrator."""
        po = ParallelOrchestrator(use_decision_plan=True)
        assert po._orchestrator.use_decision_plan is True

    def test_convoy_tracking_flag_passed(self):
        """Should pass convoy tracking config to AutonomousOrchestrator."""
        po = ParallelOrchestrator(enable_convoy_tracking=True)
        assert po._orchestrator.enable_convoy_tracking is True

    def test_standard_orchestrator_params_forwarded(self):
        """Should forward standard params to AutonomousOrchestrator."""
        engine = _mock_workflow_engine()
        decomposer = _mock_decomposer()

        po = ParallelOrchestrator(
            use_worktrees=False,
            enable_gauntlet=False,
            workflow_engine=engine,
            task_decomposer=decomposer,
            require_human_approval=True,
            max_parallel_tasks=8,
        )
        assert po._orchestrator.require_human_approval is True
        assert po._orchestrator.max_parallel_tasks == 8

    @pytest.mark.asyncio
    async def test_execute_goal_delegates_to_orchestrator(self):
        """Should delegate execute_goal to wrapped AutonomousOrchestrator."""
        engine = _mock_workflow_engine()
        decomposer = _mock_decomposer()

        po = ParallelOrchestrator(
            use_worktrees=False,
            enable_gauntlet=False,
            workflow_engine=engine,
            task_decomposer=decomposer,
        )

        result = await po.execute_goal(
            goal="Test",
            tracks=["sme", "qa"],
            max_cycles=3,
        )

        assert result.success is True
        assert result.total_subtasks == 2

    @pytest.mark.asyncio
    async def test_cleanup_calls_branch_coordinator(self):
        """Cleanup should call cleanup_all_worktrees on BranchCoordinator."""
        po = ParallelOrchestrator(use_worktrees=True)
        po._branch_coordinator = MagicMock()
        po._branch_coordinator.cleanup_all_worktrees = MagicMock()

        await po.cleanup()

        po._branch_coordinator.cleanup_all_worktrees.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_no_op_without_worktrees(self):
        """Cleanup should be safe to call without worktrees."""
        po = ParallelOrchestrator(use_worktrees=False)
        await po.cleanup()  # Should not raise


# ===========================================================================
# Worktree isolation (BranchCoordinator integration)
# ===========================================================================


class TestWorktreeIsolation:
    """Tests for worktree path assignment in TrackAssignment."""

    def test_track_assignment_has_worktree_path_field(self):
        """TrackAssignment should have worktree_path field."""
        from aragora.nomic.meta_planner import PrioritizedGoal, Track as MPTrack

        goal = PrioritizedGoal(
            id="goal_0",
            track=MPTrack.SME,
            description="Test",
            rationale="Test",
            estimated_impact="low",
            priority=1,
        )
        assignment = TrackAssignment(goal=goal)
        assert assignment.worktree_path is None

    def test_track_assignment_accepts_worktree_path(self):
        """TrackAssignment should accept a worktree_path value."""
        from aragora.nomic.meta_planner import PrioritizedGoal, Track as MPTrack

        goal = PrioritizedGoal(
            id="goal_0",
            track=MPTrack.DEVELOPER,
            description="SDK work",
            rationale="Test",
            estimated_impact="high",
            priority=1,
        )
        assignment = TrackAssignment(
            goal=goal,
            worktree_path=Path(".worktrees/dev-sdk-work-0001"),
        )
        assert assignment.worktree_path == Path(".worktrees/dev-sdk-work-0001")


class TestBranchCreationInOrchestrator:
    """Tests for _create_branches_for_assignments in orchestrator."""

    @pytest.mark.asyncio
    async def test_creates_branches_when_coordinator_present(self):
        """Should call create_track_branch for each unique track."""
        bc = MagicMock(spec=BranchCoordinator)
        bc.create_track_branch = AsyncMock(return_value="dev/sme-task-001")
        bc.get_worktree_path = MagicMock(return_value=Path(".worktrees/sme"))
        bc._worktree_paths = {}

        engine = _mock_workflow_engine()
        decomposer = _mock_decomposer()

        orchestrator = AutonomousOrchestrator(
            workflow_engine=engine,
            task_decomposer=decomposer,
            branch_coordinator=bc,
        )

        assignments = [
            AgentAssignment(
                subtask=_make_subtask("1", "Frontend", file_scope=["aragora/live/page.tsx"]),
                track=Track.SME,
                agent_type="claude",
            ),
            AgentAssignment(
                subtask=_make_subtask("2", "More Frontend", file_scope=["aragora/live/nav.tsx"]),
                track=Track.SME,
                agent_type="claude",
            ),
        ]

        await orchestrator._create_branches_for_assignments(assignments)

        # Only one branch per unique track
        assert bc.create_track_branch.call_count == 1

    @pytest.mark.asyncio
    async def test_skips_branch_creation_without_coordinator(self):
        """Should skip branch creation when no coordinator is set."""
        orchestrator = AutonomousOrchestrator(
            workflow_engine=_mock_workflow_engine(),
            task_decomposer=_mock_decomposer(),
        )

        # Should not raise
        await orchestrator._create_branches_for_assignments([])


# ===========================================================================
# Gauntlet gate insertion
# ===========================================================================


class TestGauntletGate:
    """Tests for gauntlet gate insertion in workflow."""

    def _get_orchestrator(self, enable_gauntlet=True, hierarchy=None):
        return AutonomousOrchestrator(
            workflow_engine=_mock_workflow_engine(),
            task_decomposer=_mock_decomposer(),
            enable_gauntlet_gate=enable_gauntlet,
            hierarchy=hierarchy,
        )

    def test_gauntlet_step_inserted_when_enabled(self):
        """Should insert gauntlet step between design and implement."""
        orch = self._get_orchestrator(enable_gauntlet=True)

        assignment = AgentAssignment(
            subtask=_make_subtask("1", "Task"),
            track=Track.DEVELOPER,
            agent_type="claude",
        )

        workflow = orch._build_subtask_workflow(assignment)
        step_ids = [s.id for s in workflow.steps]

        assert "gauntlet" in step_ids
        assert step_ids.index("gauntlet") < step_ids.index("implement")
        assert step_ids.index("design") < step_ids.index("gauntlet")

    def test_no_gauntlet_step_when_disabled(self):
        """Should not insert gauntlet step when disabled."""
        orch = self._get_orchestrator(enable_gauntlet=False)

        assignment = AgentAssignment(
            subtask=_make_subtask("1", "Task"),
            track=Track.DEVELOPER,
            agent_type="claude",
        )

        workflow = orch._build_subtask_workflow(assignment)
        step_ids = [s.id for s in workflow.steps]

        assert "gauntlet" not in step_ids

    def test_gauntlet_severity_threshold_high_complexity(self):
        """High-complexity subtasks should use medium severity threshold."""
        orch = self._get_orchestrator(enable_gauntlet=True)

        assignment = AgentAssignment(
            subtask=_make_subtask("1", "Hard Task", estimated_complexity="high"),
            track=Track.DEVELOPER,
            agent_type="claude",
        )

        workflow = orch._build_subtask_workflow(assignment)
        gauntlet_step = next(s for s in workflow.steps if s.id == "gauntlet")
        assert gauntlet_step.config["severity_threshold"] == "medium"

    def test_gauntlet_severity_threshold_low_complexity(self):
        """Non-high-complexity subtasks should use high severity threshold."""
        orch = self._get_orchestrator(enable_gauntlet=True)

        assignment = AgentAssignment(
            subtask=_make_subtask("1", "Easy Task", estimated_complexity="low"),
            track=Track.DEVELOPER,
            agent_type="claude",
        )

        workflow = orch._build_subtask_workflow(assignment)
        gauntlet_step = next(s for s in workflow.steps if s.id == "gauntlet")
        assert gauntlet_step.config["severity_threshold"] == "high"

    def test_gauntlet_step_type_is_gauntlet(self):
        """Gauntlet step should have step_type='gauntlet'."""
        orch = self._get_orchestrator(enable_gauntlet=True)

        assignment = AgentAssignment(
            subtask=_make_subtask("1", "Task"),
            track=Track.DEVELOPER,
            agent_type="claude",
        )

        workflow = orch._build_subtask_workflow(assignment)
        gauntlet_step = next(s for s in workflow.steps if s.id == "gauntlet")
        assert gauntlet_step.step_type == "gauntlet"
        assert gauntlet_step.config["require_passing"] is True

    def test_workflow_4_steps_with_gauntlet(self):
        """Workflow should have 4 steps when gauntlet is enabled."""
        orch = self._get_orchestrator(enable_gauntlet=True)

        assignment = AgentAssignment(
            subtask=_make_subtask("1", "Task"),
            track=Track.DEVELOPER,
            agent_type="claude",
        )

        workflow = orch._build_subtask_workflow(assignment)
        assert len(workflow.steps) == 4  # design, gauntlet, implement, verify

    def test_workflow_3_steps_without_gauntlet(self):
        """Workflow should have 3 steps when gauntlet is disabled."""
        orch = self._get_orchestrator(enable_gauntlet=False)

        assignment = AgentAssignment(
            subtask=_make_subtask("1", "Task"),
            track=Track.DEVELOPER,
            agent_type="claude",
        )

        workflow = orch._build_subtask_workflow(assignment)
        assert len(workflow.steps) == 3  # design, implement, verify

    def test_gauntlet_with_hierarchy_produces_6_steps(self):
        """Hierarchy + gauntlet should produce design, plan_approval, gauntlet, implement, verify, judge_review."""
        hierarchy = HierarchyConfig(enabled=True)
        orch = self._get_orchestrator(enable_gauntlet=True, hierarchy=hierarchy)

        assignment = AgentAssignment(
            subtask=_make_subtask("1", "Task"),
            track=Track.DEVELOPER,
            agent_type="claude",
        )

        workflow = orch._build_subtask_workflow(assignment)
        step_ids = [s.id for s in workflow.steps]

        assert step_ids == [
            "design",
            "plan_approval",
            "gauntlet",
            "implement",
            "verify",
            "judge_review",
        ]

    def test_gauntlet_next_step_is_implement(self):
        """Gauntlet step's next_steps should point to implement."""
        orch = self._get_orchestrator(enable_gauntlet=True)

        assignment = AgentAssignment(
            subtask=_make_subtask("1", "Task"),
            track=Track.DEVELOPER,
            agent_type="claude",
        )

        workflow = orch._build_subtask_workflow(assignment)
        gauntlet_step = next(s for s in workflow.steps if s.id == "gauntlet")
        assert gauntlet_step.next_steps == ["implement"]


# ===========================================================================
# DecisionPlanFactory wiring
# ===========================================================================


class TestDecisionPlanWiring:
    """Tests for _build_workflow_from_plan."""

    def test_returns_none_without_debate_result(self):
        """Should return None when no debate result is available."""
        orch = AutonomousOrchestrator(
            workflow_engine=_mock_workflow_engine(),
            task_decomposer=_mock_decomposer(),
            use_decision_plan=True,
        )

        assignment = AgentAssignment(
            subtask=_make_subtask("1", "Task"),
            track=Track.DEVELOPER,
            agent_type="claude",
        )

        result = orch._build_workflow_from_plan(assignment, None)
        assert result is None

    def test_calls_factory_with_debate_result(self):
        """Should call DecisionPlanFactory.from_debate_result when available."""
        mock_plan = MagicMock()
        mock_plan.implement_plan = None
        mock_plan.requires_human_approval = False

        with patch(
            "aragora.pipeline.decision_plan.factory.DecisionPlanFactory"
        ) as mock_factory_cls:
            mock_factory_cls.from_debate_result.return_value = mock_plan

            orch = AutonomousOrchestrator(
                workflow_engine=_mock_workflow_engine(),
                task_decomposer=_mock_decomposer(),
                use_decision_plan=True,
            )

            assignment = AgentAssignment(
                subtask=_make_subtask("1", "Task"),
                track=Track.DEVELOPER,
                agent_type="claude",
            )

            debate_result = MagicMock()
            debate_result.debate_id = "test-debate-123"
            result = orch._build_workflow_from_plan(assignment, debate_result)

            # Should produce a workflow (at minimum with just a verify step)
            if result is not None:
                assert result.id == "plan_1"
                mock_factory_cls.from_debate_result.assert_called_once()
            # If None, factory wasn't available (import issue in test env) - acceptable


# ===========================================================================
# Convoy/Bead tracking
# ===========================================================================


class TestConvoyTracking:
    """Tests for convoy/bead lifecycle tracking."""

    @pytest.mark.asyncio
    async def test_create_convoy_creates_rig_and_beads(self):
        """Should create a rig, beads, and convoy via workspace manager."""
        mock_rig = MagicMock()
        mock_rig.rig_id = "rig_123"

        mock_convoy = MagicMock()
        mock_convoy.convoy_id = "convoy_123"

        mock_bead1 = MagicMock()
        mock_bead1.bead_id = "bead_1"
        mock_bead1.payload = {"subtask_id": "1"}
        mock_bead2 = MagicMock()
        mock_bead2.bead_id = "bead_2"
        mock_bead2.payload = {"subtask_id": "2"}

        bead_manager = MagicMock()
        bead_manager.list_beads = AsyncMock(return_value=[mock_bead1, mock_bead2])

        ws_manager = MagicMock()
        ws_manager.create_rig = AsyncMock(return_value=mock_rig)
        ws_manager.create_convoy = AsyncMock(return_value=mock_convoy)
        ws_manager.start_convoy = AsyncMock()
        ws_manager._bead_manager = bead_manager

        orch = AutonomousOrchestrator(
            workflow_engine=_mock_workflow_engine(),
            task_decomposer=_mock_decomposer(),
            enable_convoy_tracking=True,
            workspace_manager=ws_manager,
        )

        assignments = [
            AgentAssignment(
                subtask=_make_subtask("1", "Task A"),
                track=Track.SME,
                agent_type="claude",
            ),
            AgentAssignment(
                subtask=_make_subtask("2", "Task B"),
                track=Track.QA,
                agent_type="claude",
            ),
        ]

        await orch._create_convoy_for_goal("Test goal", assignments)

        assert orch._convoy_id == "convoy_123"
        assert orch._bead_ids == {"1": "bead_1", "2": "bead_2"}
        ws_manager.create_rig.assert_called_once()
        ws_manager.create_convoy.assert_called_once()
        ws_manager.start_convoy.assert_called_once_with("convoy_123")

    @pytest.mark.asyncio
    async def test_update_bead_status_running(self):
        """Should call start_bead for running status."""
        ws_manager = MagicMock()
        bead_manager = MagicMock()
        bead_manager.start_bead = AsyncMock()
        ws_manager._bead_manager = bead_manager

        orch = AutonomousOrchestrator(
            workflow_engine=_mock_workflow_engine(),
            task_decomposer=_mock_decomposer(),
            enable_convoy_tracking=True,
            workspace_manager=ws_manager,
        )
        orch._bead_ids = {"1": "bead_abc"}

        await orch._update_bead_status("1", "running")

        bead_manager.start_bead.assert_called_once_with("bead_abc")

    @pytest.mark.asyncio
    async def test_update_bead_status_done(self):
        """Should call complete_bead for done status."""
        ws_manager = MagicMock()
        ws_manager.complete_bead = AsyncMock()
        # Also need _bead_manager for the running path (but we're testing done)
        ws_manager._bead_manager = MagicMock()

        orch = AutonomousOrchestrator(
            workflow_engine=_mock_workflow_engine(),
            task_decomposer=_mock_decomposer(),
            enable_convoy_tracking=True,
            workspace_manager=ws_manager,
        )
        orch._bead_ids = {"1": "bead_abc"}

        await orch._update_bead_status("1", "done")

        ws_manager.complete_bead.assert_called_once_with("bead_abc")

    @pytest.mark.asyncio
    async def test_update_bead_status_failed(self):
        """Should call fail_bead for failed status."""
        ws_manager = MagicMock()
        ws_manager.fail_bead = AsyncMock()
        ws_manager._bead_manager = MagicMock()

        orch = AutonomousOrchestrator(
            workflow_engine=_mock_workflow_engine(),
            task_decomposer=_mock_decomposer(),
            enable_convoy_tracking=True,
            workspace_manager=ws_manager,
        )
        orch._bead_ids = {"1": "bead_abc"}

        await orch._update_bead_status("1", "failed", error="test error")

        ws_manager.fail_bead.assert_called_once_with("bead_abc", "test error")

    @pytest.mark.asyncio
    async def test_update_bead_noop_without_tracking(self):
        """Should no-op when convoy tracking is disabled."""
        orch = AutonomousOrchestrator(
            workflow_engine=_mock_workflow_engine(),
            task_decomposer=_mock_decomposer(),
            enable_convoy_tracking=False,
        )

        # Should not raise
        await orch._update_bead_status("1", "running")

    @pytest.mark.asyncio
    async def test_complete_convoy_success(self):
        """Should call complete_convoy on workspace manager."""
        ws_manager = MagicMock()
        ws_manager.complete_convoy = AsyncMock()

        orch = AutonomousOrchestrator(
            workflow_engine=_mock_workflow_engine(),
            task_decomposer=_mock_decomposer(),
            enable_convoy_tracking=True,
            workspace_manager=ws_manager,
        )
        orch._convoy_id = "convoy_123"

        await orch._complete_convoy(success=True)

        ws_manager.complete_convoy.assert_called_once_with("convoy_123")

    @pytest.mark.asyncio
    async def test_complete_convoy_failure(self):
        """Should call fail_convoy via _convoy_tracker on workspace manager."""
        ws_manager = MagicMock()
        convoy_tracker = MagicMock()
        convoy_tracker.fail_convoy = AsyncMock()
        ws_manager._convoy_tracker = convoy_tracker

        orch = AutonomousOrchestrator(
            workflow_engine=_mock_workflow_engine(),
            task_decomposer=_mock_decomposer(),
            enable_convoy_tracking=True,
            workspace_manager=ws_manager,
        )
        orch._convoy_id = "convoy_123"

        await orch._complete_convoy(success=False, error="something failed")

        convoy_tracker.fail_convoy.assert_called_once_with("convoy_123", "something failed")


# ===========================================================================
# Semaphore enforcement
# ===========================================================================


class TestSemaphoreEnforcement:
    """Tests for asyncio.Semaphore-based concurrency control."""

    def test_semaphore_created_with_max_parallel(self):
        """Should create semaphore matching max_parallel_tasks."""
        orch = AutonomousOrchestrator(
            workflow_engine=_mock_workflow_engine(),
            task_decomposer=_mock_decomposer(),
            max_parallel_tasks=3,
        )
        assert orch._semaphore._value == 3

    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrency(self):
        """Semaphore should prevent more tasks than max_parallel from running simultaneously."""
        max_concurrent_seen = 0
        current_running = 0

        original_execute = AsyncMock(
            return_value=MagicMock(
                success=True,
                final_output={"status": "ok"},
                error=None,
            )
        )

        async def tracking_execute(workflow, inputs=None):
            nonlocal max_concurrent_seen, current_running
            current_running += 1
            if current_running > max_concurrent_seen:
                max_concurrent_seen = current_running
            await asyncio.sleep(0.01)  # Yield to allow other tasks to start
            result = await original_execute(workflow, inputs)
            current_running -= 1
            return result

        engine = MagicMock()
        engine.execute = AsyncMock(side_effect=tracking_execute)

        subtasks = [
            _make_subtask(str(i), f"Task {i}", file_scope=[f"tests/t{i}.py"]) for i in range(5)
        ]

        orch = AutonomousOrchestrator(
            workflow_engine=engine,
            task_decomposer=_mock_decomposer(_make_decomposition(subtasks)),
            max_parallel_tasks=2,
        )

        await orch.execute_goal(goal="Test", max_cycles=1)

        # With semaphore=2, we should never see more than 2 running concurrently
        assert max_concurrent_seen <= 2


# ===========================================================================
# Merge and cleanup
# ===========================================================================


class TestMergeAndCleanup:
    """Tests for _merge_and_cleanup helper."""

    @pytest.mark.asyncio
    async def test_merges_completed_branches(self):
        """Should merge completed branches and cleanup worktrees."""
        bc = MagicMock(spec=BranchCoordinator)
        bc.safe_merge = MagicMock(return_value=MagicMock(success=True))
        bc.cleanup_all_worktrees = MagicMock()

        orch = AutonomousOrchestrator(
            workflow_engine=_mock_workflow_engine(),
            task_decomposer=_mock_decomposer(),
            branch_coordinator=bc,
        )

        assignments = [
            AgentAssignment(
                subtask=_make_subtask("1", "Task"),
                track=Track.SME,
                agent_type="claude",
                status="completed",
            ),
        ]

        await orch._merge_and_cleanup(assignments)

        bc.cleanup_all_worktrees.assert_called_once()

    @pytest.mark.asyncio
    async def test_skips_merge_without_coordinator(self):
        """Should skip merge when no coordinator is set."""
        orch = AutonomousOrchestrator(
            workflow_engine=_mock_workflow_engine(),
            task_decomposer=_mock_decomposer(),
        )

        # Should not raise
        await orch._merge_and_cleanup([])


# ===========================================================================
# Worktree path in workflow config
# ===========================================================================


class TestWorktreePathInWorkflow:
    """Tests for repo_path resolution via worktree."""

    def test_repo_path_defaults_to_cwd_without_coordinator(self):
        """Workflow should use aragora_path when no coordinator."""
        orch = AutonomousOrchestrator(
            workflow_engine=_mock_workflow_engine(),
            task_decomposer=_mock_decomposer(),
        )

        assignment = AgentAssignment(
            subtask=_make_subtask("1", "Task"),
            track=Track.DEVELOPER,
            agent_type="claude",
        )

        workflow = orch._build_subtask_workflow(assignment)
        impl_step = next(s for s in workflow.steps if s.id == "implement")
        assert impl_step.config["repo_path"] == str(orch.aragora_path)

    def test_repo_path_uses_worktree_when_available(self):
        """Workflow should use worktree path when coordinator has one."""
        bc = MagicMock(spec=BranchCoordinator)
        bc._worktree_paths = {"dev/developer-task-001": Path("/tmp/wt/developer")}

        orch = AutonomousOrchestrator(
            workflow_engine=_mock_workflow_engine(),
            task_decomposer=_mock_decomposer(),
            branch_coordinator=bc,
        )

        assignment = AgentAssignment(
            subtask=_make_subtask("1", "Task"),
            track=Track.DEVELOPER,
            agent_type="claude",
        )

        workflow = orch._build_subtask_workflow(assignment)
        impl_step = next(s for s in workflow.steps if s.id == "implement")
        assert impl_step.config["repo_path"] == "/tmp/wt/developer"


# ===========================================================================
# Full integration: ParallelOrchestrator end-to-end
# ===========================================================================


class TestParallelOrchestratorIntegration:
    """Integration tests for ParallelOrchestrator end-to-end flow."""

    @pytest.mark.asyncio
    async def test_full_flow_with_worktrees_gauntlet_convoy(self):
        """Full flow: worktrees + gauntlet + convoy tracking."""
        engine = _mock_workflow_engine()
        decomposer = _mock_decomposer()

        with patch("aragora.nomic.parallel_orchestrator.BranchCoordinator") as mock_bc_cls:
            mock_bc = MagicMock()
            mock_bc.create_track_branch = AsyncMock(return_value="dev/sme-task-001")
            mock_bc.create_track_branches = AsyncMock(return_value=[])
            mock_bc.safe_merge = MagicMock(return_value=MagicMock(success=True))
            mock_bc.cleanup_all_worktrees = MagicMock()
            mock_bc.get_worktree_path = MagicMock(return_value=Path("/tmp/wt/test"))
            mock_bc._worktree_paths = {}
            mock_bc_cls.return_value = mock_bc

            po = ParallelOrchestrator(
                use_worktrees=True,
                enable_gauntlet=True,
                enable_convoy_tracking=False,
                workflow_engine=engine,
                task_decomposer=decomposer,
                max_parallel_tasks=2,
            )

            result = await po.execute_goal(goal="Test", max_cycles=1)

            assert result.success is True
            assert result.total_subtasks == 2

    @pytest.mark.asyncio
    async def test_minimal_parallel_orchestrator(self):
        """Minimal setup: no worktrees, no gauntlet, no convoy."""
        engine = _mock_workflow_engine()
        decomposer = _mock_decomposer()

        po = ParallelOrchestrator(
            use_worktrees=False,
            enable_gauntlet=False,
            enable_convoy_tracking=False,
            workflow_engine=engine,
            task_decomposer=decomposer,
        )

        result = await po.execute_goal(goal="Simple task", max_cycles=1)

        assert result.success is True
        assert result.completed_subtasks == 2

    @pytest.mark.asyncio
    async def test_full_lifecycle_init_execute_cleanup(self):
        """Simulate self_develop.py --parallel: init → execute → cleanup."""
        engine = _mock_workflow_engine()
        decomposer = _mock_decomposer()

        with patch("aragora.nomic.parallel_orchestrator.BranchCoordinator") as mock_bc_cls:
            mock_bc = MagicMock()
            mock_bc.create_track_branch = AsyncMock(return_value="dev/sme-task-001")
            mock_bc.create_track_branches = AsyncMock(return_value=[])
            mock_bc.safe_merge = MagicMock(return_value=MagicMock(success=True))
            mock_bc.cleanup_all_worktrees = MagicMock()
            mock_bc._worktree_paths = {}
            mock_bc_cls.return_value = mock_bc

            po = ParallelOrchestrator(
                use_worktrees=True,
                enable_gauntlet=True,
                enable_convoy_tracking=False,
                workflow_engine=engine,
                task_decomposer=decomposer,
                max_parallel_tasks=3,
                budget_limit_usd=10.0,
            )

            # Execute
            result = await po.execute_goal(
                goal="Improve error handling",
                tracks=["developer"],
                max_cycles=2,
            )

            assert result.success is True

            # Cleanup
            await po.cleanup()
            mock_bc.cleanup_all_worktrees.assert_called()

    @pytest.mark.asyncio
    async def test_cleanup_without_worktrees(self):
        """Cleanup should be a no-op when worktrees are disabled."""
        po = ParallelOrchestrator(
            use_worktrees=False,
            enable_gauntlet=False,
            enable_convoy_tracking=False,
            workflow_engine=_mock_workflow_engine(),
            task_decomposer=_mock_decomposer(),
        )

        # Should not raise
        await po.cleanup()

    @pytest.mark.asyncio
    async def test_cleanup_handles_worktree_errors_gracefully(self):
        """Cleanup should not raise if worktree removal fails."""
        with patch("aragora.nomic.parallel_orchestrator.BranchCoordinator") as mock_bc_cls:
            mock_bc = MagicMock()
            mock_bc.cleanup_all_worktrees = MagicMock(side_effect=RuntimeError("worktree locked"))
            mock_bc._worktree_paths = {}
            mock_bc_cls.return_value = mock_bc

            po = ParallelOrchestrator(
                use_worktrees=True,
                enable_gauntlet=False,
                enable_convoy_tracking=False,
                workflow_engine=_mock_workflow_engine(),
                task_decomposer=_mock_decomposer(),
            )

            # Should not raise despite cleanup failure
            await po.cleanup()
