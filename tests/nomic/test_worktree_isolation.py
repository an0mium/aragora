"""Tests for BranchCoordinator git worktree isolation.

Verifies that:
- Worktrees are created/cleaned up correctly
- Branch paths are resolved properly
- Configuration controls worktree vs checkout mode
- Orchestrator passes worktree paths to workflow steps
"""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from aragora.nomic.branch_coordinator import (
    BranchCoordinator,
    BranchCoordinatorConfig,
    TrackAssignment,
    MergeResult,
)
from aragora.nomic.meta_planner import PrioritizedGoal, Track


def _make_goal(track: Track, description: str = "Test goal") -> PrioritizedGoal:
    """Helper to create a PrioritizedGoal for testing."""
    return PrioritizedGoal(
        id="goal_0",
        track=track,
        description=description,
        rationale="test",
        estimated_impact="medium",
        priority=1,
    )


class TestWorktreeConfig:
    """Tests for worktree configuration."""

    def test_default_config_enables_worktrees(self):
        """Worktrees should be enabled by default."""
        config = BranchCoordinatorConfig()
        assert config.use_worktrees is True

    def test_config_worktree_dir_default(self):
        """Worktree dir should default to {repo}/.worktrees/."""
        coordinator = BranchCoordinator(repo_path=Path("/tmp/test-repo"))
        assert coordinator._worktree_dir == Path("/tmp/test-repo/.worktrees")

    def test_config_custom_worktree_dir(self):
        """Custom worktree directory should be respected."""
        config = BranchCoordinatorConfig(
            worktree_base_dir=Path("/custom/worktrees"),
        )
        coordinator = BranchCoordinator(
            repo_path=Path("/tmp/test-repo"),
            config=config,
        )
        assert coordinator._worktree_dir == Path("/custom/worktrees")

    def test_disable_worktrees(self):
        """Should be possible to disable worktrees."""
        config = BranchCoordinatorConfig(use_worktrees=False)
        assert config.use_worktrees is False


class TestGetWorktreePath:
    """Tests for get_worktree_path method."""

    def test_returns_none_for_unknown_branch(self):
        """Should return None for branches without worktrees."""
        coordinator = BranchCoordinator(repo_path=Path("/tmp/test-repo"))
        assert coordinator.get_worktree_path("unknown-branch") is None

    def test_returns_path_after_creation(self):
        """Should return the worktree path after branch creation."""
        coordinator = BranchCoordinator(repo_path=Path("/tmp/test-repo"))
        # Simulate a worktree being registered
        branch = "dev/sme-improve-dashboard-0215"
        wt_path = Path("/tmp/test-repo/.worktrees/dev-sme-improve-dashboard-0215")
        coordinator._worktree_paths[branch] = wt_path

        assert coordinator.get_worktree_path(branch) == wt_path


class TestWorktreeBranchCreation:
    """Tests for create_track_branch with worktrees."""

    @pytest.mark.asyncio
    async def test_worktree_add_command(self):
        """Should call 'git worktree add -b' for new branches."""
        config = BranchCoordinatorConfig(use_worktrees=True)
        coordinator = BranchCoordinator(
            repo_path=Path("/tmp/test-repo"),
            config=config,
        )

        with patch.object(coordinator, "_run_git") as mock_git, \
             patch.object(coordinator, "branch_exists", return_value=False):
            mock_git.return_value = MagicMock(returncode=0, stdout="", stderr="")

            branch = await coordinator.create_track_branch(
                track=Track.SME,
                goal="improve dashboard",
            )

            assert "sme" in branch
            # Should call worktree add (not checkout)
            worktree_calls = [
                c for c in mock_git.call_args_list
                if len(c[0]) > 0 and c[0][0] == "worktree"
            ]
            assert len(worktree_calls) == 1
            args = worktree_calls[0][0]
            assert args[0] == "worktree"
            assert args[1] == "add"
            assert "-b" in args

    @pytest.mark.asyncio
    async def test_worktree_path_registered(self):
        """Should register worktree path after creation."""
        config = BranchCoordinatorConfig(use_worktrees=True)
        coordinator = BranchCoordinator(
            repo_path=Path("/tmp/test-repo"),
            config=config,
        )

        with patch.object(coordinator, "_run_git") as mock_git, \
             patch.object(coordinator, "branch_exists", return_value=False):
            mock_git.return_value = MagicMock(returncode=0, stdout="", stderr="")

            branch = await coordinator.create_track_branch(
                track=Track.QA,
                goal="add tests",
            )

            wt_path = coordinator.get_worktree_path(branch)
            assert wt_path is not None
            assert ".worktrees" in str(wt_path)

    @pytest.mark.asyncio
    async def test_existing_branch_reuses_worktree(self):
        """Should handle existing branches gracefully."""
        config = BranchCoordinatorConfig(use_worktrees=True)
        coordinator = BranchCoordinator(
            repo_path=Path("/tmp/test-repo"),
            config=config,
        )

        with patch.object(coordinator, "_run_git") as mock_git, \
             patch.object(coordinator, "branch_exists", return_value=True):
            mock_git.return_value = MagicMock(returncode=0, stdout="", stderr="")

            branch = await coordinator.create_track_branch(
                track=Track.SME,
                goal="improve dashboard",
            )

            # Should still be tracked
            assert branch in coordinator._active_branches

    @pytest.mark.asyncio
    async def test_checkout_mode_no_worktree(self):
        """Disabling worktrees should fall back to checkout."""
        config = BranchCoordinatorConfig(use_worktrees=False)
        coordinator = BranchCoordinator(
            repo_path=Path("/tmp/test-repo"),
            config=config,
        )

        with patch.object(coordinator, "_run_git") as mock_git, \
             patch.object(coordinator, "branch_exists", return_value=False), \
             patch.object(coordinator, "get_current_branch", return_value="main"):
            mock_git.return_value = MagicMock(returncode=0, stdout="", stderr="")

            branch = await coordinator.create_track_branch(
                track=Track.DEVELOPER,
                goal="update SDK",
            )

            # Should use checkout, not worktree
            all_args = [c[0] for c in mock_git.call_args_list]
            worktree_calls = [a for a in all_args if a[0] == "worktree"]
            checkout_calls = [a for a in all_args if a[0] == "checkout"]
            assert len(worktree_calls) == 0
            assert len(checkout_calls) >= 1


class TestWorktreeCleanup:
    """Tests for worktree cleanup."""

    def test_cleanup_removes_worktree(self):
        """cleanup_branches should remove worktrees."""
        coordinator = BranchCoordinator(repo_path=Path("/tmp/test-repo"))
        branch = "dev/sme-test-0215"
        wt_path = Path("/tmp/test-repo/.worktrees/dev-sme-test-0215")

        coordinator._active_branches.append(branch)
        coordinator._worktree_paths[branch] = wt_path

        with patch.object(coordinator, "_run_git") as mock_git, \
             patch.object(coordinator, "branch_exists", return_value=True), \
             patch("pathlib.Path.exists", return_value=True):
            # Simulate branch merged
            mock_git.return_value = MagicMock(
                returncode=0,
                stdout=f"  main\n  {branch}\n",
                stderr="",
            )

            deleted = coordinator.cleanup_branches([branch])

            assert deleted == 1
            # Should have called worktree remove
            worktree_remove_calls = [
                c for c in mock_git.call_args_list
                if len(c[0]) >= 2 and c[0][0] == "worktree" and c[0][1] == "remove"
            ]
            assert len(worktree_remove_calls) >= 1

    def test_remove_worktree_clears_path_map(self):
        """_remove_worktree should clear the path from the map."""
        coordinator = BranchCoordinator(repo_path=Path("/tmp/test-repo"))
        branch = "dev/sme-test-0215"
        wt_path = Path("/tmp/test-repo/.worktrees/dev-sme-test-0215")
        coordinator._worktree_paths[branch] = wt_path

        with patch.object(coordinator, "_run_git") as mock_git, \
             patch("pathlib.Path.exists", return_value=True):
            mock_git.return_value = MagicMock(returncode=0)

            coordinator._remove_worktree(branch)

            assert branch not in coordinator._worktree_paths


class TestRunAssignmentWithWorktrees:
    """Tests for _run_assignment with worktree isolation."""

    @pytest.mark.asyncio
    async def test_no_checkout_in_worktree_mode(self):
        """Should not call git checkout when using worktrees."""
        config = BranchCoordinatorConfig(use_worktrees=True)
        coordinator = BranchCoordinator(
            repo_path=Path("/tmp/test-repo"),
            config=config,
        )

        branch = "dev/sme-test-0215"
        coordinator._worktree_paths[branch] = Path("/tmp/test-repo/.worktrees/dev-sme-test-0215")

        goal = _make_goal(Track.SME, "test goal")
        assignment = TrackAssignment(goal=goal, branch_name=branch)

        async def mock_nomic_fn(a):
            return {"success": True}

        with patch.object(coordinator, "_run_git") as mock_git:
            mock_git.return_value = MagicMock(returncode=0)

            await coordinator._run_assignment(assignment, mock_nomic_fn)

            # Should NOT have called checkout
            checkout_calls = [
                c for c in mock_git.call_args_list
                if len(c[0]) > 0 and c[0][0] == "checkout"
            ]
            assert len(checkout_calls) == 0

        assert assignment.status == "completed"

    @pytest.mark.asyncio
    async def test_checkout_in_legacy_mode(self):
        """Should call git checkout when worktrees are disabled."""
        config = BranchCoordinatorConfig(use_worktrees=False)
        coordinator = BranchCoordinator(
            repo_path=Path("/tmp/test-repo"),
            config=config,
        )

        branch = "dev/sme-test-0215"
        goal = _make_goal(Track.SME, "test goal")
        assignment = TrackAssignment(goal=goal, branch_name=branch)

        async def mock_nomic_fn(a):
            return {"success": True}

        with patch.object(coordinator, "_run_git") as mock_git:
            mock_git.return_value = MagicMock(returncode=0)

            await coordinator._run_assignment(assignment, mock_nomic_fn)

            # Should have called checkout for the branch and to return to base
            checkout_calls = [
                c for c in mock_git.call_args_list
                if len(c[0]) > 0 and c[0][0] == "checkout"
            ]
            assert len(checkout_calls) >= 1


class TestOrchestratorWorktreeIntegration:
    """Tests for AutonomousOrchestrator worktree path resolution."""

    def test_orchestrator_accepts_branch_coordinator(self):
        """Orchestrator should accept branch_coordinator parameter."""
        from aragora.nomic.autonomous_orchestrator import AutonomousOrchestrator

        mock_coordinator = MagicMock()
        orch = AutonomousOrchestrator(branch_coordinator=mock_coordinator)
        assert orch.branch_coordinator is mock_coordinator

    def test_build_workflow_uses_worktree_path(self):
        """Workflow should use worktree repo_path when available."""
        from aragora.nomic.autonomous_orchestrator import (
            AutonomousOrchestrator,
            AgentAssignment,
            Track as OrcTrack,
        )
        from aragora.nomic.task_decomposer import SubTask

        mock_coordinator = MagicMock()
        mock_coordinator._worktree_paths = {
            "dev/sme-improve-0215": Path("/tmp/repo/.worktrees/dev-sme-improve-0215"),
        }

        orch = AutonomousOrchestrator(
            aragora_path=Path("/tmp/repo"),
            branch_coordinator=mock_coordinator,
        )

        assignment = AgentAssignment(
            subtask=SubTask(
                id="subtask_1",
                title="Improve dashboard",
                description="Add charts",
                file_scope=["aragora/live/src/app/page.tsx"],
            ),
            track=OrcTrack.SME,
            agent_type="claude",
        )

        workflow = orch._build_subtask_workflow(assignment)

        # Find the implementation step
        impl_step = next(s for s in workflow.steps if s.step_type == "implementation")
        repo_path = impl_step.config["repo_path"]

        # Should use worktree path since the branch name contains "sme"
        assert "worktrees" in repo_path

    def test_build_workflow_default_path_without_coordinator(self):
        """Without coordinator, should use default aragora_path."""
        from aragora.nomic.autonomous_orchestrator import (
            AutonomousOrchestrator,
            AgentAssignment,
            Track as OrcTrack,
        )
        from aragora.nomic.task_decomposer import SubTask

        orch = AutonomousOrchestrator(aragora_path=Path("/tmp/repo"))

        assignment = AgentAssignment(
            subtask=SubTask(
                id="subtask_1",
                title="Improve SDK",
                description="Add methods",
                file_scope=["sdk/python/client.py"],
            ),
            track=OrcTrack.DEVELOPER,
            agent_type="claude",
        )

        workflow = orch._build_subtask_workflow(assignment)
        impl_step = next(s for s in workflow.steps if s.step_type == "implementation")

        assert impl_step.config["repo_path"] == "/tmp/repo"


class TestCreateTrackBranchesWorktree:
    """Tests for create_track_branches with worktrees."""

    @pytest.mark.asyncio
    async def test_no_checkout_base_after_worktree_creation(self):
        """Should not checkout base branch when using worktrees."""
        config = BranchCoordinatorConfig(use_worktrees=True)
        coordinator = BranchCoordinator(
            repo_path=Path("/tmp/test-repo"),
            config=config,
        )

        goal = _make_goal(Track.SME, "test goal")
        assignments = [TrackAssignment(goal=goal)]

        with patch.object(coordinator, "_run_git") as mock_git, \
             patch.object(coordinator, "branch_exists", return_value=False):
            mock_git.return_value = MagicMock(returncode=0, stdout="", stderr="")

            result = await coordinator.create_track_branches(assignments)

            # Should not have a bare "checkout main" call at the end
            checkout_base_calls = [
                c for c in mock_git.call_args_list
                if len(c[0]) >= 2 and c[0][0] == "checkout" and c[0][1] == "main"
            ]
            assert len(checkout_base_calls) == 0
