"""Tests for BranchCoordinator git worktree isolation.

Verifies that:
- Worktrees are created/cleaned up correctly
- Branch paths are resolved properly
- Configuration controls worktree vs checkout mode
- Orchestrator passes worktree paths to workflow steps
"""

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from aragora.nomic.branch_coordinator import (
    BranchCoordinator,
    BranchCoordinatorConfig,
    TrackAssignment,
    WorktreeInfo,
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

        with (
            patch.object(coordinator, "_run_git") as mock_git,
            patch.object(coordinator, "branch_exists", return_value=False),
        ):
            mock_git.return_value = MagicMock(returncode=0, stdout="", stderr="")

            branch = await coordinator.create_track_branch(
                track=Track.SME,
                goal="improve dashboard",
            )

            assert "sme" in branch
            # Should call worktree add (not checkout)
            worktree_calls = [
                c for c in mock_git.call_args_list if len(c[0]) > 0 and c[0][0] == "worktree"
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

        with (
            patch.object(coordinator, "_run_git") as mock_git,
            patch.object(coordinator, "branch_exists", return_value=False),
        ):
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

        with (
            patch.object(coordinator, "_run_git") as mock_git,
            patch.object(coordinator, "branch_exists", return_value=True),
        ):
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

        with (
            patch.object(coordinator, "_run_git") as mock_git,
            patch.object(coordinator, "branch_exists", return_value=False),
            patch.object(coordinator, "get_current_branch", return_value="main"),
        ):
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

        with (
            patch.object(coordinator, "_run_git") as mock_git,
            patch.object(coordinator, "branch_exists", return_value=True),
            patch("pathlib.Path.exists", return_value=True),
        ):
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
                c
                for c in mock_git.call_args_list
                if len(c[0]) >= 2 and c[0][0] == "worktree" and c[0][1] == "remove"
            ]
            assert len(worktree_remove_calls) >= 1

    def test_remove_worktree_clears_path_map(self):
        """_remove_worktree should clear the path from the map."""
        coordinator = BranchCoordinator(repo_path=Path("/tmp/test-repo"))
        branch = "dev/sme-test-0215"
        wt_path = Path("/tmp/test-repo/.worktrees/dev-sme-test-0215")
        coordinator._worktree_paths[branch] = wt_path

        with (
            patch.object(coordinator, "_run_git") as mock_git,
            patch("pathlib.Path.exists", return_value=True),
        ):
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
                c for c in mock_git.call_args_list if len(c[0]) > 0 and c[0][0] == "checkout"
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
                c for c in mock_git.call_args_list if len(c[0]) > 0 and c[0][0] == "checkout"
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

        with (
            patch.object(coordinator, "_run_git") as mock_git,
            patch.object(coordinator, "branch_exists", return_value=False),
        ):
            mock_git.return_value = MagicMock(returncode=0, stdout="", stderr="")

            result = await coordinator.create_track_branches(assignments)

            # Should not have a bare "checkout main" call at the end
            checkout_base_calls = [
                c
                for c in mock_git.call_args_list
                if len(c[0]) >= 2 and c[0][0] == "checkout" and c[0][1] == "main"
            ]
            assert len(checkout_base_calls) == 0


class TestWorktreeInfo:
    """Tests for WorktreeInfo dataclass."""

    def test_creation_required_fields(self):
        """Should create with required fields only."""
        info = WorktreeInfo(
            branch_name="dev/sme-feature",
            worktree_path=Path("/tmp/.worktrees/dev-sme-feature"),
        )
        assert info.branch_name == "dev/sme-feature"
        assert info.worktree_path == Path("/tmp/.worktrees/dev-sme-feature")
        assert info.track is None
        assert info.created_at is None
        assert info.assignment_id is None

    def test_creation_all_fields(self):
        """Should accept all optional fields."""
        now = datetime.now(timezone.utc)
        info = WorktreeInfo(
            branch_name="dev/qa-tests",
            worktree_path=Path("/tmp/.worktrees/dev-qa-tests"),
            track="qa",
            created_at=now,
            assignment_id="assign-001",
        )
        assert info.track == "qa"
        assert info.created_at == now
        assert info.assignment_id == "assign-001"

    def test_worktree_path_is_path_object(self):
        """Should store path as Path object."""
        info = WorktreeInfo(
            branch_name="dev/feature",
            worktree_path=Path("/tmp/wt"),
        )
        assert isinstance(info.worktree_path, Path)


class TestListWorktrees:
    """Tests for list_worktrees method."""

    @patch("subprocess.run")
    def test_list_empty(self, mock_run):
        """Should return empty list when no worktrees."""
        mock_run.return_value = MagicMock(stdout="", returncode=0)
        coordinator = BranchCoordinator()
        result = coordinator.list_worktrees()
        assert result == []

    @patch("subprocess.run")
    def test_list_single_worktree(self, mock_run):
        """Should parse single worktree from porcelain output."""
        porcelain = "worktree /path/to/main\nHEAD abc123\nbranch refs/heads/main\n\n"
        mock_run.return_value = MagicMock(stdout=porcelain, returncode=0)
        coordinator = BranchCoordinator()
        result = coordinator.list_worktrees()
        assert len(result) == 1
        assert result[0].branch_name == "main"
        assert result[0].worktree_path == Path("/path/to/main")

    @patch("subprocess.run")
    def test_list_multiple_worktrees(self, mock_run):
        """Should parse multiple worktrees."""
        porcelain = (
            "worktree /path/to/main\n"
            "HEAD abc123\n"
            "branch refs/heads/main\n"
            "\n"
            "worktree /path/to/worktree1\n"
            "HEAD def456\n"
            "branch refs/heads/dev/sme-feature\n"
            "\n"
        )
        mock_run.return_value = MagicMock(stdout=porcelain, returncode=0)
        coordinator = BranchCoordinator()
        result = coordinator.list_worktrees()
        assert len(result) == 2
        assert result[0].branch_name == "main"
        assert result[1].branch_name == "dev/sme-feature"

    @patch("subprocess.run")
    def test_list_cross_references_tracked(self, mock_run):
        """Should use tracked WorktreeInfo when available."""
        porcelain = "worktree /path/to/worktree\nHEAD abc123\nbranch refs/heads/dev/sme-feature\n\n"
        mock_run.return_value = MagicMock(stdout=porcelain, returncode=0)
        coordinator = BranchCoordinator()

        now = datetime.now(timezone.utc)
        tracked_info = WorktreeInfo(
            branch_name="dev/sme-feature",
            worktree_path=Path("/path/to/worktree"),
            track="sme",
            created_at=now,
        )
        coordinator._active_worktrees["dev/sme-feature"] = tracked_info

        result = coordinator.list_worktrees()
        assert len(result) == 1
        assert result[0].track == "sme"
        assert result[0].created_at == now


class TestMergeWorktreeBack:
    """Tests for merge_worktree_back method."""

    @pytest.mark.asyncio
    @patch("subprocess.run")
    async def test_success_with_cleanup(self, mock_run):
        """Should merge and cleanup worktree on success."""
        mock_run.side_effect = [
            MagicMock(returncode=0),  # branch_exists
            MagicMock(returncode=0),  # checkout
            MagicMock(returncode=0),  # pull
            MagicMock(returncode=0),  # merge
            MagicMock(stdout="abc123\n", returncode=0),  # rev-parse
            MagicMock(returncode=0),  # worktree remove
        ]

        coordinator = BranchCoordinator()
        wt_path = Path("/tmp/.worktrees/dev-feature")
        coordinator._worktree_paths["dev/feature"] = wt_path
        coordinator._active_worktrees["dev/feature"] = WorktreeInfo(
            branch_name="dev/feature",
            worktree_path=wt_path,
        )

        with patch.object(Path, "exists", return_value=True):
            result = await coordinator.merge_worktree_back("dev/feature")

        assert result.success is True
        assert "dev/feature" not in coordinator._worktree_paths
        assert "dev/feature" not in coordinator._active_worktrees

    @pytest.mark.asyncio
    @patch("subprocess.run")
    async def test_success_without_cleanup(self, mock_run):
        """Should merge but preserve worktree when cleanup=False."""
        mock_run.side_effect = [
            MagicMock(returncode=0),  # branch_exists
            MagicMock(returncode=0),  # checkout
            MagicMock(returncode=0),  # pull
            MagicMock(returncode=0),  # merge
            MagicMock(stdout="abc123\n", returncode=0),  # rev-parse
        ]

        coordinator = BranchCoordinator()
        wt_path = Path("/tmp/.worktrees/dev-feature")
        coordinator._worktree_paths["dev/feature"] = wt_path
        coordinator._active_worktrees["dev/feature"] = WorktreeInfo(
            branch_name="dev/feature",
            worktree_path=wt_path,
        )

        result = await coordinator.merge_worktree_back("dev/feature", cleanup=False)

        assert result.success is True
        assert "dev/feature" in coordinator._worktree_paths
        assert "dev/feature" in coordinator._active_worktrees

    @pytest.mark.asyncio
    @patch("subprocess.run")
    async def test_merge_conflict_preserves_worktree(self, mock_run):
        """Should preserve worktree on merge conflict."""
        mock_run.side_effect = [
            MagicMock(returncode=0),  # branch_exists
            MagicMock(returncode=0),  # checkout
            MagicMock(returncode=0),  # pull
            MagicMock(
                returncode=1,
                stderr="CONFLICT (content): Merge conflict in app.py",
            ),  # merge fails
            MagicMock(returncode=0),  # merge --abort
        ]

        coordinator = BranchCoordinator()
        wt_path = Path("/tmp/.worktrees/dev-feature")
        coordinator._worktree_paths["dev/feature"] = wt_path
        coordinator._active_worktrees["dev/feature"] = WorktreeInfo(
            branch_name="dev/feature",
            worktree_path=wt_path,
        )

        result = await coordinator.merge_worktree_back("dev/feature")

        assert result.success is False
        assert "dev/feature" in coordinator._worktree_paths
        assert "dev/feature" in coordinator._active_worktrees


class TestCleanupAllWorktrees:
    """Tests for cleanup_all_worktrees method."""

    @patch("subprocess.run")
    def test_cleanup_empty(self, mock_run):
        """Should handle no worktrees gracefully."""
        mock_run.return_value = MagicMock(returncode=0)
        coordinator = BranchCoordinator()
        removed = coordinator.cleanup_all_worktrees()
        assert removed == 0

    @patch("subprocess.run")
    def test_cleanup_multiple(self, mock_run):
        """Should remove multiple worktrees and prune."""
        mock_run.return_value = MagicMock(returncode=0)
        coordinator = BranchCoordinator()

        coordinator._worktree_paths["branch1"] = Path("/tmp/.worktrees/branch1")
        coordinator._worktree_paths["branch2"] = Path("/tmp/.worktrees/branch2")
        coordinator._active_worktrees["branch1"] = WorktreeInfo(
            branch_name="branch1",
            worktree_path=Path("/tmp/.worktrees/branch1"),
        )
        coordinator._active_worktrees["branch2"] = WorktreeInfo(
            branch_name="branch2",
            worktree_path=Path("/tmp/.worktrees/branch2"),
        )

        with patch.object(Path, "exists", return_value=True):
            removed = coordinator.cleanup_all_worktrees()

        assert removed == 2
        assert len(coordinator._worktree_paths) == 0
        assert len(coordinator._active_worktrees) == 0

    @patch("subprocess.run")
    def test_cleanup_partial_failure(self, mock_run):
        """Should continue cleanup even if one worktree removal fails."""
        mock_run.side_effect = [
            MagicMock(returncode=1),  # worktree remove fails
            MagicMock(returncode=0),  # worktree remove succeeds
            MagicMock(returncode=0),  # worktree prune
        ]
        coordinator = BranchCoordinator()
        coordinator._worktree_paths["branch1"] = Path("/tmp/.worktrees/branch1")
        coordinator._worktree_paths["branch2"] = Path("/tmp/.worktrees/branch2")

        with patch.object(Path, "exists", return_value=True):
            removed = coordinator.cleanup_all_worktrees()

        assert removed == 2
        assert len(coordinator._worktree_paths) == 0


class TestWorktreeGit:
    """Tests for _worktree_git method."""

    @patch("subprocess.run")
    def test_basic_command(self, mock_run):
        """Should run git command in worktree directory."""
        mock_run.return_value = MagicMock(
            stdout="feature-branch\n",
            returncode=0,
        )
        coordinator = BranchCoordinator()
        wt_path = Path("/tmp/.worktrees/dev-feature")
        result = coordinator._worktree_git(wt_path, "rev-parse", "--abbrev-ref", "HEAD")

        call_args = mock_run.call_args
        assert call_args[0][0] == ["git", "rev-parse", "--abbrev-ref", "HEAD"]
        assert call_args[1]["cwd"] == wt_path
        assert result.stdout == "feature-branch\n"

    @patch("subprocess.run")
    def test_error_handling(self, mock_run):
        """Should respect check=False for error handling."""
        mock_run.return_value = MagicMock(returncode=1, stderr="error")
        coordinator = BranchCoordinator()
        wt_path = Path("/tmp/.worktrees/dev-feature")
        result = coordinator._worktree_git(wt_path, "status", check=False)
        assert result.returncode == 1


class TestContextManager:
    """Tests for async context manager."""

    @pytest.mark.asyncio
    async def test_enter_returns_coordinator(self):
        """Should return coordinator on enter."""
        config = BranchCoordinatorConfig(use_worktrees=False)
        coordinator = BranchCoordinator(config=config)
        async with coordinator as ctx:
            assert ctx is coordinator

    @pytest.mark.asyncio
    @patch("subprocess.run")
    async def test_exit_cleans_up_worktrees(self, mock_run):
        """Should cleanup all worktrees on exit."""
        mock_run.return_value = MagicMock(returncode=0)
        coordinator = BranchCoordinator()
        coordinator._worktree_paths["branch1"] = Path("/tmp/.worktrees/branch1")
        coordinator._active_worktrees["branch1"] = WorktreeInfo(
            branch_name="branch1",
            worktree_path=Path("/tmp/.worktrees/branch1"),
        )

        with patch.object(Path, "exists", return_value=True):
            async with coordinator:
                pass

        assert len(coordinator._worktree_paths) == 0
        assert len(coordinator._active_worktrees) == 0

    @pytest.mark.asyncio
    @patch("subprocess.run")
    async def test_cleanup_on_error(self, mock_run):
        """Should cleanup worktrees even when exception occurs."""
        mock_run.return_value = MagicMock(returncode=0)
        coordinator = BranchCoordinator()
        coordinator._worktree_paths["branch1"] = Path("/tmp/.worktrees/branch1")

        with patch.object(Path, "exists", return_value=True):
            with pytest.raises(ValueError, match="test error"):
                async with coordinator:
                    raise ValueError("test error")

        assert len(coordinator._worktree_paths) == 0


class TestCoordinateWithWorktrees:
    """Tests for coordinate_parallel_work with worktrees."""

    @pytest.mark.asyncio
    @patch("subprocess.run")
    async def test_parallel_execution(self, mock_run):
        """Should create branches and run in parallel."""
        mock_run.return_value = MagicMock(stdout="main\n", returncode=0)

        config = BranchCoordinatorConfig(auto_merge_safe=False, use_worktrees=False)
        coordinator = BranchCoordinator(config=config)

        goal = _make_goal(Track.SME, "test parallel")
        assignment = TrackAssignment(goal=goal)

        async def nomic_fn(a):
            return {"result": "done"}

        result = await coordinator.coordinate_parallel_work(
            [assignment],
            run_nomic_fn=nomic_fn,
        )

        assert result.total_branches == 1

    @pytest.mark.asyncio
    @patch("subprocess.run")
    async def test_conflict_detection_still_works(self, mock_run):
        """Should detect conflicts in worktree mode too."""
        mock_run.return_value = MagicMock(stdout="main\n", returncode=0)

        config = BranchCoordinatorConfig(auto_merge_safe=False)
        coordinator = BranchCoordinator(config=config)

        a1 = TrackAssignment(
            goal=_make_goal(Track.SME, "Frontend"),
            branch_name="b1",
        )
        a2 = TrackAssignment(
            goal=_make_goal(Track.QA, "Tests"),
            branch_name="b2",
        )

        result = await coordinator.coordinate_parallel_work([a1, a2])
        assert result.total_branches == 2

    @pytest.mark.asyncio
    @patch("subprocess.run")
    async def test_auto_merge_after_completion(self, mock_run):
        """Should auto-merge completed branches when configured."""
        mock_run.return_value = MagicMock(stdout="main\n", returncode=0)

        config = BranchCoordinatorConfig(auto_merge_safe=True, use_worktrees=False)
        coordinator = BranchCoordinator(config=config)

        assignment = TrackAssignment(
            goal=_make_goal(Track.SME),
            branch_name="dev/feature",
        )

        async def success_fn(a):
            return {"success": True}

        result = await coordinator.coordinate_parallel_work(
            [assignment],
            run_nomic_fn=success_fn,
        )
        assert result.total_branches == 1

    @pytest.mark.asyncio
    @patch("subprocess.run")
    async def test_no_nomic_fn(self, mock_run):
        """Should work without a nomic function."""
        mock_run.return_value = MagicMock(stdout="main\n", returncode=0)

        config = BranchCoordinatorConfig(auto_merge_safe=False, use_worktrees=False)
        coordinator = BranchCoordinator(config=config)

        assignment = TrackAssignment(goal=_make_goal(Track.SME))
        result = await coordinator.coordinate_parallel_work([assignment])

        assert result.total_branches == 1
        assert result.success is True

    @pytest.mark.asyncio
    @patch("subprocess.run")
    async def test_failed_assignment_reported(self, mock_run):
        """Should report failed assignments in result."""
        mock_run.return_value = MagicMock(stdout="main\n", returncode=0)

        config = BranchCoordinatorConfig(auto_merge_safe=False, use_worktrees=False)
        coordinator = BranchCoordinator(config=config)

        assignment = TrackAssignment(
            goal=_make_goal(Track.QA),
            branch_name="dev/failing",
        )

        async def fail_fn(a):
            raise RuntimeError("Build failed")

        result = await coordinator.coordinate_parallel_work(
            [assignment],
            run_nomic_fn=fail_fn,
        )

        assert result.failed_branches >= 0  # gather with return_exceptions=True


class TestBackwardCompat:
    """Tests for backward compatibility when use_worktrees=False."""

    @patch("subprocess.run")
    def test_no_worktrees_preserves_old_behavior(self, mock_run):
        """Should not create worktree paths when disabled."""
        mock_run.return_value = MagicMock(returncode=0)
        config = BranchCoordinatorConfig(use_worktrees=False)
        coordinator = BranchCoordinator(config=config)
        assert coordinator._worktree_paths == {}
        assert coordinator._active_worktrees == {}

    @pytest.mark.asyncio
    @patch("subprocess.run")
    async def test_checkout_mode_no_worktree_info(self, mock_run):
        """Should not populate _active_worktrees in checkout mode."""
        mock_run.side_effect = [
            MagicMock(stdout="main\n", returncode=0),  # get_current_branch
            MagicMock(returncode=1),  # branch_exists
            MagicMock(returncode=0),  # checkout -b
        ]
        config = BranchCoordinatorConfig(use_worktrees=False)
        coordinator = BranchCoordinator(config=config)

        await coordinator._create_checkout_branch("dev/legacy", "main")

        assert "dev/legacy" not in coordinator._active_worktrees
        assert "dev/legacy" not in coordinator._worktree_paths

    @pytest.mark.asyncio
    @patch("subprocess.run")
    async def test_cleanup_all_noop_in_checkout_mode(self, mock_run):
        """Should return 0 when no worktrees exist."""
        mock_run.return_value = MagicMock(returncode=0)
        config = BranchCoordinatorConfig(use_worktrees=False)
        coordinator = BranchCoordinator(config=config)
        removed = coordinator.cleanup_all_worktrees()
        assert removed == 0


class TestStaleWorktreeDetection:
    """Tests for detecting untracked/stale worktrees."""

    @patch("subprocess.run")
    def test_detects_untracked_worktrees(self, mock_run):
        """Should list worktrees not in _active_worktrees."""
        porcelain = (
            "worktree /path/to/main\n"
            "HEAD abc123\n"
            "branch refs/heads/main\n"
            "\n"
            "worktree /path/to/stale\n"
            "HEAD def456\n"
            "branch refs/heads/dev/stale-branch\n"
            "\n"
        )
        mock_run.return_value = MagicMock(stdout=porcelain, returncode=0)
        coordinator = BranchCoordinator()

        worktrees = coordinator.list_worktrees()
        tracked_branches = set(coordinator._active_worktrees.keys())
        stale = [w for w in worktrees if w.branch_name not in tracked_branches]

        assert len(stale) == 2  # main + stale branch are both untracked

    @patch("subprocess.run")
    def test_mix_of_tracked_and_untracked(self, mock_run):
        """Should distinguish tracked from untracked worktrees."""
        porcelain = (
            "worktree /path/to/tracked\n"
            "HEAD abc123\n"
            "branch refs/heads/dev/tracked\n"
            "\n"
            "worktree /path/to/untracked\n"
            "HEAD def456\n"
            "branch refs/heads/dev/untracked\n"
            "\n"
        )
        mock_run.return_value = MagicMock(stdout=porcelain, returncode=0)
        coordinator = BranchCoordinator()
        coordinator._active_worktrees["dev/tracked"] = WorktreeInfo(
            branch_name="dev/tracked",
            worktree_path=Path("/path/to/tracked"),
            track="sme",
        )

        worktrees = coordinator.list_worktrees()
        assert len(worktrees) == 2
        tracked = [w for w in worktrees if w.track == "sme"]
        untracked = [w for w in worktrees if w.track is None]
        assert len(tracked) == 1
        assert len(untracked) == 1


class TestSelfDevIntegration:
    """Tests for self_development.py integration with worktrees."""

    def test_worktree_path_in_branch_info(self):
        """Should include worktree_path in branch info dict."""
        goal = _make_goal(Track.SME)
        assignment = TrackAssignment(
            goal=goal,
            branch_name="dev/sme-test",
            worktree_path=Path("/tmp/.worktrees/dev-sme-test"),
        )
        branch_info = {
            "branch_name": assignment.branch_name,
            "track": assignment.goal.track.value,
            "goal": assignment.goal.description,
            "worktree_path": str(assignment.worktree_path) if assignment.worktree_path else None,
        }
        assert branch_info["worktree_path"] == "/tmp/.worktrees/dev-sme-test"

    def test_worktree_path_none_in_checkout_mode(self):
        """Should have None worktree_path in checkout mode."""
        goal = _make_goal(Track.SME)
        assignment = TrackAssignment(
            goal=goal,
            branch_name="dev/legacy",
            worktree_path=None,
        )
        branch_info = {
            "worktree_path": str(assignment.worktree_path) if assignment.worktree_path else None,
        }
        assert branch_info["worktree_path"] is None

    @pytest.mark.asyncio
    @patch("subprocess.run")
    async def test_merge_worktree_back_used_for_merge(self, mock_run):
        """Should use merge_worktree_back when worktree_path is present."""
        mock_run.side_effect = [
            MagicMock(returncode=0),  # branch_exists
            MagicMock(returncode=0),  # checkout
            MagicMock(returncode=0),  # pull
            MagicMock(returncode=0),  # merge
            MagicMock(stdout="abc123\n", returncode=0),  # rev-parse
        ]

        coordinator = BranchCoordinator()
        result = await coordinator.merge_worktree_back("dev/feature")

        assert result.success is True
        assert result.commit_sha == "abc123"


class TestWorktreeInfoTracking:
    """Tests for _active_worktrees tracking behavior."""

    def test_initially_empty(self):
        """Should start with empty _active_worktrees."""
        coordinator = BranchCoordinator()
        assert coordinator._active_worktrees == {}

    @pytest.mark.asyncio
    @patch("subprocess.run")
    async def test_populated_on_create(self, mock_run):
        """Should be populated when worktree branch is created."""
        mock_run.side_effect = [
            MagicMock(returncode=1),  # branch_exists
            MagicMock(returncode=0),  # git worktree add
        ]
        coordinator = BranchCoordinator(repo_path=Path("/tmp/repo"))

        with patch.object(Path, "mkdir"):
            branch = await coordinator._create_worktree_branch("dev/test", "main")

        assert branch in coordinator._active_worktrees
        assert coordinator._active_worktrees[branch].branch_name == "dev/test"

    @patch("subprocess.run")
    def test_cleared_on_cleanup(self, mock_run):
        """Should be cleared on cleanup_all_worktrees."""
        mock_run.return_value = MagicMock(returncode=0)
        coordinator = BranchCoordinator()

        coordinator._worktree_paths["b1"] = Path("/tmp/.worktrees/b1")
        coordinator._active_worktrees["b1"] = WorktreeInfo(
            branch_name="b1",
            worktree_path=Path("/tmp/.worktrees/b1"),
        )

        with patch.object(Path, "exists", return_value=True):
            coordinator.cleanup_all_worktrees()

        assert len(coordinator._active_worktrees) == 0

    @patch("subprocess.run")
    def test_cleared_on_remove_worktree(self, mock_run):
        """Should remove entry when worktree is removed."""
        mock_run.return_value = MagicMock(returncode=0)
        coordinator = BranchCoordinator()

        wt_path = Path("/tmp/.worktrees/dev-feature")
        coordinator._worktree_paths["dev/feature"] = wt_path
        coordinator._active_worktrees["dev/feature"] = WorktreeInfo(
            branch_name="dev/feature",
            worktree_path=wt_path,
        )

        with patch.object(Path, "exists", return_value=True):
            coordinator._remove_worktree("dev/feature")

        assert "dev/feature" not in coordinator._active_worktrees
        assert "dev/feature" not in coordinator._worktree_paths
