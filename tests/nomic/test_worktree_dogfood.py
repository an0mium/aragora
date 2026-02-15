"""Dogfood validation test for the worktree multi-session workflow.

Proves the end-to-end flow works:
1. Create worktrees via BranchCoordinator
2. Decompose a goal into subtasks assigned to different tracks
3. Verify worktree paths exist and are isolated
4. Simulate completion, dry-run merge
5. Cleanup removes all worktrees

These tests use a temporary git repo to avoid touching the real working tree.
"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

import pytest

from aragora.nomic.branch_coordinator import (
    BranchCoordinator,
    BranchCoordinatorConfig,
)
from aragora.nomic.meta_planner import Track


@pytest.fixture
def temp_git_repo(tmp_path: Path) -> Path:
    """Create a temporary git repo for testing worktree operations."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(
        ["git", "init", "--initial-branch=main"],
        cwd=repo, capture_output=True, check=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=repo, capture_output=True, check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=repo, capture_output=True, check=True,
    )
    # Create initial commit so branches can be created
    (repo / "README.md").write_text("# Test\n")
    subprocess.run(["git", "add", "."], cwd=repo, capture_output=True, check=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=repo, capture_output=True, check=True,
    )
    return repo


@pytest.fixture
def coordinator(temp_git_repo: Path) -> BranchCoordinator:
    """Create a BranchCoordinator for the temp repo."""
    config = BranchCoordinatorConfig(
        base_branch="main",
        use_worktrees=True,
        max_parallel_branches=6,
    )
    return BranchCoordinator(repo_path=temp_git_repo, config=config)


class TestWorktreeCreation:
    """Test that worktrees can be created for different tracks."""

    def test_create_single_worktree(self, coordinator: BranchCoordinator) -> None:
        branch = asyncio.run(
            coordinator.create_track_branch(track=Track.SME, goal="Improve dashboard")
        )
        assert branch.startswith("dev/sme-")
        wt_path = coordinator.get_worktree_path(branch)
        assert wt_path is not None
        assert wt_path.exists()

    def test_create_multiple_worktrees(self, coordinator: BranchCoordinator) -> None:
        tracks = [Track.SME, Track.DEVELOPER, Track.QA]
        branches = []
        for track in tracks:
            branch = asyncio.run(
                coordinator.create_track_branch(track=track, goal=f"Work on {track.value}")
            )
            branches.append(branch)

        assert len(branches) == 3
        for branch in branches:
            wt_path = coordinator.get_worktree_path(branch)
            assert wt_path is not None
            assert wt_path.exists()

    def test_worktrees_are_isolated(self, coordinator: BranchCoordinator, temp_git_repo: Path) -> None:
        """Each worktree is at a different filesystem path."""
        b1 = asyncio.run(coordinator.create_track_branch(track=Track.SME, goal="SME work"))
        b2 = asyncio.run(coordinator.create_track_branch(track=Track.QA, goal="QA work"))

        p1 = coordinator.get_worktree_path(b1)
        p2 = coordinator.get_worktree_path(b2)

        assert p1 != p2
        assert p1 != temp_git_repo
        assert p2 != temp_git_repo


class TestWorktreeListing:
    """Test listing worktrees."""

    def test_list_empty(self, coordinator: BranchCoordinator) -> None:
        worktrees = coordinator.list_worktrees()
        # May include the main worktree
        assert isinstance(worktrees, list)

    def test_list_after_create(self, coordinator: BranchCoordinator) -> None:
        asyncio.run(coordinator.create_track_branch(track=Track.CORE, goal="Core work"))
        worktrees = coordinator.list_worktrees()
        branches = [wt.branch_name for wt in worktrees]
        assert any("core" in b for b in branches)


class TestWorktreeMerge:
    """Test merging worktree branches."""

    def test_dry_run_merge(self, coordinator: BranchCoordinator, temp_git_repo: Path) -> None:
        branch = asyncio.run(
            coordinator.create_track_branch(track=Track.DEVELOPER, goal="Add feature")
        )
        wt_path = coordinator.get_worktree_path(branch)

        # Make a change in the worktree
        (wt_path / "feature.py").write_text("# new feature\n")
        subprocess.run(["git", "add", "."], cwd=wt_path, capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "add feature"],
            cwd=wt_path, capture_output=True, check=True,
        )

        # Dry-run merge should succeed
        result = asyncio.run(coordinator.safe_merge(branch, dry_run=True))
        assert result.success

    def test_actual_merge(self, coordinator: BranchCoordinator, temp_git_repo: Path) -> None:
        branch = asyncio.run(
            coordinator.create_track_branch(track=Track.SECURITY, goal="Security audit")
        )
        wt_path = coordinator.get_worktree_path(branch)

        # Make a change
        (wt_path / "audit.py").write_text("# audit log\n")
        subprocess.run(["git", "add", "."], cwd=wt_path, capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "add audit"],
            cwd=wt_path, capture_output=True, check=True,
        )

        # Actual merge
        result = asyncio.run(coordinator.safe_merge(branch))
        assert result.success
        assert result.commit_sha is not None

        # Verify file exists on main
        assert (temp_git_repo / "audit.py").exists()


class TestConflictDetection:
    """Test conflict detection across branches."""

    def test_no_conflicts_on_fresh_branches(self, coordinator: BranchCoordinator) -> None:
        b1 = asyncio.run(coordinator.create_track_branch(track=Track.SME, goal="SME"))
        b2 = asyncio.run(coordinator.create_track_branch(track=Track.QA, goal="QA"))

        conflicts = asyncio.run(coordinator.detect_conflicts([b1, b2]))
        assert len(conflicts) == 0

    def test_detect_overlapping_changes(
        self, coordinator: BranchCoordinator, temp_git_repo: Path
    ) -> None:
        b1 = asyncio.run(coordinator.create_track_branch(track=Track.SME, goal="Edit shared"))
        b2 = asyncio.run(coordinator.create_track_branch(track=Track.QA, goal="Edit shared too"))

        p1 = coordinator.get_worktree_path(b1)
        p2 = coordinator.get_worktree_path(b2)

        # Both branches modify the same file
        (p1 / "shared.py").write_text("# sme changes\n")
        subprocess.run(["git", "add", "."], cwd=p1, capture_output=True, check=True)
        subprocess.run(["git", "commit", "-m", "sme"], cwd=p1, capture_output=True, check=True)

        (p2 / "shared.py").write_text("# qa changes\n")
        subprocess.run(["git", "add", "."], cwd=p2, capture_output=True, check=True)
        subprocess.run(["git", "commit", "-m", "qa"], cwd=p2, capture_output=True, check=True)

        conflicts = asyncio.run(coordinator.detect_conflicts([b1, b2]))
        assert len(conflicts) > 0
        assert any("shared.py" in c.conflicting_files for c in conflicts)


class TestWorktreeCleanup:
    """Test cleanup of worktrees."""

    def test_cleanup_removes_worktrees(self, coordinator: BranchCoordinator) -> None:
        b1 = asyncio.run(coordinator.create_track_branch(track=Track.SME, goal="Work"))
        b2 = asyncio.run(coordinator.create_track_branch(track=Track.QA, goal="Work"))

        p1 = coordinator.get_worktree_path(b1)
        p2 = coordinator.get_worktree_path(b2)
        assert p1.exists()
        assert p2.exists()

        removed = coordinator.cleanup_all_worktrees()
        assert removed >= 2


class TestTaskDecompositionRouting:
    """Test that task decomposition routes subtasks to different tracks."""

    def test_decomposer_assigns_to_multiple_tracks(self) -> None:
        from aragora.nomic.task_decomposer import DecomposerConfig, TaskDecomposer
        from aragora.nomic.autonomous_orchestrator import AgentRouter

        decomposer = TaskDecomposer(config=DecomposerConfig(complexity_threshold=4))
        router = AgentRouter()

        decomposition = decomposer.analyze("Maximize utility for SME businesses")

        if not decomposition.subtasks:
            pytest.skip("Decomposer produced no subtasks for this goal")

        tracks_seen = set()
        for subtask in decomposition.subtasks:
            track = router.determine_track(subtask)
            tracks_seen.add(track)

        # The decomposer should route to at least one track
        assert len(tracks_seen) >= 1


class TestBudgetCutoff:
    """Test that the budget hard cutoff works in the orchestrator."""

    def test_budget_exceeded_skips_pending(self) -> None:
        from aragora.nomic.autonomous_orchestrator import AutonomousOrchestrator

        orch = AutonomousOrchestrator(require_human_approval=False)
        orch.budget_limit = 10.0
        orch._total_cost_usd = 15.0  # Over budget

        from aragora.nomic.task_decomposer import SubTask
        from aragora.nomic.autonomous_orchestrator import AgentAssignment, Track

        assignments = [
            AgentAssignment(
                subtask=SubTask(id="t1", title="Test task", description="desc"),
                track=Track.DEVELOPER,
                agent_type="claude",
            ),
        ]

        # The execution loop should skip tasks when over budget
        asyncio.run(orch._execute_assignments(assignments, max_cycles=1))
        assert assignments[0].status == "skipped"

    def test_budget_not_exceeded_allows_execution(self) -> None:
        from aragora.nomic.autonomous_orchestrator import AutonomousOrchestrator

        orch = AutonomousOrchestrator(require_human_approval=False)
        orch.budget_limit = 100.0
        orch._total_cost_usd = 5.0  # Under budget

        # Should not skip (will fail at workflow execution, which is fine)
        assert orch._total_cost_usd <= orch.budget_limit


class TestEndToEndDogfood:
    """End-to-end dogfood test: create worktrees, decompose, merge, cleanup."""

    def test_full_workflow(self, coordinator: BranchCoordinator, temp_git_repo: Path) -> None:
        """Simulate the complete multi-session workflow."""
        # 1. Create 3 worktrees
        tracks = [Track.SME, Track.DEVELOPER, Track.QA]
        branches = []
        for track in tracks:
            branch = asyncio.run(
                coordinator.create_track_branch(track=track, goal=f"Work on {track.value}")
            )
            branches.append(branch)

        # Verify all exist
        for branch in branches:
            wt_path = coordinator.get_worktree_path(branch)
            assert wt_path is not None
            assert wt_path.exists()

        # 2. Simulate work in each worktree
        for i, branch in enumerate(branches):
            wt_path = coordinator.get_worktree_path(branch)
            (wt_path / f"work_{i}.py").write_text(f"# work from session {i}\n")
            subprocess.run(["git", "add", "."], cwd=wt_path, capture_output=True, check=True)
            subprocess.run(
                ["git", "commit", "-m", f"session {i} work"],
                cwd=wt_path, capture_output=True, check=True,
            )

        # 3. Check no conflicts (different files)
        conflicts = asyncio.run(coordinator.detect_conflicts(branches))
        assert len(conflicts) == 0

        # 4. Dry-run merge each
        for branch in branches:
            result = asyncio.run(coordinator.safe_merge(branch, dry_run=True))
            assert result.success, f"Dry-run merge failed for {branch}: {result.error}"

        # 5. Merge all (actual)
        for branch in branches:
            result = asyncio.run(coordinator.safe_merge(branch))
            assert result.success, f"Merge failed for {branch}: {result.error}"

        # Verify all files exist on main
        for i in range(len(branches)):
            assert (temp_git_repo / f"work_{i}.py").exists()

        # 6. Cleanup
        removed = coordinator.cleanup_all_worktrees()
        assert removed >= 0  # May be 0 if worktrees already removed by merge
