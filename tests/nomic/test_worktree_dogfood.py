"""End-to-end worktree workflow tests.

Validates the integration between TaskDecomposer, BranchCoordinator,
safe_merge, and cleanup -- the full worktree lifecycle that the Nomic
Loop uses for parallel development.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from aragora.nomic.branch_coordinator import (
    BranchCoordinator,
    BranchCoordinatorConfig,
    MergeResult,
)
from aragora.nomic.meta_planner import Track
from aragora.nomic.task_decomposer import DecomposerConfig, TaskDecomposer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_git_success(stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess:
    """Create a successful subprocess.CompletedProcess."""
    return subprocess.CompletedProcess(
        args=["git"], returncode=0, stdout=stdout, stderr=stderr,
    )


def _make_git_failure(stderr: str = "") -> subprocess.CompletedProcess:
    """Create a failed subprocess.CompletedProcess."""
    return subprocess.CompletedProcess(
        args=["git"], returncode=1, stdout="", stderr=stderr,
    )


TRACKS = [Track.SME, Track.DEVELOPER, Track.QA]
GOALS = [
    "Improve SME onboarding flow",
    "Add SDK pagination helpers",
    "Increase handler test coverage",
]


def _build_coordinator(tmp_path: Path) -> BranchCoordinator:
    """Build a BranchCoordinator pointing at a temporary directory."""
    config = BranchCoordinatorConfig(
        base_branch="main",
        branch_prefix="dev",
        use_worktrees=True,
        max_parallel_branches=3,
        worktree_base_dir=tmp_path / ".worktrees",
    )
    return BranchCoordinator(repo_path=tmp_path, config=config)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_create_worktrees_for_tracks(tmp_path: Path) -> None:
    """Creating worktrees for three tracks produces distinct branches and paths."""
    coordinator = _build_coordinator(tmp_path)

    with patch("subprocess.run", return_value=_make_git_success()):
        branches: list[str] = []
        for track, goal in zip(TRACKS, GOALS):
            branch = await coordinator.create_track_branch(track, goal)
            branches.append(branch)

    # Each branch name should contain the track value
    for branch, track in zip(branches, TRACKS):
        assert track.value in branch, f"Branch {branch} missing track {track.value}"

    # All branches are unique
    assert len(set(branches)) == 3

    # Worktree paths are recorded and distinct
    paths = [coordinator.get_worktree_path(b) for b in branches]
    assert all(p is not None for p in paths)
    assert len(set(str(p) for p in paths)) == 3


def test_decomposer_assigns_different_tracks() -> None:
    """TaskDecomposer on 'Maximize utility for SMEs' produces subtasks spanning tracks.

    Uses threshold=4 so the vague-goal expansion path fires (score=3 < 4),
    which cross-references development track configs and populates file_scope.
    """
    config = DecomposerConfig(complexity_threshold=4, max_subtasks=5)
    decomposer = TaskDecomposer(config=config)

    result = decomposer.analyze("Maximize utility for SMEs")

    assert result.should_decompose, "Vague strategic goal should trigger decomposition"
    assert len(result.subtasks) >= 2, "Should produce at least 2 subtasks"

    # Subtasks should have distinct titles (no duplicates)
    titles = [s.title for s in result.subtasks]
    assert len(titles) == len(set(titles)), f"Duplicate subtask titles: {titles}"

    # Each subtask should have an id and description
    for subtask in result.subtasks:
        assert subtask.id, "Subtask must have an id"
        assert subtask.description, "Subtask must have a description"

    # At least some subtasks should have file_scope set (from track expansion)
    scoped = [s for s in result.subtasks if s.file_scope]
    assert len(scoped) >= 1, "At least one subtask should have file_scope populated"


@pytest.mark.asyncio
async def test_worktree_paths_are_isolated(tmp_path: Path) -> None:
    """Each worktree lives under .worktrees/ with a unique directory name."""
    coordinator = _build_coordinator(tmp_path)
    worktree_base = tmp_path / ".worktrees"

    with patch("subprocess.run", return_value=_make_git_success()):
        for track, goal in zip(TRACKS, GOALS):
            await coordinator.create_track_branch(track, goal)

    for _branch, info in coordinator._active_worktrees.items():
        assert info.worktree_path.parent == worktree_base
        assert info.created_at is not None


@pytest.mark.asyncio
async def test_safe_merge_dry_run(tmp_path: Path) -> None:
    """safe_merge with dry_run=True checks mergeability without committing."""
    coordinator = _build_coordinator(tmp_path)

    def _side_effect(*args, **kwargs):
        cmd = args[0] if args else kwargs.get("args", [])
        if "rev-parse" in cmd and "--verify" in cmd:
            # branch_exists check -- report branch exists
            return _make_git_success()
        if "merge" in cmd and "--no-commit" in cmd:
            # dry-run merge succeeds
            return _make_git_success()
        if "merge" in cmd and "--abort" in cmd:
            return _make_git_success()
        return _make_git_success()

    with patch("subprocess.run", side_effect=_side_effect):
        result: MergeResult = await coordinator.safe_merge(
            "dev/sme-improve-sme-onboarding-flow-0216",
            target="main",
            dry_run=True,
        )

    assert result.success is True
    assert result.source_branch == "dev/sme-improve-sme-onboarding-flow-0216"
    assert result.target_branch == "main"
    # dry_run should not produce a commit SHA
    assert result.commit_sha is None


@pytest.mark.asyncio
async def test_safe_merge_dry_run_detects_conflicts(tmp_path: Path) -> None:
    """safe_merge dry_run reports conflicts when git merge --no-commit fails."""
    coordinator = _build_coordinator(tmp_path)

    def _side_effect(*args, **kwargs):
        cmd = args[0] if args else kwargs.get("args", [])
        if "rev-parse" in cmd and "--verify" in cmd:
            return _make_git_success()
        if "merge" in cmd and "--no-commit" in cmd:
            return _make_git_failure(
                stderr="CONFLICT (content): Merge conflict in aragora/server/app.py"
            )
        if "merge" in cmd and "--abort" in cmd:
            return _make_git_success()
        return _make_git_success()

    with patch("subprocess.run", side_effect=_side_effect):
        result = await coordinator.safe_merge("dev/feature-x", dry_run=True)

    assert result.success is False
    assert "aragora/server/app.py" in result.conflicts


@pytest.mark.asyncio
async def test_cleanup_removes_worktrees(tmp_path: Path) -> None:
    """cleanup_all_worktrees removes all tracked worktrees and returns count."""
    coordinator = _build_coordinator(tmp_path)

    with patch("subprocess.run", return_value=_make_git_success()):
        for track, goal in zip(TRACKS, GOALS):
            await coordinator.create_track_branch(track, goal)

    assert len(coordinator._worktree_paths) == 3

    with patch("subprocess.run", return_value=_make_git_success()):
        removed = coordinator.cleanup_all_worktrees()

    assert removed == 3
    assert len(coordinator._worktree_paths) == 0
    assert len(coordinator._active_worktrees) == 0


@pytest.mark.asyncio
async def test_end_to_end_worktree_workflow(tmp_path: Path) -> None:
    """Full lifecycle: decompose -> create worktrees -> merge -> cleanup."""
    # Step 1: Decompose a strategic goal into subtasks
    # threshold=4 triggers vague-goal expansion (score=3 < 4) for richer subtasks
    config = DecomposerConfig(complexity_threshold=4, max_subtasks=5)
    decomposer = TaskDecomposer(config=config)
    decomposition = decomposer.analyze("Maximize utility for SMEs")

    assert decomposition.should_decompose
    subtask_count = len(decomposition.subtasks)
    assert subtask_count >= 2

    # Step 2: Create worktrees for each subtask (map to tracks round-robin)
    coordinator = _build_coordinator(tmp_path)
    track_cycle = [Track.SME, Track.DEVELOPER, Track.QA, Track.CORE, Track.SECURITY]
    branches: list[str] = []

    with patch("subprocess.run", return_value=_make_git_success()):
        for i, subtask in enumerate(decomposition.subtasks):
            track = track_cycle[i % len(track_cycle)]
            branch = await coordinator.create_track_branch(track, subtask.title)
            branches.append(branch)

    assert len(branches) == subtask_count

    # Step 3: Verify worktree isolation
    paths = [coordinator.get_worktree_path(b) for b in branches]
    assert len(set(str(p) for p in paths)) == subtask_count

    # Step 4: Dry-run merge each branch
    def _merge_side_effect(*args, **kwargs):
        cmd = args[0] if args else kwargs.get("args", [])
        if "rev-parse" in cmd and "--verify" in cmd:
            return _make_git_success()
        if "merge" in cmd and "--no-commit" in cmd:
            return _make_git_success()
        return _make_git_success()

    with patch("subprocess.run", side_effect=_merge_side_effect):
        for branch in branches:
            result = await coordinator.safe_merge(branch, dry_run=True)
            assert result.success is True, f"Dry-run merge failed for {branch}"

    # Step 5: Cleanup all worktrees
    with patch("subprocess.run", return_value=_make_git_success()):
        removed = coordinator.cleanup_all_worktrees()

    assert removed == subtask_count
    assert len(coordinator._worktree_paths) == 0
    assert len(coordinator._active_worktrees) == 0
