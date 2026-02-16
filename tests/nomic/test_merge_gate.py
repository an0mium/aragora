"""Tests for MergeGate infrastructure in WorktreeManager and BranchCoordinator.

Verifies:
- MergeGateConfig and MergeGateResult dataclass behavior
- Pre-merge test failure aborts merge
- Successful merge with passing pre/post tests
- Post-merge test failure triggers auto-revert
- auto_revert=False skips revert on post-merge failure
- BranchCoordinator.safe_merge_with_gate end-to-end flow
"""

import subprocess
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.nomic.worktree_manager import (
    MergeGateConfig,
    MergeGateResult,
    WorktreeContext,
    WorktreeManager,
)
from aragora.nomic.branch_coordinator import (
    BranchCoordinator,
    BranchCoordinatorConfig,
    MergeResult,
)


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------


class TestMergeGateConfig:
    """Tests for MergeGateConfig dataclass."""

    def test_defaults(self):
        """Should have sensible defaults."""
        config = MergeGateConfig()
        assert config.pre_merge_test_paths == []
        assert config.post_merge_test_paths == []
        assert config.auto_revert is True
        assert config.test_timeout == 300
        assert config.require_pre_merge_pass is True
        assert config.require_post_merge_pass is True

    def test_custom_values(self):
        """Should accept custom values."""
        config = MergeGateConfig(
            pre_merge_test_paths=["tests/unit/"],
            post_merge_test_paths=["tests/integration/"],
            auto_revert=False,
            test_timeout=60,
            require_pre_merge_pass=False,
            require_post_merge_pass=False,
        )
        assert config.pre_merge_test_paths == ["tests/unit/"]
        assert config.post_merge_test_paths == ["tests/integration/"]
        assert config.auto_revert is False
        assert config.test_timeout == 60
        assert config.require_pre_merge_pass is False
        assert config.require_post_merge_pass is False


class TestMergeGateResult:
    """Tests for MergeGateResult dataclass."""

    def test_success_result(self):
        """Should represent a fully successful gate."""
        result = MergeGateResult(
            success=True,
            pre_merge_passed=True,
            merge_succeeded=True,
            post_merge_passed=True,
            commit_sha="abc123",
        )
        assert result.success is True
        assert result.reverted is False
        assert result.error is None

    def test_failure_result(self):
        """Should represent a failed gate with revert."""
        result = MergeGateResult(
            success=False,
            pre_merge_passed=True,
            merge_succeeded=True,
            post_merge_passed=False,
            reverted=True,
            commit_sha="abc123",
            error="Post-merge tests failed",
            post_merge_output="FAILED test_something",
        )
        assert result.success is False
        assert result.reverted is True
        assert result.error == "Post-merge tests failed"

    def test_defaults(self):
        """Should have correct defaults for optional fields."""
        result = MergeGateResult(success=False)
        assert result.pre_merge_passed is None
        assert result.merge_succeeded is None
        assert result.post_merge_passed is None
        assert result.reverted is False
        assert result.commit_sha is None
        assert result.error is None
        assert result.pre_merge_output == ""
        assert result.post_merge_output == ""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def git_repo(tmp_path):
    """Create a minimal git repo for testing."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(
        ["git", "init", "-b", "main"], cwd=repo, capture_output=True, check=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=repo, capture_output=True, check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=repo, capture_output=True, check=True,
    )
    (repo / "README.md").write_text("# Test")
    subprocess.run(["git", "add", "."], cwd=repo, capture_output=True, check=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=repo, capture_output=True, check=True,
    )
    return repo


@pytest.fixture
def repo_with_branch(git_repo):
    """Create a git repo with a feature branch containing a change."""
    repo = git_repo
    # Create a feature branch
    subprocess.run(
        ["git", "checkout", "-b", "feature/test"],
        cwd=repo, capture_output=True, check=True,
    )
    (repo / "feature.py").write_text("# Feature code\nprint('hello')\n")
    subprocess.run(["git", "add", "."], cwd=repo, capture_output=True, check=True)
    subprocess.run(
        ["git", "commit", "-m", "Add feature"],
        cwd=repo, capture_output=True, check=True,
    )
    # Switch back to main
    subprocess.run(
        ["git", "checkout", "main"],
        cwd=repo, capture_output=True, check=True,
    )
    return repo


@pytest.fixture
def manager(git_repo):
    """Create a WorktreeManager for the test repo."""
    return WorktreeManager(repo_path=git_repo, base_branch="main")


@pytest.fixture
def ctx(git_repo):
    """Create a WorktreeContext pointing at a feature branch."""
    return WorktreeContext(
        subtask_id="test-subtask-1",
        worktree_path=git_repo,  # Use repo itself as worktree for simplicity
        branch_name="feature/test",
        track="sme",
        agent_type="claude",
    )


# ---------------------------------------------------------------------------
# WorktreeManager.merge_with_gate tests
# ---------------------------------------------------------------------------


class TestMergeWithGatePreMerge:
    """Tests for pre-merge test phase."""

    @pytest.mark.asyncio
    async def test_pre_merge_failure_aborts_merge(self, manager, ctx):
        """Pre-merge test failure should abort the merge without merging."""
        config = MergeGateConfig(
            pre_merge_test_paths=["tests/unit/"],
            require_pre_merge_pass=True,
        )

        # Mock run_tests_in_worktree to return failure
        with patch.object(
            manager, "run_tests_in_worktree",
            return_value={"success": False, "output": "FAILED test_foo", "exit_code": 1, "duration": 1.0},
        ) as mock_tests, patch.object(manager, "_run_git") as mock_git:

            result = await manager.merge_with_gate(ctx, config)

        assert result.success is False
        assert result.pre_merge_passed is False
        assert result.merge_succeeded is None  # Never attempted
        assert result.error == "Pre-merge tests failed"
        assert "FAILED test_foo" in result.pre_merge_output
        # Merge should never have been called
        mock_git.assert_not_called()

    @pytest.mark.asyncio
    async def test_skip_pre_merge_when_no_paths(self, manager, ctx, repo_with_branch):
        """Should skip pre-merge tests when no test paths are configured."""
        manager.repo_path = repo_with_branch
        config = MergeGateConfig(
            pre_merge_test_paths=[],  # No pre-merge tests
            require_pre_merge_pass=True,
        )

        result = await manager.merge_with_gate(ctx, config)

        # Should have proceeded to merge (pre_merge_passed is None = skipped)
        assert result.pre_merge_passed is None
        assert result.merge_succeeded is True

    @pytest.mark.asyncio
    async def test_skip_pre_merge_when_not_required(self, manager, ctx, repo_with_branch):
        """Should skip pre-merge tests when require_pre_merge_pass=False."""
        manager.repo_path = repo_with_branch
        config = MergeGateConfig(
            pre_merge_test_paths=["tests/unit/"],
            require_pre_merge_pass=False,
        )

        result = await manager.merge_with_gate(ctx, config)

        # Should have proceeded to merge without running pre-merge tests
        assert result.pre_merge_passed is None
        assert result.merge_succeeded is True


class TestMergeWithGateSuccess:
    """Tests for successful merge gate flow."""

    @pytest.mark.asyncio
    async def test_full_success_no_tests(self, repo_with_branch):
        """Should succeed with default config (no test paths)."""
        manager = WorktreeManager(repo_path=repo_with_branch, base_branch="main")
        ctx_obj = WorktreeContext(
            subtask_id="test-1",
            worktree_path=repo_with_branch,
            branch_name="feature/test",
            track="sme",
            agent_type="claude",
        )

        result = await manager.merge_with_gate(ctx_obj)

        assert result.success is True
        assert result.merge_succeeded is True
        assert result.commit_sha is not None
        assert len(result.commit_sha) > 0
        assert ctx_obj.status == "completed"

    @pytest.mark.asyncio
    async def test_full_success_with_pre_merge(self, repo_with_branch):
        """Should succeed when pre-merge tests pass."""
        manager = WorktreeManager(repo_path=repo_with_branch, base_branch="main")
        ctx_obj = WorktreeContext(
            subtask_id="test-1",
            worktree_path=repo_with_branch,
            branch_name="feature/test",
            track="sme",
            agent_type="claude",
        )

        config = MergeGateConfig(
            pre_merge_test_paths=["tests/unit/"],
            require_pre_merge_pass=True,
        )

        with patch.object(
            manager, "run_tests_in_worktree",
            return_value={"success": True, "output": "1 passed", "exit_code": 0, "duration": 0.5},
        ):
            result = await manager.merge_with_gate(ctx_obj, config)

        assert result.success is True
        assert result.pre_merge_passed is True
        assert result.merge_succeeded is True

    @pytest.mark.asyncio
    async def test_full_success_with_post_merge(self, repo_with_branch):
        """Should succeed when both pre and post merge tests pass."""
        manager = WorktreeManager(repo_path=repo_with_branch, base_branch="main")
        ctx_obj = WorktreeContext(
            subtask_id="test-1",
            worktree_path=repo_with_branch,
            branch_name="feature/test",
            track="sme",
            agent_type="claude",
        )

        config = MergeGateConfig(
            pre_merge_test_paths=["tests/unit/"],
            post_merge_test_paths=["tests/integration/"],
            require_pre_merge_pass=True,
            require_post_merge_pass=True,
        )

        # Mock pre-merge tests pass, and subprocess for post-merge tests pass
        with patch.object(
            manager, "run_tests_in_worktree",
            return_value={"success": True, "output": "1 passed", "exit_code": 0, "duration": 0.5},
        ), patch("asyncio.to_thread") as mock_to_thread:
            mock_to_thread.return_value = MagicMock(
                returncode=0, stdout="1 passed\n", stderr="",
            )
            result = await manager.merge_with_gate(ctx_obj, config)

        assert result.success is True
        assert result.pre_merge_passed is True
        assert result.merge_succeeded is True
        assert result.post_merge_passed is True
        assert result.commit_sha is not None


class TestMergeWithGatePostMergeFailure:
    """Tests for post-merge test failure and auto-revert."""

    @pytest.mark.asyncio
    async def test_post_merge_failure_triggers_revert(self, repo_with_branch):
        """Post-merge test failure with auto_revert should revert the merge."""
        manager = WorktreeManager(repo_path=repo_with_branch, base_branch="main")
        ctx_obj = WorktreeContext(
            subtask_id="test-1",
            worktree_path=repo_with_branch,
            branch_name="feature/test",
            track="sme",
            agent_type="claude",
        )

        config = MergeGateConfig(
            post_merge_test_paths=["tests/integration/"],
            auto_revert=True,
            require_post_merge_pass=True,
        )

        # Post-merge tests will fail
        with patch("asyncio.to_thread") as mock_to_thread:
            mock_to_thread.return_value = MagicMock(
                returncode=1, stdout="FAILED\n", stderr="1 failed",
            )
            result = await manager.merge_with_gate(ctx_obj, config)

        assert result.success is False
        assert result.merge_succeeded is True
        assert result.post_merge_passed is False
        assert result.reverted is True
        assert result.commit_sha is not None
        assert result.error == "Post-merge tests failed"
        assert ctx_obj.status == "failed"

    @pytest.mark.asyncio
    async def test_post_merge_failure_no_revert(self, repo_with_branch):
        """Post-merge test failure with auto_revert=False should NOT revert."""
        manager = WorktreeManager(repo_path=repo_with_branch, base_branch="main")
        ctx_obj = WorktreeContext(
            subtask_id="test-1",
            worktree_path=repo_with_branch,
            branch_name="feature/test",
            track="sme",
            agent_type="claude",
        )

        config = MergeGateConfig(
            post_merge_test_paths=["tests/integration/"],
            auto_revert=False,
            require_post_merge_pass=True,
        )

        # Post-merge tests will fail
        with patch("asyncio.to_thread") as mock_to_thread:
            mock_to_thread.return_value = MagicMock(
                returncode=1, stdout="FAILED\n", stderr="1 failed",
            )
            # Also spy on _revert_merge to make sure it's NOT called
            with patch.object(manager, "_revert_merge") as mock_revert:
                result = await manager.merge_with_gate(ctx_obj, config)
                mock_revert.assert_not_called()

        assert result.success is False
        assert result.merge_succeeded is True
        assert result.post_merge_passed is False
        assert result.reverted is False
        assert result.error == "Post-merge tests failed"

    @pytest.mark.asyncio
    async def test_post_merge_timeout(self, repo_with_branch):
        """Post-merge test timeout should be treated as failure."""
        import asyncio as aio

        manager = WorktreeManager(repo_path=repo_with_branch, base_branch="main")
        ctx_obj = WorktreeContext(
            subtask_id="test-1",
            worktree_path=repo_with_branch,
            branch_name="feature/test",
            track="sme",
            agent_type="claude",
        )

        config = MergeGateConfig(
            post_merge_test_paths=["tests/slow/"],
            auto_revert=True,
            require_post_merge_pass=True,
            test_timeout=1,  # Very short timeout
        )

        with patch("asyncio.wait_for", side_effect=aio.TimeoutError()):
            result = await manager.merge_with_gate(ctx_obj, config)

        assert result.success is False
        assert result.post_merge_passed is False
        assert result.reverted is True


class TestMergeWithGateMergeFailure:
    """Tests for when the git merge itself fails."""

    @pytest.mark.asyncio
    async def test_merge_conflict_returns_failure(self, manager, ctx):
        """Merge conflict should return failure without post-merge tests."""
        config = MergeGateConfig(
            post_merge_test_paths=["tests/"],
        )

        with patch.object(manager, "_run_git") as mock_git:
            # checkout succeeds, merge fails, abort succeeds
            mock_git.side_effect = [
                MagicMock(returncode=0),  # checkout main
                MagicMock(returncode=1, stderr="CONFLICT"),  # merge --no-ff
                MagicMock(returncode=0),  # merge --abort
            ]

            result = await manager.merge_with_gate(ctx, config)

        assert result.success is False
        assert result.merge_succeeded is False
        assert result.post_merge_passed is None
        assert result.reverted is False
        assert ctx.status == "failed"


class TestRevertMerge:
    """Tests for _revert_merge method."""

    def test_successful_revert(self, repo_with_branch):
        """Should return True when revert succeeds."""
        manager = WorktreeManager(repo_path=repo_with_branch, base_branch="main")

        # Merge first to get a commit to revert
        subprocess.run(
            ["git", "merge", "--no-ff", "-m", "Merge feature", "feature/test"],
            cwd=repo_with_branch, capture_output=True, check=True,
        )
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_with_branch, capture_output=True, text=True, check=True,
        ).stdout.strip()

        result = manager._revert_merge(sha)
        assert result is True

    def test_failed_revert(self, manager):
        """Should return False when revert fails."""
        with patch.object(manager, "_run_git", return_value=MagicMock(returncode=1)):
            result = manager._revert_merge("nonexistent_sha")
        assert result is False


# ---------------------------------------------------------------------------
# BranchCoordinator.safe_merge_with_gate tests
# ---------------------------------------------------------------------------


class TestSafeMergeWithGate:
    """Tests for BranchCoordinator.safe_merge_with_gate."""

    @pytest.mark.asyncio
    async def test_nonexistent_branch(self):
        """Should fail for a branch that does not exist."""
        coordinator = BranchCoordinator(repo_path=Path("/tmp/fake-repo"))

        with patch.object(coordinator, "branch_exists", return_value=False):
            result = await coordinator.safe_merge_with_gate("nonexistent")

        assert result.success is False
        assert "does not exist" in result.error

    @pytest.mark.asyncio
    async def test_dry_run_conflict_aborts(self):
        """Should abort if dry-run merge detects conflicts."""
        coordinator = BranchCoordinator(repo_path=Path("/tmp/fake-repo"))

        with patch.object(coordinator, "branch_exists", return_value=True), \
             patch.object(coordinator, "_get_branch_files", return_value=[]), \
             patch.object(
                 coordinator, "safe_merge",
                 return_value=MergeResult(
                     source_branch="feature",
                     target_branch="main",
                     success=False,
                     conflicts=["file.py"],
                 ),
             ):
            result = await coordinator.safe_merge_with_gate("feature")

        assert result.success is False
        assert "conflicts" in result.error.lower()

    @pytest.mark.asyncio
    async def test_pre_merge_test_failure(self):
        """Should abort when pre-merge tests fail."""
        coordinator = BranchCoordinator(repo_path=Path("/tmp/fake-repo"))

        with patch.object(coordinator, "branch_exists", return_value=True), \
             patch.object(coordinator, "_get_branch_files", return_value=["aragora/foo/bar.py"]), \
             patch.object(
                 coordinator, "safe_merge",
                 return_value=MergeResult(
                     source_branch="feature",
                     target_branch="main",
                     success=True,
                 ),
             ), \
             patch("asyncio.to_thread") as mock_to_thread:
            # Pre-merge tests fail
            mock_to_thread.return_value = MagicMock(
                returncode=1, stdout="FAILED\n", stderr="",
            )
            result = await coordinator.safe_merge_with_gate(
                "feature", test_paths=["tests/foo/test_bar.py"],
            )

        assert result.success is False
        assert "pre-merge" in result.error.lower()

    @pytest.mark.asyncio
    async def test_post_merge_failure_with_revert(self):
        """Should revert on post-merge test failure when auto_revert=True."""
        coordinator = BranchCoordinator(repo_path=Path("/tmp/fake-repo"))

        call_count = 0

        async def mock_to_thread_fn(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Pre-merge: pass
                return MagicMock(returncode=0, stdout="OK", stderr="")
            else:
                # Post-merge: fail
                return MagicMock(returncode=1, stdout="FAILED", stderr="")

        with patch.object(coordinator, "branch_exists", return_value=True), \
             patch.object(coordinator, "_get_branch_files", return_value=[]), \
             patch.object(
                 coordinator, "safe_merge",
                 return_value=MergeResult(
                     source_branch="feature",
                     target_branch="main",
                     success=True,
                 ),
             ), \
             patch.object(coordinator, "_run_git") as mock_git, \
             patch("asyncio.to_thread", side_effect=mock_to_thread_fn):

            mock_git.return_value = MagicMock(returncode=0, stdout="abc123\n", stderr="")

            result = await coordinator.safe_merge_with_gate(
                "feature",
                test_paths=["tests/test_foo.py"],
                auto_revert=True,
            )

        assert result.success is False
        assert "post-merge" in result.error.lower()
        assert "(reverted)" in result.error

    @pytest.mark.asyncio
    async def test_post_merge_failure_no_revert(self):
        """Should NOT revert when auto_revert=False."""
        coordinator = BranchCoordinator(repo_path=Path("/tmp/fake-repo"))

        call_count = 0

        async def mock_to_thread_fn(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return MagicMock(returncode=0, stdout="OK", stderr="")
            else:
                return MagicMock(returncode=1, stdout="FAILED", stderr="")

        with patch.object(coordinator, "branch_exists", return_value=True), \
             patch.object(coordinator, "_get_branch_files", return_value=[]), \
             patch.object(
                 coordinator, "safe_merge",
                 return_value=MergeResult(
                     source_branch="feature",
                     target_branch="main",
                     success=True,
                 ),
             ), \
             patch.object(coordinator, "_run_git") as mock_git, \
             patch("asyncio.to_thread", side_effect=mock_to_thread_fn):

            mock_git.return_value = MagicMock(returncode=0, stdout="abc123\n", stderr="")

            result = await coordinator.safe_merge_with_gate(
                "feature",
                test_paths=["tests/test_foo.py"],
                auto_revert=False,
            )

        assert result.success is False
        assert "post-merge" in result.error.lower()
        assert "revert" not in result.error.lower()

    @pytest.mark.asyncio
    async def test_full_success(self):
        """Should succeed when all tests pass."""
        coordinator = BranchCoordinator(repo_path=Path("/tmp/fake-repo"))

        async def mock_to_thread_fn(*args, **kwargs):
            return MagicMock(returncode=0, stdout="OK", stderr="")

        with patch.object(coordinator, "branch_exists", return_value=True), \
             patch.object(coordinator, "_get_branch_files", return_value=[]), \
             patch.object(
                 coordinator, "safe_merge",
                 return_value=MergeResult(
                     source_branch="feature",
                     target_branch="main",
                     success=True,
                 ),
             ), \
             patch.object(coordinator, "_run_git") as mock_git, \
             patch("asyncio.to_thread", side_effect=mock_to_thread_fn):

            mock_git.return_value = MagicMock(returncode=0, stdout="abc123\n", stderr="")

            result = await coordinator.safe_merge_with_gate(
                "feature",
                test_paths=["tests/test_foo.py"],
            )

        assert result.success is True
        assert result.commit_sha == "abc123"

    @pytest.mark.asyncio
    async def test_no_test_paths_skips_tests(self):
        """Should skip test phases when no test_paths provided and inference returns empty."""
        coordinator = BranchCoordinator(repo_path=Path("/tmp/fake-repo"))

        with patch.object(coordinator, "branch_exists", return_value=True), \
             patch.object(coordinator, "_get_branch_files", return_value=[]), \
             patch.object(
                 coordinator, "safe_merge",
                 return_value=MergeResult(
                     source_branch="feature",
                     target_branch="main",
                     success=True,
                 ),
             ), \
             patch.object(coordinator, "_run_git") as mock_git, \
             patch(
                 "aragora.nomic.autonomous_orchestrator.AutonomousOrchestrator._infer_test_paths",
                 return_value=[],
             ):

            mock_git.return_value = MagicMock(returncode=0, stdout="abc123\n", stderr="")

            result = await coordinator.safe_merge_with_gate("feature")

        assert result.success is True

    @pytest.mark.asyncio
    async def test_infer_test_paths_fallback(self):
        """Should use _infer_test_paths when test_paths not provided."""
        coordinator = BranchCoordinator(repo_path=Path("/tmp/fake-repo"))

        async def mock_to_thread_fn(*args, **kwargs):
            return MagicMock(returncode=0, stdout="OK", stderr="")

        with patch.object(coordinator, "branch_exists", return_value=True), \
             patch.object(
                 coordinator, "_get_branch_files",
                 return_value=["aragora/nomic/worktree_manager.py"],
             ), \
             patch.object(
                 coordinator, "safe_merge",
                 return_value=MergeResult(
                     source_branch="feature",
                     target_branch="main",
                     success=True,
                 ),
             ), \
             patch.object(coordinator, "_run_git") as mock_git, \
             patch("asyncio.to_thread", side_effect=mock_to_thread_fn) as mock_thread:

            mock_git.return_value = MagicMock(returncode=0, stdout="abc123\n", stderr="")

            result = await coordinator.safe_merge_with_gate("feature")

        # Should have inferred test path and run tests
        assert result.success is True
        # Verify pytest was called with inferred paths
        assert mock_thread.call_count >= 1  # At least pre-merge tests


class TestSafeMergeWithGateIntegration:
    """Integration tests using real git repos."""

    @pytest.mark.asyncio
    async def test_real_merge_no_tests(self, repo_with_branch):
        """Should perform a real merge when no tests are configured."""
        coordinator = BranchCoordinator(
            repo_path=repo_with_branch,
            config=BranchCoordinatorConfig(base_branch="main"),
        )

        result = await coordinator.safe_merge_with_gate(
            "feature/test",
            test_paths=[],  # Explicit empty = skip tests
        )

        assert result.success is True
        assert result.commit_sha is not None

        # Verify merge happened
        log = subprocess.run(
            ["git", "log", "--oneline", "-5"],
            cwd=repo_with_branch, capture_output=True, text=True,
        )
        assert "Merge feature/test into main" in log.stdout
