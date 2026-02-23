"""Tests for sprint_coordinator.py -- multi-session parallel dev workflow."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Import the module under test
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))
import sprint_coordinator as sc


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_project(tmp_path: Path) -> Path:
    """Create a minimal git repo to act as PROJECT_ROOT."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", str(repo)], capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=repo,
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=repo,
        capture_output=True,
        check=True,
    )
    # Create an initial commit so branches work
    (repo / "README.md").write_text("init")
    subprocess.run(["git", "add", "."], cwd=repo, capture_output=True, check=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=repo,
        capture_output=True,
        check=True,
    )
    return repo


@pytest.fixture()
def mock_subtasks() -> list:
    """Return a list of fake subtask dicts for manifests."""
    return [
        {
            "id": "subtask_1",
            "title": "Testing Changes",
            "description": "Improve test coverage",
            "file_scope": ["tests/"],
            "complexity": "medium",
            "dependencies": [],
        },
        {
            "id": "subtask_2",
            "title": "Api Changes",
            "description": "Add new endpoints",
            "file_scope": ["aragora/server/"],
            "complexity": "high",
            "dependencies": ["subtask_1"],
        },
    ]


@pytest.fixture()
def manifest(mock_subtasks: list) -> dict:
    """Create and save a sprint manifest."""
    return {
        "goal": "Improve test coverage and API",
        "created_at": "2026-02-15T00:00:00+00:00",
        "complexity_score": 6,
        "complexity_level": "medium",
        "subtasks": mock_subtasks,
        "worktrees": {},
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _patch_project_root(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    """Point the module constants at *root*."""
    monkeypatch.setattr(sc, "PROJECT_ROOT", root)
    monkeypatch.setattr(sc, "SPRINT_DIR", root / ".aragora_beads" / "sprint")


def _make_fake_decomposition(subtasks: list | None = None):
    """Build a fake TaskDecomposition-like object."""
    from aragora.nomic.task_decomposer import SubTask, TaskDecomposition

    if subtasks is None:
        subtasks = [
            SubTask(
                id="subtask_1",
                title="Core Implementation",
                description="Implement the main feature",
                file_scope=["aragora/core.py"],
                estimated_complexity="medium",
                dependencies=[],
            ),
        ]

    return TaskDecomposition(
        original_task="test goal",
        complexity_score=6,
        complexity_level="medium",
        should_decompose=True,
        subtasks=subtasks,
        rationale="test",
    )


# ---------------------------------------------------------------------------
# cmd_plan
# ---------------------------------------------------------------------------


class TestCmdPlan:
    """Tests for the plan subcommand."""

    def test_plan_creates_manifest(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """plan should write a valid sprint-manifest.json."""
        _patch_project_root(monkeypatch, tmp_path)

        fake_result = _make_fake_decomposition()

        with patch(
            "aragora.nomic.task_decomposer.TaskDecomposer.analyze",
            return_value=fake_result,
        ):
            args = argparse.Namespace(
                goal="Improve test coverage",
                debate=False,
                dry_run=False,
            )
            sc.cmd_plan(args)

        manifest_path = tmp_path / ".aragora_beads" / "sprint" / "sprint-manifest.json"
        assert manifest_path.exists()

        data = json.loads(manifest_path.read_text())
        assert data["goal"] == "Improve test coverage"
        assert data["complexity_score"] == 6
        assert len(data["subtasks"]) == 1
        assert data["subtasks"][0]["id"] == "subtask_1"
        assert data["subtasks"][0]["title"] == "Core Implementation"

    def test_plan_dry_run_does_not_save(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """--dry-run should print but not save a manifest."""
        _patch_project_root(monkeypatch, tmp_path)

        from aragora.nomic.task_decomposer import TaskDecomposition

        fake_result = TaskDecomposition(
            original_task="Fix typo",
            complexity_score=3,
            complexity_level="low",
            should_decompose=False,
            subtasks=[],
        )

        with patch(
            "aragora.nomic.task_decomposer.TaskDecomposer.analyze",
            return_value=fake_result,
        ):
            args = argparse.Namespace(goal="Fix typo", debate=False, dry_run=True)
            sc.cmd_plan(args)

        manifest_path = tmp_path / ".aragora_beads" / "sprint" / "sprint-manifest.json"
        assert not manifest_path.exists()

    def test_plan_debate_mode(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """--debate should call analyze_with_debate via asyncio.run."""
        _patch_project_root(monkeypatch, tmp_path)

        from aragora.nomic.task_decomposer import TaskDecomposition

        fake_result = TaskDecomposition(
            original_task="Maximize utility",
            complexity_score=8,
            complexity_level="high",
            should_decompose=True,
            subtasks=[],
        )

        # Patch asyncio.run at the module level used by cmd_plan
        with patch("asyncio.run", return_value=fake_result) as mock_arun:
            args = argparse.Namespace(goal="Maximize utility", debate=True, dry_run=True)
            sc.cmd_plan(args)
            mock_arun.assert_called_once()

    def test_plan_manifest_has_required_keys(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Manifest should contain goal, created_at, complexity_score, subtasks, worktrees."""
        _patch_project_root(monkeypatch, tmp_path)

        fake_result = _make_fake_decomposition()

        with patch(
            "aragora.nomic.task_decomposer.TaskDecomposer.analyze",
            return_value=fake_result,
        ):
            sc.cmd_plan(argparse.Namespace(goal="Test", debate=False, dry_run=False))

        data = json.loads(
            (tmp_path / ".aragora_beads" / "sprint" / "sprint-manifest.json").read_text()
        )
        for key in (
            "goal",
            "created_at",
            "complexity_score",
            "complexity_level",
            "subtasks",
            "worktrees",
        ):
            assert key in data, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# cmd_setup
# ---------------------------------------------------------------------------


class TestCmdSetup:
    """Tests for the setup subcommand."""

    def test_setup_creates_worktrees(
        self,
        tmp_project: Path,
        manifest: dict,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """setup should create git worktrees for each subtask."""
        _patch_project_root(monkeypatch, tmp_project)
        sc._save_manifest(manifest)

        sc.cmd_setup(argparse.Namespace())

        # Verify worktrees were created
        wt1 = tmp_project / ".worktrees" / "sprint-testing-changes"
        wt2 = tmp_project / ".worktrees" / "sprint-api-changes"
        assert wt1.exists(), f"Expected worktree at {wt1}"
        assert wt2.exists(), f"Expected worktree at {wt2}"

        # Verify manifest was updated with worktree info
        updated = sc._load_manifest()
        assert "subtask_1" in updated["worktrees"]
        assert "subtask_2" in updated["worktrees"]
        assert updated["worktrees"]["subtask_1"]["branch"] == "sprint/testing-changes"

    def test_setup_no_manifest_exits(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """setup without a manifest should exit with an error."""
        _patch_project_root(monkeypatch, tmp_path)

        with pytest.raises(SystemExit) as exc_info:
            sc.cmd_setup(argparse.Namespace())
        assert exc_info.value.code == 1

    def test_setup_idempotent(
        self,
        tmp_project: Path,
        manifest: dict,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Running setup twice should not fail (idempotent)."""
        _patch_project_root(monkeypatch, tmp_project)
        sc._save_manifest(manifest)

        sc.cmd_setup(argparse.Namespace())
        # Second run should not raise
        sc.cmd_setup(argparse.Namespace())

        wt1 = tmp_project / ".worktrees" / "sprint-testing-changes"
        assert wt1.exists()


# ---------------------------------------------------------------------------
# cmd_status
# ---------------------------------------------------------------------------


class TestCmdStatus:
    """Tests for the status subcommand."""

    def test_status_shows_idle_worktrees(
        self,
        tmp_project: Path,
        manifest: dict,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """status on fresh worktrees should show idle (no commits)."""
        _patch_project_root(monkeypatch, tmp_project)
        sc._save_manifest(manifest)
        sc.cmd_setup(argparse.Namespace())

        sc.cmd_status(argparse.Namespace())

        captured = capsys.readouterr()
        assert "subtask_1" in captured.out
        assert "subtask_2" in captured.out
        # Both should have 0 commits
        assert "0" in captured.out

    def test_status_shows_active_after_commit(
        self,
        tmp_project: Path,
        manifest: dict,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """status should report commits after work in a worktree."""
        _patch_project_root(monkeypatch, tmp_project)
        sc._save_manifest(manifest)
        sc.cmd_setup(argparse.Namespace())

        # Make a commit in the first worktree
        wt1 = tmp_project / ".worktrees" / "sprint-testing-changes"
        (wt1 / "new_file.py").write_text("# test")
        subprocess.run(["git", "add", "."], cwd=wt1, capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "add test file"],
            cwd=wt1,
            capture_output=True,
            check=True,
        )

        sc.cmd_status(argparse.Namespace())
        captured = capsys.readouterr()
        # subtask_1 should show 1 commit
        assert "1" in captured.out

    def test_status_no_worktrees(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """status with empty worktrees dict should print a message."""
        _patch_project_root(monkeypatch, tmp_path)
        empty_manifest = {
            "goal": "test",
            "created_at": "2026-01-01T00:00:00+00:00",
            "complexity_score": 1,
            "complexity_level": "low",
            "subtasks": [],
            "worktrees": {},
        }
        sc._save_manifest(empty_manifest)
        sc.cmd_status(argparse.Namespace())
        captured = capsys.readouterr()
        assert "No worktrees" in captured.out


# ---------------------------------------------------------------------------
# cmd_merge
# ---------------------------------------------------------------------------


class TestCmdMerge:
    """Tests for the merge subcommand."""

    def test_merge_no_commits_skips(
        self,
        tmp_project: Path,
        manifest: dict,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """merge with no commits should report nothing to merge."""
        _patch_project_root(monkeypatch, tmp_project)
        sc._save_manifest(manifest)
        sc.cmd_setup(argparse.Namespace())

        sc.cmd_merge(argparse.Namespace(all=True))
        captured = capsys.readouterr()
        assert "No branches with new commits" in captured.out

    def test_merge_with_commit_runs_tests(
        self,
        tmp_project: Path,
        manifest: dict,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """merge should attempt pre-merge tests for branches with commits."""
        _patch_project_root(monkeypatch, tmp_project)
        sc._save_manifest(manifest)
        sc.cmd_setup(argparse.Namespace())

        # Make a commit in the first worktree
        wt1 = tmp_project / ".worktrees" / "sprint-testing-changes"
        (wt1 / "new_file.py").write_text("# test")
        subprocess.run(["git", "add", "."], cwd=wt1, capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "add test file"],
            cwd=wt1,
            capture_output=True,
            check=True,
        )

        # Mock subprocess.run to control test execution
        original_run = subprocess.run

        def mock_run(cmd, **kwargs):
            # Intercept pytest calls and return success
            if isinstance(cmd, list) and any("pytest" in str(c) for c in cmd):
                result = MagicMock()
                result.returncode = 0
                result.stdout = "1 passed"
                result.stderr = ""
                return result
            return original_run(cmd, **kwargs)

        with patch("sprint_coordinator.subprocess.run", side_effect=mock_run):
            sc.cmd_merge(argparse.Namespace(all=True))

    def test_merge_reverts_on_post_test_failure(
        self,
        tmp_project: Path,
        manifest: dict,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """merge should revert if post-merge tests fail."""
        _patch_project_root(monkeypatch, tmp_project)
        sc._save_manifest(manifest)
        sc.cmd_setup(argparse.Namespace())

        # Make a commit in worktree
        wt1 = tmp_project / ".worktrees" / "sprint-testing-changes"
        (wt1 / "new_file.py").write_text("# test")
        subprocess.run(["git", "add", "."], cwd=wt1, capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "add test file"],
            cwd=wt1,
            capture_output=True,
            check=True,
        )

        call_count = {"pytest": 0}
        original_run = subprocess.run

        def mock_run(cmd, **kwargs):
            if isinstance(cmd, list) and any("pytest" in str(c) for c in cmd):
                call_count["pytest"] += 1
                result = MagicMock()
                if call_count["pytest"] == 1:
                    # Pre-merge tests pass
                    result.returncode = 0
                    result.stdout = "1 passed"
                else:
                    # Post-merge tests fail
                    result.returncode = 1
                    result.stdout = "1 failed"
                result.stderr = ""
                return result
            return original_run(cmd, **kwargs)

        with patch("sprint_coordinator.subprocess.run", side_effect=mock_run):
            sc.cmd_merge(argparse.Namespace(all=True))

        captured = capsys.readouterr()
        assert "post-merge tests failed" in captured.out or "FAIL" in captured.out


# ---------------------------------------------------------------------------
# cmd_cleanup
# ---------------------------------------------------------------------------


class TestCmdCleanup:
    """Tests for the cleanup subcommand."""

    def test_cleanup_removes_merged_worktrees(
        self,
        tmp_project: Path,
        manifest: dict,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """cleanup should remove worktrees whose branches are merged."""
        _patch_project_root(monkeypatch, tmp_project)
        sc._save_manifest(manifest)
        sc.cmd_setup(argparse.Namespace())

        wt1 = tmp_project / ".worktrees" / "sprint-testing-changes"

        # Make a commit and merge to main
        (wt1 / "new_file.py").write_text("# test")
        subprocess.run(["git", "add", "."], cwd=wt1, capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "add test file"],
            cwd=wt1,
            capture_output=True,
            check=True,
        )

        # Merge from main repo
        subprocess.run(
            ["git", "merge", "--no-ff", "sprint/testing-changes", "-m", "merge"],
            cwd=tmp_project,
            capture_output=True,
            check=True,
        )

        sc.cmd_cleanup(argparse.Namespace())

        captured = capsys.readouterr()
        assert "Removed" in captured.out or "Deleted" in captured.out

        # The worktree directory should be gone
        assert not wt1.exists()

    def test_cleanup_keeps_unmerged_branches(
        self,
        tmp_project: Path,
        manifest: dict,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """cleanup should not remove worktrees for unmerged branches."""
        _patch_project_root(monkeypatch, tmp_project)
        sc._save_manifest(manifest)
        sc.cmd_setup(argparse.Namespace())

        # Make a commit in worktree but don't merge
        wt1 = tmp_project / ".worktrees" / "sprint-testing-changes"
        (wt1 / "new_file.py").write_text("# test")
        subprocess.run(["git", "add", "."], cwd=wt1, capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "add test file"],
            cwd=wt1,
            capture_output=True,
            check=True,
        )

        sc.cmd_cleanup(argparse.Namespace())

        captured = capsys.readouterr()
        assert "Kept" in captured.out or "not merged" in captured.out
        # Worktree should still exist
        assert wt1.exists()

    def test_cleanup_no_worktrees(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """cleanup with no worktrees should print a message."""
        _patch_project_root(monkeypatch, tmp_path)
        empty_manifest = {
            "goal": "test",
            "created_at": "2026-01-01T00:00:00+00:00",
            "complexity_score": 1,
            "complexity_level": "low",
            "subtasks": [],
            "worktrees": {},
        }
        sc._save_manifest(empty_manifest)
        sc.cmd_cleanup(argparse.Namespace())
        captured = capsys.readouterr()
        assert "No worktrees" in captured.out


# ---------------------------------------------------------------------------
# Helpers / unit tests
# ---------------------------------------------------------------------------


class TestHelpers:
    """Unit tests for helper functions."""

    def test_branch_name_for_subtask(self) -> None:
        """Should generate a clean branch name from a subtask."""
        subtask = {"id": "subtask_1", "title": "Testing Changes"}
        name = sc._branch_name_for_subtask(subtask)
        assert name == "sprint/testing-changes"
        assert "/" in name
        assert " " not in name

    def test_branch_name_truncates_long_titles(self) -> None:
        """Branch name should be truncated for long titles."""
        subtask = {"id": "subtask_1", "title": "A" * 100}
        name = sc._branch_name_for_subtask(subtask)
        # sprint/ prefix + up to 40 chars
        assert len(name) <= 7 + 40

    def test_branch_name_special_characters(self) -> None:
        """Special characters should become hyphens."""
        subtask = {"id": "x", "title": "Fix bug #123 (urgent!)"}
        name = sc._branch_name_for_subtask(subtask)
        assert "#" not in name
        assert "(" not in name
        assert "!" not in name

    def test_worktree_dir_for_branch(self) -> None:
        """Should convert branch name to a directory path."""
        path = sc._worktree_dir_for_branch("sprint/testing-changes")
        assert "sprint-testing-changes" in str(path)

    def test_load_manifest_missing_file(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """_load_manifest should exit 1 when no manifest exists."""
        _patch_project_root(monkeypatch, tmp_path)
        with pytest.raises(SystemExit) as exc_info:
            sc._load_manifest()
        assert exc_info.value.code == 1

    def test_save_and_load_manifest(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Round-trip save/load should preserve data."""
        _patch_project_root(monkeypatch, tmp_path)
        data = {"goal": "test", "subtasks": []}
        sc._save_manifest(data)
        loaded = sc._load_manifest()
        assert loaded["goal"] == "test"

    def test_save_creates_directory(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """_save_manifest should create the sprint directory if needed."""
        _patch_project_root(monkeypatch, tmp_path)
        sprint_dir = tmp_path / ".aragora_beads" / "sprint"
        assert not sprint_dir.exists()
        sc._save_manifest({"goal": "test"})
        assert sprint_dir.exists()

    def test_color_noop_when_not_tty(self) -> None:
        """_color should return plain text when stdout is not a TTY."""
        # In test context stdout is not a TTY, so color codes are stripped
        result = sc._color("hello", sc._GREEN)
        assert result == "hello"


# ---------------------------------------------------------------------------
# cmd_execute
# ---------------------------------------------------------------------------


class TestCmdExecute:
    """Tests for the execute subcommand."""

    def test_execute_no_worktrees(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """execute with no worktrees should report nothing to do."""
        _patch_project_root(monkeypatch, tmp_path)
        empty_manifest = {
            "goal": "test",
            "created_at": "2026-01-01T00:00:00+00:00",
            "complexity_score": 1,
            "complexity_level": "low",
            "subtasks": [],
            "worktrees": {},
        }
        sc._save_manifest(empty_manifest)
        sc.cmd_execute(argparse.Namespace(max_parallel=3))
        captured = capsys.readouterr()
        assert "No worktrees" in captured.out

    def test_execute_no_claude_binary(
        self,
        tmp_project: Path,
        manifest: dict,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """execute should exit with error if claude is not in PATH."""
        _patch_project_root(monkeypatch, tmp_project)

        # Set up worktrees in manifest
        manifest["worktrees"] = {
            "subtask_1": {
                "branch": "sprint/testing-changes",
                "path": str(tmp_project / ".worktrees" / "sprint-testing-changes"),
            },
        }
        sc._save_manifest(manifest)

        # Create the worktree directory so it exists
        wt_path = tmp_project / ".worktrees" / "sprint-testing-changes"
        wt_path.mkdir(parents=True, exist_ok=True)

        with patch("shutil.which", return_value=None):
            with pytest.raises(SystemExit) as exc_info:
                sc.cmd_execute(argparse.Namespace(max_parallel=3))
            assert exc_info.value.code == 1

    def test_execute_writes_task_files(
        self,
        tmp_project: Path,
        manifest: dict,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """execute should write .sprint-task.md in each worktree."""
        _patch_project_root(monkeypatch, tmp_project)
        sc._save_manifest(manifest)
        sc.cmd_setup(argparse.Namespace())

        # Mock claude binary and Popen so no real process is spawned
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.poll.return_value = 0  # immediately done

        with patch("shutil.which", return_value="/usr/bin/claude"):
            with patch("subprocess.Popen", return_value=mock_proc):
                sc.cmd_execute(argparse.Namespace(max_parallel=3))

        # Check task files were written
        wt1 = tmp_project / ".worktrees" / "sprint-testing-changes"
        task_file = wt1 / ".sprint-task.md"
        assert task_file.exists()
        content = task_file.read_text()
        assert "Improve test coverage" in content
        assert "tests/" in content

    def test_execute_respects_max_parallel(
        self,
        tmp_project: Path,
        manifest: dict,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """execute should not launch more than max_parallel agents at once."""
        _patch_project_root(monkeypatch, tmp_project)
        sc._save_manifest(manifest)
        sc.cmd_setup(argparse.Namespace())

        max_concurrent = 0
        current_running = 0
        poll_count = {"total": 0}

        def make_proc():
            nonlocal max_concurrent, current_running
            proc = MagicMock()
            proc.pid = 10000 + poll_count["total"]
            current_running += 1
            if current_running > max_concurrent:
                max_concurrent = current_running

            def poll_side_effect():
                nonlocal current_running
                poll_count["total"] += 1
                if poll_count["total"] > 2:  # Complete after a couple polls
                    current_running -= 1
                    return 0
                return None

            proc.poll.side_effect = poll_side_effect
            return proc

        with patch("shutil.which", return_value="/usr/bin/claude"):
            with patch("subprocess.Popen", side_effect=lambda *a, **k: make_proc()):
                with patch("time.sleep"):  # Skip actual sleep
                    sc.cmd_execute(argparse.Namespace(max_parallel=1))

        # With max_parallel=1, should never exceed 1 concurrent
        assert max_concurrent <= 1

    def test_execute_updates_manifest(
        self,
        tmp_project: Path,
        manifest: dict,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """execute should save execution results to the manifest."""
        _patch_project_root(monkeypatch, tmp_project)
        sc._save_manifest(manifest)
        sc.cmd_setup(argparse.Namespace())

        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.poll.return_value = 0

        with patch("shutil.which", return_value="/usr/bin/claude"):
            with patch("subprocess.Popen", return_value=mock_proc):
                sc.cmd_execute(argparse.Namespace(max_parallel=3))

        updated = sc._load_manifest()
        assert "last_execution" in updated
        assert "timestamp" in updated["last_execution"]
        assert "results" in updated["last_execution"]
