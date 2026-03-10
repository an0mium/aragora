"""Regression tests for exit-141 salvage staging (#901).

Covers the bug where a worker exits with SIGPIPE (141) leaving valid but
unstaged changes in the worktree.  The salvage path must stage those
changes and produce a concrete commit-backed deliverable.

The fix adds a ``_has_working_tree_changes`` fallback so the salvage
condition is satisfied even when ``git diff HEAD`` returns empty (timeout,
error, binary-only files).  ``_auto_commit`` now also verifies return
codes and checks for staged changes before committing.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from aragora.swarm.worker_launcher import (
    _SALVAGEABLE_EXIT_CODES,
    LaunchConfig,
    WorkerLauncher,
    WorkerProcess,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(args), cwd=cwd, text=True, capture_output=True, check=True)


def _head(repo: Path) -> str:
    return _run(repo, "git", "rev-parse", "HEAD").stdout.strip()


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    """Minimal git repo with one initial commit."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _run(repo, "git", "init", "-b", "main")
    _run(repo, "git", "config", "user.email", "test@example.com")
    _run(repo, "git", "config", "user.name", "Test User")
    (repo / "README.md").write_text("hello\n", encoding="utf-8")
    _run(repo, "git", "add", ".")
    _run(repo, "git", "commit", "-m", "initial")
    return repo


# ---------------------------------------------------------------------------
# Core: salvageable 141 with unstaged work -> concrete deliverable
# ---------------------------------------------------------------------------


class TestSalvageableExit141WithUnstagedWork:
    """Exit 141 with valid unstaged changes must produce a commit."""

    @pytest.mark.asyncio
    async def test_wait_path_stages_and_commits(self, repo: Path) -> None:
        """The wait() path must stage unstaged changes for exit 141."""
        initial_head = _head(repo)
        (repo / "README.md").write_text("updated\n", encoding="utf-8")
        # Changes are in the working tree but NOT staged
        assert "README.md" in _run(repo, "git", "status", "--porcelain").stdout

        launcher = WorkerLauncher(config=LaunchConfig())
        worker = WorkerProcess(
            work_order_id="wo-141",
            agent="codex",
            worktree_path=str(repo),
            branch="main",
            exit_code=141,
            initial_head=initial_head,
        )
        launcher._workers["wo-141"] = worker

        # Simulate process already completed — skip real subprocess
        with patch.object(launcher, "_processes", {"wo-141": _FakeFinishedProc(141)}):
            result = await launcher.wait("wo-141")

        assert result.exit_code == 141
        assert result.commit_shas, "Expected at least one salvage commit"
        assert result.head_sha != initial_head
        assert _head(repo) != initial_head, "HEAD must advance after salvage commit"

    @pytest.mark.asyncio
    async def test_detached_path_stages_and_commits(self, repo: Path) -> None:
        """collect_detached_result must stage unstaged changes for exit 141."""
        initial_head = _head(repo)
        (repo / "README.md").write_text("updated\n", encoding="utf-8")
        assert "README.md" in _run(repo, "git", "status", "--porcelain").stdout

        # Simulate session meta indicating exit 141
        import json

        meta = {
            "pid": 99999,
            "session_id": "test",
            "agent": "codex",
            "started_at": "2026-03-10T05:00:00Z",
            "ended_at": "2026-03-10T05:01:00Z",
            "exit_code": 141,
        }
        (repo / ".codex_session_meta.json").write_text(json.dumps(meta) + "\n", encoding="utf-8")

        result = await WorkerLauncher.collect_detached_result(
            work_order_id="wo-141",
            agent="codex",
            worktree_path=str(repo),
            branch="main",
            pid=None,
            initial_head=initial_head,
            auto_commit=True,
        )

        assert result is not None
        assert result.exit_code == 141
        assert result.commit_shas, "Expected at least one salvage commit"
        assert result.head_sha != initial_head

    @pytest.mark.asyncio
    async def test_salvage_when_diff_returns_empty(self, repo: Path) -> None:
        """Salvage must work even when _collect_diff returns empty (fallback)."""
        initial_head = _head(repo)
        (repo / "README.md").write_text("updated\n", encoding="utf-8")

        launcher = WorkerLauncher(config=LaunchConfig())
        worker = WorkerProcess(
            work_order_id="wo-141-nodiff",
            agent="codex",
            worktree_path=str(repo),
            branch="main",
            exit_code=141,
            initial_head=initial_head,
        )
        launcher._workers["wo-141-nodiff"] = worker

        # Force _collect_diff to return empty, simulating timeout/error
        with (
            patch.object(launcher, "_processes", {"wo-141-nodiff": _FakeFinishedProc(141)}),
            patch.object(WorkerLauncher, "_collect_diff", new_callable=AsyncMock, return_value=""),
        ):
            result = await launcher.wait("wo-141-nodiff")

        assert result.commit_shas, "Fallback should still produce a salvage commit"
        assert _head(repo) != initial_head


# ---------------------------------------------------------------------------
# Guard: non-salvageable non-zero exit -> no deliverable
# ---------------------------------------------------------------------------


class TestNonSalvageableExitNoDeliverable:
    """Non-salvageable non-zero exits must NOT produce commits."""

    @pytest.mark.asyncio
    async def test_exit_1_no_commit(self, repo: Path) -> None:
        """Exit code 1 is not salvageable — changes must stay uncommitted."""
        initial_head = _head(repo)
        (repo / "README.md").write_text("bad work\n", encoding="utf-8")

        launcher = WorkerLauncher(config=LaunchConfig())
        worker = WorkerProcess(
            work_order_id="wo-fail",
            agent="codex",
            worktree_path=str(repo),
            branch="main",
            exit_code=1,
            initial_head=initial_head,
        )
        launcher._workers["wo-fail"] = worker

        with patch.object(launcher, "_processes", {"wo-fail": _FakeFinishedProc(1)}):
            result = await launcher.wait("wo-fail")

        assert result.exit_code == 1
        assert not result.commit_shas, "Non-salvageable exit must not produce commits"
        assert _head(repo) == initial_head

    @pytest.mark.asyncio
    async def test_exit_2_detached_no_commit(self, repo: Path) -> None:
        """Exit code 2 via detached path must not produce commits."""
        import json

        initial_head = _head(repo)
        (repo / "README.md").write_text("bad work\n", encoding="utf-8")

        meta = {
            "pid": 99999,
            "ended_at": "2026-03-10T05:01:00Z",
            "exit_code": 2,
        }
        (repo / ".codex_session_meta.json").write_text(json.dumps(meta) + "\n", encoding="utf-8")

        result = await WorkerLauncher.collect_detached_result(
            work_order_id="wo-fail-detached",
            agent="codex",
            worktree_path=str(repo),
            branch="main",
            pid=None,
            initial_head=initial_head,
            auto_commit=True,
        )

        assert result is not None
        assert result.exit_code == 2
        assert not result.commit_shas
        assert _head(repo) == initial_head


# ---------------------------------------------------------------------------
# Guard: dirty worktree only does not count unless salvage produces deliverable
# ---------------------------------------------------------------------------


class TestDirtyWorktreeRequiresSalvageDeliverable:
    """Dirty worktree alone is not a deliverable — salvage must create a commit."""

    @pytest.mark.asyncio
    async def test_dirty_tree_exit_0_produces_commit(self, repo: Path) -> None:
        """Clean exit with dirty tree should auto-commit (normal path)."""
        initial_head = _head(repo)
        (repo / "README.md").write_text("clean exit work\n", encoding="utf-8")

        launcher = WorkerLauncher(config=LaunchConfig())
        worker = WorkerProcess(
            work_order_id="wo-clean",
            agent="codex",
            worktree_path=str(repo),
            branch="main",
            exit_code=0,
            initial_head=initial_head,
        )
        launcher._workers["wo-clean"] = worker

        with patch.object(launcher, "_processes", {"wo-clean": _FakeFinishedProc(0)}):
            result = await launcher.wait("wo-clean")

        assert result.commit_shas, "Exit 0 with dirty tree should produce commit"

    @pytest.mark.asyncio
    async def test_dirty_tree_no_auto_commit_flag(self, repo: Path) -> None:
        """With auto_commit=False, dirty tree produces no commit (no deliverable)."""
        initial_head = _head(repo)
        (repo / "README.md").write_text("work\n", encoding="utf-8")

        launcher = WorkerLauncher(config=LaunchConfig(auto_commit=False))
        worker = WorkerProcess(
            work_order_id="wo-noac",
            agent="codex",
            worktree_path=str(repo),
            branch="main",
            exit_code=0,
            initial_head=initial_head,
        )
        launcher._workers["wo-noac"] = worker

        with patch.object(launcher, "_processes", {"wo-noac": _FakeFinishedProc(0)}):
            result = await launcher.wait("wo-noac")

        assert not result.commit_shas, "auto_commit=False must not create commits"
        assert _head(repo) == initial_head


# ---------------------------------------------------------------------------
# Unit: _has_working_tree_changes
# ---------------------------------------------------------------------------


class TestHasWorkingTreeChanges:
    """Unit tests for the fallback dirty-tree detector."""

    @pytest.mark.asyncio
    async def test_clean_repo(self, repo: Path) -> None:
        assert not await WorkerLauncher._has_working_tree_changes(str(repo))

    @pytest.mark.asyncio
    async def test_modified_file(self, repo: Path) -> None:
        (repo / "README.md").write_text("changed\n", encoding="utf-8")
        assert await WorkerLauncher._has_working_tree_changes(str(repo))

    @pytest.mark.asyncio
    async def test_only_session_artifacts(self, repo: Path) -> None:
        """Session artifacts alone should not count as real changes."""
        (repo / ".codex_session_meta.json").write_text("{}\n", encoding="utf-8")
        (repo / ".codex_session.log").write_text("log\n", encoding="utf-8")
        (repo / ".codex_session_active").write_text("1\n", encoding="utf-8")
        assert not await WorkerLauncher._has_working_tree_changes(str(repo))

    @pytest.mark.asyncio
    async def test_real_changes_plus_artifacts(self, repo: Path) -> None:
        """Real changes should be detected even when artifacts are present."""
        (repo / ".codex_session_meta.json").write_text("{}\n", encoding="utf-8")
        (repo / "README.md").write_text("changed\n", encoding="utf-8")
        assert await WorkerLauncher._has_working_tree_changes(str(repo))


# ---------------------------------------------------------------------------
# Unit: _SALVAGEABLE_EXIT_CODES
# ---------------------------------------------------------------------------


class TestSalvageableExitCodes:
    """Verify the set of salvageable exit codes."""

    def test_141_is_salvageable(self) -> None:
        assert 141 in _SALVAGEABLE_EXIT_CODES

    def test_0_is_not_salvageable(self) -> None:
        assert 0 not in _SALVAGEABLE_EXIT_CODES

    def test_1_is_not_salvageable(self) -> None:
        assert 1 not in _SALVAGEABLE_EXIT_CODES

    def test_minus1_is_not_salvageable(self) -> None:
        assert -1 not in _SALVAGEABLE_EXIT_CODES


# ---------------------------------------------------------------------------
# Fake process for wait() tests
# ---------------------------------------------------------------------------


class _FakeFinishedProc:
    """Minimal asyncio.subprocess.Process stand-in that is already finished."""

    def __init__(self, returncode: int) -> None:
        self.returncode = returncode
        self.pid = 99999

    async def communicate(self) -> tuple[bytes, bytes]:
        return b"", b""

    async def wait(self) -> int:
        return self.returncode

    def kill(self) -> None:
        pass
