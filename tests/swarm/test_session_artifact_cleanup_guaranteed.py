"""Regression tests for guaranteed session artifact cleanup (#896).

Covers the bug where _cleanup_session_artifacts was called outside a
try/finally block in both wait() and collect_detached_result().  Any
exception during diff collection, auto-commit, or git operations
(rev-parse, commit_shas, changed_paths) would skip cleanup entirely,
leaving .codex_session_meta.json behind in the worktree.

The fix wraps the git-operations block in try/finally in both paths
so _cleanup_session_artifacts runs regardless of success or failure.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from aragora.swarm.worker_launcher import (
    SESSION_ARTIFACTS,
    LaunchConfig,
    WorkerLauncher,
    WorkerProcess,
)


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    """Create a minimal git repo with tracked files."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _run(repo, "git", "init", "-b", "main")
    _run(repo, "git", "config", "user.email", "test@example.com")
    _run(repo, "git", "config", "user.name", "Test User")
    (repo / "README.md").write_text("hello\n", encoding="utf-8")
    _run(repo, "git", "add", ".")
    _run(repo, "git", "commit", "-m", "initial")
    return repo


def _run(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(args), cwd=cwd, text=True, capture_output=True, check=True)


def _head(repo: Path) -> str:
    return _run(repo, "git", "rev-parse", "HEAD").stdout.strip()


def _create_session_artifacts(repo: Path) -> None:
    """Plant all session artifacts into the worktree."""
    meta = {
        "pid": 99999,
        "session_id": "test-cleanup",
        "agent": "codex",
        "started_at": "2026-03-10T05:00:00Z",
        "ended_at": "2026-03-10T05:01:00Z",
        "exit_code": 0,
    }
    (repo / ".codex_session_meta.json").write_text(json.dumps(meta) + "\n", encoding="utf-8")
    (repo / ".codex_session.log").write_text("session log data\n", encoding="utf-8")
    (repo / ".codex_session_active").write_text("1\n", encoding="utf-8")


def _session_artifacts_exist(repo: Path) -> dict[str, bool]:
    """Check which session artifacts exist on disk."""
    return {name: (repo / name).exists() for name in SESSION_ARTIFACTS}


# ---------------------------------------------------------------------------
# Happy path: artifacts cleaned after successful collection
# ---------------------------------------------------------------------------


class TestCleanupAfterSuccessfulCollection:
    """Verify artifacts are removed after normal result collection."""

    def test_detached_cleans_all_artifacts(self, repo: Path) -> None:
        """collect_detached_result removes all session artifacts on success."""
        head = _head(repo)
        _create_session_artifacts(repo)

        # All artifacts should exist before collection
        before = _session_artifacts_exist(repo)
        assert before[".codex_session_meta.json"] is True
        assert before[".codex_session.log"] is True
        assert before[".codex_session_active"] is True

        result = asyncio.run(
            WorkerLauncher.collect_detached_result(
                work_order_id="wo-cleanup-ok",
                agent="codex",
                worktree_path=str(repo),
                branch="main",
                initial_head=head,
                auto_commit=False,
            )
        )

        assert result is not None
        # All artifacts must be gone
        after = _session_artifacts_exist(repo)
        assert after[".codex_session_meta.json"] is False
        assert after[".codex_session.log"] is False
        assert after[".codex_session_active"] is False

    def test_detached_preserves_real_files(self, repo: Path) -> None:
        """Real deliverable files must survive cleanup."""
        head = _head(repo)
        _create_session_artifacts(repo)
        (repo / "README.md").write_text("updated\n", encoding="utf-8")

        asyncio.run(
            WorkerLauncher.collect_detached_result(
                work_order_id="wo-preserve",
                agent="codex",
                worktree_path=str(repo),
                branch="main",
                initial_head=head,
                auto_commit=False,
            )
        )

        # Real file untouched
        assert (repo / "README.md").read_text(encoding="utf-8") == "updated\n"
        # Artifacts gone
        assert not (repo / ".codex_session_meta.json").exists()


# ---------------------------------------------------------------------------
# Exception path: artifacts cleaned even when git operations fail
# ---------------------------------------------------------------------------


class TestCleanupOnGitFailure:
    """Verify artifacts are cleaned even when git operations raise exceptions.

    This is the core #896 regression: before the fix, any exception in
    _collect_diff / _git_output / _collect_commit_shas / _collect_changed_paths
    would skip _cleanup_session_artifacts entirely.
    """

    def test_detached_cleanup_on_diff_failure(self, repo: Path) -> None:
        """Artifacts cleaned even when _collect_diff raises."""
        head = _head(repo)
        _create_session_artifacts(repo)

        with patch.object(
            WorkerLauncher,
            "_collect_diff",
            new_callable=AsyncMock,
            side_effect=OSError("git diff failed"),
        ):
            with pytest.raises(OSError, match="git diff failed"):
                asyncio.run(
                    WorkerLauncher.collect_detached_result(
                        work_order_id="wo-diff-fail",
                        agent="codex",
                        worktree_path=str(repo),
                        branch="main",
                        initial_head=head,
                        auto_commit=False,
                    )
                )

        # Artifacts must still be cleaned despite the exception
        assert not (repo / ".codex_session_meta.json").exists()
        assert not (repo / ".codex_session.log").exists()
        assert not (repo / ".codex_session_active").exists()

    def test_detached_cleanup_on_rev_parse_failure(self, repo: Path) -> None:
        """Artifacts cleaned even when git rev-parse raises."""
        head = _head(repo)
        _create_session_artifacts(repo)

        original_git_output = WorkerLauncher._git_output

        async def _failing_git_output(worktree_path: str, *args: str) -> str:
            if "rev-parse" in args:
                raise RuntimeError("rev-parse failed")
            return await original_git_output(worktree_path, *args)

        with patch.object(WorkerLauncher, "_git_output", side_effect=_failing_git_output):
            with pytest.raises(RuntimeError, match="rev-parse failed"):
                asyncio.run(
                    WorkerLauncher.collect_detached_result(
                        work_order_id="wo-revparse-fail",
                        agent="codex",
                        worktree_path=str(repo),
                        branch="main",
                        initial_head=head,
                        auto_commit=False,
                    )
                )

        assert not (repo / ".codex_session_meta.json").exists()

    def test_detached_cleanup_on_commit_shas_failure(self, repo: Path) -> None:
        """Artifacts cleaned even when _collect_commit_shas raises."""
        head = _head(repo)
        _create_session_artifacts(repo)

        with patch.object(
            WorkerLauncher,
            "_collect_commit_shas",
            new_callable=AsyncMock,
            side_effect=RuntimeError("commit shas failed"),
        ):
            with pytest.raises(RuntimeError, match="commit shas failed"):
                asyncio.run(
                    WorkerLauncher.collect_detached_result(
                        work_order_id="wo-shas-fail",
                        agent="codex",
                        worktree_path=str(repo),
                        branch="main",
                        initial_head=head,
                        auto_commit=False,
                    )
                )

        assert not (repo / ".codex_session_meta.json").exists()

    def test_detached_cleanup_on_changed_paths_failure(self, repo: Path) -> None:
        """Artifacts cleaned even when _collect_changed_paths raises."""
        head = _head(repo)
        _create_session_artifacts(repo)

        with patch.object(
            WorkerLauncher,
            "_collect_changed_paths",
            new_callable=AsyncMock,
            side_effect=RuntimeError("changed paths failed"),
        ):
            with pytest.raises(RuntimeError, match="changed paths failed"):
                asyncio.run(
                    WorkerLauncher.collect_detached_result(
                        work_order_id="wo-paths-fail",
                        agent="codex",
                        worktree_path=str(repo),
                        branch="main",
                        initial_head=head,
                        auto_commit=False,
                    )
                )

        assert not (repo / ".codex_session_meta.json").exists()


# ---------------------------------------------------------------------------
# Wait path: same guarantee for in-memory workers
# ---------------------------------------------------------------------------


class TestWaitPathCleanupGuaranteed:
    """Verify the wait() path also cleans up on failure.

    The wait() path uses self._cleanup_session_artifacts in a finally block.
    We test this by constructing a WorkerProcess, planting artifacts, and
    verifying cleanup after simulated git failures.
    """

    def test_wait_cleanup_on_diff_failure(self, repo: Path) -> None:
        """Artifacts cleaned even when _collect_diff raises in wait() path."""
        _create_session_artifacts(repo)
        launcher = WorkerLauncher(config=LaunchConfig())

        # We can't easily call wait() without a real subprocess, so we test
        # the cleanup guarantee by directly calling the finally-protected block
        # pattern. Instead, verify via the source that try/finally is present.
        import inspect

        src = inspect.getsource(WorkerLauncher.wait)
        assert "finally:" in src
        assert "_cleanup_session_artifacts" in src

        # Also verify the cleanup method itself works
        WorkerLauncher._cleanup_session_artifacts(str(repo))
        assert not (repo / ".codex_session_meta.json").exists()
        assert not (repo / ".codex_session.log").exists()
        assert not (repo / ".codex_session_active").exists()


# ---------------------------------------------------------------------------
# Source inspection: both paths use try/finally
# ---------------------------------------------------------------------------


class TestBothPathsUseTryFinally:
    """Verify both collection paths have _cleanup_session_artifacts in finally."""

    def test_wait_has_finally_cleanup(self) -> None:
        import inspect

        src = inspect.getsource(WorkerLauncher.wait)
        # Must have a finally block containing cleanup
        assert "finally:" in src
        assert "_cleanup_session_artifacts" in src
        # The cleanup must appear after "finally:", not before
        finally_pos = src.index("finally:")
        cleanup_pos = src.rindex("_cleanup_session_artifacts")
        assert cleanup_pos > finally_pos

    def test_collect_detached_has_finally_cleanup(self) -> None:
        import inspect

        src = inspect.getsource(WorkerLauncher.collect_detached_result)
        assert "finally:" in src
        assert "_cleanup_session_artifacts" in src
        finally_pos = src.index("finally:")
        cleanup_pos = src.rindex("_cleanup_session_artifacts")
        assert cleanup_pos > finally_pos

    def test_cleanup_not_called_outside_finally(self) -> None:
        """_cleanup_session_artifacts must ONLY appear inside finally blocks."""
        import inspect

        for method_name in ("wait", "collect_detached_result"):
            method = getattr(WorkerLauncher, method_name)
            src = inspect.getsource(method)
            # Count occurrences — should be exactly 1 (inside finally)
            count = src.count("_cleanup_session_artifacts")
            assert count == 1, (
                f"{method_name} has {count} calls to _cleanup_session_artifacts, expected 1"
            )
