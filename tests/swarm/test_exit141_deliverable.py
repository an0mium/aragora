"""Regression tests for worker exit-141 deliverable salvage (#895).

Covers the bug where a Codex worker performs valid bounded work (modifies
the correct files) but exits with SIGPIPE (141) before committing.  The
auto-commit gate in ``wait()`` previously required ``exit_code == 0``,
silently discarding valid work from signal-terminated workers.

The fix removes the exit-code gate so auto-commit salvages valid diffs
regardless of exit code, turning dirty worktrees into concrete
deliverables (committed branches).
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from aragora.swarm.worker_launcher import LaunchConfig, WorkerLauncher, WorkerProcess


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    """Create a minimal git repo with a tracked file."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _run(repo, "git", "init", "-b", "main")
    _run(repo, "git", "config", "user.email", "test@example.com")
    _run(repo, "git", "config", "user.name", "Test User")
    # Create a tracked file so modifications show in `git diff HEAD`
    (repo / "package.json").write_text('{"version": "1.0.0"}\n', encoding="utf-8")
    (repo / "README.md").write_text("hello\n", encoding="utf-8")
    _run(repo, "git", "add", ".")
    _run(repo, "git", "commit", "-m", "initial")
    return repo


def _run(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(args), cwd=cwd, text=True, capture_output=True, check=True)


def _head(repo: Path) -> str:
    return _run(repo, "git", "rev-parse", "HEAD").stdout.strip()


class TestAutoCommitSalvage:
    """Verify _auto_commit salvages work from non-zero exit workers."""

    def test_exit_141_with_diff_produces_salvage_commit(self, repo: Path) -> None:
        """SIGPIPE exit should auto-commit with 'salvage' message."""
        head = _head(repo)
        (repo / "package.json").write_text('{"version": "1.0.1"}\n', encoding="utf-8")

        worker = WorkerProcess(
            work_order_id="wo-141",
            agent="codex",
            worktree_path=str(repo),
            branch="main",
            pid=None,
            initial_head=head,
        )
        worker.exit_code = 141
        worker.diff = _run(repo, "git", "diff", "HEAD").stdout

        asyncio.run(WorkerLauncher._auto_commit(worker))

        new_head = _head(repo)
        assert new_head != head
        log = _run(repo, "git", "log", "--oneline", "-1").stdout
        assert "salvage" in log
        assert "exit 141" in log
        assert "wo-141" in log

    def test_exit_0_produces_clean_commit(self, repo: Path) -> None:
        """Clean exit produces normal commit message without 'salvage'."""
        head = _head(repo)
        (repo / "package.json").write_text('{"version": "2.0.0"}\n', encoding="utf-8")

        worker = WorkerProcess(
            work_order_id="wo-clean",
            agent="codex",
            worktree_path=str(repo),
            branch="main",
            pid=None,
            initial_head=head,
        )
        worker.exit_code = 0
        worker.diff = _run(repo, "git", "diff", "HEAD").stdout

        asyncio.run(WorkerLauncher._auto_commit(worker))

        log = _run(repo, "git", "log", "--oneline", "-1").stdout
        assert "completed" in log
        assert "salvage" not in log

    def test_exit_137_sigkill_produces_salvage_commit(self, repo: Path) -> None:
        """SIGKILL (137) should also salvage."""
        head = _head(repo)
        (repo / "package.json").write_text('{"version": "3.0.0"}\n', encoding="utf-8")

        worker = WorkerProcess(
            work_order_id="wo-137",
            agent="codex",
            worktree_path=str(repo),
            branch="main",
            pid=None,
            initial_head=head,
        )
        worker.exit_code = 137
        worker.diff = _run(repo, "git", "diff", "HEAD").stdout

        asyncio.run(WorkerLauncher._auto_commit(worker))

        log = _run(repo, "git", "log", "--oneline", "-1").stdout
        assert "salvage" in log
        assert "exit 137" in log

    def test_no_diff_no_commit(self, repo: Path) -> None:
        """Without diff, no commit regardless of exit code."""
        head = _head(repo)
        # No changes to tracked files
        assert _head(repo) == head


class TestWaitGateRemoved:
    """Verify wait() no longer gates auto-commit on exit_code == 0.

    Tests the code path by checking source — the actual auto-commit
    behavior is tested above via _auto_commit directly.
    """

    def test_wait_method_does_not_gate_on_exit_code(self) -> None:
        """The wait() method should call auto-commit without exit_code check."""
        import inspect

        src = inspect.getsource(WorkerLauncher.wait)
        # The old code had: `worker.exit_code == 0`
        assert "exit_code == 0" not in src
        # The new code has: `self.config.auto_commit and worker.diff`
        assert "auto_commit" in src
        assert "worker.diff" in src


class TestDetachedCollectionSalvage:
    """Verify collect_detached_result auto-commits on exit 141."""

    def test_detached_exit_141_commits_tracked_changes(self, repo: Path) -> None:
        """collect_detached_result should commit modified tracked files."""
        head = _head(repo)
        # Modify a TRACKED file so git diff HEAD shows it
        (repo / "package.json").write_text('{"version": "1.0.1"}\n', encoding="utf-8")

        meta = {
            "pid": 99999,
            "session_id": "test-detached",
            "agent": "codex",
            "started_at": "2026-03-10T05:00:00Z",
            "ended_at": "2026-03-10T05:01:00Z",
            "exit_code": 141,
        }
        (repo / ".codex_session_meta.json").write_text(json.dumps(meta) + "\n", encoding="utf-8")

        result = asyncio.run(
            WorkerLauncher.collect_detached_result(
                work_order_id="wo-detached-141",
                agent="codex",
                worktree_path=str(repo),
                branch="main",
                initial_head=head,
                auto_commit=True,
            )
        )

        assert result is not None
        assert result.exit_code == 141
        new_head = _head(repo)
        assert new_head != head
        assert len(result.commit_shas) > 0
        log = _run(repo, "git", "log", "--oneline", "-1").stdout
        assert "salvage" in log
        # Session artifact should not be in the commit
        diff_files = _run(repo, "git", "diff", "--name-only", f"{head}..{new_head}").stdout
        assert ".codex_session_meta.json" not in diff_files
        assert "package.json" in diff_files

    def test_detached_exit_0_commits_clean(self, repo: Path) -> None:
        """collect_detached_result with clean exit uses normal message."""
        head = _head(repo)
        (repo / "package.json").write_text('{"version": "2.0.0"}\n', encoding="utf-8")

        meta = {
            "pid": 99999,
            "session_id": "test-clean",
            "agent": "codex",
            "started_at": "2026-03-10T05:00:00Z",
            "ended_at": "2026-03-10T05:01:00Z",
            "exit_code": 0,
        }
        (repo / ".codex_session_meta.json").write_text(json.dumps(meta) + "\n", encoding="utf-8")

        result = asyncio.run(
            WorkerLauncher.collect_detached_result(
                work_order_id="wo-detached-clean",
                agent="codex",
                worktree_path=str(repo),
                branch="main",
                initial_head=head,
                auto_commit=True,
            )
        )

        assert result is not None
        assert result.exit_code == 0
        assert len(result.commit_shas) > 0
        log = _run(repo, "git", "log", "--oneline", "-1").stdout
        assert "completed" in log
        assert "salvage" not in log

    def test_detached_no_diff_no_commit(self, repo: Path) -> None:
        """No changes means no commit, even with terminal session."""
        head = _head(repo)

        meta = {
            "pid": 99999,
            "session_id": "test-noop",
            "agent": "codex",
            "started_at": "2026-03-10T05:00:00Z",
            "ended_at": "2026-03-10T05:01:00Z",
            "exit_code": 141,
        }
        (repo / ".codex_session_meta.json").write_text(json.dumps(meta) + "\n", encoding="utf-8")

        result = asyncio.run(
            WorkerLauncher.collect_detached_result(
                work_order_id="wo-noop",
                agent="codex",
                worktree_path=str(repo),
                branch="main",
                initial_head=head,
                auto_commit=True,
            )
        )

        assert result is not None
        assert result.exit_code == 141
        assert _head(repo) == head
        assert result.commit_shas == []


class TestDeliverableGateIntegration:
    """Verify salvaged commits pass the deliverable gate from #893."""

    def test_salvaged_commit_is_concrete_deliverable(self) -> None:
        """A work order with commit_shas from salvaged work passes the gate."""
        from aragora.swarm.boss_loop import _extract_deliverable

        run_dict: dict[str, Any] = {
            "work_orders": [
                {
                    "work_order_id": "wo-salvaged",
                    "status": "completed",
                    "branch": "codex/swarm-work-abc",
                    "commit_shas": ["abc123"],
                    "pr_url": "",
                },
            ],
        }
        result = _extract_deliverable(run_dict)
        assert result is not None
        assert result["type"] == "branch"
        assert result["commit_shas"] == ["abc123"]

    def test_unsalvaged_exit_141_no_deliverable(self) -> None:
        """Without auto-commit, exit 141 produces no deliverable."""
        from aragora.swarm.boss_loop import _extract_deliverable

        run_dict: dict[str, Any] = {
            "work_orders": [
                {
                    "work_order_id": "wo-unsalvaged",
                    "status": "completed",
                    "branch": "",
                    "commit_shas": [],
                    "pr_url": "",
                    "changed_paths": ["package.json"],
                },
            ],
        }
        assert _extract_deliverable(run_dict) is None
