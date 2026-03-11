"""Regression tests for worker exit-141 deliverable salvage (#895).

Covers the bug where a Codex worker performs valid bounded work (modifies
the correct files) but exits with SIGPIPE (141) before committing.  The
auto-commit gate in ``wait()`` previously required ``exit_code == 0``,
silently discarding valid work from signal-terminated workers.

The fix adds SIGPIPE (141) to a narrow ``_SALVAGEABLE_EXIT_CODES`` set so
auto-commit fires for that specific transport failure.  All other non-zero
exits remain blocked — crashed or partial executions must NOT become
concrete deliverables.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from aragora.swarm.worker_launcher import (
    LaunchConfig,
    WorkerLauncher,
    WorkerProcess,
    _SALVAGEABLE_EXIT_CODES,
)


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    """Create a minimal git repo with tracked files matching #873 shape."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _run(repo, "git", "init", "-b", "main")
    _run(repo, "git", "config", "user.email", "test@example.com")
    _run(repo, "git", "config", "user.name", "Test User")
    (repo / "package.json").write_text('{"version": "1.0.0"}\n', encoding="utf-8")
    (repo / "package-lock.json").write_text('{"lockfileVersion": 3}\n', encoding="utf-8")
    (repo / "README.md").write_text("hello\n", encoding="utf-8")
    _run(repo, "git", "add", ".")
    _run(repo, "git", "commit", "-m", "initial")
    return repo


def _run(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(args), cwd=cwd, text=True, capture_output=True, check=True)


def _head(repo: Path) -> str:
    return _run(repo, "git", "rev-parse", "HEAD").stdout.strip()


def _modify_packages(repo: Path) -> None:
    """Simulate the exact #873 bounded work: bump tracked dependency files."""
    (repo / "package.json").write_text('{"version": "1.0.1"}\n', encoding="utf-8")
    (repo / "package-lock.json").write_text(
        '{"lockfileVersion": 3, "bumped": true}\n', encoding="utf-8"
    )


class TestSalvageableExitCodes:
    """Verify only the intended exit codes are in the salvageable set."""

    def test_141_is_salvageable(self) -> None:
        assert 141 in _SALVAGEABLE_EXIT_CODES

    def test_137_is_not_salvageable(self) -> None:
        """SIGKILL (137) represents a crash — must NOT be salvaged."""
        assert 137 not in _SALVAGEABLE_EXIT_CODES

    def test_1_is_not_salvageable(self) -> None:
        """Generic error (1) must NOT be salvaged."""
        assert 1 not in _SALVAGEABLE_EXIT_CODES

    def test_0_is_not_in_set(self) -> None:
        """Clean exit is handled separately, not via salvage set."""
        assert 0 not in _SALVAGEABLE_EXIT_CODES


class TestAutoCommitSalvage:
    """Verify _auto_commit message distinguishes salvage from clean."""

    def test_exit_141_produces_salvage_commit(self, repo: Path) -> None:
        """SIGPIPE exit should auto-commit with 'salvage' message."""
        head = _head(repo)
        _modify_packages(repo)

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

        # Verify both package files are in the commit
        diff_files = _run(repo, "git", "diff", "--name-only", f"{head}..{new_head}").stdout
        assert "package.json" in diff_files
        assert "package-lock.json" in diff_files

    def test_exit_0_produces_clean_commit(self, repo: Path) -> None:
        """Clean exit produces normal commit without 'salvage'."""
        head = _head(repo)
        _modify_packages(repo)

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

    def test_exit_0_untracked_file_produces_commit(self, repo: Path) -> None:
        """Clean exit with NEW (untracked) file should auto-commit.

        This covers the Phase 0A governance task pattern: worker creates a new
        markdown file that ``git diff HEAD`` does not detect because the file
        is untracked, not modified. The porcelain fallback must detect it.
        """
        head = _head(repo)
        # Create a new untracked file (not a modification of existing tracked file)
        (repo / "docs").mkdir(exist_ok=True)
        (repo / "docs" / "new-governance-doc.md").write_text(
            "# New Document\n\nCreated by worker.\n", encoding="utf-8"
        )
        # git diff HEAD is empty for untracked files
        diff = _run(repo, "git", "diff", "HEAD").stdout
        assert diff == ""

        worker = WorkerProcess(
            work_order_id="wo-untracked",
            agent="codex",
            worktree_path=str(repo),
            branch="main",
            pid=None,
            initial_head=head,
        )
        worker.exit_code = 0
        worker.diff = diff  # empty — simulates the real failure

        asyncio.run(WorkerLauncher._auto_commit(worker))

        new_head = _head(repo)
        assert new_head != head, "auto-commit should have created a commit for untracked file"
        diff_files = _run(repo, "git", "diff", "--name-only", f"{head}..{new_head}").stdout
        assert "docs/new-governance-doc.md" in diff_files


class TestWaitGateNarrowed:
    """Verify wait() only salvages _SALVAGEABLE_EXIT_CODES, not all non-zero."""

    def test_wait_source_checks_salvageable_set(self) -> None:
        """The wait() method should gate on _SALVAGEABLE_EXIT_CODES."""
        import inspect

        src = inspect.getsource(WorkerLauncher.wait)
        # Must NOT have the old blanket gate
        assert "exit_code == 0" not in src or "_SALVAGEABLE_EXIT_CODES" in src
        # Must reference the salvageable set
        assert "_SALVAGEABLE_EXIT_CODES" in src


class TestNonSalvageableExitsBlocked:
    """Verify non-salvageable non-zero exits do NOT produce commits."""

    @pytest.mark.parametrize("exit_code", [1, 2, 137, 139, 143])
    def test_non_salvageable_exit_no_commit_detached(self, repo: Path, exit_code: int) -> None:
        """Non-salvageable exits must NOT auto-commit even with valid diff."""
        head = _head(repo)
        _modify_packages(repo)

        meta = {
            "pid": 99999,
            "session_id": "test-blocked",
            "agent": "codex",
            "started_at": "2026-03-10T05:00:00Z",
            "ended_at": "2026-03-10T05:01:00Z",
            "exit_code": exit_code,
        }
        (repo / ".codex_session_meta.json").write_text(json.dumps(meta) + "\n", encoding="utf-8")

        result = asyncio.run(
            WorkerLauncher.collect_detached_result(
                work_order_id=f"wo-blocked-{exit_code}",
                agent="codex",
                worktree_path=str(repo),
                branch="main",
                initial_head=head,
                auto_commit=True,
            )
        )

        assert result is not None
        assert result.exit_code == exit_code
        # HEAD must NOT advance — no commit created
        assert _head(repo) == head
        assert result.commit_shas == []


class TestDetachedSalvagePath:
    """Verify collect_detached_result salvages only SIGPIPE (141)."""

    def test_detached_exit_141_commits_tracked_changes(self, repo: Path) -> None:
        """collect_detached_result should commit modified tracked files on 141."""
        head = _head(repo)
        _modify_packages(repo)

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
        _modify_packages(repo)

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
        """No changes means no commit, even with salvageable exit code."""
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


class TestUntrackedFileDetection:
    """Verify clean-exit workers with only untracked files get auto-committed.

    This is the B-0 root cause for Phase 0A governance tasks: workers create
    NEW files (not modifications), ``git diff HEAD`` returns empty, and the
    auto-commit gate must fall back to ``git status --porcelain``.
    """

    def _create_untracked_file(self, repo: Path) -> None:
        (repo / "docs").mkdir(exist_ok=True)
        (repo / "docs" / "new-governance-doc.md").write_text(
            "# New Document\n\nCreated by worker.\n", encoding="utf-8"
        )

    def test_detached_exit_0_untracked_file_commits(self, repo: Path) -> None:
        """collect_detached_result must commit untracked files on clean exit."""
        head = _head(repo)
        self._create_untracked_file(repo)
        assert _run(repo, "git", "diff", "HEAD").stdout == ""

        meta = {
            "pid": 99999,
            "session_id": "test-untracked-detached",
            "agent": "codex",
            "started_at": "2026-03-10T05:00:00Z",
            "ended_at": "2026-03-10T05:01:00Z",
            "exit_code": 0,
        }
        (repo / ".codex_session_meta.json").write_text(json.dumps(meta) + "\n", encoding="utf-8")

        result = asyncio.run(
            WorkerLauncher.collect_detached_result(
                work_order_id="wo-untracked-detached",
                agent="codex",
                worktree_path=str(repo),
                branch="main",
                initial_head=head,
                auto_commit=True,
            )
        )

        assert result is not None
        assert result.exit_code == 0
        new_head = _head(repo)
        assert new_head != head, "untracked file must be committed via porcelain fallback"
        assert len(result.commit_shas) > 0
        diff_files = _run(repo, "git", "diff", "--name-only", f"{head}..{new_head}").stdout
        assert "docs/new-governance-doc.md" in diff_files

    def test_wait_exit_0_untracked_file_commits(self, repo: Path) -> None:
        """wait() must commit untracked files on clean exit.

        Exercises the full gating logic in wait() rather than calling
        _auto_commit() directly.
        """
        head = _head(repo)
        self._create_untracked_file(repo)

        launcher = WorkerLauncher(config=LaunchConfig(auto_commit=True))

        worker = WorkerProcess(
            work_order_id="wo-untracked-wait",
            agent="codex",
            worktree_path=str(repo),
            branch="main",
            pid=None,
            initial_head=head,
        )
        launcher._workers["wo-untracked-wait"] = worker

        # Use MagicMock to simulate a finished process with exit code 0
        from unittest.mock import AsyncMock, MagicMock

        proc = MagicMock()
        proc.returncode = 0
        proc.communicate = AsyncMock(return_value=(b"done", b""))
        launcher._processes["wo-untracked-wait"] = proc

        result = asyncio.run(launcher.wait("wo-untracked-wait"))

        assert result.exit_code == 0
        new_head = _head(repo)
        assert new_head != head, "untracked file must be committed via porcelain fallback in wait()"
        assert len(result.commit_shas) > 0
        diff_files = _run(repo, "git", "diff", "--name-only", f"{head}..{new_head}").stdout
        assert "docs/new-governance-doc.md" in diff_files


class TestDeliverableGateIntegration:
    """Verify salvaged commits pass the #893 gate; unsalvaged do not."""

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

    def test_unsalvaged_dirty_worktree_no_deliverable(self) -> None:
        """Without salvage commit, dirty worktree is not a deliverable."""
        from aragora.swarm.boss_loop import _extract_deliverable

        run_dict: dict[str, Any] = {
            "work_orders": [
                {
                    "work_order_id": "wo-unsalvaged",
                    "status": "completed",
                    "branch": "",
                    "commit_shas": [],
                    "pr_url": "",
                    "changed_paths": ["package.json", "package-lock.json"],
                },
            ],
        }
        assert _extract_deliverable(run_dict) is None

    def test_non_salvageable_crash_no_deliverable(self) -> None:
        """A crashed worker (no commits) produces no deliverable."""
        from aragora.swarm.boss_loop import _extract_deliverable

        run_dict: dict[str, Any] = {
            "work_orders": [
                {
                    "work_order_id": "wo-crashed",
                    "status": "completed",
                    "branch": "codex/swarm-work-xyz",
                    "commit_shas": [],
                    "pr_url": "",
                    "changed_paths": ["package.json"],
                },
            ],
        }
        assert _extract_deliverable(run_dict) is None
