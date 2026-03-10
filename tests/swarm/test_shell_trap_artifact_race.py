"""Regression tests for shell-trap session artifact recreation race (#902).

Covers the bug where the ``codex_session.sh`` EXIT trap handler rewrites
``.codex_session_meta.json`` and appends to ``.codex_session.log`` after
Python-side ``_cleanup_session_artifacts`` has already deleted them.

The fix adds ``_wait_for_pid_exit`` in ``collect_detached_result``'s
finally block, ensuring the shell process (and its trap) has fully
terminated before artifact deletion.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from aragora.swarm.worker_launcher import (
    SESSION_ARTIFACTS,
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


def _create_session_artifacts(
    repo: Path,
    *,
    pid: int = 99999,
    ended_at: str = "2026-03-10T05:01:00Z",
    exit_code: int = 0,
) -> None:
    """Plant session artifacts simulating a terminated worker."""
    meta: dict[str, Any] = {
        "pid": pid,
        "session_id": "test-race",
        "agent": "codex",
        "started_at": "2026-03-10T05:00:00Z",
        "ended_at": ended_at,
        "exit_code": exit_code,
    }
    (repo / ".codex_session_meta.json").write_text(json.dumps(meta) + "\n", encoding="utf-8")
    (repo / ".codex_session.log").write_text("log data\n", encoding="utf-8")
    (repo / ".codex_session_active").write_text("1\n", encoding="utf-8")


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
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
# Core: artifacts cleaned after detached result collection
# ---------------------------------------------------------------------------


class TestArtifactsCleanedAfterDetachedCollection:
    """After collect_detached_result, all session artifacts must be gone."""

    @pytest.mark.asyncio
    async def test_meta_and_log_cleaned(self, repo: Path) -> None:
        """Both .codex_session_meta.json and .codex_session.log must be removed."""
        _create_session_artifacts(repo)
        assert (repo / ".codex_session_meta.json").exists()
        assert (repo / ".codex_session.log").exists()

        result = await WorkerLauncher.collect_detached_result(
            work_order_id="wo-race",
            agent="codex",
            worktree_path=str(repo),
            branch="main",
            pid=None,
            initial_head=_head(repo),
            auto_commit=True,
        )

        assert result is not None
        assert not (repo / ".codex_session_meta.json").exists()
        assert not (repo / ".codex_session.log").exists()
        assert not (repo / ".codex_session_active").exists()

    @pytest.mark.asyncio
    async def test_real_changes_preserved(self, repo: Path) -> None:
        """Real worktree changes must survive artifact cleanup."""
        (repo / "README.md").write_text("updated\n", encoding="utf-8")
        _run(repo, "git", "add", "README.md")
        _run(repo, "git", "commit", "-m", "worker change")
        new_head = _head(repo)

        _create_session_artifacts(repo)
        initial_head = _run(repo, "git", "rev-parse", "HEAD~1").stdout.strip()

        result = await WorkerLauncher.collect_detached_result(
            work_order_id="wo-preserve",
            agent="codex",
            worktree_path=str(repo),
            branch="main",
            pid=None,
            initial_head=initial_head,
            auto_commit=True,
        )

        assert result is not None
        assert result.commit_shas == [new_head]
        assert "README.md" in result.changed_paths
        # Artifacts cleaned, real content intact
        assert not (repo / ".codex_session_meta.json").exists()
        assert (repo / "README.md").read_text(encoding="utf-8") == "updated\n"


# ---------------------------------------------------------------------------
# _wait_for_pid_exit
# ---------------------------------------------------------------------------


class TestWaitForPidExit:
    """Unit tests for _wait_for_pid_exit."""

    @pytest.mark.asyncio
    async def test_returns_immediately_for_dead_pid(self) -> None:
        """Should return instantly when PID does not exist."""
        # PID 2**30 is extremely unlikely to exist
        await WorkerLauncher._wait_for_pid_exit(2**30, timeout=1.0)

    @pytest.mark.asyncio
    async def test_waits_for_process_to_exit(self) -> None:
        """Should wait until a running process terminates."""
        # Spawn a short-lived subprocess
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            "import time; time.sleep(0.3)",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        pid = proc.pid
        assert pid is not None

        # Process should still be running
        assert WorkerLauncher._is_pid_running(pid)

        # Wait should block until it exits
        await WorkerLauncher._wait_for_pid_exit(pid, timeout=5.0)

        # After wait returns, process should be gone
        # (need to reap the zombie first)
        await proc.wait()
        assert not WorkerLauncher._is_pid_running(pid)

    @pytest.mark.asyncio
    async def test_timeout_does_not_raise(self) -> None:
        """Should not raise even if timeout expires (graceful degradation)."""
        # Spawn a long-lived process
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            "import time; time.sleep(30)",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        try:
            # Very short timeout — should return without error
            await WorkerLauncher._wait_for_pid_exit(proc.pid, timeout=0.2)
        finally:
            proc.kill()
            await proc.wait()


# ---------------------------------------------------------------------------
# Race simulation: shell trap recreates artifacts after cleanup
# ---------------------------------------------------------------------------


class TestShellTrapRaceEliminated:
    """Simulate the race where a trap handler rewrites artifacts after cleanup."""

    @pytest.mark.asyncio
    async def test_wait_prevents_recreation(self, repo: Path) -> None:
        """When PID wait completes before cleanup, artifacts stay deleted.

        Simulates the race by spawning a subprocess that recreates
        artifacts after a delay, then verifying that _wait_for_pid_exit
        blocks until the subprocess is done.
        """
        _create_session_artifacts(repo)

        # Spawn a helper that simulates the shell trap: sleeps briefly,
        # then rewrites session meta.  This mimics codex_session.sh's
        # EXIT trap running concurrently with Python cleanup.
        trap_script = textwrap.dedent(f"""\
            import json, time
            from pathlib import Path
            time.sleep(0.3)
            meta = {{"pid": 0, "ended_at": "2026-03-10T05:01:00Z", "exit_code": 0}}
            Path("{repo / ".codex_session_meta.json"}").write_text(
                json.dumps(meta) + "\\n", encoding="utf-8"
            )
        """)
        trap_proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            trap_script,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )

        # Wait for the trap simulator to finish, then clean up
        await WorkerLauncher._wait_for_pid_exit(trap_proc.pid, timeout=5.0)
        await trap_proc.wait()

        # The trap has rewritten the meta file
        assert (repo / ".codex_session_meta.json").exists()

        # NOW clean up — since the trap is done, files stay deleted
        WorkerLauncher._cleanup_session_artifacts(str(repo))

        assert not (repo / ".codex_session_meta.json").exists()
        assert not (repo / ".codex_session.log").exists()
        assert not (repo / ".codex_session_active").exists()

    @pytest.mark.asyncio
    async def test_collect_detached_uses_meta_pid(self, repo: Path) -> None:
        """collect_detached_result uses session_meta PID when pid param is None."""
        # Spawn a short-lived process whose PID we'll embed in session meta
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            "import time; time.sleep(0.2)",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        real_pid = proc.pid

        _create_session_artifacts(repo, pid=real_pid)
        initial_head = _head(repo)

        # Call with pid=None — should fall back to session_meta["pid"]
        with patch.object(
            WorkerLauncher,
            "_wait_for_pid_exit",
            new_callable=AsyncMock,
        ) as mock_wait:
            result = await WorkerLauncher.collect_detached_result(
                work_order_id="wo-meta-pid",
                agent="codex",
                worktree_path=str(repo),
                branch="main",
                pid=None,
                initial_head=initial_head,
                auto_commit=True,
            )

        # _wait_for_pid_exit should have been called with the meta PID
        mock_wait.assert_awaited_once_with(real_pid)

        # Cleanup subprocess
        await proc.wait()

    @pytest.mark.asyncio
    async def test_pid_param_takes_precedence(self, repo: Path) -> None:
        """When pid param is provided, it takes precedence over session_meta."""
        _create_session_artifacts(repo, pid=11111)
        initial_head = _head(repo)

        with patch.object(
            WorkerLauncher,
            "_wait_for_pid_exit",
            new_callable=AsyncMock,
        ) as mock_wait:
            # The PID 22222 won't be found running, so collect proceeds
            with patch.object(WorkerLauncher, "_is_pid_running", return_value=False):
                result = await WorkerLauncher.collect_detached_result(
                    work_order_id="wo-pid-prio",
                    agent="codex",
                    worktree_path=str(repo),
                    branch="main",
                    pid=22222,
                    initial_head=initial_head,
                    auto_commit=True,
                )

        # Should use the pid parameter (22222), not session_meta pid (11111)
        mock_wait.assert_awaited_once_with(22222)
