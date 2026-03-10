"""Regression tests for pid_alive=False fallback routing (#905).

Covers the bug where the supervisor's pid_alive=False fallback path
marked needs_human directly, bypassing ``collect_detached_result()``
entirely.  This meant exit-141 salvage (#903), session-artifact cleanup
(#896/#904), and deliverable surfacing never executed for workers whose
PID died between the first ``collect_detached_result`` call (which
returned None due to a race) and the ``snapshot_progress`` check.

The fix routes the fallback through detached result collection, matching
the pattern established by the timeout path in #899.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from aragora.nomic.dev_coordination import DevCoordinationStore
from aragora.swarm.supervisor import SwarmSupervisor
from aragora.swarm.worker_launcher import (
    SESSION_ARTIFACTS,
    LaunchConfig,
    WorkerLauncher,
    WorkerProcess,
)

UTC = timezone.utc


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
    _run(repo, "git", "remote", "add", "origin", str(repo))
    _run(repo, "git", "update-ref", "refs/remotes/origin/main", "HEAD")
    return repo


@pytest.fixture()
def store(repo: Path) -> DevCoordinationStore:
    return DevCoordinationStore(repo_root=repo)


def _run(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(args), cwd=cwd, text=True, capture_output=True, check=True)


def _head(repo: Path) -> str:
    return _run(repo, "git", "rev-parse", "HEAD").stdout.strip()


def _create_session_artifacts(repo: Path) -> None:
    """Plant session artifacts simulating a still-running worker (no ended_at)."""
    meta: dict[str, Any] = {
        "pid": 99999,
        "session_id": "test-fallback",
        "agent": "codex",
        "started_at": "2026-03-10T05:00:00Z",
    }
    (repo / ".codex_session_meta.json").write_text(json.dumps(meta) + "\n", encoding="utf-8")
    (repo / ".codex_session.log").write_text("log data\n", encoding="utf-8")
    (repo / ".codex_session_active").write_text("1\n", encoding="utf-8")


def _make_supervisor(repo: Path, store: DevCoordinationStore) -> SwarmSupervisor:
    config = LaunchConfig(no_progress_timeout_seconds=1.0)
    launcher = WorkerLauncher(config=config)
    return SwarmSupervisor(repo_root=repo, store=store, launcher=launcher)


def _make_dispatched_item(
    repo: Path,
    *,
    work_order_id: str = "wo-fallback",
    initial_head: str | None = None,
) -> dict[str, Any]:
    """Build a dispatched work-order item."""
    head = initial_head or _head(repo)
    dispatched_at = datetime(2020, 1, 1, tzinfo=UTC).isoformat()
    return {
        "work_order_id": work_order_id,
        "status": "dispatched",
        "target_agent": "codex",
        "worktree_path": str(repo),
        "branch": "main",
        "initial_head": head,
        "lease_id": "",
        "file_scope": [],
        "pid": 99999,
        "dispatched_at": dispatched_at,
        "last_progress_at": dispatched_at,
        "changed_paths": [],
    }


def _create_run(store: DevCoordinationStore, item: dict[str, Any]) -> str:
    record = store.create_supervisor_run(
        goal="test fallback",
        target_branch="main",
        supervisor_agents={},
        approval_policy={},
        spec={"raw_goal": "test fallback"},
        work_orders=[item],
    )
    return record["run_id"]


# ---------------------------------------------------------------------------
# Core: pid_alive=False path now routes through collect_detached_result
# ---------------------------------------------------------------------------


class TestPidAliveFallbackRoutesThoughCollection:
    """Verify the pid_alive=False fallback calls collect_detached_result."""

    @pytest.mark.asyncio
    async def test_fallback_calls_collect_detached_result(
        self, repo: Path, store: DevCoordinationStore
    ) -> None:
        """The pid_alive=False fallback must attempt detached result collection."""
        supervisor = _make_supervisor(repo, store)
        item = _make_dispatched_item(repo)
        run_id = _create_run(store, item)

        _create_session_artifacts(repo)

        no_deliverable = WorkerProcess(
            work_order_id="wo-fallback",
            agent="codex",
            worktree_path=str(repo),
            branch="main",
            exit_code=0,
            changed_paths=[],
            commit_shas=[],
            head_sha=_head(repo),
        )

        with (
            patch.object(
                supervisor.launcher,
                "collect_finished",
                new_callable=AsyncMock,
                return_value=[],
            ),
            # First collect_detached_result call (line 445): returns None
            # (simulating the race where PID check returned True).
            # Second call (pid_alive=False block): returns the result.
            patch.object(
                WorkerLauncher,
                "collect_detached_result",
                new_callable=AsyncMock,
                side_effect=[None, no_deliverable],
            ) as mock_collect,
            patch.object(
                supervisor.launcher,
                "snapshot_progress",
                new_callable=AsyncMock,
                return_value={"pid_alive": False},
            ),
        ):
            await supervisor.collect_finished_results(run_id)

        assert mock_collect.call_count == 2


# ---------------------------------------------------------------------------
# Salvage: deliverable from pid_alive=False path
# ---------------------------------------------------------------------------


class TestPidAliveFallbackSurfacesDeliverable:
    """Verify the fallback surfaces a salvaged deliverable."""

    @pytest.mark.asyncio
    async def test_salvaged_commits_surface_normally(
        self, repo: Path, store: DevCoordinationStore
    ) -> None:
        """A commit-backed deliverable from the fallback must enter finished list."""
        supervisor = _make_supervisor(repo, store)
        head = _head(repo)

        # Create a real commit to simulate worker deliverable
        (repo / "README.md").write_text("updated\n", encoding="utf-8")
        _run(repo, "git", "add", "README.md")
        _run(repo, "git", "commit", "-m", "worker deliverable")
        new_head = _head(repo)

        item = _make_dispatched_item(repo, initial_head=head)
        run_id = _create_run(store, item)
        _create_session_artifacts(repo)

        salvaged_result = WorkerProcess(
            work_order_id="wo-fallback",
            agent="codex",
            worktree_path=str(repo),
            branch="main",
            exit_code=141,
            changed_paths=["README.md"],
            commit_shas=[new_head],
            head_sha=new_head,
        )

        with (
            patch.object(
                supervisor.launcher,
                "collect_finished",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch.object(
                WorkerLauncher,
                "collect_detached_result",
                new_callable=AsyncMock,
                side_effect=[None, salvaged_result],
            ),
            patch.object(
                supervisor.launcher,
                "snapshot_progress",
                new_callable=AsyncMock,
                return_value={"pid_alive": False},
            ),
        ):
            results = await supervisor.collect_finished_results(run_id)

        assert len(results) == 1
        assert results[0].commit_shas == [new_head]

        record = store.get_supervisor_run(run_id)
        updated_item = record["work_orders"][0]
        assert updated_item["status"] != "needs_human"


# ---------------------------------------------------------------------------
# Cleanup: artifacts cleaned from pid_alive=False path
# ---------------------------------------------------------------------------


class TestPidAliveFallbackCleansArtifacts:
    """Verify session artifacts are cleaned via the fallback."""

    @pytest.mark.asyncio
    async def test_artifacts_cleaned_on_exit(self, repo: Path, store: DevCoordinationStore) -> None:
        """Session artifacts must be removed after fallback result collection."""
        supervisor = _make_supervisor(repo, store)
        head = _head(repo)
        item = _make_dispatched_item(repo, initial_head=head)
        run_id = _create_run(store, item)

        _create_session_artifacts(repo)
        assert (repo / ".codex_session_meta.json").exists()

        with (
            patch.object(
                supervisor.launcher,
                "collect_finished",
                new_callable=AsyncMock,
                return_value=[],
            ),
            # First call: returns None (race). Second call: real collection
            # runs through to the try/finally cleanup.
            patch.object(WorkerLauncher, "_is_pid_running", return_value=False),
            patch.object(
                supervisor.launcher,
                "snapshot_progress",
                new_callable=AsyncMock,
                return_value={"pid_alive": False},
            ),
        ):
            # Let real collect_detached_result run on the second call.
            # First call: _is_pid_running=False + no ended_at → proceeds,
            # try/finally cleans up.  We need first call to return None
            # so the fallback is reached.  Mock only the first call.
            original = WorkerLauncher.collect_detached_result.__func__
            call_count = 0

            async def first_none_then_real(**kwargs: Any) -> WorkerProcess | None:
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return None
                return await original(WorkerLauncher, **kwargs)

            with patch.object(
                WorkerLauncher,
                "collect_detached_result",
                side_effect=first_none_then_real,
            ):
                await supervisor.collect_finished_results(run_id)

        assert not (repo / ".codex_session_meta.json").exists()
        assert not (repo / ".codex_session.log").exists()
        assert not (repo / ".codex_session_active").exists()


# ---------------------------------------------------------------------------
# Fail-closed: no deliverable still marks needs_human
# ---------------------------------------------------------------------------


class TestPidAliveFallbackNoDeliverableBlocked:
    """Verify fallback with no deliverable still marks needs_human."""

    @pytest.mark.asyncio
    async def test_no_commits_marks_needs_human(
        self, repo: Path, store: DevCoordinationStore
    ) -> None:
        """Fallback with empty commit_shas must produce needs_human."""
        supervisor = _make_supervisor(repo, store)
        head = _head(repo)
        item = _make_dispatched_item(repo, initial_head=head)
        run_id = _create_run(store, item)

        _create_session_artifacts(repo)

        no_deliverable = WorkerProcess(
            work_order_id="wo-fallback",
            agent="codex",
            worktree_path=str(repo),
            branch="main",
            exit_code=141,
            changed_paths=[],
            commit_shas=[],
            head_sha=head,
        )

        with (
            patch.object(
                supervisor.launcher,
                "collect_finished",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch.object(
                WorkerLauncher,
                "collect_detached_result",
                new_callable=AsyncMock,
                side_effect=[None, no_deliverable],
            ),
            patch.object(
                supervisor.launcher,
                "snapshot_progress",
                new_callable=AsyncMock,
                return_value={"pid_alive": False},
            ),
        ):
            await supervisor.collect_finished_results(run_id)

        record = store.get_supervisor_run(run_id)
        updated_item = record["work_orders"][0]
        assert updated_item["status"] == "needs_human"
        assert "exited without receipt" in updated_item.get("dispatch_error", "")

    @pytest.mark.asyncio
    async def test_collection_exception_still_blocks(
        self, repo: Path, store: DevCoordinationStore
    ) -> None:
        """If collect_detached_result raises, the item should still be needs_human."""
        supervisor = _make_supervisor(repo, store)
        item = _make_dispatched_item(repo)
        run_id = _create_run(store, item)

        call_count = 0

        async def failing_on_second_call(**kwargs: Any) -> WorkerProcess | None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return None
            raise OSError("git broken")

        with (
            patch.object(
                supervisor.launcher,
                "collect_finished",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch.object(
                WorkerLauncher,
                "collect_detached_result",
                side_effect=failing_on_second_call,
            ),
            patch.object(
                supervisor.launcher,
                "snapshot_progress",
                new_callable=AsyncMock,
                return_value={"pid_alive": False},
            ),
        ):
            await supervisor.collect_finished_results(run_id)

        record = store.get_supervisor_run(run_id)
        updated_item = record["work_orders"][0]
        assert updated_item["status"] == "needs_human"
        assert "exited without receipt" in updated_item.get("dispatch_error", "")
