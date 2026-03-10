"""Regression tests for no-progress timeout cleanup (#899).

Covers the bug where the supervisor's no-progress timeout path bypassed
``collect_detached_result()`` entirely, so session artifacts were never
cleaned and any salvageable deliverable from the timed-out worker was lost.

The fix routes timeout termination through detached result collection,
ensuring the try/finally cleanup from #896 fires and any concrete
deliverable is surfaced through the normal result path.
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
        "session_id": "test-timeout",
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
    work_order_id: str = "wo-timeout",
    initial_head: str | None = None,
) -> dict[str, Any]:
    """Build a dispatched work-order item with an expired no-progress window."""
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
    """Create a supervisor run and return its run_id."""
    record = store.create_supervisor_run(
        goal="test timeout",
        target_branch="main",
        supervisor_agents={},
        approval_policy={},
        spec={"raw_goal": "test timeout"},
        work_orders=[item],
    )
    return record["run_id"]


def _mock_kill_worker(item: dict[str, Any]) -> None:
    """Simulate _kill_worker: pop pid from item (matching real behavior)."""
    item.pop("pid", None)


# ---------------------------------------------------------------------------
# Core regression: timeout now routes through result collection
# ---------------------------------------------------------------------------


class TestTimeoutRoutesThoughCollection:
    """Verify the no-progress timeout calls collect_detached_result."""

    @pytest.mark.asyncio
    async def test_timeout_calls_collect_detached_result(
        self, repo: Path, store: DevCoordinationStore
    ) -> None:
        """The no-progress timeout must attempt detached result collection."""
        supervisor = _make_supervisor(repo, store)
        item = _make_dispatched_item(repo)
        run_id = _create_run(store, item)

        _create_session_artifacts(repo)

        no_deliverable = WorkerProcess(
            work_order_id="wo-timeout",
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
                supervisor.launcher, "collect_finished", new_callable=AsyncMock, return_value=[]
            ),
            patch.object(supervisor, "_exceeded_no_progress_timeout", return_value=True),
            patch.object(
                supervisor, "_kill_worker", new_callable=AsyncMock, side_effect=_mock_kill_worker
            ),
            # First call (line-445 normal path) returns None (worker still running);
            # second call (timeout block) returns the result.
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
                return_value={"pid_alive": True},
            ),
        ):
            await supervisor.collect_finished_results(run_id)

        assert mock_collect.call_count == 2


class TestTimeoutCleansSessionArtifacts:
    """Verify no-progress timeout cleans session artifacts via result collection."""

    @pytest.mark.asyncio
    async def test_artifacts_cleaned_on_timeout(
        self, repo: Path, store: DevCoordinationStore
    ) -> None:
        """Session artifacts must be removed after timeout result collection."""
        supervisor = _make_supervisor(repo, store)
        head = _head(repo)
        item = _make_dispatched_item(repo, initial_head=head)
        run_id = _create_run(store, item)

        _create_session_artifacts(repo)
        assert (repo / ".codex_session_meta.json").exists()

        with (
            patch.object(
                supervisor.launcher, "collect_finished", new_callable=AsyncMock, return_value=[]
            ),
            patch.object(supervisor, "_exceeded_no_progress_timeout", return_value=True),
            patch.object(
                supervisor, "_kill_worker", new_callable=AsyncMock, side_effect=_mock_kill_worker
            ),
            # Make the first collect call return None (worker still alive)
            patch.object(WorkerLauncher, "_is_pid_running", return_value=True),
            patch.object(
                supervisor.launcher,
                "snapshot_progress",
                new_callable=AsyncMock,
                return_value={"pid_alive": True},
            ),
        ):
            # Let real collect_detached_result run.
            # First call: _is_pid_running=True → returns None.
            # Kill pops pid, so second call: pid=None → proceeds, try/finally cleans up.
            await supervisor.collect_finished_results(run_id)

        # All session artifacts must be gone
        assert not (repo / ".codex_session_meta.json").exists()
        assert not (repo / ".codex_session.log").exists()
        assert not (repo / ".codex_session_active").exists()


class TestTimeoutNoDeliverableBlocked:
    """Verify timeout with no deliverable marks needs_human."""

    @pytest.mark.asyncio
    async def test_timeout_no_commits_marks_needs_human(
        self, repo: Path, store: DevCoordinationStore
    ) -> None:
        """Timeout with empty commit_shas must produce needs_human, not completed."""
        supervisor = _make_supervisor(repo, store)
        head = _head(repo)
        item = _make_dispatched_item(repo, initial_head=head)
        run_id = _create_run(store, item)

        _create_session_artifacts(repo)

        with (
            patch.object(
                supervisor.launcher, "collect_finished", new_callable=AsyncMock, return_value=[]
            ),
            patch.object(supervisor, "_exceeded_no_progress_timeout", return_value=True),
            patch.object(
                supervisor, "_kill_worker", new_callable=AsyncMock, side_effect=_mock_kill_worker
            ),
            patch.object(WorkerLauncher, "_is_pid_running", return_value=True),
            patch.object(
                supervisor.launcher,
                "snapshot_progress",
                new_callable=AsyncMock,
                return_value={"pid_alive": True},
            ),
        ):
            await supervisor.collect_finished_results(run_id)

        record = store.get_supervisor_run(run_id)
        updated_item = record["work_orders"][0]
        assert updated_item["status"] == "needs_human"
        assert "no-progress timeout" in updated_item.get("dispatch_error", "")


class TestTimeoutWithDeliverableSurfaced:
    """Verify timeout with salvageable deliverable surfaces through normal path."""

    @pytest.mark.asyncio
    async def test_timeout_with_commits_applies_result(
        self, repo: Path, store: DevCoordinationStore
    ) -> None:
        """Timeout with concrete commit_shas should apply the result normally."""
        supervisor = _make_supervisor(repo, store)
        head = _head(repo)

        # Create a real commit in the repo to simulate worker deliverable
        (repo / "README.md").write_text("updated\n", encoding="utf-8")
        _run(repo, "git", "add", "README.md")
        _run(repo, "git", "commit", "-m", "worker deliverable")
        new_head = _head(repo)

        item = _make_dispatched_item(repo, initial_head=head)
        run_id = _create_run(store, item)

        _create_session_artifacts(repo)

        salvaged_result = WorkerProcess(
            work_order_id="wo-timeout",
            agent="codex",
            worktree_path=str(repo),
            branch="main",
            exit_code=0,
            changed_paths=["README.md"],
            commit_shas=[new_head],
            head_sha=new_head,
        )

        with (
            patch.object(
                supervisor.launcher, "collect_finished", new_callable=AsyncMock, return_value=[]
            ),
            patch.object(supervisor, "_exceeded_no_progress_timeout", return_value=True),
            patch.object(
                supervisor, "_kill_worker", new_callable=AsyncMock, side_effect=_mock_kill_worker
            ),
            # First call returns None (worker still running), second returns salvaged result
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
                return_value={"pid_alive": True},
            ),
        ):
            results = await supervisor.collect_finished_results(run_id)

        # The salvaged result should be in the finished list
        assert len(results) == 1
        assert results[0].commit_shas == [new_head]

        # Work order should be processed through normal _apply_worker_result
        record = store.get_supervisor_run(run_id)
        updated_item = record["work_orders"][0]
        # Should NOT be needs_human since there's a real deliverable
        assert updated_item["status"] != "needs_human"


class TestTimeoutKillsWorker:
    """Verify timeout kills the worker process before collection."""

    @pytest.mark.asyncio
    async def test_kill_worker_called_on_timeout(
        self, repo: Path, store: DevCoordinationStore
    ) -> None:
        """The worker must be killed before result collection on timeout."""
        supervisor = _make_supervisor(repo, store)
        item = _make_dispatched_item(repo)
        run_id = _create_run(store, item)

        _create_session_artifacts(repo)
        call_order: list[str] = []

        async def tracking_kill(it: dict[str, Any]) -> None:
            call_order.append("kill")
            _mock_kill_worker(it)

        async def tracking_collect(**kwargs: Any) -> WorkerProcess | None:
            if not call_order:
                # First call (line-445 normal path) — worker still running
                return None
            call_order.append("collect")
            return WorkerProcess(
                work_order_id="wo-timeout",
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
                supervisor.launcher, "collect_finished", new_callable=AsyncMock, return_value=[]
            ),
            patch.object(supervisor, "_exceeded_no_progress_timeout", return_value=True),
            patch.object(supervisor, "_kill_worker", side_effect=tracking_kill),
            patch.object(WorkerLauncher, "collect_detached_result", side_effect=tracking_collect),
            patch.object(
                supervisor.launcher,
                "snapshot_progress",
                new_callable=AsyncMock,
                return_value={"pid_alive": True},
            ),
        ):
            await supervisor.collect_finished_results(run_id)

        assert call_order == ["kill", "collect"]


class TestTimeoutCollectionFailureStillMarksBlocked:
    """Verify timeout still marks needs_human even if collection raises."""

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
                return None  # First call: worker still running
            raise OSError("git broken")

        with (
            patch.object(
                supervisor.launcher, "collect_finished", new_callable=AsyncMock, return_value=[]
            ),
            patch.object(supervisor, "_exceeded_no_progress_timeout", return_value=True),
            patch.object(
                supervisor, "_kill_worker", new_callable=AsyncMock, side_effect=_mock_kill_worker
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
                return_value={"pid_alive": True},
            ),
        ):
            await supervisor.collect_finished_results(run_id)

        record = store.get_supervisor_run(run_id)
        updated_item = record["work_orders"][0]
        assert updated_item["status"] == "needs_human"
        assert "no-progress timeout" in updated_item.get("dispatch_error", "")
