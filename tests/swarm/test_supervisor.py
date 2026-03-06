"""Tests for supervisor-backed swarm execution."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from aragora.nomic.dev_coordination import DevCoordinationStore
from aragora.nomic.task_decomposer import SubTask, TaskDecomposition
from aragora.swarm.spec import SwarmSpec
from aragora.swarm.supervisor import SwarmSupervisor
from aragora.worktree.lifecycle import ManagedWorktreeSession


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _run(repo, "git", "init", "-b", "main")
    _run(repo, "git", "config", "user.email", "test@example.com")
    _run(repo, "git", "config", "user.name", "Test User")
    (repo / "README.md").write_text("hello\n", encoding="utf-8")
    _run(repo, "git", "add", "README.md")
    _run(repo, "git", "commit", "-m", "initial")
    _run(repo, "git", "remote", "add", "origin", str(repo))
    _run(repo, "git", "update-ref", "refs/remotes/origin/main", "HEAD")
    return repo


@pytest.fixture()
def store(repo: Path) -> DevCoordinationStore:
    return DevCoordinationStore(repo_root=repo)


def _run(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(args),
        cwd=cwd,
        text=True,
        capture_output=True,
        check=True,
    )


def test_start_run_creates_leased_work_orders(repo: Path, store: DevCoordinationStore) -> None:
    sessions = [
        ManagedWorktreeSession(
            session_id="swarm-a",
            agent="codex",
            branch="codex/swarm-a",
            path=repo / "wt-a",
            created=True,
            reconcile_status="up_to_date",
            payload={},
        ),
        ManagedWorktreeSession(
            session_id="swarm-b",
            agent="claude",
            branch="codex/swarm-b",
            path=repo / "wt-b",
            created=True,
            reconcile_status="up_to_date",
            payload={},
        ),
    ]
    sessions[0].path.mkdir()
    sessions[1].path.mkdir()

    lifecycle = MagicMock()
    lifecycle.ensure_managed_worktree.side_effect = sessions
    decomposer = MagicMock()
    decomposer.analyze.return_value = TaskDecomposition(
        original_task="Goal",
        complexity_score=8,
        complexity_level="high",
        should_decompose=True,
        subtasks=[
            SubTask(
                id="wo-1",
                title="Server lane",
                description="Implement server lane",
                file_scope=["aragora/server/handlers/foo.py"],
            ),
            SubTask(
                id="wo-2",
                title="Test lane",
                description="Implement test lane",
                file_scope=["tests/server/test_foo.py"],
            ),
        ],
    )
    supervisor = SwarmSupervisor(
        repo_root=repo,
        store=store,
        lifecycle=lifecycle,
        decomposer=decomposer,
    )
    spec = SwarmSpec(raw_goal="Goal", refined_goal="Goal")

    run = supervisor.start_run(spec=spec, max_concurrency=2)

    assert run.status == "active"
    assert len(run.work_orders) == 2
    assert {item["status"] for item in run.work_orders} == {"leased"}
    assert store.status_summary()["counts"]["supervisor_runs"] == 1
    assert store.status_summary()["counts"]["active_leases"] == 2


def test_refresh_run_scales_queued_work_after_completion(
    repo: Path, store: DevCoordinationStore
) -> None:
    counter = {"value": 0}

    def _ensure_session(**_kwargs: object) -> ManagedWorktreeSession:
        counter["value"] += 1
        path = repo / f"wt-{counter['value']}"
        path.mkdir(exist_ok=True)
        return ManagedWorktreeSession(
            session_id=f"swarm-{counter['value']}",
            agent="codex" if counter["value"] % 2 else "claude",
            branch=f"codex/swarm-{counter['value']}",
            path=path,
            created=True,
            reconcile_status="up_to_date",
            payload={},
        )

    lifecycle = MagicMock()
    lifecycle.ensure_managed_worktree.side_effect = _ensure_session
    decomposer = MagicMock()
    decomposer.analyze.return_value = TaskDecomposition(
        original_task="Goal",
        complexity_score=8,
        complexity_level="high",
        should_decompose=True,
        subtasks=[
            SubTask(
                id="wo-1",
                title="Lane one",
                description="Lane one",
                file_scope=["aragora/server/handlers/foo.py"],
            ),
            SubTask(
                id="wo-2",
                title="Lane two",
                description="Lane two",
                file_scope=["tests/server/test_foo.py"],
            ),
        ],
    )
    supervisor = SwarmSupervisor(
        repo_root=repo,
        store=store,
        lifecycle=lifecycle,
        decomposer=decomposer,
    )
    spec = SwarmSpec(raw_goal="Goal", refined_goal="Goal")

    run = supervisor.start_run(spec=spec, max_concurrency=1)
    leased = [item for item in run.work_orders if item["status"] == "leased"]
    queued = [item for item in run.work_orders if item["status"] == "queued"]
    assert len(leased) == 1
    assert len(queued) == 1

    first = leased[0]
    store.record_completion(
        lease_id=str(first["lease_id"]),
        owner_agent=str(first["target_agent"]),
        owner_session_id=str(first["owner_session_id"]),
        branch=str(first["branch"]),
        worktree_path=str(first["worktree_path"]),
        commit_shas=["abc12345"],
        changed_paths=["aragora/server/handlers/foo.py"],
    )

    refreshed = supervisor.refresh_run(run.run_id)
    leased_after = [item for item in refreshed.work_orders if item["status"] == "leased"]
    completed_after = [item for item in refreshed.work_orders if item["status"] == "completed"]

    assert len(leased_after) == 1
    assert len(completed_after) == 1
    assert counter["value"] >= 2


# ---------- dispatch_workers / collect_results tests ----------

from unittest.mock import AsyncMock, patch
from aragora.swarm.worker_launcher import WorkerLauncher, WorkerProcess


@pytest.mark.asyncio
async def test_dispatch_workers_launches_leased_orders(
    repo: Path, store: DevCoordinationStore
) -> None:
    """dispatch_workers should call launcher.launch for each leased work order."""
    sessions = [
        ManagedWorktreeSession(
            path=repo,
            branch="swarm-dispatch-1",
            session_id="dispatch-sess-1",
            agent="claude",
            created=True,
            reconcile_status=None,
            payload={},
        )
    ]
    session_iter = iter(sessions)

    lifecycle = MagicMock()
    lifecycle.ensure_managed_worktree = MagicMock(side_effect=lambda **kw: next(session_iter))

    decomposer = MagicMock()
    decomposer.analyze.return_value = TaskDecomposition(
        original_task="dispatch test",
        complexity_score=3,
        complexity_level="moderate",
        should_decompose=True,
        subtasks=[
            SubTask(
                id="dispatch-task",
                title="Dispatch test",
                description="Test dispatch",
                file_scope=["aragora/test.py"],
            )
        ],
    )

    mock_launcher = MagicMock(spec=WorkerLauncher)
    mock_worker = WorkerProcess(
        work_order_id="dispatch-task",
        agent="claude",
        worktree_path=str(repo),
        branch="swarm-dispatch-1",
        pid=999,
    )
    mock_launcher.launch = AsyncMock(return_value=mock_worker)

    supervisor = SwarmSupervisor(
        repo_root=repo,
        store=store,
        lifecycle=lifecycle,
        decomposer=decomposer,
        launcher=mock_launcher,
    )

    spec = SwarmSpec.from_dict(
        {
            "raw_goal": "dispatch test",
            "refined_goal": "dispatch test",
        }
    )

    run = supervisor.start_run(spec=spec, refresh_scaling=True)
    leased = [w for w in run.work_orders if w.get("status") == "leased"]
    assert len(leased) >= 1

    launched = await supervisor.dispatch_workers(run.run_id)
    assert len(launched) >= 1
    assert launched[0].pid == 999
    mock_launcher.launch.assert_called()


@pytest.mark.asyncio
async def test_collect_results_updates_work_orders(repo: Path, store: DevCoordinationStore) -> None:
    """collect_results should wait for dispatched workers and update statuses."""
    # Create a run with one dispatched work order
    run_record = store.create_supervisor_run(
        goal="collect test",
        target_branch="main",
        supervisor_agents={"planner": "codex"},
        approval_policy={},
        spec={"raw_goal": "collect test"},
        work_orders=[
            {
                "work_order_id": "wo-collect",
                "status": "dispatched",
                "worktree_path": str(repo),
                "branch": "main",
                "target_agent": "claude",
            }
        ],
        status="active",
    )
    run_id = run_record["run_id"]

    # Mock launcher with a completed worker
    mock_launcher = MagicMock(spec=WorkerLauncher)
    completed_worker = WorkerProcess(
        work_order_id="wo-collect",
        agent="claude",
        worktree_path=str(repo),
        branch="main",
        pid=100,
        exit_code=0,
        completed_at="2026-03-06T20:00:00+00:00",
        diff="diff --git a/test.py",
    )
    mock_launcher.get_worker = MagicMock(return_value=completed_worker)
    mock_launcher.wait = AsyncMock(return_value=completed_worker)

    supervisor = SwarmSupervisor(
        repo_root=repo,
        store=store,
        launcher=mock_launcher,
    )

    results = await supervisor.collect_results(run_id)
    assert len(results) == 1
    assert results[0].exit_code == 0

    # Verify the run was updated
    updated = store.get_supervisor_run(run_id)
    wo = updated["work_orders"][0]
    assert wo["status"] == "completed"


@pytest.mark.asyncio
async def test_dispatch_handles_missing_cli(repo: Path, store: DevCoordinationStore) -> None:
    """dispatch_workers should mark orders as dispatch_failed when CLI is missing."""
    run_record = store.create_supervisor_run(
        goal="missing cli test",
        target_branch="main",
        supervisor_agents={},
        approval_policy={},
        spec={"raw_goal": "test"},
        work_orders=[
            {
                "work_order_id": "wo-fail",
                "status": "leased",
                "worktree_path": str(repo),
                "branch": "main",
                "target_agent": "claude",
            }
        ],
        status="active",
    )
    run_id = run_record["run_id"]

    mock_launcher = MagicMock(spec=WorkerLauncher)
    mock_launcher.launch = AsyncMock(side_effect=FileNotFoundError("claude CLI not found"))

    supervisor = SwarmSupervisor(
        repo_root=repo,
        store=store,
        launcher=mock_launcher,
    )

    launched = await supervisor.dispatch_workers(run_id)
    assert len(launched) == 0

    updated = store.get_supervisor_run(run_id)
    wo = updated["work_orders"][0]
    assert wo["status"] == "dispatch_failed"
    assert "CLI not found" in wo["dispatch_error"]
