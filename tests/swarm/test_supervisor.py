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
