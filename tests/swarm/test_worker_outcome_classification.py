"""Tests for WorkerOutcome classification (#908).

Verifies that the supervisor sets ``worker_outcome`` on work-order items
for each distinct terminal state: completed, clean_exit_no_effect, crash,
crash_with_salvage, timeout_no_progress, timeout_with_salvage,
scope_violation.
"""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from aragora.nomic.dev_coordination import DevCoordinationStore
from aragora.swarm.supervisor import SwarmSupervisor, WorkerOutcome
from aragora.swarm.worker_launcher import (
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


def _make_supervisor(repo: Path, store: DevCoordinationStore) -> SwarmSupervisor:
    config = LaunchConfig(no_progress_timeout_seconds=1.0)
    launcher = WorkerLauncher(config=config)
    return SwarmSupervisor(repo_root=repo, store=store, launcher=launcher)


def _make_dispatched_item(
    repo: Path,
    *,
    work_order_id: str = "wo-outcome",
    initial_head: str | None = None,
) -> dict[str, Any]:
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
        goal="test outcome",
        target_branch="main",
        supervisor_agents={},
        approval_policy={},
        spec={"raw_goal": "test outcome"},
        work_orders=[item],
    )
    return record["run_id"]


def _mock_kill_worker(item: dict[str, Any]) -> None:
    item.pop("pid", None)


# ---------------------------------------------------------------------------
# COMPLETED: exit 0 with real deliverable
# ---------------------------------------------------------------------------


class TestOutcomeCompleted:
    @pytest.mark.asyncio
    async def test_exit_0_with_commits_is_completed(
        self, repo: Path, store: DevCoordinationStore
    ) -> None:
        supervisor = _make_supervisor(repo, store)
        head = _head(repo)

        (repo / "README.md").write_text("updated\n", encoding="utf-8")
        _run(repo, "git", "add", "README.md")
        _run(repo, "git", "commit", "-m", "worker change")
        new_head = _head(repo)

        item = _make_dispatched_item(repo, initial_head=head)
        run_id = _create_run(store, item)

        result = WorkerProcess(
            work_order_id="wo-outcome",
            agent="codex",
            worktree_path=str(repo),
            branch="main",
            exit_code=0,
            changed_paths=["README.md"],
            commit_shas=[new_head],
            head_sha=new_head,
        )

        with patch.object(
            supervisor.launcher,
            "collect_finished",
            new_callable=AsyncMock,
            return_value=[result],
        ):
            await supervisor.collect_finished_results(run_id)

        record = store.get_supervisor_run(run_id)
        wo = record["work_orders"][0]
        assert wo["worker_outcome"] == WorkerOutcome.COMPLETED.value


# ---------------------------------------------------------------------------
# CLEAN_EXIT_NO_EFFECT: exit 0, zero commits, zero changed paths
# ---------------------------------------------------------------------------


class TestOutcomeCleanExitNoEffect:
    @pytest.mark.asyncio
    async def test_exit_0_no_changes_is_clean_exit_no_effect(
        self, repo: Path, store: DevCoordinationStore
    ) -> None:
        supervisor = _make_supervisor(repo, store)
        head = _head(repo)
        item = _make_dispatched_item(repo, initial_head=head)
        run_id = _create_run(store, item)

        result = WorkerProcess(
            work_order_id="wo-outcome",
            agent="codex",
            worktree_path=str(repo),
            branch="main",
            exit_code=0,
            changed_paths=[],
            commit_shas=[],
            head_sha=head,
        )

        with patch.object(
            supervisor.launcher,
            "collect_finished",
            new_callable=AsyncMock,
            return_value=[result],
        ):
            await supervisor.collect_finished_results(run_id)

        record = store.get_supervisor_run(run_id)
        wo = record["work_orders"][0]
        assert wo["worker_outcome"] == WorkerOutcome.CLEAN_EXIT_NO_EFFECT.value

    @pytest.mark.asyncio
    async def test_exit_0_only_session_artifacts_is_clean_exit_no_effect(
        self, repo: Path, store: DevCoordinationStore
    ) -> None:
        """Worker produced only session artifacts (stripped to empty)."""
        supervisor = _make_supervisor(repo, store)
        head = _head(repo)
        item = _make_dispatched_item(repo, initial_head=head)
        run_id = _create_run(store, item)

        result = WorkerProcess(
            work_order_id="wo-outcome",
            agent="codex",
            worktree_path=str(repo),
            branch="main",
            exit_code=0,
            changed_paths=[".codex_session_meta.json", ".codex_session.log"],
            commit_shas=["abc123"],
            head_sha=head,
        )

        with patch.object(
            supervisor.launcher,
            "collect_finished",
            new_callable=AsyncMock,
            return_value=[result],
        ):
            await supervisor.collect_finished_results(run_id)

        record = store.get_supervisor_run(run_id)
        wo = record["work_orders"][0]
        assert wo["worker_outcome"] == WorkerOutcome.CLEAN_EXIT_NO_EFFECT.value
        assert wo["status"] == "needs_human"


# ---------------------------------------------------------------------------
# CRASH: non-zero exit, no salvage
# ---------------------------------------------------------------------------


class TestOutcomeCrash:
    @pytest.mark.asyncio
    async def test_nonzero_exit_no_commits_is_crash(
        self, repo: Path, store: DevCoordinationStore
    ) -> None:
        supervisor = _make_supervisor(repo, store)
        head = _head(repo)
        item = _make_dispatched_item(repo, initial_head=head)
        run_id = _create_run(store, item)

        result = WorkerProcess(
            work_order_id="wo-outcome",
            agent="codex",
            worktree_path=str(repo),
            branch="main",
            exit_code=1,
            changed_paths=[],
            commit_shas=[],
            head_sha=head,
        )

        with patch.object(
            supervisor.launcher,
            "collect_finished",
            new_callable=AsyncMock,
            return_value=[result],
        ):
            await supervisor.collect_finished_results(run_id)

        record = store.get_supervisor_run(run_id)
        wo = record["work_orders"][0]
        assert wo["worker_outcome"] == WorkerOutcome.CRASH.value
        assert wo["status"] == "failed"


# ---------------------------------------------------------------------------
# CRASH_WITH_SALVAGE: pid_alive=False fallback with commit_shas
# ---------------------------------------------------------------------------


class TestOutcomeCrashWithSalvage:
    @pytest.mark.asyncio
    async def test_pid_dead_with_salvage_is_crash_with_salvage(
        self, repo: Path, store: DevCoordinationStore
    ) -> None:
        supervisor = _make_supervisor(repo, store)
        head = _head(repo)

        (repo / "README.md").write_text("salvaged\n", encoding="utf-8")
        _run(repo, "git", "add", "README.md")
        _run(repo, "git", "commit", "-m", "salvaged work")
        new_head = _head(repo)

        item = _make_dispatched_item(repo, initial_head=head)
        run_id = _create_run(store, item)

        salvaged = WorkerProcess(
            work_order_id="wo-outcome",
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
                side_effect=[None, salvaged],
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
        wo = record["work_orders"][0]
        assert wo["worker_outcome"] == WorkerOutcome.CRASH_WITH_SALVAGE.value


# ---------------------------------------------------------------------------
# CRASH (via pid_alive=False fallback, no salvage)
# ---------------------------------------------------------------------------


class TestOutcomeCrashViaPidFallback:
    @pytest.mark.asyncio
    async def test_pid_dead_no_salvage_is_crash(
        self, repo: Path, store: DevCoordinationStore
    ) -> None:
        supervisor = _make_supervisor(repo, store)
        item = _make_dispatched_item(repo)
        run_id = _create_run(store, item)

        no_deliverable = WorkerProcess(
            work_order_id="wo-outcome",
            agent="codex",
            worktree_path=str(repo),
            branch="main",
            exit_code=141,
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
        wo = record["work_orders"][0]
        assert wo["worker_outcome"] == WorkerOutcome.CRASH.value
        assert wo["status"] == "needs_human"


# ---------------------------------------------------------------------------
# TIMEOUT_NO_PROGRESS: no-progress timeout, no salvage
# ---------------------------------------------------------------------------


class TestOutcomeTimeoutNoProgress:
    @pytest.mark.asyncio
    async def test_timeout_no_salvage_is_timeout_no_progress(
        self, repo: Path, store: DevCoordinationStore
    ) -> None:
        supervisor = _make_supervisor(repo, store)
        item = _make_dispatched_item(repo)
        run_id = _create_run(store, item)

        no_deliverable = WorkerProcess(
            work_order_id="wo-outcome",
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
            patch.object(supervisor, "_exceeded_no_progress_timeout", return_value=True),
            patch.object(
                supervisor, "_kill_worker", new_callable=AsyncMock, side_effect=_mock_kill_worker
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
                return_value={"pid_alive": True},
            ),
        ):
            await supervisor.collect_finished_results(run_id)

        record = store.get_supervisor_run(run_id)
        wo = record["work_orders"][0]
        assert wo["worker_outcome"] == WorkerOutcome.TIMEOUT_NO_PROGRESS.value
        assert wo["status"] == "needs_human"


# ---------------------------------------------------------------------------
# TIMEOUT_WITH_SALVAGE: no-progress timeout, but salvageable commits
# ---------------------------------------------------------------------------


class TestOutcomeTimeoutWithSalvage:
    @pytest.mark.asyncio
    async def test_timeout_with_salvage_is_timeout_with_salvage(
        self, repo: Path, store: DevCoordinationStore
    ) -> None:
        supervisor = _make_supervisor(repo, store)
        head = _head(repo)

        (repo / "README.md").write_text("timeout salvage\n", encoding="utf-8")
        _run(repo, "git", "add", "README.md")
        _run(repo, "git", "commit", "-m", "partial work")
        new_head = _head(repo)

        item = _make_dispatched_item(repo, initial_head=head)
        run_id = _create_run(store, item)

        salvaged = WorkerProcess(
            work_order_id="wo-outcome",
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
                supervisor.launcher,
                "collect_finished",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch.object(supervisor, "_exceeded_no_progress_timeout", return_value=True),
            patch.object(
                supervisor, "_kill_worker", new_callable=AsyncMock, side_effect=_mock_kill_worker
            ),
            patch.object(
                WorkerLauncher,
                "collect_detached_result",
                new_callable=AsyncMock,
                side_effect=[None, salvaged],
            ),
            patch.object(
                supervisor.launcher,
                "snapshot_progress",
                new_callable=AsyncMock,
                return_value={"pid_alive": True},
            ),
        ):
            results = await supervisor.collect_finished_results(run_id)

        assert len(results) == 1
        record = store.get_supervisor_run(run_id)
        wo = record["work_orders"][0]
        assert wo["worker_outcome"] == WorkerOutcome.TIMEOUT_WITH_SALVAGE.value


# ---------------------------------------------------------------------------
# SCOPE_VIOLATION: worker touched out-of-scope files
# ---------------------------------------------------------------------------


class TestOutcomeScopeViolation:
    @pytest.mark.asyncio
    async def test_scope_violation_is_scope_violation(
        self, repo: Path, store: DevCoordinationStore
    ) -> None:
        supervisor = _make_supervisor(repo, store)
        head = _head(repo)
        item = _make_dispatched_item(repo, initial_head=head)
        item["file_scope"] = ["aragora/live/**"]
        run_id = _create_run(store, item)

        result = WorkerProcess(
            work_order_id="wo-outcome",
            agent="codex",
            worktree_path=str(repo),
            branch="main",
            exit_code=0,
            changed_paths=["aragora/server/main.py"],
            commit_shas=[head],
            head_sha=head,
        )

        with patch.object(
            supervisor.launcher,
            "collect_finished",
            new_callable=AsyncMock,
            return_value=[result],
        ):
            await supervisor.collect_finished_results(run_id)

        record = store.get_supervisor_run(run_id)
        wo = record["work_orders"][0]
        assert wo["worker_outcome"] == WorkerOutcome.SCOPE_VIOLATION.value
        assert wo["status"] == "scope_violation"


# ---------------------------------------------------------------------------
# Boss loop: worker_outcome surfaces in iteration status
# ---------------------------------------------------------------------------


class TestBossLoopOutcomeSurface:
    def test_extract_worker_outcome_from_run_dict(self) -> None:
        from aragora.swarm.boss_loop import _extract_worker_outcome

        run_dict: dict[str, Any] = {
            "work_orders": [
                {"worker_outcome": "clean_exit_no_effect", "status": "completed"},
            ]
        }
        assert _extract_worker_outcome(run_dict) == "clean_exit_no_effect"

    def test_extract_worker_outcome_none_when_missing(self) -> None:
        from aragora.swarm.boss_loop import _extract_worker_outcome

        run_dict: dict[str, Any] = {"work_orders": [{"status": "completed"}]}
        assert _extract_worker_outcome(run_dict) is None

    def test_iteration_status_includes_worker_outcome(self) -> None:
        from aragora.swarm.boss_loop import BossIterationStatus

        status = BossIterationStatus(
            iteration=1,
            run_id="test",
            timestamp="2026-01-01T00:00:00Z",
            runner_freshness={},
            selected_issue=None,
            worker_status="needs_human",
            stop_reason="needs_human",
            needs_human_reasons=["test"],
            next_actions=[],
            worker_outcome="clean_exit_no_effect",
        )
        d = status.to_dict()
        assert d["worker_outcome"] == "clean_exit_no_effect"

    def test_iteration_status_omits_worker_outcome_when_none(self) -> None:
        from aragora.swarm.boss_loop import BossIterationStatus

        status = BossIterationStatus(
            iteration=1,
            run_id="test",
            timestamp="2026-01-01T00:00:00Z",
            runner_freshness={},
            selected_issue=None,
            worker_status="idle",
            stop_reason=None,
            needs_human_reasons=[],
            next_actions=[],
        )
        d = status.to_dict()
        assert "worker_outcome" not in d
