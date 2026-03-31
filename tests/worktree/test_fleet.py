"""Tests for shared fleet coordination utilities."""

from __future__ import annotations

import json
import os
import sqlite3
import shutil
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import aragora.worktree.fleet as fleet
from aragora.nomic.dev_coordination import DevCoordinationStore
from aragora.worktree.fleet import (
    FleetCoordinationStore,
    infer_orchestration_pattern,
)


def _run(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(args),
        cwd=cwd,
        text=True,
        capture_output=True,
        check=True,
    )


def _make_repo(tmp_path: Path) -> Path:
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


def _backdate_all_claims(store: FleetCoordinationStore, *, hours: int = 2) -> None:
    state = json.loads(store.path.read_text(encoding="utf-8"))
    old_ts = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    for claim in state.get("claims", []):
        claim["claimed_at"] = old_ts
        claim["updated_at"] = old_ts
    store.path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def test_infer_orchestration_pattern_from_framework() -> None:
    pattern = infer_orchestration_pattern({"framework": "CrewAI"})
    assert pattern == "crewai"


def test_infer_orchestration_pattern_from_command() -> None:
    pattern = infer_orchestration_pattern({"command": "python scripts/gastown_migrate_state.py"})
    assert pattern == "gastown"


def test_claim_paths_detects_conflicts(tmp_path: Path) -> None:
    store = FleetCoordinationStore(tmp_path)
    first = store.claim_paths(
        session_id="session-a",
        paths=["aragora/server/handlers/a.py"],
        mode="exclusive",
    )
    assert first["conflicts"] == []

    second = store.claim_paths(
        session_id="session-b",
        paths=["aragora/server/handlers/a.py"],
        mode="exclusive",
    )
    assert len(second["conflicts"]) == 1
    assert second["conflicts"][0]["session_id"] == "session-a"


def test_claim_paths_detects_glob_to_file_conflicts(tmp_path: Path) -> None:
    store = FleetCoordinationStore(tmp_path)
    first = store.claim_paths(
        session_id="session-a",
        paths=["aragora/server/**"],
        mode="exclusive",
    )
    assert first["conflicts"] == []

    second = store.claim_paths(
        session_id="session-b",
        paths=["aragora/server/handlers/a.py"],
        mode="exclusive",
    )
    assert len(second["conflicts"]) == 1
    assert second["conflicts"][0]["session_id"] == "session-a"


def test_audit_session_paths_detects_unowned_and_foreign_claims(tmp_path: Path) -> None:
    store = FleetCoordinationStore(tmp_path)
    store.claim_paths(
        session_id="session-a",
        paths=["aragora/server/**"],
        mode="exclusive",
    )
    store.claim_paths(
        session_id="session-b",
        paths=["aragora/cli/**"],
        mode="exclusive",
    )

    audit = store.audit_session_paths(
        session_id="session-a",
        paths=["aragora/server/handlers/a.py", "aragora/cli/main.py"],
        branch="codex/session-a",
    )

    assert audit["owned_paths"] == ["aragora/server/handlers/a.py"]
    assert audit["unowned_paths"] == ["aragora/cli/main.py"]
    assert audit["conflicts"][0]["session_id"] == "session-b"
    assert audit["ok"] is False


def test_release_paths_by_subset(tmp_path: Path) -> None:
    store = FleetCoordinationStore(tmp_path)
    store.claim_paths(session_id="session-a", paths=["a.py", "b.py"])
    result = store.release_paths(session_id="session-a", paths=["a.py"])
    assert result["released"] == 1
    claims = store.list_claims()
    assert len(claims) == 1
    assert claims[0]["path"] == "b.py"


def test_enqueue_merge_deduplicates_active_branch(tmp_path: Path) -> None:
    store = FleetCoordinationStore(tmp_path)
    first = store.enqueue_merge(session_id="session-a", branch="codex/session-a", priority=70)
    assert first["queued"] is True
    second = store.enqueue_merge(session_id="session-b", branch="codex/session-a", priority=80)
    assert second["duplicate"] is True
    queue = store.list_merge_queue()
    assert len(queue) == 1


def test_claim_next_merge_prefers_highest_priority(tmp_path: Path) -> None:
    store = FleetCoordinationStore(tmp_path)
    store.enqueue_merge(session_id="session-a", branch="codex/low", priority=20)
    high = store.enqueue_merge(session_id="session-b", branch="codex/high", priority=90)

    claimed = store.claim_next_merge(worker_session_id="integrator-1")

    assert claimed is not None
    assert claimed["id"] == high["item"]["id"]
    assert claimed["status"] == "validating"
    assert claimed["metadata"]["worker_session_id"] == "integrator-1"


def test_update_merge_queue_item_persists_status_and_metadata(tmp_path: Path) -> None:
    store = FleetCoordinationStore(tmp_path)
    queued = store.enqueue_merge(session_id="session-a", branch="codex/session-a", priority=50)

    updated = store.update_merge_queue_item(
        item_id=queued["item"]["id"],
        status="integrating",
        metadata={"receipt_id": "rcpt-1", "note": "approved"},
    )

    assert updated["status"] == "integrating"
    assert updated["metadata"]["receipt_id"] == "rcpt-1"
    listed = store.list_merge_queue()
    assert listed[0]["status"] == "integrating"


def test_update_merge_queue_item_enforces_expected_status(tmp_path: Path) -> None:
    store = FleetCoordinationStore(tmp_path)
    queued = store.enqueue_merge(session_id="session-a", branch="codex/session-a", priority=50)
    item_id = queued["item"]["id"]

    store.update_merge_queue_item(
        item_id=item_id,
        status="validating",
        expected_status="queued",
        metadata={"worker_session_id": "integrator-1"},
    )

    with pytest.raises(KeyError):
        store.update_merge_queue_item(
            item_id=item_id,
            status="integrating",
            expected_status="queued",
            metadata={"worker_session_id": "integrator-2"},
        )


def test_reap_stale_claims_removes_orphaned_sessions(tmp_path: Path) -> None:
    store = FleetCoordinationStore(tmp_path)
    store.claim_paths(session_id="stale-session", paths=["aragora/live/**"])
    _backdate_all_claims(store)

    result = store.reap_stale_claims()

    assert result["released"] == 1
    assert result["reaped_sessions"][0]["session_id"] == "stale-session"
    assert store.list_claims() == []


def test_reap_stale_claims_keeps_recent_claims_inside_grace_window(tmp_path: Path) -> None:
    store = FleetCoordinationStore(tmp_path)
    store.claim_paths(session_id="recent-session", paths=["aragora/live/**"])

    result = store.reap_stale_claims()

    assert result["released"] == 0
    assert result["kept_sessions"][0]["session_id"] == "recent-session"
    assert store.list_claims()[0]["session_id"] == "recent-session"


def test_reap_stale_claims_releases_inactive_managed_sessions_even_inside_grace_window(
    tmp_path: Path,
) -> None:
    repo = _make_repo(tmp_path)
    managed_root = repo / ".worktrees" / "codex-auto"
    managed_root.mkdir(parents=True)
    worktree_path = managed_root / "swarm-test-subtask_1"
    worktree_path.mkdir(parents=True)
    (managed_root / "state.json").write_text(
        json.dumps(
            {
                "sessions": [
                    {
                        "session_id": "swarm-test-subtask_1",
                        "path": str(worktree_path),
                        "tracked_worktree": True,
                        "active_session": False,
                        "lease_status": "released",
                        "lifecycle_state": "grace",
                    }
                ]
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    store = FleetCoordinationStore(repo)
    store.claim_paths(session_id="swarm-test-subtask_1", paths=["aragora/live/**"])

    result = store.reap_stale_claims()

    assert result["released"] == 1
    assert result["reaped_sessions"][0]["session_id"] == "swarm-test-subtask_1"
    assert result["reaped_sessions"][0]["reason"] == "inactive_managed_session"
    assert store.list_claims() == []


def test_reap_stale_claims_keeps_live_worktree_sessions(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    worktree_path = tmp_path / "session-live"
    _run(repo, "git", "worktree", "add", "-b", "codex/session-live", str(worktree_path), "HEAD")
    (worktree_path / ".codex_session_active").write_text(
        f"pid={os.getpid()}\nsession_id=session-live\n",
        encoding="utf-8",
    )

    store = FleetCoordinationStore(repo)
    store.claim_paths(session_id="session-live", paths=["aragora/live/**"])
    _backdate_all_claims(store)

    result = store.reap_stale_claims()

    assert result["released"] == 0
    assert result["kept_sessions"][0]["reason"] == "live_session"
    assert store.list_claims()[0]["session_id"] == "session-live"


def test_reap_stale_claims_handles_missing_worktree_paths(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    worktree_path = tmp_path / "session-missing"
    _run(
        repo,
        "git",
        "worktree",
        "add",
        "-b",
        "codex/session-missing",
        str(worktree_path),
        "HEAD",
    )

    store = FleetCoordinationStore(repo)
    store.claim_paths(session_id="session-missing", paths=["aragora/live/**"])
    _backdate_all_claims(store)

    shutil.rmtree(worktree_path)

    result = store.reap_stale_claims()

    assert result["released"] == 1
    assert result["reaped_sessions"][0]["session_id"] == "session-missing"
    assert store.list_claims() == []


def test_count_dirty_handles_worktree_disappearing_before_git_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _make_repo(tmp_path)
    worktree_path = tmp_path / "session-race"
    _run(repo, "git", "worktree", "add", "-b", "codex/session-race", str(worktree_path), "HEAD")

    def _explode(*args, **kwargs):
        raise FileNotFoundError("worktree disappeared")

    monkeypatch.setattr(fleet.subprocess, "run", _explode)

    assert fleet._count_dirty(worktree_path) == 0
    assert fleet._ahead_behind(worktree_path, "main") == (None, None)


def test_reap_stale_claims_keeps_sessions_with_active_leases(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    store = DevCoordinationStore(repo_root=repo)
    store.fleet_store.claim_paths(session_id="lease-backed", paths=["aragora/live/**"])
    _backdate_all_claims(store.fleet_store)
    lease = store.claim_lease(
        task_id="lease-backed-task",
        title="Lease backed lane",
        owner_agent="codex",
        owner_session_id="lease-backed",
        branch="codex/lease-backed",
        worktree_path="/tmp/wt-lease-backed",
        claimed_paths=["aragora/server/auth_checks.py"],
    )

    result = store.fleet_store.reap_stale_claims()

    assert result["released"] == 0
    assert result["kept_sessions"][0]["reason"] == "active_lease"
    assert any(claim["session_id"] == "lease-backed" for claim in store.fleet_store.list_claims())
    store.release_lease(lease.lease_id)


def test_active_lease_session_ids_uses_coordination_db_busy_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _make_repo(tmp_path)
    store = DevCoordinationStore(repo_root=repo)
    lease = store.claim_lease(
        task_id="lease-timeout-task",
        title="Lease timeout lane",
        owner_agent="codex",
        owner_session_id="lease-timeout",
        branch="codex/lease-timeout",
        worktree_path="/tmp/wt-lease-timeout",
        claimed_paths=["aragora/server/auth_checks.py"],
    )

    observed: dict[str, object] = {}
    real_connect = sqlite3.connect

    class _TrackingConnection:
        def __init__(self, conn: sqlite3.Connection) -> None:
            self._conn = conn

        def execute(self, sql: str, params: tuple[object, ...] = ()) -> sqlite3.Cursor:
            observed.setdefault("statements", []).append(sql)
            return self._conn.execute(sql, params)

        def commit(self) -> None:
            self._conn.commit()

        def close(self) -> None:
            self._conn.close()

    def _tracking_connect(*args, **kwargs):
        observed["timeout"] = kwargs.get("timeout")
        return _TrackingConnection(real_connect(*args, **kwargs))

    monkeypatch.setattr(fleet.sqlite3, "connect", _tracking_connect)

    session_ids = fleet._active_lease_session_ids(repo)

    assert "lease-timeout" in session_ids
    assert observed["timeout"] == 60.0
    assert any(
        str(sql).strip() == "PRAGMA busy_timeout=60000" for sql in observed.get("statements", [])
    )
    monkeypatch.undo()
    store.release_lease(lease.lease_id)


def test_claim_paths_reaps_stale_conflicts_before_reporting_conflict(tmp_path: Path) -> None:
    store = FleetCoordinationStore(tmp_path)
    store.claim_paths(session_id="stale-session", paths=["aragora/live/**"], mode="exclusive")
    _backdate_all_claims(store)

    result = store.claim_paths(
        session_id="fresh-session",
        paths=["aragora/live/src/app/page.tsx"],
        mode="exclusive",
    )

    assert result["conflicts"] == []
    assert result["claimed"] == ["aragora/live/src/app/page.tsx"]
    claims = store.list_claims()
    assert len(claims) == 1
    assert claims[0]["session_id"] == "fresh-session"
