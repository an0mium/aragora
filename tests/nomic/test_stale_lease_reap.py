"""Regression tests for stale lease auto-release (V8 blocker).

Covers the gap where a worker session dies but its worktree still exists,
leaving the lease active and blocking new work.
"""

from __future__ import annotations

import os
import subprocess
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from aragora.nomic.dev_coordination import (
    DevCoordinationStore,
    LeaseStatus,
    _safe_kill_probe,
    _utcnow,
)


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(
        ["git", "init", "-b", "main"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=True,
    )
    (repo / "README.md").write_text("hello\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "README.md"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "remote", "add", "origin", str(repo)],
        cwd=repo,
        text=True,
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "update-ref", "refs/remotes/origin/main", "HEAD"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=True,
    )
    return repo


@pytest.fixture()
def store(repo: Path) -> DevCoordinationStore:
    return DevCoordinationStore(repo_root=repo)


# ---------------------------------------------------------------------------
# _safe_kill_probe unit tests
# ---------------------------------------------------------------------------


def test_safe_kill_probe_alive_process() -> None:
    """Current process should be reported alive."""
    assert _safe_kill_probe(os.getpid()) is None


def test_safe_kill_probe_dead_pid() -> None:
    """A PID that does not exist should return ProcessLookupError."""
    exc = _safe_kill_probe(999999999)
    assert isinstance(exc, ProcessLookupError)


def test_safe_kill_probe_invalid_value() -> None:
    """Non-integer values should return an exception, not crash."""
    exc = _safe_kill_probe("not-a-pid")
    assert exc is not None


# ---------------------------------------------------------------------------
# reap_stale_leases — worker_pid_dead path
# ---------------------------------------------------------------------------


def test_reap_stale_leases_releases_dead_pid(store: DevCoordinationStore) -> None:
    """Lease with a dead worker_pid should be reaped."""
    lease = store.claim_lease(
        task_id="task-1",
        title="Dead worker task",
        owner_agent="codex",
        owner_session_id="sess-dead",
        branch="codex/dead",
        worktree_path="/tmp/wt-dead",
        claimed_paths=["aragora/swarm/supervisor.py"],
        metadata={"worker_pid": 999999999},
    )
    assert lease.lease_id in {l.lease_id for l in store.list_active_leases()}

    reaped = store.reap_stale_leases()
    assert len(reaped) == 1
    assert reaped[0].lease_id == lease.lease_id

    # Lease should no longer be active.
    assert lease.lease_id not in {l.lease_id for l in store.list_active_leases()}


def test_reap_stale_leases_keeps_live_pid(store: DevCoordinationStore) -> None:
    """Lease whose worker_pid is still running should NOT be reaped."""
    lease = store.claim_lease(
        task_id="task-alive",
        title="Alive worker task",
        owner_agent="codex",
        owner_session_id="sess-alive",
        branch="codex/alive",
        worktree_path="/tmp/wt-alive",
        claimed_paths=["aragora/swarm/campaign.py"],
        metadata={"worker_pid": os.getpid()},
    )

    reaped = store.reap_stale_leases()
    assert len(reaped) == 0
    assert lease.lease_id in {l.lease_id for l in store.list_active_leases()}


# ---------------------------------------------------------------------------
# reap_stale_leases — heartbeat_timeout path
# ---------------------------------------------------------------------------


def test_reap_stale_leases_heartbeat_timeout(store: DevCoordinationStore) -> None:
    """Lease without worker_pid and stale updated_at should be reaped."""
    lease = store.claim_lease(
        task_id="task-stale",
        title="Stale heartbeat task",
        owner_agent="codex",
        owner_session_id="sess-stale",
        branch="codex/stale",
        worktree_path="/tmp/wt-stale",
        claimed_paths=["docs/guides/OPERATOR.md"],
    )

    # Backdate updated_at to 2 hours ago.
    two_hours_ago = (_utcnow() - timedelta(hours=2)).isoformat()
    conn = store._connect()
    try:
        conn.execute(
            "UPDATE leases SET updated_at = ? WHERE lease_id = ?",
            (two_hours_ago, lease.lease_id),
        )
        conn.commit()
    finally:
        conn.close()

    reaped = store.reap_stale_leases()
    assert len(reaped) == 1
    assert reaped[0].lease_id == lease.lease_id


def test_reap_stale_leases_fresh_heartbeat_not_reaped(
    store: DevCoordinationStore,
) -> None:
    """Lease without worker_pid but recently heartbeated should NOT be reaped."""
    lease = store.claim_lease(
        task_id="task-fresh",
        title="Fresh heartbeat task",
        owner_agent="codex",
        owner_session_id="sess-fresh",
        branch="codex/fresh",
        worktree_path="/tmp/wt-fresh",
        claimed_paths=["aragora/billing/cost_tracker.py"],
    )

    reaped = store.reap_stale_leases()
    assert len(reaped) == 0
    assert lease.lease_id in {l.lease_id for l in store.list_active_leases()}


# ---------------------------------------------------------------------------
# reap_stale_leases — already-expired leases not double-counted
# ---------------------------------------------------------------------------


def test_reap_stale_leases_skips_already_expired(store: DevCoordinationStore) -> None:
    """Leases past TTL should be left to reap_expired_leases, not double-counted."""
    # Create a lease directly via SQL to avoid claim_lease's auto-reap side effects.
    import time

    lease = store.claim_lease(
        task_id="task-expired",
        title="Expired task",
        owner_agent="codex",
        owner_session_id="sess-exp",
        branch="codex/exp",
        worktree_path="/tmp/wt-exp",
        claimed_paths=["aragora/events/dispatcher.py"],
        ttl_hours=0.0001,  # Expires almost immediately.
        metadata={"worker_pid": 999999999},
    )

    # Wait briefly for TTL to pass.
    time.sleep(0.5)

    # The expired-reaper should handle this lease.
    expired = store.reap_expired_leases()
    assert len(expired) >= 1

    # Stale reaper should not try to reap an already-expired lease.
    stale = store.reap_stale_leases()
    assert lease.lease_id not in {l.lease_id for l in stale}


# ---------------------------------------------------------------------------
# update_lease_metadata
# ---------------------------------------------------------------------------


def test_update_lease_metadata_persists_pid(store: DevCoordinationStore) -> None:
    """update_lease_metadata should merge keys into existing metadata."""
    lease = store.claim_lease(
        task_id="task-meta",
        title="Metadata task",
        owner_agent="codex",
        owner_session_id="sess-meta",
        branch="codex/meta",
        worktree_path="/tmp/wt-meta",
        claimed_paths=["aragora/nomic/dev_coordination.py"],
        metadata={"supervisor_run_id": "run-123"},
    )

    store.update_lease_metadata(lease.lease_id, {"worker_pid": 42})

    # Re-read the lease and check metadata.
    active = store.list_active_leases()
    found = [l for l in active if l.lease_id == lease.lease_id]
    assert len(found) == 1
    assert found[0].metadata["worker_pid"] == 42
    assert found[0].metadata["supervisor_run_id"] == "run-123"


# ---------------------------------------------------------------------------
# claim_lease triggers stale reaping
# ---------------------------------------------------------------------------


def test_claim_lease_reaps_stale_before_conflict_check(
    store: DevCoordinationStore,
) -> None:
    """New claim_lease should auto-release stale leases before checking conflicts."""
    stale = store.claim_lease(
        task_id="task-blocking",
        title="Blocking dead task",
        owner_agent="codex",
        owner_session_id="sess-blocking",
        branch="codex/blocking",
        worktree_path="/tmp/wt-blocking",
        claimed_paths=["aragora/swarm/supervisor.py"],
        metadata={"worker_pid": 999999999},
    )
    assert stale.lease_id in {l.lease_id for l in store.list_active_leases()}

    # Claiming the same scope should succeed because the stale lease is auto-reaped.
    new_lease = store.claim_lease(
        task_id="task-replacement",
        title="Replacement task",
        owner_agent="codex",
        owner_session_id="sess-replacement",
        branch="codex/replacement",
        worktree_path="/tmp/wt-replacement",
        claimed_paths=["aragora/swarm/supervisor.py"],
    )
    assert new_lease.lease_id
    assert stale.lease_id not in {l.lease_id for l in store.list_active_leases()}
