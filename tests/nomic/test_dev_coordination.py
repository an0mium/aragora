"""Tests for development coordination primitives."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from aragora.nomic.dev_coordination import (
    CompletionReceipt,
    DevCoordinationStore,
    FileScopeViolationError,
    IntegrationDecisionType,
    LeaseConflictError,
    SalvageStatus,
)
from aragora.nomic.global_work_queue import GlobalWorkQueue, WorkStatus
from aragora.swarm.lane_telemetry import LaneTelemetryCollector


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


def _backdate_fleet_claims(store: DevCoordinationStore, *, hours: int = 2) -> None:
    state = json.loads(store.fleet_store.path.read_text(encoding="utf-8"))
    old_ts = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    for claim in state.get("claims", []):
        claim["claimed_at"] = old_ts
        claim["updated_at"] = old_ts
    store.fleet_store.path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def test_claim_lease_detects_conflicting_scope(store: DevCoordinationStore) -> None:
    lease = store.claim_lease(
        task_id="clb-1",
        title="Spec path hardening",
        owner_agent="codex",
        owner_session_id="sess-a",
        branch="codex/a",
        worktree_path="/tmp/wt-a",
        allowed_globs=["aragora/prompt_engine/**"],
        expected_tests=["python -m pytest tests/prompt_engine -q"],
    )

    assert lease.is_active is True
    assert store.fleet_store.list_claims()[0]["path"] == "aragora/prompt_engine/**"

    with pytest.raises(LeaseConflictError) as exc_info:
        store.claim_lease(
            task_id="clb-2",
            title="Overlapping spec change",
            owner_agent="claude",
            owner_session_id="sess-b",
            branch="codex/b",
            worktree_path="/tmp/wt-b",
            claimed_paths=["aragora/prompt_engine/spec_builder.py"],
        )

    assert exc_info.value.conflicts[0]["lease_id"] == lease.lease_id


def test_claim_lease_detects_existing_fleet_claim(store: DevCoordinationStore) -> None:
    store.fleet_store.claim_paths(
        session_id="external-session",
        paths=["aragora/server/auth_checks.py"],
        branch="codex/external",
    )

    with pytest.raises(LeaseConflictError) as exc_info:
        store.claim_lease(
            task_id="clb-fleet",
            title="Auth checks hardening",
            owner_agent="codex",
            owner_session_id="sess-local",
            branch="codex/local",
            worktree_path="/tmp/wt-local",
            claimed_paths=["aragora/server/auth_checks.py"],
        )

    assert exc_info.value.conflicts[0]["source"] == "fleet_claim"


def test_claim_lease_reaps_stale_fleet_claim_conflicts(store: DevCoordinationStore) -> None:
    store.fleet_store.claim_paths(
        session_id="external-session",
        paths=["aragora/server/auth_checks.py"],
        branch="codex/external",
    )
    _backdate_fleet_claims(store)

    lease = store.claim_lease(
        task_id="clb-fleet-stale",
        title="Auth checks hardening",
        owner_agent="codex",
        owner_session_id="sess-local",
        branch="codex/local",
        worktree_path="/tmp/wt-local",
        claimed_paths=["aragora/server/auth_checks.py"],
    )

    assert lease.is_active is True
    claims = store.fleet_store.list_claims()
    assert len(claims) == 1
    assert claims[0]["session_id"] == "sess-local"


def test_claim_lease_allows_disjoint_scopes(store: DevCoordinationStore) -> None:
    store.claim_lease(
        task_id="clb-1",
        title="Spec path hardening",
        owner_agent="codex",
        owner_session_id="sess-a",
        branch="codex/a",
        worktree_path="/tmp/wt-a",
        allowed_globs=["aragora/prompt_engine/**"],
    )

    second = store.claim_lease(
        task_id="clb-3",
        title="Frontend polish",
        owner_agent="claude",
        owner_session_id="sess-c",
        branch="codex/c",
        worktree_path="/tmp/wt-c",
        allowed_globs=["aragora/live/src/**"],
    )

    assert second.is_active is True
    assert len(store.list_active_leases()) == 2
    assert len(store.fleet_store.list_claims()) == 2


def test_record_completion_creates_pending_integration(store: DevCoordinationStore) -> None:
    lease = store.claim_lease(
        task_id="clb-4",
        title="Receipt gate",
        owner_agent="codex",
        owner_session_id="sess-a",
        branch="codex/a",
        worktree_path="/tmp/wt-a",
        claimed_paths=["aragora/pipeline/backbone_contracts.py"],
        expected_tests=["python -m pytest tests/pipeline/test_backbone_contracts.py -q"],
    )

    receipt = store.record_completion(
        lease_id=lease.lease_id,
        owner_agent="codex",
        owner_session_id="sess-a",
        branch="codex/a",
        worktree_path="/tmp/wt-a",
        commit_shas=["deadbeef"],
        changed_paths=["aragora/pipeline/backbone_contracts.py"],
        tests_run=["python -m pytest tests/pipeline/test_backbone_contracts.py -q"],
        assumptions=["No schema drift"],
        confidence=0.82,
    )

    assert isinstance(receipt, CompletionReceipt)
    assert receipt.artifact_hash
    assert receipt.task_id == "clb-4"
    assert receipt.validations_run == [
        "python -m pytest tests/pipeline/test_backbone_contracts.py -q"
    ]
    assert store.list_active_leases() == []

    pending = store.list_integration_decisions(only_pending=True)
    assert len(pending) == 1
    assert pending[0].receipt_id == receipt.receipt_id
    assert pending[0].decision == IntegrationDecisionType.PENDING_REVIEW.value
    assert store.fleet_store.list_claims() == []
    merge_queue = store.fleet_store.list_merge_queue()
    assert len(merge_queue) == 1
    assert merge_queue[0]["metadata"]["receipt_id"] == receipt.receipt_id
    assert merge_queue[0]["metadata"]["task_id"] == "clb-4"


def test_record_completion_persists_extended_receipt_provenance(
    store: DevCoordinationStore,
) -> None:
    lease = store.claim_lease(
        task_id="wo-extended",
        title="Extended receipt lane",
        owner_agent="codex",
        owner_session_id="sess-extended",
        branch="codex/extended",
        worktree_path="/tmp/wt-extended",
        claimed_paths=["aragora/swarm/supervisor.py"],
        expected_tests=["python -m pytest tests/swarm/test_supervisor.py -q"],
        metadata={
            "supervisor_run_id": "run-123",
            "work_order_id": "wo-extended",
            "task_key": "run-123:wo-extended",
            "reviewer_agent": "claude",
            "risk_level": "high",
        },
    )

    receipt = store.record_completion(
        lease_id=lease.lease_id,
        owner_agent="codex",
        owner_session_id="sess-extended",
        branch="codex/extended",
        worktree_path="/tmp/wt-extended",
        base_sha="abc12345",
        head_sha="def67890",
        commit_shas=["def67890"],
        changed_paths=["aragora/swarm/supervisor.py"],
        tests_run=["python -m pytest tests/swarm/test_supervisor.py -q"],
        validations_run=[
            "python -m pytest tests/swarm/test_supervisor.py -q",
            "ruff check aragora/swarm/supervisor.py",
        ],
        blockers=["needs integrator review"],
        outcome="deliverable_created",
        risks=["merge-risk:review"],
        pr_url="https://github.com/synaptent/aragora/pull/1044",
        pr_number=1044,
        confidence=0.91,
        metadata={"verification_results": [{"command": "ruff check", "passed": True}]},
    )

    stored = store.get_completion_receipt(receipt.receipt_id)

    assert stored is not None
    assert stored.task_id == "wo-extended"
    assert stored.base_sha == "abc12345"
    assert stored.head_sha == "def67890"
    assert stored.outcome == "deliverable_created"
    assert stored.risks == ["merge-risk:review"]
    assert stored.pr_url == "https://github.com/synaptent/aragora/pull/1044"
    assert stored.pr_number == 1044
    assert stored.metadata["task_key"] == "run-123:wo-extended"
    assert stored.metadata["verification_results"][0]["command"] == "ruff check"
    assert store.list_completion_receipts(task_id="wo-extended")[0].receipt_id == receipt.receipt_id


def test_record_completion_rejects_out_of_scope_changes(store: DevCoordinationStore) -> None:
    lease = store.claim_lease(
        task_id="clb-scope",
        title="Scope guarded lane",
        owner_agent="codex",
        owner_session_id="sess-scope",
        branch="codex/scope",
        worktree_path="/tmp/wt-scope",
        claimed_paths=["aragora/server/auth_checks.py"],
    )

    with pytest.raises(FileScopeViolationError) as exc_info:
        store.record_completion(
            lease_id=lease.lease_id,
            owner_agent="codex",
            owner_session_id="sess-scope",
            branch="codex/scope",
            worktree_path="/tmp/wt-scope",
            commit_shas=["abc12345"],
            changed_paths=["aragora/server/handlers/playground.py"],
        )

    violation_types = {item["type"] for item in exc_info.value.violations}
    assert "out_of_scope" in violation_types
    assert "unowned_path" in violation_types
    assert store.list_completion_receipts() == []
    assert store.fleet_store.list_merge_queue() == []

    summary = store.status_summary()
    assert summary["counts"]["scope_violations"] == 1
    assert summary["scope_violations"][0]["owner_session_id"] == "sess-scope"
    assert summary["scope_violations"][0]["violations"][0]["type"] in violation_types


def test_record_completion_rejects_protected_hot_paths(store: DevCoordinationStore) -> None:
    lease = store.claim_lease(
        task_id="clb-hot",
        title="Hot path lane",
        owner_agent="codex",
        owner_session_id="sess-hot",
        branch="codex/hot",
        worktree_path="/tmp/wt-hot",
        allowed_globs=["aragora/server/**"],
        metadata={"hot_paths": ["aragora/server/handlers/**"]},
    )

    with pytest.raises(FileScopeViolationError) as exc_info:
        store.record_completion(
            lease_id=lease.lease_id,
            owner_agent="codex",
            owner_session_id="sess-hot",
            branch="codex/hot",
            worktree_path="/tmp/wt-hot",
            commit_shas=["abc12345"],
            changed_paths=["aragora/server/handlers/playground.py"],
        )

    assert any(item["type"] == "protected_path" for item in exc_info.value.violations)


def test_supervisor_run_tracks_lease_completion_and_decision(store: DevCoordinationStore) -> None:
    run = store.create_supervisor_run(
        goal="Ship bounded swarm lane",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={
            "require_merge_approval": True,
            "require_external_action_approval": True,
        },
        spec={"raw_goal": "Ship bounded swarm lane", "refined_goal": "Ship bounded swarm lane"},
        work_orders=[
            {
                "work_order_id": "wo-1",
                "title": "Implement lane",
                "file_scope": ["aragora/swarm/commander.py"],
                "status": "queued",
                "target_agent": "codex",
                "reviewer_agent": "claude",
            }
        ],
    )

    lease = store.claim_lease(
        task_id="wo-1",
        title="Implement lane",
        owner_agent="codex",
        owner_session_id="sess-swarm",
        branch="codex/swarm-lane",
        worktree_path="/tmp/wt-swarm",
        claimed_paths=["aragora/swarm/commander.py"],
        metadata={"supervisor_run_id": run["run_id"], "work_order_id": "wo-1"},
    )
    refreshed = store.get_supervisor_run(run["run_id"])
    assert refreshed is not None
    assert refreshed["work_orders"][0]["status"] == "leased"
    assert refreshed["work_orders"][0]["lease_id"] == lease.lease_id

    receipt = store.record_completion(
        lease_id=lease.lease_id,
        owner_agent="codex",
        owner_session_id="sess-swarm",
        branch="codex/swarm-lane",
        worktree_path="/tmp/wt-swarm",
        commit_shas=["abc12345"],
        changed_paths=["aragora/swarm/commander.py"],
        tests_run=["python -m pytest tests/swarm/test_commander.py -q"],
        confidence=0.9,
    )
    refreshed = store.get_supervisor_run(run["run_id"])
    assert refreshed is not None
    assert refreshed["work_orders"][0]["status"] == "completed"
    assert refreshed["work_orders"][0]["receipt_id"] == receipt.receipt_id
    assert refreshed["status"] == "completed"

    store.record_integration_decision(
        receipt_id=receipt.receipt_id,
        decision=IntegrationDecisionType.REQUEST_CHANGES,
        decided_by="claude-review",
        rationale="Need heterogeneous review follow-up",
    )
    refreshed = store.get_supervisor_run(run["run_id"])
    assert refreshed is not None
    assert refreshed["work_orders"][0]["status"] == "changes_requested"
    assert refreshed["status"] == "needs_human"


def test_mark_supervisor_run_merged_records_canonical_lane_telemetry(
    store: DevCoordinationStore,
) -> None:
    run = store.create_supervisor_run(
        goal="Ship merged swarm lane",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={
            "require_merge_approval": True,
            "require_external_action_approval": True,
        },
        spec={"raw_goal": "Ship merged swarm lane", "refined_goal": "Ship merged swarm lane"},
        work_orders=[
            {
                "work_order_id": "wo-merge",
                "title": "Integrate lane",
                "file_scope": ["aragora/swarm/reporter.py"],
                "status": "queued",
                "target_agent": "codex",
                "reviewer_agent": "claude",
            }
        ],
    )

    lease = store.claim_lease(
        task_id="wo-merge",
        title="Integrate lane",
        owner_agent="codex",
        owner_session_id="sess-merge",
        branch="codex/merge-lane",
        worktree_path="/tmp/wt-merge",
        claimed_paths=["aragora/swarm/reporter.py"],
        metadata={
            "supervisor_run_id": run["run_id"],
            "work_order_id": "wo-merge",
            "task_key": f"{run['run_id']}:wo-merge",
        },
    )
    receipt = store.record_completion(
        lease_id=lease.lease_id,
        owner_agent="codex",
        owner_session_id="sess-merge",
        branch="codex/merge-lane",
        worktree_path="/tmp/wt-merge",
        commit_shas=["deadbeef"],
        changed_paths=["aragora/swarm/reporter.py"],
        tests_run=["python -m pytest tests/swarm/test_reporter.py -q"],
        confidence=0.93,
    )
    merged_at = (datetime.fromisoformat(receipt.created_at) + timedelta(minutes=5)).isoformat()
    collector = LaneTelemetryCollector(db_path=":memory:")

    with patch("aragora.nomic.dev_coordination._LANE_TELEMETRY", collector):
        store.mark_supervisor_run_merged(
            receipt_id=receipt.receipt_id,
            merge_commit_sha="mergeabc123",
            merged_at=merged_at,
        )

    refreshed = store.get_supervisor_run(run["run_id"])
    assert refreshed is not None
    assert refreshed["work_orders"][0]["status"] == "merged"
    assert refreshed["work_orders"][0]["merge_commit_sha"] == "mergeabc123"
    assert refreshed["work_orders"][0]["merged_at"] == merged_at
    record = collector.get_lane("supervisor_work_order", f"{run['run_id']}:wo-merge")
    assert record is not None
    assert record.terminal_outcome == "deliverable_created"
    assert record.deliverable_type == "branch"
    assert record.receipt_id == receipt.receipt_id
    assert record.merge_ref == "mergeabc123"
    assert record.merged_at == merged_at
    assert record.time_to_merge_seconds == 300.0
    assert record.human_intervention_required is False


def test_list_developer_tasks_flattens_supervisor_runs(store: DevCoordinationStore) -> None:
    run = store.create_supervisor_run(
        goal="Ship canonical queue",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={
            "require_merge_approval": True,
            "require_external_action_approval": True,
        },
        spec={"raw_goal": "Ship canonical queue", "refined_goal": "Ship canonical queue"},
        work_orders=[
            {
                "work_order_id": "wo-queue",
                "title": "Queue lane",
                "file_scope": ["aragora/nomic/dev_coordination.py"],
                "expected_tests": ["python -m pytest tests/nomic/test_dev_coordination.py -q"],
                "success_criteria": {
                    "tests": ["python -m pytest tests/nomic/test_dev_coordination.py -q"]
                },
                "status": "queued",
                "target_agent": "codex",
                "reviewer_agent": "claude",
                "risk_level": "high",
            }
        ],
    )

    task = store.get_developer_task(f"{run['run_id']}:wo-queue")
    assert task is not None
    assert task.status == "queued"
    assert task.priority == 75
    assert task.allowed_paths == ["aragora/nomic/dev_coordination.py"]
    assert "python -m pytest tests/nomic/test_dev_coordination.py -q" in task.acceptance_checks

    lease = store.claim_lease(
        task_id="wo-queue",
        title="Queue lane",
        owner_agent="codex",
        owner_session_id="sess-queue",
        branch="codex/queue",
        worktree_path="/tmp/wt-queue",
        claimed_paths=["aragora/nomic/dev_coordination.py"],
        metadata={"supervisor_run_id": run["run_id"], "work_order_id": "wo-queue"},
    )

    refreshed = store.get_developer_task(f"{run['run_id']}:wo-queue")
    assert refreshed is not None
    assert refreshed.status == "leased"
    assert refreshed.lease_id == lease.lease_id
    assert refreshed.owner_session_id == "sess-queue"


def test_record_integration_decision_updates_queue(store: DevCoordinationStore) -> None:
    lease = store.claim_lease(
        task_id="clb-5",
        title="Merge lane",
        owner_agent="codex",
        owner_session_id="sess-a",
        branch="codex/a",
        worktree_path="/tmp/wt-a",
        claimed_paths=["aragora/server/handlers/playground.py"],
    )
    receipt = store.record_completion(
        lease_id=lease.lease_id,
        owner_agent="codex",
        owner_session_id="sess-a",
        branch="codex/a",
        worktree_path="/tmp/wt-a",
        commit_shas=["abc12345"],
    )

    decision = store.record_integration_decision(
        receipt_id=receipt.receipt_id,
        decision=IntegrationDecisionType.CHERRY_PICK,
        decided_by="integrator",
        rationale="Keep only the isolated handler fix",
        chosen_commits=["abc12345"],
        followups=["run share-flow tests"],
    )

    assert decision.decision == IntegrationDecisionType.CHERRY_PICK.value
    assert len(store.list_integration_decisions(receipt_id=receipt.receipt_id)) == 2
    assert (
        store.pending_work_items()[0].metadata["decision"]
        == IntegrationDecisionType.PENDING_REVIEW.value
    )
    merge_queue = store.fleet_store.list_merge_queue()
    assert merge_queue[0]["status"] == "integrating"
    assert merge_queue[0]["metadata"]["integration_decision"] == "cherry_pick"


def test_scope_violation_is_terminal_for_supervisor_run_status() -> None:
    assert (
        DevCoordinationStore._derive_supervisor_run_status([{"status": "scope_violation"}])
        == "completed"
    )
    assert (
        DevCoordinationStore._derive_supervisor_run_status(
            [{"status": "scope_violation"}, {"status": "dispatched"}]
        )
        == "active"
    )
    assert (
        DevCoordinationStore._derive_supervisor_run_status([{"status": "waiting_conflict"}])
        == "needs_human"
    )


def test_heartbeat_lease_refreshes_expiry_and_fleet_claim(store: DevCoordinationStore) -> None:
    lease = store.claim_lease(
        task_id="clb-heartbeat",
        title="Heartbeat path",
        owner_agent="codex",
        owner_session_id="sess-heartbeat",
        branch="codex/heartbeat",
        worktree_path="/tmp/wt-heartbeat",
        claimed_paths=["aragora/server/auth_checks.py"],
        ttl_hours=0.01,
    )

    original_expiry = lease.expires_at
    refreshed = store.heartbeat_lease(lease.lease_id, ttl_hours=2.0)

    assert refreshed.expires_at > original_expiry
    claims = store.fleet_store.list_claims()
    assert len(claims) == 1
    assert claims[0]["session_id"] == "sess-heartbeat"


def test_reap_expired_leases_releases_fleet_claims(store: DevCoordinationStore) -> None:
    lease = store.claim_lease(
        task_id="clb-expired",
        title="Expired path",
        owner_agent="codex",
        owner_session_id="sess-expired",
        branch="codex/expired",
        worktree_path="/tmp/wt-expired",
        claimed_paths=["aragora/server/auth_checks.py"],
        ttl_hours=2.0,
    )
    assert store.fleet_store.list_claims()

    conn = store._connect()
    try:
        conn.execute(
            "UPDATE leases SET expires_at = ?, updated_at = ? WHERE lease_id = ?",
            (
                "2000-01-01T00:00:00+00:00",
                "2000-01-01T00:00:00+00:00",
                lease.lease_id,
            ),
        )
        conn.commit()
    finally:
        conn.close()

    expired = store.reap_expired_leases()

    assert [item.lease_id for item in expired] == [lease.lease_id]
    assert store.list_active_leases() == []
    assert store.fleet_store.list_claims() == []


def test_status_summary_does_not_reap_expired_leases(store: DevCoordinationStore) -> None:
    lease = store.claim_lease(
        task_id="clb-status-read",
        title="Status read should be side-effect free",
        owner_agent="codex",
        owner_session_id="sess-status",
        branch="codex/status",
        worktree_path="/tmp/wt-status",
        claimed_paths=["aragora/server/auth_checks.py"],
        ttl_hours=2.0,
    )

    conn = store._connect()
    try:
        conn.execute(
            "UPDATE leases SET expires_at = ?, updated_at = ? WHERE lease_id = ?",
            (
                "2000-01-01T00:00:00+00:00",
                "2000-01-01T00:00:00+00:00",
                lease.lease_id,
            ),
        )
        conn.commit()
    finally:
        conn.close()

    summary = store.status_summary()

    assert summary["counts"]["active_leases"] == 0
    assert summary["counts"]["fleet_claims"] == 1
    claims = store.fleet_store.list_claims()
    assert len(claims) == 1
    assert claims[0]["session_id"] == "sess-status"


def test_claim_lease_reaps_expired_claims_before_conflict_check(
    store: DevCoordinationStore,
) -> None:
    lease = store.claim_lease(
        task_id="clb-reclaim",
        title="Original lease",
        owner_agent="codex",
        owner_session_id="sess-old",
        branch="codex/old",
        worktree_path="/tmp/wt-old",
        claimed_paths=["aragora/server/auth_checks.py"],
        ttl_hours=2.0,
    )

    conn = store._connect()
    try:
        conn.execute(
            "UPDATE leases SET expires_at = ?, updated_at = ? WHERE lease_id = ?",
            (
                "2000-01-01T00:00:00+00:00",
                "2000-01-01T00:00:00+00:00",
                lease.lease_id,
            ),
        )
        conn.commit()
    finally:
        conn.close()

    replacement = store.claim_lease(
        task_id="clb-reclaim-2",
        title="Replacement lease",
        owner_agent="claude",
        owner_session_id="sess-new",
        branch="codex/new",
        worktree_path="/tmp/wt-new",
        claimed_paths=["aragora/server/auth_checks.py"],
    )

    claims = store.fleet_store.list_claims()
    assert replacement.is_active is True
    assert len(claims) == 1
    assert claims[0]["session_id"] == "sess-new"


def test_scan_salvage_sources_finds_worktree_and_stash(
    repo: Path, store: DevCoordinationStore
) -> None:
    worktree_path = repo.parent / "dirty-wt"
    _run(repo, "git", "worktree", "add", "-b", "codex/dirty", str(worktree_path), "main")
    (worktree_path / "dirty.txt").write_text("dirty\n", encoding="utf-8")

    (repo / "stashed.txt").write_text("stash me\n", encoding="utf-8")
    _run(repo, "git", "add", "stashed.txt")
    _run(repo, "git", "stash", "push", "-u", "-m", "useful stash")

    candidates = store.scan_salvage_sources()
    by_kind = {item.source_kind: item for item in candidates}

    assert "worktree" in by_kind
    assert by_kind["worktree"].source_ref == "codex/dirty"
    assert by_kind["worktree"].status == SalvageStatus.DETECTED.value
    assert "stash" in by_kind
    assert by_kind["stash"].source_ref.startswith("stash@{")

    work_items = store.pending_work_items()
    assert any(item.metadata.get("source_kind") == "worktree" for item in work_items)
    assert any(item.metadata.get("source_kind") == "stash" for item in work_items)


@pytest.mark.asyncio
async def test_sync_pending_work_queue_projects_items(
    repo: Path, store: DevCoordinationStore
) -> None:
    lease = store.claim_lease(
        task_id="clb-sync",
        title="Queue sync lane",
        owner_agent="codex",
        owner_session_id="sess-sync",
        branch="codex/sync",
        worktree_path="/tmp/wt-sync",
        claimed_paths=["aragora/server/auth_checks.py"],
    )
    receipt = store.record_completion(
        lease_id=lease.lease_id,
        owner_agent="codex",
        owner_session_id="sess-sync",
        branch="codex/sync",
        worktree_path="/tmp/wt-sync",
        commit_shas=["abc12345"],
    )
    salvage = store.upsert_salvage_candidate(
        source_kind="stash",
        source_ref="stash@{0}",
        stash_ref="stash@{0}",
        changed_paths=["aragora/server/auth_checks.py"],
        summary="useful stash",
        likely_value=0.75,
    )
    queue = GlobalWorkQueue(storage_dir=repo / ".work_queue")

    counts = await store.sync_pending_work_queue(queue)

    assert counts["created"] == 2
    items = await queue.list_items(limit=10)
    item_ids = {item.id for item in items}
    assert f"salvage:{salvage.candidate_id}" in item_ids
    assert any(item_id.startswith("integration:") for item_id in item_ids)
    integration_item = next(item for item in items if item.id.startswith("integration:"))
    assert integration_item.metadata["receipt_id"] == receipt.receipt_id


@pytest.mark.asyncio
async def test_sync_developer_task_queue_projects_open_tasks(
    repo: Path, store: DevCoordinationStore
) -> None:
    run = store.create_supervisor_run(
        goal="Project queued developer task",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={
            "require_merge_approval": True,
            "require_external_action_approval": True,
        },
        spec={
            "raw_goal": "Project queued developer task",
            "refined_goal": "Project queued developer task",
        },
        work_orders=[
            {
                "work_order_id": "wo-sync-task",
                "title": "Queue me",
                "file_scope": ["aragora/nomic/dev_coordination.py"],
                "status": "queued",
                "target_agent": "codex",
                "reviewer_agent": "claude",
            }
        ],
    )
    queue = GlobalWorkQueue(storage_dir=repo / ".work_queue")

    counts = await store.sync_developer_task_queue(queue)
    item = await queue.get(f"task:{run['run_id']}:wo-sync-task")

    assert counts["created"] == 1
    assert item is not None
    assert item.status == WorkStatus.READY
    assert item.metadata["task_id"] == "wo-sync-task"


@pytest.mark.asyncio
async def test_sync_developer_task_queue_completes_resolved_tasks(
    repo: Path, store: DevCoordinationStore
) -> None:
    run = store.create_supervisor_run(
        goal="Complete task queue item",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={
            "require_merge_approval": True,
            "require_external_action_approval": True,
        },
        spec={"raw_goal": "Complete task queue item", "refined_goal": "Complete task queue item"},
        work_orders=[
            {
                "work_order_id": "wo-close-task",
                "title": "Close me",
                "file_scope": ["aragora/nomic/dev_coordination.py"],
                "status": "queued",
                "target_agent": "codex",
                "reviewer_agent": "claude",
            }
        ],
    )
    queue = GlobalWorkQueue(storage_dir=repo / ".work_queue")

    await store.sync_developer_task_queue(queue)
    run_record = store.get_supervisor_run(run["run_id"])
    assert run_record is not None
    run_record["work_orders"][0]["status"] = "merged"
    store.update_supervisor_run(run["run_id"], work_orders=run_record["work_orders"])

    counts = await store.sync_developer_task_queue(queue)
    item = await queue.get(f"task:{run['run_id']}:wo-close-task")

    assert counts["completed"] == 1
    assert item is not None
    assert item.status == WorkStatus.COMPLETED


@pytest.mark.asyncio
async def test_sync_pending_work_queue_completes_resolved_items(
    repo: Path, store: DevCoordinationStore
) -> None:
    candidate = store.upsert_salvage_candidate(
        source_kind="stash",
        source_ref="stash@{0}",
        stash_ref="stash@{0}",
        changed_paths=["aragora/server/auth_checks.py"],
        summary="useful stash",
        likely_value=0.75,
    )
    queue = GlobalWorkQueue(storage_dir=repo / ".work_queue")

    await store.sync_pending_work_queue(queue)
    store.upsert_salvage_candidate(
        source_kind="stash",
        source_ref="stash@{0}",
        stash_ref="stash@{0}",
        changed_paths=["aragora/server/auth_checks.py"],
        summary="useful stash",
        likely_value=0.75,
        status=SalvageStatus.DISCARDED,
    )

    counts = await store.sync_pending_work_queue(queue)
    work = await queue.get(f"salvage:{candidate.candidate_id}")

    assert counts["completed"] == 1
    assert work is not None
    assert work.status == WorkStatus.COMPLETED
    assert work.metadata["result"]["reason"] == "no_longer_pending"


@pytest.mark.asyncio
async def test_sync_pending_work_queue_reopens_terminal_items(
    repo: Path, store: DevCoordinationStore
) -> None:
    candidate = store.upsert_salvage_candidate(
        source_kind="stash",
        source_ref="stash@{0}",
        stash_ref="stash@{0}",
        changed_paths=["aragora/server/auth_checks.py"],
        summary="useful stash",
        likely_value=0.75,
    )
    queue = GlobalWorkQueue(storage_dir=repo / ".work_queue")

    await store.sync_pending_work_queue(queue)
    await queue.complete(
        f"salvage:{candidate.candidate_id}",
        result={"source": "test", "reason": "closed early"},
    )

    counts = await store.sync_pending_work_queue(queue)
    work = await queue.get(f"salvage:{candidate.candidate_id}")

    assert counts["reopened"] == 1
    assert work is not None
    assert work.status in (WorkStatus.PENDING, WorkStatus.READY)


def test_release_lease_releases_fleet_claims(store: DevCoordinationStore) -> None:
    lease = store.claim_lease(
        task_id="clb-release",
        title="Release path",
        owner_agent="codex",
        owner_session_id="sess-release",
        branch="codex/release",
        worktree_path="/tmp/wt-release",
        claimed_paths=["aragora/server/handlers/playground.py"],
    )

    assert store.fleet_store.list_claims()

    released = store.release_lease(lease.lease_id)

    assert released.status == "released"
    assert store.fleet_store.list_claims() == []
