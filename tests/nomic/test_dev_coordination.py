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
    LeaseStatus,
    SalvageStatus,
    _targeted_replay_expected_tests_for_work_order,
    _verification_timeout_for_command,
    _verification_result_looks_environment_blocked,
)
from aragora.nomic.global_work_queue import GlobalWorkQueue, WorkStatus
from aragora.swarm.lane_telemetry import LaneTelemetryCollector
from aragora.swarm.worker_launcher import WorkerLauncher


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


def test_connect_sets_busy_timeout(store: DevCoordinationStore) -> None:
    conn = store._connect()
    try:
        timeout_ms = conn.execute("PRAGMA busy_timeout").fetchone()[0]
    finally:
        conn.close()
    assert timeout_ms == 60_000


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
    assert stored.metadata["pr_created_at"] == stored.created_at
    assert stored.metadata["task_key"] == "run-123:wo-extended"
    assert stored.metadata["verification_results"][0]["command"] == "ruff check"
    assert store.list_completion_receipts(task_id="wo-extended")[0].receipt_id == receipt.receipt_id
    merge_queue = store.fleet_store.list_merge_queue()
    assert len(merge_queue) == 1
    assert merge_queue[0]["metadata"]["pr_created_at"] == stored.created_at


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


def test_record_completion_can_skip_live_session_ownership_for_historical_backfill(
    store: DevCoordinationStore,
) -> None:
    lease = store.claim_lease(
        task_id="clb-backfill",
        title="Historical backfill lane",
        owner_agent="codex",
        owner_session_id="sess-backfill",
        branch="codex/backfill",
        worktree_path="/tmp/wt-backfill",
        claimed_paths=["aragora/server/auth_checks.py"],
    )
    store.fleet_store.release_paths(session_id="sess-backfill")

    with pytest.raises(FileScopeViolationError) as exc_info:
        store.record_completion(
            lease_id=lease.lease_id,
            owner_agent="codex",
            owner_session_id="sess-backfill",
            branch="codex/backfill",
            worktree_path="/tmp/wt-backfill",
            commit_shas=["abc12345"],
            changed_paths=["aragora/server/auth_checks.py"],
        )

    assert any(item["type"] == "unowned_path" for item in exc_info.value.violations)

    receipt = store.record_completion(
        lease_id=lease.lease_id,
        owner_agent="codex",
        owner_session_id="sess-backfill",
        branch="codex/backfill",
        worktree_path="/tmp/wt-backfill",
        commit_shas=["abc12345"],
        changed_paths=["aragora/server/auth_checks.py"],
        metadata={"backfilled_receipt": True},
        require_session_ownership=False,
    )

    assert receipt.receipt_id
    assert receipt.metadata["backfilled_receipt"] is True
    assert receipt.outcome == "deliverable_created"


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
        pr_url="https://github.com/synaptent/aragora/pull/9999",
        pr_number=9999,
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
    assert record.deliverable_type == "pr"
    assert record.receipt_id == receipt.receipt_id
    assert record.pr_url == "https://github.com/synaptent/aragora/pull/9999"
    assert record.pr_number == 9999
    assert record.merge_ref == "mergeabc123"
    assert record.merged_at == merged_at
    assert record.time_to_pr_seconds == 0.0
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


def test_reap_expired_leases_preserves_receipt_backed_work_order(
    store: DevCoordinationStore,
) -> None:
    run = store.create_supervisor_run(
        goal="Preserve completed lane on expiry reap",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={
            "require_merge_approval": True,
            "require_external_action_approval": True,
        },
        spec={
            "raw_goal": "Preserve completed lane on expiry reap",
            "refined_goal": "Preserve completed lane on expiry reap",
        },
        work_orders=[
            {
                "work_order_id": "wo-expired-reap",
                "title": "Deliverable lane",
                "file_scope": ["aragora/swarm/reporter.py"],
                "status": "queued",
                "target_agent": "codex",
                "reviewer_agent": "claude",
            }
        ],
    )

    lease = store.claim_lease(
        task_id="wo-expired-reap",
        title="Deliverable lane",
        owner_agent="codex",
        owner_session_id="sess-expired-reap",
        branch="codex/expired-reap",
        worktree_path="/tmp/wt-expired-reap",
        claimed_paths=["aragora/swarm/reporter.py"],
        metadata={"supervisor_run_id": run["run_id"], "work_order_id": "wo-expired-reap"},
    )
    receipt = store.record_completion(
        lease_id=lease.lease_id,
        owner_agent="codex",
        owner_session_id="sess-expired-reap",
        branch="codex/expired-reap",
        worktree_path="/tmp/wt-expired-reap",
        commit_shas=["abc12345"],
        changed_paths=["aragora/swarm/reporter.py"],
    )

    conn = store._connect()
    try:
        conn.execute(
            "UPDATE leases SET status = ?, expires_at = ?, updated_at = ? WHERE lease_id = ?",
            (
                "active",
                "2000-01-01T00:00:00+00:00",
                "2000-01-01T00:00:00+00:00",
                lease.lease_id,
            ),
        )
        conn.commit()
    finally:
        conn.close()

    expired = store.reap_expired_leases()
    refreshed = store.get_supervisor_run(run["run_id"])

    assert [item.lease_id for item in expired] == [lease.lease_id]
    assert refreshed is not None
    assert refreshed["work_orders"][0]["status"] == "completed"
    assert refreshed["work_orders"][0]["receipt_id"] == receipt.receipt_id
    assert "failure_reason" not in refreshed["work_orders"][0]


def test_reap_stale_leases_preserves_receipt_backed_work_order(store: DevCoordinationStore) -> None:
    run = store.create_supervisor_run(
        goal="Preserve completed lane on stale reap",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={
            "require_merge_approval": True,
            "require_external_action_approval": True,
        },
        spec={
            "raw_goal": "Preserve completed lane on stale reap",
            "refined_goal": "Preserve completed lane on stale reap",
        },
        work_orders=[
            {
                "work_order_id": "wo-stale-reap",
                "title": "Deliverable lane",
                "file_scope": ["aragora/swarm/reporter.py"],
                "status": "queued",
                "target_agent": "codex",
                "reviewer_agent": "claude",
            }
        ],
    )

    lease = store.claim_lease(
        task_id="wo-stale-reap",
        title="Deliverable lane",
        owner_agent="codex",
        owner_session_id="sess-stale-reap",
        branch="codex/stale-reap",
        worktree_path="/tmp/wt-stale-reap",
        claimed_paths=["aragora/swarm/reporter.py"],
        metadata={"supervisor_run_id": run["run_id"], "work_order_id": "wo-stale-reap"},
    )
    receipt = store.record_completion(
        lease_id=lease.lease_id,
        owner_agent="codex",
        owner_session_id="sess-stale-reap",
        branch="codex/stale-reap",
        worktree_path="/tmp/wt-stale-reap",
        commit_shas=["abc12345"],
        changed_paths=["aragora/swarm/reporter.py"],
    )

    conn = store._connect()
    try:
        conn.execute(
            "UPDATE leases SET status = ?, expires_at = ?, updated_at = ? WHERE lease_id = ?",
            (
                "active",
                "2999-01-01T00:00:00+00:00",
                "2000-01-01T00:00:00+00:00",
                lease.lease_id,
            ),
        )
        conn.commit()
    finally:
        conn.close()

    stale = store.reap_stale_leases(stale_threshold_seconds=1.0)
    refreshed = store.get_supervisor_run(run["run_id"])

    assert [item.lease_id for item in stale] == [lease.lease_id]
    assert refreshed is not None
    assert refreshed["work_orders"][0]["status"] == "completed"
    assert refreshed["work_orders"][0]["receipt_id"] == receipt.receipt_id
    assert "failure_reason" not in refreshed["work_orders"][0]


def test_archive_reaped_no_receipt_work_orders_discards_old_backlog(
    store: DevCoordinationStore,
) -> None:
    run = store.create_supervisor_run(
        goal="Archive stale reaped backlog",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={
            "require_merge_approval": True,
            "require_external_action_approval": True,
        },
        spec={
            "raw_goal": "Archive stale reaped backlog",
            "refined_goal": "Archive stale reaped backlog",
        },
        work_orders=[
            {
                "work_order_id": "wo-archive-stale",
                "title": "Old stale lane",
                "file_scope": ["aragora/swarm/reporter.py"],
                "status": "needs_human",
                "failure_reason": "stale_lease_reaped",
                "target_agent": "codex",
                "reviewer_agent": "claude",
            }
        ],
    )

    conn = store._connect()
    try:
        conn.execute(
            "UPDATE supervisor_runs SET created_at = ?, updated_at = ? WHERE run_id = ?",
            ("2000-01-01T00:00:00+00:00", "2000-01-01T00:00:00+00:00", run["run_id"]),
        )
        conn.commit()
    finally:
        conn.close()

    archived = store.archive_reaped_no_receipt_work_orders(grace_period_hours=6.0)
    refreshed = store.get_supervisor_run(run["run_id"])

    assert archived == 1
    assert refreshed is not None
    work_order = refreshed["work_orders"][0]
    assert work_order["status"] == "discarded"
    assert work_order["failure_reason"] == "stale_lease_reaped"
    assert work_order["metadata"]["archived_due_to"] == "reaped_no_receipt"
    assert work_order["metadata"]["archive_reason"] == "stale_lease_reaped"
    assert work_order["metadata"]["previous_status"] == "needs_human"
    assert store.get_developer_task(f"{run['run_id']}:wo-archive-stale") is not None
    assert store.list_developer_tasks(open_only=True) == []


def test_archive_reaped_no_receipt_work_orders_preserves_active_or_receipt_backed_lanes(
    store: DevCoordinationStore,
) -> None:
    stale_run = store.create_supervisor_run(
        goal="Keep active stale lane open",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={
            "require_merge_approval": True,
            "require_external_action_approval": True,
        },
        spec={
            "raw_goal": "Keep active stale lane open",
            "refined_goal": "Keep active stale lane open",
        },
        work_orders=[
            {
                "work_order_id": "wo-active-stale",
                "title": "Active stale lane",
                "file_scope": ["aragora/swarm/reporter.py"],
                "status": "needs_human",
                "failure_reason": "stale_lease_reaped",
                "target_agent": "codex",
                "reviewer_agent": "claude",
                "lease_id": "lease-active-stale",
            }
        ],
    )
    receipt_run = store.create_supervisor_run(
        goal="Keep receipt lane reviewable",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={
            "require_merge_approval": True,
            "require_external_action_approval": True,
        },
        spec={
            "raw_goal": "Keep receipt lane reviewable",
            "refined_goal": "Keep receipt lane reviewable",
        },
        work_orders=[
            {
                "work_order_id": "wo-receipt-stale",
                "title": "Receipt stale lane",
                "file_scope": ["aragora/swarm/reporter.py"],
                "status": "needs_human",
                "failure_reason": "stale_lease_reaped",
                "target_agent": "codex",
                "reviewer_agent": "claude",
                "receipt_id": "rcpt-stale-keep",
            }
        ],
    )

    conn = store._connect()
    try:
        conn.execute(
            """
            INSERT INTO leases VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "lease-active-stale",
                "wo-active-stale",
                "Active stale lane",
                "codex",
                "sess-active-stale",
                "codex/active-stale",
                "/tmp/wt-active-stale",
                "[]",
                '["aragora/swarm/reporter.py"]',
                "[]",
                LeaseStatus.ACTIVE.value,
                "2000-01-01T00:00:00+00:00",
                "2000-01-01T00:00:00+00:00",
                "2999-01-01T00:00:00+00:00",
                "{}",
            ),
        )
        conn.execute(
            "UPDATE supervisor_runs SET updated_at = ? WHERE run_id IN (?, ?)",
            (
                "2000-01-01T00:00:00+00:00",
                stale_run["run_id"],
                receipt_run["run_id"],
            ),
        )
        conn.commit()
    finally:
        conn.close()

    archived = store.archive_reaped_no_receipt_work_orders(grace_period_hours=6.0)
    stale_refreshed = store.get_supervisor_run(stale_run["run_id"])
    receipt_refreshed = store.get_supervisor_run(receipt_run["run_id"])

    assert archived == 0
    assert stale_refreshed is not None
    assert stale_refreshed["work_orders"][0]["status"] == "needs_human"
    assert receipt_refreshed is not None
    assert receipt_refreshed["work_orders"][0]["status"] == "needs_human"


def test_archive_reaped_no_receipt_work_orders_discards_implicit_expired_inflight_lane(
    store: DevCoordinationStore,
) -> None:
    run = store.create_supervisor_run(
        goal="Archive implicit expired inflight lane",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={
            "require_merge_approval": True,
            "require_external_action_approval": True,
        },
        spec={
            "raw_goal": "Archive implicit expired inflight lane",
            "refined_goal": "Archive implicit expired inflight lane",
        },
        work_orders=[
            {
                "work_order_id": "wo-implicit-expired",
                "title": "Implicit expired lane",
                "file_scope": ["aragora/swarm/reporter.py"],
                "status": "dispatched",
                "completed_at": "None",
                "lease_id": "lease-implicit-expired",
                "target_agent": "codex",
                "reviewer_agent": "claude",
            }
        ],
    )

    conn = store._connect()
    try:
        conn.execute(
            """
            INSERT INTO leases VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "lease-implicit-expired",
                "wo-implicit-expired",
                "Implicit expired lane",
                "codex",
                "sess-implicit-expired",
                "codex/implicit-expired",
                "/tmp/wt-implicit-expired",
                "[]",
                '["aragora/swarm/reporter.py"]',
                "[]",
                LeaseStatus.EXPIRED.value,
                "2000-01-01T00:00:00+00:00",
                "2000-01-01T00:00:00+00:00",
                "2000-01-01T01:00:00+00:00",
                "{}",
            ),
        )
        conn.execute(
            "UPDATE supervisor_runs SET updated_at = ? WHERE run_id = ?",
            ("2000-01-01T00:00:00+00:00", run["run_id"]),
        )
        conn.commit()
    finally:
        conn.close()

    archived = store.archive_reaped_no_receipt_work_orders(grace_period_hours=6.0)
    refreshed = store.get_supervisor_run(run["run_id"])

    assert archived == 1
    assert refreshed is not None
    work_order = refreshed["work_orders"][0]
    assert work_order["status"] == "discarded"
    assert work_order["failure_reason"] == "expired_lease_reaped"
    assert work_order["metadata"]["archived_due_to"] == "reaped_no_receipt"
    assert work_order["metadata"]["archive_reason"] == "expired_lease_reaped"
    assert work_order["metadata"]["previous_status"] == "dispatched"


def test_archive_scope_violation_no_deliverable_work_orders_discards_old_backlog(
    store: DevCoordinationStore,
) -> None:
    run = store.create_supervisor_run(
        goal="Archive old scope violation backlog",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={
            "require_merge_approval": True,
            "require_external_action_approval": True,
        },
        spec={
            "raw_goal": "Archive old scope violation backlog",
            "refined_goal": "Archive old scope violation backlog",
        },
        work_orders=[
            {
                "work_order_id": "wo-scope-archive",
                "title": "Old scope violation lane",
                "file_scope": ["aragora/swarm/reporter.py"],
                "status": "scope_violation",
                "completed_at": "None",
                "target_agent": "codex",
                "reviewer_agent": "claude",
            }
        ],
    )

    conn = store._connect()
    try:
        conn.execute(
            "UPDATE supervisor_runs SET created_at = ?, updated_at = ? WHERE run_id = ?",
            ("2000-01-01T00:00:00+00:00", "2000-01-01T00:00:00+00:00", run["run_id"]),
        )
        conn.commit()
    finally:
        conn.close()

    archived = store.archive_scope_violation_no_deliverable_work_orders(grace_period_hours=6.0)
    refreshed = store.get_supervisor_run(run["run_id"])

    assert archived == 1
    assert refreshed is not None
    work_order = refreshed["work_orders"][0]
    assert work_order["status"] == "discarded"
    assert work_order["failure_reason"] == "scope_violation"
    assert work_order["metadata"]["archived_due_to"] == "scope_violation_no_deliverable"
    assert work_order["metadata"]["archive_reason"] == "scope_violation"
    assert work_order["metadata"]["previous_status"] == "scope_violation"


def test_archive_scope_violation_no_deliverable_work_orders_preserves_deliverable_lane(
    store: DevCoordinationStore,
) -> None:
    run = store.create_supervisor_run(
        goal="Keep deliverable scope violation lane",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={
            "require_merge_approval": True,
            "require_external_action_approval": True,
        },
        spec={
            "raw_goal": "Keep deliverable scope violation lane",
            "refined_goal": "Keep deliverable scope violation lane",
        },
        work_orders=[
            {
                "work_order_id": "wo-scope-deliverable",
                "title": "Deliverable scope violation lane",
                "file_scope": ["aragora/swarm/reporter.py"],
                "status": "scope_violation",
                "branch": "codex/scope-deliverable",
                "commit_shas": ["deadbeef"],
                "target_agent": "codex",
                "reviewer_agent": "claude",
            }
        ],
    )

    conn = store._connect()
    try:
        conn.execute(
            "UPDATE supervisor_runs SET created_at = ?, updated_at = ? WHERE run_id = ?",
            ("2000-01-01T00:00:00+00:00", "2000-01-01T00:00:00+00:00", run["run_id"]),
        )
        conn.commit()
    finally:
        conn.close()

    archived = store.archive_scope_violation_no_deliverable_work_orders(grace_period_hours=6.0)
    refreshed = store.get_supervisor_run(run["run_id"])

    assert archived == 0
    assert refreshed is not None
    assert refreshed["work_orders"][0]["status"] == "scope_violation"


def test_archive_failed_no_deliverable_work_orders_discards_old_backlog(
    store: DevCoordinationStore,
) -> None:
    run = store.create_supervisor_run(
        goal="Archive old failed backlog",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={
            "require_merge_approval": True,
            "require_external_action_approval": True,
        },
        spec={
            "raw_goal": "Archive old failed backlog",
            "refined_goal": "Archive old failed backlog",
        },
        work_orders=[
            {
                "work_order_id": "wo-failed-archive",
                "title": "Old failed lane",
                "file_scope": ["aragora/swarm/reporter.py"],
                "status": "failed",
                "blockers": ["Worker crashed before producing an acceptable terminal result."],
                "target_agent": "codex",
                "reviewer_agent": "claude",
            }
        ],
    )

    conn = store._connect()
    try:
        conn.execute(
            "UPDATE supervisor_runs SET created_at = ?, updated_at = ? WHERE run_id = ?",
            ("2000-01-01T00:00:00+00:00", "2000-01-01T00:00:00+00:00", run["run_id"]),
        )
        conn.commit()
    finally:
        conn.close()

    archived = store.archive_failed_no_deliverable_work_orders(grace_period_hours=24.0)
    refreshed = store.get_supervisor_run(run["run_id"])

    assert archived == 1
    assert refreshed is not None
    work_order = refreshed["work_orders"][0]
    assert work_order["status"] == "discarded"
    assert (
        work_order["failure_reason"]
        == "Worker crashed before producing an acceptable terminal result."
    )
    assert work_order["metadata"]["archived_due_to"] == "failed_no_deliverable"
    assert (
        work_order["metadata"]["archive_reason"]
        == "Worker crashed before producing an acceptable terminal result."
    )
    assert work_order["metadata"]["previous_status"] == "failed"


def test_archive_failed_no_deliverable_work_orders_preserves_deliverable_lane(
    store: DevCoordinationStore,
) -> None:
    run = store.create_supervisor_run(
        goal="Keep failed deliverable lane",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={
            "require_merge_approval": True,
            "require_external_action_approval": True,
        },
        spec={
            "raw_goal": "Keep failed deliverable lane",
            "refined_goal": "Keep failed deliverable lane",
        },
        work_orders=[
            {
                "work_order_id": "wo-failed-deliverable",
                "title": "Failed deliverable lane",
                "file_scope": ["aragora/swarm/reporter.py"],
                "status": "failed",
                "branch": "codex/failed-deliverable",
                "commit_shas": ["deadbeef"],
                "target_agent": "codex",
                "reviewer_agent": "claude",
            }
        ],
    )

    conn = store._connect()
    try:
        conn.execute(
            "UPDATE supervisor_runs SET created_at = ?, updated_at = ? WHERE run_id = ?",
            ("2000-01-01T00:00:00+00:00", "2000-01-01T00:00:00+00:00", run["run_id"]),
        )
        conn.commit()
    finally:
        conn.close()

    archived = store.archive_failed_no_deliverable_work_orders(grace_period_hours=24.0)
    refreshed = store.get_supervisor_run(run["run_id"])

    assert archived == 0
    assert refreshed is not None
    assert refreshed["work_orders"][0]["status"] == "failed"


def test_archive_failed_no_deliverable_work_orders_discards_old_timeout_needs_human_backlog(
    store: DevCoordinationStore,
) -> None:
    run = store.create_supervisor_run(
        goal="Archive old timeout backlog",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={
            "require_merge_approval": True,
            "require_external_action_approval": True,
        },
        spec={
            "raw_goal": "Archive old timeout backlog",
            "refined_goal": "Archive old timeout backlog",
        },
        work_orders=[
            {
                "work_order_id": "wo-timeout-archive",
                "title": "Old timeout lane",
                "file_scope": ["aragora/swarm/reporter.py"],
                "status": "needs_human",
                "worker_outcome": "timeout_no_progress",
                "blockers": ["worker exceeded no-progress timeout (120s)"],
                "target_agent": "codex",
                "reviewer_agent": "claude",
            }
        ],
    )

    conn = store._connect()
    try:
        conn.execute(
            "UPDATE supervisor_runs SET created_at = ?, updated_at = ? WHERE run_id = ?",
            (
                "2000-01-01T00:00:00+00:00",
                "2000-01-01T00:00:00+00:00",
                run["run_id"],
            ),
        )
        conn.commit()
    finally:
        conn.close()

    archived = store.archive_failed_no_deliverable_work_orders(grace_period_hours=24.0)
    refreshed = store.get_supervisor_run(run["run_id"])

    assert archived == 1
    assert refreshed is not None
    work_order = refreshed["work_orders"][0]
    assert work_order["status"] == "discarded"
    assert work_order["metadata"]["archived_due_to"] == "failed_no_deliverable"
    assert work_order["metadata"]["archive_reason"] == "worker exceeded no-progress timeout (120s)"
    assert work_order["metadata"]["previous_status"] == "needs_human"


def test_archive_clean_exit_no_deliverable_work_orders_discards_old_backlog(
    store: DevCoordinationStore,
) -> None:
    run = store.create_supervisor_run(
        goal="Archive old clean exit backlog",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={
            "require_merge_approval": True,
            "require_external_action_approval": True,
        },
        spec={
            "raw_goal": "Archive old clean exit backlog",
            "refined_goal": "Archive old clean exit backlog",
        },
        work_orders=[
            {
                "work_order_id": "wo-clean-exit-archive",
                "title": "Old clean exit lane",
                "file_scope": ["aragora/swarm/reporter.py"],
                "status": "completed",
                "blockers": ["Run ended without a concrete deliverable."],
                "target_agent": "codex",
                "reviewer_agent": "claude",
            }
        ],
    )

    conn = store._connect()
    try:
        conn.execute(
            "UPDATE supervisor_runs SET created_at = ?, updated_at = ? WHERE run_id = ?",
            (
                "2000-01-01T00:00:00+00:00",
                "2000-01-01T00:00:00+00:00",
                run["run_id"],
            ),
        )
        conn.commit()
    finally:
        conn.close()

    archived = store.archive_clean_exit_no_deliverable_work_orders(grace_period_hours=24.0)
    refreshed = store.get_supervisor_run(run["run_id"])

    assert archived == 1
    assert refreshed is not None
    work_order = refreshed["work_orders"][0]
    assert work_order["status"] == "discarded"
    assert work_order["metadata"]["archived_due_to"] == "clean_exit_no_deliverable"
    assert work_order["metadata"]["archive_reason"] == "Run ended without a concrete deliverable."
    assert work_order["metadata"]["previous_status"] == "completed"


def test_archive_clean_exit_no_deliverable_work_orders_uses_run_created_at_anchor(
    store: DevCoordinationStore,
) -> None:
    run = store.create_supervisor_run(
        goal="Archive old clean exit backlog despite fresh run updates",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={
            "require_merge_approval": True,
            "require_external_action_approval": True,
        },
        spec={
            "raw_goal": "Archive old clean exit backlog despite fresh run updates",
            "refined_goal": "Archive old clean exit backlog despite fresh run updates",
        },
        work_orders=[
            {
                "work_order_id": "wo-clean-exit-created-anchor",
                "title": "Old clean exit lane with fresh sync timestamp",
                "file_scope": ["aragora/swarm/supervisor.py"],
                "status": "needs_human",
                "failure_reason": "clean_exit_no_deliverable",
                "blockers": ["Run ended without a concrete deliverable."],
                "target_agent": "codex",
                "reviewer_agent": "claude",
            }
        ],
    )

    conn = store._connect()
    try:
        conn.execute(
            "UPDATE supervisor_runs SET created_at = ?, updated_at = ? WHERE run_id = ?",
            (
                "2000-01-01T00:00:00+00:00",
                "2999-01-01T00:00:00+00:00",
                run["run_id"],
            ),
        )
        conn.commit()
    finally:
        conn.close()

    archived = store.archive_clean_exit_no_deliverable_work_orders(grace_period_hours=24.0)
    refreshed = store.get_supervisor_run(run["run_id"])

    assert archived == 1
    assert refreshed is not None
    work_order = refreshed["work_orders"][0]
    assert work_order["status"] == "discarded"
    assert work_order["metadata"]["archived_due_to"] == "clean_exit_no_deliverable"
    assert work_order["metadata"]["previous_status"] == "needs_human"


def test_archive_clean_exit_no_deliverable_work_orders_preserves_deliverable_lane(
    store: DevCoordinationStore,
) -> None:
    run = store.create_supervisor_run(
        goal="Keep clean exit deliverable lane",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={
            "require_merge_approval": True,
            "require_external_action_approval": True,
        },
        spec={
            "raw_goal": "Keep clean exit deliverable lane",
            "refined_goal": "Keep clean exit deliverable lane",
        },
        work_orders=[
            {
                "work_order_id": "wo-clean-exit-deliverable",
                "title": "Completed deliverable lane",
                "status": "completed",
                "branch": "codex/clean-exit-deliverable",
                "commit_shas": ["deadbeef"],
                "target_agent": "codex",
                "reviewer_agent": "claude",
            }
        ],
    )

    conn = store._connect()
    try:
        conn.execute(
            "UPDATE supervisor_runs SET created_at = ?, updated_at = ? WHERE run_id = ?",
            (
                "2000-01-01T00:00:00+00:00",
                "2000-01-01T00:00:00+00:00",
                run["run_id"],
            ),
        )
        conn.commit()
    finally:
        conn.close()

    archived = store.archive_clean_exit_no_deliverable_work_orders(grace_period_hours=24.0)
    refreshed = store.get_supervisor_run(run["run_id"])

    assert archived == 0
    assert refreshed is not None
    assert refreshed["work_orders"][0]["status"] == "completed"


def test_archive_clean_exit_no_deliverable_work_orders_infers_historical_completed_no_artifact(
    store: DevCoordinationStore,
) -> None:
    run = store.create_supervisor_run(
        goal="Archive historical inferred clean exit",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={
            "require_merge_approval": True,
            "require_external_action_approval": True,
        },
        spec={
            "raw_goal": "Archive historical inferred clean exit",
            "refined_goal": "Archive historical inferred clean exit",
        },
        work_orders=[
            {
                "work_order_id": "wo-clean-exit-inferred",
                "title": "Historical completed no-artifact lane",
                "status": "completed",
                "changed_paths": [],
                "commit_shas": [],
                "target_agent": "codex",
                "reviewer_agent": "claude",
            }
        ],
    )

    conn = store._connect()
    try:
        conn.execute(
            "UPDATE supervisor_runs SET created_at = ?, updated_at = ? WHERE run_id = ?",
            (
                "2000-01-01T00:00:00+00:00",
                "2000-01-01T00:00:00+00:00",
                run["run_id"],
            ),
        )
        conn.commit()
    finally:
        conn.close()

    archived = store.archive_clean_exit_no_deliverable_work_orders(grace_period_hours=24.0)
    refreshed = store.get_supervisor_run(run["run_id"])

    assert archived == 1
    assert refreshed is not None
    work_order = refreshed["work_orders"][0]
    assert work_order["status"] == "discarded"
    assert work_order["metadata"]["archived_due_to"] == "clean_exit_no_deliverable"
    assert work_order["metadata"]["archive_reason"] == "clean_exit_no_deliverable"


def test_archive_duplicate_branch_deliverable_work_orders_keeps_canonical_lane(
    store: DevCoordinationStore,
) -> None:
    run = store.create_supervisor_run(
        goal="Collapse duplicate branch family",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={
            "require_merge_approval": True,
            "require_external_action_approval": True,
        },
        spec={
            "raw_goal": "Collapse duplicate branch family",
            "refined_goal": "Collapse duplicate branch family",
        },
        work_orders=[
            {
                "work_order_id": "subtask_1",
                "title": "Duplicate branch sibling 1",
                "status": "completed",
                "branch": "codex/swarm-duplicate-subtask_",
                "head_sha": "aaa111",
                "commit_shas": ["aaa111"],
            },
            {
                "work_order_id": "subtask_2",
                "title": "Duplicate branch sibling 2",
                "status": "completed",
                "branch": "codex/swarm-duplicate-subtask_",
                "head_sha": "bbb222",
                "commit_shas": ["aaa111", "bbb222"],
            },
            {
                "work_order_id": "subtask_3",
                "title": "Duplicate branch canonical lane",
                "status": "completed",
                "branch": "codex/swarm-duplicate-subtask_",
                "receipt_id": "rcpt-duplicate",
                "head_sha": "ccc333",
                "commit_shas": ["aaa111", "bbb222", "ccc333"],
            },
        ],
    )

    conn = store._connect()
    try:
        conn.execute(
            "UPDATE supervisor_runs SET updated_at = ? WHERE run_id = ?",
            ("2000-01-01T00:00:00+00:00", run["run_id"]),
        )
        conn.commit()
    finally:
        conn.close()

    archived = store.archive_duplicate_branch_deliverable_work_orders(grace_period_hours=24.0)
    refreshed = store.get_supervisor_run(run["run_id"])

    assert archived == 2
    assert refreshed is not None
    work_orders = {item["work_order_id"]: item for item in refreshed["work_orders"]}
    assert work_orders["subtask_3"]["status"] == "completed"
    assert work_orders["subtask_3"]["receipt_id"] == "rcpt-duplicate"
    assert work_orders["subtask_1"]["status"] == "discarded"
    assert work_orders["subtask_2"]["status"] == "discarded"
    assert work_orders["subtask_1"]["metadata"]["archived_due_to"] == "duplicate_branch_deliverable"
    assert work_orders["subtask_2"]["metadata"]["canonical_work_order_id"] == "subtask_3"
    assert work_orders["subtask_1"]["failure_reason"] == "duplicate_branch_deliverable"


def test_archive_superseded_waiting_conflict_work_orders_discards_old_overlapping_siblings(
    store: DevCoordinationStore,
) -> None:
    run = store.create_supervisor_run(
        goal="Collapse stale waiting conflict siblings",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={
            "require_merge_approval": True,
            "require_external_action_approval": True,
        },
        spec={
            "raw_goal": "Collapse stale waiting conflict siblings",
            "refined_goal": "Collapse stale waiting conflict siblings",
        },
        work_orders=[
            {
                "work_order_id": "subtask_1",
                "title": "Canonical deliverable lane",
                "status": "completed",
                "branch": "codex/superseded-keeper",
                "head_sha": "abc123",
                "commit_shas": ["abc123"],
                "file_scope": ["docs/ADR/021-storage-layer-consolidation.md"],
                "changed_paths": ["docs/ADR/021-storage-layer-consolidation.md"],
                "receipt_id": "rcpt-superseded-keeper",
            },
            {
                "work_order_id": "subtask_2",
                "title": "Directory-wide waiting conflict",
                "status": "waiting_conflict",
                "file_scope": ["docs/ADR/"],
            },
            {
                "work_order_id": "subtask_3",
                "title": "Exact-file waiting conflict",
                "status": "waiting_conflict",
                "file_scope": ["docs/ADR/021-storage-layer-consolidation.md"],
            },
        ],
    )

    conn = store._connect()
    try:
        conn.execute(
            "UPDATE supervisor_runs SET created_at = ?, updated_at = ? WHERE run_id = ?",
            ("2000-01-01T00:00:00+00:00", "2000-01-01T00:00:00+00:00", run["run_id"]),
        )
        conn.commit()
    finally:
        conn.close()

    archived = store.archive_superseded_waiting_conflict_work_orders(grace_period_hours=24.0)
    refreshed = store.get_supervisor_run(run["run_id"])

    assert archived == 2
    assert refreshed is not None
    work_orders = {item["work_order_id"]: item for item in refreshed["work_orders"]}
    assert work_orders["subtask_1"]["status"] == "completed"
    assert work_orders["subtask_2"]["status"] == "discarded"
    assert work_orders["subtask_3"]["status"] == "discarded"
    assert work_orders["subtask_2"]["metadata"]["archived_due_to"] == "superseded_waiting_conflict"
    assert work_orders["subtask_3"]["metadata"]["canonical_work_order_id"] == "subtask_1"
    assert work_orders["subtask_2"]["failure_reason"] == "superseded_waiting_conflict"


def test_archive_superseded_waiting_conflict_work_orders_preserves_non_overlapping_lane(
    store: DevCoordinationStore,
) -> None:
    run = store.create_supervisor_run(
        goal="Keep unrelated waiting conflict lane",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={
            "require_merge_approval": True,
            "require_external_action_approval": True,
        },
        spec={
            "raw_goal": "Keep unrelated waiting conflict lane",
            "refined_goal": "Keep unrelated waiting conflict lane",
        },
        work_orders=[
            {
                "work_order_id": "subtask_1",
                "title": "Completed README lane",
                "status": "completed",
                "branch": "codex/unrelated-keeper",
                "head_sha": "def456",
                "commit_shas": ["def456"],
                "file_scope": ["README.md"],
                "changed_paths": ["README.md"],
                "receipt_id": "rcpt-unrelated-keeper",
            },
            {
                "work_order_id": "subtask_2",
                "title": "Compliance waiting conflict",
                "status": "waiting_conflict",
                "file_scope": ["aragora/compliance/"],
            },
        ],
    )

    conn = store._connect()
    try:
        conn.execute(
            "UPDATE supervisor_runs SET updated_at = ? WHERE run_id = ?",
            ("2000-01-01T00:00:00+00:00", run["run_id"]),
        )
        conn.commit()
    finally:
        conn.close()

    archived = store.archive_superseded_waiting_conflict_work_orders(grace_period_hours=24.0)
    refreshed = store.get_supervisor_run(run["run_id"])

    assert archived == 0
    assert refreshed is not None
    assert refreshed["work_orders"][1]["status"] == "waiting_conflict"


def test_archive_superseded_waiting_conflict_work_orders_discards_duplicate_same_scope_siblings(
    store: DevCoordinationStore,
) -> None:
    run = store.create_supervisor_run(
        goal="Collapse duplicate waiting conflict siblings without deliverables",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={
            "require_merge_approval": True,
            "require_external_action_approval": True,
        },
        spec={
            "raw_goal": "Collapse duplicate waiting conflict siblings without deliverables",
            "refined_goal": "Collapse duplicate waiting conflict siblings without deliverables",
        },
        work_orders=[
            {
                "work_order_id": "subtask_1",
                "title": "Canonical waiting conflict",
                "status": "waiting_conflict",
                "file_scope": [
                    "aragora/cli/commands/quickstart.py",
                    "tests/cli/test_quickstart.py",
                ],
            },
            {
                "work_order_id": "subtask_2",
                "title": "Duplicate waiting conflict sibling",
                "status": "waiting_conflict",
                "file_scope": [
                    "aragora/cli/commands/quickstart.py",
                    "tests/cli/test_quickstart.py",
                ],
            },
            {
                "work_order_id": "subtask_3",
                "title": "Distinct waiting conflict sibling",
                "status": "waiting_conflict",
                "file_scope": ["aragora/server/handlers/playground.py"],
            },
        ],
    )

    conn = store._connect()
    try:
        conn.execute(
            "UPDATE supervisor_runs SET created_at = ?, updated_at = ? WHERE run_id = ?",
            ("2000-01-01T00:00:00+00:00", "2000-01-01T00:00:00+00:00", run["run_id"]),
        )
        conn.commit()
    finally:
        conn.close()

    archived = store.archive_superseded_waiting_conflict_work_orders(grace_period_hours=24.0)
    refreshed = store.get_supervisor_run(run["run_id"])

    assert archived == 1
    assert refreshed is not None
    work_orders = {item["work_order_id"]: item for item in refreshed["work_orders"]}
    kept = [
        item["work_order_id"]
        for item in refreshed["work_orders"]
        if item["status"] == "waiting_conflict"
    ]
    assert sorted(kept) == ["subtask_1", "subtask_3"]
    assert work_orders["subtask_2"]["status"] == "discarded"
    assert work_orders["subtask_2"]["metadata"]["archived_due_to"] == "superseded_waiting_conflict"
    assert (
        work_orders["subtask_2"]["metadata"]["archive_reason"]
        == "duplicate_waiting_conflict_sibling"
    )
    assert work_orders["subtask_2"]["metadata"]["canonical_work_order_id"] == "subtask_1"
    assert work_orders["subtask_2"]["failure_reason"] == "superseded_waiting_conflict"


def test_archive_superseded_waiting_conflict_work_orders_discards_same_scope_siblings_with_reordered_paths(
    store: DevCoordinationStore,
) -> None:
    run = store.create_supervisor_run(
        goal="Collapse duplicate waiting conflict siblings with reordered scope",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={
            "require_merge_approval": True,
            "require_external_action_approval": True,
        },
        spec={
            "raw_goal": "Collapse duplicate waiting conflict siblings with reordered scope",
            "refined_goal": "Collapse duplicate waiting conflict siblings with reordered scope",
        },
        work_orders=[
            {
                "work_order_id": "subtask_1",
                "title": "Canonical waiting conflict",
                "status": "waiting_conflict",
                "file_scope": ["aragora/ralph/classifier.py", "tests/ralph/test_classifier.py"],
            },
            {
                "work_order_id": "subtask_2",
                "title": "Reordered waiting conflict sibling",
                "status": "waiting_conflict",
                "file_scope": ["tests/ralph/test_classifier.py", "aragora/ralph/classifier.py"],
            },
        ],
    )

    conn = store._connect()
    try:
        conn.execute(
            "UPDATE supervisor_runs SET created_at = ?, updated_at = ? WHERE run_id = ?",
            ("2000-01-01T00:00:00+00:00", "2000-01-01T00:00:00+00:00", run["run_id"]),
        )
        conn.commit()
    finally:
        conn.close()

    archived = store.archive_superseded_waiting_conflict_work_orders(grace_period_hours=24.0)
    refreshed = store.get_supervisor_run(run["run_id"])

    assert archived == 1
    assert refreshed is not None
    work_orders = {item["work_order_id"]: item for item in refreshed["work_orders"]}
    assert work_orders["subtask_1"]["status"] == "waiting_conflict"
    assert work_orders["subtask_2"]["status"] == "discarded"
    assert (
        work_orders["subtask_2"]["metadata"]["archive_reason"]
        == "duplicate_waiting_conflict_sibling"
    )


def test_archive_superseded_waiting_conflict_work_orders_discards_contained_narrower_sibling(
    store: DevCoordinationStore,
) -> None:
    run = store.create_supervisor_run(
        goal="Collapse contained waiting conflict sibling",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={
            "require_merge_approval": True,
            "require_external_action_approval": True,
        },
        spec={
            "raw_goal": "Collapse contained waiting conflict sibling",
            "refined_goal": "Collapse contained waiting conflict sibling",
        },
        work_orders=[
            {
                "work_order_id": "subtask_1",
                "title": "Broader waiting conflict",
                "status": "waiting_conflict",
                "file_scope": ["docs/governance/"],
            },
            {
                "work_order_id": "subtask_2",
                "title": "Contained narrower waiting conflict",
                "status": "waiting_conflict",
                "file_scope": ["docs/governance/duplicate-subsystem-resolution.md"],
            },
        ],
    )

    conn = store._connect()
    try:
        conn.execute(
            "UPDATE supervisor_runs SET created_at = ?, updated_at = ? WHERE run_id = ?",
            ("2000-01-01T00:00:00+00:00", "2000-01-01T00:00:00+00:00", run["run_id"]),
        )
        conn.commit()
    finally:
        conn.close()

    archived = store.archive_superseded_waiting_conflict_work_orders(grace_period_hours=24.0)
    refreshed = store.get_supervisor_run(run["run_id"])

    assert archived == 1
    assert refreshed is not None
    work_orders = {item["work_order_id"]: item for item in refreshed["work_orders"]}
    assert work_orders["subtask_1"]["status"] == "waiting_conflict"
    assert work_orders["subtask_2"]["status"] == "discarded"
    assert (
        work_orders["subtask_2"]["metadata"]["archive_reason"]
        == "contained_waiting_conflict_sibling"
    )
    assert work_orders["subtask_2"]["metadata"]["canonical_work_order_id"] == "subtask_1"


def test_archive_superseded_waiting_conflict_work_orders_discards_cross_run_contained_sibling(
    store: DevCoordinationStore,
) -> None:
    older = store.create_supervisor_run(
        goal="Write an ADR for standardizing health check endpoints across all deploy surfaces.",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={},
        spec={},
        work_orders=[
            {
                "work_order_id": "subtask_1",
                "title": "Narrow ADR lane",
                "status": "waiting_conflict",
                "file_scope": ["docs/ADR/019-standardized-health-check-endpoints.md"],
            }
        ],
    )
    newer = store.create_supervisor_run(
        goal="Write an ADR for standardizing health check endpoints across all deploy surfaces.",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={},
        spec={},
        work_orders=[
            {
                "work_order_id": "subtask_2",
                "title": "Broader ADR lane",
                "status": "waiting_conflict",
                "file_scope": ["docs/ADR/"],
            }
        ],
    )

    conn = store._connect()
    try:
        conn.execute(
            "UPDATE supervisor_runs SET created_at = ?, updated_at = ? WHERE run_id = ?",
            ("2000-01-01T00:00:00+00:00", "2000-01-01T00:00:00+00:00", older["run_id"]),
        )
        conn.execute(
            "UPDATE supervisor_runs SET created_at = ?, updated_at = ? WHERE run_id = ?",
            ("2000-01-02T00:00:00+00:00", "2000-01-02T00:00:00+00:00", newer["run_id"]),
        )
        conn.commit()
    finally:
        conn.close()

    archived = store.archive_superseded_waiting_conflict_work_orders(grace_period_hours=24.0)
    older_refreshed = store.get_supervisor_run(older["run_id"])
    newer_refreshed = store.get_supervisor_run(newer["run_id"])

    assert archived == 1
    assert older_refreshed is not None
    assert newer_refreshed is not None
    assert older_refreshed["work_orders"][0]["status"] == "discarded"
    assert (
        older_refreshed["work_orders"][0]["metadata"]["archive_reason"]
        == "cross_run_contained_waiting_conflict_sibling"
    )
    assert older_refreshed["work_orders"][0]["metadata"]["canonical_run_id"] == newer["run_id"]
    assert older_refreshed["work_orders"][0]["metadata"]["canonical_work_order_id"] == "subtask_2"
    assert newer_refreshed["work_orders"][0]["status"] == "waiting_conflict"


def test_archive_superseded_waiting_conflict_work_orders_discards_cross_run_contained_same_lane_with_goal_drift(
    store: DevCoordinationStore,
) -> None:
    older = store.create_supervisor_run(
        goal="Tighten the design partner motion into a bounded founder sales artifact.",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={},
        spec={},
        work_orders=[
            {
                "work_order_id": "proj-001",
                "title": "Narrow docs lane",
                "status": "waiting_conflict",
                "file_scope": ["docs/outreach/**", "docs/plans/**", "docs/strategy/**"],
                "metadata": {
                    "source": "explicit_spec_work_order",
                    "tranche_lane_id": "proj-001",
                },
            }
        ],
    )
    newer = store.create_supervisor_run(
        goal="Connect the results page to backend endpoints with truthful live-state behavior.",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={},
        spec={},
        work_orders=[
            {
                "work_order_id": "proj-001",
                "title": "Broader docs lane",
                "status": "waiting_conflict",
                "file_scope": ["aragora/live/**", "tests/e2e/**", "tests/handlers/**", "docs/**"],
                "metadata": {
                    "source": "explicit_spec_work_order",
                    "tranche_lane_id": "proj-001",
                },
            }
        ],
    )

    conn = store._connect()
    try:
        conn.execute(
            "UPDATE supervisor_runs SET created_at = ?, updated_at = ? WHERE run_id = ?",
            ("2000-01-01T00:00:00+00:00", "2000-01-01T00:00:00+00:00", older["run_id"]),
        )
        conn.execute(
            "UPDATE supervisor_runs SET created_at = ?, updated_at = ? WHERE run_id = ?",
            ("2000-01-02T00:00:00+00:00", "2000-01-02T00:00:00+00:00", newer["run_id"]),
        )
        conn.commit()
    finally:
        conn.close()

    archived = store.archive_superseded_waiting_conflict_work_orders(grace_period_hours=24.0)
    older_refreshed = store.get_supervisor_run(older["run_id"])
    newer_refreshed = store.get_supervisor_run(newer["run_id"])

    assert archived == 1
    assert older_refreshed is not None
    assert newer_refreshed is not None
    assert older_refreshed["work_orders"][0]["status"] == "discarded"
    assert (
        older_refreshed["work_orders"][0]["metadata"]["archive_reason"]
        == "cross_run_contained_waiting_conflict_sibling"
    )
    assert older_refreshed["work_orders"][0]["metadata"]["canonical_run_id"] == newer["run_id"]
    assert older_refreshed["work_orders"][0]["metadata"]["canonical_work_order_id"] == "proj-001"
    assert newer_refreshed["work_orders"][0]["status"] == "waiting_conflict"


def test_archive_superseded_waiting_conflict_work_orders_discards_glob_and_trailing_slash_variants(
    store: DevCoordinationStore,
) -> None:
    run = store.create_supervisor_run(
        goal="Collapse live-like waiting conflict scope variants",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={
            "require_merge_approval": True,
            "require_external_action_approval": True,
        },
        spec={
            "raw_goal": "Collapse live-like waiting conflict scope variants",
            "refined_goal": "Collapse live-like waiting conflict scope variants",
        },
        work_orders=[
            {
                "work_order_id": "subtask_1",
                "title": "Canonical waiting conflict",
                "status": "waiting_conflict",
                "file_scope": [
                    "aragora/live/**",
                    "aragora/live/handlers/",
                    "docs/**",
                    "tests/e2e/**",
                    "tests/handlers/",
                    "tests/handlers/**",
                ],
            },
            {
                "work_order_id": "subtask_2",
                "title": "Wildcard-only sibling",
                "status": "waiting_conflict",
                "file_scope": [
                    "aragora/live/**",
                    "docs/**",
                    "tests/e2e/**",
                    "tests/handlers/**",
                ],
            },
            {
                "work_order_id": "subtask_3",
                "title": "Trailing slash sibling",
                "status": "waiting_conflict",
                "file_scope": [
                    "aragora/live/**",
                    "docs/**",
                    "tests/e2e/",
                    "tests/e2e/**",
                    "tests/handlers/",
                    "tests/handlers/**",
                ],
            },
            {
                "work_order_id": "subtask_4",
                "title": "Docs slash variant sibling",
                "status": "waiting_conflict",
                "file_scope": [
                    "aragora/live/**",
                    "docs/",
                    "docs/**",
                    "tests/e2e/**",
                    "tests/handlers/**",
                ],
            },
        ],
    )

    conn = store._connect()
    try:
        conn.execute(
            "UPDATE supervisor_runs SET created_at = ?, updated_at = ? WHERE run_id = ?",
            ("2000-01-01T00:00:00+00:00", "2000-01-01T00:00:00+00:00", run["run_id"]),
        )
        conn.commit()
    finally:
        conn.close()

    archived = store.archive_superseded_waiting_conflict_work_orders(grace_period_hours=24.0)
    refreshed = store.get_supervisor_run(run["run_id"])

    assert archived == 3
    assert refreshed is not None
    work_orders = {item["work_order_id"]: item for item in refreshed["work_orders"]}
    kept = [
        item["work_order_id"]
        for item in refreshed["work_orders"]
        if item["status"] == "waiting_conflict"
    ]
    assert kept == ["subtask_1"]
    for discarded_id in ("subtask_2", "subtask_3", "subtask_4"):
        assert work_orders[discarded_id]["status"] == "discarded"
        assert (
            work_orders[discarded_id]["metadata"]["archive_reason"]
            == "duplicate_waiting_conflict_sibling"
        )
        assert work_orders[discarded_id]["metadata"]["canonical_work_order_id"] == "subtask_1"


def test_archive_work_order_leasing_failed_work_orders_discards_old_no_deliverable_lane(
    store: DevCoordinationStore,
) -> None:
    run = store.create_supervisor_run(
        goal="Archive stale leasing failure without deliverable",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={
            "require_merge_approval": True,
            "require_external_action_approval": True,
        },
        spec={
            "raw_goal": "Archive stale leasing failure without deliverable",
            "refined_goal": "Archive stale leasing failure without deliverable",
        },
        work_orders=[
            {
                "work_order_id": "subtask_1",
                "title": "Stale leasing failure",
                "status": "needs_human",
                "failure_reason": "work_order_leasing_failed",
                "dispatch_error": "autopilot ensure failed (1): branch already exists",
                "file_scope": ["aragora/cli/commands/quickstart.py"],
            }
        ],
    )

    conn = store._connect()
    try:
        conn.execute(
            "UPDATE supervisor_runs SET created_at = ?, updated_at = ? WHERE run_id = ?",
            ("2000-01-01T00:00:00+00:00", "2000-01-01T00:00:00+00:00", run["run_id"]),
        )
        conn.commit()
    finally:
        conn.close()

    archived = store.archive_work_order_leasing_failed_work_orders(grace_period_hours=24.0)
    refreshed = store.get_supervisor_run(run["run_id"])

    assert archived == 1
    assert refreshed is not None
    work_order = refreshed["work_orders"][0]
    assert work_order["status"] == "discarded"
    assert work_order["metadata"]["archived_due_to"] == "work_order_leasing_failed"
    assert work_order["metadata"]["archive_reason"] == "work_order_leasing_failed"


def test_archive_duplicate_waiting_conflict_work_orders_discards_older_cross_run_duplicate(
    store: DevCoordinationStore,
) -> None:
    older = store.create_supervisor_run(
        goal="Add --json output flag to aragora quickstart CLI",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={
            "require_merge_approval": True,
            "require_external_action_approval": True,
        },
        spec={
            "raw_goal": "Add --json output flag to aragora quickstart CLI",
            "refined_goal": "Add --json output flag to aragora quickstart CLI",
        },
        work_orders=[
            {
                "work_order_id": "subtask_1",
                "title": "Test Changes",
                "status": "waiting_conflict",
                "file_scope": [
                    "aragora/cli/commands/quickstart.py",
                    "aragora/cli/parser.py",
                    "tests/cli/test_quickstart.py",
                ],
            }
        ],
    )
    newer = store.create_supervisor_run(
        goal="Add --json output flag to aragora quickstart CLI",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={
            "require_merge_approval": True,
            "require_external_action_approval": True,
        },
        spec={
            "raw_goal": "Add --json output flag to aragora quickstart CLI",
            "refined_goal": "Add --json output flag to aragora quickstart CLI",
        },
        work_orders=[
            {
                "work_order_id": "subtask_2",
                "title": "Test Changes",
                "status": "waiting_conflict",
                "file_scope": [
                    "aragora/cli/commands/quickstart.py",
                    "aragora/cli/parser.py",
                    "tests/cli/test_quickstart.py",
                ],
            }
        ],
    )
    distinct = store.create_supervisor_run(
        goal="Add JSONL metrics logging per boss loop iteration",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={
            "require_merge_approval": True,
            "require_external_action_approval": True,
        },
        spec={
            "raw_goal": "Add JSONL metrics logging per boss loop iteration",
            "refined_goal": "Add JSONL metrics logging per boss loop iteration",
        },
        work_orders=[
            {
                "work_order_id": "subtask_3",
                "title": "Test Changes",
                "status": "waiting_conflict",
                "file_scope": [
                    "aragora/cli/commands/quickstart.py",
                    "aragora/cli/parser.py",
                    "tests/cli/test_quickstart.py",
                ],
            }
        ],
    )

    conn = store._connect()
    try:
        conn.execute(
            "UPDATE supervisor_runs SET created_at = ?, updated_at = ? WHERE run_id = ?",
            ("2000-01-01T00:00:00+00:00", "2000-01-01T00:00:00+00:00", older["run_id"]),
        )
        conn.execute(
            "UPDATE supervisor_runs SET created_at = ?, updated_at = ? WHERE run_id = ?",
            ("2000-01-02T00:00:00+00:00", "2000-01-02T00:00:00+00:00", newer["run_id"]),
        )
        conn.execute(
            "UPDATE supervisor_runs SET created_at = ?, updated_at = ? WHERE run_id = ?",
            ("2000-01-03T00:00:00+00:00", "2000-01-03T00:00:00+00:00", distinct["run_id"]),
        )
        conn.commit()
    finally:
        conn.close()

    archived = store.archive_duplicate_waiting_conflict_work_orders()
    older_refreshed = store.get_supervisor_run(older["run_id"])
    newer_refreshed = store.get_supervisor_run(newer["run_id"])
    distinct_refreshed = store.get_supervisor_run(distinct["run_id"])

    assert archived == 1
    assert older_refreshed is not None
    assert newer_refreshed is not None
    assert distinct_refreshed is not None
    older_item = older_refreshed["work_orders"][0]
    newer_item = newer_refreshed["work_orders"][0]
    distinct_item = distinct_refreshed["work_orders"][0]
    assert older_item["status"] == "discarded"
    assert older_item["metadata"]["archived_due_to"] == "duplicate_waiting_conflict"
    assert older_item["metadata"]["archive_reason"] == "duplicate_waiting_conflict"
    assert older_item["metadata"]["canonical_run_id"] == newer["run_id"]
    assert older_item["metadata"]["canonical_work_order_id"] == "subtask_2"
    assert newer_item["status"] == "waiting_conflict"
    assert distinct_item["status"] == "waiting_conflict"


def test_archive_duplicate_waiting_conflict_work_orders_discards_same_tranche_lane_with_goal_drift(
    store: DevCoordinationStore,
) -> None:
    older = store.create_supervisor_run(
        goal="Replace integrations UI path with truthful live-state behavior.",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={},
        spec={},
        work_orders=[
            {
                "work_order_id": "lane-old",
                "title": "Integrations slice",
                "status": "waiting_conflict",
                "file_scope": [
                    "aragora/live/**",
                    "aragora/server/handlers/features/**",
                    "tests/e2e/**",
                    "tests/handlers/**",
                    "docs/**",
                ],
                "metadata": {
                    "source": "explicit_spec_work_order",
                    "tranche_lane_id": "integrations-ui-truthful-status-slice",
                },
            }
        ],
    )
    newer = store.create_supervisor_run(
        goal=(
            "Replace one non-trustworthy integrations UI path with live-state behavior "
            "and truthful status messaging, preferring a status or edit flow that "
            "already has backend state available.\n\nVerification commands:\n"
            "- python3 -m pytest tests/handlers/features/test_integrations.py -q"
        ),
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={},
        spec={},
        work_orders=[
            {
                "work_order_id": "lane-new",
                "title": "Integrations slice",
                "status": "waiting_conflict",
                "file_scope": [
                    "aragora/live/**",
                    "aragora/server/handlers/features/**",
                    "tests/e2e/**",
                    "tests/handlers/**",
                    "docs/**",
                ],
                "metadata": {
                    "source": "explicit_spec_work_order",
                    "tranche_lane_id": "integrations-ui-truthful-status-slice",
                },
            }
        ],
    )

    conn = store._connect()
    try:
        conn.execute(
            "UPDATE supervisor_runs SET created_at = ?, updated_at = ? WHERE run_id = ?",
            ("2000-01-01T00:00:00+00:00", "2000-01-01T00:00:00+00:00", older["run_id"]),
        )
        conn.execute(
            "UPDATE supervisor_runs SET created_at = ?, updated_at = ? WHERE run_id = ?",
            ("2000-01-02T00:00:00+00:00", "2000-01-02T00:00:00+00:00", newer["run_id"]),
        )
        conn.commit()
    finally:
        conn.close()

    archived = store.archive_duplicate_waiting_conflict_work_orders()
    older_refreshed = store.get_supervisor_run(older["run_id"])
    newer_refreshed = store.get_supervisor_run(newer["run_id"])

    assert archived == 1
    assert older_refreshed is not None
    assert newer_refreshed is not None
    assert older_refreshed["work_orders"][0]["status"] == "discarded"
    assert older_refreshed["work_orders"][0]["metadata"]["archived_due_to"] == (
        "duplicate_waiting_conflict"
    )
    assert older_refreshed["work_orders"][0]["metadata"]["canonical_run_id"] == newer["run_id"]
    assert newer_refreshed["work_orders"][0]["status"] == "waiting_conflict"


def test_archive_duplicate_waiting_conflict_work_orders_discards_scope_less_duplicate(
    store: DevCoordinationStore,
) -> None:
    older = store.create_supervisor_run(
        goal="Write an ADR for standardizing health check endpoints across all deploy surfaces.",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={},
        spec={},
        work_orders=[
            {
                "work_order_id": "subtask_1",
                "title": "ADR lane",
                "status": "waiting_conflict",
            }
        ],
    )
    newer = store.create_supervisor_run(
        goal="Write an ADR for standardizing health check endpoints across all deploy surfaces.",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={},
        spec={},
        work_orders=[
            {
                "work_order_id": "subtask_2",
                "title": "ADR lane",
                "status": "waiting_conflict",
            }
        ],
    )

    conn = store._connect()
    try:
        conn.execute(
            "UPDATE supervisor_runs SET created_at = ?, updated_at = ? WHERE run_id = ?",
            ("2000-01-01T00:00:00+00:00", "2000-01-01T00:00:00+00:00", older["run_id"]),
        )
        conn.execute(
            "UPDATE supervisor_runs SET created_at = ?, updated_at = ? WHERE run_id = ?",
            ("2000-01-02T00:00:00+00:00", "2000-01-02T00:00:00+00:00", newer["run_id"]),
        )
        conn.commit()
    finally:
        conn.close()

    archived = store.archive_duplicate_waiting_conflict_work_orders()
    older_refreshed = store.get_supervisor_run(older["run_id"])
    newer_refreshed = store.get_supervisor_run(newer["run_id"])

    assert archived == 1
    assert older_refreshed is not None
    assert newer_refreshed is not None
    assert older_refreshed["work_orders"][0]["status"] == "discarded"
    assert older_refreshed["work_orders"][0]["metadata"]["archived_due_to"] == (
        "duplicate_waiting_conflict"
    )
    assert older_refreshed["work_orders"][0]["metadata"]["canonical_run_id"] == newer["run_id"]
    assert newer_refreshed["work_orders"][0]["status"] == "waiting_conflict"


def test_archive_duplicate_waiting_conflict_work_orders_discards_scope_less_same_tranche_lane(
    store: DevCoordinationStore,
) -> None:
    older = store.create_supervisor_run(
        goal="Connect the results page to backend endpoints.",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={},
        spec={},
        work_orders=[
            {
                "work_order_id": "proj-001",
                "title": "Results page lane",
                "status": "waiting_conflict",
                "metadata": {
                    "source": "explicit_spec_work_order",
                    "tranche_lane_id": "proj-001",
                },
            }
        ],
    )
    newer = store.create_supervisor_run(
        goal="Rebuild the 30/60/90 execution map from current repo reality.",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={},
        spec={},
        work_orders=[
            {
                "work_order_id": "proj-001",
                "title": "Results page lane",
                "status": "waiting_conflict",
                "metadata": {
                    "source": "explicit_spec_work_order",
                    "tranche_lane_id": "proj-001",
                },
            }
        ],
    )

    conn = store._connect()
    try:
        conn.execute(
            "UPDATE supervisor_runs SET created_at = ?, updated_at = ? WHERE run_id = ?",
            ("2000-01-01T00:00:00+00:00", "2000-01-01T00:00:00+00:00", older["run_id"]),
        )
        conn.execute(
            "UPDATE supervisor_runs SET created_at = ?, updated_at = ? WHERE run_id = ?",
            ("2000-01-02T00:00:00+00:00", "2000-01-02T00:00:00+00:00", newer["run_id"]),
        )
        conn.commit()
    finally:
        conn.close()

    archived = store.archive_duplicate_waiting_conflict_work_orders()
    older_refreshed = store.get_supervisor_run(older["run_id"])
    newer_refreshed = store.get_supervisor_run(newer["run_id"])

    assert archived == 1
    assert older_refreshed is not None
    assert newer_refreshed is not None
    assert older_refreshed["work_orders"][0]["status"] == "discarded"
    assert older_refreshed["work_orders"][0]["metadata"]["archived_due_to"] == (
        "duplicate_waiting_conflict"
    )
    assert older_refreshed["work_orders"][0]["metadata"]["canonical_run_id"] == newer["run_id"]
    assert newer_refreshed["work_orders"][0]["status"] == "waiting_conflict"


def test_archive_work_order_leasing_failed_work_orders_preserves_deliverable_backed_lane(
    store: DevCoordinationStore,
) -> None:
    run = store.create_supervisor_run(
        goal="Preserve leasing failure with deliverable",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={
            "require_merge_approval": True,
            "require_external_action_approval": True,
        },
        spec={
            "raw_goal": "Preserve leasing failure with deliverable",
            "refined_goal": "Preserve leasing failure with deliverable",
        },
        work_orders=[
            {
                "work_order_id": "subtask_1",
                "title": "Deliverable-backed leasing failure",
                "status": "needs_human",
                "failure_reason": "work_order_leasing_failed",
                "dispatch_error": "autopilot ensure failed (1): branch already exists",
                "branch": "codex/subtask_1",
                "commit_shas": ["deadbeef"],
                "file_scope": ["aragora/cli/commands/quickstart.py"],
            }
        ],
    )

    conn = store._connect()
    try:
        conn.execute(
            "UPDATE supervisor_runs SET created_at = ?, updated_at = ? WHERE run_id = ?",
            ("2000-01-01T00:00:00+00:00", "2000-01-01T00:00:00+00:00", run["run_id"]),
        )
        conn.commit()
    finally:
        conn.close()

    archived = store.archive_work_order_leasing_failed_work_orders(grace_period_hours=24.0)
    refreshed = store.get_supervisor_run(run["run_id"])

    assert archived == 0
    assert refreshed is not None
    assert refreshed["work_orders"][0]["status"] == "needs_human"


def test_archive_work_order_leasing_failed_work_orders_uses_created_at_before_run_updated_at(
    store: DevCoordinationStore,
) -> None:
    run = store.create_supervisor_run(
        goal="Archive stale leasing failure despite refreshed run timestamp",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={},
        spec={},
        work_orders=[
            {
                "work_order_id": "subtask_1",
                "title": "Stale leasing failure with refreshed run",
                "status": "needs_human",
                "failure_reason": "work_order_leasing_failed",
                "dispatch_error": "autopilot ensure failed (1): branch already exists",
                "file_scope": ["aragora/cli/commands/quickstart.py"],
            }
        ],
    )

    conn = store._connect()
    try:
        conn.execute(
            "UPDATE supervisor_runs SET created_at = ?, updated_at = ? WHERE run_id = ?",
            ("2000-01-01T00:00:00+00:00", "2999-01-01T00:00:00+00:00", run["run_id"]),
        )
        conn.commit()
    finally:
        conn.close()

    archived = store.archive_work_order_leasing_failed_work_orders(grace_period_hours=24.0)
    refreshed = store.get_supervisor_run(run["run_id"])

    assert archived == 1
    assert refreshed is not None
    assert refreshed["work_orders"][0]["status"] == "discarded"


def test_archive_worker_type_blocked_work_orders_discards_old_no_deliverable_lane(
    store: DevCoordinationStore,
) -> None:
    run = store.create_supervisor_run(
        goal="Archive stale worker-type blocked lane",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={},
        spec={},
        work_orders=[
            {
                "work_order_id": "subtask_1",
                "title": "Stale worker-type blocked lane",
                "status": "needs_human",
                "failure_reason": "worker_type_blocked",
                "dispatch_error": "worker dispatch blocked: claude breaker open until 2000-01-01T00:00:00+00:00 after agent_capacity",
                "file_scope": ["aragora/live/src/**"],
            }
        ],
    )

    conn = store._connect()
    try:
        conn.execute(
            "UPDATE supervisor_runs SET created_at = ?, updated_at = ? WHERE run_id = ?",
            ("2000-01-01T00:00:00+00:00", "2999-01-01T00:00:00+00:00", run["run_id"]),
        )
        conn.commit()
    finally:
        conn.close()

    archived = store.archive_worker_type_blocked_work_orders(grace_period_hours=24.0)
    refreshed = store.get_supervisor_run(run["run_id"])

    assert archived == 1
    assert refreshed is not None
    assert refreshed["work_orders"][0]["status"] == "discarded"


def test_archive_worker_type_blocked_work_orders_preserves_deliverable_backed_lane(
    store: DevCoordinationStore,
) -> None:
    run = store.create_supervisor_run(
        goal="Preserve worker-type blocked lane with deliverable",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={},
        spec={},
        work_orders=[
            {
                "work_order_id": "subtask_1",
                "title": "Deliverable-backed worker-type blocked lane",
                "status": "needs_human",
                "failure_reason": "worker_type_blocked",
                "dispatch_error": "worker dispatch blocked: claude breaker open until 2000-01-01T00:00:00+00:00 after agent_capacity",
                "branch": "codex/subtask_1",
                "commit_shas": ["deadbeef"],
                "file_scope": ["aragora/live/src/**"],
            }
        ],
    )

    conn = store._connect()
    try:
        conn.execute(
            "UPDATE supervisor_runs SET created_at = ?, updated_at = ? WHERE run_id = ?",
            ("2000-01-01T00:00:00+00:00", "2999-01-01T00:00:00+00:00", run["run_id"]),
        )
        conn.commit()
    finally:
        conn.close()

    archived = store.archive_worker_type_blocked_work_orders(grace_period_hours=24.0)
    refreshed = store.get_supervisor_run(run["run_id"])

    assert archived == 0
    assert refreshed is not None
    assert refreshed["work_orders"][0]["status"] == "needs_human"


def test_archive_duplicate_work_order_leasing_failed_work_orders_discards_older_duplicate_runs(
    store: DevCoordinationStore,
) -> None:
    older = store.create_supervisor_run(
        goal="Add --json output flag to aragora quickstart CLI",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={
            "require_merge_approval": True,
            "require_external_action_approval": True,
        },
        spec={
            "raw_goal": "Add --json output flag to aragora quickstart CLI",
            "refined_goal": "Add --json output flag to aragora quickstart CLI",
        },
        work_orders=[
            {
                "work_order_id": "subtask_1",
                "title": "Older duplicate leasing failure",
                "status": "needs_human",
                "failure_reason": "work_order_leasing_failed",
                "dispatch_error": "autopilot ensure failed (1): branch already exists",
                "file_scope": [
                    "aragora/cli/commands/quickstart.py",
                    "aragora/cli/parser.py",
                    "tests/cli/test_quickstart.py",
                ],
            }
        ],
    )
    newer = store.create_supervisor_run(
        goal="Add --json output flag to aragora quickstart CLI",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={
            "require_merge_approval": True,
            "require_external_action_approval": True,
        },
        spec={
            "raw_goal": "Add --json output flag to aragora quickstart CLI",
            "refined_goal": "Add --json output flag to aragora quickstart CLI",
        },
        work_orders=[
            {
                "work_order_id": "subtask_9",
                "title": "Newer duplicate leasing failure",
                "status": "needs_human",
                "failure_reason": "work_order_leasing_failed",
                "dispatch_error": "autopilot ensure failed (1): branch already exists",
                "file_scope": [
                    "aragora/cli/commands/quickstart.py",
                    "aragora/cli/parser.py",
                    "tests/cli/test_quickstart.py",
                ],
            }
        ],
    )
    distinct = store.create_supervisor_run(
        goal="Add JSONL metrics logging per boss loop iteration",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={
            "require_merge_approval": True,
            "require_external_action_approval": True,
        },
        spec={
            "raw_goal": "Add JSONL metrics logging per boss loop iteration",
            "refined_goal": "Add JSONL metrics logging per boss loop iteration",
        },
        work_orders=[
            {
                "work_order_id": "subtask_2",
                "title": "Distinct leasing failure",
                "status": "needs_human",
                "failure_reason": "work_order_leasing_failed",
                "dispatch_error": "autopilot ensure failed (1): branch already exists",
                "file_scope": [
                    "aragora/cli/commands/quickstart.py",
                    "aragora/cli/parser.py",
                    "tests/cli/test_quickstart.py",
                ],
            }
        ],
    )

    conn = store._connect()
    try:
        conn.execute(
            "UPDATE supervisor_runs SET created_at = ?, updated_at = ? WHERE run_id = ?",
            ("2000-01-01T00:00:00+00:00", "2000-01-01T00:00:00+00:00", older["run_id"]),
        )
        conn.execute(
            "UPDATE supervisor_runs SET created_at = ?, updated_at = ? WHERE run_id = ?",
            ("2000-01-02T00:00:00+00:00", "2000-01-02T00:00:00+00:00", newer["run_id"]),
        )
        conn.execute(
            "UPDATE supervisor_runs SET created_at = ?, updated_at = ? WHERE run_id = ?",
            ("2000-01-03T00:00:00+00:00", "2000-01-03T00:00:00+00:00", distinct["run_id"]),
        )
        conn.commit()
    finally:
        conn.close()

    archived = store.archive_duplicate_work_order_leasing_failed_work_orders()
    older_refreshed = store.get_supervisor_run(older["run_id"])
    newer_refreshed = store.get_supervisor_run(newer["run_id"])
    distinct_refreshed = store.get_supervisor_run(distinct["run_id"])

    assert archived == 1
    assert older_refreshed is not None
    assert newer_refreshed is not None
    assert distinct_refreshed is not None
    assert older_refreshed["work_orders"][0]["status"] == "discarded"
    assert (
        older_refreshed["work_orders"][0]["metadata"]["archive_reason"]
        == "duplicate_work_order_leasing_failed"
    )
    assert older_refreshed["work_orders"][0]["metadata"]["canonical_run_id"] == newer["run_id"]
    assert newer_refreshed["work_orders"][0]["status"] == "needs_human"
    assert distinct_refreshed["work_orders"][0]["status"] == "needs_human"


def test_archive_superseded_clean_exit_no_deliverable_work_orders_discards_helper_lane(
    store: DevCoordinationStore,
) -> None:
    run = store.create_supervisor_run(
        goal="Archive helper clean-exit lane when deliverable sibling exists",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={},
        spec={},
        work_orders=[
            {
                "work_order_id": "subtask_1",
                "title": "Implementation lane",
                "status": "completed",
                "receipt_id": "receipt-impl",
                "branch": "codex/subtask_1",
                "commit_shas": ["abc12345"],
                "file_scope": [
                    "aragora/swarm/outcome_signals.py",
                    "tests/swarm/test_outcome_signals.py",
                ],
            },
            {
                "work_order_id": "subtask_2",
                "title": "Validation helper lane",
                "status": "needs_human",
                "failure_reason": "clean_exit_no_deliverable",
                "worker_outcome": "clean_exit_no_effect",
                "dispatch_error": "worker exited 0 with no commits and no changed paths",
                "file_scope": [
                    "aragora/swarm/outcome_signals.py",
                    "tests/swarm/test_outcome_signals.py",
                ],
            },
        ],
    )

    archived = store.archive_superseded_clean_exit_no_deliverable_work_orders()
    refreshed = store.get_supervisor_run(run["run_id"])

    assert archived == 1
    assert refreshed is not None
    helper = refreshed["work_orders"][1]
    assert helper["status"] == "discarded"
    assert helper["metadata"]["archive_reason"] == "superseded_clean_exit_no_deliverable"
    assert helper["metadata"]["canonical_work_order_id"] == "subtask_1"


def test_archive_superseded_clean_exit_no_deliverable_work_orders_preserves_standalone_lane(
    store: DevCoordinationStore,
) -> None:
    run = store.create_supervisor_run(
        goal="Preserve standalone clean-exit lane",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={},
        spec={},
        work_orders=[
            {
                "work_order_id": "subtask_1",
                "title": "Standalone no-op lane",
                "status": "needs_human",
                "failure_reason": "clean_exit_no_deliverable",
                "worker_outcome": "clean_exit_no_effect",
                "dispatch_error": "worker exited 0 with no commits and no changed paths",
                "file_scope": [
                    "aragora/cli/commands/receipt.py",
                    "tests/cli/test_receipt_command.py",
                ],
            },
            {
                "work_order_id": "subtask_2",
                "title": "Queued sibling without deliverable",
                "status": "queued",
                "file_scope": [
                    "aragora/cli/commands/receipt.py",
                    "tests/cli/test_receipt_command.py",
                ],
            },
        ],
    )

    archived = store.archive_superseded_clean_exit_no_deliverable_work_orders()
    refreshed = store.get_supervisor_run(run["run_id"])

    assert archived == 0
    assert refreshed is not None
    assert refreshed["work_orders"][0]["status"] == "needs_human"


def test_archive_superseded_clean_exit_no_deliverable_work_orders_discards_helper_lane_with_open_sibling(
    store: DevCoordinationStore,
) -> None:
    run = store.create_supervisor_run(
        goal="Archive helper clean-exit lane with queued sibling",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={},
        spec={},
        work_orders=[
            {
                "work_order_id": "subtask_1",
                "title": "Read existing receipt.py code and understand current CLI structure",
                "status": "needs_human",
                "failure_reason": "clean_exit_no_deliverable",
                "worker_outcome": "clean_exit_no_effect",
                "dispatch_error": "worker exited 0 with no commits and no changed paths",
                "file_scope": [
                    "aragora/cli/commands/receipt.py",
                    "tests/cli/test_receipt_command.py",
                ],
            },
            {
                "work_order_id": "subtask_2",
                "title": "Write tests for --format markdown functionality",
                "status": "queued",
                "file_scope": [
                    "tests/cli/test_receipt_command.py",
                    "aragora/cli/commands/receipt.py",
                ],
            },
        ],
    )

    archived = store.archive_superseded_clean_exit_no_deliverable_work_orders()
    refreshed = store.get_supervisor_run(run["run_id"])

    assert archived == 1
    assert refreshed is not None
    helper = refreshed["work_orders"][0]
    assert helper["status"] == "discarded"
    assert helper["metadata"]["archive_reason"] == "helper_clean_exit_no_deliverable"
    assert helper["metadata"]["canonical_work_order_id"] == "subtask_2"


def test_archive_superseded_stale_lease_reaped_work_orders_discards_helper_lane_with_open_sibling(
    store: DevCoordinationStore,
) -> None:
    run = store.create_supervisor_run(
        goal="Archive helper stale-lease lane with queued sibling",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={},
        spec={},
        work_orders=[
            {
                "work_order_id": "subtask_1",
                "title": "Read existing receipt.py command structure and add --format flag",
                "status": "needs_human",
                "failure_reason": "stale_lease_reaped",
                "file_scope": [
                    "aragora/cli/commands/receipt.py",
                    "tests/cli/test_receipt_command.py",
                ],
            },
            {
                "work_order_id": "subtask_2",
                "title": "Add test cases for --format markdown functionality",
                "status": "queued",
                "file_scope": [
                    "tests/cli/test_receipt_command.py",
                    "aragora/cli/commands/receipt.py",
                ],
            },
        ],
    )

    archived = store.archive_superseded_stale_lease_reaped_work_orders()
    refreshed = store.get_supervisor_run(run["run_id"])

    assert archived == 1
    assert refreshed is not None
    helper = refreshed["work_orders"][0]
    assert helper["status"] == "discarded"
    assert helper["metadata"]["archive_reason"] == "helper_stale_lease_reaped"
    assert helper["metadata"]["canonical_work_order_id"] == "subtask_2"


def test_backfill_missing_blocker_metadata_infers_missing_verification_plan(
    store: DevCoordinationStore,
) -> None:
    run = store.create_supervisor_run(
        goal="Backfill merge gate blocker metadata",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={
            "require_merge_approval": True,
            "require_external_action_approval": True,
        },
        spec={
            "raw_goal": "Backfill merge gate blocker metadata",
            "refined_goal": "Backfill merge gate blocker metadata",
        },
        work_orders=[
            {
                "work_order_id": "subtask_1",
                "title": "Historical merge gate lane",
                "status": "needs_human",
                "worker_outcome": "merge_gate_failed",
                "dispatch_error": "merge gate blocked: missing verification plan for code-change lane",
            }
        ],
    )

    updated = store.backfill_missing_blocker_metadata()
    refreshed = store.get_supervisor_run(run["run_id"])

    assert updated == 1
    assert refreshed is not None
    work_order = refreshed["work_orders"][0]
    assert work_order["failure_reason"] == "missing_verification_plan"
    assert "verification command" in work_order["blocking_question"]
    assert work_order["blocker"]["reason"] == "missing_verification_plan"


def test_backfill_missing_blocker_metadata_infers_clean_exit_no_deliverable(
    store: DevCoordinationStore,
) -> None:
    run = store.create_supervisor_run(
        goal="Backfill clean exit blocker metadata",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={
            "require_merge_approval": True,
            "require_external_action_approval": True,
        },
        spec={
            "raw_goal": "Backfill clean exit blocker metadata",
            "refined_goal": "Backfill clean exit blocker metadata",
        },
        work_orders=[
            {
                "work_order_id": "subtask_1",
                "title": "Historical clean exit lane",
                "status": "needs_human",
                "worker_outcome": "clean_exit_no_effect",
                "dispatch_error": "worker exited 0 with no commits and no changed paths",
            }
        ],
    )

    updated = store.backfill_missing_blocker_metadata()
    refreshed = store.get_supervisor_run(run["run_id"])

    assert updated == 1
    assert refreshed is not None
    work_order = refreshed["work_orders"][0]
    assert work_order["failure_reason"] == "clean_exit_no_deliverable"
    assert "concrete branch, commit, or PR" in work_order["blocking_question"]
    assert work_order["blocker"]["reason"] == "clean_exit_no_deliverable"


def test_backfill_missing_blocker_metadata_infers_waiting_conflict(
    store: DevCoordinationStore,
) -> None:
    run = store.create_supervisor_run(
        goal="Backfill waiting conflict blocker metadata",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={
            "require_merge_approval": True,
            "require_external_action_approval": True,
        },
        spec={
            "raw_goal": "Backfill waiting conflict blocker metadata",
            "refined_goal": "Backfill waiting conflict blocker metadata",
        },
        work_orders=[
            {
                "work_order_id": "subtask_1",
                "title": "Historical waiting conflict lane",
                "status": "waiting_conflict",
                "file_scope": ["docs/strategy/**"],
            }
        ],
    )

    updated = store.backfill_missing_blocker_metadata()
    refreshed = store.get_supervisor_run(run["run_id"])

    assert updated == 1
    assert refreshed is not None
    work_order = refreshed["work_orders"][0]
    assert work_order["failure_reason"] == "waiting_conflict"
    assert "overlapping lane" in work_order["blocking_question"]
    assert work_order["blocker"]["reason"] == "waiting_conflict"


def test_backfill_missing_blocker_metadata_preserves_blocked_deliverable_lane(
    store: DevCoordinationStore,
) -> None:
    run = store.create_supervisor_run(
        goal="Backfill blocked deliverable metadata",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={
            "require_merge_approval": True,
            "require_external_action_approval": True,
        },
        spec={
            "raw_goal": "Backfill blocked deliverable metadata",
            "refined_goal": "Backfill blocked deliverable metadata",
        },
        work_orders=[
            {
                "work_order_id": "subtask_1",
                "title": "Blocked deliverable lane",
                "status": "needs_human",
                "worker_outcome": "merge_gate_failed",
                "dispatch_error": "merge gate blocked: missing verification plan for code-change lane",
                "branch": "codex/blocked-deliverable",
                "commit_shas": ["abc12345"],
            }
        ],
    )

    updated = store.backfill_missing_blocker_metadata()
    refreshed = store.get_supervisor_run(run["run_id"])

    assert updated == 1
    assert refreshed is not None
    work_order = refreshed["work_orders"][0]
    assert work_order["failure_reason"] == "missing_verification_plan"
    assert work_order["branch"] == "codex/blocked-deliverable"
    assert work_order["commit_shas"] == ["abc12345"]


def test_backfill_missing_blocker_metadata_reclassifies_missing_verification_target(
    store: DevCoordinationStore,
) -> None:
    run = store.create_supervisor_run(
        goal="Backfill missing verification target metadata",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={
            "require_merge_approval": True,
            "require_external_action_approval": True,
        },
        spec={
            "raw_goal": "Backfill missing verification target metadata",
            "refined_goal": "Backfill missing verification target metadata",
        },
        work_orders=[
            {
                "work_order_id": "subtask_1",
                "title": "Historical missing target lane",
                "status": "needs_human",
                "worker_outcome": "merge_gate_failed",
                "failure_reason": "merge_gate_failed",
                "dispatch_error": (
                    "merge gate blocked: verification failed: python -m pytest "
                    "tests/orchestrator/test_budget_gate.py -q (exit 4) - "
                    "ERROR: file or directory not found: tests/orchestrator/test_budget_gate.py"
                ),
                "verification_results": [
                    {
                        "command": "python -m pytest tests/orchestrator/test_budget_gate.py -q",
                        "exit_code": 4,
                        "passed": False,
                        "stdout": "\\nno tests ran in 0.00s\\n",
                        "stderr": (
                            "ERROR: file or directory not found: "
                            "tests/orchestrator/test_budget_gate.py\\n\\n"
                        ),
                        "duration_seconds": 0.123,
                    }
                ],
                "branch": "codex/historical-budget-gate",
                "commit_shas": ["deadbeef"],
            }
        ],
    )

    updated = store.backfill_missing_blocker_metadata()
    refreshed = store.get_supervisor_run(run["run_id"])

    assert updated == 1
    assert refreshed is not None
    work_order = refreshed["work_orders"][0]
    assert work_order["failure_reason"] == "verification_target_missing"
    assert "replace the missing path" in work_order["blocking_question"]
    assert work_order["blocker"]["reason"] == "verification_target_missing"
    assert work_order["branch"] == "codex/historical-budget-gate"
    assert work_order["commit_shas"] == ["deadbeef"]


def test_backfill_missing_verification_plans_infers_test_from_scope(
    store: DevCoordinationStore,
) -> None:
    run = store.create_supervisor_run(
        goal="Backfill missing verification command",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={
            "require_merge_approval": True,
            "require_external_action_approval": True,
        },
        spec={
            "raw_goal": "Backfill missing verification command",
            "refined_goal": "Backfill missing verification command",
        },
        work_orders=[
            {
                "work_order_id": "subtask_1",
                "title": "Historical merge gate lane",
                "status": "needs_human",
                "worker_outcome": "merge_gate_failed",
                "failure_reason": "missing_verification_plan",
                "dispatch_error": "merge gate blocked: missing verification plan for code-change lane",
                "file_scope": [
                    "aragora/swarm/supervisor.py",
                    "tests/swarm/test_supervisor.py",
                ],
                "metadata": {
                    "acceptance_criteria": [
                        "Behavior is covered by automated tests for supervisor flows",
                    ]
                },
            }
        ],
    )

    updated = store.backfill_missing_verification_plans()
    refreshed = store.get_supervisor_run(run["run_id"])

    assert updated == 1
    assert refreshed is not None
    work_order = refreshed["work_orders"][0]
    assert work_order["expected_tests"] == ["python -m pytest tests/swarm/test_supervisor.py -q"]
    assert work_order["success_criteria"]["tests"] == (
        "python -m pytest tests/swarm/test_supervisor.py -q"
    )
    assert work_order["failure_reason"] == "merge_gate_failed"
    assert work_order["worker_outcome"] == "merge_gate_failed"
    assert work_order["merge_gate"]["verification_missing_reason"] is None
    assert work_order["merge_gate"]["expected_checks"] == [
        "python -m pytest tests/swarm/test_supervisor.py -q"
    ]
    assert work_order["blocking_question"] == (
        "Which required verification or acceptance check must pass before approval?"
    )
    assert work_order["blocker"]["reason"] == "merge_gate_failed"
    assert work_order["dispatch_error"].startswith(
        "merge gate blocked: required verification did not run"
    )


def test_backfill_missing_verification_plans_infers_test_from_changed_paths(
    store: DevCoordinationStore,
) -> None:
    run = store.create_supervisor_run(
        goal="Backfill verification plan from changed test paths",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={
            "require_merge_approval": True,
            "require_external_action_approval": True,
        },
        spec={
            "raw_goal": "Backfill verification plan from changed test paths",
            "refined_goal": "Backfill verification plan from changed test paths",
        },
        work_orders=[
            {
                "work_order_id": "subtask_1",
                "title": "Glob-scoped merge gate lane",
                "status": "needs_human",
                "worker_outcome": "merge_gate_failed",
                "failure_reason": "missing_verification_plan",
                "dispatch_error": "merge gate blocked: missing verification plan for code-change lane",
                "file_scope": ["aragora/cli/**", "tests/cli/**", "docs/**"],
                "changed_paths": [
                    "aragora/cli/commands/quickstart.py",
                    "tests/cli/test_quickstart.py",
                ],
            }
        ],
    )

    updated = store.backfill_missing_verification_plans()
    refreshed = store.get_supervisor_run(run["run_id"])

    assert updated == 1
    assert refreshed is not None
    work_order = refreshed["work_orders"][0]
    assert work_order["expected_tests"] == ["python -m pytest tests/cli/test_quickstart.py -q"]
    assert work_order["failure_reason"] == "merge_gate_failed"
    assert work_order["merge_gate"]["expected_checks"] == [
        "python -m pytest tests/cli/test_quickstart.py -q"
    ]


def test_backfill_missing_verification_plans_infers_test_from_acceptance_text(
    store: DevCoordinationStore,
) -> None:
    run = store.create_supervisor_run(
        goal="Backfill verification plan from acceptance text",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={
            "require_merge_approval": True,
            "require_external_action_approval": True,
        },
        spec={
            "raw_goal": "Backfill verification plan from acceptance text",
            "refined_goal": "Backfill verification plan from acceptance text",
        },
        work_orders=[
            {
                "work_order_id": "subtask_1",
                "title": "Acceptance-driven merge gate lane",
                "status": "needs_human",
                "worker_outcome": "merge_gate_failed",
                "failure_reason": "missing_verification_plan",
                "dispatch_error": "merge gate blocked: missing verification plan for code-change lane",
                "file_scope": ["aragora/swarm/campaign.py"],
                "changed_paths": ["aragora/swarm/campaign.py"],
                "metadata": {
                    "acceptance_criteria": [
                        "_build_prompt includes actual diff content from git diff",
                        "Tests pass: tests/swarm/test_campaign_reviewer_diff.py, tests/swarm/test_campaign.py",
                    ]
                },
            }
        ],
    )

    updated = store.backfill_missing_verification_plans()
    refreshed = store.get_supervisor_run(run["run_id"])

    assert updated == 1
    assert refreshed is not None
    work_order = refreshed["work_orders"][0]
    assert work_order["expected_tests"] == [
        "python -m pytest tests/swarm/test_campaign_reviewer_diff.py -q",
        "python -m pytest tests/swarm/test_campaign.py -q",
    ]
    assert work_order["failure_reason"] == "merge_gate_failed"
    assert work_order["merge_gate"]["expected_checks"] == [
        "python -m pytest tests/swarm/test_campaign_reviewer_diff.py -q",
        "python -m pytest tests/swarm/test_campaign.py -q",
    ]


def test_backfill_missing_verification_plans_skips_uninferable_lane(
    store: DevCoordinationStore,
) -> None:
    run = store.create_supervisor_run(
        goal="Preserve missing verification plan when no test is inferable",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={
            "require_merge_approval": True,
            "require_external_action_approval": True,
        },
        spec={
            "raw_goal": "Preserve missing verification plan when no test is inferable",
            "refined_goal": "Preserve missing verification plan when no test is inferable",
        },
        work_orders=[
            {
                "work_order_id": "subtask_1",
                "title": "Uninferable merge gate lane",
                "status": "needs_human",
                "worker_outcome": "merge_gate_failed",
                "failure_reason": "missing_verification_plan",
                "dispatch_error": "merge gate blocked: missing verification plan for code-change lane",
                "file_scope": ["aragora/swarm/supervisor.py"],
                "metadata": {
                    "acceptance_criteria": [
                        "Supervisor records breaker state and avoids dispatching blocked worker types"
                    ]
                },
            }
        ],
    )

    updated = store.backfill_missing_verification_plans()
    refreshed = store.get_supervisor_run(run["run_id"])

    assert updated == 0
    assert refreshed is not None
    work_order = refreshed["work_orders"][0]
    assert work_order.get("expected_tests", []) == []
    assert work_order["failure_reason"] == "missing_verification_plan"
    assert work_order["dispatch_error"] == (
        "merge gate blocked: missing verification plan for code-change lane"
    )


def test_rehabilitate_docs_only_missing_verification_plan_work_orders(
    store: DevCoordinationStore,
) -> None:
    run = store.create_supervisor_run(
        goal="Rehabilitate docs-only lane",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={
            "require_merge_approval": True,
            "require_external_action_approval": True,
        },
        spec={
            "raw_goal": "Rehabilitate docs-only lane",
            "refined_goal": "Rehabilitate docs-only lane",
        },
        work_orders=[
            {
                "work_order_id": "subtask_1",
                "title": "Docs-only merge gate lane",
                "status": "needs_human",
                "review_status": "changes_requested",
                "worker_outcome": "merge_gate_failed",
                "failure_reason": "missing_verification_plan",
                "dispatch_error": "merge gate blocked: missing verification plan for code-change lane",
                "file_scope": ["docs/**"],
                "changed_paths": ["docs/status/DESIGN_PARTNER_PROGRAM.md"],
                "branch": "codex/docs-lane",
                "commit_shas": ["abc12345"],
                "receipt_id": "rcpt-docs-lane",
                "merge_gate": {
                    "enabled": True,
                    "expected_checks": [],
                    "verification_results": [],
                    "verification_missing_reason": "missing_verification_plan",
                    "checks_passed": False,
                    "human_approval_required": True,
                    "merge_eligible": False,
                    "blocked_reasons": [
                        "merge gate blocked: missing verification plan for code-change lane"
                    ],
                },
            }
        ],
    )

    updated = store.rehabilitate_docs_only_missing_verification_plan_work_orders()
    refreshed = store.get_supervisor_run(run["run_id"])

    assert updated == 1
    assert refreshed is not None
    work_order = refreshed["work_orders"][0]
    assert work_order["status"] == "completed"
    assert work_order["review_status"] == "pending_heterogeneous_review"
    assert work_order["worker_outcome"] == "completed"
    assert work_order["merge_gate"]["checks_passed"] is True
    assert work_order["merge_gate"]["verification_missing_reason"] is None
    assert work_order["blockers"] == []
    assert "failure_reason" not in work_order
    assert "dispatch_error" not in work_order


def test_rehabilitate_deliverable_backed_clean_exit_no_deliverable_work_orders(
    store: DevCoordinationStore,
) -> None:
    run = store.create_supervisor_run(
        goal="Rehabilitate contradictory clean-exit lane",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={
            "require_merge_approval": True,
            "require_external_action_approval": True,
        },
        spec={
            "raw_goal": "Rehabilitate contradictory clean-exit lane",
            "refined_goal": "Rehabilitate contradictory clean-exit lane",
        },
        work_orders=[
            {
                "work_order_id": "subtask_1",
                "title": "Contradictory clean-exit lane",
                "status": "needs_human",
                "review_status": "changes_requested",
                "worker_outcome": "clean_exit_no_effect",
                "failure_reason": "clean_exit_no_deliverable",
                "dispatch_error": "worker produced only session artifacts, no real deliverables",
                "file_scope": ["aragora/ralph/classifier.py"],
                "branch": "codex/subtask_1",
                "commit_shas": ["abc12345"],
                "receipt_id": "rcpt-clean-exit",
            }
        ],
    )

    updated = store.rehabilitate_deliverable_backed_clean_exit_no_deliverable_work_orders()
    refreshed = store.get_supervisor_run(run["run_id"])

    assert updated == 1
    assert refreshed is not None
    work_order = refreshed["work_orders"][0]
    assert work_order["status"] == "completed"
    assert work_order["review_status"] == "pending_heterogeneous_review"
    assert work_order["worker_outcome"] == "completed"
    assert work_order["blockers"] == []
    assert "failure_reason" not in work_order
    assert "dispatch_error" not in work_order


def test_rehabilitate_deliverable_backed_clean_exit_no_deliverable_work_orders_preserves_receiptless_lane(
    store: DevCoordinationStore,
) -> None:
    run = store.create_supervisor_run(
        goal="Preserve true clean-exit lane",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={
            "require_merge_approval": True,
            "require_external_action_approval": True,
        },
        spec={
            "raw_goal": "Preserve true clean-exit lane",
            "refined_goal": "Preserve true clean-exit lane",
        },
        work_orders=[
            {
                "work_order_id": "subtask_1",
                "title": "Receiptless clean-exit lane",
                "status": "needs_human",
                "worker_outcome": "clean_exit_no_effect",
                "failure_reason": "clean_exit_no_deliverable",
                "dispatch_error": "worker exited 0 with no commits and no changed paths",
                "file_scope": ["aragora/ralph/classifier.py"],
            }
        ],
    )

    updated = store.rehabilitate_deliverable_backed_clean_exit_no_deliverable_work_orders()
    refreshed = store.get_supervisor_run(run["run_id"])

    assert updated == 0
    assert refreshed is not None
    assert refreshed["work_orders"][0]["status"] == "needs_human"


def test_reclassify_branch_snapshot_stale_review_work_orders(
    store: DevCoordinationStore,
) -> None:
    run = store.create_supervisor_run(
        goal="Move branch-stale lane into review bucket",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={},
        spec={},
        work_orders=[
            {
                "work_order_id": "subtask_1",
                "title": "Branch-stale lane",
                "status": "needs_human",
                "review_status": "changes_requested",
                "worker_outcome": "branch_snapshot_stale",
                "failure_reason": "branch_snapshot_stale",
                "receipt_id": "receipt-branch-stale",
                "branch": "codex/branch-stale",
                "commit_shas": ["abc12345"],
                "metadata": {"mainline_verification_passed": True},
            }
        ],
    )

    updated = store.reclassify_branch_snapshot_stale_review_work_orders()
    refreshed = store.get_supervisor_run(run["run_id"])

    assert updated == 1
    assert refreshed is not None
    assert refreshed["work_orders"][0]["status"] == "changes_requested"
    assert refreshed["work_orders"][0]["review_status"] == "changes_requested"


def test_reclassify_deliverable_changes_requested_work_orders(
    store: DevCoordinationStore,
) -> None:
    run = store.create_supervisor_run(
        goal="Move deliverable changes-requested lane into review bucket",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={},
        spec={},
        work_orders=[
            {
                "work_order_id": "subtask_1",
                "title": "Verification target missing lane",
                "status": "needs_human",
                "review_status": "changes_requested",
                "worker_outcome": "merge_gate_failed",
                "failure_reason": "verification_target_missing",
                "receipt_id": "receipt-verification-target-missing",
                "branch": "codex/subtask_1",
                "commit_shas": ["abc12345"],
            }
        ],
    )

    updated = store.reclassify_deliverable_changes_requested_work_orders()
    refreshed = store.get_supervisor_run(run["run_id"])

    assert updated == 1
    assert refreshed is not None
    assert refreshed["work_orders"][0]["status"] == "changes_requested"
    assert refreshed["work_orders"][0]["review_status"] == "changes_requested"


def test_backfill_missing_completion_receipts_for_historical_deliverable(
    repo: Path, store: DevCoordinationStore
) -> None:
    run = store.create_supervisor_run(
        goal="Backfill historical deliverable receipt",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={
            "require_merge_approval": True,
            "require_external_action_approval": True,
        },
        spec={
            "raw_goal": "Backfill historical deliverable receipt",
            "refined_goal": "Backfill historical deliverable receipt",
        },
        work_orders=[
            {
                "work_order_id": "wo-backfill-store",
                "title": "Completed deliverable lane",
                "file_scope": [],
                "status": "completed",
                "target_agent": "codex",
                "reviewer_agent": "claude",
                "owner_session_id": "sess-backfill-store",
                "branch": "codex/backfill-store",
                "worktree_path": str(repo),
                "changed_paths": ["aragora/nomic/dev_coordination.py"],
                "commit_shas": ["abc12345"],
                "tests_run": ["python -m pytest tests/nomic/test_dev_coordination.py -q"],
                "confidence": 0.73,
            }
        ],
    )
    lease = store.claim_lease(
        task_id="wo-backfill-store",
        title="Completed deliverable lane",
        owner_agent="codex",
        owner_session_id="sess-backfill-store",
        branch="codex/backfill-store",
        worktree_path=str(repo),
        claimed_paths=["aragora/nomic/dev_coordination.py"],
        metadata={
            "supervisor_run_id": run["run_id"],
            "work_order_id": "wo-backfill-store",
            "task_key": f"{run['run_id']}:wo-backfill-store",
        },
    )
    updated = store.get_supervisor_run(run["run_id"])
    assert updated is not None
    updated["work_orders"][0]["lease_id"] = lease.lease_id
    updated["work_orders"][0]["status"] = "completed"
    store.update_supervisor_run(run["run_id"], work_orders=updated["work_orders"])

    backfilled = store.backfill_missing_completion_receipts()
    refreshed = store.get_supervisor_run(run["run_id"])

    assert backfilled == 1
    assert refreshed is not None
    assert refreshed["work_orders"][0]["file_scope"] == ["aragora/nomic/dev_coordination.py"]
    assert (
        refreshed["work_orders"][0]["metadata"]["backfilled_file_scope_from_changed_paths"] is True
    )
    receipt_id = refreshed["work_orders"][0]["receipt_id"]
    assert receipt_id is not None
    receipt = store.get_completion_receipt(receipt_id)
    assert receipt is not None
    assert receipt.outcome == "deliverable_created"
    assert receipt.metadata["backfilled_receipt"] is True


def test_backfill_missing_completion_receipts_rehydrates_empty_lease_scope(
    repo: Path, store: DevCoordinationStore
) -> None:
    run = store.create_supervisor_run(
        goal="Rehydrate empty lease scope during receipt backfill",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={
            "require_merge_approval": True,
            "require_external_action_approval": True,
        },
        spec={
            "raw_goal": "Rehydrate empty lease scope during receipt backfill",
            "refined_goal": "Rehydrate empty lease scope during receipt backfill",
        },
        work_orders=[
            {
                "work_order_id": "wo-rehydrate-lease-scope",
                "title": "Historical merge-gate deliverable",
                "file_scope": ["tests/nomic/test_dev_coordination.py"],
                "status": "needs_human",
                "failure_reason": "merge_gate_failed",
                "worker_outcome": "merge_gate_failed",
                "target_agent": "codex",
                "reviewer_agent": "claude",
                "owner_session_id": "sess-rehydrate-lease-scope",
                "branch": "codex/rehydrate-lease-scope",
                "worktree_path": str(repo),
                "initial_head": "abc12345",
                "head_sha": "def67890",
                "changed_paths": ["tests/nomic/test_dev_coordination.py"],
                "commit_shas": ["def67890"],
                "tests_run": [],
                "verification_results": [],
                "confidence": 0.82,
            }
        ],
    )
    lease = store.claim_lease(
        task_id="wo-rehydrate-lease-scope",
        title="Historical merge-gate deliverable",
        owner_agent="codex",
        owner_session_id="sess-rehydrate-lease-scope",
        branch="codex/rehydrate-lease-scope",
        worktree_path=str(repo),
        claimed_paths=[],
        allowed_globs=[],
        metadata={
            "supervisor_run_id": run["run_id"],
            "work_order_id": "wo-rehydrate-lease-scope",
            "task_key": f"{run['run_id']}:wo-rehydrate-lease-scope",
            "last_scope_violation": {
                "detected_at": "2026-03-30T00:00:00+00:00",
                "changed_paths": ["tests/nomic/test_dev_coordination.py"],
                "violations": [
                    {
                        "type": "undeclared_scope",
                        "path": "tests/nomic/test_dev_coordination.py",
                    }
                ],
            },
        },
    )
    updated = store.get_supervisor_run(run["run_id"])
    assert updated is not None
    updated["work_orders"][0]["lease_id"] = lease.lease_id
    updated["work_orders"][0]["status"] = "needs_human"
    store.update_supervisor_run(run["run_id"], work_orders=updated["work_orders"])

    backfilled = store.backfill_missing_completion_receipts()
    refreshed = store.get_supervisor_run(run["run_id"])

    assert backfilled == 1
    assert refreshed is not None
    receipt_id = refreshed["work_orders"][0]["receipt_id"]
    assert receipt_id is not None
    receipt = store.get_completion_receipt(receipt_id)
    assert receipt is not None
    assert receipt.outcome == "deliverable_created"
    assert receipt.metadata["backfilled_receipt"] is True
    assert receipt.metadata["lease_scope_rehydrated"] is True

    refreshed_lease = next(
        candidate
        for candidate in store.list_leases(limit=50)
        if candidate.lease_id == lease.lease_id
    )
    assert refreshed_lease.claimed_paths == ["tests/nomic/test_dev_coordination.py"]
    assert refreshed_lease.allowed_globs == []
    assert "last_scope_violation" not in refreshed_lease.metadata


def test_replay_missing_verification_for_merge_gate_failures_marks_lane_completed(
    repo: Path, store: DevCoordinationStore
) -> None:
    test_path = repo / "tests" / "test_verification_pass.py"
    test_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.write_text(
        "def test_verification_passes():\n    assert True\n",
        encoding="utf-8",
    )
    changed_path = "tests/test_verification_pass.py"
    test_command = f"python -m pytest {changed_path} -q"

    run = store.create_supervisor_run(
        goal="Replay missing verification and complete lane",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={
            "require_merge_approval": True,
            "require_external_action_approval": True,
        },
        spec={},
        work_orders=[
            {
                "work_order_id": "wo-verification-replay-pass",
                "title": "Receipt-backed merge-gate lane",
                "file_scope": [changed_path],
                "status": "needs_human",
                "review_status": "changes_requested",
                "failure_reason": "merge_gate_failed",
                "worker_outcome": "merge_gate_failed",
                "target_agent": "codex",
                "reviewer_agent": "claude",
                "owner_session_id": "sess-verification-replay-pass",
                "branch": "codex/verification-replay-pass",
                "worktree_path": str(repo),
                "head_sha": "def67890",
                "commit_shas": ["def67890"],
                "changed_paths": [changed_path],
                "expected_tests": [test_command],
                "tests_run": [],
                "verification_results": [],
            }
        ],
    )
    lease = store.claim_lease(
        task_id="wo-verification-replay-pass",
        title="Receipt-backed merge-gate lane",
        owner_agent="codex",
        owner_session_id="sess-verification-replay-pass",
        branch="codex/verification-replay-pass",
        worktree_path=str(repo),
        claimed_paths=[changed_path],
        metadata={
            "supervisor_run_id": run["run_id"],
            "work_order_id": "wo-verification-replay-pass",
            "task_key": f"{run['run_id']}:wo-verification-replay-pass",
        },
    )
    receipt = store.record_completion(
        lease_id=lease.lease_id,
        owner_agent="codex",
        owner_session_id="sess-verification-replay-pass",
        branch="codex/verification-replay-pass",
        worktree_path=str(repo),
        head_sha="def67890",
        commit_shas=["def67890"],
        changed_paths=[changed_path],
        tests_run=[],
        validations_run=[],
        assumptions=[],
        blockers=[],
        outcome="deliverable_created",
        risks=[],
        confidence=0.81,
        metadata={"backfilled_receipt": True},
        require_session_ownership=False,
    )

    updated = store.get_supervisor_run(run["run_id"])
    assert updated is not None
    updated["work_orders"][0]["lease_id"] = lease.lease_id
    updated["work_orders"][0]["receipt_id"] = receipt.receipt_id
    updated["work_orders"][0]["status"] = "needs_human"
    updated["work_orders"][0]["review_status"] = "changes_requested"
    updated["work_orders"][0]["expected_tests"] = [test_command]
    store.update_supervisor_run(run["run_id"], work_orders=updated["work_orders"])

    with patch.object(
        DevCoordinationStore,
        "_resolve_verification_worktree",
        return_value=(str(repo), None),
    ):
        replayed = store.replay_missing_verification_for_merge_gate_failures()
    refreshed = store.get_supervisor_run(run["run_id"])

    assert replayed == 1
    assert refreshed is not None
    item = refreshed["work_orders"][0]
    assert item["status"] == "completed"
    assert item["review_status"] == "pending_heterogeneous_review"
    assert item["worker_outcome"] == "completed"
    assert item["tests_run"] == [test_command]
    assert item["verification_results"][0]["command"] == test_command
    assert item["verification_results"][0]["passed"] is True
    assert item["metadata"]["verification_replayed"] is True
    assert "failure_reason" not in item
    assert "dispatch_error" not in item

    refreshed_receipt = store.get_completion_receipt(receipt.receipt_id)
    assert refreshed_receipt is not None
    assert refreshed_receipt.tests_run == [test_command]
    assert refreshed_receipt.metadata["verification_replayed"] is True


def test_replay_missing_verification_for_merge_gate_failures_keeps_lane_blocked_on_fail(
    repo: Path, store: DevCoordinationStore
) -> None:
    test_path = repo / "tests" / "test_verification_fail.py"
    test_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.write_text(
        "def test_verification_fails():\n    assert False\n",
        encoding="utf-8",
    )
    changed_path = "tests/test_verification_fail.py"
    test_command = f"python -m pytest {changed_path} -q"

    run = store.create_supervisor_run(
        goal="Replay missing verification and keep lane blocked",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={
            "require_merge_approval": True,
            "require_external_action_approval": True,
        },
        spec={},
        work_orders=[
            {
                "work_order_id": "wo-verification-replay-fail",
                "title": "Receipt-backed failing merge-gate lane",
                "file_scope": [changed_path],
                "status": "needs_human",
                "review_status": "changes_requested",
                "failure_reason": "merge_gate_failed",
                "worker_outcome": "merge_gate_failed",
                "target_agent": "codex",
                "reviewer_agent": "claude",
                "owner_session_id": "sess-verification-replay-fail",
                "branch": "codex/verification-replay-fail",
                "worktree_path": str(repo),
                "head_sha": "abc98765",
                "commit_shas": ["abc98765"],
                "changed_paths": [changed_path],
                "expected_tests": [test_command],
                "tests_run": [],
                "verification_results": [],
            }
        ],
    )
    lease = store.claim_lease(
        task_id="wo-verification-replay-fail",
        title="Receipt-backed failing merge-gate lane",
        owner_agent="codex",
        owner_session_id="sess-verification-replay-fail",
        branch="codex/verification-replay-fail",
        worktree_path=str(repo),
        claimed_paths=[changed_path],
        metadata={
            "supervisor_run_id": run["run_id"],
            "work_order_id": "wo-verification-replay-fail",
            "task_key": f"{run['run_id']}:wo-verification-replay-fail",
        },
    )
    receipt = store.record_completion(
        lease_id=lease.lease_id,
        owner_agent="codex",
        owner_session_id="sess-verification-replay-fail",
        branch="codex/verification-replay-fail",
        worktree_path=str(repo),
        head_sha="abc98765",
        commit_shas=["abc98765"],
        changed_paths=[changed_path],
        tests_run=[],
        validations_run=[],
        assumptions=[],
        blockers=[],
        outcome="deliverable_created",
        risks=[],
        confidence=0.61,
        metadata={"backfilled_receipt": True},
        require_session_ownership=False,
    )

    updated = store.get_supervisor_run(run["run_id"])
    assert updated is not None
    updated["work_orders"][0]["lease_id"] = lease.lease_id
    updated["work_orders"][0]["receipt_id"] = receipt.receipt_id
    updated["work_orders"][0]["status"] = "needs_human"
    updated["work_orders"][0]["review_status"] = "changes_requested"
    updated["work_orders"][0]["expected_tests"] = [test_command]
    store.update_supervisor_run(run["run_id"], work_orders=updated["work_orders"])

    with patch.object(
        DevCoordinationStore,
        "_resolve_verification_worktree",
        return_value=(str(repo), None),
    ):
        replayed = store.replay_missing_verification_for_merge_gate_failures()
    refreshed = store.get_supervisor_run(run["run_id"])

    assert replayed == 1
    assert refreshed is not None
    item = refreshed["work_orders"][0]
    assert item["status"] == "changes_requested"
    assert item["review_status"] == "changes_requested"
    assert item["worker_outcome"] == "merge_gate_failed"
    assert item["failure_reason"] == "merge_gate_failed"
    assert item["tests_run"] == [test_command]
    assert item["verification_results"][0]["command"] == test_command
    assert item["verification_results"][0]["passed"] is False
    assert item["dispatch_error"].startswith("merge gate blocked: verification failed:")
    assert item["metadata"]["verification_replayed"] is True

    refreshed_receipt = store.get_completion_receipt(receipt.receipt_id)
    assert refreshed_receipt is not None
    assert refreshed_receipt.tests_run == [test_command]
    assert refreshed_receipt.metadata["verification_replayed"] is True


def test_replay_missing_verification_for_merge_gate_failures_uses_temp_worktree_when_missing(
    repo: Path, store: DevCoordinationStore
) -> None:
    test_path = repo / "tests" / "test_verification_fallback.py"
    test_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.write_text(
        "def test_verification_fallback():\n    assert True\n",
        encoding="utf-8",
    )
    changed_path = "tests/test_verification_fallback.py"
    test_command = f"python -m pytest {changed_path} -q"
    _run(repo, "git", "add", changed_path)
    _run(repo, "git", "commit", "-m", "add fallback verification test")
    branch = "codex/verification-replay-fallback"
    _run(repo, "git", "branch", branch, "HEAD")

    run = store.create_supervisor_run(
        goal="Replay missing verification from fallback worktree",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={
            "require_merge_approval": True,
            "require_external_action_approval": True,
        },
        spec={},
        work_orders=[
            {
                "work_order_id": "wo-verification-replay-fallback",
                "title": "Receipt-backed lane with missing worktree",
                "file_scope": [changed_path],
                "status": "needs_human",
                "review_status": "changes_requested",
                "failure_reason": "merge_gate_failed",
                "worker_outcome": "merge_gate_failed",
                "target_agent": "codex",
                "reviewer_agent": "claude",
                "owner_session_id": "sess-verification-replay-fallback",
                "branch": branch,
                "worktree_path": str(repo / "missing-worktree"),
                "head_sha": "fedcba98",
                "commit_shas": ["fedcba98"],
                "changed_paths": [changed_path],
                "expected_tests": [test_command],
                "tests_run": [],
                "verification_results": [],
            }
        ],
    )
    lease = store.claim_lease(
        task_id="wo-verification-replay-fallback",
        title="Receipt-backed lane with missing worktree",
        owner_agent="codex",
        owner_session_id="sess-verification-replay-fallback",
        branch=branch,
        worktree_path=str(repo / "missing-worktree"),
        claimed_paths=[changed_path],
        metadata={
            "supervisor_run_id": run["run_id"],
            "work_order_id": "wo-verification-replay-fallback",
            "task_key": f"{run['run_id']}:wo-verification-replay-fallback",
        },
    )
    receipt = store.record_completion(
        lease_id=lease.lease_id,
        owner_agent="codex",
        owner_session_id="sess-verification-replay-fallback",
        branch=branch,
        worktree_path=str(repo),
        head_sha="fedcba98",
        commit_shas=["fedcba98"],
        changed_paths=[changed_path],
        tests_run=[],
        validations_run=[],
        assumptions=[],
        blockers=[],
        outcome="deliverable_created",
        risks=[],
        confidence=0.74,
        metadata={"backfilled_receipt": True},
        require_session_ownership=False,
    )

    updated = store.get_supervisor_run(run["run_id"])
    assert updated is not None
    updated["work_orders"][0]["lease_id"] = lease.lease_id
    updated["work_orders"][0]["receipt_id"] = receipt.receipt_id
    updated["work_orders"][0]["status"] = "needs_human"
    updated["work_orders"][0]["review_status"] = "changes_requested"
    updated["work_orders"][0]["expected_tests"] = [test_command]
    store.update_supervisor_run(run["run_id"], work_orders=updated["work_orders"])

    replayed = store.replay_missing_verification_for_merge_gate_failures()
    refreshed = store.get_supervisor_run(run["run_id"])

    assert replayed == 1
    assert refreshed is not None
    item = refreshed["work_orders"][0]
    assert item["status"] == "completed"
    assert item["review_status"] == "pending_heterogeneous_review"
    assert item["tests_run"] == [test_command]

    fallback_parent = repo / ".worktrees" / "verification-replay"
    assert not fallback_parent.exists() or not any(fallback_parent.iterdir())


def test_replay_docs_only_merge_gate_failures_prepends_capability_matrix_generation(
    repo: Path, store: DevCoordinationStore
) -> None:
    changed_path = "docs/status/ACTIVE_EXECUTION_ISSUES.md"
    reconcile_command = "python3 scripts/reconcile_status_docs.py --strict --output /tmp/report.md"
    version_command = "python3 scripts/check_version_alignment.py"
    expected_commands = [
        "python3 scripts/generate_capability_matrix.py",
        "python3 scripts/generate_capability_matrix.py --out docs-site/docs/contributing/capability-matrix.md",
        reconcile_command,
        version_command,
    ]

    run = store.create_supervisor_run(
        goal="Replay docs-only merge gate after capability matrix drift",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={},
        spec={},
        work_orders=[
            {
                "work_order_id": "wo-docs-replay-pass",
                "title": "Docs-only capability matrix drift",
                "status": "needs_human",
                "review_status": "changes_requested",
                "failure_reason": "merge_gate_failed",
                "worker_outcome": "merge_gate_failed",
                "branch": "codex/docs-replay-pass",
                "worktree_path": str(repo),
                "head_sha": "abc12345",
                "commit_shas": ["abc12345"],
                "changed_paths": [changed_path],
                "expected_tests": [reconcile_command, version_command],
                "tests_run": [reconcile_command],
                "verification_results": [
                    {
                        "command": reconcile_command,
                        "passed": False,
                        "exit_code": 1,
                        "stdout": "Matrix is out of sync with YAML. Run: python scripts/generate_capability_matrix.py",
                        "stderr": "",
                        "duration_seconds": 1.0,
                    }
                ],
                "dispatch_error": (
                    f"merge gate blocked: verification failed: {reconcile_command} (exit 1)"
                ),
            }
        ],
    )
    lease = store.claim_lease(
        task_id="wo-docs-replay-pass",
        title="Docs-only capability matrix drift",
        owner_agent="codex",
        owner_session_id="sess-docs-replay-pass",
        branch="codex/docs-replay-pass",
        worktree_path=str(repo),
        claimed_paths=[changed_path],
        metadata={
            "supervisor_run_id": run["run_id"],
            "work_order_id": "wo-docs-replay-pass",
            "task_key": f"{run['run_id']}:wo-docs-replay-pass",
        },
    )
    receipt = store.record_completion(
        lease_id=lease.lease_id,
        owner_agent="codex",
        owner_session_id="sess-docs-replay-pass",
        branch="codex/docs-replay-pass",
        worktree_path=str(repo),
        head_sha="abc12345",
        commit_shas=["abc12345"],
        changed_paths=[changed_path],
        tests_run=[reconcile_command],
        validations_run=[reconcile_command],
        assumptions=[],
        blockers=[],
        outcome="deliverable_created",
        risks=[],
        confidence=0.79,
        metadata={"backfilled_receipt": True},
        require_session_ownership=False,
    )

    updated = store.get_supervisor_run(run["run_id"])
    assert updated is not None
    updated["work_orders"][0]["lease_id"] = lease.lease_id
    updated["work_orders"][0]["receipt_id"] = receipt.receipt_id
    updated["work_orders"][0]["status"] = "needs_human"
    updated["work_orders"][0]["review_status"] = "changes_requested"
    updated["work_orders"][0]["failure_reason"] = "merge_gate_failed"
    updated["work_orders"][0]["worker_outcome"] = "merge_gate_failed"
    updated["work_orders"][0]["expected_tests"] = [reconcile_command, version_command]
    store.update_supervisor_run(run["run_id"], work_orders=updated["work_orders"])

    captured: dict[str, list[str]] = {}

    def _run_commands(
        worktree_path: str, commands: list[str], *, timeout: float
    ) -> list[dict[str, object]]:
        captured["commands"] = list(commands)
        assert worktree_path == str(repo)
        assert timeout == 900.0
        return [
            {
                "command": command,
                "passed": True,
                "exit_code": 0,
                "stdout": "",
                "stderr": "",
                "duration_seconds": 1.0,
            }
            for command in commands
        ]

    with (
        patch.object(
            DevCoordinationStore,
            "_resolve_verification_worktree",
            return_value=(str(repo), None),
        ),
        patch.object(
            DevCoordinationStore,
            "_run_verification_commands_sync",
            side_effect=_run_commands,
        ),
    ):
        replayed = store.replay_docs_only_merge_gate_failures()

    refreshed = store.get_supervisor_run(run["run_id"])

    assert replayed == 1
    assert captured["commands"] == expected_commands
    assert refreshed is not None
    item = refreshed["work_orders"][0]
    assert item["status"] == "completed"
    assert item["review_status"] == "pending_heterogeneous_review"
    assert item["worker_outcome"] == "completed"
    assert item["tests_run"] == expected_commands
    assert item["metadata"]["verification_docs_replayed"] is True
    assert "failure_reason" not in item
    assert "dispatch_error" not in item

    refreshed_receipt = store.get_completion_receipt(receipt.receipt_id)
    assert refreshed_receipt is not None
    assert refreshed_receipt.tests_run == expected_commands
    assert refreshed_receipt.metadata["verification_replayed"] is True


def test_replay_docs_only_merge_gate_failures_uses_capability_sync_command_when_present(
    repo: Path, store: DevCoordinationStore
) -> None:
    changed_path = "docs/status/NEXT_STEPS_CANONICAL.md"
    reconcile_command = "python3 scripts/reconcile_status_docs.py --strict --output /tmp/report.md"
    capability_command = "python3 scripts/check_capability_matrix_sync.py"
    expected_commands = [
        "python3 scripts/generate_capability_matrix.py",
        "python3 scripts/generate_capability_matrix.py --out docs-site/docs/contributing/capability-matrix.md",
        reconcile_command,
        capability_command,
    ]

    run = store.create_supervisor_run(
        goal="Replay docs-only merge gate with explicit capability sync check",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={},
        spec={},
        work_orders=[
            {
                "work_order_id": "wo-docs-replay-capability",
                "title": "Docs-only explicit capability sync",
                "status": "needs_human",
                "review_status": "changes_requested",
                "failure_reason": "merge_gate_failed",
                "worker_outcome": "merge_gate_failed",
                "branch": "codex/docs-replay-capability",
                "worktree_path": str(repo),
                "head_sha": "def67890",
                "commit_shas": ["def67890"],
                "changed_paths": [changed_path],
                "expected_tests": [reconcile_command, capability_command],
                "tests_run": [capability_command],
                "verification_results": [
                    {
                        "command": capability_command,
                        "passed": False,
                        "exit_code": 1,
                        "stdout": "Capability matrix files are out of date.",
                        "stderr": "",
                        "duration_seconds": 1.0,
                    }
                ],
                "dispatch_error": (
                    f"merge gate blocked: verification failed: {capability_command} (exit 1)"
                ),
            }
        ],
    )
    lease = store.claim_lease(
        task_id="wo-docs-replay-capability",
        title="Docs-only explicit capability sync",
        owner_agent="codex",
        owner_session_id="sess-docs-replay-capability",
        branch="codex/docs-replay-capability",
        worktree_path=str(repo),
        claimed_paths=[changed_path],
        metadata={
            "supervisor_run_id": run["run_id"],
            "work_order_id": "wo-docs-replay-capability",
            "task_key": f"{run['run_id']}:wo-docs-replay-capability",
        },
    )
    receipt = store.record_completion(
        lease_id=lease.lease_id,
        owner_agent="codex",
        owner_session_id="sess-docs-replay-capability",
        branch="codex/docs-replay-capability",
        worktree_path=str(repo),
        head_sha="def67890",
        commit_shas=["def67890"],
        changed_paths=[changed_path],
        tests_run=[capability_command],
        validations_run=[capability_command],
        assumptions=[],
        blockers=[],
        outcome="deliverable_created",
        risks=[],
        confidence=0.72,
        metadata={"backfilled_receipt": True},
        require_session_ownership=False,
    )

    updated = store.get_supervisor_run(run["run_id"])
    assert updated is not None
    updated["work_orders"][0]["lease_id"] = lease.lease_id
    updated["work_orders"][0]["receipt_id"] = receipt.receipt_id
    updated["work_orders"][0]["status"] = "needs_human"
    updated["work_orders"][0]["review_status"] = "changes_requested"
    updated["work_orders"][0]["failure_reason"] = "merge_gate_failed"
    updated["work_orders"][0]["worker_outcome"] = "merge_gate_failed"
    updated["work_orders"][0]["expected_tests"] = [reconcile_command, capability_command]
    store.update_supervisor_run(run["run_id"], work_orders=updated["work_orders"])

    captured: dict[str, list[str]] = {}

    def _run_commands(
        worktree_path: str, commands: list[str], *, timeout: float
    ) -> list[dict[str, object]]:
        captured["commands"] = list(commands)
        assert worktree_path == str(repo)
        assert timeout == 900.0
        return [
            {
                "command": command,
                "passed": True,
                "exit_code": 0,
                "stdout": "",
                "stderr": "",
                "duration_seconds": 1.0,
            }
            for command in commands
        ]

    with (
        patch.object(
            DevCoordinationStore,
            "_resolve_verification_worktree",
            return_value=(str(repo), None),
        ),
        patch.object(
            DevCoordinationStore,
            "_run_verification_commands_sync",
            side_effect=_run_commands,
        ),
    ):
        replayed = store.replay_docs_only_merge_gate_failures()

    refreshed = store.get_supervisor_run(run["run_id"])

    assert replayed == 1
    assert captured["commands"] == expected_commands
    assert refreshed is not None
    item = refreshed["work_orders"][0]
    assert item["status"] == "completed"
    assert item["metadata"]["verification_docs_replayed"] is True


def test_replay_missing_required_merge_gate_failures_runs_only_missing_checks(
    repo: Path, store: DevCoordinationStore
) -> None:
    existing_test = repo / "tests" / "test_partial_existing.py"
    existing_test.parent.mkdir(parents=True, exist_ok=True)
    existing_test.write_text("def test_partial_existing():\n    assert True\n", encoding="utf-8")
    missing_test = repo / "tests" / "test_partial_missing.py"
    missing_test.write_text("def test_partial_missing():\n    assert True\n", encoding="utf-8")

    existing_command = "python -m pytest tests/test_partial_existing.py -q"
    missing_command = "python -m pytest tests/test_partial_missing.py -q"

    run = store.create_supervisor_run(
        goal="Replay only missing required verification commands",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={},
        spec={},
        work_orders=[
            {
                "work_order_id": "wo-missing-required",
                "title": "Replay missing required verification",
                "status": "needs_human",
                "review_status": "changes_requested",
                "failure_reason": "merge_gate_failed",
                "worker_outcome": "merge_gate_failed",
                "receipt_id": "receipt-missing-required",
                "branch": "codex/missing-required",
                "worktree_path": str(repo),
                "head_sha": "abc12345",
                "commit_shas": ["abc12345"],
                "changed_paths": [
                    "tests/test_partial_existing.py",
                    "tests/test_partial_missing.py",
                ],
                "expected_tests": [existing_command, missing_command],
                "tests_run": [existing_command],
                "verification_results": [
                    {
                        "command": existing_command,
                        "passed": True,
                        "exit_code": 0,
                        "stdout": "",
                        "stderr": "",
                        "duration_seconds": 1.0,
                    }
                ],
                "dispatch_error": (
                    f"merge gate blocked: required verification did not run: {missing_command}"
                ),
            }
        ],
    )
    lease = store.claim_lease(
        task_id="wo-missing-required",
        title="Replay missing required verification",
        owner_agent="codex",
        owner_session_id="sess-missing-required",
        branch="codex/missing-required",
        worktree_path=str(repo),
        claimed_paths=[
            "tests/test_partial_existing.py",
            "tests/test_partial_missing.py",
        ],
        metadata={
            "supervisor_run_id": run["run_id"],
            "work_order_id": "wo-missing-required",
            "task_key": f"{run['run_id']}:wo-missing-required",
        },
    )
    receipt = store.record_completion(
        lease_id=lease.lease_id,
        owner_agent="codex",
        owner_session_id="sess-missing-required",
        branch="codex/missing-required",
        worktree_path=str(repo),
        head_sha="abc12345",
        commit_shas=["abc12345"],
        changed_paths=[
            "tests/test_partial_existing.py",
            "tests/test_partial_missing.py",
        ],
        tests_run=[existing_command],
        validations_run=[existing_command],
        assumptions=[],
        blockers=[],
        outcome="deliverable_created",
        risks=[],
        confidence=0.75,
        metadata={"backfilled_receipt": True},
        require_session_ownership=False,
    )

    updated = store.get_supervisor_run(run["run_id"])
    assert updated is not None
    updated["work_orders"][0]["lease_id"] = lease.lease_id
    updated["work_orders"][0]["receipt_id"] = receipt.receipt_id
    updated["work_orders"][0]["status"] = "needs_human"
    updated["work_orders"][0]["review_status"] = "changes_requested"
    updated["work_orders"][0]["failure_reason"] = "merge_gate_failed"
    updated["work_orders"][0]["worker_outcome"] = "merge_gate_failed"
    store.update_supervisor_run(run["run_id"], work_orders=updated["work_orders"])

    with patch.object(
        DevCoordinationStore,
        "_resolve_verification_worktree",
        return_value=(str(repo), None),
    ):
        replayed = store.replay_missing_required_merge_gate_failures()

    refreshed = store.get_supervisor_run(run["run_id"])

    assert replayed == 1
    assert refreshed is not None
    item = refreshed["work_orders"][0]
    assert item["status"] == "completed"
    assert item["review_status"] == "pending_heterogeneous_review"
    assert item["worker_outcome"] == "completed"
    assert item["tests_run"] == [missing_command, existing_command]
    assert [entry["command"] for entry in item["verification_results"]] == [
        missing_command,
        existing_command,
    ]
    assert item["verification_results"][0]["passed"] is True
    assert item["verification_results"][1]["passed"] is True
    assert item["metadata"]["verification_missing_required_replayed"] is True
    assert "failure_reason" not in item
    assert "dispatch_error" not in item


def test_replay_narrow_pytest_merge_gate_failures_marks_lane_completed(
    repo: Path, store: DevCoordinationStore
) -> None:
    test_path = repo / "tests" / "test_narrow_replay_pass.py"
    test_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.write_text("def test_narrow_replay_pass():\n    assert True\n", encoding="utf-8")
    changed_path = "tests/test_narrow_replay_pass.py"
    test_command = f"python -m pytest {changed_path} -q"

    run = store.create_supervisor_run(
        goal="Replay narrow pytest merge gate",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={},
        spec={},
        work_orders=[
            {
                "work_order_id": "wo-narrow-pytest-replay",
                "title": "Receipt-backed narrow pytest lane",
                "status": "needs_human",
                "review_status": "changes_requested",
                "failure_reason": "merge_gate_failed",
                "worker_outcome": "merge_gate_failed",
                "branch": "codex/narrow-pytest-replay",
                "worktree_path": str(repo),
                "head_sha": "abc12345",
                "commit_shas": ["abc12345"],
                "changed_paths": [changed_path],
                "expected_tests": [test_command],
                "tests_run": [test_command],
                "verification_results": [
                    {
                        "command": test_command,
                        "passed": False,
                        "exit_code": 1,
                        "stdout": "AssertionError: old branch state",
                        "stderr": "",
                        "duration_seconds": 1.0,
                    }
                ],
                "dispatch_error": (
                    f"merge gate blocked: verification failed: {test_command} (exit 1)"
                ),
            }
        ],
    )
    lease = store.claim_lease(
        task_id="wo-narrow-pytest-replay",
        title="Receipt-backed narrow pytest lane",
        owner_agent="codex",
        owner_session_id="sess-narrow-pytest-replay",
        branch="codex/narrow-pytest-replay",
        worktree_path=str(repo),
        claimed_paths=[changed_path],
        metadata={
            "supervisor_run_id": run["run_id"],
            "work_order_id": "wo-narrow-pytest-replay",
            "task_key": f"{run['run_id']}:wo-narrow-pytest-replay",
        },
    )
    receipt = store.record_completion(
        lease_id=lease.lease_id,
        owner_agent="codex",
        owner_session_id="sess-narrow-pytest-replay",
        branch="codex/narrow-pytest-replay",
        worktree_path=str(repo),
        head_sha="abc12345",
        commit_shas=["abc12345"],
        changed_paths=[changed_path],
        tests_run=[test_command],
        validations_run=[test_command],
        assumptions=[],
        blockers=[],
        outcome="deliverable_created",
        risks=[],
        confidence=0.77,
        metadata={"backfilled_receipt": True},
        require_session_ownership=False,
    )

    updated = store.get_supervisor_run(run["run_id"])
    assert updated is not None
    updated["work_orders"][0]["lease_id"] = lease.lease_id
    updated["work_orders"][0]["receipt_id"] = receipt.receipt_id
    updated["work_orders"][0]["status"] = "needs_human"
    updated["work_orders"][0]["review_status"] = "changes_requested"
    updated["work_orders"][0]["failure_reason"] = "merge_gate_failed"
    updated["work_orders"][0]["worker_outcome"] = "merge_gate_failed"
    updated["work_orders"][0]["expected_tests"] = [test_command]
    store.update_supervisor_run(run["run_id"], work_orders=updated["work_orders"])

    with patch.object(
        DevCoordinationStore,
        "_resolve_verification_worktree",
        return_value=(str(repo), None),
    ):
        replayed = store.replay_narrow_pytest_merge_gate_failures()

    refreshed = store.get_supervisor_run(run["run_id"])
    assert replayed == 1
    assert refreshed is not None
    item = refreshed["work_orders"][0]
    assert item["status"] == "completed"
    assert item["review_status"] == "pending_heterogeneous_review"
    assert item["worker_outcome"] == "completed"
    assert item["tests_run"] == [test_command]
    assert item["metadata"]["verification_narrow_pytest_replayed"] is True


def test_replay_narrow_pytest_merge_gate_failures_skips_overbroad_commands(
    repo: Path, store: DevCoordinationStore
) -> None:
    broad_command = "python -m pytest tests/ -q -k 'not benchmark and not load' --timeout=60 -x"

    run = store.create_supervisor_run(
        goal="Do not replay overbroad pytest merge gates here",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={},
        spec={},
        work_orders=[
            {
                "work_order_id": "wo-narrow-pytest-skip",
                "title": "Overbroad pytest lane",
                "status": "needs_human",
                "review_status": "changes_requested",
                "failure_reason": "merge_gate_failed",
                "worker_outcome": "merge_gate_failed",
                "receipt_id": "receipt-narrow-pytest-skip",
                "branch": "codex/narrow-pytest-skip",
                "worktree_path": str(repo),
                "head_sha": "fedcba98",
                "commit_shas": ["fedcba98"],
                "changed_paths": ["aragora/debate/memory_manager.py"],
                "expected_tests": [broad_command],
                "tests_run": [broad_command],
                "verification_results": [
                    {
                        "command": broad_command,
                        "passed": False,
                        "exit_code": -1,
                        "stdout": "",
                        "stderr": "Timed out after 90s",
                        "duration_seconds": 90.0,
                    }
                ],
                "dispatch_error": (
                    f"merge gate blocked: verification failed: {broad_command} (exit -1)"
                ),
            }
        ],
    )

    with patch.object(
        DevCoordinationStore,
        "_run_verification_commands_sync",
        side_effect=AssertionError("should not replay overbroad pytest lane"),
    ):
        replayed = store.replay_narrow_pytest_merge_gate_failures()

    refreshed = store.get_supervisor_run(run["run_id"])
    assert replayed == 0
    assert refreshed is not None
    item = refreshed["work_orders"][0]
    assert item["status"] == "needs_human"
    assert item["expected_tests"] == [broad_command]


def test_replay_narrow_pytest_merge_gate_failures_can_target_specific_task_keys(
    repo: Path, store: DevCoordinationStore
) -> None:
    first_path = repo / "tests" / "test_narrow_targeted_first.py"
    second_path = repo / "tests" / "test_narrow_targeted_second.py"
    first_path.parent.mkdir(parents=True, exist_ok=True)
    first_path.write_text("def test_narrow_targeted_first():\n    assert True\n", encoding="utf-8")
    second_path.write_text(
        "def test_narrow_targeted_second():\n    assert True\n", encoding="utf-8"
    )
    first_changed = "tests/test_narrow_targeted_first.py"
    second_changed = "tests/test_narrow_targeted_second.py"
    first_command = f"python -m pytest {first_changed} -q"
    second_command = f"python -m pytest {second_changed} -q"

    run = store.create_supervisor_run(
        goal="Replay one narrow pytest lane by task key",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={},
        spec={},
        work_orders=[
            {
                "work_order_id": "wo-targeted-first",
                "title": "First targeted narrow lane",
                "status": "needs_human",
                "review_status": "changes_requested",
                "failure_reason": "merge_gate_failed",
                "worker_outcome": "merge_gate_failed",
                "branch": "codex/targeted-first",
                "worktree_path": str(repo),
                "head_sha": "abc12345",
                "commit_shas": ["abc12345"],
                "changed_paths": [first_changed],
                "expected_tests": [first_command],
                "tests_run": [first_command],
                "verification_results": [
                    {
                        "command": first_command,
                        "passed": False,
                        "exit_code": 1,
                        "stdout": "AssertionError: stale failure",
                        "stderr": "",
                        "duration_seconds": 1.0,
                    }
                ],
                "dispatch_error": f"merge gate blocked: verification failed: {first_command} (exit 1)",
            },
            {
                "work_order_id": "wo-targeted-second",
                "title": "Second targeted narrow lane",
                "status": "needs_human",
                "review_status": "changes_requested",
                "failure_reason": "merge_gate_failed",
                "worker_outcome": "merge_gate_failed",
                "branch": "codex/targeted-second",
                "worktree_path": str(repo),
                "head_sha": "def67890",
                "commit_shas": ["def67890"],
                "changed_paths": [second_changed],
                "expected_tests": [second_command],
                "tests_run": [second_command],
                "verification_results": [
                    {
                        "command": second_command,
                        "passed": False,
                        "exit_code": 1,
                        "stdout": "AssertionError: stale failure",
                        "stderr": "",
                        "duration_seconds": 1.0,
                    }
                ],
                "dispatch_error": f"merge gate blocked: verification failed: {second_command} (exit 1)",
            },
        ],
    )

    for work_order_id, branch, changed, command in (
        ("wo-targeted-first", "codex/targeted-first", first_changed, first_command),
        ("wo-targeted-second", "codex/targeted-second", second_changed, second_command),
    ):
        lease = store.claim_lease(
            task_id=work_order_id,
            title=work_order_id,
            owner_agent="codex",
            owner_session_id=f"sess-{work_order_id}",
            branch=branch,
            worktree_path=str(repo),
            claimed_paths=[changed],
            metadata={
                "supervisor_run_id": run["run_id"],
                "work_order_id": work_order_id,
                "task_key": f"{run['run_id']}:{work_order_id}",
            },
        )
        receipt = store.record_completion(
            lease_id=lease.lease_id,
            owner_agent="codex",
            owner_session_id=f"sess-{work_order_id}",
            branch=branch,
            worktree_path=str(repo),
            head_sha="abc12345",
            commit_shas=["abc12345"],
            changed_paths=[changed],
            tests_run=[command],
            validations_run=[command],
            assumptions=[],
            blockers=[],
            outcome="deliverable_created",
            risks=[],
            confidence=0.7,
            metadata={"backfilled_receipt": True},
            require_session_ownership=False,
        )
        updated = store.get_supervisor_run(run["run_id"])
        assert updated is not None
        for item in updated["work_orders"]:
            if item["work_order_id"] != work_order_id:
                continue
            item["lease_id"] = lease.lease_id
            item["receipt_id"] = receipt.receipt_id
            item["status"] = "needs_human"
            item["review_status"] = "changes_requested"
            item["failure_reason"] = "merge_gate_failed"
            item["worker_outcome"] = "merge_gate_failed"
            item["expected_tests"] = [command]
        store.update_supervisor_run(run["run_id"], work_orders=updated["work_orders"])

    target_task_key = f"{run['run_id']}:wo-targeted-first"
    with patch.object(
        DevCoordinationStore,
        "_resolve_verification_worktree",
        return_value=(str(repo), None),
    ):
        replayed = store.replay_narrow_pytest_merge_gate_failures(task_keys=[target_task_key])

    refreshed = store.get_supervisor_run(run["run_id"])
    assert replayed == 1
    assert refreshed is not None
    first_item = next(
        item for item in refreshed["work_orders"] if item["work_order_id"] == "wo-targeted-first"
    )
    second_item = next(
        item for item in refreshed["work_orders"] if item["work_order_id"] == "wo-targeted-second"
    )
    assert first_item["status"] == "completed"
    assert first_item["metadata"]["verification_narrow_pytest_replayed"] is True
    assert second_item["status"] == "needs_human"
    assert "verification_narrow_pytest_replayed" not in (second_item.get("metadata") or {})


def test_replay_environment_blocked_merge_gate_failures_marks_lane_completed(
    repo: Path, store: DevCoordinationStore
) -> None:
    changed_path = "tests/test_environment_replay.py"
    test_command = f"python -m pytest {changed_path} -q"

    run = store.create_supervisor_run(
        goal="Replay environment-blocked verification and complete lane",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={
            "require_merge_approval": True,
            "require_external_action_approval": True,
        },
        spec={},
        work_orders=[
            {
                "work_order_id": "wo-verification-env-replay",
                "title": "Receipt-backed environment-blocked lane",
                "file_scope": [changed_path],
                "status": "needs_human",
                "review_status": "changes_requested",
                "failure_reason": "merge_gate_failed",
                "worker_outcome": "merge_gate_failed",
                "target_agent": "codex",
                "reviewer_agent": "claude",
                "owner_session_id": "sess-verification-env-replay",
                "branch": "codex/verification-env-replay",
                "worktree_path": str(repo),
                "head_sha": "def67890",
                "commit_shas": ["def67890"],
                "changed_paths": [changed_path],
                "expected_tests": [test_command],
                "tests_run": [test_command],
                "verification_results": [
                    {
                        "command": test_command,
                        "passed": False,
                        "exit_code": 1,
                        "stdout": "ModuleNotFoundError: No module named 'aragora_debate'",
                        "stderr": "",
                        "duration_seconds": 1.0,
                    }
                ],
                "dispatch_error": (
                    "merge gate blocked: verification failed: "
                    f"{test_command} (exit 1) - ModuleNotFoundError: No module named 'aragora_debate'"
                ),
            }
        ],
    )
    lease = store.claim_lease(
        task_id="wo-verification-env-replay",
        title="Receipt-backed environment-blocked lane",
        owner_agent="codex",
        owner_session_id="sess-verification-env-replay",
        branch="codex/verification-env-replay",
        worktree_path=str(repo),
        claimed_paths=[changed_path],
        metadata={
            "supervisor_run_id": run["run_id"],
            "work_order_id": "wo-verification-env-replay",
            "task_key": f"{run['run_id']}:wo-verification-env-replay",
        },
    )
    receipt = store.record_completion(
        lease_id=lease.lease_id,
        owner_agent="codex",
        owner_session_id="sess-verification-env-replay",
        branch="codex/verification-env-replay",
        worktree_path=str(repo),
        head_sha="def67890",
        commit_shas=["def67890"],
        changed_paths=[changed_path],
        tests_run=[test_command],
        validations_run=[test_command],
        assumptions=[],
        blockers=[],
        outcome="deliverable_created",
        risks=[],
        confidence=0.81,
        metadata={"backfilled_receipt": True},
        require_session_ownership=False,
    )

    updated = store.get_supervisor_run(run["run_id"])
    assert updated is not None
    updated["work_orders"][0]["lease_id"] = lease.lease_id
    updated["work_orders"][0]["receipt_id"] = receipt.receipt_id
    updated["work_orders"][0]["status"] = "needs_human"
    updated["work_orders"][0]["review_status"] = "changes_requested"
    updated["work_orders"][0]["expected_tests"] = [test_command]
    updated["work_orders"][0]["worker_outcome"] = "merge_gate_failed"
    updated["work_orders"][0]["failure_reason"] = "merge_gate_failed"
    updated["work_orders"][0]["dispatch_error"] = (
        "merge gate blocked: verification failed: "
        f"{test_command} (exit 1) - ModuleNotFoundError: No module named 'aragora_debate'"
    )
    store.update_supervisor_run(run["run_id"], work_orders=updated["work_orders"])

    with (
        patch.object(
            DevCoordinationStore,
            "_resolve_verification_worktree",
            return_value=(str(repo), None),
        ),
        patch.object(
            DevCoordinationStore,
            "_run_verification_commands_sync",
            return_value=[
                {
                    "command": test_command,
                    "passed": True,
                    "exit_code": 0,
                    "stdout": "",
                    "stderr": "",
                    "duration_seconds": 1.0,
                }
            ],
        ),
        patch.object(DevCoordinationStore, "_cleanup_verification_worktree"),
    ):
        replayed = store.replay_environment_blocked_merge_gate_failures()

    refreshed = store.get_supervisor_run(run["run_id"])

    assert replayed == 1
    assert refreshed is not None
    item = refreshed["work_orders"][0]
    assert item["status"] == "completed"
    assert item["review_status"] == "pending_heterogeneous_review"
    assert item["worker_outcome"] == "completed"
    assert item["verification_results"][0]["passed"] is True
    assert item["metadata"]["verification_environment_replayed"] is True
    assert "failure_reason" not in item
    assert "dispatch_error" not in item

    refreshed_receipt = store.get_completion_receipt(receipt.receipt_id)
    assert refreshed_receipt is not None
    assert refreshed_receipt.tests_run == [test_command]
    assert refreshed_receipt.metadata["verification_replayed"] is True


def test_replay_environment_blocked_merge_gate_failures_skips_non_environment_failures(
    repo: Path, store: DevCoordinationStore
) -> None:
    changed_path = "tests/test_environment_skip.py"
    test_command = f"python -m pytest {changed_path} -q"

    run = store.create_supervisor_run(
        goal="Do not replay genuine verification failures",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={},
        spec={},
        work_orders=[
            {
                "work_order_id": "wo-verification-env-skip",
                "title": "Receipt-backed real failure lane",
                "file_scope": [changed_path],
                "status": "changes_requested",
                "review_status": "changes_requested",
                "failure_reason": "merge_gate_failed",
                "worker_outcome": "merge_gate_failed",
                "branch": "codex/verification-env-skip",
                "worktree_path": str(repo),
                "head_sha": "abc12345",
                "commit_shas": ["abc12345"],
                "changed_paths": [changed_path],
                "expected_tests": [test_command],
                "tests_run": [test_command],
                "verification_results": [
                    {
                        "command": test_command,
                        "passed": False,
                        "exit_code": 1,
                        "stdout": "AssertionError: expected 1 == 2",
                        "stderr": "",
                        "duration_seconds": 1.0,
                    }
                ],
                "receipt_id": "receipt-env-skip",
                "dispatch_error": (
                    f"merge gate blocked: verification failed: {test_command} (exit 1)"
                ),
            }
        ],
    )

    with patch.object(
        DevCoordinationStore,
        "_run_verification_commands_sync",
        side_effect=AssertionError("should not replay non-environment failure"),
    ):
        replayed = store.replay_environment_blocked_merge_gate_failures()

    refreshed = store.get_supervisor_run(run["run_id"])

    assert replayed == 0
    assert refreshed is not None
    item = refreshed["work_orders"][0]
    assert item["status"] == "changes_requested"
    assert item["failure_reason"] == "merge_gate_failed"
    assert item.get("metadata", {}) == {}


def test_replay_targeted_merge_gate_failures_retargets_broad_timeout_and_completes_lane(
    repo: Path, store: DevCoordinationStore
) -> None:
    test_path = repo / "tests" / "test_targeted_replay_pass.py"
    test_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.write_text("def test_targeted_replay_pass():\n    assert True\n", encoding="utf-8")
    changed_path = "tests/test_targeted_replay_pass.py"
    broad_command = "python -m pytest tests/ -q -k 'not benchmark and not load' --timeout=60 -x"
    targeted_command = f"python -m pytest {changed_path} -q"

    run = store.create_supervisor_run(
        goal="Retarget broad merge-gate timeout to concrete changed test",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={},
        spec={},
        work_orders=[
            {
                "work_order_id": "wo-targeted-replay-pass",
                "title": "Retarget broad verification timeout",
                "status": "needs_human",
                "review_status": "changes_requested",
                "failure_reason": "merge_gate_failed",
                "worker_outcome": "merge_gate_failed",
                "receipt_id": "receipt-targeted-pass",
                "branch": "codex/targeted-replay-pass",
                "worktree_path": str(repo),
                "head_sha": "abc12345",
                "commit_shas": ["abc12345"],
                "file_scope": ["tests", "aragora/**", "tests/**"],
                "changed_paths": [changed_path],
                "expected_tests": [broad_command],
                "tests_run": [broad_command],
                "verification_results": [
                    {
                        "command": broad_command,
                        "passed": False,
                        "exit_code": -1,
                        "stdout": "",
                        "stderr": "Timed out after 90s",
                        "duration_seconds": 90.0,
                    }
                ],
                "dispatch_error": (
                    "merge gate blocked: verification failed: "
                    f"{broad_command} (exit -1) - Timed out after 90s"
                ),
                "metadata": {
                    "acceptance_criteria": [
                        "All tests pass",
                        "```bash",
                        broad_command,
                        "```",
                    ]
                },
            }
        ],
    )
    lease = store.claim_lease(
        task_id="wo-targeted-replay-pass",
        title="Retarget broad verification timeout",
        owner_agent="codex",
        owner_session_id="sess-targeted-replay-pass",
        branch="codex/targeted-replay-pass",
        worktree_path=str(repo),
        claimed_paths=[changed_path],
        metadata={
            "supervisor_run_id": run["run_id"],
            "work_order_id": "wo-targeted-replay-pass",
            "task_key": f"{run['run_id']}:wo-targeted-replay-pass",
        },
    )
    receipt = store.record_completion(
        lease_id=lease.lease_id,
        owner_agent="codex",
        owner_session_id="sess-targeted-replay-pass",
        branch="codex/targeted-replay-pass",
        worktree_path=str(repo),
        head_sha="abc12345",
        commit_shas=["abc12345"],
        changed_paths=[changed_path],
        tests_run=[broad_command],
        validations_run=[broad_command],
        assumptions=[],
        blockers=[],
        outcome="deliverable_created",
        risks=[],
        confidence=0.78,
        metadata={"backfilled_receipt": True},
        require_session_ownership=False,
    )

    updated = store.get_supervisor_run(run["run_id"])
    assert updated is not None
    updated["work_orders"][0]["lease_id"] = lease.lease_id
    updated["work_orders"][0]["receipt_id"] = receipt.receipt_id
    updated["work_orders"][0]["status"] = "needs_human"
    updated["work_orders"][0]["review_status"] = "changes_requested"
    updated["work_orders"][0]["failure_reason"] = "merge_gate_failed"
    updated["work_orders"][0]["worker_outcome"] = "merge_gate_failed"
    updated["work_orders"][0]["expected_tests"] = [broad_command]
    store.update_supervisor_run(run["run_id"], work_orders=updated["work_orders"])

    with patch.object(
        DevCoordinationStore,
        "_resolve_verification_worktree",
        return_value=(str(repo), None),
    ):
        replayed = store.replay_targeted_merge_gate_failures()
    refreshed = store.get_supervisor_run(run["run_id"])

    assert replayed == 1
    assert refreshed is not None
    item = refreshed["work_orders"][0]
    assert item["status"] == "completed"
    assert item["review_status"] == "pending_heterogeneous_review"
    assert item["worker_outcome"] == "completed"
    assert item["expected_tests"] == [targeted_command]
    assert item["tests_run"] == [targeted_command]
    assert item["verification_results"][0]["command"] == targeted_command
    assert item["verification_results"][0]["passed"] is True
    assert item["metadata"]["verification_targeted_replayed"] is True
    assert item["metadata"]["verification_targeted_previous_expected_tests"] == [broad_command]
    assert "failure_reason" not in item
    assert "dispatch_error" not in item


def test_replay_targeted_merge_gate_failures_keeps_lane_blocked_on_real_targeted_failure(
    repo: Path, store: DevCoordinationStore
) -> None:
    test_path = repo / "tests" / "test_targeted_replay_fail.py"
    test_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.write_text("def test_targeted_replay_fail():\n    assert False\n", encoding="utf-8")
    changed_path = "tests/test_targeted_replay_fail.py"
    broad_command = "python -m pytest tests/ -q -k 'not benchmark and not load' --timeout=60 -x"
    targeted_command = f"python -m pytest {changed_path} -q"

    run = store.create_supervisor_run(
        goal="Retarget broad merge-gate timeout but keep real failing lane blocked",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={},
        spec={},
        work_orders=[
            {
                "work_order_id": "wo-targeted-replay-fail",
                "title": "Retarget broad verification timeout to failing test",
                "status": "needs_human",
                "review_status": "changes_requested",
                "failure_reason": "merge_gate_failed",
                "worker_outcome": "merge_gate_failed",
                "receipt_id": "receipt-targeted-fail",
                "branch": "codex/targeted-replay-fail",
                "worktree_path": str(repo),
                "head_sha": "def67890",
                "commit_shas": ["def67890"],
                "file_scope": ["tests", "aragora/**", "tests/**"],
                "changed_paths": [changed_path],
                "expected_tests": [broad_command],
                "tests_run": [broad_command],
                "verification_results": [
                    {
                        "command": broad_command,
                        "passed": False,
                        "exit_code": -1,
                        "stdout": "",
                        "stderr": "Timed out after 90s",
                        "duration_seconds": 90.0,
                    }
                ],
                "dispatch_error": (
                    "merge gate blocked: verification failed: "
                    f"{broad_command} (exit -1) - Timed out after 90s"
                ),
                "metadata": {
                    "acceptance_criteria": [
                        "All tests pass",
                        "```bash",
                        broad_command,
                        "```",
                    ]
                },
            }
        ],
    )
    lease = store.claim_lease(
        task_id="wo-targeted-replay-fail",
        title="Retarget broad verification timeout to failing test",
        owner_agent="codex",
        owner_session_id="sess-targeted-replay-fail",
        branch="codex/targeted-replay-fail",
        worktree_path=str(repo),
        claimed_paths=[changed_path],
        metadata={
            "supervisor_run_id": run["run_id"],
            "work_order_id": "wo-targeted-replay-fail",
            "task_key": f"{run['run_id']}:wo-targeted-replay-fail",
        },
    )
    receipt = store.record_completion(
        lease_id=lease.lease_id,
        owner_agent="codex",
        owner_session_id="sess-targeted-replay-fail",
        branch="codex/targeted-replay-fail",
        worktree_path=str(repo),
        head_sha="def67890",
        commit_shas=["def67890"],
        changed_paths=[changed_path],
        tests_run=[broad_command],
        validations_run=[broad_command],
        assumptions=[],
        blockers=[],
        outcome="deliverable_created",
        risks=[],
        confidence=0.61,
        metadata={"backfilled_receipt": True},
        require_session_ownership=False,
    )

    updated = store.get_supervisor_run(run["run_id"])
    assert updated is not None
    updated["work_orders"][0]["lease_id"] = lease.lease_id
    updated["work_orders"][0]["receipt_id"] = receipt.receipt_id
    updated["work_orders"][0]["status"] = "needs_human"
    updated["work_orders"][0]["review_status"] = "changes_requested"
    updated["work_orders"][0]["failure_reason"] = "merge_gate_failed"
    updated["work_orders"][0]["worker_outcome"] = "merge_gate_failed"
    updated["work_orders"][0]["expected_tests"] = [broad_command]
    store.update_supervisor_run(run["run_id"], work_orders=updated["work_orders"])

    with patch.object(
        DevCoordinationStore,
        "_resolve_verification_worktree",
        return_value=(str(repo), None),
    ):
        replayed = store.replay_targeted_merge_gate_failures()
    refreshed = store.get_supervisor_run(run["run_id"])

    assert replayed == 1
    assert refreshed is not None
    item = refreshed["work_orders"][0]
    assert item["status"] == "changes_requested"
    assert item["review_status"] == "changes_requested"
    assert item["worker_outcome"] == "merge_gate_failed"
    assert item["failure_reason"] == "merge_gate_failed"
    assert item["expected_tests"] == [targeted_command]
    assert item["tests_run"] == [targeted_command]
    assert item["verification_results"][0]["command"] == targeted_command
    assert item["verification_results"][0]["passed"] is False
    assert item["dispatch_error"].startswith(
        f"merge gate blocked: verification failed: {targeted_command}"
    )
    assert item["metadata"]["verification_targeted_replayed"] is True


def test_replay_targeted_merge_gate_failures_skips_rows_without_narrower_target(
    repo: Path, store: DevCoordinationStore
) -> None:
    broad_command = "python -m pytest tests/ -q -k 'not benchmark and not load' --timeout=60 -x"

    run = store.create_supervisor_run(
        goal="Do not retarget broad verification when no concrete test can be inferred",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={},
        spec={},
        work_orders=[
            {
                "work_order_id": "wo-targeted-replay-skip",
                "title": "Broad verification with no narrower target",
                "status": "changes_requested",
                "review_status": "changes_requested",
                "failure_reason": "merge_gate_failed",
                "worker_outcome": "merge_gate_failed",
                "receipt_id": "receipt-targeted-skip",
                "branch": "codex/targeted-replay-skip",
                "worktree_path": str(repo),
                "head_sha": "fedcba98",
                "commit_shas": ["fedcba98"],
                "file_scope": ["aragora/debate/memory_manager.py", "aragora/modes/deep_audit.py"],
                "changed_paths": [
                    "aragora/debate/memory_manager.py",
                    "aragora/modes/deep_audit.py",
                ],
                "expected_tests": [broad_command],
                "tests_run": [broad_command],
                "verification_results": [
                    {
                        "command": broad_command,
                        "passed": False,
                        "exit_code": -1,
                        "stdout": "",
                        "stderr": "Timed out after 90s",
                        "duration_seconds": 90.0,
                    }
                ],
                "dispatch_error": (
                    "merge gate blocked: verification failed: "
                    f"{broad_command} (exit -1) - Timed out after 90s"
                ),
            }
        ],
    )

    with patch.object(
        DevCoordinationStore,
        "_run_verification_commands_sync",
        side_effect=AssertionError("should not replay when no narrower target exists"),
    ):
        replayed = store.replay_targeted_merge_gate_failures()

    refreshed = store.get_supervisor_run(run["run_id"])
    assert replayed == 0
    assert refreshed is not None
    item = refreshed["work_orders"][0]
    assert item["status"] == "changes_requested"
    assert item["expected_tests"] == [broad_command]
    assert item["tests_run"] == [broad_command]


def test_targeted_replay_expected_tests_for_work_order_uses_existing_narrow_history() -> None:
    work_order = {
        "expected_tests": [
            "python -m pytest tests/ -q -k 'not benchmark and not load' --timeout=60 -x"
        ],
        "tests_run": [
            "python -m pytest tests/test_modes_deep_audit.py -q",
        ],
        "verification_results": [
            {
                "command": "python -m pytest tests/test_debate_memory_manager.py -q",
                "passed": False,
                "exit_code": 1,
                "stdout": "",
                "stderr": "failed",
            }
        ],
        "changed_paths": [
            "aragora/debate/memory_manager.py",
            "aragora/modes/deep_audit.py",
        ],
    }

    targeted = _targeted_replay_expected_tests_for_work_order(work_order)

    assert targeted == [
        "python -m pytest tests/test_modes_deep_audit.py -q",
        "python -m pytest tests/test_debate_memory_manager.py -q",
    ]


def test_targeted_replay_expected_tests_for_work_order_skips_directory_targets() -> None:
    work_order = {
        "expected_tests": [
            "python3 -m pytest tests/canvas/ tests/pipeline/ -q --tb=short",
        ],
        "changed_paths": [
            "aragora/pipeline/adapters.py",
            "tests/pipeline/test_adapters.py",
            "tests/pipeline/test_graph_store.py",
        ],
    }

    targeted = _targeted_replay_expected_tests_for_work_order(work_order)

    assert targeted == [
        "python -m pytest tests/pipeline/test_adapters.py -q",
        "python -m pytest tests/pipeline/test_graph_store.py -q",
    ]


def test_verification_timeout_for_command_extends_single_file_pytest() -> None:
    assert (
        _verification_timeout_for_command(
            "python -m pytest tests/handlers/test_playground.py -q",
            180.0,
        )
        == 300.0
    )
    assert (
        _verification_timeout_for_command(
            "python -m pytest tests/ -q -k 'not benchmark and not load' --timeout=60 -x",
            180.0,
        )
        == 180.0
    )


def test_replay_targeted_merge_gate_failures_honors_task_keys(
    repo: Path, store: DevCoordinationStore
) -> None:
    first_test = repo / "tests" / "test_targeted_replay_keyed_a.py"
    first_test.parent.mkdir(parents=True, exist_ok=True)
    first_test.write_text(
        "def test_targeted_replay_keyed_a():\n    assert True\n", encoding="utf-8"
    )
    second_test = repo / "tests" / "test_targeted_replay_keyed_b.py"
    second_test.write_text(
        "def test_targeted_replay_keyed_b():\n    assert True\n", encoding="utf-8"
    )

    broad_command = "python -m pytest tests/ -q -k 'not benchmark and not load' --timeout=60 -x"
    first_changed = "tests/test_targeted_replay_keyed_a.py"
    second_changed = "tests/test_targeted_replay_keyed_b.py"

    run = store.create_supervisor_run(
        goal="Replay only the requested task key",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={},
        spec={},
        work_orders=[
            {
                "work_order_id": "wo-targeted-key-a",
                "title": "Keyed replay A",
                "status": "needs_human",
                "review_status": "changes_requested",
                "failure_reason": "merge_gate_failed",
                "worker_outcome": "merge_gate_failed",
                "receipt_id": "receipt-targeted-key-a",
                "branch": "codex/targeted-key-a",
                "worktree_path": str(repo),
                "head_sha": "aaa11111",
                "commit_shas": ["aaa11111"],
                "file_scope": ["tests", "tests/**"],
                "changed_paths": [first_changed],
                "expected_tests": [broad_command],
                "tests_run": [broad_command],
                "verification_results": [
                    {
                        "command": broad_command,
                        "passed": False,
                        "exit_code": -1,
                        "stdout": "",
                        "stderr": "Timed out after 90s",
                        "duration_seconds": 90.0,
                    }
                ],
                "metadata": {"task_key": "run-keyed:wo-targeted-key-a"},
            },
            {
                "work_order_id": "wo-targeted-key-b",
                "title": "Keyed replay B",
                "status": "needs_human",
                "review_status": "changes_requested",
                "failure_reason": "merge_gate_failed",
                "worker_outcome": "merge_gate_failed",
                "receipt_id": "receipt-targeted-key-b",
                "branch": "codex/targeted-key-b",
                "worktree_path": str(repo),
                "head_sha": "bbb22222",
                "commit_shas": ["bbb22222"],
                "file_scope": ["tests", "tests/**"],
                "changed_paths": [second_changed],
                "expected_tests": [broad_command],
                "tests_run": [broad_command],
                "verification_results": [
                    {
                        "command": broad_command,
                        "passed": False,
                        "exit_code": -1,
                        "stdout": "",
                        "stderr": "Timed out after 90s",
                        "duration_seconds": 90.0,
                    }
                ],
                "metadata": {"task_key": "run-keyed:wo-targeted-key-b"},
            },
        ],
    )

    with patch.object(
        DevCoordinationStore,
        "_resolve_verification_worktree",
        return_value=(str(repo), None),
    ):
        replayed = store.replay_targeted_merge_gate_failures(
            task_keys=["run-keyed:wo-targeted-key-a"]
        )

    refreshed = store.get_supervisor_run(run["run_id"])

    assert replayed == 1
    assert refreshed is not None
    first_item, second_item = refreshed["work_orders"]
    assert first_item["status"] == "completed"
    assert first_item["tests_run"] == [f"python -m pytest {first_changed} -q"]
    assert second_item["status"] == "needs_human"
    assert second_item["tests_run"] == [broad_command]


def test_reclassify_branch_stale_merge_gate_failures_when_mainline_passes(
    repo: Path, store: DevCoordinationStore
) -> None:
    test_path = repo / "tests" / "test_branch_snapshot_stale.py"
    test_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.write_text("def test_branch_snapshot_stale():\n    assert True\n", encoding="utf-8")
    command = "python -m pytest tests/test_branch_snapshot_stale.py -q"

    run = store.create_supervisor_run(
        goal="Mark stale branch merge-gate failures when mainline passes",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={},
        spec={},
        work_orders=[
            {
                "work_order_id": "wo-branch-stale",
                "title": "Branch stale merge gate",
                "status": "needs_human",
                "review_status": "changes_requested",
                "failure_reason": "merge_gate_failed",
                "worker_outcome": "merge_gate_failed",
                "receipt_id": "receipt-branch-stale",
                "branch": "codex/branch-stale",
                "worktree_path": str(repo),
                "head_sha": "abc12345",
                "commit_shas": ["abc12345"],
                "changed_paths": ["aragora/debate/memory_manager.py"],
                "expected_tests": [command],
                "tests_run": [command],
                "verification_results": [
                    {
                        "command": command,
                        "passed": False,
                        "exit_code": 1,
                        "stdout": "",
                        "stderr": "old branch snapshot failure",
                        "duration_seconds": 1.0,
                    }
                ],
                "metadata": {"task_key": "run-stale:wo-branch-stale"},
            }
        ],
    )

    with patch.object(
        DevCoordinationStore,
        "_resolve_verification_worktree",
        return_value=(str(repo), None),
    ):
        reclassified = store.reclassify_branch_stale_merge_gate_failures(
            task_keys=["run-stale:wo-branch-stale"]
        )

    refreshed = store.get_supervisor_run(run["run_id"])

    assert reclassified == 1
    assert refreshed is not None
    item = refreshed["work_orders"][0]
    assert item["status"] == "changes_requested"
    assert item["review_status"] == "changes_requested"
    assert item["worker_outcome"] == "branch_snapshot_stale"
    assert item["failure_reason"] == "branch_snapshot_stale"
    assert item["verification_results"][0]["passed"] is False
    assert item["metadata"]["mainline_verification_passed"] is True
    assert item["metadata"]["mainline_verification_commands"] == [command]
    assert item["metadata"]["mainline_verification_results"][0]["passed"] is True
    assert item["blocking_question"] == (
        "Should this deliverable be rebased, regenerated, or otherwise refreshed on current main before review?"
    )


def test_reconcile_merge_gate_failed_work_orders_normalizes_existing_results(
    store: DevCoordinationStore,
) -> None:
    run = store.create_supervisor_run(
        goal="Reconcile merge-gate rows with normalized commands",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={},
        spec={},
        work_orders=[
            {
                "work_order_id": "wo-merge-gate-python3",
                "title": "Python command normalization",
                "status": "needs_human",
                "review_status": "changes_requested",
                "failure_reason": "merge_gate_failed",
                "worker_outcome": "merge_gate_failed",
                "receipt_id": "receipt-python3",
                "branch": "codex/python3-normalization",
                "commit_shas": ["abc12345"],
                "expected_tests": ["python -m pytest tests/swarm/test_supervisor.py -q"],
                "tests_run": ["python3 -m pytest tests/swarm/test_supervisor.py -q"],
                "verification_results": [
                    {
                        "command": "python3 -m pytest tests/swarm/test_supervisor.py -q",
                        "passed": True,
                        "exit_code": 0,
                        "stdout": "",
                        "stderr": "",
                        "duration_seconds": 1.0,
                    }
                ],
                "dispatch_error": (
                    "merge gate blocked: required verification did not run: "
                    "python -m pytest tests/swarm/test_supervisor.py -q"
                ),
            },
            {
                "work_order_id": "wo-merge-gate-bash-lc",
                "title": "bash -lc command normalization",
                "status": "needs_human",
                "review_status": "changes_requested",
                "failure_reason": "merge_gate_failed",
                "worker_outcome": "merge_gate_failed",
                "receipt_id": "receipt-bash-lc",
                "branch": "codex/bash-lc-normalization",
                "commit_shas": ["def67890"],
                "expected_tests": ["cd aragora/live && npx tsc --noEmit"],
                "tests_run": ["bash -lc 'cd aragora/live && npx tsc --noEmit'"],
                "verification_results": [
                    {
                        "command": "bash -lc 'cd aragora/live && npx tsc --noEmit'",
                        "passed": True,
                        "exit_code": 0,
                        "stdout": "",
                        "stderr": "",
                        "duration_seconds": 1.0,
                    }
                ],
                "dispatch_error": (
                    "merge gate blocked: required verification did not run: "
                    "cd aragora/live && npx tsc --noEmit"
                ),
            },
            {
                "work_order_id": "wo-merge-gate-broad-pytest",
                "title": "Broader pytest command covers expected file",
                "status": "needs_human",
                "review_status": "changes_requested",
                "failure_reason": "merge_gate_failed",
                "worker_outcome": "merge_gate_failed",
                "receipt_id": "receipt-broad-pytest",
                "branch": "codex/broad-pytest-normalization",
                "commit_shas": ["9876fedc"],
                "expected_tests": ["python -m pytest tests/swarm/test_supervisor.py -q"],
                "tests_run": [
                    "python -m pytest tests/ -q -k 'not benchmark and not load' --timeout=60 -x"
                ],
                "verification_results": [
                    {
                        "command": "python -m pytest tests/ -q -k 'not benchmark and not load' --timeout=60 -x",
                        "passed": True,
                        "exit_code": 0,
                        "stdout": "",
                        "stderr": "",
                        "duration_seconds": 1.0,
                    }
                ],
                "dispatch_error": (
                    "merge gate blocked: required verification did not run: "
                    "python -m pytest tests/swarm/test_supervisor.py -q"
                ),
            },
        ],
    )

    reconciled = store.reconcile_merge_gate_failed_work_orders()
    refreshed = store.get_supervisor_run(run["run_id"])

    assert reconciled == 3
    assert refreshed is not None
    for item in refreshed["work_orders"]:
        assert item["status"] == "completed"
        assert item["review_status"] == "pending_heterogeneous_review"
        assert item["worker_outcome"] == "completed"
        assert item["metadata"]["merge_gate_reconciled"] is True
        assert "failure_reason" not in item
        assert "dispatch_error" not in item


def test_run_verification_commands_sync_uses_shared_verification_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, object] = {}

    monkeypatch.setattr(
        WorkerLauncher,
        "_verification_environment",
        staticmethod(lambda worktree_path: {"CUSTOM_ENV": worktree_path}),
    )

    def fake_run(
        cmd: list[str],
        *,
        cwd: str,
        capture_output: bool,
        text: bool,
        timeout: float,
        check: bool,
        env: dict[str, str],
    ) -> subprocess.CompletedProcess[str]:
        seen["cwd"] = cwd
        seen["env"] = env
        return subprocess.CompletedProcess(cmd, 0, "ok\n", "")

    monkeypatch.setattr("aragora.nomic.dev_coordination.subprocess.run", fake_run)

    results = DevCoordinationStore._run_verification_commands_sync(
        str(tmp_path),
        ["python -m pytest tests/swarm/test_supervisor.py -q"],
        timeout=30.0,
    )

    assert results[0]["passed"] is True
    assert seen["cwd"] == str(tmp_path)
    assert seen["env"] == {"CUSTOM_ENV": str(tmp_path)}


def test_run_verification_commands_sync_uses_prepared_pytest_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, object] = {}

    monkeypatch.setattr(
        WorkerLauncher,
        "_verification_environment",
        staticmethod(lambda worktree_path: {"CUSTOM_ENV": worktree_path}),
    )
    monkeypatch.setattr(
        WorkerLauncher,
        "_prepare_verification_command",
        staticmethod(lambda command: "python - <<'PY'\nimport pytest\nPY"),
    )

    def fake_run(
        cmd: list[str],
        *,
        cwd: str,
        capture_output: bool,
        text: bool,
        timeout: float,
        check: bool,
        env: dict[str, str],
    ) -> subprocess.CompletedProcess[str]:
        seen["cmd"] = cmd
        seen["cwd"] = cwd
        seen["env"] = env
        return subprocess.CompletedProcess(cmd, 0, "ok\n", "")

    monkeypatch.setattr("aragora.nomic.dev_coordination.subprocess.run", fake_run)

    results = DevCoordinationStore._run_verification_commands_sync(
        str(tmp_path),
        ["python -m pytest tests/swarm/test_supervisor.py -q"],
        timeout=30.0,
    )

    assert results[0]["passed"] is True
    assert seen["cwd"] == str(tmp_path)
    assert seen["env"] == {"CUSTOM_ENV": str(tmp_path)}
    assert seen["cmd"] == ["/bin/bash", "-lc", "python - <<'PY'\nimport pytest\nPY"]


def test_update_completion_receipt_verification_locked_coerces_bytes_metadata(
    store: DevCoordinationStore,
) -> None:
    lease = store.claim_lease(
        task_id="wo-bytes-verification",
        title="Bytes verification metadata",
        owner_agent="codex",
        owner_session_id="sess-bytes",
        branch="codex/bytes",
        worktree_path="/tmp/wt-bytes",
        claimed_paths=["tests/nomic/test_dev_coordination.py"],
    )
    receipt = store.record_completion(
        lease_id=lease.lease_id,
        owner_agent="codex",
        owner_session_id="sess-bytes",
        branch="codex/bytes",
        worktree_path="/tmp/wt-bytes",
        commit_shas=["abc12345"],
        changed_paths=["tests/nomic/test_dev_coordination.py"],
        outcome="deliverable_created",
        require_session_ownership=False,
    )

    conn = store._connect()
    try:
        updated = DevCoordinationStore._update_completion_receipt_verification_locked(
            conn,
            receipt_id=receipt.receipt_id,
            verification_results=[
                {
                    "command": "python -m pytest tests/nomic/test_dev_coordination.py -q",
                    "passed": False,
                    "exit_code": 1,
                    "stdout": b"stdout-bytes",
                    "stderr": b"stderr-bytes",
                    "duration_seconds": 1.0,
                }
            ],
            replayed_at=datetime.now(timezone.utc).isoformat(),
        )
        conn.commit()
    finally:
        conn.close()

    assert updated is True
    refreshed = store.get_completion_receipt(receipt.receipt_id)
    assert refreshed is not None
    result = refreshed.metadata["verification_results"][0]
    assert result["stdout"] == "stdout-bytes"
    assert result["stderr"] == "stderr-bytes"


def test_environment_blocked_detection_covers_dependency_and_interpreter_failures() -> None:
    assert _verification_result_looks_environment_blocked(
        {
            "passed": False,
            "stdout": "ModuleNotFoundError: No module named 'pydantic_settings'",
            "stderr": "",
        }
    )
    assert _verification_result_looks_environment_blocked(
        {
            "passed": False,
            "stdout": "",
            "stderr": "/bin/bash: /Users/armand/local/aws-cli/python: cannot execute binary file",
        }
    )


def test_list_developer_tasks_preserves_terminal_truth_metadata(
    store: DevCoordinationStore,
) -> None:
    run = store.create_supervisor_run(
        goal="Preserve task deliverable truth",
        target_branch="main",
        supervisor_agents={"lead": "codex"},
        approval_policy={"mode": "manual"},
        spec={},
        work_orders=[
            {
                "task_id": "subtask_1",
                "title": "Task projection preserves branch deliverable",
                "status": "completed",
                "target_agent": "codex",
                "reviewer_agent": "claude",
                "lease_id": "lease-123",
                "owner_session_id": "sess-123",
                "branch": "codex/preserved-branch",
                "worktree_path": "/tmp/wt-preserved",
                "base_sha": "abc12345",
                "head_sha": "def67890",
                "commit_shas": ["def67890"],
                "changed_paths": ["tests/swarm/test_reporter.py"],
                "pr_url": "https://github.com/synaptent/aragora/pull/1234",
                "pr_number": 1234,
                "worker_outcome": "completed",
                "failure_reason": "scope_violation",
                "dispatch_error": "dispatch failed once",
                "blocking_question": "Does this need a narrower scope?",
                "blocker": {"reason": "scope_violation"},
                "file_scope": ["tests/swarm/test_reporter.py"],
            }
        ],
        status="completed",
    )

    task = store.get_developer_task(f"{run['run_id']}:subtask_1")

    assert task is not None
    metadata = task.metadata
    assert metadata["base_sha"] == "abc12345"
    assert metadata["head_sha"] == "def67890"
    assert metadata["commit_shas"] == ["def67890"]
    assert metadata["changed_paths"] == ["tests/swarm/test_reporter.py"]
    assert metadata["pr_url"] == "https://github.com/synaptent/aragora/pull/1234"
    assert metadata["pr_number"] == 1234
    assert metadata["worker_outcome"] == "completed"
    assert metadata["failure_reason"] == "scope_violation"
    assert metadata["dispatch_error"] == "dispatch failed once"
    assert metadata["blocking_question"] == "Does this need a narrower scope?"
    assert metadata["blocker"] == {"reason": "scope_violation"}


def test_backfill_file_scope_from_changed_paths_skips_explicit_scope_violation(
    repo: Path, store: DevCoordinationStore
) -> None:
    run = store.create_supervisor_run(
        goal="Skip scope-violation scope backfill",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={
            "require_merge_approval": True,
            "require_external_action_approval": True,
        },
        spec={
            "raw_goal": "Skip scope-violation scope backfill",
            "refined_goal": "Skip scope-violation scope backfill",
        },
        work_orders=[
            {
                "work_order_id": "wo-scope-violation",
                "title": "Scope violation lane",
                "file_scope": [],
                "status": "scope_violation",
                "changed_paths": ["aragora/nomic/dev_coordination.py"],
                "scope_violation": {
                    "detected_at": "2026-03-30T00:00:00+00:00",
                    "changed_paths": ["aragora/nomic/dev_coordination.py"],
                    "violations": [
                        {
                            "type": "out_of_scope",
                            "path": "aragora/nomic/dev_coordination.py",
                        }
                    ],
                },
            }
        ],
    )

    backfilled = store.backfill_file_scope_from_changed_paths()
    refreshed = store.get_supervisor_run(run["run_id"])

    assert backfilled == 0
    assert refreshed is not None
    assert refreshed["work_orders"][0]["file_scope"] == []


def test_backfill_missing_completion_receipts_skips_no_deliverable_lane(
    repo: Path, store: DevCoordinationStore
) -> None:
    run = store.create_supervisor_run(
        goal="Skip no-deliverable receipt backfill",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={
            "require_merge_approval": True,
            "require_external_action_approval": True,
        },
        spec={
            "raw_goal": "Skip no-deliverable receipt backfill",
            "refined_goal": "Skip no-deliverable receipt backfill",
        },
        work_orders=[
            {
                "work_order_id": "wo-no-deliverable",
                "title": "Needs human without deliverable",
                "file_scope": ["aragora/nomic/dev_coordination.py"],
                "status": "needs_human",
                "failure_reason": "merge_gate_failed",
                "target_agent": "codex",
                "reviewer_agent": "claude",
                "owner_session_id": "sess-no-deliverable",
                "branch": "codex/no-deliverable",
                "worktree_path": str(repo),
            }
        ],
    )
    lease = store.claim_lease(
        task_id="wo-no-deliverable",
        title="Needs human without deliverable",
        owner_agent="codex",
        owner_session_id="sess-no-deliverable",
        branch="codex/no-deliverable",
        worktree_path=str(repo),
        claimed_paths=["aragora/nomic/dev_coordination.py"],
        metadata={"supervisor_run_id": run["run_id"], "work_order_id": "wo-no-deliverable"},
    )
    updated = store.get_supervisor_run(run["run_id"])
    assert updated is not None
    updated["work_orders"][0]["lease_id"] = lease.lease_id
    updated["work_orders"][0]["status"] = "needs_human"
    store.update_supervisor_run(run["run_id"], work_orders=updated["work_orders"])

    backfilled = store.backfill_missing_completion_receipts()
    refreshed = store.get_supervisor_run(run["run_id"])

    assert backfilled == 0
    assert refreshed is not None
    assert refreshed["work_orders"][0].get("receipt_id") is None


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
async def test_sync_developer_task_queue_completes_archived_reaped_tasks(
    repo: Path, store: DevCoordinationStore
) -> None:
    run = store.create_supervisor_run(
        goal="Archive stale reaped queue item",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={
            "require_merge_approval": True,
            "require_external_action_approval": True,
        },
        spec={
            "raw_goal": "Archive stale reaped queue item",
            "refined_goal": "Archive stale reaped queue item",
        },
        work_orders=[
            {
                "work_order_id": "wo-archive-queue",
                "title": "Archive me later",
                "file_scope": ["aragora/nomic/dev_coordination.py"],
                "status": "needs_human",
                "failure_reason": "stale_lease_reaped",
                "target_agent": "codex",
                "reviewer_agent": "claude",
            }
        ],
    )
    queue = GlobalWorkQueue(storage_dir=repo / ".work_queue")

    await store.sync_developer_task_queue(queue)
    task_item = await queue.get(f"task:{run['run_id']}:wo-archive-queue")
    assert task_item is not None
    assert task_item.status == WorkStatus.BLOCKED

    conn = store._connect()
    try:
        conn.execute(
            "UPDATE supervisor_runs SET updated_at = ? WHERE run_id = ?",
            ("2000-01-01T00:00:00+00:00", run["run_id"]),
        )
        conn.commit()
    finally:
        conn.close()

    counts = await store.sync_developer_task_queue(queue)
    refreshed = await queue.get(f"task:{run['run_id']}:wo-archive-queue")
    run_record = store.get_supervisor_run(run["run_id"])

    assert counts["completed"] == 1
    assert refreshed is not None
    assert refreshed.status == WorkStatus.COMPLETED
    assert run_record is not None
    assert run_record["work_orders"][0]["status"] == "discarded"


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


def test_resolve_verification_worktree_rejects_sha_mismatch(
    tmp_path: Path,
) -> None:
    """Existing worktree whose HEAD does not match recorded head_sha must be
    rejected so a fresh worktree is created at the correct revision."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, capture_output=True, check=True)
    subprocess.run(
        ["git", "commit", "--allow-empty", "-m", "init"],
        cwd=repo,
        capture_output=True,
        check=True,
        env={
            **__import__("os").environ,
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@t",
        },
    )

    store = DevCoordinationStore(repo)
    worktree_dir = tmp_path / "existing_wt"
    worktree_dir.mkdir()
    # Create a valid git dir so git rev-parse HEAD works
    subprocess.run(
        ["git", "clone", str(repo), str(worktree_dir)],
        capture_output=True,
        check=True,
    )
    actual_head = subprocess.run(
        ["git", "-C", str(worktree_dir), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    # Use a fake sha that won't match
    fake_sha = "0000000000000000000000000000000000000000"
    assert not actual_head.startswith(fake_sha)

    work_order = {
        "worktree_path": str(worktree_dir),
        "head_sha": fake_sha,
        "branch": "main",
    }

    result_path, cleanup = store._resolve_verification_worktree(work_order)
    # The existing worktree should be rejected since HEAD != head_sha.
    # It should try to create a fresh worktree (which may fail in test
    # environment, returning empty string) -- the key assertion is that
    # it does NOT return the existing worktree_dir.
    if result_path:
        assert result_path != str(worktree_dir)
    if cleanup:
        store._cleanup_verification_worktree(cleanup)
