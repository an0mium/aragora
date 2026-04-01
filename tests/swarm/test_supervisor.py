"""Tests for supervisor-backed swarm execution."""

from __future__ import annotations

import subprocess
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.nomic.dev_coordination import DevCoordinationStore
from aragora.nomic.task_decomposer import SubTask, TaskDecomposition
from aragora.swarm.spec import SwarmSpec
from aragora.swarm.supervisor import (
    CAMPAIGN_OUTCOME_METADATA_KEY,
    WORKER_TYPE_CIRCUIT_BREAKERS_KEY,
    WORKER_TYPE_CIRCUIT_BREAKER_POLICY_KEY,
    SwarmSupervisor,
)
from aragora.swarm.worker_launcher import WorkerLauncher, WorkerProcess
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


def test_start_run_discards_duplicate_open_non_deliverable_lane(
    repo: Path, store: DevCoordinationStore
) -> None:
    store.create_supervisor_run(
        goal="Add --json output flag to aragora quickstart CLI",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={},
        spec={},
        work_orders=[
            {
                "work_order_id": "existing",
                "title": "Existing quickstart lane",
                "status": "waiting_conflict",
                "file_scope": [
                    "aragora/cli/commands/quickstart.py",
                    "aragora/cli/parser.py",
                    "tests/cli/test_quickstart.py",
                ],
            }
        ],
    )

    lifecycle = MagicMock()
    decomposer = MagicMock()
    decomposer.analyze.return_value = TaskDecomposition(
        original_task="Goal",
        complexity_score=2,
        complexity_level="low",
        should_decompose=True,
        subtasks=[
            SubTask(
                id="subtask_1",
                title="Test Changes",
                description="Add --json output flag to aragora quickstart CLI",
                file_scope=[
                    "aragora/cli/commands/quickstart.py",
                    "aragora/cli/parser.py",
                    "tests/cli/test_quickstart.py",
                ],
            )
        ],
    )
    supervisor = SwarmSupervisor(
        repo_root=repo,
        store=store,
        lifecycle=lifecycle,
        decomposer=decomposer,
    )

    run = supervisor.start_run(
        spec=SwarmSpec(
            raw_goal="Add --json output flag to aragora quickstart CLI",
            refined_goal="Add --json output flag to aragora quickstart CLI",
        ),
        refresh_scaling=False,
    )

    work_order = run.work_orders[0]
    assert work_order["status"] == "discarded"
    assert work_order["metadata"]["archived_due_to"] == "duplicate_open_work_order"
    assert work_order["metadata"]["archive_reason"] == "duplicate_open_work_order"
    assert work_order["metadata"]["canonical_task_key"].endswith(":existing")


def test_start_run_preserves_duplicate_scope_when_existing_lane_has_deliverable(
    repo: Path, store: DevCoordinationStore
) -> None:
    store.create_supervisor_run(
        goal="Add --json output flag to aragora quickstart CLI",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={},
        spec={},
        work_orders=[
            {
                "work_order_id": "existing",
                "title": "Reviewed quickstart lane",
                "status": "changes_requested",
                "receipt_id": "receipt-existing",
                "branch": "codex/existing-quickstart",
                "commit_shas": ["deadbeef"],
                "file_scope": [
                    "aragora/cli/commands/quickstart.py",
                    "aragora/cli/parser.py",
                    "tests/cli/test_quickstart.py",
                ],
            }
        ],
    )

    lifecycle = MagicMock()
    decomposer = MagicMock()
    decomposer.analyze.return_value = TaskDecomposition(
        original_task="Goal",
        complexity_score=2,
        complexity_level="low",
        should_decompose=True,
        subtasks=[
            SubTask(
                id="subtask_1",
                title="Test Changes",
                description="Add --json output flag to aragora quickstart CLI",
                file_scope=[
                    "aragora/cli/commands/quickstart.py",
                    "aragora/cli/parser.py",
                    "tests/cli/test_quickstart.py",
                ],
            )
        ],
    )
    supervisor = SwarmSupervisor(
        repo_root=repo,
        store=store,
        lifecycle=lifecycle,
        decomposer=decomposer,
    )

    run = supervisor.start_run(
        spec=SwarmSpec(
            raw_goal="Add --json output flag to aragora quickstart CLI",
            refined_goal="Add --json output flag to aragora quickstart CLI",
        ),
        refresh_scaling=False,
    )

    assert run.work_orders[0]["status"] == "queued"


def test_start_run_discards_duplicate_explicit_lane_by_tranche_lane_id(
    repo: Path, store: DevCoordinationStore
) -> None:
    store.create_supervisor_run(
        goal="Replace integrations UI path with truthful live-state behavior.",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={},
        spec={},
        work_orders=[
            {
                "work_order_id": "existing",
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

    supervisor = SwarmSupervisor(
        repo_root=repo,
        store=store,
        lifecycle=MagicMock(),
        decomposer=MagicMock(),
    )

    spec = SwarmSpec(
        raw_goal=(
            "Replace one non-trustworthy integrations UI path with live-state behavior "
            "and truthful status messaging."
        ),
        refined_goal=(
            "Replace one non-trustworthy integrations UI path with live-state behavior "
            "and truthful status messaging."
        ),
        work_orders=[
            {
                "work_order_id": "integrations-ui-truthful-status-slice",
                "title": "Integrations slice",
                "description": "Replace the status/edit path with truthful live-state behavior.",
                "file_scope": [
                    "aragora/live/**",
                    "aragora/server/handlers/features/**",
                    "tests/e2e/**",
                    "tests/handlers/**",
                    "docs/**",
                ],
                "metadata": {
                    "tranche_lane_id": "integrations-ui-truthful-status-slice",
                },
                "target_agent": "codex",
                "reviewer_agent": "claude",
            }
        ],
    )

    run = supervisor.start_run(spec=spec, refresh_scaling=False)

    work_order = run.work_orders[0]
    assert work_order["status"] == "discarded"
    assert work_order["metadata"]["archived_due_to"] == "duplicate_open_work_order"
    assert work_order["metadata"]["canonical_task_key"].endswith(":existing")


def test_start_run_discards_duplicate_scope_less_open_lane(
    repo: Path, store: DevCoordinationStore
) -> None:
    store.create_supervisor_run(
        goal="Write an ADR for standardizing health check endpoints across all deploy surfaces.",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={},
        spec={},
        work_orders=[
            {
                "work_order_id": "existing",
                "title": "ADR lane",
                "status": "waiting_conflict",
            }
        ],
    )

    lifecycle = MagicMock()
    decomposer = MagicMock()
    decomposer.analyze.return_value = TaskDecomposition(
        original_task="Goal",
        complexity_score=2,
        complexity_level="low",
        should_decompose=True,
        subtasks=[
            SubTask(
                id="subtask_1",
                title="ADR lane",
                description="Write an ADR for standardizing health check endpoints across all deploy surfaces.",
                file_scope=[],
            )
        ],
    )
    supervisor = SwarmSupervisor(
        repo_root=repo,
        store=store,
        lifecycle=lifecycle,
        decomposer=decomposer,
    )

    run = supervisor.start_run(
        spec=SwarmSpec(
            raw_goal="Write an ADR for standardizing health check endpoints across all deploy surfaces.",
            refined_goal="Write an ADR for standardizing health check endpoints across all deploy surfaces.",
        ),
        refresh_scaling=False,
    )

    work_order = run.work_orders[0]
    assert work_order["status"] == "discarded"
    assert work_order["metadata"]["archived_due_to"] == "duplicate_open_work_order"
    assert work_order["metadata"]["canonical_task_key"].endswith(":existing")


def test_start_run_discards_duplicate_open_lane_when_goal_differs_only_by_boilerplate(
    repo: Path, store: DevCoordinationStore
) -> None:
    store.create_supervisor_run(
        goal=(
            "Connect the results page to backend endpoints to display debate outcomes, "
            "consensus/dissent analysis, and confidence scores.\n\n"
            "Tranche objective: Make one already reachable core page functional."
        ),
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={},
        spec={},
        work_orders=[
            {
                "work_order_id": "existing",
                "title": "Results lane",
                "status": "waiting_conflict",
                "file_scope": ["aragora/live/**", "tests/e2e/**", "tests/handlers/**", "docs/**"],
            }
        ],
    )

    lifecycle = MagicMock()
    decomposer = MagicMock()
    decomposer.analyze.return_value = TaskDecomposition(
        original_task="Goal",
        complexity_score=2,
        complexity_level="low",
        should_decompose=True,
        subtasks=[
            SubTask(
                id="subtask_1",
                title="Results lane",
                description="Connect the results page to backend endpoints and include empty-state handling.",
                file_scope=["aragora/live/**", "tests/e2e/**", "tests/handlers/**", "docs/**"],
            )
        ],
    )
    supervisor = SwarmSupervisor(
        repo_root=repo,
        store=store,
        lifecycle=lifecycle,
        decomposer=decomposer,
    )

    run = supervisor.start_run(
        spec=SwarmSpec(
            raw_goal=(
                "Connect the results page to backend endpoints to display debate outcomes, "
                "consensus/dissent analysis, and confidence scores. Include empty-state handling.\n\n"
                "Verification commands:\n- python3 -m pytest tests/swarm/test_tranche_e2e.py -q"
            ),
            refined_goal=(
                "Connect the results page to backend endpoints to display debate outcomes, "
                "consensus/dissent analysis, and confidence scores. Include empty-state handling.\n\n"
                "Verification commands:\n- python3 -m pytest tests/swarm/test_tranche_e2e.py -q"
            ),
        ),
        refresh_scaling=False,
    )

    work_order = run.work_orders[0]
    assert work_order["status"] == "discarded"
    assert work_order["metadata"]["archived_due_to"] == "duplicate_open_work_order"
    assert work_order["metadata"]["canonical_task_key"].endswith(":existing")


def test_start_run_discards_duplicate_scope_less_explicit_lane_by_tranche_lane_id(
    repo: Path, store: DevCoordinationStore
) -> None:
    store.create_supervisor_run(
        goal="Connect the results page to backend endpoints.",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={},
        spec={},
        work_orders=[
            {
                "work_order_id": "existing",
                "title": "Results page lane",
                "status": "waiting_conflict",
                "metadata": {
                    "source": "explicit_spec_work_order",
                    "tranche_lane_id": "proj-001",
                },
            }
        ],
    )

    supervisor = SwarmSupervisor(
        repo_root=repo,
        store=store,
        lifecycle=MagicMock(),
        decomposer=MagicMock(),
    )

    spec = SwarmSpec(
        raw_goal="Rebuild the 30/60/90 execution map from current repo reality.",
        refined_goal="Rebuild the 30/60/90 execution map from current repo reality.",
        work_orders=[
            {
                "work_order_id": "proj-001",
                "title": "Results page lane",
                "description": "Connect the results page to backend endpoints.",
                "file_scope": [],
                "metadata": {
                    "tranche_lane_id": "proj-001",
                },
                "target_agent": "codex",
                "reviewer_agent": "claude",
            }
        ],
    )

    run = supervisor.start_run(spec=spec, refresh_scaling=False)

    work_order = run.work_orders[0]
    assert work_order["status"] == "discarded"
    assert work_order["metadata"]["archived_due_to"] == "duplicate_open_work_order"
    assert work_order["metadata"]["canonical_task_key"].endswith(":existing")


def test_start_run_discards_same_goal_contained_scope_duplicate_open_lane(
    repo: Path, store: DevCoordinationStore
) -> None:
    store.create_supervisor_run(
        goal="Write an ADR for standardizing health check endpoints across all deploy surfaces.",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={},
        spec={},
        work_orders=[
            {
                "work_order_id": "existing",
                "title": "Broader ADR lane",
                "status": "waiting_conflict",
                "file_scope": ["docs/ADR/"],
            }
        ],
    )

    lifecycle = MagicMock()
    decomposer = MagicMock()
    decomposer.analyze.return_value = TaskDecomposition(
        original_task="Goal",
        complexity_score=2,
        complexity_level="low",
        should_decompose=True,
        subtasks=[
            SubTask(
                id="subtask_1",
                title="Narrow ADR lane",
                description="Write an ADR for standardizing health check endpoints across all deploy surfaces.",
                file_scope=["docs/ADR/019-standardized-health-check-endpoints.md"],
            )
        ],
    )
    supervisor = SwarmSupervisor(
        repo_root=repo,
        store=store,
        lifecycle=lifecycle,
        decomposer=decomposer,
    )

    run = supervisor.start_run(
        spec=SwarmSpec(
            raw_goal="Write an ADR for standardizing health check endpoints across all deploy surfaces.",
            refined_goal="Write an ADR for standardizing health check endpoints across all deploy surfaces.",
        ),
        refresh_scaling=False,
    )

    work_order = run.work_orders[0]
    assert work_order["status"] == "discarded"
    assert work_order["metadata"]["archived_due_to"] == "duplicate_open_work_order"
    assert work_order["metadata"]["canonical_task_key"].endswith(":existing")


def test_start_run_preserves_same_goal_non_overlapping_scope_lane(
    repo: Path, store: DevCoordinationStore
) -> None:
    store.create_supervisor_run(
        goal="Write an ADR for standardizing health check endpoints across all deploy surfaces.",
        target_branch="main",
        supervisor_agents={"planner": "codex", "judge": "claude"},
        approval_policy={},
        spec={},
        work_orders=[
            {
                "work_order_id": "existing",
                "title": "ADR lane",
                "status": "waiting_conflict",
                "file_scope": ["docs/ADR/019-standardized-health-check-endpoints.md"],
            }
        ],
    )

    lifecycle = MagicMock()
    decomposer = MagicMock()
    decomposer.analyze.return_value = TaskDecomposition(
        original_task="Goal",
        complexity_score=2,
        complexity_level="low",
        should_decompose=True,
        subtasks=[
            SubTask(
                id="subtask_2",
                title="Deploy truth table lane",
                description="Write an ADR for standardizing health check endpoints across all deploy surfaces.",
                file_scope=["docs/deploy-truth-table.md"],
            )
        ],
    )
    supervisor = SwarmSupervisor(
        repo_root=repo,
        store=store,
        lifecycle=lifecycle,
        decomposer=decomposer,
    )

    run = supervisor.start_run(
        spec=SwarmSpec(
            raw_goal="Write an ADR for standardizing health check endpoints across all deploy surfaces.",
            refined_goal="Write an ADR for standardizing health check endpoints across all deploy surfaces.",
        ),
        refresh_scaling=False,
    )

    assert run.work_orders[0]["status"] == "queued"


def test_start_run_passes_acceptance_and_constraints_to_decomposer(
    repo: Path, store: DevCoordinationStore
) -> None:
    session = ManagedWorktreeSession(
        session_id="swarm-acceptance",
        agent="codex",
        branch="codex/swarm-acceptance",
        path=repo / "wt-acceptance",
        created=True,
        reconcile_status="up_to_date",
        payload={},
    )
    session.path.mkdir()

    lifecycle = MagicMock()
    lifecycle.ensure_managed_worktree.return_value = session
    decomposer = MagicMock()
    decomposer.analyze.return_value = TaskDecomposition(
        original_task="Goal",
        complexity_score=2,
        complexity_level="low",
        should_decompose=True,
        subtasks=[
            SubTask(
                id="wo-1",
                title="Server lane",
                description="Implement server lane",
                file_scope=["README.md"],
            )
        ],
    )
    supervisor = SwarmSupervisor(
        repo_root=repo,
        store=store,
        lifecycle=lifecycle,
        decomposer=decomposer,
    )
    spec = SwarmSpec(
        raw_goal="Goal",
        refined_goal="Goal",
        acceptance_criteria=["python -m pytest tests/swarm/test_supervisor.py -q"],
        constraints=["Keep merge gate enabled", "Human approval required"],
    )

    supervisor.start_run(spec=spec, max_concurrency=1)

    decomposer.analyze.assert_called_once()
    kwargs = decomposer.analyze.call_args.kwargs
    assert kwargs["acceptance_criteria"] == ["python -m pytest tests/swarm/test_supervisor.py -q"]
    assert kwargs["constraints"] == ["Keep merge gate enabled", "Human approval required"]


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


def test_refresh_run_requeues_leased_work_order_when_active_lease_is_missing(
    repo: Path, store: DevCoordinationStore
) -> None:
    session_path = repo / "wt-released-requeue"
    session_path.mkdir()
    lifecycle = MagicMock()
    lifecycle.ensure_managed_worktree.return_value = ManagedWorktreeSession(
        session_id="swarm-released-requeue",
        agent="codex",
        branch="codex/swarm-released-requeue",
        path=session_path,
        created=True,
        reconcile_status="up_to_date",
        payload={},
    )
    supervisor = SwarmSupervisor(repo_root=repo, store=store, lifecycle=lifecycle)
    run_record = store.create_supervisor_run(
        goal="requeue stale leased work order",
        target_branch="main",
        supervisor_agents={},
        approval_policy={},
        spec={"raw_goal": "requeue stale leased work order"},
        metadata={"max_concurrency": 1},
        work_orders=[
            {
                "work_order_id": "wo-stale-lease",
                "title": "Requeue stale lane",
                "description": "Requeue stale lane",
                "status": "leased",
                "target_agent": "codex",
                "reviewer_agent": "claude",
                "lease_id": "missing-lease",
                "worktree_path": str(repo / "missing-worktree"),
                "file_scope": ["aragora/swarm/supervisor.py"],
                "expected_tests": ["python -m pytest tests/swarm/test_supervisor.py -q"],
                "receipt_id": "receipt-stale",
                "confidence": 0.91,
                "worker_outcome": "crash_with_salvage",
                "completed_at": "2026-03-31T12:34:56+00:00",
                "initial_head": "old-base-sha",
                "head_sha": "old-head-sha",
                "commit_shas": ["old-commit"],
                "changed_paths": ["aragora/swarm/supervisor.py"],
                "diff": "diff --git a/aragora/swarm/supervisor.py b/aragora/swarm/supervisor.py",
                "stdout_tail": "worker output",
                "stderr_tail": "worker stderr",
                "tests_run": ["python -m pytest tests/swarm/test_supervisor.py -q"],
                "verification_results": [{"command": "pytest", "passed": True}],
                "merge_gate": {"checks_passed": True},
                "verification_missing_reason": "old missing verification",
                "pr_url": "https://github.com/synaptent/aragora/pull/9999",
                "adopted_pr": "https://github.com/synaptent/aragora/pull/9999",
                "last_observed_at": "2026-03-31T12:35:00+00:00",
                "last_progress_at": "2026-03-31T12:35:00+00:00",
                "progress_fingerprint": {
                    "head_sha": "old-head-sha",
                    "changed_paths": ["aragora/swarm/supervisor.py"],
                    "diff_lines": 12,
                },
                "conflicts": [
                    {
                        "source": "lease",
                        "lease_id": "old-conflict-lease",
                        "worktree_path": str(repo / "missing-conflict-worktree"),
                    }
                ],
            }
        ],
        status="active",
    )

    refreshed = supervisor.refresh_run(run_record["run_id"])

    work_order = refreshed.work_orders[0]
    assert work_order["status"] == "leased"
    assert work_order["lease_id"] != "missing-lease"
    assert "conflicts" not in work_order
    assert work_order["owner_session_id"] == "swarm-released-requeue"
    for cleared_key in (
        "receipt_id",
        "confidence",
        "worker_outcome",
        "completed_at",
        "initial_head",
        "head_sha",
        "commit_shas",
        "changed_paths",
        "diff",
        "stdout_tail",
        "stderr_tail",
        "tests_run",
        "verification_results",
        "merge_gate",
        "verification_missing_reason",
        "pr_url",
        "adopted_pr",
        "last_observed_at",
        "last_progress_at",
        "progress_fingerprint",
    ):
        assert cleared_key not in work_order


def test_refresh_run_rebinds_released_work_order_to_new_active_lease(
    repo: Path, store: DevCoordinationStore
) -> None:
    session_path = repo / "wt-rebound-lease"
    session_path.mkdir()
    supervisor = SwarmSupervisor(repo_root=repo, store=store)
    old_lease = store.claim_lease(
        task_id="wo-rebound-lease",
        title="Rebind stale lane",
        owner_agent="codex",
        owner_session_id="swarm-rebound-lease",
        branch="codex/swarm-rebound-lease",
        worktree_path=str(session_path),
        claimed_paths=["aragora/swarm/supervisor.py"],
        metadata={
            "supervisor_run_id": "run-rebound-lease",
            "work_order_id": "wo-rebound-lease",
            "task_key": "run-rebound-lease:wo-rebound-lease",
        },
    )
    store.release_lease(old_lease.lease_id)
    new_lease = store.claim_lease(
        task_id="wo-rebound-lease",
        title="Rebind stale lane",
        owner_agent="codex",
        owner_session_id="swarm-rebound-lease",
        branch="codex/swarm-rebound-lease",
        worktree_path=str(session_path),
        claimed_paths=["aragora/swarm/supervisor.py"],
        metadata={
            "supervisor_run_id": "run-rebound-lease",
            "work_order_id": "wo-rebound-lease",
            "task_key": "run-rebound-lease:wo-rebound-lease",
        },
    )
    run_record = store.create_supervisor_run(
        goal="rebind released lease",
        target_branch="main",
        supervisor_agents={},
        approval_policy={},
        spec={"raw_goal": "rebind released lease"},
        metadata={"max_concurrency": 1},
        work_orders=[
            {
                "work_order_id": "wo-rebound-lease",
                "title": "Rebind stale lane",
                "description": "Rebind stale lane",
                "status": "leased",
                "target_agent": "codex",
                "reviewer_agent": "claude",
                "lease_id": old_lease.lease_id,
                "owner_session_id": "swarm-rebound-lease",
                "task_key": "run-rebound-lease:wo-rebound-lease",
                "branch": "codex/swarm-rebound-lease",
                "worktree_path": str(session_path),
                "file_scope": ["aragora/swarm/supervisor.py"],
                "expected_tests": ["python -m pytest tests/swarm/test_supervisor.py -q"],
            }
        ],
        status="active",
    )

    refreshed = supervisor.refresh_run(run_record["run_id"])

    work_order = refreshed.work_orders[0]
    assert work_order["status"] == "leased"
    assert work_order["lease_id"] == new_lease.lease_id
    assert work_order["owner_session_id"] == "swarm-rebound-lease"


def test_refresh_run_rebinds_stale_dispatched_lane_back_to_leased_without_worker_pid(
    repo: Path, store: DevCoordinationStore
) -> None:
    session_path = repo / "wt-rebound-dispatched"
    session_path.mkdir()
    supervisor = SwarmSupervisor(repo_root=repo, store=store)
    supervisor._collect_finished_results_before_reap = MagicMock(return_value=None)
    store.reap_stale_leases = MagicMock(return_value=[])
    store.reap_expired_leases = MagicMock(return_value=[])
    old_lease = store.claim_lease(
        task_id="wo-rebound-dispatched",
        title="Rebind dispatched lane",
        owner_agent="codex",
        owner_session_id="swarm-rebound-dispatched",
        branch="codex/swarm-rebound-dispatched",
        worktree_path=str(session_path),
        claimed_paths=["aragora/swarm/supervisor.py"],
        metadata={
            "supervisor_run_id": "run-rebound-dispatched",
            "work_order_id": "wo-rebound-dispatched",
            "task_key": "run-rebound-dispatched:wo-rebound-dispatched",
            "worker_pid": 12345,
        },
    )
    store.release_lease(old_lease.lease_id)
    new_lease = store.claim_lease(
        task_id="wo-rebound-dispatched",
        title="Rebind dispatched lane",
        owner_agent="codex",
        owner_session_id="swarm-rebound-dispatched",
        branch="codex/swarm-rebound-dispatched",
        worktree_path=str(session_path),
        claimed_paths=["aragora/swarm/supervisor.py"],
        metadata={
            "supervisor_run_id": "run-rebound-dispatched",
            "work_order_id": "wo-rebound-dispatched",
            "task_key": "run-rebound-dispatched:wo-rebound-dispatched",
        },
    )
    run_record = store.create_supervisor_run(
        goal="rebind stale dispatched lease",
        target_branch="main",
        supervisor_agents={},
        approval_policy={},
        spec={"raw_goal": "rebind stale dispatched lease"},
        metadata={"max_concurrency": 1},
        work_orders=[
            {
                "work_order_id": "wo-rebound-dispatched",
                "title": "Rebind dispatched lane",
                "description": "Rebind dispatched lane",
                "status": "dispatched",
                "target_agent": "codex",
                "reviewer_agent": "claude",
                "lease_id": old_lease.lease_id,
                "owner_session_id": "swarm-rebound-dispatched",
                "task_key": "run-rebound-dispatched:wo-rebound-dispatched",
                "branch": "codex/swarm-rebound-dispatched",
                "worktree_path": str(session_path),
                "file_scope": ["aragora/swarm/supervisor.py"],
                "expected_tests": ["python -m pytest tests/swarm/test_supervisor.py -q"],
                "review_status": "changes_requested",
                "pid": 12345,
                "dispatched_at": "2026-03-31T12:00:00+00:00",
                "last_observed_at": "2026-03-31T12:01:00+00:00",
                "last_progress_at": "2026-03-31T12:01:00+00:00",
                "progress_fingerprint": {
                    "head_sha": "old-head",
                    "changed_paths": ["aragora/swarm/supervisor.py"],
                    "diff_lines": 5,
                },
                "receipt_id": "receipt-stale",
                "worker_outcome": "completed",
                "completed_at": "2026-03-31T12:02:00+00:00",
                "head_sha": "old-head",
                "commit_shas": ["abc123"],
                "changed_paths": ["aragora/swarm/supervisor.py"],
                "diff_lines": 5,
                "stdout_tail": "old stdout",
                "stderr_tail": "old stderr",
                "verification_results": [{"command": "pytest", "passed": True}],
                "merge_gate": {"checks_passed": True},
                "verification_missing_reason": "missing_verification_plan",
                "dispatch_error": "old dispatch failure",
                "failure_reason": "worker_crash",
                "blocking_question": "Old question?",
                "blocker": {"reason": "worker_crash", "question": "Old question?"},
                "blockers": ["old blocker"],
                "scope_violation": {"changed_paths": ["aragora/swarm/supervisor.py"]},
            }
        ],
        status="active",
    )

    refreshed = supervisor.refresh_run(run_record["run_id"])

    work_order = refreshed.work_orders[0]
    assert work_order["status"] == "leased"
    assert work_order["lease_id"] == new_lease.lease_id
    assert work_order["owner_session_id"] == "swarm-rebound-dispatched"
    assert work_order["review_status"] == "pending"
    for cleared_key in (
        "pid",
        "dispatched_at",
        "last_observed_at",
        "last_progress_at",
        "progress_fingerprint",
        "receipt_id",
        "worker_outcome",
        "completed_at",
        "head_sha",
        "commit_shas",
        "changed_paths",
        "diff_lines",
        "stdout_tail",
        "stderr_tail",
        "verification_results",
        "merge_gate",
        "verification_missing_reason",
        "dispatch_error",
        "failure_reason",
        "blocking_question",
        "blocker",
        "blockers",
        "scope_violation",
    ):
        assert cleared_key not in work_order


def test_refresh_run_rebinds_stale_dispatched_lane_to_active_worker_without_stale_terminal_state(
    repo: Path, store: DevCoordinationStore
) -> None:
    session_path = repo / "wt-rebound-active-dispatched"
    session_path.mkdir()
    supervisor = SwarmSupervisor(repo_root=repo, store=store)
    supervisor._collect_finished_results_before_reap = MagicMock(return_value=None)
    store.reap_stale_leases = MagicMock(return_value=[])
    store.reap_expired_leases = MagicMock(return_value=[])
    old_lease = store.claim_lease(
        task_id="wo-rebound-active-dispatched",
        title="Rebind dispatched lane",
        owner_agent="codex",
        owner_session_id="swarm-rebound-active-dispatched",
        branch="codex/swarm-rebound-active-dispatched",
        worktree_path=str(session_path),
        claimed_paths=["aragora/swarm/supervisor.py"],
        metadata={
            "supervisor_run_id": "run-rebound-active-dispatched",
            "work_order_id": "wo-rebound-active-dispatched",
            "task_key": "run-rebound-active-dispatched:wo-rebound-active-dispatched",
            "worker_pid": 12345,
        },
    )
    store.release_lease(old_lease.lease_id)
    replacement_lease = store.claim_lease(
        task_id="wo-rebound-active-dispatched",
        title="Rebind dispatched lane",
        owner_agent="claude",
        owner_session_id="swarm-rebound-active-dispatched",
        branch="codex/swarm-rebound-active-dispatched",
        worktree_path=str(session_path),
        claimed_paths=["aragora/swarm/supervisor.py"],
        metadata={
            "supervisor_run_id": "run-rebound-active-dispatched",
            "work_order_id": "wo-rebound-active-dispatched",
            "task_key": "run-rebound-active-dispatched:wo-rebound-active-dispatched",
            "worker_pid": 67890,
        },
    )
    run_record = store.create_supervisor_run(
        goal="rebind stale dispatched lane to active worker",
        target_branch="main",
        supervisor_agents={},
        approval_policy={},
        spec={"raw_goal": "rebind stale dispatched lane to active worker"},
        metadata={"max_concurrency": 1},
        work_orders=[
            {
                "work_order_id": "wo-rebound-active-dispatched",
                "title": "Rebind dispatched lane",
                "description": "Rebind dispatched lane",
                "status": "dispatched",
                "target_agent": "codex",
                "reviewer_agent": "claude",
                "lease_id": old_lease.lease_id,
                "owner_session_id": "swarm-rebound-active-dispatched",
                "task_key": "run-rebound-active-dispatched:wo-rebound-active-dispatched",
                "branch": "codex/swarm-rebound-active-dispatched",
                "worktree_path": str(session_path),
                "file_scope": ["aragora/swarm/supervisor.py"],
                "review_status": "changes_requested",
                "pid": 12345,
                "receipt_id": "receipt-stale",
                "worker_outcome": "completed",
                "completed_at": "2026-03-31T12:02:00+00:00",
                "head_sha": "old-head",
                "commit_shas": ["abc123"],
                "changed_paths": ["aragora/swarm/supervisor.py"],
                "stdout_tail": "old stdout",
                "stderr_tail": "old stderr",
                "verification_results": [{"command": "pytest", "passed": True}],
                "merge_gate": {"checks_passed": True},
                "dispatch_error": "old dispatch failure",
                "failure_reason": "worker_crash",
                "blocking_question": "Old question?",
                "blocker": {"reason": "worker_crash", "question": "Old question?"},
                "blockers": ["old blocker"],
            }
        ],
        status="active",
    )

    refreshed = supervisor.refresh_run(run_record["run_id"])

    work_order = refreshed.work_orders[0]
    assert work_order["status"] == "dispatched"
    assert work_order["lease_id"] == replacement_lease.lease_id
    assert work_order["owner_session_id"] == "swarm-rebound-active-dispatched"
    assert work_order["target_agent"] == "claude"
    assert work_order["pid"] == 67890
    assert work_order["review_status"] == "pending"
    for cleared_key in (
        "receipt_id",
        "worker_outcome",
        "completed_at",
        "head_sha",
        "commit_shas",
        "changed_paths",
        "stdout_tail",
        "stderr_tail",
        "verification_results",
        "merge_gate",
        "dispatch_error",
        "failure_reason",
        "blocking_question",
        "blocker",
        "blockers",
    ):
        assert cleared_key not in work_order


def test_refresh_run_does_not_rebind_completed_lane_to_active_replacement_lease(
    repo: Path, store: DevCoordinationStore
) -> None:
    session_path = repo / "wt-terminal-rebind"
    session_path.mkdir()
    supervisor = SwarmSupervisor(repo_root=repo, store=store)
    store.reap_stale_leases = MagicMock(return_value=[])
    store.reap_expired_leases = MagicMock(return_value=[])
    old_lease = store.claim_lease(
        task_id="wo-terminal",
        title="Completed lane",
        owner_agent="codex",
        owner_session_id="swarm-terminal",
        branch="codex/swarm-terminal",
        worktree_path=str(session_path),
        claimed_paths=["aragora/swarm/supervisor.py"],
        metadata={
            "supervisor_run_id": "run-terminal",
            "work_order_id": "wo-terminal",
            "task_key": "run-terminal:wo-terminal",
        },
    )
    store.release_lease(old_lease.lease_id)
    replacement = store.claim_lease(
        task_id="wo-terminal",
        title="Completed lane",
        owner_agent="claude",
        owner_session_id="swarm-terminal",
        branch="codex/swarm-terminal",
        worktree_path=str(session_path),
        claimed_paths=["aragora/swarm/supervisor.py"],
        metadata={
            "supervisor_run_id": "run-terminal",
            "work_order_id": "wo-terminal",
            "task_key": "run-terminal:wo-terminal",
        },
    )
    run_record = store.create_supervisor_run(
        goal="completed lane should remain terminal",
        target_branch="main",
        supervisor_agents={},
        approval_policy={},
        spec={"raw_goal": "completed lane should remain terminal"},
        metadata={"max_concurrency": 1},
        work_orders=[
            {
                "work_order_id": "wo-terminal",
                "title": "Completed lane",
                "description": "Already done",
                "status": "completed",
                "target_agent": "codex",
                "reviewer_agent": "claude",
                "lease_id": old_lease.lease_id,
                "owner_session_id": "swarm-terminal",
                "task_key": "run-terminal:wo-terminal",
                "branch": "codex/swarm-terminal",
                "worktree_path": str(session_path),
                "file_scope": ["aragora/swarm/supervisor.py"],
                "receipt_id": "receipt-123",
                "commit_shas": ["abc123"],
                "changed_paths": ["aragora/swarm/supervisor.py"],
                "head_sha": "abc123",
                "worker_outcome": "completed",
            }
        ],
        status="completed",
    )

    refreshed = supervisor.refresh_run(run_record["run_id"])

    work_order = refreshed.work_orders[0]
    assert work_order["status"] == "completed"
    assert work_order["lease_id"] == old_lease.lease_id
    assert work_order["owner_session_id"] == "swarm-terminal"
    assert work_order["target_agent"] == "codex"
    assert work_order["receipt_id"] == "receipt-123"
    assert work_order["commit_shas"] == ["abc123"]
    assert work_order["changed_paths"] == ["aragora/swarm/supervisor.py"]
    assert replacement.lease_id in {lease.lease_id for lease in store.list_active_leases()}


def test_refresh_run_respects_dispatched_workers_as_active(
    repo: Path, store: DevCoordinationStore
) -> None:
    """Dispatched workers must count toward max_concurrency.

    Before the fix, refresh_run only counted 'leased' orders as active,
    allowing new orders to be leased even when dispatched workers were still
    running. This caused more workers than max_concurrency to run in parallel.
    """
    counter = {"value": 0}

    def _ensure_session(**_kwargs: object) -> ManagedWorktreeSession:
        counter["value"] += 1
        path = repo / f"wt-conc-{counter['value']}"
        path.mkdir(exist_ok=True)
        return ManagedWorktreeSession(
            session_id=f"swarm-conc-{counter['value']}",
            agent="codex",
            branch=f"codex/swarm-conc-{counter['value']}",
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
                id=f"wo-{i}", title=f"Lane {i}", description=f"Lane {i}", file_scope=[f"file{i}.py"]
            )
            for i in range(3)
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
    assert len(leased) == 1, "start_run should lease exactly 1 work order"

    # Simulate dispatch: change leased → dispatched (what dispatch_workers does)
    run_record = store.get_supervisor_run(run.run_id)
    for item in run_record["work_orders"]:
        if item["status"] == "leased":
            item["status"] = "dispatched"
    store.update_supervisor_run(
        run.run_id,
        work_orders=run_record["work_orders"],
    )

    # refresh_run should NOT lease another order because max_concurrency=1
    # and there's already 1 dispatched worker.
    refreshed = supervisor.refresh_run(run.run_id)
    dispatched = [item for item in refreshed.work_orders if item["status"] == "dispatched"]
    newly_leased = [item for item in refreshed.work_orders if item["status"] == "leased"]
    queued = [item for item in refreshed.work_orders if item["status"] == "queued"]

    assert len(dispatched) == 1, "dispatched worker should remain dispatched"
    assert len(newly_leased) == 0, (
        "no new orders should be leased while a dispatched worker is active"
    )
    assert len(queued) == 2, "remaining orders should stay queued"


def test_refresh_run_releases_orphaned_conflicts_and_retries(
    repo: Path, store: DevCoordinationStore
) -> None:
    missing_path = repo / "missing-docs-worktree"
    orphaned = store.claim_lease(
        task_id="old-docs-lane",
        title="Old docs lane",
        owner_agent="codex",
        owner_session_id="stale-session",
        branch="codex/stale-docs",
        worktree_path=str(missing_path),
        claimed_paths=["docs/guides/SWARM_DOGFOOD_OPERATOR.md"],
    )

    lifecycle = MagicMock()
    session_path = repo / "wt-docs"
    session_path.mkdir()
    lifecycle.ensure_managed_worktree.return_value = ManagedWorktreeSession(
        session_id="fresh-docs",
        agent="codex",
        branch="codex/fresh-docs",
        path=session_path,
        created=True,
        reconcile_status="up_to_date",
        payload={},
    )

    decomposer = MagicMock()
    decomposer.analyze.return_value = TaskDecomposition(
        original_task="Goal",
        complexity_score=2,
        complexity_level="low",
        should_decompose=True,
        subtasks=[
            SubTask(
                id="docs-lane",
                title="Write operator guide",
                description="Add operator guide.",
                file_scope=["docs/guides/SWARM_DOGFOOD_OPERATOR.md"],
            )
        ],
    )

    supervisor = SwarmSupervisor(
        repo_root=repo,
        store=store,
        lifecycle=lifecycle,
        decomposer=decomposer,
    )
    run = supervisor.start_run(spec=SwarmSpec(raw_goal="Goal", refined_goal="Goal"))

    work_order = run.work_orders[0]
    assert work_order["status"] == "leased"
    assert work_order["lease_id"]
    active_lease_ids = {lease.lease_id for lease in store.list_active_leases()}
    assert orphaned.lease_id not in active_lease_ids


def test_refresh_run_releases_managed_conflicts_without_active_session(
    repo: Path, store: DevCoordinationStore
) -> None:
    stale_path = repo / ".worktrees" / "codex-auto" / "swarm-stale-docs"
    stale_path.mkdir(parents=True)
    (stale_path / ".swarm_worker_stderr.log").write_text("old worker log\n", encoding="utf-8")
    orphaned = store.claim_lease(
        task_id="old-docs-lane",
        title="Old docs lane",
        owner_agent="codex",
        owner_session_id="swarm-stale-docs",
        branch="codex/stale-docs",
        worktree_path=str(stale_path),
        claimed_paths=["docs/guides/SWARM_DOGFOOD_OPERATOR.md"],
    )

    lifecycle = MagicMock()
    session_path = repo / "wt-docs-fresh"
    session_path.mkdir()
    lifecycle.ensure_managed_worktree.return_value = ManagedWorktreeSession(
        session_id="fresh-docs",
        agent="codex",
        branch="codex/fresh-docs",
        path=session_path,
        created=True,
        reconcile_status="up_to_date",
        payload={},
    )

    decomposer = MagicMock()
    decomposer.analyze.return_value = TaskDecomposition(
        original_task="Goal",
        complexity_score=2,
        complexity_level="low",
        should_decompose=True,
        subtasks=[
            SubTask(
                id="docs-lane",
                title="Write operator guide",
                description="Add operator guide.",
                file_scope=["docs/guides/SWARM_DOGFOOD_OPERATOR.md"],
            )
        ],
    )

    supervisor = SwarmSupervisor(
        repo_root=repo,
        store=store,
        lifecycle=lifecycle,
        decomposer=decomposer,
    )
    run = supervisor.start_run(spec=SwarmSpec(raw_goal="Goal", refined_goal="Goal"))

    work_order = run.work_orders[0]
    assert work_order["status"] == "leased"
    assert work_order["lease_id"]
    active_lease_ids = {lease.lease_id for lease in store.list_active_leases()}
    assert orphaned.lease_id not in active_lease_ids


def test_refresh_run_keeps_conflict_for_live_managed_session(
    repo: Path, store: DevCoordinationStore
) -> None:
    live_path = repo / ".worktrees" / "codex-auto" / "swarm-live-docs"
    live_path.mkdir(parents=True)
    (live_path / ".codex_session_active").write_text(
        f"pid={os.getpid()}\nsession_id=swarm-live-docs\n",
        encoding="utf-8",
    )
    blocking = store.claim_lease(
        task_id="old-docs-lane",
        title="Old docs lane",
        owner_agent="codex",
        owner_session_id="swarm-live-docs",
        branch="codex/stale-docs",
        worktree_path=str(live_path),
        claimed_paths=["docs/guides/SWARM_DOGFOOD_OPERATOR.md"],
    )

    lifecycle = MagicMock()
    session_path = repo / "wt-docs-fresh-live"
    session_path.mkdir()
    lifecycle.ensure_managed_worktree.return_value = ManagedWorktreeSession(
        session_id="fresh-docs",
        agent="codex",
        branch="codex/fresh-docs",
        path=session_path,
        created=True,
        reconcile_status="up_to_date",
        payload={},
    )

    decomposer = MagicMock()
    decomposer.analyze.return_value = TaskDecomposition(
        original_task="Goal",
        complexity_score=2,
        complexity_level="low",
        should_decompose=True,
        subtasks=[
            SubTask(
                id="docs-lane",
                title="Write operator guide",
                description="Add operator guide.",
                file_scope=["docs/guides/SWARM_DOGFOOD_OPERATOR.md"],
            )
        ],
    )

    supervisor = SwarmSupervisor(
        repo_root=repo,
        store=store,
        lifecycle=lifecycle,
        decomposer=decomposer,
    )
    run = supervisor.start_run(spec=SwarmSpec(raw_goal="Goal", refined_goal="Goal"))

    work_order = run.work_orders[0]
    assert work_order["status"] == "waiting_conflict"
    assert not work_order["lease_id"]
    assert work_order["conflicts"][0]["lease_id"] == blocking.lease_id
    assert work_order["failure_reason"] == "waiting_conflict"
    assert "overlapping lane" in work_order["blocking_question"]
    assert work_order["blocker"]["reason"] == "waiting_conflict"
    assert any(
        str(entry).startswith("scope already claimed:") for entry in work_order.get("blockers", [])
    )
    active_lease_ids = {lease.lease_id for lease in store.list_active_leases()}
    assert blocking.lease_id in active_lease_ids


def test_refresh_run_marks_resource_wait_on_disk_full(
    repo: Path, store: DevCoordinationStore
) -> None:
    lifecycle = MagicMock()
    lifecycle.ensure_managed_worktree.side_effect = RuntimeError("No space left on device")
    decomposer = MagicMock()
    decomposer.analyze.return_value = TaskDecomposition(
        original_task="Goal",
        complexity_score=2,
        complexity_level="low",
        should_decompose=True,
        subtasks=[
            SubTask(
                id="cli-tests-lane",
                title="Run CLI coverage",
                description="Add CLI coverage.",
                file_scope=["tests/cli/test_swarm_command.py"],
            )
        ],
    )

    supervisor = SwarmSupervisor(
        repo_root=repo,
        store=store,
        lifecycle=lifecycle,
        decomposer=decomposer,
    )
    run = supervisor.start_run(spec=SwarmSpec(raw_goal="Goal", refined_goal="Goal"))

    # waiting_resource with no forward-progress path escalates to needs_human
    # (fixes #883 deadlock where dead-end statuses kept the run active).
    assert run.status == "needs_human"
    work_order = run.work_orders[0]
    assert work_order["status"] == "waiting_resource"
    assert "No space left on device" in work_order["resource_error"]


def test_start_run_prefers_explicit_spec_work_orders(
    repo: Path, store: DevCoordinationStore
) -> None:
    session_path = repo / "wt-explicit"
    session_path.mkdir()
    lifecycle = MagicMock()
    lifecycle.ensure_managed_worktree.return_value = ManagedWorktreeSession(
        session_id="swarm-explicit",
        agent="codex",
        branch="codex/swarm-explicit",
        path=session_path,
        created=True,
        reconcile_status="up_to_date",
        payload={},
    )
    decomposer = MagicMock()

    supervisor = SwarmSupervisor(
        repo_root=repo,
        store=store,
        lifecycle=lifecycle,
        decomposer=decomposer,
    )
    spec = SwarmSpec(
        raw_goal="Dogfood the supervised swarm",
        refined_goal="Dogfood the supervised swarm",
        acceptance_criteria=["python -m pytest tests/swarm/test_commander.py -q"],
        work_orders=[
            {
                "work_order_id": "docs-lane",
                "title": "Write operator guide",
                "description": "Add the operator guide.",
                "file_scope": ["docs/guides/SWARM_DOGFOOD_OPERATOR.md"],
                "expected_tests": [],
                "target_agent": "codex",
                "reviewer_agent": "claude",
                "metadata": {"lane": "docs"},
            }
        ],
    )

    run = supervisor.start_run(spec=spec, max_concurrency=1)

    decomposer.analyze.assert_not_called()
    assert run.status == "active"
    assert len(run.work_orders) == 1
    work_order = run.work_orders[0]
    assert work_order["work_order_id"] == "docs-lane"
    assert work_order["target_agent"] == "codex"
    assert work_order["reviewer_agent"] == "claude"
    assert work_order["file_scope"] == ["docs/guides/SWARM_DOGFOOD_OPERATOR.md"]
    assert work_order["metadata"]["lane"] == "docs"
    assert work_order["metadata"]["source"] == "explicit_spec_work_order"


def test_explicit_work_orders_merge_spec_file_scope_hints(
    repo: Path, store: DevCoordinationStore
) -> None:
    """Regression test for #884: explicit work orders with empty file_scope
    must inherit spec.file_scope_hints so workers stay in the intended surface."""
    session_path = repo / "wt-scope-merge"
    session_path.mkdir()
    lifecycle = MagicMock()
    lifecycle.ensure_managed_worktree.return_value = ManagedWorktreeSession(
        session_id="swarm-scope-merge",
        agent="codex",
        branch="codex/swarm-scope-merge",
        path=session_path,
        created=True,
        reconcile_status="up_to_date",
        payload={},
    )
    decomposer = MagicMock()

    supervisor = SwarmSupervisor(
        repo_root=repo,
        store=store,
        lifecycle=lifecycle,
        decomposer=decomposer,
    )
    spec = SwarmSpec(
        raw_goal="Bump @eslint/eslintrc in /aragora/live",
        refined_goal="Bump @eslint/eslintrc from 3.2.0 to 3.3.0 in /aragora/live",
        file_scope_hints=["aragora/live"],
        work_orders=[
            {
                "work_order_id": "bump-eslint",
                "title": "Bump eslintrc dependency",
                "description": "Update @eslint/eslintrc to 3.3.0",
                "file_scope": [],  # Empty — must be backfilled from spec hints
                "target_agent": "codex",
                "reviewer_agent": "claude",
            }
        ],
    )

    run = supervisor.start_run(spec=spec, max_concurrency=1)

    assert len(run.work_orders) == 1
    work_order = run.work_orders[0]
    assert work_order["file_scope"] == ["aragora/live"], (
        "Empty file_scope on explicit work order must be backfilled from spec.file_scope_hints"
    )


def test_start_run_narrows_broad_scope_when_goal_names_specific_doc_path(
    repo: Path, store: DevCoordinationStore
) -> None:
    lifecycle = MagicMock()
    decomposer = MagicMock()
    decomposer.analyze.return_value = TaskDecomposition(
        original_task="Goal",
        complexity_score=2,
        complexity_level="low",
        should_decompose=True,
        subtasks=[
            SubTask(
                id="subtask_1",
                title="Write governance ADR",
                description="Write docs/governance/duplicate-subsystem-resolution.md.",
                file_scope=["docs/governance/"],
            )
        ],
    )
    supervisor = SwarmSupervisor(
        repo_root=repo,
        store=store,
        lifecycle=lifecycle,
        decomposer=decomposer,
    )

    run = supervisor.start_run(
        spec=SwarmSpec(
            raw_goal="Write docs/governance/duplicate-subsystem-resolution.md with the resolution plan.",
            refined_goal="Write docs/governance/duplicate-subsystem-resolution.md with the resolution plan.",
        ),
        refresh_scaling=False,
    )

    assert run.work_orders[0]["file_scope"] == ["docs/governance/duplicate-subsystem-resolution.md"]


def test_start_run_narrows_explicit_spec_broad_scope_when_description_names_specific_path(
    repo: Path, store: DevCoordinationStore
) -> None:
    supervisor = SwarmSupervisor(
        repo_root=repo,
        store=store,
        lifecycle=MagicMock(),
        decomposer=MagicMock(),
    )

    spec = SwarmSpec(
        raw_goal="Write docs/plans/phase0b_campaign_manifest.yaml from the bootstrap plan.",
        refined_goal="Write docs/plans/phase0b_campaign_manifest.yaml from the bootstrap plan.",
        work_orders=[
            {
                "work_order_id": "manifest-lane",
                "title": "Manifest lane",
                "description": "Write docs/plans/phase0b_campaign_manifest.yaml from the bootstrap plan.",
                "file_scope": ["docs/"],
                "target_agent": "codex",
                "reviewer_agent": "claude",
            }
        ],
    )

    run = supervisor.start_run(spec=spec, refresh_scaling=False)

    assert run.work_orders[0]["file_scope"] == ["docs/plans/phase0b_campaign_manifest.yaml"]


def test_start_run_narrows_docs_only_scope_to_doc_hints(
    repo: Path, store: DevCoordinationStore
) -> None:
    (repo / "docs" / "ADR").mkdir(parents=True, exist_ok=True)

    lifecycle = MagicMock()
    decomposer = MagicMock()
    decomposer.analyze.return_value = TaskDecomposition(
        original_task="Goal",
        complexity_score=2,
        complexity_level="low",
        should_decompose=True,
        subtasks=[
            SubTask(
                id="subtask_1",
                title="Improve Developer Track",
                description="Enhance capabilities in the Developer track. Key folders: sdk/, docs/, tests/sdk/.",
                file_scope=["sdk/", "docs/", "tests/sdk/"],
            )
        ],
    )
    supervisor = SwarmSupervisor(
        repo_root=repo,
        store=store,
        lifecycle=lifecycle,
        decomposer=decomposer,
    )

    run = supervisor.start_run(
        spec=SwarmSpec(
            raw_goal="Write the worker-model ADR with canonical command, deploy mapping, and compatibility notes.",
            refined_goal="Write the worker-model ADR with canonical command, deploy mapping, and compatibility notes.",
            acceptance_criteria=["ADR committed under docs/ADR"],
            constraints=["Documentation only"],
        ),
        refresh_scaling=False,
    )

    assert run.work_orders[0]["file_scope"] == ["docs/ADR"]


def test_start_run_drops_non_actionable_explicit_spec_validation_lane(
    repo: Path, store: DevCoordinationStore
) -> None:
    supervisor = SwarmSupervisor(
        repo_root=repo,
        store=store,
        lifecycle=MagicMock(),
        decomposer=MagicMock(),
    )

    spec = SwarmSpec(
        raw_goal="Define the founder-facing PMF scorecard and roadmap dependency map.",
        refined_goal="Define the founder-facing PMF scorecard and roadmap dependency map.",
        file_scope_hints=["ROADMAP.md", "docs/plans/**", "docs/strategy/**"],
        work_orders=[
            {
                "work_order_id": "pmf-scorecard",
                "title": "Define the founder-facing PMF scorecard, evidence thresholds, and weekly operating cadence for Aragora's current wedge.",
                "description": "Produce one concrete PMF scorecard artifact.",
                "file_scope": ["ROADMAP.md", "docs/plans/**", "docs/strategy/**"],
                "target_agent": "codex",
                "reviewer_agent": "claude",
            },
            {
                "work_order_id": "proj-001",
                "title": "Validation Changes",
                "description": "## Validation\n\n- Changed files stay within the allowed scope",
                "file_scope": ["ROADMAP.md", "docs/plans/**", "docs/strategy/**"],
                "success_criteria": {"notes": "Complete bounded task: Validation Changes"},
                "target_agent": "claude",
                "reviewer_agent": "codex",
            },
        ],
    )

    run = supervisor.start_run(spec=spec, refresh_scaling=False)

    assert [item["work_order_id"] for item in run.work_orders] == ["pmf-scorecard"]
    assert run.work_orders[0]["metadata"]["source"] == "explicit_spec_work_order"


def test_start_run_drops_explicit_spec_umbrella_lane_when_specific_sibling_exists(
    repo: Path, store: DevCoordinationStore
) -> None:
    supervisor = SwarmSupervisor(
        repo_root=repo,
        store=store,
        lifecycle=MagicMock(),
        decomposer=MagicMock(),
    )

    spec = SwarmSpec(
        raw_goal=(
            "Make one already reachable core page functional with real data flow, empty-state "
            "handling, and tests, choosing the smallest verifiable page from the issue instead "
            "of broad page churn."
        ),
        refined_goal=(
            "Make one already reachable core page functional with real data flow, empty-state "
            "handling, and tests, choosing the smallest verifiable page from the issue instead "
            "of broad page churn."
        ),
        work_orders=[
            {
                "work_order_id": "core-pages-functional-slice",
                "title": (
                    "Make one already reachable core page functional with real data flow, "
                    "empty-state handling, and tests, choosing the smallest verifiable page "
                    "from the issue instead of broad page churn."
                ),
                "description": (
                    "Make one already reachable core page functional with real data flow, "
                    "empty-state handling, and tests, choosing the smallest verifiable page "
                    "from the issue instead of broad page churn."
                ),
                "file_scope": ["aragora/live/**", "tests/e2e/**", "tests/handlers/**", "docs/**"],
            },
            {
                "work_order_id": "proj-001",
                "title": "Implement functional Results page with real data flow",
                "description": (
                    "Connect the results page to backend endpoints to display debate outcomes."
                ),
                "file_scope": ["aragora/live/**", "tests/e2e/**", "tests/handlers/**", "docs/**"],
            },
        ],
    )

    run = supervisor.start_run(spec=spec, refresh_scaling=False)

    assert [item["work_order_id"] for item in run.work_orders] == ["proj-001"]


def test_start_run_preserves_broad_scope_without_explicit_path_hints(
    repo: Path, store: DevCoordinationStore
) -> None:
    lifecycle = MagicMock()
    decomposer = MagicMock()
    decomposer.analyze.return_value = TaskDecomposition(
        original_task="Goal",
        complexity_score=2,
        complexity_level="low",
        should_decompose=True,
        subtasks=[
            SubTask(
                id="subtask_1",
                title="Write governance ADR",
                description="Write the governance ADR.",
                file_scope=["docs/governance/"],
            )
        ],
    )
    supervisor = SwarmSupervisor(
        repo_root=repo,
        store=store,
        lifecycle=lifecycle,
        decomposer=decomposer,
    )

    run = supervisor.start_run(
        spec=SwarmSpec(
            raw_goal="Write the governance ADR.",
            refined_goal="Write the governance ADR.",
        ),
        refresh_scaling=False,
    )

    assert run.work_orders[0]["file_scope"] == ["docs/governance/"]


def test_start_run_fails_closed_when_work_order_scope_remains_empty(
    repo: Path, store: DevCoordinationStore
) -> None:
    lifecycle = MagicMock()
    decomposer = MagicMock()
    decomposer.analyze.return_value = TaskDecomposition(
        original_task="Document follow-up",
        complexity_score=1,
        complexity_level="low",
        should_decompose=False,
        subtasks=[
            SubTask(
                id="wo-empty-scope",
                title="Analysis Changes",
                description="Investigate the issue and summarize next steps.",
                file_scope=[],
            )
        ],
    )

    supervisor = SwarmSupervisor(
        repo_root=repo,
        store=store,
        lifecycle=lifecycle,
        decomposer=decomposer,
    )

    run = supervisor.start_run(
        spec=SwarmSpec(raw_goal="Document follow-up", refined_goal="Document follow-up"),
        max_concurrency=1,
    )

    lifecycle.ensure_managed_worktree.assert_not_called()
    assert run.status == "needs_human"
    assert store.status_summary()["counts"]["active_leases"] == 0
    work_order = run.work_orders[0]
    assert work_order["status"] == "needs_human"
    assert work_order["failure_reason"] == "scope_violation"
    assert (
        work_order["dispatch_error"]
        == "Work order has no declared file scope; declare scope before dispatch."
    )
    assert work_order["lease_id"] is None


def test_start_run_fails_closed_when_validated_scope_resolves_to_empty(
    repo: Path, store: DevCoordinationStore
) -> None:
    session_path = repo / "wt-invalid-scope"
    session_path.mkdir()
    (session_path / ".git").write_text("gitdir: /tmp/fake\n", encoding="utf-8")
    lifecycle = MagicMock()
    lifecycle.ensure_managed_worktree.return_value = ManagedWorktreeSession(
        session_id="swarm-invalid-scope",
        agent="codex",
        branch="codex/swarm-invalid-scope",
        path=session_path,
        created=True,
        reconcile_status="up_to_date",
        payload={},
    )
    decomposer = MagicMock()
    decomposer.analyze.return_value = TaskDecomposition(
        original_task="Fix bug",
        complexity_score=1,
        complexity_level="low",
        should_decompose=False,
        subtasks=[
            SubTask(
                id="wo-invalid-scope",
                title="Fix bug",
                description="Repair the issue",
                file_scope=["src/not-real.py"],
            )
        ],
    )

    supervisor = SwarmSupervisor(
        repo_root=repo,
        store=store,
        lifecycle=lifecycle,
        decomposer=decomposer,
    )

    run = supervisor.start_run(
        spec=SwarmSpec(raw_goal="Fix bug", refined_goal="Fix bug"),
        max_concurrency=1,
    )

    lifecycle.ensure_managed_worktree.assert_called_once()
    assert run.status == "needs_human"
    assert store.status_summary()["counts"]["active_leases"] == 0
    work_order = run.work_orders[0]
    assert work_order["status"] == "needs_human"
    assert work_order["file_scope"] == []
    assert work_order["failure_reason"] == "scope_violation"
    assert (
        work_order["dispatch_error"]
        == "Declared file scope resolved to no valid in-repo paths; declare scope before dispatch."
    )
    assert work_order["lease_id"] is None


def test_explicit_work_orders_merge_spec_hints_into_existing_scope(
    repo: Path, store: DevCoordinationStore
) -> None:
    """When an explicit work order already has a file_scope, spec hints are merged in."""
    session_path = repo / "wt-scope-merge2"
    session_path.mkdir()
    lifecycle = MagicMock()
    lifecycle.ensure_managed_worktree.return_value = ManagedWorktreeSession(
        session_id="swarm-scope-merge2",
        agent="codex",
        branch="codex/swarm-scope-merge2",
        path=session_path,
        created=True,
        reconcile_status="up_to_date",
        payload={},
    )
    decomposer = MagicMock()

    supervisor = SwarmSupervisor(
        repo_root=repo,
        store=store,
        lifecycle=lifecycle,
        decomposer=decomposer,
    )
    spec = SwarmSpec(
        raw_goal="Fix live frontend",
        refined_goal="Fix live frontend linting",
        file_scope_hints=["aragora/live", "tests/live"],
        work_orders=[
            {
                "work_order_id": "fix-lint",
                "title": "Fix linting",
                "description": "Fix eslint config",
                "file_scope": ["aragora/live/package.json"],
                "target_agent": "codex",
                "reviewer_agent": "claude",
            }
        ],
    )

    run = supervisor.start_run(spec=spec, max_concurrency=1)

    work_order = run.work_orders[0]
    assert "aragora/live/package.json" in work_order["file_scope"]
    assert "aragora/live" in work_order["file_scope"]
    assert "tests/live" in work_order["file_scope"]


def test_start_run_collapses_redundant_identical_scope_work_orders(
    repo: Path, store: DevCoordinationStore
) -> None:
    session_path = repo / "wt-collapsed"
    session_path.mkdir()
    lifecycle = MagicMock()
    lifecycle.ensure_managed_worktree.return_value = ManagedWorktreeSession(
        session_id="swarm-collapsed",
        agent="codex",
        branch="codex/swarm-collapsed",
        path=session_path,
        created=True,
        reconcile_status="up_to_date",
        payload={},
    )
    decomposer = MagicMock()
    decomposer.analyze.return_value = TaskDecomposition(
        original_task="Add quickstart json output",
        complexity_score=4,
        complexity_level="moderate",
        should_decompose=True,
        subtasks=[
            SubTask(
                id="wo-cli",
                title="CLI Changes",
                description="Update quickstart command output.",
                file_scope=["aragora/cli/commands/quickstart.py", "tests/cli/test_quickstart.py"],
            ),
            SubTask(
                id="wo-tests",
                title="Tests Changes",
                description="Add JSON output regression tests.",
                file_scope=["aragora/cli/commands/quickstart.py", "tests/cli/test_quickstart.py"],
            ),
        ],
    )
    supervisor = SwarmSupervisor(
        repo_root=repo,
        store=store,
        lifecycle=lifecycle,
        decomposer=decomposer,
    )
    spec = SwarmSpec(
        raw_goal="Add quickstart json output",
        refined_goal="Add --json output to quickstart",
        file_scope_hints=["aragora/cli/commands/quickstart.py", "tests/cli/test_quickstart.py"],
    )

    run = supervisor.start_run(spec=spec, max_concurrency=1)

    assert len(run.work_orders) == 1
    work_order = run.work_orders[0]
    assert work_order["status"] == "leased"
    assert work_order["file_scope"] == [
        "aragora/cli/commands/quickstart.py",
        "tests/cli/test_quickstart.py",
    ]
    assert work_order["metadata"]["collapsed_redundant_work_orders"] == ["wo-cli", "wo-tests"]
    lifecycle.ensure_managed_worktree.assert_called_once()


def test_start_run_single_lane_uses_clean_goal_description(
    repo: Path, store: DevCoordinationStore
) -> None:
    session_path = repo / "wt-single-clean"
    session_path.mkdir()
    lifecycle = MagicMock()
    lifecycle.ensure_managed_worktree.return_value = ManagedWorktreeSession(
        session_id="swarm-single-clean",
        agent="codex",
        branch="codex/swarm-single-clean",
        path=session_path,
        created=True,
        reconcile_status="up_to_date",
        payload={},
    )
    decomposer = MagicMock()
    decomposer.analyze.return_value = TaskDecomposition(
        original_task="Original goal",
        complexity_score=1,
        complexity_level="low",
        should_decompose=True,
        subtasks=[
            SubTask(
                id="wo-single",
                title="Planner lane",
                description="Planner-specific decomposition text",
                file_scope=["aragora/swarm/supervisor.py"],
            )
        ],
    )
    supervisor = SwarmSupervisor(
        repo_root=repo,
        store=store,
        lifecycle=lifecycle,
        decomposer=decomposer,
    )
    spec = SwarmSpec(
        raw_goal="Original goal",
        refined_goal="Use the cleaned boss-loop goal text",
        file_scope_hints=["aragora/swarm/supervisor.py"],
    )

    run = supervisor.start_run(spec=spec, max_concurrency=1)

    assert len(run.work_orders) == 1
    assert run.work_orders[0]["description"] == "Use the cleaned boss-loop goal text"


def test_start_run_preserves_distinct_scope_work_orders(
    repo: Path, store: DevCoordinationStore
) -> None:
    sessions = [
        ManagedWorktreeSession(
            session_id="swarm-distinct-a",
            agent="codex",
            branch="codex/swarm-distinct-a",
            path=repo / "wt-distinct-a",
            created=True,
            reconcile_status="up_to_date",
            payload={},
        ),
        ManagedWorktreeSession(
            session_id="swarm-distinct-b",
            agent="claude",
            branch="codex/swarm-distinct-b",
            path=repo / "wt-distinct-b",
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
        original_task="Fix CLI and server",
        complexity_score=5,
        complexity_level="moderate",
        should_decompose=True,
        subtasks=[
            SubTask(
                id="wo-cli",
                title="CLI Changes",
                description="Update quickstart command output.",
                file_scope=["aragora/cli/commands/quickstart.py"],
            ),
            SubTask(
                id="wo-server",
                title="Server Changes",
                description="Update receipt route output.",
                file_scope=["aragora/server/fastapi/receipts.py"],
            ),
        ],
    )
    supervisor = SwarmSupervisor(
        repo_root=repo,
        store=store,
        lifecycle=lifecycle,
        decomposer=decomposer,
    )
    spec = SwarmSpec(
        raw_goal="Fix CLI and server",
        refined_goal="Fix CLI and server",
        file_scope_hints=[
            "aragora/cli/commands/quickstart.py",
            "aragora/server/fastapi/receipts.py",
        ],
    )

    run = supervisor.start_run(spec=spec, max_concurrency=2)

    assert len(run.work_orders) == 2
    assert lifecycle.ensure_managed_worktree.call_count == 2


def test_start_run_initializes_worker_type_circuit_breaker_metadata(
    repo: Path, store: DevCoordinationStore
) -> None:
    session_path = repo / "wt-breaker-init"
    session_path.mkdir()
    lifecycle = MagicMock()
    lifecycle.ensure_managed_worktree.return_value = ManagedWorktreeSession(
        session_id="swarm-breaker-init",
        agent="codex",
        branch="codex/swarm-breaker-init",
        path=session_path,
        created=True,
        reconcile_status="up_to_date",
        payload={},
    )
    decomposer = MagicMock()
    decomposer.analyze.return_value = TaskDecomposition(
        original_task="Goal",
        complexity_score=2,
        complexity_level="low",
        should_decompose=True,
        subtasks=[
            SubTask(
                id="breaker-init",
                title="Initialize breaker metadata",
                description="Seed default breaker metadata on the run.",
                file_scope=["tests/swarm/test_supervisor.py"],
            )
        ],
    )

    supervisor = SwarmSupervisor(
        repo_root=repo,
        store=store,
        lifecycle=lifecycle,
        decomposer=decomposer,
    )

    run = supervisor.start_run(spec=SwarmSpec(raw_goal="Goal", refined_goal="Goal"))

    assert run.metadata[WORKER_TYPE_CIRCUIT_BREAKER_POLICY_KEY] == {
        "failure_threshold": 2,
        "reset_timeout_seconds": 900.0,
    }
    assert run.metadata[WORKER_TYPE_CIRCUIT_BREAKERS_KEY] == {}


# ---------- dispatch_workers / collect_results tests ----------

UTC = timezone.utc


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
                file_scope=["README.md"],
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
                "expected_tests": ["python -m pytest tests/swarm/test_supervisor.py -q"],
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
        changed_paths=["test.py"],
        commit_shas=["abc123"],
        stdout="worker stdout\n" + ("a" * 5000),
        stderr="worker stderr\n",
        tests_run=["python -m pytest tests/swarm/test_supervisor.py -q"],
        verification_results=[
            {
                "command": "python -m pytest tests/swarm/test_supervisor.py -q",
                "exit_code": 0,
                "passed": True,
                "stdout": "1 passed",
                "stderr": "",
                "duration_seconds": 0.2,
            }
        ],
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
    assert wo["stdout_tail"] == completed_worker.stdout[-4000:]
    assert wo["stderr_tail"] == "worker stderr\n"


@pytest.mark.asyncio
async def test_collect_results_records_passed_merge_gate_checks(
    repo: Path, store: DevCoordinationStore
) -> None:
    lease = store.claim_lease(
        task_id="merge-pass-lane",
        title="Merge pass lane",
        owner_agent="claude",
        owner_session_id="merge-pass-session",
        branch="main",
        worktree_path=str(repo),
        claimed_paths=["aragora/swarm/supervisor.py"],
        expected_tests=["python -m pytest tests/swarm/test_supervisor.py -q"],
    )
    run_record = store.create_supervisor_run(
        goal="merge gate pass",
        target_branch="main",
        supervisor_agents={},
        approval_policy={},
        spec={"raw_goal": "merge gate pass"},
        work_orders=[
            {
                "work_order_id": "wo-merge-pass",
                "status": "dispatched",
                "worktree_path": str(repo),
                "branch": "main",
                "target_agent": "claude",
                "owner_session_id": "merge-pass-session",
                "lease_id": lease.lease_id,
                "review_status": "pending",
                "file_scope": ["aragora/swarm/supervisor.py"],
                "expected_tests": ["python -m pytest tests/swarm/test_supervisor.py -q"],
                "receipt_id": None,
            }
        ],
        status="active",
    )
    run_id = run_record["run_id"]

    mock_launcher = MagicMock(spec=WorkerLauncher)
    completed_worker = WorkerProcess(
        work_order_id="wo-merge-pass",
        agent="claude",
        worktree_path=str(repo),
        branch="main",
        session_id="merge-pass-session",
        pid=100,
        exit_code=0,
        completed_at="2026-03-06T20:00:00+00:00",
        diff="diff --git a/aragora/swarm/supervisor.py",
        changed_paths=["aragora/swarm/supervisor.py"],
        commit_shas=["abc12345"],
        tests_run=["python -m pytest tests/swarm/test_supervisor.py -q"],
        verification_results=[
            {
                "command": "python -m pytest tests/swarm/test_supervisor.py -q",
                "exit_code": 0,
                "passed": True,
                "stdout": "1 passed",
                "stderr": "",
                "duration_seconds": 0.4,
            }
        ],
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

    updated = store.get_supervisor_run(run_id)
    assert updated is not None
    wo = updated["work_orders"][0]
    assert wo["status"] == "completed"
    assert wo["review_status"] == "pending_heterogeneous_review"
    assert wo["receipt_id"] is not None
    assert store.get_completion_receipt(wo["receipt_id"]) is not None
    assert wo["merge_gate"]["checks_passed"] is True
    assert wo["merge_gate"]["human_approval_required"] is True


@pytest.mark.asyncio
async def test_collect_results_blocks_merge_gate_when_required_checks_fail(
    repo: Path, store: DevCoordinationStore
) -> None:
    lease = store.claim_lease(
        task_id="merge-fail-lane",
        title="Merge fail lane",
        owner_agent="claude",
        owner_session_id="merge-fail-session",
        branch="main",
        worktree_path=str(repo),
        claimed_paths=["aragora/swarm/supervisor.py"],
        expected_tests=["python -m pytest tests/swarm/test_supervisor.py -q"],
    )
    run_record = store.create_supervisor_run(
        goal="merge gate fail",
        target_branch="main",
        supervisor_agents={},
        approval_policy={},
        spec={"raw_goal": "merge gate fail"},
        work_orders=[
            {
                "work_order_id": "wo-merge-fail",
                "status": "dispatched",
                "worktree_path": str(repo),
                "branch": "main",
                "target_agent": "claude",
                "owner_session_id": "merge-fail-session",
                "lease_id": lease.lease_id,
                "review_status": "pending",
                "file_scope": ["aragora/swarm/supervisor.py"],
                "expected_tests": ["python -m pytest tests/swarm/test_supervisor.py -q"],
            }
        ],
        status="active",
    )
    run_id = run_record["run_id"]

    mock_launcher = MagicMock(spec=WorkerLauncher)
    completed_worker = WorkerProcess(
        work_order_id="wo-merge-fail",
        agent="claude",
        worktree_path=str(repo),
        branch="main",
        session_id="merge-fail-session",
        pid=100,
        exit_code=0,
        completed_at="2026-03-06T20:00:00+00:00",
        diff="diff --git a/aragora/swarm/supervisor.py",
        changed_paths=["aragora/swarm/supervisor.py"],
        commit_shas=["abc12345"],
        tests_run=["python -m pytest tests/swarm/test_supervisor.py -q"],
        verification_results=[
            {
                "command": "python -m pytest tests/swarm/test_supervisor.py -q",
                "exit_code": 1,
                "passed": False,
                "stdout": "",
                "stderr": "FAILED tests/swarm/test_supervisor.py::test_regression",
                "duration_seconds": 0.7,
            }
        ],
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

    updated = store.get_supervisor_run(run_id)
    assert updated is not None
    wo = updated["work_orders"][0]
    assert wo["status"] == "needs_human"
    assert wo["review_status"] == "changes_requested"
    assert wo["receipt_id"] is None
    assert wo["worker_outcome"] == "merge_gate_failed"
    assert wo["merge_gate"]["checks_passed"] is False
    assert "merge gate blocked" in wo["dispatch_error"]

    summary = store.status_summary()
    assert summary["counts"]["active_leases"] == 0


@pytest.mark.asyncio
async def test_collect_results_blocks_merge_gate_without_verification_plan(
    repo: Path, store: DevCoordinationStore
) -> None:
    lease = store.claim_lease(
        task_id="merge-missing-plan",
        title="Merge missing plan lane",
        owner_agent="claude",
        owner_session_id="merge-missing-plan-session",
        branch="main",
        worktree_path=str(repo),
        claimed_paths=["aragora/swarm/supervisor.py"],
        expected_tests=[],
    )
    run_record = store.create_supervisor_run(
        goal="merge gate missing verification plan",
        target_branch="main",
        supervisor_agents={},
        approval_policy={},
        spec={"raw_goal": "merge gate missing verification plan"},
        work_orders=[
            {
                "work_order_id": "wo-merge-missing-plan",
                "status": "dispatched",
                "worktree_path": str(repo),
                "branch": "main",
                "target_agent": "claude",
                "owner_session_id": "merge-missing-plan-session",
                "lease_id": lease.lease_id,
                "review_status": "pending",
                "file_scope": ["aragora/swarm/supervisor.py"],
                "expected_tests": [],
            }
        ],
        status="active",
    )
    run_id = run_record["run_id"]

    mock_launcher = MagicMock(spec=WorkerLauncher)
    completed_worker = WorkerProcess(
        work_order_id="wo-merge-missing-plan",
        agent="claude",
        worktree_path=str(repo),
        branch="main",
        session_id="merge-missing-plan-session",
        pid=100,
        exit_code=0,
        completed_at="2026-03-06T20:00:00+00:00",
        diff="diff --git a/aragora/swarm/supervisor.py",
        changed_paths=["aragora/swarm/supervisor.py"],
        commit_shas=["abc12345"],
        tests_run=[],
        verification_results=[],
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

    updated = store.get_supervisor_run(run_id)
    assert updated is not None
    wo = updated["work_orders"][0]
    assert wo["status"] == "needs_human"
    assert wo["review_status"] == "changes_requested"
    assert wo["receipt_id"] is None
    assert wo["worker_outcome"] == "merge_gate_failed"
    assert wo["merge_gate"]["checks_passed"] is False
    assert wo["merge_gate"]["verification_missing_reason"] == "missing_verification_plan"
    assert wo["verification_missing_reason"] == "missing_verification_plan"
    assert wo["failure_reason"] == "missing_verification_plan"
    assert "verification command" in wo["blocking_question"]
    assert "missing verification plan" in wo["dispatch_error"]


@pytest.mark.asyncio
async def test_collect_results_backfills_receipt_for_salvaged_deliverable(
    repo: Path, store: DevCoordinationStore
) -> None:
    lease = store.claim_lease(
        task_id="salvage-direct-lane",
        title="Salvage direct lane",
        owner_agent="claude",
        owner_session_id="salvage-direct-session",
        branch="main",
        worktree_path=str(repo),
        claimed_paths=["aragora/swarm/supervisor.py"],
    )
    run_record = store.create_supervisor_run(
        goal="salvage direct",
        target_branch="main",
        supervisor_agents={},
        approval_policy={},
        spec={"raw_goal": "salvage direct"},
        work_orders=[
            {
                "work_order_id": "wo-salvage-direct",
                "status": "dispatched",
                "worktree_path": str(repo),
                "branch": "main",
                "target_agent": "claude",
                "owner_session_id": "salvage-direct-session",
                "lease_id": lease.lease_id,
                "review_status": "pending",
                "file_scope": ["aragora/swarm/supervisor.py"],
                "receipt_id": None,
            }
        ],
        status="active",
    )
    run_id = run_record["run_id"]

    mock_launcher = MagicMock(spec=WorkerLauncher)
    salvaged_worker = WorkerProcess(
        work_order_id="wo-salvage-direct",
        agent="claude",
        worktree_path=str(repo),
        branch="main",
        session_id="salvage-direct-session",
        pid=100,
        exit_code=143,
        completed_at="2026-03-06T20:00:00+00:00",
        diff="diff --git a/aragora/swarm/supervisor.py",
        changed_paths=["aragora/swarm/supervisor.py"],
        commit_shas=["abc12345"],
        head_sha="abc12345",
    )
    mock_launcher.get_worker = MagicMock(return_value=salvaged_worker)
    mock_launcher.wait = AsyncMock(return_value=salvaged_worker)

    supervisor = SwarmSupervisor(repo_root=repo, store=store, launcher=mock_launcher)

    results = await supervisor.collect_results(run_id)
    assert len(results) == 1

    updated = store.get_supervisor_run(run_id)
    assert updated is not None
    wo = updated["work_orders"][0]
    assert wo["status"] == "completed"
    assert wo["review_status"] == "pending_heterogeneous_review"
    assert wo["worker_outcome"] == "crash_with_salvage"
    assert wo["receipt_id"] is not None
    receipt = store.get_completion_receipt(wo["receipt_id"])
    assert receipt is not None
    assert receipt.outcome == "deliverable_created"


def test_merge_gate_state_allows_docs_only_lane_without_verification_plan() -> None:
    state = SwarmSupervisor._merge_gate_state(
        {
            "file_scope": ["docs/**"],
            "changed_paths": ["docs/status/DESIGN_PARTNER_PROGRAM.md"],
            "expected_tests": [],
            "verification_results": [],
        }
    )

    assert state["checks_passed"] is True
    assert state["merge_eligible"] is True
    assert state["verification_missing_reason"] is None
    assert state["blocked_reasons"] == []


def test_merge_gate_state_normalizes_python_command_equivalence() -> None:
    state = SwarmSupervisor._merge_gate_state(
        {
            "expected_tests": ["python -m pytest tests/swarm/test_supervisor.py -q"],
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
        }
    )

    assert state["checks_passed"] is True
    assert state["merge_eligible"] is True
    assert state["blocked_reasons"] == []


def test_merge_gate_state_rejects_broader_pytest_with_k_selector() -> None:
    """A recorded command with -k selectors must NOT satisfy an expected check
    via path-based equivalence -- the selectors may filter out the required tests."""
    state = SwarmSupervisor._merge_gate_state(
        {
            "expected_tests": ["python -m pytest tests/swarm/test_supervisor.py -q"],
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
        }
    )

    assert state["checks_passed"] is False
    assert state["merge_eligible"] is False
    assert len(state["blocked_reasons"]) > 0


def test_refresh_run_backfills_missing_receipt_for_completed_deliverable(
    repo: Path, store: DevCoordinationStore
) -> None:
    run_record = store.create_supervisor_run(
        goal="backfill receipt lane",
        target_branch="main",
        supervisor_agents={},
        approval_policy={},
        spec={"raw_goal": "backfill receipt lane"},
        work_orders=[
            {
                "work_order_id": "wo-backfill",
                "status": "queued",
                "worktree_path": str(repo),
                "branch": "main",
                "target_agent": "claude",
                "owner_session_id": "backfill-session",
                "review_status": "pending",
                "file_scope": ["aragora/swarm/supervisor.py"],
            }
        ],
        status="active",
    )
    run_id = run_record["run_id"]
    lease = store.claim_lease(
        task_id="wo-backfill",
        title="Backfill lane",
        owner_agent="claude",
        owner_session_id="backfill-session",
        branch="main",
        worktree_path=str(repo),
        claimed_paths=["aragora/swarm/supervisor.py"],
        expected_tests=["python -m pytest tests/swarm/test_supervisor.py -q"],
        metadata={"supervisor_run_id": run_id, "work_order_id": "wo-backfill"},
    )
    store.update_supervisor_run(
        run_id,
        status="completed",
        work_orders=[
            {
                "work_order_id": "wo-backfill",
                "status": "completed",
                "worktree_path": str(repo),
                "branch": "main",
                "target_agent": "claude",
                "owner_session_id": "backfill-session",
                "lease_id": lease.lease_id,
                "receipt_id": None,
                "review_status": "pending_heterogeneous_review",
                "file_scope": ["aragora/swarm/supervisor.py"],
                "changed_paths": ["aragora/swarm/supervisor.py"],
                "commit_shas": ["abc12345"],
                "tests_run": ["python -m pytest tests/swarm/test_supervisor.py -q"],
                "worker_outcome": "completed",
                "initial_head": "base123",
                "head_sha": "head456",
                "confidence": 0.82,
            }
        ],
    )
    store.fleet_store.release_paths(session_id="backfill-session")

    supervisor = SwarmSupervisor(repo_root=repo, store=store)
    refreshed = supervisor.refresh_run(run_id)

    wo = refreshed.work_orders[0]
    assert wo["status"] == "completed"
    assert wo["receipt_id"] is not None
    receipt = store.get_completion_receipt(wo["receipt_id"])
    assert receipt is not None
    assert receipt.lease_id == lease.lease_id
    assert receipt.metadata["backfilled_receipt"] is True


@pytest.mark.asyncio
async def test_collect_results_marks_scope_violation_needs_human(
    repo: Path, store: DevCoordinationStore
) -> None:
    lease = store.claim_lease(
        task_id="scope-lane",
        title="Scope lane",
        owner_agent="claude",
        owner_session_id="scope-session",
        branch="main",
        worktree_path=str(repo),
        claimed_paths=["aragora/server/auth_checks.py"],
    )
    run_record = store.create_supervisor_run(
        goal="scope violation",
        target_branch="main",
        supervisor_agents={},
        approval_policy={},
        spec={"raw_goal": "scope violation"},
        work_orders=[
            {
                "work_order_id": "wo-scope",
                "status": "dispatched",
                "worktree_path": str(repo),
                "branch": "main",
                "target_agent": "claude",
                "owner_session_id": "scope-session",
                "lease_id": lease.lease_id,
                "file_scope": ["aragora/server/auth_checks.py"],
                "review_status": "pending",
            }
        ],
        status="active",
    )
    run_id = run_record["run_id"]

    mock_launcher = MagicMock(spec=WorkerLauncher)
    completed_worker = WorkerProcess(
        work_order_id="wo-scope",
        agent="claude",
        worktree_path=str(repo),
        branch="main",
        session_id="scope-session",
        pid=100,
        exit_code=0,
        completed_at="2026-03-06T20:00:00+00:00",
        diff="diff --git a/aragora/server/handlers/playground.py",
        changed_paths=["aragora/server/handlers/playground.py"],
        commit_shas=["abc12345"],
        tests_run=["pytest -q tests/swarm/test_supervisor.py"],
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

    updated = store.get_supervisor_run(run_id)
    assert updated is not None
    wo = updated["work_orders"][0]
    assert wo["status"] == "scope_violation"
    assert wo["review_status"] == "changes_requested"
    assert "outside permitted scope" in wo["dispatch_error"]
    assert wo["failure_reason"] == "scope_violation"
    assert "stay in scope" in wo["blocking_question"]
    assert wo.get("receipt_id") is None
    assert wo["lease_id"] == lease.lease_id
    assert wo["scope_violation"]["violations"][0]["type"] == "out_of_scope"

    summary = store.status_summary()
    assert summary["counts"]["scope_violations"] == 0
    assert summary["counts"]["active_leases"] == 0


@pytest.mark.asyncio
async def test_collect_finished_results_uses_terminal_session_meta_even_when_pid_looks_alive(
    repo: Path, store: DevCoordinationStore
) -> None:
    initial_head = _run(repo, "git", "rev-parse", "HEAD").stdout.strip()
    (repo / "README.md").write_text("wrong work\n", encoding="utf-8")
    _run(repo, "git", "add", "README.md")
    _run(repo, "git", "commit", "-m", "wrong work")
    (repo / ".codex_session_meta.json").write_text(
        ('{\n  "ended_at": "2026-03-09T13:33:17Z",\n  "exit_code": 0\n}\n'),
        encoding="utf-8",
    )

    lease = store.claim_lease(
        task_id="citation-lane",
        title="Citation lane",
        owner_agent="codex",
        owner_session_id="scope-session",
        branch="main",
        worktree_path=str(repo),
        claimed_paths=["docs/citations.md"],
    )
    run_record = store.create_supervisor_run(
        goal="detect terminal session meta",
        target_branch="main",
        supervisor_agents={},
        approval_policy={},
        spec={"raw_goal": "detect terminal session meta"},
        work_orders=[
            {
                "work_order_id": "wo-session-meta",
                "status": "dispatched",
                "worktree_path": str(repo),
                "branch": "main",
                "target_agent": "codex",
                "owner_session_id": "scope-session",
                "lease_id": lease.lease_id,
                "pid": 96969,
                "initial_head": initial_head,
                "review_status": "pending",
                "file_scope": ["docs/citations.md"],
            }
        ],
        status="active",
    )

    mock_launcher = MagicMock(spec=WorkerLauncher)
    mock_launcher.collect_finished = AsyncMock(return_value=[])
    mock_launcher.snapshot_progress = AsyncMock()
    mock_launcher.config = SimpleNamespace(auto_commit=False, no_progress_timeout_seconds=120.0)

    supervisor = SwarmSupervisor(repo_root=repo, store=store, launcher=mock_launcher)

    with patch.object(WorkerLauncher, "_is_pid_running", return_value=True):
        completed = await supervisor.collect_finished_results(run_record["run_id"])

    assert len(completed) == 1
    assert completed[0].exit_code == 0
    mock_launcher.snapshot_progress.assert_not_awaited()

    updated = store.get_supervisor_run(run_record["run_id"])
    assert updated is not None
    assert updated["status"] == "completed"
    work_order = updated["work_orders"][0]
    assert work_order["status"] == "scope_violation"
    assert work_order["review_status"] == "changes_requested"
    violation_paths = {
        item["path"] for item in work_order["scope_violation"]["violations"] if "path" in item
    }
    assert "README.md" in violation_paths

    summary = store.status_summary()
    assert summary["counts"]["active_leases"] == 0
    assert summary["counts"]["scope_violations"] == 0


def test_refresh_run_collects_finished_detached_worker_before_stale_lease_reap(
    repo: Path, store: DevCoordinationStore
) -> None:
    expected_test = "python -m pytest tests/swarm/test_supervisor.py -q"
    initial_head = _run(repo, "git", "rev-parse", "HEAD").stdout.strip()
    lease = store.claim_lease(
        task_id="detached-pre-reap",
        title="Detached pre-reap lane",
        owner_agent="codex",
        owner_session_id="detached-pre-reap-session",
        branch="main",
        worktree_path=str(repo),
        claimed_paths=["aragora/swarm/supervisor.py"],
        expected_tests=[expected_test],
    )
    run_record = store.create_supervisor_run(
        goal="collect detached result before stale reaping",
        target_branch="main",
        supervisor_agents={},
        approval_policy={},
        spec={"raw_goal": "collect detached result before stale reaping"},
        work_orders=[
            {
                "work_order_id": "wo-detached-pre-reap",
                "status": "dispatched",
                "worktree_path": str(repo),
                "branch": "main",
                "target_agent": "codex",
                "owner_session_id": "detached-pre-reap-session",
                "lease_id": lease.lease_id,
                "pid": 424242,
                "initial_head": initial_head,
                "review_status": "pending",
                "file_scope": ["aragora/swarm/supervisor.py"],
                "expected_tests": [expected_test],
            }
        ],
        status="active",
    )
    run_id = run_record["run_id"]

    finished_worker = WorkerProcess(
        work_order_id="wo-detached-pre-reap",
        agent="codex",
        worktree_path=str(repo),
        branch="main",
        session_id="detached-pre-reap-session",
        pid=424242,
        initial_head=initial_head,
        exit_code=0,
        completed_at="2026-03-21T01:00:00+00:00",
        diff="diff --git a/aragora/swarm/supervisor.py",
        changed_paths=["aragora/swarm/supervisor.py"],
        commit_shas=["abc12345"],
        head_sha="abc12345",
        stdout="worker stdout",
        stderr="",
        tests_run=[expected_test],
        verification_results=[
            {
                "command": expected_test,
                "exit_code": 0,
                "passed": True,
                "stdout": "1 passed",
                "stderr": "",
                "duration_seconds": 0.2,
            }
        ],
    )

    mock_launcher = MagicMock(spec=WorkerLauncher)
    mock_launcher.collect_finished = AsyncMock(return_value=[])
    mock_launcher.snapshot_progress = AsyncMock()
    mock_launcher.get_worker = MagicMock(return_value=None)
    mock_launcher.config = SimpleNamespace(auto_commit=False, no_progress_timeout_seconds=120.0)

    supervisor = SwarmSupervisor(repo_root=repo, store=store, launcher=mock_launcher)

    observed: dict[str, str] = {}

    def _reap_stale() -> list[object]:
        current = store.get_supervisor_run(run_id)
        assert current is not None
        observed["status_before_reap"] = current["work_orders"][0]["status"]
        return []

    store.reap_stale_leases = MagicMock(side_effect=_reap_stale)
    store.reap_expired_leases = MagicMock(return_value=[])

    with patch.object(
        WorkerLauncher,
        "collect_detached_result",
        new=AsyncMock(return_value=finished_worker),
    ):
        refreshed = supervisor.refresh_run(run_id)

    assert observed["status_before_reap"] == "completed"
    assert refreshed.work_orders[0]["status"] == "completed"
    assert refreshed.work_orders[0]["review_status"] == "pending_heterogeneous_review"
    assert refreshed.work_orders[0]["merge_gate"]["checks_passed"] is True


@pytest.mark.asyncio
async def test_collect_finished_results_passes_expected_tests_to_detached_collection(
    repo: Path, store: DevCoordinationStore
) -> None:
    expected_test = "python -m pytest tests/swarm/test_supervisor.py -q"
    lease = store.claim_lease(
        task_id="detached-merge-pass",
        title="Detached merge pass lane",
        owner_agent="codex",
        owner_session_id="detached-merge-session",
        branch="main",
        worktree_path=str(repo),
        claimed_paths=["aragora/swarm/supervisor.py"],
        expected_tests=[expected_test],
    )
    initial_head = _run(repo, "git", "rev-parse", "HEAD").stdout.strip()
    run_record = store.create_supervisor_run(
        goal="detached merge gate pass",
        target_branch="main",
        supervisor_agents={},
        approval_policy={},
        spec={"raw_goal": "detached merge gate pass"},
        work_orders=[
            {
                "work_order_id": "wo-detached-merge-pass",
                "status": "dispatched",
                "worktree_path": str(repo),
                "branch": "main",
                "target_agent": "codex",
                "owner_session_id": "detached-merge-session",
                "lease_id": lease.lease_id,
                "pid": 424242,
                "initial_head": initial_head,
                "review_status": "pending",
                "file_scope": ["aragora/swarm/supervisor.py"],
                "expected_tests": [expected_test],
            }
        ],
        status="active",
    )

    detached_worker = WorkerProcess(
        work_order_id="wo-detached-merge-pass",
        agent="codex",
        worktree_path=str(repo),
        branch="main",
        session_id="detached-merge-session",
        pid=424242,
        initial_head=initial_head,
        exit_code=0,
        completed_at="2026-03-21T01:00:00+00:00",
        diff="diff --git a/aragora/swarm/supervisor.py",
        changed_paths=["aragora/swarm/supervisor.py"],
        commit_shas=["abc12345"],
        head_sha="abc12345",
        stdout="worker stdout",
        stderr="",
        tests_run=[expected_test],
        verification_results=[
            {
                "command": expected_test,
                "exit_code": 0,
                "passed": True,
                "stdout": "1 passed",
                "stderr": "",
                "duration_seconds": 0.2,
            }
        ],
    )

    mock_launcher = MagicMock(spec=WorkerLauncher)
    mock_launcher.collect_finished = AsyncMock(return_value=[])
    mock_launcher.snapshot_progress = AsyncMock()
    mock_launcher.config = SimpleNamespace(auto_commit=False, no_progress_timeout_seconds=120.0)

    supervisor = SwarmSupervisor(repo_root=repo, store=store, launcher=mock_launcher)

    collect_detached = AsyncMock(return_value=detached_worker)
    with patch.object(WorkerLauncher, "collect_detached_result", new=collect_detached):
        completed = await supervisor.collect_finished_results(run_record["run_id"])

    assert len(completed) == 1
    collect_detached.assert_awaited_once()
    assert collect_detached.await_args.kwargs["expected_tests"] == [expected_test]
    mock_launcher.snapshot_progress.assert_not_awaited()

    updated = store.get_supervisor_run(run_record["run_id"])
    assert updated is not None
    work_order = updated["work_orders"][0]
    assert work_order["status"] == "completed"
    assert work_order["review_status"] == "pending_heterogeneous_review"
    assert work_order["merge_gate"]["checks_passed"] is True
    assert work_order["tests_run"] == [expected_test]
    assert work_order["verification_results"][0]["command"] == expected_test


@pytest.mark.asyncio
async def test_collect_finished_results_backfills_receipt_for_salvaged_detached_deliverable(
    repo: Path, store: DevCoordinationStore
) -> None:
    lease = store.claim_lease(
        task_id="detached-salvage-lane",
        title="Detached salvage lane",
        owner_agent="codex",
        owner_session_id="detached-salvage-session",
        branch="main",
        worktree_path=str(repo),
        claimed_paths=["aragora/swarm/supervisor.py"],
    )
    initial_head = _run(repo, "git", "rev-parse", "HEAD").stdout.strip()
    run_record = store.create_supervisor_run(
        goal="detached salvage",
        target_branch="main",
        supervisor_agents={},
        approval_policy={},
        spec={"raw_goal": "detached salvage"},
        work_orders=[
            {
                "work_order_id": "wo-detached-salvage",
                "status": "dispatched",
                "worktree_path": str(repo),
                "branch": "main",
                "target_agent": "codex",
                "owner_session_id": "detached-salvage-session",
                "lease_id": lease.lease_id,
                "pid": 424242,
                "initial_head": initial_head,
                "review_status": "pending",
                "file_scope": ["aragora/swarm/supervisor.py"],
                "receipt_id": None,
            }
        ],
        status="active",
    )

    detached_worker = WorkerProcess(
        work_order_id="wo-detached-salvage",
        agent="codex",
        worktree_path=str(repo),
        branch="main",
        session_id="detached-salvage-session",
        pid=424242,
        initial_head=initial_head,
        exit_code=143,
        completed_at="2026-03-21T01:00:00+00:00",
        diff="diff --git a/aragora/swarm/supervisor.py",
        changed_paths=["aragora/swarm/supervisor.py"],
        commit_shas=["abc12345"],
        head_sha="abc12345",
        stdout="worker stdout",
        stderr="",
    )

    mock_launcher = MagicMock(spec=WorkerLauncher)
    mock_launcher.collect_finished = AsyncMock(return_value=[])
    mock_launcher.snapshot_progress = AsyncMock()
    mock_launcher.config = SimpleNamespace(auto_commit=False, no_progress_timeout_seconds=120.0)

    supervisor = SwarmSupervisor(repo_root=repo, store=store, launcher=mock_launcher)

    collect_detached = AsyncMock(return_value=detached_worker)
    with patch.object(WorkerLauncher, "collect_detached_result", new=collect_detached):
        completed = await supervisor.collect_finished_results(run_record["run_id"])

    assert len(completed) == 1
    updated = store.get_supervisor_run(run_record["run_id"])
    assert updated is not None
    work_order = updated["work_orders"][0]
    assert work_order["status"] == "completed"
    assert work_order["review_status"] == "pending_heterogeneous_review"
    assert work_order["worker_outcome"] == "crash_with_salvage"
    assert work_order["receipt_id"] is not None
    receipt = store.get_completion_receipt(work_order["receipt_id"])
    assert receipt is not None
    assert receipt.outcome == "deliverable_created"


@pytest.mark.asyncio
async def test_dispatch_handles_missing_cli(repo: Path, store: DevCoordinationStore) -> None:
    """dispatch_workers should requeue onto the fallback agent when one CLI is unavailable."""
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
    assert wo["status"] == "leased"
    assert wo["target_agent"] == "codex"
    assert wo["reviewer_agent"] == "claude"
    assert wo["metadata"]["last_failure_reason"] == "agent_unavailable"
    assert "CLI not found" in wo["metadata"]["last_failure_detail"]
    assert wo.get("lease_id") is None


@pytest.mark.asyncio
async def test_dispatch_workers_trips_and_skips_worker_type_circuit_breaker(
    repo: Path, store: DevCoordinationStore
) -> None:
    run_record = store.create_supervisor_run(
        goal="trip worker breaker",
        target_branch="main",
        supervisor_agents={},
        approval_policy={},
        spec={"raw_goal": "trip worker breaker"},
        work_orders=[
            {
                "work_order_id": "wo-1",
                "status": "leased",
                "worktree_path": str(repo),
                "branch": "main",
                "target_agent": "claude",
                "reviewer_agent": "codex",
            },
            {
                "work_order_id": "wo-2",
                "status": "leased",
                "worktree_path": str(repo),
                "branch": "main",
                "target_agent": "claude",
                "reviewer_agent": "codex",
            },
            {
                "work_order_id": "wo-3",
                "status": "leased",
                "worktree_path": str(repo),
                "branch": "main",
                "target_agent": "claude",
                "reviewer_agent": "codex",
            },
        ],
        status="active",
        metadata={
            "max_concurrency": 3,
            "managed_dir_pattern": ".worktrees/custom-{agent}",
            WORKER_TYPE_CIRCUIT_BREAKER_POLICY_KEY: {
                "failure_threshold": 2,
                "reset_timeout_seconds": 300.0,
            },
        },
    )

    mock_launcher = MagicMock(spec=WorkerLauncher)
    mock_launcher.launch = AsyncMock(
        side_effect=[
            FileNotFoundError("claude CLI not found"),
            FileNotFoundError("claude CLI not found"),
        ]
    )

    supervisor = SwarmSupervisor(repo_root=repo, store=store, launcher=mock_launcher)

    launched = await supervisor.dispatch_workers(run_record["run_id"])

    assert launched == []
    assert mock_launcher.launch.await_count == 2

    updated = store.get_supervisor_run(run_record["run_id"])
    assert updated is not None
    assert updated["metadata"]["max_concurrency"] == 3
    assert updated["metadata"]["managed_dir_pattern"] == ".worktrees/custom-{agent}"
    breaker = updated["metadata"][WORKER_TYPE_CIRCUIT_BREAKERS_KEY]["claude"]
    assert breaker["status"] == "open"
    assert breaker["failure_count"] == 2
    assert breaker["trip_count"] == 1
    assert breaker["last_failure_reason"] == "agent_unavailable"
    assert breaker["blocked_until"]

    work_orders = updated["work_orders"]
    assert [item["target_agent"] for item in work_orders] == ["codex", "codex", "codex"]
    assert [item["status"] for item in work_orders] == ["leased", "leased", "leased"]
    assert work_orders[2]["metadata"]["last_failure_reason"] == "worker_type_blocked"


@pytest.mark.asyncio
async def test_dispatch_workers_marks_needs_human_when_all_worker_types_blocked(
    repo: Path, store: DevCoordinationStore
) -> None:
    lease = store.claim_lease(
        task_id="wo-blocked",
        title="blocked worker dispatch",
        owner_agent="claude",
        owner_session_id="blocked-session",
        branch="main",
        worktree_path=str(repo),
        claimed_paths=["tests/swarm/test_supervisor.py"],
    )
    blocked_until = (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()
    run_record = store.create_supervisor_run(
        goal="all workers blocked",
        target_branch="main",
        supervisor_agents={},
        approval_policy={},
        spec={"raw_goal": "all workers blocked"},
        work_orders=[
            {
                "work_order_id": "wo-blocked",
                "status": "leased",
                "worktree_path": str(repo),
                "branch": "main",
                "target_agent": "claude",
                "reviewer_agent": "codex",
                "lease_id": lease.lease_id,
            }
        ],
        status="active",
        metadata={
            WORKER_TYPE_CIRCUIT_BREAKERS_KEY: {
                "claude": {
                    "status": "open",
                    "failure_count": 2,
                    "failure_threshold": 2,
                    "reset_timeout_seconds": 300.0,
                    "opened_at": datetime.now(timezone.utc).isoformat(),
                    "blocked_until": blocked_until,
                    "last_failure_reason": "agent_unavailable",
                    "last_failure_detail": "claude CLI not found",
                    "trip_count": 1,
                },
                "codex": {
                    "status": "open",
                    "failure_count": 2,
                    "failure_threshold": 2,
                    "reset_timeout_seconds": 300.0,
                    "opened_at": datetime.now(timezone.utc).isoformat(),
                    "blocked_until": blocked_until,
                    "last_failure_reason": "agent_capacity",
                    "last_failure_detail": "Credit balance is too low",
                    "trip_count": 1,
                },
            },
        },
    )

    supervisor = SwarmSupervisor(
        repo_root=repo, store=store, launcher=MagicMock(spec=WorkerLauncher)
    )

    launched = await supervisor.dispatch_workers(run_record["run_id"])

    assert launched == []
    updated = store.get_supervisor_run(run_record["run_id"])
    assert updated is not None
    work_order = updated["work_orders"][0]
    assert work_order["status"] == "needs_human"
    assert "worker dispatch blocked" in work_order["dispatch_error"]
    assert work_order["failure_reason"] == "worker_type_blocked"
    assert "worker type or capacity issue" in work_order["blocking_question"]
    assert updated["status"] == "needs_human"
    assert updated["metadata"][CAMPAIGN_OUTCOME_METADATA_KEY] == "needs_human"
    assert store.status_summary()["counts"]["active_leases"] == 0


@pytest.mark.asyncio
async def test_dispatch_workers_resets_closed_worker_type_circuit_breaker_after_successful_launch(
    repo: Path, store: DevCoordinationStore
) -> None:
    run_record = store.create_supervisor_run(
        goal="reset closed worker breaker",
        target_branch="main",
        supervisor_agents={},
        approval_policy={},
        spec={"raw_goal": "reset closed worker breaker"},
        work_orders=[
            {
                "work_order_id": "wo-reset",
                "status": "leased",
                "worktree_path": str(repo),
                "branch": "main",
                "target_agent": "claude",
                "reviewer_agent": "codex",
            }
        ],
        status="active",
        metadata={
            WORKER_TYPE_CIRCUIT_BREAKER_POLICY_KEY: {
                "failure_threshold": 2,
                "reset_timeout_seconds": 300.0,
            },
            WORKER_TYPE_CIRCUIT_BREAKERS_KEY: {
                "claude": {
                    "status": "closed",
                    "failure_count": 1,
                    "failure_threshold": 2,
                    "reset_timeout_seconds": 300.0,
                    "opened_at": None,
                    "blocked_until": None,
                    "last_failure_at": datetime.now(timezone.utc).isoformat(),
                    "last_failure_reason": "agent_launch_failed",
                    "last_failure_detail": "temporary launch error",
                    "trip_count": 0,
                    "last_reset_at": None,
                }
            },
        },
    )

    mock_launcher = MagicMock(spec=WorkerLauncher)
    mock_launcher.launch = AsyncMock(
        return_value=WorkerProcess(
            work_order_id="wo-reset",
            agent="claude",
            worktree_path=str(repo),
            branch="main",
            pid=321,
            initial_head="abc123",
        )
    )

    supervisor = SwarmSupervisor(repo_root=repo, store=store, launcher=mock_launcher)

    launched = await supervisor.dispatch_workers(run_record["run_id"])

    assert len(launched) == 1
    assert mock_launcher.launch.await_count == 1

    updated = store.get_supervisor_run(run_record["run_id"])
    assert updated is not None
    assert updated["work_orders"][0]["status"] == "dispatched"

    breaker = updated["metadata"][WORKER_TYPE_CIRCUIT_BREAKERS_KEY]["claude"]
    assert breaker["status"] == "closed"
    assert breaker["failure_count"] == 0
    assert breaker["opened_at"] is None
    assert breaker["blocked_until"] is None
    assert breaker["last_reset_at"]
    assert breaker["last_failure_reason"] == "agent_launch_failed"


@pytest.mark.asyncio
async def test_collect_finished_results_requeues_capacity_failure_to_fallback_agent(
    repo: Path, store: DevCoordinationStore
) -> None:
    lifecycle = MagicMock()
    lifecycle.ensure_managed_worktree.return_value = ManagedWorktreeSession(
        path=repo,
        branch="swarm-fallback-1",
        session_id="dispatch-sess-fallback",
        agent="claude",
        created=True,
        reconcile_status=None,
        payload={},
    )

    mock_launcher = MagicMock(spec=WorkerLauncher)
    mock_launcher.collect_finished = AsyncMock(
        return_value=[
            WorkerProcess(
                work_order_id="fallback-task",
                agent="claude",
                worktree_path=str(repo),
                branch="swarm-fallback-1",
                pid=777,
                session_id="dispatch-sess-fallback",
                lease_id="lease-1",
                exit_code=1,
                completed_at="2026-03-06T00:00:00+00:00",
                stdout="Credit balance is too low\n",
                stderr="",
            )
        ]
    )
    mock_launcher.collect_detached_finished = AsyncMock(return_value=None)

    decomposer = MagicMock()
    decomposer.analyze.return_value = TaskDecomposition(
        original_task="Goal",
        complexity_score=2,
        complexity_level="low",
        should_decompose=True,
        subtasks=[
            SubTask(
                id="fallback-task",
                title="Fallback test",
                description="Test capacity fallback",
                file_scope=["README.md"],
            )
        ],
    )

    supervisor = SwarmSupervisor(
        repo_root=repo,
        store=store,
        lifecycle=lifecycle,
        launcher=mock_launcher,
        decomposer=decomposer,
    )
    run = supervisor.start_run(
        spec=SwarmSpec(raw_goal="Goal", refined_goal="Goal"),
        max_concurrency=4,
        managed_dir_pattern=".worktrees/phase0b-{agent}",
    )
    run.work_orders[0]["status"] = "dispatched"
    run.work_orders[0]["target_agent"] = "claude"
    run.work_orders[0]["reviewer_agent"] = "codex"
    run.work_orders[0]["pid"] = 777
    run.work_orders[0]["worker_outcome"] = "crash_with_salvage"
    run.work_orders[0]["confidence"] = 0.84
    run.work_orders[0]["initial_head"] = "old-base"
    run.work_orders[0]["head_sha"] = "old-head"
    run.work_orders[0]["commit_shas"] = ["old-commit"]
    run.work_orders[0]["changed_paths"] = ["README.md"]
    run.work_orders[0]["diff"] = "diff --git a/README.md b/README.md"
    run.work_orders[0]["diff_lines"] = 12
    run.work_orders[0]["stdout_tail"] = "old stdout"
    run.work_orders[0]["stderr_tail"] = "old stderr"
    run.work_orders[0]["tests_run"] = ["python -m pytest tests/swarm/test_supervisor.py -q"]
    run.work_orders[0]["verification_results"] = [{"command": "pytest", "passed": True}]
    run.work_orders[0]["merge_gate"] = {"checks_passed": True}
    run.work_orders[0]["verification_missing_reason"] = "missing_verification_plan"
    run.work_orders[0]["pr_url"] = "https://github.com/synaptent/aragora/pull/9999"
    run.work_orders[0]["adopted_pr"] = "https://github.com/synaptent/aragora/pull/9999"
    run.work_orders[0]["scope_violation"] = {"violations": [{"path": "README.md"}]}
    store.update_supervisor_run(run.run_id, work_orders=run.work_orders, status="active")

    completed = await supervisor.collect_finished_results(run.run_id)

    assert len(completed) == 1
    refreshed = store.get_supervisor_run(run.run_id)
    assert refreshed is not None
    assert refreshed["metadata"]["max_concurrency"] == 4
    assert refreshed["metadata"]["managed_dir_pattern"] == ".worktrees/phase0b-{agent}"
    work_order = refreshed["work_orders"][0]
    assert work_order["status"] == "leased"
    assert work_order["target_agent"] == "codex"
    assert work_order["reviewer_agent"] == "claude"
    assert work_order["metadata"]["last_failure_reason"] == "agent_capacity"
    assert work_order["metadata"]["attempted_agents"] == ["claude"]
    assert work_order["lease_id"] == run.work_orders[0]["lease_id"]
    for cleared_key in (
        "worker_outcome",
        "confidence",
        "initial_head",
        "head_sha",
        "commit_shas",
        "changed_paths",
        "diff",
        "diff_lines",
        "stdout_tail",
        "stderr_tail",
        "tests_run",
        "verification_results",
        "merge_gate",
        "verification_missing_reason",
        "pr_url",
        "adopted_pr",
        "scope_violation",
    ):
        assert cleared_key not in work_order
    breaker = refreshed["metadata"][WORKER_TYPE_CIRCUIT_BREAKERS_KEY]["claude"]
    assert breaker["status"] == "open"
    assert breaker["failure_count"] == breaker["failure_threshold"]
    assert breaker["last_failure_reason"] == "agent_capacity"
    assert breaker["last_failure_detail"] == "Credit balance is too low"
    assert store.status_summary()["counts"]["active_leases"] == 1


def test_reset_worker_type_circuit_breaker_preserves_run_metadata(
    repo: Path, store: DevCoordinationStore
) -> None:
    run_record = store.create_supervisor_run(
        goal="reset breaker metadata",
        target_branch="main",
        supervisor_agents={},
        approval_policy={},
        spec={"raw_goal": "reset breaker metadata"},
        work_orders=[],
        status="active",
        metadata={
            "max_concurrency": 5,
            "managed_dir_pattern": ".worktrees/preserve-{agent}",
            WORKER_TYPE_CIRCUIT_BREAKER_POLICY_KEY: {
                "failure_threshold": 2,
                "reset_timeout_seconds": 300.0,
            },
            WORKER_TYPE_CIRCUIT_BREAKERS_KEY: {
                "claude": {
                    "status": "open",
                    "failure_count": 2,
                    "failure_threshold": 2,
                    "reset_timeout_seconds": 300.0,
                    "opened_at": datetime.now(timezone.utc).isoformat(),
                    "blocked_until": (
                        datetime.now(timezone.utc) + timedelta(minutes=5)
                    ).isoformat(),
                    "last_failure_reason": "agent_unavailable",
                    "last_failure_detail": "CLI not found",
                }
            },
        },
    )

    supervisor = SwarmSupervisor(repo_root=repo, store=store)

    updated = supervisor.reset_worker_type_circuit_breaker(run_record["run_id"], "claude")

    assert updated.metadata["max_concurrency"] == 5
    assert updated.metadata["managed_dir_pattern"] == ".worktrees/preserve-{agent}"
    breaker = updated.metadata[WORKER_TYPE_CIRCUIT_BREAKERS_KEY]["claude"]
    assert breaker["status"] == "closed"
    assert breaker["failure_count"] == 0
    assert breaker["last_reset_at"]


@pytest.mark.asyncio
async def test_dispatch_workers_expires_worker_type_circuit_breaker(
    repo: Path, store: DevCoordinationStore
) -> None:
    expired_at = datetime.now(UTC) - timedelta(minutes=10)
    run_record = store.create_supervisor_run(
        goal="expire worker breaker",
        target_branch="main",
        supervisor_agents={},
        approval_policy={},
        spec={"raw_goal": "expire worker breaker"},
        work_orders=[
            {
                "work_order_id": "wo-expire",
                "status": "leased",
                "worktree_path": str(repo),
                "branch": "main",
                "target_agent": "claude",
                "reviewer_agent": "codex",
            }
        ],
        status="active",
        metadata={
            WORKER_TYPE_CIRCUIT_BREAKER_POLICY_KEY: {
                "failure_threshold": 2,
                "reset_timeout_seconds": 60.0,
            },
            WORKER_TYPE_CIRCUIT_BREAKERS_KEY: {
                "claude": {
                    "status": "open",
                    "failure_count": 2,
                    "failure_threshold": 2,
                    "reset_timeout_seconds": 60.0,
                    "opened_at": expired_at.isoformat(),
                    "blocked_until": (expired_at + timedelta(minutes=1)).isoformat(),
                    "last_failure_reason": "agent_capacity",
                    "last_failure_detail": "Credit balance is too low",
                }
            },
        },
    )

    mock_launcher = MagicMock(spec=WorkerLauncher)
    mock_worker = WorkerProcess(
        work_order_id="wo-expire",
        agent="claude",
        worktree_path=str(repo),
        branch="main",
        pid=444,
        initial_head="abc123",
    )
    mock_launcher.launch = AsyncMock(return_value=mock_worker)

    supervisor = SwarmSupervisor(repo_root=repo, store=store, launcher=mock_launcher)

    launched = await supervisor.dispatch_workers(run_record["run_id"])

    assert len(launched) == 1
    assert mock_launcher.launch.await_count == 1

    updated = store.get_supervisor_run(run_record["run_id"])
    assert updated is not None
    work_order = updated["work_orders"][0]
    assert work_order["status"] == "dispatched"
    assert work_order["target_agent"] == "claude"

    breaker = updated["metadata"][WORKER_TYPE_CIRCUIT_BREAKERS_KEY]["claude"]
    assert breaker["status"] == "closed"
    assert breaker["failure_count"] == 0
    assert breaker["opened_at"] is None
    assert breaker["blocked_until"] is None
    assert breaker["last_reset_at"]


@pytest.mark.asyncio
async def test_collect_finished_results_expires_open_breaker(
    repo: Path, store: DevCoordinationStore
) -> None:
    expired_at = datetime.now(UTC) - timedelta(minutes=10)
    run_record = store.create_supervisor_run(
        goal="expire worker breaker during collect",
        target_branch="main",
        supervisor_agents={},
        approval_policy={},
        spec={"raw_goal": "expire worker breaker during collect"},
        work_orders=[
            {
                "work_order_id": "wo-finished",
                "status": "dispatched",
                "worktree_path": str(repo),
                "branch": "main",
                "target_agent": "codex",
                "reviewer_agent": "claude",
                "expected_tests": ["python -m pytest tests/swarm/test_supervisor.py -q"],
            },
            {
                "work_order_id": "wo-dispatchable",
                "status": "leased",
                "worktree_path": str(repo),
                "branch": "main",
                "target_agent": "claude",
                "reviewer_agent": "codex",
            },
        ],
        status="active",
        metadata={
            WORKER_TYPE_CIRCUIT_BREAKER_POLICY_KEY: {
                "failure_threshold": 2,
                "reset_timeout_seconds": 60.0,
            },
            WORKER_TYPE_CIRCUIT_BREAKERS_KEY: {
                "claude": {
                    "status": "open",
                    "failure_count": 2,
                    "failure_threshold": 2,
                    "reset_timeout_seconds": 60.0,
                    "opened_at": expired_at.isoformat(),
                    "blocked_until": (expired_at + timedelta(minutes=1)).isoformat(),
                    "last_failure_reason": "agent_capacity",
                    "last_failure_detail": "Credit balance is too low",
                    "trip_count": 1,
                }
            },
        },
    )

    finished_worker = WorkerProcess(
        work_order_id="wo-finished",
        agent="codex",
        worktree_path=str(repo),
        branch="main",
        pid=445,
        exit_code=0,
        completed_at="2026-03-06T20:00:00+00:00",
        diff="diff --git a/test.py",
        changed_paths=["test.py"],
        commit_shas=["abc123"],
        tests_run=["python -m pytest tests/swarm/test_supervisor.py -q"],
    )
    launched_worker = WorkerProcess(
        work_order_id="wo-dispatchable",
        agent="claude",
        worktree_path=str(repo),
        branch="main",
        pid=446,
        initial_head="def456",
    )

    mock_launcher = MagicMock(spec=WorkerLauncher)
    mock_launcher.collect_finished = AsyncMock(return_value=[finished_worker])
    mock_launcher.launch = AsyncMock(return_value=launched_worker)

    supervisor = SwarmSupervisor(repo_root=repo, store=store, launcher=mock_launcher)

    completed = await supervisor.collect_finished_results(run_record["run_id"])

    assert [worker.work_order_id for worker in completed] == ["wo-finished"]
    updated = store.get_supervisor_run(run_record["run_id"])
    assert updated is not None
    breaker = updated["metadata"][WORKER_TYPE_CIRCUIT_BREAKERS_KEY]["claude"]
    assert breaker["status"] == "closed"
    assert breaker["failure_count"] == 0
    assert breaker["blocked_until"] is None
    assert breaker["last_reset_at"]

    launched = await supervisor.dispatch_workers(run_record["run_id"])

    assert len(launched) == 1
    assert launched[0].work_order_id == "wo-dispatchable"
    assert mock_launcher.launch.await_count == 1

    refreshed = store.get_supervisor_run(run_record["run_id"])
    assert refreshed is not None
    dispatchable = next(
        item for item in refreshed["work_orders"] if item["work_order_id"] == "wo-dispatchable"
    )
    assert dispatchable["status"] == "dispatched"
    assert dispatchable["target_agent"] == "claude"


@pytest.mark.asyncio
async def test_record_worker_type_success_noops_when_breaker_open(
    repo: Path, store: DevCoordinationStore
) -> None:
    blocked_until = datetime.now(UTC) + timedelta(minutes=5)
    run_record = store.create_supervisor_run(
        goal="late success leaves breaker open",
        target_branch="main",
        supervisor_agents={},
        approval_policy={},
        spec={"raw_goal": "late success leaves breaker open"},
        work_orders=[
            {
                "work_order_id": "wo-late-success",
                "status": "dispatched",
                "worktree_path": str(repo),
                "branch": "main",
                "target_agent": "claude",
                "reviewer_agent": "codex",
                "expected_tests": ["python -m pytest tests/swarm/test_supervisor.py -q"],
            }
        ],
        status="active",
        metadata={
            WORKER_TYPE_CIRCUIT_BREAKER_POLICY_KEY: {
                "failure_threshold": 2,
                "reset_timeout_seconds": 300.0,
            },
            WORKER_TYPE_CIRCUIT_BREAKERS_KEY: {
                "claude": {
                    "status": "open",
                    "failure_count": 2,
                    "failure_threshold": 2,
                    "reset_timeout_seconds": 300.0,
                    "opened_at": datetime.now(UTC).isoformat(),
                    "blocked_until": blocked_until.isoformat(),
                    "last_failure_reason": "agent_capacity",
                    "last_failure_detail": "Credit balance is too low",
                    "trip_count": 1,
                }
            },
        },
    )

    finished_worker = WorkerProcess(
        work_order_id="wo-late-success",
        agent="claude",
        worktree_path=str(repo),
        branch="main",
        pid=447,
        exit_code=0,
        completed_at="2026-03-06T20:00:00+00:00",
        diff="diff --git a/test.py",
        changed_paths=["test.py"],
        commit_shas=["abc123"],
        tests_run=["python -m pytest tests/swarm/test_supervisor.py -q"],
    )

    mock_launcher = MagicMock(spec=WorkerLauncher)
    mock_launcher.collect_finished = AsyncMock(return_value=[finished_worker])

    supervisor = SwarmSupervisor(repo_root=repo, store=store, launcher=mock_launcher)

    completed = await supervisor.collect_finished_results(run_record["run_id"])

    assert [worker.work_order_id for worker in completed] == ["wo-late-success"]
    updated = store.get_supervisor_run(run_record["run_id"])
    assert updated is not None
    work_order = updated["work_orders"][0]
    assert work_order["status"] == "completed"

    breaker = updated["metadata"][WORKER_TYPE_CIRCUIT_BREAKERS_KEY]["claude"]
    assert breaker["status"] == "open"
    assert breaker["failure_count"] == 2
    assert breaker["blocked_until"] == blocked_until.isoformat()
    assert breaker.get("last_reset_at") is None


@pytest.mark.asyncio
async def test_collect_finished_results_updates_progress_heartbeat(
    repo: Path, store: DevCoordinationStore
) -> None:
    old_progress = "2026-03-06T00:00:00+00:00"
    run_record = store.create_supervisor_run(
        goal="progress heartbeat",
        target_branch="main",
        supervisor_agents={},
        approval_policy={},
        spec={"raw_goal": "progress heartbeat"},
        work_orders=[
            {
                "work_order_id": "wo-progress",
                "status": "dispatched",
                "worktree_path": str(repo),
                "branch": "main",
                "target_agent": "codex",
                "pid": 321,
                "initial_head": "abc123",
                "dispatched_at": old_progress,
                "last_progress_at": old_progress,
                "progress_fingerprint": {
                    "head_sha": "abc123",
                    "changed_paths": [],
                    "diff_lines": 0,
                },
            }
        ],
        status="active",
    )

    mock_launcher = MagicMock(spec=WorkerLauncher)
    mock_launcher.collect_finished = AsyncMock(return_value=[])
    mock_launcher.snapshot_progress = AsyncMock(
        return_value={
            "pid_alive": True,
            "head_sha": "def456",
            "changed_paths": ["aragora/swarm/supervisor.py"],
            "diff_lines": 12,
        }
    )
    mock_launcher.config = SimpleNamespace(auto_commit=True, no_progress_timeout_seconds=120.0)

    supervisor = SwarmSupervisor(repo_root=repo, store=store, launcher=mock_launcher)

    with patch.object(WorkerLauncher, "collect_detached_result", new=AsyncMock(return_value=None)):
        completed = await supervisor.collect_finished_results(run_record["run_id"])

    assert completed == []
    updated = store.get_supervisor_run(run_record["run_id"])
    assert updated is not None
    work_order = updated["work_orders"][0]
    assert work_order["status"] == "dispatched"
    assert work_order["last_progress_at"] != old_progress
    assert work_order["head_sha"] == "def456"
    assert work_order["changed_paths"] == ["aragora/swarm/supervisor.py"]
    assert work_order["diff_lines"] == 12


@pytest.mark.asyncio
async def test_collect_finished_results_persists_log_tails_without_git_progress(
    repo: Path, store: DevCoordinationStore
) -> None:
    old_progress = (datetime.now(UTC) - timedelta(seconds=5)).isoformat()
    run_record = store.create_supervisor_run(
        goal="log tail heartbeat",
        target_branch="main",
        supervisor_agents={},
        approval_policy={},
        spec={"raw_goal": "log tail heartbeat"},
        work_orders=[
            {
                "work_order_id": "wo-log-tail",
                "status": "dispatched",
                "worktree_path": str(repo),
                "branch": "main",
                "target_agent": "codex",
                "pid": 321,
                "initial_head": "abc123",
                "dispatched_at": old_progress,
                "last_progress_at": old_progress,
                "progress_fingerprint": {
                    "head_sha": "abc123",
                    "changed_paths": [],
                    "diff_lines": 0,
                },
            }
        ],
        status="active",
    )

    mock_launcher = MagicMock(spec=WorkerLauncher)
    mock_launcher.collect_finished = AsyncMock(return_value=[])
    mock_launcher.snapshot_progress = AsyncMock(
        return_value={
            "pid_alive": True,
            "head_sha": "abc123",
            "changed_paths": [],
            "diff_lines": 0,
            "stdout_tail": "still validating\n",
            "stderr_tail": "warning line\n",
        }
    )
    mock_launcher.config = SimpleNamespace(auto_commit=True, no_progress_timeout_seconds=120.0)

    supervisor = SwarmSupervisor(repo_root=repo, store=store, launcher=mock_launcher)

    with patch.object(WorkerLauncher, "collect_detached_result", new=AsyncMock(return_value=None)):
        completed = await supervisor.collect_finished_results(run_record["run_id"])

    assert completed == []
    updated = store.get_supervisor_run(run_record["run_id"])
    assert updated is not None
    work_order = updated["work_orders"][0]
    assert work_order["status"] == "dispatched"
    assert work_order["last_progress_at"] == old_progress
    assert work_order["stdout_tail"] == "still validating\n"
    assert work_order["stderr_tail"] == "warning line\n"


@pytest.mark.asyncio
async def test_collect_finished_results_marks_dead_dispatched_worker_needs_human(
    repo: Path, store: DevCoordinationStore
) -> None:
    run_record = store.create_supervisor_run(
        goal="dead worker",
        target_branch="main",
        supervisor_agents={},
        approval_policy={},
        spec={"raw_goal": "dead worker"},
        work_orders=[
            {
                "work_order_id": "wo-dead",
                "status": "dispatched",
                "worktree_path": str(repo),
                "branch": "main",
                "target_agent": "codex",
                "pid": 9999,
                "initial_head": "abc123",
                "dispatched_at": "2026-03-06T00:00:00+00:00",
                "last_progress_at": "2026-03-06T00:00:00+00:00",
                "progress_fingerprint": {
                    "head_sha": "abc123",
                    "changed_paths": [],
                    "diff_lines": 0,
                },
            }
        ],
        status="active",
    )

    mock_launcher = MagicMock(spec=WorkerLauncher)
    mock_launcher.collect_finished = AsyncMock(return_value=[])
    mock_launcher.snapshot_progress = AsyncMock(
        return_value={
            "pid_alive": False,
            "head_sha": "abc123",
            "changed_paths": [],
            "diff_lines": 0,
        }
    )
    mock_launcher.config = SimpleNamespace(auto_commit=True, no_progress_timeout_seconds=120.0)

    supervisor = SwarmSupervisor(repo_root=repo, store=store, launcher=mock_launcher)

    with patch.object(WorkerLauncher, "collect_detached_result", new=AsyncMock(return_value=None)):
        completed = await supervisor.collect_finished_results(run_record["run_id"])

    assert completed == []
    updated = store.get_supervisor_run(run_record["run_id"])
    assert updated is not None
    work_order = updated["work_orders"][0]
    assert work_order["status"] == "needs_human"
    assert "without receipt or exit marker" in work_order["dispatch_error"]
    assert work_order["failure_reason"] == "worker_exited_without_receipt"
    assert "existing worktree" in work_order["blocking_question"]
    assert "pid" not in work_order


@pytest.mark.asyncio
async def test_collect_finished_results_falls_back_when_in_memory_collection_raises(
    repo: Path, store: DevCoordinationStore
) -> None:
    run_record = store.create_supervisor_run(
        goal="dead worker with broken in-memory collector",
        target_branch="main",
        supervisor_agents={},
        approval_policy={},
        spec={"raw_goal": "dead worker with broken in-memory collector"},
        work_orders=[
            {
                "work_order_id": "wo-dead-fallback",
                "status": "dispatched",
                "worktree_path": str(repo),
                "branch": "main",
                "target_agent": "codex",
                "pid": 9999,
                "initial_head": "abc123",
                "dispatched_at": "2026-03-06T00:00:00+00:00",
                "last_progress_at": "2026-03-06T00:00:00+00:00",
                "progress_fingerprint": {
                    "head_sha": "abc123",
                    "changed_paths": [],
                    "diff_lines": 0,
                },
            }
        ],
        status="active",
    )

    mock_launcher = MagicMock(spec=WorkerLauncher)
    mock_launcher.collect_finished = AsyncMock(side_effect=RuntimeError("event loop is closed"))
    mock_launcher.snapshot_progress = AsyncMock(
        return_value={
            "pid_alive": False,
            "head_sha": "abc123",
            "changed_paths": [],
            "diff_lines": 0,
        }
    )
    mock_launcher.config = SimpleNamespace(auto_commit=True, no_progress_timeout_seconds=120.0)

    supervisor = SwarmSupervisor(repo_root=repo, store=store, launcher=mock_launcher)

    with patch.object(WorkerLauncher, "collect_detached_result", new=AsyncMock(return_value=None)):
        completed = await supervisor.collect_finished_results(run_record["run_id"])

    assert completed == []
    updated = store.get_supervisor_run(run_record["run_id"])
    assert updated is not None
    work_order = updated["work_orders"][0]
    assert work_order["status"] == "needs_human"
    assert "without receipt or exit marker" in work_order["dispatch_error"]
    assert work_order["failure_reason"] == "worker_exited_without_receipt"
    assert "pid" not in work_order


@pytest.mark.asyncio
async def test_collect_finished_results_isolates_initial_detached_collection_failure(
    repo: Path, store: DevCoordinationStore
) -> None:
    run_record = store.create_supervisor_run(
        goal="mixed detached collection failure",
        target_branch="main",
        supervisor_agents={},
        approval_policy={},
        spec={"raw_goal": "mixed detached collection failure"},
        work_orders=[
            {
                "work_order_id": "wo-good-detached-isolation",
                "status": "dispatched",
                "worktree_path": str(repo),
                "branch": "main",
                "target_agent": "codex",
                "pid": 1001,
                "initial_head": "abc123",
                "lease_id": "lease-good",
                "owner_session_id": "session-good",
            },
            {
                "work_order_id": "wo-bad-detached-isolation",
                "status": "dispatched",
                "worktree_path": str(repo),
                "branch": "main",
                "target_agent": "codex",
                "pid": 1002,
                "initial_head": "abc123",
            },
        ],
        status="active",
    )

    finished_worker = WorkerProcess(
        work_order_id="wo-good-detached-isolation",
        agent="codex",
        worktree_path=str(repo),
        branch="main",
        lease_id="lease-good",
        session_id="session-good",
        exit_code=0,
        head_sha="def456",
        commit_shas=["def456"],
        changed_paths=["aragora/swarm/supervisor.py"],
        completed_at=datetime.now(UTC).isoformat(),
    )

    mock_launcher = MagicMock(spec=WorkerLauncher)
    mock_launcher.collect_finished = AsyncMock(return_value=[finished_worker])
    mock_launcher.snapshot_progress = AsyncMock(
        return_value={
            "pid_alive": False,
            "head_sha": "abc123",
            "changed_paths": [],
            "diff_lines": 0,
        }
    )
    mock_launcher.config = SimpleNamespace(auto_commit=True, no_progress_timeout_seconds=120.0)

    supervisor = SwarmSupervisor(repo_root=repo, store=store, launcher=mock_launcher)

    async def _collect_detached_result(**kwargs):
        if kwargs["work_order_id"] == "wo-bad-detached-isolation":
            raise RuntimeError("broken detached collection")
        return None

    with patch.object(
        WorkerLauncher,
        "collect_detached_result",
        new=AsyncMock(side_effect=_collect_detached_result),
    ):
        completed = await supervisor.collect_finished_results(run_record["run_id"])

    assert [worker.work_order_id for worker in completed] == ["wo-good-detached-isolation"]
    updated = store.get_supervisor_run(run_record["run_id"])
    assert updated is not None
    good = next(
        item
        for item in updated["work_orders"]
        if item["work_order_id"] == "wo-good-detached-isolation"
    )
    bad = next(
        item
        for item in updated["work_orders"]
        if item["work_order_id"] == "wo-bad-detached-isolation"
    )
    assert good["status"] != "dispatched"
    assert good["commit_shas"] == ["def456"]
    assert good["changed_paths"] == ["aragora/swarm/supervisor.py"]
    assert bad["status"] == "dispatched"
    assert bad["head_sha"] == "abc123"
    assert bad["changed_paths"] == []


@pytest.mark.asyncio
async def test_collect_finished_results_isolates_progress_snapshot_failure(
    repo: Path, store: DevCoordinationStore
) -> None:
    run_record = store.create_supervisor_run(
        goal="mixed progress snapshot failure",
        target_branch="main",
        supervisor_agents={},
        approval_policy={},
        spec={"raw_goal": "mixed progress snapshot failure"},
        work_orders=[
            {
                "work_order_id": "wo-good-progress-isolation",
                "status": "dispatched",
                "worktree_path": str(repo),
                "branch": "main",
                "target_agent": "codex",
                "pid": 2001,
                "initial_head": "abc123",
                "lease_id": "lease-good-progress",
                "owner_session_id": "session-good-progress",
            },
            {
                "work_order_id": "wo-bad-progress-isolation",
                "status": "dispatched",
                "worktree_path": str(repo),
                "branch": "main",
                "target_agent": "codex",
                "pid": 2002,
                "initial_head": "abc123",
            },
        ],
        status="active",
    )

    finished_worker = WorkerProcess(
        work_order_id="wo-good-progress-isolation",
        agent="codex",
        worktree_path=str(repo),
        branch="main",
        lease_id="lease-good-progress",
        session_id="session-good-progress",
        exit_code=0,
        head_sha="fedcba",
        commit_shas=["fedcba"],
        changed_paths=["aragora/swarm/worker_launcher.py"],
        completed_at=datetime.now(UTC).isoformat(),
    )

    mock_launcher = MagicMock(spec=WorkerLauncher)
    mock_launcher.collect_finished = AsyncMock(return_value=[finished_worker])
    mock_launcher.config = SimpleNamespace(auto_commit=True, no_progress_timeout_seconds=120.0)

    async def _snapshot_progress(item):
        if item["work_order_id"] == "wo-bad-progress-isolation":
            raise RuntimeError("snapshot exploded")
        return {
            "pid_alive": True,
            "head_sha": "abc123",
            "changed_paths": [],
            "diff_lines": 0,
        }

    mock_launcher.snapshot_progress = AsyncMock(side_effect=_snapshot_progress)

    supervisor = SwarmSupervisor(repo_root=repo, store=store, launcher=mock_launcher)

    with patch.object(WorkerLauncher, "collect_detached_result", new=AsyncMock(return_value=None)):
        completed = await supervisor.collect_finished_results(run_record["run_id"])

    assert [worker.work_order_id for worker in completed] == ["wo-good-progress-isolation"]
    updated = store.get_supervisor_run(run_record["run_id"])
    assert updated is not None
    good = next(
        item
        for item in updated["work_orders"]
        if item["work_order_id"] == "wo-good-progress-isolation"
    )
    bad = next(
        item
        for item in updated["work_orders"]
        if item["work_order_id"] == "wo-bad-progress-isolation"
    )
    assert good["status"] != "dispatched"
    assert good["commit_shas"] == ["fedcba"]
    assert good["changed_paths"] == ["aragora/swarm/worker_launcher.py"]
    assert bad["status"] == "dispatched"


@pytest.mark.asyncio
async def test_collect_finished_results_marks_no_progress_timeout_needs_human(
    repo: Path, store: DevCoordinationStore
) -> None:
    stale_time = (datetime.now(UTC) - timedelta(minutes=10)).isoformat()
    run_record = store.create_supervisor_run(
        goal="no progress timeout",
        target_branch="main",
        supervisor_agents={},
        approval_policy={},
        spec={"raw_goal": "no progress timeout"},
        work_orders=[
            {
                "work_order_id": "wo-stalled",
                "status": "dispatched",
                "worktree_path": str(repo),
                "branch": "main",
                "target_agent": "claude",
                "pid": 4242,
                "initial_head": "abc123",
                "dispatched_at": stale_time,
                "last_progress_at": stale_time,
                "progress_fingerprint": {
                    "head_sha": "abc123",
                    "changed_paths": [],
                    "diff_lines": 0,
                },
            }
        ],
        status="active",
    )

    mock_launcher = MagicMock(spec=WorkerLauncher)
    mock_launcher.collect_finished = AsyncMock(return_value=[])
    mock_launcher.snapshot_progress = AsyncMock(
        return_value={
            "pid_alive": True,
            "head_sha": "abc123",
            "changed_paths": [],
            "diff_lines": 0,
        }
    )
    mock_launcher.config = SimpleNamespace(auto_commit=True, no_progress_timeout_seconds=60.0)

    supervisor = SwarmSupervisor(repo_root=repo, store=store, launcher=mock_launcher)

    with patch.object(WorkerLauncher, "collect_detached_result", new=AsyncMock(return_value=None)):
        completed = await supervisor.collect_finished_results(run_record["run_id"])

    assert completed == []
    updated = store.get_supervisor_run(run_record["run_id"])
    assert updated is not None
    work_order = updated["work_orders"][0]
    assert work_order["status"] == "needs_human"
    assert "no-progress timeout" in work_order["dispatch_error"]
    assert work_order["failure_reason"] == "worker_no_progress_timeout"
    assert "stalled lane" in work_order["blocking_question"]


def test_session_key_unique_per_work_order() -> None:
    """Regression: subtask_1/subtask_2/subtask_3 must NOT collide into one worktree."""
    run_id = "abcdef12-3456"
    work_orders = [
        {"work_order_id": "subtask_1"},
        {"work_order_id": "subtask_2"},
        {"work_order_id": "subtask_3"},
    ]
    keys = set()
    for wo in work_orders:
        wo_id = str(wo.get("work_order_id", "task"))
        session_key = f"swarm-{run_id[:8]}-{wo_id}"
        keys.add(session_key)
    assert len(keys) == 3, f"Session keys collide: {keys}"
    assert "swarm-abcdef12-subtask_1" in keys
    assert "swarm-abcdef12-subtask_2" in keys
    assert "swarm-abcdef12-subtask_3" in keys


def test_worker_prompt_includes_boss_lane_contract() -> None:
    prompt = WorkerLauncher._build_prompt(
        {
            "title": "Implement boss-facing reporter output",
            "description": "Emit stable coordinator output for one lane.",
            "file_scope": ["aragora/swarm/reporter.py", "tests/swarm/test_supervisor.py"],
            "expected_tests": ["python -m pytest tests/swarm/test_supervisor.py -q"],
            "approval_required": True,
            "lease_id": "lease-123",
            "metadata": {
                "acceptance_criteria": ["Output includes run_id and next actions"],
                "constraints": ["Do not widen beyond the swarm package"],
            },
        }
    )

    assert "Aragora-managed CLI worker lane" in prompt
    assert "FILE SCOPE GUIDANCE" in prompt
    assert "Expected validation:" in prompt
    assert "Acceptance criteria:" in prompt
    assert "Constraints:" in prompt
    assert "Decision boundary:" in prompt
    assert "Receipt expectation:" in prompt
    assert "Lease id: lease-123" in prompt
    assert "Stop condition:" in prompt


def test_refresh_run_reaps_expired_leases(repo: Path, store: DevCoordinationStore) -> None:
    """refresh_run proactively reaps TTL-expired leases so stuck waiting_conflict
    work orders can recover even when no new claim_lease calls occur."""
    # Create a lease with normal TTL, then manually expire it
    expired = store.claim_lease(
        task_id="old-task",
        title="Old task",
        owner_agent="codex",
        owner_session_id="dead-session",
        branch="codex/dead-branch",
        worktree_path=str(repo / "dead-wt"),
        claimed_paths=["some/file.py"],
    )
    assert expired.lease_id in {l.lease_id for l in store.list_active_leases()}
    # Backdate expiry to the past so reap_expired_leases picks it up
    conn = store._connect()
    try:
        conn.execute(
            "UPDATE leases SET expires_at = ? WHERE lease_id = ?",
            ("2020-01-01T00:00:00+00:00", expired.lease_id),
        )
        conn.commit()
    finally:
        conn.close()

    # Set up a minimal supervisor with a live run
    lifecycle = MagicMock()
    session_path = repo / "wt-fresh"
    session_path.mkdir()
    lifecycle.ensure_managed_worktree.return_value = ManagedWorktreeSession(
        session_id="fresh-session",
        agent="codex",
        branch="codex/fresh-branch",
        path=session_path,
        created=True,
        reconcile_status="up_to_date",
        payload={},
    )

    decomposer = MagicMock()
    decomposer.analyze.return_value = TaskDecomposition(
        original_task="Goal",
        complexity_score=2,
        complexity_level="low",
        should_decompose=True,
        subtasks=[
            SubTask(
                id="fresh-lane",
                title="Fresh work",
                description="Do fresh work.",
                file_scope=["other/file.py"],
            )
        ],
    )

    supervisor = SwarmSupervisor(
        repo_root=repo,
        store=store,
        lifecycle=lifecycle,
        decomposer=decomposer,
    )
    run = supervisor.start_run(spec=SwarmSpec(raw_goal="Goal", refined_goal="Goal"))

    # refresh_run should have reaped the expired lease
    supervisor.refresh_run(run.run_id)
    active_lease_ids = {l.lease_id for l in store.list_active_leases()}
    assert expired.lease_id not in active_lease_ids


def test_refresh_run_marks_expired_leased_work_order(
    repo: Path, store: DevCoordinationStore
) -> None:
    """TTL expiry must reconcile the supervisor lane, not just release the lease."""
    lifecycle = MagicMock()
    session_path = repo / "wt-expired"
    session_path.mkdir()
    lifecycle.ensure_managed_worktree.return_value = ManagedWorktreeSession(
        session_id="expired-worker",
        agent="codex",
        branch="codex/expired-worker",
        path=session_path,
        created=True,
        reconcile_status="up_to_date",
        payload={},
    )

    decomposer = MagicMock()
    decomposer.analyze.return_value = TaskDecomposition(
        original_task="Goal",
        complexity_score=2,
        complexity_level="low",
        should_decompose=True,
        subtasks=[
            SubTask(
                id="expired-lane",
                title="Expired work",
                description="Do expired work.",
                file_scope=["other/file.py"],
            )
        ],
    )

    supervisor = SwarmSupervisor(
        repo_root=repo,
        store=store,
        lifecycle=lifecycle,
        decomposer=decomposer,
    )
    run = supervisor.start_run(spec=SwarmSpec(raw_goal="Goal", refined_goal="Goal"))
    lease_id = str(run.work_orders[0]["lease_id"])
    assert lease_id

    conn = store._connect()
    try:
        conn.execute(
            "UPDATE leases SET expires_at = ? WHERE lease_id = ?",
            ("2020-01-01T00:00:00+00:00", lease_id),
        )
        conn.commit()
    finally:
        conn.close()

    refreshed = supervisor.refresh_run(run.run_id)

    assert refreshed.status == "needs_human"
    work_order = refreshed.work_orders[0]
    assert work_order["status"] == "needs_human"
    assert work_order["failure_reason"] == "expired_lease_reaped"
    assert lease_id not in {lease.lease_id for lease in store.list_active_leases()}


def test_refresh_run_requeues_conflict_only_needs_human_when_fleet_claims_are_stale(
    repo: Path, store: DevCoordinationStore
) -> None:
    session_path = repo / "wt-fleet-requeue"
    session_path.mkdir()
    lifecycle = MagicMock()
    lifecycle.ensure_managed_worktree.return_value = ManagedWorktreeSession(
        session_id="swarm-fleet-requeue",
        agent="codex",
        branch="codex/swarm-fleet-requeue",
        path=session_path,
        created=True,
        reconcile_status="up_to_date",
        payload={},
    )
    supervisor = SwarmSupervisor(repo_root=repo, store=store, lifecycle=lifecycle)
    run_record = store.create_supervisor_run(
        goal="requeue stale fleet conflicts",
        target_branch="main",
        supervisor_agents={},
        approval_policy={},
        spec={"raw_goal": "requeue stale fleet conflicts"},
        metadata={"max_concurrency": 1},
        work_orders=[
            {
                "work_order_id": "wo-stale-fleet",
                "title": "Stale fleet lane",
                "description": "Stale fleet lane",
                "status": "needs_human",
                "target_agent": "codex",
                "reviewer_agent": "claude",
                "file_scope": ["aragora/swarm/reconciler.py"],
                "expected_tests": ["python -m pytest tests/swarm/test_reconciler.py -q"],
                "exit_code": 143,
                "diff_lines": 9,
                "scope_violation": {"violations": [{"path": "aragora/swarm/reconciler.py"}]},
                "conflicts": [
                    {
                        "source": "fleet_claim",
                        "session_id": "swarm-missing-session",
                        "path": "aragora/swarm/reconciler.py",
                    }
                ],
            }
        ],
        status="needs_human",
    )

    refreshed = supervisor.refresh_run(run_record["run_id"])

    work_order = refreshed.work_orders[0]
    assert work_order["status"] == "leased"
    assert work_order["owner_session_id"] == "swarm-fleet-requeue"
    assert "conflicts" not in work_order
    assert "exit_code" not in work_order
    assert "diff_lines" not in work_order
    assert "scope_violation" not in work_order


def test_refresh_run_reaps_stale_leased_work_order(repo: Path, store: DevCoordinationStore) -> None:
    """refresh_run should not leave dead leased work orders active forever."""
    lifecycle = MagicMock()
    session_path = repo / "wt-stale"
    session_path.mkdir()
    lifecycle.ensure_managed_worktree.return_value = ManagedWorktreeSession(
        session_id="stale-worker",
        agent="codex",
        branch="codex/stale-worker",
        path=session_path,
        created=True,
        reconcile_status="up_to_date",
        payload={},
    )

    decomposer = MagicMock()
    decomposer.analyze.return_value = TaskDecomposition(
        original_task="Goal",
        complexity_score=2,
        complexity_level="low",
        should_decompose=True,
        subtasks=[
            SubTask(
                id="stale-lane",
                title="Stale work",
                description="Do stale work.",
                file_scope=["other/file.py"],
            )
        ],
    )

    supervisor = SwarmSupervisor(
        repo_root=repo,
        store=store,
        lifecycle=lifecycle,
        decomposer=decomposer,
    )
    run = supervisor.start_run(spec=SwarmSpec(raw_goal="Goal", refined_goal="Goal"))
    lease_id = str(run.work_orders[0]["lease_id"])
    assert lease_id

    conn = store._connect()
    try:
        conn.execute(
            "UPDATE leases SET updated_at = ? WHERE lease_id = ?",
            ("2020-01-01T00:00:00+00:00", lease_id),
        )
        conn.commit()
    finally:
        conn.close()

    refreshed = supervisor.refresh_run(run.run_id)

    assert refreshed.status == "needs_human"
    work_order = refreshed.work_orders[0]
    assert work_order["status"] == "needs_human"
    assert work_order["failure_reason"] == "stale_lease_reaped"
    assert lease_id not in {lease.lease_id for lease in store.list_active_leases()}


# ---------------------------------------------------------------------------
# Pre-reap salvage path regression tests
# ---------------------------------------------------------------------------


def test_pre_reap_salvage_backfills_commit_shas_from_dead_worker(
    repo: Path, store: DevCoordinationStore
) -> None:
    """When a dispatched worker's PID is dead but it committed, salvage the SHAs."""
    import subprocess

    lifecycle = MagicMock()
    session_path = repo / "wt-salvage"
    session_path.mkdir()
    lifecycle.ensure_managed_worktree.return_value = ManagedWorktreeSession(
        session_id="swarm-salvage",
        agent="codex",
        branch="codex/swarm-salvage",
        path=session_path,
        created=True,
        reconcile_status="up_to_date",
        payload={},
    )
    decomposer = MagicMock()
    decomposer.analyze.return_value = TaskDecomposition(
        original_task="salvage test",
        complexity_score=2,
        complexity_level="low",
        should_decompose=False,
        subtasks=[
            SubTask(
                id="wo-1",
                title="Test salvage",
                description="Test pre-reap salvage",
                file_scope=["salvage_test.py"],
            )
        ],
    )
    supervisor = SwarmSupervisor(
        repo_root=repo, store=store, lifecycle=lifecycle, decomposer=decomposer
    )
    spec = SwarmSpec(raw_goal="salvage test", file_scope_hints=["salvage_test.py"])
    run = supervisor.start_run(spec=spec)

    # Simulate a dispatched work order with a dead PID
    work_orders = run.work_orders
    assert len(work_orders) >= 1
    wo = work_orders[0]

    # Create a real git worktree with a commit
    wt_path = repo / "salvage-wt"
    subprocess.run(
        ["git", "worktree", "add", str(wt_path), "-b", "salvage-branch"],
        cwd=repo,
        capture_output=True,
    )
    test_file = wt_path / "salvage_test.py"
    test_file.write_text("# salvage test\n")
    subprocess.run(["git", "add", "salvage_test.py"], cwd=wt_path, capture_output=True)
    subprocess.run(["git", "commit", "-m", "salvage commit"], cwd=wt_path, capture_output=True)

    initial_head = subprocess.run(
        ["git", "rev-parse", "HEAD~1"], cwd=wt_path, capture_output=True, text=True
    ).stdout.strip()

    # Update work order to look dispatched with a dead PID
    record = store.get_supervisor_run(run.run_id)
    record["work_orders"][0]["status"] = "dispatched"
    record["work_orders"][0]["pid"] = 99999  # dead PID
    record["work_orders"][0]["worktree_path"] = str(wt_path)
    record["work_orders"][0]["initial_head"] = initial_head
    store.update_supervisor_run(run.run_id, work_orders=record["work_orders"])

    # Run pre-reap salvage
    supervisor._collect_finished_workers_sync(run.run_id)

    # Verify the dead worker was fully reconciled into a salvage completion
    updated = store.get_supervisor_run(run.run_id)
    wo_updated = updated["work_orders"][0]
    assert wo_updated["status"] == "completed"
    assert wo_updated["worker_outcome"] == "crash_with_salvage"
    assert wo_updated["review_status"] == "pending_heterogeneous_review"
    assert wo_updated.get("commit_shas"), "Expected commit SHAs to be salvaged"
    assert len(wo_updated["commit_shas"]) >= 1
    assert wo_updated.get("head_sha"), "Expected head SHA to be set"
    assert wo_updated.get("receipt_id"), "Expected salvage receipt to be backfilled"

    # Cleanup
    subprocess.run(
        ["git", "worktree", "remove", str(wt_path), "--force"], cwd=repo, capture_output=True
    )


def test_pre_reap_salvage_skips_live_workers(repo: Path, store: DevCoordinationStore) -> None:
    """Pre-reap salvage should NOT touch workers with live PIDs."""
    import os

    session_path = repo / "wt-live-test"
    session_path.mkdir()
    lifecycle = MagicMock()
    lifecycle.ensure_managed_worktree.return_value = ManagedWorktreeSession(
        session_id="swarm-live",
        agent="codex",
        branch="codex/swarm-live",
        path=session_path,
        created=True,
        reconcile_status="up_to_date",
        payload={},
    )
    decomposer = MagicMock()
    decomposer.analyze.return_value = TaskDecomposition(
        original_task="live test",
        complexity_score=2,
        complexity_level="low",
        should_decompose=False,
        subtasks=[SubTask(id="wo-1", title="T", description="D", file_scope=["t.py"])],
    )
    supervisor = SwarmSupervisor(
        repo_root=repo, store=store, lifecycle=lifecycle, decomposer=decomposer
    )

    # Create a fake run with dispatched work order using OUR PID (definitely alive)
    spec = SwarmSpec(raw_goal="live test", file_scope_hints=["t.py"])
    run = supervisor.start_run(spec=spec)
    record = store.get_supervisor_run(run.run_id)
    record["work_orders"][0]["status"] = "dispatched"
    record["work_orders"][0]["pid"] = os.getpid()  # THIS process — alive
    record["work_orders"][0]["worktree_path"] = str(repo)
    store.update_supervisor_run(run.run_id, work_orders=record["work_orders"])

    # Run pre-reap — should skip because PID is alive
    supervisor._collect_finished_workers_sync(run.run_id)

    # Verify no commit SHAs were added
    updated = store.get_supervisor_run(run.run_id)
    assert not updated["work_orders"][0].get("commit_shas")


def test_pre_reap_salvage_ignores_invalid_pid_probe(
    repo: Path, store: DevCoordinationStore
) -> None:
    """Invalid PID metadata must not hit os.kill(0) or process-group probes."""
    session_path = repo / "wt-invalid-pid-test"
    session_path.mkdir()
    lifecycle = MagicMock()
    lifecycle.ensure_managed_worktree.return_value = ManagedWorktreeSession(
        session_id="swarm-invalid-pid",
        agent="codex",
        branch="codex/swarm-invalid-pid",
        path=session_path,
        created=True,
        reconcile_status="up_to_date",
        payload={},
    )
    decomposer = MagicMock()
    decomposer.analyze.return_value = TaskDecomposition(
        original_task="invalid pid test",
        complexity_score=2,
        complexity_level="low",
        should_decompose=False,
        subtasks=[SubTask(id="wo-1", title="T", description="D", file_scope=["t.py"])],
    )
    supervisor = SwarmSupervisor(
        repo_root=repo, store=store, lifecycle=lifecycle, decomposer=decomposer
    )

    spec = SwarmSpec(raw_goal="invalid pid test", file_scope_hints=["t.py"])
    run = supervisor.start_run(spec=spec)
    record = store.get_supervisor_run(run.run_id)
    record["work_orders"][0]["status"] = "dispatched"
    record["work_orders"][0]["pid"] = 0
    record["work_orders"][0]["worktree_path"] = str(repo)
    store.update_supervisor_run(run.run_id, work_orders=record["work_orders"])

    with (
        patch("os.kill") as mock_kill,
        patch.object(
            supervisor, "_build_dead_worker_salvage_result", return_value=None
        ) as mock_salvage,
    ):
        supervisor._collect_finished_workers_sync(run.run_id)

    mock_kill.assert_not_called()
    mock_salvage.assert_called_once()


def test_pre_reap_salvage_handles_missing_worktree(repo: Path, store: DevCoordinationStore) -> None:
    """Pre-reap salvage should handle missing worktree paths gracefully."""
    session_path = repo / "wt-missing-test"
    session_path.mkdir()
    lifecycle = MagicMock()
    lifecycle.ensure_managed_worktree.return_value = ManagedWorktreeSession(
        session_id="swarm-missing",
        agent="codex",
        branch="codex/swarm-missing",
        path=session_path,
        created=True,
        reconcile_status="up_to_date",
        payload={},
    )
    decomposer = MagicMock()
    decomposer.analyze.return_value = TaskDecomposition(
        original_task="missing wt",
        complexity_score=2,
        complexity_level="low",
        should_decompose=False,
        subtasks=[SubTask(id="wo-1", title="T", description="D", file_scope=["t.py"])],
    )
    supervisor = SwarmSupervisor(
        repo_root=repo, store=store, lifecycle=lifecycle, decomposer=decomposer
    )

    spec = SwarmSpec(raw_goal="missing wt", file_scope_hints=["t.py"])
    run = supervisor.start_run(spec=spec)
    record = store.get_supervisor_run(run.run_id)
    record["work_orders"][0]["status"] = "dispatched"
    record["work_orders"][0]["pid"] = 99999  # dead PID
    record["work_orders"][0]["worktree_path"] = "/nonexistent/path"
    store.update_supervisor_run(run.run_id, work_orders=record["work_orders"])

    # Should not crash
    supervisor._collect_finished_workers_sync(run.run_id)

    # No SHAs salvaged from nonexistent path
    updated = store.get_supervisor_run(run.run_id)
    assert not updated["work_orders"][0].get("commit_shas")


@pytest.mark.asyncio
async def test_refresh_run_async_context_reconciles_dead_worker_salvage(
    repo: Path, store: DevCoordinationStore
) -> None:
    session_path = repo / "wt-async-salvage"
    session_path.mkdir()
    lifecycle = MagicMock()
    lifecycle.ensure_managed_worktree.return_value = ManagedWorktreeSession(
        session_id="swarm-async-salvage",
        agent="codex",
        branch="codex/swarm-async-salvage",
        path=session_path,
        created=True,
        reconcile_status="up_to_date",
        payload={},
    )
    decomposer = MagicMock()
    decomposer.analyze.return_value = TaskDecomposition(
        original_task="async salvage",
        complexity_score=2,
        complexity_level="low",
        should_decompose=False,
        subtasks=[
            SubTask(
                id="wo-1",
                title="Async salvage lane",
                description="Prove sync reconciliation in async refresh",
                file_scope=["salvage_async.py"],
            )
        ],
    )
    supervisor = SwarmSupervisor(
        repo_root=repo, store=store, lifecycle=lifecycle, decomposer=decomposer
    )
    run = supervisor.start_run(
        spec=SwarmSpec(raw_goal="async salvage", file_scope_hints=["salvage_async.py"])
    )

    wt_path = repo / "async-salvage-wt"
    subprocess.run(
        ["git", "worktree", "add", str(wt_path), "-b", "async-salvage-branch"],
        cwd=repo,
        capture_output=True,
        check=False,
    )
    test_file = wt_path / "salvage_async.py"
    test_file.write_text("print('salvage')\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "salvage_async.py"], cwd=wt_path, capture_output=True, check=False
    )
    subprocess.run(
        ["git", "commit", "-m", "async salvage commit"],
        cwd=wt_path,
        capture_output=True,
        check=False,
    )
    initial_head = subprocess.run(
        ["git", "rev-parse", "HEAD~1"],
        cwd=wt_path,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()

    record = store.get_supervisor_run(run.run_id)
    assert record is not None
    record["work_orders"][0]["status"] = "dispatched"
    record["work_orders"][0]["pid"] = 99999
    record["work_orders"][0]["worktree_path"] = str(wt_path)
    record["work_orders"][0]["initial_head"] = initial_head
    store.update_supervisor_run(run.run_id, work_orders=record["work_orders"])

    refreshed = supervisor.refresh_run(run.run_id)

    work_order = refreshed.work_orders[0]
    assert work_order["status"] == "completed"
    assert work_order["worker_outcome"] == "crash_with_salvage"
    assert work_order["receipt_id"]
    assert work_order["commit_shas"]

    subprocess.run(
        ["git", "worktree", "remove", str(wt_path), "--force"],
        cwd=repo,
        capture_output=True,
        check=False,
    )


@pytest.mark.asyncio
async def test_refresh_run_async_context_collects_finished_in_memory_worker(
    repo: Path, store: DevCoordinationStore
) -> None:
    lease = store.claim_lease(
        task_id="async-finished-lane",
        title="Async finished lane",
        owner_agent="codex",
        owner_session_id="async-finished-session",
        branch="main",
        worktree_path=str(repo),
        claimed_paths=["docs/notes.md"],
    )
    run_record = store.create_supervisor_run(
        goal="async finished worker",
        target_branch="main",
        supervisor_agents={},
        approval_policy={},
        spec={"raw_goal": "async finished worker"},
        work_orders=[
            {
                "work_order_id": "wo-async-finished",
                "status": "dispatched",
                "worktree_path": str(repo),
                "branch": "main",
                "target_agent": "codex",
                "owner_session_id": "async-finished-session",
                "lease_id": lease.lease_id,
                "pid": 4242,
                "initial_head": "base123",
                "review_status": "pending",
                "file_scope": ["docs/notes.md"],
            }
        ],
        status="active",
    )

    finished_worker = WorkerProcess(
        work_order_id="wo-async-finished",
        agent="codex",
        worktree_path=str(repo),
        branch="main",
        pid=4242,
        exit_code=0,
        completed_at="2026-03-31T12:00:00+00:00",
        diff="diff --git a/docs/notes.md",
        initial_head="base123",
        head_sha="abc12345",
        changed_paths=["docs/notes.md"],
        commit_shas=["abc12345"],
    )

    mock_launcher = MagicMock(spec=WorkerLauncher)
    mock_launcher.get_worker.return_value = finished_worker
    mock_launcher.collect_finished_sync.return_value = [finished_worker]
    mock_launcher.config = SimpleNamespace(auto_commit=False, no_progress_timeout_seconds=120.0)

    supervisor = SwarmSupervisor(repo_root=repo, store=store, launcher=mock_launcher)

    refreshed = supervisor.refresh_run(run_record["run_id"])

    mock_launcher.collect_finished_sync.assert_called_once_with(
        work_order_ids=["wo-async-finished"]
    )
    work_order = refreshed.work_orders[0]
    assert work_order["status"] == "completed"
    assert work_order["review_status"] == "pending_heterogeneous_review"
    assert work_order["receipt_id"]
    assert work_order["commit_shas"] == ["abc12345"]
    assert work_order["head_sha"] == "abc12345"


# --- Finding 1: Filtered pytest commands with selectors ---


def test_pytest_command_has_selectors_detects_k_flag() -> None:
    assert SwarmSupervisor._pytest_command_has_selectors("python -m pytest tests/ -k 'not slow'")


def test_pytest_command_has_selectors_detects_m_flag() -> None:
    assert SwarmSupervisor._pytest_command_has_selectors("pytest tests/ -m integration")


def test_pytest_command_has_selectors_false_for_plain_command() -> None:
    assert not SwarmSupervisor._pytest_command_has_selectors("python -m pytest tests/swarm/ -x -q")


# --- Finding 4: Docs-only path allowlist ---


def test_is_docs_only_path_accepts_docs_prefix() -> None:
    assert SwarmSupervisor._is_docs_only_path("docs/STATUS.md")


def test_is_docs_only_path_accepts_docs_site_prefix() -> None:
    assert SwarmSupervisor._is_docs_only_path("docs-site/docs/guide.md")


def test_is_docs_only_path_accepts_changelog() -> None:
    assert SwarmSupervisor._is_docs_only_path("CHANGELOG.md")


def test_is_docs_only_path_rejects_claude_md() -> None:
    assert not SwarmSupervisor._is_docs_only_path("CLAUDE.md")


def test_is_docs_only_path_rejects_readme_md() -> None:
    assert not SwarmSupervisor._is_docs_only_path("README.md")


def test_is_docs_only_path_rejects_scripts_txt() -> None:
    assert not SwarmSupervisor._is_docs_only_path("scripts/pii_allowlist.txt")


def test_is_docs_only_path_rejects_arbitrary_md_in_src() -> None:
    assert not SwarmSupervisor._is_docs_only_path("aragora/notes.md")


@pytest.mark.asyncio
async def test_kill_worker_ignores_invalid_pid_metadata() -> None:
    supervisor = SwarmSupervisor(
        repo_root=Path("/tmp/repo"),
        store=MagicMock(),
        launcher=MagicMock(spec=WorkerLauncher),
    )
    item = {"pid": 0}

    with patch("os.kill") as mock_kill:
        await supervisor._kill_worker(item)

    mock_kill.assert_not_called()
    assert "pid" not in item
