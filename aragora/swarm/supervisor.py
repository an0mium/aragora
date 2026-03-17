"""Supervisor-driven Codex/Claude swarm orchestration.

Builds bounded work orders from a SwarmSpec, provisions managed worktrees for
Codex/Claude execution targets, claims bounded leases, and persists a
SupervisorRun in the existing development coordination store.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from aragora.nomic.approval import ApprovalLevel, ApprovalPolicy
from aragora.nomic.dev_coordination import (
    DevCoordinationStore,
    FileScopeViolationError,
    LeaseConflictError,
    LeaseStatus,
)
from aragora.nomic.pipeline_bridge import BoundedWorkOrder, NomicPipelineBridge
from aragora.nomic.task_decomposer import SubTask, TaskDecomposer
from aragora.swarm.spec import SwarmSpec
from aragora.swarm.worker_launcher import SESSION_ARTIFACTS, WorkerLauncher, WorkerProcess
from aragora.worktree.lifecycle import WorktreeLifecycleService

UTC = timezone.utc
logger = logging.getLogger(__name__)

WORKER_TYPE_CIRCUIT_BREAKERS_KEY = "worker_type_circuit_breakers"
WORKER_TYPE_CIRCUIT_BREAKER_POLICY_KEY = "worker_type_circuit_breaker_policy"
CAMPAIGN_OUTCOME_METADATA_KEY = "campaign_outcome"
CAMPAIGN_REQUEUE_ELIGIBLE_METADATA_KEY = "campaign_requeue_eligible"
CAMPAIGN_BLOCKERS_METADATA_KEY = "campaign_blockers"
DEFAULT_BREAKER_FAILURE_THRESHOLD = 2
DEFAULT_BREAKER_RESET_TIMEOUT_SECONDS = 900.0


def _path_in_scope(path: str, scope_pattern: str) -> bool:
    """Check if a file path falls within a scope pattern.

    Delegates to the coordination layer's proven ``_path_matches_glob`` which
    supports exact paths, directory prefixes, ``/**`` recursive globs, and
    ``PurePosixPath.match()`` for standard glob patterns like ``*.json`` or
    ``**/*.ts``.
    """
    from aragora.nomic.dev_coordination import _path_matches_glob

    clean_path = path.strip().removeprefix("./").rstrip("/")
    clean_scope = scope_pattern.strip().removeprefix("./").rstrip("/")
    if not clean_path or not clean_scope:
        return False
    return _path_matches_glob(clean_path, clean_scope)


class SupervisorRunStatus(str, Enum):
    """Lifecycle state for a supervised swarm run."""

    PLANNED = "planned"
    ACTIVE = "active"
    NEEDS_HUMAN = "needs_human"
    COMPLETED = "completed"


class WorkerOutcome(str, Enum):
    """Structured classification of a worker's terminal state.

    Set on each work-order item as ``worker_outcome`` so operators and
    higher-level orchestrators (campaign, boss loop) can distinguish between
    fundamentally different failure modes without parsing free-text fields.
    """

    COMPLETED = "completed"
    CLEAN_EXIT_NO_EFFECT = "clean_exit_no_effect"
    CRASH = "crash"
    CRASH_WITH_SALVAGE = "crash_with_salvage"
    TIMEOUT_NO_PROGRESS = "timeout_no_progress"
    TIMEOUT_WITH_SALVAGE = "timeout_with_salvage"
    SCOPE_VIOLATION = "scope_violation"
    MERGE_GATE_FAILED = "merge_gate_failed"


@dataclass(slots=True)
class SwarmApprovalPolicy:
    """Explicit human-gating policy for supervised swarm runs."""

    require_merge_approval: bool = True
    require_external_action_approval: bool = True
    protected_patterns: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "require_merge_approval": self.require_merge_approval,
            "require_external_action_approval": self.require_external_action_approval,
            "protected_patterns": list(self.protected_patterns),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> SwarmApprovalPolicy:
        payload = dict(data or {})
        return cls(
            require_merge_approval=bool(payload.get("require_merge_approval", True)),
            require_external_action_approval=bool(
                payload.get("require_external_action_approval", True)
            ),
            protected_patterns=[
                str(item) for item in payload.get("protected_patterns", []) if str(item).strip()
            ],
        )


@dataclass(slots=True)
class SupervisorRun:
    """Top-level artifact for one supervised swarm execution."""

    run_id: str
    goal: str
    target_branch: str
    status: str
    supervisor_agents: dict[str, Any]
    approval_policy: SwarmApprovalPolicy
    spec: SwarmSpec
    work_orders: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "goal": self.goal,
            "target_branch": self.target_branch,
            "status": self.status,
            "supervisor_agents": dict(self.supervisor_agents),
            "approval_policy": self.approval_policy.to_dict(),
            "spec": self.spec.to_dict(),
            "work_orders": [dict(item) for item in self.work_orders],
            "metadata": dict(self.metadata),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> SupervisorRun:
        return cls(
            run_id=str(record.get("run_id", "")),
            goal=str(record.get("goal", "")),
            target_branch=str(record.get("target_branch", "main")),
            status=str(record.get("status", SupervisorRunStatus.PLANNED.value)),
            supervisor_agents=dict(record.get("supervisor_agents") or {}),
            approval_policy=SwarmApprovalPolicy.from_dict(record.get("approval_policy")),
            spec=SwarmSpec.from_dict(dict(record.get("spec") or {})),
            work_orders=[dict(item) for item in record.get("work_orders", [])],
            metadata=dict(record.get("metadata") or {}),
            created_at=str(record.get("created_at", datetime.now(UTC).isoformat())),
            updated_at=str(record.get("updated_at", datetime.now(UTC).isoformat())),
        )


class SwarmSupervisor:
    """Coordinate a bounded Codex/Claude worker pool using existing primitives."""

    def __init__(
        self,
        repo_root: Path | None = None,
        *,
        store: DevCoordinationStore | None = None,
        lifecycle: WorktreeLifecycleService | None = None,
        bridge: NomicPipelineBridge | None = None,
        decomposer: TaskDecomposer | None = None,
        approval_policy: ApprovalPolicy | None = None,
        launcher: WorkerLauncher | None = None,
    ) -> None:
        self.repo_root = (repo_root or Path.cwd()).resolve()
        self.store = store or DevCoordinationStore(repo_root=self.repo_root)
        self.lifecycle = lifecycle or WorktreeLifecycleService(repo_root=self.repo_root)
        self.bridge = bridge or NomicPipelineBridge(repo_path=self.repo_root)
        self.decomposer = decomposer or TaskDecomposer()
        self.approval_policy = approval_policy or ApprovalPolicy()
        self.launcher = launcher or WorkerLauncher()

    def start_run(
        self,
        *,
        spec: SwarmSpec,
        target_branch: str = "main",
        max_concurrency: int = 8,
        managed_dir_pattern: str = ".worktrees/{agent}-auto",
        approval_policy: SwarmApprovalPolicy | None = None,
        refresh_scaling: bool = True,
        default_target_agent: str | None = None,
        default_reviewer_agent: str | None = None,
    ) -> SupervisorRun:
        goal = spec.refined_goal or spec.raw_goal
        policy = approval_policy or SwarmApprovalPolicy()
        policy.require_merge_approval = True
        work_orders = [item.to_dict() for item in self._build_supervised_work_orders(spec)]
        if default_target_agent:
            for item in work_orders:
                item["target_agent"] = default_target_agent
                if not default_reviewer_agent and not str(item.get("reviewer_agent", "")).strip():
                    item["reviewer_agent"] = (
                        "claude" if default_target_agent == "codex" else "codex"
                    )
        if default_reviewer_agent:
            for item in work_orders:
                item["reviewer_agent"] = default_reviewer_agent
        for item in work_orders:
            item.setdefault("status", "queued")
            item.setdefault("lease_id", None)
            item.setdefault("receipt_id", None)
            item.setdefault("review_status", "pending")

        record = self.store.create_supervisor_run(
            goal=goal,
            target_branch=target_branch,
            supervisor_agents={"planner": "codex", "judge": "claude"},
            approval_policy=policy.to_dict(),
            spec=spec.to_dict(),
            work_orders=work_orders,
            status=SupervisorRunStatus.PLANNED.value,
            metadata={
                "max_concurrency": min(max(1, int(max_concurrency)), 8),
                "managed_dir_pattern": managed_dir_pattern,
                WORKER_TYPE_CIRCUIT_BREAKERS_KEY: {},
                WORKER_TYPE_CIRCUIT_BREAKER_POLICY_KEY: {
                    "failure_threshold": DEFAULT_BREAKER_FAILURE_THRESHOLD,
                    "reset_timeout_seconds": DEFAULT_BREAKER_RESET_TIMEOUT_SECONDS,
                },
            },
        )
        run = SupervisorRun.from_record(record)
        if refresh_scaling:
            return self.refresh_run(run.run_id)
        return run

    def refresh_run(self, run_id: str) -> SupervisorRun:
        record = self.store.get_supervisor_run(run_id)
        if record is None:
            raise KeyError(f"Unknown supervisor run: {run_id}")

        max_concurrency = min(max(1, int(record.get("metadata", {}).get("max_concurrency", 8))), 8)
        managed_dir_pattern = str(
            record.get("metadata", {}).get("managed_dir_pattern", ".worktrees/{agent}-auto")
        )
        work_orders = [dict(item) for item in record.get("work_orders", [])]
        active_count = sum(1 for item in work_orders if str(item.get("status", "")) == "leased")

        if active_count < max_concurrency:
            for item in work_orders:
                if active_count >= max_concurrency:
                    break
                if str(item.get("status", "queued")) not in {"queued", "waiting_conflict"}:
                    continue
                try:
                    self._lease_work_order(
                        run_id=run_id,
                        target_branch=str(record.get("target_branch", "main")),
                        work_order=item,
                        managed_dir_pattern=managed_dir_pattern,
                        approval_policy=SwarmApprovalPolicy.from_dict(
                            record.get("approval_policy")
                        ),
                    )
                    active_count += 1
                except LeaseConflictError as exc:
                    released = self._release_orphaned_conflict_leases(exc.conflicts)
                    if released:
                        try:
                            self._lease_work_order(
                                run_id=run_id,
                                target_branch=str(record.get("target_branch", "main")),
                                work_order=item,
                                managed_dir_pattern=managed_dir_pattern,
                                approval_policy=SwarmApprovalPolicy.from_dict(
                                    record.get("approval_policy")
                                ),
                            )
                            active_count += 1
                            continue
                        except LeaseConflictError as retry_exc:
                            exc = retry_exc
                    item["status"] = "waiting_conflict"
                    item["conflicts"] = list(exc.conflicts)
                except RuntimeError as exc:
                    if self._is_resource_constraint_error(exc):
                        item["status"] = "waiting_resource"
                        item["resource_error"] = str(exc)
                    else:
                        item["status"] = "needs_human"
                        item["dispatch_error"] = str(exc)
                    break

        refreshed = self.store.update_supervisor_run(
            run_id,
            status=self._derive_status(work_orders),
            work_orders=work_orders,
            metadata=self._campaign_metadata(
                dict(record.get("metadata") or {}),
                work_orders,
            ),
        )
        return SupervisorRun.from_record(refreshed)

    def status_summary(
        self,
        *,
        run_id: str | None = None,
        limit: int = 20,
        refresh_scaling: bool = False,
    ) -> dict[str, Any]:
        records = (
            [self.store.get_supervisor_run(run_id)]
            if run_id
            else self.store.list_supervisor_runs(limit=limit)
        )
        runs: list[SupervisorRun] = []
        for record in records:
            if not record:
                continue
            current = (
                self.refresh_run(record["run_id"])
                if refresh_scaling
                else SupervisorRun.from_record(record)
            )
            runs.append(current)
        coordination = self.store.status_summary()
        return {
            "runs": [run.to_dict() for run in runs],
            "counts": {
                "runs": len(runs),
                "queued_work_orders": sum(
                    1
                    for run in runs
                    for item in run.work_orders
                    if str(item.get("status", "")) == "queued"
                ),
                "leased_work_orders": sum(
                    1
                    for run in runs
                    for item in run.work_orders
                    if str(item.get("status", "")) == "leased"
                ),
                "completed_work_orders": sum(
                    1
                    for run in runs
                    for item in run.work_orders
                    if str(item.get("status", "")) == "completed"
                ),
            },
            "coordination": coordination,
        }

    def reset_worker_type_circuit_breaker(
        self,
        run_id: str,
        worker_type: str,
    ) -> SupervisorRun:
        """Manually reset the circuit breaker for a worker type on a run."""
        record = self.store.get_supervisor_run(run_id)
        if record is None:
            raise KeyError(f"Unknown supervisor run: {run_id}")

        metadata = dict(record.get("metadata") or {})
        circuit_breakers = self._worker_type_circuit_breakers(metadata)
        normalized_worker_type = str(worker_type).strip().lower()
        if not normalized_worker_type:
            raise ValueError("worker_type must be non-empty")
        self._reset_worker_type_circuit_breaker_entry(
            circuit_breakers,
            normalized_worker_type,
        )

        updated = self.store.update_supervisor_run(
            run_id,
            metadata=self._worker_type_circuit_breaker_metadata(
                metadata,
                circuit_breakers,
            ),
        )
        return SupervisorRun.from_record(updated)

    async def dispatch_workers(self, run_id: str) -> list[WorkerProcess]:
        """Launch worker processes for all leased work orders in a run.

        Call this after start_run() to actually spawn the CLI processes.
        Only launches workers for orders in 'leased' status that have a
        worktree_path assigned.

        Returns:
            List of WorkerProcess objects for launched workers.
        """
        record = self.store.get_supervisor_run(run_id)
        if record is None:
            raise KeyError(f"Unknown supervisor run: {run_id}")

        work_orders = [dict(item) for item in record.get("work_orders", [])]
        metadata = dict(record.get("metadata") or {})
        worker_type_circuit_breaker_policy = self._worker_type_circuit_breaker_policy(metadata)
        worker_type_circuit_breakers = self._worker_type_circuit_breakers(metadata)
        self._expire_worker_type_circuit_breakers(worker_type_circuit_breakers)
        launched: list[WorkerProcess] = []

        for item in work_orders:
            if str(item.get("status", "")) != "leased":
                continue
            target_agent = str(item.get("target_agent", "codex")).strip().lower() or "codex"
            if self._worker_type_circuit_breaker_is_open(
                worker_type_circuit_breakers,
                target_agent,
            ):
                breaker = dict(worker_type_circuit_breakers.get(target_agent) or {})
                detail = self._worker_type_circuit_breaker_detail(target_agent, breaker)
                fallback_requeued = self._requeue_with_fallback(
                    item,
                    reason="worker_type_blocked",
                    detail=detail,
                    worker_type_circuit_breakers=worker_type_circuit_breakers,
                )
                if not fallback_requeued:
                    self._mark_worker_type_blocked(
                        item,
                        worker_type=target_agent,
                        detail=detail,
                    )
                continue
            worktree_path = str(item.get("worktree_path", "")).strip()
            branch = str(item.get("branch", "main")).strip()
            if not worktree_path:
                continue

            try:
                worker = await self.launcher.launch(
                    item,
                    worktree_path=worktree_path,
                    branch=branch,
                )
                self._record_worker_type_success(
                    worker_type_circuit_breakers,
                    target_agent,
                )
                dispatch_time = datetime.now(UTC).isoformat()
                item["status"] = "dispatched"
                item["pid"] = worker.pid
                item["initial_head"] = worker.initial_head
                item["dispatched_at"] = dispatch_time
                item["last_observed_at"] = dispatch_time
                item["last_progress_at"] = dispatch_time
                item["progress_fingerprint"] = {
                    "head_sha": worker.initial_head,
                    "changed_paths": [],
                    "diff_lines": 0,
                }
                launched.append(worker)
            except (FileNotFoundError, RuntimeError, OSError) as exc:
                self._record_worker_type_failure(
                    worker_type_circuit_breakers,
                    target_agent,
                    reason=self._dispatch_failure_reason(exc),
                    detail=str(exc),
                    policy=worker_type_circuit_breaker_policy,
                )
                fallback_requeued = self._requeue_after_dispatch_error(
                    item,
                    exc,
                    worker_type_circuit_breakers=worker_type_circuit_breakers,
                )
                if not fallback_requeued:
                    item["status"] = "dispatch_failed"
                    item["dispatch_error"] = str(exc)
                import logging

                logging.getLogger(__name__).warning(
                    "Failed to dispatch %s: %s",
                    item.get("work_order_id"),
                    exc,
                )

        self.store.update_supervisor_run(
            run_id,
            status=self._derive_status(work_orders),
            work_orders=work_orders,
            metadata=self._campaign_metadata(
                self._worker_type_circuit_breaker_metadata(
                    metadata,
                    worker_type_circuit_breakers,
                ),
                work_orders,
            ),
        )
        return launched

    async def collect_results(
        self,
        run_id: str,
        *,
        timeout: float | None = None,
    ) -> list[WorkerProcess]:
        """Wait for all dispatched workers to complete and update the run.

        Returns:
            List of completed WorkerProcess objects.
        """
        record = self.store.get_supervisor_run(run_id)
        if record is None:
            raise KeyError(f"Unknown supervisor run: {run_id}")

        work_orders = [dict(item) for item in record.get("work_orders", [])]
        metadata = dict(record.get("metadata") or {})
        worker_type_circuit_breaker_policy = self._worker_type_circuit_breaker_policy(metadata)
        worker_type_circuit_breakers = self._worker_type_circuit_breakers(metadata)
        self._expire_worker_type_circuit_breakers(worker_type_circuit_breakers)
        completed: list[WorkerProcess] = []

        for item in work_orders:
            if str(item.get("status", "")) != "dispatched":
                continue
            work_order_id = str(item.get("work_order_id", ""))
            worker = self.launcher.get_worker(work_order_id)
            if worker is None:
                continue

            result = await self.launcher.wait(work_order_id, timeout=timeout)
            self._apply_worker_result(
                item,
                result,
                worker_type_circuit_breakers=worker_type_circuit_breakers,
                worker_type_circuit_breaker_policy=worker_type_circuit_breaker_policy,
            )
            completed.append(result)

        self.store.update_supervisor_run(
            run_id,
            status=self._derive_status(work_orders),
            work_orders=work_orders,
            metadata=self._campaign_metadata(
                self._worker_type_circuit_breaker_metadata(
                    metadata,
                    worker_type_circuit_breakers,
                ),
                work_orders,
            ),
        )
        return completed

    async def collect_finished_results(self, run_id: str) -> list[WorkerProcess]:
        """Collect only workers that have already finished.

        Tries in-memory process collection first (same-process workers).
        Falls back to detached PID-based collection for workers spawned
        by a previous process (e.g. --dispatch-only mode).
        """
        record = self.store.get_supervisor_run(run_id)
        if record is None:
            raise KeyError(f"Unknown supervisor run: {run_id}")

        work_orders = [dict(item) for item in record.get("work_orders", [])]
        metadata = dict(record.get("metadata") or {})
        worker_type_circuit_breaker_policy = self._worker_type_circuit_breaker_policy(metadata)
        worker_type_circuit_breakers = self._worker_type_circuit_breakers(metadata)
        self._expire_worker_type_circuit_breakers(worker_type_circuit_breakers)
        dispatched_ids = [
            str(item.get("work_order_id", "")).strip()
            for item in work_orders
            if str(item.get("status", "")) == "dispatched"
        ]

        # Try in-memory collection first (same process that launched workers)
        finished = await self.launcher.collect_finished(work_order_ids=dispatched_ids)
        changed = False

        # Fall back to detached collection for workers not in memory
        # (parent process restarted, or --dispatch-only mode)
        finished_ids = {w.work_order_id for w in finished}
        for item in work_orders:
            woid = str(item.get("work_order_id", "")).strip()
            if str(item.get("status", "")) != "dispatched":
                continue
            if woid in finished_ids:
                continue
            worktree_path = str(item.get("worktree_path", "")).strip()
            if not worktree_path:
                continue
            result = await WorkerLauncher.collect_detached_result(
                work_order_id=woid,
                agent=str(item.get("target_agent", "codex")),
                worktree_path=worktree_path,
                branch=str(item.get("branch", "main")),
                pid=item.get("pid"),
                initial_head=str(item.get("initial_head", "")),
                auto_commit=self.launcher.config.auto_commit,
            )
            if result is not None:
                finished.append(result)
                finished_ids.add(woid)
                continue

            progress = await self.launcher.snapshot_progress(item)
            observed_at = datetime.now(UTC).isoformat()
            item["last_observed_at"] = observed_at
            progress_fingerprint = self._progress_fingerprint(progress)
            if progress_fingerprint != self._progress_fingerprint(item.get("progress_fingerprint")):
                item["progress_fingerprint"] = progress_fingerprint
                item["last_progress_at"] = observed_at
                if progress_fingerprint["head_sha"]:
                    item["head_sha"] = progress_fingerprint["head_sha"]
                progress_paths = self._strip_session_artifacts(
                    list(progress_fingerprint["changed_paths"])
                )
                item["changed_paths"] = progress_paths
                item["diff_lines"] = int(progress_fingerprint["diff_lines"])
                changed = True

                # Fail closed: check file-scope constraints on every progress snapshot
                scope_violations = self._check_file_scope_violations(item, progress_paths)
                if scope_violations:
                    self._mark_scope_violation(item, scope_violations)
                    item["worker_outcome"] = WorkerOutcome.SCOPE_VIOLATION.value
                    self._release_terminal_lease(item)
                    await self._kill_worker(item)
                    changed = True

                continue

            if not bool(progress.get("pid_alive")):
                # Check scope on exit too — worker may have edited wrong files then died
                exit_violations = self._check_file_scope_violations(
                    item, list(item.get("changed_paths", []))
                )
                if exit_violations:
                    self._mark_scope_violation(item, exit_violations)
                    item["worker_outcome"] = WorkerOutcome.SCOPE_VIOLATION.value
                else:
                    # Attempt detached result collection — worker may have
                    # committed successfully before dying without an exit marker.
                    detached_result: WorkerProcess | None = None
                    try:
                        detached_result = await WorkerLauncher.collect_detached_result(
                            work_order_id=woid,
                            agent=str(item.get("target_agent", "codex")),
                            worktree_path=worktree_path,
                            branch=str(item.get("branch", "main")),
                            pid=item.get("pid"),
                            initial_head=str(item.get("initial_head", "")),
                            auto_commit=self.launcher.config.auto_commit,
                            expected_tests=[
                                str(test).strip()
                                for test in item.get("expected_tests", [])
                                if str(test).strip()
                            ],
                        )
                    except Exception:
                        logger.debug(
                            "Detached result collection failed for %s", woid, exc_info=True
                        )

                    if detached_result is not None and detached_result.commit_shas:
                        finished.append(detached_result)
                        finished_ids.add(woid)
                        item["worker_outcome"] = WorkerOutcome.CRASH_WITH_SALVAGE.value
                        self._release_terminal_lease(item)
                        changed = True
                        continue

                    self._mark_needs_human(
                        item,
                        "worker process exited without receipt or exit marker",
                    )
                    item["worker_outcome"] = WorkerOutcome.CRASH.value
                self._release_terminal_lease(item)
                changed = True
                continue

            if self._exceeded_no_progress_timeout(item):
                reason = (
                    "worker exceeded no-progress timeout "
                    f"({int(self._no_progress_timeout_seconds())}s)"
                )
                # Kill the worker before collecting results so the process
                # releases file handles and the worktree is stable for git.
                await self._kill_worker(item)

                # Attempt detached result collection so the try/finally in
                # collect_detached_result cleans session artifacts (#896) and
                # any salvageable deliverable is surfaced (#899).
                worktree_path = str(item.get("worktree_path", "")).strip()
                timeout_result: WorkerProcess | None = None
                if worktree_path:
                    try:
                        timeout_result = await WorkerLauncher.collect_detached_result(
                            work_order_id=woid,
                            agent=str(item.get("target_agent", "codex")),
                            worktree_path=worktree_path,
                            branch=str(item.get("branch", "main")),
                            pid=item.get("pid"),
                            initial_head=str(item.get("initial_head", "")),
                            auto_commit=self.launcher.config.auto_commit,
                            expected_tests=[
                                str(test).strip()
                                for test in item.get("expected_tests", [])
                                if str(test).strip()
                            ],
                        )
                    except Exception:
                        logger.debug("Timeout result collection failed for %s", woid, exc_info=True)

                if timeout_result is not None and timeout_result.commit_shas:
                    # Worker produced a concrete deliverable before timing out.
                    # Surface it through the normal result path.
                    finished.append(timeout_result)
                    finished_ids.add(woid)
                    item["worker_outcome"] = WorkerOutcome.TIMEOUT_WITH_SALVAGE.value
                else:
                    # No deliverable — check scope violations and mark blocked.
                    collected_paths = self._strip_session_artifacts(
                        list(
                            (timeout_result.changed_paths if timeout_result else None)
                            or item.get("changed_paths", [])
                        )
                    )
                    timeout_violations = self._check_file_scope_violations(item, collected_paths)
                    if timeout_violations:
                        self._mark_scope_violation(item, timeout_violations, extra_reason=reason)
                        item["worker_outcome"] = WorkerOutcome.SCOPE_VIOLATION.value
                    else:
                        self._mark_needs_human(item, reason)
                        item["worker_outcome"] = WorkerOutcome.TIMEOUT_NO_PROGRESS.value
                    self._release_terminal_lease(item)
                changed = True

        if not finished and not changed:
            return []

        finished_by_id = {worker.work_order_id: worker for worker in finished}
        for item in work_orders:
            worker = finished_by_id.get(str(item.get("work_order_id", "")).strip())
            if worker is None:
                continue
            self._apply_worker_result(
                item,
                worker,
                worker_type_circuit_breakers=worker_type_circuit_breakers,
                worker_type_circuit_breaker_policy=worker_type_circuit_breaker_policy,
            )

        self.store.update_supervisor_run(
            run_id,
            status=self._derive_status(work_orders),
            work_orders=work_orders,
            metadata=self._campaign_metadata(
                self._worker_type_circuit_breaker_metadata(
                    metadata,
                    worker_type_circuit_breakers,
                ),
                work_orders,
            ),
        )
        return finished

    def _build_supervised_work_orders(self, spec: SwarmSpec) -> list[BoundedWorkOrder]:
        explicit = self._explicit_work_orders_from_spec(spec)
        if explicit:
            return explicit

        goal = spec.refined_goal or spec.raw_goal
        spec_hints = list(spec.file_scope_hints) if spec.file_scope_hints else []
        decomposition = self.decomposer.analyze(
            self._task_prompt(spec),
            file_scope_hints=spec_hints or None,
        )
        subtasks = list(decomposition.subtasks)
        if not subtasks:
            subtasks = [
                SubTask(
                    id=f"work-{uuid.uuid4().hex[:8]}",
                    title=goal[:80] or "Swarm task",
                    description=goal,
                    file_scope=list(spec.file_scope_hints),
                    success_criteria={
                        "tests": self._tests_from_acceptance(spec.acceptance_criteria),
                        "acceptance_criteria": list(spec.acceptance_criteria),
                    },
                )
            ]
        work_orders = self.bridge.build_work_orders(subtasks)
        for item in work_orders:
            # Override file_scope from spec hints when the decomposer left it
            # empty OR produced scopes with zero overlap with the hints.
            if spec_hints:
                if not item.file_scope:
                    item.file_scope = list(spec_hints)
                    logger.info(
                        "Backfilled empty file_scope on work order %s from spec hints: %s",
                        item.work_order_id,
                        spec_hints,
                    )
                elif not self._scope_overlaps_hints(item.file_scope, spec_hints):
                    logger.warning(
                        "Decomposer file_scope %s on work order %s has no overlap "
                        "with spec hints %s — overriding with spec hints",
                        item.file_scope,
                        item.work_order_id,
                        spec_hints,
                    )
                    item.file_scope = list(spec_hints)
            item.expected_tests = self._default_tests(item, spec)
            item.risk_level = self._risk_level_for_scope(item.file_scope)
            item.approval_required = True
            item.metadata = {
                **dict(item.metadata),
                "acceptance_criteria": list(spec.acceptance_criteria),
                "constraints": list(spec.constraints),
            }
        return work_orders

    def _explicit_work_orders_from_spec(self, spec: SwarmSpec) -> list[BoundedWorkOrder]:
        if not spec.work_orders:
            return []

        work_orders: list[BoundedWorkOrder] = []
        pipeline_id_by_work_order: dict[str, str] = {}
        normalized_payloads = [
            dict(payload) for payload in spec.work_orders if isinstance(payload, dict)
        ]
        explicit_ids: list[str] = []

        for index, payload in enumerate(normalized_payloads, start=1):
            work_order_id = str(payload.get("work_order_id", "")).strip() or f"work-{index}"
            explicit_ids.append(work_order_id)
            pipeline_id_by_work_order[work_order_id] = (
                str(payload.get("pipeline_task_id", "")).strip() or f"task-{index}"
            )

        for index, payload in enumerate(normalized_payloads, start=1):
            work_order_id = explicit_ids[index - 1]
            pipeline_task_id = pipeline_id_by_work_order[work_order_id]
            target_agent = str(payload.get("target_agent", "")).strip()
            reviewer_agent = str(payload.get("reviewer_agent", "")).strip()
            if not target_agent:
                target_agent = "codex" if (index - 1) % 2 == 0 else "claude"
            if not reviewer_agent:
                reviewer_agent = "claude" if target_agent == "codex" else "codex"

            success_criteria = dict(payload.get("success_criteria") or {})
            expected_tests = [
                str(item).strip() for item in payload.get("expected_tests", []) if str(item).strip()
            ]
            if expected_tests and "tests" not in success_criteria:
                success_criteria["tests"] = list(expected_tests)

            estimated_complexity = (
                str(payload.get("estimated_complexity", "medium")).strip() or "medium"
            )
            risk_level = str(payload.get("risk_level", "")).strip() or self._risk_level_for_scope(
                [str(item) for item in payload.get("file_scope", []) if str(item).strip()]
            )

            dependency_ids = [
                str(dep).strip() for dep in payload.get("dependency_ids", []) if str(dep).strip()
            ]
            if not dependency_ids:
                dependency_ids = [
                    pipeline_id_by_work_order.get(str(dep).strip(), str(dep).strip())
                    for dep in payload.get("dependencies", [])
                    if str(dep).strip()
                ]

            work_orders.append(
                BoundedWorkOrder(
                    work_order_id=work_order_id,
                    pipeline_task_id=pipeline_task_id,
                    title=str(payload.get("title", "")).strip() or work_order_id,
                    description=str(payload.get("description", "")).strip()
                    or str(payload.get("title", "")).strip()
                    or spec.refined_goal
                    or spec.raw_goal,
                    file_scope=[
                        str(item).strip()
                        for item in payload.get("file_scope", [])
                        if str(item).strip()
                    ],
                    dependency_ids=dependency_ids,
                    success_criteria=success_criteria,
                    expected_tests=expected_tests,
                    estimated_complexity=estimated_complexity,
                    risk_level=risk_level,
                    target_agent=target_agent,
                    reviewer_agent=reviewer_agent,
                    approval_required=bool(payload.get("approval_required", False)),
                    metadata={
                        **dict(payload.get("metadata") or {}),
                        "source": "explicit_spec_work_order",
                    },
                )
            )

        for item in work_orders:
            item.expected_tests = self._default_tests(item, spec)
            item.risk_level = str(item.risk_level).strip() or self._risk_level_for_scope(
                item.file_scope
            )
            item.approval_required = True
            item.metadata = {
                **dict(item.metadata),
                "acceptance_criteria": list(spec.acceptance_criteria),
                "constraints": list(spec.constraints),
            }

        return work_orders

    def _lease_work_order(
        self,
        *,
        run_id: str,
        target_branch: str,
        work_order: dict[str, Any],
        managed_dir_pattern: str,
        approval_policy: SwarmApprovalPolicy,
    ) -> None:
        target_agent = str(work_order.get("target_agent", "codex")).strip() or "codex"
        managed_dir = self._managed_dir_for_agent(managed_dir_pattern, target_agent)
        wo_id = str(work_order.get("work_order_id", "task"))
        session_key = f"swarm-{run_id[:8]}-{wo_id}"
        session = self.lifecycle.ensure_managed_worktree(
            managed_dir=managed_dir,
            base_branch=target_branch,
            agent=target_agent,
            session_id=session_key,
            reconcile=True,
            strategy="ff-only",
        )
        file_scope = [str(item) for item in work_order.get("file_scope", []) if str(item).strip()]
        claimed_paths = [item for item in file_scope if not self._looks_like_glob(item)]
        allowed_globs = [item for item in file_scope if self._looks_like_glob(item)]
        if not allowed_globs and not claimed_paths and file_scope:
            claimed_paths = list(file_scope)

        lease = self.store.claim_lease(
            task_id=str(work_order.get("work_order_id", "")),
            title=str(work_order.get("title", "") or work_order.get("work_order_id", "task")),
            owner_agent=target_agent,
            owner_session_id=session.session_id,
            branch=session.branch,
            worktree_path=str(session.path),
            allowed_globs=allowed_globs,
            claimed_paths=claimed_paths,
            expected_tests=[str(item) for item in work_order.get("expected_tests", [])],
            metadata={
                "supervisor_run_id": run_id,
                "work_order_id": str(work_order.get("work_order_id", "")),
                "reviewer_agent": str(work_order.get("reviewer_agent", "")),
                "risk_level": str(work_order.get("risk_level", "review")),
                "approval_required": True,
            },
        )
        work_order.update(
            {
                "status": "leased",
                "lease_id": lease.lease_id,
                "owner_session_id": session.session_id,
                "branch": session.branch,
                "worktree_path": str(session.path),
                "target_agent": target_agent,
                "approval_required": True,
            }
        )

    @staticmethod
    def _strip_session_artifacts(paths: list[str]) -> list[str]:
        """Remove harness session metadata from a list of changed paths.

        Session artifacts like ``.codex_session_meta.json`` are infrastructure
        metadata created by the harness, not user deliverables.  Stripping them
        prevents workers from claiming credit for non-work output.
        """
        return [p for p in paths if Path(p).name not in SESSION_ARTIFACTS]

    def _campaign_metadata(
        self,
        metadata: dict[str, Any],
        work_orders: list[dict[str, Any]],
    ) -> dict[str, Any]:
        payload = dict(metadata)
        outcome, blockers = self._campaign_outcome_for_work_orders(work_orders)
        if not outcome:
            payload.pop(CAMPAIGN_OUTCOME_METADATA_KEY, None)
            payload.pop(CAMPAIGN_REQUEUE_ELIGIBLE_METADATA_KEY, None)
            payload.pop(CAMPAIGN_BLOCKERS_METADATA_KEY, None)
            return payload

        payload[CAMPAIGN_OUTCOME_METADATA_KEY] = outcome
        payload[CAMPAIGN_REQUEUE_ELIGIBLE_METADATA_KEY] = self._campaign_requeue_eligible(outcome)
        if blockers:
            payload[CAMPAIGN_BLOCKERS_METADATA_KEY] = blockers[:10]
        else:
            payload.pop(CAMPAIGN_BLOCKERS_METADATA_KEY, None)
        return payload

    @classmethod
    def _campaign_outcome_for_work_orders(
        cls,
        work_orders: list[dict[str, Any]],
    ) -> tuple[str | None, list[str]]:
        statuses: set[str] = set()
        worker_outcomes: set[str] = set()
        blockers: list[str] = []
        has_deliverable = False

        for item in work_orders:
            if not isinstance(item, dict):
                continue
            status = str(item.get("status", "")).strip().lower()
            if status:
                statuses.add(status)

            worker_outcome = str(item.get("worker_outcome", "")).strip().lower()
            if worker_outcome:
                worker_outcomes.add(worker_outcome)

            deliverable_type = cls._work_order_deliverable_type(item)
            if deliverable_type == "pr_adopted":
                return deliverable_type, cls._campaign_blockers_from_work_orders(work_orders)
            if deliverable_type == "deliverable_created":
                has_deliverable = True

            for value in item.get("blockers", []):
                text = str(value).strip()
                if text and text not in blockers:
                    blockers.append(text)
            dispatch_error = str(item.get("dispatch_error", "")).strip()
            if dispatch_error and dispatch_error not in blockers:
                blockers.append(dispatch_error)

        forward_progress_statuses = {"queued", "leased", "dispatched"}
        stalled_wait_statuses = {"waiting_conflict", "waiting_resource"}
        stalled_dead_end = bool(statuses & stalled_wait_statuses) and not (
            statuses & forward_progress_statuses
        )
        stalled_no_progress = WorkerOutcome.TIMEOUT_NO_PROGRESS.value in worker_outcomes

        if has_deliverable:
            return "deliverable_created", blockers
        if "scope_violation" in worker_outcomes or "scope_violation" in statuses:
            return "blocked", blockers
        if any(outcome.startswith("crash") for outcome in worker_outcomes):
            return "crash", blockers
        if stalled_no_progress or stalled_dead_end:
            return "stalled", blockers
        if (
            any(outcome.startswith("timeout") for outcome in worker_outcomes)
            or "timed_out" in statuses
        ):
            return "timeout", blockers
        if "failed" in statuses:
            return "crash", blockers
        if "clean_exit_no_effect" in worker_outcomes:
            return "clean_exit_no_deliverable", blockers
        if "needs_human" in statuses:
            return "needs_human", blockers
        if statuses and statuses <= {"completed"}:
            return "clean_exit_no_deliverable", blockers
        return None, blockers

    @staticmethod
    def _campaign_blockers_from_work_orders(work_orders: list[dict[str, Any]]) -> list[str]:
        blockers: list[str] = []
        for item in work_orders:
            if not isinstance(item, dict):
                continue
            for value in item.get("blockers", []):
                text = str(value).strip()
                if text and text not in blockers:
                    blockers.append(text)
            dispatch_error = str(item.get("dispatch_error", "")).strip()
            if dispatch_error and dispatch_error not in blockers:
                blockers.append(dispatch_error)
        return blockers

    @staticmethod
    def _campaign_requeue_eligible(outcome: str) -> bool:
        return outcome in {
            "clean_exit_no_deliverable",
            "timeout",
            "crash",
        }

    @staticmethod
    def _work_order_deliverable_type(item: dict[str, Any]) -> str | None:
        if str(item.get("adopted_pr", "")).strip():
            return "pr_adopted"
        if str(item.get("pr_url", "")).strip():
            return "deliverable_created"
        branch = str(item.get("branch", "")).strip()
        commit_shas = [str(sha).strip() for sha in item.get("commit_shas", []) if str(sha).strip()]
        if branch and commit_shas:
            return "deliverable_created"
        return None

    def _apply_worker_result(
        self,
        item: dict[str, Any],
        result: WorkerProcess,
        *,
        worker_type_circuit_breakers: dict[str, dict[str, Any]] | None = None,
        worker_type_circuit_breaker_policy: dict[str, Any] | None = None,
    ) -> None:
        # Strip session artifacts before any qualification logic runs
        clean_paths = self._strip_session_artifacts(list(result.changed_paths))
        item["completed_at"] = result.completed_at
        item["diff_lines"] = result.diff.count("\n")
        item["changed_paths"] = clean_paths
        item["tests_run"] = list(result.tests_run)
        item["verification_results"] = self._verification_results_from_result(result)
        item["commit_shas"] = list(result.commit_shas)
        item["head_sha"] = result.head_sha
        item.pop("pid", None)

        # Preserve worker_outcome if already set by detached/timeout collection
        # paths — those have more specific context (e.g. timeout_with_salvage).
        _pre_outcome = str(item.get("worker_outcome", "")).strip()

        # Fail closed: check file-scope before accepting any result as successful
        scope_violations = self._check_file_scope_violations(item, clean_paths)
        if scope_violations:
            self._mark_scope_violation(item, scope_violations)
            if not _pre_outcome:
                item["worker_outcome"] = WorkerOutcome.SCOPE_VIOLATION.value
            lease_id = str(item.get("lease_id", "")).strip()
            if lease_id:
                self.store.release_lease(lease_id, status=LeaseStatus.RELEASED)
            item["exit_code"] = result.exit_code
            return

        lease_id = str(item.get("lease_id", "")).strip()
        if result.exit_code == 0:
            # Fail closed: if there are no real deliverables but the worker
            # produced commits or had pre-strip changed paths, reject.  This
            # covers both direct workers (result.changed_paths non-empty before
            # strip) and detached workers (changed_paths already stripped by
            # _collect_changed_paths, but commit_shas populated from auto-commit).
            if not clean_paths and (result.changed_paths or result.commit_shas):
                self._mark_needs_human(
                    item,
                    "worker produced only session artifacts, no real deliverables",
                )
                if not _pre_outcome:
                    item["worker_outcome"] = WorkerOutcome.CLEAN_EXIT_NO_EFFECT.value
                self._release_terminal_lease(item)
                item["exit_code"] = result.exit_code
                return

            # Clean exit with zero changes of any kind — fail closed
            if not clean_paths and not result.commit_shas:
                if not _pre_outcome:
                    item["worker_outcome"] = WorkerOutcome.CLEAN_EXIT_NO_EFFECT.value
                logger.warning(
                    "Worker %s exited 0 with no commits and no changed paths — "
                    "clean_exit_no_effect (branch=%s, initial_head=%s, head_sha=%s)",
                    item.get("work_order_id"),
                    item.get("branch"),
                    result.initial_head,
                    result.head_sha,
                )
                self._mark_needs_human(
                    item,
                    "worker exited 0 with no commits and no changed paths",
                )
                self._release_terminal_lease(item)
                item["exit_code"] = result.exit_code
                return
            elif not _pre_outcome:
                item["worker_outcome"] = WorkerOutcome.COMPLETED.value

            merge_gate = self._merge_gate_state(item)
            item["merge_gate"] = merge_gate
            if merge_gate.get("verification_missing_reason"):
                item["verification_missing_reason"] = merge_gate["verification_missing_reason"]
            if not bool(merge_gate.get("checks_passed")):
                self._mark_needs_human(item, self._merge_gate_failure_reason(merge_gate))
                item["review_status"] = "changes_requested"
                item["receipt_id"] = None
                if not _pre_outcome:
                    item["worker_outcome"] = WorkerOutcome.MERGE_GATE_FAILED.value
                self._release_terminal_lease(item)
                item["exit_code"] = result.exit_code
                return

            receipt_id = str(item.get("receipt_id", "")).strip()
            if lease_id and not receipt_id:
                try:
                    receipt = self.store.record_completion(
                        lease_id=lease_id,
                        owner_agent=str(item.get("target_agent", result.agent)),
                        owner_session_id=str(item.get("owner_session_id", result.session_id)),
                        branch=str(item.get("branch", result.branch)),
                        worktree_path=str(item.get("worktree_path", result.worktree_path)),
                        commit_shas=list(result.commit_shas),
                        changed_paths=clean_paths,
                        tests_run=list(result.tests_run),
                        assumptions=[],
                        blockers=[],
                        confidence=self._completion_confidence(item, result),
                    )
                except FileScopeViolationError as exc:
                    self._mark_needs_human(
                        item,
                        "worker completion violated file-scope ownership; narrow or split the lane",
                    )
                    item["review_status"] = "changes_requested"
                    item["receipt_id"] = None
                    item["scope_violation"] = {
                        "violations": list(exc.violations),
                        "changed_paths": clean_paths,
                    }
                    self._release_terminal_lease(item)
                    item["exit_code"] = result.exit_code
                    return
                item["receipt_id"] = receipt.receipt_id
                item["confidence"] = receipt.confidence
            if worker_type_circuit_breakers is not None:
                self._record_worker_type_success(
                    worker_type_circuit_breakers,
                    str(item.get("target_agent", result.agent)),
                )
            item["status"] = "completed"
            item["review_status"] = "pending_heterogeneous_review"
            return

        # Non-zero exit: classify as crash (with or without salvage)
        if not _pre_outcome:
            if result.commit_shas and clean_paths:
                item["worker_outcome"] = WorkerOutcome.CRASH_WITH_SALVAGE.value
            else:
                item["worker_outcome"] = WorkerOutcome.CRASH.value

        capacity_failure_detail = self._capacity_failure_detail(result)
        if (
            capacity_failure_detail
            and worker_type_circuit_breakers is not None
            and worker_type_circuit_breaker_policy is not None
        ):
            self._record_worker_type_failure(
                worker_type_circuit_breakers,
                str(item.get("target_agent", result.agent)),
                reason="agent_capacity",
                detail=capacity_failure_detail,
                open_immediately=True,
                policy=worker_type_circuit_breaker_policy,
            )

        if self._requeue_after_worker_failure(
            item,
            result,
            worker_type_circuit_breakers=worker_type_circuit_breakers,
        ):
            return

        if lease_id:
            self.store.release_lease(lease_id, status=LeaseStatus.RELEASED)
        item["status"] = "timed_out" if result.exit_code == -1 else "failed"
        item["exit_code"] = result.exit_code
        if result.stderr.strip():
            item["blockers"] = [result.stderr.strip()]

    def _release_terminal_lease(self, item: dict[str, Any]) -> None:
        lease_id = str(item.get("lease_id", "")).strip()
        if not lease_id:
            return
        try:
            self.store.release_lease(lease_id, status=LeaseStatus.RELEASED)
        except KeyError:
            return

    def _requeue_after_dispatch_error(
        self,
        item: dict[str, Any],
        exc: Exception,
        *,
        worker_type_circuit_breakers: dict[str, dict[str, Any]] | None = None,
    ) -> bool:
        message = str(exc).strip()
        lowered = message.lower()
        if "cli not found" not in lowered and "not found" not in lowered:
            return False
        return self._requeue_with_fallback(
            item,
            reason="agent_unavailable",
            detail=message,
            worker_type_circuit_breakers=worker_type_circuit_breakers,
        )

    def _requeue_after_worker_failure(
        self,
        item: dict[str, Any],
        result: WorkerProcess,
        *,
        worker_type_circuit_breakers: dict[str, dict[str, Any]] | None = None,
    ) -> bool:
        capacity_failure_detail = self._capacity_failure_detail(result)
        if not capacity_failure_detail:
            return False
        return self._requeue_with_fallback(
            item,
            reason="agent_capacity",
            detail=capacity_failure_detail,
            worker_type_circuit_breakers=worker_type_circuit_breakers,
        )

    def _requeue_with_fallback(
        self,
        item: dict[str, Any],
        *,
        reason: str,
        detail: str,
        worker_type_circuit_breakers: dict[str, dict[str, Any]] | None = None,
    ) -> bool:
        current_agent = str(item.get("target_agent", "")).strip().lower()
        fallback_agent = self._alternate_agent(current_agent)
        if not fallback_agent:
            return False
        if worker_type_circuit_breakers is not None and self._worker_type_circuit_breaker_is_open(
            worker_type_circuit_breakers,
            fallback_agent,
        ):
            return False

        metadata = dict(item.get("metadata") or {})
        attempted_agents = [
            str(agent).strip().lower()
            for agent in metadata.get("attempted_agents", [])
            if str(agent).strip()
        ]
        if current_agent and current_agent not in attempted_agents:
            attempted_agents.append(current_agent)
        if fallback_agent in attempted_agents:
            return False

        fallback_history = list(metadata.get("fallback_history", []))
        fallback_history.append(
            {
                "from_agent": current_agent,
                "to_agent": fallback_agent,
                "reason": reason,
                "detail": detail[:500],
                "at": datetime.now(UTC).isoformat(),
            }
        )
        metadata.update(
            {
                "requested_target_agent": metadata.get("requested_target_agent", current_agent),
                "requested_reviewer_agent": metadata.get(
                    "requested_reviewer_agent",
                    str(item.get("reviewer_agent", "")).strip().lower(),
                ),
                "attempted_agents": attempted_agents,
                "fallback_history": fallback_history,
                "last_failure_reason": reason,
                "last_failure_detail": detail[:1000],
                "reuse_existing_worktree": True,
            }
        )

        item.update(
            {
                "status": "leased",
                "target_agent": fallback_agent,
                "reviewer_agent": self._alternate_agent(fallback_agent)
                or str(item.get("reviewer_agent", "")),
                "metadata": metadata,
                "review_status": "pending",
                "receipt_id": None,
                "dispatch_error": None,
                "exit_code": None,
                "completed_at": None,
            }
        )
        item.pop("pid", None)
        item.pop("blockers", None)
        item.pop("dispatched_at", None)
        item.pop("last_observed_at", None)
        item.pop("last_progress_at", None)
        item.pop("progress_fingerprint", None)
        return True

    @staticmethod
    def _dispatch_failure_reason(exc: Exception) -> str:
        message = str(exc).strip().lower()
        if "cli not found" in message or "not found" in message:
            return "agent_unavailable"
        return "agent_launch_failed"

    @staticmethod
    def _capacity_failure_detail(result: WorkerProcess) -> str:
        combined = "\n".join(part for part in (result.stdout, result.stderr) if part).strip()
        lowered = combined.lower()
        capacity_patterns = (
            "credit balance is too low",
            "insufficient credit",
            "insufficient balance",
            "out of credits",
            "quota exceeded",
            "usage limit reached",
            "rate limit exceeded",
            "billing",
            "payment required",
        )
        if any(pattern in lowered for pattern in capacity_patterns):
            return combined or f"{result.agent} worker failed"
        return ""

    def _worker_type_circuit_breaker_policy(self, metadata: dict[str, Any]) -> dict[str, Any]:
        payload = dict(metadata.get(WORKER_TYPE_CIRCUIT_BREAKER_POLICY_KEY) or {})
        try:
            failure_threshold = max(
                1,
                int(payload.get("failure_threshold", DEFAULT_BREAKER_FAILURE_THRESHOLD)),
            )
        except (TypeError, ValueError):
            failure_threshold = DEFAULT_BREAKER_FAILURE_THRESHOLD
        try:
            reset_timeout_seconds = max(
                1.0,
                float(
                    payload.get(
                        "reset_timeout_seconds",
                        DEFAULT_BREAKER_RESET_TIMEOUT_SECONDS,
                    )
                ),
            )
        except (TypeError, ValueError):
            reset_timeout_seconds = DEFAULT_BREAKER_RESET_TIMEOUT_SECONDS
        return {
            "failure_threshold": failure_threshold,
            "reset_timeout_seconds": reset_timeout_seconds,
        }

    def _worker_type_circuit_breakers(
        self,
        metadata: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        policy = self._worker_type_circuit_breaker_policy(metadata)
        raw_breakers = dict(metadata.get(WORKER_TYPE_CIRCUIT_BREAKERS_KEY) or {})
        normalized: dict[str, dict[str, Any]] = {}

        for raw_worker_type, raw_entry in raw_breakers.items():
            worker_type = str(raw_worker_type).strip().lower()
            if not worker_type:
                continue
            entry = self._default_worker_type_circuit_breaker(policy)
            payload = dict(raw_entry or {})
            entry["status"] = (
                str(payload.get("status", entry["status"])).strip().lower() or "closed"
            )
            if entry["status"] not in {"open", "closed"}:
                entry["status"] = "closed"
            try:
                entry["failure_count"] = max(0, int(payload.get("failure_count", 0) or 0))
            except (TypeError, ValueError):
                entry["failure_count"] = 0
            try:
                entry["trip_count"] = max(0, int(payload.get("trip_count", 0) or 0))
            except (TypeError, ValueError):
                entry["trip_count"] = 0
            entry["last_failure_reason"] = str(payload.get("last_failure_reason", "")).strip()
            entry["last_failure_detail"] = str(payload.get("last_failure_detail", "")).strip()[
                :1000
            ]
            entry["last_failure_at"] = self._normalized_timestamp(payload.get("last_failure_at"))
            entry["opened_at"] = self._normalized_timestamp(payload.get("opened_at"))
            blocked_until = self._normalized_timestamp(payload.get("blocked_until"))
            if entry["status"] == "open" and not blocked_until and entry["opened_at"]:
                opened_at = self._parse_timestamp(entry["opened_at"])
                if opened_at is not None:
                    blocked_until = (
                        opened_at + timedelta(seconds=entry["reset_timeout_seconds"])
                    ).isoformat()
            entry["blocked_until"] = blocked_until if entry["status"] == "open" else None
            entry["last_reset_at"] = self._normalized_timestamp(payload.get("last_reset_at"))
            normalized[worker_type] = entry

        return normalized

    def _worker_type_circuit_breaker_metadata(
        self,
        metadata: dict[str, Any],
        circuit_breakers: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        policy = self._worker_type_circuit_breaker_policy(metadata)
        normalized_breakers = {
            worker_type: dict(entry) for worker_type, entry in sorted(circuit_breakers.items())
        }
        return {
            **dict(metadata),
            WORKER_TYPE_CIRCUIT_BREAKER_POLICY_KEY: policy,
            WORKER_TYPE_CIRCUIT_BREAKERS_KEY: normalized_breakers,
        }

    @staticmethod
    def _default_worker_type_circuit_breaker(policy: dict[str, Any]) -> dict[str, Any]:
        return {
            "status": "closed",
            "failure_count": 0,
            "failure_threshold": int(policy["failure_threshold"]),
            "reset_timeout_seconds": float(policy["reset_timeout_seconds"]),
            "opened_at": None,
            "blocked_until": None,
            "last_failure_at": None,
            "last_failure_reason": "",
            "last_failure_detail": "",
            "trip_count": 0,
            "last_reset_at": None,
        }

    @staticmethod
    def _normalized_timestamp(value: Any) -> str | None:
        text = str(value or "").strip()
        if not text:
            return None
        return text

    @staticmethod
    def _worker_type_circuit_breaker_is_open(
        circuit_breakers: dict[str, dict[str, Any]],
        worker_type: str,
    ) -> bool:
        entry = circuit_breakers.get(str(worker_type).strip().lower()) or {}
        return str(entry.get("status", "")).strip().lower() == "open"

    def _worker_type_circuit_breaker_detail(
        self,
        worker_type: str,
        breaker: dict[str, Any],
    ) -> str:
        detail = f"{worker_type} breaker open"
        blocked_until = str(breaker.get("blocked_until", "")).strip()
        if blocked_until:
            detail += f" until {blocked_until}"
        last_reason = str(breaker.get("last_failure_reason", "")).strip()
        if last_reason:
            detail += f" after {last_reason}"
        return detail

    def _record_worker_type_failure(
        self,
        circuit_breakers: dict[str, dict[str, Any]],
        worker_type: str,
        *,
        reason: str,
        detail: str,
        open_immediately: bool = False,
        policy: dict[str, Any] | None = None,
    ) -> None:
        normalized_worker_type = str(worker_type).strip().lower()
        if not normalized_worker_type:
            return
        entry = circuit_breakers.get(normalized_worker_type)
        if entry is None:
            entry = self._default_worker_type_circuit_breaker(
                policy
                or {
                    "failure_threshold": DEFAULT_BREAKER_FAILURE_THRESHOLD,
                    "reset_timeout_seconds": DEFAULT_BREAKER_RESET_TIMEOUT_SECONDS,
                }
            )
            circuit_breakers[normalized_worker_type] = entry

        now = datetime.now(UTC)
        threshold = max(1, int(entry.get("failure_threshold", DEFAULT_BREAKER_FAILURE_THRESHOLD)))
        reset_timeout_seconds = max(
            1.0,
            float(entry.get("reset_timeout_seconds", DEFAULT_BREAKER_RESET_TIMEOUT_SECONDS)),
        )
        was_open = str(entry.get("status", "")).strip().lower() == "open"
        failure_count = threshold if open_immediately else int(entry.get("failure_count", 0)) + 1

        entry["failure_count"] = max(
            failure_count, threshold if open_immediately else failure_count
        )
        entry["last_failure_at"] = now.isoformat()
        entry["last_failure_reason"] = str(reason).strip()
        entry["last_failure_detail"] = str(detail).strip()[:1000]

        if open_immediately or entry["failure_count"] >= threshold:
            entry["status"] = "open"
            if not was_open:
                entry["trip_count"] = int(entry.get("trip_count", 0) or 0) + 1
                entry["opened_at"] = now.isoformat()
            elif not entry.get("opened_at"):
                entry["opened_at"] = now.isoformat()
            entry["blocked_until"] = (now + timedelta(seconds=reset_timeout_seconds)).isoformat()
            return

        entry["status"] = "closed"
        entry["opened_at"] = None
        entry["blocked_until"] = None

    def _record_worker_type_success(
        self,
        circuit_breakers: dict[str, dict[str, Any]],
        worker_type: str,
    ) -> None:
        normalized_worker_type = str(worker_type).strip().lower()
        if not normalized_worker_type:
            return
        entry = circuit_breakers.get(normalized_worker_type)
        if entry is None:
            return
        if str(entry.get("status", "")).strip().lower() == "open":
            return
        if (
            int(entry.get("failure_count", 0) or 0) == 0
            and not entry.get("opened_at")
            and not entry.get("blocked_until")
        ):
            return
        self._reset_worker_type_circuit_breaker_entry(
            circuit_breakers,
            normalized_worker_type,
        )

    def _reset_worker_type_circuit_breaker_entry(
        self,
        circuit_breakers: dict[str, dict[str, Any]],
        worker_type: str,
        *,
        now: datetime | None = None,
    ) -> None:
        normalized_worker_type = str(worker_type).strip().lower()
        if not normalized_worker_type:
            return
        entry = circuit_breakers.get(normalized_worker_type)
        if entry is None:
            return
        reset_at = (now or datetime.now(UTC)).isoformat()
        entry["status"] = "closed"
        entry["failure_count"] = 0
        entry["opened_at"] = None
        entry["blocked_until"] = None
        entry["last_reset_at"] = reset_at

    def _expire_worker_type_circuit_breakers(
        self,
        circuit_breakers: dict[str, dict[str, Any]],
    ) -> None:
        now = datetime.now(UTC)
        for worker_type, entry in circuit_breakers.items():
            if str(entry.get("status", "")).strip().lower() != "open":
                continue
            blocked_until = self._parse_timestamp(entry.get("blocked_until"))
            if blocked_until is None:
                opened_at = self._parse_timestamp(entry.get("opened_at"))
                if opened_at is not None:
                    blocked_until = opened_at + timedelta(
                        seconds=float(
                            entry.get(
                                "reset_timeout_seconds",
                                DEFAULT_BREAKER_RESET_TIMEOUT_SECONDS,
                            )
                        )
                    )
                    entry["blocked_until"] = blocked_until.isoformat()
            if blocked_until is None:
                continue
            if now >= blocked_until:
                self._reset_worker_type_circuit_breaker_entry(
                    circuit_breakers,
                    worker_type,
                    now=now,
                )

    def _mark_worker_type_blocked(
        self,
        item: dict[str, Any],
        *,
        worker_type: str,
        detail: str,
    ) -> None:
        metadata = dict(item.get("metadata") or {})
        metadata.update(
            {
                "last_failure_reason": "worker_type_blocked",
                "last_failure_detail": str(detail).strip()[:1000],
                "blocked_worker_type": str(worker_type).strip().lower(),
                "reuse_existing_worktree": True,
            }
        )
        item["metadata"] = metadata
        self._mark_needs_human(
            item,
            f"worker dispatch blocked: {detail}",
        )
        self._release_terminal_lease(item)

    def _release_orphaned_conflict_leases(self, conflicts: list[dict[str, Any]]) -> int:
        released = 0
        for conflict in conflicts:
            if str(conflict.get("source", "lease")).strip() not in {"lease", ""}:
                continue
            lease_id = str(conflict.get("lease_id", "")).strip()
            worktree_path = str(conflict.get("worktree_path", "")).strip()
            if not lease_id or not worktree_path:
                continue
            if Path(worktree_path).exists():
                continue
            self.store.release_lease(lease_id, status=LeaseStatus.RELEASED)
            released += 1
        return released

    @staticmethod
    def _is_resource_constraint_error(exc: Exception) -> bool:
        lowered = str(exc).lower()
        return "no space left on device" in lowered or "disk full" in lowered

    @staticmethod
    def _alternate_agent(agent: str | None) -> str | None:
        value = str(agent or "").strip().lower()
        if value == "claude":
            return "codex"
        if value == "codex":
            return "claude"
        return None

    @staticmethod
    def _completion_confidence(item: dict[str, Any], result: WorkerProcess) -> float:
        expected_tests = [str(test) for test in item.get("expected_tests", []) if str(test).strip()]
        if result.exit_code != 0:
            return 0.0
        if expected_tests:
            return 0.8 if result.tests_run else 0.65
        if result.commit_shas or result.changed_paths:
            return 0.6
        return 0.4

    @staticmethod
    def _verification_results_from_result(result: WorkerProcess) -> list[dict[str, Any]]:
        raw_results = list(getattr(result, "verification_results", []) or [])
        normalized: list[dict[str, Any]] = []
        for entry in raw_results:
            if not isinstance(entry, dict):
                continue
            command = str(entry.get("command", "")).strip()
            if not command:
                continue
            try:
                exit_code = int(entry.get("exit_code", 0))
            except (TypeError, ValueError):
                exit_code = -1
            try:
                duration_seconds = float(entry.get("duration_seconds", 0.0) or 0.0)
            except (TypeError, ValueError):
                duration_seconds = 0.0
            normalized.append(
                {
                    "command": command,
                    "exit_code": exit_code,
                    "passed": bool(entry.get("passed", exit_code == 0)),
                    "stdout": str(entry.get("stdout", "")),
                    "stderr": str(entry.get("stderr", "")),
                    "duration_seconds": duration_seconds,
                }
            )
        if normalized:
            return normalized
        return [
            {
                "command": str(command).strip(),
                "exit_code": 0,
                "passed": True,
                "stdout": "",
                "stderr": "",
                "duration_seconds": 0.0,
            }
            for command in result.tests_run
            if str(command).strip()
        ]

    @classmethod
    def _merge_gate_state(cls, item: dict[str, Any]) -> dict[str, Any]:
        expected_checks = [
            str(test).strip() for test in item.get("expected_tests", []) if str(test).strip()
        ]
        verification_results = [
            dict(entry)
            for entry in item.get("verification_results", [])
            if isinstance(entry, dict) and str(entry.get("command", "")).strip()
        ]
        seen_commands = {
            str(entry.get("command", "")).strip()
            for entry in verification_results
            if str(entry.get("command", "")).strip()
        }
        missing_checks = [command for command in expected_checks if command not in seen_commands]
        failed_checks = [
            dict(entry)
            for entry in verification_results
            if str(entry.get("command", "")).strip() in expected_checks
            and not bool(entry.get("passed", False))
        ]

        blocked_reasons: list[str] = []
        verification_missing_reason: str | None = None
        if not expected_checks:
            verification_missing_reason = "missing_verification_plan"
            blocked_reasons.append(
                "merge gate blocked: missing verification plan for code-change lane"
            )
        if missing_checks:
            blocked_reasons.append(
                "merge gate blocked: required verification did not run: "
                + ", ".join(missing_checks[:3])
            )
        if failed_checks:
            first = failed_checks[0]
            reason = (
                "merge gate blocked: verification failed: "
                f"{first.get('command', '')} (exit {first.get('exit_code', -1)})"
            )
            stderr = str(first.get("stderr", "")).strip()
            if stderr:
                reason = f"{reason} - {stderr.splitlines()[0][:200]}"
            blocked_reasons.append(reason)

        checks_passed = bool(expected_checks) and not missing_checks and not failed_checks
        return {
            "enabled": True,
            "expected_checks": expected_checks,
            "verification_results": verification_results,
            "verification_missing_reason": verification_missing_reason,
            "checks_passed": checks_passed,
            "human_approval_required": True,
            "merge_eligible": checks_passed,
            "blocked_reasons": blocked_reasons,
        }

    @staticmethod
    def _merge_gate_failure_reason(merge_gate: dict[str, Any]) -> str:
        reasons = [
            str(reason).strip()
            for reason in merge_gate.get("blocked_reasons", [])
            if str(reason).strip()
        ]
        return reasons[0] if reasons else "merge gate blocked"

    @staticmethod
    def _progress_fingerprint(source: Any) -> dict[str, Any]:
        payload = dict(source or {})
        return {
            "head_sha": str(payload.get("head_sha", "")).strip(),
            "changed_paths": sorted(
                str(path).strip() for path in payload.get("changed_paths", []) if str(path).strip()
            ),
            "diff_lines": int(payload.get("diff_lines", 0) or 0),
        }

    def _no_progress_timeout_seconds(self) -> float:
        raw = getattr(self.launcher.config, "no_progress_timeout_seconds", 120.0)
        try:
            return max(1.0, float(raw))
        except (TypeError, ValueError):
            return 120.0

    def _exceeded_no_progress_timeout(self, item: dict[str, Any]) -> bool:
        since = self._parse_timestamp(item.get("last_progress_at")) or self._parse_timestamp(
            item.get("dispatched_at")
        )
        if since is None:
            return False
        elapsed = (datetime.now(UTC) - since).total_seconds()
        return elapsed >= self._no_progress_timeout_seconds()

    @staticmethod
    def _parse_timestamp(value: Any) -> datetime | None:
        text = str(value or "").strip()
        if not text:
            return None
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=UTC)
        return parsed

    @staticmethod
    def _mark_needs_human(item: dict[str, Any], reason: str) -> None:
        item["status"] = "needs_human"
        item["dispatch_error"] = reason
        blockers = [str(value).strip() for value in item.get("blockers", []) if str(value).strip()]
        if reason not in blockers:
            blockers.append(reason)
        item["blockers"] = blockers
        item.pop("pid", None)

    def _mark_scope_violation(
        self,
        item: dict[str, Any],
        violations: list[dict[str, Any]],
        *,
        extra_reason: str = "",
    ) -> None:
        """Mark a work order as failed due to file-scope violation.

        This is the fail-closed enforcement gate: workers that edit outside
        their permitted scope are stopped immediately rather than allowed to
        continue producing wrong work.

        Persists the violation into the lease metadata so fleet/integrator
        views can surface it without relying on in-memory work-order state.
        """
        out_of_scope_paths = [
            str(v.get("path", "")) for v in violations if v.get("type") == "out_of_scope"
        ]
        reason = "worker edited files outside permitted scope: " + ", ".join(out_of_scope_paths[:5])
        if extra_reason:
            reason = f"{extra_reason}; {reason}"
        item["status"] = "scope_violation"
        item["dispatch_error"] = reason
        item["review_status"] = "changes_requested"
        scope_violation_detail = {
            "violations": violations,
            "changed_paths": list(item.get("changed_paths", [])),
            "detected_at": datetime.now(UTC).isoformat(),
        }
        item["scope_violation"] = scope_violation_detail
        blockers = [str(v).strip() for v in item.get("blockers", []) if str(v).strip()]
        if reason not in blockers:
            blockers.append(reason)
        item["blockers"] = blockers
        item.pop("pid", None)

        # Write violation metadata into the lease so status_summary() surfaces
        # it.  The lease stays *active* — matching what record_completion() does
        # — so list_active_leases() picks it up for fleet/integrator views.
        lease_id = str(item.get("lease_id", "")).strip()
        if lease_id:
            try:
                self.store.persist_scope_violation(
                    lease_id,
                    changed_paths=list(item.get("changed_paths", [])),
                    violations=violations,
                )
            except Exception:
                pass  # Best-effort — local item is already marked

    @staticmethod
    def _check_file_scope_violations(
        work_order: dict[str, Any],
        changed_paths: list[str],
    ) -> list[dict[str, Any]]:
        """Check whether changed paths fall within the work order's file scope.

        Returns a list of violation dicts (empty = no violations).
        File-scope enforcement is strict: every changed path must match at
        least one scope pattern. If the work order has no file_scope declared,
        no enforcement is applied (open scope).
        """
        file_scope = [
            str(item).strip() for item in work_order.get("file_scope", []) if str(item).strip()
        ]
        if not file_scope or not changed_paths:
            return []

        violations: list[dict[str, Any]] = []
        for path in changed_paths:
            normalized = str(path).strip().removeprefix("./")
            if not normalized:
                continue
            if not any(_path_in_scope(normalized, scope) for scope in file_scope):
                violations.append(
                    {
                        "type": "out_of_scope",
                        "path": normalized,
                        "allowed_scope": list(file_scope),
                    }
                )
        return violations

    async def _kill_worker(self, item: dict[str, Any]) -> None:
        """Kill a running worker process by PID."""
        import signal

        raw_pid = item.get("pid")
        if raw_pid is None:
            return
        try:
            pid = int(raw_pid)
        except (TypeError, ValueError):
            return
        try:
            import os as _os

            _os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        item.pop("pid", None)

    @staticmethod
    def _derive_status(work_orders: list[dict[str, Any]]) -> str:
        statuses = {str(item.get("status", "")).strip() for item in work_orders if item}
        if not statuses:
            return SupervisorRunStatus.PLANNED.value
        terminal = {
            "merged",
            "discarded",
            "salvage",
            "completed",
            "failed",
            "timed_out",
            "scope_violation",
        }
        if statuses <= terminal:
            return SupervisorRunStatus.COMPLETED.value
        if "needs_human" in statuses or "changes_requested" in statuses:
            return SupervisorRunStatus.NEEDS_HUMAN.value
        if "dispatch_failed" in statuses:
            return SupervisorRunStatus.NEEDS_HUMAN.value
        # Deadlocked: only waiting_conflict/waiting_resource remain with no
        # forward-progress statuses (queued/leased/dispatched).  Escalate
        # instead of polling indefinitely.
        forward_progress = {"queued", "leased", "dispatched"}
        non_terminal = statuses - terminal
        if non_terminal and not (non_terminal & forward_progress):
            return SupervisorRunStatus.NEEDS_HUMAN.value
        return SupervisorRunStatus.ACTIVE.value

    @staticmethod
    def _managed_dir_for_agent(pattern: str, agent: str) -> str:
        if "{agent}" in pattern:
            return pattern.format(agent=agent)
        cleaned = pattern.rstrip("/")
        if cleaned.endswith("-auto"):
            return cleaned.replace("codex-auto", f"{agent}-auto")
        return f"{cleaned}/{agent}-auto"

    @staticmethod
    def _looks_like_glob(path: str) -> bool:
        return any(token in path for token in ("*", "?", "["))

    @staticmethod
    def _tests_from_acceptance(acceptance_criteria: list[str]) -> list[str]:
        tests: list[str] = []
        for item in acceptance_criteria:
            text = str(item).strip()
            if text.startswith("python -m pytest") or text.startswith("pytest"):
                tests.append(text)
        return tests

    def _default_tests(self, work_order: BoundedWorkOrder, spec: SwarmSpec) -> list[str]:
        tests = [str(item) for item in work_order.expected_tests if str(item).strip()]
        if tests:
            return tests
        for path in work_order.file_scope:
            if path.startswith("tests/") and path.endswith(".py"):
                tests.append(f"python -m pytest {path} -q")
        if tests:
            return tests
        return self._tests_from_acceptance(spec.acceptance_criteria)

    def _risk_level_for_scope(self, file_scope: list[str]) -> str:
        if not file_scope:
            return "review"
        level = ApprovalLevel.INFO
        for path in file_scope:
            next_level = self.approval_policy.get_approval_level(path)
            if next_level == ApprovalLevel.CRITICAL:
                return "critical"
            if next_level == ApprovalLevel.REVIEW:
                level = ApprovalLevel.REVIEW
        return "review" if level == ApprovalLevel.REVIEW else "info"

    @staticmethod
    def _scope_overlaps_hints(file_scope: list[str], hints: list[str]) -> bool:
        """Check whether any decomposer-assigned scope overlaps with spec hints.

        Delegates to the coordination layer's ``_glob_overlap`` which supports
        exact paths, directory prefixes with ``/`` boundary checks, ``/**``
        recursive globs, and ``PurePosixPath.match()`` for standard glob
        patterns — the same semantics used by file-scope enforcement.

        Pre-strips ``./`` prefixes that ``_glob_overlap`` does not normalize.
        """
        from aragora.nomic.dev_coordination import _glob_overlap

        for scope_path in file_scope:
            clean_scope = scope_path.strip().removeprefix("./")
            if not clean_scope:
                continue
            for hint in hints:
                clean_hint = hint.strip().removeprefix("./")
                if not clean_hint:
                    continue
                if _glob_overlap(clean_scope, clean_hint):
                    return True
        return False

    @staticmethod
    def _task_prompt(spec: SwarmSpec) -> str:
        parts = [spec.refined_goal or spec.raw_goal]
        if spec.file_scope_hints:
            parts.append("File scope hints: " + ", ".join(spec.file_scope_hints))
        if spec.constraints:
            parts.append("Constraints: " + "; ".join(spec.constraints))
        if spec.acceptance_criteria:
            parts.append("Acceptance: " + "; ".join(spec.acceptance_criteria))
        return "\n".join(part for part in parts if part)
