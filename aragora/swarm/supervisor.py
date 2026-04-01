"""Supervisor-driven Codex/Claude swarm orchestration.

Builds bounded work orders from a SwarmSpec, provisions managed worktrees for
Codex/Claude execution targets, claims bounded leases, and persists a
SupervisorRun in the existing development coordination store.
"""

from __future__ import annotations

import asyncio
import logging
import re
import shlex
import subprocess
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aragora.nomic.approval import ApprovalLevel, ApprovalPolicy
from aragora.nomic.dev_coordination import (
    DevCoordinationStore,
    FileScopeViolationError,
    LeaseConflictError,
    LeaseStatus,
)
from aragora.nomic.pipeline_bridge import BoundedWorkOrder, NomicPipelineBridge
from aragora.nomic.task_decomposer import SubTask, TaskDecomposer
from aragora.swarm.lane_telemetry import LaneTelemetryCollector, LaneTelemetryRecord
from aragora.swarm.spec import SwarmSpec
from aragora.swarm.terminal_truth import (
    extract_work_order_deliverable,
    qualify_run_terminal_state,
    qualify_work_order_terminal_state,
)
from aragora.swarm.worker_launcher import SESSION_ARTIFACTS, WorkerLauncher, WorkerProcess
from aragora.worktree.lifecycle import WorktreeLifecycleService

if TYPE_CHECKING:
    from aragora.swarm.pr_registry import PullRequestRegistry

UTC = timezone.utc
logger = logging.getLogger(__name__)
_LANE_TELEMETRY = LaneTelemetryCollector()

WORKER_TYPE_CIRCUIT_BREAKERS_KEY = "worker_type_circuit_breakers"
WORKER_TYPE_CIRCUIT_BREAKER_POLICY_KEY = "worker_type_circuit_breaker_policy"
CAMPAIGN_OUTCOME_METADATA_KEY = "campaign_outcome"
CAMPAIGN_REQUEUE_ELIGIBLE_METADATA_KEY = "campaign_requeue_eligible"
CAMPAIGN_BLOCKERS_METADATA_KEY = "campaign_blockers"
MAX_WORKER_LOG_TAIL_CHARS = 4000
DEFAULT_BREAKER_FAILURE_THRESHOLD = 2
DEFAULT_BREAKER_RESET_TIMEOUT_SECONDS = 900.0
SESSION_LOCK_FILES = (
    ".claude-session-active",
    ".codex_session_active",
    ".nomic-session-active",
)


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


def _ensure_work_order_scope(
    item: BoundedWorkOrder,
    spec: SwarmSpec,
) -> BoundedWorkOrder:
    """Ensure a work order has non-empty file_scope through a 3-tier fallback.

    1. If the work order already has ``file_scope`` entries, merge in spec hints
       (preserving deduplication) so the worker can touch any file the project
       declares.
    2. If the work order has empty ``file_scope`` but the spec carries
       ``file_scope_hints``, backfill them directly.
    3. If both are empty, attempt keyword-based inference from the work order
       title and description via ``SwarmSpec.infer_file_scope_hints()``.

    A warning is logged when scope remains empty after all attempts — this is
    advisory, not blocking, to maintain backward compatibility.

    Returns the (mutated) work order for convenience.
    """
    spec_hints = list(spec.file_scope_hints) if spec.file_scope_hints else []

    if item.file_scope and spec_hints:
        # Merge: keep work order scope, append any new spec hints
        merged = list(dict.fromkeys(item.file_scope + spec_hints))
        if set(merged) != set(item.file_scope):
            logger.info(
                "Merged spec hints into work order %s file_scope: %s -> %s",
                item.work_order_id,
                item.file_scope,
                merged,
            )
        item.file_scope = merged
    elif not item.file_scope and spec_hints:
        # Backfill from spec hints
        item.file_scope = list(spec_hints)
        logger.info(
            "Backfilled empty file_scope on work order %s from spec hints: %s",
            item.work_order_id,
            spec_hints,
        )
    elif not item.file_scope and not spec_hints:
        # Last resort: infer from task title + description
        inference_text = " ".join(filter(None, [item.title or "", item.description or ""]))
        inferred = SwarmSpec.infer_file_scope_hints(inference_text)
        if inferred:
            item.file_scope = inferred
            logger.info(
                "Inferred file_scope on work order %s from title/description: %s",
                item.work_order_id,
                inferred,
            )
        else:
            logger.warning(
                "Work order %s has empty file_scope after all inference attempts "
                "(spec hints empty, keyword inference found nothing). "
                "Scope enforcement will be open for this work order.",
                item.work_order_id,
            )

    return item


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

    _LLM_CALL_TIMEOUT: float = 60.0  # seconds for LLM adjudication/evaluation calls

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
        self._pr_registry: PullRequestRegistry | None = None

    def _get_pr_registry(self) -> PullRequestRegistry:
        """Lazily create and return the shared PullRequestRegistry."""
        if self._pr_registry is None:
            from aragora.swarm.pr_registry import PullRequestRegistry

            state_dir = self.repo_root / ".aragora"
            self._pr_registry = PullRequestRegistry(state_dir=state_dir)
        return self._pr_registry

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
        worker_env: dict[str, str] | None = None,
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
        normalized_worker_env = {
            str(key).strip(): str(value)
            for key, value in dict(worker_env or {}).items()
            if str(key).strip()
        }
        for item in work_orders:
            item.setdefault("status", "queued")
            item.setdefault("lease_id", None)
            item.setdefault("receipt_id", None)
            item.setdefault("review_status", "pending")
            if normalized_worker_env:
                metadata = dict(item.get("metadata") or {})
                metadata["worker_env"] = normalized_worker_env
                item["metadata"] = metadata
        self._suppress_duplicate_open_work_orders(goal, work_orders)

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

        self._collect_finished_results_before_reap(run_id, record)
        # Keep the direct dead-PID salvage path as a second chance before
        # lease reaping so detached workers with completed deliverables do not
        # get downgraded into stale state first.
        try:
            self._collect_finished_workers_sync(run_id)
        except Exception:
            logger.debug("pre-reap worker collection failed", exc_info=True)

        # Reap dead-session active leases before deriving run status so
        # orphaned leased work orders do not remain "active" indefinitely.
        try:
            stale = self.store.reap_stale_leases()
            if stale:
                logger.info("reaped %d stale leases during refresh_run", len(stale))
        except Exception:
            logger.debug("reap_stale_leases failed during refresh_run", exc_info=True)

        # Proactively reap TTL-expired leases so stale locks don't accumulate
        # when no new claim_lease() calls are attempted (e.g. all work orders
        # stuck in waiting_conflict).
        try:
            reaped = self.store.reap_expired_leases()
            if reaped:
                logger.info("reaped %d expired leases during refresh_run", len(reaped))
        except Exception:
            logger.debug("reap_expired_leases failed during refresh_run", exc_info=True)

        record = self.store.get_supervisor_run(run_id)
        if record is None:
            raise KeyError(f"Unknown supervisor run: {run_id}")

        max_concurrency = min(max(1, int(record.get("metadata", {}).get("max_concurrency", 8))), 8)
        managed_dir_pattern = str(
            record.get("metadata", {}).get("managed_dir_pattern", ".worktrees/{agent}-auto")
        )
        work_orders = [dict(item) for item in record.get("work_orders", [])]
        for item in work_orders:
            self._backfill_missing_completion_receipt(item)
        self._reconcile_stale_work_order_state(work_orders)
        active_count = sum(
            1 for item in work_orders if str(item.get("status", "")) in {"leased", "dispatched"}
        )

        if active_count < max_concurrency:
            for item in work_orders:
                if active_count >= max_concurrency:
                    break
                if str(item.get("status", "queued")) not in {"queued", "waiting_conflict"}:
                    continue
                try:
                    leased = self._lease_work_order(
                        run_id=run_id,
                        target_branch=str(record.get("target_branch", "main")),
                        work_order=item,
                        managed_dir_pattern=managed_dir_pattern,
                        approval_policy=SwarmApprovalPolicy.from_dict(
                            record.get("approval_policy")
                        ),
                    )
                    if leased:
                        active_count += 1
                except LeaseConflictError as exc:
                    released = self._release_orphaned_conflict_leases(exc.conflicts)
                    if released:
                        try:
                            leased = self._lease_work_order(
                                run_id=run_id,
                                target_branch=str(record.get("target_branch", "main")),
                                work_order=item,
                                managed_dir_pattern=managed_dir_pattern,
                                approval_policy=SwarmApprovalPolicy.from_dict(
                                    record.get("approval_policy")
                                ),
                            )
                            if leased:
                                active_count += 1
                            continue
                        except LeaseConflictError as retry_exc:
                            exc = retry_exc
                    self._mark_waiting_conflict(item, exc.conflicts)
                except RuntimeError as exc:
                    if self._is_resource_constraint_error(exc):
                        item["status"] = "waiting_resource"
                        item["resource_error"] = str(exc)
                    else:
                        self._mark_needs_human(
                            item,
                            str(exc),
                            failure_reason="work_order_leasing_failed",
                        )
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

    def _collect_finished_workers_sync(self, run_id: str) -> None:
        """Synchronously reconcile dead dispatched workers before lease reaping.

        This path runs when ``refresh_run()`` is called from an async context
        and the normal ``collect_finished_results()`` coroutine cannot be used
        safely. For dead workers with real git commits, synthesize a minimal
        worker result and flow it through the normal result-application path so
        the lane becomes a truthful terminal salvage state before reaping.
        """
        import os
        import subprocess

        record = self.store.get_supervisor_run(run_id)
        if record is None:
            return
        work_orders = [
            dict(item) for item in record.get("work_orders", []) if isinstance(item, dict)
        ]
        metadata = dict(record.get("metadata") or {})
        worker_type_circuit_breaker_policy = self._worker_type_circuit_breaker_policy(metadata)
        worker_type_circuit_breakers = self._worker_type_circuit_breakers(metadata)
        self._expire_worker_type_circuit_breakers(worker_type_circuit_breakers)
        changed = False
        dispatched_ids = [
            str(item.get("work_order_id", "")).strip()
            for item in work_orders
            if str(item.get("status", "")) == "dispatched"
        ]

        try:
            finished = self.launcher.collect_finished_sync(work_order_ids=dispatched_ids)
        except Exception:
            logger.debug(
                "sync finished-worker collection failed for run %s",
                run_id,
                exc_info=True,
            )
            finished = []

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
            self._backfill_missing_completion_receipt(item)
            changed = True

        for item in work_orders:
            if str(item.get("status", "")) != "dispatched":
                continue
            if str(item.get("work_order_id", "")).strip() in finished_by_id:
                continue
            pid = item.get("pid")
            if pid is None:
                continue
            # Check if PID is still alive
            try:
                os.kill(int(pid), 0)
                continue  # Still running
            except (OSError, ValueError):
                pass  # Dead — check for commits

            worktree_path = str(item.get("worktree_path", "")).strip()
            initial_head = str(item.get("initial_head", "")).strip()
            if not worktree_path or not os.path.isdir(worktree_path):
                continue

            try:
                result = self._build_dead_worker_salvage_result(
                    item,
                    worktree_path=worktree_path,
                    initial_head=initial_head,
                )
                if result is None or not result.commit_shas:
                    continue

                item["worker_outcome"] = WorkerOutcome.CRASH_WITH_SALVAGE.value
                self._apply_worker_result(
                    item,
                    result,
                    worker_type_circuit_breakers=worker_type_circuit_breakers,
                    worker_type_circuit_breaker_policy=worker_type_circuit_breaker_policy,
                )
                self._backfill_missing_completion_receipt(item)
                logger.info(
                    "Pre-reap salvage reconciled dead worker %s with %d commits",
                    item.get("work_order_id"),
                    len(result.commit_shas),
                )
                changed = True
            except (subprocess.TimeoutExpired, OSError) as exc:
                logger.debug(
                    "Pre-reap git check failed for %s: %s",
                    item.get("work_order_id"),
                    exc,
                )

        if changed:
            self._record_terminal_work_order_telemetry(run_id, work_orders)
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

    @staticmethod
    def _run_git_capture_sync(
        worktree_path: str,
        *args: str,
        timeout: float = 10.0,
    ) -> subprocess.CompletedProcess[str]:
        import subprocess

        return subprocess.run(
            ["git", *args],
            cwd=worktree_path,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )

    def _build_dead_worker_salvage_result(
        self,
        item: dict[str, Any],
        *,
        worktree_path: str,
        initial_head: str,
    ) -> WorkerProcess | None:
        head_result = self._run_git_capture_sync(worktree_path, "rev-parse", "HEAD")
        if head_result.returncode != 0:
            return None
        head_sha = head_result.stdout.strip()
        base_ref = initial_head or "origin/main"

        rev_result = self._run_git_capture_sync(
            worktree_path,
            "rev-list",
            "--reverse",
            f"{base_ref}..{head_sha}",
        )
        if rev_result.returncode != 0:
            return None
        commit_shas = [line.strip() for line in rev_result.stdout.splitlines() if line.strip()]
        if not commit_shas:
            return None

        paths_result = self._run_git_capture_sync(
            worktree_path,
            "diff",
            "--name-only",
            f"{base_ref}..{head_sha}",
        )
        diff_result = self._run_git_capture_sync(
            worktree_path,
            "diff",
            f"{base_ref}..{head_sha}",
        )
        changed_paths = [line.strip() for line in paths_result.stdout.splitlines() if line.strip()]
        return WorkerProcess(
            work_order_id=str(item.get("work_order_id", "")).strip(),
            agent=str(item.get("target_agent", "codex")).strip() or "codex",
            worktree_path=worktree_path,
            branch=str(item.get("branch", "main")).strip() or "main",
            pid=item.get("pid"),
            session_id=str(item.get("owner_session_id", "")).strip(),
            lease_id=str(item.get("lease_id", "")).strip(),
            completed_at=datetime.now(UTC).isoformat(),
            exit_code=1,
            stdout=str(item.get("stdout_tail", "")).strip(),
            stderr=str(item.get("stderr_tail", "")).strip(),
            diff=diff_result.stdout if diff_result.returncode == 0 else "",
            initial_head=initial_head,
            head_sha=head_sha,
            commit_shas=commit_shas,
            changed_paths=changed_paths,
            expected_tests=[
                str(test).strip() for test in item.get("expected_tests", []) if str(test).strip()
            ],
        )

    def _collect_finished_results_before_reap(
        self,
        run_id: str,
        record: dict[str, Any],
    ) -> None:
        work_orders = [
            dict(item) for item in record.get("work_orders", []) if isinstance(item, dict)
        ]
        if not any(self._should_precollect_finished_result(item) for item in work_orders):
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            return
        try:
            completed = asyncio.run(self.collect_finished_results(run_id))
            if completed:
                logger.info(
                    "pre_reap_collect_finished_results run_id=%s completed=%d",
                    run_id,
                    len(completed),
                )
        except Exception:
            logger.debug(
                "collect_finished_results failed during pre-reap refresh for %s",
                run_id,
                exc_info=True,
            )

    def _should_precollect_finished_result(self, item: dict[str, Any]) -> bool:
        if str(item.get("status", "")).strip() != "dispatched":
            return False
        work_order_id = str(item.get("work_order_id", "")).strip()
        if work_order_id and self.launcher.get_worker(work_order_id) is not None:
            return True
        pid = item.get("pid")
        try:
            return int(pid or 0) > 0
        except (TypeError, ValueError):
            return False

    def _reconcile_stale_work_order_state(self, work_orders: list[dict[str, Any]]) -> None:
        active_leases = {lease.lease_id: lease for lease in self.store.list_active_leases()}
        live_claims = [
            claim for claim in self.store.fleet_store.list_claims() if isinstance(claim, dict)
        ]
        for item in work_orders:
            self._prune_stale_conflicts(item, active_leases, live_claims)
            status = str(item.get("status", "")).strip()
            if status in {"leased", "dispatched"}:
                replacement_lease = self._replacement_active_lease(item, active_leases)
                if replacement_lease is not None:
                    self._apply_active_lease_binding(item, replacement_lease)
                    continue
                if self._should_requeue_stale_work_order(item, active_leases):
                    self._reset_work_order_for_requeue(item)
                    continue
            if self._should_requeue_conflict_only_needs_human(item):
                self._reset_work_order_for_requeue(item)

    @staticmethod
    def _replacement_active_lease(
        item: dict[str, Any],
        active_leases: dict[str, Any],
    ) -> Any | None:
        current_lease_id = str(item.get("lease_id", "")).strip()
        if current_lease_id in active_leases:
            return None

        owner_session_id = str(item.get("owner_session_id", "")).strip()
        work_order_id = str(item.get("work_order_id", "")).strip()
        task_key = str(item.get("task_key", "")).strip()
        branch = str(item.get("branch", "")).strip()
        worktree_path = str(item.get("worktree_path", "")).strip()

        candidates = []
        for lease in active_leases.values():
            if str(getattr(lease, "lease_id", "")).strip() == current_lease_id:
                continue
            if (
                owner_session_id
                and str(getattr(lease, "owner_session_id", "")).strip() != owner_session_id
            ):
                continue
            metadata = getattr(lease, "metadata", {}) or {}
            lease_work_order_id = (
                str(metadata.get("work_order_id", "")).strip()
                or str(getattr(lease, "task_id", "")).strip()
            )
            lease_task_key = str(metadata.get("task_key", "")).strip()
            if work_order_id and lease_work_order_id and lease_work_order_id != work_order_id:
                continue
            if task_key and lease_task_key and lease_task_key != task_key:
                continue
            if branch and str(getattr(lease, "branch", "")).strip() not in {"", branch}:
                continue
            if worktree_path and str(getattr(lease, "worktree_path", "")).strip() not in {
                "",
                worktree_path,
            }:
                continue
            candidates.append(lease)

        if not candidates:
            return None
        return max(candidates, key=lambda lease: str(getattr(lease, "updated_at", "")).strip())

    @staticmethod
    def _apply_active_lease_binding(item: dict[str, Any], lease: Any) -> None:
        item["lease_id"] = str(getattr(lease, "lease_id", "")).strip() or item.get("lease_id")
        item["owner_session_id"] = str(getattr(lease, "owner_session_id", "")).strip() or item.get(
            "owner_session_id"
        )
        item["branch"] = str(getattr(lease, "branch", "")).strip() or item.get("branch")
        item["worktree_path"] = str(getattr(lease, "worktree_path", "")).strip() or item.get(
            "worktree_path"
        )
        if getattr(lease, "owner_agent", None):
            item["target_agent"] = str(getattr(lease, "owner_agent")).strip()
        expected_tests = [
            str(test).strip() for test in getattr(lease, "expected_tests", []) if str(test).strip()
        ]
        if expected_tests:
            item["expected_tests"] = expected_tests
        metadata = getattr(lease, "metadata", {}) or {}
        worker_pid = metadata.get("worker_pid")
        try:
            pid_value = int(worker_pid)
        except (TypeError, ValueError):
            pid_value = None
        if pid_value and pid_value > 0:
            item["pid"] = pid_value
            item["status"] = "dispatched"
            return

        # Rebinding onto an active lease with no worker PID means the lease is
        # live but not yet dispatched. Drop stale dispatch-only state from the
        # replaced lease so the lane can launch cleanly on the next iteration.
        item["status"] = "leased"
        for key in (
            "pid",
            "dispatched_at",
            "last_observed_at",
            "last_progress_at",
            "progress_fingerprint",
        ):
            item.pop(key, None)

    def _prune_stale_conflicts(
        self,
        item: dict[str, Any],
        active_leases: dict[str, Any],
        live_claims: list[dict[str, Any]],
    ) -> None:
        raw_conflicts = item.get("conflicts")
        if not isinstance(raw_conflicts, list) or not raw_conflicts:
            return

        current_lease_id = str(item.get("lease_id", "")).strip()
        live_claim_keys = {
            (
                str(claim.get("session_id", "")).strip(),
                str(claim.get("path", "")).strip(),
            )
            for claim in live_claims
        }
        live_claim_sessions = {
            str(claim.get("session_id", "")).strip()
            for claim in live_claims
            if claim.get("session_id")
        }

        kept: list[dict[str, Any]] = []
        for conflict in raw_conflicts:
            if not isinstance(conflict, dict):
                continue
            source = str(conflict.get("source", "lease")).strip()
            if source in {"lease", ""}:
                conflict_lease_id = str(conflict.get("lease_id", "")).strip()
                if conflict_lease_id == current_lease_id:
                    continue
                if conflict_lease_id and conflict_lease_id in active_leases:
                    kept.append(conflict)
                    continue
                worktree_path = str(conflict.get("worktree_path", "")).strip()
                if worktree_path and self._orphaned_conflict_reason(worktree_path):
                    continue
                if conflict_lease_id and conflict_lease_id not in active_leases:
                    continue
                kept.append(conflict)
                continue
            if source == "fleet_claim":
                session_id = str(conflict.get("session_id", "")).strip()
                path = str(conflict.get("path", "")).strip()
                if (session_id, path) in live_claim_keys:
                    kept.append(conflict)
                    continue
                if session_id and session_id in live_claim_sessions:
                    kept.append(conflict)
                continue
            kept.append(conflict)

        if kept:
            item["conflicts"] = kept
        else:
            item.pop("conflicts", None)

    def _should_requeue_stale_work_order(
        self,
        item: dict[str, Any],
        active_leases: dict[str, Any],
    ) -> bool:
        status = str(item.get("status", "")).strip()
        if status not in {"leased", "dispatched"}:
            return False
        lease_id = str(item.get("lease_id", "")).strip()
        if not lease_id or lease_id in active_leases:
            return False
        pid = item.get("pid")
        try:
            running = int(pid or 0) > 0 and WorkerLauncher._is_pid_running(int(pid))
        except (TypeError, ValueError):
            running = False
        return not running

    @staticmethod
    def _should_requeue_conflict_only_needs_human(item: dict[str, Any]) -> bool:
        if str(item.get("status", "")).strip() != "needs_human":
            return False
        if item.get("conflicts"):
            return False
        if str(item.get("worker_outcome", "")).strip():
            return False
        if str(item.get("dispatch_error", "")).strip():
            return False
        failure_reason = str(item.get("failure_reason", "")).strip()
        if failure_reason and failure_reason != "needs_human":
            return False
        default_question = "What human input is required before rerunning this lane?"
        blocking_question = str(item.get("blocking_question", "")).strip()
        if blocking_question and blocking_question != default_question:
            return False
        blocker = item.get("blocker")
        if isinstance(blocker, dict):
            blocker_reason = str(blocker.get("reason", "")).strip()
            blocker_question = str(blocker.get("question", "")).strip()
            if blocker_reason and blocker_reason != "needs_human":
                return False
            if blocker_question and blocker_question != default_question:
                return False
        blockers = [
            str(blocker_text).strip()
            for blocker_text in item.get("blockers", [])
            if str(blocker_text).strip()
        ]
        if blockers:
            return False
        return True

    @staticmethod
    def _reset_work_order_for_requeue(item: dict[str, Any]) -> None:
        item["status"] = "queued"
        item["review_status"] = "pending"
        # Requeued lanes must start from a clean attempt state. Preserve only
        # dispatch inputs (scope/tests/agent hints), not terminal artifacts from
        # the dead or conflict-only attempt we are replacing.
        for key in (
            "lease_id",
            "owner_session_id",
            "worktree_path",
            "initial_head",
            "pid",
            "dispatched_at",
            "dispatch_error",
            "blocking_question",
            "failure_reason",
            "resource_error",
            "blocker",
            "conflicts",
            "receipt_id",
            "confidence",
            "worker_outcome",
            "completed_at",
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
            item.pop(key, None)
        item.pop("blockers", None)

    def _backfill_missing_completion_receipt(self, item: dict[str, Any]) -> None:
        """Heal older completed lanes that predate receipt propagation fixes."""
        if str(item.get("status", "")).strip().lower() != "completed":
            return
        lease_id = str(item.get("lease_id", "") or "").strip()
        receipt_id = str(item.get("receipt_id") or "").strip()
        deliverable_type = self._work_order_deliverable_type(item)
        if not lease_id or receipt_id or not deliverable_type:
            return
        try:
            receipt = self.store.record_completion(
                lease_id=lease_id,
                owner_agent=str(item.get("target_agent", "")).strip(),
                owner_session_id=str(item.get("owner_session_id", "")).strip(),
                branch=str(item.get("branch", "")).strip(),
                worktree_path=str(item.get("worktree_path", "")).strip(),
                base_sha=str(item.get("initial_head", "")).strip(),
                head_sha=str(item.get("head_sha", "")).strip(),
                commit_shas=list(item.get("commit_shas", []) or []),
                changed_paths=list(item.get("changed_paths", []) or []),
                tests_run=list(item.get("tests_run", []) or []),
                validations_run=list(item.get("tests_run", []) or []),
                assumptions=[],
                blockers=[
                    str(blocker).strip()
                    for blocker in item.get("blockers", [])
                    if str(blocker).strip()
                ],
                outcome=deliverable_type,
                risks=[
                    str(blocker).strip()
                    for blocker in item.get("blockers", [])
                    if str(blocker).strip()
                ],
                pr_url=str(item.get("pr_url", "") or item.get("adopted_pr", "")).strip(),
                pr_number=self._extract_pr_number(
                    str(item.get("pr_url", "") or item.get("adopted_pr", "")).strip()
                ),
                confidence=float(item.get("confidence", 0.0) or 0.0),
                metadata={
                    "task_key": str(item.get("task_key", "")).strip() or None,
                    "verification_results": list(item.get("verification_results", []) or []),
                    "worker_outcome": str(item.get("worker_outcome", "")).strip() or None,
                    "approval_required": bool(item.get("approval_required", False)),
                    "risk_level": str(item.get("risk_level", "")).strip() or None,
                    "success_criteria": dict(item.get("success_criteria") or {}),
                    "backfilled_receipt": True,
                },
                require_session_ownership=False,
            )
        except (FileScopeViolationError, KeyError, ValueError):
            logger.debug("completion receipt backfill skipped", exc_info=True)
            return
        item["receipt_id"] = receipt.receipt_id
        item["confidence"] = receipt.confidence

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
        coordination = self.store.status_summary(include_integrator_artifacts=True)
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
            target_agent = str(item.get("target_agent", "claude")).strip().lower() or "claude"
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
                # Persist worker PID in lease metadata so reap_stale_leases
                # can detect dead processes even if this supervisor dies.
                lease_id = str(item.get("lease_id", "")).strip()
                if lease_id and worker.pid is not None:
                    try:
                        self.store.update_lease_metadata(lease_id, {"worker_pid": worker.pid})
                    except Exception:
                        logger.debug(
                            "Failed to persist worker PID to lease %s",
                            lease_id,
                            exc_info=True,
                        )
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
            self._backfill_missing_completion_receipt(item)
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
        try:
            finished = await self.launcher.collect_finished(work_order_ids=dispatched_ids)
        except Exception:
            logger.debug(
                "in-memory finished-worker collection failed for run %s",
                run_id,
                exc_info=True,
            )
            finished = []
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
            try:
                result = await WorkerLauncher.collect_detached_result(
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
                logger.debug("Detached result collection failed for %s", woid, exc_info=True)
                result = None
            if result is not None:
                finished.append(result)
                finished_ids.add(woid)
                continue

            try:
                progress = await self.launcher.snapshot_progress(item)
            except Exception:
                logger.debug("Progress snapshot failed for %s", woid, exc_info=True)
                continue
            observed_at = datetime.now(UTC).isoformat()
            item["last_observed_at"] = observed_at
            if self._update_log_tails(
                item,
                stdout=str(progress.get("stdout_tail", "")),
                stderr=str(progress.get("stderr_tail", "")),
            ):
                changed = True
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
                    scope_violations = self._llm_adjudicate_scope(item, scope_violations)
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
                        failure_reason="worker_exited_without_receipt",
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
                        self._mark_needs_human(
                            item,
                            reason,
                            failure_reason="worker_no_progress_timeout",
                        )
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
            self._backfill_missing_completion_receipt(item)

        self._record_terminal_work_order_telemetry(run_id, work_orders)
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
            goal,
            file_scope_hints=spec_hints or None,
            acceptance_criteria=list(spec.acceptance_criteria) or None,
            constraints=list(spec.constraints) or None,
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
        work_orders = self._collapse_redundant_work_orders(work_orders, spec)
        if len(work_orders) == 1:
            work_orders[0].description = goal
        for item in work_orders:
            _ensure_work_order_scope(item, spec)
            item.expected_tests = self._default_tests(item, spec)
            item.risk_level = self._risk_level_for_scope(item.file_scope)
            item.approval_required = True
            item.metadata = {
                **dict(item.metadata),
                "acceptance_criteria": list(spec.acceptance_criteria),
                "constraints": list(spec.constraints),
            }
        return work_orders

    @staticmethod
    def _normalized_scope_signature(paths: list[str]) -> tuple[str, ...]:
        normalized: list[str] = []
        seen: set[str] = set()
        for raw in paths:
            path = str(raw).strip()
            if not path or path in seen:
                continue
            seen.add(path)
            normalized.append(path)
        return tuple(sorted(normalized))

    @staticmethod
    def _normalized_goal_signature(value: Any) -> str:
        return " ".join(str(value or "").split()).strip().lower()

    @staticmethod
    def _task_has_concrete_deliverable(task: Any) -> bool:
        metadata = getattr(task, "metadata", {}) or {}
        commit_shas = [
            str(item).strip() for item in (metadata.get("commit_shas") or []) if str(item).strip()
        ]
        pr_url = str(metadata.get("pr_url") or "").strip()
        adopted_pr = str(metadata.get("adopted_pr") or "").strip()
        return bool(getattr(task, "receipt_id", None) or commit_shas or pr_url or adopted_pr)

    def _duplicate_open_work_order_group_key(
        self,
        goal: str,
        file_scope: list[str],
        metadata: dict[str, Any] | None,
    ) -> tuple[str, str, tuple[str, ...]] | None:
        scope = self._normalized_scope_signature(file_scope)
        payload = dict(metadata or {})
        tranche_lane_id = str(payload.get("tranche_lane_id") or "").strip()
        if tranche_lane_id:
            return ("tranche_lane_id", tranche_lane_id, scope)
        goal_key = self._normalized_goal_signature(goal)
        if not goal_key:
            return None
        return ("goal", goal_key, scope)

    @staticmethod
    def _scope_signature_contains(
        container: tuple[str, ...],
        containee: tuple[str, ...],
    ) -> bool:
        if not container or not containee:
            return False
        return all(
            any(_path_in_scope(path, scope_pattern) for scope_pattern in container)
            for path in containee
        )

    def _suppress_duplicate_open_work_orders(
        self,
        goal: str,
        work_orders: list[dict[str, Any]],
    ) -> None:
        active_duplicate_statuses = {
            "queued",
            "leased",
            "dispatched",
            "active",
            "waiting_conflict",
            "dispatch_failed",
            "needs_human",
            "timed_out",
            "failed",
        }
        goal_key = self._normalized_goal_signature(goal)
        existing_by_group: dict[tuple[str, str, tuple[str, ...]], str] = {}
        existing_overlap_candidates: list[tuple[str, str, str, tuple[str, ...]]] = []
        for task in self.store.list_developer_tasks(open_only=True, limit=1000):
            if str(getattr(task, "status", "")).strip().lower() not in active_duplicate_statuses:
                continue
            if self._task_has_concrete_deliverable(task):
                continue
            task_goal = self._normalized_goal_signature(str(getattr(task, "goal", "") or ""))
            task_metadata = getattr(task, "metadata", {}) or {}
            task_scope = self._normalized_scope_signature(
                list(getattr(task, "allowed_paths", []) or [])
            )
            task_lane = str(task_metadata.get("tranche_lane_id") or "").strip()
            task_key = str(getattr(task, "task_key", "")).strip()
            group_key = self._duplicate_open_work_order_group_key(
                str(getattr(task, "goal", "") or ""),
                list(getattr(task, "allowed_paths", []) or []),
                task_metadata,
            )
            if not group_key or group_key in existing_by_group:
                if task_scope and task_key and task_goal:
                    existing_overlap_candidates.append((task_key, task_goal, task_lane, task_scope))
                continue
            existing_by_group[group_key] = task_key
            if task_scope and task_key and task_goal:
                existing_overlap_candidates.append((task_key, task_goal, task_lane, task_scope))

        if not existing_by_group and not existing_overlap_candidates:
            return

        now = datetime.now(UTC).isoformat()
        for item in work_orders:
            if str(item.get("status", "")).strip().lower() == "discarded":
                continue
            item_scope = self._normalized_scope_signature(
                [str(path) for path in item.get("file_scope", []) if str(path).strip()]
            )
            item_lane = str((item.get("metadata") or {}).get("tranche_lane_id") or "").strip()
            group_key = self._duplicate_open_work_order_group_key(
                goal,
                [str(path) for path in item.get("file_scope", []) if str(path).strip()],
                dict(item.get("metadata") or {}),
            )
            canonical_task_key = existing_by_group.get(group_key) if group_key else None
            if not canonical_task_key and item_scope:
                for (
                    task_key,
                    existing_goal,
                    existing_lane,
                    existing_scope,
                ) in existing_overlap_candidates:
                    same_lane = bool(item_lane and existing_lane and item_lane == existing_lane)
                    same_goal = bool(goal_key and existing_goal == goal_key)
                    if not same_lane and not same_goal:
                        continue
                    if self._scope_signature_contains(existing_scope, item_scope) or (
                        self._scope_signature_contains(item_scope, existing_scope)
                    ):
                        canonical_task_key = task_key
                        break
            if not group_key or not canonical_task_key:
                if group_key:
                    existing_by_group.setdefault(
                        group_key, str(item.get("work_order_id", "")).strip()
                    )
                if item_scope and goal_key:
                    existing_overlap_candidates.append(
                        (
                            str(item.get("work_order_id", "")).strip(),
                            goal_key,
                            item_lane,
                            item_scope,
                        )
                    )
                continue
            metadata = dict(item.get("metadata") or {})
            metadata.update(
                {
                    "archived_due_to": "duplicate_open_work_order",
                    "archived_at": now,
                    "archive_reason": "duplicate_open_work_order",
                    "canonical_task_key": canonical_task_key,
                    "previous_status": str(item.get("status") or "queued").strip() or "queued",
                }
            )
            item["metadata"] = metadata
            item["status"] = "discarded"

    def _collapse_redundant_work_orders(
        self,
        work_orders: list[BoundedWorkOrder],
        spec: SwarmSpec,
    ) -> list[BoundedWorkOrder]:
        """Collapse decomposition noise when every lane targets the same bounded scope.

        Boss-loop issue bodies can sometimes be over-decomposed into multiple
        phase-style work orders ("CLI Changes", "Tests Changes", etc.) that all
        claim the same file scope. Those lanes cannot make independent forward
        progress because lease enforcement serializes identical scopes anyway.
        Converting them back into one bounded work order preserves the file
        contract while avoiding waiting_conflict fan-out.
        """

        if len(work_orders) <= 1:
            return work_orders

        spec_scope = self._normalized_scope_signature(list(spec.file_scope_hints))
        if not spec_scope:
            return work_orders

        order_scopes = {
            self._normalized_scope_signature(list(item.file_scope)) for item in work_orders
        }
        if order_scopes != {spec_scope}:
            return work_orders

        tests: list[str] = []
        seen_tests: set[str] = set()
        for item in work_orders:
            for test in item.expected_tests:
                normalized = str(test).strip()
                if not normalized or normalized in seen_tests:
                    continue
                seen_tests.add(normalized)
                tests.append(normalized)

        first = work_orders[0]
        collapsed = BoundedWorkOrder(
            work_order_id=first.work_order_id,
            pipeline_task_id=first.pipeline_task_id,
            title=first.title,
            description=spec.refined_goal or spec.raw_goal or first.description,
            file_scope=list(spec_scope),
            dependency_ids=[],
            success_criteria={
                **dict(first.success_criteria),
                "tests": tests or list(first.success_criteria.get("tests", [])),
            },
            expected_tests=tests or list(first.expected_tests),
            estimated_complexity=first.estimated_complexity,
            risk_level=first.risk_level,
            target_agent=first.target_agent,
            reviewer_agent=first.reviewer_agent,
            approval_required=True,
            metadata={
                **dict(first.metadata),
                "collapsed_redundant_work_orders": [item.work_order_id for item in work_orders],
                "source": "collapsed_decomposition",
            },
        )
        logger.info(
            "Collapsed %d redundant work orders with identical scope %s into %s",
            len(work_orders),
            list(spec_scope),
            collapsed.work_order_id,
        )
        return [collapsed]

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

        # Merge spec.file_scope_hints into explicit work orders so scope
        # enforcement is never bypassed when the upstream planner leaves
        # file_scope empty on individual work orders (fixes #884).
        spec_hints = list(spec.file_scope_hints) if spec.file_scope_hints else []
        for item in work_orders:
            if spec_hints:
                if not item.file_scope:
                    item.file_scope = list(spec_hints)
                    logger.info(
                        "Backfilled empty file_scope on explicit work order %s from spec hints: %s",
                        item.work_order_id,
                        spec_hints,
                    )
                else:
                    merged = list(dict.fromkeys(item.file_scope + list(spec_hints)))
                    if set(merged) != set(item.file_scope):
                        logger.info(
                            "Merged spec hints into explicit work order %s file_scope: %s -> %s",
                            item.work_order_id,
                            item.file_scope,
                            merged,
                        )
                    item.file_scope = merged
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
    ) -> bool:
        target_agent = str(work_order.get("target_agent", "codex")).strip() or "codex"
        managed_dir = self._managed_dir_for_agent(managed_dir_pattern, target_agent)
        wo_id = str(work_order.get("work_order_id", "task"))
        task_key = f"{run_id}:{wo_id}"
        session_key = f"swarm-{run_id[:8]}-{wo_id}"
        raw_scope = [str(item) for item in work_order.get("file_scope", []) if str(item).strip()]
        if not raw_scope:
            self._mark_needs_human(
                work_order,
                "Work order has no declared file scope; declare scope before dispatch.",
                failure_reason="scope_violation",
            )
            return False
        session = self.lifecycle.ensure_managed_worktree(
            managed_dir=managed_dir,
            base_branch=target_branch,
            agent=target_agent,
            session_id=session_key,
            reconcile=True,
            strategy="ff-only",
        )
        file_scope = self._validate_file_scope(raw_scope, str(session.path))
        if len(file_scope) != len(raw_scope):
            work_order["file_scope"] = file_scope
        if not file_scope:
            self._mark_needs_human(
                work_order,
                "Declared file scope resolved to no valid in-repo paths; declare scope before dispatch.",
                failure_reason="scope_violation",
            )
            return False
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
                "task_key": task_key,
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
                "task_key": task_key,
            }
        )
        return True

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
        has_deliverable = any(self._work_order_deliverable_type(item) for item in work_orders)
        if not outcome:
            payload.pop(CAMPAIGN_OUTCOME_METADATA_KEY, None)
            payload.pop(CAMPAIGN_REQUEUE_ELIGIBLE_METADATA_KEY, None)
            payload.pop(CAMPAIGN_BLOCKERS_METADATA_KEY, None)
            return payload

        payload[CAMPAIGN_OUTCOME_METADATA_KEY] = outcome
        payload[CAMPAIGN_REQUEUE_ELIGIBLE_METADATA_KEY] = (
            self._campaign_requeue_eligible(outcome) and not has_deliverable
        )
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
        statuses = {
            str(item.get("status", "")).strip().lower()
            for item in work_orders
            if isinstance(item, dict) and str(item.get("status", "")).strip()
        }
        worker_outcomes = {
            str(item.get("worker_outcome", "")).strip().lower()
            for item in work_orders
            if isinstance(item, dict) and str(item.get("worker_outcome", "")).strip()
        }
        # Definitive terminal signals take precedence over stalled — a crash
        # or scope_violation is a concrete outcome that trumps waiting states.
        crash_outcomes = {
            WorkerOutcome.CRASH.value,
            WorkerOutcome.CRASH_WITH_SALVAGE.value,
        }
        if worker_outcomes & crash_outcomes:
            blockers = cls._campaign_blockers_from_work_orders(work_orders)
            return "crash", blockers
        if "scope_violation" in statuses:
            blockers = cls._campaign_blockers_from_work_orders(work_orders)
            return "blocked", blockers

        forward_progress_statuses = {"queued", "leased", "dispatched"}
        stalled_wait_statuses = {"waiting_conflict", "waiting_resource"}
        stalled_dead_end = bool(statuses & stalled_wait_statuses) and not (
            statuses & forward_progress_statuses
        )
        if stalled_dead_end or WorkerOutcome.TIMEOUT_NO_PROGRESS.value in worker_outcomes:
            blockers = cls._campaign_blockers_from_work_orders(work_orders)
            return "stalled", blockers

        qualification = qualify_run_terminal_state(
            {
                "status": cls._derive_status(work_orders),
                "work_orders": [dict(item) for item in work_orders if isinstance(item, dict)],
            }
        )
        if qualification.terminal_outcome == "unknown":
            return None, qualification.reasons
        return qualification.terminal_outcome, qualification.reasons

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
        deliverable = extract_work_order_deliverable(item, require_terminal_status=False)
        if not deliverable:
            return None
        deliverable_type = str(deliverable.get("type", "")).strip()
        if deliverable_type == "adopted_pr":
            return "pr_adopted"
        return "deliverable_created"

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
        self._update_log_tails(item, stdout=result.stdout, stderr=result.stderr)
        item.pop("pid", None)

        # Preserve worker_outcome if already set by detached/timeout collection
        # paths — those have more specific context (e.g. timeout_with_salvage).
        _pre_outcome = str(item.get("worker_outcome", "")).strip()

        # Fail closed: check file-scope before accepting any result as successful
        scope_violations = self._check_file_scope_violations(item, clean_paths)
        if scope_violations:
            # LLM adjudication: ask frontier model if violations are justified
            scope_violations = self._llm_adjudicate_scope(item, scope_violations)
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
                    failure_reason="clean_exit_no_deliverable",
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
                    failure_reason="clean_exit_no_deliverable",
                )
                self._release_terminal_lease(item)
                item["exit_code"] = result.exit_code
                return
            elif not _pre_outcome:
                item["worker_outcome"] = WorkerOutcome.COMPLETED.value

            # Salvaged deliverables skip the merge gate — they are best-effort
            # recoveries where strict verification is inappropriate.
            _salvage_outcomes = {
                WorkerOutcome.TIMEOUT_WITH_SALVAGE.value,
                WorkerOutcome.CRASH_WITH_SALVAGE.value,
            }
            _is_salvage = str(item.get("worker_outcome", "")).strip() in _salvage_outcomes
            merge_gate = self._merge_gate_state(item)
            item["merge_gate"] = merge_gate
            if merge_gate.get("verification_missing_reason"):
                item["verification_missing_reason"] = merge_gate["verification_missing_reason"]
            if not _is_salvage and not bool(merge_gate.get("checks_passed")):
                # LLM second opinion: is the merge gate failure genuine?
                can_override_merge_gate = not bool(merge_gate.get("verification_missing_reason"))
                if can_override_merge_gate and self._llm_override_merge_gate(item, merge_gate):
                    # LLM says deliverable is ready despite gate failure
                    merge_gate["checks_passed"] = True
                    merge_gate["llm_override"] = True
                    item["merge_gate"] = merge_gate
                else:
                    self._mark_needs_human(
                        item,
                        self._merge_gate_failure_reason(merge_gate),
                        failure_reason=str(
                            merge_gate.get("verification_missing_reason", "") or "merge_gate_failed"
                        ).strip()
                        or "merge_gate_failed",
                        blocking_question=self._merge_gate_blocking_question(merge_gate),
                    )
                    item["review_status"] = "changes_requested"
                    item["receipt_id"] = None
                    if not _pre_outcome:
                        item["worker_outcome"] = WorkerOutcome.MERGE_GATE_FAILED.value
                    self._release_terminal_lease(item)
                    item["exit_code"] = result.exit_code
                    return

            receipt_id = str(item.get("receipt_id") or "").strip()
            if lease_id and not receipt_id:
                try:
                    receipt = self.store.record_completion(
                        lease_id=lease_id,
                        owner_agent=str(item.get("target_agent", result.agent)),
                        owner_session_id=str(item.get("owner_session_id", result.session_id)),
                        branch=str(item.get("branch", result.branch)),
                        worktree_path=str(item.get("worktree_path", result.worktree_path)),
                        base_sha=str(item.get("initial_head", result.initial_head)),
                        head_sha=str(result.head_sha or item.get("head_sha", "")),
                        commit_shas=list(result.commit_shas),
                        changed_paths=clean_paths,
                        tests_run=list(result.tests_run),
                        validations_run=list(result.tests_run),
                        assumptions=[],
                        blockers=[
                            str(blocker).strip()
                            for blocker in item.get("blockers", [])
                            if str(blocker).strip()
                        ],
                        outcome=self._work_order_deliverable_type(item) or "completed",
                        risks=[
                            str(blocker).strip()
                            for blocker in item.get("blockers", [])
                            if str(blocker).strip()
                        ],
                        pr_url=str(item.get("pr_url", "") or item.get("adopted_pr", "")).strip(),
                        pr_number=self._extract_pr_number(
                            str(item.get("pr_url", "") or item.get("adopted_pr", "")).strip()
                        ),
                        confidence=self._completion_confidence(item, result),
                        metadata={
                            "task_key": str(item.get("task_key", "")).strip() or None,
                            "verification_results": list(
                                item.get("verification_results", []) or []
                            ),
                            "worker_outcome": str(item.get("worker_outcome", "")).strip() or None,
                            "approval_required": bool(item.get("approval_required", False)),
                            "risk_level": str(item.get("risk_level", "")).strip() or None,
                            "success_criteria": dict(item.get("success_criteria") or {}),
                        },
                    )
                except FileScopeViolationError as exc:
                    self._mark_needs_human(
                        item,
                        "worker completion violated file-scope ownership; narrow or split the lane",
                        failure_reason="scope_violation",
                        blocking_question=(
                            "Which files should stay in scope, or should this lane be split "
                            "before it is rerun?"
                        ),
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
            # Register PR in canonical registry if the work order produced one
            self._register_pr_if_present(item, result)
            item["status"] = "completed"
            item["review_status"] = "pending_heterogeneous_review"
            item.pop("failure_reason", None)
            item.pop("blocking_question", None)
            item.pop("blocker", None)
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

        deliverable_present = bool(self._work_order_deliverable_type(item))
        salvage_outcome = str(item.get("worker_outcome", "")).strip()
        is_salvage = salvage_outcome in {
            WorkerOutcome.CRASH_WITH_SALVAGE.value,
            WorkerOutcome.TIMEOUT_WITH_SALVAGE.value,
        }
        if deliverable_present and is_salvage:
            # Salvaged deliverables proceed to completion — the recovery was
            # intentional and the deliverable (commits/PR) is real.
            item["status"] = "completed"
            item["review_status"] = "pending_heterogeneous_review"
            item["exit_code"] = result.exit_code
            self._release_terminal_lease(item)
            self._register_pr_if_present(item, result)
            return
        if deliverable_present:
            failure_reason = "worker_crash_with_deliverable"
            self._mark_needs_human(
                item,
                "worker exited non-zero after producing a recoverable deliverable",
                failure_reason=failure_reason,
                blocking_question=(
                    "Should the recovered deliverable be adopted as-is, amended, or rerun before integration?"
                ),
            )
            item["review_status"] = "changes_requested"
            item["receipt_id"] = None
            self._release_terminal_lease(item)
            item["exit_code"] = result.exit_code
            return

        if lease_id:
            self.store.release_lease(lease_id, status=LeaseStatus.RELEASED)
        failure_reason = (
            "worker_timeout_no_deliverable" if result.exit_code == -1 else "worker_crash"
        )
        blocking_question = self._default_blocking_question(failure_reason)
        stderr_text = result.stderr.strip()
        item["status"] = "timed_out" if result.exit_code == -1 else "failed"
        item["dispatch_error"] = stderr_text or (
            "worker timed out before producing a deliverable"
            if result.exit_code == -1
            else "worker crashed before producing a deliverable"
        )
        item["failure_reason"] = failure_reason
        item["blocking_question"] = blocking_question
        item["blocker"] = {
            "reason": failure_reason,
            "question": blocking_question,
        }
        blockers = [str(value).strip() for value in item.get("blockers", []) if str(value).strip()]
        if item["dispatch_error"] not in blockers:
            blockers.append(item["dispatch_error"])
        item["blockers"] = blockers
        item["review_status"] = "changes_requested"
        item["exit_code"] = result.exit_code

    def _release_terminal_lease(self, item: dict[str, Any]) -> None:
        lease_id = str(item.get("lease_id", "")).strip()
        if not lease_id:
            return
        try:
            self.store.release_lease(lease_id, status=LeaseStatus.RELEASED)
        except KeyError:
            return

    def _record_terminal_work_order_telemetry(
        self,
        run_id: str,
        work_orders: list[dict[str, Any]],
    ) -> None:
        for item in work_orders:
            if not isinstance(item, dict):
                continue
            qualification = qualify_work_order_terminal_state(item)
            if qualification.terminal_outcome == "unknown":
                continue
            started_at = self._parse_timestamp(item.get("started_at")) or self._parse_timestamp(
                item.get("dispatched_at")
            )
            completed_at = self._parse_timestamp(item.get("completed_at"))
            duration_seconds = 0.0
            if started_at is not None and completed_at is not None:
                duration_seconds = max(0.0, (completed_at - started_at).total_seconds())
            receipt_id = str(item.get("receipt_id") or "").strip()
            pr_reference = str(item.get("pr_url", "") or item.get("adopted_pr", "") or "").strip()
            false_success_candidate = (
                qualification.terminal_outcome
                in {
                    "deliverable_created",
                    "pr_adopted",
                }
                and qualification.deliverable is None
            )
            try:
                _LANE_TELEMETRY.record_lane(
                    LaneTelemetryRecord(
                        lane_kind="supervisor_work_order",
                        lane_id=str(
                            item.get("task_key")
                            or item.get("work_order_id")
                            or item.get("lease_id")
                            or ""
                        ).strip(),
                        run_id=run_id,
                        task_id=str(item.get("task_key", "")).strip(),
                        work_order_id=str(item.get("work_order_id", "")).strip(),
                        terminal_outcome=qualification.terminal_outcome,
                        worker_outcome=str(item.get("worker_outcome", "")).strip(),
                        deliverable_type=str(qualification.deliverable_type or ""),
                        receipt_id=receipt_id,
                        human_intervention_required=qualification.human_intervention_required,
                        duration_seconds=duration_seconds,
                        pr_url=pr_reference,
                        pr_number=self._extract_pr_number(pr_reference),
                        false_success_candidate=false_success_candidate,
                        metadata={
                            "status": str(item.get("status", "")).strip() or None,
                            "failure_reason": str(item.get("failure_reason", "")).strip() or None,
                            "blocking_question": str(item.get("blocking_question", "")).strip()
                            or None,
                        },
                    )
                )
            except Exception:
                logger.debug("Supervisor lane telemetry emission skipped", exc_info=True)

    def _register_pr_if_present(self, item: dict[str, Any], result: WorkerProcess) -> None:
        """Register the work order's PR in the canonical PR registry if present."""
        pr_url = str(item.get("pr_url", "") or "").strip()
        adopted_pr = str(item.get("adopted_pr", "") or "").strip()
        url = pr_url or adopted_pr
        if not url:
            return
        branch = str(item.get("branch", "") or result.branch or "").strip()
        if not branch:
            return
        creator = str(
            item.get("work_order_id", "") or item.get("target_agent", "") or result.agent
        ).strip()
        try:
            self._get_pr_registry().register(branch, url, creator=creator)
        except Exception:
            logger.debug("Failed to register PR %s in registry", url, exc_info=True)

    @staticmethod
    def _extract_pr_number(pr_reference: str) -> int | None:
        text = str(pr_reference or "").strip().rstrip("/")
        if not text:
            return None
        tail = text.rsplit("/", 1)[-1]
        return int(tail) if tail.isdigit() else None

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
        # A fallback retry is a fresh attempt in the same lease/worktree. It
        # must not inherit terminal artifacts from the failed worker.
        for key in (
            "pid",
            "blockers",
            "dispatched_at",
            "last_observed_at",
            "last_progress_at",
            "progress_fingerprint",
            "failure_reason",
            "blocking_question",
            "blocker",
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
            item.pop(key, None)
        return True

    @staticmethod
    def _dispatch_failure_reason(exc: Exception) -> str:
        message = str(exc).strip().lower()
        if "cli not found" in message or "not found" in message:
            return "agent_unavailable"
        return "agent_launch_failed"

    def _capacity_failure_detail(self, result: WorkerProcess) -> str:
        """Detect capacity/billing failures in worker output.

        Uses LLM classification first, falling back to keyword patterns.
        """
        combined = "\n".join(part for part in (result.stdout, result.stderr) if part).strip()
        if not combined:
            return ""

        # --- LLM classification ---
        llm_succeeded = False
        try:
            from concurrent.futures import ThreadPoolExecutor

            from aragora.ralph.llm_classifier import LLMBlockerClassifier

            import asyncio

            classifier = LLMBlockerClassifier()
            with ThreadPoolExecutor(max_workers=1) as pool:
                verdict = pool.submit(
                    asyncio.run,
                    classifier.detect_capacity_failure(
                        stdout=result.stdout or "",
                        stderr=result.stderr or "",
                        agent_name=result.agent or "unknown",
                    ),
                ).result(timeout=self._LLM_CALL_TIMEOUT)
            # Only trust the LLM verdict if it actually ran (not a fallback default)
            if verdict.reasoning != "LLM call failed":
                llm_succeeded = True
                if verdict.is_capacity:
                    logger.info(
                        "LLM capacity detection: is_capacity=%s (reasoning: %s)",
                        verdict.is_capacity,
                        verdict.reasoning,
                    )
                    return verdict.detail or combined or f"{result.agent} worker failed"
                return ""
        except Exception:
            logger.debug("LLM capacity detection failed, using keyword fallback", exc_info=True)

        # --- keyword fallback (when LLM unavailable) ---
        if not llm_succeeded:
            return self._keyword_capacity_failure_detail(combined, result.agent or "unknown")
        return ""

    @staticmethod
    def _keyword_capacity_failure_detail(combined: str, agent_name: str) -> str:
        """Keyword-based fallback for capacity failure detection."""
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
            return combined or f"{agent_name} worker failed"
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
            failure_reason="worker_type_blocked",
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
            orphaned_reason = self._orphaned_conflict_reason(worktree_path)
            if not orphaned_reason:
                continue
            self.store.release_lease(lease_id, status=LeaseStatus.RELEASED)
            logger.info(
                "released_orphaned_conflict_lease lease_id=%s reason=%s worktree=%s",
                lease_id,
                orphaned_reason,
                worktree_path,
            )
            released += 1
        return released

    def _orphaned_conflict_reason(self, worktree_path: str) -> str | None:
        path = Path(worktree_path)
        if not path.exists():
            return "missing_worktree"

        lock_state = self._session_lock_state(path)
        if lock_state == "active":
            return None
        if lock_state == "stale":
            return "dead_session_lock"

        session_meta = WorkerLauncher._read_session_meta(str(path))
        if session_meta:
            if str(session_meta.get("ended_at", "")).strip():
                return "session_ended"
            raw_pid = session_meta.get("pid")
            try:
                pid = int(raw_pid)
            except (TypeError, ValueError):
                pid = None
            if pid is None:
                return None
            if WorkerLauncher._is_pid_running(pid):
                return None
            return "dead_session_pid"

        if self._is_managed_worktree(path):
            return "managed_worktree_without_active_session"
        return None

    def _is_managed_worktree(self, path: Path) -> bool:
        managed_root = (self.repo_root / ".worktrees").resolve()
        try:
            return path.resolve().is_relative_to(managed_root)
        except ValueError:
            return False

    @classmethod
    def _session_lock_state(cls, worktree_path: Path) -> str:
        found_lock = False
        for lock_name in SESSION_LOCK_FILES:
            lock_path = worktree_path / lock_name
            if not lock_path.exists():
                continue
            found_lock = True
            try:
                pids = cls._parse_session_lock_pids(lock_path)
            except OSError:
                return "active"
            if not pids:
                return "active"
            if any(WorkerLauncher._is_pid_running(pid) for pid in pids):
                return "active"
        return "stale" if found_lock else "missing"

    @staticmethod
    def _parse_session_lock_pids(lock_path: Path) -> list[int]:
        raw = lock_path.read_text(encoding="utf-8")
        pids: list[int] = []
        for line in raw.splitlines():
            entry = line.strip()
            if "=" not in entry:
                continue
            key, value = entry.split("=", 1)
            if key.strip() not in {"pid", "ppid"}:
                continue
            value = value.strip()
            if value.isdigit():
                pids.append(int(value))
        return pids

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

    @staticmethod
    def _canonical_verification_command(command: Any) -> str:
        text = str(command or "").strip()
        if not text:
            return ""
        for prefix in ("bash -lc ", "/bin/bash -lc "):
            if text.startswith(prefix):
                text = text[len(prefix) :].strip()
                if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
                    text = text[1:-1]
                break
        text = re.sub(r"^(?P<prefix>\s*)python3(?=\s|$)", r"\g<prefix>python", text)
        return WorkerLauncher._normalize_verification_command(text).strip()

    @classmethod
    def _pytest_command_targets(cls, command: Any) -> list[str]:
        text = cls._canonical_verification_command(command)
        if not text:
            return []
        try:
            tokens = shlex.split(text)
        except ValueError:
            return []
        if not tokens:
            return []
        start = 0
        if len(tokens) >= 3 and tokens[1] == "-m" and tokens[2] == "pytest":
            start = 3
        elif tokens[0].endswith("pytest"):
            start = 1
        else:
            return []

        targets: list[str] = []
        skip_next = False
        options_with_values = {"-k", "-m", "--maxfail", "--timeout", "--tb", "-c", "--rootdir"}
        for token in tokens[start:]:
            if skip_next:
                skip_next = False
                continue
            if token in options_with_values:
                skip_next = True
                continue
            if token.startswith("-"):
                continue
            normalized = str(token).strip().removeprefix("./").rstrip("/")
            if token.endswith("/") or "/" in normalized or normalized.endswith(".py"):
                targets.append(normalized)
        return targets

    @classmethod
    def _pytest_command_has_selectors(cls, command: Any) -> bool:
        """Return True if the pytest command contains -k or -m selectors."""
        text = cls._canonical_verification_command(command)
        if not text:
            return False
        try:
            tokens = shlex.split(text)
        except ValueError:
            return False
        start = 0
        if len(tokens) >= 3 and tokens[1] == "-m" and tokens[2] == "pytest":
            start = 3
        elif tokens and tokens[0].endswith("pytest"):
            start = 1
        else:
            return False
        skip_next = False
        for token in tokens[start:]:
            if skip_next:
                skip_next = False
                continue
            if token in {"-k", "-m"}:
                return True
            options_with_values = {"--maxfail", "--timeout", "--tb", "-c", "--rootdir"}
            if token in options_with_values:
                skip_next = True
                continue
        return False

    @classmethod
    def _verification_command_covers_expected(
        cls, recorded_command: Any, expected_command: Any
    ) -> bool:
        recorded = cls._canonical_verification_command(recorded_command)
        expected = cls._canonical_verification_command(expected_command)
        if not recorded or not expected:
            return False
        if recorded == expected:
            return True
        recorded_targets = cls._pytest_command_targets(recorded)
        expected_targets = cls._pytest_command_targets(expected)
        if not recorded_targets or not expected_targets:
            return False
        if cls._pytest_command_has_selectors(recorded) or cls._pytest_command_has_selectors(
            expected
        ):
            return False
        for expected_target in expected_targets:
            if not any(
                expected_target == recorded_target
                or expected_target.startswith(recorded_target.rstrip("/") + "/")
                for recorded_target in recorded_targets
            ):
                return False
        return True

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
        missing_checks = [
            command
            for command in expected_checks
            if not any(
                cls._verification_command_covers_expected(entry.get("command", ""), command)
                for entry in verification_results
            )
        ]
        failed_checks = [
            dict(entry)
            for entry in verification_results
            if any(
                cls._verification_command_covers_expected(entry.get("command", ""), command)
                for command in expected_checks
            )
            and not bool(entry.get("passed", False))
        ]

        blocked_reasons: list[str] = []
        verification_missing_reason: str | None = None
        if not expected_checks:
            if cls._work_order_is_docs_only(item):
                return {
                    "enabled": True,
                    "expected_checks": expected_checks,
                    "verification_results": verification_results,
                    "verification_missing_reason": None,
                    "checks_passed": True,
                    "human_approval_required": True,
                    "merge_eligible": True,
                    "blocked_reasons": [],
                }
            verification_missing_reason = "missing_verification_plan"
            blocked_reasons.append(
                "merge gate blocked: missing verification plan or verification command"
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

        checks_passed = not blocked_reasons
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

    _DOCS_SAFE_PREFIXES = ("docs/", "docs-site/")
    _DOCS_SAFE_FILENAMES = frozenset(
        {
            "CHANGELOG.md",
            "CODE_OF_CONDUCT.md",
            "CONTRIBUTING.md",
            "LICENSE",
            "LICENSE.md",
        }
    )

    @staticmethod
    def _is_docs_only_path(path: Any) -> bool:
        normalized = str(path or "").strip().removeprefix("./")
        if not normalized:
            return False
        if any(normalized.startswith(prefix) for prefix in SwarmSupervisor._DOCS_SAFE_PREFIXES):
            return True
        basename = normalized.rsplit("/", 1)[-1] if "/" in normalized else normalized
        return basename in SwarmSupervisor._DOCS_SAFE_FILENAMES

    @classmethod
    def _work_order_is_docs_only(cls, item: dict[str, Any]) -> bool:
        candidates = [
            str(path).strip()
            for path in item.get("changed_paths", []) or item.get("file_scope", [])
            if str(path).strip()
        ]
        if not candidates:
            return False
        return all(cls._is_docs_only_path(path) for path in candidates)

    @staticmethod
    def _merge_gate_failure_reason(merge_gate: dict[str, Any]) -> str:
        reasons = [
            str(reason).strip()
            for reason in merge_gate.get("blocked_reasons", [])
            if str(reason).strip()
        ]
        return reasons[0] if reasons else "merge gate blocked"

    @staticmethod
    def _merge_gate_blocking_question(merge_gate: dict[str, Any]) -> str:
        missing = str(merge_gate.get("verification_missing_reason", "")).strip()
        if missing == "missing_verification_plan":
            return (
                "Which verification command or acceptance check should be added before rerunning?"
            )
        return "Which required verification or acceptance check must pass before approval?"

    @classmethod
    def _update_log_tails(
        cls,
        item: dict[str, Any],
        *,
        stdout: str,
        stderr: str,
    ) -> bool:
        changed = False
        for key, value in {
            "stdout_tail": cls._log_tail(stdout),
            "stderr_tail": cls._log_tail(stderr),
        }.items():
            if value:
                if str(item.get(key, "")) != value:
                    item[key] = value
                    changed = True
            elif key in item:
                item.pop(key, None)
                changed = True
        return changed

    @staticmethod
    def _log_tail(text: str, *, max_chars: int = MAX_WORKER_LOG_TAIL_CHARS) -> str:
        if len(text) <= max_chars:
            return text
        return text[-max_chars:]

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
    def _default_blocking_question(reason_code: str) -> str:
        mapping = {
            "waiting_conflict": (
                "Which overlapping lane should finish, be discarded, or be split before this task can proceed?"
            ),
            "clean_exit_no_deliverable": (
                "What concrete branch, commit, or PR should this lane produce before rerunning?"
            ),
            "merge_gate_failed": (
                "Which required verification or acceptance check must pass before approval?"
            ),
            "missing_verification_plan": (
                "Which verification command or acceptance check should be added before rerunning?"
            ),
            "scope_violation": (
                "Which files should stay in scope, or should this lane be split before rerunning?"
            ),
            "worker_exited_without_receipt": (
                "Should this lane be rerun, or recovered manually from the existing worktree?"
            ),
            "worker_no_progress_timeout": (
                "Should this stalled lane be rerun, split, or investigated in its current worktree?"
            ),
            "worker_timeout_with_salvage": (
                "Should the recovered timed-out deliverable be adopted, amended, or rerun before integration?"
            ),
            "worker_timeout_no_deliverable": (
                "Should this timed-out lane be rerun, split, or investigated before retrying?"
            ),
            "worker_crash_with_salvage": (
                "Should the recovered crashed deliverable be adopted, amended, or rerun before integration?"
            ),
            "worker_crash": (
                "Should this crashed lane be rerun, reassigned, or investigated before retrying?"
            ),
            "worker_type_blocked": (
                "Which worker type or capacity issue must be resolved before rerunning this lane?"
            ),
            "work_order_leasing_failed": (
                "What missing environment, resource, or policy input must be resolved first?"
            ),
        }
        return mapping.get(
            reason_code,
            "What human input is required before rerunning this lane?",
        )

    @classmethod
    def _infer_failure_reason(cls, item: dict[str, Any], reason: str) -> str:
        merge_gate = item.get("merge_gate")
        if isinstance(merge_gate, dict):
            missing = str(merge_gate.get("verification_missing_reason", "")).strip()
            if missing:
                return missing
        lowered = str(reason or "").strip().lower()
        if "scope" in lowered and "ownership" in lowered:
            return "scope_violation"
        if "without receipt or exit marker" in lowered:
            return "worker_exited_without_receipt"
        if "no-progress timeout" in lowered:
            return "worker_no_progress_timeout"
        if "recoverable deliverable" in lowered and "timed out" in lowered:
            return "worker_timeout_with_salvage"
        if "recoverable deliverable" in lowered and "non-zero" in lowered:
            return "worker_crash_with_salvage"
        if "timed out before producing a deliverable" in lowered:
            return "worker_timeout_no_deliverable"
        if "crashed before producing a deliverable" in lowered:
            return "worker_crash"
        if "no commits and no changed paths" in lowered or "no real deliverables" in lowered:
            return "clean_exit_no_deliverable"
        if "merge gate" in lowered:
            return "merge_gate_failed"
        if "dispatch blocked" in lowered:
            return "worker_type_blocked"
        return "needs_human"

    @classmethod
    def _mark_needs_human(
        cls,
        item: dict[str, Any],
        reason: str,
        *,
        failure_reason: str | None = None,
        blocking_question: str | None = None,
    ) -> None:
        item["status"] = "needs_human"
        item["dispatch_error"] = reason
        normalized_reason = (
            str(failure_reason or cls._infer_failure_reason(item, reason)).strip() or "needs_human"
        )
        normalized_question = str(
            blocking_question or cls._default_blocking_question(normalized_reason)
        ).strip()
        item["failure_reason"] = normalized_reason
        item["blocking_question"] = normalized_question
        item["blocker"] = {
            "reason": normalized_reason,
            "question": normalized_question,
        }
        blockers = [str(value).strip() for value in item.get("blockers", []) if str(value).strip()]
        if reason not in blockers:
            blockers.append(reason)
        item["blockers"] = blockers
        item.pop("pid", None)

    @classmethod
    def _mark_waiting_conflict(
        cls,
        item: dict[str, Any],
        conflicts: list[dict[str, Any]],
    ) -> None:
        item["status"] = "waiting_conflict"
        item["conflicts"] = list(conflicts)
        item["failure_reason"] = "waiting_conflict"
        item["blocking_question"] = cls._default_blocking_question("waiting_conflict")
        item["blocker"] = {
            "reason": "waiting_conflict",
            "question": item["blocking_question"],
        }
        blockers: list[str] = []
        for conflict in conflicts:
            if not isinstance(conflict, dict):
                continue
            scope = (
                str(conflict.get("path", "")).strip()
                or ", ".join(
                    str(value).strip()
                    for value in (conflict.get("claimed_paths") or [])
                    if str(value).strip()
                )
                or ", ".join(
                    str(value).strip()
                    for value in (conflict.get("allowed_globs") or [])
                    if str(value).strip()
                )
            )
            if not scope:
                continue
            summary = f"scope already claimed: {scope}"
            if summary not in blockers:
                blockers.append(summary)
        if not blockers:
            blockers.append("waiting_conflict")
        item["blockers"] = blockers
        item.pop("dispatch_error", None)
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
        item["failure_reason"] = "scope_violation"
        item["blocking_question"] = (
            "Which files should stay in scope, or should this lane be split before rerunning?"
        )
        item["blocker"] = {
            "reason": "scope_violation",
            "question": item["blocking_question"],
        }
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

    def _llm_adjudicate_scope(
        self,
        item: dict[str, Any],
        violations: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Use LLM to filter false-positive scope violations.

        Returns the reduced list of violations (may be empty if all justified).
        On any failure, returns the original violations unchanged (fail-closed).
        """
        try:
            from aragora.ralph.llm_classifier import LLMBlockerClassifier

            classifier = LLMBlockerClassifier()
            task_desc = str(item.get("task_description", item.get("title", "")))
            declared_scope = [str(s).strip() for s in item.get("file_scope", []) if str(s).strip()]
            changed_paths = [str(p) for p in item.get("changed_paths", [])]

            import asyncio
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                verdict = pool.submit(
                    asyncio.run,
                    classifier.adjudicate_scope(
                        task_description=task_desc,
                        declared_scope=declared_scope,
                        changed_paths=changed_paths,
                        violations=violations,
                    ),
                ).result(timeout=self._LLM_CALL_TIMEOUT)

            if verdict.justified_paths:
                logger.info(
                    "LLM scope adjudicator justified %d paths: %s (%s)",
                    len(verdict.justified_paths),
                    verdict.justified_paths,
                    verdict.reasoning,
                )
            justified_set = set(verdict.justified_paths)
            remaining = [v for v in violations if str(v.get("path", "")) not in justified_set]
            return remaining
        except Exception:
            logger.debug("LLM scope adjudication failed, keeping all violations", exc_info=True)
            return violations

    def _llm_override_merge_gate(
        self,
        item: dict[str, Any],
        merge_gate: dict[str, Any],
    ) -> bool:
        """Ask LLM if merge gate failure is cosmetic or genuine.

        Returns True if the LLM says the deliverable is ready despite the
        gate failure.  Returns False on any error (fail-closed).
        """
        if str(merge_gate.get("verification_missing_reason", "")).strip() == (
            "missing_verification_plan"
        ):
            return False
        try:
            from aragora.ralph.llm_classifier import LLMBlockerClassifier

            classifier = LLMBlockerClassifier()
            acceptance_criteria = [
                str(c).strip() for c in item.get("acceptance_criteria", []) if str(c).strip()
            ]
            verification_results = merge_gate.get("verification_results", [])
            changed_paths = [str(p) for p in item.get("changed_paths", [])]
            diff_summary = str(item.get("diff_summary", ""))[:2000]

            import asyncio
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                verdict = pool.submit(
                    asyncio.run,
                    classifier.evaluate_merge_readiness(
                        acceptance_criteria=acceptance_criteria,
                        verification_results=verification_results,
                        changed_paths=changed_paths,
                        diff_summary=diff_summary,
                    ),
                ).result(timeout=self._LLM_CALL_TIMEOUT)

            logger.info(
                "LLM merge evaluation: ready=%s blocking=%s advisory=%s (%s)",
                verdict.ready,
                verdict.blocking_issues,
                verdict.advisory_issues,
                verdict.reasoning,
            )
            return verdict.ready
        except Exception:
            logger.debug("LLM merge evaluation failed, fail-closed", exc_info=True)
            return False

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
    def _validate_file_scope(file_scope: list[str], worktree_path: str) -> list[str]:
        """Drop file_scope entries whose top-level directory does not exist.

        LLM planners sometimes hallucinate paths (e.g. ``src/orchestrator/``
        when the real code lives at ``aragora/nomic/``).  These entries block
        workers at two layers:

        1. The prompt tells the worker to stay in scope → worker refuses to
           edit real files and exits with zero deliverables.
        2. The supervisor enforces scope on changed paths → any real edits
           are rejected as ``scope_violation``.

        This method strips entries whose first path component (e.g. ``src``)
        does not exist in the worktree, so that both prompt and enforcement
        operate on the real codebase structure.  Valid entries (e.g.
        ``aragora/nomic/foo.py``) pass through unchanged.
        """
        if not file_scope or not worktree_path:
            return file_scope
        wt = Path(worktree_path)
        if not wt.is_dir():
            return file_scope
        # Only validate against real git checkouts (have .git file/dir).
        # Test fixtures create bare directories without .git.
        dot_git = wt / ".git"
        if not dot_git.exists():
            return file_scope
        valid: list[str] = []
        for scope_path in file_scope:
            clean = scope_path.removeprefix("./").strip()
            if not clean:
                continue
            root = clean.split("/")[0]
            if (wt / root).exists():
                valid.append(scope_path)
            else:
                logger.warning(
                    "Dropping hallucinated file_scope entry %r (root %r not found in %s)",
                    scope_path,
                    root,
                    worktree_path,
                )
        return valid

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
