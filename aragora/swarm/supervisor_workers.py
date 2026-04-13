"""Worker lifecycle helpers for the swarm supervisor."""

from __future__ import annotations

import asyncio
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aragora.nomic.dev_coordination import FileScopeViolationError, LeaseConflictError, LeaseStatus
from aragora.swarm import supervisor as _supervisor
from aragora.swarm.lane_telemetry import LaneTelemetryRecord
from aragora.swarm.terminal_truth import qualify_work_order_terminal_state
from aragora.swarm.worker_launcher import WorkerLauncher, WorkerProcess

if TYPE_CHECKING:
    from aragora.swarm.supervisor import SupervisorRun, SwarmApprovalPolicy

UTC = _supervisor.UTC
logger = _supervisor.logger
_LANE_TELEMETRY = _supervisor._LANE_TELEMETRY
DEFAULT_BREAKER_FAILURE_THRESHOLD = _supervisor.DEFAULT_BREAKER_FAILURE_THRESHOLD
DEFAULT_BREAKER_RESET_TIMEOUT_SECONDS = _supervisor.DEFAULT_BREAKER_RESET_TIMEOUT_SECONDS
LAUNCHER_CONFIG_METADATA_KEY = _supervisor.LAUNCHER_CONFIG_METADATA_KEY
SESSION_LOCK_FILES = _supervisor.SESSION_LOCK_FILES
SupervisorRun = _supervisor.SupervisorRun
SwarmApprovalPolicy = _supervisor.SwarmApprovalPolicy
WORKER_TYPE_CIRCUIT_BREAKERS_KEY = _supervisor.WORKER_TYPE_CIRCUIT_BREAKERS_KEY
WORKER_TYPE_CIRCUIT_BREAKER_POLICY_KEY = _supervisor.WORKER_TYPE_CIRCUIT_BREAKER_POLICY_KEY
WorkerOutcome = _supervisor.WorkerOutcome
_strict_bool = _supervisor._strict_bool


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
    except (OSError, RuntimeError, KeyError, subprocess.SubprocessError):
        logger.debug("pre-reap worker collection failed", exc_info=True)

    # Reap dead-session active leases before deriving run status so
    # orphaned leased work orders do not remain "active" indefinitely.
    try:
        stale = self.store.reap_stale_leases()
        if stale:
            logger.info("reaped %d stale leases during refresh_run", len(stale))
    except (OSError, RuntimeError, KeyError, ValueError):
        logger.debug("reap_stale_leases failed during refresh_run", exc_info=True)

    # Proactively reap TTL-expired leases so stale locks don't accumulate
    # when no new claim_lease() calls are attempted (e.g. all work orders
    # stuck in waiting_conflict).
    try:
        reaped = self.store.reap_expired_leases()
        if reaped:
            logger.info("reaped %d expired leases during refresh_run", len(reaped))
    except (OSError, RuntimeError, KeyError, ValueError):
        logger.debug("reap_expired_leases failed during refresh_run", exc_info=True)

    record = self.store.get_supervisor_run(run_id)
    if record is None:
        raise KeyError(f"Unknown supervisor run: {run_id}")

    metadata = dict(record.get("metadata") or {})
    worker_type_circuit_breaker_policy = self._worker_type_circuit_breaker_policy(metadata)
    worker_type_circuit_breakers = self._worker_type_circuit_breakers(metadata)
    self._expire_worker_type_circuit_breakers(worker_type_circuit_breakers)

    max_concurrency = min(max(1, int(metadata.get("max_concurrency", 8))), 8)
    managed_dir_pattern = str(metadata.get("managed_dir_pattern", ".worktrees/{agent}-auto"))
    work_orders = [dict(item) for item in record.get("work_orders", [])]
    for item in work_orders:
        self._backfill_missing_completion_receipt(item)
    self._recover_reaped_needs_human_deliverables(
        work_orders,
        worker_type_circuit_breakers=worker_type_circuit_breakers,
        worker_type_circuit_breaker_policy=worker_type_circuit_breaker_policy,
    )
    self._reconcile_stale_work_order_state(
        work_orders,
        worker_type_circuit_breakers=worker_type_circuit_breakers,
        worker_type_circuit_breaker_policy=worker_type_circuit_breaker_policy,
    )
    for item in work_orders:
        self._sync_dependency_context_metadata(
            item,
            work_orders,
            prompt_ready=str(item.get("status", "")).strip() in {"leased", "dispatched"},
        )
    active_count = sum(
        1 for item in work_orders if str(item.get("status", "")) in {"leased", "dispatched"}
    )

    if active_count < max_concurrency:
        for item in work_orders:
            if active_count >= max_concurrency:
                break
            if str(item.get("status", "queued")) not in {"queued", "waiting_conflict"}:
                continue
            if not self._dependencies_ready_for_dispatch(item, work_orders):
                continue
            try:
                leased = self._lease_work_order(
                    run_id=run_id,
                    target_branch=str(record.get("target_branch", "main")),
                    work_order=item,
                    work_orders=work_orders,
                    managed_dir_pattern=managed_dir_pattern,
                    approval_policy=SwarmApprovalPolicy.from_dict(record.get("approval_policy")),
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
                            work_orders=work_orders,
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
                    self._mark_waiting_resource(item, str(exc))
                    break
                else:
                    self._clear_stale_prelaunch_deliverable_state(item)
                    self._mark_needs_human(
                        item,
                        str(exc),
                        failure_reason="work_order_leasing_failed",
                    )
                    continue

    refreshed = self.store.update_supervisor_run(
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
    work_orders = [dict(item) for item in record.get("work_orders", []) if isinstance(item, dict)]
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
    except (OSError, RuntimeError, subprocess.SubprocessError):
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
        pid = WorkerLauncher._normalized_pid(item.get("pid"))
        if pid is None:
            pass
        else:
            # Check if PID is still alive
            try:
                os.kill(pid, 0)
                continue  # Still running
            except OSError:
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
        prompt_chars=int(item.get("prompt_chars") or 0),
        enriched_context_chars=int(item.get("enriched_context_chars") or 0),
    )


def _recover_commit_backed_terminal_result(
    self,
    item: dict[str, Any],
    *,
    candidate: WorkerProcess | None,
    worktree_path: str,
    initial_head: str,
) -> WorkerProcess | None:
    """Recover a commit-backed terminal result when detached collection is incomplete.

    Detached collection intentionally returns changed-path evidence without
    synthesizing commits when the worker died without a terminal marker. If the
    worktree HEAD actually advanced, rebuild a truthful salvage result from git
    history before classifying the lane as receiptless.
    """
    if candidate is not None and candidate.commit_shas:
        return candidate
    if not worktree_path or not initial_head or not Path(worktree_path).is_dir():
        return None
    recovered = self._build_dead_worker_salvage_result(
        item,
        worktree_path=worktree_path,
        initial_head=initial_head,
    )
    if recovered is None or not recovered.commit_shas:
        return None
    if candidate is not None:
        if candidate.completed_at:
            recovered.completed_at = candidate.completed_at
        if candidate.stdout and not recovered.stdout:
            recovered.stdout = candidate.stdout
        if candidate.stderr and not recovered.stderr:
            recovered.stderr = candidate.stderr
        if candidate.pid is not None:
            recovered.pid = candidate.pid
        if candidate.session_id and not recovered.session_id:
            recovered.session_id = candidate.session_id
        if candidate.lease_id and not recovered.lease_id:
            recovered.lease_id = candidate.lease_id
    return recovered


def _collect_finished_results_before_reap(
    self,
    run_id: str,
    record: dict[str, Any],
) -> None:
    work_orders = [dict(item) for item in record.get("work_orders", []) if isinstance(item, dict)]
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
    except (OSError, RuntimeError, subprocess.SubprocessError):
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


def _reconcile_stale_work_order_state(
    self,
    work_orders: list[dict[str, Any]],
    *,
    worker_type_circuit_breakers: dict[str, dict[str, Any]] | None = None,
    worker_type_circuit_breaker_policy: dict[str, Any] | None = None,
) -> None:
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
        if self._should_requeue_recoverable_work_order_leasing_failed(item, active_leases):
            self._reset_work_order_for_requeue(item)
            continue
        if self._should_requeue_reaped_needs_human(item, active_leases):
            self._reset_work_order_for_requeue(item)
            continue
        if self._should_requeue_ignorable_scope_violation(item):
            self._reset_work_order_for_requeue(item)
            continue
        if self._should_requeue_terminal_dependency_failure(item, work_orders):
            self._reset_work_order_for_requeue(item)
            continue
        if self._should_requeue_conflict_only_needs_human(item):
            self._reset_work_order_for_requeue(item)
            continue
        self._rehabilitate_validation_marker_crash_work_order(
            item,
            worker_type_circuit_breakers=worker_type_circuit_breakers,
            worker_type_circuit_breaker_policy=worker_type_circuit_breaker_policy,
        )


def _recover_reaped_needs_human_deliverables(
    self,
    work_orders: list[dict[str, Any]],
    *,
    worker_type_circuit_breakers: dict[str, dict[str, Any]] | None = None,
    worker_type_circuit_breaker_policy: dict[str, Any] | None = None,
) -> None:
    stale_failure_reasons = {"stale_lease_reaped", "expired_lease_reaped"}
    for item in work_orders:
        if str(item.get("status", "")).strip() != "needs_human":
            continue
        failure_reason = str(item.get("failure_reason", "")).strip().lower()
        if failure_reason not in stale_failure_reasons:
            continue
        if str(item.get("receipt_id") or "").strip():
            continue
        if item.get("commit_shas"):
            continue
        worktree_path = str(item.get("worktree_path", "")).strip()
        initial_head = str(item.get("initial_head", "")).strip()
        if not worktree_path or not initial_head or not Path(worktree_path).is_dir():
            continue
        try:
            result = self._build_dead_worker_salvage_result(
                item,
                worktree_path=worktree_path,
                initial_head=initial_head,
            )
        except (subprocess.TimeoutExpired, OSError):
            logger.debug(
                "stale-reaped salvage check failed for %s",
                item.get("work_order_id"),
                exc_info=True,
            )
            continue
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
                str(blocker).strip() for blocker in item.get("blockers", []) if str(blocker).strip()
            ],
            outcome=deliverable_type,
            risks=[
                str(blocker).strip() for blocker in item.get("blockers", []) if str(blocker).strip()
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
    previous_launcher_config = self._launcher_config_snapshot(self._launcher_config())
    self._apply_launcher_config_snapshot(metadata.get(LAUNCHER_CONFIG_METADATA_KEY))

    try:
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
                item["first_output_at"] = None
                item["last_output_at"] = None
                item["progress_fingerprint"] = {
                    "head_sha": worker.initial_head,
                    "changed_paths": [],
                    "diff_lines": 0,
                }
                item["output_fingerprint"] = {
                    "stdout_size": 0,
                    "stderr_size": 0,
                    "stdout_mtime_ns": 0,
                    "stderr_mtime_ns": 0,
                    "has_output": False,
                }
                item["prompt_chars"] = int(worker.prompt_chars or 0)
                item["enriched_context_chars"] = int(worker.enriched_context_chars or 0)
                # Persist worker PID in lease metadata so reap_stale_leases
                # can detect dead processes even if this supervisor dies.
                lease_id = str(item.get("lease_id", "")).strip()
                if lease_id and worker.pid is not None:
                    try:
                        self.store.update_lease_metadata(lease_id, {"worker_pid": worker.pid})
                    except (OSError, RuntimeError, KeyError, ValueError):
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
                    lease_id = str(item.get("lease_id", "")).strip()
                    if lease_id:
                        self.store.release_lease(lease_id, status=LeaseStatus.RELEASED)
                    self._mark_dispatch_failed(item, str(exc))
                import logging

                logging.getLogger(__name__).warning(
                    "Failed to dispatch %s: %s",
                    item.get("work_order_id"),
                    exc,
                )
    finally:
        self._apply_launcher_config_snapshot(previous_launcher_config)

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
    except (OSError, RuntimeError, subprocess.SubprocessError):
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
        except (OSError, RuntimeError, subprocess.SubprocessError, ValueError):
            logger.debug("Detached result collection failed for %s", woid, exc_info=True)
            result = None
        if result is not None:
            initial_recovered_result: WorkerProcess | None = None
            try:
                initial_recovered_result = _recover_commit_backed_terminal_result(
                    self,
                    item,
                    candidate=result,
                    worktree_path=worktree_path,
                    initial_head=str(item.get("initial_head", "")),
                )
            except (subprocess.TimeoutExpired, OSError):
                logger.debug(
                    "Initial detached salvage reconstruction failed for %s",
                    woid,
                    exc_info=True,
                )
            if initial_recovered_result is not None:
                finished.append(initial_recovered_result)
                finished_ids.add(woid)
                continue

        try:
            progress = await self.launcher.snapshot_progress(item)
        except (OSError, RuntimeError, subprocess.SubprocessError):
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
        output_fingerprint = self._output_fingerprint(progress)
        if output_fingerprint != self._output_fingerprint(item.get("output_fingerprint")):
            item["output_fingerprint"] = output_fingerprint
            if output_fingerprint["has_output"]:
                if not item.get("first_output_at"):
                    item["first_output_at"] = observed_at
                item["last_output_at"] = observed_at
                item["last_progress_at"] = observed_at
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
                        preserve_incomplete_artifacts=False,
                    )
                except (OSError, RuntimeError, subprocess.SubprocessError, ValueError):
                    logger.debug("Detached result collection failed for %s", woid, exc_info=True)

                exit_recovered_result: WorkerProcess | None = None
                try:
                    exit_recovered_result = _recover_commit_backed_terminal_result(
                        self,
                        item,
                        candidate=detached_result,
                        worktree_path=worktree_path,
                        initial_head=str(item.get("initial_head", "")),
                    )
                except (subprocess.TimeoutExpired, OSError):
                    logger.debug(
                        "Dead-worker salvage reconstruction failed for %s",
                        woid,
                        exc_info=True,
                    )

                if exit_recovered_result is not None:
                    finished.append(exit_recovered_result)
                    finished_ids.add(woid)
                    item["worker_outcome"] = WorkerOutcome.CRASH_WITH_SALVAGE.value
                    self._release_terminal_lease(item)
                    changed = True
                    continue

                self._clear_stale_runtime_deliverable_state(item)
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
                f"worker exceeded no-progress timeout ({int(self._no_progress_timeout_seconds())}s)"
            )
            if (
                worktree_path
                and WorkerLauncher._normalized_pid(item.get("pid")) is None
                and WorkerLauncher._active_session_lock_blocks_collection(worktree_path, None)
            ):
                continue
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
                        allow_session_meta_pid_fallback=False,
                        preserve_incomplete_artifacts=False,
                    )
                except (OSError, RuntimeError, subprocess.SubprocessError, ValueError):
                    logger.debug("Timeout result collection failed for %s", woid, exc_info=True)

            recovered_timeout_result: WorkerProcess | None = None
            try:
                recovered_timeout_result = _recover_commit_backed_terminal_result(
                    self,
                    item,
                    candidate=timeout_result,
                    worktree_path=worktree_path,
                    initial_head=str(item.get("initial_head", "")),
                )
            except (subprocess.TimeoutExpired, OSError):
                logger.debug(
                    "Timeout salvage reconstruction failed for %s",
                    woid,
                    exc_info=True,
                )

            if recovered_timeout_result is not None:
                # Worker produced a concrete deliverable before timing out.
                # Surface it through the normal result path.
                finished.append(recovered_timeout_result)
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
                    self._clear_stale_runtime_deliverable_state(item)
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


def _lease_work_order(
    self,
    *,
    run_id: str,
    target_branch: str,
    work_order: dict[str, Any],
    work_orders: list[dict[str, Any]],
    managed_dir_pattern: str,
    approval_policy: SwarmApprovalPolicy,
) -> bool:
    target_agent = str(work_order.get("target_agent", "codex")).strip() or "codex"
    managed_dir = self._managed_dir_for_agent(managed_dir_pattern, target_agent)
    wo_id = str(work_order.get("work_order_id", "task"))
    task_key = f"{run_id}:{wo_id}"
    session_key = f"swarm-{run_id[:8]}-{wo_id}"
    metadata = dict(work_order.get("metadata") or {})
    handoff_key = str(metadata.get("handoff_key", "")).strip()
    journal_lookup: dict[str, str] = {}
    if handoff_key:
        journal_lookup["handoff_key"] = handoff_key
    else:
        # Work-order ids such as "micro-1" are reused across boss-loop issues;
        # only fall back to run-scoped keys when no cross-host handoff key exists.
        journal_lookup["task_key"] = task_key
    persisted_journals = self.store.list_worker_repair_journals(**journal_lookup)
    if persisted_journals:
        metadata["repair_journal"] = [
            dict(record.get("entry") or {}) for record in persisted_journals if record.get("entry")
        ][-3:]
        work_order["metadata"] = metadata
    raw_scope = [str(item) for item in work_order.get("file_scope", []) if str(item).strip()]
    if not raw_scope:
        self._clear_stale_prelaunch_deliverable_state(work_order)
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
        self._clear_stale_prelaunch_deliverable_state(work_order)
        self._mark_needs_human(
            work_order,
            "Declared file scope resolved to no valid in-repo paths; declare scope before dispatch.",
            failure_reason="scope_violation",
        )
        return False
    dependency_base = self._dependency_base_reference(work_order, work_orders)
    if dependency_base is not None:
        dependency_ref, dependency_id = dependency_base
        if not self._reseed_dependent_session_branch(
            session=session,
            work_order=work_order,
            dependency_ref=dependency_ref,
            dependency_id=dependency_id,
        ):
            return False
    else:
        work_order.pop("dependency_base_ref", None)
        work_order.pop("dependency_base_source", None)
    dependency_payload = self._sync_dependency_context_metadata(
        work_order,
        work_orders,
        prompt_ready=True,
    )
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
            "handoff_key": handoff_key,
            "reviewer_agent": str(work_order.get("reviewer_agent", "")),
            "risk_level": str(work_order.get("risk_level", "review")),
            "approval_required": True,
            "repair_journal_count": len(metadata.get("repair_journal", []) or []),
            "dependency_context_ready": bool(dependency_payload["ready_for_dispatch"]),
            "dependency_context_count": len(dependency_payload["contexts"]),
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
                        "blocking_question": str(item.get("blocking_question", "")).strip() or None,
                    },
                )
            )
        except (OSError, RuntimeError, ValueError, TypeError):
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
    except (OSError, RuntimeError, KeyError, ValueError):
        logger.debug("Failed to register PR %s in registry", url, exc_info=True)


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
    metadata = dict(item.get("metadata") or {})
    requested_target_agent = str(metadata.get("requested_target_agent", "")).strip().lower()
    sticky_target_agent = _strict_bool(metadata.get("sticky_target_agent")) is True
    fallback_agent = self._alternate_agent(current_agent)
    if not fallback_agent:
        return False
    if sticky_target_agent and requested_target_agent == current_agent:
        metadata.update(
            {
                "last_failure_reason": reason,
                "last_failure_detail": detail[:1000],
                "fallback_suppressed_reason": "sticky_requested_target_agent",
                "fallback_suppressed_agent": fallback_agent,
            }
        )
        item["metadata"] = metadata
        return False
    if worker_type_circuit_breakers is not None and self._worker_type_circuit_breaker_is_open(
        worker_type_circuit_breakers,
        fallback_agent,
    ):
        return False

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
        "conflicts",
        "dispatched_at",
        "last_observed_at",
        "last_progress_at",
        "first_output_at",
        "last_output_at",
        "progress_fingerprint",
        "output_fingerprint",
        "failure_reason",
        "blocking_question",
        "blocker",
        "resource_error",
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


def _dispatch_failure_reason(exc: Exception) -> str:
    message = str(exc).strip().lower()
    if "cli not found" in message or "not found" in message:
        return "agent_unavailable"
    return "agent_launch_failed"


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
        entry["status"] = str(payload.get("status", entry["status"])).strip().lower() or "closed"
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
        entry["last_failure_detail"] = str(payload.get("last_failure_detail", "")).strip()[:1000]
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


def _normalized_timestamp(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    return text


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

    entry["failure_count"] = max(failure_count, threshold if open_immediately else failure_count)
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
    self._clear_stale_prelaunch_deliverable_state(item)
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
    item.pop("lease_id", None)
    item.pop("owner_session_id", None)


def _mark_dispatch_failed(item: dict[str, Any], reason: str) -> None:
    """Persist a pre-launch failure without carrying stale deliverable state."""
    item["status"] = "dispatch_failed"
    item["review_status"] = "pending"
    item["dispatch_error"] = str(reason)
    for key in (
        "lease_id",
        "owner_session_id",
        "resource_error",
        "conflicts",
        "receipt_id",
        "confidence",
        "worker_outcome",
        "completed_at",
        "exit_code",
        "pid",
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
        "dispatched_at",
        "failure_reason",
        "blocking_question",
        "blocker",
        "last_observed_at",
        "last_progress_at",
        "first_output_at",
        "last_output_at",
        "progress_fingerprint",
        "output_fingerprint",
    ):
        item.pop(key, None)
    item.pop("blockers", None)


def _clear_stale_prelaunch_deliverable_state(item: dict[str, Any]) -> None:
    """Drop stale completion and wait-state metadata before a pre-launch blocker."""
    for key in (
        "dispatch_error",
        "resource_error",
        "failure_reason",
        "blocking_question",
        "blocker",
        "conflicts",
        "receipt_id",
        "confidence",
        "worker_outcome",
        "completed_at",
        "exit_code",
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
    item.pop("blockers", None)


def _clear_stale_runtime_deliverable_state(item: dict[str, Any]) -> None:
    """Drop stale deliverable metadata while preserving runtime log evidence."""
    for key in (
        "receipt_id",
        "confidence",
        "worker_outcome",
        "completed_at",
        "exit_code",
        "head_sha",
        "commit_shas",
        "changed_paths",
        "diff",
        "diff_lines",
        "tests_run",
        "verification_results",
        "merge_gate",
        "verification_missing_reason",
        "pr_url",
        "adopted_pr",
        "resource_error",
        "conflicts",
        "blockers",
        "scope_violation",
    ):
        item.pop(key, None)


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
        pid = WorkerLauncher._normalized_pid(session_meta.get("pid"))
        if pid is None:
            if self._is_managed_worktree(path):
                return "managed_worktree_without_active_session"
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


def _is_resource_constraint_error(exc: Exception) -> bool:
    lowered = str(exc).lower()
    return "no space left on device" in lowered or "disk full" in lowered


def _alternate_agent(agent: str | None) -> str | None:
    value = str(agent or "").strip().lower()
    if value == "claude":
        return "codex"
    if value == "codex":
        return "claude"
    return None


async def _kill_worker(self, item: dict[str, Any]) -> None:
    """Kill a running worker process by PID."""
    import signal

    if "pid" not in item:
        return
    pid = WorkerLauncher._normalized_pid(item.get("pid"))
    if pid is None:
        item.pop("pid", None)
        return
    try:
        import os as _os

        _os.kill(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
    item.pop("pid", None)
