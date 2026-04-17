#!/usr/bin/env python3
"""Run a bounded proof-first unattended shift."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from aragora.nomic.checkpoints import load_latest_checkpoint  # noqa: E402
from aragora.nomic.shift_controller import ShiftConfig, ShiftController, ShiftState  # noqa: E402
from aragora.swarm.shift_ledger import ShiftLedger  # noqa: E402
from scripts.reconcile_proof_first_queue import reconcile_proof_first_queue  # noqa: E402
from scripts.watch_boss_lane import collect_snapshot  # noqa: E402

DEFAULT_REPO = "synaptent/aragora"
DEFAULT_SHIFT_DIRNAME = "proof_first_shift"
DEFAULT_AUTOMATION_BACKLOG_LIMIT = 12
DEFAULT_REFRESH_MINUTES = 120
DEFAULT_MAX_HOURS = 12.0
DEFAULT_INTERVAL_SECONDS = 120
DEFAULT_BOSS_LABEL = "com.aragora.swarm-boss-loop"
DEFAULT_MERGE_LABEL = "com.aragora.swarm-merge-arbiter"
DEFAULT_BOSS_PROCESS_PATTERN = r"aragora\.cli\.main swarm boss-loop|run_boss_cycle\.sh"
DEFAULT_MERGE_PROCESS_PATTERN = r"aragora\.cli\.main swarm merge-arbiter"
DEFAULT_LAUNCHD_START_TIMEOUT_SECONDS = 45.0
LAUNCHD_THROTTLE_GRACE_SECONDS = 60.0
AUTH_FAILURE_STOP_AFTER = 2
PUBLICATION_FAILURE_STOP_AFTER = 2
RUNTIME_FAILURE_STOP_AFTER = 1
SERVICE_FAILURE_STOP_AFTER = 1
FAILURE_POLICIES: dict[str, dict[str, Any]] = {
    "auth_failure": {
        "stop_after": AUTH_FAILURE_STOP_AFTER,
        "self_heal": "retry_benchmark_once",
        "description": "Allow one benchmark auth failure, then fail closed on the next one.",
    },
    "publication_failure": {
        "stop_after": PUBLICATION_FAILURE_STOP_AFTER,
        "self_heal": "retry_benchmark_once",
        "description": "Allow one benchmark publication handoff failure, then fail closed.",
    },
    "rate_limit": {
        "stop_after": PUBLICATION_FAILURE_STOP_AFTER,
        "self_heal": "retry_benchmark_once",
        "description": "Allow one benchmark rate limit failure, then fail closed.",
    },
    "permission_mismatch": {
        "stop_after": PUBLICATION_FAILURE_STOP_AFTER,
        "self_heal": "retry_benchmark_once",
        "description": "Allow one benchmark permission mismatch, then fail closed.",
    },
    "runtime_failure": {
        "stop_after": RUNTIME_FAILURE_STOP_AFTER,
        "self_heal": "none",
        "description": "Stop immediately when the shift runtime loses truthful control surfaces.",
    },
    "service_failure": {
        "stop_after": SERVICE_FAILURE_STOP_AFTER,
        "self_heal": "restart_service_once",
        "description": "Attempt one service restart when work is pending, then fail closed.",
    },
}
AUTOMATION_BRANCH_PREFIXES = (
    "codex/",
    "factory/",
    "aragora/boss-harvest/",
    "benchmark-truth-publication/",
)
RECOVERY_BUDGET_PER_FAILURE_CLASS = 1
AUTH_DRIFT_FAILURE = "auth_drift"
GITHUB_OUTAGE_FAILURE = "github_outage"
PUBLICATION_FAILURE = "benchmark_publication_failure"
RATE_LIMIT_FAILURE = "rate_limit"
PERMISSION_MISMATCH_FAILURE = "permission_mismatch"
BOSS_RESTART_FAILURE = "boss_restart_failure"
MERGE_RESTART_FAILURE = "merge_restart_failure"
RECOVERY_FAILURE_CLASSES = (
    AUTH_DRIFT_FAILURE,
    GITHUB_OUTAGE_FAILURE,
    PUBLICATION_FAILURE,
    RATE_LIMIT_FAILURE,
    PERMISSION_MISMATCH_FAILURE,
    BOSS_RESTART_FAILURE,
    MERGE_RESTART_FAILURE,
)
RECOVERY_STOP_REASONS = {
    AUTH_DRIFT_FAILURE: "RepeatedAuthFailure: benchmark publication auth drift persisted after one automatic recovery attempt",
    GITHUB_OUTAGE_FAILURE: "RepeatedGitHubOutage: proof-first queue reconciliation could not reach GitHub after one automatic recovery attempt",
    PUBLICATION_FAILURE: "RepeatedPublicationFailure: benchmark publication PR handoff persisted after one automatic recovery attempt",
    RATE_LIMIT_FAILURE: "RepeatedRateLimitFailure: benchmark publication rate limit persisted after one automatic recovery attempt",
    PERMISSION_MISMATCH_FAILURE: "RepeatedPermissionMismatch: benchmark publication permission mismatch persisted after one automatic recovery attempt",
    BOSS_RESTART_FAILURE: "RepeatedBossRestartFailure: boss loop still required intervention after one automatic recovery attempt",
    MERGE_RESTART_FAILURE: "RepeatedMergeArbiterRestartFailure: merge arbiter still required intervention after one automatic recovery attempt",
}


@dataclass
class ProofFirstRuntimeState:
    recovery_shift_id: str | None = None
    boss_restart_count: int = 0
    merge_restart_count: int = 0
    auth_failure_count: int = 0
    publication_failure_count: int = 0
    rate_limit_failure_count: int = 0
    permission_mismatch_count: int = 0
    runtime_failure_count: int = 0
    github_outage_count: int = 0
    recovery_attempt_counts: dict[str, int] = field(default_factory=dict)
    last_benchmark_run_id: int | None = None
    last_triggered_benchmark_run_id: int | None = None


@dataclass(frozen=True)
class LaunchdServiceStatus:
    state: str = ""
    minimum_runtime_seconds: int | None = None
    detail: str = ""


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _assessment_id() -> str:
    return f"proof-first-{_utc_now().strftime('%Y%m%dT%H%M%SZ')}"


def _runtime_state_path(repo_root: Path) -> Path:
    return repo_root / ".aragora" / DEFAULT_SHIFT_DIRNAME / "runtime_state.json"


def _default_recovery_attempt_counts() -> dict[str, int]:
    return dict.fromkeys(RECOVERY_FAILURE_CLASSES, 0)


def _normalize_recovery_attempt_counts(payload: Any) -> dict[str, int]:
    counts = _default_recovery_attempt_counts()
    if isinstance(payload, dict):
        for failure_class in RECOVERY_FAILURE_CLASSES:
            counts[failure_class] = int(payload.get(failure_class, 0) or 0)
    return counts


def load_runtime_state(path: Path) -> ProofFirstRuntimeState:
    if not path.exists():
        return ProofFirstRuntimeState()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return ProofFirstRuntimeState()
    if not isinstance(payload, dict):
        return ProofFirstRuntimeState()
    try:
        return ProofFirstRuntimeState(
            recovery_shift_id=(
                str(payload["recovery_shift_id"]) if payload.get("recovery_shift_id") else None
            ),
            boss_restart_count=int(payload.get("boss_restart_count", 0) or 0),
            merge_restart_count=int(payload.get("merge_restart_count", 0) or 0),
            auth_failure_count=int(payload.get("auth_failure_count", 0) or 0),
            publication_failure_count=int(payload.get("publication_failure_count", 0) or 0),
            rate_limit_failure_count=int(payload.get("rate_limit_failure_count", 0) or 0),
            permission_mismatch_count=int(payload.get("permission_mismatch_count", 0) or 0),
            runtime_failure_count=int(payload.get("runtime_failure_count", 0) or 0),
            github_outage_count=int(payload.get("github_outage_count", 0) or 0),
            recovery_attempt_counts=_normalize_recovery_attempt_counts(
                payload.get("recovery_attempt_counts")
            ),
            last_benchmark_run_id=(
                int(payload["last_benchmark_run_id"])
                if payload.get("last_benchmark_run_id")
                else None
            ),
            last_triggered_benchmark_run_id=(
                int(payload["last_triggered_benchmark_run_id"])
                if payload.get("last_triggered_benchmark_run_id")
                else None
            ),
        )
    except (TypeError, ValueError):
        return ProofFirstRuntimeState()


def save_runtime_state(path: Path, state: ProofFirstRuntimeState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(state), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def reset_recovery_budget_state(state: ProofFirstRuntimeState) -> None:
    state.boss_restart_count = 0
    state.merge_restart_count = 0
    state.auth_failure_count = 0
    state.publication_failure_count = 0
    state.rate_limit_failure_count = 0
    state.permission_mismatch_count = 0
    state.runtime_failure_count = 0
    state.github_outage_count = 0
    state.recovery_attempt_counts = _default_recovery_attempt_counts()


def bind_runtime_state_to_shift(state: ProofFirstRuntimeState, shift_id: str) -> None:
    if state.recovery_shift_id == shift_id:
        return
    reset_recovery_budget_state(state)
    state.recovery_shift_id = shift_id


def recovery_budget_remaining(runtime_state: ProofFirstRuntimeState, failure_class: str) -> int:
    used = int(runtime_state.recovery_attempt_counts.get(failure_class, 0) or 0)
    return max(0, RECOVERY_BUDGET_PER_FAILURE_CLASS - used)


def consume_recovery_attempt(
    runtime_state: ProofFirstRuntimeState,
    *,
    failure_class: str,
) -> tuple[int, int]:
    attempt_number = int(runtime_state.recovery_attempt_counts.get(failure_class, 0) or 0) + 1
    runtime_state.recovery_attempt_counts[failure_class] = attempt_number
    remaining_budget = max(0, RECOVERY_BUDGET_PER_FAILURE_CLASS - attempt_number)
    return attempt_number, remaining_budget


def record_recovery_attempt(
    ledger: ShiftLedger | None,
    runtime_state: ProofFirstRuntimeState,
    *,
    failure_class: str,
    action: str,
    success: bool,
    detail: str = "",
) -> None:
    attempt_number, remaining_budget = consume_recovery_attempt(
        runtime_state,
        failure_class=failure_class,
    )
    if ledger:
        ledger.append(
            "recovery_attempt",
            failure_class=failure_class,
            action=action,
            success=success,
            detail=detail,
            attempt_number=attempt_number,
            budget_limit=RECOVERY_BUDGET_PER_FAILURE_CLASS,
            remaining_budget=remaining_budget,
        )


def failure_budget_summary(
    runtime_state: ProofFirstRuntimeState,
) -> dict[str, dict[str, int | bool]]:
    summary: dict[str, dict[str, int | bool]] = {}
    for failure_class in RECOVERY_FAILURE_CLASSES:
        attempts_used = int(runtime_state.recovery_attempt_counts.get(failure_class, 0) or 0)
        summary[failure_class] = {
            "budget": RECOVERY_BUDGET_PER_FAILURE_CLASS,
            "attempts_used": attempts_used,
            "remaining": max(0, RECOVERY_BUDGET_PER_FAILURE_CLASS - attempts_used),
            "exhausted": attempts_used >= RECOVERY_BUDGET_PER_FAILURE_CLASS,
        }
    return summary


def _run(
    args: list[str],
    *,
    cwd: Path,
    timeout: int = 30,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=cwd,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def _run_json(args: list[str], *, cwd: Path, timeout: int = 30) -> Any:
    proc = _run(args, cwd=cwd, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(
            proc.stderr.strip() or proc.stdout.strip() or f"command failed: {' '.join(args)}"
        )
    return json.loads(proc.stdout or "{}")


def restore_shift_controller(
    controller: ShiftController,
    *,
    checkpoint_dir: Path,
) -> ShiftState | None:
    payload = load_latest_checkpoint(str(checkpoint_dir))
    if not isinstance(payload, dict):
        return None
    shift_payload = payload.get("shift_state")
    if not isinstance(shift_payload, dict):
        return None
    state = ShiftState.from_dict(shift_payload)
    controller._state = state  # noqa: SLF001
    return state


def collect_boss_lane_snapshot(*, repo_root: Path, repo: str) -> dict[str, Any]:
    args = argparse.Namespace(
        repo_root=str(repo_root),
        repo=repo,
        watch_session=[],
        watch_pr=[],
        pr_search_issue=0,
        queue_limit=12,
        stale_minutes=30,
    )
    return collect_snapshot(args)


def list_open_prs(*, repo_root: Path, repo: str) -> list[dict[str, Any]]:
    payload = _run_json(
        [
            "gh",
            "pr",
            "list",
            "--repo",
            repo,
            "--state",
            "open",
            "--limit",
            "100",
            "--json",
            "number,title,headRefName,isDraft,url",
        ],
        cwd=repo_root,
        timeout=45,
    )
    if not isinstance(payload, list):
        raise RuntimeError("gh pr list returned a non-list payload")
    return [item for item in payload if isinstance(item, dict)]


def count_automation_backlog(prs: list[dict[str, Any]]) -> int:
    return sum(
        1
        for pr in prs
        if not bool(pr.get("isDraft"))
        and str(pr.get("headRefName") or "").startswith(AUTOMATION_BRANCH_PREFIXES)
    )


def actionable_open_prs(prs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [pr for pr in prs if not bool(pr.get("isDraft"))]


def has_open_benchmark_publication_pr(prs: list[dict[str, Any]]) -> bool:
    return any(
        str(pr.get("headRefName") or "").startswith("benchmark-truth-publication/") for pr in prs
    )


def latest_benchmark_run(*, repo_root: Path, repo: str) -> dict[str, Any] | None:
    payload = _run_json(
        [
            "gh",
            "run",
            "list",
            "--repo",
            repo,
            "--workflow",
            "Benchmark Truth Publication",
            "--limit",
            "1",
            "--json",
            "databaseId,createdAt,status,conclusion",
        ],
        cwd=repo_root,
        timeout=45,
    )
    if not isinstance(payload, list) or not payload:
        return None
    run = payload[0]
    return run if isinstance(run, dict) else None


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(UTC)


def should_trigger_benchmark_rerun(
    *,
    benchmark_mode: str,
    latest_run: dict[str, Any] | None,
    has_open_publication_pr: bool,
    automation_backlog: int,
    automation_backlog_limit: int,
    last_triggered_run_id: int | None,
    max_age_hours: float = 24.0,
) -> tuple[bool, str]:
    if benchmark_mode != "hybrid":
        return False, "benchmark_mode_disabled"
    if has_open_publication_pr:
        return False, "benchmark_pr_open"
    if automation_backlog >= automation_backlog_limit:
        return False, "automation_backlog_full"
    if latest_run is None:
        if int(last_triggered_run_id or 0) == -1:
            return False, "awaiting_first_run_visibility"
        return True, "no_prior_run"

    run_id = int(latest_run.get("databaseId", 0) or 0)
    created_at = _parse_timestamp(str(latest_run.get("createdAt") or ""))
    if created_at is not None:
        age_hours = (_utc_now() - created_at).total_seconds() / 3600.0
        if age_hours >= max_age_hours:
            if run_id and run_id == int(last_triggered_run_id or 0):
                return False, "awaiting_new_benchmark_run"
            return True, "stale_publication_window"

    conclusion = str(latest_run.get("conclusion") or "").strip().lower()
    if conclusion == "failure" and run_id != int(last_triggered_run_id or 0):
        return True, "retry_after_failed_run"
    return False, "fresh_enough"


def classify_benchmark_failure_log(text: str) -> str:
    lowered = str(text or "").lower()
    if any(
        token in lowered
        for token in (
            "authentication",
            "auth required",
            "codex login",
            "not logged in",
            "api token",
        )
    ):
        return "auth_failure"
    if any(
        token in lowered
        for token in (
            "resource not accessible by integration",
            "not permitted to create pull requests",
            "pr_creation_forbidden",
            "pull request creation is not allowed",
        )
    ):
        return "publication_failure"
    if any(
        token in lowered
        for token in (
            "api rate limit exceeded",
            "http 429",
            "rate limit",
            "rate-limited",
            "secondary rate limit",
            "too many requests",
        )
    ):
        return RATE_LIMIT_FAILURE
    if any(
        token in lowered
        for token in (
            "permission denied",
            "permission mismatch",
            "insufficient permission",
            "insufficient permissions",
            "requires write permission",
            "workflow permission",
        )
    ):
        return PERMISSION_MISMATCH_FAILURE
    return "other_failure"


def github_unavailable_stop_reason(queue_report: dict[str, Any]) -> str:
    github_status = dict(queue_report.get("github_status") or {})
    detail = str(github_status.get("error") or "").strip()
    if detail:
        return f"GitHubUnavailable: {detail}"
    return "GitHubUnavailable: proof-first queue reconciliation could not reach GitHub"


def build_failure_policy_status(
    runtime_state: ProofFirstRuntimeState,
    *,
    service_failure_count: int = 0,
) -> dict[str, dict[str, Any]]:
    counts = {
        "auth_failure": int(runtime_state.auth_failure_count or 0),
        "publication_failure": int(runtime_state.publication_failure_count or 0),
        "rate_limit": int(runtime_state.rate_limit_failure_count or 0),
        "permission_mismatch": int(runtime_state.permission_mismatch_count or 0),
        "runtime_failure": int(runtime_state.runtime_failure_count or 0),
        "service_failure": int(service_failure_count or 0),
    }
    status: dict[str, dict[str, Any]] = {}
    for failure_type, policy in FAILURE_POLICIES.items():
        stop_after = int(policy["stop_after"])
        count = counts[failure_type]
        status[failure_type] = {
            "count": count,
            "stop_after": stop_after,
            "remaining_self_heal_attempts": max(0, stop_after - count - 1),
            "self_heal": str(policy["self_heal"]),
            "description": str(policy["description"]),
            "will_stop": count >= stop_after,
        }
    return status


def fetch_benchmark_failure_log(*, repo_root: Path, run_id: int) -> str:
    proc = _run(
        ["gh", "run", "view", str(run_id), "--log-failed"],
        cwd=repo_root,
        timeout=60,
    )
    return proc.stdout or proc.stderr or ""


def process_running(pattern: str) -> bool:
    proc = subprocess.run(
        ["pgrep", "-f", pattern],
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    return proc.returncode == 0 and bool(proc.stdout.strip())


def inspect_launchd_service(label: str) -> LaunchdServiceStatus:
    """Inspect launchd state without failing the shift when launchctl is unavailable."""
    uid = os.getuid()
    try:
        proc = subprocess.run(
            ["launchctl", "print", f"gui/{uid}/{label}"],
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return LaunchdServiceStatus(detail=f"launchctl print timed out for {label}")
    detail = "\n".join(part.strip() for part in (proc.stdout, proc.stderr) if part.strip())
    if proc.returncode != 0:
        return LaunchdServiceStatus(detail=detail or f"launchctl print failed for {label}")

    state = ""
    minimum_runtime_seconds: int | None = None
    for raw_line in detail.splitlines():
        line = raw_line.strip()
        if line.startswith("state = ") and not state:
            state = line.removeprefix("state = ").strip()
        elif line.startswith("minimum runtime = "):
            value = line.removeprefix("minimum runtime = ").strip()
            if value.isdigit():
                minimum_runtime_seconds = int(value)
    return LaunchdServiceStatus(
        state=state,
        minimum_runtime_seconds=minimum_runtime_seconds,
        detail=detail,
    )


def launchd_start_timeout_seconds(
    status: LaunchdServiceStatus,
    *,
    fallback_seconds: float = DEFAULT_LAUNCHD_START_TIMEOUT_SECONDS,
) -> float:
    """Return a process wait long enough for launchd throttle-backed spawns."""
    if status.state == "spawn scheduled" and status.minimum_runtime_seconds:
        return max(
            fallback_seconds,
            float(status.minimum_runtime_seconds) + LAUNCHD_THROTTLE_GRACE_SECONDS,
        )
    return fallback_seconds


def wait_for_process(
    pattern: str, *, timeout_seconds: float = 45.0, interval_seconds: float = 1.0
) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while True:
        if process_running(pattern):
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(interval_seconds)


def should_restart_service(
    *,
    is_running: bool,
    pending_count: int,
    restart_count: int,
    restart_limit: int = 1,
) -> bool:
    return (not is_running) and pending_count > 0 and restart_count < restart_limit


def kickstart_launchd(label: str) -> tuple[bool, str]:
    uid = os.getuid()
    try:
        proc = subprocess.run(
            ["launchctl", "kickstart", f"gui/{uid}/{label}"],
            text=True,
            capture_output=True,
            timeout=30,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.strip() if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr.strip() if isinstance(exc.stderr, str) else ""
        detail = stderr or stdout or f"launchctl kickstart timed out for {label}"
        return False, detail
    if proc.returncode != 0:
        return (
            False,
            proc.stderr.strip() or proc.stdout.strip() or f"launchctl kickstart failed for {label}",
        )
    return True, ""


def restart_service_via_launchd(
    *,
    label: str,
    process_pattern: str,
    start_timeout_seconds: float | None = None,
) -> tuple[bool, str]:
    initial_status = (
        inspect_launchd_service(label) if start_timeout_seconds is None else LaunchdServiceStatus()
    )
    wait_timeout_seconds = (
        launchd_start_timeout_seconds(initial_status)
        if start_timeout_seconds is None
        else start_timeout_seconds
    )
    kicked, detail = kickstart_launchd(label)
    if start_timeout_seconds is None:
        current_status = inspect_launchd_service(label)
        wait_timeout_seconds = max(
            wait_timeout_seconds,
            launchd_start_timeout_seconds(current_status),
        )
    if wait_for_process(process_pattern, timeout_seconds=wait_timeout_seconds):
        return True, "" if kicked else detail
    state_detail = _read_launchd_failure_detail(label)
    if kicked:
        message = (
            f"launchctl kickstart returned success for {label}, but process pattern "
            f"{process_pattern!r} is still not running"
        )
        if state_detail:
            message = f"{message}; {state_detail}"
        return False, message
    if state_detail:
        return False, f"{detail}; {state_detail}"
    return False, detail


def _read_launchd_failure_detail(label: str) -> str:
    """Return a redacted, truthful state summary for ``label`` after a stuck restart."""

    try:
        from scripts.probe_boss_loop_launchd import read_launchd_state
    except Exception:
        return ""
    state = read_launchd_state(label, timeout_seconds=5.0)
    if state is None:
        return f"launchd state unavailable for {label}"
    parts = [f"launchd state={state.state!r}", f"runs={state.runs}"]
    if state.last_exit_code is not None:
        parts.append(f"last_exit={state.last_exit_code}")
    if state.minimum_runtime_seconds is not None:
        parts.append(f"min_runtime={state.minimum_runtime_seconds}s")
    if state.is_spawn_scheduled:
        parts.append("hint=ThrottleInterval may not have elapsed")
    return ", ".join(parts)


def run_merge_arbiter_apply(
    *,
    repo_root: Path,
    repo: str,
    limit: int,
) -> dict[str, Any]:
    payload = _run_json(
        [
            "python3",
            "scripts/merge_codex_automation_prs.py",
            "--root",
            str(repo_root),
            "--repo",
            repo,
            "--limit",
            str(limit),
            "--apply",
            "--json",
        ],
        cwd=repo_root,
        timeout=120,
    )
    if not isinstance(payload, dict):
        raise RuntimeError("merge arbiter helper returned a non-object payload")
    return payload


def trigger_benchmark_workflow(*, repo_root: Path, repo: str) -> None:
    proc = _run(
        ["gh", "workflow", "run", "Benchmark Truth Publication", "--repo", repo],
        cwd=repo_root,
        timeout=45,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or "gh workflow run failed")


def benchmark_fresh_within_window(
    latest_run: dict[str, Any] | None,
    *,
    max_age_hours: float,
) -> bool:
    if latest_run is None:
        return False
    conclusion = str(latest_run.get("conclusion") or "").strip().lower()
    if conclusion != "success":
        return False
    created_at = _parse_timestamp(str(latest_run.get("createdAt") or ""))
    if created_at is None:
        return False
    age_hours = (_utc_now() - created_at).total_seconds() / 3600.0
    return age_hours < max_age_hours


def evaluate_green_shift(
    *,
    queue_report: dict[str, Any],
    queue_count: int,
    open_pr_count: int,
    boss_running: bool | None,
    merge_running: bool | None,
    latest_run: dict[str, Any] | None,
    benchmark_drift_open: bool,
    runtime_state: ProofFirstRuntimeState,
    max_age_hours: float,
    stop_reason: str,
    repeated_failure_classes: list[str],
) -> dict[str, Any]:
    queue_canonical_or_empty = True
    if dict(queue_report.get("github_status") or {}).get("available") is False:
        queue_canonical_or_empty = False

    boss_ok = bool(boss_running) or queue_count == 0
    merge_ok = bool(merge_running) or open_pr_count == 0
    criteria = {
        "benchmark_fresh_within_window": benchmark_fresh_within_window(
            latest_run,
            max_age_hours=max_age_hours,
        ),
        "queue_canonical_or_empty": queue_canonical_or_empty,
        "services_healthy_or_intentionally_idle": boss_ok and merge_ok,
        "failure_budgets_respected": not repeated_failure_classes,
        "no_open_benchmark_publication_drift": not benchmark_drift_open,
    }
    blocking_reasons = [name for name, passed in criteria.items() if not passed]
    if stop_reason:
        blocking_reasons.append("shift_stop_reason_present")
    return {
        "is_green": all(criteria.values()) and not stop_reason,
        "criteria": criteria,
        "blocking_reasons": blocking_reasons,
        "repeated_failure_classes": repeated_failure_classes,
        "recovery_budgets": failure_budget_summary(runtime_state),
    }


def run_shift_cycle(
    *,
    repo_root: Path,
    repo: str,
    benchmark_mode: str,
    automation_backlog_limit: int,
    runtime_state: ProofFirstRuntimeState,
    ledger: ShiftLedger | None = None,
    max_hours: float = DEFAULT_MAX_HOURS,
) -> dict[str, Any]:
    queue_report = reconcile_proof_first_queue(repo=repo, repo_root=repo_root, apply=True)
    github_status = dict(queue_report.get("github_status") or {})
    queue_removed_count = len(queue_report.get("removed") or [])
    if github_status.get("available") is False:
        detail = github_unavailable_stop_reason(queue_report)
        runtime_state.github_outage_count += 1
        if ledger:
            ledger.record_failure(failure_type=GITHUB_OUTAGE_FAILURE, detail=detail)
        actions: list[str] = []
        stop_reason = ""
        repeated_failure_classes: list[str] = []
        if recovery_budget_remaining(runtime_state, GITHUB_OUTAGE_FAILURE) > 0:
            record_recovery_attempt(
                ledger,
                runtime_state,
                failure_class=GITHUB_OUTAGE_FAILURE,
                action="retry_queue_reconciliation_next_cycle",
                success=False,
                detail=detail,
            )
            actions.append("retry_github_outage_next_cycle")
        else:
            stop_reason = RECOVERY_STOP_REASONS[GITHUB_OUTAGE_FAILURE]
            repeated_failure_classes.append(GITHUB_OUTAGE_FAILURE)
        failure_policy = build_failure_policy_status(runtime_state)
        if ledger:
            ledger.record_cycle_tick(
                queue_size=len(queue_report.get("kept") or []),
                queue_removed=queue_removed_count,
                open_prs=0,
                boss_running=False,
                merge_running=False,
                benchmark_fresh=False,
                actions=actions,
                stop_reason=stop_reason,
            )
        return {
            "queue_report": queue_report,
            "snapshot": {},
            "open_pr_count": 0,
            "automation_backlog": 0,
            "latest_benchmark_run": None,
            "actions": actions,
            "stop_reason": stop_reason,
            "failure_policy": failure_policy,
            "green_shift_evaluation": evaluate_green_shift(
                queue_report=queue_report,
                queue_count=0,
                open_pr_count=0,
                boss_running=None,
                merge_running=None,
                latest_run=None,
                benchmark_drift_open=True,
                runtime_state=runtime_state,
                max_age_hours=max_hours,
                stop_reason=stop_reason,
                repeated_failure_classes=repeated_failure_classes,
            ),
        }
    snapshot = collect_boss_lane_snapshot(repo_root=repo_root, repo=repo)
    prs = list_open_prs(repo_root=repo_root, repo=repo)
    actionable_prs = actionable_open_prs(prs)
    automation_backlog = count_automation_backlog(actionable_prs)
    open_pr_count = len(actionable_prs)
    draft_pr_count = len(prs) - open_pr_count
    canonical_queue_count = len(queue_report["kept"])

    boss_running = process_running(DEFAULT_BOSS_PROCESS_PATTERN)
    merge_running = process_running(DEFAULT_MERGE_PROCESS_PATTERN)

    actions: list[str] = []
    service_failures: list[str] = []
    runtime_failures: list[str] = []
    repeated_failure_classes: list[str] = []
    if should_restart_service(
        is_running=boss_running,
        pending_count=canonical_queue_count,
        restart_count=runtime_state.boss_restart_count,
    ):
        restarted, detail = restart_service_via_launchd(
            label=DEFAULT_BOSS_LABEL,
            process_pattern=DEFAULT_BOSS_PROCESS_PATTERN,
        )
        runtime_state.boss_restart_count += 1
        record_recovery_attempt(
            ledger,
            runtime_state,
            failure_class=BOSS_RESTART_FAILURE,
            action="restart_boss_loop",
            success=restarted,
            detail=detail,
        )
        if restarted:
            boss_running = True
            actions.append("restart_boss_loop")
            if ledger:
                ledger.record_service_restart(service="boss_loop", success=True)
        else:
            actions.append("restart_boss_loop_failed")
            stop_reason = f"BossRestartFailed: {detail or 'boss loop restart did not produce a running process'}"
            service_failures.append(stop_reason)
            if ledger:
                ledger.record_service_restart(service="boss_loop", success=False, detail=detail)
                ledger.record_failure(failure_type="service_failure", detail=stop_reason)
    elif (not boss_running) and canonical_queue_count > 0 and runtime_state.boss_restart_count >= 1:
        repeated_failure_classes.append(BOSS_RESTART_FAILURE)
        service_failures.append(RECOVERY_STOP_REASONS[BOSS_RESTART_FAILURE])

    if should_restart_service(
        is_running=merge_running,
        pending_count=open_pr_count,
        restart_count=runtime_state.merge_restart_count,
    ):
        restarted, detail = restart_service_via_launchd(
            label=DEFAULT_MERGE_LABEL,
            process_pattern=DEFAULT_MERGE_PROCESS_PATTERN,
        )
        runtime_state.merge_restart_count += 1
        record_recovery_attempt(
            ledger,
            runtime_state,
            failure_class=MERGE_RESTART_FAILURE,
            action="restart_merge_arbiter",
            success=restarted,
            detail=detail,
        )
        if restarted:
            merge_running = True
            actions.append("restart_merge_arbiter")
            if ledger:
                ledger.record_service_restart(service="merge_arbiter", success=True)
        else:
            actions.append("restart_merge_arbiter_failed")
            stop_reason = f"MergeArbiterRestartFailed: {detail or 'merge arbiter restart did not produce a running process'}"
            service_failures.append(stop_reason)
            if ledger:
                ledger.record_service_restart(service="merge_arbiter", success=False, detail=detail)
                ledger.record_failure(failure_type="service_failure", detail=stop_reason)
    elif (not merge_running) and open_pr_count > 0 and runtime_state.merge_restart_count >= 1:
        repeated_failure_classes.append(MERGE_RESTART_FAILURE)
        service_failures.append(RECOVERY_STOP_REASONS[MERGE_RESTART_FAILURE])

    merge_report = run_merge_arbiter_apply(
        repo_root=repo_root,
        repo=repo,
        limit=automation_backlog_limit,
    )
    merged_numbers = list(merge_report.get("merged") or [])
    if merged_numbers:
        actions.append(f"merged_prs:{','.join(str(number) for number in merged_numbers)}")
        if ledger:
            for num in merged_numbers:
                ledger.record_pr_merged(pr_number=int(num))

    latest_run = latest_benchmark_run(repo_root=repo_root, repo=repo)
    latest_failure_class = ""
    latest_failure_detail = ""
    if latest_run is not None:
        run_id = int(latest_run.get("databaseId", 0) or 0)
        conclusion = str(latest_run.get("conclusion") or "").strip().lower()
        if run_id and run_id != int(runtime_state.last_benchmark_run_id or 0):
            runtime_state.last_benchmark_run_id = run_id
            if ledger:
                ledger.record_benchmark_run(
                    run_id=run_id,
                    conclusion=conclusion,
                    created_at=str(latest_run.get("createdAt", "")),
                )
            if conclusion == "failure":
                failure_log = fetch_benchmark_failure_log(repo_root=repo_root, run_id=run_id)
                failure_class = classify_benchmark_failure_log(failure_log)
                if failure_class == "auth_failure":
                    runtime_state.auth_failure_count += 1
                    latest_failure_class = AUTH_DRIFT_FAILURE
                    latest_failure_detail = failure_log[:500]
                    if ledger:
                        ledger.record_failure(failure_type="auth_failure", detail=failure_log[:500])
                elif failure_class == "publication_failure":
                    runtime_state.publication_failure_count += 1
                    latest_failure_class = PUBLICATION_FAILURE
                    latest_failure_detail = failure_log[:500]
                    if ledger:
                        ledger.record_failure(
                            failure_type="publication_failure", detail=failure_log[:500]
                        )
                elif failure_class == RATE_LIMIT_FAILURE:
                    runtime_state.rate_limit_failure_count += 1
                    latest_failure_class = RATE_LIMIT_FAILURE
                    latest_failure_detail = failure_log[:500]
                    if ledger:
                        ledger.record_failure(
                            failure_type=RATE_LIMIT_FAILURE, detail=failure_log[:500]
                        )
                elif failure_class == PERMISSION_MISMATCH_FAILURE:
                    runtime_state.permission_mismatch_count += 1
                    latest_failure_class = PERMISSION_MISMATCH_FAILURE
                    latest_failure_detail = failure_log[:500]
                    if ledger:
                        ledger.record_failure(
                            failure_type=PERMISSION_MISMATCH_FAILURE,
                            detail=failure_log[:500],
                        )
                else:
                    runtime_state.runtime_failure_count += 1
                    runtime_failures.append(
                        "RuntimeFailure: benchmark publication failed with an unclassified runtime error"
                    )
                    if ledger:
                        ledger.record_failure(
                            failure_type="runtime_failure",
                            detail=failure_log[:500]
                            or "benchmark publication failed with an unclassified runtime error",
                        )
            elif conclusion == "success":
                runtime_state.auth_failure_count = 0
                runtime_state.publication_failure_count = 0
                runtime_state.rate_limit_failure_count = 0
                runtime_state.permission_mismatch_count = 0
                runtime_state.runtime_failure_count = 0

    should_trigger, trigger_reason = should_trigger_benchmark_rerun(
        benchmark_mode=benchmark_mode,
        latest_run=latest_run,
        has_open_publication_pr=has_open_benchmark_publication_pr(prs),
        automation_backlog=automation_backlog,
        automation_backlog_limit=automation_backlog_limit,
        last_triggered_run_id=runtime_state.last_triggered_benchmark_run_id,
        max_age_hours=max_hours,
    )
    stop_reason = ""
    if should_trigger:
        if trigger_reason == "retry_after_failed_run" and latest_failure_class:
            if recovery_budget_remaining(runtime_state, latest_failure_class) <= 0:
                repeated_failure_classes.append(latest_failure_class)
                stop_reason = RECOVERY_STOP_REASONS[latest_failure_class]
            else:
                try:
                    trigger_benchmark_workflow(repo_root=repo_root, repo=repo)
                except RuntimeError as exc:
                    detail = str(exc).strip() or "benchmark workflow trigger failed"
                    runtime_state.runtime_failure_count += 1
                    runtime_failures.append(f"RuntimeFailure: {detail}")
                    actions.append(f"trigger_benchmark_failed:{trigger_reason}")
                    record_recovery_attempt(
                        ledger,
                        runtime_state,
                        failure_class=latest_failure_class,
                        action="trigger_benchmark_workflow",
                        success=False,
                        detail=detail,
                    )
                    if ledger:
                        ledger.record_failure(failure_type="runtime_failure", detail=detail)
                else:
                    actions.append(f"trigger_benchmark:{trigger_reason}")
                    record_recovery_attempt(
                        ledger,
                        runtime_state,
                        failure_class=latest_failure_class,
                        action="trigger_benchmark_workflow",
                        success=True,
                        detail=latest_failure_detail or trigger_reason,
                    )
                    if latest_run is not None:
                        runtime_state.last_triggered_benchmark_run_id = int(
                            latest_run.get("databaseId", 0) or 0
                        )
                    else:
                        runtime_state.last_triggered_benchmark_run_id = -1
        else:
            try:
                trigger_benchmark_workflow(repo_root=repo_root, repo=repo)
            except RuntimeError as exc:
                detail = str(exc).strip() or "benchmark workflow trigger failed"
                runtime_state.runtime_failure_count += 1
                runtime_failures.append(f"RuntimeFailure: {detail}")
                actions.append(f"trigger_benchmark_failed:{trigger_reason}")
                if ledger:
                    ledger.record_failure(failure_type="runtime_failure", detail=detail)
            else:
                actions.append(f"trigger_benchmark:{trigger_reason}")
                if latest_run is not None:
                    runtime_state.last_triggered_benchmark_run_id = int(
                        latest_run.get("databaseId", 0) or 0
                    )
                else:
                    runtime_state.last_triggered_benchmark_run_id = -1

    if not stop_reason and runtime_failures:
        stop_reason = runtime_failures[0]
    elif not stop_reason and service_failures:
        stop_reason = service_failures[0]
    elif not stop_reason and runtime_state.auth_failure_count >= AUTH_FAILURE_STOP_AFTER:
        stop_reason = "RepeatedAuthFailure: benchmark publication failed auth twice"
    elif (
        not stop_reason
        and runtime_state.publication_failure_count >= PUBLICATION_FAILURE_STOP_AFTER
    ):
        stop_reason = "RepeatedPublicationFailure: benchmark publication failed PR handoff twice"
    elif (
        not stop_reason and runtime_state.rate_limit_failure_count >= PUBLICATION_FAILURE_STOP_AFTER
    ):
        stop_reason = RECOVERY_STOP_REASONS[RATE_LIMIT_FAILURE]
        repeated_failure_classes.append(RATE_LIMIT_FAILURE)
    elif (
        not stop_reason
        and runtime_state.permission_mismatch_count >= PUBLICATION_FAILURE_STOP_AFTER
    ):
        stop_reason = RECOVERY_STOP_REASONS[PERMISSION_MISMATCH_FAILURE]
        repeated_failure_classes.append(PERMISSION_MISMATCH_FAILURE)
    elif not stop_reason and runtime_state.runtime_failure_count >= RUNTIME_FAILURE_STOP_AFTER:
        stop_reason = "RuntimeFailure: proof-first shift lost a required runtime control surface"

    failure_policy = build_failure_policy_status(
        runtime_state,
        service_failure_count=len(service_failures),
    )

    # Record cycle tick in ledger
    benchmark_fresh = benchmark_fresh_within_window(latest_run, max_age_hours=max_hours)
    if ledger:
        ledger.record_cycle_tick(
            queue_size=canonical_queue_count,
            queue_removed=queue_removed_count,
            open_prs=open_pr_count,
            boss_running=boss_running,
            merge_running=merge_running,
            benchmark_fresh=benchmark_fresh,
            actions=actions,
            stop_reason=stop_reason,
        )

    benchmark_drift_open = has_open_benchmark_publication_pr(prs) or should_trigger

    return {
        "queue_report": queue_report,
        "snapshot": snapshot,
        "open_pr_count": open_pr_count,
        "draft_pr_count": draft_pr_count,
        "total_open_pr_count": len(prs),
        "automation_backlog": automation_backlog,
        "latest_benchmark_run": latest_run,
        "actions": actions,
        "stop_reason": stop_reason,
        "failure_policy": failure_policy,
        "green_shift_evaluation": evaluate_green_shift(
            queue_report=queue_report,
            queue_count=canonical_queue_count,
            open_pr_count=open_pr_count,
            boss_running=boss_running,
            merge_running=merge_running,
            latest_run=latest_run,
            benchmark_drift_open=benchmark_drift_open,
            runtime_state=runtime_state,
            max_age_hours=max_hours,
            stop_reason=stop_reason,
            repeated_failure_classes=repeated_failure_classes,
        ),
    }


def build_cycle_objective(report: dict[str, Any]) -> str:
    queue_kept = len(report["queue_report"]["kept"])
    queue_removed = len(report["queue_report"]["removed"])
    actions = ",".join(report.get("actions") or []) or "steady_state"
    return (
        "proof-first shift tick: "
        f"queue_kept={queue_kept} queue_removed={queue_removed} "
        f"open_prs={int(report.get('open_pr_count', 0) or 0)} "
        f"automation_backlog={int(report.get('automation_backlog', 0) or 0)} "
        f"actions={actions}"
    )


async def _run_shift(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(args.repo_root).resolve()
    checkpoint_dir = repo_root / ".aragora" / DEFAULT_SHIFT_DIRNAME / "checkpoints"
    runtime_state_path = _runtime_state_path(repo_root)
    runtime_state = load_runtime_state(runtime_state_path)
    ledger = ShiftLedger(path=repo_root / ".aragora" / DEFAULT_SHIFT_DIRNAME / "shift_ledger.jsonl")
    shift_id = _assessment_id()
    shift_start_time = time.monotonic()

    controller = ShiftController(
        ShiftConfig(
            max_duration_hours=float(args.max_hours),
            refresh_interval_minutes=float(args.refresh_minutes),
            budget_limit_usd=9999.0,
            max_cycles=10000,
            require_fresh_assessment=False,
            repo_path=str(repo_root),
            checkpoint_dir=str(checkpoint_dir),
        )
    )
    restored = restore_shift_controller(controller, checkpoint_dir=checkpoint_dir)
    if restored is None:
        await controller.start_shift(assessment_id=shift_id)
    elif restored.status == "paused_for_refresh":
        await controller.resume_after_refresh(shift_id)
    elif restored.status != "running":
        await controller.start_shift(assessment_id=shift_id)

    active_shift_id = controller.state.shift_id if controller.state is not None else shift_id
    bind_runtime_state_to_shift(runtime_state, active_shift_id)
    save_runtime_state(runtime_state_path, runtime_state)

    # Record shift start in ledger
    ledger.record_shift_start(
        shift_id=shift_id,
        max_hours=float(args.max_hours),
        benchmark_mode=args.benchmark_mode,
        queue_size=0,  # will be updated on first tick
    )

    cycle_reports: list[dict[str, Any]] = []
    cycle_count = 0
    while True:
        should_stop, reason = controller.check_should_stop()
        if should_stop:
            if reason.startswith("RefreshDue"):
                await controller.pause_for_refresh(reason)
                await controller.resume_after_refresh(_assessment_id())
            else:
                controller.complete_shift(reason)
                ledger.record_shift_stop(
                    shift_id=shift_id,
                    reason=reason,
                    cycles=cycle_count,
                    duration_seconds=time.monotonic() - shift_start_time,
                )
                break

        report = run_shift_cycle(
            repo_root=repo_root,
            repo=args.repo,
            benchmark_mode=args.benchmark_mode,
            automation_backlog_limit=int(args.automation_backlog_limit),
            runtime_state=runtime_state,
            ledger=ledger,
            max_hours=float(args.max_hours),
        )
        cycle_reports.append(report)
        cycle_count += 1
        save_runtime_state(runtime_state_path, runtime_state)

        objective = build_cycle_objective(report)
        await controller.run_cycle(objective)
        if report["stop_reason"]:
            controller.complete_shift(str(report["stop_reason"]))
            ledger.record_shift_stop(
                shift_id=shift_id,
                reason=str(report["stop_reason"]),
                cycles=cycle_count,
                duration_seconds=time.monotonic() - shift_start_time,
            )
            break
        if args.once:
            controller.complete_shift("completed")
            ledger.record_shift_stop(
                shift_id=shift_id,
                reason="completed",
                cycles=cycle_count,
                duration_seconds=time.monotonic() - shift_start_time,
            )
            break
        await asyncio.sleep(float(args.interval_seconds))

    summary = {
        "shift": controller.get_progress_summary(),
        "runtime_state": asdict(runtime_state),
        "cycles": cycle_reports,
        "ledger_status": ledger.get_status_summary(max_age_hours=float(args.max_hours)),
        "green_shift_evaluation": (
            cycle_reports[-1]["green_shift_evaluation"]
            if cycle_reports
            else evaluate_green_shift(
                queue_report={"kept": [], "removed": []},
                queue_count=0,
                open_pr_count=0,
                boss_running=None,
                merge_running=None,
                latest_run=None,
                benchmark_drift_open=True,
                runtime_state=runtime_state,
                max_age_hours=float(args.max_hours),
                stop_reason=str(controller.get_progress_summary().get("stop_reason") or ""),
                repeated_failure_classes=[],
            )
        ),
        "recovery_budgets": failure_budget_summary(runtime_state),
    }
    save_runtime_state(runtime_state_path, runtime_state)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a bounded proof-first unattended shift.")
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--max-hours", type=float, default=DEFAULT_MAX_HOURS)
    parser.add_argument("--refresh-minutes", type=float, default=DEFAULT_REFRESH_MINUTES)
    parser.add_argument("--benchmark-mode", default="hybrid", choices=["hybrid", "disabled"])
    parser.add_argument("--queue-source", default="canonical", choices=["canonical"])
    parser.add_argument(
        "--automation-backlog-limit",
        type=int,
        default=DEFAULT_AUTOMATION_BACKLOG_LIMIT,
    )
    parser.add_argument("--interval-seconds", type=float, default=DEFAULT_INTERVAL_SECONDS)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    summary = asyncio.run(_run_shift(args))
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        shift = summary["shift"]
        print(
            f"shift_id={shift.get('shift_id')} status={shift.get('status')} "
            f"cycles={shift.get('cycles_completed')} stop_reason={shift.get('stop_reason')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
