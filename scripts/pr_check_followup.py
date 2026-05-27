#!/usr/bin/env python3
"""Diagnose PR check state and generate a bounded follow-up prompt.

The helper is intentionally read-only by default: it shells out to ``gh`` for
current PR state and run/job diagnostics, but never reruns jobs, pushes, or
edits files. When ``--wait-run`` is provided, it polls one Actions run until it
completes or times out, then switches the generated prompt to failed-job
diagnosis or narrow current-head cancelled-job rerun commands.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


INCREMENTAL_PROGRESS_SENTENCE = (
    "If the prompt above accomplishes no incremental progress make the next prompt one that "
    "does, include this sentence in all subsequent prompts to ensure they converge towards "
    "prompts that make incremental progress."
)
META_AUTOMATION_SENTENCE = (
    "If any of the work involved can be better automated and made more autonomous by "
    "improving aragora tooling at a meta level create a plan to do that instead, include "
    "this sentence in all subsequent prompts to ensure they converge towards automating "
    "and making more autonomous all steps at a meta level where this makes sense for the "
    "repo and project."
)

FAILURE_CONCLUSIONS = {"FAILURE", "TIMED_OUT", "ACTION_REQUIRED"}
GREEN_CONCLUSIONS = {"SUCCESS", "SKIPPED", "NEUTRAL"}
PENDING_STATUSES = {"IN_PROGRESS", "QUEUED", "PENDING", "EXPECTED", "REQUESTED", "WAITING"}
GREEN_BUCKETS = {"pass", "skipping"}
PENDING_BUCKETS = {"pending"}
FAILURE_BUCKETS = {"fail"}
CANCEL_BUCKETS = {"cancel"}
CHECKOUT_MARKERS = ("checkout", "sparse-checkout", "repository checkout")
SUBSTANTIVE_MARKERS = (
    "verify",
    "build",
    "test",
    "smoke",
    "drift",
    "consistency",
    "quorum",
    "readiness",
)
MODEL_QUORUM_MARKERS = (
    "needs_model_review_quorum",
    "collect_model_quorum_before_merge",
    "model quorum incomplete",
    "focused adversarial dogfood evidence is required",
)


@dataclass
class CheckDiagnosis:
    """Normalized status for a PR check row or Actions job."""

    workflow: str
    name: str
    status: str
    conclusion: str
    classification: str
    details_url: str = ""
    run_id: str | None = None
    job_id: str | None = None
    run_head_sha: str | None = None
    summary: str = ""
    rerun_command: str | None = None
    log_command: str | None = None
    log_summary: list[str] = field(default_factory=list)


@dataclass
class CheckRow:
    """Normalized ``gh pr checks`` row."""

    workflow: str
    name: str
    state: str
    bucket: str
    link: str = ""
    run_id: str | None = None
    job_id: str | None = None


@dataclass
class MergePacketSummary:
    """Small merge-packet view needed by the check watcher."""

    available: bool
    admin_squash_allowed: bool
    not_ready: list[Any]
    head_sha: str | None = None
    status: str = ""
    verdict: str = ""


@dataclass
class SettlementGuard:
    """Fail-closed settlement-readiness decision from direct checks and packet."""

    direct_required_checks_green: bool
    merge_packet_authorizes: bool
    required_non_green: list[CheckRow]
    full_rollup_pending: list[CheckRow]
    full_rollup_non_green: list[CheckRow]
    stale_merge_quorum: CheckRow | None
    safe_stale_merge_quorum_rerun: bool
    blocker: str


@dataclass
class RerunAttempt:
    """Result of the explicitly requested stale-quorum rerun."""

    requested: bool
    executed: bool
    command: str | None = None
    returncode: int | None = None
    stdout: str = ""
    stderr: str = ""
    reason: str = ""


@dataclass
class WaitRunDiagnosis:
    """Summary for an optionally waited GitHub Actions run."""

    run_id: str
    status: str
    conclusion: str
    workflow: str
    head_sha: str
    timed_out: bool
    jobs: list[CheckDiagnosis]


@dataclass
class FollowupResult:
    """Machine-readable PR follow-up decision."""

    pr: int
    head: str
    mergeable: str
    merge_state_status: str
    expected_head: str | None
    action: str
    checks: list[CheckDiagnosis]
    rerun_commands: list[str]
    prompt: str
    wait_run: WaitRunDiagnosis | None = None
    required_checks: list[CheckRow] = field(default_factory=list)
    full_rollup_checks: list[CheckRow] = field(default_factory=list)
    merge_packet_summary: MergePacketSummary | None = None
    settlement_guard: SettlementGuard | None = None
    rerun_attempt: RerunAttempt | None = None


def parse_run_job_ids(details_url: str) -> tuple[str | None, str | None]:
    """Extract GitHub Actions run/job ids from a details URL."""
    match = re.search(r"/actions/runs/(\d+)/job/(\d+)", details_url or "")
    if not match:
        return None, None
    return match.group(1), match.group(2)


def check_row_from_gh(row: dict[str, Any]) -> CheckRow:
    """Normalize one ``gh pr checks --json`` row."""
    link = str(row.get("link") or row.get("detailsUrl") or "").strip()
    run_id, job_id = parse_run_job_ids(link)
    return CheckRow(
        workflow=str(row.get("workflow") or row.get("workflowName") or "").strip(),
        name=str(row.get("name") or row.get("context") or "").strip(),
        state=str(row.get("state") or row.get("status") or row.get("conclusion") or "").strip(),
        bucket=str(row.get("bucket") or "").strip().lower(),
        link=link,
        run_id=run_id,
        job_id=job_id,
    )


def _check_row_identity(row: CheckRow) -> str:
    workflow = row.workflow.strip().lower()
    name = row.name.strip().lower()
    return f"{workflow}:{name}" if workflow else name


def _is_row_green(row: CheckRow) -> bool:
    state = row.state.strip().upper()
    return row.bucket in GREEN_BUCKETS or state in GREEN_CONCLUSIONS


def _is_row_pending(row: CheckRow) -> bool:
    return row.bucket in PENDING_BUCKETS or row.state.strip().upper() in PENDING_STATUSES


def _is_row_failure(row: CheckRow) -> bool:
    state = row.state.strip().upper()
    return row.bucket in FAILURE_BUCKETS or state in FAILURE_CONCLUSIONS


def _is_row_cancelled(row: CheckRow) -> bool:
    return row.bucket in CANCEL_BUCKETS or row.state.strip().upper() == "CANCELLED"


def _is_merge_quorum_row(row: CheckRow) -> bool:
    identity = _check_row_identity(row)
    return "merge quorum" in identity or "merge-quorum" in identity


def _rerun_command_for_row(row: CheckRow) -> str | None:
    if not row.run_id:
        return None
    if row.job_id:
        return f"gh run rerun {row.run_id} --job {row.job_id}"
    return f"gh run rerun {row.run_id}"


def _merge_packet_entry(packet: dict[str, Any], pr_number: int) -> dict[str, Any]:
    entries = packet.get("entries")
    if isinstance(entries, list):
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if int(entry.get("pr_number") or entry.get("number") or 0) == pr_number:
                return entry
        if len(entries) == 1 and isinstance(entries[0], dict):
            return entries[0]
    return packet if int(packet.get("pr_number") or packet.get("number") or 0) == pr_number else {}


def summarize_merge_packet(packet: dict[str, Any] | None, pr_number: int) -> MergePacketSummary:
    """Extract the small packet view needed for guarded settlement prompts."""
    if not packet:
        return MergePacketSummary(
            available=False,
            admin_squash_allowed=False,
            not_ready=[],
            verdict="missing_merge_packet",
        )
    entry = _merge_packet_entry(packet, pr_number)
    raw_not_ready = packet.get("not_ready")
    not_ready: list[Any] = raw_not_ready if isinstance(raw_not_ready, list) else []
    entry_allowed = bool(entry.get("admin_squash_allowed"))
    top_allowed = bool(packet.get("admin_squash_allowed"))
    admin_order = packet.get("admin_squash_order")
    order_allows = isinstance(admin_order, list) and pr_number in admin_order
    return MergePacketSummary(
        available=True,
        admin_squash_allowed=(entry_allowed or top_allowed or order_allows) and not not_ready,
        not_ready=not_ready,
        head_sha=str(entry.get("head_sha") or entry.get("headRefOid") or "") or None,
        status=str(entry.get("status") or ""),
        verdict=str(entry.get("verdict") or ""),
    )


def check_identity(check: dict[str, Any]) -> str:
    """Return a stable check identity for latest-row collapse."""
    workflow = str(check.get("workflowName") or check.get("workflow") or "").strip()
    name = str(check.get("name") or check.get("context") or "").strip()
    if not name:
        return ""
    return f"{workflow}:{name}" if workflow else name


def latest_status_check_rollup(checks: list[Any]) -> list[dict[str, Any]]:
    """Collapse superseded check rows to the latest row per workflow/job identity."""
    latest: dict[str, tuple[str, int, dict[str, Any]]] = {}
    passthrough: list[dict[str, Any]] = []
    for index, check in enumerate(checks):
        if not isinstance(check, dict):
            continue
        identity = check_identity(check)
        if not identity:
            passthrough.append(check)
            continue
        timestamp = str(
            check.get("completedAt")
            or check.get("startedAt")
            or check.get("createdAt")
            or check.get("updatedAt")
            or ""
        )
        previous = latest.get(identity)
        if previous is None or (timestamp, index) >= (previous[0], previous[1]):
            latest[identity] = (timestamp, index, check)
    return passthrough + [item[2] for item in sorted(latest.values(), key=lambda item: item[1])]


def _parse_datetime(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def _duration_seconds(check: dict[str, Any]) -> float | None:
    started = _parse_datetime(str(check.get("startedAt") or ""))
    completed = _parse_datetime(str(check.get("completedAt") or ""))
    if not started or not completed:
        return None
    return max(0.0, (completed - started).total_seconds())


def classify_check(
    check: dict[str, Any], pr_head: str, run_data: dict[str, Any] | None = None
) -> CheckDiagnosis:
    """Classify a statusCheckRollup row into operator-relevant buckets."""
    workflow = str(check.get("workflowName") or check.get("workflow") or "").strip()
    name = str(check.get("name") or check.get("context") or "").strip()
    status = str(check.get("status") or check.get("state") or "").upper()
    conclusion = str(check.get("conclusion") or "").upper()
    if not conclusion and status in FAILURE_CONCLUSIONS | GREEN_CONCLUSIONS | {
        "CANCELLED",
        "STALE",
    }:
        conclusion = status

    details_url = str(check.get("detailsUrl") or check.get("link") or "").strip()
    run_id, job_id = parse_run_job_ids(details_url)
    run_head = str((run_data or {}).get("headSha") or "").strip() or None
    classification = "unknown"
    summary = ""

    if conclusion == "CANCELLED":
        if run_head and run_head != pr_head:
            classification = "stale_cancelled"
            summary = f"cancelled on stale head {run_head}"
        else:
            duration = _duration_seconds(check)
            if _is_early_cancelled_job(run_data, job_id) or (
                duration is not None and duration <= 120
            ):
                classification = "early_cancelled"
                summary = "cancelled before substantive verification"
            else:
                classification = "unknown"
                summary = "cancelled after job startup; inspect log before rerun"
    elif conclusion in FAILURE_CONCLUSIONS:
        classification = "real_failure"
        summary = (
            _failure_after_checkout_summary(_job_steps(run_data, job_id))
            or "current-head failure requiring repair"
        )
    elif status in PENDING_STATUSES or not conclusion:
        classification = "in_progress"
        summary = "check still running or pending"
    elif conclusion in GREEN_CONCLUSIONS:
        classification = "green"
        summary = "green or green-equivalent"
    elif conclusion == "STALE":
        classification = "stale_cancelled"
        summary = "stale status context"

    rerun_command = f"gh run rerun {run_id} --job {job_id}" if run_id and job_id else None
    log_command = f"gh run view {run_id} --job {job_id} --log" if run_id and job_id else None
    return CheckDiagnosis(
        workflow=workflow,
        name=name,
        status=status,
        conclusion=conclusion,
        classification=classification,
        details_url=details_url,
        run_id=run_id,
        job_id=job_id,
        run_head_sha=run_head,
        summary=summary,
        rerun_command=rerun_command,
        log_command=log_command,
    )


def _job_id(job: dict[str, Any]) -> str:
    return str(job.get("databaseId") or job.get("id") or "").strip()


def _step_names_before_first_cancel(steps: list[dict[str, Any]]) -> tuple[list[str], bool]:
    names = [str(step.get("name") or "").lower() for step in steps]
    first_cancelled_index = next(
        (
            index
            for index, step in enumerate(steps)
            if str(step.get("conclusion") or "").upper() == "CANCELLED"
        ),
        None,
    )
    if first_cancelled_index is None:
        return names, False
    return names[: first_cancelled_index + 1], True


def _is_early_cancelled_steps(steps: list[dict[str, Any]]) -> bool:
    if not steps:
        return False
    before_cancel, had_cancel = _step_names_before_first_cancel(steps)
    if not had_cancel:
        return False
    return any(
        any(marker in name for marker in CHECKOUT_MARKERS) for name in before_cancel
    ) and not any(
        any(marker in name for marker in SUBSTANTIVE_MARKERS) for name in before_cancel[:-1]
    )


def _is_early_cancelled_job(run_data: dict[str, Any] | None, job_id: str | None) -> bool:
    if not run_data or not job_id:
        return False
    for job in run_data.get("jobs") or []:
        if not isinstance(job, dict):
            continue
        database_id = _job_id(job)
        if database_id and database_id != job_id:
            continue
        steps = [step for step in job.get("steps") or [] if isinstance(step, dict)]
        return _is_early_cancelled_steps(steps)
    return False


def _job_steps(run_data: dict[str, Any] | None, job_id: str | None) -> list[dict[str, Any]]:
    if not run_data or not job_id:
        return []
    for job in run_data.get("jobs") or []:
        if not isinstance(job, dict):
            continue
        database_id = _job_id(job)
        if database_id and database_id != job_id:
            continue
        return [step for step in job.get("steps") or [] if isinstance(step, dict)]
    return []


def _failure_after_checkout_summary(steps: list[dict[str, Any]]) -> str | None:
    """Return the first failing substantive step after checkout, if available."""
    checkout_index: int | None = None
    for index, step in enumerate(steps):
        name = str(step.get("name") or "").lower()
        conclusion = str(step.get("conclusion") or "").upper()
        if checkout_index is None and any(marker in name for marker in CHECKOUT_MARKERS):
            if conclusion == "SUCCESS":
                checkout_index = index
            continue
        if checkout_index is None or index <= checkout_index:
            continue
        if conclusion in FAILURE_CONCLUSIONS or conclusion == "FAILURE":
            step_name = str(step.get("name") or "substantive step").strip()
            return f"failed after checkout during {step_name}"
    return None


def _is_merge_quorum_check(check: CheckDiagnosis) -> bool:
    workflow = check.workflow.lower()
    name = check.name.lower()
    return "merge quorum" in workflow or "merge-quorum" in workflow or "merge-quorum" in name


def _has_model_quorum_blocker(check: CheckDiagnosis) -> bool:
    if check.classification != "real_failure" or not _is_merge_quorum_check(check):
        return False
    text = "\n".join([check.summary, *check.log_summary]).lower()
    return any(marker in text for marker in MODEL_QUORUM_MARKERS)


def _wait_check_selector_parts(selector: str) -> tuple[str, str] | None:
    """Parse a workflow/name selector into normalized parts."""
    parts = [part.strip().lower() for part in selector.split("/", maxsplit=1)]
    if len(parts) != 2 or not parts[0] or not parts[1]:
        return None
    return parts[0], parts[1]


def wait_check_run_id(pr_data: dict[str, Any], selector: str) -> str | None:
    """Return the run id for a matching in-progress status check, if any."""
    parsed = _wait_check_selector_parts(selector)
    if parsed is None:
        return None
    expected_workflow, expected_name = parsed
    pr_head = str(pr_data.get("headRefOid") or "").strip()
    for check in latest_status_check_rollup(list(pr_data.get("statusCheckRollup") or [])):
        workflow = str(check.get("workflowName") or check.get("workflow") or "").strip().lower()
        name = str(check.get("name") or check.get("context") or "").strip().lower()
        if workflow != expected_workflow or name != expected_name:
            continue
        diagnosis = classify_check(check, pr_head)
        if diagnosis.classification == "in_progress":
            return diagnosis.run_id
        return None
    return None


def _has_branch_conflict(pr_data: dict[str, Any]) -> bool:
    mergeable = str(pr_data.get("mergeable") or "").upper()
    merge_state = str(pr_data.get("mergeStateStatus") or "").upper()
    return mergeable == "CONFLICTING" or merge_state == "DIRTY"


def classify_run_job(
    job: dict[str, Any],
    run_id: str,
    run_data: dict[str, Any],
    *,
    pr_head: str | None = None,
) -> CheckDiagnosis:
    """Classify an Actions job from gh run view JSON."""
    job_id = _job_id(job)
    status = str(job.get("status") or "").upper()
    conclusion = str(job.get("conclusion") or "").upper()
    workflow = str(run_data.get("workflowName") or "").strip()
    name = str(job.get("name") or job_id or "").strip()
    head_sha = str(run_data.get("headSha") or "").strip() or None
    classification = "unknown"
    summary = ""

    if pr_head and head_sha and head_sha != pr_head:
        classification = "stale_cancelled"
        summary = f"run belongs to stale head {head_sha}; current PR head is {pr_head}"
    elif conclusion == "CANCELLED":
        if _is_early_cancelled_steps(
            [step for step in job.get("steps") or [] if isinstance(step, dict)]
        ):
            classification = "early_cancelled"
            summary = "cancelled before substantive verification"
        else:
            classification = "unknown"
            summary = "cancelled after job startup; inspect log before rerun"
    elif conclusion in FAILURE_CONCLUSIONS:
        classification = "real_failure"
        summary = (
            _failure_after_checkout_summary(
                [step for step in job.get("steps") or [] if isinstance(step, dict)]
            )
            or "current-head failure requiring repair"
        )
    elif status in PENDING_STATUSES or not conclusion:
        classification = "in_progress"
        summary = "job still running or pending"
    elif conclusion in GREEN_CONCLUSIONS:
        classification = "green"
        summary = "green or green-equivalent"

    rerun_command = f"gh run rerun {run_id} --job {job_id}" if job_id else None
    log_command = f"gh run view {run_id} --job {job_id} --log" if job_id else None
    return CheckDiagnosis(
        workflow=workflow,
        name=name,
        status=status,
        conclusion=conclusion,
        classification=classification,
        run_id=run_id,
        job_id=job_id or None,
        run_head_sha=head_sha,
        summary=summary,
        rerun_command=rerun_command,
        log_command=log_command,
    )


def summarize_log(log_text: str, max_lines: int = 12) -> list[str]:
    """Extract high-signal failure lines without dumping complete logs."""
    markers = (
        "##[error]",
        "FAIL:",
        "FAILED",
        "Out of date:",
        " E ",
        "Error:",
        "error:",
        "verdict=",
        "status=needs_model_review_quorum",
        "model quorum incomplete",
        "focused adversarial dogfood evidence is required",
    )
    lines: list[str] = []
    for raw in log_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if any(marker in line for marker in markers):
            lines.append(line)
    return lines[-max_lines:]


def diagnose_wait_run(
    run_id: str,
    run_data: dict[str, Any],
    *,
    pr_head: str | None = None,
    timed_out: bool = False,
    log_summary_by_job: dict[str, list[str]] | None = None,
) -> WaitRunDiagnosis:
    """Build a diagnosis for a waited Actions run."""
    jobs = [
        classify_run_job(job, run_id, run_data, pr_head=pr_head)
        for job in run_data.get("jobs") or []
        if isinstance(job, dict)
    ]
    for diagnosis in jobs:
        if diagnosis.job_id and log_summary_by_job:
            diagnosis.log_summary = log_summary_by_job.get(diagnosis.job_id, [])
    return WaitRunDiagnosis(
        run_id=run_id,
        status=str(run_data.get("status") or "").lower(),
        conclusion=str(run_data.get("conclusion") or "").lower(),
        workflow=str(run_data.get("workflowName") or ""),
        head_sha=str(run_data.get("headSha") or ""),
        timed_out=timed_out,
        jobs=jobs,
    )


def _derive_action(
    checks: list[CheckDiagnosis],
    wait_run: WaitRunDiagnosis | None,
    *,
    expected_head: str | None,
    head: str,
    branch_conflict: bool = False,
) -> str:
    if expected_head and head != expected_head:
        return "head_drift"
    if wait_run and wait_run.head_sha and head and wait_run.head_sha != head:
        return "stale_wait_run"

    wait_jobs = wait_run.jobs if wait_run else []
    if wait_run and (
        wait_run.timed_out or any(job.classification == "in_progress" for job in wait_jobs)
    ):
        return "monitor"
    if any(job.classification == "real_failure" for job in wait_jobs):
        if all(
            job.classification != "real_failure" or _has_model_quorum_blocker(job)
            for job in wait_jobs
        ):
            return "collect_model_evidence"
        return "repair_failures"
    if any(job.classification == "unknown" for job in wait_jobs):
        return "diagnose_unknown"
    if (
        wait_jobs
        and any(job.classification == "early_cancelled" for job in wait_jobs)
        and not any(job.classification == "real_failure" for job in wait_jobs)
    ):
        return "rerun_cancelled"

    if any(check.classification == "real_failure" for check in checks):
        if all(
            check.classification != "real_failure" or _has_model_quorum_blocker(check)
            for check in checks
        ):
            return "collect_model_evidence"
        return "repair_failures"
    if any(check.classification == "in_progress" for check in checks):
        return "monitor"
    if any(check.classification == "unknown" for check in checks):
        return "diagnose_unknown"
    if any(check.classification == "early_cancelled" for check in checks):
        return "rerun_cancelled"
    if branch_conflict:
        return "diagnose_branch_conflict"
    return "green"


def build_settlement_guard(
    *,
    required_checks: list[CheckRow],
    full_rollup_checks: list[CheckRow],
    merge_packet_summary: MergePacketSummary,
) -> SettlementGuard:
    """Build the fail-closed settlement guard from direct checks and packet."""
    required_non_green = [row for row in required_checks if not _is_row_green(row)]
    full_rollup_pending = [row for row in full_rollup_checks if _is_row_pending(row)]
    full_rollup_non_green = [
        row for row in full_rollup_checks if not _is_row_green(row) and not _is_row_pending(row)
    ]
    quorum_candidates = [
        row for row in required_non_green if _is_merge_quorum_row(row) and _is_row_cancelled(row)
    ]
    stale_merge_quorum = quorum_candidates[0] if len(quorum_candidates) == 1 else None
    only_required_blocker_is_quorum = (
        stale_merge_quorum is not None and len(required_non_green) == 1
    )
    full_non_green_without_quorum = [
        row
        for row in full_rollup_non_green
        if not (_is_merge_quorum_row(row) and _is_row_cancelled(row))
    ]
    safe_stale_merge_quorum_rerun = (
        merge_packet_summary.admin_squash_allowed
        and only_required_blocker_is_quorum
        and not full_rollup_pending
        and not full_non_green_without_quorum
    )

    blocker = ""
    if any(_is_row_pending(row) for row in required_non_green):
        blocker = "required_checks_pending"
    elif only_required_blocker_is_quorum and full_rollup_pending:
        blocker = "full_rollup_pending"
    elif required_non_green and safe_stale_merge_quorum_rerun:
        blocker = "stale_merge_quorum"
    elif required_non_green:
        blocker = "required_checks_failed"
    elif full_rollup_pending:
        blocker = "full_rollup_pending"
    elif full_non_green_without_quorum:
        blocker = "full_rollup_failed"
    elif not merge_packet_summary.admin_squash_allowed:
        blocker = "merge_packet_blocked"
    else:
        blocker = "ready_for_operator_authorization"

    return SettlementGuard(
        direct_required_checks_green=not required_non_green,
        merge_packet_authorizes=merge_packet_summary.admin_squash_allowed,
        required_non_green=required_non_green,
        full_rollup_pending=full_rollup_pending,
        full_rollup_non_green=full_rollup_non_green,
        stale_merge_quorum=stale_merge_quorum,
        safe_stale_merge_quorum_rerun=safe_stale_merge_quorum_rerun,
        blocker=blocker,
    )


def _derive_settlement_action(
    *,
    base_action: str,
    branch_conflict: bool,
    guard: SettlementGuard,
) -> str:
    """Overlay settlement-specific safety precedence on the base check action."""
    if guard.blocker == "required_checks_pending":
        return "monitor_required_checks"
    if guard.blocker == "required_checks_failed":
        return "repair_failures"
    if guard.blocker == "full_rollup_pending":
        return "monitor_full_rollup"
    if guard.blocker == "full_rollup_failed":
        return "repair_failures"
    if guard.blocker == "stale_merge_quorum":
        return "rerun_stale_merge_quorum"
    if branch_conflict:
        return "diagnose_branch_conflict"
    if guard.blocker == "ready_for_operator_authorization":
        return "ready_for_operator_authorization"
    if guard.blocker == "merge_packet_blocked":
        return "packet_blocked"
    return base_action


def build_followup_result(
    pr_data: dict[str, Any],
    *,
    expected_head: str | None = None,
    run_data_by_id: dict[str, dict[str, Any]] | None = None,
    log_summary_by_job: dict[str, list[str]] | None = None,
    wait_run: WaitRunDiagnosis | None = None,
    allow_rerun_commands: bool = False,
    required_checks: list[dict[str, Any]] | None = None,
    full_rollup_checks: list[dict[str, Any]] | None = None,
    merge_packet: dict[str, Any] | None = None,
    settlement_guard: bool = False,
) -> FollowupResult:
    """Build the follow-up decision from already-fetched PR/run data."""
    pr_number = int(pr_data.get("number") or pr_data.get("pr") or 0)
    head = str(pr_data.get("headRefOid") or "").strip()
    mergeable = str(pr_data.get("mergeable") or "").strip()
    merge_state_status = str(pr_data.get("mergeStateStatus") or "").strip()
    checks: list[CheckDiagnosis] = []
    for check in latest_status_check_rollup(list(pr_data.get("statusCheckRollup") or [])):
        run_id, _job_id_value = parse_run_job_ids(str(check.get("detailsUrl") or ""))
        diagnosis = classify_check(check, head, (run_data_by_id or {}).get(run_id or ""))
        if diagnosis.job_id and log_summary_by_job:
            diagnosis.log_summary = log_summary_by_job.get(diagnosis.job_id, [])
        checks.append(diagnosis)

    if wait_run and not (wait_run.head_sha and head and wait_run.head_sha != head):
        waited_job_ids = {job.job_id for job in wait_run.jobs if job.job_id}
        checks = [check for check in checks if check.job_id not in waited_job_ids]
        checks.extend(wait_run.jobs)

    branch_conflict = _has_branch_conflict(pr_data)
    action = _derive_action(
        checks,
        wait_run,
        expected_head=expected_head,
        head=head,
        branch_conflict=branch_conflict,
    )
    required_rows = [check_row_from_gh(row) for row in required_checks or []]
    full_rollup_rows = [check_row_from_gh(row) for row in full_rollup_checks or []]
    merge_summary = summarize_merge_packet(merge_packet, pr_number) if settlement_guard else None
    settlement = (
        build_settlement_guard(
            required_checks=required_rows,
            full_rollup_checks=full_rollup_rows,
            merge_packet_summary=merge_summary,
        )
        if settlement_guard and merge_summary is not None
        else None
    )
    if settlement and action not in {"head_drift", "stale_wait_run"}:
        action = _derive_settlement_action(
            base_action=action,
            branch_conflict=branch_conflict,
            guard=settlement,
        )
    rerun_commands = [
        check.rerun_command
        for check in checks
        if check.classification == "early_cancelled" and check.rerun_command
    ]
    if settlement and action == "rerun_stale_merge_quorum" and settlement.stale_merge_quorum:
        rerun_command = _rerun_command_for_row(settlement.stale_merge_quorum)
        rerun_commands = [rerun_command] if rerun_command else []
    if not allow_rerun_commands or action != "rerun_cancelled":
        if action != "rerun_stale_merge_quorum":
            rerun_commands = []
    if action == "rerun_stale_merge_quorum" and not allow_rerun_commands:
        rerun_commands = []

    prompt = build_prompt(
        pr_number=pr_number,
        head=head,
        expected_head=expected_head,
        action=action,
        checks=checks,
        rerun_commands=rerun_commands,
        wait_run=wait_run,
        required_checks=required_rows,
        full_rollup_checks=full_rollup_rows,
        merge_packet_summary=merge_summary,
        settlement_guard=settlement,
    )
    return FollowupResult(
        pr=pr_number,
        head=head,
        mergeable=mergeable,
        merge_state_status=merge_state_status,
        expected_head=expected_head,
        action=action,
        checks=checks,
        rerun_commands=rerun_commands,
        prompt=prompt,
        wait_run=wait_run,
        required_checks=required_rows,
        full_rollup_checks=full_rollup_rows,
        merge_packet_summary=merge_summary,
        settlement_guard=settlement,
    )


def build_prompt(
    *,
    pr_number: int,
    head: str,
    expected_head: str | None,
    action: str,
    checks: list[CheckDiagnosis],
    rerun_commands: list[str],
    wait_run: WaitRunDiagnosis | None = None,
    required_checks: list[CheckRow] | None = None,
    full_rollup_checks: list[CheckRow] | None = None,
    merge_packet_summary: MergePacketSummary | None = None,
    settlement_guard: SettlementGuard | None = None,
) -> str:
    """Render the recursive best-next prompt."""
    lines = [
        "Start from live repo truth in /Users/armand/Development/aragora. Do not trust prior transcript state. Check your Aragora operator-steering mailbox before lane work.",
        "",
    ]
    pin = expected_head or head
    if action == "head_drift":
        lines.extend(
            [
                f"Goal: refresh #{pr_number} follow-up because the live head drifted from {expected_head} to {head}. Do not merge, rerun, push, edit files, start cleanup, or start broader queue settlement.",
                "",
            ]
        )
    elif action == "stale_wait_run":
        run_head = wait_run.head_sha if wait_run else "unknown"
        lines.extend(
            [
                f"Goal: refresh #{pr_number} follow-up because the waited Actions run belongs to stale head {run_head}, while the live PR head is {head}. Do not merge, rerun, push, edit files, start cleanup, or start broader queue settlement.",
                "",
            ]
        )
    elif action == "repair_failures":
        lines.extend(
            [
                f"Goal: make one bounded progress increment on #{pr_number} by repairing only current-head real CI failures at head {pin}. Do not merge, start broader queue settlement, rerun cancelled jobs, start cleanup, or touch unrelated PRs/files.",
                "",
            ]
        )
    elif action == "collect_model_evidence":
        lines.extend(
            [
                f"Goal: make one bounded progress increment on #{pr_number} by collecting exactly one current-head non-Codex model/dogfood evidence signal at head {pin}. Do not merge, start broader queue settlement, rerun cancelled jobs, start cleanup, push unrelated work, or touch unrelated PRs/files.",
                "",
            ]
        )
    elif action == "rerun_cancelled":
        lines.extend(
            [
                f"Goal: make one bounded progress increment on #{pr_number} by rerunning only current-head early-cancelled jobs at head {pin}. Do not merge, push, edit files, start cleanup, or touch unrelated PRs/files.",
                "",
            ]
        )
    elif action == "monitor":
        lines.extend(
            [
                f"Goal: monitor #{pr_number} at head {pin} until checks settle. Do not merge, rerun, push, edit files, start cleanup, or start broader queue settlement.",
                "",
            ]
        )
    elif action == "monitor_required_checks":
        lines.extend(
            [
                f"Goal: monitor #{pr_number} at head {pin} until direct required checks settle. Do not merge, rerun, push, edit files, start cleanup, or start broader queue settlement.",
                "",
            ]
        )
    elif action == "monitor_full_rollup":
        lines.extend(
            [
                f"Goal: monitor #{pr_number} at head {pin} until full-rollup checks settle. Do not merge, rerun, push, edit files, start cleanup, or start broader queue settlement.",
                "",
            ]
        )
    elif action == "rerun_stale_merge_quorum":
        lines.extend(
            [
                f"Goal: rerun only the current-head stale/cancelled aragora-merge-quorum job for #{pr_number} at head {pin}, then watch required checks settle. Do not merge, push, edit files, start cleanup, or touch unrelated PRs/files.",
                "",
            ]
        )
    elif action == "ready_for_operator_authorization":
        lines.extend(
            [
                f"Goal: request explicit exact-head operator authorization for #{pr_number} at head {pin}. Do not merge in this run unless a later prompt gives exact-head authorization.",
                "",
            ]
        )
    elif action == "packet_blocked":
        lines.extend(
            [
                f"Goal: classify #{pr_number}'s merge-packet blocker at head {pin} after direct checks passed. Do not merge, rerun, push, edit files, start cleanup, or start broader queue settlement.",
                "",
            ]
        )
    elif action == "diagnose_branch_conflict":
        lines.extend(
            [
                f"Goal: diagnose only #{pr_number}'s live branch conflict or mergeability blocker at head {pin}. Do not merge, rerun CI, push, edit files outside a later explicitly authorized conflict repair, start cleanup, or start broader queue settlement.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                f"Goal: continue exact-head follow-up for #{pr_number} at head {pin}. Do not merge, push, edit files, start cleanup, or touch unrelated PRs/files.",
                "",
            ]
        )

    lines.extend(
        [
            "Run read-only first:",
            "- git status --short --branch --untracked-files=all",
            "- python3 scripts/agent_bridge.py --json health || true",
            f"- python3 scripts/identify_lane_owner.py --pr {pr_number} --json || true",
            f"- gh pr view {pr_number} --json number,state,isDraft,headRefName,headRefOid,mergeable,mergeStateStatus,statusCheckRollup,url",
            "",
            f"If any active lane owns #{pr_number} repair/settlement, stop and report owner. If #{pr_number} head drifted from {pin}, stop and produce a refreshed prompt pinned to live head.",
            "",
        ]
    )

    if wait_run:
        lines.extend(
            [
                f"Wait-run diagnosis: run {wait_run.run_id} / {wait_run.workflow} is {wait_run.status} with conclusion {wait_run.conclusion or 'none'} at head {wait_run.head_sha or 'unknown'}.",
                "",
            ]
        )

    interesting = [check for check in checks if check.classification != "green"]
    if interesting:
        lines.append("Current non-green diagnosis:")
        for check in interesting:
            run_job = (
                f" run {check.run_id}, job {check.job_id}" if check.run_id and check.job_id else ""
            )
            lines.append(
                f"- {check.workflow} / {check.name}: {check.classification}{run_job}; {check.summary}"
            )
            if (
                action in {"repair_failures", "collect_model_evidence"}
                and check.classification == "real_failure"
                and check.log_command
            ):
                lines.append(f"  inspect: {check.log_command}")
            for item in check.log_summary[-3:]:
                lines.append(f"  log: {item}")
        lines.append("")

    if settlement_guard:
        lines.append("Settlement guard state:")
        if settlement_guard.required_non_green:
            lines.append("Direct required checks are still pending/failing:")
            for row in settlement_guard.required_non_green:
                lines.append(f"- {row.workflow} / {row.name}: {row.bucket or row.state}")
        else:
            lines.append("- Direct required checks: green/green-equivalent")
        if settlement_guard.full_rollup_pending:
            lines.append("Full-rollup pending checks:")
            for row in settlement_guard.full_rollup_pending:
                lines.append(f"- {row.workflow} / {row.name}: {row.bucket or row.state}")
        if merge_packet_summary:
            packet_status = (
                "authorizes"
                if merge_packet_summary.admin_squash_allowed
                else f"blocked ({merge_packet_summary.verdict or 'unknown'})"
            )
            lines.append(f"- Merge-packet: {packet_status}")
        lines.append("")

    if action == "repair_failures":
        lines.append(
            "Repair only the real failed checks. Do not rerun cancelled rows until the substantive failures are green."
        )
    elif action == "collect_model_evidence":
        lines.append(
            "If merge-packet still blocks on model quorum or focused adversarial dogfood, collect exactly one current-head non-Codex model/dogfood evidence signal."
        )
        lines.append(
            "Post exactly one valid PR comment only if the evidence is current-head, non-Codex, lists files reviewed, puts findings first, includes validation run/not-run reasons, includes focused adversarial dogfood verdict, and states that it is not merge authorization."
        )
        lines.append(
            "Then rerun review-queue merge-packet for the PR and report the next blocker. Do not merge."
        )
    elif action == "rerun_cancelled" and rerun_commands:
        lines.append("If the same rows remain current-head early cancellations, run only:")
        for command in rerun_commands:
            lines.append(f"- {command}")
        lines.append("Then monitor until those rerun jobs settle.")
    elif action == "rerun_cancelled":
        lines.append(
            "Only early-cancelled rows remain; rerun commands were intentionally withheld by the helper. Re-run the helper with --allow-rerun-commands to generate exact commands."
        )
    elif action == "stale_wait_run":
        lines.append(
            "Ignore the stale waited run for current-head decisions. Re-run the helper against a current-head run/check or omit --wait-run and use live statusCheckRollup."
        )
    elif action == "monitor" and wait_run and wait_run.timed_out:
        lines.append(
            f"Run {wait_run.run_id} did not settle before the wait timeout. Monitor again before diagnosing or rerunning."
        )
    elif action == "monitor":
        lines.append("If checks are still in progress, report exact names and stop.")
    elif action == "monitor_required_checks":
        lines.append(
            "If direct required checks are still pending/failing, report the exact names and stop. Do not use merge-packet authorization while direct checks disagree."
        )
    elif action == "monitor_full_rollup":
        lines.append(
            "If full-rollup checks are still pending, wait only for those checks to settle before diagnosing or rerunning aragora-merge-quorum."
        )
    elif action == "rerun_stale_merge_quorum" and rerun_commands:
        lines.append("Run only this stale/cancelled quorum rerun command:")
        for command in rerun_commands:
            lines.append(f"- {command}")
        lines.append(
            "Then watch required checks and rerun review-queue merge-packet. Do not merge."
        )
    elif action == "rerun_stale_merge_quorum":
        lines.append(
            "Only stale/cancelled aragora-merge-quorum remains, but the rerun command was withheld. Re-run the helper with --allow-rerun-commands or --rerun-stale-merge-quorum-once when explicitly authorized."
        )
    elif action == "ready_for_operator_authorization":
        lines.extend(
            [
                "Exact operator authorization prompt:",
                f"I explicitly authorize exact-head admin squash merge for PR #{pr_number} at head {pin} if live gates still pass.",
                "",
                "Run root status, active-session conflicts, identify_lane_owner for the PR and branch, gh pr view/checks, and merge-packet.",
                "",
                f"Proceed only if #{pr_number} is unowned, open, non-draft, exact-head stable, required checks are green, merge-packet says admin_squash_allowed=true with not_ready=[], and unresolved_dissent=false.",
                "",
                "If clean, merge exactly:",
                f"gh pr merge {pr_number} --squash --admin --match-head-commit {pin}",
                "",
                "Then record settlement and rerun merge-packet. Do not touch unrelated surfaces.",
            ]
        )
    elif action == "packet_blocked":
        lines.append(
            "Direct checks are green, but merge-packet does not authorize settlement. Report the exact packet blocker and do not merge."
        )
    elif action == "green":
        lines.append(
            f"If #{pr_number} remains green/green-equivalent, run review-queue merge-packet for the PR. Do not merge."
        )
    elif action == "diagnose_branch_conflict":
        lines.append(
            f"Checks are green/green-equivalent, but #{pr_number} is still CONFLICTING/DIRTY. Diagnose only the live branch conflict or merge-packet blocker and produce a bounded repair prompt. Do not merge."
        )

    lines.extend(
        [
            "",
            "Final report must include root state, active/conflict lanes, head/check state, action withheld/taken, and a recursive best next prompt.",
            INCREMENTAL_PROGRESS_SENTENCE,
            META_AUTOMATION_SENTENCE,
        ]
    )
    return "\n".join(lines)


def _run_gh_json(args: list[str]) -> dict[str, Any]:
    completed = subprocess.run(args, check=True, text=True, capture_output=True)
    return json.loads(completed.stdout)


def _run_json_allow_pending(args: list[str]) -> Any:
    """Run a JSON command, accepting ``gh pr checks`` exit code 8 for pending checks."""
    completed = subprocess.run(args, check=False, text=True, capture_output=True)
    if completed.returncode not in {0, 8}:
        raise subprocess.CalledProcessError(
            completed.returncode,
            args,
            output=completed.stdout,
            stderr=completed.stderr,
        )
    return json.loads(completed.stdout or "[]")


def _run_command(command: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(shlex.split(command), check=False, text=True, capture_output=True)


def _run_gh_log(args: list[str]) -> str:
    completed = subprocess.run(args, check=True, text=True, capture_output=True)
    return completed.stdout


def wait_for_run(
    run_id: str,
    *,
    poll_interval_seconds: float,
    timeout_seconds: float,
) -> tuple[dict[str, Any], bool]:
    """Poll a GitHub Actions run until it completes or the timeout expires."""
    deadline = time.monotonic() + max(0.0, timeout_seconds)
    latest: dict[str, Any] | None = None
    while True:
        latest = _run_gh_json(
            [
                "gh",
                "run",
                "view",
                run_id,
                "--json",
                "status,conclusion,event,headSha,workflowName,jobs",
            ]
        )
        if str(latest.get("status") or "").lower() == "completed":
            return latest, False
        if time.monotonic() >= deadline:
            return latest, True
        time.sleep(max(0.0, poll_interval_seconds))


def fetch_pr_checks_rows(pr_number: int, *, required: bool) -> list[dict[str, Any]]:
    """Fetch PR checks through ``gh pr checks`` while preserving pending rows."""
    args = [
        "gh",
        "pr",
        "checks",
        str(pr_number),
        "--watch=false",
        "--json",
        "name,state,bucket,workflow,link",
    ]
    if required:
        args.insert(4, "--required")
    rows = _run_json_allow_pending(args)
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def wait_for_full_rollup_checks(
    pr_number: int,
    *,
    poll_interval_seconds: float,
    timeout_seconds: float,
) -> tuple[list[dict[str, Any]], bool]:
    """Poll full PR checks until no rows are pending or timeout expires."""
    deadline = time.monotonic() + max(0.0, timeout_seconds)
    latest: list[dict[str, Any]] = []
    while True:
        latest = fetch_pr_checks_rows(pr_number, required=False)
        if not any(_is_row_pending(check_row_from_gh(row)) for row in latest):
            return latest, False
        if time.monotonic() >= deadline:
            return latest, True
        time.sleep(max(0.0, poll_interval_seconds))


def _fetch_logs_for_failed_jobs(run_id: str, run_data: dict[str, Any]) -> dict[str, list[str]]:
    summaries: dict[str, list[str]] = {}
    for job in run_data.get("jobs") or []:
        if not isinstance(job, dict):
            continue
        job_id = _job_id(job)
        conclusion = str(job.get("conclusion") or "").upper()
        if not job_id or conclusion not in FAILURE_CONCLUSIONS:
            continue
        try:
            log_text = _run_gh_log(["gh", "run", "view", run_id, "--job", job_id, "--log"])
            summaries[job_id] = summarize_log(log_text)
        except subprocess.CalledProcessError:
            summaries[job_id] = ["failed to fetch job log"]
    return summaries


def fetch_live_result(
    pr_number: int,
    *,
    expected_head: str | None,
    include_logs: bool,
    allow_rerun_commands: bool,
    wait_run_id: str | None = None,
    wait_check: str | None = None,
    settlement_guard: bool = False,
    wait_full_rollup: bool = False,
    rerun_stale_merge_quorum_once: bool = False,
    wait_interval_seconds: float = 180.0,
    wait_timeout_seconds: float = 1800.0,
) -> FollowupResult:
    """Fetch live PR/run data through gh and build a follow-up result."""
    pr_data = _run_gh_json(
        [
            "gh",
            "pr",
            "view",
            str(pr_number),
            "--json",
            "number,state,isDraft,headRefName,headRefOid,mergeable,mergeStateStatus,statusCheckRollup,url",
        ]
    )
    head = str(pr_data.get("headRefOid") or "")
    run_data_by_id: dict[str, dict[str, Any]] = {}
    log_summary_by_job: dict[str, list[str]] = {}
    wait_run: WaitRunDiagnosis | None = None
    required_checks: list[dict[str, Any]] = []
    full_rollup_checks: list[dict[str, Any]] = []
    merge_packet: dict[str, Any] | None = None
    rerun_attempt: RerunAttempt | None = None

    if wait_run_id is None and wait_check:
        wait_run_id = wait_check_run_id(pr_data, wait_check)

    if wait_run_id:
        run_data, timed_out = wait_for_run(
            wait_run_id,
            poll_interval_seconds=wait_interval_seconds,
            timeout_seconds=wait_timeout_seconds,
        )
        run_data_by_id[wait_run_id] = run_data
        log_summary_by_job.update(_fetch_logs_for_failed_jobs(wait_run_id, run_data))
        wait_run = diagnose_wait_run(
            wait_run_id,
            run_data,
            pr_head=head,
            timed_out=timed_out,
            log_summary_by_job=log_summary_by_job,
        )

    for check in latest_status_check_rollup(list(pr_data.get("statusCheckRollup") or [])):
        diagnosis = classify_check(check, head)
        if diagnosis.classification not in {"real_failure", "early_cancelled", "unknown"}:
            continue
        if not diagnosis.run_id or not diagnosis.job_id or diagnosis.run_id in run_data_by_id:
            continue
        try:
            run_data_by_id[diagnosis.run_id] = _run_gh_json(
                [
                    "gh",
                    "run",
                    "view",
                    diagnosis.run_id,
                    "--json",
                    "status,conclusion,event,headSha,workflowName,jobs",
                ]
            )
        except (subprocess.CalledProcessError, json.JSONDecodeError):
            continue
        should_fetch_log = include_logs or (
            diagnosis.classification == "real_failure" and _is_merge_quorum_check(diagnosis)
        )
        if should_fetch_log and diagnosis.classification == "real_failure":
            try:
                log_text = _run_gh_log(
                    ["gh", "run", "view", diagnosis.run_id, "--job", diagnosis.job_id, "--log"]
                )
                log_summary_by_job[diagnosis.job_id] = summarize_log(log_text)
            except subprocess.CalledProcessError:
                log_summary_by_job[diagnosis.job_id] = ["failed to fetch job log"]

    if settlement_guard:
        required_checks = fetch_pr_checks_rows(pr_number, required=True)
        if wait_full_rollup:
            full_rollup_checks, _timed_out = wait_for_full_rollup_checks(
                pr_number,
                poll_interval_seconds=wait_interval_seconds,
                timeout_seconds=wait_timeout_seconds,
            )
        else:
            full_rollup_checks = fetch_pr_checks_rows(pr_number, required=False)
        merge_packet = _run_gh_json(
            [
                sys.executable,
                "-m",
                "aragora.cli.main",
                "review-queue",
                "merge-packet",
                "--pr",
                str(pr_number),
                "--json",
            ]
        )

    result = build_followup_result(
        pr_data,
        expected_head=expected_head,
        run_data_by_id=run_data_by_id,
        log_summary_by_job=log_summary_by_job,
        wait_run=wait_run,
        allow_rerun_commands=allow_rerun_commands or rerun_stale_merge_quorum_once,
        required_checks=required_checks,
        full_rollup_checks=full_rollup_checks,
        merge_packet=merge_packet,
        settlement_guard=settlement_guard,
    )
    if rerun_stale_merge_quorum_once:
        command = (
            result.rerun_commands[0]
            if result.action == "rerun_stale_merge_quorum" and result.rerun_commands
            else None
        )
        if command:
            completed = _run_command(command)
            rerun_attempt = RerunAttempt(
                requested=True,
                executed=True,
                command=command,
                returncode=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
                reason="executed stale/cancelled aragora-merge-quorum rerun",
            )
        else:
            rerun_attempt = RerunAttempt(
                requested=True,
                executed=False,
                reason=f"rerun not safe for action {result.action}",
            )
        result.rerun_attempt = rerun_attempt
    return result


def _result_to_json(result: FollowupResult) -> str:
    payload = asdict(result)
    payload["generated_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return json.dumps(payload, indent=2, sort_keys=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pr", type=int, required=True, help="Pull request number to inspect")
    parser.add_argument("--head", help="Expected PR head SHA; head drift stops the prompt")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    parser.add_argument("--prompt", action="store_true", help="Print the recursive prompt")
    parser.add_argument(
        "--include-logs",
        action="store_true",
        help="Fetch failed job logs and include concise failure snippets",
    )
    parser.add_argument(
        "--allow-rerun-commands",
        action="store_true",
        help="Include exact rerun commands when only current-head early cancellations remain",
    )
    parser.add_argument(
        "--wait-run",
        help="Poll one GitHub Actions run until completion before generating the prompt",
    )
    parser.add_argument(
        "--wait-check",
        help="Poll the run for an in-progress status check selected as 'Workflow/Check name'",
    )
    parser.add_argument(
        "--settlement-guard",
        action="store_true",
        help="Fetch direct required checks and merge-packet before surfacing settlement prompts",
    )
    parser.add_argument(
        "--wait-full-rollup",
        action="store_true",
        help="Poll full PR check rollup until no rows remain pending before deciding",
    )
    parser.add_argument(
        "--rerun-stale-merge-quorum-once",
        action="store_true",
        help="Mutating opt-in: rerun exactly one safe stale/cancelled aragora-merge-quorum job",
    )
    parser.add_argument(
        "--wait-interval-seconds",
        type=float,
        default=180.0,
        help="Polling interval for --wait-run (default: 180)",
    )
    parser.add_argument(
        "--wait-timeout-seconds",
        type=float,
        default=1800.0,
        help="Maximum wait duration for --wait-run (default: 1800)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = fetch_live_result(
        args.pr,
        expected_head=args.head,
        include_logs=args.include_logs,
        allow_rerun_commands=args.allow_rerun_commands,
        wait_run_id=args.wait_run,
        wait_check=args.wait_check,
        settlement_guard=args.settlement_guard,
        wait_full_rollup=args.wait_full_rollup,
        rerun_stale_merge_quorum_once=args.rerun_stale_merge_quorum_once,
        wait_interval_seconds=args.wait_interval_seconds,
        wait_timeout_seconds=args.wait_timeout_seconds,
    )
    if args.json:
        print(_result_to_json(result))
    if args.prompt or not args.json:
        print(result.prompt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
