#!/usr/bin/env python3
"""Diagnose PR check state and generate a bounded follow-up prompt.

The helper is intentionally read-only: it shells out to ``gh`` for current PR
state and run/job diagnostics, but never reruns jobs, pushes, or edits files.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


INCREMENTAL_PROGRESS_SENTENCE = (
    "If the prompt above accomplishes no incremental progress make the next prompt one that "
    "does, include this sentence in all subsequent prompts to ensure they converge towards "
    "prompts that make incremental progress."
)

FAILURE_CONCLUSIONS = {"FAILURE", "TIMED_OUT", "ACTION_REQUIRED"}
GREEN_CONCLUSIONS = {"SUCCESS", "SKIPPED", "NEUTRAL"}
PENDING_STATUSES = {"IN_PROGRESS", "QUEUED", "PENDING", "EXPECTED", "REQUESTED", "WAITING"}
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


@dataclass
class CheckDiagnosis:
    """Normalized status for a single PR check row."""

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
    log_summary: list[str] = field(default_factory=list)


@dataclass
class FollowupResult:
    """Machine-readable PR follow-up decision."""

    pr: int
    head: str
    expected_head: str | None
    action: str
    checks: list[CheckDiagnosis]
    rerun_commands: list[str]
    prompt: str


def parse_run_job_ids(details_url: str) -> tuple[str | None, str | None]:
    """Extract GitHub Actions run/job ids from a details URL."""
    match = re.search(r"/actions/runs/(\d+)/job/(\d+)", details_url or "")
    if not match:
        return None, None
    return match.group(1), match.group(2)


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
        summary = "current-head failure requiring repair"
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
    )


def _is_early_cancelled_job(run_data: dict[str, Any] | None, job_id: str | None) -> bool:
    if not run_data or not job_id:
        return False
    for job in run_data.get("jobs") or []:
        if not isinstance(job, dict):
            continue
        database_id = str(job.get("databaseId") or job.get("id") or "").strip()
        if database_id and database_id != job_id:
            continue
        steps = [step for step in job.get("steps") or [] if isinstance(step, dict)]
        names = [str(step.get("name") or "").lower() for step in steps]
        if not names:
            return False
        first_cancelled_index = next(
            (
                index
                for index, step in enumerate(steps)
                if str(step.get("conclusion") or "").upper() == "CANCELLED"
            ),
            None,
        )
        if first_cancelled_index is None:
            return False
        before_cancel = names[: first_cancelled_index + 1]
        return any(
            any(marker in name for marker in CHECKOUT_MARKERS) for name in before_cancel
        ) and not any(
            any(marker in name for marker in SUBSTANTIVE_MARKERS)
            for name in names[:first_cancelled_index]
        )
    return False


def summarize_log(log_text: str, max_lines: int = 12) -> list[str]:
    """Extract high-signal failure lines without dumping complete logs."""
    markers = ("##[error]", "FAIL:", "FAILED", "Out of date:", " E ", "Error:", "error:")
    lines: list[str] = []
    for raw in log_text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if any(marker in line for marker in markers):
            lines.append(line)
    return lines[-max_lines:]


def build_followup_result(
    pr_data: dict[str, Any],
    *,
    expected_head: str | None = None,
    run_data_by_id: dict[str, dict[str, Any]] | None = None,
    log_summary_by_job: dict[str, list[str]] | None = None,
    allow_rerun_commands: bool = False,
) -> FollowupResult:
    """Build the follow-up decision from already-fetched PR/run data."""
    pr_number = int(pr_data.get("number") or pr_data.get("pr") or 0)
    head = str(pr_data.get("headRefOid") or "").strip()
    checks = []
    for check in latest_status_check_rollup(list(pr_data.get("statusCheckRollup") or [])):
        run_id, job_id = parse_run_job_ids(str(check.get("detailsUrl") or ""))
        diagnosis = classify_check(check, head, (run_data_by_id or {}).get(run_id or ""))
        if diagnosis.job_id and log_summary_by_job:
            diagnosis.log_summary = log_summary_by_job.get(diagnosis.job_id, [])
        checks.append(diagnosis)

    if expected_head and head != expected_head:
        action = "head_drift"
    elif any(check.classification == "real_failure" for check in checks):
        action = "repair_failures"
    elif any(check.classification == "in_progress" for check in checks):
        action = "monitor"
    elif any(check.classification == "unknown" for check in checks):
        action = "diagnose_unknown"
    elif any(check.classification == "early_cancelled" for check in checks):
        action = "rerun_cancelled"
    else:
        action = "green"

    rerun_commands = [
        check.rerun_command
        for check in checks
        if check.classification == "early_cancelled" and check.rerun_command
    ]
    if not allow_rerun_commands or action != "rerun_cancelled":
        rerun_commands = []

    prompt = build_prompt(
        pr_number=pr_number,
        head=head,
        expected_head=expected_head,
        action=action,
        checks=checks,
        rerun_commands=rerun_commands,
    )
    return FollowupResult(
        pr=pr_number,
        head=head,
        expected_head=expected_head,
        action=action,
        checks=checks,
        rerun_commands=rerun_commands,
        prompt=prompt,
    )


def build_prompt(
    *,
    pr_number: int,
    head: str,
    expected_head: str | None,
    action: str,
    checks: list[CheckDiagnosis],
    rerun_commands: list[str],
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
    elif action == "repair_failures":
        lines.extend(
            [
                f"Goal: make one bounded progress increment on #{pr_number} by repairing only current-head real CI failures at head {pin}. Do not merge, start broader queue settlement, rerun cancelled jobs, start cleanup, or touch unrelated PRs/files.",
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
            for item in check.log_summary[:3]:
                lines.append(f"  log: {item}")
        lines.append("")

    if action == "repair_failures":
        lines.append(
            "Repair only the real failed checks. Do not rerun cancelled rows until the substantive failures are green."
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
    elif action == "monitor":
        lines.append("If checks are still in progress, report exact names and stop.")
    elif action == "green":
        lines.append(
            f"If #{pr_number} remains green/green-equivalent, run review-queue merge-packet for the PR. Do not merge."
        )

    lines.extend(
        [
            "",
            "Final report must include root state, active/conflict lanes, head/check state, action withheld/taken, and a recursive best next prompt.",
            INCREMENTAL_PROGRESS_SENTENCE,
        ]
    )
    return "\n".join(lines)


def _run_gh_json(args: list[str]) -> dict[str, Any]:
    completed = subprocess.run(args, check=True, text=True, capture_output=True)
    return json.loads(completed.stdout)


def _run_gh_log(args: list[str]) -> str:
    completed = subprocess.run(args, check=True, text=True, capture_output=True)
    return completed.stdout


def fetch_live_result(
    pr_number: int,
    *,
    expected_head: str | None,
    include_logs: bool,
    allow_rerun_commands: bool,
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
        if include_logs and diagnosis.classification == "real_failure":
            try:
                log_text = _run_gh_log(
                    ["gh", "run", "view", diagnosis.run_id, "--job", diagnosis.job_id, "--log"]
                )
                log_summary_by_job[diagnosis.job_id] = summarize_log(log_text)
            except subprocess.CalledProcessError:
                log_summary_by_job[diagnosis.job_id] = ["failed to fetch job log"]

    return build_followup_result(
        pr_data,
        expected_head=expected_head,
        run_data_by_id=run_data_by_id,
        log_summary_by_job=log_summary_by_job,
        allow_rerun_commands=allow_rerun_commands,
    )


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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    result = fetch_live_result(
        args.pr,
        expected_head=args.head,
        include_logs=args.include_logs,
        allow_rerun_commands=args.allow_rerun_commands,
    )
    if args.json:
        print(_result_to_json(result))
    if args.prompt or not args.json:
        print(result.prompt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
