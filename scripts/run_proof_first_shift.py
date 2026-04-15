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
from dataclasses import asdict, dataclass
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
AUTOMATION_BRANCH_PREFIXES = (
    "codex/",
    "factory/",
    "aragora/boss-harvest/",
    "benchmark-truth-publication/",
)


@dataclass
class ProofFirstRuntimeState:
    boss_restart_count: int = 0
    merge_restart_count: int = 0
    auth_failure_count: int = 0
    publication_failure_count: int = 0
    last_benchmark_run_id: int | None = None
    last_triggered_benchmark_run_id: int | None = None


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _assessment_id() -> str:
    return f"proof-first-{_utc_now().strftime('%Y%m%dT%H%M%SZ')}"


def _runtime_state_path(repo_root: Path) -> Path:
    return repo_root / ".aragora" / DEFAULT_SHIFT_DIRNAME / "runtime_state.json"


def load_runtime_state(path: Path) -> ProofFirstRuntimeState:
    if not path.exists():
        return ProofFirstRuntimeState()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return ProofFirstRuntimeState()
    return ProofFirstRuntimeState(
        boss_restart_count=int(payload.get("boss_restart_count", 0) or 0),
        merge_restart_count=int(payload.get("merge_restart_count", 0) or 0),
        auth_failure_count=int(payload.get("auth_failure_count", 0) or 0),
        publication_failure_count=int(payload.get("publication_failure_count", 0) or 0),
        last_benchmark_run_id=(
            int(payload["last_benchmark_run_id"]) if payload.get("last_benchmark_run_id") else None
        ),
        last_triggered_benchmark_run_id=(
            int(payload["last_triggered_benchmark_run_id"])
            if payload.get("last_triggered_benchmark_run_id")
            else None
        ),
    )


def save_runtime_state(path: Path, state: ProofFirstRuntimeState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(state), indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
        1 for pr in prs if str(pr.get("headRefName") or "").startswith(AUTOMATION_BRANCH_PREFIXES)
    )


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
    return "other_failure"


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
) -> tuple[bool, str]:
    kicked, detail = kickstart_launchd(label)
    if process_running(process_pattern):
        return True, "" if kicked else detail
    if kicked:
        return (
            False,
            f"launchctl kickstart returned success for {label}, but process pattern {process_pattern!r} is still not running",
        )
    return False, detail


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


def run_shift_cycle(
    *,
    repo_root: Path,
    repo: str,
    benchmark_mode: str,
    automation_backlog_limit: int,
    runtime_state: ProofFirstRuntimeState,
    ledger: ShiftLedger | None = None,
) -> dict[str, Any]:
    queue_report = reconcile_proof_first_queue(repo=repo, repo_root=repo_root, apply=True)
    snapshot = collect_boss_lane_snapshot(repo_root=repo_root, repo=repo)
    prs = list_open_prs(repo_root=repo_root, repo=repo)
    automation_backlog = count_automation_backlog(prs)
    open_pr_count = len(prs)
    canonical_queue_count = len(queue_report["kept"])

    boss_running = process_running(DEFAULT_BOSS_PROCESS_PATTERN)
    merge_running = process_running(DEFAULT_MERGE_PROCESS_PATTERN)

    if boss_running:
        runtime_state.boss_restart_count = 0
    if merge_running:
        runtime_state.merge_restart_count = 0

    actions: list[str] = []
    service_failures: list[str] = []
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
        if restarted:
            actions.append("restart_boss_loop")
            if ledger:
                ledger.record_service_restart(service="boss_loop", success=True)
        else:
            actions.append("restart_boss_loop_failed")
            service_failures.append(
                f"BossRestartFailed: {detail or 'boss loop restart did not produce a running process'}"
            )
            if ledger:
                ledger.record_service_restart(service="boss_loop", success=False, detail=detail)

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
        if restarted:
            actions.append("restart_merge_arbiter")
            if ledger:
                ledger.record_service_restart(service="merge_arbiter", success=True)
        else:
            actions.append("restart_merge_arbiter_failed")
            service_failures.append(
                f"MergeArbiterRestartFailed: {detail or 'merge arbiter restart did not produce a running process'}"
            )
            if ledger:
                ledger.record_service_restart(service="merge_arbiter", success=False, detail=detail)

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
                    if ledger:
                        ledger.record_failure(failure_type="auth_failure", detail=failure_log[:500])
                elif failure_class == "publication_failure":
                    runtime_state.publication_failure_count += 1
                    if ledger:
                        ledger.record_failure(
                            failure_type="publication_failure", detail=failure_log[:500]
                        )
            elif conclusion == "success":
                runtime_state.auth_failure_count = 0
                runtime_state.publication_failure_count = 0

    should_trigger, trigger_reason = should_trigger_benchmark_rerun(
        benchmark_mode=benchmark_mode,
        latest_run=latest_run,
        has_open_publication_pr=has_open_benchmark_publication_pr(prs),
        automation_backlog=automation_backlog,
        automation_backlog_limit=automation_backlog_limit,
        last_triggered_run_id=runtime_state.last_triggered_benchmark_run_id,
    )
    if should_trigger:
        trigger_benchmark_workflow(repo_root=repo_root, repo=repo)
        actions.append(f"trigger_benchmark:{trigger_reason}")
        if latest_run is not None:
            runtime_state.last_triggered_benchmark_run_id = int(
                latest_run.get("databaseId", 0) or 0
            )
        else:
            runtime_state.last_triggered_benchmark_run_id = -1

    stop_reason = ""
    if runtime_state.auth_failure_count >= 2:
        stop_reason = "RepeatedAuthFailure: benchmark publication failed auth twice"
    elif runtime_state.publication_failure_count >= 2:
        stop_reason = "RepeatedPublicationFailure: benchmark publication failed PR handoff twice"
    elif service_failures:
        stop_reason = service_failures[0]

    # Record cycle tick in ledger
    benchmark_fresh = False
    if latest_run is not None:
        conclusion = str(latest_run.get("conclusion") or "").strip().lower()
        benchmark_fresh = conclusion == "success"
    if ledger:
        ledger.record_cycle_tick(
            queue_size=canonical_queue_count,
            open_prs=open_pr_count,
            boss_running=boss_running,
            merge_running=merge_running,
            benchmark_fresh=benchmark_fresh,
            actions=actions,
        )

    return {
        "queue_report": queue_report,
        "snapshot": snapshot,
        "open_pr_count": open_pr_count,
        "automation_backlog": automation_backlog,
        "latest_benchmark_run": latest_run,
        "actions": actions,
        "stop_reason": stop_reason,
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
