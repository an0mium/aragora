#!/usr/bin/env python3
"""Bounded Aragora disk recovery coordinator.

This script is intentionally conservative: it only removes worktrees through
scripts/safe_worktree_cleanup.py and treats every timeout as a preservation
blocker recorded in a local quarantine file.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

UTC = timezone.utc
DEFAULT_QUARANTINE = Path(".aragora/cleanup-state/worktree-quarantine.jsonl")
DEFAULT_LOG = Path(".aragora/cleanup-state/disk-recovery-coordinator.jsonl")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _run(
    argv: list[str],
    *,
    cwd: Path,
    timeout: float,
) -> dict[str, Any]:
    started = time.monotonic()
    proc = subprocess.Popen(
        argv,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, stderr = proc.communicate()
        timeout_stdout = _as_text(stdout) or _as_text(exc.stdout)
        timeout_stderr = _as_text(stderr) or _as_text(exc.stderr)
        return {
            "argv": argv,
            "returncode": 124,
            "stdout": timeout_stdout,
            "stderr": timeout_stderr + f"\ntimed out after {exc.timeout}s",
            "timed_out": True,
            "elapsed": round(time.monotonic() - started, 3),
        }
    return {
        "argv": argv,
        "returncode": proc.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "timed_out": timed_out,
        "elapsed": round(time.monotonic() - started, 3),
    }


def _as_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


def _run_json(
    argv: list[str], *, cwd: Path, timeout: float
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    result = _run(argv, cwd=cwd, timeout=timeout)
    try:
        payload = json.loads(result["stdout"] or "{}")
    except json.JSONDecodeError:
        payload = None
    return payload, result


def _free_gib(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return usage.free / (1024**3)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, sort_keys=True) + "\n")


def _load_quarantine(path: Path) -> set[str]:
    quarantined: set[str] = set()
    if not path.exists():
        return quarantined
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        candidate = str(payload.get("path") or "").strip()
        if candidate:
            quarantined.add(candidate)
    return quarantined


def _quarantine(path: Path, candidate: str, reason: str, detail: dict[str, Any]) -> None:
    _append_jsonl(
        path,
        {
            "recorded_at": _utc_now(),
            "path": candidate,
            "reason": reason,
            "detail": detail,
        },
    )


def _git_worktree_paths(repo: Path, timeout: float) -> list[str]:
    result = _run(["git", "worktree", "list", "--porcelain"], cwd=repo, timeout=timeout)
    if result["returncode"] != 0:
        return []
    return [
        line.split(" ", 1)[1]
        for line in str(result["stdout"]).splitlines()
        if line.startswith("worktree ")
    ]


def _active_external_worktrees(prefix: str) -> set[str]:
    proc = subprocess.run(
        ["lsof", "-a", "-d", "cwd", "-Fpcn"],
        text=True,
        capture_output=True,
        check=False,
    )
    active: set[str] = set()
    current: dict[str, Any] = {}
    records: list[dict[str, Any]] = []
    for line in proc.stdout.splitlines():
        if not line:
            continue
        key, value = line[0], line[1:]
        if key == "p":
            if current:
                records.append(current)
            current = {"pid": value}
        elif key == "n":
            current.setdefault("cwds", []).append(value)
    if current:
        records.append(current)

    for record in records:
        for cwd in record.get("cwds", []):
            if not isinstance(cwd, str) or not cwd.startswith(prefix):
                continue
            parts = cwd[len(prefix) :].split("/")
            if len(parts) >= 2 and parts[1] == "aragora":
                active.add(prefix + parts[0] + "/aragora")
    return active


def _root_clean_current(
    repo: Path, timeout: float, *, allow_branch_ahead: bool
) -> tuple[bool, dict[str, Any]]:
    status = _run(["git", "status", "--short"], cwd=repo, timeout=timeout)
    head = _run(["git", "rev-parse", "HEAD"], cwd=repo, timeout=timeout)
    origin = _run(["git", "rev-parse", "origin/main"], cwd=repo, timeout=timeout)
    ancestor = _run(
        ["git", "merge-base", "--is-ancestor", "origin/main", "HEAD"], cwd=repo, timeout=timeout
    )
    same_head = str(head["stdout"]).strip() == str(origin["stdout"]).strip()
    branch_ahead_allowed = allow_branch_ahead and ancestor["returncode"] == 0
    ok = (
        status["returncode"] == 0
        and not str(status["stdout"]).strip()
        and head["returncode"] == 0
        and origin["returncode"] == 0
        and (same_head or branch_ahead_allowed)
    )
    return ok, {
        "status": status,
        "head": str(head["stdout"]).strip(),
        "origin_main": str(origin["stdout"]).strip(),
        "same_head": same_head,
        "branch_ahead_allowed": branch_ahead_allowed,
    }


def _runtime_cleanup(repo: Path, args: argparse.Namespace) -> dict[str, Any]:
    argv = ["python3", "scripts/cleanup_runtime_artifacts.py", "--purge-caches"]
    if args.apply:
        argv.append("--apply")
    return _run(argv, cwd=repo, timeout=args.command_timeout)


def _reconcile_outbox(repo: Path, args: argparse.Namespace) -> dict[str, Any]:
    dry_payload, dry = _run_json(
        [
            "python3",
            "scripts/reconcile_automation_outbox.py",
            "--repo",
            str(repo),
            "--base",
            "origin/main",
            "--dry-run",
            "--json",
        ],
        cwd=repo,
        timeout=args.command_timeout,
    )
    apply_result: dict[str, Any] | None = None
    counts = dry_payload.get("counts", {}) if isinstance(dry_payload, dict) else {}
    satisfied = sum(
        int(counts.get(key, 0) or 0)
        for key in (
            "satisfied_by_existing_receipt",
            "satisfied_by_landed_on_main",
            "satisfied_by_open_pr_merged",
            "satisfied_by_superseded_handoff",
        )
    )
    if args.apply and satisfied > 0:
        _, apply_result = _run_json(
            [
                "python3",
                "scripts/reconcile_automation_outbox.py",
                "--repo",
                str(repo),
                "--base",
                "origin/main",
                "--apply",
                "--json",
            ],
            cwd=repo,
            timeout=args.command_timeout,
        )
    return {"dry_run": dry, "dry_payload": dry_payload, "apply": apply_result}


def _autopilot_cleanup(repo: Path, args: argparse.Namespace) -> dict[str, Any]:
    if not args.apply:
        return {"skipped": True, "reason": "dry_run"}
    payload, result = _run_json(
        [
            "python3",
            "scripts/codex_worktree_autopilot.py",
            "cleanup",
            "--base",
            "main",
            "--ttl-hours",
            "24",
            "--no-delete-branches",
            "--json",
        ],
        cwd=repo,
        timeout=args.command_timeout,
    )
    return {"result": result, "payload": payload}


def _external_cleanup(repo: Path, args: argparse.Namespace) -> dict[str, Any]:
    prefix = "/Users/armand/.codex/worktrees/"
    active = _active_external_worktrees(prefix)
    quarantined = _load_quarantine(args.quarantine_file)
    paths = [
        path
        for path in _git_worktree_paths(repo, args.command_timeout)
        if path.startswith(prefix)
        and path not in active
        and path not in quarantined
        and not path.startswith(prefix + "b968/")
    ]

    inspected = 0
    removed: list[str] = []
    would_remove: list[str] = []
    blocked: list[dict[str, Any]] = []

    for candidate in paths:
        if len(removed) + len(would_remove) >= args.max_cleanup_per_cycle:
            break
        if inspected >= args.max_inspect_per_cycle:
            break
        inspected += 1
        inspect_payload, inspect_result = _run_json(
            [
                "python3",
                "scripts/safe_worktree_cleanup.py",
                "--repo",
                str(repo),
                "inspect",
                candidate,
                "--json",
            ],
            cwd=repo,
            timeout=args.inspect_timeout,
        )
        if inspect_result["timed_out"]:
            _quarantine(args.quarantine_file, candidate, "inspect_timeout", inspect_result)
            blocked.append({"path": candidate, "reason": "inspect_timeout"})
            continue
        if not isinstance(inspect_payload, dict):
            _quarantine(args.quarantine_file, candidate, "inspect_parse_failed", inspect_result)
            blocked.append({"path": candidate, "reason": "inspect_parse_failed"})
            continue
        if not inspect_payload.get("removable"):
            blocked.append(
                {
                    "path": candidate,
                    "reason": "not_removable",
                    "blockers": inspect_payload.get("blockers", []),
                }
            )
            continue

        if not args.apply:
            would_remove.append(candidate)
            continue

        remove_payload, remove_result = _run_json(
            [
                "python3",
                "scripts/safe_worktree_cleanup.py",
                "--repo",
                str(repo),
                "remove",
                candidate,
                "--json",
            ],
            cwd=repo,
            timeout=args.remove_timeout,
        )
        if remove_result["timed_out"]:
            _quarantine(args.quarantine_file, candidate, "remove_timeout", remove_result)
            blocked.append({"path": candidate, "reason": "remove_timeout"})
            continue
        if remove_result["returncode"] == 0 and not Path(candidate).exists():
            removed.append(candidate)
        else:
            blocked.append(
                {
                    "path": candidate,
                    "reason": "remove_failed",
                    "payload": remove_payload,
                    "returncode": remove_result["returncode"],
                }
            )

    return {
        "active_external_skipped": len(active),
        "quarantined_skipped": len(quarantined),
        "inactive_candidates_seen": len(paths),
        "inspected": inspected,
        "max_inspect_per_cycle": args.max_inspect_per_cycle,
        "removed": removed,
        "would_remove": would_remove,
        "blocked": blocked[:10],
    }


def _salvage_snapshot(repo: Path, args: argparse.Namespace) -> dict[str, Any]:
    if args.max_salvage_per_cycle <= 0:
        return {"skipped": True, "reason": "disabled"}
    payload, result = _run_json(
        [
            "python3",
            "scripts/audit_codex_branch_backlog.py",
            "--repo",
            str(repo),
            "--base",
            "origin/main",
            "--prefix",
            "codex/",
            "--summary-only",
            "--json",
            "--include-patch-equivalence",
            "--patch-equivalence-time-budget-seconds",
            "30",
            "--examples",
            str(args.max_salvage_per_cycle),
        ],
        cwd=repo,
        timeout=args.command_timeout,
    )
    return {"payload": payload, "result": result}


def _run_cycle(repo: Path, args: argparse.Namespace, cycle: int) -> dict[str, Any]:
    before = _free_gib(repo)
    clean_current, root = _root_clean_current(
        repo, args.command_timeout, allow_branch_ahead=args.allow_branch_ahead
    )
    if not clean_current:
        return {
            "cycle": cycle,
            "started_at": _utc_now(),
            "ok": False,
            "stop_reason": "root_not_clean_current",
            "root": root,
            "free_gib_before": round(before, 2),
        }

    result = {
        "cycle": cycle,
        "started_at": _utc_now(),
        "ok": True,
        "free_gib_before": round(before, 2),
        "runtime_cleanup": _runtime_cleanup(repo, args),
        "outbox": _reconcile_outbox(repo, args),
        "autopilot_cleanup": _autopilot_cleanup(repo, args),
        "external_cleanup": _external_cleanup(repo, args),
        "salvage_snapshot": _salvage_snapshot(repo, args),
    }
    result["free_gib_after"] = round(_free_gib(repo), 2)
    result["completed_at"] = _utc_now()
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run bounded Aragora disk recovery cycles.")
    parser.add_argument("--repo", default=".", help="Repository path.")
    parser.add_argument("--target-free-gib", type=float, default=100.0)
    parser.add_argument("--hours", type=float, default=6.0)
    parser.add_argument("--cycle-seconds", type=float, default=900.0)
    parser.add_argument("--max-cleanup-per-cycle", type=int, default=5)
    parser.add_argument(
        "--max-inspect-per-cycle",
        type=int,
        default=25,
        help="Maximum external worktree candidates to inspect per cycle.",
    )
    parser.add_argument("--max-salvage-per-cycle", type=int, default=2)
    parser.add_argument("--inspect-timeout", type=float, default=30.0)
    parser.add_argument("--remove-timeout", type=float, default=60.0)
    parser.add_argument("--command-timeout", type=float, default=180.0)
    parser.add_argument("--quarantine-file", type=Path, default=DEFAULT_QUARANTINE)
    parser.add_argument("--jsonl-log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--apply", action="store_true", help="Apply safe cleanup actions.")
    parser.add_argument(
        "--allow-branch-ahead",
        action="store_true",
        help="Allow a clean branch whose HEAD is ahead of origin/main for PR dogfood runs.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one cycle even when --apply is set.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    repo = Path(args.repo).resolve()
    args.quarantine_file = (repo / args.quarantine_file).resolve()
    args.jsonl_log = (repo / args.jsonl_log).resolve()

    deadline = time.monotonic() + args.hours * 3600
    cycle = 0
    last_result: dict[str, Any] | None = None
    while True:
        cycle += 1
        last_result = _run_cycle(repo, args, cycle)
        _append_jsonl(args.jsonl_log, last_result)
        print(json.dumps(last_result, indent=2, sort_keys=True))
        if not last_result.get("ok"):
            return 2
        if float(last_result.get("free_gib_after", 0.0)) >= args.target_free_gib:
            return 0
        if args.once or not args.apply:
            return 0
        if time.monotonic() >= deadline:
            return 1
        time.sleep(args.cycle_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
