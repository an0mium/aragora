#!/usr/bin/env python3
"""Select one safe bounded action for an Aragora overnight cycle.

The conductor is intentionally conservative: it gathers live state, writes an
optional ledger row, and emits the next prompt. It does not merge, mark PRs
ready, rerun CI, record Tier 4 settlement, or mutate branch protection.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

GREEN_STATES = {"PASS", "SUCCESS", "SKIPPED"}
PENDING_STATES = {"EXPECTED", "PENDING", "QUEUED", "REQUESTED", "STARTUP_FAILURE", "WAITING"}
FAIL_STATES = {"ACTION_REQUIRED", "CANCELLED", "ERROR", "FAILURE", "FAILED", "STALE", "TIMED_OUT"}
# UNSTABLE often means optional rollup noise. It is eligible only for prompt
# selection because gather_state filters required checks and every emitted merge
# prompt still requires merge-packet and settle_one rechecks before action.
MERGE_PROMPT_STATES = {"CLEAN", "UNSTABLE"}

MAC_RUNNER = "mac-studio-m3ultra"
HETZNER_PREFIX = "aragora-hetzner-cpu"

FORBIDDEN_ACTIONS = [
    "merge",
    "mark_ready",
    "rerun_required_ci",
    "record_tier4_settlement",
    "mutate_branch_protection",
    "mutate_dirty_root_source_files",
]


@dataclass(frozen=True)
class CommandResult:
    args: list[str]
    returncode: int
    stdout: str = ""
    stderr: str = ""

    @property
    def ok(self) -> bool:
        return self.returncode == 0


class CommandRunner:
    def __init__(self, cwd: Path):
        self.cwd = cwd

    def run(self, args: list[str], *, timeout: int = 60) -> CommandResult:
        try:
            completed = subprocess.run(
                args,
                cwd=self.cwd,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return CommandResult(args=args, returncode=124, stderr=str(exc))
        return CommandResult(
            args=args,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()


def _json_payload(result: CommandResult) -> Any:
    if not result.ok:
        return None
    try:
        return json.loads(result.stdout or "null")
    except json.JSONDecodeError:
        return None


def _probe_json(runner: CommandRunner, args: list[str], *, timeout: int = 60) -> dict[str, Any]:
    result = runner.run(args, timeout=timeout)
    return {
        "ok": result.ok,
        "args": args,
        "returncode": result.returncode,
        "data": _json_payload(result),
        "stderr": result.stderr.strip(),
    }


def check_summary(rows: list[dict[str, Any]] | None) -> dict[str, Any]:
    rows = rows or []
    non_green: list[dict[str, str]] = []
    pending: list[dict[str, str]] = []
    real_failures: list[dict[str, str]] = []

    for row in rows:
        state = str(row.get("state") or row.get("conclusion") or "").upper()
        bucket = str(row.get("bucket") or "").lower()
        compact = {
            "name": str(row.get("name") or row.get("context") or "unknown"),
            "state": state or "UNKNOWN",
            "bucket": bucket or "unknown",
        }
        if state not in GREEN_STATES:
            non_green.append(compact)
        if state in PENDING_STATES or bucket in {"pending", "queued", "in_progress"}:
            pending.append(compact)
        if state in FAIL_STATES or bucket in {"fail", "failure"}:
            real_failures.append(compact)

    return {
        "total": len(rows),
        "green": bool(rows) and not non_green,
        "non_green": non_green,
        "pending": pending,
        "real_failures": real_failures,
    }


def required_checks_args(pr_number: int) -> list[str]:
    return [
        "gh",
        "pr",
        "checks",
        str(pr_number),
        "--required",
        "--json",
        "name,state,bucket,workflow,link,startedAt,completedAt",
    ]


def runner_blockers(runner_payload: dict[str, Any] | list[dict[str, Any]] | None) -> list[str]:
    if isinstance(runner_payload, dict):
        runners = runner_payload.get("runners")
    else:
        runners = runner_payload
    if not isinstance(runners, list):
        return ["runner inventory unavailable"]

    by_name = {str(runner.get("name")): runner for runner in runners}
    blockers: list[str] = []

    mac = by_name.get(MAC_RUNNER)
    if not mac:
        blockers.append(f"{MAC_RUNNER} missing from runner inventory")
    elif str(mac.get("status")).lower() != "online":
        blockers.append(f"{MAC_RUNNER} is {mac.get('status') or 'unknown'}")

    hetzners = [runner for name, runner in by_name.items() if name.startswith(HETZNER_PREFIX)]
    online_hetzners = [
        runner for runner in hetzners if str(runner.get("status")).lower() == "online"
    ]
    if not hetzners:
        blockers.append(f"{HETZNER_PREFIX}* runners missing from runner inventory")
    elif not online_hetzners:
        blockers.append(f"all {HETZNER_PREFIX}* runners are offline")

    return blockers


def _lane_conflicts(snapshot: dict[str, Any] | None) -> list[Any]:
    if not isinstance(snapshot, dict):
        return []
    conflicts = snapshot.get("lane_conflicts")
    if isinstance(conflicts, list):
        return conflicts
    summary = snapshot.get("summary")
    if isinstance(summary, dict) and summary.get("conflict_lanes"):
        return [f"{summary['conflict_lanes']} active lane conflict(s)"]
    return []


def _health_issues(snapshot: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(snapshot, dict):
        return []
    health = snapshot.get("health")
    if isinstance(health, dict):
        issues = health.get("issues")
        if isinstance(issues, list):
            return [issue for issue in issues if isinstance(issue, dict)]
    return []


def _pr_is_clean(pr: dict[str, Any]) -> bool:
    return (
        str(pr.get("state", "OPEN")).upper() in {"", "OPEN"}
        and pr.get("mergeable") == "MERGEABLE"
        and str(pr.get("mergeStateStatus") or "").upper() in MERGE_PROMPT_STATES
    )


def _pr_identity(pr: dict[str, Any]) -> dict[str, Any]:
    return {
        "pr": pr.get("number"),
        "branch": pr.get("headRefName"),
        "head": pr.get("headRefOid"),
        "url": pr.get("url"),
    }


def _base_prompt_prefix(pr: dict[str, Any]) -> str:
    return (
        "Start from live repo truth in the Aragora repo root. "
        "Do not trust prior transcript state. Do not duplicate active lanes. "
        "Do not touch unrelated PRs. Do not use or mutate dirty root source files.\n\n"
        f"Target PR #{pr['number']} at exact head {pr['headRefOid']} "
        f"on branch {pr['headRefName']}.\n"
    )


def _merge_ready_prompt(pr: dict[str, Any]) -> str:
    return (
        _base_prompt_prefix(pr)
        + "\nGoal: re-check mailbox, owner state, active sessions, gh pr view/checks, "
        "full statusCheckRollup, review-queue merge-packet, and scripts/settle_one_pr.py "
        "from a clean current origin/main checkout. If merge-packet reports "
        "admin_squash_allowed=true with not_ready=[] and settle_one_pr.py reports "
        "packet_authorized_dry_run with no blockers, output the exact merge "
        "authorization prompt. Do not merge without separate explicit operator "
        "authorization."
    )


def _draft_gate_prompt(pr: dict[str, Any]) -> str:
    return (
        _base_prompt_prefix(pr)
        + "\nGoal: continue watching this draft PR after green checks. Re-check mailbox, "
        "owner state, active sessions, gh pr view/checks, full statusCheckRollup, "
        "review-queue merge-packet, and scripts/settle_one_pr.py. Stop on active "
        "ownership, head drift, dirty/conflicting state, pending checks, real check "
        "failures, out-of-scope files, or Tier 3/4 human-risk mutation. If all gates "
        "are green and the PR remains draft, output the exact ready authorization "
        "prompt. Do not mark ready or merge without separate explicit operator "
        "authorization."
    )


def _failing_check_prompt(pr: dict[str, Any], failure: dict[str, str]) -> str:
    return (
        _base_prompt_prefix(pr) + "\nGoal: repair only the exact-head real check failure "
        f"{failure['name']} ({failure['state']}). Use a fresh clean implementation "
        "worktree from the PR head. Re-check mailbox, owner state, active sessions, "
        "gh pr view/checks, full statusCheckRollup, merge-packet, and settle_one_pr.py. "
        "Stop on active ownership, head drift, dirty/conflicting state, or out-of-scope "
        "files. Patch the smallest repo-supported fix, validate with focused tests, "
        "ruff check, ruff format --check, and automation_pr_preflight, then push only "
        "the bounded repair branch. Do not rerun CI, mark ready, or merge."
    )


def _publisher_prompt(state: dict[str, Any]) -> str:
    publisher = state.get("publisher", {}).get("data") or {}
    verdict = publisher.get("verdict") or publisher.get("status") or "unknown"
    return (
        "Start from live repo truth in the Aragora repo root. Use a clean "
        "current origin/main checkout only. Goal: refresh exactly one current-base "
        f"outbox/publication artifact because publisher health is {verdict!r}. "
        "Re-check GitHub CLI health, publisher_freshness_check.py, active owner state, "
        "and automation outbox freshness. If GitHub auth or publication is unavailable, "
        "write a precise blocker report instead of retry-looping. Do not touch dirty "
        "root files, merge, mark ready, rerun CI, or record Tier 4 settlement."
    )


def _stale_owner_prompt(issue: dict[str, Any]) -> str:
    detail = issue.get("detail") or issue.get("session") or "stale owner health issue"
    return (
        "Start from live repo truth in the Aragora repo root. Use a clean "
        "current origin/main checkout only. Goal: coordinate the stale or incomplete "
        f"owner lane reported as: {detail}. Re-check mailbox, owner state, active "
        "sessions, and agent_bridge.py operator-snapshot. If the owner is still live, "
        "send a repo-supported steering message to that owner or skip. If it is stale, "
        "emit the exact operator coordination prompt. Do not supersede, collect "
        "evidence, rerun checks, mark ready, or merge without explicit authorization."
    )


def _blocker_prompt(blockers: list[str]) -> str:
    joined = "; ".join(blockers)
    return (
        "Start from live repo truth in the Aragora repo root. Use a clean "
        "current origin/main checkout only. Goal: produce a compact blocker report for "
        f"the overnight conductor. Current blocker(s): {joined}. Do not mutate source "
        "files, rerun CI, mark statuses, record settlement, or merge."
    )


def select_action(state: dict[str, Any]) -> dict[str, Any]:
    blockers = runner_blockers(state.get("runners", {}).get("data"))
    if blockers:
        return {
            "kind": "blocker_report",
            "reason": "runner fleet unavailable",
            "blockers": blockers,
            "prompt": _blocker_prompt(blockers),
        }

    snapshot = state.get("operator_snapshot", {}).get("data")
    conflicts = _lane_conflicts(snapshot)
    if conflicts:
        blocker_text = [str(item) for item in conflicts]
        return {
            "kind": "blocker_report",
            "reason": "active lane conflicts",
            "blockers": blocker_text,
            "prompt": _blocker_prompt(blocker_text),
        }

    prs = state.get("open_prs", {}).get("data")
    if not isinstance(prs, list):
        prs = []
    checks_by_pr = state.get("checks_by_pr") or {}

    for pr in prs:
        summary = checks_by_pr.get(str(pr.get("number")), {}).get("summary", {})
        if _pr_is_clean(pr) and not pr.get("isDraft") and summary.get("green"):
            return {
                "kind": "merge_ready_prompt",
                "reason": "first mergeable non-draft PR has green required checks",
                "target": _pr_identity(pr),
                "prompt": _merge_ready_prompt(pr),
            }

    for pr in prs:
        summary = checks_by_pr.get(str(pr.get("number")), {}).get("summary", {})
        if _pr_is_clean(pr) and pr.get("isDraft") and summary.get("green"):
            return {
                "kind": "draft_gate_preparation",
                "reason": "first mergeable draft PR has green required checks",
                "target": _pr_identity(pr),
                "prompt": _draft_gate_prompt(pr),
            }

    for pr in prs:
        summary = checks_by_pr.get(str(pr.get("number")), {}).get("summary", {})
        failures = summary.get("real_failures") or []
        if _pr_is_clean(pr) and failures:
            return {
                "kind": "failing_check_repair_prompt",
                "reason": "first mergeable PR has a real required-check failure",
                "target": _pr_identity(pr),
                "failure": failures[0],
                "prompt": _failing_check_prompt(pr, failures[0]),
            }

    publisher = state.get("publisher", {}).get("data")
    if isinstance(publisher, dict):
        verdict = str(publisher.get("verdict") or publisher.get("status") or "").lower()
        if verdict and verdict not in {"ready", "fresh", "ok"}:
            return {
                "kind": "outbox_publication_refresh_prompt",
                "reason": "publisher/outbox health needs bounded refresh",
                "target": {"publisher_verdict": verdict},
                "prompt": _publisher_prompt(state),
            }

    for issue in _health_issues(snapshot):
        issue_type = str(issue.get("type") or "")
        if "lane_missing" in issue_type or "owner" in issue_type:
            return {
                "kind": "stale_owner_coordination_prompt",
                "reason": "operator snapshot reports stale or incomplete owner state",
                "target": {"issue": issue_type, "session": issue.get("session")},
                "prompt": _stale_owner_prompt(issue),
            }

    final_blockers = ["no safe mutation candidate found"]
    return {
        "kind": "blocker_report",
        "reason": "no safe candidate",
        "blockers": final_blockers,
        "prompt": _blocker_prompt(final_blockers),
    }


def gather_state(repo_root: Path, *, max_prs: int, runner: CommandRunner) -> dict[str, Any]:
    state: dict[str, Any] = {
        "generated_at": utc_now(),
        "repo_root": str(repo_root),
        "forbidden_actions": FORBIDDEN_ACTIONS,
    }

    status = runner.run(["git", "status", "--short", "--branch", "--untracked-files=all"])
    head = runner.run(["git", "rev-parse", "HEAD"])
    origin_main = runner.run(["git", "rev-parse", "origin/main"])
    state["git"] = {
        "status_ok": status.ok,
        "status": status.stdout,
        "head": head.stdout.strip() if head.ok else None,
        "origin_main": origin_main.stdout.strip() if origin_main.ok else None,
    }

    state["runners"] = _probe_json(
        runner,
        ["gh", "api", "/repos/synaptent/aragora/actions/runners", "--paginate"],
        timeout=45,
    )
    state["operator_snapshot"] = _probe_json(
        runner,
        [
            "python3",
            "scripts/agent_bridge.py",
            "operator-snapshot",
            "--json",
            "--summary-only",
        ],
        timeout=60,
    )
    state["active_sessions"] = _probe_json(
        runner,
        [
            "python3",
            "scripts/list_active_agent_sessions.py",
            "--json",
            "--conflicts-only",
            "--codex-session-scan-limit",
            "120",
        ],
        timeout=90,
    )
    state["work_robot"] = _probe_json(
        runner,
        ["python3", "-m", "aragora.cli.main", "work", "robot", "--json"],
        timeout=120,
    )
    state["open_prs"] = _probe_json(
        runner,
        [
            "gh",
            "pr",
            "list",
            "--state",
            "open",
            "--json",
            "number,title,isDraft,headRefName,headRefOid,mergeable,mergeStateStatus,updatedAt,url",
            "--limit",
            str(max_prs),
        ],
        timeout=60,
    )
    state["publisher"] = _probe_json(
        runner,
        ["python3", "scripts/publisher_freshness_check.py", "--json", "--summary-only"],
        timeout=60,
    )
    state["github_health"] = _probe_json(
        runner,
        ["python3", "scripts/github_cli_health.py", "--json", "--timeout-seconds", "8"],
        timeout=30,
    )

    prs = state["open_prs"].get("data")
    checks_by_pr: dict[str, Any] = {}
    if isinstance(prs, list):
        for pr in prs[:max_prs]:
            number = pr.get("number")
            if not number:
                continue
            checks = _probe_json(
                runner,
                required_checks_args(int(number)),
                timeout=60,
            )
            rows = checks.get("data")
            summary = check_summary(rows if isinstance(rows, list) else [])
            checks_by_pr[str(number)] = {
                "ok": checks["ok"],
                "summary": summary,
                "stderr": checks.get("stderr"),
            }
    state["checks_by_pr"] = checks_by_pr
    return state


def append_ledger(path: Path, packet: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(packet, sort_keys=True) + "\n")


def build_packet(state: dict[str, Any]) -> dict[str, Any]:
    action = select_action(state)
    operator_snapshot = state.get("operator_snapshot", {}).get("data") or {}
    operator_health = operator_snapshot.get("health") if isinstance(operator_snapshot, dict) else {}
    if not isinstance(operator_health, dict):
        operator_health = {}
    return {
        "version": 1,
        "generated_at": state["generated_at"],
        "repo_root": state["repo_root"],
        "git": state.get("git", {}),
        "forbidden_actions": FORBIDDEN_ACTIONS,
        "action": action,
        "summary": {
            "runner_blockers": runner_blockers(state.get("runners", {}).get("data")),
            "open_pr_count": len(state.get("open_prs", {}).get("data") or []),
            "operator_health_ok": operator_health.get("ok"),
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--max-prs", type=int, default=10)
    parser.add_argument("--ledger", type=Path, default=Path(".aragora/overnight/conductor.jsonl"))
    parser.add_argument("--write-ledger", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo.resolve()
    runner = CommandRunner(repo_root)
    state = gather_state(repo_root, max_prs=args.max_prs, runner=runner)
    packet = build_packet(state)

    if args.write_ledger:
        ledger = args.ledger
        if not ledger.is_absolute():
            ledger = repo_root / ledger
        append_ledger(ledger, packet)

    if args.json:
        print(json.dumps(packet, indent=2, sort_keys=True))
    else:
        action = packet["action"]
        print(f"overnight conductor selected: {action['kind']}")
        print(f"reason: {action['reason']}")
        if action.get("target"):
            print(f"target: {json.dumps(action['target'], sort_keys=True)}")
        print("\nnext prompt:\n")
        print(action["prompt"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
