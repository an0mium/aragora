#!/usr/bin/env python3
"""Run queue commands behind a root hygiene guard.

The wrapper is intentionally conservative: it refuses to run the requested
command from a dirty checkout, snapshots branch/head/status before and after
the command, and reports checkout drift instead of letting queue work continue
from ambiguous root state.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
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
DEFAULT_PROCESS_PATTERNS = (
    "work robot",
    "review-queue merge-packet",
    "settle_one_pr",
    "pr_check_followup",
)


@dataclass(frozen=True)
class CommandResult:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class RootSnapshot:
    branch: str
    head: str
    status_lines: list[str]
    dirty_paths: list[str]
    dirty: bool


def _run(args: list[str], *, cwd: Path, timeout: int = 120) -> CommandResult:
    proc = subprocess.run(
        args,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return CommandResult(
        command=args,
        returncode=proc.returncode,
        stdout=proc.stdout.strip(),
        stderr=proc.stderr.strip(),
    )


def _repo_root(cwd: Path) -> Path:
    result = _run(["git", "rev-parse", "--show-toplevel"], cwd=cwd)
    if result.returncode != 0 or not result.stdout:
        return cwd
    return Path(result.stdout)


def _snapshot(cwd: Path) -> RootSnapshot:
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd).stdout
    head = _run(["git", "rev-parse", "HEAD"], cwd=cwd).stdout
    status = _run(["git", "status", "--short", "--branch", "--untracked-files=all"], cwd=cwd)
    lines = [line for line in status.stdout.splitlines() if line]
    dirty_paths = [line[3:].strip() for line in lines if not line.startswith("##")]
    return RootSnapshot(
        branch=branch,
        head=head,
        status_lines=lines,
        dirty_paths=dirty_paths,
        dirty=bool(dirty_paths),
    )


def _owner_attribution(cwd: Path, *, branch: str, pr: int | None) -> dict[str, Any]:
    script = cwd / "scripts" / "identify_lane_owner.py"
    if not script.exists():
        return {"available": False, "reason": f"{script} not found"}
    lookups: list[dict[str, Any]] = []
    selectors: list[list[str]] = [["--branch", branch]]
    if pr is not None:
        selectors.append(["--pr", str(pr)])
    for selector in selectors:
        result = _run(["python3", str(script), *selector, "--json"], cwd=cwd)
        parsed: Any | None = None
        if result.stdout:
            try:
                parsed = json.loads(result.stdout)
            except json.JSONDecodeError:
                parsed = None
        lookups.append(
            {
                "selector": selector,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "json": parsed,
            }
        )
    return {"available": True, "lookups": lookups}


def _process_attribution(patterns: list[str]) -> dict[str, Any]:
    result = _run(["ps", "-axo", "pid,ppid,etime,state,time,%cpu,command"], cwd=Path.cwd())
    matches: list[str] = []
    current_pid = os.getpid()
    parent_pid = os.getppid()
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split(None, 2)
        pid = int(parts[0]) if parts and parts[0].isdigit() else None
        if pid in {current_pid, parent_pid}:
            continue
        if any(pattern in stripped for pattern in patterns):
            matches.append(stripped)
    return {
        "patterns": patterns,
        "matches": matches,
        "command": result.command,
        "returncode": result.returncode,
    }


def _read_only_sequence(pr: int | None) -> str:
    if pr is None:
        return "rerun mailbox/root/owner checks, then run `python3 -m aragora.cli.main work robot --json`."
    return (
        f"rerun mailbox/owner checks for #{pr}, `gh pr view {pr}`, "
        f"`gh pr checks {pr} --required`, "
        f"`python3 -m aragora.cli.main review-queue merge-packet --pr {pr} --json`, and "
        f"`python3 scripts/settle_one_pr.py --pr {pr} --json`."
    )


def _next_prompt(
    *,
    status: str,
    before: RootSnapshot,
    after: RootSnapshot | None,
    pr: int | None,
    expected_head: str | None,
) -> str:
    head_text = f" at exact head `{expected_head}`" if expected_head else ""
    if status == "blocked_dirty_root":
        body = (
            f"Goal: resolve root hygiene for `{before.branch}`, then continue only after root is "
            f"clean or by using a clean isolated worktree. Dirty paths: "
            f"{', '.join(before.dirty_paths) or 'unknown'}. Do not reset, clean, stash, switch, "
            "or commit without explicit preserve/revert/switch authorization."
        )
    elif status == "blocked_root_drift":
        body = (
            f"Goal: resolve root drift after a guarded queue command. Before was "
            f"`{before.branch}` @ `{before.head}`; after is "
            f"`{after.branch if after else 'unknown'}` @ `{after.head if after else 'unknown'}`. "
            "Do not continue queue work until the drift is attributed and root is clean."
        )
    else:
        body = (
            f"Goal: continue read-only queue routing{head_text}. Start from live truth and "
            f"{_read_only_sequence(pr)} Do not mutate PR state without separate explicit "
            "authorization."
        )
    return (
        "Check your mailbox for steering messages first, read-only/no receipt if possible. "
        "Start from live truth in /Users/armand/Development/aragora. Do not trust prior "
        f"transcript state.\n\n{body}\n\n{INCREMENTAL_PROGRESS_SENTENCE} "
        f"{META_AUTOMATION_SENTENCE}"
    )


def _drift(before: RootSnapshot, after: RootSnapshot) -> list[str]:
    reasons: list[str] = []
    if before.branch != after.branch:
        reasons.append(f"branch drift: {before.branch} -> {after.branch}")
    if before.head != after.head:
        reasons.append(f"head drift: {before.head} -> {after.head}")
    if after.dirty:
        reasons.append(f"dirty root after command: {', '.join(after.dirty_paths)}")
    return reasons


def run_guard(args: argparse.Namespace) -> dict[str, Any]:
    cwd = _repo_root(Path(args.cwd).resolve() if args.cwd else Path.cwd())
    command = list(args.command or [])
    if command and command[0] == "--":
        command = command[1:]
    before = _snapshot(cwd)
    process_patterns = list(args.process_pattern or DEFAULT_PROCESS_PATTERNS)
    attribution = {
        "owner": _owner_attribution(cwd, branch=before.branch, pr=args.pr),
        "processes": _process_attribution(process_patterns),
    }
    if before.dirty:
        return {
            "version": "root_guarded_queue.v1",
            "status": "blocked_dirty_root",
            "before": asdict(before),
            "after": None,
            "drift_reasons": [],
            "attribution": attribution,
            "command_result": None,
            "next_prompt": _next_prompt(
                status="blocked_dirty_root",
                before=before,
                after=None,
                pr=args.pr,
                expected_head=args.expected_head,
            ),
        }
    command_result: CommandResult | None = None
    if command:
        command_result = _run(command, cwd=cwd, timeout=args.timeout)
    after = _snapshot(cwd)
    drift_reasons = _drift(before, after)
    if drift_reasons:
        status = "blocked_root_drift"
    elif command_result and command_result.returncode != 0:
        status = "command_failed"
    else:
        status = "completed"
    return {
        "version": "root_guarded_queue.v1",
        "status": status,
        "before": asdict(before),
        "after": asdict(after),
        "drift_reasons": drift_reasons,
        "attribution": attribution,
        "command_result": asdict(command_result) if command_result else None,
        "next_prompt": _next_prompt(
            status=status,
            before=before,
            after=after,
            pr=args.pr,
            expected_head=args.expected_head,
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument("--cwd", help="repo checkout to guard; defaults to current directory")
    parser.add_argument("--pr", type=int, help="PR number for owner lookup and next prompt")
    parser.add_argument("--expected-head", help="expected PR head for generated prompts")
    parser.add_argument(
        "--process-pattern",
        action="append",
        help="process substring to include in attribution; repeatable",
    )
    parser.add_argument("--timeout", type=int, default=600, help="command timeout in seconds")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="queue command after --")
    return parser


def _print_text(report: dict[str, Any]) -> None:
    print(f"status: {report['status']}")
    print(f"before: {report['before']['branch']} @ {report['before']['head']}")
    if report.get("after"):
        print(f"after: {report['after']['branch']} @ {report['after']['head']}")
    if report.get("drift_reasons"):
        print("drift:")
        for reason in report["drift_reasons"]:
            print(f"- {reason}")
    if report["before"].get("dirty_paths"):
        print("dirty paths:")
        for path in report["before"]["dirty_paths"]:
            print(f"- {path}")
    print("\nnext prompt:")
    print(report["next_prompt"])


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    report = run_guard(args)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_text(report)
    if report["status"] in {"blocked_dirty_root", "blocked_root_drift"}:
        return 2
    if report["status"] == "command_failed":
        command = report.get("command_result") or {}
        return int(command.get("returncode") or 1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
