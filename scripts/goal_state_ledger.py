#!/usr/bin/env python3
"""Persist resumable goal state without adding a new autonomous conductor."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

SCHEMA_VERSION = "aragora-goal-state/0.1"
DEFAULT_LEDGER_ROOT = Path(".aragora") / "goal-state"
SUPPORTED_GOALS = {
    "merge_authorized_prs": (
        "Read-only dry-run over existing settle_one_pr/merge-packet gates. "
        "Reports one next safe merge action, but never executes it."
    )
}


class CommandRunner(Protocol):
    def __call__(
        self,
        command: list[str],
        *,
        cwd: Path,
        timeout: int,
    ) -> subprocess.CompletedProcess[str]: ...


def run_command(
    command: list[str],
    *,
    cwd: Path,
    timeout: int,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _load_json_output(process: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    if process.returncode != 0:
        raise RuntimeError(
            "settle_one_pr failed "
            f"(exit={process.returncode}): {process.stderr.strip() or process.stdout.strip()}"
        )
    try:
        payload = json.loads(process.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"settle_one_pr returned non-JSON output: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("settle_one_pr returned a non-object JSON payload")
    return payload


def _settle_one_pr_report(
    *,
    cwd: Path,
    runner: CommandRunner,
    limit: int,
    repo: str | None,
    timeout: int,
) -> dict[str, Any]:
    command = ["python3", "scripts/settle_one_pr.py", "--json", "--limit", str(limit)]
    if repo:
        command.extend(["--repo", repo])
    process = runner(command, cwd=cwd, timeout=timeout)
    return _load_json_output(process)


def _next_action_for_merge_authorized_prs(report: dict[str, Any]) -> dict[str, Any]:
    status = str(report.get("status") or "unknown")
    blockers = [str(blocker) for blocker in report.get("blockers") or []]
    selected_pr = report.get("selected_pr")
    head_sha = report.get("head_sha")
    suggested_commands = [str(command) for command in report.get("suggested_commands") or []]

    if (
        status == "packet_authorized_dry_run"
        and not blockers
        and selected_pr is not None
        and head_sha
    ):
        return {
            "kind": "merge_authorized_pr",
            "safe_to_execute": False,
            "reason": (
                "Existing Aragora gates report this PR authorized for normal protected "
                "squash, but goal_state_ledger is read-only and does not execute merges."
            ),
            "pr": selected_pr,
            "head_sha": head_sha,
            "commands": suggested_commands[:1],
            "blockers": [],
        }

    if status == "ready_for_minimum_evidence":
        return {
            "kind": "collect_minimum_evidence",
            "safe_to_execute": False,
            "reason": "A candidate exists, but existing gates require evidence before merge.",
            "pr": selected_pr,
            "head_sha": head_sha,
            "commands": suggested_commands,
            "blockers": blockers,
        }

    if status == "needs_packet_rerun":
        return {
            "kind": "rerun_merge_packet",
            "safe_to_execute": False,
            "reason": "The existing settlement primitive needs a refreshed merge packet.",
            "pr": selected_pr,
            "head_sha": head_sha,
            "commands": suggested_commands,
            "blockers": blockers,
        }

    return {
        "kind": "blocked" if blockers else "no_candidate",
        "safe_to_execute": False,
        "reason": ("No dry-run merge action is currently safe under existing Aragora gates."),
        "pr": selected_pr,
        "head_sha": head_sha,
        "commands": [],
        "blockers": blockers or [f"settle_one_pr status={status}"],
    }


def _state_paths(ledger_root: Path, goal_id: str) -> tuple[Path, Path]:
    goal_dir = ledger_root / goal_id
    return goal_dir / "latest.json", goal_dir / "ledger.jsonl"


def _compact_source_report(report: dict[str, Any]) -> dict[str, Any]:
    """Keep resumable state small; do not persist raw command transcripts."""

    keys = [
        "version",
        "generated_at",
        "dry_run",
        "packet_summary",
        "selected_pr",
        "head_sha",
        "status",
        "blockers",
        "evidence",
        "checks",
        "load_warnings",
        "policy_exclusions",
        "validation",
        "suggested_commands",
    ]
    return {key: report[key] for key in keys if key in report}


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def write_goal_state(payload: dict[str, Any], *, ledger_root: Path, goal_id: str) -> None:
    latest_path, ledger_path = _state_paths(ledger_root, goal_id)
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(latest_path, payload)
    with ledger_path.open("a", encoding="utf-8") as receipt_file:
        receipt_file.write(json.dumps(payload, sort_keys=True) + "\n")


def build_goal_state(
    *,
    goal_id: str,
    cwd: Path,
    ledger_root: Path = DEFAULT_LEDGER_ROOT,
    runner: CommandRunner = run_command,
    write_ledger: bool = False,
    limit: int = 100,
    repo: str | None = None,
    timeout: int = 180,
) -> dict[str, Any]:
    if goal_id not in SUPPORTED_GOALS:
        supported = ", ".join(sorted(SUPPORTED_GOALS))
        raise ValueError(f"unsupported goal {goal_id!r}; supported goals: {supported}")

    cwd = cwd.resolve()
    ledger_root = ledger_root if ledger_root.is_absolute() else cwd / ledger_root
    latest_path, ledger_path = _state_paths(ledger_root, goal_id)

    settle_report = _settle_one_pr_report(
        cwd=cwd,
        runner=runner,
        limit=limit,
        repo=repo,
        timeout=timeout,
    )
    next_action = _next_action_for_merge_authorized_prs(settle_report)

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _utc_now(),
        "goal_id": goal_id,
        "goal_description": SUPPORTED_GOALS[goal_id],
        "cwd": str(cwd),
        "dry_run": True,
        "write_ledger": write_ledger,
        "state_path": str(latest_path),
        "receipt_ledger_path": str(ledger_path),
        "selected_pr": settle_report.get("selected_pr"),
        "head_sha": settle_report.get("head_sha"),
        "source_status": settle_report.get("status"),
        "source_blockers": settle_report.get("blockers") or [],
        "next_action": next_action,
        "resume_prompt": settle_report.get("recursive_best_next_prompt") or "",
        "source_report": _compact_source_report(settle_report),
    }

    if write_ledger:
        write_goal_state(payload, ledger_root=ledger_root, goal_id=goal_id)

    return payload


def _human_summary(payload: dict[str, Any]) -> str:
    action = payload["next_action"]
    lines = [
        f"goal: {payload['goal_id']}",
        f"status: {payload['source_status']}",
        f"next_action: {action['kind']}",
        f"pr: {action.get('pr')}",
        f"head: {action.get('head_sha')}",
    ]
    blockers = action.get("blockers") or []
    if blockers:
        lines.append("blockers:")
        lines.extend(f"- {blocker}" for blocker in blockers)
    commands = action.get("commands") or []
    if commands:
        lines.append("dry-run command:")
        lines.extend(f"- {command}" for command in commands)
    if payload.get("write_ledger"):
        lines.append(f"state_path: {payload['state_path']}")
        lines.append(f"receipt_ledger_path: {payload['receipt_ledger_path']}")
    return "\n".join(lines)


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Record resumable Aragora goal state around existing primitives. "
            "The first goal is read-only and never executes PR mutation."
        )
    )
    parser.add_argument(
        "--goal",
        choices=sorted(SUPPORTED_GOALS),
        default="merge_authorized_prs",
        help="Goal id to evaluate.",
    )
    parser.add_argument(
        "--cwd",
        type=Path,
        default=Path.cwd(),
        help="Repository root to run existing Aragora primitives from.",
    )
    parser.add_argument(
        "--ledger-root",
        type=Path,
        default=DEFAULT_LEDGER_ROOT,
        help="Directory for latest.json and ledger.jsonl when --write-ledger is set.",
    )
    parser.add_argument(
        "--write-ledger",
        action="store_true",
        help="Persist latest.json and append a JSONL receipt. Does not execute actions.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Candidate limit passed to scripts/settle_one_pr.py.",
    )
    parser.add_argument("--repo", help="Optional GitHub repo passed to settle_one_pr.py.")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    try:
        payload = build_goal_state(
            goal_id=args.goal,
            cwd=args.cwd,
            ledger_root=args.ledger_root,
            write_ledger=args.write_ledger,
            limit=args.limit,
            repo=args.repo,
            timeout=args.timeout,
        )
    except Exception as exc:
        if args.json:
            print(json.dumps({"ok": False, "error": str(exc)}, indent=2, sort_keys=True))
        else:
            print(f"goal_state_ledger: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(_human_summary(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
