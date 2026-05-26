#!/usr/bin/env python3
"""Read-only follow-up packet for exact-head Tier 4 settlement work.

The helper intentionally reports live state and suggested next prompts only. It
does not merge, push, create worktrees, rerun CI, apply settlement, or clean up.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[1]
CONVERGENCE_SENTENCE = (
    "If the prompt above accomplishes no incremental progress make the next prompt one "
    "that does, include this sentence in all subsequent prompts to ensure they converge "
    "towards prompts that make incremental progress."
)
AUTOMATION_SENTENCE = (
    "If any of the work involved can be better automated and made more autonomous by "
    "improving aragora tooling at a meta level create a plan to do that instead, "
    "include this sentence in all subsequent prompts to ensure they converge towards "
    "automating and making more autonomous all steps at a meta level where this makes "
    "sense for the repo and project."
)

CommandRunner = Callable[[list[str], Path], subprocess.CompletedProcess[str]]


def _default_runner(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    return subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=180,
        env=env,
    )


def _run_text(command: list[str], cwd: Path, runner: CommandRunner) -> dict[str, Any]:
    try:
        result = runner(command, cwd)
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "command": command,
            "returncode": 127,
            "stdout": "",
            "stderr": str(exc),
        }
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout or "",
        "stderr": result.stderr or "",
    }


def _parse_json(text: str) -> Any:
    stripped = text.strip()
    if not stripped:
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return {"raw": stripped}


def _run_json(command: list[str], cwd: Path, runner: CommandRunner) -> dict[str, Any]:
    result = _run_text(command, cwd, runner)
    payload = _parse_json(str(result.get("stdout") or ""))
    result["payload"] = payload
    return result


def _status_packet(repo_root: Path, runner: CommandRunner) -> dict[str, Any]:
    result = _run_text(
        ["git", "status", "--short", "--branch", "--untracked-files=all"],
        repo_root,
        runner,
    )
    lines = [line for line in str(result["stdout"]).splitlines() if line.strip()]
    dirty_paths: list[str] = []
    for line in lines:
        if line.startswith("##"):
            continue
        dirty_paths.append(line[3:].strip() if len(line) > 3 else line.strip())
    return {
        "dirty": bool(dirty_paths),
        "dirty_paths": dirty_paths,
        "status": lines,
        "returncode": result["returncode"],
    }


def _normalize_branch(branch: str) -> str:
    text = branch.strip()
    for prefix in ("refs/heads/", "origin/"):
        if text.startswith(prefix):
            return text[len(prefix) :]
    return text


def _remote_branch_packet(
    repo_root: Path,
    *,
    branch: str,
    expected_oid: str | None,
    runner: CommandRunner,
) -> dict[str, Any]:
    normalized = _normalize_branch(branch)
    result = _run_text(
        ["git", "ls-remote", "origin", f"refs/heads/{normalized}"],
        repo_root,
        runner,
    )
    line = next((item for item in result["stdout"].splitlines() if item.strip()), "")
    remote_oid = line.split()[0] if line else None
    matches_expected = remote_oid == expected_oid if expected_oid else None
    return {
        "input": branch,
        "branch": normalized,
        "remote_ref": f"refs/heads/{normalized}",
        "published": bool(remote_oid),
        "remote_oid": remote_oid,
        "expected_oid": expected_oid,
        "matches_expected": matches_expected,
        "returncode": result["returncode"],
        "stderr": result["stderr"],
    }


def _worktree_list(repo_root: Path, runner: CommandRunner) -> list[dict[str, str]]:
    result = _run_text(["git", "worktree", "list", "--porcelain"], repo_root, runner)
    records: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for line in result["stdout"].splitlines():
        if not line.strip():
            if current:
                records.append(current)
                current = {}
            continue
        key, _, value = line.partition(" ")
        current[key] = value
    if current:
        records.append(current)
    return records


def _repair_worktree_path(
    repo_root: Path,
    *,
    branch: str,
    explicit_path: Path | None,
    runner: CommandRunner,
) -> Path | None:
    if explicit_path is not None:
        return explicit_path
    normalized = _normalize_branch(branch)
    for record in _worktree_list(repo_root, runner):
        if record.get("branch") == f"refs/heads/{normalized}":
            return Path(record["worktree"])
    return None


def _repair_worktree_packet(
    repo_root: Path,
    *,
    branch: str,
    expected_oid: str | None,
    explicit_path: Path | None,
    runner: CommandRunner,
) -> dict[str, Any]:
    path = _repair_worktree_path(
        repo_root,
        branch=branch,
        explicit_path=explicit_path,
        runner=runner,
    )
    if path is None:
        return {
            "found": False,
            "path": None,
            "dirty": None,
            "head": None,
            "matches_expected": None,
        }
    status = _status_packet(path, runner)
    head_result = _run_text(["git", "rev-parse", "HEAD"], path, runner)
    head = str(head_result["stdout"]).strip() or None
    return {
        "found": True,
        "path": str(path),
        "dirty": status["dirty"],
        "dirty_paths": status["dirty_paths"],
        "status": status["status"],
        "head": head,
        "expected_oid": expected_oid,
        "matches_expected": head == expected_oid if expected_oid else None,
    }


def _tail_lines(text: str, *, limit: int = 20) -> list[str]:
    lines = [line for line in text.splitlines() if line.strip()]
    return lines[-limit:]


def _settlement_check_packet(
    *,
    pr: int,
    head: str,
    repair_cwd: Path | None,
    runner: CommandRunner,
) -> dict[str, Any]:
    if repair_cwd is None:
        return {
            "ran": False,
            "ok": False,
            "reason": "repair worktree not found",
            "blockers": ["repair worktree not found"],
        }
    command = [
        sys.executable,
        "scripts/settle_tier4_pr.py",
        "--check",
        "--pr",
        str(pr),
        "--head",
        head,
        "--json",
    ]
    result = _run_json(command, repair_cwd, runner)
    payload = result.get("payload")
    gate = payload.get("gate") if isinstance(payload, dict) else None
    blockers = gate.get("blockers") if isinstance(gate, dict) else None
    return {
        "ran": True,
        "command": command,
        "returncode": result["returncode"],
        "payload": payload,
        "ok": bool(isinstance(gate, dict) and gate.get("ok")),
        "blockers": blockers if isinstance(blockers, list) else [],
        "stderr": result["stderr"],
    }


def _focused_tests_packet(
    *,
    repair_cwd: Path | None,
    runner: CommandRunner,
) -> dict[str, Any]:
    if repair_cwd is None:
        return {
            "ran": False,
            "ok": False,
            "reason": "repair worktree not found",
        }
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-p",
        "no:cacheprovider",
        "tests/scripts/test_settle_tier4_pr.py",
    ]
    result = _run_text(command, repair_cwd, runner)
    return {
        "ran": True,
        "command": command,
        "returncode": result["returncode"],
        "ok": result["returncode"] == 0,
        "stdout_tail": _tail_lines(str(result["stdout"])),
        "stderr_tail": _tail_lines(str(result["stderr"])),
    }


def _prompt_for(packet: dict[str, Any]) -> str:
    pr = packet["pr"]
    head = packet["head"]
    repair = packet["repair_branch"]
    repair_branch = repair["branch"]
    repair_oid = repair.get("remote_oid") or packet.get("repair_head") or "<repair-head>"
    root_dirty = packet["root"]["dirty"]
    pr_state = packet["pr_state"].get("payload") or {}
    mergeable = pr_state.get("mergeable")
    merge_state = pr_state.get("mergeStateStatus")
    gate = packet["validation"]["settlement_check"]
    blockers = gate.get("blockers") or []

    if root_dirty:
        next_action = "First classify and preserve the root dirty state before any mutation."
    elif blockers or merge_state in {"DIRTY", "CONFLICTING"} or mergeable == "CONFLICTING":
        next_action = "Diagnose only the live branch conflict or merge-packet blocker."
    elif gate.get("ok"):
        next_action = (
            "Prepare exact-head settlement apply only if a fresh prompt explicitly authorizes it."
        )
    else:
        next_action = (
            "Re-run the read-only settlement follow-up helper and report the exact blocker."
        )

    return (
        "Start from live repo truth in `/Users/armand/Development/aragora`. "
        "Do not trust prior transcript state. Check Aragora operator-steering mailbox "
        "before lane work. "
        f"Goal: continue #{pr} settlement follow-up from repaired branch "
        f"`origin/{repair_branch}` at `{repair_oid}` without applying settlement. "
        "Do not merge, push, cleanup, rerun CI, apply settlement, or start unrelated queue work "
        "unless explicitly authorized. "
        f"First run `python3 scripts/settlement_followup.py --pr {pr} --head {head} "
        f"--repair-branch origin/{repair_branch}"
        + (f" --repair-head {repair_oid}" if repair_oid != "<repair-head>" else "")
        + " --json --prompt` and report root dirty paths, bridge health, PR mergeability, "
        "repair-branch publish state, focused validation, and exact apply blockers. "
        f"{next_action} "
        f"{CONVERGENCE_SENTENCE} {AUTOMATION_SENTENCE}"
    )


def build_followup_packet(
    *,
    pr: int,
    head: str,
    repair_branch: str,
    repair_head: str | None = None,
    repair_worktree: Path | None = None,
    repo_root: Path = DEFAULT_REPO_ROOT,
    include_prompt: bool = False,
    run_focused_tests: bool = True,
    runner: CommandRunner = _default_runner,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    owner = _run_json(
        [sys.executable, "scripts/identify_lane_owner.py", "--pr", str(pr), "--json"],
        repo_root,
        runner,
    )
    root = _status_packet(repo_root, runner)
    bridge_health = _run_json(
        [sys.executable, "scripts/agent_bridge.py", "--json", "health"],
        repo_root,
        runner,
    )
    pr_state = _run_json(
        [
            "gh",
            "pr",
            "view",
            str(pr),
            "--json",
            "number,state,isDraft,headRefName,headRefOid,mergeable,mergeStateStatus,url",
        ],
        repo_root,
        runner,
    )
    remote = _remote_branch_packet(
        repo_root,
        branch=repair_branch,
        expected_oid=repair_head,
        runner=runner,
    )
    worktree = _repair_worktree_packet(
        repo_root,
        branch=remote["branch"],
        expected_oid=repair_head,
        explicit_path=repair_worktree,
        runner=runner,
    )
    repair_cwd = Path(worktree["path"]) if worktree.get("path") else None
    settlement_check = _settlement_check_packet(
        pr=pr,
        head=head,
        repair_cwd=repair_cwd,
        runner=runner,
    )
    focused_tests = (
        _focused_tests_packet(repair_cwd=repair_cwd, runner=runner)
        if run_focused_tests
        else {"ran": False, "ok": None, "reason": "disabled"}
    )
    packet = {
        "pr": pr,
        "head": head,
        "repair_head": repair_head,
        "owner": owner.get("payload"),
        "mailbox": _mailbox_summary(owner.get("payload")),
        "root": root,
        "bridge_health": bridge_health.get("payload"),
        "pr_state": pr_state,
        "repair_branch": remote,
        "repair_worktree": worktree,
        "validation": {
            "settlement_check": settlement_check,
            "focused_tests": focused_tests,
        },
        "apply_blockers": settlement_check.get("blockers", []),
    }
    if include_prompt:
        packet["next_prompt"] = _prompt_for(packet)
    return packet


def _mailbox_summary(owner_payload: Any) -> dict[str, Any]:
    if not isinstance(owner_payload, dict):
        return {}
    keys = (
        "owner_session",
        "steering_inbox_path",
        "pending_message_count",
        "unread_message_count",
        "read_receipt_count",
        "latest_read_receipt",
        "status",
        "last_steering_outcome",
    )
    return {key: owner_payload.get(key) for key in keys}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pr", type=int, required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--repair-branch", required=True)
    parser.add_argument("--repair-head")
    parser.add_argument("--repair-worktree", type=Path)
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--skip-focused-tests", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--prompt", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    packet = build_followup_packet(
        pr=args.pr,
        head=args.head,
        repair_branch=args.repair_branch,
        repair_head=args.repair_head,
        repair_worktree=args.repair_worktree,
        repo_root=args.repo_root,
        include_prompt=args.prompt,
        run_focused_tests=not args.skip_focused_tests,
    )
    if args.json:
        print(json.dumps(packet, indent=2, sort_keys=True))
    else:
        print(f"PR #{packet['pr']} settlement follow-up")
        print(f"root_dirty={packet['root']['dirty']}")
        pr_payload = packet["pr_state"].get("payload") or {}
        print(
            "pr_state="
            f"{pr_payload.get('headRefOid')} {pr_payload.get('mergeable')} "
            f"{pr_payload.get('mergeStateStatus')}"
        )
        print(f"bridge_ok={(packet.get('bridge_health') or {}).get('ok')}")
        print(f"repair_published={packet['repair_branch']['published']}")
        print(f"settlement_ok={packet['validation']['settlement_check'].get('ok')}")
        for blocker in packet["apply_blockers"]:
            print(f"- {blocker}")
        if args.prompt:
            print()
            print(packet["next_prompt"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
