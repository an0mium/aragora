#!/usr/bin/env python3
"""Build a concise owner-aware next prompt from live Aragora coordination state."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from collections.abc import Callable
from collections.abc import Sequence
from pathlib import Path
from typing import Any

DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_RELATIVE_PATH = Path(".aragora") / "agent-bridge" / "lanes.json"
ACTIVE_STATUSES = {
    "active",
    "running",
    "pending",
    "queued",
    "claimed",
    "waiting_for_steering",
    "acknowledged",
    "working",
    "blocked",
}
SENSITIVE_KEYS = {
    "messages",
    "prompt",
    "raw_prompt",
    "raw_transcript",
    "transcript_file",
    "transcript_path",
}
CommandRunner = Callable[[list[str]], subprocess.CompletedProcess[str]]
CONVERGENCE_SENTENCE = (
    "If the prompt above accomplishes no incremental progress make the next prompt one "
    "that does, include this sentence in all subsequent prompts to ensure they converge "
    "towards prompts that make incremental progress."
)


def _read_lanes(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(payload, list):
        return []
    return [row for row in payload if isinstance(row, dict)]


def _find_lane(
    lanes: list[dict[str, Any]],
    *,
    lane_id: str | None = None,
    pr: int | None = None,
    branch: str | None = None,
) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    for row in lanes:
        if lane_id and str(row.get("lane_id") or "") == lane_id:
            candidates.append(row)
        elif pr is not None and row.get("pr_number") == pr:
            candidates.append(row)
        elif branch and str(row.get("branch") or "") == branch:
            candidates.append(row)
    if not candidates:
        return None
    active = [row for row in candidates if str(row.get("status") or "") in ACTIVE_STATUSES]
    return active[0] if active else candidates[0]


def _sanitize(value: Any) -> Any:
    """Drop transcript/prompt-bearing fields from live-truth packets."""

    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if str(key).lower() in SENSITIVE_KEYS:
                continue
            out[str(key)] = _sanitize(item)
        return out
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    return value


def _default_runner(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command, cwd=DEFAULT_REPO_ROOT, capture_output=True, text=True, timeout=120
    )


def _repo_runner(repo_root: Path) -> CommandRunner:
    def run(command: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(command, cwd=repo_root, capture_output=True, text=True, timeout=120)

    return run


def _json_or_empty(result: subprocess.CompletedProcess[str]) -> Any:
    if result.returncode != 0:
        return {"error": result.stderr.strip(), "returncode": result.returncode}
    text = (result.stdout or "").strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw": text}


def _run_json(command: list[str], command_runner: CommandRunner) -> Any:
    try:
        result = command_runner(command)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"error": str(exc)}
    return _sanitize(_json_or_empty(result))


def _run_text(command: list[str], command_runner: CommandRunner) -> dict[str, Any]:
    try:
        result = command_runner(command)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"stdout": "", "stderr": str(exc), "returncode": 127}
    return {
        "stdout": result.stdout or "",
        "stderr": result.stderr or "",
        "returncode": result.returncode,
    }


def _root_packet(command_runner: CommandRunner) -> dict[str, Any]:
    status = _run_text(
        ["git", "status", "--short", "--branch", "--untracked-files=all"],
        command_runner,
    )
    lines = [line for line in status["stdout"].splitlines() if line.strip()]
    dirty = any(not line.startswith("##") for line in lines)
    return {"dirty": dirty, "status": lines, "returncode": status["returncode"]}


def _disk_outbox_packet(command_runner: CommandRunner) -> dict[str, Any]:
    df = _run_text(["df", "-h", "."], command_runner)
    outbox = _run_text(["find", ".aragora/automation-outbox", "-type", "f"], command_runner)
    files = [line for line in outbox["stdout"].splitlines() if line.strip()]
    return {
        "df": df["stdout"].splitlines(),
        "outbox_file_count": len(files) if outbox["returncode"] == 0 else None,
        "outbox_returncode": outbox["returncode"],
    }


def _active_owner_map(lanes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lane in lanes:
        if str(lane.get("status") or "") not in ACTIVE_STATUSES:
            continue
        rows.append(
            _sanitize(
                {
                    "lane_id": lane.get("lane_id"),
                    "owner_session": lane.get("owner_session"),
                    "status": lane.get("status"),
                    "branch": lane.get("branch"),
                    "worktree": lane.get("worktree"),
                    "pr_number": lane.get("pr_number"),
                    "next_action": lane.get("next_action"),
                }
            )
        )
    return rows


def build_decision_packet(
    *,
    registry_path: Path,
    repo_root: Path = DEFAULT_REPO_ROOT,
    lane_id: str | None = None,
    pr: int | None = None,
    branch: str | None = None,
    command_runner: CommandRunner = _default_runner,
) -> dict[str, Any]:
    """Build machine-readable live-truth inputs for owner-aware prompts."""

    lanes = _read_lanes(registry_path)
    runner = _repo_runner(repo_root) if command_runner is _default_runner else command_runner
    lane = _find_lane(lanes, lane_id=lane_id, pr=pr, branch=branch)
    blockers: list[str] = []
    root = _root_packet(runner)
    if root["dirty"]:
        blockers.append("dirty root")
    if lane and str(lane.get("status") or "") in ACTIVE_STATUSES:
        blockers.append("active owner exists for target")

    packet: dict[str, Any] = {
        "owner": _sanitize(lane) if lane else None,
        "root": root,
        "owner_map": _active_owner_map(lanes),
        "bridge_health": _run_json(
            ["python3", "scripts/agent_bridge.py", "--json", "health"],
            runner,
        ),
        "operator_snapshot": _run_json(
            ["python3", "scripts/agent_bridge.py", "operator-snapshot", "--json", "--summary-only"],
            runner,
        ),
        "active_sessions": _run_json(
            [
                "python3",
                "scripts/list_active_agent_sessions.py",
                "--json",
                "--codex-session-scan-limit",
                "120",
            ],
            runner,
        ),
        "disk_outbox": _disk_outbox_packet(runner),
        "pr": {},
        "checks": {"required": []},
        "merge_packet": {},
        "blockers": blockers,
        "selected_action": "read_only_owner_routing"
        if "active owner exists for target" in blockers
        else "repair_or_stop"
        if root["dirty"]
        else "queue_prompt",
    }

    if pr is not None:
        packet["pr"] = _run_json(
            [
                "gh",
                "pr",
                "view",
                str(pr),
                "--json",
                "number,state,isDraft,headRefOid,headRefName,mergeable,mergeStateStatus,reviewDecision,statusCheckRollup,url",
            ],
            runner,
        )
        checks = _run_json(
            [
                "gh",
                "pr",
                "checks",
                str(pr),
                "--required",
                "--json",
                "name,state,bucket,workflow,link",
            ],
            runner,
        )
        packet["checks"] = {"required": checks if isinstance(checks, list) else []}
        packet["merge_packet"] = _run_json(
            [
                "python3",
                "-m",
                "aragora.cli.main",
                "review-queue",
                "merge-packet",
                "--pr",
                str(pr),
                "--json",
            ],
            runner,
        )
    return packet


def _mailbox_command(lane: dict[str, Any] | None, *, pr: int | None, branch: str | None) -> str:
    if lane and lane.get("lane_id"):
        return (
            "python3 scripts/read_operator_steering.py --lane-id "
            f"{shlex.quote(str(lane['lane_id']))} --json || true"
        )
    if pr is not None:
        return f"python3 scripts/read_operator_steering.py --pr {pr} --json || true"
    if branch:
        return (
            "python3 scripts/read_operator_steering.py --branch "
            f"{shlex.quote(branch)} --json || true"
        )
    return "python3 scripts/agent_bridge.py operator-snapshot --json --summary-only || true"


def build_prompt(
    *,
    registry_path: Path,
    repo_root: Path = DEFAULT_REPO_ROOT,
    lane_id: str | None = None,
    pr: int | None = None,
    branch: str | None = None,
) -> str:
    lanes = _read_lanes(registry_path)
    lane = _find_lane(lanes, lane_id=lane_id, pr=pr, branch=branch)
    mailbox = _mailbox_command(lane, pr=pr, branch=branch)
    target = (
        f"lane {lane_id}"
        if lane_id
        else f"PR #{pr}"
        if pr is not None
        else branch or "the live queue"
    )

    lines = [
        f"Start from live repo truth in {repo_root}. Do not trust prior transcript state.",
        "",
        "Before lane work, check your Aragora operator-steering mailbox:",
        mailbox,
        "If a steering message redirects or says stop, obey it before doing anything else. Do not delete, edit, move, or acknowledge mailbox files.",
        "",
        "Do not paste raw transcripts into this prompt or into follow-up prompts; rebuild live truth from Aragora tooling.",
        "",
        "Run read-only live truth first:",
        "git status --short --branch --untracked-files=all",
        "python3 scripts/agent_bridge.py --json health || true",
        "python3 scripts/agent_bridge.py operator-snapshot --json --summary-only || true",
        "python3 scripts/list_active_agent_sessions.py --json --codex-session-scan-limit 120",
    ]
    if pr is not None:
        lines.extend(
            [
                f"gh pr view {pr} --json number,state,isDraft,headRefOid,mergeable,mergeStateStatus,reviewDecision,statusCheckRollup,url",
                f"python3 -m aragora.cli.main review-queue merge-packet --pr {pr} --json || true",
            ]
        )

    lines.append("")
    if lane:
        owner_session = str(lane.get("owner_session") or "")
        status = str(lane.get("status") or "")
        lines.extend(
            [
                f"Goal: make incremental progress on {target} without duplicating active owners.",
                f"Continue only if you are owner_session {owner_session} for lane {lane.get('lane_id')}. If not, stop with NOT_OWNER and report the active owner.",
                f"Current registry status to verify, not trust: status={status}, branch={lane.get('branch') or ''}, pr={lane.get('pr_number') or ''}, next_action={lane.get('next_action') or ''}.",
                "If you are the owner, perform only the next_action after live gates pass. If the lane is blocked, produce the smallest concrete unblock prompt instead of widening scope.",
            ]
        )
    else:
        lines.extend(
            [
                f"Goal: identify one safe non-overlapping action for {target}.",
                "If you cannot map yourself to a lane, run read-only only.",
                "If an active owner appears for the target PR, branch, files, queue gate, disk cleanup, or steering work, do not mutate; report owner_session, lane_id, worktree, and exact next steering message.",
                "If no owner exists and live gates are clean, produce one bounded prompt for the highest-value unowned queue action. Do not start PR work in the same run.",
            ]
        )
    lines.extend(
        [
            "",
            "Final report: mailbox state, owner/session mapping, active/conflict lanes, target PR/head/checks if applicable, action taken or withheld, exact blocker, and a fresh recursive best-next prompt that starts with mailbox checking.",
            CONVERGENCE_SENTENCE,
        ]
    )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument("--lane-id")
    selector.add_argument("--pr", type=int)
    selector.add_argument("--branch")
    parser.add_argument(
        "--registry-path",
        type=Path,
        default=DEFAULT_REPO_ROOT / REGISTRY_RELATIVE_PATH,
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=DEFAULT_REPO_ROOT,
        help="Repo root used in generated prompt text and default live-truth commands.",
    )
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prompt = build_prompt(
        registry_path=args.registry_path,
        repo_root=args.repo_root,
        lane_id=args.lane_id,
        pr=args.pr,
        branch=args.branch,
    )
    if args.json:
        packet = build_decision_packet(
            registry_path=args.registry_path,
            repo_root=args.repo_root,
            lane_id=args.lane_id,
            pr=args.pr,
            branch=args.branch,
        )
        print(json.dumps({"prompt": prompt, "decision_packet": packet}, indent=2, sort_keys=True))
    else:
        print(prompt, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
