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
PENDING_CHECK_STATES = {
    "ACTION_REQUIRED",
    "EXPECTED",
    "IN_PROGRESS",
    "PENDING",
    "QUEUED",
    "REQUESTED",
    "WAITING",
}


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


def _active_target_lanes(
    lanes: list[dict[str, Any]],
    *,
    lane: dict[str, Any] | None,
    lane_id: str | None = None,
    pr: int | None = None,
    branch: str | None = None,
) -> list[dict[str, Any]]:
    """Return active lanes that appear to own the selected PR/branch/worktree."""

    keys: list[tuple[str, Any]] = []
    if lane_id:
        keys.append(("lane_id", lane_id))
    if pr is not None:
        keys.append(("pr_number", pr))
    if branch:
        keys.append(("branch", branch))
    if lane:
        for key in ("pr_number", "branch", "worktree"):
            value = lane.get(key)
            if value not in (None, ""):
                keys.append((key, value))

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in lanes:
        if str(row.get("status") or "") not in ACTIVE_STATUSES:
            continue
        if not any(row.get(key) == value for key, value in keys):
            continue
        row_key = str(row.get("lane_id") or id(row))
        if row_key in seen:
            continue
        seen.add(row_key)
        rows.append(
            _sanitize(
                {
                    "lane_id": row.get("lane_id"),
                    "owner_session": row.get("owner_session"),
                    "status": row.get("status"),
                    "branch": row.get("branch"),
                    "worktree": row.get("worktree"),
                    "pr_number": row.get("pr_number"),
                    "next_action": row.get("next_action"),
                }
            )
        )
    return rows


def _merge_packet_entry(merge_packet: Any, pr: int | None) -> dict[str, Any]:
    if not isinstance(merge_packet, dict):
        return {}
    entries = merge_packet.get("entries")
    if isinstance(entries, list):
        for entry in entries:
            if isinstance(entry, dict) and pr is not None and entry.get("pr_number") == pr:
                return entry
        return {}
    return merge_packet


def _packet_not_ready(merge_packet: Any) -> list[Any]:
    if not isinstance(merge_packet, dict):
        return []
    not_ready = merge_packet.get("not_ready")
    return not_ready if isinstance(not_ready, list) else []


def _packet_authorizes(merge_packet: Any, *, pr: int | None) -> bool:
    entry = _merge_packet_entry(merge_packet, pr)
    if not entry.get("admin_squash_allowed"):
        return False
    not_ready = _packet_not_ready(merge_packet)
    return not not_ready or (pr is not None and pr not in not_ready)


def _packet_authorization_reason(merge_packet: Any, *, pr: int | None) -> str | None:
    if not isinstance(merge_packet, dict) or not merge_packet:
        return "merge-packet authorization is missing or malformed"
    entry = _merge_packet_entry(merge_packet, pr)
    if not entry:
        target = f"PR #{pr}" if pr is not None else "target PR"
        return f"merge-packet has no entry for {target}"
    if not entry.get("admin_squash_allowed"):
        return "merge-packet does not authorize admin squash"
    not_ready = _packet_not_ready(merge_packet)
    if pr is not None and pr in not_ready:
        return f"merge-packet still lists PR #{pr} as not_ready"
    if pr is None and not_ready:
        return "merge-packet still has not_ready entries"
    return None


def _pending_required_checks(checks: Any) -> list[dict[str, str]]:
    if not isinstance(checks, list):
        return []
    pending: list[dict[str, str]] = []
    for check in checks:
        if not isinstance(check, dict):
            continue
        bucket = str(check.get("bucket") or "").lower()
        state = str(check.get("state") or "").upper()
        if bucket == "pending" or state in PENDING_CHECK_STATES:
            pending.append(
                {
                    "name": str(check.get("name") or ""),
                    "workflow": str(check.get("workflow") or ""),
                    "state": str(check.get("state") or ""),
                    "bucket": str(check.get("bucket") or ""),
                }
            )
    return pending


def build_settlement_guard(
    packet: dict[str, Any],
    *,
    pr: int | None = None,
    expected_head: str | None = None,
) -> dict[str, Any]:
    """Fail-closed settlement preflight for exact-head prompts."""

    pr_packet = packet.get("pr") if isinstance(packet.get("pr"), dict) else {}
    merge_packet = (
        packet.get("merge_packet") if isinstance(packet.get("merge_packet"), dict) else {}
    )
    entry = _merge_packet_entry(merge_packet, pr)
    live_head = str(pr_packet.get("headRefOid") or "") if isinstance(pr_packet, dict) else ""
    packet_head = str(entry.get("head_sha") or entry.get("headRefOid") or "")
    pending_checks = _pending_required_checks(packet.get("checks", {}).get("required"))
    target_lanes = packet.get("target_active_lanes")
    target_lanes = target_lanes if isinstance(target_lanes, list) else []
    reasons: list[str] = []
    authorization_reason = _packet_authorization_reason(merge_packet, pr=pr)

    if expected_head and live_head and expected_head != live_head:
        reasons.append(f"expected head {expected_head} does not match live head {live_head}")
    if authorization_reason:
        reasons.append(authorization_reason)
    if len(target_lanes) > 1:
        owners = ", ".join(
            str(row.get("owner_session") or row.get("lane_id")) for row in target_lanes
        )
        reasons.append(f"multiple active target owners: {owners}")
    if packet_head and live_head and packet_head != live_head:
        reasons.append(f"merge-packet head {packet_head} does not match live head {live_head}")
    if pending_checks and not authorization_reason:
        names = ", ".join(
            f"{check['workflow']} / {check['name']}".strip(" /") for check in pending_checks
        )
        reasons.append(f"merge-packet authorizes settlement while checks are pending: {names}")

    return {
        "allowed": not reasons,
        "verdict": "pass" if not reasons else "fail_closed",
        "reasons": reasons,
        "expected_head": expected_head,
        "live_head": live_head or None,
        "merge_packet_head": packet_head or None,
        "pending_checks": pending_checks,
        "target_active_lanes": target_lanes,
        "merge_packet_authorizes": _packet_authorizes(merge_packet, pr=pr),
    }


def build_decision_packet(
    *,
    registry_path: Path,
    repo_root: Path = DEFAULT_REPO_ROOT,
    lane_id: str | None = None,
    pr: int | None = None,
    branch: str | None = None,
    expected_head: str | None = None,
    command_runner: CommandRunner = _default_runner,
) -> dict[str, Any]:
    """Build machine-readable live-truth inputs for owner-aware prompts."""

    lanes = _read_lanes(registry_path)
    runner = _repo_runner(repo_root) if command_runner is _default_runner else command_runner
    lane = _find_lane(lanes, lane_id=lane_id, pr=pr, branch=branch)
    target_active_lanes = _active_target_lanes(
        lanes, lane=lane, lane_id=lane_id, pr=pr, branch=branch
    )
    blockers: list[str] = []
    root = _root_packet(runner)
    if root["dirty"]:
        blockers.append("dirty root")
    if lane and str(lane.get("status") or "") in ACTIVE_STATUSES:
        blockers.append("active owner exists for target")
    if len(target_active_lanes) > 1:
        blockers.append("multiple active owners exist for target")

    packet: dict[str, Any] = {
        "owner": _sanitize(lane) if lane else None,
        "target_active_lanes": target_active_lanes,
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
    packet["settlement_guard"] = build_settlement_guard(packet, pr=pr, expected_head=expected_head)
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


def build_settlement_guard_prompt(
    packet: dict[str, Any],
    *,
    repo_root: Path = DEFAULT_REPO_ROOT,
    pr: int | None = None,
    branch: str | None = None,
) -> str:
    guard = packet.get("settlement_guard")
    guard = guard if isinstance(guard, dict) else {}
    owner = packet.get("owner") if isinstance(packet.get("owner"), dict) else None
    owners = guard.get("target_active_lanes")
    owners = owners if isinstance(owners, list) else []
    active_owner = owners[0] if len(owners) == 1 and isinstance(owners[0], dict) else None
    mailbox = _mailbox_command(active_owner, pr=pr, branch=branch)
    owner_summary = (
        ", ".join(
            f"{row.get('lane_id')} / {row.get('owner_session')}"
            for row in owners
            if isinstance(row, dict)
        )
        or "none"
    )
    pending = guard.get("pending_checks")
    pending = pending if isinstance(pending, list) else []
    pending_summary = (
        ", ".join(
            f"{row.get('workflow')} / {row.get('name')}".strip(" /")
            for row in pending
            if isinstance(row, dict)
        )
        or "none"
    )
    reasons = guard.get("reasons")
    reasons = reasons if isinstance(reasons, list) else []
    reason_summary = "; ".join(str(reason) for reason in reasons) or "none"
    target = f"PR #{pr}" if pr is not None else branch or "the live queue"

    return "\n".join(
        [
            f"Start from live repo truth in {repo_root}. Do not trust prior transcript state.",
            "",
            "Before acting, check your Aragora operator-steering mailbox:",
            mailbox,
            "If a steering message redirects or says stop, obey it before doing anything else. Do not delete, edit, move, or acknowledge mailbox files.",
            "",
            f"Goal: settlement-guard {target} before any edit, push, comment, merge, mark-ready, cleanup, or workflow rerun.",
            f"Guard verdict to verify, not trust: {guard.get('verdict') or 'unknown'}.",
            f"Expected head: {guard.get('expected_head') or 'not supplied'}",
            f"Live head: {guard.get('live_head') or 'unknown'}",
            f"Merge-packet head: {guard.get('merge_packet_head') or 'unknown'}",
            f"Active target owners: {owner_summary}",
            f"Pending required checks: {pending_summary}",
            f"Fail-closed reasons: {reason_summary}",
            "",
            "Re-check git status, lanes, identify_lane_owner.py, gh pr view/checks, and merge-packet before mutating.",
            "If the guard still fails closed, do not mutate; report the exact blockers and produce the next bounded prompt.",
            "If the guard passes and merge-packet returns admin_squash_allowed=true and not_ready=[], exact-head settlement may proceed only within the prompt's PR/tier constraints.",
            CONVERGENCE_SENTENCE,
            "",
        ]
    )


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
    parser.add_argument("--expected-head", help="Exact head SHA the prompt intends to handle.")
    parser.add_argument(
        "--settlement-guard",
        action="store_true",
        help="Emit a fail-closed settlement guard prompt populated from live truth.",
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
    packet: dict[str, Any] | None = None
    guard_prompt: str | None = None
    if args.json or args.settlement_guard:
        packet = build_decision_packet(
            registry_path=args.registry_path,
            repo_root=args.repo_root,
            lane_id=args.lane_id,
            pr=args.pr,
            branch=args.branch,
            expected_head=args.expected_head,
        )
        guard_prompt = build_settlement_guard_prompt(
            packet,
            repo_root=args.repo_root,
            pr=args.pr,
            branch=args.branch,
        )
    if args.json:
        print(
            json.dumps(
                {
                    "prompt": prompt,
                    "settlement_guard_prompt": guard_prompt,
                    "decision_packet": packet,
                },
                indent=2,
                sort_keys=True,
            )
        )
    elif args.settlement_guard:
        print(guard_prompt or "", end="")
    else:
        print(prompt, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
