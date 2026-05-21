#!/usr/bin/env python3
"""Build a concise owner-aware next prompt from live Aragora coordination state."""

from __future__ import annotations

import argparse
import json
import sys
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


def _mailbox_command(lane: dict[str, Any] | None, *, pr: int | None, branch: str | None) -> str:
    if lane and lane.get("lane_id"):
        return (
            f"python3 scripts/read_operator_steering.py --lane-id {lane['lane_id']} --json || true"
        )
    if pr is not None:
        return f"python3 scripts/read_operator_steering.py --pr {pr} --json || true"
    if branch:
        return f"python3 scripts/read_operator_steering.py --branch {branch} --json || true"
    return "python3 scripts/agent_bridge.py operator-snapshot --json --summary-only || true"


def build_prompt(
    *,
    registry_path: Path,
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
        "Start from live repo truth in /Users/armand/Development/aragora. Do not trust prior transcript state.",
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
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prompt = build_prompt(
        registry_path=args.registry_path,
        lane_id=args.lane_id,
        pr=args.pr,
        branch=args.branch,
    )
    if args.json:
        print(json.dumps({"prompt": prompt}, indent=2, sort_keys=True))
    else:
        print(prompt, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
