#!/usr/bin/env python3
"""Backfill ``contact_method`` on existing lane registry rows (R05 of the
agent-dispatch reach plan).

Phase 1 (R01, PR #7336) added ``contact_method`` + ``contact_payload``
fields to ``LaneRecord``. New lane claims auto-populate when made from
inside an aragora tmux pane, but the **existing** rows in
``.aragora/agent-bridge/lanes.json`` were claimed before R01 shipped and
have no ``contact_method`` set. R05 walks the registry and infers
``contact_method`` for rows that meet a high-confidence rule:

  - ``owner_session`` matches a currently-live tmux window in the
    ``aragora`` session → ``tmux:<window-name>``
  - else if a corresponding live process is detected via
    ``agent_bridge.py processes --json`` → ``tmux:<window-name>``
    (when the process is in an aragora tmux pane)
  - else: leave unset (so ``wake_agent.sh`` falls back to ``mailbox-only``
    at dispatch time, which is the correct semantic for unknown
    contact)

Default mode is ``--dry-run``: produces a report of inferred
``contact_method`` per row but does NOT mutate the registry. ``--apply``
opts into the mutate path, which requires R01's CLI flag to be
available (calls ``claim_active_agent_lane.py --contact-method`` to
do the update so collision detection and atomic writes are reused).

This script ships before R01 lands on main. In that state the apply
path will fail loudly (claim_active_agent_lane.py won't recognize
``--contact-method``), so operators should run ``--dry-run`` until R01
merges, then ``--apply``.

Usage:
    python3 scripts/sweep_lane_contact_methods.py             # dry-run, JSON to stdout
    python3 scripts/sweep_lane_contact_methods.py --apply
    python3 scripts/sweep_lane_contact_methods.py --status released

Pure stdlib. Read-only by default. No new pip deps.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_LANE_RELATIVE_PATH = Path(".aragora") / "agent-bridge" / "lanes.json"
USER_LANE_PATH = Path.home() / ".aragora" / "agent-bridge" / "lanes.json"

ACTIVE_LIKE_STATUSES = {"active", "running", "claimed", "pending", "queued"}
ALL_STATUSES = {
    "active",
    "running",
    "claimed",
    "pending",
    "queued",
    "released",
    "completed",
    "conflict",
}


def resolve_registry_path(*, repo_root: Path, explicit: Path | None = None) -> Path:
    if explicit is not None:
        return explicit
    repo_lane = repo_root / REPO_LANE_RELATIVE_PATH
    if repo_lane.parent.exists():
        return repo_lane
    return USER_LANE_PATH


def read_registry(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(payload, list):
        return []
    return [row for row in payload if isinstance(row, dict)]


def list_tmux_window_names(*, session: str = "aragora") -> list[str]:
    """Return the list of window names in the aragora tmux session.

    Empty list if tmux is unavailable, the session doesn't exist, or the
    `tmux` binary fails for any reason. Pure stdlib subprocess.
    """
    try:
        result = subprocess.run(
            ["tmux", "list-windows", "-t", session, "-F", "#W"],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return []
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def infer_contact_method(
    row: dict[str, Any],
    *,
    tmux_windows: set[str],
) -> tuple[str | None, str]:
    """Return (inferred_contact_method, reason).

    Heuristic order:
    1. owner_session matches a live tmux window in `aragora` session
       exactly → ``tmux:<owner_session>``
    2. owner_session matches a tmux window name as substring →
       ``tmux:<window-name>``
    3. else → (None, 'no-live-match')
    """
    if row.get("contact_method"):
        return (None, "already-set")
    owner = str(row.get("owner_session") or "").strip()
    if not owner:
        return (None, "no-owner")
    if owner in tmux_windows:
        return (f"tmux:{owner}", "owner-equals-window")
    # Substring match — useful for "claude-79AAF84B" matching "claude-79"
    for window in tmux_windows:
        if window and (window in owner or owner in window):
            return (f"tmux:{window}", f"owner-substring-window={window}")
    return (None, "no-live-match")


def sweep(
    *,
    registry_path: Path,
    status_filter: set[str] | None,
    tmux_session: str,
) -> dict[str, Any]:
    rows = read_registry(registry_path)
    tmux_windows = set(list_tmux_window_names(session=tmux_session))
    results: list[dict[str, Any]] = []
    counts = {"total": len(rows), "considered": 0, "inferred": 0, "skipped": 0}
    for row in rows:
        status = str(row.get("status") or "")
        if status_filter and status not in status_filter:
            counts["skipped"] += 1
            continue
        counts["considered"] += 1
        inferred, reason = infer_contact_method(row, tmux_windows=tmux_windows)
        entry = {
            "lane_id": row.get("lane_id"),
            "owner_session": row.get("owner_session"),
            "status": status,
            "existing_contact_method": row.get("contact_method"),
            "inferred_contact_method": inferred,
            "reason": reason,
        }
        if inferred:
            counts["inferred"] += 1
        results.append(entry)
    return {
        "schema_version": "aragora-sweep-lane-contact-methods/1.0",
        "registry_path": str(registry_path),
        "tmux_session": tmux_session,
        "tmux_window_count": len(tmux_windows),
        "counts": counts,
        "results": results,
    }


def apply_inferred(
    report: dict[str, Any],
    *,
    repo_root: Path,
    registry_path: Path,
) -> dict[str, Any]:
    """Re-claim each row with --contact-method via claim_active_agent_lane.py.

    Requires R01 (PR #7336) to be on main — exits with applied_errors
    populated if --contact-method is rejected by argparse.
    """
    claim_script = repo_root / "scripts" / "claim_active_agent_lane.py"
    applied: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for entry in report["results"]:
        if not entry.get("inferred_contact_method"):
            continue
        cmd = [
            sys.executable,
            str(claim_script),
            "--lane-id",
            entry["lane_id"],
            "--owner-session",
            entry["owner_session"],
            "--status",
            entry["status"],
            "--contact-method",
            entry["inferred_contact_method"],
            "--registry-path",
            str(registry_path),
            "--force",
            "--json",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            errors.append(
                {
                    "lane_id": entry["lane_id"],
                    "stderr": (result.stderr or "").strip()[:200],
                    "returncode": result.returncode,
                }
            )
            continue
        applied.append(
            {
                "lane_id": entry["lane_id"],
                "contact_method": entry["inferred_contact_method"],
            }
        )
    return {
        "applied_count": len(applied),
        "applied": applied,
        "error_count": len(errors),
        "errors": errors,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry-path",
        type=Path,
        default=None,
        help="Explicit lane-registry JSON path. Default: repo-local or user-level fallback.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=DEFAULT_REPO_ROOT,
        help="Repo root for default registry/scripts resolution.",
    )
    parser.add_argument(
        "--status",
        action="append",
        default=None,
        help=(
            "Only consider rows with this status. May repeat (e.g. "
            "--status active --status running). Default: all statuses."
        ),
    )
    parser.add_argument(
        "--active-only",
        action="store_true",
        help="Convenience: only consider active-like statuses.",
    )
    parser.add_argument(
        "--tmux-session",
        default="aragora",
        help="Tmux session name to scan for live windows. Default: aragora.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Mutate path: re-claim each inferred row with --contact-method "
            "via claim_active_agent_lane.py. Requires R01 (PR #7336) to be "
            "on main; will fail loudly otherwise."
        ),
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    registry_path = resolve_registry_path(
        repo_root=args.repo_root,
        explicit=args.registry_path,
    )
    status_filter: set[str] | None = None
    if args.active_only:
        status_filter = set(ACTIVE_LIKE_STATUSES)
    elif args.status:
        status_filter = {s.strip() for s in args.status if s.strip()}
        unknown = status_filter - ALL_STATUSES
        if unknown:
            print(
                f"error: unknown status(es): {sorted(unknown)}; valid: {sorted(ALL_STATUSES)}",
                file=sys.stderr,
            )
            return 1
    report = sweep(
        registry_path=registry_path,
        status_filter=status_filter,
        tmux_session=args.tmux_session,
    )
    report["mode"] = "apply" if args.apply else "dry-run"
    if args.apply:
        apply_report = apply_inferred(
            report,
            repo_root=args.repo_root,
            registry_path=registry_path,
        )
        report["apply"] = apply_report
    indent = 2 if args.pretty else None
    print(json.dumps(report, indent=indent, sort_keys=True))
    if args.apply and report.get("apply", {}).get("error_count"):
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
