#!/usr/bin/env python3
"""Write a single lane claim into the agent-bridge lane registry.

The lane registry is the cross-agent coordination primitive shipped by
``scripts/agent_bridge.py`` (see ``LaneRecord``). It already records lane
claims when sessions explicitly call into the agent_bridge command, but
in practice many sessions (Claude Code, Codex CLI, Factory Droid,
standalone scripts) never write a lane claim, so the registry tends to
stay empty even when 5-10 concurrent agents are running. The companion
script ``scripts/list_active_agent_sessions.py`` reads the registry and
folds lane branches/worktrees/PR numbers into its overlap report so an
operator can see at a glance whether a new branch would collide with an
in-flight agent.

This helper is the thin write-side complement: it claims (or refreshes)
a single lane row in ``.aragora/agent-bridge/lanes.json`` (preferred) or
``~/.aragora/agent-bridge/lanes.json`` (fallback), matching the existing
``LaneRecord`` schema so that ``scripts/agent_bridge.py operator-snapshot``
will pick it up immediately on the next invocation.

Design constraints (kept tight so this script can be used during
bootstraps and from any agent):

- Pure Python stdlib, no aragora package import.
- Serialize the read-modify-write under a sibling lock file when ``fcntl``
  is available, while preserving atomic file integrity via write to
  ``<file>.tmp`` + rename.
- Schema matches ``LaneRecord`` exactly so the existing agent_bridge
  reader picks it up without modification: ``lane_id``, ``owner_session``,
  ``goal``, ``source``, ``status``, ``next_action``, ``updated_at``,
  ``branch``, ``worktree``, ``pr_number``, ``conflict_session``,
  ``conflict_reason``.
- A claim with the same ``lane_id`` from the same ``owner_session``
  overwrites the existing row (idempotent refresh). A claim with the
  same ``lane_id`` from a *different* ``owner_session`` is rejected
  unless ``--force`` is supplied; the helper exits 2 and prints a
  conflict hint so the caller can pick a different lane name or wait.
- Never deletes rows. Never writes a lane row with a missing
  ``lane_id`` or empty ``owner_session``.
- Never invokes git, gh, or any network call.
- Per-file lock via ``fcntl.flock`` on POSIX serializes concurrent claims.
  On platforms without ``fcntl`` we fall back to atomic rename-only and
  accept a small TOCTOU window.

Usage:

    python3 scripts/claim_active_agent_lane.py \\
        --lane-id droid/phase4-freshness-launchagent \\
        --owner-session claude-20260517-144003-b211740c \\
        --branch droid/phase4-freshness-launchagent-20260517 \\
        --worktree /Users/armand/Development/aragora/.worktrees/codex-auto/claude-20260517-144003-b211740c \\
        --goal "phase 4: freshness probe LaunchAgent shim" \\
        --source plan

To release a lane, write a ``--status released`` claim with the same
``lane_id`` and ``owner_session``. To mark a conflict, write a
``--status conflict --conflict-session <other> --conflict-reason <text>``
claim.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from contextlib import contextmanager
import datetime as dt
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

try:
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - exercised only on non-POSIX systems.
    _fcntl = None  # type: ignore[assignment]

DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_LANE_RELATIVE_PATH = Path(".aragora") / "agent-bridge" / "lanes.json"
USER_LANE_PATH = Path.home() / ".aragora" / "agent-bridge" / "lanes.json"

ALLOWED_STATUSES = (
    "active",
    "running",
    "pending",
    "queued",
    "claimed",
    "released",
    "completed",
    "conflict",
)

LANE_RECORD_KEYS = (
    "lane_id",
    "owner_session",
    "goal",
    "source",
    "status",
    "next_action",
    "updated_at",
    "branch",
    "worktree",
    "pr_number",
    "conflict_session",
    "conflict_reason",
)


class ClaimError(Exception):
    """Raised when a lane claim would conflict with an existing row."""


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def resolve_registry_path(
    *,
    repo_root: Path,
    explicit: Path | None = None,
) -> Path:
    if explicit is not None:
        return explicit
    repo_lane = repo_root / REPO_LANE_RELATIVE_PATH
    if repo_lane.parent.exists():
        return repo_lane
    return USER_LANE_PATH


def _read_existing(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(payload, list):
        return []
    return [entry for entry in payload if isinstance(entry, dict)]


def _atomic_write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".tmp.", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=2, sort_keys=True)
            fh.write("\n")
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


@contextmanager
def _registry_write_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(path.name + ".lock")
    with lock_path.open("a", encoding="utf-8") as lock_fh:
        if _fcntl is not None:
            _fcntl.flock(lock_fh.fileno(), _fcntl.LOCK_EX)
        try:
            yield
        finally:
            if _fcntl is not None:
                _fcntl.flock(lock_fh.fileno(), _fcntl.LOCK_UN)


def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in LANE_RECORD_KEYS:
        value = row.get(key)
        if value is None or value == "":
            continue
        out[key] = value
    return out


def claim_lane(
    *,
    registry_path: Path,
    lane_id: str,
    owner_session: str,
    goal: str = "",
    source: str = "",
    status: str = "active",
    next_action: str = "",
    branch: str = "",
    worktree: str = "",
    pr_number: int | None = None,
    conflict_session: str = "",
    conflict_reason: str = "",
    force: bool = False,
    updated_at: str | None = None,
) -> dict[str, Any]:
    """Write or refresh a single lane claim, returning the persisted row."""
    if not lane_id:
        raise ValueError("lane_id must not be empty")
    if not owner_session:
        raise ValueError("owner_session must not be empty")
    if status not in ALLOWED_STATUSES:
        raise ValueError(f"status {status!r} is not in {sorted(ALLOWED_STATUSES)}")

    with _registry_write_lock(registry_path):
        rows = _read_existing(registry_path)
        timestamp = updated_at or _utc_now_iso()

        new_row: dict[str, Any] = {
            "lane_id": lane_id,
            "owner_session": owner_session,
            "goal": goal,
            "source": source,
            "status": status,
            "next_action": next_action,
            "updated_at": timestamp,
            "branch": branch,
            "worktree": worktree,
            "pr_number": pr_number,
            "conflict_session": conflict_session,
            "conflict_reason": conflict_reason,
        }
        normalized = _normalize_row(new_row)

        out_rows: list[dict[str, Any]] = []
        replaced = False
        for existing in rows:
            existing_id = str(existing.get("lane_id") or "")
            if existing_id != lane_id:
                out_rows.append(existing)
                continue
            existing_owner = str(existing.get("owner_session") or "")
            if existing_owner and existing_owner != owner_session and not force:
                raise ClaimError(
                    f"lane_id={lane_id!r} already claimed by "
                    f"owner_session={existing_owner!r}; refusing to overwrite "
                    f"(use --force to override or pick a different lane id)"
                )
            replaced = True
            out_rows.append(normalized)
        if not replaced:
            out_rows.append(normalized)

        _atomic_write(registry_path, out_rows)
        return normalized


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane-id", required=True)
    parser.add_argument("--owner-session", required=True)
    parser.add_argument("--goal", default="")
    parser.add_argument("--source", default="")
    parser.add_argument(
        "--status",
        default="active",
        choices=ALLOWED_STATUSES,
        help=f"Lane status (default: active). Choices: {ALLOWED_STATUSES}",
    )
    parser.add_argument("--next-action", default="")
    parser.add_argument("--branch", default="")
    parser.add_argument("--worktree", default="")
    parser.add_argument("--pr-number", type=int, default=None)
    parser.add_argument("--conflict-session", default="")
    parser.add_argument("--conflict-reason", default="")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing claim owned by a different session.",
    )
    parser.add_argument(
        "--registry-path",
        type=Path,
        default=None,
        help=(
            "Override the lane registry path. Defaults to the repo-local "
            ".aragora/agent-bridge/lanes.json if its directory exists, else "
            "~/.aragora/agent-bridge/lanes.json."
        ),
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=DEFAULT_REPO_ROOT,
        help=f"Repo root (default: {DEFAULT_REPO_ROOT}).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the persisted row as JSON on stdout.",
    )
    parser.add_argument(
        "--updated-at",
        default=None,
        help="Override timestamp (RFC3339 UTC, e.g. 2026-05-17T15:00:00Z).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    registry_path = resolve_registry_path(
        repo_root=args.repo_root,
        explicit=args.registry_path,
    )
    try:
        row = claim_lane(
            registry_path=registry_path,
            lane_id=args.lane_id,
            owner_session=args.owner_session,
            goal=args.goal,
            source=args.source,
            status=args.status,
            next_action=args.next_action,
            branch=args.branch,
            worktree=args.worktree,
            pr_number=args.pr_number,
            conflict_session=args.conflict_session,
            conflict_reason=args.conflict_reason,
            force=args.force,
            updated_at=args.updated_at,
        )
    except ClaimError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(row, indent=2, sort_keys=True))
    else:
        print(
            f"claimed lane_id={row['lane_id']} owner={row['owner_session']} "
            f"status={row['status']} registry={registry_path}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
