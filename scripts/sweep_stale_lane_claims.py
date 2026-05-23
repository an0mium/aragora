"""Detect and (optionally) expire stale active lane claims.

The lane registry written by ``scripts/claim_active_agent_lane.py`` and read
by ``scripts/agent_bridge.py`` accumulates rows over time. Most "active"
rows get released by the owner when the lane completes, but some never do:

- A session crashed before releasing the lane.
- A session was killed by the operator.
- An owner script bailed mid-run.

Without periodic cleanup these zombie rows hold the lane_id, branch, and
worktree slots against new owners, so collision detection rejects valid
work as duplicate. This sweeper detects stale active rows via three
independent signals and (with ``--apply``) rewrites them in-place with
``status=expired`` and a ``conflict_reason`` describing the cause.

Detection signals (any triggers expiration):

1. ``branch_missing`` -- the lane.branch is not present in ``git
   branch --list <branch>`` AND not present in
   ``git ls-remote --heads origin <branch>``. Strongest orphan signal.
2. ``worktree_missing`` -- lane.worktree is set but the path does not
   exist on disk.
3. ``stale_updated_at`` -- updated_at is older than
   ``--max-active-age-hours`` (default 24 h).

Default is ``--dry-run`` (print findings as JSON, exit 0). ``--apply``
rewrites stale rows in-place.

Output schema:

    {
      "registry_path": ".../lanes.json",
      "scanned_at": "2026-05-18T17:55:00Z",
      "total_rows": 51,
      "active_rows": 5,
      "stale_rows": 1,
      "stale_records": [
        {"lane_id": "...", "owner_session": "...", "reasons": ["branch_missing", "stale_updated_at"]}
      ],
      "applied": false
    }
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

_fcntl: Any
try:
    import fcntl as _fcntl
except ImportError:
    _fcntl = None


REPO_LANE_RELATIVE_PATH = Path(".aragora") / "agent-bridge" / "lanes.json"
USER_LANE_PATH = Path.home() / ".aragora" / "agent-bridge" / "lanes.json"

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


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.UTC).replace(microsecond=0)


def _utc_now_iso() -> str:
    return _utc_now().isoformat().replace("+00:00", "Z")


def resolve_registry_path(*, repo_root: Path, explicit: Path | None = None) -> Path:
    if explicit is not None:
        return explicit
    repo_lane = repo_root / REPO_LANE_RELATIVE_PATH
    if repo_lane.parent.exists():
        return repo_lane
    return USER_LANE_PATH


def _parse_utc_timestamp(value: object) -> dt.datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    raw = value.strip()
    try:
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        return dt.datetime.fromisoformat(raw)
    except ValueError:
        return None


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


def branch_exists_locally(repo: Path, branch: str, *, timeout: int = 5) -> bool:
    if not branch:
        return False
    if shutil.which("git") is None:
        return True
    try:
        proc = subprocess.run(
            ["git", "branch", "--list", branch],
            cwd=repo,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return True
    return bool(proc.stdout.strip())


def branch_exists_remotely(repo: Path, branch: str, *, timeout: int = 10) -> bool:
    if not branch:
        return False
    if shutil.which("git") is None:
        return True
    try:
        proc = subprocess.run(
            ["git", "ls-remote", "--heads", "origin", branch],
            cwd=repo,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return True
    return bool(proc.stdout.strip())


def evaluate_row(
    row: dict[str, Any],
    *,
    now: dt.datetime,
    repo: Path,
    max_age_hours: float,
    check_branches: bool,
    check_remote: bool,
    branch_grace_hours: float,
) -> list[str]:
    if row.get("status") not in ACTIVE_STATUSES:
        return []
    reasons: list[str] = []

    ts = _parse_utc_timestamp(row.get("updated_at"))
    age_hours = (now - ts).total_seconds() / 3600 if ts is not None else 0.0

    branch = row.get("branch")
    if check_branches and isinstance(branch, str) and branch and age_hours >= branch_grace_hours:
        local = branch_exists_locally(repo, branch)
        remote = branch_exists_remotely(repo, branch) if (not local and check_remote) else local
        if not local and not remote:
            reasons.append("branch_missing")

    worktree = row.get("worktree")
    if isinstance(worktree, str) and worktree and not Path(worktree).exists():
        reasons.append("worktree_missing")

    if ts is not None and age_hours > max_age_hours:
        reasons.append("stale_updated_at")

    return reasons


def sweep(
    *,
    registry_path: Path,
    repo: Path,
    max_age_hours: float,
    apply: bool,
    check_branches: bool,
    check_remote: bool,
    now: dt.datetime | None = None,
    branch_grace_hours: float = 1.0,
) -> dict[str, Any]:
    rows = _read_existing(registry_path)
    if now is None:
        now = _utc_now()
    active_rows = [r for r in rows if r.get("status") in ACTIVE_STATUSES]
    stale_records: list[dict[str, Any]] = []
    for row in active_rows:
        reasons = evaluate_row(
            row,
            now=now,
            repo=repo,
            max_age_hours=max_age_hours,
            check_branches=check_branches,
            check_remote=check_remote,
            branch_grace_hours=branch_grace_hours,
        )
        if reasons:
            stale_records.append(
                {
                    "lane_id": row.get("lane_id"),
                    "owner_session": row.get("owner_session"),
                    "branch": row.get("branch"),
                    "worktree": row.get("worktree"),
                    "updated_at": row.get("updated_at"),
                    "reasons": reasons,
                }
            )

    applied = False
    if apply and stale_records:
        stale_keys = {(r["lane_id"], r["owner_session"]) for r in stale_records}
        reason_lookup = {(r["lane_id"], r["owner_session"]): r["reasons"] for r in stale_records}
        out_rows: list[dict[str, Any]] = []
        for row in rows:
            key = (row.get("lane_id"), row.get("owner_session"))
            if key in stale_keys:
                row = dict(row)
                row["status"] = "expired"
                row["updated_at"] = now.isoformat().replace("+00:00", "Z")
                row["conflict_reason"] = "stale: " + ",".join(reason_lookup[key])
            out_rows.append(row)
        _atomic_write(registry_path, out_rows)
        applied = True

    return {
        "registry_path": str(registry_path),
        "scanned_at": now.isoformat().replace("+00:00", "Z"),
        "total_rows": len(rows),
        "active_rows": len(active_rows),
        "stale_rows": len(stale_records),
        "stale_records": stale_records,
        "applied": applied,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--registry-path",
        type=Path,
        help="Override path to lanes.json (default: <repo>/.aragora/agent-bridge/lanes.json)",
    )
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path.cwd(),
        help="Repository root used for git branch lookups (default: cwd)",
    )
    parser.add_argument(
        "--max-active-age-hours",
        type=float,
        default=24.0,
        help="Active rows older than this many hours are flagged stale (default 24)",
    )
    parser.add_argument(
        "--branch-grace-hours",
        type=float,
        default=1.0,
        help=(
            "Skip branch_missing detection for active rows newer than this. "
            "Allows freshly-claimed lanes time to push their branch (default 1h)"
        ),
    )
    parser.add_argument(
        "--skip-branch-check",
        action="store_true",
        help="Skip local+remote git branch existence checks (default off)",
    )
    parser.add_argument(
        "--skip-remote-check",
        action="store_true",
        help="Only check local branches; skip ls-remote (default off)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Rewrite stale active rows in-place with status=expired (default dry-run)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Explicit no-op alias for the default behavior (no rewrite). "
            "Mutually exclusive with --apply. Useful in scripted invocations "
            "(Makefile, cron) where the intent should be self-documenting."
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full machine-readable JSON to stdout",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.dry_run and args.apply:
        parser.error("--dry-run and --apply are mutually exclusive")
    repo = args.repo.expanduser().resolve()
    registry_path = resolve_registry_path(repo_root=repo, explicit=args.registry_path)
    report = sweep(
        registry_path=registry_path,
        repo=repo,
        max_age_hours=args.max_active_age_hours,
        apply=args.apply,
        check_branches=not args.skip_branch_check,
        check_remote=not args.skip_remote_check,
        branch_grace_hours=args.branch_grace_hours,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(
            f"registry={report['registry_path']} total={report['total_rows']} "
            f"active={report['active_rows']} stale={report['stale_rows']} "
            f"applied={report['applied']}"
        )
        for record in report["stale_records"]:
            print(
                f"  STALE lane={record['lane_id']} owner={record['owner_session']} "
                f"reasons={','.join(record['reasons'])}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
