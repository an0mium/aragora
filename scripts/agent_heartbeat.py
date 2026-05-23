#!/usr/bin/env python3
"""Write harness heartbeat metadata for an active Aragora lane.

This is the repo-local identity hook for Codex, Claude, Droid, and Factory
wrappers. It records enough process/worktree metadata for other sessions to
distinguish a live owner from a mailbox-only or stale lane without reading raw
transcripts.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import tempfile
from collections.abc import Iterator
from collections.abc import Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

_fcntl: Any
try:
    import fcntl as _fcntl
except ImportError:
    _fcntl = None

DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[1]
HEARTBEAT_RELATIVE_PATH = Path(".aragora") / "agent-bridge" / "heartbeats.json"
SAFE_OWNER_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _validate_owner_session(owner_session: str) -> None:
    if (
        not owner_session
        or owner_session in {".", ".."}
        or owner_session.startswith(".")
        or not SAFE_OWNER_RE.fullmatch(owner_session)
    ):
        raise ValueError("unsafe owner_session: use a non-empty alphanumeric/dash/underscore slug")


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(payload, list):
        return []
    return [row for row in payload if isinstance(row, dict)]


def _atomic_write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".tmp.", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


@contextmanager
def _heartbeat_write_lock(path: Path) -> Iterator[None]:
    """Serialize heartbeat read-modify-write cycles across harnesses."""

    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("a+", encoding="utf-8") as handle:
        if _fcntl is not None:
            _fcntl.flock(handle.fileno(), _fcntl.LOCK_EX)
        try:
            yield
        finally:
            if _fcntl is not None:
                _fcntl.flock(handle.fileno(), _fcntl.LOCK_UN)


def _compact(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if value not in ("", None)}


def record_heartbeat(
    *,
    heartbeat_path: Path,
    lane_id: str,
    owner_session: str,
    thread_id: str = "",
    pid: int | None = None,
    cwd: str = "",
    worktree: str = "",
    branch: str = "",
    pr_number: int | None = None,
    last_seen_at: str | None = None,
) -> dict[str, Any]:
    """Upsert a heartbeat row keyed by ``lane_id`` and ``owner_session``."""
    if not lane_id:
        raise ValueError("lane_id must not be empty")
    _validate_owner_session(owner_session)
    row = _compact(
        {
            "schema_version": "aragora-agent-heartbeat/1.0",
            "lane_id": lane_id,
            "owner_session": owner_session,
            "thread_id": thread_id,
            "pid": pid,
            "cwd": cwd or os.getcwd(),
            "worktree": worktree,
            "branch": branch,
            "pr_number": pr_number,
            "last_seen_at": last_seen_at or _utc_now_iso(),
        }
    )

    with _heartbeat_write_lock(heartbeat_path):
        rows = _read_rows(heartbeat_path)
        out: list[dict[str, Any]] = []
        replaced = False
        for existing in rows:
            if (
                str(existing.get("lane_id") or "") == lane_id
                and str(existing.get("owner_session") or "") == owner_session
            ):
                out.append(row)
                replaced = True
            else:
                out.append(existing)
        if not replaced:
            out.append(row)
        _atomic_write(heartbeat_path, out)
    return row


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane-id", required=True)
    parser.add_argument("--owner-session", required=True)
    parser.add_argument("--thread-id", default=os.environ.get("CODEX_THREAD_ID", ""))
    parser.add_argument("--pid", type=int, default=os.getpid())
    parser.add_argument("--cwd", default=os.getcwd())
    parser.add_argument("--worktree", default="")
    parser.add_argument("--branch", default="")
    parser.add_argument("--pr-number", type=int, default=None)
    parser.add_argument("--last-seen-at", default=None)
    parser.add_argument(
        "--heartbeat-path",
        type=Path,
        default=None,
        help="Override heartbeat sidecar path.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=DEFAULT_REPO_ROOT,
        help=f"Repo root (default: {DEFAULT_REPO_ROOT}).",
    )
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    heartbeat_path = args.heartbeat_path or args.repo_root / HEARTBEAT_RELATIVE_PATH
    try:
        row = record_heartbeat(
            heartbeat_path=heartbeat_path,
            lane_id=args.lane_id,
            owner_session=args.owner_session,
            thread_id=args.thread_id,
            pid=args.pid,
            cwd=args.cwd,
            worktree=args.worktree,
            branch=args.branch,
            pr_number=args.pr_number,
            last_seen_at=args.last_seen_at,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(row, indent=2, sort_keys=True))
    else:
        print(
            f"heartbeat lane_id={row['lane_id']} owner_session={row['owner_session']} "
            f"last_seen_at={row['last_seen_at']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
