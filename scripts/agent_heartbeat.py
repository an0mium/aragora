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
AUTOMATION_STATE_ROOT_ENV = "ARAGORA_AUTOMATION_STATE_ROOT"
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


def _state_dir(root: Path) -> Path:
    expanded = root.expanduser()
    if expanded.name == ".aragora":
        return expanded
    return expanded / ".aragora"


def _has_agent_bridge_state(root: Path) -> bool:
    return (_state_dir(root) / "agent-bridge").is_dir()


def _automation_state_root(repo_root: Path) -> Path:
    """Return the checkout or direct .aragora dir backing heartbeat state."""

    if _has_agent_bridge_state(repo_root):
        return repo_root

    configured = os.environ.get(AUTOMATION_STATE_ROOT_ENV)
    candidates: list[Path] = []
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.append(Path.home() / "Development" / "aragora")

    for candidate in candidates:
        if _has_agent_bridge_state(candidate):
            return candidate
    return repo_root


def _automation_state_default_path(state_root: Path, default_relative: Path) -> Path:
    expanded = state_root.expanduser()
    if default_relative.parts[:1] == (".aragora",) and expanded.name == ".aragora":
        return expanded.joinpath(*default_relative.parts[1:])
    return expanded / default_relative


def resolve_heartbeat_path(*, repo_root: Path, explicit: Path | None = None) -> Path:
    if explicit is not None:
        return explicit
    return _automation_state_default_path(
        _automation_state_root(repo_root), HEARTBEAT_RELATIVE_PATH
    )


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
        help=(
            f"Repo root (default: {DEFAULT_REPO_ROOT}). Heartbeat state defaults to this "
            f"root's .aragora, then ${AUTOMATION_STATE_ROOT_ENV}, then ~/Development/aragora."
        ),
    )
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    heartbeat_path = resolve_heartbeat_path(repo_root=args.repo_root, explicit=args.heartbeat_path)
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


def _coerce_exit_code(code: object) -> int:
    if code is None:
        return 0
    if isinstance(code, int):
        return code
    print(code, file=sys.stderr)
    return 1


def _cli_entrypoint() -> int:
    try:
        rc = main()
    except BrokenPipeError:
        return 0
    except SystemExit as exc:
        rc = _coerce_exit_code(exc.code)
    try:
        sys.stdout.flush()
    except BrokenPipeError:
        os._exit(0)
    return rc


if __name__ == "__main__":
    raise SystemExit(_cli_entrypoint())
