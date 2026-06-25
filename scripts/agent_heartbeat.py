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
FINALIZER_RECEIPTS_RELATIVE_PATH = (
    Path(".aragora") / "agent-bridge" / "heartbeat-finalizer-receipts.jsonl"
)
AUTOMATION_STATE_ROOT_ENV = "ARAGORA_AUTOMATION_STATE_ROOT"
SAFE_OWNER_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
TERMINAL_OUTCOMES = frozenset({"completed", "failed", "cancelled", "handoff"})


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _validate_owner_session(owner_session: str) -> None:
    if (
        not owner_session
        or owner_session in {".", ".."}
        or owner_session.startswith(".")
        or not SAFE_OWNER_RE.fullmatch(owner_session)
    ):
        raise ValueError(
            "unsafe owner_session: use a non-empty alphanumeric/dash/dot/underscore slug"
        )


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


def resolve_finalizer_receipt_path(*, repo_root: Path, explicit: Path | None = None) -> Path:
    if explicit is not None:
        return explicit
    return _automation_state_default_path(
        _automation_state_root(repo_root), FINALIZER_RECEIPTS_RELATIVE_PATH
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


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        json.dump(row, handle, sort_keys=True)
        handle.write("\n")


def _heartbeat_identity_matches(row: dict[str, Any], *, lane_id: str, owner_session: str) -> bool:
    return (
        str(row.get("lane_id") or "") == lane_id
        and str(row.get("owner_session") or "") == owner_session
    )


def _thread_id(row: dict[str, Any]) -> str:
    return str(row.get("thread_id") or "").strip()


def _superseded_thread_ids(row: dict[str, Any]) -> list[str]:
    value = row.get("superseded_thread_ids")
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _with_superseded_identity(
    row: dict[str, Any],
    existing: dict[str, Any],
) -> dict[str, Any]:
    superseded = _superseded_thread_ids(existing)
    existing_thread = _thread_id(existing)
    incoming_thread = _thread_id(row)
    if existing_thread and existing_thread != incoming_thread:
        superseded.append(existing_thread)
    if not superseded:
        return row
    unique = list(dict.fromkeys(superseded))[-8:]
    return {**row, "superseded_thread_ids": unique}


def _same_comparable_identity(existing: dict[str, Any], incoming: dict[str, Any]) -> bool:
    """Return true only when all available non-thread identity fields agree."""

    comparable = False
    for key in ("pid", "cwd", "worktree", "branch", "pr_number"):
        existing_value = existing.get(key)
        incoming_value = incoming.get(key)
        if existing_value is None or incoming_value is None:
            continue
        if isinstance(existing_value, str) and not existing_value:
            continue
        if isinstance(incoming_value, str) and not incoming_value:
            continue
        comparable = True
        if existing_value != incoming_value:
            return False
    return comparable


def _has_comparable_identity_delta(
    existing: dict[str, Any],
    incoming: dict[str, Any],
) -> bool:
    """Return true when a non-thread identity field proves a different run."""

    for key in ("pid", "cwd", "worktree", "branch", "pr_number"):
        existing_value = existing.get(key)
        incoming_value = incoming.get(key)
        if existing_value is None or incoming_value is None:
            continue
        if isinstance(existing_value, str) and not existing_value:
            continue
        if isinstance(incoming_value, str) and not incoming_value:
            continue
        if existing_value != incoming_value:
            return True
    return False


def _same_finalizer_identity(existing: dict[str, Any], receipt: dict[str, Any]) -> bool:
    """Match legacy finalizers only when durable identity, not ambient cwd, agrees."""

    comparable = False
    for key in ("pid", "worktree", "branch", "pr_number"):
        existing_value = existing.get(key)
        receipt_value = receipt.get(key)
        if existing_value is None or receipt_value is None:
            continue
        if isinstance(existing_value, str) and not existing_value:
            continue
        if isinstance(receipt_value, str) and not receipt_value:
            continue
        comparable = True
        if existing_value != receipt_value:
            return False
    return comparable


def _has_finalizer_identity_delta(existing: dict[str, Any], receipt: dict[str, Any]) -> bool:
    for key in ("worktree", "branch", "pr_number"):
        existing_value = existing.get(key)
        receipt_value = receipt.get(key)
        if existing_value is None or receipt_value is None:
            continue
        if isinstance(existing_value, str) and not existing_value:
            continue
        if isinstance(receipt_value, str) and not receipt_value:
            continue
        if existing_value != receipt_value:
            return True
    return False


def _same_pidless_finalizer_identity(
    existing: dict[str, Any],
    receipt: dict[str, Any],
) -> bool:
    if _has_finalizer_identity_delta(existing, receipt):
        return False
    existing_worktree = str(existing.get("worktree") or "").strip()
    receipt_worktree = str(receipt.get("worktree") or "").strip()
    return bool(existing_worktree and existing_worktree == receipt_worktree)


def _terminal_heartbeat_fields(
    receipt: dict[str, Any],
    *,
    receipt_recorded: bool,
    receipt_error: str = "",
) -> dict[str, Any]:
    fields = {
        "terminal": True,
        "terminal_outcome": receipt["outcome"],
        "terminal_reason": receipt["reason"],
        "terminal_finalized_at": receipt["finalized_at"],
        "terminal_receipt_recorded": receipt_recorded,
    }
    if receipt_error:
        fields["terminal_receipt_error"] = receipt_error
    return fields


def _is_terminal_heartbeat(row: dict[str, Any]) -> bool:
    return bool(
        row.get("terminal") is True
        or row.get("terminal_outcome")
        or row.get("terminal_finalized_at")
    )


def _should_preserve_terminal_heartbeat(
    existing: dict[str, Any],
    incoming: dict[str, Any],
) -> bool:
    """Keep terminal owner proof unless the renewal carries a new run identity."""

    if not _is_terminal_heartbeat(existing):
        return False

    existing_thread = _thread_id(existing)
    incoming_thread = _thread_id(incoming)
    if existing_thread and incoming_thread:
        return existing_thread == incoming_thread
    if incoming_thread and not existing_thread:
        return False
    if existing_thread and not incoming_thread:
        return not _has_comparable_identity_delta(existing, incoming)

    existing_pid = existing.get("pid")
    incoming_pid = incoming.get("pid")
    if existing_pid is not None and incoming_pid is not None and existing_pid != incoming_pid:
        return False

    # PID-only same-PID or pidless terminal renewals are ambiguous: the late
    # renewal might be an in-flight heartbeat, a pidless caller, PID reuse, or
    # a new wrapper with no old run id to compare against. Preserve the
    # receipt-backed terminal state unless a comparable wrapper run id or
    # clearly different PID proves a relaunch.
    return True


def _should_preserve_existing_heartbeat(
    existing: dict[str, Any],
    incoming: dict[str, Any],
) -> bool:
    if _should_preserve_terminal_heartbeat(existing, incoming):
        return True

    existing_thread = _thread_id(existing)
    incoming_thread = _thread_id(incoming)
    if incoming_thread and incoming_thread in _superseded_thread_ids(existing):
        return True
    if existing_thread and not incoming_thread:
        return _same_comparable_identity(existing, incoming)
    if existing_thread and incoming_thread and existing_thread != incoming_thread:
        return False

    return False


def _finalizer_matches_heartbeat(existing: dict[str, Any], *, receipt: dict[str, Any]) -> bool:
    if not _heartbeat_identity_matches(
        existing,
        lane_id=str(receipt["lane_id"]),
        owner_session=str(receipt["owner_session"]),
    ):
        return False

    existing_thread = _thread_id(existing)
    receipt_thread = _thread_id(receipt)
    if receipt_thread and receipt_thread in _superseded_thread_ids(existing):
        return False
    if existing_thread and receipt_thread and existing_thread != receipt_thread:
        return False
    same_thread_identity = bool(existing_thread and receipt_thread)
    if existing_thread and not receipt_thread:
        return _same_finalizer_identity(existing, receipt)
    if receipt_thread and not existing_thread:
        return _same_finalizer_identity(existing, receipt)

    existing_pid = existing.get("pid")
    receipt_pid = receipt.get("pid")
    if existing_pid is not None and receipt_pid is None:
        if same_thread_identity:
            return not _has_finalizer_identity_delta(existing, receipt)
        return _same_pidless_finalizer_identity(existing, receipt)
    if existing_pid is None and receipt_pid is not None:
        return _same_finalizer_identity(existing, receipt)
    if existing_pid is not None and receipt_pid is not None and existing_pid != receipt_pid:
        return False

    return True


def _mark_matching_heartbeat_terminal(
    heartbeat_path: Path,
    *,
    receipt: dict[str, Any],
    receipt_recorded: bool,
    receipt_error: str = "",
) -> None:
    rows = _read_rows(heartbeat_path)
    if not rows:
        return
    terminal_fields = _terminal_heartbeat_fields(
        receipt,
        receipt_recorded=receipt_recorded,
        receipt_error=receipt_error,
    )
    out: list[dict[str, Any]] = []
    changed = False
    for existing in rows:
        if _finalizer_matches_heartbeat(existing, receipt=receipt):
            out.append({**existing, **terminal_fields})
            changed = True
        else:
            out.append(existing)
    if changed:
        _atomic_write(heartbeat_path, out)


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
                if _should_preserve_existing_heartbeat(existing, row):
                    out.append(existing)
                    row = existing
                else:
                    row = _with_superseded_identity(row, existing)
                    out.append(row)
                replaced = True
            else:
                out.append(existing)
        if not replaced:
            out.append(row)
        _atomic_write(heartbeat_path, out)
    return row


def record_finalizer_receipt(
    *,
    heartbeat_path: Path,
    receipt_path: Path,
    lane_id: str,
    owner_session: str,
    outcome: str,
    reason: str,
    thread_id: str = "",
    pid: int | None = None,
    cwd: str = "",
    worktree: str = "",
    branch: str = "",
    pr_number: int | None = None,
    finalized_at: str | None = None,
) -> dict[str, Any]:
    """Append terminal owner proof and mark the matching heartbeat non-live."""
    if not lane_id:
        raise ValueError("lane_id must not be empty")
    _validate_owner_session(owner_session)
    if outcome not in TERMINAL_OUTCOMES:
        allowed = ", ".join(sorted(TERMINAL_OUTCOMES))
        raise ValueError(f"outcome must be one of: {allowed}")
    if not reason.strip():
        raise ValueError("reason must not be empty")

    row = _compact(
        {
            "schema_version": "aragora-agent-finalizer-receipt/1.0",
            "lane_id": lane_id,
            "owner_session": owner_session,
            "thread_id": thread_id,
            "pid": pid,
            "cwd": cwd or os.getcwd(),
            "worktree": worktree,
            "branch": branch,
            "pr_number": pr_number,
            "outcome": outcome,
            "reason": reason,
            "finalized_at": finalized_at or _utc_now_iso(),
        }
    )

    with _heartbeat_write_lock(heartbeat_path):
        _mark_matching_heartbeat_terminal(
            heartbeat_path,
            receipt=row,
            receipt_recorded=False,
        )
        try:
            _append_jsonl(receipt_path, row)
        except Exception as exc:
            _mark_matching_heartbeat_terminal(
                heartbeat_path,
                receipt=row,
                receipt_recorded=False,
                receipt_error=f"{type(exc).__name__}: {exc}",
            )
            raise
        _mark_matching_heartbeat_terminal(
            heartbeat_path,
            receipt=row,
            receipt_recorded=True,
        )
    return row


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--finalize",
        action="store_true",
        help="Append a terminal owner lifecycle receipt instead of a heartbeat row.",
    )
    parser.add_argument("--lane-id", required=True)
    parser.add_argument("--owner-session", required=True)
    parser.add_argument("--thread-id", default=None)
    parser.add_argument("--pid", type=int, default=None)
    parser.add_argument("--cwd", default=os.getcwd())
    parser.add_argument("--worktree", default="")
    parser.add_argument("--branch", default="")
    parser.add_argument("--pr-number", type=int, default=None)
    parser.add_argument("--last-seen-at", default=None)
    parser.add_argument(
        "--outcome",
        choices=sorted(TERMINAL_OUTCOMES),
        default=None,
        help="Terminal outcome for --finalize receipts.",
    )
    parser.add_argument("--reason", default="", help="Human-readable terminal receipt reason.")
    parser.add_argument("--finalized-at", default=None)
    parser.add_argument(
        "--heartbeat-path",
        type=Path,
        default=None,
        help="Override heartbeat sidecar path.",
    )
    parser.add_argument(
        "--finalizer-receipt-path",
        type=Path,
        default=None,
        help="Override finalizer receipt JSONL path.",
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
    try:
        pid = args.pid
        thread_id = args.thread_id
        if args.finalize:
            # A finalizer commonly runs in its own short-lived process. Treat
            # ambient process identity as non-identity unless the caller
            # intentionally supplies --pid/--thread-id for a stricter match.
            if thread_id is None:
                thread_id = ""
            receipt_path = resolve_finalizer_receipt_path(
                repo_root=args.repo_root,
                explicit=args.finalizer_receipt_path,
            )
            if args.outcome is None:
                raise ValueError("--outcome is required with --finalize")
            row = record_finalizer_receipt(
                heartbeat_path=resolve_heartbeat_path(
                    repo_root=args.repo_root,
                    explicit=args.heartbeat_path,
                ),
                receipt_path=receipt_path,
                lane_id=args.lane_id,
                owner_session=args.owner_session,
                thread_id=thread_id,
                pid=pid,
                cwd=args.cwd,
                worktree=args.worktree,
                branch=args.branch,
                pr_number=args.pr_number,
                outcome=args.outcome,
                reason=args.reason,
                finalized_at=args.finalized_at,
            )
        else:
            heartbeat_path = resolve_heartbeat_path(
                repo_root=args.repo_root, explicit=args.heartbeat_path
            )
            if thread_id is None:
                thread_id = os.environ.get("CODEX_THREAD_ID", "")
            row = record_heartbeat(
                heartbeat_path=heartbeat_path,
                lane_id=args.lane_id,
                owner_session=args.owner_session,
                thread_id=thread_id,
                pid=pid if pid is not None else os.getpid(),
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
    elif args.finalize:
        print(
            f"finalizer lane_id={row['lane_id']} owner_session={row['owner_session']} "
            f"outcome={row['outcome']} finalized_at={row['finalized_at']}"
        )
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
