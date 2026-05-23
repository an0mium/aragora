#!/usr/bin/env python3
"""Resolve stale lane conflict rows with append-only receipts.

The resolver only handles the safe case where a ``status=conflict`` row points
at an owner session that no longer has an active lane. It never deletes rows;
``--apply`` marks the conflict row ``superseded`` and writes a sidecar receipt.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import secrets
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
REGISTRY_RELATIVE_PATH = Path(".aragora") / "agent-bridge" / "lanes.json"
RECEIPT_RELATIVE_DIR = Path(".aragora") / "agent-bridge" / "conflict-resolution-receipts"
RECEIPT_SCHEMA_VERSION = "aragora-lane-conflict-resolution/1.0"
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
INACTIVE_OWNER_STATUSES = {"released", "completed", "superseded"}


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


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
def _registry_write_lock(path: Path) -> Iterator[None]:
    """Serialize conflict-resolution registry writes."""

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


def _owner_is_inactive(rows: list[dict[str, Any]], owner_session: str) -> bool:
    owner_rows = [row for row in rows if str(row.get("owner_session") or "") == owner_session]
    if not owner_rows:
        return False
    return all(str(row.get("status") or "") in INACTIVE_OWNER_STATUSES for row in owner_rows)


def _unknown_conflict_sessions_from_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    known_sessions = {str(row.get("owner_session") or "") for row in rows}
    unknown: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("status") or "") != "conflict":
            continue
        conflict_session = str(row.get("conflict_session") or "")
        if conflict_session and conflict_session not in known_sessions:
            unknown.append(
                {
                    "lane_id": row.get("lane_id"),
                    "owner_session": row.get("owner_session"),
                    "conflict_session": conflict_session,
                    "conflict_reason": row.get("conflict_reason"),
                    "current_status": row.get("status"),
                    "resolution": "requires_manual_review_unknown_conflict_session",
                }
            )
    return unknown


def _find_resolvable_conflicts_from_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("status") or "") != "conflict":
            continue
        conflict_session = str(row.get("conflict_session") or "")
        if not conflict_session:
            continue
        if _owner_is_inactive(rows, conflict_session):
            candidates.append(
                {
                    "lane_id": row.get("lane_id"),
                    "owner_session": row.get("owner_session"),
                    "conflict_session": conflict_session,
                    "conflict_reason": row.get("conflict_reason"),
                    "current_status": row.get("status"),
                    "new_status": "superseded",
                    "resolution": "conflict_session_has_only_inactive_rows",
                }
            )
    return candidates


def find_resolvable_conflicts(registry_path: Path) -> list[dict[str, Any]]:
    return _find_resolvable_conflicts_from_rows(_read_rows(registry_path))


def _write_receipt(
    *,
    receipt_dir: Path,
    receipt: dict[str, Any],
) -> Path:
    receipt_dir.mkdir(parents=True, exist_ok=True)
    ts = str(receipt["resolved_at_utc"]).replace(":", "-")
    path = receipt_dir / f"{ts}-{secrets.token_hex(4)}.json"
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp-", suffix=".json", dir=str(receipt_dir))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(receipt, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return path


def resolve_conflicts(
    *,
    registry_path: Path,
    receipt_dir: Path,
    apply: bool = False,
    resolved_at: str | None = None,
) -> dict[str, Any]:
    resolved_at = resolved_at or _utc_now_iso()
    receipt_paths: list[str] = []
    with _registry_write_lock(registry_path):
        rows = _read_rows(registry_path)
        candidates = _find_resolvable_conflicts_from_rows(rows)
        unknown_conflicts = _unknown_conflict_sessions_from_rows(rows)
        if apply and candidates:
            candidate_keys = {
                (
                    str(candidate.get("lane_id") or ""),
                    str(candidate.get("owner_session") or ""),
                    str(candidate.get("conflict_session") or ""),
                )
                for candidate in candidates
            }
            out_rows: list[dict[str, Any]] = []
            for row in rows:
                row = dict(row)
                row_key = (
                    str(row.get("lane_id") or ""),
                    str(row.get("owner_session") or ""),
                    str(row.get("conflict_session") or ""),
                )
                if row_key in candidate_keys and row.get("status") == "conflict":
                    row["status"] = "superseded"
                    row["updated_at"] = resolved_at
                    row["last_steering_outcome"] = "superseded"
                    receipt = {
                        "schema_version": RECEIPT_SCHEMA_VERSION,
                        "lane_id": row.get("lane_id"),
                        "owner_session": row.get("owner_session"),
                        "conflict_session": row.get("conflict_session"),
                        "conflict_reason": row.get("conflict_reason"),
                        "old_status": "conflict",
                        "new_status": "superseded",
                        "resolved_at_utc": resolved_at,
                        "resolution": "conflict_session_has_only_inactive_rows",
                    }
                    receipt_paths.append(
                        str(_write_receipt(receipt_dir=receipt_dir, receipt=receipt))
                    )
                out_rows.append(row)
            _atomic_write(registry_path, out_rows)

    return {
        "registry_path": str(registry_path),
        "dry_run": not apply,
        "resolved_count": len(candidates) if apply else 0,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "unknown_session_count": len(unknown_conflicts),
        "candidates_unknown": unknown_conflicts,
        "receipt_paths": receipt_paths,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--dry-run", action="store_true", default=True)
    action.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--registry-path",
        type=Path,
        default=DEFAULT_REPO_ROOT / REGISTRY_RELATIVE_PATH,
    )
    parser.add_argument(
        "--receipt-dir",
        type=Path,
        default=DEFAULT_REPO_ROOT / RECEIPT_RELATIVE_DIR,
    )
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = resolve_conflicts(
        registry_path=args.registry_path,
        receipt_dir=args.receipt_dir,
        apply=bool(args.apply),
    )
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        verb = "resolved" if args.apply else "candidate"
        print(f"{verb}_count={result['resolved_count' if args.apply else 'candidate_count']}")
        for candidate in result["candidates"]:
            print(
                f"- lane_id={candidate['lane_id']} conflict_session="
                f"{candidate['conflict_session']} -> superseded"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
