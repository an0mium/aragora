#!/usr/bin/env python3
"""Read operator-steering messages and write append-only read receipts.

This is a sidecar receipt protocol, not an ack protocol: top-level
``*.json`` messages remain in place and still count as pending until a
future explicit ack/move flow exists.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import secrets
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import identify_lane_owner as owner_lookup
import send_operator_steering as steering_writer

REPO_ROOT = Path(__file__).resolve().parents[1]
STEERING_INBOX_ROOT_DEFAULT = REPO_ROOT / ".aragora" / "operator-steering"
LANE_REGISTRY_DEFAULT = REPO_ROOT / ".aragora" / "agent-bridge" / "lanes.json"
READ_RECEIPT_SCHEMA_VERSION = "aragora-operator-steering-read-receipt/1.0"
OUTCOME_CHOICES = ("read", "obeyed", "held", "stale", "superseded", "blocked", "completed")


def _now_utc_iso() -> str:
    return (
        datetime.datetime.now(datetime.timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _filename_timestamp(iso: str) -> str:
    return iso.replace(":", "-").replace(".", "-")


def _resolve_owner_session(
    *,
    to_session: str | None,
    lane_id: str | None,
    pr: int | None,
    branch: str | None,
    registry_path: Path,
    steering_inbox_root: Path,
) -> tuple[str, str, dict[str, Any] | None]:
    if to_session:
        steering_writer.validate_to_session(to_session, steering_inbox_root=steering_inbox_root)
        return to_session, "direct", None

    records = owner_lookup.load_lane_records(registry_path)
    lane = owner_lookup.find_lane(records, lane_id=lane_id, pr=pr, branch=branch)
    if lane is None:
        raise ValueError("no lane matched the requested selector")
    owner_session = str(lane.get("owner_session") or "")
    if not owner_session:
        raise ValueError("matched lane has no owner_session")
    steering_writer.validate_to_session(owner_session, steering_inbox_root=steering_inbox_root)
    if lane_id:
        resolved_via = "lane-id"
    elif pr is not None:
        resolved_via = "pr"
    else:
        resolved_via = "branch"
    return owner_session, resolved_via, lane


def _message_files(owner_session: str, *, steering_inbox_root: Path) -> list[Path]:
    inbox = steering_writer.validate_to_session(
        owner_session, steering_inbox_root=steering_inbox_root
    )
    if not inbox.is_dir():
        return []
    return sorted((p for p in inbox.glob("*.json") if p.is_file()), key=lambda p: p.name)


def _load_message(path: Path) -> tuple[dict[str, Any], bool]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}, False
    if not isinstance(data, dict):
        return {}, False
    _, _, sha_ok = steering_writer.verify_message_sha256(data)
    return data, sha_ok


def _message_summary(path: Path) -> dict[str, Any]:
    data, sha_ok = _load_message(path)
    return {
        "filename": path.name,
        "path": str(path),
        "schema_version": data.get("schema_version"),
        "to_session": data.get("to_session"),
        "from": data.get("from"),
        "sent_at_utc": data.get("sent_at_utc"),
        "lane_id_hint": data.get("lane_id_hint"),
        "pr_hint": data.get("pr_hint"),
        "priority": data.get("priority"),
        "subject": data.get("subject"),
        "message_sha256": data.get("message_sha256"),
        "sha256_valid": sha_ok,
    }


def build_read_receipt(
    *,
    owner_session: str,
    read_by_session: str,
    message_path: Path,
    outcome: str = "read",
    outcome_note: str | None = None,
    read_at_utc: str | None = None,
) -> dict[str, Any]:
    data, _sha_ok = _load_message(message_path)
    receipt: dict[str, Any] = {
        "schema_version": READ_RECEIPT_SCHEMA_VERSION,
        "owner_session": owner_session,
        "read_by_session": read_by_session,
        "read_at_utc": read_at_utc or _now_utc_iso(),
        "message_filename": message_path.name,
        "message_sha256": data.get("message_sha256"),
        "message_sent_at_utc": data.get("sent_at_utc"),
        "priority": data.get("priority"),
        "lane_id_hint": data.get("lane_id_hint"),
        "pr_hint": data.get("pr_hint"),
        "subject": data.get("subject"),
        "outcome": outcome,
    }
    if outcome_note:
        receipt["outcome_note"] = outcome_note
    return receipt


def write_read_receipt(
    receipt: dict[str, Any],
    *,
    steering_inbox_root: Path = STEERING_INBOX_ROOT_DEFAULT,
) -> Path:
    owner_session = str(receipt.get("owner_session") or "")
    inbox = steering_writer.validate_to_session(
        owner_session, steering_inbox_root=steering_inbox_root
    )
    receipt_dir = inbox / "_read_receipts"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    ts = _filename_timestamp(str(receipt.get("read_at_utc") or _now_utc_iso()))
    final_path = receipt_dir / f"{ts}-{secrets.token_hex(4)}.json"
    body = json.dumps(receipt, indent=2, sort_keys=True)

    fd, tmp_path = tempfile.mkstemp(prefix=".tmp-", suffix=".json", dir=str(receipt_dir))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(body)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, final_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    return final_path


def _default_read_by_session(owner_session: str) -> str:
    return (
        os.environ.get("ARAGORA_SESSION_ID") or os.environ.get("CODEX_SESSION_ID") or owner_session
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="read_operator_steering.py",
        description="Read one operator-steering mailbox and optionally write read receipts.",
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--to", metavar="OWNER_SESSION", help="Read this owner_session mailbox.")
    target.add_argument("--lane-id", help="Resolve owner_session from lane id.")
    target.add_argument("--pr", type=int, help="Resolve owner_session from PR number.")
    target.add_argument("--branch", help="Resolve owner_session from branch name.")
    parser.add_argument(
        "--read-by-session",
        default=None,
        help="Session id recorded as receipt.read_by_session. Defaults to env/session target.",
    )
    parser.add_argument("--outcome", choices=OUTCOME_CHOICES, default="read")
    parser.add_argument("--outcome-note", default=None)
    parser.add_argument("--no-receipt", action="store_true", help="Read/list without writing.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable output.")
    parser.add_argument(
        "--quiet-empty",
        action="store_true",
        help="Print nothing and exit 0 when the selected mailbox has no messages.",
    )
    parser.add_argument(
        "--steering-inbox-root",
        type=Path,
        default=STEERING_INBOX_ROOT_DEFAULT,
        help="Override .aragora/operator-steering root.",
    )
    parser.add_argument(
        "--registry-path",
        type=Path,
        default=LANE_REGISTRY_DEFAULT,
        help="Override .aragora/agent-bridge/lanes.json.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        owner_session, resolved_via, lane = _resolve_owner_session(
            to_session=args.to,
            lane_id=args.lane_id,
            pr=args.pr,
            branch=args.branch,
            registry_path=args.registry_path,
            steering_inbox_root=args.steering_inbox_root,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    files = _message_files(owner_session, steering_inbox_root=args.steering_inbox_root)
    read_by = args.read_by_session or _default_read_by_session(owner_session)
    receipt_paths: list[Path] = []
    if not args.no_receipt:
        for path in files:
            receipt = build_read_receipt(
                owner_session=owner_session,
                read_by_session=read_by,
                message_path=path,
                outcome=args.outcome,
                outcome_note=args.outcome_note,
            )
            receipt_paths.append(
                write_read_receipt(receipt, steering_inbox_root=args.steering_inbox_root)
            )

    out = {
        "owner_session": owner_session,
        "resolved_via": resolved_via,
        "lane_id": lane.get("lane_id") if isinstance(lane, dict) else None,
        "pr_number": lane.get("pr_number") if isinstance(lane, dict) else args.pr,
        "branch": lane.get("branch") if isinstance(lane, dict) else args.branch,
        "steering_inbox_path": str(args.steering_inbox_root / owner_session),
        "message_count": len(files),
        "receipt_count": len(receipt_paths),
        "read_by_session": read_by,
        "messages": [_message_summary(path) for path in files],
        "read_receipt_paths": [str(path) for path in receipt_paths],
        "no_receipt": bool(args.no_receipt),
    }
    if args.quiet_empty and not files:
        return 0
    if args.json:
        print(json.dumps(out, indent=2, sort_keys=True))
    else:
        print(f"owner_session: {owner_session}")
        print(f"steering_inbox_path: {out['steering_inbox_path']}")
        print(f"message_count: {len(files)}")
        print(f"receipt_count: {len(receipt_paths)}")
        for msg in out["messages"]:
            print(
                f"- {msg['filename']} priority={msg['priority']} "
                f"sent_at_utc={msg['sent_at_utc']} sha256_valid={msg['sha256_valid']} "
                f"subject={msg['subject']}"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
