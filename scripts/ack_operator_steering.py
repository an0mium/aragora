#!/usr/bin/env python3
"""Acknowledge operator-steering messages after terminal read outcomes.

Read receipts are intentionally advisory sidecars: top-level ``*.json``
messages remain pending until an explicit ack/move step runs. This tool is that
bounded move step. It only moves messages whose current content is bound to a
terminal read receipt.
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

try:
    import agent_bridge_sessions  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - compatibility for stale script copies.
    agent_bridge_sessions = None  # type: ignore[assignment]

REPO_ROOT = Path(__file__).resolve().parents[1]
READ_RECEIPT_SCHEMA_VERSION = "aragora-operator-steering-read-receipt/1.0"
ACK_SCHEMA_VERSION = "aragora-operator-steering-ack/1.0"
TERMINAL_OUTCOMES = {"obeyed", "stale", "superseded", "completed"}


def _default_state_dir() -> Path:
    configured = os.environ.get("ARAGORA_AUTOMATION_STATE_ROOT")
    if configured:
        root = Path(configured).expanduser()
        return root if root.name == ".aragora" else root / ".aragora"
    if agent_bridge_sessions is not None:
        try:
            return agent_bridge_sessions.resolve_canonical_repo_root(REPO_ROOT) / ".aragora"
        except (OSError, RuntimeError, ValueError):
            pass
    return REPO_ROOT / ".aragora"


def _default_steering_inbox_root() -> Path:
    return _default_state_dir() / "operator-steering"


def _default_lane_registry_path() -> Path:
    return _default_state_dir() / "agent-bridge" / "lanes.json"


def _now_utc_iso() -> str:
    return (
        datetime.datetime.now(datetime.timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


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


def _load_json_object(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _message_hash(path: Path) -> tuple[str, str, bool] | None:
    payload = _load_json_object(path)
    if payload is None:
        return None
    return steering_writer.verify_message_sha256(payload)


def _receipt_files(inbox: Path) -> list[Path]:
    receipt_dir = inbox / "_read_receipts"
    if not receipt_dir.is_dir():
        return []
    return sorted(path for path in receipt_dir.glob("*.json") if path.is_file())


def _matching_terminal_receipt(
    *,
    inbox: Path,
    message_filename: str,
    expected_sha256: str,
) -> tuple[Path | None, str | None, list[dict[str, Any]]]:
    candidates: list[tuple[Path, dict[str, Any]]] = []
    for path in _receipt_files(inbox):
        payload = _load_json_object(path)
        if payload is None:
            continue
        if payload.get("schema_version") != READ_RECEIPT_SCHEMA_VERSION:
            continue
        if payload.get("message_filename") != message_filename:
            continue
        candidates.append((path, payload))

    summaries = [
        {
            "filename": path.name,
            "message_sha256": payload.get("message_sha256"),
            "outcome": payload.get("outcome"),
            "read_at_utc": payload.get("read_at_utc"),
        }
        for path, payload in candidates
    ]
    sha_matches = [
        (path, payload)
        for path, payload in candidates
        if payload.get("message_sha256") == expected_sha256
    ]
    terminal = [
        (path, payload)
        for path, payload in sha_matches
        if str(payload.get("outcome") or "") in TERMINAL_OUTCOMES
    ]
    if terminal:
        terminal.sort(
            key=lambda item: (
                str(item[1].get("read_at_utc") or ""),
                item[0].name,
            )
        )
        return terminal[-1][0], None, summaries
    if candidates and not sha_matches:
        return None, "sha_mismatch", summaries
    if sha_matches:
        return None, "non_terminal_outcome", summaries
    return None, "missing_receipt", summaries


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, indent=2, sort_keys=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp-", suffix=".json", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(body)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _acked_sidecar_path(acked_dir: Path, message_filename: str) -> Path:
    return acked_dir / f"{message_filename}.ack.json"


def _select_messages(inbox: Path, message: str | None) -> list[Path]:
    if message:
        if "/" in message or "\\" in message or Path(message).name != message:
            raise ValueError("--message must be a plain filename")
        return [inbox / message]
    if not inbox.is_dir():
        return []
    return sorted(path for path in inbox.glob("*.json") if path.is_file())


def _acked_noop(acked_dir: Path, message_filename: str) -> dict[str, Any] | None:
    acked_message = acked_dir / message_filename
    sidecar = _acked_sidecar_path(acked_dir, message_filename)
    if not acked_message.exists():
        return None
    return {
        "message_filename": message_filename,
        "status": "already_acked",
        "acked_message_path": str(acked_message),
        "ack_path": str(sidecar) if sidecar.exists() else None,
        "ok": True,
    }


def _plan_message(inbox: Path, acked_dir: Path, path: Path) -> dict[str, Any]:
    already = _acked_noop(acked_dir, path.name)
    if already and not path.exists():
        return already
    if not path.exists():
        return {
            "message_filename": path.name,
            "status": "refused",
            "reason": "message_not_found",
            "ok": False,
        }
    if already:
        return {
            "message_filename": path.name,
            "status": "refused",
            "reason": "acked_copy_already_exists",
            "ok": False,
            "acked_message_path": already["acked_message_path"],
        }

    hash_info = _message_hash(path)
    if hash_info is None:
        return {
            "message_filename": path.name,
            "status": "refused",
            "reason": "message_unreadable",
            "ok": False,
        }
    claimed, recomputed, valid = hash_info
    if not valid:
        return {
            "message_filename": path.name,
            "status": "refused",
            "reason": "message_sha_mismatch",
            "claimed_message_sha256": claimed,
            "recomputed_message_sha256": recomputed,
            "ok": False,
        }

    receipt, reason, receipt_candidates = _matching_terminal_receipt(
        inbox=inbox,
        message_filename=path.name,
        expected_sha256=claimed,
    )
    if receipt is None:
        return {
            "message_filename": path.name,
            "status": "refused",
            "reason": reason or "missing_receipt",
            "message_sha256": claimed,
            "receipt_candidates": receipt_candidates,
            "ok": False,
        }

    return {
        "message_filename": path.name,
        "status": "ready",
        "message_path": str(path),
        "message_sha256": claimed,
        "receipt_filename": receipt.name,
        "receipt_path": str(receipt),
        "acked_message_path": str(acked_dir / path.name),
        "ack_path": str(_acked_sidecar_path(acked_dir, path.name)),
        "ok": True,
    }


def _ack_message(
    *,
    owner_session: str,
    acked_by_session: str,
    plan: dict[str, Any],
    acked_at_utc: str,
) -> dict[str, Any]:
    if plan["status"] == "already_acked":
        return plan

    message_path = Path(str(plan["message_path"]))
    acked_message_path = Path(str(plan["acked_message_path"]))
    ack_path = Path(str(plan["ack_path"]))
    receipt_path = Path(str(plan["receipt_path"]))
    ack_payload = {
        "schema_version": ACK_SCHEMA_VERSION,
        "acked_at_utc": acked_at_utc,
        "acked_by_session": acked_by_session,
        "owner_session": owner_session,
        "message_filename": plan["message_filename"],
        "message_sha256": plan["message_sha256"],
        "receipt_filename": plan["receipt_filename"],
        "receipt_path": str(receipt_path),
    }

    acked_message_path.parent.mkdir(parents=True, exist_ok=True)
    os.replace(message_path, acked_message_path)
    _atomic_write_json(ack_path, ack_payload)
    result = dict(plan)
    result["status"] = "acked"
    return result


def _default_acked_by_session(owner_session: str) -> str:
    return (
        os.environ.get("ARAGORA_SESSION_ID") or os.environ.get("CODEX_SESSION_ID") or owner_session
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ack_operator_steering.py",
        description="Move terminally handled operator-steering messages out of pending state.",
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--to", metavar="OWNER_SESSION", help="Ack this owner_session mailbox.")
    target.add_argument("--lane-id", help="Resolve owner_session from lane id.")
    target.add_argument("--pr", type=int, help="Resolve owner_session from PR number.")
    target.add_argument("--branch", help="Resolve owner_session from branch name.")
    parser.add_argument("--message", help="Ack only this pending message filename.")
    parser.add_argument(
        "--acked-by-session",
        default=None,
        help="Session id recorded in ack sidecars. Defaults to env/session target.",
    )
    parser.add_argument("--apply", action="store_true", help="Move messages into _acked/.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable output.")
    parser.add_argument(
        "--steering-inbox-root",
        type=Path,
        default=_default_steering_inbox_root(),
        help="Override .aragora/operator-steering root.",
    )
    parser.add_argument(
        "--registry-path",
        type=Path,
        default=_default_lane_registry_path(),
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
        inbox = steering_writer.validate_to_session(
            owner_session, steering_inbox_root=args.steering_inbox_root
        )
        selected = _select_messages(inbox, args.message)
    except ValueError as exc:
        out = {
            "ok": False,
            "error": str(exc),
            "message_count": 0,
            "acked_count": 0,
            "already_acked_count": 0,
            "refused_count": 0,
            "messages": [],
        }
        if args.json:
            print(json.dumps(out, indent=2, sort_keys=True))
        else:
            print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    acked_dir = inbox / "_acked"
    plans = [_plan_message(inbox, acked_dir, path) for path in selected]
    refused = [plan for plan in plans if not plan.get("ok")]
    results = plans
    if args.apply and not refused:
        acked_at = _now_utc_iso()
        acked_by = args.acked_by_session or _default_acked_by_session(owner_session)
        results = [
            _ack_message(
                owner_session=owner_session,
                acked_by_session=acked_by,
                plan=plan,
                acked_at_utc=acked_at,
            )
            for plan in plans
        ]

    out = {
        "ok": not refused,
        "dry_run": not args.apply,
        "applied": bool(args.apply and not refused),
        "owner_session": owner_session,
        "resolved_via": resolved_via,
        "lane_id": lane.get("lane_id") if isinstance(lane, dict) else args.lane_id,
        "pr_number": lane.get("pr_number") if isinstance(lane, dict) else args.pr,
        "branch": lane.get("branch") if isinstance(lane, dict) else args.branch,
        "steering_inbox_path": str(inbox),
        "message_count": len(results),
        "acked_count": sum(1 for result in results if result.get("status") == "acked"),
        "already_acked_count": sum(
            1 for result in results if result.get("status") == "already_acked"
        ),
        "refused_count": len(refused),
        "messages": results,
    }

    if args.json:
        print(json.dumps(out, indent=2, sort_keys=True))
    else:
        print(f"owner_session: {owner_session}")
        print(f"steering_inbox_path: {inbox}")
        print(f"message_count: {out['message_count']}")
        print(f"acked_count: {out['acked_count']}")
        print(f"already_acked_count: {out['already_acked_count']}")
        print(f"refused_count: {out['refused_count']}")
        for item in results:
            suffix = f" reason={item.get('reason')}" if item.get("reason") else ""
            print(f"- {item.get('message_filename')} status={item.get('status')}{suffix}")
    return 0 if not refused else 1


if __name__ == "__main__":
    sys.exit(main())
