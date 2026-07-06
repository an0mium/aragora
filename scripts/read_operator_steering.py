#!/usr/bin/env python3
"""Read, receipt, and explicitly ack operator-steering messages.

Read receipts are sidecar proof-of-read records: top-level ``*.json``
messages remain in place and still count as pending. Ack mode is the explicit
move protocol for messages whose newest valid read receipt records a terminal
outcome.
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
OUTCOME_CHOICES = ("read", "obeyed", "held", "stale", "superseded", "blocked", "completed")
TERMINAL_ACK_OUTCOMES = frozenset({"obeyed", "stale", "superseded", "completed"})


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


STEERING_INBOX_ROOT_DEFAULT = _default_steering_inbox_root()
LANE_REGISTRY_DEFAULT = _default_lane_registry_path()


def _now_utc_iso() -> str:
    return (
        datetime.datetime.now(datetime.timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _parse_utc_timestamp(value: Any) -> datetime.datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=datetime.timezone.utc)
    return parsed.astimezone(datetime.timezone.utc)


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
    steering_inbox_root: Path | None = None,
) -> Path:
    if steering_inbox_root is None:
        steering_inbox_root = _default_steering_inbox_root()
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


def _read_receipt_files(inbox: Path) -> list[Path]:
    receipt_dir = inbox / "_read_receipts"
    return sorted(receipt_dir.glob("*.json")) if receipt_dir.is_dir() else []


def _load_json_object(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        return None, f"read_failed:{type(exc).__name__}:{exc}"
    except json.JSONDecodeError as exc:
        return None, f"invalid_json:{exc.msg}"
    if not isinstance(data, dict):
        return None, "invalid_shape:not_object"
    return data, None


def _receipt_summary(path: Path, receipt: dict[str, Any]) -> dict[str, Any]:
    return {
        "filename": path.name,
        "path": str(path),
        "schema_version": receipt.get("schema_version"),
        "message_filename": receipt.get("message_filename"),
        "message_sha256": receipt.get("message_sha256"),
        "outcome": receipt.get("outcome"),
        "read_at_utc": receipt.get("read_at_utc"),
    }


def _terminal_ack_plan_for_message(
    message_path: Path,
    *,
    inbox: Path,
    owner_session: str,
    ack_by_session: str,
    now: datetime.datetime,
) -> tuple[dict[str, Any] | None, str | None]:
    message, message_error = _load_json_object(message_path)
    if message is None:
        return None, f"{message_path.name}:message_{message_error}"

    claimed_sha, recomputed_sha, sha_ok = steering_writer.verify_message_sha256(message)
    if not sha_ok or not claimed_sha:
        return (
            None,
            f"{message_path.name}:message_sha256_mismatch"
            f":claimed={claimed_sha}:recomputed={recomputed_sha}",
        )

    sent_at = _parse_utc_timestamp(message.get("sent_at_utc"))
    if sent_at is None:
        return None, f"{message_path.name}:invalid_message_sent_at"

    matching: list[tuple[Path, dict[str, Any], datetime.datetime]] = []
    for receipt_path in _read_receipt_files(inbox):
        receipt, receipt_error = _load_json_object(receipt_path)
        if receipt is None:
            continue
        if receipt.get("schema_version") != READ_RECEIPT_SCHEMA_VERSION:
            continue
        if receipt.get("message_filename") != message_path.name:
            continue

        read_at = _parse_utc_timestamp(receipt.get("read_at_utc"))
        if read_at is None:
            return None, f"{message_path.name}:{receipt_path.name}:invalid_read_at_utc"
        if read_at > now:
            return None, f"{message_path.name}:{receipt_path.name}:future_dated_receipt"
        if read_at < sent_at:
            return None, f"{message_path.name}:{receipt_path.name}:receipt_before_message"
        if str(receipt.get("message_sha256") or "") != claimed_sha:
            return None, f"{message_path.name}:{receipt_path.name}:message_sha256_mismatch"
        matching.append((receipt_path, receipt, read_at))

    if not matching:
        return None, f"{message_path.name}:no_bound_read_receipt"

    newest_time = max(read_at for _path, _receipt, read_at in matching)
    newest = [(path, receipt) for path, receipt, read_at in matching if read_at == newest_time]
    if len(newest) != 1:
        names = ",".join(path.name for path, _receipt in newest)
        return None, f"{message_path.name}:ambiguous_newest_receipt:{names}"

    receipt_path, receipt = newest[0]
    outcome = str(receipt.get("outcome") or "").strip().lower()
    if outcome not in TERMINAL_ACK_OUTCOMES:
        return None, f"{message_path.name}:{receipt_path.name}:nonterminal_outcome:{outcome}"

    acked_dir = inbox / "_acked"
    target_message_path = acked_dir / message_path.name
    if target_message_path.exists():
        return None, f"{message_path.name}:ack_destination_exists"

    ack_at_utc = _now_utc_iso()
    ack_receipt = {
        "schema_version": ACK_SCHEMA_VERSION,
        "owner_session": owner_session,
        "actor": ack_by_session,
        "acked_at_utc": ack_at_utc,
        "message_filename": message_path.name,
        "message_sha256": claimed_sha,
        "message_sent_at_utc": message.get("sent_at_utc"),
        "referenced_read_receipt_filename": receipt_path.name,
        "referenced_read_receipt_path": str(receipt_path),
        "outcome": outcome,
        "acked_message_path": str(target_message_path),
    }
    return (
        {
            "message_path": message_path,
            "acked_message_path": target_message_path,
            "referenced_read_receipt_path": receipt_path,
            "referenced_read_receipt": _receipt_summary(receipt_path, receipt),
            "ack_receipt": ack_receipt,
        },
        None,
    )


def _write_ack_receipt(acked_dir: Path, receipt: dict[str, Any]) -> Path:
    receipt_dir = acked_dir / "_ack_receipts"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    ts = _filename_timestamp(str(receipt.get("acked_at_utc") or _now_utc_iso()))
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


def build_ack_result(
    *,
    owner_session: str,
    read_by_session: str,
    files: Sequence[Path],
    steering_inbox_root: Path,
    apply: bool,
) -> dict[str, Any]:
    inbox = steering_writer.validate_to_session(
        owner_session, steering_inbox_root=steering_inbox_root
    )
    now = datetime.datetime.now(datetime.timezone.utc)
    blockers: list[str] = []
    plans: list[dict[str, Any]] = []
    for path in files:
        plan, blocker = _terminal_ack_plan_for_message(
            path,
            inbox=inbox,
            owner_session=owner_session,
            ack_by_session=read_by_session,
            now=now,
        )
        if blocker:
            blockers.append(blocker)
        elif plan is not None:
            plans.append(plan)

    acked_messages: list[dict[str, Any]] = []
    if apply and plans:
        acked_dir = inbox / "_acked"
        acked_dir.mkdir(parents=True, exist_ok=True)
        for plan in plans:
            message_path = plan["message_path"]
            target_path = plan["acked_message_path"]
            os.replace(message_path, target_path)
            ack_receipt_path = _write_ack_receipt(acked_dir, plan["ack_receipt"])
            acked_messages.append(
                {
                    "message_filename": message_path.name,
                    "acked_message_path": str(target_path),
                    "ack_receipt_path": str(ack_receipt_path),
                    "referenced_read_receipt_path": str(plan["referenced_read_receipt_path"]),
                    "outcome": plan["ack_receipt"]["outcome"],
                }
            )

    return {
        "ack": True,
        "dry_run": not apply,
        "apply": apply,
        "ack_safe": not blockers,
        "ack_count": len(acked_messages) if apply else 0,
        "ack_candidate_count": len(plans),
        "blockers": blockers,
        "ack_candidates": [
            {
                "message_filename": plan["message_path"].name,
                "message_path": str(plan["message_path"]),
                "target_path": str(plan["acked_message_path"]),
                "referenced_read_receipt": plan["referenced_read_receipt"],
                "outcome": plan["ack_receipt"]["outcome"],
            }
            for plan in plans
        ],
        "acked_messages": acked_messages,
    }


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
    parser.add_argument(
        "--ack",
        action="store_true",
        help=(
            "Acknowledge terminally receipted messages by moving them to _acked/. "
            "Defaults to dry-run; use --apply to mutate."
        ),
    )
    apply_group = parser.add_mutually_exclusive_group()
    apply_group.add_argument("--dry-run", action="store_true", help="Validate ack without moving.")
    apply_group.add_argument("--apply", action="store_true", help="Apply --ack move if safe.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable output.")
    parser.add_argument(
        "--quiet-empty",
        action="store_true",
        help="Print nothing and exit 0 when the selected mailbox has no messages.",
    )
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
    except ValueError as exc:
        if args.json and not args.to:
            out = {
                "ok": False,
                "error": str(exc),
                "owner_session": None,
                "resolved_via": None,
                "lane_id": args.lane_id,
                "pr_number": args.pr,
                "branch": args.branch,
                "steering_inbox_root": str(args.steering_inbox_root),
                "registry_path": str(args.registry_path),
                "message_count": 0,
                "receipt_count": 0,
                "messages": [],
                "read_receipt_paths": [],
                "no_receipt": bool(args.no_receipt),
            }
            print(json.dumps(out, indent=2, sort_keys=True))
            return 2
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    files = _message_files(owner_session, steering_inbox_root=args.steering_inbox_root)
    read_by = args.read_by_session or _default_read_by_session(owner_session)
    receipt_paths: list[Path] = []
    ack_result: dict[str, Any] | None = None
    if args.ack:
        ack_result = build_ack_result(
            owner_session=owner_session,
            read_by_session=read_by,
            files=files,
            steering_inbox_root=args.steering_inbox_root,
            apply=bool(args.apply),
        )
    elif not args.no_receipt:
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
    if ack_result is not None:
        out.update(ack_result)
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
        if ack_result is not None:
            if ack_result["blockers"]:
                print("ack_blockers:")
                for blocker in ack_result["blockers"]:
                    print(f"- {blocker}")
            else:
                action = "would ack" if ack_result["dry_run"] else "acked"
                print(f"{action}: {ack_result['ack_candidate_count']}")
    if ack_result is not None and args.apply and ack_result["blockers"]:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
