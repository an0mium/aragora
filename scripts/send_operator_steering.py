#!/usr/bin/env python3
"""Atomic steering-message writer for operator → agent-session inbox.

Phase B of the agent-steering primitive (ships under lane id
``P29-steering-mailbox-writer`` because the original P28-B namespace
was contested by concurrent P28-* worktree-inventory work — see
journal rows 29-31).

Writes one JSON message file per invocation into a per-recipient
mailbox dir:

    .aragora/operator-steering/<to_session>/<utc-ts>-<short-uuid>.json

Each message follows the frozen v1.0 schema documented inline and
honoured by Phase A's ``scripts/identify_lane_owner.py``
``steering_inbox_for()`` reader. The recipient session sees the count
in that consolidator's ``pending_message_count`` field next time it
runs Phase 0 of the fan-out prompt. Phase C
(``operator-snapshot`` extension) will also surface the count directly
in the agent_bridge snapshot once it ships.

Atomic write: tempfile + ``os.replace`` so partial writes don't
appear in the inbox glob even under SIGINT. Each message is its own
file — no shared-file lock contention.

Pure stdlib. No ``aragora.*`` imports. NEVER touches GitHub, the lane
registry, or any path outside ``.aragora/operator-steering/``.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import secrets
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
STEERING_INBOX_ROOT_DEFAULT = REPO_ROOT / ".aragora" / "operator-steering"
LANE_REGISTRY_DEFAULT = REPO_ROOT / ".aragora" / "agent-bridge" / "lanes.json"
USER_LANE_REGISTRY_DEFAULT = Path.home() / ".aragora" / "agent-bridge" / "lanes.json"

SCHEMA_VERSION = "aragora-operator-steering/1.0"
SUBJECT_MAX_CHARS = 80
PRIORITY_CHOICES = ("low", "normal", "high", "blocking")
SHORT_UUID_BYTES = 4  # 8 hex chars; collision risk negligible per-second
ACTIVE_STATUSES = {"active", "running", "pending", "queued", "claimed"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_utc_iso() -> str:
    return (
        datetime.datetime.now(datetime.timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _filename_timestamp(iso: str) -> str:
    """Make an ISO timestamp safe for a filesystem filename."""

    return iso.replace(":", "-").replace(".", "-")


def canonical_json(value: Any) -> str:
    """Canonical JSON matching the TS-side helper (sort_keys, no whitespace, UTF-8)."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def derive_subject(body: str, *, max_chars: int = SUBJECT_MAX_CHARS) -> str:
    """First line of body, truncated to ``max_chars``."""

    first_line = body.strip().splitlines()[0] if body.strip() else ""
    return first_line[:max_chars]


def build_message(
    *,
    to_session: str,
    body: str,
    from_label: str = "operator",
    lane_id_hint: str | None = None,
    pr_hint: int | None = None,
    priority: str = "normal",
    sent_at_utc: str | None = None,
) -> dict[str, Any]:
    """Compose the v1.0 message envelope and stamp the binding sha256."""

    ts = sent_at_utc if sent_at_utc is not None else _now_utc_iso()
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "to_session": to_session,
        "from": from_label,
        "sent_at_utc": ts,
        "lane_id_hint": lane_id_hint,
        "pr_hint": pr_hint,
        "priority": priority,
        "subject": derive_subject(body),
        "body": body,
    }
    canonical = canonical_json(payload)
    payload["message_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def verify_message_sha256(message: dict[str, Any]) -> tuple[str, str, bool]:
    """Re-derive the message's sha256 and compare to the stored value."""

    claimed = str(message.get("message_sha256", ""))
    verify_copy = {k: v for k, v in message.items() if k != "message_sha256"}
    recomputed = hashlib.sha256(canonical_json(verify_copy).encode("utf-8")).hexdigest()
    return claimed, recomputed, claimed == recomputed


def validate_to_session(to_session: str, *, steering_inbox_root: Path) -> Path:
    """Return the bounded inbox path for a safe session identifier."""

    if not to_session:
        raise ValueError("message.to_session must be a non-empty string")
    if to_session != to_session.strip():
        raise ValueError("message.to_session must not have leading or trailing whitespace")
    if "/" in to_session or "\\" in to_session:
        raise ValueError("message.to_session must not contain path separators")

    session_path = Path(to_session)
    if session_path.is_absolute() or to_session in {".", ".."} or ".." in session_path.parts:
        raise ValueError("message.to_session must be a plain session identifier")

    root = steering_inbox_root.resolve(strict=False)
    inbox = (root / to_session).resolve(strict=False)
    try:
        inbox.relative_to(root)
    except ValueError as exc:
        raise ValueError("message.to_session resolves outside the steering inbox root") from exc
    return inbox


def _load_lane_records(registry_path: Path) -> list[dict[str, Any]]:
    try:
        data = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return [record for record in data if isinstance(record, dict)] if isinstance(data, list) else []


def _load_default_lane_records(registry_path: Path) -> list[dict[str, Any]]:
    if registry_path != LANE_REGISTRY_DEFAULT:
        return _load_lane_records(registry_path)
    records: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for path in (USER_LANE_REGISTRY_DEFAULT, LANE_REGISTRY_DEFAULT):
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        if resolved in seen:
            continue
        seen.add(resolved)
        records.extend(_load_lane_records(path))
    return records


def _is_active_lane(record: dict[str, Any]) -> bool:
    return str(record.get("status") or "").strip().lower() in ACTIVE_STATUSES


def _identity_matches(
    record: dict[str, Any],
    *,
    pr: int | None = None,
    branch: str | None = None,
    worktree: str | None = None,
) -> bool:
    if pr is not None:
        try:
            if int(record.get("pr_number") or 0) == int(pr):
                return True
        except (TypeError, ValueError):
            pass
    if branch and record.get("branch") == branch:
        return True
    if worktree:
        record_worktree = record.get("worktree")
        if record_worktree and os.path.normpath(str(record_worktree)) == os.path.normpath(worktree):
            return True
    return False


def _updated_at_sort_key(record: dict[str, Any]) -> str:
    return str(record.get("updated_at") or "")


def route_payload_for_record(
    owner_record: dict[str, Any],
    *,
    resolved_via: str,
    steering_inbox_root: Path,
) -> dict[str, Any]:
    """Return target metadata for a resolved active-owner route without writing."""

    owner_session = str(owner_record.get("owner_session") or "")
    inbox_path = validate_to_session(owner_session, steering_inbox_root=steering_inbox_root)
    return {
        "resolved_via": resolved_via,
        "lane_id": owner_record.get("lane_id"),
        "owner_session": owner_session,
        "pr_number": owner_record.get("pr_number"),
        "branch": owner_record.get("branch"),
        "worktree": owner_record.get("worktree"),
        "status": owner_record.get("status"),
        "updated_at": owner_record.get("updated_at"),
        "steering_inbox_path": str(inbox_path),
        "dispatchable": True,
        "dispatch_blocker": None,
    }


def direct_route_payload(to_session: str, *, steering_inbox_root: Path) -> dict[str, Any]:
    """Return target metadata for direct ``--to`` dispatch without writing."""

    inbox_path = validate_to_session(to_session, steering_inbox_root=steering_inbox_root)
    return {
        "resolved_via": "direct",
        "lane_id": None,
        "owner_session": to_session,
        "pr_number": None,
        "branch": None,
        "worktree": None,
        "status": None,
        "updated_at": None,
        "steering_inbox_path": str(inbox_path),
        "dispatchable": True,
        "dispatch_blocker": None,
    }


def resolve_active_owner(
    *,
    registry_path: Path | None = None,
    pr: int | None = None,
    branch: str | None = None,
    worktree: str | None = None,
) -> dict[str, Any]:
    """Resolve a single active owner by PR/branch/worktree, failing closed.

    This is intentionally stricter than ``identify_lane_owner.py``: steering
    should only route to an active owner. Released/completed rows are useful
    history, but they are not a live dispatch target.
    """

    if pr is None and branch is None and worktree is None:
        raise ValueError("provide at least one owner selector")
    records = _load_default_lane_records(registry_path or LANE_REGISTRY_DEFAULT)
    matches = [
        record
        for record in records
        if _is_active_lane(record)
        and _identity_matches(record, pr=pr, branch=branch, worktree=worktree)
    ]
    if not matches:
        raise ValueError("no active owner matched the requested PR/branch/worktree")

    owners = {str(record.get("owner_session") or "") for record in matches}
    lanes = {str(record.get("lane_id") or "") for record in matches}
    if len(owners) != 1 or len(lanes) != 1:
        compact = [
            {
                "lane_id": record.get("lane_id"),
                "owner_session": record.get("owner_session"),
                "pr_number": record.get("pr_number"),
                "branch": record.get("branch"),
                "worktree": record.get("worktree"),
                "status": record.get("status"),
            }
            for record in sorted(matches, key=_updated_at_sort_key, reverse=True)
        ]
        raise ValueError(f"multiple active owners matched; resolve conflict first: {compact}")

    chosen = sorted(matches, key=_updated_at_sort_key, reverse=True)[0]
    owner = str(chosen.get("owner_session") or "")
    if not owner:
        raise ValueError("matched active lane has no owner_session")
    return chosen


def _mailbox_filename(sent_at_utc: str) -> str:
    return f"{_filename_timestamp(sent_at_utc)}-{secrets.token_hex(SHORT_UUID_BYTES)}.json"


def write_message(
    message: dict[str, Any],
    *,
    steering_inbox_root: Path = STEERING_INBOX_ROOT_DEFAULT,
) -> Path:
    """Atomically persist ``message`` into the recipient's mailbox.

    Creates the recipient dir on first message (idempotent). Writes
    to a tempfile in the same dir, then ``os.replace`` to the final
    path so partial writes never appear under the inbox glob.

    Returns the absolute path of the written message.
    """

    to_session = str(message.get("to_session") or "")
    inbox = validate_to_session(to_session, steering_inbox_root=steering_inbox_root)
    inbox.mkdir(parents=True, exist_ok=True)
    final_name = _mailbox_filename(str(message.get("sent_at_utc") or _now_utc_iso()))
    final_path = inbox / final_name
    body_text = json.dumps(message, indent=2, sort_keys=True)

    # Atomic write: tmp in same dir, then rename.
    fd, tmp_path = tempfile.mkstemp(
        prefix=".tmp-",
        suffix=".json",
        dir=str(inbox),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(body_text)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, final_path)
    except Exception:
        # Clean up tempfile on failure.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
    return final_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="send_operator_steering.py",
        description=(
            "Write an operator → agent-session steering message into the "
            "atomic mailbox under .aragora/operator-steering/<to_session>/."
        ),
    )
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--to",
        metavar="OWNER_SESSION",
        help="Target owner_session identifier (e.g., codex-p19-repair-7292).",
    )
    target_group.add_argument(
        "--to-owner-pr",
        type=int,
        metavar="N",
        help="Resolve the active owner_session for PR N from lanes.json before writing.",
    )
    target_group.add_argument(
        "--to-owner-branch",
        metavar="BRANCH",
        help="Resolve the active owner_session for BRANCH from lanes.json before writing.",
    )
    target_group.add_argument(
        "--to-owner-worktree",
        metavar="PATH",
        help="Resolve the active owner_session for worktree PATH from lanes.json before writing.",
    )
    parser.add_argument(
        "--lane-id",
        default=None,
        help="Optional lane_id hint (passed through to message.lane_id_hint).",
    )
    parser.add_argument(
        "--pr",
        type=int,
        default=None,
        metavar="N",
        help="Optional PR number hint (passed through to message.pr_hint).",
    )
    parser.add_argument(
        "--from",
        dest="from_label",
        default="operator",
        metavar="OPERATOR_NAME",
        help="Sender label (default: 'operator').",
    )
    parser.add_argument(
        "--priority",
        choices=PRIORITY_CHOICES,
        default="normal",
        help="Message priority (default: normal).",
    )
    body_group = parser.add_mutually_exclusive_group(required=False)
    body_group.add_argument(
        "--body",
        default=None,
        help="Inline message body (markdown).",
    )
    body_group.add_argument(
        "--body-file",
        type=Path,
        default=None,
        metavar="PATH",
        help="Read message body from a file (markdown).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the written record + final path as JSON to stdout.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve and validate the target, but do not write a steering message.",
    )
    parser.add_argument(
        "--print-target",
        action="store_true",
        help="Print the resolved steering target without requiring a message body or writing.",
    )
    parser.add_argument(
        "--steering-inbox-root",
        type=Path,
        default=STEERING_INBOX_ROOT_DEFAULT,
        help="Override the steering inbox root (used by tests).",
    )
    parser.add_argument(
        "--lane-registry-path",
        type=Path,
        default=LANE_REGISTRY_DEFAULT,
        help="Override path to lanes.json for --to-owner-* resolution (used by tests).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _build_parser().parse_args(argv)
    except SystemExit as exc:
        # argparse exits 2 on bad args; preserve that contract.
        return int(exc.code) if isinstance(exc.code, int) else 2

    no_write = bool(args.dry_run or args.print_target)

    # Validate / resolve body.
    body_text: str
    if args.body is not None:
        body_text = args.body
    elif args.body_file is not None:
        body_path: Path = args.body_file
        if not body_path.exists():
            print(f"ERROR: --body-file not found: {body_path}", file=sys.stderr)
            return 2
        try:
            body_text = body_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            print(f"ERROR: cannot read --body-file: {exc}", file=sys.stderr)
            return 2
    else:
        body_text = ""

    if not body_text.strip() and not args.print_target:
        print("ERROR: message body is empty", file=sys.stderr)
        return 2

    route: dict[str, Any] | None = None
    to_session = args.to
    if to_session is None:
        try:
            owner_record = resolve_active_owner(
                registry_path=args.lane_registry_path,
                pr=args.to_owner_pr,
                branch=args.to_owner_branch,
                worktree=args.to_owner_worktree,
            )
        except ValueError as exc:
            print(f"ERROR: failed to resolve active owner: {exc}", file=sys.stderr)
            return 2
        to_session = str(owner_record.get("owner_session") or "")
        resolved_via = (
            "pr"
            if args.to_owner_pr is not None
            else "branch"
            if args.to_owner_branch
            else "worktree"
        )
        route = route_payload_for_record(
            owner_record,
            resolved_via=resolved_via,
            steering_inbox_root=args.steering_inbox_root,
        )
    else:
        try:
            route = direct_route_payload(to_session, steering_inbox_root=args.steering_inbox_root)
        except ValueError as exc:
            print(f"ERROR: failed to validate target: {exc}", file=sys.stderr)
            return 2

    if args.print_target and not body_text.strip():
        out = {
            "_dry_run": True,
            "_would_write": False,
            "_target": route,
        }
        if args.json:
            print(json.dumps(out, indent=2, sort_keys=True))
        else:
            print(
                f"target {route['owner_session']} -> {route['steering_inbox_path']} "
                "(no message body supplied; no write)"
            )
        return 0

    message = build_message(
        to_session=to_session,
        body=body_text,
        from_label=args.from_label,
        lane_id_hint=args.lane_id,
        pr_hint=args.pr,
        priority=args.priority,
    )

    written_path: Path | None = None
    if not no_write:
        try:
            written_path = write_message(message, steering_inbox_root=args.steering_inbox_root)
        except (OSError, ValueError) as exc:
            print(f"ERROR: failed to write message: {exc}", file=sys.stderr)
            return 2

    if args.json:
        out = dict(message)
        out["_dry_run"] = no_write
        out["_would_write"] = no_write
        out["_written_path"] = str(written_path) if written_path is not None else None
        if route is not None:
            out["_route"] = route
        print(json.dumps(out, indent=2, sort_keys=True))
    elif no_write:
        print(
            f"would write to {route['steering_inbox_path']}  "
            f"(to={route['owner_session']}, priority={message['priority']})"
        )
    else:
        print(
            f"wrote {written_path}  "
            f"(sha256 {message['message_sha256'][:10]}…, "
            f"priority={message['priority']})"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
