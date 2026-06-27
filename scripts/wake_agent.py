#!/usr/bin/env python3
"""Resolve an agent owner and deliver or queue a steering prompt.

Default mode is dry-run.  ``--apply`` is required before the command sends to
tmux, resumes a Codex thread, or writes a mailbox message.  Every attempted
dispatch writes an auditable receipt unless ``--no-receipt`` is supplied.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
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
LANE_REGISTRY_DEFAULT = REPO_ROOT / ".aragora" / "agent-bridge" / "lanes.json"
STEERING_INBOX_ROOT_DEFAULT = REPO_ROOT / ".aragora" / "operator-steering"
DISPATCH_RECEIPT_ROOT_DEFAULT = REPO_ROOT / ".aragora" / "dispatch-receipts"
RECEIPT_SCHEMA_VERSION = "aragora-wake-agent-receipt/1.0"


class WakeError(Exception):
    """Raised when a prompt cannot be resolved or delivered safely."""

    def __init__(self, message: str, *, exit_code: int = 1) -> None:
        super().__init__(message)
        self.exit_code = exit_code


def _now_utc_iso() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _filename_timestamp(iso: str) -> str:
    return iso.replace(":", "-").replace(".", "-")


def _prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _read_prompt(*, prompt: str | None, prompt_file: Path | None) -> str:
    if bool(prompt) == bool(prompt_file):
        raise WakeError("provide exactly one of --prompt or --prompt-file", exit_code=2)
    if prompt_file is not None:
        if not prompt_file.is_file():
            raise WakeError(f"prompt file not found: {prompt_file}", exit_code=2)
        body = prompt_file.read_text(encoding="utf-8")
    else:
        body = prompt or ""
    if not body.strip():
        raise WakeError("prompt must not be empty", exit_code=2)
    return body


def _resolve_lane(
    *,
    to_session: str | None,
    lane_id: str | None,
    pr: int | None,
    branch: str | None,
    worktree: str | None,
    registry_path: Path,
    steering_inbox_root: Path,
) -> tuple[str, dict[str, Any] | None, str]:
    if to_session:
        steering_writer.validate_to_session(to_session, steering_inbox_root=steering_inbox_root)
        return to_session, None, "direct"

    records = owner_lookup.load_lane_records(registry_path)
    lane = owner_lookup.find_lane(
        records,
        lane_id=lane_id,
        pr=pr,
        branch=branch,
        worktree=worktree,
    )
    if lane is None:
        raise WakeError("no lane matched the requested selector", exit_code=2)
    owner_session = str(lane.get("owner_session") or "")
    if not owner_session:
        raise WakeError("matched lane has no owner_session", exit_code=2)
    steering_writer.validate_to_session(owner_session, steering_inbox_root=steering_inbox_root)
    if str(lane.get("status") or "").strip().lower() not in owner_lookup.ACTIVE_STATUSES:
        raise WakeError(
            f"matched lane status is {lane.get('status')!r}; refusing live dispatch",
            exit_code=3,
        )
    if lane_id:
        return owner_session, lane, "lane-id"
    if pr is not None:
        return owner_session, lane, "pr"
    if branch:
        return owner_session, lane, "branch"
    return owner_session, lane, "worktree"


def _contact_payload(lane: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(lane, dict):
        return {}
    payload = lane.get("contact_payload")
    return payload if isinstance(payload, dict) else {}


def choose_transport(
    *,
    lane: dict[str, Any] | None,
    fallback: str,
) -> tuple[str, dict[str, Any]]:
    """Return ``(transport, details)``.

    ``transport`` is one of ``tmux``, ``codex-exec-resume``, ``mailbox``, or
    ``blocked``.  Codex app-server is intentionally not enabled until a lane
    registers a reachable socket and the repository ships a tested protocol
    adapter; mailbox fallback is safer than blind Desktop tab injection.
    """

    method = str((lane or {}).get("contact_method") or "").strip()
    payload = _contact_payload(lane)
    thread_id = str((lane or {}).get("codex_thread_id") or "").strip()

    if method.startswith("tmux:"):
        target = method.removeprefix("tmux:").strip()
        if target:
            return "tmux", {"target": target, "contact_method": method}
        return "blocked", {"reason": "tmux contact_method has no target"}

    if method.startswith("codex-exec-resume:"):
        method_thread = method.removeprefix("codex-exec-resume:").strip()
        if method_thread:
            return "codex-exec-resume", {
                "thread_id": method_thread,
                "contact_method": method,
            }
        return "blocked", {"reason": "codex-exec-resume contact_method has no thread id"}

    if method.startswith("codex-app-server:"):
        if fallback == "fail":
            return "blocked", {
                "reason": "codex-app-server transport requires a shipped protocol adapter"
            }
        return "mailbox", {
            "fallback_reason": "codex-app-server transport not yet enabled",
            "socket": payload.get("socket"),
            "thread_id": payload.get("thread_id") or thread_id,
        }

    if method == "mailbox-only":
        return "mailbox", {"contact_method": method}

    if thread_id:
        return "codex-exec-resume", {"thread_id": thread_id, "source": "codex_thread_id"}

    if method and fallback == "fail":
        return "blocked", {"reason": f"unsupported contact_method {method!r}"}
    return "mailbox", {"fallback_reason": "no live contact method registered"}


def _send_tmux(target: str, prompt: str) -> None:
    if "\n" in prompt:
        subprocess.run(
            ["tmux", "load-buffer", "-"], input=prompt, text=True, check=True, timeout=10
        )
        subprocess.run(["tmux", "paste-buffer", "-d", "-t", target], check=True, timeout=10)
        subprocess.run(["tmux", "send-keys", "-t", target, "Enter"], check=True, timeout=10)
    else:
        subprocess.run(["tmux", "send-keys", "-t", target, prompt, "Enter"], check=True, timeout=10)


def _send_codex_exec_resume(thread_id: str, prompt: str, cwd: Path) -> None:
    subprocess.run(
        ["codex", "exec", "resume", thread_id, "-"],
        input=prompt,
        text=True,
        cwd=str(cwd),
        check=True,
        timeout=3600,
    )


def _write_mailbox(
    *,
    owner_session: str,
    prompt: str,
    lane: dict[str, Any] | None,
    priority: str,
    steering_inbox_root: Path,
) -> Path:
    raw_pr = (lane or {}).get("pr_number")
    try:
        pr_hint = int(raw_pr) if raw_pr is not None else None
    except (TypeError, ValueError):
        pr_hint = None
    message = steering_writer.build_message(
        to_session=owner_session,
        body=prompt,
        priority=priority,
        lane_id_hint=str((lane or {}).get("lane_id") or "") or None,
        pr_hint=pr_hint,
    )
    return steering_writer.write_message(message, steering_inbox_root=steering_inbox_root)


def _write_receipt(
    *,
    receipt_root: Path,
    payload: dict[str, Any],
) -> Path:
    ts = _filename_timestamp(str(payload.get("completed_at_utc") or _now_utc_iso()))
    lane_part = str(payload.get("lane_id") or payload.get("owner_session") or "direct")
    safe_lane = "".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in lane_part)[:80]
    path = receipt_root / f"{ts}-{safe_lane}-{str(payload['prompt_sha256'])[:8]}.json"
    _atomic_write_json(path, payload)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--to", help="Direct owner_session recipient.")
    target.add_argument("--lane", "--lane-id", dest="lane_id", help="Lane id selector.")
    target.add_argument("--pr", type=int, help="PR number selector.")
    target.add_argument("--branch", help="Branch selector.")
    target.add_argument("--worktree", help="Worktree selector.")
    body = parser.add_mutually_exclusive_group(required=True)
    body.add_argument("--prompt", help="Prompt body.")
    body.add_argument("--prompt-file", type=Path, help="Prompt file.")
    parser.add_argument("--priority", choices=steering_writer.PRIORITY_CHOICES, default="normal")
    parser.add_argument("--fallback", choices=("mailbox-only", "fail"), default="mailbox-only")
    parser.add_argument("--apply", action="store_true", help="Actually deliver instead of dry-run.")
    parser.add_argument("--no-receipt", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--registry-path", type=Path, default=LANE_REGISTRY_DEFAULT)
    parser.add_argument("--steering-inbox-root", type=Path, default=STEERING_INBOX_ROOT_DEFAULT)
    parser.add_argument("--receipt-root", type=Path, default=DISPATCH_RECEIPT_ROOT_DEFAULT)
    parser.add_argument("--cwd", type=Path, default=REPO_ROOT)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    prompt = _read_prompt(prompt=args.prompt, prompt_file=args.prompt_file)
    owner_session, lane, resolved_via = _resolve_lane(
        to_session=args.to,
        lane_id=args.lane_id,
        pr=args.pr,
        branch=args.branch,
        worktree=args.worktree,
        registry_path=args.registry_path,
        steering_inbox_root=args.steering_inbox_root,
    )
    transport, details = choose_transport(lane=lane, fallback=args.fallback)
    prompt_hash = _prompt_sha256(prompt)
    now = _now_utc_iso()
    result: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "ok": False,
        "dry_run": not args.apply,
        "owner_session": owner_session,
        "resolved_via": resolved_via,
        "lane_id": (lane or {}).get("lane_id"),
        "pr_number": (lane or {}).get("pr_number"),
        "branch": (lane or {}).get("branch"),
        "worktree": (lane or {}).get("worktree"),
        "transport": transport,
        "transport_details": details,
        "mailbox_dispatchable": transport in {"mailbox", "tmux", "codex-exec-resume"},
        "live_prompt_dispatchable": transport in {"tmux", "codex-exec-resume"},
        "prompt_sha256": prompt_hash,
        "created_at_utc": now,
        "completed_at_utc": now,
        "receipt_path": None,
    }

    try:
        if transport == "blocked":
            raise WakeError(str(details.get("reason") or "transport blocked"), exit_code=3)
        if not args.apply:
            result["ok"] = True
            result["status"] = "dry-run"
            return result
        if transport == "tmux":
            _send_tmux(str(details["target"]), prompt)
            result["status"] = "sent"
        elif transport == "codex-exec-resume":
            _send_codex_exec_resume(str(details["thread_id"]), prompt, args.cwd)
            result["status"] = "sent"
        elif transport == "mailbox":
            path = _write_mailbox(
                owner_session=owner_session,
                prompt=prompt,
                lane=lane,
                priority=args.priority,
                steering_inbox_root=args.steering_inbox_root,
            )
            result["status"] = "mailbox-written"
            result["mailbox_message_path"] = str(path)
        result["ok"] = True
        return result
    except (OSError, subprocess.SubprocessError, WakeError) as exc:
        result["ok"] = False
        result["status"] = "blocked"
        result["error"] = str(exc)
        return result
    finally:
        result["completed_at_utc"] = _now_utc_iso()
        if not args.no_receipt:
            receipt_path = _write_receipt(receipt_root=args.receipt_root, payload=result)
            result["receipt_path"] = str(receipt_path)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run(args)
    except WakeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return exc.exit_code
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(
            f"{result['status']}: owner={result['owner_session']} "
            f"transport={result['transport']} dry_run={result['dry_run']}"
        )
        if result.get("error"):
            print(f"error: {result['error']}", file=sys.stderr)
        if result.get("receipt_path"):
            print(f"receipt: {result['receipt_path']}")
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
