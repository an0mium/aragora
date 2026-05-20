#!/usr/bin/env python3
"""Local Codex/agent steering dispatch queue.

Jobs are JSON files under ``.aragora/codex-dispatch-queue/pending``.  The
runner resolves the target with ``wake_agent.py`` and moves each job to
``delivered`` or ``blocked`` with an auditable receipt.  Dry-run processing is
the default and never sends prompts or writes mailbox messages.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
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

import wake_agent

REPO_ROOT = Path(__file__).resolve().parents[1]
QUEUE_ROOT_DEFAULT = REPO_ROOT / ".aragora" / "codex-dispatch-queue"
JOB_SCHEMA_VERSION = "aragora-codex-dispatch-job/1.0"
RECEIPT_SCHEMA_VERSION = "aragora-codex-dispatch-receipt/1.0"


class QueueError(Exception):
    """Raised when a queue operation cannot proceed safely."""

    def __init__(self, message: str, *, exit_code: int = 1) -> None:
        super().__init__(message)
        self.exit_code = exit_code


def _now_utc_iso() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _filename_timestamp(value: str) -> str:
    return value.replace(":", "-").replace(".", "-")


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


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


def _queue_dirs(root: Path) -> dict[str, Path]:
    return {name: root / name for name in ("pending", "running", "delivered", "blocked")}


def _read_prompt(*, prompt: str | None, prompt_file: Path | None) -> str:
    if bool(prompt) == bool(prompt_file):
        raise QueueError("provide exactly one of --prompt or --prompt-file", exit_code=2)
    if prompt_file is not None:
        if not prompt_file.is_file():
            raise QueueError(f"prompt file not found: {prompt_file}", exit_code=2)
        body = prompt_file.read_text(encoding="utf-8")
    else:
        body = prompt or ""
    if not body.strip():
        raise QueueError("prompt must not be empty", exit_code=2)
    return body


def enqueue(args: argparse.Namespace) -> dict[str, Any]:
    prompt = _read_prompt(prompt=args.prompt, prompt_file=args.prompt_file)
    selectors = {
        "to": args.to,
        "lane_id": args.lane_id,
        "pr": args.pr,
        "branch": args.branch,
        "worktree": args.worktree,
    }
    if sum(value is not None and value != "" for value in selectors.values()) != 1:
        raise QueueError("provide exactly one target selector", exit_code=2)
    created_at = _now_utc_iso()
    job_id = f"{_filename_timestamp(created_at)}-{secrets.token_hex(4)}"
    job = {
        "schema_version": JOB_SCHEMA_VERSION,
        "id": job_id,
        "created_at_utc": created_at,
        "priority": args.priority,
        "selectors": selectors,
        "expected_head": args.expected_head,
        "prompt": prompt,
        "prompt_sha256": _sha256(prompt),
        "status": "pending",
    }
    pending = _queue_dirs(args.queue_root)["pending"]
    path = pending / f"{job_id}.json"
    _atomic_write_json(path, job)
    return {"ok": True, "job": job, "path": str(path)}


def _load_job(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QueueError(f"cannot read job {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise QueueError(f"job {path} is not a JSON object")
    if payload.get("schema_version") != JOB_SCHEMA_VERSION:
        raise QueueError(f"job {path} has unsupported schema_version")
    return payload


def _wake_argv_for_job(
    job: dict[str, Any],
    *,
    apply: bool,
    registry_path: Path,
    steering_inbox_root: Path,
    receipt_root: Path,
) -> list[str]:
    raw_selectors = job.get("selectors")
    selectors: dict[str, Any] = raw_selectors if isinstance(raw_selectors, dict) else {}
    argv: list[str] = []
    if selectors.get("to"):
        argv.extend(["--to", str(selectors["to"])])
    elif selectors.get("lane_id"):
        argv.extend(["--lane-id", str(selectors["lane_id"])])
    elif selectors.get("pr") is not None:
        argv.extend(["--pr", str(selectors["pr"])])
    elif selectors.get("branch"):
        argv.extend(["--branch", str(selectors["branch"])])
    elif selectors.get("worktree"):
        argv.extend(["--worktree", str(selectors["worktree"])])
    else:
        raise QueueError("queued job has no usable selector")
    argv.extend(["--prompt", str(job.get("prompt") or "")])
    argv.extend(["--priority", str(job.get("priority") or "normal")])
    argv.extend(["--registry-path", str(registry_path)])
    argv.extend(["--steering-inbox-root", str(steering_inbox_root)])
    argv.extend(["--receipt-root", str(receipt_root)])
    argv.extend(["--json"])
    if apply:
        argv.append("--apply")
    return argv


def _write_run_receipt(
    *,
    job: dict[str, Any],
    queue_root: Path,
    wake_result: dict[str, Any] | None,
    status: str,
    error: str | None = None,
) -> Path:
    completed_at = _now_utc_iso()
    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "job_id": job.get("id"),
        "status": status,
        "completed_at_utc": completed_at,
        "prompt_sha256": job.get("prompt_sha256"),
        "wake_result": wake_result,
        "error": error,
    }
    path = queue_root / status / f"{job.get('id', 'unknown')}.receipt.json"
    _atomic_write_json(path, receipt)
    return path


def run_queue(args: argparse.Namespace) -> dict[str, Any]:
    dirs = _queue_dirs(args.queue_root)
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    pending = sorted(dirs["pending"].glob("*.json"))
    processed: list[dict[str, Any]] = []
    for pending_path in pending[: args.limit]:
        job = _load_job(pending_path)
        running_path = dirs["running"] / pending_path.name
        os.replace(pending_path, running_path)
        wake_result: dict[str, Any] | None = None
        status = "blocked"
        error = None
        try:
            wake_args = wake_agent.build_parser().parse_args(
                _wake_argv_for_job(
                    job,
                    apply=args.apply,
                    registry_path=args.registry_path,
                    steering_inbox_root=args.steering_inbox_root,
                    receipt_root=args.receipt_root,
                )
            )
            result = wake_agent.run(wake_args)
            wake_result = result
            if args.apply and result.get("ok"):
                status = "delivered"
            else:
                status = "blocked"
                if not args.apply:
                    error = "dry-run; no prompt delivered"
                elif not result.get("ok"):
                    error = str(result.get("error") or "wake_agent blocked")
        except Exception as exc:  # noqa: BLE001 - queue receipts should capture all blockers.
            error = str(exc)
        receipt_path = _write_run_receipt(
            job=job,
            queue_root=args.queue_root,
            wake_result=wake_result,
            status=status,
            error=error,
        )
        final_job = dirs[status] / pending_path.name
        os.replace(running_path, final_job)
        processed.append(
            {
                "job_id": job.get("id"),
                "status": status,
                "job_path": str(final_job),
                "receipt_path": str(receipt_path),
                "wake_result": wake_result,
                "error": error,
            }
        )
    return {"ok": True, "dry_run": not args.apply, "processed": processed}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    enqueue_p = sub.add_parser("enqueue")
    target = enqueue_p.add_mutually_exclusive_group(required=True)
    target.add_argument("--to")
    target.add_argument("--lane", "--lane-id", dest="lane_id")
    target.add_argument("--pr", type=int)
    target.add_argument("--branch")
    target.add_argument("--worktree")
    body = enqueue_p.add_mutually_exclusive_group(required=True)
    body.add_argument("--prompt")
    body.add_argument("--prompt-file", type=Path)
    enqueue_p.add_argument(
        "--priority", choices=("low", "normal", "high", "blocking"), default="normal"
    )
    enqueue_p.add_argument("--expected-head", default=None)
    enqueue_p.add_argument("--queue-root", type=Path, default=QUEUE_ROOT_DEFAULT)
    enqueue_p.add_argument("--json", action="store_true")

    run_p = sub.add_parser("run")
    run_p.add_argument("--queue-root", type=Path, default=QUEUE_ROOT_DEFAULT)
    run_p.add_argument("--registry-path", type=Path, default=wake_agent.LANE_REGISTRY_DEFAULT)
    run_p.add_argument(
        "--steering-inbox-root", type=Path, default=wake_agent.STEERING_INBOX_ROOT_DEFAULT
    )
    run_p.add_argument(
        "--receipt-root", type=Path, default=wake_agent.DISPATCH_RECEIPT_ROOT_DEFAULT
    )
    run_p.add_argument("--limit", type=int, default=1)
    run_p.add_argument("--apply", action="store_true", help="Deliver queued prompts.")
    run_p.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = enqueue(args) if args.command == "enqueue" else run_queue(args)
    except QueueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return exc.exit_code
    if getattr(args, "json", False):
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        if args.command == "enqueue":
            print(f"queued {result['job']['id']} at {result['path']}")
        else:
            print(f"processed {len(result['processed'])} job(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
