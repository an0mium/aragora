#!/usr/bin/env python3
"""Write a generated next prompt as an automation-outbox handoff.

This is transport glue only: it does not call GitHub and it does not decide
whether a prompt is safe to execute. It converts a prompt that was already
built by repo-native tooling into the JSON contract consumed by
``scripts/publish_automation_handoffs.py`` so the existing publisher can move
prompt handoffs to repo-visible GitHub issues.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

DEFAULT_OUTBOX_DIR = Path(".aragora") / "automation-outbox"
DEFAULT_REPO = "synaptent/aragora"
SCHEMA_VERSION = "aragora-prompt-handoff/1.0"


def _utc_now() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def _iso_utc(value: datetime) -> str:
    return value.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_utc(value: str) -> datetime:
    text = value.strip()
    if not text:
        raise ValueError("timestamp must not be empty")
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC).replace(microsecond=0)


def _slug(text: str, *, limit: int = 80) -> str:
    lowered = text.strip().lower()
    slug = re.sub(r"[^a-z0-9_.-]+", "-", lowered).strip("-._")
    slug = re.sub(r"-{2,}", "-", slug)
    if not slug:
        return "prompt-handoff"
    return slug[:limit].strip("-._") or "prompt-handoff"


def _read_prompt(args: argparse.Namespace) -> str:
    if args.prompt is not None:
        return str(args.prompt)
    raw_path = str(args.prompt_file)
    if raw_path == "-":
        return sys.stdin.read()
    return Path(raw_path).expanduser().read_text(encoding="utf-8")


def _target_fields(args: argparse.Namespace) -> dict[str, Any]:
    target: dict[str, Any] = {}
    if args.pr is not None:
        target["pr"] = int(args.pr)
    if args.branch:
        target["branch"] = str(args.branch)
    if args.lane_id:
        target["lane_id"] = str(args.lane_id)
    if args.expected_head:
        target["expected_head"] = str(args.expected_head)
    return target


def build_payload(
    *,
    prompt: str,
    task: str,
    repo: str,
    source: str,
    priority: str,
    created_at: datetime,
    expires_hours: float | None,
    validation: list[str],
    target: dict[str, Any],
    idempotency_key: str | None = None,
) -> dict[str, Any]:
    """Return a publisher-compatible automation-outbox payload."""

    prompt_sha = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    key = idempotency_key or f"prompt-handoff-{_slug(task)}-{prompt_sha[:12]}"
    local_evidence: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "prompt_handoff",
        "source": source,
        "prompt_sha256": prompt_sha,
        "prompt_chars": len(prompt),
        "prompt": prompt,
    }
    if target:
        local_evidence["target"] = target

    requested_action: dict[str, Any] = {
        "type": "prompt_handoff",
        "prompt_sha256": prompt_sha,
    }
    if target:
        requested_action["target"] = target

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "task": task,
        "requires_github": True,
        "requested_action": requested_action,
        "repo": repo,
        "priority": priority,
        "created_at": _iso_utc(created_at),
        "idempotency_key": key,
        "local_evidence": local_evidence,
        "validation": validation,
    }
    if expires_hours is not None and expires_hours > 0:
        payload["expires_at"] = _iso_utc(created_at + timedelta(hours=expires_hours))
    return payload


def _output_path(outbox_dir: Path, idempotency_key: str) -> Path:
    return outbox_dir / f"{_slug(idempotency_key, limit=140)}.json"


def _default_task(prompt: str) -> str:
    first_line = next((line.strip() for line in prompt.splitlines() if line.strip()), "")
    if not first_line:
        return "Prompt handoff"
    return f"Prompt handoff: {first_line[:96]}"


def _default_validation(source: str) -> list[str]:
    return [
        f"Prompt source: {source}",
        "python3 scripts/publish_automation_handoffs.py --summary-only --json",
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt", help="Prompt text to store in the handoff")
    prompt_group.add_argument(
        "--prompt-file",
        help="Path to prompt text, or '-' to read from stdin",
    )
    parser.add_argument("--task", help="GitHub issue task title for the handoff")
    parser.add_argument("--repo", default=DEFAULT_REPO, help="Target GitHub OWNER/REPO")
    parser.add_argument(
        "--source",
        default="manual-prompt",
        help="Human-readable source command or artifact path for the prompt",
    )
    parser.add_argument("--priority", default="automation-quality")
    parser.add_argument("--pr", type=int, help="Optional PR target carried as metadata")
    parser.add_argument("--branch", help="Optional branch target carried as metadata")
    parser.add_argument("--lane-id", help="Optional lane target carried as metadata")
    parser.add_argument("--expected-head", help="Optional exact head carried as metadata")
    parser.add_argument(
        "--validation",
        action="append",
        default=[],
        help="Validation or publication command to include; repeatable",
    )
    parser.add_argument(
        "--expires-hours",
        type=float,
        default=72.0,
        help="Expiry horizon for stale prompts; <=0 omits expires_at",
    )
    parser.add_argument(
        "--outbox-dir",
        type=Path,
        default=DEFAULT_OUTBOX_DIR,
        help="Automation outbox directory to write under with --apply",
    )
    parser.add_argument("--idempotency-key", help="Override deterministic handoff key")
    parser.add_argument("--created-at", help=argparse.SUPPRESS)
    parser.add_argument("--apply", action="store_true", help="Write the JSON handoff")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing handoff file")
    parser.add_argument("--json", action="store_true", help="Emit JSON result")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        prompt = _read_prompt(args).strip()
    except OSError as exc:
        print(f"error: failed to read prompt: {exc}", file=sys.stderr)
        return 2
    if not prompt:
        print("error: prompt must not be empty", file=sys.stderr)
        return 2

    try:
        created_at = _parse_utc(args.created_at) if args.created_at else _utc_now()
    except ValueError as exc:
        print(f"error: invalid --created-at: {exc}", file=sys.stderr)
        return 2

    validation = list(args.validation) or _default_validation(str(args.source))
    payload = build_payload(
        prompt=prompt,
        task=args.task or _default_task(prompt),
        repo=str(args.repo),
        source=str(args.source),
        priority=str(args.priority),
        created_at=created_at,
        expires_hours=args.expires_hours,
        validation=validation,
        target=_target_fields(args),
        idempotency_key=args.idempotency_key,
    )

    outbox_dir = args.outbox_dir.expanduser()
    outbox_path = _output_path(outbox_dir, str(payload["idempotency_key"]))
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "apply": bool(args.apply),
        "wrote": False,
        "outbox_path": str(outbox_path),
        "payload": payload,
    }

    if args.apply:
        if outbox_path.exists() and not args.force:
            print(f"error: handoff already exists: {outbox_path}", file=sys.stderr)
            return 2
        outbox_dir.mkdir(parents=True, exist_ok=True)
        outbox_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        result["wrote"] = True

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        action = "wrote" if result["wrote"] else "would write"
        print(f"{action}: {outbox_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
