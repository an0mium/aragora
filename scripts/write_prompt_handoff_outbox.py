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
MAX_INLINE_PROMPT_CHARS = 40_000
PUBLISHER_MAX_ISSUE_BODY_BYTES = 60_000
INLINE_PROMPT_BODY_SAFETY_BYTES = 2_000
MAX_INLINE_PROMPT_BODY_BYTES = PUBLISHER_MAX_ISSUE_BODY_BYTES - INLINE_PROMPT_BODY_SAFETY_BYTES
PROMPT_ARTIFACT_DIR = "_prompt-artifacts"
PROMPT_PREVIEW_CHARS = 2_000


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


def _prompt_sha(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _default_idempotency_key(
    *, task: str, prompt_sha: str, repo: str, target: dict[str, Any]
) -> str:
    target_material = json.dumps(
        {"repo": repo, "target": target},
        sort_keys=True,
        separators=(",", ":"),
    )
    target_sha = hashlib.sha256(target_material.encode("utf-8")).hexdigest()
    return f"prompt-handoff-{_slug(task)}-{prompt_sha[:12]}-{target_sha[:12]}"


def _format_json_block(value: Any) -> str:
    if value is None or value == "":
        return "NONE"
    if isinstance(value, str):
        return value.strip() or "NONE"
    return json.dumps(value, indent=2, sort_keys=True)


def _format_outbox_body(payload: dict[str, Any], source_file: Path) -> str:
    fields = [
        ("Task", payload.get("task")),
        ("Requested Action", payload.get("requested_action")),
        ("Requires GitHub", payload.get("requires_github")),
        ("Repo", payload.get("repo")),
        ("Created At", payload.get("created_at")),
        ("Idempotency Key", payload.get("idempotency_key")),
        ("Local Evidence", payload.get("local_evidence")),
        ("Validation", payload.get("validation")),
    ]
    lines: list[str] = []
    for label, value in fields:
        formatted = _format_json_block(value)
        lines.append(f"{label}:")
        if "\n" in formatted or formatted.startswith("{") or formatted.startswith("["):
            lines.append("```json" if formatted.startswith(("{", "[")) else "```")
            lines.append(formatted)
            lines.append("```")
        else:
            lines.append(formatted)
        lines.append("")
    lines.append("---")
    lines.append(f"Published from automation outbox: `{source_file}`")
    return "\n".join(lines).strip()


def _body_utf8_bytes(value: str) -> int:
    return len(value.encode("utf-8"))


def _inline_prompt_body_bytes(payload: dict[str, Any], source_file: Path) -> int:
    return _body_utf8_bytes(_format_outbox_body(payload, source_file))


def _needs_prompt_artifact(
    prompt: str,
    *,
    inline_payload: dict[str, Any] | None = None,
    source_file: Path | None = None,
) -> bool:
    if len(prompt) > MAX_INLINE_PROMPT_CHARS:
        return True
    if inline_payload is not None and source_file is not None:
        return _inline_prompt_body_bytes(inline_payload, source_file) > MAX_INLINE_PROMPT_BODY_BYTES
    return False


def _prompt_artifact_path(outbox_dir: Path, idempotency_key: str, prompt_sha: str) -> Path:
    filename = f"{_slug(idempotency_key, limit=100)}-{prompt_sha[:12]}.prompt.md"
    return outbox_dir / PROMPT_ARTIFACT_DIR / filename


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
    prompt_artifact_path: str | None = None,
) -> dict[str, Any]:
    """Return a publisher-compatible automation-outbox payload."""

    prompt_sha = _prompt_sha(prompt)
    key = idempotency_key or _default_idempotency_key(
        task=task,
        prompt_sha=prompt_sha,
        repo=repo,
        target=target,
    )
    local_evidence: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "prompt_handoff",
        "source": source,
        "prompt_sha256": prompt_sha,
        "prompt_chars": len(prompt),
    }
    requested_action: dict[str, Any] = {
        "type": "prompt_handoff",
        "prompt_sha256": prompt_sha,
    }
    if prompt_artifact_path:
        prompt_preview = prompt[:PROMPT_PREVIEW_CHARS]
        prompt_overflow = {
            "prompt_truncated": True,
            "prompt_artifact_path": prompt_artifact_path,
            "prompt_artifact_sha256": prompt_sha,
            "prompt_artifact_publication": "github_issue_comments",
            "prompt_preview": prompt_preview,
            "prompt_preview_chars": len(prompt_preview),
            "prompt_omitted_chars": max(len(prompt) - len(prompt_preview), 0),
        }
        local_evidence.update(prompt_overflow)
        requested_action.update(
            {
                "prompt_truncated": True,
                "prompt_artifact_path": prompt_artifact_path,
                "prompt_artifact_sha256": prompt_sha,
                "prompt_artifact_publication": "github_issue_comments",
            }
        )
    else:
        local_evidence["prompt"] = prompt

    branch = str(target.get("branch") or "").strip()
    expected_head = str(target.get("expected_head") or "").strip()
    target_payload = dict(target)
    if branch:
        local_evidence["branch"] = branch
        requested_action["branch"] = branch
    if expected_head:
        local_evidence["desired_head_sha"] = expected_head
        local_evidence["head_sha"] = expected_head
        requested_action["desired_head_sha"] = expected_head
        requested_action["head_sha"] = expected_head
        target_payload.setdefault("desired_head_sha", expected_head)
        target_payload.setdefault("head_sha", expected_head)
    if target_payload:
        local_evidence["target"] = target_payload

    if target_payload:
        requested_action["target"] = target_payload

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
    task = args.task or _default_task(prompt)
    repo = str(args.repo)
    target = _target_fields(args)
    prompt_sha = _prompt_sha(prompt)
    key = args.idempotency_key or _default_idempotency_key(
        task=task,
        prompt_sha=prompt_sha,
        repo=repo,
        target=target,
    )
    outbox_dir = args.outbox_dir.expanduser()
    outbox_path = _output_path(outbox_dir, key)
    inline_payload = build_payload(
        prompt=prompt,
        task=task,
        repo=repo,
        source=str(args.source),
        priority=str(args.priority),
        created_at=created_at,
        expires_hours=args.expires_hours,
        validation=validation,
        target=target,
        idempotency_key=key,
        prompt_artifact_path=None,
    )
    prompt_artifact_path = (
        _prompt_artifact_path(outbox_dir, key, prompt_sha)
        if _needs_prompt_artifact(prompt, inline_payload=inline_payload, source_file=outbox_path)
        else None
    )
    payload = (
        build_payload(
            prompt=prompt,
            task=task,
            repo=repo,
            source=str(args.source),
            priority=str(args.priority),
            created_at=created_at,
            expires_hours=args.expires_hours,
            validation=validation,
            target=target,
            idempotency_key=key,
            prompt_artifact_path=str(prompt_artifact_path),
        )
        if prompt_artifact_path is not None
        else inline_payload
    )

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "apply": bool(args.apply),
        "wrote": False,
        "outbox_path": str(outbox_path),
        "payload": payload,
    }
    if prompt_artifact_path is not None:
        result["prompt_artifact_path"] = str(prompt_artifact_path)

    if args.apply:
        if outbox_path.exists() and not args.force:
            print(f"error: handoff already exists: {outbox_path}", file=sys.stderr)
            return 2
        if prompt_artifact_path is not None and prompt_artifact_path.exists() and not args.force:
            print(f"error: prompt artifact already exists: {prompt_artifact_path}", file=sys.stderr)
            return 2
        outbox_dir.mkdir(parents=True, exist_ok=True)
        if prompt_artifact_path is not None:
            prompt_artifact_path.parent.mkdir(parents=True, exist_ok=True)
            prompt_artifact_path.write_text(prompt, encoding="utf-8")
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
