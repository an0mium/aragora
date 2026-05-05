#!/usr/bin/env python3
"""Audit Codex Desktop automation definitions for Aragora autonomy readiness."""

from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

CORE_WRITERS = {
    "engineering-autopilot": 5,
    "engineering-autopilot-2": 20,
    "engineering-autopilot-3": 35,
    "engineering-autopilot-3-2": 50,
}
PROMPT_WORDS_BY_ROLE = {
    "writer": ("memory", "outbox", "branch", "validate", "preflight"),
    "hygiene": ("local", "dirty", "worktree"),
    "steward": ("memory", "outbox", "receipts"),
}


@dataclass(frozen=True)
class AutomationRecord:
    id: str
    name: str
    kind: str
    status: str
    rrule: str
    prompt: str
    path: str
    byminute: int | None
    role: str


@dataclass(frozen=True)
class AuditIssue:
    automation_id: str
    severity: str
    code: str
    message: str


def _parse_byminute(rrule: str) -> int | None:
    match = re.search(r"(?:^|;)BYMINUTE=(\d{1,2})(?:;|$)", rrule.removeprefix("RRULE:"))
    if not match:
        return None
    minute = int(match.group(1))
    if 0 <= minute <= 59:
        return minute
    return None


def _role_for(record_id: str, name: str) -> str:
    normalized = f"{record_id} {name}".lower()
    if record_id in CORE_WRITERS or "writer" in normalized:
        return "writer"
    if "hygiene" in normalized or "cleanup" in normalized:
        return "hygiene"
    if "steward" in normalized or "scout" in normalized or "triage" in normalized:
        return "steward"
    return "other"


def _load_record(path: Path) -> AutomationRecord:
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    record_id = str(payload.get("id") or path.parent.name)
    name = str(payload.get("name") or record_id)
    rrule = str(payload.get("rrule") or "")
    role = _role_for(record_id, name)
    return AutomationRecord(
        id=record_id,
        name=name,
        kind=str(payload.get("kind") or ""),
        status=str(payload.get("status") or ""),
        rrule=rrule,
        prompt=str(payload.get("prompt") or ""),
        path=str(path),
        byminute=_parse_byminute(rrule),
        role=role,
    )


def _load_records_with_issues(root: Path) -> tuple[list[AutomationRecord], list[AuditIssue]]:
    records: list[AutomationRecord] = []
    issues: list[AuditIssue] = []
    for path in sorted(root.glob("*/automation.toml")):
        try:
            records.append(_load_record(path))
        except (OSError, tomllib.TOMLDecodeError, UnicodeError) as exc:
            issues.append(
                AuditIssue(
                    path.parent.name,
                    "error",
                    "invalid_automation_toml",
                    f"failed to read {path}: {exc}",
                )
            )
    return records, issues


def load_automations(root: Path) -> list[AutomationRecord]:
    records, _issues = _load_records_with_issues(root)
    return records


def _automation_file_count(root: Path) -> int:
    return sum(1 for _path in root.glob("*/automation.toml"))


def audit(records: list[AutomationRecord]) -> list[AuditIssue]:
    issues: list[AuditIssue] = []
    by_id = {record.id: record for record in records}

    for writer_id, expected_minute in CORE_WRITERS.items():
        record = by_id.get(writer_id)
        if record is None:
            issues.append(
                AuditIssue(writer_id, "error", "missing_core_writer", "core writer is absent")
            )
            continue
        if record.status != "ACTIVE":
            issues.append(
                AuditIssue(writer_id, "error", "core_writer_inactive", "core writer is not active")
            )
        if record.kind != "cron":
            issues.append(
                AuditIssue(writer_id, "error", "core_writer_not_cron", "core writer is not cron")
            )
        if record.byminute != expected_minute:
            issues.append(
                AuditIssue(
                    writer_id,
                    "warning",
                    "writer_not_staggered",
                    f"expected BYMINUTE={expected_minute}, found {record.byminute}",
                )
            )

    active_writer_minutes: dict[int, list[str]] = {}
    for record in records:
        prompt_lower = record.prompt.lower()
        if record.status == "ACTIVE" and "paused" in prompt_lower:
            issues.append(
                AuditIssue(
                    record.id,
                    "warning",
                    "active_paused_prompt",
                    "automation is active but prompt says paused",
                )
            )
        if record.role == "writer" and record.status == "ACTIVE" and record.byminute is not None:
            active_writer_minutes.setdefault(record.byminute, []).append(record.id)
        required_words = PROMPT_WORDS_BY_ROLE.get(record.role, ())
        for word in required_words:
            if record.status == "ACTIVE" and word not in prompt_lower:
                issues.append(
                    AuditIssue(
                        record.id,
                        "warning",
                        f"missing_prompt_word_{word}",
                        f"active {record.role} prompt does not mention {word}",
                    )
                )

    for minute, ids in sorted(active_writer_minutes.items()):
        if len(ids) > 1:
            issues.append(
                AuditIssue(
                    ",".join(ids),
                    "warning",
                    "duplicate_writer_minute",
                    f"multiple active writers scheduled at BYMINUTE={minute}",
                )
            )

    return issues


def build_payload(root: Path) -> dict[str, Any]:
    records, load_issues = _load_records_with_issues(root)
    issues = [*load_issues, *audit(records)]
    return {
        "root": str(root),
        "automation_count": _automation_file_count(root),
        "core_writers": {
            writer_id: next((asdict(r) for r in records if r.id == writer_id), None)
            for writer_id in CORE_WRITERS
        },
        "issues": [asdict(issue) for issue in issues],
        "summary": {
            "error_count": sum(1 for issue in issues if issue.severity == "error"),
            "warning_count": sum(1 for issue in issues if issue.severity == "warning"),
            "active_count": sum(1 for record in records if record.status == "ACTIVE"),
        },
    }


def summary_only_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a compact payload suitable for recurring automation gates."""

    compact = dict(payload)
    compact["core_writers"] = {
        writer_id: (
            {
                key: record[key]
                for key in ("id", "name", "kind", "status", "path", "byminute", "role")
                if key in record
            }
            if isinstance(record, dict)
            else None
        )
        for writer_id, record in payload.get("core_writers", {}).items()
    }
    compact["prompt_details_omitted"] = True
    return compact


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.home() / ".codex" / "automations",
        help="Codex Desktop automation directory",
    )
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Omit full automation prompts from JSON output for compact startup gates.",
    )
    args = parser.parse_args(argv)

    payload = build_payload(args.root.expanduser())
    if args.summary_only:
        payload = summary_only_payload(payload)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        summary = payload["summary"]
        print(
            f"{payload['automation_count']} automations; "
            f"{summary['error_count']} error(s), {summary['warning_count']} warning(s)"
        )
        for issue in payload["issues"]:
            print(
                f"{issue['severity'].upper():<7} {issue['automation_id']:<30} "
                f"{issue['code']}: {issue['message']}"
            )
    return 1 if payload["summary"]["error_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
