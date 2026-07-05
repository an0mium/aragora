#!/usr/bin/env python3
"""Render pending founder/operator decision packets.

This is a read-only transport-compression helper. It scans local founder
decision packet markdown files, plus optional exported issue comments, and
prints one table of pending rulings with their expected one-word replies.
It never posts, mutates PRs, reruns checks, settles, or merges.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

UTC = timezone.utc
DEFAULT_DECISIONS_ROOT = Path(".aragora/founder-decisions")


@dataclass(frozen=True)
class DecisionItem:
    item: str
    target: str
    requested_action: str
    expected_reply: str
    source: str
    packet_generated_at: datetime | None = None

    def dedupe_key(self) -> tuple[str, str, str]:
        return (
            _compact_text(self.target).lower(),
            _compact_text(self.requested_action).lower(),
            _compact_text(self.expected_reply).lower(),
        )


def _compact_text(value: str) -> str:
    return " ".join(value.strip().split())


def _strip_markdown_code(value: str) -> str:
    stripped = value.strip()
    if len(stripped) >= 2 and stripped.startswith("`") and stripped.endswith("`"):
        return stripped[1:-1].strip()
    return stripped


def _parse_datetime(value: str) -> datetime | None:
    raw = value.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _packet_generated_at(markdown: str) -> datetime | None:
    match = re.search(r"^Generated:\s*(.+?)\s*$", markdown, flags=re.MULTILINE)
    if not match:
        return None
    return _parse_datetime(match.group(1))


def _split_table_row(line: str) -> list[str] | None:
    stripped = line.strip()
    if not stripped.startswith("|") or not stripped.endswith("|"):
        return None
    cells = [cell.strip() for cell in stripped.strip("|").split("|")]
    if not cells:
        return None
    return cells


def _is_separator_row(cells: list[str]) -> bool:
    return all(cell and set(cell) <= {"-", ":", " "} for cell in cells)


def _pending_ruling_rows(markdown: str) -> list[list[str]]:
    lines = markdown.splitlines()
    in_section = False
    rows: list[list[str]] = []
    for line in lines:
        if line.startswith("## "):
            heading = line.lstrip("#").strip().lower()
            if heading == "pending rulings":
                in_section = True
                continue
            if in_section:
                break
        if not in_section:
            continue
        cells = _split_table_row(line)
        if cells is None:
            continue
        rows.append(cells)
    return rows


def parse_decision_packet(markdown: str, *, source: str) -> list[DecisionItem]:
    """Parse a founder decision packet markdown body."""

    rows = _pending_ruling_rows(markdown)
    if not rows:
        return []
    header: list[str] | None = None
    data_rows: list[list[str]] = []
    for row in rows:
        if _is_separator_row(row):
            continue
        if header is None:
            header = [cell.lower().strip() for cell in row]
            continue
        data_rows.append(row)
    if header is None:
        return []

    def index(name: str) -> int | None:
        try:
            return header.index(name)
        except ValueError:
            return None

    priority_i = index("priority")
    link_i = index("link")
    action_i = index("requested action")
    reply_i = index("one-word reply")
    if link_i is None or action_i is None or reply_i is None:
        return []

    generated_at = _packet_generated_at(markdown)
    items: list[DecisionItem] = []
    for row in data_rows:
        if max(link_i, action_i, reply_i) >= len(row):
            continue
        priority = row[priority_i] if priority_i is not None and priority_i < len(row) else ""
        target = _compact_text(row[link_i])
        requested_action = _compact_text(row[action_i])
        expected_reply = _strip_markdown_code(_compact_text(row[reply_i]))
        if not target or not expected_reply:
            continue
        item = f"Priority {priority}" if priority else target
        items.append(
            DecisionItem(
                item=item,
                target=target,
                requested_action=requested_action,
                expected_reply=expected_reply,
                source=source,
                packet_generated_at=generated_at,
            )
        )
    return items


def _load_issue_comment_bodies(path: Path) -> list[tuple[str, str]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(payload, list):
        return []
    bodies: list[tuple[str, str]] = []
    for index, comment in enumerate(payload):
        if not isinstance(comment, dict):
            continue
        body = comment.get("body")
        if not isinstance(body, str) or "## Pending Rulings" not in body:
            continue
        source = str(comment.get("html_url") or comment.get("url") or f"{path}#{index}")
        bodies.append((source, body))
    return bodies


def collect_decision_items(
    *,
    decisions_root: Path = DEFAULT_DECISIONS_ROOT,
    packet_files: Iterable[Path] = (),
    issue_comments_json: Path | None = None,
) -> list[DecisionItem]:
    sources: list[tuple[str, str]] = []
    if decisions_root.exists():
        for path in sorted(decisions_root.glob("*.md")):
            try:
                sources.append((str(path), path.read_text(encoding="utf-8")))
            except OSError:
                continue
    for path in packet_files:
        try:
            sources.append((str(path), path.read_text(encoding="utf-8")))
        except OSError:
            continue
    if issue_comments_json is not None:
        sources.extend(_load_issue_comment_bodies(issue_comments_json))

    deduped: dict[tuple[str, str, str], DecisionItem] = {}
    for source, body in sources:
        for item in parse_decision_packet(body, source=source):
            deduped.setdefault(item.dedupe_key(), item)
    return list(deduped.values())


def _format_age(item: DecisionItem, *, now: datetime) -> str:
    if item.packet_generated_at is None:
        return "unknown"
    seconds = max(0.0, (now - item.packet_generated_at).total_seconds())
    hours = seconds / 3600.0
    if hours < 1:
        return f"{int(seconds // 60)}m"
    if hours < 48:
        return f"{hours:.1f}h"
    return f"{hours / 24.0:.1f}d"


def render_markdown(items: Iterable[DecisionItem], *, now: datetime | None = None) -> str:
    now_dt = (now or datetime.now(tz=UTC)).astimezone(UTC)
    rows = list(items)
    lines = [
        "# Founder Decision Queue",
        "",
        f"Generated: {now_dt.isoformat().replace('+00:00', 'Z')}",
        "",
    ]
    if not rows:
        lines.append("No pending operator rulings found.")
        return "\n".join(lines) + "\n"

    lines.extend(
        [
            "| Item | PR/Issue | Expected reply | Age | Source |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for item in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    _escape_table_cell(item.item),
                    _escape_table_cell(item.target),
                    f"`{_escape_table_cell(item.expected_reply)}`",
                    _format_age(item, now=now_dt),
                    _escape_table_cell(item.source),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _escape_table_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ").strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render pending founder/operator one-word decision packets."
    )
    parser.add_argument(
        "--decisions-root",
        default=str(DEFAULT_DECISIONS_ROOT),
        help="Directory of local founder decision packet markdown files.",
    )
    parser.add_argument(
        "--packet-file",
        action="append",
        default=[],
        help="Additional founder decision packet markdown file to parse.",
    )
    parser.add_argument(
        "--issue-comments-json",
        default=None,
        help="Optional JSON export of issue comments containing founder decision packets.",
    )
    parser.add_argument(
        "--now",
        default=None,
        help="Override current UTC timestamp for deterministic rendering.",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of Markdown.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.now:
        now = _parse_datetime(args.now)
        if now is None:
            raise SystemExit(f"invalid --now timestamp: {args.now}")
    else:
        now = datetime.now(tz=UTC)
    items = collect_decision_items(
        decisions_root=Path(args.decisions_root),
        packet_files=[Path(path) for path in args.packet_file],
        issue_comments_json=Path(args.issue_comments_json) if args.issue_comments_json else None,
    )
    if args.json:
        payload = {
            "generated_at": now.isoformat().replace("+00:00", "Z"),
            "count": len(items),
            "items": [
                {
                    "item": item.item,
                    "target": item.target,
                    "requested_action": item.requested_action,
                    "expected_reply": item.expected_reply,
                    "source": item.source,
                    "packet_generated_at": (
                        item.packet_generated_at.isoformat().replace("+00:00", "Z")
                        if item.packet_generated_at
                        else None
                    ),
                    "age": _format_age(item, now=now),
                }
                for item in items
            ],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    print(render_markdown(items, now=now), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
