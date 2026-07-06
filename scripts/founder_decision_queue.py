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
import subprocess
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


@dataclass(frozen=True)
class DecisionSource:
    source: str
    body: str
    thread_key: str
    source_created_at: datetime | None = None
    thread_open: bool = True
    later_thread_comments: tuple[tuple[datetime | None, str], ...] = ()

    @property
    def packet_time(self) -> datetime | None:
        return self.source_created_at or _packet_generated_at(self.body)


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
        if line.startswith("#"):
            heading = line.lstrip("#").strip().lower()
            if not in_section and heading == "pending rulings":
                in_section = True
                continue
            if in_section:
                break
        if not in_section:
            continue
        cells = _split_table_row(line)
        if cells is None:
            if rows and line.strip():
                break
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


def _packet_sort_key(source: DecisionSource) -> tuple[datetime, str]:
    return (
        source.packet_time or datetime.min.replace(tzinfo=UTC),
        source.source,
    )


def _newest_sources_by_thread(sources: Iterable[DecisionSource]) -> list[DecisionSource]:
    newest: dict[str, DecisionSource] = {}
    for source in sources:
        current = newest.get(source.thread_key)
        if current is None or _packet_sort_key(source) > _packet_sort_key(current):
            newest[source.thread_key] = source
    return list(newest.values())


def _local_thread_key(path: Path, *, decisions_root: Path | None = None) -> str:
    if decisions_root is not None:
        return f"local:{decisions_root.resolve()}"
    parent = path.parent if path.parent != Path("") else Path(".")
    return f"local:{parent.resolve()}"


def _thread_key_from_comment(comment: dict[str, Any], *, fallback: str) -> str:
    for key in ("issue_url", "pull_request_url", "thread_url"):
        value = comment.get(key)
        if isinstance(value, str) and value:
            return value
    for key in ("html_url", "url"):
        value = comment.get(key)
        if not isinstance(value, str) or not value:
            continue
        match = re.search(r"/(?:issues|pulls|pull)/(\d+)(?:[/#?]|$)", value)
        if match:
            return f"github-thread:{match.group(1)}"
    return fallback


def _thread_open_from_comment(comment: dict[str, Any], *, default: bool) -> bool:
    for key in ("thread_state", "issue_state", "pr_state", "state"):
        value = comment.get(key)
        if isinstance(value, str):
            return value.lower() == "open"
    return default


def _comment_created_at(comment: dict[str, Any]) -> datetime | None:
    value = (
        comment.get("created_at")
        or comment.get("createdAt")
        or comment.get("updated_at")
        or comment.get("updatedAt")
    )
    if not isinstance(value, str):
        return None
    return _parse_datetime(value)


def _first_nonempty_line(body: str) -> str:
    for line in body.splitlines():
        compact = _compact_text(line)
        if compact:
            return compact
    return ""


def _target_number(target: str) -> str | None:
    match = re.search(r"(?:pull|issues)/(\d+)|#(\d+)", target)
    if not match:
        return None
    return match.group(1) or match.group(2)


def _comment_resolves_item(body: str, item: DecisionItem) -> bool:
    first_line = _strip_markdown_code(_first_nonempty_line(body))
    expected = item.expected_reply
    if first_line and first_line == expected:
        return True
    number = _target_number(item.target)
    if number is None:
        return False
    lowered = body.lower()
    if "settlement" not in lowered:
        return False
    number_pattern = re.escape(number)
    target_patterns = [
        rf"#\s*{number_pattern}\b",
        rf"/(?:issues|pull|pulls)/{number_pattern}\b",
        rf"\b(?:issue|pr|pull request)\s+#?\s*{number_pattern}\b",
        rf"\b{number_pattern}\b",
    ]
    return any(re.search(pattern, lowered) for pattern in target_patterns)


def _item_resolved_after_packet(source: DecisionSource, item: DecisionItem) -> bool:
    packet_time = source.packet_time
    for created_at, body in source.later_thread_comments:
        if packet_time is not None and created_at is not None and created_at <= packet_time:
            continue
        if _comment_resolves_item(body, item):
            return True
    return False


def _load_issue_comment_sources(path: Path) -> list[DecisionSource]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []

    default_thread_open = True
    if isinstance(payload, dict):
        state = payload.get("state") or payload.get("thread_state")
        if isinstance(state, str):
            default_thread_open = state.lower() == "open"
        comments_payload = payload.get("comments")
    else:
        comments_payload = payload

    if not isinstance(comments_payload, list):
        return []
    comments = [comment for comment in comments_payload if isinstance(comment, dict)]
    sources: list[DecisionSource] = []
    for index, comment in enumerate(comments):
        body = comment.get("body")
        if not isinstance(body, str) or "## Pending Rulings" not in body:
            continue
        source = str(comment.get("html_url") or comment.get("url") or f"{path}#{index}")
        thread_key = _thread_key_from_comment(comment, fallback=f"{path}")
        created_at = _comment_created_at(comment)
        later_comments: list[tuple[datetime | None, str]] = []
        for other in comments:
            other_body = other.get("body")
            if not isinstance(other_body, str) or "## Pending Rulings" in other_body:
                continue
            if _thread_key_from_comment(other, fallback=f"{path}") != thread_key:
                continue
            later_comments.append((_comment_created_at(other), other_body))
        sources.append(
            DecisionSource(
                source=source,
                body=body,
                thread_key=thread_key,
                source_created_at=created_at,
                thread_open=_thread_open_from_comment(comment, default=default_thread_open),
                later_thread_comments=tuple(later_comments),
            )
        )
    return sources


def _github_issue_comment_sources(
    *,
    repo: str,
    issue: str,
    payload: dict[str, Any],
) -> list[DecisionSource]:
    comments_payload = payload.get("comments")
    if not isinstance(comments_payload, list):
        return []
    comments = [comment for comment in comments_payload if isinstance(comment, dict)]
    thread_open = str(payload.get("state") or "open").lower() == "open"
    thread_url = str(payload.get("url") or f"https://github.com/{repo}/issues/{issue}")
    thread_key = f"github-thread:{issue}"
    sources: list[DecisionSource] = []
    for index, comment in enumerate(comments):
        body = comment.get("body")
        if not isinstance(body, str) or "## Pending Rulings" not in body:
            continue
        source = str(comment.get("url") or f"{thread_url}#comment-{index}")
        created_at = _comment_created_at(comment)
        later_comments: list[tuple[datetime | None, str]] = []
        for other in comments:
            other_body = other.get("body")
            if not isinstance(other_body, str) or "## Pending Rulings" in other_body:
                continue
            later_comments.append((_comment_created_at(other), other_body))
        sources.append(
            DecisionSource(
                source=source,
                body=body,
                thread_key=thread_key,
                source_created_at=created_at,
                thread_open=thread_open,
                later_thread_comments=tuple(later_comments),
            )
        )
    return sources


def _load_github_issue_sources(*, repo: str, issue: str) -> list[DecisionSource]:
    completed = subprocess.run(
        [
            "gh",
            "issue",
            "view",
            issue,
            "--repo",
            repo,
            "--json",
            "state,url,comments",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip() or "unknown gh error"
        raise RuntimeError(f"gh issue view failed for {repo}#{issue}: {message}")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"gh issue view returned malformed JSON for {repo}#{issue}") from exc
    if not isinstance(payload, dict):
        return []
    return _github_issue_comment_sources(repo=repo, issue=issue, payload=payload)


def _legacy_load_issue_comment_bodies(path: Path) -> list[tuple[str, str]]:
    """Compatibility shim for tests or callers that imported the private helper."""

    bodies: list[tuple[str, str]] = []
    for source in _load_issue_comment_sources(path):
        bodies.append((source.source, source.body))
    return bodies


_load_issue_comment_bodies = _legacy_load_issue_comment_bodies


def _collect_sources(
    *,
    decisions_root: Path,
    packet_files: Iterable[Path],
    issue_comments_json: Path | None,
    github_issues: Iterable[str],
    repo: str,
) -> list[DecisionSource]:
    sources: list[DecisionSource] = []
    if decisions_root.exists():
        for path in sorted(decisions_root.glob("*.md")):
            try:
                body = path.read_text(encoding="utf-8")
            except OSError:
                continue
            sources.append(
                DecisionSource(
                    source=str(path),
                    body=body,
                    thread_key=_local_thread_key(path, decisions_root=decisions_root),
                )
            )
    for path in packet_files:
        try:
            body = path.read_text(encoding="utf-8")
        except OSError:
            continue
        sources.append(
            DecisionSource(
                source=str(path),
                body=body,
                thread_key=_local_thread_key(path),
            )
        )
    if issue_comments_json is not None:
        sources.extend(_load_issue_comment_sources(issue_comments_json))
    for issue in github_issues:
        sources.extend(_load_github_issue_sources(repo=repo, issue=issue))
    return sources


def collect_decision_items(
    *,
    decisions_root: Path = DEFAULT_DECISIONS_ROOT,
    packet_files: Iterable[Path] = (),
    issue_comments_json: Path | None = None,
    github_issues: Iterable[str] = (),
    repo: str = "synaptent/aragora",
) -> list[DecisionItem]:
    deduped: dict[tuple[str, str, str], DecisionItem] = {}
    sources = _collect_sources(
        decisions_root=decisions_root,
        packet_files=packet_files,
        issue_comments_json=issue_comments_json,
        github_issues=github_issues,
        repo=repo,
    )
    for source in _newest_sources_by_thread(sources):
        if not source.thread_open:
            continue
        for item in parse_decision_packet(source.body, source=source.source):
            if _item_resolved_after_packet(source, item):
                continue
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
        "--github-issue",
        action="append",
        default=[],
        help="Read-only GitHub issue number to scan for founder decision packet comments.",
    )
    parser.add_argument(
        "--repo",
        default="synaptent/aragora",
        help="GitHub repository for --github-issue lookups.",
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
        github_issues=args.github_issue,
        repo=args.repo,
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
