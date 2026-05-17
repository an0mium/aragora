"""CLI handlers for ``aragora codex sessions {list,show,tail}``.

Read-only inspector for Codex Desktop local state. All output is secret-redacted
by default; full-transcript output (``--full``) writes to a file under
``.aragora/codex_sessions/`` unless ``--out -`` forces stdout.

Heavy imports are deferred to invocation time so adding this CLI does not slow
``aragora --help`` startup (see ``aragora/cli/backup.py`` for the same pattern).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aragora.codex.desktop_paths import CodexDesktopPaths

DEFAULT_OUTPUT_ROOT = Path(".aragora/codex_sessions")
DEFAULT_SINCE = "4h"
DEFAULT_LIMIT = 50
DEFAULT_MAX_EVENTS = 2000
DEFAULT_TAIL_INTERVAL_SECONDS = 5.0


def _parse_since(value: str) -> timedelta:
    from aragora.codex.duration import parse_duration

    return parse_duration(value)


def _resolve_paths(args: argparse.Namespace) -> "CodexDesktopPaths":
    from aragora.codex.desktop_paths import resolve

    return resolve(getattr(args, "codex_home", None))


def _missing_db_message(sqlite_path: Path) -> str:
    from aragora.codex.desktop_inspector import redact_display

    display_path = redact_display(sqlite_path)
    return (
        f"error: Codex Desktop state DB not found at {display_path}\n"
        "       Pass --codex-home <path> or set ARAGORA_CODEX_HOME."
    )


def _print_json(payload: Any) -> None:
    json.dump(payload, sys.stdout, indent=2, sort_keys=True, default=str)
    sys.stdout.write("\n")


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _output_path_inside_codex_home(out_path: Path, paths: "CodexDesktopPaths") -> bool:
    resolved_out = out_path.expanduser().resolve(strict=False)
    resolved_home = paths.home.expanduser().resolve(strict=False)
    return _is_relative_to(resolved_out, resolved_home)


def _format_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    if not rows:
        return "(no rows)"
    widths = {key: len(label) for key, label in columns}
    for row in rows:
        for key, _ in columns:
            value = str(row.get(key, ""))
            widths[key] = max(widths[key], len(value))
    header = "  ".join(label.ljust(widths[key]) for key, label in columns)
    separator = "  ".join("-" * widths[key] for key, _ in columns)
    body = "\n".join(
        "  ".join(str(row.get(key, "")).ljust(widths[key]) for key, _ in columns) for row in rows
    )
    return f"{header}\n{separator}\n{body}"


def cmd_codex_sessions_list(args: argparse.Namespace) -> int:
    """List Codex Desktop threads updated within ``--since``."""
    from aragora.codex.desktop_inspector import (
        humanize_ago,
        list_active_threads,
        redact_display,
        truncate,
    )

    if args.limit < 0:
        print("error: --limit must be >= 0", file=sys.stderr)
        return 2

    try:
        since = _parse_since(args.since)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    paths = _resolve_paths(args)
    if not paths.sqlite_path.exists():
        print(_missing_db_message(paths.sqlite_path), file=sys.stderr)
        return 1

    threads = list_active_threads(
        since=since,
        include_archived=args.include_archived,
        limit=args.limit,
        paths=paths,
    )

    if args.json:
        _print_json(
            {
                "schema": "aragora-codex-sessions-list/1.0",
                "generated_at": datetime.now(UTC).isoformat(),
                "codex_home": redact_display(paths.home),
                "since": args.since,
                "since_seconds": int(since.total_seconds()),
                "include_archived": bool(args.include_archived),
                "limit": int(args.limit),
                "count": len(threads),
                "threads": [t.to_list_dict() for t in threads],
            }
        )
        return 0

    now = datetime.now(UTC)
    rows = [
        {
            "id": t.id[:12],
            "ago": humanize_ago(t.updated_at, now=now),
            "model": (t.model or "")[:24],
            "tokens": str(t.tokens_used),
            "branch": (t.git_branch or "")[:18],
            "cwd": truncate(t.cwd, width=40),
            "title": truncate(t.title or "(no title)", width=60),
        }
        for t in threads
    ]
    table = _format_table(
        rows,
        columns=[
            ("id", "ID"),
            ("ago", "AGO"),
            ("model", "MODEL"),
            ("tokens", "TOKENS"),
            ("branch", "BRANCH"),
            ("cwd", "CWD"),
            ("title", "TITLE"),
        ],
    )
    print(table)
    print(f"\n{len(rows)} thread(s) updated since {since}.")
    return 0


def _resolve_rollout(
    target: str, args: argparse.Namespace
) -> "tuple[Path, str | None] | tuple[None, None]":
    """Resolve ``target`` (thread id or path) to a (rollout_path, thread_id) pair."""
    from aragora.codex.desktop_inspector import find_thread

    candidate = Path(target).expanduser()
    if candidate.exists() and candidate.is_file():
        return candidate, None
    paths = _resolve_paths(args)
    if not paths.sqlite_path.exists():
        return None, None
    summary = find_thread(target, paths=paths)
    if summary is None:
        return None, None
    return summary.rollout_path, summary.id


def cmd_codex_sessions_show(args: argparse.Namespace) -> int:
    """Summarize one session; with ``--full`` write the redacted transcript to a file."""
    from aragora.codex.desktop_inspector import (
        find_thread,
        iter_session_events,
        redact_display,
        summarize_session,
    )

    rollout, thread_id = _resolve_rollout(args.target, args)
    if rollout is None or not rollout.exists():
        print(
            f"error: could not resolve '{args.target}' to a rollout file\n"
            "       Pass a thread id (or 8+ char prefix) or a rollout path.",
            file=sys.stderr,
        )
        return 1

    paths = _resolve_paths(args)
    thread = find_thread(thread_id, paths=paths) if thread_id else None
    summary = summarize_session(rollout, max_events=args.max_events)

    if not args.full:
        if args.json:
            payload = {
                "thread": thread.to_dict() if thread else None,
                "summary": summary.to_dict(),
            }
            _print_json(payload)
            return 0
        if thread:
            print(f"Thread:    {thread.id}")
            print(f"Title:     {thread.title or '(no title)'}")
            print(f"Cwd:       {thread.cwd}")
            print(f"Model:     {thread.model or '(unknown)'}")
            print(f"Updated:   {thread.updated_at.isoformat()}")
            print(f"Tokens:    {thread.tokens_used}")
            if thread.git_branch:
                print(f"Branch:    {thread.git_branch}")
            if thread.git_sha:
                print(f"Sha:       {thread.git_sha[:12]}")
        print(f"Rollout:   {redact_display(summary.rollout_path)}")
        print(f"Scanned:   {summary.events_scanned}{' (truncated)' if summary.truncated else ''}")
        if summary.event_type_counts:
            print("Events:")
            for name, count in sorted(summary.event_type_counts.items(), key=lambda kv: -kv[1]):
                print(f"    {count:6}  {name}")
        if summary.tool_call_counts:
            print("Tool calls:")
            for name, count in sorted(summary.tool_call_counts.items(), key=lambda kv: -kv[1]):
                print(f"    {count:6}  {name}")
        if summary.first_user_message:
            print("First user:")
            for line in summary.first_user_message.splitlines()[:5]:
                print(f"    {line}")
        if summary.last_user_message and summary.last_user_message != summary.first_user_message:
            print("Last user:")
            for line in summary.last_user_message.splitlines()[:5]:
                print(f"    {line}")
        return 0

    # --full path
    out_arg = args.out or ""
    if out_arg == "-":
        out_handle = sys.stdout
        out_path: Path | None = None
    else:
        out_path = (
            Path(out_arg).expanduser()
            if out_arg
            else DEFAULT_OUTPUT_ROOT / f"{(thread_id or rollout.stem)}.jsonl"
        )
        if _output_path_inside_codex_home(out_path, paths):
            print(
                "error: refusing to write --full output inside Codex Desktop home; "
                "use --out - for stdout or choose a path outside --codex-home",
                file=sys.stderr,
            )
            return 2
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_handle = out_path.open("w", encoding="utf-8")

    written = 0
    bytes_written = 0
    try:
        for event in iter_session_events(rollout, redact=True):
            line = json.dumps(event, sort_keys=True, default=str)
            out_handle.write(line + "\n")
            written += 1
            bytes_written += len((line + "\n").encode("utf-8"))
    finally:
        if out_path is not None:
            out_handle.close()

    if out_path is not None:
        print(f"wrote {out_path} ({written} events, {bytes_written} bytes redacted)")
    return 0


def cmd_codex_sessions_tail(args: argparse.Namespace) -> int:
    """Poll active sessions and print new redacted events as they arrive.

    Default ``--since 4h`` window controls which sessions are watched. The
    loop never exits on its own; ``Ctrl-C`` to stop. Suitable for piping to
    grep or pumping into the Monitor tool.
    """
    from aragora.codex.desktop_inspector import iter_session_events_from_offset, list_active_threads

    if args.interval <= 0:
        print("error: --interval must be > 0", file=sys.stderr)
        return 2

    try:
        since = _parse_since(args.since)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    paths = _resolve_paths(args)
    if not paths.sqlite_path.exists():
        print(_missing_db_message(paths.sqlite_path), file=sys.stderr)
        return 1

    # Track byte offsets per rollout so each poll only emits new events.
    offsets: dict[Path, int] = {}
    initialized = False
    try:
        while True:
            threads = list_active_threads(since=since, paths=paths)
            for thread in threads:
                rollout = thread.rollout_path
                if not rollout.exists():
                    continue
                current_size = rollout.stat().st_size
                if rollout in offsets:
                    prev = offsets[rollout]
                else:
                    prev = 0 if args.from_start or initialized else current_size
                    offsets[rollout] = prev
                if current_size < prev:
                    prev = 0
                    offsets[rollout] = 0
                if current_size == prev:
                    continue
                for event, next_offset in iter_session_events_from_offset(
                    rollout,
                    offset=prev,
                    redact=True,
                ):
                    if not event:
                        offsets[rollout] = next_offset
                        continue
                    serialized = json.dumps(event, sort_keys=True, default=str)
                    print(f"{thread.id[:12]}  {serialized}")
                    offsets[rollout] = next_offset
            initialized = True
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 0
