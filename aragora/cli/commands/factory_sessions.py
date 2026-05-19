"""CLI handlers for ``aragora factory sessions brief``.

The command surfaces Factory/Droid local session metadata as redacted operator
briefs. It intentionally does not read raw Factory transcripts, prompts, logs,
or history files.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_SINCE = "4h"
DEFAULT_LIMIT = 50


def _parse_since(value: str):
    from aragora.codex.duration import parse_duration

    return parse_duration(value)


def _print_json(payload: Any) -> None:
    json.dump(payload, sys.stdout, indent=2, sort_keys=True, default=str)
    sys.stdout.write("\n")


def _format_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    if not rows:
        return "(no rows)"
    widths = {key: len(label) for key, label in columns}
    for row in rows:
        for key, _ in columns:
            widths[key] = max(widths[key], len(str(row.get(key, ""))))
    header = "  ".join(label.ljust(widths[key]) for key, label in columns)
    separator = "  ".join("-" * widths[key] for key, _ in columns)
    body = "\n".join(
        "  ".join(str(row.get(key, "")).ljust(widths[key]) for key, _ in columns) for row in rows
    )
    return f"{header}\n{separator}\n{body}"


def _compact_brief(row: dict[str, Any]) -> dict[str, Any]:
    raw_router = row.get("router")
    router: dict[str, Any] = raw_router if isinstance(raw_router, dict) else {}
    return {
        "provider": row.get("provider"),
        "session_id": row.get("session_id"),
        "age": row.get("age"),
        "branch": row.get("branch"),
        "pr_number": row.get("pr_number"),
        "route": router.get("category"),
        "conflict_risk": row.get("conflict_risk"),
        "prompt_needed": row.get("prompt_needed"),
        "prompt_needed_reason": row.get("prompt_needed_reason"),
        "direct_steering_available": row.get("direct_steering_available"),
        "recommended_next_prompt": router.get("recommended_next_prompt"),
    }


def _format_brief_table(briefs: list[dict[str, Any]]) -> str:
    rows = []
    for brief in briefs:
        raw_router = brief.get("router")
        router: dict[str, Any] = raw_router if isinstance(raw_router, dict) else {}
        raw_lane = brief.get("matched_lane")
        lane: dict[str, Any] = raw_lane if isinstance(raw_lane, dict) else {}
        rows.append(
            {
                "id": str(brief.get("session_id") or "")[:18],
                "age": brief.get("age") or "",
                "route": router.get("category") or "",
                "pr": f"#{brief['pr_number']}" if brief.get("pr_number") else "",
                "branch": str(brief.get("branch") or "")[:28],
                "lane": str(lane.get("lane_id") or "")[:24],
                "conflict": brief.get("conflict_risk") or "",
            }
        )
    return _format_table(
        rows,
        columns=[
            ("id", "ID"),
            ("age", "AGE"),
            ("route", "ROUTE"),
            ("pr", "PR"),
            ("branch", "BRANCH"),
            ("lane", "LANE"),
            ("conflict", "CONFLICT"),
        ],
    )


def cmd_factory_sessions_brief(args: argparse.Namespace) -> int:
    from aragora.factory.session_inspector import build_factory_session_briefs, redact_display

    if args.limit < 0:
        print("error: --limit must be >= 0", file=sys.stderr)
        return 2
    try:
        since = _parse_since(args.since)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    repo_root = getattr(args, "repo_root", None)
    if repo_root == "":
        repo_root = None
    factory_home = getattr(args, "factory_home", None)
    briefs = [
        brief.to_dict()
        for brief in build_factory_session_briefs(
            factory_home=factory_home,
            repo_root=repo_root,
            since=since,
            limit=args.limit,
            session=getattr(args, "session", None),
        )
    ]
    compact = bool(getattr(args, "compact", False))
    output_briefs = [_compact_brief(row) for row in briefs] if compact else briefs
    payload = {
        "schema": "aragora-factory-sessions-brief/1.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "factory_home": redact_display(Path(factory_home).expanduser())
        if factory_home
        else redact_display(Path("~/.factory").expanduser()),
        "repo_root": redact_display(Path(repo_root).expanduser()) if repo_root else None,
        "since": args.since,
        "since_seconds": int(since.total_seconds()),
        "limit": int(args.limit),
        "session": redact_display(getattr(args, "session", None)),
        "compact": compact,
        "count": len(output_briefs),
        "briefs": output_briefs,
    }
    if args.json:
        _print_json(payload)
        return 0

    print(_format_brief_table(output_briefs))
    print(f"\n{len(output_briefs)} Factory/Droid brief(s) updated since {since}.")
    return 0
