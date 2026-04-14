#!/usr/bin/env python3
"""Harvest repeated rescue classes from the RescueEvent ledger."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from aragora.swarm.rescue_events import DEFAULT_RESCUE_LEDGER_PATH, RescueEventLedger


def harvest_repeated_rescue_classes(
    ledger: RescueEventLedger,
    *,
    threshold: int = 2,
    recent_limit: int = 500,
    example_limit: int = 5,
) -> list[dict[str, Any]]:
    """Return repeated rescue classes plus example issue numbers."""
    repeated = ledger.repeated_classes(threshold=threshold)
    if not repeated:
        return []

    issue_numbers_by_class: dict[str, list[int]] = defaultdict(list)
    for event in ledger.recent(limit=recent_limit):
        if event.issue_number is None:
            continue
        key = f"{event.event_type}:{event.reason[:60]}"
        bucket = issue_numbers_by_class[key]
        if event.issue_number not in bucket:
            bucket.append(event.issue_number)

    rows: list[dict[str, Any]] = []
    for item in repeated:
        class_name = str(item.get("class", "")).strip()
        count = int(item.get("count", 0) or 0)
        event_type, _, reason_excerpt = class_name.partition(":")
        rows.append(
            {
                "class": class_name,
                "event_type": event_type,
                "reason_excerpt": reason_excerpt,
                "count": count,
                "issue_numbers": issue_numbers_by_class.get(class_name, [])[:example_limit],
            }
        )
    return rows


def _format_issue_numbers(issue_numbers: list[int]) -> str:
    if not issue_numbers:
        return "-"
    return ", ".join(f"#{issue_number}" for issue_number in issue_numbers)


def render_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "No repeated rescue classes found."

    headers = ("class", "count", "issue_numbers")
    table_rows = [
        (
            str(row.get("class", "")).strip(),
            str(int(row.get("count", 0) or 0)),
            _format_issue_numbers(list(row.get("issue_numbers", []) or [])),
        )
        for row in rows
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in table_rows))
        for index in range(len(headers))
    ]

    def _render_row(values: tuple[str, str, str]) -> str:
        return " | ".join(value.ljust(widths[index]) for index, value in enumerate(values))

    separator = "-+-".join("-" * width for width in widths)
    lines = [_render_row(headers), separator]
    lines.extend(_render_row(row) for row in table_rows)
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Harvest repeated rescue classes from the RescueEvent ledger."
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=DEFAULT_RESCUE_LEDGER_PATH,
        help="Path to the RescueEvent JSONL ledger.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=2,
        help="Minimum count required before a rescue class is reported.",
    )
    parser.add_argument(
        "--recent-limit",
        type=int,
        default=500,
        help="How many recent ledger events to inspect for examples.",
    )
    parser.add_argument(
        "--example-limit",
        type=int,
        default=5,
        help="Maximum number of example issue numbers to print per rescue class.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of a table.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    ledger = RescueEventLedger(path=args.path)
    rows = harvest_repeated_rescue_classes(
        ledger,
        threshold=max(1, args.threshold),
        recent_limit=max(1, args.recent_limit),
        example_limit=max(1, args.example_limit),
    )
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        print(render_table(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
