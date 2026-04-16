#!/usr/bin/env python3
"""Harvest repeated rescue classes from the RescueEvent ledger."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from aragora.swarm.rescue_events import DEFAULT_RESCUE_LEDGER_PATH, RescueEventLedger

DEFAULT_PRODUCTIZATION_MAP_PATH = REPO_ROOT / "docs" / "benchmarks" / "rescue_productization.json"


def load_productization_map(path: Path) -> dict[str, dict[str, str]]:
    """Load rescue-class linkage metadata from a tracked JSON file."""
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Productization map at {path} must be a JSON object")
    raw_entries = payload.get("entries", [])
    if not isinstance(raw_entries, list):
        raise ValueError(f"Productization map at {path} must contain an 'entries' list")

    entries: dict[str, dict[str, str]] = {}
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, dict):
            continue
        class_name = str(raw_entry.get("class", "") or "").strip()
        if not class_name:
            continue
        entries[class_name] = {
            "target_kind": str(raw_entry.get("target_kind", "") or "").strip(),
            "target": str(raw_entry.get("target", "") or "").strip(),
            "title": str(raw_entry.get("title", "") or "").strip(),
            "notes": str(raw_entry.get("notes", "") or "").strip(),
        }
    return entries


def _rescue_class_key(event: Any) -> str:
    return f"{event.event_type}:{event.reason[:60]}"


def _productization_fields(
    class_name: str,
    *,
    productization_map: dict[str, dict[str, str]] | None = None,
) -> dict[str, str]:
    entry = dict((productization_map or {}).get(class_name, {}) or {})
    target_kind = str(entry.get("target_kind", "") or "").strip()
    target = str(entry.get("target", "") or "").strip()
    if target_kind == "fixture" and target:
        status = "linked_fixture"
    elif target_kind == "issue" and target:
        status = "linked_issue"
    elif target:
        status = "linked_other"
    else:
        status = "unlinked"
    return {
        "productization_status": status,
        "productization_target_kind": target_kind,
        "productization_target": target,
        "productization_title": str(entry.get("title", "") or "").strip(),
        "productization_notes": str(entry.get("notes", "") or "").strip(),
    }


def _class_counts(
    ledger: RescueEventLedger,
    *,
    recent_limit: int,
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for event in ledger.recent(limit=recent_limit):
        counts[_rescue_class_key(event)] += 1
    return counts


def harvest_repeated_rescue_classes(
    ledger: RescueEventLedger,
    *,
    threshold: int = 2,
    recent_limit: int = 500,
    example_limit: int = 5,
    productization_map: dict[str, dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    """Return repeated rescue classes plus example issue numbers."""
    repeated = ledger.repeated_classes(threshold=threshold)
    if not repeated:
        return []

    issue_numbers_by_class: dict[str, list[int]] = defaultdict(list)
    for event in ledger.recent(limit=recent_limit):
        if event.issue_number is None:
            continue
        key = _rescue_class_key(event)
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
                **_productization_fields(
                    class_name,
                    productization_map=productization_map,
                ),
            }
        )
    return rows


def summarize_rescue_classes(
    ledger: RescueEventLedger,
    *,
    threshold: int = 2,
    recent_limit: int = 500,
    example_limit: int = 5,
    one_off_limit: int = 20,
    productization_map: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Summarize repeated rescue patterns versus one-off or below-threshold noise."""
    counts = _class_counts(ledger, recent_limit=recent_limit)
    repeated_rows = harvest_repeated_rescue_classes(
        ledger,
        threshold=threshold,
        recent_limit=recent_limit,
        example_limit=example_limit,
        productization_map=productization_map,
    )
    repeated_class_names = {str(row.get("class", "")).strip() for row in repeated_rows}

    one_off_rows: list[dict[str, Any]] = []
    below_threshold_rows: list[dict[str, Any]] = []
    for class_name, count in counts.most_common():
        if class_name in repeated_class_names:
            continue
        event_type, _, reason_excerpt = class_name.partition(":")
        row = {
            "class": class_name,
            "event_type": event_type,
            "reason_excerpt": reason_excerpt,
            "count": count,
            **_productization_fields(
                class_name,
                productization_map=productization_map,
            ),
        }
        if count <= 1:
            one_off_rows.append(row)
        else:
            below_threshold_rows.append(row)

    linked_fixture_count = sum(
        1 for row in repeated_rows if row.get("productization_status") == "linked_fixture"
    )
    linked_issue_count = sum(
        1 for row in repeated_rows if row.get("productization_status") == "linked_issue"
    )
    linked_other_count = sum(
        1 for row in repeated_rows if row.get("productization_status") == "linked_other"
    )

    return {
        "summary": {
            "threshold": threshold,
            "recent_limit": recent_limit,
            "total_unique_classes": len(counts),
            "repeated_class_count": len(repeated_rows),
            "one_off_class_count": sum(1 for count in counts.values() if count <= 1),
            "below_threshold_class_count": sum(
                1 for count in counts.values() if 1 < count < threshold
            ),
            "linked_fixture_count": linked_fixture_count,
            "linked_issue_count": linked_issue_count,
            "linked_other_count": linked_other_count,
            "unlinked_repeated_class_count": sum(
                1 for row in repeated_rows if row.get("productization_status") == "unlinked"
            ),
        },
        "repeated_classes": repeated_rows,
        "one_off_classes": one_off_rows[: max(0, one_off_limit)],
        "below_threshold_classes": below_threshold_rows[: max(0, one_off_limit)],
    }


def _format_issue_numbers(issue_numbers: list[int]) -> str:
    if not issue_numbers:
        return "-"
    return ", ".join(f"#{issue_number}" for issue_number in issue_numbers)


def render_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "No repeated rescue classes found."

    headers = ("class", "count", "productization", "target", "issue_numbers")
    table_rows = [
        (
            str(row.get("class", "")).strip(),
            str(int(row.get("count", 0) or 0)),
            str(row.get("productization_status", "")).strip() or "unlinked",
            str(row.get("productization_target", "")).strip() or "-",
            _format_issue_numbers(list(row.get("issue_numbers", []) or [])),
        )
        for row in rows
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in table_rows))
        for index in range(len(headers))
    ]

    def _render_row(values: tuple[str, str, str, str, str]) -> str:
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
    parser.add_argument(
        "--report-json",
        action="store_true",
        help="Emit a richer JSON report with repeated, one-off, and below-threshold classes.",
    )
    parser.add_argument(
        "--productization-map",
        type=Path,
        default=DEFAULT_PRODUCTIZATION_MAP_PATH,
        help="Path to the tracked rescue-productization map.",
    )
    parser.add_argument(
        "--one-off-limit",
        type=int,
        default=20,
        help="Maximum number of one-off or below-threshold classes to emit in report JSON.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ledger = RescueEventLedger(path=args.path)
    productization_map = load_productization_map(args.productization_map)
    rows = harvest_repeated_rescue_classes(
        ledger,
        threshold=max(1, args.threshold),
        recent_limit=max(1, args.recent_limit),
        example_limit=max(1, args.example_limit),
        productization_map=productization_map,
    )
    if args.report_json:
        print(
            json.dumps(
                summarize_rescue_classes(
                    ledger,
                    threshold=max(1, args.threshold),
                    recent_limit=max(1, args.recent_limit),
                    example_limit=max(1, args.example_limit),
                    one_off_limit=max(0, args.one_off_limit),
                    productization_map=productization_map,
                ),
                indent=2,
            )
        )
    elif args.json:
        print(json.dumps(rows, indent=2))
    else:
        print(render_table(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
