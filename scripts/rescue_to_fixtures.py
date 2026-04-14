#!/usr/bin/env python3
"""TW-03: Convert repeated rescue classes into linked fixture or issue work."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from aragora.swarm.rescue_events import DEFAULT_RESCUE_LEDGER_PATH, RescueEventLedger
from scripts.harvest_rescue_classes import (
    DEFAULT_PRODUCTIZATION_MAP_PATH,
    load_productization_map,
    summarize_rescue_classes,
)


def load_rescue_productization_report(
    *,
    ledger_path: Path = DEFAULT_RESCUE_LEDGER_PATH,
    threshold: int = 2,
    recent_limit: int = 500,
    example_limit: int = 5,
    one_off_limit: int = 20,
    productization_map_path: Path = DEFAULT_PRODUCTIZATION_MAP_PATH,
) -> dict[str, Any]:
    ledger = RescueEventLedger(path=ledger_path)
    productization_map = load_productization_map(productization_map_path)
    return summarize_rescue_classes(
        ledger,
        threshold=threshold,
        recent_limit=recent_limit,
        example_limit=example_limit,
        one_off_limit=one_off_limit,
        productization_map=productization_map,
    )


def _slugify_fragment(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.strip().lower())
    return slug.strip("-") or "repeated-rescue-class"


def build_issue_drafts(report: dict[str, Any]) -> list[dict[str, Any]]:
    repeated_rows = list(report.get("repeated_classes") or [])
    drafts: list[dict[str, Any]] = []
    for row in repeated_rows:
        if str(row.get("productization_status") or "").strip() != "unlinked":
            continue
        rescue_class = str(row.get("class") or "").strip()
        count = int(row.get("count", 0) or 0)
        issue_numbers = [
            int(item)
            for item in list(row.get("issue_numbers") or [])
            if isinstance(item, int) and item > 0
        ]
        title = (
            f"[TW-03] Productize repeated rescue class: "
            f"{_slugify_fragment(rescue_class)} ({count}x)"
        )
        examples_text = (
            "\n".join(f"- #{issue_number}" for issue_number in issue_numbers) or "- none"
        )
        body = (
            "## Goal\n"
            f"Productize the repeated rescue class `{rescue_class}` so it stops requiring ad hoc human intervention.\n\n"
            "## Why now\n"
            f"This rescue class appeared {count} times in the recent rescue ledger and is still unlinked.\n\n"
            "## Evidence\n"
            f"- Rescue class: `{rescue_class}`\n"
            f"- Count: {count}\n"
            f"- Event type: `{str(row.get('event_type') or '').strip()}`\n"
            f"- Reason excerpt: `{str(row.get('reason_excerpt') or '').strip()}`\n\n"
            "### Example issue numbers\n"
            f"{examples_text}\n\n"
            "## Acceptance\n"
            "- Either add a benchmark fixture path to `docs/benchmarks/rescue_productization.json`\n"
            "- Or link the bounded substrate/control-plane issue that resolves this class\n"
            "- Keep the harvest/report loop truthful so this class shows up as linked on the next pass\n"
        )
        drafts.append(
            {
                "class": rescue_class,
                "count": count,
                "issue_numbers": issue_numbers,
                "title": title,
                "body": body,
            }
        )
    return drafts


def render_report(report: dict[str, Any], *, issue_drafts: list[dict[str, Any]]) -> str:
    summary = dict(report.get("summary") or {})
    repeated_rows = list(report.get("repeated_classes") or [])
    if not repeated_rows:
        return "No repeated rescue classes found."

    linked_rows = [
        row
        for row in repeated_rows
        if str(row.get("productization_status") or "").strip() != "unlinked"
    ]
    unlinked_rows = [
        row
        for row in repeated_rows
        if str(row.get("productization_status") or "").strip() == "unlinked"
    ]

    lines = [
        "=" * 60,
        "  TW-03: RESCUE PRODUCTIZATION REPORT",
        "=" * 60,
        f"  repeated classes:        {int(summary.get('repeated_class_count', 0) or 0)}",
        f"  linked repeated classes: {len(linked_rows)}",
        f"  issue drafts queued:     {len(issue_drafts)}",
        "",
    ]
    if linked_rows:
        lines.append("  Already linked:")
        for row in linked_rows:
            lines.append(
                "    - "
                f"{row['class']} -> "
                f"{str(row.get('productization_target_kind') or '').strip()}:"
                f"{str(row.get('productization_target') or '').strip()}"
            )
        lines.append("")
    if unlinked_rows:
        lines.append("  Still unlinked:")
        for row in unlinked_rows:
            issue_numbers = (
                ", ".join(f"#{num}" for num in list(row.get("issue_numbers") or [])) or "-"
            )
            lines.append(f"    - {row['class']} ({row['count']}x) [{issue_numbers}]")
        lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)


def create_substrate_issues(
    issue_drafts: list[dict[str, Any]],
    *,
    repo: str = "synaptent/aragora",
    dry_run: bool = False,
) -> list[str]:
    created: list[str] = []
    for draft in issue_drafts:
        if dry_run:
            created.append(f"DRY-RUN: would create '{draft['title']}'")
            continue

        try:
            result = subprocess.run(
                [
                    "gh",
                    "issue",
                    "create",
                    "--repo",
                    repo,
                    "--title",
                    str(draft["title"]),
                    "--label",
                    "boss-ready",
                    "--body",
                    str(draft["body"]),
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            if result.returncode == 0:
                created.append(result.stdout.strip())
            else:
                created.append(f"FAILED: {result.stderr.strip()[:100]}")
        except (subprocess.TimeoutExpired, OSError) as exc:
            created.append(f"ERROR: {exc}")
    return created


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TW-03: Convert repeated rescue classes into fixture or issue work."
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=DEFAULT_RESCUE_LEDGER_PATH,
        help="Path to the RescueEvent JSONL ledger.",
    )
    parser.add_argument(
        "--productization-map",
        type=Path,
        default=DEFAULT_PRODUCTIZATION_MAP_PATH,
        help="Path to the tracked rescue-productization map.",
    )
    parser.add_argument("--threshold", type=int, default=2)
    parser.add_argument("--recent-limit", type=int, default=500)
    parser.add_argument("--example-limit", type=int, default=5)
    parser.add_argument("--one-off-limit", type=int, default=20)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--create-issues", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--repo", default="synaptent/aragora")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = load_rescue_productization_report(
        ledger_path=args.path,
        threshold=max(1, args.threshold),
        recent_limit=max(1, args.recent_limit),
        example_limit=max(1, args.example_limit),
        one_off_limit=max(0, args.one_off_limit),
        productization_map_path=args.productization_map,
    )
    issue_drafts = build_issue_drafts(report)

    if args.json:
        payload = {
            **report,
            "issue_drafts": issue_drafts,
        }
        print(json.dumps(payload, indent=2))
    else:
        print(render_report(report, issue_drafts=issue_drafts))

    if args.create_issues or args.dry_run:
        results = create_substrate_issues(
            issue_drafts,
            repo=str(args.repo),
            dry_run=bool(args.dry_run),
        )
        for item in results:
            print(f"  {item}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
