#!/usr/bin/env python3
"""Analyze boss-loop crash patterns from prompt-era metrics rows."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from aragora.swarm.issue_scanner import (  # noqa: E402
    DEFAULT_BOSS_METRICS_PATH,
    historical_category_stats,
)
from aragora.swarm.terminal_truth import classify_from_metrics  # noqa: E402


def load_prompt_rows(metrics_file: Path) -> list[dict[str, Any]]:
    """Load prompt-era rows only (`prompt_chars > 0`)."""
    if not metrics_file.exists():
        return []

    rows: list[dict[str, Any]] = []
    with metrics_file.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            if float(row.get("prompt_chars", 0) or 0) <= 0:
                continue
            rows.append(row)
    return rows


def terminal_class_counts(rows: list[dict[str, Any]]) -> Counter[str]:
    """Count prompt-era rows by terminal class."""
    return Counter(classify_from_metrics(row).value for row in rows)


def low_success_categories(
    category_stats: dict[str, dict[str, float]],
    *,
    threshold: float,
) -> list[str]:
    """Return categories whose historical success rate is below the threshold."""
    return sorted(
        category
        for category, stats in category_stats.items()
        if stats.get("success_rate", 0.0) < threshold
    )


def render_report(
    rows: list[dict[str, Any]],
    counts: Counter[str],
    category_stats: dict[str, dict[str, float]],
    *,
    threshold: float,
) -> str:
    """Render a human-readable crash analysis report."""
    lines = [
        f"Prompt-era rows analyzed: {len(rows)}",
        "",
        "Terminal class counts:",
    ]
    for terminal_class, count in counts.most_common():
        lines.append(f"  {terminal_class}: {count}")

    if not category_stats:
        lines.extend(
            [
                "",
                "Category analysis:",
                "  No category stats available. Metrics rows are missing issue titles and title lookup returned no data.",
            ]
        )
        return "\n".join(lines)

    lines.extend(["", "Per-category stats:"])
    for category in sorted(category_stats):
        stats = category_stats[category]
        lines.append(
            "  "
            f"{category}: success_rate={stats['success_rate']:.3f} "
            f"crash_rate={stats['crash_rate']:.3f} "
            f"avg_elapsed={stats['avg_elapsed_seconds']:.1f}s "
            f"total={int(stats['total'])}"
        )

    deprioritized = low_success_categories(category_stats, threshold=threshold)
    lines.extend(["", f"Categories below success threshold ({threshold:.2f}):"])
    if deprioritized:
        lines.extend(f"  - {category}" for category in deprioritized)
    else:
        lines.append("  none")

    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze boss-loop crash patterns.")
    parser.add_argument(
        "--metrics-file",
        type=Path,
        default=REPO_ROOT / DEFAULT_BOSS_METRICS_PATH,
        help="Path to boss_metrics.jsonl",
    )
    parser.add_argument(
        "--min-success-rate",
        type=float,
        default=0.3,
        help="Threshold below which categories should be deprioritized",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = load_prompt_rows(args.metrics_file)
    if not rows:
        print(f"No prompt-era rows found in {args.metrics_file}")
        return 1

    counts = terminal_class_counts(rows)
    category_stats = historical_category_stats(args.metrics_file)
    print(
        render_report(
            rows,
            counts,
            category_stats,
            threshold=args.min_success_rate,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
