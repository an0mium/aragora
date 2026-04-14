#!/usr/bin/env python3
"""B0 Benchmark Scorecard: measure autonomous execution truth metrics.

Reads boss_metrics.jsonl and produces a no-rescue scorecard showing:
- Total ticks / unique issues attempted
- No-rescue success rate (deliverable_pr_created without human intervention)
- Terminal class distribution
- Failure class breakdown
- Per-category success rates (when outcome learner data exists)

This is the primary truth metric for the wedge-first roadmap (TW-01/TW-02).

Usage:
  python3 scripts/measure_b0_scorecard.py
  python3 scripts/measure_b0_scorecard.py --metrics .aragora/overnight/boss_metrics.jsonl
  python3 scripts/measure_b0_scorecard.py --json
  python3 scripts/measure_b0_scorecard.py --ci --threshold 0.5
  python3 scripts/measure_b0_scorecard.py --window 100
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

DEFAULT_METRICS_PATH = Path(".aragora/overnight/boss_metrics.jsonl")

# Terminal classes that count as autonomous success (no human rescue)
SUCCESS_CLASSES = frozenset(
    {
        "success_merged",
        "success_pr_created",
        "deliverable_pr_created",
    }
)

# Terminal classes that indicate the system tried and failed
FAILURE_CLASSES = frozenset(
    {
        "rescue_timeout",
        "rescue_worker_crash",
        "rescue_no_deliverable",
        "blocked_not_dispatch_bounded",
        "blocked_validation_target_missing",
        "blocked_sanitation_failed",
        "blocked_auth_failure",
        "blocked_no_runner",
    }
)


def load_metrics(path: Path, window: int | None = None) -> list[dict[str, Any]]:
    """Load boss metrics JSONL, optionally limiting to last N rows."""
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    except OSError:
        return []
    if window and window > 0:
        return rows[-window:]
    return rows


def compute_scorecard(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute the B0 benchmark scorecard from metrics rows."""
    if not rows:
        return {"status": "no_data", "total_ticks": 0}

    terminal_classes: Counter[str] = Counter()
    issues_attempted: set[int] = set()
    issues_succeeded: set[int] = set()
    issues_failed: set[int] = set()
    elapsed_times: list[float] = []

    for row in rows:
        tc = str(row.get("terminal_class", "")).strip()
        if tc:
            terminal_classes[tc] += 1

        issue_num = row.get("issue_number")
        if isinstance(issue_num, int) and issue_num > 0:
            issues_attempted.add(issue_num)
            if tc in SUCCESS_CLASSES:
                issues_succeeded.add(issue_num)
            elif tc in FAILURE_CLASSES:
                issues_failed.add(issue_num)

        elapsed = row.get("elapsed_seconds")
        if isinstance(elapsed, (int, float)) and elapsed > 0:
            elapsed_times.append(float(elapsed))

    total_ticks = len(rows)
    success_ticks = sum(terminal_classes[tc] for tc in SUCCESS_CLASSES)
    failure_ticks = sum(terminal_classes[tc] for tc in FAILURE_CLASSES)

    return {
        "status": "active",
        "total_ticks": total_ticks,
        "unique_issues_attempted": len(issues_attempted),
        "unique_issues_succeeded": len(issues_succeeded),
        "unique_issues_failed": len(issues_failed),
        "no_rescue_success_rate": round(len(issues_succeeded) / len(issues_attempted), 3)
        if issues_attempted
        else 0.0,
        "tick_success_rate": round(success_ticks / total_ticks, 3) if total_ticks else 0.0,
        "terminal_class_distribution": dict(terminal_classes.most_common()),
        "success_classes": {
            tc: terminal_classes[tc] for tc in sorted(SUCCESS_CLASSES) if terminal_classes[tc]
        },
        "failure_classes": {
            tc: terminal_classes[tc] for tc in sorted(FAILURE_CLASSES) if terminal_classes[tc]
        },
        "median_elapsed_seconds": round(sorted(elapsed_times)[len(elapsed_times) // 2], 1)
        if elapsed_times
        else 0.0,
        "mean_elapsed_seconds": round(sum(elapsed_times) / len(elapsed_times), 1)
        if elapsed_times
        else 0.0,
    }


def print_scorecard(scorecard: dict[str, Any]) -> None:
    """Print human-readable scorecard."""
    if scorecard.get("status") == "no_data":
        print("No boss metrics data found.")
        return

    print("=" * 60)
    print("  B0 BENCHMARK SCORECARD")
    print("=" * 60)
    print(f"  Total ticks:              {scorecard['total_ticks']}")
    print(f"  Unique issues attempted:  {scorecard['unique_issues_attempted']}")
    print(f"  Unique issues succeeded:  {scorecard['unique_issues_succeeded']}")
    print(f"  Unique issues failed:     {scorecard['unique_issues_failed']}")
    print(f"  No-rescue success rate:   {scorecard['no_rescue_success_rate']:.1%}")
    print(f"  Per-tick success rate:    {scorecard['tick_success_rate']:.1%}")
    print(f"  Median elapsed time:      {scorecard['median_elapsed_seconds']:.0f}s")
    print(f"  Mean elapsed time:        {scorecard['mean_elapsed_seconds']:.0f}s")
    print()
    print("  Terminal class distribution:")
    for tc, count in scorecard.get("terminal_class_distribution", {}).items():
        marker = "+" if tc in SUCCESS_CLASSES else "-" if tc in FAILURE_CLASSES else " "
        print(f"    {marker} {tc}: {count}")
    print("=" * 60)


def render_ci_summary(scorecard: dict[str, Any], *, threshold: float) -> str:
    success_rate = float(scorecard.get("no_rescue_success_rate", 0.0) or 0.0)
    status = (
        "pass" if success_rate >= threshold and scorecard.get("status") != "no_data" else "fail"
    )
    return " ".join(
        [
            f"status={status}",
            f"scorecard_status={scorecard.get('status', 'unknown')}",
            f"success_rate={success_rate:.3f}",
            f"threshold={threshold:.3f}",
            f"total_ticks={int(scorecard.get('total_ticks', 0) or 0)}",
            f"unique_issues_attempted={int(scorecard.get('unique_issues_attempted', 0) or 0)}",
            f"unique_issues_succeeded={int(scorecard.get('unique_issues_succeeded', 0) or 0)}",
            f"unique_issues_failed={int(scorecard.get('unique_issues_failed', 0) or 0)}",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="B0 Benchmark Scorecard")
    parser.add_argument(
        "--metrics",
        type=Path,
        default=DEFAULT_METRICS_PATH,
        help="Path to boss_metrics.jsonl",
    )
    parser.add_argument("--window", type=int, default=None, help="Last N ticks only")
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument("--json", action="store_true", help="JSON output")
    output_group.add_argument(
        "--ci",
        action="store_true",
        help="CI output: one-line summary and threshold-based exit status",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Minimum no-rescue success rate required in --ci mode (default: 0.5)",
    )
    args = parser.parse_args(argv)

    rows = load_metrics(args.metrics, args.window)
    scorecard = compute_scorecard(rows)

    if args.json:
        print(json.dumps(scorecard, indent=2))
    elif args.ci:
        print(render_ci_summary(scorecard, threshold=args.threshold))
        return (
            0
            if scorecard.get("status") != "no_data"
            and scorecard.get("no_rescue_success_rate", 0.0) >= args.threshold
            else 1
        )
    else:
        print_scorecard(scorecard)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
