#!/usr/bin/env python3
"""TW-03: Convert repeated rescue classes into boss-ready substrate tickets.

Reads the RescueEvent ledger AND boss_metrics.jsonl to identify repeated
failure patterns, then creates GitHub issues that fix the root cause.

Usage:
  python3 scripts/rescue_to_fixtures.py                    # report only
  python3 scripts/rescue_to_fixtures.py --create-issues    # create GitHub issues
  python3 scripts/rescue_to_fixtures.py --json             # machine-readable
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METRICS_PATH = REPO_ROOT / ".aragora" / "overnight" / "boss_metrics.jsonl"
THRESHOLD = 3  # minimum occurrences to be considered "repeated"


def load_failure_patterns(
    metrics_path: Path = DEFAULT_METRICS_PATH,
    window: int = 200,
) -> list[dict[str, Any]]:
    """Identify repeated failure patterns from boss metrics."""
    if not metrics_path.exists():
        return []

    rows: list[dict[str, Any]] = []
    for line in metrics_path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    rows = rows[-window:]

    # Group failures by terminal class + failure pattern
    pattern_counter: Counter[str] = Counter()
    pattern_examples: dict[str, list[dict[str, Any]]] = {}

    for row in rows:
        tc = str(row.get("terminal_class", "")).strip()
        if not tc or tc.startswith("success") or tc.startswith("deliverable"):
            continue

        # Build a pattern key from terminal class + sanitizer outcome
        sanitizer = str(row.get("sanitizer_outcome", "")).strip()
        pattern_key = f"{tc}"
        if sanitizer and sanitizer != "accepted":
            pattern_key = f"{tc}:{sanitizer}"

        pattern_counter[pattern_key] += 1
        examples = pattern_examples.setdefault(pattern_key, [])
        if len(examples) < 5:
            examples.append(
                {
                    "issue_number": row.get("issue_number"),
                    "issue_title": str(row.get("issue_title", ""))[:80],
                    "elapsed_seconds": row.get("elapsed_seconds"),
                }
            )

    # Filter to repeated patterns
    repeated = []
    for pattern, count in pattern_counter.most_common():
        if count < THRESHOLD:
            continue
        repeated.append(
            {
                "pattern": pattern,
                "count": count,
                "percentage": round(count / len(rows) * 100, 1) if rows else 0,
                "examples": pattern_examples.get(pattern, []),
                "suggested_fix": _suggest_fix(pattern),
            }
        )

    return repeated


def _suggest_fix(pattern: str) -> str:
    """Generate a human-readable fix suggestion for a failure pattern."""
    fixes = {
        "blocked_not_dispatch_bounded": "Improve issue upgrader to add missing file scope and acceptance criteria",
        "blocked_sanitation_failed": "Tighten sanitizer false positive rules or add RescuePlanner LLM override",
        "blocked_sanitation_failed:quarantined": "Fix sanitizer contradictory_scope false positive pattern",
        "blocked_sanitation_failed:dropped": "Improve issue body generation to include validation contracts",
        "blocked_validation_target_missing": "Update issue generator to check validation target existence",
        "rescue_no_deliverable": "Wire Conductor retry recommendations into boss loop retry path",
        "rescue_timeout": "Increase worker timeout or decompose into smaller tasks",
        "rescue_worker_crash": "Add worker crash recovery with session state resume",
        "rescue_verification_failed": "Improve worker prompt to include verification commands",
    }
    for key, fix in fixes.items():
        if pattern.startswith(key):
            return fix
    return "Investigate and create a substrate fix for this failure pattern"


def render_report(patterns: list[dict[str, Any]]) -> str:
    """Render a human-readable report of repeated failure patterns."""
    if not patterns:
        return "No repeated failure patterns found above threshold."

    lines = [
        "=" * 60,
        "  TW-03: REPEATED RESCUE CLASS REPORT",
        "=" * 60,
        "",
    ]
    for p in patterns:
        lines.append(f"  {p['count']}x ({p['percentage']}%) — {p['pattern']}")
        lines.append(f"     Fix: {p['suggested_fix']}")
        for ex in p["examples"][:3]:
            lines.append(f"     e.g. #{ex['issue_number']} {ex['issue_title']}")
        lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)


def create_substrate_issues(
    patterns: list[dict[str, Any]],
    repo: str = "synaptent/aragora",
    dry_run: bool = False,
) -> list[str]:
    """Create boss-ready GitHub issues for each repeated failure pattern."""
    created: list[str] = []

    for p in patterns:
        title = f"[TW-03] Fix repeated failure: {p['pattern']} ({p['count']}x)"
        examples_text = "\n".join(
            f"- #{ex['issue_number']} {ex['issue_title']}" for ex in p["examples"][:5]
        )
        body = f"""## Goal
Fix the repeated failure pattern `{p["pattern"]}` which accounts for {p["percentage"]}% of recent boss loop failures.

## Suggested Fix
{p["suggested_fix"]}

## Evidence
This pattern appeared {p["count"]} times in the last 200 boss loop ticks.

### Example issues
{examples_text}

## Validation
python3 scripts/measure_b0_scorecard.py
# Success rate should improve after fix
"""
        if dry_run:
            created.append(f"DRY-RUN: would create '{title}'")
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
                    title,
                    "--label",
                    "boss-ready",
                    "--body",
                    body,
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            if result.returncode == 0:
                url = result.stdout.strip()
                created.append(url)
            else:
                created.append(f"FAILED: {result.stderr.strip()[:100]}")
        except (subprocess.TimeoutExpired, OSError) as exc:
            created.append(f"ERROR: {exc}")

    return created


def main() -> int:
    parser = argparse.ArgumentParser(description="TW-03: Rescue to fixture conversion")
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--window", type=int, default=200)
    parser.add_argument("--threshold", type=int, default=THRESHOLD)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--create-issues", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--repo", default="synaptent/aragora")
    args = parser.parse_args()

    patterns = load_failure_patterns(args.metrics, args.window)

    if args.json:
        print(json.dumps(patterns, indent=2))
    else:
        print(render_report(patterns))

    if args.create_issues or args.dry_run:
        results = create_substrate_issues(patterns, repo=args.repo, dry_run=args.dry_run)
        for r in results:
            print(f"  {r}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
