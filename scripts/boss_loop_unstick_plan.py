#!/usr/bin/env python3
"""Boss-loop ``unstick`` planning CLI (dry-run only).

Reads two JSON files (the shape returned by ``gh issue list`` and
``gh pr list``) and writes an ``UnstickRecommendation`` plan to
stdout (JSON or markdown) for operator review.

This script performs **no** GitHub mutations.

Example
-------

::

    gh issue list --state open --label boss-stuck --limit 200 \\
        --json number,state,labels > /tmp/stuck-issues.json
    gh pr list --state all --limit 500 \\
        --search 'head:aragora/boss-harvest/' \\
        --json number,state,headRefName > /tmp/pr-records.json

    python3 scripts/boss_loop_unstick_plan.py \\
        --issues /tmp/stuck-issues.json \\
        --pr-records /tmp/pr-records.json \\
        --json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from aragora.swarm.unstick import (  # noqa: E402
    plan_unstick,
    render_markdown,
    summarize_plan,
)


def _load_json_list(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--issues",
        type=Path,
        required=True,
        help="JSON file with `gh issue list ... --json number,state,labels` output",
    )
    parser.add_argument(
        "--pr-records",
        type=Path,
        required=True,
        help="JSON file with `gh pr list ... --json number,state,headRefName` output",
    )
    parser.add_argument(
        "--format",
        choices=("json", "md"),
        default="json",
        help="Output format (default: json)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path; default writes to stdout",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    issues = _load_json_list(args.issues)
    pr_records = _load_json_list(args.pr_records)
    plan = plan_unstick(stuck_issue_records=issues, pr_records=pr_records)
    if args.format == "md":
        text = render_markdown(plan)
    else:
        payload = {
            "summary": summarize_plan(plan),
            "recommendations": [asdict(r) for r in plan],
        }
        text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + ("\n" if not text.endswith("\n") else ""))
        print(str(args.output))
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
