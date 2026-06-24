#!/usr/bin/env python3
"""Classify active Aragora automation handoffs without mutating state."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from handoff_state import (  # noqa: E402
    DEFAULT_REPO_ROOT,
    classify_handoffs,
    compact_summary,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        type=Path,
        default=DEFAULT_REPO_ROOT,
        help="Repository root used for local state and gh cwd (default: current directory).",
    )
    parser.add_argument(
        "--state-root",
        type=Path,
        default=None,
        help=(
            "Checkout or .aragora directory that owns shared automation state. "
            "Defaults to --repo/.aragora."
        ),
    )
    parser.add_argument(
        "--github-repo",
        default=None,
        help="GitHub repository in owner/name form. Defaults to remote.origin.url.",
    )
    parser.add_argument(
        "--outbox-file",
        default=None,
        help="Classify only this outbox JSON file. Relative names resolve inside automation-outbox.",
    )
    parser.add_argument(
        "--no-github",
        action="store_true",
        help="Disable narrow GitHub REST/ref reads and classify from local state only.",
    )
    parser.add_argument(
        "--owner-timeout-seconds",
        type=int,
        default=20,
        help="Timeout for the read-only owner/liveness helper per branch (default: 20).",
    )
    parser.add_argument(
        "--with-liveness-helper",
        action="store_true",
        help=(
            "Use identify_lane_owner.py --liveness for each item. Default uses the "
            "faster local lane registry to keep whole-outbox classification bounded."
        ),
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON.")
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="With --json, omit per-item details and emit compact counts only.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    payload = classify_handoffs(
        repo_root=args.repo,
        state_root=args.state_root,
        github_repo=args.github_repo,
        outbox_file=args.outbox_file,
        no_github=args.no_github,
        owner_timeout_seconds=args.owner_timeout_seconds,
        with_liveness_helper=args.with_liveness_helper,
    )
    output = compact_summary(payload) if args.summary_only else payload
    if args.json:
        print(json.dumps(output, indent=2, sort_keys=True))
        return 0

    print(f"schema_version: {output['schema_version']}")
    print(f"generated_at: {output['generated_at']}")
    print(f"repo: {output['repo']}")
    print(f"outbox_count: {output['outbox_count']}")
    print("counts:")
    for state, count in sorted((output.get("counts") or {}).items()):
        print(f"  {state}: {count}")
    if not args.summary_only:
        print("items:")
        for item in output.get("items", []):
            print(f"  {item.get('outbox_file')}: {item.get('state')} ({item.get('reason')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
