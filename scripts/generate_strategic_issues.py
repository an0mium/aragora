#!/usr/bin/env python3
"""Generate strategic boss-ready issue candidates.

This CLI is a dry-run preview tool. It never creates GitHub issues.
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aragora.swarm.strategic_issue_bridge import (
    StrategicIssueBridge,
    StrategicIssueBridgeConfig,
    dump_candidates_json,
    format_candidates_markdown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate strategic issue candidates")
    parser.add_argument("--repo", default=".", help="Repository root")
    parser.add_argument("--max-issues", type=int, default=10)
    parser.add_argument("--max-per-theme", type=int, default=4)
    parser.add_argument("--categories", nargs="*", default=[], help="Filter by theme/category")
    parser.add_argument("--dry-run", action="store_true", help="Preview only")
    parser.add_argument("--heuristic-only", action="store_true")
    parser.add_argument("--no-scanner", action="store_true")
    parser.add_argument("--enable-llm", action="store_true")
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo).resolve()

    config = StrategicIssueBridgeConfig(
        max_issues=args.max_issues,
        max_per_theme=args.max_per_theme,
        categories=list(args.categories),
        heuristic_only=args.heuristic_only,
        enable_scanner=not args.no_scanner,
        enable_llm=args.enable_llm,
    )

    bridge = StrategicIssueBridge(repo_root=repo_root, config=config)
    candidates = bridge.generate_candidates()

    if args.json:
        print(dump_candidates_json(candidates))
    else:
        header = f"Strategic issue candidates: {len(candidates)}"
        print(header)
        print("" if args.dry_run else "")
        print(format_candidates_markdown(candidates))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
