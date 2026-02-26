#!/usr/bin/env python3
"""Generate a single-pane fleet coordination status report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aragora.coordination.fleet import create_fleet_coordinator


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate fleet status report for active sessions."
    )
    parser.add_argument(
        "--repo", default=".", help="Path inside repository (default: current directory)"
    )
    parser.add_argument(
        "--tail",
        type=int,
        default=500,
        help="Tail N lines from each session log (default: 500)",
    )
    parser.add_argument(
        "--output-dir",
        default="docs/status",
        help="Output directory for report markdown (default: docs/status)",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (repo / output_dir).resolve()

    fleet = create_fleet_coordinator(repo_root=repo)
    status = fleet.fleet_status(tail_lines=max(1, int(args.tail)))
    report = fleet.write_report(status=status, output_dir=output_dir)

    payload = {
        "ok": True,
        "generated_at": status.get("generated_at"),
        "report_path": report.get("report_path"),
        "active_sessions": status.get("active_sessions", 0),
        "total_sessions": status.get("total_sessions", 0),
        "claim_conflicts": status.get("claims", {}).get("conflict_count", 0),
        "merge_queue_items": status.get("merge_queue", {}).get("total", 0),
        "actionable_failures": len(status.get("actionable_failures", [])),
    }

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"Report written: {payload['report_path']}")
        print(
            "Summary: "
            f"sessions={payload['active_sessions']}/{payload['total_sessions']} "
            f"claims={payload['claim_conflicts']} queue={payload['merge_queue_items']} "
            f"failures={payload['actionable_failures']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
