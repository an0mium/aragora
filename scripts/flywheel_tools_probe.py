#!/usr/bin/env python3
"""Probe optional local Agent Flywheel-style tools.

The probe is read-only: it never installs tools, mutates shell profiles, starts
tmux sessions, launches agents, or calls model APIs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aragora.integrations.flywheel import probe_flywheel_tools, summarize_probe


def _json_payload(*, include_help: bool, timeout_seconds: float) -> dict[str, Any]:
    statuses = probe_flywheel_tools(include_help=include_help, timeout_seconds=timeout_seconds)
    return {
        "schema_version": "flywheel_tools_probe.v1",
        "mode": "read_only_local_probe",
        "summary": summarize_probe(statuses),
        "tools": [status.to_dict() for status in statuses],
        "non_claims": [
            "no tools were installed",
            "no shell profiles were modified",
            "no model calls were made",
            "no GitHub runner or AWS environment was touched",
        ],
    }


def _print_human(payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    print("Flywheel local tool probe")
    print(f"available: {summary['available_count']} / {summary['tool_count']}")
    for tool in payload["tools"]:
        marker = "available" if tool["available"] else "missing"
        source = tool["matched_command"] or ", ".join(tool["marker_paths_found"]) or "-"
        print(f"- {tool['name']}: {marker} ({source})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    parser.add_argument(
        "--no-help",
        action="store_true",
        help="Skip --help probes and only run lightweight availability/version checks",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=2.0,
        help="Per-command timeout for version/help probes",
    )
    args = parser.parse_args(argv)

    payload = _json_payload(include_help=not args.no_help, timeout_seconds=args.timeout_seconds)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _print_human(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
