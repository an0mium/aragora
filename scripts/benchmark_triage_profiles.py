#!/usr/bin/env python3
"""Benchmark inbox-triage baseline vs staged_v1 on a fixed fixture set."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from aragora.inbox.triage_profile_benchmark import render_benchmark_report, run_fixture_benchmark


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare baseline and staged_v1 inbox-triage profiles on a fixed message set.",
    )
    parser.add_argument(
        "--fixtures",
        type=Path,
        required=True,
        help="Path to a JSON array of fixture messages.",
    )
    parser.add_argument(
        "--diagnostics-dir",
        type=Path,
        help="Optional root directory for per-profile diagnostics artifacts.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write the full JSON benchmark report.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Mirror captured diagnostics inline while benchmarking.",
    )
    parser.add_argument(
        "--fail-on-thresholds",
        action="store_true",
        help="Exit 1 when the staged profile misses any acceptance threshold.",
    )
    return parser.parse_args(argv)


def _positive_int(value: Any, *, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def validate_report_shape(report: dict[str, Any]) -> None:
    comparison = report.get("comparison")
    profiles = report.get("profiles")
    if not isinstance(comparison, dict):
        raise ValueError("benchmark report comparison must be an object")
    if not isinstance(profiles, dict):
        raise ValueError("benchmark report profiles must be an object")

    message_count = _positive_int(
        comparison.get("message_count"),
        label="comparison.message_count",
    )
    for profile in ("baseline", "staged_v1"):
        profile_payload = profiles.get(profile)
        if not isinstance(profile_payload, dict):
            raise ValueError(f"profiles.{profile} must be an object")
        profile_count = _positive_int(
            profile_payload.get("message_count"),
            label=f"profiles.{profile}.message_count",
        )
        if profile_count != message_count:
            raise ValueError(
                f"profiles.{profile}.message_count ({profile_count}) must match "
                f"comparison.message_count ({message_count})"
            )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = asyncio.run(
        run_fixture_benchmark(
            args.fixtures,
            diagnostics_root=args.diagnostics_dir,
            verbose=args.verbose,
        )
    )
    try:
        validate_report_shape(report)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(render_benchmark_report(report))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nReport written to {args.output}")

    if args.fail_on_thresholds and not report["comparison"]["passes_all_thresholds"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
