#!/usr/bin/env python3
"""Evaluate the PR-scoped zero-coverage gate.

The zero-coverage workflow currently has two distinct signal sources:

1. ``scripts/nomic_ci_test_selector.py`` decides whether a PR changed any
   Python modules and whether those modules map to tests.
2. ``pytest --cov`` emits ``cov.json`` for the selected tests.

This script joins those signals into one explicit decision:

- selector status ``skipped`` is the only case where missing coverage is OK
- selector status ``unmapped_python_changes`` is a hard failure
- missing or invalid ``cov.json`` is a hard failure in every non-skipped case
- newly zero-covered files not present in the baseline are a hard failure
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_baseline(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def _new_zero_coverage_files(coverage: dict[str, Any], baseline: set[str]) -> list[str]:
    zero_cov: list[str] = []
    for file, info in coverage.get("files", {}).items():
        percent = info.get("summary", {}).get("percent_covered", 100)
        if percent == 0 and file not in baseline:
            zero_cov.append(file)
    return sorted(zero_cov)


def evaluate_zero_coverage_gate(
    *,
    coverage_path: Path,
    baseline_path: Path,
    selector_status: str,
    changed_python_count: int,
) -> tuple[bool, list[str]]:
    if selector_status == "skipped":
        return True, ["No changed Python files required PR-scoped zero-coverage probing; skipping."]

    if selector_status == "unmapped_python_changes":
        count_msg = (
            f" for {changed_python_count} changed Python file(s)"
            if changed_python_count > 0
            else ""
        )
        return False, [
            f"::error::PR-scoped coverage selector reported unmapped Python changes{count_msg}",
            "::error::Changed Python modules without mapped tests must fail the zero-coverage gate",
        ]

    if not coverage_path.exists():
        return False, [
            f"::error::Coverage data not available at {coverage_path}; zero-coverage gate cannot verify this change",
        ]

    try:
        coverage = json.loads(coverage_path.read_text())
    except json.JSONDecodeError:
        return False, [
            f"::error::Coverage data at {coverage_path} is invalid JSON; zero-coverage gate cannot verify this change",
        ]

    baseline = _load_baseline(baseline_path)
    zero_cov = _new_zero_coverage_files(coverage, baseline)
    if zero_cov:
        messages = ["::error::New zero-coverage files detected:"]
        messages.extend(f"  - {path}" for path in zero_cov[:20])
        if len(zero_cov) > 20:
            messages.append(f"  ... and {len(zero_cov) - 20} more")
        messages.append("Add to tests/.zero_coverage_baseline if intentional")
        return False, messages

    return True, ["No new zero-coverage files detected"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--coverage-path",
        type=Path,
        default=Path("cov.json"),
        help="Path to the JSON coverage report emitted by pytest-cov.",
    )
    parser.add_argument(
        "--baseline-path",
        type=Path,
        default=Path("tests/.zero_coverage_baseline"),
        help="Path to the committed zero-coverage baseline file.",
    )
    parser.add_argument(
        "--selector-status",
        default="",
        help="Status emitted by scripts/nomic_ci_test_selector.py.",
    )
    parser.add_argument(
        "--changed-python-count",
        type=int,
        default=0,
        help="Count of changed Python files from the selector result.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ok, messages = evaluate_zero_coverage_gate(
        coverage_path=args.coverage_path,
        baseline_path=args.baseline_path,
        selector_status=args.selector_status,
        changed_python_count=args.changed_python_count,
    )
    for message in messages:
        print(message)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
