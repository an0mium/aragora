#!/usr/bin/env python3
"""Score terminal-truth benchmark fixtures against classify_from_metrics().

RS-03 scoring lane — loads JSON fixture files produced by RS-02, runs
``classify_from_metrics()`` on every example, and reports per-file and
aggregate pass/fail results.

Exit codes:
    0  — all examples classified correctly
    1  — one or more mismatches detected
    2  — fixtures directory missing or empty

Usage:
    python scripts/score_benchmark.py
    python scripts/score_benchmark.py --fixtures-dir path/to/fixtures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure the local repo takes precedence over any editable install so that
# worktree-local changes to aragora are picked up when the script is invoked
# directly (sys.path[0] defaults to the *scripts/* directory, not the repo root).
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from aragora.swarm.terminal_truth import classify_from_metrics  # noqa: E402

DEFAULT_FIXTURES_DIR = REPO_ROOT / "benchmarks" / "fixtures" / "swarm" / "terminal_truth"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score terminal-truth benchmark fixtures against classify_from_metrics().",
    )
    parser.add_argument(
        "--fixtures-dir",
        type=Path,
        default=DEFAULT_FIXTURES_DIR,
        help=(
            f"Path to directory containing .json fixture files (default: {DEFAULT_FIXTURES_DIR})"
        ),
    )
    return parser


def _is_str(value: object) -> bool:
    return isinstance(value, str)


def _is_numeric(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_row(row: object) -> list[str]:
    if not isinstance(row, dict):
        return ["row must be a JSON object"]

    validators = {
        "worker_status": ("str", _is_str),
        "worker_outcome": ("str", _is_str),
        "elapsed_seconds": ("int|float", _is_numeric),
        "files_changed": ("int", _is_int),
        "has_deliverable": ("bool", lambda value: isinstance(value, bool)),
        "publish_action": ("str", _is_str),
        "expected_class": ("str", _is_str),
    }
    errors: list[str] = []
    for key, (expected_type, validator) in validators.items():
        if key not in row:
            errors.append(f"missing required key {key!r}")
            continue
        if not validator(row[key]):
            errors.append(f"invalid {key!r}: expected {expected_type}")
    return errors


def score_fixtures(fixtures_dir: Path) -> tuple[bool, str]:
    """Load and score all fixture files.

    Returns ``(all_passed, report_text)``.
    """
    if not fixtures_dir.is_dir():
        return False, f"ERROR: fixtures directory does not exist: {fixtures_dir}"

    fixture_files = sorted(fixtures_dir.glob("*.json"))
    if not fixture_files:
        return False, f"ERROR: no .json files found in {fixtures_dir}"

    total_examples = 0
    total_pass = 0
    total_fail = 0
    lines: list[str] = []

    for fixture_file in fixture_files:
        with fixture_file.open() as fh:
            examples = json.load(fh)

        file_pass = 0
        file_fail = 0
        file_errors: list[str] = []

        if not isinstance(examples, list):
            total_fail += 1
            lines.append(f"  FAIL  {fixture_file.name} (0/1)")
            lines.append("  [root] fixture root must be a JSON array")
            continue

        for idx, row in enumerate(examples):
            row_errors = _validate_row(row)
            if row_errors:
                file_fail += 1
                for err in row_errors:
                    file_errors.append(f"  [{idx}] schema error: {err}")
                continue

            expected = row.get("expected_class", "")
            result = classify_from_metrics(row)
            if result.value == expected:
                file_pass += 1
            else:
                file_fail += 1
                file_errors.append(f"  [{idx}] expected {expected!r}, got {result.value!r}")

        total_examples += file_pass + file_fail
        total_pass += file_pass
        total_fail += file_fail

        status = "PASS" if file_fail == 0 else "FAIL"
        lines.append(f"  {status}  {fixture_file.name} ({file_pass}/{file_pass + file_fail})")
        for err in file_errors:
            lines.append(err)

    # Aggregate summary
    all_passed = total_fail == 0
    summary_status = "PASS" if all_passed else "FAIL"
    header = (
        f"Terminal-truth benchmark: {summary_status}\n"
        f"Files: {len(fixture_files)}  |  "
        f"Examples: {total_examples}  |  "
        f"Pass: {total_pass}  |  Fail: {total_fail}\n"
    )
    report = header + "\n".join(lines)
    return all_passed, report


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    fixtures_dir: Path = args.fixtures_dir

    all_passed, report = score_fixtures(fixtures_dir)
    print(report)

    if report.startswith("ERROR:"):
        return 2

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
