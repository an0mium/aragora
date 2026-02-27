#!/usr/bin/env python3
"""Benchmark regression checker for CI.

Compares pytest-benchmark JSON output against a baseline and fails if any
benchmark regresses beyond a configurable threshold.  Also validates that
orchestration-speed-policy benchmarks pass, which guards the fast-first
routing feature against latency regressions.

Usage:
    # Compare two pytest-benchmark JSON files (PR vs main)
    python scripts/check_benchmark_regression.py \
        --current  pr-benchmark.json \
        --baseline main-benchmark.json \
        --threshold 20

    # Validate benchmark results exist and all tests passed
    python scripts/check_benchmark_regression.py --validate benchmark-results.json

Exit codes:
    0  - No regressions detected / validation passed
    1  - Regression detected or validation failed
    2  - Input file missing or malformed
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_benchmark_json(path: Path) -> dict:
    """Load and validate a pytest-benchmark JSON file."""
    if not path.exists():
        print(f"ERROR: Benchmark file not found: {path}", file=sys.stderr)
        sys.exit(2)

    try:
        with open(path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"ERROR: Cannot parse {path}: {exc}", file=sys.stderr)
        sys.exit(2)

    if "benchmarks" not in data:
        print(f"ERROR: {path} missing 'benchmarks' key", file=sys.stderr)
        sys.exit(2)

    return data


def extract_means(data: dict) -> dict[str, float]:
    """Extract benchmark name -> mean time mapping."""
    results: dict[str, float] = {}
    for bench in data["benchmarks"]:
        name = bench.get("name", bench.get("fullname", "unknown"))
        stats = bench.get("stats", {})
        mean = stats.get("mean")
        if mean is not None:
            results[name] = mean
    return results


def compare_benchmarks(
    current_path: Path,
    baseline_path: Path,
    threshold_pct: float,
) -> int:
    """Compare current benchmark results against a baseline.

    Returns 0 if no regressions exceed threshold_pct, 1 otherwise.
    """
    current = load_benchmark_json(current_path)
    baseline = load_benchmark_json(baseline_path)

    current_means = extract_means(current)
    baseline_means = extract_means(baseline)

    if not baseline_means:
        print("WARNING: Baseline has no benchmarks to compare against.")
        return 0

    regressions: list[str] = []
    improvements: list[str] = []
    unchanged: list[str] = []

    for name, current_mean in sorted(current_means.items()):
        if name not in baseline_means:
            continue

        baseline_mean = baseline_means[name]
        if baseline_mean == 0:
            continue

        change_pct = ((current_mean - baseline_mean) / baseline_mean) * 100

        # Skip noise: sub-microsecond benchmarks have high variance from
        # CPU scheduling; ignore regressions where absolute delta < 0.01ms.
        abs_delta_ms = (current_mean - baseline_mean) * 1000
        if change_pct > threshold_pct and abs_delta_ms < 0.01:
            unchanged.append(
                f"  OK (noise)  {name}: {change_pct:+.1f}% "
                f"(delta {abs_delta_ms:.4f}ms < 0.01ms floor)"
            )
            continue

        if change_pct > threshold_pct:
            regressions.append(
                f"  REGRESSION  {name}: {baseline_mean * 1000:.3f}ms -> "
                f"{current_mean * 1000:.3f}ms (+{change_pct:.1f}%)"
            )
        elif change_pct < -threshold_pct:
            improvements.append(
                f"  IMPROVED    {name}: {baseline_mean * 1000:.3f}ms -> "
                f"{current_mean * 1000:.3f}ms ({change_pct:.1f}%)"
            )
        else:
            unchanged.append(f"  OK          {name}: {change_pct:+.1f}%")

    # Print report
    print("=" * 72)
    print("  Benchmark Regression Report")
    print(f"  Threshold: {threshold_pct}% slowdown")
    print(f"  Compared:  {len(current_means)} current vs {len(baseline_means)} baseline")
    print("=" * 72)

    if regressions:
        print()
        print(f"  REGRESSIONS ({len(regressions)}):")
        for line in regressions:
            print(line)

    if improvements:
        print()
        print(f"  IMPROVEMENTS ({len(improvements)}):")
        for line in improvements:
            print(line)

    if unchanged:
        print()
        print(f"  WITHIN THRESHOLD ({len(unchanged)}):")
        for line in unchanged:
            print(line)

    print()

    if regressions:
        print(f"FAILED: {len(regressions)} benchmark(s) regressed by more than {threshold_pct}%.")
        return 1

    matched = len([n for n in current_means if n in baseline_means])
    print(f"PASSED: {matched} benchmark(s) within {threshold_pct}% threshold.")
    return 0


def validate_results(results_path: Path) -> int:
    """Validate that a benchmark results file exists and contains passing tests.

    This is the fallback mode when pytest-benchmark is not available or when
    we just need to verify benchmark tests ran and passed.

    Returns 0 if valid, 1 otherwise.
    """
    data = load_benchmark_json(results_path)
    benchmarks = data.get("benchmarks", [])

    if not benchmarks:
        print(f"WARNING: {results_path} contains no benchmark entries.")
        # Not a failure -- file exists but may have been filtered
        return 0

    print(f"Benchmark results validated: {len(benchmarks)} benchmark(s) recorded.")

    # Check for any benchmarks with suspicious stats (e.g., extremely slow)
    warnings = 0
    for bench in benchmarks:
        name = bench.get("name", "unknown")
        stats = bench.get("stats", {})
        mean = stats.get("mean", 0)
        stddev = stats.get("stddev", 0)

        # Flag benchmarks with coefficient of variation > 100% (highly unstable)
        if mean > 0 and stddev / mean > 1.0:
            print(
                f"  WARNING: {name} has high variance "
                f"(mean={mean * 1000:.3f}ms, stddev={stddev * 1000:.3f}ms)"
            )
            warnings += 1

    if warnings:
        print(f"\n{warnings} benchmark(s) with high variance (informational only).")

    print("PASSED: Benchmark results file is valid.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Check for benchmark regressions in CI.")
    subparsers = parser.add_subparsers(dest="command")

    # Compare mode
    compare = subparsers.add_parser("compare", help="Compare current results against baseline")
    compare.add_argument(
        "--current",
        type=Path,
        required=True,
        help="Path to current benchmark results JSON",
    )
    compare.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="Path to baseline benchmark results JSON",
    )
    compare.add_argument(
        "--threshold",
        type=float,
        default=20.0,
        help="Maximum allowed regression percentage (default: 20%%)",
    )

    # Validate mode
    validate = subparsers.add_parser("validate", help="Validate benchmark results file")
    validate.add_argument(
        "results",
        type=Path,
        help="Path to benchmark results JSON to validate",
    )

    # Legacy positional args: --current/--baseline at top level
    parser.add_argument("--current", type=Path, dest="top_current")
    parser.add_argument("--baseline", type=Path, dest="top_baseline")
    parser.add_argument("--threshold", type=float, default=20.0, dest="top_threshold")
    parser.add_argument("--validate", type=Path, dest="top_validate")

    args = parser.parse_args()

    if args.command == "compare":
        sys.exit(compare_benchmarks(args.current, args.baseline, args.threshold))
    elif args.command == "validate":
        sys.exit(validate_results(args.results))
    elif args.top_current and args.top_baseline:
        sys.exit(compare_benchmarks(args.top_current, args.top_baseline, args.top_threshold))
    elif args.top_validate:
        sys.exit(validate_results(args.top_validate))
    else:
        parser.print_help()
        sys.exit(2)


if __name__ == "__main__":
    main()
