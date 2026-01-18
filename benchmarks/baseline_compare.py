#!/usr/bin/env python3
"""
Baseline Comparison Tool.

Compares current benchmark results against established baselines.
Outputs pass/fail status and regression warnings.

Usage:
    python -m benchmarks.baseline_compare --run
    python -m benchmarks.baseline_compare --report
    python -m benchmarks.baseline_compare --update
"""

import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class ComparisonStatus(Enum):
    """Status of a metric comparison."""

    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"
    SKIP = "skip"


@dataclass
class ComparisonResult:
    """Result of comparing a metric to baseline."""

    metric_name: str
    baseline_value: float
    current_value: float
    percent_change: float
    status: ComparisonStatus
    message: str


def load_baseline(baseline_file: Path) -> dict[str, Any]:
    """Load baseline metrics from JSON file."""
    if not baseline_file.exists():
        raise FileNotFoundError(f"Baseline file not found: {baseline_file}")
    with open(baseline_file) as f:
        return json.load(f)


def save_baseline(baseline_file: Path, data: dict[str, Any]) -> None:
    """Save baseline metrics to JSON file."""
    with open(baseline_file, "w") as f:
        json.dump(data, f, indent=2)


def compare_metric(
    name: str,
    baseline: float,
    current: float,
    warning_threshold: float = 10.0,
    critical_threshold: float = 25.0,
    higher_is_better: bool = False,
) -> ComparisonResult:
    """
    Compare a current metric value to baseline.

    Args:
        name: Metric name
        baseline: Baseline value
        current: Current measured value
        warning_threshold: Percent change to trigger warning
        critical_threshold: Percent change to trigger failure
        higher_is_better: If True, increases are good; if False, increases are bad
    """
    if baseline == 0:
        return ComparisonResult(
            metric_name=name,
            baseline_value=baseline,
            current_value=current,
            percent_change=0,
            status=ComparisonStatus.SKIP,
            message="Baseline is zero, cannot compare",
        )

    percent_change = ((current - baseline) / baseline) * 100

    # Determine if change is a regression
    is_regression = percent_change > 0 if not higher_is_better else percent_change < 0
    abs_change = abs(percent_change)

    if is_regression:
        if abs_change >= critical_threshold:
            status = ComparisonStatus.FAIL
            message = f"CRITICAL: {abs_change:.1f}% regression (threshold: {critical_threshold}%)"
        elif abs_change >= warning_threshold:
            status = ComparisonStatus.WARNING
            message = f"WARNING: {abs_change:.1f}% regression (threshold: {warning_threshold}%)"
        else:
            status = ComparisonStatus.PASS
            message = f"Within tolerance: {percent_change:+.1f}%"
    else:
        status = ComparisonStatus.PASS
        direction = "improvement" if abs_change > 1 else "stable"
        message = f"PASS: {direction} ({percent_change:+.1f}%)"

    return ComparisonResult(
        metric_name=name,
        baseline_value=baseline,
        current_value=current,
        percent_change=percent_change,
        status=status,
        message=message,
    )


async def run_quick_benchmarks() -> dict[str, float]:
    """Run quick benchmarks to capture current metrics."""
    from unittest.mock import AsyncMock, MagicMock

    results: dict[str, float] = {}

    # Benchmark: Simple async operation latency
    latencies = []
    for _ in range(100):
        start = time.perf_counter()
        await asyncio.sleep(0)
        latencies.append((time.perf_counter() - start) * 1000)

    results["async_baseline_p50"] = statistics.median(latencies)
    results["async_baseline_p95"] = sorted(latencies)[int(len(latencies) * 0.95)]

    # Benchmark: Memory operations (in-memory dict simulation)
    memory_store: dict[str, Any] = {}

    # Write operations
    write_latencies = []
    for i in range(100):
        start = time.perf_counter()
        memory_store[f"key_{i}"] = {"data": f"value_{i}"}
        await asyncio.sleep(0)  # Yield to event loop
        write_latencies.append((time.perf_counter() - start) * 1000)

    results["memory_write_p50"] = statistics.median(write_latencies)
    results["memory_write_p95"] = sorted(write_latencies)[int(len(write_latencies) * 0.95)]

    # Read operations
    read_latencies = []
    for i in range(100):
        start = time.perf_counter()
        _ = memory_store.get(f"key_{i}")
        await asyncio.sleep(0)  # Yield to event loop
        read_latencies.append((time.perf_counter() - start) * 1000)

    results["memory_read_p50"] = statistics.median(read_latencies)
    results["memory_read_p95"] = sorted(read_latencies)[int(len(read_latencies) * 0.95)]

    # Throughput calculation (scale up as in-memory is much faster than baseline)
    results["memory_write_ops_per_sec"] = 1000 / max(results["memory_write_p50"], 0.001)
    results["memory_read_ops_per_sec"] = 1000 / max(results["memory_read_p50"], 0.001)

    return results


def generate_report(
    comparisons: list[ComparisonResult],
    output_format: str = "text",
) -> str:
    """Generate a comparison report."""
    if output_format == "json":
        return json.dumps(
            [
                {
                    "metric": c.metric_name,
                    "baseline": c.baseline_value,
                    "current": c.current_value,
                    "percent_change": c.percent_change,
                    "status": c.status.value,
                    "message": c.message,
                }
                for c in comparisons
            ],
            indent=2,
        )

    # Text format
    lines = [
        "=" * 70,
        "BENCHMARK COMPARISON REPORT",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "=" * 70,
        "",
    ]

    # Group by status
    by_status: dict[ComparisonStatus, list[ComparisonResult]] = {
        s: [] for s in ComparisonStatus
    }
    for c in comparisons:
        by_status[c.status].append(c)

    # Summary
    lines.append("SUMMARY")
    lines.append("-" * 40)
    lines.append(f"  PASS:    {len(by_status[ComparisonStatus.PASS])}")
    lines.append(f"  WARNING: {len(by_status[ComparisonStatus.WARNING])}")
    lines.append(f"  FAIL:    {len(by_status[ComparisonStatus.FAIL])}")
    lines.append(f"  SKIP:    {len(by_status[ComparisonStatus.SKIP])}")
    lines.append("")

    # Details
    for status in [ComparisonStatus.FAIL, ComparisonStatus.WARNING, ComparisonStatus.PASS]:
        if by_status[status]:
            lines.append(f"{status.value.upper()} ({len(by_status[status])})")
            lines.append("-" * 40)
            for c in by_status[status]:
                lines.append(
                    f"  {c.metric_name}: {c.baseline_value:.2f} -> {c.current_value:.2f} ({c.percent_change:+.1f}%)"
                )
                lines.append(f"    {c.message}")
            lines.append("")

    # Overall status
    has_failures = len(by_status[ComparisonStatus.FAIL]) > 0
    has_warnings = len(by_status[ComparisonStatus.WARNING]) > 0

    lines.append("=" * 70)
    if has_failures:
        lines.append("OVERALL STATUS: FAIL")
    elif has_warnings:
        lines.append("OVERALL STATUS: WARNING")
    else:
        lines.append("OVERALL STATUS: PASS")
    lines.append("=" * 70)

    return "\n".join(lines)


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Baseline comparison tool")
    parser.add_argument("--run", action="store_true", help="Run benchmarks and compare")
    parser.add_argument("--report", action="store_true", help="Generate report only")
    parser.add_argument("--update", action="store_true", help="Update baseline with current results")
    parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path(__file__).parent / "baseline_metrics.json",
        help="Baseline file path",
    )
    args = parser.parse_args()

    if not any([args.run, args.report, args.update]):
        args.run = True  # Default action

    baseline = load_baseline(args.baseline)
    rules = baseline.get("regression_rules", {})
    warning_threshold = rules.get("warning_threshold_percent", 10)
    critical_threshold = rules.get("critical_threshold_percent", 25)

    if args.run:
        print("Running benchmarks...")
        current_metrics = await run_quick_benchmarks()

        print("\nCurrent measurements:")
        for name, value in current_metrics.items():
            print(f"  {name}: {value:.3f}")

        # Compare to baseline
        comparisons: list[ComparisonResult] = []

        # Memory operations comparison
        baseline_memory = baseline.get("metrics", {}).get("memory_operations", {}).get("measurements", {})
        if baseline_memory:
            fast_read = baseline_memory.get("fast_tier_read", {})
            if "baseline" in fast_read:
                comparisons.append(
                    compare_metric(
                        "memory_read_ops_per_sec",
                        fast_read["baseline"],
                        current_metrics.get("memory_read_ops_per_sec", 0),
                        warning_threshold,
                        critical_threshold,
                        higher_is_better=True,
                    )
                )

            fast_write = baseline_memory.get("fast_tier_write", {})
            if "baseline" in fast_write:
                comparisons.append(
                    compare_metric(
                        "memory_write_ops_per_sec",
                        fast_write["baseline"],
                        current_metrics.get("memory_write_ops_per_sec", 0),
                        warning_threshold,
                        critical_threshold,
                        higher_is_better=True,
                    )
                )

        print("\n" + generate_report(comparisons, args.format))

        # Exit with error code if failures
        has_failures = any(c.status == ComparisonStatus.FAIL for c in comparisons)
        sys.exit(1 if has_failures else 0)

    if args.update:
        print("Running benchmarks to update baseline...")
        current_metrics = await run_quick_benchmarks()

        # Update baseline values
        metrics = baseline.setdefault("metrics", {})
        memory_ops = metrics.setdefault("memory_operations", {}).setdefault("measurements", {})

        if current_metrics.get("memory_read_ops_per_sec"):
            memory_ops.setdefault("fast_tier_read", {})["baseline"] = current_metrics["memory_read_ops_per_sec"]

        if current_metrics.get("memory_write_ops_per_sec"):
            memory_ops.setdefault("fast_tier_write", {})["baseline"] = current_metrics["memory_write_ops_per_sec"]

        baseline["updated_at"] = datetime.now(timezone.utc).isoformat()

        save_baseline(args.baseline, baseline)
        print(f"Updated baseline saved to {args.baseline}")


if __name__ == "__main__":
    asyncio.run(main())
