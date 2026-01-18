"""
Performance Benchmark Suite for Aragora.

Benchmarks for:
- RLM compression efficiency
- Tenant isolation overhead
- Extended debate memory usage
- Connector parallel syncs

Usage:
    # Run all benchmarks
    python -m benchmarks

    # Run specific benchmark
    python -m benchmarks.rlm_compression
    python -m benchmarks.tenant_isolation
    python -m benchmarks.extended_debates
    python -m benchmarks.connector_parallel

Output includes:
- Execution time metrics (p50, p95, p99)
- Memory usage (peak, steady-state)
- Throughput (operations/second)
- Comparison with baseline
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run Aragora benchmarks")
    parser.add_argument(
        "--suite",
        choices=["all", "rlm", "tenant", "debate", "connector"],
        default="all",
        help="Benchmark suite to run",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file for results (JSON)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of iterations per benchmark",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Warmup iterations",
    )

    args = parser.parse_args()

    print(f"Aragora Benchmark Suite")
    print(f"======================")
    print(f"Suite: {args.suite}")
    print(f"Iterations: {args.iterations}")
    print(f"Warmup: {args.warmup}")
    print()

    results = {}

    if args.suite in ("all", "rlm"):
        from benchmarks.rlm_compression import run_rlm_benchmarks
        results["rlm"] = asyncio.run(run_rlm_benchmarks(args.iterations, args.warmup))

    if args.suite in ("all", "tenant"):
        from benchmarks.tenant_isolation import run_tenant_benchmarks
        results["tenant"] = asyncio.run(run_tenant_benchmarks(args.iterations, args.warmup))

    if args.suite in ("all", "debate"):
        from benchmarks.extended_debates import run_debate_benchmarks
        results["debate"] = asyncio.run(run_debate_benchmarks(args.iterations, args.warmup))

    if args.suite in ("all", "connector"):
        from benchmarks.connector_parallel import run_connector_benchmarks
        results["connector"] = asyncio.run(run_connector_benchmarks(args.iterations, args.warmup))

    # Add metadata
    results["_meta"] = {
        "timestamp": datetime.utcnow().isoformat(),
        "suite": args.suite,
        "iterations": args.iterations,
        "warmup": args.warmup,
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    else:
        print("\nResults:")
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
