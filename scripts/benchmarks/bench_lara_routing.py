#!/usr/bin/env python3
"""
Benchmark LaRA routing decisions.

Usage:
    python scripts/benchmarks/bench_lara_routing.py --iterations 200
"""

from __future__ import annotations

import argparse
import random
import statistics
import time
from dataclasses import dataclass, field

from aragora.knowledge.mound.api.router import DocumentFeatures, LaRARouter


@dataclass
class RoutingBenchmarkResult:
    iterations: int
    route_counts: dict[str, int] = field(default_factory=dict)
    times_ms: list[float] = field(default_factory=list)

    @property
    def mean_ms(self) -> float:
        return statistics.mean(self.times_ms) if self.times_ms else 0.0

    @property
    def median_ms(self) -> float:
        return statistics.median(self.times_ms) if self.times_ms else 0.0


def sample_queries() -> list[str]:
    return [
        "graph:node_123",
        "node:fact_456",
        "summarize contract obligations",
        "compare agent approaches to calibration",
        "timeline of changes",
        "short query",
        "owner id mapping",
        "id:knowledge_789",
        "why did the debate stop early",
        "explain late-stage fragility",
    ]


def _require_positive_int(value: int, *, label: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")


def _require_non_negative_int(value: int, *, label: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")


def _validate_benchmark_inputs(iterations: int, min_nodes: int, max_nodes: int) -> None:
    _require_positive_int(iterations, label="iterations")
    _require_non_negative_int(min_nodes, label="min_nodes")
    _require_non_negative_int(max_nodes, label="max_nodes")
    if min_nodes > max_nodes:
        raise ValueError("min_nodes must be less than or equal to max_nodes")


def run_benchmark(iterations: int, min_nodes: int, max_nodes: int) -> RoutingBenchmarkResult:
    _validate_benchmark_inputs(iterations, min_nodes, max_nodes)

    rng = random.Random(42)
    router = LaRARouter()
    queries = sample_queries()
    result = RoutingBenchmarkResult(iterations=iterations)

    for _ in range(iterations):
        query = rng.choice(queries)
        total_nodes = rng.randint(min_nodes, max_nodes)
        start = time.perf_counter()
        decision = router.route(
            query,
            DocumentFeatures(total_nodes=total_nodes),
            supports_rlm=True,
        )
        elapsed = (time.perf_counter() - start) * 1000
        result.times_ms.append(elapsed)
        result.route_counts[decision.route] = result.route_counts.get(decision.route, 0) + 1

    return result


def _positive_int(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if value <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _non_negative_int(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a non-negative integer") from exc
    if value < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark LaRA routing decisions")
    parser.add_argument("--iterations", type=_positive_int, default=200)
    parser.add_argument("--min-nodes", type=_non_negative_int, default=0)
    parser.add_argument("--max-nodes", type=_non_negative_int, default=5000)
    args = parser.parse_args(argv)
    if args.min_nodes > args.max_nodes:
        parser.error("--min-nodes must be less than or equal to --max-nodes")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = run_benchmark(args.iterations, args.min_nodes, args.max_nodes)

    print("LaRA Routing Benchmark")
    print(f"Iterations: {result.iterations}")
    print(f"Mean latency: {result.mean_ms:.4f} ms")
    print(f"Median latency: {result.median_ms:.4f} ms")
    print("Route distribution:")
    for route, count in sorted(result.route_counts.items()):
        print(f"  {route}: {count}")


if __name__ == "__main__":
    main()
