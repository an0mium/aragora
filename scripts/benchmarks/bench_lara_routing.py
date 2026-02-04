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


def run_benchmark(iterations: int, min_nodes: int, max_nodes: int) -> RoutingBenchmarkResult:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark LaRA routing decisions")
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--min-nodes", type=int, default=0)
    parser.add_argument("--max-nodes", type=int, default=5000)
    args = parser.parse_args()

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
