#!/usr/bin/env python3
"""
Memory Tier Latency Benchmarks.

Measures performance of the ContinuumMemory system across all tiers:
- Fast tier (1h half-life): Immediate patterns
- Medium tier (24h half-life): Tactical learning
- Slow tier (7d half-life): Strategic learning
- Glacial tier (30d half-life): Foundational knowledge

Metrics:
- Write latency (store operations)
- Read latency (query operations)
- Cache hit rates
- Tier promotion/demotion latency
- Concurrent access performance

Usage:
    python -m benchmarks.memory_tiers
    python -m benchmarks.memory_tiers --iterations 1000
    python -m benchmarks.memory_tiers --concurrent 50
"""

import argparse
import asyncio
import logging
import statistics
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from aragora.memory.continuum import ContinuumMemory, ContinuumMemoryEntry
from aragora.memory.tier_manager import MemoryTier, TierManager

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""

    name: str
    iterations: int
    total_time_ms: float
    min_latency_ms: float
    max_latency_ms: float
    mean_latency_ms: float
    median_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    std_dev_ms: float
    throughput_ops_per_sec: float
    extra: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"{self.name}:\n"
            f"  Iterations: {self.iterations}\n"
            f"  Total time: {self.total_time_ms:.2f}ms\n"
            f"  Latency (ms): min={self.min_latency_ms:.3f}, "
            f"max={self.max_latency_ms:.3f}, "
            f"mean={self.mean_latency_ms:.3f}, "
            f"median={self.median_latency_ms:.3f}\n"
            f"  Percentiles (ms): p95={self.p95_latency_ms:.3f}, "
            f"p99={self.p99_latency_ms:.3f}\n"
            f"  Std dev: {self.std_dev_ms:.3f}ms\n"
            f"  Throughput: {self.throughput_ops_per_sec:.1f} ops/sec"
        )


def calculate_percentile(data: List[float], percentile: float) -> float:
    """Calculate percentile of a sorted list."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    index = int(len(sorted_data) * percentile / 100)
    return sorted_data[min(index, len(sorted_data) - 1)]


def create_benchmark_result(
    name: str, latencies: List[float], extra: dict = None
) -> BenchmarkResult:
    """Create a BenchmarkResult from latency measurements."""
    if not latencies:
        return BenchmarkResult(
            name=name,
            iterations=0,
            total_time_ms=0,
            min_latency_ms=0,
            max_latency_ms=0,
            mean_latency_ms=0,
            median_latency_ms=0,
            p95_latency_ms=0,
            p99_latency_ms=0,
            std_dev_ms=0,
            throughput_ops_per_sec=0,
            extra=extra or {},
        )

    total_time = sum(latencies)
    return BenchmarkResult(
        name=name,
        iterations=len(latencies),
        total_time_ms=total_time,
        min_latency_ms=min(latencies),
        max_latency_ms=max(latencies),
        mean_latency_ms=statistics.mean(latencies),
        median_latency_ms=statistics.median(latencies),
        p95_latency_ms=calculate_percentile(latencies, 95),
        p99_latency_ms=calculate_percentile(latencies, 99),
        std_dev_ms=statistics.stdev(latencies) if len(latencies) > 1 else 0,
        throughput_ops_per_sec=(len(latencies) / total_time * 1000) if total_time > 0 else 0,
        extra=extra or {},
    )


class MemoryTierBenchmark:
    """Benchmark suite for memory tier operations."""

    def __init__(self, db_path: Optional[Path] = None, iterations: int = 100):
        """
        Initialize benchmark.

        Args:
            db_path: Path to database file (uses temp file if None)
            iterations: Number of iterations per benchmark
        """
        self.iterations = iterations
        self._temp_dir = None
        if db_path is None:
            self._temp_dir = tempfile.mkdtemp(prefix="aragora_bench_")
            db_path = Path(self._temp_dir) / "benchmark_memory.db"
        self.db_path = db_path
        self.memory: Optional[ContinuumMemory] = None
        self.results: List[BenchmarkResult] = []

    def setup(self):
        """Initialize memory system."""
        self.memory = ContinuumMemory(db_path=str(self.db_path))
        # Pre-warm with some data
        for i in range(10):
            for tier in MemoryTier:
                self.memory.store(
                    content=f"Warmup content {i} for {tier.value}",
                    tier=tier,
                    importance=0.5,
                )

    def cleanup(self):
        """Clean up resources."""
        if self.memory:
            self.memory.close()
        if self._temp_dir:
            import shutil

            shutil.rmtree(self._temp_dir, ignore_errors=True)

    def benchmark_write_latency(self, tier: MemoryTier) -> BenchmarkResult:
        """Benchmark write latency for a specific tier."""
        latencies = []

        for i in range(self.iterations):
            content = f"Benchmark content {i} for {tier.value} at {time.time()}"
            start = time.perf_counter()
            self.memory.store(
                content=content,
                tier=tier,
                importance=0.5 + (i % 10) / 20,  # Vary importance
                metadata={"iteration": i, "tier": tier.value},
            )
            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)

        return create_benchmark_result(
            f"Write latency ({tier.value})",
            latencies,
            extra={"tier": tier.value},
        )

    def benchmark_read_latency(self, tier: MemoryTier) -> BenchmarkResult:
        """Benchmark read latency for a specific tier."""
        # First, ensure there's data to query
        for i in range(50):
            self.memory.store(
                content=f"Searchable content {i} for {tier.value}",
                tier=tier,
                importance=0.5,
            )

        latencies = []
        for i in range(self.iterations):
            start = time.perf_counter()
            entries = self.memory.query(
                tier=tier,
                limit=10,
                min_importance=0.3,
            )
            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)

        return create_benchmark_result(
            f"Read latency ({tier.value})",
            latencies,
            extra={"tier": tier.value, "avg_results": len(entries) if entries else 0},
        )

    def benchmark_search_latency(self) -> BenchmarkResult:
        """Benchmark content search across all tiers."""
        # Store searchable content
        keywords = ["algorithm", "optimization", "performance", "memory", "cache"]
        for i, keyword in enumerate(keywords):
            for tier in MemoryTier:
                self.memory.store(
                    content=f"Content about {keyword} in {tier.value} tier - iteration {i}",
                    tier=tier,
                    importance=0.7,
                )

        latencies = []
        for i in range(self.iterations):
            keyword = keywords[i % len(keywords)]
            start = time.perf_counter()
            entries = self.memory.search(content_like=keyword, limit=20)
            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)

        return create_benchmark_result(
            "Search latency (all tiers)",
            latencies,
        )

    def benchmark_tier_promotion(self) -> BenchmarkResult:
        """Benchmark tier promotion operations."""
        # Create entries in slow tier
        entry_ids = []
        for i in range(self.iterations):
            entry = self.memory.store(
                content=f"Promotable content {i}",
                tier=MemoryTier.SLOW,
                importance=0.8,
            )
            entry_ids.append(entry.id)

        latencies = []
        for entry_id in entry_ids:
            start = time.perf_counter()
            # Simulate promotion by updating with high surprise score
            self.memory.update(
                entry_id,
                surprise_score=0.9,  # High surprise triggers promotion check
            )
            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)

        return create_benchmark_result(
            "Tier promotion check",
            latencies,
        )

    def benchmark_cache_hit_rate(self) -> BenchmarkResult:
        """Benchmark cache hit rates under repeated access patterns."""
        # Store fixed set of entries
        entry_ids = []
        for i in range(20):
            entry = self.memory.store(
                content=f"Cacheable content {i}",
                tier=MemoryTier.FAST,
                importance=0.6,
            )
            entry_ids.append(entry.id)

        # Access pattern: 80% recent (last 5), 20% random
        hits = 0
        misses = 0
        latencies = []

        import random

        for i in range(self.iterations):
            if random.random() < 0.8:
                # Access recent
                entry_id = entry_ids[-1 - (i % 5)]
            else:
                # Access random
                entry_id = random.choice(entry_ids)

            start = time.perf_counter()
            entry = self.memory.get(entry_id)
            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)

            if entry:
                hits += 1
            else:
                misses += 1

        hit_rate = hits / (hits + misses) if (hits + misses) > 0 else 0

        return create_benchmark_result(
            "Cache hit rate test",
            latencies,
            extra={"hit_rate": hit_rate, "hits": hits, "misses": misses},
        )

    def benchmark_concurrent_writes(self, concurrency: int = 10) -> BenchmarkResult:
        """Benchmark concurrent write operations."""

        async def write_task(task_id: int, tier: MemoryTier) -> List[float]:
            latencies = []
            per_task = self.iterations // concurrency
            for i in range(per_task):
                content = f"Concurrent write {task_id}-{i} to {tier.value}"
                start = time.perf_counter()
                # Note: ContinuumMemory uses sqlite which handles concurrency
                self.memory.store(content=content, tier=tier, importance=0.5)
                elapsed = (time.perf_counter() - start) * 1000
                latencies.append(elapsed)
            return latencies

        async def run_concurrent():
            tasks = []
            for i in range(concurrency):
                tier = list(MemoryTier)[i % len(MemoryTier)]
                tasks.append(write_task(i, tier))
            results = await asyncio.gather(*tasks)
            return [lat for task_lats in results for lat in task_lats]

        all_latencies = asyncio.run(run_concurrent())

        return create_benchmark_result(
            f"Concurrent writes (n={concurrency})",
            all_latencies,
            extra={"concurrency": concurrency},
        )

    def benchmark_cleanup_performance(self) -> BenchmarkResult:
        """Benchmark cleanup/garbage collection performance."""
        # Store many entries across tiers
        for i in range(500):
            tier = list(MemoryTier)[i % len(MemoryTier)]
            self.memory.store(
                content=f"Cleanup test content {i}",
                tier=tier,
                importance=0.3,  # Low importance for cleanup eligibility
            )

        latencies = []
        for _ in range(min(self.iterations, 20)):  # Cleanup is expensive, limit iterations
            start = time.perf_counter()
            removed = self.memory.cleanup(max_entries_per_tier=100)
            elapsed = (time.perf_counter() - start) * 1000
            latencies.append(elapsed)

        return create_benchmark_result(
            "Cleanup performance",
            latencies,
        )

    def run_all(self) -> List[BenchmarkResult]:
        """Run all benchmarks."""
        print(f"Running memory tier benchmarks ({self.iterations} iterations each)...\n")

        self.setup()
        try:
            # Write latency per tier
            for tier in MemoryTier:
                result = self.benchmark_write_latency(tier)
                self.results.append(result)
                print(result)
                print()

            # Read latency per tier
            for tier in MemoryTier:
                result = self.benchmark_read_latency(tier)
                self.results.append(result)
                print(result)
                print()

            # Search latency
            result = self.benchmark_search_latency()
            self.results.append(result)
            print(result)
            print()

            # Tier promotion
            result = self.benchmark_tier_promotion()
            self.results.append(result)
            print(result)
            print()

            # Cache hit rate
            result = self.benchmark_cache_hit_rate()
            self.results.append(result)
            print(result)
            if "hit_rate" in result.extra:
                print(f"  Cache hit rate: {result.extra['hit_rate']:.1%}")
            print()

            # Concurrent writes
            result = self.benchmark_concurrent_writes(concurrency=10)
            self.results.append(result)
            print(result)
            print()

            # Cleanup performance
            result = self.benchmark_cleanup_performance()
            self.results.append(result)
            print(result)
            print()

        finally:
            self.cleanup()

        return self.results

    def summary(self) -> str:
        """Generate summary of all benchmark results."""
        lines = ["=" * 60, "MEMORY TIER BENCHMARK SUMMARY", "=" * 60, ""]

        # Group by operation type
        write_results = [r for r in self.results if "Write" in r.name]
        read_results = [r for r in self.results if "Read" in r.name]
        other_results = [r for r in self.results if "Write" not in r.name and "Read" not in r.name]

        if write_results:
            lines.append("Write Operations:")
            for r in write_results:
                lines.append(
                    f"  {r.name}: {r.mean_latency_ms:.3f}ms avg, {r.throughput_ops_per_sec:.0f} ops/sec"
                )
            lines.append("")

        if read_results:
            lines.append("Read Operations:")
            for r in read_results:
                lines.append(
                    f"  {r.name}: {r.mean_latency_ms:.3f}ms avg, {r.throughput_ops_per_sec:.0f} ops/sec"
                )
            lines.append("")

        if other_results:
            lines.append("Other Operations:")
            for r in other_results:
                lines.append(f"  {r.name}: {r.mean_latency_ms:.3f}ms avg")
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Memory Tier Latency Benchmarks")
    parser.add_argument("--iterations", type=int, default=100, help="Iterations per benchmark")
    parser.add_argument("--concurrent", type=int, default=10, help="Concurrency level")
    parser.add_argument(
        "--db-path", type=str, default=None, help="Database path (temp if not specified)"
    )
    args = parser.parse_args()

    benchmark = MemoryTierBenchmark(
        db_path=Path(args.db_path) if args.db_path else None,
        iterations=args.iterations,
    )
    benchmark.run_all()
    print(benchmark.summary())


if __name__ == "__main__":
    main()
