#!/usr/bin/env python3
"""
Knowledge Mound Adapter Performance Benchmarks.

Measures performance of KM adapters under various load conditions:
- Forward sync throughput (subsystem → KM)
- Reverse query latency (KM → subsystem)
- Semantic search performance
- Validation feedback processing
- Concurrent adapter operations

Metrics:
- Write latency (forward sync)
- Query latency (reverse queries)
- Semantic search latency
- Cache hit rates
- Concurrent operation performance
- SLO compliance percentages

Usage:
    python -m benchmarks.km_adapters
    python -m benchmarks.km_adapters --iterations 500
    python -m benchmarks.km_adapters --concurrent 100
    python -m benchmarks.km_adapters --adapters continuum,consensus
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
from unittest.mock import AsyncMock, MagicMock
import uuid

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

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
    k = (len(sorted_data) - 1) * percentile / 100
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_data) else f
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def create_result(name: str, latencies: List[float]) -> BenchmarkResult:
    """Create a BenchmarkResult from a list of latencies in seconds."""
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
        )

    latencies_ms = [lat * 1000 for lat in latencies]
    total_time_s = sum(latencies)

    return BenchmarkResult(
        name=name,
        iterations=len(latencies),
        total_time_ms=total_time_s * 1000,
        min_latency_ms=min(latencies_ms),
        max_latency_ms=max(latencies_ms),
        mean_latency_ms=statistics.mean(latencies_ms),
        median_latency_ms=statistics.median(latencies_ms),
        p95_latency_ms=calculate_percentile(latencies_ms, 95),
        p99_latency_ms=calculate_percentile(latencies_ms, 99),
        std_dev_ms=statistics.stdev(latencies_ms) if len(latencies_ms) > 1 else 0,
        throughput_ops_per_sec=len(latencies) / total_time_s if total_time_s > 0 else 0,
    )


class KMAdapterBenchmark:
    """Benchmark suite for Knowledge Mound adapters."""

    def __init__(
        self,
        iterations: int = 100,
        concurrent: int = 10,
        adapters: Optional[List[str]] = None,
    ):
        self.iterations = iterations
        self.concurrent = concurrent
        self.adapters = adapters or [
            "continuum",
            "consensus",
            "evidence",
            "elo",
            "insights",
            "belief",
        ]
        self.results: List[BenchmarkResult] = []
        self._temp_dir: Optional[tempfile.TemporaryDirectory] = None

    async def setup(self) -> None:
        """Set up benchmark environment."""
        self._temp_dir = tempfile.TemporaryDirectory()

    async def teardown(self) -> None:
        """Clean up benchmark environment."""
        if self._temp_dir:
            self._temp_dir.cleanup()

    async def benchmark_forward_sync(self, adapter_name: str) -> BenchmarkResult:
        """Benchmark forward sync (subsystem → KM) throughput."""
        from aragora.knowledge.mound.adapters.continuum_adapter import ContinuumAdapter
        from aragora.knowledge.mound.core import KnowledgeMound
        from aragora.knowledge.mound.types import MoundConfig

        config = MoundConfig(workspace_id=f"bench_{adapter_name}")

        # Create mock mound for benchmarking
        mound = MagicMock(spec=KnowledgeMound)
        mound.ingest = AsyncMock(return_value=f"km_{uuid.uuid4().hex[:8]}")

        adapter = ContinuumAdapter(mound, config)

        latencies = []
        for i in range(self.iterations):
            entry_data = {
                "content": f"Benchmark entry {i} for {adapter_name}",
                "debate_id": f"debate_{i}",
                "confidence": 0.85,
                "source": "benchmark",
            }

            start = time.perf_counter()
            await adapter.forward_sync(entry_data)
            end = time.perf_counter()

            latencies.append(end - start)

        return create_result(f"{adapter_name}_forward_sync", latencies)

    async def benchmark_reverse_query(self, adapter_name: str) -> BenchmarkResult:
        """Benchmark reverse query (KM → subsystem) latency."""
        from aragora.knowledge.mound.adapters.continuum_adapter import ContinuumAdapter
        from aragora.knowledge.mound.core import KnowledgeMound
        from aragora.knowledge.mound.types import MoundConfig

        config = MoundConfig(workspace_id=f"bench_{adapter_name}")

        # Create mock mound with query results
        mound = MagicMock(spec=KnowledgeMound)
        mound.query = AsyncMock(
            return_value=[
                {"id": f"km_{i}", "content": f"Result {i}", "confidence": 0.8} for i in range(5)
            ]
        )

        adapter = ContinuumAdapter(mound, config)

        latencies = []
        topics = [
            "climate change",
            "artificial intelligence",
            "economics",
            "healthcare",
            "education",
        ]

        for i in range(self.iterations):
            topic = topics[i % len(topics)]

            start = time.perf_counter()
            await adapter.reverse_query(topic=topic, limit=10)
            end = time.perf_counter()

            latencies.append(end - start)

        return create_result(f"{adapter_name}_reverse_query", latencies)

    async def benchmark_semantic_search(self, adapter_name: str) -> BenchmarkResult:
        """Benchmark semantic search performance."""
        from aragora.knowledge.mound.adapters.continuum_adapter import ContinuumAdapter
        from aragora.knowledge.mound.core import KnowledgeMound
        from aragora.knowledge.mound.types import MoundConfig

        config = MoundConfig(workspace_id=f"bench_{adapter_name}")

        # Create mock mound with semantic search
        mound = MagicMock(spec=KnowledgeMound)
        mound.semantic_search = AsyncMock(
            return_value=[
                {
                    "id": f"km_{i}",
                    "content": f"Semantic result {i}",
                    "similarity": 0.9 - i * 0.05,
                }
                for i in range(10)
            ]
        )

        adapter = ContinuumAdapter(mound, config)

        latencies = []
        queries = [
            "What are the main arguments about climate policy?",
            "How does machine learning work?",
            "Economic implications of automation",
            "Healthcare system improvements",
            "Education reform strategies",
        ]

        for i in range(self.iterations):
            query = queries[i % len(queries)]

            start = time.perf_counter()
            await adapter.semantic_search(query=query, limit=10, min_similarity=0.6)
            end = time.perf_counter()

            latencies.append(end - start)

        return create_result(f"{adapter_name}_semantic_search", latencies)

    async def benchmark_concurrent_operations(self, adapter_name: str) -> BenchmarkResult:
        """Benchmark concurrent adapter operations."""
        from aragora.knowledge.mound.adapters.continuum_adapter import ContinuumAdapter
        from aragora.knowledge.mound.core import KnowledgeMound
        from aragora.knowledge.mound.types import MoundConfig

        config = MoundConfig(workspace_id=f"bench_{adapter_name}")

        mound = MagicMock(spec=KnowledgeMound)
        mound.ingest = AsyncMock(return_value=f"km_{uuid.uuid4().hex[:8]}")
        mound.query = AsyncMock(
            return_value=[{"id": "km_1", "content": "Result", "confidence": 0.8}]
        )
        mound.semantic_search = AsyncMock(
            return_value=[{"id": "km_1", "content": "Result", "similarity": 0.9}]
        )

        adapter = ContinuumAdapter(mound, config)

        async def mixed_operation(idx: int) -> float:
            """Perform a mixed operation and return latency."""
            op_type = idx % 3
            start = time.perf_counter()

            if op_type == 0:
                await adapter.forward_sync({"content": f"Entry {idx}", "confidence": 0.8})
            elif op_type == 1:
                await adapter.reverse_query(topic="test topic", limit=5)
            else:
                await adapter.semantic_search(query="test query", limit=5)

            return time.perf_counter() - start

        # Run concurrent batches
        all_latencies = []
        batch_count = self.iterations // self.concurrent

        for batch in range(batch_count):
            tasks = [mixed_operation(batch * self.concurrent + i) for i in range(self.concurrent)]
            batch_latencies = await asyncio.gather(*tasks)
            all_latencies.extend(batch_latencies)

        return create_result(f"{adapter_name}_concurrent", all_latencies)

    async def benchmark_validation_feedback(self, adapter_name: str) -> BenchmarkResult:
        """Benchmark validation feedback processing."""
        from aragora.knowledge.mound.adapters.continuum_adapter import ContinuumAdapter
        from aragora.knowledge.mound.core import KnowledgeMound
        from aragora.knowledge.mound.types import MoundConfig

        config = MoundConfig(workspace_id=f"bench_{adapter_name}")

        mound = MagicMock(spec=KnowledgeMound)
        mound.update_confidence = AsyncMock(return_value=True)
        mound.add_validation = AsyncMock(return_value=True)

        adapter = ContinuumAdapter(mound, config)

        latencies = []
        for i in range(self.iterations):
            validation = {
                "item_id": f"km_{i}",
                "is_positive": i % 3 != 0,  # 2/3 positive
                "debate_id": f"debate_{i}",
                "confidence_delta": 0.1 if i % 3 != 0 else -0.2,
            }

            start = time.perf_counter()
            await adapter.process_validation(validation)
            end = time.perf_counter()

            latencies.append(end - start)

        return create_result(f"{adapter_name}_validation", latencies)

    async def benchmark_batch_ingestion(self, adapter_name: str) -> BenchmarkResult:
        """Benchmark batch ingestion performance."""
        from aragora.knowledge.mound.adapters.continuum_adapter import ContinuumAdapter
        from aragora.knowledge.mound.core import KnowledgeMound
        from aragora.knowledge.mound.types import MoundConfig

        config = MoundConfig(workspace_id=f"bench_{adapter_name}")

        mound = MagicMock(spec=KnowledgeMound)
        mound.ingest_batch = AsyncMock(return_value=[f"km_{i}" for i in range(100)])

        adapter = ContinuumAdapter(mound, config)

        latencies = []
        batch_size = 100

        for batch_num in range(self.iterations // 10):  # Fewer iterations for batches
            entries = [
                {
                    "content": f"Batch {batch_num} entry {i}",
                    "confidence": 0.8,
                    "source": "benchmark",
                }
                for i in range(batch_size)
            ]

            start = time.perf_counter()
            await adapter.forward_sync_batch(entries)
            end = time.perf_counter()

            latencies.append(end - start)

        result = create_result(f"{adapter_name}_batch_ingestion", latencies)
        result.extra["batch_size"] = batch_size
        result.extra["items_per_second"] = (
            len(latencies) * batch_size / (sum(latencies) if latencies else 1)
        )
        return result

    async def run_all(self) -> List[BenchmarkResult]:
        """Run all benchmarks."""
        await self.setup()

        try:
            for adapter in self.adapters:
                # Forward sync
                result = await self.benchmark_forward_sync(adapter)
                self.results.append(result)

                # Reverse query
                result = await self.benchmark_reverse_query(adapter)
                self.results.append(result)

                # Semantic search
                result = await self.benchmark_semantic_search(adapter)
                self.results.append(result)

                # Concurrent operations
                result = await self.benchmark_concurrent_operations(adapter)
                self.results.append(result)

                # Validation feedback
                result = await self.benchmark_validation_feedback(adapter)
                self.results.append(result)

                # Batch ingestion
                result = await self.benchmark_batch_ingestion(adapter)
                self.results.append(result)
                if "items_per_second" in result.extra:
                    pass

            # Summary

            slo_checks = {
                "forward_sync": {"p95": 300, "p99": 800},
                "reverse_query": {"p95": 150, "p99": 500},
                "semantic_search": {"p95": 300, "p99": 1000},
                "validation": {"p95": 500, "p99": 1500},
            }

            for result in self.results:
                for slo_name, thresholds in slo_checks.items():
                    if slo_name in result.name:
                        result.p95_latency_ms <= thresholds["p95"]
                        result.p99_latency_ms <= thresholds["p99"]

            return self.results

        finally:
            await self.teardown()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="KM Adapter Performance Benchmarks")
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of iterations per benchmark (default: 100)",
    )
    parser.add_argument(
        "--concurrent",
        type=int,
        default=10,
        help="Concurrency level for concurrent benchmarks (default: 10)",
    )
    parser.add_argument(
        "--adapters",
        type=str,
        default=None,
        help="Comma-separated list of adapters to benchmark (default: all)",
    )

    args = parser.parse_args()

    adapters = args.adapters.split(",") if args.adapters else None

    benchmark = KMAdapterBenchmark(
        iterations=args.iterations,
        concurrent=args.concurrent,
        adapters=adapters,
    )

    asyncio.run(benchmark.run_all())


if __name__ == "__main__":
    main()
