#!/usr/bin/env python3
"""
Knowledge Mound Maintenance Benchmarks.

Measures performance of deduplication and pruning operations:
- Duplicate detection speed (similarity search)
- Cluster merging latency
- Pruning evaluation time
- Batch archive/delete operations
- Confidence decay application

Metrics:
- Items per second processed
- Memory usage during large batch operations
- Query latency for duplicate detection
- Merge operation throughput

Usage:
    python -m benchmarks.knowledge_maintenance
    python -m benchmarks.knowledge_maintenance --items 10000
    python -m benchmarks.knowledge_maintenance --workspace test_workspace
"""

import argparse
import asyncio
import logging
import statistics
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional
from unittest.mock import AsyncMock, MagicMock

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

    total_time_ms = sum(latencies)
    return BenchmarkResult(
        name=name,
        iterations=len(latencies),
        total_time_ms=total_time_ms,
        min_latency_ms=min(latencies),
        max_latency_ms=max(latencies),
        mean_latency_ms=statistics.mean(latencies),
        median_latency_ms=statistics.median(latencies),
        p95_latency_ms=calculate_percentile(latencies, 95),
        p99_latency_ms=calculate_percentile(latencies, 99),
        std_dev_ms=statistics.stdev(latencies) if len(latencies) > 1 else 0,
        throughput_ops_per_sec=len(latencies) / (total_time_ms / 1000) if total_time_ms > 0 else 0,
        extra=extra or {},
    )


class MockStore:
    """Mock store for benchmarking without real DB."""

    def __init__(self, item_count: int):
        self.items = {}
        self.archived = set()
        self.deleted = set()
        self._generate_items(item_count)

    def _generate_items(self, count: int):
        """Generate mock knowledge items."""
        base_time = datetime.now() - timedelta(days=60)
        for i in range(count):
            node_id = f"node_{uuid.uuid4().hex[:8]}"
            # Create some duplicates (10% of items)
            if i > 0 and i % 10 == 0:
                content_hash = self.items[list(self.items.keys())[i - 1]]["content_hash"]
            else:
                content_hash = f"hash_{uuid.uuid4().hex[:16]}"

            self.items[node_id] = {
                "node_id": node_id,
                "content": f"Test content for knowledge item {i}",
                "content_hash": content_hash,
                "confidence": 0.3 + (i % 7) * 0.1,  # 0.3 - 0.9
                "staleness_score": 0.5 + (i % 5) * 0.1,  # 0.5 - 0.9
                "tier": ["fast", "medium", "slow", "glacial"][i % 4],
                "created_at": base_time + timedelta(days=i % 45),
                "retrieval_count": i % 20,
                "last_retrieved_at": base_time + timedelta(days=(i % 30)),
            }

    async def get_nodes_for_workspace(self, workspace_id: str) -> List[dict]:
        return list(self.items.values())

    async def search_similar(self, content: str, threshold: float) -> List[dict]:
        # Simulate similarity search delay
        await asyncio.sleep(0.001)
        return list(self.items.values())[:10]

    async def get_nodes_by_content_hash(self, workspace_id: str) -> dict:
        hash_groups = {}
        for item in self.items.values():
            h = item["content_hash"]
            if h not in hash_groups:
                hash_groups[h] = []
            hash_groups[h].append(item["node_id"])
        return hash_groups

    async def archive_node(self, node_id: str):
        if node_id in self.items:
            self.archived.add(node_id)

    async def delete_node(self, node_id: str):
        if node_id in self.items:
            self.deleted.add(node_id)
            del self.items[node_id]

    async def update_node(self, node_id: str, updates: dict):
        if node_id in self.items:
            self.items[node_id].update(updates)

    async def get_node(self, node_id: str) -> Optional[dict]:
        return self.items.get(node_id)


async def benchmark_duplicate_detection(
    store: MockStore,
    iterations: int,
) -> BenchmarkResult:
    """Benchmark duplicate detection speed."""
    latencies = []

    for _ in range(iterations):
        start = time.perf_counter()

        # Simulate finding duplicates by content hash
        hash_groups = await store.get_nodes_by_content_hash("test_workspace")
        duplicates_found = sum(1 for ids in hash_groups.values() if len(ids) > 1)

        elapsed = (time.perf_counter() - start) * 1000
        latencies.append(elapsed)

    return create_benchmark_result(
        "Duplicate Detection",
        latencies,
        {"duplicates_found": duplicates_found},
    )


async def benchmark_similarity_search(
    store: MockStore,
    iterations: int,
) -> BenchmarkResult:
    """Benchmark similarity search for near-duplicates."""
    latencies = []

    sample_content = "Test content for knowledge item"

    for _ in range(iterations):
        start = time.perf_counter()

        # Simulate similarity search
        similar = await store.search_similar(sample_content, threshold=0.9)

        elapsed = (time.perf_counter() - start) * 1000
        latencies.append(elapsed)

    return create_benchmark_result(
        "Similarity Search",
        latencies,
        {"avg_results": len(similar)},
    )


async def benchmark_pruning_evaluation(
    store: MockStore,
    iterations: int,
    staleness_threshold: float = 0.9,
    min_age_days: int = 30,
) -> BenchmarkResult:
    """Benchmark pruning candidate evaluation."""
    latencies = []
    prunable_counts = []

    cutoff_date = datetime.now() - timedelta(days=min_age_days)

    for _ in range(iterations):
        start = time.perf_counter()

        # Evaluate which items are prunable
        prunable = [
            item
            for item in store.items.values()
            if item["staleness_score"] >= staleness_threshold
            and item["created_at"] < cutoff_date
            and item["tier"] != "glacial"
        ]

        elapsed = (time.perf_counter() - start) * 1000
        latencies.append(elapsed)
        prunable_counts.append(len(prunable))

    return create_benchmark_result(
        "Pruning Evaluation",
        latencies,
        {"avg_prunable_items": statistics.mean(prunable_counts) if prunable_counts else 0},
    )


async def benchmark_batch_archive(
    store: MockStore,
    batch_size: int,
    iterations: int,
) -> BenchmarkResult:
    """Benchmark batch archive operations."""
    latencies = []

    # Get IDs to archive
    all_ids = list(store.items.keys())

    for i in range(iterations):
        batch_start = (i * batch_size) % len(all_ids)
        batch_ids = all_ids[batch_start : batch_start + batch_size]

        if not batch_ids:
            continue

        start = time.perf_counter()

        # Archive batch
        for node_id in batch_ids:
            await store.archive_node(node_id)

        elapsed = (time.perf_counter() - start) * 1000
        latencies.append(elapsed)

    return create_benchmark_result(
        f"Batch Archive ({batch_size} items)",
        latencies,
        {"total_archived": len(store.archived)},
    )


async def benchmark_confidence_decay(
    store: MockStore,
    iterations: int,
    decay_rate: float = 0.01,
) -> BenchmarkResult:
    """Benchmark confidence decay application."""
    latencies = []
    items_decayed_counts = []

    for _ in range(iterations):
        start = time.perf_counter()

        items_decayed = 0
        for item in store.items.values():
            old_conf = item["confidence"]
            new_conf = max(0.1, old_conf * (1 - decay_rate))
            if new_conf != old_conf:
                await store.update_node(item["node_id"], {"confidence": new_conf})
                items_decayed += 1

        elapsed = (time.perf_counter() - start) * 1000
        latencies.append(elapsed)
        items_decayed_counts.append(items_decayed)

    return create_benchmark_result(
        "Confidence Decay",
        latencies,
        {"avg_items_decayed": statistics.mean(items_decayed_counts) if items_decayed_counts else 0},
    )


async def benchmark_merge_operation(
    store: MockStore,
    iterations: int,
) -> BenchmarkResult:
    """Benchmark cluster merge operations."""
    latencies = []

    # Find duplicate groups
    hash_groups = await store.get_nodes_by_content_hash("test_workspace")
    duplicate_groups = [ids for ids in hash_groups.values() if len(ids) > 1]

    for i, group in enumerate(duplicate_groups[:iterations]):
        if len(group) < 2:
            continue

        start = time.perf_counter()

        # Keep first, archive rest
        primary_id = group[0]
        for dup_id in group[1:]:
            await store.archive_node(dup_id)

        elapsed = (time.perf_counter() - start) * 1000
        latencies.append(elapsed)

    return create_benchmark_result(
        "Merge Operation",
        latencies,
        {"groups_merged": len(latencies)},
    )


async def run_benchmarks(
    item_count: int = 1000,
    iterations: int = 100,
) -> List[BenchmarkResult]:
    """Run all knowledge maintenance benchmarks."""
    print(f"\n{'='*60}")
    print("Knowledge Maintenance Benchmarks")
    print(f"{'='*60}")
    print(f"Item count: {item_count}")
    print(f"Iterations: {iterations}")
    print(f"{'='*60}\n")

    # Create mock store with test data
    store = MockStore(item_count)

    results = []

    # Run benchmarks
    print("Running duplicate detection benchmark...")
    result = await benchmark_duplicate_detection(store, iterations)
    print(result)
    results.append(result)
    print()

    print("Running similarity search benchmark...")
    result = await benchmark_similarity_search(store, iterations)
    print(result)
    results.append(result)
    print()

    print("Running pruning evaluation benchmark...")
    result = await benchmark_pruning_evaluation(store, iterations)
    print(result)
    results.append(result)
    print()

    print("Running batch archive benchmark (10 items)...")
    result = await benchmark_batch_archive(store, 10, iterations)
    print(result)
    results.append(result)
    print()

    print("Running batch archive benchmark (100 items)...")
    result = await benchmark_batch_archive(store, 100, iterations // 10)
    print(result)
    results.append(result)
    print()

    print("Running confidence decay benchmark...")
    result = await benchmark_confidence_decay(store, iterations // 10)
    print(result)
    results.append(result)
    print()

    print("Running merge operation benchmark...")
    result = await benchmark_merge_operation(store, iterations)
    print(result)
    results.append(result)
    print()

    # Summary
    print(f"{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in results:
        print(f"  {r.name}: {r.mean_latency_ms:.3f}ms avg, {r.throughput_ops_per_sec:.1f} ops/sec")

    return results


def main():
    parser = argparse.ArgumentParser(description="Knowledge Maintenance Benchmarks")
    parser.add_argument(
        "--items",
        type=int,
        default=1000,
        help="Number of items to generate (default: 1000)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of iterations per benchmark (default: 100)",
    )
    args = parser.parse_args()

    asyncio.run(run_benchmarks(args.items, args.iterations))


if __name__ == "__main__":
    main()
