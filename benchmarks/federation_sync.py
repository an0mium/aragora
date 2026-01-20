#!/usr/bin/env python3
"""
Federation Sync Benchmarks.

Measures performance of multi-region federation operations:
- Region registration latency
- Push sync throughput
- Pull sync throughput
- Bidirectional sync performance
- Scope filtering overhead (full, summary, metadata)
- Large batch sync performance

Metrics:
- Items per second synced
- Latency per sync operation
- Memory usage during large syncs
- Regional status query time

Usage:
    python -m benchmarks.federation_sync
    python -m benchmarks.federation_sync --items 10000
    python -m benchmarks.federation_sync --regions 5
"""

import argparse
import asyncio
import logging
import statistics
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

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


class MockFederationStore:
    """Mock store for benchmarking federation operations."""

    def __init__(self, item_count: int):
        self.items: Dict[str, Dict[str, Any]] = {}
        self.regions: Dict[str, Dict[str, Any]] = {}
        self._generate_items(item_count)

    def _generate_items(self, count: int):
        """Generate mock knowledge items for sync."""
        base_time = datetime.now() - timedelta(days=30)
        for i in range(count):
            node_id = f"node_{uuid.uuid4().hex[:8]}"
            self.items[node_id] = {
                "id": node_id,
                "content": f"Knowledge item {i} with some content for testing sync operations",
                "confidence": 0.5 + (i % 5) * 0.1,
                "visibility": ["workspace", "organization", "public"][i % 3],
                "created_at": (base_time + timedelta(days=i % 20)).isoformat(),
                "metadata": {
                    "topics": [f"topic_{i % 10}"],
                    "node_type": "fact",
                },
            }

    def register_region(self, region_id: str, endpoint: str, mode: str) -> Dict:
        """Register a federated region."""
        region = {
            "region_id": region_id,
            "endpoint_url": endpoint,
            "mode": mode,
            "enabled": True,
            "last_sync_at": None,
        }
        self.regions[region_id] = region
        return region

    def unregister_region(self, region_id: str) -> bool:
        """Unregister a region."""
        if region_id in self.regions:
            del self.regions[region_id]
            return True
        return False

    async def get_items_for_sync(
        self,
        visibility_levels: List[str],
        scope: str = "full",
        limit: int = 1000,
    ) -> List[Dict]:
        """Get items filtered by visibility for sync."""
        # Simulate filtering
        await asyncio.sleep(0.001)

        items = [
            item for item in self.items.values()
            if item["visibility"] in visibility_levels
        ][:limit]

        # Apply scope filtering
        if scope == "metadata":
            return [{"id": i["id"], "metadata": i["metadata"]} for i in items]
        elif scope == "summary":
            return [
                {"id": i["id"], "content": i["content"][:100], "metadata": i["metadata"]}
                for i in items
            ]
        return items

    async def push_to_remote(self, region_id: str, items: List[Dict]) -> int:
        """Simulate pushing items to remote region."""
        # Simulate network latency
        await asyncio.sleep(0.005 * len(items) / 100)
        return len(items)

    async def pull_from_remote(self, region_id: str, since: Optional[datetime] = None) -> List[Dict]:
        """Simulate pulling items from remote region."""
        # Simulate network latency and return random items
        await asyncio.sleep(0.01)
        return list(self.items.values())[:50]

    async def ingest_items(self, items: List[Dict]) -> int:
        """Simulate ingesting pulled items."""
        await asyncio.sleep(0.001 * len(items))
        return len(items)


async def benchmark_region_registration(
    store: MockFederationStore,
    num_regions: int,
    iterations: int,
) -> BenchmarkResult:
    """Benchmark region registration performance."""
    latencies = []

    for _ in range(iterations):
        # Clean up regions
        store.regions.clear()

        start = time.perf_counter()

        # Register multiple regions
        for i in range(num_regions):
            store.register_region(
                f"region_{i}",
                f"https://region-{i}.example.com",
                "bidirectional",
            )

        elapsed = (time.perf_counter() - start) * 1000
        latencies.append(elapsed)

    return create_benchmark_result(
        f"Region Registration ({num_regions} regions)",
        latencies,
        {"regions_registered": num_regions},
    )


async def benchmark_push_sync(
    store: MockFederationStore,
    items_per_sync: int,
    iterations: int,
) -> BenchmarkResult:
    """Benchmark push sync performance."""
    latencies = []
    total_items_synced = 0

    # Ensure we have a region
    store.register_region("test_region", "https://test.example.com", "push")

    for _ in range(iterations):
        start = time.perf_counter()

        # Get items and push
        items = await store.get_items_for_sync(
            ["organization", "public"],
            scope="full",
            limit=items_per_sync,
        )
        synced = await store.push_to_remote("test_region", items)
        total_items_synced += synced

        elapsed = (time.perf_counter() - start) * 1000
        latencies.append(elapsed)

    return create_benchmark_result(
        f"Push Sync ({items_per_sync} items)",
        latencies,
        {"total_items_synced": total_items_synced},
    )


async def benchmark_pull_sync(
    store: MockFederationStore,
    iterations: int,
) -> BenchmarkResult:
    """Benchmark pull sync performance."""
    latencies = []
    total_items_pulled = 0

    # Ensure we have a region
    store.register_region("test_region", "https://test.example.com", "pull")

    for _ in range(iterations):
        start = time.perf_counter()

        # Pull and ingest
        items = await store.pull_from_remote("test_region")
        ingested = await store.ingest_items(items)
        total_items_pulled += ingested

        elapsed = (time.perf_counter() - start) * 1000
        latencies.append(elapsed)

    return create_benchmark_result(
        "Pull Sync",
        latencies,
        {"total_items_pulled": total_items_pulled},
    )


async def benchmark_scope_filtering(
    store: MockFederationStore,
    iterations: int,
) -> BenchmarkResult:
    """Benchmark scope filtering overhead."""
    latencies_by_scope = {"full": [], "summary": [], "metadata": []}

    for scope in ["full", "summary", "metadata"]:
        for _ in range(iterations):
            start = time.perf_counter()
            await store.get_items_for_sync(
                ["organization", "public"],
                scope=scope,
                limit=500,
            )
            elapsed = (time.perf_counter() - start) * 1000
            latencies_by_scope[scope].append(elapsed)

    # Report combined results
    all_latencies = []
    for scope_latencies in latencies_by_scope.values():
        all_latencies.extend(scope_latencies)

    return create_benchmark_result(
        "Scope Filtering (full/summary/metadata)",
        all_latencies,
        {
            "full_avg_ms": statistics.mean(latencies_by_scope["full"]),
            "summary_avg_ms": statistics.mean(latencies_by_scope["summary"]),
            "metadata_avg_ms": statistics.mean(latencies_by_scope["metadata"]),
        },
    )


async def benchmark_bidirectional_sync(
    store: MockFederationStore,
    items_per_sync: int,
    iterations: int,
) -> BenchmarkResult:
    """Benchmark bidirectional sync (push + pull)."""
    latencies = []

    store.register_region("test_region", "https://test.example.com", "bidirectional")

    for _ in range(iterations):
        start = time.perf_counter()

        # Push
        items = await store.get_items_for_sync(
            ["organization", "public"],
            limit=items_per_sync,
        )
        await store.push_to_remote("test_region", items)

        # Pull
        pulled = await store.pull_from_remote("test_region")
        await store.ingest_items(pulled)

        elapsed = (time.perf_counter() - start) * 1000
        latencies.append(elapsed)

    return create_benchmark_result(
        f"Bidirectional Sync ({items_per_sync} items)",
        latencies,
    )


async def benchmark_multi_region_sync(
    store: MockFederationStore,
    num_regions: int,
    items_per_sync: int,
    iterations: int,
) -> BenchmarkResult:
    """Benchmark syncing to multiple regions."""
    latencies = []

    # Register multiple regions
    for i in range(num_regions):
        store.register_region(f"region_{i}", f"https://region-{i}.example.com", "push")

    for _ in range(iterations):
        start = time.perf_counter()

        items = await store.get_items_for_sync(
            ["organization", "public"],
            limit=items_per_sync,
        )

        # Sync to all regions concurrently
        tasks = [
            store.push_to_remote(f"region_{i}", items)
            for i in range(num_regions)
        ]
        await asyncio.gather(*tasks)

        elapsed = (time.perf_counter() - start) * 1000
        latencies.append(elapsed)

    return create_benchmark_result(
        f"Multi-Region Sync ({num_regions} regions, {items_per_sync} items)",
        latencies,
        {"regions": num_regions},
    )


async def benchmark_status_query(
    store: MockFederationStore,
    num_regions: int,
    iterations: int,
) -> BenchmarkResult:
    """Benchmark federation status queries."""
    latencies = []

    # Register regions with varying states
    for i in range(num_regions):
        region = store.register_region(
            f"region_{i}",
            f"https://region-{i}.example.com",
            "bidirectional",
        )
        region["last_sync_at"] = datetime.now().isoformat() if i % 2 == 0 else None
        region["enabled"] = i % 3 != 0

    for _ in range(iterations):
        start = time.perf_counter()

        # Build status report
        status = {}
        for region_id, region in store.regions.items():
            status[region_id] = {
                "endpoint_url": region["endpoint_url"],
                "mode": region["mode"],
                "enabled": region["enabled"],
                "last_sync_at": region["last_sync_at"],
            }

        elapsed = (time.perf_counter() - start) * 1000
        latencies.append(elapsed)

    return create_benchmark_result(
        f"Status Query ({num_regions} regions)",
        latencies,
    )


async def run_benchmarks(
    item_count: int = 1000,
    num_regions: int = 3,
    iterations: int = 100,
) -> List[BenchmarkResult]:
    """Run all federation sync benchmarks."""
    print(f"\n{'='*60}")
    print("Federation Sync Benchmarks")
    print(f"{'='*60}")
    print(f"Item count: {item_count}")
    print(f"Regions: {num_regions}")
    print(f"Iterations: {iterations}")
    print(f"{'='*60}\n")

    # Create mock store with test data
    store = MockFederationStore(item_count)

    results = []

    # Run benchmarks
    print("Running region registration benchmark...")
    result = await benchmark_region_registration(store, num_regions, iterations)
    print(result)
    results.append(result)
    print()

    print("Running push sync benchmark (100 items)...")
    result = await benchmark_push_sync(store, 100, iterations)
    print(result)
    results.append(result)
    print()

    print("Running push sync benchmark (500 items)...")
    result = await benchmark_push_sync(store, 500, iterations // 2)
    print(result)
    results.append(result)
    print()

    print("Running pull sync benchmark...")
    result = await benchmark_pull_sync(store, iterations)
    print(result)
    results.append(result)
    print()

    print("Running scope filtering benchmark...")
    result = await benchmark_scope_filtering(store, iterations // 3)
    print(result)
    results.append(result)
    print()

    print("Running bidirectional sync benchmark...")
    result = await benchmark_bidirectional_sync(store, 100, iterations // 2)
    print(result)
    results.append(result)
    print()

    print("Running multi-region sync benchmark...")
    result = await benchmark_multi_region_sync(store, num_regions, 100, iterations // 2)
    print(result)
    results.append(result)
    print()

    print("Running status query benchmark...")
    result = await benchmark_status_query(store, num_regions * 2, iterations)
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
    parser = argparse.ArgumentParser(description="Federation Sync Benchmarks")
    parser.add_argument(
        "--items",
        type=int,
        default=1000,
        help="Number of items to generate (default: 1000)",
    )
    parser.add_argument(
        "--regions",
        type=int,
        default=3,
        help="Number of regions (default: 3)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of iterations per benchmark (default: 100)",
    )
    args = parser.parse_args()

    asyncio.run(run_benchmarks(args.items, args.regions, args.iterations))


if __name__ == "__main__":
    main()
