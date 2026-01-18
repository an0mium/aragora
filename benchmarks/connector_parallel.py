"""
Connector Parallelization Benchmarks.

Measures:
- Parallel sync throughput
- Connection pool efficiency
- Rate limiting impact
- Error recovery overhead
"""

from __future__ import annotations

import asyncio
import gc
import statistics
import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch


@dataclass
class BenchmarkResult:
    name: str
    iterations: int
    times_ms: List[float] = field(default_factory=list)
    memory_mb: List[float] = field(default_factory=list)
    custom_metrics: Dict[str, List[float]] = field(default_factory=dict)

    @property
    def p50_ms(self) -> float:
        return statistics.median(self.times_ms) if self.times_ms else 0

    @property
    def p95_ms(self) -> float:
        if not self.times_ms:
            return 0
        idx = int(len(self.times_ms) * 0.95)
        return sorted(self.times_ms)[idx]

    @property
    def p99_ms(self) -> float:
        if not self.times_ms:
            return 0
        idx = int(len(self.times_ms) * 0.99)
        return sorted(self.times_ms)[idx]

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "name": self.name,
            "iterations": self.iterations,
            "latency_ms": {
                "p50": round(self.p50_ms, 3),
                "p95": round(self.p95_ms, 3),
                "p99": round(self.p99_ms, 3),
            },
            "memory_mb": {
                "avg": round(statistics.mean(self.memory_mb), 2) if self.memory_mb else 0,
                "peak": round(max(self.memory_mb), 2) if self.memory_mb else 0,
            },
        }
        for key, values in self.custom_metrics.items():
            if values:
                result[key] = {
                    "avg": round(statistics.mean(values), 3),
                    "max": round(max(values), 3),
                }
        return result


class MockConnector:
    """Mock connector for benchmarking."""

    def __init__(self, name: str, latency_ms: float = 10):
        self.name = name
        self.latency_ms = latency_ms
        self.sync_count = 0

    async def initialize(self) -> bool:
        await asyncio.sleep(self.latency_ms / 1000)
        return True

    async def sync(self) -> Dict[str, Any]:
        await asyncio.sleep(self.latency_ms / 1000)
        self.sync_count += 1
        return {
            "status": "success",
            "items_synced": 100,
            "connector": self.name,
        }


async def benchmark_parallel_sync(iterations: int = 50) -> BenchmarkResult:
    """Benchmark parallel connector sync throughput."""
    result = BenchmarkResult(name="parallel_sync", iterations=iterations)
    result.custom_metrics["connectors_synced"] = []
    result.custom_metrics["items_per_second"] = []

    parallelism_levels = [2, 5, 10, 20]

    for parallelism in parallelism_levels:
        for _ in range(iterations // len(parallelism_levels)):
            connectors = [
                MockConnector(f"connector-{i}", latency_ms=5)
                for i in range(parallelism)
            ]

            gc.collect()
            tracemalloc.start()

            start = time.perf_counter()

            # Initialize all
            await asyncio.gather(*[c.initialize() for c in connectors])

            # Sync all
            results = await asyncio.gather(*[c.sync() for c in connectors])

            elapsed = time.perf_counter() - start
            elapsed_ms = elapsed * 1000

            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            total_items = sum(r["items_synced"] for r in results)
            items_per_sec = total_items / elapsed if elapsed > 0 else 0

            result.times_ms.append(elapsed_ms)
            result.memory_mb.append(peak / 1024 / 1024)
            result.custom_metrics["connectors_synced"].append(parallelism)
            result.custom_metrics["items_per_second"].append(items_per_sec)

    return result


async def benchmark_connection_pool(iterations: int = 100) -> BenchmarkResult:
    """Benchmark connection pool efficiency."""
    result = BenchmarkResult(name="connection_pool", iterations=iterations)
    result.custom_metrics["pool_utilization"] = []

    # Simulate connection pool
    class ConnectionPool:
        def __init__(self, size: int):
            self.size = size
            self.available = asyncio.Semaphore(size)
            self.in_use = 0

        async def acquire(self):
            await self.available.acquire()
            self.in_use += 1

        def release(self):
            self.in_use -= 1
            self.available.release()

        @property
        def utilization(self) -> float:
            return self.in_use / self.size

    pool_sizes = [5, 10, 20]

    for pool_size in pool_sizes:
        for _ in range(iterations // len(pool_sizes)):
            pool = ConnectionPool(pool_size)

            async def work_item():
                await pool.acquire()
                try:
                    await asyncio.sleep(0.001)  # Simulate work
                finally:
                    pool.release()

            gc.collect()

            start = time.perf_counter()

            # Run concurrent work items
            await asyncio.gather(*[work_item() for _ in range(50)])

            elapsed_ms = (time.perf_counter() - start) * 1000

            result.times_ms.append(elapsed_ms)
            result.custom_metrics["pool_utilization"].append(pool_size)

    return result


async def benchmark_rate_limiting(iterations: int = 100) -> BenchmarkResult:
    """Benchmark rate limiting overhead."""
    result = BenchmarkResult(name="rate_limiting", iterations=iterations)
    result.custom_metrics["requests_throttled"] = []

    class RateLimiter:
        def __init__(self, rate: int, per_seconds: float = 1.0):
            self.rate = rate
            self.per_seconds = per_seconds
            self.tokens = rate
            self.last_update = time.monotonic()
            self.throttled = 0

        async def acquire(self) -> bool:
            now = time.monotonic()
            elapsed = now - self.last_update
            self.tokens = min(self.rate, self.tokens + elapsed * self.rate / self.per_seconds)
            self.last_update = now

            if self.tokens >= 1:
                self.tokens -= 1
                return True
            else:
                self.throttled += 1
                await asyncio.sleep(0.001)
                return False

    for _ in range(iterations):
        limiter = RateLimiter(rate=100, per_seconds=0.1)

        gc.collect()

        start = time.perf_counter()

        # Try 200 requests
        for _ in range(200):
            await limiter.acquire()

        elapsed_ms = (time.perf_counter() - start) * 1000

        result.times_ms.append(elapsed_ms)
        result.custom_metrics["requests_throttled"].append(limiter.throttled)

    return result


async def benchmark_error_recovery(iterations: int = 50) -> BenchmarkResult:
    """Benchmark error recovery overhead."""
    result = BenchmarkResult(name="error_recovery", iterations=iterations)
    result.custom_metrics["retries_performed"] = []

    class RetryingConnector:
        def __init__(self, fail_rate: float = 0.3, max_retries: int = 3):
            self.fail_rate = fail_rate
            self.max_retries = max_retries
            self.retries = 0

        async def sync_with_retry(self) -> Dict[str, Any]:
            import random

            for attempt in range(self.max_retries):
                if random.random() > self.fail_rate:
                    return {"status": "success", "attempt": attempt + 1}

                self.retries += 1
                await asyncio.sleep(0.001 * (2 ** attempt))  # Exponential backoff

            return {"status": "failed", "attempts": self.max_retries}

    for _ in range(iterations):
        connector = RetryingConnector(fail_rate=0.3, max_retries=3)

        gc.collect()

        start = time.perf_counter()

        # Run 10 syncs
        results = await asyncio.gather(*[
            connector.sync_with_retry() for _ in range(10)
        ])

        elapsed_ms = (time.perf_counter() - start) * 1000

        result.times_ms.append(elapsed_ms)
        result.custom_metrics["retries_performed"].append(connector.retries)

    return result


async def run_connector_benchmarks(iterations: int = 100, warmup: int = 10) -> Dict[str, Any]:
    """Run all connector parallelization benchmarks."""
    print("Running Connector Parallelization Benchmarks...")
    print("-" * 40)

    results = {}

    # Warmup
    print(f"Warming up ({warmup} iterations)...")
    await benchmark_parallel_sync(warmup)

    benchmarks = [
        ("Parallel Sync", benchmark_parallel_sync),
        ("Connection Pool", benchmark_connection_pool),
        ("Rate Limiting", benchmark_rate_limiting),
        ("Error Recovery", benchmark_error_recovery),
    ]

    for name, bench_func in benchmarks:
        print(f"  Running: {name}...")
        try:
            result = await bench_func(iterations)
            results[result.name] = result.to_dict()
            print(f"    p50: {result.p50_ms:.2f}ms, p99: {result.p99_ms:.2f}ms")
        except Exception as e:
            print(f"    Failed: {e}")
            results[name.lower().replace(" ", "_")] = {"error": str(e)}

    return results


if __name__ == "__main__":
    results = asyncio.run(run_connector_benchmarks())
    import json
    print(json.dumps(results, indent=2))
