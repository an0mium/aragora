"""
Tenant Isolation Benchmarks.

Measures:
- Context switch overhead
- Query filtering performance
- Concurrent tenant request handling
- Quota check latency
"""

from __future__ import annotations

import asyncio
import gc
import statistics
import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Any, Dict, List
import uuid


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
        return {
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


async def benchmark_context_switch(iterations: int = 100) -> BenchmarkResult:
    """Benchmark tenant context switch overhead."""
    from aragora.tenancy import TenantContext, get_current_tenant

    result = BenchmarkResult(name="context_switch", iterations=iterations)

    tenants = [f"tenant-{uuid.uuid4().hex[:8]}" for _ in range(10)]

    for i in range(iterations):
        tenant_id = tenants[i % len(tenants)]

        gc.collect()
        tracemalloc.start()

        start = time.perf_counter()
        with TenantContext(tenant_id):
            current = get_current_tenant()
            assert current == tenant_id
        elapsed = (time.perf_counter() - start) * 1000

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        result.times_ms.append(elapsed)
        result.memory_mb.append(peak / 1024 / 1024)

    return result


async def benchmark_query_filtering(iterations: int = 100) -> BenchmarkResult:
    """Benchmark SQL query tenant filtering."""
    from aragora.tenancy.isolation import TenantDataIsolation
    from aragora.tenancy import TenantContext

    result = BenchmarkResult(name="query_filtering", iterations=iterations)

    isolation = TenantDataIsolation()
    tenant_id = f"tenant-{uuid.uuid4().hex[:8]}"

    queries = [
        "SELECT * FROM facts WHERE type = 'knowledge'",
        "SELECT id, content FROM documents WHERE status = 'active'",
        "SELECT * FROM debates WHERE created_at > '2024-01-01'",
        "SELECT COUNT(*) FROM events WHERE event_type = 'sync'",
    ]

    with TenantContext(tenant_id):
        for i in range(iterations):
            query = queries[i % len(queries)]

            gc.collect()

            start = time.perf_counter()
            filtered = isolation.apply_tenant_filter(query)
            elapsed = (time.perf_counter() - start) * 1000

            result.times_ms.append(elapsed)

            # Verify tenant filter was added
            assert tenant_id in filtered

    return result


async def benchmark_concurrent_tenants(iterations: int = 100) -> BenchmarkResult:
    """Benchmark concurrent multi-tenant request handling."""
    from aragora.tenancy import TenantContext, get_current_tenant

    result = BenchmarkResult(name="concurrent_tenants", iterations=iterations)
    result.custom_metrics["concurrent_requests"] = []

    tenants = [f"tenant-{uuid.uuid4().hex[:8]}" for _ in range(100)]

    async def tenant_request(tenant_id: str) -> str:
        with TenantContext(tenant_id):
            await asyncio.sleep(0.001)  # Simulate work
            return get_current_tenant()

    concurrency_levels = [10, 50, 100]

    for concurrency in concurrency_levels:
        for _ in range(iterations // len(concurrency_levels)):
            selected_tenants = tenants[:concurrency]

            gc.collect()
            tracemalloc.start()

            start = time.perf_counter()
            results = await asyncio.gather(*[
                tenant_request(t) for t in selected_tenants
            ])
            elapsed = (time.perf_counter() - start) * 1000

            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            # Verify isolation
            for i, r in enumerate(results):
                assert r == selected_tenants[i]

            result.times_ms.append(elapsed)
            result.memory_mb.append(peak / 1024 / 1024)
            result.custom_metrics["concurrent_requests"].append(concurrency)

    return result


async def benchmark_quota_check(iterations: int = 100) -> BenchmarkResult:
    """Benchmark quota check latency."""
    from aragora.tenancy import TenantContext
    from aragora.tenancy.quotas import QuotaManager

    result = BenchmarkResult(name="quota_check", iterations=iterations)

    manager = QuotaManager()
    tenant_id = f"tenant-{uuid.uuid4().hex[:8]}"

    # Configure tenant quotas
    manager.configure_tenant(tenant_id, {
        "api_requests_per_minute": 1000,
        "storage_bytes": 1024 * 1024 * 100,  # 100MB
        "max_debate_rounds": 50,
    })

    quota_types = ["api_requests", "storage", "debate_rounds"]

    with TenantContext(tenant_id):
        for i in range(iterations):
            quota_type = quota_types[i % len(quota_types)]

            gc.collect()

            start = time.perf_counter()
            try:
                await manager.check_quota(quota_type)
            except Exception:
                pass  # Quota may be exceeded
            elapsed = (time.perf_counter() - start) * 1000

            result.times_ms.append(elapsed)

    return result


async def run_tenant_benchmarks(iterations: int = 100, warmup: int = 10) -> Dict[str, Any]:
    """Run all tenant isolation benchmarks."""
    print("Running Tenant Isolation Benchmarks...")
    print("-" * 40)

    results = {}

    # Warmup
    print(f"Warming up ({warmup} iterations)...")
    await benchmark_context_switch(warmup)

    benchmarks = [
        ("Context Switch", benchmark_context_switch),
        ("Query Filtering", benchmark_query_filtering),
        ("Concurrent Tenants", benchmark_concurrent_tenants),
        ("Quota Check", benchmark_quota_check),
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
    results = asyncio.run(run_tenant_benchmarks())
    import json
    print(json.dumps(results, indent=2))
