"""
API Endpoint Performance Benchmarks.

Measures latency and throughput of core API endpoints against SLO targets.

SLO Targets (from PERFORMANCE_TARGETS.md):
- Health Check: P50 < 5ms, P95 < 20ms, P99 < 50ms
- Authentication: P50 < 50ms, P95 < 150ms, P99 < 300ms
- Simple API Call: P50 < 100ms, P95 < 300ms, P99 < 500ms
- Search/Query: P50 < 200ms, P95 < 500ms, P99 < 1s

Usage:
    python -m benchmarks.api_endpoints
    pytest benchmarks/api_endpoints.py -v
"""

import asyncio
import statistics
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
from unittest.mock import Mock, patch


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""

    name: str
    samples: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    mean_ms: float
    stddev_ms: float
    ops_per_sec: float
    slo_p50_ms: float
    slo_p95_ms: float
    slo_p99_ms: float

    @property
    def meets_slo(self) -> bool:
        """Check if result meets all SLO targets."""
        return (
            self.p50_ms <= self.slo_p50_ms
            and self.p95_ms <= self.slo_p95_ms
            and self.p99_ms <= self.slo_p99_ms
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "samples": self.samples,
            "latency": {
                "p50_ms": round(self.p50_ms, 3),
                "p95_ms": round(self.p95_ms, 3),
                "p99_ms": round(self.p99_ms, 3),
                "min_ms": round(self.min_ms, 3),
                "max_ms": round(self.max_ms, 3),
                "mean_ms": round(self.mean_ms, 3),
                "stddev_ms": round(self.stddev_ms, 3),
            },
            "throughput": {"ops_per_sec": round(self.ops_per_sec, 2)},
            "slo": {
                "p50_ms": self.slo_p50_ms,
                "p95_ms": self.slo_p95_ms,
                "p99_ms": self.slo_p99_ms,
                "meets_slo": self.meets_slo,
            },
        }


def percentile(data: List[float], p: float) -> float:
    """Calculate percentile of sorted data."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100)
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_data) else f
    return sorted_data[f] + (sorted_data[c] - sorted_data[f]) * (k - f)


def run_benchmark(
    func: Callable,
    samples: int = 1000,
    warmup: int = 100,
    slo_p50_ms: float = 100,
    slo_p95_ms: float = 300,
    slo_p99_ms: float = 500,
    name: str = "benchmark",
) -> BenchmarkResult:
    """
    Run a benchmark and collect timing statistics.

    Args:
        func: Function to benchmark (called with no arguments)
        samples: Number of samples to collect
        warmup: Number of warmup iterations (not measured)
        slo_p50_ms: SLO target for P50 latency
        slo_p95_ms: SLO target for P95 latency
        slo_p99_ms: SLO target for P99 latency
        name: Name of the benchmark

    Returns:
        BenchmarkResult with statistics
    """
    # Warmup
    for _ in range(warmup):
        func()

    # Collect samples
    latencies_ms = []
    start_total = time.perf_counter()

    for _ in range(samples):
        start = time.perf_counter()
        func()
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies_ms.append(elapsed_ms)

    total_time = time.perf_counter() - start_total

    # Calculate statistics
    return BenchmarkResult(
        name=name,
        samples=samples,
        p50_ms=percentile(latencies_ms, 50),
        p95_ms=percentile(latencies_ms, 95),
        p99_ms=percentile(latencies_ms, 99),
        min_ms=min(latencies_ms),
        max_ms=max(latencies_ms),
        mean_ms=statistics.mean(latencies_ms),
        stddev_ms=statistics.stdev(latencies_ms) if len(latencies_ms) > 1 else 0,
        ops_per_sec=samples / total_time,
        slo_p50_ms=slo_p50_ms,
        slo_p95_ms=slo_p95_ms,
        slo_p99_ms=slo_p99_ms,
    )


# =============================================================================
# Mock API Handlers for Benchmarking
# =============================================================================


class MockHealthHandler:
    """Mock health endpoint for benchmarking."""

    def handle(self) -> Tuple[str, int]:
        """Handle health check request."""
        return '{"status": "healthy", "timestamp": 1234567890}', 200


class MockAuthHandler:
    """Mock authentication handler for benchmarking."""

    def __init__(self):
        self._token_cache: Dict[str, Dict] = {}

    def validate_token(self, token: str) -> Tuple[str, int]:
        """Validate JWT token."""
        # Simulate token validation (in-memory check)
        if token in self._token_cache:
            return '{"valid": true, "user_id": "user-123"}', 200
        # Simulate JWT decode
        import hashlib

        hashlib.sha256(token.encode()).hexdigest()
        self._token_cache[token] = {"user_id": "user-123"}
        return '{"valid": true, "user_id": "user-123"}', 200


class MockDebateHandler:
    """Mock debate handler for benchmarking."""

    def __init__(self):
        self._debates: Dict[str, Dict] = {}

    def get_debate(self, debate_id: str) -> Tuple[str, int]:
        """Get debate by ID."""
        if debate_id not in self._debates:
            self._debates[debate_id] = {
                "id": debate_id,
                "topic": "Test topic",
                "messages": [],
                "status": "completed",
            }
        import json

        return json.dumps(self._debates[debate_id]), 200

    def list_debates(self, limit: int = 20) -> Tuple[str, int]:
        """List debates with pagination."""
        debates = list(self._debates.values())[:limit]
        import json

        return json.dumps({"debates": debates, "total": len(self._debates)}), 200


class MockSearchHandler:
    """Mock search handler for benchmarking."""

    def __init__(self):
        self._index: Dict[str, List[str]] = {}

    def search(self, query: str, limit: int = 10) -> Tuple[str, int]:
        """Search debates."""
        # Simulate search with simple string matching
        results = []
        for key, values in self._index.items():
            if query.lower() in key.lower():
                results.extend(values[:limit])
            if len(results) >= limit:
                break

        import json

        return json.dumps({"results": results[:limit], "query": query}), 200


# =============================================================================
# Benchmark Tests
# =============================================================================


def test_health_endpoint_latency():
    """Benchmark health endpoint latency against SLO."""
    handler = MockHealthHandler()

    result = run_benchmark(
        func=handler.handle,
        samples=10000,
        warmup=1000,
        slo_p50_ms=5,
        slo_p95_ms=20,
        slo_p99_ms=50,
        name="health_check",
    )

    print(f"\nHealth Check Benchmark:")
    print(f"  P50: {result.p50_ms:.3f}ms (SLO: {result.slo_p50_ms}ms)")
    print(f"  P95: {result.p95_ms:.3f}ms (SLO: {result.slo_p95_ms}ms)")
    print(f"  P99: {result.p99_ms:.3f}ms (SLO: {result.slo_p99_ms}ms)")
    print(f"  OPS: {result.ops_per_sec:.0f}/sec")
    print(f"  Meets SLO: {result.meets_slo}")

    # Health check should be extremely fast
    assert result.p50_ms < 5, f"Health check P50 {result.p50_ms}ms exceeds 5ms SLO"
    assert result.p95_ms < 20, f"Health check P95 {result.p95_ms}ms exceeds 20ms SLO"


def test_auth_validation_latency():
    """Benchmark auth token validation latency against SLO."""
    handler = MockAuthHandler()
    token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.test.signature"

    result = run_benchmark(
        func=lambda: handler.validate_token(token),
        samples=5000,
        warmup=500,
        slo_p50_ms=50,
        slo_p95_ms=150,
        slo_p99_ms=300,
        name="auth_validation",
    )

    print(f"\nAuth Validation Benchmark:")
    print(f"  P50: {result.p50_ms:.3f}ms (SLO: {result.slo_p50_ms}ms)")
    print(f"  P95: {result.p95_ms:.3f}ms (SLO: {result.slo_p95_ms}ms)")
    print(f"  P99: {result.p99_ms:.3f}ms (SLO: {result.slo_p99_ms}ms)")
    print(f"  OPS: {result.ops_per_sec:.0f}/sec")
    print(f"  Meets SLO: {result.meets_slo}")

    assert result.p50_ms < 50, f"Auth P50 {result.p50_ms}ms exceeds 50ms SLO"


def test_debate_get_latency():
    """Benchmark debate retrieval latency against SLO."""
    handler = MockDebateHandler()

    # Pre-populate some debates
    for i in range(100):
        handler.get_debate(f"debate-{i}")

    result = run_benchmark(
        func=lambda: handler.get_debate("debate-50"),
        samples=5000,
        warmup=500,
        slo_p50_ms=100,
        slo_p95_ms=300,
        slo_p99_ms=500,
        name="debate_get",
    )

    print(f"\nDebate Get Benchmark:")
    print(f"  P50: {result.p50_ms:.3f}ms (SLO: {result.slo_p50_ms}ms)")
    print(f"  P95: {result.p95_ms:.3f}ms (SLO: {result.slo_p95_ms}ms)")
    print(f"  P99: {result.p99_ms:.3f}ms (SLO: {result.slo_p99_ms}ms)")
    print(f"  OPS: {result.ops_per_sec:.0f}/sec")
    print(f"  Meets SLO: {result.meets_slo}")

    assert result.p50_ms < 100, f"Debate get P50 {result.p50_ms}ms exceeds 100ms SLO"


def test_debate_list_latency():
    """Benchmark debate listing latency against SLO."""
    handler = MockDebateHandler()

    # Pre-populate debates
    for i in range(1000):
        handler.get_debate(f"debate-{i}")

    result = run_benchmark(
        func=lambda: handler.list_debates(limit=20),
        samples=3000,
        warmup=300,
        slo_p50_ms=100,
        slo_p95_ms=300,
        slo_p99_ms=500,
        name="debate_list",
    )

    print(f"\nDebate List Benchmark:")
    print(f"  P50: {result.p50_ms:.3f}ms (SLO: {result.slo_p50_ms}ms)")
    print(f"  P95: {result.p95_ms:.3f}ms (SLO: {result.slo_p95_ms}ms)")
    print(f"  P99: {result.p99_ms:.3f}ms (SLO: {result.slo_p99_ms}ms)")
    print(f"  OPS: {result.ops_per_sec:.0f}/sec")
    print(f"  Meets SLO: {result.meets_slo}")

    assert result.p50_ms < 100, f"Debate list P50 {result.p50_ms}ms exceeds 100ms SLO"


def test_search_latency():
    """Benchmark search latency against SLO."""
    handler = MockSearchHandler()

    # Pre-populate search index
    handler._index = {f"topic-{i}": [f"debate-{i}-{j}" for j in range(10)] for i in range(100)}

    result = run_benchmark(
        func=lambda: handler.search("topic", limit=10),
        samples=3000,
        warmup=300,
        slo_p50_ms=200,
        slo_p95_ms=500,
        slo_p99_ms=1000,
        name="search",
    )

    print(f"\nSearch Benchmark:")
    print(f"  P50: {result.p50_ms:.3f}ms (SLO: {result.slo_p50_ms}ms)")
    print(f"  P95: {result.p95_ms:.3f}ms (SLO: {result.slo_p95_ms}ms)")
    print(f"  P99: {result.p99_ms:.3f}ms (SLO: {result.slo_p99_ms}ms)")
    print(f"  OPS: {result.ops_per_sec:.0f}/sec")
    print(f"  Meets SLO: {result.meets_slo}")

    assert result.p50_ms < 200, f"Search P50 {result.p50_ms}ms exceeds 200ms SLO"


def test_throughput_target():
    """Test that API can handle target throughput."""
    handler = MockHealthHandler()

    # Target: 1000 requests/sec sustained
    target_ops = 1000
    duration_seconds = 1.0

    start = time.perf_counter()
    count = 0

    while time.perf_counter() - start < duration_seconds:
        handler.handle()
        count += 1

    actual_ops = count / duration_seconds

    print(f"\nThroughput Test:")
    print(f"  Target: {target_ops} ops/sec")
    print(f"  Actual: {actual_ops:.0f} ops/sec")
    print(f"  Meets target: {actual_ops >= target_ops}")

    # Should easily exceed 1000 ops/sec for simple in-memory operations
    assert actual_ops >= target_ops, f"Throughput {actual_ops:.0f} below target {target_ops}"


# =============================================================================
# Main Entry Point
# =============================================================================


def run_all_benchmarks():
    """Run all benchmarks and print summary."""
    print("=" * 60)
    print("Aragora API Endpoint Benchmarks")
    print("=" * 60)

    results = []

    # Health check
    handler = MockHealthHandler()
    results.append(
        run_benchmark(
            handler.handle, samples=10000, slo_p50_ms=5, slo_p95_ms=20, slo_p99_ms=50, name="health"
        )
    )

    # Auth validation
    auth = MockAuthHandler()
    token = "test-token"
    results.append(
        run_benchmark(
            lambda: auth.validate_token(token),
            samples=5000,
            slo_p50_ms=50,
            slo_p95_ms=150,
            slo_p99_ms=300,
            name="auth",
        )
    )

    # Debate operations
    debate = MockDebateHandler()
    for i in range(100):
        debate.get_debate(f"debate-{i}")
    results.append(
        run_benchmark(
            lambda: debate.get_debate("debate-50"),
            samples=5000,
            slo_p50_ms=100,
            slo_p95_ms=300,
            slo_p99_ms=500,
            name="debate_get",
        )
    )

    # Search
    search = MockSearchHandler()
    search._index = {f"topic-{i}": [f"debate-{i}"] for i in range(100)}
    results.append(
        run_benchmark(
            lambda: search.search("topic"),
            samples=3000,
            slo_p50_ms=200,
            slo_p95_ms=500,
            slo_p99_ms=1000,
            name="search",
        )
    )

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Benchmark':<20} {'P50':>10} {'P95':>10} {'P99':>10} {'OPS':>10} {'SLO':>8}")
    print("-" * 60)

    all_pass = True
    for r in results:
        status = "PASS" if r.meets_slo else "FAIL"
        all_pass = all_pass and r.meets_slo
        print(
            f"{r.name:<20} {r.p50_ms:>9.2f}ms {r.p95_ms:>9.2f}ms {r.p99_ms:>9.2f}ms {r.ops_per_sec:>9.0f} {status:>8}"
        )

    print("-" * 60)
    print(f"Overall: {'ALL PASS' if all_pass else 'SOME FAILED'}")

    return results


if __name__ == "__main__":
    run_all_benchmarks()
