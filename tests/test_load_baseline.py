"""
Load test baseline for Aragora server.

Establishes baseline performance metrics for:
- Concurrent API requests
- Request latency under load
- Memory usage patterns
- WebSocket connections

These tests are marked with @pytest.mark.load and can be run with:
    pytest tests/test_load_baseline.py -v -m load

Note: Requires a running server for full tests. Unit tests (no server needed)
test concurrency primitives and baseline measurements.
"""

import asyncio
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, Optional
from unittest.mock import MagicMock, patch

import pytest


@dataclass
class LoadTestResult:
    """Results from a load test run."""

    total_requests: int
    successful_requests: int
    failed_requests: int
    total_duration_seconds: float
    latencies_ms: list[float] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests

    @property
    def requests_per_second(self) -> float:
        if self.total_duration_seconds == 0:
            return 0.0
        return self.successful_requests / self.total_duration_seconds

    @property
    def avg_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        return statistics.mean(self.latencies_ms)

    @property
    def p50_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        return statistics.median(self.latencies_ms)

    @property
    def p95_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        sorted_latencies = sorted(self.latencies_ms)
        idx = int(len(sorted_latencies) * 0.95)
        return sorted_latencies[min(idx, len(sorted_latencies) - 1)]

    @property
    def p99_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        sorted_latencies = sorted(self.latencies_ms)
        idx = int(len(sorted_latencies) * 0.99)
        return sorted_latencies[min(idx, len(sorted_latencies) - 1)]

    def summary(self) -> str:
        return (
            f"Requests: {self.successful_requests}/{self.total_requests} "
            f"({self.success_rate*100:.1f}% success)\n"
            f"Duration: {self.total_duration_seconds:.2f}s "
            f"({self.requests_per_second:.1f} req/s)\n"
            f"Latency: avg={self.avg_latency_ms:.1f}ms, "
            f"p50={self.p50_latency_ms:.1f}ms, "
            f"p95={self.p95_latency_ms:.1f}ms, "
            f"p99={self.p99_latency_ms:.1f}ms"
        )


def run_concurrent_load_test(
    request_fn: Callable[[], bool],
    num_requests: int,
    max_workers: int = 10,
) -> LoadTestResult:
    """Run a concurrent load test.

    Args:
        request_fn: Function that makes a request and returns (success, latency_ms)
        num_requests: Total number of requests to make
        max_workers: Maximum concurrent workers

    Returns:
        LoadTestResult with metrics
    """
    result = LoadTestResult(
        total_requests=num_requests,
        successful_requests=0,
        failed_requests=0,
        total_duration_seconds=0.0,
        latencies_ms=[],
    )

    start_time = time.perf_counter()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(request_fn) for _ in range(num_requests)]

        for future in as_completed(futures):
            try:
                success, latency_ms = future.result()
                if success:
                    result.successful_requests += 1
                    result.latencies_ms.append(latency_ms)
                else:
                    result.failed_requests += 1
            except Exception:
                result.failed_requests += 1

    result.total_duration_seconds = time.perf_counter() - start_time
    return result


async def run_async_load_test(
    request_fn: Callable[[], bool],
    num_requests: int,
    concurrency: int = 10,
) -> LoadTestResult:
    """Run an async concurrent load test.

    Args:
        request_fn: Async function that returns (success, latency_ms)
        num_requests: Total number of requests to make
        concurrency: Maximum concurrent tasks

    Returns:
        LoadTestResult with metrics
    """
    result = LoadTestResult(
        total_requests=num_requests,
        successful_requests=0,
        failed_requests=0,
        total_duration_seconds=0.0,
        latencies_ms=[],
    )

    semaphore = asyncio.Semaphore(concurrency)

    async def limited_request():
        async with semaphore:
            return await request_fn()

    start_time = time.perf_counter()

    tasks = [asyncio.create_task(limited_request()) for _ in range(num_requests)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for res in results:
        if isinstance(res, Exception):
            result.failed_requests += 1
        else:
            success, latency_ms = res
            if success:
                result.successful_requests += 1
                result.latencies_ms.append(latency_ms)
            else:
                result.failed_requests += 1

    result.total_duration_seconds = time.perf_counter() - start_time
    return result


# =============================================================================
# Unit Tests (no server required)
# =============================================================================


class TestLoadTestResult:
    """Tests for LoadTestResult metrics calculations."""

    def test_success_rate_calculation(self):
        result = LoadTestResult(
            total_requests=100,
            successful_requests=95,
            failed_requests=5,
            total_duration_seconds=1.0,
        )
        assert result.success_rate == 0.95

    def test_success_rate_zero_requests(self):
        result = LoadTestResult(
            total_requests=0,
            successful_requests=0,
            failed_requests=0,
            total_duration_seconds=0.0,
        )
        assert result.success_rate == 0.0

    def test_requests_per_second(self):
        result = LoadTestResult(
            total_requests=100,
            successful_requests=100,
            failed_requests=0,
            total_duration_seconds=2.0,
        )
        assert result.requests_per_second == 50.0

    def test_latency_percentiles(self):
        result = LoadTestResult(
            total_requests=100,
            successful_requests=100,
            failed_requests=0,
            total_duration_seconds=1.0,
            latencies_ms=list(range(1, 101)),  # 1-100ms
        )
        assert result.avg_latency_ms == 50.5
        assert result.p50_latency_ms == 50.5
        # p95 at index 95 = value 96 (0-indexed list of 1-100)
        assert result.p95_latency_ms == 96
        assert result.p99_latency_ms == 100

    def test_summary_output(self):
        result = LoadTestResult(
            total_requests=100,
            successful_requests=95,
            failed_requests=5,
            total_duration_seconds=1.0,
            latencies_ms=[10, 20, 30, 40, 50],
        )
        summary = result.summary()
        assert "95/100" in summary
        assert "95.0% success" in summary


class TestConcurrentLoadRunner:
    """Tests for concurrent load test execution."""

    def test_synchronous_load_test(self):
        """Test thread-based concurrent load runner."""
        call_count = 0

        def mock_request():
            nonlocal call_count
            call_count += 1
            time.sleep(0.001)  # Simulate 1ms latency
            return True, 1.0

        result = run_concurrent_load_test(
            mock_request,
            num_requests=20,
            max_workers=5,
        )

        assert call_count == 20
        assert result.total_requests == 20
        assert result.successful_requests == 20
        assert result.success_rate == 1.0

    def test_load_test_with_failures(self):
        """Test load runner handles failures correctly."""
        counter = {"count": 0}

        def flaky_request():
            counter["count"] += 1
            # Fail every 5th request
            if counter["count"] % 5 == 0:
                return False, 0.0
            return True, 1.0

        result = run_concurrent_load_test(
            flaky_request,
            num_requests=20,
            max_workers=5,
        )

        assert result.total_requests == 20
        assert result.failed_requests == 4  # 4 failures out of 20
        assert result.successful_requests == 16

    @pytest.mark.asyncio
    async def test_async_load_test(self):
        """Test async concurrent load runner."""
        call_count = 0

        async def mock_async_request():
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.001)
            return True, 1.0

        result = await run_async_load_test(
            mock_async_request,
            num_requests=20,
            concurrency=5,
        )

        assert call_count == 20
        assert result.total_requests == 20
        assert result.successful_requests == 20


# =============================================================================
# Integration Load Tests (require mocked or real server)
# =============================================================================


class TestAPIEndpointLatency:
    """Baseline latency tests for API endpoints using mocks."""

    def test_handler_routing_baseline(self):
        """Test baseline routing performance."""
        from aragora.server.handlers.base import BaseHandler

        # Mock handler
        class TestHandler(BaseHandler):
            ROUTES = ["/api/test"]

            def can_handle(self, path: str) -> bool:
                return path == "/api/test"

            def handle(self, path, query_params, handler):
                return {"status": 200, "body": b'{"ok": true}'}

        handler = TestHandler({})

        # Measure routing overhead
        iterations = 1000
        start = time.perf_counter()
        for _ in range(iterations):
            handler.can_handle("/api/test")
            handler.handle("/api/test", {}, None)
        elapsed = time.perf_counter() - start

        avg_latency_us = (elapsed / iterations) * 1_000_000
        print(f"\nHandler routing baseline: {avg_latency_us:.2f}µs per request")

        # Assert reasonable baseline (< 100µs per request)
        assert avg_latency_us < 100, f"Routing overhead too high: {avg_latency_us}µs"

    def test_json_serialization_baseline(self):
        """Test JSON serialization performance."""
        from aragora.server.handlers.base import json_response

        test_data = {
            "debates": [
                {"id": f"debate-{i}", "topic": "Test topic", "status": "completed"}
                for i in range(100)
            ],
            "total": 100,
        }

        iterations = 1000
        start = time.perf_counter()
        for _ in range(iterations):
            json_response(test_data)
        elapsed = time.perf_counter() - start

        avg_latency_us = (elapsed / iterations) * 1_000_000
        print(f"\nJSON serialization baseline: {avg_latency_us:.2f}µs per response")

        # Assert reasonable baseline (< 500µs for 100-item response)
        assert avg_latency_us < 500, f"JSON serialization too slow: {avg_latency_us}µs"


class TestCachePerformance:
    """Baseline performance tests for caching."""

    def test_ttl_cache_performance(self):
        """Test TTL cache hit performance."""
        from aragora.server.handlers.base import ttl_cache

        @ttl_cache(ttl_seconds=60, key_prefix="test")
        def expensive_function(x: int) -> int:
            return x * x

        # Warm up
        expensive_function(5)

        # Measure cache hits
        iterations = 10000
        start = time.perf_counter()
        for _ in range(iterations):
            expensive_function(5)  # Same key = cache hit
        elapsed = time.perf_counter() - start

        avg_latency_ns = (elapsed / iterations) * 1_000_000_000
        print(f"\nCache hit baseline: {avg_latency_ns:.2f}ns per lookup")

        # Assert reasonable baseline (< 10µs per cache hit)
        assert avg_latency_ns < 10000, f"Cache hit too slow: {avg_latency_ns}ns"


class TestDatabaseQueryBaseline:
    """Baseline tests for database operations."""

    def test_sqlite_insert_baseline(self, tmp_path):
        """Test SQLite insert performance."""
        import sqlite3

        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            """
            CREATE TABLE test (
                id INTEGER PRIMARY KEY,
                data TEXT,
                created_at TEXT
            )
        """
        )
        conn.commit()

        iterations = 1000
        start = time.perf_counter()
        for i in range(iterations):
            conn.execute(
                "INSERT INTO test (data, created_at) VALUES (?, ?)",
                (f"data-{i}", "2024-01-01"),
            )
        conn.commit()
        elapsed = time.perf_counter() - start

        avg_latency_ms = (elapsed / iterations) * 1000
        print(f"\nSQLite insert baseline: {avg_latency_ms:.3f}ms per insert")

        conn.close()

        # Assert reasonable baseline (< 1ms per insert with batched commit)
        assert avg_latency_ms < 1.0, f"Insert too slow: {avg_latency_ms}ms"

    def test_sqlite_select_baseline(self, tmp_path):
        """Test SQLite select performance."""
        import sqlite3

        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, data TEXT)")

        # Insert test data
        for i in range(1000):
            conn.execute("INSERT INTO test (data) VALUES (?)", (f"data-{i}",))
        conn.commit()

        # Create index
        conn.execute("CREATE INDEX idx_data ON test(data)")
        conn.commit()

        iterations = 1000
        start = time.perf_counter()
        for i in range(iterations):
            conn.execute("SELECT * FROM test WHERE data = ?", (f"data-{i % 1000}",))
        elapsed = time.perf_counter() - start

        avg_latency_ms = (elapsed / iterations) * 1000
        print(f"\nSQLite indexed select baseline: {avg_latency_ms:.3f}ms per select")

        conn.close()

        # Assert reasonable baseline (< 0.5ms per indexed select)
        assert avg_latency_ms < 0.5, f"Select too slow: {avg_latency_ms}ms"


# =============================================================================
# Performance Regression Markers
# =============================================================================

# These are baseline values that should be updated if intentional performance
# changes are made. Tests fail if performance degrades beyond these thresholds.

PERFORMANCE_THRESHOLDS = {
    "handler_routing_us": 100,  # Max microseconds for route matching
    "json_serialize_us": 500,  # Max microseconds for JSON response
    "cache_hit_ns": 10000,  # Max nanoseconds for cache hit
    "db_insert_ms": 1.0,  # Max milliseconds per insert
    "db_select_indexed_ms": 0.5,  # Max milliseconds per indexed select
}


@pytest.mark.load
class TestPerformanceRegression:
    """Regression tests to catch performance degradation."""

    def test_no_regression_marker(self):
        """Marker test to verify load tests are discoverable."""
        assert PERFORMANCE_THRESHOLDS["handler_routing_us"] == 100
