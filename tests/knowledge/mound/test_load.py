"""
Load tests for Knowledge Mound adapters.

These tests verify adapter performance under load and ensure SLO compliance.
Run with: pytest tests/knowledge/mound/test_load.py -v
"""

import asyncio
import statistics
import tempfile
import time
from typing import List
from unittest.mock import AsyncMock, MagicMock
import uuid

import pytest


def calculate_percentile(data: List[float], percentile: float) -> float:
    """Calculate percentile of a sorted list."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * percentile / 100
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_data) else f
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


class TestContinuumAdapterLoad:
    """Load tests for ContinuumAdapter operations."""

    @pytest.fixture
    def continuum(self):
        """Create a ContinuumMemory instance for testing."""
        from aragora.memory.continuum import ContinuumMemory

        with tempfile.TemporaryDirectory() as tmpdir:
            memory = ContinuumMemory(db_path=f"{tmpdir}/test.db")
            yield memory

    @pytest.fixture
    def adapter(self, continuum):
        """Create a ContinuumAdapter."""
        from aragora.knowledge.mound.adapters.continuum_adapter import ContinuumAdapter

        return ContinuumAdapter(continuum, enable_dual_write=False)

    def test_search_throughput_100_ops(self, adapter):
        """Test search can handle 100 operations efficiently."""
        latencies = []
        keywords = ["test", "debate", "climate", "ai", "policy"]

        for i in range(100):
            keyword = keywords[i % len(keywords)]
            start = time.perf_counter()
            adapter.search_by_keyword(keyword, limit=10)
            latencies.append(time.perf_counter() - start)

        mean_ms = statistics.mean(latencies) * 1000
        p95_ms = calculate_percentile(latencies, 95) * 1000

        assert mean_ms < 50, f"Mean latency {mean_ms:.2f}ms exceeds 50ms"
        assert p95_ms < 100, f"p95 latency {p95_ms:.2f}ms exceeds 100ms"

    def test_adapter_get_stats(self, adapter):
        """Test adapter stats retrieval performance."""
        latencies = []

        for _ in range(100):
            start = time.perf_counter()
            stats = adapter.get_stats()
            latencies.append(time.perf_counter() - start)

        mean_ms = statistics.mean(latencies) * 1000
        p95_ms = calculate_percentile(latencies, 95) * 1000

        assert mean_ms < 10, f"Mean stats latency {mean_ms:.2f}ms exceeds 10ms"
        assert p95_ms < 50, f"p95 stats latency {p95_ms:.2f}ms exceeds 50ms"
        assert isinstance(stats, dict)


class TestConsensusAdapterLoad:
    """Load tests for ConsensusAdapter operations."""

    @pytest.fixture
    def consensus(self):
        """Create a ConsensusMemory instance for testing."""
        from aragora.memory.consensus import ConsensusMemory

        with tempfile.TemporaryDirectory() as tmpdir:
            memory = ConsensusMemory(db_path=f"{tmpdir}/consensus.db")
            yield memory

    @pytest.fixture
    def adapter(self, consensus):
        """Create a ConsensusAdapter."""
        from aragora.knowledge.mound.adapters.consensus_adapter import ConsensusAdapter

        return ConsensusAdapter(consensus, enable_dual_write=False)

    def test_search_throughput_100_ops(self, adapter):
        """Test search can handle 100 operations efficiently."""
        latencies = []
        topics = ["climate", "AI ethics", "economics", "healthcare", "education"]

        for i in range(100):
            topic = topics[i % len(topics)]
            start = time.perf_counter()
            adapter.search_by_topic(topic, limit=10)
            latencies.append(time.perf_counter() - start)

        mean_ms = statistics.mean(latencies) * 1000
        p95_ms = calculate_percentile(latencies, 95) * 1000

        assert mean_ms < 50, f"Mean latency {mean_ms:.2f}ms exceeds 50ms"
        assert p95_ms < 100, f"p95 latency {p95_ms:.2f}ms exceeds 100ms"


class TestMockedKMLoad:
    """Load tests using mocked KM operations for consistent benchmarking."""

    @pytest.fixture
    def mock_mound(self):
        """Create a mock KnowledgeMound."""
        mound = MagicMock()
        mound.ingest = AsyncMock(return_value="km_test")
        mound.query = AsyncMock(return_value=[
            {"id": f"km_{i}", "content": f"Result {i}", "confidence": 0.8}
            for i in range(5)
        ])
        mound.semantic_search = AsyncMock(return_value=[
            {"id": f"km_{i}", "content": f"Result {i}", "similarity": 0.9 - i * 0.05}
            for i in range(5)
        ])
        return mound

    @pytest.mark.asyncio
    async def test_ingest_throughput(self, mock_mound):
        """Test ingest throughput with mocked backend."""
        latencies = []

        for i in range(100):
            start = time.perf_counter()
            await mock_mound.ingest({"content": f"Test {i}", "confidence": 0.8})
            latencies.append(time.perf_counter() - start)

        mean_ms = statistics.mean(latencies) * 1000
        p95_ms = calculate_percentile(latencies, 95) * 1000

        # With mocks, operations should be very fast
        assert mean_ms < 5, f"Mean latency {mean_ms:.2f}ms exceeds 5ms"
        assert p95_ms < 10, f"p95 latency {p95_ms:.2f}ms exceeds 10ms"

    @pytest.mark.asyncio
    async def test_query_throughput(self, mock_mound):
        """Test query throughput with mocked backend."""
        latencies = []
        queries = ["climate", "AI", "economics", "health", "education"]

        for i in range(100):
            query = queries[i % len(queries)]
            start = time.perf_counter()
            await mock_mound.query(query, limit=10)
            latencies.append(time.perf_counter() - start)

        mean_ms = statistics.mean(latencies) * 1000
        p95_ms = calculate_percentile(latencies, 95) * 1000

        assert mean_ms < 5, f"Mean latency {mean_ms:.2f}ms exceeds 5ms"
        assert p95_ms < 10, f"p95 latency {p95_ms:.2f}ms exceeds 10ms"

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, mock_mound):
        """Test concurrent operations performance."""

        async def mixed_op(idx: int) -> float:
            start = time.perf_counter()
            if idx % 2 == 0:
                await mock_mound.ingest({"content": f"Test {idx}"})
            else:
                await mock_mound.query(f"query_{idx}")
            return time.perf_counter() - start

        latencies = await asyncio.gather(*[mixed_op(i) for i in range(100)])

        mean_ms = statistics.mean(latencies) * 1000
        max_ms = max(latencies) * 1000

        assert mean_ms < 10, f"Mean concurrent latency {mean_ms:.2f}ms exceeds 10ms"
        assert max_ms < 50, f"Max concurrent latency {max_ms:.2f}ms exceeds 50ms"


class TestSLOBenchmarks:
    """Benchmark tests that verify SLO targets are met."""

    def test_adapter_operation_overhead(self):
        """Measure pure adapter operation overhead (no IO)."""
        from aragora.knowledge.mound.adapters.continuum_adapter import ContinuumAdapter
        from aragora.memory.continuum import ContinuumMemory, ContinuumMemoryEntry
        from aragora.memory.tier_manager import MemoryTier

        with tempfile.TemporaryDirectory() as tmpdir:
            continuum = ContinuumMemory(db_path=f"{tmpdir}/test.db")
            adapter = ContinuumAdapter(continuum, enable_dual_write=False)

            # Measure conversion overhead
            entry = ContinuumMemoryEntry(
                id="test_entry",
                tier=MemoryTier.FAST,
                content="Test content for performance measurement",
                importance=0.8,
                surprise_score=0.5,
                consolidation_score=0.7,
                update_count=1,
                success_count=5,
                failure_count=0,
                created_at="2024-01-01T00:00:00",
                updated_at="2024-01-01T00:00:00",
            )

            latencies = []
            for _ in range(1000):
                start = time.perf_counter()
                adapter.to_knowledge_item(entry)
                latencies.append(time.perf_counter() - start)

            mean_us = statistics.mean(latencies) * 1_000_000
            p99_us = calculate_percentile(latencies, 99) * 1_000_000

            # Conversion should be sub-millisecond
            assert mean_us < 500, f"Mean conversion overhead {mean_us:.2f}us exceeds 500us"
            assert p99_us < 2000, f"p99 conversion overhead {p99_us:.2f}us exceeds 2000us"

    def test_search_slo_compliance(self):
        """Verify search operations meet SLO: p95 < 100ms."""
        from aragora.knowledge.mound.adapters.continuum_adapter import ContinuumAdapter
        from aragora.memory.continuum import ContinuumMemory

        with tempfile.TemporaryDirectory() as tmpdir:
            continuum = ContinuumMemory(db_path=f"{tmpdir}/test.db")
            adapter = ContinuumAdapter(continuum, enable_dual_write=False)

            latencies = []
            keywords = ["test", "debate", "climate"]

            for i in range(200):
                keyword = keywords[i % len(keywords)]
                start = time.perf_counter()
                adapter.search_by_keyword(keyword, limit=10)
                latencies.append(time.perf_counter() - start)

            p95_ms = calculate_percentile(latencies, 95) * 1000
            p99_ms = calculate_percentile(latencies, 99) * 1000

            assert p95_ms < 100, f"Search p95 {p95_ms:.2f}ms exceeds SLO 100ms"
            assert p99_ms < 300, f"Search p99 {p99_ms:.2f}ms exceeds SLO 300ms"


class TestHighConcurrency:
    """Tests for high concurrency scenarios."""

    @pytest.mark.asyncio
    async def test_100_concurrent_queries(self):
        """Test 100 concurrent query operations."""
        mock_mound = MagicMock()
        mock_mound.query = AsyncMock(return_value=[{"id": "km_1", "content": "Result"}])

        async def query_one(idx: int) -> float:
            start = time.perf_counter()
            await mock_mound.query(f"query_{idx}", limit=10)
            return time.perf_counter() - start

        latencies = await asyncio.gather(*[query_one(i) for i in range(100)])

        mean_ms = statistics.mean(latencies) * 1000
        max_ms = max(latencies) * 1000

        # Under high concurrency, latency should still be reasonable
        assert mean_ms < 20, f"Mean latency {mean_ms:.2f}ms exceeds 20ms at 100 concurrency"
        assert max_ms < 100, f"Max latency {max_ms:.2f}ms exceeds 100ms at 100 concurrency"

    @pytest.mark.asyncio
    async def test_burst_load(self):
        """Test handling of burst load (sudden spike in requests)."""
        mock_mound = MagicMock()
        mock_mound.query = AsyncMock(return_value=[])
        mock_mound.ingest = AsyncMock(return_value="km_test")

        all_latencies: List[float] = []

        # Simulate 5 bursts of 50 operations each
        for _ in range(5):
            async def burst_op(idx: int) -> float:
                start = time.perf_counter()
                if idx % 3 == 0:
                    await mock_mound.ingest({"content": f"Test {idx}"})
                else:
                    await mock_mound.query(f"query_{idx}")
                return time.perf_counter() - start

            burst_latencies = await asyncio.gather(*[burst_op(i) for i in range(50)])
            all_latencies.extend(burst_latencies)

        mean_ms = statistics.mean(all_latencies) * 1000
        p99_ms = calculate_percentile(all_latencies, 99) * 1000

        assert mean_ms < 20, f"Mean burst latency {mean_ms:.2f}ms exceeds 20ms"
        assert p99_ms < 100, f"p99 burst latency {p99_ms:.2f}ms exceeds 100ms"
