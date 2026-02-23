"""Benchmark-scale memory gateway load tests (10x of standard load tests).

These tests run at 10x scale vs test_gateway_load.py and include timing
assertions. Marked with @pytest.mark.benchmark for nightly CI runs only.

Run with:
    pytest tests/memory/test_gateway_benchmark.py -v --timeout=120 -m benchmark
"""

from __future__ import annotations

import time

import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from aragora.memory.dedup import CrossSystemDedupEngine
from aragora.memory.gateway import (
    MemoryGateway,
    UnifiedMemoryQuery,
)
from aragora.memory.gateway_config import MemoryGatewayConfig
from aragora.memory.retention_gate import RetentionGate, RetentionGateConfig


def _make_continuum_results(n, prefix="continuum"):
    """Create mock continuum search results."""
    return [
        SimpleNamespace(
            id=f"{prefix}_{i}",
            content=f"{prefix} content {i}",
            importance=0.5 + (i % 5) * 0.1,
            surprise_score=0.3,
            tier="medium",
        )
        for i in range(n)
    ]


def _make_km_query_result(n, prefix="km"):
    """Create mock KM query result with .items attribute."""
    items = [
        SimpleNamespace(
            id=f"{prefix}_{i}",
            content=f"{prefix} content {i}",
            confidence=0.6 + (i % 4) * 0.1,
        )
        for i in range(n)
    ]
    return SimpleNamespace(items=items)


def _make_supermemory_results(n, prefix="supermemory"):
    """Create mock supermemory search results."""
    return [
        SimpleNamespace(
            memory_id=f"{prefix}_{i}",
            content=f"{prefix} content {i}",
            similarity=0.7 + (i % 3) * 0.1,
        )
        for i in range(n)
    ]


def _make_claude_mem_results(n, prefix="claude_mem"):
    """Create mock claude-mem observation results."""
    return [
        {"id": f"{prefix}_{i}", "content": f"{prefix} content {i}", "metadata": {}}
        for i in range(n)
    ]


@pytest.mark.benchmark
class TestDedupEngineBenchmark:
    """10x scale dedup engine tests with timing assertions."""

    def test_register_10000_items(self):
        """Hash index handles 10,000 items under 2 seconds."""
        engine = CrossSystemDedupEngine()
        start = time.time()
        for i in range(10_000):
            engine.register_item(f"item_{i}", f"source_{i % 4}", f"unique content {i}")
        elapsed = time.time() - start
        assert engine.get_hash_index_size() == 10_000
        assert elapsed < 2.0, f"Registration took {elapsed:.2f}s, expected <2s"

    @pytest.mark.asyncio
    async def test_scan_2000_with_50pct_duplicates(self):
        """Exact dedup at 1000 duplicate pairs under 30 seconds."""
        items = []
        for i in range(1000):
            content = f"shared content {i}"
            items.append({"id": f"a_{i}", "source": "continuum", "content": content})
            items.append({"id": f"b_{i}", "source": "km", "content": content})

        engine = CrossSystemDedupEngine()
        start = time.time()
        report = await engine.scan_cross_system_duplicates(items)
        elapsed = time.time() - start
        assert report.total_items_scanned == 2000
        assert report.exact_duplicates == 1000
        assert elapsed < 30.0, f"Scan took {elapsed:.2f}s, expected <30s"

    @pytest.mark.asyncio
    async def test_check_before_write_1000_lookups_in_10k_index(self):
        """1000 lookups against 10,000-item index under 2 seconds."""
        engine = CrossSystemDedupEngine()
        for i in range(10_000):
            engine.register_item(f"item_{i}", "source", f"content {i}")

        start = time.time()
        for i in range(1000):
            await engine.check_duplicate_before_write(f"new content {i}")
        elapsed_ms = (time.time() - start) * 1000
        assert elapsed_ms < 2000, f"Lookups took {elapsed_ms:.0f}ms, expected <2000ms"


@pytest.mark.benchmark
class TestGatewayFanOutBenchmark:
    """10x scale gateway fan-out tests."""

    @pytest.mark.asyncio
    async def test_parallel_4_sources_2000_results(self):
        """Fan-out merges 500 results from each of 4 sources."""
        continuum = SimpleNamespace(search=lambda query, limit: _make_continuum_results(500))
        km = AsyncMock()
        km.query = AsyncMock(return_value=_make_km_query_result(500))
        supermemory = AsyncMock()
        supermemory.search_memories = AsyncMock(return_value=_make_supermemory_results(500))
        claude_mem = AsyncMock()
        claude_mem.search_observations = AsyncMock(return_value=_make_claude_mem_results(500))

        gateway = MemoryGateway(
            config=MemoryGatewayConfig(
                enabled=True, parallel_queries=True, query_timeout_seconds=10.0
            ),
            continuum_memory=continuum,
            knowledge_mound=km,
            supermemory_adapter=supermemory,
            claude_mem_adapter=claude_mem,
        )

        start = time.time()
        response = await gateway.query(UnifiedMemoryQuery(query="test", limit=2000, dedup=False))
        elapsed = time.time() - start
        assert response.total_found == 2000
        assert len(response.sources_queried) == 4
        assert not response.errors
        assert elapsed < 5.0, f"Fan-out took {elapsed:.2f}s, expected <5s"

    @pytest.mark.asyncio
    async def test_dedup_across_sources_500_items(self):
        """Cross-source dedup at 500 items (250 duplicated across 2 sources)."""
        n = 250
        continuum_results = [
            SimpleNamespace(
                id=f"c_{i}",
                content=f"shared content {i}",
                importance=0.8,
                surprise_score=0.5,
                tier="fast",
            )
            for i in range(n)
        ]
        km_items = [
            SimpleNamespace(
                id=f"k_{i}",
                content=f"shared content {i}",
                confidence=0.9,
            )
            for i in range(n)
        ]

        continuum = SimpleNamespace(search=lambda query, limit: continuum_results)
        km = AsyncMock()
        km.query = AsyncMock(return_value=SimpleNamespace(items=km_items))

        gateway = MemoryGateway(
            config=MemoryGatewayConfig(enabled=True, parallel_queries=True),
            continuum_memory=continuum,
            knowledge_mound=km,
        )

        start = time.time()
        response = await gateway.query(UnifiedMemoryQuery(query="test", limit=500, dedup=True))
        elapsed = time.time() - start
        assert len(response.results) == n  # Dedup removes 250 duplicates
        assert response.duplicates_removed == n
        assert elapsed < 5.0, f"Dedup took {elapsed:.2f}s, expected <5s"


@pytest.mark.benchmark
class TestRetentionGateBenchmark:
    """10x scale retention gate tests."""

    def test_batch_evaluate_5000_items(self):
        """Batch evaluate 5000 items under 5 seconds."""
        gate = RetentionGate(RetentionGateConfig(adaptive_decay_enabled=True))
        items = [
            {
                "item_id": f"item_{i}",
                "source": "km",
                "content": f"content {i}",
                "outcome_surprise": (i % 100) / 100.0,
                "current_confidence": 0.5,
                "access_count": i % 10,
                "is_red_line": False,
            }
            for i in range(5000)
        ]

        start = time.time()
        decisions = gate.batch_evaluate(items)
        elapsed = time.time() - start
        assert len(decisions) == 5000
        actions = {d.action for d in decisions}
        assert len(actions) >= 2
        assert elapsed < 5.0, f"Batch took {elapsed:.2f}s, expected <5s"

    def test_red_line_protection_1000_items(self):
        """All 1000 red-line items retained under 2 seconds."""
        gate = RetentionGate(RetentionGateConfig(red_line_protection=True))
        items = [
            {
                "item_id": f"rl_{i}",
                "source": "km",
                "content": f"critical content {i}",
                "outcome_surprise": 0.01,
                "current_confidence": 0.1,
                "is_red_line": True,
            }
            for i in range(1000)
        ]

        start = time.time()
        decisions = gate.batch_evaluate(items)
        elapsed = time.time() - start
        assert all(d.action == "retain" for d in decisions)
        assert elapsed < 2.0, f"Red-line took {elapsed:.2f}s, expected <2s"
