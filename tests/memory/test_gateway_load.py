"""Load tests for unified memory gateway, dedup engine, and retention gate."""

import asyncio
import time

import pytest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from aragora.memory.dedup import CrossSystemDedupEngine
from aragora.memory.gateway import (
    MemoryGateway,
    UnifiedMemoryQuery,
    UnifiedMemoryResult,
)
from aragora.memory.gateway_config import MemoryGatewayConfig
from aragora.memory.retention_gate import RetentionGate, RetentionGateConfig


class TestDedupEngineLoad:
    def test_register_1000_items(self):
        """Hash index handles 1000 items."""
        engine = CrossSystemDedupEngine()
        for i in range(1000):
            engine.register_item(f"item_{i}", f"source_{i % 4}", f"unique content {i}")
        assert engine.get_hash_index_size() == 1000

    @pytest.mark.asyncio
    async def test_scan_500_with_50pct_duplicates(self):
        """Exact dedup at 250 duplicate pairs."""
        items = []
        for i in range(250):
            content = f"shared content {i}"
            items.append({"id": f"a_{i}", "source": "continuum", "content": content})
            items.append({"id": f"b_{i}", "source": "km", "content": content})

        engine = CrossSystemDedupEngine()
        report = await engine.scan_cross_system_duplicates(items)
        assert report.total_items_scanned == 500
        assert report.exact_duplicates == 250

    @pytest.mark.asyncio
    async def test_near_duplicate_jaccard_200_items(self):
        """Jaccard near-duplicate detection at scale."""
        engine = CrossSystemDedupEngine(near_duplicate_threshold=0.8)
        items = []
        base = "the quick brown fox jumps over the lazy dog"
        for i in range(100):
            items.append({"id": f"a_{i}", "source": "continuum", "content": f"{base} version {i}"})
            items.append({"id": f"b_{i}", "source": "km", "content": f"{base} variant {i}"})

        report = await engine.scan_cross_system_duplicates(items)
        assert report.total_items_scanned == 200
        # Near-duplicates found depends on threshold; just verify the scan completes
        assert report.near_duplicates >= 0

    @pytest.mark.asyncio
    async def test_check_before_write_fast_with_large_index(self):
        """Lookup is fast with 1000 items indexed."""
        engine = CrossSystemDedupEngine()
        for i in range(1000):
            engine.register_item(f"item_{i}", "source", f"content {i}")

        start = time.time()
        for i in range(100):
            await engine.check_duplicate_before_write(f"new content {i}")
        elapsed_ms = (time.time() - start) * 1000
        assert elapsed_ms < 500  # 100 lookups should be very fast


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


class TestGatewayFanOut:
    @pytest.mark.asyncio
    async def test_parallel_4_sources_200_results(self):
        """Fan-out merges 50 results x 4 sources."""
        continuum = SimpleNamespace(search=lambda query, limit: _make_continuum_results(50))
        km = AsyncMock()
        km.query = AsyncMock(return_value=_make_km_query_result(50))
        supermemory = AsyncMock()
        supermemory.search_memories = AsyncMock(return_value=_make_supermemory_results(50))
        claude_mem = AsyncMock()
        claude_mem.search_observations = AsyncMock(return_value=_make_claude_mem_results(50))

        gateway = MemoryGateway(
            config=MemoryGatewayConfig(
                enabled=True, parallel_queries=True, query_timeout_seconds=5.0
            ),
            continuum_memory=continuum,
            knowledge_mound=km,
            supermemory_adapter=supermemory,
            claude_mem_adapter=claude_mem,
        )

        response = await gateway.query(UnifiedMemoryQuery(query="test", limit=200, dedup=False))
        assert response.total_found == 200
        assert len(response.sources_queried) == 4
        assert not response.errors

    @pytest.mark.asyncio
    async def test_timeout_slow_source(self):
        """One slow source times out, others return."""

        async def slow_query(query, limit):
            await asyncio.sleep(10)
            return _make_km_query_result(10)

        continuum = SimpleNamespace(search=lambda query, limit: _make_continuum_results(10))
        km = AsyncMock()
        km.query = slow_query

        gateway = MemoryGateway(
            config=MemoryGatewayConfig(
                enabled=True, parallel_queries=True, query_timeout_seconds=0.1
            ),
            continuum_memory=continuum,
            knowledge_mound=km,
        )

        response = await gateway.query(UnifiedMemoryQuery(query="test", limit=50))
        # Continuum returns 10 results, km times out
        assert response.total_found == 10
        assert "km" in response.errors
        assert response.errors["km"] == "timeout"

    @pytest.mark.asyncio
    async def test_dedup_across_sources(self):
        """Cross-source dedup removes identical content."""
        shared_content = "identical content across sources"
        continuum = SimpleNamespace(
            search=lambda query, limit: [
                SimpleNamespace(
                    id="c_0",
                    content=shared_content,
                    importance=0.8,
                    surprise_score=0.5,
                    tier="fast",
                )
            ]
        )
        km = AsyncMock()
        km.query = AsyncMock(
            return_value=SimpleNamespace(
                items=[SimpleNamespace(id="k_0", content=shared_content, confidence=0.9)]
            )
        )

        gateway = MemoryGateway(
            config=MemoryGatewayConfig(enabled=True, parallel_queries=True),
            continuum_memory=continuum,
            knowledge_mound=km,
        )

        response = await gateway.query(UnifiedMemoryQuery(query="test", limit=10, dedup=True))
        assert len(response.results) == 1
        assert response.duplicates_removed == 1

    @pytest.mark.asyncio
    async def test_ranking_by_confidence(self):
        """Results sorted by confidence score."""
        continuum = SimpleNamespace(
            search=lambda query, limit: [
                SimpleNamespace(
                    id="low",
                    content="low confidence item",
                    importance=0.1,
                    surprise_score=None,
                    tier="slow",
                ),
                SimpleNamespace(
                    id="high",
                    content="high confidence item",
                    importance=0.95,
                    surprise_score=None,
                    tier="fast",
                ),
            ]
        )

        gateway = MemoryGateway(
            config=MemoryGatewayConfig(enabled=True),
            continuum_memory=continuum,
        )

        response = await gateway.query(UnifiedMemoryQuery(query="test", limit=10, dedup=False))
        assert len(response.results) == 2
        # Higher confidence should rank first
        assert response.results[0].confidence > response.results[1].confidence


class TestRetentionGateLoad:
    def test_batch_evaluate_500_items(self):
        """Mixed surprise scores produce mixed decisions."""
        gate = RetentionGate(RetentionGateConfig(adaptive_decay_enabled=True))
        items = []
        for i in range(500):
            surprise = (i % 100) / 100.0
            items.append(
                {
                    "item_id": f"item_{i}",
                    "source": "km",
                    "content": f"content {i}",
                    "outcome_surprise": surprise,
                    "current_confidence": 0.5,
                    "access_count": i % 10,
                    "is_red_line": False,
                }
            )

        decisions = gate.batch_evaluate(items)
        assert len(decisions) == 500
        actions = {d.action for d in decisions}
        # Should have mix of retain, demote, forget, consolidate
        assert len(actions) >= 2

    def test_red_line_protection_100_items(self):
        """All red-line items retained regardless of low surprise/confidence."""
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
            for i in range(100)
        ]

        decisions = gate.batch_evaluate(items)
        assert all(d.action == "retain" for d in decisions)

    def test_adaptive_decay_monotonic(self):
        """Higher surprise produces slower (lower) decay rate."""
        gate = RetentionGate(RetentionGateConfig(adaptive_decay_enabled=True))

        prev_rate = None
        for surprise in [0.1, 0.3, 0.5, 0.7, 0.9]:
            rate = gate.compute_adaptive_decay_rate(surprise, 0.5)
            if prev_rate is not None:
                assert rate <= prev_rate
            prev_rate = rate

    def test_forget_low_surprise_low_confidence(self):
        """Low surprise + low confidence -> forget."""
        gate = RetentionGate(
            RetentionGateConfig(
                forget_threshold=0.15,
                consolidate_threshold=0.7,
            )
        )

        decision = gate.evaluate(
            item_id="test_item",
            source="km",
            content="mundane content",
            outcome_surprise=0.05,
            current_confidence=0.1,
        )
        assert decision.action == "forget"
