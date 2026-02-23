"""End-to-end integration tests for the Unified Memory Architecture.

Tests the full flow: Arena config → lazy gateway creation → query fan-out →
dedup → ranking → response, plus retention gate enrichment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from aragora.debate.arena_sub_configs import UnifiedMemorySubConfig
from aragora.memory.gateway import MemoryGateway, UnifiedMemoryQuery
from aragora.memory.gateway_config import MemoryGatewayConfig
from aragora.memory.dedup import CrossSystemDedupEngine
from aragora.memory.retention_gate import RetentionGate, RetentionGateConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class FakeContinuumEntry:
    id: str = "cm_1"
    content: str = "rate limiting with token bucket"
    importance: float = 0.8
    surprise_score: float = 0.6
    tier: str = "medium"


@dataclass
class FakeKMItem:
    id: str = "km_1"
    content: str = "token bucket rate limiter design"
    confidence: float = 0.9


@dataclass
class FakeKMQueryResult:
    items: list[Any] | None = None


@dataclass
class FakeSupermemoryResult:
    content: str = "external rate limit advice"
    similarity: float = 0.75
    memory_id: str = "sm_1"


def _mock_continuum(entries=None):
    mock = MagicMock()
    mock.search.return_value = entries or [FakeContinuumEntry()]
    return mock


def _mock_km(items=None):
    mock = AsyncMock()
    qr = FakeKMQueryResult(items=items or [FakeKMItem()])
    mock.query = AsyncMock(return_value=qr)
    return mock


def _mock_supermemory(results=None):
    mock = AsyncMock()
    mock.search_memories = AsyncMock(return_value=results or [FakeSupermemoryResult()])
    return mock


def _mock_claude_mem(observations=None):
    mock = AsyncMock()
    default_obs = [{"id": "obs_1", "content": "claude-mem rate limit insight", "metadata": {}}]
    mock.search_observations = AsyncMock(return_value=observations or default_obs)
    return mock


# ---------------------------------------------------------------------------
# E2E: Lazy Gateway Creation from Arena Config
# ---------------------------------------------------------------------------


class TestLazyGatewayE2E:
    def test_full_lazy_creation_flow(self):
        """Config → lazy factory → gateway with all sources."""
        from aragora.debate.lazy_subsystems import create_lazy_memory_gateway

        arena = MagicMock()
        arena.enable_unified_memory = True
        arena.enable_retention_gate = True
        arena.continuum_memory = _mock_continuum()
        arena.knowledge_mound = _mock_km()
        arena.supermemory_adapter = _mock_supermemory()

        gateway = create_lazy_memory_gateway(arena)
        assert gateway is not None
        sources = gateway._available_sources()
        assert "continuum" in sources
        assert "km" in sources
        assert "supermemory" in sources
        assert gateway.retention_gate is not None


# ---------------------------------------------------------------------------
# E2E: Full Query Pipeline
# ---------------------------------------------------------------------------


class TestQueryPipelineE2E:
    @pytest.mark.asyncio
    async def test_full_fan_out_dedup_rank(self):
        """Query fans out to all 4 sources, deduplicates, ranks."""
        gw = MemoryGateway(
            config=MemoryGatewayConfig(enabled=True),
            continuum_memory=_mock_continuum(),
            knowledge_mound=_mock_km(),
            supermemory_adapter=_mock_supermemory(),
            claude_mem_adapter=_mock_claude_mem(),
        )

        resp = await gw.query(
            UnifiedMemoryQuery(
                query="rate limiting",
                limit=20,
            )
        )

        assert resp.total_found == 4  # 1 from each source
        assert len(resp.sources_queried) == 4
        assert "continuum" in resp.sources_queried
        assert "km" in resp.sources_queried
        assert "supermemory" in resp.sources_queried
        assert "claude_mem" in resp.sources_queried
        assert resp.query_time_ms > 0

    @pytest.mark.asyncio
    async def test_cross_system_dedup_in_pipeline(self):
        """Identical content across sources gets deduped."""
        same = "identical rate limiting advice"
        gw = MemoryGateway(
            config=MemoryGatewayConfig(enabled=True),
            continuum_memory=_mock_continuum([FakeContinuumEntry(content=same)]),
            knowledge_mound=_mock_km([FakeKMItem(content=same)]),
        )

        resp = await gw.query(UnifiedMemoryQuery(query="test"))
        assert resp.duplicates_removed >= 1
        assert len(resp.results) == 1

    @pytest.mark.asyncio
    async def test_min_confidence_filter(self):
        """Low-confidence results filtered out."""
        gw = MemoryGateway(
            config=MemoryGatewayConfig(enabled=True),
            continuum_memory=_mock_continuum([FakeContinuumEntry(importance=0.2)]),
            knowledge_mound=_mock_km([FakeKMItem(confidence=0.95)]),
        )

        resp = await gw.query(
            UnifiedMemoryQuery(
                query="test",
                min_confidence=0.5,
            )
        )
        assert all(r.confidence >= 0.5 for r in resp.results)

    @pytest.mark.asyncio
    async def test_source_filter(self):
        """Only querying specific sources."""
        gw = MemoryGateway(
            config=MemoryGatewayConfig(enabled=True),
            continuum_memory=_mock_continuum(),
            knowledge_mound=_mock_km(),
            supermemory_adapter=_mock_supermemory(),
        )

        resp = await gw.query(
            UnifiedMemoryQuery(
                query="test",
                sources=["km"],
            )
        )
        assert resp.sources_queried == ["km"]
        assert all(r.source_system == "km" for r in resp.results)


# ---------------------------------------------------------------------------
# E2E: Retention Gate Integration
# ---------------------------------------------------------------------------


class TestRetentionGateE2E:
    def test_retention_gate_evaluates_items(self):
        """RetentionGate evaluates items with surprise scores."""
        gate = RetentionGate(
            config=RetentionGateConfig(
                enable_surprise_driven_decay=True,
                forget_threshold=0.15,
                consolidate_threshold=0.7,
            )
        )

        # High surprise → consolidate
        decision = gate.evaluate(
            item_id="cm_1",
            source="continuum",
            content="surprising discovery",
            outcome_surprise=0.9,
            current_confidence=0.8,
        )
        assert decision.action == "consolidate"

        # Low surprise + low confidence → forget
        decision = gate.evaluate(
            item_id="cm_2",
            source="continuum",
            content="boring fact",
            outcome_surprise=0.05,
            current_confidence=0.1,
        )
        assert decision.action == "forget"

    def test_red_line_protection(self):
        """Red-line items are always retained."""
        gate = RetentionGate(config=RetentionGateConfig())

        decision = gate.evaluate(
            item_id="cm_3",
            source="continuum",
            content="critical safety constraint",
            outcome_surprise=0.01,
            current_confidence=0.05,
            is_red_line=True,
        )
        assert decision.action == "retain"

    def test_batch_evaluate(self):
        """Batch evaluation processes multiple items."""
        gate = RetentionGate(config=RetentionGateConfig())

        items = [
            {
                "item_id": f"item_{i}",
                "source": "continuum",
                "content": f"content {i}",
                "outcome_surprise": 0.1 * i,
                "current_confidence": 0.5,
            }
            for i in range(5)
        ]
        decisions = gate.batch_evaluate(items)
        assert len(decisions) == 5
        assert all(d.item_id.startswith("item_") for d in decisions)


# ---------------------------------------------------------------------------
# E2E: RLM Navigator Integration
# ---------------------------------------------------------------------------


class TestNavigatorE2E:
    @pytest.mark.asyncio
    async def test_navigator_builds_context_from_gateway(self):
        """Navigator builds hierarchical context from gateway results."""
        from aragora.rlm.memory_navigator import RLMMemoryNavigator

        gw = MemoryGateway(
            config=MemoryGatewayConfig(enabled=True),
            continuum_memory=_mock_continuum(),
            knowledge_mound=_mock_km(),
        )

        nav = RLMMemoryNavigator(gateway=gw)
        ctx = await nav.build_context_hierarchy("rate limiting")
        assert ctx.total_items == 2
        assert ctx.query == "rate limiting"
        assert "km" in ctx.by_source
        assert "continuum" in ctx.by_source

    @pytest.mark.asyncio
    async def test_navigator_search_all(self):
        """Navigator search_all returns items from gateway."""
        from aragora.rlm.memory_navigator import RLMMemoryNavigator

        gw = MemoryGateway(
            config=MemoryGatewayConfig(enabled=True),
            continuum_memory=_mock_continuum(),
        )

        nav = RLMMemoryNavigator(gateway=gw)
        results = await nav.search_all("rate limiting")
        assert len(results) == 1
        assert results[0].source == "continuum"


# ---------------------------------------------------------------------------
# E2E: HTTP Handler Integration
# ---------------------------------------------------------------------------


class TestHandlerE2E:
    @pytest.mark.asyncio
    async def test_handler_search_with_real_gateway(self):
        """Handler delegates to real gateway and returns formatted results."""
        from aragora.server.handlers.memory.unified_handler import UnifiedMemoryHandler

        gw = MemoryGateway(
            config=MemoryGatewayConfig(enabled=True),
            continuum_memory=_mock_continuum(),
            knowledge_mound=_mock_km(),
        )

        handler = UnifiedMemoryHandler(gateway=gw)
        result = await handler.handle_search({"query": "rate limiting"})
        assert "results" in result
        assert len(result["results"]) == 2
        assert result["total_found"] == 2
        assert "km" in result["sources_queried"]
        assert "continuum" in result["sources_queried"]

    @pytest.mark.asyncio
    async def test_handler_stats_with_real_gateway(self):
        """Handler stats endpoint returns gateway statistics."""
        from aragora.server.handlers.memory.unified_handler import UnifiedMemoryHandler

        gw = MemoryGateway(
            config=MemoryGatewayConfig(enabled=True),
            continuum_memory=_mock_continuum(),
        )

        handler = UnifiedMemoryHandler(gateway=gw)
        result = await handler.handle_stats()
        assert "continuum" in result["available_sources"]
        assert result["config"]["enabled"] is True


# ---------------------------------------------------------------------------
# E2E: Dedup Engine Integration
# ---------------------------------------------------------------------------


class TestDedupE2E:
    @pytest.mark.asyncio
    async def test_cross_system_scan_detects_duplicates(self):
        """Full dedup scan across systems finds exact and near duplicates."""
        engine = CrossSystemDedupEngine(near_duplicate_threshold=0.5)

        items = [
            {
                "id": "cm_1",
                "source": "continuum",
                "content": "rate limiting with token bucket algorithm",
            },
            {"id": "km_1", "source": "km", "content": "rate limiting with token bucket algorithm"},
            {
                "id": "sm_1",
                "source": "supermemory",
                "content": "rate limiting with token bucket strategy",
            },
            {"id": "obs_1", "source": "claude_mem", "content": "completely unrelated content"},
        ]

        report = await engine.scan_cross_system_duplicates(items)
        assert report.total_items_scanned == 4
        assert report.exact_duplicates >= 1  # cm_1 and km_1 are exact
        assert report.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_dedup_before_write(self):
        """Write dedup prevents storing duplicate content."""
        engine = CrossSystemDedupEngine()
        engine.register_item("km_1", "km", "existing knowledge about rate limiting")

        # Same content → duplicate detected
        result = await engine.check_duplicate_before_write("existing knowledge about rate limiting")
        assert result.is_duplicate is True
        assert result.existing_id == "km_1"

        # Different content → no duplicate
        result = await engine.check_duplicate_before_write("something entirely new")
        assert result.is_duplicate is False
