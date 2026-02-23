"""Tests for the Unified Memory Gateway."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.memory.gateway import (
    MemoryGateway,
    MemoryGatewayConfig,
    UnifiedMemoryQuery,
    UnifiedMemoryResponse,
    UnifiedMemoryResult,
)
from aragora.memory.gateway_config import MemoryGatewayConfig as GWConfig


# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------


@dataclass
class FakeContinuumEntry:
    id: str = "cm_1"
    content: str = "rate limiting pattern"
    importance: float = 0.8
    surprise_score: float = 0.5
    tier: str = "medium"


@dataclass
class FakeKMItem:
    id: str = "km_1"
    content: str = "knowledge about rate limiting"
    confidence: float = 0.9


@dataclass
class FakeKMQueryResult:
    items: list[Any] | None = None


@dataclass
class FakeSupermemoryResult:
    content: str = "supermemory rate limit info"
    similarity: float = 0.75
    memory_id: str = "sm_1"


def _make_continuum_mock(entries: list[FakeContinuumEntry] | None = None):
    mock = MagicMock()
    mock.search.return_value = entries or [FakeContinuumEntry()]
    return mock


def _make_km_mock(items: list[FakeKMItem] | None = None):
    mock = AsyncMock()
    qr = FakeKMQueryResult(items=items or [FakeKMItem()])
    mock.query = AsyncMock(return_value=qr)
    return mock


def _make_supermemory_mock(results: list[FakeSupermemoryResult] | None = None):
    mock = AsyncMock()
    mock.search_memories = AsyncMock(return_value=results or [FakeSupermemoryResult()])
    return mock


def _make_claude_mem_mock(observations: list[dict] | None = None):
    mock = AsyncMock()
    default_obs = [{"id": "obs_1", "content": "claude-mem observation", "metadata": {}}]
    mock.search_observations = AsyncMock(return_value=observations or default_obs)
    return mock


# ---------------------------------------------------------------------------
# Tests: Gateway Config
# ---------------------------------------------------------------------------


class TestMemoryGatewayConfig:
    def test_defaults(self):
        cfg = MemoryGatewayConfig()
        assert cfg.enabled is False
        assert cfg.query_timeout_seconds == 15.0
        assert cfg.dedup_threshold == 0.95
        assert cfg.default_sources is None
        assert cfg.parallel_queries is True

    def test_custom_values(self):
        cfg = MemoryGatewayConfig(
            enabled=True,
            query_timeout_seconds=5.0,
            default_sources=["km", "continuum"],
        )
        assert cfg.enabled is True
        assert cfg.default_sources == ["km", "continuum"]


# ---------------------------------------------------------------------------
# Tests: UnifiedMemoryQuery / Result / Response
# ---------------------------------------------------------------------------


class TestDataclasses:
    def test_query_defaults(self):
        q = UnifiedMemoryQuery(query="test")
        assert q.limit == 10
        assert q.min_confidence == 0.0
        assert q.sources is None
        assert q.dedup is True

    def test_result_fields(self):
        r = UnifiedMemoryResult(
            id="r1",
            content="test content",
            source_system="km",
            confidence=0.9,
        )
        assert r.surprise_score is None
        assert r.content_hash == ""
        assert r.metadata == {}

    def test_response_defaults(self):
        resp = UnifiedMemoryResponse()
        assert resp.results == []
        assert resp.total_found == 0
        assert resp.duplicates_removed == 0


# ---------------------------------------------------------------------------
# Tests: Available Sources
# ---------------------------------------------------------------------------


class TestAvailableSources:
    def test_no_sources(self):
        gw = MemoryGateway()
        assert gw._available_sources() == []

    def test_all_sources(self):
        gw = MemoryGateway(
            continuum_memory=MagicMock(),
            knowledge_mound=AsyncMock(),
            supermemory_adapter=AsyncMock(),
            claude_mem_adapter=AsyncMock(),
        )
        sources = gw._available_sources()
        assert "continuum" in sources
        assert "km" in sources
        assert "supermemory" in sources
        assert "claude_mem" in sources

    def test_partial_sources(self):
        gw = MemoryGateway(continuum_memory=MagicMock())
        assert gw._available_sources() == ["continuum"]


# ---------------------------------------------------------------------------
# Tests: Query Fan-Out
# ---------------------------------------------------------------------------


class TestQueryFanOut:
    @pytest.mark.asyncio
    async def test_query_all_systems(self):
        gw = MemoryGateway(
            continuum_memory=_make_continuum_mock(),
            knowledge_mound=_make_km_mock(),
            supermemory_adapter=_make_supermemory_mock(),
            claude_mem_adapter=_make_claude_mem_mock(),
        )
        resp = await gw.query(UnifiedMemoryQuery(query="test", limit=20))
        assert resp.total_found == 4  # 1 from each source
        assert len(resp.sources_queried) == 4

    @pytest.mark.asyncio
    async def test_query_specific_sources(self):
        gw = MemoryGateway(
            continuum_memory=_make_continuum_mock(),
            knowledge_mound=_make_km_mock(),
        )
        resp = await gw.query(UnifiedMemoryQuery(query="test", sources=["km"]))
        assert "km" in resp.sources_queried
        assert all(r.source_system == "km" for r in resp.results)

    @pytest.mark.asyncio
    async def test_query_unavailable_source_ignored(self):
        gw = MemoryGateway(continuum_memory=_make_continuum_mock())
        resp = await gw.query(UnifiedMemoryQuery(query="test", sources=["km", "continuum"]))
        # Only continuum is available
        assert resp.sources_queried == ["continuum"]

    @pytest.mark.asyncio
    async def test_query_handles_source_error(self):
        cm = _make_continuum_mock()
        cm.search.side_effect = RuntimeError("connection lost")
        gw = MemoryGateway(continuum_memory=cm)
        resp = await gw.query(UnifiedMemoryQuery(query="test"))
        assert "continuum" in resp.errors

    @pytest.mark.asyncio
    async def test_query_with_min_confidence_filter(self):
        gw = MemoryGateway(
            continuum_memory=_make_continuum_mock([FakeContinuumEntry(importance=0.3)]),
            knowledge_mound=_make_km_mock([FakeKMItem(confidence=0.9)]),
        )
        resp = await gw.query(UnifiedMemoryQuery(query="test", min_confidence=0.5))
        assert all(r.confidence >= 0.5 for r in resp.results)

    @pytest.mark.asyncio
    async def test_sequential_queries(self):
        gw = MemoryGateway(
            config=MemoryGatewayConfig(parallel_queries=False),
            continuum_memory=_make_continuum_mock(),
            knowledge_mound=_make_km_mock(),
        )
        resp = await gw.query(UnifiedMemoryQuery(query="test"))
        assert resp.total_found == 2


# ---------------------------------------------------------------------------
# Tests: Dedup
# ---------------------------------------------------------------------------


class TestDedup:
    @pytest.mark.asyncio
    async def test_dedup_removes_cross_system_duplicates(self):
        """Same content in continuum and KM should be deduped."""
        same_content = "identical rate limiting advice"
        gw = MemoryGateway(
            continuum_memory=_make_continuum_mock([FakeContinuumEntry(content=same_content)]),
            knowledge_mound=_make_km_mock([FakeKMItem(content=same_content)]),
        )
        resp = await gw.query(UnifiedMemoryQuery(query="test"))
        assert resp.duplicates_removed >= 1
        assert len(resp.results) == 1

    @pytest.mark.asyncio
    async def test_dedup_disabled(self):
        same_content = "identical content"
        gw = MemoryGateway(
            continuum_memory=_make_continuum_mock([FakeContinuumEntry(content=same_content)]),
            knowledge_mound=_make_km_mock([FakeKMItem(content=same_content)]),
        )
        resp = await gw.query(UnifiedMemoryQuery(query="test", dedup=False))
        assert resp.duplicates_removed == 0
        assert len(resp.results) == 2


# ---------------------------------------------------------------------------
# Tests: Ranking
# ---------------------------------------------------------------------------


class TestRanking:
    @pytest.mark.asyncio
    async def test_results_ranked_by_confidence(self):
        gw = MemoryGateway(
            continuum_memory=_make_continuum_mock(
                [FakeContinuumEntry(content="low", importance=0.2)]
            ),
            knowledge_mound=_make_km_mock([FakeKMItem(content="high", confidence=0.95)]),
        )
        resp = await gw.query(UnifiedMemoryQuery(query="test"))
        assert resp.results[0].source_system == "km"


# ---------------------------------------------------------------------------
# Tests: Store
# ---------------------------------------------------------------------------


class TestStore:
    @pytest.mark.asyncio
    async def test_store_without_coordinator(self):
        gw = MemoryGateway()
        result = await gw.store("test content")
        assert result == {}

    @pytest.mark.asyncio
    async def test_store_with_coordinator(self):
        gw = MemoryGateway(coordinator=MagicMock())
        result = await gw.store("new content")
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Tests: Graceful Degradation
# ---------------------------------------------------------------------------


class TestGracefulDegradation:
    @pytest.mark.asyncio
    async def test_single_source_failure_doesnt_break_others(self):
        bad_cm = _make_continuum_mock()
        bad_cm.search.side_effect = OSError("disk error")
        gw = MemoryGateway(
            continuum_memory=bad_cm,
            knowledge_mound=_make_km_mock(),
        )
        resp = await gw.query(UnifiedMemoryQuery(query="test"))
        assert len(resp.results) >= 1  # KM still works
        assert "continuum" in resp.errors

    @pytest.mark.asyncio
    async def test_all_sources_fail(self):
        bad_cm = _make_continuum_mock()
        bad_cm.search.side_effect = OSError("fail")
        bad_km = _make_km_mock()
        bad_km.query.side_effect = OSError("fail")
        gw = MemoryGateway(continuum_memory=bad_cm, knowledge_mound=bad_km)
        resp = await gw.query(UnifiedMemoryQuery(query="test"))
        assert resp.results == []
        assert len(resp.errors) == 2


# ---------------------------------------------------------------------------
# Tests: Stats
# ---------------------------------------------------------------------------


class TestGatewayStats:
    def test_stats_reflect_config(self):
        gw = MemoryGateway(
            config=MemoryGatewayConfig(enabled=True),
            continuum_memory=MagicMock(),
        )
        stats = gw.get_stats()
        assert stats["config"]["enabled"] is True
        assert "continuum" in stats["available_sources"]
        assert stats["has_coordinator"] is False
