"""Tests for the RLM Memory Navigator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from aragora.rlm.memory_navigator import (
    RLMMemoryNavigator,
    UnifiedMemoryContext,
    UnifiedMemoryItem,
    filter_by_confidence,
    filter_by_source,
    sort_by_confidence,
    sort_by_surprise,
)


# ---------------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------------


@dataclass
class FakeUnifiedMemoryResult:
    id: str = "r1"
    content: str = "rate limiting advice"
    source_system: str = "km"
    confidence: float = 0.9
    surprise_score: float | None = None
    content_hash: str = ""
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}


@dataclass
class FakeResponse:
    results: list[Any] | None = None
    total_found: int = 0
    sources_queried: list[str] | None = None
    duplicates_removed: int = 0
    query_time_ms: float = 1.5
    errors: dict[str, str] | None = None

    def __post_init__(self) -> None:
        if self.results is None:
            self.results = []
        if self.sources_queried is None:
            self.sources_queried = []
        if self.errors is None:
            self.errors = {}


def _make_gateway_mock(results: list[FakeUnifiedMemoryResult] | None = None):
    gw = AsyncMock()
    resp = FakeResponse(
        results=results or [FakeUnifiedMemoryResult()],
        total_found=len(results) if results else 1,
        sources_queried=["km", "continuum"],
    )
    gw.query = AsyncMock(return_value=resp)
    gw.knowledge_mound = None
    gw.continuum_memory = None
    return gw


# ---------------------------------------------------------------------------
# Tests: Dataclasses
# ---------------------------------------------------------------------------


class TestDataclasses:
    def test_unified_memory_item_defaults(self):
        item = UnifiedMemoryItem(id="x", source="km", content="test", confidence=0.5)
        assert item.surprise_score is None
        assert item.retention_action is None
        assert item.metadata == {}

    def test_unified_memory_context_defaults(self):
        ctx = UnifiedMemoryContext(
            items=[],
            by_source={},
            by_id={},
            total_items=0,
            source_counts={},
            query="test",
        )
        assert ctx.sources_queried == []
        assert ctx.duplicates_removed == 0
        assert ctx.query_time_ms == 0.0


# ---------------------------------------------------------------------------
# Tests: REPL Helpers
# ---------------------------------------------------------------------------


class TestREPLHelpers:
    def test_get_repl_helpers_keys(self):
        nav = RLMMemoryNavigator()
        helpers = nav.get_repl_helpers()
        assert "search_all" in helpers
        assert "build_context_hierarchy" in helpers
        assert "drill_into" in helpers
        assert "get_by_surprise" in helpers
        assert "filter_by_source" in helpers
        assert "filter_by_confidence" in helpers
        assert "sort_by_confidence" in helpers
        assert "sort_by_surprise" in helpers
        assert "UnifiedMemoryItem" in helpers
        assert "UnifiedMemoryContext" in helpers

    def test_helpers_include_types(self):
        nav = RLMMemoryNavigator()
        helpers = nav.get_repl_helpers()
        assert helpers["UnifiedMemoryItem"] is UnifiedMemoryItem
        assert helpers["UnifiedMemoryContext"] is UnifiedMemoryContext


# ---------------------------------------------------------------------------
# Tests: search_all
# ---------------------------------------------------------------------------


class TestSearchAll:
    @pytest.mark.asyncio
    async def test_search_all_delegates_to_gateway(self):
        gw = _make_gateway_mock(
            [
                FakeUnifiedMemoryResult(id="r1", content="rate limiting", source_system="km"),
            ]
        )
        nav = RLMMemoryNavigator(gateway=gw)
        results = await nav.search_all("rate limiting")
        assert len(results) == 1
        assert results[0].id == "r1"
        assert results[0].source == "km"
        gw.query.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_all_no_gateway(self):
        nav = RLMMemoryNavigator()
        results = await nav.search_all("test")
        assert results == []

    @pytest.mark.asyncio
    async def test_search_all_with_source_filter(self):
        gw = _make_gateway_mock()
        nav = RLMMemoryNavigator(gateway=gw)
        await nav.search_all("test", sources=["km"])
        call_args = gw.query.call_args[0][0]
        assert call_args.sources == ["km"]


# ---------------------------------------------------------------------------
# Tests: build_context_hierarchy
# ---------------------------------------------------------------------------


class TestBuildContextHierarchy:
    @pytest.mark.asyncio
    async def test_builds_context_from_gateway(self):
        gw = _make_gateway_mock(
            [
                FakeUnifiedMemoryResult(id="km_1", source_system="km"),
                FakeUnifiedMemoryResult(id="cm_1", source_system="continuum"),
            ]
        )
        nav = RLMMemoryNavigator(gateway=gw)
        ctx = await nav.build_context_hierarchy("rate limiting")
        assert ctx.total_items == 2
        assert "km" in ctx.source_counts
        assert "continuum" in ctx.source_counts
        assert ctx.query == "rate limiting"
        assert ctx.by_id["km_1"].source == "km"
        assert ctx.by_id["cm_1"].source == "continuum"

    @pytest.mark.asyncio
    async def test_build_context_no_gateway(self):
        nav = RLMMemoryNavigator()
        ctx = await nav.build_context_hierarchy("test")
        assert ctx.total_items == 0
        assert ctx.query == "test"

    @pytest.mark.asyncio
    async def test_build_context_with_retention_enrichment(self):
        gw = _make_gateway_mock(
            [
                FakeUnifiedMemoryResult(id="r1", source_system="km", surprise_score=0.8),
            ]
        )

        @dataclass
        class FakeDecision:
            action: str = "retain"

        gate = MagicMock()
        gate.batch_evaluate.return_value = [FakeDecision(action="consolidate")]

        nav = RLMMemoryNavigator(gateway=gw, retention_gate=gate)
        ctx = await nav.build_context_hierarchy("test")
        assert ctx.items[0].retention_action == "consolidate"
        gate.batch_evaluate.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: drill_into
# ---------------------------------------------------------------------------


class TestDrillInto:
    @pytest.mark.asyncio
    async def test_drill_into_no_gateway(self):
        nav = RLMMemoryNavigator()
        result = await nav.drill_into("km", "km_1")
        assert result["error"] == "No gateway configured"

    @pytest.mark.asyncio
    async def test_drill_into_km(self):
        @dataclass
        class FakeKMItem:
            content: str = "detailed content"
            confidence: float = 0.95
            tags: list[str] | None = None
            created_at: str = "2026-01-01"

        gw = _make_gateway_mock()
        km_mock = AsyncMock()
        km_mock.get_item = AsyncMock(return_value=FakeKMItem())
        gw.knowledge_mound = km_mock

        nav = RLMMemoryNavigator(gateway=gw)
        result = await nav.drill_into("km", "km_42")
        assert result["content"] == "detailed content"
        assert result["confidence"] == 0.95
        assert result["source"] == "km"

    @pytest.mark.asyncio
    async def test_drill_into_continuum(self):
        @dataclass
        class FakeEntry:
            content: str = "memory entry"
            importance: float = 0.7
            surprise_score: float = 0.6
            tier: str = "medium"

        gw = _make_gateway_mock()
        cm_mock = MagicMock()
        cm_mock.get_entry.return_value = FakeEntry()
        gw.continuum_memory = cm_mock

        nav = RLMMemoryNavigator(gateway=gw)
        result = await nav.drill_into("continuum", "cm_1")
        assert result["content"] == "memory entry"
        assert result["importance"] == 0.7
        assert result["tier"] == "medium"

    @pytest.mark.asyncio
    async def test_drill_into_fallback(self):
        gw = _make_gateway_mock()
        nav = RLMMemoryNavigator(gateway=gw)
        result = await nav.drill_into("supermemory", "sm_1")
        assert "note" in result


# ---------------------------------------------------------------------------
# Tests: get_by_surprise
# ---------------------------------------------------------------------------


class TestGetBySurprise:
    @pytest.mark.asyncio
    async def test_filters_by_surprise(self):
        gw = _make_gateway_mock(
            [
                FakeUnifiedMemoryResult(id="r1", surprise_score=0.9, source_system="continuum"),
                FakeUnifiedMemoryResult(id="r2", surprise_score=0.3, source_system="continuum"),
                FakeUnifiedMemoryResult(id="r3", surprise_score=0.7, source_system="continuum"),
            ]
        )
        nav = RLMMemoryNavigator(gateway=gw)
        results = await nav.get_by_surprise(min_surprise=0.5)
        assert len(results) == 2
        assert results[0].id == "r1"  # Highest surprise first
        assert results[1].id == "r3"

    @pytest.mark.asyncio
    async def test_get_by_surprise_no_gateway(self):
        nav = RLMMemoryNavigator()
        results = await nav.get_by_surprise()
        assert results == []

    @pytest.mark.asyncio
    async def test_get_by_surprise_excludes_none(self):
        gw = _make_gateway_mock(
            [
                FakeUnifiedMemoryResult(id="r1", surprise_score=None),
                FakeUnifiedMemoryResult(id="r2", surprise_score=0.8, source_system="continuum"),
            ]
        )
        nav = RLMMemoryNavigator(gateway=gw)
        results = await nav.get_by_surprise(min_surprise=0.5)
        assert len(results) == 1
        assert results[0].id == "r2"


# ---------------------------------------------------------------------------
# Tests: Filter/Sort Helpers
# ---------------------------------------------------------------------------


class TestFilterSortHelpers:
    def test_filter_by_source(self):
        items = [
            UnifiedMemoryItem(id="1", source="km", content="a", confidence=0.5),
            UnifiedMemoryItem(id="2", source="continuum", content="b", confidence=0.5),
            UnifiedMemoryItem(id="3", source="km", content="c", confidence=0.5),
        ]
        filtered = filter_by_source(items, "km")
        assert len(filtered) == 2
        assert all(i.source == "km" for i in filtered)

    def test_filter_by_confidence(self):
        items = [
            UnifiedMemoryItem(id="1", source="km", content="a", confidence=0.3),
            UnifiedMemoryItem(id="2", source="km", content="b", confidence=0.8),
            UnifiedMemoryItem(id="3", source="km", content="c", confidence=0.9),
        ]
        filtered = filter_by_confidence(items, threshold=0.7)
        assert len(filtered) == 2

    def test_sort_by_confidence(self):
        items = [
            UnifiedMemoryItem(id="1", source="km", content="a", confidence=0.3),
            UnifiedMemoryItem(id="2", source="km", content="b", confidence=0.9),
            UnifiedMemoryItem(id="3", source="km", content="c", confidence=0.6),
        ]
        sorted_items = sort_by_confidence(items)
        assert sorted_items[0].confidence == 0.9
        assert sorted_items[-1].confidence == 0.3

    def test_sort_by_surprise(self):
        items = [
            UnifiedMemoryItem(id="1", source="km", content="a", confidence=0.5, surprise_score=0.2),
            UnifiedMemoryItem(id="2", source="km", content="b", confidence=0.5, surprise_score=0.9),
            UnifiedMemoryItem(
                id="3", source="km", content="c", confidence=0.5, surprise_score=None
            ),
        ]
        sorted_items = sort_by_surprise(items)
        assert sorted_items[0].surprise_score == 0.9
        assert sorted_items[-1].surprise_score is None  # None sorted last


# ---------------------------------------------------------------------------
# Tests: Bridge Integration
# ---------------------------------------------------------------------------


class TestBridgeIntegration:
    def test_inject_unified_memory_helpers(self):
        """Test that AragoraRLM.inject_unified_memory_helpers returns helpers."""
        from aragora.rlm.bridge import AragoraRLM

        rlm = AragoraRLM()
        gw = _make_gateway_mock()
        result = rlm.inject_unified_memory_helpers(gateway=gw)
        assert "helpers" in result
        assert "search_all" in result["helpers"]
        assert "build_context_hierarchy" in result["helpers"]

    def test_inject_unified_memory_helpers_no_gateway(self):
        """Test graceful handling when gateway is None."""
        from aragora.rlm.bridge import AragoraRLM

        rlm = AragoraRLM()
        result = rlm.inject_unified_memory_helpers(gateway=None)
        assert "helpers" in result
        # Navigator still created, just with no gateway
        assert "search_all" in result["helpers"]
