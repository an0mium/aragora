"""E2E tests for the Unified Memory Gateway fan-out query.

Verifies that ``MemoryGateway.query()`` fans out to multiple backends,
deduplicates results, ranks them, and returns a merged response — exercising
the full read path end-to-end with lightweight in-process fakes.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest

from aragora.memory.gateway import (
    MemoryGateway,
    UnifiedMemoryQuery,
    UnifiedMemoryResponse,
    UnifiedMemoryResult,
)
from aragora.memory.gateway_config import MemoryGatewayConfig


# ---------------------------------------------------------------------------
# Lightweight in-process backend fakes (no mocking framework)
# ---------------------------------------------------------------------------


@dataclass
class FakeEntry:
    id: str = ""
    content: str = ""
    importance: float = 0.5
    surprise_score: float | None = None
    tier: str = "medium"


class FakeContinuumMemory:
    """Minimal ContinuumMemory stand-in that stores/retrieves entries."""

    def __init__(self, entries: list[FakeEntry] | None = None) -> None:
        self._entries = list(entries or [])

    def add(self, entry: FakeEntry) -> None:
        self._entries.append(entry)

    def retrieve(self, *, query: str = "", limit: int = 10) -> list[FakeEntry]:
        return self._entries[:limit]


@dataclass
class FakeKMItem:
    id: str = ""
    content: str = ""
    confidence: float = 0.5


@dataclass
class FakeKMQueryResult:
    items: list[FakeKMItem] = field(default_factory=list)


class FakeKnowledgeMound:
    """Minimal KnowledgeMound stand-in."""

    def __init__(self, items: list[FakeKMItem] | None = None) -> None:
        self._items = list(items or [])

    def add(self, item: FakeKMItem) -> None:
        self._items.append(item)

    async def query(self, *, query: str = "", limit: int = 10) -> FakeKMQueryResult:
        return FakeKMQueryResult(items=self._items[:limit])


@dataclass
class FakeSMResult:
    content: str = ""
    similarity: float = 0.5
    memory_id: str = ""


class FakeSupermemoryAdapter:
    """Minimal Supermemory adapter stand-in."""

    def __init__(self, results: list[FakeSMResult] | None = None) -> None:
        self._results = list(results or [])

    async def search_memories(self, *, query: str = "", limit: int = 10) -> list[FakeSMResult]:
        return self._results[:limit]


class FakeClaudeMemAdapter:
    """Minimal claude-mem adapter stand-in."""

    def __init__(self, observations: list[dict[str, Any]] | None = None) -> None:
        self._observations = list(observations or [])

    async def search_observations(
        self, *, query: str = "", limit: int = 10
    ) -> list[dict[str, Any]]:
        return self._observations[:limit]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def continuum() -> FakeContinuumMemory:
    return FakeContinuumMemory(
        [
            FakeEntry(id="cm_1", content="rate limiting best practices", importance=0.8),
            FakeEntry(id="cm_2", content="circuit breaker patterns", importance=0.7),
        ]
    )


@pytest.fixture()
def km() -> FakeKnowledgeMound:
    return FakeKnowledgeMound(
        [
            FakeKMItem(id="km_1", content="knowledge about rate limiting", confidence=0.9),
        ]
    )


@pytest.fixture()
def supermemory() -> FakeSupermemoryAdapter:
    return FakeSupermemoryAdapter(
        [
            FakeSMResult(content="supermemory rate limit info", similarity=0.75, memory_id="sm_1"),
        ]
    )


@pytest.fixture()
def claude_mem() -> FakeClaudeMemAdapter:
    return FakeClaudeMemAdapter(
        [
            {"id": "obs_1", "content": "claude-mem observation on rate limits", "metadata": {}},
        ]
    )


@pytest.fixture()
def gateway(
    continuum: FakeContinuumMemory,
    km: FakeKnowledgeMound,
    supermemory: FakeSupermemoryAdapter,
    claude_mem: FakeClaudeMemAdapter,
) -> MemoryGateway:
    return MemoryGateway.create_for_testing(
        continuum_memory=continuum,
        knowledge_mound=km,
        supermemory_adapter=supermemory,
        claude_mem_adapter=claude_mem,
    )


# ---------------------------------------------------------------------------
# E2E tests
# ---------------------------------------------------------------------------


class TestUnifiedMemoryGatewayE2E:
    """End-to-end tests for the gateway fan-out query pipeline."""

    @pytest.mark.asyncio
    async def test_fan_out_queries_all_sources(self, gateway: MemoryGateway) -> None:
        """Gateway fans out to all 4 backends and merges results."""
        resp = await gateway.query(UnifiedMemoryQuery(query="rate limiting", limit=20))

        assert isinstance(resp, UnifiedMemoryResponse)
        assert resp.sources_queried == ["continuum", "km", "supermemory", "claude_mem"]
        # 2 continuum + 1 km + 1 supermemory + 1 claude_mem = 5 (before dedup)
        assert resp.total_found == 5
        assert len(resp.results) >= 4  # at least 4 unique results
        assert resp.errors == {}
        assert resp.query_time_ms > 0

    @pytest.mark.asyncio
    async def test_fan_out_respects_source_filter(self, gateway: MemoryGateway) -> None:
        """When sources are restricted, only those backends are queried."""
        resp = await gateway.query(
            UnifiedMemoryQuery(query="rate limiting", sources=["continuum", "km"])
        )

        assert set(resp.sources_queried) == {"continuum", "km"}
        # Only results from continuum (2) and km (1)
        assert resp.total_found == 3

    @pytest.mark.asyncio
    async def test_dedup_removes_exact_duplicates(self) -> None:
        """Duplicate content across sources is deduplicated."""
        shared_content = "rate limiting best practices"
        gw = MemoryGateway.create_for_testing(
            continuum_memory=FakeContinuumMemory(
                [FakeEntry(id="cm_dup", content=shared_content, importance=0.8)]
            ),
            knowledge_mound=FakeKnowledgeMound(
                [FakeKMItem(id="km_dup", content=shared_content, confidence=0.9)]
            ),
        )

        resp = await gw.query(UnifiedMemoryQuery(query="rate limiting"))

        assert resp.total_found == 2  # both found
        assert resp.duplicates_removed == 1  # one removed as dup
        assert len(resp.results) == 1

    @pytest.mark.asyncio
    async def test_dedup_can_be_disabled(self) -> None:
        """Setting dedup=False keeps all results including duplicates."""
        shared = "same content"
        gw = MemoryGateway.create_for_testing(
            continuum_memory=FakeContinuumMemory([FakeEntry(id="a", content=shared)]),
            knowledge_mound=FakeKnowledgeMound([FakeKMItem(id="b", content=shared)]),
        )

        resp = await gw.query(UnifiedMemoryQuery(query="x", dedup=False))

        assert resp.duplicates_removed == 0
        assert len(resp.results) == 2

    @pytest.mark.asyncio
    async def test_min_confidence_filter(self, gateway: MemoryGateway) -> None:
        """Results below min_confidence are excluded."""
        resp = await gateway.query(UnifiedMemoryQuery(query="rate limiting", min_confidence=0.85))

        for r in resp.results:
            assert r.confidence >= 0.85

    @pytest.mark.asyncio
    async def test_results_ranked_by_confidence_and_source(self, gateway: MemoryGateway) -> None:
        """Results are ranked with KM (highest source priority) first when confidence is close."""
        resp = await gateway.query(UnifiedMemoryQuery(query="rate limiting", limit=20))

        if len(resp.results) >= 2:
            # Verify sort order is non-increasing by the ranking score
            scores = []
            source_priority = {"km": 1.0, "continuum": 0.9, "supermemory": 0.8, "claude_mem": 0.7}
            for r in resp.results:
                p = source_priority.get(r.source_system, 0.5)
                scores.append(r.confidence * 0.7 + p * 0.3)
            assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_limit_caps_results(self, gateway: MemoryGateway) -> None:
        """The limit parameter caps the number of returned results."""
        resp = await gateway.query(UnifiedMemoryQuery(query="rate limiting", limit=2))

        assert len(resp.results) <= 2

    @pytest.mark.asyncio
    async def test_empty_backends(self) -> None:
        """Gateway with empty backends returns empty response gracefully."""
        gw = MemoryGateway.create_for_testing(
            continuum_memory=FakeContinuumMemory([]),
            knowledge_mound=FakeKnowledgeMound([]),
        )

        resp = await gw.query(UnifiedMemoryQuery(query="anything"))

        assert resp.total_found == 0
        assert len(resp.results) == 0
        assert resp.errors == {}

    @pytest.mark.asyncio
    async def test_no_backends_configured(self) -> None:
        """Gateway with no backends returns empty response."""
        gw = MemoryGateway.create_for_testing()

        resp = await gw.query(UnifiedMemoryQuery(query="anything"))

        assert resp.sources_queried == []
        assert resp.total_found == 0

    @pytest.mark.asyncio
    async def test_sequential_mode(
        self,
        continuum: FakeContinuumMemory,
        km: FakeKnowledgeMound,
    ) -> None:
        """Fan-out also works in sequential (non-parallel) mode."""
        gw = MemoryGateway.create_for_testing(
            continuum_memory=continuum,
            knowledge_mound=km,
            parallel=False,
        )

        resp = await gw.query(UnifiedMemoryQuery(query="rate limiting"))

        assert set(resp.sources_queried) == {"continuum", "km"}
        assert resp.total_found == 3

    @pytest.mark.asyncio
    async def test_backend_error_isolated(self) -> None:
        """A failing backend reports error but other backends still return results."""

        class FailingKM:
            async def query(self, *, query: str = "", limit: int = 10) -> Any:
                raise RuntimeError("KM unavailable")

        gw = MemoryGateway.create_for_testing(
            continuum_memory=FakeContinuumMemory([FakeEntry(id="ok", content="still works")]),
            knowledge_mound=FailingKM(),  # type: ignore[arg-type]
        )

        resp = await gw.query(UnifiedMemoryQuery(query="test"))

        assert "km" in resp.errors
        assert resp.total_found == 1
        assert resp.results[0].source_system == "continuum"

    @pytest.mark.asyncio
    async def test_get_stats(self, gateway: MemoryGateway) -> None:
        """get_stats reflects available sources and config."""
        stats = gateway.get_stats()

        assert set(stats["available_sources"]) == {
            "continuum",
            "km",
            "supermemory",
            "claude_mem",
        }
        assert stats["config"]["enabled"] is True
        assert stats["config"]["parallel_queries"] is True

    @pytest.mark.asyncio
    async def test_create_for_testing_defaults(self) -> None:
        """create_for_testing produces a gateway with sensible test defaults."""
        gw = MemoryGateway.create_for_testing()

        assert gw.config.enabled is True
        assert gw.config.query_timeout_seconds == 5.0
        assert gw.config.parallel_queries is True

    @pytest.mark.asyncio
    async def test_result_metadata_preserved(self, gateway: MemoryGateway) -> None:
        """Source-specific metadata is preserved through the pipeline."""
        resp = await gateway.query(UnifiedMemoryQuery(query="rate limiting", limit=20))

        continuum_results = [r for r in resp.results if r.source_system == "continuum"]
        assert len(continuum_results) > 0
        # Continuum results carry tier metadata
        for r in continuum_results:
            assert "tier" in r.metadata
