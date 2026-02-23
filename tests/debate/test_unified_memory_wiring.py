"""Tests for unified memory arena wiring and HTTP handler."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from aragora.debate.arena_sub_configs import UnifiedMemorySubConfig
from aragora.server.handlers.memory.unified_handler import UnifiedMemoryHandler


# ---------------------------------------------------------------------------
# Tests: UnifiedMemorySubConfig
# ---------------------------------------------------------------------------


class TestUnifiedMemorySubConfig:
    def test_defaults(self):
        cfg = UnifiedMemorySubConfig()
        assert cfg.enable_unified_memory is False
        assert cfg.enable_retention_gate is False
        assert cfg.query_timeout_seconds == 15.0
        assert cfg.dedup_threshold == 0.95
        assert cfg.default_sources is None
        assert cfg.parallel_queries is True

    def test_custom_values(self):
        cfg = UnifiedMemorySubConfig(
            enable_unified_memory=True,
            enable_retention_gate=True,
            query_timeout_seconds=5.0,
            default_sources=["km", "continuum"],
        )
        assert cfg.enable_unified_memory is True
        assert cfg.enable_retention_gate is True
        assert cfg.default_sources == ["km", "continuum"]


# ---------------------------------------------------------------------------
# Tests: Gateway Creation from Config
# ---------------------------------------------------------------------------


class TestGatewayFromConfig:
    def test_gateway_created_when_enabled(self):
        """MemoryGateway can be created from UnifiedMemorySubConfig."""
        from aragora.memory.gateway import MemoryGateway
        from aragora.memory.gateway_config import MemoryGatewayConfig

        cfg = UnifiedMemorySubConfig(
            enable_unified_memory=True,
            query_timeout_seconds=10.0,
            dedup_threshold=0.9,
            parallel_queries=False,
        )

        gw_config = MemoryGatewayConfig(
            enabled=cfg.enable_unified_memory,
            query_timeout_seconds=cfg.query_timeout_seconds,
            dedup_threshold=cfg.dedup_threshold,
            default_sources=cfg.default_sources,
            parallel_queries=cfg.parallel_queries,
        )

        gateway = MemoryGateway(config=gw_config)
        stats = gateway.get_stats()
        assert stats["config"]["enabled"] is True
        assert stats["config"]["query_timeout_seconds"] == 10.0
        assert stats["config"]["parallel_queries"] is False

    def test_gateway_not_created_when_disabled(self):
        """No gateway when unified memory is disabled."""
        cfg = UnifiedMemorySubConfig(enable_unified_memory=False)
        # Simulates orchestrator check
        gateway = None
        if cfg.enable_unified_memory:
            from aragora.memory.gateway import MemoryGateway

            gateway = MemoryGateway()
        assert gateway is None

    def test_retention_gate_wired_when_both_enabled(self):
        """RetentionGate wired when both flags enabled."""
        from aragora.memory.gateway import MemoryGateway
        from aragora.memory.retention_gate import RetentionGate, RetentionGateConfig

        cfg = UnifiedMemorySubConfig(
            enable_unified_memory=True,
            enable_retention_gate=True,
        )

        gate = None
        if cfg.enable_retention_gate:
            gate = RetentionGate(config=RetentionGateConfig())

        gateway = MemoryGateway(retention_gate=gate)
        assert gateway.retention_gate is not None

    def test_retention_gate_not_wired_when_disabled(self):
        """RetentionGate not wired when enable_retention_gate is False."""
        from aragora.memory.gateway import MemoryGateway

        cfg = UnifiedMemorySubConfig(
            enable_unified_memory=True,
            enable_retention_gate=False,
        )

        gate = None
        if cfg.enable_retention_gate:
            from aragora.memory.retention_gate import RetentionGate

            gate = RetentionGate()

        gateway = MemoryGateway(retention_gate=gate)
        assert gateway.retention_gate is None


# ---------------------------------------------------------------------------
# Tests: Unified Memory Handler
# ---------------------------------------------------------------------------


@dataclass
class FakeResult:
    id: str = "r1"
    content: str = "test content"
    source_system: str = "km"
    confidence: float = 0.9
    surprise_score: float | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}


@dataclass
class FakeResponse:
    results: list[Any] | None = None
    total_found: int = 1
    sources_queried: list[str] | None = None
    duplicates_removed: int = 0
    query_time_ms: float = 2.5
    errors: dict[str, str] | None = None

    def __post_init__(self) -> None:
        if self.results is None:
            self.results = [FakeResult()]
        if self.sources_queried is None:
            self.sources_queried = ["km"]
        if self.errors is None:
            self.errors = {}


class TestUnifiedMemoryHandler:
    @pytest.mark.asyncio
    async def test_search_returns_results(self):
        gw = AsyncMock()
        gw.query = AsyncMock(return_value=FakeResponse())
        handler = UnifiedMemoryHandler(gateway=gw)
        result = await handler.handle_search({"query": "rate limiting"})
        assert len(result["results"]) == 1
        assert result["results"][0]["id"] == "r1"
        assert result["total_found"] == 1
        assert result["sources_queried"] == ["km"]

    @pytest.mark.asyncio
    async def test_search_missing_query(self):
        handler = UnifiedMemoryHandler(gateway=AsyncMock())
        result = await handler.handle_search({})
        assert "error" in result
        assert "query" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_search_no_gateway(self):
        handler = UnifiedMemoryHandler()
        result = await handler.handle_search({"query": "test"})
        assert "error" in result
        assert result["results"] == []

    @pytest.mark.asyncio
    async def test_search_passes_all_params(self):
        gw = AsyncMock()
        gw.query = AsyncMock(return_value=FakeResponse())
        handler = UnifiedMemoryHandler(gateway=gw)
        await handler.handle_search(
            {
                "query": "test",
                "limit": 5,
                "min_confidence": 0.5,
                "sources": ["km"],
                "dedup": False,
            }
        )
        call_args = gw.query.call_args[0][0]
        assert call_args.query == "test"
        assert call_args.limit == 5
        assert call_args.min_confidence == 0.5
        assert call_args.sources == ["km"]
        assert call_args.dedup is False

    @pytest.mark.asyncio
    async def test_stats_returns_gateway_stats(self):
        gw = MagicMock()
        gw.get_stats.return_value = {
            "available_sources": ["km"],
            "config": {"enabled": True},
        }
        handler = UnifiedMemoryHandler(gateway=gw)
        result = await handler.handle_stats()
        assert result["available_sources"] == ["km"]

    @pytest.mark.asyncio
    async def test_stats_no_gateway(self):
        handler = UnifiedMemoryHandler()
        result = await handler.handle_stats()
        assert "error" in result
