"""
Tests for KnowledgeVelocityHandler.

Covers the Knowledge Mound Learning Velocity handler:
- GET /api/v1/knowledge/velocity - Get learning velocity metrics
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.knowledge.velocity import (
    ADAPTER_NAMES,
    KnowledgeVelocityHandler,
    _velocity_limiter,
)


# =============================================================================
# Helpers
# =============================================================================


def _body(result) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if result is None:
        return {}
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if result is None:
        return 0
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


def _data(result) -> dict:
    """Extract the response body (velocity endpoint does not use 'data' envelope)."""
    return _body(result)


# =============================================================================
# Mock data objects
# =============================================================================


@dataclass
class MockMoundStats:
    """Mock stats from KnowledgeMound.get_stats()."""

    total_nodes: int = 500
    nodes_by_type: dict = field(
        default_factory=lambda: {
            "continuum": 100,
            "consensus": 80,
            "critique": 60,
            "evidence": 50,
            "belief": 40,
        }
    )
    stale_nodes_count: int = 15
    average_confidence: float = 0.72


class _MockHTTPHandler:
    """Lightweight mock for the HTTP handler."""

    def __init__(
        self,
        method: str = "GET",
        path: str = "/api/v1/knowledge/velocity",
    ):
        self.command = method
        self.path = path
        self.client_address = ("127.0.0.1", 12345)
        self.headers = {"Content-Length": "0", "Host": "localhost:8080"}

    def get_client_ip(self) -> str:
        return "127.0.0.1"


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def handler():
    """Create a KnowledgeVelocityHandler instance."""
    return KnowledgeVelocityHandler(ctx={})


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """Reset the rate limiter between tests."""
    _velocity_limiter._requests.clear()
    yield


def _make_mock_mound(stats: MockMoundStats | None = None):
    """Create a mock KnowledgeMound."""
    mound = AsyncMock()
    mound.get_stats = AsyncMock(return_value=stats or MockMoundStats())
    return mound


# =============================================================================
# ROUTES / can_handle
# =============================================================================


class TestRoutes:
    """Test the can_handle method."""

    def test_can_handle_velocity(self, handler):
        assert handler.can_handle("/api/v1/knowledge/velocity")

    def test_can_handle_velocity_prefix(self, handler):
        """Should handle anything starting with the velocity prefix."""
        assert handler.can_handle("/api/v1/knowledge/velocity/extra")

    def test_rejects_other_paths(self, handler):
        assert not handler.can_handle("/api/v1/knowledge/gaps")
        assert not handler.can_handle("/api/v1/knowledge")
        assert not handler.can_handle("")


# =============================================================================
# ADAPTER_NAMES constant
# =============================================================================


class TestAdapterNames:
    """Test the ADAPTER_NAMES constant."""

    def test_adapter_names_count(self):
        """Should have 34 registered adapter names."""
        assert len(ADAPTER_NAMES) == 34

    def test_adapter_names_includes_key_adapters(self):
        """Should include well-known adapter names."""
        for name in [
            "continuum",
            "consensus",
            "critique",
            "evidence",
            "elo",
            "pulse",
            "debate",
            "compliance",
            "claude_mem",
        ]:
            assert name in ADAPTER_NAMES

    def test_adapter_names_unique(self):
        """All adapter names should be unique."""
        assert len(ADAPTER_NAMES) == len(set(ADAPTER_NAMES))


# =============================================================================
# GET /api/v1/knowledge/velocity (with KM available)
# =============================================================================


class TestVelocityMetricsWithKM:
    """Test velocity metrics when KnowledgeMound is available."""

    @pytest.mark.asyncio
    async def test_velocity_returns_all_fields(self, handler):
        """Should return all expected metric fields."""
        mound = _make_mock_mound()

        with patch(
            "aragora.server.handlers.knowledge.velocity._safe_import",
        ) as mock_import:
            # get_knowledge_mound returns our mock mound
            mock_get_km = MagicMock(return_value=mound)
            mock_get_continuum = MagicMock(return_value=None)

            def import_side_effect(module, attr):
                if attr == "get_knowledge_mound":
                    return mock_get_km
                if attr == "get_continuum_memory":
                    return mock_get_continuum
                return None

            mock_import.side_effect = import_side_effect

            result = await handler._get_velocity_metrics("default")

        data = _data(result)
        assert "total_entries" in data
        assert "entries_by_adapter" in data
        assert "adapter_count" in data
        assert "daily_growth" in data
        assert "growth_rate" in data
        assert "contradiction_count" in data
        assert "resolution_count" in data
        assert "resolution_rate" in data
        assert "confidence_distribution" in data
        assert "top_topics" in data
        assert "workspace_id" in data
        assert "timestamp" in data

    @pytest.mark.asyncio
    async def test_velocity_total_entries(self, handler):
        """Total entries should match mound stats."""
        mound = _make_mock_mound(MockMoundStats(total_nodes=250))

        with patch(
            "aragora.server.handlers.knowledge.velocity._safe_import",
        ) as mock_import:
            mock_get_km = MagicMock(return_value=mound)

            def import_side_effect(module, attr):
                if attr == "get_knowledge_mound":
                    return mock_get_km
                return None

            mock_import.side_effect = import_side_effect

            result = await handler._get_velocity_metrics("default")

        data = _data(result)
        assert data["total_entries"] == 250

    @pytest.mark.asyncio
    async def test_velocity_adapter_count(self, handler):
        """Adapter count should match ADAPTER_NAMES length."""
        result = await handler._get_velocity_metrics("default")
        data = _data(result)
        assert data["adapter_count"] == len(ADAPTER_NAMES)

    @pytest.mark.asyncio
    async def test_velocity_daily_growth_length(self, handler):
        """Daily growth should have 7 entries (last 7 days)."""
        mound = _make_mock_mound()

        with patch(
            "aragora.server.handlers.knowledge.velocity._safe_import",
        ) as mock_import:
            mock_get_km = MagicMock(return_value=mound)

            def import_side_effect(module, attr):
                if attr == "get_knowledge_mound":
                    return mock_get_km
                return None

            mock_import.side_effect = import_side_effect

            result = await handler._get_velocity_metrics("default")

        data = _data(result)
        assert len(data["daily_growth"]) == 7

    @pytest.mark.asyncio
    async def test_velocity_daily_growth_structure(self, handler):
        """Each daily growth entry should have date and count."""
        mound = _make_mock_mound()

        with patch(
            "aragora.server.handlers.knowledge.velocity._safe_import",
        ) as mock_import:
            mock_get_km = MagicMock(return_value=mound)

            def import_side_effect(module, attr):
                if attr == "get_knowledge_mound":
                    return mock_get_km
                return None

            mock_import.side_effect = import_side_effect

            result = await handler._get_velocity_metrics("default")

        data = _data(result)
        for entry in data["daily_growth"]:
            assert "date" in entry
            assert "count" in entry

    @pytest.mark.asyncio
    async def test_velocity_confidence_distribution_buckets(self, handler):
        """Confidence distribution should have 5 buckets."""
        mound = _make_mock_mound()

        with patch(
            "aragora.server.handlers.knowledge.velocity._safe_import",
        ) as mock_import:
            mock_get_km = MagicMock(return_value=mound)

            def import_side_effect(module, attr):
                if attr == "get_knowledge_mound":
                    return mock_get_km
                return None

            mock_import.side_effect = import_side_effect

            result = await handler._get_velocity_metrics("default")

        data = _data(result)
        buckets = data["confidence_distribution"]
        assert len(buckets) == 5
        expected_keys = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
        for key in expected_keys:
            assert key in buckets

    @pytest.mark.asyncio
    async def test_velocity_top_topics(self, handler):
        """Top topics should be derived from adapter entries, sorted by count."""
        stats = MockMoundStats(nodes_by_type={"continuum": 100, "consensus": 50, "critique": 30})
        mound = _make_mock_mound(stats)

        with patch(
            "aragora.server.handlers.knowledge.velocity._safe_import",
        ) as mock_import:
            mock_get_km = MagicMock(return_value=mound)

            def import_side_effect(module, attr):
                if attr == "get_knowledge_mound":
                    return mock_get_km
                return None

            mock_import.side_effect = import_side_effect

            result = await handler._get_velocity_metrics("default")

        data = _data(result)
        assert len(data["top_topics"]) == 3
        # Sorted descending by count
        assert data["top_topics"][0]["topic"] == "continuum"
        assert data["top_topics"][0]["count"] == 100

    @pytest.mark.asyncio
    async def test_velocity_contradiction_count(self, handler):
        """Contradiction count should come from stale_nodes_count."""
        stats = MockMoundStats(stale_nodes_count=42)
        mound = _make_mock_mound(stats)

        with patch(
            "aragora.server.handlers.knowledge.velocity._safe_import",
        ) as mock_import:
            mock_get_km = MagicMock(return_value=mound)

            def import_side_effect(module, attr):
                if attr == "get_knowledge_mound":
                    return mock_get_km
                return None

            mock_import.side_effect = import_side_effect

            result = await handler._get_velocity_metrics("default")

        data = _data(result)
        assert data["contradiction_count"] == 42


# =============================================================================
# GET /api/v1/knowledge/velocity (without KM)
# =============================================================================


class TestVelocityMetricsWithoutKM:
    """Test velocity metrics when KnowledgeMound is not available."""

    @pytest.mark.asyncio
    async def test_velocity_defaults_when_km_unavailable(self, handler):
        """When _safe_import returns None, should return default values."""
        with patch(
            "aragora.server.handlers.knowledge.velocity._safe_import",
            return_value=None,
        ):
            result = await handler._get_velocity_metrics("default")

        data = _data(result)
        assert data["total_entries"] == 0
        assert data["entries_by_adapter"] == {}
        assert data["contradiction_count"] == 0
        assert data["resolution_count"] == 0
        assert data["resolution_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_velocity_daily_growth_zeros(self, handler):
        """Daily growth should show all zeros when no entries."""
        with patch(
            "aragora.server.handlers.knowledge.velocity._safe_import",
            return_value=None,
        ):
            result = await handler._get_velocity_metrics("default")

        data = _data(result)
        assert len(data["daily_growth"]) == 7
        for entry in data["daily_growth"]:
            assert entry["count"] == 0

    @pytest.mark.asyncio
    async def test_velocity_growth_rate_zero(self, handler):
        """Growth rate should be 0.0 when no entries."""
        with patch(
            "aragora.server.handlers.knowledge.velocity._safe_import",
            return_value=None,
        ):
            result = await handler._get_velocity_metrics("default")

        data = _data(result)
        assert data["growth_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_velocity_empty_top_topics(self, handler):
        """Top topics should be empty when no entries."""
        with patch(
            "aragora.server.handlers.knowledge.velocity._safe_import",
            return_value=None,
        ):
            result = await handler._get_velocity_metrics("default")

        data = _data(result)
        assert data["top_topics"] == []


# =============================================================================
# Continuum memory integration
# =============================================================================


class TestContinuumIntegration:
    """Test resolution_count from continuum memory."""

    @pytest.mark.asyncio
    async def test_resolution_count_from_continuum(self, handler):
        """resolution_count should come from continuum._km_adapter."""
        mock_adapter = MagicMock()
        mock_adapter.get_stats.return_value = {"km_validated_entries": 25}

        mock_continuum = MagicMock()
        mock_continuum._km_adapter = mock_adapter

        with patch(
            "aragora.server.handlers.knowledge.velocity._safe_import",
        ) as mock_import:

            def import_side_effect(module, attr):
                if attr == "get_continuum_memory":
                    return MagicMock(return_value=mock_continuum)
                return None

            mock_import.side_effect = import_side_effect

            result = await handler._get_velocity_metrics("default")

        data = _data(result)
        assert data["resolution_count"] == 25

    @pytest.mark.asyncio
    async def test_resolution_rate_calculation(self, handler):
        """resolution_rate = resolution / (contradictions + resolutions)."""
        mock_adapter = MagicMock()
        mock_adapter.get_stats.return_value = {"km_validated_entries": 10}

        mock_continuum = MagicMock()
        mock_continuum._km_adapter = mock_adapter

        stats = MockMoundStats(stale_nodes_count=40)
        mound = _make_mock_mound(stats)

        with patch(
            "aragora.server.handlers.knowledge.velocity._safe_import",
        ) as mock_import:

            def import_side_effect(module, attr):
                if attr == "get_knowledge_mound":
                    return MagicMock(return_value=mound)
                if attr == "get_continuum_memory":
                    return MagicMock(return_value=mock_continuum)
                return None

            mock_import.side_effect = import_side_effect

            result = await handler._get_velocity_metrics("default")

        data = _data(result)
        # 10 / (40 + 10) = 0.2
        assert data["resolution_rate"] == 0.2

    @pytest.mark.asyncio
    async def test_resolution_rate_zero_when_both_zero(self, handler):
        """resolution_rate should be 0.0 when both counts are zero."""
        with patch(
            "aragora.server.handlers.knowledge.velocity._safe_import",
            return_value=None,
        ):
            result = await handler._get_velocity_metrics("default")

        data = _data(result)
        assert data["resolution_rate"] == 0.0


# =============================================================================
# handle() dispatch
# =============================================================================


class TestHandleDispatch:
    """Test the handle() route dispatch."""

    @pytest.mark.asyncio
    async def test_handle_routes_to_velocity(self, handler):
        """GET /api/v1/knowledge/velocity should return velocity data."""
        with patch(
            "aragora.server.handlers.knowledge.velocity._safe_import",
            return_value=None,
        ):
            result = await handler.handle(
                "/api/v1/knowledge/velocity",
                {},
                _MockHTTPHandler(),
            )

        data = _data(result)
        assert "total_entries" in data
        assert "workspace_id" in data

    @pytest.mark.asyncio
    async def test_handle_returns_none_for_unknown(self, handler):
        """Non-matching paths should return None."""
        result = await handler.handle(
            "/api/v1/knowledge/velocity/unknown",
            {},
            _MockHTTPHandler(path="/api/v1/knowledge/velocity/unknown"),
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_handle_workspace_from_params(self, handler):
        """workspace_id should come from query_params."""
        with patch(
            "aragora.server.handlers.knowledge.velocity._safe_import",
            return_value=None,
        ):
            result = await handler.handle(
                "/api/v1/knowledge/velocity",
                {"workspace_id": "my-ws"},
                _MockHTTPHandler(),
            )

        data = _data(result)
        assert data["workspace_id"] == "my-ws"


# =============================================================================
# Rate limiting
# =============================================================================


class TestRateLimiting:
    """Test rate limiting on the velocity handler."""

    @pytest.mark.asyncio
    async def test_rate_limit_allows_normal(self, handler):
        """Normal requests should be allowed."""
        with patch(
            "aragora.server.handlers.knowledge.velocity._safe_import",
            return_value=None,
        ):
            result = await handler.handle(
                "/api/v1/knowledge/velocity",
                {},
                _MockHTTPHandler(),
            )
        assert _status(result) == 200

    @pytest.mark.asyncio
    async def test_rate_limit_rejects_excessive(self, handler):
        """Exceeding rate limit should return 429."""
        # Exhaust the rate limit
        for _ in range(35):
            _velocity_limiter.is_allowed("127.0.0.1")

        with patch(
            "aragora.server.handlers.knowledge.velocity._safe_import",
            return_value=None,
        ):
            result = await handler.handle(
                "/api/v1/knowledge/velocity",
                {},
                _MockHTTPHandler(),
            )
        assert _status(result) == 429


# =============================================================================
# Error resilience
# =============================================================================


class TestErrorResilience:
    """Test graceful degradation when backends fail."""

    @pytest.mark.asyncio
    async def test_km_stats_error_graceful(self, handler):
        """RuntimeError from mound.get_stats should be caught gracefully."""
        mound = AsyncMock()
        mound.get_stats = AsyncMock(side_effect=RuntimeError("db down"))

        with patch(
            "aragora.server.handlers.knowledge.velocity._safe_import",
        ) as mock_import:
            mock_get_km = MagicMock(return_value=mound)

            def import_side_effect(module, attr):
                if attr == "get_knowledge_mound":
                    return mock_get_km
                return None

            mock_import.side_effect = import_side_effect

            result = await handler._get_velocity_metrics("default")

        data = _data(result)
        # Should still return valid response with defaults
        assert data["total_entries"] == 0
        assert data["workspace_id"] == "default"

    @pytest.mark.asyncio
    async def test_continuum_error_graceful(self, handler):
        """Error from continuum should be caught gracefully."""
        mock_continuum_fn = MagicMock(side_effect=RuntimeError("fail"))

        with patch(
            "aragora.server.handlers.knowledge.velocity._safe_import",
        ) as mock_import:

            def import_side_effect(module, attr):
                if attr == "get_continuum_memory":
                    return mock_continuum_fn
                return None

            mock_import.side_effect = import_side_effect

            result = await handler._get_velocity_metrics("default")

        data = _data(result)
        assert data["resolution_count"] == 0


# =============================================================================
# _safe_import helper
# =============================================================================


class TestSafeImport:
    """Test the _safe_import helper function."""

    def test_safe_import_returns_none_on_import_error(self):
        from aragora.server.handlers.knowledge.velocity import _safe_import

        result = _safe_import("nonexistent.module.path", "SomeClass")
        assert result is None

    def test_safe_import_returns_none_on_attribute_error(self):
        from aragora.server.handlers.knowledge.velocity import _safe_import

        result = _safe_import("json", "nonexistent_attr_xyz")
        assert result is None

    def test_safe_import_returns_valid_attr(self):
        from aragora.server.handlers.knowledge.velocity import _safe_import

        result = _safe_import("json", "dumps")
        assert result is not None
        assert callable(result)
