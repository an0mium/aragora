"""
Tests for KnowledgeMound Dashboard Operations Mixin.

Tests dashboard and metrics endpoints:
- GET /api/knowledge/mound/dashboard/health - Health status
- GET /api/knowledge/mound/dashboard/metrics - Detailed metrics
- GET /api/knowledge/mound/dashboard/adapters - Adapter status
- GET /api/knowledge/mound/dashboard/queries - Federated query stats
- POST /api/knowledge/mound/dashboard/metrics/reset - Reset metrics
- GET /api/knowledge/mound/dashboard/batcher-stats - Batcher statistics
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.base import HandlerResult

pytestmark = pytest.mark.asyncio


# ===========================================================================
# Mock Classes
# ===========================================================================


class MockHealthReport:
    """Mock health report."""

    def __init__(
        self,
        status: str = "healthy",
        checks: dict | None = None,
        recommendations: list | None = None,
    ):
        self.status = status
        self.checks = checks or {"mound": True, "adapters": True, "memory": True}
        self.recommendations = recommendations or []
        self.timestamp = datetime.now(timezone.utc)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "checks": self.checks,
            "recommendations": self.recommendations,
            "timestamp": self.timestamp.isoformat(),
        }


class MockMetrics:
    """Mock metrics instance."""

    def __init__(self):
        self.stats = {
            "operations": {"read": 100, "write": 50, "delete": 10},
            "latency_ms": {"avg": 25.5, "p95": 100, "p99": 250},
            "cache_hits": 500,
            "cache_misses": 50,
        }
        self.health_report = MockHealthReport()
        self.uptime_seconds = 3600

    def get_health(self) -> MockHealthReport:
        return self.health_report

    def to_dict(self) -> dict[str, Any]:
        return {
            "stats": self.stats,
            "health": self.health_report.to_dict(),
            "uptime_seconds": self.uptime_seconds,
        }

    def reset(self) -> None:
        self.stats = {"operations": {}, "latency_ms": {}, "cache_hits": 0, "cache_misses": 0}


class MockCoordinator:
    """Mock bidirectional coordinator."""

    def __init__(self, adapters: dict | None = None):
        self._adapters = adapters or {
            "continuum": {
                "enabled": True,
                "priority": 10,
                "forward_sync_count": 100,
                "reverse_sync_count": 50,
                "last_sync": datetime.now(timezone.utc).isoformat(),
                "errors": 0,
            },
            "consensus": {
                "enabled": True,
                "priority": 5,
                "forward_sync_count": 75,
                "reverse_sync_count": 25,
                "last_sync": datetime.now(timezone.utc).isoformat(),
                "errors": 2,
            },
        }

    def get_stats(self) -> dict[str, Any]:
        return {
            "adapters": self._adapters,
            "total_adapters": len(self._adapters),
            "enabled_adapters": sum(1 for a in self._adapters.values() if a.get("enabled")),
            "last_full_sync": datetime.now(timezone.utc).isoformat(),
        }


class MockAggregator:
    """Mock federated query aggregator."""

    def __init__(self):
        self.total_queries = 1000
        self.successful_queries = 950
        self.sources = ["mound", "consensus", "continuum"]

    def get_stats(self) -> dict[str, Any]:
        return {
            "total_queries": self.total_queries,
            "successful_queries": self.successful_queries,
            "success_rate": self.successful_queries / self.total_queries
            if self.total_queries
            else 0,
            "sources": self.sources,
        }


class MockKMBridge:
    """Mock WebSocket bridge."""

    def __init__(self):
        self.running = True
        self.total_events = 500
        self.total_batches = 50

    def get_stats(self) -> dict[str, Any]:
        return {
            "running": self.running,
            "total_events_queued": self.total_events,
            "total_events_emitted": self.total_events - 10,
            "total_batches_emitted": self.total_batches,
            "average_batch_size": (self.total_events - 10) / self.total_batches
            if self.total_batches
            else 0,
            "pending_events": 10,
        }


class MockDashboardHandler:
    """Mock handler class that uses DashboardOperationsMixin."""

    def __init__(self, mound=None, server_context=None):
        self._mound = mound
        self.server_context = server_context or {}

    def _get_mound(self):
        return self._mound

    async def _check_knowledge_permission(self, request, action: str = "read"):
        """Mock RBAC check - always allow by default."""
        return None


# Create a test instance that inherits from both
def create_handler(mound=None, server_context=None, rbac_denied=False):
    """Create a handler instance for testing."""
    from aragora.server.handlers.knowledge_base.mound.dashboard import DashboardOperationsMixin

    class TestHandler(DashboardOperationsMixin):
        def __init__(self, mound, server_context, rbac_denied):
            self._mound = mound
            self.server_context = server_context or {}
            self._rbac_denied = rbac_denied

        def _get_mound(self):
            return self._mound

        async def _check_knowledge_permission(self, request, action: str = "read"):
            if self._rbac_denied:
                from aragora.server.handlers.base import error_response

                return error_response("Permission denied: knowledge.read required", status=403)
            return None

    return TestHandler(mound, server_context, rbac_denied)


def create_mock_request(user_id: str = "user_123", roles: set | None = None) -> MagicMock:
    """Create a mock aiohttp request."""
    request = MagicMock()
    request.get = MagicMock(
        side_effect=lambda key, default=None: {
            "user_id": user_id,
            "org_id": "org_123",
            "roles": roles or {"admin"},
        }.get(key, default)
    )
    return request


def parse_result(result: HandlerResult) -> tuple[dict[str, Any], bool]:
    """Parse handler result into (data, success)."""
    if hasattr(result, "body"):
        body = json.loads(result.body) if result.body else {}
        return body.get("data", body), result.status_code == 200
    return result.data if hasattr(result, "data") else {}, getattr(result, "success", False)


# ===========================================================================
# Test: Dashboard Health
# ===========================================================================


class TestDashboardHealth:
    """Tests for GET /api/knowledge/mound/dashboard/health."""

    @pytest.mark.asyncio
    async def test_health_success(self):
        """Should return health status successfully."""
        mock_metrics = MockMetrics()
        handler = create_handler()
        request = create_mock_request()

        with patch(
            "aragora.knowledge.mound.metrics.get_metrics",
            return_value=mock_metrics,
        ):
            result = await handler.handle_dashboard_health(request)

        data, success = parse_result(result)
        assert success
        assert data["status"] == "healthy"
        assert "checks" in data
        assert "recommendations" in data

    @pytest.mark.asyncio
    async def test_health_degraded(self):
        """Should return degraded status when checks fail."""
        mock_metrics = MockMetrics()
        mock_metrics.health_report = MockHealthReport(
            status="degraded",
            checks={"mound": True, "adapters": False, "memory": True},
            recommendations=["Check adapter connections"],
        )
        handler = create_handler()
        request = create_mock_request()

        with patch(
            "aragora.knowledge.mound.metrics.get_metrics",
            return_value=mock_metrics,
        ):
            result = await handler.handle_dashboard_health(request)

        data, success = parse_result(result)
        assert success
        assert data["status"] == "degraded"
        assert data["checks"]["adapters"] is False
        assert len(data["recommendations"]) == 1

    @pytest.mark.asyncio
    async def test_health_metrics_unavailable(self):
        """Should return 503 when metrics module unavailable."""
        handler = create_handler()
        request = create_mock_request()

        with patch.dict("sys.modules", {"aragora.knowledge.mound.metrics": None}):
            result = await handler.handle_dashboard_health(request)

        assert result.status_code == 503

    @pytest.mark.asyncio
    async def test_health_rbac_denied(self):
        """Should return 403 when RBAC denies access."""
        handler = create_handler(rbac_denied=True)
        request = create_mock_request(roles={"guest"})

        result = await handler.handle_dashboard_health(request)

        assert result.status_code == 403


# ===========================================================================
# Test: Dashboard Metrics
# ===========================================================================


class TestDashboardMetrics:
    """Tests for GET /api/knowledge/mound/dashboard/metrics."""

    @pytest.mark.asyncio
    async def test_metrics_success(self):
        """Should return detailed metrics successfully."""
        mock_metrics = MockMetrics()
        handler = create_handler()
        request = create_mock_request()

        with patch(
            "aragora.knowledge.mound.metrics.get_metrics",
            return_value=mock_metrics,
        ):
            result = await handler.handle_dashboard_metrics(request)

        data, success = parse_result(result)
        assert success
        assert "stats" in data
        assert "health" in data
        assert "uptime_seconds" in data

    @pytest.mark.asyncio
    async def test_metrics_with_operations(self):
        """Should include operation statistics."""
        mock_metrics = MockMetrics()
        handler = create_handler()
        request = create_mock_request()

        with patch(
            "aragora.knowledge.mound.metrics.get_metrics",
            return_value=mock_metrics,
        ):
            result = await handler.handle_dashboard_metrics(request)

        data, success = parse_result(result)
        assert success
        assert data["stats"]["operations"]["read"] == 100
        assert data["stats"]["operations"]["write"] == 50

    @pytest.mark.asyncio
    async def test_metrics_unavailable(self):
        """Should return 503 when metrics module unavailable."""
        handler = create_handler()
        request = create_mock_request()

        with patch.dict("sys.modules", {"aragora.knowledge.mound.metrics": None}):
            result = await handler.handle_dashboard_metrics(request)

        assert result.status_code == 503

    @pytest.mark.asyncio
    async def test_metrics_rbac_denied(self):
        """Should return 403 when RBAC denies access."""
        handler = create_handler(rbac_denied=True)
        request = create_mock_request(roles={"viewer"})

        result = await handler.handle_dashboard_metrics(request)

        assert result.status_code == 403


# ===========================================================================
# Test: Dashboard Adapters
# ===========================================================================


class TestDashboardAdapters:
    """Tests for GET /api/knowledge/mound/dashboard/adapters."""

    @pytest.mark.asyncio
    async def test_adapters_with_coordinator(self):
        """Should return adapter status when coordinator available."""
        mock_mound = MagicMock()
        mock_mound._coordinator = MockCoordinator()
        handler = create_handler(mound=mock_mound, server_context={})
        request = create_mock_request()

        result = await handler.handle_dashboard_adapters(request)

        data, success = parse_result(result)
        assert success
        assert data["total"] == 2
        assert data["enabled"] == 2
        assert len(data["adapters"]) == 2

    @pytest.mark.asyncio
    async def test_adapters_from_server_context(self):
        """Should get coordinator from server context if not on mound."""
        mock_mound = MagicMock()
        mock_mound._coordinator = None
        mock_coordinator = MockCoordinator()
        handler = create_handler(
            mound=mock_mound, server_context={"km_coordinator": mock_coordinator}
        )
        request = create_mock_request()

        result = await handler.handle_dashboard_adapters(request)

        data, success = parse_result(result)
        assert success
        assert data["total"] == 2

    @pytest.mark.asyncio
    async def test_adapters_no_mound(self):
        """Should return empty list when mound not initialized."""
        handler = create_handler(mound=None)
        request = create_mock_request()

        result = await handler.handle_dashboard_adapters(request)

        data, success = parse_result(result)
        assert success
        assert data["adapters"] == []
        assert data["total"] == 0
        assert "not initialized" in data.get("message", "")

    @pytest.mark.asyncio
    async def test_adapters_no_coordinator(self):
        """Should return empty list when coordinator unavailable."""
        mock_mound = MagicMock()
        mock_mound._coordinator = None
        handler = create_handler(mound=mock_mound, server_context={})
        request = create_mock_request()

        result = await handler.handle_dashboard_adapters(request)

        data, success = parse_result(result)
        assert success
        assert data["adapters"] == []
        assert "No coordinator" in data.get("message", "")

    @pytest.mark.asyncio
    async def test_adapters_with_errors(self):
        """Should include error counts for adapters."""
        adapters = {
            "continuum": {"enabled": True, "priority": 10, "errors": 5},
            "consensus": {"enabled": False, "priority": 5, "errors": 0},
        }
        mock_mound = MagicMock()
        mock_mound._coordinator = MockCoordinator(adapters)
        handler = create_handler(mound=mock_mound)
        request = create_mock_request()

        result = await handler.handle_dashboard_adapters(request)

        data, success = parse_result(result)
        assert success
        adapter_names = [a["name"] for a in data["adapters"]]
        assert "continuum" in adapter_names

    @pytest.mark.asyncio
    async def test_adapters_rbac_denied(self):
        """Should return 403 when RBAC denies access."""
        handler = create_handler(rbac_denied=True)
        request = create_mock_request()

        result = await handler.handle_dashboard_adapters(request)

        assert result.status_code == 403


# ===========================================================================
# Test: Dashboard Queries
# ===========================================================================


class TestDashboardQueries:
    """Tests for GET /api/knowledge/mound/dashboard/queries."""

    @pytest.mark.asyncio
    async def test_queries_with_aggregator(self):
        """Should return query stats when aggregator available."""
        mock_aggregator = MockAggregator()
        handler = create_handler(server_context={"km_aggregator": mock_aggregator})
        request = create_mock_request()

        with patch(
            "aragora.knowledge.mound.federated_query.FederatedQueryAggregator",
            return_value=MockAggregator(),
        ):
            result = await handler.handle_dashboard_queries(request)

        data, success = parse_result(result)
        assert success
        assert data["total_queries"] == 1000
        assert data["successful_queries"] == 950
        assert data["success_rate"] == 0.95

    @pytest.mark.asyncio
    async def test_queries_no_aggregator_creates_new(self):
        """Should create temporary aggregator when none available."""
        handler = create_handler(server_context={})
        request = create_mock_request()

        with patch(
            "aragora.knowledge.mound.federated_query.FederatedQueryAggregator",
            return_value=MockAggregator(),
        ):
            result = await handler.handle_dashboard_queries(request)

        data, success = parse_result(result)
        assert success
        assert "total_queries" in data

    @pytest.mark.asyncio
    async def test_queries_module_unavailable(self):
        """Should return 503 when federated query module unavailable."""
        handler = create_handler()
        request = create_mock_request()

        with patch.dict("sys.modules", {"aragora.knowledge.mound.federated_query": None}):
            result = await handler.handle_dashboard_queries(request)

        assert result.status_code == 503


# ===========================================================================
# Test: Dashboard Metrics Reset
# ===========================================================================


class TestDashboardMetricsReset:
    """Tests for POST /api/knowledge/mound/dashboard/metrics/reset."""

    @pytest.mark.asyncio
    async def test_reset_success(self):
        """Should reset metrics successfully."""
        mock_metrics = MockMetrics()
        handler = create_handler()
        request = create_mock_request()

        with patch(
            "aragora.knowledge.mound.metrics.get_metrics",
            return_value=mock_metrics,
        ):
            result = await handler.handle_dashboard_metrics_reset(request)

        data, success = parse_result(result)
        assert success
        assert "reset" in data.get("message", "").lower()

    @pytest.mark.asyncio
    async def test_reset_unavailable(self):
        """Should return 503 when metrics module unavailable."""
        handler = create_handler()
        request = create_mock_request()

        with patch.dict("sys.modules", {"aragora.knowledge.mound.metrics": None}):
            result = await handler.handle_dashboard_metrics_reset(request)

        assert result.status_code == 503


# ===========================================================================
# Test: Dashboard Batcher Stats
# ===========================================================================


class TestDashboardBatcherStats:
    """Tests for GET /api/knowledge/mound/dashboard/batcher-stats."""

    @pytest.mark.asyncio
    async def test_batcher_stats_success(self):
        """Should return batcher stats when bridge available."""
        handler = create_handler()
        request = create_mock_request()

        with patch(
            "aragora.knowledge.mound.websocket_bridge.get_km_bridge",
            return_value=MockKMBridge(),
        ):
            result = await handler.handle_dashboard_batcher_stats(request)

        data, success = parse_result(result)
        assert success
        assert data["running"] is True
        assert data["total_events_queued"] == 500
        assert data["total_batches_emitted"] == 50

    @pytest.mark.asyncio
    async def test_batcher_stats_no_bridge(self):
        """Should return not initialized when bridge unavailable."""
        handler = create_handler()
        request = create_mock_request()

        with patch(
            "aragora.knowledge.mound.websocket_bridge.get_km_bridge",
            return_value=None,
        ):
            result = await handler.handle_dashboard_batcher_stats(request)

        data, success = parse_result(result)
        assert success
        assert data["running"] is False
        assert "not initialized" in data.get("message", "")

    @pytest.mark.asyncio
    async def test_batcher_stats_module_unavailable(self):
        """Should return 503 when WebSocket bridge module unavailable."""
        handler = create_handler()
        request = create_mock_request()

        with patch.dict("sys.modules", {"aragora.knowledge.mound.websocket_bridge": None}):
            result = await handler.handle_dashboard_batcher_stats(request)

        assert result.status_code == 503

    @pytest.mark.asyncio
    async def test_batcher_stats_with_pending(self):
        """Should include pending events count."""
        mock_bridge = MockKMBridge()
        handler = create_handler()
        request = create_mock_request()

        with patch(
            "aragora.knowledge.mound.websocket_bridge.get_km_bridge",
            return_value=mock_bridge,
        ):
            result = await handler.handle_dashboard_batcher_stats(request)

        data, success = parse_result(result)
        assert success
        assert data["pending_events"] == 10
