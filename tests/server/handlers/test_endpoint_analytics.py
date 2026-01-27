"""
Tests for Endpoint Performance Analytics handler.

Tests cover:
- Endpoint metrics recording
- All endpoints listing
- Slowest endpoints ranking
- Error endpoints ranking
- Specific endpoint performance
- Health summary
"""

from __future__ import annotations

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.server.handlers.endpoint_analytics import (
    EndpointAnalyticsHandler,
    EndpointMetrics,
    EndpointMetricsStore,
    record_endpoint_request,
    get_metrics_store,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def metrics_store():
    """Create a fresh metrics store for testing."""
    return EndpointMetricsStore()


@pytest.fixture
def populated_store(metrics_store):
    """Create a metrics store with sample data."""
    # Fast endpoint - low latency, no errors
    for _ in range(100):
        metrics_store.record_request("/api/health", "GET", 200, 0.01)

    # Medium endpoint - moderate latency, some errors
    for i in range(50):
        status = 200 if i < 45 else 500  # 10% error rate
        metrics_store.record_request("/api/debates", "GET", status, 0.1)

    # Slow endpoint - high latency, many errors
    for i in range(30):
        status = 200 if i < 20 else 500  # 33% error rate
        metrics_store.record_request("/api/analyze", "POST", status, 0.5)

    # Very slow endpoint - very high latency
    for _ in range(10):
        metrics_store.record_request("/api/batch", "POST", 200, 2.0)

    return metrics_store


@pytest.fixture
def handler():
    """Create an endpoint analytics handler."""
    return EndpointAnalyticsHandler({})


@pytest.fixture
def mock_handler():
    """Create a mock HTTP handler."""
    handler = MagicMock()
    handler.headers = {}
    handler.client_address = ("127.0.0.1", 12345)
    return handler


@pytest.fixture
def mock_auth_context():
    """Create a mock authentication context with analytics permission."""
    ctx = MagicMock()
    ctx.user_id = "test-user"
    ctx.permissions = {"analytics:read"}
    ctx.has_permission = lambda p: p in ctx.permissions
    return ctx


# ============================================================================
# EndpointMetrics Tests
# ============================================================================


class TestEndpointMetrics:
    """Tests for EndpointMetrics dataclass."""

    def test_finalize_computes_percentiles(self):
        """Test that finalize computes correct percentiles."""
        metrics = EndpointMetrics(endpoint="/api/test", method="GET")
        # Add 100 latencies from 0.001 to 0.1 seconds
        metrics.latencies = [i * 0.001 for i in range(1, 101)]
        metrics.total_requests = 100
        metrics.success_count = 95
        metrics.error_count = 5

        metrics.finalize(window_seconds=100.0)

        assert metrics.error_rate == 5.0
        assert metrics.requests_per_second == 1.0
        assert metrics.min_latency_ms == pytest.approx(1.0, rel=0.1)
        assert metrics.max_latency_ms == pytest.approx(100.0, rel=0.1)
        # p50 should be around 50ms
        assert 40 < metrics.p50_latency_ms < 60

    def test_finalize_empty_latencies(self):
        """Test finalize with no latencies."""
        metrics = EndpointMetrics(endpoint="/api/test", method="GET")
        metrics.total_requests = 10
        metrics.success_count = 10

        metrics.finalize()

        assert metrics.avg_latency_ms == 0.0
        assert metrics.p50_latency_ms == 0.0
        assert metrics.error_rate == 0.0

    def test_to_dict(self):
        """Test serialization to dictionary."""
        metrics = EndpointMetrics(endpoint="/api/test", method="GET")
        metrics.total_requests = 100
        metrics.success_count = 95
        metrics.error_count = 5
        metrics.latencies = [0.05] * 100
        metrics.finalize()

        data = metrics.to_dict()

        assert data["endpoint"] == "/api/test"
        assert data["method"] == "GET"
        assert data["total_requests"] == 100
        assert data["error_rate_percent"] == 5.0
        assert "latency" in data
        assert data["latency"]["avg_ms"] == pytest.approx(50.0, rel=0.1)


# ============================================================================
# EndpointMetricsStore Tests
# ============================================================================


class TestEndpointMetricsStore:
    """Tests for EndpointMetricsStore."""

    def test_record_request_creates_entry(self, metrics_store):
        """Test that recording a request creates a metrics entry."""
        metrics_store.record_request("/api/test", "GET", 200, 0.05)

        assert "GET:/api/test" in metrics_store._metrics
        metrics = metrics_store._metrics["GET:/api/test"]
        assert metrics.total_requests == 1
        assert metrics.success_count == 1
        assert metrics.error_count == 0
        assert len(metrics.latencies) == 1

    def test_record_request_counts_errors(self, metrics_store):
        """Test that error responses are counted correctly."""
        metrics_store.record_request("/api/test", "GET", 200, 0.01)
        metrics_store.record_request("/api/test", "GET", 500, 0.01)
        metrics_store.record_request("/api/test", "GET", 404, 0.01)

        metrics = metrics_store._metrics["GET:/api/test"]
        assert metrics.total_requests == 3
        assert metrics.success_count == 1  # Only 200
        assert metrics.error_count == 2  # 500 and 404

    def test_record_request_different_methods(self, metrics_store):
        """Test that different HTTP methods are tracked separately."""
        metrics_store.record_request("/api/resource", "GET", 200, 0.01)
        metrics_store.record_request("/api/resource", "POST", 200, 0.02)
        metrics_store.record_request("/api/resource", "PUT", 200, 0.03)

        assert "GET:/api/resource" in metrics_store._metrics
        assert "POST:/api/resource" in metrics_store._metrics
        assert "PUT:/api/resource" in metrics_store._metrics

    def test_get_all_endpoints(self, populated_store):
        """Test getting all endpoint metrics."""
        endpoints = populated_store.get_all_endpoints()

        assert len(endpoints) == 4  # health, debates, analyze, batch
        endpoint_names = {e.endpoint for e in endpoints}
        assert "/api/health" in endpoint_names
        assert "/api/debates" in endpoint_names

    def test_get_endpoint(self, populated_store):
        """Test getting specific endpoint metrics."""
        metrics = populated_store.get_endpoint("/api/health", "GET")

        assert metrics is not None
        assert metrics.total_requests == 100
        assert metrics.error_count == 0

    def test_get_nonexistent_endpoint(self, populated_store):
        """Test getting metrics for nonexistent endpoint."""
        metrics = populated_store.get_endpoint("/api/nonexistent", "GET")
        assert metrics is None

    def test_reset(self, populated_store):
        """Test resetting the metrics store."""
        assert len(populated_store._metrics) > 0

        populated_store.reset()

        assert len(populated_store._metrics) == 0


# ============================================================================
# EndpointAnalyticsHandler Tests
# ============================================================================


class TestEndpointAnalyticsHandler:
    """Tests for EndpointAnalyticsHandler."""

    def test_can_handle_routes(self, handler):
        """Test that handler recognizes its routes."""
        assert handler.can_handle("/api/analytics/endpoints")
        assert handler.can_handle("/api/analytics/endpoints/slowest")
        assert handler.can_handle("/api/analytics/endpoints/errors")
        assert handler.can_handle("/api/analytics/endpoints/health")
        assert handler.can_handle("/api/analytics/endpoints/api%2Ftest/performance")

    def test_cannot_handle_other_routes(self, handler):
        """Test that handler doesn't match other routes."""
        assert not handler.can_handle("/api/debates")
        assert not handler.can_handle("/api/analytics/debates/overview")
        assert not handler.can_handle("/api/other")

    @pytest.mark.asyncio
    async def test_get_all_endpoints(
        self, handler, mock_handler, mock_auth_context, populated_store
    ):
        """Test GET /api/analytics/endpoints."""
        with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
            mock_auth.return_value = mock_auth_context
            with patch.object(handler, "check_permission"):
                with patch(
                    "aragora.server.handlers.endpoint_analytics._metrics_store",
                    populated_store,
                ):
                    result = await handler.handle(
                        "/api/analytics/endpoints",
                        {"sort": "requests", "limit": "10"},
                        mock_handler,
                    )

                    assert result is not None
                    assert result[1] == 200
                    body = result[0]
                    assert "endpoints" in body
                    assert len(body["endpoints"]) == 4
                    # Should be sorted by requests (desc by default)
                    assert (
                        body["endpoints"][0]["total_requests"]
                        >= body["endpoints"][1]["total_requests"]
                    )

    @pytest.mark.asyncio
    async def test_get_slowest_endpoints(
        self, handler, mock_handler, mock_auth_context, populated_store
    ):
        """Test GET /api/analytics/endpoints/slowest."""
        with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
            mock_auth.return_value = mock_auth_context
            with patch.object(handler, "check_permission"):
                with patch(
                    "aragora.server.handlers.endpoint_analytics._metrics_store",
                    populated_store,
                ):
                    result = await handler.handle(
                        "/api/analytics/endpoints/slowest",
                        {"limit": "5", "percentile": "p95"},
                        mock_handler,
                    )

                    assert result is not None
                    assert result[1] == 200
                    body = result[0]
                    assert "slowest_endpoints" in body
                    # /api/batch should be slowest (2s latency)
                    assert body["slowest_endpoints"][0]["endpoint"] == "/api/batch"

    @pytest.mark.asyncio
    async def test_get_error_endpoints(
        self, handler, mock_handler, mock_auth_context, populated_store
    ):
        """Test GET /api/analytics/endpoints/errors."""
        with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
            mock_auth.return_value = mock_auth_context
            with patch.object(handler, "check_permission"):
                with patch(
                    "aragora.server.handlers.endpoint_analytics._metrics_store",
                    populated_store,
                ):
                    result = await handler.handle(
                        "/api/analytics/endpoints/errors",
                        {"limit": "5", "min_requests": "5"},
                        mock_handler,
                    )

                    assert result is not None
                    assert result[1] == 200
                    body = result[0]
                    assert "error_endpoints" in body
                    # /api/analyze should have highest error rate (~33%)
                    if body["error_endpoints"]:
                        assert body["error_endpoints"][0]["endpoint"] == "/api/analyze"

    @pytest.mark.asyncio
    async def test_get_health_summary(
        self, handler, mock_handler, mock_auth_context, populated_store
    ):
        """Test GET /api/analytics/endpoints/health."""
        with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
            mock_auth.return_value = mock_auth_context
            with patch.object(handler, "check_permission"):
                with patch(
                    "aragora.server.handlers.endpoint_analytics._metrics_store",
                    populated_store,
                ):
                    result = await handler.handle(
                        "/api/analytics/endpoints/health",
                        {},
                        mock_handler,
                    )

                    assert result is not None
                    assert result[1] == 200
                    body = result[0]
                    assert "status" in body
                    assert body["status"] in ["healthy", "warning", "degraded"]
                    assert "summary" in body
                    assert body["summary"]["total_endpoints"] == 4
                    assert "endpoint_health" in body

    @pytest.mark.asyncio
    async def test_get_specific_endpoint_performance(
        self, handler, mock_handler, mock_auth_context, populated_store
    ):
        """Test GET /api/analytics/endpoints/{endpoint}/performance."""
        with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
            mock_auth.return_value = mock_auth_context
            with patch.object(handler, "check_permission"):
                with patch(
                    "aragora.server.handlers.endpoint_analytics._metrics_store",
                    populated_store,
                ):
                    result = await handler.handle(
                        "/api/analytics/endpoints/%2Fapi%2Fhealth/performance",
                        {"method": "GET"},
                        mock_handler,
                    )

                    assert result is not None
                    assert result[1] == 200
                    body = result[0]
                    assert "endpoint" in body
                    assert body["endpoint"]["endpoint"] == "/api/health"
                    assert body["endpoint"]["total_requests"] == 100

    @pytest.mark.asyncio
    async def test_unauthorized_access(self, handler, mock_handler):
        """Test that unauthorized requests are rejected."""
        from aragora.server.handlers.secure import UnauthorizedError

        with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
            mock_auth.side_effect = UnauthorizedError("Not authenticated")

            result = await handler.handle("/api/analytics/endpoints", {}, mock_handler)

            assert result is not None
            assert result[1] == 401

    @pytest.mark.asyncio
    async def test_forbidden_access(self, handler, mock_handler, mock_auth_context):
        """Test that requests without permission are rejected."""
        from aragora.server.handlers.secure import ForbiddenError

        mock_auth_context.permissions = set()  # No permissions

        with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
            mock_auth.return_value = mock_auth_context
            with patch.object(handler, "check_permission") as mock_check:
                mock_check.side_effect = ForbiddenError("Permission denied")

                result = await handler.handle("/api/analytics/endpoints", {}, mock_handler)

                assert result is not None
                assert result[1] == 403


# ============================================================================
# Integration Tests
# ============================================================================


class TestEndpointAnalyticsIntegration:
    """Integration tests for endpoint analytics."""

    def test_record_and_retrieve_flow(self):
        """Test the full flow of recording and retrieving metrics."""
        store = EndpointMetricsStore()

        # Record various requests
        for i in range(100):
            latency = 0.01 + (i * 0.001)  # 10ms to 110ms
            status = 200 if i < 95 else 500  # 5% error rate
            store.record_request("/api/flow-test", "GET", status, latency)

        # Retrieve metrics
        metrics = store.get_endpoint("/api/flow-test", "GET")

        assert metrics is not None
        assert metrics.total_requests == 100
        assert metrics.success_count == 95
        assert metrics.error_count == 5
        assert metrics.error_rate == 5.0
        assert metrics.avg_latency_ms > 0
        assert metrics.p95_latency_ms > metrics.p50_latency_ms

    def test_global_store_singleton(self):
        """Test that get_metrics_store returns the global instance."""
        store1 = get_metrics_store()
        store2 = get_metrics_store()

        assert store1 is store2

    def test_record_endpoint_request_function(self):
        """Test the convenience function for recording requests."""
        # Reset global store
        global_store = get_metrics_store()
        global_store.reset()

        record_endpoint_request("/api/convenience", "POST", 201, 0.05)
        record_endpoint_request("/api/convenience", "POST", 500, 0.1)

        metrics = global_store.get_endpoint("/api/convenience", "POST")
        assert metrics is not None
        assert metrics.total_requests == 2
        assert metrics.error_count == 1


__all__ = [
    "TestEndpointMetrics",
    "TestEndpointMetricsStore",
    "TestEndpointAnalyticsHandler",
    "TestEndpointAnalyticsIntegration",
]
