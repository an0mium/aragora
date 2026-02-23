"""Tests for EndpointAnalyticsHandler.

Covers:
- Handler initialization and route matching (can_handle)
- GET /api/analytics/endpoints (list all endpoints with metrics)
- GET /api/analytics/endpoints/slowest (top N slowest endpoints)
- GET /api/analytics/endpoints/errors (top N error endpoints)
- GET /api/analytics/endpoints/health (API health summary)
- GET /api/analytics/endpoints/{endpoint}/performance (specific endpoint)
- Rate limiting
- RBAC authentication and authorization (no_auto_auth)
- Sorting, filtering, pagination
- Edge cases (empty store, no latencies, single entry)
- EndpointMetrics dataclass and EndpointMetricsStore unit tests
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.endpoint_analytics import (
    EndpointAnalyticsHandler,
    EndpointMetrics,
    EndpointMetricsStore,
    _endpoint_analytics_limiter,
    _metrics_store,
    get_metrics_store,
    record_endpoint_request,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result) -> dict:
    """Extract decoded JSON body from HandlerResult."""
    return result[0]


def _status(result) -> int:
    """Extract HTTP status code from HandlerResult."""
    return result[1]


def _run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler():
    """Create an EndpointAnalyticsHandler with empty context."""
    return EndpointAnalyticsHandler(ctx={})


@pytest.fixture
def mock_http_handler():
    """Create a mock HTTP handler with client address."""
    h = MagicMock()
    h.client_address = ("127.0.0.1", 54321)
    h.headers = {"Content-Length": "0", "Host": "localhost:8080"}
    return h


@pytest.fixture(autouse=True)
def reset_metrics_store():
    """Reset the global metrics store and rate limiter between tests."""
    _metrics_store.reset()
    _endpoint_analytics_limiter._requests.clear()
    yield
    _metrics_store.reset()


def _seed_store(entries: list[tuple[str, str, int, float]] | None = None):
    """Seed the global metrics store with sample data.

    Each entry is (endpoint, method, status_code, latency_seconds).
    """
    if entries is None:
        entries = [
            ("/api/debates", "GET", 200, 0.05),
            ("/api/debates", "GET", 200, 0.08),
            ("/api/debates", "GET", 500, 0.12),
            ("/api/agents", "GET", 200, 0.02),
            ("/api/agents", "GET", 200, 0.03),
            ("/api/users", "POST", 200, 0.15),
            ("/api/users", "POST", 201, 0.10),
            ("/api/users", "POST", 400, 0.01),
            ("/api/users", "POST", 500, 0.20),
        ]
    for endpoint, method, status, latency in entries:
        _metrics_store.record_request(endpoint, method, status, latency)


# ===========================================================================
# EndpointMetrics dataclass tests
# ===========================================================================


class TestEndpointMetrics:
    """Unit tests for the EndpointMetrics dataclass."""

    def test_finalize_with_latencies(self):
        m = EndpointMetrics(endpoint="/api/test", method="GET")
        m.total_requests = 10
        m.success_count = 8
        m.error_count = 2
        m.latencies = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10]
        m.finalize(window_seconds=100.0)

        assert m.error_rate == 20.0
        assert m.requests_per_second == 0.1
        assert m.avg_latency_ms > 0
        assert m.min_latency_ms == 10.0
        assert m.max_latency_ms == 100.0
        assert m.p50_latency_ms > 0
        assert m.p95_latency_ms > 0
        assert m.p99_latency_ms > 0

    def test_finalize_no_latencies(self):
        m = EndpointMetrics(endpoint="/api/empty", method="GET")
        m.total_requests = 5
        m.error_count = 1
        m.success_count = 4
        m.finalize()

        assert m.error_rate == 20.0
        assert m.avg_latency_ms == 0.0
        assert m.min_latency_ms == 0.0
        assert m.max_latency_ms == 0.0

    def test_finalize_zero_requests(self):
        m = EndpointMetrics(endpoint="/api/zero")
        m.finalize()

        assert m.error_rate == 0.0
        assert m.requests_per_second == 0.0
        assert m.avg_latency_ms == 0.0

    def test_finalize_single_latency(self):
        m = EndpointMetrics(endpoint="/api/single")
        m.total_requests = 1
        m.success_count = 1
        m.latencies = [0.05]
        m.finalize()

        assert m.avg_latency_ms == 50.0
        assert m.min_latency_ms == 50.0
        assert m.max_latency_ms == 50.0
        assert m.p50_latency_ms == 50.0
        assert m.p95_latency_ms == 50.0
        assert m.p99_latency_ms == 50.0

    def test_to_dict_structure(self):
        m = EndpointMetrics(endpoint="/api/test", method="POST")
        m.total_requests = 100
        m.success_count = 90
        m.error_count = 10
        m.latencies = [0.1] * 100
        m.finalize()

        d = m.to_dict()
        assert d["endpoint"] == "/api/test"
        assert d["method"] == "POST"
        assert d["total_requests"] == 100
        assert d["success_count"] == 90
        assert d["error_count"] == 10
        assert d["error_rate_percent"] == 10.0
        assert "latency" in d
        assert "avg_ms" in d["latency"]
        assert "p50_ms" in d["latency"]
        assert "p95_ms" in d["latency"]
        assert "p99_ms" in d["latency"]
        assert "min_ms" in d["latency"]
        assert "max_ms" in d["latency"]


# ===========================================================================
# EndpointMetricsStore tests
# ===========================================================================


class TestEndpointMetricsStore:
    """Unit tests for the EndpointMetricsStore."""

    def test_record_request_creates_entry(self):
        store = EndpointMetricsStore()
        store.record_request("/api/foo", "GET", 200, 0.05)

        metrics = store.get_endpoint("/api/foo", "GET")
        assert metrics is not None
        assert metrics.total_requests == 1
        assert metrics.success_count == 1
        assert metrics.error_count == 0

    def test_record_request_increments(self):
        store = EndpointMetricsStore()
        store.record_request("/api/foo", "GET", 200, 0.05)
        store.record_request("/api/foo", "GET", 200, 0.06)
        store.record_request("/api/foo", "GET", 500, 0.10)

        metrics = store.get_endpoint("/api/foo", "GET")
        assert metrics.total_requests == 3
        assert metrics.success_count == 2
        assert metrics.error_count == 1

    def test_record_status_codes(self):
        store = EndpointMetricsStore()
        # 2xx and 3xx are success
        store.record_request("/api/a", "GET", 200, 0.01)
        store.record_request("/api/a", "GET", 301, 0.01)
        # 4xx and 5xx are errors
        store.record_request("/api/a", "GET", 400, 0.01)
        store.record_request("/api/a", "GET", 500, 0.01)

        metrics = store.get_endpoint("/api/a", "GET")
        assert metrics.success_count == 2
        assert metrics.error_count == 2

    def test_different_methods_separate(self):
        store = EndpointMetricsStore()
        store.record_request("/api/foo", "GET", 200, 0.05)
        store.record_request("/api/foo", "POST", 201, 0.10)

        get_metrics = store.get_endpoint("/api/foo", "GET")
        post_metrics = store.get_endpoint("/api/foo", "POST")

        assert get_metrics.total_requests == 1
        assert post_metrics.total_requests == 1

    def test_max_entries_bounded(self):
        store = EndpointMetricsStore(max_entries_per_endpoint=5)
        for i in range(10):
            store.record_request("/api/bounded", "GET", 200, 0.01 * i)

        metrics = store.get_endpoint("/api/bounded", "GET")
        assert len(metrics.latencies) == 5  # bounded at max
        assert metrics.total_requests == 10  # count is not bounded

    def test_get_all_endpoints(self):
        store = EndpointMetricsStore()
        store.record_request("/api/a", "GET", 200, 0.01)
        store.record_request("/api/b", "POST", 200, 0.02)

        all_eps = store.get_all_endpoints()
        assert len(all_eps) == 2

    def test_get_endpoint_not_found(self):
        store = EndpointMetricsStore()
        assert store.get_endpoint("/api/missing", "GET") is None

    def test_reset_clears_all(self):
        store = EndpointMetricsStore()
        store.record_request("/api/a", "GET", 200, 0.01)
        store.reset()

        assert store.get_all_endpoints() == []
        assert store.get_endpoint("/api/a", "GET") is None


# ===========================================================================
# Module-level helpers tests
# ===========================================================================


class TestModuleHelpers:
    """Tests for module-level utility functions."""

    def test_get_metrics_store_returns_global(self):
        store = get_metrics_store()
        assert store is _metrics_store

    def test_record_endpoint_request_uses_global(self):
        record_endpoint_request("/api/test", "GET", 200, 0.05)
        metrics = _metrics_store.get_endpoint("/api/test", "GET")
        assert metrics is not None
        assert metrics.total_requests == 1


# ===========================================================================
# can_handle tests
# ===========================================================================


class TestCanHandle:
    """Tests for route matching via can_handle."""

    def test_handles_endpoints_list(self, handler):
        assert handler.can_handle("/api/analytics/endpoints") is True

    def test_handles_versioned_endpoints_list(self, handler):
        assert handler.can_handle("/api/v1/analytics/endpoints") is True

    def test_handles_slowest(self, handler):
        assert handler.can_handle("/api/analytics/endpoints/slowest") is True

    def test_handles_versioned_slowest(self, handler):
        assert handler.can_handle("/api/v1/analytics/endpoints/slowest") is True

    def test_handles_errors(self, handler):
        assert handler.can_handle("/api/analytics/endpoints/errors") is True

    def test_handles_health(self, handler):
        assert handler.can_handle("/api/analytics/endpoints/health") is True

    def test_handles_endpoint_performance(self, handler):
        assert handler.can_handle("/api/analytics/endpoints/my-endpoint/performance") is True

    def test_handles_versioned_endpoint_performance(self, handler):
        assert handler.can_handle("/api/v1/analytics/endpoints/my-endpoint/performance") is True

    def test_rejects_unrelated_path(self, handler):
        assert handler.can_handle("/api/debates") is False
        assert handler.can_handle("/api/agents") is False
        assert handler.can_handle("/api/analytics/other") is False

    def test_rejects_partial_match(self, handler):
        assert handler.can_handle("/api/analytics/endpoints/slowest/extra") is False


# ===========================================================================
# Rate limiting tests
# ===========================================================================


class TestRateLimiting:
    """Tests for rate limiting in the handler."""

    def test_rate_limit_exceeded_returns_429(self, handler, mock_http_handler):
        with patch(
            "aragora.server.handlers.endpoint_analytics._endpoint_analytics_limiter"
        ) as mock_limiter:
            mock_limiter.is_allowed.return_value = False
            result = _run(
                handler.handle("/api/analytics/endpoints", {}, mock_http_handler)
            )

        assert _status(result) == 429
        body = _body(result)
        assert "rate limit" in body.get("error", {}).get("message", "").lower() or \
               "rate limit" in body.get("error", "").lower()

    def test_rate_limit_allowed_passes(self, handler, mock_http_handler):
        _seed_store()
        result = _run(
            handler.handle("/api/analytics/endpoints", {}, mock_http_handler)
        )
        assert _status(result) == 200


# ===========================================================================
# RBAC / Auth tests (opt out of auto-auth)
# ===========================================================================


class TestRBAC:
    """Tests for authentication and authorization enforcement."""

    @pytest.mark.no_auto_auth
    def test_unauthenticated_returns_401(self, mock_http_handler):
        from aragora.server.handlers.secure import SecureHandler
        from aragora.server.handlers.utils.auth import UnauthorizedError

        handler = EndpointAnalyticsHandler(ctx={})

        async def mock_raise_unauth(self, request, require_auth=False):
            raise UnauthorizedError("No token")

        with patch.object(SecureHandler, "get_auth_context", mock_raise_unauth):
            result = _run(
                handler.handle("/api/analytics/endpoints", {}, mock_http_handler)
            )

        assert _status(result) == 401

    @pytest.mark.no_auto_auth
    def test_forbidden_returns_403(self, mock_http_handler):
        from aragora.rbac.models import AuthorizationContext
        from aragora.server.handlers.secure import SecureHandler
        from aragora.server.handlers.utils.auth import ForbiddenError

        handler = EndpointAnalyticsHandler(ctx={})

        mock_ctx = AuthorizationContext(
            user_id="user-1",
            user_email="user@example.com",
            org_id="org-1",
            roles={"viewer"},
            permissions=set(),
        )

        async def mock_get_auth(self, request, require_auth=False):
            return mock_ctx

        def mock_check_perm(self, ctx, perm):
            raise ForbiddenError(f"Missing {perm}")

        with patch.object(SecureHandler, "get_auth_context", mock_get_auth), \
             patch.object(SecureHandler, "check_permission", mock_check_perm):
            result = _run(
                handler.handle("/api/analytics/endpoints", {}, mock_http_handler)
            )

        assert _status(result) == 403


# ===========================================================================
# GET /api/analytics/endpoints
# ===========================================================================


class TestGetAllEndpoints:
    """Tests for the all-endpoints listing route."""

    def test_empty_store_returns_empty_list(self, handler, mock_http_handler):
        result = _run(
            handler.handle("/api/analytics/endpoints", {}, mock_http_handler)
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["endpoints"] == []
        assert body["total_endpoints"] == 0

    def test_returns_seeded_endpoints(self, handler, mock_http_handler):
        _seed_store()
        result = _run(
            handler.handle("/api/analytics/endpoints", {}, mock_http_handler)
        )
        assert _status(result) == 200
        body = _body(result)
        assert len(body["endpoints"]) == 3  # /api/debates GET, /api/agents GET, /api/users POST
        assert body["total_endpoints"] == 3
        assert "window_seconds" in body
        assert "generated_at" in body

    def test_sort_by_latency(self, handler, mock_http_handler):
        _seed_store()
        result = _run(
            handler.handle(
                "/api/analytics/endpoints",
                {"sort": "latency", "order": "desc"},
                mock_http_handler,
            )
        )
        assert _status(result) == 200
        body = _body(result)
        endpoints = body["endpoints"]
        # Verify descending order by p95 latency
        latencies = [ep["latency"]["p95_ms"] for ep in endpoints]
        assert latencies == sorted(latencies, reverse=True)

    def test_sort_by_errors(self, handler, mock_http_handler):
        _seed_store()
        result = _run(
            handler.handle(
                "/api/analytics/endpoints",
                {"sort": "errors", "order": "desc"},
                mock_http_handler,
            )
        )
        assert _status(result) == 200
        body = _body(result)
        endpoints = body["endpoints"]
        error_rates = [ep["error_rate_percent"] for ep in endpoints]
        assert error_rates == sorted(error_rates, reverse=True)

    def test_sort_by_requests_asc(self, handler, mock_http_handler):
        _seed_store()
        result = _run(
            handler.handle(
                "/api/analytics/endpoints",
                {"sort": "requests", "order": "asc"},
                mock_http_handler,
            )
        )
        assert _status(result) == 200
        body = _body(result)
        endpoints = body["endpoints"]
        totals = [ep["total_requests"] for ep in endpoints]
        assert totals == sorted(totals)

    def test_limit_parameter(self, handler, mock_http_handler):
        _seed_store()
        result = _run(
            handler.handle(
                "/api/analytics/endpoints",
                {"limit": "1"},
                mock_http_handler,
            )
        )
        assert _status(result) == 200
        body = _body(result)
        assert len(body["endpoints"]) == 1

    def test_custom_window(self, handler, mock_http_handler):
        _seed_store()
        result = _run(
            handler.handle(
                "/api/analytics/endpoints",
                {"window": "600"},
                mock_http_handler,
            )
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["window_seconds"] == 600.0

    def test_versioned_path(self, handler, mock_http_handler):
        _seed_store()
        result = _run(
            handler.handle("/api/v1/analytics/endpoints", {}, mock_http_handler)
        )
        assert _status(result) == 200
        assert len(_body(result)["endpoints"]) == 3


# ===========================================================================
# GET /api/analytics/endpoints/slowest
# ===========================================================================


class TestGetSlowestEndpoints:
    """Tests for the slowest endpoints route."""

    def test_empty_store(self, handler, mock_http_handler):
        result = _run(
            handler.handle("/api/analytics/endpoints/slowest", {}, mock_http_handler)
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["slowest_endpoints"] == []
        assert body["percentile"] == "p95"

    def test_returns_sorted_by_p95_default(self, handler, mock_http_handler):
        _seed_store()
        result = _run(
            handler.handle("/api/analytics/endpoints/slowest", {}, mock_http_handler)
        )
        assert _status(result) == 200
        body = _body(result)
        latencies = [ep["latency"]["p95_ms"] for ep in body["slowest_endpoints"]]
        assert latencies == sorted(latencies, reverse=True)

    def test_percentile_p50(self, handler, mock_http_handler):
        _seed_store()
        result = _run(
            handler.handle(
                "/api/analytics/endpoints/slowest",
                {"percentile": "p50"},
                mock_http_handler,
            )
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["percentile"] == "p50"
        latencies = [ep["latency"]["p50_ms"] for ep in body["slowest_endpoints"]]
        assert latencies == sorted(latencies, reverse=True)

    def test_percentile_p99(self, handler, mock_http_handler):
        _seed_store()
        result = _run(
            handler.handle(
                "/api/analytics/endpoints/slowest",
                {"percentile": "p99"},
                mock_http_handler,
            )
        )
        assert _status(result) == 200
        assert _body(result)["percentile"] == "p99"

    def test_invalid_percentile_returns_400(self, handler, mock_http_handler):
        result = _run(
            handler.handle(
                "/api/analytics/endpoints/slowest",
                {"percentile": "p75"},
                mock_http_handler,
            )
        )
        assert _status(result) == 400
        body = _body(result)
        assert "INVALID_PERCENTILE" in str(body)

    def test_limit_parameter(self, handler, mock_http_handler):
        _seed_store()
        result = _run(
            handler.handle(
                "/api/analytics/endpoints/slowest",
                {"limit": "1"},
                mock_http_handler,
            )
        )
        assert _status(result) == 200
        body = _body(result)
        assert len(body["slowest_endpoints"]) == 1
        assert body["limit"] == 1

    def test_versioned_path(self, handler, mock_http_handler):
        _seed_store()
        result = _run(
            handler.handle("/api/v1/analytics/endpoints/slowest", {}, mock_http_handler)
        )
        assert _status(result) == 200


# ===========================================================================
# GET /api/analytics/endpoints/errors
# ===========================================================================


class TestGetErrorEndpoints:
    """Tests for the error endpoints route."""

    def test_empty_store(self, handler, mock_http_handler):
        result = _run(
            handler.handle("/api/analytics/endpoints/errors", {}, mock_http_handler)
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["error_endpoints"] == []

    def test_min_requests_filters(self, handler, mock_http_handler):
        # Seed with low-volume endpoints that won't pass the min_requests=10 threshold
        _seed_store()
        result = _run(
            handler.handle(
                "/api/analytics/endpoints/errors",
                {"min_requests": "10"},
                mock_http_handler,
            )
        )
        assert _status(result) == 200
        body = _body(result)
        # All seeded endpoints have < 10 requests, so all should be filtered out
        assert body["error_endpoints"] == []
        assert body["min_requests_threshold"] == 10

    def test_returns_endpoints_above_threshold(self, handler, mock_http_handler):
        # Seed enough requests to pass the threshold
        entries = [("/api/err", "GET", 500, 0.1)] * 5 + [("/api/err", "GET", 200, 0.1)] * 5
        _seed_store(entries)
        result = _run(
            handler.handle(
                "/api/analytics/endpoints/errors",
                {"min_requests": "5"},
                mock_http_handler,
            )
        )
        assert _status(result) == 200
        body = _body(result)
        assert len(body["error_endpoints"]) == 1
        assert body["error_endpoints"][0]["error_rate_percent"] == 50.0

    def test_sorted_by_error_rate_desc(self, handler, mock_http_handler):
        entries = (
            [("/api/high-err", "GET", 500, 0.1)] * 8
            + [("/api/high-err", "GET", 200, 0.1)] * 2
            + [("/api/low-err", "GET", 500, 0.1)] * 2
            + [("/api/low-err", "GET", 200, 0.1)] * 8
        )
        _seed_store(entries)
        result = _run(
            handler.handle(
                "/api/analytics/endpoints/errors",
                {"min_requests": "1"},
                mock_http_handler,
            )
        )
        assert _status(result) == 200
        body = _body(result)
        error_rates = [ep["error_rate_percent"] for ep in body["error_endpoints"]]
        assert error_rates == sorted(error_rates, reverse=True)

    def test_limit_parameter(self, handler, mock_http_handler):
        entries = (
            [("/api/a", "GET", 500, 0.1)] * 3
            + [("/api/b", "GET", 500, 0.1)] * 3
            + [("/api/c", "GET", 500, 0.1)] * 3
        )
        _seed_store(entries)
        result = _run(
            handler.handle(
                "/api/analytics/endpoints/errors",
                {"limit": "1", "min_requests": "1"},
                mock_http_handler,
            )
        )
        assert _status(result) == 200
        assert len(_body(result)["error_endpoints"]) == 1

    def test_versioned_path(self, handler, mock_http_handler):
        result = _run(
            handler.handle("/api/v1/analytics/endpoints/errors", {}, mock_http_handler)
        )
        assert _status(result) == 200


# ===========================================================================
# GET /api/analytics/endpoints/{endpoint}/performance
# ===========================================================================


class TestGetEndpointPerformance:
    """Tests for specific endpoint performance route."""

    def test_found_endpoint(self, handler, mock_http_handler):
        _metrics_store.record_request("/api/debates", "GET", 200, 0.05)
        _metrics_store.record_request("/api/debates", "GET", 200, 0.08)

        result = _run(
            handler.handle(
                "/api/analytics/endpoints/%2Fapi%2Fdebates/performance",
                {},
                mock_http_handler,
            )
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["endpoint"]["endpoint"] == "/api/debates"
        assert body["endpoint"]["total_requests"] == 2
        assert "generated_at" in body

    def test_endpoint_not_found_returns_404(self, handler, mock_http_handler):
        result = _run(
            handler.handle(
                "/api/analytics/endpoints/nonexistent/performance",
                {},
                mock_http_handler,
            )
        )
        assert _status(result) == 404
        body = _body(result)
        assert "ENDPOINT_NOT_FOUND" in str(body)

    def test_endpoint_with_leading_slash_fallback(self, handler, mock_http_handler):
        """Test that the handler tries adding a leading slash."""
        _metrics_store.record_request("/api/test", "GET", 200, 0.05)

        # Request without leading slash - should find it via fallback
        result = _run(
            handler.handle(
                "/api/analytics/endpoints/api%2Ftest/performance",
                {},
                mock_http_handler,
            )
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["endpoint"]["endpoint"] == "/api/test"

    def test_method_query_param(self, handler, mock_http_handler):
        _metrics_store.record_request("/api/users", "POST", 201, 0.10)

        result = _run(
            handler.handle(
                "/api/analytics/endpoints/%2Fapi%2Fusers/performance",
                {"method": "post"},
                mock_http_handler,
            )
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["endpoint"]["method"] == "POST"

    def test_method_not_found(self, handler, mock_http_handler):
        _metrics_store.record_request("/api/users", "GET", 200, 0.05)

        result = _run(
            handler.handle(
                "/api/analytics/endpoints/%2Fapi%2Fusers/performance",
                {"method": "DELETE"},
                mock_http_handler,
            )
        )
        assert _status(result) == 404

    def test_versioned_path(self, handler, mock_http_handler):
        _metrics_store.record_request("/api/debates", "GET", 200, 0.05)

        result = _run(
            handler.handle(
                "/api/v1/analytics/endpoints/%2Fapi%2Fdebates/performance",
                {},
                mock_http_handler,
            )
        )
        assert _status(result) == 200


# ===========================================================================
# GET /api/analytics/endpoints/health
# ===========================================================================


class TestGetHealthSummary:
    """Tests for the health summary route."""

    def test_empty_store_returns_unknown(self, handler, mock_http_handler):
        result = _run(
            handler.handle("/api/analytics/endpoints/health", {}, mock_http_handler)
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "unknown"
        assert body["total_endpoints"] == 0
        assert "generated_at" in body

    def test_healthy_status(self, handler, mock_http_handler):
        # All requests succeed with low latency
        entries = [("/api/a", "GET", 200, 0.01)] * 20
        _seed_store(entries)

        result = _run(
            handler.handle("/api/analytics/endpoints/health", {}, mock_http_handler)
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "healthy"
        assert body["summary"]["total_requests"] == 20
        assert body["summary"]["total_errors"] == 0
        assert body["summary"]["overall_error_rate_percent"] == 0.0

    def test_warning_status_due_to_error_rate(self, handler, mock_http_handler):
        # Error rate between 1% and 5%
        entries = [("/api/a", "GET", 200, 0.01)] * 97 + [("/api/a", "GET", 500, 0.01)] * 3
        _seed_store(entries)

        result = _run(
            handler.handle("/api/analytics/endpoints/health", {}, mock_http_handler)
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "warning"

    def test_degraded_status_due_to_error_rate(self, handler, mock_http_handler):
        # Error rate > 5%
        entries = [("/api/a", "GET", 200, 0.01)] * 90 + [("/api/a", "GET", 500, 0.01)] * 10
        _seed_store(entries)

        result = _run(
            handler.handle("/api/analytics/endpoints/health", {}, mock_http_handler)
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "degraded"

    def test_degraded_status_due_to_latency(self, handler, mock_http_handler):
        # High latency (>2000ms p95)
        entries = [("/api/slow", "GET", 200, 3.0)] * 20
        _seed_store(entries)

        result = _run(
            handler.handle("/api/analytics/endpoints/health", {}, mock_http_handler)
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "degraded"

    def test_warning_status_due_to_latency(self, handler, mock_http_handler):
        # Moderate latency (500ms < p95 < 2000ms)
        entries = [("/api/moderate", "GET", 200, 0.8)] * 20
        _seed_store(entries)

        result = _run(
            handler.handle("/api/analytics/endpoints/health", {}, mock_http_handler)
        )
        assert _status(result) == 200
        body = _body(result)
        assert body["status"] == "warning"

    def test_health_summary_structure(self, handler, mock_http_handler):
        _seed_store()
        result = _run(
            handler.handle("/api/analytics/endpoints/health", {}, mock_http_handler)
        )
        assert _status(result) == 200
        body = _body(result)

        assert "summary" in body
        summary = body["summary"]
        assert "total_endpoints" in summary
        assert "total_requests" in summary
        assert "total_errors" in summary
        assert "overall_error_rate_percent" in summary
        assert "avg_p95_latency_ms" in summary

        assert "endpoint_health" in body
        eh = body["endpoint_health"]
        assert "healthy" in eh
        assert "warning" in eh
        assert "degraded" in eh

        assert "thresholds" in body

    def test_endpoint_health_counts(self, handler, mock_http_handler):
        # One healthy endpoint, one with high error rate (degraded)
        entries = (
            [("/api/healthy", "GET", 200, 0.01)] * 20
            + [("/api/degraded", "GET", 500, 0.01)] * 10
            + [("/api/degraded", "GET", 200, 0.01)] * 10
        )
        _seed_store(entries)

        result = _run(
            handler.handle("/api/analytics/endpoints/health", {}, mock_http_handler)
        )
        assert _status(result) == 200
        body = _body(result)
        # /api/healthy has 0% error rate -> healthy
        # /api/degraded has 50% error rate -> degraded
        assert body["endpoint_health"]["healthy"] >= 1
        assert body["endpoint_health"]["degraded"] >= 1

    def test_versioned_path(self, handler, mock_http_handler):
        result = _run(
            handler.handle("/api/v1/analytics/endpoints/health", {}, mock_http_handler)
        )
        assert _status(result) == 200


# ===========================================================================
# Unknown path tests
# ===========================================================================


class TestUnknownPath:
    """Tests for paths that match can_handle but not any specific route."""

    def test_unknown_path_returns_404(self, handler, mock_http_handler):
        # Directly call handle with a path that passes rate-limit/auth
        # but doesn't match any route or pattern.
        # The ENDPOINT_PATTERN won't match without "/performance" suffix.
        result = _run(
            handler.handle(
                "/api/analytics/endpoints/something/unknown",
                {},
                mock_http_handler,
            )
        )
        assert _status(result) == 404


# ===========================================================================
# Error handling / internal errors
# ===========================================================================


class TestInternalErrors:
    """Tests for error handling within handler methods."""

    def test_all_endpoints_internal_error(self, handler, mock_http_handler):
        with patch(
            "aragora.server.handlers.endpoint_analytics._metrics_store"
        ) as mock_store:
            mock_store.get_all_endpoints.side_effect = TypeError("test error")
            result = _run(
                handler.handle("/api/analytics/endpoints", {}, mock_http_handler)
            )
        assert _status(result) == 500

    def test_slowest_endpoints_internal_error(self, handler, mock_http_handler):
        with patch(
            "aragora.server.handlers.endpoint_analytics._metrics_store"
        ) as mock_store:
            mock_store.get_all_endpoints.side_effect = ValueError("test error")
            result = _run(
                handler.handle(
                    "/api/analytics/endpoints/slowest",
                    {},
                    mock_http_handler,
                )
            )
        assert _status(result) == 500

    def test_error_endpoints_internal_error(self, handler, mock_http_handler):
        with patch(
            "aragora.server.handlers.endpoint_analytics._metrics_store"
        ) as mock_store:
            mock_store.get_all_endpoints.side_effect = AttributeError("test error")
            result = _run(
                handler.handle(
                    "/api/analytics/endpoints/errors",
                    {},
                    mock_http_handler,
                )
            )
        assert _status(result) == 500

    def test_endpoint_performance_internal_error(self, handler, mock_http_handler):
        with patch(
            "aragora.server.handlers.endpoint_analytics._metrics_store"
        ) as mock_store:
            mock_store.get_endpoint.side_effect = KeyError("test error")
            result = _run(
                handler.handle(
                    "/api/analytics/endpoints/myep/performance",
                    {},
                    mock_http_handler,
                )
            )
        assert _status(result) == 500

    def test_health_summary_internal_error(self, handler, mock_http_handler):
        with patch(
            "aragora.server.handlers.endpoint_analytics._metrics_store"
        ) as mock_store:
            mock_store.get_all_endpoints.side_effect = TypeError("test error")
            result = _run(
                handler.handle(
                    "/api/analytics/endpoints/health",
                    {},
                    mock_http_handler,
                )
            )
        assert _status(result) == 500
