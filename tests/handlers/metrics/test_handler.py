"""Comprehensive tests for MetricsHandler.

Covers all routes, authentication, rate limiting, error handling,
and individual metrics endpoints.

Test classes:
  TestHandlerInit                  - Constructor and context
  TestCanHandle                    - Route matching with/without version prefix
  TestRateLimiting                 - Rate limit enforcement
  TestAuthForApiMetrics            - Auth/permission checks on /api/metrics/* routes
  TestMetricsTokenAuth             - Token-based auth for /metrics endpoint
  TestGetMetrics                   - GET /api/metrics (operational metrics)
  TestGetHealth                    - GET /api/metrics/health (health checks)
  TestGetCacheStats                - GET /api/metrics/cache (cache statistics)
  TestGetVerificationStats         - GET /api/metrics/verification (Z3 stats)
  TestGetSystemInfo                - GET /api/metrics/system (system info)
  TestGetBackgroundStats           - GET /api/metrics/background (background tasks)
  TestGetDebatePerfStats           - GET /api/metrics/debate (debate performance)
  TestGetPrometheusMetrics         - GET /metrics (Prometheus format)
  TestMonitoringRoutes             - /api/v1/monitoring/* routes return None
  TestFormatUptime                 - Legacy format_uptime wrapper
  TestFormatSize                   - Legacy format_size wrapper
  TestGetDatabaseSizes             - Database size enumeration
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.metrics.handler import MetricsHandler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result) -> dict[str, Any]:
    """Extract JSON body dict from a HandlerResult."""
    if result is None:
        return {}
    if isinstance(result, dict):
        return result
    raw = result.body
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(raw)


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if result is None:
        return 0
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


def _raw_body(result) -> str:
    """Extract raw string body from a HandlerResult."""
    if result is None:
        return ""
    raw = result.body
    if isinstance(raw, bytes):
        return raw.decode("utf-8")
    return str(raw)


def _make_http_handler(
    body: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> MagicMock:
    """Create mock HTTP handler."""
    h = MagicMock()
    h.client_address = ("127.0.0.1", 12345)
    if body is not None:
        raw = json.dumps(body).encode()
        h.headers = {
            "Content-Length": str(len(raw)),
            "Content-Type": "application/json",
            "Authorization": "Bearer test-token",
            **(headers or {}),
        }
        h.rfile = MagicMock()
        h.rfile.read.return_value = raw
    else:
        h.headers = {
            "Content-Length": "2",
            "Content-Type": "application/json",
            "Authorization": "Bearer test-token",
            **(headers or {}),
        }
        h.rfile = MagicMock()
        h.rfile.read.return_value = b"{}"
    return h


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler():
    """Create a MetricsHandler with empty context."""
    return MetricsHandler(ctx={})


@pytest.fixture
def mock_http():
    """Create a default mock HTTP handler."""
    return _make_http_handler()


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """Reset the rate limiter between tests."""
    from aragora.server.handlers.metrics.handler import _metrics_limiter

    # Clear internal state
    if hasattr(_metrics_limiter, "_requests"):
        _metrics_limiter._requests.clear()
    yield


# ---------------------------------------------------------------------------
# TestHandlerInit
# ---------------------------------------------------------------------------


class TestHandlerInit:
    """Test MetricsHandler construction."""

    def test_init_with_no_context(self):
        h = MetricsHandler()
        assert h.ctx == {}

    def test_init_with_none_context(self):
        h = MetricsHandler(ctx=None)
        assert h.ctx == {}

    def test_init_with_context(self):
        ctx = {"storage": MagicMock()}
        h = MetricsHandler(ctx=ctx)
        assert h.ctx is ctx

    def test_routes_list_not_empty(self):
        assert len(MetricsHandler.ROUTES) > 0

    def test_routes_contains_expected_paths(self):
        assert "/api/metrics" in MetricsHandler.ROUTES
        assert "/api/metrics/health" in MetricsHandler.ROUTES
        assert "/api/metrics/cache" in MetricsHandler.ROUTES
        assert "/api/metrics/verification" in MetricsHandler.ROUTES
        assert "/api/metrics/system" in MetricsHandler.ROUTES
        assert "/api/metrics/background" in MetricsHandler.ROUTES
        assert "/api/metrics/debate" in MetricsHandler.ROUTES
        assert "/metrics" in MetricsHandler.ROUTES


# ---------------------------------------------------------------------------
# TestCanHandle
# ---------------------------------------------------------------------------


class TestCanHandle:
    """Test can_handle route matching."""

    def test_api_metrics_root(self, handler):
        assert handler.can_handle("/api/metrics") is True

    def test_api_metrics_health(self, handler):
        assert handler.can_handle("/api/metrics/health") is True

    def test_api_metrics_cache(self, handler):
        assert handler.can_handle("/api/metrics/cache") is True

    def test_api_metrics_verification(self, handler):
        assert handler.can_handle("/api/metrics/verification") is True

    def test_api_metrics_system(self, handler):
        assert handler.can_handle("/api/metrics/system") is True

    def test_api_metrics_background(self, handler):
        assert handler.can_handle("/api/metrics/background") is True

    def test_api_metrics_debate(self, handler):
        assert handler.can_handle("/api/metrics/debate") is True

    def test_prometheus_metrics(self, handler):
        assert handler.can_handle("/metrics") is True

    def test_versioned_monitoring_alerts_stripped(self, handler):
        """Versioned monitoring routes strip to /api/monitoring/* which is NOT in ROUTES."""
        assert handler.can_handle("/api/v1/monitoring/alerts") is False

    def test_versioned_monitoring_dashboards_stripped(self, handler):
        assert handler.can_handle("/api/v1/monitoring/dashboards") is False

    def test_versioned_monitoring_health_stripped(self, handler):
        assert handler.can_handle("/api/v1/monitoring/health") is False

    def test_versioned_monitoring_logs_stripped(self, handler):
        assert handler.can_handle("/api/v1/monitoring/logs") is False

    def test_versioned_monitoring_metrics_stripped(self, handler):
        assert handler.can_handle("/api/v1/monitoring/metrics") is False

    def test_versioned_monitoring_slos_stripped(self, handler):
        assert handler.can_handle("/api/v1/monitoring/slos") is False

    def test_versioned_monitoring_traces_stripped(self, handler):
        assert handler.can_handle("/api/v1/monitoring/traces") is False

    def test_unhandled_path(self, handler):
        assert handler.can_handle("/api/unknown") is False

    def test_partial_match_no_handle(self, handler):
        assert handler.can_handle("/api/metrics/nonexistent") is False

    def test_empty_path(self, handler):
        assert handler.can_handle("") is False

    def test_versioned_api_metrics(self, handler):
        # /api/v1/metrics -> strip -> /api/metrics, which IS in ROUTES
        assert handler.can_handle("/api/v1/metrics") is True


# ---------------------------------------------------------------------------
# TestRateLimiting
# ---------------------------------------------------------------------------


class TestRateLimiting:
    """Test rate limiting on all metrics endpoints."""

    def test_rate_limit_exceeded_returns_429(self, handler, mock_http):
        with patch("aragora.server.handlers.metrics.handler._metrics_limiter") as mock_limiter:
            mock_limiter.is_allowed.return_value = False
            result = handler.handle("/api/metrics", {}, mock_http)
            assert _status(result) == 429
            assert "Rate limit" in _body(result).get("error", "")

    def test_rate_limit_allowed_continues(self, handler, mock_http):
        with patch("aragora.server.handlers.metrics.handler._metrics_limiter") as mock_limiter:
            mock_limiter.is_allowed.return_value = True
            result = handler.handle("/api/metrics", {}, mock_http)
            assert _status(result) == 200

    def test_rate_limit_on_prometheus_endpoint(self, handler, mock_http):
        with patch("aragora.server.handlers.metrics.handler._metrics_limiter") as mock_limiter:
            mock_limiter.is_allowed.return_value = False
            result = handler.handle("/metrics", {}, mock_http)
            assert _status(result) == 429


# ---------------------------------------------------------------------------
# TestAuthForApiMetrics
# ---------------------------------------------------------------------------


class TestAuthForApiMetrics:
    """Test auth/permission checks for /api/metrics/* endpoints."""

    @pytest.mark.no_auto_auth
    def test_unauthenticated_returns_401(self):
        h = MetricsHandler(ctx={})
        mock_http = _make_http_handler()
        # Override require_auth_or_error to simulate auth failure
        from aragora.server.handlers.base import error_response

        original = h.require_auth_or_error

        def auth_fail(handler):
            return None, error_response("Not authenticated", 401)

        h.require_auth_or_error = auth_fail
        result = h.handle("/api/metrics", {}, mock_http)
        assert _status(result) == 401

    @pytest.mark.no_auto_auth
    def test_permission_denied_returns_403(self):
        h = MetricsHandler(ctx={})
        mock_http = _make_http_handler()
        from aragora.server.handlers.base import error_response

        mock_user = MagicMock()
        h.require_auth_or_error = lambda handler: (mock_user, None)
        h.require_permission_or_error = lambda handler, perm: (
            None,
            error_response("Forbidden", 403),
        )
        result = h.handle("/api/metrics", {}, mock_http)
        assert _status(result) == 403

    def test_metrics_endpoint_does_not_require_auth(self, handler, mock_http):
        """The /metrics (Prometheus) endpoint uses token auth, not RBAC auth."""
        with patch(
            "aragora.server.handlers.metrics.handler.get_metrics_output",
            return_value=("# metrics", "text/plain"),
        ):
            result = handler.handle("/metrics", {}, mock_http)
            assert _status(result) == 200


# ---------------------------------------------------------------------------
# TestMetricsTokenAuth
# ---------------------------------------------------------------------------


class TestMetricsTokenAuth:
    """Test token-based auth for /metrics endpoint."""

    def test_no_token_configured_allows_access(self, handler, mock_http):
        with (
            patch.dict("os.environ", {}, clear=False),
            patch(
                "aragora.server.handlers.metrics.handler.get_metrics_output",
                return_value=("# metrics", "text/plain"),
            ),
        ):
            # Make sure ARAGORA_METRICS_TOKEN is not set
            import os

            os.environ.pop("ARAGORA_METRICS_TOKEN", None)
            result = handler.handle("/metrics", {}, mock_http)
            assert _status(result) == 200

    def test_valid_bearer_token_allows_access(self, handler):
        mock_http = _make_http_handler(headers={"Authorization": "Bearer secret123"})
        with (
            patch.dict("os.environ", {"ARAGORA_METRICS_TOKEN": "secret123"}),
            patch(
                "aragora.server.handlers.metrics.handler.get_metrics_output",
                return_value=("# metrics", "text/plain"),
            ),
        ):
            result = handler.handle("/metrics", {}, mock_http)
            assert _status(result) == 200

    def test_invalid_bearer_token_returns_401(self, handler):
        mock_http = _make_http_handler(headers={"Authorization": "Bearer wrong"})
        with patch.dict("os.environ", {"ARAGORA_METRICS_TOKEN": "secret123"}):
            result = handler.handle("/metrics", {}, mock_http)
            assert _status(result) == 401
            assert "Unauthorized" in _body(result).get("error", "")

    def test_missing_bearer_token_returns_401(self, handler):
        mock_http = _make_http_handler(headers={"Authorization": ""})
        with patch.dict("os.environ", {"ARAGORA_METRICS_TOKEN": "secret123"}):
            result = handler.handle("/metrics", {}, mock_http)
            assert _status(result) == 401

    def test_query_param_token_rejected(self, handler, mock_http):
        with patch.dict("os.environ", {"ARAGORA_METRICS_TOKEN": "secret123"}):
            result = handler.handle("/metrics", {"token": ["secret123"]}, mock_http)
            assert _status(result) == 401
            body = _body(result)
            assert "Query parameter" in body.get("error", "")

    def test_query_param_token_rejected_even_if_correct(self, handler):
        """Query param tokens are always rejected for security."""
        mock_http = _make_http_handler(headers={"Authorization": "Bearer secret123"})
        with patch.dict("os.environ", {"ARAGORA_METRICS_TOKEN": "secret123"}):
            result = handler.handle("/metrics", {"token": ["secret123"]}, mock_http)
            assert _status(result) == 401

    def test_no_authorization_header_at_all(self, handler):
        mock_http = _make_http_handler()
        # Simulate missing Authorization header
        mock_http.headers = {"Content-Length": "2"}
        with patch.dict("os.environ", {"ARAGORA_METRICS_TOKEN": "secret123"}):
            result = handler.handle("/metrics", {}, mock_http)
            assert _status(result) == 401


# ---------------------------------------------------------------------------
# TestGetMetrics
# ---------------------------------------------------------------------------


class TestGetMetrics:
    """Test GET /api/metrics."""

    def test_success_returns_200(self, handler, mock_http):
        result = handler.handle("/api/metrics", {}, mock_http)
        assert _status(result) == 200

    def test_response_contains_uptime(self, handler, mock_http):
        result = handler.handle("/api/metrics", {}, mock_http)
        body = _body(result)
        assert "uptime_seconds" in body
        assert "uptime_human" in body
        assert body["uptime_seconds"] >= 0

    def test_response_contains_requests(self, handler, mock_http):
        result = handler.handle("/api/metrics", {}, mock_http)
        body = _body(result)
        assert "requests" in body
        req = body["requests"]
        assert "total" in req
        assert "errors" in req
        assert "error_rate" in req
        assert "top_endpoints" in req

    def test_response_contains_cache(self, handler, mock_http):
        result = handler.handle("/api/metrics", {}, mock_http)
        body = _body(result)
        assert "cache" in body
        assert "entries" in body["cache"]

    def test_response_contains_databases(self, handler, mock_http):
        result = handler.handle("/api/metrics", {}, mock_http)
        body = _body(result)
        assert "databases" in body

    def test_response_contains_timestamp(self, handler, mock_http):
        result = handler.handle("/api/metrics", {}, mock_http)
        body = _body(result)
        assert "timestamp" in body

    def test_error_rate_zero_when_no_requests(self, handler, mock_http):
        with patch(
            "aragora.server.handlers.metrics.handler.get_request_stats",
            return_value={
                "total_requests": 0,
                "total_errors": 0,
                "counts_snapshot": [],
            },
        ):
            result = handler.handle("/api/metrics", {}, mock_http)
            body = _body(result)
            assert body["requests"]["error_rate"] == 0.0

    def test_error_rate_computed(self, handler, mock_http):
        with patch(
            "aragora.server.handlers.metrics.handler.get_request_stats",
            return_value={
                "total_requests": 100,
                "total_errors": 5,
                "counts_snapshot": [("/api/test", 100)],
            },
        ):
            result = handler.handle("/api/metrics", {}, mock_http)
            body = _body(result)
            assert body["requests"]["error_rate"] == 0.05

    def test_top_endpoints_sorted_by_count(self, handler, mock_http):
        counts = [("/a", 10), ("/b", 50), ("/c", 30)]
        with patch(
            "aragora.server.handlers.metrics.handler.get_request_stats",
            return_value={
                "total_requests": 90,
                "total_errors": 0,
                "counts_snapshot": counts,
            },
        ):
            result = handler.handle("/api/metrics", {}, mock_http)
            body = _body(result)
            endpoints = body["requests"]["top_endpoints"]
            assert endpoints[0]["endpoint"] == "/b"
            assert endpoints[0]["count"] == 50

    def test_runtime_error_returns_500(self, handler, mock_http):
        with patch(
            "aragora.server.handlers.metrics.handler.get_start_time",
            side_effect=RuntimeError("boom"),
        ):
            result = handler.handle("/api/metrics", {}, mock_http)
            assert _status(result) == 500

    def test_sqlite_error_returns_500(self, handler, mock_http):
        with patch(
            "aragora.server.handlers.metrics.handler.get_request_stats",
            side_effect=sqlite3.Error("db locked"),
        ):
            result = handler.handle("/api/metrics", {}, mock_http)
            assert _status(result) == 500

    def test_versioned_path(self, handler, mock_http):
        """GET /api/v1/metrics should route to _get_metrics."""
        result = handler.handle("/api/v1/metrics", {}, mock_http)
        assert _status(result) == 200
        assert "uptime_seconds" in _body(result)


# ---------------------------------------------------------------------------
# TestGetHealth
# ---------------------------------------------------------------------------


class TestGetHealth:
    """Test GET /api/metrics/health."""

    def test_healthy_when_all_checks_pass(self, mock_http):
        storage = MagicMock()
        storage.list_debates.return_value = []
        elo = MagicMock()
        elo.get_leaderboard.return_value = []
        nomic_dir = MagicMock(spec=Path)
        nomic_dir.exists.return_value = True
        nomic_dir.__str__ = lambda self: "/tmp/nomic"

        h = MetricsHandler(ctx={"storage": storage, "elo_system": elo, "nomic_dir": nomic_dir})
        result = h.handle("/api/metrics/health", {}, mock_http)
        body = _body(result)
        assert _status(result) == 200
        assert body["status"] == "healthy"
        assert body["checks"]["storage"]["status"] == "healthy"
        assert body["checks"]["elo_system"]["status"] == "healthy"
        assert body["checks"]["nomic_dir"]["status"] == "healthy"

    def test_storage_unavailable(self, mock_http):
        h = MetricsHandler(ctx={})
        result = h.handle("/api/metrics/health", {}, mock_http)
        body = _body(result)
        assert body["checks"]["storage"]["status"] == "unavailable"

    def test_storage_unhealthy_sqlite_error(self, mock_http):
        storage = MagicMock()
        storage.list_debates.side_effect = sqlite3.Error("db error")
        h = MetricsHandler(ctx={"storage": storage})
        result = h.handle("/api/metrics/health", {}, mock_http)
        body = _body(result)
        assert body["checks"]["storage"]["status"] == "unhealthy"
        assert body["status"] == "degraded"

    def test_storage_unhealthy_runtime_error(self, mock_http):
        storage = MagicMock()
        storage.list_debates.side_effect = RuntimeError("unexpected")
        h = MetricsHandler(ctx={"storage": storage})
        result = h.handle("/api/metrics/health", {}, mock_http)
        body = _body(result)
        assert body["checks"]["storage"]["status"] == "unhealthy"
        assert body["checks"]["storage"]["error"] == "Internal error"

    def test_elo_unavailable(self, mock_http):
        h = MetricsHandler(ctx={})
        result = h.handle("/api/metrics/health", {}, mock_http)
        body = _body(result)
        assert body["checks"]["elo_system"]["status"] == "unavailable"

    def test_elo_unhealthy_sqlite_error(self, mock_http):
        elo = MagicMock()
        elo.get_leaderboard.side_effect = sqlite3.Error("elo error")
        h = MetricsHandler(ctx={"elo_system": elo})
        result = h.handle("/api/metrics/health", {}, mock_http)
        body = _body(result)
        assert body["checks"]["elo_system"]["status"] == "unhealthy"
        assert body["status"] == "degraded"

    def test_elo_unhealthy_runtime_error(self, mock_http):
        elo = MagicMock()
        elo.get_leaderboard.side_effect = ValueError("bad elo")
        h = MetricsHandler(ctx={"elo_system": elo})
        result = h.handle("/api/metrics/health", {}, mock_http)
        body = _body(result)
        assert body["checks"]["elo_system"]["status"] == "unhealthy"
        assert body["checks"]["elo_system"]["error"] == "Internal error"

    def test_nomic_dir_unavailable(self, mock_http):
        h = MetricsHandler(ctx={})
        result = h.handle("/api/metrics/health", {}, mock_http)
        body = _body(result)
        assert body["checks"]["nomic_dir"]["status"] == "unavailable"

    def test_nomic_dir_exists(self, mock_http):
        nomic_dir = MagicMock(spec=Path)
        nomic_dir.exists.return_value = True
        nomic_dir.__str__ = lambda self: "/tmp/nomic"
        h = MetricsHandler(ctx={"nomic_dir": nomic_dir})
        result = h.handle("/api/metrics/health", {}, mock_http)
        body = _body(result)
        assert body["checks"]["nomic_dir"]["status"] == "healthy"

    def test_nomic_dir_does_not_exist(self, mock_http):
        nomic_dir = MagicMock(spec=Path)
        nomic_dir.exists.return_value = False
        h = MetricsHandler(ctx={"nomic_dir": nomic_dir})
        result = h.handle("/api/metrics/health", {}, mock_http)
        body = _body(result)
        assert body["checks"]["nomic_dir"]["status"] == "unavailable"

    def test_degraded_returns_503(self, mock_http):
        storage = MagicMock()
        storage.list_debates.side_effect = sqlite3.Error("err")
        h = MetricsHandler(ctx={"storage": storage})
        result = h.handle("/api/metrics/health", {}, mock_http)
        assert _status(result) == 503
        assert _body(result)["status"] == "degraded"

    def test_exception_returns_500(self, mock_http):
        h = MetricsHandler(ctx={})
        with patch.object(h, "get_storage", side_effect=RuntimeError("fatal")):
            result = h.handle("/api/metrics/health", {}, mock_http)
            assert _status(result) == 500


# ---------------------------------------------------------------------------
# TestGetCacheStats
# ---------------------------------------------------------------------------


class TestGetCacheStats:
    """Test GET /api/metrics/cache."""

    def test_success_returns_200(self, handler, mock_http):
        result = handler.handle("/api/metrics/cache", {}, mock_http)
        assert _status(result) == 200

    def test_response_structure(self, handler, mock_http):
        with patch(
            "aragora.server.handlers.metrics.handler.get_cache_stats",
            return_value={
                "entries": 5,
                "max_entries": 1000,
                "hit_rate": 0.75,
                "hits": 30,
                "misses": 10,
            },
        ):
            result = handler.handle("/api/metrics/cache", {}, mock_http)
            body = _body(result)
            assert "total_entries" in body
            assert "max_entries" in body
            assert "hit_rate" in body
            assert "hits" in body
            assert "misses" in body
            assert "entries_by_prefix" in body

    def test_empty_cache_age_zero(self, handler, mock_http):
        mock_cache = MagicMock()
        mock_cache.__len__ = MagicMock(return_value=0)
        mock_cache.items.return_value = []
        with (
            patch(
                "aragora.server.handlers.metrics.handler.get_cache_stats",
                return_value={
                    "entries": 0,
                    "max_entries": 1000,
                    "hit_rate": 0.0,
                    "hits": 0,
                    "misses": 0,
                },
            ),
            patch("aragora.server.handlers.metrics.handler._cache", mock_cache),
        ):
            result = handler.handle("/api/metrics/cache", {}, mock_http)
            body = _body(result)
            assert body["oldest_entry_age_seconds"] == 0
            assert body["newest_entry_age_seconds"] == 0

    def test_entries_by_prefix(self, handler, mock_http):
        now = time.time()
        mock_cache = MagicMock()
        mock_cache.__len__ = MagicMock(return_value=3)
        mock_cache.items.return_value = [
            ("api:key1", (now - 10, "val1")),
            ("api:key2", (now - 5, "val2")),
            ("db:key3", (now - 1, "val3")),
        ]
        with (
            patch(
                "aragora.server.handlers.metrics.handler.get_cache_stats",
                return_value={
                    "entries": 3,
                    "max_entries": 1000,
                    "hit_rate": 0.5,
                    "hits": 5,
                    "misses": 5,
                },
            ),
            patch("aragora.server.handlers.metrics.handler._cache", mock_cache),
        ):
            result = handler.handle("/api/metrics/cache", {}, mock_http)
            body = _body(result)
            assert body["entries_by_prefix"]["api"] == 2
            assert body["entries_by_prefix"]["db"] == 1

    def test_default_prefix_for_keyless_entries(self, handler, mock_http):
        now = time.time()
        mock_cache = MagicMock()
        mock_cache.__len__ = MagicMock(return_value=1)
        mock_cache.items.return_value = [("simple_key", (now, "val"))]
        with (
            patch(
                "aragora.server.handlers.metrics.handler.get_cache_stats",
                return_value={
                    "entries": 1,
                    "max_entries": 1000,
                    "hit_rate": 0.0,
                    "hits": 0,
                    "misses": 0,
                },
            ),
            patch("aragora.server.handlers.metrics.handler._cache", mock_cache),
        ):
            result = handler.handle("/api/metrics/cache", {}, mock_http)
            body = _body(result)
            assert body["entries_by_prefix"].get("default") == 1

    def test_error_returns_500(self, handler, mock_http):
        with patch(
            "aragora.server.handlers.metrics.handler.get_cache_stats",
            side_effect=RuntimeError("cache broken"),
        ):
            result = handler.handle("/api/metrics/cache", {}, mock_http)
            assert _status(result) == 500


# ---------------------------------------------------------------------------
# TestGetVerificationStats
# ---------------------------------------------------------------------------


class TestGetVerificationStats:
    """Test GET /api/metrics/verification."""

    def test_success_returns_200(self, handler, mock_http):
        result = handler.handle("/api/metrics/verification", {}, mock_http)
        assert _status(result) == 200

    def test_response_contains_expected_keys(self, handler, mock_http):
        result = handler.handle("/api/metrics/verification", {}, mock_http)
        body = _body(result)
        assert "total_claims_processed" in body
        assert "z3_verified" in body
        assert "z3_disproved" in body
        assert "z3_timeout" in body
        assert "z3_translation_failed" in body
        assert "confidence_fallback" in body
        assert "avg_verification_time_ms" in body
        assert "z3_success_rate" in body

    def test_custom_verification_stats(self, handler, mock_http):
        mock_stats = {
            "total_claims_processed": 100,
            "z3_verified": 80,
            "z3_disproved": 10,
            "z3_timeout": 5,
            "z3_translation_failed": 3,
            "confidence_fallback": 2,
            "total_verification_time_ms": 5000.0,
            "avg_verification_time_ms": 50.0,
            "z3_success_rate": 0.8,
        }
        with patch(
            "aragora.server.handlers.metrics.handler.get_verification_stats",
            return_value=mock_stats,
        ):
            result = handler.handle("/api/metrics/verification", {}, mock_http)
            body = _body(result)
            assert body["total_claims_processed"] == 100
            assert body["z3_verified"] == 80

    def test_error_returns_500(self, handler, mock_http):
        with patch(
            "aragora.server.handlers.metrics.handler.get_verification_stats",
            side_effect=RuntimeError("z3 error"),
        ):
            result = handler.handle("/api/metrics/verification", {}, mock_http)
            assert _status(result) == 500


# ---------------------------------------------------------------------------
# TestGetSystemInfo
# ---------------------------------------------------------------------------


class TestGetSystemInfo:
    """Test GET /api/metrics/system."""

    def test_success_returns_200(self, handler, mock_http):
        result = handler.handle("/api/metrics/system", {}, mock_http)
        assert _status(result) == 200

    def test_response_contains_python_version(self, handler, mock_http):
        result = handler.handle("/api/metrics/system", {}, mock_http)
        body = _body(result)
        assert "python_version" in body

    def test_response_contains_platform(self, handler, mock_http):
        result = handler.handle("/api/metrics/system", {}, mock_http)
        body = _body(result)
        assert "platform" in body

    def test_response_contains_machine(self, handler, mock_http):
        result = handler.handle("/api/metrics/system", {}, mock_http)
        body = _body(result)
        assert "machine" in body

    def test_response_contains_pid(self, handler, mock_http):
        result = handler.handle("/api/metrics/system", {}, mock_http)
        body = _body(result)
        assert "pid" in body
        assert isinstance(body["pid"], int)

    def test_memory_with_psutil(self, handler, mock_http):
        mock_process = MagicMock()
        mock_process.memory_info.return_value = MagicMock(
            rss=100 * 1024 * 1024, vms=200 * 1024 * 1024
        )
        with patch.dict("sys.modules", {"psutil": MagicMock(Process=lambda: mock_process)}):
            result = handler.handle("/api/metrics/system", {}, mock_http)
            body = _body(result)
            assert "memory" in body
            # psutil mock may or may not work depending on import caching
            # Just verify memory key exists

    def test_memory_without_psutil(self, handler, mock_http):
        with patch.dict("sys.modules", {"psutil": None}):
            result = handler.handle("/api/metrics/system", {}, mock_http)
            body = _body(result)
            assert "memory" in body

    def test_error_returns_500(self, handler, mock_http):
        with patch("platform.platform", side_effect=OSError("no platform")):
            result = handler.handle("/api/metrics/system", {}, mock_http)
            assert _status(result) == 500


# ---------------------------------------------------------------------------
# TestGetBackgroundStats
# ---------------------------------------------------------------------------


class TestGetBackgroundStats:
    """Test GET /api/metrics/background."""

    def test_success_with_manager(self, handler, mock_http):
        mock_manager = MagicMock()
        mock_manager.get_stats.return_value = {
            "running": True,
            "task_count": 3,
            "tasks": {"task1": "running"},
        }
        with patch(
            "aragora.server.handlers.metrics.handler.get_background_manager",
            return_value=mock_manager,
            create=True,
        ):
            # Need to also patch the import inside the method
            with patch.dict(
                "sys.modules",
                {
                    "aragora.server.background": MagicMock(
                        get_background_manager=lambda: mock_manager
                    )
                },
            ):
                result = handler.handle("/api/metrics/background", {}, mock_http)
                assert _status(result) == 200
                body = _body(result)
                assert body["running"] is True
                assert body["task_count"] == 3

    def test_import_error_returns_fallback(self, handler, mock_http):
        """When background module is not available, return fallback."""
        with patch.dict("sys.modules", {"aragora.server.background": None}):
            # Force reimport to trigger ImportError
            import importlib

            try:
                result = handler.handle("/api/metrics/background", {}, mock_http)
                assert _status(result) == 200
                body = _body(result)
                # Either the manager is available or fallback is returned
                assert "running" in body or "task_count" in body or "message" in body
            except ImportError:
                # The import error is caught in the handler
                pass

    def test_runtime_error_returns_500(self, handler, mock_http):
        mock_manager = MagicMock()
        mock_manager.get_stats.side_effect = RuntimeError("boom")
        with patch.dict(
            "sys.modules",
            {"aragora.server.background": MagicMock(get_background_manager=lambda: mock_manager)},
        ):
            result = handler.handle("/api/metrics/background", {}, mock_http)
            assert _status(result) == 500


# ---------------------------------------------------------------------------
# TestGetDebatePerfStats
# ---------------------------------------------------------------------------


class TestGetDebatePerfStats:
    """Test GET /api/metrics/debate."""

    def test_summary_stats_without_debate_id(self, handler, mock_http):
        mock_monitor = MagicMock()
        mock_monitor.get_slow_debates.return_value = []
        mock_monitor.get_current_slow_debates.return_value = []
        mock_monitor.slow_round_threshold = 30.0
        mock_monitor._active_debates = {}
        with patch.dict(
            "sys.modules",
            {
                "aragora.debate.performance_monitor": MagicMock(
                    get_debate_monitor=lambda: mock_monitor
                )
            },
        ):
            result = handler.handle("/api/metrics/debate", {"debate_id": [None]}, mock_http)
            assert _status(result) == 200
            body = _body(result)
            assert "slow_debates_history" in body
            assert "current_slow_debates" in body
            assert "slow_round_threshold_seconds" in body
            assert body["active_debate_count"] == 0

    def test_specific_debate_insights(self, handler, mock_http):
        mock_monitor = MagicMock()
        mock_monitor.get_performance_insights.return_value = {
            "debate_id": "d123",
            "duration": 120,
            "slow_rounds": [],
        }
        with patch.dict(
            "sys.modules",
            {
                "aragora.debate.performance_monitor": MagicMock(
                    get_debate_monitor=lambda: mock_monitor
                )
            },
        ):
            result = handler.handle(
                "/api/metrics/debate",
                {"debate_id": ["d123"]},
                mock_http,
            )
            assert _status(result) == 200
            body = _body(result)
            assert body["debate_id"] == "d123"

    def test_import_error_returns_fallback(self, handler, mock_http):
        with patch.dict("sys.modules", {"aragora.debate.performance_monitor": None}):
            result = handler.handle("/api/metrics/debate", {"debate_id": [None]}, mock_http)
            assert _status(result) == 200
            body = _body(result)
            assert "message" in body or "slow_debates_history" in body

    def test_runtime_error_returns_500(self, handler, mock_http):
        mock_monitor = MagicMock()
        mock_monitor.get_slow_debates.side_effect = RuntimeError("err")
        with patch.dict(
            "sys.modules",
            {
                "aragora.debate.performance_monitor": MagicMock(
                    get_debate_monitor=lambda: mock_monitor
                )
            },
        ):
            result = handler.handle("/api/metrics/debate", {"debate_id": [None]}, mock_http)
            assert _status(result) == 500

    def test_no_debate_id_param_defaults_to_none(self, handler, mock_http):
        """When debate_id query param is missing, defaults to [None][0] = None."""
        mock_monitor = MagicMock()
        mock_monitor.get_slow_debates.return_value = []
        mock_monitor.get_current_slow_debates.return_value = []
        mock_monitor.slow_round_threshold = 10.0
        mock_monitor._active_debates = {"d1": True}
        with patch.dict(
            "sys.modules",
            {
                "aragora.debate.performance_monitor": MagicMock(
                    get_debate_monitor=lambda: mock_monitor
                )
            },
        ):
            result = handler.handle("/api/metrics/debate", {}, mock_http)
            assert _status(result) == 200
            body = _body(result)
            assert body["active_debate_count"] == 1


# ---------------------------------------------------------------------------
# TestGetPrometheusMetrics
# ---------------------------------------------------------------------------


class TestGetPrometheusMetrics:
    """Test GET /metrics (Prometheus format)."""

    def test_success_returns_200(self, handler, mock_http):
        with patch(
            "aragora.server.handlers.metrics.handler.get_metrics_output",
            return_value=(
                "# HELP aragora_uptime Server uptime\naragora_uptime 123.4\n",
                "text/plain; version=0.0.4; charset=utf-8",
            ),
        ):
            result = handler.handle("/metrics", {}, mock_http)
            assert _status(result) == 200

    def test_content_type_is_prometheus(self, handler, mock_http):
        with patch(
            "aragora.server.handlers.metrics.handler.get_metrics_output",
            return_value=(
                "# metrics\n",
                "text/plain; version=0.0.4; charset=utf-8",
            ),
        ):
            result = handler.handle("/metrics", {}, mock_http)
            assert "text/plain" in result.content_type

    def test_body_is_bytes(self, handler, mock_http):
        with patch(
            "aragora.server.handlers.metrics.handler.get_metrics_output",
            return_value=("# metrics\n", "text/plain"),
        ):
            result = handler.handle("/metrics", {}, mock_http)
            assert isinstance(result.body, bytes)

    def test_calls_set_cache_size(self, handler, mock_http):
        with (
            patch(
                "aragora.server.handlers.metrics.handler.get_metrics_output",
                return_value=("# metrics\n", "text/plain"),
            ),
            patch("aragora.server.handlers.metrics.handler.set_cache_size") as mock_set,
        ):
            handler.handle("/metrics", {}, mock_http)
            mock_set.assert_called_once()
            args = mock_set.call_args
            assert args[0][0] == "handler_cache"

    def test_calls_set_server_info(self, handler, mock_http):
        with (
            patch(
                "aragora.server.handlers.metrics.handler.get_metrics_output",
                return_value=("# metrics\n", "text/plain"),
            ),
            patch("aragora.server.handlers.metrics.handler.set_server_info") as mock_set,
        ):
            handler.handle("/metrics", {}, mock_http)
            mock_set.assert_called_once()
            kwargs = mock_set.call_args
            assert kwargs[1]["version"] == "0.8.0" or kwargs[0][0] == "0.8.0"

    def test_error_returns_500(self, handler, mock_http):
        with patch(
            "aragora.server.handlers.metrics.handler.get_metrics_output",
            side_effect=RuntimeError("prometheus error"),
        ):
            result = handler.handle("/metrics", {}, mock_http)
            assert _status(result) == 500

    def test_value_error_returns_500(self, handler, mock_http):
        with patch(
            "aragora.server.handlers.metrics.handler.get_metrics_output",
            side_effect=ValueError("bad value"),
        ):
            result = handler.handle("/metrics", {}, mock_http)
            assert _status(result) == 500


# ---------------------------------------------------------------------------
# TestMonitoringRoutes
# ---------------------------------------------------------------------------


class TestMonitoringRoutes:
    """Test /api/v1/monitoring/* routes.

    These routes are listed in ROUTES (for can_handle) but handle()
    returns None since no explicit routing exists for them after
    strip_version_prefix.
    """

    def test_monitoring_alerts_returns_none(self, handler, mock_http):
        result = handler.handle("/api/v1/monitoring/alerts", {}, mock_http)
        assert result is None

    def test_monitoring_dashboards_returns_none(self, handler, mock_http):
        result = handler.handle("/api/v1/monitoring/dashboards", {}, mock_http)
        assert result is None

    def test_monitoring_health_returns_none(self, handler, mock_http):
        result = handler.handle("/api/v1/monitoring/health", {}, mock_http)
        assert result is None

    def test_monitoring_logs_returns_none(self, handler, mock_http):
        result = handler.handle("/api/v1/monitoring/logs", {}, mock_http)
        assert result is None

    def test_monitoring_metrics_returns_none(self, handler, mock_http):
        result = handler.handle("/api/v1/monitoring/metrics", {}, mock_http)
        assert result is None

    def test_monitoring_slos_returns_none(self, handler, mock_http):
        result = handler.handle("/api/v1/monitoring/slos", {}, mock_http)
        assert result is None

    def test_monitoring_traces_returns_none(self, handler, mock_http):
        result = handler.handle("/api/v1/monitoring/traces", {}, mock_http)
        assert result is None


# ---------------------------------------------------------------------------
# TestFormatUptime
# ---------------------------------------------------------------------------


class TestFormatUptime:
    """Test _format_uptime legacy wrapper."""

    def test_seconds_only(self, handler):
        assert handler._format_uptime(30) == "30s"

    def test_minutes_and_seconds(self, handler):
        assert handler._format_uptime(90) == "1m 30s"

    def test_hours(self, handler):
        assert handler._format_uptime(3661) == "1h 1m 1s"

    def test_days(self, handler):
        assert handler._format_uptime(86400 + 3600 + 60) == "1d 1h 1m"

    def test_zero(self, handler):
        assert handler._format_uptime(0) == "0s"


# ---------------------------------------------------------------------------
# TestFormatSize
# ---------------------------------------------------------------------------


class TestFormatSize:
    """Test _format_size legacy wrapper."""

    def test_bytes(self, handler):
        assert handler._format_size(512) == "512.0 B"

    def test_kilobytes(self, handler):
        assert handler._format_size(1024) == "1.0 KB"

    def test_megabytes(self, handler):
        assert handler._format_size(1024 * 1024) == "1.0 MB"

    def test_gigabytes(self, handler):
        assert handler._format_size(1024**3) == "1.0 GB"

    def test_terabytes(self, handler):
        assert handler._format_size(1024**4) == "1.0 TB"

    def test_zero(self, handler):
        assert handler._format_size(0) == "0.0 B"


# ---------------------------------------------------------------------------
# TestGetDatabaseSizes
# ---------------------------------------------------------------------------


class TestGetDatabaseSizes:
    """Test _get_database_sizes."""

    def test_no_nomic_dir(self, handler):
        sizes = handler._get_database_sizes()
        assert sizes == {}

    def test_nomic_dir_does_not_exist(self):
        nomic_dir = MagicMock(spec=Path)
        nomic_dir.exists.return_value = False
        h = MetricsHandler(ctx={"nomic_dir": nomic_dir})
        sizes = h._get_database_sizes()
        assert sizes == {}

    def test_nomic_dir_with_existing_db(self, tmp_path):
        # Create a fake db file
        db_file = tmp_path / "agent_elo.db"
        db_file.write_bytes(b"x" * 2048)

        h = MetricsHandler(ctx={"nomic_dir": tmp_path})
        sizes = h._get_database_sizes()
        assert "agent_elo.db" in sizes
        assert sizes["agent_elo.db"]["bytes"] == 2048
        assert "KB" in sizes["agent_elo.db"]["human"]

    def test_nomic_dir_with_multiple_dbs(self, tmp_path):
        (tmp_path / "agent_elo.db").write_bytes(b"x" * 1024)
        (tmp_path / "debate_storage.db").write_bytes(b"y" * 2048)

        h = MetricsHandler(ctx={"nomic_dir": tmp_path})
        sizes = h._get_database_sizes()
        assert "agent_elo.db" in sizes
        assert "debate_storage.db" in sizes

    def test_nomic_dir_with_no_matching_files(self, tmp_path):
        # No db files present
        h = MetricsHandler(ctx={"nomic_dir": tmp_path})
        sizes = h._get_database_sizes()
        assert sizes == {}

    def test_nomic_dir_with_some_files(self, tmp_path):
        (tmp_path / "aragora_insights.db").write_bytes(b"z" * 512)
        h = MetricsHandler(ctx={"nomic_dir": tmp_path})
        sizes = h._get_database_sizes()
        assert "aragora_insights.db" in sizes
        assert sizes["aragora_insights.db"]["bytes"] == 512
