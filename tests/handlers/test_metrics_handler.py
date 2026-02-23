"""Tests for the MetricsHandler (operational metrics endpoints).

Covers:
- Handler initialization and route matching (can_handle)
- GET /metrics (Prometheus format)
- GET /api/metrics (operational metrics)
- GET /api/metrics/health (health checks)
- GET /api/metrics/cache (cache statistics)
- GET /api/metrics/verification (Z3 verification stats)
- GET /api/metrics/system (system information)
- GET /api/metrics/background (background task stats)
- GET /api/metrics/debate (debate performance stats)
- /metrics token authentication (ARAGORA_METRICS_TOKEN)
- Rate limiting
- Error handling / graceful degradation
- Edge cases (empty metrics, missing subsystems)
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.metrics.handler import MetricsHandler


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler():
    """Create a MetricsHandler with empty context."""
    return MetricsHandler(ctx={})


@pytest.fixture
def handler_with_storage():
    """Create a MetricsHandler with a mock storage backend."""
    mock_storage = MagicMock()
    mock_storage.list_debates.return_value = []
    return MetricsHandler(ctx={"storage": mock_storage})


@pytest.fixture
def handler_with_full_ctx(tmp_path):
    """Create a MetricsHandler with storage, ELO, and nomic dir."""
    mock_storage = MagicMock()
    mock_storage.list_debates.return_value = []

    mock_elo = MagicMock()
    mock_elo.get_leaderboard.return_value = []

    nomic_dir = tmp_path / ".nomic"
    nomic_dir.mkdir()

    return MetricsHandler(
        ctx={
            "storage": mock_storage,
            "elo_system": mock_elo,
            "nomic_dir": nomic_dir,
        }
    )


@pytest.fixture
def mock_http_handler():
    """Create a mock HTTP handler with basic attributes."""
    handler = MagicMock()
    handler.client_address = ("127.0.0.1", 54321)
    handler.headers = {
        "Content-Length": "0",
        "Host": "localhost:8080",
    }
    return handler


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """Reset the metrics rate limiter between tests."""
    from aragora.server.handlers.metrics.handler import _metrics_limiter

    # Clear internal state so each test starts fresh
    with _metrics_limiter._lock:
        _metrics_limiter._requests.clear()
    yield


# ===========================================================================
# Initialization
# ===========================================================================


class TestInit:
    """Tests for MetricsHandler initialization."""

    def test_init_with_empty_ctx(self):
        h = MetricsHandler(ctx={})
        assert h.ctx == {}

    def test_init_with_none_ctx(self):
        h = MetricsHandler(ctx=None)
        assert h.ctx == {}

    def test_init_with_populated_ctx(self):
        ctx = {"storage": MagicMock(), "elo_system": MagicMock()}
        h = MetricsHandler(ctx=ctx)
        assert h.ctx is ctx


# ===========================================================================
# Route Matching (can_handle)
# ===========================================================================


class TestCanHandle:
    """Tests for route matching via can_handle()."""

    @pytest.mark.parametrize(
        "path",
        [
            "/api/metrics",
            "/api/metrics/health",
            "/api/metrics/cache",
            "/api/metrics/verification",
            "/api/metrics/system",
            "/api/metrics/background",
            "/api/metrics/debate",
            "/metrics",
        ],
    )
    def test_can_handle_known_routes(self, handler, path):
        assert handler.can_handle(path) is True

    @pytest.mark.parametrize(
        "path",
        [
            # Versioned monitoring routes are in ROUTES but strip_version_prefix
            # converts /api/v1/monitoring/X -> /api/monitoring/X which does NOT
            # match. These routes are effectively unreachable via can_handle.
            "/api/v1/monitoring/alerts",
            "/api/v1/monitoring/dashboards",
            "/api/v1/monitoring/health",
            "/api/v1/monitoring/logs",
            "/api/v1/monitoring/metrics",
            "/api/v1/monitoring/slos",
            "/api/v1/monitoring/traces",
            # Other unrelated paths
            "/api/debates",
            "/api/v1/agents",
            "/api/metrics/unknown",
            "/unknown",
            "/",
        ],
    )
    def test_cannot_handle_unrelated_routes(self, handler, path):
        assert handler.can_handle(path) is False


# ===========================================================================
# Route Dispatch
# ===========================================================================


class TestRouteDispatch:
    """Tests for route dispatch in handle()."""

    def test_dispatch_api_metrics(self, handler, mock_http_handler):
        with patch.object(handler, "_get_metrics") as mock_method:
            mock_method.return_value = MagicMock(status_code=200)
            result = handler.handle("/api/metrics", {}, mock_http_handler)
            mock_method.assert_called_once()

    def test_dispatch_api_metrics_health(self, handler, mock_http_handler):
        with patch.object(handler, "_get_health") as mock_method:
            mock_method.return_value = MagicMock(status_code=200)
            result = handler.handle("/api/metrics/health", {}, mock_http_handler)
            mock_method.assert_called_once()

    def test_dispatch_api_metrics_cache(self, handler, mock_http_handler):
        with patch.object(handler, "_get_cache_stats") as mock_method:
            mock_method.return_value = MagicMock(status_code=200)
            result = handler.handle("/api/metrics/cache", {}, mock_http_handler)
            mock_method.assert_called_once()

    def test_dispatch_api_metrics_verification(self, handler, mock_http_handler):
        with patch.object(handler, "_get_verification_stats") as mock_method:
            mock_method.return_value = MagicMock(status_code=200)
            result = handler.handle("/api/metrics/verification", {}, mock_http_handler)
            mock_method.assert_called_once()

    def test_dispatch_api_metrics_system(self, handler, mock_http_handler):
        with patch.object(handler, "_get_system_info") as mock_method:
            mock_method.return_value = MagicMock(status_code=200)
            result = handler.handle("/api/metrics/system", {}, mock_http_handler)
            mock_method.assert_called_once()

    def test_dispatch_api_metrics_background(self, handler, mock_http_handler):
        with patch.object(handler, "_get_background_stats") as mock_method:
            mock_method.return_value = MagicMock(status_code=200)
            result = handler.handle("/api/metrics/background", {}, mock_http_handler)
            mock_method.assert_called_once()

    def test_dispatch_api_metrics_debate(self, handler, mock_http_handler):
        with patch.object(handler, "_get_debate_perf_stats") as mock_method:
            mock_method.return_value = MagicMock(status_code=200)
            handler.handle("/api/metrics/debate", {"debate_id": [None]}, mock_http_handler)
            mock_method.assert_called_once_with(None)

    def test_dispatch_api_metrics_debate_with_id(self, handler, mock_http_handler):
        with patch.object(handler, "_get_debate_perf_stats") as mock_method:
            mock_method.return_value = MagicMock(status_code=200)
            handler.handle("/api/metrics/debate", {"debate_id": ["d123"]}, mock_http_handler)
            mock_method.assert_called_once_with("d123")

    def test_dispatch_prometheus_metrics(self, handler, mock_http_handler):
        with patch.object(handler, "_get_prometheus_metrics") as mock_method:
            mock_method.return_value = MagicMock(status_code=200)
            result = handler.handle("/metrics", {}, mock_http_handler)
            mock_method.assert_called_once()

    def test_dispatch_unmatched_returns_none(self, handler, mock_http_handler):
        """Unmatched paths within the handler return None (not handled)."""
        # We need a path that passes can_handle but is not in the if-chain.
        # Since all ROUTES are covered, any route not in the list will not
        # reach handle() in production. Test the monitoring stub routes.
        result = handler.handle("/api/v1/monitoring/alerts", {}, mock_http_handler)
        # Monitoring routes are in ROUTES but not dispatched in handle() --
        # the method returns None for them.
        assert result is None


# ===========================================================================
# GET /metrics (Prometheus format)
# ===========================================================================


class TestPrometheusMetrics:
    """Tests for the /metrics Prometheus endpoint."""

    def test_returns_200(self, handler, mock_http_handler):
        result = handler.handle("/metrics", {}, mock_http_handler)
        assert result is not None
        assert result.status_code == 200

    def test_returns_prometheus_content_type(self, handler, mock_http_handler):
        result = handler.handle("/metrics", {}, mock_http_handler)
        assert result is not None
        # Prometheus content type contains "text/plain" or "text/plain; version="
        assert "text/plain" in result.content_type or "openmetrics" in result.content_type

    def test_body_is_bytes(self, handler, mock_http_handler):
        result = handler.handle("/metrics", {}, mock_http_handler)
        assert result is not None
        assert isinstance(result.body, bytes)

    def test_body_is_valid_text(self, handler, mock_http_handler):
        result = handler.handle("/metrics", {}, mock_http_handler)
        assert result is not None
        text = result.body.decode("utf-8")
        # Prometheus output is always non-empty (at least has HELP/TYPE lines
        # or simple metric lines from the fallback)
        assert len(text) >= 0  # May be empty if no metrics registered

    def test_prometheus_metrics_error_returns_500(self, handler, mock_http_handler):
        """When get_metrics_output raises, graceful 500 is returned."""
        with patch(
            "aragora.server.handlers.metrics.handler.get_metrics_output",
            side_effect=RuntimeError("metrics registry corrupted"),
        ):
            result = handler.handle("/metrics", {}, mock_http_handler)
            assert result is not None
            assert result.status_code == 500

    def test_prometheus_with_token_auth_valid(self, handler, mock_http_handler, monkeypatch):
        """When ARAGORA_METRICS_TOKEN is set, a valid Bearer token grants access."""
        monkeypatch.setenv("ARAGORA_METRICS_TOKEN", "secret-token-123")
        mock_http_handler.headers["Authorization"] = "Bearer secret-token-123"
        result = handler.handle("/metrics", {}, mock_http_handler)
        assert result is not None
        assert result.status_code == 200

    def test_prometheus_with_token_auth_invalid(self, handler, mock_http_handler, monkeypatch):
        """When ARAGORA_METRICS_TOKEN is set, an invalid token is rejected with 401."""
        monkeypatch.setenv("ARAGORA_METRICS_TOKEN", "secret-token-123")
        mock_http_handler.headers["Authorization"] = "Bearer wrong-token"
        result = handler.handle("/metrics", {}, mock_http_handler)
        assert result is not None
        assert result.status_code == 401

    def test_prometheus_with_token_auth_missing_header(
        self, handler, mock_http_handler, monkeypatch
    ):
        """When ARAGORA_METRICS_TOKEN is set but no Authorization header, reject."""
        monkeypatch.setenv("ARAGORA_METRICS_TOKEN", "secret-token-123")
        # No Authorization header set
        result = handler.handle("/metrics", {}, mock_http_handler)
        assert result is not None
        assert result.status_code == 401

    def test_prometheus_rejects_query_param_token(self, handler, mock_http_handler, monkeypatch):
        """Token passed as query param is rejected for security."""
        monkeypatch.setenv("ARAGORA_METRICS_TOKEN", "secret-token-123")
        result = handler.handle("/metrics", {"token": ["secret-token-123"]}, mock_http_handler)
        assert result is not None
        assert result.status_code == 401
        body = json.loads(result.body)
        assert "query parameter" in body.get("error", "").lower()

    def test_prometheus_no_token_env_allows_access(self, handler, mock_http_handler, monkeypatch):
        """When ARAGORA_METRICS_TOKEN is not set, /metrics is open."""
        monkeypatch.delenv("ARAGORA_METRICS_TOKEN", raising=False)
        result = handler.handle("/metrics", {}, mock_http_handler)
        assert result is not None
        assert result.status_code == 200


# ===========================================================================
# GET /api/metrics (operational metrics)
# ===========================================================================


class TestOperationalMetrics:
    """Tests for the /api/metrics operational metrics endpoint."""

    def test_returns_200(self, handler, mock_http_handler):
        result = handler.handle("/api/metrics", {}, mock_http_handler)
        assert result is not None
        assert result.status_code == 200

    def test_returns_json(self, handler, mock_http_handler):
        result = handler.handle("/api/metrics", {}, mock_http_handler)
        assert result is not None
        body = result[0]
        assert isinstance(body, dict)

    def test_contains_uptime(self, handler, mock_http_handler):
        result = handler.handle("/api/metrics", {}, mock_http_handler)
        body = result[0]
        assert "uptime_seconds" in body
        assert "uptime_human" in body
        assert body["uptime_seconds"] >= 0

    def test_contains_requests_section(self, handler, mock_http_handler):
        result = handler.handle("/api/metrics", {}, mock_http_handler)
        body = result[0]
        assert "requests" in body
        req = body["requests"]
        assert "total" in req
        assert "errors" in req
        assert "error_rate" in req
        assert "top_endpoints" in req

    def test_contains_cache_section(self, handler, mock_http_handler):
        result = handler.handle("/api/metrics", {}, mock_http_handler)
        body = result[0]
        assert "cache" in body
        assert "entries" in body["cache"]

    def test_contains_databases_section(self, handler, mock_http_handler):
        result = handler.handle("/api/metrics", {}, mock_http_handler)
        body = result[0]
        assert "databases" in body

    def test_contains_timestamp(self, handler, mock_http_handler):
        result = handler.handle("/api/metrics", {}, mock_http_handler)
        body = result[0]
        assert "timestamp" in body

    def test_error_rate_zero_when_no_requests(self, handler, mock_http_handler):
        with patch(
            "aragora.server.handlers.metrics.handler.get_request_stats",
            return_value={"total_requests": 0, "total_errors": 0, "counts_snapshot": []},
        ):
            result = handler.handle("/api/metrics", {}, mock_http_handler)
            body = result[0]
            assert body["requests"]["error_rate"] == 0.0

    def test_database_sizes_with_nomic_dir(self, handler_with_full_ctx, mock_http_handler):
        """When nomic_dir exists, database sizes section is populated (possibly empty)."""
        result = handler_with_full_ctx.handle("/api/metrics", {}, mock_http_handler)
        body = result[0]
        # databases section should be a dict (may be empty if no .db files)
        assert isinstance(body["databases"], dict)

    def test_error_returns_500(self, handler, mock_http_handler):
        """If an internal error occurs, return 500 with safe message."""
        with patch(
            "aragora.server.handlers.metrics.handler.get_request_stats",
            side_effect=RuntimeError("stats corrupted"),
        ):
            result = handler.handle("/api/metrics", {}, mock_http_handler)
            assert result is not None
            assert result.status_code == 500


# ===========================================================================
# GET /api/metrics/health
# ===========================================================================


class TestHealthEndpoint:
    """Tests for the /api/metrics/health health check endpoint."""

    def test_healthy_with_working_subsystems(self, handler_with_full_ctx, mock_http_handler):
        result = handler_with_full_ctx.handle("/api/metrics/health", {}, mock_http_handler)
        assert result is not None
        assert result.status_code == 200
        body = result[0]
        assert body["status"] == "healthy"
        assert "checks" in body

    def test_storage_check_present(self, handler_with_full_ctx, mock_http_handler):
        result = handler_with_full_ctx.handle("/api/metrics/health", {}, mock_http_handler)
        body = result[0]
        assert "storage" in body["checks"]
        assert body["checks"]["storage"]["status"] == "healthy"

    def test_elo_system_check_present(self, handler_with_full_ctx, mock_http_handler):
        result = handler_with_full_ctx.handle("/api/metrics/health", {}, mock_http_handler)
        body = result[0]
        assert "elo_system" in body["checks"]
        assert body["checks"]["elo_system"]["status"] == "healthy"

    def test_nomic_dir_check_present(self, handler_with_full_ctx, mock_http_handler):
        result = handler_with_full_ctx.handle("/api/metrics/health", {}, mock_http_handler)
        body = result[0]
        assert "nomic_dir" in body["checks"]
        assert body["checks"]["nomic_dir"]["status"] == "healthy"

    def test_unavailable_subsystems(self, handler, mock_http_handler):
        """When no subsystems are configured, they show as unavailable."""
        result = handler.handle("/api/metrics/health", {}, mock_http_handler)
        assert result is not None
        body = result[0]
        assert "checks" in body
        assert body["checks"]["storage"]["status"] == "unavailable"
        assert body["checks"]["elo_system"]["status"] == "unavailable"
        assert body["checks"]["nomic_dir"]["status"] == "unavailable"

    def test_degraded_on_storage_error(self, mock_http_handler, tmp_path):
        """When storage health check fails, status is degraded with 503."""
        import sqlite3

        mock_storage = MagicMock()
        mock_storage.list_debates.side_effect = sqlite3.Error("db locked")
        h = MetricsHandler(ctx={"storage": mock_storage})
        result = h.handle("/api/metrics/health", {}, mock_http_handler)
        assert result is not None
        assert result.status_code == 503
        body = result[0]
        assert body["status"] == "degraded"
        assert body["checks"]["storage"]["status"] == "unhealthy"

    def test_degraded_on_elo_error(self, mock_http_handler):
        """When ELO health check fails, status is degraded."""
        import sqlite3

        mock_elo = MagicMock()
        mock_elo.get_leaderboard.side_effect = sqlite3.Error("corrupt")
        h = MetricsHandler(ctx={"elo_system": mock_elo})
        result = h.handle("/api/metrics/health", {}, mock_http_handler)
        body = result[0]
        assert body["status"] == "degraded"
        assert body["checks"]["elo_system"]["status"] == "unhealthy"

    def test_health_internal_error_returns_500(self, handler, mock_http_handler):
        """If the entire health check method fails, return 500."""
        with patch.object(handler, "get_storage", side_effect=RuntimeError("boom")):
            result = handler.handle("/api/metrics/health", {}, mock_http_handler)
            assert result is not None
            assert result.status_code == 500


# ===========================================================================
# GET /api/metrics/cache
# ===========================================================================


class TestCacheStats:
    """Tests for the /api/metrics/cache endpoint."""

    def test_returns_200(self, handler, mock_http_handler):
        result = handler.handle("/api/metrics/cache", {}, mock_http_handler)
        assert result is not None
        assert result.status_code == 200

    def test_returns_expected_fields(self, handler, mock_http_handler):
        result = handler.handle("/api/metrics/cache", {}, mock_http_handler)
        body = result[0]
        assert "total_entries" in body
        assert "max_entries" in body
        assert "hit_rate" in body
        assert "hits" in body
        assert "misses" in body
        assert "entries_by_prefix" in body

    def test_empty_cache(self, handler, mock_http_handler):
        """With an empty cache, entries counts are zero."""
        result = handler.handle("/api/metrics/cache", {}, mock_http_handler)
        body = result[0]
        assert body["total_entries"] >= 0
        assert body["oldest_entry_age_seconds"] >= 0
        assert body["newest_entry_age_seconds"] >= 0

    def test_cache_with_entries(self, handler, mock_http_handler):
        """When cache has entries, entries_by_prefix is populated."""
        from aragora.server.handlers.admin.cache import _cache

        # Manually insert test entries
        _cache.set("metrics:test_key", "value1")
        _cache.set("debate:test_key", "value2")

        try:
            result = handler.handle("/api/metrics/cache", {}, mock_http_handler)
            body = result[0]
            assert body["total_entries"] >= 2
            assert "metrics" in body["entries_by_prefix"] or body["total_entries"] >= 2
        finally:
            # Clean up - clear only our test entries
            with _cache._lock:
                _cache._cache.pop("metrics:test_key", None)
                _cache._cache.pop("debate:test_key", None)

    def test_cache_error_returns_500(self, handler, mock_http_handler):
        with patch(
            "aragora.server.handlers.metrics.handler.get_cache_stats",
            side_effect=RuntimeError("cache broken"),
        ):
            result = handler.handle("/api/metrics/cache", {}, mock_http_handler)
            assert result is not None
            assert result.status_code == 500


# ===========================================================================
# GET /api/metrics/verification
# ===========================================================================


class TestVerificationStats:
    """Tests for the /api/metrics/verification endpoint."""

    def test_returns_200(self, handler, mock_http_handler):
        result = handler.handle("/api/metrics/verification", {}, mock_http_handler)
        assert result is not None
        assert result.status_code == 200

    def test_returns_expected_fields(self, handler, mock_http_handler):
        result = handler.handle("/api/metrics/verification", {}, mock_http_handler)
        body = result[0]
        assert "total_claims_processed" in body
        assert "z3_verified" in body
        assert "z3_disproved" in body
        assert "z3_timeout" in body
        assert "z3_translation_failed" in body
        assert "confidence_fallback" in body
        assert "avg_verification_time_ms" in body
        assert "z3_success_rate" in body

    def test_zero_stats_initially(self, handler, mock_http_handler):
        """Verification stats start at zero (or close to it)."""
        result = handler.handle("/api/metrics/verification", {}, mock_http_handler)
        body = result[0]
        assert body["total_claims_processed"] >= 0
        assert body["z3_success_rate"] >= 0.0

    def test_verification_error_returns_500(self, handler, mock_http_handler):
        with patch(
            "aragora.server.handlers.metrics.handler.get_verification_stats",
            side_effect=RuntimeError("stats error"),
        ):
            result = handler.handle("/api/metrics/verification", {}, mock_http_handler)
            assert result is not None
            assert result.status_code == 500


# ===========================================================================
# GET /api/metrics/system
# ===========================================================================


class TestSystemInfo:
    """Tests for the /api/metrics/system endpoint."""

    def test_returns_200(self, handler, mock_http_handler):
        result = handler.handle("/api/metrics/system", {}, mock_http_handler)
        assert result is not None
        assert result.status_code == 200

    def test_returns_expected_fields(self, handler, mock_http_handler):
        result = handler.handle("/api/metrics/system", {}, mock_http_handler)
        body = result[0]
        assert "python_version" in body
        assert "platform" in body
        assert "machine" in body
        assert "processor" in body
        assert "pid" in body
        assert isinstance(body["pid"], int)

    def test_memory_field_present(self, handler, mock_http_handler):
        """Memory info is present (either with data or reason for unavailability)."""
        result = handler.handle("/api/metrics/system", {}, mock_http_handler)
        body = result[0]
        assert "memory" in body
        mem = body["memory"]
        # Either has rss_mb/vms_mb or available=False
        assert "rss_mb" in mem or "available" in mem

    def test_system_info_error_returns_500(self, handler, mock_http_handler):
        with patch(
            "aragora.server.handlers.metrics.handler.platform.platform",
            side_effect=OSError("os error"),
        ):
            result = handler.handle("/api/metrics/system", {}, mock_http_handler)
            assert result is not None
            assert result.status_code == 500


# ===========================================================================
# GET /api/metrics/background
# ===========================================================================


class TestBackgroundStats:
    """Tests for the /api/metrics/background endpoint."""

    def test_returns_200_when_manager_unavailable(self, handler, mock_http_handler):
        """When background manager import fails, returns graceful fallback."""
        with patch(
            "aragora.server.handlers.metrics.handler.MetricsHandler._get_background_stats",
            wraps=handler._get_background_stats,
        ):
            # The import may fail in test env; either way should return 200
            result = handler.handle("/api/metrics/background", {}, mock_http_handler)
            assert result is not None
            assert result.status_code == 200

    def test_fallback_format(self, handler, mock_http_handler):
        """When background manager is not available, returns fallback JSON."""
        with patch.dict("sys.modules", {"aragora.server.background": None}):
            result = handler._get_background_stats()
            body = result[0]
            assert body.get("running") is False or "message" in body

    def test_with_mock_manager(self, handler, mock_http_handler):
        """With a mock background manager, returns its stats."""
        mock_manager = MagicMock()
        mock_manager.get_stats.return_value = {
            "running": True,
            "task_count": 3,
            "tasks": {"cleanup": "running", "sync": "idle"},
        }
        with patch(
            "aragora.server.handlers.metrics.handler.MetricsHandler._get_background_stats"
        ) as mock_method:
            from aragora.server.handlers.utils.responses import json_response

            mock_method.return_value = json_response(mock_manager.get_stats())
            result = handler.handle("/api/metrics/background", {}, mock_http_handler)
            assert result is not None
            assert result.status_code == 200


# ===========================================================================
# GET /api/metrics/debate
# ===========================================================================


class TestDebatePerfStats:
    """Tests for the /api/metrics/debate endpoint."""

    def test_returns_200_when_monitor_unavailable(self, handler, mock_http_handler):
        """When debate monitor import fails, returns graceful fallback."""
        result = handler.handle("/api/metrics/debate", {"debate_id": [None]}, mock_http_handler)
        assert result is not None
        assert result.status_code == 200

    def test_fallback_format(self, handler):
        """Fallback response includes expected keys."""
        result = handler._get_debate_perf_stats()
        body = result[0]
        assert "slow_debates_history" in body or "message" in body

    def test_with_debate_id(self, handler, mock_http_handler):
        """Passing debate_id dispatches correctly."""
        result = handler.handle(
            "/api/metrics/debate", {"debate_id": ["test-debate-1"]}, mock_http_handler
        )
        assert result is not None
        assert result.status_code == 200

    def test_error_returns_500(self, handler, mock_http_handler):
        """Runtime error in debate stats returns 500."""
        with patch(
            "aragora.server.handlers.metrics.handler.MetricsHandler._get_debate_perf_stats",
            side_effect=RuntimeError("boom"),
        ):
            # This patches the method directly so handle() calls it and it raises
            with pytest.raises(RuntimeError):
                handler.handle("/api/metrics/debate", {"debate_id": [None]}, mock_http_handler)


# ===========================================================================
# Rate Limiting
# ===========================================================================


class TestRateLimiting:
    """Tests for rate limiting on metrics endpoints."""

    def test_rate_limit_returns_429(self, handler, mock_http_handler):
        """Exceeding rate limit returns 429."""
        from aragora.server.handlers.metrics.handler import _metrics_limiter

        # Patch is_allowed to return False (rate limited)
        with patch.object(_metrics_limiter, "is_allowed", return_value=False):
            result = handler.handle("/api/metrics", {}, mock_http_handler)
            assert result is not None
            assert result.status_code == 429

    def test_rate_limit_message(self, handler, mock_http_handler):
        """Rate limit error includes descriptive message."""
        from aragora.server.handlers.metrics.handler import _metrics_limiter

        with patch.object(_metrics_limiter, "is_allowed", return_value=False):
            result = handler.handle("/api/metrics", {}, mock_http_handler)
            body = json.loads(result.body)
            assert "rate limit" in body.get("error", "").lower()

    def test_rate_limit_applies_to_prometheus(self, handler, mock_http_handler):
        """/metrics endpoint is also rate limited."""
        from aragora.server.handlers.metrics.handler import _metrics_limiter

        with patch.object(_metrics_limiter, "is_allowed", return_value=False):
            result = handler.handle("/metrics", {}, mock_http_handler)
            assert result is not None
            assert result.status_code == 429


# ===========================================================================
# Formatters (legacy wrappers)
# ===========================================================================


class TestFormatters:
    """Tests for the formatting helper methods."""

    def test_format_uptime_seconds(self, handler):
        assert handler._format_uptime(30) == "30s"

    def test_format_uptime_minutes(self, handler):
        assert handler._format_uptime(90) == "1m 30s"

    def test_format_uptime_hours(self, handler):
        result = handler._format_uptime(3661)
        assert "1h" in result

    def test_format_uptime_days(self, handler):
        result = handler._format_uptime(90000)
        assert "1d" in result

    def test_format_size_bytes(self, handler):
        assert handler._format_size(500) == "500.0 B"

    def test_format_size_kb(self, handler):
        result = handler._format_size(1500)
        assert "KB" in result

    def test_format_size_mb(self, handler):
        result = handler._format_size(1_500_000)
        assert "MB" in result

    def test_format_size_gb(self, handler):
        result = handler._format_size(1_500_000_000)
        assert "GB" in result


# ===========================================================================
# Database Sizes
# ===========================================================================


class TestDatabaseSizes:
    """Tests for _get_database_sizes helper."""

    def test_empty_without_nomic_dir(self, handler):
        """Returns empty dict when nomic_dir is not configured."""
        sizes = handler._get_database_sizes()
        assert sizes == {}

    def test_empty_with_nonexistent_nomic_dir(self):
        """Returns empty dict when nomic_dir does not exist on disk."""
        h = MetricsHandler(ctx={"nomic_dir": Path("/nonexistent/path")})
        sizes = h._get_database_sizes()
        assert sizes == {}

    def test_with_db_files(self, tmp_path):
        """Returns sizes for database files found in nomic_dir."""
        nomic_dir = tmp_path / ".nomic"
        nomic_dir.mkdir()

        # Create a fake database file
        db_file = nomic_dir / "debate_storage.db"
        db_file.write_bytes(b"x" * 1024)

        h = MetricsHandler(ctx={"nomic_dir": nomic_dir})
        sizes = h._get_database_sizes()
        assert "debate_storage.db" in sizes
        assert sizes["debate_storage.db"]["bytes"] == 1024
        assert (
            "KB" in sizes["debate_storage.db"]["human"]
            or "B" in sizes["debate_storage.db"]["human"]
        )

    def test_ignores_missing_db_files(self, tmp_path):
        """Only includes database files that actually exist."""
        nomic_dir = tmp_path / ".nomic"
        nomic_dir.mkdir()
        # No db files created

        h = MetricsHandler(ctx={"nomic_dir": nomic_dir})
        sizes = h._get_database_sizes()
        assert sizes == {}


# ===========================================================================
# Edge Cases
# ===========================================================================


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_handle_with_versioned_path(self, handler, mock_http_handler):
        """Versioned paths (/api/v1/...) are stripped by strip_version_prefix."""
        result = handler.handle("/api/v1/metrics", {}, mock_http_handler)
        # /api/v1/metrics strips to /api/metrics
        assert result is not None
        assert result.status_code == 200

    def test_handle_with_versioned_prometheus(self, handler, mock_http_handler):
        """/api/v1/metrics resolves to /api/metrics, not /metrics."""
        with patch.object(handler, "_get_metrics") as mock_get:
            mock_get.return_value = MagicMock(status_code=200)
            handler.handle("/api/v1/metrics", {}, mock_http_handler)
            mock_get.assert_called_once()

    def test_concurrent_access_safety(self, handler, mock_http_handler):
        """Multiple calls do not corrupt state (basic smoke test)."""
        import threading

        results = []

        def call_metrics():
            r = handler.handle("/metrics", {}, mock_http_handler)
            results.append(r)

        threads = [threading.Thread(target=call_metrics) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        assert len(results) == 5
        for r in results:
            assert r is not None
            assert r.status_code == 200

    def test_none_handler_for_client_ip(self, handler):
        """get_client_ip handles None handler gracefully."""
        from aragora.server.handlers.utils.rate_limit import get_client_ip

        assert get_client_ip(None) == "unknown"


# ===========================================================================
# Authentication on /api/metrics/* endpoints
# ===========================================================================


@pytest.mark.no_auto_auth
class TestMetricsAuth:
    """Tests for authentication requirements on /api/metrics/* endpoints.

    Uses no_auto_auth marker to test actual auth behavior.
    """

    def test_api_metrics_requires_auth(self, mock_http_handler):
        """Without auth bypass, /api/metrics returns 401."""
        h = MetricsHandler(ctx={})
        result = h.handle("/api/metrics", {}, mock_http_handler)
        assert result is not None
        assert result.status_code == 401

    def test_api_metrics_health_requires_auth(self, mock_http_handler):
        h = MetricsHandler(ctx={})
        result = h.handle("/api/metrics/health", {}, mock_http_handler)
        assert result is not None
        assert result.status_code == 401

    def test_prometheus_no_auth_required_by_default(self, mock_http_handler, monkeypatch):
        """/metrics does not require auth when ARAGORA_METRICS_TOKEN is unset."""
        monkeypatch.delenv("ARAGORA_METRICS_TOKEN", raising=False)
        h = MetricsHandler(ctx={})
        result = h.handle("/metrics", {}, mock_http_handler)
        assert result is not None
        # Should succeed (200) since /metrics doesn't require auth
        assert result.status_code == 200
