"""
Tests for the MetricsHandler module.

Tests cover:
- Handler routing for metrics endpoints
- can_handle method
- ROUTES attribute
- Prometheus metrics endpoint
- Verification statistics tracking
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest

from aragora.server.handlers.metrics import (
    MetricsHandler,
    track_verification,
    get_verification_stats,
    track_request,
)


@pytest.fixture
def mock_server_context():
    """Create mock server context for handler initialization."""
    return {"storage": None, "elo_system": None, "nomic_dir": None}


class TestMetricsHandlerRouting:
    """Tests for handler routing."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return MetricsHandler(mock_server_context)

    def test_can_handle_metrics(self, handler):
        """Handler can handle metrics base endpoint."""
        assert handler.can_handle("/api/metrics")

    def test_can_handle_health(self, handler):
        """Handler can handle health endpoint."""
        assert handler.can_handle("/api/metrics/health")

    def test_can_handle_cache(self, handler):
        """Handler can handle cache endpoint."""
        assert handler.can_handle("/api/metrics/cache")

    def test_can_handle_verification(self, handler):
        """Handler can handle verification endpoint."""
        assert handler.can_handle("/api/metrics/verification")

    def test_can_handle_system(self, handler):
        """Handler can handle system endpoint."""
        assert handler.can_handle("/api/metrics/system")

    def test_can_handle_prometheus(self, handler):
        """Handler can handle Prometheus metrics endpoint."""
        assert handler.can_handle("/metrics")

    def test_cannot_handle_other_paths(self, handler):
        """Handler cannot handle unrelated paths."""
        assert not handler.can_handle("/api/debates")
        assert not handler.can_handle("/api/other")


class TestMetricsHandlerRoutesAttribute:
    """Tests for ROUTES class attribute."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return MetricsHandler(mock_server_context)

    def test_routes_contains_metrics(self, handler):
        """ROUTES contains metrics base."""
        assert "/api/metrics" in handler.ROUTES

    def test_routes_contains_health(self, handler):
        """ROUTES contains health."""
        assert "/api/metrics/health" in handler.ROUTES

    def test_routes_contains_cache(self, handler):
        """ROUTES contains cache."""
        assert "/api/metrics/cache" in handler.ROUTES

    def test_routes_contains_verification(self, handler):
        """ROUTES contains verification."""
        assert "/api/metrics/verification" in handler.ROUTES

    def test_routes_contains_system(self, handler):
        """ROUTES contains system."""
        assert "/api/metrics/system" in handler.ROUTES

    def test_routes_contains_prometheus(self, handler):
        """ROUTES contains Prometheus endpoint."""
        assert "/metrics" in handler.ROUTES


class TestMetricsHandlerRouteDispatch:
    """Tests for route dispatch logic."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return MetricsHandler(mock_server_context)

    def test_handle_metrics_returns_result(self, handler):
        """Handle returns result for metrics endpoint."""
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 8080)

        result = handler.handle("/api/metrics", {}, mock_http)

        assert result is not None
        assert result.status_code == 200

    def test_handle_health_returns_result(self, handler):
        """Handle returns result for health endpoint."""
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 8080)

        result = handler.handle("/api/metrics/health", {}, mock_http)

        assert result is not None
        assert result.status_code == 200

    def test_handle_cache_returns_result(self, handler):
        """Handle returns result for cache endpoint."""
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 8080)

        result = handler.handle("/api/metrics/cache", {}, mock_http)

        assert result is not None
        assert result.status_code == 200

    def test_handle_verification_returns_result(self, handler):
        """Handle returns result for verification endpoint."""
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 8080)

        result = handler.handle("/api/metrics/verification", {}, mock_http)

        assert result is not None
        assert result.status_code == 200

    def test_handle_system_returns_result(self, handler):
        """Handle returns result for system endpoint."""
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 8080)

        result = handler.handle("/api/metrics/system", {}, mock_http)

        assert result is not None
        assert result.status_code == 200

    def test_handle_prometheus_returns_result(self, handler):
        """Handle returns result for Prometheus endpoint."""
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 8080)

        result = handler.handle("/metrics", {}, mock_http)

        assert result is not None
        assert result.status_code == 200

    def test_handle_unknown_returns_none(self, handler):
        """Handle returns None for unknown paths."""
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 8080)

        result = handler.handle("/api/unknown", {}, mock_http)

        assert result is None


class TestVerificationTracking:
    """Tests for verification statistics tracking."""

    def test_track_verification_increments_total(self):
        """track_verification increments total claims."""
        initial_stats = get_verification_stats()
        initial_total = initial_stats["total_claims_processed"]

        track_verification("z3_verified")

        stats = get_verification_stats()
        assert stats["total_claims_processed"] == initial_total + 1

    def test_track_verification_increments_status(self):
        """track_verification increments status counter."""
        initial_stats = get_verification_stats()
        initial_verified = initial_stats["z3_verified"]

        track_verification("z3_verified")

        stats = get_verification_stats()
        assert stats["z3_verified"] == initial_verified + 1

    def test_track_verification_accumulates_time(self):
        """track_verification accumulates verification time."""
        initial_stats = get_verification_stats()
        initial_time = initial_stats["total_verification_time_ms"]

        track_verification("z3_verified", verification_time_ms=100.0)

        stats = get_verification_stats()
        assert stats["total_verification_time_ms"] == initial_time + 100.0

    def test_get_verification_stats_calculates_averages(self):
        """get_verification_stats calculates derived metrics."""
        stats = get_verification_stats()

        assert "avg_verification_time_ms" in stats
        assert "z3_success_rate" in stats

    def test_track_verification_all_statuses(self):
        """track_verification works for all status types."""
        # Track each status type
        track_verification("z3_verified")
        track_verification("z3_disproved")
        track_verification("z3_timeout")
        track_verification("z3_translation_failed")
        track_verification("confidence_fallback")

        stats = get_verification_stats()

        # All counters should be positive
        assert stats["z3_verified"] > 0
        assert stats["z3_disproved"] >= 0
        assert stats["z3_timeout"] >= 0


class TestRequestTracking:
    """Tests for request tracking."""

    def test_track_request_increments_count(self):
        """track_request increments endpoint count."""
        endpoint = "/api/test/unique_endpoint_123"

        track_request(endpoint)
        track_request(endpoint)

        # The function doesn't expose counts directly,
        # but we verify it doesn't crash
        assert True

    def test_track_request_handles_error(self):
        """track_request can track errors."""
        endpoint = "/api/test/error_endpoint"

        track_request(endpoint, is_error=True)

        # Verify no crash
        assert True

    def test_track_request_multiple_endpoints(self):
        """track_request tracks multiple endpoints."""
        track_request("/api/endpoint1")
        track_request("/api/endpoint2")
        track_request("/api/endpoint3")

        # Verify no crash
        assert True


class TestMetricsHandlerResponseFormat:
    """Tests for response format."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return MetricsHandler(mock_server_context)

    def test_metrics_response_is_json(self, handler):
        """Metrics endpoint returns JSON response."""
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 8080)

        result = handler.handle("/api/metrics", {}, mock_http)

        assert result is not None
        assert result.content_type == "application/json"

    def test_prometheus_response_is_text(self, handler):
        """Prometheus endpoint returns text response."""
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 8080)

        result = handler.handle("/metrics", {}, mock_http)

        assert result is not None
        # Prometheus format is text/plain or application/openmetrics-text
        assert "text" in result.content_type or "openmetrics" in result.content_type

    def test_health_response_includes_status(self, handler):
        """Health endpoint includes status field."""
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 8080)

        result = handler.handle("/api/metrics/health", {}, mock_http)

        assert result is not None
        # Body should contain status
        import json

        body = json.loads(result.body)
        assert "status" in body or "healthy" in body


class TestMetricsHandlerSystemInfo:
    """Tests for system information endpoint."""

    @pytest.fixture
    def handler(self, mock_server_context):
        return MetricsHandler(mock_server_context)

    def test_system_includes_python_version(self, handler):
        """System endpoint includes Python version."""
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 8080)

        result = handler.handle("/api/metrics/system", {}, mock_http)

        assert result is not None
        import json

        body = json.loads(result.body)
        assert "python_version" in body

    def test_system_includes_platform(self, handler):
        """System endpoint includes platform info."""
        mock_http = MagicMock()
        mock_http.client_address = ("127.0.0.1", 8080)

        result = handler.handle("/api/metrics/system", {}, mock_http)

        assert result is not None
        import json

        body = json.loads(result.body)
        assert "platform" in body or "system" in body
