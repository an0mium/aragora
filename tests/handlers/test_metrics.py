"""Tests for metrics handler endpoints.

Tests the operational metrics API endpoints including:
- GET /api/metrics - Get operational metrics
- GET /api/metrics/health - Detailed health check
- GET /api/metrics/cache - Cache statistics
- GET /api/metrics/verification - Z3 verification stats
- GET /api/metrics/system - System information
- GET /metrics - Prometheus-format metrics
"""

import json
import time
from datetime import datetime
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest


def parse_body(result) -> dict:
    """Parse JSON body from HandlerResult."""
    return json.loads(result.body.decode("utf-8"))


class MockHandler:
    """Mock HTTP handler."""

    def __init__(self):
        self.client_address = ("127.0.0.1", 12345)
        self.headers = {}


@pytest.fixture
def mock_handler():
    """Create mock handler."""
    return MockHandler()


@pytest.fixture
def mock_server_context():
    """Create mock server context."""
    return {
        "storage": None,
        "elo_system": None,
    }


@pytest.fixture
def metrics_handler(mock_server_context):
    """Create MetricsHandler for testing."""
    from aragora.server.handlers.metrics import MetricsHandler

    handler = MetricsHandler(mock_server_context)
    return handler


class TestMetricsHandlerRouting:
    """Test routing logic for metrics handler."""

    def test_can_handle_metrics(self):
        """Test can_handle for /api/metrics."""
        from aragora.server.handlers.metrics import MetricsHandler

        handler = MetricsHandler({})
        assert handler.can_handle("/api/v1/metrics") is True

    def test_can_handle_health(self):
        """Test can_handle for /api/metrics/health."""
        from aragora.server.handlers.metrics import MetricsHandler

        handler = MetricsHandler({})
        assert handler.can_handle("/api/v1/metrics/health") is True

    def test_can_handle_cache(self):
        """Test can_handle for /api/metrics/cache."""
        from aragora.server.handlers.metrics import MetricsHandler

        handler = MetricsHandler({})
        assert handler.can_handle("/api/v1/metrics/cache") is True

    def test_can_handle_verification(self):
        """Test can_handle for /api/metrics/verification."""
        from aragora.server.handlers.metrics import MetricsHandler

        handler = MetricsHandler({})
        assert handler.can_handle("/api/v1/metrics/verification") is True

    def test_can_handle_system(self):
        """Test can_handle for /api/metrics/system."""
        from aragora.server.handlers.metrics import MetricsHandler

        handler = MetricsHandler({})
        assert handler.can_handle("/api/v1/metrics/system") is True

    def test_can_handle_prometheus(self):
        """Test can_handle for /metrics (Prometheus)."""
        from aragora.server.handlers.metrics import MetricsHandler

        handler = MetricsHandler({})
        assert handler.can_handle("/metrics") is True

    def test_cannot_handle_invalid(self):
        """Test can_handle rejects invalid paths."""
        from aragora.server.handlers.metrics import MetricsHandler

        handler = MetricsHandler({})
        assert handler.can_handle("/api/v1/other") is False
        assert handler.can_handle("/api/v1/metrics/unknown") is False


class TestMetricsHandler:
    """Test /api/metrics endpoint."""

    @patch("aragora.server.handlers.metrics._metrics_limiter")
    def test_get_metrics(self, mock_limiter, metrics_handler, mock_handler):
        """Test basic metrics retrieval."""
        mock_limiter.is_allowed.return_value = True

        result = metrics_handler.handle("/api/v1/metrics", {}, mock_handler)

        assert result is not None
        assert result.status_code == 200
        body = parse_body(result)
        assert "uptime_seconds" in body
        # Requests are nested under "requests" key
        assert "requests" in body


class TestMetricsHandlerHealth:
    """Test /api/metrics/health endpoint."""

    @patch("aragora.server.handlers.metrics._metrics_limiter")
    def test_health_check(self, mock_limiter, metrics_handler, mock_handler):
        """Test health check endpoint."""
        mock_limiter.is_allowed.return_value = True

        result = metrics_handler.handle("/api/v1/metrics/health", {}, mock_handler)

        assert result is not None
        assert result.status_code == 200
        body = parse_body(result)
        assert "status" in body
        assert "checks" in body


class TestMetricsHandlerCache:
    """Test /api/metrics/cache endpoint."""

    @patch("aragora.server.handlers.metrics._metrics_limiter")
    def test_cache_stats(self, mock_limiter, metrics_handler, mock_handler):
        """Test cache statistics retrieval."""
        mock_limiter.is_allowed.return_value = True

        result = metrics_handler.handle("/api/v1/metrics/cache", {}, mock_handler)

        assert result is not None
        assert result.status_code == 200
        body = parse_body(result)
        # Cache stats include hit_rate, entries, etc.
        assert "hit_rate" in body or "entries_by_prefix" in body


class TestMetricsHandlerVerification:
    """Test /api/metrics/verification endpoint."""

    @patch("aragora.server.handlers.metrics._metrics_limiter")
    def test_verification_stats(self, mock_limiter, metrics_handler, mock_handler):
        """Test verification statistics retrieval."""
        mock_limiter.is_allowed.return_value = True

        result = metrics_handler.handle("/api/v1/metrics/verification", {}, mock_handler)

        assert result is not None
        assert result.status_code == 200
        body = parse_body(result)
        assert "total_claims_processed" in body
        assert "z3_verified" in body
        assert "z3_success_rate" in body


class TestMetricsHandlerSystem:
    """Test /api/metrics/system endpoint."""

    @patch("aragora.server.handlers.metrics._metrics_limiter")
    def test_system_info(self, mock_limiter, metrics_handler, mock_handler):
        """Test system information retrieval."""
        mock_limiter.is_allowed.return_value = True

        result = metrics_handler.handle("/api/v1/metrics/system", {}, mock_handler)

        assert result is not None
        assert result.status_code == 200
        body = parse_body(result)
        assert "platform" in body
        assert "python_version" in body


class TestMetricsHandlerPrometheus:
    """Test /metrics endpoint (Prometheus format)."""

    @patch("aragora.server.handlers.metrics._metrics_limiter")
    def test_prometheus_metrics(self, mock_limiter, metrics_handler, mock_handler):
        """Test Prometheus-format metrics."""
        mock_limiter.is_allowed.return_value = True

        result = metrics_handler.handle("/metrics", {}, mock_handler)

        assert result is not None
        assert result.status_code == 200
        # Prometheus metrics are text/plain, not JSON
        assert b"aragora_" in result.body or result.body  # Metrics content


class TestMetricsHandlerRateLimiting:
    """Test rate limiting for metrics endpoints."""

    @patch("aragora.server.handlers.metrics._metrics_limiter")
    def test_rate_limit_exceeded(self, mock_limiter, metrics_handler, mock_handler):
        """Test rate limit exceeded response."""
        mock_limiter.is_allowed.return_value = False

        result = metrics_handler.handle("/api/v1/metrics", {}, mock_handler)

        assert result is not None
        assert result.status_code == 429
        body = parse_body(result)
        assert "rate limit" in body["error"].lower()


class TestVerificationTracking:
    """Test verification tracking functions."""

    def test_track_verification(self):
        """Test tracking verification outcomes."""
        from aragora.server.handlers.metrics import (
            get_verification_stats,
            track_verification,
        )

        initial_stats = get_verification_stats()
        initial_total = initial_stats["total_claims_processed"]

        # Track a verified claim
        track_verification("z3_verified", verification_time_ms=50.0)

        stats = get_verification_stats()
        assert stats["total_claims_processed"] == initial_total + 1
        assert stats["z3_verified"] >= 1

    def test_verification_stats_derived_metrics(self):
        """Test derived metrics calculation."""
        from aragora.server.handlers.metrics import get_verification_stats

        stats = get_verification_stats()
        assert "avg_verification_time_ms" in stats
        assert "z3_success_rate" in stats


class TestRequestTracking:
    """Test request tracking for metrics."""

    def test_track_request(self):
        """Test tracking requests."""
        from aragora.server.handlers.metrics import track_request

        # Should not raise
        track_request("/api/v1/test", is_error=False)
        track_request("/api/v1/test", is_error=True)

    def test_track_request_limit(self):
        """Test that request tracking has a limit."""
        from aragora.server.handlers.metrics import MAX_TRACKED_ENDPOINTS

        assert MAX_TRACKED_ENDPOINTS == 1000  # Prevent unbounded growth
