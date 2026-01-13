"""
Tests for MetricsHandler.

Tests operational metrics, health checks, cache stats, system info,
and Prometheus-format metrics endpoints.
"""

import json
import time
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from aragora.server.handlers.metrics import (
    MetricsHandler,
    track_request,
    _request_counts,
    _error_counts,
    _start_time,
)
from aragora.server.handlers.base import clear_cache, _cache


@pytest.fixture
def handler(tmp_path):
    """Create MetricsHandler with mock context."""
    ctx = {
        "storage": Mock(),
        "elo_system": Mock(),
        "nomic_dir": tmp_path,
    }
    return MetricsHandler(ctx)


@pytest.fixture
def handler_no_components():
    """Create MetricsHandler without storage/elo."""
    ctx = {
        "storage": None,
        "elo_system": None,
        "nomic_dir": None,
    }
    return MetricsHandler(ctx)


@pytest.fixture(autouse=True)
def reset_metrics():
    """Reset metrics state between tests."""
    _request_counts.clear()
    _error_counts.clear()
    clear_cache()
    yield
    _request_counts.clear()
    _error_counts.clear()
    clear_cache()


class TestMetricsHandlerRouting:
    """Test route matching and dispatch."""

    def test_can_handle_metrics_endpoint(self, handler):
        """can_handle returns True for /api/metrics."""
        assert handler.can_handle("/api/metrics") is True

    def test_can_handle_health_endpoint(self, handler):
        """can_handle returns True for /api/metrics/health."""
        assert handler.can_handle("/api/metrics/health") is True

    def test_can_handle_cache_endpoint(self, handler):
        """can_handle returns True for /api/metrics/cache."""
        assert handler.can_handle("/api/metrics/cache") is True

    def test_can_handle_system_endpoint(self, handler):
        """can_handle returns True for /api/metrics/system."""
        assert handler.can_handle("/api/metrics/system") is True

    def test_can_handle_prometheus_endpoint(self, handler):
        """can_handle returns True for /metrics."""
        assert handler.can_handle("/metrics") is True

    def test_cannot_handle_unknown_paths(self, handler):
        """can_handle returns False for unknown paths."""
        assert handler.can_handle("/api/other") is False
        assert handler.can_handle("/api/metrics/unknown") is False
        assert handler.can_handle("/api/leaderboard") is False

    def test_handle_returns_none_for_unknown(self, handler):
        """handle returns None for paths it doesn't handle."""
        result = handler.handle("/api/unknown", {}, None)
        assert result is None


class TestMetricsEndpoint:
    """Test /api/metrics endpoint."""

    def test_returns_200_success(self, handler):
        """Returns 200 status for successful request."""
        result = handler.handle("/api/metrics", {}, None)
        assert result.status_code == 200
        assert result.content_type == "application/json"

    def test_returns_uptime(self, handler):
        """Returns uptime in seconds and human-readable format."""
        result = handler.handle("/api/metrics", {}, None)
        data = json.loads(result.body)

        assert "uptime_seconds" in data
        assert "uptime_human" in data
        assert isinstance(data["uptime_seconds"], (int, float))
        assert isinstance(data["uptime_human"], str)

    def test_returns_request_stats(self, handler):
        """Returns request statistics."""
        # Track some requests first
        track_request("/api/test")
        track_request("/api/test")
        track_request("/api/other", is_error=True)

        result = handler.handle("/api/metrics", {}, None)
        data = json.loads(result.body)

        assert "requests" in data
        assert data["requests"]["total"] == 3
        assert data["requests"]["errors"] == 1
        assert data["requests"]["error_rate"] > 0

    def test_returns_zero_requests(self, handler):
        """Handles zero requests gracefully."""
        result = handler.handle("/api/metrics", {}, None)
        data = json.loads(result.body)

        assert data["requests"]["total"] == 0
        assert data["requests"]["errors"] == 0
        assert data["requests"]["error_rate"] == 0.0

    def test_returns_top_endpoints(self, handler):
        """Returns top endpoints by request count."""
        track_request("/api/popular")
        track_request("/api/popular")
        track_request("/api/popular")
        track_request("/api/other")

        result = handler.handle("/api/metrics", {}, None)
        data = json.loads(result.body)

        top = data["requests"]["top_endpoints"]
        assert len(top) > 0
        assert top[0]["endpoint"] == "/api/popular"
        assert top[0]["count"] == 3

    def test_returns_cache_info(self, handler):
        """Returns cache entry count."""
        result = handler.handle("/api/metrics", {}, None)
        data = json.loads(result.body)

        assert "cache" in data
        assert "entries" in data["cache"]

    def test_returns_database_sizes(self, handler, tmp_path):
        """Returns database file sizes when they exist."""
        # Create a test database file - must match DB_ELO_PATH ("agent_elo.db")
        db_path = tmp_path / "agent_elo.db"
        db_path.write_bytes(b"x" * 1024)

        result = handler.handle("/api/metrics", {}, None)
        data = json.loads(result.body)

        assert "databases" in data
        assert "agent_elo.db" in data["databases"]
        assert data["databases"]["agent_elo.db"]["bytes"] == 1024

    def test_returns_timestamp(self, handler):
        """Returns current timestamp."""
        result = handler.handle("/api/metrics", {}, None)
        data = json.loads(result.body)

        assert "timestamp" in data
        assert "T" in data["timestamp"]  # ISO format


class TestHealthEndpoint:
    """Test /api/metrics/health endpoint."""

    def test_returns_healthy_when_all_ok(self, handler):
        """Returns healthy status when all components work."""
        handler.ctx["storage"].list_debates.return_value = []
        handler.ctx["elo_system"].get_leaderboard.return_value = []

        result = handler.handle("/api/metrics/health", {}, None)
        data = json.loads(result.body)

        assert result.status_code == 200
        assert data["status"] == "healthy"
        assert data["checks"]["storage"]["status"] == "healthy"
        assert data["checks"]["elo_system"]["status"] == "healthy"

    def test_returns_degraded_when_storage_fails(self, handler):
        """Returns degraded status when storage check fails."""
        handler.ctx["storage"].list_debates.side_effect = Exception("DB error")
        handler.ctx["elo_system"].get_leaderboard.return_value = []

        result = handler.handle("/api/metrics/health", {}, None)
        data = json.loads(result.body)

        assert result.status_code == 503
        assert data["status"] == "degraded"
        assert data["checks"]["storage"]["status"] == "unhealthy"

    def test_returns_degraded_when_elo_fails(self, handler):
        """Returns degraded status when ELO check fails."""
        handler.ctx["storage"].list_debates.return_value = []
        handler.ctx["elo_system"].get_leaderboard.side_effect = Exception("ELO error")

        result = handler.handle("/api/metrics/health", {}, None)
        data = json.loads(result.body)

        assert result.status_code == 503
        assert data["status"] == "degraded"
        assert data["checks"]["elo_system"]["status"] == "unhealthy"

    def test_handles_unavailable_components(self, handler_no_components):
        """Handles missing storage/elo gracefully."""
        result = handler_no_components.handle("/api/metrics/health", {}, None)
        data = json.loads(result.body)

        assert result.status_code == 200
        assert data["checks"]["storage"]["status"] == "unavailable"
        assert data["checks"]["elo_system"]["status"] == "unavailable"
        assert data["checks"]["nomic_dir"]["status"] == "unavailable"

    def test_checks_nomic_dir(self, handler, tmp_path):
        """Reports nomic_dir status correctly."""
        handler.ctx["storage"].list_debates.return_value = []
        handler.ctx["elo_system"].get_leaderboard.return_value = []

        result = handler.handle("/api/metrics/health", {}, None)
        data = json.loads(result.body)

        assert data["checks"]["nomic_dir"]["status"] == "healthy"
        assert data["checks"]["nomic_dir"]["path"] == str(tmp_path)


class TestCacheEndpoint:
    """Test /api/metrics/cache endpoint."""

    def test_returns_200_success(self, handler):
        """Returns 200 for cache stats."""
        result = handler.handle("/api/metrics/cache", {}, None)
        assert result.status_code == 200
        assert result.content_type == "application/json"

    def test_returns_cache_stats_structure(self, handler):
        """Returns expected cache stats fields."""
        result = handler.handle("/api/metrics/cache", {}, None)
        data = json.loads(result.body)

        assert "total_entries" in data
        assert "max_entries" in data
        assert "hit_rate" in data
        assert "hits" in data
        assert "misses" in data

    def test_returns_empty_cache_stats(self, handler):
        """Handles empty cache gracefully."""
        result = handler.handle("/api/metrics/cache", {}, None)
        data = json.loads(result.body)

        assert data["total_entries"] == 0
        assert data["oldest_entry_age_seconds"] == 0
        assert data["newest_entry_age_seconds"] == 0

    def test_returns_entries_by_prefix(self, handler):
        """Groups cache entries by prefix."""
        # Add some cache entries using the cache's set method
        _cache.set("prefix1:key1", "value1")
        _cache.set("prefix1:key2", "value2")
        _cache.set("prefix2:key1", "value3")

        result = handler.handle("/api/metrics/cache", {}, None)
        data = json.loads(result.body)

        assert "entries_by_prefix" in data
        assert data["entries_by_prefix"].get("prefix1") == 2
        assert data["entries_by_prefix"].get("prefix2") == 1


class TestSystemEndpoint:
    """Test /api/metrics/system endpoint."""

    def test_returns_200_success(self, handler):
        """Returns 200 for system info."""
        result = handler.handle("/api/metrics/system", {}, None)
        assert result.status_code == 200
        assert result.content_type == "application/json"

    def test_returns_system_info(self, handler):
        """Returns expected system info fields."""
        result = handler.handle("/api/metrics/system", {}, None)
        data = json.loads(result.body)

        assert "python_version" in data
        assert "platform" in data
        assert "machine" in data
        assert "pid" in data
        assert isinstance(data["pid"], int)

    def test_returns_memory_info_with_psutil(self, handler):
        """Returns memory info when psutil is available."""
        result = handler.handle("/api/metrics/system", {}, None)
        data = json.loads(result.body)

        # psutil should be available in test environment
        if "rss_mb" in data.get("memory", {}):
            assert isinstance(data["memory"]["rss_mb"], (int, float))
            assert isinstance(data["memory"]["vms_mb"], (int, float))

    @patch.dict("sys.modules", {"psutil": None})
    def test_handles_missing_psutil(self, handler):
        """Gracefully handles missing psutil."""
        # This test verifies the fallback behavior exists
        result = handler.handle("/api/metrics/system", {}, None)
        data = json.loads(result.body)

        assert "memory" in data
        # Either has memory info or indicates unavailable
        assert "rss_mb" in data["memory"] or "available" in data["memory"]


class TestPrometheusEndpoint:
    """Test /metrics Prometheus endpoint."""

    def test_returns_200_success(self, handler):
        """Returns 200 for Prometheus metrics."""
        result = handler.handle("/metrics", {}, None)
        assert result.status_code == 200

    def test_returns_text_content_type(self, handler):
        """Returns text content type for Prometheus format."""
        result = handler.handle("/metrics", {}, None)
        # Prometheus uses text/plain or openmetrics format
        assert "text" in result.content_type or "openmetrics" in result.content_type

    def test_returns_metric_lines(self, handler):
        """Returns Prometheus-format metric lines."""
        result = handler.handle("/metrics", {}, None)
        body = result.body.decode("utf-8")

        # Should contain some metric-like content
        assert len(body) > 0

    @patch("aragora.server.handlers.metrics.get_metrics_output")
    def test_handles_prometheus_error(self, mock_output, handler):
        """Handles errors during Prometheus generation."""
        mock_output.side_effect = Exception("Prometheus error")

        result = handler.handle("/metrics", {}, None)

        assert result.status_code == 500
        data = json.loads(result.body)
        assert "error" in data


class TestTrackRequest:
    """Test request tracking function."""

    def test_tracks_normal_requests(self):
        """Tracks normal requests correctly."""
        track_request("/api/test")
        track_request("/api/test")

        assert _request_counts["/api/test"] == 2
        assert "/api/test" not in _error_counts

    def test_tracks_error_requests(self):
        """Tracks error requests correctly."""
        track_request("/api/test", is_error=True)

        assert _request_counts["/api/test"] == 1
        assert _error_counts["/api/test"] == 1

    def test_tracks_multiple_endpoints(self):
        """Tracks multiple endpoints separately."""
        track_request("/api/one")
        track_request("/api/two")
        track_request("/api/one")

        assert _request_counts["/api/one"] == 2
        assert _request_counts["/api/two"] == 1


class TestFormatHelpers:
    """Test formatting helper methods."""

    def test_format_uptime_seconds(self, handler):
        """Formats seconds correctly."""
        result = handler._format_uptime(45)
        assert result == "45s"

    def test_format_uptime_minutes(self, handler):
        """Formats minutes correctly."""
        result = handler._format_uptime(125)  # 2m 5s
        assert "2m" in result
        assert "5s" in result

    def test_format_uptime_hours(self, handler):
        """Formats hours correctly."""
        result = handler._format_uptime(3725)  # 1h 2m 5s
        assert "1h" in result
        assert "2m" in result

    def test_format_uptime_days(self, handler):
        """Formats days correctly."""
        result = handler._format_uptime(90061)  # 1d 1h 1m
        assert "1d" in result
        assert "1h" in result

    def test_format_size_bytes(self, handler):
        """Formats bytes correctly."""
        result = handler._format_size(512)
        assert "512" in result
        assert "B" in result

    def test_format_size_kilobytes(self, handler):
        """Formats kilobytes correctly."""
        result = handler._format_size(2048)
        assert "KB" in result

    def test_format_size_megabytes(self, handler):
        """Formats megabytes correctly."""
        result = handler._format_size(5 * 1024 * 1024)
        assert "MB" in result

    def test_format_size_gigabytes(self, handler):
        """Formats gigabytes correctly."""
        result = handler._format_size(3 * 1024 * 1024 * 1024)
        assert "GB" in result


class TestDatabaseSizes:
    """Test database size detection."""

    def test_returns_empty_without_nomic_dir(self, handler_no_components):
        """Returns empty dict when nomic_dir unavailable."""
        sizes = handler_no_components._get_database_sizes()
        assert sizes == {}

    def test_finds_existing_databases(self, handler, tmp_path):
        """Finds and reports existing database files."""
        # Create test database file - must match DB_ELO_PATH ("agent_elo.db")
        db_path = tmp_path / "agent_elo.db"
        db_path.write_bytes(b"x" * 2048)

        sizes = handler._get_database_sizes()

        assert "agent_elo.db" in sizes
        assert sizes["agent_elo.db"]["bytes"] == 2048
        assert "human" in sizes["agent_elo.db"]

    def test_ignores_missing_databases(self, handler, tmp_path):
        """Ignores database files that don't exist."""
        sizes = handler._get_database_sizes()

        # Should not include non-existent files
        assert "debate_storage.db" not in sizes
