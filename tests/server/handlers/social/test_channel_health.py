"""Tests for aragora.server.handlers.social.channel_health - Channel Health Handler."""

import time
from unittest.mock import MagicMock, patch

import pytest

from .conftest import MockHandler, install_social_slack_stubs

install_social_slack_stubs()

from aragora.server.handlers.social.channel_health import ChannelHealthHandler


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture(autouse=True)
def reset_module_state():
    """Reset module-level state before each test."""
    try:
        from aragora.server.handlers.social import channel_health

        channel_health._health_limiter._buckets.clear()
    except Exception:
        pass
    yield


@pytest.fixture
def mock_channel_monitor():
    monitor = MagicMock()
    monitor.get_channel_status.return_value = {
        "channel_id": "C12345",
        "platform": "slack",
        "status": "healthy",
        "last_message_at": time.time() - 60,
        "message_count_24h": 150,
        "error_count_24h": 2,
    }
    monitor.get_all_channels.return_value = [
        {"channel_id": "C12345", "platform": "slack", "status": "healthy"},
        {"channel_id": "C67890", "platform": "slack", "status": "degraded"},
    ]
    monitor.check_health.return_value = {
        "overall_status": "healthy",
        "channels_checked": 5,
        "healthy": 4,
        "degraded": 1,
        "unhealthy": 0,
    }
    return monitor


@pytest.fixture
def health_handler(handler_context, mock_channel_monitor):
    handler = ChannelHealthHandler()
    yield handler


# ===========================================================================
# Routing Tests
# ===========================================================================


class TestRouting:
    """Tests for route handling."""

    def test_can_handle_health_status(self, health_handler):
        """Test handler recognizes health status endpoint."""
        assert health_handler.can_handle("/api/v1/social/channels/health") is True

    def test_can_handle_channel_status(self, health_handler):
        """Test handler recognizes channel status endpoint."""
        assert health_handler.can_handle("/api/v1/social/channels/C12345/health") is True

    def test_can_handle_metrics(self, health_handler):
        """Test handler recognizes metrics endpoint."""
        assert health_handler.can_handle("/api/v1/social/channels/health/metrics") is True

    def test_cannot_handle_unknown_path(self, health_handler):
        """Test handler rejects unknown paths."""
        assert health_handler.can_handle("/api/v1/unknown") is False


# ===========================================================================
# Overall Health Tests
# ===========================================================================


class TestOverallHealth:
    """Tests for overall health status."""

    def test_get_overall_health_success(self, health_handler, mock_user):
        """Test successful overall health retrieval."""
        http_handler = MockHandler(path="/api/v1/social/channels/health", method="GET")

        result = health_handler.handle(
            "/api/v1/social/channels/health", {}, http_handler, method="GET"
        )
        assert result is not None

    def test_get_overall_health_with_platform_filter(self, health_handler, mock_user):
        """Test health retrieval with platform filter."""
        http_handler = MockHandler(path="/api/v1/social/channels/health", method="GET")

        result = health_handler.handle(
            "/api/v1/social/channels/health",
            {"platform": "slack"},
            http_handler,
            method="GET",
        )
        assert result is not None


# ===========================================================================
# Channel Health Tests
# ===========================================================================


class TestChannelHealth:
    """Tests for individual channel health."""

    def test_get_channel_health_success(self, health_handler, mock_user):
        """Test successful channel health retrieval."""
        http_handler = MockHandler(path="/api/v1/social/channels/C12345/health", method="GET")

        result = health_handler.handle(
            "/api/v1/social/channels/C12345/health", {}, http_handler, method="GET"
        )
        assert result is not None

    def test_get_channel_health_not_found(self, health_handler, mock_channel_monitor):
        """Test channel not found error."""
        mock_channel_monitor.get_channel_status.return_value = None
        http_handler = MockHandler(path="/api/v1/social/channels/C99999/health", method="GET")

        result = health_handler.handle(
            "/api/v1/social/channels/C99999/health", {}, http_handler, method="GET"
        )
        assert result is not None


# ===========================================================================
# Metrics Tests
# ===========================================================================


class TestMetrics:
    """Tests for health metrics."""

    def test_get_metrics_success(self, health_handler, mock_user):
        """Test successful metrics retrieval."""
        http_handler = MockHandler(path="/api/v1/social/channels/health/metrics", method="GET")

        result = health_handler.handle(
            "/api/v1/social/channels/health/metrics", {}, http_handler, method="GET"
        )
        assert result is not None

    def test_get_metrics_with_time_range(self, health_handler, mock_user):
        """Test metrics retrieval with time range."""
        http_handler = MockHandler(path="/api/v1/social/channels/health/metrics", method="GET")

        result = health_handler.handle(
            "/api/v1/social/channels/health/metrics",
            {"from": str(int(time.time()) - 3600), "to": str(int(time.time()))},
            http_handler,
            method="GET",
        )
        assert result is not None


# ===========================================================================
# Rate Limiting Tests
# ===========================================================================


class TestRateLimiting:
    """Tests for rate limiting."""

    def test_rate_limit_exceeded(self, health_handler):
        """Test rate limit enforcement."""
        http_handler = MockHandler(path="/api/v1/social/channels/health", method="GET")

        with patch(
            "aragora.server.handlers.social.channel_health._health_limiter.is_allowed",
            return_value=False,
        ):
            result = health_handler.handle(
                "/api/v1/social/channels/health", {}, http_handler, method="GET"
            )
            assert result is not None
            assert result.status_code == 429


# ===========================================================================
# Method Not Allowed Tests
# ===========================================================================


class TestMethodNotAllowed:
    """Tests for method not allowed responses."""

    def test_health_get_only(self, health_handler):
        """Test only GET allowed for health endpoint."""
        http_handler = MockHandler(path="/api/v1/social/channels/health", method="POST")

        result = health_handler.handle(
            "/api/v1/social/channels/health", {}, http_handler, method="POST"
        )
        assert result is not None
        assert result.status_code == 405

    def test_metrics_get_only(self, health_handler):
        """Test only GET allowed for metrics endpoint."""
        http_handler = MockHandler(path="/api/v1/social/channels/health/metrics", method="DELETE")

        result = health_handler.handle(
            "/api/v1/social/channels/health/metrics", {}, http_handler, method="DELETE"
        )
        assert result is not None
        assert result.status_code == 405
