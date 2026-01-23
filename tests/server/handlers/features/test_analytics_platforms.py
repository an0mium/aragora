"""
Tests for Analytics Platforms Handler.

Tests cover basic platform configuration and handler creation.
"""

import pytest

from aragora.server.handlers.features.analytics_platforms import (
    AnalyticsPlatformsHandler,
    SUPPORTED_PLATFORMS,
)


class TestSupportedPlatforms:
    """Tests for analytics platform configuration."""

    def test_all_platforms_defined(self):
        """Test that analytics platforms are configured."""
        # At least google_analytics and mixpanel should be defined
        assert "google_analytics" in SUPPORTED_PLATFORMS
        assert "mixpanel" in SUPPORTED_PLATFORMS

    def test_platform_has_required_fields(self):
        """Test that all platforms have required configuration."""
        for platform_id, config in SUPPORTED_PLATFORMS.items():
            assert "name" in config
            assert "description" in config
            assert "features" in config


class TestAnalyticsPlatformsHandler:
    """Tests for AnalyticsPlatformsHandler class."""

    def test_handler_creation(self):
        """Test creating handler instance."""
        handler = AnalyticsPlatformsHandler(server_context={})
        assert handler is not None

    def test_handler_has_routes(self):
        """Test that handler has route definitions."""
        handler = AnalyticsPlatformsHandler(server_context={})
        assert hasattr(handler, "handle_get") or hasattr(handler, "handle_request")
