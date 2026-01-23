"""
Tests for Cross-Platform Analytics Handler.

Tests cover enums, dataclasses, and basic handler creation.
"""

import pytest

from aragora.server.handlers.features.cross_platform_analytics import (
    CrossPlatformAnalyticsHandler,
    Platform,
    MetricType,
    AlertSeverity,
    AlertStatus,
    TimeRange,
)


class TestPlatformEnum:
    """Tests for Platform enum."""

    def test_all_platforms_defined(self):
        """Test that all platforms are defined."""
        expected = ["aragora", "google_analytics", "mixpanel", "metabase", "segment"]
        for platform in expected:
            assert Platform(platform) is not None

    def test_platform_values(self):
        """Test platform enum values."""
        assert Platform.ARAGORA.value == "aragora"
        assert Platform.GOOGLE_ANALYTICS.value == "google_analytics"
        assert Platform.MIXPANEL.value == "mixpanel"


class TestMetricTypeEnum:
    """Tests for MetricType enum."""

    def test_all_metric_types_defined(self):
        """Test that all metric types are defined."""
        expected = ["counter", "gauge", "histogram", "rate"]
        for metric_type in expected:
            assert MetricType(metric_type) is not None


class TestAlertSeverityEnum:
    """Tests for AlertSeverity enum."""

    def test_all_severities_defined(self):
        """Test that all severities are defined."""
        expected = ["info", "warning", "critical"]
        for severity in expected:
            assert AlertSeverity(severity) is not None


class TestAlertStatusEnum:
    """Tests for AlertStatus enum."""

    def test_all_statuses_defined(self):
        """Test that all alert statuses are defined."""
        expected = ["active", "acknowledged", "resolved"]
        for status in expected:
            assert AlertStatus(status) is not None


class TestTimeRangeEnum:
    """Tests for TimeRange enum."""

    def test_all_time_ranges_defined(self):
        """Test that all time ranges are defined."""
        expected = ["1h", "24h", "7d", "30d", "90d", "365d"]
        for time_range in expected:
            assert TimeRange(time_range) is not None

    def test_time_range_values(self):
        """Test time range enum values."""
        assert TimeRange.LAST_HOUR.value == "1h"
        assert TimeRange.LAST_DAY.value == "24h"
        assert TimeRange.LAST_WEEK.value == "7d"
        assert TimeRange.LAST_MONTH.value == "30d"
        assert TimeRange.LAST_QUARTER.value == "90d"
        assert TimeRange.LAST_YEAR.value == "365d"


class TestCrossPlatformAnalyticsHandler:
    """Tests for CrossPlatformAnalyticsHandler class."""

    def test_handler_creation(self):
        """Test creating handler instance."""
        handler = CrossPlatformAnalyticsHandler(server_context={})
        assert handler is not None

    def test_handler_routes(self):
        """Test that handler has route definitions."""
        assert hasattr(CrossPlatformAnalyticsHandler, "ROUTES")
        routes = CrossPlatformAnalyticsHandler.ROUTES
        assert "/api/v1/analytics/cross-platform/summary" in routes
        assert "/api/v1/analytics/cross-platform/metrics" in routes
        assert "/api/v1/analytics/cross-platform/trends" in routes
        assert "/api/v1/analytics/cross-platform/comparison" in routes
        assert "/api/v1/analytics/cross-platform/anomalies" in routes
        assert "/api/v1/analytics/cross-platform/alerts" in routes
        assert "/api/v1/analytics/cross-platform/demo" in routes

    def test_handler_has_handle_method(self):
        """Test that handler has async handle method."""
        handler = CrossPlatformAnalyticsHandler(server_context={})
        assert hasattr(handler, "handle")
        assert callable(handler.handle)
