"""
Tests for Cross-Platform Analytics Handler.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.server.handlers.features.cross_platform_analytics import (
    CrossPlatformAnalyticsHandler,
    get_cross_platform_analytics_handler,
    handle_cross_platform_analytics,
    Platform,
    MetricType,
    AlertSeverity,
    AlertStatus,
    TimeRange,
    MetricValue,
    AggregatedMetric,
    Anomaly,
    AlertRule,
    Alert,
    calculate_trend,
    detect_anomalies,
    calculate_correlation,
)


class TestPlatformEnum:
    """Tests for Platform enum."""

    def test_platform_values(self):
        """Test platform enum values."""
        assert Platform.ARAGORA.value == "aragora"
        assert Platform.GOOGLE_ANALYTICS.value == "google_analytics"
        assert Platform.MIXPANEL.value == "mixpanel"
        assert Platform.METABASE.value == "metabase"
        assert Platform.SEGMENT.value == "segment"


class TestMetricType:
    """Tests for MetricType enum."""

    def test_metric_type_values(self):
        """Test metric type enum values."""
        assert MetricType.COUNTER.value == "counter"
        assert MetricType.GAUGE.value == "gauge"
        assert MetricType.HISTOGRAM.value == "histogram"
        assert MetricType.RATE.value == "rate"


class TestAlertSeverity:
    """Tests for AlertSeverity enum."""

    def test_severity_values(self):
        """Test alert severity enum values."""
        assert AlertSeverity.INFO.value == "info"
        assert AlertSeverity.WARNING.value == "warning"
        assert AlertSeverity.CRITICAL.value == "critical"


class TestMetricValue:
    """Tests for MetricValue dataclass."""

    def test_metric_value_creation(self):
        """Test metric value creation."""
        metric = MetricValue(
            name="users",
            value=1234.5,
            platform=Platform.GOOGLE_ANALYTICS,
            timestamp=datetime.now(timezone.utc),
        )

        assert metric.name == "users"
        assert metric.value == 1234.5
        assert metric.platform == Platform.GOOGLE_ANALYTICS

    def test_metric_value_to_dict(self):
        """Test metric value serialization."""
        metric = MetricValue(
            name="events",
            value=5678,
            platform=Platform.MIXPANEL,
            timestamp=datetime.now(timezone.utc),
            dimensions={"country": "US"},
        )

        data = metric.to_dict()

        assert data["name"] == "events"
        assert data["value"] == 5678
        assert data["platform"] == "mixpanel"
        assert data["dimensions"]["country"] == "US"


class TestAggregatedMetric:
    """Tests for AggregatedMetric dataclass."""

    def test_aggregated_metric_creation(self):
        """Test aggregated metric creation."""
        metric = AggregatedMetric(
            name="total_users",
            total=35000,
            by_platform={"google_analytics": 20000, "mixpanel": 15000},
            trend="up",
            change_percent=12.5,
            period="7d",
        )

        assert metric.total == 35000
        assert metric.trend == "up"

    def test_aggregated_metric_to_dict(self):
        """Test aggregated metric serialization."""
        metric = AggregatedMetric(
            name="total_events",
            total=100000,
            by_platform={"ga4": 50000, "mixpanel": 50000},
            trend="stable",
            change_percent=0.5,
            period="30d",
        )

        data = metric.to_dict()

        assert data["total"] == 100000
        assert data["trend"] == "stable"


class TestAnomaly:
    """Tests for Anomaly dataclass."""

    def test_anomaly_creation(self):
        """Test anomaly creation."""
        anomaly = Anomaly(
            id="anom_123",
            metric_name="error_rate",
            platform=Platform.ARAGORA,
            timestamp=datetime.now(timezone.utc),
            expected_value=0.02,
            actual_value=0.08,
            deviation=3.0,
            severity=AlertSeverity.WARNING,
            description="Error rate spike",
        )

        assert anomaly.deviation == 3.0
        assert anomaly.severity == AlertSeverity.WARNING

    def test_anomaly_to_dict(self):
        """Test anomaly serialization."""
        anomaly = Anomaly(
            id="anom_456",
            metric_name="latency",
            platform=Platform.METABASE,
            timestamp=datetime.now(timezone.utc),
            expected_value=100,
            actual_value=500,
            deviation=4.0,
            severity=AlertSeverity.CRITICAL,
            description="High latency",
        )

        data = anomaly.to_dict()

        assert data["metric_name"] == "latency"
        assert data["severity"] == "critical"


class TestAlertRule:
    """Tests for AlertRule dataclass."""

    def test_alert_rule_creation(self):
        """Test alert rule creation."""
        rule = AlertRule(
            id="rule_123",
            name="High Error Rate",
            metric_name="error_rate",
            condition="above",
            threshold=0.05,
            severity=AlertSeverity.WARNING,
        )

        assert rule.condition == "above"
        assert rule.enabled is True

    def test_alert_rule_to_dict(self):
        """Test alert rule serialization."""
        rule = AlertRule(
            id="rule_456",
            name="Low Retention",
            metric_name="retention_day7",
            condition="below",
            threshold=0.2,
            severity=AlertSeverity.CRITICAL,
            platforms=[Platform.MIXPANEL],
        )

        data = rule.to_dict()

        assert data["condition"] == "below"
        assert data["platforms"] == ["mixpanel"]


class TestAlert:
    """Tests for Alert dataclass."""

    def test_alert_creation(self):
        """Test alert creation."""
        alert = Alert(
            id="alert_123",
            rule_id="rule_456",
            rule_name="High Error Rate",
            metric_name="error_rate",
            platform=Platform.ARAGORA,
            severity=AlertSeverity.WARNING,
            status=AlertStatus.ACTIVE,
            triggered_at=datetime.now(timezone.utc),
            current_value=0.08,
            threshold=0.05,
            message="Error rate exceeded threshold",
        )

        assert alert.status == AlertStatus.ACTIVE

    def test_alert_to_dict(self):
        """Test alert serialization."""
        alert = Alert(
            id="alert_789",
            rule_id="rule_abc",
            rule_name="Low Sessions",
            metric_name="sessions",
            platform=Platform.GOOGLE_ANALYTICS,
            severity=AlertSeverity.INFO,
            status=AlertStatus.ACKNOWLEDGED,
            triggered_at=datetime.now(timezone.utc),
            current_value=100,
            threshold=500,
            message="Sessions dropped",
            acknowledged_by="user_123",
        )

        data = alert.to_dict()

        assert data["status"] == "acknowledged"
        assert data["acknowledged_by"] == "user_123"


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_calculate_trend_up(self):
        """Test trend calculation - upward."""
        trend, change = calculate_trend(110, 100)
        assert trend == "up"
        assert change == pytest.approx(10.0)

    def test_calculate_trend_down(self):
        """Test trend calculation - downward."""
        trend, change = calculate_trend(90, 100)
        assert trend == "down"
        assert change == pytest.approx(-10.0)

    def test_calculate_trend_stable(self):
        """Test trend calculation - stable."""
        trend, change = calculate_trend(101, 100)
        assert trend == "stable"

    def test_calculate_trend_zero_previous(self):
        """Test trend calculation with zero previous."""
        trend, change = calculate_trend(100, 0)
        assert trend == "up"
        assert change == 100.0

    def test_detect_anomalies_empty(self):
        """Test anomaly detection with insufficient data."""
        anomalies = detect_anomalies([], "test", Platform.ARAGORA)
        assert len(anomalies) == 0

    def test_detect_anomalies_normal(self):
        """Test anomaly detection with normal values."""
        values = [100, 101, 99, 100, 102, 98, 101]
        anomalies = detect_anomalies(values, "test", Platform.ARAGORA)
        assert len(anomalies) == 0

    def test_detect_anomalies_with_spike(self):
        """Test anomaly detection with spike."""
        values = [100, 101, 99, 100, 102, 98, 500]  # Last value is anomaly
        anomalies = detect_anomalies(values, "test", Platform.ARAGORA, threshold_std=2.0)
        assert len(anomalies) > 0

    def test_calculate_correlation_same(self):
        """Test correlation of identical series."""
        series = [1, 2, 3, 4, 5]
        corr = calculate_correlation(series, series)
        assert corr == pytest.approx(1.0)

    def test_calculate_correlation_inverse(self):
        """Test correlation of inverse series."""
        series_a = [1, 2, 3, 4, 5]
        series_b = [5, 4, 3, 2, 1]
        corr = calculate_correlation(series_a, series_b)
        assert corr == pytest.approx(-1.0)


class TestCrossPlatformAnalyticsHandler:
    """Tests for CrossPlatformAnalyticsHandler."""

    def test_handler_routes(self):
        """Test handler has expected routes."""
        handler = CrossPlatformAnalyticsHandler()

        expected_routes = [
            "/api/v1/analytics/cross-platform/summary",
            "/api/v1/analytics/cross-platform/metrics",
            "/api/v1/analytics/cross-platform/trends",
            "/api/v1/analytics/cross-platform/comparison",
            "/api/v1/analytics/cross-platform/correlation",
            "/api/v1/analytics/cross-platform/anomalies",
            "/api/v1/analytics/cross-platform/alerts",
            "/api/v1/analytics/cross-platform/export",
            "/api/v1/analytics/cross-platform/demo",
        ]

        for route in expected_routes:
            assert any(route in r for r in handler.ROUTES), f"Missing route: {route}"

    def test_get_handler_instance(self):
        """Test getting handler instance."""
        handler1 = get_cross_platform_analytics_handler()
        handler2 = get_cross_platform_analytics_handler()

        assert handler1 is handler2


class TestSummaryEndpoint:
    """Tests for summary endpoint."""

    @pytest.mark.asyncio
    async def test_summary_default(self):
        """Test summary with default parameters."""
        handler = CrossPlatformAnalyticsHandler()

        request = MagicMock()
        request.tenant_id = "test_tenant"
        request.query = {}

        result = await handler.handle(request, "/api/v1/analytics/cross-platform/summary", "GET")

        assert result is not None
        assert result.status_code == 200
        assert b"platforms_connected" in result.body
        assert b"key_metrics" in result.body

    @pytest.mark.asyncio
    async def test_summary_with_range(self):
        """Test summary with time range."""
        handler = CrossPlatformAnalyticsHandler()

        request = MagicMock()
        request.tenant_id = "test_tenant"
        request.query = {"range": "7d"}

        result = await handler.handle(request, "/api/v1/analytics/cross-platform/summary", "GET")

        assert result is not None
        assert result.status_code == 200


class TestMetricsEndpoint:
    """Tests for metrics endpoint."""

    @pytest.mark.asyncio
    async def test_metrics_all_platforms(self):
        """Test metrics from all platforms."""
        handler = CrossPlatformAnalyticsHandler()

        request = MagicMock()
        request.tenant_id = "test_tenant"
        request.query = {}

        result = await handler.handle(request, "/api/v1/analytics/cross-platform/metrics", "GET")

        assert result is not None
        assert result.status_code == 200
        assert b"metrics_by_platform" in result.body

    @pytest.mark.asyncio
    async def test_metrics_single_platform(self):
        """Test metrics from single platform."""
        handler = CrossPlatformAnalyticsHandler()

        request = MagicMock()
        request.tenant_id = "test_tenant"
        request.query = {"platform": "aragora"}

        result = await handler.handle(request, "/api/v1/analytics/cross-platform/metrics", "GET")

        assert result is not None
        assert result.status_code == 200


class TestTrendsEndpoint:
    """Tests for trends endpoint."""

    @pytest.mark.asyncio
    async def test_trends_default(self):
        """Test trends with default parameters."""
        handler = CrossPlatformAnalyticsHandler()

        request = MagicMock()
        request.tenant_id = "test_tenant"
        request.query = {}

        result = await handler.handle(request, "/api/v1/analytics/cross-platform/trends", "GET")

        assert result is not None
        assert result.status_code == 200
        assert b"data_points" in result.body

    @pytest.mark.asyncio
    async def test_trends_with_metric(self):
        """Test trends with specific metric."""
        handler = CrossPlatformAnalyticsHandler()

        request = MagicMock()
        request.tenant_id = "test_tenant"
        request.query = {"metric": "events", "range": "30d"}

        result = await handler.handle(request, "/api/v1/analytics/cross-platform/trends", "GET")

        assert result is not None
        assert result.status_code == 200


class TestComparisonEndpoint:
    """Tests for comparison endpoint."""

    @pytest.mark.asyncio
    async def test_comparison_engagement(self):
        """Test engagement comparison."""
        handler = CrossPlatformAnalyticsHandler()

        request = MagicMock()
        request.tenant_id = "test_tenant"
        request.query = {"type": "engagement"}

        result = await handler.handle(request, "/api/v1/analytics/cross-platform/comparison", "GET")

        assert result is not None
        assert result.status_code == 200
        assert b"comparisons" in result.body

    @pytest.mark.asyncio
    async def test_comparison_performance(self):
        """Test performance comparison."""
        handler = CrossPlatformAnalyticsHandler()

        request = MagicMock()
        request.tenant_id = "test_tenant"
        request.query = {"type": "performance"}

        result = await handler.handle(request, "/api/v1/analytics/cross-platform/comparison", "GET")

        assert result is not None
        assert result.status_code == 200


class TestCorrelationEndpoint:
    """Tests for correlation endpoint."""

    @pytest.mark.asyncio
    async def test_correlation(self):
        """Test correlation calculation."""
        handler = CrossPlatformAnalyticsHandler()

        request = MagicMock()
        request.tenant_id = "test_tenant"
        request.query = {"range": "30d"}

        result = await handler.handle(
            request, "/api/v1/analytics/cross-platform/correlation", "GET"
        )

        assert result is not None
        assert result.status_code == 200
        assert b"correlation_matrix" in result.body
        assert b"strong_correlations" in result.body


class TestAnomaliesEndpoint:
    """Tests for anomalies endpoint."""

    @pytest.mark.asyncio
    async def test_anomalies_default(self):
        """Test anomalies detection."""
        handler = CrossPlatformAnalyticsHandler()

        request = MagicMock()
        request.tenant_id = "test_tenant"
        request.query = {}

        result = await handler.handle(request, "/api/v1/analytics/cross-platform/anomalies", "GET")

        assert result is not None
        assert result.status_code == 200
        assert b"anomalies" in result.body

    @pytest.mark.asyncio
    async def test_anomalies_with_severity(self):
        """Test anomalies with severity filter."""
        handler = CrossPlatformAnalyticsHandler()

        request = MagicMock()
        request.tenant_id = "test_tenant"
        request.query = {"severity": "warning"}

        result = await handler.handle(request, "/api/v1/analytics/cross-platform/anomalies", "GET")

        assert result is not None
        assert result.status_code == 200


class TestQueryEndpoint:
    """Tests for custom query endpoint."""

    @pytest.mark.asyncio
    async def test_query_requires_metrics(self):
        """Test query requires metrics."""
        handler = CrossPlatformAnalyticsHandler()

        request = MagicMock()
        request.tenant_id = "test_tenant"
        request.json = AsyncMock(return_value={})

        result = await handler.handle(request, "/api/v1/analytics/cross-platform/query", "POST")

        assert result is not None
        assert result.status_code == 400

    @pytest.mark.asyncio
    async def test_query_success(self):
        """Test successful query."""
        handler = CrossPlatformAnalyticsHandler()

        request = MagicMock()
        request.tenant_id = "test_tenant"
        request.json = AsyncMock(
            return_value={
                "metrics": ["users", "events"],
                "platforms": ["aragora", "google_analytics"],
                "range": "7d",
            }
        )

        result = await handler.handle(request, "/api/v1/analytics/cross-platform/query", "POST")

        assert result is not None
        assert result.status_code == 200
        assert b"results" in result.body


class TestAlertsEndpoint:
    """Tests for alerts endpoint."""

    @pytest.mark.asyncio
    async def test_list_alerts_empty(self):
        """Test listing alerts when empty."""
        handler = CrossPlatformAnalyticsHandler()

        request = MagicMock()
        request.tenant_id = "empty_tenant"
        request.query = {}

        result = await handler.handle(request, "/api/v1/analytics/cross-platform/alerts", "GET")

        assert result is not None
        assert result.status_code == 200
        assert b"alerts" in result.body

    @pytest.mark.asyncio
    async def test_create_alert_rule(self):
        """Test creating alert rule."""
        handler = CrossPlatformAnalyticsHandler()

        request = MagicMock()
        request.tenant_id = "test_tenant"
        request.json = AsyncMock(
            return_value={
                "name": "High Error Rate",
                "metric_name": "error_rate",
                "condition": "above",
                "threshold": 0.05,
                "severity": "warning",
            }
        )

        result = await handler.handle(request, "/api/v1/analytics/cross-platform/alerts", "POST")

        assert result is not None
        assert result.status_code == 200
        assert b"created" in result.body

    @pytest.mark.asyncio
    async def test_create_alert_missing_fields(self):
        """Test creating alert with missing fields."""
        handler = CrossPlatformAnalyticsHandler()

        request = MagicMock()
        request.tenant_id = "test_tenant"
        request.json = AsyncMock(return_value={"name": "Test"})

        result = await handler.handle(request, "/api/v1/analytics/cross-platform/alerts", "POST")

        assert result is not None
        assert result.status_code == 400


class TestExportEndpoint:
    """Tests for export endpoint."""

    @pytest.mark.asyncio
    async def test_export_json(self):
        """Test JSON export."""
        handler = CrossPlatformAnalyticsHandler()

        request = MagicMock()
        request.tenant_id = "test_tenant"
        request.query = {"format": "json", "range": "7d"}

        result = await handler.handle(request, "/api/v1/analytics/cross-platform/export", "GET")

        assert result is not None
        assert result.status_code == 200
        assert b"data" in result.body

    @pytest.mark.asyncio
    async def test_export_csv(self):
        """Test CSV export."""
        handler = CrossPlatformAnalyticsHandler()

        request = MagicMock()
        request.tenant_id = "test_tenant"
        request.query = {"format": "csv"}

        result = await handler.handle(request, "/api/v1/analytics/cross-platform/export", "GET")

        assert result is not None
        assert result.status_code == 200
        assert result.content_type == "text/csv"

    @pytest.mark.asyncio
    async def test_export_unsupported_format(self):
        """Test unsupported export format."""
        handler = CrossPlatformAnalyticsHandler()

        request = MagicMock()
        request.tenant_id = "test_tenant"
        request.query = {"format": "xml"}

        result = await handler.handle(request, "/api/v1/analytics/cross-platform/export", "GET")

        assert result is not None
        assert result.status_code == 400


class TestDemoEndpoint:
    """Tests for demo endpoint."""

    @pytest.mark.asyncio
    async def test_demo_endpoint(self):
        """Test demo endpoint returns mock data."""
        handler = CrossPlatformAnalyticsHandler()

        request = MagicMock()
        request.tenant_id = "test_tenant"

        result = await handler.handle(request, "/api/v1/analytics/cross-platform/demo", "GET")

        assert result is not None
        assert result.status_code == 200
        assert b"is_demo" in result.body
        assert b"platforms" in result.body


class TestHandleFunction:
    """Tests for handle_cross_platform_analytics entry point."""

    @pytest.mark.asyncio
    async def test_entry_point(self):
        """Test entry point function."""
        request = MagicMock()
        request.tenant_id = "test"
        request.query = {}

        result = await handle_cross_platform_analytics(
            request, "/api/v1/analytics/cross-platform/summary", "GET"
        )

        assert result is not None


class TestNotFoundRoute:
    """Tests for not found route."""

    @pytest.mark.asyncio
    async def test_unknown_route(self):
        """Test handling unknown route."""
        handler = CrossPlatformAnalyticsHandler()

        request = MagicMock()
        request.tenant_id = "test_tenant"

        result = await handler.handle(request, "/api/v1/analytics/cross-platform/unknown", "GET")

        assert result is not None
        assert result.status_code == 404


class TestImports:
    """Test that imports work correctly."""

    def test_import_from_package(self):
        """Test imports from features package."""
        from aragora.server.handlers.features import (
            CrossPlatformAnalyticsHandler,
            handle_cross_platform_analytics,
            get_cross_platform_analytics_handler,
            Platform,
            MetricType,
            MetricValue,
            AggregatedMetric,
        )

        assert CrossPlatformAnalyticsHandler is not None
        assert handle_cross_platform_analytics is not None
        assert Platform is not None
        assert MetricValue is not None
