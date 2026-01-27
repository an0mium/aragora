"""Tests for autonomous monitoring handler."""

import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web

from aragora.server.handlers.autonomous import monitoring


# =============================================================================
# Mock Classes
# =============================================================================


class MockTrendMonitor:
    """Mock TrendMonitor for testing."""

    def __init__(self):
        self._data = {}
        self._trends = {}

    def record(self, metric_name, value):
        if metric_name not in self._data:
            self._data[metric_name] = []
        self._data[metric_name].append(value)

    def get_trend(self, metric_name):
        return self._trends.get(metric_name)

    def get_all_trends(self):
        return self._trends


class MockAnomaly:
    """Mock anomaly data."""

    def __init__(
        self,
        metric_name: str = "cpu_usage",
        value: float = 95.0,
        z_score: float = 3.5,
        timestamp=None,
    ):
        self.metric_name = metric_name
        self.value = value
        self.z_score = z_score
        self.timestamp = timestamp or datetime.now()


class MockAnomalyDetector:
    """Mock AnomalyDetector for testing."""

    def __init__(self):
        self._data = {}
        self._anomalies = []
        self._next_anomaly = None

    def record(self, metric_name, value):
        if metric_name not in self._data:
            self._data[metric_name] = []
        self._data[metric_name].append(value)
        return self._next_anomaly

    def get_recent_anomalies(self, limit=10):
        return self._anomalies[:limit]


class MockAuthContext:
    """Mock authorization context."""

    def __init__(self, user_id="test-user"):
        self.user_id = user_id


class MockPermissionDecision:
    """Mock permission decision."""

    def __init__(self, allowed=True, reason=None):
        self.allowed = allowed
        self.reason = reason or ""


class MockPermissionChecker:
    """Mock permission checker."""

    def check_permission(self, ctx, permission):
        return MockPermissionDecision(allowed=True)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_trend_monitor():
    """Create mock trend monitor."""
    return MockTrendMonitor()


@pytest.fixture
def mock_anomaly_detector():
    """Create mock anomaly detector."""
    return MockAnomalyDetector()


@pytest.fixture
def mock_auth_context():
    """Create mock auth context."""
    return MockAuthContext()


@pytest.fixture
def mock_permission_checker():
    """Create mock permission checker."""
    return MockPermissionChecker()


# =============================================================================
# Test MonitoringHandler.record_metric
# =============================================================================


class TestMonitoringHandlerRecordMetric:
    """Tests for POST /api/autonomous/monitoring/record endpoint."""

    @pytest.mark.asyncio
    async def test_record_metric_success(
        self,
        mock_trend_monitor,
        mock_anomaly_detector,
        mock_auth_context,
        mock_permission_checker,
    ):
        """Should record metric successfully."""
        with (
            patch.object(monitoring, "get_trend_monitor", return_value=mock_trend_monitor),
            patch.object(monitoring, "get_anomaly_detector", return_value=mock_anomaly_detector),
            patch.object(
                monitoring,
                "get_auth_context",
                AsyncMock(return_value=mock_auth_context),
            ),
            patch.object(
                monitoring,
                "get_permission_checker",
                return_value=mock_permission_checker,
            ),
        ):
            request = MagicMock()
            request.json = AsyncMock(return_value={"metric_name": "cpu_usage", "value": 75.0})

            response = await monitoring.MonitoringHandler.record_metric(request)

            assert response.status == 200
            body = json.loads(response.body)
            assert body["success"] is True
            assert body["metric_name"] == "cpu_usage"

    @pytest.mark.asyncio
    async def test_record_metric_missing_fields(
        self,
        mock_trend_monitor,
        mock_anomaly_detector,
        mock_auth_context,
        mock_permission_checker,
    ):
        """Should return 400 for missing required fields."""
        with (
            patch.object(monitoring, "get_trend_monitor", return_value=mock_trend_monitor),
            patch.object(monitoring, "get_anomaly_detector", return_value=mock_anomaly_detector),
            patch.object(
                monitoring,
                "get_auth_context",
                AsyncMock(return_value=mock_auth_context),
            ),
            patch.object(
                monitoring,
                "get_permission_checker",
                return_value=mock_permission_checker,
            ),
        ):
            request = MagicMock()
            request.json = AsyncMock(return_value={})

            response = await monitoring.MonitoringHandler.record_metric(request)

            assert response.status == 400

    @pytest.mark.asyncio
    async def test_record_metric_unauthorized(self, mock_trend_monitor, mock_anomaly_detector):
        """Should return 401 when unauthorized."""
        with (
            patch.object(monitoring, "get_trend_monitor", return_value=mock_trend_monitor),
            patch.object(monitoring, "get_anomaly_detector", return_value=mock_anomaly_detector),
            patch.object(
                monitoring,
                "get_auth_context",
                AsyncMock(side_effect=monitoring.UnauthorizedError("Not authenticated")),
            ),
        ):
            request = MagicMock()
            response = await monitoring.MonitoringHandler.record_metric(request)

            assert response.status == 401


# =============================================================================
# Test Route Registration
# =============================================================================


class TestMonitoringHandlerRoutes:
    """Tests for route registration."""

    def test_register_routes(self):
        """Should register all monitoring routes."""
        app = web.Application()
        monitoring.MonitoringHandler.register_routes(app)

        routes = [r.resource.canonical for r in app.router.routes()]
        assert "/api/v1/autonomous/monitoring/record" in routes
        assert "/api/v1/autonomous/monitoring/trends" in routes
        assert "/api/v1/autonomous/monitoring/trends/{metric_name}" in routes
        assert "/api/v1/autonomous/monitoring/anomalies" in routes


# =============================================================================
# Test Global Functions
# =============================================================================


class TestMonitoringSingletons:
    """Tests for monitoring singleton functions."""

    def test_get_trend_monitor_creates_singleton(self):
        """get_trend_monitor should return same instance."""
        monitoring._trend_monitor = None

        monitor1 = monitoring.get_trend_monitor()
        monitor2 = monitoring.get_trend_monitor()

        assert monitor1 is monitor2

        # Clean up
        monitoring._trend_monitor = None

    def test_get_anomaly_detector_creates_singleton(self):
        """get_anomaly_detector should return same instance."""
        monitoring._anomaly_detector = None

        detector1 = monitoring.get_anomaly_detector()
        detector2 = monitoring.get_anomaly_detector()

        assert detector1 is detector2

        # Clean up
        monitoring._anomaly_detector = None

    def test_set_trend_monitor(self):
        """set_trend_monitor should update the global instance."""
        mock = MockTrendMonitor()
        monitoring.set_trend_monitor(mock)

        assert monitoring.get_trend_monitor() is mock

        # Clean up
        monitoring._trend_monitor = None

    def test_set_anomaly_detector(self):
        """set_anomaly_detector should update the global instance."""
        mock = MockAnomalyDetector()
        monitoring.set_anomaly_detector(mock)

        assert monitoring.get_anomaly_detector() is mock

        # Clean up
        monitoring._anomaly_detector = None
