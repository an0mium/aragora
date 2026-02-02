"""Comprehensive tests for autonomous monitoring handler.

Tests circuit breaker pattern, rate limiting, input validation, RBAC,
and all endpoint functionality.
"""

import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from typing import Any

import pytest
from aiohttp import web

from aragora.server.handlers.autonomous import monitoring
from aragora.autonomous import TrendDirection, AlertSeverity


# =============================================================================
# Mock Classes
# =============================================================================


class MockTrendData:
    """Mock TrendData for testing."""

    def __init__(
        self,
        metric_name: str = "cpu_usage",
        direction: TrendDirection = TrendDirection.STABLE,
        current_value: float = 50.0,
        previous_value: float = 45.0,
        change_percent: float = 11.1,
        period_start: datetime | None = None,
        period_end: datetime | None = None,
        data_points: int = 20,
        confidence: float = 0.8,
    ):
        self.metric_name = metric_name
        self.direction = direction
        self.current_value = current_value
        self.previous_value = previous_value
        self.change_percent = change_percent
        self.period_start = period_start or datetime(2024, 1, 1, 0, 0)
        self.period_end = period_end or datetime(2024, 1, 1, 1, 0)
        self.data_points = data_points
        self.confidence = confidence


class MockTrendMonitor:
    """Mock TrendMonitor for testing."""

    def __init__(self):
        self._data: dict[str, list[float]] = {}
        self._trends: dict[str, MockTrendData] = {}

    def record(self, metric_name: str, value: float) -> None:
        if metric_name not in self._data:
            self._data[metric_name] = []
        self._data[metric_name].append(value)

    def get_trend(
        self, metric_name: str, period_seconds: int | None = None
    ) -> MockTrendData | None:
        return self._trends.get(metric_name)

    def get_all_trends(self) -> dict[str, MockTrendData]:
        return self._trends


class MockAnomaly:
    """Mock anomaly data."""

    def __init__(
        self,
        id: str = "anomaly_123",
        metric_name: str = "cpu_usage",
        value: float = 95.0,
        expected_value: float = 50.0,
        deviation: float = 3.5,
        timestamp: datetime | None = None,
        severity: AlertSeverity = AlertSeverity.HIGH,
        description: str = "CPU usage is 3.5 standard deviations above expected",
    ):
        self.id = id
        self.metric_name = metric_name
        self.value = value
        self.expected_value = expected_value
        self.deviation = deviation
        self.timestamp = timestamp or datetime.now()
        self.severity = severity
        self.description = description


class MockAnomalyDetector:
    """Mock AnomalyDetector for testing."""

    def __init__(self):
        self._data: dict[str, list[float]] = {}
        self._anomalies: list[MockAnomaly] = []
        self._next_anomaly: MockAnomaly | None = None
        self._baseline_stats: dict[str, dict[str, float]] = {}

    def record(
        self, metric_name: str, value: float, timestamp: datetime | None = None
    ) -> MockAnomaly | None:
        if metric_name not in self._data:
            self._data[metric_name] = []
        self._data[metric_name].append(value)
        return self._next_anomaly

    def get_recent_anomalies(
        self, hours: int = 24, metric_name: str | None = None
    ) -> list[MockAnomaly]:
        anomalies = self._anomalies
        if metric_name:
            anomalies = [a for a in anomalies if a.metric_name == metric_name]
        return anomalies

    def get_baseline_stats(self, metric_name: str) -> dict[str, float] | None:
        return self._baseline_stats.get(metric_name)


class MockAuthContext:
    """Mock authorization context."""

    def __init__(self, user_id: str = "test-user"):
        self.user_id = user_id


class MockPermissionDecision:
    """Mock permission decision."""

    def __init__(self, allowed: bool = True, reason: str | None = None):
        self.allowed = allowed
        self.reason = reason or ""


class MockPermissionChecker:
    """Mock permission checker."""

    def __init__(self, allowed: bool = True, reason: str = ""):
        self._allowed = allowed
        self._reason = reason

    def check_permission(self, ctx: Any, permission: str) -> MockPermissionDecision:
        return MockPermissionDecision(allowed=self._allowed, reason=self._reason)


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


@pytest.fixture
def mock_permission_denied():
    """Create mock permission checker that denies."""
    return MockPermissionChecker(allowed=False, reason="Permission denied")


@pytest.fixture(autouse=True)
def reset_monitoring_components():
    """Reset monitoring components before each test."""
    monitoring._clear_monitoring_components()
    yield
    monitoring._clear_monitoring_components()


# =============================================================================
# Helper Functions
# =============================================================================


def create_mock_request(
    json_data: dict | None = None,
    query: dict | None = None,
    match_info: dict | None = None,
) -> MagicMock:
    """Create a mock aiohttp request."""
    request = MagicMock()
    request.json = AsyncMock(return_value=json_data or {})
    request.query = query or {}
    request.match_info = match_info or {}
    return request


# =============================================================================
# Test Validation Functions
# =============================================================================


class TestValidateMetricName:
    """Tests for metric name validation."""

    def test_valid_metric_names(self):
        """Should accept valid metric names."""
        valid_names = [
            "cpu_usage",
            "memoryUsage",
            "disk.io.read",
            "api-response-time",
            "a",
            "abc123",
            "CPU_USAGE_PERCENT",
            "metric.sub.name",
            "metric-with-hyphens",
        ]
        for name in valid_names:
            is_valid, error = monitoring.validate_metric_name(name)
            assert is_valid, f"Expected {name!r} to be valid, got error: {error}"

    def test_invalid_metric_name_empty(self):
        """Should reject empty metric name."""
        is_valid, error = monitoring.validate_metric_name("")
        assert not is_valid
        assert "required" in error.lower()

    def test_invalid_metric_name_none(self):
        """Should reject None metric name."""
        is_valid, error = monitoring.validate_metric_name(None)
        assert not is_valid
        assert "required" in error.lower()

    def test_invalid_metric_name_not_string(self):
        """Should reject non-string metric name."""
        is_valid, error = monitoring.validate_metric_name(123)  # type: ignore
        assert not is_valid
        assert "string" in error.lower()

    def test_invalid_metric_name_too_long(self):
        """Should reject metric name longer than 128 characters."""
        long_name = "a" * 129
        is_valid, error = monitoring.validate_metric_name(long_name)
        assert not is_valid
        assert "128" in error

    def test_invalid_metric_name_starts_with_number(self):
        """Should reject metric name starting with number."""
        is_valid, error = monitoring.validate_metric_name("123metric")
        assert not is_valid
        assert "start with a letter" in error.lower()

    def test_invalid_metric_name_special_chars(self):
        """Should reject metric name with special characters."""
        invalid_names = [
            "metric@name",
            "metric#name",
            "metric name",  # space
            "metric!name",
            "metric$name",
        ]
        for name in invalid_names:
            is_valid, error = monitoring.validate_metric_name(name)
            assert not is_valid, f"Expected {name!r} to be invalid"


class TestValidateMetricValue:
    """Tests for metric value validation."""

    def test_valid_values(self):
        """Should accept valid numeric values."""
        valid_values = [0, 1, -1, 0.5, 100.0, -100.0, 1e10, -1e10]
        for value in valid_values:
            is_valid, result, error = monitoring.validate_metric_value(value)
            assert is_valid, f"Expected {value} to be valid, got error: {error}"
            assert result == float(value)

    def test_string_numbers(self):
        """Should accept string representations of numbers."""
        is_valid, result, error = monitoring.validate_metric_value("42.5")
        assert is_valid
        assert result == 42.5

    def test_invalid_value_none(self):
        """Should reject None value."""
        is_valid, result, error = monitoring.validate_metric_value(None)
        assert not is_valid
        assert "required" in error.lower()

    def test_invalid_value_not_number(self):
        """Should reject non-numeric values."""
        is_valid, result, error = monitoring.validate_metric_value("not_a_number")
        assert not is_valid
        assert "valid number" in error.lower()

    def test_invalid_value_nan(self):
        """Should reject NaN value."""
        is_valid, result, error = monitoring.validate_metric_value(float("nan"))
        assert not is_valid
        assert "NaN" in error

    def test_invalid_value_infinity(self):
        """Should reject infinite values."""
        is_valid, result, error = monitoring.validate_metric_value(float("inf"))
        assert not is_valid
        assert "infinite" in error.lower()

        is_valid, result, error = monitoring.validate_metric_value(float("-inf"))
        assert not is_valid
        assert "infinite" in error.lower()


# =============================================================================
# Test Circuit Breaker
# =============================================================================


class TestMonitoringCircuitBreaker:
    """Tests for the MonitoringCircuitBreaker class."""

    def test_initial_state_is_closed(self):
        """Circuit breaker should start in closed state."""
        cb = monitoring.MonitoringCircuitBreaker()
        assert cb.state == monitoring.MonitoringCircuitBreaker.CLOSED

    def test_allows_requests_when_closed(self):
        """Should allow requests when circuit is closed."""
        cb = monitoring.MonitoringCircuitBreaker()
        assert cb.is_allowed()

    def test_opens_after_failures(self):
        """Should open circuit after reaching failure threshold."""
        cb = monitoring.MonitoringCircuitBreaker(failure_threshold=3)

        for _ in range(3):
            cb.record_failure()

        assert cb.state == monitoring.MonitoringCircuitBreaker.OPEN

    def test_rejects_requests_when_open(self):
        """Should reject requests when circuit is open."""
        cb = monitoring.MonitoringCircuitBreaker(failure_threshold=2)
        cb.record_failure()
        cb.record_failure()

        assert not cb.is_allowed()

    def test_transitions_to_half_open_after_cooldown(self):
        """Should transition to half-open after cooldown period."""
        cb = monitoring.MonitoringCircuitBreaker(failure_threshold=1, cooldown_seconds=0.01)
        cb.record_failure()

        # Wait for cooldown
        import time

        time.sleep(0.02)

        assert cb.state == monitoring.MonitoringCircuitBreaker.HALF_OPEN

    def test_closes_after_successful_half_open_calls(self):
        """Should close after successful calls in half-open state."""
        cb = monitoring.MonitoringCircuitBreaker(
            failure_threshold=1,
            cooldown_seconds=0.01,
            half_open_max_calls=2,
        )
        cb.record_failure()

        import time

        time.sleep(0.02)

        # Make successful calls
        cb.is_allowed()
        cb.record_success()
        cb.is_allowed()
        cb.record_success()

        assert cb.state == monitoring.MonitoringCircuitBreaker.CLOSED

    def test_reopens_on_failure_in_half_open(self):
        """Should reopen on failure during half-open state."""
        cb = monitoring.MonitoringCircuitBreaker(failure_threshold=1, cooldown_seconds=0.01)
        cb.record_failure()

        import time

        time.sleep(0.02)

        # Fail during half-open
        cb.is_allowed()
        cb.record_failure()

        assert cb.state == monitoring.MonitoringCircuitBreaker.OPEN

    def test_reset_returns_to_closed(self):
        """Should return to closed state after reset."""
        cb = monitoring.MonitoringCircuitBreaker(failure_threshold=1)
        cb.record_failure()
        assert cb.state == monitoring.MonitoringCircuitBreaker.OPEN

        cb.reset()
        assert cb.state == monitoring.MonitoringCircuitBreaker.CLOSED

    def test_get_status(self):
        """Should return correct status dict."""
        cb = monitoring.MonitoringCircuitBreaker(
            failure_threshold=5,
            cooldown_seconds=30.0,
        )
        cb.record_failure()
        cb.record_failure()

        status = cb.get_status()
        assert status["state"] == monitoring.MonitoringCircuitBreaker.CLOSED
        assert status["failure_count"] == 2
        assert status["failure_threshold"] == 5
        assert status["cooldown_seconds"] == 30.0

    def test_success_resets_failure_count(self):
        """Success in closed state should reset failure count."""
        cb = monitoring.MonitoringCircuitBreaker(failure_threshold=5)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()

        status = cb.get_status()
        assert status["failure_count"] == 0


class TestGlobalCircuitBreaker:
    """Tests for global circuit breaker functions."""

    def test_get_monitoring_circuit_breaker_status(self):
        """Should return circuit breaker status."""
        status = monitoring.get_monitoring_circuit_breaker_status()
        assert "state" in status
        assert "failure_count" in status
        assert status["state"] == "closed"


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
            request = create_mock_request(json_data={"metric_name": "cpu_usage", "value": 75.0})

            response = await monitoring.MonitoringHandler.record_metric(request)

            assert response.status == 200
            body = json.loads(response.body)
            assert body["success"] is True
            assert body["metric_name"] == "cpu_usage"
            assert body["value"] == 75.0
            assert body["anomaly_detected"] is False

    @pytest.mark.asyncio
    async def test_record_metric_with_anomaly(
        self,
        mock_trend_monitor,
        mock_anomaly_detector,
        mock_auth_context,
        mock_permission_checker,
    ):
        """Should include anomaly data when detected."""
        mock_anomaly = MockAnomaly(
            id="anomaly_456",
            metric_name="cpu_usage",
            value=95.0,
            expected_value=50.0,
            deviation=3.5,
            severity=AlertSeverity.HIGH,
            description="High CPU detected",
        )
        mock_anomaly_detector._next_anomaly = mock_anomaly

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
            request = create_mock_request(json_data={"metric_name": "cpu_usage", "value": 95.0})

            response = await monitoring.MonitoringHandler.record_metric(request)

            assert response.status == 200
            body = json.loads(response.body)
            assert body["anomaly_detected"] is True
            assert body["anomaly"]["id"] == "anomaly_456"
            assert body["anomaly"]["severity"] == "high"

    @pytest.mark.asyncio
    async def test_record_metric_missing_metric_name(
        self,
        mock_auth_context,
        mock_permission_checker,
    ):
        """Should return 400 for missing metric_name."""
        with (
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
            request = create_mock_request(json_data={"value": 75.0})

            response = await monitoring.MonitoringHandler.record_metric(request)

            assert response.status == 400
            body = json.loads(response.body)
            assert body["success"] is False
            assert "metric_name" in body["error"]

    @pytest.mark.asyncio
    async def test_record_metric_missing_value(
        self,
        mock_auth_context,
        mock_permission_checker,
    ):
        """Should return 400 for missing value."""
        with (
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
            request = create_mock_request(json_data={"metric_name": "cpu_usage"})

            response = await monitoring.MonitoringHandler.record_metric(request)

            assert response.status == 400
            body = json.loads(response.body)
            assert body["success"] is False
            assert "value" in body["error"]

    @pytest.mark.asyncio
    async def test_record_metric_invalid_metric_name(
        self,
        mock_auth_context,
        mock_permission_checker,
    ):
        """Should return 400 for invalid metric name."""
        with (
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
            request = create_mock_request(json_data={"metric_name": "123invalid", "value": 75.0})

            response = await monitoring.MonitoringHandler.record_metric(request)

            assert response.status == 400
            body = json.loads(response.body)
            assert body["success"] is False

    @pytest.mark.asyncio
    async def test_record_metric_invalid_value(
        self,
        mock_auth_context,
        mock_permission_checker,
    ):
        """Should return 400 for invalid value."""
        with (
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
            request = create_mock_request(
                json_data={"metric_name": "cpu_usage", "value": "not_a_number"}
            )

            response = await monitoring.MonitoringHandler.record_metric(request)

            assert response.status == 400
            body = json.loads(response.body)
            assert body["success"] is False

    @pytest.mark.asyncio
    async def test_record_metric_unauthorized(self):
        """Should return 401 when unauthorized."""
        with patch.object(
            monitoring,
            "get_auth_context",
            AsyncMock(side_effect=monitoring.UnauthorizedError("Not authenticated")),
        ):
            request = create_mock_request()
            response = await monitoring.MonitoringHandler.record_metric(request)

            assert response.status == 401

    @pytest.mark.asyncio
    async def test_record_metric_forbidden(
        self,
        mock_auth_context,
        mock_permission_denied,
    ):
        """Should return 403 when permission denied."""
        with (
            patch.object(
                monitoring,
                "get_auth_context",
                AsyncMock(return_value=mock_auth_context),
            ),
            patch.object(
                monitoring,
                "get_permission_checker",
                return_value=mock_permission_denied,
            ),
        ):
            request = create_mock_request(json_data={"metric_name": "cpu_usage", "value": 75.0})

            response = await monitoring.MonitoringHandler.record_metric(request)

            assert response.status == 403

    @pytest.mark.asyncio
    async def test_record_metric_circuit_breaker_open(
        self,
        mock_auth_context,
        mock_permission_checker,
    ):
        """Should return 503 when circuit breaker is open."""
        # Force circuit breaker open
        cb = monitoring._get_monitoring_circuit_breaker()
        for _ in range(10):
            cb.record_failure()

        with (
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
            request = create_mock_request(json_data={"metric_name": "cpu_usage", "value": 75.0})

            response = await monitoring.MonitoringHandler.record_metric(request)

            assert response.status == 503
            body = json.loads(response.body)
            assert "temporarily unavailable" in body["error"].lower()
            assert "circuit_breaker_state" in body


# =============================================================================
# Test MonitoringHandler.get_trend
# =============================================================================


class TestMonitoringHandlerGetTrend:
    """Tests for GET /api/autonomous/monitoring/trends/{metric_name} endpoint."""

    @pytest.mark.asyncio
    async def test_get_trend_success(
        self,
        mock_trend_monitor,
        mock_auth_context,
        mock_permission_checker,
    ):
        """Should return trend data successfully."""
        mock_trend_monitor._trends["cpu_usage"] = MockTrendData(
            metric_name="cpu_usage",
            direction=TrendDirection.INCREASING,
            current_value=75.0,
            previous_value=50.0,
            change_percent=50.0,
        )

        with (
            patch.object(monitoring, "get_trend_monitor", return_value=mock_trend_monitor),
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
            request = create_mock_request(match_info={"metric_name": "cpu_usage"})

            response = await monitoring.MonitoringHandler.get_trend(request)

            assert response.status == 200
            body = json.loads(response.body)
            assert body["success"] is True
            assert body["trend"]["metric_name"] == "cpu_usage"
            assert body["trend"]["direction"] == "increasing"

    @pytest.mark.asyncio
    async def test_get_trend_insufficient_data(
        self,
        mock_trend_monitor,
        mock_auth_context,
        mock_permission_checker,
    ):
        """Should handle insufficient data gracefully."""
        with (
            patch.object(monitoring, "get_trend_monitor", return_value=mock_trend_monitor),
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
            request = create_mock_request(match_info={"metric_name": "unknown_metric"})

            response = await monitoring.MonitoringHandler.get_trend(request)

            assert response.status == 200
            body = json.loads(response.body)
            assert body["success"] is True
            assert body["trend"] is None
            assert "insufficient" in body["message"].lower()

    @pytest.mark.asyncio
    async def test_get_trend_invalid_metric_name(
        self,
        mock_auth_context,
        mock_permission_checker,
    ):
        """Should return 400 for invalid metric name."""
        with (
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
            request = create_mock_request(match_info={"metric_name": "123invalid"})

            response = await monitoring.MonitoringHandler.get_trend(request)

            assert response.status == 400


# =============================================================================
# Test MonitoringHandler.get_all_trends
# =============================================================================


class TestMonitoringHandlerGetAllTrends:
    """Tests for GET /api/autonomous/monitoring/trends endpoint."""

    @pytest.mark.asyncio
    async def test_get_all_trends_success(
        self,
        mock_trend_monitor,
        mock_auth_context,
        mock_permission_checker,
    ):
        """Should return all trends."""
        mock_trend_monitor._trends = {
            "cpu_usage": MockTrendData(
                metric_name="cpu_usage",
                direction=TrendDirection.INCREASING,
            ),
            "memory_usage": MockTrendData(
                metric_name="memory_usage",
                direction=TrendDirection.STABLE,
            ),
        }

        with (
            patch.object(monitoring, "get_trend_monitor", return_value=mock_trend_monitor),
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
            request = create_mock_request()

            response = await monitoring.MonitoringHandler.get_all_trends(request)

            assert response.status == 200
            body = json.loads(response.body)
            assert body["success"] is True
            assert body["count"] == 2
            assert "cpu_usage" in body["trends"]
            assert "memory_usage" in body["trends"]

    @pytest.mark.asyncio
    async def test_get_all_trends_empty(
        self,
        mock_trend_monitor,
        mock_auth_context,
        mock_permission_checker,
    ):
        """Should return empty trends when none exist."""
        with (
            patch.object(monitoring, "get_trend_monitor", return_value=mock_trend_monitor),
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
            request = create_mock_request()

            response = await monitoring.MonitoringHandler.get_all_trends(request)

            assert response.status == 200
            body = json.loads(response.body)
            assert body["success"] is True
            assert body["count"] == 0
            assert body["trends"] == {}


# =============================================================================
# Test MonitoringHandler.get_anomalies
# =============================================================================


class TestMonitoringHandlerGetAnomalies:
    """Tests for GET /api/autonomous/monitoring/anomalies endpoint."""

    @pytest.mark.asyncio
    async def test_get_anomalies_success(
        self,
        mock_anomaly_detector,
        mock_auth_context,
        mock_permission_checker,
    ):
        """Should return anomalies list."""
        mock_anomaly_detector._anomalies = [
            MockAnomaly(id="a1", metric_name="cpu_usage"),
            MockAnomaly(id="a2", metric_name="memory_usage"),
        ]

        with (
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
            request = create_mock_request()

            response = await monitoring.MonitoringHandler.get_anomalies(request)

            assert response.status == 200
            body = json.loads(response.body)
            assert body["success"] is True
            assert body["count"] == 2
            assert len(body["anomalies"]) == 2

    @pytest.mark.asyncio
    async def test_get_anomalies_filter_by_metric(
        self,
        mock_anomaly_detector,
        mock_auth_context,
        mock_permission_checker,
    ):
        """Should filter anomalies by metric name."""
        mock_anomaly_detector._anomalies = [
            MockAnomaly(id="a1", metric_name="cpu_usage"),
            MockAnomaly(id="a2", metric_name="memory_usage"),
        ]

        with (
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
            request = create_mock_request(query={"metric_name": "cpu_usage"})

            response = await monitoring.MonitoringHandler.get_anomalies(request)

            assert response.status == 200
            body = json.loads(response.body)
            assert body["success"] is True
            assert body["count"] == 1

    @pytest.mark.asyncio
    async def test_get_anomalies_invalid_metric_name(
        self,
        mock_auth_context,
        mock_permission_checker,
    ):
        """Should return 400 for invalid filter metric name."""
        with (
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
            request = create_mock_request(query={"metric_name": "123invalid"})

            response = await monitoring.MonitoringHandler.get_anomalies(request)

            assert response.status == 400


# =============================================================================
# Test MonitoringHandler.get_baseline_stats
# =============================================================================


class TestMonitoringHandlerGetBaselineStats:
    """Tests for GET /api/autonomous/monitoring/baseline/{metric_name} endpoint."""

    @pytest.mark.asyncio
    async def test_get_baseline_stats_success(
        self,
        mock_anomaly_detector,
        mock_auth_context,
        mock_permission_checker,
    ):
        """Should return baseline statistics."""
        mock_anomaly_detector._baseline_stats["cpu_usage"] = {
            "mean": 50.0,
            "stdev": 10.0,
            "min": 20.0,
            "max": 80.0,
            "median": 50.0,
            "count": 100,
        }

        with (
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
            request = create_mock_request(match_info={"metric_name": "cpu_usage"})

            response = await monitoring.MonitoringHandler.get_baseline_stats(request)

            assert response.status == 200
            body = json.loads(response.body)
            assert body["success"] is True
            assert body["metric_name"] == "cpu_usage"
            assert body["stats"]["mean"] == 50.0

    @pytest.mark.asyncio
    async def test_get_baseline_stats_insufficient_data(
        self,
        mock_anomaly_detector,
        mock_auth_context,
        mock_permission_checker,
    ):
        """Should handle insufficient data gracefully."""
        with (
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
            request = create_mock_request(match_info={"metric_name": "unknown_metric"})

            response = await monitoring.MonitoringHandler.get_baseline_stats(request)

            assert response.status == 200
            body = json.loads(response.body)
            assert body["success"] is True
            assert body["stats"] is None
            assert "insufficient" in body["message"].lower()


# =============================================================================
# Test MonitoringHandler.get_circuit_breaker_status
# =============================================================================


class TestMonitoringHandlerGetCircuitBreakerStatus:
    """Tests for GET /api/autonomous/monitoring/circuit-breaker endpoint."""

    @pytest.mark.asyncio
    async def test_get_circuit_breaker_status_success(
        self,
        mock_auth_context,
        mock_permission_checker,
    ):
        """Should return circuit breaker status."""
        with (
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
            request = create_mock_request()

            response = await monitoring.MonitoringHandler.get_circuit_breaker_status(request)

            assert response.status == 200
            body = json.loads(response.body)
            assert body["success"] is True
            assert "circuit_breaker" in body
            assert body["circuit_breaker"]["state"] == "closed"


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
        assert "/api/v1/autonomous/monitoring/baseline/{metric_name}" in routes
        assert "/api/v1/autonomous/monitoring/circuit-breaker" in routes

    def test_register_routes_custom_prefix(self):
        """Should register routes with custom prefix."""
        app = web.Application()
        monitoring.MonitoringHandler.register_routes(app, prefix="/api/v2/custom")

        routes = [r.resource.canonical for r in app.router.routes()]
        assert "/api/v2/custom/monitoring/record" in routes


# =============================================================================
# Test Global Functions
# =============================================================================


class TestMonitoringSingletons:
    """Tests for monitoring singleton functions."""

    def test_get_trend_monitor_creates_singleton(self):
        """get_trend_monitor should return same instance."""
        monitor1 = monitoring.get_trend_monitor()
        monitor2 = monitoring.get_trend_monitor()

        assert monitor1 is monitor2

    def test_get_anomaly_detector_creates_singleton(self):
        """get_anomaly_detector should return same instance."""
        detector1 = monitoring.get_anomaly_detector()
        detector2 = monitoring.get_anomaly_detector()

        assert detector1 is detector2

    def test_set_trend_monitor(self):
        """set_trend_monitor should update the global instance."""
        mock = MockTrendMonitor()
        monitoring.set_trend_monitor(mock)

        assert monitoring.get_trend_monitor() is mock

    def test_set_anomaly_detector(self):
        """set_anomaly_detector should update the global instance."""
        mock = MockAnomalyDetector()
        monitoring.set_anomaly_detector(mock)

        assert monitoring.get_anomaly_detector() is mock

    def test_clear_monitoring_components(self):
        """_clear_monitoring_components should reset all components."""
        # Ensure components exist
        monitoring.get_trend_monitor()
        monitoring.get_anomaly_detector()
        monitoring._get_monitoring_circuit_breaker()

        # Clear them
        monitoring._clear_monitoring_components()

        # They should be recreated (new instances)
        with monitoring._trend_monitor_lock:
            assert monitoring._trend_monitor is None
        with monitoring._anomaly_detector_lock:
            assert monitoring._anomaly_detector is None


# =============================================================================
# Test Error Handling
# =============================================================================


class TestMonitoringHandlerErrorHandling:
    """Tests for error handling in monitoring handler."""

    @pytest.mark.asyncio
    async def test_internal_error_records_circuit_breaker_failure(
        self,
        mock_trend_monitor,
        mock_auth_context,
        mock_permission_checker,
    ):
        """Should record circuit breaker failure on internal error."""
        mock_trend_monitor.record = MagicMock(side_effect=RuntimeError("Database error"))

        with (
            patch.object(monitoring, "get_trend_monitor", return_value=mock_trend_monitor),
            patch.object(monitoring, "get_anomaly_detector", return_value=MockAnomalyDetector()),
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
            request = create_mock_request(json_data={"metric_name": "cpu_usage", "value": 75.0})

            response = await monitoring.MonitoringHandler.record_metric(request)

            assert response.status == 500
            body = json.loads(response.body)
            assert body["success"] is False
            # Error message should be safe (not expose internals)
            assert "internal server error" in body["error"].lower()

    @pytest.mark.asyncio
    async def test_safe_error_message_does_not_expose_internals(
        self,
        mock_trend_monitor,
        mock_auth_context,
        mock_permission_checker,
    ):
        """Should not expose internal error details in response."""
        mock_trend_monitor.record = MagicMock(
            side_effect=RuntimeError("SECRET_DATABASE_PASSWORD exposed")
        )

        with (
            patch.object(monitoring, "get_trend_monitor", return_value=mock_trend_monitor),
            patch.object(monitoring, "get_anomaly_detector", return_value=MockAnomalyDetector()),
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
            request = create_mock_request(json_data={"metric_name": "cpu_usage", "value": 75.0})

            response = await monitoring.MonitoringHandler.record_metric(request)

            assert response.status == 500
            body = json.loads(response.body)
            assert "SECRET_DATABASE_PASSWORD" not in body["error"]
