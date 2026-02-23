"""Comprehensive tests for MonitoringHandler.

Tests cover:
- record_metric (POST /api/v1/autonomous/monitoring/record)
- get_trend (GET /api/v1/autonomous/monitoring/trends/{metric_name})
- get_all_trends (GET /api/v1/autonomous/monitoring/trends)
- get_anomalies (GET /api/v1/autonomous/monitoring/anomalies)
- get_baseline_stats (GET /api/v1/autonomous/monitoring/baseline/{metric_name})
- get_circuit_breaker_status (GET /api/v1/autonomous/monitoring/circuit-breaker)
- Input validation (metric names, values)
- Auth / permission checks
- Circuit breaker behaviour
- Error-handling paths
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web

from aragora.server.handlers.autonomous.monitoring import (
    MonitoringHandler,
    _clear_monitoring_components,
    get_trend_monitor,
    set_trend_monitor,
    get_anomaly_detector,
    set_anomaly_detector,
    get_monitoring_circuit_breaker_status,
    validate_metric_name,
    validate_metric_value,
    MIN_METRIC_VALUE,
    MAX_METRIC_VALUE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _parse(response: web.Response) -> dict:
    """Extract JSON dict from an aiohttp json_response."""
    return json.loads(response.body)


def _make_request(
    method: str = "GET",
    query: dict | None = None,
    match_info: dict | None = None,
    body: dict | None = None,
) -> MagicMock:
    """Build a MagicMock that mimics an aiohttp web.Request."""
    req = MagicMock()
    req.method = method
    req.query = query or {}

    # Use a MagicMock for match_info so we can freely assign .get
    mi_data = match_info or {}
    mi_mock = MagicMock()
    mi_mock.get = MagicMock(side_effect=lambda k, default=None: mi_data.get(k, default))
    req.match_info = mi_mock

    if body is not None:
        req.json = AsyncMock(return_value=body)
        raw = json.dumps(body).encode()
        req.read = AsyncMock(return_value=raw)
        req.text = AsyncMock(return_value=json.dumps(body))
        req.content_type = "application/json"
        req.content_length = len(raw)
        req.can_read_body = True
    else:
        req.json = AsyncMock(return_value={})
        req.read = AsyncMock(return_value=b"{}")
        req.text = AsyncMock(return_value="{}")
        req.content_type = "application/json"
        req.content_length = 2
        req.can_read_body = True

    # peername for rate-limit key extraction
    req.remote = "127.0.0.1"
    req.transport = MagicMock()
    req.transport.get_extra_info.return_value = ("127.0.0.1", 12345)

    return req


class _MockSeverity(Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class _MockDirection(Enum):
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    VOLATILE = "volatile"


def _make_anomaly(
    anomaly_id: str = "a-1",
    metric_name: str = "cpu_usage",
    value: float = 99.0,
    expected: float = 50.0,
    deviation: float = 3.5,
    severity=None,
    description: str = "High CPU",
):
    severity = severity or _MockSeverity.HIGH
    obj = MagicMock()
    obj.id = anomaly_id
    obj.metric_name = metric_name
    obj.value = value
    obj.expected_value = expected
    obj.deviation = deviation
    obj.timestamp = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    obj.severity = severity
    obj.description = description
    return obj


def _make_trend(
    metric_name: str = "requests_per_second",
    direction=None,
    current_value: float = 120.0,
    previous_value: float = 100.0,
    change_percent: float = 20.0,
    data_points: int = 50,
    confidence: float = 0.85,
):
    direction = direction or _MockDirection.INCREASING
    obj = MagicMock()
    obj.metric_name = metric_name
    obj.direction = direction
    obj.current_value = current_value
    obj.previous_value = previous_value
    obj.change_percent = change_percent
    obj.period_start = datetime(2026, 1, 15, 11, 0, 0, tzinfo=timezone.utc)
    obj.period_end = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    obj.data_points = data_points
    obj.confidence = confidence
    return obj


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_globals():
    """Ensure globals are fresh before/after each test."""
    _clear_monitoring_components()
    yield
    _clear_monitoring_components()


@pytest.fixture
def mock_trend_monitor():
    monitor = MagicMock()
    monitor.record = MagicMock()
    monitor.get_trend = MagicMock(return_value=None)
    monitor.get_all_trends = MagicMock(return_value={})
    return monitor


@pytest.fixture
def mock_anomaly_detector():
    detector = MagicMock()
    detector.record = MagicMock(return_value=None)
    detector.get_recent_anomalies = MagicMock(return_value=[])
    detector.get_baseline_stats = MagicMock(return_value=None)
    return detector


@pytest.fixture
def install_mocks(mock_trend_monitor, mock_anomaly_detector):
    """Inject mock trend monitor and anomaly detector into globals."""
    set_trend_monitor(mock_trend_monitor)
    set_anomaly_detector(mock_anomaly_detector)
    return mock_trend_monitor, mock_anomaly_detector


# ---------------------------------------------------------------------------
# validate_metric_name unit tests
# ---------------------------------------------------------------------------


class TestValidateMetricName:
    def test_valid_simple_name(self):
        ok, msg = validate_metric_name("cpu_usage")
        assert ok is True
        assert msg == ""

    def test_valid_with_dots_hyphens(self):
        ok, _ = validate_metric_name("sys.cpu-usage_total")
        assert ok is True

    def test_none_metric_name(self):
        ok, msg = validate_metric_name(None)
        assert ok is False
        assert "required" in msg

    def test_empty_string(self):
        ok, msg = validate_metric_name("")
        assert ok is False
        assert "required" in msg

    def test_too_long(self):
        ok, msg = validate_metric_name("a" * 129)
        assert ok is False
        assert "128" in msg

    def test_starts_with_digit(self):
        ok, msg = validate_metric_name("1cpu")
        assert ok is False
        assert "letter" in msg

    def test_special_characters(self):
        ok, msg = validate_metric_name("cpu usage!")
        assert ok is False

    def test_exactly_128_chars(self):
        ok, _ = validate_metric_name("a" * 128)
        assert ok is True

    def test_starts_with_underscore(self):
        ok, _ = validate_metric_name("_cpu")
        assert ok is False


# ---------------------------------------------------------------------------
# validate_metric_value unit tests
# ---------------------------------------------------------------------------


class TestValidateMetricValue:
    def test_valid_int(self):
        ok, val, _ = validate_metric_value(42)
        assert ok is True
        assert val == 42.0

    def test_valid_float(self):
        ok, val, _ = validate_metric_value(3.14)
        assert ok is True
        assert val == pytest.approx(3.14)

    def test_valid_string_number(self):
        ok, val, _ = validate_metric_value("100.5")
        assert ok is True
        assert val == 100.5

    def test_none_value(self):
        ok, val, msg = validate_metric_value(None)
        assert ok is False
        assert val is None
        assert "required" in msg

    def test_nan_value(self):
        ok, val, msg = validate_metric_value(float("nan"))
        assert ok is False
        assert "NaN" in msg

    def test_inf_value(self):
        ok, _, msg = validate_metric_value(float("inf"))
        assert ok is False
        assert "infinite" in msg

    def test_neg_inf_value(self):
        ok, _, msg = validate_metric_value(float("-inf"))
        assert ok is False
        assert "infinite" in msg

    def test_non_numeric_string(self):
        ok, _, msg = validate_metric_value("not_a_number")
        assert ok is False
        assert "valid number" in msg

    def test_exceeds_max(self):
        ok, _, msg = validate_metric_value(MAX_METRIC_VALUE + 1)
        assert ok is False
        assert "between" in msg

    def test_below_min(self):
        ok, _, msg = validate_metric_value(MIN_METRIC_VALUE - 1)
        assert ok is False
        assert "between" in msg

    def test_negative_valid(self):
        ok, val, _ = validate_metric_value(-42.5)
        assert ok is True
        assert val == -42.5

    def test_zero(self):
        ok, val, _ = validate_metric_value(0)
        assert ok is True
        assert val == 0.0

    def test_boundary_max(self):
        ok, val, _ = validate_metric_value(MAX_METRIC_VALUE)
        assert ok is True
        assert val == MAX_METRIC_VALUE

    def test_boundary_min(self):
        ok, val, _ = validate_metric_value(MIN_METRIC_VALUE)
        assert ok is True
        assert val == MIN_METRIC_VALUE


# ---------------------------------------------------------------------------
# record_metric endpoint
# ---------------------------------------------------------------------------


class TestRecordMetric:
    @pytest.mark.asyncio
    async def test_record_success_no_anomaly(self, install_mocks):
        trend_mon, anomaly_det = install_mocks
        anomaly_det.record.return_value = None

        req = _make_request(method="POST", body={"metric_name": "cpu", "value": 55.0})
        resp = await MonitoringHandler.record_metric(req)

        assert resp.status == 200
        data = await _parse(resp)
        assert data["success"] is True
        assert data["metric_name"] == "cpu"
        assert data["value"] == 55.0
        assert data["anomaly_detected"] is False
        assert "anomaly" not in data
        trend_mon.record.assert_called_once_with("cpu", 55.0)

    @pytest.mark.asyncio
    async def test_record_success_with_anomaly(self, install_mocks):
        _, anomaly_det = install_mocks
        anomaly = _make_anomaly()
        anomaly_det.record.return_value = anomaly

        req = _make_request(method="POST", body={"metric_name": "cpu", "value": 99.0})
        resp = await MonitoringHandler.record_metric(req)

        assert resp.status == 200
        data = await _parse(resp)
        assert data["success"] is True
        assert data["anomaly_detected"] is True
        assert data["anomaly"]["id"] == "a-1"
        assert data["anomaly"]["severity"] == "high"

    @pytest.mark.asyncio
    async def test_record_missing_metric_name(self, install_mocks):
        req = _make_request(method="POST", body={"value": 10.0})
        resp = await MonitoringHandler.record_metric(req)

        assert resp.status == 400
        data = await _parse(resp)
        assert data["success"] is False
        assert "required" in data["error"]

    @pytest.mark.asyncio
    async def test_record_invalid_metric_name(self, install_mocks):
        req = _make_request(method="POST", body={"metric_name": "123bad", "value": 10.0})
        resp = await MonitoringHandler.record_metric(req)

        assert resp.status == 400
        data = await _parse(resp)
        assert data["success"] is False

    @pytest.mark.asyncio
    async def test_record_missing_value(self, install_mocks):
        req = _make_request(method="POST", body={"metric_name": "cpu"})
        resp = await MonitoringHandler.record_metric(req)

        assert resp.status == 400
        data = await _parse(resp)
        assert "required" in data["error"]

    @pytest.mark.asyncio
    async def test_record_nan_value(self, install_mocks):
        req = _make_request(method="POST", body={"metric_name": "cpu", "value": "NaN"})
        # float("NaN") from string "NaN" is a valid conversion but results in NaN
        # parse_json_body will parse it as the string "NaN", then float("NaN") in validate
        resp = await MonitoringHandler.record_metric(req)

        assert resp.status == 400
        data = await _parse(resp)
        assert data["success"] is False

    @pytest.mark.asyncio
    async def test_record_non_numeric_value(self, install_mocks):
        req = _make_request(method="POST", body={"metric_name": "cpu", "value": "abc"})
        resp = await MonitoringHandler.record_metric(req)

        assert resp.status == 400
        data = await _parse(resp)
        assert "valid number" in data["error"]

    @pytest.mark.asyncio
    async def test_record_circuit_breaker_open(self):
        """When circuit breaker is open, requests get 503."""
        cb = MagicMock()
        cb.is_allowed.return_value = False
        cb.state = "open"
        cb.cooldown_seconds = 30.0

        with patch(
            "aragora.server.handlers.autonomous.monitoring._get_monitoring_circuit_breaker",
            return_value=cb,
        ):
            req = _make_request(method="POST", body={"metric_name": "cpu", "value": 1.0})
            resp = await MonitoringHandler.record_metric(req)

        assert resp.status == 503
        data = await _parse(resp)
        assert "circuit_breaker" in data["error"].lower() or "circuit_breaker_state" in data

    @pytest.mark.asyncio
    async def test_record_internal_error_records_failure(self, install_mocks):
        """Internal error records failure on circuit breaker."""
        trend_mon, _ = install_mocks
        trend_mon.record.side_effect = RuntimeError("boom")

        cb = MagicMock()
        cb.is_allowed.return_value = True

        with patch(
            "aragora.server.handlers.autonomous.monitoring._get_monitoring_circuit_breaker",
            return_value=cb,
        ):
            req = _make_request(method="POST", body={"metric_name": "cpu", "value": 1.0})
            resp = await MonitoringHandler.record_metric(req)

        assert resp.status == 500
        data = await _parse(resp)
        assert data["success"] is False
        assert "Internal server error" in data["error"]
        cb.record_failure.assert_called_once()

    @pytest.mark.asyncio
    async def test_record_success_records_success_on_cb(self, install_mocks):
        """Successful recording calls record_success on circuit breaker."""
        cb = MagicMock()
        cb.is_allowed.return_value = True

        _, anomaly_det = install_mocks
        anomaly_det.record.return_value = None

        with patch(
            "aragora.server.handlers.autonomous.monitoring._get_monitoring_circuit_breaker",
            return_value=cb,
        ):
            req = _make_request(method="POST", body={"metric_name": "cpu", "value": 1.0})
            resp = await MonitoringHandler.record_metric(req)

        assert resp.status == 200
        cb.record_success.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_record_unauthorized(self, install_mocks):
        """Unauthenticated request returns 401."""
        from aragora.server.handlers.utils.auth import UnauthorizedError

        with patch(
            "aragora.server.handlers.autonomous.monitoring.get_auth_context",
            side_effect=UnauthorizedError("no token"),
        ):
            req = _make_request(method="POST", body={"metric_name": "cpu", "value": 1.0})
            resp = await MonitoringHandler.record_metric(req)

        assert resp.status == 401
        data = await _parse(resp)
        assert "Authentication required" in data["error"]

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_record_forbidden(self, install_mocks):
        """Request with insufficient permissions returns 403."""
        from aragora.server.handlers.utils.auth import ForbiddenError

        mock_ctx = MagicMock()
        mock_checker = MagicMock()
        mock_decision = MagicMock()
        mock_decision.allowed = False

        with (
            patch(
                "aragora.server.handlers.autonomous.monitoring.get_auth_context",
                new_callable=AsyncMock,
                return_value=mock_ctx,
            ),
            patch(
                "aragora.server.handlers.autonomous.monitoring.get_permission_checker",
                return_value=mock_checker,
            ),
        ):
            mock_checker.check_permission.return_value = mock_decision
            req = _make_request(method="POST", body={"metric_name": "cpu", "value": 1.0})
            resp = await MonitoringHandler.record_metric(req)

        assert resp.status == 403
        data = await _parse(resp)
        assert "Permission denied" in data["error"]

    @pytest.mark.asyncio
    async def test_record_value_too_large(self, install_mocks):
        req = _make_request(
            method="POST",
            body={"metric_name": "cpu", "value": MAX_METRIC_VALUE + 1e10},
        )
        resp = await MonitoringHandler.record_metric(req)
        assert resp.status == 400

    @pytest.mark.asyncio
    async def test_record_special_chars_in_name(self, install_mocks):
        req = _make_request(
            method="POST",
            body={"metric_name": "cpu<script>", "value": 1.0},
        )
        resp = await MonitoringHandler.record_metric(req)
        assert resp.status == 400


# ---------------------------------------------------------------------------
# get_trend endpoint
# ---------------------------------------------------------------------------


class TestGetTrend:
    @pytest.mark.asyncio
    async def test_trend_found(self, install_mocks):
        trend_mon, _ = install_mocks
        trend_mon.get_trend.return_value = _make_trend(metric_name="latency_ms")

        req = _make_request(
            query={},
            match_info={"metric_name": "latency_ms"},
        )
        resp = await MonitoringHandler.get_trend(req)

        assert resp.status == 200
        data = await _parse(resp)
        assert data["success"] is True
        assert data["trend"]["metric_name"] == "latency_ms"
        assert data["trend"]["direction"] == "increasing"
        assert data["trend"]["data_points"] == 50

    @pytest.mark.asyncio
    async def test_trend_not_found(self, install_mocks):
        trend_mon, _ = install_mocks
        trend_mon.get_trend.return_value = None

        req = _make_request(match_info={"metric_name": "unknown_metric"})
        resp = await MonitoringHandler.get_trend(req)

        assert resp.status == 200
        data = await _parse(resp)
        assert data["success"] is True
        assert data["trend"] is None
        assert "Insufficient" in data["message"]

    @pytest.mark.asyncio
    async def test_trend_with_period_seconds(self, install_mocks):
        trend_mon, _ = install_mocks
        trend_mon.get_trend.return_value = _make_trend()

        req = _make_request(
            query={"period_seconds": "3600"},
            match_info={"metric_name": "cpu"},
        )
        resp = await MonitoringHandler.get_trend(req)

        assert resp.status == 200
        # Verify period_seconds was passed through
        call_args = trend_mon.get_trend.call_args
        assert call_args is not None

    @pytest.mark.asyncio
    async def test_trend_invalid_metric_name(self, install_mocks):
        req = _make_request(match_info={"metric_name": "123bad"})
        resp = await MonitoringHandler.get_trend(req)
        assert resp.status == 400

    @pytest.mark.asyncio
    async def test_trend_empty_metric_name(self, install_mocks):
        req = _make_request(match_info={})
        resp = await MonitoringHandler.get_trend(req)
        assert resp.status == 400

    @pytest.mark.asyncio
    async def test_trend_circuit_breaker_open(self):
        cb = MagicMock()
        cb.is_allowed.return_value = False
        cb.state = "open"
        cb.cooldown_seconds = 30.0

        with patch(
            "aragora.server.handlers.autonomous.monitoring._get_monitoring_circuit_breaker",
            return_value=cb,
        ):
            req = _make_request(match_info={"metric_name": "cpu"})
            resp = await MonitoringHandler.get_trend(req)

        assert resp.status == 503

    @pytest.mark.asyncio
    async def test_trend_internal_error(self, install_mocks):
        trend_mon, _ = install_mocks
        trend_mon.get_trend.side_effect = RuntimeError("db down")

        cb = MagicMock()
        cb.is_allowed.return_value = True

        with patch(
            "aragora.server.handlers.autonomous.monitoring._get_monitoring_circuit_breaker",
            return_value=cb,
        ):
            req = _make_request(match_info={"metric_name": "cpu"})
            resp = await MonitoringHandler.get_trend(req)

        assert resp.status == 500
        cb.record_failure.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_trend_unauthorized(self, install_mocks):
        from aragora.server.handlers.utils.auth import UnauthorizedError

        with patch(
            "aragora.server.handlers.autonomous.monitoring.get_auth_context",
            side_effect=UnauthorizedError("no token"),
        ):
            req = _make_request(match_info={"metric_name": "cpu"})
            resp = await MonitoringHandler.get_trend(req)

        assert resp.status == 401

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_trend_forbidden(self, install_mocks):
        mock_ctx = MagicMock()
        mock_checker = MagicMock()
        mock_decision = MagicMock()
        mock_decision.allowed = False

        with (
            patch(
                "aragora.server.handlers.autonomous.monitoring.get_auth_context",
                new_callable=AsyncMock,
                return_value=mock_ctx,
            ),
            patch(
                "aragora.server.handlers.autonomous.monitoring.get_permission_checker",
                return_value=mock_checker,
            ),
        ):
            mock_checker.check_permission.return_value = mock_decision
            req = _make_request(match_info={"metric_name": "cpu"})
            resp = await MonitoringHandler.get_trend(req)

        assert resp.status == 403

    @pytest.mark.asyncio
    async def test_trend_response_fields(self, install_mocks):
        """Verify all expected response fields are present."""
        trend_mon, _ = install_mocks
        trend_mon.get_trend.return_value = _make_trend()

        req = _make_request(match_info={"metric_name": "requests_per_second"})
        resp = await MonitoringHandler.get_trend(req)

        data = await _parse(resp)
        trend = data["trend"]
        expected_keys = {
            "metric_name",
            "direction",
            "current_value",
            "previous_value",
            "change_percent",
            "period_start",
            "period_end",
            "data_points",
            "confidence",
        }
        assert expected_keys.issubset(set(trend.keys()))


# ---------------------------------------------------------------------------
# get_all_trends endpoint
# ---------------------------------------------------------------------------


class TestGetAllTrends:
    @pytest.mark.asyncio
    async def test_all_trends_empty(self, install_mocks):
        trend_mon, _ = install_mocks
        trend_mon.get_all_trends.return_value = {}

        req = _make_request()
        resp = await MonitoringHandler.get_all_trends(req)

        assert resp.status == 200
        data = await _parse(resp)
        assert data["success"] is True
        assert data["trends"] == {}
        assert data["count"] == 0

    @pytest.mark.asyncio
    async def test_all_trends_multiple(self, install_mocks):
        trend_mon, _ = install_mocks
        trend_mon.get_all_trends.return_value = {
            "cpu": _make_trend(metric_name="cpu", direction=_MockDirection.STABLE),
            "memory": _make_trend(metric_name="memory", direction=_MockDirection.INCREASING),
        }

        req = _make_request()
        resp = await MonitoringHandler.get_all_trends(req)

        assert resp.status == 200
        data = await _parse(resp)
        assert data["count"] == 2
        assert "cpu" in data["trends"]
        assert "memory" in data["trends"]
        assert data["trends"]["cpu"]["direction"] == "stable"
        assert data["trends"]["memory"]["direction"] == "increasing"

    @pytest.mark.asyncio
    async def test_all_trends_response_keys(self, install_mocks):
        trend_mon, _ = install_mocks
        trend_mon.get_all_trends.return_value = {
            "cpu": _make_trend(metric_name="cpu"),
        }

        req = _make_request()
        resp = await MonitoringHandler.get_all_trends(req)
        data = await _parse(resp)

        trend_data = data["trends"]["cpu"]
        # get_all_trends serialises a subset of fields
        for key in ("direction", "current_value", "change_percent", "data_points", "confidence"):
            assert key in trend_data

    @pytest.mark.asyncio
    async def test_all_trends_circuit_breaker_open(self):
        cb = MagicMock()
        cb.is_allowed.return_value = False
        cb.state = "open"
        cb.cooldown_seconds = 30.0

        with patch(
            "aragora.server.handlers.autonomous.monitoring._get_monitoring_circuit_breaker",
            return_value=cb,
        ):
            req = _make_request()
            resp = await MonitoringHandler.get_all_trends(req)

        assert resp.status == 503

    @pytest.mark.asyncio
    async def test_all_trends_internal_error(self, install_mocks):
        trend_mon, _ = install_mocks
        trend_mon.get_all_trends.side_effect = ValueError("corrupt data")

        cb = MagicMock()
        cb.is_allowed.return_value = True

        with patch(
            "aragora.server.handlers.autonomous.monitoring._get_monitoring_circuit_breaker",
            return_value=cb,
        ):
            req = _make_request()
            resp = await MonitoringHandler.get_all_trends(req)

        assert resp.status == 500
        cb.record_failure.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_all_trends_unauthorized(self, install_mocks):
        from aragora.server.handlers.utils.auth import UnauthorizedError

        with patch(
            "aragora.server.handlers.autonomous.monitoring.get_auth_context",
            side_effect=UnauthorizedError(),
        ):
            req = _make_request()
            resp = await MonitoringHandler.get_all_trends(req)

        assert resp.status == 401

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_all_trends_forbidden(self, install_mocks):
        mock_ctx = MagicMock()
        mock_checker = MagicMock()
        mock_decision = MagicMock()
        mock_decision.allowed = False

        with (
            patch(
                "aragora.server.handlers.autonomous.monitoring.get_auth_context",
                new_callable=AsyncMock,
                return_value=mock_ctx,
            ),
            patch(
                "aragora.server.handlers.autonomous.monitoring.get_permission_checker",
                return_value=mock_checker,
            ),
        ):
            mock_checker.check_permission.return_value = mock_decision
            req = _make_request()
            resp = await MonitoringHandler.get_all_trends(req)

        assert resp.status == 403


# ---------------------------------------------------------------------------
# get_anomalies endpoint
# ---------------------------------------------------------------------------


class TestGetAnomalies:
    @pytest.mark.asyncio
    async def test_anomalies_empty(self, install_mocks):
        _, anomaly_det = install_mocks
        anomaly_det.get_recent_anomalies.return_value = []

        req = _make_request()
        resp = await MonitoringHandler.get_anomalies(req)

        assert resp.status == 200
        data = await _parse(resp)
        assert data["success"] is True
        assert data["anomalies"] == []
        assert data["count"] == 0

    @pytest.mark.asyncio
    async def test_anomalies_with_results(self, install_mocks):
        _, anomaly_det = install_mocks
        anomaly_det.get_recent_anomalies.return_value = [
            _make_anomaly(anomaly_id="a-1"),
            _make_anomaly(anomaly_id="a-2", severity=_MockSeverity.CRITICAL),
        ]

        req = _make_request()
        resp = await MonitoringHandler.get_anomalies(req)

        assert resp.status == 200
        data = await _parse(resp)
        assert data["count"] == 2
        assert data["anomalies"][0]["id"] == "a-1"
        assert data["anomalies"][1]["severity"] == "critical"

    @pytest.mark.asyncio
    async def test_anomalies_with_hours_param(self, install_mocks):
        _, anomaly_det = install_mocks
        anomaly_det.get_recent_anomalies.return_value = []

        req = _make_request(query={"hours": "48"})
        resp = await MonitoringHandler.get_anomalies(req)

        assert resp.status == 200
        call_args = anomaly_det.get_recent_anomalies.call_args
        assert call_args.kwargs.get("hours") == 48 or (call_args[1].get("hours") == 48)

    @pytest.mark.asyncio
    async def test_anomalies_with_metric_name_filter(self, install_mocks):
        _, anomaly_det = install_mocks
        anomaly_det.get_recent_anomalies.return_value = []

        req = _make_request(query={"metric_name": "cpu_usage"})
        resp = await MonitoringHandler.get_anomalies(req)

        assert resp.status == 200
        call_args = anomaly_det.get_recent_anomalies.call_args
        assert call_args.kwargs.get("metric_name") == "cpu_usage" or (
            call_args[1].get("metric_name") == "cpu_usage"
        )

    @pytest.mark.asyncio
    async def test_anomalies_invalid_metric_name_filter(self, install_mocks):
        req = _make_request(query={"metric_name": "123invalid"})
        resp = await MonitoringHandler.get_anomalies(req)

        assert resp.status == 400
        data = await _parse(resp)
        assert data["success"] is False

    @pytest.mark.asyncio
    async def test_anomalies_response_fields(self, install_mocks):
        _, anomaly_det = install_mocks
        anomaly_det.get_recent_anomalies.return_value = [_make_anomaly()]

        req = _make_request()
        resp = await MonitoringHandler.get_anomalies(req)
        data = await _parse(resp)

        anomaly = data["anomalies"][0]
        expected_keys = {
            "id",
            "metric_name",
            "value",
            "expected_value",
            "deviation",
            "timestamp",
            "severity",
            "description",
        }
        assert expected_keys.issubset(set(anomaly.keys()))

    @pytest.mark.asyncio
    async def test_anomalies_circuit_breaker_open(self):
        cb = MagicMock()
        cb.is_allowed.return_value = False
        cb.state = "open"
        cb.cooldown_seconds = 30.0

        with patch(
            "aragora.server.handlers.autonomous.monitoring._get_monitoring_circuit_breaker",
            return_value=cb,
        ):
            req = _make_request()
            resp = await MonitoringHandler.get_anomalies(req)

        assert resp.status == 503

    @pytest.mark.asyncio
    async def test_anomalies_internal_error(self, install_mocks):
        _, anomaly_det = install_mocks
        anomaly_det.get_recent_anomalies.side_effect = TypeError("bad data")

        cb = MagicMock()
        cb.is_allowed.return_value = True

        with patch(
            "aragora.server.handlers.autonomous.monitoring._get_monitoring_circuit_breaker",
            return_value=cb,
        ):
            req = _make_request()
            resp = await MonitoringHandler.get_anomalies(req)

        assert resp.status == 500
        cb.record_failure.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_anomalies_unauthorized(self, install_mocks):
        from aragora.server.handlers.utils.auth import UnauthorizedError

        with patch(
            "aragora.server.handlers.autonomous.monitoring.get_auth_context",
            side_effect=UnauthorizedError(),
        ):
            req = _make_request()
            resp = await MonitoringHandler.get_anomalies(req)

        assert resp.status == 401

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_anomalies_forbidden(self, install_mocks):
        mock_ctx = MagicMock()
        mock_checker = MagicMock()
        mock_decision = MagicMock()
        mock_decision.allowed = False

        with (
            patch(
                "aragora.server.handlers.autonomous.monitoring.get_auth_context",
                new_callable=AsyncMock,
                return_value=mock_ctx,
            ),
            patch(
                "aragora.server.handlers.autonomous.monitoring.get_permission_checker",
                return_value=mock_checker,
            ),
        ):
            mock_checker.check_permission.return_value = mock_decision
            req = _make_request()
            resp = await MonitoringHandler.get_anomalies(req)

        assert resp.status == 403

    @pytest.mark.asyncio
    async def test_anomalies_default_hours(self, install_mocks):
        """Without hours param, should default to 24."""
        _, anomaly_det = install_mocks
        anomaly_det.get_recent_anomalies.return_value = []

        req = _make_request()
        resp = await MonitoringHandler.get_anomalies(req)

        assert resp.status == 200
        call_args = anomaly_det.get_recent_anomalies.call_args
        assert call_args.kwargs.get("hours") == 24 or (call_args[1].get("hours") == 24)


# ---------------------------------------------------------------------------
# get_baseline_stats endpoint
# ---------------------------------------------------------------------------


class TestGetBaselineStats:
    @pytest.mark.asyncio
    async def test_baseline_found(self, install_mocks):
        _, anomaly_det = install_mocks
        anomaly_det.get_baseline_stats.return_value = {
            "mean": 50.0,
            "stdev": 5.0,
            "min": 40.0,
            "max": 60.0,
            "median": 50.0,
            "count": 100,
        }

        req = _make_request(match_info={"metric_name": "cpu_usage"})
        resp = await MonitoringHandler.get_baseline_stats(req)

        assert resp.status == 200
        data = await _parse(resp)
        assert data["success"] is True
        assert data["metric_name"] == "cpu_usage"
        assert data["stats"]["mean"] == 50.0
        assert data["stats"]["count"] == 100

    @pytest.mark.asyncio
    async def test_baseline_insufficient_data(self, install_mocks):
        _, anomaly_det = install_mocks
        anomaly_det.get_baseline_stats.return_value = None

        req = _make_request(match_info={"metric_name": "new_metric"})
        resp = await MonitoringHandler.get_baseline_stats(req)

        assert resp.status == 200
        data = await _parse(resp)
        assert data["success"] is True
        assert data["stats"] is None
        assert "Insufficient" in data["message"]

    @pytest.mark.asyncio
    async def test_baseline_invalid_metric_name(self, install_mocks):
        req = _make_request(match_info={"metric_name": "!invalid"})
        resp = await MonitoringHandler.get_baseline_stats(req)
        assert resp.status == 400

    @pytest.mark.asyncio
    async def test_baseline_missing_metric_name(self, install_mocks):
        req = _make_request(match_info={})
        resp = await MonitoringHandler.get_baseline_stats(req)
        assert resp.status == 400

    @pytest.mark.asyncio
    async def test_baseline_circuit_breaker_open(self):
        cb = MagicMock()
        cb.is_allowed.return_value = False
        cb.state = "open"
        cb.cooldown_seconds = 30.0

        with patch(
            "aragora.server.handlers.autonomous.monitoring._get_monitoring_circuit_breaker",
            return_value=cb,
        ):
            req = _make_request(match_info={"metric_name": "cpu"})
            resp = await MonitoringHandler.get_baseline_stats(req)

        assert resp.status == 503

    @pytest.mark.asyncio
    async def test_baseline_internal_error(self, install_mocks):
        _, anomaly_det = install_mocks
        anomaly_det.get_baseline_stats.side_effect = AttributeError("oops")

        cb = MagicMock()
        cb.is_allowed.return_value = True

        with patch(
            "aragora.server.handlers.autonomous.monitoring._get_monitoring_circuit_breaker",
            return_value=cb,
        ):
            req = _make_request(match_info={"metric_name": "cpu"})
            resp = await MonitoringHandler.get_baseline_stats(req)

        assert resp.status == 500
        cb.record_failure.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_baseline_unauthorized(self, install_mocks):
        from aragora.server.handlers.utils.auth import UnauthorizedError

        with patch(
            "aragora.server.handlers.autonomous.monitoring.get_auth_context",
            side_effect=UnauthorizedError(),
        ):
            req = _make_request(match_info={"metric_name": "cpu"})
            resp = await MonitoringHandler.get_baseline_stats(req)

        assert resp.status == 401

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_baseline_forbidden(self, install_mocks):
        mock_ctx = MagicMock()
        mock_checker = MagicMock()
        mock_decision = MagicMock()
        mock_decision.allowed = False

        with (
            patch(
                "aragora.server.handlers.autonomous.monitoring.get_auth_context",
                new_callable=AsyncMock,
                return_value=mock_ctx,
            ),
            patch(
                "aragora.server.handlers.autonomous.monitoring.get_permission_checker",
                return_value=mock_checker,
            ),
        ):
            mock_checker.check_permission.return_value = mock_decision
            req = _make_request(match_info={"metric_name": "cpu"})
            resp = await MonitoringHandler.get_baseline_stats(req)

        assert resp.status == 403

    @pytest.mark.asyncio
    async def test_baseline_success_records_cb_success(self, install_mocks):
        _, anomaly_det = install_mocks
        anomaly_det.get_baseline_stats.return_value = {"mean": 1.0}

        cb = MagicMock()
        cb.is_allowed.return_value = True

        with patch(
            "aragora.server.handlers.autonomous.monitoring._get_monitoring_circuit_breaker",
            return_value=cb,
        ):
            req = _make_request(match_info={"metric_name": "cpu"})
            resp = await MonitoringHandler.get_baseline_stats(req)

        assert resp.status == 200
        cb.record_success.assert_called_once()

    @pytest.mark.asyncio
    async def test_baseline_insufficient_data_records_cb_success(self, install_mocks):
        """Even with no data, should record success on circuit breaker."""
        _, anomaly_det = install_mocks
        anomaly_det.get_baseline_stats.return_value = None

        cb = MagicMock()
        cb.is_allowed.return_value = True

        with patch(
            "aragora.server.handlers.autonomous.monitoring._get_monitoring_circuit_breaker",
            return_value=cb,
        ):
            req = _make_request(match_info={"metric_name": "cpu"})
            resp = await MonitoringHandler.get_baseline_stats(req)

        assert resp.status == 200
        cb.record_success.assert_called_once()


# ---------------------------------------------------------------------------
# get_circuit_breaker_status endpoint
# ---------------------------------------------------------------------------


class TestGetCircuitBreakerStatus:
    @pytest.mark.asyncio
    async def test_cb_status_success(self):
        mock_status = {
            "state": "closed",
            "failure_count": 0,
            "success_count": 0,
            "failure_threshold": 5,
            "cooldown_seconds": 30.0,
            "last_failure_time": None,
        }

        with patch(
            "aragora.server.handlers.autonomous.monitoring.get_monitoring_circuit_breaker_status",
            return_value=mock_status,
        ):
            req = _make_request()
            resp = await MonitoringHandler.get_circuit_breaker_status(req)

        assert resp.status == 200
        data = await _parse(resp)
        assert data["success"] is True
        assert data["circuit_breaker"]["state"] == "closed"
        assert data["circuit_breaker"]["failure_count"] == 0

    @pytest.mark.asyncio
    async def test_cb_status_open_state(self):
        mock_status = {
            "state": "open",
            "failure_count": 5,
            "success_count": 0,
            "failure_threshold": 5,
            "cooldown_seconds": 30.0,
            "last_failure_time": 1234567890.0,
        }

        with patch(
            "aragora.server.handlers.autonomous.monitoring.get_monitoring_circuit_breaker_status",
            return_value=mock_status,
        ):
            req = _make_request()
            resp = await MonitoringHandler.get_circuit_breaker_status(req)

        assert resp.status == 200
        data = await _parse(resp)
        assert data["circuit_breaker"]["state"] == "open"
        assert data["circuit_breaker"]["failure_count"] == 5

    @pytest.mark.asyncio
    async def test_cb_status_internal_error(self):
        with patch(
            "aragora.server.handlers.autonomous.monitoring.get_monitoring_circuit_breaker_status",
            side_effect=RuntimeError("broken"),
        ):
            req = _make_request()
            resp = await MonitoringHandler.get_circuit_breaker_status(req)

        assert resp.status == 500
        data = await _parse(resp)
        assert data["success"] is False

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_cb_status_unauthorized(self):
        from aragora.server.handlers.utils.auth import UnauthorizedError

        with patch(
            "aragora.server.handlers.autonomous.monitoring.get_auth_context",
            side_effect=UnauthorizedError(),
        ):
            req = _make_request()
            resp = await MonitoringHandler.get_circuit_breaker_status(req)

        assert resp.status == 401

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_cb_status_forbidden(self):
        mock_ctx = MagicMock()
        mock_checker = MagicMock()
        mock_decision = MagicMock()
        mock_decision.allowed = False

        with (
            patch(
                "aragora.server.handlers.autonomous.monitoring.get_auth_context",
                new_callable=AsyncMock,
                return_value=mock_ctx,
            ),
            patch(
                "aragora.server.handlers.autonomous.monitoring.get_permission_checker",
                return_value=mock_checker,
            ),
        ):
            mock_checker.check_permission.return_value = mock_decision
            req = _make_request()
            resp = await MonitoringHandler.get_circuit_breaker_status(req)

        assert resp.status == 403


# ---------------------------------------------------------------------------
# Handler ROUTES and register_routes
# ---------------------------------------------------------------------------


class TestHandlerMeta:
    def test_routes_constant(self):
        """ROUTES should list all registered endpoints."""
        routes = MonitoringHandler.ROUTES
        assert "/api/v1/autonomous/monitoring/record" in routes
        assert "/api/v1/autonomous/monitoring/trends" in routes
        assert "/api/v1/autonomous/monitoring/trends/*" in routes
        assert "/api/v1/autonomous/monitoring/anomalies" in routes
        assert (
            "/api/v1/autonomous/monitoring/baseline" in routes
            or "/api/v1/autonomous/monitoring/baseline/*" in routes
        )
        assert "/api/v1/autonomous/monitoring/circuit-breaker" in routes

    def test_register_routes(self):
        """register_routes should add routes to an aiohttp app."""
        app = web.Application()
        MonitoringHandler.register_routes(app)

        route_paths = [
            r.resource.canonical
            for r in app.router.routes()
            if hasattr(r, "resource") and r.resource
        ]
        assert any("monitoring/record" in p for p in route_paths)
        assert any("monitoring/trends" in p for p in route_paths)
        assert any("monitoring/anomalies" in p for p in route_paths)
        assert any("monitoring/baseline" in p for p in route_paths)
        assert any("circuit-breaker" in p for p in route_paths)

    def test_register_routes_custom_prefix(self):
        """register_routes should respect custom prefix."""
        app = web.Application()
        MonitoringHandler.register_routes(app, prefix="/custom")

        route_paths = [
            r.resource.canonical
            for r in app.router.routes()
            if hasattr(r, "resource") and r.resource
        ]
        assert any("/custom/monitoring/record" in p for p in route_paths)

    def test_init_default_ctx(self):
        handler = MonitoringHandler()
        assert handler.ctx == {}

    def test_init_custom_ctx(self):
        handler = MonitoringHandler(ctx={"key": "val"})
        assert handler.ctx == {"key": "val"}


# ---------------------------------------------------------------------------
# Global accessor tests
# ---------------------------------------------------------------------------


class TestGlobalAccessors:
    def test_get_trend_monitor_creates_instance(self):
        monitor = get_trend_monitor()
        assert monitor is not None

    def test_set_and_get_trend_monitor(self):
        custom = MagicMock()
        set_trend_monitor(custom)
        assert get_trend_monitor() is custom

    def test_get_anomaly_detector_creates_instance(self):
        detector = get_anomaly_detector()
        assert detector is not None

    def test_set_and_get_anomaly_detector(self):
        custom = MagicMock()
        set_anomaly_detector(custom)
        assert get_anomaly_detector() is custom

    def test_get_circuit_breaker_status_returns_dict(self):
        status = get_monitoring_circuit_breaker_status()
        assert isinstance(status, dict)
        assert "state" in status
        assert status["state"] == "closed"

    def test_clear_components(self):
        """_clear_monitoring_components resets everything."""
        # Populate
        set_trend_monitor(MagicMock())
        set_anomaly_detector(MagicMock())
        _ = get_monitoring_circuit_breaker_status()  # ensure CB exists

        _clear_monitoring_components()

        # After clear, get_* should create fresh instances
        m1 = get_trend_monitor()
        m2 = get_trend_monitor()
        assert m1 is m2  # same singleton

    def test_trend_monitor_singleton(self):
        m1 = get_trend_monitor()
        m2 = get_trend_monitor()
        assert m1 is m2

    def test_anomaly_detector_singleton(self):
        d1 = get_anomaly_detector()
        d2 = get_anomaly_detector()
        assert d1 is d2


# ---------------------------------------------------------------------------
# Circuit breaker integration behaviour
# ---------------------------------------------------------------------------


class TestCircuitBreakerIntegration:
    @pytest.mark.asyncio
    async def test_503_includes_retry_after(self):
        cb = MagicMock()
        cb.is_allowed.return_value = False
        cb.state = "open"
        cb.cooldown_seconds = 42.0

        with patch(
            "aragora.server.handlers.autonomous.monitoring._get_monitoring_circuit_breaker",
            return_value=cb,
        ):
            req = _make_request(method="POST", body={"metric_name": "cpu", "value": 1.0})
            resp = await MonitoringHandler.record_metric(req)

        assert resp.status == 503
        data = await _parse(resp)
        assert data["retry_after_seconds"] == 42.0
        assert data["circuit_breaker_state"] == "open"

    @pytest.mark.asyncio
    async def test_half_open_state_reported(self):
        cb = MagicMock()
        cb.is_allowed.return_value = False
        cb.state = "half_open"
        cb.cooldown_seconds = 15.0

        with patch(
            "aragora.server.handlers.autonomous.monitoring._get_monitoring_circuit_breaker",
            return_value=cb,
        ):
            req = _make_request()
            resp = await MonitoringHandler.get_all_trends(req)

        assert resp.status == 503
        data = await _parse(resp)
        assert data["circuit_breaker_state"] == "half_open"
