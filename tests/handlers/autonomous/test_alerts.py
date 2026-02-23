"""Comprehensive tests for AlertHandler.

Tests cover:
- list_active (GET /api/v1/autonomous/alerts/active)
- acknowledge (POST /api/v1/autonomous/alerts/{alert_id}/acknowledge)
- resolve (POST /api/v1/autonomous/alerts/{alert_id}/resolve)
- set_threshold (POST /api/v1/autonomous/alerts/thresholds)
- check_metric (POST /api/v1/autonomous/alerts/check)
- Auth / permission checks (unauthorized, forbidden)
- Error-handling paths (KeyError, ValueError, TypeError, AttributeError, RuntimeError)
- Input validation (missing fields, empty values)
- Global accessors (get/set_alert_analyzer, circuit breaker status)
- register_routes
- Edge cases (path traversal, injection, invalid JSON)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web

from aragora.server.handlers.autonomous.alerts import (
    AlertHandler,
    get_alert_analyzer,
    set_alert_analyzer,
    get_alert_circuit_breaker,
    get_alert_circuit_breaker_status,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _parse(response: web.Response) -> dict:
    """Extract JSON dict from an aiohttp json_response."""
    return json.loads(response.body)


class _MockSeverity(Enum):
    """Mock alert severity for tests."""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


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

    req.remote = "127.0.0.1"
    req.transport = MagicMock()
    req.transport.get_extra_info.return_value = ("127.0.0.1", 12345)

    return req


def _make_alert(
    alert_id: str = "alert-1",
    severity=None,
    title: str = "High CPU Usage",
    description: str = "CPU usage exceeded threshold",
    source: str = "system",
    acknowledged: bool = False,
    acknowledged_by: str | None = None,
    debate_triggered: bool = False,
    debate_id: str | None = None,
    metadata: dict | None = None,
) -> MagicMock:
    """Build a mock Alert object."""
    severity = severity or _MockSeverity.HIGH
    obj = MagicMock()
    obj.id = alert_id
    obj.severity = severity
    obj.title = title
    obj.description = description
    obj.source = source
    obj.timestamp = datetime(2026, 2, 1, 12, 0, 0, tzinfo=timezone.utc)
    obj.acknowledged = acknowledged
    obj.acknowledged_by = acknowledged_by
    obj.debate_triggered = debate_triggered
    obj.debate_id = debate_id
    obj.metadata = metadata or {}
    return obj


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_alert_globals():
    """Reset the global alert analyzer between tests."""
    import aragora.server.handlers.autonomous.alerts as mod

    old_analyzer = mod._alert_analyzer
    mod._alert_analyzer = None
    yield
    mod._alert_analyzer = old_analyzer


@pytest.fixture
def mock_analyzer():
    """Create a mock AlertAnalyzer instance."""
    analyzer = MagicMock()
    analyzer.get_active_alerts = MagicMock(return_value=[])
    analyzer.acknowledge_alert = MagicMock(return_value=True)
    analyzer.resolve_alert = MagicMock(return_value=True)
    analyzer.set_threshold = MagicMock()
    analyzer.check_metric = AsyncMock(return_value=None)
    return analyzer


@pytest.fixture
def install_analyzer(mock_analyzer):
    """Set mock analyzer as the global singleton."""
    set_alert_analyzer(mock_analyzer)
    return mock_analyzer


# ---------------------------------------------------------------------------
# list_active endpoint
# ---------------------------------------------------------------------------


class TestListActive:
    @pytest.mark.asyncio
    async def test_list_empty(self, install_analyzer):
        install_analyzer.get_active_alerts.return_value = []
        req = _make_request()
        resp = await AlertHandler.list_active(req)

        assert resp.status == 200
        data = await _parse(resp)
        assert data["success"] is True
        assert data["alerts"] == []
        assert data["count"] == 0

    @pytest.mark.asyncio
    async def test_list_single_alert(self, install_analyzer):
        alert = _make_alert()
        install_analyzer.get_active_alerts.return_value = [alert]

        req = _make_request()
        resp = await AlertHandler.list_active(req)

        assert resp.status == 200
        data = await _parse(resp)
        assert data["success"] is True
        assert data["count"] == 1
        assert data["alerts"][0]["id"] == "alert-1"
        assert data["alerts"][0]["severity"] == "high"
        assert data["alerts"][0]["title"] == "High CPU Usage"
        assert data["alerts"][0]["description"] == "CPU usage exceeded threshold"
        assert data["alerts"][0]["source"] == "system"
        assert data["alerts"][0]["acknowledged"] is False
        assert data["alerts"][0]["acknowledged_by"] is None
        assert data["alerts"][0]["debate_triggered"] is False
        assert data["alerts"][0]["debate_id"] is None
        assert data["alerts"][0]["metadata"] == {}

    @pytest.mark.asyncio
    async def test_list_multiple_alerts(self, install_analyzer):
        alerts = [
            _make_alert(alert_id="a-1", title="Alert 1"),
            _make_alert(alert_id="a-2", title="Alert 2", severity=_MockSeverity.CRITICAL),
            _make_alert(alert_id="a-3", title="Alert 3", acknowledged=True, acknowledged_by="admin"),
        ]
        install_analyzer.get_active_alerts.return_value = alerts

        req = _make_request()
        resp = await AlertHandler.list_active(req)

        assert resp.status == 200
        data = await _parse(resp)
        assert data["count"] == 3
        assert data["alerts"][0]["id"] == "a-1"
        assert data["alerts"][1]["severity"] == "critical"
        assert data["alerts"][2]["acknowledged"] is True
        assert data["alerts"][2]["acknowledged_by"] == "admin"

    @pytest.mark.asyncio
    async def test_list_alert_with_debate(self, install_analyzer):
        alert = _make_alert(debate_triggered=True, debate_id="debate-42")
        install_analyzer.get_active_alerts.return_value = [alert]

        req = _make_request()
        resp = await AlertHandler.list_active(req)

        data = await _parse(resp)
        assert data["alerts"][0]["debate_triggered"] is True
        assert data["alerts"][0]["debate_id"] == "debate-42"

    @pytest.mark.asyncio
    async def test_list_alert_with_metadata(self, install_analyzer):
        alert = _make_alert(metadata={"metric": "cpu", "node": "us-east-1"})
        install_analyzer.get_active_alerts.return_value = [alert]

        req = _make_request()
        resp = await AlertHandler.list_active(req)

        data = await _parse(resp)
        assert data["alerts"][0]["metadata"]["metric"] == "cpu"
        assert data["alerts"][0]["metadata"]["node"] == "us-east-1"

    @pytest.mark.asyncio
    async def test_list_alert_timestamp_format(self, install_analyzer):
        alert = _make_alert()
        install_analyzer.get_active_alerts.return_value = [alert]

        req = _make_request()
        resp = await AlertHandler.list_active(req)

        data = await _parse(resp)
        assert data["alerts"][0]["timestamp"] == "2026-02-01T12:00:00+00:00"

    @pytest.mark.asyncio
    async def test_list_response_keys(self, install_analyzer):
        alert = _make_alert()
        install_analyzer.get_active_alerts.return_value = [alert]

        req = _make_request()
        resp = await AlertHandler.list_active(req)

        data = await _parse(resp)
        expected_keys = {
            "id", "severity", "title", "description", "source",
            "timestamp", "acknowledged", "acknowledged_by",
            "debate_triggered", "debate_id", "metadata",
        }
        assert expected_keys == set(data["alerts"][0].keys())

    @pytest.mark.asyncio
    async def test_list_runtime_error(self, install_analyzer):
        install_analyzer.get_active_alerts.side_effect = RuntimeError("db down")

        req = _make_request()
        resp = await AlertHandler.list_active(req)

        assert resp.status == 500
        data = await _parse(resp)
        assert data["success"] is False
        assert "Failed to list alerts" in data["error"]

    @pytest.mark.asyncio
    async def test_list_key_error(self, install_analyzer):
        install_analyzer.get_active_alerts.side_effect = KeyError("missing")

        req = _make_request()
        resp = await AlertHandler.list_active(req)

        assert resp.status == 500

    @pytest.mark.asyncio
    async def test_list_value_error(self, install_analyzer):
        install_analyzer.get_active_alerts.side_effect = ValueError("invalid")

        req = _make_request()
        resp = await AlertHandler.list_active(req)

        assert resp.status == 500

    @pytest.mark.asyncio
    async def test_list_type_error(self, install_analyzer):
        install_analyzer.get_active_alerts.side_effect = TypeError("bad type")

        req = _make_request()
        resp = await AlertHandler.list_active(req)

        assert resp.status == 500

    @pytest.mark.asyncio
    async def test_list_attribute_error(self, install_analyzer):
        install_analyzer.get_active_alerts.side_effect = AttributeError("missing attr")

        req = _make_request()
        resp = await AlertHandler.list_active(req)

        assert resp.status == 500

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_list_unauthorized(self, install_analyzer):
        from aragora.server.handlers.utils.auth import UnauthorizedError

        with patch(
            "aragora.server.handlers.autonomous.alerts.get_auth_context",
            side_effect=UnauthorizedError("no token"),
        ):
            req = _make_request()
            resp = await AlertHandler.list_active(req)

        assert resp.status == 401
        data = await _parse(resp)
        assert "Authentication required" in data["error"]

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_list_forbidden(self, install_analyzer):
        mock_ctx = MagicMock()
        mock_checker = MagicMock()
        mock_decision = MagicMock()
        mock_decision.allowed = False

        with (
            patch(
                "aragora.server.handlers.autonomous.alerts.get_auth_context",
                new_callable=AsyncMock,
                return_value=mock_ctx,
            ),
            patch(
                "aragora.server.handlers.autonomous.alerts.get_permission_checker",
                return_value=mock_checker,
            ),
        ):
            mock_checker.check_permission.return_value = mock_decision
            req = _make_request()
            resp = await AlertHandler.list_active(req)

        assert resp.status == 403
        data = await _parse(resp)
        assert "Permission denied" in data["error"]

    @pytest.mark.asyncio
    async def test_list_all_severities(self, install_analyzer):
        """Verify all severity levels serialize correctly."""
        alerts = [
            _make_alert(alert_id="a-info", severity=_MockSeverity.INFO),
            _make_alert(alert_id="a-low", severity=_MockSeverity.LOW),
            _make_alert(alert_id="a-med", severity=_MockSeverity.MEDIUM),
            _make_alert(alert_id="a-high", severity=_MockSeverity.HIGH),
            _make_alert(alert_id="a-crit", severity=_MockSeverity.CRITICAL),
        ]
        install_analyzer.get_active_alerts.return_value = alerts

        req = _make_request()
        resp = await AlertHandler.list_active(req)

        data = await _parse(resp)
        assert data["count"] == 5
        severities = [a["severity"] for a in data["alerts"]]
        assert severities == ["info", "low", "medium", "high", "critical"]


# ---------------------------------------------------------------------------
# acknowledge endpoint
# ---------------------------------------------------------------------------


class TestAcknowledge:
    @pytest.mark.asyncio
    async def test_acknowledge_success(self, install_analyzer):
        install_analyzer.acknowledge_alert.return_value = True

        req = _make_request(
            method="POST",
            match_info={"alert_id": "alert-1"},
            body={"acknowledged_by": "admin-user"},
        )
        resp = await AlertHandler.acknowledge(req)

        assert resp.status == 200
        data = await _parse(resp)
        assert data["success"] is True
        assert data["alert_id"] == "alert-1"
        assert data["acknowledged_by"] == "admin-user"
        install_analyzer.acknowledge_alert.assert_called_once_with("alert-1", "admin-user")

    @pytest.mark.asyncio
    async def test_acknowledge_uses_auth_user_id_when_no_body_field(self, install_analyzer):
        """When acknowledged_by is not in body, fall back to auth_ctx.user_id."""
        install_analyzer.acknowledge_alert.return_value = True

        req = _make_request(
            method="POST",
            match_info={"alert_id": "alert-2"},
            body={},
        )
        resp = await AlertHandler.acknowledge(req)

        assert resp.status == 200
        data = await _parse(resp)
        assert data["success"] is True
        # Should fall back to the auth context user_id ("test-user-001")
        assert data["acknowledged_by"] == "test-user-001"

    @pytest.mark.asyncio
    async def test_acknowledge_not_found(self, install_analyzer):
        install_analyzer.acknowledge_alert.return_value = False

        req = _make_request(
            method="POST",
            match_info={"alert_id": "nonexistent"},
            body={"acknowledged_by": "admin"},
        )
        resp = await AlertHandler.acknowledge(req)

        assert resp.status == 404
        data = await _parse(resp)
        assert data["success"] is False
        assert "not found" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_acknowledge_runtime_error(self, install_analyzer):
        install_analyzer.acknowledge_alert.side_effect = RuntimeError("db error")

        req = _make_request(
            method="POST",
            match_info={"alert_id": "alert-1"},
            body={"acknowledged_by": "admin"},
        )
        resp = await AlertHandler.acknowledge(req)

        assert resp.status == 500
        data = await _parse(resp)
        assert "Failed to acknowledge alert" in data["error"]

    @pytest.mark.asyncio
    async def test_acknowledge_key_error(self, install_analyzer):
        install_analyzer.acknowledge_alert.side_effect = KeyError("missing")

        req = _make_request(
            method="POST",
            match_info={"alert_id": "alert-1"},
            body={"acknowledged_by": "admin"},
        )
        resp = await AlertHandler.acknowledge(req)

        assert resp.status == 500

    @pytest.mark.asyncio
    async def test_acknowledge_value_error(self, install_analyzer):
        install_analyzer.acknowledge_alert.side_effect = ValueError("invalid")

        req = _make_request(
            method="POST",
            match_info={"alert_id": "alert-1"},
            body={"acknowledged_by": "admin"},
        )
        resp = await AlertHandler.acknowledge(req)

        assert resp.status == 500

    @pytest.mark.asyncio
    async def test_acknowledge_type_error(self, install_analyzer):
        install_analyzer.acknowledge_alert.side_effect = TypeError("bad type")

        req = _make_request(
            method="POST",
            match_info={"alert_id": "alert-1"},
            body={"acknowledged_by": "admin"},
        )
        resp = await AlertHandler.acknowledge(req)

        assert resp.status == 500

    @pytest.mark.asyncio
    async def test_acknowledge_attribute_error(self, install_analyzer):
        install_analyzer.acknowledge_alert.side_effect = AttributeError("no attr")

        req = _make_request(
            method="POST",
            match_info={"alert_id": "alert-1"},
            body={"acknowledged_by": "admin"},
        )
        resp = await AlertHandler.acknowledge(req)

        assert resp.status == 500

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_acknowledge_unauthorized(self, install_analyzer):
        from aragora.server.handlers.utils.auth import UnauthorizedError

        with patch(
            "aragora.server.handlers.autonomous.alerts.get_auth_context",
            side_effect=UnauthorizedError("no token"),
        ):
            req = _make_request(
                method="POST",
                match_info={"alert_id": "alert-1"},
                body={"acknowledged_by": "admin"},
            )
            resp = await AlertHandler.acknowledge(req)

        assert resp.status == 401
        data = await _parse(resp)
        assert "Authentication required" in data["error"]

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_acknowledge_forbidden(self, install_analyzer):
        mock_ctx = MagicMock()
        mock_checker = MagicMock()
        mock_decision = MagicMock()
        mock_decision.allowed = False

        with (
            patch(
                "aragora.server.handlers.autonomous.alerts.get_auth_context",
                new_callable=AsyncMock,
                return_value=mock_ctx,
            ),
            patch(
                "aragora.server.handlers.autonomous.alerts.get_permission_checker",
                return_value=mock_checker,
            ),
        ):
            mock_checker.check_permission.return_value = mock_decision
            req = _make_request(
                method="POST",
                match_info={"alert_id": "alert-1"},
                body={"acknowledged_by": "admin"},
            )
            resp = await AlertHandler.acknowledge(req)

        assert resp.status == 403
        data = await _parse(resp)
        assert "Permission denied" in data["error"]

    @pytest.mark.asyncio
    async def test_acknowledge_invalid_json(self, install_analyzer):
        """Malformed JSON body returns 400 via parse_json_body."""
        req = MagicMock()
        req.method = "POST"
        req.json = AsyncMock(side_effect=ValueError("bad json"))
        req.read = AsyncMock(return_value=b"not json")
        req.text = AsyncMock(return_value="not json")
        req.content_type = "application/json"
        req.content_length = 8
        req.can_read_body = True
        req.remote = "127.0.0.1"
        req.transport = MagicMock()
        req.transport.get_extra_info.return_value = ("127.0.0.1", 12345)
        mi = MagicMock()
        mi.get = MagicMock(return_value="alert-1")
        req.match_info = mi

        resp = await AlertHandler.acknowledge(req)
        assert resp.status == 400

    @pytest.mark.asyncio
    async def test_acknowledge_none_alert_id(self, install_analyzer):
        """When match_info returns None for alert_id."""
        install_analyzer.acknowledge_alert.return_value = False

        req = _make_request(
            method="POST",
            match_info={},
            body={"acknowledged_by": "admin"},
        )
        resp = await AlertHandler.acknowledge(req)

        assert resp.status == 404

    @pytest.mark.asyncio
    async def test_acknowledge_empty_acknowledged_by(self, install_analyzer):
        """Empty acknowledged_by string should fall back to auth user_id."""
        install_analyzer.acknowledge_alert.return_value = True

        req = _make_request(
            method="POST",
            match_info={"alert_id": "alert-1"},
            body={"acknowledged_by": ""},
        )
        resp = await AlertHandler.acknowledge(req)

        assert resp.status == 200
        data = await _parse(resp)
        # Empty string is falsy, so it should fall back to auth_ctx.user_id
        assert data["acknowledged_by"] == "test-user-001"


# ---------------------------------------------------------------------------
# resolve endpoint
# ---------------------------------------------------------------------------


class TestResolve:
    @pytest.mark.asyncio
    async def test_resolve_success(self, install_analyzer):
        install_analyzer.resolve_alert.return_value = True

        req = _make_request(
            method="POST",
            match_info={"alert_id": "alert-1"},
        )
        resp = await AlertHandler.resolve(req)

        assert resp.status == 200
        data = await _parse(resp)
        assert data["success"] is True
        assert data["alert_id"] == "alert-1"
        assert data["resolved"] is True
        install_analyzer.resolve_alert.assert_called_once_with("alert-1")

    @pytest.mark.asyncio
    async def test_resolve_not_found(self, install_analyzer):
        install_analyzer.resolve_alert.return_value = False

        req = _make_request(
            method="POST",
            match_info={"alert_id": "nonexistent"},
        )
        resp = await AlertHandler.resolve(req)

        assert resp.status == 404
        data = await _parse(resp)
        assert data["success"] is False
        assert "not found" in data["error"].lower()

    @pytest.mark.asyncio
    async def test_resolve_runtime_error(self, install_analyzer):
        install_analyzer.resolve_alert.side_effect = RuntimeError("db error")

        req = _make_request(
            method="POST",
            match_info={"alert_id": "alert-1"},
        )
        resp = await AlertHandler.resolve(req)

        assert resp.status == 500
        data = await _parse(resp)
        assert "Failed to resolve alert" in data["error"]

    @pytest.mark.asyncio
    async def test_resolve_key_error(self, install_analyzer):
        install_analyzer.resolve_alert.side_effect = KeyError("missing")

        req = _make_request(
            method="POST",
            match_info={"alert_id": "alert-1"},
        )
        resp = await AlertHandler.resolve(req)

        assert resp.status == 500

    @pytest.mark.asyncio
    async def test_resolve_value_error(self, install_analyzer):
        install_analyzer.resolve_alert.side_effect = ValueError("invalid")

        req = _make_request(
            method="POST",
            match_info={"alert_id": "alert-1"},
        )
        resp = await AlertHandler.resolve(req)

        assert resp.status == 500

    @pytest.mark.asyncio
    async def test_resolve_type_error(self, install_analyzer):
        install_analyzer.resolve_alert.side_effect = TypeError("bad type")

        req = _make_request(
            method="POST",
            match_info={"alert_id": "alert-1"},
        )
        resp = await AlertHandler.resolve(req)

        assert resp.status == 500

    @pytest.mark.asyncio
    async def test_resolve_attribute_error(self, install_analyzer):
        install_analyzer.resolve_alert.side_effect = AttributeError("no attr")

        req = _make_request(
            method="POST",
            match_info={"alert_id": "alert-1"},
        )
        resp = await AlertHandler.resolve(req)

        assert resp.status == 500

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_resolve_unauthorized(self, install_analyzer):
        from aragora.server.handlers.utils.auth import UnauthorizedError

        with patch(
            "aragora.server.handlers.autonomous.alerts.get_auth_context",
            side_effect=UnauthorizedError("no token"),
        ):
            req = _make_request(
                method="POST",
                match_info={"alert_id": "alert-1"},
            )
            resp = await AlertHandler.resolve(req)

        assert resp.status == 401
        data = await _parse(resp)
        assert "Authentication required" in data["error"]

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_resolve_forbidden(self, install_analyzer):
        mock_ctx = MagicMock()
        mock_checker = MagicMock()
        mock_decision = MagicMock()
        mock_decision.allowed = False

        with (
            patch(
                "aragora.server.handlers.autonomous.alerts.get_auth_context",
                new_callable=AsyncMock,
                return_value=mock_ctx,
            ),
            patch(
                "aragora.server.handlers.autonomous.alerts.get_permission_checker",
                return_value=mock_checker,
            ),
        ):
            mock_checker.check_permission.return_value = mock_decision
            req = _make_request(
                method="POST",
                match_info={"alert_id": "alert-1"},
            )
            resp = await AlertHandler.resolve(req)

        assert resp.status == 403
        data = await _parse(resp)
        assert "Permission denied" in data["error"]

    @pytest.mark.asyncio
    async def test_resolve_none_alert_id(self, install_analyzer):
        """When match_info returns None for alert_id."""
        install_analyzer.resolve_alert.return_value = False

        req = _make_request(
            method="POST",
            match_info={},
        )
        resp = await AlertHandler.resolve(req)

        assert resp.status == 404


# ---------------------------------------------------------------------------
# set_threshold endpoint
# ---------------------------------------------------------------------------


class TestSetThreshold:
    @pytest.mark.asyncio
    async def test_set_threshold_success(self, install_analyzer):
        req = _make_request(
            method="POST",
            body={
                "metric_name": "cpu_usage",
                "warning_threshold": 80.0,
                "critical_threshold": 95.0,
                "comparison": "gt",
                "enabled": True,
            },
        )
        resp = await AlertHandler.set_threshold(req)

        assert resp.status == 200
        data = await _parse(resp)
        assert data["success"] is True
        assert data["metric_name"] == "cpu_usage"
        assert data["threshold_set"] is True
        install_analyzer.set_threshold.assert_called_once_with(
            metric_name="cpu_usage",
            warning_threshold=80.0,
            critical_threshold=95.0,
            comparison="gt",
            enabled=True,
        )

    @pytest.mark.asyncio
    async def test_set_threshold_minimal(self, install_analyzer):
        """Only metric_name is required."""
        req = _make_request(
            method="POST",
            body={"metric_name": "memory_usage"},
        )
        resp = await AlertHandler.set_threshold(req)

        assert resp.status == 200
        data = await _parse(resp)
        assert data["success"] is True
        assert data["metric_name"] == "memory_usage"

    @pytest.mark.asyncio
    async def test_set_threshold_defaults(self, install_analyzer):
        """Verify default values for comparison and enabled."""
        req = _make_request(
            method="POST",
            body={"metric_name": "latency"},
        )
        resp = await AlertHandler.set_threshold(req)

        assert resp.status == 200
        call_kwargs = install_analyzer.set_threshold.call_args[1]
        assert call_kwargs["comparison"] == "gt"
        assert call_kwargs["enabled"] is True

    @pytest.mark.asyncio
    async def test_set_threshold_missing_metric_name(self, install_analyzer):
        req = _make_request(
            method="POST",
            body={"warning_threshold": 80.0},
        )
        resp = await AlertHandler.set_threshold(req)

        assert resp.status == 400
        data = await _parse(resp)
        assert data["success"] is False
        assert "metric_name" in data["error"]

    @pytest.mark.asyncio
    async def test_set_threshold_empty_metric_name(self, install_analyzer):
        req = _make_request(
            method="POST",
            body={"metric_name": ""},
        )
        resp = await AlertHandler.set_threshold(req)

        assert resp.status == 400
        data = await _parse(resp)
        assert data["success"] is False
        assert "metric_name" in data["error"]

    @pytest.mark.asyncio
    async def test_set_threshold_null_metric_name(self, install_analyzer):
        req = _make_request(
            method="POST",
            body={"metric_name": None},
        )
        resp = await AlertHandler.set_threshold(req)

        assert resp.status == 400

    @pytest.mark.asyncio
    async def test_set_threshold_disabled(self, install_analyzer):
        req = _make_request(
            method="POST",
            body={
                "metric_name": "error_rate",
                "enabled": False,
            },
        )
        resp = await AlertHandler.set_threshold(req)

        assert resp.status == 200
        call_kwargs = install_analyzer.set_threshold.call_args[1]
        assert call_kwargs["enabled"] is False

    @pytest.mark.asyncio
    async def test_set_threshold_lt_comparison(self, install_analyzer):
        req = _make_request(
            method="POST",
            body={
                "metric_name": "uptime",
                "comparison": "lt",
                "warning_threshold": 99.0,
            },
        )
        resp = await AlertHandler.set_threshold(req)

        assert resp.status == 200
        call_kwargs = install_analyzer.set_threshold.call_args[1]
        assert call_kwargs["comparison"] == "lt"

    @pytest.mark.asyncio
    async def test_set_threshold_runtime_error(self, install_analyzer):
        install_analyzer.set_threshold.side_effect = RuntimeError("db error")

        req = _make_request(
            method="POST",
            body={"metric_name": "cpu_usage"},
        )
        resp = await AlertHandler.set_threshold(req)

        assert resp.status == 500
        data = await _parse(resp)
        assert "Failed to set alert threshold" in data["error"]

    @pytest.mark.asyncio
    async def test_set_threshold_key_error(self, install_analyzer):
        install_analyzer.set_threshold.side_effect = KeyError("missing")

        req = _make_request(
            method="POST",
            body={"metric_name": "cpu_usage"},
        )
        resp = await AlertHandler.set_threshold(req)

        assert resp.status == 500

    @pytest.mark.asyncio
    async def test_set_threshold_value_error(self, install_analyzer):
        install_analyzer.set_threshold.side_effect = ValueError("invalid")

        req = _make_request(
            method="POST",
            body={"metric_name": "cpu_usage"},
        )
        resp = await AlertHandler.set_threshold(req)

        assert resp.status == 500

    @pytest.mark.asyncio
    async def test_set_threshold_type_error(self, install_analyzer):
        install_analyzer.set_threshold.side_effect = TypeError("bad type")

        req = _make_request(
            method="POST",
            body={"metric_name": "cpu_usage"},
        )
        resp = await AlertHandler.set_threshold(req)

        assert resp.status == 500

    @pytest.mark.asyncio
    async def test_set_threshold_attribute_error(self, install_analyzer):
        install_analyzer.set_threshold.side_effect = AttributeError("no attr")

        req = _make_request(
            method="POST",
            body={"metric_name": "cpu_usage"},
        )
        resp = await AlertHandler.set_threshold(req)

        assert resp.status == 500

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_set_threshold_unauthorized(self, install_analyzer):
        from aragora.server.handlers.utils.auth import UnauthorizedError

        with patch(
            "aragora.server.handlers.autonomous.alerts.get_auth_context",
            side_effect=UnauthorizedError("no token"),
        ):
            req = _make_request(
                method="POST",
                body={"metric_name": "cpu_usage"},
            )
            resp = await AlertHandler.set_threshold(req)

        assert resp.status == 401
        data = await _parse(resp)
        assert "Authentication required" in data["error"]

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_set_threshold_forbidden(self, install_analyzer):
        mock_ctx = MagicMock()
        mock_checker = MagicMock()
        mock_decision = MagicMock()
        mock_decision.allowed = False

        with (
            patch(
                "aragora.server.handlers.autonomous.alerts.get_auth_context",
                new_callable=AsyncMock,
                return_value=mock_ctx,
            ),
            patch(
                "aragora.server.handlers.autonomous.alerts.get_permission_checker",
                return_value=mock_checker,
            ),
        ):
            mock_checker.check_permission.return_value = mock_decision
            req = _make_request(
                method="POST",
                body={"metric_name": "cpu_usage"},
            )
            resp = await AlertHandler.set_threshold(req)

        assert resp.status == 403
        data = await _parse(resp)
        assert "Permission denied" in data["error"]

    @pytest.mark.asyncio
    async def test_set_threshold_invalid_json(self, install_analyzer):
        req = MagicMock()
        req.method = "POST"
        req.json = AsyncMock(side_effect=ValueError("bad json"))
        req.read = AsyncMock(return_value=b"not json")
        req.text = AsyncMock(return_value="not json")
        req.content_type = "application/json"
        req.content_length = 8
        req.can_read_body = True
        req.remote = "127.0.0.1"
        req.transport = MagicMock()
        req.transport.get_extra_info.return_value = ("127.0.0.1", 12345)

        resp = await AlertHandler.set_threshold(req)
        assert resp.status == 400

    @pytest.mark.asyncio
    async def test_set_threshold_only_warning(self, install_analyzer):
        req = _make_request(
            method="POST",
            body={
                "metric_name": "disk_usage",
                "warning_threshold": 75.0,
            },
        )
        resp = await AlertHandler.set_threshold(req)

        assert resp.status == 200
        call_kwargs = install_analyzer.set_threshold.call_args[1]
        assert call_kwargs["warning_threshold"] == 75.0
        assert call_kwargs["critical_threshold"] is None

    @pytest.mark.asyncio
    async def test_set_threshold_only_critical(self, install_analyzer):
        req = _make_request(
            method="POST",
            body={
                "metric_name": "disk_usage",
                "critical_threshold": 95.0,
            },
        )
        resp = await AlertHandler.set_threshold(req)

        assert resp.status == 200
        call_kwargs = install_analyzer.set_threshold.call_args[1]
        assert call_kwargs["warning_threshold"] is None
        assert call_kwargs["critical_threshold"] == 95.0


# ---------------------------------------------------------------------------
# check_metric endpoint
# ---------------------------------------------------------------------------


class TestCheckMetric:
    @pytest.mark.asyncio
    async def test_check_metric_no_alert(self, install_analyzer):
        install_analyzer.check_metric.return_value = None

        req = _make_request(
            method="POST",
            body={"metric_name": "cpu_usage", "value": 50.0},
        )
        resp = await AlertHandler.check_metric(req)

        assert resp.status == 200
        data = await _parse(resp)
        assert data["success"] is True
        assert data["alert_generated"] is False

    @pytest.mark.asyncio
    async def test_check_metric_alert_generated(self, install_analyzer):
        alert = _make_alert(alert_id="gen-1", title="CPU Critical")
        install_analyzer.check_metric.return_value = alert

        req = _make_request(
            method="POST",
            body={"metric_name": "cpu_usage", "value": 99.0},
        )
        resp = await AlertHandler.check_metric(req)

        assert resp.status == 200
        data = await _parse(resp)
        assert data["success"] is True
        assert data["alert_generated"] is True
        assert data["alert"]["id"] == "gen-1"
        assert data["alert"]["severity"] == "high"
        assert data["alert"]["title"] == "CPU Critical"
        assert data["alert"]["description"] == "CPU usage exceeded threshold"

    @pytest.mark.asyncio
    async def test_check_metric_with_source(self, install_analyzer):
        install_analyzer.check_metric.return_value = None

        req = _make_request(
            method="POST",
            body={"metric_name": "latency", "value": 100.0, "source": "prometheus"},
        )
        resp = await AlertHandler.check_metric(req)

        assert resp.status == 200
        call_kwargs = install_analyzer.check_metric.call_args[1]
        assert call_kwargs["source"] == "prometheus"

    @pytest.mark.asyncio
    async def test_check_metric_default_source(self, install_analyzer):
        install_analyzer.check_metric.return_value = None

        req = _make_request(
            method="POST",
            body={"metric_name": "latency", "value": 50.0},
        )
        resp = await AlertHandler.check_metric(req)

        assert resp.status == 200
        call_kwargs = install_analyzer.check_metric.call_args[1]
        assert call_kwargs["source"] == "api"

    @pytest.mark.asyncio
    async def test_check_metric_with_metadata(self, install_analyzer):
        install_analyzer.check_metric.return_value = None

        req = _make_request(
            method="POST",
            body={
                "metric_name": "latency",
                "value": 50.0,
                "metadata": {"region": "us-east-1"},
            },
        )
        resp = await AlertHandler.check_metric(req)

        assert resp.status == 200
        call_kwargs = install_analyzer.check_metric.call_args[1]
        assert call_kwargs["metadata"] == {"region": "us-east-1"}

    @pytest.mark.asyncio
    async def test_check_metric_missing_metric_name(self, install_analyzer):
        req = _make_request(
            method="POST",
            body={"value": 50.0},
        )
        resp = await AlertHandler.check_metric(req)

        assert resp.status == 400
        data = await _parse(resp)
        assert data["success"] is False
        assert "metric_name" in data["error"]

    @pytest.mark.asyncio
    async def test_check_metric_missing_value(self, install_analyzer):
        req = _make_request(
            method="POST",
            body={"metric_name": "cpu_usage"},
        )
        resp = await AlertHandler.check_metric(req)

        assert resp.status == 400
        data = await _parse(resp)
        assert data["success"] is False
        assert "value" in data["error"]

    @pytest.mark.asyncio
    async def test_check_metric_missing_both(self, install_analyzer):
        req = _make_request(
            method="POST",
            body={},
        )
        resp = await AlertHandler.check_metric(req)

        assert resp.status == 400

    @pytest.mark.asyncio
    async def test_check_metric_null_metric_name(self, install_analyzer):
        req = _make_request(
            method="POST",
            body={"metric_name": None, "value": 50.0},
        )
        resp = await AlertHandler.check_metric(req)

        assert resp.status == 400

    @pytest.mark.asyncio
    async def test_check_metric_empty_metric_name(self, install_analyzer):
        req = _make_request(
            method="POST",
            body={"metric_name": "", "value": 50.0},
        )
        resp = await AlertHandler.check_metric(req)

        assert resp.status == 400

    @pytest.mark.asyncio
    async def test_check_metric_value_zero(self, install_analyzer):
        """Value of 0 should be accepted (not treated as missing)."""
        install_analyzer.check_metric.return_value = None

        req = _make_request(
            method="POST",
            body={"metric_name": "error_count", "value": 0},
        )
        resp = await AlertHandler.check_metric(req)

        assert resp.status == 200
        call_kwargs = install_analyzer.check_metric.call_args[1]
        assert call_kwargs["value"] == 0.0

    @pytest.mark.asyncio
    async def test_check_metric_value_negative(self, install_analyzer):
        """Negative values should be accepted."""
        install_analyzer.check_metric.return_value = None

        req = _make_request(
            method="POST",
            body={"metric_name": "temperature", "value": -10.5},
        )
        resp = await AlertHandler.check_metric(req)

        assert resp.status == 200
        call_kwargs = install_analyzer.check_metric.call_args[1]
        assert call_kwargs["value"] == -10.5

    @pytest.mark.asyncio
    async def test_check_metric_value_as_string_number(self, install_analyzer):
        """String numeric value should be converted via float()."""
        install_analyzer.check_metric.return_value = None

        req = _make_request(
            method="POST",
            body={"metric_name": "latency", "value": "42.5"},
        )
        resp = await AlertHandler.check_metric(req)

        assert resp.status == 200
        call_kwargs = install_analyzer.check_metric.call_args[1]
        assert call_kwargs["value"] == 42.5

    @pytest.mark.asyncio
    async def test_check_metric_value_as_invalid_string(self, install_analyzer):
        """Non-numeric string value should trigger error path."""
        req = _make_request(
            method="POST",
            body={"metric_name": "latency", "value": "not_a_number"},
        )
        resp = await AlertHandler.check_metric(req)

        # float("not_a_number") raises ValueError, caught by error handler
        assert resp.status == 500

    @pytest.mark.asyncio
    async def test_check_metric_runtime_error(self, install_analyzer):
        install_analyzer.check_metric.side_effect = RuntimeError("service down")

        req = _make_request(
            method="POST",
            body={"metric_name": "cpu_usage", "value": 50.0},
        )
        resp = await AlertHandler.check_metric(req)

        assert resp.status == 500
        data = await _parse(resp)
        assert "Failed to check metric" in data["error"]

    @pytest.mark.asyncio
    async def test_check_metric_key_error(self, install_analyzer):
        install_analyzer.check_metric.side_effect = KeyError("missing")

        req = _make_request(
            method="POST",
            body={"metric_name": "cpu_usage", "value": 50.0},
        )
        resp = await AlertHandler.check_metric(req)

        assert resp.status == 500

    @pytest.mark.asyncio
    async def test_check_metric_value_error(self, install_analyzer):
        install_analyzer.check_metric.side_effect = ValueError("invalid")

        req = _make_request(
            method="POST",
            body={"metric_name": "cpu_usage", "value": 50.0},
        )
        resp = await AlertHandler.check_metric(req)

        assert resp.status == 500

    @pytest.mark.asyncio
    async def test_check_metric_type_error(self, install_analyzer):
        install_analyzer.check_metric.side_effect = TypeError("bad type")

        req = _make_request(
            method="POST",
            body={"metric_name": "cpu_usage", "value": 50.0},
        )
        resp = await AlertHandler.check_metric(req)

        assert resp.status == 500

    @pytest.mark.asyncio
    async def test_check_metric_attribute_error(self, install_analyzer):
        install_analyzer.check_metric.side_effect = AttributeError("no attr")

        req = _make_request(
            method="POST",
            body={"metric_name": "cpu_usage", "value": 50.0},
        )
        resp = await AlertHandler.check_metric(req)

        assert resp.status == 500

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_check_metric_unauthorized(self, install_analyzer):
        from aragora.server.handlers.utils.auth import UnauthorizedError

        with patch(
            "aragora.server.handlers.autonomous.alerts.get_auth_context",
            side_effect=UnauthorizedError("no token"),
        ):
            req = _make_request(
                method="POST",
                body={"metric_name": "cpu_usage", "value": 50.0},
            )
            resp = await AlertHandler.check_metric(req)

        assert resp.status == 401
        data = await _parse(resp)
        assert "Authentication required" in data["error"]

    @pytest.mark.asyncio
    @pytest.mark.no_auto_auth
    async def test_check_metric_forbidden(self, install_analyzer):
        mock_ctx = MagicMock()
        mock_checker = MagicMock()
        mock_decision = MagicMock()
        mock_decision.allowed = False

        with (
            patch(
                "aragora.server.handlers.autonomous.alerts.get_auth_context",
                new_callable=AsyncMock,
                return_value=mock_ctx,
            ),
            patch(
                "aragora.server.handlers.autonomous.alerts.get_permission_checker",
                return_value=mock_checker,
            ),
        ):
            mock_checker.check_permission.return_value = mock_decision
            req = _make_request(
                method="POST",
                body={"metric_name": "cpu_usage", "value": 50.0},
            )
            resp = await AlertHandler.check_metric(req)

        assert resp.status == 403
        data = await _parse(resp)
        assert "Permission denied" in data["error"]

    @pytest.mark.asyncio
    async def test_check_metric_invalid_json(self, install_analyzer):
        req = MagicMock()
        req.method = "POST"
        req.json = AsyncMock(side_effect=ValueError("bad json"))
        req.read = AsyncMock(return_value=b"not json")
        req.text = AsyncMock(return_value="not json")
        req.content_type = "application/json"
        req.content_length = 8
        req.can_read_body = True
        req.remote = "127.0.0.1"
        req.transport = MagicMock()
        req.transport.get_extra_info.return_value = ("127.0.0.1", 12345)

        resp = await AlertHandler.check_metric(req)
        assert resp.status == 400

    @pytest.mark.asyncio
    async def test_check_metric_alert_response_keys(self, install_analyzer):
        alert = _make_alert(alert_id="gen-1", severity=_MockSeverity.CRITICAL)
        install_analyzer.check_metric.return_value = alert

        req = _make_request(
            method="POST",
            body={"metric_name": "cpu_usage", "value": 99.0},
        )
        resp = await AlertHandler.check_metric(req)

        data = await _parse(resp)
        expected_alert_keys = {"id", "severity", "title", "description"}
        assert expected_alert_keys == set(data["alert"].keys())

    @pytest.mark.asyncio
    async def test_check_metric_value_converted_to_float(self, install_analyzer):
        """Integer value should be converted to float."""
        install_analyzer.check_metric.return_value = None

        req = _make_request(
            method="POST",
            body={"metric_name": "count", "value": 42},
        )
        resp = await AlertHandler.check_metric(req)

        assert resp.status == 200
        call_kwargs = install_analyzer.check_metric.call_args[1]
        assert isinstance(call_kwargs["value"], float)
        assert call_kwargs["value"] == 42.0


# ---------------------------------------------------------------------------
# register_routes
# ---------------------------------------------------------------------------


class TestRegisterRoutes:
    def test_register_routes_default_prefix(self):
        app = web.Application()
        AlertHandler.register_routes(app)

        route_paths = [
            r.resource.canonical
            for r in app.router.routes()
            if hasattr(r, "resource") and r.resource
        ]

        assert any("/alerts/active" in p for p in route_paths)
        assert any("alert_id" in p and "/acknowledge" in p for p in route_paths)
        assert any("alert_id" in p and "/resolve" in p for p in route_paths)
        assert any("/alerts/thresholds" in p for p in route_paths)
        assert any("/alerts/check" in p for p in route_paths)

    def test_register_routes_custom_prefix(self):
        app = web.Application()
        AlertHandler.register_routes(app, prefix="/custom/api")

        route_paths = [
            r.resource.canonical
            for r in app.router.routes()
            if hasattr(r, "resource") and r.resource
        ]

        assert any("/custom/api/alerts/active" in p for p in route_paths)
        assert any("/custom/api/alerts/thresholds" in p for p in route_paths)
        assert any("/custom/api/alerts/check" in p for p in route_paths)

    def test_register_routes_count(self):
        """There should be 6 routes: 5 explicit + 1 implicit HEAD for GET."""
        app = web.Application()
        AlertHandler.register_routes(app)

        route_count = sum(
            1
            for r in app.router.routes()
            if hasattr(r, "resource") and r.resource
        )
        # 1 GET (active) + 4 POST + 1 HEAD auto-added for GET = 6
        assert route_count == 6

    def test_register_routes_methods(self):
        """Verify correct HTTP methods for each route."""
        app = web.Application()
        AlertHandler.register_routes(app)

        method_map = {}
        for r in app.router.routes():
            if hasattr(r, "resource") and r.resource:
                key = r.resource.canonical
                if key not in method_map:
                    method_map[key] = []
                method_map[key].append(r.method)

        # Find routes by suffix
        for canonical, methods in method_map.items():
            if canonical.endswith("/alerts/active"):
                assert "GET" in methods
            elif canonical.endswith("/alerts/thresholds"):
                assert "POST" in methods
            elif canonical.endswith("/alerts/check"):
                assert "POST" in methods


# ---------------------------------------------------------------------------
# Handler init and global accessors
# ---------------------------------------------------------------------------


class TestHandlerInit:
    def test_handler_init_default(self):
        handler = AlertHandler()
        assert handler.ctx == {}

    def test_handler_init_custom_ctx(self):
        handler = AlertHandler(ctx={"env": "test"})
        assert handler.ctx == {"env": "test"}

    def test_handler_init_none_ctx(self):
        handler = AlertHandler(ctx=None)
        assert handler.ctx == {}


class TestGlobalAccessors:
    def test_get_alert_analyzer_creates_instance(self):
        analyzer = get_alert_analyzer()
        assert analyzer is not None

    def test_get_alert_analyzer_singleton(self):
        a1 = get_alert_analyzer()
        a2 = get_alert_analyzer()
        assert a1 is a2

    def test_set_and_get_alert_analyzer(self):
        custom = MagicMock()
        set_alert_analyzer(custom)
        assert get_alert_analyzer() is custom

    def test_set_alert_analyzer_replaces(self):
        first = MagicMock()
        second = MagicMock()
        set_alert_analyzer(first)
        set_alert_analyzer(second)
        assert get_alert_analyzer() is second

    def test_get_circuit_breaker(self):
        cb = get_alert_circuit_breaker()
        assert cb is not None
        assert cb.name == "alert_handler"

    def test_get_circuit_breaker_status(self):
        status = get_alert_circuit_breaker_status()
        assert isinstance(status, dict)
        assert "name" in status
        assert status["name"] == "alert_handler"


# ---------------------------------------------------------------------------
# Security edge cases
# ---------------------------------------------------------------------------


class TestSecurityEdgeCases:
    @pytest.mark.asyncio
    async def test_path_traversal_in_alert_id(self, install_analyzer):
        """Path traversal attempts in alert_id should not cause issues."""
        install_analyzer.acknowledge_alert.return_value = False

        req = _make_request(
            method="POST",
            match_info={"alert_id": "../../../etc/passwd"},
            body={"acknowledged_by": "admin"},
        )
        resp = await AlertHandler.acknowledge(req)

        assert resp.status == 404
        install_analyzer.acknowledge_alert.assert_called_once_with(
            "../../../etc/passwd", "admin"
        )

    @pytest.mark.asyncio
    async def test_sql_injection_in_alert_id(self, install_analyzer):
        """SQL injection in alert_id should be handled safely."""
        install_analyzer.resolve_alert.return_value = False

        req = _make_request(
            method="POST",
            match_info={"alert_id": "'; DROP TABLE alerts; --"},
        )
        resp = await AlertHandler.resolve(req)

        assert resp.status == 404

    @pytest.mark.asyncio
    async def test_xss_in_metric_name(self, install_analyzer):
        """XSS payload in metric_name should be handled safely."""
        req = _make_request(
            method="POST",
            body={
                "metric_name": "<script>alert('xss')</script>",
                "warning_threshold": 80.0,
            },
        )
        resp = await AlertHandler.set_threshold(req)

        assert resp.status == 200
        data = await _parse(resp)
        assert data["success"] is True
        # The handler passes it through; XSS protection is at the rendering layer
        assert data["metric_name"] == "<script>alert('xss')</script>"

    @pytest.mark.asyncio
    async def test_very_long_alert_id(self, install_analyzer):
        """Very long alert ID should be handled safely."""
        install_analyzer.resolve_alert.return_value = False
        long_id = "a" * 10000

        req = _make_request(
            method="POST",
            match_info={"alert_id": long_id},
        )
        resp = await AlertHandler.resolve(req)

        assert resp.status == 404

    @pytest.mark.asyncio
    async def test_unicode_in_acknowledged_by(self, install_analyzer):
        """Unicode characters in acknowledged_by should work."""
        install_analyzer.acknowledge_alert.return_value = True

        req = _make_request(
            method="POST",
            match_info={"alert_id": "alert-1"},
            body={"acknowledged_by": "usuario-espanol"},
        )
        resp = await AlertHandler.acknowledge(req)

        assert resp.status == 200
        data = await _parse(resp)
        assert data["acknowledged_by"] == "usuario-espanol"

    @pytest.mark.asyncio
    async def test_special_chars_in_metric_name(self, install_analyzer):
        """Special characters in metric_name should be passed through."""
        install_analyzer.check_metric.return_value = None

        req = _make_request(
            method="POST",
            body={"metric_name": "sys.cpu/usage@total", "value": 50.0},
        )
        resp = await AlertHandler.check_metric(req)

        assert resp.status == 200
        call_kwargs = install_analyzer.check_metric.call_args[1]
        assert call_kwargs["metric_name"] == "sys.cpu/usage@total"


# ---------------------------------------------------------------------------
# Integration / cross-endpoint edge cases
# ---------------------------------------------------------------------------


class TestIntegrationEdgeCases:
    @pytest.mark.asyncio
    async def test_list_calls_get_alert_analyzer(self):
        """Verify list_active goes through get_alert_analyzer()."""
        mock_a = MagicMock()
        mock_a.get_active_alerts.return_value = []
        set_alert_analyzer(mock_a)

        req = _make_request()
        resp = await AlertHandler.list_active(req)

        assert resp.status == 200
        mock_a.get_active_alerts.assert_called_once()

    @pytest.mark.asyncio
    async def test_acknowledge_calls_get_alert_analyzer(self):
        """Verify acknowledge goes through get_alert_analyzer()."""
        mock_a = MagicMock()
        mock_a.acknowledge_alert.return_value = True
        set_alert_analyzer(mock_a)

        req = _make_request(
            method="POST",
            match_info={"alert_id": "x-1"},
            body={"acknowledged_by": "admin"},
        )
        resp = await AlertHandler.acknowledge(req)

        assert resp.status == 200
        mock_a.acknowledge_alert.assert_called_once()

    @pytest.mark.asyncio
    async def test_resolve_calls_get_alert_analyzer(self):
        """Verify resolve goes through get_alert_analyzer()."""
        mock_a = MagicMock()
        mock_a.resolve_alert.return_value = True
        set_alert_analyzer(mock_a)

        req = _make_request(
            method="POST",
            match_info={"alert_id": "x-1"},
        )
        resp = await AlertHandler.resolve(req)

        assert resp.status == 200
        mock_a.resolve_alert.assert_called_once_with("x-1")

    @pytest.mark.asyncio
    async def test_set_threshold_calls_get_alert_analyzer(self):
        """Verify set_threshold goes through get_alert_analyzer()."""
        mock_a = MagicMock()
        set_alert_analyzer(mock_a)

        req = _make_request(
            method="POST",
            body={"metric_name": "cpu"},
        )
        resp = await AlertHandler.set_threshold(req)

        assert resp.status == 200
        mock_a.set_threshold.assert_called_once()

    @pytest.mark.asyncio
    async def test_check_metric_calls_get_alert_analyzer(self):
        """Verify check_metric goes through get_alert_analyzer()."""
        mock_a = MagicMock()
        mock_a.check_metric = AsyncMock(return_value=None)
        set_alert_analyzer(mock_a)

        req = _make_request(
            method="POST",
            body={"metric_name": "latency", "value": 10.0},
        )
        resp = await AlertHandler.check_metric(req)

        assert resp.status == 200
        mock_a.check_metric.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_check_metric_value_none_is_rejected(self, install_analyzer):
        """Explicit null value should be rejected."""
        req = _make_request(
            method="POST",
            body={"metric_name": "cpu_usage", "value": None},
        )
        resp = await AlertHandler.check_metric(req)

        assert resp.status == 400

    @pytest.mark.asyncio
    async def test_check_metric_value_false_is_accepted(self, install_analyzer):
        """Boolean False (0.0 as float) should be treated as 0."""
        install_analyzer.check_metric.return_value = None

        req = _make_request(
            method="POST",
            body={"metric_name": "flag", "value": False},
        )
        resp = await AlertHandler.check_metric(req)

        # False is not None, so it passes the `value is None` check
        # float(False) == 0.0
        assert resp.status == 200

    @pytest.mark.asyncio
    async def test_empty_body_threshold(self, install_analyzer):
        """Completely empty body should fail validation (no metric_name)."""
        req = _make_request(
            method="POST",
            body={},
        )
        resp = await AlertHandler.set_threshold(req)

        assert resp.status == 400
