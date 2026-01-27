"""Tests for autonomous alerts handler."""

import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import AioHTTPTestCase, unittest_run_loop

from aragora.server.handlers.autonomous import alerts


# =============================================================================
# Mock Alert Classes
# =============================================================================


class MockAlertSeverity:
    """Mock alert severity enum."""

    WARNING = MagicMock(value="warning")
    CRITICAL = MagicMock(value="critical")
    INFO = MagicMock(value="info")


class MockAlert:
    """Mock alert for testing."""

    def __init__(
        self,
        id: str = "test-alert-001",
        severity_value: str = "warning",
        title: str = "Test Alert",
        description: str = "Test description",
        source: str = "test",
        acknowledged: bool = False,
        acknowledged_by: str = None,
        debate_triggered: bool = False,
        debate_id: str = None,
        metadata: dict = None,
    ):
        self.id = id
        self.severity = MagicMock(value=severity_value)
        self.title = title
        self.description = description
        self.source = source
        self.timestamp = datetime.now()
        self.acknowledged = acknowledged
        self.acknowledged_by = acknowledged_by
        self.debate_triggered = debate_triggered
        self.debate_id = debate_id
        self.metadata = metadata or {}


class MockAlertAnalyzer:
    """Mock AlertAnalyzer for testing."""

    def __init__(self):
        self._alerts = []
        self._thresholds = {}

    def get_active_alerts(self):
        return [a for a in self._alerts if not getattr(a, "resolved", False)]

    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        for alert in self._alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                alert.acknowledged_by = acknowledged_by
                return True
        return False

    def resolve_alert(self, alert_id: str) -> bool:
        for alert in self._alerts:
            if alert.id == alert_id:
                alert.resolved = True
                return True
        return False

    def set_threshold(self, **kwargs):
        metric_name = kwargs.get("metric_name")
        self._thresholds[metric_name] = kwargs

    async def check_metric(self, **kwargs) -> MockAlert:
        return None


class MockAuthContext:
    """Mock authorization context."""

    def __init__(self, user_id="test-user", permissions=None):
        self.user_id = user_id
        self.permissions = permissions or {"alerts:read", "alerts:write", "alerts:admin"}


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
def mock_analyzer():
    """Create mock alert analyzer."""
    return MockAlertAnalyzer()


@pytest.fixture
def mock_auth_context():
    """Create mock auth context."""
    return MockAuthContext()


@pytest.fixture
def mock_permission_checker():
    """Create mock permission checker."""
    return MockPermissionChecker()


# =============================================================================
# Test AlertHandler.list_active
# =============================================================================


class TestAlertHandlerListActive:
    """Tests for GET /api/autonomous/alerts/active endpoint."""

    @pytest.mark.asyncio
    async def test_list_active_empty(
        self, mock_analyzer, mock_auth_context, mock_permission_checker
    ):
        """Should return empty list when no active alerts."""
        with (
            patch.object(alerts, "get_alert_analyzer", return_value=mock_analyzer),
            patch.object(
                alerts,
                "get_auth_context",
                AsyncMock(return_value=mock_auth_context),
            ),
            patch.object(
                alerts,
                "get_permission_checker",
                return_value=mock_permission_checker,
            ),
        ):
            request = MagicMock()
            response = await alerts.AlertHandler.list_active(request)

            assert response.status == 200
            body = json.loads(response.body)
            assert body["success"] is True
            assert body["alerts"] == []
            assert body["count"] == 0

    @pytest.mark.asyncio
    async def test_list_active_with_alerts(
        self, mock_analyzer, mock_auth_context, mock_permission_checker
    ):
        """Should return active alerts."""
        mock_analyzer._alerts = [
            MockAlert(id="alert-1", title="Alert 1"),
            MockAlert(id="alert-2", title="Alert 2"),
        ]

        with (
            patch.object(alerts, "get_alert_analyzer", return_value=mock_analyzer),
            patch.object(
                alerts,
                "get_auth_context",
                AsyncMock(return_value=mock_auth_context),
            ),
            patch.object(
                alerts,
                "get_permission_checker",
                return_value=mock_permission_checker,
            ),
        ):
            request = MagicMock()
            response = await alerts.AlertHandler.list_active(request)

            assert response.status == 200
            body = json.loads(response.body)
            assert body["success"] is True
            assert len(body["alerts"]) == 2
            assert body["count"] == 2

    @pytest.mark.asyncio
    async def test_list_active_unauthorized(self, mock_analyzer):
        """Should return 401 when unauthorized."""
        with (
            patch.object(alerts, "get_alert_analyzer", return_value=mock_analyzer),
            patch.object(
                alerts,
                "get_auth_context",
                AsyncMock(side_effect=alerts.UnauthorizedError("Not authenticated")),
            ),
        ):
            request = MagicMock()
            response = await alerts.AlertHandler.list_active(request)

            assert response.status == 401
            body = json.loads(response.body)
            assert body["success"] is False

    @pytest.mark.asyncio
    async def test_list_active_forbidden(self, mock_analyzer, mock_auth_context):
        """Should return 403 when permission denied."""
        mock_checker = MockPermissionChecker()
        mock_checker.check_permission = MagicMock(
            return_value=MockPermissionDecision(allowed=False, reason="No permission")
        )

        with (
            patch.object(alerts, "get_alert_analyzer", return_value=mock_analyzer),
            patch.object(
                alerts,
                "get_auth_context",
                AsyncMock(return_value=mock_auth_context),
            ),
            patch.object(
                alerts,
                "get_permission_checker",
                return_value=mock_checker,
            ),
        ):
            request = MagicMock()
            response = await alerts.AlertHandler.list_active(request)

            assert response.status == 403
            body = json.loads(response.body)
            assert body["success"] is False


# =============================================================================
# Test AlertHandler.acknowledge
# =============================================================================


class TestAlertHandlerAcknowledge:
    """Tests for POST /api/autonomous/alerts/{alert_id}/acknowledge endpoint."""

    @pytest.mark.asyncio
    async def test_acknowledge_success(
        self, mock_analyzer, mock_auth_context, mock_permission_checker
    ):
        """Should acknowledge alert successfully."""
        mock_analyzer._alerts = [MockAlert(id="alert-1")]

        with (
            patch.object(alerts, "get_alert_analyzer", return_value=mock_analyzer),
            patch.object(
                alerts,
                "get_auth_context",
                AsyncMock(return_value=mock_auth_context),
            ),
            patch.object(
                alerts,
                "get_permission_checker",
                return_value=mock_permission_checker,
            ),
        ):
            request = MagicMock()
            request.match_info.get.return_value = "alert-1"
            request.json = AsyncMock(return_value={"acknowledged_by": "test-user"})

            response = await alerts.AlertHandler.acknowledge(request)

            assert response.status == 200
            body = json.loads(response.body)
            assert body["success"] is True
            assert body["alert_id"] == "alert-1"

    @pytest.mark.asyncio
    async def test_acknowledge_not_found(
        self, mock_analyzer, mock_auth_context, mock_permission_checker
    ):
        """Should return 404 for non-existent alert."""
        with (
            patch.object(alerts, "get_alert_analyzer", return_value=mock_analyzer),
            patch.object(
                alerts,
                "get_auth_context",
                AsyncMock(return_value=mock_auth_context),
            ),
            patch.object(
                alerts,
                "get_permission_checker",
                return_value=mock_permission_checker,
            ),
        ):
            request = MagicMock()
            request.match_info.get.return_value = "non-existent"
            request.json = AsyncMock(return_value={})

            response = await alerts.AlertHandler.acknowledge(request)

            assert response.status == 404
            body = json.loads(response.body)
            assert body["success"] is False


# =============================================================================
# Test AlertHandler.resolve
# =============================================================================


class TestAlertHandlerResolve:
    """Tests for POST /api/autonomous/alerts/{alert_id}/resolve endpoint."""

    @pytest.mark.asyncio
    async def test_resolve_success(self, mock_analyzer, mock_auth_context, mock_permission_checker):
        """Should resolve alert successfully."""
        mock_analyzer._alerts = [MockAlert(id="alert-1")]

        with (
            patch.object(alerts, "get_alert_analyzer", return_value=mock_analyzer),
            patch.object(
                alerts,
                "get_auth_context",
                AsyncMock(return_value=mock_auth_context),
            ),
            patch.object(
                alerts,
                "get_permission_checker",
                return_value=mock_permission_checker,
            ),
        ):
            request = MagicMock()
            request.match_info.get.return_value = "alert-1"

            response = await alerts.AlertHandler.resolve(request)

            assert response.status == 200
            body = json.loads(response.body)
            assert body["success"] is True
            assert body["resolved"] is True

    @pytest.mark.asyncio
    async def test_resolve_not_found(
        self, mock_analyzer, mock_auth_context, mock_permission_checker
    ):
        """Should return 404 for non-existent alert."""
        with (
            patch.object(alerts, "get_alert_analyzer", return_value=mock_analyzer),
            patch.object(
                alerts,
                "get_auth_context",
                AsyncMock(return_value=mock_auth_context),
            ),
            patch.object(
                alerts,
                "get_permission_checker",
                return_value=mock_permission_checker,
            ),
        ):
            request = MagicMock()
            request.match_info.get.return_value = "non-existent"

            response = await alerts.AlertHandler.resolve(request)

            assert response.status == 404


# =============================================================================
# Test AlertHandler.set_threshold
# =============================================================================


class TestAlertHandlerSetThreshold:
    """Tests for POST /api/autonomous/alerts/thresholds endpoint."""

    @pytest.mark.asyncio
    async def test_set_threshold_success(
        self, mock_analyzer, mock_auth_context, mock_permission_checker
    ):
        """Should set threshold successfully."""
        with (
            patch.object(alerts, "get_alert_analyzer", return_value=mock_analyzer),
            patch.object(
                alerts,
                "get_auth_context",
                AsyncMock(return_value=mock_auth_context),
            ),
            patch.object(
                alerts,
                "get_permission_checker",
                return_value=mock_permission_checker,
            ),
        ):
            request = MagicMock()
            request.json = AsyncMock(
                return_value={
                    "metric_name": "cpu_usage",
                    "warning_threshold": 70.0,
                    "critical_threshold": 90.0,
                    "comparison": "gt",
                }
            )

            response = await alerts.AlertHandler.set_threshold(request)

            assert response.status == 200
            body = json.loads(response.body)
            assert body["success"] is True
            assert body["metric_name"] == "cpu_usage"

    @pytest.mark.asyncio
    async def test_set_threshold_missing_metric_name(
        self, mock_analyzer, mock_auth_context, mock_permission_checker
    ):
        """Should return 400 when metric_name missing."""
        with (
            patch.object(alerts, "get_alert_analyzer", return_value=mock_analyzer),
            patch.object(
                alerts,
                "get_auth_context",
                AsyncMock(return_value=mock_auth_context),
            ),
            patch.object(
                alerts,
                "get_permission_checker",
                return_value=mock_permission_checker,
            ),
        ):
            request = MagicMock()
            request.json = AsyncMock(return_value={})

            response = await alerts.AlertHandler.set_threshold(request)

            assert response.status == 400


# =============================================================================
# Test AlertHandler.check_metric
# =============================================================================


class TestAlertHandlerCheckMetric:
    """Tests for POST /api/autonomous/alerts/check endpoint."""

    @pytest.mark.asyncio
    async def test_check_metric_no_alert(
        self, mock_analyzer, mock_auth_context, mock_permission_checker
    ):
        """Should return no alert when below threshold."""
        with (
            patch.object(alerts, "get_alert_analyzer", return_value=mock_analyzer),
            patch.object(
                alerts,
                "get_auth_context",
                AsyncMock(return_value=mock_auth_context),
            ),
            patch.object(
                alerts,
                "get_permission_checker",
                return_value=mock_permission_checker,
            ),
        ):
            request = MagicMock()
            request.json = AsyncMock(
                return_value={
                    "metric_name": "cpu_usage",
                    "value": 50.0,
                }
            )

            response = await alerts.AlertHandler.check_metric(request)

            assert response.status == 200
            body = json.loads(response.body)
            assert body["success"] is True
            assert body["alert_generated"] is False

    @pytest.mark.asyncio
    async def test_check_metric_missing_fields(
        self, mock_analyzer, mock_auth_context, mock_permission_checker
    ):
        """Should return 400 when required fields missing."""
        with (
            patch.object(alerts, "get_alert_analyzer", return_value=mock_analyzer),
            patch.object(
                alerts,
                "get_auth_context",
                AsyncMock(return_value=mock_auth_context),
            ),
            patch.object(
                alerts,
                "get_permission_checker",
                return_value=mock_permission_checker,
            ),
        ):
            request = MagicMock()
            request.json = AsyncMock(return_value={"metric_name": "cpu_usage"})

            response = await alerts.AlertHandler.check_metric(request)

            assert response.status == 400


# =============================================================================
# Test Route Registration
# =============================================================================


class TestAlertHandlerRoutes:
    """Tests for route registration."""

    def test_register_routes(self):
        """Should register all alert routes."""
        app = web.Application()
        alerts.AlertHandler.register_routes(app)

        routes = [r.resource.canonical for r in app.router.routes()]
        assert "/api/v1/autonomous/alerts/active" in routes
        assert "/api/v1/autonomous/alerts/{alert_id}/acknowledge" in routes
        assert "/api/v1/autonomous/alerts/{alert_id}/resolve" in routes
        assert "/api/v1/autonomous/alerts/thresholds" in routes
        assert "/api/v1/autonomous/alerts/check" in routes


# =============================================================================
# Test Global Functions
# =============================================================================


class TestAlertAnalyzerSingleton:
    """Tests for alert analyzer singleton functions."""

    def test_get_alert_analyzer_creates_singleton(self):
        """get_alert_analyzer should return same instance."""
        alerts._alert_analyzer = None

        analyzer1 = alerts.get_alert_analyzer()
        analyzer2 = alerts.get_alert_analyzer()

        assert analyzer1 is analyzer2

        # Clean up
        alerts._alert_analyzer = None

    def test_set_alert_analyzer(self):
        """set_alert_analyzer should update the global instance."""
        mock = MockAlertAnalyzer()
        alerts.set_alert_analyzer(mock)

        assert alerts.get_alert_analyzer() is mock

        # Clean up
        alerts._alert_analyzer = None
