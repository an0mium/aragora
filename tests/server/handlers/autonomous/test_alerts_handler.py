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


# =============================================================================
# Test Circuit Breaker Integration
# =============================================================================


class TestAlertCircuitBreaker:
    """Tests for circuit breaker integration."""

    def test_get_alert_circuit_breaker(self):
        """Test getting circuit breaker instance."""
        cb = alerts.get_alert_circuit_breaker()
        assert cb is not None
        assert cb.name == "alert_handler"

    def test_get_alert_circuit_breaker_status(self):
        """Test getting circuit breaker status dict."""
        status = alerts.get_alert_circuit_breaker_status()
        assert isinstance(status, dict)
        assert "name" in status
        assert status["name"] == "alert_handler"

    def test_circuit_breaker_is_singleton(self):
        """Test circuit breaker returns same instance."""
        cb1 = alerts.get_alert_circuit_breaker()
        cb2 = alerts.get_alert_circuit_breaker()
        assert cb1 is cb2

    def test_circuit_breaker_has_correct_config(self):
        """Test circuit breaker has expected configuration."""
        cb = alerts.get_alert_circuit_breaker()
        assert cb.failure_threshold == 5
        assert cb.cooldown_seconds == 30.0


# =============================================================================
# Test AlertHandler Initialization
# =============================================================================


class TestAlertHandlerInit:
    """Tests for AlertHandler initialization."""

    def test_init_with_none_context(self):
        """Test handler initialization with None context."""
        handler = alerts.AlertHandler(None)
        assert handler.ctx == {}

    def test_init_with_context(self):
        """Test handler initialization with context."""
        ctx = {"key": "value"}
        handler = alerts.AlertHandler(ctx)
        assert handler.ctx == ctx


# =============================================================================
# Test Alert Check Metric with Alert Generation
# =============================================================================


class TestAlertHandlerCheckMetricWithAlert:
    """Tests for check_metric endpoint with alert generation."""

    @pytest.mark.asyncio
    async def test_check_metric_generates_alert(
        self, mock_analyzer, mock_auth_context, mock_permission_checker
    ):
        """Should return alert when threshold exceeded."""
        generated_alert = MockAlert(
            id="generated-alert-1",
            severity_value="critical",
            title="CPU Critical",
            description="CPU usage critical",
        )
        mock_analyzer.check_metric = AsyncMock(return_value=generated_alert)

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
                    "value": 95.0,
                    "source": "monitoring",
                }
            )

            response = await alerts.AlertHandler.check_metric(request)

            assert response.status == 200
            body = json.loads(response.body)
            assert body["success"] is True
            assert body["alert_generated"] is True
            assert body["alert"]["id"] == "generated-alert-1"
            assert body["alert"]["severity"] == "critical"


# =============================================================================
# Test Error Handling
# =============================================================================


class TestAlertHandlerErrorHandling:
    """Tests for error handling in AlertHandler."""

    @pytest.mark.asyncio
    async def test_list_active_handles_exception(self, mock_auth_context, mock_permission_checker):
        """Should return 500 when analyzer raises exception."""
        mock_analyzer = MagicMock()
        mock_analyzer.get_active_alerts.side_effect = RuntimeError("Database error")

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

            assert response.status == 500
            body = json.loads(response.body)
            assert body["success"] is False
            assert "Database error" in body["error"]

    @pytest.mark.asyncio
    async def test_acknowledge_handles_exception(
        self, mock_analyzer, mock_auth_context, mock_permission_checker
    ):
        """Should return 500 when acknowledge raises exception."""
        mock_analyzer.acknowledge_alert = MagicMock(side_effect=RuntimeError("DB error"))

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
            request.json = AsyncMock(return_value={})

            response = await alerts.AlertHandler.acknowledge(request)

            assert response.status == 500

    @pytest.mark.asyncio
    async def test_resolve_handles_exception(
        self, mock_analyzer, mock_auth_context, mock_permission_checker
    ):
        """Should return 500 when resolve raises exception."""
        mock_analyzer.resolve_alert = MagicMock(side_effect=RuntimeError("DB error"))

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

            assert response.status == 500

    @pytest.mark.asyncio
    async def test_set_threshold_handles_exception(
        self, mock_analyzer, mock_auth_context, mock_permission_checker
    ):
        """Should return 500 when set_threshold raises exception."""
        mock_analyzer.set_threshold = MagicMock(side_effect=RuntimeError("Config error"))

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
            request.json = AsyncMock(return_value={"metric_name": "cpu"})

            response = await alerts.AlertHandler.set_threshold(request)

            assert response.status == 500

    @pytest.mark.asyncio
    async def test_check_metric_handles_exception(
        self, mock_analyzer, mock_auth_context, mock_permission_checker
    ):
        """Should return 500 when check_metric raises exception."""
        mock_analyzer.check_metric = AsyncMock(side_effect=RuntimeError("Check error"))

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
            request.json = AsyncMock(return_value={"metric_name": "cpu", "value": 50})

            response = await alerts.AlertHandler.check_metric(request)

            assert response.status == 500


# =============================================================================
# Test RBAC Permissions
# =============================================================================


class TestAlertHandlerPermissions:
    """Tests for RBAC permission checks."""

    @pytest.mark.asyncio
    async def test_acknowledge_forbidden(self, mock_analyzer, mock_auth_context):
        """Should return 403 when alerts:write permission denied."""
        mock_checker = MockPermissionChecker()
        mock_checker.check_permission = MagicMock(
            return_value=MockPermissionDecision(allowed=False, reason="No write permission")
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
            request.match_info.get.return_value = "alert-1"
            request.json = AsyncMock(return_value={})

            response = await alerts.AlertHandler.acknowledge(request)

            assert response.status == 403

    @pytest.mark.asyncio
    async def test_resolve_forbidden(self, mock_analyzer, mock_auth_context):
        """Should return 403 when alerts:write permission denied for resolve."""
        mock_checker = MockPermissionChecker()
        mock_checker.check_permission = MagicMock(
            return_value=MockPermissionDecision(allowed=False, reason="No write permission")
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
            request.match_info.get.return_value = "alert-1"

            response = await alerts.AlertHandler.resolve(request)

            assert response.status == 403

    @pytest.mark.asyncio
    async def test_set_threshold_forbidden(self, mock_analyzer, mock_auth_context):
        """Should return 403 when alerts:admin permission denied."""
        mock_checker = MockPermissionChecker()
        mock_checker.check_permission = MagicMock(
            return_value=MockPermissionDecision(allowed=False, reason="No admin permission")
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
            request.json = AsyncMock(return_value={"metric_name": "cpu"})

            response = await alerts.AlertHandler.set_threshold(request)

            assert response.status == 403

    @pytest.mark.asyncio
    async def test_check_metric_forbidden(self, mock_analyzer, mock_auth_context):
        """Should return 403 when alerts:write permission denied for check."""
        mock_checker = MockPermissionChecker()
        mock_checker.check_permission = MagicMock(
            return_value=MockPermissionDecision(allowed=False, reason="No write permission")
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
            request.json = AsyncMock(return_value={"metric_name": "cpu", "value": 50})

            response = await alerts.AlertHandler.check_metric(request)

            assert response.status == 403


# =============================================================================
# Test Authentication
# =============================================================================


class TestAlertHandlerAuthentication:
    """Tests for authentication handling."""

    @pytest.mark.asyncio
    async def test_acknowledge_unauthorized(self, mock_analyzer):
        """Should return 401 when not authenticated for acknowledge."""
        with (
            patch.object(alerts, "get_alert_analyzer", return_value=mock_analyzer),
            patch.object(
                alerts,
                "get_auth_context",
                AsyncMock(side_effect=alerts.UnauthorizedError("Not authenticated")),
            ),
        ):
            request = MagicMock()
            request.match_info.get.return_value = "alert-1"
            request.json = AsyncMock(return_value={})

            response = await alerts.AlertHandler.acknowledge(request)

            assert response.status == 401

    @pytest.mark.asyncio
    async def test_resolve_unauthorized(self, mock_analyzer):
        """Should return 401 when not authenticated for resolve."""
        with (
            patch.object(alerts, "get_alert_analyzer", return_value=mock_analyzer),
            patch.object(
                alerts,
                "get_auth_context",
                AsyncMock(side_effect=alerts.UnauthorizedError("Not authenticated")),
            ),
        ):
            request = MagicMock()
            request.match_info.get.return_value = "alert-1"

            response = await alerts.AlertHandler.resolve(request)

            assert response.status == 401

    @pytest.mark.asyncio
    async def test_set_threshold_unauthorized(self, mock_analyzer):
        """Should return 401 when not authenticated for set_threshold."""
        with (
            patch.object(alerts, "get_alert_analyzer", return_value=mock_analyzer),
            patch.object(
                alerts,
                "get_auth_context",
                AsyncMock(side_effect=alerts.UnauthorizedError("Not authenticated")),
            ),
        ):
            request = MagicMock()
            request.json = AsyncMock(return_value={"metric_name": "cpu"})

            response = await alerts.AlertHandler.set_threshold(request)

            assert response.status == 401

    @pytest.mark.asyncio
    async def test_check_metric_unauthorized(self, mock_analyzer):
        """Should return 401 when not authenticated for check_metric."""
        with (
            patch.object(alerts, "get_alert_analyzer", return_value=mock_analyzer),
            patch.object(
                alerts,
                "get_auth_context",
                AsyncMock(side_effect=alerts.UnauthorizedError("Not authenticated")),
            ),
        ):
            request = MagicMock()
            request.json = AsyncMock(return_value={"metric_name": "cpu", "value": 50})

            response = await alerts.AlertHandler.check_metric(request)

            assert response.status == 401


# =============================================================================
# Test Threshold Configuration
# =============================================================================


class TestThresholdConfiguration:
    """Tests for threshold configuration options."""

    @pytest.mark.asyncio
    async def test_set_threshold_with_all_options(
        self, mock_analyzer, mock_auth_context, mock_permission_checker
    ):
        """Should set threshold with all configuration options."""
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
                    "metric_name": "memory_usage",
                    "warning_threshold": 75.0,
                    "critical_threshold": 90.0,
                    "comparison": "gte",
                    "enabled": True,
                }
            )

            response = await alerts.AlertHandler.set_threshold(request)

            assert response.status == 200
            body = json.loads(response.body)
            assert body["success"] is True
            assert body["metric_name"] == "memory_usage"

            # Verify threshold was stored in analyzer
            assert "memory_usage" in mock_analyzer._thresholds
            stored = mock_analyzer._thresholds["memory_usage"]
            assert stored["warning_threshold"] == 75.0
            assert stored["critical_threshold"] == 90.0
            assert stored["comparison"] == "gte"
            assert stored["enabled"] is True

    @pytest.mark.asyncio
    async def test_set_threshold_with_disabled(
        self, mock_analyzer, mock_auth_context, mock_permission_checker
    ):
        """Should set threshold as disabled."""
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
                    "metric_name": "disk_usage",
                    "enabled": False,
                }
            )

            response = await alerts.AlertHandler.set_threshold(request)

            assert response.status == 200


# =============================================================================
# Test Check Metric with Metadata
# =============================================================================


class TestCheckMetricMetadata:
    """Tests for check_metric with metadata."""

    @pytest.mark.asyncio
    async def test_check_metric_with_metadata(
        self, mock_analyzer, mock_auth_context, mock_permission_checker
    ):
        """Should pass metadata to check_metric."""
        mock_analyzer.check_metric = AsyncMock(return_value=None)

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
                    "metric_name": "request_latency",
                    "value": 250.0,
                    "source": "api_gateway",
                    "metadata": {"endpoint": "/api/v1/debates", "method": "POST"},
                }
            )

            response = await alerts.AlertHandler.check_metric(request)

            assert response.status == 200

            # Verify metadata was passed
            mock_analyzer.check_metric.assert_called_once()
            call_kwargs = mock_analyzer.check_metric.call_args[1]
            assert call_kwargs["metadata"] == {"endpoint": "/api/v1/debates", "method": "POST"}


# =============================================================================
# Test Route Registration with Custom Prefix
# =============================================================================


class TestRouteRegistrationPrefix:
    """Tests for route registration with custom prefix."""

    def test_register_routes_custom_prefix(self):
        """Should register routes with custom prefix."""
        app = web.Application()
        alerts.AlertHandler.register_routes(app, prefix="/custom/api")

        routes = [r.resource.canonical for r in app.router.routes()]
        assert "/custom/api/alerts/active" in routes
        assert "/custom/api/alerts/{alert_id}/acknowledge" in routes
        assert "/custom/api/alerts/{alert_id}/resolve" in routes
        assert "/custom/api/alerts/thresholds" in routes
        assert "/custom/api/alerts/check" in routes


# =============================================================================
# Test Acknowledge with User ID from Auth Context
# =============================================================================


class TestAcknowledgeUserId:
    """Tests for acknowledged_by user ID handling."""

    @pytest.mark.asyncio
    async def test_acknowledge_uses_auth_user_when_not_provided(
        self, mock_analyzer, mock_auth_context, mock_permission_checker
    ):
        """Should use auth context user_id when acknowledged_by not in body."""
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
            request.json = AsyncMock(return_value={})  # No acknowledged_by

            response = await alerts.AlertHandler.acknowledge(request)

            assert response.status == 200
            body = json.loads(response.body)
            assert body["acknowledged_by"] == "test-user"  # From mock_auth_context
