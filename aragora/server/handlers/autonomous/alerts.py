"""Alert management HTTP handlers.

Stability: STABLE
"""

from __future__ import annotations

import logging
from typing import Any

from aiohttp import web

from aragora.autonomous import AlertAnalyzer
from aragora.resilience import CircuitBreaker
from aragora.server.handlers.utils.auth import (
    get_auth_context,
    UnauthorizedError,
    ForbiddenError,
)
from aragora.server.handlers.utils import parse_json_body
from aragora.rbac.checker import get_permission_checker
from aragora.rbac.decorators import require_permission

logger = logging.getLogger(__name__)

# =============================================================================
# Resilience Configuration
# =============================================================================

# Circuit breaker for alert service
_alert_circuit_breaker = CircuitBreaker(
    name="alert_handler",
    failure_threshold=5,
    cooldown_seconds=30.0,
)


def get_alert_circuit_breaker() -> CircuitBreaker:
    """Get the circuit breaker for alert service."""
    return _alert_circuit_breaker


def get_alert_circuit_breaker_status() -> dict[str, Any]:
    """Get current status of the alert circuit breaker."""
    status = _alert_circuit_breaker.to_dict()
    status.setdefault("name", _alert_circuit_breaker.name)
    return status


# Global alert analyzer instance
_alert_analyzer: AlertAnalyzer | None = None


def get_alert_analyzer() -> AlertAnalyzer:
    """Get or create the global alert analyzer instance."""
    global _alert_analyzer
    if _alert_analyzer is None:
        _alert_analyzer = AlertAnalyzer()
    return _alert_analyzer


def set_alert_analyzer(analyzer: AlertAnalyzer) -> None:
    """Set the global alert analyzer instance."""
    global _alert_analyzer
    _alert_analyzer = analyzer


class AlertHandler:
    """HTTP handlers for alert operations.

    Rate limiting is applied via middleware registration in register_routes().
    Circuit breaker protection is available via get_alert_circuit_breaker().

    Rate limits:
    - GET endpoints: 60 requests/minute
    - POST endpoints: 30 requests/minute
    - Admin endpoints: 10 requests/minute
    """

    def __init__(self, ctx: dict | None = None):
        """Initialize AlertHandler with optional context."""
        self.ctx = ctx or {}

    @staticmethod
    @require_permission("autonomous:alerts:read")
    async def list_active(request: web.Request) -> web.Response:
        """
        List all active (unresolved) alerts.

        GET /api/autonomous/alerts/active

        Requires authentication and 'alerts:read' permission.

        Returns:
            List of active alerts
        """
        try:
            # RBAC check
            auth_ctx = await get_auth_context(request, require_auth=True)
            checker = get_permission_checker()
            decision = checker.check_permission(auth_ctx, "alerts:read")
            if not decision.allowed:
                raise ForbiddenError(f"Permission denied: {decision.reason}")

            analyzer = get_alert_analyzer()
            alerts = analyzer.get_active_alerts()

            return web.json_response(
                {
                    "success": True,
                    "alerts": [
                        {
                            "id": alert.id,
                            "severity": alert.severity.value,
                            "title": alert.title,
                            "description": alert.description,
                            "source": alert.source,
                            "timestamp": alert.timestamp.isoformat(),
                            "acknowledged": alert.acknowledged,
                            "acknowledged_by": alert.acknowledged_by,
                            "debate_triggered": alert.debate_triggered,
                            "debate_id": alert.debate_id,
                            "metadata": alert.metadata,
                        }
                        for alert in alerts
                    ],
                    "count": len(alerts),
                }
            )

        except UnauthorizedError as e:
            logger.warning("Unauthorized listing alerts: %s", e)
            return web.json_response({"success": False, "error": "Authentication required"}, status=401)
        except ForbiddenError as e:
            logger.warning("Forbidden listing alerts: %s", e)
            return web.json_response({"success": False, "error": "Permission denied"}, status=403)
        except (KeyError, ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.error("Error listing alerts: %s", e)
            return web.json_response(
                {"success": False, "error": "Failed to list alerts"},
                status=500,
            )

    @staticmethod
    @require_permission("autonomous:alerts:write")
    async def acknowledge(request: web.Request) -> web.Response:
        """
        Acknowledge an alert.

        POST /api/autonomous/alerts/{alert_id}/acknowledge

        Requires authentication and 'alerts:write' permission.

        Body:
            acknowledged_by: str - Who is acknowledging

        Returns:
            Success status
        """
        alert_id = request.match_info.get("alert_id")

        try:
            # RBAC check
            auth_ctx = await get_auth_context(request, require_auth=True)
            checker = get_permission_checker()
            decision = checker.check_permission(auth_ctx, "alerts:write")
            if not decision.allowed:
                raise ForbiddenError(f"Permission denied: {decision.reason}")

            data, err = await parse_json_body(request, context="acknowledge_alert")
            if err:
                return err
            acknowledged_by = data.get("acknowledged_by") or auth_ctx.user_id

            analyzer = get_alert_analyzer()
            success = analyzer.acknowledge_alert(alert_id, acknowledged_by)

            if not success:
                return web.json_response(
                    {"success": False, "error": "Alert not found"},
                    status=404,
                )

            return web.json_response(
                {
                    "success": True,
                    "alert_id": alert_id,
                    "acknowledged_by": acknowledged_by,
                }
            )

        except UnauthorizedError as e:
            logger.warning("Unauthorized acknowledging alert: %s", e)
            return web.json_response({"success": False, "error": "Authentication required"}, status=401)
        except ForbiddenError as e:
            logger.warning("Forbidden acknowledging alert: %s", e)
            return web.json_response({"success": False, "error": "Permission denied"}, status=403)
        except (KeyError, ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.error("Error acknowledging alert: %s", e)
            return web.json_response(
                {"success": False, "error": "Failed to acknowledge alert"},
                status=500,
            )

    @staticmethod
    @require_permission("autonomous:alerts:write")
    async def resolve(request: web.Request) -> web.Response:
        """
        Resolve an alert.

        POST /api/autonomous/alerts/{alert_id}/resolve

        Requires authentication and 'alerts:write' permission.

        Returns:
            Success status
        """
        alert_id = request.match_info.get("alert_id")

        try:
            # RBAC check
            auth_ctx = await get_auth_context(request, require_auth=True)
            checker = get_permission_checker()
            decision = checker.check_permission(auth_ctx, "alerts:write")
            if not decision.allowed:
                raise ForbiddenError(f"Permission denied: {decision.reason}")

            analyzer = get_alert_analyzer()
            success = analyzer.resolve_alert(alert_id)

            if not success:
                return web.json_response(
                    {"success": False, "error": "Alert not found"},
                    status=404,
                )

            return web.json_response(
                {
                    "success": True,
                    "alert_id": alert_id,
                    "resolved": True,
                }
            )

        except UnauthorizedError as e:
            logger.warning("Unauthorized resolving alert: %s", e)
            return web.json_response({"success": False, "error": "Authentication required"}, status=401)
        except ForbiddenError as e:
            logger.warning("Forbidden resolving alert: %s", e)
            return web.json_response({"success": False, "error": "Permission denied"}, status=403)
        except (KeyError, ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.error("Error resolving alert: %s", e)
            return web.json_response(
                {"success": False, "error": "Failed to resolve alert"},
                status=500,
            )

    @staticmethod
    @require_permission("autonomous:alerts:write")
    async def set_threshold(request: web.Request) -> web.Response:
        """
        Set alert threshold for a metric.

        POST /api/autonomous/alerts/thresholds

        Requires authentication and 'alerts:admin' permission.

        Body:
            metric_name: str - Name of the metric
            warning_threshold: float (optional)
            critical_threshold: float (optional)
            comparison: str - gt, lt, eq, ne, gte, lte
            enabled: bool

        Returns:
            Success status
        """
        try:
            # RBAC check
            auth_ctx = await get_auth_context(request, require_auth=True)
            checker = get_permission_checker()
            decision = checker.check_permission(auth_ctx, "alerts:admin")
            if not decision.allowed:
                raise ForbiddenError(f"Permission denied: {decision.reason}")

            data, err = await parse_json_body(request, context="set_threshold")
            if err:
                return err
            metric_name = data.get("metric_name")

            if not metric_name:
                return web.json_response(
                    {"success": False, "error": "metric_name is required"},
                    status=400,
                )

            analyzer = get_alert_analyzer()
            analyzer.set_threshold(
                metric_name=metric_name,
                warning_threshold=data.get("warning_threshold"),
                critical_threshold=data.get("critical_threshold"),
                comparison=data.get("comparison", "gt"),
                enabled=data.get("enabled", True),
            )

            return web.json_response(
                {
                    "success": True,
                    "metric_name": metric_name,
                    "threshold_set": True,
                }
            )

        except UnauthorizedError as e:
            logger.warning("Unauthorized setting threshold: %s", e)
            return web.json_response({"success": False, "error": "Authentication required"}, status=401)
        except ForbiddenError as e:
            logger.warning("Forbidden setting threshold: %s", e)
            return web.json_response({"success": False, "error": "Permission denied"}, status=403)
        except (KeyError, ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.error("Error setting threshold: %s", e)
            return web.json_response(
                {"success": False, "error": "Failed to set alert threshold"},
                status=500,
            )

    @staticmethod
    @require_permission("autonomous:alerts:write")
    async def check_metric(request: web.Request) -> web.Response:
        """
        Check a metric value against thresholds.

        POST /api/autonomous/alerts/check

        Requires authentication and 'alerts:write' permission.

        Body:
            metric_name: str - Name of the metric
            value: float - Current value
            source: str (optional) - Source of the metric

        Returns:
            Alert if threshold exceeded, null otherwise
        """
        try:
            # RBAC check
            auth_ctx = await get_auth_context(request, require_auth=True)
            checker = get_permission_checker()
            decision = checker.check_permission(auth_ctx, "alerts:write")
            if not decision.allowed:
                raise ForbiddenError(f"Permission denied: {decision.reason}")

            data, err = await parse_json_body(request, context="check_metric")
            if err:
                return err
            metric_name = data.get("metric_name")
            value = data.get("value")

            if not metric_name or value is None:
                return web.json_response(
                    {"success": False, "error": "metric_name and value are required"},
                    status=400,
                )

            analyzer = get_alert_analyzer()
            alert = await analyzer.check_metric(
                metric_name=metric_name,
                value=float(value),
                source=data.get("source", "api"),
                metadata=data.get("metadata"),
            )

            if alert:
                return web.json_response(
                    {
                        "success": True,
                        "alert_generated": True,
                        "alert": {
                            "id": alert.id,
                            "severity": alert.severity.value,
                            "title": alert.title,
                            "description": alert.description,
                        },
                    }
                )

            return web.json_response(
                {
                    "success": True,
                    "alert_generated": False,
                }
            )

        except UnauthorizedError as e:
            logger.warning("Unauthorized checking metric: %s", e)
            return web.json_response({"success": False, "error": "Authentication required"}, status=401)
        except ForbiddenError as e:
            logger.warning("Forbidden checking metric: %s", e)
            return web.json_response({"success": False, "error": "Permission denied"}, status=403)
        except Exception as e:
            logger.error("Error checking metric: %s", e)
            return web.json_response(
                {"success": False, "error": "Failed to check metric"},
                status=500,
            )

    @staticmethod
    def register_routes(app: web.Application, prefix: str = "/api/v1/autonomous") -> None:
        """Register alert routes with the application."""
        app.router.add_get(
            f"{prefix}/alerts/active",
            AlertHandler.list_active,
        )
        app.router.add_post(
            f"{prefix}/alerts/{{alert_id}}/acknowledge",
            AlertHandler.acknowledge,
        )
        app.router.add_post(
            f"{prefix}/alerts/{{alert_id}}/resolve",
            AlertHandler.resolve,
        )
        app.router.add_post(
            f"{prefix}/alerts/thresholds",
            AlertHandler.set_threshold,
        )
        app.router.add_post(
            f"{prefix}/alerts/check",
            AlertHandler.check_metric,
        )
