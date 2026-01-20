"""Alert management HTTP handlers."""

import logging
from typing import Optional

from aiohttp import web

from aragora.autonomous import AlertAnalyzer, AlertSeverity

logger = logging.getLogger(__name__)

# Global alert analyzer instance
_alert_analyzer: Optional[AlertAnalyzer] = None


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
    """HTTP handlers for alert operations."""

    @staticmethod
    async def list_active(request: web.Request) -> web.Response:
        """
        List all active (unresolved) alerts.

        GET /api/autonomous/alerts/active

        Returns:
            List of active alerts
        """
        try:
            analyzer = get_alert_analyzer()
            alerts = analyzer.get_active_alerts()

            return web.json_response({
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
            })

        except Exception as e:
            logger.error(f"Error listing alerts: {e}")
            return web.json_response(
                {"success": False, "error": str(e)},
                status=500,
            )

    @staticmethod
    async def acknowledge(request: web.Request) -> web.Response:
        """
        Acknowledge an alert.

        POST /api/autonomous/alerts/{alert_id}/acknowledge

        Body:
            acknowledged_by: str - Who is acknowledging

        Returns:
            Success status
        """
        alert_id = request.match_info.get("alert_id")

        try:
            data = await request.json()
            acknowledged_by = data.get("acknowledged_by", "api_user")

            analyzer = get_alert_analyzer()
            success = analyzer.acknowledge_alert(alert_id, acknowledged_by)

            if not success:
                return web.json_response(
                    {"success": False, "error": "Alert not found"},
                    status=404,
                )

            return web.json_response({
                "success": True,
                "alert_id": alert_id,
                "acknowledged_by": acknowledged_by,
            })

        except Exception as e:
            logger.error(f"Error acknowledging alert: {e}")
            return web.json_response(
                {"success": False, "error": str(e)},
                status=500,
            )

    @staticmethod
    async def resolve(request: web.Request) -> web.Response:
        """
        Resolve an alert.

        POST /api/autonomous/alerts/{alert_id}/resolve

        Returns:
            Success status
        """
        alert_id = request.match_info.get("alert_id")

        try:
            analyzer = get_alert_analyzer()
            success = analyzer.resolve_alert(alert_id)

            if not success:
                return web.json_response(
                    {"success": False, "error": "Alert not found"},
                    status=404,
                )

            return web.json_response({
                "success": True,
                "alert_id": alert_id,
                "resolved": True,
            })

        except Exception as e:
            logger.error(f"Error resolving alert: {e}")
            return web.json_response(
                {"success": False, "error": str(e)},
                status=500,
            )

    @staticmethod
    async def set_threshold(request: web.Request) -> web.Response:
        """
        Set alert threshold for a metric.

        POST /api/autonomous/alerts/thresholds

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
            data = await request.json()
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

            return web.json_response({
                "success": True,
                "metric_name": metric_name,
                "threshold_set": True,
            })

        except Exception as e:
            logger.error(f"Error setting threshold: {e}")
            return web.json_response(
                {"success": False, "error": str(e)},
                status=500,
            )

    @staticmethod
    async def check_metric(request: web.Request) -> web.Response:
        """
        Check a metric value against thresholds.

        POST /api/autonomous/alerts/check

        Body:
            metric_name: str - Name of the metric
            value: float - Current value
            source: str (optional) - Source of the metric

        Returns:
            Alert if threshold exceeded, null otherwise
        """
        try:
            data = await request.json()
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
                return web.json_response({
                    "success": True,
                    "alert_generated": True,
                    "alert": {
                        "id": alert.id,
                        "severity": alert.severity.value,
                        "title": alert.title,
                        "description": alert.description,
                    },
                })

            return web.json_response({
                "success": True,
                "alert_generated": False,
            })

        except Exception as e:
            logger.error(f"Error checking metric: {e}")
            return web.json_response(
                {"success": False, "error": str(e)},
                status=500,
            )

    @staticmethod
    def register_routes(app: web.Application, prefix: str = "/api/autonomous") -> None:
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
