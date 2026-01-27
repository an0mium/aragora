"""Monitoring HTTP handlers (trends and anomalies)."""

import logging
from typing import Optional

from aiohttp import web

from aragora.autonomous import TrendMonitor, AnomalyDetector
from aragora.server.handlers.utils.auth import (
    get_auth_context,
    UnauthorizedError,
    ForbiddenError,
)
from aragora.rbac.checker import get_permission_checker

logger = logging.getLogger(__name__)

# Global instances
_trend_monitor: Optional[TrendMonitor] = None
_anomaly_detector: Optional[AnomalyDetector] = None


def get_trend_monitor() -> TrendMonitor:
    """Get or create the global trend monitor instance."""
    global _trend_monitor
    if _trend_monitor is None:
        _trend_monitor = TrendMonitor()
    return _trend_monitor


def set_trend_monitor(monitor: TrendMonitor) -> None:
    """Set the global trend monitor instance."""
    global _trend_monitor
    _trend_monitor = monitor


def get_anomaly_detector() -> AnomalyDetector:
    """Get or create the global anomaly detector instance."""
    global _anomaly_detector
    if _anomaly_detector is None:
        _anomaly_detector = AnomalyDetector()
    return _anomaly_detector


def set_anomaly_detector(detector: AnomalyDetector) -> None:
    """Set the global anomaly detector instance."""
    global _anomaly_detector
    _anomaly_detector = detector


class MonitoringHandler:
    """HTTP handlers for monitoring operations."""

    @staticmethod
    async def record_metric(request: web.Request) -> web.Response:
        """
        Record a metric value for trend and anomaly detection.

        POST /api/autonomous/monitoring/record

        Requires authentication and 'autonomous:write' permission.

        Body:
            metric_name: str - Name of the metric
            value: float - Current value

        Returns:
            Trend data and anomaly if detected
        """
        try:
            # RBAC check
            auth_ctx = await get_auth_context(request, require_auth=True)
            checker = get_permission_checker()
            decision = checker.check_permission(auth_ctx, "autonomous:write")
            if not decision.allowed:
                raise ForbiddenError(f"Permission denied: {decision.reason}")

            data = await request.json()
            metric_name = data.get("metric_name")
            value = data.get("value")

            if not metric_name or value is None:
                return web.json_response(
                    {"success": False, "error": "metric_name and value are required"},
                    status=400,
                )

            value = float(value)

            # Record in trend monitor
            trend_monitor = get_trend_monitor()
            trend_monitor.record(metric_name, value)

            # Record in anomaly detector
            anomaly_detector = get_anomaly_detector()
            anomaly = anomaly_detector.record(metric_name, value)

            response = {
                "success": True,
                "metric_name": metric_name,
                "value": value,
                "anomaly_detected": anomaly is not None,
            }

            if anomaly:
                response["anomaly"] = {
                    "id": anomaly.id,
                    "severity": anomaly.severity.value,
                    "deviation": anomaly.deviation,
                    "expected_value": anomaly.expected_value,
                    "description": anomaly.description,
                }

            return web.json_response(response)

        except UnauthorizedError as e:
            return web.json_response({"success": False, "error": str(e)}, status=401)
        except ForbiddenError as e:
            return web.json_response({"success": False, "error": str(e)}, status=403)
        except Exception as e:
            logger.error(f"Error recording metric: {e}")
            return web.json_response(
                {"success": False, "error": str(e)},
                status=500,
            )

    @staticmethod
    async def get_trend(request: web.Request) -> web.Response:
        """
        Get trend for a specific metric.

        GET /api/autonomous/monitoring/trends/{metric_name}

        Requires authentication and 'autonomous:read' permission.

        Query params:
            period_seconds: int (optional) - Time period to analyze

        Returns:
            Trend data
        """
        metric_name = request.match_info.get("metric_name")
        period_seconds = request.query.get("period_seconds")

        try:
            # RBAC check
            auth_ctx = await get_auth_context(request, require_auth=True)
            checker = get_permission_checker()
            decision = checker.check_permission(auth_ctx, "autonomous:read")
            if not decision.allowed:
                raise ForbiddenError(f"Permission denied: {decision.reason}")

            trend_monitor = get_trend_monitor()
            trend = trend_monitor.get_trend(
                metric_name,
                period_seconds=int(period_seconds) if period_seconds else None,
            )

            if not trend:
                return web.json_response(
                    {
                        "success": True,
                        "trend": None,
                        "message": "Insufficient data for trend analysis",
                    }
                )

            return web.json_response(
                {
                    "success": True,
                    "trend": {
                        "metric_name": trend.metric_name,
                        "direction": trend.direction.value,
                        "current_value": trend.current_value,
                        "previous_value": trend.previous_value,
                        "change_percent": trend.change_percent,
                        "period_start": trend.period_start.isoformat(),
                        "period_end": trend.period_end.isoformat(),
                        "data_points": trend.data_points,
                        "confidence": trend.confidence,
                    },
                }
            )

        except UnauthorizedError as e:
            return web.json_response({"success": False, "error": str(e)}, status=401)
        except ForbiddenError as e:
            return web.json_response({"success": False, "error": str(e)}, status=403)
        except Exception as e:
            logger.error(f"Error getting trend: {e}")
            return web.json_response(
                {"success": False, "error": str(e)},
                status=500,
            )

    @staticmethod
    async def get_all_trends(request: web.Request) -> web.Response:
        """
        Get trends for all monitored metrics.

        GET /api/autonomous/monitoring/trends

        Requires authentication and 'autonomous:read' permission.

        Returns:
            Dict of metric_name -> trend data
        """
        try:
            # RBAC check
            auth_ctx = await get_auth_context(request, require_auth=True)
            checker = get_permission_checker()
            decision = checker.check_permission(auth_ctx, "autonomous:read")
            if not decision.allowed:
                raise ForbiddenError(f"Permission denied: {decision.reason}")

            trend_monitor = get_trend_monitor()
            trends = trend_monitor.get_all_trends()

            return web.json_response(
                {
                    "success": True,
                    "trends": {
                        name: {
                            "direction": trend.direction.value,
                            "current_value": trend.current_value,
                            "change_percent": trend.change_percent,
                            "data_points": trend.data_points,
                            "confidence": trend.confidence,
                        }
                        for name, trend in trends.items()
                    },
                    "count": len(trends),
                }
            )

        except UnauthorizedError as e:
            return web.json_response({"success": False, "error": str(e)}, status=401)
        except ForbiddenError as e:
            return web.json_response({"success": False, "error": str(e)}, status=403)
        except Exception as e:
            logger.error(f"Error getting trends: {e}")
            return web.json_response(
                {"success": False, "error": str(e)},
                status=500,
            )

    @staticmethod
    async def get_anomalies(request: web.Request) -> web.Response:
        """
        Get recent anomalies.

        GET /api/autonomous/monitoring/anomalies

        Requires authentication and 'autonomous:read' permission.

        Query params:
            hours: int (optional) - Hours to look back (default: 24)
            metric_name: str (optional) - Filter by metric

        Returns:
            List of recent anomalies
        """
        try:
            # RBAC check
            auth_ctx = await get_auth_context(request, require_auth=True)
            checker = get_permission_checker()
            decision = checker.check_permission(auth_ctx, "autonomous:read")
            if not decision.allowed:
                raise ForbiddenError(f"Permission denied: {decision.reason}")

            hours = int(request.query.get("hours", "24"))
            metric_name = request.query.get("metric_name")

            anomaly_detector = get_anomaly_detector()
            anomalies = anomaly_detector.get_recent_anomalies(
                hours=hours,
                metric_name=metric_name,
            )

            return web.json_response(
                {
                    "success": True,
                    "anomalies": [
                        {
                            "id": a.id,
                            "metric_name": a.metric_name,
                            "value": a.value,
                            "expected_value": a.expected_value,
                            "deviation": a.deviation,
                            "timestamp": a.timestamp.isoformat(),
                            "severity": a.severity.value,
                            "description": a.description,
                        }
                        for a in anomalies
                    ],
                    "count": len(anomalies),
                }
            )

        except UnauthorizedError as e:
            return web.json_response({"success": False, "error": str(e)}, status=401)
        except ForbiddenError as e:
            return web.json_response({"success": False, "error": str(e)}, status=403)
        except Exception as e:
            logger.error(f"Error getting anomalies: {e}")
            return web.json_response(
                {"success": False, "error": str(e)},
                status=500,
            )

    @staticmethod
    async def get_baseline_stats(request: web.Request) -> web.Response:
        """
        Get baseline statistics for a metric.

        GET /api/autonomous/monitoring/baseline/{metric_name}

        Requires authentication and 'autonomous:read' permission.

        Returns:
            Baseline statistics (mean, stdev, min, max, median)
        """
        metric_name = request.match_info.get("metric_name")

        try:
            # RBAC check
            auth_ctx = await get_auth_context(request, require_auth=True)
            checker = get_permission_checker()
            decision = checker.check_permission(auth_ctx, "autonomous:read")
            if not decision.allowed:
                raise ForbiddenError(f"Permission denied: {decision.reason}")

            anomaly_detector = get_anomaly_detector()
            stats = anomaly_detector.get_baseline_stats(metric_name)

            if not stats:
                return web.json_response(
                    {
                        "success": True,
                        "stats": None,
                        "message": "Insufficient data for baseline statistics",
                    }
                )

            return web.json_response(
                {
                    "success": True,
                    "metric_name": metric_name,
                    "stats": stats,
                }
            )

        except UnauthorizedError as e:
            return web.json_response({"success": False, "error": str(e)}, status=401)
        except ForbiddenError as e:
            return web.json_response({"success": False, "error": str(e)}, status=403)
        except Exception as e:
            logger.error(f"Error getting baseline stats: {e}")
            return web.json_response(
                {"success": False, "error": str(e)},
                status=500,
            )

    @staticmethod
    def register_routes(app: web.Application, prefix: str = "/api/v1/autonomous") -> None:
        """Register monitoring routes with the application."""
        app.router.add_post(
            f"{prefix}/monitoring/record",
            MonitoringHandler.record_metric,
        )
        app.router.add_get(
            f"{prefix}/monitoring/trends",
            MonitoringHandler.get_all_trends,
        )
        app.router.add_get(
            f"{prefix}/monitoring/trends/{{metric_name}}",
            MonitoringHandler.get_trend,
        )
        app.router.add_get(
            f"{prefix}/monitoring/anomalies",
            MonitoringHandler.get_anomalies,
        )
        app.router.add_get(
            f"{prefix}/monitoring/baseline/{{metric_name}}",
            MonitoringHandler.get_baseline_stats,
        )
