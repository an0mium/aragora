"""
Monitoring HTTP handlers (trends and anomalies).

Stability: STABLE

Provides API endpoints for monitoring operations:
- Recording metrics for trend and anomaly detection
- Retrieving trend data for specific metrics
- Listing all monitored trends
- Fetching recent anomalies
- Getting baseline statistics

Features:
- Circuit breaker pattern for resilient monitoring system access
- Rate limiting (30-60 requests/minute depending on endpoint)
- RBAC permission checks (autonomous:read, autonomous:write)
- Input validation with safe metric name patterns
- Comprehensive error handling with safe error messages
"""

from __future__ import annotations

__all__ = [
    "MonitoringHandler",
    "MonitoringCircuitBreaker",
    "get_monitoring_circuit_breaker_status",
    "get_trend_monitor",
    "set_trend_monitor",
    "get_anomaly_detector",
    "set_anomaly_detector",
    "_clear_monitoring_components",
]

import logging
import re
import threading
import time
from typing import Any

from aiohttp import web

from aragora.autonomous import TrendMonitor, AnomalyDetector
from aragora.server.handlers.utils.auth import (
    get_auth_context,
    UnauthorizedError,
    ForbiddenError,
)
from aragora.server.handlers.utils import parse_json_body
from aragora.rbac.checker import get_permission_checker
from aragora.rbac.decorators import require_permission
from aragora.server.validation.query_params import safe_query_int
from aragora.server.handlers.utils.rate_limit import rate_limit

logger = logging.getLogger(__name__)

# RBAC permission keys for autonomous operations
AUTONOMOUS_READ_PERMISSION = "autonomous:read"
AUTONOMOUS_WRITE_PERMISSION = "autonomous:write"

# Metric name validation pattern (alphanumeric, underscores, hyphens, dots)
SAFE_METRIC_NAME_PATTERN = re.compile(r"^[a-zA-Z][a-zA-Z0-9_\-\.]{0,127}$")

# Value bounds for metric recording
MIN_METRIC_VALUE = -1e15
MAX_METRIC_VALUE = 1e15


def validate_metric_name(metric_name: str | None) -> tuple[bool, str]:
    """Validate a metric name.

    Args:
        metric_name: The metric name to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not metric_name:
        return False, "metric_name is required"

    if not isinstance(metric_name, str):
        return False, "metric_name must be a string"

    if len(metric_name) > 128:
        return False, "metric_name must be 128 characters or less"

    if not SAFE_METRIC_NAME_PATTERN.match(metric_name):
        return False, (
            "metric_name must start with a letter and contain only "
            "alphanumeric characters, underscores, hyphens, and dots"
        )

    return True, ""


def validate_metric_value(value: Any) -> tuple[bool, float | None, str]:
    """Validate and convert a metric value.

    Args:
        value: The value to validate

    Returns:
        Tuple of (is_valid, converted_value, error_message)
    """
    if value is None:
        return False, None, "value is required"

    try:
        float_value = float(value)
    except (TypeError, ValueError):
        return False, None, "value must be a valid number"

    if float_value != float_value:  # NaN check
        return False, None, "value cannot be NaN"

    if float_value in (float("inf"), float("-inf")):
        return False, None, "value cannot be infinite"

    if float_value < MIN_METRIC_VALUE or float_value > MAX_METRIC_VALUE:
        return False, None, f"value must be between {MIN_METRIC_VALUE} and {MAX_METRIC_VALUE}"

    return True, float_value, ""


# =============================================================================
# Circuit Breaker for Monitoring System Access
# =============================================================================


class MonitoringCircuitBreaker:
    """Circuit breaker for monitoring system access.

    Prevents cascading failures when monitoring components are unavailable.
    Uses a simple state machine: CLOSED -> OPEN -> HALF_OPEN -> CLOSED.
    """

    # State constants
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(
        self,
        failure_threshold: int = 5,
        cooldown_seconds: float = 30.0,
        half_open_max_calls: int = 3,
    ):
        """Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            cooldown_seconds: Time to wait before allowing test calls
            half_open_max_calls: Number of test calls in half-open state
        """
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self.half_open_max_calls = half_open_max_calls

        self._state = self.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: float | None = None
        self._half_open_calls = 0
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        """Get current circuit state."""
        with self._lock:
            return self._check_state()

    def _check_state(self) -> str:
        """Check and potentially transition state (must hold lock)."""
        if self._state == self.OPEN:
            # Check if cooldown has elapsed
            if (
                self._last_failure_time is not None
                and time.time() - self._last_failure_time >= self.cooldown_seconds
            ):
                logger.info("Monitoring circuit breaker transitioning to HALF_OPEN")
                self._state = self.HALF_OPEN
                self._half_open_calls = 0
        return self._state

    def is_allowed(self) -> bool:
        """Check if a request should be allowed through.

        Returns:
            True if request is allowed, False if circuit is open.
        """
        with self._lock:
            state = self._check_state()

            if state == self.CLOSED:
                return True
            elif state == self.HALF_OPEN:
                # Allow limited calls in half-open state
                if self._half_open_calls < self.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False
            else:  # OPEN
                return False

    def record_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            if self._state == self.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.half_open_max_calls:
                    logger.info("Monitoring circuit breaker closing after successful tests")
                    self._state = self.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
            elif self._state == self.CLOSED:
                # Reset failure count on success
                if self._failure_count > 0:
                    self._failure_count = 0

    def record_failure(self) -> None:
        """Record a failed call."""
        with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()

            if self._state == self.HALF_OPEN:
                logger.warning("Monitoring circuit breaker opening after half-open failure")
                self._state = self.OPEN
                self._success_count = 0
            elif self._state == self.CLOSED and self._failure_count >= self.failure_threshold:
                logger.warning(
                    "Monitoring circuit breaker opening after %d failures",
                    self._failure_count,
                )
                self._state = self.OPEN

    def reset(self) -> None:
        """Reset circuit breaker to closed state."""
        with self._lock:
            self._state = self.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._last_failure_time = None
            self._half_open_calls = 0

    def get_status(self) -> dict[str, Any]:
        """Get current circuit breaker status."""
        with self._lock:
            return {
                "state": self._check_state(),
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "failure_threshold": self.failure_threshold,
                "cooldown_seconds": self.cooldown_seconds,
                "last_failure_time": self._last_failure_time,
            }


# Global circuit breaker for monitoring
_monitoring_circuit_breaker: MonitoringCircuitBreaker | None = None
_circuit_breaker_lock = threading.Lock()


def _get_monitoring_circuit_breaker() -> MonitoringCircuitBreaker:
    """Get or create the global monitoring circuit breaker."""
    global _monitoring_circuit_breaker
    with _circuit_breaker_lock:
        if _monitoring_circuit_breaker is None:
            _monitoring_circuit_breaker = MonitoringCircuitBreaker()
        return _monitoring_circuit_breaker


def get_monitoring_circuit_breaker_status() -> dict[str, Any]:
    """Get the current monitoring circuit breaker status.

    Returns:
        dict with state, failure_count, success_count, etc.
    """
    return _get_monitoring_circuit_breaker().get_status()


# Global instances for trend monitor and anomaly detector
_trend_monitor: TrendMonitor | None = None
_anomaly_detector: AnomalyDetector | None = None
_trend_monitor_lock = threading.Lock()
_anomaly_detector_lock = threading.Lock()


def get_trend_monitor() -> TrendMonitor:
    """Get or create the global trend monitor instance."""
    global _trend_monitor
    with _trend_monitor_lock:
        if _trend_monitor is None:
            _trend_monitor = TrendMonitor()
        return _trend_monitor


def set_trend_monitor(monitor: TrendMonitor) -> None:
    """Set the global trend monitor instance."""
    global _trend_monitor
    with _trend_monitor_lock:
        _trend_monitor = monitor


def get_anomaly_detector() -> AnomalyDetector:
    """Get or create the global anomaly detector instance."""
    global _anomaly_detector
    with _anomaly_detector_lock:
        if _anomaly_detector is None:
            _anomaly_detector = AnomalyDetector()
        return _anomaly_detector


def set_anomaly_detector(detector: AnomalyDetector) -> None:
    """Set the global anomaly detector instance."""
    global _anomaly_detector
    with _anomaly_detector_lock:
        _anomaly_detector = detector


def _clear_monitoring_components() -> None:
    """Clear cached monitoring components (for testing)."""
    global _monitoring_circuit_breaker, _trend_monitor, _anomaly_detector
    with _circuit_breaker_lock:
        if _monitoring_circuit_breaker is not None:
            _monitoring_circuit_breaker.reset()
            _monitoring_circuit_breaker = None
    with _trend_monitor_lock:
        _trend_monitor = None
    with _anomaly_detector_lock:
        _anomaly_detector = None


class MonitoringHandler:
    """HTTP handlers for monitoring operations.

    Provides endpoints for:
    - Recording metrics for trend and anomaly detection
    - Retrieving trend data for specific or all metrics
    - Fetching recent anomalies
    - Getting baseline statistics
    - Circuit breaker status

    Features:
    - Circuit breaker pattern for resilient monitoring system access
    - Rate limiting (30-60 requests/minute depending on endpoint)
    - RBAC permission checks (autonomous:read, autonomous:write)
    - Comprehensive input validation
    """

    ROUTES = [
        "/api/v1/autonomous/monitoring/record",
        "/api/v1/autonomous/monitoring/trends",
        "/api/v1/autonomous/monitoring/trends/*",
        "/api/v1/autonomous/monitoring/anomalies",
        "/api/v1/autonomous/monitoring/baseline",
        "/api/v1/autonomous/monitoring/baseline/*",
        "/api/v1/autonomous/monitoring/circuit-breaker",
    ]

    def __init__(self, ctx: dict | None = None):
        """Initialize handler with optional context."""
        self.ctx = ctx or {}

    @staticmethod
    def _check_circuit_breaker() -> web.Response | None:
        """Check if circuit breaker allows the request.

        Returns:
            Error response if circuit is open, None if allowed.
        """
        circuit_breaker = _get_monitoring_circuit_breaker()
        if not circuit_breaker.is_allowed():
            logger.warning("Monitoring circuit breaker is open, rejecting request")
            return web.json_response(
                {
                    "success": False,
                    "error": "Monitoring service temporarily unavailable (circuit_breaker open)",
                    "circuit_breaker_state": circuit_breaker.state,
                    "retry_after_seconds": circuit_breaker.cooldown_seconds,
                },
                status=503,
            )
        return None

    @staticmethod
    @rate_limit(requests_per_minute=60, limiter_name="monitoring.record")
    @require_permission("autonomous:monitoring:write")
    async def record_metric(request: web.Request) -> web.Response:
        """
        Record a metric value for trend and anomaly detection.

        POST /api/autonomous/monitoring/record

        Requires authentication and 'autonomous:write' permission.
        Rate limited to 60 requests per minute.

        Body:
            metric_name: str - Name of the metric (alphanumeric, underscores, hyphens, dots)
            value: float - Current value (must be a finite number)

        Returns:
            Trend data and anomaly if detected
        """
        circuit_breaker = _get_monitoring_circuit_breaker()

        try:
            # Check circuit breaker
            cb_error = MonitoringHandler._check_circuit_breaker()
            if cb_error:
                return cb_error

            # RBAC check
            auth_ctx = await get_auth_context(request, require_auth=True)
            checker = get_permission_checker()
            decision = checker.check_permission(auth_ctx, AUTONOMOUS_WRITE_PERMISSION)
            if not decision.allowed:
                raise ForbiddenError("Permission denied")

            data, err = await parse_json_body(request, context="record_metric")
            if err:
                return err

            # Validate metric_name
            metric_name = data.get("metric_name")
            is_valid, error_msg = validate_metric_name(metric_name)
            if not is_valid:
                return web.json_response(
                    {"success": False, "error": error_msg},
                    status=400,
                )

            # Validate value
            raw_value = data.get("value")
            is_valid, value, error_msg = validate_metric_value(raw_value)
            if not is_valid:
                return web.json_response(
                    {"success": False, "error": error_msg},
                    status=400,
                )

            # Record in trend monitor
            trend_monitor = get_trend_monitor()
            trend_monitor.record(metric_name, value)

            # Record in anomaly detector
            anomaly_detector = get_anomaly_detector()
            anomaly = anomaly_detector.record(metric_name, value)

            response: dict[str, Any] = {
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

            circuit_breaker.record_success()
            return web.json_response(response)

        except UnauthorizedError as e:
            logger.warning("Unauthorized recording metric: %s", e)
            return web.json_response({"success": False, "error": "Authentication required"}, status=401)
        except ForbiddenError as e:
            logger.warning("Forbidden recording metric: %s", e)
            return web.json_response({"success": False, "error": "Permission denied"}, status=403)
        except (KeyError, ValueError, TypeError, AttributeError, RuntimeError) as e:
            circuit_breaker.record_failure()
            logger.error("Error recording metric: %s", e)
            return web.json_response(
                {"success": False, "error": "Internal server error"},
                status=500,
            )

    @staticmethod
    @rate_limit(requests_per_minute=60, limiter_name="monitoring.get_trend")
    @require_permission("autonomous:monitoring:read")
    async def get_trend(request: web.Request) -> web.Response:
        """
        Get trend for a specific metric.

        GET /api/autonomous/monitoring/trends/{metric_name}

        Requires authentication and 'autonomous:read' permission.
        Rate limited to 60 requests per minute.

        Path params:
            metric_name: str - Name of the metric to analyze

        Query params:
            period_seconds: int (optional) - Time period to analyze (0-86400)

        Returns:
            Trend data
        """
        circuit_breaker = _get_monitoring_circuit_breaker()

        try:
            # Check circuit breaker
            cb_error = MonitoringHandler._check_circuit_breaker()
            if cb_error:
                return cb_error

            # RBAC check
            auth_ctx = await get_auth_context(request, require_auth=True)
            checker = get_permission_checker()
            decision = checker.check_permission(auth_ctx, AUTONOMOUS_READ_PERMISSION)
            if not decision.allowed:
                raise ForbiddenError("Permission denied")

            # Validate metric_name
            metric_name = request.match_info.get("metric_name")
            is_valid, error_msg = validate_metric_name(metric_name)
            if not is_valid:
                return web.json_response(
                    {"success": False, "error": error_msg},
                    status=400,
                )

            trend_monitor = get_trend_monitor()
            trend = trend_monitor.get_trend(
                metric_name,
                period_seconds=safe_query_int(
                    request.query, "period_seconds", default=0, min_val=0, max_val=86400
                )
                or None,
            )

            if not trend:
                circuit_breaker.record_success()
                return web.json_response(
                    {
                        "success": True,
                        "trend": None,
                        "message": "Insufficient data for trend analysis",
                    }
                )

            circuit_breaker.record_success()
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
            logger.warning("Unauthorized getting trend: %s", e)
            return web.json_response({"success": False, "error": "Authentication required"}, status=401)
        except ForbiddenError as e:
            logger.warning("Forbidden getting trend: %s", e)
            return web.json_response({"success": False, "error": "Permission denied"}, status=403)
        except (KeyError, ValueError, TypeError, AttributeError, RuntimeError) as e:
            circuit_breaker.record_failure()
            logger.error("Error getting trend: %s", e)
            return web.json_response(
                {"success": False, "error": "Internal server error"},
                status=500,
            )

    @staticmethod
    @rate_limit(requests_per_minute=30, limiter_name="monitoring.get_all_trends")
    @require_permission("autonomous:monitoring:read")
    async def get_all_trends(request: web.Request) -> web.Response:
        """
        Get trends for all monitored metrics.

        GET /api/autonomous/monitoring/trends

        Requires authentication and 'autonomous:read' permission.
        Rate limited to 30 requests per minute (heavier operation).

        Returns:
            Dict of metric_name -> trend data
        """
        circuit_breaker = _get_monitoring_circuit_breaker()

        try:
            # Check circuit breaker
            cb_error = MonitoringHandler._check_circuit_breaker()
            if cb_error:
                return cb_error

            # RBAC check
            auth_ctx = await get_auth_context(request, require_auth=True)
            checker = get_permission_checker()
            decision = checker.check_permission(auth_ctx, AUTONOMOUS_READ_PERMISSION)
            if not decision.allowed:
                raise ForbiddenError("Permission denied")

            trend_monitor = get_trend_monitor()
            trends = trend_monitor.get_all_trends()

            circuit_breaker.record_success()
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
            logger.warning("Unauthorized getting trends: %s", e)
            return web.json_response({"success": False, "error": "Authentication required"}, status=401)
        except ForbiddenError as e:
            logger.warning("Forbidden getting trends: %s", e)
            return web.json_response({"success": False, "error": "Permission denied"}, status=403)
        except (KeyError, ValueError, TypeError, AttributeError, RuntimeError) as e:
            circuit_breaker.record_failure()
            logger.error("Error getting trends: %s", e)
            return web.json_response(
                {"success": False, "error": "Internal server error"},
                status=500,
            )

    @staticmethod
    @rate_limit(requests_per_minute=60, limiter_name="monitoring.get_anomalies")
    @require_permission("autonomous:monitoring:read")
    async def get_anomalies(request: web.Request) -> web.Response:
        """
        Get recent anomalies.

        GET /api/autonomous/monitoring/anomalies

        Requires authentication and 'autonomous:read' permission.
        Rate limited to 60 requests per minute.

        Query params:
            hours: int (optional) - Hours to look back (1-720, default: 24)
            metric_name: str (optional) - Filter by metric name

        Returns:
            List of recent anomalies
        """
        circuit_breaker = _get_monitoring_circuit_breaker()

        try:
            # Check circuit breaker
            cb_error = MonitoringHandler._check_circuit_breaker()
            if cb_error:
                return cb_error

            # RBAC check
            auth_ctx = await get_auth_context(request, require_auth=True)
            checker = get_permission_checker()
            decision = checker.check_permission(auth_ctx, AUTONOMOUS_READ_PERMISSION)
            if not decision.allowed:
                raise ForbiddenError("Permission denied")

            hours = safe_query_int(request.query, "hours", default=24, min_val=1, max_val=720)
            metric_name = None
            try:
                metric_name = request.query.get("metric_name")
            except (AttributeError, TypeError):
                metric_name = None
            if not isinstance(metric_name, str):
                metric_name = None

            # Validate metric_name if provided
            if metric_name:
                is_valid, error_msg = validate_metric_name(metric_name)
                if not is_valid:
                    return web.json_response(
                        {"success": False, "error": error_msg},
                        status=400,
                    )

            anomaly_detector = get_anomaly_detector()
            anomalies = anomaly_detector.get_recent_anomalies(
                hours=hours,
                metric_name=metric_name,
            )

            circuit_breaker.record_success()
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
            logger.warning("Unauthorized getting anomalies: %s", e)
            return web.json_response({"success": False, "error": "Authentication required"}, status=401)
        except ForbiddenError as e:
            logger.warning("Forbidden getting anomalies: %s", e)
            return web.json_response({"success": False, "error": "Permission denied"}, status=403)
        except (KeyError, ValueError, TypeError, AttributeError, RuntimeError) as e:
            circuit_breaker.record_failure()
            logger.error("Error getting anomalies: %s", e)
            return web.json_response(
                {"success": False, "error": "Internal server error"},
                status=500,
            )

    @staticmethod
    @rate_limit(requests_per_minute=60, limiter_name="monitoring.get_baseline")
    @require_permission("autonomous:monitoring:read")
    async def get_baseline_stats(request: web.Request) -> web.Response:
        """
        Get baseline statistics for a metric.

        GET /api/autonomous/monitoring/baseline/{metric_name}

        Requires authentication and 'autonomous:read' permission.
        Rate limited to 60 requests per minute.

        Path params:
            metric_name: str - Name of the metric

        Returns:
            Baseline statistics (mean, stdev, min, max, median)
        """
        circuit_breaker = _get_monitoring_circuit_breaker()

        try:
            # Check circuit breaker
            cb_error = MonitoringHandler._check_circuit_breaker()
            if cb_error:
                return cb_error

            # RBAC check
            auth_ctx = await get_auth_context(request, require_auth=True)
            checker = get_permission_checker()
            decision = checker.check_permission(auth_ctx, AUTONOMOUS_READ_PERMISSION)
            if not decision.allowed:
                raise ForbiddenError("Permission denied")

            # Validate metric_name
            metric_name = request.match_info.get("metric_name")
            is_valid, error_msg = validate_metric_name(metric_name)
            if not is_valid:
                return web.json_response(
                    {"success": False, "error": error_msg},
                    status=400,
                )

            anomaly_detector = get_anomaly_detector()
            stats = anomaly_detector.get_baseline_stats(metric_name)

            if not stats:
                circuit_breaker.record_success()
                return web.json_response(
                    {
                        "success": True,
                        "stats": None,
                        "message": "Insufficient data for baseline statistics",
                    }
                )

            circuit_breaker.record_success()
            return web.json_response(
                {
                    "success": True,
                    "metric_name": metric_name,
                    "stats": stats,
                }
            )

        except UnauthorizedError as e:
            logger.warning("Unauthorized getting baseline stats: %s", e)
            return web.json_response({"success": False, "error": "Authentication required"}, status=401)
        except ForbiddenError as e:
            logger.warning("Forbidden getting baseline stats: %s", e)
            return web.json_response({"success": False, "error": "Permission denied"}, status=403)
        except (KeyError, ValueError, TypeError, AttributeError, RuntimeError) as e:
            circuit_breaker.record_failure()
            logger.error("Error getting baseline stats: %s", e)
            return web.json_response(
                {"success": False, "error": "Internal server error"},
                status=500,
            )

    @staticmethod
    @rate_limit(requests_per_minute=60, limiter_name="monitoring.circuit_breaker")
    @require_permission("autonomous:monitoring:read")
    async def get_circuit_breaker_status(request: web.Request) -> web.Response:
        """
        Get the current circuit breaker status.

        GET /api/autonomous/monitoring/circuit-breaker

        Requires authentication and 'autonomous:read' permission.
        Rate limited to 60 requests per minute.

        Returns:
            Circuit breaker status (state, failure_count, etc.)
        """
        try:
            # RBAC check
            auth_ctx = await get_auth_context(request, require_auth=True)
            checker = get_permission_checker()
            decision = checker.check_permission(auth_ctx, AUTONOMOUS_READ_PERMISSION)
            if not decision.allowed:
                raise ForbiddenError("Permission denied")

            status = get_monitoring_circuit_breaker_status()
            return web.json_response(
                {
                    "success": True,
                    "circuit_breaker": status,
                }
            )

        except UnauthorizedError as e:
            logger.warning("Unauthorized getting circuit breaker status: %s", e)
            return web.json_response({"success": False, "error": "Authentication required"}, status=401)
        except ForbiddenError as e:
            logger.warning("Forbidden getting circuit breaker status: %s", e)
            return web.json_response({"success": False, "error": "Permission denied"}, status=403)
        except (KeyError, ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.error("Error getting circuit breaker status: %s", e)
            return web.json_response(
                {"success": False, "error": "Internal server error"},
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
        app.router.add_get(
            f"{prefix}/monitoring/circuit-breaker",
            MonitoringHandler.get_circuit_breaker_status,
        )
