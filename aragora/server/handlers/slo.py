"""
SLO (Service Level Objective) endpoint handlers.

Exposes SLO status, error budgets, and violation data via HTTP API.

Endpoints:
- GET /api/slos/status - Overall SLO compliance status
- GET /api/slos/{slo_name} - Individual SLO details
- GET /api/slos/error-budget - Error budget timeline
- GET /api/slos/violations - Recent SLO violations
- GET /api/v1/slos/status - Versioned endpoint

Usage:
    # Check overall SLO health
    curl http://localhost:8080/api/slos/status

    # Get specific SLO details
    curl http://localhost:8080/api/slos/availability

    # Get error budget information
    curl http://localhost:8080/api/slos/error-budget
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from aragora.observability.slo import (
    check_alerts,
    check_availability_slo,
    check_debate_success_slo,
    check_latency_slo,
    get_slo_status,
    get_slo_status_json,
    get_slo_targets,
)
from aragora.server.versioning.compat import strip_version_prefix

from .base import BaseHandler, HandlerResult, error_response, json_response
from .utils.rate_limit import RateLimiter, get_client_ip

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Rate limiter for SLO endpoints (30 requests per minute)
_slo_limiter = RateLimiter(requests_per_minute=30)


class SLOHandler(BaseHandler):
    """Handler for SLO monitoring endpoints."""

    ROUTES = [
        "/api/slos/status",
        "/api/slos/error-budget",
        "/api/slos/violations",
        "/api/slos/targets",
        "/api/slos/availability",
        "/api/slos/latency",
        "/api/slos/debate-success",
    ]

    # Individual SLO names for dynamic routing
    SLO_NAMES = {"availability", "latency", "latency_p99", "debate_success", "debate-success"}

    def can_handle(self, path: str) -> bool:
        """Check if this handler can process the given path."""
        path = strip_version_prefix(path)

        if path in self.ROUTES:
            return True

        # Handle dynamic /api/slos/{slo_name} routes
        if path.startswith("/api/slos/"):
            slo_name = path.split("/")[-1]
            return slo_name in self.SLO_NAMES

        return False

    def handle(self, path: str, query_params: dict, handler: Any) -> Optional[HandlerResult]:
        """Route SLO requests to appropriate methods."""
        path = strip_version_prefix(path)

        # Rate limit check
        client_ip = get_client_ip(handler)
        if not _slo_limiter.is_allowed(client_ip):
            logger.warning(f"Rate limit exceeded for SLO endpoint: {client_ip}")
            return error_response("Rate limit exceeded. Please try again later.", 429)

        try:
            if path == "/api/slos/status":
                return self._handle_slo_status()
            elif path == "/api/slos/error-budget":
                return self._handle_error_budget()
            elif path == "/api/slos/violations":
                return self._handle_violations()
            elif path == "/api/slos/targets":
                return self._handle_targets()
            elif path == "/api/slos/availability":
                return self._handle_slo_detail("availability")
            elif path == "/api/slos/latency":
                return self._handle_slo_detail("latency_p99")
            elif path == "/api/slos/debate-success":
                return self._handle_slo_detail("debate_success")
            elif path.startswith("/api/slos/"):
                slo_name = path.split("/")[-1]
                # Normalize name
                if slo_name == "latency":
                    slo_name = "latency_p99"
                elif slo_name == "debate-success":
                    slo_name = "debate_success"
                return self._handle_slo_detail(slo_name)
            else:
                return error_response(f"Unknown SLO endpoint: {path}", 404)

        except Exception as e:
            logger.exception(f"Error handling SLO request: {e}")
            return error_response(f"Internal error: {str(e)}", 500)

    def _handle_slo_status(self) -> HandlerResult:
        """GET /api/slos/status - Overall SLO compliance status."""
        try:
            status_json = get_slo_status_json()
            return json_response(status_json)
        except Exception as e:
            logger.exception(f"Failed to get SLO status: {e}")
            return error_response(f"Failed to get SLO status: {str(e)}", 500)

    def _handle_slo_detail(self, slo_name: str) -> HandlerResult:
        """GET /api/slos/{slo_name} - Individual SLO details."""
        try:
            if slo_name == "availability":
                result = check_availability_slo()
            elif slo_name == "latency_p99":
                result = check_latency_slo()
            elif slo_name == "debate_success":
                result = check_debate_success_slo()
            else:
                return error_response(f"Unknown SLO: {slo_name}", 404)

            return json_response(
                {
                    "name": result.name,
                    "target": result.target,
                    "current": result.current,
                    "compliant": result.compliant,
                    "compliance_percentage": result.compliance_percentage,
                    "error_budget_remaining": result.error_budget_remaining,
                    "burn_rate": result.burn_rate,
                    "window": {
                        "start": result.window_start.isoformat(),
                        "end": result.window_end.isoformat(),
                    },
                }
            )
        except Exception as e:
            logger.exception(f"Failed to get SLO detail for {slo_name}: {e}")
            return error_response(f"Failed to get SLO detail: {str(e)}", 500)

    def _handle_error_budget(self) -> HandlerResult:
        """GET /api/slos/error-budget - Error budget timeline."""
        try:
            status = get_slo_status()

            # Build error budget response
            budgets = []
            for name, result in [
                ("availability", status.availability),
                ("latency_p99", status.latency_p99),
                ("debate_success", status.debate_success),
            ]:
                budgets.append(
                    {
                        "slo_name": result.name,
                        "slo_id": name,
                        "target": result.target,
                        "error_budget_total": 100.0,  # Percentage
                        "error_budget_remaining": result.error_budget_remaining,
                        "error_budget_consumed": 100.0 - result.error_budget_remaining,
                        "burn_rate": result.burn_rate,
                        "projected_exhaustion": self._calculate_exhaustion_time(result),
                        "window": {
                            "start": result.window_start.isoformat(),
                            "end": result.window_end.isoformat(),
                        },
                    }
                )

            return json_response(
                {
                    "timestamp": status.timestamp.isoformat(),
                    "budgets": budgets,
                }
            )
        except Exception as e:
            logger.exception(f"Failed to get error budget: {e}")
            return error_response(f"Failed to get error budget: {str(e)}", 500)

    def _handle_violations(self) -> HandlerResult:
        """GET /api/slos/violations - Recent SLO violations."""
        try:
            status = get_slo_status()
            alerts = check_alerts(status)

            violations = []
            for alert, result in alerts:
                violations.append(
                    {
                        "slo_name": alert.slo_name,
                        "severity": alert.severity,
                        "message": alert.message,
                        "current_value": result.current,
                        "target_value": result.target,
                        "error_budget_remaining": result.error_budget_remaining,
                        "burn_rate": result.burn_rate,
                        "detected_at": result.window_end.isoformat(),
                    }
                )

            return json_response(
                {
                    "timestamp": status.timestamp.isoformat(),
                    "violation_count": len(violations),
                    "violations": violations,
                    "overall_healthy": status.overall_healthy,
                }
            )
        except Exception as e:
            logger.exception(f"Failed to get violations: {e}")
            return error_response(f"Failed to get violations: {str(e)}", 500)

    def _handle_targets(self) -> HandlerResult:
        """GET /api/slos/targets - Get configured SLO targets."""
        try:
            targets = get_slo_targets()

            targets_response = []
            for key, target in targets.items():
                targets_response.append(
                    {
                        "id": key,
                        "name": target.name,
                        "target": target.target,
                        "unit": target.unit,
                        "description": target.description,
                        "comparison": target.comparison,
                    }
                )

            return json_response(
                {
                    "targets": targets_response,
                }
            )
        except Exception as e:
            logger.exception(f"Failed to get targets: {e}")
            return error_response(f"Failed to get targets: {str(e)}", 500)

    def _calculate_exhaustion_time(self, result: Any) -> Optional[str]:
        """Calculate projected error budget exhaustion time.

        Args:
            result: SLOResult with burn_rate and error_budget_remaining

        Returns:
            ISO timestamp of projected exhaustion, or None if not applicable
        """
        from datetime import timedelta

        if result.burn_rate <= 0 or result.error_budget_remaining <= 0:
            return None

        # Calculate hours until exhaustion at current burn rate
        # Error budget window is 1 hour, so burn_rate of 1.0 = consuming at expected rate
        # If burn_rate > 1.0, exhaustion happens faster
        if result.burn_rate <= 1.0:
            return None  # Budget is sustainable

        # Hours remaining = (remaining / 100) / (burn_rate * expected_hourly_consumption)
        # Simplified: hours = remaining_pct / (burn_rate * (100 - target) * 100)
        hours_remaining = result.error_budget_remaining / (result.burn_rate * 10)
        hours_remaining = max(0, min(hours_remaining, 720))  # Cap at 30 days

        exhaustion_time = result.window_end + timedelta(hours=hours_remaining)
        return exhaustion_time.isoformat()


__all__ = ["SLOHandler"]
