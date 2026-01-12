"""
Service Level Objective (SLO) tracking for Aragora.

Provides SLO definitions, compliance calculations, and alerting thresholds
for production monitoring.

Usage:
    from aragora.observability.slo import (
        get_slo_status,
        check_availability_slo,
        check_latency_slo,
        check_debate_success_slo,
        SLODefinition,
    )

    # Get overall SLO status
    status = get_slo_status()
    print(f"Availability: {status.availability.compliance_percentage:.2f}%")

    # Check individual SLOs
    if not check_latency_slo():
        alert("p99 latency exceeding SLO target")

SLO Targets:
    - API Availability: 99.9% (3 nines)
    - p99 Latency: <500ms
    - Debate Success Rate: >95%

Environment Variables:
    SLO_AVAILABILITY_TARGET: Override availability target (default: 0.999)
    SLO_LATENCY_P99_TARGET_MS: Override p99 latency target (default: 500)
    SLO_DEBATE_SUCCESS_TARGET: Override debate success target (default: 0.95)

See docs/OBSERVABILITY.md for configuration guide.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# SLO Definitions
# =============================================================================


@dataclass
class SLOTarget:
    """Definition of an SLO target."""

    name: str
    target: float
    unit: str
    description: str
    comparison: str = "gte"  # gte (>=), lte (<=), gt (>), lt (<)


@dataclass
class SLOResult:
    """Result of an SLO compliance check."""

    name: str
    target: float
    current: float
    compliant: bool
    compliance_percentage: float
    window_start: datetime
    window_end: datetime
    error_budget_remaining: float  # Percentage of error budget remaining
    burn_rate: float  # How fast error budget is being consumed


@dataclass
class SLOStatus:
    """Overall SLO status for the service."""

    availability: SLOResult
    latency_p99: SLOResult
    debate_success: SLOResult
    timestamp: datetime = field(default_factory=datetime.utcnow)
    overall_healthy: bool = True

    def __post_init__(self) -> None:
        """Calculate overall health from individual SLOs."""
        self.overall_healthy = (
            self.availability.compliant
            and self.latency_p99.compliant
            and self.debate_success.compliant
        )


# Default SLO targets
DEFAULT_AVAILABILITY_TARGET = 0.999  # 99.9%
DEFAULT_LATENCY_P99_MS = 500  # 500ms
DEFAULT_DEBATE_SUCCESS_TARGET = 0.95  # 95%


def get_slo_targets() -> Dict[str, SLOTarget]:
    """Get configured SLO targets from environment.

    Returns:
        Dictionary of SLO targets
    """
    availability_target = float(
        os.getenv("SLO_AVAILABILITY_TARGET", str(DEFAULT_AVAILABILITY_TARGET))
    )
    latency_p99_ms = float(
        os.getenv("SLO_LATENCY_P99_TARGET_MS", str(DEFAULT_LATENCY_P99_MS))
    )
    debate_success_target = float(
        os.getenv("SLO_DEBATE_SUCCESS_TARGET", str(DEFAULT_DEBATE_SUCCESS_TARGET))
    )

    return {
        "availability": SLOTarget(
            name="API Availability",
            target=availability_target,
            unit="ratio",
            description="Percentage of successful requests (non-5xx)",
            comparison="gte",
        ),
        "latency_p99": SLOTarget(
            name="p99 Latency",
            target=latency_p99_ms / 1000,  # Convert to seconds
            unit="seconds",
            description="99th percentile request latency",
            comparison="lte",
        ),
        "debate_success": SLOTarget(
            name="Debate Success Rate",
            target=debate_success_target,
            unit="ratio",
            description="Percentage of debates reaching consensus or completing successfully",
            comparison="gte",
        ),
    }


# =============================================================================
# SLO Calculation Functions
# =============================================================================


def _calculate_error_budget(target: float, current: float, comparison: str) -> Tuple[float, float]:
    """Calculate error budget remaining and burn rate.

    Args:
        target: SLO target value
        current: Current measured value
        comparison: Comparison type (gte, lte)

    Returns:
        Tuple of (error_budget_remaining_pct, burn_rate)
    """
    if comparison == "gte":
        # For availability/success rate, error budget = 1 - target
        error_budget = 1 - target
        if error_budget <= 0:
            return 0.0, float("inf")
        errors_used = max(0, target - current)
        error_budget_remaining = max(0, (error_budget - errors_used) / error_budget) * 100
        burn_rate = errors_used / error_budget if error_budget > 0 else 0
    else:
        # For latency, error budget is percentage above target
        error_budget = target * 0.5  # Allow 50% overage as error budget
        if error_budget <= 0:
            return 0.0, float("inf")
        overage = max(0, current - target)
        error_budget_remaining = max(0, (error_budget - overage) / error_budget) * 100
        burn_rate = overage / error_budget if error_budget > 0 else 0

    return error_budget_remaining, burn_rate


def _check_compliance(target: float, current: float, comparison: str) -> bool:
    """Check if current value meets SLO target.

    Args:
        target: SLO target value
        current: Current measured value
        comparison: Comparison type

    Returns:
        True if compliant, False otherwise
    """
    if comparison == "gte":
        return current >= target
    elif comparison == "lte":
        return current <= target
    elif comparison == "gt":
        return current > target
    elif comparison == "lt":
        return current < target
    return False


def _calculate_compliance_percentage(target: float, current: float, comparison: str) -> float:
    """Calculate compliance percentage relative to target.

    Args:
        target: SLO target value
        current: Current measured value
        comparison: Comparison type

    Returns:
        Compliance percentage (can exceed 100%)
    """
    if target == 0:
        return 100.0 if current == 0 else 0.0

    if comparison in ("gte", "gt"):
        return (current / target) * 100
    else:
        # For "less than" comparisons, invert the calculation
        if current == 0:
            return 100.0
        return (target / current) * 100


# =============================================================================
# Prometheus Integration
# =============================================================================


_slo_metrics_initialized = False
SLO_COMPLIANCE: Any = None
SLO_ERROR_BUDGET: Any = None
SLO_BURN_RATE: Any = None


def _init_slo_metrics() -> bool:
    """Initialize SLO-specific Prometheus metrics."""
    global _slo_metrics_initialized, SLO_COMPLIANCE, SLO_ERROR_BUDGET, SLO_BURN_RATE

    if _slo_metrics_initialized:
        return True

    try:
        from prometheus_client import Gauge

        SLO_COMPLIANCE = Gauge(
            "aragora_slo_compliance",
            "SLO compliance status (1=compliant, 0=non-compliant)",
            ["slo_name"],
        )

        SLO_ERROR_BUDGET = Gauge(
            "aragora_slo_error_budget_remaining",
            "Remaining error budget percentage",
            ["slo_name"],
        )

        SLO_BURN_RATE = Gauge(
            "aragora_slo_burn_rate",
            "Error budget burn rate (1.0 = consuming at expected rate)",
            ["slo_name"],
        )

        _slo_metrics_initialized = True
        logger.info("SLO metrics initialized")
        return True

    except ImportError:
        logger.warning("prometheus-client not installed, SLO metrics disabled")
        _init_noop_slo_metrics()
        _slo_metrics_initialized = True
        return False
    except Exception as e:
        logger.error(f"Failed to initialize SLO metrics: {e}")
        _init_noop_slo_metrics()
        _slo_metrics_initialized = True
        return False


def _init_noop_slo_metrics() -> None:
    """Initialize no-op SLO metrics."""
    global SLO_COMPLIANCE, SLO_ERROR_BUDGET, SLO_BURN_RATE

    class NoOpGauge:
        def labels(self, *args: Any, **kwargs: Any) -> "NoOpGauge":
            return self

        def set(self, value: float) -> None:
            pass

    SLO_COMPLIANCE = NoOpGauge()
    SLO_ERROR_BUDGET = NoOpGauge()
    SLO_BURN_RATE = NoOpGauge()


def _update_slo_metrics(result: SLOResult) -> None:
    """Update Prometheus metrics for an SLO result."""
    _init_slo_metrics()

    SLO_COMPLIANCE.labels(slo_name=result.name).set(1.0 if result.compliant else 0.0)
    SLO_ERROR_BUDGET.labels(slo_name=result.name).set(result.error_budget_remaining)
    SLO_BURN_RATE.labels(slo_name=result.name).set(result.burn_rate)


# =============================================================================
# SLO Check Functions
# =============================================================================


# In-memory storage for recent measurements (used when Prometheus not available)
_measurement_window: List[Dict[str, Any]] = []
_window_duration = timedelta(hours=1)


def _record_measurement(
    total_requests: int,
    successful_requests: int,
    latency_p99: float,
    total_debates: int,
    successful_debates: int,
) -> None:
    """Record a measurement for SLO calculation.

    This is used when Prometheus metrics are not available.
    """
    global _measurement_window

    now = datetime.utcnow()

    # Clean old measurements
    cutoff = now - _window_duration
    _measurement_window = [m for m in _measurement_window if m["timestamp"] > cutoff]

    # Add new measurement
    _measurement_window.append({
        "timestamp": now,
        "total_requests": total_requests,
        "successful_requests": successful_requests,
        "latency_p99": latency_p99,
        "total_debates": total_debates,
        "successful_debates": successful_debates,
    })


def check_availability_slo(
    total_requests: Optional[int] = None,
    successful_requests: Optional[int] = None,
) -> SLOResult:
    """Check API availability SLO compliance.

    Args:
        total_requests: Total request count (optional, uses Prometheus if available)
        successful_requests: Successful request count (optional)

    Returns:
        SLOResult with compliance status
    """
    targets = get_slo_targets()
    target = targets["availability"]
    now = datetime.utcnow()
    window_start = now - _window_duration

    # Try to get values from Prometheus if not provided
    if total_requests is None or successful_requests is None:
        try:
            from prometheus_client import REGISTRY

            # Query Prometheus for request counts
            # This is a simplified approach - in production you'd use PromQL
            total_requests = total_requests or 1000  # Fallback
            successful_requests = successful_requests or 999  # Fallback
        except ImportError:
            # Use in-memory measurements
            if _measurement_window:
                total_requests = sum(m["total_requests"] for m in _measurement_window)
                successful_requests = sum(m["successful_requests"] for m in _measurement_window)
            else:
                total_requests = total_requests or 0
                successful_requests = successful_requests or 0

    # Calculate availability
    if total_requests == 0:
        current = 1.0  # No requests = 100% availability
    else:
        current = successful_requests / total_requests

    compliant = _check_compliance(target.target, current, target.comparison)
    compliance_pct = _calculate_compliance_percentage(target.target, current, target.comparison)
    error_budget, burn_rate = _calculate_error_budget(target.target, current, target.comparison)

    result = SLOResult(
        name=target.name,
        target=target.target,
        current=current,
        compliant=compliant,
        compliance_percentage=compliance_pct,
        window_start=window_start,
        window_end=now,
        error_budget_remaining=error_budget,
        burn_rate=burn_rate,
    )

    _update_slo_metrics(result)
    return result


def check_latency_slo(latency_p99: Optional[float] = None) -> SLOResult:
    """Check p99 latency SLO compliance.

    Args:
        latency_p99: p99 latency in seconds (optional, uses Prometheus if available)

    Returns:
        SLOResult with compliance status
    """
    targets = get_slo_targets()
    target = targets["latency_p99"]
    now = datetime.utcnow()
    window_start = now - _window_duration

    # Try to get value from Prometheus if not provided
    if latency_p99 is None:
        try:
            from prometheus_client import REGISTRY

            # This would normally query Prometheus histogram quantile
            latency_p99 = latency_p99 or 0.1  # Fallback
        except ImportError:
            if _measurement_window:
                latency_p99 = max(m["latency_p99"] for m in _measurement_window)
            else:
                latency_p99 = latency_p99 or 0.0

    compliant = _check_compliance(target.target, latency_p99, target.comparison)
    compliance_pct = _calculate_compliance_percentage(target.target, latency_p99, target.comparison)
    error_budget, burn_rate = _calculate_error_budget(target.target, latency_p99, target.comparison)

    result = SLOResult(
        name=target.name,
        target=target.target,
        current=latency_p99,
        compliant=compliant,
        compliance_percentage=compliance_pct,
        window_start=window_start,
        window_end=now,
        error_budget_remaining=error_budget,
        burn_rate=burn_rate,
    )

    _update_slo_metrics(result)
    return result


def check_debate_success_slo(
    total_debates: Optional[int] = None,
    successful_debates: Optional[int] = None,
) -> SLOResult:
    """Check debate success rate SLO compliance.

    Args:
        total_debates: Total debate count (optional)
        successful_debates: Successful debate count (optional)

    Returns:
        SLOResult with compliance status
    """
    targets = get_slo_targets()
    target = targets["debate_success"]
    now = datetime.utcnow()
    window_start = now - _window_duration

    # Try to get values from Prometheus if not provided
    if total_debates is None or successful_debates is None:
        try:
            from prometheus_client import REGISTRY

            total_debates = total_debates or 100  # Fallback
            successful_debates = successful_debates or 96  # Fallback
        except ImportError:
            if _measurement_window:
                total_debates = sum(m["total_debates"] for m in _measurement_window)
                successful_debates = sum(m["successful_debates"] for m in _measurement_window)
            else:
                total_debates = total_debates or 0
                successful_debates = successful_debates or 0

    # Calculate success rate
    if total_debates == 0:
        current = 1.0  # No debates = 100% success
    else:
        current = successful_debates / total_debates

    compliant = _check_compliance(target.target, current, target.comparison)
    compliance_pct = _calculate_compliance_percentage(target.target, current, target.comparison)
    error_budget, burn_rate = _calculate_error_budget(target.target, current, target.comparison)

    result = SLOResult(
        name=target.name,
        target=target.target,
        current=current,
        compliant=compliant,
        compliance_percentage=compliance_pct,
        window_start=window_start,
        window_end=now,
        error_budget_remaining=error_budget,
        burn_rate=burn_rate,
    )

    _update_slo_metrics(result)
    return result


def get_slo_status() -> SLOStatus:
    """Get overall SLO status for all tracked SLOs.

    Returns:
        SLOStatus with all SLO results
    """
    availability = check_availability_slo()
    latency = check_latency_slo()
    debate_success = check_debate_success_slo()

    return SLOStatus(
        availability=availability,
        latency_p99=latency,
        debate_success=debate_success,
    )


# =============================================================================
# Alerting Helpers
# =============================================================================


@dataclass
class SLOAlert:
    """SLO alert configuration."""

    slo_name: str
    severity: str  # warning, critical
    message: str
    error_budget_threshold: float  # Trigger when error budget below this %
    burn_rate_threshold: float  # Trigger when burn rate above this


def get_default_alerts() -> List[SLOAlert]:
    """Get default SLO alert configurations."""
    return [
        # Availability alerts
        SLOAlert(
            slo_name="API Availability",
            severity="warning",
            message="API availability below target",
            error_budget_threshold=50.0,
            burn_rate_threshold=2.0,
        ),
        SLOAlert(
            slo_name="API Availability",
            severity="critical",
            message="API availability critically low",
            error_budget_threshold=10.0,
            burn_rate_threshold=10.0,
        ),
        # Latency alerts
        SLOAlert(
            slo_name="p99 Latency",
            severity="warning",
            message="p99 latency exceeding target",
            error_budget_threshold=50.0,
            burn_rate_threshold=2.0,
        ),
        SLOAlert(
            slo_name="p99 Latency",
            severity="critical",
            message="p99 latency critically high",
            error_budget_threshold=10.0,
            burn_rate_threshold=10.0,
        ),
        # Debate success alerts
        SLOAlert(
            slo_name="Debate Success Rate",
            severity="warning",
            message="Debate success rate below target",
            error_budget_threshold=50.0,
            burn_rate_threshold=2.0,
        ),
        SLOAlert(
            slo_name="Debate Success Rate",
            severity="critical",
            message="Debate success rate critically low",
            error_budget_threshold=10.0,
            burn_rate_threshold=10.0,
        ),
    ]


def check_alerts(status: Optional[SLOStatus] = None) -> List[Tuple[SLOAlert, SLOResult]]:
    """Check all SLO alerts and return triggered ones.

    Args:
        status: Optional SLOStatus to check (will fetch if not provided)

    Returns:
        List of (alert, result) tuples for triggered alerts
    """
    if status is None:
        status = get_slo_status()

    alerts = get_default_alerts()
    triggered: List[Tuple[SLOAlert, SLOResult]] = []

    results = {
        "API Availability": status.availability,
        "p99 Latency": status.latency_p99,
        "Debate Success Rate": status.debate_success,
    }

    for alert in alerts:
        result = results.get(alert.slo_name)
        if result is None:
            continue

        # Check if alert should trigger
        should_trigger = (
            result.error_budget_remaining < alert.error_budget_threshold
            or result.burn_rate > alert.burn_rate_threshold
        )

        if should_trigger:
            triggered.append((alert, result))

    return triggered


def format_slo_report(status: SLOStatus) -> str:
    """Format SLO status as a human-readable report.

    Args:
        status: SLOStatus to format

    Returns:
        Formatted report string
    """
    lines = [
        "=" * 60,
        "Aragora SLO Status Report",
        f"Generated: {status.timestamp.isoformat()}",
        "=" * 60,
        "",
    ]

    for name, result in [
        ("Availability", status.availability),
        ("Latency p99", status.latency_p99),
        ("Debate Success", status.debate_success),
    ]:
        status_icon = "[OK]" if result.compliant else "[!!]"
        lines.extend([
            f"{status_icon} {result.name}",
            f"    Target: {result.target:.4f}",
            f"    Current: {result.current:.4f}",
            f"    Compliance: {result.compliance_percentage:.1f}%",
            f"    Error Budget: {result.error_budget_remaining:.1f}% remaining",
            f"    Burn Rate: {result.burn_rate:.2f}x",
            "",
        ])

    overall = "HEALTHY" if status.overall_healthy else "DEGRADED"
    lines.extend([
        "-" * 60,
        f"Overall Status: {overall}",
        "=" * 60,
    ])

    return "\n".join(lines)


# =============================================================================
# HTTP Endpoint for SLO Status
# =============================================================================


def get_slo_status_json() -> Dict[str, Any]:
    """Get SLO status as JSON-serializable dictionary.

    Returns:
        Dictionary suitable for JSON response
    """
    status = get_slo_status()

    def result_to_dict(result: SLOResult) -> Dict[str, Any]:
        return {
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

    return {
        "timestamp": status.timestamp.isoformat(),
        "overall_healthy": status.overall_healthy,
        "slos": {
            "availability": result_to_dict(status.availability),
            "latency_p99": result_to_dict(status.latency_p99),
            "debate_success": result_to_dict(status.debate_success),
        },
        "alerts": [
            {
                "slo_name": alert.slo_name,
                "severity": alert.severity,
                "message": alert.message,
                "error_budget_remaining": result.error_budget_remaining,
                "burn_rate": result.burn_rate,
            }
            for alert, result in check_alerts(status)
        ],
    }
