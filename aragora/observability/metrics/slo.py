"""
SLO (Service Level Objective) Prometheus metrics.

Provides metrics for monitoring SLO compliance:
- SLO check totals by operation and result
- SLO violation counters
- Latency histograms per operation
- Webhook notifications for violations

Usage:
    from aragora.observability.metrics.slo import (
        record_slo_check,
        record_slo_violation,
        track_operation_slo,
    )

    # Record an SLO check
    record_slo_check("km_query", passed=True, percentile="p99")

    # Record a violation with context
    record_slo_violation("km_query", "p99", latency_ms=550.0, threshold_ms=500.0)

    # Context manager for automatic SLO tracking
    with track_operation_slo("km_query") as ctx:
        result = await mound.query(...)
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, Generator, List, Optional

from aragora.observability.config import get_metrics_config
from aragora.observability.metrics.base import NoOpMetric

logger = logging.getLogger(__name__)

__all__ = [
    # Core metrics
    "init_slo_metrics",
    "record_slo_check",
    "record_slo_violation",
    "record_operation_latency",
    "check_and_record_slo",
    "check_and_record_slo_with_recovery",
    "track_operation_slo",
    "get_slo_metrics_summary",
    # Webhook integration
    "init_slo_webhooks",
    "notify_slo_violation",
    "notify_slo_recovery",
    "get_slo_webhook_status",
    "get_violation_state",
    "SLOWebhookConfig",
    "SEVERITY_ORDER",
]

# Webhook notification callback (set by init_slo_webhooks)
_webhook_callback: Optional[Callable[[Dict[str, Any]], bool]] = None

# Violation buffer for batching webhook notifications
_violation_buffer: List[Dict[str, Any]] = []
_buffer_lock: Optional[Any] = None  # threading.Lock set on init


# Config for webhook notifications
@dataclass
class SLOWebhookConfig:
    """Configuration for SLO webhook notifications."""

    enabled: bool = True
    min_severity: str = "minor"  # minor, moderate, major, critical
    batch_size: int = 10  # Max violations per webhook call
    cooldown_seconds: float = 60.0  # Min time between notifications for same operation


# Severity ordering for filtering
SEVERITY_ORDER = {"minor": 0, "moderate": 1, "major": 2, "critical": 3}

# Cooldown tracking
_last_notification: Dict[str, float] = {}

# Track violation state for recovery detection
_violation_state: Dict[str, Dict[str, Any]] = {}  # operation -> {in_violation, last_severity, ...}

# Prometheus metrics - initialized lazily
_initialized = False

# Metric instances (set during initialization)
SLO_CHECKS_TOTAL: Any = None
SLO_VIOLATIONS_TOTAL: Any = None
SLO_LATENCY_HISTOGRAM: Any = None
SLO_VIOLATION_MARGIN: Any = None  # How much over the threshold


def init_slo_metrics() -> bool:
    """Initialize SLO Prometheus metrics lazily.

    Returns:
        True if metrics were successfully initialized
    """
    global _initialized
    global SLO_CHECKS_TOTAL, SLO_VIOLATIONS_TOTAL
    global SLO_LATENCY_HISTOGRAM, SLO_VIOLATION_MARGIN

    if _initialized:
        return True

    config = get_metrics_config()
    if not config.enabled:
        # Use NoOp metrics
        SLO_CHECKS_TOTAL = NoOpMetric()
        SLO_VIOLATIONS_TOTAL = NoOpMetric()
        SLO_LATENCY_HISTOGRAM = NoOpMetric()
        SLO_VIOLATION_MARGIN = NoOpMetric()
        _initialized = True
        return False

    try:
        from prometheus_client import Counter, Histogram, Gauge

        SLO_CHECKS_TOTAL = Counter(
            "aragora_slo_checks_total",
            "Total number of SLO checks performed",
            ["operation", "percentile", "result"],
        )

        SLO_VIOLATIONS_TOTAL = Counter(
            "aragora_slo_violations_total",
            "Total number of SLO violations",
            ["operation", "percentile", "severity"],
        )

        # Buckets aligned with typical SLO thresholds
        SLO_LATENCY_HISTOGRAM = Histogram(
            "aragora_slo_operation_latency_ms",
            "Operation latency in milliseconds for SLO tracking",
            ["operation"],
            buckets=[10, 25, 50, 100, 150, 200, 300, 500, 1000, 2000, 5000, 10000, 30000],
        )

        SLO_VIOLATION_MARGIN = Gauge(
            "aragora_slo_violation_margin_ms",
            "How much the latency exceeded the SLO threshold (0 if within SLO)",
            ["operation", "percentile"],
        )

        _initialized = True
        logger.info("SLO metrics initialized")
        return True

    except ImportError:
        logger.warning("prometheus-client not installed, SLO metrics disabled")
        SLO_CHECKS_TOTAL = NoOpMetric()
        SLO_VIOLATIONS_TOTAL = NoOpMetric()
        SLO_LATENCY_HISTOGRAM = NoOpMetric()
        SLO_VIOLATION_MARGIN = NoOpMetric()
        _initialized = True
        return False


def record_slo_check(
    operation: str,
    passed: bool,
    percentile: str = "p99",
) -> None:
    """Record an SLO check result.

    Args:
        operation: Operation name (e.g., "km_query", "consensus_ingestion")
        passed: Whether the check passed
        percentile: SLO percentile checked (p50, p90, p99)
    """
    if not _initialized:
        init_slo_metrics()

    result = "pass" if passed else "fail"
    SLO_CHECKS_TOTAL.labels(
        operation=operation,
        percentile=percentile,
        result=result,
    ).inc()


def record_slo_violation(
    operation: str,
    percentile: str,
    latency_ms: float,
    threshold_ms: float,
    severity: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    notify_webhook: bool = True,
) -> str:
    """Record an SLO violation with context and optionally notify via webhook.

    Args:
        operation: Operation name
        percentile: SLO percentile that was violated
        latency_ms: Actual latency in milliseconds
        threshold_ms: SLO threshold in milliseconds
        severity: Violation severity (auto-calculated if not provided)
        context: Optional additional context for webhook notification
        notify_webhook: Whether to send webhook notification (default True)

    Returns:
        The calculated severity level
    """
    if not _initialized:
        init_slo_metrics()

    # Auto-calculate severity based on how much threshold was exceeded
    if severity is None:
        margin_pct = ((latency_ms - threshold_ms) / threshold_ms) * 100 if threshold_ms > 0 else 0
        if margin_pct < 20:
            severity = "minor"  # < 20% over
        elif margin_pct < 50:
            severity = "moderate"  # 20-50% over
        elif margin_pct < 100:
            severity = "major"  # 50-100% over
        else:
            severity = "critical"  # > 100% over (2x threshold)

    SLO_VIOLATIONS_TOTAL.labels(
        operation=operation,
        percentile=percentile,
        severity=severity,
    ).inc()

    # Record the margin for alerting
    margin = max(0, latency_ms - threshold_ms)
    SLO_VIOLATION_MARGIN.labels(
        operation=operation,
        percentile=percentile,
    ).set(margin)

    # Send webhook notification if enabled
    if notify_webhook:
        notify_slo_violation(
            operation=operation,
            percentile=percentile,
            latency_ms=latency_ms,
            threshold_ms=threshold_ms,
            severity=severity,
            context=context,
        )

    return severity


def record_operation_latency(operation: str, latency_ms: float) -> None:
    """Record operation latency for SLO histogram.

    Args:
        operation: Operation name
        latency_ms: Latency in milliseconds
    """
    if not _initialized:
        init_slo_metrics()

    SLO_LATENCY_HISTOGRAM.labels(operation=operation).observe(latency_ms)


def check_and_record_slo(
    operation: str,
    latency_ms: float,
    percentile: str = "p99",
) -> tuple[bool, str]:
    """Check SLO and record metrics in one call.

    Combines check_latency_slo with metric recording.

    Args:
        operation: Operation name (must match SLOConfig attributes)
        latency_ms: Measured latency in milliseconds
        percentile: SLO percentile to check (p50, p90, p99)

    Returns:
        Tuple of (is_within_slo, message)
    """
    from aragora.config.performance_slos import check_latency_slo, get_slo_config

    if not _initialized:
        init_slo_metrics()

    # Record latency in histogram
    record_operation_latency(operation, latency_ms)

    # Check against SLO
    passed, message = check_latency_slo(operation, latency_ms, percentile)

    # Record the check result
    record_slo_check(operation, passed, percentile)

    # If failed, record violation with threshold
    if not passed:
        config = get_slo_config()
        slo = getattr(config, operation, None)
        if slo:
            threshold_ms = getattr(slo, f"{percentile}_ms", slo.p99_ms)
            record_slo_violation(operation, percentile, latency_ms, threshold_ms)
            logger.warning(message)

    return passed, message


@contextmanager
def track_operation_slo(
    operation: str,
    percentile: str = "p99",
    log_violations: bool = True,
) -> Generator[dict, None, None]:
    """Context manager for tracking operation SLO compliance.

    Automatically measures latency and records SLO metrics.

    Args:
        operation: Operation name (must match SLOConfig attributes)
        percentile: SLO percentile to check
        log_violations: Whether to log warning on violation

    Yields:
        Dict that can be used to store context (e.g., {"size_bytes": 1024})

    Example:
        with track_operation_slo("km_query") as ctx:
            result = await mound.query(...)
            ctx["result_count"] = len(result.items)
    """
    if not _initialized:
        init_slo_metrics()

    ctx: dict = {}
    start_time = time.perf_counter()

    try:
        yield ctx
    finally:
        latency_ms = (time.perf_counter() - start_time) * 1000
        passed, message = check_and_record_slo(operation, latency_ms, percentile)

        if not passed and log_violations:
            logger.warning(
                "SLO violation: %s (context: %s)",
                message,
                ctx if ctx else "none",
            )


def get_slo_metrics_summary() -> dict:
    """Get a summary of SLO metrics for observability endpoints.

    Returns:
        Dict with metric summaries
    """
    if not _initialized:
        init_slo_metrics()

    # This would need prometheus_client inspection which varies
    # Return basic status for now
    return {
        "initialized": _initialized,
        "metrics_enabled": get_metrics_config().enabled,
        "webhooks_enabled": _webhook_callback is not None,
        "tracked_operations": [
            "km_query",
            "km_ingestion",
            "km_checkpoint",
            "consensus_ingestion",
            "consensus_detection",
            "adapter_sync",
            "event_dispatch",
            "handler_execution",
            "memory_store",
            "memory_recall",
            "debate_round",
            "api_endpoint",
        ],
    }


# --- Webhook Integration ---


def init_slo_webhooks(
    webhook_config: Optional[SLOWebhookConfig] = None,
) -> bool:
    """Initialize SLO webhook notifications.

    Connects SLO violations to the webhook dispatcher for external alerting.

    Args:
        webhook_config: Optional configuration for webhook behavior

    Returns:
        True if webhooks were successfully initialized
    """
    global _webhook_callback, _buffer_lock

    try:
        import threading
        from aragora.integrations.webhooks import get_dispatcher

        _buffer_lock = threading.Lock()

        dispatcher = get_dispatcher()
        if dispatcher is None:
            logger.debug("Webhook dispatcher not available, SLO webhooks disabled")
            return False

        # Create callback that sends to webhook dispatcher
        config = webhook_config or SLOWebhookConfig()

        def send_violation_webhook(violation_data: Dict[str, Any]) -> bool:
            """Send violation to webhook dispatcher."""
            severity = violation_data.get("severity", "minor")

            # Check severity threshold
            if SEVERITY_ORDER.get(severity, 0) < SEVERITY_ORDER.get(config.min_severity, 0):
                return False

            # Build webhook event
            operation = violation_data.get("operation", "unknown")
            event = {
                "type": "slo_violation",
                "timestamp": datetime.now().isoformat(),
                "operation": operation,
                "percentile": violation_data.get("percentile", "p99"),
                "severity": severity,
                "latency_ms": violation_data.get("latency_ms", 0),
                "threshold_ms": violation_data.get("threshold_ms", 0),
                "margin_ms": violation_data.get("margin_ms", 0),
                "margin_percent": violation_data.get("margin_percent", 0),
                "context": violation_data.get("context", {}),
            }

            return dispatcher.enqueue(event)

        _webhook_callback = send_violation_webhook
        logger.info("SLO webhook notifications initialized")
        return True

    except ImportError as e:
        logger.debug(f"Could not initialize SLO webhooks: {e}")
        return False
    except Exception as e:
        logger.warning(f"Failed to initialize SLO webhooks: {e}")
        return False


def notify_slo_violation(
    operation: str,
    percentile: str,
    latency_ms: float,
    threshold_ms: float,
    severity: str,
    context: Optional[Dict[str, Any]] = None,
    cooldown_seconds: float = 60.0,
) -> bool:
    """Send SLO violation notification via webhook.

    Args:
        operation: Operation name that violated SLO
        percentile: SLO percentile that was violated
        latency_ms: Actual latency in milliseconds
        threshold_ms: SLO threshold in milliseconds
        severity: Violation severity (minor, moderate, major, critical)
        context: Optional additional context
        cooldown_seconds: Minimum time between notifications for same operation

    Returns:
        True if notification was sent successfully
    """
    if _webhook_callback is None:
        return False

    # Check cooldown
    now = time.time()
    last_time = _last_notification.get(operation, 0)
    if now - last_time < cooldown_seconds:
        logger.debug(f"SLO webhook cooldown for {operation}, skipping notification")
        return False

    margin_ms = latency_ms - threshold_ms
    margin_percent = (margin_ms / threshold_ms) * 100 if threshold_ms > 0 else 0

    violation_data = {
        "operation": operation,
        "percentile": percentile,
        "latency_ms": latency_ms,
        "threshold_ms": threshold_ms,
        "margin_ms": margin_ms,
        "margin_percent": margin_percent,
        "severity": severity,
        "context": context or {},
    }

    try:
        result = _webhook_callback(violation_data)
        if result:
            _last_notification[operation] = now
        return result
    except Exception as e:
        logger.debug(f"Failed to send SLO violation webhook: {e}")
        return False


def get_slo_webhook_status() -> Dict[str, Any]:
    """Get status of SLO webhook integration.

    Returns:
        Dict with webhook status information
    """
    return {
        "enabled": _webhook_callback is not None,
        "cooldown_active": {op: time.time() - ts < 60.0 for op, ts in _last_notification.items()},
        "buffer_size": len(_violation_buffer),
        "operations_in_violation": [
            op for op, state in _violation_state.items() if state.get("in_violation", False)
        ],
    }


def notify_slo_recovery(
    operation: str,
    percentile: str,
    latency_ms: float,
    threshold_ms: float,
    violation_duration_seconds: float,
    context: Optional[Dict[str, Any]] = None,
) -> bool:
    """Send SLO recovery notification via webhook.

    Called when an operation returns to SLO compliance after being in violation.

    Args:
        operation: Operation name that recovered
        percentile: SLO percentile that was violated
        latency_ms: Current latency (now within SLO)
        threshold_ms: SLO threshold in milliseconds
        violation_duration_seconds: How long the violation lasted
        context: Optional additional context

    Returns:
        True if notification was sent successfully
    """
    if _webhook_callback is None:
        return False

    {
        "operation": operation,
        "percentile": percentile,
        "latency_ms": latency_ms,
        "threshold_ms": threshold_ms,
        "margin_ms": threshold_ms - latency_ms,  # How much under threshold
        "violation_duration_seconds": violation_duration_seconds,
        "context": context or {},
    }

    try:
        # Import here to avoid circular imports
        from aragora.integrations.webhooks import get_dispatcher
        from datetime import datetime

        dispatcher = get_dispatcher()
        if not dispatcher:
            return False

        event = {
            "type": "slo_recovery",
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "percentile": percentile,
            "latency_ms": latency_ms,
            "threshold_ms": threshold_ms,
            "margin_ms": threshold_ms - latency_ms,
            "violation_duration_seconds": violation_duration_seconds,
            "context": context or {},
        }

        return dispatcher.enqueue(event)

    except Exception as e:
        logger.debug(f"Failed to send SLO recovery webhook: {e}")
        return False


def check_and_record_slo_with_recovery(
    operation: str,
    latency_ms: float,
    percentile: str = "p99",
    context: Optional[Dict[str, Any]] = None,
) -> tuple[bool, str]:
    """Check SLO, record metrics, and handle violation/recovery state.

    This is an enhanced version of check_and_record_slo that also:
    - Tracks violation state for each operation
    - Sends recovery notifications when an operation returns to compliance

    Args:
        operation: Operation name (must match SLOConfig attributes)
        latency_ms: Measured latency in milliseconds
        percentile: SLO percentile to check (p50, p90, p99)
        context: Optional context to include in webhook notifications

    Returns:
        Tuple of (is_within_slo, message)
    """
    from aragora.config.performance_slos import check_latency_slo, get_slo_config

    if not _initialized:
        init_slo_metrics()

    # Record latency in histogram
    record_operation_latency(operation, latency_ms)

    # Check against SLO
    passed, message = check_latency_slo(operation, latency_ms, percentile)

    # Record the check result
    record_slo_check(operation, passed, percentile)

    # Get current violation state for this operation
    current_state = _violation_state.get(operation, {"in_violation": False})

    if not passed:
        # SLO violated
        config = get_slo_config()
        slo = getattr(config, operation, None)
        if slo:
            threshold_ms = getattr(slo, f"{percentile}_ms", slo.p99_ms)
            severity = record_slo_violation(
                operation,
                percentile,
                latency_ms,
                threshold_ms,
                context=context,
                notify_webhook=True,
            )

            # Update violation state
            if not current_state.get("in_violation"):
                # Entering violation state
                _violation_state[operation] = {
                    "in_violation": True,
                    "violation_start": time.time(),
                    "last_severity": severity,
                    "percentile": percentile,
                    "threshold_ms": threshold_ms,
                }
            else:
                # Already in violation, update severity if worse
                if SEVERITY_ORDER.get(severity, 0) > SEVERITY_ORDER.get(
                    current_state.get("last_severity", "minor"), 0
                ):
                    _violation_state[operation]["last_severity"] = severity

            logger.warning(message)

    else:
        # SLO passed
        if current_state.get("in_violation"):
            # Recovering from violation!
            violation_start = current_state.get("violation_start", time.time())
            violation_duration = time.time() - violation_start
            threshold_ms = current_state.get("threshold_ms", 0)

            # Send recovery notification
            notify_slo_recovery(
                operation=operation,
                percentile=current_state.get("percentile", percentile),
                latency_ms=latency_ms,
                threshold_ms=threshold_ms,
                violation_duration_seconds=violation_duration,
                context=context,
            )

            logger.info(
                f"SLO recovered for {operation}: latency={latency_ms:.1f}ms "
                f"(threshold={threshold_ms:.1f}ms), "
                f"violation lasted {violation_duration:.1f}s"
            )

            # Clear violation state
            _violation_state[operation] = {"in_violation": False}

    return passed, message


def get_violation_state(operation: Optional[str] = None) -> Dict[str, Any]:
    """Get current violation state for operation(s).

    Args:
        operation: Specific operation to check, or None for all

    Returns:
        Dict with violation state information
    """
    if operation:
        return _violation_state.get(operation, {"in_violation": False})
    return dict(_violation_state)
