"""
SLO (Service Level Objective) Prometheus metrics.

Provides metrics for monitoring SLO compliance:
- SLO check totals by operation and result
- SLO violation counters
- Latency histograms per operation

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
from typing import Any, Generator, Optional

from aragora.observability.config import get_metrics_config
from aragora.observability.metrics.base import NoOpMetric

logger = logging.getLogger(__name__)

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
) -> None:
    """Record an SLO violation with context.

    Args:
        operation: Operation name
        percentile: SLO percentile that was violated
        latency_ms: Actual latency in milliseconds
        threshold_ms: SLO threshold in milliseconds
        severity: Violation severity (auto-calculated if not provided)
    """
    if not _initialized:
        init_slo_metrics()

    # Auto-calculate severity based on how much threshold was exceeded
    if severity is None:
        margin_pct = ((latency_ms - threshold_ms) / threshold_ms) * 100
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
