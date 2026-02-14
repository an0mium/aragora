"""
Custom and application-level metrics.

Provides Prometheus metrics for application-specific functionality including:
- Gauntlet exports
- Backup operations
- Handler/endpoint tracking decorators
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, TypeVar, cast
from collections.abc import Callable, Coroutine, Generator

from aragora.observability.metrics.base import NoOpMetric, get_metrics_enabled

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

# Gauntlet metrics
GAUNTLET_EXPORTS_TOTAL: Any = None
GAUNTLET_EXPORT_LATENCY: Any = None
GAUNTLET_EXPORT_SIZE: Any = None

# Backup metrics
BACKUP_DURATION: Any = None
BACKUP_SIZE: Any = None
BACKUP_SUCCESS: Any = None
LAST_BACKUP_TIMESTAMP: Any = None
BACKUP_VERIFICATION_DURATION: Any = None
BACKUP_VERIFICATION_SUCCESS: Any = None
BACKUP_RESTORE_SUCCESS: Any = None

_initialized = False


def init_custom_metrics() -> None:
    """Initialize custom and application-level metrics."""
    global _initialized
    global GAUNTLET_EXPORTS_TOTAL, GAUNTLET_EXPORT_LATENCY, GAUNTLET_EXPORT_SIZE
    global BACKUP_DURATION, BACKUP_SIZE, BACKUP_SUCCESS, LAST_BACKUP_TIMESTAMP
    global BACKUP_VERIFICATION_DURATION, BACKUP_VERIFICATION_SUCCESS
    global BACKUP_RESTORE_SUCCESS

    if _initialized:
        return

    if not get_metrics_enabled():
        _init_noop_metrics()
        _initialized = True
        return

    try:
        from prometheus_client import Counter, Gauge, Histogram

        # Gauntlet metrics
        GAUNTLET_EXPORTS_TOTAL = Counter(
            "aragora_gauntlet_exports_total",
            "Total Gauntlet exports",
            ["format", "type", "status"],
        )

        GAUNTLET_EXPORT_LATENCY = Histogram(
            "aragora_gauntlet_export_latency_seconds",
            "Gauntlet export latency",
            ["format", "type"],
            buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0],
        )

        GAUNTLET_EXPORT_SIZE = Histogram(
            "aragora_gauntlet_export_size_bytes",
            "Gauntlet export size",
            ["format", "type"],
            buckets=[1000, 10000, 100000, 1000000, 10000000],
        )

        # Backup metrics
        BACKUP_DURATION = Histogram(
            "aragora_backup_duration_seconds",
            "Backup operation duration",
            ["type"],
            buckets=[1, 5, 10, 30, 60, 120, 300, 600],
        )

        BACKUP_SIZE = Gauge(
            "aragora_backup_size_bytes",
            "Size of last backup",
            ["type"],
        )

        BACKUP_SUCCESS = Counter(
            "aragora_backup_success_total",
            "Successful backup operations",
            ["type"],
        )

        LAST_BACKUP_TIMESTAMP = Gauge(
            "aragora_last_backup_timestamp",
            "Unix timestamp of last successful backup",
            ["type"],
        )

        BACKUP_VERIFICATION_DURATION = Histogram(
            "aragora_backup_verification_duration_seconds",
            "Backup verification duration",
            buckets=[0.5, 1, 2, 5, 10, 30],
        )

        BACKUP_VERIFICATION_SUCCESS = Counter(
            "aragora_backup_verification_success_total",
            "Successful backup verifications",
        )

        BACKUP_RESTORE_SUCCESS = Counter(
            "aragora_backup_restore_success_total",
            "Successful backup restores",
        )

        _initialized = True
        logger.debug("Custom metrics initialized")

    except (ImportError, ValueError):
        _init_noop_metrics()
        _initialized = True


def _init_noop_metrics() -> None:
    """Initialize no-op metrics when Prometheus is disabled."""
    global GAUNTLET_EXPORTS_TOTAL, GAUNTLET_EXPORT_LATENCY, GAUNTLET_EXPORT_SIZE
    global BACKUP_DURATION, BACKUP_SIZE, BACKUP_SUCCESS, LAST_BACKUP_TIMESTAMP
    global BACKUP_VERIFICATION_DURATION, BACKUP_VERIFICATION_SUCCESS
    global BACKUP_RESTORE_SUCCESS

    noop = NoOpMetric()
    GAUNTLET_EXPORTS_TOTAL = noop
    GAUNTLET_EXPORT_LATENCY = noop
    GAUNTLET_EXPORT_SIZE = noop
    BACKUP_DURATION = noop
    BACKUP_SIZE = noop
    BACKUP_SUCCESS = noop
    LAST_BACKUP_TIMESTAMP = noop
    BACKUP_VERIFICATION_DURATION = noop
    BACKUP_VERIFICATION_SUCCESS = noop
    BACKUP_RESTORE_SUCCESS = noop


def _ensure_init() -> None:
    """Ensure metrics are initialized."""
    if not _initialized:
        init_custom_metrics()


# =============================================================================
# Gauntlet Recording Functions
# =============================================================================


def record_gauntlet_export(
    format: str,
    export_type: str,
    success: bool,
    latency_seconds: float,
    size_bytes: int = 0,
) -> None:
    """Record a Gauntlet export operation.

    Args:
        format: Export format (e.g., "json", "csv")
        export_type: Type of export (e.g., "findings", "receipts")
        success: Whether the export succeeded
        latency_seconds: Export latency in seconds
        size_bytes: Size of export in bytes
    """
    _ensure_init()
    status = "success" if success else "error"
    GAUNTLET_EXPORTS_TOTAL.labels(format=format, type=export_type, status=status).inc()
    GAUNTLET_EXPORT_LATENCY.labels(format=format, type=export_type).observe(latency_seconds)
    if size_bytes > 0:
        GAUNTLET_EXPORT_SIZE.labels(format=format, type=export_type).observe(size_bytes)


@contextmanager
def track_gauntlet_export(format: str, export_type: str) -> Generator[dict[str, Any], None, None]:
    """Context manager to track Gauntlet export operations.

    Args:
        format: Export format
        export_type: Type of export

    Yields:
        A dict where 'size_bytes' can be set to record export size

    Example:
        with track_gauntlet_export("json", "findings") as ctx:
            data = export_findings()
            ctx["size_bytes"] = len(data)
    """
    _ensure_init()
    start = time.perf_counter()
    ctx: dict[str, Any] = {"size_bytes": 0}
    success = True
    try:
        yield ctx
    except Exception:  # noqa: BLE001 - Intentional: set success flag for metrics before re-raising
        success = False
        raise
    finally:
        latency = time.perf_counter() - start
        record_gauntlet_export(format, export_type, success, latency, ctx.get("size_bytes", 0))


# =============================================================================
# Backup Recording Functions
# =============================================================================


def record_backup_operation(
    backup_type: str,
    duration_seconds: float,
    size_bytes: int,
    success: bool = True,
) -> None:
    """Record a backup operation.

    Args:
        backup_type: Type of backup (e.g., "full", "incremental")
        duration_seconds: Backup duration in seconds
        size_bytes: Size of backup in bytes
        success: Whether the backup succeeded
    """
    _ensure_init()
    BACKUP_DURATION.labels(type=backup_type).observe(duration_seconds)
    if success:
        BACKUP_SIZE.labels(type=backup_type).set(size_bytes)
        BACKUP_SUCCESS.labels(type=backup_type).inc()
        LAST_BACKUP_TIMESTAMP.labels(type=backup_type).set(time.time())


def record_backup_verification(duration_seconds: float, success: bool = True) -> None:
    """Record a backup verification operation.

    Args:
        duration_seconds: Verification duration in seconds
        success: Whether the verification succeeded
    """
    _ensure_init()
    BACKUP_VERIFICATION_DURATION.observe(duration_seconds)
    if success:
        BACKUP_VERIFICATION_SUCCESS.inc()


def record_backup_restore(success: bool = True) -> None:
    """Record a backup restore operation.

    Args:
        success: Whether the restore succeeded
    """
    _ensure_init()
    if success:
        BACKUP_RESTORE_SUCCESS.inc()


# =============================================================================
# Handler Tracking Decorator
# =============================================================================


def track_handler(handler_name: str, method: str = "POST") -> Callable[[F], F]:
    """Decorator factory to track handler metrics (supports both sync and async).

    Tracks:
    - Request count with success/error status
    - Request latency (p50/p95/p99 via histogram)
    - Error rate

    Args:
        handler_name: The handler name for labeling (e.g., "email/prioritize")
        method: HTTP method for this handler (default: POST)

    Example:
        @track_handler("email/prioritize")
        async def handle_prioritize_email(data):
            ...

        @track_handler("payments/process", method="POST")
        def handle_process_payment(data):  # Also works with sync handlers
            ...
    """
    # Import here to avoid circular imports
    from aragora.observability.metrics.request import record_request

    def _extract_status(result: Any) -> int:
        """Extract status code from result if it indicates failure."""
        if isinstance(result, dict):
            if result.get("success") is False:
                return result.get("status", 400)
            elif "error" in result and "success" not in result:
                return result.get("status", 500)
        return 200

    def decorator(func: F) -> F:
        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            status = 200
            try:
                result = func(*args, **kwargs)
                status = _extract_status(result)
                return result
            except Exception:  # noqa: BLE001 - Intentional: set status for metrics before re-raising
                status = 500
                raise
            finally:
                latency = time.perf_counter() - start
                record_request(method, handler_name, status, latency)

        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            status = 200
            try:
                result = await cast(Coroutine[Any, Any, Any], func(*args, **kwargs))
                status = _extract_status(result)
                return result
            except Exception:  # noqa: BLE001 - Intentional: set status for metrics before re-raising
                status = 500
                raise
            finally:
                latency = time.perf_counter() - start
                record_request(method, handler_name, status, latency)

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return cast(F, async_wrapper)
        return cast(F, sync_wrapper)

    return decorator


__all__ = [
    # Gauntlet metrics
    "GAUNTLET_EXPORTS_TOTAL",
    "GAUNTLET_EXPORT_LATENCY",
    "GAUNTLET_EXPORT_SIZE",
    # Backup metrics
    "BACKUP_DURATION",
    "BACKUP_SIZE",
    "BACKUP_SUCCESS",
    "LAST_BACKUP_TIMESTAMP",
    "BACKUP_VERIFICATION_DURATION",
    "BACKUP_VERIFICATION_SUCCESS",
    "BACKUP_RESTORE_SUCCESS",
    # Functions
    "init_custom_metrics",
    "record_gauntlet_export",
    "track_gauntlet_export",
    "record_backup_operation",
    "record_backup_verification",
    "record_backup_restore",
    "track_handler",
]
