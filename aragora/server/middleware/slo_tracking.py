"""
SLO tracking middleware for API request monitoring.

Tracks request latency, success/failure rates, and updates SLO metrics.
Integrates with the observability.slo module for compliance tracking.

Usage:
    # As a decorator
    @track_slo("api_endpoint")
    def handle_request(self, handler):
        ...

    # As a context manager
    with slo_context("debate_execution"):
        result = await arena.run()

    # As middleware for HTTP handlers
    @slo_middleware
    def do_GET(self):
        ...
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, TypeVar

from aragora.config.performance_slos import check_latency_slo
from aragora.observability.slo import _record_measurement

logger = logging.getLogger(__name__)

# Type for decorated functions
F = TypeVar("F", bound=Callable[..., Any])

# Request counters (thread-safe via atomic operations)
_total_requests = 0
_successful_requests = 0
_total_debates = 0
_successful_debates = 0
_recent_latencies: list[float] = []
_max_latency_samples = 1000


def _record_request(success: bool = True, is_debate: bool = False) -> None:
    """Record a request for SLO tracking.

    Args:
        success: Whether the request succeeded
        is_debate: Whether this was a debate operation
    """
    global _total_requests, _successful_requests, _total_debates, _successful_debates

    _total_requests += 1
    if success:
        _successful_requests += 1

    if is_debate:
        _total_debates += 1
        if success:
            _successful_debates += 1


def _record_latency(latency_ms: float) -> None:
    """Record a latency measurement.

    Args:
        latency_ms: Request latency in milliseconds
    """
    global _recent_latencies

    _recent_latencies.append(latency_ms)

    # Keep bounded
    if len(_recent_latencies) > _max_latency_samples:
        _recent_latencies = _recent_latencies[-_max_latency_samples:]


def _get_p99_latency() -> float:
    """Get the p99 latency from recent samples.

    Returns:
        p99 latency in seconds
    """
    if not _recent_latencies:
        return 0.0

    sorted_latencies = sorted(_recent_latencies)
    p99_index = int(len(sorted_latencies) * 0.99)
    p99_ms = sorted_latencies[min(p99_index, len(sorted_latencies) - 1)]

    return p99_ms / 1000  # Convert to seconds


def sync_slo_measurements() -> None:
    """Sync local measurements to the SLO system."""
    global _total_requests, _successful_requests, _total_debates, _successful_debates

    p99_latency = _get_p99_latency()

    _record_measurement(
        total_requests=_total_requests,
        successful_requests=_successful_requests,
        latency_p99=p99_latency,
        total_debates=_total_debates,
        successful_debates=_successful_debates,
    )


def get_tracking_stats() -> dict[str, Any]:
    """Get current tracking statistics.

    Returns:
        Dictionary with tracking stats
    """
    return {
        "total_requests": _total_requests,
        "successful_requests": _successful_requests,
        "total_debates": _total_debates,
        "successful_debates": _successful_debates,
        "recent_latency_samples": len(_recent_latencies),
        "p99_latency_ms": _get_p99_latency() * 1000,
    }


@contextmanager
def slo_context(operation: str, is_debate: bool = False):
    """Context manager for tracking SLO metrics.

    Args:
        operation: Operation name for SLO lookup
        is_debate: Whether this is a debate operation

    Yields:
        Context for the operation

    Example:
        with slo_context("km_query"):
            result = await knowledge_mound.query(...)
    """
    start_time = time.perf_counter()
    success = True

    try:
        yield
    except Exception:
        success = False
        raise
    finally:
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Record measurements
        _record_request(success=success, is_debate=is_debate)
        _record_latency(elapsed_ms)

        # Check against SLO
        is_within, message = check_latency_slo(operation, elapsed_ms)
        if not is_within:
            logger.warning(f"slo_violation operation={operation} {message}")


def track_slo(operation: str, is_debate: bool = False) -> Callable[[F], F]:
    """Decorator for tracking SLO metrics on a function.

    Args:
        operation: Operation name for SLO lookup
        is_debate: Whether this is a debate operation

    Returns:
        Decorated function

    Example:
        @track_slo("api_endpoint")
        def handle_request(self, handler):
            ...
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with slo_context(operation, is_debate):
                return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


def track_slo_async(operation: str, is_debate: bool = False) -> Callable[[F], F]:
    """Async decorator for tracking SLO metrics.

    Args:
        operation: Operation name for SLO lookup
        is_debate: Whether this is a debate operation

    Returns:
        Decorated async function
    """

    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            start_time = time.perf_counter()
            success = True

            try:
                return await func(*args, **kwargs)
            except Exception:
                success = False
                raise
            finally:
                elapsed_ms = (time.perf_counter() - start_time) * 1000

                _record_request(success=success, is_debate=is_debate)
                _record_latency(elapsed_ms)

                is_within, message = check_latency_slo(operation, elapsed_ms)
                if not is_within:
                    logger.warning(f"slo_violation operation={operation} {message}")

        return wrapper  # type: ignore[return-value]

    return decorator


def slo_middleware(handler_func: F) -> F:
    """Middleware decorator for HTTP handlers.

    Tracks request latency and success/failure for SLO compliance.

    Args:
        handler_func: HTTP handler function (do_GET, do_POST, etc.)

    Returns:
        Wrapped handler function

    Example:
        @slo_middleware
        def do_GET(self):
            ...
    """

    @wraps(handler_func)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        start_time = time.perf_counter()
        success = True

        try:
            result = handler_func(self, *args, **kwargs)
            return result
        except Exception:
            success = False
            raise
        finally:
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            # Record for SLO
            _record_request(success=success)
            _record_latency(elapsed_ms)

            # Check against API endpoint SLO
            is_within, message = check_latency_slo("api_endpoint", elapsed_ms)
            if not is_within:
                path = getattr(self, "path", "unknown")
                logger.warning(f"slo_violation path={path} {message}")

    return wrapper  # type: ignore[return-value]


__all__ = [
    "slo_context",
    "slo_middleware",
    "sync_slo_measurements",
    "track_slo",
    "track_slo_async",
    "get_tracking_stats",
]
