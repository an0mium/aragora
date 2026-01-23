"""
Handler Instrumentation for Aragora.

Provides decorators and middleware for instrumenting HTTP handlers with
metrics and tracing. This bridges the gap between the observability
infrastructure and the API handlers.

Usage:
    from aragora.observability.handler_instrumentation import (
        instrument_handler,
        track_request,
        MetricsMiddleware,
    )

    @instrument_handler("control_plane.agents.list")
    def handle_list_agents(self, query_params, handler):
        ...

    # Or use middleware for automatic instrumentation
    middleware = MetricsMiddleware()
    result = middleware.wrap(handler_func, path, method)
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Dict, Iterator, Optional, TypeVar

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


# Endpoint categorization for metric labels
ENDPOINT_CATEGORIES = {
    "/api/control-plane/agents": "control_plane.agents",
    "/api/control-plane/tasks": "control_plane.tasks",
    "/api/control-plane/deliberations": "control_plane.deliberations",
    "/api/control-plane/health": "control_plane.health",
    "/api/control-plane/stats": "control_plane.stats",
    "/api/debates": "debates",
    "/api/agents": "agents",
    "/api/rankings": "rankings",
    "/api/webhooks": "webhooks",
    "/api/knowledge": "knowledge",
    "/api/workflows": "workflows",
}


def _categorize_endpoint(path: str) -> str:
    """Categorize an endpoint path for metric labels."""
    for prefix, category in ENDPOINT_CATEGORIES.items():
        if path.startswith(prefix):
            return category

    # Extract first two path segments as fallback
    parts = path.strip("/").split("/")
    if len(parts) >= 2:
        return f"{parts[0]}.{parts[1]}"
    return "other"


def _safe_record_request(
    method: str,
    endpoint: str,
    status: int,
    latency: float,
) -> None:
    """Safely record request metrics."""
    try:
        from aragora.observability.metrics import record_request

        record_request(method, endpoint, status, latency)
    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"Failed to record request metrics: {e}")


def _safe_start_span(name: str, attributes: Optional[Dict[str, Any]] = None) -> Any:
    """Safely start a tracing span."""
    try:
        from aragora.observability.tracing import get_tracer

        tracer = get_tracer()
        span = tracer.start_as_current_span(name)
        if attributes and hasattr(span, "set_attributes"):
            span.set_attributes(attributes)
        return span
    except ImportError:
        return _NoOpContextManager()
    except Exception as e:
        logger.debug(f"Failed to start span: {e}")
        return _NoOpContextManager()


class _NoOpContextManager:
    """No-op context manager for when observability is disabled."""

    def __enter__(self) -> "_NoOpContextManager":
        return self

    def __exit__(self, *args: Any) -> None:
        pass

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def set_attributes(self, attributes: Dict[str, Any]) -> None:
        pass

    def record_exception(self, exc: BaseException) -> None:
        pass


@contextmanager
def track_request(
    method: str,
    path: str,
    *,
    include_tracing: bool = True,
) -> Iterator[Dict[str, Any]]:
    """Context manager for tracking request metrics and tracing.

    Args:
        method: HTTP method (GET, POST, etc.)
        path: Request path
        include_tracing: Whether to create a tracing span

    Yields:
        Context dict for setting response info

    Example:
        with track_request("GET", "/api/debates") as ctx:
            result = handle_request()
            ctx["status"] = 200
    """
    category = _categorize_endpoint(path)
    context: Dict[str, Any] = {"status": 200, "error": None}
    start_time = time.perf_counter()

    span = None
    if include_tracing:
        span = _safe_start_span(
            f"http.{method.lower()}.{category}",
            {"http.method": method, "http.path": path},
        )

    try:
        if hasattr(span, "__enter__"):
            span.__enter__()
        yield context
    except Exception as e:
        context["status"] = 500
        context["error"] = e
        if span and hasattr(span, "record_exception"):
            span.record_exception(e)
        raise
    finally:
        latency = time.perf_counter() - start_time
        _safe_record_request(method, category, context["status"], latency)

        if span and hasattr(span, "__exit__"):
            span.__exit__(None, None, None)


def instrument_handler(
    name: str,
    *,
    include_tracing: bool = True,
    record_args: bool = False,
) -> Callable[[F], F]:
    """Decorator to instrument a handler method with metrics and tracing.

    Args:
        name: Handler name for metrics (e.g., "control_plane.agents.list")
        include_tracing: Whether to create a tracing span
        record_args: Whether to record query params as span attributes

    Returns:
        Decorated function with instrumentation

    Example:
        @instrument_handler("control_plane.agents.list")
        def handle_list_agents(self, query_params, handler):
            ...
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Extract method and path from arguments
            handler = None
            path = "unknown"
            method = "UNKNOWN"

            # Try to find handler in args/kwargs
            for arg in args:
                if hasattr(arg, "path") and hasattr(arg, "command"):
                    handler = arg
                    path = getattr(arg, "path", "unknown")
                    method = getattr(arg, "command", "GET")
                    break

            if handler is None:
                handler = kwargs.get("handler")
                if handler:
                    path = getattr(handler, "path", "unknown")
                    method = getattr(handler, "command", "GET")

            start_time = time.perf_counter()
            status = 200

            # Start span if tracing enabled
            span = None
            if include_tracing:
                attrs = {"http.method": method, "http.path": path, "handler": name}
                if record_args and kwargs.get("query_params"):
                    # Only record safe params
                    safe_params = {
                        k: v
                        for k, v in kwargs["query_params"].items()
                        if k in ("limit", "offset", "status", "type", "sort")
                    }
                    if safe_params:
                        attrs["query_params"] = str(safe_params)
                span = _safe_start_span(name, attrs)

            try:
                if hasattr(span, "__enter__"):
                    span.__enter__()
                result = func(*args, **kwargs)

                # Extract status from result if possible
                if isinstance(result, tuple) and len(result) >= 2:
                    status = result[1] if isinstance(result[1], int) else 200
                elif hasattr(result, "status"):
                    status = result.status

                return result
            except Exception as e:
                status = 500
                if span and hasattr(span, "record_exception"):
                    span.record_exception(e)
                raise
            finally:
                latency = time.perf_counter() - start_time
                _safe_record_request(method, name, status, latency)

                if span and hasattr(span, "__exit__"):
                    span.__exit__(None, None, None)

        return wrapper  # type: ignore[return-value]

    return decorator


class MetricsMiddleware:
    """Middleware for automatic handler instrumentation.

    Wraps handler functions to automatically record metrics and traces
    for all requests without requiring individual decorators.

    Example:
        middleware = MetricsMiddleware()

        # In request handler
        def do_GET(self):
            result = middleware.wrap(
                self.handler_func,
                self.path,
                "GET",
                self.query_params,
            )
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        include_tracing: bool = True,
        exclude_paths: Optional[list[str]] = None,
    ):
        """Initialize middleware.

        Args:
            enabled: Whether metrics collection is enabled
            include_tracing: Whether to create tracing spans
            exclude_paths: Paths to exclude from instrumentation
        """
        self.enabled = enabled
        self.include_tracing = include_tracing
        self.exclude_paths = exclude_paths or [
            "/health",
            "/metrics",
            "/favicon.ico",
        ]

    def should_instrument(self, path: str) -> bool:
        """Check if a path should be instrumented."""
        if not self.enabled:
            return False

        for exclude in self.exclude_paths:
            if path.startswith(exclude):
                return False

        return True

    def wrap(
        self,
        func: Callable[..., Any],
        path: str,
        method: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Wrap a handler function with instrumentation.

        Args:
            func: Handler function to call
            path: Request path
            method: HTTP method
            *args, **kwargs: Arguments to pass to the handler

        Returns:
            Handler result
        """
        if not self.should_instrument(path):
            return func(*args, **kwargs)

        category = _categorize_endpoint(path)
        start_time = time.perf_counter()
        status = 200

        span = None
        if self.include_tracing:
            span = _safe_start_span(
                f"http.{method.lower()}.{category}",
                {"http.method": method, "http.path": path},
            )

        try:
            if hasattr(span, "__enter__"):
                span.__enter__()
            result = func(*args, **kwargs)

            # Extract status from result
            if isinstance(result, tuple) and len(result) >= 2:
                status = result[1] if isinstance(result[1], int) else 200
            elif hasattr(result, "status"):
                status = result.status

            return result
        except Exception as e:
            status = 500
            if span and hasattr(span, "record_exception"):
                span.record_exception(e)
            raise
        finally:
            latency = time.perf_counter() - start_time
            _safe_record_request(method, category, status, latency)

            if span and hasattr(span, "__exit__"):
                span.__exit__(None, None, None)


# Control Plane specific metrics recording functions
def record_control_plane_operation(
    operation: str,
    status: str,
    latency: Optional[float] = None,
) -> None:
    """Record a control plane operation metric.

    Args:
        operation: Operation type (agent_register, task_submit, deliberation_start, etc.)
        status: Operation status (success, failure, timeout)
        latency: Optional operation latency in seconds
    """
    try:
        from aragora.observability.metrics import (
            _init_metrics,
            REQUEST_COUNT,
        )

        _init_metrics()

        if REQUEST_COUNT is not None:
            REQUEST_COUNT.labels(
                method="control_plane",
                endpoint=operation,
                status=status,
            ).inc()
    except Exception as e:
        logger.debug(f"Failed to record control plane operation: {e}")


def record_agent_registration(
    agent_id: str,
    success: bool,
    latency: Optional[float] = None,
) -> None:
    """Record agent registration metric."""
    record_control_plane_operation(
        "agent_register",
        "success" if success else "failure",
        latency,
    )


def record_task_submission(
    task_type: str,
    success: bool,
    latency: Optional[float] = None,
) -> None:
    """Record task submission metric."""
    record_control_plane_operation(
        f"task_submit.{task_type}",
        "success" if success else "failure",
        latency,
    )


def record_deliberation_start(
    request_id: str,
    agent_count: int,
) -> None:
    """Record deliberation start metric."""
    record_control_plane_operation(
        "deliberation_start",
        "started",
    )


def record_deliberation_complete(
    request_id: str,
    success: bool,
    consensus_reached: bool,
    duration_seconds: float,
    sla_compliant: bool,
) -> None:
    """Record deliberation completion metric."""
    status = "consensus" if consensus_reached else "no_consensus"
    if not success:
        status = "failure"

    record_control_plane_operation(
        "deliberation_complete",
        status,
        duration_seconds,
    )

    # Record SLA compliance
    if sla_compliant:
        record_control_plane_operation("deliberation_sla", "compliant")
    else:
        record_control_plane_operation("deliberation_sla", "violated")


__all__ = [
    # Decorators
    "instrument_handler",
    "track_request",
    # Middleware
    "MetricsMiddleware",
    # Control plane recording
    "record_control_plane_operation",
    "record_agent_registration",
    "record_task_submission",
    "record_deliberation_start",
    "record_deliberation_complete",
    # Utilities
    "_categorize_endpoint",
]
