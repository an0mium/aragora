"""
OpenTelemetry integration for Agent Fabric tracing.

Provides distributed tracing capabilities with W3C Trace Context propagation,
correlation ID management, and graceful degradation when OpenTelemetry is not installed.
"""

from __future__ import annotations

import functools
import logging
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Generator, TypeVar
import uuid

logger = logging.getLogger(__name__)

# Type variables for generic decorator
F = TypeVar("F", bound=Callable[..., Any])

# Try to import OpenTelemetry - graceful degradation if not installed
_OTEL_AVAILABLE = False
_tracer = None

try:
    from opentelemetry import trace
    from opentelemetry.context import Context
    from opentelemetry.trace import SpanKind, Status, StatusCode, Tracer
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

    _OTEL_AVAILABLE = True
    _propagator = TraceContextTextMapPropagator()
except ImportError:
    trace = None  # type: ignore[assignment]
    Context = None  # type: ignore[assignment, misc]
    SpanKind = None  # type: ignore[assignment, misc]
    Status = None  # type: ignore[assignment, misc]
    StatusCode = None  # type: ignore[assignment, misc]
    Tracer = None  # type: ignore[assignment, misc]
    _propagator = None


@dataclass
class TraceContext:
    """Trace context for correlation and distributed tracing."""

    trace_id: str
    span_id: str
    correlation_id: str
    parent_span_id: str | None = None
    sampled: bool = True
    baggage: dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_w3c_traceparent(self) -> str:
        """Generate W3C traceparent header value."""
        flags = "01" if self.sampled else "00"
        return f"00-{self.trace_id}-{self.span_id}-{flags}"

    @classmethod
    def from_w3c_traceparent(
        cls, traceparent: str, correlation_id: str | None = None
    ) -> TraceContext | None:
        """Parse W3C traceparent header value."""
        try:
            parts = traceparent.split("-")
            if len(parts) != 4 or parts[0] != "00":
                return None

            return cls(
                trace_id=parts[1],
                span_id=parts[2],
                correlation_id=correlation_id or str(uuid.uuid4()),
                sampled=parts[3] == "01",
            )
        except (ValueError, IndexError):
            return None


# Context variables for trace propagation
_trace_context: ContextVar[TraceContext | None] = ContextVar("trace_context", default=None)
_correlation_id: ContextVar[str | None] = ContextVar("correlation_id", default=None)


def _generate_id(length: int = 32) -> str:
    """Generate a random hex ID of specified length."""
    return uuid.uuid4().hex[:length]


def get_current_trace_id() -> str | None:
    """
    Get the current trace ID from context.

    Returns:
        The current trace ID or None if no trace is active.
    """
    if _OTEL_AVAILABLE and trace is not None:
        span = trace.get_current_span()
        if span and span.is_recording():
            ctx = span.get_span_context()
            if ctx.is_valid:
                return format(ctx.trace_id, "032x")

    # Fall back to our context var
    ctx = _trace_context.get()
    return ctx.trace_id if ctx else None


def get_current_span_id() -> str | None:
    """
    Get the current span ID from context.

    Returns:
        The current span ID or None if no span is active.
    """
    if _OTEL_AVAILABLE and trace is not None:
        span = trace.get_current_span()
        if span and span.is_recording():
            ctx = span.get_span_context()
            if ctx.is_valid:
                return format(ctx.span_id, "016x")

    # Fall back to our context var
    ctx = _trace_context.get()
    return ctx.span_id if ctx else None


def get_correlation_id() -> str:
    """
    Get or create a correlation ID for the current context.

    This ID is used to correlate logs, traces, and other telemetry
    across service boundaries.

    Returns:
        The current correlation ID or a newly generated one.
    """
    # Check context var first
    corr_id = _correlation_id.get()
    if corr_id:
        return corr_id

    # Check trace context
    ctx = _trace_context.get()
    if ctx:
        return ctx.correlation_id

    # Generate new correlation ID
    new_id = str(uuid.uuid4())
    _correlation_id.set(new_id)
    return new_id


def set_correlation_id(correlation_id: str) -> None:
    """
    Set the correlation ID for the current context.

    Args:
        correlation_id: The correlation ID to set.
    """
    _correlation_id.set(correlation_id)


@contextmanager
def with_correlation_id(correlation_id: str) -> Generator[str, None, None]:
    """
    Context manager to set correlation ID for a block of code.

    Args:
        correlation_id: The correlation ID to use.

    Yields:
        The correlation ID.

    Example:
        with with_correlation_id("request-123"):
            # All code here will have correlation_id = "request-123"
            do_something()
    """
    old_id = _correlation_id.get()
    _correlation_id.set(correlation_id)
    try:
        yield correlation_id
    finally:
        _correlation_id.set(old_id)


@contextmanager
def with_trace_context(ctx: TraceContext) -> Generator[TraceContext, None, None]:
    """
    Context manager to set trace context for a block of code.

    Args:
        ctx: The trace context to use.

    Yields:
        The trace context.
    """
    old_ctx = _trace_context.get()
    old_corr_id = _correlation_id.get()
    _trace_context.set(ctx)
    _correlation_id.set(ctx.correlation_id)
    try:
        yield ctx
    finally:
        _trace_context.set(old_ctx)
        _correlation_id.set(old_corr_id)


def create_trace_context(
    correlation_id: str | None = None,
    parent_context: TraceContext | None = None,
) -> TraceContext:
    """
    Create a new trace context.

    Args:
        correlation_id: Optional correlation ID (generated if not provided).
        parent_context: Optional parent context for trace hierarchy.

    Returns:
        A new TraceContext instance.
    """
    if parent_context:
        return TraceContext(
            trace_id=parent_context.trace_id,
            span_id=_generate_id(16),
            correlation_id=parent_context.correlation_id,
            parent_span_id=parent_context.span_id,
            sampled=parent_context.sampled,
        )

    return TraceContext(
        trace_id=_generate_id(32),
        span_id=_generate_id(16),
        correlation_id=correlation_id or str(uuid.uuid4()),
    )


def extract_trace_context(headers: dict[str, str]) -> TraceContext | None:
    """
    Extract trace context from W3C Trace Context headers.

    Args:
        headers: Dictionary containing HTTP headers.

    Returns:
        Extracted TraceContext or None if not present/invalid.
    """
    traceparent = headers.get("traceparent")
    if not traceparent:
        return None

    correlation_id = headers.get("x-correlation-id") or headers.get("x-request-id")
    return TraceContext.from_w3c_traceparent(traceparent, correlation_id)


def inject_trace_context(
    headers: dict[str, str], ctx: TraceContext | None = None
) -> dict[str, str]:
    """
    Inject trace context into headers for W3C Trace Context propagation.

    Args:
        headers: Dictionary to inject headers into.
        ctx: Optional TraceContext (uses current context if not provided).

    Returns:
        The headers dictionary with trace context added.
    """
    if ctx is None:
        ctx = _trace_context.get()

    if ctx:
        headers["traceparent"] = ctx.to_w3c_traceparent()
        headers["x-correlation-id"] = ctx.correlation_id

    # Also use OpenTelemetry propagation if available
    if _OTEL_AVAILABLE and _propagator is not None:
        _propagator.inject(headers)

    return headers


def _get_tracer(tracer_name: str = "aragora.fabric") -> Any:
    """Get or create a tracer instance."""
    global _tracer

    if not _OTEL_AVAILABLE:
        return None

    if _tracer is None and trace is not None:
        _tracer = trace.get_tracer(tracer_name)

    return _tracer


class CorrelationContext:
    """
    Context holder for correlation and tracing information.

    This class provides a convenient way to manage correlation IDs
    and trace context across async boundaries.
    """

    def __init__(
        self,
        correlation_id: str | None = None,
        trace_context: TraceContext | None = None,
    ) -> None:
        """
        Initialize correlation context.

        Args:
            correlation_id: Optional correlation ID.
            trace_context: Optional trace context.
        """
        self._correlation_id = correlation_id or str(uuid.uuid4())
        self._trace_context = trace_context or create_trace_context(self._correlation_id)

    @property
    def correlation_id(self) -> str:
        """Get the correlation ID."""
        return self._correlation_id

    @property
    def trace_context(self) -> TraceContext:
        """Get the trace context."""
        return self._trace_context

    @property
    def trace_id(self) -> str:
        """Get the trace ID."""
        return self._trace_context.trace_id

    @property
    def span_id(self) -> str:
        """Get the current span ID."""
        return self._trace_context.span_id

    def to_log_context(self) -> dict[str, str]:
        """
        Get context as a dictionary suitable for log extra fields.

        Returns:
            Dictionary with correlation_id, trace_id, and span_id.
        """
        return {
            "correlation_id": self._correlation_id,
            "trace_id": self._trace_context.trace_id,
            "span_id": self._trace_context.span_id,
        }

    def to_headers(self) -> dict[str, str]:
        """
        Get context as HTTP headers for propagation.

        Returns:
            Dictionary with traceparent and correlation headers.
        """
        return inject_trace_context({}, self._trace_context)

    @classmethod
    def from_headers(cls, headers: dict[str, str]) -> CorrelationContext:
        """
        Create CorrelationContext from HTTP headers.

        Args:
            headers: Dictionary containing HTTP headers.

        Returns:
            New CorrelationContext instance.
        """
        trace_ctx = extract_trace_context(headers)
        correlation_id = headers.get("x-correlation-id") or headers.get("x-request-id")

        if trace_ctx:
            return cls(
                correlation_id=trace_ctx.correlation_id,
                trace_context=trace_ctx,
            )

        return cls(correlation_id=correlation_id)

    def __enter__(self) -> CorrelationContext:
        """Enter context manager."""
        self._token = _trace_context.set(self._trace_context)
        self._corr_token = _correlation_id.set(self._correlation_id)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager."""
        _trace_context.set(None)
        _correlation_id.set(None)


def trace_span(
    name: str | None = None,
    attributes: dict[str, str] | None = None,
    record_exception: bool = True,
    set_status_on_exception: bool = True,
) -> Callable[[F], F]:
    """
    Decorator to wrap a function with a tracing span.

    Works with both sync and async functions. When OpenTelemetry is not installed,
    this decorator gracefully degrades to a no-op while still managing
    correlation context.

    Args:
        name: Span name (defaults to function name).
        attributes: Optional span attributes.
        record_exception: Whether to record exceptions in the span.
        set_status_on_exception: Whether to set error status on exception.

    Returns:
        Decorated function.

    Example:
        @trace_span("process_request", attributes={"service": "fabric"})
        async def process_request(request_id: str) -> dict:
            return {"status": "ok"}
    """

    def decorator(func: F) -> F:
        span_name = name or func.__name__
        span_attrs = attributes or {}

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            # Add correlation_id to log context
            corr_id = get_correlation_id()

            if _OTEL_AVAILABLE and trace is not None:
                tracer = _get_tracer()
                if tracer is not None:
                    with tracer.start_as_current_span(
                        span_name,
                        kind=SpanKind.INTERNAL,
                    ) as span:
                        # Set attributes
                        span.set_attribute("correlation_id", corr_id)
                        for key, value in span_attrs.items():
                            span.set_attribute(key, value)

                        try:
                            result = await func(*args, **kwargs)
                            return result
                        except Exception as e:
                            if record_exception:
                                span.record_exception(e)
                            if set_status_on_exception:
                                span.set_status(Status(StatusCode.ERROR, str(e)))
                            raise

            # Fallback: no OpenTelemetry, but still manage trace context
            current_ctx = _trace_context.get()
            if current_ctx is None:
                new_ctx = create_trace_context(corr_id)
                with with_trace_context(new_ctx):
                    return await func(*args, **kwargs)

            return await func(*args, **kwargs)

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            # Add correlation_id to log context
            corr_id = get_correlation_id()

            if _OTEL_AVAILABLE and trace is not None:
                tracer = _get_tracer()
                if tracer is not None:
                    with tracer.start_as_current_span(
                        span_name,
                        kind=SpanKind.INTERNAL,
                    ) as span:
                        # Set attributes
                        span.set_attribute("correlation_id", corr_id)
                        for key, value in span_attrs.items():
                            span.set_attribute(key, value)

                        try:
                            result = func(*args, **kwargs)
                            return result
                        except Exception as e:
                            if record_exception:
                                span.record_exception(e)
                            if set_status_on_exception:
                                span.set_status(Status(StatusCode.ERROR, str(e)))
                            raise

            # Fallback: no OpenTelemetry, but still manage trace context
            current_ctx = _trace_context.get()
            if current_ctx is None:
                new_ctx = create_trace_context(corr_id)
                with with_trace_context(new_ctx):
                    return func(*args, **kwargs)

            return func(*args, **kwargs)

        # Return appropriate wrapper based on function type
        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore[return-value]
        return sync_wrapper  # type: ignore[return-value]

    return decorator


class CorrelationLogFilter(logging.Filter):
    """
    Logging filter that adds correlation context to log records.

    This filter adds correlation_id, trace_id, and span_id to all log records,
    making it easy to correlate logs with traces.

    Example:
        handler = logging.StreamHandler()
        handler.addFilter(CorrelationLogFilter())
        logger.addHandler(handler)
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """Add correlation context to log record."""
        record.correlation_id = get_correlation_id()  # type: ignore[attr-defined]
        record.trace_id = get_current_trace_id() or ""  # type: ignore[attr-defined]
        record.span_id = get_current_span_id() or ""  # type: ignore[attr-defined]
        return True


def configure_logging_with_correlation(
    log_format: str | None = None,
    level: int = logging.INFO,
) -> None:
    """
    Configure logging with correlation context.

    This sets up the root logger with a format that includes
    correlation_id, trace_id, and span_id.

    Args:
        log_format: Optional custom log format (must include correlation fields).
        level: Logging level.
    """
    if log_format is None:
        log_format = (
            "%(asctime)s - %(name)s - %(levelname)s - "
            "[correlation_id=%(correlation_id)s trace_id=%(trace_id)s span_id=%(span_id)s] - "
            "%(message)s"
        )

    handler = logging.StreamHandler()
    handler.addFilter(CorrelationLogFilter())
    handler.setFormatter(logging.Formatter(log_format))

    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(level)


def is_otel_available() -> bool:
    """Check if OpenTelemetry is available."""
    return _OTEL_AVAILABLE
