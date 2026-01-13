"""
Distributed Tracing Middleware.

Provides request tracing for observability:
- Generates unique trace IDs for each request
- Propagates trace IDs via X-Trace-ID header
- Integrates with structured logging
- Supports parent/child span relationships

Usage:
    from aragora.server.middleware.tracing import (
        TracingMiddleware,
        get_trace_id,
        get_span_id,
        trace_context,
    )

    # Get current trace ID
    trace_id = get_trace_id()

    # Create child span
    with trace_context(operation="debate.create") as span:
        # ... operation code ...
        span.set_tag("debate_id", debate_id)
"""

import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Callable, Dict, Generator, List, Optional

# Trace ID header names (W3C Trace Context compatible)
TRACE_ID_HEADER = "X-Trace-ID"
SPAN_ID_HEADER = "X-Span-ID"
PARENT_SPAN_HEADER = "X-Parent-Span-ID"

# W3C Trace Context header (if using OpenTelemetry format)
TRACEPARENT_HEADER = "traceparent"

# Context variables for trace propagation
_trace_id: ContextVar[Optional[str]] = ContextVar("trace_id", default=None)
_span_id: ContextVar[Optional[str]] = ContextVar("span_id", default=None)
_parent_span_id: ContextVar[Optional[str]] = ContextVar("parent_span_id", default=None)
_span_stack: ContextVar[List["Span"]] = ContextVar("span_stack", default=[])


def generate_trace_id() -> str:
    """Generate a unique trace ID.

    Uses UUID4 with hex encoding for 32-character trace IDs,
    compatible with most tracing systems.

    Returns:
        32-character hex trace ID
    """
    return uuid.uuid4().hex


def generate_span_id() -> str:
    """Generate a unique span ID.

    Uses UUID4 truncated to 16 characters for span IDs,
    following OpenTelemetry conventions.

    Returns:
        16-character hex span ID
    """
    return uuid.uuid4().hex[:16]


def get_trace_id() -> Optional[str]:
    """Get the current trace ID.

    Returns:
        Current trace ID or None if not in trace context
    """
    return _trace_id.get()


def get_span_id() -> Optional[str]:
    """Get the current span ID.

    Returns:
        Current span ID or None if not in trace context
    """
    return _span_id.get()


def get_parent_span_id() -> Optional[str]:
    """Get the parent span ID.

    Returns:
        Parent span ID or None if no parent
    """
    return _parent_span_id.get()


def set_trace_id(trace_id: str) -> None:
    """Set the current trace ID.

    Args:
        trace_id: The trace ID to set
    """
    _trace_id.set(trace_id)


def set_span_id(span_id: str) -> None:
    """Set the current span ID.

    Args:
        span_id: The span ID to set
    """
    _span_id.set(span_id)


@dataclass
class Span:
    """Represents a single operation within a trace.

    Tracks timing, tags, and events for observability.
    """

    trace_id: str
    span_id: str
    operation: str
    parent_span_id: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    tags: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "ok"
    error: Optional[str] = None

    def set_tag(self, key: str, value: Any) -> None:
        """Set a tag on the span.

        Args:
            key: Tag name
            value: Tag value
        """
        self.tags[key] = value

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """Add an event to the span.

        Args:
            name: Event name
            attributes: Optional event attributes
        """
        self.events.append(
            {
                "name": name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "attributes": attributes or {},
            }
        )

    def set_error(self, error: Exception) -> None:
        """Mark the span as errored.

        Args:
            error: The exception that occurred
        """
        self.status = "error"
        self.error = f"{type(error).__name__}: {str(error)}"
        self.add_event(
            "exception",
            {
                "type": type(error).__name__,
                "message": str(error),
            },
        )

    def finish(self) -> None:
        """Mark the span as finished."""
        self.end_time = time.time()

    @property
    def duration_ms(self) -> float:
        """Get span duration in milliseconds."""
        end = self.end_time or time.time()
        return (end - self.start_time) * 1000

    def to_dict(self) -> Dict[str, Any]:
        """Convert span to dictionary for logging/export."""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "operation": self.operation,
            "start_time": datetime.fromtimestamp(self.start_time, tz=timezone.utc).isoformat(),
            "end_time": (
                datetime.fromtimestamp(self.end_time, tz=timezone.utc).isoformat()
                if self.end_time
                else None
            ),
            "duration_ms": round(self.duration_ms, 2),
            "status": self.status,
            "error": self.error,
            "tags": self.tags,
            "events": self.events,
        }


@contextmanager
def trace_context(
    operation: str,
    trace_id: Optional[str] = None,
    parent_span_id: Optional[str] = None,
) -> Generator[Span, None, None]:
    """Context manager for creating a traced operation.

    Creates a new span for the operation and propagates trace context.

    Args:
        operation: Name of the operation being traced
        trace_id: Optional trace ID (uses current or generates new if not set)
        parent_span_id: Optional parent span ID for nested spans

    Yields:
        Span object for the current operation

    Example:
        with trace_context("debate.create") as span:
            span.set_tag("agents", ["claude", "gpt4"])
            debate = await create_debate(...)
            span.set_tag("debate_id", debate.id)
    """
    # Get or generate trace ID
    current_trace_id = trace_id or get_trace_id() or generate_trace_id()

    # Get parent span ID (current span becomes parent for this new span)
    current_span_id = get_span_id()
    actual_parent = parent_span_id or current_span_id

    # Generate new span ID
    new_span_id = generate_span_id()

    # Create span
    span = Span(
        trace_id=current_trace_id,
        span_id=new_span_id,
        operation=operation,
        parent_span_id=actual_parent,
    )

    # Push span to stack
    stack = _span_stack.get().copy()
    stack.append(span)

    # Set context
    old_trace = _trace_id.set(current_trace_id)
    old_span = _span_id.set(new_span_id)
    old_parent = _parent_span_id.set(actual_parent)
    old_stack = _span_stack.set(stack)

    try:
        yield span
    except Exception as e:
        span.set_error(e)
        raise
    finally:
        span.finish()

        # Pop span from stack
        stack = _span_stack.get().copy()
        if stack:
            stack.pop()

        # Restore context
        _trace_id.reset(old_trace)
        _span_id.reset(old_span)
        _parent_span_id.reset(old_parent)
        _span_stack.reset(old_stack)


def traced(operation: Optional[str] = None) -> Callable:
    """Decorator for tracing function execution.

    Args:
        operation: Operation name (defaults to function name)

    Returns:
        Decorator function

    Example:
        @traced("debate.create")
        async def create_debate(task: str) -> Debate:
            ...

        @traced()  # Uses function name as operation
        def process_message(msg: dict) -> None:
            ...
    """

    def decorator(func: Callable) -> Callable:
        op_name = operation or func.__name__

        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            with trace_context(op_name) as span:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    span.set_error(e)
                    raise

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            with trace_context(op_name) as span:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    span.set_error(e)
                    raise

        import asyncio

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper

    return decorator


class TracingMiddleware:
    """HTTP middleware for distributed tracing.

    Extracts or generates trace IDs and propagates them through the request.

    Usage:
        middleware = TracingMiddleware()

        # In request handling
        trace_id = middleware.extract_trace_id(request_headers)
        middleware.set_response_headers(response_headers, trace_id)
    """

    def __init__(self, service_name: str = "aragora"):
        """Initialize the tracing middleware.

        Args:
            service_name: Name of the service for span tagging
        """
        self.service_name = service_name

    def extract_trace_id(self, headers: Dict[str, str]) -> str:
        """Extract trace ID from request headers or generate new one.

        Supports multiple header formats:
        - X-Trace-ID (custom)
        - traceparent (W3C Trace Context)

        Args:
            headers: Request headers dictionary

        Returns:
            Trace ID (extracted or generated)
        """
        # Check custom header first
        trace_id = headers.get(TRACE_ID_HEADER) or headers.get(TRACE_ID_HEADER.lower())
        if trace_id:
            return trace_id

        # Check W3C traceparent header
        traceparent = headers.get(TRACEPARENT_HEADER) or headers.get(TRACEPARENT_HEADER.lower())
        if traceparent:
            # Format: version-trace_id-parent_id-flags
            parts = traceparent.split("-")
            if len(parts) >= 2:
                return parts[1]

        # Generate new trace ID
        return generate_trace_id()

    def extract_parent_span_id(self, headers: Dict[str, str]) -> Optional[str]:
        """Extract parent span ID from request headers.

        Args:
            headers: Request headers dictionary

        Returns:
            Parent span ID or None
        """
        parent_id = headers.get(PARENT_SPAN_HEADER) or headers.get(PARENT_SPAN_HEADER.lower())
        if parent_id:
            return parent_id

        # Check W3C traceparent header
        traceparent = headers.get(TRACEPARENT_HEADER) or headers.get(TRACEPARENT_HEADER.lower())
        if traceparent:
            parts = traceparent.split("-")
            if len(parts) >= 3:
                return parts[2]

        return None

    def set_response_headers(
        self,
        headers: Dict[str, str],
        trace_id: str,
        span_id: Optional[str] = None,
    ) -> None:
        """Add tracing headers to response.

        Args:
            headers: Response headers dictionary (modified in place)
            trace_id: Trace ID to include
            span_id: Optional span ID to include
        """
        headers[TRACE_ID_HEADER] = trace_id
        if span_id:
            headers[SPAN_ID_HEADER] = span_id

    def start_request_span(
        self,
        method: str,
        path: str,
        headers: Dict[str, str],
    ) -> Span:
        """Start a span for an incoming HTTP request.

        Args:
            method: HTTP method
            path: Request path
            headers: Request headers

        Returns:
            New span for the request
        """
        trace_id = self.extract_trace_id(headers)
        parent_span_id = self.extract_parent_span_id(headers)
        span_id = generate_span_id()

        # Set global context
        set_trace_id(trace_id)
        set_span_id(span_id)

        # Create span
        span = Span(
            trace_id=trace_id,
            span_id=span_id,
            operation=f"{method} {path}",
            parent_span_id=parent_span_id,
        )

        span.set_tag("http.method", method)
        span.set_tag("http.path", path)
        span.set_tag("service", self.service_name)

        return span

    def finish_request_span(
        self,
        span: Span,
        status_code: int,
        error: Optional[Exception] = None,
    ) -> None:
        """Finish a request span.

        Args:
            span: The span to finish
            status_code: HTTP response status code
            error: Optional exception if request failed
        """
        span.set_tag("http.status_code", status_code)

        if error:
            span.set_error(error)
        elif status_code >= 400:
            span.status = "error"
            span.error = f"HTTP {status_code}"

        span.finish()


# WebSocket tracing support


def trace_websocket_event(
    event_type: str,
    event_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Add tracing context to a WebSocket event.

    Args:
        event_type: Type of WebSocket event
        event_data: Event data dictionary

    Returns:
        Event data with tracing context added
    """
    data = event_data or {}

    # Add trace context
    trace_id = get_trace_id()
    span_id = get_span_id()

    if trace_id:
        data["_trace"] = {
            "trace_id": trace_id,
            "span_id": span_id,
        }

    return data


def extract_websocket_trace(event_data: Dict[str, Any]) -> Optional[str]:
    """Extract trace ID from WebSocket event data.

    Args:
        event_data: Event data dictionary

    Returns:
        Trace ID or None if not present
    """
    trace_info = event_data.get("_trace", {})
    return trace_info.get("trace_id")


# Error response tracing


def add_trace_to_error(error_response: Dict[str, Any]) -> Dict[str, Any]:
    """Add tracing context to error response.

    Args:
        error_response: Error response dictionary

    Returns:
        Error response with trace context
    """
    trace_id = get_trace_id()
    if trace_id:
        error_response["trace_id"] = trace_id
    return error_response


__all__ = [
    # Header constants
    "TRACE_ID_HEADER",
    "SPAN_ID_HEADER",
    "PARENT_SPAN_HEADER",
    # ID generators
    "generate_trace_id",
    "generate_span_id",
    # Context getters/setters
    "get_trace_id",
    "get_span_id",
    "get_parent_span_id",
    "set_trace_id",
    "set_span_id",
    # Span
    "Span",
    # Context manager
    "trace_context",
    # Decorator
    "traced",
    # Middleware
    "TracingMiddleware",
    # WebSocket support
    "trace_websocket_event",
    "extract_websocket_trace",
    # Error support
    "add_trace_to_error",
]
