"""
Trace context propagation for background task queues.

Bridges the request-scoped tracing context (``X-Trace-ID``, ``X-Span-ID``)
into background jobs so that a job can be correlated with the HTTP request
that enqueued it.

Trace context is serialised into the job payload on enqueue and restored
into :mod:`contextvars` on dequeue, ensuring that any spans created by
the worker are children of the original request span.

Usage — enqueue side::

    from aragora.queue.tracing import inject_trace_context

    payload = {"debate_id": "d-123", "task": "summarise"}
    payload = inject_trace_context(payload)   # adds _trace key
    await queue.enqueue(Job(payload=payload))

Usage — worker side::

    from aragora.queue.tracing import extract_and_activate, traced_job

    @traced_job("debate.execute")
    async def execute_job(job: Job) -> dict:
        # Trace context from the original request is now active
        ...

    # Or manually:
    ctx = extract_and_activate(job.payload)
    with trace_context("debate.execute"):
        await run_debate(job)
"""

from __future__ import annotations

import logging
from contextvars import copy_context
from dataclasses import dataclass
from functools import wraps
from typing import Any, Awaitable, Callable, TypeVar

from aragora.server.middleware.correlation import (
    CorrelationContext,
    get_or_create_correlation,
    init_correlation,
)
from aragora.server.middleware.tracing import (
    generate_span_id,
    get_span_id,
    get_trace_id,
    set_span_id,
    set_trace_id,
    trace_context,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Key used inside job payloads
_TRACE_KEY = "_trace"


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


@dataclass
class TraceCarrier:
    """Serialisable trace context that rides inside a job payload."""

    trace_id: str
    parent_span_id: str | None
    request_id: str | None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "trace_id": self.trace_id,
            "parent_span_id": self.parent_span_id,
            "request_id": self.request_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TraceCarrier":
        return cls(
            trace_id=data.get("trace_id", ""),
            parent_span_id=data.get("parent_span_id"),
            request_id=data.get("request_id"),
        )


# ---------------------------------------------------------------------------
# Inject (enqueue side)
# ---------------------------------------------------------------------------


def inject_trace_context(payload: dict[str, Any]) -> dict[str, Any]:
    """Inject current trace context into a job payload.

    Adds a ``_trace`` key to *payload* containing the trace ID and
    span ID from the current request context.  Safe to call even when
    there is no active trace — the key will simply be omitted.

    Args:
        payload: Mutable job payload dict.

    Returns:
        The same dict (mutated) for fluent usage.
    """
    trace_id = get_trace_id()
    if not trace_id:
        return payload

    correlation = get_or_create_correlation()
    carrier = TraceCarrier(
        trace_id=trace_id,
        parent_span_id=get_span_id(),
        request_id=correlation.request_id if correlation else None,
    )
    payload[_TRACE_KEY] = carrier.to_dict()
    return payload


def extract_trace_carrier(payload: dict[str, Any]) -> TraceCarrier | None:
    """Extract (but don't activate) the trace carrier from a job payload.

    Returns ``None`` if the payload has no trace context.
    """
    raw = payload.get(_TRACE_KEY)
    if not raw or not isinstance(raw, dict):
        return None
    try:
        return TraceCarrier.from_dict(raw)
    except Exception:
        logger.debug("Failed to parse trace carrier from payload", exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Extract + activate (worker side)
# ---------------------------------------------------------------------------


def extract_and_activate(payload: dict[str, Any]) -> CorrelationContext | None:
    """Extract trace context from a job payload and activate it.

    Sets the :mod:`contextvars` so that subsequent calls to
    ``get_trace_id()``, ``get_span_id()``, etc. return the original
    request's trace context with a **new child span** for the worker.

    Args:
        payload: Job payload dict.

    Returns:
        Activated :class:`CorrelationContext`, or ``None`` if the
        payload had no trace context.
    """
    carrier = extract_trace_carrier(payload)
    if carrier is None:
        return None

    # Create a new child span so the worker work appears as a child
    # of the enqueuing span.
    child_span = generate_span_id()

    set_trace_id(carrier.trace_id)
    set_span_id(child_span)

    ctx = init_correlation(
        request_id=carrier.request_id,
        trace_id=carrier.trace_id,
        span_id=child_span,
        parent_span_id=carrier.parent_span_id,
    )

    logger.debug(
        "Activated trace context for background job: trace=%s parent=%s span=%s",
        carrier.trace_id,
        carrier.parent_span_id,
        child_span,
    )
    return ctx


# ---------------------------------------------------------------------------
# Decorator for traced jobs
# ---------------------------------------------------------------------------


def traced_job(
    operation: str | None = None,
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    """Decorator that activates trace context and wraps the job in a span.

    Usage::

        @traced_job("debate.execute")
        async def execute_debate(job: Job) -> dict:
            ...  # runs inside a span linked to the original request

    Args:
        operation: Span name.  Defaults to the function name.
    """

    def decorator(fn: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        op = operation or fn.__name__

        @wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            # Try to find the job payload in positional args
            payload: dict[str, Any] | None = None
            for arg in args:
                if hasattr(arg, "payload") and isinstance(arg.payload, dict):
                    payload = arg.payload
                    break
                if isinstance(arg, dict) and _TRACE_KEY in arg:
                    payload = arg
                    break

            if payload is not None:
                extract_and_activate(payload)

            with trace_context(op) as span:
                span.set_tag("job.traced", True)
                try:
                    return await fn(*args, **kwargs)
                except Exception as exc:
                    span.set_error(exc)
                    raise

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

_injected = 0
_extracted = 0


def get_tracing_stats() -> dict[str, int]:
    """Return inject/extract counters (for monitoring)."""
    return {"injected": _injected, "extracted": _extracted}


__all__ = [
    "TraceCarrier",
    "extract_and_activate",
    "extract_trace_carrier",
    "get_tracing_stats",
    "inject_trace_context",
    "traced_job",
]
