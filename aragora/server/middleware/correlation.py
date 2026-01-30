"""
Request Correlation Middleware.

Unifies the Request ID (X-Request-ID) and Trace ID (X-Trace-ID) systems
so that logs, metrics, and traces can be correlated across the full
request lifecycle.

Before this module, request_logging.py and tracing.py maintained separate
context variables with no cross-reference. This module:

1. Provides a single ``CorrelationContext`` holding both IDs.
2. Exposes a logging ``Filter`` that injects request_id, trace_id, and
   span_id into every log record automatically.
3. Adds helper to build the W3C ``traceparent`` header that embeds the
   correlation context for outgoing calls.

Usage:
    from aragora.server.middleware.correlation import (
        CorrelationContext,
        init_correlation,
        get_correlation,
        CorrelationLogFilter,
    )

    # At request entry point
    ctx = init_correlation(request_headers)

    # In application code
    ctx = get_correlation()
    logger.info("Processing debate", extra={"debate_id": "d-123"})
    # Log record automatically includes request_id, trace_id, span_id

    # For outgoing HTTP calls
    headers = ctx.as_headers()
"""

from __future__ import annotations

import logging
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from aragora.server.middleware.request_logging import (
    REQUEST_ID_HEADER,
    generate_request_id,
    set_current_request_id,
)
from aragora.server.middleware.tracing import (
    PARENT_SPAN_HEADER,
    SPAN_ID_HEADER,
    TRACE_ID_HEADER,
    TRACEPARENT_HEADER,
    generate_span_id,
    generate_trace_id,
    set_span_id,
    set_trace_id,
)

logger = logging.getLogger(__name__)

# Single context variable holding the correlated IDs
_correlation: ContextVar["CorrelationContext | None"] = ContextVar(
    "correlation_context", default=None
)


@dataclass
class CorrelationContext:
    """Unified correlation context binding request and trace identifiers.

    Attributes:
        request_id: Short human-friendly identifier (``req-...``).
        trace_id: 32-hex-char distributed trace identifier.
        span_id: 16-hex-char span identifier for the current unit of work.
        parent_span_id: Span that spawned this one (if any).
    """

    request_id: str
    trace_id: str
    span_id: str
    parent_span_id: str | None = None

    # Extra fields that downstream code can set for richer correlation
    extras: dict[str, str] = field(default_factory=dict)

    def as_headers(self) -> dict[str, str]:
        """Return HTTP headers for propagating correlation to outgoing calls."""
        headers: dict[str, str] = {
            REQUEST_ID_HEADER: self.request_id,
            TRACE_ID_HEADER: self.trace_id,
            SPAN_ID_HEADER: self.span_id,
            # W3C traceparent
            TRACEPARENT_HEADER: (
                f"00-{self.trace_id:0>32}-{self.span_id:0>16}-01"
            ),
        }
        if self.parent_span_id:
            headers[PARENT_SPAN_HEADER] = self.parent_span_id
        return headers

    def as_log_dict(self) -> dict[str, str]:
        """Return a flat dict suitable for structured log extra fields."""
        d = {
            "request_id": self.request_id,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
        }
        if self.parent_span_id:
            d["parent_span_id"] = self.parent_span_id
        d.update(self.extras)
        return d


# ---------------------------------------------------------------------------
# Context helpers
# ---------------------------------------------------------------------------


def init_correlation(
    headers: dict[str, str] | None = None,
    *,
    request_id: str | None = None,
    trace_id: str | None = None,
    span_id: str | None = None,
    parent_span_id: str | None = None,
) -> CorrelationContext:
    """Initialise correlation context from incoming request headers.

    Extracts ``X-Request-ID``, ``X-Trace-ID``, ``X-Span-ID`` and
    ``traceparent`` from *headers*, falling back to generating new IDs.

    Also sets the legacy per-system context variables so existing code
    that calls ``get_trace_id()`` or ``get_current_request_id()`` still
    works.

    Args:
        headers: Incoming HTTP headers (optional).
        request_id: Explicit request ID override.
        trace_id: Explicit trace ID override.
        span_id: Explicit span ID override.
        parent_span_id: Explicit parent span ID override.

    Returns:
        Populated ``CorrelationContext``.
    """
    hdrs = headers or {}

    # --- Request ID ---
    rid = request_id or hdrs.get(REQUEST_ID_HEADER) or hdrs.get(
        REQUEST_ID_HEADER.lower()
    ) or generate_request_id()

    # --- Trace ID ---
    tid = trace_id
    psid = parent_span_id

    if not tid:
        tid = hdrs.get(TRACE_ID_HEADER) or hdrs.get(TRACE_ID_HEADER.lower())

    if not tid:
        traceparent = hdrs.get(TRACEPARENT_HEADER) or hdrs.get(
            TRACEPARENT_HEADER.lower()
        )
        if traceparent:
            parts = traceparent.split("-")
            if len(parts) >= 2:
                tid = parts[1]
            if len(parts) >= 3 and not psid:
                psid = parts[2]

    if not tid:
        tid = generate_trace_id()

    # --- Span ID ---
    sid = span_id or generate_span_id()

    if not psid:
        psid = hdrs.get(PARENT_SPAN_HEADER) or hdrs.get(
            PARENT_SPAN_HEADER.lower()
        )

    ctx = CorrelationContext(
        request_id=rid,
        trace_id=tid,
        span_id=sid,
        parent_span_id=psid,
    )

    # Store in context var
    _correlation.set(ctx)

    # Back-populate legacy context vars so existing callers keep working
    set_current_request_id(rid)
    set_trace_id(tid)
    set_span_id(sid)

    return ctx


def get_correlation() -> CorrelationContext | None:
    """Get the current correlation context.

    Returns:
        Active ``CorrelationContext`` or ``None``.
    """
    return _correlation.get()


def get_or_create_correlation() -> CorrelationContext:
    """Get or create a correlation context.

    If there is no active context (e.g. background task), a new one is
    created with fresh IDs.

    Returns:
        Active or new ``CorrelationContext``.
    """
    ctx = _correlation.get()
    if ctx is None:
        ctx = init_correlation()
    return ctx


# ---------------------------------------------------------------------------
# Logging filter
# ---------------------------------------------------------------------------


class CorrelationLogFilter(logging.Filter):
    """Logging filter that injects correlation IDs into every log record.

    Adds ``request_id``, ``trace_id``, and ``span_id`` attributes to
    each :class:`logging.LogRecord` so they appear in structured log
    output.

    Usage::

        handler = logging.StreamHandler()
        handler.addFilter(CorrelationLogFilter())
        logging.getLogger().addHandler(handler)

    Or with a dict-config formatter that references ``%(trace_id)s``.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        ctx = _correlation.get()
        if ctx is not None:
            setattr(record, "request_id", ctx.request_id)
            setattr(record, "trace_id", ctx.trace_id)
            setattr(record, "span_id", ctx.span_id)
        else:
            setattr(record, "request_id", "")
            setattr(record, "trace_id", "")
            setattr(record, "span_id", "")
        return True


# ---------------------------------------------------------------------------
# Convenience: structured log with correlation
# ---------------------------------------------------------------------------


def correlation_log_extra() -> dict[str, Any]:
    """Return ``extra`` dict suitable for ``logger.info(..., extra=...)``.

    Example::

        logger.info("Starting debate", extra=correlation_log_extra())
    """
    ctx = _correlation.get()
    if ctx is not None:
        return ctx.as_log_dict()
    return {}


__all__ = [
    "CorrelationContext",
    "CorrelationLogFilter",
    "correlation_log_extra",
    "get_correlation",
    "get_or_create_correlation",
    "init_correlation",
]
