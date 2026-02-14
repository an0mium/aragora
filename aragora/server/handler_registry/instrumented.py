"""Automatic handler instrumentation for tracing and metrics.

Provides decorators that wrap handler methods with automatic tracing
and latency recording at registration time.  This avoids manually
instrumenting 250+ handlers: instead, ``auto_instrument_handler`` is
called once during ``_init_handlers`` to decorate every handle method
on every registered handler class.

The module gracefully degrades when the observability subsystem is not
installed or configured -- handlers continue to work without any
performance overhead beyond a single boolean check.
"""

from __future__ import annotations

import logging
import time
from functools import wraps
from typing import Any
from collections.abc import Callable

logger = logging.getLogger(__name__)

# Lazy-initialised import cache ------------------------------------------------

_tracing_available: bool | None = None


def _get_tracing() -> tuple[Any, Any]:
    """Return (``track_request``, ``record_request``) or ``(None, None)``.

    The first successful or failed import is cached so subsequent calls
    are essentially free.
    """
    global _tracing_available

    if _tracing_available is False:
        return None, None

    try:
        from aragora.observability.handler_instrumentation import (
            _safe_record_request,
            _safe_start_span,
        )

        _tracing_available = True
        return _safe_start_span, _safe_record_request
    except ImportError:
        _tracing_available = False
        return None, None


# ---------------------------------------------------------------------------
# Per-method decorator
# ---------------------------------------------------------------------------


def instrumented_handler(handler_name: str, method_name: str = "handle") -> Callable:
    """Decorator that adds automatic tracing and metrics to a handler method.

    Args:
        handler_name: Human-readable handler class name used in span/metric labels.
        method_name: The method being wrapped (e.g. ``"handle"``, ``"handle_post"``).

    Returns:
        A decorator that wraps the target callable.
    """

    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            safe_start_span, safe_record_request = _get_tracing()
            start = time.monotonic()
            status_code = 500  # default to error
            try:
                result = fn(*args, **kwargs)
                # Extract status_code from the result when possible.
                # HandlerResult is typically (status_code, headers, body)
                # or a dataclass with a ``status_code`` attribute.
                if result is not None:
                    if isinstance(result, tuple) and len(result) >= 1:
                        status_code = result[0]
                    elif hasattr(result, "status_code"):
                        status_code = result.status_code
                    else:
                        status_code = 200
                else:
                    status_code = 200
                return result
            except Exception:  # noqa: BLE001 - Intentional: set status_code for metrics before re-raising
                status_code = 500
                raise
            finally:
                duration = time.monotonic() - start
                span_name = f"handler.{handler_name}.{method_name}"
                if safe_record_request is not None:
                    try:
                        safe_record_request(
                            method_name.replace("handle_", "").upper() or "GET",
                            handler_name,
                            status_code,
                            duration,
                        )
                    except (TypeError, AttributeError, RuntimeError):
                        logger.debug("Failed to record request metrics", exc_info=True)
                if safe_start_span is not None:
                    try:
                        span = safe_start_span(
                            span_name,
                            {
                                "handler.name": handler_name,
                                "handler.method": method_name,
                                "http.status_code": status_code,
                                "handler.duration_ms": round(duration * 1000, 2),
                            },
                        )
                        # Immediately close the span -- we only record it.
                        if hasattr(span, "__enter__"):
                            span.__enter__()
                        if hasattr(span, "__exit__"):
                            span.__exit__(None, None, None)
                    except (TypeError, AttributeError, RuntimeError):
                        logger.debug("Failed to record span", exc_info=True)

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Class-level auto-instrumentation
# ---------------------------------------------------------------------------

_INSTRUMENTABLE_METHODS = (
    "handle",
    "handle_post",
    "handle_put",
    "handle_delete",
    "handle_patch",
)


def auto_instrument_handler(handler_instance: Any) -> Any:
    """Auto-instrument all handle methods on a handler *instance*.

    Called during handler registration (``_init_handlers``) so that
    every handler is transparently wrapped with tracing and metrics.

    Args:
        handler_instance: An instantiated handler object.

    Returns:
        The same handler instance with its handle methods wrapped.
    """
    handler_name = handler_instance.__class__.__name__

    for method_name in _INSTRUMENTABLE_METHODS:
        original = getattr(handler_instance, method_name, None)
        if original is not None and callable(original):
            decorated = instrumented_handler(handler_name, method_name)(original)
            try:
                setattr(handler_instance, method_name, decorated)
            except (AttributeError, TypeError):
                # Some objects disallow setattr -- skip silently.
                logger.debug(f"[instrumented] Could not instrument {handler_name}.{method_name}")

    return handler_instance
