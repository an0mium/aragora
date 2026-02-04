"""
Handler Event Emission Helpers.

Provides a lightweight, fail-safe interface for emitting events from HTTP handlers.
Uses the existing webhook dispatcher for delivery.

Usage:
    from aragora.events.handler_events import emit_handler_event

    # In any handler method:
    emit_handler_event("debates", "created", {"debate_id": debate_id, "question": question})
"""

from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

# Standard event action constants
CREATED = "created"
UPDATED = "updated"
DELETED = "deleted"
COMPLETED = "completed"
FAILED = "failed"
STARTED = "started"
APPROVED = "approved"
REJECTED = "rejected"
QUERIED = "queried"

# Track whether dispatcher is available
_dispatcher_available: bool | None = None


def _get_dispatch_fn():
    """Lazy-load the dispatch function to avoid circular imports."""
    global _dispatcher_available
    if _dispatcher_available is False:
        return None
    try:
        from aragora.events.dispatcher import dispatch_event

        _dispatcher_available = True
        return dispatch_event
    except ImportError:
        _dispatcher_available = False
        logger.debug("Event dispatcher not available — handler events will be silenced")
        return None


def emit_handler_event(
    handler: str,
    action: str,
    payload: dict[str, Any] | None = None,
    *,
    user_id: str | None = None,
    resource_id: str | None = None,
) -> None:
    """Emit an event from a handler.

    This is a fire-and-forget, non-blocking call. If the dispatcher
    is unavailable, the event is silently dropped.

    Args:
        handler: Handler name (e.g., "debates", "knowledge", "agents")
        action: Action performed (e.g., "created", "updated", "deleted")
        payload: Event-specific data dict
        user_id: Optional user who triggered the action
        resource_id: Optional resource ID affected
    """
    dispatch_fn = _get_dispatch_fn()
    if dispatch_fn is None:
        return

    event_type = f"{handler}.{action}"
    data: dict[str, Any] = {
        "handler": handler,
        "action": action,
        "timestamp": time.time(),
    }

    if payload:
        data.update(payload)
    if user_id:
        data["user_id"] = user_id
    if resource_id:
        data["resource_id"] = resource_id

    # Add trace context if available
    try:
        from aragora.server.middleware.tracing import get_trace_id

        trace_id = get_trace_id()
        if trace_id:
            data["trace_id"] = trace_id
    except ImportError:
        pass

    try:
        dispatch_fn(event_type, data)
    except Exception as e:
        # Never let event emission break handler logic
        logger.debug("Failed to emit handler event %s: %s", event_type, e)
