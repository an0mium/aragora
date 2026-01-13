"""
Aragora Events Package.

Provides webhook delivery and event routing for external integrations.

Usage:
    from aragora.events import WebhookDispatcher, get_dispatcher

    # Get the global dispatcher
    dispatcher = get_dispatcher()

    # Subscribe to events
    dispatcher.subscribe_to_stream(event_emitter)

    # Or manually dispatch an event
    from aragora.events.dispatcher import dispatch_event
    dispatch_event("debate_end", {"debate_id": "...", "consensus": True})
"""

from .dispatcher import (
    WebhookDispatcher,
    dispatch_event,
    dispatch_webhook,
    get_dispatcher,
    shutdown_dispatcher,
)

__all__ = [
    "WebhookDispatcher",
    "get_dispatcher",
    "dispatch_event",
    "dispatch_webhook",
    "shutdown_dispatcher",
]
