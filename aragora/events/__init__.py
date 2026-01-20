"""
Aragora Events Package.

Provides event types, webhook delivery, and event routing for external integrations.

The events layer is a shared dependency that can be imported by any layer
(CLI, debate, memory, server) without creating circular dependencies.

Usage:
    # Import event types (available to all layers)
    from aragora.events import StreamEvent, StreamEventType, EventEmitter

    # Import webhook dispatcher
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
from .types import (
    AudienceMessage,
    EventEmitter,
    StreamEvent,
    StreamEventType,
)
from .cross_subscribers import (
    CrossSubscriberManager,
    SubscriberStats,
    get_cross_subscriber_manager,
)

__all__ = [
    # Event types (shared layer)
    "StreamEventType",
    "StreamEvent",
    "AudienceMessage",
    "EventEmitter",
    # Webhook dispatcher
    "WebhookDispatcher",
    "get_dispatcher",
    "dispatch_event",
    "dispatch_webhook",
    "shutdown_dispatcher",
    # Cross-subsystem subscribers
    "CrossSubscriberManager",
    "SubscriberStats",
    "get_cross_subscriber_manager",
]
