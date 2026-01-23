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
    reset_cross_subscriber_manager,
)
from .arena_bridge import (
    ArenaEventBridge,
    create_arena_bridge,
    EVENT_TYPE_MAP,
)
from .webhook_verify import (
    VerificationResult,
    generate_signature,
    verify_signature,
    verify_timestamp,
    verify_webhook_request,
    create_test_webhook_payload,
)
from .batch_dispatcher import (
    BatchWebhookDispatcher,
    BatchedEvent,
    EventBatch,
    get_batch_dispatcher,
    queue_batched_event,
    shutdown_batch_dispatcher,
)
from .async_dispatcher import (
    AsyncWebhookDispatcher,
    AsyncDeliveryResult,
    get_async_dispatcher,
    dispatch_webhook_async,
    dispatch_webhook_async_with_retry,
    shutdown_async_dispatcher,
)
from .security_events import (  # noqa: F401
    # Event types
    SecurityEventType,
    SecuritySeverity,
    SecurityFinding,
    SecurityEvent,
    # Emitter
    SecurityEventEmitter,
    SecurityEventHandler,
    get_security_emitter,
    set_security_emitter,
    # Debate integration
    trigger_security_debate,
    build_security_debate_question,
    get_security_debate_result,
    list_security_debates,
    # Convenience functions
    create_vulnerability_event,
    create_secret_event,
    create_scan_completed_event,
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
    "reset_cross_subscriber_manager",
    # Arena event bridge
    "ArenaEventBridge",
    "create_arena_bridge",
    "EVENT_TYPE_MAP",
    # Webhook verification
    "VerificationResult",
    "generate_signature",
    "verify_signature",
    "verify_timestamp",
    "verify_webhook_request",
    "create_test_webhook_payload",
    # Batch dispatcher
    "BatchWebhookDispatcher",
    "BatchedEvent",
    "EventBatch",
    "get_batch_dispatcher",
    "queue_batched_event",
    "shutdown_batch_dispatcher",
    # Async dispatcher
    "AsyncWebhookDispatcher",
    "AsyncDeliveryResult",
    "get_async_dispatcher",
    "dispatch_webhook_async",
    "dispatch_webhook_async_with_retry",
    "shutdown_async_dispatcher",
    # Security events
    "SecurityEventType",
    "SecuritySeverity",
    "SecurityFinding",
    "SecurityEvent",
    "SecurityEventEmitter",
    "SecurityEventHandler",
    "get_security_emitter",
    "set_security_emitter",
    "trigger_security_debate",
    "build_security_debate_question",
    "get_security_debate_result",
    "list_security_debates",
    "create_vulnerability_event",
    "create_secret_event",
    "create_scan_completed_event",
]
