"""
Aragora Webhooks Package.

Provides reliable webhook delivery infrastructure with:
- Persistent retry queues with exponential backoff
- Dead-letter handling for permanently failed deliveries
- Multiple storage backends (in-memory, Redis)
- Integration with the event dispatcher

Usage:
    from aragora.webhooks import (
        WebhookRetryQueue,
        WebhookDelivery,
        DeliveryStatus,
        get_retry_queue,
    )

    # Get global retry queue
    queue = get_retry_queue()
    await queue.start()

    # Enqueue a delivery
    delivery = WebhookDelivery(
        id="delivery-123",
        url="https://example.com/webhook",
        payload={"event": "debate_end", "data": {...}},
    )
    await queue.enqueue(delivery)

    # Stop when done
    await queue.stop()
"""

from aragora.webhooks.retry_queue import (
    # Enums
    DeliveryStatus,
    # Data classes
    WebhookDelivery,
    # Storage backends
    WebhookDeliveryStore,
    InMemoryDeliveryStore,
    RedisDeliveryStore,
    # Queue
    WebhookRetryQueue,
    # Factory functions
    get_retry_queue,
    set_retry_queue,
    reset_retry_queue,
    # Type aliases
    DeliveryCallback,
)

__all__ = [
    # Enums
    "DeliveryStatus",
    # Data classes
    "WebhookDelivery",
    # Storage backends
    "WebhookDeliveryStore",
    "InMemoryDeliveryStore",
    "RedisDeliveryStore",
    # Queue
    "WebhookRetryQueue",
    # Factory functions
    "get_retry_queue",
    "set_retry_queue",
    "reset_retry_queue",
    # Type aliases
    "DeliveryCallback",
]
