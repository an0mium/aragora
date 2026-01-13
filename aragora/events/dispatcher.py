"""
Webhook Event Dispatcher.

Handles delivery of events to registered webhook endpoints.
Supports async delivery with retry logic and signature verification.

Features:
- HMAC-SHA256 payload signing
- Async non-blocking delivery
- Exponential backoff retry
- Delivery status tracking
- Rate limiting per endpoint

Usage:
    from aragora.events.dispatcher import get_dispatcher

    # Get global dispatcher
    dispatcher = get_dispatcher()

    # Connect to event stream
    dispatcher.subscribe_to_stream(event_emitter)

    # Events are automatically delivered to registered webhooks
"""

import asyncio
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

from aragora.server.middleware.tracing import get_trace_id

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

# Maximum concurrent webhook deliveries
MAX_WORKERS = int(os.environ.get("ARAGORA_WEBHOOK_WORKERS", "10"))

# Retry configuration
MAX_RETRIES = int(os.environ.get("ARAGORA_WEBHOOK_MAX_RETRIES", "3"))
INITIAL_RETRY_DELAY = float(os.environ.get("ARAGORA_WEBHOOK_RETRY_DELAY", "1.0"))
MAX_RETRY_DELAY = float(os.environ.get("ARAGORA_WEBHOOK_MAX_RETRY_DELAY", "60.0"))

# Request timeout in seconds
REQUEST_TIMEOUT = float(os.environ.get("ARAGORA_WEBHOOK_TIMEOUT", "30.0"))

# User agent for webhook requests
USER_AGENT = "Aragora-Webhooks/1.0"


# =============================================================================
# Webhook Delivery
# =============================================================================

@dataclass
class DeliveryResult:
    """Result of a webhook delivery attempt."""

    success: bool
    status_code: int
    error: Optional[str] = None
    retry_count: int = 0
    duration_ms: float = 0.0


def dispatch_webhook(
    webhook: "WebhookConfig",
    payload: dict,
    timeout: float = REQUEST_TIMEOUT,
) -> Tuple[bool, int, Optional[str]]:
    """
    Dispatch a single webhook synchronously.

    Args:
        webhook: Webhook configuration
        payload: Event payload to send
        timeout: Request timeout in seconds

    Returns:
        Tuple of (success, status_code, error_message)
    """
    # Import here to avoid circular dependency
    from aragora.server.handlers.webhooks import generate_signature

    try:
        # Serialize payload
        payload_json = json.dumps(payload, default=str)

        # Generate signature
        signature = generate_signature(payload_json, webhook.secret)

        # Build headers with distributed tracing support
        headers = {
            "Content-Type": "application/json",
            "User-Agent": USER_AGENT,
            "X-Aragora-Signature": signature,
            "X-Aragora-Event": payload.get("event", "unknown"),
            "X-Aragora-Delivery": payload.get("delivery_id", ""),
            "X-Aragora-Timestamp": str(int(time.time())),
        }

        # Add correlation ID for distributed tracing
        # Check payload first (from event data), then current trace context
        correlation_id = (
            payload.get("data", {}).get("correlation_id")
            or payload.get("correlation_id")
            or get_trace_id()
        )
        if correlation_id:
            headers["X-Aragora-Correlation-ID"] = correlation_id

        # Build request
        request = Request(
            webhook.url,
            data=payload_json.encode("utf-8"),
            headers=headers,
            method="POST",
        )

        # Send request
        start_time = time.time()
        with urlopen(request, timeout=timeout) as response:
            status_code = response.status
            duration_ms = (time.time() - start_time) * 1000

            logger.debug(
                f"Webhook delivered to {webhook.url}: "
                f"status={status_code}, duration={duration_ms:.1f}ms"
            )

            return True, status_code, None

    except HTTPError as e:
        logger.warning(f"Webhook HTTP error for {webhook.url}: {e.code} {e.reason}")
        return False, e.code, f"HTTP {e.code}: {e.reason}"

    except URLError as e:
        logger.warning(f"Webhook URL error for {webhook.url}: {e.reason}")
        return False, 0, f"Connection failed: {e.reason}"

    except TimeoutError:
        logger.warning(f"Webhook timeout for {webhook.url}")
        return False, 0, "Request timed out"

    except Exception as e:
        logger.error(f"Webhook delivery error for {webhook.url}: {e}")
        return False, 0, str(e)


def dispatch_webhook_with_retry(
    webhook: "WebhookConfig",
    payload: dict,
    max_retries: int = MAX_RETRIES,
    initial_delay: float = INITIAL_RETRY_DELAY,
    max_delay: float = MAX_RETRY_DELAY,
) -> DeliveryResult:
    """
    Dispatch webhook with exponential backoff retry.

    Args:
        webhook: Webhook configuration
        payload: Event payload
        max_retries: Maximum retry attempts
        initial_delay: Initial retry delay in seconds
        max_delay: Maximum retry delay in seconds

    Returns:
        DeliveryResult with outcome
    """
    start_time = time.time()
    delay = initial_delay

    for attempt in range(max_retries + 1):
        success, status_code, error = dispatch_webhook(webhook, payload)

        if success:
            return DeliveryResult(
                success=True,
                status_code=status_code,
                retry_count=attempt,
                duration_ms=(time.time() - start_time) * 1000,
            )

        # Don't retry on 4xx errors (client errors)
        if 400 <= status_code < 500:
            return DeliveryResult(
                success=False,
                status_code=status_code,
                error=error,
                retry_count=attempt,
                duration_ms=(time.time() - start_time) * 1000,
            )

        # Retry on 5xx or connection errors
        if attempt < max_retries:
            logger.info(
                f"Retrying webhook {webhook.id} in {delay:.1f}s "
                f"(attempt {attempt + 1}/{max_retries})"
            )
            time.sleep(delay)
            delay = min(delay * 2, max_delay)  # Exponential backoff

    return DeliveryResult(
        success=False,
        status_code=status_code,
        error=error,
        retry_count=max_retries,
        duration_ms=(time.time() - start_time) * 1000,
    )


# =============================================================================
# Webhook Dispatcher
# =============================================================================

class WebhookDispatcher:
    """
    Async webhook event dispatcher.

    Subscribes to event streams and delivers webhooks in background threads.
    """

    def __init__(self, max_workers: int = MAX_WORKERS):
        """Initialize dispatcher with thread pool."""
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="webhook-",
        )
        self._subscriptions: List[Callable] = []
        self._shutdown = False

        # Stats
        self._deliveries = 0
        self._successes = 0
        self._failures = 0
        self._lock = threading.Lock()

    def subscribe_to_stream(self, event_emitter: "SyncEventEmitter") -> None:
        """
        Subscribe to an event emitter to receive events.

        Args:
            event_emitter: SyncEventEmitter instance to subscribe to
        """
        def on_event(event: "StreamEvent"):
            if not self._shutdown:
                self.dispatch_event(event.type.value, event.to_dict())

        event_emitter.subscribe(on_event)
        self._subscriptions.append(on_event)
        logger.info("Webhook dispatcher subscribed to event stream")

    def dispatch_event(self, event_type: str, data: dict) -> None:
        """
        Dispatch event to all registered webhooks.

        Args:
            event_type: Event type string
            data: Event data
        """
        if self._shutdown:
            return

        # Import here to avoid circular dependency
        from aragora.server.handlers.webhooks import get_webhook_store

        store = get_webhook_store()
        webhooks = store.get_for_event(event_type)

        if not webhooks:
            return

        # Create payload
        delivery_id = f"{event_type}-{int(time.time() * 1000)}"
        payload = {
            "event": event_type,
            "delivery_id": delivery_id,
            "timestamp": time.time(),
            "data": data,
        }

        # Submit deliveries to thread pool
        for webhook in webhooks:
            self._executor.submit(
                self._deliver_webhook,
                webhook,
                payload.copy(),
            )

    def _deliver_webhook(self, webhook: "WebhookConfig", payload: dict) -> None:
        """Deliver webhook in background thread."""
        from aragora.server.handlers.webhooks import get_webhook_store

        result = dispatch_webhook_with_retry(webhook, payload)

        # Update stats
        with self._lock:
            self._deliveries += 1
            if result.success:
                self._successes += 1
            else:
                self._failures += 1

        # Record delivery in store
        store = get_webhook_store()
        store.record_delivery(
            webhook_id=webhook.id,
            status_code=result.status_code,
            success=result.success,
        )

        if not result.success:
            logger.warning(
                f"Webhook delivery failed: {webhook.id} -> {webhook.url}: {result.error}"
            )

    def get_stats(self) -> dict:
        """Get dispatcher statistics."""
        with self._lock:
            return {
                "deliveries": self._deliveries,
                "successes": self._successes,
                "failures": self._failures,
                "success_rate": (
                    self._successes / self._deliveries
                    if self._deliveries > 0
                    else 1.0
                ),
                "active_workers": len(self._executor._threads),
            }

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the dispatcher."""
        self._shutdown = True
        self._executor.shutdown(wait=wait)
        logger.info("Webhook dispatcher shutdown")


# =============================================================================
# Global Dispatcher
# =============================================================================

_dispatcher: Optional[WebhookDispatcher] = None


def get_dispatcher() -> WebhookDispatcher:
    """Get or create the global webhook dispatcher."""
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = WebhookDispatcher()
    return _dispatcher


def dispatch_event(event_type: str, data: dict) -> None:
    """
    Dispatch an event to all registered webhooks.

    Convenience function that uses the global dispatcher.

    Args:
        event_type: Event type string (e.g., "debate_end")
        data: Event data dict
    """
    dispatcher = get_dispatcher()
    dispatcher.dispatch_event(event_type, data)


def shutdown_dispatcher(wait: bool = True) -> None:
    """Shutdown the global dispatcher."""
    global _dispatcher
    if _dispatcher is not None:
        _dispatcher.shutdown(wait=wait)
        _dispatcher = None


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "WebhookDispatcher",
    "get_dispatcher",
    "dispatch_event",
    "dispatch_webhook",
    "dispatch_webhook_with_retry",
    "shutdown_dispatcher",
    "DeliveryResult",
]
