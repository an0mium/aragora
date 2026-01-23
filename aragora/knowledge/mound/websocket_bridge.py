"""
WebSocket Bridge for Knowledge Mound Events.

Connects the EventBatcher to WebSocket broadcasting for efficient
real-time Knowledge Mound event delivery.

Usage:
    from aragora.knowledge.mound.websocket_bridge import KMWebSocketBridge

    # Create bridge with broadcaster
    bridge = KMWebSocketBridge(broadcaster)

    # Start batching events
    bridge.start()

    # Queue events from adapters
    bridge.queue_event("knowledge_indexed", {"source": "evidence", "id": "ev_001"})

    # Stop when done
    await bridge.stop()
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from threading import Lock
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

from aragora.knowledge.mound.event_batcher import EventBatcher, AdapterEventBatcher

if TYPE_CHECKING:
    from aragora.server.stream.broadcaster import WebSocketBroadcaster

logger = logging.getLogger(__name__)


@dataclass
class KMSubscription:
    """A subscription to KM events."""

    client_id: str
    event_types: Set[str] = field(default_factory=set)
    sources: Set[str] = field(default_factory=set)  # Empty = all sources
    min_confidence: float = 0.0
    workspace_id: Optional[str] = None  # None = all workspaces
    created_at: float = field(default_factory=lambda: __import__("time").time())

    def matches(self, event_type: str, data: Dict[str, Any]) -> bool:
        """Check if event matches this subscription."""
        # Check event type (empty set = subscribe to all)
        if self.event_types and event_type not in self.event_types:
            return False

        # Check source filter
        if self.sources:
            source = data.get("source", data.get("adapter", ""))
            if source and source not in self.sources:
                return False

        # Check confidence threshold
        confidence = data.get("confidence", 1.0)
        if confidence < self.min_confidence:
            return False

        # Check workspace
        if self.workspace_id:
            ws = data.get("workspace_id", data.get("workspace", ""))
            if ws and ws != self.workspace_id:
                return False

        return True


class KMSubscriptionManager:
    """
    Manages KM event subscriptions per client.

    Allows clients to subscribe to specific:
    - Event types (knowledge_indexed, mound_updated, etc.)
    - Sources (evidence, belief, insights, etc.)
    - Confidence thresholds
    - Workspaces

    Thread-safe for concurrent access.
    """

    def __init__(self):
        self._subscriptions: Dict[str, KMSubscription] = {}
        self._lock = Lock()

    def subscribe(
        self,
        client_id: str,
        event_types: Optional[List[str]] = None,
        sources: Optional[List[str]] = None,
        min_confidence: float = 0.0,
        workspace_id: Optional[str] = None,
    ) -> KMSubscription:
        """
        Subscribe a client to KM events.

        Args:
            client_id: Unique client identifier
            event_types: Event types to receive (None = all)
            sources: Knowledge sources to receive (None = all)
            min_confidence: Minimum confidence threshold
            workspace_id: Specific workspace (None = all)

        Returns:
            The created subscription
        """
        subscription = KMSubscription(
            client_id=client_id,
            event_types=set(event_types) if event_types else set(),
            sources=set(sources) if sources else set(),
            min_confidence=min_confidence,
            workspace_id=workspace_id,
        )

        with self._lock:
            self._subscriptions[client_id] = subscription

        logger.debug(
            f"[km_subscriptions] Client {client_id} subscribed to "
            f"types={event_types or 'all'}, sources={sources or 'all'}"
        )
        return subscription

    def unsubscribe(self, client_id: str) -> bool:
        """
        Unsubscribe a client.

        Args:
            client_id: Client to unsubscribe

        Returns:
            True if client was subscribed
        """
        with self._lock:
            if client_id in self._subscriptions:
                del self._subscriptions[client_id]
                logger.debug(f"[km_subscriptions] Client {client_id} unsubscribed")
                return True
        return False

    def update_subscription(
        self,
        client_id: str,
        add_types: Optional[List[str]] = None,
        remove_types: Optional[List[str]] = None,
        add_sources: Optional[List[str]] = None,
        remove_sources: Optional[List[str]] = None,
    ) -> Optional[KMSubscription]:
        """
        Update an existing subscription.

        Args:
            client_id: Client to update
            add_types: Event types to add
            remove_types: Event types to remove
            add_sources: Sources to add
            remove_sources: Sources to remove

        Returns:
            Updated subscription or None if not found
        """
        with self._lock:
            sub = self._subscriptions.get(client_id)
            if not sub:
                return None

            if add_types:
                sub.event_types.update(add_types)
            if remove_types:
                sub.event_types.difference_update(remove_types)
            if add_sources:
                sub.sources.update(add_sources)
            if remove_sources:
                sub.sources.difference_update(remove_sources)

            return sub

    def get_subscribers(
        self,
        event_type: str,
        data: Dict[str, Any],
    ) -> List[str]:
        """
        Get clients that should receive an event.

        Args:
            event_type: The event type
            data: Event payload

        Returns:
            List of client IDs that match
        """
        matching = []
        with self._lock:
            for client_id, sub in self._subscriptions.items():
                if sub.matches(event_type, data):
                    matching.append(client_id)
        return matching

    def get_subscription(self, client_id: str) -> Optional[KMSubscription]:
        """Get a client's subscription."""
        with self._lock:
            return self._subscriptions.get(client_id)

    def get_all_subscriptions(self) -> Dict[str, KMSubscription]:
        """Get all subscriptions (copy)."""
        with self._lock:
            return dict(self._subscriptions)

    def get_stats(self) -> Dict[str, Any]:
        """Get subscription statistics."""
        with self._lock:
            type_counts: Dict[str, int] = {}
            source_counts: Dict[str, int] = {}

            for sub in self._subscriptions.values():
                for t in sub.event_types:
                    type_counts[t] = type_counts.get(t, 0) + 1
                for s in sub.sources:
                    source_counts[s] = source_counts.get(s, 0) + 1

            return {
                "total_subscribers": len(self._subscriptions),
                "event_type_subscriptions": type_counts,
                "source_subscriptions": source_counts,
            }


class KMWebSocketBridge:
    """
    Bridges KM events to WebSocket clients via batching.

    Uses EventBatcher to collect events and emit them in efficient batches
    to avoid flooding clients during bulk KM operations.

    Features:
    - Event batching for efficiency
    - Selective subscriptions per client
    - Source and workspace filtering
    - Confidence threshold filtering
    """

    def __init__(
        self,
        broadcaster: Optional["WebSocketBroadcaster"] = None,
        batch_interval_ms: float = 100.0,
        max_batch_size: int = 50,
        passthrough_events: Optional[list[str]] = None,
        enable_subscriptions: bool = True,
    ):
        """
        Initialize the bridge.

        Args:
            broadcaster: WebSocket broadcaster for sending events
            batch_interval_ms: Batch interval in milliseconds (default 100ms)
            max_batch_size: Maximum batch size before flush (default 50)
            passthrough_events: Event types to send immediately (e.g., errors)
            enable_subscriptions: Enable per-client filtering (default True)
        """
        self._broadcaster = broadcaster
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._enable_subscriptions = enable_subscriptions

        # Subscription manager for per-client filtering
        self._subscriptions = KMSubscriptionManager()

        # Create the event batcher
        self._batcher = EventBatcher(
            callback=self._emit_callback,
            batch_interval_ms=batch_interval_ms,
            max_batch_size=max_batch_size,
            passthrough_event_types=passthrough_events
            or [
                "km_error",
                "km_sync_failed",
            ],
        )

        # Create adapter callback wrapper
        self._adapter_batcher = AdapterEventBatcher(self._batcher, prefix="km")

    def set_broadcaster(self, broadcaster: "WebSocketBroadcaster") -> None:
        """Set the broadcaster after initialization."""
        self._broadcaster = broadcaster

    def start(self, loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        """Start the event batching loop."""
        self._loop = loop or asyncio.get_event_loop()
        self._batcher.start(self._loop)
        logger.info("[km_websocket] Event batching started")

    async def stop(self) -> None:
        """Stop batching and flush remaining events."""
        await self._batcher.stop()
        logger.info("[km_websocket] Event batching stopped")

    def queue_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Queue a KM event for batched delivery.

        Args:
            event_type: Event type (e.g., "knowledge_indexed")
            data: Event payload data
        """
        self._batcher.queue_event(event_type, data)

    @property
    def adapter_callback(self):
        """Get the callback function for use with adapters."""
        return self._adapter_batcher.event_callback

    @property
    def subscriptions(self) -> KMSubscriptionManager:
        """Get the subscription manager."""
        return self._subscriptions

    def subscribe(
        self,
        client_id: str,
        event_types: Optional[List[str]] = None,
        sources: Optional[List[str]] = None,
        min_confidence: float = 0.0,
        workspace_id: Optional[str] = None,
    ) -> KMSubscription:
        """
        Subscribe a client to KM events.

        Args:
            client_id: Unique client identifier
            event_types: Event types to receive (None = all)
            sources: Knowledge sources to receive (None = all)
            min_confidence: Minimum confidence threshold
            workspace_id: Specific workspace (None = all)

        Returns:
            The created subscription
        """
        return self._subscriptions.subscribe(
            client_id=client_id,
            event_types=event_types,
            sources=sources,
            min_confidence=min_confidence,
            workspace_id=workspace_id,
        )

    def unsubscribe(self, client_id: str) -> bool:
        """Unsubscribe a client."""
        return self._subscriptions.unsubscribe(client_id)

    def _emit_callback(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Callback invoked by batcher to emit events.

        Args:
            event_type: Either individual event type or "km_batch"
            data: Event data (or batch payload for km_batch)
        """
        if not self._broadcaster:
            return

        try:
            from aragora.events.types import StreamEvent, StreamEventType

            # Determine event type enum
            if event_type == "km_batch":
                event_enum = StreamEventType.KM_BATCH
            else:
                # Map individual event types
                event_map = {
                    "km_knowledge_indexed": StreamEventType.KNOWLEDGE_INDEXED,
                    "km_knowledge_queried": StreamEventType.KNOWLEDGE_QUERIED,
                    "km_mound_updated": StreamEventType.MOUND_UPDATED,
                    "km_knowledge_stale": StreamEventType.KNOWLEDGE_STALE,
                    "km_belief_converged": StreamEventType.BELIEF_CONVERGED,
                    "km_crux_detected": StreamEventType.CRUX_DETECTED,
                    "knowledge_indexed": StreamEventType.KNOWLEDGE_INDEXED,
                    "knowledge_queried": StreamEventType.KNOWLEDGE_QUERIED,
                    "mound_updated": StreamEventType.MOUND_UPDATED,
                    "knowledge_stale": StreamEventType.KNOWLEDGE_STALE,
                }
                event_enum = event_map.get(event_type)
                if event_enum is None:
                    # Generic batch event for unrecognized types
                    event_enum = StreamEventType.KM_BATCH

            # Create stream event
            stream_event = StreamEvent(event_type=event_enum, data=data)  # type: ignore[arg-type,call-arg]

            # Schedule async broadcast
            if self._loop and self._loop.is_running():
                asyncio.ensure_future(
                    self._broadcaster.broadcast(stream_event),
                    loop=self._loop,  # type: ignore[arg-type]
                )

                # Record Prometheus metric
                try:
                    from aragora.observability.metrics import record_km_event_emitted  # type: ignore[attr-defined]

                    record_km_event_emitted(event_type)
                except ImportError:
                    pass
            else:
                logger.debug("[km_websocket] No event loop available, event not broadcast")

        except Exception as e:
            logger.warning(f"[km_websocket] Failed to emit event: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get batching and subscription statistics."""
        stats = self._batcher.get_stats()
        stats["subscriptions"] = self._subscriptions.get_stats()
        return stats


# Global bridge instance (can be replaced per-workspace)
_global_bridge: Optional[KMWebSocketBridge] = None


def get_km_bridge() -> Optional[KMWebSocketBridge]:
    """Get the global KM WebSocket bridge."""
    return _global_bridge


def set_km_bridge(bridge: KMWebSocketBridge) -> None:
    """Set the global KM WebSocket bridge."""
    global _global_bridge
    _global_bridge = bridge


def create_km_bridge(
    broadcaster: Optional["WebSocketBroadcaster"] = None,
    **kwargs,
) -> KMWebSocketBridge:
    """
    Create and set the global KM WebSocket bridge.

    Args:
        broadcaster: WebSocket broadcaster
        **kwargs: Additional arguments for KMWebSocketBridge

    Returns:
        The created bridge instance
    """
    global _global_bridge
    _global_bridge = KMWebSocketBridge(broadcaster, **kwargs)
    return _global_bridge


__all__ = [
    "KMWebSocketBridge",
    "KMSubscription",
    "KMSubscriptionManager",
    "get_km_bridge",
    "set_km_bridge",
    "create_km_bridge",
]
