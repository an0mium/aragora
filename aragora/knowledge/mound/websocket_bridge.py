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
from typing import TYPE_CHECKING, Any, Dict, Optional

from aragora.knowledge.mound.event_batcher import EventBatcher, AdapterEventBatcher

if TYPE_CHECKING:
    from aragora.server.stream.broadcaster import WebSocketBroadcaster
    from aragora.events.types import StreamEvent

logger = logging.getLogger(__name__)


class KMWebSocketBridge:
    """
    Bridges KM events to WebSocket clients via batching.

    Uses EventBatcher to collect events and emit them in efficient batches
    to avoid flooding clients during bulk KM operations.
    """

    def __init__(
        self,
        broadcaster: Optional["WebSocketBroadcaster"] = None,
        batch_interval_ms: float = 100.0,
        max_batch_size: int = 50,
        passthrough_events: Optional[list[str]] = None,
    ):
        """
        Initialize the bridge.

        Args:
            broadcaster: WebSocket broadcaster for sending events
            batch_interval_ms: Batch interval in milliseconds (default 100ms)
            max_batch_size: Maximum batch size before flush (default 50)
            passthrough_events: Event types to send immediately (e.g., errors)
        """
        self._broadcaster = broadcaster
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Create the event batcher
        self._batcher = EventBatcher(
            callback=self._emit_callback,
            batch_interval_ms=batch_interval_ms,
            max_batch_size=max_batch_size,
            passthrough_event_types=passthrough_events or [
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
            stream_event = StreamEvent(event_type=event_enum, data=data)

            # Schedule async broadcast
            if self._loop and self._loop.is_running():
                asyncio.ensure_future(
                    self._broadcaster.broadcast(stream_event),
                    loop=self._loop,
                )
            else:
                logger.debug(
                    "[km_websocket] No event loop available, event not broadcast"
                )

        except Exception as e:
            logger.warning(f"[km_websocket] Failed to emit event: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get batching statistics."""
        return self._batcher.get_stats()


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
    "get_km_bridge",
    "set_km_bridge",
    "create_km_bridge",
]
