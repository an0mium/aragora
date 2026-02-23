"""Cache invalidation event bus."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)


@dataclass
class CacheInvalidationEvent:
    """Event for cache invalidation."""

    event_type: str  # "node_updated", "node_deleted", "query_invalidated"
    workspace_id: str
    item_id: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "workspace_id": self.workspace_id,
            "item_id": self.item_id,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


class CacheInvalidationBus:
    """
    Event bus for cache invalidation.

    Allows subscribers to receive invalidation events for
    coordinated cache updates across the system.
    """

    def __init__(self, max_log_size: int = 1000) -> None:
        self._subscribers: list[Callable[[CacheInvalidationEvent], Awaitable[None]]] = []
        self._event_log: list[CacheInvalidationEvent] = []
        self._max_log_size = max_log_size

    def subscribe(
        self, callback: Callable[[CacheInvalidationEvent], Awaitable[None]]
    ) -> Callable[[], None]:
        """
        Subscribe to cache invalidation events.

        Returns an unsubscribe function.
        """
        self._subscribers.append(callback)

        def unsubscribe() -> None:
            if callback in self._subscribers:
                self._subscribers.remove(callback)

        return unsubscribe

    async def publish(self, event: CacheInvalidationEvent) -> None:
        """Publish a cache invalidation event to all subscribers."""
        # Log event
        self._event_log.append(event)
        if len(self._event_log) > self._max_log_size:
            self._event_log = self._event_log[-self._max_log_size // 2 :]

        # Notify subscribers
        errors = []
        for subscriber in self._subscribers:
            try:
                await subscriber(event)
            except (ValueError, TypeError, RuntimeError, AttributeError) as e:
                logger.warning("Cache invalidation subscriber error (expected): %s", e)
                errors.append("Subscriber notification failed")
            except (RuntimeError, ValueError, OSError, AttributeError) as e:
                logger.exception("Cache invalidation subscriber error (unexpected): %s", e)
                errors.append("Subscriber notification failed")

        if errors:
            logger.warning("Cache invalidation had %s subscriber errors", len(errors))

    async def publish_node_update(self, workspace_id: str, node_id: str, **metadata: Any) -> None:
        """Convenience method for node update events."""
        await self.publish(
            CacheInvalidationEvent(
                event_type="node_updated",
                workspace_id=workspace_id,
                item_id=node_id,
                metadata=metadata,
            )
        )

    async def publish_node_delete(self, workspace_id: str, node_id: str, **metadata: Any) -> None:
        """Convenience method for node deletion events."""
        await self.publish(
            CacheInvalidationEvent(
                event_type="node_deleted",
                workspace_id=workspace_id,
                item_id=node_id,
                metadata=metadata,
            )
        )

    async def publish_query_invalidation(self, workspace_id: str, **metadata: Any) -> None:
        """Convenience method for query cache invalidation."""
        await self.publish(
            CacheInvalidationEvent(
                event_type="query_invalidated",
                workspace_id=workspace_id,
                metadata=metadata,
            )
        )

    def get_recent_events(self, limit: int = 100) -> list[dict[str, Any]]:
        """Get recent invalidation events."""
        return [e.to_dict() for e in self._event_log[-limit:]]


# Global cache invalidation bus
_invalidation_bus: CacheInvalidationBus | None = None


def get_invalidation_bus() -> CacheInvalidationBus:
    """Get or create the global cache invalidation bus."""
    global _invalidation_bus
    if _invalidation_bus is None:
        _invalidation_bus = CacheInvalidationBus()
    return _invalidation_bus
