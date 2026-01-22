"""
Inbox sync WebSocket events.

Provides real-time sync progress updates for Gmail inbox synchronization:
- inbox_sync_start: Sync operation started
- inbox_sync_progress: Progress update with message count
- inbox_sync_complete: Sync finished successfully
- inbox_sync_error: Sync failed with error
- new_priority_email: High-priority email detected
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Optional, Set
from datetime import datetime

logger = logging.getLogger(__name__)


class InboxSyncEventType(str, Enum):
    """Types of inbox sync events."""

    SYNC_START = "inbox_sync_start"
    SYNC_PROGRESS = "inbox_sync_progress"
    SYNC_COMPLETE = "inbox_sync_complete"
    SYNC_ERROR = "inbox_sync_error"
    NEW_PRIORITY_EMAIL = "new_priority_email"


@dataclass
class InboxSyncEvent:
    """An inbox sync event for WebSocket broadcast."""

    type: InboxSyncEventType
    user_id: str
    data: Dict[str, Any]
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.utcnow().isoformat() + "Z"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type.value if isinstance(self.type, InboxSyncEventType) else self.type,
            "user_id": self.user_id,
            "data": self.data,
            "timestamp": self.timestamp,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


class InboxSyncEmitter:
    """
    Emitter for inbox sync WebSocket events.

    Manages user subscriptions and broadcasts events to connected clients.
    Thread-safe for concurrent access.
    """

    def __init__(self):
        # Map user_id -> set of websocket connections
        self._subscriptions: Dict[str, Set[Any]] = {}
        self._lock = asyncio.Lock()
        # Callbacks for event handling (e.g., for testing)
        self._event_callbacks: list[Callable[[InboxSyncEvent], None]] = []

    async def subscribe(self, user_id: str, websocket: Any) -> None:
        """Subscribe a WebSocket connection to a user's sync events."""
        async with self._lock:
            if user_id not in self._subscriptions:
                self._subscriptions[user_id] = set()
            self._subscriptions[user_id].add(websocket)
            logger.debug(
                f"[InboxSync] User {user_id} subscribed, total: {len(self._subscriptions[user_id])}"
            )

    async def unsubscribe(self, user_id: str, websocket: Any) -> None:
        """Unsubscribe a WebSocket connection from a user's sync events."""
        async with self._lock:
            if user_id in self._subscriptions:
                self._subscriptions[user_id].discard(websocket)
                if not self._subscriptions[user_id]:
                    del self._subscriptions[user_id]
                logger.debug(f"[InboxSync] User {user_id} unsubscribed")

    async def emit(self, event: InboxSyncEvent) -> int:
        """
        Emit an event to all subscribed clients for the user.

        Returns the number of clients that received the event.
        """
        user_id = event.user_id
        sent_count = 0

        # Call registered callbacks
        for callback in self._event_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"[InboxSync] Callback error: {e}")

        async with self._lock:
            clients = self._subscriptions.get(user_id, set()).copy()

        if not clients:
            return 0

        message = event.to_json()
        dead_clients = []

        for websocket in clients:
            try:
                await websocket.send(message)
                sent_count += 1
            except Exception as e:
                logger.debug(f"[InboxSync] Failed to send to client: {e}")
                dead_clients.append(websocket)

        # Clean up dead connections
        if dead_clients:
            async with self._lock:
                if user_id in self._subscriptions:
                    for ws in dead_clients:
                        self._subscriptions[user_id].discard(ws)

        return sent_count

    async def emit_sync_start(
        self,
        user_id: str,
        total_messages: int = 0,
        phase: str = "Starting sync...",
    ) -> int:
        """Emit sync start event."""
        event = InboxSyncEvent(
            type=InboxSyncEventType.SYNC_START,
            user_id=user_id,
            data={
                "total_messages": total_messages,
                "phase": phase,
            },
        )
        return await self.emit(event)

    async def emit_sync_progress(
        self,
        user_id: str,
        progress: int,
        messages_synced: int,
        total_messages: int,
        phase: str = "Syncing...",
    ) -> int:
        """Emit sync progress event."""
        event = InboxSyncEvent(
            type=InboxSyncEventType.SYNC_PROGRESS,
            user_id=user_id,
            data={
                "progress": progress,
                "messages_synced": messages_synced,
                "total_messages": total_messages,
                "phase": phase,
            },
        )
        return await self.emit(event)

    async def emit_sync_complete(
        self,
        user_id: str,
        messages_synced: int,
    ) -> int:
        """Emit sync complete event."""
        event = InboxSyncEvent(
            type=InboxSyncEventType.SYNC_COMPLETE,
            user_id=user_id,
            data={
                "messages_synced": messages_synced,
            },
        )
        return await self.emit(event)

    async def emit_sync_error(
        self,
        user_id: str,
        error: str,
    ) -> int:
        """Emit sync error event."""
        event = InboxSyncEvent(
            type=InboxSyncEventType.SYNC_ERROR,
            user_id=user_id,
            data={
                "error": error,
            },
        )
        return await self.emit(event)

    async def emit_new_priority_email(
        self,
        user_id: str,
        email_id: str,
        subject: str,
        from_address: str,
        priority: str,
    ) -> int:
        """Emit new priority email notification."""
        event = InboxSyncEvent(
            type=InboxSyncEventType.NEW_PRIORITY_EMAIL,
            user_id=user_id,
            data={
                "email": {
                    "id": email_id,
                    "subject": subject,
                    "from_address": from_address,
                    "priority": priority,
                },
            },
        )
        return await self.emit(event)

    def add_callback(self, callback: Callable[[InboxSyncEvent], None]) -> None:
        """Add a callback for all events (useful for testing/logging)."""
        self._event_callbacks.append(callback)

    def remove_callback(self, callback: Callable[[InboxSyncEvent], None]) -> None:
        """Remove a previously added callback."""
        if callback in self._event_callbacks:
            self._event_callbacks.remove(callback)


# Global emitter instance
_inbox_sync_emitter: Optional[InboxSyncEmitter] = None


def get_inbox_sync_emitter() -> InboxSyncEmitter:
    """Get the global inbox sync emitter instance."""
    global _inbox_sync_emitter
    if _inbox_sync_emitter is None:
        _inbox_sync_emitter = InboxSyncEmitter()
    return _inbox_sync_emitter


# Convenience functions for direct usage
async def emit_sync_start(user_id: str, **kwargs) -> int:
    """Emit sync start event."""
    return await get_inbox_sync_emitter().emit_sync_start(user_id, **kwargs)


async def emit_sync_progress(user_id: str, **kwargs) -> int:
    """Emit sync progress event."""
    return await get_inbox_sync_emitter().emit_sync_progress(user_id, **kwargs)


async def emit_sync_complete(user_id: str, messages_synced: int) -> int:
    """Emit sync complete event."""
    return await get_inbox_sync_emitter().emit_sync_complete(user_id, messages_synced)


async def emit_sync_error(user_id: str, error: str) -> int:
    """Emit sync error event."""
    return await get_inbox_sync_emitter().emit_sync_error(user_id, error)


async def emit_new_priority_email(
    user_id: str,
    email_id: str,
    subject: str,
    from_address: str,
    priority: str,
) -> int:
    """Emit new priority email notification."""
    return await get_inbox_sync_emitter().emit_new_priority_email(
        user_id, email_id, subject, from_address, priority
    )
