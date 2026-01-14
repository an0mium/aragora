"""
Event Bus for Aragora Debates.

Centralized event emission and subscription system for debate events.
Extracted from Arena to enable cleaner event handling and testing.

Usage:
    from aragora.debate.event_bus import EventBus, DebateEvent

    # Create event bus
    bus = EventBus()

    # Subscribe to events
    async def handler(event: DebateEvent):
        print(f"Received: {event.event_type}")

    bus.subscribe("debate_start", handler)

    # Emit events
    await bus.emit("debate_start", debate_id="123", task="Design a rate limiter")
"""

import asyncio
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine, Dict, List, Optional

from aragora.server.middleware.tracing import get_span_id, get_trace_id

logger = logging.getLogger(__name__)


@dataclass
class DebateEvent:
    """Represents a debate event with distributed tracing support."""

    event_type: str
    debate_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    data: Dict[str, Any] = field(default_factory=dict)
    # Distributed tracing fields for correlation across services
    correlation_id: Optional[str] = field(default=None)
    span_id: Optional[str] = field(default=None)

    def __post_init__(self):
        """Auto-populate correlation_id from current trace context if not provided."""
        if self.correlation_id is None:
            self.correlation_id = get_trace_id()
        if self.span_id is None:
            self.span_id = get_span_id()

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization."""
        result = {
            "event_type": self.event_type,
            "debate_id": self.debate_id,
            "timestamp": self.timestamp.isoformat(),
            **self.data,
        }
        # Include tracing fields if present
        if self.correlation_id:
            result["correlation_id"] = self.correlation_id
        if self.span_id:
            result["span_id"] = self.span_id
        return result


# Type alias for event handlers
EventHandler = Callable[[DebateEvent], Coroutine[Any, Any, None]]
SyncEventHandler = Callable[[DebateEvent], None]


class EventBus:
    """
    Centralized event bus for debate events.

    Features:
    - Async event emission
    - Multiple subscribers per event type
    - User event queuing (thread-safe)
    - Event bridge integration for WebSocket streaming
    - Health event broadcasting
    """

    def __init__(
        self,
        event_bridge: Optional[Any] = None,  # EventEmitterBridge
        audience_manager: Optional[Any] = None,  # AudienceManager
        immune_system: Optional[Any] = None,  # TransparentImmuneSystem
        spectator: Optional[Any] = None,  # SpectatorStream
    ):
        """
        Initialize the event bus.

        Args:
            event_bridge: Optional EventEmitterBridge for external notifications
            audience_manager: Optional AudienceManager for user participation
            immune_system: Optional TransparentImmuneSystem for health events
            spectator: Optional SpectatorStream for WebSocket broadcasting
        """
        self._event_bridge = event_bridge
        self._audience_manager = audience_manager
        self._immune_system = immune_system
        self._spectator = spectator

        # Event subscribers
        self._async_handlers: Dict[str, List[EventHandler]] = {}
        self._sync_handlers: Dict[str, List[SyncEventHandler]] = {}

        # User event queue (thread-safe for external input)
        self._user_event_queue: queue.Queue[Dict[str, Any]] = queue.Queue()
        self._user_event_lock = threading.Lock()

        # Metrics
        self._events_emitted: int = 0
        self._events_by_type: Dict[str, int] = {}

    # =========================================================================
    # Subscription Management
    # =========================================================================

    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        """
        Subscribe an async handler to an event type.

        Args:
            event_type: Type of event to subscribe to
            handler: Async handler function
        """
        if event_type not in self._async_handlers:
            self._async_handlers[event_type] = []
        self._async_handlers[event_type].append(handler)
        logger.debug(f"Subscribed async handler to '{event_type}'")

    def subscribe_sync(self, event_type: str, handler: SyncEventHandler) -> None:
        """
        Subscribe a sync handler to an event type.

        Args:
            event_type: Type of event to subscribe to
            handler: Sync handler function
        """
        if event_type not in self._sync_handlers:
            self._sync_handlers[event_type] = []
        self._sync_handlers[event_type].append(handler)
        logger.debug(f"Subscribed sync handler to '{event_type}'")

    def unsubscribe(self, event_type: str, handler: EventHandler) -> bool:
        """
        Unsubscribe an async handler from an event type.

        Args:
            event_type: Type of event to unsubscribe from
            handler: Handler function to remove

        Returns:
            True if handler was found and removed
        """
        if event_type in self._async_handlers:
            try:
                self._async_handlers[event_type].remove(handler)
                return True
            except ValueError:
                pass
        return False

    # =========================================================================
    # Event Emission
    # =========================================================================

    async def emit(
        self,
        event_type: str,
        debate_id: str = "",
        correlation_id: Optional[str] = None,
        **data: Any,
    ) -> None:
        """
        Emit an event to all subscribers.

        Args:
            event_type: Type of event
            debate_id: ID of the debate
            correlation_id: Optional correlation ID for distributed tracing
                           (auto-populated from trace context if not provided)
            **data: Additional event data
        """
        event = DebateEvent(
            event_type=event_type,
            debate_id=debate_id,
            correlation_id=correlation_id,
            data=data,
        )

        # Update metrics
        self._events_emitted += 1
        self._events_by_type[event_type] = self._events_by_type.get(event_type, 0) + 1

        # Notify event bridge (WebSocket streaming)
        if self._event_bridge is not None:
            try:
                event_data = event.to_dict()
                event_data.pop("event_type", None)  # Already passed as first arg
                self._event_bridge.notify(event_type, **event_data)
            except Exception as e:
                logger.warning(f"Event bridge notification failed: {e}")

        # Notify spectator stream
        if self._spectator is not None:
            try:
                self._spectator.emit(event_type, event.to_dict())
            except Exception as e:
                logger.warning(f"Spectator notification failed: {e}")

        # Call sync handlers
        for handler in self._sync_handlers.get(event_type, []):
            try:
                handler(event)
            except Exception as e:
                logger.warning(f"Sync handler error for '{event_type}': {e}")

        # Call async handlers
        async_handlers = self._async_handlers.get(event_type, [])
        if async_handlers:
            await asyncio.gather(
                *[self._safe_call_handler(handler, event) for handler in async_handlers],
                return_exceptions=True,
            )

        logger.debug(f"Emitted event '{event_type}' for debate '{debate_id}'")

    async def _safe_call_handler(
        self,
        handler: EventHandler,
        event: DebateEvent,
    ) -> None:
        """Safely call an async handler with error handling."""
        try:
            await handler(event)
        except Exception as e:
            logger.warning(f"Async handler error for '{event.event_type}': {e}")

    def emit_sync(
        self,
        event_type: str,
        debate_id: str = "",
        correlation_id: Optional[str] = None,
        **data: Any,
    ) -> None:
        """
        Emit an event synchronously (for non-async contexts).

        Only notifies sync handlers and event bridge.

        Args:
            event_type: Type of event
            debate_id: ID of the debate
            correlation_id: Optional correlation ID for distributed tracing
                           (auto-populated from trace context if not provided)
            **data: Additional event data
        """
        event = DebateEvent(
            event_type=event_type,
            debate_id=debate_id,
            correlation_id=correlation_id,
            data=data,
        )

        # Update metrics
        self._events_emitted += 1
        self._events_by_type[event_type] = self._events_by_type.get(event_type, 0) + 1

        # Notify event bridge
        if self._event_bridge is not None:
            try:
                event_data = event.to_dict()
                event_data.pop("event_type", None)  # Already passed as first arg
                self._event_bridge.notify(event_type, **event_data)
            except Exception as e:
                logger.warning(f"Event bridge notification failed: {e}")

        # Call sync handlers only
        for handler in self._sync_handlers.get(event_type, []):
            try:
                handler(event)
            except Exception as e:
                logger.warning(f"Sync handler error for '{event_type}': {e}")

    # =========================================================================
    # Specialized Event Methods (extracted from Arena)
    # =========================================================================

    async def notify_spectator(
        self,
        event_type: str,
        debate_id: str,
        **data: Any,
    ) -> None:
        """
        Notify spectators of a debate event.

        Extracted from Arena._notify_spectator().

        Args:
            event_type: Type of event
            debate_id: ID of the debate
            **data: Event data
        """
        await self.emit(event_type, debate_id=debate_id, **data)

    async def emit_moment_event(
        self,
        debate_id: str,
        moment_type: str,
        description: str,
        agent: Optional[str] = None,
        round_num: Optional[int] = None,
        significance: float = 0.5,
        **extra: Any,
    ) -> None:
        """
        Emit a significant moment event.

        Extracted from Arena._emit_moment_event().

        Args:
            debate_id: ID of the debate
            moment_type: Type of moment (e.g., "breakthrough", "consensus")
            description: Human-readable description
            agent: Optional agent name involved
            round_num: Optional round number
            significance: Significance score (0-1)
            **extra: Additional data
        """
        await self.emit(
            "moment",
            debate_id=debate_id,
            moment_type=moment_type,
            description=description,
            agent=agent,
            round_num=round_num,
            significance=significance,
            **extra,
        )

    async def broadcast_health_event(
        self,
        debate_id: str,
        health_status: Dict[str, Any],
    ) -> None:
        """
        Broadcast immune system health status.

        Extracted from Arena._broadcast_health_event().

        Args:
            debate_id: ID of the debate
            health_status: Health metrics from immune system
        """
        if self._immune_system is None:
            return

        await self.emit(
            "health_update",
            debate_id=debate_id,
            health=health_status,
            timestamp=time.time(),
        )

    # =========================================================================
    # User Event Queue (Thread-Safe)
    # =========================================================================

    def queue_user_event(self, event: Dict[str, Any]) -> None:
        """
        Queue a user participation event (thread-safe).

        Extracted from Arena._handle_user_event().

        Args:
            event: User event data (vote, suggestion, etc.)
        """
        with self._user_event_lock:
            self._user_event_queue.put(event)
        logger.debug(f"Queued user event: {event.get('type', 'unknown')}")

    async def drain_user_events(self, debate_id: str) -> List[Dict[str, Any]]:
        """
        Process all queued user events.

        Extracted from Arena._drain_user_events().

        Args:
            debate_id: ID of the debate

        Returns:
            List of processed events
        """
        events = []
        with self._user_event_lock:
            while not self._user_event_queue.empty():
                try:
                    event = self._user_event_queue.get_nowait()
                    events.append(event)
                except queue.Empty:
                    break

        # Process events through audience manager if available
        if self._audience_manager is not None and events:
            for event in events:
                try:
                    await self._process_user_event(event, debate_id)
                except Exception as e:
                    logger.warning(f"Failed to process user event: {e}")

        return events

    async def _process_user_event(
        self,
        event: Dict[str, Any],
        debate_id: str,
    ) -> None:
        """Process a single user event."""
        event_type = event.get("type", "")

        if event_type == "vote" and self._audience_manager:
            await self._audience_manager.record_vote(
                debate_id=debate_id,
                user_id=event.get("user_id", "anonymous"),
                vote=event.get("vote"),
            )
        elif event_type == "suggestion" and self._audience_manager:
            await self._audience_manager.add_suggestion(
                debate_id=debate_id,
                user_id=event.get("user_id", "anonymous"),
                content=event.get("content", ""),
            )

    # =========================================================================
    # Metrics
    # =========================================================================

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get event bus metrics.

        Returns:
            Dictionary with event statistics
        """
        return {
            "total_events_emitted": self._events_emitted,
            "events_by_type": self._events_by_type.copy(),
            "async_subscribers": {k: len(v) for k, v in self._async_handlers.items()},
            "sync_subscribers": {k: len(v) for k, v in self._sync_handlers.items()},
            "pending_user_events": self._user_event_queue.qsize(),
        }

    def reset_metrics(self) -> None:
        """Reset event metrics."""
        self._events_emitted = 0
        self._events_by_type.clear()


# Singleton instance for global access (optional pattern)
_default_bus: Optional[EventBus] = None


def get_event_bus() -> EventBus:
    """Get the default event bus instance."""
    global _default_bus
    if _default_bus is None:
        _default_bus = EventBus()
    return _default_bus


def set_event_bus(bus: EventBus) -> None:
    """Set the default event bus instance."""
    global _default_bus
    _default_bus = bus


__all__ = [
    "EventBus",
    "DebateEvent",
    "EventHandler",
    "SyncEventHandler",
    "get_event_bus",
    "set_event_bus",
]
