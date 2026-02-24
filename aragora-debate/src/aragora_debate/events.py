"""Event/callback system for debate observation.

Provides a lightweight event emitter that supports both sync and async
callbacks, enabling real-time monitoring of debate progress without
coupling to any specific framework.

Example::

    from aragora_debate.events import EventEmitter, EventType

    emitter = EventEmitter()

    @emitter.on(EventType.PROPOSAL)
    def on_proposal(event):
        print(f"{event.agent} proposed in round {event.round_num}")

    @emitter.on(EventType.CONSENSUS_CHECK)
    async def on_consensus(event):
        await log_consensus(event.data)
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from collections.abc import Callable

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """Types of events emitted during a debate."""

    DEBATE_START = "debate_start"
    DEBATE_END = "debate_end"
    ROUND_START = "round_start"
    ROUND_END = "round_end"
    PROPOSAL = "proposal"
    CRITIQUE = "critique"
    VOTE = "vote"
    CONSENSUS_CHECK = "consensus_check"
    TRICKSTER_INTERVENTION = "trickster_intervention"
    CONVERGENCE_DETECTED = "convergence_detected"


@dataclass
class DebateEvent:
    """An event emitted during a debate."""

    event_type: EventType
    round_num: int = 0
    agent: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# Type alias for event callbacks
EventCallback = Callable[[DebateEvent], Any]


class EventEmitter:
    """Lightweight event emitter supporting sync and async callbacks.

    Listeners are registered per event type. When an event is emitted,
    all registered callbacks for that type are invoked. Async callbacks
    are awaited; sync callbacks are called directly.

    Example::

        emitter = EventEmitter()
        emitter.on(EventType.PROPOSAL)(lambda e: print(e.agent))
        await emitter.emit(EventType.PROPOSAL, agent="claude", data={...})
    """

    def __init__(self) -> None:
        self._listeners: dict[EventType, list[EventCallback]] = {}

    def on(self, event_type: EventType) -> Callable[[EventCallback], EventCallback]:
        """Register a callback for an event type.

        Can be used as a decorator::

            @emitter.on(EventType.PROPOSAL)
            def handler(event):
                ...

        Or called directly::

            emitter.on(EventType.PROPOSAL)(my_handler)
        """

        def decorator(fn: EventCallback) -> EventCallback:
            if event_type not in self._listeners:
                self._listeners[event_type] = []
            self._listeners[event_type].append(fn)
            return fn

        return decorator

    def off(self, event_type: EventType, fn: EventCallback) -> None:
        """Remove a callback for an event type."""
        if event_type in self._listeners:
            self._listeners[event_type] = [cb for cb in self._listeners[event_type] if cb is not fn]

    async def emit(
        self,
        event_type: EventType,
        *,
        round_num: int = 0,
        agent: str = "",
        data: dict[str, Any] | None = None,
    ) -> DebateEvent:
        """Emit an event, invoking all registered callbacks.

        Args:
            event_type: The type of event to emit.
            round_num: Current debate round.
            agent: Agent associated with this event.
            data: Additional event data.

        Returns:
            The emitted DebateEvent.
        """
        event = DebateEvent(
            event_type=event_type,
            round_num=round_num,
            agent=agent,
            data=data or {},
        )

        callbacks = self._listeners.get(event_type, [])
        for cb in callbacks:
            try:
                result = cb(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                logger.exception("Error in event callback for %s", event_type.value)

        return event

    def listener_count(self, event_type: EventType) -> int:
        """Return the number of listeners for an event type."""
        return len(self._listeners.get(event_type, []))

    def clear(self) -> None:
        """Remove all listeners."""
        self._listeners.clear()


__all__ = [
    "EventType",
    "DebateEvent",
    "EventCallback",
    "EventEmitter",
]
