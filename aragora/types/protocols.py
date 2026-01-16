"""
Protocols for structural typing in Aragora.

These protocols define interfaces for key abstractions, enabling
duck typing while maintaining type safety through structural subtyping.

Usage:
    from aragora.types.protocols import EventEmitterProtocol

    def setup_listener(emitter: EventEmitterProtocol) -> None:
        emitter.subscribe("debate_start", handler)
"""

from __future__ import annotations

from typing import Any, Callable, Coroutine, Optional, Protocol, runtime_checkable


# Type alias for event data
EventData = dict[str, Any]

# Handler type aliases
EventHandlerProtocol = Callable[[Any], Coroutine[Any, Any, None]]
SyncEventHandlerProtocol = Callable[[Any], None]


@runtime_checkable
class EventEmitterProtocol(Protocol):
    """
    Protocol for event emitters.

    Defines the interface for event emission and subscription.
    Both EventBus and external emitters can satisfy this protocol.

    Example:
        def process_events(emitter: EventEmitterProtocol) -> None:
            async def handler(event):
                print(f"Received: {event}")

            emitter.subscribe("my_event", handler)
    """

    def subscribe(
        self,
        event_type: str,
        handler: EventHandlerProtocol,
    ) -> None:
        """Subscribe an async handler to an event type."""
        ...

    def subscribe_sync(
        self,
        event_type: str,
        handler: SyncEventHandlerProtocol,
    ) -> None:
        """Subscribe a sync handler to an event type."""
        ...

    def unsubscribe(
        self,
        event_type: str,
        handler: EventHandlerProtocol,
    ) -> bool:
        """Unsubscribe a handler from an event type."""
        ...

    async def emit(
        self,
        event_type: str,
        debate_id: str = "",
        correlation_id: Optional[str] = None,
        **data: Any,
    ) -> None:
        """Emit an event asynchronously."""
        ...

    def emit_sync(
        self,
        event_type: str,
        debate_id: str = "",
        correlation_id: Optional[str] = None,
        **data: Any,
    ) -> None:
        """Emit an event synchronously."""
        ...


@runtime_checkable
class StorageProtocol(Protocol):
    """
    Protocol for storage backends.

    Defines the interface for persistent storage operations.
    """

    async def get(self, key: str) -> Optional[Any]:
        """Get a value by key."""
        ...

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set a value with optional TTL."""
        ...

    async def delete(self, key: str) -> bool:
        """Delete a value by key."""
        ...

    async def exists(self, key: str) -> bool:
        """Check if a key exists."""
        ...


@runtime_checkable
class CacheProtocol(Protocol):
    """
    Protocol for caching backends.

    Defines the interface for cache operations with TTL support.
    """

    def get(self, key: str) -> Optional[Any]:
        """Get a cached value."""
        ...

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set a cached value with optional TTL."""
        ...

    def delete(self, key: str) -> None:
        """Delete a cached value."""
        ...

    def clear(self) -> None:
        """Clear all cached values."""
        ...


@runtime_checkable
class AgentProtocol(Protocol):
    """
    Protocol for debate agents.

    Defines the minimal interface that all agents must implement.
    """

    @property
    def name(self) -> str:
        """Agent's display name."""
        ...

    @property
    def model(self) -> str:
        """Model identifier."""
        ...

    async def generate(
        self,
        prompt: str,
        context: Optional[dict[str, Any]] = None,
    ) -> str:
        """Generate a response to a prompt."""
        ...


@runtime_checkable
class MemoryProtocol(Protocol):
    """
    Protocol for memory systems.

    Defines the interface for storing and retrieving debate memories.
    """

    async def store(
        self,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Store content in memory, returns memory ID."""
        ...

    async def retrieve(
        self,
        query: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Retrieve relevant memories for a query."""
        ...

    async def forget(self, memory_id: str) -> bool:
        """Remove a memory by ID."""
        ...
