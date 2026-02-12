"""Event and handler protocol definitions.

Provides Protocol classes for event emission systems
and HTTP endpoint handlers.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Protocol, runtime_checkable


@runtime_checkable
class EventEmitterProtocol(Protocol):
    """Protocol for event emission systems."""

    def emit(self, event: Any, data: dict[str, Any] | None = None) -> None:
        """Emit an event. Can be called with event object or (event_type, data)."""
        ...

    def on(self, event_type: str, callback: Callable[..., Any]) -> None:
        """Register an event listener."""
        ...


@runtime_checkable
class AsyncEventEmitterProtocol(Protocol):
    """Protocol for async event emission."""

    async def emit_async(self, event_type: str, data: dict[str, Any]) -> None:
        """Emit an event asynchronously."""
        ...


@runtime_checkable
class HandlerProtocol(Protocol):
    """Protocol for HTTP endpoint handlers."""

    def can_handle(self, path: str) -> bool:
        """Check if handler can process this path."""
        ...

    def handle(
        self,
        path: str,
        query: dict[str, Any],
        request_handler: Any,
    ) -> Any | None:
        """Handle the request and return result."""
        ...


@runtime_checkable
class BaseHandlerProtocol(HandlerProtocol, Protocol):
    """Extended handler protocol with common patterns."""

    ROUTES: list[str]
    ctx: dict[str, Any]

    def read_json_body(self, handler: Any) -> dict[str, Any] | None:
        """Read and parse JSON body from request."""
        ...


__all__ = [
    "EventEmitterProtocol",
    "AsyncEventEmitterProtocol",
    "HandlerProtocol",
    "BaseHandlerProtocol",
]
