"""
Protocol Handler Registry for Aragora Debates.

Provides a registry pattern for handling protocol messages.
Enables extensible message processing with type-safe handlers.

Inspired by gastown's protocol handler pattern (protocol/handlers.go).
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Coroutine, Dict, List, Optional

from .messages import ProtocolMessage, ProtocolMessageType

logger = logging.getLogger(__name__)

# Type alias for async handlers
AsyncHandler = Callable[[ProtocolMessage], Coroutine[Any, Any, None]]


class ProtocolHandler(ABC):
    """
    Abstract base class for protocol message handlers.

    Handlers process specific message types and can trigger side effects
    like notifications, state updates, or downstream workflows.
    """

    @property
    @abstractmethod
    def message_types(self) -> List[ProtocolMessageType]:
        """List of message types this handler processes."""
        pass

    @abstractmethod
    async def handle(self, message: ProtocolMessage) -> None:
        """
        Handle a protocol message.

        Args:
            message: The protocol message to handle.
        """
        pass

    async def on_error(self, message: ProtocolMessage, error: Exception) -> None:
        """
        Called when handler raises an exception.

        Override to customize error handling.

        Args:
            message: The message that caused the error.
            error: The exception that was raised.
        """
        logger.error(
            f"Handler {self.__class__.__name__} failed for {message.message_type.value}: {error}",
            exc_info=True,
        )


class ProtocolHandlerRegistry:
    """
    Registry for protocol message handlers.

    Features:
    - Register handlers by message type
    - Support for multiple handlers per type
    - Async message dispatch
    - Error isolation (one handler failure doesn't affect others)
    - Handler priority ordering
    """

    def __init__(self):
        """Initialize the handler registry."""
        self._handlers: Dict[ProtocolMessageType, List[tuple[int, AsyncHandler]]] = {}
        self._class_handlers: Dict[ProtocolMessageType, List[tuple[int, ProtocolHandler]]] = {}
        self._global_handlers: List[tuple[int, AsyncHandler]] = []

    def register(
        self,
        message_type: ProtocolMessageType,
        handler: AsyncHandler,
        priority: int = 100,
    ) -> None:
        """
        Register a function handler for a message type.

        Args:
            message_type: The message type to handle.
            handler: Async function to call for matching messages.
            priority: Handler priority (lower runs first). Default 100.
        """
        if message_type not in self._handlers:
            self._handlers[message_type] = []

        self._handlers[message_type].append((priority, handler))
        self._handlers[message_type].sort(key=lambda x: x[0])

        logger.debug(f"Registered handler for {message_type.value} with priority {priority}")

    def register_handler(self, handler: ProtocolHandler, priority: int = 100) -> None:
        """
        Register a class-based handler.

        The handler will be called for all message types it declares.

        Args:
            handler: The protocol handler instance.
            priority: Handler priority (lower runs first). Default 100.
        """
        for message_type in handler.message_types:
            if message_type not in self._class_handlers:
                self._class_handlers[message_type] = []

            self._class_handlers[message_type].append((priority, handler))
            self._class_handlers[message_type].sort(key=lambda x: x[0])

        logger.debug(
            f"Registered class handler {handler.__class__.__name__} "
            f"for {[mt.value for mt in handler.message_types]}"
        )

    def register_global(self, handler: AsyncHandler, priority: int = 100) -> None:
        """
        Register a global handler that receives all messages.

        Args:
            handler: Async function to call for all messages.
            priority: Handler priority (lower runs first). Default 100.
        """
        self._global_handlers.append((priority, handler))
        self._global_handlers.sort(key=lambda x: x[0])
        logger.debug(f"Registered global handler with priority {priority}")

    def unregister(
        self,
        message_type: ProtocolMessageType,
        handler: AsyncHandler,
    ) -> bool:
        """
        Unregister a function handler.

        Args:
            message_type: The message type.
            handler: The handler to remove.

        Returns:
            True if handler was found and removed.
        """
        if message_type not in self._handlers:
            return False

        original_len = len(self._handlers[message_type])
        self._handlers[message_type] = [
            (p, h) for p, h in self._handlers[message_type] if h != handler
        ]

        return len(self._handlers[message_type]) < original_len

    async def dispatch(self, message: ProtocolMessage) -> int:
        """
        Dispatch a message to all registered handlers.

        Handlers are called in priority order. Errors are logged but don't
        stop other handlers from running.

        Args:
            message: The protocol message to dispatch.

        Returns:
            Number of handlers that successfully processed the message.
        """
        handlers_called = 0

        # Run global handlers first
        for priority, handler in self._global_handlers:
            try:
                await handler(message)
                handlers_called += 1
            except Exception as e:
                logger.error(
                    f"Global handler failed for {message.message_type.value}: {e}",
                    exc_info=True,
                )

        # Run type-specific function handlers
        if message.message_type in self._handlers:
            for priority, handler in self._handlers[message.message_type]:
                try:
                    await handler(message)
                    handlers_called += 1
                except Exception as e:
                    logger.error(
                        f"Handler failed for {message.message_type.value}: {e}",
                        exc_info=True,
                    )

        # Run type-specific class handlers
        if message.message_type in self._class_handlers:
            for priority, handler in self._class_handlers[message.message_type]:
                try:
                    await handler.handle(message)
                    handlers_called += 1
                except Exception as e:
                    await handler.on_error(message, e)

        return handlers_called

    async def dispatch_concurrent(self, message: ProtocolMessage) -> int:
        """
        Dispatch a message to all handlers concurrently.

        Unlike dispatch(), all handlers run in parallel. Use when handlers
        are independent and order doesn't matter.

        Args:
            message: The protocol message to dispatch.

        Returns:
            Number of handlers that successfully processed the message.
        """
        tasks = []

        # Collect all handlers
        for priority, handler in self._global_handlers:
            tasks.append(self._safe_call(handler, message))

        if message.message_type in self._handlers:
            for priority, handler in self._handlers[message.message_type]:
                tasks.append(self._safe_call(handler, message))

        if message.message_type in self._class_handlers:
            for priority, handler in self._class_handlers[message.message_type]:
                tasks.append(self._safe_call_class(handler, message))

        if not tasks:
            return 0

        results = await asyncio.gather(*tasks, return_exceptions=True)
        return sum(1 for r in results if r is True)

    async def _safe_call(self, handler: AsyncHandler, message: ProtocolMessage) -> bool:
        """Safely call a function handler."""
        try:
            await handler(message)
            return True
        except Exception as e:
            logger.error(
                f"Handler failed for {message.message_type.value}: {e}",
                exc_info=True,
            )
            return False

    async def _safe_call_class(self, handler: ProtocolHandler, message: ProtocolMessage) -> bool:
        """Safely call a class handler."""
        try:
            await handler.handle(message)
            return True
        except Exception as e:
            await handler.on_error(message, e)
            return False

    def get_handlers(self, message_type: ProtocolMessageType) -> List[AsyncHandler]:
        """Get all function handlers for a message type."""
        handlers = [h for _, h in self._handlers.get(message_type, [])]
        return handlers

    def clear(self) -> None:
        """Clear all registered handlers."""
        self._handlers.clear()
        self._class_handlers.clear()
        self._global_handlers.clear()
        logger.debug("Cleared all protocol handlers")


# Default global registry
_default_registry: Optional[ProtocolHandlerRegistry] = None


def get_handler_registry() -> ProtocolHandlerRegistry:
    """Get the default protocol handler registry."""
    global _default_registry
    if _default_registry is None:
        _default_registry = ProtocolHandlerRegistry()
    return _default_registry


# Decorator for registering handlers


def handles(*message_types: ProtocolMessageType, priority: int = 100):
    """
    Decorator for registering a function as a protocol handler.

    Usage:
        @handles(ProtocolMessageType.PROPOSAL_SUBMITTED)
        async def on_proposal(message: ProtocolMessage):
            print(f"Received proposal: {message.message_id}")

    Args:
        *message_types: Message types to handle.
        priority: Handler priority (lower runs first).
    """

    def decorator(func: AsyncHandler) -> AsyncHandler:
        registry = get_handler_registry()
        for message_type in message_types:
            registry.register(message_type, func, priority)
        return func

    return decorator


# Built-in handlers for common operations


class LoggingHandler(ProtocolHandler):
    """Handler that logs all protocol messages."""

    @property
    def message_types(self) -> List[ProtocolMessageType]:
        return list(ProtocolMessageType)

    async def handle(self, message: ProtocolMessage) -> None:
        logger.info(
            f"Protocol: {message.message_type.value} | "
            f"debate={message.debate_id[:8]}... | "
            f"agent={message.agent_id or 'N/A'} | "
            f"round={message.round_number}"
        )


class MetricsHandler(ProtocolHandler):
    """Handler that tracks protocol message metrics."""

    def __init__(self):
        self._counts: Dict[ProtocolMessageType, int] = {}
        self._debate_counts: Dict[str, int] = {}

    @property
    def message_types(self) -> List[ProtocolMessageType]:
        return list(ProtocolMessageType)

    async def handle(self, message: ProtocolMessage) -> None:
        # Increment type count
        self._counts[message.message_type] = self._counts.get(message.message_type, 0) + 1

        # Increment debate count
        self._debate_counts[message.debate_id] = self._debate_counts.get(message.debate_id, 0) + 1

    def get_counts(self) -> Dict[str, int]:
        """Get message counts by type."""
        return {k.value: v for k, v in self._counts.items()}

    def get_debate_count(self, debate_id: str) -> int:
        """Get message count for a debate."""
        return self._debate_counts.get(debate_id, 0)
