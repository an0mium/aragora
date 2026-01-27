"""
Propulsion Engine.

Implements the Gastown propulsion pattern for push-based work assignment
in multi-agent debates. Instead of agents polling for work, work is
pushed to the next stage as soon as it's ready.

Key concepts:
- PropulsionPayload: Work item to be pushed to the next stage
- PropulsionHandler: Handler for a specific event type
- PropulsionEngine: Orchestrates push-based work distribution

Usage:
    engine = PropulsionEngine()

    # Register handlers for different stages
    engine.register_handler("proposals_ready", handle_critiques)
    engine.register_handler("critiques_ready", handle_revisions)

    # Push work to next stage
    await engine.propel("proposals_ready", PropulsionPayload(
        data={"proposals": proposals},
        source_molecule_id="debate-123",
    ))

    # Chain multiple stages
    await engine.chain([
        ("proposals_ready", payload1),
        ("critiques_ready", payload2),
    ])
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional, Union

from aragora.debate.hooks import HookManager, HookType

logger = logging.getLogger(__name__)


class PropulsionPriority(Enum):
    """Priority levels for propulsion events."""

    CRITICAL = 0  # Must be processed immediately
    HIGH = 1  # Process before normal priority
    NORMAL = 2  # Default priority
    LOW = 3  # Process when no higher priority work
    BACKGROUND = 4  # Process during idle time


@dataclass
class PropulsionPayload:
    """
    Payload for a propulsion event.

    Contains the data to be pushed to the next stage along with
    metadata for routing, prioritization, and tracking.
    """

    data: Dict[str, Any]
    priority: PropulsionPriority = PropulsionPriority.NORMAL
    deadline: Optional[datetime] = None
    source_molecule_id: Optional[str] = None

    # Tracking
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source_stage: Optional[str] = None
    target_stage: Optional[str] = None

    # Routing hints
    routing_key: Optional[str] = None
    agent_affinity: Optional[str] = None  # Prefer specific agent

    # Retry tracking
    attempt_count: int = 0
    max_attempts: int = 3
    last_error: Optional[str] = None

    def is_expired(self) -> bool:
        """Check if payload has passed its deadline."""
        if not self.deadline:
            return False
        return datetime.now(timezone.utc) > self.deadline

    def can_retry(self) -> bool:
        """Check if payload can be retried."""
        return self.attempt_count < self.max_attempts

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "data": self.data,
            "priority": self.priority.value,
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "source_molecule_id": self.source_molecule_id,
            "created_at": self.created_at.isoformat(),
            "source_stage": self.source_stage,
            "target_stage": self.target_stage,
            "routing_key": self.routing_key,
            "agent_affinity": self.agent_affinity,
            "attempt_count": self.attempt_count,
            "max_attempts": self.max_attempts,
            "last_error": self.last_error,
        }


@dataclass
class PropulsionResult:
    """Result of a propulsion event."""

    payload_id: str
    success: bool
    handler_name: str
    result: Optional[Any] = None
    error_message: Optional[str] = None
    duration_ms: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "payload_id": self.payload_id,
            "success": self.success,
            "handler_name": self.handler_name,
            "result": self.result,
            "error_message": self.error_message,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp.isoformat(),
        }


# Type alias for propulsion handlers
PropulsionHandler = Union[
    Callable[[PropulsionPayload], Any],
    Callable[[PropulsionPayload], Coroutine[Any, Any, Any]],
]


@dataclass
class RegisteredHandler:
    """A registered propulsion handler."""

    name: str
    handler: PropulsionHandler
    priority: PropulsionPriority
    filter_fn: Optional[Callable[[PropulsionPayload], bool]] = None


class PropulsionEngine:
    """
    Push-based work assignment engine (Gastown pattern).

    The propulsion engine manages the flow of work between stages
    in a multi-agent debate or workflow. Instead of polling, work
    is actively pushed to handlers when it becomes available.

    Features:
    - Priority-based handler execution
    - Deadline-aware scheduling
    - Handler filtering based on payload attributes
    - Integration with HookManager for lifecycle events
    - Chained stage execution
    - Retry with backoff on failure
    """

    def __init__(
        self,
        hook_manager: Optional[HookManager] = None,
        max_concurrent: int = 10,
    ):
        """
        Initialize the propulsion engine.

        Args:
            hook_manager: Optional HookManager for lifecycle events
            max_concurrent: Maximum concurrent handler executions
        """
        self._handlers: Dict[str, List[RegisteredHandler]] = defaultdict(list)
        self._hook_manager = hook_manager or HookManager()
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._pending_payloads: List[PropulsionPayload] = []
        self._results: Dict[str, PropulsionResult] = {}
        self._lock = asyncio.Lock()
        self._running = False
        self._stats = {
            "total_propelled": 0,
            "successful": 0,
            "failed": 0,
            "retried": 0,
        }

    def register_handler(
        self,
        event_type: str,
        handler: PropulsionHandler,
        *,
        name: Optional[str] = None,
        priority: PropulsionPriority = PropulsionPriority.NORMAL,
        filter_fn: Optional[Callable[[PropulsionPayload], bool]] = None,
    ) -> Callable[[], None]:
        """
        Register a handler for a propulsion event type.

        Args:
            event_type: The event type to handle (e.g., "proposals_ready")
            handler: Async or sync callback function
            name: Optional handler name (for debugging)
            priority: Handler execution priority
            filter_fn: Optional filter function (return True to handle)

        Returns:
            Unregister function to remove the handler
        """
        handler_name = name or f"{event_type}_{len(self._handlers[event_type])}"

        registered = RegisteredHandler(
            name=handler_name,
            handler=handler,
            priority=priority,
            filter_fn=filter_fn,
        )

        self._handlers[event_type].append(registered)
        # Sort by priority (lower value = higher priority)
        self._handlers[event_type].sort(key=lambda h: h.priority.value)

        logger.debug(f"Registered propulsion handler: {handler_name} for {event_type}")

        def unregister():
            if registered in self._handlers[event_type]:
                self._handlers[event_type].remove(registered)
                logger.debug(f"Unregistered propulsion handler: {handler_name}")

        return unregister

    def unregister_handler(self, event_type: str, name: str) -> bool:
        """
        Unregister a handler by name.

        Args:
            event_type: The event type
            name: Handler name

        Returns:
            True if handler was found and removed
        """
        for handler in self._handlers[event_type]:
            if handler.name == name:
                self._handlers[event_type].remove(handler)
                return True
        return False

    async def propel(
        self,
        event_type: str,
        payload: PropulsionPayload,
    ) -> List[PropulsionResult]:
        """
        Push work to the next stage by firing an event.

        All registered handlers for the event type will be invoked
        with the payload. Handlers are executed in priority order.

        Args:
            event_type: The event type to fire
            payload: The payload to propel

        Returns:
            List of PropulsionResults from all handlers
        """
        payload.target_stage = event_type
        payload.attempt_count += 1
        self._stats["total_propelled"] += 1

        # Fire ON_PROPEL hook
        await self._hook_manager.trigger(
            HookType.ON_PROPEL,
            source_stage=payload.source_stage,
            target_stage=event_type,
            payload=payload.to_dict(),
        )

        handlers = self._handlers.get(event_type, [])
        if not handlers:
            logger.warning(f"No handlers registered for event: {event_type}")
            return []

        # Check deadline
        if payload.is_expired():
            logger.warning(f"Payload {payload.id} expired before processing")
            return [
                PropulsionResult(
                    payload_id=payload.id,
                    success=False,
                    handler_name="deadline_check",
                    error_message="Payload expired before processing",
                )
            ]

        results: List[PropulsionResult] = []

        for registered in handlers:
            # Apply filter if present
            if registered.filter_fn and not registered.filter_fn(payload):
                logger.debug(f"Handler {registered.name} filtered out payload {payload.id}")
                continue

            result = await self._execute_handler(registered, payload)
            results.append(result)
            self._results[f"{payload.id}:{registered.name}"] = result

            if result.success:
                self._stats["successful"] += 1
            else:
                self._stats["failed"] += 1

        return results

    async def _execute_handler(
        self,
        registered: RegisteredHandler,
        payload: PropulsionPayload,
    ) -> PropulsionResult:
        """Execute a single handler with the payload."""
        start_time = time.time()

        async with self._semaphore:
            try:
                result = registered.handler(payload)
                if asyncio.iscoroutine(result):
                    result = await result

                duration_ms = (time.time() - start_time) * 1000

                return PropulsionResult(
                    payload_id=payload.id,
                    success=True,
                    handler_name=registered.name,
                    result=result,
                    duration_ms=duration_ms,
                )

            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                payload.last_error = str(e)

                logger.error(
                    f"Propulsion handler {registered.name} failed: {e}",
                    exc_info=True,
                )

                return PropulsionResult(
                    payload_id=payload.id,
                    success=False,
                    handler_name=registered.name,
                    error_message=str(e),
                    duration_ms=duration_ms,
                )

    async def chain(
        self,
        events: List[tuple[str, PropulsionPayload]],
        stop_on_failure: bool = True,
    ) -> List[List[PropulsionResult]]:
        """
        Execute a chain of propulsion events in sequence.

        Each event is propelled after the previous one completes.
        The payload of each stage can access results from previous stages.

        Args:
            events: List of (event_type, payload) tuples
            stop_on_failure: If True, stop chain on first failure

        Returns:
            List of result lists for each stage
        """
        all_results: List[List[PropulsionResult]] = []
        previous_stage = None

        for event_type, payload in events:
            # Link to previous stage
            payload.source_stage = previous_stage

            results = await self.propel(event_type, payload)
            all_results.append(results)

            # Check for failures
            if stop_on_failure:
                failures = [r for r in results if not r.success]
                if failures:
                    logger.warning(
                        f"Chain stopped at {event_type} due to failures: "
                        f"{[f.error_message for f in failures]}"
                    )
                    break

            previous_stage = event_type

        return all_results

    async def propel_with_retry(
        self,
        event_type: str,
        payload: PropulsionPayload,
        max_retries: Optional[int] = None,
        backoff_base: float = 1.0,
    ) -> List[PropulsionResult]:
        """
        Propel with automatic retry on failure.

        Args:
            event_type: The event type
            payload: The payload
            max_retries: Maximum retries (default from payload)
            backoff_base: Base delay for exponential backoff

        Returns:
            Results from successful attempt or last failed attempt
        """
        max_attempts = max_retries if max_retries is not None else payload.max_attempts

        for attempt in range(max_attempts):
            results = await self.propel(event_type, payload)

            # Check if all handlers succeeded
            all_success = all(r.success for r in results)
            if all_success or attempt == max_attempts - 1:
                return results

            # Exponential backoff
            delay = backoff_base * (2**attempt)
            logger.info(
                f"Retrying propulsion {payload.id} in {delay:.1f}s "
                f"(attempt {attempt + 2}/{max_attempts})"
            )
            self._stats["retried"] += 1
            await asyncio.sleep(delay)

        return results

    async def broadcast(
        self,
        event_types: List[str],
        payload: PropulsionPayload,
    ) -> Dict[str, List[PropulsionResult]]:
        """
        Broadcast a payload to multiple event types simultaneously.

        Args:
            event_types: List of event types to broadcast to
            payload: The payload to broadcast

        Returns:
            Dict mapping event type to results
        """
        tasks = [
            asyncio.create_task(self.propel(event_type, payload)) for event_type in event_types
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        return {
            event_type: result if isinstance(result, list) else []
            for event_type, result in zip(event_types, results)
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get propulsion statistics."""
        return {
            **self._stats,
            "registered_handlers": {
                event_type: len(handlers) for event_type, handlers in self._handlers.items()
            },
            "pending_payloads": len(self._pending_payloads),
        }

    def get_result(self, payload_id: str, handler_name: str) -> Optional[PropulsionResult]:
        """Get a specific result by payload ID and handler name."""
        return self._results.get(f"{payload_id}:{handler_name}")

    def clear_results(self) -> None:
        """Clear stored results."""
        self._results.clear()


# Global propulsion engine singleton
_default_engine: Optional[PropulsionEngine] = None


def get_propulsion_engine() -> PropulsionEngine:
    """Get the default propulsion engine instance."""
    global _default_engine
    if _default_engine is None:
        _default_engine = PropulsionEngine()
    return _default_engine


def reset_propulsion_engine() -> None:
    """Reset the default engine (for testing)."""
    global _default_engine
    _default_engine = None


# Convenience decorators


def propulsion_handler(
    event_type: str,
    priority: PropulsionPriority = PropulsionPriority.NORMAL,
):
    """
    Decorator to register a function as a propulsion handler.

    Usage:
        @propulsion_handler("proposals_ready")
        async def handle_proposals(payload: PropulsionPayload):
            # Process proposals
            pass
    """

    def decorator(func: PropulsionHandler) -> PropulsionHandler:
        # Register with default engine on import
        engine = get_propulsion_engine()
        engine.register_handler(
            event_type,
            func,
            name=func.__name__,
            priority=priority,
        )
        return func

    return decorator
