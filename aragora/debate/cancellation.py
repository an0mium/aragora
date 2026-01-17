"""
Cancellation tokens for cooperative task cancellation.

Provides a mechanism for graceful cancellation of long-running debate operations.
Inspired by claude-code-by-agents patterns for abort control.

Usage:
    token = CancellationToken()
    ctx.cancellation_token = token

    # In an async phase:
    async def execute(self, ctx: DebateContext):
        for round in range(self.rounds):
            if ctx.cancellation_token.is_cancelled:
                raise DebateCancelled(ctx.cancellation_token.reason)
            await self._run_round(ctx)

    # To cancel from outside:
    token.cancel("User requested cancellation")
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Optional

__all__ = [
    "CancellationToken",
    "CancellationScope",
    "CancellationReason",
    "DebateCancelled",
    "create_linked_token",
]

logger = logging.getLogger(__name__)


class CancellationReason(Enum):
    """Standard reasons for cancellation."""

    USER_REQUESTED = "user_requested"
    TIMEOUT = "timeout"
    RESOURCE_LIMIT = "resource_limit"
    PARENT_CANCELLED = "parent_cancelled"
    ERROR = "error"
    SHUTDOWN = "shutdown"


class DebateCancelled(Exception):
    """Exception raised when a debate is cancelled."""

    def __init__(
        self,
        reason: str,
        reason_type: CancellationReason = CancellationReason.USER_REQUESTED,
        partial_result: Optional[Any] = None,
    ):
        self.reason = reason
        self.reason_type = reason_type
        self.partial_result = partial_result
        super().__init__(f"Debate cancelled: {reason}")


@dataclass
class CancellationToken:
    """
    Cooperative cancellation token for long-running operations.

    Thread-safe and asyncio-safe mechanism for signaling cancellation.
    Supports:
    - Cancellation with reason tracking
    - Callback registration for cleanup
    - Linked tokens (child cancelled when parent cancelled)
    - Async waiting for cancellation
    """

    _cancelled: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
    _reason: Optional[str] = field(default=None, repr=False)
    _reason_type: CancellationReason = field(default=CancellationReason.USER_REQUESTED, repr=False)
    _cancelled_at: Optional[datetime] = field(default=None, repr=False)
    _callbacks: list[Callable[["CancellationToken"], None]] = field(
        default_factory=list, repr=False
    )
    _children: list["CancellationToken"] = field(default_factory=list, repr=False)
    _parent: Optional["CancellationToken"] = field(default=None, repr=False)

    def cancel(
        self,
        reason: str = "Cancellation requested",
        reason_type: CancellationReason = CancellationReason.USER_REQUESTED,
    ) -> None:
        """
        Signal cancellation.

        Args:
            reason: Human-readable cancellation reason
            reason_type: Structured reason type for programmatic handling
        """
        if self._cancelled.is_set():
            return  # Already cancelled

        self._reason = reason
        self._reason_type = reason_type
        self._cancelled_at = datetime.utcnow()
        self._cancelled.set()

        logger.info(f"cancellation_signalled reason={reason} type={reason_type.value}")

        # Execute registered callbacks
        for callback in self._callbacks:
            try:
                callback(self)
            except Exception as e:
                logger.warning(f"Cancellation callback failed: {e}")

        # Propagate to children
        for child in self._children:
            if not child.is_cancelled:
                child.cancel(
                    reason=f"Parent cancelled: {reason}",
                    reason_type=CancellationReason.PARENT_CANCELLED,
                )

    @property
    def is_cancelled(self) -> bool:
        """Check if cancellation has been requested."""
        return self._cancelled.is_set()

    @property
    def reason(self) -> Optional[str]:
        """Get the cancellation reason, if cancelled."""
        return self._reason

    @property
    def reason_type(self) -> CancellationReason:
        """Get the structured cancellation reason type."""
        return self._reason_type

    @property
    def cancelled_at(self) -> Optional[datetime]:
        """Get the timestamp when cancellation was requested."""
        return self._cancelled_at

    async def wait_for_cancellation(self, timeout: Optional[float] = None) -> str:
        """
        Wait asynchronously until cancellation is requested.

        Args:
            timeout: Optional timeout in seconds

        Returns:
            The cancellation reason

        Raises:
            asyncio.TimeoutError: If timeout expires before cancellation
        """
        if timeout is not None:
            await asyncio.wait_for(self._cancelled.wait(), timeout=timeout)
        else:
            await self._cancelled.wait()
        return self._reason or "Unknown reason"

    def check(self) -> None:
        """
        Check if cancelled and raise DebateCancelled if so.

        Use this at cancellation points in your code:
            token.check()  # Raises if cancelled
        """
        if self.is_cancelled:
            raise DebateCancelled(
                reason=self._reason or "Cancellation requested",
                reason_type=self._reason_type,
            )

    def register_callback(
        self, callback: Callable[["CancellationToken"], None]
    ) -> Callable[[], None]:
        """
        Register a callback to be called on cancellation.

        Args:
            callback: Function to call when cancelled

        Returns:
            Unregister function to remove the callback
        """
        self._callbacks.append(callback)

        def unregister():
            if callback in self._callbacks:
                self._callbacks.remove(callback)

        # If already cancelled, call immediately
        if self.is_cancelled:
            try:
                callback(self)
            except Exception as e:
                logger.warning(f"Cancellation callback failed: {e}")

        return unregister

    def link_child(self, child: "CancellationToken") -> None:
        """
        Link a child token to this parent.

        When this token is cancelled, all children will be cancelled too.
        """
        self._children.append(child)
        child._parent = self

        # If already cancelled, cancel child immediately
        if self.is_cancelled:
            child.cancel(
                reason=f"Parent cancelled: {self._reason}",
                reason_type=CancellationReason.PARENT_CANCELLED,
            )

    def create_child(self) -> "CancellationToken":
        """
        Create a linked child token.

        The child will be cancelled when this token is cancelled.
        """
        child = CancellationToken()
        self.link_child(child)
        return child

    def __bool__(self) -> bool:
        """Allow using token in boolean context: if token: ..."""
        return not self.is_cancelled


def create_linked_token(parent: Optional[CancellationToken] = None) -> CancellationToken:
    """
    Create a cancellation token, optionally linked to a parent.

    Args:
        parent: Optional parent token to link to

    Returns:
        New CancellationToken linked to parent if provided
    """
    token = CancellationToken()
    if parent is not None:
        parent.link_child(token)
    return token


class CancellationScope:
    """
    Context manager for scoped cancellation.

    Creates a child token that is automatically cleaned up on exit.

    Usage:
        async with CancellationScope(parent_token) as token:
            # token is a child of parent_token
            # Will be cancelled if parent is cancelled
            await do_work(token)
    """

    def __init__(
        self,
        parent: Optional[CancellationToken] = None,
        timeout: Optional[float] = None,
    ):
        """
        Initialize cancellation scope.

        Args:
            parent: Optional parent token
            timeout: Optional timeout in seconds (auto-cancels on timeout)
        """
        self._parent = parent
        self._timeout = timeout
        self._token: Optional[CancellationToken] = None
        self._timeout_handle: Optional[asyncio.TimerHandle] = None

    async def __aenter__(self) -> CancellationToken:
        """Enter the scope, creating a child token."""
        self._token = create_linked_token(self._parent)

        # Set up timeout if specified
        if self._timeout is not None:
            loop = asyncio.get_event_loop()
            token = self._token  # Capture for closure
            timeout = self._timeout  # Capture for closure
            self._timeout_handle = loop.call_later(
                timeout,
                lambda: token.cancel(
                    reason=f"Timeout after {timeout}s",
                    reason_type=CancellationReason.TIMEOUT,
                ),
            )

        return self._token

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Exit the scope, cleaning up."""
        # Cancel timeout if it hasn't fired
        if self._timeout_handle is not None:
            self._timeout_handle.cancel()
            self._timeout_handle = None

        # Remove from parent's children list
        if self._parent is not None and self._token in self._parent._children:
            self._parent._children.remove(self._token)

        # Don't suppress exceptions
        return False

    @property
    def token(self) -> Optional[CancellationToken]:
        """Get the scoped token (only valid inside the context)."""
        return self._token
