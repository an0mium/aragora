"""Track in-flight requests for graceful shutdown.

Provides a mechanism to track active HTTP requests and coordinate
graceful draining during server shutdown. This ensures that:
1. New requests are rejected when shutdown begins
2. In-flight requests can complete before the server stops
3. A timeout ensures the server doesn't wait indefinitely
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

logger = logging.getLogger(__name__)


class ServiceUnavailable(Exception):
    """Raised when server is shutting down and cannot accept requests."""

    pass


class RequestTracker:
    """Track active requests and support graceful drain.

    Thread-safe tracking of in-flight HTTP requests with support for
    graceful shutdown. During shutdown, new requests are rejected and
    the drain waits for existing requests to complete.

    Usage:
        tracker = RequestTracker()

        # In request handler:
        async with tracker.track_request():
            # Handle request
            ...

        # During shutdown:
        success = await tracker.start_drain(timeout=30.0)
    """

    def __init__(self) -> None:
        """Initialize the request tracker."""
        self._active_count: int = 0
        self._lock: asyncio.Lock = asyncio.Lock()
        self._draining: bool = False
        self._drain_complete: asyncio.Event = asyncio.Event()

    @property
    def active_count(self) -> int:
        """Return the number of currently active requests."""
        return self._active_count

    @property
    def is_draining(self) -> bool:
        """Return True if the tracker is in drain mode."""
        return self._draining

    @asynccontextmanager
    async def track_request(self) -> AsyncIterator[None]:
        """Context manager to track a request.

        Raises:
            ServiceUnavailable: If the server is shutting down

        Usage:
            async with tracker.track_request():
                # Process request
                await handle_request()
        """
        if self._draining:
            raise ServiceUnavailable("Server is shutting down")

        async with self._lock:
            self._active_count += 1

        try:
            yield
        finally:
            async with self._lock:
                self._active_count -= 1
                if self._draining and self._active_count == 0:
                    self._drain_complete.set()

    async def start_drain(self, timeout: float = 30.0) -> bool:
        """Start draining and wait for active requests to complete.

        Sets the draining flag to reject new requests, then waits for
        all in-flight requests to complete up to the specified timeout.

        Args:
            timeout: Maximum seconds to wait for requests to complete

        Returns:
            True if all requests completed, False if timeout was reached
        """
        self._draining = True
        logger.info(f"Starting request drain (active: {self._active_count})")

        if self._active_count == 0:
            logger.info("No active requests, drain complete immediately")
            return True

        try:
            await asyncio.wait_for(self._drain_complete.wait(), timeout)
            logger.info("All in-flight requests completed")
            return True
        except asyncio.TimeoutError:
            logger.warning(
                f"Request drain timeout after {timeout}s "
                f"({self._active_count} requests still active)"
            )
            return False

    def reset(self) -> None:
        """Reset the tracker state.

        Primarily useful for testing. Resets drain state and clears
        the completion event.
        """
        self._draining = False
        self._active_count = 0
        self._drain_complete.clear()


# Global singleton instance for use across the server
request_tracker = RequestTracker()


def get_request_tracker() -> RequestTracker:
    """Get the global request tracker instance.

    Returns:
        The global RequestTracker singleton
    """
    return request_tracker
