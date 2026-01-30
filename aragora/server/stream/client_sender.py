"""
Timeout-safe WebSocket client sender with health tracking.

Solves the problem where slow clients block the entire drain loop by:
1. Wrapping all send operations with configurable timeouts
2. Tracking client health via consecutive failure counts
3. Quarantining slow clients that exceed failure thresholds

Usage:
    from aragora.server.stream.client_sender import TimeoutSender, ClientHealth

    sender = TimeoutSender(timeout=2.0, max_failures=3)

    # Send with timeout
    success = await sender.send(client, message)

    # Check client health
    if sender.is_quarantined(client):
        await cleanup_client(client)
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class ClientStatus(Enum):
    """Health status of a WebSocket client."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"  # Some failures, but not quarantined
    QUARANTINED = "quarantined"  # Too many failures, skip sending


@dataclass
class ClientHealth:
    """Tracks health metrics for a single WebSocket client.

    Attributes:
        client_id: Unique identifier for the client (usually id(ws))
        consecutive_failures: Number of consecutive send failures
        total_failures: Total failures since connection
        total_sends: Total send attempts
        last_failure_time: Timestamp of last failure
        last_success_time: Timestamp of last successful send
        quarantined_until: If set, client is quarantined until this time
        avg_latency_ms: Rolling average send latency in milliseconds
    """

    client_id: int
    consecutive_failures: int = 0
    total_failures: int = 0
    total_sends: int = 0
    last_failure_time: float | None = None
    last_success_time: float | None = None
    quarantined_until: float | None = None
    avg_latency_ms: float = 0.0
    _latency_samples: list[float] = field(default_factory=list)

    @property
    def status(self) -> ClientStatus:
        """Current health status."""
        if self.quarantined_until and time.time() < self.quarantined_until:
            return ClientStatus.QUARANTINED
        if self.consecutive_failures >= 2:
            return ClientStatus.DEGRADED
        return ClientStatus.HEALTHY

    @property
    def failure_rate(self) -> float:
        """Failure rate as percentage."""
        if self.total_sends == 0:
            return 0.0
        return (self.total_failures / self.total_sends) * 100

    def record_success(self, latency_ms: float) -> None:
        """Record a successful send."""
        self.consecutive_failures = 0
        self.total_sends += 1
        self.last_success_time = time.time()
        self.quarantined_until = None  # Clear quarantine on success

        # Update rolling average (keep last 10 samples)
        self._latency_samples.append(latency_ms)
        if len(self._latency_samples) > 10:
            self._latency_samples.pop(0)
        self.avg_latency_ms = sum(self._latency_samples) / len(self._latency_samples)

    def record_failure(self, quarantine_duration: float = 0.0) -> None:
        """Record a failed send.

        Args:
            quarantine_duration: If > 0, quarantine client for this many seconds
        """
        self.consecutive_failures += 1
        self.total_failures += 1
        self.total_sends += 1
        self.last_failure_time = time.time()

        if quarantine_duration > 0:
            self.quarantined_until = time.time() + quarantine_duration


class TimeoutSender:
    """
    Timeout-safe WebSocket sender with client health tracking.

    Wraps send operations with timeouts to prevent slow clients from
    blocking the drain loop. Tracks health per client and quarantines
    clients that consistently fail.

    Attributes:
        timeout: Default send timeout in seconds
        max_failures: Consecutive failures before quarantine
        quarantine_duration: How long to quarantine slow clients (seconds)
    """

    def __init__(
        self,
        timeout: float = 2.0,
        max_failures: int = 3,
        quarantine_duration: float = 10.0,
    ):
        """Initialize the timeout sender.

        Args:
            timeout: Send timeout in seconds (default: 2.0)
            max_failures: Consecutive failures before quarantine (default: 3)
            quarantine_duration: Quarantine duration in seconds (default: 10.0)
        """
        self._timeout = timeout
        self._max_failures = max_failures
        self._quarantine_duration = quarantine_duration
        self._client_health: dict[int, ClientHealth] = {}
        self._lock = asyncio.Lock()

        # Metrics
        self._total_timeouts = 0
        self._total_quarantines = 0
        self._total_sends = 0
        self._total_failures = 0

    async def send(
        self,
        client: Any,
        message: str,
        timeout: float | None = None,
    ) -> bool:
        """Send a message to a client with timeout protection.

        Args:
            client: WebSocket client (aiohttp WebSocketResponse or similar)
            message: Message string to send
            timeout: Override default timeout (optional)

        Returns:
            True if sent successfully, False if failed or skipped
        """
        client_id = id(client)
        send_timeout = timeout if timeout is not None else self._timeout

        # Get or create health record
        health = self._get_or_create_health(client_id)

        # Skip quarantined clients
        if health.status == ClientStatus.QUARANTINED:
            logger.debug(
                "Skipping quarantined client %d (until %.1fs)",
                client_id,
                (health.quarantined_until or 0) - time.time(),
            )
            return False

        # Attempt send with timeout
        start_time = time.time()
        self._total_sends += 1

        try:
            await asyncio.wait_for(
                self._do_send(client, message),
                timeout=send_timeout,
            )

            # Record success
            latency_ms = (time.time() - start_time) * 1000
            health.record_success(latency_ms)
            return True

        except asyncio.TimeoutError:
            self._total_timeouts += 1
            self._total_failures += 1
            logger.warning(
                "WebSocket send timeout for client %d (%.1fs)",
                client_id,
                send_timeout,
            )
            self._handle_failure(health)
            return False

        except (OSError, ConnectionError, RuntimeError) as e:
            self._total_failures += 1
            logger.debug(
                "WebSocket send error for client %d: %s",
                client_id,
                type(e).__name__,
            )
            self._handle_failure(health)
            return False

    async def send_many(
        self,
        clients: list[Any],
        message: str,
        timeout: float | None = None,
    ) -> tuple[int, list[Any]]:
        """Send a message to multiple clients with timeout protection.

        Uses asyncio.gather for concurrent sends, but each client has
        its own timeout to prevent one slow client from affecting others.

        Args:
            clients: List of WebSocket clients
            message: Message string to send
            timeout: Override default timeout (optional)

        Returns:
            Tuple of (success_count, list of failed/dead clients)
        """
        if not clients:
            return 0, []

        # Send to all clients concurrently
        results = await asyncio.gather(
            *[self.send(client, message, timeout) for client in clients],
            return_exceptions=True,
        )

        success_count = 0
        dead_clients = []

        for client, result in zip(clients, results):
            if result is True:
                success_count += 1
            else:
                # Check if client should be considered dead
                health = self._get_or_create_health(id(client))
                if health.consecutive_failures >= self._max_failures:
                    dead_clients.append(client)

        return success_count, dead_clients

    def is_quarantined(self, client: Any) -> bool:
        """Check if a client is currently quarantined.

        Args:
            client: WebSocket client

        Returns:
            True if quarantined, False otherwise
        """
        health = self._client_health.get(id(client))
        if health is None:
            return False
        return health.status == ClientStatus.QUARANTINED

    def is_dead(self, client: Any) -> bool:
        """Check if a client should be considered dead (too many failures).

        Args:
            client: WebSocket client

        Returns:
            True if client exceeded max failures and should be removed
        """
        health = self._client_health.get(id(client))
        if health is None:
            return False
        return health.consecutive_failures >= self._max_failures

    def get_health(self, client: Any) -> ClientHealth | None:
        """Get health record for a client.

        Args:
            client: WebSocket client

        Returns:
            ClientHealth or None if not tracked
        """
        return self._client_health.get(id(client))

    def remove_client(self, client: Any) -> None:
        """Remove a client from health tracking.

        Args:
            client: WebSocket client to remove
        """
        client_id = id(client)
        self._client_health.pop(client_id, None)

    def get_stats(self) -> dict[str, Any]:
        """Get sender statistics.

        Returns:
            Dict with metrics about sends, timeouts, quarantines
        """
        quarantined_count = sum(
            1 for h in self._client_health.values() if h.status == ClientStatus.QUARANTINED
        )
        degraded_count = sum(
            1 for h in self._client_health.values() if h.status == ClientStatus.DEGRADED
        )

        return {
            "total_sends": self._total_sends,
            "total_failures": self._total_failures,
            "total_timeouts": self._total_timeouts,
            "total_quarantines": self._total_quarantines,
            "tracked_clients": len(self._client_health),
            "quarantined_clients": quarantined_count,
            "degraded_clients": degraded_count,
            "failure_rate": (
                (self._total_failures / self._total_sends * 100) if self._total_sends > 0 else 0.0
            ),
        }

    def _get_or_create_health(self, client_id: int) -> ClientHealth:
        """Get or create health record for a client."""
        if client_id not in self._client_health:
            self._client_health[client_id] = ClientHealth(client_id=client_id)
        return self._client_health[client_id]

    def _handle_failure(self, health: ClientHealth) -> None:
        """Handle a send failure, potentially quarantining the client."""
        if health.consecutive_failures >= self._max_failures:
            # Quarantine the client
            health.record_failure(quarantine_duration=self._quarantine_duration)
            self._total_quarantines += 1
            logger.warning(
                "Quarantined slow client %d for %.1fs (failures: %d)",
                health.client_id,
                self._quarantine_duration,
                health.consecutive_failures,
            )
        else:
            health.record_failure()

    async def _do_send(self, client: Any, message: str) -> None:
        """Execute the actual send operation.

        Supports both aiohttp WebSocketResponse and websockets library.
        """
        # aiohttp WebSocketResponse
        if hasattr(client, "send_str"):
            await client.send_str(message)
        # websockets library
        elif hasattr(client, "send"):
            await client.send(message)
        else:
            raise TypeError(f"Unsupported WebSocket client type: {type(client)}")


# Global sender instance (can be overridden in tests)
_default_sender: TimeoutSender | None = None


def get_timeout_sender(
    timeout: float = 2.0,
    max_failures: int = 3,
    quarantine_duration: float = 10.0,
) -> TimeoutSender:
    """Get or create the global timeout sender.

    Args:
        timeout: Send timeout in seconds
        max_failures: Consecutive failures before quarantine
        quarantine_duration: Quarantine duration in seconds

    Returns:
        TimeoutSender instance
    """
    global _default_sender
    if _default_sender is None:
        _default_sender = TimeoutSender(
            timeout=timeout,
            max_failures=max_failures,
            quarantine_duration=quarantine_duration,
        )
    return _default_sender


def reset_timeout_sender() -> None:
    """Reset the global timeout sender (for testing)."""
    global _default_sender
    _default_sender = None
