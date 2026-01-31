"""
Drain Coordinator for Zero-Downtime Deployments.

Manages the transition from ``serving`` to ``stopped`` through a ``draining``
phase that allows load balancers to remove the instance before the shutdown
sequence begins.

State machine::

    SERVING ──► DRAINING ──► SHUTTING_DOWN ──► STOPPED
                  │                │
                  │ (drain_seconds) │ (shutdown_sequence.py)
                  ▼                ▼
           health → 503       connections closed

During the ``DRAINING`` phase:
- ``/healthz`` and ``/readyz`` return **503**, signalling the load
  balancer to stop routing new traffic.
- Existing in-flight requests are allowed to complete.
- After ``drain_seconds`` the coordinator advances to ``SHUTTING_DOWN``
  and delegates to the existing :class:`ShutdownSequence`.

Usage:
    from aragora.server.drain_coordinator import (
        DrainCoordinator,
        ServerState,
        get_drain_coordinator,
    )

    coordinator = get_drain_coordinator()

    # In health endpoint
    if coordinator.is_healthy():
        return 200, {"status": "ok"}
    else:
        return 503, {"status": coordinator.state.value}

    # On SIGTERM
    await coordinator.begin_drain()
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine

logger = logging.getLogger(__name__)

# Configurable via environment
DEFAULT_DRAIN_SECONDS = float(os.environ.get("ARAGORA_DRAIN_SECONDS", "10"))
DEFAULT_SHUTDOWN_TIMEOUT = float(os.environ.get("ARAGORA_SHUTDOWN_TIMEOUT", "30"))


class ServerState(Enum):
    """Server lifecycle states."""

    STARTING = "starting"
    SERVING = "serving"
    DRAINING = "draining"
    SHUTTING_DOWN = "shutting_down"
    STOPPED = "stopped"


@dataclass
class DrainStats:
    """Statistics about the drain/shutdown process."""

    state_transitions: list[tuple[str, float]] = field(default_factory=list)
    drain_started_at: float | None = None
    shutdown_started_at: float | None = None
    stopped_at: float | None = None
    in_flight_at_drain: int = 0
    in_flight_at_shutdown: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_transitions": [{"state": s, "timestamp": t} for s, t in self.state_transitions],
            "drain_started_at": self.drain_started_at,
            "shutdown_started_at": self.shutdown_started_at,
            "stopped_at": self.stopped_at,
            "in_flight_at_drain": self.in_flight_at_drain,
            "in_flight_at_shutdown": self.in_flight_at_shutdown,
            "total_drain_seconds": (
                (self.shutdown_started_at - self.drain_started_at)
                if self.drain_started_at and self.shutdown_started_at
                else None
            ),
            "total_shutdown_seconds": (
                (self.stopped_at - self.shutdown_started_at)
                if self.shutdown_started_at and self.stopped_at
                else None
            ),
        }


class DrainCoordinator:
    """Coordinates the drain → shutdown lifecycle.

    Integrates with the existing :class:`ShutdownSequence` by inserting
    a drain period before the shutdown phases run.

    Attributes:
        state: Current server state.
        drain_seconds: How long to stay in DRAINING before shutting down.
    """

    def __init__(
        self,
        drain_seconds: float = DEFAULT_DRAIN_SECONDS,
        shutdown_timeout: float = DEFAULT_SHUTDOWN_TIMEOUT,
    ) -> None:
        self._state = ServerState.STARTING
        self._drain_seconds = drain_seconds
        self._shutdown_timeout = shutdown_timeout
        self._stats = DrainStats()

        # In-flight request counter
        self._in_flight = 0
        self._in_flight_lock = asyncio.Lock()

        # Optional shutdown callback
        self._shutdown_fn: Callable[[], Coroutine[Any, Any, Any]] | None = None

        # Callbacks notified on state changes
        self._state_callbacks: list[Callable[[ServerState], None]] = []

        self._record_transition(ServerState.STARTING)

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    @property
    def state(self) -> ServerState:
        """Current server state."""
        return self._state

    def is_healthy(self) -> bool:
        """Return ``True`` if the server should accept new requests.

        Load balancer health checks should call this.  Returns ``False``
        once draining begins, causing the LB to remove this instance.
        """
        return self._state == ServerState.SERVING

    def is_ready(self) -> bool:
        """Return ``True`` if the server is ready to handle requests.

        More permissive than :meth:`is_healthy` — allows in-flight
        requests during drain.
        """
        return self._state in (ServerState.SERVING, ServerState.DRAINING)

    def is_alive(self) -> bool:
        """Return ``True`` if the server process is alive (liveness probe).

        Only returns ``False`` when fully stopped.
        """
        return self._state != ServerState.STOPPED

    # ------------------------------------------------------------------
    # In-flight tracking
    # ------------------------------------------------------------------

    async def request_started(self) -> bool:
        """Track a new in-flight request.

        Returns:
            ``True`` if the request should proceed.
            ``False`` if the server is shutting down and the request
            should be rejected with 503.
        """
        if self._state in (ServerState.SHUTTING_DOWN, ServerState.STOPPED):
            return False
        async with self._in_flight_lock:
            self._in_flight += 1
        return True

    async def request_finished(self) -> None:
        """Mark an in-flight request as complete."""
        async with self._in_flight_lock:
            self._in_flight = max(0, self._in_flight - 1)

    @property
    def in_flight_count(self) -> int:
        """Number of currently in-flight requests."""
        return self._in_flight

    # ------------------------------------------------------------------
    # Lifecycle transitions
    # ------------------------------------------------------------------

    def mark_serving(self) -> None:
        """Transition from STARTING to SERVING (called after startup)."""
        if self._state == ServerState.STARTING:
            self._set_state(ServerState.SERVING)
            logger.info("Server is now SERVING")

    async def begin_drain(self) -> dict[str, Any]:
        """Begin the drain → shutdown lifecycle.

        1. Transitions to DRAINING (health checks return 503).
        2. Waits ``drain_seconds`` for the load balancer to react.
        3. Transitions to SHUTTING_DOWN and runs the shutdown callback.
        4. Transitions to STOPPED.

        Returns:
            Summary dict with timing and in-flight counts.
        """
        if self._state not in (ServerState.SERVING, ServerState.STARTING):
            logger.warning("begin_drain called in state %s; ignoring", self._state.value)
            return self._stats.to_dict()

        # --- DRAINING ---
        self._set_state(ServerState.DRAINING)
        self._stats.drain_started_at = time.time()
        self._stats.in_flight_at_drain = self._in_flight
        logger.info(
            "Entering DRAINING state (drain_seconds=%.1f, in_flight=%d)",
            self._drain_seconds,
            self._in_flight,
        )

        # Wait for drain period
        await asyncio.sleep(self._drain_seconds)

        # --- SHUTTING_DOWN ---
        self._set_state(ServerState.SHUTTING_DOWN)
        self._stats.shutdown_started_at = time.time()
        self._stats.in_flight_at_shutdown = self._in_flight
        logger.info("Entering SHUTTING_DOWN state (in_flight=%d)", self._in_flight)

        # Run shutdown callback if registered
        if self._shutdown_fn is not None:
            try:
                await asyncio.wait_for(
                    self._shutdown_fn(),
                    timeout=self._shutdown_timeout,
                )
            except asyncio.TimeoutError:
                logger.error(
                    "Shutdown callback timed out after %.1fs",
                    self._shutdown_timeout,
                )
            except Exception:
                logger.exception("Shutdown callback failed")

        # --- STOPPED ---
        self._set_state(ServerState.STOPPED)
        self._stats.stopped_at = time.time()
        logger.info("Server is now STOPPED")

        return self._stats.to_dict()

    def set_shutdown_callback(self, fn: Callable[[], Coroutine[Any, Any, Any]]) -> None:
        """Register the async function to call during SHUTTING_DOWN.

        Typically this wraps ``ShutdownSequence.execute_all()``.
        """
        self._shutdown_fn = fn

    def on_state_change(self, callback: Callable[[ServerState], None]) -> None:
        """Register a callback invoked on every state transition."""
        self._state_callbacks.append(callback)

    # ------------------------------------------------------------------
    # Stats / health endpoint data
    # ------------------------------------------------------------------

    def health_response(self) -> tuple[dict[str, Any], int]:
        """Generate a health check response and HTTP status code.

        Returns:
            (body, status_code) — 200 when serving, 503 otherwise.
        """
        body: dict[str, Any] = {
            "status": self._state.value,
            "in_flight": self._in_flight,
        }
        if self._state == ServerState.SERVING:
            return body, 200
        return body, 503

    def get_stats(self) -> dict[str, Any]:
        """Return drain/shutdown statistics."""
        data = self._stats.to_dict()
        data["current_state"] = self._state.value
        data["in_flight"] = self._in_flight
        data["drain_seconds"] = self._drain_seconds
        data["shutdown_timeout"] = self._shutdown_timeout
        return data

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _set_state(self, new_state: ServerState) -> None:
        self._state = new_state
        self._stats.state_transitions.append((new_state.value, time.time()))
        for cb in self._state_callbacks:
            try:
                cb(new_state)
            except Exception:
                logger.debug("State change callback failed", exc_info=True)

    def _record_transition(self, state: ServerState) -> None:
        self._stats.state_transitions.append((state.value, time.time()))


# ---------------------------------------------------------------------------
# Global instance
# ---------------------------------------------------------------------------

_coordinator: DrainCoordinator | None = None


def get_drain_coordinator() -> DrainCoordinator:
    """Get or create the global drain coordinator."""
    global _coordinator
    if _coordinator is None:
        _coordinator = DrainCoordinator()
    return _coordinator


def reset_drain_coordinator() -> None:
    """Reset the global drain coordinator (for testing)."""
    global _coordinator
    _coordinator = None


__all__ = [
    "DrainCoordinator",
    "DrainStats",
    "ServerState",
    "get_drain_coordinator",
    "reset_drain_coordinator",
]
