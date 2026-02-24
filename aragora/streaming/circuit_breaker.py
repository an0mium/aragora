"""
Stream-specific circuit breaker with per-debate isolation.

Provides a ``StreamCircuitBreaker`` that wraps the core
:class:`~aragora.resilience.circuit_breaker.CircuitBreaker` with streaming-
specific semantics:

- Per-debate-id isolation (one debate's failures do not affect others).
- Configurable failure threshold and time window.
- Three states: CLOSED -> OPEN -> HALF_OPEN.
- Prometheus metrics emission on state transitions.

Usage::

    from aragora.streaming.circuit_breaker import StreamCircuitBreaker

    breaker = StreamCircuitBreaker()

    if breaker.can_send("debate-123"):
        try:
            await ws.send(event)
            breaker.record_success("debate-123")
        except Exception:
            breaker.record_failure("debate-123")
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class StreamCircuitState(str, Enum):
    """Possible states for a stream circuit."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class StreamCircuitBreakerConfig:
    """Configuration for a stream circuit breaker.

    Attributes:
        failure_threshold: Number of failures within ``failure_window_seconds``
            to trip the breaker open.
        failure_window_seconds: Rolling window for counting failures.
        cooldown_seconds: How long the breaker stays open before transitioning
            to half-open.
        half_open_max_attempts: Number of trial sends in half-open before
            deciding to close or reopen.
        half_open_success_threshold: Consecutive successes needed to close
            from half-open.
    """

    failure_threshold: int = 5
    failure_window_seconds: float = 60.0
    cooldown_seconds: float = 30.0
    half_open_max_attempts: int = 3
    half_open_success_threshold: int = 2


@dataclass
class _DebateCircuitState:
    """Internal per-debate circuit state."""

    state: StreamCircuitState = StreamCircuitState.CLOSED
    failure_timestamps: list[float] = field(default_factory=list)
    opened_at: float = 0.0
    half_open_attempts: int = 0
    half_open_successes: int = 0
    total_failures: int = 0
    total_successes: int = 0
    total_rejected: int = 0
    transition_count: int = 0


# Metrics callback type: (debate_id, old_state, new_state) -> None
MetricsCallback = Any


class StreamCircuitBreaker:
    """Per-debate stream circuit breaker.

    Each debate ID gets its own isolated circuit, so failures in one debate
    do not trip the breaker for other debates.

    Example::

        breaker = StreamCircuitBreaker(
            config=StreamCircuitBreakerConfig(
                failure_threshold=5,
                cooldown_seconds=30.0,
            )
        )

        if breaker.can_send("debate-abc"):
            # ... send event ...
            breaker.record_success("debate-abc")
        else:
            # Circuit is open; skip or buffer
            pass
    """

    def __init__(
        self,
        config: StreamCircuitBreakerConfig | None = None,
        *,
        metrics_callback: MetricsCallback | None = None,
    ) -> None:
        self._config = config or StreamCircuitBreakerConfig()
        self._states: dict[str, _DebateCircuitState] = {}
        self._lock = threading.Lock()
        self._metrics_callback = metrics_callback

    @property
    def config(self) -> StreamCircuitBreakerConfig:
        """Return the active configuration."""
        return self._config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def can_send(self, debate_id: str) -> bool:
        """Check whether sending to *debate_id* is allowed.

        Returns ``True`` when the circuit is CLOSED or HALF_OPEN (trial).
        Returns ``False`` when the circuit is OPEN.
        """
        with self._lock:
            cs = self._get_or_create(debate_id)
            return self._can_proceed_locked(debate_id, cs)

    def record_failure(self, debate_id: str) -> bool:
        """Record a send failure for *debate_id*.

        Returns ``True`` if the circuit just transitioned to OPEN.
        """
        with self._lock:
            cs = self._get_or_create(debate_id)
            cs.total_failures += 1

            now = time.monotonic()
            cs.failure_timestamps.append(now)

            # Prune old failures outside the window
            cutoff = now - self._config.failure_window_seconds
            cs.failure_timestamps = [ts for ts in cs.failure_timestamps if ts > cutoff]

            if cs.state == StreamCircuitState.HALF_OPEN:
                # Any failure in half-open reopens the circuit
                self._transition(debate_id, cs, StreamCircuitState.OPEN)
                cs.opened_at = now
                cs.half_open_attempts = 0
                cs.half_open_successes = 0
                return True

            if cs.state == StreamCircuitState.CLOSED:
                if len(cs.failure_timestamps) >= self._config.failure_threshold:
                    self._transition(debate_id, cs, StreamCircuitState.OPEN)
                    cs.opened_at = now
                    return True

            return False

    def record_success(self, debate_id: str) -> bool:
        """Record a send success for *debate_id*.

        Returns ``True`` if the circuit just transitioned to CLOSED from
        HALF_OPEN.
        """
        with self._lock:
            cs = self._get_or_create(debate_id)
            cs.total_successes += 1

            if cs.state == StreamCircuitState.HALF_OPEN:
                cs.half_open_successes += 1
                if cs.half_open_successes >= self._config.half_open_success_threshold:
                    self._transition(debate_id, cs, StreamCircuitState.CLOSED)
                    cs.failure_timestamps.clear()
                    cs.half_open_attempts = 0
                    cs.half_open_successes = 0
                    return True
            elif cs.state == StreamCircuitState.CLOSED:
                # Reset failure timestamps on success in closed state
                cs.failure_timestamps.clear()

            return False

    def get_state(self, debate_id: str) -> StreamCircuitState:
        """Return the current circuit state for *debate_id*."""
        with self._lock:
            cs = self._states.get(debate_id)
            if cs is None:
                return StreamCircuitState.CLOSED
            self._maybe_half_open(debate_id, cs)
            return cs.state

    def get_stats(self, debate_id: str) -> dict[str, Any]:
        """Return circuit statistics for *debate_id*."""
        with self._lock:
            cs = self._states.get(debate_id)
            if cs is None:
                return {
                    "state": StreamCircuitState.CLOSED.value,
                    "total_failures": 0,
                    "total_successes": 0,
                    "total_rejected": 0,
                    "transition_count": 0,
                    "recent_failures": 0,
                }
            self._maybe_half_open(debate_id, cs)
            now = time.monotonic()
            cutoff = now - self._config.failure_window_seconds
            recent = len([ts for ts in cs.failure_timestamps if ts > cutoff])
            return {
                "state": cs.state.value,
                "total_failures": cs.total_failures,
                "total_successes": cs.total_successes,
                "total_rejected": cs.total_rejected,
                "transition_count": cs.transition_count,
                "recent_failures": recent,
            }

    def get_all_stats(self) -> dict[str, dict[str, Any]]:
        """Return circuit statistics for all known debates."""
        with self._lock:
            debate_ids = list(self._states.keys())
        return {did: self.get_stats(did) for did in debate_ids}

    def reset(self, debate_id: str | None = None) -> None:
        """Reset circuit state.

        If *debate_id* is ``None``, resets all circuits.
        """
        with self._lock:
            if debate_id is None:
                self._states.clear()
                logger.info("[StreamCB] Reset all circuit breakers")
            else:
                self._states.pop(debate_id, None)
                logger.info("[StreamCB] Reset circuit for %s", debate_id)

    def remove(self, debate_id: str) -> None:
        """Remove tracking for a finished debate."""
        with self._lock:
            self._states.pop(debate_id, None)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_create(self, debate_id: str) -> _DebateCircuitState:
        """Get or create state for a debate (caller must hold lock)."""
        if debate_id not in self._states:
            self._states[debate_id] = _DebateCircuitState()
        return self._states[debate_id]

    def _can_proceed_locked(self, debate_id: str, cs: _DebateCircuitState) -> bool:
        """Check if sending is allowed (caller must hold lock)."""
        self._maybe_half_open(debate_id, cs)

        if cs.state == StreamCircuitState.CLOSED:
            return True

        if cs.state == StreamCircuitState.HALF_OPEN:
            if cs.half_open_attempts < self._config.half_open_max_attempts:
                cs.half_open_attempts += 1
                return True
            # Already hit max attempts in half-open without enough successes
            cs.total_rejected += 1
            return False

        # OPEN
        cs.total_rejected += 1
        return False

    def _maybe_half_open(self, debate_id: str, cs: _DebateCircuitState) -> None:
        """Transition OPEN -> HALF_OPEN if cooldown has elapsed (caller holds lock)."""
        if cs.state != StreamCircuitState.OPEN:
            return
        elapsed = time.monotonic() - cs.opened_at
        if elapsed >= self._config.cooldown_seconds:
            self._transition(debate_id, cs, StreamCircuitState.HALF_OPEN)
            cs.half_open_attempts = 0
            cs.half_open_successes = 0

    def _transition(
        self,
        debate_id: str,
        cs: _DebateCircuitState,
        new_state: StreamCircuitState,
    ) -> None:
        """Perform a state transition with logging and metrics (caller holds lock)."""
        old_state = cs.state
        if old_state == new_state:
            return
        cs.state = new_state
        cs.transition_count += 1

        logger.info(
            "[StreamCB] %s: %s -> %s",
            debate_id,
            old_state.value,
            new_state.value,
        )

        # Emit metrics
        self._emit_transition_metric(debate_id, old_state, new_state)

    def _emit_transition_metric(
        self,
        debate_id: str,
        old_state: StreamCircuitState,
        new_state: StreamCircuitState,
    ) -> None:
        """Emit a metrics callback for a state transition."""
        if self._metrics_callback is not None:
            try:
                self._metrics_callback(debate_id, old_state, new_state)
            except Exception:  # noqa: BLE001 - metrics must never break callers
                logger.debug("[StreamCB] Metrics callback error for %s", debate_id)

        # Also try Prometheus metrics
        try:
            from aragora.observability.metrics.base import get_or_create_counter

            counter = get_or_create_counter(
                "aragora_stream_circuit_transitions_total",
                "Stream circuit breaker state transitions",
                ["debate_id", "from_state", "to_state"],
            )
            counter.labels(
                debate_id=debate_id,
                from_state=old_state.value,
                to_state=new_state.value,
            ).inc()
        except (ImportError, RuntimeError, TypeError, ValueError):
            pass  # Prometheus not available


__all__ = [
    "StreamCircuitBreaker",
    "StreamCircuitBreakerConfig",
    "StreamCircuitState",
]
