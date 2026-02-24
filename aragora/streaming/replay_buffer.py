"""
Enhanced event replay buffer for streaming reconnection.

Provides an ``EventReplayBuffer`` with configurable time windows, memory
bounds, and thread-safe ring buffer semantics.  Designed for the production
streaming path where clients reconnect after brief disconnections and need
to catch up on missed events.

Unlike the per-debate replay buffer in ``aragora.server.stream.replay_buffer``
(which stores ``(seq, json_str)`` tuples per debate), this module provides a
*global* event buffer with time-based pruning and memory bounds that can be
used by any streaming component.

Usage::

    from aragora.streaming.replay_buffer import EventReplayBuffer

    buf = EventReplayBuffer(max_events=1000, window_seconds=300)
    buf.append(seq=1, data='{"type":"agent_message",...}', debate_id="d-1")

    # Client reconnects and wants everything after seq 42
    missed = buf.replay_from_seq(debate_id="d-1", seq_number=42)

    # Periodic cleanup
    pruned = buf.prune_expired()
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_MAX_EVENTS = 1000
DEFAULT_WINDOW_SECONDS = 300.0  # 5 minutes
DEFAULT_MAX_BYTES = 50 * 1024 * 1024  # 50 MB


@dataclass(frozen=True)
class BufferedEvent:
    """A single buffered event with metadata for pruning.

    Attributes:
        seq: Monotonically increasing sequence number.
        data: Serialized event payload (JSON string).
        debate_id: Debate this event belongs to.
        timestamp: Monotonic time when the event was buffered.
        size_bytes: Approximate size of the serialized data.
    """

    seq: int
    data: str
    debate_id: str
    timestamp: float
    size_bytes: int


class EventReplayBuffer:
    """Thread-safe event replay buffer with time and memory bounds.

    Events are stored in a ring buffer (deque) ordered by sequence number.
    Each event is tagged with a debate ID, a timestamp, and a byte-size
    estimate.  The buffer enforces three limits:

    1. ``max_events`` -- hard cap on total events (oldest evicted first).
    2. ``window_seconds`` -- events older than this are eligible for pruning.
    3. ``max_bytes`` -- total memory cap; oldest events evicted when exceeded.

    Thread safety is provided via a ``threading.Lock``.  The lock is held
    only for short critical sections so blocking is minimal.
    """

    def __init__(
        self,
        *,
        max_events: int = DEFAULT_MAX_EVENTS,
        window_seconds: float = DEFAULT_WINDOW_SECONDS,
        max_bytes: int = DEFAULT_MAX_BYTES,
    ) -> None:
        self._max_events = max_events
        self._window_seconds = window_seconds
        self._max_bytes = max_bytes

        self._buffer: deque[BufferedEvent] = deque(maxlen=max_events)
        self._lock = threading.Lock()

        # Tracking
        self._current_bytes: int = 0
        self._total_appended: int = 0
        self._total_evicted: int = 0
        self._total_pruned: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def max_events(self) -> int:
        """Maximum number of events the buffer can hold."""
        return self._max_events

    @property
    def window_seconds(self) -> float:
        """Time window for event retention."""
        return self._window_seconds

    @property
    def max_bytes(self) -> int:
        """Maximum total bytes the buffer can hold."""
        return self._max_bytes

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def append(
        self,
        *,
        seq: int,
        data: str,
        debate_id: str,
    ) -> bool:
        """Append an event to the buffer.

        Returns ``True`` if the event was stored, ``False`` if it was
        rejected (e.g. duplicate or invalid).
        """
        if not data or not debate_id:
            return False

        size = sys.getsizeof(data)
        now = time.monotonic()
        event = BufferedEvent(
            seq=seq,
            data=data,
            debate_id=debate_id,
            timestamp=now,
            size_bytes=size,
        )

        with self._lock:
            # If the deque is at maxlen, the oldest will be auto-evicted
            if len(self._buffer) == self._max_events and self._buffer:
                evicted = self._buffer[0]  # Will be removed by deque
                self._current_bytes -= evicted.size_bytes
                self._total_evicted += 1

            self._buffer.append(event)
            self._current_bytes += size
            self._total_appended += 1

            # Enforce memory bound
            self._enforce_memory_limit_locked()

        return True

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def replay_from_seq(
        self,
        debate_id: str,
        seq_number: int,
    ) -> list[str]:
        """Return serialized events for *debate_id* with ``seq > seq_number``.

        Events are returned in sequence order.
        """
        with self._lock:
            return [
                ev.data for ev in self._buffer if ev.debate_id == debate_id and ev.seq > seq_number
            ]

    def get_latest_seq(self, debate_id: str) -> int:
        """Return the highest sequence number for *debate_id* (0 if empty)."""
        with self._lock:
            for ev in reversed(self._buffer):
                if ev.debate_id == debate_id:
                    return ev.seq
            return 0

    def get_oldest_seq(self, debate_id: str) -> int:
        """Return the lowest sequence number for *debate_id* (0 if empty)."""
        with self._lock:
            for ev in self._buffer:
                if ev.debate_id == debate_id:
                    return ev.seq
            return 0

    def count(self, debate_id: str | None = None) -> int:
        """Return the number of buffered events, optionally filtered by debate."""
        with self._lock:
            if debate_id is None:
                return len(self._buffer)
            return sum(1 for ev in self._buffer if ev.debate_id == debate_id)

    def current_bytes(self) -> int:
        """Return approximate total bytes of buffered data."""
        with self._lock:
            return self._current_bytes

    # ------------------------------------------------------------------
    # Pruning
    # ------------------------------------------------------------------

    def prune_expired(self) -> int:
        """Remove events older than ``window_seconds``.

        Returns the number of events pruned.
        """
        now = time.monotonic()
        cutoff = now - self._window_seconds
        pruned = 0

        with self._lock:
            while self._buffer and self._buffer[0].timestamp < cutoff:
                evicted = self._buffer.popleft()
                self._current_bytes -= evicted.size_bytes
                pruned += 1

            self._total_pruned += pruned

        if pruned > 0:
            logger.debug("[ReplayBuffer] Pruned %d expired events", pruned)

        return pruned

    def remove_debate(self, debate_id: str) -> int:
        """Remove all events for a specific debate.

        Returns the number of events removed.
        """
        with self._lock:
            before = len(self._buffer)
            remaining: deque[BufferedEvent] = deque(maxlen=self._max_events)
            kept_bytes = 0
            for ev in self._buffer:
                if ev.debate_id != debate_id:
                    remaining.append(ev)
                    kept_bytes += ev.size_bytes
            removed = before - len(remaining)
            self._buffer = remaining
            self._current_bytes = kept_bytes
            self._total_pruned += removed

        if removed > 0:
            logger.debug(
                "[ReplayBuffer] Removed %d events for debate %s",
                removed,
                debate_id,
            )

        return removed

    def clear(self) -> None:
        """Remove all buffered events."""
        with self._lock:
            self._buffer.clear()
            self._current_bytes = 0

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def get_metrics(self) -> dict[str, Any]:
        """Return buffer metrics for observability."""
        with self._lock:
            return {
                "buffered_events": len(self._buffer),
                "current_bytes": self._current_bytes,
                "max_events": self._max_events,
                "max_bytes": self._max_bytes,
                "window_seconds": self._window_seconds,
                "total_appended": self._total_appended,
                "total_evicted": self._total_evicted,
                "total_pruned": self._total_pruned,
            }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _enforce_memory_limit_locked(self) -> None:
        """Evict oldest events until under memory limit (caller holds lock)."""
        while self._current_bytes > self._max_bytes and self._buffer:
            evicted = self._buffer.popleft()
            self._current_bytes -= evicted.size_bytes
            self._total_evicted += 1


__all__ = [
    "BufferedEvent",
    "EventReplayBuffer",
]
