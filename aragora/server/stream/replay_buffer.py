"""Per-debate event replay buffer for WebSocket reconnection support.

Stores recent events in a bounded ring buffer so that clients reconnecting
after a brief disconnect can request replay of missed events via
``replay_from_seq``.

Usage::

    buf = EventReplayBuffer()
    buf.append(event)                     # feed every broadcast event
    missed = buf.replay_since(loop_id, 42)  # get JSON strings for seq > 42
    buf.remove(loop_id)                    # cleanup on loop unregister
"""

from __future__ import annotations

import os
import threading
from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .events import StreamEvent

# Default buffer size per debate (overridable via env var)
EVENT_REPLAY_BUFFER_SIZE = int(
    os.getenv("ARAGORA_EVENT_REPLAY_BUFFER_SIZE", "200")
)


class EventReplayBuffer:
    """Thread-safe per-debate ring buffer storing ``(seq, json_str)`` tuples.

    Each debate gets its own ``deque(maxlen=N)`` so only the most recent
    *N* events are retained.  Older events are silently dropped.
    """

    def __init__(self, max_per_debate: int = EVENT_REPLAY_BUFFER_SIZE) -> None:
        self._max = max_per_debate
        self._buffers: dict[str, deque[tuple[int, str]]] = {}
        self._lock = threading.Lock()

    # -- write path -----------------------------------------------------------

    def append(self, event: StreamEvent) -> None:
        """Store an event in the appropriate debate buffer."""
        loop_id = event.loop_id
        if not loop_id:
            return
        seq = event.seq
        serialized = event.to_json()
        with self._lock:
            if loop_id not in self._buffers:
                self._buffers[loop_id] = deque(maxlen=self._max)
            self._buffers[loop_id].append((seq, serialized))

    # -- read path ------------------------------------------------------------

    def replay_since(self, loop_id: str, since_seq: int) -> list[str]:
        """Return JSON strings for all buffered events with ``seq > since_seq``."""
        with self._lock:
            buf = self._buffers.get(loop_id)
            if not buf:
                return []
            return [s for (seq, s) in buf if seq > since_seq]

    def get_latest_seq(self, loop_id: str) -> int:
        """Return the highest buffered sequence number for *loop_id* (0 if empty)."""
        with self._lock:
            buf = self._buffers.get(loop_id)
            if not buf:
                return 0
            return buf[-1][0]

    # -- cleanup --------------------------------------------------------------

    def remove(self, loop_id: str) -> None:
        """Drop the buffer for a finished debate."""
        with self._lock:
            self._buffers.pop(loop_id, None)

    def cleanup_stale(self, active_loop_ids: set[str]) -> int:
        """Remove buffers for debates no longer active.  Returns count removed."""
        with self._lock:
            stale = [lid for lid in self._buffers if lid not in active_loop_ids]
            for lid in stale:
                del self._buffers[lid]
            return len(stale)


__all__ = ["EventReplayBuffer", "EVENT_REPLAY_BUFFER_SIZE"]
