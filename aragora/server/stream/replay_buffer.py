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
import time
from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .events import StreamEvent

# Default buffer size per debate (overridable via env var).
# 1000 events covers ~5 minutes of a high-throughput debate with token
# streaming, giving reconnecting clients a generous replay window.
EVENT_REPLAY_BUFFER_SIZE = int(os.getenv("ARAGORA_EVENT_REPLAY_BUFFER_SIZE", "1000"))


class EventReplayBuffer:
    """Thread-safe per-debate ring buffer storing ``(seq, json_str)`` tuples.

    Each debate gets its own ``deque(maxlen=N)`` so only the most recent
    *N* events are retained.  Older events are silently dropped.

    Also tracks per-debate metrics (total appended, total dropped due to
    ring-buffer eviction) for connection quality reporting.
    """

    def __init__(self, max_per_debate: int = EVENT_REPLAY_BUFFER_SIZE) -> None:
        self._max = max_per_debate
        self._buffers: dict[str, deque[tuple[int, str]]] = {}
        self._lock = threading.Lock()
        # Per-debate metrics: {loop_id: {"appended": int, "evicted": int}}
        self._metrics: dict[str, dict[str, int]] = {}

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
                self._metrics[loop_id] = {"appended": 0, "evicted": 0}
            buf = self._buffers[loop_id]
            was_full = len(buf) == buf.maxlen
            buf.append((seq, serialized))
            metrics = self._metrics[loop_id]
            metrics["appended"] += 1
            if was_full:
                metrics["evicted"] += 1

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

    def get_oldest_seq(self, loop_id: str) -> int:
        """Return the lowest buffered sequence number for *loop_id* (0 if empty)."""
        with self._lock:
            buf = self._buffers.get(loop_id)
            if not buf:
                return 0
            return buf[0][0]

    def get_buffered_count(self, loop_id: str) -> int:
        """Return the number of events currently buffered for *loop_id*."""
        with self._lock:
            buf = self._buffers.get(loop_id)
            return len(buf) if buf else 0

    def get_metrics(self, loop_id: str) -> dict[str, int]:
        """Return buffer metrics for a debate (appended, evicted counts)."""
        with self._lock:
            return dict(self._metrics.get(loop_id, {"appended": 0, "evicted": 0}))

    # -- cleanup --------------------------------------------------------------

    def remove(self, loop_id: str) -> None:
        """Drop the buffer for a finished debate."""
        with self._lock:
            self._buffers.pop(loop_id, None)
            self._metrics.pop(loop_id, None)

    def cleanup_stale(self, active_loop_ids: set[str]) -> int:
        """Remove buffers for debates no longer active.  Returns count removed."""
        with self._lock:
            stale = [lid for lid in self._buffers if lid not in active_loop_ids]
            for lid in stale:
                del self._buffers[lid]
                self._metrics.pop(lid, None)
            return len(stale)


class ConnectionQualityTracker:
    """Tracks per-client connection quality metrics.

    Metrics tracked per client (by ws_id):
    - reconnect_count: number of reconnections
    - messages_received: total messages sent to client
    - replay_requests: number of replay requests
    - total_replayed: total events replayed
    - last_seen_seq: last sequence number the client acknowledged
    - connected_at: timestamp of initial connection
    - last_reconnect_at: timestamp of most recent reconnection
    - latency_samples: recent round-trip latency measurements (bounded)
    """

    _MAX_LATENCY_SAMPLES = 50

    def __init__(self) -> None:
        self._metrics: dict[int, dict] = {}
        self._lock = threading.Lock()

    def register(self, ws_id: int) -> None:
        """Register a new client connection."""
        with self._lock:
            self._metrics[ws_id] = {
                "reconnect_count": 0,
                "messages_received": 0,
                "replay_requests": 0,
                "total_replayed": 0,
                "last_seen_seq": 0,
                "connected_at": time.time(),
                "last_reconnect_at": None,
                "latency_samples": deque(maxlen=self._MAX_LATENCY_SAMPLES),
            }

    def unregister(self, ws_id: int) -> dict | None:
        """Unregister a client, returning its final metrics (or None)."""
        with self._lock:
            metrics = self._metrics.pop(ws_id, None)
            if metrics:
                # Convert deque to list for serialization
                metrics["latency_samples"] = list(metrics["latency_samples"])
            return metrics

    def record_reconnect(self, ws_id: int) -> None:
        """Record a client reconnection."""
        with self._lock:
            m = self._metrics.get(ws_id)
            if m:
                m["reconnect_count"] += 1
                m["last_reconnect_at"] = time.time()

    def record_replay(self, ws_id: int, event_count: int) -> None:
        """Record a replay request and the number of events replayed."""
        with self._lock:
            m = self._metrics.get(ws_id)
            if m:
                m["replay_requests"] += 1
                m["total_replayed"] += event_count

    def record_message_sent(self, ws_id: int) -> None:
        """Record that a message was sent to the client."""
        with self._lock:
            m = self._metrics.get(ws_id)
            if m:
                m["messages_received"] += 1

    def update_last_seen_seq(self, ws_id: int, seq: int) -> None:
        """Update the last sequence number acknowledged by the client."""
        with self._lock:
            m = self._metrics.get(ws_id)
            if m and seq > m["last_seen_seq"]:
                m["last_seen_seq"] = seq

    def record_latency(self, ws_id: int, latency_ms: float) -> None:
        """Record a round-trip latency measurement."""
        with self._lock:
            m = self._metrics.get(ws_id)
            if m:
                m["latency_samples"].append(latency_ms)

    def get_quality(self, ws_id: int) -> dict | None:
        """Return connection quality summary for a client."""
        with self._lock:
            m = self._metrics.get(ws_id)
            if not m:
                return None
            samples = list(m["latency_samples"])
            avg_latency = sum(samples) / len(samples) if samples else 0.0
            return {
                "reconnect_count": m["reconnect_count"],
                "messages_received": m["messages_received"],
                "replay_requests": m["replay_requests"],
                "total_replayed": m["total_replayed"],
                "last_seen_seq": m["last_seen_seq"],
                "connected_at": m["connected_at"],
                "last_reconnect_at": m["last_reconnect_at"],
                "avg_latency_ms": round(avg_latency, 2),
                "latency_sample_count": len(samples),
                "uptime_seconds": round(time.time() - m["connected_at"], 1),
            }

    def get_all_qualities(self) -> dict[int, dict]:
        """Return connection quality summaries for all clients."""
        with self._lock:
            ws_ids = list(self._metrics.keys())
        # Use get_quality (acquires lock each time) to avoid holding lock too long
        result = {}
        for ws_id in ws_ids:
            quality = self.get_quality(ws_id)
            if quality:
                result[ws_id] = quality
        return result


__all__ = [
    "EventReplayBuffer",
    "EVENT_REPLAY_BUFFER_SIZE",
    "ConnectionQualityTracker",
]
