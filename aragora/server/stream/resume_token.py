"""
WebSocket Resume Token Support.

Enables clients to reconnect to a debate WebSocket and resume from the
last received event, avoiding data loss on transient disconnects.

Architecture:
  - Each StreamEvent already has a `seq` field (global sequence number).
  - On each event sent to a client, a resume_token is included.
  - The token encodes (debate_id, seq) with HMAC integrity.
  - On reconnect, the client sends the token; the server replays
    buffered events with seq > token.last_seq.
  - Event buffers are kept per-debate with bounded size and TTL.
"""

import hashlib
import hmac
import json
import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Configuration
RESUME_TOKEN_SECRET = os.environ.get("ARAGORA_RESUME_SECRET", "aragora-ws-resume-v1")
EVENT_BUFFER_MAX_SIZE = 500  # Max events buffered per debate
EVENT_BUFFER_TTL_SECONDS = 300  # 5 minutes
RESUME_TOKEN_TTL_SECONDS = 300  # Token valid for 5 minutes


@dataclass
class BufferedEvent:
    """An event stored in the replay buffer."""

    seq: int
    data: dict[str, Any]
    timestamp: float = field(default_factory=time.time)


class EventReplayBuffer:
    """Thread-safe bounded event buffer for a single debate.

    Stores recent events so that reconnecting clients can receive
    events they missed during a disconnect.
    """

    def __init__(
        self,
        debate_id: str,
        max_size: int = EVENT_BUFFER_MAX_SIZE,
        ttl_seconds: float = EVENT_BUFFER_TTL_SECONDS,
    ):
        self.debate_id = debate_id
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._buffer: deque[BufferedEvent] = deque(maxlen=max_size)
        self._lock = threading.Lock()

    def append(self, seq: int, event_data: dict[str, Any]) -> None:
        """Add an event to the buffer."""
        with self._lock:
            self._buffer.append(BufferedEvent(seq=seq, data=event_data))

    def get_events_after(self, last_seq: int) -> list[dict[str, Any]]:
        """Return all buffered events with seq > last_seq.

        Also evicts expired events.
        """
        cutoff = time.time() - self.ttl_seconds
        result = []
        with self._lock:
            for event in self._buffer:
                if event.timestamp < cutoff:
                    continue
                if event.seq > last_seq:
                    result.append(event.data)
        return result

    def __len__(self) -> int:
        with self._lock:
            return len(self._buffer)


class ResumeTokenManager:
    """Manages resume tokens and event replay buffers for all debates.

    Thread-safe singleton that tracks event buffers per debate and
    generates/validates resume tokens.
    """

    def __init__(self, secret: str = RESUME_TOKEN_SECRET):
        self._secret = secret.encode("utf-8")
        self._buffers: dict[str, EventReplayBuffer] = {}
        self._lock = threading.Lock()

    def get_or_create_buffer(self, debate_id: str) -> EventReplayBuffer:
        """Get or create an event buffer for a debate."""
        with self._lock:
            if debate_id not in self._buffers:
                self._buffers[debate_id] = EventReplayBuffer(debate_id)
            return self._buffers[debate_id]

    def remove_buffer(self, debate_id: str) -> None:
        """Remove a debate's buffer (e.g., when debate ends)."""
        with self._lock:
            self._buffers.pop(debate_id, None)

    def buffer_event(self, debate_id: str, seq: int, event_data: dict[str, Any]) -> None:
        """Buffer an event for potential replay."""
        buf = self.get_or_create_buffer(debate_id)
        buf.append(seq, event_data)

    def generate_token(self, debate_id: str, last_seq: int) -> str:
        """Generate an HMAC-signed resume token.

        Token format: base64(json({debate_id, last_seq, expires_at})).signature
        """
        expires_at = time.time() + RESUME_TOKEN_TTL_SECONDS
        payload = json.dumps(
            {
                "d": debate_id,
                "s": last_seq,
                "e": expires_at,
            },
            separators=(",", ":"),
        )

        sig = hmac.new(self._secret, payload.encode("utf-8"), hashlib.sha256).hexdigest()[:16]
        return f"{payload}.{sig}"

    def validate_token(self, token: str) -> tuple[str, int] | None:
        """Validate a resume token and return (debate_id, last_seq) or None.

        Returns None if token is invalid, expired, or tampered with.
        """
        parts = token.rsplit(".", 1)
        if len(parts) != 2:
            return None

        payload_str, sig = parts

        # Verify HMAC signature
        expected_sig = hmac.new(
            self._secret, payload_str.encode("utf-8"), hashlib.sha256
        ).hexdigest()[:16]
        if not hmac.compare_digest(sig, expected_sig):
            logger.warning("Invalid resume token signature")
            return None

        try:
            payload = json.loads(payload_str)
        except (json.JSONDecodeError, ValueError):
            return None

        debate_id = payload.get("d", "")
        last_seq = payload.get("s", 0)
        expires_at = payload.get("e", 0)

        if time.time() > expires_at:
            logger.debug("Resume token expired for debate %s", debate_id)
            return None

        return debate_id, last_seq

    def get_replay_events(self, token: str) -> list[dict[str, Any]] | None:
        """Validate token and return missed events, or None if token invalid."""
        result = self.validate_token(token)
        if result is None:
            return None

        debate_id, last_seq = result
        buf = self.get_or_create_buffer(debate_id)
        return buf.get_events_after(last_seq)

    def cleanup_expired(self) -> int:
        """Remove buffers for debates that have no recent events.

        Returns number of buffers removed.
        """
        cutoff = time.time() - EVENT_BUFFER_TTL_SECONDS * 2
        to_remove = []
        with self._lock:
            for debate_id, buf in self._buffers.items():
                with buf._lock:
                    if not buf._buffer or buf._buffer[-1].timestamp < cutoff:
                        to_remove.append(debate_id)
            for debate_id in to_remove:
                del self._buffers[debate_id]
        return len(to_remove)


# Module-level singleton
_manager: ResumeTokenManager | None = None
_manager_lock = threading.Lock()


def get_resume_token_manager() -> ResumeTokenManager:
    """Get the singleton ResumeTokenManager."""
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                _manager = ResumeTokenManager()
    return _manager


def inject_resume_token(event_data: dict[str, Any], debate_id: str) -> dict[str, Any]:
    """Add a resume_token to an event dict before sending to client.

    This should be called just before serializing an event for WebSocket send.
    The event must have a 'seq' field.
    """
    seq = event_data.get("seq", 0)
    if not seq:
        return event_data

    mgr = get_resume_token_manager()
    mgr.buffer_event(debate_id, seq, event_data)
    event_data["resume_token"] = mgr.generate_token(debate_id, seq)
    return event_data


__all__ = [
    "ResumeTokenManager",
    "EventReplayBuffer",
    "get_resume_token_manager",
    "inject_resume_token",
]
