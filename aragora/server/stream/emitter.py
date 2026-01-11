"""
Event emitter and audience participation classes.

Provides the SyncEventEmitter which bridges synchronous Arena code with
async WebSocket broadcasts, along with audience participation classes
for user votes and suggestions.
"""

import logging
import os
import queue
import threading
import time
from collections import deque
from typing import Any, Callable, Optional

from aragora.config import MAX_EVENT_QUEUE_SIZE
from aragora.server.stream.events import (
    StreamEvent,
    StreamEventType,
    AudienceMessage,
)

logger = logging.getLogger(__name__)


def normalize_intensity(value: Any, default: int = 5, min_val: int = 1, max_val: int = 10) -> int:
    """
    Safely normalize vote intensity to a clamped integer.

    Args:
        value: Raw intensity value from user input (may be string, float, None, etc.)
        default: Default intensity if value is invalid
        min_val: Minimum allowed intensity
        max_val: Maximum allowed intensity

    Returns:
        Clamped integer intensity between min_val and max_val
    """
    if value is None:
        return default

    try:
        intensity = int(float(value))
    except (ValueError, TypeError):
        return default

    return max(min_val, min(max_val, intensity))


class TokenBucket:
    """
    Token bucket rate limiter for audience message throttling.

    Allows burst traffic up to burst_size, then limits to rate_per_minute.
    Thread-safe for concurrent access.
    """

    def __init__(self, rate_per_minute: float, burst_size: int):
        """
        Initialize token bucket.

        Args:
            rate_per_minute: Token refill rate (tokens per minute)
            burst_size: Maximum tokens (bucket capacity)
        """
        self.rate_per_minute = rate_per_minute
        self.burst_size = burst_size
        self.tokens = float(burst_size)  # Start full
        self.last_refill = time.monotonic()
        self._lock = threading.Lock()

    def consume(self, tokens: int = 1) -> bool:
        """
        Attempt to consume tokens from the bucket.

        Args:
            tokens: Number of tokens to consume

        Returns:
            True if tokens were available and consumed, False otherwise
        """
        with self._lock:
            # Refill tokens based on elapsed time
            now = time.monotonic()
            elapsed_minutes = (now - self.last_refill) / 60.0
            refill_amount = elapsed_minutes * self.rate_per_minute
            self.tokens = min(self.burst_size, self.tokens + refill_amount)
            self.last_refill = now

            # Try to consume
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False


class AudienceInbox:
    """
    Thread-safe queue for audience messages.

    Collects votes and suggestions from WebSocket clients for processing
    by the debate arena.

    Uses a bounded deque to prevent unbounded memory growth under spam.
    Maximum size is configurable via ARAGORA_AUDIENCE_INBOX_MAX_SIZE env var.
    """

    # Maximum messages to retain (prevents memory exhaustion under spam)
    # Configurable via environment variable
    MAX_MESSAGES = int(os.environ.get("ARAGORA_AUDIENCE_INBOX_MAX_SIZE", "1000"))

    def __init__(self, max_messages: int | None = None):
        """
        Initialize audience inbox.

        Args:
            max_messages: Maximum messages to retain (defaults to MAX_MESSAGES)
        """
        max_size = max_messages if max_messages is not None else self.MAX_MESSAGES
        self._messages: deque[AudienceMessage] = deque(maxlen=max_size)
        self._lock = threading.Lock()
        self._overflow_count = 0  # Track dropped messages for monitoring

    def put(self, message: AudienceMessage) -> None:
        """Add a message to the inbox (thread-safe).

        If the inbox is at capacity, the oldest message is automatically dropped.
        """
        with self._lock:
            was_full = len(self._messages) == self._messages.maxlen
            self._messages.append(message)
            if was_full:
                self._overflow_count += 1
                if self._overflow_count % 100 == 1:  # Log every 100 drops
                    logger.warning(
                        f"[audience] Inbox overflow, dropping old messages "
                        f"(total dropped: {self._overflow_count})"
                    )

    def get_all(self) -> list[AudienceMessage]:
        """
        Drain all messages from the inbox (thread-safe).

        Returns:
            List of all queued messages, emptying the inbox
        """
        with self._lock:
            messages = list(self._messages)
            self._messages.clear()
            return messages

    def get_summary(self, loop_id: str | None = None) -> dict:
        """
        Get a summary of current inbox state without draining.

        Args:
            loop_id: Optional loop ID to filter messages by (multi-tenant support)

        Returns:
            Dict with vote counts, suggestions, histograms, and conviction distribution
        """
        with self._lock:
            votes: dict[str, int] = {}
            suggestions = 0
            # Per-choice intensity histograms: {choice: {intensity: count}}
            histograms: dict[str, dict[int, int]] = {}
            # Global conviction distribution: {intensity: count}
            conviction_distribution: dict[int, int] = {i: 0 for i in range(1, 11)}

            for msg in self._messages:
                # Filter by loop_id if provided
                if loop_id and msg.loop_id != loop_id:
                    continue

                if msg.type == "vote":
                    choice = msg.payload.get("choice", "unknown")
                    intensity = normalize_intensity(msg.payload.get("intensity"))

                    # Basic vote count
                    votes[choice] = votes.get(choice, 0) + 1

                    # Per-choice histogram
                    if choice not in histograms:
                        histograms[choice] = {i: 0 for i in range(1, 11)}
                    histograms[choice][intensity] = histograms[choice].get(intensity, 0) + 1

                    # Global conviction distribution
                    conviction_distribution[intensity] = conviction_distribution.get(intensity, 0) + 1

                elif msg.type == "suggestion":
                    suggestions += 1

            # Calculate weighted votes using intensity
            weighted_votes = {}
            for choice, histogram in histograms.items():
                weighted_sum = sum(
                    count * (0.5 + (intensity - 1) * 0.1667)  # Linear scale: 1->0.5, 10->2.0
                    for intensity, count in histogram.items()
                )
                weighted_votes[choice] = round(weighted_sum, 2)

            return {
                "votes": votes,
                "weighted_votes": weighted_votes,
                "suggestions": suggestions,
                "total": len(self._messages) if not loop_id else sum(votes.values()) + suggestions,
                "histograms": histograms,
                "conviction_distribution": conviction_distribution,
            }

    def drain_suggestions(self, loop_id: str | None = None) -> list[dict]:
        """
        Drain and return all suggestion messages, optionally filtered by loop_id.

        Args:
            loop_id: Optional loop ID to filter suggestions by

        Returns:
            List of suggestion payloads
        """
        with self._lock:
            suggestions = []
            remaining = []

            for msg in self._messages:
                # Filter by loop_id if provided
                if loop_id and msg.loop_id != loop_id:
                    remaining.append(msg)
                    continue

                if msg.type == "suggestion":
                    suggestions.append(msg.payload)
                else:
                    remaining.append(msg)

            # Clear and repopulate the deque to preserve maxlen
            self._messages.clear()
            self._messages.extend(remaining)
            return suggestions


class SyncEventEmitter:
    """
    Thread-safe event emitter bridging sync Arena code with async WebSocket.

    Events are queued synchronously via emit() and consumed by async drain().
    This pattern avoids needing to rewrite Arena to be fully async.

    Sequence numbers are automatically assigned to enable:
    - Global ordering (seq) for detecting message reordering
    - Per-agent ordering (agent_seq) for token stream integrity
    """

    # Maximum queue size to prevent memory exhaustion (DoS protection)
    # Configurable via ARAGORA_MAX_EVENT_QUEUE_SIZE environment variable
    MAX_QUEUE_SIZE = MAX_EVENT_QUEUE_SIZE

    def __init__(self, loop_id: str = ""):
        self._queue: queue.Queue[StreamEvent] = queue.Queue(maxsize=MAX_EVENT_QUEUE_SIZE)
        self._subscribers: list[Callable[[StreamEvent], None]] = []
        self._loop_id = loop_id  # Default loop_id for all events
        self._overflow_count = 0  # Track dropped events for monitoring
        self._global_seq = 0  # Global sequence counter
        self._agent_seqs: dict[str, int] = {}  # Per-agent sequence counters
        self._seq_lock = threading.Lock()  # Thread-safe sequence assignment

    def set_loop_id(self, loop_id: str) -> None:
        """Set the loop_id to attach to all emitted events."""
        self._loop_id = loop_id

    def reset_sequences(self) -> None:
        """Reset sequence counters (call when starting a new debate)."""
        with self._seq_lock:
            self._global_seq = 0
            self._agent_seqs.clear()

    def emit(self, event: StreamEvent) -> None:
        """Emit event (safe to call from sync code).

        Automatically assigns sequence numbers for ordering:
        - seq: Global sequence across all events
        - agent_seq: Per-agent sequence for token stream integrity
        """
        # Add loop_id to event if not already set
        if self._loop_id and not event.loop_id:
            event.loop_id = self._loop_id

        # Assign sequence numbers (thread-safe)
        with self._seq_lock:
            self._global_seq += 1
            event.seq = self._global_seq

            # Per-agent sequence for token events
            if event.agent:
                if event.agent not in self._agent_seqs:
                    self._agent_seqs[event.agent] = 0
                self._agent_seqs[event.agent] += 1
                event.agent_seq = self._agent_seqs[event.agent]

        # Enforce queue size limit to prevent memory exhaustion
        if self._queue.qsize() >= self.MAX_QUEUE_SIZE:
            # Drop oldest event to make room (backpressure)
            try:
                self._queue.get_nowait()
                self._overflow_count += 1
                logger.warning(f"[stream] Queue overflow, dropped event (total: {self._overflow_count})")
            except queue.Empty:
                pass

        self._queue.put(event)
        for sub in self._subscribers:
            try:
                sub(event)
            except Exception as e:
                logger.warning(f"[stream] Subscriber callback error: {e}")

    def subscribe(self, callback: Callable[[StreamEvent], None]) -> None:
        """Add synchronous subscriber for immediate event handling."""
        self._subscribers.append(callback)

    def drain(self, max_batch_size: int = 100) -> list[StreamEvent]:
        """Get queued events (non-blocking) with backpressure limit."""
        events: list[StreamEvent] = []
        try:
            while len(events) < max_batch_size:
                events.append(self._queue.get_nowait())
        except queue.Empty:
            pass
        return events

    def broadcast_event(
        self,
        event_type: StreamEventType,
        data: dict,
        agent: str = "",
        round_num: int = 0,
        redactor: Optional[Callable[[dict], dict]] = None,
    ) -> bool:
        """
        Broadcast an event with optional redaction for telemetry.

        This method respects the TelemetryConfig settings:
        - SILENT: Event is not emitted
        - DIAGNOSTIC: Event is logged but not broadcast
        - CONTROLLED: Event is emitted with redaction applied
        - SPECTACLE: Event is emitted without redaction

        Args:
            event_type: Type of event to emit
            data: Event payload data
            agent: Agent name (optional)
            round_num: Debate round number (optional)
            redactor: Optional function to redact sensitive data

        Returns:
            True if event was emitted, False if suppressed
        """
        try:
            from aragora.debate.telemetry_config import TelemetryConfig

            config = TelemetryConfig.get_instance()

            # Check if telemetry should be suppressed
            if config.is_silent():
                return False

            # Check if this is a telemetry-specific event
            is_telemetry_event = event_type.value.startswith("telemetry_")

            if is_telemetry_event:
                if config.is_diagnostic():
                    # Log only, don't broadcast
                    logger.debug(f"[telemetry] {event_type.value}: {data}")
                    return False

                # Apply redaction if in controlled mode
                if config.should_redact() and redactor is not None:
                    try:
                        data = redactor(data)
                        # Emit redaction notification
                        self.emit(StreamEvent(
                            type=StreamEventType.TELEMETRY_REDACTION,
                            data={"agent": agent, "event_type": event_type.value},
                            agent=agent,
                            round=round_num,
                        ))
                    except Exception as e:
                        logger.warning(f"[telemetry] Redaction failed: {e}")
                        # On redaction failure, suppress the event for security
                        return False

            # Emit the event
            self.emit(StreamEvent(
                type=event_type,
                data=data,
                agent=agent,
                round=round_num,
            ))
            return True

        except ImportError:
            # TelemetryConfig not available, emit without telemetry controls
            self.emit(StreamEvent(
                type=event_type,
                data=data,
                agent=agent,
                round=round_num,
            ))
            return True
        except Exception as e:
            logger.error(f"[telemetry] broadcast_event failed: {e}")
            return False


__all__ = [
    "TokenBucket",
    "AudienceInbox",
    "SyncEventEmitter",
    "normalize_intensity",
]
