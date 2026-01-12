"""
Audience participation manager for debates.

Handles user event queuing, draining, and processing using the
Stadium Mailbox pattern for thread-safe audience participation.

Extracted from orchestrator.py to separate audience participation
concerns from core debate orchestration.
"""

from __future__ import annotations

import logging
import queue
import threading
from collections import deque
from typing import TYPE_CHECKING, Callable, Optional

from aragora.config import USER_EVENT_QUEUE_SIZE

if TYPE_CHECKING:
    from aragora.server.stream.events import StreamEventType

logger = logging.getLogger(__name__)


class AudienceManager:
    """
    Manages audience participation in debates.

    Implements the Stadium Mailbox pattern:
    1. Events arrive asynchronously via handle_event() (thread-safe enqueue)
    2. Events are processed synchronously via drain_events() (digest phase)

    This separation ensures thread safety while allowing the debate loop
    to process audience input at controlled checkpoints.

    Usage:
        manager = AudienceManager(loop_id="debate-123")
        manager.set_notify_callback(arena._notify_spectator)

        # From WebSocket thread (thread-safe)
        manager.handle_event(event)

        # From debate loop (at safe checkpoints)
        manager.drain_events()
        votes = manager.get_votes()
        suggestions = manager.get_suggestions()
    """

    def __init__(
        self,
        loop_id: Optional[str] = None,
        strict_loop_scoping: bool = False,
        queue_size: int = USER_EVENT_QUEUE_SIZE,
    ):
        """
        Initialize the audience manager.

        Args:
            loop_id: Unique identifier for this debate loop (for event filtering)
            strict_loop_scoping: If True, drop events without matching loop_id
            queue_size: Maximum size of the event queue and deques
        """
        self.loop_id = loop_id
        self.strict_loop_scoping = strict_loop_scoping

        # Thread-safe queue for incoming events (mailbox)
        self._event_queue: queue.Queue = queue.Queue(maxsize=queue_size)

        # Processed events (bounded deques for O(1) operations)
        self._votes: deque[dict] = deque(maxlen=queue_size)
        self._suggestions: deque[dict] = deque(maxlen=queue_size)
        self._data_lock = threading.Lock()  # Protects votes/suggestions access

        # Optional callback for notifications
        self._notify_callback: Optional[Callable[[str], None]] = None

    def set_notify_callback(
        self, callback: Callable[..., None]
    ) -> None:
        """Set callback for spectator notifications."""
        self._notify_callback = callback  # type: ignore[assignment]

    def subscribe_to_emitter(self, event_emitter) -> None:
        """Subscribe to an event emitter for user participation events."""
        if event_emitter:
            event_emitter.subscribe(self.handle_event)

    def handle_event(self, event) -> None:
        """
        Handle incoming user participation event (thread-safe).

        Events are enqueued for later processing by drain_events().
        This method may be called from any thread (e.g., WebSocket server).

        Args:
            event: Event with type and data attributes
        """
        from aragora.server.stream.events import StreamEventType

        # Ignore events from other loops to prevent cross-contamination
        event_loop_id = getattr(event, "loop_id", None)
        if event_loop_id and event_loop_id != self.loop_id:
            return

        # In strict scoping mode, drop events without a loop_id
        if self.strict_loop_scoping and not event_loop_id:
            return

        # Enqueue for processing (thread-safe)
        if event.type in (StreamEventType.USER_VOTE, StreamEventType.USER_SUGGESTION):
            try:
                self._event_queue.put_nowait((event.type, event.data))
            except queue.Full:
                logger.warning(f"User event queue full, dropping {event.type}")

    def drain_events(self) -> int:
        """
        Drain pending user events from queue into working lists.

        This method should be called at safe points in the debate loop:
        - Before building prompts that include audience suggestions
        - Before vote aggregation that includes user votes

        This is the 'digest' phase of the Stadium Mailbox pattern.

        Returns:
            Number of events processed
        """
        from aragora.server.stream.events import StreamEventType

        drained_count = 0
        while True:
            try:
                event_type, event_data = self._event_queue.get_nowait()
                with self._data_lock:
                    if event_type == StreamEventType.USER_VOTE:
                        self._votes.append(event_data)  # deque auto-evicts oldest
                    elif event_type == StreamEventType.USER_SUGGESTION:
                        self._suggestions.append(event_data)  # deque auto-evicts oldest
                drained_count += 1
            except queue.Empty:
                break

        if drained_count > 0 and self._notify_callback:
            self._notify_callback(  # type: ignore[call-arg]
                "audience_drain",
                details=f"Processed {drained_count} audience events",
            )

        return drained_count

    def get_votes(self) -> list[dict]:
        """Get all drained user votes (thread-safe)."""
        with self._data_lock:
            return list(self._votes)

    def get_suggestions(self) -> list[dict]:
        """Get all drained user suggestions (thread-safe)."""
        with self._data_lock:
            return list(self._suggestions)

    def clear_votes(self) -> None:
        """Clear processed votes (thread-safe)."""
        with self._data_lock:
            self._votes.clear()

    def clear_suggestions(self) -> None:
        """Clear processed suggestions (thread-safe)."""
        with self._data_lock:
            self._suggestions.clear()

    def clear_all(self) -> None:
        """Clear all processed events and drain queue (thread-safe)."""
        with self._data_lock:
            self._votes.clear()
            self._suggestions.clear()
        # Drain and discard any pending events (queue is already thread-safe)
        while True:
            try:
                self._event_queue.get_nowait()
            except queue.Empty:
                break

    @property
    def pending_count(self) -> int:
        """Get approximate count of pending (unprocessed) events."""
        return self._event_queue.qsize()

    @property
    def votes_count(self) -> int:
        """Get count of processed votes (thread-safe)."""
        with self._data_lock:
            return len(self._votes)

    @property
    def suggestions_count(self) -> int:
        """Get count of processed suggestions (thread-safe)."""
        with self._data_lock:
            return len(self._suggestions)


__all__ = ["AudienceManager"]
