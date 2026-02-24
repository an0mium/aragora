"""WebSocket bridge for SpectatorStream events.

Connects the terminal-only SpectatorStream to WebSocket event delivery,
enabling real-time debate and pipeline visualization in the dashboard.

Usage:
    from aragora.spectate.ws_bridge import SpectateWebSocketBridge

    bridge = SpectateWebSocketBridge()
    bridge.start()  # Begin forwarding events
    # ... debates and pipelines run ...
    bridge.stop()
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class SpectateEvent:
    """A spectate event ready for WebSocket delivery."""

    event_type: str  # debate_start, round_start, proposal, critique, vote, consensus, etc.
    timestamp: str
    data: dict[str, Any] = field(default_factory=dict)
    debate_id: str | None = None
    pipeline_id: str | None = None
    agent_name: str | None = None
    round_number: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return {
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "data": self.data,
            "debate_id": self.debate_id,
            "pipeline_id": self.pipeline_id,
            "agent_name": self.agent_name,
            "round_number": self.round_number,
        }


class SpectateWebSocketBridge:
    """Bridges SpectatorStream events to WebSocket delivery.

    Intercepts SpectatorStream.emit() calls and converts them into
    structured SpectateEvent objects that can be sent to WebSocket clients.

    Thread-safe: debates may run in background threads while subscribers
    are added/removed from the main thread.
    """

    def __init__(self, max_buffer: int = 1000) -> None:
        self._subscribers: list[Callable[[SpectateEvent], None]] = []
        self._event_buffer: list[SpectateEvent] = []
        self._max_buffer = max_buffer
        self._running = False
        self._lock = threading.Lock()
        self._original_emit: Any = None

    def start(self) -> None:
        """Start the bridge, hooking into SpectatorStream.

        Monkey-patches SpectatorStream.emit() to forward events to
        all registered subscribers while preserving the original
        terminal output behavior.
        """
        if self._running:
            return

        try:
            from aragora.spectate.stream import SpectatorStream

            # Save original emit method
            self._original_emit = SpectatorStream.emit
            bridge = self
            original_emit = self._original_emit

            def patched_emit(
                self_stream: Any,
                event_type: str,
                agent: str = "",
                details: str = "",
                metric: float | None = None,
                round_number: int | None = None,
            ) -> None:
                # Call original emit (terminal output)
                original_emit(
                    self_stream,
                    event_type,
                    agent=agent,
                    details=details,
                    metric=metric,
                    round_number=round_number,
                )
                # Only forward to bridge if the stream is enabled
                if not getattr(self_stream, "enabled", True):
                    return
                # Forward to bridge
                bridge._forward_event(
                    event_type,
                    agent=agent,
                    details=details,
                    metric=metric,
                    round_number=round_number,
                )

            SpectatorStream.emit = patched_emit  # type: ignore[assignment]
            self._running = True
            logger.info("spectate_ws_bridge_started")
        except ImportError:
            logger.debug("SpectatorStream not available, bridge inactive")

    def stop(self) -> None:
        """Stop the bridge, restoring original SpectatorStream.emit()."""
        if self._original_emit is not None:
            try:
                from aragora.spectate.stream import SpectatorStream

                SpectatorStream.emit = self._original_emit  # type: ignore[assignment]
            except ImportError:
                pass
            self._original_emit = None
        self._running = False
        logger.info("spectate_ws_bridge_stopped")

    @property
    def running(self) -> bool:
        """Whether the bridge is currently active."""
        return self._running

    def subscribe(self, callback: Callable[[SpectateEvent], None]) -> None:
        """Subscribe to spectate events.

        Args:
            callback: Function called with each SpectateEvent as it arrives.
        """
        with self._lock:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[SpectateEvent], None]) -> None:
        """Unsubscribe from spectate events.

        Args:
            callback: Previously registered callback to remove.
                      Uses equality (==) for matching, so bound methods
                      that compare equal will be correctly removed.
        """
        with self._lock:
            self._subscribers = [s for s in self._subscribers if s != callback]

    @property
    def subscriber_count(self) -> int:
        """Number of active subscribers."""
        with self._lock:
            return len(self._subscribers)

    @property
    def buffer_size(self) -> int:
        """Current number of events in the buffer."""
        with self._lock:
            return len(self._event_buffer)

    def get_recent_events(self, count: int = 50) -> list[SpectateEvent]:
        """Get recent events from the buffer.

        Args:
            count: Maximum number of recent events to return.

        Returns:
            List of recent SpectateEvent objects, newest last.
        """
        with self._lock:
            return list(self._event_buffer[-count:])

    def clear_buffer(self) -> None:
        """Clear the event buffer."""
        with self._lock:
            self._event_buffer.clear()

    def _forward_event(
        self,
        event_type: str,
        agent: str = "",
        details: str = "",
        metric: float | None = None,
        round_number: int | None = None,
    ) -> None:
        """Convert and forward a SpectatorStream event.

        Builds a SpectateEvent from the emit() parameters, buffers it,
        and dispatches to all subscribers. Subscriber errors are caught
        and logged to prevent spectating from breaking debates.
        """
        # Build data dict from non-empty keyword args
        data: dict[str, Any] = {}
        if details:
            data["details"] = details
        if metric is not None:
            data["metric"] = metric

        event = SpectateEvent(
            event_type=event_type,
            timestamp=datetime.now(timezone.utc).isoformat(),
            data=data,
            agent_name=agent or None,
            round_number=round_number,
        )

        with self._lock:
            self._event_buffer.append(event)
            if len(self._event_buffer) > self._max_buffer:
                self._event_buffer = self._event_buffer[-self._max_buffer :]

            subscribers = list(self._subscribers)

        # Dispatch outside the lock to avoid holding it during callbacks
        for subscriber in subscribers:
            try:
                subscriber(event)
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception:  # noqa: BLE001 - external subscriber callbacks must not crash event dispatch
                logger.debug("spectate_subscriber_error", exc_info=True)


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_bridge_instance: SpectateWebSocketBridge | None = None
_bridge_lock = threading.Lock()


def get_spectate_bridge() -> SpectateWebSocketBridge:
    """Get the global spectate bridge instance.

    Returns:
        The singleton SpectateWebSocketBridge instance.
    """
    global _bridge_instance
    if _bridge_instance is None:
        with _bridge_lock:
            if _bridge_instance is None:
                _bridge_instance = SpectateWebSocketBridge()
    return _bridge_instance


def reset_spectate_bridge() -> None:
    """Reset the global bridge instance. Intended for testing only."""
    global _bridge_instance
    with _bridge_lock:
        if _bridge_instance is not None:
            _bridge_instance.stop()
        _bridge_instance = None
