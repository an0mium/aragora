"""
EventBatcher - Batch KM events for efficient WebSocket transmission.

Instead of emitting each KM event individually (which can flood WebSocket clients
during bulk operations), this batcher collects events and emits them in batches.

Usage:
    from aragora.knowledge.mound.event_batcher import EventBatcher

    # Create batcher with a callback
    batcher = EventBatcher(
        callback=websocket_emit,
        batch_interval_ms=100,  # Batch every 100ms
        max_batch_size=50,  # Or when we hit 50 events
    )

    # Queue events (will be batched automatically)
    batcher.queue_event("knowledge_indexed", {"source": "evidence", "id": "ev_001"})
    batcher.queue_event("belief_converged", {"debate_id": "d_001", "claim": "..."})

    # Events are emitted as:
    # {"type": "km_batch", "events": [{"type": "knowledge_indexed", ...}, ...], "count": 2}

    # Cleanup when done
    await batcher.flush()
    await batcher.stop()
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class BatchedEvent:
    """A single event in a batch."""

    event_type: str
    data: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "type": event_type,
            "data": self.data,
            "timestamp": self.timestamp,
        }


@dataclass
class EventBatch:
    """A batch of events ready for emission."""

    events: List[BatchedEvent] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    @property
    def count(self) -> int:
        return len(self.events)

    @property
    def age_ms(self) -> float:
        return (time.time() - self.created_at) * 1000

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "type": "km_batch",
            "events": [
                {"type": e.event_type, "data": e.data, "timestamp": e.timestamp}
                for e in self.events
            ],
            "count": self.count,
            "batch_created_at": self.created_at,
        }


# Type alias for callbacks
EventCallback = Callable[[str, Dict[str, Any]], None]
AsyncEventCallback = Callable[[str, Dict[str, Any]], "asyncio.Future[None]"]


class EventBatcher:
    """
    Batches KM events for efficient WebSocket transmission.

    Features:
    - Configurable batch interval (time-based batching)
    - Configurable max batch size (count-based batching)
    - Event type filtering (only batch specific event types)
    - Passthrough mode for high-priority events
    - Statistics tracking
    - Async and sync callback support

    The batcher will emit a batch when:
    1. The batch interval expires (e.g., 100ms)
    2. The batch size reaches max_batch_size
    3. flush() is called manually
    """

    def __init__(
        self,
        callback: Optional[Union[EventCallback, AsyncEventCallback]] = None,
        batch_interval_ms: float = 100.0,
        max_batch_size: int = 50,
        batch_event_types: Optional[List[str]] = None,
        passthrough_event_types: Optional[List[str]] = None,
    ):
        """
        Initialize the event batcher.

        Args:
            callback: Function to call with batched events
            batch_interval_ms: Max time to wait before emitting batch (default 100ms)
            max_batch_size: Max events in a batch (default 50)
            batch_event_types: Event types to batch (None = batch all)
            passthrough_event_types: Event types to emit immediately (bypass batching)
        """
        self._callback = callback
        self._batch_interval_ms = batch_interval_ms
        self._max_batch_size = max_batch_size
        self._batch_event_types = set(batch_event_types) if batch_event_types else None
        self._passthrough_event_types = set(passthrough_event_types or [])

        # Current batch
        self._batch = EventBatch()
        self._lock = Lock()

        # Background task for time-based emission
        self._flush_task: Optional[asyncio.Task] = None
        self._running = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Statistics
        self._total_events_queued = 0
        self._total_events_emitted = 0
        self._total_batches_emitted = 0
        self._passthrough_events = 0

    def set_callback(self, callback: Union[EventCallback, AsyncEventCallback]) -> None:
        """Set the callback for emitting batched events."""
        self._callback = callback

    def start(self, loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        """Start the background flush task."""
        if self._running:
            return

        self._running = True
        self._loop = loop or asyncio.get_event_loop()

        try:
            self._flush_task = self._loop.create_task(self._flush_loop())
        except RuntimeError:
            # No running event loop, will flush synchronously
            logger.debug("No event loop available, events will be flushed synchronously")

    async def stop(self) -> None:
        """Stop the background flush task and emit remaining events."""
        self._running = False

        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
            self._flush_task = None

        # Flush any remaining events
        await self.flush()

    def queue_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Queue an event for batching.

        Args:
            event_type: The type of event (e.g., "knowledge_indexed")
            data: Event payload data
        """
        self._total_events_queued += 1

        # Check if this event type should pass through immediately
        if event_type in self._passthrough_event_types:
            self._emit_immediately(event_type, data)
            self._passthrough_events += 1
            return

        # Check if this event type should be batched
        if self._batch_event_types and event_type not in self._batch_event_types:
            self._emit_immediately(event_type, data)
            self._passthrough_events += 1
            return

        # Add to batch
        event = BatchedEvent(event_type=event_type, data=data)

        should_flush = False
        with self._lock:
            self._batch.events.append(event)

            # Check if we should flush due to size
            if len(self._batch.events) >= self._max_batch_size:
                should_flush = True

        # Schedule flush outside the lock to avoid deadlock
        if should_flush:
            self._schedule_flush()

    def _emit_immediately(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit an event immediately (bypass batching)."""
        if not self._callback:
            return

        try:
            result = self._callback(event_type, data)
            if asyncio.iscoroutine(result):
                # Schedule coroutine on event loop
                if self._loop and self._loop.is_running():
                    asyncio.ensure_future(result, loop=self._loop)
            self._total_events_emitted += 1
        except Exception as e:
            logger.warning(f"Failed to emit event: {e}")

    def _schedule_flush(self) -> None:
        """Schedule an immediate flush."""
        if self._loop and self._loop.is_running():
            asyncio.ensure_future(self.flush(), loop=self._loop)
        else:
            # Synchronous flush
            self._flush_sync()

    async def _flush_loop(self) -> None:
        """Background loop that flushes batches at regular intervals."""
        while self._running:
            try:
                await asyncio.sleep(self._batch_interval_ms / 1000.0)
                await self.flush()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Error in flush loop: {e}")

    async def flush(self) -> int:
        """
        Flush the current batch, emitting all queued events.

        Returns:
            Number of events flushed
        """
        with self._lock:
            if not self._batch.events:
                return 0

            batch = self._batch
            self._batch = EventBatch()

        return await self._emit_batch(batch)

    def _flush_sync(self) -> int:
        """Synchronous flush for when no event loop is available."""
        with self._lock:
            if not self._batch.events:
                return 0

            batch = self._batch
            self._batch = EventBatch()

        if not self._callback:
            return 0

        try:
            # Emit as batch event
            batch_data = batch.to_dict()
            self._callback("km_batch", batch_data)
            self._total_events_emitted += batch.count
            self._total_batches_emitted += 1
            return batch.count
        except Exception as e:
            logger.warning(f"Failed to emit batch: {e}")
            return 0

    async def _emit_batch(self, batch: EventBatch) -> int:
        """Emit a batch of events."""
        if not self._callback or not batch.events:
            return 0

        try:
            # Emit as batch event
            batch_data = batch.to_dict()
            result = self._callback("km_batch", batch_data)
            if asyncio.iscoroutine(result):
                await result

            self._total_events_emitted += batch.count
            self._total_batches_emitted += 1

            logger.debug(f"Emitted batch with {batch.count} events")
            return batch.count

        except Exception as e:
            logger.warning(f"Failed to emit batch: {e}")
            return 0

    def get_stats(self) -> Dict[str, Any]:
        """Get batching statistics."""
        with self._lock:
            pending = len(self._batch.events)

        return {
            "total_events_queued": self._total_events_queued,
            "total_events_emitted": self._total_events_emitted,
            "total_batches_emitted": self._total_batches_emitted,
            "passthrough_events": self._passthrough_events,
            "pending_events": pending,
            "batch_interval_ms": self._batch_interval_ms,
            "max_batch_size": self._max_batch_size,
            "running": self._running,
            "average_batch_size": (
                self._total_events_emitted / max(self._total_batches_emitted, 1)
            ),
        }

    def reset_stats(self) -> None:
        """Reset statistics."""
        self._total_events_queued = 0
        self._total_events_emitted = 0
        self._total_batches_emitted = 0
        self._passthrough_events = 0


class AdapterEventBatcher:
    """
    Convenience wrapper for using EventBatcher with KM adapters.

    Provides a callback that can be passed to adapters that automatically
    queues events through the batcher.

    Usage:
        batcher = EventBatcher(callback=websocket_emit)
        adapter_batcher = AdapterEventBatcher(batcher)

        # Pass to adapter
        adapter = EvidenceAdapter(event_callback=adapter_batcher.event_callback)

        # Or set on multiple adapters
        for adapter in adapters:
            adapter.set_event_callback(adapter_batcher.event_callback)
    """

    def __init__(
        self,
        batcher: EventBatcher,
        prefix: str = "km",
    ):
        """
        Initialize adapter event batcher.

        Args:
            batcher: The EventBatcher to use
            prefix: Prefix to add to event types (default "km")
        """
        self._batcher = batcher
        self._prefix = prefix

    def event_callback(self, event_type: str, data: Dict[str, Any]) -> None:
        """
        Callback function to pass to adapters.

        This queues events through the batcher automatically.
        """
        # Add prefix if not already present
        if self._prefix and not event_type.startswith(self._prefix):
            event_type = f"{self._prefix}_{event_type}"

        self._batcher.queue_event(event_type, data)

    @property
    def stats(self) -> Dict[str, Any]:
        """Get batcher statistics."""
        return self._batcher.get_stats()


__all__ = [
    "EventBatcher",
    "EventBatch",
    "BatchedEvent",
    "AdapterEventBatcher",
]
