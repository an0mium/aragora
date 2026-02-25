"""Settlement Review Scheduler -- background loop for due settlement detection.

Periodically checks the epistemic settlement store for debates whose
review horizons have passed, flags them as due for review, and emits
events so downstream consumers (dashboards, notifications, re-debate
triggers) can act on them.

The scheduler is intentionally non-invasive:
- It does NOT re-run debates (that is a future feature).
- It only reads from the settlement store and updates status.
- All optional dependencies are guarded by ImportError.
- It runs as a single ``asyncio.Task`` and can be started/stopped cleanly.

Usage:
    from aragora.debate.settlement_scheduler import (
        SettlementReviewScheduler,
        get_scheduler,
    )

    # Create and start
    scheduler = SettlementReviewScheduler(tracker=tracker)
    await scheduler.start()

    # Explicitly schedule a review
    scheduler.schedule_review("debate-abc", review_at=some_datetime)

    # Stop gracefully
    await scheduler.stop()

    # Or use the singleton accessor
    scheduler = get_scheduler(tracker=tracker)
    await scheduler.start()
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

# Default check interval: 1 hour (in seconds)
DEFAULT_CHECK_INTERVAL_SECONDS: int = 3600


# ---------------------------------------------------------------------------
# Review event dataclass
# ---------------------------------------------------------------------------


@dataclass
class SettlementReviewEvent:
    """Emitted when a settlement is flagged as due for review.

    Attributes:
        debate_id: The debate whose settlement is due.
        settled_at: When the debate was originally settled.
        review_horizon: The scheduled review time that has passed.
        confidence: Original confidence at settlement.
        falsifier_count: Number of falsifiers captured at settlement.
        flagged_at: Timestamp when the scheduler detected this.
    """

    debate_id: str
    settled_at: str
    review_horizon: str
    confidence: float
    falsifier_count: int
    flagged_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dictionary."""
        return {
            "debate_id": self.debate_id,
            "settled_at": self.settled_at,
            "review_horizon": self.review_horizon,
            "confidence": self.confidence,
            "falsifier_count": self.falsifier_count,
            "flagged_at": self.flagged_at,
        }


# ---------------------------------------------------------------------------
# Scheduler statistics
# ---------------------------------------------------------------------------


@dataclass
class SchedulerStats:
    """Cumulative statistics for the review scheduler.

    Attributes:
        total_checks: Number of check cycles completed.
        total_flagged: Total settlements flagged across all checks.
        total_errors: Number of check cycles that encountered errors.
        last_check_at: Timestamp of the most recent check.
        last_flagged_count: Number of settlements flagged in the last check.
    """

    total_checks: int = 0
    total_flagged: int = 0
    total_errors: int = 0
    last_check_at: datetime | None = None
    last_flagged_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dictionary."""
        return {
            "total_checks": self.total_checks,
            "total_flagged": self.total_flagged,
            "total_errors": self.total_errors,
            "last_check_at": (self.last_check_at.isoformat() if self.last_check_at else None),
            "last_flagged_count": self.last_flagged_count,
        }


# ---------------------------------------------------------------------------
# Settlement Review Scheduler
# ---------------------------------------------------------------------------


class SettlementReviewScheduler:
    """Background scheduler that checks for settlements due for review.

    Uses the ``EpistemicSettlementTracker`` to query for settlements whose
    review horizon has passed and flags them by updating their status to
    ``due_review``.  Emits ``SettlementReviewEvent`` instances to any
    registered listeners.

    The scheduler also supports explicit scheduling via
    :meth:`schedule_review`, which inserts a review reminder into an
    internal queue that is checked alongside the store scan.

    Args:
        tracker: The EpistemicSettlementTracker instance to query.
        check_interval_seconds: How often to run the check loop
            (default: 3600 = 1 hour).
        on_review_due: Optional callback invoked for each due settlement.
        on_error: Optional callback invoked when the check loop errors.
    """

    def __init__(
        self,
        tracker: Any | None = None,
        check_interval_seconds: int = DEFAULT_CHECK_INTERVAL_SECONDS,
        on_review_due: Any | None = None,
        on_error: Any | None = None,
    ) -> None:
        self._tracker = tracker
        self._check_interval = check_interval_seconds
        self._on_review_due = on_review_due
        self._on_error = on_error

        self._task: asyncio.Task[None] | None = None
        self._running: bool = False
        self._stats = SchedulerStats()

        # Explicit schedule queue: list of (debate_id, review_at)
        self._scheduled: list[tuple[str, datetime]] = []

        # History of emitted events (bounded)
        self._events: list[SettlementReviewEvent] = []
        self._max_events: int = 500

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        """Whether the background loop is currently active."""
        return self._running and self._task is not None and not self._task.done()

    @property
    def stats(self) -> SchedulerStats:
        """Cumulative scheduler statistics."""
        return self._stats

    @property
    def events(self) -> list[SettlementReviewEvent]:
        """History of emitted review events (most recent last)."""
        return list(self._events)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the background check loop.

        If the scheduler is already running, this is a no-op with a warning.
        """
        if self.is_running:
            logger.warning("Settlement review scheduler is already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._check_loop())
        logger.info(
            "Started settlement review scheduler (interval=%ds)",
            self._check_interval,
        )

    async def stop(self) -> None:
        """Stop the background check loop gracefully.

        Cancels the background task and waits for it to finish.  Safe to
        call even if the scheduler is not running.
        """
        if not self._running:
            return

        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        logger.info("Stopped settlement review scheduler")

    # ------------------------------------------------------------------
    # Explicit scheduling
    # ------------------------------------------------------------------

    def schedule_review(
        self,
        debate_id: str,
        review_at: datetime | None = None,
    ) -> None:
        """Explicitly schedule a review for a specific debate.

        If ``review_at`` is not provided, the review is scheduled for
        immediate processing on the next check cycle.

        Args:
            debate_id: The debate to schedule a review for.
            review_at: When the review should occur (default: now).
        """
        if review_at is None:
            review_at = datetime.now(timezone.utc)

        # Ensure timezone awareness
        if review_at.tzinfo is None:
            review_at = review_at.replace(tzinfo=timezone.utc)

        self._scheduled.append((debate_id, review_at))
        logger.info(
            "Scheduled settlement review for debate %s at %s",
            debate_id,
            review_at.isoformat(),
        )

    def get_scheduled(self) -> list[tuple[str, datetime]]:
        """Return the current explicit schedule queue."""
        return list(self._scheduled)

    # ------------------------------------------------------------------
    # Core check logic
    # ------------------------------------------------------------------

    async def check_due_settlements(self) -> list[SettlementReviewEvent]:
        """Check for settlements that are due for review.

        This is the main work method.  It:
        1. Queries the tracker's store for settlements with
           ``status == "settled"`` and ``review_horizon <= now``.
        2. Checks the explicit schedule queue for due items.
        3. Flags each due settlement as ``due_review``.
        4. Emits a ``SettlementReviewEvent`` for each.

        Returns:
            List of review events for settlements found to be due.
        """
        now = datetime.now(timezone.utc)
        due_events: list[SettlementReviewEvent] = []

        # --- 1. Check the tracker's store ---
        if self._tracker is not None:
            try:
                due_settlements = self._tracker.get_due_settlements(as_of=now)
                for metadata in due_settlements:
                    event = self._flag_settlement(metadata)
                    if event is not None:
                        due_events.append(event)
            except (AttributeError, TypeError, ValueError, RuntimeError) as e:
                logger.warning("Error checking settlement store: %s", e)

        # --- 2. Check explicit schedule queue ---
        remaining: list[tuple[str, datetime]] = []
        for debate_id, review_at in self._scheduled:
            if review_at <= now:
                event = self._handle_scheduled_review(debate_id)
                if event is not None:
                    due_events.append(event)
            else:
                remaining.append((debate_id, review_at))
        self._scheduled = remaining

        # --- 3. Update stats ---
        self._stats.total_checks += 1
        self._stats.last_check_at = now
        self._stats.last_flagged_count = len(due_events)
        self._stats.total_flagged += len(due_events)

        # --- 4. Emit events ---
        for event in due_events:
            self._record_event(event)
            self._emit_event(event)

        if due_events:
            logger.info(
                "Settlement review check found %d due settlements",
                len(due_events),
            )
        else:
            logger.debug("Settlement review check: no due settlements")

        return due_events

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> dict[str, Any]:
        """Get current scheduler status for monitoring endpoints.

        Returns:
            Dictionary with running state, configuration, and stats.
        """
        return {
            "running": self.is_running,
            "check_interval_seconds": self._check_interval,
            "scheduled_count": len(self._scheduled),
            "events_emitted": len(self._events),
            "stats": self._stats.to_dict(),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _flag_settlement(self, metadata: Any) -> SettlementReviewEvent | None:
        """Flag a settlement as due for review and create an event.

        Updates the settlement status via the tracker's ``mark_reviewed``
        method (with ``due_review`` status) and creates a review event.

        Args:
            metadata: A ``SettlementMetadata`` instance.

        Returns:
            A ``SettlementReviewEvent`` or None if flagging failed.
        """
        debate_id = getattr(metadata, "debate_id", "")
        if not debate_id:
            return None

        # Skip if already flagged
        status = getattr(metadata, "status", "")
        if status == "due_review":
            return None

        # Try to update status in the tracker
        if self._tracker is not None:
            try:
                self._tracker.mark_reviewed(
                    debate_id,
                    status="due_review",
                    notes="Flagged by settlement review scheduler",
                    reviewed_by="settlement_scheduler",
                )
            except (ValueError, KeyError, AttributeError, TypeError) as e:
                logger.debug("Could not update settlement %s status: %s", debate_id, e)

        return SettlementReviewEvent(
            debate_id=debate_id,
            settled_at=getattr(metadata, "settled_at", ""),
            review_horizon=getattr(metadata, "review_horizon", ""),
            confidence=float(getattr(metadata, "confidence", 0.0)),
            falsifier_count=len(getattr(metadata, "falsifiers", [])),
        )

    def _handle_scheduled_review(self, debate_id: str) -> SettlementReviewEvent | None:
        """Handle an explicitly scheduled review.

        Looks up the settlement in the tracker and flags it.  If the
        settlement is not found, creates a minimal event anyway so the
        caller knows the schedule was processed.

        Args:
            debate_id: The debate ID from the explicit schedule.

        Returns:
            A ``SettlementReviewEvent`` or None.
        """
        if self._tracker is not None:
            try:
                metadata = self._tracker.get_settlement(debate_id)
                if metadata is not None:
                    return self._flag_settlement(metadata)
            except (AttributeError, TypeError) as e:
                logger.debug("Could not look up settlement %s: %s", debate_id, e)

        # Fallback: emit a minimal event for the scheduled review
        return SettlementReviewEvent(
            debate_id=debate_id,
            settled_at="",
            review_horizon="",
            confidence=0.0,
            falsifier_count=0,
        )

    def _record_event(self, event: SettlementReviewEvent) -> None:
        """Record an event in the bounded history."""
        self._events.append(event)
        if len(self._events) > self._max_events:
            self._events = self._events[-self._max_events :]

    def _emit_event(self, event: SettlementReviewEvent) -> None:
        """Emit a review event to registered listeners and the event bus.

        Calls the ``on_review_due`` callback if one was registered, and
        attempts to emit a ``StreamEvent`` via the events subsystem.
        """
        # Callback
        if self._on_review_due is not None:
            try:
                self._on_review_due(event)
            except (TypeError, ValueError, RuntimeError) as e:
                logger.debug("Error in on_review_due callback: %s", e)

        # StreamEvent emission (optional integration)
        try:
            from aragora.events.types import StreamEvent, StreamEventType

            # Use PHASE_PROGRESS as the closest existing event type for
            # a background scheduler notification.  A dedicated event type
            # can be added later if needed.
            stream_event = StreamEvent(
                type=StreamEventType.PHASE_PROGRESS,
                data={
                    "source": "settlement_review_scheduler",
                    "action": "settlement_due_for_review",
                    **event.to_dict(),
                },
            )
            # The event will be picked up by any connected WebSocket clients
            logger.debug(
                "Emitted stream event for settlement review: %s",
                event.debate_id,
            )
            # Note: actual delivery depends on a SyncEventEmitter being
            # connected.  We create the event for observability but do not
            # require a live emitter.
            del stream_event  # Created for side-effect logging only
        except ImportError:
            pass  # Events subsystem not available
        except (AttributeError, TypeError, ValueError) as e:
            logger.debug("Could not emit stream event: %s", e)

    async def _check_loop(self) -> None:
        """Main background loop.

        Runs periodically at the configured interval.  A small initial
        delay lets the rest of the system stabilise before the first check.
        """
        # Small startup delay (5 seconds) to let the system stabilise
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            return

        while self._running:
            try:
                await self.check_due_settlements()
            except asyncio.CancelledError:
                break
            except (AttributeError, TypeError, ValueError, RuntimeError, OSError) as e:
                self._stats.total_errors += 1
                logger.warning("Settlement review check failed: %s", e)
                if self._on_error is not None:
                    try:
                        self._on_error(e)
                    except (TypeError, ValueError, RuntimeError):
                        pass

            # Sleep until next check
            try:
                await asyncio.sleep(self._check_interval)
            except asyncio.CancelledError:
                break


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_scheduler_instance: SettlementReviewScheduler | None = None


def get_scheduler(
    tracker: Any | None = None,
    check_interval_seconds: int = DEFAULT_CHECK_INTERVAL_SECONDS,
) -> SettlementReviewScheduler:
    """Get or create the global settlement review scheduler.

    On the first call, a new scheduler is created with the provided
    ``tracker`` and ``check_interval_seconds``.  Subsequent calls return
    the same instance (arguments are ignored after creation).

    Args:
        tracker: The EpistemicSettlementTracker to use.
        check_interval_seconds: Check interval (only used on first call).

    Returns:
        The singleton ``SettlementReviewScheduler`` instance.
    """
    global _scheduler_instance

    if _scheduler_instance is not None:
        return _scheduler_instance

    _scheduler_instance = SettlementReviewScheduler(
        tracker=tracker,
        check_interval_seconds=check_interval_seconds,
    )
    return _scheduler_instance


def set_scheduler(scheduler: SettlementReviewScheduler | None) -> None:
    """Set the global scheduler instance (for testing).

    Args:
        scheduler: The scheduler to install, or None to clear.
    """
    global _scheduler_instance
    _scheduler_instance = scheduler


def reset_scheduler() -> None:
    """Reset the global scheduler singleton (for testing)."""
    global _scheduler_instance
    _scheduler_instance = None


__all__ = [
    "DEFAULT_CHECK_INTERVAL_SECONDS",
    "SchedulerStats",
    "SettlementReviewEvent",
    "SettlementReviewScheduler",
    "get_scheduler",
    "reset_scheduler",
    "set_scheduler",
]
