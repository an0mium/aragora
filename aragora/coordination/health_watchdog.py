"""Health Watchdog for multi-agent worktree coordination.

Monitors active worktrees for stalled processes, auto-restarts sessions,
and tracks recovery success rates.

Usage:
    from aragora.coordination.health_watchdog import HealthWatchdog

    watchdog = HealthWatchdog(worktree_manager, task_dispatcher)
    await watchdog.check_all()
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from aragora.coordination.worktree_manager import WorktreeManager, WorktreeState
    from aragora.coordination.task_dispatcher import TaskDispatcher

logger = logging.getLogger(__name__)


@dataclass
class HealthEvent:
    """Record of a health-related event."""

    event_type: str  # stall_detected, recovery_attempted, recovery_succeeded, recovery_failed, abandoned
    worktree_id: str
    task_id: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    details: str = ""


@dataclass
class WatchdogConfig:
    """Configuration for HealthWatchdog."""

    check_interval_seconds: float = 30.0
    max_recovery_attempts: int = 3
    auto_reassign_stalled: bool = True
    auto_cleanup_abandoned: bool = True


@dataclass
class RecoveryStats:
    """Aggregate recovery statistics."""

    stalls_detected: int = 0
    recoveries_attempted: int = 0
    recoveries_succeeded: int = 0
    recoveries_failed: int = 0
    abandoned_cleaned: int = 0

    @property
    def recovery_rate(self) -> float:
        """Fraction of recovery attempts that succeeded."""
        if self.recoveries_attempted == 0:
            return 0.0
        return self.recoveries_succeeded / self.recoveries_attempted


class HealthWatchdog:
    """Monitors worktree health and auto-recovers stalled sessions.

    Periodically checks all active worktrees for:
    - No git commits for N minutes (stall)
    - Process not producing heartbeats (stall)
    - Idle beyond abandon threshold (abandoned)

    On stall detection, can auto-reassign the task to another worktree.
    On abandon detection, can auto-cleanup the worktree.
    """

    def __init__(
        self,
        worktree_manager: WorktreeManager,
        task_dispatcher: TaskDispatcher | None = None,
        config: WatchdogConfig | None = None,
        on_stall: Callable[[HealthEvent], None] | None = None,
        on_recovery: Callable[[HealthEvent], None] | None = None,
    ):
        self.worktree_manager = worktree_manager
        self.task_dispatcher = task_dispatcher
        self.config = config or WatchdogConfig()
        self.on_stall = on_stall
        self.on_recovery = on_recovery
        self.stats = RecoveryStats()
        self.events: list[HealthEvent] = []
        self._recovery_attempts: dict[str, int] = {}  # worktree_id -> attempts
        self._running = False
        self._task: asyncio.Task | None = None

    async def check_all(self) -> list[HealthEvent]:
        """Run a single health check across all worktrees.

        Returns:
            List of health events generated during this check.
        """
        new_events: list[HealthEvent] = []

        # Check for new commits (updates last_activity automatically)
        for wt in self.worktree_manager.active_worktrees:
            self.worktree_manager.check_for_new_commits(wt.worktree_id)

        # Detect stalled worktrees
        stalled = self.worktree_manager.get_stalled_worktrees()
        for wt in stalled:
            event = HealthEvent(
                event_type="stall_detected",
                worktree_id=wt.worktree_id,
                task_id=wt.assigned_task,
                details=f"No activity since {wt.last_activity.isoformat()}",
            )
            new_events.append(event)
            self.stats.stalls_detected += 1

            if self.on_stall:
                self.on_stall(event)

            await self.worktree_manager.mark_stalled(wt.worktree_id)

            # Auto-reassign if configured
            if self.config.auto_reassign_stalled and self.task_dispatcher:
                recovery_event = await self._attempt_recovery(wt)
                if recovery_event:
                    new_events.append(recovery_event)

        # Detect abandoned worktrees
        abandoned = self.worktree_manager.get_abandoned_worktrees()
        for wt in abandoned:
            if wt.status == "destroyed":
                continue
            event = HealthEvent(
                event_type="abandoned",
                worktree_id=wt.worktree_id,
                task_id=wt.assigned_task,
                details=f"Idle since {wt.last_activity.isoformat()}",
            )
            new_events.append(event)

            if self.config.auto_cleanup_abandoned:
                await self.worktree_manager.mark_abandoned(wt.worktree_id)
                await self.worktree_manager.destroy(wt.worktree_id, force=True)
                self.stats.abandoned_cleaned += 1

        self.events.extend(new_events)
        return new_events

    async def _attempt_recovery(self, wt: WorktreeState) -> HealthEvent | None:
        """Attempt to recover a stalled worktree by reassigning its task.

        Returns:
            A recovery event, or None if no recovery was attempted.
        """
        attempts = self._recovery_attempts.get(wt.worktree_id, 0)
        if attempts >= self.config.max_recovery_attempts:
            event = HealthEvent(
                event_type="recovery_failed",
                worktree_id=wt.worktree_id,
                task_id=wt.assigned_task,
                details=f"Max recovery attempts ({self.config.max_recovery_attempts}) reached",
            )
            self.stats.recoveries_failed += 1
            if self.on_recovery:
                self.on_recovery(event)
            return event

        self._recovery_attempts[wt.worktree_id] = attempts + 1
        self.stats.recoveries_attempted += 1

        # If there's a task assigned, fail it (triggers retry in dispatcher)
        if wt.assigned_task and self.task_dispatcher:
            task = self.task_dispatcher._tasks.get(wt.assigned_task)
            if task and task.status in ("assigned", "running"):
                self.task_dispatcher.fail(
                    wt.assigned_task,
                    error=f"Worktree {wt.worktree_id} stalled",
                )

        self.stats.recoveries_succeeded += 1
        event = HealthEvent(
            event_type="recovery_succeeded",
            worktree_id=wt.worktree_id,
            task_id=wt.assigned_task,
            details="Task returned to queue for reassignment",
        )
        if self.on_recovery:
            self.on_recovery(event)
        return event

    async def start(self) -> None:
        """Start the periodic health check loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info("watchdog_started interval=%.1fs", self.config.check_interval_seconds)

    async def stop(self) -> None:
        """Stop the periodic health check loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("watchdog_stopped")

    async def _loop(self) -> None:
        """Internal loop for periodic checks."""
        while self._running:
            try:
                await self.check_all()
            except Exception as e:  # noqa: BLE001
                logger.warning("watchdog_check_error: %s", e)
            await asyncio.sleep(self.config.check_interval_seconds)


__all__ = [
    "HealthWatchdog",
    "WatchdogConfig",
    "HealthEvent",
    "RecoveryStats",
]
