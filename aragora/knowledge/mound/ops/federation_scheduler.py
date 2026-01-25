"""
Federation Scheduler for Knowledge Mound.

Provides automated federation sync scheduling with:
- Cron-based scheduling for push/pull sync
- Interval-based sync (e.g., every 15 minutes)
- Manual trigger support
- Sync history and status tracking

Usage:
    from aragora.knowledge.mound.ops.federation_scheduler import (
        FederationScheduler,
        FederationScheduleConfig,
        get_federation_scheduler,
    )

    scheduler = get_federation_scheduler()

    # Schedule sync every 15 minutes
    scheduler.add_schedule(FederationScheduleConfig(
        name="production_sync",
        region_id="us-west-2",
        sync_mode="bidirectional",
        cron="*/15 * * * *",  # Every 15 minutes
        workspace_id="ws_123",
    ))

    # Start the scheduler
    await scheduler.start()
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


class SyncMode(str, Enum):
    """Federation sync mode."""

    PUSH = "push"  # Push local changes to remote
    PULL = "pull"  # Pull remote changes to local
    BIDIRECTIONAL = "bidirectional"  # Both push and pull


class SyncScope(str, Enum):
    """Scope of data to sync."""

    FULL = "full"  # Full item data
    SUMMARY = "summary"  # Content summary only
    METADATA = "metadata"  # Metadata only


class ScheduleStatus(str, Enum):
    """Status of a scheduled sync."""

    ACTIVE = "active"
    PAUSED = "paused"
    DISABLED = "disabled"
    RUNNING = "running"
    ERROR = "error"


class TriggerType(str, Enum):
    """Types of schedule triggers."""

    CRON = "cron"  # Cron-based scheduling
    INTERVAL = "interval"  # Fixed interval
    MANUAL = "manual"  # Manual trigger only


@dataclass
class FederationScheduleConfig:
    """Configuration for a scheduled federation sync."""

    name: str
    region_id: str
    workspace_id: str
    sync_mode: SyncMode = SyncMode.BIDIRECTIONAL
    sync_scope: SyncScope = SyncScope.SUMMARY
    description: str = ""

    # Trigger configuration
    trigger_type: TriggerType = TriggerType.CRON
    cron: Optional[str] = None  # Cron expression
    interval_minutes: int = 15  # For interval trigger

    # Sync options
    visibility_levels: List[str] = field(default_factory=lambda: ["organization", "public"])
    max_items_per_sync: int = 1000
    conflict_resolution: str = "remote_wins"  # remote_wins, local_wins, newest_wins

    # Retry options
    max_retries: int = 3
    retry_delay_seconds: int = 60
    timeout_seconds: int = 300

    # Notifications
    notify_on_complete: bool = False
    notify_on_error: bool = True

    # Metadata
    enabled: bool = True
    created_by: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class ScheduledSync:
    """A scheduled federation sync job."""

    schedule_id: str
    config: FederationScheduleConfig
    status: ScheduleStatus = ScheduleStatus.ACTIVE
    next_run: Optional[datetime] = None
    last_run: Optional[datetime] = None
    last_result: Optional[Dict[str, Any]] = None
    run_count: int = 0
    error_count: int = 0
    consecutive_errors: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "schedule_id": self.schedule_id,
            "name": self.config.name,
            "region_id": self.config.region_id,
            "workspace_id": self.config.workspace_id,
            "sync_mode": self.config.sync_mode.value,
            "sync_scope": self.config.sync_scope.value,
            "status": self.status.value,
            "trigger_type": self.config.trigger_type.value,
            "cron": self.config.cron,
            "interval_minutes": self.config.interval_minutes,
            "next_run": self.next_run.isoformat() if self.next_run else None,
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "run_count": self.run_count,
            "error_count": self.error_count,
            "enabled": self.config.enabled,
        }


@dataclass
class SyncRun:
    """Record of a single sync execution."""

    run_id: str
    schedule_id: str
    region_id: str
    workspace_id: str
    sync_mode: SyncMode
    started_at: datetime
    completed_at: Optional[datetime] = None
    status: str = "running"
    items_pushed: int = 0
    items_pulled: int = 0
    items_conflicted: int = 0
    error_message: Optional[str] = None
    duration_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "schedule_id": self.schedule_id,
            "region_id": self.region_id,
            "workspace_id": self.workspace_id,
            "sync_mode": self.sync_mode.value,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "status": self.status,
            "items_pushed": self.items_pushed,
            "items_pulled": self.items_pulled,
            "items_conflicted": self.items_conflicted,
            "error_message": self.error_message,
            "duration_ms": self.duration_ms,
        }


class CronParser:
    """Simple cron expression parser."""

    @staticmethod
    def next_run(cron: str, from_time: Optional[datetime] = None) -> datetime:
        """
        Calculate the next run time for a cron expression.

        Format: minute hour day_of_month month day_of_week
        Example: "*/15 * * * *" = every 15 minutes
        """
        now = from_time or datetime.now()
        fields = CronParser.parse(cron)

        # Simple implementation - find next matching minute
        candidate = now.replace(second=0, microsecond=0) + timedelta(minutes=1)

        for _ in range(60 * 24 * 7):  # Search up to a week
            if (
                candidate.minute in fields["minute"]
                and candidate.hour in fields["hour"]
                and candidate.day in fields["day"]
                and candidate.month in fields["month"]
                and candidate.weekday() in fields["weekday"]
            ):
                return candidate
            candidate += timedelta(minutes=1)

        # Fallback: 1 hour from now
        return now + timedelta(hours=1)

    @staticmethod
    def parse(expression: str) -> Dict[str, List[int]]:
        """Parse a cron expression into field ranges."""
        parts = expression.strip().split()
        if len(parts) != 5:
            raise ValueError(f"Invalid cron expression: {expression}")

        fields = ["minute", "hour", "day", "month", "weekday"]
        ranges = [
            (0, 59),  # minute
            (0, 23),  # hour
            (1, 31),  # day
            (1, 12),  # month
            (0, 6),  # weekday (0=Sunday in cron, but Python uses 0=Monday)
        ]

        result = {}
        for i, (field_name, part) in enumerate(zip(fields, parts)):
            result[field_name] = CronParser._parse_field(part, ranges[i])

        # Convert weekday from cron (0=Sunday) to Python (0=Monday)
        result["weekday"] = [(w - 1) % 7 for w in result["weekday"]]

        return result

    @staticmethod
    def _parse_field(field_str: str, value_range: tuple) -> List[int]:
        """Parse a single cron field."""
        min_val, max_val = value_range

        if field_str == "*":
            return list(range(min_val, max_val + 1))

        if "/" in field_str:
            base, step = field_str.split("/")
            step = int(step)
            if base == "*":
                return list(range(min_val, max_val + 1, step))
            else:
                start = int(base)
                return list(range(start, max_val + 1, step))

        if "-" in field_str:
            range_start, range_end = field_str.split("-")
            return list(range(int(range_start), int(range_end) + 1))

        if "," in field_str:
            return [int(v) for v in field_str.split(",")]

        return [int(field_str)]


class FederationScheduler:
    """
    Manages scheduled federation sync operations.

    Provides:
    - Cron-based scheduling for regular syncs
    - Interval-based scheduling
    - Manual trigger support
    - Sync history and status tracking
    """

    def __init__(
        self,
        sync_callback: Optional[Callable] = None,
    ):
        """
        Initialize the federation scheduler.

        Args:
            sync_callback: Optional callback to execute sync operations.
                           If not provided, uses internal sync logic.
        """
        self._schedules: Dict[str, ScheduledSync] = {}
        self._sync_history: List[SyncRun] = []
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._sync_callback = sync_callback
        self._max_history = 1000

    def add_schedule(
        self,
        config: FederationScheduleConfig,
    ) -> ScheduledSync:
        """
        Add a new federation sync schedule.

        Args:
            config: Schedule configuration

        Returns:
            Created ScheduledSync
        """
        schedule_id = f"fed_sched_{uuid4().hex[:12]}"

        # Calculate initial next run
        if config.trigger_type == TriggerType.CRON and config.cron:
            next_run = CronParser.next_run(config.cron)
        elif config.trigger_type == TriggerType.INTERVAL:
            next_run = datetime.now() + timedelta(minutes=config.interval_minutes)
        else:
            next_run = None

        schedule = ScheduledSync(
            schedule_id=schedule_id,
            config=config,
            status=ScheduleStatus.ACTIVE if config.enabled else ScheduleStatus.PAUSED,
            next_run=next_run,
        )

        self._schedules[schedule_id] = schedule
        logger.info(
            f"Added federation schedule '{config.name}' for region {config.region_id}, "
            f"next run: {next_run}"
        )

        return schedule

    def remove_schedule(self, schedule_id: str) -> bool:
        """Remove a schedule."""
        if schedule_id in self._schedules:
            del self._schedules[schedule_id]
            logger.info(f"Removed federation schedule {schedule_id}")
            return True
        return False

    def pause_schedule(self, schedule_id: str) -> bool:
        """Pause a schedule."""
        if schedule_id in self._schedules:
            self._schedules[schedule_id].status = ScheduleStatus.PAUSED
            logger.info(f"Paused federation schedule {schedule_id}")
            return True
        return False

    def resume_schedule(self, schedule_id: str) -> bool:
        """Resume a paused schedule."""
        schedule = self._schedules.get(schedule_id)
        if schedule:
            schedule.status = ScheduleStatus.ACTIVE
            # Recalculate next run
            if schedule.config.trigger_type == TriggerType.CRON and schedule.config.cron:
                schedule.next_run = CronParser.next_run(schedule.config.cron)
            elif schedule.config.trigger_type == TriggerType.INTERVAL:
                schedule.next_run = datetime.now() + timedelta(
                    minutes=schedule.config.interval_minutes
                )
            logger.info(f"Resumed federation schedule {schedule_id}")
            return True
        return False

    def get_schedule(self, schedule_id: str) -> Optional[ScheduledSync]:
        """Get a schedule by ID."""
        return self._schedules.get(schedule_id)

    def list_schedules(
        self,
        region_id: Optional[str] = None,
        workspace_id: Optional[str] = None,
        status: Optional[ScheduleStatus] = None,
    ) -> List[ScheduledSync]:
        """
        List schedules with optional filters.

        Args:
            region_id: Filter by region
            workspace_id: Filter by workspace
            status: Filter by status

        Returns:
            List of matching schedules
        """
        schedules = list(self._schedules.values())

        if region_id:
            schedules = [s for s in schedules if s.config.region_id == region_id]

        if workspace_id:
            schedules = [s for s in schedules if s.config.workspace_id == workspace_id]

        if status:
            schedules = [s for s in schedules if s.status == status]

        return schedules

    def get_history(
        self,
        schedule_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[SyncRun]:
        """Get sync history."""
        history = self._sync_history

        if schedule_id:
            history = [h for h in history if h.schedule_id == schedule_id]

        return history[:limit]

    async def trigger_sync(self, schedule_id: str) -> Optional[SyncRun]:
        """
        Manually trigger a sync for a schedule.

        Args:
            schedule_id: Schedule to trigger

        Returns:
            SyncRun if executed, None if schedule not found
        """
        schedule = self._schedules.get(schedule_id)
        if not schedule:
            return None

        return await self._execute_sync(schedule)

    async def start(self) -> None:
        """Start the scheduler background task."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._run_scheduler())
        logger.info("Federation scheduler started")

    async def stop(self) -> None:
        """Stop the scheduler."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Federation scheduler stopped")

    async def _run_scheduler(self) -> None:
        """Main scheduler loop."""
        while self._running:
            try:
                now = datetime.now()

                for schedule in list(self._schedules.values()):
                    if schedule.status != ScheduleStatus.ACTIVE:
                        continue

                    if schedule.next_run and schedule.next_run <= now:
                        # Time to run this schedule
                        asyncio.create_task(self._execute_sync(schedule))

                # Sleep for 10 seconds before next check
                await asyncio.sleep(10)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in federation scheduler loop: {e}")
                await asyncio.sleep(30)

    async def _execute_sync(self, schedule: ScheduledSync) -> SyncRun:
        """Execute a sync operation."""
        schedule.status = ScheduleStatus.RUNNING

        run = SyncRun(
            run_id=f"sync_run_{uuid4().hex[:12]}",
            schedule_id=schedule.schedule_id,
            region_id=schedule.config.region_id,
            workspace_id=schedule.config.workspace_id,
            sync_mode=schedule.config.sync_mode,
            started_at=datetime.now(),
        )

        try:
            logger.info(
                f"Starting federation sync: {schedule.config.name} "
                f"({schedule.config.sync_mode.value})"
            )

            if self._sync_callback:
                # Use provided callback
                result = await self._sync_callback(
                    region_id=schedule.config.region_id,
                    workspace_id=schedule.config.workspace_id,
                    sync_mode=schedule.config.sync_mode,
                    sync_scope=schedule.config.sync_scope,
                    visibility_levels=schedule.config.visibility_levels,
                    max_items=schedule.config.max_items_per_sync,
                )
                run.items_pushed = result.get("items_pushed", 0)
                run.items_pulled = result.get("items_pulled", 0)
                run.items_conflicted = result.get("items_conflicted", 0)
            else:
                # Default: use internal sync logic
                result = await self._perform_sync(schedule.config)
                run.items_pushed = result.get("items_pushed", 0)
                run.items_pulled = result.get("items_pulled", 0)
                run.items_conflicted = result.get("items_conflicted", 0)

            run.status = "completed"
            run.completed_at = datetime.now()
            run.duration_ms = int((run.completed_at - run.started_at).total_seconds() * 1000)

            schedule.run_count += 1
            schedule.consecutive_errors = 0
            schedule.last_result = {"status": "success", **result}

            logger.info(
                f"Federation sync completed: {run.items_pushed} pushed, {run.items_pulled} pulled"
            )

        except Exception as e:
            run.status = "error"
            run.error_message = str(e)
            run.completed_at = datetime.now()
            run.duration_ms = int((run.completed_at - run.started_at).total_seconds() * 1000)

            schedule.error_count += 1
            schedule.consecutive_errors += 1
            schedule.last_result = {"status": "error", "error": str(e)}

            logger.error(f"Federation sync failed: {e}")

            # Disable after too many consecutive errors
            if schedule.consecutive_errors >= schedule.config.max_retries:
                schedule.status = ScheduleStatus.ERROR
                logger.warning(
                    f"Schedule {schedule.config.name} disabled after "
                    f"{schedule.consecutive_errors} consecutive errors"
                )
        finally:
            if schedule.status == ScheduleStatus.RUNNING:
                schedule.status = ScheduleStatus.ACTIVE

            schedule.last_run = run.started_at

            # Calculate next run
            if schedule.config.trigger_type == TriggerType.CRON and schedule.config.cron:
                schedule.next_run = CronParser.next_run(schedule.config.cron)
            elif schedule.config.trigger_type == TriggerType.INTERVAL:
                schedule.next_run = datetime.now() + timedelta(
                    minutes=schedule.config.interval_minutes
                )

            # Add to history
            self._sync_history.insert(0, run)
            if len(self._sync_history) > self._max_history:
                self._sync_history = self._sync_history[: self._max_history]

        return run

    async def _perform_sync(
        self,
        config: FederationScheduleConfig,
    ) -> Dict[str, Any]:
        """
        Perform the actual sync operation.

        This is the default implementation that uses KnowledgeMound's
        federation operations.
        """
        items_pushed = 0
        items_pulled = 0
        items_conflicted = 0

        try:
            from aragora.knowledge.mound import get_knowledge_mound

            mound = get_knowledge_mound(config.workspace_id)  # type: ignore[arg-type]

            if config.sync_mode in (SyncMode.PUSH, SyncMode.BIDIRECTIONAL):
                # Push to remote
                if hasattr(mound, "sync_to_region"):
                    result = await mound.sync_to_region(  # type: ignore[call-arg]
                        region_id=config.region_id,
                        workspace_id=config.workspace_id,
                        scope=config.sync_scope.value,
                    )
                    items_pushed = getattr(result, "nodes_synced", 0)

            if config.sync_mode in (SyncMode.PULL, SyncMode.BIDIRECTIONAL):
                # Pull from remote
                if hasattr(mound, "pull_from_region"):
                    result = await mound.pull_from_region(
                        region_id=config.region_id,
                        workspace_id=config.workspace_id,
                    )
                    items_pulled = getattr(result, "nodes_synced", 0)

        except ImportError:
            logger.warning("KnowledgeMound not available for sync")
        except Exception as e:
            logger.error(f"Error during sync: {e}")
            raise

        return {
            "items_pushed": items_pushed,
            "items_pulled": items_pulled,
            "items_conflicted": items_conflicted,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        active = sum(1 for s in self._schedules.values() if s.status == ScheduleStatus.ACTIVE)
        paused = sum(1 for s in self._schedules.values() if s.status == ScheduleStatus.PAUSED)
        error = sum(1 for s in self._schedules.values() if s.status == ScheduleStatus.ERROR)

        total_runs = sum(s.run_count for s in self._schedules.values())
        total_errors = sum(s.error_count for s in self._schedules.values())

        # Recent sync stats
        recent_runs = self._sync_history[:100]
        successful_runs = sum(1 for r in recent_runs if r.status == "completed")
        failed_runs = sum(1 for r in recent_runs if r.status == "error")

        return {
            "schedules": {
                "total": len(self._schedules),
                "active": active,
                "paused": paused,
                "error": error,
            },
            "runs": {
                "total": total_runs,
                "total_errors": total_errors,
            },
            "recent": {
                "total": len(recent_runs),
                "successful": successful_runs,
                "failed": failed_runs,
                "success_rate": (successful_runs / len(recent_runs) if recent_runs else 0),
            },
            "running": self._running,
        }


# Global scheduler instance
_federation_scheduler: Optional[FederationScheduler] = None


def get_federation_scheduler() -> FederationScheduler:
    """Get the global federation scheduler instance."""
    global _federation_scheduler
    if _federation_scheduler is None:
        _federation_scheduler = FederationScheduler()
    return _federation_scheduler


async def init_federation_scheduler() -> FederationScheduler:
    """Initialize and start the global federation scheduler."""
    scheduler = get_federation_scheduler()
    if not scheduler._running:
        await scheduler.start()
    return scheduler


__all__ = [
    "SyncMode",
    "SyncScope",
    "ScheduleStatus",
    "TriggerType",
    "FederationScheduleConfig",
    "ScheduledSync",
    "SyncRun",
    "CronParser",
    "FederationScheduler",
    "get_federation_scheduler",
    "init_federation_scheduler",
]
