"""
Sync Scheduler for Enterprise Connectors.

Provides cron-based scheduling, webhook handling, and sync history tracking.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from aragora.connectors.enterprise.base import EnterpriseConnector, SyncResult

logger = logging.getLogger(__name__)


class SyncStatus(Enum):
    """Status of a sync job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class RetryPolicy:
    """Exponential backoff retry policy for sync operations."""

    max_retries: int = 5
    base_delay: float = 1.0  # seconds
    max_delay: float = 300.0  # 5 minutes
    exponential_base: float = 2.0
    jitter: bool = True  # Add randomness to prevent thundering herd

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for a given attempt number (0-indexed)."""
        import random

        # Exponential backoff
        delay = self.base_delay * (self.exponential_base**attempt)

        # Cap at max delay
        delay = min(delay, self.max_delay)

        # Add jitter (up to 25% variation)
        if self.jitter:
            jitter_range = delay * 0.25
            delay += random.uniform(-jitter_range, jitter_range)

        return max(0, delay)

    async def execute_with_retry(
        self,
        func: Callable,
        *args: Any,
        on_retry: Optional[Callable[[int, Exception], None]] = None,
        **kwargs: Any,
    ) -> Any:
        """
        Execute a function with retry on failure.

        Args:
            func: Async function to execute
            *args: Positional arguments for func
            on_retry: Optional callback called on each retry with (attempt, exception)
            **kwargs: Keyword arguments for func

        Returns:
            Result from func

        Raises:
            Exception: Last exception if all retries exhausted
        """
        last_exception: Optional[Exception] = None

        for attempt in range(self.max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except asyncio.CancelledError:
                # Don't retry on cancellation
                raise
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries:
                    delay = self.calculate_delay(attempt)
                    logger.warning(
                        f"Retry {attempt + 1}/{self.max_retries} after {delay:.1f}s: {e}"
                    )
                    if on_retry:
                        on_retry(attempt + 1, e)
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All {self.max_retries} retries exhausted: {e}")

        if last_exception:
            raise last_exception

        # This shouldn't happen, but satisfy type checker
        raise RuntimeError("Unexpected retry state")


@dataclass
class SyncSchedule:
    """
    Schedule configuration for a connector sync.

    Supports cron-like expressions and intervals.
    """

    # Schedule type
    schedule_type: str = "interval"  # "cron", "interval", "webhook_only"

    # For interval scheduling
    interval_minutes: int = 60

    # For cron scheduling (cron expression)
    cron_expression: Optional[str] = None

    # Schedule constraints
    enabled: bool = True
    start_time: Optional[datetime] = None  # Don't run before this
    end_time: Optional[datetime] = None  # Don't run after this
    max_concurrent: int = 1  # Max concurrent syncs for this job

    # Retry configuration
    retry_on_failure: bool = True
    max_retries: int = 3
    retry_delay_minutes: int = 5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schedule_type": self.schedule_type,
            "interval_minutes": self.interval_minutes,
            "cron_expression": self.cron_expression,
            "enabled": self.enabled,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "max_concurrent": self.max_concurrent,
            "retry_on_failure": self.retry_on_failure,
            "max_retries": self.max_retries,
            "retry_delay_minutes": self.retry_delay_minutes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SyncSchedule":
        return cls(
            schedule_type=data.get("schedule_type", "interval"),
            interval_minutes=data.get("interval_minutes", 60),
            cron_expression=data.get("cron_expression"),
            enabled=data.get("enabled", True),
            start_time=(
                datetime.fromisoformat(data["start_time"]) if data.get("start_time") else None
            ),
            end_time=datetime.fromisoformat(data["end_time"]) if data.get("end_time") else None,
            max_concurrent=data.get("max_concurrent", 1),
            retry_on_failure=data.get("retry_on_failure", True),
            max_retries=data.get("max_retries", 3),
            retry_delay_minutes=data.get("retry_delay_minutes", 5),
        )


@dataclass
class SyncHistory:
    """Record of a sync execution."""

    id: str
    job_id: str
    connector_id: str
    tenant_id: str
    status: SyncStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    items_synced: int = 0
    items_total: int = 0
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> Optional[float]:
        if self.completed_at and self.started_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "job_id": self.job_id,
            "connector_id": self.connector_id,
            "tenant_id": self.tenant_id,
            "status": self.status.value,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "items_synced": self.items_synced,
            "items_total": self.items_total,
            "errors": self.errors,
            "duration_seconds": self.duration_seconds,
            "metadata": self.metadata,
        }


@dataclass
class SyncJob:
    """
    A scheduled sync job for a connector.
    """

    id: str
    connector_id: str
    tenant_id: str
    schedule: SyncSchedule
    connector: Optional["EnterpriseConnector"] = None

    # Runtime state
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    current_run_id: Optional[str] = None
    consecutive_failures: int = 0

    # Callbacks
    on_complete: Optional[Callable[["SyncResult"], None]] = None
    on_error: Optional[Callable[[Exception], None]] = None

    def __post_init__(self):
        if self.next_run is None and self.schedule.enabled:
            self._calculate_next_run()

    def _calculate_next_run(self):
        """Calculate the next run time based on schedule."""
        now = datetime.now(timezone.utc)

        if self.schedule.schedule_type == "interval":
            if self.last_run:
                self.next_run = self.last_run + timedelta(minutes=self.schedule.interval_minutes)
                # If next run is in the past, schedule from now
                if self.next_run < now:
                    self.next_run = now + timedelta(minutes=self.schedule.interval_minutes)
            else:
                self.next_run = now + timedelta(minutes=1)  # Start in 1 minute

        elif self.schedule.schedule_type == "cron" and self.schedule.cron_expression:
            self.next_run = self._parse_cron_next(self.schedule.cron_expression)

        elif self.schedule.schedule_type == "webhook_only":
            self.next_run = None  # Only triggered by webhooks

        # Apply time constraints
        if self.next_run:
            if self.schedule.start_time and self.next_run < self.schedule.start_time:
                self.next_run = self.schedule.start_time
            if self.schedule.end_time and self.next_run > self.schedule.end_time:
                self.next_run = None  # Don't schedule if past end time

    def _parse_cron_next(self, cron_expr: str) -> Optional[datetime]:
        """Parse cron expression and get next run time."""
        try:
            from croniter import croniter  # type: ignore[import-untyped]

            cron = croniter(cron_expr, datetime.now(timezone.utc))
            return cron.get_next(datetime)
        except ImportError:
            logger.warning("croniter not installed, falling back to interval")
            return datetime.now(timezone.utc) + timedelta(hours=1)
        except Exception as e:
            logger.error(f"Invalid cron expression '{cron_expr}': {e}")
            return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "connector_id": self.connector_id,
            "tenant_id": self.tenant_id,
            "schedule": self.schedule.to_dict(),
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "next_run": self.next_run.isoformat() if self.next_run else None,
            "consecutive_failures": self.consecutive_failures,
        }


class SyncScheduler:
    """
    Scheduler for managing connector sync jobs.

    Features:
    - Cron and interval-based scheduling
    - Multi-tenant job isolation
    - Webhook triggering
    - Sync history tracking
    - Concurrent execution limits
    """

    def __init__(
        self,
        state_dir: Optional[Path] = None,
        max_concurrent_syncs: int = 5,
        history_retention_days: int = 30,
    ):
        self.state_dir = state_dir or Path.home() / ".aragora" / "sync"
        self.max_concurrent_syncs = max_concurrent_syncs
        self.history_retention_days = history_retention_days

        self._jobs: Dict[str, SyncJob] = {}
        self._connectors: Dict[str, "EnterpriseConnector"] = {}
        self._history: List[SyncHistory] = []
        self._running_syncs: Dict[str, asyncio.Task] = {}
        self._scheduler_task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

        # Ensure state directory exists
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def register_connector(
        self,
        connector: "EnterpriseConnector",
        schedule: Optional[SyncSchedule] = None,
        tenant_id: str = "default",
    ) -> SyncJob:
        """
        Register a connector with an optional schedule.

        Returns the created SyncJob.
        """
        job_id = f"{tenant_id}:{connector.connector_id}"

        job = SyncJob(
            id=job_id,
            connector_id=connector.connector_id,
            tenant_id=tenant_id,
            schedule=schedule or SyncSchedule(schedule_type="webhook_only"),
            connector=connector,
        )

        self._jobs[job_id] = job
        self._connectors[connector.connector_id] = connector

        logger.info(f"Registered connector: {connector.name} (job_id={job_id})")
        return job

    def unregister_connector(self, connector_id: str, tenant_id: str = "default"):
        """Unregister a connector and remove its job."""
        job_id = f"{tenant_id}:{connector_id}"

        if job_id in self._jobs:
            del self._jobs[job_id]

        if connector_id in self._connectors:
            del self._connectors[connector_id]

        logger.info(f"Unregistered connector: {connector_id}")

    def get_job(self, job_id: str) -> Optional[SyncJob]:
        """Get a sync job by ID."""
        return self._jobs.get(job_id)

    def list_jobs(self, tenant_id: Optional[str] = None) -> List[SyncJob]:
        """List all jobs, optionally filtered by tenant."""
        jobs = list(self._jobs.values())
        if tenant_id:
            jobs = [j for j in jobs if j.tenant_id == tenant_id]
        return sorted(jobs, key=lambda j: j.next_run or datetime.max.replace(tzinfo=timezone.utc))

    async def trigger_sync(
        self,
        connector_id: str,
        tenant_id: str = "default",
        full_sync: bool = False,
    ) -> Optional[str]:
        """
        Manually trigger a sync for a connector.

        Returns the run ID if started, None if already running.
        """
        job_id = f"{tenant_id}:{connector_id}"
        job = self._jobs.get(job_id)

        if not job:
            logger.warning(f"No job found for {job_id}")
            return None

        if job.current_run_id:
            logger.warning(f"Sync already running for {job_id}")
            return job.current_run_id

        return await self._execute_sync(job, full_sync=full_sync)

    async def handle_webhook(
        self,
        connector_id: str,
        payload: Dict[str, Any],
        tenant_id: str = "default",
    ) -> bool:
        """
        Handle a webhook event for a connector.

        Returns True if the webhook triggered a sync.
        """
        job_id = f"{tenant_id}:{connector_id}"
        job = self._jobs.get(job_id)

        if not job or not job.connector:
            return False

        # Let the connector handle the webhook
        handled = await job.connector.handle_webhook(payload)

        if handled:
            logger.info(f"Webhook triggered sync for {connector_id}")

        return handled

    async def _execute_sync(self, job: SyncJob, full_sync: bool = False) -> str:
        """Execute a sync for a job."""
        run_id = str(uuid.uuid4())[:8]
        job.current_run_id = run_id

        # Create history record
        history = SyncHistory(
            id=run_id,
            job_id=job.id,
            connector_id=job.connector_id,
            tenant_id=job.tenant_id,
            status=SyncStatus.RUNNING,
            started_at=datetime.now(timezone.utc),
        )
        self._history.append(history)

        try:
            if not job.connector:
                raise ValueError(f"No connector registered for job {job.id}")

            # Check concurrent sync limit
            if len(self._running_syncs) >= self.max_concurrent_syncs:
                history.status = SyncStatus.PENDING
                logger.warning(f"Max concurrent syncs reached, queueing {job.id}")
                return run_id

            # Run sync
            logger.info(f"Starting sync for {job.connector_id} (run_id={run_id})")

            result = await job.connector.sync(full_sync=full_sync)

            # Update history
            history.status = SyncStatus.COMPLETED if result.success else SyncStatus.FAILED
            history.completed_at = datetime.now(timezone.utc)
            history.items_synced = result.items_synced
            history.items_total = result.items_total
            history.errors = result.errors

            # Update job state
            job.last_run = datetime.now(timezone.utc)
            job._calculate_next_run()
            job.current_run_id = None

            if result.success:
                job.consecutive_failures = 0
                if job.on_complete:
                    job.on_complete(result)
            else:
                job.consecutive_failures += 1

            logger.info(
                f"Sync completed for {job.connector_id}: "
                f"{result.items_synced}/{result.items_total} items "
                f"({history.duration_seconds:.1f}s)"
            )

        except Exception as e:
            logger.error(f"Sync failed for {job.connector_id}: {e}")
            history.status = SyncStatus.FAILED
            history.completed_at = datetime.now(timezone.utc)
            history.errors.append(str(e))

            job.consecutive_failures += 1
            job.current_run_id = None

            if job.on_error:
                job.on_error(e)

            # Schedule retry if configured
            if (
                job.schedule.retry_on_failure
                and job.consecutive_failures <= job.schedule.max_retries
            ):
                job.next_run = datetime.now(timezone.utc) + timedelta(
                    minutes=job.schedule.retry_delay_minutes
                )
                logger.info(f"Retry scheduled for {job.connector_id} at {job.next_run}")

        finally:
            self._running_syncs.pop(run_id, None)

        return run_id

    async def start(self):
        """Start the scheduler loop."""
        if self._scheduler_task:
            return

        self._stop_event.clear()
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        logger.info("Sync scheduler started")

    async def stop(self):
        """Stop the scheduler loop."""
        self._stop_event.set()

        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
            self._scheduler_task = None

        # Cancel running syncs
        for run_id, task in list(self._running_syncs.items()):
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self._running_syncs.clear()
        logger.info("Sync scheduler stopped")

    async def _scheduler_loop(self):
        """Main scheduler loop with exponential backoff on errors."""
        consecutive_errors = 0
        error_backoff = RetryPolicy(
            max_retries=999,  # Keep trying
            base_delay=10.0,
            max_delay=300.0,
            exponential_base=1.5,
            jitter=True,
        )

        while not self._stop_event.is_set():
            try:
                now = datetime.now(timezone.utc)

                # Find jobs that are due
                for job in self._jobs.values():
                    if not job.schedule.enabled:
                        continue

                    if job.next_run and job.next_run <= now:
                        if not job.current_run_id:
                            # Create task for sync
                            task = asyncio.create_task(self._execute_sync(job))
                            self._running_syncs[job.id] = task

                # Clean up old history
                self._cleanup_history()

                # Reset error count on successful iteration
                consecutive_errors = 0

                # Wait before next check
                await asyncio.sleep(10)

            except asyncio.CancelledError:
                break
            except Exception as e:
                consecutive_errors += 1
                delay = error_backoff.calculate_delay(consecutive_errors - 1)

                logger.error(
                    f"Scheduler loop error (attempt {consecutive_errors}): {e}. "
                    f"Retrying in {delay:.1f}s",
                    exc_info=consecutive_errors <= 3,  # Full trace only for first few
                )

                # Alert if we're having persistent issues
                if consecutive_errors == 5:
                    logger.critical(
                        f"Scheduler experiencing persistent errors: {consecutive_errors} consecutive failures"
                    )

                await asyncio.sleep(delay)

    def _cleanup_history(self):
        """Remove old history entries."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.history_retention_days)
        self._history = [h for h in self._history if h.started_at >= cutoff]

    def get_history(
        self,
        job_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        status: Optional[SyncStatus] = None,
        limit: int = 100,
    ) -> List[SyncHistory]:
        """Get sync history with optional filters."""
        history = self._history

        if job_id:
            history = [h for h in history if h.job_id == job_id]
        if tenant_id:
            history = [h for h in history if h.tenant_id == tenant_id]
        if status:
            history = [h for h in history if h.status == status]

        return sorted(history, key=lambda h: h.started_at, reverse=True)[:limit]

    def get_stats(self, tenant_id: Optional[str] = None) -> Dict[str, Any]:
        """Get scheduler statistics."""
        jobs = self.list_jobs(tenant_id)
        history = self.get_history(tenant_id=tenant_id)

        # Calculate stats
        total_syncs = len(history)
        successful_syncs = len([h for h in history if h.status == SyncStatus.COMPLETED])
        failed_syncs = len([h for h in history if h.status == SyncStatus.FAILED])

        total_items = sum(h.items_synced for h in history)
        avg_duration = (
            sum(h.duration_seconds or 0 for h in history) / total_syncs if total_syncs > 0 else 0
        )

        return {
            "total_jobs": len(jobs),
            "enabled_jobs": len([j for j in jobs if j.schedule.enabled]),
            "running_syncs": len(self._running_syncs),
            "total_syncs": total_syncs,
            "successful_syncs": successful_syncs,
            "failed_syncs": failed_syncs,
            "success_rate": successful_syncs / total_syncs if total_syncs > 0 else 1.0,
            "total_items_synced": total_items,
            "average_duration_seconds": avg_duration,
        }

    async def save_state(self):
        """Save scheduler state to disk."""
        state = {
            "jobs": {job_id: job.to_dict() for job_id, job in self._jobs.items()},
            "history": [h.to_dict() for h in self._history[-1000:]],  # Keep last 1000
        }

        state_file = self.state_dir / "scheduler_state.json"
        with open(state_file, "w") as f:
            json.dump(state, f, indent=2, default=str)

        logger.debug(f"Saved scheduler state to {state_file}")

    async def load_state(self):
        """Load scheduler state from disk."""
        state_file = self.state_dir / "scheduler_state.json"

        if not state_file.exists():
            return

        try:
            with open(state_file) as f:
                state = json.load(f)

            # Restore history
            for h_data in state.get("history", []):
                history = SyncHistory(
                    id=h_data["id"],
                    job_id=h_data["job_id"],
                    connector_id=h_data["connector_id"],
                    tenant_id=h_data["tenant_id"],
                    status=SyncStatus(h_data["status"]),
                    started_at=datetime.fromisoformat(h_data["started_at"]),
                    completed_at=(
                        datetime.fromisoformat(h_data["completed_at"])
                        if h_data.get("completed_at")
                        else None
                    ),
                    items_synced=h_data.get("items_synced", 0),
                    items_total=h_data.get("items_total", 0),
                    errors=h_data.get("errors", []),
                )
                self._history.append(history)

            logger.info(f"Loaded scheduler state: {len(self._history)} history entries")

        except Exception as e:
            logger.warning(f"Failed to load scheduler state: {e}")
