"""
Audit Scheduler.

Provides automated audit scheduling with:
- Cron-like time-based scheduling
- Webhook triggers (Git push, file upload)
- CI/CD integration hooks
- Recurring audit jobs

Usage:
    from aragora.scheduler import AuditScheduler, ScheduleConfig

    scheduler = AuditScheduler()

    # Schedule daily audit at 2 AM
    scheduler.add_schedule(ScheduleConfig(
        name="nightly_security_scan",
        cron="0 2 * * *",
        preset="Code Security",
        document_scope="workspace:ws_123",
    ))

    # Start the scheduler
    await scheduler.start()
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


class TriggerType(str, Enum):
    """Types of schedule triggers."""

    CRON = "cron"  # Time-based cron schedule
    WEBHOOK = "webhook"  # HTTP webhook trigger
    GIT_PUSH = "git_push"  # Git push event
    FILE_UPLOAD = "file_upload"  # New file uploaded
    MANUAL = "manual"  # Manual trigger
    INTERVAL = "interval"  # Fixed interval


class ScheduleStatus(str, Enum):
    """Status of a scheduled job."""

    ACTIVE = "active"
    PAUSED = "paused"
    DISABLED = "disabled"
    RUNNING = "running"
    ERROR = "error"


@dataclass
class ScheduleConfig:
    """Configuration for a scheduled audit."""

    name: str
    description: str = ""

    # Trigger configuration
    trigger_type: TriggerType = TriggerType.CRON
    cron: Optional[str] = None  # Cron expression (e.g., "0 2 * * *")
    interval_minutes: Optional[int] = None  # For interval trigger
    webhook_secret: Optional[str] = None  # For webhook verification

    # Audit configuration
    preset: Optional[str] = None  # Preset name to use
    audit_types: list[str] = field(default_factory=list)  # Or specific types
    custom_config: dict[str, Any] = field(default_factory=dict)

    # Scope
    workspace_id: Optional[str] = None
    document_scope: Optional[str] = None  # "workspace:id", "folder:path", "tag:name"
    document_ids: list[str] = field(default_factory=list)

    # Options
    notify_on_complete: bool = True
    notify_on_findings: bool = True
    finding_severity_threshold: str = "medium"  # Notify for this severity and above
    max_retries: int = 3
    timeout_minutes: int = 60

    # Metadata
    created_by: Optional[str] = None
    tags: list[str] = field(default_factory=list)


@dataclass
class ScheduledJob:
    """A scheduled audit job instance."""

    job_id: str
    schedule_id: str
    config: ScheduleConfig
    status: ScheduleStatus
    next_run: Optional[datetime] = None
    last_run: Optional[datetime] = None
    last_result: Optional[dict] = None
    run_count: int = 0
    error_count: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "job_id": self.job_id,
            "schedule_id": self.schedule_id,
            "name": self.config.name,
            "status": self.status.value,
            "trigger_type": self.config.trigger_type.value,
            "next_run": self.next_run.isoformat() if self.next_run else None,
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "run_count": self.run_count,
            "error_count": self.error_count,
        }


@dataclass
class JobRun:
    """Record of a single job execution."""

    run_id: str
    job_id: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    status: str = "running"
    session_id: Optional[str] = None
    findings_count: int = 0
    error_message: Optional[str] = None
    duration_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "job_id": self.job_id,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "status": self.status,
            "session_id": self.session_id,
            "findings_count": self.findings_count,
            "error_message": self.error_message,
            "duration_ms": self.duration_ms,
        }


class CronParser:
    """Simple cron expression parser."""

    @staticmethod
    def parse(expression: str) -> dict[str, list[int]]:
        """
        Parse a cron expression into field ranges.

        Format: minute hour day_of_month month day_of_week
        Example: "0 2 * * *" = 2:00 AM daily
        """
        parts = expression.strip().split()
        if len(parts) != 5:
            raise ValueError(f"Invalid cron expression: {expression}")

        fields = ["minute", "hour", "day", "month", "weekday"]
        ranges = [
            (0, 59),  # minute
            (0, 23),  # hour
            (1, 31),  # day
            (1, 12),  # month
            (0, 6),  # weekday (0=Sunday)
        ]

        result = {}
        for i, (field_name, part) in enumerate(zip(fields, parts)):
            result[field_name] = CronParser._parse_field(part, ranges[i])

        return result

    @staticmethod
    def _parse_field(field: str, value_range: tuple[int, int]) -> list[int]:
        """Parse a single cron field."""
        min_val, max_val = value_range

        if field == "*":
            return list(range(min_val, max_val + 1))

        if "/" in field:
            base, step = field.split("/")
            step = int(step)
            if base == "*":
                return list(range(min_val, max_val + 1, step))
            else:
                start = int(base)
                return list(range(start, max_val + 1, step))

        if "-" in field:
            start, end = field.split("-")
            return list(range(int(start), int(end) + 1))

        if "," in field:
            return [int(v) for v in field.split(",")]

        return [int(field)]

    @staticmethod
    def next_run(expression: str, after: Optional[datetime] = None) -> datetime:
        """Calculate the next run time for a cron expression."""
        if after is None:
            after = datetime.utcnow()

        parsed = CronParser.parse(expression)

        # Start from the next minute
        candidate = after.replace(second=0, microsecond=0) + timedelta(minutes=1)

        # Find the next matching time (limit iterations to prevent infinite loop)
        for _ in range(366 * 24 * 60):  # Max ~1 year of minutes
            if (
                candidate.minute in parsed["minute"]
                and candidate.hour in parsed["hour"]
                and candidate.day in parsed["day"]
                and candidate.month in parsed["month"]
                and candidate.weekday() in parsed["weekday"]
            ):
                return candidate
            candidate += timedelta(minutes=1)

        raise ValueError(f"Could not find next run time for: {expression}")


class AuditScheduler:
    """
    Scheduler for automated audit jobs.

    Supports cron schedules, webhooks, and event triggers.
    """

    def __init__(self):
        """Initialize the scheduler."""
        self._jobs: dict[str, ScheduledJob] = {}
        self._runs: dict[str, JobRun] = {}
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._callbacks: dict[str, list[Callable]] = {
            "job_started": [],
            "job_completed": [],
            "job_failed": [],
            "findings_detected": [],
        }

    def add_schedule(self, config: ScheduleConfig) -> ScheduledJob:
        """
        Add a new scheduled audit job.

        Args:
            config: Schedule configuration

        Returns:
            The created ScheduledJob
        """
        schedule_id = f"sched_{uuid4().hex[:8]}"
        job_id = f"job_{uuid4().hex[:8]}"

        # Calculate next run time
        next_run = None
        if config.trigger_type == TriggerType.CRON and config.cron:
            next_run = CronParser.next_run(config.cron)
        elif config.trigger_type == TriggerType.INTERVAL and config.interval_minutes:
            next_run = datetime.utcnow() + timedelta(minutes=config.interval_minutes)

        job = ScheduledJob(
            job_id=job_id,
            schedule_id=schedule_id,
            config=config,
            status=ScheduleStatus.ACTIVE,
            next_run=next_run,
        )

        self._jobs[job_id] = job
        logger.info(f"Added scheduled job: {config.name} (ID: {job_id})")

        return job

    def remove_schedule(self, job_id: str) -> bool:
        """Remove a scheduled job."""
        if job_id in self._jobs:
            del self._jobs[job_id]
            logger.info(f"Removed scheduled job: {job_id}")
            return True
        return False

    def pause_schedule(self, job_id: str) -> bool:
        """Pause a scheduled job."""
        if job_id in self._jobs:
            self._jobs[job_id].status = ScheduleStatus.PAUSED
            return True
        return False

    def resume_schedule(self, job_id: str) -> bool:
        """Resume a paused scheduled job."""
        if job_id in self._jobs:
            job = self._jobs[job_id]
            if job.status == ScheduleStatus.PAUSED:
                job.status = ScheduleStatus.ACTIVE
                # Recalculate next run
                if job.config.trigger_type == TriggerType.CRON and job.config.cron:
                    job.next_run = CronParser.next_run(job.config.cron)
                return True
        return False

    def get_job(self, job_id: str) -> Optional[ScheduledJob]:
        """Get a job by ID."""
        return self._jobs.get(job_id)

    def list_jobs(
        self,
        status: Optional[ScheduleStatus] = None,
        workspace_id: Optional[str] = None,
    ) -> list[ScheduledJob]:
        """List all scheduled jobs, optionally filtered."""
        jobs = list(self._jobs.values())

        if status:
            jobs = [j for j in jobs if j.status == status]

        if workspace_id:
            jobs = [j for j in jobs if j.config.workspace_id == workspace_id]

        return jobs

    def get_job_history(self, job_id: str, limit: int = 10) -> list[JobRun]:
        """Get run history for a job."""
        runs = [r for r in self._runs.values() if r.job_id == job_id]
        runs.sort(key=lambda r: r.started_at, reverse=True)
        return runs[:limit]

    async def trigger_job(self, job_id: str) -> Optional[JobRun]:
        """Manually trigger a job execution."""
        if job_id not in self._jobs:
            return None

        job = self._jobs[job_id]
        return await self._execute_job(job)

    async def handle_webhook(
        self,
        webhook_id: str,
        payload: dict[str, Any],
        signature: Optional[str] = None,
    ) -> list[JobRun]:
        """
        Handle an incoming webhook trigger.

        Args:
            webhook_id: ID of the webhook configuration
            payload: Webhook payload data
            signature: Optional HMAC signature for verification

        Returns:
            List of triggered job runs
        """
        runs = []

        # Find jobs configured for this webhook
        for job in self._jobs.values():
            if job.config.trigger_type != TriggerType.WEBHOOK:
                continue
            if job.status != ScheduleStatus.ACTIVE:
                continue

            # Verify signature if secret is configured
            if job.config.webhook_secret and signature:
                if not self._verify_webhook_signature(
                    payload, signature, job.config.webhook_secret
                ):
                    logger.warning(f"Webhook signature verification failed for job {job.job_id}")
                    continue

            # Execute the job
            run = await self._execute_job(job)
            if run:
                runs.append(run)

        return runs

    async def handle_git_push(
        self,
        repository: str,
        branch: str,
        commit_sha: str,
        changed_files: list[str],
    ) -> list[JobRun]:
        """
        Handle a Git push event trigger.

        Args:
            repository: Repository identifier
            branch: Branch name
            commit_sha: Commit SHA
            changed_files: List of changed file paths

        Returns:
            List of triggered job runs
        """
        runs = []

        for job in self._jobs.values():
            if job.config.trigger_type != TriggerType.GIT_PUSH:
                continue
            if job.status != ScheduleStatus.ACTIVE:
                continue

            # Check if this push matches the job's scope
            # (In a full implementation, would check branch patterns, paths, etc.)

            run = await self._execute_job(
                job,
                context={
                    "trigger": "git_push",
                    "repository": repository,
                    "branch": branch,
                    "commit": commit_sha,
                    "changed_files": changed_files,
                },
            )
            if run:
                runs.append(run)

        return runs

    async def handle_file_upload(
        self,
        workspace_id: str,
        document_ids: list[str],
    ) -> list[JobRun]:
        """
        Handle a file upload event trigger.

        Args:
            workspace_id: Workspace where files were uploaded
            document_ids: IDs of uploaded documents

        Returns:
            List of triggered job runs
        """
        runs = []

        for job in self._jobs.values():
            if job.config.trigger_type != TriggerType.FILE_UPLOAD:
                continue
            if job.status != ScheduleStatus.ACTIVE:
                continue
            if job.config.workspace_id and job.config.workspace_id != workspace_id:
                continue

            # Override document scope with uploaded files
            run = await self._execute_job(
                job,
                context={
                    "trigger": "file_upload",
                    "document_ids": document_ids,
                },
            )
            if run:
                runs.append(run)

        return runs

    def on(self, event: str, callback: Callable) -> None:
        """Register an event callback."""
        if event in self._callbacks:
            self._callbacks[event].append(callback)

    async def start(self) -> None:
        """Start the scheduler background task."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._scheduler_loop())
        logger.info("Audit scheduler started")

    async def stop(self) -> None:
        """Stop the scheduler."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Audit scheduler stopped")

    async def _scheduler_loop(self) -> None:
        """Main scheduler loop."""
        while self._running:
            try:
                now = datetime.utcnow()

                # Check each job for due execution
                for job in list(self._jobs.values()):
                    if job.status != ScheduleStatus.ACTIVE:
                        continue

                    if job.next_run and job.next_run <= now:
                        # Execute the job
                        asyncio.create_task(self._execute_job(job))

                        # Calculate next run time
                        if job.config.trigger_type == TriggerType.CRON and job.config.cron:
                            job.next_run = CronParser.next_run(job.config.cron)
                        elif (
                            job.config.trigger_type == TriggerType.INTERVAL
                            and job.config.interval_minutes
                        ):
                            job.next_run = now + timedelta(minutes=job.config.interval_minutes)

                # Sleep until next check (every minute)
                await asyncio.sleep(60)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scheduler loop error: {e}")
                await asyncio.sleep(60)

    async def _execute_job(
        self,
        job: ScheduledJob,
        context: Optional[dict[str, Any]] = None,
    ) -> Optional[JobRun]:
        """Execute a scheduled job."""
        run_id = f"run_{uuid4().hex[:12]}"
        run = JobRun(
            run_id=run_id,
            job_id=job.job_id,
            started_at=datetime.utcnow(),
        )
        self._runs[run_id] = run

        job.status = ScheduleStatus.RUNNING
        job.last_run = run.started_at
        job.run_count += 1

        # Emit job_started event
        await self._emit("job_started", job, run)

        try:
            # Get document auditor
            from aragora.audit import get_document_auditor, AuditConfig

            auditor = get_document_auditor()

            # Determine document IDs
            document_ids = job.config.document_ids.copy()
            if context and "document_ids" in context:
                document_ids = context["document_ids"]

            # Create audit session
            session = await auditor.create_session(
                document_ids=document_ids,
                audit_types=job.config.audit_types if job.config.audit_types else None,
                name=f"Scheduled: {job.config.name}",
                model=job.config.custom_config.get("model", "gemini-1.5-flash"),
            )

            run.session_id = session.id

            # Run the audit
            result = await asyncio.wait_for(
                auditor.run_audit(session.id),
                timeout=job.config.timeout_minutes * 60,
            )

            run.findings_count = len(result.findings)
            run.status = "completed"
            run.completed_at = datetime.utcnow()
            run.duration_ms = int((run.completed_at - run.started_at).total_seconds() * 1000)

            job.status = ScheduleStatus.ACTIVE
            job.last_result = {
                "session_id": session.id,
                "findings_count": run.findings_count,
                "status": "completed",
            }

            # Emit completion events
            await self._emit("job_completed", job, run)

            if run.findings_count > 0 and job.config.notify_on_findings:
                await self._emit("findings_detected", job, run, result.findings)

            logger.info(
                f"Job {job.job_id} completed: {run.findings_count} findings in {run.duration_ms}ms"
            )

        except asyncio.TimeoutError:
            run.status = "timeout"
            run.error_message = f"Job timed out after {job.config.timeout_minutes} minutes"
            run.completed_at = datetime.utcnow()
            job.status = ScheduleStatus.ACTIVE
            job.error_count += 1
            await self._emit("job_failed", job, run)
            logger.error(f"Job {job.job_id} timed out")

        except Exception as e:
            run.status = "error"
            run.error_message = str(e)
            run.completed_at = datetime.utcnow()
            job.status = (
                ScheduleStatus.ERROR
                if job.error_count >= job.config.max_retries
                else ScheduleStatus.ACTIVE
            )
            job.error_count += 1
            await self._emit("job_failed", job, run)
            logger.error(f"Job {job.job_id} failed: {e}")

        return run

    async def _emit(self, event: str, *args: Any) -> None:
        """Emit an event to registered callbacks."""
        for callback in self._callbacks.get(event, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(*args)
                else:
                    callback(*args)
            except Exception as e:
                logger.error(f"Callback error for {event}: {e}")

    def _verify_webhook_signature(
        self,
        payload: dict[str, Any],
        signature: str,
        secret: str,
    ) -> bool:
        """Verify a webhook signature."""
        import hmac
        import json

        payload_bytes = json.dumps(payload, sort_keys=True).encode()
        expected = hmac.new(
            secret.encode(),
            payload_bytes,
            hashlib.sha256,
        ).hexdigest()

        return hmac.compare_digest(signature, expected)


# Singleton scheduler instance
_scheduler: Optional[AuditScheduler] = None


def get_scheduler() -> AuditScheduler:
    """Get the global scheduler instance."""
    global _scheduler
    if _scheduler is None:
        _scheduler = AuditScheduler()
    return _scheduler


__all__ = [
    "AuditScheduler",
    "ScheduleConfig",
    "ScheduledJob",
    "JobRun",
    "TriggerType",
    "ScheduleStatus",
    "CronParser",
    "get_scheduler",
]
