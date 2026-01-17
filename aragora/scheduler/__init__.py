"""
Aragora Scheduler Module.

Provides automated audit scheduling with cron expressions,
webhook triggers, and CI/CD integration.

Usage:
    from aragora.scheduler import AuditScheduler, ScheduleConfig, get_scheduler

    scheduler = get_scheduler()

    # Add a daily security scan
    scheduler.add_schedule(ScheduleConfig(
        name="Daily Security Scan",
        cron="0 2 * * *",
        preset="Code Security",
        workspace_id="ws_123",
    ))

    # Add webhook-triggered audit
    scheduler.add_schedule(ScheduleConfig(
        name="PR Security Check",
        trigger_type=TriggerType.WEBHOOK,
        webhook_secret="your-secret",
        preset="Code Security",
    ))

    # Start the scheduler
    await scheduler.start()
"""

from .audit_scheduler import (
    AuditScheduler,
    ScheduleConfig,
    ScheduledJob,
    JobRun,
    TriggerType,
    ScheduleStatus,
    CronParser,
    get_scheduler,
)

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
