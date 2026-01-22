"""
Aragora Scheduler Module.

Provides automated scheduling for:
- Audit scheduling with cron expressions and webhook triggers
- Access review automation (SOC 2 CC6.1)
- DR drill scheduling (SOC 2 CC9)
- Secrets rotation (SOC 2 CC6.2)

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

SOC 2 Compliance Modules:
    from aragora.scheduler import (
        AccessReviewScheduler,
        get_access_review_scheduler,
        DRDrillScheduler,
        get_dr_drill_scheduler,
        SecretsRotationScheduler,
        get_secrets_rotation_scheduler,
    )
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

from .access_review_scheduler import (
    AccessReviewScheduler,
    AccessReviewConfig,
    AccessReview,
    AccessReviewItem,
    ReviewType,
    ReviewStatus,
    ReviewItemStatus,
    get_access_review_scheduler,
    schedule_access_review,
)

from .dr_drill_scheduler import (
    DRDrillScheduler,
    DRDrillConfig,
    DRDrillResult,
    DrillStep,
    DrillType,
    DrillStatus,
    ComponentType,
    get_dr_drill_scheduler,
    schedule_dr_drill,
)

from .secrets_rotation_scheduler import (
    SecretsRotationScheduler,
    SecretsRotationConfig,
    SecretMetadata,
    RotationResult,
    SecretType,
    RotationStatus,
    RotationTrigger,
    get_secrets_rotation_scheduler,
    rotate_secret,
)

__all__ = [
    # Audit scheduler
    "AuditScheduler",
    "ScheduleConfig",
    "ScheduledJob",
    "JobRun",
    "TriggerType",
    "ScheduleStatus",
    "CronParser",
    "get_scheduler",
    # Access review scheduler (SOC 2 CC6.1)
    "AccessReviewScheduler",
    "AccessReviewConfig",
    "AccessReview",
    "AccessReviewItem",
    "ReviewType",
    "ReviewStatus",
    "ReviewItemStatus",
    "get_access_review_scheduler",
    "schedule_access_review",
    # DR drill scheduler (SOC 2 CC9)
    "DRDrillScheduler",
    "DRDrillConfig",
    "DRDrillResult",
    "DrillStep",
    "DrillType",
    "DrillStatus",
    "ComponentType",
    "get_dr_drill_scheduler",
    "schedule_dr_drill",
    # Secrets rotation scheduler (SOC 2 CC6.2)
    "SecretsRotationScheduler",
    "SecretsRotationConfig",
    "SecretMetadata",
    "RotationResult",
    "SecretType",
    "RotationStatus",
    "RotationTrigger",
    "get_secrets_rotation_scheduler",
    "rotate_secret",
]
