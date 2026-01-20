"""
Aragora Notifications Module.

Multi-channel notification system for audit alerts and team collaboration.

Supported channels:
- Slack (webhooks and Bot API)
- Email (SMTP)
- Webhooks (custom endpoints)

Usage:
    from aragora.notifications import (
        NotificationService,
        get_notification_service,
        Notification,
        NotificationChannel,
        notify_finding_created,
    )

    # Send a notification
    service = get_notification_service()
    await service.notify(
        Notification(
            title="Critical Finding",
            message="...",
            severity="critical",
        ),
        channels=[NotificationChannel.SLACK],
    )

    # Or use convenience functions
    await notify_finding_created(
        finding_id="f-123",
        title="SQL Injection Detected",
        severity="critical",
        workspace_id="ws-456",
    )
"""

from .service import (
    NotificationService,
    NotificationChannel,
    NotificationPriority,
    Notification,
    NotificationResult,
    SlackConfig,
    EmailConfig,
    WebhookEndpoint,
    NotificationProvider,
    SlackProvider,
    EmailProvider,
    WebhookProvider,
    get_notification_service,
    init_notification_service,
    notify_finding_created,
    notify_audit_completed,
    notify_checkpoint_approval_requested,
    notify_checkpoint_escalation,
    notify_checkpoint_resolved,
)

__all__ = [
    # Core service
    "NotificationService",
    "NotificationChannel",
    "NotificationPriority",
    "Notification",
    "NotificationResult",
    # Configuration
    "SlackConfig",
    "EmailConfig",
    "WebhookEndpoint",
    # Providers
    "NotificationProvider",
    "SlackProvider",
    "EmailProvider",
    "WebhookProvider",
    # Global access
    "get_notification_service",
    "init_notification_service",
    # Convenience functions
    "notify_finding_created",
    "notify_audit_completed",
    # Checkpoint notifications
    "notify_checkpoint_approval_requested",
    "notify_checkpoint_escalation",
    "notify_checkpoint_resolved",
]
