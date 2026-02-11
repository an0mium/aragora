"""
Notification Service.

Multi-channel notification system supporting:
- Slack (channel messages, direct messages)
- Email (SMTP, templates)
- Webhooks (configurable endpoints)

Usage:
    from aragora.notifications import (
        NotificationService,
        get_notification_service,
        Notification,
        NotificationChannel,
    )

    service = get_notification_service()

    # Send notification
    await service.notify(
        notification=Notification(
            title="Critical Finding Detected",
            message="A new critical vulnerability was found...",
            severity="critical",
            resource_type="finding",
            resource_id="f-123",
        ),
        channels=[NotificationChannel.SLACK, NotificationChannel.EMAIL],
        recipients=["security-team"],
    )
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from .models import (
    EmailConfig,
    Notification,
    NotificationChannel,
    NotificationPriority,
    NotificationResult,
    SlackConfig,
    WebhookEndpoint,
)
from .providers import (
    EmailProvider,
    NotificationProvider,
    SlackProvider,
    WebhookProvider,
    _record_notification_metric,
)

logger = logging.getLogger(__name__)

# Re-export everything for backward compatibility
__all__ = [
    # Models
    "NotificationChannel",
    "NotificationPriority",
    "Notification",
    "NotificationResult",
    "SlackConfig",
    "EmailConfig",
    "WebhookEndpoint",
    # Providers
    "NotificationProvider",
    "SlackProvider",
    "EmailProvider",
    "WebhookProvider",
    # Service
    "NotificationService",
    # Global access
    "get_notification_service",
    "init_notification_service",
    # Convenience functions
    "notify_finding_created",
    "notify_audit_completed",
    "notify_checkpoint_approval_requested",
    "notify_checkpoint_escalation",
    "notify_checkpoint_resolved",
    "notify_webhook_delivery_failure",
    "notify_webhook_circuit_breaker_opened",
    "notify_batch_job_failed",
    "notify_batch_job_completed",
    # Internal helpers (used by tests)
    "_severity_to_priority",
    "_record_notification_metric",
]


class NotificationService:
    """
    Main notification service orchestrating multiple channels.

    Handles routing notifications to appropriate channels and
    managing provider configurations.
    """

    def __init__(
        self,
        slack_config: SlackConfig | None = None,
        email_config: EmailConfig | None = None,
    ):
        self.providers: dict[NotificationChannel, NotificationProvider] = {}

        # Initialize providers
        if slack_config is None:
            slack_config = SlackConfig.from_env()
        self.providers[NotificationChannel.SLACK] = SlackProvider(slack_config)

        if email_config is None:
            email_config = EmailConfig.from_env()
        self.providers[NotificationChannel.EMAIL] = EmailProvider(email_config)

        self.providers[NotificationChannel.WEBHOOK] = WebhookProvider()

        # Notification history (in-memory, could be persisted)
        self._history: list[tuple[Notification, list[NotificationResult]]] = []
        self._history_limit = 1000

    def get_provider(self, channel: NotificationChannel) -> NotificationProvider | None:
        """Get a provider by channel."""
        return self.providers.get(channel)

    @property
    def webhook_provider(self) -> WebhookProvider:
        """Get the webhook provider for endpoint management."""
        provider = self.providers[NotificationChannel.WEBHOOK]
        if not isinstance(provider, WebhookProvider):
            raise TypeError(f"Expected WebhookProvider, got {type(provider).__name__}")
        return provider

    def get_configured_channels(self) -> list[NotificationChannel]:
        """Get list of configured channels."""
        return [channel for channel, provider in self.providers.items() if provider.is_configured()]

    async def notify(
        self,
        notification: Notification,
        channels: list[NotificationChannel] | None = None,
        recipients: dict[NotificationChannel, list[str] | None] = None,
    ) -> list[NotificationResult]:
        """
        Send notification to specified channels and recipients.

        Args:
            notification: The notification to send
            channels: Channels to use (defaults to all configured)
            recipients: Channel -> recipients mapping

        Returns:
            List of results from each channel/recipient
        """
        if channels is None:
            channels = self.get_configured_channels()

        results = []

        for channel in channels:
            provider = self.providers.get(channel)
            if not provider or not provider.is_configured():
                continue

            # Get recipients for this channel
            channel_recipients: list[str] = []
            if recipients and channel in recipients:
                channel_recipients = recipients[channel] or []
            else:
                # Default recipients
                channel_recipients = self._get_default_recipients(channel, notification)

            for recipient in channel_recipients:
                result = await provider.send(notification, recipient)
                results.append(result)

        # Store in history
        self._add_to_history(notification, results)

        return results

    async def notify_all_webhooks(
        self,
        notification: Notification,
        event_type: str,
    ) -> list[NotificationResult]:
        """Send to all webhooks matching the event type."""
        webhook_provider = self.webhook_provider
        return await webhook_provider.send_to_matching(notification, event_type)

    def _get_default_recipients(
        self,
        channel: NotificationChannel,
        notification: Notification,
    ) -> list[str]:
        """Get default recipients for a channel based on notification."""
        if channel == NotificationChannel.SLACK:
            provider = self.providers[channel]
            if isinstance(provider, SlackProvider):
                return [provider.config.default_channel]

        if channel == NotificationChannel.WEBHOOK:
            # Return all enabled webhook endpoint IDs
            webhook_provider = self.webhook_provider
            return [ep.id for ep in webhook_provider.endpoints.values() if ep.enabled]

        return []

    def _add_to_history(
        self,
        notification: Notification,
        results: list[NotificationResult],
    ) -> None:
        """Add notification to history."""
        self._history.append((notification, results))
        if len(self._history) > self._history_limit:
            self._history = self._history[-self._history_limit :]

    def get_history(
        self,
        limit: int = 100,
        channel: NotificationChannel | None = None,
    ) -> list[dict]:
        """Get notification history."""
        history = []
        for notification, results in reversed(self._history):
            if channel:
                results = [r for r in results if r.channel == channel]
                if not results:
                    continue

            history.append(
                {
                    "notification": notification.to_dict(),
                    "results": [r.to_dict() for r in results],
                }
            )

            if len(history) >= limit:
                break

        return history


# =============================================================================
# Convenience notification functions
# =============================================================================


def _severity_to_priority(severity: str) -> NotificationPriority:
    """Map severity to notification priority."""
    mapping = {
        "critical": NotificationPriority.URGENT,
        "high": NotificationPriority.HIGH,
        "medium": NotificationPriority.NORMAL,
        "low": NotificationPriority.LOW,
        "info": NotificationPriority.LOW,
    }
    return mapping.get(severity.lower(), NotificationPriority.NORMAL)


async def notify_finding_created(
    finding_id: str,
    title: str,
    severity: str,
    workspace_id: str,
    details: str | None = None,
) -> list[NotificationResult]:
    """Send notification for new finding."""
    service = get_notification_service()

    notification = Notification(
        title=f"New Finding: {title}",
        message=details or f"A new {severity} severity finding has been detected.",
        severity=severity,
        priority=_severity_to_priority(severity),
        resource_type="finding",
        resource_id=finding_id,
        workspace_id=workspace_id,
        action_label="View Finding",
    )

    results = await service.notify(notification)
    await service.notify_all_webhooks(notification, "finding.created")

    return results


async def notify_audit_completed(
    session_id: str,
    workspace_id: str,
    finding_count: int,
    critical_count: int,
) -> list[NotificationResult]:
    """Send notification for completed audit."""
    service = get_notification_service()

    severity = "critical" if critical_count > 0 else "info"

    notification = Notification(
        title="Audit Completed",
        message=(
            f"Audit session completed with {finding_count} findings ({critical_count} critical)."
        ),
        severity=severity,
        resource_type="audit_session",
        resource_id=session_id,
        workspace_id=workspace_id,
        action_label="View Results",
    )

    results = await service.notify(notification)
    await service.notify_all_webhooks(notification, "audit.completed")

    return results


# =============================================================================
# Human Checkpoint Notifications
# =============================================================================


async def notify_checkpoint_approval_requested(
    request_id: str,
    workflow_id: str,
    step_id: str,
    title: str,
    description: str,
    workspace_id: str | None = None,
    assignees: list[str] | None = None,
    timeout_seconds: float | None = None,
    action_url: str | None = None,
) -> list[NotificationResult]:
    """
    Send notification when a human checkpoint approval is requested.

    Args:
        request_id: ID of the approval request
        workflow_id: ID of the workflow
        step_id: ID of the checkpoint step
        title: Title of the approval request
        description: Description for the approver
        workspace_id: Optional workspace ID
        assignees: Optional list of assignee emails/slack handles
        timeout_seconds: Timeout before escalation
        action_url: URL to view/respond to the approval request

    Returns:
        List of notification results
    """
    service = get_notification_service()

    timeout_info = ""
    if timeout_seconds:
        hours = int(timeout_seconds // 3600)
        minutes = int((timeout_seconds % 3600) // 60)
        if hours > 0:
            timeout_info = f"\n\nThis request will timeout in {hours}h {minutes}m."
        else:
            timeout_info = f"\n\nThis request will timeout in {minutes} minutes."

    notification = Notification(
        title=f"Approval Required: {title}",
        message=f"{description}{timeout_info}",
        severity="warning",
        priority=NotificationPriority.HIGH,
        resource_type="approval_request",
        resource_id=request_id,
        workspace_id=workspace_id,
        action_url=action_url,
        action_label="Review & Approve",
        metadata={
            "workflow_id": workflow_id,
            "step_id": step_id,
            "timeout_seconds": timeout_seconds,
        },
    )

    # Build recipients mapping
    recipients: dict[NotificationChannel, list[str]] = {}
    if assignees:
        # Split by channel type
        slack_recipients = [a for a in assignees if a.startswith("#") or a.startswith("@")]
        email_recipients = [a for a in assignees if "@" in a and not a.startswith("@")]

        if slack_recipients:
            recipients[NotificationChannel.SLACK] = slack_recipients
        if email_recipients:
            recipients[NotificationChannel.EMAIL] = email_recipients

    # Send to configured channels (or specified recipients)
    results = await service.notify(
        notification,
        recipients=recipients if recipients else None,
    )

    # Also send to webhooks
    await service.notify_all_webhooks(notification, "checkpoint.approval_requested")

    return results


async def notify_checkpoint_escalation(
    request_id: str,
    workflow_id: str,
    step_id: str,
    title: str,
    escalation_emails: list[str],
    workspace_id: str | None = None,
    original_timeout_seconds: float | None = None,
    action_url: str | None = None,
) -> list[NotificationResult]:
    """
    Send escalation notification when a checkpoint approval times out.

    Args:
        request_id: ID of the approval request
        workflow_id: ID of the workflow
        step_id: ID of the checkpoint step
        title: Title of the approval request
        escalation_emails: List of emails to escalate to
        workspace_id: Optional workspace ID
        original_timeout_seconds: The original timeout that was exceeded
        action_url: URL to view/respond to the approval request

    Returns:
        List of notification results
    """
    service = get_notification_service()

    timeout_info = ""
    if original_timeout_seconds:
        hours = int(original_timeout_seconds // 3600)
        minutes = int((original_timeout_seconds % 3600) // 60)
        if hours > 0:
            timeout_info = f" after {hours}h {minutes}m"
        else:
            timeout_info = f" after {minutes} minutes"

    notification = Notification(
        title=f"ESCALATION: {title}",
        message=f"An approval request has timed out{timeout_info} and requires immediate attention.",
        severity="critical",
        priority=NotificationPriority.URGENT,
        resource_type="approval_request",
        resource_id=request_id,
        workspace_id=workspace_id,
        action_url=action_url,
        action_label="Review Urgently",
        metadata={
            "workflow_id": workflow_id,
            "step_id": step_id,
            "escalation": True,
        },
    )

    # Send to escalation recipients
    recipients = {
        NotificationChannel.EMAIL: escalation_emails,
    }

    # Also try Slack if escalation emails contain Slack handles
    slack_recipients = [e for e in escalation_emails if e.startswith("#") or e.startswith("@")]
    if slack_recipients:
        recipients[NotificationChannel.SLACK] = slack_recipients

    results = await service.notify(
        notification,
        recipients=recipients,
    )

    # Also send to webhooks
    await service.notify_all_webhooks(notification, "checkpoint.escalation")

    return results


async def notify_checkpoint_resolved(
    request_id: str,
    workflow_id: str,
    step_id: str,
    title: str,
    status: str,  # approved, rejected
    responder_id: str | None = None,
    responder_notes: str | None = None,
    workspace_id: str | None = None,
) -> list[NotificationResult]:
    """
    Send notification when a checkpoint approval is resolved.

    Args:
        request_id: ID of the approval request
        workflow_id: ID of the workflow
        step_id: ID of the checkpoint step
        title: Title of the approval request
        status: Resolution status (approved/rejected)
        responder_id: ID of the person who responded
        responder_notes: Notes from the responder
        workspace_id: Optional workspace ID

    Returns:
        List of notification results
    """
    service = get_notification_service()

    if status == "approved":
        severity = "info"
        status_text = "APPROVED"
        emoji = "✓"
    else:
        severity = "warning"
        status_text = "REJECTED"
        emoji = "✗"

    message_parts = [f"Checkpoint '{title}' has been {status_text.lower()}."]
    if responder_id:
        message_parts.append(f"Resolved by: {responder_id}")
    if responder_notes:
        message_parts.append(f"Notes: {responder_notes}")

    notification = Notification(
        title=f"{emoji} Checkpoint {status_text}: {title}",
        message="\n".join(message_parts),
        severity=severity,
        priority=NotificationPriority.NORMAL,
        resource_type="approval_request",
        resource_id=request_id,
        workspace_id=workspace_id,
        metadata={
            "workflow_id": workflow_id,
            "step_id": step_id,
            "status": status,
            "responder_id": responder_id,
        },
    )

    results = await service.notify(notification)

    # Also send to webhooks
    await service.notify_all_webhooks(notification, f"checkpoint.{status}")

    return results


# =============================================================================
# Webhook Delivery Failure Notifications
# =============================================================================


async def notify_webhook_delivery_failure(
    webhook_id: str,
    webhook_url: str,
    event_type: str,
    error_message: str,
    attempt_count: int,
    workspace_id: str | None = None,
    owner_email: str | None = None,
) -> list[NotificationResult]:
    """
    Send notification when a webhook delivery fails.

    Args:
        webhook_id: ID of the webhook
        webhook_url: URL of the webhook endpoint
        event_type: Type of event being delivered
        error_message: Error message from delivery attempt
        attempt_count: Number of delivery attempts made
        workspace_id: Optional workspace ID
        owner_email: Email of webhook owner to notify

    Returns:
        List of notification results
    """
    service = get_notification_service()

    # Determine severity based on attempt count
    if attempt_count >= 5:
        severity = "critical"
        priority = NotificationPriority.URGENT
        title = "Webhook Delivery Failed - Max Retries Exceeded"
    elif attempt_count >= 3:
        severity = "error"
        priority = NotificationPriority.HIGH
        title = "Webhook Delivery Failing - Multiple Retries"
    else:
        severity = "warning"
        priority = NotificationPriority.NORMAL
        title = "Webhook Delivery Failed"

    # Truncate URL for display
    display_url = webhook_url if len(webhook_url) <= 50 else webhook_url[:47] + "..."

    notification = Notification(
        title=title,
        message=(
            f"Webhook delivery to {display_url} failed.\n\n"
            f"Event type: {event_type}\n"
            f"Attempt: {attempt_count}\n"
            f"Error: {error_message}"
        ),
        severity=severity,
        priority=priority,
        resource_type="webhook",
        resource_id=webhook_id,
        workspace_id=workspace_id,
        action_label="View Webhook",
        metadata={
            "webhook_url": webhook_url,
            "event_type": event_type,
            "attempt_count": attempt_count,
            "error_message": error_message,
        },
    )

    # Build recipients
    recipients: dict[NotificationChannel, list[str]] = {}
    if owner_email:
        recipients[NotificationChannel.EMAIL] = [owner_email]

    results = await service.notify(
        notification,
        recipients=recipients if recipients else None,
    )

    return results


async def notify_webhook_circuit_breaker_opened(
    webhook_id: str,
    webhook_url: str,
    failure_count: int,
    cooldown_seconds: float,
    workspace_id: str | None = None,
    owner_email: str | None = None,
) -> list[NotificationResult]:
    """
    Send notification when a webhook's circuit breaker opens.

    Args:
        webhook_id: ID of the webhook
        webhook_url: URL of the webhook endpoint
        failure_count: Number of failures that triggered the circuit breaker
        cooldown_seconds: Cooldown period before retrying
        workspace_id: Optional workspace ID
        owner_email: Email of webhook owner to notify

    Returns:
        List of notification results
    """
    service = get_notification_service()

    cooldown_minutes = int(cooldown_seconds / 60)
    display_url = webhook_url if len(webhook_url) <= 50 else webhook_url[:47] + "..."

    notification = Notification(
        title="Webhook Circuit Breaker Opened",
        message=(
            f"The circuit breaker for webhook {display_url} has opened "
            f"after {failure_count} consecutive failures.\n\n"
            f"Deliveries will be paused for {cooldown_minutes} minutes before retrying.\n"
            f"Please check that the webhook endpoint is accessible and returning success responses."
        ),
        severity="critical",
        priority=NotificationPriority.URGENT,
        resource_type="webhook",
        resource_id=webhook_id,
        workspace_id=workspace_id,
        action_label="Check Webhook",
        metadata={
            "webhook_url": webhook_url,
            "failure_count": failure_count,
            "cooldown_seconds": cooldown_seconds,
            "circuit_state": "open",
        },
    )

    recipients: dict[NotificationChannel, list[str]] = {}
    if owner_email:
        recipients[NotificationChannel.EMAIL] = [owner_email]

    results = await service.notify(
        notification,
        recipients=recipients if recipients else None,
    )

    # Also send to webhooks (other endpoints might want to know)
    await service.notify_all_webhooks(notification, "webhook.circuit_breaker_opened")

    return results


async def notify_batch_job_failed(
    job_id: str,
    total_debates: int,
    success_count: int,
    failure_count: int,
    error_message: str | None = None,
    workspace_id: str | None = None,
    user_email: str | None = None,
) -> list[NotificationResult]:
    """
    Send notification when a batch explainability job fails.

    Args:
        job_id: ID of the batch job
        total_debates: Total debates in the batch
        success_count: Number of successful explanations
        failure_count: Number of failed explanations
        error_message: Optional error message
        workspace_id: Optional workspace ID
        user_email: Email of user who created the job

    Returns:
        List of notification results
    """
    service = get_notification_service()

    if failure_count == total_debates:
        severity = "critical"
        title = "Batch Explainability Job Failed Completely"
    elif failure_count > success_count:
        severity = "error"
        title = "Batch Explainability Job Mostly Failed"
    else:
        severity = "warning"
        title = "Batch Explainability Job Partially Failed"

    message_parts = [
        f"Batch job {job_id[:12]}... has completed with failures.",
        "",
        "Results:",
        f"- Total debates: {total_debates}",
        f"- Successful: {success_count}",
        f"- Failed: {failure_count}",
    ]
    if error_message:
        message_parts.append("")
        message_parts.append(f"Error: {error_message}")

    notification = Notification(
        title=title,
        message="\n".join(message_parts),
        severity=severity,
        priority=(
            NotificationPriority.HIGH
            if failure_count > success_count
            else NotificationPriority.NORMAL
        ),
        resource_type="batch_job",
        resource_id=job_id,
        workspace_id=workspace_id,
        action_label="View Results",
        metadata={
            "total_debates": total_debates,
            "success_count": success_count,
            "failure_count": failure_count,
        },
    )

    recipients: dict[NotificationChannel, list[str]] = {}
    if user_email:
        recipients[NotificationChannel.EMAIL] = [user_email]

    results = await service.notify(
        notification,
        recipients=recipients if recipients else None,
    )

    return results


async def notify_batch_job_completed(
    job_id: str,
    total_debates: int,
    success_count: int,
    elapsed_seconds: float,
    workspace_id: str | None = None,
    user_email: str | None = None,
) -> list[NotificationResult]:
    """
    Send notification when a batch explainability job completes successfully.

    Args:
        job_id: ID of the batch job
        total_debates: Total debates processed
        success_count: Number of successful explanations
        elapsed_seconds: Total processing time
        workspace_id: Optional workspace ID
        user_email: Email of user who created the job

    Returns:
        List of notification results
    """
    service = get_notification_service()

    elapsed_min = int(elapsed_seconds / 60)
    elapsed_sec = int(elapsed_seconds % 60)
    time_str = f"{elapsed_min}m {elapsed_sec}s" if elapsed_min > 0 else f"{elapsed_sec}s"

    notification = Notification(
        title="Batch Explainability Job Completed",
        message=(
            f"Batch job {job_id[:12]}... has completed successfully.\n\n"
            f"Processed {total_debates} debates in {time_str}.\n"
            f"All {success_count} explanations generated successfully."
        ),
        severity="info",
        priority=NotificationPriority.NORMAL,
        resource_type="batch_job",
        resource_id=job_id,
        workspace_id=workspace_id,
        action_label="View Results",
        metadata={
            "total_debates": total_debates,
            "success_count": success_count,
            "elapsed_seconds": elapsed_seconds,
        },
    )

    recipients: dict[NotificationChannel, list[str]] = {}
    if user_email:
        recipients[NotificationChannel.EMAIL] = [user_email]

    results = await service.notify(
        notification,
        recipients=recipients if recipients else None,
    )

    return results


# =============================================================================
# Global instance
# =============================================================================

_notification_service: NotificationService | None = None
_lock = threading.Lock()


def get_notification_service() -> NotificationService:
    """Get the global notification service instance."""
    global _notification_service

    if _notification_service is None:
        with _lock:
            if _notification_service is None:
                _notification_service = NotificationService()

    return _notification_service


def init_notification_service(
    slack_config: SlackConfig | None = None,
    email_config: EmailConfig | None = None,
) -> NotificationService:
    """Initialize the global notification service with custom config."""
    global _notification_service

    with _lock:
        _notification_service = NotificationService(
            slack_config=slack_config,
            email_config=email_config,
        )

    return _notification_service
