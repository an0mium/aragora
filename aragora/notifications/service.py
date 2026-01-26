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

import asyncio
import hashlib
import hmac
import json
import logging
import os
import smtplib
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from typing import Any, Optional

from aragora.exceptions import SlackNotificationError, WebhookDeliveryError

logger = logging.getLogger(__name__)


def _record_notification_metric(
    channel: str,
    severity: str,
    priority: str,
    success: bool,
    latency_seconds: float,
    error_type: Optional[str] = None,
) -> None:
    """Record notification metrics (imported lazily to avoid circular imports)."""
    try:
        from aragora.observability.metrics import (
            record_notification_sent,
            record_notification_error,
        )

        record_notification_sent(channel, severity, priority, success, latency_seconds)
        if not success and error_type:
            record_notification_error(channel, error_type)
    except ImportError:
        pass  # Metrics not available


class NotificationChannel(str, Enum):
    """Available notification channels."""

    SLACK = "slack"
    EMAIL = "email"
    WEBHOOK = "webhook"
    IN_APP = "in_app"


class NotificationPriority(str, Enum):
    """Notification priority levels."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class Notification:
    """A notification to be sent."""

    title: str
    message: str
    severity: str = "info"  # info, warning, error, critical
    priority: NotificationPriority = NotificationPriority.NORMAL

    # Context
    resource_type: Optional[str] = None  # finding, document, session
    resource_id: Optional[str] = None
    workspace_id: Optional[str] = None

    # Metadata
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    # Links
    action_url: Optional[str] = None
    action_label: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "message": self.message,
            "severity": self.severity,
            "priority": self.priority.value,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "workspace_id": self.workspace_id,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
            "action_url": self.action_url,
            "action_label": self.action_label,
        }


@dataclass
class NotificationResult:
    """Result of sending a notification."""

    success: bool
    channel: NotificationChannel
    recipient: str
    notification_id: str
    error: Optional[str] = None
    external_id: Optional[str] = None  # Message ID from external service

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "channel": self.channel.value,
            "recipient": self.recipient,
            "notification_id": self.notification_id,
            "error": self.error,
            "external_id": self.external_id,
        }


@dataclass
class SlackConfig:
    """Slack integration configuration."""

    webhook_url: Optional[str] = None
    bot_token: Optional[str] = None
    default_channel: str = "#notifications"
    username: str = "Aragora"
    icon_emoji: str = ":robot_face:"

    @classmethod
    def from_env(cls) -> SlackConfig:
        """Create from environment variables."""
        return cls(
            webhook_url=os.environ.get("SLACK_WEBHOOK_URL"),
            bot_token=os.environ.get("SLACK_BOT_TOKEN"),
            default_channel=os.environ.get("SLACK_DEFAULT_CHANNEL", "#notifications"),
        )


@dataclass
class EmailConfig:
    """Email (SMTP) configuration."""

    smtp_host: str = "localhost"
    smtp_port: int = 587
    smtp_user: Optional[str] = None
    smtp_password: Optional[str] = None
    use_tls: bool = True
    from_address: str = "notifications@aragora.local"
    from_name: str = "Aragora Notifications"

    @classmethod
    def from_env(cls) -> EmailConfig:
        """Create from environment variables."""
        return cls(
            smtp_host=os.environ.get("SMTP_HOST", "localhost"),
            smtp_port=int(os.environ.get("SMTP_PORT", "587")),
            smtp_user=os.environ.get("SMTP_USER"),
            smtp_password=os.environ.get("SMTP_PASSWORD"),
            use_tls=os.environ.get("SMTP_USE_TLS", "true").lower() == "true",
            from_address=os.environ.get("SMTP_FROM", "notifications@aragora.local"),
            from_name=os.environ.get("SMTP_FROM_NAME", "Aragora Notifications"),
        )


@dataclass
class WebhookEndpoint:
    """A configured webhook endpoint."""

    id: str
    url: str
    secret: Optional[str] = None
    events: list[str] = field(default_factory=list)  # Empty = all events
    headers: dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    workspace_id: Optional[str] = None

    def matches_event(self, event_type: str) -> bool:
        """Check if this endpoint should receive the event."""
        if not self.events:
            return True  # All events
        return event_type in self.events


class NotificationProvider(ABC):
    """Abstract base class for notification providers."""

    @property
    @abstractmethod
    def channel(self) -> NotificationChannel:
        """Get the channel this provider handles."""
        ...

    @abstractmethod
    async def send(
        self,
        notification: Notification,
        recipient: str,
    ) -> NotificationResult:
        """Send a notification to a recipient."""
        ...

    @abstractmethod
    def is_configured(self) -> bool:
        """Check if the provider is properly configured."""
        ...


class SlackProvider(NotificationProvider):
    """Slack notification provider."""

    def __init__(self, config: SlackConfig):
        self.config = config

    @property
    def channel(self) -> NotificationChannel:
        return NotificationChannel.SLACK

    def is_configured(self) -> bool:
        return bool(self.config.webhook_url or self.config.bot_token)

    async def send(
        self,
        notification: Notification,
        recipient: str,
    ) -> NotificationResult:
        """Send notification to Slack."""
        start_time = time.perf_counter()

        if not self.is_configured():
            latency = time.perf_counter() - start_time
            _record_notification_metric(
                "slack",
                notification.severity,
                notification.priority.value,
                False,
                latency,
                "not_configured",
            )
            return NotificationResult(
                success=False,
                channel=self.channel,
                recipient=recipient,
                notification_id=notification.id,
                error="Slack not configured",
            )

        try:
            # Build Slack message
            message = self._build_message(notification)

            if self.config.webhook_url:
                await self._send_webhook(message, recipient)
            elif self.config.bot_token:
                await self._send_api(message, recipient)

            latency = time.perf_counter() - start_time
            _record_notification_metric(
                "slack", notification.severity, notification.priority.value, True, latency
            )
            return NotificationResult(
                success=True,
                channel=self.channel,
                recipient=recipient,
                notification_id=notification.id,
            )

        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
            latency = time.perf_counter() - start_time
            error_type = "rate_limited" if "rate" in str(e).lower() else "delivery_error"
            _record_notification_metric(
                "slack",
                notification.severity,
                notification.priority.value,
                False,
                latency,
                error_type,
            )
            return NotificationResult(
                success=False,
                channel=self.channel,
                recipient=recipient,
                notification_id=notification.id,
                error=str(e),
            )

    def _build_message(self, notification: Notification) -> dict:
        """Build Slack message payload."""
        # Map severity to color
        colors = {
            "info": "#2196F3",
            "warning": "#FF9800",
            "error": "#F44336",
            "critical": "#B71C1C",
        }
        color = colors.get(notification.severity, "#9E9E9E")

        # Build attachment
        attachment = {
            "color": color,
            "title": notification.title,
            "text": notification.message,
            "ts": int(notification.created_at.timestamp()),
        }

        # Add fields
        fields = []
        if notification.severity:
            fields.append(
                {
                    "title": "Severity",
                    "value": notification.severity.upper(),
                    "short": True,
                }
            )
        if notification.resource_type:
            fields.append(
                {
                    "title": "Resource",
                    "value": f"{notification.resource_type}/{notification.resource_id}",
                    "short": True,
                }
            )

        if fields:
            attachment["fields"] = fields

        # Add action button
        if notification.action_url:
            attachment["actions"] = [
                {
                    "type": "button",
                    "text": notification.action_label or "View Details",
                    "url": notification.action_url,
                }
            ]

        return {
            "username": self.config.username,
            "icon_emoji": self.config.icon_emoji,
            "attachments": [attachment],
        }

    async def _send_webhook(self, message: dict, channel: str) -> None:
        """Send via webhook URL."""
        import aiohttp

        from aragora.http_client import WEBHOOK_TIMEOUT

        # Add channel to message
        if channel.startswith("#") or channel.startswith("@"):
            message["channel"] = channel

        async with aiohttp.ClientSession(timeout=WEBHOOK_TIMEOUT) as session:
            async with session.post(
                self.config.webhook_url,
                json=message,
            ) as response:
                if response.status != 200:
                    text = await response.text()
                    raise SlackNotificationError(
                        f"Slack webhook failed: {text}",
                        status_code=response.status,
                    )

    async def _send_api(self, message: dict, channel: str) -> None:
        """Send via Slack API."""
        import aiohttp

        from aragora.http_client import DEFAULT_TIMEOUT

        message["channel"] = channel

        async with aiohttp.ClientSession(timeout=DEFAULT_TIMEOUT) as session:
            async with session.post(
                "https://slack.com/api/chat.postMessage",
                json=message,
                headers={"Authorization": f"Bearer {self.config.bot_token}"},
            ) as response:
                data = await response.json()
                if not data.get("ok"):
                    raise SlackNotificationError(
                        f"Slack API error: {data.get('error')}",
                        error_code=data.get("error"),
                    )


class EmailProvider(NotificationProvider):
    """Email notification provider."""

    def __init__(self, config: EmailConfig):
        self.config = config

    @property
    def channel(self) -> NotificationChannel:
        return NotificationChannel.EMAIL

    def is_configured(self) -> bool:
        return bool(self.config.smtp_host)

    async def send(
        self,
        notification: Notification,
        recipient: str,
    ) -> NotificationResult:
        """Send email notification."""
        start_time = time.perf_counter()

        if not self.is_configured():
            latency = time.perf_counter() - start_time
            _record_notification_metric(
                "email",
                notification.severity,
                notification.priority.value,
                False,
                latency,
                "not_configured",
            )
            return NotificationResult(
                success=False,
                channel=self.channel,
                recipient=recipient,
                notification_id=notification.id,
                error="Email not configured",
            )

        try:
            # Run SMTP in thread pool (it's blocking)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._send_email,
                notification,
                recipient,
            )

            latency = time.perf_counter() - start_time
            _record_notification_metric(
                "email", notification.severity, notification.priority.value, True, latency
            )
            return NotificationResult(
                success=True,
                channel=self.channel,
                recipient=recipient,
                notification_id=notification.id,
            )

        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
            latency = time.perf_counter() - start_time
            error_type = "connection_error" if "connection" in str(e).lower() else "delivery_error"
            _record_notification_metric(
                "email",
                notification.severity,
                notification.priority.value,
                False,
                latency,
                error_type,
            )
            return NotificationResult(
                success=False,
                channel=self.channel,
                recipient=recipient,
                notification_id=notification.id,
                error=str(e),
            )

    def _send_email(self, notification: Notification, recipient: str) -> None:
        """Send email via SMTP."""
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"[{notification.severity.upper()}] {notification.title}"
        msg["From"] = f"{self.config.from_name} <{self.config.from_address}>"
        msg["To"] = recipient

        # Plain text version
        text = f"{notification.title}\n\n{notification.message}"
        if notification.action_url:
            text += f"\n\nView details: {notification.action_url}"

        # HTML version
        html = self._build_html(notification)

        msg.attach(MIMEText(text, "plain"))
        msg.attach(MIMEText(html, "html"))

        # Send
        with smtplib.SMTP(self.config.smtp_host, self.config.smtp_port) as server:
            if self.config.use_tls:
                server.starttls()
            if self.config.smtp_user and self.config.smtp_password:
                server.login(self.config.smtp_user, self.config.smtp_password)
            server.send_message(msg)

    def _build_html(self, notification: Notification) -> str:
        """Build HTML email content."""
        colors = {
            "info": "#2196F3",
            "warning": "#FF9800",
            "error": "#F44336",
            "critical": "#B71C1C",
        }
        color = colors.get(notification.severity, "#9E9E9E")

        action_html = ""
        if notification.action_url:
            action_html = f"""
            <p style="margin-top: 20px;">
                <a href="{notification.action_url}"
                   style="background-color: {color}; color: white; padding: 10px 20px;
                          text-decoration: none; border-radius: 4px;">
                    {notification.action_label or "View Details"}
                </a>
            </p>
            """

        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background-color: {
            color
        }; color: white; padding: 15px; border-radius: 4px 4px 0 0; }}
                .content {{ background-color: #f5f5f5; padding: 20px; border-radius: 0 0 4px 4px; }}
                .meta {{ color: #666; font-size: 0.9em; margin-top: 15px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h2 style="margin: 0;">{notification.title}</h2>
                </div>
                <div class="content">
                    <p>{notification.message}</p>
                    {action_html}
                    <div class="meta">
                        <p>Severity: {notification.severity.upper()}</p>
                        {
            f"<p>Resource: {notification.resource_type}/{notification.resource_id}</p>"
            if notification.resource_type
            else ""
        }
                    </div>
                </div>
            </div>
        </body>
        </html>
        """


class WebhookProvider(NotificationProvider):
    """Webhook notification provider."""

    def __init__(self):
        self.endpoints: dict[str, WebhookEndpoint] = {}

    @property
    def channel(self) -> NotificationChannel:
        return NotificationChannel.WEBHOOK

    def is_configured(self) -> bool:
        return len(self.endpoints) > 0

    def add_endpoint(self, endpoint: WebhookEndpoint) -> None:
        """Register a webhook endpoint."""
        self.endpoints[endpoint.id] = endpoint

    def remove_endpoint(self, endpoint_id: str) -> bool:
        """Remove a webhook endpoint."""
        if endpoint_id in self.endpoints:
            del self.endpoints[endpoint_id]
            return True
        return False

    async def send(
        self,
        notification: Notification,
        recipient: str,  # endpoint_id
    ) -> NotificationResult:
        """Send notification to webhook endpoint."""
        start_time = time.perf_counter()

        endpoint = self.endpoints.get(recipient)
        if not endpoint:
            latency = time.perf_counter() - start_time
            _record_notification_metric(
                "webhook",
                notification.severity,
                notification.priority.value,
                False,
                latency,
                "endpoint_not_found",
            )
            return NotificationResult(
                success=False,
                channel=self.channel,
                recipient=recipient,
                notification_id=notification.id,
                error=f"Webhook endpoint not found: {recipient}",
            )

        if not endpoint.enabled:
            latency = time.perf_counter() - start_time
            _record_notification_metric(
                "webhook",
                notification.severity,
                notification.priority.value,
                False,
                latency,
                "endpoint_disabled",
            )
            return NotificationResult(
                success=False,
                channel=self.channel,
                recipient=recipient,
                notification_id=notification.id,
                error="Webhook endpoint is disabled",
            )

        try:
            import aiohttp

            payload = notification.to_dict()
            body = json.dumps(payload)

            headers = {
                "Content-Type": "application/json",
                **endpoint.headers,
            }

            # Add signature if secret is configured
            if endpoint.secret:
                signature = hmac.new(
                    endpoint.secret.encode(),
                    body.encode(),
                    hashlib.sha256,
                ).hexdigest()
                headers["X-Aragora-Signature"] = f"sha256={signature}"

            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.post(
                    endpoint.url,
                    data=body,
                    headers=headers,
                ) as response:
                    if response.status >= 400:
                        text = await response.text()
                        raise WebhookDeliveryError(
                            webhook_url=endpoint.url,
                            status_code=response.status,
                            message=text,
                        )

            latency = time.perf_counter() - start_time
            _record_notification_metric(
                "webhook", notification.severity, notification.priority.value, True, latency
            )
            return NotificationResult(
                success=True,
                channel=self.channel,
                recipient=recipient,
                notification_id=notification.id,
            )

        except Exception as e:
            logger.error(f"Failed to send webhook notification: {e}")
            latency = time.perf_counter() - start_time
            error_type = "timeout" if "timeout" in str(e).lower() else "delivery_error"
            _record_notification_metric(
                "webhook",
                notification.severity,
                notification.priority.value,
                False,
                latency,
                error_type,
            )
            return NotificationResult(
                success=False,
                channel=self.channel,
                recipient=recipient,
                notification_id=notification.id,
                error=str(e),
            )

    async def send_to_matching(
        self,
        notification: Notification,
        event_type: str,
    ) -> list[NotificationResult]:
        """Send to all endpoints matching the event type."""
        results = []
        for endpoint in self.endpoints.values():
            if endpoint.matches_event(event_type):
                result = await self.send(notification, endpoint.id)
                results.append(result)
        return results


class NotificationService:
    """
    Main notification service orchestrating multiple channels.

    Handles routing notifications to appropriate channels and
    managing provider configurations.
    """

    def __init__(
        self,
        slack_config: Optional[SlackConfig] = None,
        email_config: Optional[EmailConfig] = None,
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

    def get_provider(self, channel: NotificationChannel) -> Optional[NotificationProvider]:
        """Get a provider by channel."""
        return self.providers.get(channel)

    @property
    def webhook_provider(self) -> WebhookProvider:
        """Get the webhook provider for endpoint management."""
        provider = self.providers[NotificationChannel.WEBHOOK]
        assert isinstance(provider, WebhookProvider)
        return provider

    def get_configured_channels(self) -> list[NotificationChannel]:
        """Get list of configured channels."""
        return [channel for channel, provider in self.providers.items() if provider.is_configured()]

    async def notify(
        self,
        notification: Notification,
        channels: Optional[list[NotificationChannel]] = None,
        recipients: Optional[dict[NotificationChannel, list[str]]] = None,
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
            channel_recipients = []
            if recipients and channel in recipients:
                channel_recipients = recipients[channel]
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
        channel: Optional[NotificationChannel] = None,
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


# Convenience notification functions
async def notify_finding_created(
    finding_id: str,
    title: str,
    severity: str,
    workspace_id: str,
    details: Optional[str] = None,
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


# =============================================================================
# Human Checkpoint Notifications
# =============================================================================


async def notify_checkpoint_approval_requested(
    request_id: str,
    workflow_id: str,
    step_id: str,
    title: str,
    description: str,
    workspace_id: Optional[str] = None,
    assignees: Optional[list[str]] = None,
    timeout_seconds: Optional[float] = None,
    action_url: Optional[str] = None,
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
    workspace_id: Optional[str] = None,
    original_timeout_seconds: Optional[float] = None,
    action_url: Optional[str] = None,
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
    responder_id: Optional[str] = None,
    responder_notes: Optional[str] = None,
    workspace_id: Optional[str] = None,
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
    workspace_id: Optional[str] = None,
    owner_email: Optional[str] = None,
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
    workspace_id: Optional[str] = None,
    owner_email: Optional[str] = None,
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
    error_message: Optional[str] = None,
    workspace_id: Optional[str] = None,
    user_email: Optional[str] = None,
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
    workspace_id: Optional[str] = None,
    user_email: Optional[str] = None,
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


# Global instance
_notification_service: Optional[NotificationService] = None
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
    slack_config: Optional[SlackConfig] = None,
    email_config: Optional[EmailConfig] = None,
) -> NotificationService:
    """Initialize the global notification service with custom config."""
    global _notification_service

    with _lock:
        _notification_service = NotificationService(
            slack_config=slack_config,
            email_config=email_config,
        )

    return _notification_service
