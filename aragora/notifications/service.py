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
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from typing import Any, Optional
import uuid

logger = logging.getLogger(__name__)


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
        if not self.is_configured():
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

            return NotificationResult(
                success=True,
                channel=self.channel,
                recipient=recipient,
                notification_id=notification.id,
            )

        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
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

        # Add channel to message
        if channel.startswith("#") or channel.startswith("@"):
            message["channel"] = channel

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.config.webhook_url,
                json=message,
            ) as response:
                if response.status != 200:
                    text = await response.text()
                    raise Exception(f"Slack webhook failed: {response.status} {text}")

    async def _send_api(self, message: dict, channel: str) -> None:
        """Send via Slack API."""
        import aiohttp

        message["channel"] = channel

        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://slack.com/api/chat.postMessage",
                json=message,
                headers={"Authorization": f"Bearer {self.config.bot_token}"},
            ) as response:
                data = await response.json()
                if not data.get("ok"):
                    raise Exception(f"Slack API error: {data.get('error')}")


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
        if not self.is_configured():
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

            return NotificationResult(
                success=True,
                channel=self.channel,
                recipient=recipient,
                notification_id=notification.id,
            )

        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
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
                .header {{ background-color: {color}; color: white; padding: 15px; border-radius: 4px 4px 0 0; }}
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
                        {f"<p>Resource: {notification.resource_type}/{notification.resource_id}</p>"
                         if notification.resource_type else ""}
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
        endpoint = self.endpoints.get(recipient)
        if not endpoint:
            return NotificationResult(
                success=False,
                channel=self.channel,
                recipient=recipient,
                notification_id=notification.id,
                error=f"Webhook endpoint not found: {recipient}",
            )

        if not endpoint.enabled:
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

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    endpoint.url,
                    data=body,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    if response.status >= 400:
                        text = await response.text()
                        raise Exception(f"Webhook failed: {response.status} {text}")

            return NotificationResult(
                success=True,
                channel=self.channel,
                recipient=recipient,
                notification_id=notification.id,
            )

        except Exception as e:
            logger.error(f"Failed to send webhook notification: {e}")
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
        return self.providers[NotificationChannel.WEBHOOK]

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
            f"Audit session completed with {finding_count} findings "
            f"({critical_count} critical)."
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
