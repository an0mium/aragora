"""
Channel Notifications for the Aragora Control Plane.

Provides notification routing to external channels:
- Slack integration for team notifications
- Microsoft Teams webhooks
- Email notifications (via SMTP or SendGrid)
- Custom webhook endpoints

Notifications are triggered by control plane events:
- Task completions
- Deliberation consensus
- SLA violations
- Agent failures
- Policy violations

This module integrates with the existing connectors for Slack/Teams.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Types
# =============================================================================


class NotificationChannel(Enum):
    """Supported notification channels."""

    SLACK = "slack"
    TEAMS = "teams"
    EMAIL = "email"
    WEBHOOK = "webhook"


class NotificationPriority(Enum):
    """Notification priority levels."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class NotificationEventType(Enum):
    """Types of events that trigger notifications."""

    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    DELIBERATION_STARTED = "deliberation_started"
    DELIBERATION_CONSENSUS = "deliberation_consensus"
    DELIBERATION_FAILED = "deliberation_failed"
    AGENT_REGISTERED = "agent_registered"
    AGENT_OFFLINE = "agent_offline"
    AGENT_ERROR = "agent_error"
    SLA_WARNING = "sla_warning"
    SLA_VIOLATION = "sla_violation"
    POLICY_VIOLATION = "policy_violation"
    CONNECTOR_SYNC_COMPLETE = "connector_sync_complete"
    CONNECTOR_SYNC_FAILED = "connector_sync_failed"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class NotificationMessage:
    """A notification message to be sent."""

    event_type: NotificationEventType
    title: str
    body: str
    priority: NotificationPriority = NotificationPriority.NORMAL
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    workspace_id: Optional[str] = None
    link_url: Optional[str] = None
    link_text: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_type": self.event_type.value,
            "title": self.title,
            "body": self.body,
            "priority": self.priority.value,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "workspace_id": self.workspace_id,
            "link_url": self.link_url,
            "link_text": self.link_text,
        }


@dataclass
class ChannelConfig:
    """Configuration for a notification channel."""

    channel_type: NotificationChannel
    enabled: bool = True
    workspace_id: Optional[str] = None

    # Slack settings
    slack_webhook_url: Optional[str] = None
    slack_channel: Optional[str] = None
    slack_bot_token: Optional[str] = None

    # Teams settings
    teams_webhook_url: Optional[str] = None

    # Email settings
    email_recipients: List[str] = field(default_factory=list)
    email_from: Optional[str] = None
    smtp_host: Optional[str] = None
    smtp_port: int = 587

    # Webhook settings
    webhook_url: Optional[str] = None
    webhook_headers: Dict[str, str] = field(default_factory=dict)

    # Filtering
    event_types: Optional[List[NotificationEventType]] = None  # None = all events
    min_priority: NotificationPriority = NotificationPriority.LOW


@dataclass
class NotificationResult:
    """Result of sending a notification."""

    success: bool
    channel: NotificationChannel
    message_id: Optional[str] = None
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# =============================================================================
# Channel Providers
# =============================================================================


class ChannelProvider(ABC):
    """Abstract base class for notification channel providers."""

    @abstractmethod
    async def send(self, message: NotificationMessage, config: ChannelConfig) -> NotificationResult:
        """Send a notification through this channel."""
        pass

    @abstractmethod
    def format_message(self, message: NotificationMessage) -> Any:
        """Format message for this channel."""
        pass


class SlackProvider(ChannelProvider):
    """Slack notification provider using webhooks or Bot API."""

    async def send(self, message: NotificationMessage, config: ChannelConfig) -> NotificationResult:
        """Send notification to Slack."""
        try:
            import aiohttp

            payload = self.format_message(message)

            # Use webhook or Bot API
            if config.slack_webhook_url:
                url = config.slack_webhook_url
                headers = {"Content-Type": "application/json"}
            elif config.slack_bot_token and config.slack_channel:
                url = "https://slack.com/api/chat.postMessage"
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {config.slack_bot_token}",
                }
                payload["channel"] = config.slack_channel
            else:
                return NotificationResult(
                    success=False,
                    channel=NotificationChannel.SLACK,
                    error="No Slack webhook URL or bot token configured",
                )

            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return NotificationResult(
                            success=True,
                            channel=NotificationChannel.SLACK,
                            message_id=data.get("ts"),
                        )
                    else:
                        error_text = await resp.text()
                        return NotificationResult(
                            success=False,
                            channel=NotificationChannel.SLACK,
                            error=f"Slack API error: {resp.status} - {error_text}",
                        )

        except ImportError:
            return NotificationResult(
                success=False,
                channel=NotificationChannel.SLACK,
                error="aiohttp not installed",
            )
        except Exception as e:
            return NotificationResult(
                success=False,
                channel=NotificationChannel.SLACK,
                error=str(e),
            )

    def format_message(self, message: NotificationMessage) -> Dict[str, Any]:
        """Format message as Slack Block Kit."""
        # Priority emoji
        priority_emoji = {
            NotificationPriority.LOW: ":small_blue_diamond:",
            NotificationPriority.NORMAL: ":large_blue_diamond:",
            NotificationPriority.HIGH: ":warning:",
            NotificationPriority.URGENT: ":rotating_light:",
        }.get(message.priority, ":large_blue_diamond:")

        blocks: list[dict[str, Any]] = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{priority_emoji} {message.title}",
                    "emoji": True,
                },
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": message.body,
                },
            },
        ]

        # Add link button if provided
        if message.link_url:
            blocks.append(
                {
                    "type": "actions",
                    "elements": [
                        {
                            "type": "button",
                            "text": {
                                "type": "plain_text",
                                "text": message.link_text or "View Details",
                            },
                            "url": message.link_url,
                        }
                    ],
                }
            )

        # Add context footer
        blocks.append(
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"Aragora Control Plane | {message.event_type.value} | {message.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}",
                    }
                ],
            }
        )

        return {"blocks": blocks}


class TeamsProvider(ChannelProvider):
    """Microsoft Teams notification provider using webhooks."""

    async def send(self, message: NotificationMessage, config: ChannelConfig) -> NotificationResult:
        """Send notification to Microsoft Teams."""
        try:
            import aiohttp

            if not config.teams_webhook_url:
                return NotificationResult(
                    success=False,
                    channel=NotificationChannel.TEAMS,
                    error="No Teams webhook URL configured",
                )

            payload = self.format_message(message)

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    config.teams_webhook_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                ) as resp:
                    if resp.status == 200:
                        return NotificationResult(
                            success=True,
                            channel=NotificationChannel.TEAMS,
                        )
                    else:
                        error_text = await resp.text()
                        return NotificationResult(
                            success=False,
                            channel=NotificationChannel.TEAMS,
                            error=f"Teams webhook error: {resp.status} - {error_text}",
                        )

        except ImportError:
            return NotificationResult(
                success=False,
                channel=NotificationChannel.TEAMS,
                error="aiohttp not installed",
            )
        except Exception as e:
            return NotificationResult(
                success=False,
                channel=NotificationChannel.TEAMS,
                error=str(e),
            )

    def format_message(self, message: NotificationMessage) -> Dict[str, Any]:
        """Format message as Teams Adaptive Card."""
        # Priority color
        priority_color = {
            NotificationPriority.LOW: "default",
            NotificationPriority.NORMAL: "accent",
            NotificationPriority.HIGH: "warning",
            NotificationPriority.URGENT: "attention",
        }.get(message.priority, "accent")

        card = {
            "type": "message",
            "attachments": [
                {
                    "contentType": "application/vnd.microsoft.card.adaptive",
                    "content": {
                        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                        "type": "AdaptiveCard",
                        "version": "1.4",
                        "body": [
                            {
                                "type": "TextBlock",
                                "text": message.title,
                                "weight": "bolder",
                                "size": "large",
                                "color": priority_color,
                            },
                            {
                                "type": "TextBlock",
                                "text": message.body,
                                "wrap": True,
                            },
                            {
                                "type": "TextBlock",
                                "text": f"Event: {message.event_type.value} | {message.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}",
                                "size": "small",
                                "isSubtle": True,
                            },
                        ],
                    },
                }
            ],
        }

        # Add action button if link provided
        if message.link_url:
            card["attachments"][0]["content"]["actions"] = [
                {
                    "type": "Action.OpenUrl",
                    "title": message.link_text or "View Details",
                    "url": message.link_url,
                }
            ]

        return card


class WebhookProvider(ChannelProvider):
    """Generic webhook notification provider."""

    async def send(self, message: NotificationMessage, config: ChannelConfig) -> NotificationResult:
        """Send notification to a webhook endpoint."""
        try:
            import aiohttp

            if not config.webhook_url:
                return NotificationResult(
                    success=False,
                    channel=NotificationChannel.WEBHOOK,
                    error="No webhook URL configured",
                )

            payload = self.format_message(message)
            headers = {"Content-Type": "application/json", **config.webhook_headers}

            async with aiohttp.ClientSession() as session:
                async with session.post(config.webhook_url, json=payload, headers=headers) as resp:
                    if resp.status < 300:
                        return NotificationResult(
                            success=True,
                            channel=NotificationChannel.WEBHOOK,
                        )
                    else:
                        error_text = await resp.text()
                        return NotificationResult(
                            success=False,
                            channel=NotificationChannel.WEBHOOK,
                            error=f"Webhook error: {resp.status} - {error_text}",
                        )

        except ImportError:
            return NotificationResult(
                success=False,
                channel=NotificationChannel.WEBHOOK,
                error="aiohttp not installed",
            )
        except Exception as e:
            return NotificationResult(
                success=False,
                channel=NotificationChannel.WEBHOOK,
                error=str(e),
            )

    def format_message(self, message: NotificationMessage) -> Dict[str, Any]:
        """Format message as generic JSON payload."""
        return message.to_dict()


# =============================================================================
# Notification Manager
# =============================================================================


class NotificationManager:
    """
    Manages notification routing to configured channels.

    Usage:
        manager = NotificationManager()

        # Configure channels
        manager.add_channel(ChannelConfig(
            channel_type=NotificationChannel.SLACK,
            slack_webhook_url="https://hooks.slack.com/...",
            event_types=[NotificationEventType.TASK_COMPLETED],
        ))

        # Send notification
        await manager.notify(
            event_type=NotificationEventType.TASK_COMPLETED,
            title="Task Completed",
            body="Task xyz finished successfully",
        )
    """

    def __init__(self) -> None:
        self._channels: List[ChannelConfig] = []
        self._providers: Dict[NotificationChannel, ChannelProvider] = {
            NotificationChannel.SLACK: SlackProvider(),
            NotificationChannel.TEAMS: TeamsProvider(),
            NotificationChannel.WEBHOOK: WebhookProvider(),
        }
        self._event_handlers: Dict[NotificationEventType, List[Callable[..., None]]] = {}
        self._notification_history: List[NotificationResult] = []
        self._max_history = 1000

    def add_channel(self, config: ChannelConfig) -> None:
        """Add a notification channel configuration."""
        self._channels.append(config)
        logger.info(
            f"Added notification channel: {config.channel_type.value}",
            extra={"workspace_id": config.workspace_id},
        )

    def remove_channel(
        self, channel_type: NotificationChannel, workspace_id: Optional[str] = None
    ) -> bool:
        """Remove a notification channel."""
        initial_count = len(self._channels)
        self._channels = [
            c
            for c in self._channels
            if not (c.channel_type == channel_type and c.workspace_id == workspace_id)
        ]
        return len(self._channels) < initial_count

    def get_channels(self, workspace_id: Optional[str] = None) -> List[ChannelConfig]:
        """Get configured channels, optionally filtered by workspace."""
        if workspace_id:
            return [
                c
                for c in self._channels
                if c.workspace_id == workspace_id or c.workspace_id is None
            ]
        return list(self._channels)

    async def notify(
        self,
        event_type: NotificationEventType,
        title: str,
        body: str,
        priority: NotificationPriority = NotificationPriority.NORMAL,
        metadata: Optional[Dict[str, Any]] = None,
        workspace_id: Optional[str] = None,
        link_url: Optional[str] = None,
        link_text: Optional[str] = None,
    ) -> List[NotificationResult]:
        """
        Send a notification to all applicable channels.

        Args:
            event_type: Type of event triggering notification
            title: Notification title
            body: Notification body text
            priority: Notification priority
            metadata: Additional metadata
            workspace_id: Workspace for filtering channels
            link_url: Optional URL for action button
            link_text: Text for action button

        Returns:
            List of NotificationResult for each channel attempted
        """
        message = NotificationMessage(
            event_type=event_type,
            title=title,
            body=body,
            priority=priority,
            metadata=metadata or {},
            workspace_id=workspace_id,
            link_url=link_url,
            link_text=link_text,
        )

        # Find applicable channels
        applicable_channels = self._filter_channels(message)

        if not applicable_channels:
            logger.debug(f"No channels configured for event {event_type.value}")
            return []

        # Send to all applicable channels in parallel
        tasks = [self._send_to_channel(message, config) for config in applicable_channels]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        notification_results: List[NotificationResult] = []
        for result in results:
            if isinstance(result, NotificationResult):
                notification_results.append(result)
                self._add_to_history(result)
            elif isinstance(result, Exception):
                error_result = NotificationResult(
                    success=False,
                    channel=NotificationChannel.WEBHOOK,  # Generic
                    error=str(result),
                )
                notification_results.append(error_result)
                self._add_to_history(error_result)

        return notification_results

    def _filter_channels(self, message: NotificationMessage) -> List[ChannelConfig]:
        """Filter channels based on message properties."""
        applicable = []

        for config in self._channels:
            # Check if enabled
            if not config.enabled:
                continue

            # Check workspace filter
            if config.workspace_id and message.workspace_id != config.workspace_id:
                continue

            # Check event type filter
            if config.event_types and message.event_type not in config.event_types:
                continue

            # Check priority filter
            priority_order = list(NotificationPriority)
            if priority_order.index(message.priority) < priority_order.index(config.min_priority):
                continue

            applicable.append(config)

        return applicable

    async def _send_to_channel(
        self, message: NotificationMessage, config: ChannelConfig
    ) -> NotificationResult:
        """Send message to a specific channel."""
        provider = self._providers.get(config.channel_type)

        if not provider:
            return NotificationResult(
                success=False,
                channel=config.channel_type,
                error=f"No provider for channel type: {config.channel_type.value}",
            )

        try:
            result = await provider.send(message, config)
            if result.success:
                logger.info(
                    f"Notification sent to {config.channel_type.value}",
                    extra={"event_type": message.event_type.value},
                )
            else:
                logger.warning(
                    f"Failed to send notification to {config.channel_type.value}: {result.error}",
                )
            return result
        except Exception as e:
            logger.error(f"Error sending notification to {config.channel_type.value}: {e}")
            return NotificationResult(
                success=False,
                channel=config.channel_type,
                error=str(e),
            )

    def _add_to_history(self, result: NotificationResult) -> None:
        """Add result to history with size limit."""
        self._notification_history.append(result)
        if len(self._notification_history) > self._max_history:
            self._notification_history = self._notification_history[-self._max_history :]

    def get_stats(self) -> Dict[str, Any]:
        """Get notification statistics."""
        total = len(self._notification_history)
        successful = sum(1 for r in self._notification_history if r.success)
        by_channel: Dict[str, int] = {}

        for result in self._notification_history:
            channel = result.channel.value
            by_channel[channel] = by_channel.get(channel, 0) + 1

        return {
            "total_sent": total,
            "successful": successful,
            "failed": total - successful,
            "success_rate": successful / total if total > 0 else 0,
            "by_channel": by_channel,
            "channels_configured": len(self._channels),
        }


# =============================================================================
# Helper Functions
# =============================================================================


def create_task_completed_notification(
    task_id: str,
    task_type: str,
    agent_id: str,
    duration_seconds: float,
    ui_base_url: str = "https://app.aragora.ai",
) -> NotificationMessage:
    """Create a notification for task completion."""
    return NotificationMessage(
        event_type=NotificationEventType.TASK_COMPLETED,
        title=f"Task Completed: {task_type}",
        body=f"Task `{task_id[:8]}...` completed by agent `{agent_id}` in {duration_seconds:.1f}s",
        priority=NotificationPriority.NORMAL,
        metadata={"task_id": task_id, "task_type": task_type, "agent_id": agent_id},
        link_url=f"{ui_base_url}/control-plane?task={task_id}",
        link_text="View Task",
    )


def create_deliberation_consensus_notification(
    task_id: str,
    question: str,
    answer: str,
    confidence: float,
    ui_base_url: str = "https://app.aragora.ai",
) -> NotificationMessage:
    """Create a notification for deliberation consensus."""
    return NotificationMessage(
        event_type=NotificationEventType.DELIBERATION_CONSENSUS,
        title="Deliberation Consensus Reached",
        body=f"**Question:** {question[:100]}...\n\n**Answer:** {answer[:200]}...\n\n**Confidence:** {confidence:.0%}",
        priority=NotificationPriority.NORMAL,
        metadata={"task_id": task_id, "confidence": confidence},
        link_url=f"{ui_base_url}/deliberations/{task_id}",
        link_text="View Deliberation",
    )


def create_sla_violation_notification(
    task_id: str,
    task_type: str,
    elapsed_seconds: float,
    timeout_seconds: float,
) -> NotificationMessage:
    """Create a notification for SLA violation."""
    return NotificationMessage(
        event_type=NotificationEventType.SLA_VIOLATION,
        title=f"SLA Violation: {task_type}",
        body=f"Task `{task_id[:8]}...` exceeded SLA timeout.\n\n**Elapsed:** {elapsed_seconds:.0f}s\n**Timeout:** {timeout_seconds:.0f}s",
        priority=NotificationPriority.HIGH,
        metadata={"task_id": task_id, "elapsed_seconds": elapsed_seconds},
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Enums
    "NotificationChannel",
    "NotificationPriority",
    "NotificationEventType",
    # Data Classes
    "NotificationMessage",
    "ChannelConfig",
    "NotificationResult",
    # Providers
    "ChannelProvider",
    "SlackProvider",
    "TeamsProvider",
    "WebhookProvider",
    # Manager
    "NotificationManager",
    # Helpers
    "create_task_completed_notification",
    "create_deliberation_consensus_notification",
    "create_sla_violation_notification",
]
