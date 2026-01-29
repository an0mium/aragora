"""
Budget Alert Notification Service.

Delivers budget alerts to configured Slack/Teams channels using the
channel subscription store. Integrates with BudgetManager's alert callback
system.

Usage:
    from aragora.billing.budget_alert_notifier import (
        BudgetAlertNotifier,
        setup_budget_notifications,
    )

    # Setup notifications for a budget manager
    notifier = BudgetAlertNotifier()
    manager.register_alert_callback(notifier.on_alert)

    # Or use the convenience function
    setup_budget_notifications(manager)
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from aragora.billing.budget_manager import BudgetAlert, BudgetManager

logger = logging.getLogger(__name__)


class NotificationStatus(str, Enum):
    """Status of a notification delivery attempt."""

    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"  # e.g., no subscriptions


@dataclass
class DeliveryResult:
    """Result of a notification delivery attempt."""

    channel_type: str
    channel_id: str
    status: NotificationStatus
    error: str | None = None
    timestamp: float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = datetime.now(timezone.utc).timestamp()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "channel_type": self.channel_type,
            "channel_id": self.channel_id,
            "status": self.status.value,
            "error": self.error,
            "timestamp": self.timestamp,
            "timestamp_iso": datetime.fromtimestamp(self.timestamp, tz=timezone.utc).isoformat(),
        }


class BudgetAlertNotifier:
    """Delivers budget alerts to subscribed channels.

    Integrates with:
    - ChannelSubscriptionStore for delivery targets
    - Slack/Teams connectors for message delivery
    - BudgetManager via alert callback
    """

    def __init__(
        self,
        slack_connector: Any | None = None,
        teams_connector: Any | None = None,
        subscription_store: Any | None = None,
    ):
        """Initialize the notifier.

        Args:
            slack_connector: Optional Slack connector for message delivery.
            teams_connector: Optional Teams connector for message delivery.
            subscription_store: Optional ChannelSubscriptionStore instance.
        """
        self._slack_connector = slack_connector
        self._teams_connector = teams_connector
        self._subscription_store = subscription_store
        self._delivery_history: list[DeliveryResult] = []
        self._max_history = 1000

    @property
    def subscription_store(self):
        """Lazy-load subscription store."""
        if self._subscription_store is None:
            from aragora.storage.channel_subscription_store import (
                get_channel_subscription_store,
            )

            self._subscription_store = get_channel_subscription_store()
        return self._subscription_store

    def on_alert(self, alert: "BudgetAlert") -> None:
        """Handle a budget alert from BudgetManager.

        This method is designed to be registered as a callback with
        BudgetManager.register_alert_callback().

        Args:
            alert: The budget alert to deliver.
        """
        try:
            # Run async delivery in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                results = loop.run_until_complete(self.deliver_alert(alert))
                for result in results:
                    logger.info(
                        f"Budget alert delivery to {result.channel_type}:{result.channel_id}: "
                        f"{result.status.value}"
                    )
            finally:
                loop.close()
        except Exception as e:
            logger.error(f"Failed to deliver budget alert: {e}", exc_info=True)

    async def deliver_alert(self, alert: "BudgetAlert") -> list[DeliveryResult]:
        """Deliver a budget alert to all subscribed channels.

        Args:
            alert: The budget alert to deliver.

        Returns:
            List of delivery results for each channel.
        """
        from aragora.storage.channel_subscription_store import EventType

        # Get subscriptions for budget alerts
        subscriptions = self.subscription_store.get_for_event(alert.org_id, EventType.BUDGET_ALERT)

        if not subscriptions:
            logger.debug(f"No budget alert subscriptions for org {alert.org_id}")
            return [
                DeliveryResult(
                    channel_type="none",
                    channel_id="none",
                    status=NotificationStatus.SKIPPED,
                    error="No subscriptions",
                )
            ]

        results = []
        for sub in subscriptions:
            result = await self._deliver_to_channel(alert, sub)
            results.append(result)
            self._record_delivery(result)

        return results

    async def _deliver_to_channel(
        self,
        alert: "BudgetAlert",
        subscription: Any,
    ) -> DeliveryResult:
        """Deliver alert to a specific channel.

        Args:
            alert: The budget alert.
            subscription: The channel subscription.

        Returns:
            Delivery result.
        """
        from aragora.storage.channel_subscription_store import ChannelType

        channel_type = subscription.channel_type
        if isinstance(channel_type, ChannelType):
            channel_type = channel_type.value

        try:
            message = self._format_alert_message(alert)

            if channel_type == "slack":
                await self._send_to_slack(
                    subscription.workspace_id,
                    subscription.channel_id,
                    message,
                    subscription.config,
                )
            elif channel_type == "teams":
                await self._send_to_teams(
                    subscription.workspace_id,
                    subscription.channel_id,
                    message,
                    subscription.config,
                )
            elif channel_type == "webhook":
                await self._send_to_webhook(
                    subscription.channel_id,  # URL stored as channel_id
                    alert,
                    subscription.config,
                )
            else:
                return DeliveryResult(
                    channel_type=channel_type,
                    channel_id=subscription.channel_id,
                    status=NotificationStatus.SKIPPED,
                    error=f"Unsupported channel type: {channel_type}",
                )

            return DeliveryResult(
                channel_type=channel_type,
                channel_id=subscription.channel_id,
                status=NotificationStatus.SUCCESS,
            )

        except Exception as e:
            logger.error(
                f"Failed to deliver to {channel_type}:{subscription.channel_id}: {e}",
                exc_info=True,
            )
            return DeliveryResult(
                channel_type=channel_type,
                channel_id=subscription.channel_id,
                status=NotificationStatus.FAILED,
                error=str(e),
            )

    def _format_alert_message(self, alert: "BudgetAlert") -> dict[str, Any]:
        """Format alert for display in channels.

        Args:
            alert: The budget alert.

        Returns:
            Formatted message dict with text and blocks.
        """
        usage_pct = (alert.spent_usd / alert.amount_usd * 100) if alert.amount_usd > 0 else 0
        remaining = alert.amount_usd - alert.spent_usd

        # Emoji based on threshold
        emoji = "🟡"  # Warning
        if alert.threshold_percentage >= 1.0:
            emoji = "🔴"  # Exceeded
        elif alert.threshold_percentage >= 0.9:
            emoji = "🟠"  # Critical
        elif alert.threshold_percentage <= 0.5:
            emoji = "🟢"  # Info

        text = (
            f"{emoji} Budget Alert: {alert.message}\n"
            f"Budget: ${alert.spent_usd:.2f} / ${alert.amount_usd:.2f} "
            f"({usage_pct:.1f}% used)\n"
            f"Remaining: ${remaining:.2f}"
        )

        # Slack Block Kit format
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{emoji} Budget Alert",
                    "emoji": True,
                },
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": alert.message,
                },
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Spent:* ${alert.spent_usd:.2f}",
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Budget:* ${alert.amount_usd:.2f}",
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Usage:* {usage_pct:.1f}%",
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Remaining:* ${remaining:.2f}",
                    },
                ],
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"Alert ID: {alert.alert_id} | Budget ID: {alert.budget_id}",
                    },
                ],
            },
        ]

        return {
            "text": text,
            "blocks": blocks,
            "alert_data": alert.to_dict(),
        }

    async def _send_to_slack(
        self,
        workspace_id: str,
        channel_id: str,
        message: dict[str, Any],
        config: dict[str, Any],
    ) -> None:
        """Send message to Slack channel.

        Args:
            workspace_id: Slack workspace ID.
            channel_id: Slack channel ID.
            message: Formatted message.
            config: Channel-specific config.
        """
        if self._slack_connector:
            await self._slack_connector.post_message(
                workspace_id=workspace_id,
                channel=channel_id,
                text=message["text"],
                blocks=message.get("blocks"),
            )
        else:
            # Try to import and use connector
            try:
                from aragora.connectors.chat.slack import SlackConnector
                from aragora.storage.slack_workspace_store import get_slack_workspace_store

                store = get_slack_workspace_store()
                workspace = store.get(workspace_id)
                if workspace and workspace.access_token:
                    connector = SlackConnector(token=workspace.access_token)
                    await connector.post_message(  # type: ignore[attr-defined]
                        channel=channel_id,
                        text=message["text"],
                        blocks=message.get("blocks"),
                    )
                else:
                    raise ValueError(f"No Slack workspace found: {workspace_id}")
            except ImportError:
                logger.warning("Slack connector not available")
                raise

    async def _send_to_teams(
        self,
        tenant_id: str,
        channel_id: str,
        message: dict[str, Any],
        config: dict[str, Any],
    ) -> None:
        """Send message to Teams channel.

        Args:
            tenant_id: Teams tenant ID.
            channel_id: Teams channel ID.
            message: Formatted message.
            config: Channel-specific config.
        """
        if self._teams_connector:
            await self._teams_connector.post_message(
                tenant_id=tenant_id,
                channel_id=channel_id,
                text=message["text"],
            )
        else:
            # Try to import and use connector
            try:
                from aragora.connectors.enterprise.collaboration.teams import TeamsConnector  # type: ignore[attr-defined]
                from aragora.storage.teams_workspace_store import get_teams_workspace_store

                store = get_teams_workspace_store()
                workspace = store.get(tenant_id)
                if workspace and workspace.access_token:
                    connector = TeamsConnector(token=workspace.access_token)
                    await connector.post_message(
                        channel_id=channel_id,
                        content=message["text"],
                    )
                else:
                    raise ValueError(f"No Teams workspace found: {tenant_id}")
            except ImportError:
                logger.warning("Teams connector not available")
                raise

    async def _send_to_webhook(
        self,
        url: str,
        alert: "BudgetAlert",
        config: dict[str, Any],
    ) -> None:
        """Send alert to webhook URL.

        Args:
            url: Webhook URL.
            alert: The budget alert.
            config: Webhook-specific config.
        """
        import aiohttp

        payload = {
            "event_type": "budget_alert",
            "alert": alert.to_dict(),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Add custom headers from config
        headers = {"Content-Type": "application/json"}
        if config.get("headers"):
            headers.update(config["headers"])

        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as resp:
                if resp.status >= 400:
                    raise Exception(f"Webhook returned {resp.status}")

    def _record_delivery(self, result: DeliveryResult) -> None:
        """Record delivery result in history.

        Args:
            result: Delivery result to record.
        """
        self._delivery_history.append(result)
        # Trim history if too large
        if len(self._delivery_history) > self._max_history:
            self._delivery_history = self._delivery_history[-self._max_history :]

    def get_delivery_history(
        self,
        limit: int = 50,
        status: NotificationStatus | None = None,
    ) -> list[DeliveryResult]:
        """Get delivery history.

        Args:
            limit: Maximum results to return.
            status: Filter by status.

        Returns:
            List of delivery results.
        """
        results = self._delivery_history
        if status:
            results = [r for r in results if r.status == status]
        return results[-limit:]


# Global notifier instance
_notifier: BudgetAlertNotifier | None = None


def get_budget_alert_notifier() -> BudgetAlertNotifier:
    """Get or create the global budget alert notifier."""
    global _notifier
    if _notifier is None:
        _notifier = BudgetAlertNotifier()
    return _notifier


def setup_budget_notifications(manager: "BudgetManager") -> BudgetAlertNotifier:
    """Setup budget notifications for a BudgetManager.

    Registers the notifier's on_alert callback with the manager.

    Args:
        manager: BudgetManager instance to configure.

    Returns:
        The configured BudgetAlertNotifier.
    """
    notifier = get_budget_alert_notifier()
    manager.register_alert_callback(notifier.on_alert)
    logger.info("Registered budget alert notifier with BudgetManager")
    return notifier
