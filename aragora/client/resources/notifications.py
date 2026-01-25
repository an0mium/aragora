"""
Notifications API resource for the Aragora client.

Provides methods for notification management:
- Notification preferences
- Notification delivery channels
- Notification history
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from ..client import AragoraClient

logger = logging.getLogger(__name__)


@dataclass
class NotificationChannel:
    """A notification delivery channel."""

    type: str  # email, slack, teams, webhook
    enabled: bool = True
    target: Optional[str] = None  # email address, webhook URL, etc.
    settings: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NotificationPreference:
    """Notification preferences for an event type."""

    event_type: str
    enabled: bool = True
    channels: List[str] = field(default_factory=list)
    frequency: str = "immediate"  # immediate, digest, daily, weekly


@dataclass
class Notification:
    """A notification record."""

    id: str
    type: str
    title: str
    message: str
    status: str  # pending, sent, delivered, failed, read
    priority: str  # low, normal, high, urgent
    created_at: Optional[datetime] = None
    sent_at: Optional[datetime] = None
    read_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NotificationStats:
    """Notification statistics."""

    total_sent: int = 0
    total_delivered: int = 0
    total_failed: int = 0
    total_read: int = 0
    delivery_rate: float = 0.0
    read_rate: float = 0.0


class NotificationsAPI:
    """API interface for notification management."""

    def __init__(self, client: "AragoraClient"):
        self._client = client

    # =========================================================================
    # Preferences
    # =========================================================================

    def get_preferences(self) -> List[NotificationPreference]:
        """
        Get notification preferences for the current user.

        Returns:
            List of NotificationPreference objects.
        """
        response = self._client._get("/api/v1/notifications/preferences")
        prefs = response.get("preferences", [])
        return [self._parse_preference(p) for p in prefs]

    async def get_preferences_async(self) -> List[NotificationPreference]:
        """Async version of get_preferences()."""
        response = await self._client._get_async("/api/v1/notifications/preferences")
        prefs = response.get("preferences", [])
        return [self._parse_preference(p) for p in prefs]

    def update_preference(
        self,
        event_type: str,
        enabled: Optional[bool] = None,
        channels: Optional[List[str]] = None,
        frequency: Optional[str] = None,
    ) -> NotificationPreference:
        """
        Update a notification preference.

        Args:
            event_type: The event type to configure.
            enabled: Whether notifications are enabled.
            channels: Delivery channels to use.
            frequency: Notification frequency.

        Returns:
            Updated NotificationPreference object.
        """
        body: Dict[str, Any] = {"event_type": event_type}
        if enabled is not None:
            body["enabled"] = enabled
        if channels is not None:
            body["channels"] = channels
        if frequency is not None:
            body["frequency"] = frequency

        response = self._client._patch("/api/v1/notifications/preferences", body)
        return self._parse_preference(response.get("preference", response))

    async def update_preference_async(
        self,
        event_type: str,
        enabled: Optional[bool] = None,
        channels: Optional[List[str]] = None,
        frequency: Optional[str] = None,
    ) -> NotificationPreference:
        """Async version of update_preference()."""
        body: Dict[str, Any] = {"event_type": event_type}
        if enabled is not None:
            body["enabled"] = enabled
        if channels is not None:
            body["channels"] = channels
        if frequency is not None:
            body["frequency"] = frequency

        response = await self._client._patch_async("/api/v1/notifications/preferences", body)
        return self._parse_preference(response.get("preference", response))

    def mute_all(self, duration_hours: Optional[int] = None) -> bool:
        """
        Mute all notifications.

        Args:
            duration_hours: Duration to mute (None = indefinite).

        Returns:
            True if successful.
        """
        body: Dict[str, Any] = {"action": "mute"}
        if duration_hours:
            body["duration_hours"] = duration_hours

        self._client._post("/api/v1/notifications/preferences/mute", body)
        return True

    async def mute_all_async(self, duration_hours: Optional[int] = None) -> bool:
        """Async version of mute_all()."""
        body: Dict[str, Any] = {"action": "mute"}
        if duration_hours:
            body["duration_hours"] = duration_hours

        await self._client._post_async("/api/v1/notifications/preferences/mute", body)
        return True

    def unmute_all(self) -> bool:
        """
        Unmute all notifications.

        Returns:
            True if successful.
        """
        self._client._post("/api/v1/notifications/preferences/mute", {"action": "unmute"})
        return True

    async def unmute_all_async(self) -> bool:
        """Async version of unmute_all()."""
        await self._client._post_async(
            "/api/v1/notifications/preferences/mute", {"action": "unmute"}
        )
        return True

    # =========================================================================
    # Channels
    # =========================================================================

    def list_channels(self) -> List[NotificationChannel]:
        """
        List configured notification channels.

        Returns:
            List of NotificationChannel objects.
        """
        response = self._client._get("/api/v1/notifications/channels")
        channels = response.get("channels", [])
        return [self._parse_channel(c) for c in channels]

    async def list_channels_async(self) -> List[NotificationChannel]:
        """Async version of list_channels()."""
        response = await self._client._get_async("/api/v1/notifications/channels")
        channels = response.get("channels", [])
        return [self._parse_channel(c) for c in channels]

    def configure_channel(
        self,
        channel_type: str,
        enabled: bool = True,
        target: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None,
    ) -> NotificationChannel:
        """
        Configure a notification channel.

        Args:
            channel_type: The channel type (email, slack, teams, webhook).
            enabled: Whether the channel is enabled.
            target: Target address/URL.
            settings: Channel-specific settings.

        Returns:
            Configured NotificationChannel object.
        """
        body: Dict[str, Any] = {
            "type": channel_type,
            "enabled": enabled,
        }
        if target:
            body["target"] = target
        if settings:
            body["settings"] = settings

        response = self._client._post("/api/v1/notifications/channels", body)
        return self._parse_channel(response.get("channel", response))

    async def configure_channel_async(
        self,
        channel_type: str,
        enabled: bool = True,
        target: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None,
    ) -> NotificationChannel:
        """Async version of configure_channel()."""
        body: Dict[str, Any] = {
            "type": channel_type,
            "enabled": enabled,
        }
        if target:
            body["target"] = target
        if settings:
            body["settings"] = settings

        response = await self._client._post_async("/api/v1/notifications/channels", body)
        return self._parse_channel(response.get("channel", response))

    def test_channel(self, channel_type: str) -> bool:
        """
        Send a test notification to a channel.

        Args:
            channel_type: The channel type to test.

        Returns:
            True if test was sent successfully.
        """
        body = {"type": channel_type}
        self._client._post("/api/v1/notifications/channels/test", body)
        return True

    async def test_channel_async(self, channel_type: str) -> bool:
        """Async version of test_channel()."""
        body = {"type": channel_type}
        await self._client._post_async("/api/v1/notifications/channels/test", body)
        return True

    # =========================================================================
    # Notification History
    # =========================================================================

    def list(
        self,
        status: Optional[str] = None,
        type_filter: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[List[Notification], int]:
        """
        List notifications.

        Args:
            status: Filter by status.
            type_filter: Filter by type.
            limit: Maximum number of notifications.
            offset: Offset for pagination.

        Returns:
            Tuple of (list of Notification objects, total count).
        """
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status
        if type_filter:
            params["type"] = type_filter

        response = self._client._get("/api/v1/notifications", params=params)
        notifications = [self._parse_notification(n) for n in response.get("notifications", [])]
        return notifications, response.get("total", len(notifications))

    async def list_async(
        self,
        status: Optional[str] = None,
        type_filter: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[List[Notification], int]:
        """Async version of list()."""
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status
        if type_filter:
            params["type"] = type_filter

        response = await self._client._get_async("/api/v1/notifications", params=params)
        notifications = [self._parse_notification(n) for n in response.get("notifications", [])]
        return notifications, response.get("total", len(notifications))

    def mark_as_read(self, notification_id: str) -> bool:
        """
        Mark a notification as read.

        Args:
            notification_id: The notification ID.

        Returns:
            True if successful.
        """
        self._client._post(f"/api/v1/notifications/{notification_id}/read", {})
        return True

    async def mark_as_read_async(self, notification_id: str) -> bool:
        """Async version of mark_as_read()."""
        await self._client._post_async(f"/api/v1/notifications/{notification_id}/read", {})
        return True

    def mark_all_as_read(self) -> int:
        """
        Mark all notifications as read.

        Returns:
            Number of notifications marked as read.
        """
        response = self._client._post("/api/v1/notifications/read-all", {})
        return response.get("count", 0)

    async def mark_all_as_read_async(self) -> int:
        """Async version of mark_all_as_read()."""
        response = await self._client._post_async("/api/v1/notifications/read-all", {})
        return response.get("count", 0)

    def get_stats(self) -> NotificationStats:
        """
        Get notification statistics.

        Returns:
            NotificationStats object.
        """
        response = self._client._get("/api/v1/notifications/stats")
        return self._parse_stats(response)

    async def get_stats_async(self) -> NotificationStats:
        """Async version of get_stats()."""
        response = await self._client._get_async("/api/v1/notifications/stats")
        return self._parse_stats(response)

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _parse_preference(self, data: Dict[str, Any]) -> NotificationPreference:
        """Parse preference data into NotificationPreference object."""
        return NotificationPreference(
            event_type=data.get("event_type", ""),
            enabled=data.get("enabled", True),
            channels=data.get("channels", []),
            frequency=data.get("frequency", "immediate"),
        )

    def _parse_channel(self, data: Dict[str, Any]) -> NotificationChannel:
        """Parse channel data into NotificationChannel object."""
        return NotificationChannel(
            type=data.get("type", ""),
            enabled=data.get("enabled", True),
            target=data.get("target"),
            settings=data.get("settings", {}),
        )

    def _parse_notification(self, data: Dict[str, Any]) -> Notification:
        """Parse notification data into Notification object."""
        created_at = None
        sent_at = None
        read_at = None

        if data.get("created_at"):
            try:
                created_at = datetime.fromisoformat(data["created_at"].replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass

        if data.get("sent_at"):
            try:
                sent_at = datetime.fromisoformat(data["sent_at"].replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass

        if data.get("read_at"):
            try:
                read_at = datetime.fromisoformat(data["read_at"].replace("Z", "+00:00"))
            except (ValueError, TypeError):
                pass

        return Notification(
            id=data.get("id", ""),
            type=data.get("type", ""),
            title=data.get("title", ""),
            message=data.get("message", ""),
            status=data.get("status", "pending"),
            priority=data.get("priority", "normal"),
            created_at=created_at,
            sent_at=sent_at,
            read_at=read_at,
            metadata=data.get("metadata", {}),
        )

    def _parse_stats(self, data: Dict[str, Any]) -> NotificationStats:
        """Parse stats data into NotificationStats object."""
        return NotificationStats(
            total_sent=data.get("total_sent", 0),
            total_delivered=data.get("total_delivered", 0),
            total_failed=data.get("total_failed", 0),
            total_read=data.get("total_read", 0),
            delivery_rate=data.get("delivery_rate", 0.0),
            read_rate=data.get("read_rate", 0.0),
        )


__all__ = [
    "NotificationsAPI",
    "Notification",
    "NotificationPreference",
    "NotificationChannel",
    "NotificationStats",
]
