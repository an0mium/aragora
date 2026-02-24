"""
Notifications Namespace API

Provides endpoints for managing notification preferences and sending notifications.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class NotificationsAPI:
    """Synchronous Notifications API."""

    def __init__(self, client: AragoraClient):
        self._client = client

    def send(self, **kwargs: Any) -> dict[str, Any]:
        """Send a notification."""
        return self._client.request("POST", "/api/v1/notifications/send", json=kwargs)

    def get_status(self) -> dict[str, Any]:
        """Get notification service status."""
        return self._client.request("GET", "/api/v1/notifications/status")

    def get_history(self) -> dict[str, Any]:
        """Get notification history."""
        return self._client.request("GET", "/api/v1/notifications/history")

    def configure_email(self, **kwargs: Any) -> dict[str, Any]:
        """Configure email notifications."""
        return self._client.request("POST", "/api/v1/notifications/email/config", json=kwargs)

    def list_email_recipients(self) -> dict[str, Any]:
        """List email notification recipients."""
        return self._client.request("GET", "/api/v1/notifications/email/recipients")

    def add_email_recipient(self, **kwargs: Any) -> dict[str, Any]:
        """Add an email notification recipient."""
        return self._client.request("POST", "/api/v1/notifications/email/recipient", json=kwargs)

    def remove_email_recipient(self, **kwargs: Any) -> dict[str, Any]:
        """Remove an email notification recipient."""
        return self._client.request("DELETE", "/api/v1/notifications/email/recipient", json=kwargs)

    def configure_telegram(self, **kwargs: Any) -> dict[str, Any]:
        """Configure Telegram notifications."""
        return self._client.request("POST", "/api/v1/notifications/telegram/config", json=kwargs)

    def test(self, **kwargs: Any) -> dict[str, Any]:
        """Send a test notification."""
        return self._client.request("POST", "/api/v1/notifications/test", json=kwargs)

    def get_delivery_stats(self) -> dict[str, Any]:
        """Get notification delivery statistics (success rate, latency, failures)."""
        return self._client.request("GET", "/api/notifications/delivery-stats")

    def get_preferences(self) -> dict[str, Any]:
        """Get notification preferences for the current user."""
        return self._client.request("GET", "/api/notifications/preferences")

    def update_preferences(self, **kwargs: Any) -> dict[str, Any]:
        """
        Update notification preferences.

        Args:
            **kwargs: Preferences (channels, frequency, quiet_hours, etc.)

        Returns:
            Dict with updated preferences.
        """
        return self._client.request("PUT", "/api/notifications/preferences", json=kwargs)


class AsyncNotificationsAPI:
    """Asynchronous Notifications API."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def send(self, **kwargs: Any) -> dict[str, Any]:
        """Send a notification."""
        return await self._client.request("POST", "/api/v1/notifications/send", json=kwargs)

    async def get_status(self) -> dict[str, Any]:
        """Get notification service status."""
        return await self._client.request("GET", "/api/v1/notifications/status")

    async def get_history(self) -> dict[str, Any]:
        """Get notification history."""
        return await self._client.request("GET", "/api/v1/notifications/history")

    async def configure_email(self, **kwargs: Any) -> dict[str, Any]:
        """Configure email notifications."""
        return await self._client.request("POST", "/api/v1/notifications/email/config", json=kwargs)

    async def list_email_recipients(self) -> dict[str, Any]:
        """List email notification recipients."""
        return await self._client.request("GET", "/api/v1/notifications/email/recipients")

    async def add_email_recipient(self, **kwargs: Any) -> dict[str, Any]:
        """Add an email notification recipient."""
        return await self._client.request(
            "POST", "/api/v1/notifications/email/recipient", json=kwargs
        )

    async def remove_email_recipient(self, **kwargs: Any) -> dict[str, Any]:
        """Remove an email notification recipient."""
        return await self._client.request(
            "DELETE", "/api/v1/notifications/email/recipient", json=kwargs
        )

    async def configure_telegram(self, **kwargs: Any) -> dict[str, Any]:
        """Configure Telegram notifications."""
        return await self._client.request(
            "POST", "/api/v1/notifications/telegram/config", json=kwargs
        )

    async def test(self, **kwargs: Any) -> dict[str, Any]:
        """Send a test notification."""
        return await self._client.request("POST", "/api/v1/notifications/test", json=kwargs)

    async def get_delivery_stats(self) -> dict[str, Any]:
        """Get notification delivery statistics."""
        return await self._client.request("GET", "/api/notifications/delivery-stats")

    async def get_preferences(self) -> dict[str, Any]:
        """Get notification preferences for the current user."""
        return await self._client.request("GET", "/api/notifications/preferences")

    async def update_preferences(self, **kwargs: Any) -> dict[str, Any]:
        """Update notification preferences."""
        return await self._client.request("PUT", "/api/notifications/preferences", json=kwargs)
