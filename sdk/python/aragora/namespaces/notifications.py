"""
Notifications Namespace API

Provides endpoints for managing notification preferences and sending notifications.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ..client import AragoraAsyncClient, AragoraClient


class NotificationsAPI:
    """Synchronous Notifications API."""

    def __init__(self, client: AragoraClient):
        self._client = client

    def get_status(self) -> dict[str, Any]:
        """Get notification integration status.

        Returns:
            Status of email and telegram integrations.
        """
        return self._client.request("GET", "/api/v1/notifications/status")

    def configure_email(
        self,
        smtp_host: str,
        smtp_port: int = 587,
        smtp_username: str | None = None,
        smtp_password: str | None = None,
        from_email: str = "debates@aragora.ai",
        from_name: str = "Aragora Debates",
        use_tls: bool = True,
        notify_on_consensus: bool = True,
        notify_on_debate_end: bool = True,
        notify_on_error: bool = True,
    ) -> dict[str, Any]:
        """Configure email notifications.

        Args:
            smtp_host: SMTP server hostname.
            smtp_port: SMTP server port.
            smtp_username: SMTP authentication username.
            smtp_password: SMTP authentication password.
            from_email: Sender email address.
            from_name: Sender display name.
            use_tls: Whether to use TLS.
            notify_on_consensus: Send notification when consensus reached.
            notify_on_debate_end: Send notification when debate ends.
            notify_on_error: Send notification on errors.

        Returns:
            Configuration result.
        """
        return self._client.request(
            "POST",
            "/api/v1/notifications/email/config",
            json={
                "smtp_host": smtp_host,
                "smtp_port": smtp_port,
                "smtp_username": smtp_username,
                "smtp_password": smtp_password,
                "from_email": from_email,
                "from_name": from_name,
                "use_tls": use_tls,
                "notify_on_consensus": notify_on_consensus,
                "notify_on_debate_end": notify_on_debate_end,
                "notify_on_error": notify_on_error,
            },
        )

    def configure_telegram(
        self,
        bot_token: str,
        chat_id: str,
        notify_on_consensus: bool = True,
        notify_on_debate_end: bool = True,
        notify_on_error: bool = True,
    ) -> dict[str, Any]:
        """Configure Telegram notifications.

        Args:
            bot_token: Telegram bot token.
            chat_id: Target chat ID.
            notify_on_consensus: Send notification when consensus reached.
            notify_on_debate_end: Send notification when debate ends.
            notify_on_error: Send notification on errors.

        Returns:
            Configuration result.
        """
        return self._client.request(
            "POST",
            "/api/v1/notifications/telegram/config",
            json={
                "bot_token": bot_token,
                "chat_id": chat_id,
                "notify_on_consensus": notify_on_consensus,
                "notify_on_debate_end": notify_on_debate_end,
                "notify_on_error": notify_on_error,
            },
        )

    def add_email_recipient(
        self,
        email: str,
        name: str | None = None,
        preferences: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Add an email notification recipient.

        Args:
            email: Recipient email address.
            name: Recipient display name.
            preferences: Notification preferences.

        Returns:
            Result with recipient info.
        """
        return self._client.request(
            "POST",
            "/api/v1/notifications/email/recipient",
            json={
                "email": email,
                "name": name,
                "preferences": preferences or {},
            },
        )

    def remove_email_recipient(self, email: str) -> dict[str, Any]:
        """Remove an email notification recipient.

        Args:
            email: Recipient email address to remove.

        Returns:
            Removal result.
        """
        return self._client.request(
            "DELETE",
            "/api/v1/notifications/email/recipient",
            params={"email": email},
        )

    def list_email_recipients(self) -> list[dict[str, Any]]:
        """List configured email recipients.

        Returns:
            List of email recipients.
        """
        response = self._client.request("GET", "/api/v1/notifications/email/recipients")
        return response.get("recipients", [])

    def send_test(
        self,
        notification_type: Literal["all", "email", "telegram"] = "all",
    ) -> dict[str, Any]:
        """Send a test notification.

        Args:
            notification_type: Type of notification to test.

        Returns:
            Test results for each channel.
        """
        return self._client.request(
            "POST",
            "/api/v1/notifications/test",
            json={"type": notification_type},
        )

    def send(
        self,
        subject: str,
        message: str,
        notification_type: Literal["all", "email", "telegram"] = "all",
        html_message: str | None = None,
    ) -> dict[str, Any]:
        """Send a notification.

        Args:
            subject: Notification subject.
            message: Plain text message content.
            notification_type: Type of notification to send.
            html_message: Optional HTML message content.

        Returns:
            Send results for each channel.
        """
        return self._client.request(
            "POST",
            "/api/v1/notifications/send",
            json={
                "subject": subject,
                "message": message,
                "type": notification_type,
                "html_message": html_message,
            },
        )


class AsyncNotificationsAPI:
    """Asynchronous Notifications API."""

    def __init__(self, client: AragoraAsyncClient):
        self._client = client

    async def get_status(self) -> dict[str, Any]:
        """Get notification integration status."""
        return await self._client.request("GET", "/api/v1/notifications/status")

    async def configure_email(
        self,
        smtp_host: str,
        smtp_port: int = 587,
        smtp_username: str | None = None,
        smtp_password: str | None = None,
        from_email: str = "debates@aragora.ai",
        from_name: str = "Aragora Debates",
        use_tls: bool = True,
        notify_on_consensus: bool = True,
        notify_on_debate_end: bool = True,
        notify_on_error: bool = True,
    ) -> dict[str, Any]:
        """Configure email notifications."""
        return await self._client.request(
            "POST",
            "/api/v1/notifications/email/config",
            json={
                "smtp_host": smtp_host,
                "smtp_port": smtp_port,
                "smtp_username": smtp_username,
                "smtp_password": smtp_password,
                "from_email": from_email,
                "from_name": from_name,
                "use_tls": use_tls,
                "notify_on_consensus": notify_on_consensus,
                "notify_on_debate_end": notify_on_debate_end,
                "notify_on_error": notify_on_error,
            },
        )

    async def configure_telegram(
        self,
        bot_token: str,
        chat_id: str,
        notify_on_consensus: bool = True,
        notify_on_debate_end: bool = True,
        notify_on_error: bool = True,
    ) -> dict[str, Any]:
        """Configure Telegram notifications."""
        return await self._client.request(
            "POST",
            "/api/v1/notifications/telegram/config",
            json={
                "bot_token": bot_token,
                "chat_id": chat_id,
                "notify_on_consensus": notify_on_consensus,
                "notify_on_debate_end": notify_on_debate_end,
                "notify_on_error": notify_on_error,
            },
        )

    async def add_email_recipient(
        self,
        email: str,
        name: str | None = None,
        preferences: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Add an email notification recipient."""
        return await self._client.request(
            "POST",
            "/api/v1/notifications/email/recipient",
            json={
                "email": email,
                "name": name,
                "preferences": preferences or {},
            },
        )

    async def remove_email_recipient(self, email: str) -> dict[str, Any]:
        """Remove an email notification recipient."""
        return await self._client.request(
            "DELETE",
            "/api/v1/notifications/email/recipient",
            params={"email": email},
        )

    async def list_email_recipients(self) -> list[dict[str, Any]]:
        """List configured email recipients."""
        response = await self._client.request("GET", "/api/v1/notifications/email/recipients")
        return response.get("recipients", [])

    async def send_test(
        self,
        notification_type: Literal["all", "email", "telegram"] = "all",
    ) -> dict[str, Any]:
        """Send a test notification."""
        return await self._client.request(
            "POST",
            "/api/v1/notifications/test",
            json={"type": notification_type},
        )

    async def send(
        self,
        subject: str,
        message: str,
        notification_type: Literal["all", "email", "telegram"] = "all",
        html_message: str | None = None,
    ) -> dict[str, Any]:
        """Send a notification."""
        return await self._client.request(
            "POST",
            "/api/v1/notifications/send",
            json={
                "subject": subject,
                "message": message,
                "type": notification_type,
                "html_message": html_message,
            },
        )
