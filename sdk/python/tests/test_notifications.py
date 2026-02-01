"""Tests for Notifications namespace API."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from aragora.client import AragoraAsyncClient, AragoraClient


class TestNotificationStatus:
    """Tests for notification status and configuration."""

    def test_get_status(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "email": {"configured": True},
                "telegram": {"configured": False},
            }
            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.notifications.get_status()
            mock_request.assert_called_once_with("GET", "/api/v1/notifications/status")
            assert result["email"]["configured"] is True
            assert result["telegram"]["configured"] is False
            client.close()

    def test_configure_email(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"configured": True}
            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.notifications.configure_email(
                smtp_host="smtp.example.com",
                smtp_port=465,
                smtp_username="user@example.com",
                smtp_password="secret",
                from_email="noreply@example.com",
                from_name="My Org",
                use_tls=True,
                notify_on_consensus=True,
                notify_on_debate_end=False,
                notify_on_error=True,
            )
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/notifications/email/config",
                json={
                    "smtp_host": "smtp.example.com",
                    "smtp_port": 465,
                    "smtp_username": "user@example.com",
                    "smtp_password": "secret",
                    "from_email": "noreply@example.com",
                    "from_name": "My Org",
                    "use_tls": True,
                    "notify_on_consensus": True,
                    "notify_on_debate_end": False,
                    "notify_on_error": True,
                },
            )
            assert result["configured"] is True
            client.close()

    def test_configure_telegram(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"configured": True}
            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.notifications.configure_telegram(
                bot_token="123456:ABC-DEF",
                chat_id="-100123456789",
                notify_on_consensus=False,
            )
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/notifications/telegram/config",
                json={
                    "bot_token": "123456:ABC-DEF",
                    "chat_id": "-100123456789",
                    "notify_on_consensus": False,
                    "notify_on_debate_end": True,
                    "notify_on_error": True,
                },
            )
            assert result["configured"] is True
            client.close()


class TestNotificationRecipients:
    """Tests for email recipient management."""

    def test_add_email_recipient(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"id": "rcpt_1", "email": "alice@example.com"}
            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.notifications.add_email_recipient(
                email="alice@example.com",
                name="Alice",
                preferences={"digest": True},
            )
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/notifications/email/recipient",
                json={
                    "email": "alice@example.com",
                    "name": "Alice",
                    "preferences": {"digest": True},
                },
            )
            assert result["email"] == "alice@example.com"
            client.close()

    def test_remove_email_recipient(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"removed": True}
            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.notifications.remove_email_recipient("alice@example.com")
            mock_request.assert_called_once_with(
                "DELETE",
                "/api/v1/notifications/email/recipient",
                params={"email": "alice@example.com"},
            )
            assert result["removed"] is True
            client.close()

    def test_list_email_recipients(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {
                "recipients": [
                    {"email": "alice@example.com"},
                    {"email": "bob@example.com"},
                ]
            }
            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.notifications.list_email_recipients()
            mock_request.assert_called_once_with("GET", "/api/v1/notifications/email/recipients")
            assert len(result) == 2
            assert result[0]["email"] == "alice@example.com"
            client.close()


class TestNotificationSending:
    """Tests for sending notifications."""

    def test_send_test_notification(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"email": "sent", "telegram": "sent"}
            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.notifications.send_test(notification_type="email")
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/notifications/test",
                json={"type": "email"},
            )
            assert result["email"] == "sent"
            client.close()

    def test_send_notification(self) -> None:
        with patch.object(AragoraClient, "request") as mock_request:
            mock_request.return_value = {"email": "sent", "telegram": "sent"}
            client = AragoraClient(base_url="https://api.aragora.ai")
            result = client.notifications.send(
                subject="Consensus Reached",
                message="Debate #42 reached consensus.",
                notification_type="all",
                html_message="<h1>Consensus Reached</h1>",
            )
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/notifications/send",
                json={
                    "subject": "Consensus Reached",
                    "message": "Debate #42 reached consensus.",
                    "type": "all",
                    "html_message": "<h1>Consensus Reached</h1>",
                },
            )
            assert result["email"] == "sent"
            client.close()


class TestAsyncNotifications:
    """Tests for async notification methods."""

    @pytest.mark.asyncio
    async def test_get_status(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {
                "email": {"configured": True},
                "telegram": {"configured": False},
            }
            client = AragoraAsyncClient(base_url="https://api.aragora.ai")
            result = await client.notifications.get_status()
            mock_request.assert_called_once_with("GET", "/api/v1/notifications/status")
            assert result["email"]["configured"] is True
            await client.close()

    @pytest.mark.asyncio
    async def test_send_notification(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"email": "sent"}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai")
            result = await client.notifications.send(
                subject="Test",
                message="Hello",
                notification_type="email",
            )
            mock_request.assert_called_once_with(
                "POST",
                "/api/v1/notifications/send",
                json={
                    "subject": "Test",
                    "message": "Hello",
                    "type": "email",
                    "html_message": None,
                },
            )
            assert result["email"] == "sent"
            await client.close()

    @pytest.mark.asyncio
    async def test_list_email_recipients(self) -> None:
        with patch.object(AragoraAsyncClient, "request") as mock_request:
            mock_request.return_value = {"recipients": [{"email": "alice@example.com"}]}
            client = AragoraAsyncClient(base_url="https://api.aragora.ai")
            result = await client.notifications.list_email_recipients()
            mock_request.assert_called_once_with("GET", "/api/v1/notifications/email/recipients")
            assert len(result) == 1
            await client.close()
