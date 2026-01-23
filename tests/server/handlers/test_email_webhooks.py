"""
Tests for Email Webhook Handlers.
"""

import base64
import json
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.server.handlers.features.email_webhooks import (
    EmailWebhooksHandler,
    WebhookProvider,
    WebhookStatus,
    NotificationType,
    WebhookSubscription,
    WebhookNotification,
    process_gmail_notification,
    process_outlook_notification,
    get_email_webhooks_handler,
    handle_email_webhooks,
)


class TestWebhookProvider:
    """Tests for WebhookProvider enum."""

    def test_provider_values(self):
        """Test provider enum values."""
        assert WebhookProvider.GMAIL.value == "gmail"
        assert WebhookProvider.OUTLOOK.value == "outlook"


class TestWebhookStatus:
    """Tests for WebhookStatus enum."""

    def test_status_values(self):
        """Test status enum values."""
        assert WebhookStatus.ACTIVE.value == "active"
        assert WebhookStatus.PENDING.value == "pending"
        assert WebhookStatus.EXPIRED.value == "expired"
        assert WebhookStatus.ERROR.value == "error"


class TestNotificationType:
    """Tests for NotificationType enum."""

    def test_notification_types(self):
        """Test notification type enum values."""
        assert NotificationType.MESSAGE_CREATED.value == "message_created"
        assert NotificationType.MESSAGE_UPDATED.value == "message_updated"
        assert NotificationType.MESSAGE_DELETED.value == "message_deleted"
        assert NotificationType.LABEL_CHANGED.value == "label_changed"
        assert NotificationType.SYNC_REQUESTED.value == "sync_requested"


class TestWebhookSubscription:
    """Tests for WebhookSubscription dataclass."""

    def test_subscription_creation(self):
        """Test subscription creation."""
        subscription = WebhookSubscription(
            id="sub_123",
            tenant_id="tenant_456",
            account_id="acc_789",
            provider=WebhookProvider.GMAIL,
            status=WebhookStatus.ACTIVE,
            created_at=datetime.now(timezone.utc),
            notification_url="https://api.example.com/webhooks/gmail",
            client_state="secret_state",
        )

        assert subscription.id == "sub_123"
        assert subscription.provider == WebhookProvider.GMAIL
        assert subscription.status == WebhookStatus.ACTIVE

    def test_subscription_to_dict(self):
        """Test subscription serialization."""
        now = datetime.now(timezone.utc)
        subscription = WebhookSubscription(
            id="sub_123",
            tenant_id="tenant_456",
            account_id="acc_789",
            provider=WebhookProvider.OUTLOOK,
            status=WebhookStatus.PENDING,
            created_at=now,
            expires_at=now,
            notification_count=10,
            error_count=1,
        )

        data = subscription.to_dict()

        assert data["id"] == "sub_123"
        assert data["provider"] == "outlook"
        assert data["status"] == "pending"
        assert data["notification_count"] == 10


class TestWebhookNotification:
    """Tests for WebhookNotification dataclass."""

    def test_notification_creation(self):
        """Test notification creation."""
        notification = WebhookNotification(
            provider=WebhookProvider.GMAIL,
            notification_type=NotificationType.MESSAGE_CREATED,
            account_id="acc_123",
            resource_id="msg_456",
            tenant_id="tenant_789",
            timestamp=datetime.now(timezone.utc),
            raw_data={"test": "data"},
        )

        assert notification.provider == WebhookProvider.GMAIL
        assert notification.notification_type == NotificationType.MESSAGE_CREATED

    def test_notification_to_dict(self):
        """Test notification serialization."""
        notification = WebhookNotification(
            provider=WebhookProvider.OUTLOOK,
            notification_type=NotificationType.MESSAGE_UPDATED,
            account_id="acc_123",
            resource_id="msg_456",
            tenant_id="tenant_789",
            timestamp=datetime.now(timezone.utc),
            raw_data={},
            metadata={"change_type": "updated"},
        )

        data = notification.to_dict()

        assert data["provider"] == "outlook"
        assert data["notification_type"] == "message_updated"
        assert data["metadata"]["change_type"] == "updated"


class TestProcessGmailNotification:
    """Tests for Gmail notification processing."""

    @pytest.mark.asyncio
    async def test_process_valid_notification(self):
        """Test processing valid Gmail notification."""
        # Create Gmail-style notification with base64-encoded data
        gmail_data = {
            "emailAddress": "test@gmail.com",
            "historyId": "12345"
        }
        encoded_data = base64.b64encode(json.dumps(gmail_data).encode()).decode()

        notification_data = {
            "message": {
                "data": encoded_data,
                "messageId": "msg_123",
                "publishTime": "2024-01-15T10:00:00Z",
            },
            "subscription": "projects/test/subscriptions/gmail-push",
        }

        result = await process_gmail_notification(notification_data, "tenant_123")

        assert result is not None
        assert result.provider == WebhookProvider.GMAIL
        assert result.notification_type == NotificationType.SYNC_REQUESTED
        assert result.metadata["email_address"] == "test@gmail.com"
        assert result.metadata["history_id"] == "12345"

    @pytest.mark.asyncio
    async def test_process_missing_data(self):
        """Test processing notification with missing data."""
        notification_data = {
            "message": {},
            "subscription": "projects/test/subscriptions/gmail-push",
        }

        result = await process_gmail_notification(notification_data, "tenant_123")

        assert result is None

    @pytest.mark.asyncio
    async def test_process_invalid_base64(self):
        """Test processing notification with invalid base64."""
        notification_data = {
            "message": {
                "data": "not-valid-base64!!!",
            },
        }

        result = await process_gmail_notification(notification_data, "tenant_123")

        assert result is None


class TestProcessOutlookNotification:
    """Tests for Outlook notification processing."""

    @pytest.mark.asyncio
    async def test_process_created_notification(self):
        """Test processing Outlook created notification."""
        notification_data = {
            "value": [
                {
                    "subscriptionId": "sub_123",
                    "changeType": "created",
                    "resource": "Users/user_456/Messages/msg_789",
                    "clientState": "secret",
                    "tenantId": "tenant_abc",
                }
            ]
        }

        results = await process_outlook_notification(notification_data, "tenant_123")

        assert len(results) == 1
        assert results[0].provider == WebhookProvider.OUTLOOK
        assert results[0].notification_type == NotificationType.MESSAGE_CREATED

    @pytest.mark.asyncio
    async def test_process_updated_notification(self):
        """Test processing Outlook updated notification."""
        notification_data = {
            "value": [
                {
                    "subscriptionId": "sub_123",
                    "changeType": "updated",
                    "resource": "Users/user_456/Messages/msg_789",
                    "clientState": "secret",
                }
            ]
        }

        results = await process_outlook_notification(notification_data, "tenant_123")

        assert len(results) == 1
        assert results[0].notification_type == NotificationType.MESSAGE_UPDATED

    @pytest.mark.asyncio
    async def test_process_deleted_notification(self):
        """Test processing Outlook deleted notification."""
        notification_data = {
            "value": [
                {
                    "subscriptionId": "sub_123",
                    "changeType": "deleted",
                    "resource": "Users/user_456/Messages/msg_789",
                }
            ]
        }

        results = await process_outlook_notification(notification_data, "tenant_123")

        assert len(results) == 1
        assert results[0].notification_type == NotificationType.MESSAGE_DELETED

    @pytest.mark.asyncio
    async def test_process_multiple_notifications(self):
        """Test processing multiple Outlook notifications."""
        notification_data = {
            "value": [
                {
                    "subscriptionId": "sub_1",
                    "changeType": "created",
                    "resource": "Users/user_1/Messages/msg_1",
                },
                {
                    "subscriptionId": "sub_2",
                    "changeType": "updated",
                    "resource": "Users/user_2/Messages/msg_2",
                },
            ]
        }

        results = await process_outlook_notification(notification_data, "tenant_123")

        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_client_state_verification(self):
        """Test client state verification."""
        notification_data = {
            "value": [
                {
                    "subscriptionId": "sub_123",
                    "changeType": "created",
                    "resource": "Users/user/Messages/msg",
                    "clientState": "wrong_state",
                }
            ]
        }

        # Should skip notifications with wrong client state
        results = await process_outlook_notification(
            notification_data, "tenant_123", client_state="correct_state"
        )

        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_empty_notification(self):
        """Test processing empty notification."""
        notification_data = {"value": []}

        results = await process_outlook_notification(notification_data, "tenant_123")

        assert len(results) == 0


class TestEmailWebhooksHandler:
    """Tests for EmailWebhooksHandler."""

    def test_handler_routes(self):
        """Test handler has expected routes."""
        handler = EmailWebhooksHandler()

        expected_routes = [
            "/api/v1/webhooks/gmail",
            "/api/v1/webhooks/outlook",
            "/api/v1/webhooks/status",
            "/api/v1/webhooks/subscribe",
        ]

        for route in expected_routes:
            assert any(route in r for r in handler.ROUTES), f"Missing route: {route}"

    def test_get_handler_instance(self):
        """Test getting handler instance."""
        handler1 = get_email_webhooks_handler()
        handler2 = get_email_webhooks_handler()

        assert handler1 is handler2

    @pytest.mark.asyncio
    async def test_handle_gmail_webhook(self):
        """Test handling Gmail webhook."""
        handler = EmailWebhooksHandler()

        gmail_data = {"emailAddress": "test@gmail.com", "historyId": "123"}
        encoded_data = base64.b64encode(json.dumps(gmail_data).encode()).decode()

        request = MagicMock()
        request.tenant_id = "test_tenant"
        request.json = AsyncMock(return_value={
            "message": {"data": encoded_data},
        })

        result = await handler.handle(request, "/api/v1/webhooks/gmail", "POST")

        assert result is not None

    @pytest.mark.asyncio
    async def test_handle_outlook_webhook(self):
        """Test handling Outlook webhook."""
        handler = EmailWebhooksHandler()

        request = MagicMock()
        request.tenant_id = "test_tenant"
        request.query = {}
        request.json = AsyncMock(return_value={
            "value": [
                {
                    "subscriptionId": "sub_123",
                    "changeType": "created",
                    "resource": "Users/user/Messages/msg",
                }
            ]
        })

        result = await handler.handle(request, "/api/v1/webhooks/outlook", "POST")

        assert result is not None

    @pytest.mark.asyncio
    async def test_handle_outlook_validation(self):
        """Test handling Outlook validation request."""
        handler = EmailWebhooksHandler()

        request = MagicMock()
        request.tenant_id = "test_tenant"
        request.query = {"validationToken": "test_token_123"}

        result = await handler.handle(
            request, "/api/v1/webhooks/outlook", "POST"
        )

        # Should return the validation token
        assert result is not None
        assert result.body == b"test_token_123"
        assert result.content_type == "text/plain"

    @pytest.mark.asyncio
    async def test_handle_status(self):
        """Test getting webhook status."""
        handler = EmailWebhooksHandler()

        request = MagicMock()
        request.tenant_id = "test_tenant"

        result = await handler.handle(request, "/api/v1/webhooks/status", "GET")

        assert result is not None

    @pytest.mark.asyncio
    async def test_handle_subscribe_invalid_provider(self):
        """Test subscribing with invalid provider."""
        handler = EmailWebhooksHandler()

        request = MagicMock()
        request.tenant_id = "test_tenant"
        request.json = AsyncMock(return_value={"provider": "invalid"})

        result = await handler.handle(request, "/api/v1/webhooks/subscribe", "POST")

        assert result is not None

    @pytest.mark.asyncio
    async def test_handle_subscribe_missing_account(self):
        """Test subscribing without account_id."""
        handler = EmailWebhooksHandler()

        request = MagicMock()
        request.tenant_id = "test_tenant"
        request.json = AsyncMock(return_value={"provider": "gmail"})

        result = await handler.handle(request, "/api/v1/webhooks/subscribe", "POST")

        assert result is not None

    @pytest.mark.asyncio
    async def test_handle_not_found(self):
        """Test handling unknown route."""
        handler = EmailWebhooksHandler()

        request = MagicMock()
        request.tenant_id = "test_tenant"

        result = await handler.handle(request, "/api/v1/webhooks/unknown", "GET")

        assert result is not None


class TestHandleEmailWebhooks:
    """Tests for handle_email_webhooks function."""

    @pytest.mark.asyncio
    async def test_entry_point(self):
        """Test entry point function."""
        request = MagicMock()
        request.tenant_id = "test"

        result = await handle_email_webhooks(
            request, "/api/v1/webhooks/status", "GET"
        )

        assert result is not None


class TestImports:
    """Test that imports work correctly."""

    def test_import_from_package(self):
        """Test imports from features package."""
        from aragora.server.handlers.features import (
            EmailWebhooksHandler,
            handle_email_webhooks,
            get_email_webhooks_handler,
            WebhookProvider,
            WebhookStatus,
            NotificationType,
        )

        assert EmailWebhooksHandler is not None
        assert handle_email_webhooks is not None
        assert WebhookProvider is not None
        assert NotificationType is not None
