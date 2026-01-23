"""
Tests for Outlook Sync Service.

Tests for background sync, webhook handling, and prioritization integration.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch


class TestOutlookSyncConfig:
    """Tests for OutlookSyncConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        from aragora.connectors.email.outlook_sync import OutlookSyncConfig

        config = OutlookSyncConfig()

        assert config.notification_url == ""
        assert config.initial_sync_days == 7
        assert config.max_messages_per_sync == 100
        assert "Inbox" in config.sync_folders
        assert "Deleted Items" in config.exclude_folders
        assert config.enable_prioritization is True
        assert config.subscription_expiry_minutes == 4230

    def test_custom_config(self):
        """Test custom configuration."""
        from aragora.connectors.email.outlook_sync import OutlookSyncConfig

        config = OutlookSyncConfig(
            notification_url="https://api.example.com/webhooks/outlook",
            initial_sync_days=30,
            max_messages_per_sync=500,
            sync_folders=["Inbox", "Important"],
            enable_prioritization=False,
            subscription_expiry_minutes=2880,
        )

        assert config.notification_url == "https://api.example.com/webhooks/outlook"
        assert config.initial_sync_days == 30
        assert config.max_messages_per_sync == 500
        assert "Important" in config.sync_folders
        assert config.enable_prioritization is False
        assert config.subscription_expiry_minutes == 2880


class TestOutlookSyncState:
    """Tests for OutlookSyncState."""

    def test_state_creation(self):
        """Test state creation."""
        from aragora.connectors.email.outlook_sync import OutlookSyncState

        state = OutlookSyncState(
            tenant_id="tenant_123",
            user_id="user_456",
            email_address="test@example.com",
        )

        assert state.tenant_id == "tenant_123"
        assert state.user_id == "user_456"
        assert state.email_address == "test@example.com"
        assert state.delta_link == ""
        assert state.initial_sync_complete is False

    def test_state_to_dict(self):
        """Test state serialization."""
        from aragora.connectors.email.outlook_sync import OutlookSyncState

        state = OutlookSyncState(
            tenant_id="tenant_123",
            user_id="user_456",
            email_address="test@outlook.com",
            delta_link="https://graph.microsoft.com/v1.0/delta?token=abc",
            initial_sync_complete=True,
            subscription_id="sub_789",
            total_messages_synced=100,
        )

        data = state.to_dict()

        assert data["tenant_id"] == "tenant_123"
        assert data["email_address"] == "test@outlook.com"
        assert "delta?token=abc" in data["delta_link"]
        assert data["initial_sync_complete"] is True
        assert data["subscription_id"] == "sub_789"
        assert data["total_messages_synced"] == 100

    def test_state_from_dict(self):
        """Test state deserialization."""
        from aragora.connectors.email.outlook_sync import OutlookSyncState

        data = {
            "tenant_id": "tenant_123",
            "user_id": "user_456",
            "email_address": "test@outlook.com",
            "delta_link": "https://graph.microsoft.com/v1.0/delta?token=abc",
            "initial_sync_complete": True,
            "last_sync": "2024-01-15T10:30:00+00:00",
            "subscription_id": "sub_789",
            "subscription_expiry": "2024-01-18T10:30:00+00:00",
            "total_messages_synced": 100,
            "synced_folder_ids": ["inbox_id", "important_id"],
        }

        state = OutlookSyncState.from_dict(data)

        assert state.tenant_id == "tenant_123"
        assert state.email_address == "test@outlook.com"
        assert state.initial_sync_complete is True
        assert state.subscription_id == "sub_789"
        assert state.last_sync is not None
        assert state.subscription_expiry is not None
        assert "inbox_id" in state.synced_folder_ids


class TestOutlookWebhookPayload:
    """Tests for webhook payload parsing."""

    def test_parse_graph_notification(self):
        """Test parsing Microsoft Graph change notification."""
        from aragora.connectors.email.outlook_sync import OutlookWebhookPayload

        notification = {
            "subscriptionId": "sub_123",
            "changeType": "created",
            "resource": "Users/user_456/Messages/msg_789",
            "clientState": "secret_state",
            "tenantId": "tenant_abc",
            "resourceData": {
                "@odata.type": "#Microsoft.Graph.Message",
                "id": "msg_789",
            },
        }

        webhook = OutlookWebhookPayload.from_graph(notification)

        assert webhook.subscription_id == "sub_123"
        assert webhook.change_type == "created"
        assert webhook.resource == "Users/user_456/Messages/msg_789"
        assert webhook.client_state == "secret_state"
        assert webhook.tenant_id == "tenant_abc"
        assert webhook.resource_data["id"] == "msg_789"

    def test_parse_updated_notification(self):
        """Test parsing update notification."""
        from aragora.connectors.email.outlook_sync import OutlookWebhookPayload

        notification = {
            "subscriptionId": "sub_123",
            "changeType": "updated",
            "resource": "me/messages/msg_789",
            "clientState": "secret",
            "tenantId": "tenant_abc",
        }

        webhook = OutlookWebhookPayload.from_graph(notification)

        assert webhook.change_type == "updated"
        assert webhook.resource == "me/messages/msg_789"


class TestOutlookSyncStatus:
    """Tests for OutlookSyncStatus enum."""

    def test_status_values(self):
        """Test status enum values."""
        from aragora.connectors.email.outlook_sync import OutlookSyncStatus

        assert OutlookSyncStatus.IDLE.value == "idle"
        assert OutlookSyncStatus.SYNCING.value == "syncing"
        assert OutlookSyncStatus.WATCHING.value == "watching"
        assert OutlookSyncStatus.ERROR.value == "error"
        assert OutlookSyncStatus.STOPPED.value == "stopped"


class TestOutlookSyncService:
    """Tests for OutlookSyncService."""

    def test_service_creation(self):
        """Test service creation."""
        from aragora.connectors.email.outlook_sync import (
            OutlookSyncService,
            OutlookSyncConfig,
            OutlookSyncStatus,
        )

        config = OutlookSyncConfig(
            notification_url="https://api.example.com/webhooks/outlook"
        )

        service = OutlookSyncService(
            tenant_id="tenant_123",
            user_id="user_456",
            config=config,
        )

        assert service.tenant_id == "tenant_123"
        assert service.user_id == "user_456"
        assert service.status == OutlookSyncStatus.IDLE
        assert service.state is None

    def test_service_with_callbacks(self):
        """Test service with callbacks."""
        from aragora.connectors.email.outlook_sync import OutlookSyncService

        messages_received = []

        def on_message(msg):
            messages_received.append(msg)

        service = OutlookSyncService(
            tenant_id="tenant_123",
            user_id="user_456",
            on_message_synced=on_message,
        )

        assert service._on_message_synced is not None

    def test_extract_message_id(self):
        """Test message ID extraction from resource path."""
        from aragora.connectors.email.outlook_sync import OutlookSyncService

        service = OutlookSyncService(
            tenant_id="tenant_123",
            user_id="user_456",
        )

        # Standard format
        msg_id = service._extract_message_id("Users/user_456/Messages/msg_789")
        assert msg_id == "msg_789"

        # me format
        msg_id = service._extract_message_id("me/messages/msg_abc")
        assert msg_id == "msg_abc"

        # Invalid format
        msg_id = service._extract_message_id("invalid/path")
        assert msg_id is None

    @pytest.mark.asyncio
    async def test_get_stats(self):
        """Test getting service stats."""
        from aragora.connectors.email.outlook_sync import (
            OutlookSyncService,
            OutlookSyncState,
        )

        service = OutlookSyncService(
            tenant_id="tenant_123",
            user_id="user_456",
        )

        # Set up state
        service._state = OutlookSyncState(
            tenant_id="tenant_123",
            user_id="user_456",
            email_address="test@outlook.com",
            delta_link="https://graph.microsoft.com/v1.0/delta?token=abc",
            subscription_id="sub_789",
            total_messages_synced=50,
        )

        stats = service.get_stats()

        assert stats["tenant_id"] == "tenant_123"
        assert stats["user_id"] == "user_456"
        assert stats["email_address"] == "test@outlook.com"
        assert stats["subscription_active"] is True
        assert stats["total_messages_synced"] == 50

    @pytest.mark.asyncio
    async def test_handle_validation(self):
        """Test subscription validation handling."""
        from aragora.connectors.email.outlook_sync import OutlookSyncService

        service = OutlookSyncService(
            tenant_id="tenant_123",
            user_id="user_456",
        )

        token = "validation_token_abc123"
        result = await service.handle_validation(token)

        assert result == token


class TestOutlookSyncedMessage:
    """Tests for OutlookSyncedMessage dataclass."""

    def test_synced_message_creation(self):
        """Test OutlookSyncedMessage creation."""
        from aragora.connectors.email.outlook_sync import OutlookSyncedMessage

        # Mock email message
        mock_message = MagicMock()
        mock_message.id = "msg_123"
        mock_message.subject = "Test Subject"

        synced = OutlookSyncedMessage(
            message=mock_message,
            account_id="tenant/user",
            is_new=True,
            change_type="created",
        )

        assert synced.message.id == "msg_123"
        assert synced.account_id == "tenant/user"
        assert synced.is_new is True
        assert synced.change_type == "created"
        assert synced.priority_result is None


class TestOutlookSyncPackageImports:
    """Test package imports."""

    def test_imports_from_package(self):
        """Test imports work from package."""
        from aragora.connectors.email import (
            OutlookSyncService,
            OutlookSyncConfig,
            OutlookSyncState,
            OutlookWebhookPayload,
            OutlookSyncStatus,
            OutlookSyncedMessage,
            start_outlook_sync,
        )

        assert OutlookSyncService is not None
        assert OutlookSyncConfig is not None
        assert OutlookSyncState is not None
        assert OutlookWebhookPayload is not None
        assert OutlookSyncStatus is not None
        assert OutlookSyncedMessage is not None
        assert start_outlook_sync is not None


class TestOutlookSyncIntegration:
    """Integration tests for Outlook sync (mocked)."""

    @pytest.mark.asyncio
    async def test_initial_sync_flow(self):
        """Test initial sync flow with mocked connector."""
        from aragora.connectors.email.outlook_sync import (
            OutlookSyncService,
            OutlookSyncConfig,
            OutlookSyncStatus,
        )

        config = OutlookSyncConfig(
            enable_prioritization=False,
            initial_sync_days=1,
        )

        # Create mock connector
        mock_connector = AsyncMock()
        mock_connector.get_user_info = AsyncMock(return_value={
            "mail": "test@outlook.com",
            "userPrincipalName": "test@outlook.com",
        })
        mock_connector.list_folders = AsyncMock(return_value=[
            MagicMock(id="inbox_123", display_name="Inbox"),
        ])
        mock_connector.list_messages = AsyncMock(return_value=([], None))
        mock_connector.get_delta = AsyncMock(return_value=(
            [],
            None,
            "https://graph.microsoft.com/v1.0/delta?token=new",
        ))
        mock_connector.authenticate = AsyncMock(return_value=True)

        service = OutlookSyncService(
            tenant_id="tenant_123",
            user_id="user_456",
            config=config,
            outlook_connector=mock_connector,
        )

        # Start service (without subscription since no notification_url)
        success = await service.start(do_initial_sync=True)

        assert success is True
        assert service.state is not None
        assert service.state.email_address == "test@outlook.com"
        assert service.state.initial_sync_complete is True
        assert service.status in [OutlookSyncStatus.IDLE, OutlookSyncStatus.WATCHING]

        await service.stop()

    @pytest.mark.asyncio
    async def test_webhook_handling(self):
        """Test webhook handling with mocked connector."""
        from aragora.connectors.email.outlook_sync import (
            OutlookSyncService,
            OutlookSyncConfig,
            OutlookSyncState,
        )

        config = OutlookSyncConfig(enable_prioritization=False)

        # Create mock connector
        mock_connector = AsyncMock()
        mock_message = MagicMock()
        mock_message.id = "msg_789"
        mock_connector.get_message = AsyncMock(return_value=mock_message)

        service = OutlookSyncService(
            tenant_id="tenant_123",
            user_id="user_456",
            config=config,
            outlook_connector=mock_connector,
        )

        # Set up state as if already synced
        service._state = OutlookSyncState(
            tenant_id="tenant_123",
            user_id="user_456",
            email_address="test@outlook.com",
            delta_link="https://graph.microsoft.com/v1.0/delta?token=abc",
            initial_sync_complete=True,
            client_state="secret_state",
        )
        service._running = True

        # Create webhook payload
        payload = {
            "value": [
                {
                    "subscriptionId": "sub_123",
                    "changeType": "created",
                    "resource": "Users/user_456/Messages/msg_789",
                    "clientState": "secret_state",
                    "tenantId": "tenant_abc",
                }
            ]
        }

        # Handle webhook
        messages = await service.handle_webhook(payload)

        # Should have fetched the message
        mock_connector.get_message.assert_called_once_with("msg_789")
        assert len(messages) == 1
        assert messages[0].message.id == "msg_789"

        await service.stop()

    @pytest.mark.asyncio
    async def test_wrong_client_state_ignored(self):
        """Test webhook with wrong client state is ignored."""
        from aragora.connectors.email.outlook_sync import (
            OutlookSyncService,
            OutlookSyncState,
        )

        service = OutlookSyncService(
            tenant_id="tenant_123",
            user_id="user_456",
        )

        service._state = OutlookSyncState(
            tenant_id="tenant_123",
            user_id="user_456",
            email_address="test@outlook.com",
            client_state="correct_state",
        )
        service._running = True

        # Webhook with wrong client state
        payload = {
            "value": [
                {
                    "subscriptionId": "sub_123",
                    "changeType": "created",
                    "resource": "Users/user_456/Messages/msg_789",
                    "clientState": "wrong_state",
                    "tenantId": "tenant_abc",
                }
            ]
        }

        messages = await service.handle_webhook(payload)

        # Should return empty - wrong client state
        assert messages == []
