"""
Tests for Gmail Sync Service.

Tests for background sync, webhook handling, and prioritization integration.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch


class TestGmailSyncConfig:
    """Tests for GmailSyncConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        from aragora.connectors.email.gmail_sync import GmailSyncConfig

        config = GmailSyncConfig()

        assert config.project_id == ""
        assert config.topic_name == "gmail-notifications"
        assert config.subscription_name == "gmail-sync-sub"
        assert config.initial_sync_days == 7
        assert config.max_messages_per_sync == 100
        assert "INBOX" in config.sync_labels
        assert "SPAM" in config.exclude_labels
        assert config.enable_prioritization is True

    def test_custom_config(self):
        """Test custom configuration."""
        from aragora.connectors.email.gmail_sync import GmailSyncConfig

        config = GmailSyncConfig(
            project_id="my-project",
            topic_name="custom-topic",
            initial_sync_days=30,
            max_messages_per_sync=500,
            sync_labels=["INBOX", "IMPORTANT"],
            enable_prioritization=False,
        )

        assert config.project_id == "my-project"
        assert config.topic_name == "custom-topic"
        assert config.initial_sync_days == 30
        assert config.max_messages_per_sync == 500
        assert "IMPORTANT" in config.sync_labels
        assert config.enable_prioritization is False


class TestGmailSyncState:
    """Tests for GmailSyncState."""

    def test_state_creation(self):
        """Test state creation."""
        from aragora.connectors.email.gmail_sync import GmailSyncState

        state = GmailSyncState(
            tenant_id="tenant_123",
            user_id="user_456",
            email_address="test@example.com",
        )

        assert state.tenant_id == "tenant_123"
        assert state.user_id == "user_456"
        assert state.email_address == "test@example.com"
        assert state.history_id == ""
        assert state.initial_sync_complete is False

    def test_state_to_dict(self):
        """Test state serialization."""
        from aragora.connectors.email.gmail_sync import GmailSyncState

        state = GmailSyncState(
            tenant_id="tenant_123",
            user_id="user_456",
            email_address="test@example.com",
            history_id="12345",
            initial_sync_complete=True,
            total_messages_synced=100,
        )

        data = state.to_dict()

        assert data["tenant_id"] == "tenant_123"
        assert data["email_address"] == "test@example.com"
        assert data["history_id"] == "12345"
        assert data["initial_sync_complete"] is True
        assert data["total_messages_synced"] == 100

    def test_state_from_dict(self):
        """Test state deserialization."""
        from aragora.connectors.email.gmail_sync import GmailSyncState

        data = {
            "tenant_id": "tenant_123",
            "user_id": "user_456",
            "email_address": "test@example.com",
            "history_id": "12345",
            "initial_sync_complete": True,
            "last_sync": "2024-01-15T10:30:00+00:00",
            "total_messages_synced": 100,
            "synced_labels": ["INBOX", "IMPORTANT"],
        }

        state = GmailSyncState.from_dict(data)

        assert state.tenant_id == "tenant_123"
        assert state.email_address == "test@example.com"
        assert state.initial_sync_complete is True
        assert state.total_messages_synced == 100
        assert state.last_sync is not None
        assert "INBOX" in state.synced_labels


class TestGmailWebhookPayload:
    """Tests for webhook payload parsing."""

    def test_parse_pubsub_payload(self):
        """Test parsing Pub/Sub webhook payload."""
        import base64
        import json
        from aragora.connectors.email.gmail_sync import GmailWebhookPayload

        # Create base64 encoded data
        data = {
            "emailAddress": "user@example.com",
            "historyId": "12345",
        }
        data_b64 = base64.urlsafe_b64encode(json.dumps(data).encode()).decode()

        payload = {
            "message": {
                "data": data_b64,
                "messageId": "msg_123",
                "publishTime": "2024-01-15T10:30:00Z",
            },
            "subscription": "projects/my-project/subscriptions/gmail-sync-sub",
        }

        webhook = GmailWebhookPayload.from_pubsub(payload)

        assert webhook.message_id == "msg_123"
        assert webhook.email_address == "user@example.com"
        assert webhook.history_id == "12345"
        assert "my-project" in webhook.subscription

    def test_parse_invalid_payload(self):
        """Test parsing invalid payload."""
        from aragora.connectors.email.gmail_sync import GmailWebhookPayload

        payload = {
            "message": {
                "data": "invalid-base64!!!",
                "messageId": "msg_123",
            },
        }

        webhook = GmailWebhookPayload.from_pubsub(payload)

        # Should handle gracefully
        assert webhook.message_id == "msg_123"
        assert webhook.email_address == ""
        assert webhook.history_id == ""


class TestSyncStatus:
    """Tests for SyncStatus enum."""

    def test_status_values(self):
        """Test status enum values."""
        from aragora.connectors.email.gmail_sync import SyncStatus

        assert SyncStatus.IDLE.value == "idle"
        assert SyncStatus.SYNCING.value == "syncing"
        assert SyncStatus.WATCHING.value == "watching"
        assert SyncStatus.ERROR.value == "error"
        assert SyncStatus.STOPPED.value == "stopped"


class TestGmailSyncService:
    """Tests for GmailSyncService."""

    def test_service_creation(self):
        """Test service creation."""
        from aragora.connectors.email.gmail_sync import (
            GmailSyncService,
            GmailSyncConfig,
            SyncStatus,
        )

        config = GmailSyncConfig(project_id="test-project")

        service = GmailSyncService(
            tenant_id="tenant_123",
            user_id="user_456",
            config=config,
        )

        assert service.tenant_id == "tenant_123"
        assert service.user_id == "user_456"
        assert service.status == SyncStatus.IDLE
        assert service.state is None

    def test_service_with_callbacks(self):
        """Test service with callbacks."""
        from aragora.connectors.email.gmail_sync import GmailSyncService

        messages_received = []

        def on_message(msg):
            messages_received.append(msg)

        service = GmailSyncService(
            tenant_id="tenant_123",
            user_id="user_456",
            on_message_synced=on_message,
        )

        assert service._on_message_synced is not None

    @pytest.mark.asyncio
    async def test_get_stats(self):
        """Test getting service stats."""
        from aragora.connectors.email.gmail_sync import GmailSyncService, GmailSyncState

        service = GmailSyncService(
            tenant_id="tenant_123",
            user_id="user_456",
        )

        # Set up state
        service._state = GmailSyncState(
            tenant_id="tenant_123",
            user_id="user_456",
            email_address="test@example.com",
            history_id="12345",
            total_messages_synced=50,
        )

        stats = service.get_stats()

        assert stats["tenant_id"] == "tenant_123"
        assert stats["user_id"] == "user_456"
        assert stats["email_address"] == "test@example.com"
        assert stats["total_messages_synced"] == 50


class TestSyncedMessage:
    """Tests for SyncedMessage dataclass."""

    def test_synced_message_creation(self):
        """Test SyncedMessage creation."""
        from aragora.connectors.email.gmail_sync import SyncedMessage

        # Mock email message
        mock_message = MagicMock()
        mock_message.id = "msg_123"
        mock_message.subject = "Test Subject"

        synced = SyncedMessage(
            message=mock_message,
            account_id="tenant/user",
            is_new=True,
        )

        assert synced.message.id == "msg_123"
        assert synced.account_id == "tenant/user"
        assert synced.is_new is True
        assert synced.priority_result is None


class TestGmailSyncPackageImports:
    """Test package imports."""

    def test_imports_from_package(self):
        """Test imports work from package."""
        from aragora.connectors.email import (
            GmailSyncService,
            GmailSyncConfig,
            GmailSyncState,
            GmailWebhookPayload,
            SyncStatus,
            start_gmail_sync,
        )

        assert GmailSyncService is not None
        assert GmailSyncConfig is not None
        assert GmailSyncState is not None
        assert GmailWebhookPayload is not None
        assert SyncStatus is not None
        assert start_gmail_sync is not None


class TestGmailSyncIntegration:
    """Integration tests for Gmail sync (mocked)."""

    @pytest.mark.asyncio
    async def test_initial_sync_flow(self):
        """Test initial sync flow with mocked connector."""
        from aragora.connectors.email.gmail_sync import (
            GmailSyncService,
            GmailSyncConfig,
            SyncStatus,
        )

        config = GmailSyncConfig(
            enable_prioritization=False,  # Skip prioritization for this test
            initial_sync_days=1,
        )

        # Create mock connector
        mock_connector = AsyncMock()
        mock_connector.get_user_info = AsyncMock(
            return_value={
                "emailAddress": "test@example.com",
                "historyId": "12345",
            }
        )
        mock_connector.list_messages = AsyncMock(return_value=([], None))
        mock_connector.authenticate = AsyncMock(return_value=True)

        service = GmailSyncService(
            tenant_id="tenant_123",
            user_id="user_456",
            config=config,
            gmail_connector=mock_connector,
        )

        # Start service (without watch since no project_id)
        success = await service.start(do_initial_sync=True)

        assert success is True
        assert service.state is not None
        assert service.state.email_address == "test@example.com"
        assert service.state.initial_sync_complete is True
        assert service.status in [SyncStatus.IDLE, SyncStatus.WATCHING]

        await service.stop()

    @pytest.mark.asyncio
    async def test_webhook_handling(self):
        """Test webhook handling with mocked connector."""
        import base64
        import json
        from aragora.connectors.email.gmail_sync import (
            GmailSyncService,
            GmailSyncConfig,
            GmailSyncState,
        )

        config = GmailSyncConfig(enable_prioritization=False)

        # Create mock connector
        mock_connector = AsyncMock()
        mock_connector.get_history = AsyncMock(return_value=([], None, "12346"))

        service = GmailSyncService(
            tenant_id="tenant_123",
            user_id="user_456",
            config=config,
            gmail_connector=mock_connector,
        )

        # Set up state as if already synced
        service._state = GmailSyncState(
            tenant_id="tenant_123",
            user_id="user_456",
            email_address="test@example.com",
            history_id="12345",
            initial_sync_complete=True,
        )
        service._running = True

        # Create webhook payload
        data = {
            "emailAddress": "test@example.com",
            "historyId": "12346",
        }
        data_b64 = base64.urlsafe_b64encode(json.dumps(data).encode()).decode()

        payload = {
            "message": {
                "data": data_b64,
                "messageId": "msg_123",
            },
            "subscription": "projects/test/subscriptions/test",
        }

        # Handle webhook
        messages = await service.handle_webhook(payload)

        # Should have called get_history
        mock_connector.get_history.assert_called_once()

        await service.stop()

    @pytest.mark.asyncio
    async def test_wrong_email_webhook(self):
        """Test webhook for wrong email is ignored."""
        import base64
        import json
        from aragora.connectors.email.gmail_sync import (
            GmailSyncService,
            GmailSyncState,
        )

        service = GmailSyncService(
            tenant_id="tenant_123",
            user_id="user_456",
        )

        service._state = GmailSyncState(
            tenant_id="tenant_123",
            user_id="user_456",
            email_address="correct@example.com",
        )
        service._running = True

        # Webhook for different email
        data = {
            "emailAddress": "wrong@example.com",
            "historyId": "12346",
        }
        data_b64 = base64.urlsafe_b64encode(json.dumps(data).encode()).decode()

        payload = {
            "message": {"data": data_b64, "messageId": "msg_123"},
            "subscription": "test",
        }

        messages = await service.handle_webhook(payload)

        # Should return empty - wrong email
        assert messages == []
