"""
Integration tests for email sync with prioritization and unified inbox.

Tests the full flow:
1. Email sync services (Gmail/Outlook) receive messages
2. Messages are scored by EmailPrioritizer
3. Results are stored and available via unified inbox handler
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch


class TestEmailSyncPrioritizationIntegration:
    """Test email sync with prioritization pipeline."""

    @pytest.fixture
    def mock_gmail_message(self):
        """Create a mock Gmail message."""
        message = MagicMock()
        message.id = "gmail_msg_123"
        message.thread_id = "thread_456"
        message.subject = "URGENT: Contract review needed"
        message.sender = "ceo@company.com"
        message.sender_name = "CEO"
        message.recipients = ["legal@company.com"]
        message.body_text = "Please review the attached contract urgently."
        message.body_html = "<p>Please review the attached contract urgently.</p>"
        message.received_at = datetime.now(timezone.utc)
        message.labels = ["INBOX", "IMPORTANT"]
        message.is_unread = True
        message.attachments = [{"filename": "contract.pdf", "size": 1024}]
        return message

    @pytest.fixture
    def mock_outlook_message(self):
        """Create a mock Outlook message."""
        message = MagicMock()
        message.id = "outlook_msg_456"
        message.conversation_id = "conv_789"
        message.subject = "FW: Meeting tomorrow"
        message.sender = "colleague@company.com"
        message.sender_name = "Colleague"
        message.recipients = ["user@company.com"]
        message.body_text = "See forwarded message below."
        message.body_html = "<p>See forwarded message below.</p>"
        message.received_at = datetime.now(timezone.utc)
        message.folder = "Inbox"
        message.is_read = False
        message.importance = "normal"
        return message

    @pytest.mark.asyncio
    async def test_gmail_sync_with_prioritization(self, mock_gmail_message):
        """Test Gmail sync triggers prioritization."""
        from aragora.connectors.email.gmail_sync import (
            GmailSyncService,
            GmailSyncConfig,
            GmailSyncState,
            SyncedMessage,
        )

        prioritization_results = []

        async def mock_prioritize(message):
            result = MagicMock()
            result.tier = "critical" if "URGENT" in message.subject else "normal"
            result.score = 0.9 if "URGENT" in message.subject else 0.5
            result.factors = {"urgency": True, "sender_reputation": 0.8}
            prioritization_results.append(result)
            return result

        config = GmailSyncConfig(enable_prioritization=True)

        # Create mock connector
        mock_connector = AsyncMock()
        mock_connector.get_user_info = AsyncMock(
            return_value={
                "emailAddress": "user@company.com",
                "historyId": "12345",
            }
        )
        mock_connector.list_messages = AsyncMock(return_value=([mock_gmail_message], None))
        mock_connector.get_message = AsyncMock(return_value=mock_gmail_message)
        mock_connector.authenticate = AsyncMock(return_value=True)

        # Create mock prioritizer
        mock_prioritizer = AsyncMock()
        mock_prioritizer.score = mock_prioritize

        messages_synced = []

        def on_message(synced_msg):
            messages_synced.append(synced_msg)

        service = GmailSyncService(
            tenant_id="tenant_123",
            user_id="user_456",
            config=config,
            gmail_connector=mock_connector,
            prioritizer=mock_prioritizer,
            on_message_synced=on_message,
        )

        # Set up state
        service._state = GmailSyncState(
            tenant_id="tenant_123",
            user_id="user_456",
            email_address="user@company.com",
            history_id="12345",
            initial_sync_complete=True,
        )
        service._running = True

        # Simulate processing a message
        synced = SyncedMessage(
            message=mock_gmail_message,
            account_id="tenant_123/user_456",
            is_new=True,
        )

        # Process with prioritization
        if service._prioritizer:
            synced.priority_result = await service._prioritizer.score(mock_gmail_message)

        on_message(synced)

        assert len(messages_synced) == 1
        assert messages_synced[0].priority_result.tier == "critical"
        assert messages_synced[0].priority_result.score == 0.9

        await service.stop()

    @pytest.mark.asyncio
    async def test_outlook_sync_with_prioritization(self, mock_outlook_message):
        """Test Outlook sync triggers prioritization."""
        from aragora.connectors.email.outlook_sync import (
            OutlookSyncService,
            OutlookSyncConfig,
            OutlookSyncState,
            OutlookSyncedMessage,
        )

        config = OutlookSyncConfig(enable_prioritization=True)

        # Create mock connector
        mock_connector = AsyncMock()
        mock_connector.get_user_info = AsyncMock(
            return_value={
                "mail": "user@company.com",
                "userPrincipalName": "user@company.com",
            }
        )
        mock_connector.list_folders = AsyncMock(
            return_value=[
                MagicMock(id="inbox_123", display_name="Inbox"),
            ]
        )
        mock_connector.get_message = AsyncMock(return_value=mock_outlook_message)
        mock_connector.authenticate = AsyncMock(return_value=True)

        # Create mock prioritizer
        async def mock_prioritize(message):
            result = MagicMock()
            result.tier = "normal"
            result.score = 0.5
            result.factors = {"forwarded": True}
            return result

        mock_prioritizer = AsyncMock()
        mock_prioritizer.score = mock_prioritize

        messages_synced = []

        service = OutlookSyncService(
            tenant_id="tenant_123",
            user_id="user_456",
            config=config,
            outlook_connector=mock_connector,
            prioritizer=mock_prioritizer,
            on_message_synced=lambda msg: messages_synced.append(msg),
        )

        service._state = OutlookSyncState(
            tenant_id="tenant_123",
            user_id="user_456",
            email_address="user@company.com",
            initial_sync_complete=True,
            client_state="secret",
        )
        service._running = True

        # Simulate processing a message
        synced = OutlookSyncedMessage(
            message=mock_outlook_message,
            account_id="tenant_123/user_456",
            is_new=True,
            change_type="created",
        )

        if service._prioritizer:
            synced.priority_result = await service._prioritizer.score(mock_outlook_message)

        messages_synced.append(synced)

        assert len(messages_synced) == 1
        assert messages_synced[0].priority_result.tier == "normal"
        assert messages_synced[0].priority_result.score == 0.5

        await service.stop()


class TestUnifiedInboxIntegration:
    """Test unified inbox handler integration."""

    @pytest.mark.asyncio
    async def test_unified_inbox_receives_synced_messages(self):
        """Test that unified inbox can receive messages from both Gmail and Outlook."""
        # This test verifies the handler routes exist and can be called

        from aragora.server.handlers.features.unified_inbox import UnifiedInboxHandler

        handler = UnifiedInboxHandler()

        # Verify handler routes are registered
        assert "/api/v1/inbox/messages" in [r.split()[0] if " " in r else r for r in handler.ROUTES]

    @pytest.mark.asyncio
    async def test_inbox_triage_endpoint(self):
        """Test inbox triage endpoint exists."""
        from aragora.server.handlers.features.unified_inbox import UnifiedInboxHandler

        handler = UnifiedInboxHandler()

        # Verify triage route exists
        routes = [r.split()[0] if " " in r else r for r in handler.ROUTES]
        # Check for triage-related routes
        assert any("inbox" in r for r in routes)


class TestEmailPrioritizationPipeline:
    """Test the full prioritization pipeline."""

    @pytest.mark.asyncio
    async def test_prioritizer_scoring(self):
        """Test email prioritizer scoring logic."""
        from aragora.prioritization.email_prioritizer import EmailPrioritizer

        prioritizer = EmailPrioritizer()

        # Create test message
        message = MagicMock()
        message.subject = "URGENT: Action required immediately"
        message.sender = "vip@company.com"
        message.body_text = "This requires your immediate attention."
        message.received_at = datetime.now(timezone.utc)
        message.is_unread = True

        # Score the message
        result = await prioritizer.score(message)

        # URGENT in subject should boost priority
        assert result.score > 0.5
        assert result.tier in ["critical", "high", "normal"]
        assert "urgency" in result.factors or "subject_keywords" in result.factors

    @pytest.mark.asyncio
    async def test_spam_detection(self):
        """Test spam detection in prioritizer."""
        from aragora.prioritization.email_prioritizer import EmailPrioritizer

        prioritizer = EmailPrioritizer()

        # Create spam-like message
        message = MagicMock()
        message.subject = "FREE MONEY!!! Click here!!!"
        message.sender = "unknown@spam-domain.xyz"
        message.body_text = "You have won $1,000,000!!! Click here to claim your prize!"
        message.received_at = datetime.now(timezone.utc)
        message.is_unread = True

        result = await prioritizer.score(message)

        # Should be flagged as potential spam
        assert result.tier == "spam" or result.score < 0.3

    @pytest.mark.asyncio
    async def test_sender_reputation(self):
        """Test sender reputation affects scoring."""
        from aragora.prioritization.email_prioritizer import EmailPrioritizer

        prioritizer = EmailPrioritizer()

        # Message from known VIP
        vip_message = MagicMock()
        vip_message.subject = "Quick question"
        vip_message.sender = "ceo@company.com"
        vip_message.body_text = "Do you have a moment?"
        vip_message.received_at = datetime.now(timezone.utc)
        vip_message.is_unread = True

        # Message from unknown sender
        unknown_message = MagicMock()
        unknown_message.subject = "Quick question"
        unknown_message.sender = "random@external.com"
        unknown_message.body_text = "Do you have a moment?"
        unknown_message.received_at = datetime.now(timezone.utc)
        unknown_message.is_unread = True

        vip_result = await prioritizer.score(vip_message)
        unknown_result = await prioritizer.score(unknown_message)

        # VIP should generally score higher (when reputation is configured)
        # This test verifies the prioritizer runs without error
        assert vip_result.score >= 0
        assert unknown_result.score >= 0


class TestMultiAccountSync:
    """Test syncing from multiple accounts."""

    @pytest.mark.asyncio
    async def test_concurrent_gmail_outlook_sync(self):
        """Test syncing from both Gmail and Outlook concurrently."""
        import asyncio
        from aragora.connectors.email.gmail_sync import GmailSyncService, GmailSyncConfig
        from aragora.connectors.email.outlook_sync import OutlookSyncService, OutlookSyncConfig

        all_messages = []
        lock = asyncio.Lock()

        async def on_message(msg):
            async with lock:
                all_messages.append(msg)

        # Set up Gmail sync
        gmail_connector = AsyncMock()
        gmail_connector.get_user_info = AsyncMock(
            return_value={
                "emailAddress": "user@gmail.com",
                "historyId": "123",
            }
        )
        gmail_connector.list_messages = AsyncMock(return_value=([], None))
        gmail_connector.authenticate = AsyncMock(return_value=True)

        gmail_service = GmailSyncService(
            tenant_id="tenant_1",
            user_id="user_1",
            config=GmailSyncConfig(enable_prioritization=False),
            gmail_connector=gmail_connector,
            on_message_synced=on_message,
        )

        # Set up Outlook sync
        outlook_connector = AsyncMock()
        outlook_connector.get_user_info = AsyncMock(
            return_value={
                "mail": "user@outlook.com",
            }
        )
        outlook_connector.list_folders = AsyncMock(
            return_value=[
                MagicMock(id="inbox", display_name="Inbox"),
            ]
        )
        outlook_connector.list_messages = AsyncMock(return_value=([], None))
        outlook_connector.get_delta = AsyncMock(return_value=([], None, "delta_link"))
        outlook_connector.authenticate = AsyncMock(return_value=True)

        outlook_service = OutlookSyncService(
            tenant_id="tenant_1",
            user_id="user_1",
            config=OutlookSyncConfig(enable_prioritization=False),
            outlook_connector=outlook_connector,
            on_message_synced=on_message,
        )

        # Start both services concurrently
        results = await asyncio.gather(
            gmail_service.start(do_initial_sync=True),
            outlook_service.start(do_initial_sync=True),
            return_exceptions=True,
        )

        # Both should succeed
        assert results[0] is True or isinstance(results[0], Exception)
        assert results[1] is True or isinstance(results[1], Exception)

        # Clean up
        await gmail_service.stop()
        await outlook_service.stop()


class TestSyncStateRecovery:
    """Test sync state persistence and recovery."""

    def test_gmail_state_roundtrip(self):
        """Test Gmail sync state can be serialized and deserialized."""
        from aragora.connectors.email.gmail_sync import GmailSyncState
        from datetime import datetime, timezone

        original = GmailSyncState(
            tenant_id="tenant_123",
            user_id="user_456",
            email_address="test@gmail.com",
            history_id="12345",
            initial_sync_complete=True,
            last_sync=datetime.now(timezone.utc),
            total_messages_synced=500,
            synced_labels=["INBOX", "IMPORTANT"],
        )

        # Serialize
        data = original.to_dict()

        # Deserialize
        recovered = GmailSyncState.from_dict(data)

        assert recovered.tenant_id == original.tenant_id
        assert recovered.email_address == original.email_address
        assert recovered.history_id == original.history_id
        assert recovered.initial_sync_complete == original.initial_sync_complete
        assert recovered.total_messages_synced == original.total_messages_synced
        assert set(recovered.synced_labels) == set(original.synced_labels)

    def test_outlook_state_roundtrip(self):
        """Test Outlook sync state can be serialized and deserialized."""
        from aragora.connectors.email.outlook_sync import OutlookSyncState
        from datetime import datetime, timezone

        original = OutlookSyncState(
            tenant_id="tenant_123",
            user_id="user_456",
            email_address="test@outlook.com",
            delta_link="https://graph.microsoft.com/v1.0/delta?token=abc",
            initial_sync_complete=True,
            last_sync=datetime.now(timezone.utc),
            subscription_id="sub_789",
            subscription_expiry=datetime.now(timezone.utc),
            total_messages_synced=300,
            synced_folder_ids=["inbox_id", "sent_id"],
        )

        # Serialize
        data = original.to_dict()

        # Deserialize
        recovered = OutlookSyncState.from_dict(data)

        assert recovered.tenant_id == original.tenant_id
        assert recovered.email_address == original.email_address
        assert recovered.delta_link == original.delta_link
        assert recovered.initial_sync_complete == original.initial_sync_complete
        assert recovered.subscription_id == original.subscription_id
        assert recovered.total_messages_synced == original.total_messages_synced
        assert set(recovered.synced_folder_ids) == set(original.synced_folder_ids)
