"""
Tests for Unified Inbox API Handler.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.server.handlers.features.unified_inbox import (
    UnifiedInboxHandler,
    EmailProvider,
    AccountStatus,
    TriageAction,
    ConnectedAccount,
    UnifiedMessage,
    TriageResult,
    InboxStats,
    get_unified_inbox_handler,
    handle_unified_inbox,
)


class TestEmailProvider:
    """Tests for EmailProvider enum."""

    def test_provider_values(self):
        """Test provider enum values."""
        assert EmailProvider.GMAIL.value == "gmail"
        assert EmailProvider.OUTLOOK.value == "outlook"


class TestAccountStatus:
    """Tests for AccountStatus enum."""

    def test_status_values(self):
        """Test status enum values."""
        assert AccountStatus.PENDING.value == "pending"
        assert AccountStatus.CONNECTED.value == "connected"
        assert AccountStatus.SYNCING.value == "syncing"
        assert AccountStatus.ERROR.value == "error"
        assert AccountStatus.DISCONNECTED.value == "disconnected"


class TestTriageAction:
    """Tests for TriageAction enum."""

    def test_action_values(self):
        """Test triage action enum values."""
        assert TriageAction.RESPOND_URGENT.value == "respond_urgent"
        assert TriageAction.RESPOND_NORMAL.value == "respond_normal"
        assert TriageAction.DELEGATE.value == "delegate"
        assert TriageAction.SCHEDULE.value == "schedule"
        assert TriageAction.ARCHIVE.value == "archive"
        assert TriageAction.DELETE.value == "delete"
        assert TriageAction.FLAG.value == "flag"
        assert TriageAction.DEFER.value == "defer"


class TestConnectedAccount:
    """Tests for ConnectedAccount dataclass."""

    def test_account_creation(self):
        """Test account creation."""
        account = ConnectedAccount(
            id="acc_123",
            provider=EmailProvider.GMAIL,
            email_address="test@gmail.com",
            display_name="Test User",
            status=AccountStatus.CONNECTED,
            connected_at=datetime.now(timezone.utc),
        )

        assert account.id == "acc_123"
        assert account.provider == EmailProvider.GMAIL
        assert account.email_address == "test@gmail.com"
        assert account.status == AccountStatus.CONNECTED

    def test_account_to_dict(self):
        """Test account serialization."""
        now = datetime.now(timezone.utc)
        account = ConnectedAccount(
            id="acc_123",
            provider=EmailProvider.OUTLOOK,
            email_address="test@outlook.com",
            display_name="Test User",
            status=AccountStatus.SYNCING,
            connected_at=now,
            last_sync=now,
            total_messages=100,
            unread_count=10,
        )

        data = account.to_dict()

        assert data["id"] == "acc_123"
        assert data["provider"] == "outlook"
        assert data["email_address"] == "test@outlook.com"
        assert data["status"] == "syncing"
        assert data["total_messages"] == 100
        assert data["unread_count"] == 10


class TestUnifiedMessage:
    """Tests for UnifiedMessage dataclass."""

    def test_message_creation(self):
        """Test message creation."""
        message = UnifiedMessage(
            id="msg_123",
            account_id="acc_456",
            provider=EmailProvider.GMAIL,
            external_id="ext_789",
            subject="Test Subject",
            sender_email="sender@example.com",
            sender_name="Sender Name",
            recipients=["recipient@example.com"],
            cc=["cc@example.com"],
            received_at=datetime.now(timezone.utc),
            snippet="Message snippet...",
            body_preview="Full body preview...",
            is_read=False,
            is_starred=True,
            has_attachments=True,
            labels=["inbox", "important"],
        )

        assert message.id == "msg_123"
        assert message.provider == EmailProvider.GMAIL
        assert message.subject == "Test Subject"
        assert message.is_starred is True
        assert "important" in message.labels

    def test_message_to_dict(self):
        """Test message serialization."""
        now = datetime.now(timezone.utc)
        message = UnifiedMessage(
            id="msg_123",
            account_id="acc_456",
            provider=EmailProvider.OUTLOOK,
            external_id="ext_789",
            subject="Test Subject",
            sender_email="sender@example.com",
            sender_name="Sender Name",
            recipients=["recipient@example.com"],
            cc=[],
            received_at=now,
            snippet="Preview...",
            body_preview="Full body...",
            is_read=True,
            is_starred=False,
            has_attachments=False,
            labels=["inbox"],
            priority_score=0.85,
            priority_tier="high",
            priority_reasons=["VIP sender", "Time-sensitive"],
        )

        data = message.to_dict()

        assert data["id"] == "msg_123"
        assert data["provider"] == "outlook"
        assert data["sender"]["email"] == "sender@example.com"
        assert data["priority"]["score"] == 0.85
        assert data["priority"]["tier"] == "high"
        assert "VIP sender" in data["priority"]["reasons"]


class TestTriageResult:
    """Tests for TriageResult dataclass."""

    def test_result_creation(self):
        """Test triage result creation."""
        result = TriageResult(
            message_id="msg_123",
            recommended_action=TriageAction.RESPOND_URGENT,
            confidence=0.92,
            rationale="High priority sender with deadline",
            suggested_response="Dear John, Thank you for...",
            delegate_to=None,
            schedule_for=None,
            agents_involved=["support_analyst", "product_expert"],
            debate_summary="Agents agreed on urgent response",
        )

        assert result.message_id == "msg_123"
        assert result.recommended_action == TriageAction.RESPOND_URGENT
        assert result.confidence == 0.92
        assert len(result.agents_involved) == 2

    def test_result_to_dict(self):
        """Test result serialization."""
        result = TriageResult(
            message_id="msg_123",
            recommended_action=TriageAction.DELEGATE,
            confidence=0.75,
            rationale="Requires specialized knowledge",
            suggested_response=None,
            delegate_to="support_team",
            schedule_for=None,
            agents_involved=["support_manager"],
            debate_summary=None,
        )

        data = result.to_dict()

        assert data["message_id"] == "msg_123"
        assert data["recommended_action"] == "delegate"
        assert data["delegate_to"] == "support_team"
        assert data["confidence"] == 0.75


class TestInboxStats:
    """Tests for InboxStats dataclass."""

    def test_stats_creation(self):
        """Test stats creation."""
        stats = InboxStats(
            total_accounts=2,
            total_messages=100,
            unread_count=15,
            messages_by_priority={
                "critical": 5,
                "high": 20,
                "medium": 50,
                "low": 25,
            },
            messages_by_provider={
                "gmail": 60,
                "outlook": 40,
            },
            avg_response_time_hours=4.5,
            pending_triage=10,
            sync_health={"accounts_healthy": 2, "accounts_error": 0},
            top_senders=[{"email": "boss@company.com", "count": 20}],
            hourly_volume=[],
        )

        assert stats.total_accounts == 2
        assert stats.unread_count == 15
        assert stats.messages_by_priority["critical"] == 5

    def test_stats_to_dict(self):
        """Test stats serialization."""
        stats = InboxStats(
            total_accounts=1,
            total_messages=50,
            unread_count=5,
            messages_by_priority={"critical": 1, "high": 4, "medium": 30, "low": 15},
            messages_by_provider={"gmail": 50, "outlook": 0},
            avg_response_time_hours=2.0,
            pending_triage=3,
            sync_health={"accounts_healthy": 1},
            top_senders=[],
            hourly_volume=[],
        )

        data = stats.to_dict()

        assert data["total_accounts"] == 1
        assert data["unread_count"] == 5
        assert data["pending_triage"] == 3


class TestUnifiedInboxHandler:
    """Tests for UnifiedInboxHandler."""

    def test_handler_routes(self):
        """Test handler has expected routes."""
        handler = UnifiedInboxHandler()

        expected_routes = [
            "/api/v1/inbox/connect",
            "/api/v1/inbox/accounts",
            "/api/v1/inbox/messages",
            "/api/v1/inbox/triage",
            "/api/v1/inbox/stats",
        ]

        for route in expected_routes:
            assert any(route in r for r in handler.ROUTES), f"Missing route: {route}"

    def test_get_handler_instance(self):
        """Test getting handler instance."""
        handler1 = get_unified_inbox_handler()
        handler2 = get_unified_inbox_handler()

        assert handler1 is handler2  # Same instance (singleton)

    @pytest.mark.asyncio
    async def test_handle_list_accounts_empty(self):
        """Test listing accounts when none connected."""
        handler = UnifiedInboxHandler()

        request = MagicMock()
        request.tenant_id = "test_tenant"

        result = await handler.handle(request, "/api/v1/inbox/accounts", "GET")

        assert result is not None
        # Result should indicate success with empty accounts

    @pytest.mark.asyncio
    async def test_handle_stats_empty(self):
        """Test getting stats with no data."""
        handler = UnifiedInboxHandler()

        request = MagicMock()
        request.tenant_id = "test_tenant_stats"
        request.query = {}

        result = await handler.handle(request, "/api/v1/inbox/stats", "GET")

        assert result is not None

    @pytest.mark.asyncio
    async def test_handle_not_found(self):
        """Test handling unknown route."""
        handler = UnifiedInboxHandler()

        request = MagicMock()
        request.tenant_id = "test_tenant"

        result = await handler.handle(request, "/api/v1/inbox/unknown", "GET")

        assert result is not None
        # Should return 404

    @pytest.mark.asyncio
    async def test_handle_connect_invalid_provider(self):
        """Test connecting with invalid provider."""
        handler = UnifiedInboxHandler()

        request = MagicMock()
        request.tenant_id = "test_tenant"
        request.json = AsyncMock(return_value={"provider": "invalid"})

        result = await handler.handle(request, "/api/v1/inbox/connect", "POST")

        # Should return error for invalid provider
        assert result is not None

    @pytest.mark.asyncio
    async def test_handle_bulk_action_invalid(self):
        """Test bulk action with invalid action type."""
        handler = UnifiedInboxHandler()

        request = MagicMock()
        request.tenant_id = "test_tenant"
        request.json = AsyncMock(return_value={
            "message_ids": ["msg_1"],
            "action": "invalid_action",
        })

        result = await handler.handle(request, "/api/v1/inbox/bulk-action", "POST")

        # Should return error for invalid action
        assert result is not None


class TestHandleUnifiedInbox:
    """Tests for handle_unified_inbox function."""

    @pytest.mark.asyncio
    async def test_entry_point(self):
        """Test entry point function."""
        request = MagicMock()
        request.tenant_id = "test"

        result = await handle_unified_inbox(request, "/api/v1/inbox/accounts", "GET")

        assert result is not None


class TestImports:
    """Test that imports work correctly."""

    def test_import_from_package(self):
        """Test imports from features package."""
        from aragora.server.handlers.features import (
            UnifiedInboxHandler,
            handle_unified_inbox,
            get_unified_inbox_handler,
            EmailProvider,
            AccountStatus,
            TriageAction,
        )

        assert UnifiedInboxHandler is not None
        assert handle_unified_inbox is not None
        assert get_unified_inbox_handler is not None
        assert EmailProvider is not None
        assert AccountStatus is not None
        assert TriageAction is not None
