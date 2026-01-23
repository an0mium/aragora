"""
Tests for Unified Inbox Handler.

Tests cover enums, dataclasses, and basic handler creation.
"""

import pytest
from datetime import datetime, timezone

from aragora.server.handlers.features.unified_inbox import (
    UnifiedInboxHandler,
    EmailProvider,
    AccountStatus,
    TriageAction,
    ConnectedAccount,
    UnifiedMessage,
    TriageResult,
    InboxStats,
)


class TestEmailProviderEnum:
    """Tests for EmailProvider enum."""

    def test_providers_defined(self):
        """Test that email providers are defined."""
        assert EmailProvider.GMAIL.value == "gmail"
        assert EmailProvider.OUTLOOK.value == "outlook"


class TestAccountStatusEnum:
    """Tests for AccountStatus enum."""

    def test_all_statuses_defined(self):
        """Test that all account statuses are available."""
        expected = ["pending", "connected", "syncing", "error", "disconnected"]
        for status in expected:
            assert AccountStatus(status) is not None


class TestTriageActionEnum:
    """Tests for TriageAction enum."""

    def test_all_actions_defined(self):
        """Test that all triage actions are available."""
        expected = [
            "respond_urgent",
            "respond_normal",
            "delegate",
            "schedule",
            "archive",
            "delete",
            "flag",
            "defer",
        ]
        for action in expected:
            assert TriageAction(action) is not None


class TestConnectedAccount:
    """Tests for ConnectedAccount dataclass."""

    def test_account_creation(self):
        """Test creating a connected account."""
        account = ConnectedAccount(
            id="acc_123",
            provider=EmailProvider.GMAIL,
            email_address="user@gmail.com",
            display_name="Test User",
            status=AccountStatus.CONNECTED,
            connected_at=datetime.now(timezone.utc),
            total_messages=500,
            unread_count=25,
        )

        assert account.id == "acc_123"
        assert account.provider == EmailProvider.GMAIL
        assert account.status == AccountStatus.CONNECTED
        assert account.total_messages == 500

    def test_account_defaults(self):
        """Test account with default values."""
        account = ConnectedAccount(
            id="acc_456",
            provider=EmailProvider.OUTLOOK,
            email_address="user@outlook.com",
            display_name="Outlook User",
            status=AccountStatus.PENDING,
            connected_at=datetime.now(timezone.utc),
        )

        assert account.last_sync is None
        assert account.total_messages == 0
        assert account.unread_count == 0
        assert account.sync_errors == 0

    def test_account_to_dict(self):
        """Test account serialization."""
        account = ConnectedAccount(
            id="acc_test",
            provider=EmailProvider.GMAIL,
            email_address="test@gmail.com",
            display_name="Test",
            status=AccountStatus.CONNECTED,
            connected_at=datetime.now(timezone.utc),
        )

        data = account.to_dict()
        assert data["id"] == "acc_test"
        assert data["provider"] == "gmail"
        assert data["status"] == "connected"


class TestUnifiedMessage:
    """Tests for UnifiedMessage dataclass."""

    def test_message_creation(self):
        """Test creating a unified message."""
        message = UnifiedMessage(
            id="msg_123",
            account_id="acc_456",
            provider=EmailProvider.GMAIL,
            external_id="gmail_abc123",
            subject="Urgent: Contract Review",
            sender_email="sender@example.com",
            sender_name="John Doe",
            recipients=["user@gmail.com"],
            cc=["manager@example.com"],
            received_at=datetime.now(timezone.utc),
            snippet="Please review the attached contract...",
            body_preview="Please review the attached contract for Q4...",
            is_read=False,
            is_starred=True,
            has_attachments=True,
            labels=["inbox", "important"],
            priority_score=0.95,
            priority_tier="critical",
        )

        assert message.id == "msg_123"
        assert message.subject == "Urgent: Contract Review"
        assert message.priority_tier == "critical"
        assert message.has_attachments is True

    def test_message_defaults(self):
        """Test message with default values."""
        message = UnifiedMessage(
            id="msg_789",
            account_id="acc_xyz",
            provider=EmailProvider.OUTLOOK,
            external_id="outlook_def456",
            subject="Weekly Update",
            sender_email="team@company.com",
            sender_name="Team Lead",
            recipients=["user@outlook.com"],
            cc=[],
            received_at=datetime.now(timezone.utc),
            snippet="Weekly team update...",
            body_preview="Here's our weekly update...",
            is_read=True,
            is_starred=False,
            has_attachments=False,
            labels=["inbox"],
        )

        assert message.priority_score == 0.5
        assert message.priority_tier == "medium"
        assert message.triage_action is None
        assert message.thread_id is None

    def test_message_to_dict(self):
        """Test message serialization."""
        message = UnifiedMessage(
            id="msg_test",
            account_id="acc_test",
            provider=EmailProvider.GMAIL,
            external_id="ext_test",
            subject="Test Subject",
            sender_email="sender@test.com",
            sender_name="Sender",
            recipients=["recipient@test.com"],
            cc=[],
            received_at=datetime.now(timezone.utc),
            snippet="Test snippet...",
            body_preview="Test body...",
            is_read=False,
            is_starred=False,
            has_attachments=False,
            labels=["inbox"],
        )

        data = message.to_dict()
        assert data["id"] == "msg_test"
        assert data["sender"]["email"] == "sender@test.com"
        assert data["priority"]["tier"] == "medium"


class TestTriageResult:
    """Tests for TriageResult dataclass."""

    def test_triage_result_creation(self):
        """Test creating a triage result."""
        result = TriageResult(
            message_id="msg_123",
            recommended_action=TriageAction.RESPOND_URGENT,
            confidence=0.92,
            rationale="High priority sender with time-sensitive request",
            suggested_response="Thank you for reaching out. I will review...",
            delegate_to=None,
            schedule_for=None,
            agents_involved=["support_analyst", "product_expert"],
            debate_summary="Both agents agreed on urgent response",
        )

        assert result.message_id == "msg_123"
        assert result.recommended_action == TriageAction.RESPOND_URGENT
        assert result.confidence == 0.92

    def test_triage_result_to_dict(self):
        """Test triage result serialization."""
        result = TriageResult(
            message_id="msg_test",
            recommended_action=TriageAction.ARCHIVE,
            confidence=0.85,
            rationale="Low priority newsletter",
            suggested_response=None,
            delegate_to=None,
            schedule_for=None,
            agents_involved=["classifier"],
            debate_summary=None,
        )

        data = result.to_dict()
        assert data["message_id"] == "msg_test"
        assert data["recommended_action"] == "archive"
        assert data["confidence"] == 0.85


class TestInboxStats:
    """Tests for InboxStats dataclass."""

    def test_inbox_stats_creation(self):
        """Test creating inbox stats."""
        stats = InboxStats(
            total_accounts=2,
            total_messages=500,
            unread_count=45,
            messages_by_priority={"critical": 5, "high": 15, "medium": 200, "low": 280},
            messages_by_provider={"gmail": 300, "outlook": 200},
            avg_response_time_hours=4.5,
            pending_triage=25,
            sync_health={"accounts_healthy": 2, "accounts_error": 0},
            top_senders=[{"email": "important@company.com", "count": 50}],
            hourly_volume=[{"hour": 9, "count": 25}],
        )

        assert stats.total_accounts == 2
        assert stats.total_messages == 500
        assert stats.unread_count == 45

    def test_inbox_stats_to_dict(self):
        """Test inbox stats serialization."""
        stats = InboxStats(
            total_accounts=1,
            total_messages=100,
            unread_count=10,
            messages_by_priority={"critical": 1, "high": 5, "medium": 50, "low": 44},
            messages_by_provider={"gmail": 100},
            avg_response_time_hours=2.5,
            pending_triage=5,
            sync_health={"accounts_healthy": 1},
            top_senders=[],
            hourly_volume=[],
        )

        data = stats.to_dict()
        assert data["total_accounts"] == 1
        assert data["unread_count"] == 10
        assert data["avg_response_time_hours"] == 2.5


class TestUnifiedInboxHandler:
    """Tests for UnifiedInboxHandler class."""

    def test_handler_creation(self):
        """Test creating handler instance."""
        handler = UnifiedInboxHandler(server_context={})
        assert handler is not None

    def test_handler_routes(self):
        """Test that handler has route definitions."""
        assert hasattr(UnifiedInboxHandler, "ROUTES")
        routes = UnifiedInboxHandler.ROUTES
        assert "/api/v1/inbox/connect" in routes
        assert "/api/v1/inbox/accounts" in routes
        assert "/api/v1/inbox/messages" in routes
        assert "/api/v1/inbox/triage" in routes
        assert "/api/v1/inbox/stats" in routes
