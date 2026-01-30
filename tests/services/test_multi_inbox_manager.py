"""Tests for the Multi-Inbox Manager service."""

from __future__ import annotations

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.services.multi_inbox_manager import (
    AccountType,
    CrossAccountSenderProfile,
    InboxAccount,
    MultiInboxManager,
    UnifiedEmail,
)


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------


class TestMultiInboxDataclasses:
    def test_account_type_values(self):
        assert AccountType.PERSONAL.value == "personal"
        assert AccountType.WORK.value == "work"
        assert AccountType.SHARED.value == "shared"

    def test_inbox_account_to_dict(self):
        acc = InboxAccount(
            account_id="personal",
            email_address="me@example.com",
            account_type=AccountType.PERSONAL,
            is_primary=True,
            is_connected=True,
            total_emails=50,
            unread_count=10,
        )
        d = acc.to_dict()
        assert d["account_id"] == "personal"
        assert d["email_address"] == "me@example.com"
        assert d["account_type"] == "personal"
        assert d["is_primary"] is True
        assert d["total_emails"] == 50

    def test_inbox_account_defaults(self):
        acc = InboxAccount(
            account_id="test",
            email_address="test@example.com",
        )
        assert acc.account_type == AccountType.OTHER
        assert acc.is_connected is False
        assert acc.priority_weight == 1.0

    def test_cross_account_sender_profile_account_count(self):
        profile = CrossAccountSenderProfile(sender_email="test@example.com")
        profile.seen_in_accounts = {"personal", "work"}
        assert profile.account_count == 2

    def test_cross_account_sender_profile_reply_rate(self):
        profile = CrossAccountSenderProfile(sender_email="test@example.com")
        profile.total_emails_received = 10
        profile.total_emails_replied = 5
        assert profile.reply_rate == 0.5

    def test_cross_account_sender_profile_reply_rate_zero(self):
        profile = CrossAccountSenderProfile(sender_email="test@example.com")
        assert profile.reply_rate == 0.0


# ---------------------------------------------------------------------------
# CrossAccountSenderProfile.compute_importance
# ---------------------------------------------------------------------------


class TestComputeImportance:
    def test_importance_single_account(self):
        profile = CrossAccountSenderProfile(sender_email="test@example.com")
        profile.seen_in_accounts = {"personal"}
        score = profile.compute_importance()
        assert score == 0.0  # Single account, no bonus

    def test_importance_multi_account(self):
        profile = CrossAccountSenderProfile(sender_email="test@example.com")
        profile.seen_in_accounts = {"personal", "work"}
        score = profile.compute_importance()
        assert score > 0.0

    def test_importance_replied_from_multiple(self):
        profile = CrossAccountSenderProfile(sender_email="test@example.com")
        profile.seen_in_accounts = {"personal", "work"}
        profile.replied_from_accounts = {"personal", "work"}
        score = profile.compute_importance()
        assert score >= 0.5  # Multi-account + multi-reply bonus

    def test_importance_starred(self):
        profile = CrossAccountSenderProfile(sender_email="test@example.com")
        profile.seen_in_accounts = {"personal", "work"}
        profile.starred_in_accounts = {"personal"}
        score = profile.compute_importance()
        assert score > 0.0

    def test_importance_high_reply_rate(self):
        profile = CrossAccountSenderProfile(sender_email="test@example.com")
        profile.seen_in_accounts = {"personal", "work"}
        profile.total_emails_received = 10
        profile.total_emails_replied = 8
        score = profile.compute_importance()
        assert score > 0.0

    def test_importance_capped_at_1(self):
        profile = CrossAccountSenderProfile(sender_email="test@example.com")
        profile.seen_in_accounts = {"a", "b", "c", "d"}
        profile.replied_from_accounts = {"a", "b", "c"}
        profile.starred_in_accounts = {"a", "b"}
        profile.total_emails_received = 10
        profile.total_emails_replied = 9
        score = profile.compute_importance()
        assert score <= 1.0


# ---------------------------------------------------------------------------
# MultiInboxManager init
# ---------------------------------------------------------------------------


class TestMultiInboxManagerInit:
    def test_init_defaults(self):
        manager = MultiInboxManager()
        assert manager.user_id == "default"
        assert len(manager._accounts) == 0

    def test_init_custom(self):
        manager = MultiInboxManager(user_id="user_123")
        assert manager.user_id == "user_123"


# ---------------------------------------------------------------------------
# Account management (mocking GmailConnector)
# ---------------------------------------------------------------------------


class TestAccountManagement:
    def test_get_accounts_empty(self):
        manager = MultiInboxManager()
        assert manager.get_accounts() == []

    def test_get_account_not_found(self):
        manager = MultiInboxManager()
        assert manager.get_account("nonexistent") is None

    @pytest.mark.asyncio
    async def test_remove_nonexistent_account(self):
        manager = MultiInboxManager()
        result = await manager.remove_account("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_remove_existing_account(self):
        manager = MultiInboxManager()
        # Manually add an account
        acc = InboxAccount(
            account_id="test",
            email_address="test@example.com",
            is_connected=True,
        )
        manager._accounts["test"] = acc
        manager._connectors["test"] = MagicMock()
        result = await manager.remove_account("test")
        assert result is True
        assert "test" not in manager._accounts


# ---------------------------------------------------------------------------
# Sender profile lookup
# ---------------------------------------------------------------------------


class TestSenderProfile:
    @pytest.mark.asyncio
    async def test_get_sender_profile_not_found(self):
        manager = MultiInboxManager()
        result = await manager.get_sender_profile("unknown@example.com")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_sender_profile_found(self):
        manager = MultiInboxManager()
        profile = CrossAccountSenderProfile(sender_email="test@example.com")
        profile.seen_in_accounts = {"personal"}
        manager._sender_profiles["test@example.com"] = profile
        result = await manager.get_sender_profile("test@example.com")
        assert result is not None
        assert result.sender_email == "test@example.com"


# ---------------------------------------------------------------------------
# Sender importance
# ---------------------------------------------------------------------------


class TestSenderImportance:
    @pytest.mark.asyncio
    async def test_unknown_sender(self):
        manager = MultiInboxManager()
        result = await manager.get_sender_importance("unknown@example.com")
        assert result["importance_score"] == 0.0
        assert result["is_known"] is False

    @pytest.mark.asyncio
    async def test_known_sender(self):
        manager = MultiInboxManager()
        profile = CrossAccountSenderProfile(sender_email="known@example.com")
        profile.seen_in_accounts = {"personal", "work"}
        profile.replied_from_accounts = {"personal"}
        manager._sender_profiles["known@example.com"] = profile
        result = await manager.get_sender_importance("known@example.com")
        assert result["is_known"] is True
        assert result["importance_score"] > 0.0
        assert result["account_count"] == 2


# ---------------------------------------------------------------------------
# Record action
# ---------------------------------------------------------------------------


class TestRecordAction:
    @pytest.mark.asyncio
    async def test_record_opened(self):
        manager = MultiInboxManager()
        await manager.record_action("personal", "email_1", "sender@example.com", "opened")
        profile = manager._sender_profiles["sender@example.com"]
        assert profile.total_emails_opened == 1

    @pytest.mark.asyncio
    async def test_record_replied(self):
        manager = MultiInboxManager()
        await manager.record_action("personal", "email_1", "sender@example.com", "replied")
        profile = manager._sender_profiles["sender@example.com"]
        assert profile.total_emails_replied == 1
        assert "personal" in profile.replied_from_accounts

    @pytest.mark.asyncio
    async def test_record_starred(self):
        manager = MultiInboxManager()
        await manager.record_action("personal", "email_1", "sender@example.com", "starred")
        profile = manager._sender_profiles["sender@example.com"]
        assert "personal" in profile.starred_in_accounts

    @pytest.mark.asyncio
    async def test_record_action_creates_profile(self):
        manager = MultiInboxManager()
        await manager.record_action("work", "email_1", "new@example.com", "opened")
        assert "new@example.com" in manager._sender_profiles

    @pytest.mark.asyncio
    async def test_record_action_updates_importance(self):
        manager = MultiInboxManager()
        await manager.record_action("personal", "e1", "x@example.com", "replied")
        await manager.record_action("work", "e2", "x@example.com", "replied")
        profile = manager._sender_profiles["x@example.com"]
        assert profile.cross_account_importance > 0.0


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


class TestMultiInboxStats:
    def test_empty_stats(self):
        manager = MultiInboxManager()
        stats = manager.get_stats()
        assert stats["account_count"] == 0
        assert stats["known_senders"] == 0

    def test_stats_with_data(self):
        manager = MultiInboxManager(user_id="u1")
        manager._accounts["personal"] = InboxAccount(
            account_id="personal",
            email_address="me@example.com",
        )
        profile = CrossAccountSenderProfile(sender_email="s@example.com")
        profile.seen_in_accounts = {"personal", "work"}
        manager._sender_profiles["s@example.com"] = profile
        stats = manager.get_stats()
        assert stats["account_count"] == 1
        assert stats["known_senders"] == 1
        assert stats["cross_account_senders"] == 1


# ---------------------------------------------------------------------------
# Prioritize unified emails (without external prioritizer)
# ---------------------------------------------------------------------------


class TestPrioritizeUnifiedEmails:
    @pytest.mark.asyncio
    async def test_simple_scoring_no_prioritizer(self):
        manager = MultiInboxManager()
        manager._accounts["personal"] = InboxAccount(
            account_id="personal",
            email_address="me@example.com",
            priority_weight=1.0,
        )

        mock_email = MagicMock()
        mock_email.is_starred = True
        mock_email.is_important = False

        unified = UnifiedEmail(
            email=mock_email,
            account_id="personal",
            account_type=AccountType.PERSONAL,
            is_cross_account_important=True,
        )
        result = await manager._prioritize_unified_emails([unified])
        assert len(result) == 1
        assert result[0].unified_score > 0.5  # Base + cross-account + starred
