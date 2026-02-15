"""
Tests for InboxServicesMixin - email service integration methods.

Tests cover:
- Demo email fallback when Gmail connector is absent
- Prioritized email fetching
- Inbox stats calculation
- Sender profile lookup
- Daily digest computation
- Reprioritization flow
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.inbox_services import InboxServicesMixin


# ===========================================================================
# Concrete Test Class (mixin requires composing class)
# ===========================================================================


class ConcreteInboxServices(InboxServicesMixin):
    """Concrete class combining InboxServicesMixin for testing."""

    def __init__(
        self,
        gmail_connector=None,
        prioritizer=None,
        sender_history=None,
    ):
        self.gmail_connector = gmail_connector
        self.prioritizer = prioritizer
        self.sender_history = sender_history


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture(autouse=True)
def _clear_email_cache():
    """Clear email cache between tests."""
    try:
        from aragora.server.handlers.inbox_command import _email_cache

        _email_cache.clear()
    except (ImportError, AttributeError):
        pass
    yield
    try:
        from aragora.server.handlers.inbox_command import _email_cache

        _email_cache.clear()
    except (ImportError, AttributeError):
        pass


@pytest.fixture
def service():
    """Create a ConcreteInboxServices with no connectors (demo mode)."""
    return ConcreteInboxServices()


@pytest.fixture
def service_with_gmail():
    """Create a ConcreteInboxServices with a mock Gmail connector."""
    gmail = AsyncMock()
    gmail.list_messages = AsyncMock(return_value=[])
    return ConcreteInboxServices(gmail_connector=gmail)


# ===========================================================================
# Demo Email Fallback
# ===========================================================================


class TestDemoEmails:
    """Tests for _get_demo_emails when services are unavailable."""

    def test_returns_demo_emails_list(self, service):
        emails = service._get_demo_emails(limit=10, offset=0, priority_filter=None)
        assert isinstance(emails, list)
        assert len(emails) > 0

    def test_demo_emails_have_required_fields(self, service):
        emails = service._get_demo_emails(limit=10, offset=0, priority_filter=None)
        for email in emails:
            assert "id" in email
            assert "from" in email
            assert "subject" in email
            assert "priority" in email

    def test_demo_emails_respects_limit(self, service):
        emails = service._get_demo_emails(limit=2, offset=0, priority_filter=None)
        assert len(emails) <= 2

    def test_demo_emails_respects_offset(self, service):
        all_emails = service._get_demo_emails(limit=100, offset=0, priority_filter=None)
        offset_emails = service._get_demo_emails(limit=100, offset=2, priority_filter=None)
        assert len(offset_emails) == len(all_emails) - 2

    def test_demo_emails_priority_filter(self, service):
        emails = service._get_demo_emails(limit=100, offset=0, priority_filter="critical")
        for email in emails:
            assert email["priority"] == "critical"

    def test_demo_emails_populates_cache(self, service):
        try:
            from aragora.server.handlers.inbox_command import _email_cache

            service._get_demo_emails(limit=10, offset=0, priority_filter=None)
            assert len(_email_cache) > 0
        except ImportError:
            pytest.skip("inbox_command not available")


# ===========================================================================
# Fetch Prioritized Emails
# ===========================================================================


class TestFetchPrioritizedEmails:
    """Tests for _fetch_prioritized_emails."""

    @pytest.mark.asyncio
    async def test_returns_demo_when_no_connector(self, service):
        result = await service._fetch_prioritized_emails(
            limit=10, offset=0, priority_filter=None, unread_only=False
        )
        assert isinstance(result, list)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_returns_empty_when_gmail_returns_nothing(self, service_with_gmail):
        service_with_gmail.gmail_connector.list_messages.return_value = []
        result = await service_with_gmail._fetch_prioritized_emails(
            limit=10, offset=0, priority_filter=None, unread_only=False
        )
        assert result == []

    @pytest.mark.asyncio
    async def test_falls_back_to_demo_on_gmail_error(self):
        gmail = AsyncMock()
        gmail.list_messages = AsyncMock(side_effect=ConnectionError("timeout"))
        svc = ConcreteInboxServices(gmail_connector=gmail)
        result = await svc._fetch_prioritized_emails(
            limit=10, offset=0, priority_filter=None, unread_only=False
        )
        # Should return demo emails
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_basic_list_without_prioritizer(self):
        """When no prioritizer is set, returns basic email list."""
        msg = MagicMock()
        msg.id = "msg-1"
        msg.from_address = "sender@test.com"
        msg.subject = "Test Subject"
        msg.snippet = "Hello world"
        msg.date = datetime.utcnow()
        msg.unread = True

        gmail = AsyncMock()
        gmail.list_messages = AsyncMock(return_value=[msg])

        svc = ConcreteInboxServices(gmail_connector=gmail, prioritizer=None)
        result = await svc._fetch_prioritized_emails(
            limit=10, offset=0, priority_filter=None, unread_only=False
        )
        assert len(result) == 1
        assert result[0]["priority"] == "medium"  # Default when no prioritizer


# ===========================================================================
# Inbox Stats Calculation
# ===========================================================================


class TestCalculateInboxStats:
    """Tests for _calculate_inbox_stats."""

    @pytest.mark.asyncio
    async def test_empty_list_returns_zeros(self, service):
        stats = await service._calculate_inbox_stats([])
        assert stats["total"] == 0
        assert stats["unread"] == 0
        assert stats["actionRequired"] == 0

    @pytest.mark.asyncio
    async def test_counts_priorities(self, service):
        emails = [
            {"priority": "critical", "unread": True},
            {"priority": "high", "unread": True},
            {"priority": "medium", "unread": False},
            {"priority": "low", "unread": False},
            {"priority": "defer", "unread": False},
        ]
        stats = await service._calculate_inbox_stats(emails)
        assert stats["total"] == 5
        assert stats["critical"] == 1
        assert stats["high"] == 1
        assert stats["medium"] == 1
        assert stats["low"] == 1
        assert stats["deferred"] == 1
        assert stats["actionRequired"] == 2  # critical + high

    @pytest.mark.asyncio
    async def test_counts_unread(self, service):
        emails = [
            {"priority": "medium", "unread": True},
            {"priority": "low", "unread": True},
            {"priority": "low", "unread": False},
        ]
        stats = await service._calculate_inbox_stats(emails)
        assert stats["unread"] == 2


# ===========================================================================
# Sender Profile
# ===========================================================================


class TestGetSenderProfile:
    """Tests for _get_sender_profile."""

    @pytest.mark.asyncio
    async def test_basic_profile_without_services(self, service):
        profile = await service._get_sender_profile("alice@example.com")
        assert profile["email"] == "alice@example.com"
        assert profile["name"] == "alice"
        assert profile["isVip"] is False

    @pytest.mark.asyncio
    async def test_vip_from_prioritizer_config(self):
        prioritizer = MagicMock()
        prioritizer.config.vip_addresses = {"boss@example.com"}
        prioritizer.config.vip_domains = set()
        svc = ConcreteInboxServices(prioritizer=prioritizer)
        profile = await svc._get_sender_profile("boss@example.com")
        assert profile["isVip"] is True

    @pytest.mark.asyncio
    async def test_vip_from_domain(self):
        prioritizer = MagicMock()
        prioritizer.config.vip_addresses = set()
        prioritizer.config.vip_domains = {"vip.com"}
        svc = ConcreteInboxServices(prioritizer=prioritizer)
        profile = await svc._get_sender_profile("anyone@vip.com")
        assert profile["isVip"] is True

    @pytest.mark.asyncio
    async def test_sender_history_integration(self):
        sender_history = AsyncMock()
        mock_stats = MagicMock()
        mock_stats.is_vip = True
        mock_stats.reply_rate = 0.85
        mock_stats.avg_response_time_minutes = 120
        mock_stats.total_emails = 50
        mock_stats.last_email_date = datetime(2026, 1, 15)
        sender_history.get_sender_stats.return_value = mock_stats

        svc = ConcreteInboxServices(sender_history=sender_history)
        profile = await svc._get_sender_profile("colleague@company.com")
        assert profile["isVip"] is True
        assert profile["responseRate"] == 0.85
        assert profile["totalEmails"] == 50


# ===========================================================================
# Daily Digest
# ===========================================================================


class TestCalculateDailyDigest:
    """Tests for _calculate_daily_digest."""

    @pytest.mark.asyncio
    async def test_returns_digest_structure(self, service):
        digest = await service._calculate_daily_digest()
        assert "emailsReceived" in digest
        assert "emailsProcessed" in digest
        assert "topSenders" in digest
        assert "categoryBreakdown" in digest

    @pytest.mark.asyncio
    async def test_uses_sender_history_when_available(self):
        sender_history = AsyncMock()
        sender_history.get_daily_summary = AsyncMock(
            return_value={"emailsReceived": 42, "emailsProcessed": 40}
        )
        svc = ConcreteInboxServices(sender_history=sender_history)
        digest = await svc._calculate_daily_digest()
        assert digest["emailsReceived"] == 42

    @pytest.mark.asyncio
    async def test_falls_back_to_cache_on_history_error(self):
        sender_history = AsyncMock()
        sender_history.get_daily_summary = AsyncMock(side_effect=RuntimeError("fail"))
        svc = ConcreteInboxServices(sender_history=sender_history)
        digest = await svc._calculate_daily_digest()
        assert isinstance(digest["emailsReceived"], int)
