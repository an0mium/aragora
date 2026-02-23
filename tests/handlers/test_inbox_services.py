"""Tests for inbox services mixin (aragora/server/handlers/inbox_services.py).

Covers all methods of the InboxServicesMixin class:
- _fetch_prioritized_emails: Gmail fetch + prioritization, demo fallback
- _get_demo_emails: demo email data generation
- _calculate_inbox_stats: inbox statistics computation
- _get_sender_profile: sender profile lookup via SenderHistoryService
- _calculate_daily_digest: daily digest statistics
- _reprioritize_emails: batch AI re-prioritization

Tests are organized into classes for each method:
- TestFetchPrioritizedEmails: live Gmail + prioritizer path
- TestFetchPrioritizedEmailsNoPrioritizer: no prioritizer path
- TestFetchPrioritizedEmailsFallback: error fallback to demo
- TestGetDemoEmails: demo data generation + filtering
- TestCalculateInboxStats: stats computation
- TestGetSenderProfile: sender profile with/without services
- TestCalculateDailyDigest: daily digest with/without cache
- TestReprioritizeEmails: batch reprioritization
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Mock types that mirror the real services
# ---------------------------------------------------------------------------


class _MockEmailPriority(Enum):
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    DEFER = 5


class _MockScoringTier(Enum):
    TIER_1_RULES = "tier_1_rules"
    TIER_2_LIGHTWEIGHT = "tier_2_lightweight"
    TIER_3_DEBATE = "tier_3_debate"


@dataclass
class _MockPriorityResult:
    email_id: str
    priority: _MockEmailPriority
    confidence: float
    tier_used: _MockScoringTier
    rationale: str
    sender_score: float = 0.5
    content_urgency_score: float = 0.6
    context_relevance_score: float = 0.4
    time_sensitivity_score: float = 0.3
    suggested_labels: list[str] = field(default_factory=list)
    auto_archive: bool = False


@dataclass
class _MockSenderStats:
    sender_email: str
    total_emails: int = 10
    is_vip: bool = False
    reply_rate: float = 0.5
    avg_response_time_minutes: float | None = 120.0
    last_email_date: datetime | None = None


@dataclass
class _MockEmailMessage:
    id: str
    from_address: str = "sender@example.com"
    subject: str = "Test Subject"
    snippet: str = "Test snippet text"
    date: datetime = None
    unread: bool = True

    def __post_init__(self):
        if self.date is None:
            self.date = datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_email_cache():
    """Clear the module-level email cache between tests."""
    from aragora.server.handlers.inbox_command import _email_cache, _priority_results

    _email_cache.clear()
    _priority_results.clear()
    yield
    _email_cache.clear()
    _priority_results.clear()


@pytest.fixture(autouse=True)
def _reset_rate_limiters():
    """Reset rate limiters between tests."""
    from aragora.server.handlers.utils.rate_limit import clear_all_limiters

    clear_all_limiters()
    yield
    clear_all_limiters()


@pytest.fixture
def handler():
    """Create an InboxCommandHandler with mocked services (no services by default)."""
    from aragora.server.handlers.inbox_command import InboxCommandHandler

    with patch("aragora.server.handlers.inbox_command.ServiceRegistry") as mock_registry_cls:
        mock_registry = MagicMock()
        mock_registry.has.return_value = False
        mock_registry_cls.get.return_value = mock_registry

        h = InboxCommandHandler(
            gmail_connector=None,
            prioritizer=None,
            sender_history=None,
        )
        h._initialized = True
        return h


@pytest.fixture
def gmail_connector():
    """Create a mock Gmail connector."""
    connector = AsyncMock()
    connector.list_messages = AsyncMock(return_value=[])
    connector.get_messages = AsyncMock(return_value=[])
    connector.get_message = AsyncMock(return_value=None)
    return connector


@pytest.fixture
def prioritizer():
    """Create a mock EmailPrioritizer."""
    p = AsyncMock()
    p.rank_inbox = AsyncMock(return_value=[])
    p.score_emails = AsyncMock(return_value=[])
    p.config = MagicMock()
    p.config.vip_addresses = set()
    p.config.vip_domains = set()
    p.config.auto_archive_senders = set()
    return p


@pytest.fixture
def sender_history():
    """Create a mock SenderHistoryService."""
    svc = AsyncMock()
    svc.get_sender_stats = AsyncMock(return_value=None)
    return svc


@pytest.fixture
def handler_with_services(gmail_connector, prioritizer, sender_history):
    """Create an InboxCommandHandler with all services mocked."""
    from aragora.server.handlers.inbox_command import InboxCommandHandler

    with patch("aragora.server.handlers.inbox_command.ServiceRegistry") as mock_registry_cls:
        mock_registry = MagicMock()
        mock_registry.has.return_value = False
        mock_registry_cls.get.return_value = mock_registry

        h = InboxCommandHandler(
            gmail_connector=gmail_connector,
            prioritizer=prioritizer,
            sender_history=sender_history,
        )
        h._initialized = True
        return h


# ============================================================================
# _fetch_prioritized_emails - with Gmail + Prioritizer
# ============================================================================


class TestFetchPrioritizedEmails:
    """Tests for _fetch_prioritized_emails with live services."""

    @pytest.mark.asyncio
    async def test_returns_demo_when_no_gmail(self, handler):
        """Falls back to demo emails when gmail_connector is None."""
        emails = await handler._fetch_prioritized_emails(
            limit=10,
            offset=0,
            priority_filter=None,
            unread_only=False,
        )
        assert len(emails) > 0
        assert any(e["id"].startswith("demo_") for e in emails)

    @pytest.mark.asyncio
    async def test_returns_empty_when_gmail_returns_nothing(
        self,
        handler_with_services,
        gmail_connector,
    ):
        """Returns empty list when Gmail has no messages."""
        gmail_connector.list_messages.return_value = []

        emails = await handler_with_services._fetch_prioritized_emails(
            limit=10,
            offset=0,
            priority_filter=None,
            unread_only=False,
        )
        assert emails == []

    @pytest.mark.asyncio
    async def test_prioritizes_emails_via_rank_inbox(
        self,
        handler_with_services,
        gmail_connector,
        prioritizer,
    ):
        """Fetches emails from Gmail and ranks them via prioritizer."""
        msg1 = _MockEmailMessage(id="e1", from_address="a@b.com", subject="Urgent")
        msg2 = _MockEmailMessage(id="e2", from_address="c@d.com", subject="FYI")
        gmail_connector.list_messages.return_value = [msg1, msg2]

        result1 = _MockPriorityResult(
            email_id="e1",
            priority=_MockEmailPriority.CRITICAL,
            confidence=0.95,
            tier_used=_MockScoringTier.TIER_1_RULES,
            rationale="VIP sender",
        )
        result2 = _MockPriorityResult(
            email_id="e2",
            priority=_MockEmailPriority.LOW,
            confidence=0.7,
            tier_used=_MockScoringTier.TIER_1_RULES,
            rationale="Newsletter",
            auto_archive=True,
        )
        prioritizer.rank_inbox.return_value = [result1, result2]

        emails = await handler_with_services._fetch_prioritized_emails(
            limit=50,
            offset=0,
            priority_filter=None,
            unread_only=False,
        )

        assert len(emails) == 2
        assert emails[0]["id"] == "e1"
        assert emails[0]["priority"] == "critical"
        assert emails[0]["confidence"] == 0.95
        assert emails[0]["reasoning"] == "VIP sender"
        assert emails[0]["from"] == "a@b.com"
        assert emails[0]["subject"] == "Urgent"
        assert emails[0]["tier_used"] == "tier_1_rules"
        assert "scores" in emails[0]
        assert emails[0]["scores"]["sender"] == 0.5
        assert emails[0]["scores"]["urgency"] == 0.6

        assert emails[1]["id"] == "e2"
        assert emails[1]["priority"] == "low"
        assert emails[1]["auto_archive"] is True

    @pytest.mark.asyncio
    async def test_priority_filter_applied(
        self,
        handler_with_services,
        gmail_connector,
        prioritizer,
    ):
        """Filters by priority when priority_filter is set."""
        msg1 = _MockEmailMessage(id="e1")
        msg2 = _MockEmailMessage(id="e2")
        gmail_connector.list_messages.return_value = [msg1, msg2]

        result1 = _MockPriorityResult(
            email_id="e1",
            priority=_MockEmailPriority.CRITICAL,
            confidence=0.9,
            tier_used=_MockScoringTier.TIER_1_RULES,
            rationale="Urgent",
        )
        result2 = _MockPriorityResult(
            email_id="e2",
            priority=_MockEmailPriority.LOW,
            confidence=0.7,
            tier_used=_MockScoringTier.TIER_1_RULES,
            rationale="Low priority",
        )
        prioritizer.rank_inbox.return_value = [result1, result2]

        emails = await handler_with_services._fetch_prioritized_emails(
            limit=50,
            offset=0,
            priority_filter="critical",
            unread_only=False,
        )
        assert len(emails) == 1
        assert emails[0]["priority"] == "critical"

    @pytest.mark.asyncio
    async def test_unread_only_filter(
        self,
        handler_with_services,
        gmail_connector,
        prioritizer,
    ):
        """Filters out read emails when unread_only is True."""
        msg1 = _MockEmailMessage(id="e1", unread=True)
        msg2 = _MockEmailMessage(id="e2", unread=False)
        gmail_connector.list_messages.return_value = [msg1, msg2]

        result1 = _MockPriorityResult(
            email_id="e1",
            priority=_MockEmailPriority.HIGH,
            confidence=0.8,
            tier_used=_MockScoringTier.TIER_1_RULES,
            rationale="High",
        )
        result2 = _MockPriorityResult(
            email_id="e2",
            priority=_MockEmailPriority.MEDIUM,
            confidence=0.6,
            tier_used=_MockScoringTier.TIER_1_RULES,
            rationale="Medium",
        )
        prioritizer.rank_inbox.return_value = [result1, result2]

        emails = await handler_with_services._fetch_prioritized_emails(
            limit=50,
            offset=0,
            priority_filter=None,
            unread_only=True,
        )
        assert len(emails) == 1
        assert emails[0]["id"] == "e1"
        assert emails[0]["unread"] is True

    @pytest.mark.asyncio
    async def test_pagination_with_offset_and_limit(
        self,
        handler_with_services,
        gmail_connector,
        prioritizer,
    ):
        """Applies offset and limit to paginate results."""
        messages = [_MockEmailMessage(id=f"e{i}") for i in range(5)]
        gmail_connector.list_messages.return_value = messages

        results = [
            _MockPriorityResult(
                email_id=f"e{i}",
                priority=_MockEmailPriority.MEDIUM,
                confidence=0.7,
                tier_used=_MockScoringTier.TIER_1_RULES,
                rationale=f"Result {i}",
            )
            for i in range(5)
        ]
        prioritizer.rank_inbox.return_value = results

        emails = await handler_with_services._fetch_prioritized_emails(
            limit=2,
            offset=1,
            priority_filter=None,
            unread_only=False,
        )
        assert len(emails) == 2
        assert emails[0]["id"] == "e1"
        assert emails[1]["id"] == "e2"

    @pytest.mark.asyncio
    async def test_populates_email_cache(
        self,
        handler_with_services,
        gmail_connector,
        prioritizer,
    ):
        """Caches prioritized results in _email_cache."""
        from aragora.server.handlers.inbox_command import _email_cache

        msg1 = _MockEmailMessage(id="cached_e1")
        gmail_connector.list_messages.return_value = [msg1]

        result1 = _MockPriorityResult(
            email_id="cached_e1",
            priority=_MockEmailPriority.HIGH,
            confidence=0.85,
            tier_used=_MockScoringTier.TIER_1_RULES,
            rationale="Cached",
        )
        prioritizer.rank_inbox.return_value = [result1]

        await handler_with_services._fetch_prioritized_emails(
            limit=50,
            offset=0,
            priority_filter=None,
            unread_only=False,
        )
        cached = _email_cache.get("cached_e1")
        assert cached is not None
        assert cached["priority"] == "high"

    @pytest.mark.asyncio
    async def test_snippet_truncated_to_200_chars(
        self,
        handler_with_services,
        gmail_connector,
        prioritizer,
    ):
        """Snippet is truncated to 200 characters."""
        long_snippet = "x" * 500
        msg = _MockEmailMessage(id="e1", snippet=long_snippet)
        gmail_connector.list_messages.return_value = [msg]

        result = _MockPriorityResult(
            email_id="e1",
            priority=_MockEmailPriority.MEDIUM,
            confidence=0.7,
            tier_used=_MockScoringTier.TIER_1_RULES,
            rationale="Test",
        )
        prioritizer.rank_inbox.return_value = [result]

        emails = await handler_with_services._fetch_prioritized_emails(
            limit=50,
            offset=0,
            priority_filter=None,
            unread_only=False,
        )
        assert len(emails[0]["snippet"]) == 200

    @pytest.mark.asyncio
    async def test_email_not_found_in_messages(
        self,
        handler_with_services,
        gmail_connector,
        prioritizer,
    ):
        """Handles case where priority result ID does not match any message."""
        msg = _MockEmailMessage(id="e1")
        gmail_connector.list_messages.return_value = [msg]

        # Priority result references a non-existent email ID
        result = _MockPriorityResult(
            email_id="nonexistent",
            priority=_MockEmailPriority.MEDIUM,
            confidence=0.7,
            tier_used=_MockScoringTier.TIER_1_RULES,
            rationale="Missing email",
        )
        prioritizer.rank_inbox.return_value = [result]

        emails = await handler_with_services._fetch_prioritized_emails(
            limit=50,
            offset=0,
            priority_filter=None,
            unread_only=False,
        )
        # Should still produce an entry with "unknown" from/subject
        assert len(emails) == 1
        assert emails[0]["from"] == "unknown"
        assert emails[0]["subject"] == "No subject"


# ============================================================================
# _fetch_prioritized_emails - without prioritizer
# ============================================================================


class TestFetchPrioritizedEmailsNoPrioritizer:
    """Tests for _fetch_prioritized_emails when prioritizer is None."""

    @pytest.mark.asyncio
    async def test_basic_list_without_prioritizer(self, gmail_connector):
        """Returns basic email list when prioritizer is None."""
        from aragora.server.handlers.inbox_command import InboxCommandHandler

        with patch("aragora.server.handlers.inbox_command.ServiceRegistry") as mock_reg:
            mock_reg.get.return_value = MagicMock(has=MagicMock(return_value=False))

            h = InboxCommandHandler(
                gmail_connector=gmail_connector,
                prioritizer=None,
                sender_history=None,
            )
            h._initialized = True
            # Ensure no prioritizer was auto-created by _ensure_services
            h.prioritizer = None

        msg1 = _MockEmailMessage(id="e1", from_address="a@b.com", subject="Hello")
        msg2 = _MockEmailMessage(id="e2", from_address="c@d.com", subject="World")
        gmail_connector.list_messages.return_value = [msg1, msg2]

        emails = await h._fetch_prioritized_emails(
            limit=50,
            offset=0,
            priority_filter=None,
            unread_only=False,
        )
        assert len(emails) == 2
        assert emails[0]["id"] == "e1"
        assert emails[0]["from"] == "a@b.com"
        assert emails[0]["subject"] == "Hello"
        assert emails[0]["priority"] == "medium"
        assert emails[0]["confidence"] == 0.5

    @pytest.mark.asyncio
    async def test_pagination_without_prioritizer(self, gmail_connector):
        """Applies offset and limit even without prioritizer."""
        from aragora.server.handlers.inbox_command import InboxCommandHandler

        with patch("aragora.server.handlers.inbox_command.ServiceRegistry") as mock_reg:
            mock_reg.get.return_value = MagicMock(has=MagicMock(return_value=False))

            h = InboxCommandHandler(
                gmail_connector=gmail_connector,
                prioritizer=None,
                sender_history=None,
            )
            h._initialized = True
            h.prioritizer = None

        messages = [_MockEmailMessage(id=f"e{i}") for i in range(5)]
        gmail_connector.list_messages.return_value = messages

        emails = await h._fetch_prioritized_emails(
            limit=2,
            offset=1,
            priority_filter=None,
            unread_only=False,
        )
        assert len(emails) == 2
        assert emails[0]["id"] == "e1"
        assert emails[1]["id"] == "e2"

    @pytest.mark.asyncio
    async def test_snippet_truncated_without_prioritizer(self, gmail_connector):
        """Snippet truncated to 200 chars even without prioritizer."""
        from aragora.server.handlers.inbox_command import InboxCommandHandler

        with patch("aragora.server.handlers.inbox_command.ServiceRegistry") as mock_reg:
            mock_reg.get.return_value = MagicMock(has=MagicMock(return_value=False))

            h = InboxCommandHandler(
                gmail_connector=gmail_connector,
                prioritizer=None,
                sender_history=None,
            )
            h._initialized = True
            h.prioritizer = None

        msg = _MockEmailMessage(id="e1", snippet="y" * 400)
        gmail_connector.list_messages.return_value = [msg]

        emails = await h._fetch_prioritized_emails(
            limit=50,
            offset=0,
            priority_filter=None,
            unread_only=False,
        )
        assert len(emails[0]["snippet"]) == 200


# ============================================================================
# _fetch_prioritized_emails - error fallback
# ============================================================================


class TestFetchPrioritizedEmailsFallback:
    """Tests for fallback to demo data on errors."""

    @pytest.mark.asyncio
    async def test_falls_back_on_os_error(
        self,
        handler_with_services,
        gmail_connector,
    ):
        """Falls back to demo data on OSError from Gmail."""
        gmail_connector.list_messages.side_effect = OSError("Connection refused")

        emails = await handler_with_services._fetch_prioritized_emails(
            limit=10,
            offset=0,
            priority_filter=None,
            unread_only=False,
        )
        assert len(emails) > 0
        assert any(e["id"].startswith("demo_") for e in emails)

    @pytest.mark.asyncio
    async def test_falls_back_on_connection_error(
        self,
        handler_with_services,
        gmail_connector,
    ):
        """Falls back to demo data on ConnectionError."""
        gmail_connector.list_messages.side_effect = ConnectionError("DNS failure")

        emails = await handler_with_services._fetch_prioritized_emails(
            limit=10,
            offset=0,
            priority_filter=None,
            unread_only=False,
        )
        assert any(e["id"].startswith("demo_") for e in emails)

    @pytest.mark.asyncio
    async def test_falls_back_on_runtime_error(
        self,
        handler_with_services,
        gmail_connector,
    ):
        """Falls back to demo data on RuntimeError."""
        gmail_connector.list_messages.side_effect = RuntimeError("Bad state")

        emails = await handler_with_services._fetch_prioritized_emails(
            limit=10,
            offset=0,
            priority_filter=None,
            unread_only=False,
        )
        assert any(e["id"].startswith("demo_") for e in emails)

    @pytest.mark.asyncio
    async def test_falls_back_on_value_error(
        self,
        handler_with_services,
        gmail_connector,
    ):
        """Falls back to demo data on ValueError."""
        gmail_connector.list_messages.side_effect = ValueError("Invalid token")

        emails = await handler_with_services._fetch_prioritized_emails(
            limit=10,
            offset=0,
            priority_filter=None,
            unread_only=False,
        )
        assert any(e["id"].startswith("demo_") for e in emails)

    @pytest.mark.asyncio
    async def test_falls_back_on_attribute_error(
        self,
        handler_with_services,
        gmail_connector,
    ):
        """Falls back to demo data on AttributeError."""
        gmail_connector.list_messages.side_effect = AttributeError("No method")

        emails = await handler_with_services._fetch_prioritized_emails(
            limit=10,
            offset=0,
            priority_filter=None,
            unread_only=False,
        )
        assert any(e["id"].startswith("demo_") for e in emails)


# ============================================================================
# _get_demo_emails
# ============================================================================


class TestGetDemoEmails:
    """Tests for _get_demo_emails method."""

    def test_returns_five_demo_emails(self, handler):
        """Returns 5 demo emails by default."""
        emails = handler._get_demo_emails(limit=50, offset=0, priority_filter=None)
        assert len(emails) == 5

    def test_demo_emails_have_required_fields(self, handler):
        """Each demo email has all expected fields."""
        emails = handler._get_demo_emails(limit=50, offset=0, priority_filter=None)
        for email in emails:
            assert "id" in email
            assert "from" in email
            assert "subject" in email
            assert "priority" in email
            assert "confidence" in email
            assert "timestamp" in email

    def test_demo_email_priorities(self, handler):
        """Demo emails cover multiple priority levels."""
        emails = handler._get_demo_emails(limit=50, offset=0, priority_filter=None)
        priorities = {e["priority"] for e in emails}
        assert "critical" in priorities
        assert "high" in priorities
        assert "medium" in priorities
        assert "low" in priorities
        assert "defer" in priorities

    def test_filter_by_priority(self, handler):
        """Filters demo emails by priority."""
        emails = handler._get_demo_emails(limit=50, offset=0, priority_filter="critical")
        assert len(emails) == 1
        assert emails[0]["priority"] == "critical"

    def test_filter_by_high_priority(self, handler):
        """Filters demo emails by high priority."""
        emails = handler._get_demo_emails(limit=50, offset=0, priority_filter="high")
        assert len(emails) == 1
        assert emails[0]["priority"] == "high"

    def test_filter_by_defer_priority(self, handler):
        """Filters demo emails by defer priority."""
        emails = handler._get_demo_emails(limit=50, offset=0, priority_filter="defer")
        assert len(emails) == 1
        assert emails[0]["priority"] == "defer"
        assert emails[0].get("auto_archive") is True

    def test_filter_by_nonexistent_priority(self, handler):
        """Returns empty for a non-matching priority."""
        emails = handler._get_demo_emails(limit=50, offset=0, priority_filter="nonexistent")
        assert emails == []

    def test_limit_restricts_results(self, handler):
        """Limit caps the number of returned emails."""
        emails = handler._get_demo_emails(limit=2, offset=0, priority_filter=None)
        assert len(emails) == 2

    def test_offset_skips_results(self, handler):
        """Offset skips the first N results."""
        all_emails = handler._get_demo_emails(limit=50, offset=0, priority_filter=None)
        offset_emails = handler._get_demo_emails(limit=50, offset=2, priority_filter=None)
        assert len(offset_emails) == len(all_emails) - 2
        assert offset_emails[0]["id"] == all_emails[2]["id"]

    def test_offset_and_limit_combined(self, handler):
        """Offset + limit work together for pagination."""
        emails = handler._get_demo_emails(limit=1, offset=1, priority_filter=None)
        assert len(emails) == 1

    def test_offset_beyond_range(self, handler):
        """Offset beyond available data returns empty."""
        emails = handler._get_demo_emails(limit=50, offset=100, priority_filter=None)
        assert emails == []

    def test_populates_email_cache(self, handler):
        """Demo emails are cached in _email_cache."""
        from aragora.server.handlers.inbox_command import _email_cache

        handler._get_demo_emails(limit=50, offset=0, priority_filter=None)
        assert _email_cache.get("demo_1") is not None
        assert _email_cache.get("demo_2") is not None

    def test_demo_emails_have_timestamps(self, handler):
        """Demo emails have valid ISO 8601 timestamps."""
        emails = handler._get_demo_emails(limit=50, offset=0, priority_filter=None)
        for email in emails:
            # Should parse without error
            datetime.fromisoformat(email["timestamp"])


# ============================================================================
# _calculate_inbox_stats
# ============================================================================


class TestCalculateInboxStats:
    """Tests for _calculate_inbox_stats method."""

    @pytest.mark.asyncio
    async def test_empty_list(self, handler):
        """Stats for empty email list."""
        stats = await handler._calculate_inbox_stats([])
        assert stats["total"] == 0
        assert stats["unread"] == 0
        assert stats["critical"] == 0
        assert stats["high"] == 0
        assert stats["medium"] == 0
        assert stats["low"] == 0
        assert stats["deferred"] == 0
        assert stats["actionRequired"] == 0

    @pytest.mark.asyncio
    async def test_counts_by_priority(self, handler):
        """Counts emails by priority level."""
        emails = [
            {"priority": "critical", "unread": True},
            {"priority": "critical", "unread": True},
            {"priority": "high", "unread": True},
            {"priority": "medium", "unread": False},
            {"priority": "low", "unread": False},
            {"priority": "defer", "unread": False},
        ]
        stats = await handler._calculate_inbox_stats(emails)
        assert stats["total"] == 6
        assert stats["critical"] == 2
        assert stats["high"] == 1
        assert stats["medium"] == 1
        assert stats["low"] == 1
        assert stats["deferred"] == 1

    @pytest.mark.asyncio
    async def test_unread_count(self, handler):
        """Counts unread emails."""
        emails = [
            {"priority": "medium", "unread": True},
            {"priority": "medium", "unread": True},
            {"priority": "medium", "unread": False},
        ]
        stats = await handler._calculate_inbox_stats(emails)
        assert stats["unread"] == 2

    @pytest.mark.asyncio
    async def test_action_required_count(self, handler):
        """actionRequired = critical + high."""
        emails = [
            {"priority": "critical", "unread": True},
            {"priority": "high", "unread": True},
            {"priority": "medium", "unread": True},
        ]
        stats = await handler._calculate_inbox_stats(emails)
        assert stats["actionRequired"] == 2

    @pytest.mark.asyncio
    async def test_unknown_priority_ignored(self, handler):
        """Emails with unknown priority are counted in total but not in buckets."""
        emails = [
            {"priority": "unknown_level", "unread": True},
            {"priority": "critical", "unread": True},
        ]
        stats = await handler._calculate_inbox_stats(emails)
        assert stats["total"] == 2
        assert stats["critical"] == 1
        # "unknown_level" is not a valid priority, so it is not counted
        # The sum of all known priorities should be 1
        known_sum = (
            stats["critical"] + stats["high"] + stats["medium"] + stats["low"] + stats["deferred"]
        )
        assert known_sum == 1

    @pytest.mark.asyncio
    async def test_missing_priority_defaults_to_medium(self, handler):
        """Emails without priority key default to medium."""
        emails = [{"unread": True}]
        stats = await handler._calculate_inbox_stats(emails)
        assert stats["medium"] == 1

    @pytest.mark.asyncio
    async def test_missing_unread_defaults_to_false(self, handler):
        """Emails without unread key default to False."""
        emails = [{"priority": "medium"}]
        stats = await handler._calculate_inbox_stats(emails)
        assert stats["unread"] == 0

    @pytest.mark.asyncio
    async def test_all_critical(self, handler):
        """All emails critical."""
        emails = [{"priority": "critical", "unread": True} for _ in range(3)]
        stats = await handler._calculate_inbox_stats(emails)
        assert stats["critical"] == 3
        assert stats["actionRequired"] == 3

    @pytest.mark.asyncio
    async def test_priority_case_sensitivity(self, handler):
        """Priority matching is case-sensitive (lowered before passing)."""
        emails = [{"priority": "CRITICAL", "unread": True}]
        stats = await handler._calculate_inbox_stats(emails)
        # The code does .lower() so "CRITICAL" -> "critical"
        assert stats["critical"] == 1


# ============================================================================
# _get_sender_profile
# ============================================================================


class TestGetSenderProfile:
    """Tests for _get_sender_profile method."""

    @pytest.mark.asyncio
    async def test_basic_profile_no_services(self, handler):
        """Returns basic profile when no services are available."""
        profile = await handler._get_sender_profile("user@example.com")
        assert profile["email"] == "user@example.com"
        assert profile["name"] == "user"
        assert profile["isVip"] is False
        assert profile["isInternal"] is False
        assert profile["responseRate"] == 0.0
        assert profile["avgResponseTime"] == "N/A"
        assert profile["totalEmails"] == 0
        assert profile["lastContact"] == "Unknown"

    @pytest.mark.asyncio
    async def test_name_extracted_from_email(self, handler):
        """Name is local part of email address."""
        profile = await handler._get_sender_profile("john.doe@company.com")
        assert profile["name"] == "john.doe"

    @pytest.mark.asyncio
    async def test_with_sender_history_service(self, handler_with_services, sender_history):
        """Returns full profile when SenderHistoryService has data."""
        stats = _MockSenderStats(
            sender_email="boss@company.com",
            total_emails=42,
            is_vip=True,
            reply_rate=0.75,
            avg_response_time_minutes=120.0,
            last_email_date=datetime(2026, 2, 15, tzinfo=timezone.utc),
        )
        sender_history.get_sender_stats.return_value = stats

        profile = await handler_with_services._get_sender_profile("boss@company.com")
        assert profile["email"] == "boss@company.com"
        assert profile["name"] == "boss"
        assert profile["isVip"] is True
        assert profile["responseRate"] == 0.75
        assert profile["avgResponseTime"] == "2.0h"
        assert profile["totalEmails"] == 42
        assert profile["lastContact"] == "2026-02-15"

    @pytest.mark.asyncio
    async def test_with_sender_history_no_response_time(
        self,
        handler_with_services,
        sender_history,
    ):
        """Handles None avg_response_time_minutes gracefully."""
        stats = _MockSenderStats(
            sender_email="fast@company.com",
            avg_response_time_minutes=None,
            last_email_date=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        sender_history.get_sender_stats.return_value = stats

        profile = await handler_with_services._get_sender_profile("fast@company.com")
        assert profile["avgResponseTime"] == "N/A"

    @pytest.mark.asyncio
    async def test_with_sender_history_no_last_date(
        self,
        handler_with_services,
        sender_history,
    ):
        """Handles None last_email_date gracefully."""
        stats = _MockSenderStats(
            sender_email="ghost@company.com",
            last_email_date=None,
        )
        sender_history.get_sender_stats.return_value = stats

        profile = await handler_with_services._get_sender_profile("ghost@company.com")
        assert profile["lastContact"] == "Never"

    @pytest.mark.asyncio
    async def test_sender_history_returns_none(
        self,
        handler_with_services,
        sender_history,
    ):
        """Falls back to basic profile when sender_history returns None."""
        sender_history.get_sender_stats.return_value = None

        profile = await handler_with_services._get_sender_profile("unknown@example.com")
        assert profile["totalEmails"] == 0
        assert profile["isVip"] is False

    @pytest.mark.asyncio
    async def test_sender_history_raises_error(
        self,
        handler_with_services,
        sender_history,
    ):
        """Falls back to basic profile on sender_history errors."""
        sender_history.get_sender_stats.side_effect = RuntimeError("DB down")

        profile = await handler_with_services._get_sender_profile("error@example.com")
        assert profile["email"] == "error@example.com"
        assert profile["totalEmails"] == 0

    @pytest.mark.asyncio
    async def test_sender_history_os_error(
        self,
        handler_with_services,
        sender_history,
    ):
        """Falls back gracefully on OSError from sender_history."""
        sender_history.get_sender_stats.side_effect = OSError("Network error")

        profile = await handler_with_services._get_sender_profile("err@example.com")
        assert profile["email"] == "err@example.com"
        assert profile["totalEmails"] == 0

    @pytest.mark.asyncio
    async def test_vip_from_prioritizer_address(
        self,
        handler_with_services,
        prioritizer,
        sender_history,
    ):
        """Detects VIP from prioritizer config vip_addresses."""
        sender_history.get_sender_stats.return_value = None
        prioritizer.config.vip_addresses = {"ceo@company.com"}
        prioritizer.config.vip_domains = set()

        profile = await handler_with_services._get_sender_profile("ceo@company.com")
        assert profile["isVip"] is True

    @pytest.mark.asyncio
    async def test_vip_from_prioritizer_domain(
        self,
        handler_with_services,
        prioritizer,
        sender_history,
    ):
        """Detects VIP from prioritizer config vip_domains."""
        sender_history.get_sender_stats.return_value = None
        prioritizer.config.vip_addresses = set()
        prioritizer.config.vip_domains = {"bigcorp.com"}

        profile = await handler_with_services._get_sender_profile("anyone@bigcorp.com")
        assert profile["isVip"] is True

    @pytest.mark.asyncio
    async def test_vip_case_insensitive(
        self,
        handler_with_services,
        prioritizer,
        sender_history,
    ):
        """VIP matching is case-insensitive."""
        sender_history.get_sender_stats.return_value = None
        prioritizer.config.vip_addresses = {"CEO@Company.COM"}
        prioritizer.config.vip_domains = set()

        profile = await handler_with_services._get_sender_profile("ceo@company.com")
        assert profile["isVip"] is True

    @pytest.mark.asyncio
    async def test_not_vip_when_no_match(
        self,
        handler_with_services,
        prioritizer,
        sender_history,
    ):
        """isVip is False when sender is not in VIP lists."""
        sender_history.get_sender_stats.return_value = None
        prioritizer.config.vip_addresses = {"other@vip.com"}
        prioritizer.config.vip_domains = {"vipdomain.com"}

        profile = await handler_with_services._get_sender_profile("random@example.com")
        assert profile["isVip"] is False

    @pytest.mark.asyncio
    async def test_email_without_at_sign(self, handler):
        """Handles email without @ gracefully."""
        profile = await handler._get_sender_profile("noemail")
        assert profile["email"] == "noemail"
        assert profile["name"] == "noemail"


# ============================================================================
# _calculate_daily_digest
# ============================================================================


class TestCalculateDailyDigest:
    """Tests for _calculate_daily_digest method."""

    @pytest.mark.asyncio
    async def test_empty_cache_digest(self, handler):
        """Digest with empty cache returns zeros."""
        digest = await handler._calculate_daily_digest()
        assert digest["emailsReceived"] == 0
        assert digest["emailsProcessed"] == 0
        assert digest["criticalHandled"] == 0
        assert digest["timeSaved"] == "0 min"
        assert "topSenders" in digest
        assert "categoryBreakdown" in digest

    @pytest.mark.asyncio
    async def test_digest_with_cached_emails(self, handler):
        """Digest reflects cached email data."""
        from aragora.server.handlers.inbox_command import _email_cache

        _email_cache.set(
            "e1",
            {
                "priority": "critical",
                "from": "boss@co.com",
                "category": "Work",
            },
        )
        _email_cache.set(
            "e2",
            {
                "priority": "high",
                "from": "boss@co.com",
                "category": "Work",
            },
        )
        _email_cache.set(
            "e3",
            {
                "priority": "low",
                "from": "news@blog.com",
                "category": "Newsletter",
            },
        )

        digest = await handler._calculate_daily_digest()
        assert digest["emailsReceived"] == 3
        assert digest["emailsProcessed"] == 3
        assert digest["criticalHandled"] == 1
        assert digest["timeSaved"] == "6 min"  # 3 * 2 min

    @pytest.mark.asyncio
    async def test_top_senders(self, handler):
        """Top senders are ranked by count."""
        from aragora.server.handlers.inbox_command import _email_cache

        for i in range(5):
            _email_cache.set(f"from_a_{i}", {"from": "a@co.com", "category": "Work"})
        for i in range(3):
            _email_cache.set(f"from_b_{i}", {"from": "b@co.com", "category": "Work"})
        _email_cache.set("from_c_0", {"from": "c@co.com", "category": "Work"})

        digest = await handler._calculate_daily_digest()
        senders = digest["topSenders"]
        assert len(senders) >= 3
        assert senders[0]["name"] == "a@co.com"
        assert senders[0]["count"] == 5
        assert senders[1]["name"] == "b@co.com"
        assert senders[1]["count"] == 3

    @pytest.mark.asyncio
    async def test_top_senders_max_five(self, handler):
        """Top senders list is capped at 5."""
        from aragora.server.handlers.inbox_command import _email_cache

        for i in range(7):
            _email_cache.set(f"s{i}", {"from": f"sender{i}@co.com", "category": "Work"})

        digest = await handler._calculate_daily_digest()
        assert len(digest["topSenders"]) <= 5

    @pytest.mark.asyncio
    async def test_category_breakdown(self, handler):
        """Category breakdown shows percentages."""
        from aragora.server.handlers.inbox_command import _email_cache

        _email_cache.set("e1", {"from": "a@b.com", "category": "Work"})
        _email_cache.set("e2", {"from": "b@b.com", "category": "Work"})
        _email_cache.set("e3", {"from": "c@b.com", "category": "Newsletter"})

        digest = await handler._calculate_daily_digest()
        breakdown = digest["categoryBreakdown"]
        assert len(breakdown) == 2
        # Sorted by count descending
        assert breakdown[0]["category"] == "Work"
        assert breakdown[0]["count"] == 2
        assert breakdown[0]["percentage"] == 67  # round(2/3 * 100)
        assert breakdown[1]["category"] == "Newsletter"
        assert breakdown[1]["count"] == 1
        assert breakdown[1]["percentage"] == 33

    @pytest.mark.asyncio
    async def test_default_category_is_general(self, handler):
        """Emails without category get 'General'."""
        from aragora.server.handlers.inbox_command import _email_cache

        _email_cache.set("e1", {"from": "a@b.com"})

        digest = await handler._calculate_daily_digest()
        breakdown = digest["categoryBreakdown"]
        assert breakdown[0]["category"] == "General"

    @pytest.mark.asyncio
    async def test_uses_sender_history_daily_summary(
        self,
        handler_with_services,
        sender_history,
    ):
        """Uses sender_history.get_daily_summary when available."""
        daily_data = {
            "emailsReceived": 100,
            "emailsProcessed": 95,
            "criticalHandled": 5,
            "timeSaved": "190 min",
            "topSenders": [{"name": "top@co.com", "count": 10}],
            "categoryBreakdown": [{"category": "Work", "count": 80, "percentage": 84}],
        }
        sender_history.get_daily_summary = AsyncMock(return_value=daily_data)

        digest = await handler_with_services._calculate_daily_digest()
        assert digest["emailsReceived"] == 100
        assert digest["criticalHandled"] == 5

    @pytest.mark.asyncio
    async def test_falls_back_when_daily_summary_fails(
        self,
        handler_with_services,
        sender_history,
    ):
        """Falls back to cache-based digest when get_daily_summary raises."""
        sender_history.get_daily_summary = AsyncMock(side_effect=RuntimeError("DB down"))

        digest = await handler_with_services._calculate_daily_digest()
        # Should not raise; uses cache fallback
        assert "emailsReceived" in digest

    @pytest.mark.asyncio
    async def test_falls_back_when_daily_summary_returns_none(
        self,
        handler_with_services,
        sender_history,
    ):
        """Falls back when get_daily_summary returns None."""
        sender_history.get_daily_summary = AsyncMock(return_value=None)

        digest = await handler_with_services._calculate_daily_digest()
        assert "emailsReceived" in digest

    @pytest.mark.asyncio
    async def test_no_get_daily_summary_method(
        self,
        handler_with_services,
        sender_history,
    ):
        """Falls back when sender_history has no get_daily_summary attribute."""
        # Remove the attribute so hasattr returns False
        if hasattr(sender_history, "get_daily_summary"):
            del sender_history.get_daily_summary

        digest = await handler_with_services._calculate_daily_digest()
        assert "emailsReceived" in digest


# ============================================================================
# _reprioritize_emails
# ============================================================================


class TestReprioritizeEmails:
    """Tests for _reprioritize_emails method."""

    @pytest.mark.asyncio
    async def test_no_prioritizer(self, handler):
        """Returns error dict when no prioritizer is available."""
        result = await handler._reprioritize_emails(email_ids=None, force_tier=None)
        assert result["count"] == 0
        assert result["changes"] == []
        assert "error" in result

    @pytest.mark.asyncio
    async def test_empty_cache_returns_no_changes(
        self,
        handler_with_services,
    ):
        """Returns 0 count when cache is empty."""
        result = await handler_with_services._reprioritize_emails(
            email_ids=None,
            force_tier=None,
        )
        assert result["count"] == 0
        assert result["changes"] == []

    @pytest.mark.asyncio
    async def test_specific_email_ids_not_in_cache(
        self,
        handler_with_services,
    ):
        """Returns 0 count when specified IDs are not in cache."""
        result = await handler_with_services._reprioritize_emails(
            email_ids=["missing1", "missing2"],
            force_tier=None,
        )
        assert result["count"] == 0
        assert result["changes"] == []

    @pytest.mark.asyncio
    async def test_reprioritize_with_gmail_batch_fetch(
        self,
        handler_with_services,
        gmail_connector,
        prioritizer,
    ):
        """Batch fetches from Gmail and scores emails."""
        from aragora.server.handlers.inbox_command import _email_cache

        _email_cache.set("e1", {"id": "e1", "priority": "medium", "confidence": 0.5})
        _email_cache.set("e2", {"id": "e2", "priority": "low", "confidence": 0.6})

        msg1 = _MockEmailMessage(id="e1")
        msg2 = _MockEmailMessage(id="e2")
        gmail_connector.get_messages.return_value = [msg1, msg2]

        result1 = _MockPriorityResult(
            email_id="e1",
            priority=_MockEmailPriority.CRITICAL,
            confidence=0.95,
            tier_used=_MockScoringTier.TIER_1_RULES,
            rationale="Urgent now",
            suggested_labels=["urgent"],
        )
        result2 = _MockPriorityResult(
            email_id="e2",
            priority=_MockEmailPriority.LOW,
            confidence=0.7,
            tier_used=_MockScoringTier.TIER_1_RULES,
            rationale="Still low",
        )
        prioritizer.score_emails.return_value = [result1, result2]

        result = await handler_with_services._reprioritize_emails(
            email_ids=None,
            force_tier=None,
        )

        assert result["count"] == 2
        # e1 changed from medium to critical
        assert len(result["changes"]) == 1
        change = result["changes"][0]
        assert change["email_id"] == "e1"
        assert change["old_priority"] == "medium"
        assert change["new_priority"] == "critical"
        assert change["new_confidence"] == 0.95

    @pytest.mark.asyncio
    async def test_reprioritize_specific_ids(
        self,
        handler_with_services,
        gmail_connector,
        prioritizer,
    ):
        """Reprioritizes only specified email IDs."""
        from aragora.server.handlers.inbox_command import _email_cache

        _email_cache.set("e1", {"id": "e1", "priority": "medium", "confidence": 0.5})
        _email_cache.set("e2", {"id": "e2", "priority": "low", "confidence": 0.6})

        msg1 = _MockEmailMessage(id="e1")
        gmail_connector.get_messages.return_value = [msg1]

        result1 = _MockPriorityResult(
            email_id="e1",
            priority=_MockEmailPriority.HIGH,
            confidence=0.85,
            tier_used=_MockScoringTier.TIER_2_LIGHTWEIGHT,
            rationale="Elevated",
        )
        prioritizer.score_emails.return_value = [result1]

        result = await handler_with_services._reprioritize_emails(
            email_ids=["e1"],
            force_tier=None,
        )

        # Only e1 was processed
        assert result["count"] == 1
        assert len(result["changes"]) == 1
        assert result["changes"][0]["email_id"] == "e1"

    @pytest.mark.asyncio
    async def test_reprioritize_with_force_tier(
        self,
        handler_with_services,
        gmail_connector,
        prioritizer,
    ):
        """Passes force_tier to score_emails as ScoringTier enum."""
        from aragora.server.handlers.inbox_command import _email_cache
        from aragora.services.email_prioritization import ScoringTier

        _email_cache.set("e1", {"id": "e1", "priority": "medium", "confidence": 0.5})

        msg1 = _MockEmailMessage(id="e1")
        gmail_connector.get_messages.return_value = [msg1]

        result1 = _MockPriorityResult(
            email_id="e1",
            priority=_MockEmailPriority.MEDIUM,
            confidence=0.7,
            tier_used=_MockScoringTier.TIER_3_DEBATE,
            rationale="Debate says medium",
        )
        prioritizer.score_emails.return_value = [result1]

        result = await handler_with_services._reprioritize_emails(
            email_ids=None,
            force_tier="tier_3_debate",
        )

        # Verify score_emails was called with the real ScoringTier enum
        call_kwargs = prioritizer.score_emails.call_args
        assert call_kwargs[1]["force_tier"] == ScoringTier.TIER_3_DEBATE

    @pytest.mark.asyncio
    async def test_reprioritize_no_change(
        self,
        handler_with_services,
        gmail_connector,
        prioritizer,
    ):
        """No changes recorded when priority stays the same."""
        from aragora.server.handlers.inbox_command import _email_cache

        _email_cache.set("e1", {"id": "e1", "priority": "medium", "confidence": 0.5})

        msg1 = _MockEmailMessage(id="e1")
        gmail_connector.get_messages.return_value = [msg1]

        result1 = _MockPriorityResult(
            email_id="e1",
            priority=_MockEmailPriority.MEDIUM,
            confidence=0.8,
            tier_used=_MockScoringTier.TIER_1_RULES,
            rationale="Same priority",
        )
        prioritizer.score_emails.return_value = [result1]

        result = await handler_with_services._reprioritize_emails(
            email_ids=None,
            force_tier=None,
        )
        assert result["count"] == 1
        assert result["changes"] == []

    @pytest.mark.asyncio
    async def test_reprioritize_updates_cache(
        self,
        handler_with_services,
        gmail_connector,
        prioritizer,
    ):
        """Cache is updated with new priority data after reprioritization."""
        from aragora.server.handlers.inbox_command import _email_cache

        _email_cache.set("e1", {"id": "e1", "priority": "medium", "confidence": 0.5})

        msg1 = _MockEmailMessage(id="e1")
        gmail_connector.get_messages.return_value = [msg1]

        result1 = _MockPriorityResult(
            email_id="e1",
            priority=_MockEmailPriority.CRITICAL,
            confidence=0.99,
            tier_used=_MockScoringTier.TIER_3_DEBATE,
            rationale="Very urgent now",
            suggested_labels=["urgent", "action-required"],
            auto_archive=False,
        )
        prioritizer.score_emails.return_value = [result1]

        await handler_with_services._reprioritize_emails(
            email_ids=None,
            force_tier=None,
        )

        cached = _email_cache.get("e1")
        assert cached["priority"] == "critical"
        assert cached["confidence"] == 0.99
        assert cached["reasoning"] == "Very urgent now"
        assert cached["suggested_labels"] == ["urgent", "action-required"]

    @pytest.mark.asyncio
    async def test_batch_fetch_fallback_on_error(
        self,
        handler_with_services,
        gmail_connector,
        prioritizer,
    ):
        """Falls back to individual fetches when batch fetch fails."""
        from aragora.server.handlers.inbox_command import _email_cache

        _email_cache.set("e1", {"id": "e1", "priority": "medium", "confidence": 0.5})

        gmail_connector.get_messages.side_effect = OSError("Batch failed")
        msg1 = _MockEmailMessage(id="e1")
        gmail_connector.get_message.return_value = msg1

        result1 = _MockPriorityResult(
            email_id="e1",
            priority=_MockEmailPriority.HIGH,
            confidence=0.85,
            tier_used=_MockScoringTier.TIER_1_RULES,
            rationale="High now",
        )
        prioritizer.score_emails.return_value = [result1]

        result = await handler_with_services._reprioritize_emails(
            email_ids=None,
            force_tier=None,
        )
        assert result["count"] == 1
        gmail_connector.get_message.assert_called_once_with("e1")

    @pytest.mark.asyncio
    async def test_individual_fetch_failure_skipped(
        self,
        handler_with_services,
        gmail_connector,
        prioritizer,
    ):
        """Individual fetch failures are skipped gracefully."""
        from aragora.server.handlers.inbox_command import _email_cache

        _email_cache.set("e1", {"id": "e1", "priority": "medium", "confidence": 0.5})

        gmail_connector.get_messages.side_effect = OSError("Batch failed")
        gmail_connector.get_message.side_effect = OSError("Individual failed too")

        result = await handler_with_services._reprioritize_emails(
            email_ids=None,
            force_tier=None,
        )
        # No messages were fetched so no scoring happens
        assert result["count"] == 1  # count of IDs attempted
        assert result["changes"] == []

    @pytest.mark.asyncio
    async def test_no_gmail_connector(
        self,
        handler_with_services,
        prioritizer,
    ):
        """Returns count with no changes when no Gmail connector."""
        from aragora.server.handlers.inbox_command import _email_cache

        handler_with_services.gmail_connector = None
        _email_cache.set("e1", {"id": "e1", "priority": "medium", "confidence": 0.5})

        result = await handler_with_services._reprioritize_emails(
            email_ids=None,
            force_tier=None,
        )
        assert result["count"] == 1
        assert result["changes"] == []

    @pytest.mark.asyncio
    async def test_scoring_failure_returns_error(
        self,
        handler_with_services,
        gmail_connector,
        prioritizer,
    ):
        """Returns error when batch scoring fails."""
        from aragora.server.handlers.inbox_command import _email_cache

        _email_cache.set("e1", {"id": "e1", "priority": "medium", "confidence": 0.5})

        msg1 = _MockEmailMessage(id="e1")
        gmail_connector.get_messages.return_value = [msg1]

        prioritizer.score_emails.side_effect = RuntimeError("Scoring engine down")

        result = await handler_with_services._reprioritize_emails(
            email_ids=None,
            force_tier=None,
        )
        assert result["count"] == 0
        assert "error" in result

    @pytest.mark.asyncio
    async def test_tier_used_in_result(
        self,
        handler_with_services,
        gmail_connector,
        prioritizer,
    ):
        """Result includes tier_used field."""
        from aragora.server.handlers.inbox_command import _email_cache

        _email_cache.set("e1", {"id": "e1", "priority": "medium", "confidence": 0.5})

        msg1 = _MockEmailMessage(id="e1")
        gmail_connector.get_messages.return_value = [msg1]

        result1 = _MockPriorityResult(
            email_id="e1",
            priority=_MockEmailPriority.MEDIUM,
            confidence=0.7,
            tier_used=_MockScoringTier.TIER_1_RULES,
            rationale="Same",
        )
        prioritizer.score_emails.return_value = [result1]

        result = await handler_with_services._reprioritize_emails(
            email_ids=None,
            force_tier="tier_1_rules",
        )
        assert result["tier_used"] == "tier_1_rules"

    @pytest.mark.asyncio
    async def test_auto_tier_when_no_force(
        self,
        handler_with_services,
        gmail_connector,
        prioritizer,
    ):
        """tier_used is 'auto' when no force_tier specified."""
        from aragora.server.handlers.inbox_command import _email_cache

        _email_cache.set("e1", {"id": "e1", "priority": "medium", "confidence": 0.5})

        msg1 = _MockEmailMessage(id="e1")
        gmail_connector.get_messages.return_value = [msg1]

        result1 = _MockPriorityResult(
            email_id="e1",
            priority=_MockEmailPriority.MEDIUM,
            confidence=0.7,
            tier_used=_MockScoringTier.TIER_1_RULES,
            rationale="Same",
        )
        prioritizer.score_emails.return_value = [result1]

        result = await handler_with_services._reprioritize_emails(
            email_ids=None,
            force_tier=None,
        )
        assert result["tier_used"] == "auto"

    @pytest.mark.asyncio
    async def test_force_tier_invalid_string_ignored(
        self,
        handler_with_services,
        gmail_connector,
        prioritizer,
    ):
        """Unknown force_tier string results in None scoring_tier."""
        from aragora.server.handlers.inbox_command import _email_cache

        _email_cache.set("e1", {"id": "e1", "priority": "medium", "confidence": 0.5})

        msg1 = _MockEmailMessage(id="e1")
        gmail_connector.get_messages.return_value = [msg1]

        result1 = _MockPriorityResult(
            email_id="e1",
            priority=_MockEmailPriority.MEDIUM,
            confidence=0.7,
            tier_used=_MockScoringTier.TIER_1_RULES,
            rationale="Same",
        )
        prioritizer.score_emails.return_value = [result1]

        result = await handler_with_services._reprioritize_emails(
            email_ids=None,
            force_tier="bogus_tier",
        )
        # Should still work, just passes None for scoring_tier
        call_kwargs = prioritizer.score_emails.call_args
        assert call_kwargs[1]["force_tier"] is None

    @pytest.mark.asyncio
    async def test_multiple_changes_tracked(
        self,
        handler_with_services,
        gmail_connector,
        prioritizer,
    ):
        """Multiple priority changes are all tracked."""
        from aragora.server.handlers.inbox_command import _email_cache

        _email_cache.set("e1", {"id": "e1", "priority": "low", "confidence": 0.4})
        _email_cache.set("e2", {"id": "e2", "priority": "medium", "confidence": 0.5})
        _email_cache.set("e3", {"id": "e3", "priority": "high", "confidence": 0.8})

        msg1 = _MockEmailMessage(id="e1")
        msg2 = _MockEmailMessage(id="e2")
        msg3 = _MockEmailMessage(id="e3")
        gmail_connector.get_messages.return_value = [msg1, msg2, msg3]

        prioritizer.score_emails.return_value = [
            _MockPriorityResult(
                email_id="e1",
                priority=_MockEmailPriority.HIGH,
                confidence=0.9,
                tier_used=_MockScoringTier.TIER_2_LIGHTWEIGHT,
                rationale="Elevated",
            ),
            _MockPriorityResult(
                email_id="e2",
                priority=_MockEmailPriority.CRITICAL,
                confidence=0.95,
                tier_used=_MockScoringTier.TIER_2_LIGHTWEIGHT,
                rationale="Very urgent",
            ),
            _MockPriorityResult(
                email_id="e3",
                priority=_MockEmailPriority.HIGH,
                confidence=0.85,
                tier_used=_MockScoringTier.TIER_2_LIGHTWEIGHT,
                rationale="Same high",
            ),
        ]

        result = await handler_with_services._reprioritize_emails(
            email_ids=None,
            force_tier=None,
        )
        assert result["count"] == 3
        # e1: low -> high, e2: medium -> critical, e3: high -> high (no change)
        assert len(result["changes"]) == 2
        changed_ids = {c["email_id"] for c in result["changes"]}
        assert changed_ids == {"e1", "e2"}
