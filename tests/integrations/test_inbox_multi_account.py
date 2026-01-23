"""
Tests for Inbox Multi-Account Flow.

Tests the integration of multiple email accounts including:
- Multi-account token storage and retrieval
- Cross-account inbox aggregation
- Account-specific prioritization
- Unified inbox sorting across accounts
- Account isolation and filtering
"""

from __future__ import annotations

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.storage.gmail_token_store import (
    GmailUserState,
    InMemoryGmailTokenStore,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def token_store():
    """Create an in-memory token store for testing."""
    return InMemoryGmailTokenStore()


@pytest.fixture
def sample_accounts():
    """Create sample multi-account states."""
    return [
        GmailUserState(
            user_id="user_work",
            email_address="user@company.com",
            access_token="work_access_token",
            refresh_token="work_refresh_token",
            token_expiry=datetime.now(timezone.utc) + timedelta(hours=1),
            history_id="work_history_123",
            indexed_count=500,
            connected_at=datetime.now(timezone.utc) - timedelta(days=30),
        ),
        GmailUserState(
            user_id="user_personal",
            email_address="user@gmail.com",
            access_token="personal_access_token",
            refresh_token="personal_refresh_token",
            token_expiry=datetime.now(timezone.utc) + timedelta(hours=1),
            history_id="personal_history_456",
            indexed_count=1200,
            connected_at=datetime.now(timezone.utc) - timedelta(days=60),
        ),
        GmailUserState(
            user_id="user_shared",
            email_address="team@company.com",
            access_token="shared_access_token",
            refresh_token="shared_refresh_token",
            token_expiry=datetime.now(timezone.utc) + timedelta(hours=1),
            history_id="shared_history_789",
            indexed_count=300,
            connected_at=datetime.now(timezone.utc) - timedelta(days=7),
        ),
    ]


class MockEmailMessage:
    """Mock email message for testing."""

    def __init__(
        self,
        id: str,
        account_id: str,
        subject: str,
        from_address: str,
        priority: str = "medium",
        date: datetime = None,
    ):
        self.id = id
        self.account_id = account_id
        self.subject = subject
        self.from_address = from_address
        self.priority = priority
        self.date = date or datetime.now(timezone.utc)
        self.to_addresses = ["recipient@example.com"]
        self.body_text = "Test email body"
        self.snippet = "Test snippet..."
        self.labels = ["INBOX"]
        self.is_read = False


@pytest.fixture
def sample_emails():
    """Create sample emails from multiple accounts."""
    now = datetime.now(timezone.utc)
    return [
        # Work emails
        MockEmailMessage(
            id="work_email_1",
            account_id="user_work",
            subject="Q4 Budget Review - ACTION REQUIRED",
            from_address="cfo@company.com",
            priority="critical",
            date=now - timedelta(hours=1),
        ),
        MockEmailMessage(
            id="work_email_2",
            account_id="user_work",
            subject="Team standup notes",
            from_address="manager@company.com",
            priority="medium",
            date=now - timedelta(hours=3),
        ),
        # Personal emails
        MockEmailMessage(
            id="personal_email_1",
            account_id="user_personal",
            subject="Your order has shipped",
            from_address="orders@amazon.com",
            priority="low",
            date=now - timedelta(hours=2),
        ),
        MockEmailMessage(
            id="personal_email_2",
            account_id="user_personal",
            subject="Flight confirmation",
            from_address="noreply@airline.com",
            priority="high",
            date=now - timedelta(hours=4),
        ),
        # Shared inbox emails
        MockEmailMessage(
            id="shared_email_1",
            account_id="user_shared",
            subject="Support ticket #12345",
            from_address="customer@client.com",
            priority="high",
            date=now - timedelta(minutes=30),
        ),
    ]


# =============================================================================
# Multi-Account Token Store Tests
# =============================================================================


class TestMultiAccountTokenStore:
    """Tests for multi-account token management."""

    @pytest.mark.asyncio
    async def test_store_multiple_accounts(self, token_store, sample_accounts):
        """Test storing credentials for multiple accounts."""
        for account in sample_accounts:
            await token_store.save(account)

        all_accounts = await token_store.list_all()
        assert len(all_accounts) == 3

    @pytest.mark.asyncio
    async def test_retrieve_specific_account(self, token_store, sample_accounts):
        """Test retrieving a specific account's credentials."""
        for account in sample_accounts:
            await token_store.save(account)

        work_account = await token_store.get("user_work")
        assert work_account is not None
        assert work_account.email_address == "user@company.com"
        assert work_account.access_token == "work_access_token"

    @pytest.mark.asyncio
    async def test_update_single_account(self, token_store, sample_accounts):
        """Test updating a single account without affecting others."""
        for account in sample_accounts:
            await token_store.save(account)

        # Update work account
        work_account = await token_store.get("user_work")
        work_account.indexed_count = 600
        work_account.access_token = "new_work_access_token"
        await token_store.save(work_account)

        # Verify update
        updated = await token_store.get("user_work")
        assert updated.indexed_count == 600
        assert updated.access_token == "new_work_access_token"

        # Verify other accounts unchanged
        personal = await token_store.get("user_personal")
        assert personal.access_token == "personal_access_token"

    @pytest.mark.asyncio
    async def test_delete_account_isolation(self, token_store, sample_accounts):
        """Test deleting one account doesn't affect others."""
        for account in sample_accounts:
            await token_store.save(account)

        # Delete personal account
        deleted = await token_store.delete("user_personal")
        assert deleted is True

        # Verify only personal is deleted
        all_accounts = await token_store.list_all()
        assert len(all_accounts) == 2

        work = await token_store.get("user_work")
        shared = await token_store.get("user_shared")
        personal = await token_store.get("user_personal")

        assert work is not None
        assert shared is not None
        assert personal is None

    @pytest.mark.asyncio
    async def test_expired_token_detection(self, token_store):
        """Test detection of expired tokens across accounts."""
        now = datetime.now(timezone.utc)

        accounts = [
            GmailUserState(
                user_id="expired_account",
                email_address="expired@test.com",
                access_token="expired_token",
                refresh_token="expired_refresh",
                token_expiry=now - timedelta(hours=1),  # Expired
            ),
            GmailUserState(
                user_id="valid_account",
                email_address="valid@test.com",
                access_token="valid_token",
                refresh_token="valid_refresh",
                token_expiry=now + timedelta(hours=1),  # Valid
            ),
        ]

        for account in accounts:
            await token_store.save(account)

        expired = await token_store.get("expired_account")
        valid = await token_store.get("valid_account")

        assert expired.token_expiry < now
        assert valid.token_expiry > now


# =============================================================================
# Multi-Account Inbox Aggregation Tests
# =============================================================================


class TestMultiAccountInboxAggregation:
    """Tests for aggregating emails across multiple accounts."""

    def test_aggregate_emails_from_all_accounts(self, sample_emails):
        """Test combining emails from multiple accounts."""
        all_emails = sample_emails
        assert len(all_emails) == 5

        # Verify all accounts represented
        accounts = set(email.account_id for email in all_emails)
        assert accounts == {"user_work", "user_personal", "user_shared"}

    def test_filter_emails_by_account(self, sample_emails):
        """Test filtering emails to single account."""
        work_emails = [e for e in sample_emails if e.account_id == "user_work"]
        assert len(work_emails) == 2
        assert all(e.account_id == "user_work" for e in work_emails)

    def test_sort_aggregated_by_date(self, sample_emails):
        """Test sorting aggregated emails by date across accounts."""
        sorted_emails = sorted(sample_emails, key=lambda e: e.date, reverse=True)

        # Most recent should be shared inbox (30 min ago)
        assert sorted_emails[0].id == "shared_email_1"

        # Dates should be descending
        for i in range(len(sorted_emails) - 1):
            assert sorted_emails[i].date >= sorted_emails[i + 1].date

    def test_sort_aggregated_by_priority(self, sample_emails):
        """Test sorting aggregated emails by priority across accounts."""
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}

        sorted_emails = sorted(sample_emails, key=lambda e: priority_order.get(e.priority, 99))

        # Critical work email should be first
        assert sorted_emails[0].priority == "critical"

        # Low priority should be last
        assert sorted_emails[-1].priority == "low"

    def test_combined_priority_and_date_sort(self, sample_emails):
        """Test sorting by priority first, then by date."""
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}

        sorted_emails = sorted(
            sample_emails,
            key=lambda e: (priority_order.get(e.priority, 99), -e.date.timestamp()),
        )

        # Critical should come first
        assert sorted_emails[0].priority == "critical"

        # Among same priority, more recent should come first
        high_priority = [e for e in sorted_emails if e.priority == "high"]
        if len(high_priority) > 1:
            for i in range(len(high_priority) - 1):
                assert high_priority[i].date >= high_priority[i + 1].date


# =============================================================================
# Account-Specific Prioritization Tests
# =============================================================================


class TestAccountSpecificPrioritization:
    """Tests for account-specific prioritization rules."""

    def test_work_account_vip_senders(self):
        """Test VIP sender detection for work account."""
        work_vip_domains = {"company.com", "important-client.com"}

        email_from_vip = MockEmailMessage(
            id="vip_email",
            account_id="user_work",
            subject="Project Update",
            from_address="ceo@company.com",
        )

        email_from_external = MockEmailMessage(
            id="external_email",
            account_id="user_work",
            subject="Newsletter",
            from_address="newsletter@marketing.com",
        )

        # Extract domain from email
        vip_domain = email_from_vip.from_address.split("@")[1]
        external_domain = email_from_external.from_address.split("@")[1]

        assert vip_domain in work_vip_domains
        assert external_domain not in work_vip_domains

    def test_personal_account_newsletter_detection(self):
        """Test newsletter detection for personal account."""
        newsletter_patterns = ["newsletter", "noreply", "unsubscribe", "marketing"]

        newsletter_email = MockEmailMessage(
            id="newsletter_email",
            account_id="user_personal",
            subject="Weekly Newsletter - Unsubscribe anytime",
            from_address="newsletter@news.com",
        )

        regular_email = MockEmailMessage(
            id="regular_email",
            account_id="user_personal",
            subject="Dinner plans",
            from_address="friend@gmail.com",
        )

        # Check subject and sender for newsletter patterns
        newsletter_subject = newsletter_email.subject.lower()
        newsletter_sender = newsletter_email.from_address.lower()

        is_newsletter = any(
            pattern in newsletter_subject or pattern in newsletter_sender
            for pattern in newsletter_patterns
        )
        assert is_newsletter is True

        regular_subject = regular_email.subject.lower()
        regular_sender = regular_email.from_address.lower()

        is_regular_newsletter = any(
            pattern in regular_subject or pattern in regular_sender
            for pattern in newsletter_patterns
        )
        assert is_regular_newsletter is False

    def test_shared_inbox_urgency_detection(self):
        """Test urgency detection for shared support inbox."""
        urgency_keywords = ["urgent", "asap", "critical", "emergency", "immediately"]

        urgent_ticket = MockEmailMessage(
            id="urgent_ticket",
            account_id="user_shared",
            subject="URGENT: System down - need help ASAP",
            from_address="customer@client.com",
        )

        normal_ticket = MockEmailMessage(
            id="normal_ticket",
            account_id="user_shared",
            subject="Feature request for next release",
            from_address="customer@client.com",
        )

        urgent_subject = urgent_ticket.subject.lower()
        is_urgent = any(keyword in urgent_subject for keyword in urgency_keywords)
        assert is_urgent is True

        normal_subject = normal_ticket.subject.lower()
        is_normal_urgent = any(keyword in normal_subject for keyword in urgency_keywords)
        assert is_normal_urgent is False


# =============================================================================
# Unified Inbox Statistics Tests
# =============================================================================


class TestUnifiedInboxStatistics:
    """Tests for statistics across all accounts."""

    def test_calculate_per_account_stats(self, sample_emails):
        """Test calculating statistics per account."""
        stats = {}
        for email in sample_emails:
            account = email.account_id
            if account not in stats:
                stats[account] = {"total": 0, "by_priority": {}}
            stats[account]["total"] += 1

            priority = email.priority
            if priority not in stats[account]["by_priority"]:
                stats[account]["by_priority"][priority] = 0
            stats[account]["by_priority"][priority] += 1

        assert stats["user_work"]["total"] == 2
        assert stats["user_personal"]["total"] == 2
        assert stats["user_shared"]["total"] == 1

        assert stats["user_work"]["by_priority"]["critical"] == 1
        assert stats["user_work"]["by_priority"]["medium"] == 1

    def test_calculate_unified_stats(self, sample_emails):
        """Test calculating unified statistics across all accounts."""
        stats = {
            "total": len(sample_emails),
            "by_priority": {},
            "by_account": {},
        }

        for email in sample_emails:
            # Priority counts
            priority = email.priority
            stats["by_priority"][priority] = stats["by_priority"].get(priority, 0) + 1

            # Account counts
            account = email.account_id
            stats["by_account"][account] = stats["by_account"].get(account, 0) + 1

        assert stats["total"] == 5
        assert stats["by_priority"]["critical"] == 1
        assert stats["by_priority"]["high"] == 2
        assert stats["by_priority"]["medium"] == 1
        assert stats["by_priority"]["low"] == 1
        assert len(stats["by_account"]) == 3

    def test_action_required_count(self, sample_emails):
        """Test calculating action-required count."""
        action_priorities = {"critical", "high"}
        action_required = sum(1 for email in sample_emails if email.priority in action_priorities)
        assert action_required == 3  # 1 critical + 2 high


# =============================================================================
# Account Isolation Tests
# =============================================================================


class TestAccountIsolation:
    """Tests for ensuring proper account isolation."""

    @pytest.mark.asyncio
    async def test_token_isolation(self, token_store, sample_accounts):
        """Test that tokens are isolated between accounts."""
        for account in sample_accounts:
            await token_store.save(account)

        work = await token_store.get("user_work")
        personal = await token_store.get("user_personal")

        # Tokens should be different
        assert work.access_token != personal.access_token
        assert work.refresh_token != personal.refresh_token

    def test_email_account_attribution(self, sample_emails):
        """Test that emails maintain correct account attribution."""
        for email in sample_emails:
            # Each email should have exactly one account
            assert email.account_id is not None
            assert email.account_id in {"user_work", "user_personal", "user_shared"}

    def test_cross_account_deduplication(self):
        """Test deduplication when same email appears in multiple accounts."""
        # Create duplicate emails (same subject/sender, different accounts)
        shared_meeting = MockEmailMessage(
            id="meeting_work",
            account_id="user_work",
            subject="Team Meeting Invite",
            from_address="calendar@company.com",
        )

        shared_meeting_personal = MockEmailMessage(
            id="meeting_personal",
            account_id="user_personal",
            subject="Team Meeting Invite",
            from_address="calendar@company.com",
        )

        emails = [shared_meeting, shared_meeting_personal]

        # Deduplicate by subject + sender
        seen = set()
        deduplicated = []
        for email in emails:
            key = (email.subject, email.from_address)
            if key not in seen:
                seen.add(key)
                deduplicated.append(email)

        # Should keep only first occurrence
        assert len(deduplicated) == 1
        assert deduplicated[0].account_id == "user_work"


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestMultiAccountErrorHandling:
    """Tests for error handling in multi-account scenarios."""

    @pytest.mark.asyncio
    async def test_partial_account_failure(self, token_store, sample_accounts):
        """Test handling when one account fails but others succeed."""
        for account in sample_accounts:
            await token_store.save(account)

        # Simulate one account having invalid credentials
        work = await token_store.get("user_work")
        work.refresh_token = ""  # Invalid
        await token_store.save(work)

        # Should still be able to access other accounts
        personal = await token_store.get("user_personal")
        shared = await token_store.get("user_shared")

        assert personal.refresh_token != ""
        assert shared.refresh_token != ""

    @pytest.mark.asyncio
    async def test_missing_account_graceful_handling(self, token_store, sample_accounts):
        """Test graceful handling of missing account."""
        for account in sample_accounts:
            await token_store.save(account)

        # Request non-existent account
        missing = await token_store.get("nonexistent_account")
        assert missing is None

        # Other accounts should still work
        work = await token_store.get("user_work")
        assert work is not None
