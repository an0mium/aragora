"""
Tests for Email Prioritization Service.

Tests cover:
- 3-tier scoring system
- Priority levels (critical, high, medium, low, defer)
- Newsletter detection
- Urgency keyword detection
- VIP sender handling
- Sender profile management
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.services.email_prioritization import (
    EmailPrioritizer,
    EmailPrioritizationConfig,
    EmailPriority,
    EmailPriorityResult,
    ScoringTier,
    SenderProfile,
)


# =============================================================================
# Fixtures
# =============================================================================


class MockEmailMessage:
    """Mock email message for testing."""

    def __init__(
        self,
        id: str = "test_email_1",
        thread_id: str = "thread_1",
        subject: str = "Test Subject",
        from_address: str = "sender@example.com",
        to_addresses: list = None,
        body_text: str = "Test email body",
        snippet: str = "Test snippet",
        labels: list = None,
        is_read: bool = False,
        is_starred: bool = False,
        is_important: bool = False,
        date: datetime = None,
    ):
        self.id = id
        self.thread_id = thread_id
        self.subject = subject
        self.from_address = from_address
        self.to_addresses = to_addresses or ["recipient@example.com"]
        self.cc_addresses = []
        self.bcc_addresses = []
        self.body_text = body_text
        self.body_html = ""
        self.snippet = snippet
        self.labels = labels or ["INBOX"]
        self.headers = {}
        self.attachments = []
        self.is_read = is_read
        self.is_starred = is_starred
        self.is_important = is_important
        self.date = date or datetime.now()


@pytest.fixture
def config():
    """Default test configuration."""
    return EmailPrioritizationConfig(
        vip_domains={"important-company.com", "vip.org"},
        vip_addresses={"vip@test.com", "ceo@company.com"},
        internal_domains={"mycompany.com"},
        auto_archive_senders={"noreply@spam.com"},
    )


@pytest.fixture
def prioritizer(config):
    """Email prioritizer with test config."""
    return EmailPrioritizer(config=config)


# =============================================================================
# Email Priority Tests
# =============================================================================


class TestEmailPriority:
    """Test EmailPriority enum."""

    def test_priority_ordering(self):
        """Test that priorities are correctly ordered."""
        assert EmailPriority.CRITICAL.value > EmailPriority.HIGH.value
        assert EmailPriority.HIGH.value > EmailPriority.MEDIUM.value
        assert EmailPriority.MEDIUM.value > EmailPriority.LOW.value
        assert EmailPriority.LOW.value > EmailPriority.DEFER.value

    def test_priority_values(self):
        """Test priority numeric values."""
        assert EmailPriority.CRITICAL.value == 5
        assert EmailPriority.HIGH.value == 4
        assert EmailPriority.MEDIUM.value == 3
        assert EmailPriority.LOW.value == 2
        assert EmailPriority.DEFER.value == 1


# =============================================================================
# Scoring Tier Tests
# =============================================================================


class TestScoringTier:
    """Test scoring tier enum."""

    def test_tier_values(self):
        """Test tier numeric values."""
        assert ScoringTier.TIER_1_RULES.value == 1
        assert ScoringTier.TIER_2_LIGHTWEIGHT.value == 2
        assert ScoringTier.TIER_3_DEBATE.value == 3


# =============================================================================
# Email Prioritizer Core Tests
# =============================================================================


class TestEmailPrioritizerInit:
    """Test EmailPrioritizer initialization."""

    def test_default_init(self):
        """Test initialization with defaults."""
        prioritizer = EmailPrioritizer()
        assert prioritizer.config is not None
        assert prioritizer.gmail is None
        assert prioritizer.mound is None

    def test_init_with_config(self, config):
        """Test initialization with custom config."""
        prioritizer = EmailPrioritizer(config=config)
        assert prioritizer.config == config
        assert "important-company.com" in prioritizer.config.vip_domains

    def test_patterns_compiled(self, prioritizer):
        """Test that regex patterns are compiled."""
        assert len(prioritizer._newsletter_patterns) > 0
        assert len(prioritizer._urgent_patterns) > 0
        assert len(prioritizer._deadline_patterns) > 0


# =============================================================================
# Tier 1 Rule-Based Scoring Tests
# =============================================================================


class TestTier1RuleBasedScoring:
    """Test Tier 1 rule-based scoring."""

    @pytest.mark.asyncio
    async def test_vip_sender_critical_priority(self, prioritizer):
        """VIP sender emails should be critical priority."""
        email = MockEmailMessage(
            from_address="vip@test.com",
            subject="Meeting tomorrow",
        )

        result = await prioritizer.score_email(email, force_tier=ScoringTier.TIER_1_RULES)

        assert result.priority == EmailPriority.CRITICAL
        assert result.tier_used == ScoringTier.TIER_1_RULES
        assert "VIP" in result.rationale or "vip" in result.rationale.lower()

    @pytest.mark.asyncio
    async def test_vip_domain_critical_priority(self, prioritizer):
        """Emails from VIP domains should be critical priority."""
        email = MockEmailMessage(
            from_address="anyone@important-company.com",
            subject="Important update",
        )

        result = await prioritizer.score_email(email, force_tier=ScoringTier.TIER_1_RULES)

        assert result.priority == EmailPriority.CRITICAL

    @pytest.mark.asyncio
    async def test_newsletter_deferred(self, prioritizer):
        """Newsletter emails should be deferred."""
        email = MockEmailMessage(
            from_address="news@newsletter.com",
            subject="Weekly Newsletter - Your weekly update",
            body_text="Click here to unsubscribe from this newsletter.",
        )

        result = await prioritizer.score_email(email, force_tier=ScoringTier.TIER_1_RULES)

        assert result.priority == EmailPriority.DEFER

    @pytest.mark.asyncio
    async def test_urgent_keywords_high_priority(self, prioritizer):
        """Emails with urgent keywords should be high priority."""
        email = MockEmailMessage(
            subject="URGENT: Need response ASAP",
            body_text="This is critical and needs immediate attention.",
        )

        result = await prioritizer.score_email(email, force_tier=ScoringTier.TIER_1_RULES)

        assert result.priority in [EmailPriority.CRITICAL, EmailPriority.HIGH]
        assert result.confidence >= 0.6

    @pytest.mark.asyncio
    async def test_deadline_detection(self, prioritizer):
        """Emails with deadlines should be detected."""
        email = MockEmailMessage(
            subject="Report due by Friday",
            body_text="Please submit the report by end of day Friday.",
        )

        result = await prioritizer.score_email(email, force_tier=ScoringTier.TIER_1_RULES)

        assert result.priority in [EmailPriority.HIGH, EmailPriority.MEDIUM]
        assert "deadline" in result.rationale.lower() or "due" in result.rationale.lower()

    @pytest.mark.asyncio
    async def test_internal_domain_medium_priority(self, prioritizer):
        """Internal domain emails should get medium priority by default."""
        email = MockEmailMessage(
            from_address="colleague@mycompany.com",
            subject="Team update",
        )

        result = await prioritizer.score_email(email, force_tier=ScoringTier.TIER_1_RULES)

        assert result.priority in [EmailPriority.MEDIUM, EmailPriority.HIGH]

    @pytest.mark.asyncio
    async def test_regular_email_medium_priority(self, prioritizer):
        """Regular emails should get medium priority."""
        email = MockEmailMessage(
            from_address="random@external.com",
            subject="Hello",
            body_text="Just saying hi.",
        )

        result = await prioritizer.score_email(email, force_tier=ScoringTier.TIER_1_RULES)

        assert result.priority in [EmailPriority.MEDIUM, EmailPriority.LOW]

    @pytest.mark.asyncio
    async def test_auto_archive_sender_deferred(self, prioritizer):
        """Auto-archive senders should be deferred."""
        email = MockEmailMessage(
            from_address="noreply@spam.com",
            subject="Your receipt",
        )

        result = await prioritizer.score_email(email, force_tier=ScoringTier.TIER_1_RULES)

        assert result.priority == EmailPriority.DEFER


# =============================================================================
# Newsletter Detection Tests
# =============================================================================


class TestNewsletterDetection:
    """Test newsletter detection patterns."""

    @pytest.mark.asyncio
    async def test_unsubscribe_link_detected(self, prioritizer):
        """Emails with unsubscribe links should be marked as newsletters."""
        email = MockEmailMessage(
            body_text="To unsubscribe, click here: https://example.com/unsubscribe",
        )

        result = await prioritizer.score_email(email, force_tier=ScoringTier.TIER_1_RULES)
        assert result.priority == EmailPriority.DEFER

    @pytest.mark.asyncio
    async def test_weekly_update_detected(self, prioritizer):
        """Weekly updates should be detected as newsletters."""
        email = MockEmailMessage(
            subject="Your Weekly Update - Week 42",
        )

        result = await prioritizer.score_email(email, force_tier=ScoringTier.TIER_1_RULES)
        assert result.priority == EmailPriority.DEFER

    @pytest.mark.asyncio
    async def test_noreply_sender_newsletter(self, prioritizer):
        """No-reply senders often indicate newsletters."""
        email = MockEmailMessage(
            from_address="noreply@marketing.com",
            subject="New products available",
        )

        result = await prioritizer.score_email(email, force_tier=ScoringTier.TIER_1_RULES)
        assert result.priority in [EmailPriority.DEFER, EmailPriority.LOW]


# =============================================================================
# Inbox Ranking Tests
# =============================================================================


class TestInboxRanking:
    """Test inbox ranking functionality."""

    @pytest.mark.asyncio
    async def test_rank_inbox_orders_by_priority(self, prioritizer):
        """rank_inbox should order emails by priority."""
        emails = [
            MockEmailMessage(
                id="low",
                from_address="random@example.com",
                subject="Hello",
            ),
            MockEmailMessage(
                id="critical",
                from_address="vip@test.com",
                subject="Important",
            ),
            MockEmailMessage(
                id="defer",
                from_address="news@newsletter.com",
                subject="Weekly Newsletter",
                body_text="Click to unsubscribe",
            ),
        ]

        results = await prioritizer.rank_inbox(emails)

        # VIP should be first
        assert results[0].email_id == "critical"
        # Newsletter should be last
        assert results[-1].email_id == "defer"

    @pytest.mark.asyncio
    async def test_rank_inbox_respects_limit(self, prioritizer):
        """rank_inbox should respect the limit parameter."""
        emails = [
            MockEmailMessage(id=f"email_{i}")
            for i in range(10)
        ]

        results = await prioritizer.rank_inbox(emails, limit=5)

        assert len(results) == 5


# =============================================================================
# Sender Profile Tests
# =============================================================================


class TestSenderProfile:
    """Test sender profile management."""

    @pytest.mark.asyncio
    async def test_get_sender_profile_creates_new(self, prioritizer):
        """Getting a new sender profile should create it."""
        profile = await prioritizer._get_sender_profile("new@example.com")

        assert profile is not None
        assert profile.email == "new@example.com"
        assert profile.response_rate == 0.0
        assert profile.total_emails_received == 0

    @pytest.mark.asyncio
    async def test_get_sender_profile_caches(self, prioritizer):
        """Sender profiles should be cached."""
        profile1 = await prioritizer._get_sender_profile("cached@example.com")
        profile1.total_emails_received = 5

        profile2 = await prioritizer._get_sender_profile("cached@example.com")

        assert profile1 is profile2
        assert profile2.total_emails_received == 5


# =============================================================================
# User Action Recording Tests
# =============================================================================


class TestUserActionRecording:
    """Test user action recording for learning."""

    @pytest.mark.asyncio
    async def test_record_user_action_updates_profile(self, prioritizer):
        """Recording a user action should update the sender profile."""
        email = MockEmailMessage(from_address="sender@example.com")

        # Pre-populate profile
        profile = await prioritizer._get_sender_profile("sender@example.com")
        initial_count = profile.total_emails_received

        await prioritizer.record_user_action("email_1", "replied", email)

        assert profile.total_emails_received == initial_count + 1
        assert profile.total_emails_responded == 1

    @pytest.mark.asyncio
    async def test_record_replied_updates_response_rate(self, prioritizer):
        """Replying should update response rate."""
        email = MockEmailMessage(from_address="test@example.com")

        await prioritizer.record_user_action("email_1", "replied", email)

        profile = await prioritizer._get_sender_profile("test@example.com")
        assert profile.response_rate == 1.0  # 1/1 = 100%


# =============================================================================
# Configuration Tests
# =============================================================================


class TestEmailPrioritizationConfig:
    """Test EmailPrioritizationConfig."""

    def test_default_config_values(self):
        """Test default configuration values."""
        config = EmailPrioritizationConfig()

        assert config.tier_1_confidence_threshold == 0.8
        assert config.tier_2_confidence_threshold == 0.5
        assert len(config.urgent_keywords) > 0
        assert "urgent" in config.urgent_keywords
        assert config.enable_slack_signals is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = EmailPrioritizationConfig(
            vip_addresses={"special@vip.com"},
            tier_1_confidence_threshold=0.9,
            enable_slack_signals=False,
        )

        assert "special@vip.com" in config.vip_addresses
        assert config.tier_1_confidence_threshold == 0.9
        assert config.enable_slack_signals is False


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_empty_subject(self, prioritizer):
        """Handle emails with empty subjects."""
        email = MockEmailMessage(subject="", body_text="Some content")

        result = await prioritizer.score_email(email, force_tier=ScoringTier.TIER_1_RULES)

        assert result is not None
        assert result.priority in EmailPriority

    @pytest.mark.asyncio
    async def test_empty_body(self, prioritizer):
        """Handle emails with empty bodies."""
        email = MockEmailMessage(subject="Test", body_text="")

        result = await prioritizer.score_email(email, force_tier=ScoringTier.TIER_1_RULES)

        assert result is not None

    @pytest.mark.asyncio
    async def test_very_long_subject(self, prioritizer):
        """Handle emails with very long subjects."""
        email = MockEmailMessage(subject="A" * 1000)

        result = await prioritizer.score_email(email, force_tier=ScoringTier.TIER_1_RULES)

        assert result is not None

    @pytest.mark.asyncio
    async def test_unicode_content(self, prioritizer):
        """Handle emails with unicode content."""
        email = MockEmailMessage(
            subject="Urgent: 紧急 🚨",
            body_text="Hello 你好 مرحبا",
        )

        result = await prioritizer.score_email(email, force_tier=ScoringTier.TIER_1_RULES)

        assert result is not None
        # "Urgent" keyword should still be detected
        assert result.priority in [EmailPriority.CRITICAL, EmailPriority.HIGH]

    @pytest.mark.asyncio
    async def test_empty_inbox_ranking(self, prioritizer):
        """Handle empty inbox ranking."""
        results = await prioritizer.rank_inbox([])

        assert results == []


# =============================================================================
# Result Dataclass Tests
# =============================================================================


class TestEmailPriorityResult:
    """Test EmailPriorityResult dataclass."""

    def test_to_dict(self):
        """Test to_dict conversion."""
        result = EmailPriorityResult(
            email_id="test_123",
            priority=EmailPriority.HIGH,
            confidence=0.85,
            score=0.75,
            tier_used=ScoringTier.TIER_1_RULES,
            rationale="VIP sender detected",
        )

        d = result.to_dict()

        assert d["email_id"] == "test_123"
        assert d["priority"] == "high"
        assert d["confidence"] == 0.85
        assert d["score"] == 0.75
        assert d["tier_used"] == 1
        assert d["rationale"] == "VIP sender detected"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
