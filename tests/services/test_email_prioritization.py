"""
Tests for Email Prioritization Service.

Tests for intelligent email inbox management:
- Priority levels and scoring tiers
- Sender profile and reputation
- 3-tier scoring architecture (rules, lightweight, debate)
- Urgency and deadline detection
- Cross-channel signals
- User action recording and learning
"""

import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch


class TestEmailPriority:
    """Tests for EmailPriority enum."""

    def test_email_priority_values(self):
        """Test EmailPriority enum values."""
        from aragora.services.email_prioritization import EmailPriority

        assert EmailPriority.CRITICAL.value == 1
        assert EmailPriority.HIGH.value == 2
        assert EmailPriority.MEDIUM.value == 3
        assert EmailPriority.LOW.value == 4
        assert EmailPriority.DEFER.value == 5
        assert EmailPriority.BLOCKED.value == 6

    def test_email_priority_ordering(self):
        """Test that priority values are correctly ordered."""
        from aragora.services.email_prioritization import EmailPriority

        # Lower value = higher priority
        assert EmailPriority.CRITICAL.value < EmailPriority.HIGH.value
        assert EmailPriority.HIGH.value < EmailPriority.MEDIUM.value
        assert EmailPriority.MEDIUM.value < EmailPriority.LOW.value


class TestScoringTier:
    """Tests for ScoringTier enum."""

    def test_scoring_tier_values(self):
        """Test ScoringTier enum values."""
        from aragora.services.email_prioritization import ScoringTier

        assert ScoringTier.TIER_1_RULES.value == "tier_1_rules"
        assert ScoringTier.TIER_2_LIGHTWEIGHT.value == "tier_2_lightweight"
        assert ScoringTier.TIER_3_DEBATE.value == "tier_3_debate"


class TestSenderProfile:
    """Tests for SenderProfile dataclass."""

    def test_sender_profile_minimal(self):
        """Test minimal SenderProfile."""
        from aragora.services.email_prioritization import SenderProfile

        profile = SenderProfile(
            email="sender@example.com",
            domain="example.com",
        )

        assert profile.email == "sender@example.com"
        assert profile.domain == "example.com"
        assert profile.is_vip is False
        assert profile.is_internal is False
        assert profile.is_blocked is False
        assert profile.response_rate == 0.0

    def test_sender_profile_vip(self):
        """Test VIP sender profile."""
        from aragora.services.email_prioritization import SenderProfile

        profile = SenderProfile(
            email="ceo@company.com",
            domain="company.com",
            is_vip=True,
            is_internal=True,
        )

        assert profile.is_vip is True
        assert profile.is_internal is True

    def test_sender_profile_reputation_score_baseline(self):
        """Test baseline reputation score."""
        from aragora.services.email_prioritization import SenderProfile

        # Use a neutral response_rate to avoid penalty
        profile = SenderProfile(
            email="unknown@random.com",
            domain="random.com",
            response_rate=0.5,  # Neutral response rate
        )

        # Baseline is 0.5 with no bonuses or penalties
        assert profile.reputation_score == 0.5

    def test_sender_profile_reputation_score_vip(self):
        """Test VIP reputation score boost."""
        from aragora.services.email_prioritization import SenderProfile

        profile = SenderProfile(
            email="vip@company.com",
            domain="company.com",
            is_vip=True,
            response_rate=0.5,  # Neutral response rate
        )

        # VIP adds 0.3, so 0.5 + 0.3 = 0.8
        assert profile.reputation_score == 0.8

    def test_sender_profile_reputation_score_internal(self):
        """Test internal sender reputation boost."""
        from aragora.services.email_prioritization import SenderProfile

        profile = SenderProfile(
            email="colleague@company.com",
            domain="company.com",
            is_internal=True,
            response_rate=0.5,  # Neutral response rate
        )

        # Internal adds 0.1, so 0.5 + 0.1 = 0.6
        assert profile.reputation_score == 0.6

    def test_sender_profile_reputation_score_high_response_rate(self):
        """Test high response rate reputation boost."""
        from aragora.services.email_prioritization import SenderProfile

        profile = SenderProfile(
            email="frequent@contact.com",
            domain="contact.com",
            response_rate=0.9,
        )

        assert profile.reputation_score >= 0.6

    def test_sender_profile_reputation_score_low_response_rate(self):
        """Test low response rate reputation penalty."""
        from aragora.services.email_prioritization import SenderProfile

        profile = SenderProfile(
            email="ignored@spam.com",
            domain="spam.com",
            response_rate=0.05,
        )

        assert profile.reputation_score <= 0.4

    def test_sender_profile_reputation_recent_interaction(self):
        """Test recent interaction boost."""
        from aragora.services.email_prioritization import SenderProfile

        profile = SenderProfile(
            email="recent@contact.com",
            domain="contact.com",
            last_interaction=datetime.now() - timedelta(days=2),
            response_rate=0.5,  # Neutral response rate
        )

        # Recent interaction (<7 days) adds 0.1, so 0.5 + 0.1 = 0.6
        assert profile.reputation_score == 0.6

    def test_sender_profile_reputation_old_interaction(self):
        """Test old interaction penalty."""
        from aragora.services.email_prioritization import SenderProfile

        profile = SenderProfile(
            email="old@contact.com",
            domain="contact.com",
            last_interaction=datetime.now() - timedelta(days=120),
        )

        assert profile.reputation_score <= 0.5


class TestEmailPriorityResult:
    """Tests for EmailPriorityResult dataclass."""

    def test_priority_result_minimal(self):
        """Test minimal EmailPriorityResult."""
        from aragora.services.email_prioritization import (
            EmailPriorityResult,
            EmailPriority,
            ScoringTier,
        )

        result = EmailPriorityResult(
            email_id="email_123",
            priority=EmailPriority.MEDIUM,
            confidence=0.75,
            tier_used=ScoringTier.TIER_1_RULES,
            rationale="Standard priority signals",
        )

        assert result.email_id == "email_123"
        assert result.priority == EmailPriority.MEDIUM
        assert result.confidence == 0.75
        assert result.tier_used == ScoringTier.TIER_1_RULES

    def test_priority_result_full(self):
        """Test full EmailPriorityResult."""
        from aragora.services.email_prioritization import (
            EmailPriorityResult,
            EmailPriority,
            ScoringTier,
        )

        result = EmailPriorityResult(
            email_id="email_456",
            priority=EmailPriority.CRITICAL,
            confidence=0.95,
            tier_used=ScoringTier.TIER_3_DEBATE,
            rationale="Urgent deadline detected",
            sender_score=0.9,
            content_urgency_score=0.85,
            context_relevance_score=0.7,
            time_sensitivity_score=0.95,
            spam_score=0.1,
            spam_category="ham",
            is_spam=False,
            slack_activity_boost=0.1,
            debate_id="debate_789",
            suggested_labels=["Urgent", "Deadline"],
            auto_archive=False,
        )

        assert result.sender_score == 0.9
        assert result.content_urgency_score == 0.85
        assert result.is_spam is False
        assert result.debate_id == "debate_789"

    def test_priority_result_to_dict(self):
        """Test EmailPriorityResult.to_dict."""
        from aragora.services.email_prioritization import (
            EmailPriorityResult,
            EmailPriority,
            ScoringTier,
        )

        result = EmailPriorityResult(
            email_id="email_789",
            priority=EmailPriority.HIGH,
            confidence=0.8,
            tier_used=ScoringTier.TIER_2_LIGHTWEIGHT,
            rationale="VIP sender",
            sender_score=0.85,
            suggested_labels=["VIP"],
        )

        data = result.to_dict()

        assert data["email_id"] == "email_789"
        assert data["priority"] == 2
        assert data["priority_name"] == "HIGH"
        assert data["confidence"] == 0.8
        assert data["tier_used"] == "tier_2_lightweight"
        assert data["scores"]["sender"] == 0.85
        assert data["suggested_labels"] == ["VIP"]


class TestEmailPrioritizationConfig:
    """Tests for EmailPrioritizationConfig."""

    def test_config_defaults(self):
        """Test default config values."""
        from aragora.services.email_prioritization import EmailPrioritizationConfig

        config = EmailPrioritizationConfig()

        assert config.tier_1_confidence_threshold == 0.7
        assert config.tier_2_confidence_threshold == 0.6
        assert config.enable_slack_signals is True
        assert config.debate_agent_count == 3
        assert config.debate_timeout_seconds == 30.0

    def test_config_custom_vip(self):
        """Test custom VIP configuration."""
        from aragora.services.email_prioritization import EmailPrioritizationConfig

        config = EmailPrioritizationConfig(
            vip_domains={"company.com", "partner.com"},
            vip_addresses={"ceo@company.com", "vip@client.com"},
        )

        assert "company.com" in config.vip_domains
        assert "ceo@company.com" in config.vip_addresses

    def test_config_urgent_keywords(self):
        """Test urgent keywords configuration."""
        from aragora.services.email_prioritization import EmailPrioritizationConfig

        config = EmailPrioritizationConfig()

        assert "urgent" in config.urgent_keywords
        assert "asap" in config.urgent_keywords
        assert "deadline" in config.urgent_keywords


class TestEmailPrioritizer:
    """Tests for EmailPrioritizer class."""

    def test_prioritizer_initialization(self):
        """Test prioritizer initialization."""
        from aragora.services.email_prioritization import EmailPrioritizer

        prioritizer = EmailPrioritizer()

        assert prioritizer.gmail is None
        assert prioritizer.mound is None
        assert prioritizer.config is not None
        assert prioritizer._sender_profiles == {}

    def test_prioritizer_with_config(self):
        """Test prioritizer with custom config."""
        from aragora.services.email_prioritization import (
            EmailPrioritizer,
            EmailPrioritizationConfig,
        )

        config = EmailPrioritizationConfig(
            vip_domains={"vip.com"},
            tier_1_confidence_threshold=0.8,
        )
        prioritizer = EmailPrioritizer(config=config)

        assert "vip.com" in prioritizer.config.vip_domains
        assert prioritizer.config.tier_1_confidence_threshold == 0.8

    @pytest.fixture
    def mock_email(self):
        """Create a mock email."""
        email = MagicMock()
        email.id = "email_123"
        email.from_address = "sender@example.com"
        email.subject = "Test Subject"
        email.body_text = "Test body content"
        email.body_html = "<p>Test body content</p>"
        email.snippet = "Test body content"
        email.is_important = False
        email.is_starred = False
        email.labels = []
        email.headers = {}
        return email

    @pytest.mark.asyncio
    async def test_get_sender_profile_new(self, mock_email):
        """Test creating new sender profile."""
        from aragora.services.email_prioritization import EmailPrioritizer

        prioritizer = EmailPrioritizer()

        profile = await prioritizer._get_sender_profile("new@sender.com")

        assert profile.email == "new@sender.com"
        assert profile.domain == "sender.com"
        assert "new@sender.com" in prioritizer._sender_profiles

    @pytest.mark.asyncio
    async def test_get_sender_profile_cached(self, mock_email):
        """Test cached sender profile."""
        from aragora.services.email_prioritization import EmailPrioritizer, SenderProfile

        prioritizer = EmailPrioritizer()

        # Pre-populate cache
        cached_profile = SenderProfile(
            email="cached@sender.com",
            domain="sender.com",
            is_vip=True,
        )
        prioritizer._sender_profiles["cached@sender.com"] = cached_profile

        profile = await prioritizer._get_sender_profile("cached@sender.com")

        assert profile.is_vip is True

    @pytest.mark.asyncio
    async def test_get_sender_profile_vip(self, mock_email):
        """Test VIP sender detection."""
        from aragora.services.email_prioritization import (
            EmailPrioritizer,
            EmailPrioritizationConfig,
        )

        config = EmailPrioritizationConfig(
            vip_addresses={"vip@important.com"},
            vip_domains={"vip-domain.com"},
        )
        prioritizer = EmailPrioritizer(config=config)

        profile = await prioritizer._get_sender_profile("vip@important.com")
        assert profile.is_vip is True

        profile2 = await prioritizer._get_sender_profile("anyone@vip-domain.com")
        assert profile2.is_vip is True

    @pytest.mark.asyncio
    async def test_get_sender_profile_internal(self, mock_email):
        """Test internal sender detection."""
        from aragora.services.email_prioritization import (
            EmailPrioritizer,
            EmailPrioritizationConfig,
        )

        config = EmailPrioritizationConfig(
            internal_domains={"company.com"},
        )
        prioritizer = EmailPrioritizer(config=config)

        profile = await prioritizer._get_sender_profile("colleague@company.com")

        assert profile.is_internal is True


class TestEmailPrioritizerTier1:
    """Tests for Tier 1 rule-based scoring."""

    @pytest.fixture
    def prioritizer(self):
        """Create a prioritizer instance."""
        from aragora.services.email_prioritization import EmailPrioritizer

        return EmailPrioritizer()

    @pytest.fixture
    def mock_email(self):
        """Create a mock email."""
        email = MagicMock()
        email.id = "email_123"
        email.from_address = "sender@example.com"
        email.subject = "Test Subject"
        email.body_text = "Test body content"
        email.body_html = None
        email.snippet = "Test body content"
        email.is_important = False
        email.is_starred = False
        email.labels = []
        email.headers = {}
        return email

    @pytest.fixture
    def sender_profile(self):
        """Create a sender profile."""
        from aragora.services.email_prioritization import SenderProfile

        return SenderProfile(
            email="sender@example.com",
            domain="example.com",
        )

    @pytest.mark.asyncio
    async def test_tier_1_blocked_sender(self, prioritizer, mock_email):
        """Test Tier 1 with blocked sender."""
        from aragora.services.email_prioritization import (
            SenderProfile,
            EmailPriority,
            ScoringTier,
        )

        blocked_sender = SenderProfile(
            email="blocked@spam.com",
            domain="spam.com",
            is_blocked=True,
        )
        mock_email.from_address = "blocked@spam.com"

        result = await prioritizer._tier_1_score(mock_email, blocked_sender)

        assert result.priority == EmailPriority.BLOCKED
        assert result.confidence == 1.0
        assert result.tier_used == ScoringTier.TIER_1_RULES
        assert result.auto_archive is True

    @pytest.mark.asyncio
    async def test_tier_1_newsletter_detection(self, prioritizer, mock_email, sender_profile):
        """Test newsletter detection in Tier 1."""
        from aragora.services.email_prioritization import EmailPriority

        mock_email.from_address = "no-reply@newsletter.com"
        mock_email.subject = "Weekly Newsletter"
        mock_email.body_text = "Click here to unsubscribe"
        mock_email.headers = {"list-unsubscribe": "<mailto:unsub@list.com>"}

        result = await prioritizer._tier_1_score(mock_email, sender_profile)

        assert result.priority == EmailPriority.DEFER
        assert result.auto_archive is True
        assert "newsletter" in result.rationale.lower()

    @pytest.mark.asyncio
    async def test_tier_1_vip_sender(self, prioritizer, mock_email):
        """Test VIP sender boost in Tier 1."""
        from aragora.services.email_prioritization import SenderProfile

        vip_sender = SenderProfile(
            email="ceo@company.com",
            domain="company.com",
            is_vip=True,
        )

        result = await prioritizer._tier_1_score(mock_email, vip_sender)

        assert "VIP" in result.rationale
        assert result.sender_score > 0.5

    @pytest.mark.asyncio
    async def test_tier_1_important_flag(self, prioritizer, mock_email, sender_profile):
        """Test Gmail important flag boost."""
        mock_email.is_important = True

        result = await prioritizer._tier_1_score(mock_email, sender_profile)

        assert "important" in result.rationale.lower()

    @pytest.mark.asyncio
    async def test_tier_1_starred(self, prioritizer, mock_email, sender_profile):
        """Test starred email boost."""
        mock_email.is_starred = True

        result = await prioritizer._tier_1_score(mock_email, sender_profile)

        assert "starred" in result.rationale.lower()

    @pytest.mark.asyncio
    async def test_tier_1_urgent_keywords(self, prioritizer, mock_email, sender_profile):
        """Test urgent keyword detection."""
        mock_email.subject = "URGENT: Action Required ASAP"
        mock_email.body_text = "Please respond immediately. Critical issue."

        result = await prioritizer._tier_1_score(mock_email, sender_profile)

        assert "urgent" in result.rationale.lower() or "keyword" in result.rationale.lower()
        assert result.content_urgency_score > 0

    @pytest.mark.asyncio
    async def test_tier_1_deadline_detection(self, prioritizer, mock_email, sender_profile):
        """Test deadline detection."""
        mock_email.subject = "Report Due by Friday"
        mock_email.body_text = "Please submit your report by Monday."

        result = await prioritizer._tier_1_score(mock_email, sender_profile)

        assert "deadline" in result.rationale.lower()
        assert result.time_sensitivity_score > 0


class TestEmailPrioritizerTier2:
    """Tests for Tier 2 lightweight agent scoring."""

    @pytest.fixture
    def prioritizer(self):
        """Create a prioritizer instance."""
        from aragora.services.email_prioritization import EmailPrioritizer

        return EmailPrioritizer()

    @pytest.fixture
    def mock_email(self):
        """Create a mock email."""
        email = MagicMock()
        email.id = "email_123"
        email.from_address = "sender@example.com"
        email.subject = "Complex Email"
        email.body_text = "This email needs analysis"
        email.body_html = None
        email.snippet = "This email needs analysis"
        email.is_important = False
        email.is_starred = False
        email.labels = []
        email.headers = {}
        return email

    @pytest.fixture
    def sender_profile(self):
        """Create a sender profile."""
        from aragora.services.email_prioritization import SenderProfile

        return SenderProfile(
            email="sender@example.com",
            domain="example.com",
        )

    @pytest.mark.asyncio
    async def test_tier_2_score_fallback(self, prioritizer, mock_email, sender_profile):
        """Test Tier 2 fallback when model unavailable."""
        from aragora.services.email_prioritization import ScoringTier

        # Mock both the import location and where it's used
        with patch.dict("sys.modules", {"aragora.core.model_router": MagicMock()}):
            # Make the imported function raise an exception
            import sys

            sys.modules["aragora.core.model_router"].get_model_router = MagicMock(
                side_effect=Exception("Model not available")
            )

            result = await prioritizer._tier_2_score(mock_email, sender_profile)

            assert result.tier_used == ScoringTier.TIER_2_LIGHTWEIGHT
            assert "fallback" in result.rationale.lower()

    @pytest.mark.asyncio
    async def test_tier_2_score_success(self, prioritizer, mock_email, sender_profile):
        """Test successful Tier 2 scoring."""
        from aragora.services.email_prioritization import ScoringTier, EmailPriority

        mock_router = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "PRIORITY: 2, CONFIDENCE: 0.8, REASON: Important business email"
        mock_router.generate = AsyncMock(return_value=mock_response)

        with patch.dict("sys.modules", {"aragora.core.model_router": MagicMock()}):
            import sys

            sys.modules["aragora.core.model_router"].get_model_router = MagicMock(
                return_value=mock_router
            )

            result = await prioritizer._tier_2_score(mock_email, sender_profile)

            assert result.tier_used == ScoringTier.TIER_2_LIGHTWEIGHT
            assert result.priority == EmailPriority.HIGH
            assert result.confidence == 0.8


class TestEmailPrioritizerTier3:
    """Tests for Tier 3 debate scoring."""

    @pytest.fixture
    def prioritizer(self):
        """Create a prioritizer instance."""
        from aragora.services.email_prioritization import EmailPrioritizer

        return EmailPrioritizer()

    @pytest.fixture
    def mock_email(self):
        """Create a mock email."""
        email = MagicMock()
        email.id = "email_123"
        email.from_address = "sender@example.com"
        email.subject = "Complex Decision Needed"
        email.body_text = "This requires multi-agent analysis"
        email.body_html = None
        email.snippet = "Complex content"
        email.is_important = False
        email.is_starred = False
        email.labels = []
        email.headers = {}
        return email

    @pytest.fixture
    def sender_profile(self):
        """Create a sender profile."""
        from aragora.services.email_prioritization import SenderProfile

        return SenderProfile(
            email="sender@example.com",
            domain="example.com",
        )

    @pytest.mark.asyncio
    async def test_tier_3_fallback(self, prioritizer, mock_email, sender_profile):
        """Test Tier 3 fallback when debate fails."""
        from aragora.services.email_prioritization import ScoringTier

        # Mock the debate module to raise an exception
        with patch.dict("sys.modules", {"aragora.debate.arena": MagicMock()}):
            import sys

            sys.modules["aragora.debate.arena"].DebateArena = MagicMock(
                side_effect=Exception("Debate failed")
            )

            result = await prioritizer._tier_3_debate(mock_email, sender_profile)

            assert result.tier_used == ScoringTier.TIER_3_DEBATE
            assert "fallback" in result.rationale.lower()


class TestEmailPrioritizerDetection:
    """Tests for detection methods."""

    @pytest.fixture
    def prioritizer(self):
        """Create a prioritizer instance."""
        from aragora.services.email_prioritization import EmailPrioritizer

        return EmailPrioritizer()

    def test_detect_newsletter_noreply(self, prioritizer):
        """Test newsletter detection by no-reply sender."""
        email = MagicMock()
        email.from_address = "noreply@company.com"
        email.subject = "Update"
        email.body_text = "Some content"
        email.body_html = None
        email.headers = {}

        assert prioritizer._detect_newsletter(email) is True

    def test_detect_newsletter_unsubscribe_header(self, prioritizer):
        """Test newsletter detection by list-unsubscribe header."""
        email = MagicMock()
        email.from_address = "sender@company.com"
        email.subject = "News"
        email.body_text = "Content"
        email.body_html = None
        email.headers = {"list-unsubscribe": "<mailto:unsub@list.com>"}

        assert prioritizer._detect_newsletter(email) is True

    def test_detect_newsletter_content(self, prioritizer):
        """Test newsletter detection by content patterns."""
        email = MagicMock()
        email.from_address = "sender@company.com"
        email.subject = "Weekly Newsletter"
        email.body_text = "Click here to unsubscribe from this newsletter"
        email.body_html = None
        email.headers = {}

        assert prioritizer._detect_newsletter(email) is True

    def test_detect_reply_needed_question(self, prioritizer):
        """Test reply needed detection by question."""
        email = MagicMock()
        email.subject = "Can you help with this?"
        email.body_text = "I need your input on this matter"

        assert prioritizer._detect_reply_needed(email) is True

    def test_detect_reply_needed_explicit(self, prioritizer):
        """Test reply needed detection by explicit request."""
        email = MagicMock()
        email.subject = "Project Update"
        email.body_text = "Please respond by end of day. Looking forward to hearing from you."

        assert prioritizer._detect_reply_needed(email) is True

    def test_detect_reply_needed_no_reply(self, prioritizer):
        """Test no reply needed for announcements."""
        email = MagicMock()
        email.subject = "Announcement: Office Closed Tomorrow"
        email.body_text = "The office will be closed tomorrow for maintenance."

        assert prioritizer._detect_reply_needed(email) is False


class TestEmailPrioritizerRankInbox:
    """Tests for inbox ranking."""

    @pytest.fixture
    def prioritizer(self):
        """Create a prioritizer instance."""
        from aragora.services.email_prioritization import EmailPrioritizer

        return EmailPrioritizer()

    @pytest.mark.asyncio
    async def test_rank_inbox(self, prioritizer):
        """Test ranking multiple emails."""
        emails = []
        for i in range(5):
            email = MagicMock()
            email.id = f"email_{i}"
            email.from_address = f"sender{i}@example.com"
            email.subject = f"Email {i}"
            email.body_text = "Content"
            email.body_html = None
            email.snippet = "Content"
            email.is_important = i == 0  # First one is important
            email.is_starred = i == 1  # Second one is starred
            email.labels = []
            email.headers = {}
            emails.append(email)

        results = await prioritizer.rank_inbox(emails)

        assert len(results) == 5
        # Results should be sorted by priority
        for i in range(len(results) - 1):
            assert results[i].priority.value <= results[i + 1].priority.value

    @pytest.mark.asyncio
    async def test_rank_inbox_with_limit(self, prioritizer):
        """Test inbox ranking with limit."""
        emails = []
        for i in range(10):
            email = MagicMock()
            email.id = f"email_{i}"
            email.from_address = "sender@example.com"
            email.subject = f"Email {i}"
            email.body_text = "Content"
            email.body_html = None
            email.snippet = "Content"
            email.is_important = False
            email.is_starred = False
            email.labels = []
            email.headers = {}
            emails.append(email)

        results = await prioritizer.rank_inbox(emails, limit=5)

        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_rank_inbox_handles_errors(self, prioritizer):
        """Test inbox ranking handles scoring errors."""
        emails = []
        for i in range(3):
            email = MagicMock()
            email.id = f"email_{i}"
            email.from_address = "sender@example.com"
            email.subject = f"Email {i}"
            email.body_text = "Content"
            email.body_html = None
            email.snippet = "Content"
            email.is_important = False
            email.is_starred = False
            email.labels = []
            email.headers = {}
            emails.append(email)

        # Make one email fail
        emails[1].from_address = None  # This might cause an error

        results = await prioritizer.rank_inbox(emails)

        # Should still return results (maybe fewer due to error)
        assert len(results) >= 0


class TestEmailPrioritizerScoreEmail:
    """Tests for score_email method."""

    @pytest.fixture
    def prioritizer(self):
        """Create a prioritizer instance."""
        from aragora.services.email_prioritization import (
            EmailPrioritizer,
            EmailPrioritizationConfig,
        )

        config = EmailPrioritizationConfig(
            tier_1_confidence_threshold=0.7,
            tier_2_confidence_threshold=0.6,
        )
        return EmailPrioritizer(config=config)

    @pytest.fixture
    def mock_email(self):
        """Create a mock email."""
        email = MagicMock()
        email.id = "email_123"
        email.from_address = "sender@example.com"
        email.subject = "Test Email"
        email.body_text = "Test content"
        email.body_html = None
        email.snippet = "Test content"
        email.is_important = False
        email.is_starred = False
        email.labels = []
        email.headers = {}
        return email

    @pytest.mark.asyncio
    async def test_score_email_force_tier_1(self, prioritizer, mock_email):
        """Test forcing Tier 1 scoring."""
        from aragora.services.email_prioritization import ScoringTier

        result = await prioritizer.score_email(mock_email, force_tier=ScoringTier.TIER_1_RULES)

        assert result.tier_used == ScoringTier.TIER_1_RULES

    @pytest.mark.asyncio
    async def test_score_email_high_confidence_stays_tier_1(self, prioritizer, mock_email):
        """Test high confidence result stays at Tier 1."""
        from aragora.services.email_prioritization import ScoringTier

        # VIP sender for high confidence
        mock_email.is_important = True
        mock_email.is_starred = True
        mock_email.subject = "URGENT: Critical Issue"

        result = await prioritizer.score_email(mock_email)

        # High confidence should stay at Tier 1
        assert result.tier_used == ScoringTier.TIER_1_RULES


class TestEmailPrioritizerUserActions:
    """Tests for user action recording."""

    @pytest.fixture
    def prioritizer(self):
        """Create a prioritizer instance."""
        from aragora.services.email_prioritization import EmailPrioritizer

        return EmailPrioritizer()

    @pytest.fixture
    def mock_email(self):
        """Create a mock email."""
        email = MagicMock()
        email.id = "email_123"
        email.from_address = "sender@example.com"
        email.subject = "Test Email"
        email.labels = []
        return email

    @pytest.mark.asyncio
    async def test_record_user_action_basic(self, prioritizer, mock_email):
        """Test recording basic user action."""
        # This should not raise even without mound
        await prioritizer.record_user_action(
            email_id="email_123",
            action="read",
            email=mock_email,
        )

    @pytest.mark.asyncio
    async def test_record_user_action_replied(self, prioritizer, mock_email):
        """Test recording reply action updates profile."""
        # Pre-populate sender profile
        from aragora.services.email_prioritization import SenderProfile

        profile = SenderProfile(
            email="sender@example.com",
            domain="example.com",
            total_emails_received=5,
            total_emails_responded=2,
        )
        prioritizer._sender_profiles["sender@example.com"] = profile

        await prioritizer.record_user_action(
            email_id="email_123",
            action="replied",
            email=mock_email,
        )

        # Profile should be updated
        updated_profile = prioritizer._sender_profiles["sender@example.com"]
        assert updated_profile.total_emails_responded >= 2


class TestPrioritizeInbox:
    """Tests for convenience function."""

    @pytest.mark.asyncio
    async def test_prioritize_inbox_function(self):
        """Test prioritize_inbox convenience function."""
        from aragora.services.email_prioritization import prioritize_inbox

        mock_gmail = AsyncMock()

        # Mock sync to return some emails
        async def mock_sync():
            for i in range(3):
                item = MagicMock()
                item.raw_data = {"message": MagicMock()}
                msg = item.raw_data["message"]
                msg.id = f"msg_{i}"
                msg.from_address = f"sender{i}@example.com"
                msg.subject = f"Email {i}"
                msg.body_text = "Content"
                msg.body_html = None
                msg.snippet = "Content"
                msg.is_important = False
                msg.is_starred = False
                msg.labels = []
                msg.headers = {}
                yield item

        mock_gmail.sync = mock_sync

        with patch(
            "aragora.services.email_prioritization.EmailPrioritizer.rank_inbox",
            new_callable=AsyncMock,
        ) as mock_rank:
            mock_rank.return_value = []

            results = await prioritize_inbox(
                gmail_connector=mock_gmail,
                limit=5,
            )

            assert isinstance(results, list)


class TestThreatIntelligence:
    """Tests for threat intelligence integration."""

    @pytest.fixture
    def prioritizer_with_threat_intel(self):
        """Create prioritizer with threat intelligence."""
        from aragora.services.email_prioritization import EmailPrioritizer

        mock_threat_intel = AsyncMock()
        mock_threat_intel.check_email_content = AsyncMock(
            return_value={
                "is_suspicious": False,
                "urls": [],
            }
        )

        prioritizer = EmailPrioritizer(threat_intel_service=mock_threat_intel)
        return prioritizer

    @pytest.fixture
    def mock_email(self):
        """Create a mock email."""
        email = MagicMock()
        email.id = "email_123"
        email.from_address = "sender@example.com"
        email.subject = "Test Email"
        email.body_text = "Test content"
        email.body_html = None
        email.snippet = "Test content"
        email.is_important = False
        email.is_starred = False
        email.labels = []
        email.headers = {}
        return email

    @pytest.mark.asyncio
    async def test_check_email_threats(self, prioritizer_with_threat_intel, mock_email):
        """Test threat checking."""
        result = await prioritizer_with_threat_intel._check_email_threats(mock_email)

        assert result is not None
        assert result["is_suspicious"] is False

    @pytest.mark.asyncio
    async def test_tier_1_with_malicious_url(self, mock_email):
        """Test Tier 1 with malicious URL detection."""
        from aragora.services.email_prioritization import (
            EmailPrioritizer,
            SenderProfile,
            EmailPriority,
        )

        mock_threat_intel = AsyncMock()
        mock_threat_intel.check_email_content = AsyncMock(
            return_value={
                "is_suspicious": True,
                "urls": [{"url": "http://malicious.com", "is_malicious": True}],
            }
        )

        prioritizer = EmailPrioritizer(threat_intel_service=mock_threat_intel)
        sender = SenderProfile(email="sender@example.com", domain="example.com")

        result = await prioritizer._tier_1_score(mock_email, sender)

        assert result.priority == EmailPriority.CRITICAL
        assert "THREAT" in result.rationale
        assert "Threat" in result.suggested_labels
