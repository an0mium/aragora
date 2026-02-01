"""
Tests for EmailPriorityAnalyzer.

Comprehensive tests for email priority scoring including:
- Priority score calculation
- Classification (urgent, important, normal, low)
- Rule matching
- Sender reputation scoring
- Subject line analysis
- Time-sensitivity detection
- Edge cases
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.analysis.email_priority import (
    EmailFeedbackLearner,
    EmailPriorityAnalyzer,
    EmailPriorityScore,
    UserEmailPreferences,
)


@pytest.fixture
def analyzer():
    """Create an EmailPriorityAnalyzer instance."""
    return EmailPriorityAnalyzer(user_id="test-user-123")


@pytest.fixture
def analyzer_with_prefs():
    """Create an EmailPriorityAnalyzer with pre-set preferences to avoid memory lookups."""
    analyzer = EmailPriorityAnalyzer(user_id="test-user-123")
    # Pre-set preferences to avoid memory lookups
    analyzer._preferences = UserEmailPreferences(
        user_id="test-user-123",
        important_senders=["boss@company.com"],
        important_domains=["work.io"],
        important_keywords=[],
        low_priority_senders=["noreply@marketing.com"],
        low_priority_keywords=["unsubscribe", "newsletter"],
    )
    return analyzer


@pytest.fixture
def preferences():
    """Create sample user preferences."""
    return UserEmailPreferences(
        user_id="test-user-123",
        important_senders=["boss@company.com", "vip@example.com"],
        important_domains=["company.com", "client.io"],
        important_keywords=["urgent", "deadline", "project"],
        low_priority_senders=["noreply@marketing.com", "spam@newsletter.co"],
        low_priority_keywords=["unsubscribe", "newsletter", "promotion"],
    )


class TestEmailPriorityScoreDataclass:
    """Tests for EmailPriorityScore dataclass."""

    def test_create_score(self):
        """Test creating an EmailPriorityScore."""
        score = EmailPriorityScore(
            email_id="email-123",
            score=0.75,
            reason="Important sender",
            factors={"sender": 0.9, "urgency": 0.6},
        )

        assert score.email_id == "email-123"
        assert score.score == 0.75
        assert score.reason == "Important sender"
        assert score.factors["sender"] == 0.9

    def test_to_dict(self):
        """Test serialization to dictionary."""
        score = EmailPriorityScore(
            email_id="email-456",
            score=0.5,
            reason="Normal priority",
            factors={"sender": 0.5, "content": 0.5},
        )

        result = score.to_dict()

        assert result["email_id"] == "email-456"
        assert result["score"] == 0.5
        assert result["reason"] == "Normal priority"
        assert "factors" in result


class TestUserEmailPreferences:
    """Tests for UserEmailPreferences dataclass."""

    def test_create_preferences(self):
        """Test creating preferences."""
        prefs = UserEmailPreferences(
            user_id="user-123",
            important_senders=["alice@example.com"],
            important_keywords=["project"],
        )

        assert prefs.user_id == "user-123"
        assert "alice@example.com" in prefs.important_senders
        assert "project" in prefs.important_keywords

    def test_to_dict(self):
        """Test preferences serialization."""
        prefs = UserEmailPreferences(
            user_id="user-456",
            important_domains=["company.com"],
            low_priority_keywords=["unsubscribe"],
        )

        result = prefs.to_dict()

        assert result["user_id"] == "user-456"
        assert "company.com" in result["important_domains"]


class TestSenderScoring:
    """Tests for sender-based scoring."""

    def test_score_important_sender(self, analyzer, preferences):
        """Test that known important senders get high scores."""
        score = analyzer._score_sender("boss@company.com", preferences)
        assert score == 1.0

    def test_score_low_priority_sender(self, analyzer, preferences):
        """Test that known low priority senders get low scores."""
        score = analyzer._score_sender("noreply@marketing.com", preferences)
        assert score == 0.2

    def test_score_important_domain(self, analyzer, preferences):
        """Test that emails from important domains get elevated scores."""
        score = analyzer._score_sender("anyone@client.io", preferences)
        assert score == 0.8

    def test_score_corporate_domain(self, analyzer, preferences):
        """Test that corporate domains get moderate boost."""
        score = analyzer._score_sender("contact@business.com", preferences)
        assert score == 0.6

    def test_score_personal_email(self, analyzer, preferences):
        """Test that personal emails remain neutral."""
        score = analyzer._score_sender("someone@gmail.com", preferences)
        assert score == 0.5

    def test_score_unknown_sender(self, analyzer, preferences):
        """Test neutral score for unknown senders."""
        score = analyzer._score_sender("unknown@unknown.org", preferences)
        assert score == 0.5

    def test_sender_case_insensitive(self, analyzer, preferences):
        """Test that sender matching is case insensitive."""
        score = analyzer._score_sender("BOSS@COMPANY.COM", preferences)
        assert score == 1.0


class TestGmailSignalsScoring:
    """Tests for Gmail label-based scoring."""

    def test_score_starred_email(self, analyzer):
        """Test that starred emails get highest score."""
        score = analyzer._score_gmail_signals([], is_starred=True)
        assert score == 1.0

    def test_score_important_label(self, analyzer):
        """Test that IMPORTANT labeled emails get high score."""
        score = analyzer._score_gmail_signals(["IMPORTANT"], is_starred=False)
        assert score == 0.9

    def test_score_primary_category(self, analyzer):
        """Test that primary inbox emails get elevated score."""
        score = analyzer._score_gmail_signals(["CATEGORY_PRIMARY"], is_starred=False)
        assert score == 0.7

    def test_score_promotions_category(self, analyzer):
        """Test that promotional emails get low score."""
        score = analyzer._score_gmail_signals(["CATEGORY_PROMOTIONS"], is_starred=False)
        assert score == 0.3

    def test_score_social_category(self, analyzer):
        """Test that social emails get low score."""
        score = analyzer._score_gmail_signals(["CATEGORY_SOCIAL"], is_starred=False)
        assert score == 0.3

    def test_score_updates_category(self, analyzer):
        """Test that updates category gets neutral score."""
        score = analyzer._score_gmail_signals(["CATEGORY_UPDATES"], is_starred=False)
        assert score == 0.5

    def test_score_no_labels(self, analyzer):
        """Test default score when no labels."""
        score = analyzer._score_gmail_signals([], is_starred=False)
        assert score == 0.5


class TestUrgencyScoring:
    """Tests for urgency-based scoring."""

    def test_score_urgent_keyword(self, analyzer):
        """Test that urgent keywords trigger high score."""
        score = analyzer._score_urgency("URGENT: Please respond", "This needs attention")
        assert score == 1.0

    def test_score_asap_keyword(self, analyzer):
        """Test ASAP keyword detection."""
        score = analyzer._score_urgency("Need this ASAP", "Please handle immediately")
        assert score == 1.0

    def test_score_deadline_keyword(self, analyzer):
        """Test deadline keyword detection."""
        score = analyzer._score_urgency("Project deadline approaching", "Due tomorrow")
        assert score == 1.0

    def test_score_action_required(self, analyzer):
        """Test action required keyword detection."""
        score = analyzer._score_urgency("Action required", "Please review")
        assert score == 1.0

    def test_score_medium_urgency_tomorrow(self, analyzer):
        """Test medium urgency keywords."""
        score = analyzer._score_urgency("Meeting tomorrow", "Let's discuss")
        assert score == 0.7

    def test_score_medium_urgency_reminder(self, analyzer):
        """Test reminder keyword detection."""
        score = analyzer._score_urgency("Reminder: Follow up", "Please check")
        assert score == 0.7

    def test_score_newsletter_low_priority(self, analyzer):
        """Test that newsletter keywords lower priority."""
        score = analyzer._score_urgency("Weekly Newsletter", "Click to unsubscribe")
        assert score == 0.2

    def test_score_no_urgency_signals(self, analyzer):
        """Test neutral score when no urgency signals."""
        score = analyzer._score_urgency("Regular email", "Just checking in")
        assert score == 0.5


class TestThreadScoring:
    """Tests for thread engagement scoring."""

    def test_score_unread_email(self, analyzer):
        """Test that unread emails get slight boost."""
        score = analyzer._score_thread(thread_count=1, is_read=False)
        assert score == 0.7  # 0.5 base + 0.2 for unread

    def test_score_read_email(self, analyzer):
        """Test read emails baseline."""
        score = analyzer._score_thread(thread_count=1, is_read=True)
        assert score == 0.5

    def test_score_active_thread(self, analyzer):
        """Test that active threads get boost."""
        score = analyzer._score_thread(thread_count=6, is_read=True)
        assert score == 0.8  # 0.5 + 0.3 for > 5 messages

    def test_score_moderate_thread(self, analyzer):
        """Test moderate thread activity."""
        score = analyzer._score_thread(thread_count=3, is_read=True)
        assert score == 0.6  # 0.5 + 0.1 for > 2 messages

    def test_score_unread_active_thread(self, analyzer):
        """Test unread in active thread (capped at 1.0)."""
        score = analyzer._score_thread(thread_count=10, is_read=False)
        assert score == 1.0


class TestReasonGeneration:
    """Tests for explanation generation."""

    def test_reason_important_sender(self, analyzer):
        """Test reason for high sender score."""
        factors = {"sender": 0.9, "urgency": 0.5, "gmail_signals": 0.5}
        reason = analyzer._generate_reason(factors, "boss@company.com", "Meeting")
        assert "important sender" in reason.lower()

    def test_reason_gmail_important(self, analyzer):
        """Test reason for Gmail important label."""
        factors = {"sender": 0.5, "urgency": 0.5, "gmail_signals": 0.9}
        reason = analyzer._generate_reason(factors, "test@example.com", "Test")
        assert "gmail" in reason.lower() or "important" in reason.lower()

    def test_reason_urgent(self, analyzer):
        """Test reason for urgent content."""
        factors = {"sender": 0.5, "urgency": 1.0, "gmail_signals": 0.5}
        reason = analyzer._generate_reason(factors, "test@example.com", "URGENT")
        assert "urgent" in reason.lower()

    def test_reason_low_priority_sender(self, analyzer):
        """Test reason for low priority sender."""
        factors = {"sender": 0.2, "urgency": 0.5, "gmail_signals": 0.5}
        reason = analyzer._generate_reason(factors, "spam@newsletter.com", "News")
        assert "low-priority" in reason.lower() or "priority" in reason.lower()


class TestFullEmailScoring:
    """Tests for complete email scoring."""

    @pytest.mark.asyncio
    async def test_score_email_basic(self, analyzer_with_prefs):
        """Test basic email scoring."""
        score = await analyzer_with_prefs.score_email(
            email_id="email-123",
            subject="Hello",
            from_address="friend@gmail.com",
            snippet="Just checking in",
        )

        assert isinstance(score, EmailPriorityScore)
        assert score.email_id == "email-123"
        assert 0.0 <= score.score <= 1.0
        assert score.reason != ""
        assert "sender" in score.factors

    @pytest.mark.asyncio
    async def test_score_urgent_email(self, analyzer_with_prefs):
        """Test that urgent emails get high scores."""
        score = await analyzer_with_prefs.score_email(
            email_id="urgent-123",
            subject="URGENT: Immediate action required",
            from_address="boss@company.com",
            snippet="This is time sensitive",
            labels=["IMPORTANT"],
            is_starred=True,
        )

        assert score.score > 0.7  # Should be high priority

    @pytest.mark.asyncio
    async def test_score_promotional_email(self, analyzer_with_prefs):
        """Test that promotional emails get low scores."""
        score = await analyzer_with_prefs.score_email(
            email_id="promo-123",
            subject="Weekly Newsletter - 50% Off!",
            from_address="noreply@marketing.com",
            snippet="Click here to unsubscribe",
            labels=["CATEGORY_PROMOTIONS"],
        )

        assert score.score < 0.5  # Should be low priority

    @pytest.mark.asyncio
    async def test_score_email_with_all_params(self, analyzer_with_prefs):
        """Test scoring with all parameters provided."""
        score = await analyzer_with_prefs.score_email(
            email_id="full-123",
            subject="Project Update",
            from_address="colleague@work.io",
            snippet="Here's the latest status",
            body_text="Full email body content here...",
            labels=["CATEGORY_PRIMARY", "IMPORTANT"],
            is_read=False,
            is_starred=False,
            thread_count=5,
        )

        assert isinstance(score, EmailPriorityScore)
        assert len(score.factors) >= 4  # Should have multiple factors


class TestBatchScoring:
    """Tests for batch email scoring."""

    @pytest.mark.asyncio
    async def test_score_batch_empty(self, analyzer_with_prefs):
        """Test batch scoring with empty list."""
        result = await analyzer_with_prefs.score_batch([])
        assert result == []

    @pytest.mark.asyncio
    async def test_score_batch_single(self, analyzer_with_prefs):
        """Test batch scoring with single email."""
        emails = [
            {
                "id": "batch-1",
                "subject": "Test",
                "from_address": "test@example.com",
                "snippet": "Test content",
            }
        ]

        results = await analyzer_with_prefs.score_batch(emails)

        assert len(results) == 1
        assert results[0].email_id == "batch-1"

    @pytest.mark.asyncio
    async def test_score_batch_multiple(self, analyzer_with_prefs):
        """Test batch scoring with multiple emails."""
        emails = [
            {
                "id": f"batch-{i}",
                "subject": f"Subject {i}",
                "from_address": f"sender{i}@example.com",
                "snippet": f"Content {i}",
            }
            for i in range(5)
        ]

        results = await analyzer_with_prefs.score_batch(emails)

        assert len(results) == 5
        for i, result in enumerate(results):
            assert result.email_id == f"batch-{i}"

    @pytest.mark.asyncio
    async def test_score_batch_preserves_order(self, analyzer_with_prefs):
        """Test that batch scoring preserves input order."""
        emails = [
            {"id": "first", "subject": "First", "from_address": "a@test.com", "snippet": "1"},
            {"id": "second", "subject": "Second", "from_address": "b@test.com", "snippet": "2"},
            {"id": "third", "subject": "Third", "from_address": "c@test.com", "snippet": "3"},
        ]

        results = await analyzer_with_prefs.score_batch(emails)

        assert results[0].email_id == "first"
        assert results[1].email_id == "second"
        assert results[2].email_id == "third"


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_empty_subject(self, analyzer_with_prefs):
        """Test handling of empty subject."""
        score = await analyzer_with_prefs.score_email(
            email_id="empty-subj",
            subject="",
            from_address="test@example.com",
            snippet="Some content",
        )

        assert isinstance(score, EmailPriorityScore)
        assert 0.0 <= score.score <= 1.0

    @pytest.mark.asyncio
    async def test_empty_sender(self, analyzer_with_prefs):
        """Test handling of empty sender."""
        score = await analyzer_with_prefs.score_email(
            email_id="empty-sender",
            subject="Test",
            from_address="",
            snippet="Some content",
        )

        assert isinstance(score, EmailPriorityScore)

    @pytest.mark.asyncio
    async def test_none_labels(self, analyzer_with_prefs):
        """Test handling of None labels."""
        score = await analyzer_with_prefs.score_email(
            email_id="none-labels",
            subject="Test",
            from_address="test@example.com",
            snippet="Content",
            labels=None,
        )

        assert isinstance(score, EmailPriorityScore)

    @pytest.mark.asyncio
    async def test_invalid_email_format(self, analyzer_with_prefs):
        """Test handling of invalid email format."""
        score = await analyzer_with_prefs.score_email(
            email_id="invalid-email",
            subject="Test",
            from_address="not-an-email",
            snippet="Content",
        )

        assert isinstance(score, EmailPriorityScore)
        # Should still return a valid score

    @pytest.mark.asyncio
    async def test_very_long_content(self, analyzer_with_prefs):
        """Test handling of very long content."""
        long_subject = "A" * 1000
        long_snippet = "B" * 5000

        score = await analyzer_with_prefs.score_email(
            email_id="long-content",
            subject=long_subject,
            from_address="test@example.com",
            snippet=long_snippet,
        )

        assert isinstance(score, EmailPriorityScore)

    @pytest.mark.asyncio
    async def test_special_characters_in_content(self, analyzer_with_prefs):
        """Test handling of special characters."""
        score = await analyzer_with_prefs.score_email(
            email_id="special-chars",
            subject="Test <script>alert('xss')</script>",
            from_address="test@example.com",
            snippet="Content with \x00 null bytes and unicode: \u2603",
        )

        assert isinstance(score, EmailPriorityScore)


class TestKeywordExtraction:
    """Tests for keyword extraction utility."""

    def test_extract_keywords_basic(self, analyzer):
        """Test basic keyword extraction."""
        content = "This is a test about important project deadlines"
        keywords = analyzer._extract_keywords(content)

        assert isinstance(keywords, list)
        assert len(keywords) <= 5  # Max 5 keywords
        assert all(len(kw) > 3 for kw in keywords)

    def test_extract_keywords_filters_stopwords(self, analyzer):
        """Test that stopwords are filtered."""
        content = "the is a an are was were been being"
        keywords = analyzer._extract_keywords(content)

        assert len(keywords) == 0

    def test_extract_keywords_short_words(self, analyzer):
        """Test that short words are filtered."""
        content = "a to in on at by for"
        keywords = analyzer._extract_keywords(content)

        assert len(keywords) == 0


class TestSenderExtraction:
    """Tests for sender email extraction."""

    def test_extract_sender_from_content(self, analyzer):
        """Test extracting email from content."""
        content = "Replied to email from boss@company.com about meeting"
        sender = analyzer._extract_sender(content)

        assert sender == "boss@company.com"

    def test_extract_sender_no_email(self, analyzer):
        """Test extraction when no email present."""
        content = "No email address in this content"
        sender = analyzer._extract_sender(content)

        assert sender is None

    def test_extract_sender_multiple_emails(self, analyzer):
        """Test extraction with multiple emails (returns first)."""
        content = "Contact alice@example.com or bob@example.com"
        sender = analyzer._extract_sender(content)

        assert sender == "alice@example.com"


class TestEmailFeedbackLearner:
    """Tests for EmailFeedbackLearner."""

    @pytest.fixture
    def learner(self):
        """Create an EmailFeedbackLearner instance."""
        return EmailFeedbackLearner(user_id="test-user-456")

    @pytest.mark.asyncio
    async def test_record_interaction_replied(self, learner):
        """Test recording a replied interaction."""
        # Mock memory
        mock_memory = AsyncMock()
        learner._memory = mock_memory

        result = await learner.record_interaction(
            email_id="email-123",
            action="replied",
            from_address="important@example.com",
            subject="Test Subject",
        )

        assert result is True
        mock_memory.store.assert_called_once()
        call_kwargs = mock_memory.store.call_args.kwargs
        assert "replied" in call_kwargs["content"].lower()
        assert call_kwargs["importance"] == 0.9

    @pytest.mark.asyncio
    async def test_record_interaction_starred(self, learner):
        """Test recording a starred interaction."""
        mock_memory = AsyncMock()
        learner._memory = mock_memory

        result = await learner.record_interaction(
            email_id="email-456",
            action="starred",
            from_address="vip@example.com",
            subject="Important",
        )

        assert result is True
        call_kwargs = mock_memory.store.call_args.kwargs
        assert call_kwargs["importance"] == 0.8

    @pytest.mark.asyncio
    async def test_record_interaction_deleted(self, learner):
        """Test recording a deleted interaction."""
        mock_memory = AsyncMock()
        learner._memory = mock_memory

        result = await learner.record_interaction(
            email_id="email-789",
            action="deleted",
            from_address="spam@example.com",
            subject="Spam",
        )

        assert result is True
        call_kwargs = mock_memory.store.call_args.kwargs
        assert call_kwargs["importance"] == 0.2

    @pytest.mark.asyncio
    async def test_record_interaction_unknown_action(self, learner):
        """Test recording unknown action returns False."""
        mock_memory = AsyncMock()
        learner._memory = mock_memory

        result = await learner.record_interaction(
            email_id="email-000",
            action="unknown_action",
            from_address="test@example.com",
            subject="Test",
        )

        assert result is False
        mock_memory.store.assert_not_called()

    @pytest.mark.asyncio
    async def test_record_interaction_no_memory(self, learner):
        """Test recording when memory is not available."""
        # Patch the memory getter to return None
        learner._get_memory = AsyncMock(return_value=None)

        result = await learner.record_interaction(
            email_id="email-999",
            action="replied",
            from_address="test@example.com",
            subject="Test",
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_consolidate_preferences(self, learner):
        """Test preferences consolidation."""
        mock_memory = AsyncMock()
        mock_memory.consolidate = AsyncMock()
        learner._memory = mock_memory

        result = await learner.consolidate_preferences()

        assert result is True
        mock_memory.consolidate.assert_called_once()

    @pytest.mark.asyncio
    async def test_consolidate_no_memory(self, learner):
        """Test consolidation when memory unavailable."""
        learner._get_memory = AsyncMock(return_value=None)

        result = await learner.consolidate_preferences()

        assert result is False
