"""
Tests for Email Categorization Service.

Tests for intelligent email folder assignment:
- Category patterns and detection
- Sender domain categorization
- Pattern matching and scoring
- Batch categorization
- Gmail label integration
- Statistics generation
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestEmailCategory:
    """Tests for EmailCategory enum."""

    def test_email_category_values(self):
        """Test EmailCategory enum values."""
        from aragora.services.email_categorizer import EmailCategory

        assert EmailCategory.INVOICES.value == "invoices"
        assert EmailCategory.HR.value == "hr"
        assert EmailCategory.NEWSLETTERS.value == "newsletters"
        assert EmailCategory.PROJECTS.value == "projects"
        assert EmailCategory.MEETINGS.value == "meetings"
        assert EmailCategory.SUPPORT.value == "support"
        assert EmailCategory.PERSONAL.value == "personal"
        assert EmailCategory.SECURITY.value == "security"
        assert EmailCategory.RECEIPTS.value == "receipts"
        assert EmailCategory.SOCIAL.value == "social"
        assert EmailCategory.UNCATEGORIZED.value == "uncategorized"


class TestCategoryPattern:
    """Tests for CategoryPattern dataclass."""

    def test_category_pattern_minimal(self):
        """Test minimal CategoryPattern."""
        from aragora.services.email_categorizer import CategoryPattern

        pattern = CategoryPattern(pattern=r"\binvoice\b")

        assert pattern.pattern == r"\binvoice\b"
        assert pattern.weight == 1.0
        assert pattern.field == "all"

    def test_category_pattern_full(self):
        """Test CategoryPattern with all fields."""
        from aragora.services.email_categorizer import CategoryPattern

        pattern = CategoryPattern(
            pattern=r"hr@",
            weight=0.8,
            field="from",
        )

        assert pattern.pattern == r"hr@"
        assert pattern.weight == 0.8
        assert pattern.field == "from"


class TestCategorizationResult:
    """Tests for CategorizationResult dataclass."""

    def test_result_minimal(self):
        """Test minimal CategorizationResult."""
        from aragora.services.email_categorizer import (
            CategorizationResult,
            EmailCategory,
        )

        result = CategorizationResult(
            email_id="email_123",
            category=EmailCategory.INVOICES,
            confidence=0.85,
        )

        assert result.email_id == "email_123"
        assert result.category == EmailCategory.INVOICES
        assert result.confidence == 0.85
        assert result.secondary_category is None
        assert result.auto_archive is False

    def test_result_full(self):
        """Test full CategorizationResult."""
        from aragora.services.email_categorizer import (
            CategorizationResult,
            EmailCategory,
        )

        result = CategorizationResult(
            email_id="email_456",
            category=EmailCategory.NEWSLETTERS,
            confidence=0.9,
            secondary_category=EmailCategory.SOCIAL,
            matched_patterns=[r"\bunsubscribe\b", r"\bnewsletter\b"],
            suggested_label="Aragora/Newsletters",
            auto_archive=True,
            rationale="Detected newsletter content",
        )

        assert result.secondary_category == EmailCategory.SOCIAL
        assert len(result.matched_patterns) == 2
        assert result.auto_archive is True

    def test_result_to_dict(self):
        """Test CategorizationResult.to_dict."""
        from aragora.services.email_categorizer import (
            CategorizationResult,
            EmailCategory,
        )

        result = CategorizationResult(
            email_id="email_789",
            category=EmailCategory.PROJECTS,
            confidence=0.75,
            secondary_category=EmailCategory.MEETINGS,
            matched_patterns=[r"\bsprint\b"],
            suggested_label="Aragora/Projects",
            auto_archive=False,
            rationale="Project update detected",
        )

        data = result.to_dict()

        assert data["email_id"] == "email_789"
        assert data["category"] == "projects"
        assert data["confidence"] == 0.75
        assert data["secondary_category"] == "meetings"
        assert data["matched_patterns"] == [r"\bsprint\b"]
        assert data["auto_archive"] is False


class TestEmailCategorizerConfig:
    """Tests for EmailCategorizerConfig."""

    def test_config_defaults(self):
        """Test default config values."""
        from aragora.services.email_categorizer import (
            EmailCategorizerConfig,
            EmailCategory,
        )

        config = EmailCategorizerConfig()

        assert config.high_confidence_threshold == 0.7
        assert config.low_confidence_threshold == 0.3
        assert config.auto_archive_newsletters is True
        assert config.auto_archive_social is False
        assert len(config.enabled_categories) > 0

    def test_config_custom(self):
        """Test custom config values."""
        from aragora.services.email_categorizer import (
            EmailCategorizerConfig,
            EmailCategory,
        )

        config = EmailCategorizerConfig(
            high_confidence_threshold=0.8,
            auto_archive_newsletters=False,
            auto_archive_social=True,
            custom_sender_categories={"ceo@company.com": "hr"},
        )

        assert config.high_confidence_threshold == 0.8
        assert config.auto_archive_newsletters is False
        assert config.custom_sender_categories["ceo@company.com"] == "hr"


class TestEmailCategorizer:
    """Tests for EmailCategorizer class."""

    def test_categorizer_initialization(self):
        """Test categorizer initialization."""
        from aragora.services.email_categorizer import EmailCategorizer

        categorizer = EmailCategorizer()

        assert categorizer.gmail is None
        assert categorizer.config is not None
        assert len(categorizer._compiled_patterns) > 0

    def test_categorizer_with_config(self):
        """Test categorizer with custom config."""
        from aragora.services.email_categorizer import (
            EmailCategorizer,
            EmailCategorizerConfig,
        )

        config = EmailCategorizerConfig(
            auto_archive_newsletters=False,
        )
        categorizer = EmailCategorizer(config=config)

        assert categorizer.config.auto_archive_newsletters is False

    def test_extract_domain(self):
        """Test domain extraction."""
        from aragora.services.email_categorizer import EmailCategorizer

        categorizer = EmailCategorizer()

        assert categorizer._extract_domain("user@example.com") == "example.com"
        assert categorizer._extract_domain("admin@sub.domain.org") == "sub.domain.org"
        assert categorizer._extract_domain("no-at-sign") == ""

    def test_get_text_for_field(self):
        """Test text extraction by field."""
        from aragora.services.email_categorizer import EmailCategorizer

        categorizer = EmailCategorizer()

        text = categorizer._get_text_for_field(
            "subject",
            "Invoice for March",
            "Please pay by Friday",
            "billing@company.com",
        )
        assert text == "Invoice for March"

        text = categorizer._get_text_for_field(
            "body",
            "Invoice for March",
            "Please pay by Friday",
            "billing@company.com",
        )
        assert text == "Please pay by Friday"

        text = categorizer._get_text_for_field(
            "from",
            "Invoice for March",
            "Please pay by Friday",
            "billing@company.com",
        )
        assert text == "billing@company.com"

        text = categorizer._get_text_for_field(
            "all",
            "Invoice for March",
            "Please pay by Friday",
            "billing@company.com",
        )
        assert "Invoice" in text
        assert "Friday" in text
        assert "billing" in text


class TestEmailCategorizerCategorization:
    """Tests for email categorization logic."""

    @pytest.fixture
    def categorizer(self):
        """Create a categorizer instance."""
        from aragora.services.email_categorizer import EmailCategorizer

        return EmailCategorizer()

    @pytest.fixture
    def mock_email(self):
        """Create a mock email."""
        email = MagicMock()
        email.id = "email_123"
        email.subject = "Test Email"
        email.body = "Test body content"
        email.snippet = "Test body content"
        email.sender = "sender@example.com"
        email.from_ = "sender@example.com"
        return email

    @pytest.mark.asyncio
    async def test_categorize_invoice_email(self, categorizer):
        """Test categorizing an invoice email."""
        from aragora.services.email_categorizer import EmailCategory

        email = MagicMock()
        email.id = "inv_001"
        email.subject = "Invoice #12345 - Payment Due"
        email.body = "Please find attached your invoice for $500.00. Payment due in Net 30 days."
        email.snippet = email.body
        email.sender = "accounts@vendor.com"
        email.from_ = "accounts@vendor.com"

        result = await categorizer.categorize_email(email)

        assert result.category == EmailCategory.INVOICES
        assert result.confidence > 0.5
        assert len(result.matched_patterns) > 0

    @pytest.mark.asyncio
    async def test_categorize_hr_email(self, categorizer):
        """Test categorizing an HR email."""
        from aragora.services.email_categorizer import EmailCategory

        email = MagicMock()
        email.id = "hr_001"
        email.subject = "Your PTO Request Has Been Approved"
        email.body = (
            "Your vacation leave request for next week has been approved. Enjoy your time off!"
        )
        email.snippet = email.body
        email.sender = "hr@company.com"
        email.from_ = "hr@company.com"

        result = await categorizer.categorize_email(email)

        assert result.category == EmailCategory.HR
        assert result.confidence > 0.5

    @pytest.mark.asyncio
    async def test_categorize_newsletter_email(self, categorizer):
        """Test categorizing a newsletter email."""
        from aragora.services.email_categorizer import EmailCategory

        email = MagicMock()
        email.id = "news_001"
        email.subject = "Weekly Digest - Top Stories"
        email.body = "Here are this week's top stories. View in browser. Click here to unsubscribe."
        email.snippet = email.body
        email.sender = "no-reply@newsletter.com"
        email.from_ = "no-reply@newsletter.com"

        result = await categorizer.categorize_email(email)

        assert result.category == EmailCategory.NEWSLETTERS
        assert result.auto_archive is True

    @pytest.mark.asyncio
    async def test_categorize_project_email(self, categorizer):
        """Test categorizing a project email."""
        from aragora.services.email_categorizer import EmailCategory

        email = MagicMock()
        email.id = "proj_001"
        email.subject = "[Project Alpha] Sprint 5 - Task Assigned"
        email.body = "A new task has been assigned to you in Jira. Please review the pull request."
        email.snippet = email.body
        email.sender = "notifications@atlassian.net"
        email.from_ = "notifications@atlassian.net"

        result = await categorizer.categorize_email(email)

        assert result.category == EmailCategory.PROJECTS

    @pytest.mark.asyncio
    async def test_categorize_meeting_email(self, categorizer):
        """Test categorizing a meeting email."""
        from aragora.services.email_categorizer import EmailCategory

        email = MagicMock()
        email.id = "meet_001"
        email.subject = "Meeting Invitation: Weekly Standup"
        email.body = "You're invited to a meeting. Join via zoom.us/j/123456. Agenda: Team sync."
        email.snippet = email.body
        email.sender = "calendar-notification@google.com"
        email.from_ = "calendar-notification@google.com"

        result = await categorizer.categorize_email(email)

        assert result.category == EmailCategory.MEETINGS

    @pytest.mark.asyncio
    async def test_categorize_security_email(self, categorizer):
        """Test categorizing a security email."""
        from aragora.services.email_categorizer import EmailCategory

        email = MagicMock()
        email.id = "sec_001"
        email.subject = "Security Alert: New Sign-in Detected"
        email.body = (
            "A new sign-in was detected on your account. Verification code: 123456. 2FA required."
        )
        email.snippet = email.body
        email.sender = "security@example.com"
        email.from_ = "security@example.com"

        result = await categorizer.categorize_email(email)

        assert result.category == EmailCategory.SECURITY

    @pytest.mark.asyncio
    async def test_categorize_receipt_email(self, categorizer):
        """Test categorizing a receipt email."""
        from aragora.services.email_categorizer import EmailCategory

        email = MagicMock()
        email.id = "rec_001"
        email.subject = "Order Confirmation #ORD-12345"
        email.body = (
            "Thank you for your purchase! Your order has shipped. Tracking number: 1234567890."
        )
        email.snippet = email.body
        email.sender = "orders@amazon.com"
        email.from_ = "orders@amazon.com"

        result = await categorizer.categorize_email(email)

        assert result.category == EmailCategory.RECEIPTS

    @pytest.mark.asyncio
    async def test_categorize_social_email(self, categorizer):
        """Test categorizing a social email."""
        from aragora.services.email_categorizer import EmailCategory

        email = MagicMock()
        email.id = "soc_001"
        email.subject = "John Smith liked your post"
        email.body = "Someone viewed your profile. Connect with them on LinkedIn."
        email.snippet = email.body
        email.sender = "notifications@linkedin.com"
        email.from_ = "notifications@linkedin.com"

        result = await categorizer.categorize_email(email)

        assert result.category == EmailCategory.SOCIAL

    @pytest.mark.asyncio
    async def test_categorize_uncategorized_email(self, categorizer):
        """Test email that doesn't match any category."""
        from aragora.services.email_categorizer import EmailCategory

        email = MagicMock()
        email.id = "misc_001"
        email.subject = "Hello"
        email.body = "Just wanted to say hi."
        email.snippet = email.body
        email.sender = "random@unknown.org"
        email.from_ = "random@unknown.org"

        result = await categorizer.categorize_email(email)

        assert result.category == EmailCategory.UNCATEGORIZED
        assert result.confidence == 1.0

    @pytest.mark.asyncio
    async def test_categorize_by_sender_domain(self, categorizer):
        """Test categorization by known sender domain."""
        from aragora.services.email_categorizer import EmailCategory

        # GitHub should be categorized as Projects
        email = MagicMock()
        email.id = "gh_001"
        email.subject = "Notification"
        email.body = "Some notification"
        email.snippet = email.body
        email.sender = "notifications@github.com"
        email.from_ = "notifications@github.com"

        result = await categorizer.categorize_email(email)

        assert result.category == EmailCategory.PROJECTS
        assert result.confidence == 0.85
        assert "Sender domain: github.com" in result.matched_patterns

    @pytest.mark.asyncio
    async def test_categorize_by_custom_sender(self, categorizer):
        """Test categorization by custom sender mapping."""
        from aragora.services.email_categorizer import (
            EmailCategorizer,
            EmailCategorizerConfig,
            EmailCategory,
        )

        config = EmailCategorizerConfig(custom_sender_categories={"ceo@company.com": "hr"})
        custom_categorizer = EmailCategorizer(config=config)

        email = MagicMock()
        email.id = "ceo_001"
        email.subject = "Quick question"
        email.body = "Can you help with something?"
        email.snippet = email.body
        email.sender = "ceo@company.com"
        email.from_ = "ceo@company.com"

        result = await custom_categorizer.categorize_email(email)

        assert result.category == EmailCategory.HR
        assert result.confidence == 0.9

    @pytest.mark.asyncio
    async def test_secondary_category_detection(self, categorizer):
        """Test detection of secondary category."""
        from aragora.services.email_categorizer import EmailCategory

        # Email that matches both project and meeting categories
        email = MagicMock()
        email.id = "multi_001"
        email.subject = "Meeting Invitation: Sprint Planning"
        email.body = "Let's discuss the sprint goals and task assignments."
        email.snippet = email.body
        email.sender = "pm@company.com"
        email.from_ = "pm@company.com"

        result = await categorizer.categorize_email(email)

        # Should have a primary and possibly secondary category
        assert result.category in [EmailCategory.MEETINGS, EmailCategory.PROJECTS]


class TestEmailCategorizerBatch:
    """Tests for batch categorization."""

    @pytest.mark.asyncio
    async def test_categorize_batch(self):
        """Test batch categorization."""
        from aragora.services.email_categorizer import EmailCategorizer, EmailCategory

        categorizer = EmailCategorizer()

        emails = []
        for i in range(5):
            email = MagicMock()
            email.id = f"email_{i}"
            email.subject = f"Invoice #{i}" if i % 2 == 0 else "Newsletter"
            email.body = "Content"
            email.snippet = "Content"
            email.sender = "sender@example.com"
            email.from_ = "sender@example.com"
            emails.append(email)

        results = await categorizer.categorize_batch(emails)

        assert len(results) == 5
        for result in results:
            assert result.email_id is not None

    @pytest.mark.asyncio
    async def test_categorize_batch_concurrency(self):
        """Test batch categorization with limited concurrency."""
        from aragora.services.email_categorizer import EmailCategorizer

        categorizer = EmailCategorizer()

        emails = []
        for i in range(20):
            email = MagicMock()
            email.id = f"email_{i}"
            email.subject = f"Test {i}"
            email.body = "Content"
            email.snippet = "Content"
            email.sender = "sender@example.com"
            email.from_ = "sender@example.com"
            emails.append(email)

        results = await categorizer.categorize_batch(emails, concurrency=5)

        assert len(results) == 20

    @pytest.mark.asyncio
    async def test_categorize_batch_handles_errors(self):
        """Test batch categorization handles individual errors."""
        from aragora.services.email_categorizer import EmailCategorizer, EmailCategory

        categorizer = EmailCategorizer()

        # Create emails where one will fail
        emails = []
        for i in range(3):
            email = MagicMock()
            email.id = f"email_{i}"
            email.subject = "Test"
            email.body = "Content"
            email.snippet = "Content"
            email.sender = "sender@example.com"
            email.from_ = "sender@example.com"
            emails.append(email)

        # Inject an error by making one email raise an exception
        bad_email = MagicMock()
        bad_email.id = "bad_email"

        # Make property access raise an exception
        type(bad_email).subject = property(
            lambda self: (_ for _ in ()).throw(ValueError("Bad email"))
        )
        emails.insert(1, bad_email)

        results = await categorizer.categorize_batch(emails)

        # Should still return results for all emails
        assert len(results) == 4
        # The bad email should be uncategorized
        assert results[1].category == EmailCategory.UNCATEGORIZED


class TestEmailCategorizerGmailIntegration:
    """Tests for Gmail label integration."""

    @pytest.mark.asyncio
    async def test_apply_gmail_label_no_connector(self):
        """Test applying label without Gmail connector."""
        from aragora.services.email_categorizer import EmailCategorizer, EmailCategory

        categorizer = EmailCategorizer()

        result = await categorizer.apply_gmail_label("email_123", EmailCategory.INVOICES)

        assert result is False

    @pytest.mark.asyncio
    async def test_apply_gmail_label_existing(self):
        """Test applying existing Gmail label."""
        from aragora.services.email_categorizer import EmailCategorizer, EmailCategory

        mock_gmail = AsyncMock()
        mock_label = MagicMock()
        mock_label.name = "Aragora/Invoices"
        mock_label.id = "label_123"
        mock_gmail.list_labels = AsyncMock(return_value=[mock_label])
        mock_gmail.add_label = AsyncMock()

        categorizer = EmailCategorizer(gmail_connector=mock_gmail)

        result = await categorizer.apply_gmail_label("email_123", EmailCategory.INVOICES)

        assert result is True
        mock_gmail.add_label.assert_called_once_with("email_123", "label_123")

    @pytest.mark.asyncio
    async def test_apply_gmail_label_create_new(self):
        """Test creating new Gmail label."""
        from aragora.services.email_categorizer import EmailCategorizer, EmailCategory

        mock_gmail = AsyncMock()
        mock_gmail.list_labels = AsyncMock(return_value=[])
        new_label = MagicMock()
        new_label.id = "new_label_123"
        mock_gmail.create_label = AsyncMock(return_value=new_label)
        mock_gmail.add_label = AsyncMock()

        categorizer = EmailCategorizer(gmail_connector=mock_gmail)

        result = await categorizer.apply_gmail_label("email_123", EmailCategory.HR)

        assert result is True
        mock_gmail.create_label.assert_called_once_with("Aragora/Hr")
        mock_gmail.add_label.assert_called_once_with("email_123", "new_label_123")

    @pytest.mark.asyncio
    async def test_apply_gmail_label_error(self):
        """Test handling Gmail API errors."""
        from aragora.services.email_categorizer import EmailCategorizer, EmailCategory

        mock_gmail = AsyncMock()
        mock_gmail.list_labels = AsyncMock(side_effect=Exception("API error"))

        categorizer = EmailCategorizer(gmail_connector=mock_gmail)

        result = await categorizer.apply_gmail_label("email_123", EmailCategory.INVOICES)

        assert result is False


class TestEmailCategorizerStats:
    """Tests for category statistics."""

    def test_get_category_stats_empty(self):
        """Test stats with no results."""
        from aragora.services.email_categorizer import EmailCategorizer

        categorizer = EmailCategorizer()
        stats = categorizer.get_category_stats([])

        assert stats["total"] == 0
        assert stats["categories"] == {}
        assert stats["confidence_avg"] == 0.0

    def test_get_category_stats(self):
        """Test category statistics."""
        from aragora.services.email_categorizer import (
            EmailCategorizer,
            CategorizationResult,
            EmailCategory,
        )

        categorizer = EmailCategorizer()

        results = [
            CategorizationResult(
                email_id="1",
                category=EmailCategory.INVOICES,
                confidence=0.9,
            ),
            CategorizationResult(
                email_id="2",
                category=EmailCategory.INVOICES,
                confidence=0.8,
            ),
            CategorizationResult(
                email_id="3",
                category=EmailCategory.NEWSLETTERS,
                confidence=0.95,
                auto_archive=True,
            ),
            CategorizationResult(
                email_id="4",
                category=EmailCategory.UNCATEGORIZED,
                confidence=1.0,
            ),
        ]

        stats = categorizer.get_category_stats(results)

        assert stats["total"] == 4
        assert stats["categories"]["invoices"] == 2
        assert stats["categories"]["newsletters"] == 1
        assert stats["auto_archive_count"] == 1
        assert stats["uncategorized_count"] == 1
        assert 0.9 <= stats["confidence_avg"] <= 0.95


class TestEmailCategorizerAutoArchive:
    """Tests for auto-archive logic."""

    def test_should_auto_archive_newsletters(self):
        """Test newsletter auto-archive."""
        from aragora.services.email_categorizer import (
            EmailCategorizer,
            EmailCategorizerConfig,
            EmailCategory,
        )

        config = EmailCategorizerConfig(auto_archive_newsletters=True)
        categorizer = EmailCategorizer(config=config)

        assert categorizer._should_auto_archive(EmailCategory.NEWSLETTERS) is True

    def test_should_not_auto_archive_newsletters(self):
        """Test newsletter not auto-archived when disabled."""
        from aragora.services.email_categorizer import (
            EmailCategorizer,
            EmailCategorizerConfig,
            EmailCategory,
        )

        config = EmailCategorizerConfig(auto_archive_newsletters=False)
        categorizer = EmailCategorizer(config=config)

        assert categorizer._should_auto_archive(EmailCategory.NEWSLETTERS) is False

    def test_should_auto_archive_social(self):
        """Test social auto-archive when enabled."""
        from aragora.services.email_categorizer import (
            EmailCategorizer,
            EmailCategorizerConfig,
            EmailCategory,
        )

        config = EmailCategorizerConfig(auto_archive_social=True)
        categorizer = EmailCategorizer(config=config)

        assert categorizer._should_auto_archive(EmailCategory.SOCIAL) is True

    def test_should_not_auto_archive_invoices(self):
        """Test invoices never auto-archived."""
        from aragora.services.email_categorizer import EmailCategorizer, EmailCategory

        categorizer = EmailCategorizer()

        assert categorizer._should_auto_archive(EmailCategory.INVOICES) is False
        assert categorizer._should_auto_archive(EmailCategory.HR) is False
        assert categorizer._should_auto_archive(EmailCategory.SECURITY) is False


class TestEmailCategorizerRationale:
    """Tests for rationale generation."""

    def test_generate_rationale(self):
        """Test rationale generation."""
        from aragora.services.email_categorizer import EmailCategorizer, EmailCategory

        categorizer = EmailCategorizer()

        rationale = categorizer._generate_rationale(
            EmailCategory.INVOICES,
            [r"\binvoice\b", r"\$\d+"],
        )

        assert "financial" in rationale.lower() or "billing" in rationale.lower()
        assert "2 pattern" in rationale

    def test_generate_rationale_no_patterns(self):
        """Test rationale with no matched patterns."""
        from aragora.services.email_categorizer import EmailCategorizer, EmailCategory

        categorizer = EmailCategorizer()

        rationale = categorizer._generate_rationale(EmailCategory.HR, [])

        assert rationale != ""


class TestCategorizeEmailQuick:
    """Tests for quick categorization function."""

    @pytest.mark.asyncio
    async def test_categorize_email_quick(self):
        """Test quick categorization without email object."""
        from aragora.services.email_categorizer import (
            categorize_email_quick,
            EmailCategory,
        )

        result = await categorize_email_quick(
            subject="Invoice #123 - Payment Due",
            body="Please pay the attached invoice of $500.00",
            sender="billing@vendor.com",
        )

        assert result.category == EmailCategory.INVOICES
        assert result.confidence > 0

    @pytest.mark.asyncio
    async def test_categorize_email_quick_newsletter(self):
        """Test quick categorization of newsletter."""
        from aragora.services.email_categorizer import (
            categorize_email_quick,
            EmailCategory,
        )

        result = await categorize_email_quick(
            subject="Weekly Newsletter - Tech Updates",
            body="View in browser. Unsubscribe here.",
            sender="no-reply@newsletter.com",
        )

        assert result.category == EmailCategory.NEWSLETTERS


class TestCategoryPatterns:
    """Tests for category pattern constants."""

    def test_category_patterns_exist(self):
        """Test that category patterns are defined."""
        from aragora.services.email_categorizer import (
            CATEGORY_PATTERNS,
            EmailCategory,
        )

        # Check key categories have patterns
        assert EmailCategory.INVOICES in CATEGORY_PATTERNS
        assert EmailCategory.HR in CATEGORY_PATTERNS
        assert EmailCategory.NEWSLETTERS in CATEGORY_PATTERNS
        assert EmailCategory.PROJECTS in CATEGORY_PATTERNS
        assert EmailCategory.MEETINGS in CATEGORY_PATTERNS

    def test_sender_domain_categories_exist(self):
        """Test that sender domain categories are defined."""
        from aragora.services.email_categorizer import SENDER_DOMAIN_CATEGORIES

        assert "github.com" in SENDER_DOMAIN_CATEGORIES
        assert "linkedin.com" in SENDER_DOMAIN_CATEGORIES
        assert "amazon.com" in SENDER_DOMAIN_CATEGORIES
        assert "stripe.com" in SENDER_DOMAIN_CATEGORIES
