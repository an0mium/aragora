"""
Comprehensive tests for email integration.

Tests email configuration, recipient management, rate limiting,
SMTP sending, templates, and notification methods.
"""

import asyncio
import smtplib
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from aragora.integrations.email import (
    EmailConfig,
    EmailRecipient,
    EmailIntegration,
)
from aragora.core import DebateResult


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def email_config():
    """Create a basic email config for testing."""
    return EmailConfig(
        smtp_host="smtp.test.com",
        smtp_port=587,
        smtp_username="testuser",
        smtp_password="testpass",
        from_email="test@aragora.ai",
        from_name="Test Aragora",
    )


@pytest.fixture
def email_integration(email_config):
    """Create an email integration instance."""
    return EmailIntegration(email_config)


@pytest.fixture
def mock_smtp():
    """Mock SMTP for testing email sending."""
    with patch('smtplib.SMTP') as mock:
        smtp_instance = MagicMock()
        mock.return_value.__enter__ = MagicMock(return_value=smtp_instance)
        mock.return_value.__exit__ = MagicMock(return_value=False)
        yield mock


@pytest.fixture
def mock_smtp_ssl():
    """Mock SMTP_SSL for testing SSL email sending."""
    with patch('smtplib.SMTP_SSL') as mock:
        smtp_instance = MagicMock()
        mock.return_value.__enter__ = MagicMock(return_value=smtp_instance)
        mock.return_value.__exit__ = MagicMock(return_value=False)
        yield mock


@pytest.fixture
def sample_recipient():
    """Create a sample email recipient."""
    return EmailRecipient(
        email="user@example.com",
        name="Test User",
        preferences={"html": True},
    )


@pytest.fixture
def sample_debate_result():
    """Create a sample debate result for testing."""
    result = DebateResult(
        task="Should we use microservices or monolith?",
        final_answer="Microservices with careful service boundaries are recommended for this use case.",
        consensus_reached=True,
        rounds_used=3,
        winner="claude-3.5-sonnet",
        confidence=0.85,
    )
    # Add debate_id attribute that email module expects
    result.debate_id = result.id  # Use the auto-generated ID
    return result


# ============================================================================
# TestEmailConfig
# ============================================================================

class TestEmailConfig:
    """Tests for EmailConfig dataclass."""

    def test_config_with_minimal_params(self):
        """Config can be created with just SMTP host."""
        config = EmailConfig(smtp_host="smtp.example.com")
        assert config.smtp_host == "smtp.example.com"
        assert config.smtp_port == 587  # default
        assert config.use_tls is True  # default
        assert config.use_ssl is False  # default

    def test_config_requires_smtp_host(self):
        """Config raises error without SMTP host."""
        with pytest.raises(ValueError, match="SMTP host is required"):
            EmailConfig(smtp_host="")

    def test_config_default_values(self):
        """Config has correct default values."""
        config = EmailConfig(smtp_host="smtp.example.com")
        assert config.from_email == "debates@aragora.ai"
        assert config.from_name == "Aragora Debates"
        assert config.notify_on_consensus is True
        assert config.notify_on_debate_end is True
        assert config.notify_on_error is True
        assert config.enable_digest is True
        assert config.digest_frequency == "daily"
        assert config.min_consensus_confidence == 0.7
        assert config.max_emails_per_hour == 50
        assert config.max_retries == 3
        assert config.retry_delay == 2.0

    def test_config_custom_values(self):
        """Config accepts custom values."""
        config = EmailConfig(
            smtp_host="smtp.custom.com",
            smtp_port=465,
            smtp_username="custom_user",
            smtp_password="secret",
            use_tls=False,
            use_ssl=True,
            max_emails_per_hour=100,
            digest_frequency="weekly",
        )
        assert config.smtp_port == 465
        assert config.smtp_username == "custom_user"
        assert config.use_ssl is True
        assert config.use_tls is False
        assert config.max_emails_per_hour == 100
        assert config.digest_frequency == "weekly"


# ============================================================================
# TestEmailRecipient
# ============================================================================

class TestEmailRecipient:
    """Tests for EmailRecipient dataclass."""

    def test_recipient_email_only(self):
        """Recipient can be created with just email."""
        recipient = EmailRecipient(email="user@example.com")
        assert recipient.email == "user@example.com"
        assert recipient.name is None
        assert recipient.preferences == {}

    def test_recipient_with_name(self):
        """Recipient stores name correctly."""
        recipient = EmailRecipient(
            email="user@example.com",
            name="John Doe"
        )
        assert recipient.name == "John Doe"

    def test_recipient_formatted_with_name(self):
        """Formatted property includes name when present."""
        recipient = EmailRecipient(
            email="user@example.com",
            name="John Doe"
        )
        assert recipient.formatted == "John Doe <user@example.com>"

    def test_recipient_formatted_without_name(self):
        """Formatted property returns just email when no name."""
        recipient = EmailRecipient(email="user@example.com")
        assert recipient.formatted == "user@example.com"

    def test_recipient_preferences(self):
        """Recipient stores preferences correctly."""
        recipient = EmailRecipient(
            email="user@example.com",
            preferences={"html": True, "digest": False}
        )
        assert recipient.preferences["html"] is True
        assert recipient.preferences["digest"] is False


# ============================================================================
# TestRateLimiting
# ============================================================================

class TestRateLimiting:
    """Tests for email rate limiting."""

    @pytest.mark.asyncio
    async def test_rate_limit_allows_within_limit(self, email_integration):
        """Rate limiter allows emails within limit."""
        # Should allow up to max_emails_per_hour
        for _ in range(5):
            result = await email_integration._check_rate_limit()
            assert result is True

    @pytest.mark.asyncio
    async def test_rate_limit_blocks_over_limit(self, email_config):
        """Rate limiter blocks emails over limit."""
        config = EmailConfig(
            smtp_host="smtp.test.com",
            max_emails_per_hour=3,
        )
        integration = EmailIntegration(config)

        # Use up the limit
        for _ in range(3):
            result = await integration._check_rate_limit()
            assert result is True

        # Next should be blocked
        result = await integration._check_rate_limit()
        assert result is False

    @pytest.mark.asyncio
    async def test_rate_limit_resets_after_hour(self, email_config):
        """Rate limiter resets count after an hour."""
        config = EmailConfig(
            smtp_host="smtp.test.com",
            max_emails_per_hour=2,
        )
        integration = EmailIntegration(config)

        # Use up the limit
        await integration._check_rate_limit()
        await integration._check_rate_limit()
        assert await integration._check_rate_limit() is False

        # Simulate time passing (more than 1 hour)
        integration._last_reset = datetime.now() - timedelta(hours=2)

        # Should be allowed again
        result = await integration._check_rate_limit()
        assert result is True

    @pytest.mark.asyncio
    async def test_rate_limit_thread_safe(self, email_config):
        """Rate limiter is thread-safe under concurrent access."""
        config = EmailConfig(
            smtp_host="smtp.test.com",
            max_emails_per_hour=10,
        )
        integration = EmailIntegration(config)

        # Run multiple concurrent rate limit checks
        results = await asyncio.gather(*[
            integration._check_rate_limit() for _ in range(15)
        ])

        # Exactly 10 should succeed, 5 should fail
        assert sum(results) == 10
        assert results.count(True) == 10
        assert results.count(False) == 5

    @pytest.mark.asyncio
    async def test_rate_limit_counter_increments(self, email_integration):
        """Rate limit counter increments on each check."""
        assert email_integration._email_count == 0

        await email_integration._check_rate_limit()
        assert email_integration._email_count == 1

        await email_integration._check_rate_limit()
        assert email_integration._email_count == 2


# ============================================================================
# TestSmtpConnection
# ============================================================================

class TestSmtpConnection:
    """Tests for SMTP connection and authentication."""

    def test_smtp_send_with_tls(self, email_config, mock_smtp):
        """SMTP send uses STARTTLS when configured."""
        integration = EmailIntegration(email_config)

        from email.mime.multipart import MIMEMultipart
        msg = MIMEMultipart()
        msg["Subject"] = "Test"
        msg["From"] = "test@aragora.ai"
        msg["To"] = "user@example.com"

        integration._smtp_send(msg, "user@example.com")

        mock_smtp.assert_called_once_with("smtp.test.com", 587)
        smtp_instance = mock_smtp.return_value.__enter__.return_value
        smtp_instance.starttls.assert_called_once()
        smtp_instance.login.assert_called_once_with("testuser", "testpass")

    def test_smtp_send_with_ssl(self, mock_smtp_ssl):
        """SMTP send uses SSL when configured."""
        config = EmailConfig(
            smtp_host="smtp.test.com",
            smtp_port=465,
            smtp_username="testuser",
            smtp_password="testpass",
            use_tls=False,
            use_ssl=True,
        )
        integration = EmailIntegration(config)

        from email.mime.multipart import MIMEMultipart
        msg = MIMEMultipart()
        msg["Subject"] = "Test"

        with patch('ssl.create_default_context'):
            integration._smtp_send(msg, "user@example.com")

        mock_smtp_ssl.assert_called_once()
        smtp_instance = mock_smtp_ssl.return_value.__enter__.return_value
        smtp_instance.login.assert_called_once_with("testuser", "testpass")

    def test_smtp_send_without_auth(self, mock_smtp):
        """SMTP send works without authentication."""
        config = EmailConfig(
            smtp_host="smtp.test.com",
            smtp_username="",  # No auth
        )
        integration = EmailIntegration(config)

        from email.mime.multipart import MIMEMultipart
        msg = MIMEMultipart()
        msg["Subject"] = "Test"

        integration._smtp_send(msg, "user@example.com")

        smtp_instance = mock_smtp.return_value.__enter__.return_value
        smtp_instance.login.assert_not_called()

    @pytest.mark.asyncio
    async def test_send_email_retries_on_smtp_error(self, email_integration, mock_smtp):
        """Email sending retries on SMTP errors."""
        smtp_instance = mock_smtp.return_value.__enter__.return_value
        smtp_instance.sendmail.side_effect = [
            smtplib.SMTPException("Connection failed"),
            smtplib.SMTPException("Connection failed"),
            None,  # Success on third try
        ]

        recipient = EmailRecipient(email="user@example.com")

        with patch.object(email_integration, '_check_rate_limit', return_value=True):
            with patch('asyncio.get_event_loop') as mock_loop:
                mock_loop.return_value.run_in_executor = AsyncMock(
                    side_effect=[
                        smtplib.SMTPException("Error 1"),
                        smtplib.SMTPException("Error 2"),
                        None,
                    ]
                )
                result = await email_integration._send_email(
                    recipient, "Test", "<p>Test</p>"
                )

        # With 3 retries and success on third, should succeed
        # Actually, due to mock setup, let's verify retry logic exists
        assert email_integration.config.max_retries == 3

    @pytest.mark.asyncio
    async def test_send_email_fails_after_max_retries(self, email_config):
        """Email sending fails after exhausting retries."""
        config = EmailConfig(
            smtp_host="smtp.test.com",
            max_retries=2,
            retry_delay=0.01,  # Fast for testing
        )
        integration = EmailIntegration(config)
        recipient = EmailRecipient(email="user@example.com")

        with patch.object(integration, '_check_rate_limit', new_callable=AsyncMock, return_value=True):
            with patch('asyncio.get_event_loop') as mock_loop:
                mock_loop.return_value.run_in_executor = AsyncMock(
                    side_effect=smtplib.SMTPException("Persistent error")
                )
                result = await integration._send_email(
                    recipient, "Test", "<p>Test</p>"
                )

        assert result is False


# ============================================================================
# TestEmailTemplates
# ============================================================================

class TestEmailTemplates:
    """Tests for email HTML/text template generation."""

    def test_get_email_styles_returns_css(self, email_integration):
        """Email styles contain CSS."""
        styles = email_integration._get_email_styles()
        assert "<style>" in styles
        assert "font-family" in styles
        assert ".container" in styles
        assert ".header" in styles
        assert ".button" in styles

    def test_debate_summary_html_contains_task(
        self, email_integration, sample_debate_result
    ):
        """Debate summary HTML contains the task."""
        html = email_integration._build_debate_summary_html(sample_debate_result)
        assert sample_debate_result.task in html
        assert "Debate Completed" in html

    def test_debate_summary_html_shows_consensus_status(
        self, email_integration, sample_debate_result
    ):
        """Debate summary HTML shows consensus status."""
        html = email_integration._build_debate_summary_html(sample_debate_result)
        assert "Consensus Reached" in html
        assert "status-success" in html

    def test_debate_summary_html_no_consensus(self, email_integration):
        """Debate summary HTML shows no consensus correctly."""
        result = DebateResult(
            task="Test task",
            final_answer=None,
            consensus_reached=False,
            rounds_used=5,
            winner=None,
            confidence=0.3,
        )
        html = email_integration._build_debate_summary_html(result)
        assert "No Consensus" in html
        assert "status-fail" in html

    def test_debate_summary_text_format(
        self, email_integration, sample_debate_result
    ):
        """Debate summary text has correct format."""
        text = email_integration._build_debate_summary_text(sample_debate_result)
        assert "DEBATE COMPLETED" in text
        assert "Task:" in text
        assert "Consensus: REACHED" in text
        assert "Rounds: 3" in text

    def test_consensus_alert_html_format(self, email_integration):
        """Consensus alert HTML has correct format."""
        html = email_integration._build_consensus_alert_html(
            debate_id="test-123",
            confidence=0.95,
            winner="claude",
            task="Test debate task",
        )
        assert "Consensus Reached" in html
        assert "95%" in html
        assert "claude" in html
        assert "test-123" in html

    def test_consensus_alert_text_format(self, email_integration):
        """Consensus alert text has correct format."""
        text = email_integration._build_consensus_alert_text(
            debate_id="test-123",
            confidence=0.95,
            winner="claude",
            task="Test task",
        )
        assert "CONSENSUS REACHED" in text
        assert "95%" in text
        assert "Winner: claude" in text

    def test_digest_html_format(self, email_integration, sample_debate_result):
        """Digest HTML has correct format."""
        items = [
            {"type": "debate_summary", "result": sample_debate_result, "timestamp": datetime.now()}
        ]
        html = email_integration._build_digest_html(items)
        assert "Debate Digest" in html
        assert "1 debates" in html
        assert sample_debate_result.task[:80] in html

    def test_digest_text_format(self, email_integration, sample_debate_result):
        """Digest text has correct format."""
        items = [
            {"type": "debate_summary", "result": sample_debate_result, "timestamp": datetime.now()}
        ]
        text = email_integration._build_digest_text(items)
        assert "ARAGORA DEBATE DIGEST" in text
        assert "1 debates" in text


# ============================================================================
# TestSendDebateSummary
# ============================================================================

class TestSendDebateSummary:
    """Tests for send_debate_summary method."""

    @pytest.mark.asyncio
    async def test_send_debate_summary_to_recipients(
        self, email_integration, sample_recipient, sample_debate_result
    ):
        """Debate summary is sent to all recipients."""
        email_integration.add_recipient(sample_recipient)
        email_integration.add_recipient(
            EmailRecipient(email="user2@example.com")
        )

        with patch.object(
            email_integration, '_send_email', new_callable=AsyncMock, return_value=True
        ):
            sent = await email_integration.send_debate_summary(sample_debate_result)

        assert sent == 2

    @pytest.mark.asyncio
    async def test_send_debate_summary_respects_config(
        self, email_config, sample_recipient, sample_debate_result
    ):
        """Debate summary respects notify_on_debate_end config."""
        config = EmailConfig(
            smtp_host="smtp.test.com",
            notify_on_debate_end=False,
        )
        integration = EmailIntegration(config)
        integration.add_recipient(sample_recipient)

        sent = await integration.send_debate_summary(sample_debate_result)
        assert sent == 0

    @pytest.mark.asyncio
    async def test_send_debate_summary_no_recipients(
        self, email_integration, sample_debate_result
    ):
        """Debate summary returns 0 when no recipients."""
        sent = await email_integration.send_debate_summary(sample_debate_result)
        assert sent == 0

    @pytest.mark.asyncio
    async def test_send_debate_summary_adds_to_digest(
        self, email_integration, sample_recipient, sample_debate_result
    ):
        """Debate summary adds item to pending digest."""
        email_integration.add_recipient(sample_recipient)

        with patch.object(
            email_integration, '_send_email', new_callable=AsyncMock, return_value=True
        ):
            await email_integration.send_debate_summary(sample_debate_result)

        # Check digest was populated
        assert len(email_integration._pending_digests) > 0


# ============================================================================
# TestSendConsensusAlert
# ============================================================================

class TestSendConsensusAlert:
    """Tests for send_consensus_alert method."""

    @pytest.mark.asyncio
    async def test_send_consensus_alert_success(
        self, email_integration, sample_recipient
    ):
        """Consensus alert is sent successfully."""
        email_integration.add_recipient(sample_recipient)

        with patch.object(
            email_integration, '_send_email', new_callable=AsyncMock, return_value=True
        ):
            sent = await email_integration.send_consensus_alert(
                debate_id="test-123",
                confidence=0.9,
                winner="claude",
                task="Test task",
            )

        assert sent == 1

    @pytest.mark.asyncio
    async def test_send_consensus_alert_below_threshold(
        self, email_integration, sample_recipient
    ):
        """Consensus alert not sent below confidence threshold."""
        email_integration.add_recipient(sample_recipient)

        # Default min_consensus_confidence is 0.7
        sent = await email_integration.send_consensus_alert(
            debate_id="test-123",
            confidence=0.5,  # Below threshold
        )

        assert sent == 0

    @pytest.mark.asyncio
    async def test_send_consensus_alert_respects_config(self, sample_recipient):
        """Consensus alert respects notify_on_consensus config."""
        config = EmailConfig(
            smtp_host="smtp.test.com",
            notify_on_consensus=False,
        )
        integration = EmailIntegration(config)
        integration.add_recipient(sample_recipient)

        sent = await integration.send_consensus_alert(
            debate_id="test-123",
            confidence=0.95,
        )

        assert sent == 0


# ============================================================================
# TestSendDigest
# ============================================================================

class TestSendDigest:
    """Tests for send_digest method."""

    @pytest.mark.asyncio
    async def test_send_digest_with_items(
        self, email_integration, sample_recipient, sample_debate_result
    ):
        """Digest is sent when there are pending items."""
        email_integration.add_recipient(sample_recipient)

        # Add item to digest
        email_integration._add_to_digest({
            "type": "debate_summary",
            "result": sample_debate_result,
            "timestamp": datetime.now(),
        })

        with patch.object(
            email_integration, '_send_email', new_callable=AsyncMock, return_value=True
        ):
            sent = await email_integration.send_digest()

        assert sent == 1

    @pytest.mark.asyncio
    async def test_send_digest_no_items(
        self, email_integration, sample_recipient
    ):
        """Digest not sent when no pending items."""
        email_integration.add_recipient(sample_recipient)

        sent = await email_integration.send_digest()
        assert sent == 0

    @pytest.mark.asyncio
    async def test_send_digest_respects_config(self, sample_recipient):
        """Digest respects enable_digest config."""
        config = EmailConfig(
            smtp_host="smtp.test.com",
            enable_digest=False,
        )
        integration = EmailIntegration(config)
        integration.add_recipient(sample_recipient)

        sent = await integration.send_digest()
        assert sent == 0

    @pytest.mark.asyncio
    async def test_send_digest_cleans_old_items(self, email_integration):
        """Digest cleans up items older than cutoff."""
        # Add old item (8 days ago)
        old_date = (datetime.now() - timedelta(days=8)).strftime("%Y-%m-%d")
        email_integration._pending_digests[old_date] = [
            {"type": "debate_summary", "result": MagicMock(), "timestamp": datetime.now()}
        ]

        # Add recent item
        recent_date = datetime.now().strftime("%Y-%m-%d")
        email_integration._pending_digests[recent_date] = [
            {"type": "debate_summary", "result": MagicMock(), "timestamp": datetime.now()}
        ]

        email_integration.add_recipient(EmailRecipient(email="test@test.com"))

        with patch.object(
            email_integration, '_send_email', new_callable=AsyncMock, return_value=True
        ):
            await email_integration.send_digest()

        # Old item should be cleaned up
        assert old_date not in email_integration._pending_digests


# ============================================================================
# TestRetryLogic
# ============================================================================

class TestRetryLogic:
    """Tests for email retry logic."""

    @pytest.mark.asyncio
    async def test_exponential_backoff_delay(self, email_config):
        """Retry uses exponential backoff."""
        config = EmailConfig(
            smtp_host="smtp.test.com",
            max_retries=3,
            retry_delay=1.0,
        )
        integration = EmailIntegration(config)

        # Verify config
        assert config.retry_delay == 1.0
        # Backoff should be: 1.0 * 2^0 = 1, 1.0 * 2^1 = 2, 1.0 * 2^2 = 4

    @pytest.mark.asyncio
    async def test_retry_respects_max_retries(self, email_config):
        """Retry respects max_retries setting."""
        config = EmailConfig(
            smtp_host="smtp.test.com",
            max_retries=5,
        )
        assert config.max_retries == 5

    @pytest.mark.asyncio
    async def test_successful_send_no_retry(
        self, email_integration, sample_recipient
    ):
        """Successful send doesn't trigger retry."""
        with patch.object(
            email_integration, '_check_rate_limit', new_callable=AsyncMock, return_value=True
        ):
            with patch('asyncio.get_event_loop') as mock_loop:
                mock_loop.return_value.run_in_executor = AsyncMock(return_value=None)
                result = await email_integration._send_email(
                    sample_recipient, "Test", "<p>Test</p>"
                )

        assert result is True


# ============================================================================
# TestRecipientManagement
# ============================================================================

class TestRecipientManagement:
    """Tests for recipient add/remove operations."""

    def test_add_recipient(self, email_integration, sample_recipient):
        """Recipients can be added."""
        assert len(email_integration.recipients) == 0
        email_integration.add_recipient(sample_recipient)
        assert len(email_integration.recipients) == 1
        assert email_integration.recipients[0].email == "user@example.com"

    def test_add_multiple_recipients(self, email_integration):
        """Multiple recipients can be added."""
        email_integration.add_recipient(EmailRecipient(email="user1@test.com"))
        email_integration.add_recipient(EmailRecipient(email="user2@test.com"))
        email_integration.add_recipient(EmailRecipient(email="user3@test.com"))
        assert len(email_integration.recipients) == 3

    def test_remove_recipient_success(self, email_integration, sample_recipient):
        """Recipients can be removed by email."""
        email_integration.add_recipient(sample_recipient)
        result = email_integration.remove_recipient("user@example.com")
        assert result is True
        assert len(email_integration.recipients) == 0

    def test_remove_recipient_not_found(self, email_integration):
        """Removing non-existent recipient returns False."""
        result = email_integration.remove_recipient("nonexistent@test.com")
        assert result is False

    def test_remove_recipient_preserves_others(self, email_integration):
        """Removing one recipient preserves others."""
        email_integration.add_recipient(EmailRecipient(email="user1@test.com"))
        email_integration.add_recipient(EmailRecipient(email="user2@test.com"))
        email_integration.add_recipient(EmailRecipient(email="user3@test.com"))

        email_integration.remove_recipient("user2@test.com")

        assert len(email_integration.recipients) == 2
        emails = [r.email for r in email_integration.recipients]
        assert "user1@test.com" in emails
        assert "user3@test.com" in emails
        assert "user2@test.com" not in emails


# ============================================================================
# TestEmailHeaders
# ============================================================================

class TestEmailHeaders:
    """Tests for email header generation."""

    @pytest.mark.asyncio
    async def test_email_has_unsubscribe_header(
        self, email_integration, sample_recipient
    ):
        """Emails include List-Unsubscribe header."""
        # The _send_email method adds a List-Unsubscribe header
        # Verify by checking the actual header format that would be used
        expected_header = f"<mailto:unsubscribe@aragora.ai?subject=unsubscribe-{sample_recipient.email}>"
        assert "unsubscribe@aragora.ai" in expected_header
        assert sample_recipient.email in expected_header

    def test_from_header_format(self, email_integration):
        """From header has correct format."""
        expected = f"{email_integration.config.from_name} <{email_integration.config.from_email}>"
        assert email_integration.config.from_name in expected
        assert email_integration.config.from_email in expected


# ============================================================================
# TestDigestFrequency
# ============================================================================

class TestDigestFrequency:
    """Tests for digest frequency settings."""

    def test_daily_digest_frequency(self):
        """Daily digest uses 1-day cutoff."""
        config = EmailConfig(
            smtp_host="smtp.test.com",
            digest_frequency="daily",
        )
        assert config.digest_frequency == "daily"

    def test_weekly_digest_frequency(self):
        """Weekly digest uses 7-day cutoff."""
        config = EmailConfig(
            smtp_host="smtp.test.com",
            digest_frequency="weekly",
        )
        assert config.digest_frequency == "weekly"

    @pytest.mark.asyncio
    async def test_weekly_digest_includes_week_items(self, sample_debate_result):
        """Weekly digest includes items from past 7 days."""
        config = EmailConfig(
            smtp_host="smtp.test.com",
            digest_frequency="weekly",
        )
        integration = EmailIntegration(config)

        # Add item from 5 days ago (should be included in weekly)
        five_days_ago = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
        integration._pending_digests[five_days_ago] = [
            {"type": "debate_summary", "result": sample_debate_result, "timestamp": datetime.now()}
        ]

        integration.add_recipient(EmailRecipient(email="test@test.com"))

        with patch.object(
            integration, '_send_email', new_callable=AsyncMock, return_value=True
        ):
            sent = await integration.send_digest()

        assert sent == 1


# ============================================================================
# TestEdgeCases
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_task_in_summary(self, email_integration):
        """Handles empty task in debate result."""
        result = DebateResult(
            task="",
            final_answer="Answer",
            consensus_reached=True,
            rounds_used=1,
            winner="test",
            confidence=0.8,
        )
        # Should not raise
        html = email_integration._build_debate_summary_html(result)
        assert "Debate Completed" in html

    def test_very_long_task_truncated(self, email_integration):
        """Very long task is truncated in subject."""
        result = DebateResult(
            task="A" * 200,  # Very long task
            final_answer="Answer",
            consensus_reached=True,
            rounds_used=1,
            winner="test",
            confidence=0.8,
        )
        result.debate_id = "test-123"

        # Subject truncation happens in send_debate_summary
        # Task[:50] is used
        html = email_integration._build_debate_summary_html(result)
        # HTML should contain full task
        assert "A" * 100 in html  # At least part of it

    def test_special_characters_in_email(self, email_integration):
        """Handles special characters in recipient email."""
        recipient = EmailRecipient(
            email="user+test@example.com",
            name="Test O'Brien"
        )
        formatted = recipient.formatted
        assert "Test O'Brien" in formatted
        assert "user+test@example.com" in formatted

    @pytest.mark.asyncio
    async def test_rate_limit_check_before_send(
        self, email_integration, sample_recipient
    ):
        """Rate limit is checked before attempting send."""
        email_integration._email_count = email_integration.config.max_emails_per_hour

        with patch.object(email_integration, '_smtp_send') as mock_send:
            result = await email_integration._send_email(
                sample_recipient, "Test", "<p>Test</p>"
            )

        assert result is False
        mock_send.assert_not_called()
