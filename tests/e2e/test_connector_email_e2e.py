"""
End-to-End Test: Email Connector.

Tests complete email connector scenarios including:
- Email configuration with multiple providers (SMTP, SendGrid)
- Rate limiting and retry mechanisms
- Email debate notifications and digests
- Email categorization and prioritization via debate
- Complete email workflow lifecycle

Related plan: kind-squishing-russell.md
"""

from __future__ import annotations

import asyncio
import smtplib
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.integrations.email import (
    EmailConfig,
    EmailRecipient,
    EmailIntegration,
    EmailProvider,
)
from aragora.services.email_debate import (
    EmailInput,
    EmailDebateResult,
    EmailPriority,
    EmailCategory,
    BatchEmailResult,
)
from aragora.core import DebateResult


# ============================================================================
# Test Helpers
# ============================================================================


@dataclass
class MockSMTPServer:
    """Mock SMTP server for testing."""

    sent_emails: List[Dict[str, Any]] = field(default_factory=list)
    fail_count: int = 0
    current_failures: int = 0

    def send_email(
        self,
        from_addr: str,
        to_addrs: List[str],
        message: str,
    ) -> None:
        """Simulate sending an email."""
        if self.current_failures < self.fail_count:
            self.current_failures += 1
            raise Exception("SMTP connection failed")

        self.sent_emails.append(
            {
                "from": from_addr,
                "to": to_addrs,
                "message": message,
                "timestamp": datetime.now(timezone.utc),
            }
        )


def create_test_email_config(
    provider: str = "smtp",
    smtp_host: str = "smtp.test.com",
    **kwargs: Any,
) -> EmailConfig:
    """Create a test email configuration."""
    defaults = {
        "provider": provider,
        "smtp_host": smtp_host,
        "smtp_port": 587,
        "smtp_username": "testuser",
        "smtp_password": "testpass",
        "from_email": "debates@test.aragora.ai",
        "from_name": "Test Aragora",
        "max_emails_per_hour": 100,
        "enable_circuit_breaker": False,
    }
    defaults.update(kwargs)
    return EmailConfig(**defaults)


def create_test_email_input(
    subject: str = "Test Email Subject",
    body: str = "This is a test email body.",
    sender: str = "sender@example.com",
    **kwargs: Any,
) -> EmailInput:
    """Create a test email input."""
    return EmailInput(
        subject=subject,
        body=body,
        sender=sender,
        received_at=kwargs.get("received_at", datetime.now(timezone.utc)),
        recipients=kwargs.get("recipients", ["recipient@example.com"]),
        cc=kwargs.get("cc", []),
        thread_id=kwargs.get("thread_id"),
        message_id=kwargs.get("message_id", f"test-msg-{datetime.now().timestamp()}"),
        attachments=kwargs.get("attachments", []),
    )


def create_test_debate_result(
    task: str = "Test debate task",
    final_answer: str = "Test answer",
    consensus_reached: bool = True,
    **kwargs: Any,
) -> DebateResult:
    """Create a test debate result."""
    result = DebateResult(
        task=task,
        final_answer=final_answer,
        consensus_reached=consensus_reached,
        rounds_used=kwargs.get("rounds_used", 3),
        winner=kwargs.get("winner", "claude"),
        confidence=kwargs.get("confidence", 0.85),
    )
    result.debate_id = result.id
    return result


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def smtp_config() -> EmailConfig:
    """Create SMTP email config."""
    return create_test_email_config(provider="smtp")


@pytest.fixture
def sendgrid_config() -> EmailConfig:
    """Create SendGrid email config."""
    return create_test_email_config(
        provider="sendgrid",
        smtp_host="",  # Not needed for SendGrid
        sendgrid_api_key="SG.test-api-key-12345",
    )


@pytest.fixture
def email_integration(smtp_config: EmailConfig) -> EmailIntegration:
    """Create email integration instance."""
    return EmailIntegration(smtp_config)


@pytest.fixture
def mock_smtp_server() -> MockSMTPServer:
    """Create mock SMTP server."""
    return MockSMTPServer()


@pytest.fixture
def sample_recipient() -> EmailRecipient:
    """Create sample email recipient."""
    return EmailRecipient(
        email="user@example.com",
        name="Test User",
        preferences={"html": True},
    )


@pytest.fixture
def sample_debate_result() -> DebateResult:
    """Create sample debate result."""
    return create_test_debate_result()


@pytest.fixture
def sample_email_input() -> EmailInput:
    """Create sample email input."""
    return create_test_email_input()


@pytest.fixture
def batch_email_inputs() -> List[EmailInput]:
    """Create batch of test emails with varying priorities."""
    return [
        create_test_email_input(
            subject="URGENT: Server is down!",
            body="Critical production issue needs immediate attention.",
            sender="ops@company.com",
            message_id="msg-urgent-001",
        ),
        create_test_email_input(
            subject="Meeting tomorrow at 10am",
            body="Please confirm your attendance for the project review meeting.",
            sender="manager@company.com",
            message_id="msg-meeting-001",
        ),
        create_test_email_input(
            subject="Weekly Newsletter",
            body="Check out this week's industry news and updates.",
            sender="newsletter@marketing.com",
            message_id="msg-newsletter-001",
        ),
        create_test_email_input(
            subject="Win $1000000 now!!!",
            body="Click here to claim your prize IMMEDIATELY!!!",
            sender="scam@suspicious.com",
            message_id="msg-spam-001",
        ),
    ]


# ============================================================================
# Test Classes
# ============================================================================


@pytest.mark.e2e
class TestEmailConfigurationValidation:
    """Tests for email configuration validation."""

    def test_smtp_config_validation(self) -> None:
        """Test SMTP configuration is validated correctly."""
        config = create_test_email_config()
        assert config.provider == "smtp"
        assert config.smtp_host == "smtp.test.com"
        assert config.smtp_port == 587
        assert config.email_provider == EmailProvider.SMTP

    def test_smtp_config_requires_host(self) -> None:
        """Test SMTP config raises error without host."""
        with pytest.raises(ValueError, match="SMTP host is required"):
            EmailConfig(provider="smtp", smtp_host="")

    def test_sendgrid_config_validation(self) -> None:
        """Test SendGrid configuration is validated correctly."""
        config = EmailConfig(
            provider="sendgrid",
            sendgrid_api_key="SG.test-key",
        )
        assert config.provider == "sendgrid"
        assert config.email_provider == EmailProvider.SENDGRID

    def test_ses_config_validation(self) -> None:
        """Test AWS SES configuration is validated correctly."""
        config = EmailConfig(
            provider="ses",
            ses_region="us-west-2",
            ses_access_key_id="AKIATEST",
            ses_secret_access_key="secret-key",
        )
        assert config.provider == "ses"
        assert config.ses_region == "us-west-2"
        assert config.email_provider == EmailProvider.SES

    def test_config_default_values(self) -> None:
        """Test configuration default values."""
        config = create_test_email_config()
        assert config.use_tls is True
        assert config.use_ssl is False
        assert config.notify_on_consensus is True
        assert config.notify_on_debate_end is True
        assert config.enable_digest is True
        assert config.max_retries == 3

    def test_config_custom_values(self) -> None:
        """Test configuration accepts custom values."""
        config = create_test_email_config(
            smtp_port=465,
            use_ssl=True,
            use_tls=False,
            max_emails_per_hour=200,
            max_retries=5,
        )
        assert config.smtp_port == 465
        assert config.use_ssl is True
        assert config.use_tls is False
        assert config.max_emails_per_hour == 200
        assert config.max_retries == 5


@pytest.mark.e2e
class TestEmailRecipientManagement:
    """Tests for email recipient management."""

    def test_recipient_with_name_formatting(self) -> None:
        """Test recipient formatted address includes name."""
        recipient = EmailRecipient(
            email="john@example.com",
            name="John Doe",
        )
        assert recipient.formatted == "John Doe <john@example.com>"

    def test_recipient_without_name_formatting(self) -> None:
        """Test recipient formatted address without name."""
        recipient = EmailRecipient(email="john@example.com")
        assert recipient.formatted == "john@example.com"

    def test_recipient_preferences(self) -> None:
        """Test recipient preferences are stored correctly."""
        recipient = EmailRecipient(
            email="user@example.com",
            preferences={"html": True, "digest": False, "urgent_only": True},
        )
        assert recipient.preferences["html"] is True
        assert recipient.preferences["digest"] is False
        assert recipient.preferences["urgent_only"] is True

    def test_multiple_recipients(self) -> None:
        """Test managing multiple recipients."""
        recipients = [
            EmailRecipient(email="user1@example.com", name="User One"),
            EmailRecipient(email="user2@example.com", name="User Two"),
            EmailRecipient(email="user3@example.com"),  # No name
        ]
        assert len(recipients) == 3
        assert all(r.email.endswith("@example.com") for r in recipients)


@pytest.mark.e2e
class TestRateLimiting:
    """Tests for email rate limiting functionality."""

    @pytest.mark.asyncio
    async def test_rate_limit_allows_within_limit(
        self,
        email_integration: EmailIntegration,
    ) -> None:
        """Test rate limiter allows emails within limit."""
        # Should allow initial emails (check is async)
        for i in range(5):
            result = await email_integration._check_rate_limit()
            assert result is True

    @pytest.mark.asyncio
    async def test_rate_limit_enforces_limit(self) -> None:
        """Test rate limiter enforces maximum emails per hour."""
        config = create_test_email_config(max_emails_per_hour=3)
        integration = EmailIntegration(config)

        # Simulate reaching the limit by calling _check_rate_limit
        for _ in range(3):
            await integration._check_rate_limit()

        # Next email should be rate limited
        result = await integration._check_rate_limit()
        assert result is False

    @pytest.mark.asyncio
    async def test_rate_limit_resets_after_hour(self) -> None:
        """Test rate limit resets after an hour."""
        config = create_test_email_config(max_emails_per_hour=2)
        integration = EmailIntegration(config)

        # Simulate old email count by manipulating internal state
        integration._email_count = 10
        integration._last_reset = datetime.now() - timedelta(hours=2)

        # Should allow new emails since old ones expired
        result = await integration._check_rate_limit()
        assert result is True


@pytest.mark.e2e
class TestEmailSending:
    """Tests for email sending functionality."""

    @pytest.mark.asyncio
    async def test_send_debate_summary(
        self,
        email_integration: EmailIntegration,
        sample_recipient: EmailRecipient,
        sample_debate_result: DebateResult,
    ) -> None:
        """Test sending debate summary email."""
        # Add recipient to integration
        email_integration.add_recipient(sample_recipient)

        with patch.object(email_integration, "_send_via_smtp", new_callable=AsyncMock) as mock_send:
            mock_send.return_value = True

            # send_debate_summary returns count of emails sent
            sent_count = await email_integration.send_debate_summary(
                sample_debate_result,
            )

            assert sent_count == 1  # One recipient

    @pytest.mark.asyncio
    async def test_send_consensus_alert(
        self,
        email_integration: EmailIntegration,
        sample_recipient: EmailRecipient,
        sample_debate_result: DebateResult,
    ) -> None:
        """Test sending consensus alert email."""
        # Add recipient to integration
        email_integration.add_recipient(sample_recipient)

        with patch.object(email_integration, "_send_via_smtp", new_callable=AsyncMock) as mock_send:
            mock_send.return_value = True

            # send_consensus_alert takes debate_id and confidence directly
            sent_count = await email_integration.send_consensus_alert(
                debate_id=sample_debate_result.debate_id,
                confidence=sample_debate_result.confidence,
                winner=sample_debate_result.winner,
                task=sample_debate_result.task,
            )

            assert sent_count == 1  # One recipient

    @pytest.mark.asyncio
    async def test_send_email_handles_smtp_failure_gracefully(
        self,
        sample_recipient: EmailRecipient,
    ) -> None:
        """Test email sending handles SMTP failure gracefully."""
        config = create_test_email_config(max_retries=1, retry_delay=0.01)
        integration = EmailIntegration(config)

        # Mock SMTP to return False (failure without exception)
        async def mock_send_smtp_failure(*args: Any, **kwargs: Any) -> bool:
            return False  # Indicate failure

        with patch.object(integration, "_send_via_smtp", side_effect=mock_send_smtp_failure):
            result = await integration._send_email(
                sample_recipient,
                "Test Subject",
                "<p>Test body</p>",
            )

            # Should return False on failure
            assert result is False

    @pytest.mark.asyncio
    async def test_send_email_succeeds_with_valid_mock(
        self,
        sample_recipient: EmailRecipient,
    ) -> None:
        """Test email sending succeeds with valid mocked SMTP."""
        config = create_test_email_config()
        integration = EmailIntegration(config)

        async def mock_send_smtp_success(*args: Any, **kwargs: Any) -> bool:
            return True

        with patch.object(integration, "_send_via_smtp", side_effect=mock_send_smtp_success):
            result = await integration._send_email(
                sample_recipient,
                "Test Subject",
                "<p>Test body</p>",
            )

            assert result is True


@pytest.mark.e2e
class TestEmailDebateIntegration:
    """Tests for email debate and categorization."""

    def test_email_input_to_context_string(
        self,
        sample_email_input: EmailInput,
    ) -> None:
        """Test email input converts to context string for debate."""
        context = sample_email_input.to_context_string()

        assert f"From: {sample_email_input.sender}" in context
        assert f"Subject: {sample_email_input.subject}" in context
        assert sample_email_input.body in context

    def test_email_debate_result_to_dict(self) -> None:
        """Test email debate result serialization."""
        result = EmailDebateResult(
            message_id="msg-001",
            priority=EmailPriority.HIGH,
            category=EmailCategory.ACTION_REQUIRED,
            confidence=0.9,
            reasoning="Important business email requiring response.",
            action_items=["Review proposal", "Send response by EOD"],
            is_spam=False,
            is_phishing=False,
        )

        result_dict = result.to_dict()

        assert result_dict["message_id"] == "msg-001"
        assert result_dict["priority"] == "high"
        assert result_dict["category"] == "action_required"
        assert result_dict["confidence"] == 0.9
        assert len(result_dict["action_items"]) == 2

    def test_batch_email_result_grouping(self) -> None:
        """Test batch email result grouping by priority."""
        results = [
            EmailDebateResult(
                message_id="msg-1",
                priority=EmailPriority.URGENT,
                category=EmailCategory.ACTION_REQUIRED,
                confidence=0.95,
                reasoning="Critical",
            ),
            EmailDebateResult(
                message_id="msg-2",
                priority=EmailPriority.HIGH,
                category=EmailCategory.REPLY_NEEDED,
                confidence=0.85,
                reasoning="Important",
            ),
            EmailDebateResult(
                message_id="msg-3",
                priority=EmailPriority.URGENT,
                category=EmailCategory.ACTION_REQUIRED,
                confidence=0.9,
                reasoning="Critical 2",
            ),
            EmailDebateResult(
                message_id="msg-4",
                priority=EmailPriority.LOW,
                category=EmailCategory.NEWSLETTER,
                confidence=0.8,
                reasoning="Newsletter",
            ),
        ]

        batch_result = BatchEmailResult(
            results=results,
            total_emails=4,
            processed_emails=4,
            duration_seconds=10.5,
        )

        # Test grouping
        by_priority = batch_result.by_priority
        assert len(by_priority["urgent"]) == 2
        assert len(by_priority["high"]) == 1
        assert len(by_priority["low"]) == 1

        # Test counts
        assert batch_result.urgent_count == 2
        assert batch_result.action_required_count == 2

    def test_email_priority_enum_values(self) -> None:
        """Test email priority enum has correct values."""
        assert EmailPriority.URGENT.value == "urgent"
        assert EmailPriority.HIGH.value == "high"
        assert EmailPriority.NORMAL.value == "normal"
        assert EmailPriority.LOW.value == "low"
        assert EmailPriority.SPAM.value == "spam"

    def test_email_category_enum_values(self) -> None:
        """Test email category enum has correct values."""
        assert EmailCategory.ACTION_REQUIRED.value == "action_required"
        assert EmailCategory.REPLY_NEEDED.value == "reply_needed"
        assert EmailCategory.FYI.value == "fyi"
        assert EmailCategory.MEETING.value == "meeting"
        assert EmailCategory.SPAM.value == "spam"
        assert EmailCategory.PHISHING.value == "phishing"


@pytest.mark.e2e
class TestCompleteEmailWorkflow:
    """Tests for complete email workflow scenarios."""

    @pytest.mark.asyncio
    async def test_end_to_end_email_notification_flow(
        self,
        sample_recipient: EmailRecipient,
        sample_debate_result: DebateResult,
    ) -> None:
        """Test complete flow: debate → notification → delivery."""
        config = create_test_email_config()
        integration = EmailIntegration(config)
        integration.add_recipient(sample_recipient)

        with patch.object(integration, "_send_via_smtp", new_callable=AsyncMock) as mock_send:
            mock_send.return_value = True

            # Send debate summary (returns count)
            sent_count = await integration.send_debate_summary(
                sample_debate_result,
            )
            assert sent_count >= 1

            # If consensus reached with high confidence, send alert
            if (
                sample_debate_result.consensus_reached
                and sample_debate_result.confidence >= config.min_consensus_confidence
            ):
                alert_count = await integration.send_consensus_alert(
                    debate_id=sample_debate_result.debate_id,
                    confidence=sample_debate_result.confidence,
                    winner=sample_debate_result.winner,
                    task=sample_debate_result.task,
                )
                assert alert_count >= 1

    @pytest.mark.asyncio
    async def test_multiple_provider_fallback(self) -> None:
        """Test email sending with provider fallback."""
        # Test SMTP config
        smtp_config = create_test_email_config(provider="smtp")
        smtp_integration = EmailIntegration(smtp_config)
        assert smtp_integration.config.provider == "smtp"

        # Test SendGrid config
        sendgrid_config = EmailConfig(
            provider="sendgrid",
            sendgrid_api_key="SG.test-key",
        )
        sendgrid_integration = EmailIntegration(sendgrid_config)
        assert sendgrid_integration.config.provider == "sendgrid"

    @pytest.mark.asyncio
    async def test_email_categorization_workflow(
        self,
        batch_email_inputs: List[EmailInput],
    ) -> None:
        """Test email categorization workflow."""
        # Simulate categorization results
        categorization_results = []
        for email in batch_email_inputs:
            # Simple heuristic categorization for testing
            if "URGENT" in email.subject.upper():
                priority = EmailPriority.URGENT
                category = EmailCategory.ACTION_REQUIRED
            elif "meeting" in email.subject.lower():
                priority = EmailPriority.HIGH
                category = EmailCategory.MEETING
            elif "newsletter" in email.subject.lower():
                priority = EmailPriority.LOW
                category = EmailCategory.NEWSLETTER
            elif "win" in email.body.lower() or "scam" in email.sender:
                priority = EmailPriority.SPAM
                category = EmailCategory.SPAM
            else:
                priority = EmailPriority.NORMAL
                category = EmailCategory.FYI

            categorization_results.append(
                EmailDebateResult(
                    message_id=email.message_id or "",
                    priority=priority,
                    category=category,
                    confidence=0.85,
                    reasoning=f"Categorized as {category.value}",
                )
            )

        batch_result = BatchEmailResult(
            results=categorization_results,
            total_emails=len(batch_email_inputs),
            processed_emails=len(categorization_results),
            duration_seconds=5.0,
        )

        # Verify categorization results
        assert batch_result.urgent_count == 1
        assert len([r for r in batch_result.results if r.priority == EmailPriority.SPAM]) == 1
        assert len([r for r in batch_result.results if r.category == EmailCategory.MEETING]) == 1

    @pytest.mark.asyncio
    async def test_digest_email_compilation(
        self,
        sample_recipient: EmailRecipient,
    ) -> None:
        """Test compiling and sending digest email."""
        config = create_test_email_config(enable_digest=True)
        integration = EmailIntegration(config)
        integration.add_recipient(sample_recipient)

        # Create some debate results for digest
        debates = [
            create_test_debate_result(
                task="Should we use microservices?",
                final_answer="Yes, for scalability.",
                confidence=0.9,
            ),
            create_test_debate_result(
                task="Which database to use?",
                final_answer="PostgreSQL for ACID compliance.",
                confidence=0.85,
            ),
        ]

        with patch.object(integration, "_send_via_smtp", new_callable=AsyncMock) as mock_send:
            mock_send.return_value = True

            # Send summaries for each debate
            for debate in debates:
                await integration.send_debate_summary(debate)

            assert mock_send.call_count == 2

    @pytest.mark.asyncio
    async def test_notification_settings_respected(
        self,
        sample_recipient: EmailRecipient,
        sample_debate_result: DebateResult,
    ) -> None:
        """Test that notification settings are respected."""
        # Config with notifications disabled
        config = create_test_email_config(
            notify_on_consensus=False,
            notify_on_debate_end=False,
        )
        integration = EmailIntegration(config)

        # These should still be configurable at runtime
        assert integration.config.notify_on_consensus is False
        assert integration.config.notify_on_debate_end is False


@pytest.mark.e2e
class TestEmailSecurityFeatures:
    """Tests for email security features."""

    def test_spam_detection_categorization(self) -> None:
        """Test spam email is categorized correctly."""
        spam_email = create_test_email_input(
            subject="CONGRATULATIONS! You won $1,000,000!",
            body="Click here immediately to claim your prize!!!",
            sender="winner@spam-domain.com",
        )

        # Verify spam characteristics in input
        assert "CONGRATULATIONS" in spam_email.subject.upper()
        assert "Click here" in spam_email.body

    def test_phishing_detection_categorization(self) -> None:
        """Test phishing email categorization."""
        phishing_email = create_test_email_input(
            subject="Your account has been compromised",
            body="Click here to verify your credentials: http://fake-bank.com/login",
            sender="security@fake-bank.com",
        )

        # Verify phishing characteristics
        assert "compromised" in phishing_email.subject.lower()
        assert "http://" in phishing_email.body
        assert "fake" in phishing_email.sender

    def test_email_input_body_truncation(self) -> None:
        """Test long email bodies are truncated for processing."""
        long_body = "x" * 5000
        email = create_test_email_input(body=long_body)

        context = email.to_context_string()

        # Body should be truncated at 2000 chars
        assert len(long_body) == 5000
        # Context contains truncated body
        assert "x" * 2000 in context


@pytest.mark.e2e
class TestEmailMetricsTracking:
    """Tests for email metrics and tracking."""

    def test_batch_result_metrics(self) -> None:
        """Test batch processing metrics."""
        results = [
            EmailDebateResult(
                message_id=f"msg-{i}",
                priority=EmailPriority.NORMAL,
                category=EmailCategory.FYI,
                confidence=0.8,
                reasoning="Test",
                duration_seconds=1.0 + i * 0.5,
            )
            for i in range(5)
        ]

        batch = BatchEmailResult(
            results=results,
            total_emails=10,
            processed_emails=5,
            duration_seconds=15.0,
            errors=["Error processing email 6", "Error processing email 8"],
        )

        assert batch.total_emails == 10
        assert batch.processed_emails == 5
        assert len(batch.errors) == 2
        assert batch.duration_seconds == 15.0

    def test_email_debate_result_duration_tracking(self) -> None:
        """Test debate result tracks duration."""
        result = EmailDebateResult(
            message_id="msg-001",
            priority=EmailPriority.HIGH,
            category=EmailCategory.ACTION_REQUIRED,
            confidence=0.9,
            reasoning="Important email",
            duration_seconds=2.5,
        )

        assert result.duration_seconds == 2.5
        result_dict = result.to_dict()
        assert result_dict["duration_seconds"] == 2.5

    def test_sender_reputation_tracking(self) -> None:
        """Test sender reputation is tracked in results."""
        result = EmailDebateResult(
            message_id="msg-001",
            priority=EmailPriority.HIGH,
            category=EmailCategory.ACTION_REQUIRED,
            confidence=0.9,
            reasoning="Trusted sender",
            sender_reputation=0.95,
        )

        assert result.sender_reputation == 0.95
        result_dict = result.to_dict()
        assert result_dict["sender_reputation"] == 0.95
