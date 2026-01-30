"""Tests for Email integration."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.integrations.email import (
    EmailConfig,
    EmailIntegration,
    EmailProvider,
    EmailRecipient,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def smtp_config():
    return EmailConfig(smtp_host="smtp.example.com", smtp_username="user", smtp_password="pass")


@pytest.fixture
def sendgrid_config():
    return EmailConfig(provider="sendgrid", sendgrid_api_key="SG.test_key")


@pytest.fixture
def ses_config():
    return EmailConfig(
        provider="ses",
        ses_access_key_id="AKIA_TEST",
        ses_secret_access_key="secret_test",
        ses_region="us-east-1",
    )


@pytest.fixture
def recipient():
    return EmailRecipient(email="user@example.com", name="Test User")


@pytest.fixture
def integration(smtp_config):
    return EmailIntegration(smtp_config)


def _make_debate_result(**kwargs):
    """Create a mock DebateResult."""
    result = MagicMock()
    result.task = kwargs.get("task", "Design a rate limiter")
    result.final_answer = kwargs.get("final_answer", "Use token bucket algorithm")
    result.consensus_reached = kwargs.get("consensus_reached", True)
    result.confidence = kwargs.get("confidence", 0.85)
    result.rounds_used = kwargs.get("rounds_used", 3)
    result.winner = kwargs.get("winner", "claude")
    result.debate_id = kwargs.get("debate_id", "debate-abc123")
    result.participants = kwargs.get("participants", ["claude", "gpt4"])
    return result


# =============================================================================
# EmailConfig Tests
# =============================================================================


class TestEmailConfig:
    def test_smtp_defaults(self):
        cfg = EmailConfig(smtp_host="smtp.example.com")
        assert cfg.provider == "smtp"
        assert cfg.smtp_port == 587
        assert cfg.use_tls is True
        assert cfg.from_email == "debates@aragora.ai"
        assert cfg.max_emails_per_hour == 50

    def test_smtp_requires_host(self):
        with pytest.raises(ValueError, match="SMTP host is required"):
            EmailConfig(provider="smtp")

    def test_auto_detect_sendgrid(self):
        cfg = EmailConfig(sendgrid_api_key="SG.test")
        assert cfg.provider == "sendgrid"

    def test_auto_detect_ses(self):
        cfg = EmailConfig(ses_access_key_id="AKIA_TEST", ses_secret_access_key="secret")
        assert cfg.provider == "ses"

    def test_email_provider_property(self):
        cfg = EmailConfig(provider="sendgrid", sendgrid_api_key="SG.test")
        assert cfg.email_provider == EmailProvider.SENDGRID

    def test_circuit_breaker_settings(self):
        cfg = EmailConfig(smtp_host="smtp.test.com")
        assert cfg.enable_circuit_breaker is True
        assert cfg.circuit_breaker_threshold == 5
        assert cfg.circuit_breaker_cooldown == 60.0


# =============================================================================
# EmailRecipient Tests
# =============================================================================


class TestEmailRecipient:
    def test_formatted_with_name(self):
        r = EmailRecipient(email="user@example.com", name="John Doe")
        assert r.formatted == "John Doe <user@example.com>"

    def test_formatted_without_name(self):
        r = EmailRecipient(email="user@example.com")
        assert r.formatted == "user@example.com"

    def test_default_preferences(self):
        r = EmailRecipient(email="user@example.com")
        assert r.preferences == {}


# =============================================================================
# EmailProvider Tests
# =============================================================================


class TestEmailProvider:
    def test_enum_values(self):
        assert EmailProvider.SMTP.value == "smtp"
        assert EmailProvider.SENDGRID.value == "sendgrid"
        assert EmailProvider.SES.value == "ses"


# =============================================================================
# EmailIntegration Tests
# =============================================================================


class TestEmailIntegration:
    def test_initialization(self, integration):
        assert integration.recipients == []
        assert integration._email_count == 0

    def test_add_recipient(self, integration, recipient):
        integration.add_recipient(recipient)
        assert len(integration.recipients) == 1
        assert integration.recipients[0].email == "user@example.com"

    def test_remove_recipient(self, integration, recipient):
        integration.add_recipient(recipient)
        result = integration.remove_recipient("user@example.com")
        assert result is True
        assert len(integration.recipients) == 0

    def test_remove_nonexistent_recipient(self, integration):
        result = integration.remove_recipient("nobody@example.com")
        assert result is False

    def test_get_health_status(self, integration):
        status = integration.get_health_status()
        assert status["provider"] == "smtp"
        assert status["configured"] is True
        assert status["recipients_count"] == 0
        assert status["emails_sent_this_hour"] == 0
        assert status["rate_limit"] == 50

    @pytest.mark.asyncio
    async def test_check_rate_limit_allows(self, integration):
        result = await integration._check_rate_limit()
        assert result is True
        assert integration._email_count == 1

    @pytest.mark.asyncio
    async def test_check_rate_limit_blocks(self, integration):
        integration._email_count = 50
        integration._last_reset = datetime.now()
        result = await integration._check_rate_limit()
        assert result is False

    @pytest.mark.asyncio
    async def test_check_rate_limit_resets(self, integration):
        integration._email_count = 50
        integration._last_reset = datetime.now() - timedelta(hours=2)
        result = await integration._check_rate_limit()
        assert result is True
        assert integration._email_count == 1

    def test_get_email_styles(self, integration):
        styles = integration._get_email_styles()
        assert "<style>" in styles
        assert "font-family" in styles

    def test_build_debate_summary_html(self, integration):
        result = _make_debate_result()
        html = integration._build_debate_summary_html(result)
        assert "Debate Completed" in html
        assert "Design a rate limiter" in html

    def test_build_debate_summary_text(self, integration):
        result = _make_debate_result()
        text = integration._build_debate_summary_text(result)
        assert "DEBATE COMPLETED" in text
        assert "Design a rate limiter" in text

    def test_build_debate_summary_text_no_consensus(self, integration):
        result = _make_debate_result(consensus_reached=False, final_answer="")
        text = integration._build_debate_summary_text(result)
        assert "NOT REACHED" in text

    def test_build_consensus_alert_html(self, integration):
        html = integration._build_consensus_alert_html(
            debate_id="d-123", confidence=0.9, winner="claude", task="Test"
        )
        assert "Consensus Reached" in html
        assert "90%" in html

    def test_build_consensus_alert_text(self, integration):
        text = integration._build_consensus_alert_text(
            debate_id="d-123", confidence=0.9, winner="claude", task="Test"
        )
        assert "CONSENSUS REACHED" in text
        assert "Winner: claude" in text

    @pytest.mark.asyncio
    async def test_send_debate_summary_no_recipients(self, integration):
        result = _make_debate_result()
        count = await integration.send_debate_summary(result)
        assert count == 0

    @pytest.mark.asyncio
    async def test_send_debate_summary_disabled(self, integration, recipient):
        integration.config.notify_on_debate_end = False
        integration.add_recipient(recipient)
        result = _make_debate_result()
        count = await integration.send_debate_summary(result)
        assert count == 0

    @pytest.mark.asyncio
    async def test_send_consensus_alert_below_threshold(self, integration, recipient):
        integration.add_recipient(recipient)
        count = await integration.send_consensus_alert(
            debate_id="d-123", confidence=0.5, winner="claude"
        )
        assert count == 0

    @pytest.mark.asyncio
    async def test_send_consensus_alert_disabled(self, integration, recipient):
        integration.config.notify_on_consensus = False
        integration.add_recipient(recipient)
        count = await integration.send_consensus_alert(
            debate_id="d-123", confidence=0.9, winner="claude"
        )
        assert count == 0

    @pytest.mark.asyncio
    async def test_send_digest_disabled(self, integration, recipient):
        integration.config.enable_digest = False
        integration.add_recipient(recipient)
        count = await integration.send_digest()
        assert count == 0

    @pytest.mark.asyncio
    async def test_send_digest_no_items(self, integration, recipient):
        integration.add_recipient(recipient)
        count = await integration.send_digest()
        assert count == 0

    def test_add_to_digest(self, integration):
        integration._add_to_digest({"type": "test", "data": "value"})
        assert len(integration._pending_digests) == 1

    def test_build_digest_html(self, integration):
        result = _make_debate_result()
        items = [{"type": "debate_summary", "result": result}]
        html = integration._build_digest_html(items)
        assert "Debate Digest" in html
        assert "Design a rate limiter" in html

    def test_build_digest_text(self, integration):
        result = _make_debate_result()
        items = [{"type": "debate_summary", "result": result}]
        text = integration._build_digest_text(items)
        assert "ARAGORA DEBATE DIGEST" in text

    @pytest.mark.asyncio
    async def test_context_manager(self, smtp_config):
        async with EmailIntegration(smtp_config) as email:
            assert isinstance(email, EmailIntegration)

    @pytest.mark.asyncio
    async def test_close(self, integration):
        mock_session = AsyncMock()
        mock_session.closed = False
        integration._session = mock_session
        await integration.close()
        mock_session.close.assert_called_once()
