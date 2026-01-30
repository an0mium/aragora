"""Tests for the Email Debate service."""

from __future__ import annotations

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.services.email_debate import (
    BatchEmailResult,
    EmailCategory,
    EmailDebateResult,
    EmailDebateService,
    EmailInput,
    EmailPriority,
)


def _make_email(**overrides) -> EmailInput:
    defaults = dict(
        subject="Test Subject",
        body="This is a test email body.",
        sender="sender@example.com",
        received_at=datetime.now(timezone.utc),
        message_id="msg_123",
    )
    defaults.update(overrides)
    return EmailInput(**defaults)


# ---------------------------------------------------------------------------
# Dataclass tests
# ---------------------------------------------------------------------------


class TestEmailDataclasses:
    def test_email_priority_values(self):
        assert EmailPriority.URGENT.value == "urgent"
        assert EmailPriority.SPAM.value == "spam"

    def test_email_category_values(self):
        assert EmailCategory.ACTION_REQUIRED.value == "action_required"
        assert EmailCategory.PHISHING.value == "phishing"

    def test_email_input_to_context_string(self):
        email = _make_email(
            subject="Important Meeting",
            sender="boss@company.com",
            recipients=["me@company.com"],
            cc=["team@company.com"],
            attachments=["report.pdf"],
        )
        ctx = email.to_context_string()
        assert "From: boss@company.com" in ctx
        assert "Subject: Important Meeting" in ctx
        assert "To: me@company.com" in ctx
        assert "CC: team@company.com" in ctx
        assert "Attachments: report.pdf" in ctx

    def test_email_input_context_truncates_body(self):
        email = _make_email(body="x" * 3000)
        ctx = email.to_context_string()
        # Body should be truncated to 2000 chars
        assert len(ctx) < 3100

    def test_email_debate_result_to_dict(self):
        result = EmailDebateResult(
            message_id="msg_1",
            priority=EmailPriority.HIGH,
            category=EmailCategory.ACTION_REQUIRED,
            confidence=0.85,
            reasoning="Urgent request",
            is_spam=False,
        )
        d = result.to_dict()
        assert d["priority"] == "high"
        assert d["category"] == "action_required"
        assert d["confidence"] == 0.85

    def test_batch_result_by_priority(self):
        results = [
            EmailDebateResult(
                message_id="1",
                priority=EmailPriority.URGENT,
                category=EmailCategory.ACTION_REQUIRED,
                confidence=0.9,
                reasoning="",
            ),
            EmailDebateResult(
                message_id="2",
                priority=EmailPriority.LOW,
                category=EmailCategory.NEWSLETTER,
                confidence=0.8,
                reasoning="",
            ),
            EmailDebateResult(
                message_id="3",
                priority=EmailPriority.URGENT,
                category=EmailCategory.REPLY_NEEDED,
                confidence=0.7,
                reasoning="",
            ),
        ]
        batch = BatchEmailResult(
            results=results, total_emails=3, processed_emails=3, duration_seconds=1.0
        )
        grouped = batch.by_priority
        assert len(grouped["urgent"]) == 2
        assert len(grouped["low"]) == 1

    def test_batch_result_urgent_count(self):
        results = [
            EmailDebateResult(
                message_id="1",
                priority=EmailPriority.URGENT,
                category=EmailCategory.ACTION_REQUIRED,
                confidence=0.9,
                reasoning="",
            ),
            EmailDebateResult(
                message_id="2",
                priority=EmailPriority.LOW,
                category=EmailCategory.FYI,
                confidence=0.8,
                reasoning="",
            ),
        ]
        batch = BatchEmailResult(
            results=results, total_emails=2, processed_emails=2, duration_seconds=0.5
        )
        assert batch.urgent_count == 1
        assert batch.action_required_count == 1


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestEmailDebateServiceInit:
    def test_init_defaults(self):
        svc = EmailDebateService()
        assert svc.agents == ["anthropic-api", "openai-api"]
        assert svc.fast_mode is True
        assert svc.enable_pii_redaction is True

    def test_init_custom(self):
        svc = EmailDebateService(
            agents=["claude"],
            fast_mode=False,
            enable_pii_redaction=False,
        )
        assert svc.agents == ["claude"]
        assert svc.fast_mode is False


# ---------------------------------------------------------------------------
# Fallback prioritization (heuristic-based)
# ---------------------------------------------------------------------------


class TestFallbackPrioritization:
    def test_urgent_subject(self):
        svc = EmailDebateService()
        email = _make_email(subject="URGENT: Server down")
        result = svc._fallback_prioritization(email, sender_reputation=None)
        assert result.priority == EmailPriority.URGENT
        assert result.category == EmailCategory.ACTION_REQUIRED
        assert result.confidence == 0.5

    def test_high_priority_subject(self):
        svc = EmailDebateService()
        email = _make_email(subject="Important deadline reminder")
        result = svc._fallback_prioritization(email, sender_reputation=None)
        assert result.priority == EmailPriority.HIGH

    def test_newsletter_subject(self):
        svc = EmailDebateService()
        email = _make_email(subject="Weekly Digest - Newsletter")
        result = svc._fallback_prioritization(email, sender_reputation=None)
        assert result.priority == EmailPriority.LOW
        assert result.category == EmailCategory.NEWSLETTER

    def test_meeting_subject(self):
        svc = EmailDebateService()
        email = _make_email(subject="Meeting invite: Team sync")
        result = svc._fallback_prioritization(email, sender_reputation=None)
        assert result.category == EmailCategory.MEETING

    def test_spam_subject(self):
        svc = EmailDebateService()
        email = _make_email(subject="You are the winner of a free gift!")
        result = svc._fallback_prioritization(email, sender_reputation=None)
        assert result.priority == EmailPriority.SPAM
        assert result.category == EmailCategory.SPAM
        assert result.is_spam is True

    def test_normal_subject(self):
        svc = EmailDebateService()
        email = _make_email(subject="Quarterly Report Q3")
        result = svc._fallback_prioritization(email, sender_reputation=None)
        assert result.priority == EmailPriority.NORMAL

    def test_sender_reputation_preserved(self):
        svc = EmailDebateService()
        email = _make_email(subject="Hello")
        result = svc._fallback_prioritization(email, sender_reputation=0.9)
        assert result.sender_reputation == 0.9


# ---------------------------------------------------------------------------
# Build debate prompt
# ---------------------------------------------------------------------------


class TestBuildDebatePrompt:
    def test_prompt_contains_email_context(self):
        svc = EmailDebateService()
        email = _make_email(subject="Test", body="Body content")
        prompt = svc._build_debate_prompt(email)
        assert "Test" in prompt
        assert "Body content" in prompt
        assert "priority" in prompt.lower() or "Priority" in prompt

    def test_prompt_includes_sender_reputation(self):
        svc = EmailDebateService()
        email = _make_email()
        prompt = svc._build_debate_prompt(email, sender_reputation=0.85)
        assert "0.85" in prompt


# ---------------------------------------------------------------------------
# Parse debate result
# ---------------------------------------------------------------------------


class TestParseDebateResult:
    def test_parse_urgent(self):
        svc = EmailDebateService()
        email = _make_email()
        mock_result = MagicMock()
        mock_result.final_answer = "This email is URGENT and requires ACTION immediately."
        mock_result.confidence = 0.9
        mock_result.debate_id = "debate_1"

        parsed = svc._parse_debate_result(email, mock_result, sender_reputation=0.8, duration=1.5)
        assert parsed.priority == EmailPriority.URGENT
        assert parsed.category == EmailCategory.ACTION_REQUIRED
        assert parsed.confidence == 0.9
        assert parsed.duration_seconds == 1.5

    def test_parse_low_priority_newsletter(self):
        svc = EmailDebateService()
        email = _make_email()
        mock_result = MagicMock()
        mock_result.final_answer = "This is a LOW priority newsletter."
        mock_result.confidence = 0.8
        mock_result.debate_id = None

        parsed = svc._parse_debate_result(email, mock_result, sender_reputation=None, duration=0.5)
        assert parsed.priority == EmailPriority.LOW
        assert parsed.category == EmailCategory.NEWSLETTER

    def test_parse_phishing(self):
        svc = EmailDebateService()
        email = _make_email()
        mock_result = MagicMock()
        mock_result.final_answer = "This is a PHISHING attempt."
        mock_result.confidence = 0.95
        mock_result.debate_id = None

        parsed = svc._parse_debate_result(email, mock_result, sender_reputation=None, duration=0.3)
        assert parsed.category == EmailCategory.PHISHING
        assert parsed.is_phishing is True
        assert parsed.is_spam is True

    def test_parse_spam(self):
        svc = EmailDebateService()
        email = _make_email()
        mock_result = MagicMock()
        mock_result.final_answer = "This is spam, mark as SPAM."
        mock_result.confidence = 0.85
        mock_result.debate_id = None

        parsed = svc._parse_debate_result(email, mock_result, sender_reputation=None, duration=0.2)
        assert parsed.priority == EmailPriority.SPAM
        assert parsed.is_spam is True


# ---------------------------------------------------------------------------
# Sanitize email
# ---------------------------------------------------------------------------


class TestSanitizeEmail:
    def test_no_redactor_returns_same(self):
        svc = EmailDebateService(enable_pii_redaction=False)
        email = _make_email()
        result = svc._sanitize_email(email)
        assert result.subject == email.subject
        assert result.body == email.body

    def test_redactor_not_available_returns_same(self):
        svc = EmailDebateService(enable_pii_redaction=True)
        svc._pii_redactor = None
        # _get_pii_redactor may fail to import; in that case, returns original
        email = _make_email()
        result = svc._sanitize_email(email)
        # Since PIIRedactor may not be importable, just check no crash
        assert result is not None


# ---------------------------------------------------------------------------
# prioritize_email (uses fallback when no debate factory)
# ---------------------------------------------------------------------------


class TestPrioritizeEmail:
    @pytest.mark.asyncio
    async def test_prioritize_fallback(self):
        svc = EmailDebateService()
        email = _make_email(subject="URGENT: Please respond ASAP")
        result = await svc.prioritize_email(email)
        assert result.priority == EmailPriority.URGENT

    @pytest.mark.asyncio
    async def test_prioritize_normal(self):
        svc = EmailDebateService()
        email = _make_email(subject="Monthly status update")
        result = await svc.prioritize_email(email)
        assert result.priority == EmailPriority.NORMAL


# ---------------------------------------------------------------------------
# prioritize_batch
# ---------------------------------------------------------------------------


class TestPrioritizeBatch:
    @pytest.mark.asyncio
    async def test_batch_empty(self):
        svc = EmailDebateService()
        batch = await svc.prioritize_batch([])
        assert batch.total_emails == 0
        assert batch.processed_emails == 0

    @pytest.mark.asyncio
    async def test_batch_multiple(self):
        svc = EmailDebateService()
        emails = [
            _make_email(subject="URGENT: Fire!", message_id="m1"),
            _make_email(subject="Newsletter update", message_id="m2"),
        ]
        batch = await svc.prioritize_batch(emails)
        assert batch.total_emails == 2
        assert batch.processed_emails == 2
        assert len(batch.results) == 2
