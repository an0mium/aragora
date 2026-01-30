"""Tests for debate origin message formatting utilities.

Tests cover:
1. Result message formatting (markdown, HTML, plain text)
2. Receipt summary formatting
3. Error message formatting for chat platforms
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from aragora.server.debate_origin.formatting import (
    _format_result_message,
    _format_receipt_summary,
    format_error_for_chat,
)
from aragora.server.debate_origin.models import DebateOrigin


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_origin() -> DebateOrigin:
    """Create a sample debate origin for testing."""
    return DebateOrigin(
        debate_id="debate-123",
        platform="telegram",
        channel_id="chat-456",
        user_id="user-789",
        metadata={"topic": "Test Topic"},
    )


@pytest.fixture
def sample_result() -> dict[str, Any]:
    """Create a sample debate result for testing."""
    return {
        "consensus_reached": True,
        "final_answer": "The team reached agreement on the approach.",
        "confidence": 0.85,
        "participants": ["claude", "gpt-4", "gemini"],
        "task": "Evaluate the proposal",
    }


@dataclass
class MockReceipt:
    """Mock receipt for testing."""

    verdict: str = "APPROVED"
    confidence: float = 0.92
    critical_count: int = 0
    high_count: int = 2
    cost_usd: float | None = None
    budget_limit_usd: float | None = None


# =============================================================================
# Test: Result Message Formatting
# =============================================================================


class TestFormatResultMessage:
    """Tests for _format_result_message function."""

    def test_formats_markdown_result(self, sample_origin, sample_result):
        """_format_result_message returns markdown by default."""
        message = _format_result_message(sample_result, sample_origin)

        assert "**Debate Complete!**" in message
        assert "**Topic:** Evaluate the proposal" in message
        assert "**Consensus:** Yes" in message
        assert "**Confidence:** 85%" in message
        assert "claude, gpt-4, gemini" in message
        assert "The team reached agreement" in message

    def test_formats_html_result(self, sample_origin, sample_result):
        """_format_result_message returns HTML when requested."""
        message = _format_result_message(sample_result, sample_origin, html=True)

        assert "<h2>Debate Complete!</h2>" in message
        assert "<strong>Topic:</strong>" in message
        assert "<strong>Consensus:</strong> Yes" in message
        assert "<strong>Confidence:</strong> 85%" in message
        assert "claude, gpt-4, gemini" in message

    def test_formats_plain_text_result(self, sample_origin, sample_result):
        """_format_result_message returns plain text when markdown=False."""
        message = _format_result_message(sample_result, sample_origin, markdown=False)

        assert "Debate Complete!" in message
        assert "**" not in message  # No markdown
        assert "<" not in message  # No HTML
        assert "Topic: Evaluate the proposal" in message
        assert "Consensus: Yes" in message

    def test_no_consensus_result(self, sample_origin, sample_result):
        """_format_result_message handles no consensus case."""
        sample_result["consensus_reached"] = False
        message = _format_result_message(sample_result, sample_origin)

        assert "**Consensus:** No" in message

    def test_truncates_long_answer(self, sample_origin, sample_result):
        """_format_result_message truncates answers over 800 chars."""
        sample_result["final_answer"] = "X" * 1000
        message = _format_result_message(sample_result, sample_origin)

        assert "X" * 800 in message
        assert "..." in message
        assert "X" * 801 not in message

    def test_uses_origin_topic_as_fallback(self, sample_origin, sample_result):
        """_format_result_message uses origin metadata topic when task missing."""
        del sample_result["task"]
        message = _format_result_message(sample_result, sample_origin)

        assert "**Topic:** Test Topic" in message

    def test_handles_missing_fields(self, sample_origin):
        """_format_result_message handles missing result fields."""
        minimal_result = {}
        message = _format_result_message(minimal_result, sample_origin)

        assert "Debate Complete!" in message
        assert "**Consensus:** No" in message
        assert "**Confidence:** 0%" in message
        assert "No conclusion reached." in message

    def test_limits_participants_displayed(self, sample_origin, sample_result):
        """_format_result_message shows at most 5 participants."""
        sample_result["participants"] = [
            "agent1",
            "agent2",
            "agent3",
            "agent4",
            "agent5",
            "agent6",
            "agent7",
        ]
        message = _format_result_message(sample_result, sample_origin)

        assert "agent5" in message
        assert "agent6" not in message


# =============================================================================
# Test: Receipt Summary Formatting
# =============================================================================


class TestFormatReceiptSummary:
    """Tests for _format_receipt_summary function."""

    def test_formats_approved_receipt(self):
        """_format_receipt_summary formats APPROVED verdict."""
        receipt = MockReceipt(verdict="APPROVED", confidence=0.95)
        summary = _format_receipt_summary(receipt, "https://app.example.com/receipt/123")

        assert "\u2705" in summary  # Check mark emoji
        assert "**Decision Receipt**" in summary
        assert "Verdict: APPROVED" in summary
        assert "Confidence: 95%" in summary
        assert "[View Full Receipt]" in summary
        assert "https://app.example.com/receipt/123" in summary

    def test_formats_rejected_receipt(self):
        """_format_receipt_summary formats REJECTED verdict."""
        receipt = MockReceipt(verdict="REJECTED", confidence=0.88)
        summary = _format_receipt_summary(receipt, "https://example.com/r")

        assert "\u274c" in summary  # X emoji
        assert "Verdict: REJECTED" in summary

    def test_formats_approved_with_conditions(self):
        """_format_receipt_summary formats APPROVED_WITH_CONDITIONS verdict."""
        receipt = MockReceipt(verdict="APPROVED_WITH_CONDITIONS")
        summary = _format_receipt_summary(receipt, "https://example.com/r")

        assert "\u26a0\ufe0f" in summary  # Warning emoji

    def test_formats_needs_review(self):
        """_format_receipt_summary formats NEEDS_REVIEW verdict."""
        receipt = MockReceipt(verdict="NEEDS_REVIEW")
        summary = _format_receipt_summary(receipt, "https://example.com/r")

        assert "\U0001f50d" in summary  # Magnifying glass emoji

    def test_formats_unknown_verdict(self):
        """_format_receipt_summary handles unknown verdict."""
        receipt = MockReceipt(verdict="PENDING")
        summary = _format_receipt_summary(receipt, "https://example.com/r")

        assert "\U0001f4cb" in summary  # Clipboard emoji (default)

    def test_includes_findings_count(self):
        """_format_receipt_summary shows findings counts."""
        receipt = MockReceipt(critical_count=3, high_count=5)
        summary = _format_receipt_summary(receipt, "https://example.com/r")

        assert "3 critical" in summary
        assert "5 high" in summary

    def test_includes_cost_when_present(self):
        """_format_receipt_summary includes cost when available."""
        receipt = MockReceipt(cost_usd=0.0523)
        summary = _format_receipt_summary(receipt, "https://example.com/r")

        assert "Cost: $0.0523" in summary

    def test_includes_budget_percentage(self):
        """_format_receipt_summary shows budget percentage when both cost and limit exist."""
        receipt = MockReceipt(cost_usd=0.50, budget_limit_usd=2.00)
        summary = _format_receipt_summary(receipt, "https://example.com/r")

        assert "Cost: $0.5000" in summary
        assert "25% of budget" in summary

    def test_omits_cost_when_zero(self):
        """_format_receipt_summary omits cost when zero."""
        receipt = MockReceipt(cost_usd=0.0)
        summary = _format_receipt_summary(receipt, "https://example.com/r")

        assert "Cost:" not in summary

    def test_omits_cost_when_none(self):
        """_format_receipt_summary omits cost when None."""
        receipt = MockReceipt()
        summary = _format_receipt_summary(receipt, "https://example.com/r")

        assert "Cost:" not in summary

    def test_handles_invalid_cost_value(self):
        """_format_receipt_summary handles non-numeric cost."""
        receipt = MockReceipt()
        receipt.cost_usd = "invalid"  # type: ignore
        summary = _format_receipt_summary(receipt, "https://example.com/r")

        assert "Cost:" not in summary


# =============================================================================
# Test: Error Message Formatting
# =============================================================================


class TestFormatErrorForChat:
    """Tests for format_error_for_chat function."""

    def test_formats_rate_limit_error(self):
        """format_error_for_chat handles rate limit errors."""
        message = format_error_for_chat("rate limit exceeded", "debate-123")

        assert "processed" in message.lower()
        assert "Debate ID: debate-123" in message

    def test_formats_429_error(self):
        """format_error_for_chat handles 429 status."""
        message = format_error_for_chat("HTTP 429 Too Many Requests", "d-456")

        assert "high demand" in message.lower()
        assert "d-456" in message

    def test_formats_timeout_error(self):
        """format_error_for_chat handles timeout errors."""
        message = format_error_for_chat("Request timeout after 30s", "debate-789")

        assert "delay" in message.lower() or "wait" in message.lower()

    def test_formats_not_found_error(self):
        """format_error_for_chat handles 404 errors."""
        message = format_error_for_chat("Debate not found", "d-404")

        assert "couldn't find" in message.lower() or "wasn't found" in message.lower()

    def test_formats_unauthorized_error(self):
        """format_error_for_chat handles 401 errors."""
        message = format_error_for_chat("401 Unauthorized", "d-auth")

        assert "reconnect" in message.lower() or "authentication" in message.lower()

    def test_formats_forbidden_error(self):
        """format_error_for_chat handles 403 errors."""
        message = format_error_for_chat("403 Forbidden", "d-perm")

        assert "permission" in message.lower() or "access" in message.lower()

    def test_formats_connection_error(self):
        """format_error_for_chat handles connection errors."""
        message = format_error_for_chat("Connection refused", "d-conn")

        assert "connectivity" in message.lower()

    def test_formats_service_unavailable(self):
        """format_error_for_chat handles 503 errors."""
        message = format_error_for_chat("503 Service Unavailable", "d-503")

        assert "unavailable" in message.lower() or "busy" in message.lower()

    def test_formats_internal_error(self):
        """format_error_for_chat handles 500 errors."""
        message = format_error_for_chat("500 Internal Server Error", "d-500")

        assert "unexpected" in message.lower() or "wrong" in message.lower()

    def test_formats_budget_error(self):
        """format_error_for_chat handles budget exceeded errors."""
        message = format_error_for_chat("Budget limit exceeded", "d-budget")

        assert "budget" in message.lower()
        assert "admin" in message.lower()

    def test_formats_quota_error(self):
        """format_error_for_chat handles quota exceeded errors."""
        message = format_error_for_chat("Quota exceeded for this period", "d-quota")

        assert "quota" in message.lower()
        assert "billing cycle" in message.lower() or "period" in message.lower()

    def test_formats_validation_error(self):
        """format_error_for_chat handles validation errors."""
        message = format_error_for_chat("Invalid input format", "d-valid")

        assert "couldn't be processed" in message.lower() or "check your input" in message.lower()

    def test_fallback_for_unknown_error(self):
        """format_error_for_chat uses fallback for unknown errors."""
        message = format_error_for_chat("Some obscure error XYZ123", "d-unknown")

        assert "try again" in message.lower()
        assert "d-unknown" in message

    def test_always_includes_debate_id(self):
        """format_error_for_chat always includes debate ID."""
        errors = [
            "rate limit",
            "timeout",
            "unknown error",
            "",
        ]
        for error in errors:
            message = format_error_for_chat(error, "my-debate-id")
            assert "my-debate-id" in message
