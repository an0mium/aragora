"""
Tests for the ReceiptFormatter base class and formatter registry.

Tests the abstract formatter interface, registration mechanism,
and the format_receipt_for_channel utility function.
"""

import pytest
from unittest.mock import MagicMock
from typing import Any, Optional

from aragora.channels.formatter import (
    ReceiptFormatter,
    register_formatter,
    get_formatter,
    format_receipt_for_channel,
    _FORMATTERS,
)


# =============================================================================
# Helpers - Mock receipt
# =============================================================================


def _make_receipt(**kwargs):
    """Create a mock receipt with common fields."""
    receipt = MagicMock()
    receipt.receipt_id = kwargs.get("receipt_id", "r-001")
    receipt.verdict = kwargs.get("verdict", "Approve the proposal")
    receipt.confidence = kwargs.get("confidence", 0.85)
    receipt.confidence_score = kwargs.get("confidence_score")
    receipt.topic = kwargs.get("topic", "Should we adopt microservices?")
    receipt.question = kwargs.get("question")
    receipt.decision = kwargs.get("decision")
    receipt.final_answer = kwargs.get("final_answer")
    receipt.input_summary = kwargs.get("input_summary")
    receipt.rounds = kwargs.get("rounds", 3)
    receipt.rounds_completed = kwargs.get("rounds_completed")
    receipt.agents = kwargs.get("agents", ["claude", "gpt-4", "gemini"])
    receipt.agents_involved = kwargs.get("agents_involved")
    receipt.key_arguments = kwargs.get("key_arguments")
    receipt.mitigations = kwargs.get("mitigations")
    receipt.risks = kwargs.get("risks")
    receipt.findings = kwargs.get("findings")
    receipt.dissenting_views = kwargs.get("dissenting_views")
    receipt.evidence = kwargs.get("evidence")
    receipt.timestamp = kwargs.get("timestamp", "2025-01-15T12:00:00Z")
    return receipt


# =============================================================================
# Concrete formatter for testing the base class
# =============================================================================


class _TestFormatter(ReceiptFormatter):
    """Concrete formatter for testing abstract base class."""

    @property
    def channel_type(self) -> str:
        return "test_channel"

    def format(
        self,
        receipt: Any,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {"formatted": True, "content": receipt.verdict}


# =============================================================================
# ReceiptFormatter Base Class Tests
# =============================================================================


class TestReceiptFormatterBase:
    """Tests for the abstract ReceiptFormatter base class."""

    def test_channel_type_property(self):
        """Test channel_type property on concrete implementation."""
        fmt = _TestFormatter()
        assert fmt.channel_type == "test_channel"

    def test_format_method(self):
        """Test format method on concrete implementation."""
        fmt = _TestFormatter()
        receipt = _make_receipt()
        result = fmt.format(receipt)
        assert result == {"formatted": True, "content": "Approve the proposal"}

    def test_format_summary_basic(self):
        """Test format_summary generates short text."""
        fmt = _TestFormatter()
        receipt = _make_receipt(verdict="Go ahead", confidence=0.92)
        summary = fmt.format_summary(receipt)
        assert "92%" in summary
        assert "Go ahead" in summary

    def test_format_summary_truncation(self):
        """Test format_summary truncates to max_length."""
        fmt = _TestFormatter()
        long_verdict = "A" * 300
        receipt = _make_receipt(verdict=long_verdict, confidence=0.5)
        summary = fmt.format_summary(receipt, max_length=50)
        assert len(summary) <= 50
        assert summary.endswith("...")

    def test_format_summary_no_truncation_when_short(self):
        """Test format_summary does not truncate short summaries."""
        fmt = _TestFormatter()
        receipt = _make_receipt(verdict="OK", confidence=0.9)
        summary = fmt.format_summary(receipt, max_length=280)
        assert not summary.endswith("...")

    def test_format_summary_zero_confidence(self):
        """Test format_summary with zero confidence."""
        fmt = _TestFormatter()
        receipt = _make_receipt(verdict="Uncertain", confidence=0)
        summary = fmt.format_summary(receipt)
        assert "0%" in summary

    def test_format_summary_no_verdict(self):
        """Test format_summary when verdict is None."""
        fmt = _TestFormatter()
        receipt = _make_receipt(verdict=None, confidence=0.7)
        summary = fmt.format_summary(receipt)
        assert "No decision" in summary

    def test_cannot_instantiate_abstract(self):
        """Test that ReceiptFormatter cannot be instantiated directly."""
        with pytest.raises(TypeError):
            ReceiptFormatter()


# =============================================================================
# Formatter Registry Tests
# =============================================================================


class TestFormatterRegistry:
    """Tests for the formatter registration and lookup functions."""

    def setup_method(self):
        """Save and restore the global registry."""
        self._saved = dict(_FORMATTERS)

    def teardown_method(self):
        """Restore the global registry."""
        _FORMATTERS.clear()
        _FORMATTERS.update(self._saved)

    def test_register_formatter_decorator(self):
        """Test that register_formatter adds class to registry."""

        @register_formatter
        class MyFormatter(ReceiptFormatter):
            @property
            def channel_type(self) -> str:
                return "my_channel"

            def format(self, receipt, options=None):
                return {}

        assert "my_channel" in _FORMATTERS
        assert _FORMATTERS["my_channel"] is MyFormatter

    def test_register_formatter_returns_class(self):
        """Test that register_formatter returns the class unchanged."""

        @register_formatter
        class AnotherFormatter(ReceiptFormatter):
            @property
            def channel_type(self) -> str:
                return "another"

            def format(self, receipt, options=None):
                return {}

        assert AnotherFormatter.channel_type is not None

    def test_get_formatter_found(self):
        """Test get_formatter returns a new instance."""

        @register_formatter
        class LookupFormatter(ReceiptFormatter):
            @property
            def channel_type(self) -> str:
                return "lookup_channel"

            def format(self, receipt, options=None):
                return {"found": True}

        fmt = get_formatter("lookup_channel")
        assert fmt is not None
        assert isinstance(fmt, LookupFormatter)

    def test_get_formatter_not_found(self):
        """Test get_formatter returns None for unknown channel."""
        fmt = get_formatter("nonexistent_channel_xyz")
        assert fmt is None

    def test_format_receipt_for_channel_success(self):
        """Test format_receipt_for_channel uses the right formatter."""

        @register_formatter
        class DispatchFormatter(ReceiptFormatter):
            @property
            def channel_type(self) -> str:
                return "dispatch_test"

            def format(self, receipt, options=None):
                return {"dispatched": receipt.receipt_id}

        receipt = _make_receipt(receipt_id="r-dispatch")
        result = format_receipt_for_channel(receipt, "dispatch_test")
        assert result == {"dispatched": "r-dispatch"}

    def test_format_receipt_for_channel_with_options(self):
        """Test format_receipt_for_channel passes options through."""

        @register_formatter
        class OptionsFormatter(ReceiptFormatter):
            @property
            def channel_type(self) -> str:
                return "options_test"

            def format(self, receipt, options=None):
                return {"compact": (options or {}).get("compact", False)}

        receipt = _make_receipt()
        result = format_receipt_for_channel(receipt, "options_test", options={"compact": True})
        assert result == {"compact": True}

    def test_format_receipt_for_channel_raises_on_unknown(self):
        """Test format_receipt_for_channel raises ValueError."""
        receipt = _make_receipt()
        with pytest.raises(ValueError, match="No formatter registered"):
            format_receipt_for_channel(receipt, "totally_unknown_channel")
