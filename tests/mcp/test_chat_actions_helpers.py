"""Tests for MCP chat_actions helper functions.

These tests focus on the helper functions in chat_actions.py that can be
tested directly without complex mocking of external dependencies.
"""

import pytest

from aragora.mcp.tools_module.chat_actions import (
    _create_progress_bar,
    _format_message,
    _format_poll,
    _format_receipt,
)



# =============================================================================
# Progress Bar Tests
# =============================================================================


class TestCreateProgressBar:
    """Tests for _create_progress_bar helper function."""

    def test_zero_progress(self):
        """Zero progress shows empty bar."""
        result = _create_progress_bar(0)
        assert result.startswith("[")
        assert result.endswith("]")
        assert "=" not in result

    def test_full_progress(self):
        """100% progress shows full bar."""
        result = _create_progress_bar(100)
        assert "=" in result
        assert " " not in result.strip("[]")

    def test_half_progress(self):
        """50% progress shows half-filled bar."""
        result = _create_progress_bar(50, width=20)
        # Should have roughly half filled
        bar_content = result.strip("[]")
        assert "=" in bar_content
        assert " " in bar_content

    def test_custom_width(self):
        """Custom width is respected."""
        result = _create_progress_bar(100, width=10)
        bar_content = result.strip("[]")
        assert len(bar_content) == 10

    def test_default_width(self):
        """Default width is 20."""
        result = _create_progress_bar(100)
        bar_content = result.strip("[]")
        assert len(bar_content) == 20


# =============================================================================
# Format Message Tests
# =============================================================================


class TestFormatMessage:
    """Tests for _format_message helper function."""

    def test_text_format_unchanged(self):
        """Text format leaves content unchanged."""
        content = "Hello **world**"
        result = _format_message(content, "slack", "text")
        assert result == content

    def test_markdown_slack_bold_conversion(self):
        """Slack markdown converts ** to *."""
        content = "Hello **world**"
        result = _format_message(content, "slack", "markdown")
        assert result == "Hello *world*"

    def test_markdown_slack_italic_conversion(self):
        """Slack markdown converts __ to _."""
        content = "Hello __world__"
        result = _format_message(content, "slack", "markdown")
        assert result == "Hello _world_"

    def test_markdown_discord_unchanged(self):
        """Discord markdown leaves ** unchanged."""
        content = "Hello **world**"
        result = _format_message(content, "discord", "markdown")
        assert result == content

    def test_markdown_other_platforms_unchanged(self):
        """Other platforms leave markdown unchanged."""
        content = "Hello **world**"
        result = _format_message(content, "telegram", "markdown")
        assert result == content


# =============================================================================
# Format Poll Tests
# =============================================================================


class TestFormatPoll:
    """Tests for _format_poll helper function."""

    def test_basic_poll_structure(self):
        """Poll has correct basic structure."""
        result = _format_poll("What is best?", ["Option A", "Option B"], "discord")
        assert "**Poll:**" in result
        assert "What is best?" in result
        assert "Option A" in result
        assert "Option B" in result

    def test_slack_poll_uses_emoji_numbers(self):
        """Slack polls use emoji number format."""
        result = _format_poll("Question?", ["A", "B"], "slack")
        # Slack uses :one:, :two: format
        assert ":one:" in result
        assert ":two:" in result

    def test_discord_poll_uses_numeric(self):
        """Discord polls use numeric format."""
        result = _format_poll("Question?", ["A", "B"], "discord")
        assert "1." in result
        assert "2." in result

    def test_poll_ends_with_vote_instruction(self):
        """Poll ends with voting instruction."""
        result = _format_poll("Question?", ["A", "B"], "discord")
        assert "React to vote" in result


# =============================================================================
# Format Receipt Tests
# =============================================================================


class TestFormatReceipt:
    """Tests for _format_receipt helper function."""

    def test_basic_receipt_structure(self):
        """Receipt has correct basic structure."""
        receipt = {
            "task": "Test question",
            "final_answer": "Test answer",
            "consensus_reached": True,
            "confidence": 0.85,
            "hash": "abc123xyz",
            "timestamp": "2024-01-15T10:00:00",
        }
        result = _format_receipt(receipt, "slack", include_summary=True, include_hash=True)
        assert "**Debate Receipt**" in result
        assert "Test question" in result
        assert "Test answer" in result
        assert "Yes" in result  # consensus_reached
        assert "85" in result  # confidence percentage

    def test_receipt_without_summary(self):
        """Receipt without summary excludes task and answer."""
        receipt = {
            "task": "Test question",
            "final_answer": "Test answer",
            "consensus_reached": True,
            "confidence": 0.85,
            "hash": "abc123xyz",
            "timestamp": "2024-01-15T10:00:00",
        }
        result = _format_receipt(receipt, "slack", include_summary=False, include_hash=True)
        assert "**Question:**" not in result
        assert "**Answer:**" not in result
        assert "abc123" in result  # hash should still be present

    def test_receipt_without_hash(self):
        """Receipt without hash excludes hash and timestamp."""
        receipt = {
            "task": "Test question",
            "final_answer": "Test answer",
            "consensus_reached": False,
            "confidence": 0.5,
            "hash": "abc123xyz",
            "timestamp": "2024-01-15T10:00:00",
        }
        result = _format_receipt(receipt, "slack", include_summary=True, include_hash=False)
        assert "Test question" in result
        assert "**Hash:**" not in result
        assert "**Timestamp:**" not in result

    def test_receipt_consensus_no(self):
        """Receipt shows 'No' when consensus not reached."""
        receipt = {
            "task": "Test",
            "final_answer": "Answer",
            "consensus_reached": False,
            "confidence": 0.3,
        }
        result = _format_receipt(receipt, "slack", include_summary=True, include_hash=False)
        assert "No" in result  # consensus_reached = False

    def test_receipt_missing_values(self):
        """Receipt handles missing values gracefully."""
        receipt = {}
        result = _format_receipt(receipt, "slack", include_summary=True, include_hash=True)
        assert "N/A" in result  # Default for missing values
