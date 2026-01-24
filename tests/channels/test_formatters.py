"""
Tests for channel-specific receipt formatters.

Tests the Slack, Teams, Discord, and Email formatters for decision receipts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pytest

from aragora.channels.formatter import (
    ReceiptFormatter,
    format_receipt_for_channel,
    get_formatter,
    register_formatter,
)


@dataclass
class MockReceipt:
    """Mock receipt for testing formatters."""

    receipt_id: str = "test-receipt-123"
    topic: str = "Should we adopt microservices?"
    question: str = "Should we adopt microservices architecture?"
    decision: str = "Yes, adopt microservices with gradual migration"
    verdict: str = "APPROVED"
    confidence_score: float = 0.85
    confidence: float = 0.85
    key_arguments: List[str] | None = None
    risks: List[str] | None = None
    dissenting_views: List[str] | None = None
    agents: List[str] | None = None
    agents_involved: List[str] | None = None
    rounds: int = 3
    rounds_completed: int = 3
    evidence: List[Dict[str, Any]] | None = None
    input_summary: str = "Evaluating microservices adoption"
    timestamp: str = "2024-01-15T10:30:00Z"

    def __post_init__(self):
        if self.key_arguments is None:
            self.key_arguments = [
                "Better scalability",
                "Independent deployments",
                "Team autonomy",
            ]
        if self.risks is None:
            self.risks = [
                "Increased operational complexity",
                "Network latency overhead",
            ]
        if self.agents is None:
            self.agents = ["claude", "gpt-4", "gemini"]
        if self.agents_involved is None:
            self.agents_involved = ["claude", "gpt-4", "gemini"]
        if self.evidence is None:
            self.evidence = [
                {"type": "article", "source": "Martin Fowler"},
                {"type": "case_study", "source": "Netflix"},
            ]


class TestReceiptFormatter:
    """Tests for the base ReceiptFormatter class."""

    def test_format_summary_basic(self):
        """Test basic summary formatting."""
        from aragora.channels.slack_formatter import SlackReceiptFormatter

        formatter = SlackReceiptFormatter()
        receipt = MockReceipt()

        summary = formatter.format_summary(receipt)
        assert "[85% confidence]" in summary
        assert "APPROVED" in summary

    def test_format_summary_truncation(self):
        """Test summary truncation for long decisions."""
        from aragora.channels.slack_formatter import SlackReceiptFormatter

        formatter = SlackReceiptFormatter()
        receipt = MockReceipt(verdict="A" * 300)

        summary = formatter.format_summary(receipt, max_length=100)
        assert len(summary) <= 100
        assert summary.endswith("...")

    def test_format_summary_no_verdict(self):
        """Test summary with missing verdict."""
        from aragora.channels.slack_formatter import SlackReceiptFormatter

        formatter = SlackReceiptFormatter()
        receipt = MockReceipt(verdict=None)

        summary = formatter.format_summary(receipt)
        assert "No decision" in summary


class TestSlackFormatter:
    """Tests for the Slack Block Kit formatter."""

    def test_format_basic(self):
        """Test basic Slack formatting."""
        from aragora.channels.slack_formatter import SlackReceiptFormatter

        formatter = SlackReceiptFormatter()
        receipt = MockReceipt()

        result = formatter.format(receipt)

        assert "blocks" in result
        blocks = result["blocks"]

        # Should have header
        header = next((b for b in blocks if b.get("type") == "header"), None)
        assert header is not None
        assert "Decision Receipt" in header["text"]["text"]

    def test_format_with_high_confidence(self):
        """Test formatting with high confidence shows green indicators."""
        from aragora.channels.slack_formatter import SlackReceiptFormatter

        formatter = SlackReceiptFormatter()
        receipt = MockReceipt(confidence_score=0.95)

        result = formatter.format(receipt)
        blocks = result["blocks"]

        # Header should have green checkmark emoji
        header = next((b for b in blocks if b.get("type") == "header"), None)
        assert ":white_check_mark:" in header["text"]["text"]

    def test_format_with_low_confidence(self):
        """Test formatting with low confidence shows red indicators."""
        from aragora.channels.slack_formatter import SlackReceiptFormatter

        formatter = SlackReceiptFormatter()
        receipt = MockReceipt(confidence_score=0.35)

        result = formatter.format(receipt)
        blocks = result["blocks"]

        # Header should have red circle emoji
        header = next((b for b in blocks if b.get("type") == "header"), None)
        assert ":red_circle:" in header["text"]["text"]

    def test_format_compact_mode(self):
        """Test compact mode excludes detailed sections."""
        from aragora.channels.slack_formatter import SlackReceiptFormatter

        formatter = SlackReceiptFormatter()
        receipt = MockReceipt()

        result_full = formatter.format(receipt, options={"compact": False})
        result_compact = formatter.format(receipt, options={"compact": True})

        # Compact should have fewer blocks
        assert len(result_compact["blocks"]) < len(result_full["blocks"])

    def test_format_excludes_agents(self):
        """Test option to exclude agents section."""
        from aragora.channels.slack_formatter import SlackReceiptFormatter

        formatter = SlackReceiptFormatter()
        receipt = MockReceipt()

        result = formatter.format(receipt, options={"include_agents": False})
        blocks = result["blocks"]

        # Should not contain "Agents:" text
        for block in blocks:
            if block.get("type") == "context":
                elements = block.get("elements", [])
                for elem in elements:
                    if elem.get("type") == "mrkdwn":
                        assert ":robot_face: *Agents:*" not in elem.get("text", "")

    def test_format_with_evidence(self):
        """Test evidence section is included."""
        from aragora.channels.slack_formatter import SlackReceiptFormatter

        formatter = SlackReceiptFormatter()
        receipt = MockReceipt()

        result = formatter.format(receipt, options={"include_evidence": True})
        blocks_text = str(result["blocks"])

        assert "evidence sources" in blocks_text

    def test_format_without_evidence(self):
        """Test evidence section can be excluded."""
        from aragora.channels.slack_formatter import SlackReceiptFormatter

        formatter = SlackReceiptFormatter()
        receipt = MockReceipt()

        result = formatter.format(receipt, options={"include_evidence": False})
        blocks_text = str(result["blocks"])

        assert "evidence sources" not in blocks_text

    def test_format_many_agents(self):
        """Test handling of many agents (should truncate)."""
        from aragora.channels.slack_formatter import SlackReceiptFormatter

        formatter = SlackReceiptFormatter()
        receipt = MockReceipt(
            agents=["agent1", "agent2", "agent3", "agent4", "agent5", "agent6", "agent7"]
        )

        result = formatter.format(receipt)
        blocks_text = str(result["blocks"])

        assert "+2 more" in blocks_text

    def test_channel_type(self):
        """Test channel type property."""
        from aragora.channels.slack_formatter import SlackReceiptFormatter

        formatter = SlackReceiptFormatter()
        assert formatter.channel_type == "slack"

    def test_confidence_emoji_thresholds(self):
        """Test confidence emoji selection at boundary values."""
        from aragora.channels.slack_formatter import SlackReceiptFormatter

        formatter = SlackReceiptFormatter()

        assert formatter._get_confidence_emoji(0.9) == ":white_check_mark:"
        assert formatter._get_confidence_emoji(0.89) == ":large_green_circle:"
        assert formatter._get_confidence_emoji(0.7) == ":large_green_circle:"
        assert formatter._get_confidence_emoji(0.69) == ":large_yellow_circle:"
        assert formatter._get_confidence_emoji(0.5) == ":large_yellow_circle:"
        assert formatter._get_confidence_emoji(0.49) == ":red_circle:"
        assert formatter._get_confidence_emoji(0.0) == ":red_circle:"


class TestTeamsFormatter:
    """Tests for the Microsoft Teams Adaptive Cards formatter."""

    def test_format_basic(self):
        """Test basic Teams formatting."""
        from aragora.channels.teams_formatter import TeamsReceiptFormatter

        formatter = TeamsReceiptFormatter()
        receipt = MockReceipt()

        result = formatter.format(receipt)

        assert result["type"] == "AdaptiveCard"
        assert result["version"] == "1.4"
        assert "body" in result
        assert "actions" in result

    def test_format_includes_topic(self):
        """Test that topic is included in the card."""
        from aragora.channels.teams_formatter import TeamsReceiptFormatter

        formatter = TeamsReceiptFormatter()
        receipt = MockReceipt(topic="Test Topic Question")

        result = formatter.format(receipt)
        body_text = str(result["body"])

        assert "Test Topic Question" in body_text

    def test_format_confidence_colors(self):
        """Test confidence color selection."""
        from aragora.channels.teams_formatter import TeamsReceiptFormatter

        formatter = TeamsReceiptFormatter()

        assert formatter._get_confidence_color(0.8) == "Good"
        assert formatter._get_confidence_color(0.79) == "Warning"
        assert formatter._get_confidence_color(0.5) == "Warning"
        assert formatter._get_confidence_color(0.49) == "Attention"

    def test_format_confidence_labels(self):
        """Test confidence label selection."""
        from aragora.channels.teams_formatter import TeamsReceiptFormatter

        formatter = TeamsReceiptFormatter()

        assert formatter._get_confidence_label(0.9) == "Very High"
        assert formatter._get_confidence_label(0.89) == "High"
        assert formatter._get_confidence_label(0.7) == "High"
        assert formatter._get_confidence_label(0.69) == "Moderate"
        assert formatter._get_confidence_label(0.5) == "Moderate"
        assert formatter._get_confidence_label(0.49) == "Low"

    def test_format_compact_mode(self):
        """Test compact mode excludes detailed sections."""
        from aragora.channels.teams_formatter import TeamsReceiptFormatter

        formatter = TeamsReceiptFormatter()
        receipt = MockReceipt()

        result_full = formatter.format(receipt, options={"compact": False})
        result_compact = formatter.format(receipt, options={"compact": True})

        # Compact should have fewer body items
        assert len(result_compact["body"]) < len(result_full["body"])

    def test_format_includes_view_action(self):
        """Test that view action URL is included."""
        from aragora.channels.teams_formatter import TeamsReceiptFormatter

        formatter = TeamsReceiptFormatter()
        receipt = MockReceipt(receipt_id="my-receipt-id")

        result = formatter.format(receipt)
        actions = result["actions"]

        assert len(actions) == 1
        assert actions[0]["type"] == "Action.OpenUrl"
        assert "my-receipt-id" in actions[0]["url"]

    def test_channel_type(self):
        """Test channel type property."""
        from aragora.channels.teams_formatter import TeamsReceiptFormatter

        formatter = TeamsReceiptFormatter()
        assert formatter.channel_type == "teams"


class TestFormatterRegistry:
    """Tests for the formatter registry functions."""

    def test_get_slack_formatter(self):
        """Test getting registered Slack formatter."""
        # Import to trigger registration
        import aragora.channels.slack_formatter  # noqa: F401

        formatter = get_formatter("slack")
        assert formatter is not None
        assert formatter.channel_type == "slack"

    def test_get_teams_formatter(self):
        """Test getting registered Teams formatter."""
        # Import to trigger registration
        import aragora.channels.teams_formatter  # noqa: F401

        formatter = get_formatter("teams")
        assert formatter is not None
        assert formatter.channel_type == "teams"

    def test_get_unknown_formatter(self):
        """Test getting unknown formatter returns None."""
        formatter = get_formatter("unknown_channel_xyz")
        assert formatter is None

    def test_format_receipt_for_channel(self):
        """Test convenience function for formatting."""
        import aragora.channels.slack_formatter  # noqa: F401

        receipt = MockReceipt()
        result = format_receipt_for_channel(receipt, "slack")

        assert "blocks" in result

    def test_format_receipt_for_unknown_channel(self):
        """Test formatting for unknown channel raises error."""
        receipt = MockReceipt()

        with pytest.raises(ValueError, match="No formatter registered"):
            format_receipt_for_channel(receipt, "unknown_channel_xyz")

    def test_register_formatter_decorator(self):
        """Test the register_formatter decorator."""

        @register_formatter
        class TestFormatter(ReceiptFormatter):
            @property
            def channel_type(self) -> str:
                return "test_channel_123"

            def format(
                self, receipt: Any, options: Optional[Dict[str, Any]] = None
            ) -> Dict[str, Any]:
                return {"test": True}

        formatter = get_formatter("test_channel_123")
        assert formatter is not None
        assert formatter.channel_type == "test_channel_123"


class TestDiscordFormatter:
    """Tests for the Discord formatter."""

    def test_format_basic(self):
        """Test basic Discord formatting."""
        from aragora.channels.discord_formatter import DiscordReceiptFormatter

        formatter = DiscordReceiptFormatter()
        receipt = MockReceipt()

        result = formatter.format(receipt)

        assert "embeds" in result
        embeds = result["embeds"]
        assert len(embeds) >= 1

    def test_channel_type(self):
        """Test channel type property."""
        from aragora.channels.discord_formatter import DiscordReceiptFormatter

        formatter = DiscordReceiptFormatter()
        assert formatter.channel_type == "discord"


class TestEmailFormatter:
    """Tests for the Email formatter."""

    def test_format_basic(self):
        """Test basic Email formatting."""
        from aragora.channels.email_formatter import EmailReceiptFormatter

        formatter = EmailReceiptFormatter()
        receipt = MockReceipt()

        result = formatter.format(receipt)

        assert "subject" in result
        assert "body" in result or "html" in result

    def test_channel_type(self):
        """Test channel type property."""
        from aragora.channels.email_formatter import EmailReceiptFormatter

        formatter = EmailReceiptFormatter()
        assert formatter.channel_type == "email"


class TestFormatterEdgeCases:
    """Edge case tests for formatters."""

    def test_empty_receipt(self):
        """Test formatting receipt with minimal data."""
        from aragora.channels.slack_formatter import SlackReceiptFormatter

        formatter = SlackReceiptFormatter()
        receipt = MockReceipt(
            key_arguments=[],
            risks=[],
            agents=[],
            evidence=[],
        )

        result = formatter.format(receipt)
        assert "blocks" in result

    def test_none_attributes(self):
        """Test handling of None attributes."""
        from aragora.channels.teams_formatter import TeamsReceiptFormatter

        formatter = TeamsReceiptFormatter()
        receipt = MockReceipt(
            topic=None,
            question=None,
            decision=None,
        )

        # Should not raise
        result = formatter.format(receipt)
        assert "body" in result

    def test_special_characters_in_text(self):
        """Test handling of special characters."""
        from aragora.channels.slack_formatter import SlackReceiptFormatter

        formatter = SlackReceiptFormatter()
        receipt = MockReceipt(
            topic='Topic with <special> & "characters"',
            decision="Decision with 'quotes' and `backticks`",
        )

        # Should not raise
        result = formatter.format(receipt)
        assert "blocks" in result

    def test_unicode_content(self):
        """Test handling of unicode content."""
        from aragora.channels.slack_formatter import SlackReceiptFormatter

        formatter = SlackReceiptFormatter()
        receipt = MockReceipt(
            topic="Testing with emojis: We should adopt microservices",
            decision="We recommend the cloud approach",
        )

        result = formatter.format(receipt)
        assert "blocks" in result
