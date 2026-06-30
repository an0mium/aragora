"""Tests for CLI agent sanitization to prevent 'embedded null byte' errors."""

import pytest
from aragora.agents.cli_agents import CLIAgent, CodexAgent
from aragora.core import Message


class TestCLIAgentSanitization:
    """Tests for CLIAgent._sanitize_cli_arg() method."""

    @pytest.fixture
    def agent(self):
        """Create a CLIAgent instance for testing."""
        return CodexAgent(name="test-agent", model="test")

    def test_removes_null_bytes(self, agent):
        """Should remove null bytes from CLI arguments."""
        dirty = "Hello\x00World\x00!"
        clean = agent._sanitize_cli_arg(dirty)
        assert clean == "HelloWorld!"
        assert "\x00" not in clean

    def test_removes_control_characters(self, agent):
        """Should remove control characters except newlines and tabs."""
        dirty = "Hello\x01\x02\x03World\x7f"
        clean = agent._sanitize_cli_arg(dirty)
        assert clean == "HelloWorld"

    def test_preserves_newlines_tabs(self, agent):
        """Should preserve newlines and tabs."""
        text = "Hello\nWorld\tTest"
        clean = agent._sanitize_cli_arg(text)
        assert clean == text

    def test_preserves_carriage_return(self, agent):
        """Should preserve carriage returns."""
        text = "Hello\rWorld"
        clean = agent._sanitize_cli_arg(text)
        assert clean == text

    def test_handles_non_string(self, agent):
        """Should convert non-string to string."""
        result = agent._sanitize_cli_arg(123)
        assert result == "123"

    def test_normal_text_unchanged(self, agent):
        """Should not modify normal text."""
        text = "This is a normal prompt for the agent."
        result = agent._sanitize_cli_arg(text)
        assert result == text

    def test_unicode_preserved(self, agent):
        """Should preserve valid unicode characters."""
        text = "Hello 世界 🌍 émojis"
        result = agent._sanitize_cli_arg(text)
        assert result == text


class TestBuildContextPromptSanitization:
    """Tests for CLIAgent._build_context_prompt() sanitization."""

    @pytest.fixture
    def agent(self):
        """Create a CLIAgent instance for testing."""
        return CodexAgent(name="test-agent", model="test")

    def test_sanitizes_context_with_null_bytes(self, agent):
        """Should sanitize context messages containing null bytes."""
        messages = [
            Message(
                round=1,
                role="proposer",
                agent="test",
                content="Hello\x00World",
            )
        ]
        result = agent._build_context_prompt(messages)
        assert "\x00" not in result
        assert "HelloWorld" in result

    def test_sanitizes_multiple_messages(self, agent):
        """Should sanitize all context messages."""
        messages = [
            Message(round=1, role="proposer", agent="agent1", content="First\x00message"),
            Message(round=2, role="critic", agent="agent2", content="Second\x01message"),
        ]
        result = agent._build_context_prompt(messages)
        assert "\x00" not in result
        assert "\x01" not in result
        assert "Firstmessage" in result
        assert "Secondmessage" in result

    def test_preserves_normal_content(self, agent):
        """Should preserve normal content without modification."""
        messages = [
            Message(
                round=1,
                role="proposer",
                agent="test",
                content="This is normal content with\nnewlines",
            )
        ]
        result = agent._build_context_prompt(messages)
        assert "This is normal content with\nnewlines" in result

    def test_empty_context(self, agent):
        """Should handle empty context."""
        result = agent._build_context_prompt([])
        assert result == ""

    def test_none_context(self, agent):
        """Should handle None context."""
        result = agent._build_context_prompt(None)
        assert result == ""
