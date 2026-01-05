"""Tests for the OutputSanitizer class."""

import pytest
from aragora.debate.sanitization import OutputSanitizer


class TestOutputSanitizer:
    """Tests for OutputSanitizer.sanitize_agent_output()."""

    def test_removes_null_bytes(self):
        """Should remove null bytes from output."""
        raw = "Hello\x00World\x00!"
        result = OutputSanitizer.sanitize_agent_output(raw, "test_agent")
        assert result == "HelloWorld!"
        assert "\x00" not in result

    def test_removes_control_characters(self):
        """Should remove control characters except newlines and tabs."""
        raw = "Hello\x01\x02\x03World\x7f"
        result = OutputSanitizer.sanitize_agent_output(raw, "test_agent")
        assert result == "HelloWorld"
        assert "\x01" not in result
        assert "\x7f" not in result

    def test_preserves_newlines_and_tabs(self):
        """Should preserve newlines and tabs."""
        raw = "Hello\nWorld\tTest"
        result = OutputSanitizer.sanitize_agent_output(raw, "test_agent")
        assert result == "Hello\nWorld\tTest"

    def test_is_idempotent(self):
        """Should be idempotent - running twice gives same result."""
        raw = "Hello\x00World\x01!"
        first = OutputSanitizer.sanitize_agent_output(raw, "test_agent")
        second = OutputSanitizer.sanitize_agent_output(first, "test_agent")
        assert first == second

    def test_handles_empty_string(self):
        """Should handle empty string by returning placeholder."""
        result = OutputSanitizer.sanitize_agent_output("", "test_agent")
        assert result == "(Agent produced empty output)"

    def test_handles_whitespace_only(self):
        """Should handle whitespace-only string by returning placeholder."""
        result = OutputSanitizer.sanitize_agent_output("   \n\t  ", "test_agent")
        assert result == "(Agent produced empty output)"

    def test_handles_non_string_input(self):
        """Should handle non-string input gracefully."""
        result = OutputSanitizer.sanitize_agent_output(None, "test_agent")
        assert result == "(Agent output type error)"

        result = OutputSanitizer.sanitize_agent_output(123, "test_agent")
        assert result == "(Agent output type error)"

    def test_strips_leading_trailing_whitespace(self):
        """Should strip leading and trailing whitespace."""
        raw = "   Hello World   "
        result = OutputSanitizer.sanitize_agent_output(raw, "test_agent")
        assert result == "Hello World"

    def test_normal_text_unchanged(self):
        """Should not modify normal text."""
        raw = "This is a normal response from the agent."
        result = OutputSanitizer.sanitize_agent_output(raw, "test_agent")
        assert result == raw

    def test_unicode_preserved(self):
        """Should preserve valid unicode characters."""
        raw = "Hello 世界 🌍 émojis"
        result = OutputSanitizer.sanitize_agent_output(raw, "test_agent")
        assert result == raw
