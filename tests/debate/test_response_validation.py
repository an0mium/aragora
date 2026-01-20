"""
Tests for response schema validation.

Tests cover:
- Valid response parsing
- Invalid response rejection
- Ready signal extraction (HTML, JSON, inline, natural language)
- Content sanitization
- Confidence clamping
- Truncation of oversized responses
"""

import pytest

from aragora.debate.schemas import (
    AgentResponseSchema,
    ReadySignal,
    StructuredContent,
    ValidationResult,
    sanitize_html,
    validate_agent_response,
    MAX_CONTENT_LENGTH,
)


class TestReadySignal:
    """Tests for ReadySignal schema."""

    def test_default_values(self):
        """Should have sensible defaults."""
        signal = ReadySignal()
        assert signal.confidence == 0.5
        assert signal.ready is False
        assert signal.reasoning == ""

    def test_confidence_clamping_high(self):
        """Should clamp confidence > 1.0 to 1.0."""
        signal = ReadySignal(confidence=1.5)
        assert signal.confidence == 1.0

    def test_confidence_clamping_low(self):
        """Should clamp confidence < 0.0 to 0.0."""
        signal = ReadySignal(confidence=-0.5)
        assert signal.confidence == 0.0

    def test_confidence_coercion_from_string(self):
        """Should coerce string to float."""
        signal = ReadySignal(confidence="0.75")
        assert signal.confidence == 0.75

    def test_confidence_coercion_invalid(self):
        """Should default to 0.5 for invalid confidence."""
        signal = ReadySignal(confidence="invalid")
        assert signal.confidence == 0.5

    def test_ready_coercion_from_string(self):
        """Should coerce string to boolean."""
        assert ReadySignal(ready="true").ready is True
        assert ReadySignal(ready="yes").ready is True
        assert ReadySignal(ready="1").ready is True
        assert ReadySignal(ready="false").ready is False
        assert ReadySignal(ready="no").ready is False


class TestStructuredContent:
    """Tests for StructuredContent schema."""

    def test_default_type(self):
        """Should default to 'text' type."""
        content = StructuredContent(content="test")
        assert content.type == "text"

    def test_content_sanitization(self):
        """Should handle None content."""
        content = StructuredContent(content=None)
        assert content.content == ""

    def test_metadata_limiting(self):
        """Should limit metadata to 50 keys."""
        large_meta = {f"key_{i}": i for i in range(100)}
        content = StructuredContent(content="test", metadata=large_meta)
        assert len(content.metadata) == 50


class TestAgentResponseSchema:
    """Tests for AgentResponseSchema."""

    def test_valid_response(self):
        """Should accept valid response."""
        response = AgentResponseSchema(
            content="This is a valid response",
            agent_name="claude",
        )
        assert response.content == "This is a valid response"
        assert response.agent_name == "claude"

    def test_empty_content_rejected(self):
        """Should reject empty content."""
        with pytest.raises(ValueError, match="empty"):
            AgentResponseSchema(content="", agent_name="claude")

    def test_none_content_rejected(self):
        """Should reject None content."""
        with pytest.raises(ValueError, match="None"):
            AgentResponseSchema(content=None, agent_name="claude")

    def test_agent_name_sanitization(self):
        """Should sanitize agent name."""
        response = AgentResponseSchema(
            content="test",
            agent_name="claude<script>",
        )
        assert "<script>" not in response.agent_name
        assert response.agent_name == "claude_script_"

    def test_confidence_defaults_to_half(self):
        """Should default confidence to 0.5."""
        response = AgentResponseSchema(content="test", agent_name="claude")
        assert response.confidence == 0.5

    def test_ready_signal_extraction_html(self):
        """Should extract ready signal from HTML comment."""
        content = '''Analysis here.
<!-- READY_SIGNAL: {"confidence": 0.85, "ready": true} -->'''
        response = AgentResponseSchema(content=content, agent_name="claude")
        assert response.ready_signal.confidence == 0.85
        assert response.ready_signal.ready is True

    def test_ready_signal_extraction_json_block(self):
        """Should extract ready signal from JSON block."""
        content = '''Analysis here.
```ready_signal {"confidence": 0.9, "ready": true} ```'''
        response = AgentResponseSchema(content=content, agent_name="claude")
        assert response.ready_signal.confidence == 0.9
        assert response.ready_signal.ready is True

    def test_ready_signal_extraction_inline(self):
        """Should extract ready signal from inline format."""
        content = 'Analysis [READY: confidence=0.8, ready=true] done.'
        response = AgentResponseSchema(content=content, agent_name="claude")
        assert response.ready_signal.confidence == 0.8
        assert response.ready_signal.ready is True

    def test_ready_signal_natural_language(self):
        """Should detect ready signal from natural language."""
        content = "This is my final position on the matter."
        response = AgentResponseSchema(content=content, agent_name="claude")
        assert response.ready_signal.ready is True
        assert response.ready_signal.confidence == 0.7


class TestValidateAgentResponse:
    """Tests for validate_agent_response function."""

    def test_valid_string_response(self):
        """Should validate string response."""
        result = validate_agent_response("Valid response", agent_name="claude")
        assert result.is_valid
        assert result.response.content == "Valid response"
        assert result.errors == []

    def test_valid_dict_response(self):
        """Should validate dict response with content key."""
        result = validate_agent_response(
            {"content": "Valid response"},
            agent_name="claude",
        )
        assert result.is_valid
        assert result.response.content == "Valid response"

    def test_valid_dict_response_text_key(self):
        """Should validate dict response with text key."""
        result = validate_agent_response(
            {"text": "Valid response"},
            agent_name="claude",
        )
        assert result.is_valid
        assert result.response.content == "Valid response"

    def test_valid_dict_response_message_key(self):
        """Should validate dict response with message key."""
        result = validate_agent_response(
            {"message": "Valid response"},
            agent_name="claude",
        )
        assert result.is_valid
        assert result.response.content == "Valid response"

    def test_empty_response_rejected(self):
        """Should reject empty response."""
        result = validate_agent_response("", agent_name="claude")
        assert not result.is_valid
        assert len(result.errors) > 0
        assert "empty" in result.errors[0].lower()

    def test_none_response_rejected(self):
        """Should reject None response."""
        result = validate_agent_response(None, agent_name="claude")
        assert not result.is_valid
        assert "None" in result.errors[0]

    def test_whitespace_only_rejected(self):
        """Should reject whitespace-only response."""
        result = validate_agent_response("   \n\t  ", agent_name="claude")
        assert not result.is_valid

    def test_oversized_response_truncated(self):
        """Should truncate oversized responses with warning."""
        huge_content = "x" * (MAX_CONTENT_LENGTH + 1000)
        result = validate_agent_response(huge_content, agent_name="claude")
        assert result.is_valid
        assert len(result.response.content) <= MAX_CONTENT_LENGTH
        assert result.response.content.endswith("... [truncated]")
        assert result.warnings is not None
        assert any("truncated" in w.lower() for w in result.warnings)

    def test_role_preserved(self):
        """Should preserve role parameter."""
        result = validate_agent_response(
            "test",
            agent_name="claude",
            role="critic",
        )
        assert result.response.role == "critic"

    def test_round_number_preserved(self):
        """Should preserve round number."""
        result = validate_agent_response(
            "test",
            agent_name="claude",
            round_number=3,
        )
        assert result.response.round_number == 3

    def test_metadata_preserved(self):
        """Should preserve metadata."""
        result = validate_agent_response(
            "test",
            agent_name="claude",
            metadata={"key": "value"},
        )
        assert result.response.metadata == {"key": "value"}


class TestSanitizeHtml:
    """Tests for HTML sanitization."""

    def test_escapes_script_tags(self):
        """Should escape script tags."""
        html = "<script>alert('xss')</script>"
        safe = sanitize_html(html)
        assert "<script>" not in safe
        assert "&lt;script&gt;" in safe

    def test_escapes_event_handlers(self):
        """Should escape event handlers."""
        html = '<img onerror="alert(1)" src="x">'
        safe = sanitize_html(html)
        assert "onerror" not in safe or "&quot;" in safe

    def test_preserves_plain_text(self):
        """Should preserve plain text."""
        text = "This is plain text with no HTML"
        assert sanitize_html(text) == text


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_valid_result(self):
        """Should create valid result."""
        response = AgentResponseSchema(content="test", agent_name="claude")
        result = ValidationResult(is_valid=True, response=response)
        assert result.is_valid
        assert result.response is not None
        assert result.errors == []

    def test_invalid_result(self):
        """Should create invalid result."""
        result = ValidationResult(is_valid=False, errors=["Error 1", "Error 2"])
        assert not result.is_valid
        assert result.response is None
        assert len(result.errors) == 2

    def test_result_with_warnings(self):
        """Should include warnings."""
        response = AgentResponseSchema(content="test", agent_name="claude")
        result = ValidationResult(
            is_valid=True,
            response=response,
            warnings=["Warning 1"],
        )
        assert result.is_valid
        assert len(result.warnings) == 1


class TestEdgeCases:
    """Tests for edge cases and malformed inputs."""

    def test_unicode_content(self):
        """Should handle unicode content."""
        result = validate_agent_response(
            "Unicode: 日本語 中文 العربية 🎉",
            agent_name="claude",
        )
        assert result.is_valid
        assert "日本語" in result.response.content

    def test_control_characters(self):
        """Should handle control characters."""
        result = validate_agent_response(
            "Content\x00with\x01control\x02chars",
            agent_name="claude",
        )
        assert result.is_valid

    def test_nested_json_in_content(self):
        """Should handle nested JSON in content."""
        content = '{"analysis": {"key": "value"}, "conclusion": "test"}'
        result = validate_agent_response(content, agent_name="claude")
        assert result.is_valid

    def test_malformed_ready_signal(self):
        """Should handle malformed ready signal gracefully."""
        content = '<!-- READY_SIGNAL: {malformed json} -->'
        result = validate_agent_response(content, agent_name="claude")
        assert result.is_valid  # Should still validate, just skip signal
        assert result.response.ready_signal.ready is False  # Default

    def test_multiple_ready_signals(self):
        """Should use first valid ready signal."""
        content = '''<!-- READY_SIGNAL: {"confidence": 0.9, "ready": true} -->
Analysis here.
<!-- READY_SIGNAL: {"confidence": 0.5, "ready": false} -->'''
        result = validate_agent_response(content, agent_name="claude")
        assert result.is_valid
        assert result.response.ready_signal.confidence == 0.9
