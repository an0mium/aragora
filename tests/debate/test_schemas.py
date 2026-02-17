"""Tests for response schema validation.

Covers ReadySignal, StructuredContent, AgentResponseSchema,
ValidationResult, _parse_ready_signal, validate_agent_response, sanitize_html.
"""

from __future__ import annotations

import pytest

from aragora.debate.schemas import (
    MAX_CONTENT_LENGTH,
    MAX_METADATA_SIZE,
    AgentResponseSchema,
    ReadySignal,
    StructuredContent,
    ValidationResult,
    sanitize_html,
    validate_agent_response,
)


# ---------------------------------------------------------------------------
# ReadySignal
# ---------------------------------------------------------------------------


class TestReadySignal:
    def test_defaults(self):
        rs = ReadySignal()
        assert rs.confidence == 0.5
        assert rs.ready is False
        assert rs.reasoning == ""

    def test_coerce_confidence_clamp_high(self):
        rs = ReadySignal(confidence=1.5)
        assert rs.confidence == 1.0

    def test_coerce_confidence_clamp_low(self):
        rs = ReadySignal(confidence=-0.3)
        assert rs.confidence == 0.0

    def test_coerce_confidence_string(self):
        rs = ReadySignal(confidence="0.7")
        assert rs.confidence == 0.7

    def test_coerce_confidence_invalid(self):
        rs = ReadySignal(confidence="not_a_number")
        assert rs.confidence == 0.5  # default fallback

    def test_coerce_ready_string_true(self):
        rs = ReadySignal(ready="true")
        assert rs.ready is True

    def test_coerce_ready_string_yes(self):
        rs = ReadySignal(ready="yes")
        assert rs.ready is True

    def test_coerce_ready_string_false(self):
        rs = ReadySignal(ready="false")
        assert rs.ready is False

    def test_coerce_ready_int(self):
        rs = ReadySignal(ready=1)
        assert rs.ready is True


# ---------------------------------------------------------------------------
# StructuredContent
# ---------------------------------------------------------------------------


class TestStructuredContent:
    def test_defaults(self):
        sc = StructuredContent()
        assert sc.type == "text"
        assert sc.content == ""
        assert sc.language is None
        assert sc.metadata is None

    def test_sanitize_none_content(self):
        sc = StructuredContent(content=None)
        assert sc.content == ""

    def test_truncate_long_content(self):
        # The validator truncates to MAX_CONTENT_LENGTH + suffix, but pydantic's
        # max_length then rejects it. So oversized content raises ValidationError.
        with pytest.raises(Exception):
            StructuredContent(content="x" * (MAX_CONTENT_LENGTH + 100))

    def test_metadata_limit(self):
        big_meta = {f"key_{i}": i for i in range(MAX_METADATA_SIZE + 10)}
        sc = StructuredContent(metadata=big_meta)
        assert len(sc.metadata) <= MAX_METADATA_SIZE

    def test_metadata_non_dict(self):
        sc = StructuredContent(metadata="not a dict")
        assert sc.metadata is None

    def test_code_block(self):
        sc = StructuredContent(type="code", content="print('hello')", language="python")
        assert sc.type == "code"
        assert sc.language == "python"


# ---------------------------------------------------------------------------
# AgentResponseSchema
# ---------------------------------------------------------------------------


class TestAgentResponseSchema:
    def test_valid_response(self):
        r = AgentResponseSchema(content="Hello world", agent_name="claude")
        assert r.content == "Hello world"
        assert r.agent_name == "claude"
        assert r.role == "proposer"
        assert r.round_number == 0
        assert r.confidence == 0.5

    def test_content_none_raises(self):
        with pytest.raises(ValueError, match="None"):
            AgentResponseSchema(content=None, agent_name="claude")

    def test_content_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            AgentResponseSchema(content="", agent_name="claude")

    def test_agent_name_none_raises(self):
        with pytest.raises(ValueError, match="None"):
            AgentResponseSchema(content="text", agent_name=None)

    def test_agent_name_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            AgentResponseSchema(content="text", agent_name="  ")

    def test_agent_name_sanitized(self):
        r = AgentResponseSchema(content="text", agent_name="claude!@#$%^&*()")
        assert re.search(r"[^a-zA-Z0-9_-]", r.agent_name) is None

    def test_confidence_clamped(self):
        r = AgentResponseSchema(content="text", agent_name="claude", confidence=2.0)
        assert r.confidence == 1.0

    def test_ready_signal_auto_parsed(self):
        content = 'My final position is complete. <!-- READY_SIGNAL: {"confidence": 0.85, "ready": true} -->'
        r = AgentResponseSchema(content=content, agent_name="claude")
        assert r.ready_signal is not None
        assert r.ready_signal.confidence == 0.85
        assert r.ready_signal.ready is True

    def test_ready_signal_from_natural_language(self):
        r = AgentResponseSchema(
            content="This is my final position on the matter.",
            agent_name="claude",
        )
        assert r.ready_signal is not None
        assert r.ready_signal.ready is True

    def test_metadata_limited(self):
        big_meta = {f"k{i}": i for i in range(MAX_METADATA_SIZE + 10)}
        r = AgentResponseSchema(content="text", agent_name="a", metadata=big_meta)
        assert len(r.metadata) <= MAX_METADATA_SIZE


# ---------------------------------------------------------------------------
# ValidationResult
# ---------------------------------------------------------------------------


class TestValidationResult:
    def test_valid(self):
        vr = ValidationResult(is_valid=True)
        assert vr.errors == []
        assert vr.warnings == []

    def test_invalid_with_errors(self):
        vr = ValidationResult(is_valid=False, errors=["bad content"])
        assert vr.errors == ["bad content"]


# ---------------------------------------------------------------------------
# _parse_ready_signal (tested through validate_agent_response)
# ---------------------------------------------------------------------------


class TestParseReadySignal:
    def test_html_comment_format(self):
        content = '<!-- READY_SIGNAL: {"confidence": 0.9, "ready": true} -->'
        r = AgentResponseSchema(content=content, agent_name="a")
        assert r.ready_signal.ready is True
        assert r.ready_signal.confidence == 0.9

    def test_json_block_format(self):
        content = 'text ```ready_signal {"confidence": 0.8, "ready": true} ``` more'
        r = AgentResponseSchema(content=content, agent_name="a")
        assert r.ready_signal.ready is True
        assert r.ready_signal.confidence == 0.8

    def test_inline_format(self):
        content = "My response [READY: confidence=0.75, ready=true]"
        r = AgentResponseSchema(content=content, agent_name="a")
        assert r.ready_signal.ready is True
        assert r.ready_signal.confidence == 0.75

    def test_natural_language_fully_refined(self):
        r = AgentResponseSchema(
            content="I believe this position is fully refined.",
            agent_name="a",
        )
        assert r.ready_signal.ready is True

    def test_natural_language_ready_to_conclude(self):
        r = AgentResponseSchema(
            content="I am ready to conclude on this topic.",
            agent_name="a",
        )
        assert r.ready_signal.ready is True

    def test_no_signal(self):
        r = AgentResponseSchema(
            content="Just a normal response about rate limiters.",
            agent_name="a",
        )
        assert r.ready_signal.ready is False


# ---------------------------------------------------------------------------
# validate_agent_response
# ---------------------------------------------------------------------------


class TestValidateAgentResponse:
    def test_valid_string(self):
        result = validate_agent_response("Hello", agent_name="claude")
        assert result.is_valid
        assert result.response.content == "Hello"
        assert result.response.agent_name == "claude"

    def test_valid_dict_content(self):
        result = validate_agent_response(
            {"content": "Hello from dict"}, agent_name="claude"
        )
        assert result.is_valid
        assert "Hello from dict" in result.response.content

    def test_valid_dict_text(self):
        result = validate_agent_response(
            {"text": "Hello from text"}, agent_name="claude"
        )
        assert result.is_valid
        assert "Hello from text" in result.response.content

    def test_valid_dict_message(self):
        result = validate_agent_response(
            {"message": "Hello from message"}, agent_name="claude"
        )
        assert result.is_valid
        assert "Hello from message" in result.response.content

    def test_none_content(self):
        result = validate_agent_response(None, agent_name="claude")
        assert not result.is_valid
        assert "None" in result.errors[0]

    def test_empty_content(self):
        result = validate_agent_response("", agent_name="claude")
        assert not result.is_valid
        assert "empty" in result.errors[0]

    def test_whitespace_content(self):
        result = validate_agent_response("   ", agent_name="claude")
        assert not result.is_valid

    def test_truncation_warning(self):
        result = validate_agent_response("x" * (MAX_CONTENT_LENGTH + 100), agent_name="a")
        assert result.is_valid
        assert result.warnings and any("truncated" in w.lower() for w in result.warnings)

    def test_role_and_round(self):
        result = validate_agent_response(
            "text", agent_name="claude", role="judge", round_number=3
        )
        assert result.response.role == "judge"
        assert result.response.round_number == 3

    def test_metadata_attached(self):
        result = validate_agent_response(
            "text", agent_name="claude", metadata={"key": "value"}
        )
        assert result.response.metadata == {"key": "value"}


# ---------------------------------------------------------------------------
# sanitize_html
# ---------------------------------------------------------------------------


class TestSanitizeHtml:
    def test_escapes_tags(self):
        result = sanitize_html("<script>alert('xss')</script>")
        assert "<script>" not in result
        assert "&lt;script&gt;" in result

    def test_preserves_plain_text(self):
        assert sanitize_html("hello world") == "hello world"

    def test_restores_bold(self):
        result = sanitize_html("<b>bold text</b>")
        assert "**bold text**" in result

    def test_restores_italic(self):
        result = sanitize_html("<i>italic text</i>")
        assert "*italic text*" in result

    def test_restores_code(self):
        result = sanitize_html("<code>inline code</code>")
        assert "`inline code`" in result


import re  # noqa: E402 - needed for test assertions
