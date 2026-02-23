"""Tests for Slack response utilities (aragora/server/handlers/social/slack/responses.py).

Covers all public functions:
- slack_response: basic Slack-formatted responses
- slack_blocks_response: Block Kit block responses

Tests include:
- Default parameters and return types
- Response type variants (ephemeral, in_channel)
- Attachments handling (present/absent)
- Blocks handling and fallback text
- JSON serialization correctness
- Content-type and status code
- Edge cases: empty strings, unicode, special characters, large payloads
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from aragora.server.handlers.social.slack.responses import (
    slack_blocks_response,
    slack_response,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result: object) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _status(result: object) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


# ===========================================================================
# slack_response tests
# ===========================================================================


class TestSlackResponse:
    """Tests for the slack_response helper."""

    def test_basic_text_response(self):
        """A simple text response returns valid JSON with the text."""
        result = slack_response("Hello, world!")
        body = _body(result)
        assert body["text"] == "Hello, world!"

    def test_default_response_type_is_ephemeral(self):
        """Default response_type should be ephemeral."""
        result = slack_response("test")
        body = _body(result)
        assert body["response_type"] == "ephemeral"

    def test_in_channel_response_type(self):
        """Setting response_type='in_channel' is reflected in the body."""
        result = slack_response("test", response_type="in_channel")
        body = _body(result)
        assert body["response_type"] == "in_channel"

    def test_custom_response_type(self):
        """An arbitrary response_type string is passed through."""
        result = slack_response("test", response_type="custom_type")
        body = _body(result)
        assert body["response_type"] == "custom_type"

    def test_status_code_is_200(self):
        """slack_response always returns HTTP 200."""
        result = slack_response("ok")
        assert _status(result) == 200

    def test_content_type_is_json(self):
        """Content-Type should be application/json."""
        result = slack_response("ok")
        assert result.content_type == "application/json"

    def test_body_is_bytes(self):
        """The body attribute should be bytes."""
        result = slack_response("test")
        assert isinstance(result.body, bytes)

    def test_body_is_valid_json(self):
        """The body should be valid, parseable JSON."""
        result = slack_response("test")
        parsed = json.loads(result.body)
        assert isinstance(parsed, dict)

    def test_no_attachments_by_default(self):
        """When no attachments are given, the key should be absent."""
        result = slack_response("test")
        body = _body(result)
        assert "attachments" not in body

    def test_attachments_none_excluded(self):
        """Explicitly passing attachments=None excludes the key."""
        result = slack_response("test", attachments=None)
        body = _body(result)
        assert "attachments" not in body

    def test_empty_attachments_list_excluded(self):
        """An empty list is falsy, so attachments should be excluded."""
        result = slack_response("test", attachments=[])
        body = _body(result)
        assert "attachments" not in body

    def test_attachments_included_when_provided(self):
        """Non-empty attachments list should appear in the response."""
        attachments = [{"text": "Attachment 1", "color": "#36a64f"}]
        result = slack_response("test", attachments=attachments)
        body = _body(result)
        assert body["attachments"] == attachments

    def test_multiple_attachments(self):
        """Multiple attachments should all be serialized."""
        attachments = [
            {"text": "First", "color": "#ff0000"},
            {"text": "Second", "color": "#00ff00"},
            {"text": "Third", "color": "#0000ff"},
        ]
        result = slack_response("test", attachments=attachments)
        body = _body(result)
        assert len(body["attachments"]) == 3
        assert body["attachments"][1]["text"] == "Second"

    def test_unicode_text(self):
        """Unicode characters should be properly serialized."""
        result = slack_response("Hello \u2603 \u2764\ufe0f \U0001f600")
        body = _body(result)
        assert "\u2603" in body["text"]

    def test_empty_text(self):
        """An empty string should be preserved as text."""
        result = slack_response("")
        body = _body(result)
        assert body["text"] == ""

    def test_special_characters_in_text(self):
        """Special JSON characters should be properly escaped."""
        result = slack_response('He said "hello" & <goodbye>')
        body = _body(result)
        assert body["text"] == 'He said "hello" & <goodbye>'

    def test_newlines_in_text(self):
        """Newlines in text should survive serialization."""
        result = slack_response("line1\nline2\nline3")
        body = _body(result)
        assert body["text"] == "line1\nline2\nline3"

    def test_very_long_text(self):
        """A very long text string should serialize correctly."""
        long_text = "x" * 10_000
        result = slack_response(long_text)
        body = _body(result)
        assert body["text"] == long_text
        assert len(body["text"]) == 10_000

    def test_response_only_has_expected_keys_no_attachments(self):
        """Without attachments, body should only contain response_type and text."""
        result = slack_response("test")
        body = _body(result)
        assert set(body.keys()) == {"response_type", "text"}

    def test_response_keys_with_attachments(self):
        """With attachments, body should contain response_type, text, and attachments."""
        result = slack_response("test", attachments=[{"text": "a"}])
        body = _body(result)
        assert set(body.keys()) == {"response_type", "text", "attachments"}

    def test_complex_attachment_structure(self):
        """Complex nested attachment structures should serialize correctly."""
        attachments = [
            {
                "fallback": "Required plain-text summary",
                "color": "#2eb886",
                "pretext": "Optional text above the attachment",
                "author_name": "author",
                "title": "Title",
                "title_link": "https://example.com",
                "text": "body text",
                "fields": [
                    {"title": "Priority", "value": "High", "short": True},
                    {"title": "Status", "value": "Open", "short": True},
                ],
            }
        ]
        result = slack_response("test", attachments=attachments)
        body = _body(result)
        assert body["attachments"][0]["fields"][0]["title"] == "Priority"
        assert body["attachments"][0]["color"] == "#2eb886"


# ===========================================================================
# slack_blocks_response tests
# ===========================================================================


class TestSlackBlocksResponse:
    """Tests for the slack_blocks_response helper."""

    def test_basic_blocks_response(self):
        """A simple blocks response should include the blocks."""
        blocks = [{"type": "section", "text": {"type": "mrkdwn", "text": "Hello"}}]
        result = slack_blocks_response(blocks)
        body = _body(result)
        assert body["blocks"] == blocks

    def test_default_response_type_is_ephemeral(self):
        """Default response_type should be ephemeral."""
        blocks = [{"type": "section", "text": {"type": "mrkdwn", "text": "test"}}]
        result = slack_blocks_response(blocks)
        body = _body(result)
        assert body["response_type"] == "ephemeral"

    def test_in_channel_response_type(self):
        """Setting response_type='in_channel' is reflected."""
        blocks = [{"type": "divider"}]
        result = slack_blocks_response(blocks, response_type="in_channel")
        body = _body(result)
        assert body["response_type"] == "in_channel"

    def test_status_code_is_200(self):
        """slack_blocks_response always returns HTTP 200."""
        result = slack_blocks_response([{"type": "divider"}])
        assert _status(result) == 200

    def test_content_type_is_json(self):
        """Content-Type should be application/json."""
        result = slack_blocks_response([{"type": "divider"}])
        assert result.content_type == "application/json"

    def test_body_is_bytes(self):
        """The body attribute should be bytes."""
        result = slack_blocks_response([{"type": "divider"}])
        assert isinstance(result.body, bytes)

    def test_no_text_by_default(self):
        """When text is empty (default), the key should be absent."""
        result = slack_blocks_response([{"type": "divider"}])
        body = _body(result)
        assert "text" not in body

    def test_text_empty_string_excluded(self):
        """Explicitly passing text='' should exclude the text key (falsy)."""
        result = slack_blocks_response([{"type": "divider"}], text="")
        body = _body(result)
        assert "text" not in body

    def test_fallback_text_included(self):
        """Non-empty text should appear as a fallback."""
        result = slack_blocks_response(
            [{"type": "divider"}], text="Fallback notification text"
        )
        body = _body(result)
        assert body["text"] == "Fallback notification text"

    def test_response_keys_without_text(self):
        """Without text, body should only contain response_type and blocks."""
        result = slack_blocks_response([{"type": "divider"}])
        body = _body(result)
        assert set(body.keys()) == {"response_type", "blocks"}

    def test_response_keys_with_text(self):
        """With text, body should contain response_type, blocks, and text."""
        result = slack_blocks_response([{"type": "divider"}], text="hi")
        body = _body(result)
        assert set(body.keys()) == {"response_type", "blocks", "text"}

    def test_empty_blocks_list(self):
        """An empty blocks list should still serialize."""
        result = slack_blocks_response([])
        body = _body(result)
        assert body["blocks"] == []

    def test_multiple_blocks(self):
        """Multiple blocks should all be preserved."""
        blocks = [
            {"type": "header", "text": {"type": "plain_text", "text": "Header"}},
            {"type": "section", "text": {"type": "mrkdwn", "text": "Body"}},
            {"type": "divider"},
            {"type": "section", "text": {"type": "mrkdwn", "text": "Footer"}},
        ]
        result = slack_blocks_response(blocks)
        body = _body(result)
        assert len(body["blocks"]) == 4
        assert body["blocks"][0]["type"] == "header"
        assert body["blocks"][2]["type"] == "divider"

    def test_blocks_with_actions(self):
        """Blocks containing interactive elements should serialize properly."""
        blocks = [
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Approve"},
                        "style": "primary",
                        "action_id": "approve_action",
                    },
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "Deny"},
                        "style": "danger",
                        "action_id": "deny_action",
                    },
                ],
            }
        ]
        result = slack_blocks_response(blocks)
        body = _body(result)
        elements = body["blocks"][0]["elements"]
        assert len(elements) == 2
        assert elements[0]["action_id"] == "approve_action"
        assert elements[1]["style"] == "danger"

    def test_unicode_in_blocks(self):
        """Unicode in block text should be properly serialized."""
        blocks = [
            {"type": "section", "text": {"type": "mrkdwn", "text": "Caf\u00e9 \u2603"}}
        ]
        result = slack_blocks_response(blocks)
        body = _body(result)
        assert "Caf\u00e9" in body["blocks"][0]["text"]["text"]

    def test_custom_response_type(self):
        """An arbitrary response_type string is passed through for blocks."""
        result = slack_blocks_response(
            [{"type": "divider"}], response_type="custom_value"
        )
        body = _body(result)
        assert body["response_type"] == "custom_value"


# ===========================================================================
# Cross-function consistency tests
# ===========================================================================


class TestCrossFunctionConsistency:
    """Tests ensuring consistency between slack_response and slack_blocks_response."""

    def test_both_return_same_status(self):
        """Both functions should return status 200."""
        r1 = slack_response("test")
        r2 = slack_blocks_response([{"type": "divider"}])
        assert _status(r1) == _status(r2) == 200

    def test_both_return_same_content_type(self):
        """Both functions should return application/json content type."""
        r1 = slack_response("test")
        r2 = slack_blocks_response([{"type": "divider"}])
        assert r1.content_type == r2.content_type == "application/json"

    def test_both_return_bytes_body(self):
        """Both functions should return bytes in the body field."""
        r1 = slack_response("test")
        r2 = slack_blocks_response([{"type": "divider"}])
        assert isinstance(r1.body, bytes)
        assert isinstance(r2.body, bytes)

    def test_utf8_encoding_both(self):
        """Both should produce valid UTF-8 encoded JSON."""
        r1 = slack_response("\u00e9\u00e8\u00ea")
        r2 = slack_blocks_response(
            [{"type": "section", "text": {"type": "mrkdwn", "text": "\u00e9\u00e8\u00ea"}}]
        )
        # Should not raise
        json.loads(r1.body.decode("utf-8"))
        json.loads(r2.body.decode("utf-8"))

    def test_ephemeral_default_both(self):
        """Both functions default to 'ephemeral' response_type."""
        body1 = _body(slack_response("test"))
        body2 = _body(slack_blocks_response([{"type": "divider"}]))
        assert body1["response_type"] == "ephemeral"
        assert body2["response_type"] == "ephemeral"
