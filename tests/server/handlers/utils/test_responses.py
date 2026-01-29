"""Tests for responses module."""

from __future__ import annotations

import sys
import types as _types_mod

# Pre-stub Slack modules to prevent import chain failures
_SLACK_ATTRS = [
    "SlackHandler",
    "get_slack_handler",
    "get_slack_integration",
    "get_workspace_store",
    "resolve_workspace",
    "create_tracked_task",
    "_validate_slack_url",
    "SLACK_SIGNING_SECRET",
    "SLACK_BOT_TOKEN",
    "SLACK_WEBHOOK_URL",
    "SLACK_ALLOWED_DOMAINS",
    "SignatureVerifierMixin",
    "CommandsMixin",
    "EventsMixin",
    "init_slack_handler",
]
for _mod_name in (
    "aragora.server.handlers.social.slack.handler",
    "aragora.server.handlers.social.slack",
    "aragora.server.handlers.social._slack_impl",
):
    if _mod_name not in sys.modules:
        _m = _types_mod.ModuleType(_mod_name)
        for _a in _SLACK_ATTRS:
            setattr(_m, _a, None)
        sys.modules[_mod_name] = _m

import json
from datetime import datetime

import pytest

from aragora.server.handlers.utils.responses import (
    HandlerResult,
    json_response,
    error_response,
    success_response,
    html_response,
    redirect_response,
)


# =============================================================================
# Test HandlerResult
# =============================================================================


class TestHandlerResult:
    """Tests for HandlerResult dataclass."""

    def test_creates_with_required_fields(self):
        """Should create with required fields."""
        result = HandlerResult(
            status_code=200,
            content_type="application/json",
            body=b'{"ok": true}',
        )

        assert result.status_code == 200
        assert result.content_type == "application/json"
        assert result.body == b'{"ok": true}'
        assert result.headers == {}

    def test_creates_with_optional_headers(self):
        """Should create with optional headers."""
        result = HandlerResult(
            status_code=200,
            content_type="application/json",
            body=b"{}",
            headers={"X-Custom": "value"},
        )

        assert result.headers == {"X-Custom": "value"}

    def test_post_init_defaults_headers_to_empty_dict(self):
        """Should default headers to empty dict in __post_init__."""
        result = HandlerResult(
            status_code=200,
            content_type="text/plain",
            body=b"hello",
            headers=None,
        )

        assert result.headers == {}


# =============================================================================
# Test json_response
# =============================================================================


class TestJsonResponse:
    """Tests for json_response function."""

    def test_creates_json_response_with_data(self):
        """Should create JSON response with data."""
        result = json_response({"key": "value"})

        assert result.status_code == 200
        assert result.content_type == "application/json"
        assert json.loads(result.body) == {"key": "value"}

    def test_uses_custom_status_code(self):
        """Should use custom status code."""
        result = json_response({"created": True}, status=201)

        assert result.status_code == 201

    def test_includes_custom_headers(self):
        """Should include custom headers."""
        result = json_response(
            {"data": "test"},
            headers={"X-Request-Id": "abc123"},
        )

        assert result.headers["X-Request-Id"] == "abc123"

    def test_serializes_non_serializable_with_str(self):
        """Should use str() for non-serializable types."""
        now = datetime(2025, 1, 15, 12, 0, 0)
        result = json_response({"timestamp": now})

        body = json.loads(result.body)
        assert "2025-01-15" in body["timestamp"]

    def test_handles_nested_structures(self):
        """Should handle nested data structures."""
        data = {
            "users": [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}],
            "count": 2,
        }
        result = json_response(data)

        body = json.loads(result.body)
        assert body["users"][0]["name"] == "Alice"
        assert body["count"] == 2

    def test_handles_empty_data(self):
        """Should handle empty data."""
        result = json_response({})
        assert json.loads(result.body) == {}

        result = json_response([])
        assert json.loads(result.body) == []


# =============================================================================
# Test error_response
# =============================================================================


class TestErrorResponse:
    """Tests for error_response function."""

    def test_creates_simple_error(self):
        """Should create simple error response."""
        result = error_response("Something went wrong")

        body = json.loads(result.body)
        assert body == {"error": "Something went wrong"}
        assert result.status_code == 400

    def test_uses_custom_status_code(self):
        """Should use custom status code."""
        result = error_response("Not found", status=404)

        assert result.status_code == 404

    def test_creates_structured_error_with_code(self):
        """Should create structured error when code provided."""
        result = error_response(
            "Invalid email format",
            status=400,
            code="VALIDATION_ERROR",
        )

        body = json.loads(result.body)
        assert body["error"]["message"] == "Invalid email format"
        assert body["error"]["code"] == "VALIDATION_ERROR"

    def test_includes_trace_id(self):
        """Should include trace_id when provided."""
        result = error_response(
            "Server error",
            status=500,
            trace_id="abc123",
        )

        body = json.loads(result.body)
        assert body["error"]["trace_id"] == "abc123"

    def test_includes_suggestion(self):
        """Should include suggestion when provided."""
        result = error_response(
            "Field 'name' is required",
            status=400,
            code="VALIDATION_ERROR",
            suggestion="Include 'name' in request body",
        )

        body = json.loads(result.body)
        assert body["error"]["suggestion"] == "Include 'name' in request body"

    def test_includes_details(self):
        """Should include details when provided."""
        result = error_response(
            "Multiple validation errors",
            status=400,
            details={"fields": ["name", "email"]},
        )

        body = json.loads(result.body)
        assert body["error"]["details"]["fields"] == ["name", "email"]

    def test_forced_structured_mode(self):
        """Should use structured format when structured=True."""
        result = error_response("Simple error", structured=True)

        body = json.loads(result.body)
        assert "message" in body["error"]
        assert body["error"]["message"] == "Simple error"

    def test_includes_custom_headers(self):
        """Should include custom headers."""
        result = error_response(
            "Error",
            headers={"X-Error-Code": "E001"},
        )

        assert result.headers["X-Error-Code"] == "E001"


# =============================================================================
# Test success_response
# =============================================================================


class TestSuccessResponse:
    """Tests for success_response function."""

    def test_creates_success_response(self):
        """Should create success response with data."""
        result = success_response({"id": "123"})

        body = json.loads(result.body)
        assert body["success"] is True
        assert body["data"]["id"] == "123"
        assert result.status_code == 200

    def test_includes_message(self):
        """Should include message when provided."""
        result = success_response(
            {"items": [1, 2, 3]},
            message="Found 3 items",
        )

        body = json.loads(result.body)
        assert body["message"] == "Found 3 items"

    def test_includes_custom_headers(self):
        """Should include custom headers."""
        result = success_response(
            {"ok": True},
            headers={"X-Custom": "value"},
        )

        assert result.headers["X-Custom"] == "value"


# =============================================================================
# Test html_response
# =============================================================================


class TestHtmlResponse:
    """Tests for html_response function."""

    def test_creates_html_response(self):
        """Should create HTML response."""
        result = html_response("<h1>Hello</h1>")

        assert result.status_code == 200
        assert "text/html" in result.content_type
        assert result.body == b"<h1>Hello</h1>"

    def test_uses_custom_status_code(self):
        """Should use custom status code."""
        result = html_response("<h1>Error</h1>", status=500)

        assert result.status_code == 500

    def test_includes_charset_in_content_type(self):
        """Should include charset in content type."""
        result = html_response("<p>Test</p>")

        assert "utf-8" in result.content_type.lower()

    def test_encodes_unicode_content(self):
        """Should properly encode unicode content."""
        result = html_response("<p>Hello, World!</p>")

        assert result.body == b"<p>Hello, World!</p>"


# =============================================================================
# Test redirect_response
# =============================================================================


class TestRedirectResponse:
    """Tests for redirect_response function."""

    def test_creates_redirect_response(self):
        """Should create redirect response."""
        result = redirect_response("https://example.com/new-page")

        assert result.status_code == 302
        assert result.headers["Location"] == "https://example.com/new-page"
        assert result.body == b""

    def test_uses_custom_status_code(self):
        """Should use custom status code."""
        result = redirect_response("https://example.com", status=301)

        assert result.status_code == 301

    def test_uses_307_for_temporary(self):
        """Should support 307 Temporary Redirect."""
        result = redirect_response("https://example.com", status=307)

        assert result.status_code == 307

    def test_preserves_additional_headers(self):
        """Should preserve additional headers."""
        result = redirect_response(
            "https://example.com",
            headers={"X-Redirect-Reason": "Moved"},
        )

        assert result.headers["Location"] == "https://example.com"
        assert result.headers["X-Redirect-Reason"] == "Moved"
