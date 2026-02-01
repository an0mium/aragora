"""Tests for standardized error response helpers.

Tests the error response helper functions that provide consistent,
structured error responses across the API.
"""

from __future__ import annotations

import json
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

import pytest

from aragora.server.errors import ErrorCode
from aragora.server.handlers.utils.responses import (
    error_response,
    not_found_error,
    permission_denied_error,
    rate_limit_error,
    validation_error,
)


# =============================================================================
# Test validation_error
# =============================================================================


class TestValidationError:
    """Tests for validation_error helper function."""

    def test_validation_error_format(self):
        """Verify validation error uses structured format with code."""
        result = validation_error("Field 'name' is required")

        body = json.loads(result.body)
        assert result.status_code == 400
        assert body["error"]["code"] == "VALIDATION_ERROR"
        assert body["error"]["message"] == "Field 'name' is required"

    def test_validation_error_includes_field(self):
        """Verify field is included in details when provided."""
        result = validation_error("Invalid email format", field="email")

        body = json.loads(result.body)
        assert body["error"]["details"]["field"] == "email"

    def test_validation_error_without_field(self):
        """Verify validation error works without field parameter."""
        result = validation_error("Invalid request format")

        body = json.loads(result.body)
        assert "details" not in body["error"]
        assert body["error"]["message"] == "Invalid request format"

    def test_validation_error_with_trace_id(self):
        """Verify trace_id is included when provided."""
        result = validation_error("Invalid input", trace_id="req-12345")

        body = json.loads(result.body)
        assert body["error"]["trace_id"] == "req-12345"

    def test_validation_error_with_headers(self):
        """Verify custom headers are included."""
        result = validation_error("Error", headers={"X-Custom": "value"})

        assert result.headers["X-Custom"] == "value"


# =============================================================================
# Test not_found_error
# =============================================================================


class TestNotFoundError:
    """Tests for not_found_error helper function."""

    def test_not_found_error_format(self):
        """Verify not found error uses structured format with resource info."""
        result = not_found_error("Debate", "abc123")

        body = json.loads(result.body)
        assert result.status_code == 404
        assert body["error"]["code"] == "NOT_FOUND"
        assert body["error"]["message"] == "Debate not found"
        assert body["error"]["details"]["resource_type"] == "Debate"
        assert body["error"]["details"]["resource_id"] == "abc123"

    def test_not_found_error_without_id(self):
        """Verify not found error works without resource_id."""
        result = not_found_error("User")

        body = json.loads(result.body)
        assert body["error"]["message"] == "User not found"
        assert body["error"]["details"]["resource_type"] == "User"
        assert "resource_id" not in body["error"]["details"]

    def test_not_found_error_with_trace_id(self):
        """Verify trace_id is included when provided."""
        result = not_found_error("Agent", "agent-1", trace_id="trace-999")

        body = json.loads(result.body)
        assert body["error"]["trace_id"] == "trace-999"

    def test_not_found_error_resource_type_in_message(self):
        """Verify resource type is properly included in message."""
        result = not_found_error("Organization")

        body = json.loads(result.body)
        assert "Organization" in body["error"]["message"]


# =============================================================================
# Test permission_denied_error
# =============================================================================


class TestPermissionDeniedError:
    """Tests for permission_denied_error helper function."""

    def test_permission_denied_error_format(self):
        """Verify permission denied error uses structured format."""
        result = permission_denied_error("backups:read")

        body = json.loads(result.body)
        assert result.status_code == 403
        assert body["error"]["code"] == "FORBIDDEN"
        assert body["error"]["message"] == "Permission denied"
        assert body["error"]["details"]["permission"] == "backups:read"

    def test_permission_denied_error_without_permission(self):
        """Verify permission denied works without specific permission."""
        result = permission_denied_error()

        body = json.loads(result.body)
        assert body["error"]["message"] == "Permission denied"
        assert "details" not in body["error"]

    def test_permission_denied_error_custom_message(self):
        """Verify custom message can be provided."""
        result = permission_denied_error(message="Admin access required")

        body = json.loads(result.body)
        assert body["error"]["message"] == "Admin access required"

    def test_permission_denied_error_with_trace_id(self):
        """Verify trace_id is included when provided."""
        result = permission_denied_error("debates:create", trace_id="auth-check-123")

        body = json.loads(result.body)
        assert body["error"]["trace_id"] == "auth-check-123"


# =============================================================================
# Test rate_limit_error
# =============================================================================


class TestRateLimitError:
    """Tests for rate_limit_error helper function."""

    def test_rate_limit_error_includes_retry_after(self):
        """Verify Retry-After header is included when retry_after provided."""
        result = rate_limit_error(retry_after=60)

        body = json.loads(result.body)
        assert result.status_code == 429
        assert body["error"]["code"] == "RATE_LIMITED"
        assert body["error"]["message"] == "Rate limit exceeded"
        assert body["error"]["details"]["retry_after"] == 60
        assert result.headers["Retry-After"] == "60"

    def test_rate_limit_error_without_retry_after(self):
        """Verify rate limit error works without retry_after."""
        result = rate_limit_error()

        body = json.loads(result.body)
        assert body["error"]["message"] == "Rate limit exceeded"
        assert "details" not in body["error"]
        assert "Retry-After" not in result.headers

    def test_rate_limit_error_custom_message(self):
        """Verify custom message can be provided."""
        result = rate_limit_error(message="Too many requests per minute")

        body = json.loads(result.body)
        assert body["error"]["message"] == "Too many requests per minute"

    def test_rate_limit_error_with_trace_id(self):
        """Verify trace_id is included when provided."""
        result = rate_limit_error(retry_after=30, trace_id="rate-limit-hit")

        body = json.loads(result.body)
        assert body["error"]["trace_id"] == "rate-limit-hit"

    def test_rate_limit_error_preserves_custom_headers(self):
        """Verify custom headers are preserved alongside Retry-After."""
        result = rate_limit_error(
            retry_after=45,
            headers={"X-RateLimit-Limit": "100", "X-RateLimit-Remaining": "0"},
        )

        assert result.headers["Retry-After"] == "45"
        assert result.headers["X-RateLimit-Limit"] == "100"
        assert result.headers["X-RateLimit-Remaining"] == "0"


# =============================================================================
# Test error code constants
# =============================================================================


class TestErrorCodeConstants:
    """Tests for ErrorCode constants."""

    def test_error_code_constants_unique(self):
        """Verify all error codes are unique."""
        codes = [member.value for member in ErrorCode]
        assert len(codes) == len(set(codes)), "Error codes must be unique"

    def test_error_codes_are_strings(self):
        """Verify all error codes are strings."""
        for member in ErrorCode:
            assert isinstance(member.value, str), f"{member.name} value must be a string"

    def test_expected_codes_exist(self):
        """Verify key error codes exist."""
        expected = [
            "VALIDATION_ERROR",
            "NOT_FOUND",
            "FORBIDDEN",
            "RATE_LIMITED",
            "INTERNAL_ERROR",
            "UNAUTHORIZED",
        ]
        code_values = [member.value for member in ErrorCode]
        for expected_code in expected:
            assert expected_code in code_values, f"{expected_code} should exist"


# =============================================================================
# Test backward compatibility
# =============================================================================


class TestBackwardCompatibility:
    """Tests for backward compatibility of error_response."""

    def test_backward_compatible_simple_error(self):
        """Verify error_response without code returns simple format."""
        result = error_response("Something went wrong")

        body = json.loads(result.body)
        assert body == {"error": "Something went wrong"}
        assert result.status_code == 400

    def test_backward_compatible_with_status(self):
        """Verify error_response with only status works."""
        result = error_response("Not found", status=404)

        body = json.loads(result.body)
        assert body == {"error": "Not found"}
        assert result.status_code == 404

    def test_backward_compatible_with_headers(self):
        """Verify error_response with headers but no code returns simple format."""
        result = error_response("Error", headers={"X-Custom": "value"})

        body = json.loads(result.body)
        assert body == {"error": "Error"}
        assert result.headers["X-Custom"] == "value"


# =============================================================================
# Test structured error format
# =============================================================================


class TestStructuredErrorFormat:
    """Tests for structured error format."""

    def test_structured_error_always_has_code(self):
        """Verify structured format always includes code when provided."""
        result = error_response("Test error", code="CUSTOM_ERROR")

        body = json.loads(result.body)
        assert "error" in body
        assert isinstance(body["error"], dict)
        assert body["error"]["code"] == "CUSTOM_ERROR"
        assert body["error"]["message"] == "Test error"

    def test_trace_id_included_when_provided(self):
        """Verify trace_id passthrough in structured format."""
        result = error_response("Error", trace_id="abc123")

        body = json.loads(result.body)
        assert body["error"]["trace_id"] == "abc123"

    def test_structured_with_all_fields(self):
        """Verify all fields work together in structured format."""
        result = error_response(
            "Complete error",
            status=500,
            code="INTERNAL_ERROR",
            trace_id="trace-xyz",
            suggestion="Try again later",
            details={"context": "test"},
        )

        body = json.loads(result.body)
        assert result.status_code == 500
        assert body["error"]["code"] == "INTERNAL_ERROR"
        assert body["error"]["message"] == "Complete error"
        assert body["error"]["trace_id"] == "trace-xyz"
        assert body["error"]["suggestion"] == "Try again later"
        assert body["error"]["details"]["context"] == "test"

    def test_structured_mode_forced(self):
        """Verify structured=True forces structured format even without code."""
        result = error_response("Simple message", structured=True)

        body = json.loads(result.body)
        assert isinstance(body["error"], dict)
        assert body["error"]["message"] == "Simple message"


# =============================================================================
# Test error response integration
# =============================================================================


class TestErrorResponseIntegration:
    """Integration tests for error response helpers."""

    def test_helpers_return_handler_result(self):
        """Verify all helpers return HandlerResult with correct type."""
        from aragora.server.handlers.utils.responses import HandlerResult

        results = [
            validation_error("Test"),
            not_found_error("Resource"),
            permission_denied_error(),
            rate_limit_error(),
        ]

        for result in results:
            assert isinstance(result, HandlerResult)
            assert isinstance(result.body, bytes)
            assert result.content_type == "application/json"

    def test_all_helpers_produce_valid_json(self):
        """Verify all helpers produce valid JSON."""
        results = [
            validation_error("Test", field="name"),
            not_found_error("Debate", "123"),
            permission_denied_error("read"),
            rate_limit_error(60),
        ]

        for result in results:
            # Should not raise
            body = json.loads(result.body)
            assert "error" in body

    def test_error_codes_match_status_codes(self):
        """Verify error codes match appropriate HTTP status codes."""
        # 400 - Validation Error
        result = validation_error("Invalid")
        assert result.status_code == 400

        # 403 - Permission Denied
        result = permission_denied_error()
        assert result.status_code == 403

        # 404 - Not Found
        result = not_found_error("Item")
        assert result.status_code == 404

        # 429 - Rate Limited
        result = rate_limit_error()
        assert result.status_code == 429
