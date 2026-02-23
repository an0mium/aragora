"""Comprehensive tests for aragora.server.handlers.utils.responses.

Tests cover:
- HandlerResult dataclass (construction, properties, dict conversion, iteration, indexing)
- json_response helper
- error_response helper (simple and structured formats, 5xx sanitization)
- validation_error helper
- not_found_error helper
- permission_denied_error helper
- rate_limit_error helper
- success_response helper
- error_dict helper
- html_response helper
- safe_html_response helper
- redirect_response helper
- paginated_response helper
- parse_pagination_params helper
- normalize_pagination_response helper
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.errors import ErrorCode
from aragora.server.handlers.utils.responses import (
    HandlerResult,
    error_dict,
    error_response,
    html_response,
    json_response,
    not_found_error,
    normalize_pagination_response,
    paginated_response,
    parse_pagination_params,
    permission_denied_error,
    rate_limit_error,
    redirect_response,
    safe_html_response,
    success_response,
    validation_error,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_data():
    """Return a sample dict payload for response construction."""
    return {"id": "abc123", "name": "Test", "items": [1, 2, 3]}


@pytest.fixture
def json_handler_result(sample_data):
    """Return a HandlerResult with JSON content."""
    return json_response(sample_data)


# ===========================================================================
# HandlerResult dataclass
# ===========================================================================


class TestHandlerResult:
    """Tests for the HandlerResult dataclass."""

    def test_construction_basic(self):
        result = HandlerResult(
            status_code=200,
            content_type="application/json",
            body=b'{"ok":true}',
        )
        assert result.status_code == 200
        assert result.content_type == "application/json"
        assert result.body == b'{"ok":true}'
        assert result.headers == {}

    def test_construction_with_headers(self):
        hdrs = {"X-Custom": "value"}
        result = HandlerResult(
            status_code=201,
            content_type="text/plain",
            body=b"created",
            headers=hdrs,
        )
        assert result.headers == {"X-Custom": "value"}

    def test_post_init_sets_headers_to_empty_dict_when_none(self):
        result = HandlerResult(
            status_code=200,
            content_type="text/plain",
            body=b"",
            headers=None,
        )
        assert result.headers == {}
        # Confirm it's a dict, not None
        assert isinstance(result.headers, dict)

    def test_status_property_alias(self):
        result = HandlerResult(status_code=404, content_type="text/plain", body=b"")
        assert result.status == 404
        assert result.status == result.status_code

    # -- to_dict --

    def test_to_dict_with_json_body(self):
        body = json.dumps({"key": "value"}).encode("utf-8")
        result = HandlerResult(
            status_code=200, content_type="application/json", body=body
        )
        d = result.to_dict()
        assert d["status"] == 200
        assert d["body"] == {"key": "value"}
        assert d["content_type"] == "application/json"
        assert isinstance(d["headers"], dict)

    def test_to_dict_with_empty_body(self):
        result = HandlerResult(
            status_code=204, content_type="application/json", body=b""
        )
        d = result.to_dict()
        assert d["body"] == {}

    def test_to_dict_with_non_json_body(self):
        result = HandlerResult(
            status_code=200, content_type="text/plain", body=b"\x80\x81invalid"
        )
        d = result.to_dict()
        # Should gracefully fall back to empty dict
        assert d["body"] == {}

    def test_to_dict_with_invalid_json_body(self):
        result = HandlerResult(
            status_code=200,
            content_type="application/json",
            body=b"not json",
        )
        d = result.to_dict()
        assert d["body"] == {}

    # -- _tuple_body --

    def test_tuple_body_none_body(self):
        result = HandlerResult(
            status_code=200, content_type="text/plain", body=None
        )
        assert result._tuple_body() == b""

    def test_tuple_body_json_content(self):
        body = json.dumps({"x": 1}).encode("utf-8")
        result = HandlerResult(
            status_code=200, content_type="application/json", body=body
        )
        assert result._tuple_body() == {"x": 1}

    def test_tuple_body_json_empty_string(self):
        result = HandlerResult(
            status_code=200, content_type="application/json", body=b""
        )
        assert result._tuple_body() == {}

    def test_tuple_body_scim_content_returns_raw_string(self):
        raw = '{"schemas":["urn:scim"]}'
        result = HandlerResult(
            status_code=200,
            content_type="application/scim+json",
            body=raw.encode("utf-8"),
        )
        tb = result._tuple_body()
        assert isinstance(tb, str)
        assert tb == raw

    def test_tuple_body_non_json_content(self):
        result = HandlerResult(
            status_code=200, content_type="text/html", body=b"<p>hello</p>"
        )
        assert result._tuple_body() == b"<p>hello</p>"

    def test_tuple_body_invalid_json_returns_raw_bytes(self):
        result = HandlerResult(
            status_code=200, content_type="application/json", body=b"\x80bad"
        )
        # Should hit the UnicodeDecodeError/JSONDecodeError fallback
        assert result._tuple_body() == b"\x80bad"

    # -- __iter__ --

    def test_iter_unpacking_json(self):
        body = json.dumps({"a": 1}).encode("utf-8")
        result = HandlerResult(
            status_code=200,
            content_type="application/json",
            body=body,
            headers={"X-H": "v"},
        )
        b, s, h = result
        assert b == {"a": 1}
        assert s == 200
        assert h == {"X-H": "v"}

    def test_iter_unpacking_scim(self):
        raw = '{"schemas":["urn:scim"]}'
        result = HandlerResult(
            status_code=200,
            content_type="application/scim+json",
            body=raw.encode("utf-8"),
        )
        b, s, ct = result
        assert isinstance(b, str)
        assert s == 200
        assert ct == "application/scim+json"

    def test_iter_unpacking_non_json(self):
        result = HandlerResult(
            status_code=200,
            content_type="text/plain",
            body=b"hello",
        )
        b, s, h = result
        assert b == b"hello"
        assert s == 200
        assert h == {}

    # -- __getitem__ --

    def test_getitem_int_0(self, json_handler_result):
        val = json_handler_result[0]
        assert isinstance(val, dict)

    def test_getitem_int_1(self, json_handler_result):
        assert json_handler_result[1] == 200

    def test_getitem_int_2(self, json_handler_result):
        assert isinstance(json_handler_result[2], dict)

    def test_getitem_int_out_of_range(self, json_handler_result):
        with pytest.raises(IndexError):
            json_handler_result[3]

    def test_getitem_string_status(self, json_handler_result):
        assert json_handler_result["status"] == 200

    def test_getitem_string_body(self):
        result = json_response({"x": 1})
        body_str = result["body"]
        assert isinstance(body_str, str)
        assert json.loads(body_str) == {"x": 1}

    def test_getitem_string_body_empty(self):
        result = HandlerResult(
            status_code=200, content_type="text/plain", body=b""
        )
        assert result["body"] == ""

    def test_getitem_string_body_non_decodable(self):
        result = HandlerResult(
            status_code=200, content_type="application/octet-stream", body=b"\x80\x81"
        )
        # Falls back to returning raw bytes
        assert result["body"] == b"\x80\x81"

    def test_getitem_string_headers(self, json_handler_result):
        assert isinstance(json_handler_result["headers"], dict)

    def test_getitem_string_content_type(self, json_handler_result):
        assert json_handler_result["content_type"] == "application/json"

    def test_getitem_string_invalid_key(self, json_handler_result):
        with pytest.raises(KeyError):
            json_handler_result["nonexistent"]


# ===========================================================================
# json_response
# ===========================================================================


class TestJsonResponse:
    """Tests for the json_response helper function."""

    def test_basic_dict(self, sample_data):
        result = json_response(sample_data)
        assert result.status_code == 200
        assert result.content_type == "application/json"
        body = json.loads(result.body)
        assert body == sample_data

    def test_custom_status_code(self):
        result = json_response({"created": True}, status=201)
        assert result.status_code == 201

    def test_with_headers(self):
        hdrs = {"X-Request-Id": "abc"}
        result = json_response({}, headers=hdrs)
        assert result.headers == {"X-Request-Id": "abc"}

    def test_headers_default_empty_dict(self):
        result = json_response({})
        assert result.headers == {}

    def test_list_payload(self):
        result = json_response([1, 2, 3])
        body = json.loads(result.body)
        assert body == [1, 2, 3]

    def test_string_payload(self):
        result = json_response("hello")
        body = json.loads(result.body)
        assert body == "hello"

    def test_non_serializable_uses_str_default(self):
        """Non-serializable objects should be converted via str()."""
        from datetime import datetime

        now = datetime(2025, 1, 15, 12, 0, 0)
        result = json_response({"timestamp": now})
        body = json.loads(result.body)
        assert "2025" in body["timestamp"]

    def test_null_payload(self):
        result = json_response(None)
        body = json.loads(result.body)
        assert body is None

    def test_nested_payload(self):
        data = {"a": {"b": {"c": [1, {"d": True}]}}}
        result = json_response(data)
        body = json.loads(result.body)
        assert body == data

    def test_body_is_bytes(self):
        result = json_response({"x": 1})
        assert isinstance(result.body, bytes)


# ===========================================================================
# error_response
# ===========================================================================


class TestErrorResponse:
    """Tests for the error_response helper function."""

    def test_simple_format(self):
        result = error_response("Bad input", 400)
        body = json.loads(result.body)
        assert body == {"error": "Bad input"}
        assert result.status_code == 400

    def test_default_status_is_400(self):
        result = error_response("Something wrong")
        assert result.status_code == 400

    def test_structured_format_with_code(self):
        result = error_response("Invalid", 400, code="VALIDATION_ERROR")
        body = json.loads(result.body)
        assert body["error"]["code"] == "VALIDATION_ERROR"
        assert body["error"]["message"] == "Invalid"

    def test_structured_format_with_trace_id(self):
        result = error_response("Error", trace_id="trace-123")
        body = json.loads(result.body)
        assert body["error"]["trace_id"] == "trace-123"

    def test_structured_format_with_suggestion(self):
        result = error_response("Error", suggestion="Try again")
        body = json.loads(result.body)
        assert body["error"]["suggestion"] == "Try again"

    def test_structured_format_with_details(self):
        result = error_response("Error", details={"field": "name"})
        body = json.loads(result.body)
        assert body["error"]["details"] == {"field": "name"}

    def test_structured_flag_forces_structured_format(self):
        result = error_response("Simple error", structured=True)
        body = json.loads(result.body)
        assert isinstance(body["error"], dict)
        assert body["error"]["message"] == "Simple error"

    def test_all_structured_fields(self):
        result = error_response(
            "Full error",
            422,
            code="FULL",
            trace_id="t1",
            suggestion="Fix it",
            details={"x": 1},
        )
        body = json.loads(result.body)
        err = body["error"]
        assert err["message"] == "Full error"
        assert err["code"] == "FULL"
        assert err["trace_id"] == "t1"
        assert err["suggestion"] == "Fix it"
        assert err["details"] == {"x": 1}
        assert result.status_code == 422

    def test_custom_headers(self):
        result = error_response("Error", headers={"X-H": "v"})
        assert result.headers["X-H"] == "v"

    def test_5xx_sanitization_in_production(self):
        """In production mode, 5xx error messages should be sanitized."""
        with patch.dict("os.environ", {"ARAGORA_ENV": "production"}):
            result = error_response("SQL error at /db/table", 500)
            body = json.loads(result.body)
            assert body["error"] == "Internal server error"

    def test_5xx_not_sanitized_outside_production(self):
        """Outside production, 5xx error messages are preserved."""
        with patch.dict("os.environ", {"ARAGORA_ENV": "development"}):
            result = error_response("SQL error at /db/table", 500)
            body = json.loads(result.body)
            assert body["error"] == "SQL error at /db/table"

    def test_5xx_sanitization_already_generic(self):
        """Messages already equal to 'Internal server error' should not be logged."""
        with patch.dict("os.environ", {"ARAGORA_ENV": "production"}):
            result = error_response("Internal server error", 500)
            body = json.loads(result.body)
            assert body["error"] == "Internal server error"

    def test_5xx_empty_message_in_production(self):
        """Empty message in production should not be changed (it's falsy)."""
        with patch.dict("os.environ", {"ARAGORA_ENV": "production"}):
            result = error_response("", 500)
            body = json.loads(result.body)
            # Empty string is falsy, so the `if message` check skips it
            assert body["error"] == ""

    def test_4xx_not_sanitized_in_production(self):
        """4xx errors should never be sanitized, even in production."""
        with patch.dict("os.environ", {"ARAGORA_ENV": "production"}):
            result = error_response("Specific validation error", 400)
            body = json.loads(result.body)
            assert body["error"] == "Specific validation error"

    def test_content_type_is_json(self):
        result = error_response("err")
        assert result.content_type == "application/json"


# ===========================================================================
# validation_error
# ===========================================================================


class TestValidationError:
    """Tests for the validation_error helper function."""

    def test_basic(self):
        result = validation_error("Name is required")
        body = json.loads(result.body)
        assert result.status_code == 400
        assert body["error"]["code"] == ErrorCode.VALIDATION_ERROR.value
        assert body["error"]["message"] == "Name is required"

    def test_with_field(self):
        result = validation_error("Required", field="email")
        body = json.loads(result.body)
        assert body["error"]["details"] == {"field": "email"}

    def test_without_field_no_details(self):
        result = validation_error("Invalid format")
        body = json.loads(result.body)
        assert "details" not in body["error"]

    def test_with_trace_id(self):
        result = validation_error("Bad", trace_id="t-abc")
        body = json.loads(result.body)
        assert body["error"]["trace_id"] == "t-abc"

    def test_with_headers(self):
        result = validation_error("Bad", headers={"X-H": "v"})
        assert result.headers["X-H"] == "v"


# ===========================================================================
# not_found_error
# ===========================================================================


class TestNotFoundError:
    """Tests for the not_found_error helper function."""

    def test_basic(self):
        result = not_found_error("Debate")
        body = json.loads(result.body)
        assert result.status_code == 404
        assert body["error"]["code"] == ErrorCode.NOT_FOUND.value
        assert body["error"]["message"] == "Debate not found"
        assert body["error"]["details"]["resource_type"] == "Debate"

    def test_with_resource_id(self):
        result = not_found_error("User", "usr-123")
        body = json.loads(result.body)
        assert body["error"]["details"]["resource_id"] == "usr-123"
        assert body["error"]["details"]["resource_type"] == "User"

    def test_without_resource_id(self):
        result = not_found_error("Template")
        body = json.loads(result.body)
        assert "resource_id" not in body["error"]["details"]

    def test_with_trace_id(self):
        result = not_found_error("Agent", trace_id="t-1")
        body = json.loads(result.body)
        assert body["error"]["trace_id"] == "t-1"

    def test_with_headers(self):
        result = not_found_error("Debate", headers={"X-H": "v"})
        assert result.headers["X-H"] == "v"


# ===========================================================================
# permission_denied_error
# ===========================================================================


class TestPermissionDeniedError:
    """Tests for the permission_denied_error helper function."""

    def test_basic(self):
        result = permission_denied_error()
        body = json.loads(result.body)
        assert result.status_code == 403
        assert body["error"]["code"] == ErrorCode.FORBIDDEN.value
        assert body["error"]["message"] == "Permission denied"

    def test_with_permission(self):
        result = permission_denied_error("backups:read")
        body = json.loads(result.body)
        assert body["error"]["details"]["permission"] == "backups:read"

    def test_without_permission_no_details(self):
        result = permission_denied_error()
        body = json.loads(result.body)
        assert "details" not in body["error"]

    def test_custom_message(self):
        result = permission_denied_error(message="Admin access required")
        body = json.loads(result.body)
        assert body["error"]["message"] == "Admin access required"

    def test_with_trace_id(self):
        result = permission_denied_error(trace_id="t-x")
        body = json.loads(result.body)
        assert body["error"]["trace_id"] == "t-x"

    def test_with_headers(self):
        result = permission_denied_error(headers={"X-H": "v"})
        assert result.headers["X-H"] == "v"


# ===========================================================================
# rate_limit_error
# ===========================================================================


class TestRateLimitError:
    """Tests for the rate_limit_error helper function."""

    def test_basic(self):
        result = rate_limit_error()
        body = json.loads(result.body)
        assert result.status_code == 429
        assert body["error"]["code"] == ErrorCode.RATE_LIMITED.value
        assert body["error"]["message"] == "Rate limit exceeded"

    def test_with_retry_after(self):
        result = rate_limit_error(retry_after=60)
        body = json.loads(result.body)
        assert body["error"]["details"]["retry_after"] == 60
        assert result.headers["Retry-After"] == "60"

    def test_retry_after_zero(self):
        """retry_after=0 is a valid value (immediate retry allowed)."""
        result = rate_limit_error(retry_after=0)
        body = json.loads(result.body)
        assert body["error"]["details"]["retry_after"] == 0
        assert result.headers["Retry-After"] == "0"

    def test_without_retry_after_no_header(self):
        result = rate_limit_error()
        assert "Retry-After" not in result.headers

    def test_without_retry_after_no_details(self):
        result = rate_limit_error()
        body = json.loads(result.body)
        assert "details" not in body["error"]

    def test_custom_message(self):
        result = rate_limit_error(message="Too many requests")
        body = json.loads(result.body)
        assert body["error"]["message"] == "Too many requests"

    def test_with_trace_id(self):
        result = rate_limit_error(trace_id="t-r")
        body = json.loads(result.body)
        assert body["error"]["trace_id"] == "t-r"

    def test_with_additional_headers(self):
        result = rate_limit_error(retry_after=30, headers={"X-Limit": "100"})
        assert result.headers["X-Limit"] == "100"
        assert result.headers["Retry-After"] == "30"

    def test_headers_not_mutated(self):
        """Original headers dict should not be mutated."""
        original = {"X-Existing": "value"}
        original_copy = original.copy()
        rate_limit_error(retry_after=10, headers=original)
        assert original == original_copy


# ===========================================================================
# success_response
# ===========================================================================


class TestSuccessResponse:
    """Tests for the success_response helper function."""

    def test_basic(self, sample_data):
        result = success_response(sample_data)
        body = json.loads(result.body)
        assert result.status_code == 200
        assert body["success"] is True
        assert body["data"] == sample_data

    def test_with_message(self):
        result = success_response([], message="Found 0 items")
        body = json.loads(result.body)
        assert body["message"] == "Found 0 items"

    def test_without_message(self):
        result = success_response({})
        body = json.loads(result.body)
        assert "message" not in body

    def test_with_headers(self):
        result = success_response({}, headers={"X-H": "v"})
        assert result.headers["X-H"] == "v"

    def test_none_data(self):
        result = success_response(None)
        body = json.loads(result.body)
        assert body["success"] is True
        assert body["data"] is None


# ===========================================================================
# error_dict
# ===========================================================================


class TestErrorDict:
    """Tests for the error_dict helper function."""

    def test_simple(self):
        d = error_dict("Something failed")
        assert d == {"error": "Something failed"}

    def test_with_code(self):
        d = error_dict("Not found", code="NOT_FOUND")
        assert d["code"] == "NOT_FOUND"

    def test_with_status(self):
        d = error_dict("Error", status=500)
        assert d["status"] == 500

    def test_with_all_fields(self):
        d = error_dict("Error", code="E1", status=422)
        assert d == {"error": "Error", "code": "E1", "status": 422}

    def test_no_code_no_status(self):
        d = error_dict("Plain error")
        assert "code" not in d
        assert "status" not in d

    def test_status_zero(self):
        """Status=0 is a valid integer and should be included."""
        d = error_dict("Error", status=0)
        assert d["status"] == 0


# ===========================================================================
# html_response
# ===========================================================================


class TestHtmlResponse:
    """Tests for the html_response helper function."""

    def test_basic(self):
        result = html_response("<h1>Hello</h1>")
        assert result.status_code == 200
        assert result.content_type == "text/html; charset=utf-8"
        assert result.body == b"<h1>Hello</h1>"

    def test_custom_status(self):
        result = html_response("<p>Not found</p>", status=404)
        assert result.status_code == 404

    def test_with_headers(self):
        result = html_response("<p>Hi</p>", headers={"X-H": "v"})
        assert result.headers["X-H"] == "v"

    def test_with_nonce_adds_csp_header(self):
        result = html_response("<p>Hi</p>", nonce="abc123")
        csp = result.headers["Content-Security-Policy"]
        assert "'nonce-abc123'" in csp
        assert "default-src 'self'" in csp
        assert "script-src" in csp

    def test_escape_content(self):
        """escape_content=True should escape HTML entities."""
        with patch(
            "aragora.server.middleware.xss_protection.escape_html",
            return_value="&lt;script&gt;alert(&#x27;xss&#x27;)&lt;/script&gt;",
        ) as mock_escape:
            result = html_response(
                "<script>alert('xss')</script>", escape_content=True
            )
            mock_escape.assert_called_once_with("<script>alert('xss')</script>")
        decoded = result.body.decode("utf-8")
        assert "&lt;script&gt;" in decoded

    def test_headers_not_mutated(self):
        original = {"X-Existing": "value"}
        original_copy = original.copy()
        html_response("<p>test</p>", headers=original, nonce="n1")
        assert original == original_copy

    def test_unicode_content(self):
        result = html_response("<p>Caf\u00e9 \u2603</p>")
        decoded = result.body.decode("utf-8")
        assert "Caf\u00e9" in decoded


# ===========================================================================
# safe_html_response
# ===========================================================================


class TestSafeHtmlResponse:
    """Tests for the safe_html_response helper function."""

    def test_with_string_content(self):
        result = safe_html_response("<p>Hello</p>")
        assert result.status_code == 200
        assert result.body == b"<p>Hello</p>"
        assert "text/html" in result.content_type

    def test_with_safe_html_builder(self):
        from aragora.server.middleware.xss_protection import SafeHTMLBuilder

        builder = SafeHTMLBuilder()
        builder.add_raw("<div>Built</div>")
        result = safe_html_response(builder)
        assert result.body == b"<div>Built</div>"

    def test_with_custom_status(self):
        result = safe_html_response("<p>Error</p>", status=500)
        assert result.status_code == 500

    def test_with_nonce(self):
        result = safe_html_response("<p>Hi</p>", nonce="xyz")
        assert "Content-Security-Policy" in result.headers
        assert "'nonce-xyz'" in result.headers["Content-Security-Policy"]

    def test_with_non_string_content(self):
        """Non-string, non-builder content should be converted via str()."""
        result = safe_html_response(42)
        assert result.body == b"42"


# ===========================================================================
# redirect_response
# ===========================================================================


class TestRedirectResponse:
    """Tests for the redirect_response helper function."""

    def test_basic(self):
        result = redirect_response("https://example.com")
        assert result.status_code == 302
        assert result.headers["Location"] == "https://example.com"
        assert result.body == b""
        assert result.content_type == "text/plain"

    def test_custom_status(self):
        result = redirect_response("https://example.com/new", status=301)
        assert result.status_code == 301

    def test_with_additional_headers(self):
        result = redirect_response(
            "https://example.com", headers={"X-H": "v"}
        )
        assert result.headers["Location"] == "https://example.com"
        assert result.headers["X-H"] == "v"

    def test_headers_not_mutated(self):
        original = {"X-Existing": "value"}
        original_copy = original.copy()
        redirect_response("https://example.com", headers=original)
        assert original == original_copy

    def test_relative_url(self):
        result = redirect_response("/login")
        assert result.headers["Location"] == "/login"


# ===========================================================================
# paginated_response
# ===========================================================================


class TestPaginatedResponse:
    """Tests for the paginated_response helper function."""

    def test_basic(self):
        result = paginated_response(
            items=[1, 2, 3], total=10, limit=3, offset=0
        )
        body = json.loads(result.body)
        assert body["data"] == [1, 2, 3]
        assert body["pagination"]["total"] == 10
        assert body["pagination"]["limit"] == 3
        assert body["pagination"]["offset"] == 0
        assert body["pagination"]["has_more"] is True

    def test_last_page(self):
        result = paginated_response(
            items=[8, 9, 10], total=10, limit=3, offset=9
        )
        body = json.loads(result.body)
        assert body["pagination"]["has_more"] is False

    def test_exact_fit(self):
        result = paginated_response(
            items=[1, 2, 3], total=3, limit=3, offset=0
        )
        body = json.loads(result.body)
        assert body["pagination"]["has_more"] is False

    def test_empty_items(self):
        result = paginated_response(items=[], total=0, limit=20, offset=0)
        body = json.loads(result.body)
        assert body["data"] == []
        assert body["pagination"]["has_more"] is False

    def test_with_headers(self):
        result = paginated_response(
            items=[], total=0, limit=10, headers={"X-H": "v"}
        )
        assert result.headers["X-H"] == "v"

    def test_status_is_200(self):
        result = paginated_response(items=[], total=0, limit=10)
        assert result.status_code == 200


# ===========================================================================
# parse_pagination_params
# ===========================================================================


class TestParsePaginationParams:
    """Tests for the parse_pagination_params helper function."""

    def test_defaults(self):
        limit, offset = parse_pagination_params({})
        assert limit == 20
        assert offset == 0

    def test_valid_params(self):
        limit, offset = parse_pagination_params(
            {"limit": "50", "offset": "10"}
        )
        assert limit == 50
        assert offset == 10

    def test_integer_params(self):
        limit, offset = parse_pagination_params({"limit": 30, "offset": 5})
        assert limit == 30
        assert offset == 5

    def test_negative_limit_uses_default(self):
        limit, offset = parse_pagination_params({"limit": "-5"})
        assert limit == 20

    def test_zero_limit_uses_default(self):
        limit, offset = parse_pagination_params({"limit": "0"})
        assert limit == 20

    def test_limit_exceeds_max(self):
        limit, offset = parse_pagination_params({"limit": "500"})
        assert limit == 100

    def test_negative_offset_clamped(self):
        limit, offset = parse_pagination_params({"offset": "-10"})
        assert offset == 0

    def test_custom_defaults(self):
        limit, offset = parse_pagination_params(
            {}, default_limit=50, max_limit=200
        )
        assert limit == 50

    def test_custom_max_limit(self):
        limit, offset = parse_pagination_params(
            {"limit": "150"}, max_limit=200
        )
        assert limit == 150

    def test_non_numeric_limit(self):
        limit, offset = parse_pagination_params({"limit": "abc"})
        assert limit == 20

    def test_non_numeric_offset(self):
        limit, offset = parse_pagination_params({"offset": "xyz"})
        assert offset == 0

    def test_list_value_for_limit(self):
        """Query params that come as lists (e.g., multi-value) use first value."""
        limit, offset = parse_pagination_params({"limit": ["50", "100"]})
        assert limit == 50

    def test_empty_list_value_uses_default(self):
        limit, offset = parse_pagination_params({"limit": []})
        assert limit == 20

    def test_none_value(self):
        limit, offset = parse_pagination_params({"limit": None})
        assert limit == 20


# ===========================================================================
# normalize_pagination_response
# ===========================================================================


class TestNormalizePaginationResponse:
    """Tests for the normalize_pagination_response helper function."""

    def test_already_standard_format(self):
        standard = {
            "data": [1, 2],
            "pagination": {
                "total": 5,
                "limit": 2,
                "offset": 0,
                "has_more": True,
            },
        }
        result = normalize_pagination_response(standard)
        assert result is standard  # Same object returned

    def test_items_format(self):
        legacy = {"items": [1, 2, 3], "total": 10, "limit": 3, "offset": 0}
        result = normalize_pagination_response(legacy)
        assert result["data"] == [1, 2, 3]
        assert result["pagination"]["total"] == 10
        assert result["pagination"]["limit"] == 3
        assert result["pagination"]["offset"] == 0
        assert result["pagination"]["has_more"] is True

    def test_results_format(self):
        legacy = {"results": [1, 2], "total_count": 5}
        result = normalize_pagination_response(legacy)
        assert result["data"] == [1, 2]
        assert result["pagination"]["total"] == 5

    def test_data_with_count_format(self):
        legacy = {"data": [1], "count": 3}
        result = normalize_pagination_response(legacy)
        assert result["data"] == [1]
        assert result["pagination"]["total"] == 3

    def test_has_more_false_when_all_items_shown(self):
        legacy = {"items": [1, 2, 3], "total": 3, "limit": 3, "offset": 0}
        result = normalize_pagination_response(legacy)
        assert result["pagination"]["has_more"] is False

    def test_default_limit_from_items_length(self):
        legacy = {"items": [1, 2, 3], "total": 10}
        result = normalize_pagination_response(legacy)
        assert result["pagination"]["limit"] == 3

    def test_default_limit_20_when_no_items(self):
        legacy = {"total": 10}
        result = normalize_pagination_response(legacy)
        assert result["pagination"]["limit"] == 20

    def test_default_offset_zero(self):
        legacy = {"items": [1], "total": 5}
        result = normalize_pagination_response(legacy)
        assert result["pagination"]["offset"] == 0

    def test_empty_response(self):
        result = normalize_pagination_response({})
        assert result["data"] == []
        assert result["pagination"]["total"] == 0
        assert result["pagination"]["has_more"] is False

    def test_priority_items_over_results(self):
        """'items' key takes priority over 'results'."""
        resp = {"items": [1], "results": [2], "total": 5}
        result = normalize_pagination_response(resp)
        assert result["data"] == [1]


# ===========================================================================
# Integration / cross-function tests
# ===========================================================================


class TestIntegration:
    """Cross-function integration tests."""

    def test_error_response_is_unpacked_correctly(self):
        result = error_response("Bad request", 400)
        body, status, headers = result
        assert body == {"error": "Bad request"}
        assert status == 400

    def test_success_response_is_unpacked_correctly(self):
        result = success_response({"id": 1})
        body, status, headers = result
        assert body["success"] is True
        assert body["data"] == {"id": 1}
        assert status == 200

    def test_paginated_response_to_dict(self):
        result = paginated_response(items=[1], total=5, limit=1, offset=0)
        d = result.to_dict()
        assert d["status"] == 200
        assert d["body"]["pagination"]["has_more"] is True

    def test_handler_result_iteration_count(self):
        result = json_response({"x": 1})
        parts = list(result)
        assert len(parts) == 3

    def test_error_response_structured_with_error_code_enum(self):
        result = error_response(
            "Not found",
            404,
            code=ErrorCode.NOT_FOUND.value,
        )
        body = json.loads(result.body)
        assert body["error"]["code"] == "NOT_FOUND"
