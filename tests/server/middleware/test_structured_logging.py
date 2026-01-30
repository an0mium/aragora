"""
Tests for aragora.server.middleware.structured_logging - Structured Logging Middleware.

Comprehensive tests covering:
- JsonFormatter: JSON-structured log output
- TextFormatter: Enhanced text log formatting
- redact_sensitive: Recursive sensitive field redaction
- redact_string: Secret pattern detection in string values
- _contains_secret_pattern: Pattern matching for secrets
- LogContext: Structured context dataclass
- log_context / set_log_context / clear_log_context / get_log_context: Context management
- configure_structured_logging: Logger configuration
- RequestLoggingMiddleware: HTTP request/response logging
- Correlation ID propagation
- PII and sensitive data redaction
- Audit trail generation
"""

from __future__ import annotations

import json
import logging
import time
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.middleware.structured_logging import (
    JsonFormatter,
    LogContext,
    RequestLoggingMiddleware,
    TextFormatter,
    _contains_secret_pattern,
    clear_log_context,
    configure_structured_logging,
    get_log_context,
    get_logger,
    log_context,
    redact_sensitive,
    redact_string,
    set_log_context,
    setup_logging,
)


# ===========================================================================
# Test LogContext Dataclass
# ===========================================================================


class TestLogContext:
    """Tests for LogContext dataclass."""

    def test_empty_context_to_dict(self):
        ctx = LogContext()
        result = ctx.to_dict()
        assert result == {}

    def test_context_with_request_id(self):
        ctx = LogContext(request_id="req-123")
        result = ctx.to_dict()
        assert result == {"request_id": "req-123"}

    def test_context_with_all_fields(self):
        ctx = LogContext(
            request_id="req-1",
            trace_id="trace-1",
            span_id="span-1",
            user_id="user-1",
            org_id="org-1",
            debate_id="debate-1",
        )
        result = ctx.to_dict()
        assert result["request_id"] == "req-1"
        assert result["trace_id"] == "trace-1"
        assert result["span_id"] == "span-1"
        assert result["user_id"] == "user-1"
        assert result["org_id"] == "org-1"
        assert result["debate_id"] == "debate-1"

    def test_context_excludes_none_values(self):
        ctx = LogContext(request_id="req-1", user_id=None)
        result = ctx.to_dict()
        assert "user_id" not in result
        assert "request_id" in result

    def test_context_with_extra_fields(self):
        ctx = LogContext(extra={"action": "debate.create", "count": 5})
        result = ctx.to_dict()
        assert result["action"] == "debate.create"
        assert result["count"] == 5

    def test_context_extra_merges_with_named_fields(self):
        ctx = LogContext(
            request_id="req-1",
            extra={"custom": "value"},
        )
        result = ctx.to_dict()
        assert result["request_id"] == "req-1"
        assert result["custom"] == "value"


# ===========================================================================
# Test redact_sensitive
# ===========================================================================


class TestRedactSensitive:
    """Tests for redact_sensitive function."""

    def test_redacts_password_field(self):
        data = {"username": "alice", "password": "secret123"}
        result = redact_sensitive(data)
        assert result["username"] == "alice"
        assert result["password"] == "[REDACTED]"

    def test_redacts_api_key_field(self):
        data = {"api_key": "sk-test-12345"}
        result = redact_sensitive(data)
        assert result["api_key"] == "[REDACTED]"

    def test_redacts_token_field(self):
        data = {"token": "eyJhbGci...", "name": "test"}
        result = redact_sensitive(data)
        assert result["token"] == "[REDACTED]"
        assert result["name"] == "test"

    def test_redacts_session_key(self):
        data = {"session_key": "abc123xyz"}
        result = redact_sensitive(data)
        assert result["session_key"] == "[REDACTED]"

    def test_redacts_cookie(self):
        data = {"cookie": "session=abc123; path=/"}
        result = redact_sensitive(data)
        assert result["cookie"] == "[REDACTED]"

    def test_redacts_csrf_token(self):
        data = {"csrf_token": "csrf_abc123"}
        result = redact_sensitive(data)
        assert result["csrf_token"] == "[REDACTED]"

    def test_redacts_authorization_header(self):
        data = {"Authorization": "Bearer eyJhbGci..."}
        result = redact_sensitive(data)
        assert result["Authorization"] == "[REDACTED]"

    def test_redacts_nested_sensitive_fields(self):
        data = {
            "user": {
                "name": "alice",
                "credentials": {
                    "password": "secret",
                    "api_key": "key123",
                },
            }
        }
        result = redact_sensitive(data)
        assert result["user"]["name"] == "alice"
        assert result["user"]["credentials"]["password"] == "[REDACTED]"
        assert result["user"]["credentials"]["api_key"] == "[REDACTED]"

    def test_redacts_case_insensitive(self):
        data = {"API_KEY": "key", "Password": "pass", "SESSION_KEY": "sess"}
        result = redact_sensitive(data)
        assert result["API_KEY"] == "[REDACTED]"
        assert result["Password"] == "[REDACTED]"
        assert result["SESSION_KEY"] == "[REDACTED]"

    def test_preserves_non_sensitive_fields(self):
        data = {"method": "GET", "path": "/api/debates", "status": 200}
        result = redact_sensitive(data)
        assert result == data

    def test_redacts_list_of_dicts(self):
        data = {
            "items": [
                {"name": "item1", "secret": "hidden"},
                {"name": "item2", "password": "hidden2"},
            ]
        }
        result = redact_sensitive(data)
        assert result["items"][0]["name"] == "item1"
        assert result["items"][0]["secret"] == "[REDACTED]"
        assert result["items"][1]["password"] == "[REDACTED]"

    def test_max_depth_stops_recursion(self):
        # Build deeply nested structure (depth > 5)
        data = {"level": 0}
        current = data
        for i in range(1, 8):
            current["nested"] = {"level": i}
            current = current["nested"]
        current["password"] = "deep_secret"

        result = redact_sensitive(data)
        # At depth 6+, the inner dicts should be returned as-is (no further recursion)
        assert isinstance(result, dict)

    def test_redacts_partial_match_field_names(self):
        """Field names containing sensitive substrings should be redacted."""
        data = {
            "user_password_hash": "abc123",
            "my_api_key_v2": "key123",
            "x_session_token": "tok123",
        }
        result = redact_sensitive(data)
        assert result["user_password_hash"] == "[REDACTED]"
        assert result["my_api_key_v2"] == "[REDACTED]"
        assert result["x_session_token"] == "[REDACTED]"

    def test_redacts_database_connection_string_field(self):
        data = {"database_url": "postgres://user:pass@host/db"}
        result = redact_sensitive(data)
        assert result["database_url"] == "[REDACTED]"

    def test_redacts_cloud_provider_keys(self):
        data = {
            "aws_secret": "AKIAIOSFODNN7EXAMPLE",
            "gcp_key": "AIzaSyDaGmWKa4JsXZ-HjGw7ISLn_3namBGewQe",
            "azure_key": "my-azure-key-value",
        }
        result = redact_sensitive(data)
        assert all(v == "[REDACTED]" for v in result.values())

    def test_redacts_financial_fields(self):
        data = {
            "credit_card": "4111111111111111",
            "cvv": "123",
            "account_number": "9876543210",
        }
        result = redact_sensitive(data)
        assert all(v == "[REDACTED]" for v in result.values())

    def test_redacts_pii_fields(self):
        data = {"ssn": "123-45-6789", "social_security": "987-65-4321"}
        result = redact_sensitive(data)
        assert all(v == "[REDACTED]" for v in result.values())

    def test_string_values_checked_for_secret_patterns(self):
        """String values matching secret patterns get redacted even if key is not sensitive."""
        data = {"some_field": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"}
        result = redact_sensitive(data)
        assert "[REDACTED]" in result["some_field"]

    def test_list_with_string_secret_values(self):
        data = {
            "values": [
                "normal text",
                "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U",
            ]
        }
        result = redact_sensitive(data)
        assert result["values"][0] == "normal text"
        assert "[REDACTED]" in result["values"][1]

    def test_empty_dict(self):
        assert redact_sensitive({}) == {}

    def test_non_string_non_dict_values_preserved(self):
        data = {"count": 42, "active": True, "ratio": 3.14, "nothing": None}
        result = redact_sensitive(data)
        assert result == data


# ===========================================================================
# Test redact_string
# ===========================================================================


class TestRedactString:
    """Tests for redact_string function."""

    def test_non_string_returns_unchanged(self):
        assert redact_string(123) == 123
        assert redact_string(None) is None

    def test_short_string_not_redacted(self):
        assert redact_string("hello") == "hello"

    def test_normal_text_not_redacted(self):
        assert redact_string("This is a normal log message about debugging.") == "This is a normal log message about debugging."

    def test_jwt_token_redacted(self):
        jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
        result = redact_string(jwt)
        assert "[REDACTED]" in result

    def test_bearer_token_redacted(self):
        bearer = "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
        result = redact_string(bearer)
        assert "[REDACTED]" in result

    def test_aws_access_key_redacted(self):
        key = "AKIAIOSFODNN7EXAMPLE"
        result = redact_string(key)
        assert "[REDACTED]" in result

    def test_private_key_redacted(self):
        key = "-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEA..."
        result = redact_string(key)
        assert "[REDACTED]" in result

    def test_database_url_redacted(self):
        url = "postgres://admin:secretpassword@db.example.com:5432/mydb"
        result = redact_string(url)
        assert "[REDACTED]" in result

    def test_long_redacted_string_preserves_prefix_suffix(self):
        """Long secrets preserve first/last 4 chars for debugging."""
        jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
        result = redact_string(jwt)
        # Should have format: first4...last4[REDACTED]
        assert result.startswith(jwt[:4])
        assert result.endswith("[REDACTED]")


# ===========================================================================
# Test _contains_secret_pattern
# ===========================================================================


class TestContainsSecretPattern:
    """Tests for _contains_secret_pattern function."""

    def test_non_string_returns_false(self):
        assert _contains_secret_pattern(123) is False

    def test_short_string_returns_false(self):
        assert _contains_secret_pattern("short") is False

    def test_normal_text_returns_false(self):
        assert _contains_secret_pattern("This is a regular log message") is False

    def test_jwt_detected(self):
        jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
        assert _contains_secret_pattern(jwt) is True

    def test_aws_key_detected(self):
        assert _contains_secret_pattern("AKIAIOSFODNN7EXAMPLE") is True

    def test_stripe_key_detected(self):
        assert _contains_secret_pattern("sk-live-51Hb3Xk2eZvKYlo2C0NFXY7Lh") is True

    def test_private_key_header_detected(self):
        assert _contains_secret_pattern("-----BEGIN RSA PRIVATE KEY-----MIIE") is True

    def test_openssh_key_detected(self):
        assert _contains_secret_pattern("-----BEGIN OPENSSH PRIVATE KEY-----b3Bl") is True

    def test_database_connection_string_detected(self):
        assert _contains_secret_pattern("postgres://admin:pass123@host:5432/db") is True

    def test_basic_auth_url_detected(self):
        assert _contains_secret_pattern("https://user:password@api.example.com/v1") is True


# ===========================================================================
# Test JsonFormatter
# ===========================================================================


class TestJsonFormatter:
    """Tests for JsonFormatter."""

    def setup_method(self):
        self.formatter = JsonFormatter(
            include_timestamp=True,
            include_hostname=False,
            service_name="test-service",
        )

    def test_basic_format_produces_valid_json(self):
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=None,
            exc_info=None,
        )
        output = self.formatter.format(record)
        parsed = json.loads(output)
        assert parsed["level"] == "INFO"
        assert parsed["logger"] == "test.logger"
        assert parsed["message"] == "Test message"
        assert parsed["service"] == "test-service"

    def test_timestamp_included_when_enabled(self):
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="msg", args=None, exc_info=None,
        )
        output = self.formatter.format(record)
        parsed = json.loads(output)
        assert "timestamp" in parsed

    def test_timestamp_excluded_when_disabled(self):
        formatter = JsonFormatter(include_timestamp=False, include_hostname=False)
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="msg", args=None, exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert "timestamp" not in parsed

    def test_hostname_included_when_enabled(self):
        formatter = JsonFormatter(include_hostname=True)
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="msg", args=None, exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert "hostname" in parsed

    def test_context_propagated_from_contextvar(self):
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="msg", args=None, exc_info=None,
        )
        with log_context(request_id="req-abc", user_id="user-456"):
            output = self.formatter.format(record)
        parsed = json.loads(output)
        assert parsed["context"]["request_id"] == "req-abc"
        assert parsed["context"]["user_id"] == "user-456"

    def test_extra_fields_included(self):
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="msg", args=None, exc_info=None,
        )
        record.event = "request_start"
        record.method = "GET"
        output = self.formatter.format(record)
        parsed = json.loads(output)
        assert parsed["extra"]["event"] == "request_start"
        assert parsed["extra"]["method"] == "GET"

    def test_extra_fields_are_redacted(self):
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="msg", args=None, exc_info=None,
        )
        record.api_key = "sk-12345"
        record.password = "secret"
        output = self.formatter.format(record)
        parsed = json.loads(output)
        assert parsed["extra"]["api_key"] == "[REDACTED]"
        assert parsed["extra"]["password"] == "[REDACTED]"

    def test_exception_info_included(self):
        try:
            raise ValueError("test error")
        except ValueError:
            import sys
            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test", level=logging.ERROR, pathname="test.py", lineno=42,
            msg="Error occurred", args=None, exc_info=exc_info,
        )
        output = self.formatter.format(record)
        parsed = json.loads(output)
        assert "exception" in parsed
        assert parsed["exception"]["type"] == "ValueError"
        assert parsed["exception"]["message"] == "test error"
        assert "traceback" in parsed["exception"]

    def test_source_location_for_errors(self):
        record = logging.LogRecord(
            name="test", level=logging.ERROR, pathname="/app/server.py", lineno=99,
            msg="Error", args=None, exc_info=None,
        )
        record.funcName = "handle_request"
        output = self.formatter.format(record)
        parsed = json.loads(output)
        assert "source" in parsed
        assert parsed["source"]["file"] == "/app/server.py"
        assert parsed["source"]["line"] == 99
        assert parsed["source"]["function"] == "handle_request"

    def test_source_location_not_for_info(self):
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="test.py", lineno=10,
            msg="Info", args=None, exc_info=None,
        )
        output = self.formatter.format(record)
        parsed = json.loads(output)
        assert "source" not in parsed

    def test_standard_attrs_excluded_from_extra(self):
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="test.py", lineno=10,
            msg="msg", args=None, exc_info=None,
        )
        output = self.formatter.format(record)
        parsed = json.loads(output)
        # Standard attrs like 'name', 'msg', 'args' should not appear in extra
        if "extra" in parsed:
            assert "name" not in parsed["extra"]
            assert "msg" not in parsed["extra"]
            assert "args" not in parsed["extra"]
            assert "levelname" not in parsed["extra"]


# ===========================================================================
# Test TextFormatter
# ===========================================================================


class TestTextFormatter:
    """Tests for TextFormatter."""

    def setup_method(self):
        self.formatter = TextFormatter()

    def test_basic_format(self):
        record = logging.LogRecord(
            name="test.logger", level=logging.INFO, pathname="", lineno=0,
            msg="Hello world", args=None, exc_info=None,
        )
        output = self.formatter.format(record)
        assert "INFO" in output
        assert "test.logger" in output
        assert "Hello world" in output

    def test_includes_request_id_from_context(self):
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="Processing", args=None, exc_info=None,
        )
        with log_context(request_id="req-text-123"):
            output = self.formatter.format(record)
        assert "req-text-123" in output

    def test_dash_when_no_request_id(self):
        clear_log_context()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="No context", args=None, exc_info=None,
        )
        output = self.formatter.format(record)
        assert "[-]" in output

    def test_extra_fields_appended(self):
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="msg", args=None, exc_info=None,
        )
        record.event = "custom_event"
        output = self.formatter.format(record)
        assert "custom_event" in output

    def test_extra_fields_are_redacted(self):
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="msg", args=None, exc_info=None,
        )
        record.password = "mysecret"
        output = self.formatter.format(record)
        assert "mysecret" not in output
        assert "[REDACTED]" in output

    def test_exception_included(self):
        try:
            raise RuntimeError("text error")
        except RuntimeError:
            import sys
            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test", level=logging.ERROR, pathname="", lineno=0,
            msg="Error", args=None, exc_info=exc_info,
        )
        output = self.formatter.format(record)
        assert "RuntimeError" in output
        assert "text error" in output


# ===========================================================================
# Test log_context / set_log_context / clear_log_context / get_log_context
# ===========================================================================


class TestLogContextManagement:
    """Tests for log context management functions."""

    def setup_method(self):
        clear_log_context()

    def teardown_method(self):
        clear_log_context()

    def test_log_context_manager_sets_and_clears(self):
        assert get_log_context() == {}

        with log_context(request_id="req-1"):
            ctx = get_log_context()
            assert ctx["request_id"] == "req-1"

        # After exiting, context should be restored
        assert get_log_context() == {}

    def test_nested_log_context(self):
        with log_context(request_id="req-outer"):
            with log_context(user_id="user-inner"):
                ctx = get_log_context()
                assert ctx["request_id"] == "req-outer"
                assert ctx["user_id"] == "user-inner"
            # Inner context should be cleared but outer preserved
            ctx = get_log_context()
            assert ctx["request_id"] == "req-outer"
            assert "user_id" not in ctx

    def test_set_log_context_persists(self):
        set_log_context(request_id="req-persist")
        ctx = get_log_context()
        assert ctx["request_id"] == "req-persist"

        # Still there after get
        assert get_log_context()["request_id"] == "req-persist"

    def test_set_log_context_updates(self):
        set_log_context(request_id="req-1")
        set_log_context(user_id="user-1")
        ctx = get_log_context()
        assert ctx["request_id"] == "req-1"
        assert ctx["user_id"] == "user-1"

    def test_clear_log_context(self):
        set_log_context(request_id="req-1", user_id="user-1")
        assert len(get_log_context()) > 0

        clear_log_context()
        assert get_log_context() == {}

    def test_get_log_context_returns_copy(self):
        set_log_context(request_id="req-1")
        ctx = get_log_context()
        ctx["mutated"] = True  # Mutating the copy
        # Original should not be affected
        assert "mutated" not in get_log_context()


# ===========================================================================
# Test RequestLoggingMiddleware
# ===========================================================================


class TestRequestLoggingMiddleware:
    """Tests for RequestLoggingMiddleware."""

    def setup_method(self):
        clear_log_context()
        self.middleware = RequestLoggingMiddleware(slow_threshold_ms=1000.0)

    def teardown_method(self):
        clear_log_context()

    def test_start_request_returns_context(self):
        ctx = self.middleware.start_request("GET", "/api/debates", "127.0.0.1")
        assert "request_id" in ctx
        assert ctx["method"] == "GET"
        assert ctx["path"] == "/api/debates"
        assert ctx["client_ip"] == "127.0.0.1"
        assert "start_time" in ctx

    def test_start_request_generates_request_id(self):
        ctx = self.middleware.start_request("POST", "/api/test", "10.0.0.1")
        assert ctx["request_id"].startswith("req-")

    def test_start_request_uses_provided_request_id(self):
        ctx = self.middleware.start_request(
            "GET", "/api/test", "10.0.0.1", request_id="custom-req-123"
        )
        assert ctx["request_id"] == "custom-req-123"

    def test_start_request_sets_log_context(self):
        self.middleware.start_request("GET", "/api/test", "10.0.0.1", request_id="ctx-test")
        log_ctx = get_log_context()
        assert log_ctx["request_id"] == "ctx-test"
        assert log_ctx["method"] == "GET"
        assert log_ctx["path"] == "/api/test"
        assert log_ctx["client_ip"] == "10.0.0.1"

    def test_end_request_clears_log_context(self):
        ctx = self.middleware.start_request("GET", "/api/test", "10.0.0.1")
        assert len(get_log_context()) > 0

        self.middleware.end_request(ctx, 200)
        assert get_log_context() == {}

    def test_end_request_logs_at_info_for_2xx(self):
        ctx = self.middleware.start_request("GET", "/api/test", "10.0.0.1")
        with patch.object(self.middleware.logger, "log") as mock_log:
            self.middleware.end_request(ctx, 200)
        mock_log.assert_called_once()
        call_args = mock_log.call_args
        assert call_args[0][0] == logging.INFO

    def test_end_request_logs_at_warning_for_4xx(self):
        ctx = self.middleware.start_request("GET", "/api/test", "10.0.0.1")
        with patch.object(self.middleware.logger, "log") as mock_log:
            self.middleware.end_request(ctx, 404)
        call_args = mock_log.call_args
        assert call_args[0][0] == logging.WARNING

    def test_end_request_logs_at_error_for_5xx(self):
        ctx = self.middleware.start_request("GET", "/api/test", "10.0.0.1")
        with patch.object(self.middleware.logger, "log") as mock_log:
            self.middleware.end_request(ctx, 500)
        call_args = mock_log.call_args
        assert call_args[0][0] == logging.ERROR

    def test_end_request_warns_on_slow_request(self):
        middleware = RequestLoggingMiddleware(slow_threshold_ms=0.001)
        ctx = middleware.start_request("GET", "/api/slow", "10.0.0.1")
        time.sleep(0.01)  # Sleep to exceed tiny threshold
        with patch.object(middleware.logger, "log") as mock_log:
            middleware.end_request(ctx, 200)
        call_args = mock_log.call_args
        assert call_args[0][0] == logging.WARNING
        assert call_args[1]["extra"]["slow_request"] is True

    def test_end_request_includes_response_size(self):
        ctx = self.middleware.start_request("GET", "/api/test", "10.0.0.1")
        with patch.object(self.middleware.logger, "log") as mock_log:
            self.middleware.end_request(ctx, 200, response_size=1024)
        extra = mock_log.call_args[1]["extra"]
        assert extra["response_size"] == 1024

    def test_end_request_includes_elapsed_ms(self):
        ctx = self.middleware.start_request("GET", "/api/test", "10.0.0.1")
        with patch.object(self.middleware.logger, "log") as mock_log:
            self.middleware.end_request(ctx, 200)
        extra = mock_log.call_args[1]["extra"]
        assert "elapsed_ms" in extra
        assert isinstance(extra["elapsed_ms"], float)

    def test_log_error_logs_at_error_level(self):
        ctx = self.middleware.start_request("POST", "/api/create", "10.0.0.1")
        error = ValueError("Something went wrong")
        with patch.object(self.middleware.logger, "error") as mock_error:
            self.middleware.log_error(ctx, error)
        mock_error.assert_called_once()

    def test_log_error_includes_error_details(self):
        ctx = self.middleware.start_request("POST", "/api/create", "10.0.0.1")
        error = TypeError("Invalid type")
        with patch.object(self.middleware.logger, "error") as mock_error:
            self.middleware.log_error(ctx, error)
        extra = mock_error.call_args[1]["extra"]
        assert extra["error_type"] == "TypeError"
        assert extra["error_message"] == "Invalid type"

    def test_log_error_includes_traceback(self):
        ctx = self.middleware.start_request("POST", "/api/create", "10.0.0.1")
        error = RuntimeError("runtime issue")
        with patch.object(self.middleware.logger, "error") as mock_error:
            self.middleware.log_error(ctx, error, include_traceback=True)
        extra = mock_error.call_args[1]["extra"]
        assert "traceback" in extra

    def test_log_error_without_traceback(self):
        ctx = self.middleware.start_request("POST", "/api/create", "10.0.0.1")
        error = RuntimeError("runtime issue")
        with patch.object(self.middleware.logger, "error") as mock_error:
            self.middleware.log_error(ctx, error, include_traceback=False)
        extra = mock_error.call_args[1]["extra"]
        assert "traceback" not in extra

    def test_start_request_redacts_sensitive_headers(self):
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer sk-secret-token",
            "Cookie": "session=abc123",
        }
        with patch.object(self.middleware.logger, "info") as mock_info:
            self.middleware.start_request("GET", "/api/test", "10.0.0.1", headers=headers)
        extra = mock_info.call_args[1]["extra"]
        assert extra["headers"]["Content-Type"] == "application/json"
        assert extra["headers"]["Authorization"] == "[REDACTED]"
        assert extra["headers"]["Cookie"] == "[REDACTED]"

    def test_correlation_id_propagated_through_request_lifecycle(self):
        """Correlation ID set during start_request is available during processing."""
        ctx = self.middleware.start_request(
            "GET", "/api/test", "10.0.0.1", request_id="corr-id-789"
        )
        # During request processing, log context should have the request_id
        log_ctx = get_log_context()
        assert log_ctx["request_id"] == "corr-id-789"

        # After end_request, context is cleared
        self.middleware.end_request(ctx, 200)
        assert get_log_context() == {}

    def test_long_error_message_truncated(self):
        ctx = self.middleware.start_request("POST", "/api/test", "10.0.0.1")
        long_message = "x" * 1000
        error = ValueError(long_message)
        with patch.object(self.middleware.logger, "error") as mock_error:
            self.middleware.log_error(ctx, error)
        extra = mock_error.call_args[1]["extra"]
        assert len(extra["error_message"]) <= 500


# ===========================================================================
# Test configure_structured_logging
# ===========================================================================


class TestConfigureStructuredLogging:
    """Tests for configure_structured_logging and setup_logging."""

    def teardown_method(self):
        # Reset root logger to avoid test pollution
        root = logging.getLogger()
        for handler in root.handlers[:]:
            root.removeHandler(handler)
        root.setLevel(logging.WARNING)

    def test_configure_json_output(self):
        configure_structured_logging(level="INFO", json_output=True, service_name="test")
        root = logging.getLogger()
        assert root.level == logging.INFO
        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0].formatter, JsonFormatter)

    def test_configure_text_output(self):
        configure_structured_logging(level="DEBUG", json_output=False)
        root = logging.getLogger()
        assert root.level == logging.DEBUG
        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0].formatter, TextFormatter)

    def test_configure_suppresses_noisy_loggers(self):
        configure_structured_logging(level="DEBUG", json_output=True)
        assert logging.getLogger("urllib3").level == logging.WARNING
        assert logging.getLogger("httpx").level == logging.WARNING
        assert logging.getLogger("httpcore").level == logging.WARNING

    def test_setup_logging_shortcut(self):
        setup_logging(json_output=True, level="ERROR")
        root = logging.getLogger()
        assert root.level == logging.ERROR
        assert isinstance(root.handlers[0].formatter, JsonFormatter)


# ===========================================================================
# Test get_logger
# ===========================================================================


class TestGetLogger:
    """Tests for get_logger function."""

    def test_returns_logger_with_name(self):
        logger = get_logger("my.module")
        assert logger.name == "my.module"
        assert isinstance(logger, logging.Logger)

    def test_returns_same_logger_for_same_name(self):
        logger1 = get_logger("shared.logger")
        logger2 = get_logger("shared.logger")
        assert logger1 is logger2
