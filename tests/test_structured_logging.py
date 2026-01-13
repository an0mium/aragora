"""Tests for structured logging middleware.

Tests cover:
- LogContext dataclass
- Sensitive field redaction
- JsonFormatter and TextFormatter
- Context variable propagation
- RequestLoggingMiddleware
"""

import json
import logging
import time
from io import StringIO
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.middleware.structured_logging import (
    REDACT_FIELDS,
    JsonFormatter,
    LogContext,
    RequestLoggingMiddleware,
    TextFormatter,
    clear_log_context,
    configure_structured_logging,
    get_log_context,
    get_logger,
    log_context,
    redact_sensitive,
    set_log_context,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def reset_log_context():
    """Reset log context before and after each test."""
    clear_log_context()
    yield
    clear_log_context()


@pytest.fixture
def mock_log_record():
    """Create a basic log record for testing formatters."""
    return logging.LogRecord(
        name="test.logger",
        level=logging.INFO,
        pathname="/test/path.py",
        lineno=42,
        msg="Test message",
        args=(),
        exc_info=None,
    )


@pytest.fixture
def string_handler():
    """Create a handler that writes to a StringIO."""
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    return handler, stream


# =============================================================================
# LogContext Tests
# =============================================================================


class TestLogContext:
    """Test LogContext dataclass."""

    def test_all_fields_none_by_default(self):
        """All fields are None by default."""
        ctx = LogContext()
        assert ctx.request_id is None
        assert ctx.trace_id is None
        assert ctx.span_id is None
        assert ctx.user_id is None
        assert ctx.org_id is None
        assert ctx.debate_id is None

    def test_to_dict_excludes_none_values(self):
        """to_dict excludes None values."""
        ctx = LogContext()
        assert ctx.to_dict() == {}

    def test_to_dict_includes_non_none_values(self):
        """to_dict includes non-None values."""
        ctx = LogContext(request_id="req-123", user_id="user-456")
        result = ctx.to_dict()
        assert result["request_id"] == "req-123"
        assert result["user_id"] == "user-456"
        assert "trace_id" not in result

    def test_to_dict_includes_extra_fields(self):
        """to_dict includes extra fields."""
        ctx = LogContext(extra={"custom_field": "value"})
        result = ctx.to_dict()
        assert result["custom_field"] == "value"

    def test_extra_default_empty_dict(self):
        """Extra defaults to empty dict."""
        ctx = LogContext()
        assert ctx.extra == {}

    def test_all_fields_present_in_dict(self):
        """All fields present when all are set."""
        ctx = LogContext(
            request_id="req",
            trace_id="trace",
            span_id="span",
            user_id="user",
            org_id="org",
            debate_id="debate",
        )
        result = ctx.to_dict()
        assert len(result) == 6


# =============================================================================
# redact_sensitive Tests
# =============================================================================


class TestRedactSensitive:
    """Test redact_sensitive() function."""

    def test_redacts_password_field(self):
        """Redacts password field."""
        data = {"password": "secret123"}
        result = redact_sensitive(data)
        assert result["password"] == "[REDACTED]"

    def test_redacts_secret_field(self):
        """Redacts secret field."""
        data = {"secret": "mysecret"}
        result = redact_sensitive(data)
        assert result["secret"] == "[REDACTED]"

    def test_redacts_token_field(self):
        """Redacts token field."""
        data = {"auth_token": "abc123"}
        result = redact_sensitive(data)
        assert result["auth_token"] == "[REDACTED]"

    def test_redacts_api_key_field(self):
        """Redacts api_key field."""
        data = {"api_key": "sk-xxx"}
        result = redact_sensitive(data)
        assert result["api_key"] == "[REDACTED]"

    def test_redacts_apikey_no_underscore(self):
        """Redacts apikey (no underscore)."""
        data = {"apikey": "key123"}
        result = redact_sensitive(data)
        assert result["apikey"] == "[REDACTED]"

    def test_redacts_authorization_field(self):
        """Redacts authorization field."""
        data = {"authorization": "Bearer xxx"}
        result = redact_sensitive(data)
        assert result["authorization"] == "[REDACTED]"

    def test_redacts_cookie_field(self):
        """Redacts cookie field."""
        data = {"cookie": "session=abc"}
        result = redact_sensitive(data)
        assert result["cookie"] == "[REDACTED]"

    def test_redacts_credit_card_field(self):
        """Redacts credit_card field."""
        data = {"credit_card": "4111111111111111"}
        result = redact_sensitive(data)
        assert result["credit_card"] == "[REDACTED]"

    def test_redacts_ssn_field(self):
        """Redacts ssn field."""
        data = {"ssn": "123-45-6789"}
        result = redact_sensitive(data)
        assert result["ssn"] == "[REDACTED]"

    def test_redacts_private_key_field(self):
        """Redacts private_key field."""
        data = {"private_key": "-----BEGIN RSA KEY-----"}
        result = redact_sensitive(data)
        assert result["private_key"] == "[REDACTED]"

    def test_case_insensitive_PASSWORD(self):
        """Redaction is case insensitive."""
        data = {"PASSWORD": "secret", "Password": "secret2"}
        result = redact_sensitive(data)
        assert result["PASSWORD"] == "[REDACTED]"
        assert result["Password"] == "[REDACTED]"

    def test_case_insensitive_API_KEY(self):
        """API_KEY is redacted (case insensitive)."""
        data = {"API_KEY": "key123", "X_API_KEY": "key456"}
        result = redact_sensitive(data)
        assert result["API_KEY"] == "[REDACTED]"
        assert result["X_API_KEY"] == "[REDACTED]"  # Contains "api_key" substring

    def test_redacts_nested_dicts(self):
        """Redacts sensitive fields in nested dicts."""
        data = {"user": {"password": "secret", "name": "John"}}
        result = redact_sensitive(data)
        assert result["user"]["password"] == "[REDACTED]"
        assert result["user"]["name"] == "John"

    def test_redacts_dicts_in_lists(self):
        """Redacts sensitive fields in dicts inside lists."""
        data = {"items": [{"token": "abc"}, {"name": "test"}]}
        result = redact_sensitive(data)
        assert result["items"][0]["token"] == "[REDACTED]"
        assert result["items"][1]["name"] == "test"

    def test_max_depth_limit(self):
        """Stops recursing after depth 5."""
        # Create deeply nested structure
        nested = {"password": "secret"}
        for _ in range(10):
            nested = {"nested": nested}

        result = redact_sensitive(nested)
        # Should not raise, but won't redact past depth 5
        assert result is not None

    def test_preserves_normal_fields(self):
        """Non-sensitive fields are preserved."""
        data = {"name": "John", "email": "john@example.com", "age": 30}
        result = redact_sensitive(data)
        assert result == data

    def test_preserves_non_dict_values(self):
        """Non-dict values in lists are preserved."""
        data = {"tags": ["a", "b", "c"], "count": 5}
        result = redact_sensitive(data)
        assert result["tags"] == ["a", "b", "c"]
        assert result["count"] == 5


# =============================================================================
# JsonFormatter Tests
# =============================================================================


class TestJsonFormatter:
    """Test JsonFormatter class."""

    def test_basic_json_output(self, mock_log_record):
        """Produces valid JSON output."""
        formatter = JsonFormatter()
        output = formatter.format(mock_log_record)
        parsed = json.loads(output)
        assert parsed["level"] == "INFO"
        assert parsed["logger"] == "test.logger"
        assert parsed["message"] == "Test message"

    def test_includes_timestamp_when_enabled(self, mock_log_record):
        """Includes timestamp when enabled."""
        formatter = JsonFormatter(include_timestamp=True)
        output = formatter.format(mock_log_record)
        parsed = json.loads(output)
        assert "timestamp" in parsed

    def test_excludes_timestamp_when_disabled(self, mock_log_record):
        """Excludes timestamp when disabled."""
        formatter = JsonFormatter(include_timestamp=False)
        output = formatter.format(mock_log_record)
        parsed = json.loads(output)
        assert "timestamp" not in parsed

    def test_includes_hostname_when_enabled(self, mock_log_record):
        """Includes hostname when enabled."""
        formatter = JsonFormatter(include_hostname=True)
        output = formatter.format(mock_log_record)
        parsed = json.loads(output)
        assert "hostname" in parsed

    def test_excludes_hostname_when_disabled(self, mock_log_record):
        """Excludes hostname when disabled."""
        formatter = JsonFormatter(include_hostname=False)
        output = formatter.format(mock_log_record)
        parsed = json.loads(output)
        assert "hostname" not in parsed

    def test_includes_service_name(self, mock_log_record):
        """Includes service name."""
        formatter = JsonFormatter(service_name="my-service")
        output = formatter.format(mock_log_record)
        parsed = json.loads(output)
        assert parsed["service"] == "my-service"

    def test_includes_context_from_contextvar(self, mock_log_record):
        """Includes context from ContextVar."""
        set_log_context(request_id="req-123", user_id="user-456")
        formatter = JsonFormatter()
        output = formatter.format(mock_log_record)
        parsed = json.loads(output)
        assert parsed["context"]["request_id"] == "req-123"
        assert parsed["context"]["user_id"] == "user-456"

    def test_includes_extra_fields_from_record(self, mock_log_record):
        """Includes extra fields from log record."""
        mock_log_record.action = "debate.create"
        mock_log_record.debate_id = "deb-123"
        formatter = JsonFormatter()
        output = formatter.format(mock_log_record)
        parsed = json.loads(output)
        assert parsed["extra"]["action"] == "debate.create"
        assert parsed["extra"]["debate_id"] == "deb-123"

    def test_redacts_sensitive_extra_fields(self, mock_log_record):
        """Redacts sensitive extra fields."""
        mock_log_record.api_key = "sk-secret123"
        formatter = JsonFormatter()
        output = formatter.format(mock_log_record)
        parsed = json.loads(output)
        assert parsed["extra"]["api_key"] == "[REDACTED]"

    def test_includes_exception_info(self):
        """Includes exception info when present."""
        try:
            raise ValueError("test error")
        except ValueError:
            import sys

            record = logging.LogRecord(
                name="test",
                level=logging.ERROR,
                pathname="",
                lineno=1,
                msg="Error",
                args=(),
                exc_info=sys.exc_info(),
            )
        formatter = JsonFormatter()
        output = formatter.format(record)
        parsed = json.loads(output)
        assert "exception" in parsed
        assert parsed["exception"]["type"] == "ValueError"
        assert parsed["exception"]["message"] == "test error"

    def test_includes_source_location_for_errors(self):
        """Includes source location for ERROR level."""
        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="/app/main.py",
            lineno=100,
            msg="Error",
            args=(),
            exc_info=None,
        )
        record.funcName = "handle_request"
        formatter = JsonFormatter()
        output = formatter.format(record)
        parsed = json.loads(output)
        assert "source" in parsed
        assert parsed["source"]["file"] == "/app/main.py"
        assert parsed["source"]["line"] == 100


# =============================================================================
# TextFormatter Tests
# =============================================================================


class TestTextFormatter:
    """Test TextFormatter class."""

    def test_basic_text_output(self, mock_log_record):
        """Produces text output."""
        formatter = TextFormatter()
        output = formatter.format(mock_log_record)
        assert "Test message" in output

    def test_includes_timestamp(self, mock_log_record):
        """Includes timestamp."""
        formatter = TextFormatter()
        output = formatter.format(mock_log_record)
        # Timestamp format: [YYYY-MM-DD HH:MM:SS.mmm]
        assert "[" in output

    def test_includes_level(self, mock_log_record):
        """Includes log level."""
        formatter = TextFormatter()
        output = formatter.format(mock_log_record)
        assert "INFO" in output

    def test_includes_request_id_from_context(self, mock_log_record):
        """Includes request_id from context."""
        set_log_context(request_id="req-abc123")
        formatter = TextFormatter()
        output = formatter.format(mock_log_record)
        assert "req-abc123" in output

    def test_includes_logger_name(self, mock_log_record):
        """Includes logger name."""
        formatter = TextFormatter()
        output = formatter.format(mock_log_record)
        assert "test.logger" in output

    def test_includes_message(self, mock_log_record):
        """Includes message."""
        formatter = TextFormatter()
        output = formatter.format(mock_log_record)
        assert "Test message" in output

    def test_includes_extra_fields_as_json(self, mock_log_record):
        """Extra fields are included as JSON."""
        mock_log_record.action = "create"
        formatter = TextFormatter()
        output = formatter.format(mock_log_record)
        assert '"action"' in output
        assert '"create"' in output

    def test_redacts_sensitive_extra_fields(self, mock_log_record):
        """Redacts sensitive extra fields."""
        mock_log_record.password = "secret"
        formatter = TextFormatter()
        output = formatter.format(mock_log_record)
        assert "REDACTED" in output
        assert "secret" not in output

    def test_includes_exception_traceback(self):
        """Includes exception traceback."""
        try:
            raise ValueError("test error")
        except ValueError:
            import sys

            record = logging.LogRecord(
                name="test",
                level=logging.ERROR,
                pathname="",
                lineno=1,
                msg="Error",
                args=(),
                exc_info=sys.exc_info(),
            )
        formatter = TextFormatter()
        output = formatter.format(record)
        assert "ValueError" in output
        assert "test error" in output


# =============================================================================
# configure_structured_logging Tests
# =============================================================================


class TestConfigureStructuredLogging:
    """Test configure_structured_logging() function."""

    def test_sets_log_level(self):
        """Sets root logger level."""
        configure_structured_logging(level="DEBUG")
        root = logging.getLogger()
        assert root.level == logging.DEBUG

    def test_removes_existing_handlers(self):
        """Removes existing handlers."""
        root = logging.getLogger()
        root.addHandler(logging.StreamHandler())
        initial_count = len(root.handlers)

        configure_structured_logging()

        # Should have exactly one handler after configure
        assert len(root.handlers) == 1

    def test_suppresses_urllib3_logger(self):
        """Suppresses urllib3 logger."""
        configure_structured_logging()
        logger = logging.getLogger("urllib3")
        assert logger.level >= logging.WARNING

    def test_suppresses_httpx_logger(self):
        """Suppresses httpx logger."""
        configure_structured_logging()
        logger = logging.getLogger("httpx")
        assert logger.level >= logging.WARNING

    def test_suppresses_httpcore_logger(self):
        """Suppresses httpcore logger."""
        configure_structured_logging()
        logger = logging.getLogger("httpcore")
        assert logger.level >= logging.WARNING


# =============================================================================
# log_context Tests
# =============================================================================


class TestLogContextManager:
    """Test log_context() context manager."""

    def test_sets_context_fields(self):
        """Sets context fields within manager."""
        with log_context(request_id="req-123"):
            ctx = get_log_context()
            assert ctx["request_id"] == "req-123"

    def test_clears_context_on_exit(self):
        """Context is reset on exit."""
        with log_context(request_id="req-123"):
            pass
        ctx = get_log_context()
        assert ctx == {}

    def test_nested_contexts_merge(self):
        """Nested contexts merge fields."""
        with log_context(request_id="req-123"):
            with log_context(user_id="user-456"):
                ctx = get_log_context()
                assert ctx["request_id"] == "req-123"
                assert ctx["user_id"] == "user-456"

    def test_inner_context_overrides_outer(self):
        """Inner context overrides outer for same key."""
        with log_context(request_id="outer"):
            with log_context(request_id="inner"):
                ctx = get_log_context()
                assert ctx["request_id"] == "inner"

    def test_restores_outer_context_after_inner(self):
        """Outer context is restored after inner exits."""
        with log_context(request_id="outer"):
            with log_context(request_id="inner"):
                pass
            ctx = get_log_context()
            assert ctx["request_id"] == "outer"

    def test_original_context_restored_on_exception(self):
        """Context is restored even on exception."""
        try:
            with log_context(request_id="req-123"):
                raise ValueError("test")
        except ValueError:
            pass
        ctx = get_log_context()
        assert ctx == {}

    def test_multiple_fields_at_once(self):
        """Can set multiple fields at once."""
        with log_context(request_id="req", user_id="user", debug=True):
            ctx = get_log_context()
            assert ctx["request_id"] == "req"
            assert ctx["user_id"] == "user"
            assert ctx["debug"] is True

    def test_empty_context_manager(self):
        """Empty context manager works."""
        with log_context():
            ctx = get_log_context()
            assert ctx == {}


# =============================================================================
# Context Functions Tests
# =============================================================================


class TestContextFunctions:
    """Test set_log_context, clear_log_context, get_log_context."""

    def test_set_log_context_adds_fields(self):
        """set_log_context adds fields."""
        set_log_context(request_id="req-123")
        ctx = get_log_context()
        assert ctx["request_id"] == "req-123"

    def test_set_log_context_merges(self):
        """set_log_context merges with existing context."""
        set_log_context(request_id="req-123")
        set_log_context(user_id="user-456")
        ctx = get_log_context()
        assert ctx["request_id"] == "req-123"
        assert ctx["user_id"] == "user-456"

    def test_clear_log_context_removes_all(self):
        """clear_log_context removes all fields."""
        set_log_context(request_id="req-123", user_id="user-456")
        clear_log_context()
        ctx = get_log_context()
        assert ctx == {}

    def test_get_log_context_returns_copy(self):
        """get_log_context returns a copy."""
        set_log_context(request_id="req-123")
        ctx1 = get_log_context()
        ctx2 = get_log_context()
        assert ctx1 == ctx2
        assert ctx1 is not ctx2

    def test_get_log_context_mutation_isolation(self):
        """Mutating returned context doesn't affect original."""
        set_log_context(request_id="req-123")
        ctx = get_log_context()
        ctx["request_id"] = "modified"
        original = get_log_context()
        assert original["request_id"] == "req-123"

    def test_get_logger_returns_logger(self):
        """get_logger returns a Logger instance."""
        logger = get_logger("test.module")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test.module"


# =============================================================================
# RequestLoggingMiddleware Tests
# =============================================================================


class TestRequestLoggingMiddleware:
    """Test RequestLoggingMiddleware class."""

    @pytest.fixture
    def middleware(self):
        """Create middleware instance."""
        return RequestLoggingMiddleware(slow_threshold_ms=500.0)

    def test_default_slow_threshold(self):
        """Default slow threshold is 1000ms."""
        mw = RequestLoggingMiddleware()
        assert mw.slow_threshold_ms == 1000.0

    def test_custom_slow_threshold(self):
        """Can set custom slow threshold."""
        mw = RequestLoggingMiddleware(slow_threshold_ms=250.0)
        assert mw.slow_threshold_ms == 250.0

    def test_start_request_generates_request_id(self, middleware):
        """start_request generates request ID when not provided."""
        ctx = middleware.start_request("GET", "/api/test", "127.0.0.1")
        assert ctx["request_id"].startswith("req-")
        assert len(ctx["request_id"]) == 16  # "req-" + 12 hex chars

    def test_start_request_uses_provided_request_id(self, middleware):
        """start_request uses provided request ID."""
        ctx = middleware.start_request("GET", "/api/test", "127.0.0.1", request_id="custom-id")
        assert ctx["request_id"] == "custom-id"

    def test_start_request_sets_log_context(self, middleware):
        """start_request sets log context."""
        middleware.start_request("POST", "/api/debates", "192.168.1.1")
        ctx = get_log_context()
        assert ctx["method"] == "POST"
        assert ctx["path"] == "/api/debates"
        assert ctx["client_ip"] == "192.168.1.1"

    def test_start_request_returns_context(self, middleware):
        """start_request returns context dict."""
        ctx = middleware.start_request("GET", "/health", "127.0.0.1")
        assert "request_id" in ctx
        assert "method" in ctx
        assert "path" in ctx
        assert "client_ip" in ctx
        assert "start_time" in ctx

    def test_start_request_redacts_headers(self, middleware):
        """start_request redacts sensitive headers."""
        headers = {"Authorization": "Bearer secret", "Content-Type": "application/json"}
        with patch.object(middleware.logger, "info") as mock_info:
            middleware.start_request("GET", "/", "127.0.0.1", headers=headers)
            call_kwargs = mock_info.call_args[1]
            assert call_kwargs["extra"]["headers"]["Authorization"] == "[REDACTED]"
            assert call_kwargs["extra"]["headers"]["Content-Type"] == "application/json"

    def test_end_request_clears_context(self, middleware):
        """end_request clears log context."""
        ctx = middleware.start_request("GET", "/", "127.0.0.1")
        middleware.end_request(ctx, 200)
        assert get_log_context() == {}

    def test_end_request_5xx_logs_error_level(self, middleware):
        """end_request logs at ERROR level for 5xx."""
        ctx = middleware.start_request("GET", "/", "127.0.0.1")
        with patch.object(middleware.logger, "log") as mock_log:
            middleware.end_request(ctx, 500)
            mock_log.assert_called()
            call_args = mock_log.call_args[0]
            assert call_args[0] == logging.ERROR

    def test_end_request_4xx_logs_warning_level(self, middleware):
        """end_request logs at WARNING level for 4xx."""
        ctx = middleware.start_request("GET", "/", "127.0.0.1")
        with patch.object(middleware.logger, "log") as mock_log:
            middleware.end_request(ctx, 404)
            call_args = mock_log.call_args[0]
            assert call_args[0] == logging.WARNING

    def test_end_request_slow_request_logs_warning(self, middleware):
        """end_request logs WARNING for slow requests."""
        ctx = middleware.start_request("GET", "/", "127.0.0.1")
        # Make request appear slow by backdating start time
        ctx["start_time"] = time.time() - 1.0  # 1 second ago

        with patch.object(middleware.logger, "log") as mock_log:
            middleware.end_request(ctx, 200)
            call_args = mock_log.call_args[0]
            assert call_args[0] == logging.WARNING
            call_kwargs = mock_log.call_args[1]
            assert call_kwargs["extra"]["slow_request"] is True

    def test_end_request_includes_response_size(self, middleware):
        """end_request includes response size when provided."""
        ctx = middleware.start_request("GET", "/", "127.0.0.1")
        with patch.object(middleware.logger, "log") as mock_log:
            middleware.end_request(ctx, 200, response_size=1024)
            call_kwargs = mock_log.call_args[1]
            assert call_kwargs["extra"]["response_size"] == 1024

    def test_log_error_logs_exception_details(self, middleware):
        """log_error logs exception details."""
        ctx = middleware.start_request("GET", "/", "127.0.0.1")
        error = ValueError("test error")
        with patch.object(middleware.logger, "error") as mock_error:
            middleware.log_error(ctx, error)
            call_kwargs = mock_error.call_args[1]
            assert call_kwargs["extra"]["error_type"] == "ValueError"
            assert "test error" in call_kwargs["extra"]["error_message"]

    def test_log_error_includes_traceback(self, middleware):
        """log_error includes traceback by default."""
        ctx = middleware.start_request("GET", "/", "127.0.0.1")
        error = ValueError("test error")
        with patch.object(middleware.logger, "error") as mock_error:
            middleware.log_error(ctx, error, include_traceback=True)
            call_kwargs = mock_error.call_args[1]
            assert "traceback" in call_kwargs["extra"]

    def test_log_error_omits_traceback_when_disabled(self, middleware):
        """log_error omits traceback when disabled."""
        ctx = middleware.start_request("GET", "/", "127.0.0.1")
        error = ValueError("test error")
        with patch.object(middleware.logger, "error") as mock_error:
            middleware.log_error(ctx, error, include_traceback=False)
            call_kwargs = mock_error.call_args[1]
            assert "traceback" not in call_kwargs["extra"]


# =============================================================================
# REDACT_FIELDS Tests
# =============================================================================


class TestRedactFields:
    """Test REDACT_FIELDS constant."""

    def test_all_expected_fields_present(self):
        """All expected sensitive fields are present."""
        expected = {
            "password",
            "secret",
            "token",
            "api_key",
            "apikey",
            "authorization",
            "cookie",
            "credit_card",
            "ssn",
            "private_key",
        }
        assert expected == REDACT_FIELDS

    def test_is_frozenset(self):
        """REDACT_FIELDS is immutable frozenset."""
        assert isinstance(REDACT_FIELDS, frozenset)
