"""
Tests for centralized error handling utilities.

Tests the handle_errors decorator, trace ID generation,
and exception-to-status mapping in handlers/base.py.
"""

import logging
import pytest
from unittest.mock import patch, MagicMock

from aragora.server.handlers.base import (
    handle_errors,
    generate_trace_id,
    _map_exception_to_status,
    error_response,
    json_response,
    HandlerResult,
)


# ============================================================================
# Test generate_trace_id
# ============================================================================

class TestGenerateTraceId:
    """Tests for trace ID generation."""

    def test_returns_string(self):
        """Trace ID should be a string."""
        trace_id = generate_trace_id()
        assert isinstance(trace_id, str)

    def test_correct_length(self):
        """Trace ID should be 8 characters."""
        trace_id = generate_trace_id()
        assert len(trace_id) == 8

    def test_unique_ids(self):
        """Multiple calls should return unique IDs."""
        ids = [generate_trace_id() for _ in range(100)]
        assert len(set(ids)) == 100

    def test_valid_hex_format(self):
        """Trace ID should be valid hex characters."""
        trace_id = generate_trace_id()
        # UUID prefix is hex + hyphens, but we slice first 8 which is all hex
        assert all(c in "0123456789abcdef-" for c in trace_id)


# ============================================================================
# Test _map_exception_to_status
# ============================================================================

class TestMapExceptionToStatus:
    """Tests for exception to HTTP status mapping."""

    def test_file_not_found_returns_404(self):
        """FileNotFoundError should map to 404."""
        status = _map_exception_to_status(FileNotFoundError("missing"))
        assert status == 404

    def test_key_error_returns_404(self):
        """KeyError should map to 404."""
        status = _map_exception_to_status(KeyError("key"))
        assert status == 404

    def test_value_error_returns_400(self):
        """ValueError should map to 400."""
        status = _map_exception_to_status(ValueError("bad"))
        assert status == 400

    def test_type_error_returns_400(self):
        """TypeError should map to 400."""
        status = _map_exception_to_status(TypeError("type"))
        assert status == 400

    def test_permission_error_returns_403(self):
        """PermissionError should map to 403."""
        status = _map_exception_to_status(PermissionError("denied"))
        assert status == 403

    def test_timeout_error_returns_504(self):
        """TimeoutError should map to 504."""
        status = _map_exception_to_status(TimeoutError("slow"))
        assert status == 504

    def test_connection_error_returns_502(self):
        """ConnectionError should map to 502."""
        status = _map_exception_to_status(ConnectionError("network"))
        assert status == 502

    def test_os_error_returns_500(self):
        """OSError should map to 500."""
        status = _map_exception_to_status(OSError("io"))
        assert status == 500

    def test_unknown_exception_returns_default(self):
        """Unknown exceptions should return default status."""
        status = _map_exception_to_status(RuntimeError("unknown"))
        assert status == 500

    def test_custom_default_status(self):
        """Custom default should be used for unknown exceptions."""
        status = _map_exception_to_status(RuntimeError("unknown"), default=503)
        assert status == 503


# ============================================================================
# Test error_response with headers
# ============================================================================

class TestErrorResponseHeaders:
    """Tests for error_response with header support."""

    def test_error_response_basic(self):
        """Basic error response without headers."""
        result = error_response("Test error", 400)
        assert result.status_code == 400
        assert result.headers == {}
        assert b"Test error" in result.body

    def test_error_response_with_headers(self):
        """Error response with custom headers."""
        result = error_response("Test error", 500, headers={"X-Trace-Id": "abc123"})
        assert result.status_code == 500
        assert result.headers == {"X-Trace-Id": "abc123"}

    def test_error_response_preserves_content_type(self):
        """Error response should have JSON content type."""
        result = error_response("Test error", 400)
        assert result.content_type == "application/json"


# ============================================================================
# Test json_response with headers
# ============================================================================

class TestJsonResponseHeaders:
    """Tests for json_response with header support."""

    def test_json_response_basic(self):
        """Basic JSON response without headers."""
        result = json_response({"key": "value"}, 200)
        assert result.status_code == 200
        assert result.headers == {}

    def test_json_response_with_headers(self):
        """JSON response with custom headers."""
        result = json_response(
            {"key": "value"},
            200,
            headers={"X-Custom": "header"}
        )
        assert result.headers == {"X-Custom": "header"}


# ============================================================================
# Test handle_errors decorator
# ============================================================================

class TestHandleErrorsDecorator:
    """Tests for the handle_errors decorator."""

    def test_successful_call_passes_through(self):
        """Successful function calls should return normally."""
        @handle_errors("test operation")
        def success_func():
            return json_response({"success": True})

        result = success_func()
        assert result.status_code == 200
        assert b"success" in result.body

    def test_exception_returns_error_response(self):
        """Exceptions should be caught and return error response."""
        @handle_errors("test operation")
        def failing_func():
            raise ValueError("test error")

        result = failing_func()
        assert result.status_code == 400
        assert b"error" in result.body

    def test_exception_includes_trace_id(self):
        """Error responses should include X-Trace-Id header."""
        @handle_errors("test operation")
        def failing_func():
            raise ValueError("test error")

        result = failing_func()
        assert "X-Trace-Id" in result.headers
        assert len(result.headers["X-Trace-Id"]) == 8

    def test_file_not_found_returns_404(self):
        """FileNotFoundError should result in 404 status."""
        @handle_errors("resource lookup")
        def not_found_func():
            raise FileNotFoundError("missing file")

        result = not_found_func()
        assert result.status_code == 404

    def test_permission_error_returns_403(self):
        """PermissionError should result in 403 status."""
        @handle_errors("access check")
        def permission_func():
            raise PermissionError("access denied")

        result = permission_func()
        assert result.status_code == 403

    def test_timeout_error_returns_504(self):
        """TimeoutError should result in 504 status."""
        @handle_errors("remote call")
        def timeout_func():
            raise TimeoutError("operation timed out")

        result = timeout_func()
        assert result.status_code == 504

    def test_unknown_error_returns_500(self):
        """Unknown exceptions should result in 500 status."""
        @handle_errors("general operation")
        def unknown_func():
            raise RuntimeError("unexpected error")

        result = unknown_func()
        assert result.status_code == 500

    def test_custom_default_status(self):
        """Custom default status should be used."""
        @handle_errors("service unavailable", default_status=503)
        def unavailable_func():
            raise RuntimeError("service down")

        result = unavailable_func()
        assert result.status_code == 503

    def test_logs_exception_details(self, caplog):
        """Decorator should log exception details."""
        @handle_errors("test context")
        def failing_func():
            raise ValueError("detailed error message")

        with caplog.at_level(logging.ERROR):
            failing_func()

        assert "test context" in caplog.text
        assert "ValueError" in caplog.text

    def test_logs_trace_id(self, caplog):
        """Decorator should log trace ID."""
        @handle_errors("traceable operation")
        def failing_func():
            raise ValueError("error")

        with caplog.at_level(logging.ERROR):
            result = failing_func()

        trace_id = result.headers["X-Trace-Id"]
        assert trace_id in caplog.text

    def test_preserves_function_args(self):
        """Decorator should pass through function arguments."""
        @handle_errors("parameterized operation")
        def func_with_args(a, b, c=None):
            return json_response({"a": a, "b": b, "c": c})

        result = func_with_args(1, 2, c=3)
        assert result.status_code == 200

    def test_preserves_function_name(self):
        """Decorator should preserve function name."""
        @handle_errors("named operation")
        def my_special_function():
            pass

        assert my_special_function.__name__ == "my_special_function"

    def test_works_with_class_methods(self):
        """Decorator should work with class instance methods."""
        class TestHandler:
            @handle_errors("method operation")
            def handle_request(self, query_params):
                if query_params.get("fail"):
                    raise ValueError("method failed")
                return json_response({"status": "ok"})

        handler = TestHandler()

        # Success case
        result = handler.handle_request({})
        assert result.status_code == 200

        # Failure case
        result = handler.handle_request({"fail": True})
        assert result.status_code == 400
        assert "X-Trace-Id" in result.headers


# ============================================================================
# Test sanitization integration
# ============================================================================

class TestSanitizationIntegration:
    """Tests for error message sanitization in decorator."""

    def test_error_message_sanitized(self):
        """Error messages should be sanitized (no internal paths)."""
        @handle_errors("file operation")
        def file_func():
            raise FileNotFoundError("/secret/internal/path/file.txt")

        result = file_func()
        # Should get generic message, not the path
        assert b"/secret" not in result.body
        assert b"Resource not found" in result.body or b"error" in result.body

    def test_value_error_sanitized(self):
        """ValueError should get sanitized message."""
        @handle_errors("validation")
        def validate_func():
            raise ValueError("Invalid API key: sk-12345secret")

        result = validate_func()
        # Should not expose the API key
        assert b"sk-12345" not in result.body


# ============================================================================
# Test concurrent trace IDs
# ============================================================================

class TestConcurrentTraceIds:
    """Tests for trace ID uniqueness under concurrent access."""

    def test_concurrent_requests_unique_ids(self):
        """Concurrent failing requests should have unique trace IDs."""
        import threading

        trace_ids = []
        lock = threading.Lock()

        @handle_errors("concurrent operation")
        def concurrent_func():
            raise ValueError("concurrent error")

        def call_and_record():
            result = concurrent_func()
            with lock:
                trace_ids.append(result.headers["X-Trace-Id"])

        threads = [threading.Thread(target=call_and_record) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All trace IDs should be unique
        assert len(trace_ids) == 20
        assert len(set(trace_ids)) == 20


# ============================================================================
# Test edge cases
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases in error handling."""

    def test_exception_with_no_message(self):
        """Exceptions without messages should be handled."""
        @handle_errors("empty error")
        def empty_error_func():
            raise ValueError()

        result = empty_error_func()
        assert result.status_code == 400
        assert "X-Trace-Id" in result.headers

    def test_exception_with_non_string_message(self):
        """Exceptions with non-string messages should be handled."""
        @handle_errors("non-string error")
        def non_string_func():
            raise ValueError({"complex": "error"})

        result = non_string_func()
        assert result.status_code == 400

    def test_nested_exception(self):
        """Nested exceptions should be handled."""
        @handle_errors("nested error")
        def nested_func():
            try:
                raise KeyError("original")
            except KeyError:
                raise ValueError("wrapper") from None

        result = nested_func()
        assert result.status_code == 400

    def test_keyboard_interrupt_propagates(self):
        """KeyboardInterrupt should propagate, not be caught."""
        @handle_errors("interrupt test")
        def interrupt_func():
            raise KeyboardInterrupt()

        with pytest.raises(KeyboardInterrupt):
            interrupt_func()

    def test_system_exit_propagates(self):
        """SystemExit should propagate, not be caught."""
        @handle_errors("exit test")
        def exit_func():
            raise SystemExit(1)

        with pytest.raises(SystemExit):
            exit_func()

