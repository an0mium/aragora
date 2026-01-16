"""
Tests for aragora.server.middleware.exception_handler - Exception handling middleware.

Tests cover:
- EXCEPTION_STATUS_MAP constants
- map_exception_to_status() function
- generate_trace_id() function
- ErrorResponse dataclass
- build_error_response() function
- ExceptionHandler context manager (sync)
- async_exception_handler context manager (async)
- handle_exceptions() sync decorator
- async_handle_exceptions() async decorator
- is_client_error(), is_server_error(), is_retryable(), is_authentication_error() utilities
"""

from __future__ import annotations

import pytest


# ===========================================================================
# Test Exception Status Map
# ===========================================================================


class TestExceptionStatusMap:
    """Tests for EXCEPTION_STATUS_MAP constants."""

    def test_builtin_exceptions_mapped(self):
        """Built-in Python exceptions should be mapped."""
        from aragora.server.middleware.exception_handler import EXCEPTION_STATUS_MAP

        assert EXCEPTION_STATUS_MAP["FileNotFoundError"] == 404
        assert EXCEPTION_STATUS_MAP["KeyError"] == 404
        assert EXCEPTION_STATUS_MAP["ValueError"] == 400
        assert EXCEPTION_STATUS_MAP["TypeError"] == 400
        assert EXCEPTION_STATUS_MAP["PermissionError"] == 403
        assert EXCEPTION_STATUS_MAP["TimeoutError"] == 504

    def test_validation_errors_return_400(self):
        """Validation errors should map to 400 Bad Request."""
        from aragora.server.middleware.exception_handler import EXCEPTION_STATUS_MAP

        validation_errors = [
            "ValidationError",
            "InputValidationError",
            "SchemaValidationError",
            "VoteValidationError",
            "JSONParseError",
        ]

        for error_name in validation_errors:
            assert EXCEPTION_STATUS_MAP.get(error_name) == 400, f"{error_name} should map to 400"

    def test_not_found_errors_return_404(self):
        """Not found errors should map to 404 Not Found."""
        from aragora.server.middleware.exception_handler import EXCEPTION_STATUS_MAP

        not_found_errors = [
            "DebateNotFoundError",
            "AgentNotFoundError",
            "RecordNotFoundError",
            "ModeNotFoundError",
        ]

        for error_name in not_found_errors:
            assert EXCEPTION_STATUS_MAP.get(error_name) == 404, f"{error_name} should map to 404"

    def test_auth_errors_return_401_or_403(self):
        """Auth errors should map to 401 or 403."""
        from aragora.server.middleware.exception_handler import EXCEPTION_STATUS_MAP

        assert EXCEPTION_STATUS_MAP["AuthenticationError"] == 401
        assert EXCEPTION_STATUS_MAP["AuthError"] == 401
        assert EXCEPTION_STATUS_MAP["TokenExpiredError"] == 401
        assert EXCEPTION_STATUS_MAP["AuthorizationError"] == 403

    def test_rate_limit_returns_429(self):
        """Rate limit errors should map to 429 Too Many Requests."""
        from aragora.server.middleware.exception_handler import EXCEPTION_STATUS_MAP

        assert EXCEPTION_STATUS_MAP["RateLimitExceededError"] == 429
        assert EXCEPTION_STATUS_MAP["AgentRateLimitError"] == 429

    def test_timeout_errors_return_504(self):
        """Timeout errors should map to 504 Gateway Timeout."""
        from aragora.server.middleware.exception_handler import EXCEPTION_STATUS_MAP

        timeout_errors = [
            "TimeoutError",
            "asyncio.TimeoutError",
            "AgentTimeoutError",
            "ConsensusTimeoutError",
            "VerificationTimeoutError",
        ]

        for error_name in timeout_errors:
            assert EXCEPTION_STATUS_MAP.get(error_name) == 504, f"{error_name} should map to 504"


# ===========================================================================
# Test map_exception_to_status
# ===========================================================================


class TestMapExceptionToStatus:
    """Tests for map_exception_to_status() function."""

    def test_maps_builtin_exception(self):
        """Should map built-in exceptions correctly."""
        from aragora.server.middleware.exception_handler import map_exception_to_status

        assert map_exception_to_status(ValueError("test")) == 400
        assert map_exception_to_status(KeyError("test")) == 404
        assert map_exception_to_status(FileNotFoundError("test")) == 404

    def test_returns_default_for_unknown(self):
        """Should return default for unknown exception types."""
        from aragora.server.middleware.exception_handler import map_exception_to_status

        class CustomUnknownError(Exception):
            pass

        status = map_exception_to_status(CustomUnknownError("test"))
        assert status == 500

    def test_custom_default(self):
        """Should use custom default when provided."""
        from aragora.server.middleware.exception_handler import map_exception_to_status

        class CustomError(Exception):
            pass

        status = map_exception_to_status(CustomError("test"), default=418)
        assert status == 418

    def test_maps_by_base_class(self):
        """Should check base classes if direct type not found."""
        from aragora.server.middleware.exception_handler import map_exception_to_status

        # RuntimeError is mapped to 500 in EXCEPTION_STATUS_MAP
        class MyRuntimeError(RuntimeError):
            pass

        status = map_exception_to_status(MyRuntimeError("test"))
        assert status == 500


# ===========================================================================
# Test generate_trace_id
# ===========================================================================


class TestGenerateTraceId:
    """Tests for generate_trace_id() function."""

    def test_returns_string(self):
        """Should return a string trace ID."""
        from aragora.server.middleware.exception_handler import generate_trace_id

        trace_id = generate_trace_id()
        assert isinstance(trace_id, str)

    def test_returns_8_characters(self):
        """Should return 8-character trace ID."""
        from aragora.server.middleware.exception_handler import generate_trace_id

        trace_id = generate_trace_id()
        assert len(trace_id) == 8

    def test_returns_unique_values(self):
        """Should return unique trace IDs."""
        from aragora.server.middleware.exception_handler import generate_trace_id

        trace_ids = [generate_trace_id() for _ in range(100)]
        assert len(set(trace_ids)) == 100


# ===========================================================================
# Test ErrorResponse Dataclass
# ===========================================================================


class TestErrorResponse:
    """Tests for ErrorResponse dataclass."""

    def test_to_dict(self):
        """to_dict() should return error details."""
        from aragora.server.middleware.exception_handler import ErrorResponse

        error = ErrorResponse(
            message="Test error",
            status=400,
            trace_id="abc12345",
            error_type="ValueError",
            context="test operation",
        )

        d = error.to_dict()

        assert d["error"] == "Test error"
        assert d["status"] == 400
        assert d["trace_id"] == "abc12345"
        assert d["error_type"] == "ValueError"
        assert d["context"] == "test operation"

    def test_to_handler_result(self):
        """to_handler_result() should return (body, status, headers) tuple."""
        from aragora.server.middleware.exception_handler import ErrorResponse

        error = ErrorResponse(
            message="Test error",
            status=404,
            trace_id="abc12345",
            error_type="NotFoundError",
            context="lookup",
        )

        body, status, headers = error.to_handler_result()

        assert status == 404
        assert body["error"] == "Test error"
        assert headers["X-Trace-Id"] == "abc12345"

    def test_timestamp_auto_generated(self):
        """Should auto-generate timestamp."""
        from aragora.server.middleware.exception_handler import ErrorResponse

        error = ErrorResponse(
            message="Test",
            status=500,
            trace_id="test",
            error_type="Error",
            context="test",
        )

        assert error.timestamp > 0


# ===========================================================================
# Test build_error_response
# ===========================================================================


class TestBuildErrorResponse:
    """Tests for build_error_response() function."""

    def test_builds_from_exception(self):
        """Should build error response from exception."""
        from aragora.server.middleware.exception_handler import build_error_response

        exc = ValueError("Invalid input")
        error = build_error_response(exc, "validation")

        assert error.status == 400
        assert error.error_type == "ValueError"
        assert error.context == "validation"
        assert len(error.trace_id) == 8

    def test_uses_provided_trace_id(self):
        """Should use provided trace ID."""
        from aragora.server.middleware.exception_handler import build_error_response

        exc = ValueError("test")
        error = build_error_response(exc, "test", trace_id="custom12")

        assert error.trace_id == "custom12"

    def test_uses_default_status_for_unknown(self):
        """Should use default status for unknown exceptions."""
        from aragora.server.middleware.exception_handler import build_error_response

        class UnknownError(Exception):
            pass

        error = build_error_response(
            UnknownError("test"),
            "test",
            default_status=503,
        )

        assert error.status == 503


# ===========================================================================
# Test ExceptionHandler Context Manager (Sync)
# ===========================================================================


class TestExceptionHandlerContextManager:
    """Tests for ExceptionHandler context manager."""

    def test_successful_execution(self):
        """Should track successful execution."""
        from aragora.server.middleware.exception_handler import ExceptionHandler

        with ExceptionHandler("test operation") as ctx:
            result = "success"
            ctx.success(result)

        assert ctx.error is None
        assert ctx.result == "success"
        assert ctx.status_code == 200

    def test_captures_exception(self):
        """Should capture exception and build error response."""
        from aragora.server.middleware.exception_handler import ExceptionHandler

        with ExceptionHandler("test operation") as ctx:
            raise ValueError("test error")

        assert ctx.error is not None
        assert ctx.error.status == 400
        assert ctx.error.error_type == "ValueError"
        assert ctx.exception is not None

    def test_suppresses_exception(self):
        """Should suppress exception by default."""
        from aragora.server.middleware.exception_handler import ExceptionHandler

        # Should not raise
        with ExceptionHandler("test") as ctx:
            raise ValueError("suppressed")

        assert ctx.error is not None

    def test_error_response_property(self):
        """error_response should return dict or None."""
        from aragora.server.middleware.exception_handler import ExceptionHandler

        # Success case
        with ExceptionHandler("test") as ctx:
            ctx.success("ok")

        assert ctx.error_response is None

        # Error case
        with ExceptionHandler("test") as ctx:
            raise ValueError("error")

        assert ctx.error_response is not None
        assert "error" in ctx.error_response

    def test_uses_custom_default_status(self):
        """Should use custom default status."""
        from aragora.server.middleware.exception_handler import ExceptionHandler

        class CustomError(Exception):
            pass

        with ExceptionHandler("test", default_status=418) as ctx:
            raise CustomError("custom")

        assert ctx.error.status == 418


# ===========================================================================
# Test async_exception_handler Context Manager (Async)
# ===========================================================================


class TestAsyncExceptionHandler:
    """Tests for async_exception_handler context manager."""

    @pytest.mark.asyncio
    async def test_successful_execution(self):
        """Should track successful async execution."""
        from aragora.server.middleware.exception_handler import async_exception_handler

        async with async_exception_handler("async operation") as ctx:
            result = "async success"
            ctx.success(result)

        assert ctx.error is None
        assert ctx.result == "async success"

    @pytest.mark.asyncio
    async def test_captures_exception(self):
        """Should capture exception in async context."""
        from aragora.server.middleware.exception_handler import async_exception_handler

        async with async_exception_handler("async operation") as ctx:
            raise KeyError("not found")

        assert ctx.error is not None
        assert ctx.error.status == 404
        assert ctx.error.error_type == "KeyError"

    @pytest.mark.asyncio
    async def test_suppresses_exception(self):
        """Should suppress exception by default in async context."""
        from aragora.server.middleware.exception_handler import async_exception_handler

        # Should not raise
        async with async_exception_handler("test") as ctx:
            raise ValueError("suppressed")

        assert ctx.error is not None


# ===========================================================================
# Test handle_exceptions Decorator (Sync)
# ===========================================================================


class TestHandleExceptionsDecorator:
    """Tests for handle_exceptions() sync decorator."""

    def test_successful_execution(self):
        """Should return function result on success."""
        from aragora.server.middleware.exception_handler import handle_exceptions

        @handle_exceptions("test")
        def my_handler():
            return {"data": "success"}, 200

        result = my_handler()
        assert result == ({"data": "success"}, 200)

    def test_returns_error_on_exception(self):
        """Should return error response on exception."""
        from aragora.server.middleware.exception_handler import handle_exceptions

        @handle_exceptions("test operation")
        def failing_handler():
            raise ValueError("invalid")

        body, status, headers = failing_handler()

        assert status == 400
        assert "error" in body
        assert "X-Trace-Id" in headers

    def test_reraise_option(self):
        """Should re-raise exception when reraise=True."""
        from aragora.server.middleware.exception_handler import handle_exceptions

        @handle_exceptions("test", reraise=True)
        def failing_handler():
            raise ValueError("to be raised")

        with pytest.raises(ValueError, match="to be raised"):
            failing_handler()

    def test_preserves_function_metadata(self):
        """Should preserve function name and docstring."""
        from aragora.server.middleware.exception_handler import handle_exceptions

        @handle_exceptions("test")
        def my_function():
            """My docstring."""
            pass

        assert my_function.__name__ == "my_function"
        assert my_function.__doc__ == "My docstring."


# ===========================================================================
# Test async_handle_exceptions Decorator (Async)
# ===========================================================================


class TestAsyncHandleExceptionsDecorator:
    """Tests for async_handle_exceptions() async decorator."""

    @pytest.mark.asyncio
    async def test_successful_execution(self):
        """Should return function result on success."""
        from aragora.server.middleware.exception_handler import async_handle_exceptions

        @async_handle_exceptions("test")
        async def my_handler():
            return {"data": "async success"}, 200

        result = await my_handler()
        assert result == ({"data": "async success"}, 200)

    @pytest.mark.asyncio
    async def test_returns_error_on_exception(self):
        """Should return error response on async exception."""
        from aragora.server.middleware.exception_handler import async_handle_exceptions

        @async_handle_exceptions("async test")
        async def failing_handler():
            raise FileNotFoundError("missing")

        body, status, headers = await failing_handler()

        assert status == 404
        assert "error" in body
        assert "X-Trace-Id" in headers

    @pytest.mark.asyncio
    async def test_reraise_option(self):
        """Should re-raise exception when reraise=True."""
        from aragora.server.middleware.exception_handler import async_handle_exceptions

        @async_handle_exceptions("test", reraise=True)
        async def failing_handler():
            raise KeyError("to be raised")

        with pytest.raises(KeyError, match="to be raised"):
            await failing_handler()

    @pytest.mark.asyncio
    async def test_preserves_function_metadata(self):
        """Should preserve async function name and docstring."""
        from aragora.server.middleware.exception_handler import async_handle_exceptions

        @async_handle_exceptions("test")
        async def my_async_function():
            """My async docstring."""
            pass

        assert my_async_function.__name__ == "my_async_function"
        assert my_async_function.__doc__ == "My async docstring."


# ===========================================================================
# Test Exception Type Utilities
# ===========================================================================


class TestIsClientError:
    """Tests for is_client_error() function."""

    def test_returns_true_for_4xx(self):
        """Should return True for 4xx status codes."""
        from aragora.server.middleware.exception_handler import is_client_error

        assert is_client_error(ValueError("bad")) is True  # 400
        assert is_client_error(KeyError("missing")) is True  # 404
        assert is_client_error(PermissionError("denied")) is True  # 403

    def test_returns_false_for_5xx(self):
        """Should return False for 5xx status codes."""
        from aragora.server.middleware.exception_handler import is_client_error

        assert is_client_error(RuntimeError("server")) is False  # 500


class TestIsServerError:
    """Tests for is_server_error() function."""

    def test_returns_true_for_5xx(self):
        """Should return True for 5xx status codes."""
        from aragora.server.middleware.exception_handler import is_server_error

        assert is_server_error(RuntimeError("server")) is True  # 500
        assert is_server_error(OSError("os")) is True  # 500

    def test_returns_false_for_4xx(self):
        """Should return False for 4xx status codes."""
        from aragora.server.middleware.exception_handler import is_server_error

        assert is_server_error(ValueError("bad")) is False  # 400


class TestIsRetryable:
    """Tests for is_retryable() function."""

    def test_returns_true_for_retryable_errors(self):
        """Should return True for 429, 502, 503, 504."""
        from aragora.server.middleware.exception_handler import is_retryable

        assert is_retryable(ConnectionError("conn")) is True  # 502
        assert is_retryable(TimeoutError("timeout")) is True  # 504

    def test_returns_false_for_non_retryable(self):
        """Should return False for non-retryable errors."""
        from aragora.server.middleware.exception_handler import is_retryable

        assert is_retryable(ValueError("bad")) is False  # 400
        assert is_retryable(KeyError("missing")) is False  # 404


class TestIsAuthenticationError:
    """Tests for is_authentication_error() function."""

    def test_returns_true_for_401_403(self):
        """Should return True for 401 and 403 status codes."""
        from aragora.server.middleware.exception_handler import is_authentication_error

        assert is_authentication_error(PermissionError("denied")) is True  # 403

    def test_returns_false_for_other_errors(self):
        """Should return False for non-auth errors."""
        from aragora.server.middleware.exception_handler import is_authentication_error

        assert is_authentication_error(ValueError("bad")) is False
        assert is_authentication_error(RuntimeError("server")) is False
