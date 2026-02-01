"""
Tests for aragora.server.middleware.error_handler - Standardized Error Handler Middleware.

Tests cover:
- APIError exception class
- raise_api_error helper function
- ErrorResponse dataclass
- ErrorHandlerMiddleware configuration and usage
- Exception to error code mapping
- Error response formatting
- Helper functions (validation_error, not_found_error, etc.)
- Request ID generation
- Traceback inclusion in development mode
"""

from __future__ import annotations

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch


# =============================================================================
# Test APIError Exception
# =============================================================================


class TestAPIError:
    """Tests for APIError exception class."""

    def test_create_api_error_basic(self):
        """Should create an APIError with basic fields."""
        from aragora.server.middleware.error_handler import APIError
        from aragora.server.error_codes import ErrorCode

        error = APIError(
            code=ErrorCode.VALIDATION_ERROR,
            message="Invalid input",
        )

        assert error.code == ErrorCode.VALIDATION_ERROR
        assert error.message == "Invalid input"
        assert error.status == 400  # Auto-derived from code

    def test_create_api_error_with_status(self):
        """Should allow custom status code."""
        from aragora.server.middleware.error_handler import APIError
        from aragora.server.error_codes import ErrorCode

        error = APIError(
            code=ErrorCode.VALIDATION_ERROR,
            message="Custom status",
            status=422,
        )

        assert error.status == 422

    def test_create_api_error_with_details(self):
        """Should include details dictionary."""
        from aragora.server.middleware.error_handler import APIError
        from aragora.server.error_codes import ErrorCode

        error = APIError(
            code=ErrorCode.MISSING_FIELD,
            message="Missing required field",
            details={"field": "email", "expected": "string"},
        )

        assert error.details == {"field": "email", "expected": "string"}

    def test_create_api_error_with_headers(self):
        """Should include custom headers."""
        from aragora.server.middleware.error_handler import APIError
        from aragora.server.error_codes import ErrorCode

        error = APIError(
            code=ErrorCode.RATE_LIMITED,
            message="Too many requests",
            headers={"Retry-After": "60"},
        )

        assert error.headers == {"Retry-After": "60"}

    def test_api_error_is_exception(self):
        """APIError should be a proper exception."""
        from aragora.server.middleware.error_handler import APIError
        from aragora.server.error_codes import ErrorCode

        error = APIError(
            code=ErrorCode.NOT_FOUND,
            message="Resource not found",
        )

        assert isinstance(error, Exception)
        assert str(error) == "Resource not found"

    def test_api_error_to_dict_basic(self):
        """to_dict should create standardized response structure."""
        from aragora.server.middleware.error_handler import APIError
        from aragora.server.error_codes import ErrorCode

        error = APIError(
            code=ErrorCode.VALIDATION_ERROR,
            message="Invalid email",
        )

        result = error.to_dict()

        assert "error" in result
        assert result["error"]["code"] == ErrorCode.VALIDATION_ERROR
        assert result["error"]["message"] == "Invalid email"
        assert "timestamp" in result["error"]

    def test_api_error_to_dict_with_request_id(self):
        """to_dict should include request_id when provided."""
        from aragora.server.middleware.error_handler import APIError
        from aragora.server.error_codes import ErrorCode

        error = APIError(
            code=ErrorCode.NOT_FOUND,
            message="Not found",
        )

        result = error.to_dict(request_id="req_abc123")

        assert result["error"]["request_id"] == "req_abc123"

    def test_api_error_to_dict_with_path(self):
        """to_dict should include path when provided."""
        from aragora.server.middleware.error_handler import APIError
        from aragora.server.error_codes import ErrorCode

        error = APIError(
            code=ErrorCode.NOT_FOUND,
            message="Not found",
        )

        result = error.to_dict(path="/api/debates/123")

        assert result["error"]["path"] == "/api/debates/123"

    def test_api_error_to_dict_with_details(self):
        """to_dict should include details when present."""
        from aragora.server.middleware.error_handler import APIError
        from aragora.server.error_codes import ErrorCode

        error = APIError(
            code=ErrorCode.VALIDATION_ERROR,
            message="Invalid field",
            details={"field": "name", "reason": "too short"},
        )

        result = error.to_dict()

        assert result["error"]["details"] == {"field": "name", "reason": "too short"}

    def test_api_error_timestamp_format(self):
        """Timestamp should be ISO 8601 format with Z suffix."""
        from aragora.server.middleware.error_handler import APIError
        from aragora.server.error_codes import ErrorCode

        error = APIError(
            code=ErrorCode.INTERNAL_ERROR,
            message="Error",
        )

        result = error.to_dict()
        timestamp = result["error"]["timestamp"]

        # Should end with Z
        assert timestamp.endswith("Z")
        # Should be parseable
        # Remove trailing Z for parsing
        parsed = datetime.fromisoformat(timestamp.rstrip("Z"))
        assert parsed is not None


# =============================================================================
# Test raise_api_error Helper
# =============================================================================


class TestRaiseApiError:
    """Tests for raise_api_error helper function."""

    def test_raises_api_error(self):
        """Should raise APIError with given parameters."""
        from aragora.server.middleware.error_handler import APIError, raise_api_error
        from aragora.server.error_codes import ErrorCode

        with pytest.raises(APIError) as exc_info:
            raise_api_error(
                ErrorCode.FORBIDDEN,
                "Access denied",
            )

        assert exc_info.value.code == ErrorCode.FORBIDDEN
        assert exc_info.value.message == "Access denied"

    def test_raises_api_error_with_details(self):
        """Should include details in raised error."""
        from aragora.server.middleware.error_handler import APIError, raise_api_error
        from aragora.server.error_codes import ErrorCode

        with pytest.raises(APIError) as exc_info:
            raise_api_error(
                ErrorCode.VALIDATION_ERROR,
                "Invalid input",
                details={"field": "email"},
            )

        assert exc_info.value.details == {"field": "email"}

    def test_raises_api_error_with_custom_status(self):
        """Should allow custom status code."""
        from aragora.server.middleware.error_handler import APIError, raise_api_error
        from aragora.server.error_codes import ErrorCode

        with pytest.raises(APIError) as exc_info:
            raise_api_error(
                ErrorCode.VALIDATION_ERROR,
                "Custom status",
                status=422,
            )

        assert exc_info.value.status == 422


# =============================================================================
# Test ErrorResponse
# =============================================================================


class TestErrorResponse:
    """Tests for ErrorResponse dataclass."""

    def test_create_error_response(self):
        """Should create ErrorResponse with required fields."""
        from aragora.server.middleware.error_handler import ErrorResponse

        response = ErrorResponse(
            status=400,
            body={"error": {"code": "VALIDATION_ERROR", "message": "Invalid"}},
        )

        assert response.status == 400
        assert response.body["error"]["code"] == "VALIDATION_ERROR"
        assert response.headers == {}

    def test_create_error_response_with_headers(self):
        """Should include custom headers."""
        from aragora.server.middleware.error_handler import ErrorResponse

        response = ErrorResponse(
            status=429,
            body={"error": {"code": "RATE_LIMITED", "message": "Too many"}},
            headers={"Retry-After": "30"},
        )

        assert response.headers == {"Retry-After": "30"}


# =============================================================================
# Test ErrorHandlerMiddleware
# =============================================================================


class TestErrorHandlerMiddleware:
    """Tests for ErrorHandlerMiddleware class."""

    def test_init_default_config(self):
        """Should initialize with default configuration."""
        from aragora.server.middleware.error_handler import ErrorHandlerMiddleware

        app = MagicMock()
        middleware = ErrorHandlerMiddleware(app)

        assert middleware.app == app
        assert middleware.include_traceback is False
        assert middleware.log_errors is True
        assert middleware.exclude_paths == ["/healthz", "/readyz"]

    def test_init_custom_config(self):
        """Should accept custom configuration."""
        from aragora.server.middleware.error_handler import ErrorHandlerMiddleware

        app = MagicMock()
        middleware = ErrorHandlerMiddleware(
            app,
            include_traceback=True,
            log_errors=False,
            exclude_paths=["/custom/health"],
        )

        assert middleware.include_traceback is True
        assert middleware.log_errors is False
        assert middleware.exclude_paths == ["/custom/health"]

    def test_generate_request_id_format(self):
        """Request ID should have correct format."""
        from aragora.server.middleware.error_handler import ErrorHandlerMiddleware

        app = MagicMock()
        middleware = ErrorHandlerMiddleware(app)

        request_id = middleware.generate_request_id()

        assert request_id.startswith("req_")
        assert len(request_id) == 20  # "req_" + 16 hex chars

    def test_generate_request_id_unique(self):
        """Request IDs should be unique."""
        from aragora.server.middleware.error_handler import ErrorHandlerMiddleware

        app = MagicMock()
        middleware = ErrorHandlerMiddleware(app)

        ids = [middleware.generate_request_id() for _ in range(100)]
        assert len(set(ids)) == 100


class TestErrorHandlerMiddlewareHandleException:
    """Tests for handle_exception method."""

    def test_handle_api_error(self):
        """Should handle APIError correctly."""
        from aragora.server.middleware.error_handler import (
            APIError,
            ErrorHandlerMiddleware,
        )
        from aragora.server.error_codes import ErrorCode

        app = MagicMock()
        middleware = ErrorHandlerMiddleware(app, log_errors=False)

        error = APIError(
            code=ErrorCode.NOT_FOUND,
            message="Debate not found",
            details={"debate_id": "123"},
        )

        response = middleware.handle_exception(error, "req_abc", "/api/debates/123")

        assert response.status == 404
        assert response.body["error"]["code"] == ErrorCode.NOT_FOUND
        assert response.body["error"]["message"] == "Debate not found"
        assert response.body["error"]["request_id"] == "req_abc"
        assert response.body["error"]["path"] == "/api/debates/123"

    def test_handle_api_error_with_headers(self):
        """Should include APIError headers in response."""
        from aragora.server.middleware.error_handler import (
            APIError,
            ErrorHandlerMiddleware,
        )
        from aragora.server.error_codes import ErrorCode

        app = MagicMock()
        middleware = ErrorHandlerMiddleware(app, log_errors=False)

        error = APIError(
            code=ErrorCode.RATE_LIMITED,
            message="Rate limited",
            headers={"Retry-After": "60"},
        )

        response = middleware.handle_exception(error, "req_xyz", "/api/test")

        assert response.headers == {"Retry-After": "60"}

    def test_handle_value_error(self):
        """Should map ValueError to VALIDATION_ERROR."""
        from aragora.server.middleware.error_handler import ErrorHandlerMiddleware
        from aragora.server.error_codes import ErrorCode

        app = MagicMock()
        middleware = ErrorHandlerMiddleware(app, log_errors=False)

        error = ValueError("Invalid format")

        response = middleware.handle_exception(error, "req_123", "/api/test")

        assert response.status == 400
        assert response.body["error"]["code"] == ErrorCode.VALIDATION_ERROR

    def test_handle_key_error(self):
        """Should map KeyError to MISSING_FIELD."""
        from aragora.server.middleware.error_handler import ErrorHandlerMiddleware
        from aragora.server.error_codes import ErrorCode

        app = MagicMock()
        middleware = ErrorHandlerMiddleware(app, log_errors=False)

        error = KeyError("name")

        response = middleware.handle_exception(error, "req_123", "/api/test")

        assert response.body["error"]["code"] == ErrorCode.MISSING_FIELD

    def test_handle_type_error(self):
        """Should map TypeError to INVALID_FIELD."""
        from aragora.server.middleware.error_handler import ErrorHandlerMiddleware
        from aragora.server.error_codes import ErrorCode

        app = MagicMock()
        middleware = ErrorHandlerMiddleware(app, log_errors=False)

        error = TypeError("Expected string")

        response = middleware.handle_exception(error, "req_123", "/api/test")

        assert response.body["error"]["code"] == ErrorCode.INVALID_FIELD

    def test_handle_permission_error(self):
        """Should map PermissionError to PERMISSION_DENIED."""
        from aragora.server.middleware.error_handler import ErrorHandlerMiddleware
        from aragora.server.error_codes import ErrorCode

        app = MagicMock()
        middleware = ErrorHandlerMiddleware(app, log_errors=False)

        error = PermissionError("Access denied")

        response = middleware.handle_exception(error, "req_123", "/api/test")

        assert response.status == 403
        assert response.body["error"]["code"] == ErrorCode.PERMISSION_DENIED

    def test_handle_file_not_found_error(self):
        """Should map FileNotFoundError to NOT_FOUND."""
        from aragora.server.middleware.error_handler import ErrorHandlerMiddleware
        from aragora.server.error_codes import ErrorCode

        app = MagicMock()
        middleware = ErrorHandlerMiddleware(app, log_errors=False)

        error = FileNotFoundError("File missing")

        response = middleware.handle_exception(error, "req_123", "/api/test")

        assert response.status == 404
        assert response.body["error"]["code"] == ErrorCode.NOT_FOUND

    def test_handle_timeout_error(self):
        """Should map TimeoutError to EXTERNAL_SERVICE_ERROR."""
        from aragora.server.middleware.error_handler import ErrorHandlerMiddleware
        from aragora.server.error_codes import ErrorCode

        app = MagicMock()
        middleware = ErrorHandlerMiddleware(app, log_errors=False)

        error = TimeoutError("Request timeout")

        response = middleware.handle_exception(error, "req_123", "/api/test")

        assert response.body["error"]["code"] == ErrorCode.EXTERNAL_SERVICE_ERROR

    def test_handle_connection_error(self):
        """Should map ConnectionError to EXTERNAL_SERVICE_ERROR."""
        from aragora.server.middleware.error_handler import ErrorHandlerMiddleware
        from aragora.server.error_codes import ErrorCode

        app = MagicMock()
        middleware = ErrorHandlerMiddleware(app, log_errors=False)

        error = ConnectionError("Connection refused")

        response = middleware.handle_exception(error, "req_123", "/api/test")

        assert response.body["error"]["code"] == ErrorCode.EXTERNAL_SERVICE_ERROR

    def test_handle_unknown_exception(self):
        """Should map unknown exceptions to INTERNAL_ERROR."""
        from aragora.server.middleware.error_handler import ErrorHandlerMiddleware
        from aragora.server.error_codes import ErrorCode

        app = MagicMock()
        middleware = ErrorHandlerMiddleware(app, log_errors=False)

        class CustomError(Exception):
            pass

        error = CustomError("Something went wrong")

        response = middleware.handle_exception(error, "req_123", "/api/test")

        assert response.status == 500
        assert response.body["error"]["code"] == ErrorCode.INTERNAL_ERROR

    def test_handle_5xx_error_sanitizes_message_production(self):
        """Should sanitize error message for 5xx in production."""
        from aragora.server.middleware.error_handler import ErrorHandlerMiddleware
        from aragora.server.error_codes import ErrorCode

        app = MagicMock()
        middleware = ErrorHandlerMiddleware(app, include_traceback=False, log_errors=False)

        error = Exception("Sensitive database error: password=secret")

        response = middleware.handle_exception(error, "req_123", "/api/test")

        assert response.status == 500
        # Should not expose internal details
        assert "password" not in response.body["error"]["message"]
        assert "internal error" in response.body["error"]["message"].lower()

    def test_handle_5xx_error_includes_message_development(self):
        """Should include error message for 5xx in development mode."""
        from aragora.server.middleware.error_handler import ErrorHandlerMiddleware

        app = MagicMock()
        middleware = ErrorHandlerMiddleware(app, include_traceback=True, log_errors=False)

        error = Exception("Debug error details")

        response = middleware.handle_exception(error, "req_123", "/api/test")

        assert "Debug error details" in response.body["error"]["message"]

    def test_handle_exception_includes_traceback_development(self):
        """Should include traceback in development mode."""
        from aragora.server.middleware.error_handler import ErrorHandlerMiddleware

        app = MagicMock()
        middleware = ErrorHandlerMiddleware(app, include_traceback=True, log_errors=False)

        try:
            raise Exception("Test error")
        except Exception as e:
            response = middleware.handle_exception(e, "req_123", "/api/test")

        assert "traceback" in response.body["error"]
        assert "Test error" in response.body["error"]["traceback"]


# =============================================================================
# Test create_error_response Helper
# =============================================================================


class TestCreateErrorResponse:
    """Tests for create_error_response utility function."""

    def test_create_basic_error_response(self):
        """Should create basic error response dict."""
        from aragora.server.middleware.error_handler import create_error_response
        from aragora.server.error_codes import ErrorCode

        result = create_error_response(
            code=ErrorCode.VALIDATION_ERROR,
            message="Invalid input",
        )

        assert "error" in result
        assert result["error"]["code"] == ErrorCode.VALIDATION_ERROR
        assert result["error"]["message"] == "Invalid input"
        assert "timestamp" in result["error"]

    def test_create_error_response_with_all_fields(self):
        """Should include all optional fields when provided."""
        from aragora.server.middleware.error_handler import create_error_response
        from aragora.server.error_codes import ErrorCode

        result = create_error_response(
            code=ErrorCode.NOT_FOUND,
            message="Resource not found",
            details={"resource_type": "debate"},
            request_id="req_abc123",
            path="/api/debates/123",
        )

        assert result["error"]["details"] == {"resource_type": "debate"}
        assert result["error"]["request_id"] == "req_abc123"
        assert result["error"]["path"] == "/api/debates/123"

    def test_create_error_response_omits_none_fields(self):
        """Should not include fields that are None."""
        from aragora.server.middleware.error_handler import create_error_response
        from aragora.server.error_codes import ErrorCode

        result = create_error_response(
            code=ErrorCode.INTERNAL_ERROR,
            message="Error",
        )

        assert "details" not in result["error"]
        assert "request_id" not in result["error"]
        assert "path" not in result["error"]


# =============================================================================
# Test Pre-built Error Helpers
# =============================================================================


class TestValidationError:
    """Tests for validation_error helper."""

    def test_validation_error_basic(self):
        """Should create basic validation error."""
        from aragora.server.middleware.error_handler import validation_error
        from aragora.server.error_codes import ErrorCode

        error = validation_error("Invalid email format")

        assert error.code == ErrorCode.VALIDATION_ERROR
        assert error.message == "Invalid email format"
        assert error.details is None

    def test_validation_error_with_field(self):
        """Should include field in details."""
        from aragora.server.middleware.error_handler import validation_error

        error = validation_error("Invalid format", field="email")

        assert error.details == {"field": "email"}

    def test_validation_error_with_field_and_value(self):
        """Should include field and truncated value."""
        from aragora.server.middleware.error_handler import validation_error

        error = validation_error("Invalid format", field="email", value="not-an-email")

        assert error.details["field"] == "email"
        assert error.details["value"] == "not-an-email"

    def test_validation_error_truncates_long_value(self):
        """Should truncate values longer than 100 characters."""
        from aragora.server.middleware.error_handler import validation_error

        long_value = "x" * 200
        error = validation_error("Invalid", field="data", value=long_value)

        assert len(error.details["value"]) == 100


class TestNotFoundError:
    """Tests for not_found_error helper."""

    def test_not_found_error_basic(self):
        """Should create basic not found error."""
        from aragora.server.middleware.error_handler import not_found_error
        from aragora.server.error_codes import ErrorCode

        error = not_found_error("Debate")

        assert error.code == ErrorCode.NOT_FOUND
        assert error.message == "Debate not found"

    def test_not_found_error_with_id(self):
        """Should include resource ID in message and details."""
        from aragora.server.middleware.error_handler import not_found_error

        error = not_found_error("Debate", resource_id="debate_123")

        assert error.message == "Debate 'debate_123' not found"
        assert error.details == {"resource": "Debate", "id": "debate_123"}


class TestPermissionError:
    """Tests for permission_error helper."""

    def test_permission_error_basic(self):
        """Should create basic permission error."""
        from aragora.server.middleware.error_handler import permission_error
        from aragora.server.error_codes import ErrorCode

        error = permission_error("delete")

        assert error.code == ErrorCode.PERMISSION_DENIED
        assert "delete" in error.message

    def test_permission_error_with_resource(self):
        """Should include resource in message."""
        from aragora.server.middleware.error_handler import permission_error

        error = permission_error("delete", resource="debate")

        assert "delete" in error.message
        assert "debate" in error.message
        assert error.details == {"action": "delete", "resource": "debate"}


class TestRateLimitError:
    """Tests for rate_limit_error helper."""

    def test_rate_limit_error_basic(self):
        """Should create basic rate limit error."""
        from aragora.server.middleware.error_handler import rate_limit_error
        from aragora.server.error_codes import ErrorCode

        error = rate_limit_error()

        assert error.code == ErrorCode.RATE_LIMITED
        assert "too many requests" in error.message.lower()

    def test_rate_limit_error_with_retry_after(self):
        """Should include Retry-After header and details."""
        from aragora.server.middleware.error_handler import rate_limit_error

        error = rate_limit_error(retry_after=60)

        assert error.headers == {"Retry-After": "60"}
        assert error.details == {"retry_after_seconds": 60}


# =============================================================================
# Test Exception to Error Code Mapping
# =============================================================================


class TestExceptionErrorMap:
    """Tests for EXCEPTION_ERROR_MAP."""

    def test_exception_error_map_exists(self):
        """Exception to error code mapping should exist."""
        from aragora.server.middleware.error_handler import EXCEPTION_ERROR_MAP

        assert ValueError in EXCEPTION_ERROR_MAP
        assert KeyError in EXCEPTION_ERROR_MAP
        assert TypeError in EXCEPTION_ERROR_MAP
        assert PermissionError in EXCEPTION_ERROR_MAP
        assert FileNotFoundError in EXCEPTION_ERROR_MAP
        assert TimeoutError in EXCEPTION_ERROR_MAP
        assert ConnectionError in EXCEPTION_ERROR_MAP

    def test_exception_error_map_correct_codes(self):
        """Exception mappings should have correct error codes."""
        from aragora.server.middleware.error_handler import EXCEPTION_ERROR_MAP
        from aragora.server.error_codes import ErrorCode

        assert EXCEPTION_ERROR_MAP[ValueError] == ErrorCode.VALIDATION_ERROR
        assert EXCEPTION_ERROR_MAP[KeyError] == ErrorCode.MISSING_FIELD
        assert EXCEPTION_ERROR_MAP[TypeError] == ErrorCode.INVALID_FIELD
        assert EXCEPTION_ERROR_MAP[PermissionError] == ErrorCode.PERMISSION_DENIED
        assert EXCEPTION_ERROR_MAP[FileNotFoundError] == ErrorCode.NOT_FOUND


# =============================================================================
# Test Module Exports
# =============================================================================


class TestModuleExports:
    """Tests for module exports."""

    def test_all_exports_accessible(self):
        """All __all__ exports should be accessible."""
        from aragora.server.middleware.error_handler import (
            APIError,
            ErrorHandlerMiddleware,
            ErrorResponse,
            create_error_response,
            not_found_error,
            permission_error,
            raise_api_error,
            rate_limit_error,
            validation_error,
        )

        assert APIError is not None
        assert ErrorHandlerMiddleware is not None
        assert ErrorResponse is not None
        assert create_error_response is not None
        assert raise_api_error is not None
        assert validation_error is not None
        assert not_found_error is not None
        assert permission_error is not None
        assert rate_limit_error is not None


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_api_error_with_none_status(self):
        """Should auto-derive status when None."""
        from aragora.server.middleware.error_handler import APIError
        from aragora.server.error_codes import ErrorCode

        # Not found should be 404
        error = APIError(code=ErrorCode.NOT_FOUND, message="Not found", status=None)
        assert error.status == 404

        # Rate limited should be 429
        error = APIError(code=ErrorCode.RATE_LIMITED, message="Limited", status=None)
        assert error.status == 429

    def test_api_error_with_empty_message(self):
        """Should handle empty error message."""
        from aragora.server.middleware.error_handler import APIError
        from aragora.server.error_codes import ErrorCode

        error = APIError(code=ErrorCode.INTERNAL_ERROR, message="")
        assert error.message == ""
        assert str(error) == ""

    def test_handle_exception_with_empty_error_message(self):
        """Should generate message for exceptions with no message."""
        from aragora.server.middleware.error_handler import ErrorHandlerMiddleware

        app = MagicMock()
        middleware = ErrorHandlerMiddleware(app, include_traceback=True, log_errors=False)

        class EmptyError(Exception):
            def __str__(self):
                return ""

        error = EmptyError()
        response = middleware.handle_exception(error, "req_123", "/api/test")

        # Should have a generated message
        assert "EmptyError" in response.body["error"]["message"]

    def test_validation_error_with_none_value(self):
        """Should handle None value in validation_error."""
        from aragora.server.middleware.error_handler import validation_error

        # None value should not be included in details
        error = validation_error("Invalid", field="test", value=None)
        assert "value" not in error.details

    def test_timestamp_is_utc(self):
        """Timestamps should be in UTC."""
        from aragora.server.middleware.error_handler import APIError
        from aragora.server.error_codes import ErrorCode

        error = APIError(code=ErrorCode.INTERNAL_ERROR, message="Error")
        result = error.to_dict()

        # Timestamp ends with Z (UTC)
        assert result["error"]["timestamp"].endswith("Z")

    def test_validation_error_only_value_no_field(self):
        """Should handle value without field in validation_error."""
        from aragora.server.middleware.error_handler import validation_error

        error = validation_error("Invalid", value="test-value")
        assert error.details is not None
        assert error.details["value"] == "test-value"

    def test_not_found_error_without_id_has_no_details(self):
        """not_found_error without resource_id should have no details."""
        from aragora.server.middleware.error_handler import not_found_error

        error = not_found_error("Agent")
        assert error.details is None

    def test_permission_error_without_resource_has_no_details(self):
        """permission_error without resource should have no details."""
        from aragora.server.middleware.error_handler import permission_error

        error = permission_error("read")
        assert error.details is None
        assert "read" in error.message

    def test_rate_limit_error_without_retry_after_no_headers(self):
        """rate_limit_error without retry_after should have empty headers."""
        from aragora.server.middleware.error_handler import rate_limit_error

        error = rate_limit_error()
        assert error.headers == {}
        assert error.details is None

    def test_rate_limit_error_without_retry_after_no_details(self):
        """rate_limit_error without retry_after should have no details."""
        from aragora.server.middleware.error_handler import rate_limit_error

        error = rate_limit_error()
        assert error.details is None

    def test_api_error_to_dict_no_optional_fields(self):
        """to_dict with no optional args should omit request_id, path, details."""
        from aragora.server.middleware.error_handler import APIError
        from aragora.server.error_codes import ErrorCode

        error = APIError(code=ErrorCode.VALIDATION_ERROR, message="Test")
        result = error.to_dict()

        assert "request_id" not in result["error"]
        assert "path" not in result["error"]
        assert "details" not in result["error"]

    def test_api_error_inherits_from_exception(self):
        """APIError should be catchable as Exception."""
        from aragora.server.middleware.error_handler import APIError
        from aragora.server.error_codes import ErrorCode

        try:
            raise APIError(code=ErrorCode.NOT_FOUND, message="Not found")
        except Exception as e:
            assert isinstance(e, APIError)
            assert str(e) == "Not found"

    def test_handle_exception_logs_api_error_when_enabled(self):
        """Should log APIError when log_errors is True."""
        from aragora.server.middleware.error_handler import APIError, ErrorHandlerMiddleware
        from aragora.server.error_codes import ErrorCode

        app = MagicMock()
        middleware = ErrorHandlerMiddleware(app, log_errors=True)

        error = APIError(code=ErrorCode.NOT_FOUND, message="Test")

        with patch("aragora.server.middleware.error_handler.logger") as mock_logger:
            middleware.handle_exception(error, "req_log", "/api/test")
            mock_logger.warning.assert_called_once()

    def test_handle_exception_logs_5xx_when_enabled(self):
        """Should log 5xx errors when log_errors is True."""
        from aragora.server.middleware.error_handler import ErrorHandlerMiddleware

        app = MagicMock()
        middleware = ErrorHandlerMiddleware(app, log_errors=True)

        error = RuntimeError("Unexpected crash")

        with patch("aragora.server.middleware.error_handler.logger") as mock_logger:
            middleware.handle_exception(error, "req_5xx", "/api/test")
            mock_logger.exception.assert_called_once()

    def test_handle_exception_no_log_when_disabled(self):
        """Should not log when log_errors is False."""
        from aragora.server.middleware.error_handler import ErrorHandlerMiddleware

        app = MagicMock()
        middleware = ErrorHandlerMiddleware(app, log_errors=False)

        error = RuntimeError("Error")

        with patch("aragora.server.middleware.error_handler.logger") as mock_logger:
            middleware.handle_exception(error, "req_nolog", "/api/test")
            mock_logger.warning.assert_not_called()
            mock_logger.exception.assert_not_called()

    def test_handle_exception_no_traceback_for_4xx(self):
        """Should not include traceback for 4xx errors even in dev mode."""
        from aragora.server.middleware.error_handler import ErrorHandlerMiddleware

        app = MagicMock()
        middleware = ErrorHandlerMiddleware(app, include_traceback=True, log_errors=False)

        try:
            raise ValueError("Bad input")
        except Exception as e:
            response = middleware.handle_exception(e, "req_4xx", "/api/test")

        # 400 errors should not have traceback
        assert "traceback" not in response.body["error"]

    def test_handle_api_error_with_none_status_defaults_to_400(self):
        """APIError with auto-derived status should map correctly."""
        from aragora.server.middleware.error_handler import APIError, ErrorHandlerMiddleware
        from aragora.server.error_codes import ErrorCode

        app = MagicMock()
        middleware = ErrorHandlerMiddleware(app, log_errors=False)

        error = APIError(code=ErrorCode.FORBIDDEN, message="Forbidden", status=None)
        response = middleware.handle_exception(error, "req_nil_status", "/test")

        assert response.status == 403

    def test_create_error_response_with_empty_details(self):
        """create_error_response with empty dict details should still include details."""
        from aragora.server.middleware.error_handler import create_error_response
        from aragora.server.error_codes import ErrorCode

        result = create_error_response(
            code=ErrorCode.VALIDATION_ERROR,
            message="Test",
            details={},
        )
        # Empty dict is falsy, so details should NOT be included
        assert "details" not in result["error"]

    def test_exception_error_map_length(self):
        """Should have 7 exception mappings."""
        from aragora.server.middleware.error_handler import EXCEPTION_ERROR_MAP

        assert len(EXCEPTION_ERROR_MAP) == 7

    def test_error_response_default_headers(self):
        """ErrorResponse should default to empty headers dict."""
        from aragora.server.middleware.error_handler import ErrorResponse

        response = ErrorResponse(status=200, body={})
        assert isinstance(response.headers, dict)
        assert len(response.headers) == 0

    def test_multiple_request_ids_are_unique(self):
        """Multiple request IDs generated in succession should all be unique."""
        from aragora.server.middleware.error_handler import ErrorHandlerMiddleware

        app = MagicMock()
        middleware = ErrorHandlerMiddleware(app)

        ids = set()
        for _ in range(1000):
            ids.add(middleware.generate_request_id())

        assert len(ids) == 1000

    def test_handle_exception_api_error_with_empty_headers(self):
        """APIError with None headers should result in empty response headers."""
        from aragora.server.middleware.error_handler import APIError, ErrorHandlerMiddleware
        from aragora.server.error_codes import ErrorCode

        app = MagicMock()
        middleware = ErrorHandlerMiddleware(app, log_errors=False)

        error = APIError(code=ErrorCode.NOT_FOUND, message="Not found", headers=None)
        response = middleware.handle_exception(error, "req_nh", "/test")

        assert response.headers == {}

    def test_handle_exception_response_has_path_and_request_id(self):
        """Non-APIError exceptions should include path and request_id in response."""
        from aragora.server.middleware.error_handler import ErrorHandlerMiddleware

        app = MagicMock()
        middleware = ErrorHandlerMiddleware(app, log_errors=False)

        error = ValueError("bad value")
        response = middleware.handle_exception(error, "req_meta", "/api/v1/debates")

        assert response.body["error"]["request_id"] == "req_meta"
        assert response.body["error"]["path"] == "/api/v1/debates"

    def test_handle_exception_response_has_timestamp(self):
        """Non-APIError exception responses should include a timestamp."""
        from aragora.server.middleware.error_handler import ErrorHandlerMiddleware

        app = MagicMock()
        middleware = ErrorHandlerMiddleware(app, log_errors=False)

        error = ValueError("timestamp test")
        response = middleware.handle_exception(error, "req_ts", "/test")

        assert "timestamp" in response.body["error"]
        assert response.body["error"]["timestamp"].endswith("Z")
