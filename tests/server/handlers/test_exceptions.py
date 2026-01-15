"""
Tests for handler-specific exception utilities.

Tests:
- Handler exception classes (status codes, details)
- Exception classification
- Error handling helper functions
"""

import logging
import sqlite3

import pytest

from aragora.server.handlers.exceptions import (
    EXCEPTION_MAP,
    GENERIC_ERROR_MESSAGES,
    HandlerAuthorizationError,
    HandlerConflictError,
    HandlerDatabaseError,
    HandlerError,
    HandlerExternalServiceError,
    HandlerNotFoundError,
    HandlerRateLimitError,
    HandlerValidationError,
    classify_exception,
    handle_handler_error,
    is_client_error,
    is_retryable_error,
    is_server_error,
)


class TestHandlerError:
    """Tests for base HandlerError class."""

    def test_default_status_code(self):
        """Default status code should be 500."""
        error = HandlerError("test error")
        assert error.status_code == 500
        assert str(error) == "test error"

    def test_custom_status_code(self):
        """Should accept custom status code."""
        error = HandlerError("test error", status_code=418)
        assert error.status_code == 418

    def test_with_details(self):
        """Should store details."""
        error = HandlerError("test error", details={"key": "value"})
        assert error.details == {"key": "value"}


class TestHandlerValidationError:
    """Tests for HandlerValidationError class."""

    def test_status_code(self):
        """Validation errors should return 400."""
        error = HandlerValidationError("Invalid input")
        assert error.status_code == 400

    def test_with_field(self):
        """Should include field name in details."""
        error = HandlerValidationError("Invalid email", field="email")
        assert error.field == "email"
        assert error.details["field"] == "email"

    def test_without_field(self):
        """Should work without field."""
        error = HandlerValidationError("Invalid input")
        assert error.field is None


class TestHandlerNotFoundError:
    """Tests for HandlerNotFoundError class."""

    def test_status_code(self):
        """Not found errors should return 404."""
        error = HandlerNotFoundError("Debate", "debate-123")
        assert error.status_code == 404

    def test_message_format(self):
        """Should format message correctly."""
        error = HandlerNotFoundError("Agent", "claude")
        assert "Agent not found: claude" in str(error)

    def test_stores_resource_info(self):
        """Should store resource type and ID."""
        error = HandlerNotFoundError("User", "user-456")
        assert error.resource_type == "User"
        assert error.resource_id == "user-456"
        assert error.details["resource_type"] == "User"
        assert error.details["resource_id"] == "user-456"


class TestHandlerAuthorizationError:
    """Tests for HandlerAuthorizationError class."""

    def test_status_code(self):
        """Authorization errors should return 403."""
        error = HandlerAuthorizationError("delete")
        assert error.status_code == 403

    def test_message_without_resource(self):
        """Should format message without resource."""
        error = HandlerAuthorizationError("view")
        assert "Not authorized to view" in str(error)

    def test_message_with_resource(self):
        """Should format message with resource."""
        error = HandlerAuthorizationError("edit", "debate-123")
        assert "Not authorized to edit on debate-123" in str(error)

    def test_stores_action_and_resource(self):
        """Should store action and resource."""
        error = HandlerAuthorizationError("delete", "user-456")
        assert error.action == "delete"
        assert error.resource == "user-456"


class TestHandlerConflictError:
    """Tests for HandlerConflictError class."""

    def test_status_code(self):
        """Conflict errors should return 409."""
        error = HandlerConflictError("Resource already exists")
        assert error.status_code == 409

    def test_with_resource_type(self):
        """Should include resource type in details."""
        error = HandlerConflictError("Duplicate", resource_type="debate")
        assert error.details["resource_type"] == "debate"


class TestHandlerRateLimitError:
    """Tests for HandlerRateLimitError class."""

    def test_status_code(self):
        """Rate limit errors should return 429."""
        error = HandlerRateLimitError()
        assert error.status_code == 429

    def test_default_message(self):
        """Should have default message."""
        error = HandlerRateLimitError()
        assert str(error) == "Rate limit exceeded"

    def test_custom_message(self):
        """Should accept custom message."""
        error = HandlerRateLimitError("Too many requests per minute")
        assert "Too many requests per minute" in str(error)

    def test_retry_after(self):
        """Should store retry_after."""
        error = HandlerRateLimitError(retry_after=60)
        assert error.retry_after == 60
        assert error.details["retry_after"] == 60


class TestHandlerExternalServiceError:
    """Tests for HandlerExternalServiceError class."""

    def test_default_status_code(self):
        """External service errors should return 502 by default."""
        error = HandlerExternalServiceError("OpenAI", "API timeout")
        assert error.status_code == 502

    def test_unavailable_status_code(self):
        """Should return 503 when service unavailable."""
        error = HandlerExternalServiceError("Database", "Connection refused", unavailable=True)
        assert error.status_code == 503

    def test_message_format(self):
        """Should format message correctly."""
        error = HandlerExternalServiceError("Anthropic", "Rate limited")
        assert "Anthropic service error: Rate limited" in str(error)

    def test_stores_service(self):
        """Should store service name."""
        error = HandlerExternalServiceError("Stripe", "Payment failed")
        assert error.service == "Stripe"


class TestHandlerDatabaseError:
    """Tests for HandlerDatabaseError class."""

    def test_status_code(self):
        """Database errors should return 500."""
        error = HandlerDatabaseError("insert")
        assert error.status_code == 500

    def test_message_without_details(self):
        """Should format message without details."""
        error = HandlerDatabaseError("select")
        assert "Database error during select" in str(error)

    def test_message_with_details(self):
        """Should format message with details."""
        error = HandlerDatabaseError("insert", "constraint violation")
        assert "Database error during insert: constraint violation" in str(error)


class TestClassifyException:
    """Tests for classify_exception function."""

    def test_handler_validation_error(self):
        """Should classify validation errors as 400."""
        exc = HandlerValidationError("Bad input")
        status, level, message = classify_exception(exc)
        assert status == 400
        assert level == "info"
        assert "Bad input" in message

    def test_handler_not_found_error(self):
        """Should classify not found errors as 404."""
        exc = HandlerNotFoundError("Debate", "123")
        status, level, message = classify_exception(exc)
        assert status == 404
        assert level == "info"
        assert "Debate not found: 123" in message

    def test_handler_authorization_error(self):
        """Should classify authorization errors as 403."""
        exc = HandlerAuthorizationError("delete")
        status, level, message = classify_exception(exc)
        assert status == 403
        assert level == "info"

    def test_handler_rate_limit_error(self):
        """Should classify rate limit errors as 429."""
        exc = HandlerRateLimitError()
        status, level, message = classify_exception(exc)
        assert status == 429
        assert level == "info"

    def test_handler_external_service_error(self):
        """Should classify external service errors as 502."""
        exc = HandlerExternalServiceError("API", "failed")
        status, level, message = classify_exception(exc)
        assert status == 502
        assert level == "error"

    def test_handler_database_error(self):
        """Should classify database errors as 500 with generic message."""
        exc = HandlerDatabaseError("query")
        status, level, message = classify_exception(exc)
        assert status == 500
        assert level == "error"
        # Database errors should not expose details
        assert message == "Internal server error"

    def test_value_error(self):
        """Should classify ValueError as 400."""
        exc = ValueError("invalid value")
        status, level, message = classify_exception(exc)
        assert status == 400
        assert level == "info"
        assert "invalid value" in message

    def test_key_error(self):
        """Should classify KeyError as 400 with generic message."""
        exc = KeyError("missing_key")
        status, level, message = classify_exception(exc)
        assert status == 400
        assert level == "info"
        # KeyError should not expose internal key names
        assert message == "Invalid request"

    def test_timeout_error(self):
        """Should classify TimeoutError as 504."""
        exc = TimeoutError("request timed out")
        status, level, message = classify_exception(exc)
        assert status == 504
        assert level == "warning"

    def test_unknown_exception(self):
        """Should classify unknown exceptions as 500."""
        exc = RuntimeError("something went wrong")
        status, level, message = classify_exception(exc)
        assert status == 500
        assert level == "error"
        assert message == "Internal server error"

    def test_sqlite_error(self):
        """Should classify sqlite3.Error as 500."""
        exc = sqlite3.Error("database locked")
        status, level, message = classify_exception(exc)
        assert status == 500
        assert level == "error"

    def test_sqlite_integrity_error(self):
        """Should classify sqlite3.IntegrityError as 409."""
        exc = sqlite3.IntegrityError("UNIQUE constraint failed")
        status, level, message = classify_exception(exc)
        assert status == 409
        assert level == "warning"


class TestHandleHandlerError:
    """Tests for handle_handler_error function."""

    def test_logs_and_returns_status(self):
        """Should log error and return status/message."""
        logger = logging.getLogger("test")
        exc = HandlerValidationError("Invalid email")
        status, message = handle_handler_error(exc, "create user", logger)
        assert status == 400
        assert "Invalid email" in message

    def test_logs_server_error_with_traceback(self):
        """Should log server errors with traceback."""
        logger = logging.getLogger("test")
        exc = RuntimeError("unexpected error")
        status, message = handle_handler_error(exc, "process request", logger)
        assert status == 500

    def test_include_traceback_flag(self):
        """Should respect include_traceback flag."""
        logger = logging.getLogger("test")
        exc = HandlerValidationError("Bad input")
        status, message = handle_handler_error(exc, "validate", logger, include_traceback=True)
        assert status == 400


class TestIsClientError:
    """Tests for is_client_error function."""

    def test_validation_error_is_client_error(self):
        """Validation errors should be client errors."""
        assert is_client_error(HandlerValidationError("bad"))

    def test_not_found_is_client_error(self):
        """Not found errors should be client errors."""
        assert is_client_error(HandlerNotFoundError("X", "1"))

    def test_authorization_is_client_error(self):
        """Authorization errors should be client errors."""
        assert is_client_error(HandlerAuthorizationError("view"))

    def test_rate_limit_is_client_error(self):
        """Rate limit errors should be client errors."""
        assert is_client_error(HandlerRateLimitError())

    def test_database_error_is_not_client_error(self):
        """Database errors should not be client errors."""
        assert not is_client_error(HandlerDatabaseError("query"))


class TestIsServerError:
    """Tests for is_server_error function."""

    def test_database_error_is_server_error(self):
        """Database errors should be server errors."""
        assert is_server_error(HandlerDatabaseError("query"))

    def test_external_service_error_is_server_error(self):
        """External service errors should be server errors."""
        assert is_server_error(HandlerExternalServiceError("API", "failed"))

    def test_validation_error_is_not_server_error(self):
        """Validation errors should not be server errors."""
        assert not is_server_error(HandlerValidationError("bad"))


class TestIsRetryableError:
    """Tests for is_retryable_error function."""

    def test_rate_limit_is_retryable(self):
        """Rate limit errors should be retryable."""
        assert is_retryable_error(HandlerRateLimitError())

    def test_external_service_error_is_retryable(self):
        """External service errors should be retryable."""
        assert is_retryable_error(HandlerExternalServiceError("API", "timeout"))

    def test_service_unavailable_is_retryable(self):
        """Service unavailable errors should be retryable."""
        assert is_retryable_error(HandlerExternalServiceError("DB", "down", unavailable=True))

    def test_timeout_is_retryable(self):
        """Timeout errors should be retryable."""
        assert is_retryable_error(TimeoutError("request timed out"))

    def test_validation_error_is_not_retryable(self):
        """Validation errors should not be retryable."""
        assert not is_retryable_error(HandlerValidationError("bad"))

    def test_not_found_is_not_retryable(self):
        """Not found errors should not be retryable."""
        assert not is_retryable_error(HandlerNotFoundError("X", "1"))


class TestExceptionMapAndGenericMessages:
    """Tests for EXCEPTION_MAP and GENERIC_ERROR_MESSAGES constants."""

    def test_exception_map_has_expected_types(self):
        """EXCEPTION_MAP should have common exception types."""
        assert HandlerValidationError in EXCEPTION_MAP
        assert HandlerNotFoundError in EXCEPTION_MAP
        assert HandlerAuthorizationError in EXCEPTION_MAP
        assert ValueError in EXCEPTION_MAP
        assert TimeoutError in EXCEPTION_MAP

    def test_generic_messages_has_common_codes(self):
        """GENERIC_ERROR_MESSAGES should have common status codes."""
        assert 400 in GENERIC_ERROR_MESSAGES
        assert 401 in GENERIC_ERROR_MESSAGES
        assert 403 in GENERIC_ERROR_MESSAGES
        assert 404 in GENERIC_ERROR_MESSAGES
        assert 500 in GENERIC_ERROR_MESSAGES
