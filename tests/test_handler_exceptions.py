"""Tests for handler-specific exception utilities.

Tests cover:
- Custom exception classes and their attributes
- Exception classification via EXCEPTION_MAP
- Message redaction for sensitive errors
- Status code mapping
- Retryability detection
"""

import logging
import sqlite3
from unittest.mock import MagicMock

import pytest

from aragora.exceptions import (
    AragoraError,
    AuthenticationError,
    AuthorizationError,
    DatabaseError,
    InputValidationError,
    RateLimitExceededError,
    RecordNotFoundError,
    ValidationError,
)
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


# =============================================================================
# HandlerError Tests
# =============================================================================


class TestHandlerError:
    """Test base HandlerError class."""

    def test_default_status_code(self):
        """Default status code is 500."""
        error = HandlerError("test error")
        assert error.status_code == 500

    def test_custom_status_code_override(self):
        """Can override status code."""
        error = HandlerError("test error", status_code=418)
        assert error.status_code == 418

    def test_message_stored(self):
        """Message is stored and accessible."""
        error = HandlerError("my error message")
        assert str(error) == "my error message"

    def test_details_stored(self):
        """Details dict is stored."""
        error = HandlerError("error", details={"key": "value"})
        assert error.details == {"key": "value"}

    def test_details_default_empty_dict(self):
        """Details defaults to empty dict when not provided."""
        error = HandlerError("error")
        assert error.details == {}

    def test_inherits_from_aragora_error(self):
        """HandlerError inherits from AragoraError."""
        error = HandlerError("test")
        assert isinstance(error, AragoraError)


# =============================================================================
# HandlerValidationError Tests
# =============================================================================


class TestHandlerValidationError:
    """Test HandlerValidationError (400)."""

    def test_status_code_400(self):
        """Status code is 400."""
        error = HandlerValidationError("invalid input")
        assert error.status_code == 400

    def test_field_stored(self):
        """Field attribute is stored."""
        error = HandlerValidationError("invalid", field="email")
        assert error.field == "email"

    def test_field_in_details(self):
        """Field is included in details dict."""
        error = HandlerValidationError("invalid", field="username")
        assert error.details["field"] == "username"

    def test_field_optional(self):
        """Field is optional."""
        error = HandlerValidationError("invalid input")
        assert error.field is None

    def test_message_preserved(self):
        """Message is preserved."""
        error = HandlerValidationError("Email is required")
        assert str(error) == "Email is required"

    def test_details_merging(self):
        """Field is merged into existing details."""
        error = HandlerValidationError("invalid", field="name", details={"extra": "info"})
        assert error.details["field"] == "name"
        assert error.details["extra"] == "info"

    def test_inherits_from_handler_error(self):
        """Inherits from HandlerError."""
        error = HandlerValidationError("test")
        assert isinstance(error, HandlerError)


# =============================================================================
# HandlerNotFoundError Tests
# =============================================================================


class TestHandlerNotFoundError:
    """Test HandlerNotFoundError (404)."""

    def test_status_code_404(self):
        """Status code is 404."""
        error = HandlerNotFoundError("debate", "deb-123")
        assert error.status_code == 404

    def test_resource_type_in_message(self):
        """Resource type appears in message."""
        error = HandlerNotFoundError("debate", "deb-123")
        assert "debate" in str(error)

    def test_resource_id_in_message(self):
        """Resource ID appears in message."""
        error = HandlerNotFoundError("debate", "deb-123")
        assert "deb-123" in str(error)

    def test_attributes_stored(self):
        """Resource type and ID are stored as attributes."""
        error = HandlerNotFoundError("user", "usr-456")
        assert error.resource_type == "user"
        assert error.resource_id == "usr-456"

    def test_details_contains_resource_info(self):
        """Details dict contains resource info."""
        error = HandlerNotFoundError("agent", "agent-789")
        assert error.details["resource_type"] == "agent"
        assert error.details["resource_id"] == "agent-789"


# =============================================================================
# HandlerAuthorizationError Tests
# =============================================================================


class TestHandlerAuthorizationError:
    """Test HandlerAuthorizationError (403)."""

    def test_status_code_403(self):
        """Status code is 403."""
        error = HandlerAuthorizationError("delete")
        assert error.status_code == 403

    def test_action_in_message(self):
        """Action appears in message."""
        error = HandlerAuthorizationError("delete")
        assert "delete" in str(error)

    def test_resource_in_message_when_provided(self):
        """Resource appears in message when provided."""
        error = HandlerAuthorizationError("edit", resource="debate-123")
        assert "debate-123" in str(error)

    def test_resource_optional(self):
        """Resource is optional."""
        error = HandlerAuthorizationError("view")
        assert error.resource is None

    def test_details_contain_action_and_resource(self):
        """Details contain action and resource."""
        error = HandlerAuthorizationError("modify", resource="settings")
        assert error.details["action"] == "modify"
        assert error.details["resource"] == "settings"


# =============================================================================
# HandlerConflictError Tests
# =============================================================================


class TestHandlerConflictError:
    """Test HandlerConflictError (409)."""

    def test_status_code_409(self):
        """Status code is 409."""
        error = HandlerConflictError("Resource already exists")
        assert error.status_code == 409

    def test_message_preserved(self):
        """Message is preserved."""
        error = HandlerConflictError("Debate already started")
        assert "Debate already started" in str(error)

    def test_resource_type_optional(self):
        """Resource type is optional."""
        error = HandlerConflictError("conflict")
        assert error.details["resource_type"] is None

    def test_details_contain_resource_type(self):
        """Details contain resource type when provided."""
        error = HandlerConflictError("exists", resource_type="debate")
        assert error.details["resource_type"] == "debate"


# =============================================================================
# HandlerRateLimitError Tests
# =============================================================================


class TestHandlerRateLimitError:
    """Test HandlerRateLimitError (429)."""

    def test_status_code_429(self):
        """Status code is 429."""
        error = HandlerRateLimitError()
        assert error.status_code == 429

    def test_default_message(self):
        """Default message is 'Rate limit exceeded'."""
        error = HandlerRateLimitError()
        assert str(error) == "Rate limit exceeded"

    def test_custom_message(self):
        """Can provide custom message."""
        error = HandlerRateLimitError("Too many requests, slow down")
        assert str(error) == "Too many requests, slow down"

    def test_retry_after_stored(self):
        """Retry-after value is stored."""
        error = HandlerRateLimitError(retry_after=60)
        assert error.retry_after == 60

    def test_retry_after_in_details(self):
        """Retry-after is included in details."""
        error = HandlerRateLimitError(retry_after=30)
        assert error.details["retry_after"] == 30


# =============================================================================
# HandlerExternalServiceError Tests
# =============================================================================


class TestHandlerExternalServiceError:
    """Test HandlerExternalServiceError (502/503)."""

    def test_status_code_502_default(self):
        """Default status code is 502."""
        error = HandlerExternalServiceError("OpenAI", "timeout")
        assert error.status_code == 502

    def test_status_code_503_when_unavailable(self):
        """Status code is 503 when unavailable=True."""
        error = HandlerExternalServiceError("Anthropic", "down", unavailable=True)
        assert error.status_code == 503

    def test_service_name_in_message(self):
        """Service name appears in message."""
        error = HandlerExternalServiceError("Stripe", "payment failed")
        assert "Stripe" in str(error)

    def test_service_stored(self):
        """Service name is stored as attribute."""
        error = HandlerExternalServiceError("Redis", "connection lost")
        assert error.service == "Redis"

    def test_details_contain_service_and_unavailable(self):
        """Details contain service and unavailable flag."""
        error = HandlerExternalServiceError("Database", "error", unavailable=True)
        assert error.details["service"] == "Database"
        assert error.details["unavailable"] is True


# =============================================================================
# HandlerDatabaseError Tests
# =============================================================================


class TestHandlerDatabaseError:
    """Test HandlerDatabaseError (500)."""

    def test_status_code_500(self):
        """Status code is 500."""
        error = HandlerDatabaseError("insert")
        assert error.status_code == 500

    def test_operation_in_message(self):
        """Operation appears in message."""
        error = HandlerDatabaseError("update")
        assert "update" in str(error)

    def test_optional_message_appended(self):
        """Optional message is appended."""
        error = HandlerDatabaseError("delete", message="constraint violation")
        assert "delete" in str(error)
        assert "constraint violation" in str(error)

    def test_operation_stored(self):
        """Operation is stored as attribute."""
        error = HandlerDatabaseError("query")
        assert error.operation == "query"


# =============================================================================
# classify_exception Tests
# =============================================================================


class TestClassifyException:
    """Test classify_exception() function."""

    # Handler exceptions (exact matches)
    def test_handler_validation_error(self):
        """HandlerValidationError returns 400, info, message."""
        exc = HandlerValidationError("bad input")
        status, level, msg = classify_exception(exc)
        assert status == 400
        assert level == "info"
        assert msg == "bad input"

    def test_handler_not_found_error(self):
        """HandlerNotFoundError returns 404, info, message."""
        exc = HandlerNotFoundError("debate", "123")
        status, level, msg = classify_exception(exc)
        assert status == 404
        assert level == "info"
        assert "debate" in msg

    def test_handler_authorization_error(self):
        """HandlerAuthorizationError returns 403, info, message."""
        exc = HandlerAuthorizationError("delete")
        status, level, msg = classify_exception(exc)
        assert status == 403
        assert level == "info"
        assert "delete" in msg

    def test_handler_rate_limit_error(self):
        """HandlerRateLimitError returns 429, info, message."""
        exc = HandlerRateLimitError()
        status, level, msg = classify_exception(exc)
        assert status == 429
        assert level == "info"

    def test_handler_external_service_error(self):
        """HandlerExternalServiceError returns 502, error, message."""
        exc = HandlerExternalServiceError("API", "failed")
        status, level, msg = classify_exception(exc)
        assert status == 502
        assert level == "error"

    def test_handler_database_error_generic(self):
        """HandlerDatabaseError returns 500, error, generic message."""
        exc = HandlerDatabaseError("insert")
        status, level, msg = classify_exception(exc)
        assert status == 500
        assert level == "error"
        assert msg == "Internal server error"  # Redacted

    # Base exceptions (from aragora.exceptions)
    def test_input_validation_error(self):
        """InputValidationError returns 400, info, message."""
        exc = InputValidationError("field", "invalid")
        status, level, msg = classify_exception(exc)
        assert status == 400
        assert level == "info"

    def test_record_not_found_error(self):
        """RecordNotFoundError returns 404, info, message."""
        exc = RecordNotFoundError("debate", "deb-123")
        status, level, msg = classify_exception(exc)
        assert status == 404
        assert level == "info"

    def test_authentication_error(self):
        """AuthenticationError returns 401, info, message."""
        exc = AuthenticationError("Invalid token")
        status, level, msg = classify_exception(exc)
        assert status == 401
        assert level == "info"

    def test_rate_limit_exceeded_error(self):
        """RateLimitExceededError returns 429, info, message."""
        exc = RateLimitExceededError("Too many requests", window_seconds=60)
        status, level, msg = classify_exception(exc)
        assert status == 429
        assert level == "info"

    # Generic exceptions with redaction
    def test_key_error_generic_message(self):
        """KeyError returns 400, info, generic message (redacted)."""
        exc = KeyError("internal_key_name")
        status, level, msg = classify_exception(exc)
        assert status == 400
        assert level == "info"
        assert msg == "Invalid request"  # Redacted

    def test_type_error_generic_message(self):
        """TypeError returns 400, warning, generic message (redacted)."""
        exc = TypeError("expected str, got int")
        status, level, msg = classify_exception(exc)
        assert status == 400
        assert level == "warning"
        assert msg == "Invalid request"  # Redacted

    def test_sqlite_error_generic_message(self):
        """sqlite3.Error returns 500, error, generic message."""
        exc = sqlite3.Error("database locked")
        status, level, msg = classify_exception(exc)
        assert status == 500
        assert level == "error"
        assert msg == "Internal server error"  # Redacted

    def test_sqlite_integrity_error(self):
        """sqlite3.IntegrityError returns 409, warning, generic message."""
        exc = sqlite3.IntegrityError("UNIQUE constraint failed")
        status, level, msg = classify_exception(exc)
        assert status == 409
        assert level == "warning"
        assert msg == "Resource conflict"  # Redacted

    def test_timeout_error(self):
        """TimeoutError returns 504, warning, generic message."""
        exc = TimeoutError("operation timed out")
        status, level, msg = classify_exception(exc)
        assert status == 504
        assert level == "warning"
        assert msg == "Request timeout"  # Redacted

    def test_value_error_includes_message(self):
        """ValueError returns 400, info, actual message."""
        exc = ValueError("invalid value")
        status, level, msg = classify_exception(exc)
        assert status == 400
        assert level == "info"
        assert msg == "invalid value"

    # Subclass matching
    def test_subclass_match_aragora_error(self):
        """Unknown AragoraError subclass returns 500, error, message."""
        class CustomAragoraError(AragoraError):
            pass

        exc = CustomAragoraError("custom error")
        status, level, msg = classify_exception(exc)
        assert status == 500
        assert level == "error"
        assert msg == "custom error"

    def test_subclass_match_handler_error_custom_status(self):
        """Custom HandlerError subclass with status_code is respected."""
        class CustomHandlerError(HandlerError):
            status_code = 422

        exc = CustomHandlerError("unprocessable")
        status, level, msg = classify_exception(exc)
        assert status == 422
        assert level == "error"
        assert msg == "unprocessable"

    def test_unknown_exception_returns_500(self):
        """Unknown exception returns 500, error, generic message."""
        exc = RuntimeError("something unexpected")
        status, level, msg = classify_exception(exc)
        assert status == 500
        assert level == "error"
        assert msg == "Internal server error"


# =============================================================================
# handle_handler_error Tests
# =============================================================================


class TestHandleHandlerError:
    """Test handle_handler_error() function."""

    @pytest.fixture
    def mock_logger(self):
        """Create mock logger."""
        logger = MagicMock(spec=logging.Logger)
        logger.info = MagicMock()
        logger.warning = MagicMock()
        logger.error = MagicMock()
        return logger

    def test_returns_status_and_message(self, mock_logger):
        """Returns tuple of status code and message."""
        exc = HandlerValidationError("bad input")
        status, message = handle_handler_error(exc, "validate", mock_logger)
        assert status == 400
        assert message == "bad input"

    def test_logs_at_correct_level_info(self, mock_logger):
        """Logs at info level for validation errors."""
        exc = HandlerValidationError("bad input")
        handle_handler_error(exc, "validate", mock_logger)
        mock_logger.info.assert_called_once()
        mock_logger.warning.assert_not_called()
        mock_logger.error.assert_not_called()

    def test_logs_at_correct_level_warning(self, mock_logger):
        """Logs at warning level for timeout errors."""
        exc = TimeoutError("timed out")
        handle_handler_error(exc, "fetch", mock_logger)
        mock_logger.warning.assert_called_once()

    def test_logs_at_correct_level_error(self, mock_logger):
        """Logs at error level for database errors."""
        exc = HandlerDatabaseError("insert")
        handle_handler_error(exc, "create", mock_logger)
        mock_logger.error.assert_called_once()

    def test_logs_operation_name(self, mock_logger):
        """Operation name appears in log message."""
        exc = HandlerValidationError("invalid")
        handle_handler_error(exc, "create_debate", mock_logger)
        call_args = mock_logger.info.call_args[0][0]
        assert "create_debate" in call_args

    def test_logs_exception_type(self, mock_logger):
        """Exception type appears in log message."""
        exc = HandlerValidationError("invalid")
        handle_handler_error(exc, "create", mock_logger)
        call_args = mock_logger.info.call_args[0][0]
        assert "HandlerValidationError" in call_args

    def test_include_traceback_forces_exc_info(self, mock_logger):
        """include_traceback=True adds exc_info to log call."""
        exc = HandlerValidationError("invalid")
        handle_handler_error(exc, "op", mock_logger, include_traceback=True)
        call_kwargs = mock_logger.info.call_args[1]
        assert call_kwargs.get("exc_info") is True

    def test_error_level_includes_traceback(self, mock_logger):
        """Error level automatically includes traceback."""
        exc = HandlerDatabaseError("insert")
        handle_handler_error(exc, "create", mock_logger)
        call_kwargs = mock_logger.error.call_args[1]
        assert call_kwargs.get("exc_info") is True


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestIsClientError:
    """Test is_client_error() function."""

    def test_true_for_400(self):
        """Returns True for 400 status."""
        exc = HandlerValidationError("invalid")
        assert is_client_error(exc) is True

    def test_true_for_404(self):
        """Returns True for 404 status."""
        exc = HandlerNotFoundError("resource", "id")
        assert is_client_error(exc) is True

    def test_true_for_429(self):
        """Returns True for 429 status."""
        exc = HandlerRateLimitError()
        assert is_client_error(exc) is True

    def test_false_for_500(self):
        """Returns False for 500 status."""
        exc = HandlerDatabaseError("insert")
        assert is_client_error(exc) is False

    def test_false_for_502(self):
        """Returns False for 502 status."""
        exc = HandlerExternalServiceError("API", "error")
        assert is_client_error(exc) is False


class TestIsServerError:
    """Test is_server_error() function."""

    def test_true_for_500(self):
        """Returns True for 500 status."""
        exc = HandlerDatabaseError("query")
        assert is_server_error(exc) is True

    def test_true_for_502(self):
        """Returns True for 502 status."""
        exc = HandlerExternalServiceError("API", "error")
        assert is_server_error(exc) is True

    def test_true_for_503(self):
        """Returns True for 503 status."""
        exc = HandlerExternalServiceError("API", "down", unavailable=True)
        assert is_server_error(exc) is True

    def test_false_for_400(self):
        """Returns False for 400 status."""
        exc = HandlerValidationError("invalid")
        assert is_server_error(exc) is False

    def test_false_for_404(self):
        """Returns False for 404 status."""
        exc = HandlerNotFoundError("resource", "id")
        assert is_server_error(exc) is False


class TestIsRetryableError:
    """Test is_retryable_error() function."""

    def test_retryable_429(self):
        """429 is retryable."""
        exc = HandlerRateLimitError()
        assert is_retryable_error(exc) is True

    def test_retryable_502(self):
        """502 is retryable."""
        exc = HandlerExternalServiceError("API", "error")
        assert is_retryable_error(exc) is True

    def test_retryable_503(self):
        """503 is retryable."""
        exc = HandlerExternalServiceError("API", "down", unavailable=True)
        assert is_retryable_error(exc) is True

    def test_retryable_504(self):
        """504 is retryable."""
        exc = TimeoutError("timeout")
        assert is_retryable_error(exc) is True

    def test_not_retryable_400(self):
        """400 is not retryable."""
        exc = HandlerValidationError("invalid")
        assert is_retryable_error(exc) is False

    def test_not_retryable_404(self):
        """404 is not retryable."""
        exc = HandlerNotFoundError("resource", "id")
        assert is_retryable_error(exc) is False

    def test_not_retryable_500(self):
        """500 is not retryable."""
        exc = HandlerDatabaseError("insert")
        assert is_retryable_error(exc) is False


# =============================================================================
# EXCEPTION_MAP Tests
# =============================================================================


class TestExceptionMap:
    """Test EXCEPTION_MAP structure."""

    def test_all_handler_exceptions_mapped(self):
        """All handler exceptions are in EXCEPTION_MAP."""
        handler_exceptions = [
            HandlerValidationError,
            HandlerNotFoundError,
            HandlerAuthorizationError,
            HandlerConflictError,
            HandlerRateLimitError,
            HandlerExternalServiceError,
            HandlerDatabaseError,
        ]
        for exc_type in handler_exceptions:
            assert exc_type in EXCEPTION_MAP, f"{exc_type.__name__} not in EXCEPTION_MAP"

    def test_map_values_are_tuples(self):
        """All EXCEPTION_MAP values are (status, level, include_msg) tuples."""
        for exc_type, value in EXCEPTION_MAP.items():
            assert isinstance(value, tuple), f"{exc_type.__name__} value is not tuple"
            assert len(value) == 3, f"{exc_type.__name__} tuple length != 3"
            status, level, include_msg = value
            assert isinstance(status, int)
            assert isinstance(level, str)
            assert isinstance(include_msg, bool)

    def test_log_levels_valid(self):
        """All log levels are valid."""
        valid_levels = {"debug", "info", "warning", "error", "critical"}
        for exc_type, (_, level, _) in EXCEPTION_MAP.items():
            assert level in valid_levels, f"{exc_type.__name__} has invalid level: {level}"


class TestGenericErrorMessages:
    """Test GENERIC_ERROR_MESSAGES structure."""

    def test_all_common_status_codes_covered(self):
        """Common HTTP status codes have generic messages."""
        expected_codes = [400, 401, 403, 404, 409, 429, 500, 502, 503, 504]
        for code in expected_codes:
            assert code in GENERIC_ERROR_MESSAGES, f"Status {code} not in GENERIC_ERROR_MESSAGES"

    def test_messages_are_strings(self):
        """All generic messages are non-empty strings."""
        for code, message in GENERIC_ERROR_MESSAGES.items():
            assert isinstance(message, str)
            assert len(message) > 0
