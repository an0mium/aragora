"""Tests for aragora/server/handlers/exceptions.py.

Comprehensive coverage of all handler exception classes and utilities:
1. HandlerError base class - instantiation, status codes, message formatting
2. HandlerValidationError - field tracking, default status 400
3. HandlerNotFoundError - resource_type/resource_id formatting
4. HandlerAuthorizationError - action/resource message building
5. HandlerConflictError - 409 status, resource_type tracking
6. HandlerRateLimitError - retry_after, default message
7. HandlerExternalServiceError - service name, unavailable toggle (502/503)
8. HandlerDatabaseError - operation tracking, message formatting
9. HandlerJSONParseError - source/reason, inherits from HandlerValidationError
10. HandlerTimeoutError - operation/timeout_seconds formatting
11. HandlerStreamError - stream_type, custom code
12. HandlerOAuthError - provider/reason/oauth_error/recoverable
13. EXCEPTION_MAP - correct mappings for all exception types
14. GENERIC_ERROR_MESSAGES - all status codes covered
15. classify_exception() - exact match, subclass match, fallbacks
16. handle_handler_error() - logging, status codes, traceback control
17. is_client_error() - 4xx detection
18. is_server_error() - 5xx detection
19. is_retryable_error() - 429, 502, 503, 504 detection
20. Inheritance hierarchy - all exceptions inherit correctly
21. Edge cases - None values, empty strings, boundary conditions
"""

from __future__ import annotations

import logging
import sqlite3
from unittest.mock import MagicMock, patch

import pytest

from aragora.exceptions import (
    AragoraError,
    AuthenticationError,
    AuthError,
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
    HandlerJSONParseError,
    HandlerNotFoundError,
    HandlerOAuthError,
    HandlerRateLimitError,
    HandlerStreamError,
    HandlerTimeoutError,
    HandlerValidationError,
    classify_exception,
    handle_handler_error,
    is_client_error,
    is_retryable_error,
    is_server_error,
)


# =============================================================================
# HandlerError Base Class
# =============================================================================


class TestHandlerError:
    """Test HandlerError base exception class."""

    def test_basic_instantiation(self):
        err = HandlerError("something broke")
        assert str(err) == "something broke"
        assert err.status_code == 500

    def test_custom_status_code(self):
        err = HandlerError("bad request", status_code=400)
        assert err.status_code == 400

    def test_default_status_code_is_500(self):
        err = HandlerError("internal")
        assert err.status_code == 500

    def test_with_details(self):
        details = {"key": "value", "count": 42}
        err = HandlerError("error", details=details)
        assert err.details == details

    def test_details_default_empty(self):
        err = HandlerError("error")
        assert err.details == {}

    def test_none_status_code_keeps_default(self):
        err = HandlerError("error", status_code=None)
        assert err.status_code == 500

    def test_inherits_from_aragora_error(self):
        err = HandlerError("test")
        assert isinstance(err, AragoraError)
        assert isinstance(err, Exception)

    def test_message_attribute(self):
        err = HandlerError("test message")
        assert err.message == "test message"

    def test_empty_message(self):
        err = HandlerError("")
        assert str(err) == ""

    def test_status_code_override_via_init(self):
        """Even though class default is 500, init can override."""
        err = HandlerError("err", status_code=418)
        assert err.status_code == 418

    def test_str_with_details(self):
        """AragoraError.__str__ includes details if present."""
        err = HandlerError("msg", details={"k": "v"})
        s = str(err)
        assert "msg" in s
        assert "k" in s


# =============================================================================
# HandlerValidationError
# =============================================================================


class TestHandlerValidationError:
    """Test HandlerValidationError (400 Bad Request)."""

    def test_basic_instantiation(self):
        err = HandlerValidationError("invalid input")
        assert err.status_code == 400
        assert "invalid input" in str(err)

    def test_with_field(self):
        err = HandlerValidationError("bad value", field="email")
        assert err.field == "email"
        assert err.details["field"] == "email"

    def test_field_none(self):
        err = HandlerValidationError("bad value", field=None)
        assert err.field is None
        assert "field" not in err.details

    def test_with_details(self):
        err = HandlerValidationError("bad", details={"extra": "info"})
        assert err.details["extra"] == "info"

    def test_field_added_to_existing_details(self):
        err = HandlerValidationError("bad", field="name", details={"extra": 1})
        assert err.details["field"] == "name"
        assert err.details["extra"] == 1

    def test_inherits_from_handler_error(self):
        err = HandlerValidationError("test")
        assert isinstance(err, HandlerError)
        assert isinstance(err, AragoraError)

    def test_default_details_when_none(self):
        err = HandlerValidationError("test", details=None)
        assert isinstance(err.details, dict)

    def test_empty_field_string(self):
        """Empty string field is truthy, should be included."""
        err = HandlerValidationError("bad", field="")
        # Empty string is falsy, so field should NOT be in details
        assert "field" not in err.details
        assert err.field == ""


# =============================================================================
# HandlerNotFoundError
# =============================================================================


class TestHandlerNotFoundError:
    """Test HandlerNotFoundError (404 Not Found)."""

    def test_basic_instantiation(self):
        err = HandlerNotFoundError("Debate", "abc-123")
        assert err.status_code == 404
        assert "Debate" in str(err)
        assert "abc-123" in str(err)

    def test_resource_type_attribute(self):
        err = HandlerNotFoundError("User", "user-1")
        assert err.resource_type == "User"

    def test_resource_id_attribute(self):
        err = HandlerNotFoundError("User", "user-1")
        assert err.resource_id == "user-1"

    def test_details_contain_resource_info(self):
        err = HandlerNotFoundError("Template", "tmpl-99")
        assert err.details["resource_type"] == "Template"
        assert err.details["resource_id"] == "tmpl-99"

    def test_message_format(self):
        err = HandlerNotFoundError("Agent", "agent-x")
        assert (
            str(err)
            == "Agent not found: agent-x (details: {'resource_type': 'Agent', 'resource_id': 'agent-x'})"
        )

    def test_inherits_from_handler_error(self):
        err = HandlerNotFoundError("X", "Y")
        assert isinstance(err, HandlerError)

    def test_special_characters_in_id(self):
        err = HandlerNotFoundError("File", "../../../etc/passwd")
        assert err.resource_id == "../../../etc/passwd"

    def test_empty_strings(self):
        err = HandlerNotFoundError("", "")
        assert err.resource_type == ""
        assert err.resource_id == ""


# =============================================================================
# HandlerAuthorizationError
# =============================================================================


class TestHandlerAuthorizationError:
    """Test HandlerAuthorizationError (403 Forbidden)."""

    def test_basic_instantiation(self):
        err = HandlerAuthorizationError("delete")
        assert err.status_code == 403
        assert "Not authorized to delete" in str(err)

    def test_with_resource(self):
        err = HandlerAuthorizationError("edit", resource="debate-123")
        assert "on debate-123" in str(err)

    def test_without_resource(self):
        err = HandlerAuthorizationError("create")
        # The main message part (before details) should not contain "on <resource>"
        assert err.message == "Not authorized to create"
        assert err.resource is None

    def test_action_attribute(self):
        err = HandlerAuthorizationError("admin_access")
        assert err.action == "admin_access"

    def test_resource_attribute(self):
        err = HandlerAuthorizationError("view", resource="secrets")
        assert err.resource == "secrets"

    def test_details(self):
        err = HandlerAuthorizationError("modify", resource="config")
        assert err.details["action"] == "modify"
        assert err.details["resource"] == "config"

    def test_inherits_from_handler_error(self):
        err = HandlerAuthorizationError("test")
        assert isinstance(err, HandlerError)

    def test_resource_none_in_details(self):
        err = HandlerAuthorizationError("act")
        assert err.details["resource"] is None


# =============================================================================
# HandlerConflictError
# =============================================================================


class TestHandlerConflictError:
    """Test HandlerConflictError (409 Conflict)."""

    def test_basic_instantiation(self):
        err = HandlerConflictError("Resource already exists")
        assert err.status_code == 409
        assert "already exists" in str(err)

    def test_with_resource_type(self):
        err = HandlerConflictError("Duplicate key", resource_type="User")
        assert err.details["resource_type"] == "User"

    def test_without_resource_type(self):
        err = HandlerConflictError("Conflict")
        assert err.details["resource_type"] is None

    def test_inherits_from_handler_error(self):
        err = HandlerConflictError("test")
        assert isinstance(err, HandlerError)


# =============================================================================
# HandlerRateLimitError
# =============================================================================


class TestHandlerRateLimitError:
    """Test HandlerRateLimitError (429 Too Many Requests)."""

    def test_default_message(self):
        err = HandlerRateLimitError()
        assert err.status_code == 429
        assert "Rate limit exceeded" in str(err)

    def test_custom_message(self):
        err = HandlerRateLimitError("Too fast")
        assert "Too fast" in str(err)

    def test_retry_after(self):
        err = HandlerRateLimitError(retry_after=60)
        assert err.retry_after == 60
        assert err.details["retry_after"] == 60

    def test_retry_after_none(self):
        err = HandlerRateLimitError()
        assert err.retry_after is None
        assert "retry_after" not in err.details

    def test_retry_after_zero_is_falsy(self):
        """Zero retry_after is falsy, so it should NOT be in details."""
        err = HandlerRateLimitError(retry_after=0)
        assert err.retry_after == 0
        # 0 is falsy, so details won't include it
        assert "retry_after" not in err.details

    def test_inherits_from_handler_error(self):
        err = HandlerRateLimitError()
        assert isinstance(err, HandlerError)


# =============================================================================
# HandlerExternalServiceError
# =============================================================================


class TestHandlerExternalServiceError:
    """Test HandlerExternalServiceError (502/503)."""

    def test_basic_instantiation(self):
        err = HandlerExternalServiceError("OpenAI", "timeout")
        assert err.status_code == 502
        assert "OpenAI" in str(err)
        assert "timeout" in str(err)

    def test_unavailable_sets_503(self):
        err = HandlerExternalServiceError("Redis", "down", unavailable=True)
        assert err.status_code == 503

    def test_available_keeps_502(self):
        err = HandlerExternalServiceError("API", "error", unavailable=False)
        assert err.status_code == 502

    def test_service_attribute(self):
        err = HandlerExternalServiceError("Stripe", "payment failed")
        assert err.service == "Stripe"

    def test_details(self):
        err = HandlerExternalServiceError("Slack", "webhook failed", unavailable=True)
        assert err.details["service"] == "Slack"
        assert err.details["unavailable"] is True

    def test_message_format(self):
        err = HandlerExternalServiceError("GitHub", "rate limited")
        assert "GitHub service error: rate limited" in str(err)

    def test_inherits_from_handler_error(self):
        err = HandlerExternalServiceError("X", "Y")
        assert isinstance(err, HandlerError)


# =============================================================================
# HandlerDatabaseError
# =============================================================================


class TestHandlerDatabaseError:
    """Test HandlerDatabaseError (500 Internal Server Error)."""

    def test_basic_instantiation(self):
        err = HandlerDatabaseError("insert")
        assert err.status_code == 500
        assert "Database error during insert" in str(err)

    def test_with_message(self):
        err = HandlerDatabaseError("update", message="constraint violation")
        assert "constraint violation" in str(err)

    def test_without_message(self):
        err = HandlerDatabaseError("delete")
        assert "Database error during delete" in str(err)
        assert ":" not in str(err).split("Database error during delete")[1].split("(")[0]

    def test_operation_attribute(self):
        err = HandlerDatabaseError("query")
        assert err.operation == "query"

    def test_details(self):
        err = HandlerDatabaseError("migrate")
        assert err.details["operation"] == "migrate"

    def test_inherits_from_handler_error(self):
        err = HandlerDatabaseError("test")
        assert isinstance(err, HandlerError)


# =============================================================================
# HandlerJSONParseError
# =============================================================================


class TestHandlerJSONParseError:
    """Test HandlerJSONParseError (400 Bad Request)."""

    def test_basic_instantiation(self):
        err = HandlerJSONParseError()
        assert err.status_code == 400
        assert "Invalid JSON in request body" in str(err)

    def test_custom_source(self):
        err = HandlerJSONParseError(source="query parameter")
        assert "query parameter" in str(err)

    def test_with_reason(self):
        err = HandlerJSONParseError(reason="unexpected token")
        assert "unexpected token" in str(err)

    def test_source_attribute(self):
        err = HandlerJSONParseError(source="headers")
        assert err.source == "headers"

    def test_reason_attribute(self):
        err = HandlerJSONParseError(reason="EOF")
        assert err.reason == "EOF"

    def test_details(self):
        err = HandlerJSONParseError(source="body", reason="syntax error")
        assert err.details["source"] == "body"
        assert err.details["reason"] == "syntax error"

    def test_inherits_from_handler_validation_error(self):
        err = HandlerJSONParseError()
        assert isinstance(err, HandlerValidationError)
        assert isinstance(err, HandlerError)

    def test_field_is_set_to_source(self):
        err = HandlerJSONParseError(source="payload")
        assert err.field == "payload"

    def test_default_source_is_request_body(self):
        err = HandlerJSONParseError()
        assert err.source == "request body"

    def test_reason_none(self):
        err = HandlerJSONParseError()
        assert err.reason is None


# =============================================================================
# HandlerTimeoutError
# =============================================================================


class TestHandlerTimeoutError:
    """Test HandlerTimeoutError (504 Gateway Timeout)."""

    def test_basic_instantiation(self):
        err = HandlerTimeoutError("database query")
        assert err.status_code == 504
        assert "database query" in str(err)

    def test_with_timeout_seconds(self):
        err = HandlerTimeoutError("api call", timeout_seconds=30.0)
        assert "30.0s" in str(err)
        assert err.timeout_seconds == 30.0

    def test_without_timeout_seconds(self):
        err = HandlerTimeoutError("search")
        assert err.timeout_seconds is None
        assert "timeout:" not in str(err)

    def test_operation_attribute(self):
        err = HandlerTimeoutError("index rebuild")
        assert err.operation == "index rebuild"

    def test_details(self):
        err = HandlerTimeoutError("sync", timeout_seconds=5.5)
        assert err.details["operation"] == "sync"
        assert err.details["timeout_seconds"] == 5.5

    def test_zero_timeout_is_falsy(self):
        """Zero timeout is falsy, so message won't include it."""
        err = HandlerTimeoutError("op", timeout_seconds=0.0)
        assert err.timeout_seconds == 0.0
        # 0.0 is falsy
        assert "timeout:" not in str(err).replace("timed out", "")

    def test_inherits_from_handler_error(self):
        err = HandlerTimeoutError("test")
        assert isinstance(err, HandlerError)


# =============================================================================
# HandlerStreamError
# =============================================================================


class TestHandlerStreamError:
    """Test HandlerStreamError (500 or custom code)."""

    def test_basic_instantiation(self):
        err = HandlerStreamError("connection lost")
        assert err.status_code == 500
        assert "connection lost" in str(err)

    def test_default_stream_type(self):
        err = HandlerStreamError("error")
        assert err.stream_type == "websocket"
        assert "websocket" in str(err)

    def test_custom_stream_type(self):
        err = HandlerStreamError("error", stream_type="sse")
        assert err.stream_type == "sse"
        assert "sse" in str(err)

    def test_custom_code(self):
        err = HandlerStreamError("closed", code=1006)
        assert err.status_code == 1006

    def test_code_none_keeps_default(self):
        err = HandlerStreamError("error", code=None)
        assert err.status_code == 500

    def test_code_zero_is_falsy(self):
        """Zero code is falsy, so status_code stays at default 500."""
        err = HandlerStreamError("error", code=0)
        assert err.status_code == 500

    def test_reason_attribute(self):
        err = HandlerStreamError("timeout")
        assert err.reason == "timeout"

    def test_details(self):
        err = HandlerStreamError("closed", stream_type="grpc")
        assert err.details["stream_type"] == "grpc"
        assert err.details["reason"] == "closed"

    def test_message_format(self):
        err = HandlerStreamError("reset", stream_type="http2")
        assert "Stream error (http2): reset" in str(err)

    def test_inherits_from_handler_error(self):
        err = HandlerStreamError("test")
        assert isinstance(err, HandlerError)


# =============================================================================
# HandlerOAuthError
# =============================================================================


class TestHandlerOAuthError:
    """Test HandlerOAuthError (400 Bad Request by default)."""

    def test_basic_instantiation(self):
        err = HandlerOAuthError("google", "invalid_grant")
        assert err.status_code == 400
        assert "google" in str(err)
        assert "invalid_grant" in str(err)

    def test_provider_attribute(self):
        err = HandlerOAuthError("github", "error")
        assert err.provider == "github"

    def test_reason_attribute(self):
        err = HandlerOAuthError("google", "token expired")
        assert err.reason == "token expired"

    def test_oauth_error(self):
        err = HandlerOAuthError("microsoft", "auth failed", oauth_error="invalid_client")
        assert err.oauth_error == "invalid_client"
        assert err.details["oauth_error"] == "invalid_client"

    def test_oauth_error_none_default(self):
        err = HandlerOAuthError("slack", "error")
        assert err.oauth_error is None

    def test_recoverable_default_true(self):
        err = HandlerOAuthError("google", "temp error")
        assert err.recoverable is True
        assert err.details["recoverable"] is True

    def test_recoverable_false(self):
        err = HandlerOAuthError("google", "revoked", recoverable=False)
        assert err.recoverable is False
        assert err.details["recoverable"] is False

    def test_details(self):
        err = HandlerOAuthError("x", "y", oauth_error="z", recoverable=False)
        assert err.details["provider"] == "x"
        assert err.details["reason"] == "y"
        assert err.details["oauth_error"] == "z"
        assert err.details["recoverable"] is False

    def test_message_format(self):
        err = HandlerOAuthError("Azure", "token invalid")
        assert "OAuth error with Azure: token invalid" in str(err)

    def test_inherits_from_handler_error(self):
        err = HandlerOAuthError("p", "r")
        assert isinstance(err, HandlerError)


# =============================================================================
# EXCEPTION_MAP
# =============================================================================


class TestExceptionMap:
    """Test the EXCEPTION_MAP configuration."""

    def test_handler_validation_error_mapping(self):
        status, level, include = EXCEPTION_MAP[HandlerValidationError]
        assert status == 400
        assert level == "info"
        assert include is True

    def test_input_validation_error_mapping(self):
        status, level, include = EXCEPTION_MAP[InputValidationError]
        assert status == 400
        assert level == "info"
        assert include is True

    def test_validation_error_mapping(self):
        status, level, include = EXCEPTION_MAP[ValidationError]
        assert status == 400
        assert level == "info"
        assert include is True

    def test_value_error_mapping(self):
        status, level, include = EXCEPTION_MAP[ValueError]
        assert status == 400
        assert level == "info"
        assert include is True

    def test_key_error_no_message_exposure(self):
        """KeyError should NOT expose internal key names."""
        status, level, include = EXCEPTION_MAP[KeyError]
        assert status == 400
        assert include is False

    def test_type_error_mapping(self):
        status, level, include = EXCEPTION_MAP[TypeError]
        assert status == 400
        assert level == "warning"
        assert include is False

    def test_handler_not_found_mapping(self):
        status, level, include = EXCEPTION_MAP[HandlerNotFoundError]
        assert status == 404
        assert include is True

    def test_record_not_found_mapping(self):
        status, level, include = EXCEPTION_MAP[RecordNotFoundError]
        assert status == 404
        assert include is True

    def test_handler_authorization_mapping(self):
        status, level, include = EXCEPTION_MAP[HandlerAuthorizationError]
        assert status == 403
        assert include is True

    def test_authorization_error_mapping(self):
        status, level, include = EXCEPTION_MAP[AuthorizationError]
        assert status == 403
        assert include is True

    def test_authentication_error_mapping(self):
        status, level, include = EXCEPTION_MAP[AuthenticationError]
        assert status == 401
        assert include is True

    def test_auth_error_mapping(self):
        status, level, include = EXCEPTION_MAP[AuthError]
        assert status == 401
        assert include is True

    def test_handler_rate_limit_mapping(self):
        status, level, include = EXCEPTION_MAP[HandlerRateLimitError]
        assert status == 429
        assert include is True

    def test_rate_limit_exceeded_mapping(self):
        status, level, include = EXCEPTION_MAP[RateLimitExceededError]
        assert status == 429
        assert include is True

    def test_handler_conflict_mapping(self):
        status, level, include = EXCEPTION_MAP[HandlerConflictError]
        assert status == 409
        assert include is True

    def test_handler_external_service_mapping(self):
        status, level, include = EXCEPTION_MAP[HandlerExternalServiceError]
        assert status == 502
        assert level == "error"
        assert include is True

    def test_handler_database_error_no_message(self):
        """Database errors should NOT expose internal details."""
        status, level, include = EXCEPTION_MAP[HandlerDatabaseError]
        assert status == 500
        assert include is False

    def test_database_error_no_message(self):
        status, level, include = EXCEPTION_MAP[DatabaseError]
        assert status == 500
        assert include is False

    def test_sqlite_error_no_message(self):
        status, level, include = EXCEPTION_MAP[sqlite3.Error]
        assert status == 500
        assert include is False

    def test_sqlite_integrity_conflict(self):
        status, level, include = EXCEPTION_MAP[sqlite3.IntegrityError]
        assert status == 409
        assert level == "warning"
        assert include is False

    def test_timeout_error_mapping(self):
        status, level, include = EXCEPTION_MAP[TimeoutError]
        assert status == 504
        assert level == "warning"
        assert include is False

    def test_handler_timeout_mapping(self):
        status, level, include = EXCEPTION_MAP[HandlerTimeoutError]
        assert status == 504
        assert include is True

    def test_handler_json_parse_mapping(self):
        status, level, include = EXCEPTION_MAP[HandlerJSONParseError]
        assert status == 400
        assert include is True

    def test_handler_stream_mapping(self):
        status, level, include = EXCEPTION_MAP[HandlerStreamError]
        assert status == 500
        assert level == "warning"
        assert include is True

    def test_handler_oauth_mapping(self):
        status, level, include = EXCEPTION_MAP[HandlerOAuthError]
        assert status == 400
        assert include is True

    def test_all_expected_types_present(self):
        """Verify all expected exception types are in the map."""
        expected_types = {
            HandlerValidationError,
            InputValidationError,
            ValidationError,
            ValueError,
            KeyError,
            TypeError,
            HandlerNotFoundError,
            RecordNotFoundError,
            HandlerAuthorizationError,
            AuthorizationError,
            AuthenticationError,
            AuthError,
            HandlerRateLimitError,
            RateLimitExceededError,
            HandlerConflictError,
            HandlerExternalServiceError,
            HandlerDatabaseError,
            DatabaseError,
            sqlite3.Error,
            sqlite3.IntegrityError,
            TimeoutError,
            HandlerTimeoutError,
            HandlerJSONParseError,
            HandlerStreamError,
            HandlerOAuthError,
        }
        assert expected_types.issubset(set(EXCEPTION_MAP.keys()))


# =============================================================================
# GENERIC_ERROR_MESSAGES
# =============================================================================


class TestGenericErrorMessages:
    """Test the GENERIC_ERROR_MESSAGES dict."""

    def test_400_message(self):
        assert GENERIC_ERROR_MESSAGES[400] == "Invalid request"

    def test_401_message(self):
        assert GENERIC_ERROR_MESSAGES[401] == "Authentication required"

    def test_403_message(self):
        assert GENERIC_ERROR_MESSAGES[403] == "Access denied"

    def test_404_message(self):
        assert GENERIC_ERROR_MESSAGES[404] == "Resource not found"

    def test_409_message(self):
        assert GENERIC_ERROR_MESSAGES[409] == "Resource conflict"

    def test_429_message(self):
        assert GENERIC_ERROR_MESSAGES[429] == "Too many requests"

    def test_500_message(self):
        assert GENERIC_ERROR_MESSAGES[500] == "Internal server error"

    def test_502_message(self):
        assert GENERIC_ERROR_MESSAGES[502] == "Service temporarily unavailable"

    def test_503_message(self):
        assert GENERIC_ERROR_MESSAGES[503] == "Service unavailable"

    def test_504_message(self):
        assert GENERIC_ERROR_MESSAGES[504] == "Request timeout"

    def test_all_status_codes_present(self):
        expected_codes = {400, 401, 403, 404, 409, 429, 500, 502, 503, 504}
        assert expected_codes == set(GENERIC_ERROR_MESSAGES.keys())


# =============================================================================
# classify_exception()
# =============================================================================


class TestClassifyException:
    """Test classify_exception() function."""

    # --- Exact match tests ---

    def test_handler_validation_error(self):
        exc = HandlerValidationError("bad input")
        status, level, msg = classify_exception(exc)
        assert status == 400
        assert level == "info"
        assert "bad input" in msg

    def test_value_error_includes_message(self):
        exc = ValueError("wrong value")
        status, level, msg = classify_exception(exc)
        assert status == 400
        assert "wrong value" in msg

    def test_key_error_hides_message(self):
        """KeyError should return generic message, not key name."""
        exc = KeyError("secret_internal_key")
        status, level, msg = classify_exception(exc)
        assert status == 400
        assert "secret_internal_key" not in msg
        assert msg == "Invalid request"

    def test_type_error_hides_message(self):
        exc = TypeError("unexpected argument")
        status, level, msg = classify_exception(exc)
        assert status == 400
        assert "unexpected argument" not in msg
        assert msg == "Invalid request"

    def test_handler_not_found(self):
        exc = HandlerNotFoundError("Debate", "abc")
        status, level, msg = classify_exception(exc)
        assert status == 404
        assert "Debate" in msg

    def test_handler_authorization(self):
        exc = HandlerAuthorizationError("delete", resource="debate-1")
        status, level, msg = classify_exception(exc)
        assert status == 403
        assert "Not authorized" in msg

    def test_authentication_error(self):
        exc = AuthenticationError("invalid token")
        status, level, msg = classify_exception(exc)
        assert status == 401

    def test_auth_error(self):
        exc = AuthError("auth failed")
        status, level, msg = classify_exception(exc)
        assert status == 401

    def test_handler_rate_limit(self):
        exc = HandlerRateLimitError()
        status, level, msg = classify_exception(exc)
        assert status == 429
        assert "Rate limit" in msg

    def test_handler_conflict(self):
        exc = HandlerConflictError("already exists")
        status, level, msg = classify_exception(exc)
        assert status == 409
        assert "already exists" in msg

    def test_handler_external_service(self):
        exc = HandlerExternalServiceError("API", "down")
        status, level, msg = classify_exception(exc)
        assert status == 502
        assert level == "error"

    def test_handler_database_hides_message(self):
        exc = HandlerDatabaseError("query", "table not found")
        status, level, msg = classify_exception(exc)
        assert status == 500
        assert "table not found" not in msg
        assert msg == "Internal server error"

    def test_sqlite_error(self):
        exc = sqlite3.Error("disk full")
        status, level, msg = classify_exception(exc)
        assert status == 500
        assert "disk full" not in msg

    def test_sqlite_integrity_error(self):
        exc = sqlite3.IntegrityError("UNIQUE constraint failed")
        status, level, msg = classify_exception(exc)
        assert status == 409
        assert "UNIQUE" not in msg
        assert msg == "Resource conflict"

    def test_timeout_error(self):
        exc = TimeoutError("timed out")
        status, level, msg = classify_exception(exc)
        assert status == 504
        assert "timed out" not in msg
        assert msg == "Request timeout"

    def test_handler_timeout(self):
        exc = HandlerTimeoutError("search")
        status, level, msg = classify_exception(exc)
        assert status == 504
        assert "search" in msg

    def test_handler_json_parse(self):
        exc = HandlerJSONParseError(reason="unterminated string")
        status, level, msg = classify_exception(exc)
        assert status == 400
        assert "unterminated string" in msg

    def test_handler_stream(self):
        exc = HandlerStreamError("closed unexpectedly")
        status, level, msg = classify_exception(exc)
        assert status == 500
        assert "closed unexpectedly" in msg

    def test_handler_oauth(self):
        exc = HandlerOAuthError("google", "invalid_grant")
        status, level, msg = classify_exception(exc)
        assert status == 400
        assert "google" in msg

    # --- Subclass match tests ---

    def test_subclass_of_handler_error(self):
        """Custom HandlerError subclass not in map should match via isinstance."""

        class CustomHandlerError(HandlerError):
            status_code = 422

        exc = CustomHandlerError("unprocessable")
        status, level, msg = classify_exception(exc)
        # Falls through exact match, then subclass match finds HandlerError ancestor
        # or falls to the isinstance(exc, HandlerError) check
        assert status == 422

    def test_subclass_of_validation_error(self):
        """Custom ValidationError subclass should match via isinstance."""

        class CustomValidation(ValidationError):
            pass

        exc = CustomValidation("bad data")
        status, level, msg = classify_exception(exc)
        assert status == 400
        assert "bad data" in msg

    # --- Fallback tests ---

    def test_aragora_error_fallback(self):
        """AragoraError not in map should return 500 with generic message."""
        exc = AragoraError("internal issue")
        status, level, msg = classify_exception(exc)
        assert status == 500
        assert msg == "Internal server error"

    def test_unknown_exception_fallback(self):
        """Completely unknown exception should return 500."""
        exc = RuntimeError("unexpected")
        status, level, msg = classify_exception(exc)
        assert status == 500
        assert level == "error"
        assert msg == "Internal server error"

    def test_base_exception_fallback(self):
        """Non-Exception subclass of BaseException (like SystemExit) if wrapped."""
        exc = Exception("generic")
        status, level, msg = classify_exception(exc)
        assert status == 500
        assert msg == "Internal server error"

    def test_database_error_hides_message(self):
        exc = DatabaseError("secret SQL")
        status, level, msg = classify_exception(exc)
        assert status == 500
        assert "secret SQL" not in msg

    def test_record_not_found(self):
        exc = RecordNotFoundError("users", "42")
        status, level, msg = classify_exception(exc)
        assert status == 404

    def test_authorization_error(self):
        exc = AuthorizationError("no permission")
        status, level, msg = classify_exception(exc)
        assert status == 403

    def test_rate_limit_exceeded(self):
        exc = RateLimitExceededError(100, 60)
        status, level, msg = classify_exception(exc)
        assert status == 429

    def test_input_validation_error(self):
        exc = InputValidationError("email", "invalid format")
        status, level, msg = classify_exception(exc)
        assert status == 400
        assert "email" in msg


# =============================================================================
# handle_handler_error()
# =============================================================================


class TestHandleHandlerError:
    """Test handle_handler_error() function."""

    def setup_method(self):
        self.logger = MagicMock(spec=logging.Logger)

    def test_returns_status_and_message(self):
        exc = HandlerValidationError("bad input")
        status, msg = handle_handler_error(exc, "create_debate", self.logger)
        assert status == 400
        assert "bad input" in msg

    def test_logs_at_info_level_for_validation(self):
        exc = HandlerValidationError("bad")
        handle_handler_error(exc, "operation", self.logger)
        self.logger.info.assert_called_once()

    def test_logs_at_error_level_for_database(self):
        exc = HandlerDatabaseError("query")
        handle_handler_error(exc, "operation", self.logger)
        self.logger.error.assert_called_once()

    def test_error_level_includes_exc_info(self):
        exc = HandlerDatabaseError("insert")
        handle_handler_error(exc, "save_record", self.logger)
        call_kwargs = self.logger.error.call_args
        assert call_kwargs[1].get("exc_info") is True

    def test_info_level_no_exc_info(self):
        exc = HandlerValidationError("bad")
        handle_handler_error(exc, "op", self.logger)
        call_kwargs = self.logger.info.call_args
        # info-level without include_traceback should not have exc_info=True
        assert call_kwargs[1].get("exc_info") is not True

    def test_include_traceback_forces_exc_info(self):
        exc = HandlerValidationError("bad")
        handle_handler_error(exc, "op", self.logger, include_traceback=True)
        call_kwargs = self.logger.info.call_args
        assert call_kwargs[1].get("exc_info") is True

    def test_log_message_contains_operation(self):
        exc = ValueError("wrong")
        handle_handler_error(exc, "process_request", self.logger)
        log_msg = self.logger.info.call_args[0][0]
        assert "process_request" in log_msg

    def test_log_message_contains_exception_type(self):
        exc = TypeError("mismatch")
        handle_handler_error(exc, "op", self.logger)
        log_msg = self.logger.warning.call_args[0][0]
        assert "TypeError" in log_msg

    def test_log_message_contains_exception_str(self):
        exc = ValueError("specific issue")
        handle_handler_error(exc, "op", self.logger)
        log_msg = self.logger.info.call_args[0][0]
        assert "specific issue" in log_msg

    def test_unknown_exception_logged_as_error(self):
        exc = RuntimeError("unexpected crash")
        handle_handler_error(exc, "op", self.logger)
        self.logger.error.assert_called_once()

    def test_unknown_exception_returns_500(self):
        exc = RuntimeError("crash")
        status, msg = handle_handler_error(exc, "op", self.logger)
        assert status == 500
        assert msg == "Internal server error"

    def test_handler_not_found_returns_404(self):
        exc = HandlerNotFoundError("Debate", "d1")
        status, msg = handle_handler_error(exc, "get_debate", self.logger)
        assert status == 404

    def test_handler_authorization_returns_403(self):
        exc = HandlerAuthorizationError("delete")
        status, msg = handle_handler_error(exc, "delete_debate", self.logger)
        assert status == 403

    def test_handler_rate_limit_returns_429(self):
        exc = HandlerRateLimitError()
        status, msg = handle_handler_error(exc, "api_call", self.logger)
        assert status == 429

    def test_handler_external_service_returns_502(self):
        exc = HandlerExternalServiceError("OpenAI", "timeout")
        status, msg = handle_handler_error(exc, "generate", self.logger)
        assert status == 502

    def test_handler_timeout_returns_504(self):
        exc = HandlerTimeoutError("search")
        status, msg = handle_handler_error(exc, "search", self.logger)
        assert status == 504

    def test_sqlite_integrity_returns_409(self):
        exc = sqlite3.IntegrityError("duplicate")
        status, msg = handle_handler_error(exc, "insert", self.logger)
        assert status == 409


# =============================================================================
# is_client_error()
# =============================================================================


class TestIsClientError:
    """Test is_client_error() utility."""

    def test_validation_error_is_client(self):
        assert is_client_error(HandlerValidationError("bad")) is True

    def test_not_found_is_client(self):
        assert is_client_error(HandlerNotFoundError("X", "1")) is True

    def test_authorization_is_client(self):
        assert is_client_error(HandlerAuthorizationError("act")) is True

    def test_conflict_is_client(self):
        assert is_client_error(HandlerConflictError("dup")) is True

    def test_rate_limit_is_client(self):
        assert is_client_error(HandlerRateLimitError()) is True

    def test_database_error_not_client(self):
        assert is_client_error(HandlerDatabaseError("op")) is False

    def test_external_service_not_client(self):
        assert is_client_error(HandlerExternalServiceError("s", "m")) is False

    def test_timeout_not_client(self):
        assert is_client_error(HandlerTimeoutError("op")) is False

    def test_value_error_is_client(self):
        assert is_client_error(ValueError("wrong")) is True

    def test_key_error_is_client(self):
        assert is_client_error(KeyError("missing")) is True

    def test_type_error_is_client(self):
        assert is_client_error(TypeError("mismatch")) is True

    def test_unknown_error_not_client(self):
        assert is_client_error(RuntimeError("crash")) is False

    def test_authentication_is_client(self):
        assert is_client_error(AuthenticationError("invalid")) is True

    def test_oauth_is_client(self):
        assert is_client_error(HandlerOAuthError("g", "r")) is True

    def test_json_parse_is_client(self):
        assert is_client_error(HandlerJSONParseError()) is True


# =============================================================================
# is_server_error()
# =============================================================================


class TestIsServerError:
    """Test is_server_error() utility."""

    def test_database_error_is_server(self):
        assert is_server_error(HandlerDatabaseError("op")) is True

    def test_external_service_is_server(self):
        assert is_server_error(HandlerExternalServiceError("s", "m")) is True

    def test_timeout_is_server(self):
        assert is_server_error(HandlerTimeoutError("op")) is True

    def test_stream_error_is_server(self):
        assert is_server_error(HandlerStreamError("err")) is True

    def test_unknown_error_is_server(self):
        assert is_server_error(RuntimeError("crash")) is True

    def test_validation_not_server(self):
        assert is_server_error(HandlerValidationError("bad")) is False

    def test_not_found_not_server(self):
        assert is_server_error(HandlerNotFoundError("X", "1")) is False

    def test_authorization_not_server(self):
        assert is_server_error(HandlerAuthorizationError("act")) is False

    def test_rate_limit_not_server(self):
        assert is_server_error(HandlerRateLimitError()) is False

    def test_conflict_not_server(self):
        assert is_server_error(HandlerConflictError("dup")) is False


# =============================================================================
# is_retryable_error()
# =============================================================================


class TestIsRetryableError:
    """Test is_retryable_error() utility."""

    def test_rate_limit_is_retryable(self):
        assert is_retryable_error(HandlerRateLimitError()) is True

    def test_external_service_502_is_retryable(self):
        assert is_retryable_error(HandlerExternalServiceError("s", "m")) is True

    def test_external_service_503_is_retryable(self):
        exc = HandlerExternalServiceError("s", "m", unavailable=True)
        assert is_retryable_error(exc) is True

    def test_timeout_is_retryable(self):
        assert is_retryable_error(HandlerTimeoutError("op")) is True

    def test_builtin_timeout_is_retryable(self):
        assert is_retryable_error(TimeoutError()) is True

    def test_validation_not_retryable(self):
        assert is_retryable_error(HandlerValidationError("bad")) is False

    def test_not_found_not_retryable(self):
        assert is_retryable_error(HandlerNotFoundError("X", "1")) is False

    def test_authorization_not_retryable(self):
        assert is_retryable_error(HandlerAuthorizationError("act")) is False

    def test_database_not_retryable(self):
        assert is_retryable_error(HandlerDatabaseError("op")) is False

    def test_conflict_not_retryable(self):
        assert is_retryable_error(HandlerConflictError("dup")) is False

    def test_unknown_error_not_retryable(self):
        assert is_retryable_error(RuntimeError("crash")) is False

    def test_rate_limit_exceeded_is_retryable(self):
        assert is_retryable_error(RateLimitExceededError(100, 60)) is True


# =============================================================================
# Inheritance Hierarchy
# =============================================================================


class TestInheritanceHierarchy:
    """Test the exception inheritance chain."""

    def test_handler_error_inherits_aragora_error(self):
        assert issubclass(HandlerError, AragoraError)

    def test_handler_validation_inherits_handler_error(self):
        assert issubclass(HandlerValidationError, HandlerError)

    def test_handler_not_found_inherits_handler_error(self):
        assert issubclass(HandlerNotFoundError, HandlerError)

    def test_handler_authorization_inherits_handler_error(self):
        assert issubclass(HandlerAuthorizationError, HandlerError)

    def test_handler_conflict_inherits_handler_error(self):
        assert issubclass(HandlerConflictError, HandlerError)

    def test_handler_rate_limit_inherits_handler_error(self):
        assert issubclass(HandlerRateLimitError, HandlerError)

    def test_handler_external_service_inherits_handler_error(self):
        assert issubclass(HandlerExternalServiceError, HandlerError)

    def test_handler_database_inherits_handler_error(self):
        assert issubclass(HandlerDatabaseError, HandlerError)

    def test_handler_json_parse_inherits_handler_validation(self):
        assert issubclass(HandlerJSONParseError, HandlerValidationError)

    def test_handler_json_parse_also_inherits_handler_error(self):
        assert issubclass(HandlerJSONParseError, HandlerError)

    def test_handler_timeout_inherits_handler_error(self):
        assert issubclass(HandlerTimeoutError, HandlerError)

    def test_handler_stream_inherits_handler_error(self):
        assert issubclass(HandlerStreamError, HandlerError)

    def test_handler_oauth_inherits_handler_error(self):
        assert issubclass(HandlerOAuthError, HandlerError)

    def test_all_handler_exceptions_are_exceptions(self):
        for cls in [
            HandlerError,
            HandlerValidationError,
            HandlerNotFoundError,
            HandlerAuthorizationError,
            HandlerConflictError,
            HandlerRateLimitError,
            HandlerExternalServiceError,
            HandlerDatabaseError,
            HandlerJSONParseError,
            HandlerTimeoutError,
            HandlerStreamError,
            HandlerOAuthError,
        ]:
            assert issubclass(cls, Exception), f"{cls.__name__} is not an Exception"


# =============================================================================
# Edge Cases and Security
# =============================================================================


class TestEdgeCases:
    """Test edge cases, boundary conditions, and security concerns."""

    def test_very_long_message(self):
        long_msg = "x" * 10000
        err = HandlerError(long_msg)
        assert len(str(err)) >= 10000

    def test_unicode_in_message(self):
        err = HandlerError("Error: \u2603 \u00e9\u00e8\u00ea \u4e16\u754c")
        assert "\u2603" in str(err)

    def test_newlines_in_message(self):
        err = HandlerError("line1\nline2\nline3")
        assert "line1\nline2\nline3" in str(err)

    def test_sql_injection_in_message(self):
        """SQL injection attempts in error messages should be passed through as-is (no execution)."""
        err = HandlerValidationError("'; DROP TABLE users; --")
        assert "DROP TABLE" in str(err)

    def test_xss_in_message(self):
        """XSS attempts in error messages should be passed through as-is."""
        err = HandlerValidationError("<script>alert('xss')</script>")
        assert "<script>" in str(err)

    def test_path_traversal_in_resource_id(self):
        err = HandlerNotFoundError("File", "../../../etc/passwd")
        assert err.resource_id == "../../../etc/passwd"

    def test_none_details_coerced_to_empty_dict(self):
        err = HandlerError("test", details=None)
        assert err.details == {}

    def test_handler_error_can_be_caught_as_exception(self):
        with pytest.raises(Exception):
            raise HandlerError("test")

    def test_handler_error_can_be_caught_as_aragora_error(self):
        with pytest.raises(AragoraError):
            raise HandlerError("test")

    def test_handler_validation_can_be_caught_as_handler_error(self):
        with pytest.raises(HandlerError):
            raise HandlerValidationError("test")

    def test_classify_with_none_message(self):
        """Exceptions with unusual string representations."""
        exc = ValueError()
        status, level, msg = classify_exception(exc)
        assert status == 400

    def test_handler_error_with_zero_status_code(self):
        err = HandlerError("test", status_code=0)
        assert err.status_code == 0

    def test_handler_error_with_negative_status_code(self):
        err = HandlerError("test", status_code=-1)
        assert err.status_code == -1

    def test_classify_exception_with_custom_subclass_not_in_map(self):
        """A custom AragoraError subclass not in the map."""

        class CustomAragora(AragoraError):
            pass

        exc = CustomAragora("custom error")
        status, level, msg = classify_exception(exc)
        assert status == 500
        assert msg == "Internal server error"

    def test_handle_handler_error_with_empty_operation(self):
        logger = MagicMock(spec=logging.Logger)
        exc = HandlerValidationError("bad")
        status, msg = handle_handler_error(exc, "", logger)
        assert status == 400
        log_msg = logger.info.call_args[0][0]
        assert "failed:" in log_msg

    def test_generic_message_fallback_for_unknown_status(self):
        """When status code has no generic message, 'Error' is returned."""

        class WeirdHandler(HandlerError):
            status_code = 999

        exc = WeirdHandler("weird")
        status, level, msg = classify_exception(exc)
        assert status == 999
        assert msg == "Error"


# =============================================================================
# Module __all__ Exports
# =============================================================================


class TestModuleExports:
    """Test that __all__ exports match expected symbols."""

    def test_handler_exceptions_exported(self):
        from aragora.server.handlers import exceptions as mod

        expected = [
            "HandlerError",
            "HandlerValidationError",
            "HandlerNotFoundError",
            "HandlerAuthorizationError",
            "HandlerConflictError",
            "HandlerRateLimitError",
            "HandlerExternalServiceError",
            "HandlerDatabaseError",
            "HandlerJSONParseError",
            "HandlerTimeoutError",
            "HandlerStreamError",
            "HandlerOAuthError",
        ]
        for name in expected:
            assert name in mod.__all__, f"{name} not in __all__"

    def test_utility_functions_exported(self):
        from aragora.server.handlers import exceptions as mod

        for name in [
            "classify_exception",
            "handle_handler_error",
            "is_client_error",
            "is_server_error",
            "is_retryable_error",
        ]:
            assert name in mod.__all__, f"{name} not in __all__"

    def test_reexported_base_exceptions(self):
        from aragora.server.handlers import exceptions as mod

        for name in [
            "ValidationError",
            "InputValidationError",
            "DatabaseError",
            "RecordNotFoundError",
            "AuthenticationError",
            "AuthorizationError",
            "RateLimitExceededError",
        ]:
            assert name in mod.__all__, f"{name} not in __all__"

    def test_all_exports_are_importable(self):
        from aragora.server.handlers import exceptions as mod

        for name in mod.__all__:
            assert hasattr(mod, name), f"{name} in __all__ but not importable"
