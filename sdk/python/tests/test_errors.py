"""Tests for error handling in the Aragora SDK."""

from __future__ import annotations

from unittest.mock import MagicMock

import httpx
import pytest

from aragora.client import AragoraAsyncClient, AragoraClient
from aragora.exceptions import (
    AragoraError,
    AuthenticationError,
    AuthorizationError,
    ConnectionError,
    NotFoundError,
    RateLimitError,
    ServerError,
    TimeoutError,
    ValidationError,
)


class TestAragoraError:
    """Tests for base AragoraError."""

    def test_error_with_message(self) -> None:
        """AragoraError stores message."""
        error = AragoraError("Something went wrong")
        assert error.message == "Something went wrong"
        assert "Something went wrong" in str(error)

    def test_error_with_status_code(self) -> None:
        """AragoraError stores status code."""
        error = AragoraError("Error", status_code=500)
        assert error.status_code == 500
        assert "(500)" in str(error)

    def test_error_with_response_body(self) -> None:
        """AragoraError stores response body."""
        body = {"error": "details", "code": "ERR_001"}
        error = AragoraError("Error", response_body=body)
        assert error.response_body == body
        assert error.response_body["code"] == "ERR_001"

    def test_error_str_without_status_code(self) -> None:
        """String representation without status code."""
        error = AragoraError("Simple error")
        assert str(error) == "AragoraError: Simple error"

    def test_error_with_error_code(self) -> None:
        """AragoraError stores error code."""
        error = AragoraError("Error", error_code="ERR_CODE")
        assert error.error_code == "ERR_CODE"
        assert "[ERR_CODE]" in str(error)

    def test_error_with_trace_id(self) -> None:
        """AragoraError stores trace id."""
        error = AragoraError("Error", trace_id="trace-123")
        assert error.trace_id == "trace-123"
        assert "trace: trace-123" in str(error)

    def test_error_str_with_all_fields(self) -> None:
        """String representation with all fields."""
        error = AragoraError(
            "Something went wrong",
            status_code=500,
            error_code="INTERNAL_ERROR",
            trace_id="abc-123-xyz",
        )
        result = str(error)
        assert "AragoraError" in result
        assert "(500)" in result
        assert "[INTERNAL_ERROR]" in result
        assert "Something went wrong" in result
        assert "trace: abc-123-xyz" in result


class TestAuthenticationError:
    """Tests for AuthenticationError."""

    def test_authentication_error_defaults(self) -> None:
        """AuthenticationError has correct defaults."""
        error = AuthenticationError()
        assert error.status_code == 401
        assert "Authentication failed" in error.message

    def test_authentication_error_custom_message(self) -> None:
        """AuthenticationError with custom message."""
        error = AuthenticationError("Invalid API key")
        assert error.message == "Invalid API key"
        assert error.status_code == 401

    def test_authentication_error_with_response_body(self) -> None:
        """AuthenticationError stores response body."""
        body = {"error": "Token expired", "code": "TOKEN_EXPIRED"}
        error = AuthenticationError("Token expired", response_body=body)
        assert error.response_body["code"] == "TOKEN_EXPIRED"

    def test_authentication_error_with_error_code_and_trace_id(self) -> None:
        """AuthenticationError with error_code and trace_id."""
        error = AuthenticationError(
            "Token expired",
            error_code="TOKEN_EXPIRED",
            trace_id="auth-trace-456",
        )
        assert error.error_code == "TOKEN_EXPIRED"
        assert error.trace_id == "auth-trace-456"


class TestAuthorizationError:
    """Tests for AuthorizationError."""

    def test_authorization_error_defaults(self) -> None:
        """AuthorizationError has correct defaults."""
        error = AuthorizationError()
        assert error.status_code == 403
        assert "Access denied" in error.message

    def test_authorization_error_custom_message(self) -> None:
        """AuthorizationError with custom message."""
        error = AuthorizationError("Insufficient permissions")
        assert error.message == "Insufficient permissions"
        assert error.status_code == 403

    def test_authorization_error_with_error_code_and_trace_id(self) -> None:
        """AuthorizationError with error_code and trace_id."""
        error = AuthorizationError(
            "Insufficient permissions",
            error_code="FORBIDDEN",
            trace_id="authz-trace-789",
        )
        assert error.error_code == "FORBIDDEN"
        assert error.trace_id == "authz-trace-789"


class TestNotFoundError:
    """Tests for NotFoundError."""

    def test_not_found_error_defaults(self) -> None:
        """NotFoundError has correct defaults."""
        error = NotFoundError()
        assert error.status_code == 404
        assert "Resource not found" in error.message

    def test_not_found_error_custom_message(self) -> None:
        """NotFoundError with custom message."""
        error = NotFoundError("Debate deb_123 not found")
        assert error.message == "Debate deb_123 not found"

    def test_not_found_error_with_error_code_and_trace_id(self) -> None:
        """NotFoundError with error_code and trace_id."""
        error = NotFoundError(
            "User not found",
            error_code="USER_NOT_FOUND",
            trace_id="nf-trace-001",
        )
        assert error.error_code == "USER_NOT_FOUND"
        assert error.trace_id == "nf-trace-001"


class TestRateLimitError:
    """Tests for RateLimitError."""

    def test_rate_limit_error_defaults(self) -> None:
        """RateLimitError has correct defaults."""
        error = RateLimitError()
        assert error.status_code == 429
        assert "Rate limit exceeded" in error.message
        assert error.retry_after is None

    def test_rate_limit_error_with_retry_after(self) -> None:
        """RateLimitError with retry_after."""
        error = RateLimitError("Too many requests", retry_after=60)
        assert error.retry_after == 60
        assert "retry after 60s" in str(error)

    def test_rate_limit_error_str_without_retry_after(self) -> None:
        """RateLimitError string without retry_after."""
        error = RateLimitError("Rate limited")
        assert "retry after" not in str(error)

    def test_rate_limit_error_with_error_code_and_trace_id(self) -> None:
        """RateLimitError with error_code and trace_id."""
        error = RateLimitError(
            "Too many requests",
            retry_after=120,
            error_code="RATE_LIMITED",
            trace_id="rl-trace-xyz",
        )
        assert error.error_code == "RATE_LIMITED"
        assert error.trace_id == "rl-trace-xyz"
        assert error.retry_after == 120


class TestValidationError:
    """Tests for ValidationError."""

    def test_validation_error_defaults(self) -> None:
        """ValidationError has correct defaults."""
        error = ValidationError()
        assert error.status_code == 400
        assert "Validation failed" in error.message
        assert error.errors == []

    def test_validation_error_with_field_errors(self) -> None:
        """ValidationError with field errors."""
        field_errors = [
            {"field": "task", "message": "Required field"},
            {"field": "agents", "message": "Must be a list"},
        ]
        error = ValidationError("Validation failed", errors=field_errors)
        assert len(error.errors) == 2
        assert error.errors[0]["field"] == "task"

    def test_validation_error_with_error_code_and_trace_id(self) -> None:
        """ValidationError with error_code and trace_id."""
        error = ValidationError(
            "Invalid input",
            errors=[{"field": "name", "message": "Required"}],
            error_code="VALIDATION_FAILED",
            trace_id="val-trace-123",
        )
        assert error.error_code == "VALIDATION_FAILED"
        assert error.trace_id == "val-trace-123"
        assert len(error.errors) == 1


class TestServerError:
    """Tests for ServerError."""

    def test_server_error_defaults(self) -> None:
        """ServerError has correct defaults."""
        error = ServerError()
        assert "Server error" in error.message

    def test_server_error_500(self) -> None:
        """ServerError with 500 status."""
        error = ServerError("Internal server error", status_code=500)
        assert error.status_code == 500

    def test_server_error_502(self) -> None:
        """ServerError with 502 status."""
        error = ServerError("Bad gateway", status_code=502)
        assert error.status_code == 502

    def test_server_error_503(self) -> None:
        """ServerError with 503 status."""
        error = ServerError("Service unavailable", status_code=503)
        assert error.status_code == 503

    def test_server_error_with_error_code_and_trace_id(self) -> None:
        """ServerError with error_code and trace_id."""
        error = ServerError(
            "Internal server error",
            error_code="INTERNAL_ERROR",
            trace_id="srv-trace-500",
        )
        assert error.error_code == "INTERNAL_ERROR"
        assert error.trace_id == "srv-trace-500"


class TestTimeoutError:
    """Tests for TimeoutError."""

    def test_timeout_error_defaults(self) -> None:
        """TimeoutError has correct defaults."""
        error = TimeoutError()
        assert error.message == "Request timed out"
        assert error.status_code is None
        assert error.error_code is None
        assert error.trace_id is None

    def test_timeout_error_custom_message(self) -> None:
        """TimeoutError with custom message."""
        error = TimeoutError("Operation timed out after 30s")
        assert error.message == "Operation timed out after 30s"

    def test_timeout_error_with_error_code_and_trace_id(self) -> None:
        """TimeoutError with error_code and trace_id."""
        error = TimeoutError(
            "Operation timed out after 30s",
            error_code="TIMEOUT",
            trace_id="to-trace-999",
        )
        assert error.error_code == "TIMEOUT"
        assert error.trace_id == "to-trace-999"


class TestConnectionError:
    """Tests for ConnectionError."""

    def test_connection_error_defaults(self) -> None:
        """ConnectionError has correct defaults."""
        error = ConnectionError()
        assert error.message == "Connection failed"
        assert error.status_code is None
        assert error.error_code is None
        assert error.trace_id is None

    def test_connection_error_custom_message(self) -> None:
        """ConnectionError with custom message."""
        error = ConnectionError("Could not connect to server")
        assert error.message == "Could not connect to server"

    def test_connection_error_with_error_code_and_trace_id(self) -> None:
        """ConnectionError with error_code and trace_id."""
        error = ConnectionError(
            "Could not connect to server",
            error_code="CONNECTION_REFUSED",
            trace_id="conn-trace-001",
        )
        assert error.error_code == "CONNECTION_REFUSED"
        assert error.trace_id == "conn-trace-001"


class TestClientErrorHandling:
    """Tests for client error response handling."""

    def test_handle_401_response(self) -> None:
        """Client raises AuthenticationError for 401."""
        client = AragoraClient(base_url="https://api.aragora.ai")

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 401
        mock_response.is_success = False
        mock_response.text = "Unauthorized"
        mock_response.json.return_value = {"error": "Invalid token"}
        mock_response.headers = {}

        with pytest.raises(AuthenticationError) as exc_info:
            client._handle_error_response(mock_response)

        assert exc_info.value.status_code == 401
        client.close()

    def test_handle_403_response(self) -> None:
        """Client raises AuthorizationError for 403."""
        client = AragoraClient(base_url="https://api.aragora.ai")

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 403
        mock_response.is_success = False
        mock_response.text = "Forbidden"
        mock_response.json.return_value = {"error": "Access denied"}
        mock_response.headers = {}

        with pytest.raises(AuthorizationError) as exc_info:
            client._handle_error_response(mock_response)

        assert exc_info.value.status_code == 403
        client.close()

    def test_handle_404_response(self) -> None:
        """Client raises NotFoundError for 404."""
        client = AragoraClient(base_url="https://api.aragora.ai")

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 404
        mock_response.is_success = False
        mock_response.text = "Not found"
        mock_response.json.return_value = {"error": "Resource not found"}
        mock_response.headers = {}

        with pytest.raises(NotFoundError) as exc_info:
            client._handle_error_response(mock_response)

        assert exc_info.value.status_code == 404
        client.close()

    def test_handle_429_response(self) -> None:
        """Client raises RateLimitError for 429."""
        client = AragoraClient(base_url="https://api.aragora.ai")

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 429
        mock_response.is_success = False
        mock_response.text = "Too many requests"
        mock_response.json.return_value = {"error": "Rate limit exceeded"}
        mock_response.headers = {"Retry-After": "30"}

        with pytest.raises(RateLimitError) as exc_info:
            client._handle_error_response(mock_response)

        assert exc_info.value.status_code == 429
        assert exc_info.value.retry_after == 30
        client.close()

    def test_handle_429_response_no_retry_after(self) -> None:
        """Client handles 429 without Retry-After header."""
        client = AragoraClient(base_url="https://api.aragora.ai")

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 429
        mock_response.is_success = False
        mock_response.text = "Rate limited"
        mock_response.json.return_value = {"error": "Rate limit"}
        mock_response.headers = {}

        with pytest.raises(RateLimitError) as exc_info:
            client._handle_error_response(mock_response)

        assert exc_info.value.retry_after is None
        client.close()

    def test_handle_400_response(self) -> None:
        """Client raises ValidationError for 400."""
        client = AragoraClient(base_url="https://api.aragora.ai")

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 400
        mock_response.is_success = False
        mock_response.text = "Bad request"
        mock_response.json.return_value = {
            "error": "Validation failed",
            "errors": [{"field": "task", "message": "Required"}],
        }
        mock_response.headers = {}

        with pytest.raises(ValidationError) as exc_info:
            client._handle_error_response(mock_response)

        assert exc_info.value.status_code == 400
        assert len(exc_info.value.errors) == 1
        client.close()

    def test_handle_500_response(self) -> None:
        """Client raises ServerError for 500."""
        client = AragoraClient(base_url="https://api.aragora.ai")

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 500
        mock_response.is_success = False
        mock_response.text = "Internal server error"
        mock_response.json.return_value = {"error": "Internal error"}
        mock_response.headers = {}

        with pytest.raises(ServerError) as exc_info:
            client._handle_error_response(mock_response)

        assert exc_info.value.status_code == 500
        client.close()

    def test_handle_502_response(self) -> None:
        """Client raises ServerError for 502."""
        client = AragoraClient(base_url="https://api.aragora.ai")

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 502
        mock_response.is_success = False
        mock_response.text = "Bad gateway"
        mock_response.json.return_value = {"error": "Bad gateway"}
        mock_response.headers = {}

        with pytest.raises(ServerError) as exc_info:
            client._handle_error_response(mock_response)

        assert exc_info.value.status_code == 502
        client.close()

    def test_handle_unknown_4xx_response(self) -> None:
        """Client raises AragoraError for unknown 4xx."""
        client = AragoraClient(base_url="https://api.aragora.ai")

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 418  # I'm a teapot
        mock_response.is_success = False
        mock_response.text = "I'm a teapot"
        mock_response.json.return_value = {"error": "Teapot"}
        mock_response.headers = {}

        with pytest.raises(AragoraError) as exc_info:
            client._handle_error_response(mock_response)

        assert exc_info.value.status_code == 418
        client.close()

    def test_handle_response_with_invalid_json(self) -> None:
        """Client handles response with invalid JSON body."""
        client = AragoraClient(base_url="https://api.aragora.ai")

        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 500
        mock_response.is_success = False
        mock_response.text = "Internal Server Error"
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_response.headers = {}

        with pytest.raises(ServerError) as exc_info:
            client._handle_error_response(mock_response)

        # Should fall back to response.text
        assert "Internal Server Error" in exc_info.value.message
        client.close()


class TestAsyncClientErrorHandling:
    """Tests for async client error handling."""

    @pytest.mark.asyncio
    async def test_async_handle_401_response(self) -> None:
        """Async client raises AuthenticationError for 401."""
        async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
            mock_response = MagicMock(spec=httpx.Response)
            mock_response.status_code = 401
            mock_response.is_success = False
            mock_response.text = "Unauthorized"
            mock_response.json.return_value = {"error": "Invalid token"}
            mock_response.headers = {}

            with pytest.raises(AuthenticationError):
                client._handle_error_response(mock_response)

    @pytest.mark.asyncio
    async def test_async_handle_429_response(self) -> None:
        """Async client raises RateLimitError for 429."""
        async with AragoraAsyncClient(base_url="https://api.aragora.ai") as client:
            mock_response = MagicMock(spec=httpx.Response)
            mock_response.status_code = 429
            mock_response.is_success = False
            mock_response.text = "Rate limited"
            mock_response.json.return_value = {"error": "Too many requests"}
            mock_response.headers = {"Retry-After": "45"}

            with pytest.raises(RateLimitError) as exc_info:
                client._handle_error_response(mock_response)

            assert exc_info.value.retry_after == 45


class TestErrorInheritance:
    """Tests for exception inheritance."""

    def test_authentication_error_is_aragora_error(self) -> None:
        """AuthenticationError inherits from AragoraError."""
        error = AuthenticationError()
        assert isinstance(error, AragoraError)

    def test_authorization_error_is_aragora_error(self) -> None:
        """AuthorizationError inherits from AragoraError."""
        error = AuthorizationError()
        assert isinstance(error, AragoraError)

    def test_not_found_error_is_aragora_error(self) -> None:
        """NotFoundError inherits from AragoraError."""
        error = NotFoundError()
        assert isinstance(error, AragoraError)

    def test_rate_limit_error_is_aragora_error(self) -> None:
        """RateLimitError inherits from AragoraError."""
        error = RateLimitError()
        assert isinstance(error, AragoraError)

    def test_validation_error_is_aragora_error(self) -> None:
        """ValidationError inherits from AragoraError."""
        error = ValidationError()
        assert isinstance(error, AragoraError)

    def test_server_error_is_aragora_error(self) -> None:
        """ServerError inherits from AragoraError."""
        error = ServerError()
        assert isinstance(error, AragoraError)

    def test_all_errors_are_exceptions(self) -> None:
        """All custom errors inherit from Exception."""
        errors = [
            AragoraError("test"),
            AuthenticationError(),
            AuthorizationError(),
            NotFoundError(),
            RateLimitError(),
            ValidationError(),
            ServerError(),
            TimeoutError(),
            ConnectionError(),
        ]
        for error in errors:
            assert isinstance(error, Exception)

    def test_timeout_error_is_aragora_error(self) -> None:
        """TimeoutError inherits from AragoraError."""
        error = TimeoutError()
        assert isinstance(error, AragoraError)

    def test_connection_error_is_aragora_error(self) -> None:
        """ConnectionError inherits from AragoraError."""
        error = ConnectionError()
        assert isinstance(error, AragoraError)

    def test_all_errors_catchable_as_aragora_error(self) -> None:
        """All specific errors can be caught as AragoraError."""
        errors = [
            AuthenticationError(),
            AuthorizationError(),
            NotFoundError(),
            RateLimitError(),
            ValidationError(),
            ServerError(),
            TimeoutError(),
            ConnectionError(),
        ]
        for error in errors:
            try:
                raise error
            except AragoraError as e:
                assert isinstance(e, AragoraError)
            except Exception:
                pytest.fail(f"{type(error).__name__} was not caught as AragoraError")
