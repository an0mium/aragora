"""
Tests for connector exception hierarchy.

Tests cover:
- All exception types and their attributes
- Retry eligibility detection
- Retry delay calculation
- Exception classification utilities
- Error handler context manager
- classify_connector_error for chat connectors
"""

import asyncio
import json
import pytest
import ssl

from aragora.connectors.exceptions import (
    # Base exception
    ConnectorError,
    # Specific exceptions
    ConnectorAuthError,
    ConnectorRateLimitError,
    ConnectorTimeoutError,
    ConnectorNetworkError,
    ConnectorAPIError,
    ConnectorValidationError,
    ConnectorNotFoundError,
    ConnectorQuotaError,
    ConnectorParseError,
    ConnectorConfigError,
    ConnectorCircuitOpenError,
    # Utilities
    is_retryable_error,
    get_retry_delay,
    classify_exception,
    classify_connector_error,
    connector_error_handler,
)


class TestConnectorError:
    """Tests for base ConnectorError class."""

    def test_basic_initialization(self):
        """Should initialize with message and defaults."""
        error = ConnectorError("Something went wrong")

        assert error.message == "Something went wrong"
        assert error.connector_name == "unknown"
        assert error.retry_after is None
        assert error.is_retryable is False

    def test_with_connector_name(self):
        """Should store connector name."""
        error = ConnectorError("Error", connector_name="github")

        assert error.connector_name == "github"

    def test_with_retry_after(self):
        """Should store retry_after value."""
        error = ConnectorError("Error", retry_after=30.0)

        assert error.retry_after == 30.0

    def test_with_retryable_flag(self):
        """Should store is_retryable flag."""
        error = ConnectorError("Error", is_retryable=True)

        assert error.is_retryable is True

    def test_details_dict(self):
        """Should include connector info in details."""
        error = ConnectorError(
            "Error",
            connector_name="slack",
            retry_after=60.0,
            is_retryable=True,
        )

        assert error.details["connector_name"] == "slack"
        assert error.details["is_retryable"] is True
        assert error.details["retry_after"] == 60.0

    def test_str_without_connector(self):
        """Should format string without connector name when unknown."""
        error = ConnectorError("API failed")

        str_repr = str(error)
        assert "API failed" in str_repr

    def test_str_with_connector(self):
        """Should include connector name in string representation."""
        error = ConnectorError("API failed", connector_name="twitter")

        str_repr = str(error)
        assert "[twitter]" in str_repr
        assert "API failed" in str_repr

    def test_str_with_retry_info(self):
        """Should include retry info in string representation."""
        error = ConnectorError(
            "Rate limited",
            connector_name="api",
            retry_after=30.0,
            is_retryable=True,
        )

        str_repr = str(error)
        assert "retry_after=30.0s" in str_repr
        assert "retryable=True" in str_repr


class TestConnectorAuthError:
    """Tests for ConnectorAuthError."""

    def test_not_retryable_by_default(self):
        """Auth errors should not be retryable."""
        error = ConnectorAuthError("Invalid API key", connector_name="github")

        assert error.is_retryable is False

    def test_stores_connector_name(self):
        """Should store connector name."""
        error = ConnectorAuthError("Forbidden", connector_name="slack")

        assert error.connector_name == "slack"

    def test_can_specify_retry_after(self):
        """Should allow retry_after for token refresh scenarios."""
        error = ConnectorAuthError(
            "Token expired",
            connector_name="oauth",
            retry_after=300.0,  # Refresh in 5 minutes
        )

        assert error.retry_after == 300.0
        assert error.is_retryable is False  # Still not auto-retryable


class TestConnectorRateLimitError:
    """Tests for ConnectorRateLimitError."""

    def test_retryable_by_default(self):
        """Rate limit errors should be retryable."""
        error = ConnectorRateLimitError("429 Too Many Requests")

        assert error.is_retryable is True

    def test_default_retry_after(self):
        """Should have default retry_after of 60 seconds."""
        error = ConnectorRateLimitError("Rate limited")

        assert error.retry_after == 60.0

    def test_custom_retry_after(self):
        """Should accept custom retry_after."""
        error = ConnectorRateLimitError(
            "Rate limited",
            connector_name="twitter",
            retry_after=120.0,
        )

        assert error.retry_after == 120.0


class TestConnectorTimeoutError:
    """Tests for ConnectorTimeoutError."""

    def test_retryable(self):
        """Timeout errors should be retryable."""
        error = ConnectorTimeoutError("Request timed out")

        assert error.is_retryable is True

    def test_short_retry_delay(self):
        """Should have short retry delay (5s)."""
        error = ConnectorTimeoutError("Timeout")

        assert error.retry_after == 5.0

    def test_stores_timeout_seconds(self):
        """Should store the timeout duration."""
        error = ConnectorTimeoutError(
            "Connection timed out after 30s",
            connector_name="api",
            timeout_seconds=30.0,
        )

        assert error.timeout_seconds == 30.0


class TestConnectorNetworkError:
    """Tests for ConnectorNetworkError."""

    def test_retryable(self):
        """Network errors should be retryable."""
        error = ConnectorNetworkError("Connection refused")

        assert error.is_retryable is True

    def test_retry_delay(self):
        """Should have 5 second retry delay."""
        error = ConnectorNetworkError("DNS lookup failed")

        assert error.retry_after == 5.0


class TestConnectorAPIError:
    """Tests for ConnectorAPIError."""

    def test_5xx_is_retryable(self):
        """5xx errors should be retryable."""
        for status in [500, 502, 503, 504]:
            error = ConnectorAPIError(f"HTTP {status}", status_code=status)
            assert error.is_retryable is True, f"Status {status} should be retryable"

    def test_4xx_not_retryable(self):
        """4xx errors should not be retryable."""
        for status in [400, 401, 403, 404, 422]:
            error = ConnectorAPIError(f"HTTP {status}", status_code=status)
            assert error.is_retryable is False, f"Status {status} should not be retryable"

    def test_stores_status_code(self):
        """Should store HTTP status code."""
        error = ConnectorAPIError("Internal Server Error", status_code=500)

        assert error.status_code == 500

    def test_no_status_code(self):
        """Should handle missing status code."""
        error = ConnectorAPIError("Unknown error")

        assert error.status_code is None
        assert error.is_retryable is False


class TestConnectorValidationError:
    """Tests for ConnectorValidationError."""

    def test_not_retryable(self):
        """Validation errors should not be retryable."""
        error = ConnectorValidationError("Invalid email format")

        assert error.is_retryable is False

    def test_stores_field(self):
        """Should store the invalid field name."""
        error = ConnectorValidationError(
            "Email is required",
            connector_name="form",
            field="email",
        )

        assert error.field == "email"


class TestConnectorNotFoundError:
    """Tests for ConnectorNotFoundError."""

    def test_not_retryable(self):
        """Not found errors should not be retryable."""
        error = ConnectorNotFoundError("Resource not found")

        assert error.is_retryable is False

    def test_stores_resource_id(self):
        """Should store the resource ID."""
        error = ConnectorNotFoundError(
            "Document not found",
            connector_name="gdrive",
            resource_id="doc_123",
        )

        assert error.resource_id == "doc_123"


class TestConnectorQuotaError:
    """Tests for ConnectorQuotaError."""

    def test_not_retryable(self):
        """Quota errors should not be retryable."""
        error = ConnectorQuotaError("Monthly quota exceeded")

        assert error.is_retryable is False

    def test_stores_quota_reset(self):
        """Should store quota reset time."""
        error = ConnectorQuotaError(
            "API quota exceeded",
            connector_name="openai",
            quota_reset=3600.0,  # Reset in 1 hour
        )

        assert error.quota_reset == 3600.0
        assert error.retry_after == 3600.0


class TestConnectorParseError:
    """Tests for ConnectorParseError."""

    def test_not_retryable(self):
        """Parse errors should not be retryable."""
        error = ConnectorParseError("JSON decode failed")

        assert error.is_retryable is False

    def test_stores_content_type(self):
        """Should store content type."""
        error = ConnectorParseError(
            "Invalid JSON",
            connector_name="api",
            content_type="application/json",
        )

        assert error.content_type == "application/json"


class TestConnectorConfigError:
    """Tests for ConnectorConfigError."""

    def test_not_retryable(self):
        """Config errors should not be retryable."""
        error = ConnectorConfigError("Missing API key")

        assert error.is_retryable is False

    def test_stores_config_key(self):
        """Should store config key."""
        error = ConnectorConfigError(
            "API_KEY not set",
            connector_name="service",
            config_key="API_KEY",
        )

        assert error.config_key == "API_KEY"


class TestConnectorCircuitOpenError:
    """Tests for ConnectorCircuitOpenError."""

    def test_retryable(self):
        """Circuit open errors should be retryable after cooldown."""
        error = ConnectorCircuitOpenError("Circuit breaker open")

        assert error.is_retryable is True

    def test_default_cooldown(self):
        """Should have default 60 second cooldown."""
        error = ConnectorCircuitOpenError("Circuit open")

        assert error.retry_after == 60.0

    def test_custom_cooldown(self):
        """Should accept custom cooldown."""
        error = ConnectorCircuitOpenError(
            "Circuit open",
            connector_name="api",
            cooldown_remaining=30.0,
        )

        assert error.cooldown_remaining == 30.0
        assert error.retry_after == 30.0


class TestIsRetryableError:
    """Tests for is_retryable_error utility."""

    def test_connector_error_retryable(self):
        """Should use is_retryable attribute for ConnectorError."""
        retryable = ConnectorRateLimitError("Rate limited")
        not_retryable = ConnectorAuthError("Invalid key")

        assert is_retryable_error(retryable) is True
        assert is_retryable_error(not_retryable) is False

    def test_timeout_error_retryable(self):
        """Should detect timeout errors as retryable."""
        error = TimeoutError("Connection timed out")

        assert is_retryable_error(error) is True

    def test_connection_error_retryable(self):
        """Should detect connection errors as retryable."""
        error = ConnectionError("Connection refused")

        assert is_retryable_error(error) is True

    def test_429_in_message_retryable(self):
        """Should detect rate limit indicators in message."""
        error = Exception("HTTP 429: Too Many Requests")

        assert is_retryable_error(error) is True

    def test_generic_error_not_retryable(self):
        """Generic errors should not be retryable."""
        error = ValueError("Invalid value")

        assert is_retryable_error(error) is False


class TestGetRetryDelay:
    """Tests for get_retry_delay utility."""

    def test_uses_retry_after_from_connector_error(self):
        """Should use retry_after from ConnectorError."""
        error = ConnectorRateLimitError("Rate limited", retry_after=120.0)

        assert get_retry_delay(error) == 120.0

    def test_rate_limit_default(self):
        """Rate limit errors should default to 60 seconds."""
        error = ConnectorRateLimitError("Rate limited", retry_after=None)
        # RateLimitError always has retry_after (default 60)
        assert get_retry_delay(error) == 60.0

    def test_default_delay(self):
        """Should return default delay for generic errors."""
        error = Exception("Some error")

        assert get_retry_delay(error) == 5.0

    def test_custom_default(self):
        """Should use custom default delay."""
        error = Exception("Some error")

        assert get_retry_delay(error, default=10.0) == 10.0


class TestClassifyException:
    """Tests for classify_exception utility."""

    def test_timeout_error(self):
        """Should classify TimeoutError as ConnectorTimeoutError."""
        error = TimeoutError("Timed out")
        classified = classify_exception(error, "test")

        assert isinstance(classified, ConnectorTimeoutError)
        assert classified.connector_name == "test"

    def test_asyncio_timeout(self):
        """Should classify asyncio.TimeoutError."""
        error = asyncio.TimeoutError()
        classified = classify_exception(error, "async_client")

        assert isinstance(classified, ConnectorTimeoutError)

    def test_connection_refused(self):
        """Should classify connection refused as network error."""
        error = ConnectionError("Connection refused")
        classified = classify_exception(error, "server")

        assert isinstance(classified, ConnectorNetworkError)

    def test_ssl_error(self):
        """Should classify SSL errors as network errors."""
        error = ssl.SSLError("Certificate verify failed")
        classified = classify_exception(error, "https_client")

        assert isinstance(classified, ConnectorNetworkError)

    def test_json_decode_error(self):
        """Should classify JSON errors as parse errors."""
        error = json.JSONDecodeError("Expecting value", "", 0)
        classified = classify_exception(error, "api")

        assert isinstance(classified, ConnectorParseError)
        assert classified.content_type == "application/json"

    def test_rate_limit_in_message(self):
        """Should detect rate limit from message."""
        error = Exception("429 too many requests")
        classified = classify_exception(error, "api")

        assert isinstance(classified, ConnectorRateLimitError)

    def test_auth_keywords(self):
        """Should detect auth errors from keywords."""
        # Note: classify_exception uses regex "invalid.*key" which needs "invalid" followed by "key"
        for msg in ["401 unauthorized", "403 forbidden", "authentication failed"]:
            error = Exception(msg)
            classified = classify_exception(error, "api")
            assert isinstance(classified, ConnectorAuthError), f"Failed for: {msg}"

    def test_not_found(self):
        """Should detect 404 errors."""
        error = Exception("404 not found")
        classified = classify_exception(error, "api")

        assert isinstance(classified, ConnectorNotFoundError)

    def test_server_error(self):
        """Should detect 5xx server errors."""
        error = Exception("500 internal server error")
        classified = classify_exception(error, "api")

        assert isinstance(classified, ConnectorAPIError)
        assert classified.status_code == 500

    def test_value_error_as_validation(self):
        """Should classify ValueError as validation error."""
        error = ValueError("Invalid format")
        classified = classify_exception(error, "form")

        assert isinstance(classified, ConnectorValidationError)

    def test_type_error_as_validation(self):
        """Should classify TypeError as validation error."""
        error = TypeError("Expected string")
        classified = classify_exception(error, "parser")

        assert isinstance(classified, ConnectorValidationError)

    def test_unknown_as_api_error(self):
        """Should classify unknown errors as API errors."""
        error = RuntimeError("Something unexpected")
        classified = classify_exception(error, "service")

        assert isinstance(classified, ConnectorAPIError)
        assert classified.status_code is None

    def test_existing_connector_error_passthrough(self):
        """Should update connector name on existing ConnectorError."""
        original = ConnectorAuthError("Invalid key", connector_name="unknown")
        classified = classify_exception(original, "github")

        assert classified is original
        assert classified.connector_name == "github"

    def test_existing_connector_error_preserves_name(self):
        """Should preserve connector name if already set."""
        original = ConnectorAuthError("Invalid key", connector_name="slack")
        classified = classify_exception(original, "github")

        assert classified.connector_name == "slack"


class TestConnectorErrorHandler:
    """Tests for connector_error_handler context manager."""

    def test_sync_no_error(self):
        """Should not affect normal execution."""
        with connector_error_handler("test"):
            result = 1 + 1

        assert result == 2

    def test_sync_classifies_error(self):
        """Should classify errors in sync context."""
        with pytest.raises(ConnectorTimeoutError) as exc_info:
            with connector_error_handler("sync_service"):
                raise TimeoutError("Timed out")

        assert exc_info.value.connector_name == "sync_service"

    @pytest.mark.asyncio
    async def test_async_no_error(self):
        """Should not affect normal async execution."""
        async with connector_error_handler("async_test"):
            result = await asyncio.sleep(0, result=42)

        assert result == 42

    @pytest.mark.asyncio
    async def test_async_classifies_error(self):
        """Should classify errors in async context."""
        with pytest.raises(ConnectorNetworkError) as exc_info:
            async with connector_error_handler("async_service"):
                raise ConnectionError("Connection refused")

        assert exc_info.value.connector_name == "async_service"


class TestClassifyConnectorError:
    """Tests for classify_connector_error utility for chat connectors."""

    def test_rate_limit_by_status(self):
        """Should classify 429 as rate limit."""
        error = classify_connector_error(
            "Too many requests",
            connector_name="slack",
            status_code=429,
        )

        assert isinstance(error, ConnectorRateLimitError)

    def test_rate_limit_by_keyword(self):
        """Should classify rate limit keywords."""
        for keyword in ["rate limited", "ratelimited", "too many requests", "throttled"]:
            error = classify_connector_error(keyword, connector_name="api")
            assert isinstance(error, ConnectorRateLimitError), f"Failed for: {keyword}"

    def test_auth_by_status(self):
        """Should classify 401/403 as auth error."""
        for status in [401, 403]:
            error = classify_connector_error("Error", connector_name="api", status_code=status)
            assert isinstance(error, ConnectorAuthError), f"Failed for status {status}"

    def test_auth_by_keyword(self):
        """Should classify auth keywords."""
        keywords = [
            "unauthorized",
            "forbidden",
            "invalid_auth",
            "token_expired",
            "not_authed",
        ]
        for keyword in keywords:
            error = classify_connector_error(keyword, connector_name="slack")
            assert isinstance(error, ConnectorAuthError), f"Failed for: {keyword}"

    def test_not_found_returns_api_error(self):
        """Should return APIError with 404 status for backward compatibility."""
        error = classify_connector_error(
            "Channel not found",
            connector_name="slack",
            status_code=404,
        )

        assert isinstance(error, ConnectorAPIError)
        assert error.status_code == 404

    def test_timeout_by_keyword(self):
        """Should classify timeout keywords."""
        error = classify_connector_error("Request timed out", connector_name="api")

        assert isinstance(error, ConnectorTimeoutError)

    def test_network_by_keyword(self):
        """Should classify network keywords."""
        for keyword in ["connection refused", "network error", "dns failed"]:
            error = classify_connector_error(keyword, connector_name="api")
            assert isinstance(error, ConnectorNetworkError), f"Failed for: {keyword}"

    def test_server_error(self):
        """Should classify 5xx as API error with retryable status."""
        error = classify_connector_error(
            "Internal error",
            connector_name="api",
            status_code=500,
        )

        assert isinstance(error, ConnectorAPIError)
        assert error.status_code == 500
        assert error.is_retryable is True

    def test_client_error(self):
        """Should classify 4xx as non-retryable API error."""
        error = classify_connector_error(
            "Bad request",
            connector_name="api",
            status_code=400,
        )

        assert isinstance(error, ConnectorAPIError)
        assert error.status_code == 400
        assert error.is_retryable is False

    def test_retry_after_preserved(self):
        """Should preserve retry_after value."""
        error = classify_connector_error(
            "Rate limited",
            connector_name="slack",
            status_code=429,
            retry_after=30.0,
        )

        assert error.retry_after == 30.0

    def test_default_rate_limit_retry(self):
        """Should default to 60 seconds for rate limits."""
        error = classify_connector_error(
            "Rate limited",
            connector_name="slack",
            status_code=429,
        )

        assert error.retry_after == 60.0


class TestExceptionHierarchy:
    """Tests for exception inheritance hierarchy."""

    def test_all_inherit_from_connector_error(self):
        """All specific exceptions should inherit from ConnectorError."""
        exception_classes = [
            ConnectorAuthError,
            ConnectorRateLimitError,
            ConnectorTimeoutError,
            ConnectorNetworkError,
            ConnectorAPIError,
            ConnectorValidationError,
            ConnectorNotFoundError,
            ConnectorQuotaError,
            ConnectorParseError,
            ConnectorConfigError,
            ConnectorCircuitOpenError,
        ]

        for exc_class in exception_classes:
            error = exc_class("Test error")
            assert isinstance(error, ConnectorError), (
                f"{exc_class.__name__} should inherit from ConnectorError"
            )

    def test_all_inherit_from_aragora_error(self):
        """All exceptions should inherit from AragoraError."""
        from aragora.exceptions import AragoraError

        error = ConnectorError("Test")
        assert isinstance(error, AragoraError)

    def test_exception_can_be_caught_generically(self):
        """Should be catchable as Exception."""
        try:
            raise ConnectorRateLimitError("Test")
        except Exception as e:
            assert isinstance(e, ConnectorRateLimitError)


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_message(self):
        """Should handle empty message."""
        error = ConnectorError("")
        assert str(error) is not None

    def test_very_long_message(self):
        """Should handle very long messages."""
        long_msg = "x" * 10000
        error = ConnectorError(long_msg)
        assert error.message == long_msg

    def test_unicode_message(self):
        """Should handle unicode in messages."""
        error = ConnectorError("Error: 日本語テスト 🚀")
        assert "日本語" in error.message
        assert "🚀" in error.message

    def test_none_retry_after(self):
        """Should handle None retry_after gracefully."""
        error = ConnectorError("Error", retry_after=None)
        assert error.retry_after is None
        assert get_retry_delay(error) == 5.0

    def test_zero_retry_after(self):
        """Should handle zero retry_after."""
        error = ConnectorError("Error", retry_after=0.0)
        assert error.retry_after == 0.0

    def test_negative_retry_after(self):
        """Should handle negative retry_after (edge case)."""
        error = ConnectorError("Error", retry_after=-1.0)
        assert error.retry_after == -1.0
