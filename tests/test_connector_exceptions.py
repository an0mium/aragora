"""
Tests for connector exception classification and handling utilities.

Covers:
- Exception type classification
- Retryability detection
- Context manager usage
- Error message extraction
"""

import asyncio
import json
import ssl
import pytest

from aragora.connectors.exceptions import (
    ConnectorError,
    ConnectorAuthError,
    ConnectorRateLimitError,
    ConnectorTimeoutError,
    ConnectorNetworkError,
    ConnectorAPIError,
    ConnectorValidationError,
    ConnectorNotFoundError,
    ConnectorParseError,
    classify_exception,
    connector_error_handler,
    is_retryable_error,
    get_retry_delay,
)


class TestClassifyException:
    """Tests for the classify_exception utility."""

    def test_classifies_asyncio_timeout(self):
        """asyncio.TimeoutError should become ConnectorTimeoutError."""
        error = asyncio.TimeoutError("operation timed out")
        result = classify_exception(error, "test_connector")

        assert isinstance(result, ConnectorTimeoutError)
        assert result.connector_name == "test_connector"
        assert result.is_retryable is True

    def test_classifies_builtin_timeout(self):
        """Built-in TimeoutError should become ConnectorTimeoutError."""
        error = TimeoutError("socket timed out")
        result = classify_exception(error, "test_connector")

        assert isinstance(result, ConnectorTimeoutError)

    def test_classifies_connection_error(self):
        """ConnectionError should become ConnectorNetworkError."""
        error = ConnectionError("Connection refused")
        result = classify_exception(error, "test_connector")

        assert isinstance(result, ConnectorNetworkError)
        assert result.is_retryable is True

    def test_classifies_json_decode_error(self):
        """JSONDecodeError should become ConnectorParseError."""
        error = json.JSONDecodeError("Invalid JSON", "doc", 0)
        result = classify_exception(error, "test_connector")

        assert isinstance(result, ConnectorParseError)
        assert result.content_type == "application/json"
        assert result.is_retryable is False

    def test_classifies_value_error(self):
        """ValueError should become ConnectorValidationError."""
        error = ValueError("Invalid parameter")
        result = classify_exception(error, "test_connector")

        assert isinstance(result, ConnectorValidationError)
        assert result.is_retryable is False

    def test_classifies_rate_limit_from_message(self):
        """Error with '429' in message should become ConnectorRateLimitError."""
        error = Exception("HTTP 429: Too Many Requests")
        result = classify_exception(error, "test_connector")

        assert isinstance(result, ConnectorRateLimitError)
        assert result.is_retryable is True

    def test_classifies_auth_from_message(self):
        """Error with '401' in message should become ConnectorAuthError."""
        error = Exception("HTTP 401: Unauthorized")
        result = classify_exception(error, "test_connector")

        assert isinstance(result, ConnectorAuthError)
        assert result.is_retryable is False

    def test_classifies_not_found_from_message(self):
        """Error with '404' in message should become ConnectorNotFoundError."""
        error = Exception("Resource not found: HTTP 404")
        result = classify_exception(error, "test_connector")

        assert isinstance(result, ConnectorNotFoundError)

    def test_classifies_server_error_from_message(self):
        """Error with '500' in message should become ConnectorAPIError."""
        error = Exception("Internal Server Error: 500")
        result = classify_exception(error, "test_connector")

        assert isinstance(result, ConnectorAPIError)
        assert result.status_code == 500
        assert result.is_retryable is True  # 5xx is retryable

    def test_preserves_existing_connector_error(self):
        """Existing ConnectorError should be returned with updated name."""
        original = ConnectorTimeoutError("test", "unknown")
        result = classify_exception(original, "new_name")

        assert result is original
        assert result.connector_name == "new_name"

    def test_keeps_named_connector_error(self):
        """ConnectorError with name set should keep it."""
        original = ConnectorTimeoutError("test", "original_name")
        result = classify_exception(original, "new_name")

        assert result is original
        assert result.connector_name == "original_name"  # Kept original

    def test_wraps_unknown_exception(self):
        """Unknown exceptions should become ConnectorAPIError."""

        class CustomError(Exception):
            pass

        error = CustomError("something went wrong")
        result = classify_exception(error, "test_connector")

        assert isinstance(result, ConnectorAPIError)
        assert "something went wrong" in str(result)


class TestConnectorErrorHandler:
    """Tests for the connector_error_handler context manager."""

    def test_sync_context_manager_converts_exception(self):
        """Sync context manager should convert exceptions."""
        with pytest.raises(ConnectorTimeoutError) as exc_info:
            with connector_error_handler("test_conn"):
                raise asyncio.TimeoutError("timed out")

        assert exc_info.value.connector_name == "test_conn"

    def test_sync_context_manager_no_exception(self):
        """Sync context manager should pass through without exception."""
        result = None
        with connector_error_handler("test_conn"):
            result = "success"

        assert result == "success"

    @pytest.mark.asyncio
    async def test_async_context_manager_converts_exception(self):
        """Async context manager should convert exceptions."""
        with pytest.raises(ConnectorNetworkError) as exc_info:
            async with connector_error_handler("async_conn"):
                raise ConnectionError("connection refused")

        assert exc_info.value.connector_name == "async_conn"

    @pytest.mark.asyncio
    async def test_async_context_manager_no_exception(self):
        """Async context manager should pass through without exception."""
        result = None
        async with connector_error_handler("async_conn"):
            result = "async success"

        assert result == "async success"

    def test_preserves_cause_chain(self):
        """Context manager should preserve exception cause chain."""
        with pytest.raises(ConnectorParseError) as exc_info:
            with connector_error_handler("test_conn"):
                raise json.JSONDecodeError("bad json", "doc", 0)

        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, json.JSONDecodeError)


class TestRetryability:
    """Tests for retryability detection."""

    def test_timeout_error_is_retryable(self):
        """Timeout errors should be retryable."""
        error = ConnectorTimeoutError("timeout", "test")
        assert is_retryable_error(error) is True

    def test_network_error_is_retryable(self):
        """Network errors should be retryable."""
        error = ConnectorNetworkError("connection failed", "test")
        assert is_retryable_error(error) is True

    def test_rate_limit_error_is_retryable(self):
        """Rate limit errors should be retryable."""
        error = ConnectorRateLimitError("too many requests", "test")
        assert is_retryable_error(error) is True

    def test_auth_error_not_retryable(self):
        """Auth errors should NOT be retryable."""
        error = ConnectorAuthError("invalid key", "test")
        assert is_retryable_error(error) is False

    def test_validation_error_not_retryable(self):
        """Validation errors should NOT be retryable."""
        error = ConnectorValidationError("bad input", "test")
        assert is_retryable_error(error) is False

    def test_5xx_api_error_is_retryable(self):
        """5xx API errors should be retryable."""
        error = ConnectorAPIError("server error", "test", status_code=503)
        assert is_retryable_error(error) is True

    def test_4xx_api_error_not_retryable(self):
        """4xx API errors should NOT be retryable."""
        error = ConnectorAPIError("bad request", "test", status_code=400)
        assert is_retryable_error(error) is False


class TestRetryDelay:
    """Tests for retry delay calculation."""

    def test_rate_limit_error_has_60s_delay(self):
        """Rate limit errors should have 60s default delay."""
        error = ConnectorRateLimitError("rate limited", "test")
        delay = get_retry_delay(error)
        assert delay == 60.0

    def test_custom_retry_after_respected(self):
        """Custom retry_after should be respected."""
        error = ConnectorRateLimitError("rate limited", "test", retry_after=120.0)
        delay = get_retry_delay(error)
        assert delay == 120.0

    def test_default_delay_for_unknown_error(self):
        """Unknown errors should use default delay."""
        error = Exception("unknown")
        delay = get_retry_delay(error, default=10.0)
        assert delay == 10.0

    def test_timeout_error_has_short_delay(self):
        """Timeout errors should have short delay."""
        error = ConnectorTimeoutError("timeout", "test")
        delay = get_retry_delay(error)
        assert delay == 5.0  # Short delay for transient errors
