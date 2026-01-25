"""Tests for connector retry logic and chaos scenarios.

Tests the exponential backoff retry mechanism and exception handling
for various failure scenarios (network errors, rate limits, timeouts).
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.connectors.base import BaseConnector, Evidence
from aragora.connectors.exceptions import (
    ConnectorAPIError,
    ConnectorAuthError,
    ConnectorError,
    ConnectorNetworkError,
    ConnectorNotFoundError,
    ConnectorParseError,
    ConnectorQuotaError,
    ConnectorRateLimitError,
    ConnectorTimeoutError,
    ConnectorValidationError,
    classify_exception,
    connector_error_handler,
    get_retry_delay,
    is_retryable_error,
)
from aragora.reasoning.provenance import SourceType


class MockConnector(BaseConnector):
    """Mock connector for testing."""

    @property
    def source_type(self) -> SourceType:
        return SourceType.WEB_SEARCH

    @property
    def name(self) -> str:
        return "MockConnector"

    async def search(self, query: str, limit: int = 10, **kwargs) -> list[Evidence]:
        return []

    async def fetch(self, evidence_id: str):
        return None


class TestConnectorExceptions:
    """Test connector exception hierarchy."""

    def test_connector_error_base(self):
        """Test base ConnectorError."""
        error = ConnectorError("Base error", connector_name="test")
        assert str(error).startswith("[test] Base error")
        assert error.connector_name == "test"
        assert error.is_retryable is False
        assert error.retry_after is None

    def test_connector_error_unknown_name(self):
        """Test ConnectorError with unknown name."""
        error = ConnectorError("Error")
        assert str(error).startswith("Error")
        assert error.connector_name == "unknown"

    def test_auth_error(self):
        """Test ConnectorAuthError."""
        error = ConnectorAuthError("Invalid API key", connector_name="github")
        assert error.is_retryable is False
        assert "github" in str(error)

    def test_rate_limit_error(self):
        """Test ConnectorRateLimitError."""
        error = ConnectorRateLimitError(
            "Rate limit exceeded", connector_name="twitter", retry_after=120.0
        )
        assert error.is_retryable is True
        assert error.retry_after == 120.0

    def test_rate_limit_error_default_retry_after(self):
        """Test ConnectorRateLimitError default retry_after."""
        error = ConnectorRateLimitError("Rate limited", connector_name="api")
        assert error.retry_after == 60.0  # Default

    def test_timeout_error(self):
        """Test ConnectorTimeoutError."""
        error = ConnectorTimeoutError(
            "Request timed out", connector_name="web", timeout_seconds=30.0
        )
        assert error.is_retryable is True
        assert error.timeout_seconds == 30.0
        assert error.retry_after == 5.0

    def test_network_error(self):
        """Test ConnectorNetworkError."""
        error = ConnectorNetworkError("Connection refused", connector_name="api")
        assert error.is_retryable is True
        assert error.retry_after == 5.0

    def test_api_error_5xx(self):
        """Test ConnectorAPIError with 5xx status (retryable)."""
        error = ConnectorAPIError("Server error", connector_name="api", status_code=503)
        assert error.is_retryable is True
        assert error.status_code == 503

    def test_api_error_4xx(self):
        """Test ConnectorAPIError with 4xx status (not retryable)."""
        error = ConnectorAPIError("Bad request", connector_name="api", status_code=400)
        assert error.is_retryable is False
        assert error.status_code == 400

    def test_validation_error(self):
        """Test ConnectorValidationError."""
        error = ConnectorValidationError("Invalid parameter", connector_name="api", field="query")
        assert error.is_retryable is False
        assert error.field == "query"

    def test_not_found_error(self):
        """Test ConnectorNotFoundError."""
        error = ConnectorNotFoundError(
            "Resource not found", connector_name="api", resource_id="doc-123"
        )
        assert error.is_retryable is False
        assert error.resource_id == "doc-123"

    def test_quota_error(self):
        """Test ConnectorQuotaError."""
        error = ConnectorQuotaError("Quota exhausted", connector_name="api", quota_reset=3600.0)
        assert error.is_retryable is False
        assert error.quota_reset == 3600.0
        assert error.retry_after == 3600.0

    def test_parse_error(self):
        """Test ConnectorParseError."""
        error = ConnectorParseError(
            "JSON decode failed",
            connector_name="api",
            content_type="application/json",
        )
        assert error.is_retryable is False
        assert error.content_type == "application/json"


class TestExceptionUtilities:
    """Test exception utility functions."""

    def test_is_retryable_connector_error(self):
        """Test is_retryable_error with ConnectorError."""
        assert is_retryable_error(ConnectorTimeoutError("timeout")) is True
        assert is_retryable_error(ConnectorNetworkError("network")) is True
        assert is_retryable_error(ConnectorRateLimitError("rate")) is True
        assert is_retryable_error(ConnectorAuthError("auth")) is False
        assert is_retryable_error(ConnectorValidationError("validation")) is False

    def test_is_retryable_generic_error(self):
        """Test is_retryable_error with generic exceptions."""
        assert is_retryable_error(TimeoutError("timeout")) is True
        assert is_retryable_error(ConnectionError("connection")) is True
        assert is_retryable_error(Exception("429 rate limit")) is True
        assert is_retryable_error(ValueError("invalid")) is False

    def test_get_retry_delay_connector_error(self):
        """Test get_retry_delay with ConnectorError."""
        error = ConnectorRateLimitError("rate", retry_after=120.0)
        assert get_retry_delay(error) == 120.0

    def test_get_retry_delay_no_retry_after(self):
        """Test get_retry_delay with no retry_after."""
        error = ConnectorAPIError("error")
        assert get_retry_delay(error, default=10.0) == 10.0

    def test_get_retry_delay_rate_limit(self):
        """Test get_retry_delay for rate limit error."""
        error = ConnectorRateLimitError("rate")
        # Should use default 60.0 for rate limit
        assert get_retry_delay(error) == 60.0


class TestClassifyException:
    """Test exception classification."""

    def test_classify_timeout_error(self):
        """Test classifying TimeoutError."""
        error = classify_exception(TimeoutError("timed out"), "test")
        assert isinstance(error, ConnectorTimeoutError)
        assert error.connector_name == "test"

    def test_classify_asyncio_timeout(self):
        """Test classifying asyncio.TimeoutError."""
        error = classify_exception(asyncio.TimeoutError(), "test")
        assert isinstance(error, ConnectorTimeoutError)

    def test_classify_connection_error(self):
        """Test classifying ConnectionError."""
        error = classify_exception(ConnectionError("connection refused"), "test")
        assert isinstance(error, ConnectorNetworkError)

    def test_classify_json_decode_error(self):
        """Test classifying JSON decode error."""
        import json

        try:
            json.loads("invalid json")
        except json.JSONDecodeError as e:
            error = classify_exception(e, "test")
            assert isinstance(error, ConnectorParseError)
            assert error.content_type == "application/json"

    def test_classify_rate_limit_message(self):
        """Test classifying error with rate limit message."""
        error = classify_exception(Exception("429 Too Many Requests"), "test")
        assert isinstance(error, ConnectorRateLimitError)

    def test_classify_auth_message(self):
        """Test classifying error with auth message."""
        error = classify_exception(Exception("401 Unauthorized"), "test")
        assert isinstance(error, ConnectorAuthError)

    def test_classify_not_found_message(self):
        """Test classifying error with not found message."""
        error = classify_exception(Exception("404 Not Found"), "test")
        assert isinstance(error, ConnectorNotFoundError)

    def test_classify_server_error_message(self):
        """Test classifying error with server error message."""
        error = classify_exception(Exception("500 Internal Server Error"), "test")
        assert isinstance(error, ConnectorAPIError)
        assert error.status_code == 500

    def test_classify_value_error(self):
        """Test classifying ValueError."""
        error = classify_exception(ValueError("invalid value"), "test")
        assert isinstance(error, ConnectorValidationError)

    def test_classify_already_connector_error(self):
        """Test classifying already-wrapped ConnectorError."""
        original = ConnectorTimeoutError("timeout", connector_name="original")
        error = classify_exception(original, "new")
        assert error is original
        assert error.connector_name == "original"  # Not changed

    def test_classify_unknown_error(self):
        """Test classifying unknown error."""
        error = classify_exception(Exception("unknown error"), "test")
        assert isinstance(error, ConnectorAPIError)


class TestConnectorErrorHandler:
    """Test connector_error_handler context manager."""

    def test_sync_handler_no_error(self):
        """Test sync handler with no error."""
        with connector_error_handler("test"):
            result = 1 + 1
        assert result == 2

    def test_sync_handler_with_error(self):
        """Test sync handler converts error."""
        with pytest.raises(ConnectorTimeoutError):
            with connector_error_handler("test"):
                raise TimeoutError("timed out")

    @pytest.mark.asyncio
    async def test_async_handler_no_error(self):
        """Test async handler with no error."""
        async with connector_error_handler("test"):
            result = await asyncio.sleep(0, result=42)
        assert result == 42

    @pytest.mark.asyncio
    async def test_async_handler_with_error(self):
        """Test async handler converts error."""
        with pytest.raises(ConnectorNetworkError):
            async with connector_error_handler("test"):
                raise ConnectionError("connection failed")


class TestRetryMechanism:
    """Test the retry mechanism with exponential backoff."""

    @pytest.fixture
    def connector(self):
        """Create a mock connector with short retry delays."""
        return MockConnector(
            max_retries=3,
            base_delay=0.01,  # Very short for testing
            max_delay=0.1,
        )

    def test_calculate_retry_delay_exponential(self):
        """Test exponential backoff calculation."""
        # Use larger delays to see clear exponential increase
        connector = MockConnector(
            max_retries=3,
            base_delay=1.0,
            max_delay=100.0,  # High max to avoid capping
        )
        delays = [connector._calculate_retry_delay(i) for i in range(4)]

        # Delays should generally increase (accounting for jitter)
        # Base: ~1.0, ~2.0, ~4.0, ~8.0
        # Check that delay at attempt 3 is greater than attempt 0 (with margin for jitter)
        assert delays[3] > delays[0] * 1.5  # Significant increase

    def test_calculate_retry_delay_respects_max(self):
        """Test retry delay respects max delay."""
        connector = MockConnector(
            max_retries=3,
            base_delay=1.0,
            max_delay=2.0,
        )

        # With high attempt, delay should be capped at max
        delay = connector._calculate_retry_delay(100)
        # Max delay is 2.0, jitter can add up to 30%, so max is ~2.6
        assert delay <= 2.0 * 1.5  # Allow for jitter
        # Should be at least the minimum of 0.1
        assert delay >= 0.1

    @pytest.mark.asyncio
    async def test_request_with_retry_success(self, connector):
        """Test successful request returns result."""

        async def successful_request():
            return {"data": "result"}

        result = await connector._request_with_retry(successful_request, "test")
        assert result == {"data": "result"}

    @pytest.mark.asyncio
    async def test_request_with_retry_httpx_not_available(self, connector):
        """Test fallback when httpx not available."""

        async def request():
            return "result"

        with patch.dict("sys.modules", {"httpx": None}):
            # Should still work without httpx (no retry wrapper)
            result = await connector._request_with_retry(request, "test")
            assert result == "result"


class TestRetryChaosScenarios:
    """Test chaos scenarios - various failure modes."""

    @pytest.fixture
    def connector(self):
        """Create a mock connector with short retry delays."""
        return MockConnector(
            max_retries=2,
            base_delay=0.001,  # Very short for testing
            max_delay=0.01,
        )

    @pytest.mark.asyncio
    async def test_timeout_retry_then_success(self, connector):
        """Test timeout followed by success."""
        pytest.importorskip("httpx")
        import httpx

        attempt_count = 0

        async def request_with_timeout():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 2:
                raise httpx.TimeoutException("timeout")
            return {"success": True}

        result = await connector._request_with_retry(request_with_timeout, "test")
        assert result == {"success": True}
        assert attempt_count == 2

    @pytest.mark.asyncio
    async def test_timeout_exhausts_retries(self, connector):
        """Test timeout exhausts all retries."""
        pytest.importorskip("httpx")
        import httpx

        async def always_timeout():
            raise httpx.TimeoutException("timeout")

        with pytest.raises(ConnectorTimeoutError):
            await connector._request_with_retry(always_timeout, "test")

    @pytest.mark.asyncio
    async def test_network_error_retry(self, connector):
        """Test network error triggers retry."""
        pytest.importorskip("httpx")
        import httpx

        attempt_count = 0

        async def network_then_success():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 2:
                raise httpx.ConnectError("connection refused")
            return {"success": True}

        result = await connector._request_with_retry(network_then_success, "test")
        assert result == {"success": True}
        assert attempt_count == 2

    @pytest.mark.asyncio
    async def test_rate_limit_with_retry_after(self, connector):
        """Test 429 error respects Retry-After header."""
        pytest.importorskip("httpx")
        import httpx

        attempt_count = 0

        async def rate_limit_then_success():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 2:
                response = MagicMock()
                response.status_code = 429
                response.headers = {"Retry-After": "0.001"}
                raise httpx.HTTPStatusError("rate limit", request=MagicMock(), response=response)
            return {"success": True}

        result = await connector._request_with_retry(rate_limit_then_success, "test")
        assert result == {"success": True}

    @pytest.mark.asyncio
    async def test_server_error_retry(self, connector):
        """Test 5xx error triggers retry."""
        pytest.importorskip("httpx")
        import httpx

        attempt_count = 0

        async def server_error_then_success():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 2:
                response = MagicMock()
                response.status_code = 503
                raise httpx.HTTPStatusError(
                    "service unavailable", request=MagicMock(), response=response
                )
            return {"success": True}

        result = await connector._request_with_retry(server_error_then_success, "test")
        assert result == {"success": True}

    @pytest.mark.asyncio
    async def test_client_error_no_retry(self, connector):
        """Test 4xx error does not retry."""
        pytest.importorskip("httpx")
        import httpx

        async def client_error():
            response = MagicMock()
            response.status_code = 400
            raise httpx.HTTPStatusError("bad request", request=MagicMock(), response=response)

        with pytest.raises(ConnectorAPIError) as exc_info:
            await connector._request_with_retry(client_error, "test")

        assert exc_info.value.status_code == 400
        assert exc_info.value.is_retryable is False

    @pytest.mark.asyncio
    async def test_parse_error_no_retry(self, connector):
        """Test parse error does not retry."""
        pytest.importorskip("httpx")

        async def parse_error():
            raise Exception("json decode error")

        with pytest.raises(ConnectorParseError):
            await connector._request_with_retry(parse_error, "test")


class TestCacheUnderChaos:
    """Test cache behavior under chaos conditions."""

    def test_cache_survives_concurrent_access(self):
        """Test cache handles concurrent access."""
        import threading

        connector = MockConnector()
        errors = []

        def cache_operations(thread_id: int):
            try:
                for i in range(100):
                    evidence = Evidence(
                        id=f"thread-{thread_id}-{i}",
                        source_type=SourceType.WEB_SEARCH,
                        source_id=f"test-{thread_id}-{i}",
                        content=f"Content {thread_id}-{i}",
                    )
                    connector._cache_put(f"thread-{thread_id}-{i}", evidence)
                    connector._cache_get(f"thread-{thread_id}-{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=cache_operations, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_cache_clear_expired_under_load(self):
        """Test cache expiry cleanup under load."""
        connector = MockConnector(cache_ttl_seconds=0.001, max_cache_entries=100)

        # Add many entries
        for i in range(50):
            evidence = Evidence(
                id=f"expire-{i}",
                source_type=SourceType.WEB_SEARCH,
                source_id=f"test-{i}",
                content=f"Content {i}",
            )
            connector._cache_put(f"expire-{i}", evidence)

        # Wait for expiry
        import time

        time.sleep(0.01)

        # Clear expired
        cleared = connector._cache_clear_expired()
        assert cleared > 0

        # Stats should reflect
        stats = connector._cache_stats()
        assert stats["expired_entries"] == 0
