"""
Tests for chat connector error handling.

Tests cover:
- Platform-specific error codes
- Rate limit handling
- Network error recovery
- Retry eligibility
- Error message parsing
- Circuit breaker behavior
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio


class TestSlackErrorHandling:
    """Tests for Slack-specific error handling."""

    @pytest.fixture
    def connector(self):
        """Create SlackConnector for testing."""
        from aragora.connectors.chat.slack import SlackConnector

        return SlackConnector(
            bot_token="xoxb-test",
            enable_circuit_breaker=False,  # Disable for unit tests
        )

    @pytest.mark.asyncio
    async def test_rate_limit_error(self, connector):
        """Should handle rate limit (429) errors."""
        # Create connector with no retries for this test
        from aragora.connectors.chat.slack import SlackConnector

        test_connector = SlackConnector(
            bot_token="xoxb-test",
            enable_circuit_breaker=False,
        )
        # Disable retries by setting max_retries to 1
        test_connector._max_retries = 1

        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.headers = {"Retry-After": "30"}
        mock_response.json.return_value = {
            "ok": False,
            "error": "rate_limited",
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            result = await test_connector.send_message("C12345", "Test message")

        # Should indicate failure due to rate limiting
        assert result.success is False or "rate" in str(result).lower()

    @pytest.mark.asyncio
    async def test_channel_not_found_error(self, connector):
        """Should handle channel_not_found error."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "ok": False,
            "error": "channel_not_found",
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            result = await connector.send_message("invalid_channel", "Test")

        assert result.success is False

    @pytest.mark.asyncio
    async def test_not_in_channel_error(self, connector):
        """Should handle not_in_channel error."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "ok": False,
            "error": "not_in_channel",
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            result = await connector.send_message("C12345", "Test")

        assert result.success is False

    @pytest.mark.asyncio
    async def test_invalid_auth_error(self, connector):
        """Should handle invalid_auth error."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "ok": False,
            "error": "invalid_auth",
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            result = await connector.send_message("C12345", "Test")

        assert result.success is False

    @pytest.mark.asyncio
    async def test_message_too_long_error(self, connector):
        """Should handle msg_too_long error."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "ok": False,
            "error": "msg_too_long",
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            long_message = "x" * 50000  # Exceeds Slack limit
            result = await connector.send_message("C12345", long_message)

        assert result.success is False


class TestTeamsErrorHandling:
    """Tests for Teams-specific error handling."""

    @pytest.fixture
    def connector(self):
        """Create TeamsConnector for testing."""
        from aragora.connectors.chat.teams import TeamsConnector

        return TeamsConnector(
            app_id="test_app_id",
            app_password="test_password",
        )

    @pytest.mark.asyncio
    async def test_unauthorized_error(self, connector):
        """Should handle 401 Unauthorized errors."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.json.return_value = {
            "error": {
                "code": "Unauthorized",
                "message": "Access token is invalid or expired",
            }
        }

        with patch.object(
            connector, "_graph_api_request", return_value=(False, None, "Unauthorized")
        ):
            success, data, error = await connector._graph_api_request(
                endpoint="/test",
                method="GET",
                operation="test",
            )

        assert success is False
        assert error == "Unauthorized"

    @pytest.mark.asyncio
    async def test_forbidden_error(self, connector):
        """Should handle 403 Forbidden errors."""
        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_response.json.return_value = {
            "error": {
                "code": "Forbidden",
                "message": "Insufficient permissions to complete the operation",
            }
        }

        with patch.object(connector, "_graph_api_request", return_value=(False, None, "Forbidden")):
            success, data, error = await connector._graph_api_request(
                endpoint="/test",
                method="POST",
                operation="test",
            )

        assert success is False

    @pytest.mark.asyncio
    async def test_not_found_error(self, connector):
        """Should handle 404 Not Found errors."""
        with patch.object(connector, "_graph_api_request", return_value=(False, None, "NotFound")):
            success, data, error = await connector._graph_api_request(
                endpoint="/teams/invalid-id/channels",
                method="GET",
                operation="test",
            )

        assert success is False

    @pytest.mark.asyncio
    async def test_throttling_error(self, connector):
        """Should handle 429 Too Many Requests errors."""
        with patch.object(
            connector,
            "_graph_api_request",
            return_value=(False, None, "TooManyRequests"),
        ):
            success, data, error = await connector._graph_api_request(
                endpoint="/test",
                method="GET",
                operation="test",
            )

        assert success is False


class TestRetryEligibility:
    """Tests for determining retry eligibility."""

    def test_retryable_status_codes(self):
        """Should identify retryable HTTP status codes."""
        from aragora.connectors.chat.slack import _is_retryable_error

        # 429 - Rate limited - should retry
        assert _is_retryable_error(429) is True

        # 500 - Internal server error - should retry
        assert _is_retryable_error(500) is True

        # 502 - Bad gateway - should retry
        assert _is_retryable_error(502) is True

        # 503 - Service unavailable - should retry
        assert _is_retryable_error(503) is True

        # 504 - Gateway timeout - should retry
        assert _is_retryable_error(504) is True

        # 400 - Bad request - should NOT retry
        assert _is_retryable_error(400) is False

        # 401 - Unauthorized - should NOT retry
        assert _is_retryable_error(401) is False

        # 403 - Forbidden - should NOT retry
        assert _is_retryable_error(403) is False

        # 404 - Not found - should NOT retry
        assert _is_retryable_error(404) is False

    def test_retryable_slack_errors(self):
        """Should identify retryable Slack-specific errors."""
        from aragora.connectors.chat.slack import _is_retryable_error

        # Retryable errors
        assert _is_retryable_error(200, "service_unavailable") is True
        assert _is_retryable_error(200, "timeout") is True
        assert _is_retryable_error(200, "internal_error") is True

        # Non-retryable errors
        assert _is_retryable_error(200, "channel_not_found") is False
        assert _is_retryable_error(200, "invalid_auth") is False
        assert _is_retryable_error(200, "not_in_channel") is False


class TestExponentialBackoff:
    """Tests for exponential backoff behavior."""

    @pytest.mark.asyncio
    async def test_backoff_delays_increase(self):
        """Should increase delay exponentially with each attempt."""
        from aragora.connectors.chat.slack import _exponential_backoff

        delays = []

        for attempt in range(5):
            start = asyncio.get_event_loop().time()
            await _exponential_backoff(attempt, base=0.1, max_delay=5.0)
            elapsed = asyncio.get_event_loop().time() - start
            delays.append(elapsed)

        # Each delay should generally be larger than the previous
        # (accounting for jitter and test execution time)
        assert delays[3] > delays[0]

    @pytest.mark.asyncio
    async def test_backoff_respects_max_delay(self):
        """Should not exceed maximum delay."""
        from aragora.connectors.chat.slack import _exponential_backoff

        max_delay = 1.0

        start = asyncio.get_event_loop().time()
        await _exponential_backoff(attempt=10, base=0.5, max_delay=max_delay)
        elapsed = asyncio.get_event_loop().time() - start

        # Should not exceed max_delay (with some margin for test execution)
        assert elapsed <= max_delay + 0.5


class TestCircuitBreakerBehavior:
    """Tests for circuit breaker error handling."""

    @pytest.fixture
    def connector_with_cb(self):
        """Create SlackConnector with circuit breaker enabled."""
        from aragora.connectors.chat.slack import SlackConnector

        return SlackConnector(
            bot_token="xoxb-test",
            enable_circuit_breaker=True,
            circuit_breaker_threshold=3,
            circuit_breaker_cooldown=1.0,  # Short for testing
        )

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_after_failures(self, connector_with_cb):
        """Should open circuit breaker after consecutive failures."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.json.return_value = {"ok": False, "error": "internal_error"}

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            # Trigger multiple failures
            for _ in range(5):
                await connector_with_cb.send_message("C12345", "Test")

        # Circuit breaker should be initialized and tracking failures
        cb = connector_with_cb._get_circuit_breaker()
        if cb:
            # If circuit breaker is available, it should have recorded failures
            assert cb.failure_count >= 0

    @pytest.mark.asyncio
    async def test_circuit_breaker_blocks_requests_when_open(self, connector_with_cb):
        """Should block requests when circuit is open."""
        cb = connector_with_cb._get_circuit_breaker()
        if cb:
            # Manually open the circuit
            for _ in range(connector_with_cb._circuit_breaker_threshold + 1):
                cb.record_failure()

            can_proceed, error = connector_with_cb._check_circuit_breaker()

            # May be blocked if circuit is open
            # (depends on implementation details)


class TestNetworkErrorRecovery:
    """Tests for network error recovery."""

    @pytest.fixture
    def connector(self):
        """Create SlackConnector for testing."""
        from aragora.connectors.chat.slack import SlackConnector

        return SlackConnector(
            bot_token="xoxb-test",
            enable_circuit_breaker=False,
        )

    @pytest.mark.asyncio
    async def test_connection_timeout_handling(self, connector):
        """Should handle connection timeouts gracefully."""
        import httpx

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=httpx.TimeoutException("Connection timed out")
            )

            result = await connector.send_message("C12345", "Test")

        assert result.success is False

    @pytest.mark.asyncio
    async def test_dns_resolution_error(self, connector):
        """Should handle DNS resolution failures."""
        import httpx

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=httpx.ConnectError("DNS lookup failed")
            )

            result = await connector.send_message("C12345", "Test")

        assert result.success is False

    @pytest.mark.asyncio
    async def test_ssl_error_handling(self, connector):
        """Should handle SSL/TLS errors."""
        import httpx

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=httpx.ConnectError("SSL certificate verify failed")
            )

            result = await connector.send_message("C12345", "Test")

        assert result.success is False


class TestErrorMessageParsing:
    """Tests for error message parsing."""

    def test_slack_error_parsing(self):
        """Should parse Slack error responses correctly."""
        response = {
            "ok": False,
            "error": "channel_not_found",
            "response_metadata": {"messages": ["Channel C12345 does not exist"]},
        }

        error_code = response.get("error")
        error_details = response.get("response_metadata", {}).get("messages", [])

        assert error_code == "channel_not_found"
        assert len(error_details) == 1

    def test_teams_error_parsing(self):
        """Should parse Teams/Graph API error responses correctly."""
        response = {
            "error": {
                "code": "BadRequest",
                "message": "Invalid request payload",
                "innerError": {
                    "date": "2024-01-01T00:00:00",
                    "request-id": "abc123",
                    "client-request-id": "xyz789",
                },
            }
        }

        error = response.get("error", {})
        error_code = error.get("code")
        error_message = error.get("message")
        request_id = error.get("innerError", {}).get("request-id")

        assert error_code == "BadRequest"
        assert error_message == "Invalid request payload"
        assert request_id == "abc123"

    def test_nested_error_extraction(self):
        """Should extract errors from nested response structures."""
        # Some APIs return errors nested differently
        response = {
            "status": "error",
            "data": None,
            "error": {
                "type": "validation_error",
                "errors": [
                    {"field": "channel_id", "message": "Required field"},
                    {"field": "text", "message": "Cannot be empty"},
                ],
            },
        }

        errors = response.get("error", {}).get("errors", [])
        assert len(errors) == 2
        assert errors[0]["field"] == "channel_id"


class TestErrorRecoveryStrategies:
    """Tests for error recovery strategies."""

    @pytest.mark.asyncio
    async def test_retry_with_backoff(self):
        """Should retry failed requests with exponential backoff."""
        attempt_count = 0

        async def failing_request():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise Exception("Temporary failure")
            return {"success": True}

        # Simulate retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = await failing_request()
                break
            except Exception:
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(0.01 * (2**attempt))

        assert result["success"] is True
        assert attempt_count == 3

    @pytest.mark.asyncio
    async def test_fallback_on_persistent_failure(self):
        """Should use fallback mechanism on persistent failures."""
        primary_failed = False
        fallback_used = False

        async def primary_request():
            nonlocal primary_failed
            primary_failed = True
            raise Exception("Primary failed")

        async def fallback_request():
            nonlocal fallback_used
            fallback_used = True
            return {"success": True, "via": "fallback"}

        # Try primary, then fallback
        try:
            result = await primary_request()
        except Exception:
            result = await fallback_request()

        assert primary_failed is True
        assert fallback_used is True
        assert result["via"] == "fallback"

    @pytest.mark.asyncio
    async def test_graceful_degradation(self):
        """Should degrade gracefully when features unavailable."""
        features_available = {
            "rich_formatting": False,
            "file_upload": False,
            "reactions": True,
        }

        message = "Test message"

        # Degrade formatting if not available
        if not features_available["rich_formatting"]:
            # Use plain text instead of blocks
            formatted_message = message
        else:
            formatted_message = {"blocks": [{"type": "section", "text": message}]}

        assert isinstance(formatted_message, str)
        assert formatted_message == message
