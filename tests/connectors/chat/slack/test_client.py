"""
Tests for Slack client module - API utilities, rate limiting, and error classification.

Tests cover:
- Error classification (_classify_slack_error)
- Retryable error detection (_is_retryable_error)
- Exponential backoff behavior
- Rate limit header handling
- WorkspaceRateLimit tracking
- WorkspaceRateLimitRegistry management
- Global registry access
- Rate limit state transitions
- Best workspace selection for load balancing
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Error Classification Tests
# ---------------------------------------------------------------------------


class TestClassifySlackError:
    """Tests for _classify_slack_error function."""

    def test_rate_limit_error_status_429(self):
        """Should classify 429 status as rate limit error."""
        from aragora.connectors.chat.slack.client import _classify_slack_error
        from aragora.connectors.exceptions import ConnectorRateLimitError

        error = _classify_slack_error("any error", status_code=429)
        assert isinstance(error, ConnectorRateLimitError)
        assert error.connector_name == "slack"

    def test_rate_limit_error_by_message(self):
        """Should classify error with 'rate' in message as rate limit."""
        from aragora.connectors.chat.slack.client import _classify_slack_error
        from aragora.connectors.exceptions import ConnectorRateLimitError

        error = _classify_slack_error("rate_limited")
        assert isinstance(error, ConnectorRateLimitError)

    def test_rate_limit_error_ratelimited(self):
        """Should classify 'ratelimited' error as rate limit."""
        from aragora.connectors.chat.slack.client import _classify_slack_error
        from aragora.connectors.exceptions import ConnectorRateLimitError

        error = _classify_slack_error("ratelimited")
        assert isinstance(error, ConnectorRateLimitError)

    def test_rate_limit_includes_retry_after(self):
        """Should include retry_after in rate limit error."""
        from aragora.connectors.chat.slack.client import _classify_slack_error
        from aragora.connectors.exceptions import ConnectorRateLimitError

        error = _classify_slack_error("rate_limited", retry_after=30.0)
        assert isinstance(error, ConnectorRateLimitError)
        assert error.retry_after == 30.0

    def test_auth_error_invalid_auth(self):
        """Should classify invalid_auth as auth error."""
        from aragora.connectors.chat.slack.client import _classify_slack_error
        from aragora.connectors.exceptions import ConnectorAuthError

        error = _classify_slack_error("invalid_auth")
        assert isinstance(error, ConnectorAuthError)

    def test_auth_error_token_expired(self):
        """Should classify token_expired as auth error."""
        from aragora.connectors.chat.slack.client import _classify_slack_error
        from aragora.connectors.exceptions import ConnectorAuthError

        error = _classify_slack_error("token_expired")
        assert isinstance(error, ConnectorAuthError)

    def test_auth_error_token_revoked(self):
        """Should classify token_revoked as auth error."""
        from aragora.connectors.chat.slack.client import _classify_slack_error
        from aragora.connectors.exceptions import ConnectorAuthError

        error = _classify_slack_error("token_revoked")
        assert isinstance(error, ConnectorAuthError)

    def test_auth_error_not_authed(self):
        """Should classify not_authed as auth error."""
        from aragora.connectors.chat.slack.client import _classify_slack_error
        from aragora.connectors.exceptions import ConnectorAuthError

        error = _classify_slack_error("not_authed")
        assert isinstance(error, ConnectorAuthError)

    def test_auth_error_account_inactive(self):
        """Should classify account_inactive as auth error."""
        from aragora.connectors.chat.slack.client import _classify_slack_error
        from aragora.connectors.exceptions import ConnectorAuthError

        error = _classify_slack_error("account_inactive")
        assert isinstance(error, ConnectorAuthError)

    def test_timeout_error(self):
        """Should classify timeout error correctly."""
        from aragora.connectors.chat.slack.client import _classify_slack_error
        from aragora.connectors.exceptions import ConnectorTimeoutError

        error = _classify_slack_error("Request timeout after 30s")
        assert isinstance(error, ConnectorTimeoutError)

    def test_network_error_connection(self):
        """Should classify connection error as network error."""
        from aragora.connectors.chat.slack.client import _classify_slack_error
        from aragora.connectors.exceptions import ConnectorNetworkError

        error = _classify_slack_error("Connection refused")
        assert isinstance(error, ConnectorNetworkError)

    def test_network_error_network_keyword(self):
        """Should classify network error by keyword."""
        from aragora.connectors.chat.slack.client import _classify_slack_error
        from aragora.connectors.exceptions import ConnectorNetworkError

        error = _classify_slack_error("Network unreachable")
        assert isinstance(error, ConnectorNetworkError)

    def test_server_error_500(self):
        """Should classify 500 status as API error."""
        from aragora.connectors.chat.slack.client import _classify_slack_error
        from aragora.connectors.exceptions import ConnectorAPIError

        error = _classify_slack_error("Internal server error", status_code=500)
        assert isinstance(error, ConnectorAPIError)
        assert error.status_code == 500

    def test_server_error_503(self):
        """Should classify 503 status as API error."""
        from aragora.connectors.chat.slack.client import _classify_slack_error
        from aragora.connectors.exceptions import ConnectorAPIError

        error = _classify_slack_error("Service unavailable", status_code=503)
        assert isinstance(error, ConnectorAPIError)
        assert error.status_code == 503

    def test_generic_api_error(self):
        """Should default to API error for unknown errors."""
        from aragora.connectors.chat.slack.client import _classify_slack_error
        from aragora.connectors.exceptions import ConnectorAPIError

        error = _classify_slack_error("channel_not_found", status_code=404)
        assert isinstance(error, ConnectorAPIError)
        assert error.status_code == 404

    def test_generic_error_without_status(self):
        """Should handle error without status code."""
        from aragora.connectors.chat.slack.client import _classify_slack_error
        from aragora.connectors.exceptions import ConnectorAPIError

        error = _classify_slack_error("unknown_error")
        assert isinstance(error, ConnectorAPIError)
        assert error.status_code is None


# ---------------------------------------------------------------------------
# Retryable Error Detection Tests
# ---------------------------------------------------------------------------


class TestIsRetryableError:
    """Tests for _is_retryable_error function."""

    def test_rate_limit_429_is_retryable(self):
        """429 status should be retryable."""
        from aragora.connectors.chat.slack.client import _is_retryable_error

        assert _is_retryable_error(429) is True

    def test_server_error_500_is_retryable(self):
        """500 status should be retryable."""
        from aragora.connectors.chat.slack.client import _is_retryable_error

        assert _is_retryable_error(500) is True

    def test_server_error_502_is_retryable(self):
        """502 status should be retryable."""
        from aragora.connectors.chat.slack.client import _is_retryable_error

        assert _is_retryable_error(502) is True

    def test_server_error_503_is_retryable(self):
        """503 status should be retryable."""
        from aragora.connectors.chat.slack.client import _is_retryable_error

        assert _is_retryable_error(503) is True

    def test_server_error_504_is_retryable(self):
        """504 status should be retryable."""
        from aragora.connectors.chat.slack.client import _is_retryable_error

        assert _is_retryable_error(504) is True

    def test_client_error_400_not_retryable(self):
        """400 status should not be retryable."""
        from aragora.connectors.chat.slack.client import _is_retryable_error

        assert _is_retryable_error(400) is False

    def test_client_error_404_not_retryable(self):
        """404 status should not be retryable."""
        from aragora.connectors.chat.slack.client import _is_retryable_error

        assert _is_retryable_error(404) is False

    def test_service_unavailable_error_string_retryable(self):
        """service_unavailable error string should be retryable."""
        from aragora.connectors.chat.slack.client import _is_retryable_error

        assert _is_retryable_error(200, "service_unavailable") is True

    def test_timeout_error_string_retryable(self):
        """timeout error string should be retryable."""
        from aragora.connectors.chat.slack.client import _is_retryable_error

        assert _is_retryable_error(200, "timeout") is True

    def test_internal_error_string_retryable(self):
        """internal_error error string should be retryable."""
        from aragora.connectors.chat.slack.client import _is_retryable_error

        assert _is_retryable_error(200, "internal_error") is True

    def test_fatal_error_string_retryable(self):
        """fatal_error error string should be retryable."""
        from aragora.connectors.chat.slack.client import _is_retryable_error

        assert _is_retryable_error(200, "fatal_error") is True

    def test_channel_not_found_not_retryable(self):
        """channel_not_found error should not be retryable."""
        from aragora.connectors.chat.slack.client import _is_retryable_error

        assert _is_retryable_error(200, "channel_not_found") is False


# ---------------------------------------------------------------------------
# Exponential Backoff Tests
# ---------------------------------------------------------------------------


class TestExponentialBackoff:
    """Tests for _exponential_backoff function."""

    @pytest.mark.asyncio
    async def test_backoff_first_attempt_short(self):
        """First attempt should have short delay."""
        from aragora.connectors.chat.slack.client import _exponential_backoff

        start = time.time()
        await _exponential_backoff(0, base=0.01, max_delay=1.0)
        elapsed = time.time() - start

        # Should be around 0.01-0.02 seconds (base + jitter)
        assert elapsed < 0.1

    @pytest.mark.asyncio
    async def test_backoff_increases_with_attempts(self):
        """Delay should increase with attempt number."""
        from aragora.connectors.chat.slack.client import _exponential_backoff

        start1 = time.time()
        await _exponential_backoff(0, base=0.01, max_delay=1.0)
        elapsed1 = time.time() - start1

        start2 = time.time()
        await _exponential_backoff(2, base=0.01, max_delay=1.0)
        elapsed2 = time.time() - start2

        # Second delay should be longer (exponential growth)
        assert elapsed2 > elapsed1

    @pytest.mark.asyncio
    async def test_backoff_respects_max_delay(self):
        """Should not exceed max_delay."""
        from aragora.connectors.chat.slack.client import _exponential_backoff

        start = time.time()
        await _exponential_backoff(10, base=0.01, max_delay=0.05)
        elapsed = time.time() - start

        # Should be capped at max_delay + small overhead
        assert elapsed < 0.1


# ---------------------------------------------------------------------------
# Rate Limit Header Handling Tests
# ---------------------------------------------------------------------------


class TestWaitForRateLimit:
    """Tests for _wait_for_rate_limit function."""

    @pytest.mark.asyncio
    async def test_uses_retry_after_header(self):
        """Should use Retry-After header when present."""
        from aragora.connectors.chat.slack.client import _wait_for_rate_limit

        mock_response = MagicMock()
        mock_response.headers = {"Retry-After": "1"}

        start = time.time()
        await _wait_for_rate_limit(mock_response, attempt=0, max_delay=2.0)
        elapsed = time.time() - start

        # Should wait approximately 1 second
        assert 0.9 < elapsed < 1.5

    @pytest.mark.asyncio
    async def test_caps_retry_after_at_max_delay(self):
        """Should cap Retry-After at max_delay."""
        from aragora.connectors.chat.slack.client import _wait_for_rate_limit

        mock_response = MagicMock()
        mock_response.headers = {"Retry-After": "100"}

        start = time.time()
        await _wait_for_rate_limit(mock_response, attempt=0, max_delay=0.05)
        elapsed = time.time() - start

        # Should be capped at max_delay
        assert elapsed < 0.1

    @pytest.mark.asyncio
    async def test_fallback_to_exponential_backoff(self):
        """Should fallback to exponential backoff without Retry-After."""
        from aragora.connectors.chat.slack.client import _wait_for_rate_limit

        mock_response = MagicMock()
        mock_response.headers = {}

        start = time.time()
        await _wait_for_rate_limit(mock_response, attempt=0, base=0.01, max_delay=0.1)
        elapsed = time.time() - start

        # Should use exponential backoff
        assert elapsed < 0.2

    @pytest.mark.asyncio
    async def test_handles_invalid_retry_after(self):
        """Should handle invalid Retry-After header gracefully."""
        from aragora.connectors.chat.slack.client import _wait_for_rate_limit

        mock_response = MagicMock()
        mock_response.headers = {"Retry-After": "not-a-number"}

        # Should not raise, falls back to exponential backoff
        await _wait_for_rate_limit(mock_response, attempt=0, base=0.01, max_delay=0.1)


# ---------------------------------------------------------------------------
# WorkspaceRateLimit Tests
# ---------------------------------------------------------------------------


class TestWorkspaceRateLimit:
    """Tests for WorkspaceRateLimit dataclass."""

    def test_default_initialization(self):
        """Should initialize with default values."""
        from aragora.connectors.chat.slack.client import WorkspaceRateLimit

        limit = WorkspaceRateLimit(workspace_id="W123")

        assert limit.workspace_id == "W123"
        assert limit.limit == 50
        assert limit.remaining == 50
        assert limit.reset_at == 0.0

    def test_update_from_headers(self):
        """Should update from API response headers."""
        from aragora.connectors.chat.slack.client import WorkspaceRateLimit

        limit = WorkspaceRateLimit(workspace_id="W123")
        headers = {
            "X-Rate-Limit-Limit": "100",
            "X-Rate-Limit-Remaining": "45",
            "X-Rate-Limit-Reset": "1704067200.0",
        }

        limit.update_from_headers(headers)

        assert limit.limit == 100
        assert limit.remaining == 45
        assert limit.reset_at == 1704067200.0

    def test_is_rate_limited_when_no_remaining(self):
        """Should be rate limited when remaining is 0 and before reset."""
        from aragora.connectors.chat.slack.client import WorkspaceRateLimit

        future_reset = time.time() + 60
        limit = WorkspaceRateLimit(
            workspace_id="W123",
            remaining=0,
            reset_at=future_reset,
        )

        assert limit.is_rate_limited is True

    def test_not_rate_limited_when_has_remaining(self):
        """Should not be rate limited when remaining > 0."""
        from aragora.connectors.chat.slack.client import WorkspaceRateLimit

        limit = WorkspaceRateLimit(
            workspace_id="W123",
            remaining=10,
            reset_at=time.time() + 60,
        )

        assert limit.is_rate_limited is False

    def test_not_rate_limited_after_reset(self):
        """Should not be rate limited after reset time."""
        from aragora.connectors.chat.slack.client import WorkspaceRateLimit

        past_reset = time.time() - 60
        limit = WorkspaceRateLimit(
            workspace_id="W123",
            remaining=0,
            reset_at=past_reset,
        )

        assert limit.is_rate_limited is False

    def test_seconds_until_reset_future(self):
        """Should calculate seconds until future reset."""
        from aragora.connectors.chat.slack.client import WorkspaceRateLimit

        future_reset = time.time() + 30
        limit = WorkspaceRateLimit(
            workspace_id="W123",
            reset_at=future_reset,
        )

        seconds = limit.seconds_until_reset
        assert 29 < seconds < 31

    def test_seconds_until_reset_past(self):
        """Should return 0 for past reset time."""
        from aragora.connectors.chat.slack.client import WorkspaceRateLimit

        past_reset = time.time() - 60
        limit = WorkspaceRateLimit(
            workspace_id="W123",
            reset_at=past_reset,
        )

        assert limit.seconds_until_reset == 0.0

    def test_to_dict(self):
        """Should serialize to dictionary."""
        from aragora.connectors.chat.slack.client import WorkspaceRateLimit

        limit = WorkspaceRateLimit(
            workspace_id="W123",
            limit=100,
            remaining=45,
            reset_at=1704067200.0,
        )

        result = limit.to_dict()

        assert result["workspace_id"] == "W123"
        assert result["limit"] == 100
        assert result["remaining"] == 45
        assert result["reset_at"] == 1704067200.0
        assert "is_rate_limited" in result
        assert "seconds_until_reset" in result


# ---------------------------------------------------------------------------
# WorkspaceRateLimitRegistry Tests
# ---------------------------------------------------------------------------


class TestWorkspaceRateLimitRegistry:
    """Tests for WorkspaceRateLimitRegistry class."""

    def test_get_creates_new_entry(self):
        """Should create new entry for unknown workspace."""
        from aragora.connectors.chat.slack.client import WorkspaceRateLimitRegistry

        registry = WorkspaceRateLimitRegistry()
        limit = registry.get("W_NEW")

        assert limit.workspace_id == "W_NEW"
        assert limit.limit == 50  # Default

    def test_get_returns_existing_entry(self):
        """Should return existing entry for known workspace."""
        from aragora.connectors.chat.slack.client import WorkspaceRateLimitRegistry

        registry = WorkspaceRateLimitRegistry()
        limit1 = registry.get("W_EXISTING")
        limit1.remaining = 25
        limit2 = registry.get("W_EXISTING")

        assert limit2.remaining == 25
        assert limit1 is limit2

    def test_update_creates_and_updates(self):
        """Should create and update workspace entry."""
        from aragora.connectors.chat.slack.client import WorkspaceRateLimitRegistry

        registry = WorkspaceRateLimitRegistry()
        headers = {"X-Rate-Limit-Remaining": "30"}

        limit = registry.update("W_UPDATE", headers)

        assert limit.remaining == 30

    def test_get_best_workspace_empty_list(self):
        """Should return None for empty workspace list."""
        from aragora.connectors.chat.slack.client import WorkspaceRateLimitRegistry

        registry = WorkspaceRateLimitRegistry()
        result = registry.get_best_workspace([])

        assert result is None

    def test_get_best_workspace_selects_highest_remaining(self):
        """Should select workspace with highest remaining quota."""
        from aragora.connectors.chat.slack.client import WorkspaceRateLimitRegistry

        registry = WorkspaceRateLimitRegistry()

        # Set up workspaces with different remaining quotas
        limit1 = registry.get("W1")
        limit1.remaining = 10

        limit2 = registry.get("W2")
        limit2.remaining = 50

        limit3 = registry.get("W3")
        limit3.remaining = 25

        best = registry.get_best_workspace(["W1", "W2", "W3"])

        assert best == "W2"

    def test_get_best_workspace_excludes_rate_limited(self):
        """Should exclude rate limited workspaces."""
        from aragora.connectors.chat.slack.client import WorkspaceRateLimitRegistry

        registry = WorkspaceRateLimitRegistry()

        # W1 is rate limited (remaining=0, reset in future)
        limit1 = registry.get("W1")
        limit1.remaining = 0
        limit1.reset_at = time.time() + 60

        # W2 has some remaining
        limit2 = registry.get("W2")
        limit2.remaining = 5

        best = registry.get_best_workspace(["W1", "W2"])

        assert best == "W2"

    def test_get_all_stats(self):
        """Should return stats for all tracked workspaces."""
        from aragora.connectors.chat.slack.client import WorkspaceRateLimitRegistry

        registry = WorkspaceRateLimitRegistry()
        registry.get("W1")
        registry.get("W2")

        stats = registry.get_all_stats()

        assert "W1" in stats
        assert "W2" in stats
        assert stats["W1"]["workspace_id"] == "W1"


# ---------------------------------------------------------------------------
# Global Registry Tests
# ---------------------------------------------------------------------------


class TestGlobalRegistry:
    """Tests for global rate limit registry access."""

    def test_get_rate_limit_registry_returns_singleton(self):
        """Should return the same registry instance."""
        from aragora.connectors.chat.slack.client import get_rate_limit_registry

        registry1 = get_rate_limit_registry()
        registry2 = get_rate_limit_registry()

        assert registry1 is registry2

    def test_get_rate_limit_registry_is_functional(self):
        """Should return a functional registry."""
        from aragora.connectors.chat.slack.client import get_rate_limit_registry

        registry = get_rate_limit_registry()
        limit = registry.get("W_GLOBAL_TEST")

        assert limit.workspace_id == "W_GLOBAL_TEST"
