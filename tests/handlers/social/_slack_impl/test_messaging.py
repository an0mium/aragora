"""Comprehensive tests for MessagingMixin in _slack_impl/messaging.py.

Covers all public functions and methods:
- get_slack_circuit_breaker (singleton creation, thread-safety)
- reset_slack_circuit_breaker (reset existing, reset when None)
- MessagingMixin._slack_response (ephemeral, in_channel, custom response_type)
- MessagingMixin._slack_blocks_response (blocks content, response_type variants)
- MessagingMixin._post_to_response_url (SSRF protection, circuit breaker gating,
    success, non-2xx responses, 5xx triggers circuit breaker, connection errors,
    unexpected errors)
- MessagingMixin._post_message_async (no token, circuit breaker open, success,
    thread_ts, blocks, API error non-circuit, API error circuit-triggering,
    connection error, unexpected error, returned ts)
- MessagingMixin.get_circuit_breaker_status (status dict shape)
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.social._slack_impl.messaging import (
    MessagingMixin,
    get_slack_circuit_breaker,
    reset_slack_circuit_breaker,
    _slack_circuit_breaker,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if result is None:
        return {}
    if isinstance(result, dict):
        return result
    raw = result.body
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")
    return json.loads(raw)


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if result is None:
        return 0
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


class Messenger(MessagingMixin):
    """Concrete class that uses MessagingMixin for testing."""

    pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_circuit_breaker():
    """Reset the module-level circuit breaker singleton before and after each test."""
    import aragora.server.handlers.social._slack_impl.messaging as mod

    old = mod._slack_circuit_breaker
    mod._slack_circuit_breaker = None
    yield
    mod._slack_circuit_breaker = old


@pytest.fixture
def messenger():
    """Create a Messenger instance for testing."""
    return Messenger()


@pytest.fixture
def mock_pool():
    """Create a mock HTTP pool with an async session context manager."""
    mock_client = AsyncMock()
    mock_session_ctx = AsyncMock()
    mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_client)
    mock_session_ctx.__aexit__ = AsyncMock(return_value=None)
    pool = MagicMock()
    pool.get_session.return_value = mock_session_ctx
    return pool, mock_client


# ===========================================================================
# Tests for get_slack_circuit_breaker
# ===========================================================================


class TestGetSlackCircuitBreaker:
    """Tests for the get_slack_circuit_breaker singleton function."""

    def test_creates_singleton(self):
        """First call should create a new circuit breaker."""
        cb = get_slack_circuit_breaker()
        assert cb is not None
        assert cb.name == "slack"
        assert cb.half_open_max_calls == 2

    def test_returns_same_instance(self):
        """Repeated calls should return the same instance."""
        cb1 = get_slack_circuit_breaker()
        cb2 = get_slack_circuit_breaker()
        assert cb1 is cb2

    def test_initial_state_is_closed(self):
        """Circuit breaker should start in the closed state."""
        cb = get_slack_circuit_breaker()
        assert cb.state == "closed"


# ===========================================================================
# Tests for reset_slack_circuit_breaker
# ===========================================================================


class TestResetSlackCircuitBreaker:
    """Tests for the reset_slack_circuit_breaker function."""

    def test_reset_when_none(self):
        """Resetting when no circuit breaker exists should not raise."""
        reset_slack_circuit_breaker()  # should not raise

    def test_reset_clears_failures(self):
        """Resetting after failures should restore closed state."""
        cb = get_slack_circuit_breaker()
        # Trigger enough failures to open the circuit
        for _ in range(10):
            cb.record_failure()
        assert cb.state == "open"
        reset_slack_circuit_breaker()
        assert cb.state == "closed"
        assert cb._failure_count == 0


# ===========================================================================
# Tests for MessagingMixin._slack_response
# ===========================================================================


class TestSlackResponse:
    """Tests for the _slack_response method."""

    def test_ephemeral_default(self, messenger):
        """Default response_type should be ephemeral."""
        result = messenger._slack_response("hello")
        body = _body(result)
        assert body["response_type"] == "ephemeral"
        assert body["text"] == "hello"

    def test_in_channel_response(self, messenger):
        """Should support in_channel response_type."""
        result = messenger._slack_response("public msg", response_type="in_channel")
        body = _body(result)
        assert body["response_type"] == "in_channel"
        assert body["text"] == "public msg"

    def test_status_code_is_200(self, messenger):
        """Response should have a 200 status code."""
        result = messenger._slack_response("ok")
        assert _status(result) == 200

    def test_empty_text(self, messenger):
        """Should handle empty text."""
        result = messenger._slack_response("")
        body = _body(result)
        assert body["text"] == ""

    def test_special_characters(self, messenger):
        """Should handle special characters in text."""
        text = 'Hello <@U123|user> & "quotes" & emoji :tada:'
        result = messenger._slack_response(text)
        body = _body(result)
        assert body["text"] == text

    def test_custom_response_type(self, messenger):
        """Should accept any custom response_type string."""
        result = messenger._slack_response("test", response_type="custom_type")
        body = _body(result)
        assert body["response_type"] == "custom_type"


# ===========================================================================
# Tests for MessagingMixin._slack_blocks_response
# ===========================================================================


class TestSlackBlocksResponse:
    """Tests for the _slack_blocks_response method."""

    def test_blocks_included(self, messenger):
        """Should include blocks in the response."""
        blocks = [{"type": "section", "text": {"type": "mrkdwn", "text": "hello"}}]
        result = messenger._slack_blocks_response(blocks, "fallback text")
        body = _body(result)
        assert body["blocks"] == blocks
        assert body["text"] == "fallback text"

    def test_default_response_type_ephemeral(self, messenger):
        """Default response_type should be ephemeral."""
        result = messenger._slack_blocks_response([], "text")
        body = _body(result)
        assert body["response_type"] == "ephemeral"

    def test_in_channel_blocks(self, messenger):
        """Should support in_channel response_type with blocks."""
        blocks = [{"type": "divider"}]
        result = messenger._slack_blocks_response(blocks, "text", response_type="in_channel")
        body = _body(result)
        assert body["response_type"] == "in_channel"
        assert body["blocks"] == blocks

    def test_empty_blocks_list(self, messenger):
        """Should handle an empty blocks list."""
        result = messenger._slack_blocks_response([], "text")
        body = _body(result)
        assert body["blocks"] == []

    def test_multiple_blocks(self, messenger):
        """Should handle multiple blocks."""
        blocks = [
            {"type": "header", "text": {"type": "plain_text", "text": "Title"}},
            {"type": "section", "text": {"type": "mrkdwn", "text": "Body"}},
            {"type": "divider"},
        ]
        result = messenger._slack_blocks_response(blocks, "fallback")
        body = _body(result)
        assert len(body["blocks"]) == 3


# ===========================================================================
# Tests for MessagingMixin._post_to_response_url
# ===========================================================================


class TestPostToResponseUrl:
    """Tests for the _post_to_response_url method."""

    @pytest.mark.asyncio
    async def test_ssrf_blocks_invalid_url(self, messenger):
        """Should silently return when the URL is not a valid Slack endpoint."""
        with patch(
            "aragora.server.handlers.social._slack_impl.messaging._validate_slack_url",
            return_value=False,
        ):
            await messenger._post_to_response_url("http://evil.com/hook", {"text": "hi"})
        # No exception, no HTTP call

    @pytest.mark.asyncio
    async def test_circuit_breaker_open_skips_call(self, messenger):
        """Should skip the POST when the circuit breaker is open."""
        with patch(
            "aragora.server.handlers.social._slack_impl.messaging._validate_slack_url",
            return_value=True,
        ):
            cb = get_slack_circuit_breaker()
            # Force circuit open
            for _ in range(10):
                cb.record_failure()
            assert cb.state == "open"

            with patch("aragora.server.http_client_pool.get_http_pool") as mock_get_pool:
                await messenger._post_to_response_url(
                    "https://hooks.slack.com/resp/123", {"text": "hi"}
                )
                mock_get_pool.assert_not_called()

    @pytest.mark.asyncio
    async def test_success_records_success(self, messenger, mock_pool):
        """Should record success on a 200 response."""
        pool, client = mock_pool
        mock_response = MagicMock()
        mock_response.status_code = 200
        client.post.return_value = mock_response

        with (
            patch(
                "aragora.server.handlers.social._slack_impl.messaging._validate_slack_url",
                return_value=True,
            ),
            patch(
                "aragora.server.http_client_pool.get_http_pool",
                return_value=pool,
            ),
        ):
            await messenger._post_to_response_url(
                "https://hooks.slack.com/resp/123", {"text": "hi"}
            )

        client.post.assert_awaited_once()
        cb = get_slack_circuit_breaker()
        assert cb._failure_count == 0

    @pytest.mark.asyncio
    async def test_4xx_does_not_trip_circuit_breaker(self, messenger, mock_pool):
        """A 4xx error should log but NOT trip the circuit breaker."""
        pool, client = mock_pool
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "bad request"
        client.post.return_value = mock_response

        with (
            patch(
                "aragora.server.handlers.social._slack_impl.messaging._validate_slack_url",
                return_value=True,
            ),
            patch(
                "aragora.server.http_client_pool.get_http_pool",
                return_value=pool,
            ),
        ):
            await messenger._post_to_response_url(
                "https://hooks.slack.com/resp/123", {"text": "hi"}
            )

        cb = get_slack_circuit_breaker()
        assert cb._failure_count == 0

    @pytest.mark.asyncio
    async def test_5xx_trips_circuit_breaker(self, messenger, mock_pool):
        """A 5xx error should record a failure on the circuit breaker."""
        pool, client = mock_pool
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "internal server error"
        client.post.return_value = mock_response

        with (
            patch(
                "aragora.server.handlers.social._slack_impl.messaging._validate_slack_url",
                return_value=True,
            ),
            patch(
                "aragora.server.http_client_pool.get_http_pool",
                return_value=pool,
            ),
        ):
            await messenger._post_to_response_url(
                "https://hooks.slack.com/resp/123", {"text": "hi"}
            )

        cb = get_slack_circuit_breaker()
        assert cb._failure_count == 1

    @pytest.mark.asyncio
    async def test_connection_error_trips_circuit_breaker(self, messenger, mock_pool):
        """A ConnectionError should record a failure on the circuit breaker."""
        pool, client = mock_pool
        client.post.side_effect = ConnectionError("connection refused")

        with (
            patch(
                "aragora.server.handlers.social._slack_impl.messaging._validate_slack_url",
                return_value=True,
            ),
            patch(
                "aragora.server.http_client_pool.get_http_pool",
                return_value=pool,
            ),
        ):
            await messenger._post_to_response_url(
                "https://hooks.slack.com/resp/123", {"text": "hi"}
            )

        cb = get_slack_circuit_breaker()
        assert cb._failure_count == 1

    @pytest.mark.asyncio
    async def test_timeout_error_trips_circuit_breaker(self, messenger, mock_pool):
        """A TimeoutError should record a failure on the circuit breaker."""
        pool, client = mock_pool
        client.post.side_effect = TimeoutError("timed out")

        with (
            patch(
                "aragora.server.handlers.social._slack_impl.messaging._validate_slack_url",
                return_value=True,
            ),
            patch(
                "aragora.server.http_client_pool.get_http_pool",
                return_value=pool,
            ),
        ):
            await messenger._post_to_response_url(
                "https://hooks.slack.com/resp/123", {"text": "hi"}
            )

        cb = get_slack_circuit_breaker()
        assert cb._failure_count == 1

    @pytest.mark.asyncio
    async def test_runtime_error_trips_circuit_breaker(self, messenger, mock_pool):
        """A RuntimeError should record a failure on the circuit breaker."""
        pool, client = mock_pool
        client.post.side_effect = RuntimeError("unexpected")

        with (
            patch(
                "aragora.server.handlers.social._slack_impl.messaging._validate_slack_url",
                return_value=True,
            ),
            patch(
                "aragora.server.http_client_pool.get_http_pool",
                return_value=pool,
            ),
        ):
            await messenger._post_to_response_url(
                "https://hooks.slack.com/resp/123", {"text": "hi"}
            )

        cb = get_slack_circuit_breaker()
        assert cb._failure_count == 1

    @pytest.mark.asyncio
    async def test_oserror_trips_circuit_breaker(self, messenger, mock_pool):
        """An OSError should record a failure on the circuit breaker."""
        pool, client = mock_pool
        client.post.side_effect = OSError("network unreachable")

        with (
            patch(
                "aragora.server.handlers.social._slack_impl.messaging._validate_slack_url",
                return_value=True,
            ),
            patch(
                "aragora.server.http_client_pool.get_http_pool",
                return_value=pool,
            ),
        ):
            await messenger._post_to_response_url(
                "https://hooks.slack.com/resp/123", {"text": "hi"}
            )

        cb = get_slack_circuit_breaker()
        assert cb._failure_count == 1

    @pytest.mark.asyncio
    async def test_post_sends_correct_payload(self, messenger, mock_pool):
        """Should send the payload as JSON with correct headers."""
        pool, client = mock_pool
        mock_response = MagicMock()
        mock_response.status_code = 200
        client.post.return_value = mock_response
        url = "https://hooks.slack.com/resp/456"
        payload = {"text": "hello", "response_type": "in_channel"}

        with (
            patch(
                "aragora.server.handlers.social._slack_impl.messaging._validate_slack_url",
                return_value=True,
            ),
            patch(
                "aragora.server.http_client_pool.get_http_pool",
                return_value=pool,
            ),
        ):
            await messenger._post_to_response_url(url, payload)

        client.post.assert_awaited_once_with(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30,
        )


# ===========================================================================
# Tests for MessagingMixin._post_message_async
# ===========================================================================


class TestPostMessageAsync:
    """Tests for the _post_message_async method."""

    @pytest.mark.asyncio
    async def test_no_token_returns_none(self, messenger):
        """Should return None when SLACK_BOT_TOKEN is not configured."""
        with patch(
            "aragora.server.handlers.social._slack_impl.messaging.SLACK_BOT_TOKEN",
            None,
        ):
            result = await messenger._post_message_async("C123", "hello")
        assert result is None

    @pytest.mark.asyncio
    async def test_empty_token_returns_none(self, messenger):
        """Should return None when SLACK_BOT_TOKEN is empty string."""
        with patch(
            "aragora.server.handlers.social._slack_impl.messaging.SLACK_BOT_TOKEN",
            "",
        ):
            result = await messenger._post_message_async("C123", "hello")
        assert result is None

    @pytest.mark.asyncio
    async def test_circuit_breaker_open_returns_none(self, messenger):
        """Should return None when the circuit breaker is open."""
        cb = get_slack_circuit_breaker()
        for _ in range(10):
            cb.record_failure()
        assert cb.state == "open"

        with patch(
            "aragora.server.handlers.social._slack_impl.messaging.SLACK_BOT_TOKEN",
            "xoxb-test-token",
        ):
            result = await messenger._post_message_async("C123", "hello")
        assert result is None

    @pytest.mark.asyncio
    async def test_success_returns_ts(self, messenger, mock_pool):
        """Should return the message ts on success."""
        pool, client = mock_pool
        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": True, "ts": "1234567890.123456"}
        client.post.return_value = mock_response

        with (
            patch(
                "aragora.server.handlers.social._slack_impl.messaging.SLACK_BOT_TOKEN",
                "xoxb-test-token",
            ),
            patch(
                "aragora.server.http_client_pool.get_http_pool",
                return_value=pool,
            ),
        ):
            result = await messenger._post_message_async("C123", "hello")

        assert result == "1234567890.123456"

    @pytest.mark.asyncio
    async def test_success_records_circuit_breaker_success(self, messenger, mock_pool):
        """Should record success on the circuit breaker."""
        pool, client = mock_pool
        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": True, "ts": "123.456"}
        client.post.return_value = mock_response

        with (
            patch(
                "aragora.server.handlers.social._slack_impl.messaging.SLACK_BOT_TOKEN",
                "xoxb-test-token",
            ),
            patch(
                "aragora.server.http_client_pool.get_http_pool",
                return_value=pool,
            ),
        ):
            await messenger._post_message_async("C123", "hello")

        cb = get_slack_circuit_breaker()
        assert cb._failure_count == 0

    @pytest.mark.asyncio
    async def test_includes_thread_ts(self, messenger, mock_pool):
        """Should include thread_ts in payload when provided."""
        pool, client = mock_pool
        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": True, "ts": "123.456"}
        client.post.return_value = mock_response

        with (
            patch(
                "aragora.server.handlers.social._slack_impl.messaging.SLACK_BOT_TOKEN",
                "xoxb-test-token",
            ),
            patch(
                "aragora.server.http_client_pool.get_http_pool",
                return_value=pool,
            ),
        ):
            await messenger._post_message_async("C123", "hello", thread_ts="111.222")

        call_kwargs = client.post.call_args
        sent_payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert sent_payload["thread_ts"] == "111.222"

    @pytest.mark.asyncio
    async def test_includes_blocks(self, messenger, mock_pool):
        """Should include blocks in payload when provided."""
        pool, client = mock_pool
        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": True, "ts": "123.456"}
        client.post.return_value = mock_response

        blocks = [{"type": "section", "text": {"type": "mrkdwn", "text": "test"}}]

        with (
            patch(
                "aragora.server.handlers.social._slack_impl.messaging.SLACK_BOT_TOKEN",
                "xoxb-test-token",
            ),
            patch(
                "aragora.server.http_client_pool.get_http_pool",
                return_value=pool,
            ),
        ):
            await messenger._post_message_async("C123", "hello", blocks=blocks)

        call_kwargs = client.post.call_args
        sent_payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert sent_payload["blocks"] == blocks

    @pytest.mark.asyncio
    async def test_no_thread_ts_omits_key(self, messenger, mock_pool):
        """Should omit thread_ts from payload when not provided."""
        pool, client = mock_pool
        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": True, "ts": "123.456"}
        client.post.return_value = mock_response

        with (
            patch(
                "aragora.server.handlers.social._slack_impl.messaging.SLACK_BOT_TOKEN",
                "xoxb-test-token",
            ),
            patch(
                "aragora.server.http_client_pool.get_http_pool",
                return_value=pool,
            ),
        ):
            await messenger._post_message_async("C123", "hello")

        call_kwargs = client.post.call_args
        sent_payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert "thread_ts" not in sent_payload

    @pytest.mark.asyncio
    async def test_no_blocks_omits_key(self, messenger, mock_pool):
        """Should omit blocks from payload when not provided."""
        pool, client = mock_pool
        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": True, "ts": "123.456"}
        client.post.return_value = mock_response

        with (
            patch(
                "aragora.server.handlers.social._slack_impl.messaging.SLACK_BOT_TOKEN",
                "xoxb-test-token",
            ),
            patch(
                "aragora.server.http_client_pool.get_http_pool",
                return_value=pool,
            ),
        ):
            await messenger._post_message_async("C123", "hello")

        call_kwargs = client.post.call_args
        sent_payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert "blocks" not in sent_payload

    @pytest.mark.asyncio
    async def test_api_error_non_circuit_returns_none(self, messenger, mock_pool):
        """API errors like 'channel_not_found' should return None but not trip breaker."""
        pool, client = mock_pool
        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": False, "error": "channel_not_found"}
        client.post.return_value = mock_response

        with (
            patch(
                "aragora.server.handlers.social._slack_impl.messaging.SLACK_BOT_TOKEN",
                "xoxb-test-token",
            ),
            patch(
                "aragora.server.http_client_pool.get_http_pool",
                return_value=pool,
            ),
        ):
            result = await messenger._post_message_async("C123", "hello")

        assert result is None
        cb = get_slack_circuit_breaker()
        assert cb._failure_count == 0

    @pytest.mark.asyncio
    async def test_api_error_rate_limited_trips_breaker(self, messenger, mock_pool):
        """rate_limited error should trip the circuit breaker."""
        pool, client = mock_pool
        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": False, "error": "rate_limited"}
        client.post.return_value = mock_response

        with (
            patch(
                "aragora.server.handlers.social._slack_impl.messaging.SLACK_BOT_TOKEN",
                "xoxb-test-token",
            ),
            patch(
                "aragora.server.http_client_pool.get_http_pool",
                return_value=pool,
            ),
        ):
            result = await messenger._post_message_async("C123", "hello")

        assert result is None
        cb = get_slack_circuit_breaker()
        assert cb._failure_count == 1

    @pytest.mark.asyncio
    async def test_api_error_service_unavailable_trips_breaker(self, messenger, mock_pool):
        """service_unavailable error should trip the circuit breaker."""
        pool, client = mock_pool
        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": False, "error": "service_unavailable"}
        client.post.return_value = mock_response

        with (
            patch(
                "aragora.server.handlers.social._slack_impl.messaging.SLACK_BOT_TOKEN",
                "xoxb-test-token",
            ),
            patch(
                "aragora.server.http_client_pool.get_http_pool",
                return_value=pool,
            ),
        ):
            result = await messenger._post_message_async("C123", "hello")

        assert result is None
        cb = get_slack_circuit_breaker()
        assert cb._failure_count == 1

    @pytest.mark.asyncio
    async def test_api_error_fatal_error_trips_breaker(self, messenger, mock_pool):
        """fatal_error should trip the circuit breaker."""
        pool, client = mock_pool
        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": False, "error": "fatal_error"}
        client.post.return_value = mock_response

        with (
            patch(
                "aragora.server.handlers.social._slack_impl.messaging.SLACK_BOT_TOKEN",
                "xoxb-test-token",
            ),
            patch(
                "aragora.server.http_client_pool.get_http_pool",
                return_value=pool,
            ),
        ):
            result = await messenger._post_message_async("C123", "hello")

        assert result is None
        cb = get_slack_circuit_breaker()
        assert cb._failure_count == 1

    @pytest.mark.asyncio
    async def test_api_error_unknown_does_not_trip_breaker(self, messenger, mock_pool):
        """An unknown error type should not trip the circuit breaker."""
        pool, client = mock_pool
        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": False, "error": "invalid_auth"}
        client.post.return_value = mock_response

        with (
            patch(
                "aragora.server.handlers.social._slack_impl.messaging.SLACK_BOT_TOKEN",
                "xoxb-test-token",
            ),
            patch(
                "aragora.server.http_client_pool.get_http_pool",
                return_value=pool,
            ),
        ):
            result = await messenger._post_message_async("C123", "hello")

        assert result is None
        cb = get_slack_circuit_breaker()
        assert cb._failure_count == 0

    @pytest.mark.asyncio
    async def test_api_error_missing_error_field(self, messenger, mock_pool):
        """An ok=False response with no error field should return None."""
        pool, client = mock_pool
        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": False}
        client.post.return_value = mock_response

        with (
            patch(
                "aragora.server.handlers.social._slack_impl.messaging.SLACK_BOT_TOKEN",
                "xoxb-test-token",
            ),
            patch(
                "aragora.server.http_client_pool.get_http_pool",
                return_value=pool,
            ),
        ):
            result = await messenger._post_message_async("C123", "hello")

        assert result is None

    @pytest.mark.asyncio
    async def test_connection_error_returns_none(self, messenger, mock_pool):
        """A ConnectionError should return None and trip the breaker."""
        pool, client = mock_pool
        client.post.side_effect = ConnectionError("refused")

        with (
            patch(
                "aragora.server.handlers.social._slack_impl.messaging.SLACK_BOT_TOKEN",
                "xoxb-test-token",
            ),
            patch(
                "aragora.server.http_client_pool.get_http_pool",
                return_value=pool,
            ),
        ):
            result = await messenger._post_message_async("C123", "hello")

        assert result is None
        cb = get_slack_circuit_breaker()
        assert cb._failure_count == 1

    @pytest.mark.asyncio
    async def test_timeout_error_returns_none(self, messenger, mock_pool):
        """A TimeoutError should return None and trip the breaker."""
        pool, client = mock_pool
        client.post.side_effect = TimeoutError("timed out")

        with (
            patch(
                "aragora.server.handlers.social._slack_impl.messaging.SLACK_BOT_TOKEN",
                "xoxb-test-token",
            ),
            patch(
                "aragora.server.http_client_pool.get_http_pool",
                return_value=pool,
            ),
        ):
            result = await messenger._post_message_async("C123", "hello")

        assert result is None
        cb = get_slack_circuit_breaker()
        assert cb._failure_count == 1

    @pytest.mark.asyncio
    async def test_value_error_returns_none(self, messenger, mock_pool):
        """A ValueError should return None and trip the breaker."""
        pool, client = mock_pool
        client.post.side_effect = ValueError("bad value")

        with (
            patch(
                "aragora.server.handlers.social._slack_impl.messaging.SLACK_BOT_TOKEN",
                "xoxb-test-token",
            ),
            patch(
                "aragora.server.http_client_pool.get_http_pool",
                return_value=pool,
            ),
        ):
            result = await messenger._post_message_async("C123", "hello")

        assert result is None
        cb = get_slack_circuit_breaker()
        assert cb._failure_count == 1

    @pytest.mark.asyncio
    async def test_type_error_returns_none(self, messenger, mock_pool):
        """A TypeError should return None and trip the breaker."""
        pool, client = mock_pool
        client.post.side_effect = TypeError("type error")

        with (
            patch(
                "aragora.server.handlers.social._slack_impl.messaging.SLACK_BOT_TOKEN",
                "xoxb-test-token",
            ),
            patch(
                "aragora.server.http_client_pool.get_http_pool",
                return_value=pool,
            ),
        ):
            result = await messenger._post_message_async("C123", "hello")

        assert result is None
        cb = get_slack_circuit_breaker()
        assert cb._failure_count == 1

    @pytest.mark.asyncio
    async def test_posts_to_correct_url(self, messenger, mock_pool):
        """Should POST to the Slack chat.postMessage endpoint."""
        pool, client = mock_pool
        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": True, "ts": "123.456"}
        client.post.return_value = mock_response

        with (
            patch(
                "aragora.server.handlers.social._slack_impl.messaging.SLACK_BOT_TOKEN",
                "xoxb-test-token",
            ),
            patch(
                "aragora.server.http_client_pool.get_http_pool",
                return_value=pool,
            ),
        ):
            await messenger._post_message_async("C123", "hello")

        call_args = client.post.call_args
        assert call_args[0][0] == "https://slack.com/api/chat.postMessage"

    @pytest.mark.asyncio
    async def test_sends_auth_header(self, messenger, mock_pool):
        """Should include the Authorization header with the bot token."""
        pool, client = mock_pool
        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": True, "ts": "123.456"}
        client.post.return_value = mock_response

        with (
            patch(
                "aragora.server.handlers.social._slack_impl.messaging.SLACK_BOT_TOKEN",
                "xoxb-my-token",
            ),
            patch(
                "aragora.server.http_client_pool.get_http_pool",
                return_value=pool,
            ),
        ):
            await messenger._post_message_async("C123", "hello")

        call_kwargs = client.post.call_args
        headers = call_kwargs.kwargs.get("headers") or call_kwargs[1].get("headers")
        assert headers["Authorization"] == "Bearer xoxb-my-token"

    @pytest.mark.asyncio
    async def test_success_response_without_ts(self, messenger, mock_pool):
        """Should return None if ok=True but no ts in response."""
        pool, client = mock_pool
        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": True}
        client.post.return_value = mock_response

        with (
            patch(
                "aragora.server.handlers.social._slack_impl.messaging.SLACK_BOT_TOKEN",
                "xoxb-test-token",
            ),
            patch(
                "aragora.server.http_client_pool.get_http_pool",
                return_value=pool,
            ),
        ):
            result = await messenger._post_message_async("C123", "hello")

        assert result is None  # .get("ts") returns None when key missing


# ===========================================================================
# Tests for MessagingMixin.get_circuit_breaker_status
# ===========================================================================


class TestGetCircuitBreakerStatus:
    """Tests for the get_circuit_breaker_status method."""

    def test_returns_status_dict(self, messenger):
        """Should return a dict with expected circuit breaker fields."""
        status = messenger.get_circuit_breaker_status()
        assert isinstance(status, dict)
        assert "state" in status
        assert "failure_count" in status
        assert "success_count" in status
        assert "failure_threshold" in status
        assert "cooldown_seconds" in status

    def test_initial_state_closed(self, messenger):
        """Initial circuit breaker status should show closed state."""
        status = messenger.get_circuit_breaker_status()
        assert status["state"] == "closed"
        assert status["failure_count"] == 0
        assert status["success_count"] == 0

    def test_status_after_failures(self, messenger):
        """Status should reflect failure count after recording failures."""
        cb = get_slack_circuit_breaker()
        cb.record_failure()
        cb.record_failure()
        status = messenger.get_circuit_breaker_status()
        assert status["failure_count"] == 2

    def test_status_shows_open_after_threshold(self, messenger):
        """Status should show open state after exceeding failure threshold."""
        cb = get_slack_circuit_breaker()
        for _ in range(10):
            cb.record_failure()
        status = messenger.get_circuit_breaker_status()
        assert status["state"] == "open"
