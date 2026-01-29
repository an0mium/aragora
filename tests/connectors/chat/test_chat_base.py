"""
Tests for ChatPlatformConnector abstract base class.

Validates constructor defaults, circuit breaker lifecycle, retry logic with
exponential backoff, HTTP request handling, retryable status codes, and
error paths using a concrete test subclass.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.connectors.chat.base import ChatPlatformConnector
from aragora.connectors.chat.models import (
    BotCommand,
    FileAttachment,
    SendMessageResponse,
    UserInteraction,
    WebhookEvent,
)


# ============================================================================
# Concrete Test Subclass
# ============================================================================


class StubConnector(ChatPlatformConnector):
    """Minimal concrete implementation for testing base class behaviour."""

    @property
    def platform_name(self) -> str:
        return "stub"

    @property
    def platform_display_name(self) -> str:
        return "Stub Platform"

    async def send_message(self, channel_id, text, blocks=None, thread_id=None, **kw):
        return SendMessageResponse(success=True, message_id="msg-1")

    async def update_message(self, channel_id, message_id, text, blocks=None, **kw):
        return SendMessageResponse(success=True, message_id=message_id)

    async def delete_message(self, channel_id, message_id, **kw):
        return True

    async def respond_to_command(self, command, text, blocks=None, ephemeral=True, **kw):
        return SendMessageResponse(success=True)

    async def respond_to_interaction(
        self, interaction, text, blocks=None, replace_original=False, **kw
    ):
        return SendMessageResponse(success=True)

    async def upload_file(
        self,
        channel_id,
        content,
        filename,
        content_type="application/octet-stream",
        title=None,
        thread_id=None,
        **kw,
    ):
        return FileAttachment(
            id="file-1", filename=filename, content_type=content_type, size=len(content)
        )

    async def download_file(self, file_id, **kw):
        return FileAttachment(id=file_id, filename="test.txt", content_type="text/plain", size=0)

    def format_blocks(self, title=None, body=None, fields=None, actions=None, **kw):
        return [{"type": "section", "text": body}]

    def format_button(self, text, action_id, value=None, style="default", url=None):
        return {"type": "button", "text": text, "action_id": action_id}

    def verify_webhook(self, headers, body):
        return True

    def parse_webhook_event(self, headers, body):
        return WebhookEvent(platform="stub", event_type="message")


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def reset_circuit_breakers():
    """Clear all circuit breakers before each test to ensure isolation."""
    from aragora.resilience import _circuit_breakers, _circuit_breakers_lock

    with _circuit_breakers_lock:
        _circuit_breakers.clear()
    yield
    with _circuit_breakers_lock:
        _circuit_breakers.clear()


@pytest.fixture
def connector():
    """Create a default StubConnector with circuit breaker enabled."""
    return StubConnector(bot_token="tok-123", signing_secret="sec-456")


@pytest.fixture
def connector_no_cb():
    """Create a StubConnector with circuit breaker disabled."""
    return StubConnector(bot_token="tok-123", enable_circuit_breaker=False)


# ============================================================================
# Constructor Defaults and Custom Config
# ============================================================================


class TestConstructorDefaults:
    """Verify constructor stores parameters and applies correct defaults."""

    def test_default_values(self):
        c = StubConnector()
        assert c.bot_token is None
        assert c.signing_secret is None
        assert c.webhook_url is None
        assert c._enable_circuit_breaker is True
        assert c._circuit_breaker_threshold == 5
        assert c._circuit_breaker_cooldown == 60.0
        assert c._request_timeout == 30.0
        assert c._initialized is False
        assert c._circuit_breaker is None
        assert c._circuit_breaker_initialized is False

    def test_custom_values(self):
        c = StubConnector(
            bot_token="my-token",
            signing_secret="my-secret",
            webhook_url="https://hook.example.com",
            enable_circuit_breaker=False,
            circuit_breaker_threshold=10,
            circuit_breaker_cooldown=120.0,
            request_timeout=15.0,
        )
        assert c.bot_token == "my-token"
        assert c.signing_secret == "my-secret"
        assert c.webhook_url == "https://hook.example.com"
        assert c._enable_circuit_breaker is False
        assert c._circuit_breaker_threshold == 10
        assert c._circuit_breaker_cooldown == 120.0
        assert c._request_timeout == 15.0

    def test_extra_config_kwargs_stored(self):
        c = StubConnector(custom_key="custom_val", another=42)
        assert c.config == {"custom_key": "custom_val", "another": 42}

    def test_is_configured_with_token(self):
        c = StubConnector(bot_token="tok")
        assert c.is_configured is True

    def test_is_configured_with_webhook(self):
        c = StubConnector(webhook_url="https://example.com")
        assert c.is_configured is True

    def test_is_not_configured_without_token_or_webhook(self):
        c = StubConnector()
        assert c.is_configured is False

    def test_repr(self):
        c = StubConnector()
        assert repr(c) == "StubConnector(platform=stub)"


# ============================================================================
# Circuit Breaker Lifecycle
# ============================================================================


class TestCircuitBreakerLifecycle:
    """Circuit breaker lazy init, check, record success/failure."""

    def test_lazy_initialization(self, connector):
        assert connector._circuit_breaker is None
        assert not connector._circuit_breaker_initialized

        cb = connector._get_circuit_breaker()
        assert cb is not None
        assert connector._circuit_breaker_initialized

    def test_disabled_returns_none(self, connector_no_cb):
        cb = connector_no_cb._get_circuit_breaker()
        assert cb is None

    def test_check_allows_when_closed(self, connector):
        can_proceed, error = connector._check_circuit_breaker()
        assert can_proceed is True
        assert error is None

    def test_check_blocks_when_open(self, connector):
        # Use low threshold so it opens quickly
        connector._circuit_breaker_threshold = 2
        connector._circuit_breaker_initialized = False
        connector._circuit_breaker = None

        c = StubConnector(
            bot_token="tok",
            circuit_breaker_threshold=2,
            circuit_breaker_cooldown=300.0,
        )
        cb = c._get_circuit_breaker()
        cb.record_failure()
        cb.record_failure()

        can_proceed, error = c._check_circuit_breaker()
        assert can_proceed is False
        assert error is not None
        assert "Circuit breaker open" in error
        assert "stub" in error

    def test_check_allows_when_disabled(self, connector_no_cb):
        can_proceed, error = connector_no_cb._check_circuit_breaker()
        assert can_proceed is True
        assert error is None

    def test_record_success(self, connector):
        # Should not raise even without prior _get
        connector._record_success()
        cb = connector._get_circuit_breaker()
        # After success, status should be closed
        assert cb.get_status() == "closed"

    def test_record_failure_opens_circuit(self):
        c = StubConnector(
            bot_token="tok",
            circuit_breaker_threshold=2,
        )
        c._record_failure()
        c._record_failure()

        cb = c._get_circuit_breaker()
        assert cb.get_status() == "open"

    def test_import_error_handled_gracefully(self):
        """If resilience module is unavailable, circuit breaker stays None."""
        c = StubConnector(bot_token="tok")
        with patch(
            "aragora.connectors.chat.base.ChatPlatformConnector._get_circuit_breaker"
        ) as mock_get:
            # Simulate the real implementation when import fails
            c._circuit_breaker = None
            c._circuit_breaker_initialized = True
            mock_get.return_value = None
            can_proceed, error = c._check_circuit_breaker()
            assert can_proceed is True
            assert error is None


# ============================================================================
# Retryable Status Codes
# ============================================================================


class TestRetryableStatusCodes:
    """Validate which HTTP status codes are considered retryable."""

    @pytest.mark.parametrize("code", [429, 500, 502, 503, 504])
    def test_retryable_codes(self, connector, code):
        assert connector._is_retryable_status_code(code) is True

    @pytest.mark.parametrize("code", [200, 201, 400, 401, 403, 404, 405, 409, 422])
    def test_non_retryable_codes(self, connector, code):
        assert connector._is_retryable_status_code(code) is False


# ============================================================================
# Retry Logic (_with_retry)
# ============================================================================


class TestWithRetry:
    """Async retry with exponential backoff and circuit breaker."""

    @pytest.mark.asyncio
    async def test_success_on_first_try(self, connector):
        func = AsyncMock(return_value="ok")
        result = await connector._with_retry("test_op", func)
        assert result == "ok"
        func.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_retries_on_failure_then_succeeds(self, connector_no_cb):
        func = AsyncMock(side_effect=[ValueError("fail"), "recovered"])
        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await connector_no_cb._with_retry(
                "test_op",
                func,
                max_retries=2,
                base_delay=0.01,
                retryable_exceptions=(ValueError,),
            )
        assert result == "recovered"
        assert func.await_count == 2

    @pytest.mark.asyncio
    async def test_raises_after_all_retries_exhausted(self, connector_no_cb):
        func = AsyncMock(side_effect=ValueError("always fails"))
        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(ValueError, match="always fails"):
                await connector_no_cb._with_retry(
                    "test_op",
                    func,
                    max_retries=3,
                    base_delay=0.01,
                    retryable_exceptions=(ValueError,),
                )
        assert func.await_count == 3

    @pytest.mark.asyncio
    async def test_circuit_breaker_blocks_retry(self):
        """If circuit is open, _with_retry raises ConnectionError immediately."""
        c = StubConnector(bot_token="tok", circuit_breaker_threshold=1)
        cb = c._get_circuit_breaker()
        cb.record_failure()

        func = AsyncMock()
        with pytest.raises(ConnectionError, match="Circuit breaker open"):
            await c._with_retry("blocked_op", func)
        func.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_records_success_on_circuit_breaker(self, connector):
        func = AsyncMock(return_value="ok")
        cb = connector._get_circuit_breaker()
        with patch.object(cb, "record_success") as mock_success:
            await connector._with_retry("test_op", func)
            mock_success.assert_called_once()

    @pytest.mark.asyncio
    async def test_records_failure_on_circuit_breaker(self, connector):
        func = AsyncMock(side_effect=RuntimeError("boom"))
        cb = connector._get_circuit_breaker()
        with patch.object(cb, "record_failure") as mock_fail:
            with patch("asyncio.sleep", new_callable=AsyncMock):
                with pytest.raises(RuntimeError):
                    await connector._with_retry(
                        "test_op",
                        func,
                        max_retries=2,
                        base_delay=0.01,
                    )
            assert mock_fail.call_count == 2

    @pytest.mark.asyncio
    async def test_non_retryable_exception_not_caught(self, connector_no_cb):
        """Exceptions not in retryable_exceptions propagate immediately."""
        func = AsyncMock(side_effect=TypeError("wrong type"))
        with pytest.raises(TypeError, match="wrong type"):
            await connector_no_cb._with_retry(
                "test_op",
                func,
                retryable_exceptions=(ValueError,),
            )
        # Only called once - no retry
        func.assert_awaited_once()


# ============================================================================
# HTTP Request Handling (_http_request)
# ============================================================================


class TestHttpRequest:
    """HTTP request with retry, timeout, and circuit breaker."""

    @pytest.mark.asyncio
    async def test_successful_json_response(self, connector_no_cb):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "ok"}

        mock_client = AsyncMock()
        mock_client.request.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            success, data, error = await connector_no_cb._http_request(
                "GET", "https://api.example.com/data"
            )

        assert success is True
        assert data == {"result": "ok"}
        assert error is None

    @pytest.mark.asyncio
    async def test_successful_raw_response(self, connector_no_cb):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"raw bytes"

        mock_client = AsyncMock()
        mock_client.request.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            success, data, error = await connector_no_cb._http_request(
                "GET", "https://api.example.com/file", return_raw=True
            )

        assert success is True
        assert data == b"raw bytes"
        assert error is None

    @pytest.mark.asyncio
    async def test_non_json_success_response(self, connector_no_cb):
        """When response is 200 but body is not valid JSON."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("no json")
        mock_response.text = "plain text response"

        mock_client = AsyncMock()
        mock_client.request.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            success, data, error = await connector_no_cb._http_request(
                "GET", "https://api.example.com/text"
            )

        assert success is True
        assert data == {"status": "ok", "text": "plain text response"}

    @pytest.mark.asyncio
    async def test_non_retryable_client_error(self, connector_no_cb):
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.text = "Not Found"

        mock_client = AsyncMock()
        mock_client.request.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            success, data, error = await connector_no_cb._http_request(
                "GET", "https://api.example.com/missing"
            )

        assert success is False
        assert data is None
        assert "404" in error

    @pytest.mark.asyncio
    async def test_retryable_status_code_retries(self, connector_no_cb):
        """A 503 should be retried up to max_retries times."""
        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_response.text = "Service Unavailable"

        mock_client = AsyncMock()
        mock_client.request.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                success, data, error = await connector_no_cb._http_request(
                    "GET",
                    "https://api.example.com/down",
                    max_retries=2,
                    base_delay=0.01,
                )

        assert success is False
        assert "503" in error
        # Should have been called twice (2 retries)
        assert mock_client.request.await_count == 2

    @pytest.mark.asyncio
    async def test_timeout_retries(self, connector_no_cb):
        import httpx

        mock_client = AsyncMock()
        mock_client.request.side_effect = httpx.TimeoutException("timed out")
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                success, data, error = await connector_no_cb._http_request(
                    "POST",
                    "https://api.example.com/slow",
                    max_retries=2,
                    base_delay=0.01,
                )

        assert success is False
        assert "Timeout" in error
        assert mock_client.request.await_count == 2

    @pytest.mark.asyncio
    async def test_connect_error_retries(self, connector_no_cb):
        import httpx

        mock_client = AsyncMock()
        mock_client.request.side_effect = httpx.ConnectError("refused")
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                success, data, error = await connector_no_cb._http_request(
                    "GET",
                    "https://api.example.com/down",
                    max_retries=3,
                    base_delay=0.01,
                )

        assert success is False
        assert "Connection error" in error
        assert mock_client.request.await_count == 3

    @pytest.mark.asyncio
    async def test_unexpected_error_no_retry(self, connector_no_cb):
        mock_client = AsyncMock()
        mock_client.request.side_effect = RuntimeError("unexpected")
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=mock_client):
            success, data, error = await connector_no_cb._http_request(
                "GET",
                "https://api.example.com/boom",
                max_retries=3,
            )

        assert success is False
        assert "Unexpected error" in error
        # No retry on unexpected errors
        mock_client.request.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_circuit_breaker_blocks_http_request(self):
        c = StubConnector(bot_token="tok", circuit_breaker_threshold=1)
        cb = c._get_circuit_breaker()
        cb.record_failure()

        success, data, error = await c._http_request("GET", "https://api.example.com")
        assert success is False
        assert data is None
        assert "Circuit breaker open" in error

    @pytest.mark.asyncio
    async def test_httpx_import_error(self, connector_no_cb):
        """When httpx is not available, return graceful error."""
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "httpx":
                raise ImportError("No module named 'httpx'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            success, data, error = await connector_no_cb._http_request(
                "GET", "https://api.example.com"
            )

        assert success is False
        assert error == "httpx not available"


# ============================================================================
# Utility and Default Method Behaviour
# ============================================================================


class TestUtilityMethods:
    """Test connection, health, default method implementations."""

    @pytest.mark.asyncio
    async def test_test_connection_configured(self, connector):
        result = await connector.test_connection()
        assert result["platform"] == "stub"
        assert result["success"] is True
        assert result["bot_token_configured"] is True

    @pytest.mark.asyncio
    async def test_test_connection_unconfigured(self):
        c = StubConnector()
        result = await c.test_connection()
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_send_typing_indicator_default(self, connector):
        result = await connector.send_typing_indicator("ch-1")
        assert result is False

    @pytest.mark.asyncio
    async def test_get_voice_message_default(self, connector):
        result = await connector.get_voice_message("file-1")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_channel_info_default(self, connector):
        result = await connector.get_channel_info("ch-1")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_user_info_default(self, connector):
        result = await connector.get_user_info("u-1")
        assert result is None

    @pytest.mark.asyncio
    async def test_collect_evidence_default(self, connector):
        result = await connector.collect_evidence("ch-1")
        assert result == []

    @pytest.mark.asyncio
    async def test_get_channel_history_default(self, connector):
        result = await connector.get_channel_history("ch-1")
        assert result == []

    @pytest.mark.asyncio
    async def test_send_voice_message_default_delegates_to_upload(self, connector):
        """Default voice message sends via upload_file."""
        result = await connector.send_voice_message("ch-1", b"audio-bytes")
        assert result.success is True
        assert result.message_id == "file-1"

    @pytest.mark.asyncio
    async def test_send_voice_message_handles_upload_failure(self):
        """If upload_file raises, send_voice_message returns error response."""

        class FailUploadConnector(StubConnector):
            async def upload_file(self, *args, **kwargs):
                raise RuntimeError("upload boom")

        c = FailUploadConnector(bot_token="tok")
        result = await c.send_voice_message("ch-1", b"audio")
        assert result.success is False
        assert "upload boom" in result.error


# ============================================================================
# Health Check
# ============================================================================


class TestHealthCheck:
    """Health endpoint reflecting configuration and circuit breaker state."""

    @pytest.mark.asyncio
    async def test_health_configured_healthy(self, connector):
        health = await connector.get_health()
        assert health["platform"] == "stub"
        assert health["display_name"] == "Stub Platform"
        assert health["status"] == "healthy"
        assert health["configured"] is True
        assert health["details"]["bot_token_configured"] is True
        assert health["details"]["request_timeout"] == 30.0

    @pytest.mark.asyncio
    async def test_health_unconfigured(self):
        c = StubConnector()
        health = await c.get_health()
        assert health["status"] == "unconfigured"
        assert "Missing required configuration" in health["details"]["error"]

    @pytest.mark.asyncio
    async def test_health_circuit_breaker_disabled(self, connector_no_cb):
        health = await connector_no_cb.get_health()
        assert health["circuit_breaker"] == {"enabled": False}
        assert health["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_health_circuit_breaker_open(self):
        c = StubConnector(bot_token="tok", circuit_breaker_threshold=1)
        c._record_failure()  # opens the circuit
        health = await c.get_health()
        assert health["status"] == "unhealthy"
        assert health["circuit_breaker"]["state"] == "open"
        assert "cooldown_remaining" in health["circuit_breaker"]


# ============================================================================
# Message Query and Relevance Helpers
# ============================================================================


class TestMessageQueryHelpers:
    """_message_matches_query and _compute_message_relevance."""

    def _make_msg(self, content: str):
        from aragora.connectors.chat.models import ChatChannel, ChatMessage, ChatUser

        return ChatMessage(
            id="m1",
            platform="stub",
            channel=ChatChannel(id="ch1", platform="stub"),
            author=ChatUser(id="u1", platform="stub"),
            content=content,
        )

    def test_matches_query_empty_query(self, connector):
        msg = self._make_msg("anything")
        assert connector._message_matches_query(msg, "") is True

    def test_matches_query_keyword_found(self, connector):
        msg = self._make_msg("the deployment failed")
        assert connector._message_matches_query(msg, "deployment") is True

    def test_matches_query_keyword_not_found(self, connector):
        msg = self._make_msg("everything is fine")
        assert connector._message_matches_query(msg, "deployment") is False

    def test_relevance_no_query(self, connector):
        msg = self._make_msg("anything")
        assert connector._compute_message_relevance(msg) == 1.0

    def test_relevance_all_keywords_match(self, connector):
        msg = self._make_msg("the deployment process failed")
        score = connector._compute_message_relevance(msg, "deployment failed")
        assert score == 1.0

    def test_relevance_partial_match(self, connector):
        msg = self._make_msg("the deployment succeeded")
        score = connector._compute_message_relevance(msg, "deployment failed")
        assert score == 0.5

    def test_relevance_no_match(self, connector):
        msg = self._make_msg("everything is fine")
        score = connector._compute_message_relevance(msg, "deployment failed")
        assert score == 0.0

    def test_relevance_empty_content(self, connector):
        msg = self._make_msg("")
        score = connector._compute_message_relevance(msg, "deployment")
        assert score == 0.0
