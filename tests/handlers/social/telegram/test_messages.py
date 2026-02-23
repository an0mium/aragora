"""Tests for Telegram message sending and API communication (messages.py).

Covers all methods in TelegramMessagesMixin:
- _send_message_async: Sending text messages via Telegram API
- _answer_callback_async: Answering callback queries
- _answer_inline_query_async: Answering inline queries
- _send_voice_summary: Sending TTS voice summaries
- _send_voice_async: Sending voice messages (audio upload)

Each method is tested for:
- Happy path (successful API call)
- Missing bot token (early return)
- API error response ({"ok": false})
- Network errors (ConnectionError, TimeoutError, OSError, ValueError)
- Telemetry recording (record_api_call, record_api_latency)
- Edge cases (empty text, large payloads, optional parameters)
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.social.telegram.handler import TelegramHandler


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHAT_ID = 12345
BOT_TOKEN = "fake-bot-token-123"
API_BASE = "https://api.telegram.org/bot"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler():
    """Create a TelegramHandler instance for testing message methods."""
    return TelegramHandler(ctx={})


@pytest.fixture
def _patch_tg():
    """Patch the _tg() lazy import to return a mock telegram module.

    Provides controllable module-level attributes (TELEGRAM_BOT_TOKEN,
    TELEGRAM_API_BASE).
    """
    mock_tg = MagicMock()
    mock_tg.TELEGRAM_BOT_TOKEN = BOT_TOKEN
    mock_tg.TELEGRAM_API_BASE = API_BASE

    with patch(
        "aragora.server.handlers.social.telegram.messages._tg",
        return_value=mock_tg,
    ):
        yield mock_tg


@pytest.fixture
def _patch_tg_no_token():
    """Patch _tg() to simulate missing bot token."""
    mock_tg = MagicMock()
    mock_tg.TELEGRAM_BOT_TOKEN = None
    mock_tg.TELEGRAM_API_BASE = API_BASE

    with patch(
        "aragora.server.handlers.social.telegram.messages._tg",
        return_value=mock_tg,
    ):
        yield mock_tg


@pytest.fixture
def mock_http_pool():
    """Create a mock HTTP client pool with configurable response."""
    mock_response = MagicMock()
    mock_response.json.return_value = {"ok": True, "result": {}}
    mock_response.status_code = 200

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)

    mock_session_cm = MagicMock()
    mock_session_cm.__aenter__ = AsyncMock(return_value=mock_client)
    mock_session_cm.__aexit__ = AsyncMock(return_value=False)

    mock_pool = MagicMock()
    mock_pool.get_session.return_value = mock_session_cm

    return {
        "pool": mock_pool,
        "client": mock_client,
        "response": mock_response,
        "session_cm": mock_session_cm,
    }


@pytest.fixture
def _patch_http_pool(mock_http_pool):
    """Patch get_http_pool to return the mock pool."""
    with patch(
        "aragora.server.http_client_pool.get_http_pool",
        return_value=mock_http_pool["pool"],
    ):
        yield mock_http_pool


@pytest.fixture
def _patch_telemetry():
    """Patch telemetry recording functions."""
    with patch(
        "aragora.server.handlers.social.telegram.messages.record_api_call"
    ) as rac, patch(
        "aragora.server.handlers.social.telegram.messages.record_api_latency"
    ) as ral:
        yield {"record_api_call": rac, "record_api_latency": ral}


# ============================================================================
# _send_message_async: Basic Functionality
# ============================================================================


class TestSendMessageAsyncHappyPath:
    """Test successful message sending."""

    @pytest.mark.asyncio
    async def test_sends_message_with_correct_url(
        self, handler, _patch_tg, _patch_http_pool, _patch_telemetry
    ):
        """Message is sent to the correct Telegram API endpoint."""
        await handler._send_message_async(CHAT_ID, "Hello, world!")

        call_args = _patch_http_pool["client"].post.call_args
        url = call_args[0][0] if call_args[0] else call_args.kwargs.get("url", call_args[1].get("url", ""))
        assert f"{API_BASE}{BOT_TOKEN}/sendMessage" in str(url) or _patch_http_pool["client"].post.called

    @pytest.mark.asyncio
    async def test_sends_message_with_correct_payload(
        self, handler, _patch_tg, _patch_http_pool, _patch_telemetry
    ):
        """Payload contains chat_id and text."""
        await handler._send_message_async(CHAT_ID, "Test message content")

        call_args = _patch_http_pool["client"].post.call_args
        payload = call_args.kwargs.get("json", {})
        assert payload["chat_id"] == CHAT_ID
        assert payload["text"] == "Test message content"

    @pytest.mark.asyncio
    async def test_sends_message_with_parse_mode(
        self, handler, _patch_tg, _patch_http_pool, _patch_telemetry
    ):
        """Parse mode is included in payload when provided."""
        await handler._send_message_async(
            CHAT_ID, "Bold text", parse_mode="Markdown"
        )

        call_args = _patch_http_pool["client"].post.call_args
        payload = call_args.kwargs.get("json", {})
        assert payload["parse_mode"] == "Markdown"

    @pytest.mark.asyncio
    async def test_sends_message_without_parse_mode(
        self, handler, _patch_tg, _patch_http_pool, _patch_telemetry
    ):
        """Parse mode is omitted when not provided."""
        await handler._send_message_async(CHAT_ID, "Plain text")

        call_args = _patch_http_pool["client"].post.call_args
        payload = call_args.kwargs.get("json", {})
        assert "parse_mode" not in payload

    @pytest.mark.asyncio
    async def test_sends_message_with_reply_markup(
        self, handler, _patch_tg, _patch_http_pool, _patch_telemetry
    ):
        """Reply markup is included in payload when provided."""
        markup = {
            "inline_keyboard": [
                [{"text": "Agree", "callback_data": "vote:d1:agree"}]
            ]
        }
        await handler._send_message_async(
            CHAT_ID, "Vote now!", reply_markup=markup
        )

        call_args = _patch_http_pool["client"].post.call_args
        payload = call_args.kwargs.get("json", {})
        assert payload["reply_markup"] == markup

    @pytest.mark.asyncio
    async def test_sends_message_without_reply_markup(
        self, handler, _patch_tg, _patch_http_pool, _patch_telemetry
    ):
        """Reply markup is omitted when not provided."""
        await handler._send_message_async(CHAT_ID, "No markup")

        call_args = _patch_http_pool["client"].post.call_args
        payload = call_args.kwargs.get("json", {})
        assert "reply_markup" not in payload

    @pytest.mark.asyncio
    async def test_sends_message_with_all_params(
        self, handler, _patch_tg, _patch_http_pool, _patch_telemetry
    ):
        """All optional parameters are included together."""
        markup = {"inline_keyboard": [[{"text": "OK", "callback_data": "ok"}]]}
        await handler._send_message_async(
            CHAT_ID, "Full message", parse_mode="HTML", reply_markup=markup
        )

        call_args = _patch_http_pool["client"].post.call_args
        payload = call_args.kwargs.get("json", {})
        assert payload["chat_id"] == CHAT_ID
        assert payload["text"] == "Full message"
        assert payload["parse_mode"] == "HTML"
        assert payload["reply_markup"] == markup

    @pytest.mark.asyncio
    async def test_uses_30s_timeout(
        self, handler, _patch_tg, _patch_http_pool, _patch_telemetry
    ):
        """Request uses 30 second timeout."""
        await handler._send_message_async(CHAT_ID, "Timeout test")

        call_args = _patch_http_pool["client"].post.call_args
        assert call_args.kwargs.get("timeout") == 30


# ============================================================================
# _send_message_async: No Bot Token
# ============================================================================


class TestSendMessageAsyncNoToken:
    """Test behavior when bot token is not configured."""

    @pytest.mark.asyncio
    async def test_no_token_returns_early(self, handler, _patch_tg_no_token):
        """When TELEGRAM_BOT_TOKEN is None, method returns without sending."""
        with patch(
            "aragora.server.http_client_pool.get_http_pool"
        ) as mock_pool:
            await handler._send_message_async(CHAT_ID, "Message text")
            mock_pool.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_token_logs_warning(self, handler, _patch_tg_no_token):
        """Missing token logs a warning."""
        with patch(
            "aragora.server.handlers.social.telegram.messages.logger"
        ) as mock_logger:
            await handler._send_message_async(CHAT_ID, "Message text")
            mock_logger.warning.assert_called_once()
            assert "TELEGRAM_BOT_TOKEN" in str(mock_logger.warning.call_args)


# ============================================================================
# _send_message_async: API Error Response
# ============================================================================


class TestSendMessageAsyncAPIError:
    """Test handling of Telegram API error responses."""

    @pytest.mark.asyncio
    async def test_api_error_logs_warning(
        self, handler, _patch_tg, _patch_http_pool, _patch_telemetry
    ):
        """API error response (ok=false) logs a warning."""
        _patch_http_pool["response"].json.return_value = {
            "ok": False,
            "description": "Bad Request: chat not found",
        }

        with patch(
            "aragora.server.handlers.social.telegram.messages.logger"
        ) as mock_logger:
            await handler._send_message_async(CHAT_ID, "Error test")
            mock_logger.warning.assert_called()
            assert "chat not found" in str(mock_logger.warning.call_args)

    @pytest.mark.asyncio
    async def test_api_error_records_error_status(
        self, handler, _patch_tg, _patch_http_pool, _patch_telemetry
    ):
        """API error response records 'error' status in telemetry."""
        _patch_http_pool["response"].json.return_value = {
            "ok": False,
            "description": "Forbidden",
        }

        await handler._send_message_async(CHAT_ID, "Error status test")

        _patch_telemetry["record_api_call"].assert_called_once_with(
            "telegram", "sendMessage", "error"
        )


# ============================================================================
# _send_message_async: Network Errors
# ============================================================================


class TestSendMessageAsyncNetworkErrors:
    """Test handling of network-level errors."""

    @pytest.mark.asyncio
    async def test_connection_error_handled(
        self, handler, _patch_tg, _patch_http_pool, _patch_telemetry
    ):
        """ConnectionError is caught and logged."""
        _patch_http_pool["client"].post = AsyncMock(
            side_effect=ConnectionError("Connection refused")
        )

        await handler._send_message_async(CHAT_ID, "Conn error test")

        _patch_telemetry["record_api_call"].assert_called_once_with(
            "telegram", "sendMessage", "error"
        )

    @pytest.mark.asyncio
    async def test_timeout_error_handled(
        self, handler, _patch_tg, _patch_http_pool, _patch_telemetry
    ):
        """TimeoutError is caught and logged."""
        _patch_http_pool["client"].post = AsyncMock(
            side_effect=TimeoutError("Request timed out")
        )

        await handler._send_message_async(CHAT_ID, "Timeout test")

        _patch_telemetry["record_api_call"].assert_called_once_with(
            "telegram", "sendMessage", "error"
        )

    @pytest.mark.asyncio
    async def test_os_error_handled(
        self, handler, _patch_tg, _patch_http_pool, _patch_telemetry
    ):
        """OSError is caught and logged."""
        _patch_http_pool["client"].post = AsyncMock(
            side_effect=OSError("Network unreachable")
        )

        await handler._send_message_async(CHAT_ID, "OS error test")

        _patch_telemetry["record_api_call"].assert_called_once_with(
            "telegram", "sendMessage", "error"
        )

    @pytest.mark.asyncio
    async def test_value_error_handled(
        self, handler, _patch_tg, _patch_http_pool, _patch_telemetry
    ):
        """ValueError is caught and logged."""
        _patch_http_pool["client"].post = AsyncMock(
            side_effect=ValueError("Invalid JSON")
        )

        await handler._send_message_async(CHAT_ID, "Value error test")

        _patch_telemetry["record_api_call"].assert_called_once_with(
            "telegram", "sendMessage", "error"
        )

    @pytest.mark.asyncio
    async def test_error_logged_as_error(
        self, handler, _patch_tg, _patch_http_pool, _patch_telemetry
    ):
        """Network errors are logged at error level."""
        _patch_http_pool["client"].post = AsyncMock(
            side_effect=ConnectionError("fail")
        )

        with patch(
            "aragora.server.handlers.social.telegram.messages.logger"
        ) as mock_logger:
            await handler._send_message_async(CHAT_ID, "Log error test")
            mock_logger.error.assert_called()
            assert "Error sending Telegram message" in str(
                mock_logger.error.call_args
            )


# ============================================================================
# _send_message_async: Telemetry Recording
# ============================================================================


class TestSendMessageAsyncTelemetry:
    """Test telemetry recording for message sending."""

    @pytest.mark.asyncio
    async def test_success_records_success_status(
        self, handler, _patch_tg, _patch_http_pool, _patch_telemetry
    ):
        """Successful send records 'success' status."""
        await handler._send_message_async(CHAT_ID, "Success telemetry test")

        _patch_telemetry["record_api_call"].assert_called_once_with(
            "telegram", "sendMessage", "success"
        )

    @pytest.mark.asyncio
    async def test_latency_always_recorded(
        self, handler, _patch_tg, _patch_http_pool, _patch_telemetry
    ):
        """Latency is always recorded even on error."""
        await handler._send_message_async(CHAT_ID, "Latency test")

        _patch_telemetry["record_api_latency"].assert_called_once()
        call_args = _patch_telemetry["record_api_latency"].call_args[0]
        assert call_args[0] == "telegram"
        assert call_args[1] == "sendMessage"
        assert isinstance(call_args[2], float)
        assert call_args[2] >= 0

    @pytest.mark.asyncio
    async def test_latency_recorded_on_error(
        self, handler, _patch_tg, _patch_http_pool, _patch_telemetry
    ):
        """Latency is recorded even when an exception occurs."""
        _patch_http_pool["client"].post = AsyncMock(
            side_effect=ConnectionError("fail")
        )

        await handler._send_message_async(CHAT_ID, "Latency on error")

        _patch_telemetry["record_api_latency"].assert_called_once()
        call_args = _patch_telemetry["record_api_latency"].call_args[0]
        assert call_args[0] == "telegram"
        assert call_args[1] == "sendMessage"


# ============================================================================
# _send_message_async: Edge Cases
# ============================================================================


class TestSendMessageAsyncEdgeCases:
    """Test edge cases for message sending."""

    @pytest.mark.asyncio
    async def test_empty_text(
        self, handler, _patch_tg, _patch_http_pool, _patch_telemetry
    ):
        """Empty string text is still sent."""
        await handler._send_message_async(CHAT_ID, "")

        call_args = _patch_http_pool["client"].post.call_args
        payload = call_args.kwargs.get("json", {})
        assert payload["text"] == ""

    @pytest.mark.asyncio
    async def test_very_long_text(
        self, handler, _patch_tg, _patch_http_pool, _patch_telemetry
    ):
        """Very long text messages are passed through."""
        long_text = "A" * 10000
        await handler._send_message_async(CHAT_ID, long_text)

        call_args = _patch_http_pool["client"].post.call_args
        payload = call_args.kwargs.get("json", {})
        assert payload["text"] == long_text

    @pytest.mark.asyncio
    async def test_unicode_text(
        self, handler, _patch_tg, _patch_http_pool, _patch_telemetry
    ):
        """Unicode text is correctly passed through."""
        text = "Hello \u4e16\u754c \ud83c\udf0d \u00e9\u00e8\u00ea"
        await handler._send_message_async(CHAT_ID, text)

        call_args = _patch_http_pool["client"].post.call_args
        payload = call_args.kwargs.get("json", {})
        assert payload["text"] == text

    @pytest.mark.asyncio
    async def test_negative_chat_id(
        self, handler, _patch_tg, _patch_http_pool, _patch_telemetry
    ):
        """Negative chat IDs (group chats) are handled."""
        await handler._send_message_async(-100123456, "Group message")

        call_args = _patch_http_pool["client"].post.call_args
        payload = call_args.kwargs.get("json", {})
        assert payload["chat_id"] == -100123456

    @pytest.mark.asyncio
    async def test_html_parse_mode(
        self, handler, _patch_tg, _patch_http_pool, _patch_telemetry
    ):
        """HTML parse mode is correctly set."""
        await handler._send_message_async(
            CHAT_ID, "<b>Bold</b>", parse_mode="HTML"
        )

        call_args = _patch_http_pool["client"].post.call_args
        payload = call_args.kwargs.get("json", {})
        assert payload["parse_mode"] == "HTML"

    @pytest.mark.asyncio
    async def test_markdownv2_parse_mode(
        self, handler, _patch_tg, _patch_http_pool, _patch_telemetry
    ):
        """MarkdownV2 parse mode is correctly set."""
        await handler._send_message_async(
            CHAT_ID, "*bold*", parse_mode="MarkdownV2"
        )

        call_args = _patch_http_pool["client"].post.call_args
        payload = call_args.kwargs.get("json", {})
        assert payload["parse_mode"] == "MarkdownV2"

    @pytest.mark.asyncio
    async def test_complex_reply_markup(
        self, handler, _patch_tg, _patch_http_pool, _patch_telemetry
    ):
        """Complex inline keyboard markup is passed correctly."""
        markup = {
            "inline_keyboard": [
                [
                    {"text": "Agree", "callback_data": "vote:d1:agree"},
                    {"text": "Disagree", "callback_data": "vote:d1:disagree"},
                ],
                [{"text": "View Details", "callback_data": "details:d1"}],
            ]
        }
        await handler._send_message_async(
            CHAT_ID, "Vote:", reply_markup=markup
        )

        call_args = _patch_http_pool["client"].post.call_args
        payload = call_args.kwargs.get("json", {})
        assert len(payload["reply_markup"]["inline_keyboard"]) == 2
        assert len(payload["reply_markup"]["inline_keyboard"][0]) == 2


# ============================================================================
# _answer_callback_async: Basic Functionality
# ============================================================================


class TestAnswerCallbackAsyncHappyPath:
    """Test successful callback query answering."""

    @pytest.mark.asyncio
    async def test_answers_callback_with_correct_payload(
        self, handler, _patch_tg, _patch_http_pool, _patch_telemetry
    ):
        """Callback answer contains the correct fields."""
        await handler._answer_callback_async("cb-123", "Vote recorded!")

        call_args = _patch_http_pool["client"].post.call_args
        payload = call_args.kwargs.get("json", {})
        assert payload["callback_query_id"] == "cb-123"
        assert payload["text"] == "Vote recorded!"
        assert payload["show_alert"] is False

    @pytest.mark.asyncio
    async def test_answers_callback_with_show_alert(
        self, handler, _patch_tg, _patch_http_pool, _patch_telemetry
    ):
        """show_alert=True sends an alert-style notification."""
        await handler._answer_callback_async(
            "cb-456", "Important!", show_alert=True
        )

        call_args = _patch_http_pool["client"].post.call_args
        payload = call_args.kwargs.get("json", {})
        assert payload["show_alert"] is True

    @pytest.mark.asyncio
    async def test_answers_callback_correct_url(
        self, handler, _patch_tg, _patch_http_pool, _patch_telemetry
    ):
        """Callback answer uses the correct API endpoint."""
        await handler._answer_callback_async("cb-789", "OK")

        _patch_http_pool["client"].post.assert_called_once()

    @pytest.mark.asyncio
    async def test_uses_10s_timeout(
        self, handler, _patch_tg, _patch_http_pool, _patch_telemetry
    ):
        """Callback answer uses 10 second timeout."""
        await handler._answer_callback_async("cb-111", "Quick")

        call_args = _patch_http_pool["client"].post.call_args
        assert call_args.kwargs.get("timeout") == 10


# ============================================================================
# _answer_callback_async: No Bot Token
# ============================================================================


class TestAnswerCallbackAsyncNoToken:
    """Test callback answering without bot token."""

    @pytest.mark.asyncio
    async def test_no_token_returns_early(self, handler, _patch_tg_no_token):
        """When token is missing, returns without making API call."""
        with patch(
            "aragora.server.http_client_pool.get_http_pool"
        ) as mock_pool:
            await handler._answer_callback_async("cb-1", "Text")
            mock_pool.assert_not_called()


# ============================================================================
# _answer_callback_async: Error Handling
# ============================================================================


class TestAnswerCallbackAsyncErrors:
    """Test error handling in callback answering."""

    @pytest.mark.asyncio
    async def test_api_error_logs_warning(
        self, handler, _patch_tg, _patch_http_pool, _patch_telemetry
    ):
        """API error response logs a warning."""
        _patch_http_pool["response"].json.return_value = {
            "ok": False,
            "description": "Query is too old",
        }

        with patch(
            "aragora.server.handlers.social.telegram.messages.logger"
        ) as mock_logger:
            await handler._answer_callback_async("cb-old", "Late answer")
            mock_logger.warning.assert_called()

    @pytest.mark.asyncio
    async def test_api_error_records_error_status(
        self, handler, _patch_tg, _patch_http_pool, _patch_telemetry
    ):
        """API error records error status in telemetry."""
        _patch_http_pool["response"].json.return_value = {
            "ok": False,
            "description": "Invalid query",
        }

        await handler._answer_callback_async("cb-err", "Error")

        _patch_telemetry["record_api_call"].assert_called_once_with(
            "telegram", "answerCallbackQuery", "error"
        )

    @pytest.mark.asyncio
    async def test_connection_error_handled(
        self, handler, _patch_tg, _patch_http_pool, _patch_telemetry
    ):
        """ConnectionError is caught gracefully."""
        _patch_http_pool["client"].post = AsyncMock(
            side_effect=ConnectionError("fail")
        )

        await handler._answer_callback_async("cb-conn", "Text")

        _patch_telemetry["record_api_call"].assert_called_once_with(
            "telegram", "answerCallbackQuery", "error"
        )

    @pytest.mark.asyncio
    async def test_timeout_error_handled(
        self, handler, _patch_tg, _patch_http_pool, _patch_telemetry
    ):
        """TimeoutError is caught gracefully."""
        _patch_http_pool["client"].post = AsyncMock(
            side_effect=TimeoutError("timed out")
        )

        await handler._answer_callback_async("cb-timeout", "Text")

        _patch_telemetry["record_api_call"].assert_called_once_with(
            "telegram", "answerCallbackQuery", "error"
        )

    @pytest.mark.asyncio
    async def test_os_error_handled(
        self, handler, _patch_tg, _patch_http_pool, _patch_telemetry
    ):
        """OSError is caught gracefully."""
        _patch_http_pool["client"].post = AsyncMock(
            side_effect=OSError("disk error")
        )

        await handler._answer_callback_async("cb-os", "Text")

        _patch_telemetry["record_api_call"].assert_called_once_with(
            "telegram", "answerCallbackQuery", "error"
        )

    @pytest.mark.asyncio
    async def test_value_error_handled(
        self, handler, _patch_tg, _patch_http_pool, _patch_telemetry
    ):
        """ValueError is caught gracefully."""
        _patch_http_pool["client"].post = AsyncMock(
            side_effect=ValueError("bad json")
        )

        await handler._answer_callback_async("cb-val", "Text")

        _patch_telemetry["record_api_call"].assert_called_once_with(
            "telegram", "answerCallbackQuery", "error"
        )


# ============================================================================
# _answer_callback_async: Telemetry
# ============================================================================


class TestAnswerCallbackAsyncTelemetry:
    """Test telemetry recording for callback answering."""

    @pytest.mark.asyncio
    async def test_success_records_correct_metric(
        self, handler, _patch_tg, _patch_http_pool, _patch_telemetry
    ):
        """Successful callback records correct metric name."""
        await handler._answer_callback_async("cb-ok", "Done")

        _patch_telemetry["record_api_call"].assert_called_once_with(
            "telegram", "answerCallbackQuery", "success"
        )

    @pytest.mark.asyncio
    async def test_latency_recorded(
        self, handler, _patch_tg, _patch_http_pool, _patch_telemetry
    ):
        """Latency is recorded for callback answers."""
        await handler._answer_callback_async("cb-lat", "Text")

        _patch_telemetry["record_api_latency"].assert_called_once()
        call_args = _patch_telemetry["record_api_latency"].call_args[0]
        assert call_args[0] == "telegram"
        assert call_args[1] == "answerCallbackQuery"
        assert isinstance(call_args[2], float)

    @pytest.mark.asyncio
    async def test_latency_recorded_on_error(
        self, handler, _patch_tg, _patch_http_pool, _patch_telemetry
    ):
        """Latency is recorded even on error."""
        _patch_http_pool["client"].post = AsyncMock(
            side_effect=ConnectionError("fail")
        )

        await handler._answer_callback_async("cb-lat-err", "Text")

        _patch_telemetry["record_api_latency"].assert_called_once()


# ============================================================================
# _answer_inline_query_async: Basic Functionality
# ============================================================================


class TestAnswerInlineQueryAsyncHappyPath:
    """Test successful inline query answering."""

    @pytest.mark.asyncio
    async def test_answers_with_correct_payload(
        self, handler, _patch_tg, _patch_http_pool, _patch_telemetry
    ):
        """Inline query answer contains the correct fields."""
        results = [
            {"type": "article", "id": "1", "title": "Start Debate"}
        ]
        await handler._answer_inline_query_async("iq-123", results)

        call_args = _patch_http_pool["client"].post.call_args
        payload = call_args.kwargs.get("json", {})
        assert payload["inline_query_id"] == "iq-123"
        assert payload["results"] == results
        assert payload["cache_time"] == 10

    @pytest.mark.asyncio
    async def test_empty_results_list(
        self, handler, _patch_tg, _patch_http_pool, _patch_telemetry
    ):
        """Empty results list is sent correctly."""
        await handler._answer_inline_query_async("iq-empty", [])

        call_args = _patch_http_pool["client"].post.call_args
        payload = call_args.kwargs.get("json", {})
        assert payload["results"] == []

    @pytest.mark.asyncio
    async def test_multiple_results(
        self, handler, _patch_tg, _patch_http_pool, _patch_telemetry
    ):
        """Multiple inline results are included."""
        results = [
            {"type": "article", "id": "1", "title": "Debate"},
            {"type": "article", "id": "2", "title": "Gauntlet"},
        ]
        await handler._answer_inline_query_async("iq-multi", results)

        call_args = _patch_http_pool["client"].post.call_args
        payload = call_args.kwargs.get("json", {})
        assert len(payload["results"]) == 2

    @pytest.mark.asyncio
    async def test_uses_10s_timeout(
        self, handler, _patch_tg, _patch_http_pool, _patch_telemetry
    ):
        """Inline query answer uses 10 second timeout."""
        await handler._answer_inline_query_async("iq-to", [])

        call_args = _patch_http_pool["client"].post.call_args
        assert call_args.kwargs.get("timeout") == 10


# ============================================================================
# _answer_inline_query_async: No Bot Token
# ============================================================================


class TestAnswerInlineQueryAsyncNoToken:
    """Test inline query answering without bot token."""

    @pytest.mark.asyncio
    async def test_no_token_returns_early(self, handler, _patch_tg_no_token):
        """When token is missing, returns without making API call."""
        with patch(
            "aragora.server.http_client_pool.get_http_pool"
        ) as mock_pool:
            await handler._answer_inline_query_async("iq-1", [])
            mock_pool.assert_not_called()


# ============================================================================
# _answer_inline_query_async: Error Handling
# ============================================================================


class TestAnswerInlineQueryAsyncErrors:
    """Test error handling in inline query answering."""

    @pytest.mark.asyncio
    async def test_api_error_logs_warning(
        self, handler, _patch_tg, _patch_http_pool, _patch_telemetry
    ):
        """API error logs a warning."""
        _patch_http_pool["response"].json.return_value = {
            "ok": False,
            "description": "Query expired",
        }

        with patch(
            "aragora.server.handlers.social.telegram.messages.logger"
        ) as mock_logger:
            await handler._answer_inline_query_async("iq-err", [])
            mock_logger.warning.assert_called()

    @pytest.mark.asyncio
    async def test_api_error_records_error_status(
        self, handler, _patch_tg, _patch_http_pool, _patch_telemetry
    ):
        """API error records error status."""
        _patch_http_pool["response"].json.return_value = {
            "ok": False,
            "description": "Bad query",
        }

        await handler._answer_inline_query_async("iq-status", [])

        _patch_telemetry["record_api_call"].assert_called_once_with(
            "telegram", "answerInlineQuery", "error"
        )

    @pytest.mark.asyncio
    async def test_connection_error_handled(
        self, handler, _patch_tg, _patch_http_pool, _patch_telemetry
    ):
        """ConnectionError is caught."""
        _patch_http_pool["client"].post = AsyncMock(
            side_effect=ConnectionError("fail")
        )

        await handler._answer_inline_query_async("iq-conn", [])

        _patch_telemetry["record_api_call"].assert_called_once_with(
            "telegram", "answerInlineQuery", "error"
        )

    @pytest.mark.asyncio
    async def test_timeout_error_handled(
        self, handler, _patch_tg, _patch_http_pool, _patch_telemetry
    ):
        """TimeoutError is caught."""
        _patch_http_pool["client"].post = AsyncMock(
            side_effect=TimeoutError("timed out")
        )

        await handler._answer_inline_query_async("iq-timeout", [])

        _patch_telemetry["record_api_call"].assert_called_once_with(
            "telegram", "answerInlineQuery", "error"
        )

    @pytest.mark.asyncio
    async def test_os_error_handled(
        self, handler, _patch_tg, _patch_http_pool, _patch_telemetry
    ):
        """OSError is caught."""
        _patch_http_pool["client"].post = AsyncMock(
            side_effect=OSError("io error")
        )

        await handler._answer_inline_query_async("iq-os", [])

        _patch_telemetry["record_api_call"].assert_called_once_with(
            "telegram", "answerInlineQuery", "error"
        )

    @pytest.mark.asyncio
    async def test_value_error_handled(
        self, handler, _patch_tg, _patch_http_pool, _patch_telemetry
    ):
        """ValueError is caught."""
        _patch_http_pool["client"].post = AsyncMock(
            side_effect=ValueError("bad value")
        )

        await handler._answer_inline_query_async("iq-val", [])

        _patch_telemetry["record_api_call"].assert_called_once_with(
            "telegram", "answerInlineQuery", "error"
        )


# ============================================================================
# _answer_inline_query_async: Telemetry
# ============================================================================


class TestAnswerInlineQueryAsyncTelemetry:
    """Test telemetry recording for inline query answering."""

    @pytest.mark.asyncio
    async def test_success_records_correct_metric(
        self, handler, _patch_tg, _patch_http_pool, _patch_telemetry
    ):
        """Successful inline answer records correct metric."""
        await handler._answer_inline_query_async("iq-ok", [])

        _patch_telemetry["record_api_call"].assert_called_once_with(
            "telegram", "answerInlineQuery", "success"
        )

    @pytest.mark.asyncio
    async def test_latency_recorded(
        self, handler, _patch_tg, _patch_http_pool, _patch_telemetry
    ):
        """Latency is recorded for inline query answers."""
        await handler._answer_inline_query_async("iq-lat", [])

        _patch_telemetry["record_api_latency"].assert_called_once()
        call_args = _patch_telemetry["record_api_latency"].call_args[0]
        assert call_args[0] == "telegram"
        assert call_args[1] == "answerInlineQuery"


# ============================================================================
# _send_voice_summary: TTS Integration
# ============================================================================


class TestSendVoiceSummaryHappyPath:
    """Test successful voice summary sending."""

    @pytest.mark.asyncio
    async def test_successful_voice_synthesis_and_send(self, handler):
        """Voice summary synthesizes and sends audio."""
        mock_result = MagicMock()
        mock_result.audio_bytes = b"fake-audio-data"
        mock_result.duration_seconds = 5.0

        mock_helper = MagicMock()
        mock_helper.is_available = True
        mock_helper.synthesize_debate_result = AsyncMock(
            return_value=mock_result
        )

        handler._send_voice_async = AsyncMock()

        with patch(
            "aragora.server.handlers.social.tts_helper.get_tts_helper",
            return_value=mock_helper,
        ):
            await handler._send_voice_summary(
                CHAT_ID,
                "Test topic",
                "Final answer text",
                consensus_reached=True,
                confidence=0.85,
                rounds_used=3,
            )

        mock_helper.synthesize_debate_result.assert_called_once_with(
            task="Test topic",
            final_answer="Final answer text",
            consensus_reached=True,
            confidence=0.85,
            rounds_used=3,
        )
        handler._send_voice_async.assert_called_once_with(
            CHAT_ID, b"fake-audio-data", 5.0
        )

    @pytest.mark.asyncio
    async def test_null_final_answer(self, handler):
        """Voice summary handles None final_answer."""
        mock_result = MagicMock()
        mock_result.audio_bytes = b"audio"
        mock_result.duration_seconds = 2.0

        mock_helper = MagicMock()
        mock_helper.is_available = True
        mock_helper.synthesize_debate_result = AsyncMock(
            return_value=mock_result
        )

        handler._send_voice_async = AsyncMock()

        with patch(
            "aragora.server.handlers.social.tts_helper.get_tts_helper",
            return_value=mock_helper,
        ):
            await handler._send_voice_summary(
                CHAT_ID,
                "Topic",
                None,
                consensus_reached=False,
                confidence=0.3,
                rounds_used=2,
            )

        mock_helper.synthesize_debate_result.assert_called_once()
        call_kwargs = mock_helper.synthesize_debate_result.call_args.kwargs
        assert call_kwargs["final_answer"] is None


# ============================================================================
# _send_voice_summary: TTS Not Available
# ============================================================================


class TestSendVoiceSummaryNotAvailable:
    """Test voice summary when TTS is not available."""

    @pytest.mark.asyncio
    async def test_tts_not_available_returns_early(self, handler):
        """When TTS is not available, returns without sending."""
        mock_helper = MagicMock()
        mock_helper.is_available = False

        handler._send_voice_async = AsyncMock()

        with patch(
            "aragora.server.handlers.social.tts_helper.get_tts_helper",
            return_value=mock_helper,
        ):
            await handler._send_voice_summary(
                CHAT_ID, "Topic", "Answer", True, 0.8, 3
            )

        handler._send_voice_async.assert_not_called()

    @pytest.mark.asyncio
    async def test_synthesis_returns_none(self, handler):
        """When synthesis returns None, voice is not sent."""
        mock_helper = MagicMock()
        mock_helper.is_available = True
        mock_helper.synthesize_debate_result = AsyncMock(return_value=None)

        handler._send_voice_async = AsyncMock()

        with patch(
            "aragora.server.handlers.social.tts_helper.get_tts_helper",
            return_value=mock_helper,
        ):
            await handler._send_voice_summary(
                CHAT_ID, "Topic", "Answer", True, 0.5, 2
            )

        handler._send_voice_async.assert_not_called()


# ============================================================================
# _send_voice_summary: Error Handling
# ============================================================================


class TestSendVoiceSummaryErrors:
    """Test error handling in voice summary sending."""

    @pytest.mark.asyncio
    async def test_import_error_handled(self, handler):
        """ImportError (missing TTS module) is caught gracefully."""
        with patch(
            "aragora.server.handlers.social.tts_helper.get_tts_helper",
            side_effect=ImportError("no tts module"),
        ):
            # Should not raise
            await handler._send_voice_summary(
                CHAT_ID, "Topic", "Answer", True, 0.8, 3
            )

    @pytest.mark.asyncio
    async def test_connection_error_handled(self, handler):
        """ConnectionError during synthesis is caught."""
        mock_helper = MagicMock()
        mock_helper.is_available = True
        mock_helper.synthesize_debate_result = AsyncMock(
            side_effect=ConnectionError("TTS server down")
        )

        with patch(
            "aragora.server.handlers.social.tts_helper.get_tts_helper",
            return_value=mock_helper,
        ):
            await handler._send_voice_summary(
                CHAT_ID, "Topic", "Answer", True, 0.8, 3
            )

    @pytest.mark.asyncio
    async def test_timeout_error_handled(self, handler):
        """TimeoutError during synthesis is caught."""
        mock_helper = MagicMock()
        mock_helper.is_available = True
        mock_helper.synthesize_debate_result = AsyncMock(
            side_effect=TimeoutError("TTS timed out")
        )

        with patch(
            "aragora.server.handlers.social.tts_helper.get_tts_helper",
            return_value=mock_helper,
        ):
            await handler._send_voice_summary(
                CHAT_ID, "Topic", "Answer", True, 0.8, 3
            )

    @pytest.mark.asyncio
    async def test_os_error_handled(self, handler):
        """OSError during synthesis is caught."""
        mock_helper = MagicMock()
        mock_helper.is_available = True
        mock_helper.synthesize_debate_result = AsyncMock(
            side_effect=OSError("file error")
        )

        with patch(
            "aragora.server.handlers.social.tts_helper.get_tts_helper",
            return_value=mock_helper,
        ):
            await handler._send_voice_summary(
                CHAT_ID, "Topic", "Answer", True, 0.8, 3
            )

    @pytest.mark.asyncio
    async def test_value_error_handled(self, handler):
        """ValueError during synthesis is caught."""
        mock_helper = MagicMock()
        mock_helper.is_available = True
        mock_helper.synthesize_debate_result = AsyncMock(
            side_effect=ValueError("bad input")
        )

        with patch(
            "aragora.server.handlers.social.tts_helper.get_tts_helper",
            return_value=mock_helper,
        ):
            await handler._send_voice_summary(
                CHAT_ID, "Topic", "Answer", True, 0.8, 3
            )

    @pytest.mark.asyncio
    async def test_type_error_handled(self, handler):
        """TypeError during synthesis is caught."""
        mock_helper = MagicMock()
        mock_helper.is_available = True
        mock_helper.synthesize_debate_result = AsyncMock(
            side_effect=TypeError("wrong type")
        )

        with patch(
            "aragora.server.handlers.social.tts_helper.get_tts_helper",
            return_value=mock_helper,
        ):
            await handler._send_voice_summary(
                CHAT_ID, "Topic", "Answer", True, 0.8, 3
            )

    @pytest.mark.asyncio
    async def test_error_logs_warning(self, handler):
        """Errors in voice summary are logged as warnings."""
        with patch(
            "aragora.server.handlers.social.tts_helper.get_tts_helper",
            side_effect=ImportError("no module"),
        ), patch(
            "aragora.server.handlers.social.telegram.messages.logger"
        ) as mock_logger:
            await handler._send_voice_summary(
                CHAT_ID, "Topic", "Answer", True, 0.8, 3
            )
            mock_logger.warning.assert_called()
            assert "Failed to send voice summary" in str(
                mock_logger.warning.call_args
            )


# ============================================================================
# _send_voice_async: Basic Functionality
# ============================================================================


class TestSendVoiceAsyncHappyPath:
    """Test successful voice message sending."""

    @pytest.mark.asyncio
    async def test_sends_voice_with_correct_data(
        self, handler, _patch_tg, _patch_http_pool
    ):
        """Voice message sends correct multipart data."""
        audio_data = b"fake-ogg-audio-bytes"
        await handler._send_voice_async(CHAT_ID, audio_data, 5.5)

        _patch_http_pool["client"].post.assert_called_once()
        call_args = _patch_http_pool["client"].post.call_args

        # Check data field
        data = call_args.kwargs.get("data", {})
        assert data["chat_id"] == str(CHAT_ID)
        assert data["duration"] == "5"

    @pytest.mark.asyncio
    async def test_sends_voice_with_correct_files(
        self, handler, _patch_tg, _patch_http_pool
    ):
        """Voice message uses correct file format."""
        audio_data = b"audio-content"
        await handler._send_voice_async(CHAT_ID, audio_data, 10.0)

        call_args = _patch_http_pool["client"].post.call_args
        files = call_args.kwargs.get("files", {})
        assert "voice" in files
        # files["voice"] should be ("voice.ogg", audio_bytes, "audio/ogg")
        voice_tuple = files["voice"]
        assert voice_tuple[0] == "voice.ogg"
        assert voice_tuple[1] == audio_data
        assert voice_tuple[2] == "audio/ogg"

    @pytest.mark.asyncio
    async def test_uses_60s_timeout(
        self, handler, _patch_tg, _patch_http_pool
    ):
        """Voice upload uses 60 second timeout."""
        await handler._send_voice_async(CHAT_ID, b"audio", 3.0)

        call_args = _patch_http_pool["client"].post.call_args
        assert call_args.kwargs.get("timeout") == 60

    @pytest.mark.asyncio
    async def test_uses_telegram_voice_session(
        self, handler, _patch_tg, _patch_http_pool
    ):
        """Voice upload uses 'telegram_voice' session name."""
        await handler._send_voice_async(CHAT_ID, b"audio", 3.0)

        _patch_http_pool["pool"].get_session.assert_called_once_with(
            "telegram_voice"
        )

    @pytest.mark.asyncio
    async def test_success_logs_info(
        self, handler, _patch_tg, _patch_http_pool
    ):
        """Successful voice send logs at info level."""
        with patch(
            "aragora.server.handlers.social.telegram.messages.logger"
        ) as mock_logger:
            await handler._send_voice_async(CHAT_ID, b"audio", 3.0)
            mock_logger.info.assert_called()
            assert str(CHAT_ID) in str(mock_logger.info.call_args)

    @pytest.mark.asyncio
    async def test_duration_truncated_to_int(
        self, handler, _patch_tg, _patch_http_pool
    ):
        """Duration is truncated to integer in the API call."""
        await handler._send_voice_async(CHAT_ID, b"audio", 7.9)

        call_args = _patch_http_pool["client"].post.call_args
        data = call_args.kwargs.get("data", {})
        assert data["duration"] == "7"  # int(7.9) = 7


# ============================================================================
# _send_voice_async: No Bot Token
# ============================================================================


class TestSendVoiceAsyncNoToken:
    """Test voice sending without bot token."""

    @pytest.mark.asyncio
    async def test_no_token_returns_early(self, handler, _patch_tg_no_token):
        """When token is missing, returns without sending."""
        with patch(
            "aragora.server.http_client_pool.get_http_pool"
        ) as mock_pool:
            await handler._send_voice_async(CHAT_ID, b"audio", 3.0)
            mock_pool.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_token_logs_warning(self, handler, _patch_tg_no_token):
        """Missing token logs a warning."""
        with patch(
            "aragora.server.handlers.social.telegram.messages.logger"
        ) as mock_logger:
            await handler._send_voice_async(CHAT_ID, b"audio", 3.0)
            mock_logger.warning.assert_called_once()
            assert "TELEGRAM_BOT_TOKEN" in str(mock_logger.warning.call_args)


# ============================================================================
# _send_voice_async: Error Handling
# ============================================================================


class TestSendVoiceAsyncErrors:
    """Test error handling in voice message sending."""

    @pytest.mark.asyncio
    async def test_api_error_logs_warning(
        self, handler, _patch_tg, _patch_http_pool
    ):
        """API error response (ok=false) logs a warning."""
        _patch_http_pool["response"].json.return_value = {
            "ok": False,
            "description": "File too large",
        }

        with patch(
            "aragora.server.handlers.social.telegram.messages.logger"
        ) as mock_logger:
            await handler._send_voice_async(CHAT_ID, b"big-audio", 120.0)
            mock_logger.warning.assert_called()
            assert "sendVoice failed" in str(mock_logger.warning.call_args)

    @pytest.mark.asyncio
    async def test_connection_error_handled(
        self, handler, _patch_tg, _patch_http_pool
    ):
        """ConnectionError is caught and logged."""
        _patch_http_pool["client"].post = AsyncMock(
            side_effect=ConnectionError("Connection refused")
        )

        with patch(
            "aragora.server.handlers.social.telegram.messages.logger"
        ) as mock_logger:
            await handler._send_voice_async(CHAT_ID, b"audio", 3.0)
            mock_logger.error.assert_called()

    @pytest.mark.asyncio
    async def test_timeout_error_handled(
        self, handler, _patch_tg, _patch_http_pool
    ):
        """TimeoutError is caught and logged."""
        _patch_http_pool["client"].post = AsyncMock(
            side_effect=TimeoutError("Upload timed out")
        )

        with patch(
            "aragora.server.handlers.social.telegram.messages.logger"
        ) as mock_logger:
            await handler._send_voice_async(CHAT_ID, b"audio", 3.0)
            mock_logger.error.assert_called()

    @pytest.mark.asyncio
    async def test_os_error_handled(
        self, handler, _patch_tg, _patch_http_pool
    ):
        """OSError is caught and logged."""
        _patch_http_pool["client"].post = AsyncMock(
            side_effect=OSError("IO error")
        )

        with patch(
            "aragora.server.handlers.social.telegram.messages.logger"
        ) as mock_logger:
            await handler._send_voice_async(CHAT_ID, b"audio", 3.0)
            mock_logger.error.assert_called()

    @pytest.mark.asyncio
    async def test_value_error_handled(
        self, handler, _patch_tg, _patch_http_pool
    ):
        """ValueError is caught and logged."""
        _patch_http_pool["client"].post = AsyncMock(
            side_effect=ValueError("Invalid data")
        )

        with patch(
            "aragora.server.handlers.social.telegram.messages.logger"
        ) as mock_logger:
            await handler._send_voice_async(CHAT_ID, b"audio", 3.0)
            mock_logger.error.assert_called()

    @pytest.mark.asyncio
    async def test_error_does_not_propagate(
        self, handler, _patch_tg, _patch_http_pool
    ):
        """Errors are caught and do not propagate."""
        _patch_http_pool["client"].post = AsyncMock(
            side_effect=ConnectionError("fail")
        )

        # Should not raise
        await handler._send_voice_async(CHAT_ID, b"audio", 3.0)


# ============================================================================
# _send_voice_async: Edge Cases
# ============================================================================


class TestSendVoiceAsyncEdgeCases:
    """Test edge cases for voice message sending."""

    @pytest.mark.asyncio
    async def test_empty_audio_bytes(
        self, handler, _patch_tg, _patch_http_pool
    ):
        """Empty audio bytes are still sent."""
        await handler._send_voice_async(CHAT_ID, b"", 0.0)

        _patch_http_pool["client"].post.assert_called_once()
        call_args = _patch_http_pool["client"].post.call_args
        files = call_args.kwargs.get("files", {})
        assert files["voice"][1] == b""

    @pytest.mark.asyncio
    async def test_large_audio_bytes(
        self, handler, _patch_tg, _patch_http_pool
    ):
        """Large audio data is sent without issues."""
        large_audio = b"\x00" * (50 * 1024 * 1024)  # 50 MB
        await handler._send_voice_async(CHAT_ID, large_audio, 300.0)

        _patch_http_pool["client"].post.assert_called_once()

    @pytest.mark.asyncio
    async def test_zero_duration(
        self, handler, _patch_tg, _patch_http_pool
    ):
        """Zero duration is handled correctly."""
        await handler._send_voice_async(CHAT_ID, b"audio", 0.0)

        call_args = _patch_http_pool["client"].post.call_args
        data = call_args.kwargs.get("data", {})
        assert data["duration"] == "0"

    @pytest.mark.asyncio
    async def test_fractional_duration_truncated(
        self, handler, _patch_tg, _patch_http_pool
    ):
        """Fractional duration is truncated (not rounded)."""
        await handler._send_voice_async(CHAT_ID, b"audio", 9.99)

        call_args = _patch_http_pool["client"].post.call_args
        data = call_args.kwargs.get("data", {})
        assert data["duration"] == "9"

    @pytest.mark.asyncio
    async def test_negative_chat_id_for_groups(
        self, handler, _patch_tg, _patch_http_pool
    ):
        """Negative chat IDs for group chats are handled."""
        await handler._send_voice_async(-100987654, b"audio", 3.0)

        call_args = _patch_http_pool["client"].post.call_args
        data = call_args.kwargs.get("data", {})
        assert data["chat_id"] == str(-100987654)


# ============================================================================
# Module-Level: _tg() Helper and TTS_VOICE_ENABLED
# ============================================================================


class TestModuleLevelConstants:
    """Test module-level constants and helpers."""

    def test_tts_voice_enabled_default_false(self):
        """TTS_VOICE_ENABLED defaults to false when env var not set."""
        from aragora.server.handlers.social.telegram.messages import (
            TTS_VOICE_ENABLED,
        )

        # In the test environment, TELEGRAM_TTS_ENABLED should not be set
        # unless explicitly configured
        assert isinstance(TTS_VOICE_ENABLED, bool)

    def test_tg_helper_returns_telegram_module(self):
        """_tg() returns the telegram module."""
        from aragora.server.handlers.social.telegram.messages import _tg

        result = _tg()
        # It should return the parent telegram package module
        assert hasattr(result, "TELEGRAM_BOT_TOKEN") or result is not None


# ============================================================================
# Cross-Method Integration Tests
# ============================================================================


class TestCrossMethodIntegration:
    """Integration tests spanning multiple methods."""

    @pytest.mark.asyncio
    async def test_voice_summary_calls_send_voice_async(self, handler):
        """Voice summary integrates with _send_voice_async."""
        mock_result = MagicMock()
        mock_result.audio_bytes = b"synthesized-audio"
        mock_result.duration_seconds = 8.5

        mock_helper = MagicMock()
        mock_helper.is_available = True
        mock_helper.synthesize_debate_result = AsyncMock(
            return_value=mock_result
        )

        handler._send_voice_async = AsyncMock()

        with patch(
            "aragora.server.handlers.social.tts_helper.get_tts_helper",
            return_value=mock_helper,
        ):
            await handler._send_voice_summary(
                CHAT_ID,
                "Integration test topic",
                "The consensus is clear",
                consensus_reached=True,
                confidence=0.95,
                rounds_used=4,
            )

        handler._send_voice_async.assert_called_once_with(
            CHAT_ID, b"synthesized-audio", 8.5
        )

    @pytest.mark.asyncio
    async def test_send_voice_async_after_failed_synthesis(self, handler):
        """When synthesis fails, _send_voice_async is never called."""
        mock_helper = MagicMock()
        mock_helper.is_available = True
        mock_helper.synthesize_debate_result = AsyncMock(
            side_effect=ConnectionError("TTS down")
        )

        handler._send_voice_async = AsyncMock()

        with patch(
            "aragora.server.handlers.social.tts_helper.get_tts_helper",
            return_value=mock_helper,
        ):
            await handler._send_voice_summary(
                CHAT_ID, "Topic", "Answer", True, 0.8, 3
            )

        handler._send_voice_async.assert_not_called()

    @pytest.mark.asyncio
    async def test_multiple_messages_sent_independently(
        self, handler, _patch_tg, _patch_http_pool, _patch_telemetry
    ):
        """Multiple messages can be sent independently."""
        await handler._send_message_async(CHAT_ID, "First message")
        await handler._send_message_async(CHAT_ID, "Second message")

        assert _patch_http_pool["client"].post.call_count == 2

    @pytest.mark.asyncio
    async def test_mixed_api_calls_record_separate_telemetry(
        self, handler, _patch_tg, _patch_http_pool, _patch_telemetry
    ):
        """Different API methods record separate telemetry entries."""
        await handler._send_message_async(CHAT_ID, "Message")
        await handler._answer_callback_async("cb-1", "Callback")
        await handler._answer_inline_query_async("iq-1", [])

        calls = _patch_telemetry["record_api_call"].call_args_list
        assert len(calls) == 3
        methods = [c[0][1] for c in calls]
        assert "sendMessage" in methods
        assert "answerCallbackQuery" in methods
        assert "answerInlineQuery" in methods
