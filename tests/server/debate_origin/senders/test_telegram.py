"""Tests for Telegram sender for debate origin result routing.

Tests cover:
1. Result message sending via Telegram Bot API
2. Receipt posting to Telegram chats
3. Error message delivery
4. Voice message sending
5. Authentication handling
6. Reply threading
7. Markdown formatting
8. Error handling for API failures
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import pytest

from aragora.server.debate_origin.models import DebateOrigin
from aragora.server.debate_origin.senders.telegram import (
    _send_telegram_result,
    _send_telegram_receipt,
    _send_telegram_error,
    _send_telegram_voice,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_origin() -> DebateOrigin:
    """Create a sample Telegram debate origin for testing."""
    return DebateOrigin(
        debate_id="debate-telegram-123",
        platform="telegram",
        channel_id="-1001234567890",
        user_id="user-telegram-456",
        message_id="12345",
        metadata={"topic": "Telegram Integration Test"},
    )


@pytest.fixture
def sample_origin_no_message_id() -> DebateOrigin:
    """Create a Telegram origin without message_id for threading tests."""
    return DebateOrigin(
        debate_id="debate-telegram-456",
        platform="telegram",
        channel_id="-1001234567890",
        user_id="user-telegram-789",
        metadata={"topic": "No Reply Test"},
    )


@pytest.fixture
def sample_result() -> dict[str, Any]:
    """Create a sample debate result for testing."""
    return {
        "consensus_reached": True,
        "final_answer": "The team reached agreement on the approach.",
        "confidence": 0.85,
        "participants": ["claude", "gpt-4", "gemini"],
        "task": "Evaluate the Telegram proposal",
    }


# =============================================================================
# Test: Send Telegram Result
# =============================================================================


class TestSendTelegramResult:
    """Tests for _send_telegram_result function."""

    @pytest.mark.asyncio
    async def test_sends_result_to_chat(self, sample_origin, sample_result):
        """_send_telegram_result posts message to Telegram chat."""
        mock_response = MagicMock()
        mock_response.is_success = True

        with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "test-bot-token"}):
            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_client.post = AsyncMock(return_value=mock_response)
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client_class.return_value = mock_client

                result = await _send_telegram_result(sample_origin, sample_result)

        assert result is True
        mock_client.post.assert_called_once()

        # Verify API call details
        call_args = mock_client.post.call_args
        assert "api.telegram.org" in call_args[0][0]
        assert "test-bot-token" in call_args[0][0]
        assert "/sendMessage" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_sends_with_markdown_parse_mode(self, sample_origin, sample_result):
        """_send_telegram_result uses Markdown parse mode."""
        mock_response = MagicMock()
        mock_response.is_success = True

        with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "test-token"}):
            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_client.post = AsyncMock(return_value=mock_response)
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client_class.return_value = mock_client

                await _send_telegram_result(sample_origin, sample_result)

        call_args = mock_client.post.call_args
        post_data = call_args[1]["json"]
        assert post_data["parse_mode"] == "Markdown"

    @pytest.mark.asyncio
    async def test_includes_chat_id(self, sample_origin, sample_result):
        """_send_telegram_result includes correct chat_id."""
        mock_response = MagicMock()
        mock_response.is_success = True

        with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "test-token"}):
            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_client.post = AsyncMock(return_value=mock_response)
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client_class.return_value = mock_client

                await _send_telegram_result(sample_origin, sample_result)

        call_args = mock_client.post.call_args
        post_data = call_args[1]["json"]
        assert post_data["chat_id"] == sample_origin.channel_id

    @pytest.mark.asyncio
    async def test_includes_reply_to_message_id(self, sample_origin, sample_result):
        """_send_telegram_result includes reply_to_message_id when message_id present."""
        mock_response = MagicMock()
        mock_response.is_success = True

        with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "test-token"}):
            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_client.post = AsyncMock(return_value=mock_response)
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client_class.return_value = mock_client

                await _send_telegram_result(sample_origin, sample_result)

        call_args = mock_client.post.call_args
        post_data = call_args[1]["json"]
        assert post_data["reply_to_message_id"] == sample_origin.message_id

    @pytest.mark.asyncio
    async def test_no_reply_without_message_id(self, sample_origin_no_message_id, sample_result):
        """_send_telegram_result omits reply_to_message_id when no message_id."""
        mock_response = MagicMock()
        mock_response.is_success = True

        with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "test-token"}):
            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_client.post = AsyncMock(return_value=mock_response)
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client_class.return_value = mock_client

                await _send_telegram_result(sample_origin_no_message_id, sample_result)

        call_args = mock_client.post.call_args
        post_data = call_args[1]["json"]
        assert "reply_to_message_id" not in post_data

    @pytest.mark.asyncio
    async def test_returns_false_without_bot_token(self, sample_origin, sample_result):
        """_send_telegram_result returns False when TELEGRAM_BOT_TOKEN not set."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("TELEGRAM_BOT_TOKEN", None)
            result = await _send_telegram_result(sample_origin, sample_result)

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_on_api_error(self, sample_origin, sample_result):
        """_send_telegram_result returns False on non-success HTTP status."""
        mock_response = MagicMock()
        mock_response.is_success = False
        mock_response.status_code = 403

        with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "test-token"}):
            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_client.post = AsyncMock(return_value=mock_response)
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client_class.return_value = mock_client

                result = await _send_telegram_result(sample_origin, sample_result)

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_on_connection_error(self, sample_origin, sample_result):
        """_send_telegram_result returns False on connection errors."""
        with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "test-token"}):
            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_client.post = AsyncMock(side_effect=OSError("Connection refused"))
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client_class.return_value = mock_client

                result = await _send_telegram_result(sample_origin, sample_result)

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_on_timeout(self, sample_origin, sample_result):
        """_send_telegram_result returns False on timeout errors."""
        with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "test-token"}):
            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_client.post = AsyncMock(side_effect=TimeoutError("Request timed out"))
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client_class.return_value = mock_client

                result = await _send_telegram_result(sample_origin, sample_result)

        assert result is False


# =============================================================================
# Test: Send Telegram Receipt
# =============================================================================


class TestSendTelegramReceipt:
    """Tests for _send_telegram_receipt function."""

    @pytest.mark.asyncio
    async def test_posts_receipt_summary(self, sample_origin):
        """_send_telegram_receipt posts receipt summary to chat."""
        mock_response = MagicMock()
        mock_response.is_success = True

        with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "test-token"}):
            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_client.post = AsyncMock(return_value=mock_response)
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client_class.return_value = mock_client

                result = await _send_telegram_receipt(
                    sample_origin, "Receipt: Approved with 95% confidence"
                )

        assert result is True
        call_args = mock_client.post.call_args
        post_data = call_args[1]["json"]
        assert post_data["text"] == "Receipt: Approved with 95% confidence"
        assert post_data["parse_mode"] == "Markdown"

    @pytest.mark.asyncio
    async def test_includes_reply_reference(self, sample_origin):
        """_send_telegram_receipt replies to original message when message_id present."""
        mock_response = MagicMock()
        mock_response.is_success = True

        with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "test-token"}):
            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_client.post = AsyncMock(return_value=mock_response)
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client_class.return_value = mock_client

                await _send_telegram_receipt(sample_origin, "Receipt summary")

        call_args = mock_client.post.call_args
        post_data = call_args[1]["json"]
        assert post_data["reply_to_message_id"] == sample_origin.message_id

    @pytest.mark.asyncio
    async def test_returns_false_without_token(self, sample_origin):
        """_send_telegram_receipt returns False when token not configured."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("TELEGRAM_BOT_TOKEN", None)
            result = await _send_telegram_receipt(sample_origin, "Receipt")

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_on_api_failure(self, sample_origin):
        """_send_telegram_receipt returns False on API error."""
        mock_response = MagicMock()
        mock_response.is_success = False

        with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "test-token"}):
            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_client.post = AsyncMock(return_value=mock_response)
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client_class.return_value = mock_client

                result = await _send_telegram_receipt(sample_origin, "Receipt")

        assert result is False

    @pytest.mark.asyncio
    async def test_handles_connection_error(self, sample_origin):
        """_send_telegram_receipt handles connection errors gracefully."""
        with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "test-token"}):
            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_client.post = AsyncMock(side_effect=OSError("Network error"))
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client_class.return_value = mock_client

                result = await _send_telegram_receipt(sample_origin, "Receipt")

        assert result is False


# =============================================================================
# Test: Send Telegram Error
# =============================================================================


class TestSendTelegramError:
    """Tests for _send_telegram_error function."""

    @pytest.mark.asyncio
    async def test_sends_error_message(self, sample_origin):
        """_send_telegram_error sends error message to chat."""
        mock_response = MagicMock()
        mock_response.is_success = True

        with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "test-token"}):
            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_client.post = AsyncMock(return_value=mock_response)
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client_class.return_value = mock_client

                result = await _send_telegram_error(sample_origin, "An error occurred")

        assert result is True
        call_args = mock_client.post.call_args
        post_data = call_args[1]["json"]
        assert post_data["text"] == "An error occurred"

    @pytest.mark.asyncio
    async def test_includes_reply_reference_for_error(self, sample_origin):
        """_send_telegram_error replies to original message."""
        mock_response = MagicMock()
        mock_response.is_success = True

        with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "test-token"}):
            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_client.post = AsyncMock(return_value=mock_response)
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client_class.return_value = mock_client

                await _send_telegram_error(sample_origin, "Error message")

        call_args = mock_client.post.call_args
        post_data = call_args[1]["json"]
        assert "reply_to_message_id" in post_data

    @pytest.mark.asyncio
    async def test_returns_false_without_token(self, sample_origin):
        """_send_telegram_error returns False when token not configured."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("TELEGRAM_BOT_TOKEN", None)
            result = await _send_telegram_error(sample_origin, "Error")

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_api_success_status(self, sample_origin):
        """_send_telegram_error returns the API success status."""
        mock_response = MagicMock()
        mock_response.is_success = False

        with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "test-token"}):
            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_client.post = AsyncMock(return_value=mock_response)
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client_class.return_value = mock_client

                result = await _send_telegram_error(sample_origin, "Error")

        assert result is False


# =============================================================================
# Test: Send Telegram Voice
# =============================================================================


class TestSendTelegramVoice:
    """Tests for _send_telegram_voice function."""

    @pytest.mark.asyncio
    async def test_sends_voice_message(self, sample_origin, sample_result):
        """_send_telegram_voice sends voice file via sendVoice API."""
        mock_response = MagicMock()
        mock_response.is_success = True

        with patch(
            "aragora.server.debate_origin.senders.telegram._synthesize_voice",
            new_callable=AsyncMock,
            return_value="/tmp/test_audio.ogg",
        ):
            with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "test-token"}):
                with patch("httpx.AsyncClient") as mock_client_class:
                    mock_client = AsyncMock()
                    mock_client.post = AsyncMock(return_value=mock_response)
                    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                    mock_client.__aexit__ = AsyncMock(return_value=None)
                    mock_client_class.return_value = mock_client

                    with patch("builtins.open", mock_open(read_data=b"audio data")):
                        with patch.object(Path, "unlink"):
                            result = await _send_telegram_voice(sample_origin, sample_result)

        assert result is True

        # Verify sendVoice endpoint used
        call_args = mock_client.post.call_args
        assert "/sendVoice" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_includes_chat_id_in_voice_request(self, sample_origin, sample_result):
        """_send_telegram_voice includes chat_id in request."""
        mock_response = MagicMock()
        mock_response.is_success = True

        with patch(
            "aragora.server.debate_origin.senders.telegram._synthesize_voice",
            new_callable=AsyncMock,
            return_value="/tmp/audio.ogg",
        ):
            with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "test-token"}):
                with patch("httpx.AsyncClient") as mock_client_class:
                    mock_client = AsyncMock()
                    mock_client.post = AsyncMock(return_value=mock_response)
                    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                    mock_client.__aexit__ = AsyncMock(return_value=None)
                    mock_client_class.return_value = mock_client

                    with patch("builtins.open", mock_open(read_data=b"audio")):
                        with patch.object(Path, "unlink"):
                            await _send_telegram_voice(sample_origin, sample_result)

        call_args = mock_client.post.call_args
        post_data = call_args[1]["data"]
        assert post_data["chat_id"] == sample_origin.channel_id

    @pytest.mark.asyncio
    async def test_includes_reply_for_voice(self, sample_origin, sample_result):
        """_send_telegram_voice replies to original message."""
        mock_response = MagicMock()
        mock_response.is_success = True

        with patch(
            "aragora.server.debate_origin.senders.telegram._synthesize_voice",
            new_callable=AsyncMock,
            return_value="/tmp/audio.ogg",
        ):
            with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "test-token"}):
                with patch("httpx.AsyncClient") as mock_client_class:
                    mock_client = AsyncMock()
                    mock_client.post = AsyncMock(return_value=mock_response)
                    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                    mock_client.__aexit__ = AsyncMock(return_value=None)
                    mock_client_class.return_value = mock_client

                    with patch("builtins.open", mock_open(read_data=b"audio")):
                        with patch.object(Path, "unlink"):
                            await _send_telegram_voice(sample_origin, sample_result)

        call_args = mock_client.post.call_args
        post_data = call_args[1]["data"]
        assert post_data["reply_to_message_id"] == sample_origin.message_id

    @pytest.mark.asyncio
    async def test_returns_false_when_synthesis_fails(self, sample_origin, sample_result):
        """_send_telegram_voice returns False when TTS synthesis fails."""
        with patch(
            "aragora.server.debate_origin.senders.telegram._synthesize_voice",
            new_callable=AsyncMock,
            return_value=None,
        ):
            result = await _send_telegram_voice(sample_origin, sample_result)

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_without_token(self, sample_origin, sample_result):
        """_send_telegram_voice returns False when token not configured."""
        with patch(
            "aragora.server.debate_origin.senders.telegram._synthesize_voice",
            new_callable=AsyncMock,
            return_value="/tmp/audio.ogg",
        ):
            with patch.dict(os.environ, {}, clear=True):
                os.environ.pop("TELEGRAM_BOT_TOKEN", None)
                result = await _send_telegram_voice(sample_origin, sample_result)

        assert result is False

    @pytest.mark.asyncio
    async def test_cleans_up_temp_file(self, sample_origin, sample_result):
        """_send_telegram_voice cleans up temp audio file after sending."""
        mock_response = MagicMock()
        mock_response.is_success = True

        with patch(
            "aragora.server.debate_origin.senders.telegram._synthesize_voice",
            new_callable=AsyncMock,
            return_value="/tmp/cleanup_test.ogg",
        ):
            with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "test-token"}):
                with patch("httpx.AsyncClient") as mock_client_class:
                    mock_client = AsyncMock()
                    mock_client.post = AsyncMock(return_value=mock_response)
                    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                    mock_client.__aexit__ = AsyncMock(return_value=None)
                    mock_client_class.return_value = mock_client

                    with patch("builtins.open", mock_open(read_data=b"audio")):
                        with patch.object(Path, "unlink") as mock_unlink:
                            await _send_telegram_voice(sample_origin, sample_result)

        mock_unlink.assert_called_once_with(missing_ok=True)

    @pytest.mark.asyncio
    async def test_cleans_up_on_failure(self, sample_origin, sample_result):
        """_send_telegram_voice cleans up temp file even on send failure."""
        mock_response = MagicMock()
        mock_response.is_success = False
        mock_response.status_code = 500

        with patch(
            "aragora.server.debate_origin.senders.telegram._synthesize_voice",
            new_callable=AsyncMock,
            return_value="/tmp/failure_cleanup.ogg",
        ):
            with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "test-token"}):
                with patch("httpx.AsyncClient") as mock_client_class:
                    mock_client = AsyncMock()
                    mock_client.post = AsyncMock(return_value=mock_response)
                    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                    mock_client.__aexit__ = AsyncMock(return_value=None)
                    mock_client_class.return_value = mock_client

                    with patch("builtins.open", mock_open(read_data=b"audio")):
                        with patch.object(Path, "unlink") as mock_unlink:
                            await _send_telegram_voice(sample_origin, sample_result)

        mock_unlink.assert_called_once_with(missing_ok=True)

    @pytest.mark.asyncio
    async def test_handles_cleanup_error_gracefully(self, sample_origin, sample_result):
        """_send_telegram_voice handles cleanup errors without raising."""
        mock_response = MagicMock()
        mock_response.is_success = True

        with patch(
            "aragora.server.debate_origin.senders.telegram._synthesize_voice",
            new_callable=AsyncMock,
            return_value="/tmp/cleanup_error.ogg",
        ):
            with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "test-token"}):
                with patch("httpx.AsyncClient") as mock_client_class:
                    mock_client = AsyncMock()
                    mock_client.post = AsyncMock(return_value=mock_response)
                    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                    mock_client.__aexit__ = AsyncMock(return_value=None)
                    mock_client_class.return_value = mock_client

                    with patch("builtins.open", mock_open(read_data=b"audio")):
                        with patch.object(Path, "unlink", side_effect=OSError("Permission denied")):
                            # Should not raise
                            result = await _send_telegram_voice(sample_origin, sample_result)

        assert result is True

    @pytest.mark.asyncio
    async def test_uses_longer_timeout_for_voice(self, sample_origin, sample_result):
        """_send_telegram_voice uses 60s timeout for file upload."""
        mock_response = MagicMock()
        mock_response.is_success = True

        with patch(
            "aragora.server.debate_origin.senders.telegram._synthesize_voice",
            new_callable=AsyncMock,
            return_value="/tmp/audio.ogg",
        ):
            with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "test-token"}):
                with patch("httpx.AsyncClient") as mock_client_class:
                    mock_client = AsyncMock()
                    mock_client.post = AsyncMock(return_value=mock_response)
                    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                    mock_client.__aexit__ = AsyncMock(return_value=None)
                    mock_client_class.return_value = mock_client

                    with patch("builtins.open", mock_open(read_data=b"audio")):
                        with patch.object(Path, "unlink"):
                            await _send_telegram_voice(sample_origin, sample_result)

        # Verify timeout was set to 60s
        mock_client_class.assert_called_once_with(timeout=60.0)

    @pytest.mark.asyncio
    async def test_handles_http_error(self, sample_origin, sample_result):
        """_send_telegram_voice handles HTTP errors gracefully."""
        import httpx

        with patch(
            "aragora.server.debate_origin.senders.telegram._synthesize_voice",
            new_callable=AsyncMock,
            return_value="/tmp/audio.ogg",
        ):
            with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "test-token"}):
                with patch("httpx.AsyncClient") as mock_client_class:
                    mock_client = AsyncMock()
                    mock_client.post = AsyncMock(side_effect=httpx.HTTPError("Connection failed"))
                    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                    mock_client.__aexit__ = AsyncMock(return_value=None)
                    mock_client_class.return_value = mock_client

                    with patch("builtins.open", mock_open(read_data=b"audio")):
                        with patch.object(Path, "unlink"):
                            result = await _send_telegram_voice(sample_origin, sample_result)

        assert result is False
