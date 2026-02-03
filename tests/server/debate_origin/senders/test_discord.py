"""Tests for Discord sender for debate origin result routing.

Tests cover:
1. Result message sending via Discord API
2. Receipt posting to Discord channels
3. Error message delivery
4. Voice message sending with audio attachment
5. Authentication handling
6. Reply threading
7. Error handling for API failures
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import pytest

from aragora.server.debate_origin.models import DebateOrigin
from aragora.server.debate_origin.senders.discord import (
    _send_discord_result,
    _send_discord_receipt,
    _send_discord_error,
    _send_discord_voice,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_origin() -> DebateOrigin:
    """Create a sample Discord debate origin for testing."""
    return DebateOrigin(
        debate_id="debate-discord-123",
        platform="discord",
        channel_id="1234567890123456789",
        user_id="user-discord-456",
        message_id="9876543210987654321",
        metadata={"topic": "Discord Integration Test"},
    )


@pytest.fixture
def sample_origin_no_message_id() -> DebateOrigin:
    """Create a Discord origin without message_id for threading tests."""
    return DebateOrigin(
        debate_id="debate-discord-456",
        platform="discord",
        channel_id="1234567890123456789",
        user_id="user-discord-789",
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
        "task": "Evaluate the Discord proposal",
    }


# =============================================================================
# Test: Send Discord Result
# =============================================================================


class TestSendDiscordResult:
    """Tests for _send_discord_result function."""

    @pytest.mark.asyncio
    async def test_sends_result_to_channel(self, sample_origin, sample_result):
        """_send_discord_result posts message to Discord channel."""
        mock_response = MagicMock()
        mock_response.is_success = True

        with patch.dict(os.environ, {"DISCORD_BOT_TOKEN": "test-token"}):
            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_client.post = AsyncMock(return_value=mock_response)
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client_class.return_value = mock_client

                result = await _send_discord_result(sample_origin, sample_result)

        assert result is True
        mock_client.post.assert_called_once()

        # Verify API call details
        call_args = mock_client.post.call_args
        assert f"/channels/{sample_origin.channel_id}/messages" in call_args[0][0]
        assert call_args[1]["headers"]["Authorization"] == "Bot test-token"

    @pytest.mark.asyncio
    async def test_includes_message_reference_for_reply(self, sample_origin, sample_result):
        """_send_discord_result includes message_reference when message_id present."""
        mock_response = MagicMock()
        mock_response.is_success = True

        with patch.dict(os.environ, {"DISCORD_BOT_TOKEN": "test-token"}):
            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_client.post = AsyncMock(return_value=mock_response)
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client_class.return_value = mock_client

                await _send_discord_result(sample_origin, sample_result)

        call_args = mock_client.post.call_args
        post_data = call_args[1]["json"]
        assert "message_reference" in post_data
        assert post_data["message_reference"]["message_id"] == sample_origin.message_id

    @pytest.mark.asyncio
    async def test_no_message_reference_without_message_id(
        self, sample_origin_no_message_id, sample_result
    ):
        """_send_discord_result omits message_reference when no message_id."""
        mock_response = MagicMock()
        mock_response.is_success = True

        with patch.dict(os.environ, {"DISCORD_BOT_TOKEN": "test-token"}):
            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_client.post = AsyncMock(return_value=mock_response)
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client_class.return_value = mock_client

                await _send_discord_result(sample_origin_no_message_id, sample_result)

        call_args = mock_client.post.call_args
        post_data = call_args[1]["json"]
        assert "message_reference" not in post_data

    @pytest.mark.asyncio
    async def test_returns_false_without_bot_token(self, sample_origin, sample_result):
        """_send_discord_result returns False when DISCORD_BOT_TOKEN not set."""
        with patch.dict(os.environ, {}, clear=True):
            # Ensure DISCORD_BOT_TOKEN is not in environ
            os.environ.pop("DISCORD_BOT_TOKEN", None)
            result = await _send_discord_result(sample_origin, sample_result)

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_on_api_error(self, sample_origin, sample_result):
        """_send_discord_result returns False on non-success HTTP status."""
        mock_response = MagicMock()
        mock_response.is_success = False
        mock_response.status_code = 403

        with patch.dict(os.environ, {"DISCORD_BOT_TOKEN": "test-token"}):
            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_client.post = AsyncMock(return_value=mock_response)
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client_class.return_value = mock_client

                result = await _send_discord_result(sample_origin, sample_result)

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_on_connection_error(self, sample_origin, sample_result):
        """_send_discord_result returns False on connection errors."""
        with patch.dict(os.environ, {"DISCORD_BOT_TOKEN": "test-token"}):
            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_client.post = AsyncMock(side_effect=OSError("Connection refused"))
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client_class.return_value = mock_client

                result = await _send_discord_result(sample_origin, sample_result)

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_on_timeout(self, sample_origin, sample_result):
        """_send_discord_result returns False on timeout errors."""
        with patch.dict(os.environ, {"DISCORD_BOT_TOKEN": "test-token"}):
            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_client.post = AsyncMock(side_effect=TimeoutError("Request timed out"))
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client_class.return_value = mock_client

                result = await _send_discord_result(sample_origin, sample_result)

        assert result is False

    @pytest.mark.asyncio
    async def test_uses_correct_api_version(self, sample_origin, sample_result):
        """_send_discord_result uses Discord API v10."""
        mock_response = MagicMock()
        mock_response.is_success = True

        with patch.dict(os.environ, {"DISCORD_BOT_TOKEN": "test-token"}):
            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_client.post = AsyncMock(return_value=mock_response)
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client_class.return_value = mock_client

                await _send_discord_result(sample_origin, sample_result)

        call_args = mock_client.post.call_args
        assert "discord.com/api/v10" in call_args[0][0]


# =============================================================================
# Test: Send Discord Receipt
# =============================================================================


class TestSendDiscordReceipt:
    """Tests for _send_discord_receipt function."""

    @pytest.mark.asyncio
    async def test_posts_receipt_summary(self, sample_origin):
        """_send_discord_receipt posts receipt summary to channel."""
        mock_response = MagicMock()
        mock_response.is_success = True

        with patch.dict(os.environ, {"DISCORD_BOT_TOKEN": "test-token"}):
            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_client.post = AsyncMock(return_value=mock_response)
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client_class.return_value = mock_client

                result = await _send_discord_receipt(
                    sample_origin, "Receipt: Approved with 95% confidence"
                )

        assert result is True
        call_args = mock_client.post.call_args
        post_data = call_args[1]["json"]
        assert post_data["content"] == "Receipt: Approved with 95% confidence"

    @pytest.mark.asyncio
    async def test_includes_reply_reference(self, sample_origin):
        """_send_discord_receipt replies to original message when message_id present."""
        mock_response = MagicMock()
        mock_response.is_success = True

        with patch.dict(os.environ, {"DISCORD_BOT_TOKEN": "test-token"}):
            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_client.post = AsyncMock(return_value=mock_response)
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client_class.return_value = mock_client

                await _send_discord_receipt(sample_origin, "Receipt summary")

        call_args = mock_client.post.call_args
        post_data = call_args[1]["json"]
        assert post_data["message_reference"]["message_id"] == sample_origin.message_id

    @pytest.mark.asyncio
    async def test_returns_false_without_token(self, sample_origin):
        """_send_discord_receipt returns False when token not configured."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("DISCORD_BOT_TOKEN", None)
            result = await _send_discord_receipt(sample_origin, "Receipt")

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_on_api_failure(self, sample_origin):
        """_send_discord_receipt returns False on API error."""
        mock_response = MagicMock()
        mock_response.is_success = False

        with patch.dict(os.environ, {"DISCORD_BOT_TOKEN": "test-token"}):
            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_client.post = AsyncMock(return_value=mock_response)
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client_class.return_value = mock_client

                result = await _send_discord_receipt(sample_origin, "Receipt")

        assert result is False


# =============================================================================
# Test: Send Discord Error
# =============================================================================


class TestSendDiscordError:
    """Tests for _send_discord_error function."""

    @pytest.mark.asyncio
    async def test_sends_error_message(self, sample_origin):
        """_send_discord_error sends error message to channel."""
        mock_response = MagicMock()
        mock_response.is_success = True

        with patch.dict(os.environ, {"DISCORD_BOT_TOKEN": "test-token"}):
            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_client.post = AsyncMock(return_value=mock_response)
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client_class.return_value = mock_client

                result = await _send_discord_error(sample_origin, "An error occurred")

        assert result is True
        call_args = mock_client.post.call_args
        post_data = call_args[1]["json"]
        assert post_data["content"] == "An error occurred"

    @pytest.mark.asyncio
    async def test_includes_reply_reference_for_error(self, sample_origin):
        """_send_discord_error replies to original message."""
        mock_response = MagicMock()
        mock_response.is_success = True

        with patch.dict(os.environ, {"DISCORD_BOT_TOKEN": "test-token"}):
            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_client.post = AsyncMock(return_value=mock_response)
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client_class.return_value = mock_client

                await _send_discord_error(sample_origin, "Error message")

        call_args = mock_client.post.call_args
        post_data = call_args[1]["json"]
        assert "message_reference" in post_data

    @pytest.mark.asyncio
    async def test_returns_false_without_token(self, sample_origin):
        """_send_discord_error returns False when token not configured."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("DISCORD_BOT_TOKEN", None)
            result = await _send_discord_error(sample_origin, "Error")

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_api_success_status(self, sample_origin):
        """_send_discord_error returns the API success status."""
        mock_response = MagicMock()
        mock_response.is_success = False

        with patch.dict(os.environ, {"DISCORD_BOT_TOKEN": "test-token"}):
            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_client.post = AsyncMock(return_value=mock_response)
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client_class.return_value = mock_client

                result = await _send_discord_error(sample_origin, "Error")

        assert result is False

    @pytest.mark.asyncio
    async def test_handles_connection_error(self, sample_origin):
        """_send_discord_error handles connection errors gracefully."""
        with patch.dict(os.environ, {"DISCORD_BOT_TOKEN": "test-token"}):
            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_client.post = AsyncMock(side_effect=OSError("Network error"))
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client_class.return_value = mock_client

                result = await _send_discord_error(sample_origin, "Error")

        assert result is False


# =============================================================================
# Test: Send Discord Voice
# =============================================================================


class TestSendDiscordVoice:
    """Tests for _send_discord_voice function."""

    @pytest.mark.asyncio
    async def test_sends_voice_as_attachment(self, sample_origin, sample_result):
        """_send_discord_voice sends audio file as attachment."""
        mock_response = MagicMock()
        mock_response.is_success = True

        with patch(
            "aragora.server.debate_origin.senders.discord._synthesize_voice",
            new_callable=AsyncMock,
            return_value="/tmp/test_audio.ogg",
        ):
            with patch.dict(os.environ, {"DISCORD_BOT_TOKEN": "test-token"}):
                with patch("httpx.AsyncClient") as mock_client_class:
                    mock_client = AsyncMock()
                    mock_client.post = AsyncMock(return_value=mock_response)
                    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                    mock_client.__aexit__ = AsyncMock(return_value=None)
                    mock_client_class.return_value = mock_client

                    with patch("builtins.open", mock_open(read_data=b"audio data")):
                        with patch.object(Path, "unlink"):
                            result = await _send_discord_voice(sample_origin, sample_result)

        assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_when_synthesis_fails(self, sample_origin, sample_result):
        """_send_discord_voice returns False when TTS synthesis fails."""
        with patch(
            "aragora.server.debate_origin.senders.discord._synthesize_voice",
            new_callable=AsyncMock,
            return_value=None,
        ):
            result = await _send_discord_voice(sample_origin, sample_result)

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_without_token(self, sample_origin, sample_result):
        """_send_discord_voice returns False when token not configured."""
        with patch(
            "aragora.server.debate_origin.senders.discord._synthesize_voice",
            new_callable=AsyncMock,
            return_value="/tmp/audio.ogg",
        ):
            with patch.dict(os.environ, {}, clear=True):
                os.environ.pop("DISCORD_BOT_TOKEN", None)
                result = await _send_discord_voice(sample_origin, sample_result)

        assert result is False

    @pytest.mark.asyncio
    async def test_cleans_up_temp_file(self, sample_origin, sample_result):
        """_send_discord_voice cleans up temp audio file after sending."""
        mock_response = MagicMock()
        mock_response.is_success = True
        mock_path = MagicMock()

        with patch(
            "aragora.server.debate_origin.senders.discord._synthesize_voice",
            new_callable=AsyncMock,
            return_value="/tmp/cleanup_test.ogg",
        ):
            with patch.dict(os.environ, {"DISCORD_BOT_TOKEN": "test-token"}):
                with patch("httpx.AsyncClient") as mock_client_class:
                    mock_client = AsyncMock()
                    mock_client.post = AsyncMock(return_value=mock_response)
                    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                    mock_client.__aexit__ = AsyncMock(return_value=None)
                    mock_client_class.return_value = mock_client

                    with patch("builtins.open", mock_open(read_data=b"audio")):
                        with patch.object(Path, "unlink") as mock_unlink:
                            await _send_discord_voice(sample_origin, sample_result)

        mock_unlink.assert_called_once_with(missing_ok=True)

    @pytest.mark.asyncio
    async def test_cleans_up_on_failure(self, sample_origin, sample_result):
        """_send_discord_voice cleans up temp file even on send failure."""
        mock_response = MagicMock()
        mock_response.is_success = False
        mock_response.status_code = 500

        with patch(
            "aragora.server.debate_origin.senders.discord._synthesize_voice",
            new_callable=AsyncMock,
            return_value="/tmp/failure_cleanup.ogg",
        ):
            with patch.dict(os.environ, {"DISCORD_BOT_TOKEN": "test-token"}):
                with patch("httpx.AsyncClient") as mock_client_class:
                    mock_client = AsyncMock()
                    mock_client.post = AsyncMock(return_value=mock_response)
                    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                    mock_client.__aexit__ = AsyncMock(return_value=None)
                    mock_client_class.return_value = mock_client

                    with patch("builtins.open", mock_open(read_data=b"audio")):
                        with patch.object(Path, "unlink") as mock_unlink:
                            await _send_discord_voice(sample_origin, sample_result)

        mock_unlink.assert_called_once_with(missing_ok=True)

    @pytest.mark.asyncio
    async def test_handles_cleanup_error_gracefully(self, sample_origin, sample_result):
        """_send_discord_voice handles cleanup errors without raising."""
        mock_response = MagicMock()
        mock_response.is_success = True

        with patch(
            "aragora.server.debate_origin.senders.discord._synthesize_voice",
            new_callable=AsyncMock,
            return_value="/tmp/cleanup_error.ogg",
        ):
            with patch.dict(os.environ, {"DISCORD_BOT_TOKEN": "test-token"}):
                with patch("httpx.AsyncClient") as mock_client_class:
                    mock_client = AsyncMock()
                    mock_client.post = AsyncMock(return_value=mock_response)
                    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                    mock_client.__aexit__ = AsyncMock(return_value=None)
                    mock_client_class.return_value = mock_client

                    with patch("builtins.open", mock_open(read_data=b"audio")):
                        with patch.object(Path, "unlink", side_effect=OSError("Permission denied")):
                            # Should not raise
                            result = await _send_discord_voice(sample_origin, sample_result)

        assert result is True

    @pytest.mark.asyncio
    async def test_uses_longer_timeout_for_voice(self, sample_origin, sample_result):
        """_send_discord_voice uses 60s timeout for file upload."""
        mock_response = MagicMock()
        mock_response.is_success = True

        with patch(
            "aragora.server.debate_origin.senders.discord._synthesize_voice",
            new_callable=AsyncMock,
            return_value="/tmp/audio.ogg",
        ):
            with patch.dict(os.environ, {"DISCORD_BOT_TOKEN": "test-token"}):
                with patch("httpx.AsyncClient") as mock_client_class:
                    mock_client = AsyncMock()
                    mock_client.post = AsyncMock(return_value=mock_response)
                    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                    mock_client.__aexit__ = AsyncMock(return_value=None)
                    mock_client_class.return_value = mock_client

                    with patch("builtins.open", mock_open(read_data=b"audio")):
                        with patch.object(Path, "unlink"):
                            await _send_discord_voice(sample_origin, sample_result)

        # Verify timeout was set to 60s
        mock_client_class.assert_called_once_with(timeout=60.0)
