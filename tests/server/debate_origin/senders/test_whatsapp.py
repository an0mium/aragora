"""Tests for WhatsApp sender for debate origin result routing.

Tests cover:
1. Result message sending via WhatsApp Cloud API
2. Voice message sending with media upload
3. Authentication handling (access token and phone number ID)
4. Media upload flow
5. Error handling for API failures
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import pytest

from aragora.server.debate_origin.models import DebateOrigin
from aragora.server.debate_origin.senders.whatsapp import (
    _send_whatsapp_result,
    _send_whatsapp_voice,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_origin() -> DebateOrigin:
    """Create a sample WhatsApp debate origin for testing."""
    return DebateOrigin(
        debate_id="debate-whatsapp-123",
        platform="whatsapp",
        channel_id="15551234567",  # Phone number
        user_id="user-whatsapp-456",
        metadata={"topic": "WhatsApp Integration Test"},
    )


@pytest.fixture
def sample_result() -> dict[str, Any]:
    """Create a sample debate result for testing."""
    return {
        "consensus_reached": True,
        "final_answer": "The team reached agreement on the approach.",
        "confidence": 0.85,
        "participants": ["claude", "gpt-4", "gemini"],
        "task": "Evaluate the WhatsApp proposal",
    }


@pytest.fixture
def whatsapp_env():
    """Environment variables for WhatsApp API."""
    return {
        "WHATSAPP_ACCESS_TOKEN": "test-access-token-12345",
        "WHATSAPP_PHONE_NUMBER_ID": "123456789012345",
    }


# =============================================================================
# Test: Send WhatsApp Result
# =============================================================================


class TestSendWhatsAppResult:
    """Tests for _send_whatsapp_result function."""

    @pytest.mark.asyncio
    async def test_sends_result_to_recipient(self, sample_origin, sample_result, whatsapp_env):
        """_send_whatsapp_result posts message to WhatsApp recipient."""
        mock_response = MagicMock()
        mock_response.is_success = True

        with patch.dict(os.environ, whatsapp_env):
            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_client.post = AsyncMock(return_value=mock_response)
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client_class.return_value = mock_client

                result = await _send_whatsapp_result(sample_origin, sample_result)

        assert result is True
        mock_client.post.assert_called_once()

        # Verify API call details
        call_args = mock_client.post.call_args
        assert "graph.facebook.com" in call_args[0][0]
        assert whatsapp_env["WHATSAPP_PHONE_NUMBER_ID"] in call_args[0][0]
        assert "/messages" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_uses_bearer_auth(self, sample_origin, sample_result, whatsapp_env):
        """_send_whatsapp_result uses Bearer token authentication."""
        mock_response = MagicMock()
        mock_response.is_success = True

        with patch.dict(os.environ, whatsapp_env):
            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_client.post = AsyncMock(return_value=mock_response)
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client_class.return_value = mock_client

                await _send_whatsapp_result(sample_origin, sample_result)

        call_args = mock_client.post.call_args
        headers = call_args[1]["headers"]
        assert headers["Authorization"] == f"Bearer {whatsapp_env['WHATSAPP_ACCESS_TOKEN']}"

    @pytest.mark.asyncio
    async def test_includes_correct_payload_structure(
        self, sample_origin, sample_result, whatsapp_env
    ):
        """_send_whatsapp_result sends correct WhatsApp Cloud API payload."""
        mock_response = MagicMock()
        mock_response.is_success = True

        with patch.dict(os.environ, whatsapp_env):
            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_client.post = AsyncMock(return_value=mock_response)
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client_class.return_value = mock_client

                await _send_whatsapp_result(sample_origin, sample_result)

        call_args = mock_client.post.call_args
        post_data = call_args[1]["json"]

        assert post_data["messaging_product"] == "whatsapp"
        assert post_data["recipient_type"] == "individual"
        assert post_data["to"] == sample_origin.channel_id
        assert post_data["type"] == "text"
        assert "text" in post_data
        assert "body" in post_data["text"]
        assert post_data["text"]["preview_url"] is False

    @pytest.mark.asyncio
    async def test_returns_false_without_access_token(self, sample_origin, sample_result):
        """_send_whatsapp_result returns False when WHATSAPP_ACCESS_TOKEN not set."""
        with patch.dict(os.environ, {"WHATSAPP_PHONE_NUMBER_ID": "123456"}, clear=True):
            os.environ.pop("WHATSAPP_ACCESS_TOKEN", None)
            result = await _send_whatsapp_result(sample_origin, sample_result)

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_without_phone_number_id(self, sample_origin, sample_result):
        """_send_whatsapp_result returns False when WHATSAPP_PHONE_NUMBER_ID not set."""
        with patch.dict(os.environ, {"WHATSAPP_ACCESS_TOKEN": "token"}, clear=True):
            os.environ.pop("WHATSAPP_PHONE_NUMBER_ID", None)
            result = await _send_whatsapp_result(sample_origin, sample_result)

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_without_credentials(self, sample_origin, sample_result):
        """_send_whatsapp_result returns False when both credentials missing."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("WHATSAPP_ACCESS_TOKEN", None)
            os.environ.pop("WHATSAPP_PHONE_NUMBER_ID", None)
            result = await _send_whatsapp_result(sample_origin, sample_result)

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_on_api_error(self, sample_origin, sample_result, whatsapp_env):
        """_send_whatsapp_result returns False on non-success HTTP status."""
        mock_response = MagicMock()
        mock_response.is_success = False
        mock_response.status_code = 403

        with patch.dict(os.environ, whatsapp_env):
            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_client.post = AsyncMock(return_value=mock_response)
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client_class.return_value = mock_client

                result = await _send_whatsapp_result(sample_origin, sample_result)

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_on_connection_error(
        self, sample_origin, sample_result, whatsapp_env
    ):
        """_send_whatsapp_result returns False on connection errors."""
        with patch.dict(os.environ, whatsapp_env):
            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_client.post = AsyncMock(side_effect=OSError("Connection refused"))
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client_class.return_value = mock_client

                result = await _send_whatsapp_result(sample_origin, sample_result)

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_on_timeout(self, sample_origin, sample_result, whatsapp_env):
        """_send_whatsapp_result returns False on timeout errors."""
        with patch.dict(os.environ, whatsapp_env):
            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_client.post = AsyncMock(side_effect=TimeoutError("Request timed out"))
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client_class.return_value = mock_client

                result = await _send_whatsapp_result(sample_origin, sample_result)

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_on_value_error(self, sample_origin, sample_result, whatsapp_env):
        """_send_whatsapp_result returns False on value errors."""
        with patch.dict(os.environ, whatsapp_env):
            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_client.post = AsyncMock(side_effect=ValueError("Invalid data"))
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client_class.return_value = mock_client

                result = await _send_whatsapp_result(sample_origin, sample_result)

        assert result is False

    @pytest.mark.asyncio
    async def test_uses_graph_api_v18(self, sample_origin, sample_result, whatsapp_env):
        """_send_whatsapp_result uses Graph API v18.0."""
        mock_response = MagicMock()
        mock_response.is_success = True

        with patch.dict(os.environ, whatsapp_env):
            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client = AsyncMock()
                mock_client.post = AsyncMock(return_value=mock_response)
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=None)
                mock_client_class.return_value = mock_client

                await _send_whatsapp_result(sample_origin, sample_result)

        call_args = mock_client.post.call_args
        assert "v18.0" in call_args[0][0]


# =============================================================================
# Test: Send WhatsApp Voice
# =============================================================================


class TestSendWhatsAppVoice:
    """Tests for _send_whatsapp_voice function."""

    @pytest.mark.asyncio
    async def test_uploads_media_and_sends_voice(self, sample_origin, sample_result, whatsapp_env):
        """_send_whatsapp_voice uploads media then sends audio message."""
        mock_upload_response = MagicMock()
        mock_upload_response.is_success = True
        mock_upload_response.json.return_value = {"id": "media-id-12345"}

        mock_send_response = MagicMock()
        mock_send_response.is_success = True

        with patch(
            "aragora.server.debate_origin.senders.whatsapp._synthesize_voice",
            new_callable=AsyncMock,
            return_value="/tmp/test_audio.ogg",
        ):
            with patch.dict(os.environ, whatsapp_env):
                with patch("httpx.AsyncClient") as mock_client_class:
                    mock_client = AsyncMock()
                    # First call is upload, second is send
                    mock_client.post = AsyncMock(
                        side_effect=[mock_upload_response, mock_send_response]
                    )
                    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                    mock_client.__aexit__ = AsyncMock(return_value=None)
                    mock_client_class.return_value = mock_client

                    with patch("builtins.open", mock_open(read_data=b"audio data")):
                        with patch.object(Path, "unlink"):
                            result = await _send_whatsapp_voice(sample_origin, sample_result)

        assert result is True
        assert mock_client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_upload_uses_media_endpoint(self, sample_origin, sample_result, whatsapp_env):
        """_send_whatsapp_voice uploads to media endpoint first."""
        mock_upload_response = MagicMock()
        mock_upload_response.is_success = True
        mock_upload_response.json.return_value = {"id": "media-123"}

        mock_send_response = MagicMock()
        mock_send_response.is_success = True

        with patch(
            "aragora.server.debate_origin.senders.whatsapp._synthesize_voice",
            new_callable=AsyncMock,
            return_value="/tmp/audio.ogg",
        ):
            with patch.dict(os.environ, whatsapp_env):
                with patch("httpx.AsyncClient") as mock_client_class:
                    mock_client = AsyncMock()
                    mock_client.post = AsyncMock(
                        side_effect=[mock_upload_response, mock_send_response]
                    )
                    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                    mock_client.__aexit__ = AsyncMock(return_value=None)
                    mock_client_class.return_value = mock_client

                    with patch("builtins.open", mock_open(read_data=b"audio")):
                        with patch.object(Path, "unlink"):
                            await _send_whatsapp_voice(sample_origin, sample_result)

        # First call should be to media endpoint
        first_call = mock_client.post.call_args_list[0]
        assert "/media" in first_call[0][0]

    @pytest.mark.asyncio
    async def test_send_uses_messages_endpoint_with_media_id(
        self, sample_origin, sample_result, whatsapp_env
    ):
        """_send_whatsapp_voice sends audio message with uploaded media ID."""
        mock_upload_response = MagicMock()
        mock_upload_response.is_success = True
        mock_upload_response.json.return_value = {"id": "uploaded-media-id"}

        mock_send_response = MagicMock()
        mock_send_response.is_success = True

        with patch(
            "aragora.server.debate_origin.senders.whatsapp._synthesize_voice",
            new_callable=AsyncMock,
            return_value="/tmp/audio.ogg",
        ):
            with patch.dict(os.environ, whatsapp_env):
                with patch("httpx.AsyncClient") as mock_client_class:
                    mock_client = AsyncMock()
                    mock_client.post = AsyncMock(
                        side_effect=[mock_upload_response, mock_send_response]
                    )
                    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                    mock_client.__aexit__ = AsyncMock(return_value=None)
                    mock_client_class.return_value = mock_client

                    with patch("builtins.open", mock_open(read_data=b"audio")):
                        with patch.object(Path, "unlink"):
                            await _send_whatsapp_voice(sample_origin, sample_result)

        # Second call should be to messages endpoint with media ID
        second_call = mock_client.post.call_args_list[1]
        assert "/messages" in second_call[0][0]
        send_data = second_call[1]["json"]
        assert send_data["type"] == "audio"
        assert send_data["audio"]["id"] == "uploaded-media-id"

    @pytest.mark.asyncio
    async def test_returns_false_when_synthesis_fails(self, sample_origin, sample_result):
        """_send_whatsapp_voice returns False when TTS synthesis fails."""
        with patch(
            "aragora.server.debate_origin.senders.whatsapp._synthesize_voice",
            new_callable=AsyncMock,
            return_value=None,
        ):
            result = await _send_whatsapp_voice(sample_origin, sample_result)

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_without_credentials(self, sample_origin, sample_result):
        """_send_whatsapp_voice returns False when credentials missing."""
        with patch(
            "aragora.server.debate_origin.senders.whatsapp._synthesize_voice",
            new_callable=AsyncMock,
            return_value="/tmp/audio.ogg",
        ):
            with patch.dict(os.environ, {}, clear=True):
                os.environ.pop("WHATSAPP_ACCESS_TOKEN", None)
                os.environ.pop("WHATSAPP_PHONE_NUMBER_ID", None)
                result = await _send_whatsapp_voice(sample_origin, sample_result)

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_when_upload_fails(
        self, sample_origin, sample_result, whatsapp_env
    ):
        """_send_whatsapp_voice returns False when media upload fails."""
        mock_upload_response = MagicMock()
        mock_upload_response.is_success = False
        mock_upload_response.status_code = 400

        with patch(
            "aragora.server.debate_origin.senders.whatsapp._synthesize_voice",
            new_callable=AsyncMock,
            return_value="/tmp/audio.ogg",
        ):
            with patch.dict(os.environ, whatsapp_env):
                with patch("httpx.AsyncClient") as mock_client_class:
                    mock_client = AsyncMock()
                    mock_client.post = AsyncMock(return_value=mock_upload_response)
                    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                    mock_client.__aexit__ = AsyncMock(return_value=None)
                    mock_client_class.return_value = mock_client

                    with patch("builtins.open", mock_open(read_data=b"audio")):
                        with patch.object(Path, "unlink"):
                            result = await _send_whatsapp_voice(sample_origin, sample_result)

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_when_upload_returns_no_id(
        self, sample_origin, sample_result, whatsapp_env
    ):
        """_send_whatsapp_voice returns False when upload returns no media ID."""
        mock_upload_response = MagicMock()
        mock_upload_response.is_success = True
        mock_upload_response.json.return_value = {}  # No "id" field

        with patch(
            "aragora.server.debate_origin.senders.whatsapp._synthesize_voice",
            new_callable=AsyncMock,
            return_value="/tmp/audio.ogg",
        ):
            with patch.dict(os.environ, whatsapp_env):
                with patch("httpx.AsyncClient") as mock_client_class:
                    mock_client = AsyncMock()
                    mock_client.post = AsyncMock(return_value=mock_upload_response)
                    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                    mock_client.__aexit__ = AsyncMock(return_value=None)
                    mock_client_class.return_value = mock_client

                    with patch("builtins.open", mock_open(read_data=b"audio")):
                        with patch.object(Path, "unlink"):
                            result = await _send_whatsapp_voice(sample_origin, sample_result)

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_when_send_fails(self, sample_origin, sample_result, whatsapp_env):
        """_send_whatsapp_voice returns False when audio send fails."""
        mock_upload_response = MagicMock()
        mock_upload_response.is_success = True
        mock_upload_response.json.return_value = {"id": "media-123"}

        mock_send_response = MagicMock()
        mock_send_response.is_success = False
        mock_send_response.status_code = 500

        with patch(
            "aragora.server.debate_origin.senders.whatsapp._synthesize_voice",
            new_callable=AsyncMock,
            return_value="/tmp/audio.ogg",
        ):
            with patch.dict(os.environ, whatsapp_env):
                with patch("httpx.AsyncClient") as mock_client_class:
                    mock_client = AsyncMock()
                    mock_client.post = AsyncMock(
                        side_effect=[mock_upload_response, mock_send_response]
                    )
                    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                    mock_client.__aexit__ = AsyncMock(return_value=None)
                    mock_client_class.return_value = mock_client

                    with patch("builtins.open", mock_open(read_data=b"audio")):
                        with patch.object(Path, "unlink"):
                            result = await _send_whatsapp_voice(sample_origin, sample_result)

        assert result is False

    @pytest.mark.asyncio
    async def test_cleans_up_temp_file(self, sample_origin, sample_result, whatsapp_env):
        """_send_whatsapp_voice cleans up temp audio file after sending."""
        mock_upload_response = MagicMock()
        mock_upload_response.is_success = True
        mock_upload_response.json.return_value = {"id": "media-123"}

        mock_send_response = MagicMock()
        mock_send_response.is_success = True

        with patch(
            "aragora.server.debate_origin.senders.whatsapp._synthesize_voice",
            new_callable=AsyncMock,
            return_value="/tmp/cleanup_test.ogg",
        ):
            with patch.dict(os.environ, whatsapp_env):
                with patch("httpx.AsyncClient") as mock_client_class:
                    mock_client = AsyncMock()
                    mock_client.post = AsyncMock(
                        side_effect=[mock_upload_response, mock_send_response]
                    )
                    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                    mock_client.__aexit__ = AsyncMock(return_value=None)
                    mock_client_class.return_value = mock_client

                    with patch("builtins.open", mock_open(read_data=b"audio")):
                        with patch.object(Path, "unlink") as mock_unlink:
                            await _send_whatsapp_voice(sample_origin, sample_result)

        mock_unlink.assert_called_once_with(missing_ok=True)

    @pytest.mark.asyncio
    async def test_cleans_up_on_failure(self, sample_origin, sample_result, whatsapp_env):
        """_send_whatsapp_voice cleans up temp file even on failure."""
        mock_upload_response = MagicMock()
        mock_upload_response.is_success = False
        mock_upload_response.status_code = 500

        with patch(
            "aragora.server.debate_origin.senders.whatsapp._synthesize_voice",
            new_callable=AsyncMock,
            return_value="/tmp/failure_cleanup.ogg",
        ):
            with patch.dict(os.environ, whatsapp_env):
                with patch("httpx.AsyncClient") as mock_client_class:
                    mock_client = AsyncMock()
                    mock_client.post = AsyncMock(return_value=mock_upload_response)
                    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                    mock_client.__aexit__ = AsyncMock(return_value=None)
                    mock_client_class.return_value = mock_client

                    with patch("builtins.open", mock_open(read_data=b"audio")):
                        with patch.object(Path, "unlink") as mock_unlink:
                            await _send_whatsapp_voice(sample_origin, sample_result)

        mock_unlink.assert_called_once_with(missing_ok=True)

    @pytest.mark.asyncio
    async def test_handles_cleanup_error_gracefully(
        self, sample_origin, sample_result, whatsapp_env
    ):
        """_send_whatsapp_voice handles cleanup errors without raising."""
        mock_upload_response = MagicMock()
        mock_upload_response.is_success = True
        mock_upload_response.json.return_value = {"id": "media-123"}

        mock_send_response = MagicMock()
        mock_send_response.is_success = True

        with patch(
            "aragora.server.debate_origin.senders.whatsapp._synthesize_voice",
            new_callable=AsyncMock,
            return_value="/tmp/cleanup_error.ogg",
        ):
            with patch.dict(os.environ, whatsapp_env):
                with patch("httpx.AsyncClient") as mock_client_class:
                    mock_client = AsyncMock()
                    mock_client.post = AsyncMock(
                        side_effect=[mock_upload_response, mock_send_response]
                    )
                    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                    mock_client.__aexit__ = AsyncMock(return_value=None)
                    mock_client_class.return_value = mock_client

                    with patch("builtins.open", mock_open(read_data=b"audio")):
                        with patch.object(Path, "unlink", side_effect=OSError("Permission denied")):
                            # Should not raise
                            result = await _send_whatsapp_voice(sample_origin, sample_result)

        assert result is True

    @pytest.mark.asyncio
    async def test_uses_longer_timeout_for_voice(self, sample_origin, sample_result, whatsapp_env):
        """_send_whatsapp_voice uses 60s timeout for media operations."""
        mock_upload_response = MagicMock()
        mock_upload_response.is_success = True
        mock_upload_response.json.return_value = {"id": "media-123"}

        mock_send_response = MagicMock()
        mock_send_response.is_success = True

        with patch(
            "aragora.server.debate_origin.senders.whatsapp._synthesize_voice",
            new_callable=AsyncMock,
            return_value="/tmp/audio.ogg",
        ):
            with patch.dict(os.environ, whatsapp_env):
                with patch("httpx.AsyncClient") as mock_client_class:
                    mock_client = AsyncMock()
                    mock_client.post = AsyncMock(
                        side_effect=[mock_upload_response, mock_send_response]
                    )
                    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                    mock_client.__aexit__ = AsyncMock(return_value=None)
                    mock_client_class.return_value = mock_client

                    with patch("builtins.open", mock_open(read_data=b"audio")):
                        with patch.object(Path, "unlink"):
                            await _send_whatsapp_voice(sample_origin, sample_result)

        # Verify timeout was set to 60s
        mock_client_class.assert_called_once_with(timeout=60.0)

    @pytest.mark.asyncio
    async def test_handles_connection_error(self, sample_origin, sample_result, whatsapp_env):
        """_send_whatsapp_voice handles connection errors gracefully."""
        with patch(
            "aragora.server.debate_origin.senders.whatsapp._synthesize_voice",
            new_callable=AsyncMock,
            return_value="/tmp/audio.ogg",
        ):
            with patch.dict(os.environ, whatsapp_env):
                with patch("httpx.AsyncClient") as mock_client_class:
                    mock_client = AsyncMock()
                    mock_client.post = AsyncMock(side_effect=ConnectionError("Connection failed"))
                    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                    mock_client.__aexit__ = AsyncMock(return_value=None)
                    mock_client_class.return_value = mock_client

                    with patch("builtins.open", mock_open(read_data=b"audio")):
                        with patch.object(Path, "unlink"):
                            result = await _send_whatsapp_voice(sample_origin, sample_result)

        assert result is False
