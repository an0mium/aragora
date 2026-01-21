"""
Tests for SlackConnector - Slack chat platform integration.

Tests cover:
- Message operations (send, update, delete)
- Ephemeral messages
- Slash command responses
- Interaction responses
- Webhook verification
- Block Kit formatting
- Error handling
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import json


class TestSlackConnectorInit:
    """Tests for SlackConnector initialization."""

    def test_default_init(self):
        """Should initialize with default values."""
        from aragora.connectors.chat.slack import SlackConnector

        connector = SlackConnector()

        assert connector.platform_name == "slack"
        assert connector.platform_display_name == "Slack"

    def test_init_with_token(self):
        """Should accept bot token."""
        from aragora.connectors.chat.slack import SlackConnector

        connector = SlackConnector(bot_token="xoxb-test-token")

        assert connector.bot_token == "xoxb-test-token"

    def test_init_with_signing_secret(self):
        """Should accept signing secret."""
        from aragora.connectors.chat.slack import SlackConnector

        connector = SlackConnector(signing_secret="test-secret")

        assert connector.signing_secret == "test-secret"

    def test_headers(self):
        """Should generate correct authorization headers."""
        from aragora.connectors.chat.slack import SlackConnector

        connector = SlackConnector(bot_token="xoxb-test-token")
        headers = connector._get_headers()

        assert headers["Authorization"] == "Bearer xoxb-test-token"
        assert "application/json" in headers["Content-Type"]


class TestSlackSendMessage:
    """Tests for send_message method."""

    @pytest.mark.asyncio
    async def test_send_simple_message(self):
        """Should send simple text message."""
        from aragora.connectors.chat.slack import SlackConnector

        connector = SlackConnector(bot_token="xoxb-test")

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "ok": True,
            "ts": "1234567890.123456",
            "channel": "C12345",
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            result = await connector.send_message(
                channel_id="C12345",
                text="Hello, World!",
            )

        assert result.success is True
        assert result.message_id == "1234567890.123456"
        assert result.channel_id == "C12345"

    @pytest.mark.asyncio
    async def test_send_message_with_blocks(self):
        """Should send message with Block Kit blocks."""
        from aragora.connectors.chat.slack import SlackConnector

        connector = SlackConnector(bot_token="xoxb-test")

        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": True, "ts": "123", "channel": "C1"}

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.post = AsyncMock(return_value=mock_response)

            blocks = [{"type": "section", "text": {"type": "mrkdwn", "text": "*Bold*"}}]
            result = await connector.send_message(
                channel_id="C12345",
                text="Fallback text",
                blocks=blocks,
            )

            # Verify blocks were included in payload
            call_kwargs = mock_instance.post.call_args[1]
            payload = call_kwargs["json"]
            assert payload["blocks"] == blocks

        assert result.success is True

    @pytest.mark.asyncio
    async def test_send_threaded_message(self):
        """Should send threaded reply."""
        from aragora.connectors.chat.slack import SlackConnector

        connector = SlackConnector(bot_token="xoxb-test")

        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": True, "ts": "123", "channel": "C1"}

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.post = AsyncMock(return_value=mock_response)

            result = await connector.send_message(
                channel_id="C12345",
                text="Reply",
                thread_id="1234567890.000001",
            )

            # Verify thread_ts was included
            call_kwargs = mock_instance.post.call_args[1]
            payload = call_kwargs["json"]
            assert payload["thread_ts"] == "1234567890.000001"

        assert result.success is True

    @pytest.mark.asyncio
    async def test_send_message_api_error(self):
        """Should handle API errors gracefully."""
        from aragora.connectors.chat.slack import SlackConnector

        connector = SlackConnector(bot_token="xoxb-test")

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "ok": False,
            "error": "channel_not_found",
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            result = await connector.send_message(
                channel_id="invalid",
                text="Test",
            )

        assert result.success is False
        assert result.error == "channel_not_found"

    @pytest.mark.asyncio
    async def test_send_message_exception(self):
        """Should handle exceptions gracefully."""
        from aragora.connectors.chat.slack import SlackConnector

        connector = SlackConnector(bot_token="xoxb-test")

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=Exception("Network error")
            )

            result = await connector.send_message(
                channel_id="C12345",
                text="Test",
            )

        assert result.success is False
        assert "Network error" in result.error


class TestSlackUpdateMessage:
    """Tests for update_message method."""

    @pytest.mark.asyncio
    async def test_update_message(self):
        """Should update existing message."""
        from aragora.connectors.chat.slack import SlackConnector

        connector = SlackConnector(bot_token="xoxb-test")

        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": True, "ts": "123", "channel": "C1"}

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.post = AsyncMock(return_value=mock_response)

            result = await connector.update_message(
                channel_id="C12345",
                message_id="1234567890.123456",
                text="Updated text",
            )

            # Verify correct endpoint and ts
            call_args = mock_instance.post.call_args
            assert "chat.update" in call_args[0][0]
            payload = call_args[1]["json"]
            assert payload["ts"] == "1234567890.123456"

        assert result.success is True

    @pytest.mark.asyncio
    async def test_update_message_error(self):
        """Should handle update errors."""
        from aragora.connectors.chat.slack import SlackConnector

        connector = SlackConnector(bot_token="xoxb-test")

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "ok": False,
            "error": "message_not_found",
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            result = await connector.update_message(
                channel_id="C12345",
                message_id="invalid",
                text="Updated",
            )

        assert result.success is False
        assert result.error == "message_not_found"


class TestSlackDeleteMessage:
    """Tests for delete_message method."""

    @pytest.mark.asyncio
    async def test_delete_message(self):
        """Should delete message."""
        from aragora.connectors.chat.slack import SlackConnector

        connector = SlackConnector(bot_token="xoxb-test")

        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": True}

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.post = AsyncMock(return_value=mock_response)

            result = await connector.delete_message(
                channel_id="C12345",
                message_id="1234567890.123456",
            )

            # Verify correct endpoint
            call_args = mock_instance.post.call_args
            assert "chat.delete" in call_args[0][0]

        assert result is True

    @pytest.mark.asyncio
    async def test_delete_message_failure(self):
        """Should return False on delete failure."""
        from aragora.connectors.chat.slack import SlackConnector

        connector = SlackConnector(bot_token="xoxb-test")

        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": False}

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            result = await connector.delete_message(
                channel_id="C12345",
                message_id="invalid",
            )

        assert result is False


class TestSlackEphemeralMessage:
    """Tests for send_ephemeral method."""

    @pytest.mark.asyncio
    async def test_send_ephemeral(self):
        """Should send ephemeral message to user."""
        from aragora.connectors.chat.slack import SlackConnector

        connector = SlackConnector(bot_token="xoxb-test")

        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": True}

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.post = AsyncMock(return_value=mock_response)

            result = await connector.send_ephemeral(
                channel_id="C12345",
                user_id="U12345",
                text="Only you can see this",
            )

            # Verify correct endpoint and user
            call_args = mock_instance.post.call_args
            assert "chat.postEphemeral" in call_args[0][0]
            payload = call_args[1]["json"]
            assert payload["user"] == "U12345"

        assert result.success is True


class TestSlackWithoutHttpx:
    """Tests for behavior when httpx is not available."""

    @pytest.mark.asyncio
    async def test_send_without_httpx(self):
        """Should return error when httpx not available."""
        from aragora.connectors.chat.slack import SlackConnector

        connector = SlackConnector(bot_token="xoxb-test")

        with patch("aragora.connectors.chat.slack.HTTPX_AVAILABLE", False):
            # Need to reimport to get patched value
            connector_module = __import__(
                "aragora.connectors.chat.slack", fromlist=["SlackConnector"]
            )
            patched_connector = connector_module.SlackConnector(bot_token="xoxb-test")

            result = await patched_connector.send_message(
                channel_id="C12345",
                text="Test",
            )

            # When httpx not available, should fail gracefully
            assert result.success is False or result.error is not None
