"""
Tests for DiscordConnector - Discord chat platform integration.

Tests cover:
- Message operations (send, update, delete)
- Embed formatting
- Slash command interactions
- Button components
- Webhook verification
- Error handling
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import json


class TestDiscordConnectorInit:
    """Tests for DiscordConnector initialization."""

    def test_default_init(self):
        """Should initialize with default values."""
        from aragora.connectors.chat.discord import DiscordConnector

        connector = DiscordConnector()

        assert connector.platform_name == "discord"
        assert connector.platform_display_name == "Discord"

    def test_init_with_token(self):
        """Should accept bot token."""
        from aragora.connectors.chat.discord import DiscordConnector

        connector = DiscordConnector(bot_token="test-bot-token")

        assert connector.bot_token == "test-bot-token"

    def test_init_with_application_id(self):
        """Should accept application ID."""
        from aragora.connectors.chat.discord import DiscordConnector

        connector = DiscordConnector(application_id="12345")

        assert connector.application_id == "12345"

    def test_headers(self):
        """Should generate correct authorization headers."""
        from aragora.connectors.chat.discord import DiscordConnector

        connector = DiscordConnector(bot_token="test-token")
        headers = connector._get_headers()

        assert headers["Authorization"] == "Bot test-token"
        assert headers["Content-Type"] == "application/json"


class TestDiscordSendMessage:
    """Tests for send_message method."""

    def _create_mock_client(self, mock_response):
        """Helper to create properly configured httpx mock."""
        mock_client_instance = MagicMock()
        mock_client_instance.request = AsyncMock(return_value=mock_response)
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=None)
        return mock_client_instance

    @pytest.mark.asyncio
    async def test_send_simple_message(self):
        """Should send simple text message."""
        from aragora.connectors.chat.discord import DiscordConnector

        connector = DiscordConnector(bot_token="test-token")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = ""
        mock_response.json.return_value = {
            "id": "123456789",
            "channel_id": "987654321",
        }
        mock_response.raise_for_status = MagicMock()

        mock_client_instance = self._create_mock_client(mock_response)

        with patch("httpx.AsyncClient", return_value=mock_client_instance):
            result = await connector.send_message(
                channel_id="987654321",
                text="Hello, Discord!",
            )

        assert result.success is True
        assert result.message_id == "123456789"

    @pytest.mark.asyncio
    async def test_send_message_with_embeds(self):
        """Should send message with embeds."""
        from aragora.connectors.chat.discord import DiscordConnector

        connector = DiscordConnector(bot_token="test-token")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": "123", "channel_id": "456"}
        mock_response.raise_for_status = MagicMock()

        mock_client_instance = self._create_mock_client(mock_response)

        with patch("httpx.AsyncClient", return_value=mock_client_instance):
            embeds = [
                {
                    "title": "Debate Result",
                    "description": "The conclusion",
                    "color": 0x00FF00,
                }
            ]
            result = await connector.send_message(
                channel_id="456",
                text="Fallback",
                blocks=embeds,
            )

            # Verify request was called with correct method
            call_kwargs = mock_client_instance.request.call_args[1]
            assert call_kwargs["method"] == "POST"
            payload = call_kwargs["json"]
            assert payload["embeds"] == embeds

        assert result.success is True

    @pytest.mark.asyncio
    async def test_send_message_with_components(self):
        """Should send message with button components."""
        from aragora.connectors.chat.discord import DiscordConnector

        connector = DiscordConnector(bot_token="test-token")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": "123", "channel_id": "456"}
        mock_response.raise_for_status = MagicMock()

        mock_client_instance = self._create_mock_client(mock_response)

        with patch("httpx.AsyncClient", return_value=mock_client_instance):
            components = [
                {
                    "type": 1,  # Action row
                    "components": [
                        {
                            "type": 2,  # Button
                            "style": 1,
                            "label": "Vote Yes",
                            "custom_id": "vote_yes",
                        }
                    ],
                }
            ]

            result = await connector.send_message(
                channel_id="456",
                text="Vote:",
                components=components,
            )

            call_kwargs = mock_client_instance.request.call_args[1]
            payload = call_kwargs["json"]
            assert payload["components"] == components

        assert result.success is True

    @pytest.mark.asyncio
    async def test_send_message_error(self):
        """Should handle API errors gracefully."""
        from aragora.connectors.chat.discord import DiscordConnector

        connector = DiscordConnector(bot_token="test-token")

        mock_client_instance = MagicMock()
        mock_client_instance.request = AsyncMock(side_effect=Exception("Rate limited"))
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_client_instance):
            result = await connector.send_message(
                channel_id="invalid",
                text="Test",
            )

        assert result.success is False
        assert "Rate limited" in result.error


class TestDiscordUpdateMessage:
    """Tests for update_message method."""

    @pytest.mark.asyncio
    async def test_update_message(self):
        """Should update existing message."""
        from aragora.connectors.chat.discord import DiscordConnector

        connector = DiscordConnector(bot_token="test-token")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": "123", "channel_id": "456"}
        mock_response.raise_for_status = MagicMock()

        mock_client_instance = MagicMock()
        mock_client_instance.request = AsyncMock(return_value=mock_response)
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_client_instance):
            result = await connector.update_message(
                channel_id="456",
                message_id="123",
                text="Updated text",
            )

            # Verify PATCH to correct endpoint
            call_kwargs = mock_client_instance.request.call_args[1]
            assert call_kwargs["method"] == "PATCH"
            assert "/channels/456/messages/123" in call_kwargs["url"]

        assert result.success is True


class TestDiscordDeleteMessage:
    """Tests for delete_message method."""

    @pytest.mark.asyncio
    async def test_delete_message(self):
        """Should delete message."""
        from aragora.connectors.chat.discord import DiscordConnector

        connector = DiscordConnector(bot_token="test-token")

        mock_response = MagicMock()
        mock_response.status_code = 204
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.delete = AsyncMock(return_value=mock_response)

            result = await connector.delete_message(
                channel_id="456",
                message_id="123",
            )

            # Verify DELETE was called
            mock_instance.delete.assert_called_once()
            call_url = mock_instance.delete.call_args[0][0]
            assert "/channels/456/messages/123" in call_url

        assert result is True

    @pytest.mark.asyncio
    async def test_delete_message_failure(self):
        """Should return False on delete failure."""
        from aragora.connectors.chat.discord import DiscordConnector

        connector = DiscordConnector(bot_token="test-token")

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.delete = AsyncMock(side_effect=Exception("Not found"))

            result = await connector.delete_message(
                channel_id="456",
                message_id="invalid",
            )

        assert result is False


class TestDiscordWithoutDependencies:
    """Tests for behavior when dependencies are not available."""

    def test_nacl_not_available_warning(self):
        """Should handle missing PyNaCl gracefully."""
        from aragora.connectors.chat.discord import DiscordConnector

        # Just verify connector can be created without PyNaCl
        connector = DiscordConnector(public_key="")
        assert connector.platform_name == "discord"

    @pytest.mark.asyncio
    async def test_send_without_httpx(self):
        """Should return error when httpx not available."""
        from aragora.connectors.chat.discord import DiscordConnector

        connector = DiscordConnector(bot_token="test-token")

        with patch("aragora.connectors.chat.discord.HTTPX_AVAILABLE", False):
            connector_module = __import__(
                "aragora.connectors.chat.discord", fromlist=["DiscordConnector"]
            )
            patched_connector = connector_module.DiscordConnector(bot_token="test-token")

            result = await patched_connector.send_message(
                channel_id="123",
                text="Test",
            )

            assert result.success is False or result.error is not None


class TestDiscordMetadataLookups:
    """Tests for Discord channel and user info lookups."""

    @pytest.mark.asyncio
    async def test_get_channel_info_text_channel(self):
        """Should retrieve text channel info via Discord API."""
        from aragora.connectors.chat.discord import DiscordConnector

        connector = DiscordConnector(bot_token="test-token")

        channel_response = {
            "id": "123456789",
            "type": 0,  # GUILD_TEXT
            "name": "general",
            "topic": "General discussion",
            "guild_id": "guild-123",
            "nsfw": False,
            "position": 1,
            "parent_id": "category-456",
        }

        with patch.object(connector, "_http_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = (True, channel_response, None)

            channel = await connector.get_channel_info(channel_id="123456789")

        assert channel is not None
        assert channel.id == "123456789"
        assert channel.name == "general"
        assert channel.team_id == "guild-123"
        assert channel.is_dm is False
        assert channel.is_private is False

    @pytest.mark.asyncio
    async def test_get_channel_info_dm_channel(self):
        """Should recognize DM channels."""
        from aragora.connectors.chat.discord import DiscordConnector

        connector = DiscordConnector(bot_token="test-token")

        dm_response = {
            "id": "dm-123",
            "type": 1,  # DM
            "name": None,
        }

        with patch.object(connector, "_http_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = (True, dm_response, None)

            channel = await connector.get_channel_info(channel_id="dm-123")

        assert channel is not None
        assert channel.is_dm is True

    @pytest.mark.asyncio
    async def test_get_channel_info_private_thread(self):
        """Should recognize private threads."""
        from aragora.connectors.chat.discord import DiscordConnector

        connector = DiscordConnector(bot_token="test-token")

        thread_response = {
            "id": "thread-123",
            "type": 12,  # PRIVATE_THREAD
            "name": "Private discussion",
            "guild_id": "guild-123",
        }

        with patch.object(connector, "_http_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = (True, thread_response, None)

            channel = await connector.get_channel_info(channel_id="thread-123")

        assert channel is not None
        assert channel.is_private is True

    @pytest.mark.asyncio
    async def test_get_channel_info_error(self):
        """Should return None on API error."""
        from aragora.connectors.chat.discord import DiscordConnector

        connector = DiscordConnector(bot_token="test-token")

        with patch.object(connector, "_http_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = (False, None, "Channel not found")

            channel = await connector.get_channel_info(channel_id="invalid")

        assert channel is None

    @pytest.mark.asyncio
    async def test_get_user_info(self):
        """Should retrieve user info via Discord API."""
        from aragora.connectors.chat.discord import DiscordConnector

        connector = DiscordConnector(bot_token="test-token")

        user_response = {
            "id": "user-123",
            "username": "johndoe",
            "global_name": "John Doe",
            "avatar": "abc123",
            "discriminator": "0",
            "bot": False,
            "accent_color": 16711680,
        }

        with patch.object(connector, "_http_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = (True, user_response, None)

            user = await connector.get_user_info(user_id="user-123")

        assert user is not None
        assert user.id == "user-123"
        assert user.username == "johndoe"
        assert user.display_name == "John Doe"
        assert user.is_bot is False
        assert "avatars/user-123/abc123.png" in user.avatar_url

    @pytest.mark.asyncio
    async def test_get_user_info_animated_avatar(self):
        """Should use .gif for animated avatars."""
        from aragora.connectors.chat.discord import DiscordConnector

        connector = DiscordConnector(bot_token="test-token")

        user_response = {
            "id": "user-456",
            "username": "animated_user",
            "global_name": "Animated User",
            "avatar": "a_xyz789",  # Starts with a_ = animated
            "bot": False,
        }

        with patch.object(connector, "_http_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = (True, user_response, None)

            user = await connector.get_user_info(user_id="user-456")

        assert user is not None
        assert user.avatar_url.endswith(".gif")

    @pytest.mark.asyncio
    async def test_get_user_info_no_avatar(self):
        """Should handle users without avatar."""
        from aragora.connectors.chat.discord import DiscordConnector

        connector = DiscordConnector(bot_token="test-token")

        user_response = {
            "id": "user-789",
            "username": "noavatar",
            "global_name": None,
            "avatar": None,
            "bot": False,
        }

        with patch.object(connector, "_http_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = (True, user_response, None)

            user = await connector.get_user_info(user_id="user-789")

        assert user is not None
        assert user.avatar_url is None
        assert user.display_name == "noavatar"  # Falls back to username

    @pytest.mark.asyncio
    async def test_get_user_info_bot(self):
        """Should identify bot users."""
        from aragora.connectors.chat.discord import DiscordConnector

        connector = DiscordConnector(bot_token="test-token")

        bot_response = {
            "id": "bot-123",
            "username": "mybot",
            "global_name": "My Bot",
            "avatar": "botavatar",
            "bot": True,
        }

        with patch.object(connector, "_http_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = (True, bot_response, None)

            user = await connector.get_user_info(user_id="bot-123")

        assert user is not None
        assert user.is_bot is True

    @pytest.mark.asyncio
    async def test_get_user_info_error(self):
        """Should return None on API error."""
        from aragora.connectors.chat.discord import DiscordConnector

        connector = DiscordConnector(bot_token="test-token")

        with patch.object(connector, "_http_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = (False, None, "User not found")

            user = await connector.get_user_info(user_id="invalid")

        assert user is None
