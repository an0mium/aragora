"""
Extended tests for DiscordConnector - comprehensive coverage of all methods.

Covers functionality NOT tested in test_discord_connector.py:
- Thread message routing
- Typing indicator
- Ephemeral messages
- Slash command responses (with/without interaction tokens)
- Interaction responses (button, select menu, modal)
- Interaction token response helper
- File upload and download
- Embed formatting (all combinations)
- Button formatting (all styles, URL buttons)
- Webhook verification
- Webhook event parsing (PING, slash command, button, select, modal, invalid JSON)
- Channel history retrieval (params, bot skipping, text response parsing)
- Evidence collection (query filtering, relevance, empty)
- Circuit breaker integration
- Health/connection/configuration checks
- HTTPX unavailable paths for all async methods
"""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_connector(**kwargs):
    """Create a DiscordConnector with sensible defaults for testing."""
    from aragora.connectors.chat.discord import DiscordConnector

    defaults = {
        "bot_token": "test-token",
        "application_id": "app-123",
        "public_key": "pk-abc",
    }
    defaults.update(kwargs)
    return DiscordConnector(**defaults)


def _mock_http_client(response):
    """Build a mock httpx.AsyncClient that returns *response* from .request()."""
    client = MagicMock()
    client.request = AsyncMock(return_value=response)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    return client


def _ok_response(json_body=None, status_code=200, text=""):
    """Build a mock httpx response."""
    r = MagicMock()
    r.status_code = status_code
    r.text = text
    r.json = MagicMock(return_value=json_body or {})
    r.raise_for_status = MagicMock()
    r.content = b""
    r.headers = {"content-type": "application/octet-stream"}
    return r


# ===========================================================================
# 1. send_message — thread routing
# ===========================================================================


class TestSendMessageThread:
    """Verify send_message routes to thread_id when provided."""

    @pytest.mark.asyncio
    async def test_send_message_to_thread(self):
        connector = _make_connector()
        resp = _ok_response(
            {"id": "msg-1", "channel_id": "ch-1", "timestamp": "2025-01-01T00:00:00Z"}
        )
        client = _mock_http_client(resp)

        with patch("httpx.AsyncClient", return_value=client):
            result = await connector.send_message(
                channel_id="ch-1",
                text="Hello thread",
                thread_id="thread-99",
            )

        assert result.success is True
        call_kwargs = client.request.call_args[1]
        assert "/channels/thread-99/messages" in call_kwargs["url"]


# ===========================================================================
# 2. update_message — embeds, components, error, httpx unavailable
# ===========================================================================


class TestUpdateMessageExtended:
    @pytest.mark.asyncio
    async def test_update_with_embeds(self):
        connector = _make_connector()
        resp = _ok_response({"id": "m1", "channel_id": "c1"})
        client = _mock_http_client(resp)

        with patch("httpx.AsyncClient", return_value=client):
            result = await connector.update_message(
                channel_id="c1",
                message_id="m1",
                text="updated",
                blocks=[{"title": "Update", "description": "New info"}],
            )
            payload = client.request.call_args[1]["json"]
            assert "embeds" in payload

        assert result.success is True

    @pytest.mark.asyncio
    async def test_update_with_components(self):
        connector = _make_connector()
        resp = _ok_response({"id": "m1", "channel_id": "c1"})
        client = _mock_http_client(resp)

        with patch("httpx.AsyncClient", return_value=client):
            result = await connector.update_message(
                channel_id="c1",
                message_id="m1",
                text="updated",
                components=[{"type": 1, "components": []}],
            )
            payload = client.request.call_args[1]["json"]
            assert payload["components"] == [{"type": 1, "components": []}]

        assert result.success is True

    @pytest.mark.asyncio
    async def test_update_message_error(self):
        connector = _make_connector()
        client = MagicMock()
        client.request = AsyncMock(side_effect=Exception("timeout"))
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=client):
            result = await connector.update_message("c1", "m1", "text")

        assert result.success is False

    @pytest.mark.asyncio
    async def test_update_message_httpx_unavailable(self):
        connector = _make_connector()
        with patch("aragora.connectors.chat.discord.HTTPX_AVAILABLE", False):
            result = await connector.update_message("c1", "m1", "text")
        assert result.success is False
        assert "httpx" in (result.error or "").lower()


# ===========================================================================
# 3. delete_message — httpx unavailable
# ===========================================================================


class TestDeleteMessageExtended:
    @pytest.mark.asyncio
    async def test_delete_httpx_unavailable(self):
        connector = _make_connector()
        with patch("aragora.connectors.chat.discord.HTTPX_AVAILABLE", False):
            result = await connector.delete_message("c1", "m1")
        assert result is False


# ===========================================================================
# 4. send_typing_indicator
# ===========================================================================


class TestSendTypingIndicator:
    @pytest.mark.asyncio
    async def test_typing_indicator_success(self):
        connector = _make_connector()
        with patch.object(connector, "_http_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = (True, {"status": "ok"}, None)
            result = await connector.send_typing_indicator(channel_id="c1")
        assert result is True

    @pytest.mark.asyncio
    async def test_typing_indicator_failure(self):
        connector = _make_connector()
        with patch.object(connector, "_http_request", new_callable=AsyncMock) as mock_req:
            mock_req.side_effect = OSError("connection refused")
            result = await connector.send_typing_indicator(channel_id="c1")
        assert result is False

    @pytest.mark.asyncio
    async def test_typing_indicator_httpx_unavailable(self):
        connector = _make_connector()
        with patch("aragora.connectors.chat.discord.HTTPX_AVAILABLE", False):
            result = await connector.send_typing_indicator(channel_id="c1")
        assert result is False


# ===========================================================================
# 5. send_ephemeral
# ===========================================================================


class TestSendEphemeral:
    @pytest.mark.asyncio
    async def test_ephemeral_delegates_to_send_message(self):
        connector = _make_connector()
        with patch.object(connector, "send_message", new_callable=AsyncMock) as mock_send:
            from aragora.connectors.chat.models import SendMessageResponse

            mock_send.return_value = SendMessageResponse(success=True, message_id="m1")
            result = await connector.send_ephemeral(
                channel_id="c1",
                user_id="u1",
                text="only you can see this",
            )
        assert result.success is True
        mock_send.assert_called_once()


# ===========================================================================
# 6. respond_to_command
# ===========================================================================


class TestRespondToCommand:
    @pytest.mark.asyncio
    async def test_respond_with_interaction_token(self):
        connector = _make_connector()
        from aragora.connectors.chat.models import BotCommand, ChatChannel, ChatUser

        cmd = BotCommand(
            name="debate",
            text="/debate",
            user=ChatUser(id="u1", platform="discord"),
            channel=ChatChannel(id="c1", platform="discord"),
            metadata={
                "interaction_id": "int-1",
                "interaction_token": "tok-abc",
            },
        )

        with patch.object(
            connector, "_respond_to_interaction_token", new_callable=AsyncMock
        ) as mock_respond:
            from aragora.connectors.chat.models import SendMessageResponse

            mock_respond.return_value = SendMessageResponse(success=True)
            result = await connector.respond_to_command(cmd, "Response text")

        assert result.success is True
        mock_respond.assert_called_once_with("int-1", "tok-abc", "Response text", None, True)

    @pytest.mark.asyncio
    async def test_respond_without_token_with_channel(self):
        connector = _make_connector()
        from aragora.connectors.chat.models import BotCommand, ChatChannel, ChatUser

        cmd = BotCommand(
            name="help",
            text="/help",
            user=ChatUser(id="u1", platform="discord"),
            channel=ChatChannel(id="c1", platform="discord"),
            metadata={},
        )

        with patch.object(connector, "send_message", new_callable=AsyncMock) as mock_send:
            from aragora.connectors.chat.models import SendMessageResponse

            mock_send.return_value = SendMessageResponse(success=True, message_id="m1")
            result = await connector.respond_to_command(cmd, "Help text")

        assert result.success is True
        mock_send.assert_called_once()

    @pytest.mark.asyncio
    async def test_respond_without_token_or_channel(self):
        connector = _make_connector()
        from aragora.connectors.chat.models import BotCommand

        cmd = BotCommand(
            name="help",
            text="/help",
            metadata={},
        )

        result = await connector.respond_to_command(cmd, "Help text")
        assert result.success is False
        assert "No interaction token or channel" in (result.error or "")


# ===========================================================================
# 7. respond_to_interaction
# ===========================================================================


class TestRespondToInteraction:
    @pytest.mark.asyncio
    async def test_respond_with_interaction_context(self):
        connector = _make_connector()
        from aragora.connectors.chat.models import (
            ChatChannel,
            ChatUser,
            InteractionType,
            SendMessageResponse,
            UserInteraction,
        )

        interaction = UserInteraction(
            id="int-1",
            interaction_type=InteractionType.BUTTON_CLICK,
            action_id="vote_yes",
            user=ChatUser(id="u1", platform="discord"),
            channel=ChatChannel(id="c1", platform="discord"),
            metadata={
                "interaction_id": "int-1",
                "interaction_token": "tok-abc",
            },
        )

        with patch.object(
            connector, "_respond_to_interaction_token", new_callable=AsyncMock
        ) as mock_respond:
            mock_respond.return_value = SendMessageResponse(success=True)
            result = await connector.respond_to_interaction(interaction, "Voted!")

        assert result.success is True
        # response_type=4 (CHANNEL_MESSAGE_WITH_SOURCE) when not replacing
        mock_respond.assert_called_once_with(
            "int-1", "tok-abc", "Voted!", None, ephemeral=False, response_type=4
        )

    @pytest.mark.asyncio
    async def test_respond_replace_original(self):
        connector = _make_connector()
        from aragora.connectors.chat.models import (
            ChatChannel,
            ChatUser,
            InteractionType,
            SendMessageResponse,
            UserInteraction,
        )

        interaction = UserInteraction(
            id="int-1",
            interaction_type=InteractionType.BUTTON_CLICK,
            action_id="vote_yes",
            user=ChatUser(id="u1", platform="discord"),
            channel=ChatChannel(id="c1", platform="discord"),
            metadata={
                "interaction_id": "int-1",
                "interaction_token": "tok-abc",
            },
        )

        with patch.object(
            connector, "_respond_to_interaction_token", new_callable=AsyncMock
        ) as mock_respond:
            mock_respond.return_value = SendMessageResponse(success=True)
            result = await connector.respond_to_interaction(
                interaction, "Updated!", replace_original=True
            )

        assert result.success is True
        # response_type=7 (UPDATE_MESSAGE)
        mock_respond.assert_called_once_with(
            "int-1", "tok-abc", "Updated!", None, ephemeral=False, response_type=7
        )

    @pytest.mark.asyncio
    async def test_respond_without_interaction_context_falls_back(self):
        connector = _make_connector()
        from aragora.connectors.chat.models import (
            ChatChannel,
            ChatUser,
            InteractionType,
            SendMessageResponse,
            UserInteraction,
        )

        interaction = UserInteraction(
            id="int-1",
            interaction_type=InteractionType.BUTTON_CLICK,
            action_id="vote_yes",
            user=ChatUser(id="u1", platform="discord"),
            channel=ChatChannel(id="c1", platform="discord"),
            metadata={},  # No interaction_id/token
        )

        with patch.object(connector, "send_message", new_callable=AsyncMock) as mock_send:
            mock_send.return_value = SendMessageResponse(success=True, message_id="m1")
            result = await connector.respond_to_interaction(interaction, "Fallback")

        assert result.success is True
        mock_send.assert_called_once()

    @pytest.mark.asyncio
    async def test_respond_without_context_or_channel(self):
        connector = _make_connector()
        from aragora.connectors.chat.models import InteractionType, UserInteraction

        interaction = UserInteraction(
            id="int-1",
            interaction_type=InteractionType.BUTTON_CLICK,
            action_id="vote_yes",
            metadata={},
        )

        result = await connector.respond_to_interaction(interaction, "text")
        assert result.success is False
        assert "No interaction context" in (result.error or "")


# ===========================================================================
# 8. _respond_to_interaction_token
# ===========================================================================


class TestRespondToInteractionToken:
    @pytest.mark.asyncio
    async def test_success(self):
        connector = _make_connector()
        with patch.object(connector, "_http_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = (True, {}, None)
            result = await connector._respond_to_interaction_token("int-1", "tok-abc", "Hello")
        assert result.success is True

    @pytest.mark.asyncio
    async def test_with_embeds_and_ephemeral(self):
        connector = _make_connector()
        with patch.object(connector, "_http_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = (True, {}, None)
            result = await connector._respond_to_interaction_token(
                "int-1",
                "tok-abc",
                "Secret",
                blocks=[{"title": "Embed"}],
                ephemeral=True,
            )
        assert result.success is True
        # Verify the payload constructed
        call_kwargs = mock_req.call_args[1]
        payload = call_kwargs["json"]
        assert payload["data"]["flags"] == 64  # EPHEMERAL
        assert payload["data"]["embeds"] == [{"title": "Embed"}]
        assert payload["type"] == 4  # default response type

    @pytest.mark.asyncio
    async def test_custom_response_type(self):
        connector = _make_connector()
        with patch.object(connector, "_http_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = (True, {}, None)
            result = await connector._respond_to_interaction_token(
                "int-1", "tok-abc", "Updating", response_type=7
            )
        assert result.success is True
        payload = mock_req.call_args[1]["json"]
        assert payload["type"] == 7

    @pytest.mark.asyncio
    async def test_failure(self):
        connector = _make_connector()
        with patch.object(connector, "_http_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = (False, None, "Server error")
            result = await connector._respond_to_interaction_token("int-1", "tok-abc", "text")
        assert result.success is False
        assert "Server error" in (result.error or "")

    @pytest.mark.asyncio
    async def test_httpx_unavailable(self):
        connector = _make_connector()
        with patch("aragora.connectors.chat.discord.HTTPX_AVAILABLE", False):
            result = await connector._respond_to_interaction_token("int-1", "tok-abc", "text")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_interaction_url_format(self):
        connector = _make_connector()
        with patch.object(connector, "_http_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = (True, {}, None)
            await connector._respond_to_interaction_token("int-1", "tok-abc", "text")
        url = mock_req.call_args[1]["url"]
        assert "/interactions/int-1/tok-abc/callback" in url


# ===========================================================================
# 9. upload_file
# ===========================================================================


class TestUploadFile:
    @pytest.mark.asyncio
    async def test_upload_success(self):
        connector = _make_connector()
        with patch.object(connector, "_http_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = (
                True,
                {
                    "id": "msg-1",
                    "attachments": [
                        {
                            "id": "att-1",
                            "filename": "report.pdf",
                            "content_type": "application/pdf",
                            "size": 1024,
                            "url": "https://cdn.discordapp.com/att-1/report.pdf",
                        }
                    ],
                },
                None,
            )
            result = await connector.upload_file(
                channel_id="c1",
                content=b"fake pdf",
                filename="report.pdf",
                content_type="application/pdf",
            )

        assert result.id == "att-1"
        assert result.filename == "report.pdf"
        assert result.url == "https://cdn.discordapp.com/att-1/report.pdf"

    @pytest.mark.asyncio
    async def test_upload_with_title(self):
        connector = _make_connector()
        with patch.object(connector, "_http_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = (
                True,
                {"id": "msg-1", "attachments": [{"id": "a1", "filename": "f.txt", "size": 5}]},
                None,
            )
            result = await connector.upload_file(
                channel_id="c1",
                content=b"hello",
                filename="f.txt",
                title="My File",
            )
        # Verify data param includes content (title)
        call_kwargs = mock_req.call_args[1]
        assert call_kwargs["data"] == {"content": "My File"}
        assert result.id == "a1"

    @pytest.mark.asyncio
    async def test_upload_to_thread(self):
        connector = _make_connector()
        with patch.object(connector, "_http_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = (
                True,
                {"id": "m1", "attachments": [{"id": "a1", "filename": "f.txt", "size": 5}]},
                None,
            )
            await connector.upload_file(
                channel_id="c1",
                content=b"data",
                filename="f.txt",
                thread_id="thread-42",
            )
        url = mock_req.call_args[1]["url"]
        assert "/channels/thread-42/messages" in url

    @pytest.mark.asyncio
    async def test_upload_no_attachments_in_response(self):
        connector = _make_connector()
        with patch.object(connector, "_http_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = (True, {"id": "m1", "attachments": []}, None)
            result = await connector.upload_file("c1", b"data", "f.txt")
        assert result.id == ""  # fallback

    @pytest.mark.asyncio
    async def test_upload_failure(self):
        connector = _make_connector()
        with patch.object(connector, "_http_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = (False, None, "Upload failed")
            result = await connector.upload_file("c1", b"data", "f.txt")
        assert result.id == ""

    @pytest.mark.asyncio
    async def test_upload_httpx_unavailable(self):
        connector = _make_connector()
        with patch("aragora.connectors.chat.discord.HTTPX_AVAILABLE", False):
            result = await connector.upload_file("c1", b"data", "f.txt")
        assert result.id == ""
        assert result.size == 4  # len(b"data")

    @pytest.mark.asyncio
    async def test_upload_exception(self):
        connector = _make_connector()
        with patch.object(connector, "_http_request", new_callable=AsyncMock) as mock_req:
            mock_req.side_effect = ValueError("bad value")
            result = await connector.upload_file("c1", b"data", "f.txt")
        assert result.id == ""


# ===========================================================================
# 10. download_file
# ===========================================================================


class TestDownloadFile:
    @pytest.mark.asyncio
    async def test_download_no_url(self):
        connector = _make_connector()
        result = await connector.download_file("file-1")
        assert result.id == "file-1"
        assert result.size == 0

    @pytest.mark.asyncio
    async def test_download_httpx_unavailable(self):
        connector = _make_connector()
        with patch("aragora.connectors.chat.discord.HTTPX_AVAILABLE", False):
            result = await connector.download_file("file-1", url="https://example.com/f.txt")
        assert result.size == 0

    @pytest.mark.asyncio
    async def test_download_http_request_fails(self):
        connector = _make_connector()
        with patch.object(connector, "_http_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = (False, None, "not found")
            result = await connector.download_file(
                "file-1", url="https://cdn.discordapp.com/file-1"
            )
        assert result.size == 0

    @pytest.mark.asyncio
    async def test_download_success(self):
        connector = _make_connector()

        # First _http_request returns success, then the direct httpx call succeeds
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"file-contents"
        mock_response.headers = {"content-type": "text/plain"}
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch.object(connector, "_http_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = (True, {"text": "ok"}, None)
            with patch.object(connector, "_check_circuit_breaker", return_value=(True, None)):
                with patch("httpx.AsyncClient", return_value=mock_client):
                    result = await connector.download_file(
                        "file-1",
                        url="https://cdn.discordapp.com/file-1",
                        filename="test.txt",
                    )

        assert result.size == 13  # len(b"file-contents")
        assert result.content == b"file-contents"
        assert result.filename == "test.txt"

    @pytest.mark.asyncio
    async def test_download_circuit_breaker_blocks(self):
        connector = _make_connector()
        with patch.object(connector, "_http_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = (True, {"text": "ok"}, None)
            with patch.object(connector, "_check_circuit_breaker", return_value=(False, "CB open")):
                result = await connector.download_file(
                    "file-1",
                    url="https://cdn.discordapp.com/file-1",
                )
        assert result.size == 0


# ===========================================================================
# 11. format_blocks
# ===========================================================================


class TestFormatBlocks:
    def test_title_only(self):
        connector = _make_connector()
        blocks = connector.format_blocks(title="Hello")
        assert len(blocks) == 1
        assert blocks[0]["title"] == "Hello"
        assert "description" not in blocks[0]

    def test_body_only(self):
        connector = _make_connector()
        blocks = connector.format_blocks(body="Some description")
        assert blocks[0]["description"] == "Some description"

    def test_fields(self):
        connector = _make_connector()
        blocks = connector.format_blocks(fields=[("Key", "Value"), ("A", "B")])
        assert len(blocks[0]["fields"]) == 2
        assert blocks[0]["fields"][0] == {"name": "Key", "value": "Value", "inline": True}

    def test_custom_color(self):
        connector = _make_connector()
        blocks = connector.format_blocks(title="T", color=0xFF0000)
        assert blocks[0]["color"] == 0xFF0000

    def test_default_color_green(self):
        connector = _make_connector()
        blocks = connector.format_blocks(title="T")
        assert blocks[0]["color"] == 0x00FF00

    def test_type_is_rich(self):
        connector = _make_connector()
        blocks = connector.format_blocks()
        assert blocks[0]["type"] == "rich"

    def test_all_fields_combined(self):
        connector = _make_connector()
        blocks = connector.format_blocks(
            title="Title",
            body="Body",
            fields=[("F1", "V1")],
            color=0x0000FF,
        )
        embed = blocks[0]
        assert embed["title"] == "Title"
        assert embed["description"] == "Body"
        assert len(embed["fields"]) == 1
        assert embed["color"] == 0x0000FF


# ===========================================================================
# 12. format_button
# ===========================================================================


class TestFormatButton:
    def test_default_style(self):
        connector = _make_connector()
        button = connector.format_button("Click Me", "btn_action")
        assert button["type"] == 2
        assert button["style"] == 2  # SECONDARY
        assert button["label"] == "Click Me"
        assert button["custom_id"] == "btn_action:"

    def test_primary_style(self):
        connector = _make_connector()
        button = connector.format_button("Go", "go_action", style="primary")
        assert button["style"] == 1

    def test_danger_style(self):
        connector = _make_connector()
        button = connector.format_button("Delete", "del_action", style="danger")
        assert button["style"] == 4

    def test_unknown_style_defaults_to_secondary(self):
        connector = _make_connector()
        button = connector.format_button("Other", "act", style="unknown")
        assert button["style"] == 2  # fallback to SECONDARY

    def test_with_value(self):
        connector = _make_connector()
        button = connector.format_button("Vote", "vote_action", value="yes")
        assert button["custom_id"] == "vote_action:yes"

    def test_url_button(self):
        connector = _make_connector()
        button = connector.format_button("Open Link", "link_action", url="https://example.com")
        assert button["style"] == 5  # LINK
        assert button["url"] == "https://example.com"
        assert "custom_id" not in button


# ===========================================================================
# 13. verify_webhook
# ===========================================================================


class TestVerifyWebhook:
    def test_verify_webhook_calls_ed25519_verifier(self):
        connector = _make_connector()

        mock_result = MagicMock()
        mock_result.verified = True
        mock_result.error = None

        with patch("aragora.connectors.chat.discord.Ed25519Verifier") as MockVerifier:
            # Patch at import location within verify_webhook
            pass

        # Use a different approach - patch the import inside the method
        with patch("aragora.connectors.chat.webhook_security.Ed25519Verifier") as MockVerifier:
            instance = MockVerifier.return_value
            instance.verify.return_value = mock_result
            result = connector.verify_webhook(
                headers={"X-Signature-Ed25519": "sig", "X-Signature-Timestamp": "ts"},
                body=b'{"type":1}',
            )

        assert result is True

    def test_verify_webhook_failure(self):
        connector = _make_connector()
        mock_result = MagicMock()
        mock_result.verified = False
        mock_result.error = "Invalid signature"

        with patch("aragora.connectors.chat.webhook_security.Ed25519Verifier") as MockVerifier:
            instance = MockVerifier.return_value
            instance.verify.return_value = mock_result
            result = connector.verify_webhook(
                headers={},
                body=b"bad",
            )

        assert result is False


# ===========================================================================
# 14. parse_webhook_event
# ===========================================================================


class TestParseWebhookEvent:
    def test_invalid_json(self):
        connector = _make_connector()
        event = connector.parse_webhook_event({}, b"not json")
        assert event.event_type == "error"

    def test_ping(self):
        connector = _make_connector()
        payload = {"type": 1}
        event = connector.parse_webhook_event({}, json.dumps(payload).encode())
        assert event.event_type == "ping"
        assert event.challenge == "PONG"

    def test_slash_command(self):
        connector = _make_connector()
        payload = {
            "type": 2,
            "id": "int-1",
            "token": "tok-abc",
            "channel_id": "c1",
            "guild_id": "g1",
            "member": {
                "user": {
                    "id": "u1",
                    "username": "testuser",
                    "global_name": "Test User",
                    "avatar": "av123",
                    "bot": False,
                }
            },
            "data": {
                "name": "debate",
                "options": [
                    {"name": "topic", "value": "AI safety"},
                    {"name": "rounds", "value": 3},
                ],
            },
        }
        event = connector.parse_webhook_event({}, json.dumps(payload).encode())

        assert event.event_type == "interaction_2"
        assert event.command is not None
        assert event.command.name == "debate"
        assert event.command.args == ["AI safety", 3]
        assert event.command.options == {"topic": "AI safety", "rounds": 3}
        assert event.command.user.id == "u1"
        assert event.command.channel.id == "c1"
        assert event.command.channel.team_id == "g1"
        assert event.metadata["interaction_id"] == "int-1"
        assert event.metadata["interaction_token"] == "tok-abc"

    def test_button_click(self):
        connector = _make_connector()
        payload = {
            "type": 3,
            "id": "int-2",
            "token": "tok-def",
            "channel_id": "c1",
            "guild_id": "g1",
            "member": {
                "user": {
                    "id": "u2",
                    "username": "clicker",
                    "bot": False,
                }
            },
            "data": {
                "custom_id": "vote:yes",
                "component_type": 2,  # BUTTON
            },
            "message": {"id": "msg-1"},
        }
        event = connector.parse_webhook_event({}, json.dumps(payload).encode())

        assert event.event_type == "interaction_3"
        assert event.interaction is not None
        assert event.interaction.action_id == "vote"
        assert event.interaction.value == "yes"
        assert event.interaction.message_id == "msg-1"
        from aragora.connectors.chat.models import InteractionType

        assert event.interaction.interaction_type == InteractionType.BUTTON_CLICK

    def test_select_menu(self):
        connector = _make_connector()
        payload = {
            "type": 3,
            "id": "int-3",
            "token": "tok-ghi",
            "channel_id": "c1",
            "member": {
                "user": {
                    "id": "u3",
                    "username": "selector",
                    "bot": False,
                }
            },
            "data": {
                "custom_id": "agent_select",
                "component_type": 3,  # SELECT_MENU
                "values": ["claude", "gpt4"],
            },
            "message": {"id": "msg-2"},
        }
        event = connector.parse_webhook_event({}, json.dumps(payload).encode())

        assert event.interaction is not None
        from aragora.connectors.chat.models import InteractionType

        assert event.interaction.interaction_type == InteractionType.SELECT_MENU
        assert event.interaction.values == ["claude", "gpt4"]
        assert event.interaction.action_id == "agent_select"
        # No ":" in custom_id means value should be None
        assert event.interaction.value is None

    def test_modal_submit(self):
        connector = _make_connector()
        payload = {
            "type": 5,
            "id": "int-4",
            "token": "tok-jkl",
            "channel_id": "c1",
            "user": {
                "id": "u4",
                "username": "modal_user",
                "bot": False,
            },
            "data": {
                "custom_id": "feedback_modal",
                "components": [{"type": 1, "components": [{"type": 4, "value": "Great tool!"}]}],
            },
        }
        event = connector.parse_webhook_event({}, json.dumps(payload).encode())

        assert event.interaction is not None
        from aragora.connectors.chat.models import InteractionType

        assert event.interaction.interaction_type == InteractionType.MODAL_SUBMIT
        assert event.interaction.action_id == "feedback_modal"
        assert "components" in event.interaction.metadata

    def test_user_from_top_level_user_field(self):
        """When member is absent, user should come from top-level user field."""
        connector = _make_connector()
        payload = {
            "type": 2,
            "id": "int-5",
            "token": "tok-mno",
            "channel_id": "c1",
            "user": {
                "id": "u5",
                "username": "dm_user",
                "global_name": "DM User",
                "bot": False,
            },
            "data": {"name": "ping", "options": []},
        }
        event = connector.parse_webhook_event({}, json.dumps(payload).encode())
        assert event.command.user.id == "u5"
        assert event.command.user.username == "dm_user"

    def test_avatar_url_constructed(self):
        connector = _make_connector()
        payload = {
            "type": 2,
            "id": "int-6",
            "token": "tok-pqr",
            "channel_id": "c1",
            "member": {
                "user": {
                    "id": "u6",
                    "username": "avatar_user",
                    "avatar": "hash123",
                    "bot": False,
                }
            },
            "data": {"name": "cmd", "options": []},
        }
        event = connector.parse_webhook_event({}, json.dumps(payload).encode())
        # The user in the event's command should have an avatar URL
        # Note: the avatar URL is set on the user in parse_webhook_event
        # but command.user is also set from the same user_data
        assert "avatars/u6/hash123.png" in (event.command.user.avatar_url or "")

    def test_no_avatar_url_when_no_avatar(self):
        connector = _make_connector()
        payload = {
            "type": 2,
            "id": "int-7",
            "token": "tok-stu",
            "channel_id": "c1",
            "member": {
                "user": {
                    "id": "u7",
                    "username": "noav",
                    "avatar": None,
                    "bot": False,
                }
            },
            "data": {"name": "cmd", "options": []},
        }
        event = connector.parse_webhook_event({}, json.dumps(payload).encode())
        assert event.command.user.avatar_url is None

    def test_unknown_interaction_type(self):
        """Unknown interaction types should still produce an event with metadata."""
        connector = _make_connector()
        payload = {
            "type": 99,
            "id": "int-99",
            "token": "tok-xyz",
            "channel_id": "c1",
            "member": {"user": {"id": "u1", "username": "u1", "bot": False}},
        }
        event = connector.parse_webhook_event({}, json.dumps(payload).encode())
        assert event.event_type == "interaction_99"
        # No command or interaction set for unknown types
        assert event.command is None
        assert event.interaction is None


# ===========================================================================
# 15. get_channel_history
# ===========================================================================


class TestGetChannelHistory:
    @pytest.mark.asyncio
    async def test_basic_history(self):
        connector = _make_connector()
        api_response = [
            {
                "id": "msg-1",
                "content": "Hello world",
                "author": {
                    "id": "u1",
                    "username": "alice",
                    "global_name": "Alice",
                    "avatar": "av1",
                    "bot": False,
                },
                "timestamp": "2025-01-15T12:00:00+00:00",
                "reactions": [],
                "attachments": [],
                "embeds": [],
            }
        ]
        with patch.object(connector, "_http_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = (True, api_response, None)
            messages = await connector.get_channel_history("c1", limit=10)

        assert len(messages) == 1
        assert messages[0].id == "msg-1"
        assert messages[0].content == "Hello world"
        assert messages[0].author.username == "alice"

    @pytest.mark.asyncio
    async def test_skip_bot_messages(self):
        connector = _make_connector()
        api_response = [
            {
                "id": "msg-1",
                "content": "Bot message",
                "author": {"id": "bot-1", "username": "mybot", "bot": True},
                "timestamp": "2025-01-15T12:00:00Z",
            },
            {
                "id": "msg-2",
                "content": "Human message",
                "author": {"id": "u1", "username": "alice", "bot": False},
                "timestamp": "2025-01-15T12:01:00Z",
            },
        ]
        with patch.object(connector, "_http_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = (True, api_response, None)
            messages = await connector.get_channel_history("c1", skip_bots=True)

        assert len(messages) == 1
        assert messages[0].id == "msg-2"

    @pytest.mark.asyncio
    async def test_include_bot_messages(self):
        connector = _make_connector()
        api_response = [
            {
                "id": "msg-1",
                "content": "Bot message",
                "author": {"id": "bot-1", "username": "mybot", "bot": True},
                "timestamp": "2025-01-15T12:00:00Z",
            },
        ]
        with patch.object(connector, "_http_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = (True, api_response, None)
            messages = await connector.get_channel_history("c1", skip_bots=False)

        assert len(messages) == 1

    @pytest.mark.asyncio
    async def test_history_with_oldest_and_latest_params(self):
        connector = _make_connector()
        with patch.object(connector, "_http_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = (True, [], None)
            await connector.get_channel_history(
                "c1", limit=50, oldest="msg-before", latest="msg-after"
            )
        url = mock_req.call_args[1]["url"]
        assert "limit=50" in url
        assert "after=msg-before" in url
        assert "before=msg-after" in url

    @pytest.mark.asyncio
    async def test_history_limit_capped_at_100(self):
        connector = _make_connector()
        with patch.object(connector, "_http_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = (True, [], None)
            await connector.get_channel_history("c1", limit=500)
        url = mock_req.call_args[1]["url"]
        assert "limit=100" in url

    @pytest.mark.asyncio
    async def test_history_httpx_unavailable(self):
        connector = _make_connector()
        with patch("aragora.connectors.chat.discord.HTTPX_AVAILABLE", False):
            messages = await connector.get_channel_history("c1")
        assert messages == []

    @pytest.mark.asyncio
    async def test_history_api_error(self):
        connector = _make_connector()
        with patch.object(connector, "_http_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = (False, None, "API error")
            messages = await connector.get_channel_history("c1")
        assert messages == []

    @pytest.mark.asyncio
    async def test_history_text_response_parsing(self):
        """When response is dict with 'text', parse text as JSON list."""
        connector = _make_connector()
        msg_list = [
            {
                "id": "msg-1",
                "content": "text",
                "author": {"id": "u1", "username": "a", "bot": False},
                "timestamp": "2025-01-15T12:00:00Z",
            }
        ]
        with patch.object(connector, "_http_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = (True, {"text": json.dumps(msg_list)}, None)
            messages = await connector.get_channel_history("c1")

        assert len(messages) == 1

    @pytest.mark.asyncio
    async def test_history_text_response_bad_json(self):
        connector = _make_connector()
        with patch.object(connector, "_http_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = (True, {"text": "not json"}, None)
            messages = await connector.get_channel_history("c1")
        assert messages == []

    @pytest.mark.asyncio
    async def test_history_unexpected_response_type(self):
        connector = _make_connector()
        with patch.object(connector, "_http_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = (True, "unexpected string", None)
            messages = await connector.get_channel_history("c1")
        assert messages == []

    @pytest.mark.asyncio
    async def test_invalid_timestamp_falls_back_to_now(self):
        connector = _make_connector()
        api_response = [
            {
                "id": "msg-1",
                "content": "test",
                "author": {"id": "u1", "username": "a", "bot": False},
                "timestamp": "not-a-timestamp",
            }
        ]
        with patch.object(connector, "_http_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = (True, api_response, None)
            messages = await connector.get_channel_history("c1")
        assert len(messages) == 1
        # Timestamp should be roughly now
        assert messages[0].timestamp is not None

    @pytest.mark.asyncio
    async def test_message_reference_parsed_as_thread_id(self):
        connector = _make_connector()
        api_response = [
            {
                "id": "msg-1",
                "content": "reply",
                "author": {"id": "u1", "username": "a", "bot": False},
                "timestamp": "2025-01-15T12:00:00Z",
                "message_reference": {"message_id": "original-msg"},
            }
        ]
        with patch.object(connector, "_http_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = (True, api_response, None)
            messages = await connector.get_channel_history("c1")
        assert messages[0].thread_id == "original-msg"

    @pytest.mark.asyncio
    async def test_history_avatar_url(self):
        connector = _make_connector()
        api_response = [
            {
                "id": "msg-1",
                "content": "hi",
                "author": {
                    "id": "u1",
                    "username": "a",
                    "avatar": "avhash",
                    "bot": False,
                },
                "timestamp": "2025-01-15T12:00:00Z",
            }
        ]
        with patch.object(connector, "_http_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = (True, api_response, None)
            messages = await connector.get_channel_history("c1")
        assert "avatars/u1/avhash.png" in messages[0].author.avatar_url

    @pytest.mark.asyncio
    async def test_history_no_avatar(self):
        connector = _make_connector()
        api_response = [
            {
                "id": "msg-1",
                "content": "hi",
                "author": {"id": "u1", "username": "a", "bot": False},
                "timestamp": "2025-01-15T12:00:00Z",
            }
        ]
        with patch.object(connector, "_http_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = (True, api_response, None)
            messages = await connector.get_channel_history("c1")
        assert messages[0].author.avatar_url is None


# ===========================================================================
# 16. collect_evidence
# ===========================================================================


class TestCollectEvidence:
    @pytest.mark.asyncio
    async def test_collect_basic(self):
        connector = _make_connector()
        from aragora.connectors.chat.models import ChatChannel, ChatMessage, ChatUser

        mock_messages = [
            ChatMessage(
                id="msg-1",
                platform="discord",
                channel=ChatChannel(id="c1", platform="discord"),
                author=ChatUser(id="u1", platform="discord", username="alice"),
                content="I think AI safety is important",
            ),
        ]
        with patch.object(connector, "get_channel_history", new_callable=AsyncMock) as mock_hist:
            mock_hist.return_value = mock_messages
            evidence = await connector.collect_evidence("c1")

        assert len(evidence) == 1
        assert evidence[0].content == "I think AI safety is important"

    @pytest.mark.asyncio
    async def test_collect_with_query_filter(self):
        connector = _make_connector()
        from aragora.connectors.chat.models import ChatChannel, ChatMessage, ChatUser

        mock_messages = [
            ChatMessage(
                id="msg-1",
                platform="discord",
                channel=ChatChannel(id="c1", platform="discord"),
                author=ChatUser(id="u1", platform="discord"),
                content="AI safety is crucial",
            ),
            ChatMessage(
                id="msg-2",
                platform="discord",
                channel=ChatChannel(id="c1", platform="discord"),
                author=ChatUser(id="u2", platform="discord"),
                content="The weather is nice today",
            ),
        ]
        with patch.object(connector, "get_channel_history", new_callable=AsyncMock) as mock_hist:
            mock_hist.return_value = mock_messages
            evidence = await connector.collect_evidence("c1", query="AI safety", min_relevance=0.5)

        # Only the first message should have high relevance
        assert len(evidence) >= 1
        assert evidence[0].content == "AI safety is crucial"

    @pytest.mark.asyncio
    async def test_collect_empty_channel(self):
        connector = _make_connector()
        with patch.object(connector, "get_channel_history", new_callable=AsyncMock) as mock_hist:
            mock_hist.return_value = []
            evidence = await connector.collect_evidence("c1")
        assert evidence == []

    @pytest.mark.asyncio
    async def test_collect_sorted_by_relevance(self):
        connector = _make_connector()
        from aragora.connectors.chat.models import ChatChannel, ChatMessage, ChatUser

        mock_messages = [
            ChatMessage(
                id="msg-1",
                platform="discord",
                channel=ChatChannel(id="c1", platform="discord"),
                author=ChatUser(id="u1", platform="discord"),
                content="partly about safety",
            ),
            ChatMessage(
                id="msg-2",
                platform="discord",
                channel=ChatChannel(id="c1", platform="discord"),
                author=ChatUser(id="u2", platform="discord"),
                content="AI safety alignment research safety",
            ),
        ]
        with patch.object(connector, "get_channel_history", new_callable=AsyncMock) as mock_hist:
            mock_hist.return_value = mock_messages
            evidence = await connector.collect_evidence("c1", query="safety")

        # Most relevant first
        assert len(evidence) == 2
        assert evidence[0].relevance_score >= evidence[1].relevance_score


# ===========================================================================
# 17. Circuit breaker integration
# ===========================================================================


class TestCircuitBreakerIntegration:
    @pytest.mark.asyncio
    async def test_send_message_circuit_breaker_open(self):
        connector = _make_connector()
        with patch.object(
            connector, "_check_circuit_breaker", return_value=(False, "CB open 10.0s")
        ):
            result = await connector.send_message("c1", "text")
        assert result.success is False
        assert "CB open" in (result.error or "")

    @pytest.mark.asyncio
    async def test_update_message_circuit_breaker_open(self):
        connector = _make_connector()
        with patch.object(
            connector, "_check_circuit_breaker", return_value=(False, "CB open 5.0s")
        ):
            result = await connector.update_message("c1", "m1", "text")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_get_channel_info_circuit_breaker_open(self):
        connector = _make_connector()
        with patch.object(connector, "_check_circuit_breaker", return_value=(False, "CB open")):
            result = await connector.get_channel_info("c1")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_user_info_circuit_breaker_open(self):
        connector = _make_connector()
        with patch.object(connector, "_check_circuit_breaker", return_value=(False, "CB open")):
            result = await connector.get_user_info("u1")
        assert result is None


# ===========================================================================
# 18. HTTPX unavailable for all paths
# ===========================================================================


class TestHttpxUnavailableAllPaths:
    @pytest.mark.asyncio
    async def test_get_channel_info_httpx_unavailable(self):
        connector = _make_connector()
        with patch("aragora.connectors.chat.discord.HTTPX_AVAILABLE", False):
            result = await connector.get_channel_info("c1")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_user_info_httpx_unavailable(self):
        connector = _make_connector()
        with patch("aragora.connectors.chat.discord.HTTPX_AVAILABLE", False):
            result = await connector.get_user_info("u1")
        assert result is None


# ===========================================================================
# 19. get_channel_info — metadata fields
# ===========================================================================


class TestGetChannelInfoMetadata:
    @pytest.mark.asyncio
    async def test_channel_metadata_fields(self):
        connector = _make_connector()
        data = {
            "id": "c1",
            "type": 0,
            "name": "general",
            "topic": "General discussion",
            "guild_id": "g1",
            "nsfw": True,
            "position": 3,
            "parent_id": "cat-1",
            "rate_limit_per_user": 5,
        }
        with patch.object(connector, "_http_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = (True, data, None)
            channel = await connector.get_channel_info("c1")

        assert channel is not None
        assert channel.metadata["topic"] == "General discussion"
        assert channel.metadata["nsfw"] is True
        assert channel.metadata["position"] == 3
        assert channel.metadata["parent_id"] == "cat-1"
        assert channel.metadata["rate_limit_per_user"] == 5
        assert channel.metadata["type"] == 0

    @pytest.mark.asyncio
    async def test_group_dm_channel(self):
        """Type 3 (GROUP_DM) should be marked as DM."""
        connector = _make_connector()
        data = {"id": "gdm-1", "type": 3, "name": "Group Chat"}
        with patch.object(connector, "_http_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = (True, data, None)
            channel = await connector.get_channel_info("gdm-1")

        assert channel is not None
        assert channel.is_dm is True


# ===========================================================================
# 20. get_user_info — metadata fields
# ===========================================================================


class TestGetUserInfoMetadata:
    @pytest.mark.asyncio
    async def test_user_metadata_fields(self):
        connector = _make_connector()
        data = {
            "id": "u1",
            "username": "testuser",
            "global_name": "Test User",
            "avatar": "av1",
            "discriminator": "1234",
            "accent_color": 0xFF0000,
            "banner": "banner_hash",
            "public_flags": 256,
            "bot": False,
        }
        with patch.object(connector, "_http_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = (True, data, None)
            user = await connector.get_user_info("u1")

        assert user is not None
        assert user.metadata["discriminator"] == "1234"
        assert user.metadata["accent_color"] == 0xFF0000
        assert user.metadata["banner"] == "banner_hash"
        assert user.metadata["public_flags"] == 256


# ===========================================================================
# 21. Platform properties
# ===========================================================================


class TestPlatformProperties:
    def test_platform_name(self):
        connector = _make_connector()
        assert connector.platform_name == "discord"

    def test_platform_display_name(self):
        connector = _make_connector()
        assert connector.platform_display_name == "Discord"


# ===========================================================================
# 22. Headers include trace context
# ===========================================================================


class TestHeaders:
    def test_headers_include_auth(self):
        connector = _make_connector()
        headers = connector._get_headers()
        assert headers["Authorization"] == "Bot test-token"
        assert headers["Content-Type"] == "application/json"

    def test_headers_include_trace_context(self):
        connector = _make_connector()
        with patch(
            "aragora.connectors.chat.discord.build_trace_headers",
            return_value={"traceparent": "00-trace-id"},
        ):
            headers = connector._get_headers()
        assert headers["traceparent"] == "00-trace-id"


# ===========================================================================
# 23. test_connection and is_configured
# ===========================================================================


class TestConnectionAndConfig:
    @pytest.mark.asyncio
    async def test_connection_with_token(self):
        connector = _make_connector()
        result = await connector.test_connection()
        assert result["success"] is True
        assert result["platform"] == "discord"
        assert result["bot_token_configured"] is True

    @pytest.mark.asyncio
    async def test_connection_without_token(self):
        from aragora.connectors.chat.discord import DiscordConnector

        connector = DiscordConnector(bot_token="")
        result = await connector.test_connection()
        assert result["success"] is False

    def test_is_configured_true(self):
        connector = _make_connector()
        assert connector.is_configured is True

    def test_is_configured_false(self):
        from aragora.connectors.chat.discord import DiscordConnector

        connector = DiscordConnector(bot_token="")
        assert connector.is_configured is False


# ===========================================================================
# 24. get_health
# ===========================================================================


class TestGetHealth:
    @pytest.mark.asyncio
    async def test_health_configured_no_cb(self):
        connector = _make_connector(enable_circuit_breaker=False)
        health = await connector.get_health()
        assert health["status"] == "healthy"
        assert health["configured"] is True
        assert health["circuit_breaker"] == {"enabled": False}

    @pytest.mark.asyncio
    async def test_health_unconfigured(self):
        from aragora.connectors.chat.discord import DiscordConnector

        connector = DiscordConnector(bot_token="", enable_circuit_breaker=False)
        health = await connector.get_health()
        assert health["status"] == "unconfigured"


# ===========================================================================
# 25. send_message with no data in success response
# ===========================================================================


class TestSendMessageEdgeCases:
    @pytest.mark.asyncio
    async def test_send_message_success_but_no_data(self):
        connector = _make_connector()
        with patch.object(connector, "_http_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = (True, None, None)
            result = await connector.send_message("c1", "text")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_send_message_success_non_dict_data(self):
        connector = _make_connector()
        with patch.object(connector, "_http_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = (True, "not a dict", None)
            result = await connector.send_message("c1", "text")
        assert result.success is False

    @pytest.mark.asyncio
    async def test_send_message_httpx_unavailable(self):
        connector = _make_connector()
        with patch("aragora.connectors.chat.discord.HTTPX_AVAILABLE", False):
            result = await connector.send_message("c1", "text")
        assert result.success is False
        assert "httpx" in (result.error or "").lower()


# ===========================================================================
# 26. Init with public_key
# ===========================================================================


class TestInitPublicKey:
    def test_public_key_stored(self):
        connector = _make_connector(public_key="my-pub-key")
        assert connector.public_key == "my-pub-key"

    def test_public_key_empty_default(self):
        from aragora.connectors.chat.discord import DiscordConnector

        connector = DiscordConnector()
        # Should fall back to env var or empty string
        assert isinstance(connector.public_key, str)


# ===========================================================================
# 27. Channel history - metadata on messages
# ===========================================================================


class TestChannelHistoryMetadata:
    @pytest.mark.asyncio
    async def test_message_metadata_includes_reactions_attachments_embeds(self):
        connector = _make_connector()
        api_response = [
            {
                "id": "msg-1",
                "content": "look at this",
                "author": {"id": "u1", "username": "a", "bot": False},
                "timestamp": "2025-01-15T12:00:00Z",
                "reactions": [{"emoji": {"name": "thumbsup"}, "count": 3}],
                "attachments": [{"id": "att-1", "filename": "file.png"}],
                "embeds": [{"title": "Link Preview"}],
            }
        ]
        with patch.object(connector, "_http_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = (True, api_response, None)
            messages = await connector.get_channel_history("c1")

        meta = messages[0].metadata
        assert len(meta["reactions"]) == 1
        assert meta["reactions"][0]["emoji"]["name"] == "thumbsup"
        assert len(meta["attachments"]) == 1
        assert len(meta["embeds"]) == 1


# ===========================================================================
# 28. update_message URL and HTTP method verification
# ===========================================================================


class TestUpdateMessageURL:
    @pytest.mark.asyncio
    async def test_update_message_uses_correct_url(self):
        connector = _make_connector()
        with patch.object(connector, "_http_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = (True, {}, None)
            await connector.update_message("chan-42", "msg-77", "new text")
        url = mock_req.call_args[1]["url"]
        assert "/channels/chan-42/messages/msg-77" in url
        assert mock_req.call_args[1]["method"] == "PATCH"

    @pytest.mark.asyncio
    async def test_update_message_failure_returns_error(self):
        connector = _make_connector()
        with patch.object(connector, "_http_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = (False, None, "Not Found")
            result = await connector.update_message("c1", "m1", "text")
        assert result.success is False
        assert "Not Found" in (result.error or "")


# ===========================================================================
# 29. delete_message URL verification
# ===========================================================================


class TestDeleteMessageURL:
    @pytest.mark.asyncio
    async def test_delete_uses_correct_url(self):
        connector = _make_connector()
        with patch.object(connector, "_http_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = (True, {}, None)
            result = await connector.delete_message("chan-42", "msg-77")
        assert result is True
        url = mock_req.call_args[1]["url"]
        assert "/channels/chan-42/messages/msg-77" in url
        assert mock_req.call_args[1]["method"] == "DELETE"


# ===========================================================================
# 30. send_message timestamp field
# ===========================================================================


class TestSendMessageTimestamp:
    @pytest.mark.asyncio
    async def test_send_message_returns_timestamp(self):
        connector = _make_connector()
        resp_data = {
            "id": "msg-1",
            "channel_id": "c1",
            "timestamp": "2025-06-15T10:30:00.000Z",
        }
        with patch.object(connector, "_http_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = (True, resp_data, None)
            result = await connector.send_message("c1", "hello")
        assert result.timestamp == "2025-06-15T10:30:00.000Z"
        assert result.channel_id == "c1"
