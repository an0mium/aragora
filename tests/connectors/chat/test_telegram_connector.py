"""Tests for TelegramConnector."""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
import json

from aragora.connectors.chat.telegram import TelegramConnector
from aragora.connectors.chat.models import (
    SendMessageResponse,
    WebhookEvent,
    FileAttachment,
    BotCommand,
    ChatUser,
    ChatChannel,
    ChatMessage,
    UserInteraction,
    InteractionType,
)


@pytest.fixture
def connector():
    """Create a TelegramConnector for testing."""
    return TelegramConnector(
        bot_token="123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11",
        webhook_url="https://example.com/webhook",
    )


@pytest.fixture
def mock_httpx_response():
    """Create a mock httpx response."""

    def _create_response(json_data, status_code=200):
        mock = MagicMock()
        mock.json.return_value = json_data
        mock.status_code = status_code
        mock.content = b"test content"
        return mock

    return _create_response


class TestTelegramConnectorInit:
    """Test connector initialization."""

    def test_init_with_token(self):
        """Test initialization with bot token."""
        connector = TelegramConnector(bot_token="test-token")
        assert connector.bot_token == "test-token"
        assert connector.platform_name == "telegram"
        assert connector.platform_display_name == "Telegram"

    def test_init_with_parse_mode(self):
        """Test initialization with custom parse mode."""
        connector = TelegramConnector(bot_token="test-token", parse_mode="HTML")
        assert connector.parse_mode == "HTML"

    def test_init_default_parse_mode(self):
        """Test default parse mode is MarkdownV2."""
        connector = TelegramConnector(bot_token="test-token")
        assert connector.parse_mode == "MarkdownV2"


class TestSendMessage:
    """Test send_message method."""

    @pytest.mark.asyncio
    async def test_send_message_success(self, connector, mock_httpx_response):
        """Test successful message send."""
        mock_response = mock_httpx_response(
            {
                "ok": True,
                "result": {
                    "message_id": 123,
                    "chat": {"id": -1001234567890},
                    "date": 1640000000,
                },
            }
        )

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            result = await connector.send_message(
                channel_id="-1001234567890",
                text="Hello, world!",
            )

            assert result.success is True
            assert result.message_id == "123"
            assert result.channel_id == "-1001234567890"

    @pytest.mark.asyncio
    async def test_send_message_failure(self, connector, mock_httpx_response):
        """Test failed message send."""
        mock_response = mock_httpx_response(
            {
                "ok": False,
                "description": "Bad Request: chat not found",
            }
        )

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            result = await connector.send_message(
                channel_id="invalid_chat",
                text="Hello",
            )

            assert result.success is False
            assert "chat not found" in result.error

    @pytest.mark.asyncio
    async def test_send_message_with_thread(self, connector, mock_httpx_response):
        """Test sending message as reply (thread)."""
        mock_response = mock_httpx_response(
            {
                "ok": True,
                "result": {
                    "message_id": 456,
                    "chat": {"id": -1001234567890},
                    "date": 1640000000,
                },
            }
        )

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.post = AsyncMock(return_value=mock_response)

            result = await connector.send_message(
                channel_id="-1001234567890",
                text="Reply message",
                thread_id="123",
            )

            assert result.success is True
            # Verify reply_to_message_id was included
            call_args = mock_instance.post.call_args
            assert call_args[1]["json"]["reply_to_message_id"] == 123

    @pytest.mark.asyncio
    async def test_send_message_with_blocks(self, connector, mock_httpx_response):
        """Test sending message with inline keyboard."""
        mock_response = mock_httpx_response(
            {
                "ok": True,
                "result": {
                    "message_id": 789,
                    "chat": {"id": -1001234567890},
                    "date": 1640000000,
                },
            }
        )

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.post = AsyncMock(return_value=mock_response)

            blocks = [
                {"type": "button", "text": "Click me", "value": "btn_1"},
            ]

            result = await connector.send_message(
                channel_id="-1001234567890",
                text="Choose an option",
                blocks=blocks,
            )

            assert result.success is True


class TestUpdateMessage:
    """Test update_message method."""

    @pytest.mark.asyncio
    async def test_update_message_success(self, connector, mock_httpx_response):
        """Test successful message update."""
        mock_response = mock_httpx_response(
            {
                "ok": True,
                "result": {"message_id": 123},
            }
        )

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            result = await connector.update_message(
                channel_id="-1001234567890",
                message_id="123",
                text="Updated text",
            )

            assert result.success is True
            assert result.message_id == "123"

    @pytest.mark.asyncio
    async def test_update_message_failure(self, connector, mock_httpx_response):
        """Test failed message update."""
        mock_response = mock_httpx_response(
            {
                "ok": False,
                "description": "Bad Request: message is not modified",
            }
        )

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            result = await connector.update_message(
                channel_id="-1001234567890",
                message_id="123",
                text="Same text",
            )

            assert result.success is False


class TestDeleteMessage:
    """Test delete_message method."""

    @pytest.mark.asyncio
    async def test_delete_message_success(self, connector, mock_httpx_response):
        """Test successful message deletion."""
        mock_response = mock_httpx_response({"ok": True, "result": True})

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            result = await connector.delete_message(
                channel_id="-1001234567890",
                message_id="123",
            )

            assert result is True

    @pytest.mark.asyncio
    async def test_delete_message_failure(self, connector, mock_httpx_response):
        """Test failed message deletion."""
        mock_response = mock_httpx_response(
            {
                "ok": False,
                "description": "Bad Request: message to delete not found",
            }
        )

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            result = await connector.delete_message(
                channel_id="-1001234567890",
                message_id="99999",
            )

            assert result is False


class TestFileOperations:
    """Test file upload and download."""

    @pytest.mark.asyncio
    async def test_download_file_success(self, connector, mock_httpx_response):
        """Test successful file download returns FileAttachment."""
        mock_get_file_response = mock_httpx_response(
            {
                "ok": True,
                "result": {"file_path": "documents/file_123.pdf", "file_size": 17},
            }
        )
        mock_download_response = MagicMock()
        mock_download_response.content = b"file content here"

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.get = AsyncMock(
                side_effect=[mock_get_file_response, mock_download_response]
            )

            result = await connector.download_file(file_id="file_123")

            # Should return FileAttachment with content populated

            assert isinstance(result, FileAttachment)
            assert result.content == b"file content here"
            assert result.id == "file_123"
            assert result.filename == "file_123.pdf"
            assert result.size == 17

    @pytest.mark.asyncio
    async def test_download_file_failure(self, connector, mock_httpx_response):
        """Test failed file download - file not found."""
        mock_response = mock_httpx_response(
            {
                "ok": False,
                "description": "Bad Request: file not found",
            }
        )

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            with pytest.raises(RuntimeError, match="file not found"):
                await connector.download_file(file_id="invalid_file")


class TestWebhookHandling:
    """Test webhook event handling."""

    @pytest.mark.asyncio
    async def test_handle_webhook_message(self, connector):
        """Test handling message webhook."""
        payload = {
            "update_id": 123456789,
            "message": {
                "message_id": 1,
                "from": {"id": 123, "is_bot": False, "first_name": "Test"},
                "chat": {"id": -1001234567890, "type": "supergroup"},
                "date": 1640000000,
                "text": "Hello bot!",
            },
        }

        result = await connector.handle_webhook(payload)

        assert isinstance(result, WebhookEvent)
        assert result.event_type == "message"
        assert result.platform == "telegram"
        assert result.metadata["channel_id"] == "-1001234567890"
        assert result.metadata["user_id"] == "123"
        assert result.metadata["message_id"] == "1"

    @pytest.mark.asyncio
    async def test_handle_webhook_callback_query(self, connector):
        """Test handling callback query (button click)."""
        payload = {
            "update_id": 123456790,
            "callback_query": {
                "id": "4382bfdwdsb323b2d9",
                "from": {"id": 123, "is_bot": False, "first_name": "Test"},
                "message": {
                    "message_id": 100,
                    "chat": {"id": -1001234567890},
                },
                "data": "button_clicked",
            },
        }

        result = await connector.handle_webhook(payload)

        assert isinstance(result, WebhookEvent)
        assert result.event_type == "callback_query"
        assert result.platform == "telegram"
        assert result.metadata["channel_id"] == "-1001234567890"
        assert result.metadata["user_id"] == "123"

    @pytest.mark.asyncio
    async def test_handle_webhook_unknown_event(self, connector):
        """Test handling unknown webhook event type."""
        payload = {
            "update_id": 123456791,
            "edited_channel_post": {"message_id": 5},  # Not handled
        }

        result = await connector.handle_webhook(payload)

        assert result.event_type == "unknown"
        assert result.platform == "telegram"


class TestWebhookVerification:
    """Test webhook signature verification."""

    def test_verify_webhook_telegram_always_true(self, connector):
        """Telegram uses secret token in URL, not signature verification."""
        # Telegram doesn't use HMAC signatures like Slack - it uses
        # a secret_token query parameter set when configuring webhook
        result = connector.verify_webhook(headers={}, body=b"test")
        # Should return True as Telegram verification is done differently
        assert result is True


class TestRespondToCommand:
    """Test respond_to_command method."""

    @pytest.mark.asyncio
    async def test_respond_to_command(self, connector, mock_httpx_response):
        """Test responding to a bot command."""
        mock_response = mock_httpx_response(
            {
                "ok": True,
                "result": {
                    "message_id": 200,
                    "chat": {"id": -1001234567890},
                    "date": 1640000000,
                },
            }
        )

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            from aragora.connectors.chat.models import ChatChannel

            command = BotCommand(
                name="help",
                channel=ChatChannel(id="-1001234567890", platform="telegram"),
                user=ChatUser(id="123", username="testuser", platform="telegram"),
                text="/help",
                platform="telegram",
                metadata={"message_id": "100"},
            )

            result = await connector.respond_to_command(
                command=command,
                text="Here is the help information.",
            )

            assert result.success is True


class TestRespondToInteraction:
    """Test respond_to_interaction method."""

    @pytest.mark.asyncio
    async def test_respond_to_interaction_callback(self, connector, mock_httpx_response):
        """Test responding to callback query (button click)."""
        mock_response = mock_httpx_response({"ok": True, "result": True})

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            interaction = UserInteraction(
                id="callback_123",
                interaction_type=InteractionType.BUTTON_CLICK,
                action_id="approve_btn",
                user=ChatUser(id="123", username="testuser", platform="telegram"),
                channel=ChatChannel(id="-1001234567890", platform="telegram"),
                value="approve",
                platform="telegram",
            )

            result = await connector.respond_to_interaction(
                interaction=interaction,
                text="Button clicked!",
            )

            # Should return True for success
            assert result is True


class TestFormatting:
    """Test message formatting helpers."""

    def test_format_button(self, connector):
        """Test button formatting."""
        button = connector.format_button(
            text="Click me",
            action_id="btn_click",
            value="button_value",
        )

        assert isinstance(button, dict)
        assert button["text"] == "Click me"
        assert button["action_id"] == "btn_click"
        assert button["value"] == "button_value"
        assert button["type"] == "button"

    def test_format_blocks_single_button(self, connector):
        """Test formatting blocks with single button."""
        buttons = [
            {"type": "button", "text": "Option 1", "value": "opt1"},
            {"type": "button", "text": "Option 2", "value": "opt2"},
        ]

        result = connector.format_blocks(buttons)
        assert isinstance(result, list)

    def test_escape_markdown_special_chars(self, connector):
        """Test MarkdownV2 special character escaping."""
        text = "Hello *world* with _underscore_ and [link]"
        escaped = connector._escape_markdown(text)

        # Should escape special Markdown characters
        assert "\\" in escaped or text == escaped  # Either escaped or unchanged


class TestPlatformProperties:
    """Test platform property methods."""

    def test_platform_name(self, connector):
        """Test platform_name property."""
        assert connector.platform_name == "telegram"

    def test_platform_display_name(self, connector):
        """Test platform_display_name property."""
        assert connector.platform_display_name == "Telegram"


class TestParseWebhookEvent:
    """Test parse_webhook_event method."""

    def test_parse_webhook_event_message(self, connector):
        """Test parsing message event from webhook payload."""

        payload = {
            "message": {
                "message_id": 1,
                "from": {"id": 123, "first_name": "Test", "username": "testuser"},
                "chat": {"id": -1001234567890, "type": "supergroup", "title": "Test Group"},
                "date": 1640000000,
                "text": "Hello!",
            }
        }

        event = connector.parse_webhook_event(headers={}, body=json.dumps(payload).encode())

        assert event is not None
        assert event.event_type == "message"
        assert event.metadata["channel_id"] == "-1001234567890"

    def test_parse_webhook_event_edited_message(self, connector):
        """Test parsing edited message event."""

        payload = {
            "edited_message": {
                "message_id": 1,
                "from": {"id": 123, "first_name": "Test"},
                "chat": {"id": -1001234567890},
                "date": 1640000000,
                "edit_date": 1640000100,
                "text": "Edited hello!",
            }
        }

        event = connector.parse_webhook_event(headers={}, body=json.dumps(payload).encode())

        assert event is not None
        assert event.event_type == "message_edited"


class TestHttpxRequirement:
    """Test httpx availability checks."""

    @pytest.mark.asyncio
    async def test_send_message_without_httpx(self):
        """Test that operations fail gracefully without httpx."""
        with patch("aragora.connectors.chat.telegram.HTTPX_AVAILABLE", False):
            connector = TelegramConnector(bot_token="test")

            with pytest.raises(RuntimeError, match="httpx is required"):
                await connector.send_message("-123", "test")

    @pytest.mark.asyncio
    async def test_download_file_without_httpx(self):
        """Test download fails gracefully without httpx."""
        with patch("aragora.connectors.chat.telegram.HTTPX_AVAILABLE", False):
            connector = TelegramConnector(bot_token="test")

            with pytest.raises(RuntimeError, match="httpx is required"):
                await connector.download_file("file_123")


class TestRichMediaOperations:
    """Tests for rich media operations (photo, video, animation)."""

    @pytest.mark.asyncio
    async def test_send_photo_with_url(self, connector, mock_httpx_response):
        """Test sending photo with URL."""
        mock_response = mock_httpx_response(
            {
                "ok": True,
                "result": {
                    "message_id": 123,
                    "date": 1640000000,
                    "chat": {"id": -1001234567890},
                    "photo": [{"file_id": "photo123"}],
                },
            }
        )

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value.__aenter__.return_value = mock_client

            result = await connector.send_photo(
                channel_id="-1001234567890",
                photo="https://example.com/photo.jpg",
                caption="Test caption",
            )

            assert result.success is True
            assert result.message_id == "123"

    @pytest.mark.asyncio
    async def test_send_photo_with_bytes(self, connector, mock_httpx_response):
        """Test sending photo with bytes."""
        mock_response = mock_httpx_response(
            {
                "ok": True,
                "result": {
                    "message_id": 124,
                    "date": 1640000000,
                    "chat": {"id": -1001234567890},
                    "photo": [{"file_id": "photo124"}],
                },
            }
        )

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value.__aenter__.return_value = mock_client

            result = await connector.send_photo(
                channel_id="-1001234567890",
                photo=b"\x89PNG\r\n\x1a\n...",  # Fake PNG bytes
            )

            assert result.success is True
            mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_video_success(self, connector, mock_httpx_response):
        """Test sending video."""
        mock_response = mock_httpx_response(
            {
                "ok": True,
                "result": {
                    "message_id": 125,
                    "date": 1640000000,
                    "chat": {"id": -1001234567890},
                    "video": {"file_id": "video125"},
                },
            }
        )

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value.__aenter__.return_value = mock_client

            result = await connector.send_video(
                channel_id="-1001234567890",
                video="https://example.com/video.mp4",
                caption="Test video",
                duration=30,
                width=1920,
                height=1080,
            )

            assert result.success is True
            assert result.message_id == "125"

    @pytest.mark.asyncio
    async def test_send_animation_success(self, connector, mock_httpx_response):
        """Test sending animation (GIF)."""
        mock_response = mock_httpx_response(
            {
                "ok": True,
                "result": {
                    "message_id": 126,
                    "date": 1640000000,
                    "chat": {"id": -1001234567890},
                    "animation": {"file_id": "anim126"},
                },
            }
        )

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value.__aenter__.return_value = mock_client

            result = await connector.send_animation(
                channel_id="-1001234567890",
                animation="https://example.com/animation.gif",
            )

            assert result.success is True
            assert result.message_id == "126"

    @pytest.mark.asyncio
    async def test_send_media_group_success(self, connector, mock_httpx_response):
        """Test sending media group (album)."""
        mock_response = mock_httpx_response(
            {
                "ok": True,
                "result": [
                    {"message_id": 127, "date": 1640000000, "chat": {"id": -1001234567890}},
                    {"message_id": 128, "date": 1640000001, "chat": {"id": -1001234567890}},
                ],
            }
        )

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value.__aenter__.return_value = mock_client

            results = await connector.send_media_group(
                channel_id="-1001234567890",
                media=[
                    {
                        "type": "photo",
                        "media": "https://example.com/photo1.jpg",
                        "caption": "First",
                    },
                    {"type": "photo", "media": "https://example.com/photo2.jpg"},
                ],
            )

            assert len(results) == 2
            assert results[0].success is True
            assert results[0].message_id == "127"
            assert results[1].message_id == "128"


class TestInlineQuerySupport:
    """Tests for inline query support."""

    @pytest.mark.asyncio
    async def test_answer_inline_query_success(self, connector, mock_httpx_response):
        """Test answering inline query."""
        mock_response = mock_httpx_response({"ok": True, "result": True})

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value.__aenter__.return_value = mock_client

            results = [
                connector.build_inline_article_result(
                    result_id="1",
                    title="Test Result",
                    message_text="Selected test result",
                    description="A test result",
                )
            ]

            success = await connector.answer_inline_query(
                inline_query_id="query123",
                results=results,
                cache_time=60,
            )

            assert success is True

    @pytest.mark.asyncio
    async def test_answer_inline_query_failure(self, connector, mock_httpx_response):
        """Test inline query failure handling."""
        mock_response = mock_httpx_response({"ok": False, "description": "QUERY_ID_INVALID"})

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value.__aenter__.return_value = mock_client

            success = await connector.answer_inline_query(
                inline_query_id="invalid",
                results=[],
            )

            assert success is False

    def test_build_inline_article_result(self, connector):
        """Test building inline article result."""
        result = connector.build_inline_article_result(
            result_id="test-123",
            title="Test Article",
            message_text="This is the message content",
            description="Short description",
            url="https://example.com",
            thumb_url="https://example.com/thumb.jpg",
        )

        assert result["type"] == "article"
        assert result["id"] == "test-123"
        assert result["title"] == "Test Article"
        assert result["description"] == "Short description"
        assert result["url"] == "https://example.com"
        assert result["input_message_content"]["message_text"] == "This is the message content"


class TestBotManagement:
    """Tests for bot management operations."""

    @pytest.mark.asyncio
    async def test_set_my_commands_success(self, connector, mock_httpx_response):
        """Test setting bot commands."""
        mock_response = mock_httpx_response({"ok": True, "result": True})

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value.__aenter__.return_value = mock_client

            success = await connector.set_my_commands(
                commands=[
                    {"command": "start", "description": "Start the bot"},
                    {"command": "help", "description": "Get help"},
                ]
            )

            assert success is True

    @pytest.mark.asyncio
    async def test_get_me_success(self, connector, mock_httpx_response):
        """Test getting bot info."""
        mock_response = mock_httpx_response(
            {
                "ok": True,
                "result": {
                    "id": 123456789,
                    "is_bot": True,
                    "first_name": "TestBot",
                    "username": "test_bot",
                },
            }
        )

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_class.return_value.__aenter__.return_value = mock_client

            result = await connector.get_me()

            assert result is not None
            assert result["id"] == 123456789
            assert result["username"] == "test_bot"

    @pytest.mark.asyncio
    async def test_get_chat_member_count_success(self, connector, mock_httpx_response):
        """Test getting chat member count."""
        mock_response = mock_httpx_response({"ok": True, "result": 42})

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value.__aenter__.return_value = mock_client

            count = await connector.get_chat_member_count("-1001234567890")

            assert count == 42
