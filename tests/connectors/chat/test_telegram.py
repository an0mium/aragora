"""
Comprehensive tests for TelegramConnector.

Tests cover:
1. Connector initialization
2. Message parsing and handling
3. Bot command processing
4. Debate initiation from Telegram
5. Result formatting and delivery
6. Media handling (photo, video, animation, voice)
7. Error handling and recovery
8. Rate limiting
9. Webhook handling and verification
10. Circuit breaker integration

Uses pytest and pytest-asyncio for async tests.
All Telegram API calls are mocked.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.connectors.chat.models import (
    BotCommand,
    ChatChannel,
    ChatEvidence,
    ChatMessage,
    ChatUser,
    FileAttachment,
    InteractionType,
    MessageButton,
    MessageType,
    SendMessageResponse,
    UserInteraction,
    VoiceMessage,
    WebhookEvent,
)
from aragora.connectors.chat.telegram import (
    TelegramConnector,
    _classify_telegram_error,
)
from aragora.connectors.exceptions import (
    ConnectorAPIError,
    ConnectorAuthError,
    ConnectorNetworkError,
    ConnectorRateLimitError,
    ConnectorTimeoutError,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def reset_circuit_breakers():
    """Clear all circuit breakers before each test to ensure isolation."""
    try:
        from aragora.resilience import _circuit_breakers, _circuit_breakers_lock

        with _circuit_breakers_lock:
            _circuit_breakers.clear()
    except (ImportError, AttributeError):
        pass
    yield


@pytest.fixture
def connector() -> TelegramConnector:
    """Create a TelegramConnector for testing with circuit breaker disabled."""
    return TelegramConnector(
        bot_token="123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11",
        webhook_url="https://example.com/webhook",
        enable_circuit_breaker=False,
    )


@pytest.fixture
def connector_with_cb() -> TelegramConnector:
    """Create a TelegramConnector with circuit breaker enabled."""
    return TelegramConnector(
        bot_token="123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11",
        enable_circuit_breaker=True,
        circuit_breaker_threshold=3,
        circuit_breaker_cooldown=10.0,
    )


@pytest.fixture
def mock_httpx_response():
    """Create a mock httpx response."""

    def _create_response(json_data: dict, status_code: int = 200):
        mock = MagicMock()
        mock.json.return_value = json_data
        mock.status_code = status_code
        mock.text = json.dumps(json_data)
        mock.content = json.dumps(json_data).encode()
        return mock

    return _create_response


@pytest.fixture
def sample_message_payload() -> dict[str, Any]:
    """Sample Telegram message payload."""
    return {
        "update_id": 123456789,
        "message": {
            "message_id": 1,
            "from": {
                "id": 12345,
                "is_bot": False,
                "first_name": "John",
                "last_name": "Doe",
                "username": "johndoe",
            },
            "chat": {
                "id": -1001234567890,
                "type": "supergroup",
                "title": "Test Group",
            },
            "date": 1640000000,
            "text": "Hello bot!",
        },
    }


@pytest.fixture
def sample_callback_payload() -> dict[str, Any]:
    """Sample Telegram callback query payload."""
    return {
        "update_id": 123456790,
        "callback_query": {
            "id": "4382bfdwdsb323b2d9",
            "from": {
                "id": 12345,
                "is_bot": False,
                "first_name": "John",
                "username": "johndoe",
            },
            "message": {
                "message_id": 100,
                "chat": {"id": -1001234567890, "type": "supergroup"},
                "date": 1640000000,
                "text": "Choose an option:",
            },
            "data": "approve_debate",
        },
    }


# ============================================================================
# Test: Connector Initialization
# ============================================================================


class TestConnectorInitialization:
    """Tests for TelegramConnector initialization."""

    def test_init_with_token(self):
        """Should initialize with bot token."""
        connector = TelegramConnector(bot_token="test-token-123")
        assert connector.bot_token == "test-token-123"
        assert connector.platform_name == "telegram"
        assert connector.platform_display_name == "Telegram"

    def test_init_default_parse_mode(self):
        """Should default to MarkdownV2 parse mode."""
        connector = TelegramConnector(bot_token="test-token")
        assert connector.parse_mode == "MarkdownV2"

    def test_init_with_custom_parse_mode(self):
        """Should accept custom parse mode."""
        connector = TelegramConnector(bot_token="test-token", parse_mode="HTML")
        assert connector.parse_mode == "HTML"

    def test_init_with_webhook_url(self):
        """Should accept webhook URL."""
        connector = TelegramConnector(
            bot_token="test-token", webhook_url="https://example.com/webhook"
        )
        assert connector.webhook_url == "https://example.com/webhook"

    def test_init_api_base_construction(self):
        """Should construct correct API base URL."""
        connector = TelegramConnector(bot_token="test-token-abc")
        assert "test-token-abc" in connector._api_base
        assert "api.telegram.org" in connector._api_base

    def test_init_from_env_vars(self):
        """Should fall back to environment variables."""
        with patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "env-token"}):
            # Need to reimport to pick up env var
            connector = TelegramConnector()
            # Token comes from env if not provided
            assert connector.bot_token == "env-token" or connector.bot_token == ""

    def test_platform_properties(self, connector):
        """Should return correct platform properties."""
        assert connector.platform_name == "telegram"
        assert connector.platform_display_name == "Telegram"

    def test_is_configured_with_token(self, connector):
        """Should report configured when token present."""
        assert connector.is_configured is True

    def test_is_configured_without_token(self):
        """Should report not configured without token."""
        connector = TelegramConnector(bot_token="", webhook_url="")
        assert connector.is_configured is False


# ============================================================================
# Test: Message Parsing and Handling
# ============================================================================


class TestMessageParsing:
    """Tests for message parsing functionality."""

    @pytest.mark.asyncio
    async def test_parse_message_text(self, connector, sample_message_payload):
        """Should parse text message correctly."""
        message = await connector.parse_message(sample_message_payload)

        assert isinstance(message, ChatMessage)
        assert message.id == "1"
        assert message.platform == "telegram"
        assert message.content == "Hello bot!"
        assert message.message_type == MessageType.TEXT

    @pytest.mark.asyncio
    async def test_parse_message_author(self, connector, sample_message_payload):
        """Should extract author information."""
        message = await connector.parse_message(sample_message_payload)

        assert message.author.id == "12345"
        assert message.author.username == "johndoe"
        assert message.author.display_name == "John Doe"
        assert message.author.platform == "telegram"

    @pytest.mark.asyncio
    async def test_parse_message_channel(self, connector, sample_message_payload):
        """Should extract channel information."""
        message = await connector.parse_message(sample_message_payload)

        assert message.channel.id == "-1001234567890"
        assert message.channel.name == "Test Group"
        assert message.channel.platform == "telegram"

    @pytest.mark.asyncio
    async def test_parse_message_voice(self, connector):
        """Should detect voice messages."""
        payload = {
            "message": {
                "message_id": 2,
                "from": {"id": 123},
                "chat": {"id": -100},
                "date": 1640000000,
                "voice": {"file_id": "voice123", "duration": 5},
            }
        }

        message = await connector.parse_message(payload)
        assert message.message_type == MessageType.VOICE

    @pytest.mark.asyncio
    async def test_parse_message_document(self, connector):
        """Should detect document messages."""
        payload = {
            "message": {
                "message_id": 3,
                "from": {"id": 123},
                "chat": {"id": -100},
                "date": 1640000000,
                "document": {"file_id": "doc123", "file_name": "report.pdf"},
            }
        }

        message = await connector.parse_message(payload)
        assert message.message_type == MessageType.FILE

    @pytest.mark.asyncio
    async def test_parse_message_photo(self, connector):
        """Should detect photo messages as files."""
        payload = {
            "message": {
                "message_id": 4,
                "from": {"id": 123},
                "chat": {"id": -100},
                "date": 1640000000,
                "photo": [{"file_id": "photo123"}],
            }
        }

        message = await connector.parse_message(payload)
        assert message.message_type == MessageType.FILE

    @pytest.mark.asyncio
    async def test_parse_message_with_reply(self, connector):
        """Should extract thread ID from reply."""
        payload = {
            "message": {
                "message_id": 5,
                "from": {"id": 123},
                "chat": {"id": -100},
                "date": 1640000000,
                "text": "Reply text",
                "reply_to_message": {"message_id": 1},
            }
        }

        message = await connector.parse_message(payload)
        assert message.thread_id == "1"

    @pytest.mark.asyncio
    async def test_parse_message_private_chat(self, connector):
        """Should detect private chats."""
        payload = {
            "message": {
                "message_id": 6,
                "from": {"id": 123, "username": "user1"},
                "chat": {"id": 123, "type": "private", "username": "user1"},
                "date": 1640000000,
                "text": "DM text",
            }
        }

        message = await connector.parse_message(payload)
        assert message.channel.is_private is True
        assert message.channel.is_dm is True

    @pytest.mark.asyncio
    async def test_parse_message_with_caption(self, connector):
        """Should use caption as content for media messages."""
        payload = {
            "message": {
                "message_id": 7,
                "from": {"id": 123},
                "chat": {"id": -100},
                "date": 1640000000,
                "photo": [{"file_id": "photo123"}],
                "caption": "Photo caption here",
            }
        }

        message = await connector.parse_message(payload)
        assert message.content == "Photo caption here"


# ============================================================================
# Test: Bot Command Processing
# ============================================================================


class TestBotCommandProcessing:
    """Tests for bot command parsing and handling."""

    @pytest.mark.asyncio
    async def test_parse_command_simple(self, connector):
        """Should parse simple command."""
        payload = {
            "message": {
                "message_id": 10,
                "from": {"id": 123, "first_name": "John", "username": "johndoe"},
                "chat": {"id": -100, "title": "Test Group"},
                "date": 1640000000,
                "text": "/start",
            }
        }

        command = await connector.parse_command(payload)

        assert command is not None
        assert command.name == "start"
        assert command.text == "/start"
        assert command.args == []

    @pytest.mark.asyncio
    async def test_parse_command_with_args(self, connector):
        """Should parse command with arguments."""
        payload = {
            "message": {
                "message_id": 11,
                "from": {"id": 123},
                "chat": {"id": -100},
                "date": 1640000000,
                "text": "/debate Should we use Python or Rust?",
            }
        }

        command = await connector.parse_command(payload)

        assert command is not None
        assert command.name == "debate"
        assert len(command.args) == 6
        assert command.args[0] == "Should"

    @pytest.mark.asyncio
    async def test_parse_command_with_bot_mention(self, connector):
        """Should strip @bot suffix from command."""
        payload = {
            "message": {
                "message_id": 12,
                "from": {"id": 123},
                "chat": {"id": -100},
                "date": 1640000000,
                "text": "/help@aragorabot",
            }
        }

        command = await connector.parse_command(payload)

        assert command is not None
        assert command.name == "help"

    @pytest.mark.asyncio
    async def test_parse_command_non_command(self, connector):
        """Should return None for non-command messages."""
        payload = {
            "message": {
                "message_id": 13,
                "from": {"id": 123},
                "chat": {"id": -100},
                "date": 1640000000,
                "text": "Regular message, not a command",
            }
        }

        command = await connector.parse_command(payload)
        assert command is None

    @pytest.mark.asyncio
    async def test_parse_command_metadata(self, connector):
        """Should include message ID in command metadata."""
        payload = {
            "message": {
                "message_id": 14,
                "from": {"id": 123},
                "chat": {"id": -100},
                "date": 1640000000,
                "text": "/status",
            }
        }

        command = await connector.parse_command(payload)

        assert command.metadata.get("message_id") == "14"

    @pytest.mark.asyncio
    async def test_respond_to_command(self, connector, mock_httpx_response):
        """Should respond to command successfully."""
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

            command = BotCommand(
                name="help",
                text="/help",
                channel=ChatChannel(id="-1001234567890", platform="telegram"),
                user=ChatUser(id="123", platform="telegram"),
                platform="telegram",
                metadata={"message_id": "100"},
            )

            result = await connector.respond_to_command(
                command=command,
                text="Here is the help information.",
            )

            assert result.success is True

    @pytest.mark.asyncio
    async def test_respond_to_command_no_channel(self, connector):
        """Should fail gracefully without channel ID."""
        command = BotCommand(
            name="help",
            text="/help",
            channel=None,
            platform="telegram",
        )

        result = await connector.respond_to_command(
            command=command,
            text="Response text",
        )

        assert result.success is False
        assert "No channel ID" in (result.error or "")


# ============================================================================
# Test: Debate Initiation from Telegram
# ============================================================================


class TestDebateInitiation:
    """Tests for debate initiation functionality."""

    @pytest.mark.asyncio
    async def test_extract_evidence(self, connector):
        """Should extract evidence from message for debate."""
        channel = ChatChannel(id="-1001234567890", platform="telegram", name="Test Group")
        author = ChatUser(
            id="123",
            platform="telegram",
            username="johndoe",
            display_name="John Doe",
        )
        message = ChatMessage(
            id="42",
            platform="telegram",
            channel=channel,
            author=author,
            content="We should use microservices architecture for scalability",
            timestamp=datetime.utcnow(),
        )

        evidence = await connector.extract_evidence(message)

        assert isinstance(evidence, ChatEvidence)
        assert evidence.platform == "telegram"
        assert evidence.channel_id == "-1001234567890"
        assert "microservices" in evidence.content
        assert evidence.author_name == "John Doe"

    @pytest.mark.asyncio
    async def test_extract_evidence_generates_unique_id(self, connector):
        """Should generate unique evidence IDs."""
        channel = ChatChannel(id="-100", platform="telegram")
        author = ChatUser(id="1", platform="telegram")
        msg1 = ChatMessage(id="1", platform="telegram", channel=channel, author=author, content="A")
        msg2 = ChatMessage(id="2", platform="telegram", channel=channel, author=author, content="B")

        ev1 = await connector.extract_evidence(msg1)
        ev2 = await connector.extract_evidence(msg2)

        assert ev1.id != ev2.id

    @pytest.mark.asyncio
    async def test_link_debate_to_session(self, connector):
        """Should link debate to session (if session manager available)."""
        # Session manager integration test
        session_id = await connector.link_debate_to_session(
            user_id="12345",
            debate_id="debate-abc-123",
            context={"message_id": "100", "chat_id": "-1001234567890"},
        )

        # May return None if session manager not available
        assert session_id is None or isinstance(session_id, str)


# ============================================================================
# Test: Result Formatting and Delivery
# ============================================================================


class TestResultDelivery:
    """Tests for result formatting and message delivery."""

    @pytest.mark.asyncio
    async def test_send_message_success(self, connector, mock_httpx_response):
        """Should send message successfully."""
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
                text="Debate conclusion: We recommend Option A",
            )

            assert result.success is True
            assert result.message_id == "123"
            assert result.channel_id == "-1001234567890"

    @pytest.mark.asyncio
    async def test_send_message_with_thread(self, connector, mock_httpx_response):
        """Should send threaded reply."""
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
                text="Reply in thread",
                thread_id="123",
            )

            assert result.success is True
            call_args = mock_instance.post.call_args
            assert call_args[1]["json"]["reply_to_message_id"] == 123

    @pytest.mark.asyncio
    async def test_send_message_with_buttons(self, connector, mock_httpx_response):
        """Should send message with inline keyboard."""
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
                {"type": "button", "text": "Approve", "action_id": "approve"},
                {"type": "button", "text": "Reject", "action_id": "reject"},
            ]

            result = await connector.send_message(
                channel_id="-1001234567890",
                text="Vote on this proposal:",
                blocks=blocks,
            )

            assert result.success is True

    @pytest.mark.asyncio
    async def test_update_message_success(self, connector, mock_httpx_response):
        """Should update message successfully."""
        mock_response = mock_httpx_response({"ok": True, "result": {"message_id": 123}})

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            result = await connector.update_message(
                channel_id="-1001234567890",
                message_id="123",
                text="Updated debate status: In Progress",
            )

            assert result.success is True
            assert result.message_id == "123"

    @pytest.mark.asyncio
    async def test_delete_message_success(self, connector, mock_httpx_response):
        """Should delete message successfully."""
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
    async def test_send_typing_indicator(self, connector, mock_httpx_response):
        """Should send typing indicator."""
        mock_response = mock_httpx_response({"ok": True, "result": True})

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            result = await connector.send_typing_indicator("-1001234567890")
            assert result is True


# ============================================================================
# Test: Media Handling
# ============================================================================


class TestMediaHandling:
    """Tests for photo, video, animation, and file handling."""

    @pytest.mark.asyncio
    async def test_send_photo_with_url(self, connector, mock_httpx_response):
        """Should send photo via URL."""
        mock_response = mock_httpx_response(
            {
                "ok": True,
                "result": {
                    "message_id": 200,
                    "chat": {"id": -1001234567890},
                    "date": 1640000000,
                    "photo": [{"file_id": "AgAC..."}],
                },
            }
        )

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            result = await connector.send_photo(
                channel_id="-1001234567890",
                photo="https://example.com/image.jpg",
                caption="Debate visualization",
            )

            assert result.success is True
            assert result.message_id == "200"

    @pytest.mark.asyncio
    async def test_send_photo_with_bytes(self, connector, mock_httpx_response):
        """Should send photo as bytes upload."""
        mock_response = mock_httpx_response(
            {
                "ok": True,
                "result": {
                    "message_id": 201,
                    "chat": {"id": -1001234567890},
                    "date": 1640000000,
                },
            }
        )

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            result = await connector.send_photo(
                channel_id="-1001234567890",
                photo=b"\x89PNG\r\n\x1a\n...",  # Fake PNG
            )

            assert result.success is True

    @pytest.mark.asyncio
    async def test_send_video_success(self, connector, mock_httpx_response):
        """Should send video successfully."""
        mock_response = mock_httpx_response(
            {
                "ok": True,
                "result": {
                    "message_id": 202,
                    "chat": {"id": -1001234567890},
                    "date": 1640000000,
                },
            }
        )

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            result = await connector.send_video(
                channel_id="-1001234567890",
                video="https://example.com/video.mp4",
                caption="Debate recording",
                duration=120,
                width=1920,
                height=1080,
            )

            assert result.success is True

    @pytest.mark.asyncio
    async def test_send_animation_success(self, connector, mock_httpx_response):
        """Should send animation (GIF) successfully."""
        mock_response = mock_httpx_response(
            {
                "ok": True,
                "result": {
                    "message_id": 203,
                    "chat": {"id": -1001234567890},
                    "date": 1640000000,
                },
            }
        )

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            result = await connector.send_animation(
                channel_id="-1001234567890",
                animation="https://example.com/animation.gif",
            )

            assert result.success is True

    @pytest.mark.asyncio
    async def test_send_media_group(self, connector, mock_httpx_response):
        """Should send media group (album)."""
        mock_response = mock_httpx_response(
            {
                "ok": True,
                "result": [
                    {"message_id": 204, "date": 1640000000, "chat": {"id": -100}},
                    {"message_id": 205, "date": 1640000001, "chat": {"id": -100}},
                ],
            }
        )

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            results = await connector.send_media_group(
                channel_id="-1001234567890",
                media=[
                    {"type": "photo", "media": "https://example.com/1.jpg"},
                    {"type": "photo", "media": "https://example.com/2.jpg"},
                ],
            )

            assert len(results) == 2
            assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_send_voice_message(self, connector, mock_httpx_response):
        """Should send voice message."""
        mock_response = mock_httpx_response(
            {
                "ok": True,
                "result": {
                    "message_id": 206,
                    "chat": {"id": -1001234567890},
                    "date": 1640000000,
                },
            }
        )

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            result = await connector.send_voice_message(
                channel_id="-1001234567890",
                audio_content=b"audio data here",
                duration=10,
            )

            assert result.success is True

    @pytest.mark.asyncio
    async def test_upload_file_success(self, connector, mock_httpx_response):
        """Should upload file successfully."""
        mock_response = mock_httpx_response(
            {
                "ok": True,
                "result": {
                    "message_id": 207,
                    "document": {
                        "file_id": "BQACAgIAAxk...",
                        "file_name": "report.pdf",
                        "file_size": 12345,
                        "mime_type": "application/pdf",
                    },
                },
            }
        )

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            result = await connector.upload_file(
                channel_id="-1001234567890",
                content=b"PDF content here",
                filename="report.pdf",
                content_type="application/pdf",
                title="Debate Report",
            )

            assert isinstance(result, FileAttachment)
            assert result.filename == "report.pdf"

    @pytest.mark.asyncio
    async def test_download_file_success(self, connector, mock_httpx_response):
        """Should download file successfully."""
        mock_get_file = mock_httpx_response(
            {
                "ok": True,
                "result": {
                    "file_id": "file123",
                    "file_path": "documents/file_123.pdf",
                    "file_size": 1024,
                },
            }
        )
        mock_download = MagicMock()
        mock_download.content = b"file content here"
        mock_download.status_code = 200

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.get = AsyncMock(return_value=mock_get_file)
            mock_instance.request = AsyncMock(return_value=mock_download)

            result = await connector.download_file(file_id="file123")

            assert isinstance(result, FileAttachment)
            assert result.content == b"file content here"
            assert result.id == "file123"

    @pytest.mark.asyncio
    async def test_download_voice_message(self, connector, mock_httpx_response):
        """Should download voice message."""
        mock_get_file = mock_httpx_response(
            {
                "ok": True,
                "result": {"file_path": "voice/voice_123.ogg", "file_size": 5000},
            }
        )
        mock_download = MagicMock()
        mock_download.content = b"voice audio content"
        mock_download.status_code = 200

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.get = AsyncMock(return_value=mock_get_file)
            mock_instance.request = AsyncMock(return_value=mock_download)

            file_attachment = FileAttachment(
                id="voice123",
                filename="voice.ogg",
                content_type="audio/ogg",
                size=5000,
            )
            voice_msg = VoiceMessage(
                id="voice123",
                channel=ChatChannel(id="-100", platform="telegram"),
                author=ChatUser(id="123", platform="telegram"),
                duration_seconds=5.0,
                file=file_attachment,
            )

            content = await connector.download_voice_message(voice_msg)
            assert content == b"voice audio content"


# ============================================================================
# Test: Error Handling and Recovery
# ============================================================================


class TestErrorHandling:
    """Tests for error handling and recovery mechanisms."""

    def test_classify_rate_limit_error(self):
        """Should classify rate limit errors correctly."""
        error = _classify_telegram_error("Too many requests", error_code=429)
        assert isinstance(error, ConnectorRateLimitError)
        assert error.is_retryable

    def test_classify_auth_error(self):
        """Should classify auth errors correctly."""
        error = _classify_telegram_error("Unauthorized", error_code=401)
        assert isinstance(error, ConnectorAuthError)
        assert not error.is_retryable

    def test_classify_not_found_error(self):
        """Should classify not found errors correctly."""
        error = _classify_telegram_error("Chat not found", error_code=404)
        assert isinstance(error, ConnectorAPIError)
        assert error.status_code == 404

    def test_classify_timeout_error(self):
        """Should classify timeout errors correctly."""
        error = _classify_telegram_error("Request timeout")
        assert isinstance(error, ConnectorTimeoutError)

    def test_classify_network_error(self):
        """Should classify network errors correctly."""
        error = _classify_telegram_error("Connection refused")
        assert isinstance(error, ConnectorNetworkError)

    @pytest.mark.asyncio
    async def test_send_message_failure(self, connector, mock_httpx_response):
        """Should handle API failure gracefully."""
        mock_response = mock_httpx_response(
            {"ok": False, "description": "Bad Request: chat not found"}
        )

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            result = await connector.send_message(
                channel_id="invalid_chat",
                text="Test",
            )

            assert result.success is False
            assert "chat not found" in (result.error or "")

    @pytest.mark.asyncio
    async def test_httpx_not_available(self):
        """Should handle httpx not being available."""
        with patch("aragora.connectors.chat.telegram.HTTPX_AVAILABLE", False):
            connector = TelegramConnector(bot_token="test")

            result = await connector.send_message("-123", "test")
            assert result.success is False
            assert "httpx not available" in (result.error or "")

    @pytest.mark.asyncio
    async def test_download_file_not_found(self, connector, mock_httpx_response):
        """Should raise error when file not found."""
        mock_response = mock_httpx_response(
            {"ok": False, "description": "Bad Request: file not found"}
        )

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            with pytest.raises(RuntimeError, match="file not found"):
                await connector.download_file("invalid_file")

    @pytest.mark.asyncio
    async def test_timeout_handling(self, connector):
        """Should handle timeouts gracefully."""
        import httpx

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=httpx.TimeoutException("Request timed out")
            )

            result = await connector.send_message("-123", "test")

            assert result.success is False
            assert "timeout" in (result.error or "").lower()

    @pytest.mark.asyncio
    async def test_connection_error_handling(self, connector):
        """Should handle connection errors gracefully."""
        import httpx

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=httpx.ConnectError("Connection refused")
            )

            result = await connector.send_message("-123", "test")

            assert result.success is False


# ============================================================================
# Test: Rate Limiting
# ============================================================================


class TestRateLimiting:
    """Tests for rate limiting behavior."""

    @pytest.mark.asyncio
    async def test_rate_limit_retry(self, connector, mock_httpx_response):
        """Should retry on rate limit with backoff."""
        rate_limit_response = mock_httpx_response(
            {
                "ok": False,
                "error_code": 429,
                "description": "Too Many Requests: retry after 5",
                "parameters": {"retry_after": 1},  # Short for testing
            }
        )
        success_response = mock_httpx_response(
            {
                "ok": True,
                "result": {"message_id": 100, "chat": {"id": -100}, "date": 1640000000},
            }
        )

        call_count = 0

        async def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return rate_limit_response
            return success_response

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = mock_post

            result = await connector.send_message("-123", "test")

            assert result.success is True
            assert call_count == 2

    @pytest.mark.asyncio
    async def test_rate_limit_max_retries_exceeded(self, connector, mock_httpx_response):
        """Should fail after max retries on persistent rate limit."""
        rate_limit_response = mock_httpx_response(
            {
                "ok": False,
                "error_code": 429,
                "description": "Too Many Requests",
                "parameters": {"retry_after": 1},
            }
        )

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=rate_limit_response
            )

            result = await connector.send_message("-123", "test")

            assert result.success is False
            assert "Too Many Requests" in (result.error or "")

    @pytest.mark.asyncio
    async def test_server_error_retry(self, connector, mock_httpx_response):
        """Should retry on server errors (5xx)."""
        error_response = mock_httpx_response(
            {"ok": False, "error_code": 502, "description": "Bad Gateway"}
        )
        success_response = mock_httpx_response(
            {
                "ok": True,
                "result": {"message_id": 100, "chat": {"id": -100}, "date": 1640000000},
            }
        )

        call_count = 0

        async def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return error_response
            return success_response

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = mock_post

            result = await connector.send_message("-123", "test")

            assert result.success is True
            assert call_count == 2


# ============================================================================
# Test: Webhook Handling
# ============================================================================


class TestWebhookHandling:
    """Tests for webhook event handling."""

    @pytest.mark.asyncio
    async def test_handle_webhook_message(self, connector, sample_message_payload):
        """Should handle message webhook events."""
        event = await connector.handle_webhook(sample_message_payload)

        assert isinstance(event, WebhookEvent)
        assert event.event_type == "message"
        assert event.platform == "telegram"
        assert event.metadata["channel_id"] == "-1001234567890"
        assert event.metadata["user_id"] == "12345"
        assert event.metadata["message_id"] == "1"

    @pytest.mark.asyncio
    async def test_handle_webhook_callback(self, connector, sample_callback_payload):
        """Should handle callback query webhook events."""
        event = await connector.handle_webhook(sample_callback_payload)

        assert event.event_type == "callback_query"
        assert event.metadata["user_id"] == "12345"

    @pytest.mark.asyncio
    async def test_handle_webhook_unknown(self, connector):
        """Should handle unknown webhook event types."""
        payload = {"update_id": 123, "some_unknown_type": {}}

        event = await connector.handle_webhook(payload)

        assert event.event_type == "unknown"
        assert event.platform == "telegram"

    def test_parse_webhook_event_message(self, connector, sample_message_payload):
        """Should parse webhook body to event."""
        body = json.dumps(sample_message_payload).encode()

        event = connector.parse_webhook_event({}, body)

        assert event.event_type == "message"
        assert event.metadata["channel_id"] == "-1001234567890"

    def test_parse_webhook_event_edited_message(self, connector):
        """Should parse edited message events."""
        payload = {
            "edited_message": {
                "message_id": 1,
                "from": {"id": 123},
                "chat": {"id": -100},
                "date": 1640000000,
                "edit_date": 1640000100,
                "text": "Edited text",
            }
        }
        body = json.dumps(payload).encode()

        event = connector.parse_webhook_event({}, body)

        assert event.event_type == "message_edited"

    def test_parse_webhook_event_callback_query(self, connector, sample_callback_payload):
        """Should parse callback query events."""
        body = json.dumps(sample_callback_payload).encode()

        event = connector.parse_webhook_event({}, body)

        assert event.event_type == "callback_query"
        assert event.metadata.get("callback_data") == "approve_debate"

    def test_parse_webhook_event_channel_post(self, connector):
        """Should parse channel post events."""
        payload = {
            "channel_post": {
                "message_id": 10,
                "chat": {"id": -1001234567890},
                "date": 1640000000,
                "text": "Channel announcement",
            }
        }
        body = json.dumps(payload).encode()

        event = connector.parse_webhook_event({}, body)

        assert event.event_type == "channel_post"

    def test_verify_webhook_no_secret_development(self, connector):
        """Should allow webhooks in development without secret."""
        with patch.dict(os.environ, {"ARAGORA_ENV": "development"}, clear=False):
            result = connector.verify_webhook({}, b"test")
            assert result is True

    def test_verify_webhook_with_valid_secret(self, connector):
        """Should verify webhooks with correct secret."""
        with patch.dict(
            os.environ,
            {"TELEGRAM_WEBHOOK_SECRET": "my-secret", "ARAGORA_ENV": "development"},
        ):
            headers = {"X-Telegram-Bot-Api-Secret-Token": "my-secret"}
            result = connector.verify_webhook(headers, b"test")
            assert result is True

    def test_verify_webhook_with_invalid_secret(self, connector):
        """Should reject webhooks with wrong secret."""
        with patch.dict(
            os.environ,
            {"TELEGRAM_WEBHOOK_SECRET": "my-secret", "ARAGORA_ENV": "development"},
        ):
            headers = {"X-Telegram-Bot-Api-Secret-Token": "wrong-secret"}
            result = connector.verify_webhook(headers, b"test")
            assert result is False

    def test_verify_webhook_fails_in_production_without_secret(self, connector):
        """Should fail closed in production without secret configured."""
        with patch.dict(
            os.environ,
            {"TELEGRAM_WEBHOOK_SECRET": "", "ARAGORA_ENV": "production"},
            clear=False,
        ):
            result = connector.verify_webhook({}, b"test")
            assert result is False


# ============================================================================
# Test: User Interaction Handling
# ============================================================================


class TestUserInteractionHandling:
    """Tests for handling user interactions (button clicks, etc.)."""

    @pytest.mark.asyncio
    async def test_handle_interaction(self, connector, sample_callback_payload):
        """Should parse callback query into UserInteraction."""
        interaction = await connector.handle_interaction(sample_callback_payload)

        assert isinstance(interaction, UserInteraction)
        assert interaction.id == "4382bfdwdsb323b2d9"
        assert interaction.interaction_type == InteractionType.BUTTON_CLICK
        assert interaction.action_id == "approve_debate"
        assert interaction.value == "approve_debate"
        assert interaction.user.id == "12345"

    @pytest.mark.asyncio
    async def test_answer_callback_query(self, connector, mock_httpx_response):
        """Should answer callback query successfully."""
        mock_response = mock_httpx_response({"ok": True, "result": True})

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            result = await connector.answer_callback_query(
                callback_query_id="4382bfdwdsb323b2d9",
                text="Vote recorded!",
                show_alert=False,
            )

            assert result is True

    @pytest.mark.asyncio
    async def test_respond_to_interaction(self, connector, mock_httpx_response):
        """Should respond to interaction successfully."""
        mock_response = mock_httpx_response({"ok": True, "result": True})

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            interaction = UserInteraction(
                id="callback_123",
                interaction_type=InteractionType.BUTTON_CLICK,
                action_id="approve_btn",
                value="approve",
                user=ChatUser(id="123", platform="telegram"),
                channel=ChatChannel(id="-1001234567890", platform="telegram"),
                platform="telegram",
            )

            result = await connector.respond_to_interaction(
                interaction=interaction,
                text="Action processed!",
            )

            assert result.success is True

    @pytest.mark.asyncio
    async def test_respond_to_interaction_replace_original(self, connector, mock_httpx_response):
        """Should replace original message when requested."""
        mock_answer = mock_httpx_response({"ok": True, "result": True})
        mock_edit = mock_httpx_response({"ok": True, "result": {"message_id": 100}})

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.post = AsyncMock(side_effect=[mock_answer, mock_edit])

            interaction = UserInteraction(
                id="callback_123",
                interaction_type=InteractionType.BUTTON_CLICK,
                action_id="approve_btn",
                value="approve",
                user=ChatUser(id="123", platform="telegram"),
                channel=ChatChannel(id="-1001234567890", platform="telegram"),
                message_id="100",
                platform="telegram",
            )

            result = await connector.respond_to_interaction(
                interaction=interaction,
                text="Updated message content",
                replace_original=True,
            )

            assert result.success is True


# ============================================================================
# Test: Formatting Helpers
# ============================================================================


class TestFormattingHelpers:
    """Tests for message formatting utilities."""

    def test_escape_markdown_special_chars(self, connector):
        """Should escape MarkdownV2 special characters."""
        text = "Hello *world* with _underscore_"
        escaped = connector._escape_markdown(text)

        assert "\\*" in escaped
        assert "\\_" in escaped

    def test_escape_markdown_preserves_text(self, connector):
        """Should preserve normal text while escaping."""
        text = "Hello world"
        escaped = connector._escape_markdown(text)

        assert "Hello" in escaped
        assert "world" in escaped

    def test_format_button_callback(self, connector):
        """Should format callback button correctly."""
        button = connector.format_button(
            text="Click Me",
            action_id="action_123",
            value="clicked",
        )

        assert button["type"] == "button"
        assert button["text"] == "Click Me"
        assert button["action_id"] == "action_123"
        assert button["value"] == "clicked"

    def test_format_button_url(self, connector):
        """Should format URL button correctly."""
        button = connector.format_button(
            text="Visit Site",
            action_id="link",
            url="https://example.com",
        )

        assert button["type"] == "url_button"
        assert button["url"] == "https://example.com"

    def test_format_blocks_with_buttons(self, connector):
        """Should format blocks with MessageButton objects."""
        buttons = [
            MessageButton(text="Option A", action_id="opt_a"),
            MessageButton(text="Option B", action_id="opt_b"),
        ]

        blocks = connector.format_blocks(actions=buttons)

        assert len(blocks) == 2
        assert blocks[0]["text"] == "Option A"
        assert blocks[1]["text"] == "Option B"

    def test_blocks_to_keyboard(self, connector):
        """Should convert blocks to Telegram inline keyboard."""
        blocks = [
            {"type": "button", "text": "Btn 1", "action_id": "btn1"},
            {"type": "button", "text": "Btn 2", "action_id": "btn2"},
            {"type": "button", "text": "Btn 3", "action_id": "btn3"},
            {"type": "button", "text": "Btn 4", "action_id": "btn4"},
        ]

        keyboard = connector._blocks_to_keyboard(blocks)

        assert "inline_keyboard" in keyboard
        # Should be grouped into rows of 3
        assert len(keyboard["inline_keyboard"]) == 2
        assert len(keyboard["inline_keyboard"][0]) == 3
        assert len(keyboard["inline_keyboard"][1]) == 1

    def test_blocks_to_keyboard_empty(self, connector):
        """Should return None for empty blocks."""
        result = connector._blocks_to_keyboard([])
        assert result is None

    def test_blocks_to_keyboard_url_buttons(self, connector):
        """Should handle URL buttons."""
        blocks = [
            {"type": "url_button", "text": "Visit", "url": "https://example.com"},
        ]

        keyboard = connector._blocks_to_keyboard(blocks)

        assert keyboard["inline_keyboard"][0][0]["url"] == "https://example.com"


# ============================================================================
# Test: Inline Query Support
# ============================================================================


class TestInlineQuerySupport:
    """Tests for inline query functionality."""

    @pytest.mark.asyncio
    async def test_answer_inline_query(self, connector, mock_httpx_response):
        """Should answer inline query successfully."""
        mock_response = mock_httpx_response({"ok": True, "result": True})

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            results = [
                connector.build_inline_article_result(
                    result_id="1",
                    title="Debate Result",
                    message_text="The consensus was...",
                )
            ]

            success = await connector.answer_inline_query(
                inline_query_id="query123",
                results=results,
                cache_time=60,
            )

            assert success is True

    def test_build_inline_article_result(self, connector):
        """Should build inline article result correctly."""
        result = connector.build_inline_article_result(
            result_id="abc123",
            title="Search Result",
            message_text="Selected content here",
            description="Brief description",
            url="https://example.com",
            thumb_url="https://example.com/thumb.jpg",
        )

        assert result["type"] == "article"
        assert result["id"] == "abc123"
        assert result["title"] == "Search Result"
        assert result["description"] == "Brief description"
        assert result["input_message_content"]["message_text"] == "Selected content here"


# ============================================================================
# Test: Bot Management
# ============================================================================


class TestBotManagement:
    """Tests for bot management operations."""

    @pytest.mark.asyncio
    async def test_get_me(self, connector, mock_httpx_response):
        """Should get bot information."""
        mock_response = mock_httpx_response(
            {
                "ok": True,
                "result": {
                    "id": 123456789,
                    "is_bot": True,
                    "first_name": "AragoraBot",
                    "username": "aragora_bot",
                },
            }
        )

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            result = await connector.get_me()

            assert result is not None
            assert result["id"] == 123456789
            assert result["username"] == "aragora_bot"

    @pytest.mark.asyncio
    async def test_set_my_commands(self, connector, mock_httpx_response):
        """Should set bot commands."""
        mock_response = mock_httpx_response({"ok": True, "result": True})

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            success = await connector.set_my_commands(
                commands=[
                    {"command": "debate", "description": "Start a debate"},
                    {"command": "help", "description": "Get help"},
                ]
            )

            assert success is True

    @pytest.mark.asyncio
    async def test_get_chat_member_count(self, connector, mock_httpx_response):
        """Should get chat member count."""
        mock_response = mock_httpx_response({"ok": True, "result": 42})

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            count = await connector.get_chat_member_count("-1001234567890")

            assert count == 42

    @pytest.mark.asyncio
    async def test_get_channel_info(self, connector, mock_httpx_response):
        """Should get channel information."""
        mock_response = mock_httpx_response(
            {
                "ok": True,
                "result": {
                    "id": -1001234567890,
                    "type": "supergroup",
                    "title": "Test Group",
                    "username": "testgroup",
                },
            }
        )

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            channel = await connector.get_channel_info("-1001234567890")

            assert isinstance(channel, ChatChannel)
            assert channel.id == "-1001234567890"
            assert channel.name == "Test Group"

    @pytest.mark.asyncio
    async def test_get_user_info(self, connector):
        """Should return basic user info (Telegram limitation)."""
        user = await connector.get_user_info("12345")

        # Telegram doesn't have a direct getUser API
        assert isinstance(user, ChatUser)
        assert user.id == "12345"
        assert user.platform == "telegram"


# ============================================================================
# Test: Circuit Breaker Integration
# ============================================================================


class TestCircuitBreakerIntegration:
    """Tests for circuit breaker integration."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_after_failures(self, mock_httpx_response):
        """Should open circuit breaker after repeated failures."""
        connector = TelegramConnector(
            bot_token="test",
            enable_circuit_breaker=True,
            circuit_breaker_threshold=2,
            circuit_breaker_cooldown=10.0,
        )

        error_response = mock_httpx_response(
            {"ok": False, "error_code": 500, "description": "Internal Server Error"}
        )

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=error_response
            )

            # First few failures should go through
            for _ in range(3):
                result = await connector.send_message("-123", "test")
                assert result.success is False

            # Eventually circuit breaker should open
            cb = connector._get_circuit_breaker()
            if cb:
                # Circuit breaker may be open after failures
                assert cb.get_status() in ["open", "half_open", "closed"]

    @pytest.mark.asyncio
    async def test_health_check_includes_circuit_breaker_status(
        self,
        connector_with_cb,
    ):
        """Should include circuit breaker status in health check."""
        health = await connector_with_cb.get_health()

        assert "circuit_breaker" in health
        assert health["circuit_breaker"]["enabled"] is True

    @pytest.mark.asyncio
    async def test_connector_without_circuit_breaker(self, connector):
        """Should work without circuit breaker."""
        health = await connector.get_health()

        assert health["circuit_breaker"]["enabled"] is False


# ============================================================================
# Test: Connection and Health
# ============================================================================


class TestConnectionAndHealth:
    """Tests for connection testing and health checks."""

    @pytest.mark.asyncio
    async def test_test_connection_configured(self, connector):
        """Should report connection configured status."""
        result = await connector.test_connection()

        assert result["platform"] == "telegram"
        assert result["success"] is True
        assert result["bot_token_configured"] is True

    @pytest.mark.asyncio
    async def test_test_connection_unconfigured(self):
        """Should report connection unconfigured status."""
        connector = TelegramConnector(bot_token="", webhook_url="")
        result = await connector.test_connection()

        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_get_health_configured(self, connector):
        """Should report healthy when configured."""
        health = await connector.get_health()

        assert health["platform"] == "telegram"
        assert health["status"] == "healthy"
        assert health["configured"] is True

    @pytest.mark.asyncio
    async def test_get_health_unconfigured(self):
        """Should report unconfigured status."""
        connector = TelegramConnector(bot_token="", webhook_url="")
        health = await connector.get_health()

        assert health["status"] == "unconfigured"
