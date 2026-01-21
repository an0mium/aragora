"""Tests for WhatsAppConnector."""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
import json

from aragora.connectors.chat.whatsapp import WhatsAppConnector
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
    VoiceMessage,
)


@pytest.fixture
def connector():
    """Create a WhatsAppConnector for testing."""
    return WhatsAppConnector(
        access_token="test-access-token",
        phone_number_id="123456789",
        business_account_id="987654321",
        verify_token="test-verify-token",
        app_secret="test-app-secret",
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


class TestWhatsAppConnectorInit:
    """Test connector initialization."""

    def test_init_with_tokens(self):
        """Test initialization with tokens."""
        connector = WhatsAppConnector(
            access_token="test-token",
            phone_number_id="12345",
            business_account_id="67890",
        )
        assert connector.bot_token == "test-token"
        assert connector.phone_number_id == "12345"
        assert connector.business_account_id == "67890"
        assert connector.platform_name == "whatsapp"
        assert connector.platform_display_name == "WhatsApp"

    def test_init_with_verify_token(self):
        """Test initialization with verify token."""
        connector = WhatsAppConnector(
            access_token="test-token",
            verify_token="verify-123",
        )
        assert connector.verify_token == "verify-123"


class TestSendMessage:
    """Test send_message method."""

    @pytest.mark.asyncio
    async def test_send_message_success(self, connector, mock_httpx_response):
        """Test successful message send."""
        mock_response = mock_httpx_response(
            {
                "messages": [{"id": "wamid.123456789"}],
            }
        )

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            result = await connector.send_message(
                channel_id="+1234567890",
                text="Hello, world!",
            )

            assert result.success is True
            assert result.message_id == "wamid.123456789"
            assert result.channel_id == "+1234567890"

    @pytest.mark.asyncio
    async def test_send_message_failure(self, connector, mock_httpx_response):
        """Test failed message send."""
        mock_response = mock_httpx_response(
            {
                "error": {
                    "message": "Invalid phone number",
                    "code": 400,
                }
            }
        )

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            result = await connector.send_message(
                channel_id="invalid",
                text="Hello",
            )

            assert result.success is False
            assert "Invalid phone number" in result.error

    @pytest.mark.asyncio
    async def test_send_message_with_reply(self, connector, mock_httpx_response):
        """Test sending message as reply (with context)."""
        mock_response = mock_httpx_response(
            {
                "messages": [{"id": "wamid.reply_456"}],
            }
        )

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.post = AsyncMock(return_value=mock_response)

            result = await connector.send_message(
                channel_id="+1234567890",
                text="Reply message",
                thread_id="wamid.original_123",
            )

            assert result.success is True
            # Verify context was included
            call_args = mock_instance.post.call_args
            payload = call_args[1]["json"]
            assert payload["context"]["message_id"] == "wamid.original_123"

    @pytest.mark.asyncio
    async def test_send_message_with_buttons(self, connector, mock_httpx_response):
        """Test sending interactive message with buttons."""
        mock_response = mock_httpx_response(
            {
                "messages": [{"id": "wamid.interactive_789"}],
            }
        )

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.post = AsyncMock(return_value=mock_response)

            blocks = [
                {"type": "button", "text": "Yes", "action_id": "btn_yes"},
                {"type": "button", "text": "No", "action_id": "btn_no"},
            ]

            result = await connector.send_message(
                channel_id="+1234567890",
                text="Choose an option",
                blocks=blocks,
            )

            assert result.success is True
            # Verify interactive type was set
            call_args = mock_instance.post.call_args
            payload = call_args[1]["json"]
            assert payload["type"] == "interactive"


class TestUpdateMessage:
    """Test update_message method."""

    @pytest.mark.asyncio
    async def test_update_message_sends_new_message(self, connector, mock_httpx_response):
        """Test that update sends new message (WhatsApp doesn't support editing)."""
        mock_response = mock_httpx_response(
            {
                "messages": [{"id": "wamid.new_123"}],
            }
        )

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.post = AsyncMock(return_value=mock_response)

            result = await connector.update_message(
                channel_id="+1234567890",
                message_id="wamid.original_123",
                text="Updated text",
            )

            assert result.success is True
            # Should be sent as reply
            call_args = mock_instance.post.call_args
            payload = call_args[1]["json"]
            assert payload["context"]["message_id"] == "wamid.original_123"


class TestDeleteMessage:
    """Test delete_message method."""

    @pytest.mark.asyncio
    async def test_delete_message_not_supported(self, connector):
        """Test that delete returns False (not supported by WhatsApp API)."""
        result = await connector.delete_message(
            channel_id="+1234567890",
            message_id="wamid.123",
        )

        assert result is False


class TestFileOperations:
    """Test file upload and download."""

    @pytest.mark.asyncio
    async def test_download_file_success(self, connector, mock_httpx_response):
        """Test successful file download."""
        mock_media_info_response = mock_httpx_response(
            {
                "url": "https://cdn.whatsapp.com/media/file123",
            }
        )
        mock_download_response = MagicMock()
        mock_download_response.content = b"file content here"
        mock_download_response.json.return_value = {"url": "https://cdn.whatsapp.com/media/file123"}

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.get = AsyncMock(
                side_effect=[mock_media_info_response, mock_download_response]
            )

            result = await connector.download_file(file_id="media_123")

            assert result == b"file content here"

    @pytest.mark.asyncio
    async def test_download_file_failure(self, connector, mock_httpx_response):
        """Test failed file download - media not found."""
        mock_response = mock_httpx_response(
            {
                "error": {
                    "message": "Media not found",
                }
            }
        )

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            with pytest.raises(RuntimeError, match="Media not found"):
                await connector.download_file(file_id="invalid_media")


class TestWebhookHandling:
    """Test webhook event handling."""

    @pytest.mark.asyncio
    async def test_handle_webhook_message(self, connector):
        """Test handling message webhook."""
        payload = {
            "entry": [
                {
                    "changes": [
                        {
                            "value": {
                                "messages": [
                                    {
                                        "id": "wamid.msg_123",
                                        "from": "+1234567890",
                                        "timestamp": "1640000000",
                                        "type": "text",
                                        "text": {"body": "Hello bot!"},
                                    }
                                ],
                                "contacts": [
                                    {
                                        "wa_id": "+1234567890",
                                        "profile": {"name": "Test User"},
                                    }
                                ],
                            }
                        }
                    ]
                }
            ]
        }

        result = await connector.handle_webhook(payload)

        assert isinstance(result, WebhookEvent)
        assert result.event_type == "message"
        assert result.platform == "whatsapp"
        assert result.metadata["channel_id"] == "+1234567890"
        assert result.metadata["message_id"] == "wamid.msg_123"

    @pytest.mark.asyncio
    async def test_handle_webhook_status_update(self, connector):
        """Test handling status update (delivered, read, etc.)."""
        payload = {
            "entry": [
                {
                    "changes": [
                        {
                            "value": {
                                "statuses": [
                                    {
                                        "id": "wamid.msg_123",
                                        "status": "delivered",
                                        "timestamp": "1640000000",
                                    }
                                ]
                            }
                        }
                    ]
                }
            ]
        }

        result = await connector.handle_webhook(payload)

        assert isinstance(result, WebhookEvent)
        assert result.event_type == "status_delivered"
        assert result.platform == "whatsapp"

    @pytest.mark.asyncio
    async def test_handle_webhook_unknown_event(self, connector):
        """Test handling unknown webhook event type."""
        payload = {"entry": [{"changes": [{"value": {}}]}]}

        result = await connector.handle_webhook(payload)

        assert result.event_type == "unknown"
        assert result.platform == "whatsapp"


class TestWebhookVerification:
    """Test webhook verification methods."""

    @pytest.mark.asyncio
    async def test_verify_webhook_success(self, connector):
        """Test successful webhook verification."""
        result = await connector.verify_webhook(
            mode="subscribe",
            token="test-verify-token",
            challenge="challenge_123",
        )

        assert result == "challenge_123"

    @pytest.mark.asyncio
    async def test_verify_webhook_wrong_token(self, connector):
        """Test webhook verification with wrong token."""
        result = await connector.verify_webhook(
            mode="subscribe",
            token="wrong-token",
            challenge="challenge_123",
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_verify_webhook_wrong_mode(self, connector):
        """Test webhook verification with wrong mode."""
        result = await connector.verify_webhook(
            mode="unsubscribe",
            token="test-verify-token",
            challenge="challenge_123",
        )

        assert result is None


class TestParseMessage:
    """Test parse_message method."""

    @pytest.mark.asyncio
    async def test_parse_text_message(self, connector):
        """Test parsing a text message."""
        payload = {
            "entry": [
                {
                    "changes": [
                        {
                            "value": {
                                "messages": [
                                    {
                                        "id": "wamid.msg_123",
                                        "from": "+1234567890",
                                        "timestamp": "1640000000",
                                        "type": "text",
                                        "text": {"body": "Hello!"},
                                    }
                                ],
                                "contacts": [
                                    {
                                        "wa_id": "+1234567890",
                                        "profile": {"name": "Test User"},
                                    }
                                ],
                            }
                        }
                    ]
                }
            ]
        }

        result = await connector.parse_message(payload)

        assert isinstance(result, ChatMessage)
        assert result.id == "wamid.msg_123"
        assert result.content == "Hello!"
        assert result.platform == "whatsapp"
        assert result.author.id == "+1234567890"

    @pytest.mark.asyncio
    async def test_parse_interactive_button_reply(self, connector):
        """Test parsing an interactive button reply."""
        payload = {
            "entry": [
                {
                    "changes": [
                        {
                            "value": {
                                "messages": [
                                    {
                                        "id": "wamid.msg_456",
                                        "from": "+1234567890",
                                        "timestamp": "1640000000",
                                        "type": "interactive",
                                        "interactive": {
                                            "type": "button_reply",
                                            "button_reply": {
                                                "id": "btn_yes",
                                                "title": "Yes",
                                            },
                                        },
                                    }
                                ],
                                "contacts": [
                                    {
                                        "wa_id": "+1234567890",
                                        "profile": {"name": "Test User"},
                                    }
                                ],
                            }
                        }
                    ]
                }
            ]
        }

        result = await connector.parse_message(payload)

        assert result.content == "Yes"


class TestParseCommand:
    """Test parse_command method."""

    @pytest.mark.asyncio
    async def test_parse_slash_command(self, connector):
        """Test parsing a /command."""
        payload = {
            "entry": [
                {
                    "changes": [
                        {
                            "value": {
                                "messages": [
                                    {
                                        "id": "wamid.cmd_123",
                                        "from": "+1234567890",
                                        "timestamp": "1640000000",
                                        "type": "text",
                                        "text": {"body": "/help argument1"},
                                    }
                                ],
                                "contacts": [
                                    {
                                        "wa_id": "+1234567890",
                                        "profile": {"name": "Test User"},
                                    }
                                ],
                            }
                        }
                    ]
                }
            ]
        }

        result = await connector.parse_command(payload)

        assert result is not None
        assert result.name == "help"
        assert result.args == ["argument1"]
        assert result.text == "/help argument1"

    @pytest.mark.asyncio
    async def test_parse_exclamation_command(self, connector):
        """Test parsing a !command."""
        payload = {
            "entry": [
                {
                    "changes": [
                        {
                            "value": {
                                "messages": [
                                    {
                                        "id": "wamid.cmd_456",
                                        "from": "+1234567890",
                                        "timestamp": "1640000000",
                                        "type": "text",
                                        "text": {"body": "!ping"},
                                    }
                                ],
                                "contacts": [
                                    {
                                        "wa_id": "+1234567890",
                                        "profile": {"name": "Test User"},
                                    }
                                ],
                            }
                        }
                    ]
                }
            ]
        }

        result = await connector.parse_command(payload)

        assert result is not None
        assert result.name == "ping"

    @pytest.mark.asyncio
    async def test_parse_no_command(self, connector):
        """Test that regular messages don't parse as commands."""
        payload = {
            "entry": [
                {
                    "changes": [
                        {
                            "value": {
                                "messages": [
                                    {
                                        "id": "wamid.msg_789",
                                        "from": "+1234567890",
                                        "timestamp": "1640000000",
                                        "type": "text",
                                        "text": {"body": "Just a regular message"},
                                    }
                                ],
                                "contacts": [
                                    {
                                        "wa_id": "+1234567890",
                                        "profile": {"name": "Test User"},
                                    }
                                ],
                            }
                        }
                    ]
                }
            ]
        }

        result = await connector.parse_command(payload)

        assert result is None


class TestHandleInteraction:
    """Test handle_interaction method."""

    @pytest.mark.asyncio
    async def test_handle_button_click(self, connector):
        """Test handling a button click interaction."""
        payload = {
            "entry": [
                {
                    "changes": [
                        {
                            "value": {
                                "messages": [
                                    {
                                        "id": "wamid.int_123",
                                        "from": "+1234567890",
                                        "timestamp": "1640000000",
                                        "type": "interactive",
                                        "interactive": {
                                            "type": "button_reply",
                                            "button_reply": {
                                                "id": "btn_confirm",
                                                "title": "Confirm",
                                            },
                                        },
                                    }
                                ],
                                "contacts": [
                                    {
                                        "wa_id": "+1234567890",
                                        "profile": {"name": "Test User"},
                                    }
                                ],
                            }
                        }
                    ]
                }
            ]
        }

        result = await connector.handle_interaction(payload)

        assert isinstance(result, UserInteraction)
        assert result.action_id == "btn_confirm"
        assert result.value == "Confirm"
        assert result.interaction_type == InteractionType.BUTTON_CLICK

    @pytest.mark.asyncio
    async def test_handle_list_selection(self, connector):
        """Test handling a list selection interaction."""
        payload = {
            "entry": [
                {
                    "changes": [
                        {
                            "value": {
                                "messages": [
                                    {
                                        "id": "wamid.int_456",
                                        "from": "+1234567890",
                                        "timestamp": "1640000000",
                                        "type": "interactive",
                                        "interactive": {
                                            "type": "list_reply",
                                            "list_reply": {
                                                "id": "option_1",
                                                "title": "Option One",
                                            },
                                        },
                                    }
                                ],
                                "contacts": [
                                    {
                                        "wa_id": "+1234567890",
                                        "profile": {"name": "Test User"},
                                    }
                                ],
                            }
                        }
                    ]
                }
            ]
        }

        result = await connector.handle_interaction(payload)

        assert result.action_id == "option_1"
        assert result.interaction_type == InteractionType.SELECT_MENU


class TestSendVoiceMessage:
    """Test send_voice_message method."""

    @pytest.mark.asyncio
    async def test_send_voice_message_success(self, connector, mock_httpx_response):
        """Test successful voice message send."""
        # Mock media upload response
        mock_upload_response = mock_httpx_response({"id": "media_voice_123"})
        # Mock message send response
        mock_send_response = mock_httpx_response(
            {"messages": [{"id": "wamid.voice_456"}]}
        )

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.post = AsyncMock(
                side_effect=[mock_upload_response, mock_send_response]
            )

            result = await connector.send_voice_message(
                channel_id="+1234567890",
                audio_data=b"audio data here",
            )

            assert result.success is True
            assert result.message_id == "wamid.voice_456"

    @pytest.mark.asyncio
    async def test_send_voice_message_upload_failure(self, connector, mock_httpx_response):
        """Test voice message send with upload failure."""
        mock_response = mock_httpx_response(
            {"error": {"message": "Audio upload failed"}}
        )

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.post = AsyncMock(return_value=mock_response)

            result = await connector.send_voice_message(
                channel_id="+1234567890",
                audio_data=b"audio data",
            )

            assert result.success is False
            assert "Audio upload failed" in result.error


class TestSendTemplate:
    """Test send_template method."""

    @pytest.mark.asyncio
    async def test_send_template_success(self, connector, mock_httpx_response):
        """Test successful template message send."""
        mock_response = mock_httpx_response(
            {"messages": [{"id": "wamid.template_123"}]}
        )

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.post = AsyncMock(return_value=mock_response)

            result = await connector.send_template(
                channel_id="+1234567890",
                template_name="hello_world",
                language_code="en_US",
            )

            assert result.success is True
            # Verify template payload
            call_args = mock_instance.post.call_args
            payload = call_args[1]["json"]
            assert payload["type"] == "template"
            assert payload["template"]["name"] == "hello_world"

    @pytest.mark.asyncio
    async def test_send_template_with_components(self, connector, mock_httpx_response):
        """Test template message with dynamic components."""
        mock_response = mock_httpx_response(
            {"messages": [{"id": "wamid.template_456"}]}
        )

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.post = AsyncMock(return_value=mock_response)

            components = [
                {
                    "type": "body",
                    "parameters": [
                        {"type": "text", "text": "John"},
                    ],
                }
            ]

            result = await connector.send_template(
                channel_id="+1234567890",
                template_name="greeting",
                components=components,
            )

            assert result.success is True
            call_args = mock_instance.post.call_args
            payload = call_args[1]["json"]
            assert payload["template"]["components"] == components


class TestChannelAndUserInfo:
    """Test get_channel_info and get_user_info methods."""

    @pytest.mark.asyncio
    async def test_get_channel_info(self, connector):
        """Test getting channel info."""
        result = await connector.get_channel_info("+1234567890")

        assert isinstance(result, ChatChannel)
        assert result.id == "+1234567890"
        assert result.platform == "whatsapp"
        assert result.is_dm is True

    @pytest.mark.asyncio
    async def test_get_user_info(self, connector):
        """Test getting user info."""
        result = await connector.get_user_info("+1234567890")

        assert isinstance(result, ChatUser)
        assert result.id == "+1234567890"
        assert result.platform == "whatsapp"


class TestPlatformProperties:
    """Test platform property methods."""

    def test_platform_name(self, connector):
        """Test platform_name property."""
        assert connector.platform_name == "whatsapp"

    def test_platform_display_name(self, connector):
        """Test platform_display_name property."""
        assert connector.platform_display_name == "WhatsApp"


class TestBuildInteractive:
    """Test _build_interactive helper method."""

    def test_build_interactive_buttons(self, connector):
        """Test building interactive payload with buttons."""
        blocks = [
            {"type": "button", "text": "Yes", "action_id": "btn_yes"},
            {"type": "button", "text": "No", "action_id": "btn_no"},
        ]

        result = connector._build_interactive("Choose:", blocks)

        assert result["type"] == "button"
        assert result["body"]["text"] == "Choose:"
        assert len(result["action"]["buttons"]) == 2

    def test_build_interactive_list(self, connector):
        """Test building interactive payload with list items."""
        blocks = [
            {"type": "list_item", "text": "Option 1", "action_id": "opt_1"},
            {"type": "list_item", "text": "Option 2", "action_id": "opt_2"},
        ]

        result = connector._build_interactive("Select:", blocks)

        assert result["type"] == "list"
        assert result["body"]["text"] == "Select:"
        rows = result["action"]["sections"][0]["rows"]
        assert len(rows) == 2

    def test_build_interactive_max_buttons(self, connector):
        """Test that buttons are limited to 3 max."""
        blocks = [
            {"type": "button", "text": f"Btn {i}", "action_id": f"btn_{i}"}
            for i in range(5)
        ]

        result = connector._build_interactive("Choose:", blocks)

        # WhatsApp limits to 3 buttons
        assert len(result["action"]["buttons"]) == 3


class TestHttpxRequirement:
    """Test httpx availability checks."""

    @pytest.mark.asyncio
    async def test_send_message_without_httpx(self):
        """Test that operations fail gracefully without httpx."""
        with patch("aragora.connectors.chat.whatsapp.HTTPX_AVAILABLE", False):
            connector = WhatsAppConnector(access_token="test")

            with pytest.raises(RuntimeError, match="httpx is required"):
                await connector.send_message("+123", "test")

    @pytest.mark.asyncio
    async def test_download_file_without_httpx(self):
        """Test download fails gracefully without httpx."""
        with patch("aragora.connectors.chat.whatsapp.HTTPX_AVAILABLE", False):
            connector = WhatsAppConnector(access_token="test")

            with pytest.raises(RuntimeError, match="httpx is required"):
                await connector.download_file("media_123")

    @pytest.mark.asyncio
    async def test_send_voice_without_httpx(self):
        """Test voice message fails gracefully without httpx."""
        with patch("aragora.connectors.chat.whatsapp.HTTPX_AVAILABLE", False):
            connector = WhatsAppConnector(access_token="test")

            with pytest.raises(RuntimeError, match="httpx is required"):
                await connector.send_voice_message("+123", b"audio")

    @pytest.mark.asyncio
    async def test_send_template_without_httpx(self):
        """Test template fails gracefully without httpx."""
        with patch("aragora.connectors.chat.whatsapp.HTTPX_AVAILABLE", False):
            connector = WhatsAppConnector(access_token="test")

            with pytest.raises(RuntimeError, match="httpx is required"):
                await connector.send_template("+123", "template_name")


class TestSignatureVerification:
    """Test webhook signature verification."""

    @pytest.mark.asyncio
    async def test_valid_signature(self, connector):
        """Test webhook processing with valid signature."""
        import hashlib
        import hmac

        payload = {
            "entry": [{"changes": [{"value": {"messages": []}}]}]
        }
        body = json.dumps(payload, separators=(",", ":"))
        signature = hmac.new(
            b"test-app-secret",
            body.encode(),
            hashlib.sha256,
        ).hexdigest()

        headers = {"x-hub-signature-256": f"sha256={signature}"}

        result = await connector.handle_webhook(payload, headers=headers)

        # Should process normally (not invalid_signature)
        assert result.event_type != "invalid_signature"

    @pytest.mark.asyncio
    async def test_invalid_signature(self, connector):
        """Test webhook processing with invalid signature."""
        payload = {
            "entry": [{"changes": [{"value": {"messages": []}}]}]
        }
        headers = {"x-hub-signature-256": "sha256=invalidsignature"}

        result = await connector.handle_webhook(payload, headers=headers)

        assert result.event_type == "invalid_signature"
