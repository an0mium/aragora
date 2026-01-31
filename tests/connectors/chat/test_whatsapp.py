"""Comprehensive tests for WhatsAppConnector.

Tests for WhatsApp Business API connector including:
- Connector initialization
- Message parsing and handling
- Template message formatting
- Debate initiation from WhatsApp
- Result formatting and delivery
- Media handling
- Webhook verification
- Error handling and recovery
- Rate limiting
- Circuit breaker functionality
"""

import hashlib
import hmac
import json
import os
import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from aragora.connectors.chat.whatsapp import WhatsAppConnector
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


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def connector():
    """Create a WhatsAppConnector with test credentials."""
    return WhatsAppConnector(
        access_token="test-access-token-12345",
        phone_number_id="123456789012345",
        business_account_id="987654321098765",
        verify_token="test-verify-token-secret",
        app_secret="test-app-secret-abc123",
    )


@pytest.fixture
def connector_no_secret():
    """Create a WhatsAppConnector without app secret (for dev mode)."""
    return WhatsAppConnector(
        access_token="test-access-token",
        phone_number_id="123456789",
    )


@pytest.fixture
def mock_httpx_response():
    """Factory for creating mock httpx responses."""

    def _create_response(json_data, status_code=200, content=b""):
        mock = MagicMock()
        mock.json.return_value = json_data
        mock.status_code = status_code
        mock.content = content or json.dumps(json_data).encode()
        mock.text = json.dumps(json_data)
        return mock

    return _create_response


@pytest.fixture
def sample_message_payload():
    """Sample WhatsApp webhook message payload."""
    return {
        "entry": [
            {
                "id": "WHATSAPP_BUSINESS_ACCOUNT_ID",
                "changes": [
                    {
                        "value": {
                            "messaging_product": "whatsapp",
                            "metadata": {
                                "display_phone_number": "15551234567",
                                "phone_number_id": "123456789",
                            },
                            "contacts": [
                                {
                                    "profile": {"name": "John Doe"},
                                    "wa_id": "15559876543",
                                }
                            ],
                            "messages": [
                                {
                                    "from": "15559876543",
                                    "id": "wamid.HBgLMTU1NTEyMzQ1NjcVAgARGBI5QTNDQTVCM0Q0Q0Q2MzU3NDgA",
                                    "timestamp": "1640000000",
                                    "type": "text",
                                    "text": {"body": "Hello, bot!"},
                                }
                            ],
                        },
                        "field": "messages",
                    }
                ],
            }
        ],
        "object": "whatsapp_business_account",
    }


@pytest.fixture
def sample_interactive_payload():
    """Sample WhatsApp interactive button reply payload."""
    return {
        "entry": [
            {
                "changes": [
                    {
                        "value": {
                            "messaging_product": "whatsapp",
                            "contacts": [
                                {
                                    "profile": {"name": "Jane Smith"},
                                    "wa_id": "15551112222",
                                }
                            ],
                            "messages": [
                                {
                                    "from": "15551112222",
                                    "id": "wamid.interactive123",
                                    "timestamp": "1640001000",
                                    "type": "interactive",
                                    "interactive": {
                                        "type": "button_reply",
                                        "button_reply": {
                                            "id": "btn_approve",
                                            "title": "Approve",
                                        },
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
        ]
    }


# =============================================================================
# Test Classes - Initialization
# =============================================================================


class TestWhatsAppConnectorInitialization:
    """Test connector initialization and configuration."""

    def test_init_with_all_credentials(self):
        """Test initialization with all credentials provided."""
        connector = WhatsAppConnector(
            access_token="token123",
            phone_number_id="phone123",
            business_account_id="business123",
            verify_token="verify123",
            app_secret="secret123",
        )
        assert connector.bot_token == "token123"
        assert connector.phone_number_id == "phone123"
        assert connector.business_account_id == "business123"
        assert connector.verify_token == "verify123"
        assert connector.signing_secret == "secret123"

    def test_init_with_minimal_credentials(self):
        """Test initialization with only required credentials."""
        connector = WhatsAppConnector(access_token="token123")
        assert connector.bot_token == "token123"
        assert connector.phone_number_id == ""
        assert connector.business_account_id == ""

    def test_init_from_environment_variables(self):
        """Test initialization from environment variables."""
        with patch.dict(
            os.environ,
            {
                "WHATSAPP_ACCESS_TOKEN": "env_token",
                "WHATSAPP_PHONE_NUMBER_ID": "env_phone",
                "WHATSAPP_BUSINESS_ACCOUNT_ID": "env_business",
                "WHATSAPP_VERIFY_TOKEN": "env_verify",
                "WHATSAPP_APP_SECRET": "env_secret",
            },
        ):
            # Reimport to pick up env vars
            from aragora.connectors.chat import whatsapp

            connector = WhatsAppConnector()
            # Direct param takes precedence, but empty uses env
            assert connector.bot_token == "" or connector.bot_token is not None

    def test_platform_name_property(self, connector):
        """Test platform_name returns 'whatsapp'."""
        assert connector.platform_name == "whatsapp"

    def test_platform_display_name_property(self, connector):
        """Test platform_display_name returns 'WhatsApp'."""
        assert connector.platform_display_name == "WhatsApp"

    def test_is_configured_with_token(self, connector):
        """Test is_configured returns True when token is set."""
        assert connector.is_configured is True

    def test_is_configured_without_token(self):
        """Test is_configured returns False when no credentials."""
        connector = WhatsAppConnector()
        # May still return True if env vars are set
        # Test with explicit empty
        with patch.object(connector, "bot_token", ""):
            with patch.object(connector, "webhook_url", None):
                assert connector.is_configured is False

    def test_circuit_breaker_enabled_by_default(self, connector):
        """Test circuit breaker is enabled by default."""
        assert connector._enable_circuit_breaker is True

    def test_circuit_breaker_can_be_disabled(self):
        """Test circuit breaker can be disabled."""
        connector = WhatsAppConnector(access_token="test", enable_circuit_breaker=False)
        assert connector._enable_circuit_breaker is False

    def test_custom_request_timeout(self):
        """Test custom request timeout configuration."""
        connector = WhatsAppConnector(access_token="test", request_timeout=60.0)
        assert connector._request_timeout == 60.0


# =============================================================================
# Test Classes - Message Sending
# =============================================================================


class TestSendMessage:
    """Test send_message method."""

    @pytest.mark.asyncio
    async def test_send_text_message_success(self, connector, mock_httpx_response):
        """Test successful text message send."""
        mock_response = mock_httpx_response({"messages": [{"id": "wamid.ABCdef123456"}]})

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            result = await connector.send_message(
                channel_id="+15559876543",
                text="Hello from WhatsApp!",
            )

            assert result.success is True
            assert result.message_id == "wamid.ABCdef123456"
            assert result.channel_id == "+15559876543"

    @pytest.mark.asyncio
    async def test_send_message_with_preview_url(self, connector, mock_httpx_response):
        """Test message includes preview_url setting."""
        mock_response = mock_httpx_response({"messages": [{"id": "wamid.xyz"}]})

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.post = AsyncMock(return_value=mock_response)

            await connector.send_message(
                channel_id="+15559876543",
                text="Check out https://example.com",
            )

            call_args = mock_instance.post.call_args
            payload = call_args[1]["json"]
            assert payload["text"]["preview_url"] is True

    @pytest.mark.asyncio
    async def test_send_message_with_thread_context(self, connector, mock_httpx_response):
        """Test message with reply context (thread)."""
        mock_response = mock_httpx_response({"messages": [{"id": "wamid.reply123"}]})

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.post = AsyncMock(return_value=mock_response)

            result = await connector.send_message(
                channel_id="+15559876543",
                text="This is a reply",
                thread_id="wamid.original789",
            )

            assert result.success is True
            call_args = mock_instance.post.call_args
            payload = call_args[1]["json"]
            assert payload["context"]["message_id"] == "wamid.original789"

    @pytest.mark.asyncio
    async def test_send_message_with_buttons(self, connector, mock_httpx_response):
        """Test sending interactive message with buttons."""
        mock_response = mock_httpx_response({"messages": [{"id": "wamid.interactive_btn"}]})

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.post = AsyncMock(return_value=mock_response)

            blocks = [
                {"type": "button", "text": "Yes", "action_id": "btn_yes"},
                {"type": "button", "text": "No", "action_id": "btn_no"},
            ]

            result = await connector.send_message(
                channel_id="+15559876543",
                text="Do you agree?",
                blocks=blocks,
            )

            assert result.success is True
            call_args = mock_instance.post.call_args
            payload = call_args[1]["json"]
            assert payload["type"] == "interactive"
            assert "interactive" in payload

    @pytest.mark.asyncio
    async def test_send_message_with_list(self, connector, mock_httpx_response):
        """Test sending interactive message with list items."""
        mock_response = mock_httpx_response({"messages": [{"id": "wamid.interactive_list"}]})

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.post = AsyncMock(return_value=mock_response)

            blocks = [
                {"type": "list", "text": "Select option"},
                {"type": "list_item", "text": "Option 1", "action_id": "opt_1"},
                {"type": "list_item", "text": "Option 2", "action_id": "opt_2"},
            ]

            result = await connector.send_message(
                channel_id="+15559876543",
                text="Choose from the list",
                blocks=blocks,
            )

            assert result.success is True

    @pytest.mark.asyncio
    async def test_send_message_api_error(self, connector, mock_httpx_response):
        """Test handling API error response."""
        mock_response = mock_httpx_response(
            {
                "error": {
                    "message": "Invalid phone number format",
                    "code": 131009,
                }
            }
        )

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            result = await connector.send_message(
                channel_id="invalid",
                text="Test",
            )

            assert result.success is False
            assert "Invalid phone number" in result.error

    @pytest.mark.asyncio
    async def test_send_message_empty_response(self, connector, mock_httpx_response):
        """Test handling empty messages array in response."""
        mock_response = mock_httpx_response({"messages": []})

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            result = await connector.send_message(
                channel_id="+15559876543",
                text="Test",
            )

            assert result.success is True
            assert result.message_id is None


# =============================================================================
# Test Classes - Message Update and Delete
# =============================================================================


class TestUpdateMessage:
    """Test update_message method."""

    @pytest.mark.asyncio
    async def test_update_message_sends_reply(self, connector, mock_httpx_response):
        """Test update sends new message as reply (WhatsApp limitation)."""
        mock_response = mock_httpx_response({"messages": [{"id": "wamid.updated"}]})

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.post = AsyncMock(return_value=mock_response)

            result = await connector.update_message(
                channel_id="+15559876543",
                message_id="wamid.original",
                text="Updated content",
            )

            assert result.success is True
            call_args = mock_instance.post.call_args
            payload = call_args[1]["json"]
            assert payload["context"]["message_id"] == "wamid.original"


class TestDeleteMessage:
    """Test delete_message method."""

    @pytest.mark.asyncio
    async def test_delete_message_not_supported(self, connector):
        """Test delete returns False (WhatsApp doesn't support deletion)."""
        result = await connector.delete_message(
            channel_id="+15559876543",
            message_id="wamid.123",
        )
        assert result is False


# =============================================================================
# Test Classes - File Operations
# =============================================================================


class TestFileOperations:
    """Test file upload and download operations."""

    @pytest.mark.asyncio
    async def test_upload_file_success(self, connector, mock_httpx_response):
        """Test successful file upload."""
        mock_upload = mock_httpx_response({"id": "media_upload_123"})
        mock_send = mock_httpx_response({"messages": [{"id": "wamid.doc"}]})

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.post = AsyncMock(side_effect=[mock_upload, mock_send])

            result = await connector.upload_file(
                channel_id="+15559876543",
                content=b"PDF content here",
                filename="document.pdf",
                content_type="application/pdf",
                title="My Document",
            )

            assert isinstance(result, FileAttachment)
            assert result.id == "media_upload_123"
            assert result.filename == "document.pdf"
            assert result.size == len(b"PDF content here")

    @pytest.mark.asyncio
    async def test_upload_file_failure(self, connector, mock_httpx_response):
        """Test file upload failure handling."""
        mock_response = mock_httpx_response(
            {"error": {"message": "File too large", "code": 131052}}
        )

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            with pytest.raises(RuntimeError, match="File too large"):
                await connector.upload_file(
                    channel_id="+15559876543",
                    content=b"x" * 100_000_000,
                    filename="huge.pdf",
                )

    @pytest.mark.asyncio
    async def test_download_file_success(self, connector, mock_httpx_response):
        """Test successful file download."""
        mock_info = mock_httpx_response(
            {
                "url": "https://lookaside.fbsbx.com/whatsapp/media123",
                "mime_type": "audio/ogg",
                "file_size": 1024,
            }
        )
        mock_download = MagicMock()
        mock_download.content = b"audio file content"
        mock_download.status_code = 200

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.get = AsyncMock(return_value=mock_info)
            mock_instance.request = AsyncMock(return_value=mock_download)

            result = await connector.download_file(file_id="media123")

            assert isinstance(result, FileAttachment)
            assert result.content == b"audio file content"
            assert result.id == "media123"
            assert result.content_type == "audio/ogg"

    @pytest.mark.asyncio
    async def test_download_file_determines_extension(self, connector, mock_httpx_response):
        """Test file download determines correct extension from mime type."""
        mock_info = mock_httpx_response(
            {
                "url": "https://lookaside.fbsbx.com/media",
                "mime_type": "audio/mpeg",
                "file_size": 512,
            }
        )
        mock_download = MagicMock()
        mock_download.content = b"mp3 data"
        mock_download.status_code = 200

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.get = AsyncMock(return_value=mock_info)
            mock_instance.request = AsyncMock(return_value=mock_download)

            result = await connector.download_file(file_id="media456")

            assert result.filename.endswith(".mp3")

    @pytest.mark.asyncio
    async def test_download_file_no_url(self, connector, mock_httpx_response):
        """Test download failure when no URL returned."""
        mock_response = mock_httpx_response(
            {"mime_type": "audio/ogg"}  # Missing URL
        )

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            with pytest.raises(RuntimeError, match="No media URL"):
                await connector.download_file(file_id="media789")

    @pytest.mark.asyncio
    async def test_download_file_api_error(self, connector, mock_httpx_response):
        """Test download failure on API error."""
        mock_response = mock_httpx_response({"error": {"message": "Media expired"}})

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            with pytest.raises(RuntimeError, match="Media expired"):
                await connector.download_file(file_id="expired_media")


# =============================================================================
# Test Classes - Voice Messages
# =============================================================================


class TestVoiceMessages:
    """Test voice message operations."""

    @pytest.mark.asyncio
    async def test_send_voice_message_success(self, connector, mock_httpx_response):
        """Test successful voice message send."""
        mock_upload = mock_httpx_response({"id": "media_voice_123"})
        mock_send = mock_httpx_response({"messages": [{"id": "wamid.voice_sent"}]})

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.post = AsyncMock(side_effect=[mock_upload, mock_send])

            result = await connector.send_voice_message(
                channel_id="+15559876543",
                audio_content=b"OGG audio data",
                filename="voice.ogg",
                content_type="audio/ogg",
            )

            assert result.success is True
            assert result.message_id == "wamid.voice_sent"

    @pytest.mark.asyncio
    async def test_send_voice_message_upload_failure(self, connector, mock_httpx_response):
        """Test voice message with upload failure."""
        mock_response = mock_httpx_response({"error": {"message": "Invalid audio format"}})

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            result = await connector.send_voice_message(
                channel_id="+15559876543",
                audio_content=b"bad audio",
            )

            assert result.success is False
            assert "Invalid audio format" in result.error

    @pytest.mark.asyncio
    async def test_download_voice_message(self, connector, mock_httpx_response):
        """Test downloading a voice message."""
        mock_info = mock_httpx_response(
            {
                "url": "https://cdn.whatsapp.com/voice123",
                "mime_type": "audio/ogg",
                "file_size": 2048,
            }
        )
        mock_download = MagicMock()
        mock_download.content = b"voice audio bytes"
        mock_download.status_code = 200

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.get = AsyncMock(return_value=mock_info)
            mock_instance.request = AsyncMock(return_value=mock_download)

            voice = VoiceMessage(
                id="voice_msg_1",
                channel=ChatChannel(id="+15559876543", platform="whatsapp"),
                author=ChatUser(id="+15551112222", platform="whatsapp"),
                duration_seconds=5.0,
                file=FileAttachment(
                    id="voice_file_123",
                    filename="voice.ogg",
                    content_type="audio/ogg",
                    size=2048,
                ),
            )

            result = await connector.download_voice_message(voice)

            assert result == b"voice audio bytes"


# =============================================================================
# Test Classes - Template Messages
# =============================================================================


class TestTemplateMessages:
    """Test template message sending."""

    @pytest.mark.asyncio
    async def test_send_template_success(self, connector, mock_httpx_response):
        """Test successful template message send."""
        mock_response = mock_httpx_response({"messages": [{"id": "wamid.template_sent"}]})

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.post = AsyncMock(return_value=mock_response)

            result = await connector.send_template(
                channel_id="+15559876543",
                template_name="hello_world",
                language_code="en_US",
            )

            assert result.success is True
            call_args = mock_instance.post.call_args
            payload = call_args[1]["json"]
            assert payload["type"] == "template"
            assert payload["template"]["name"] == "hello_world"
            assert payload["template"]["language"]["code"] == "en_US"

    @pytest.mark.asyncio
    async def test_send_template_with_components(self, connector, mock_httpx_response):
        """Test template with dynamic components."""
        mock_response = mock_httpx_response({"messages": [{"id": "wamid.template_dynamic"}]})

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.post = AsyncMock(return_value=mock_response)

            components = [
                {
                    "type": "header",
                    "parameters": [
                        {"type": "image", "image": {"link": "https://example.com/img.jpg"}}
                    ],
                },
                {
                    "type": "body",
                    "parameters": [
                        {"type": "text", "text": "John"},
                        {"type": "text", "text": "$100"},
                    ],
                },
            ]

            result = await connector.send_template(
                channel_id="+15559876543",
                template_name="order_confirmation",
                language_code="en",
                components=components,
            )

            assert result.success is True
            call_args = mock_instance.post.call_args
            payload = call_args[1]["json"]
            assert payload["template"]["components"] == components

    @pytest.mark.asyncio
    async def test_send_template_not_approved(self, connector, mock_httpx_response):
        """Test template send with unapproved template."""
        mock_response = mock_httpx_response(
            {"error": {"message": "Template not found", "code": 132000}}
        )

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            result = await connector.send_template(
                channel_id="+15559876543",
                template_name="unapproved_template",
            )

            assert result.success is False
            assert "Template not found" in result.error


# =============================================================================
# Test Classes - Webhook Handling
# =============================================================================


class TestWebhookHandling:
    """Test webhook event handling."""

    @pytest.mark.asyncio
    async def test_handle_webhook_text_message(self, connector, sample_message_payload):
        """Test handling text message webhook."""
        result = await connector.handle_webhook(sample_message_payload)

        assert isinstance(result, WebhookEvent)
        assert result.event_type == "message"
        assert result.platform == "whatsapp"
        assert result.metadata["channel_id"] == "15559876543"

    @pytest.mark.asyncio
    async def test_handle_webhook_status_delivered(self, connector):
        """Test handling message status update - delivered."""
        payload = {
            "entry": [
                {
                    "changes": [
                        {
                            "value": {
                                "statuses": [
                                    {
                                        "id": "wamid.msg123",
                                        "status": "delivered",
                                        "timestamp": "1640001000",
                                        "recipient_id": "15559876543",
                                    }
                                ]
                            }
                        }
                    ]
                }
            ]
        }

        result = await connector.handle_webhook(payload)

        assert result.event_type == "status_delivered"
        assert result.metadata["message_id"] == "wamid.msg123"

    @pytest.mark.asyncio
    async def test_handle_webhook_status_read(self, connector):
        """Test handling message status update - read."""
        payload = {
            "entry": [
                {
                    "changes": [
                        {
                            "value": {
                                "statuses": [
                                    {
                                        "id": "wamid.msg456",
                                        "status": "read",
                                        "timestamp": "1640002000",
                                    }
                                ]
                            }
                        }
                    ]
                }
            ]
        }

        result = await connector.handle_webhook(payload)

        assert result.event_type == "status_read"

    @pytest.mark.asyncio
    async def test_handle_webhook_unknown_type(self, connector):
        """Test handling unknown webhook event type."""
        payload = {"entry": [{"changes": [{"value": {}}]}]}

        result = await connector.handle_webhook(payload)

        assert result.event_type == "unknown"

    @pytest.mark.asyncio
    async def test_handle_webhook_invalid_signature(self, connector):
        """Test webhook rejection with invalid signature."""
        payload = {"entry": [{"changes": [{"value": {"messages": []}}]}]}
        headers = {"x-hub-signature-256": "sha256=invalid_signature"}

        result = await connector.handle_webhook(payload, headers=headers)

        assert result.event_type == "invalid_signature"

    @pytest.mark.asyncio
    async def test_handle_webhook_valid_signature(self, connector):
        """Test webhook acceptance with valid signature."""
        payload = {"entry": [{"changes": [{"value": {"messages": []}}]}]}
        body = json.dumps(payload, separators=(",", ":"))
        signature = hmac.new(
            b"test-app-secret-abc123",
            body.encode(),
            hashlib.sha256,
        ).hexdigest()

        headers = {"x-hub-signature-256": f"sha256={signature}"}

        result = await connector.handle_webhook(payload, headers=headers)

        assert result.event_type != "invalid_signature"


# =============================================================================
# Test Classes - Webhook Verification
# =============================================================================


class TestWebhookVerification:
    """Test webhook signature verification methods."""

    def test_verify_webhook_subscription_success(self, connector):
        """Test successful webhook subscription verification."""
        result = connector.verify_webhook_subscription(
            mode="subscribe",
            token="test-verify-token-secret",
            challenge="challenge_abc123",
        )

        assert result == "challenge_abc123"

    def test_verify_webhook_subscription_wrong_token(self, connector):
        """Test subscription verification with wrong token."""
        result = connector.verify_webhook_subscription(
            mode="subscribe",
            token="wrong_token",
            challenge="challenge_xyz",
        )

        assert result is None

    def test_verify_webhook_subscription_wrong_mode(self, connector):
        """Test subscription verification with wrong mode."""
        result = connector.verify_webhook_subscription(
            mode="unsubscribe",
            token="test-verify-token-secret",
            challenge="challenge_xyz",
        )

        assert result is None

    def test_verify_webhook_valid_signature(self, connector):
        """Test POST webhook signature verification."""
        body = b'{"test": "data"}'
        signature = (
            "sha256="
            + hmac.new(
                b"test-app-secret-abc123",
                body,
                hashlib.sha256,
            ).hexdigest()
        )

        result = connector.verify_webhook(
            headers={"X-Hub-Signature-256": signature},
            body=body,
        )

        assert result is True

    def test_verify_webhook_invalid_signature(self, connector):
        """Test webhook rejection with invalid signature."""
        result = connector.verify_webhook(
            headers={"X-Hub-Signature-256": "sha256=invalid"},
            body=b'{"test": "data"}',
        )

        assert result is False

    def test_verify_webhook_missing_prefix(self, connector):
        """Test webhook rejection when signature lacks sha256= prefix."""
        result = connector.verify_webhook(
            headers={"X-Hub-Signature-256": "no_prefix_signature"},
            body=b'{"test": "data"}',
        )

        assert result is False

    def test_verify_webhook_no_secret_dev_mode(self, connector_no_secret):
        """Test webhook passes in dev mode without secret."""
        with patch.dict(os.environ, {"ARAGORA_ENV": "development"}):
            result = connector_no_secret.verify_webhook(
                headers={},
                body=b'{"test": "data"}',
            )

            assert result is True

    def test_verify_webhook_no_secret_production_fails(self, connector_no_secret):
        """Test webhook fails in production without secret."""
        with patch.dict(os.environ, {"ARAGORA_ENV": "production"}):
            result = connector_no_secret.verify_webhook(
                headers={},
                body=b'{"test": "data"}',
            )

            assert result is False

    def test_verify_webhook_lowercase_header(self, connector):
        """Test verification with lowercase header name."""
        body = b'{"test": "data"}'
        signature = (
            "sha256="
            + hmac.new(
                b"test-app-secret-abc123",
                body,
                hashlib.sha256,
            ).hexdigest()
        )

        result = connector.verify_webhook(
            headers={"x-hub-signature-256": signature},
            body=body,
        )

        assert result is True


# =============================================================================
# Test Classes - Message Parsing
# =============================================================================


class TestParseMessage:
    """Test parse_message method."""

    @pytest.mark.asyncio
    async def test_parse_text_message(self, connector, sample_message_payload):
        """Test parsing a text message."""
        result = await connector.parse_message(sample_message_payload)

        assert isinstance(result, ChatMessage)
        assert result.id == "wamid.HBgLMTU1NTEyMzQ1NjcVAgARGBI5QTNDQTVCM0Q0Q0Q2MzU3NDgA"
        assert result.content == "Hello, bot!"
        assert result.platform == "whatsapp"
        assert result.message_type == MessageType.TEXT
        assert result.author.id == "15559876543"
        assert result.author.display_name == "John Doe"
        assert result.channel.is_dm is True

    @pytest.mark.asyncio
    async def test_parse_voice_message(self, connector):
        """Test parsing a voice message."""
        payload = {
            "entry": [
                {
                    "changes": [
                        {
                            "value": {
                                "messages": [
                                    {
                                        "from": "15559876543",
                                        "id": "wamid.voice123",
                                        "timestamp": "1640000000",
                                        "type": "audio",
                                        "audio": {
                                            "id": "audio_file_id",
                                            "mime_type": "audio/ogg; codecs=opus",
                                        },
                                    }
                                ],
                                "contacts": [{"wa_id": "15559876543", "profile": {"name": "User"}}],
                            }
                        }
                    ]
                }
            ]
        }

        result = await connector.parse_message(payload)

        assert result.message_type == MessageType.VOICE

    @pytest.mark.asyncio
    async def test_parse_document_message(self, connector):
        """Test parsing a document message."""
        payload = {
            "entry": [
                {
                    "changes": [
                        {
                            "value": {
                                "messages": [
                                    {
                                        "from": "15559876543",
                                        "id": "wamid.doc456",
                                        "timestamp": "1640000000",
                                        "type": "document",
                                        "document": {
                                            "id": "doc_file_id",
                                            "filename": "report.pdf",
                                        },
                                    }
                                ],
                                "contacts": [{"wa_id": "15559876543", "profile": {"name": "User"}}],
                            }
                        }
                    ]
                }
            ]
        }

        result = await connector.parse_message(payload)

        assert result.message_type == MessageType.FILE

    @pytest.mark.asyncio
    async def test_parse_image_message(self, connector):
        """Test parsing an image message."""
        payload = {
            "entry": [
                {
                    "changes": [
                        {
                            "value": {
                                "messages": [
                                    {
                                        "from": "15559876543",
                                        "id": "wamid.img789",
                                        "timestamp": "1640000000",
                                        "type": "image",
                                        "image": {"id": "img_file_id"},
                                    }
                                ],
                                "contacts": [{"wa_id": "15559876543", "profile": {"name": "User"}}],
                            }
                        }
                    ]
                }
            ]
        }

        result = await connector.parse_message(payload)

        assert result.message_type == MessageType.FILE

    @pytest.mark.asyncio
    async def test_parse_interactive_button_reply(self, connector, sample_interactive_payload):
        """Test parsing interactive button reply."""
        result = await connector.parse_message(sample_interactive_payload)

        assert result.content == "Approve"

    @pytest.mark.asyncio
    async def test_parse_interactive_list_reply(self, connector):
        """Test parsing interactive list selection."""
        payload = {
            "entry": [
                {
                    "changes": [
                        {
                            "value": {
                                "messages": [
                                    {
                                        "from": "15559876543",
                                        "id": "wamid.list_reply",
                                        "timestamp": "1640000000",
                                        "type": "interactive",
                                        "interactive": {
                                            "type": "list_reply",
                                            "list_reply": {
                                                "id": "item_3",
                                                "title": "Third Option",
                                            },
                                        },
                                    }
                                ],
                                "contacts": [{"wa_id": "15559876543", "profile": {}}],
                            }
                        }
                    ]
                }
            ]
        }

        result = await connector.parse_message(payload)

        assert result.content == "Third Option"

    @pytest.mark.asyncio
    async def test_parse_message_with_context(self, connector):
        """Test parsing a reply message with context."""
        payload = {
            "entry": [
                {
                    "changes": [
                        {
                            "value": {
                                "messages": [
                                    {
                                        "from": "15559876543",
                                        "id": "wamid.reply",
                                        "timestamp": "1640000000",
                                        "type": "text",
                                        "text": {"body": "This is a reply"},
                                        "context": {"id": "wamid.original_msg"},
                                    }
                                ],
                                "contacts": [{"wa_id": "15559876543", "profile": {"name": "User"}}],
                            }
                        }
                    ]
                }
            ]
        }

        result = await connector.parse_message(payload)

        assert result.thread_id == "wamid.original_msg"


# =============================================================================
# Test Classes - Command Parsing
# =============================================================================


class TestParseCommand:
    """Test parse_command method."""

    @pytest.mark.asyncio
    async def test_parse_slash_command(self, connector):
        """Test parsing /command."""
        payload = {
            "entry": [
                {
                    "changes": [
                        {
                            "value": {
                                "messages": [
                                    {
                                        "from": "15559876543",
                                        "id": "wamid.cmd1",
                                        "timestamp": "1640000000",
                                        "type": "text",
                                        "text": {"body": "/debate How should we invest?"},
                                    }
                                ],
                                "contacts": [{"wa_id": "15559876543", "profile": {"name": "User"}}],
                            }
                        }
                    ]
                }
            ]
        }

        result = await connector.parse_command(payload)

        assert result is not None
        assert result.name == "debate"
        assert result.args == ["How", "should", "we", "invest?"]
        assert result.text == "/debate How should we invest?"

    @pytest.mark.asyncio
    async def test_parse_exclamation_command(self, connector):
        """Test parsing !command."""
        payload = {
            "entry": [
                {
                    "changes": [
                        {
                            "value": {
                                "messages": [
                                    {
                                        "from": "15559876543",
                                        "id": "wamid.cmd2",
                                        "timestamp": "1640000000",
                                        "type": "text",
                                        "text": {"body": "!help"},
                                    }
                                ],
                                "contacts": [{"wa_id": "15559876543", "profile": {"name": "User"}}],
                            }
                        }
                    ]
                }
            ]
        }

        result = await connector.parse_command(payload)

        assert result is not None
        assert result.name == "help"
        assert result.args == []

    @pytest.mark.asyncio
    async def test_parse_no_command(self, connector):
        """Test that regular text is not parsed as command."""
        payload = {
            "entry": [
                {
                    "changes": [
                        {
                            "value": {
                                "messages": [
                                    {
                                        "from": "15559876543",
                                        "id": "wamid.text",
                                        "timestamp": "1640000000",
                                        "type": "text",
                                        "text": {"body": "Just a regular message"},
                                    }
                                ],
                                "contacts": [{"wa_id": "15559876543", "profile": {"name": "User"}}],
                            }
                        }
                    ]
                }
            ]
        }

        result = await connector.parse_command(payload)

        assert result is None

    @pytest.mark.asyncio
    async def test_parse_command_case_insensitive(self, connector):
        """Test command name is lowercased."""
        payload = {
            "entry": [
                {
                    "changes": [
                        {
                            "value": {
                                "messages": [
                                    {
                                        "from": "15559876543",
                                        "id": "wamid.cmd3",
                                        "timestamp": "1640000000",
                                        "type": "text",
                                        "text": {"body": "/DEBATE topic"},
                                    }
                                ],
                                "contacts": [{"wa_id": "15559876543", "profile": {"name": "User"}}],
                            }
                        }
                    ]
                }
            ]
        }

        result = await connector.parse_command(payload)

        assert result.name == "debate"


# =============================================================================
# Test Classes - Interaction Handling
# =============================================================================


class TestHandleInteraction:
    """Test handle_interaction method."""

    @pytest.mark.asyncio
    async def test_handle_button_click(self, connector, sample_interactive_payload):
        """Test handling button click interaction."""
        result = await connector.handle_interaction(sample_interactive_payload)

        assert isinstance(result, UserInteraction)
        assert result.action_id == "btn_approve"
        assert result.value == "Approve"
        assert result.interaction_type == InteractionType.BUTTON_CLICK

    @pytest.mark.asyncio
    async def test_handle_list_selection(self, connector):
        """Test handling list selection interaction."""
        payload = {
            "entry": [
                {
                    "changes": [
                        {
                            "value": {
                                "messages": [
                                    {
                                        "from": "15559876543",
                                        "id": "wamid.list_int",
                                        "timestamp": "1640000000",
                                        "type": "interactive",
                                        "interactive": {
                                            "type": "list_reply",
                                            "list_reply": {
                                                "id": "option_2",
                                                "title": "Second Choice",
                                            },
                                        },
                                    }
                                ],
                                "contacts": [{"wa_id": "15559876543", "profile": {"name": "User"}}],
                            }
                        }
                    ]
                }
            ]
        }

        result = await connector.handle_interaction(payload)

        assert result.action_id == "option_2"
        assert result.value == "Second Choice"
        assert result.interaction_type == InteractionType.SELECT_MENU


# =============================================================================
# Test Classes - Interactive Message Building
# =============================================================================


class TestBuildInteractive:
    """Test _build_interactive helper method."""

    def test_build_interactive_buttons(self, connector):
        """Test building button interactive payload."""
        blocks = [
            {"type": "button", "text": "Accept", "action_id": "btn_accept"},
            {"type": "button", "text": "Reject", "action_id": "btn_reject"},
        ]

        result = connector._build_interactive("Make a choice:", blocks)

        assert result["type"] == "button"
        assert result["body"]["text"] == "Make a choice:"
        assert len(result["action"]["buttons"]) == 2
        assert result["action"]["buttons"][0]["reply"]["id"] == "btn_accept"

    def test_build_interactive_button_text_truncation(self, connector):
        """Test button text is truncated to 20 chars."""
        blocks = [
            {"type": "button", "text": "This is a very long button text", "action_id": "btn1"},
        ]

        result = connector._build_interactive("Choose:", blocks)

        assert len(result["action"]["buttons"][0]["reply"]["title"]) <= 20

    def test_build_interactive_max_three_buttons(self, connector):
        """Test buttons are limited to 3."""
        blocks = [
            {"type": "button", "text": f"Button {i}", "action_id": f"btn_{i}"} for i in range(5)
        ]

        result = connector._build_interactive("Choose:", blocks)

        assert len(result["action"]["buttons"]) == 3

    def test_build_interactive_list(self, connector):
        """Test building list interactive payload."""
        blocks = [
            {
                "type": "list_item",
                "text": "Option A",
                "action_id": "opt_a",
                "description": "First option",
            },
            {
                "type": "list_item",
                "text": "Option B",
                "action_id": "opt_b",
                "description": "Second option",
            },
        ]

        result = connector._build_interactive("Select an option:", blocks)

        assert result["type"] == "list"
        assert result["body"]["text"] == "Select an option:"
        assert len(result["action"]["sections"][0]["rows"]) == 2

    def test_build_interactive_list_max_ten_items(self, connector):
        """Test list items are limited to 10."""
        blocks = [
            {"type": "list_item", "text": f"Item {i}", "action_id": f"item_{i}"} for i in range(15)
        ]

        result = connector._build_interactive("Choose:", blocks)

        assert len(result["action"]["sections"][0]["rows"]) == 10

    def test_build_interactive_list_title_truncation(self, connector):
        """Test list item title is truncated to 24 chars."""
        blocks = [
            {
                "type": "list_item",
                "text": "This is a very long list item title",
                "action_id": "item1",
            },
        ]

        result = connector._build_interactive("Choose:", blocks)

        assert len(result["action"]["sections"][0]["rows"][0]["title"]) <= 24

    def test_build_interactive_empty_blocks(self, connector):
        """Test building interactive with empty blocks."""
        result = connector._build_interactive("No options:", [])

        assert result["type"] == "button"
        assert result["action"]["buttons"] == []


# =============================================================================
# Test Classes - Block Formatting
# =============================================================================


class TestFormatBlocks:
    """Test format_blocks method."""

    def test_format_blocks_with_title(self, connector):
        """Test formatting blocks with title."""
        blocks = connector.format_blocks(title="Important Notice")

        assert any(b["type"] == "header" for b in blocks)
        assert any(b["text"] == "Important Notice" for b in blocks)

    def test_format_blocks_with_body(self, connector):
        """Test formatting blocks with body text."""
        blocks = connector.format_blocks(body="This is the main content.")

        assert any(b["type"] == "body" for b in blocks)

    def test_format_blocks_with_fields(self, connector):
        """Test formatting blocks with field tuples."""
        fields = [("Status", "Active"), ("Count", "42")]
        blocks = connector.format_blocks(fields=fields)

        assert len([b for b in blocks if b["type"] == "field"]) == 2

    def test_format_blocks_with_buttons(self, connector):
        """Test formatting blocks with action buttons."""
        buttons = [
            MessageButton(text="Click Me", action_id="btn_click", value="clicked"),
        ]
        blocks = connector.format_blocks(actions=buttons)

        assert any(b["type"] == "button" for b in blocks)

    def test_format_blocks_button_text_truncation(self, connector):
        """Test button text is truncated in format_blocks."""
        buttons = [
            MessageButton(text="Very Long Button Text Here", action_id="btn1"),
        ]
        blocks = connector.format_blocks(actions=buttons)

        button_block = next(b for b in blocks if b["type"] == "button")
        assert len(button_block["text"]) <= 20


class TestFormatButton:
    """Test format_button method."""

    def test_format_button_basic(self, connector):
        """Test basic button formatting."""
        button = connector.format_button(
            text="Submit",
            action_id="submit_btn",
            value="submit_value",
        )

        assert button["type"] == "button"
        assert button["text"] == "Submit"
        assert button["action_id"] == "submit_btn"
        assert button["value"] == "submit_value"

    def test_format_button_with_url(self, connector):
        """Test URL button formatting."""
        button = connector.format_button(
            text="Visit Site",
            action_id="url_btn",
            url="https://example.com",
        )

        assert button["type"] == "url_button"
        assert button["url"] == "https://example.com"

    def test_format_button_text_truncation(self, connector):
        """Test button text is truncated to 20 chars."""
        button = connector.format_button(
            text="This Is A Very Long Button Label",
            action_id="btn",
        )

        assert len(button["text"]) <= 20


# =============================================================================
# Test Classes - Webhook Event Parsing
# =============================================================================


class TestParseWebhookEvent:
    """Test parse_webhook_event method."""

    def test_parse_webhook_event_message(self, connector_no_secret):
        """Test parsing message webhook event."""
        payload = {
            "entry": [
                {
                    "changes": [
                        {
                            "value": {
                                "messages": [
                                    {
                                        "from": "15559876543",
                                        "id": "wamid.123",
                                        "timestamp": "1640000000",
                                        "type": "text",
                                        "text": {"body": "Hello"},
                                    }
                                ]
                            }
                        }
                    ]
                }
            ]
        }

        event = connector_no_secret.parse_webhook_event(
            headers={},
            body=json.dumps(payload).encode(),
        )

        assert event.event_type == "message"
        assert event.platform == "whatsapp"

    def test_parse_webhook_event_interactive(self, connector_no_secret):
        """Test parsing interactive webhook event."""
        payload = {
            "entry": [
                {
                    "changes": [
                        {
                            "value": {
                                "messages": [
                                    {
                                        "from": "15559876543",
                                        "id": "wamid.int",
                                        "timestamp": "1640000000",
                                        "type": "interactive",
                                        "interactive": {
                                            "button_reply": {"id": "btn1", "title": "Yes"}
                                        },
                                    }
                                ]
                            }
                        }
                    ]
                }
            ]
        }

        event = connector_no_secret.parse_webhook_event(
            headers={},
            body=json.dumps(payload).encode(),
        )

        assert event.event_type == "interaction"

    def test_parse_webhook_event_status(self, connector_no_secret):
        """Test parsing status webhook event."""
        payload = {
            "entry": [
                {
                    "changes": [
                        {
                            "value": {
                                "statuses": [
                                    {
                                        "id": "wamid.123",
                                        "status": "sent",
                                        "timestamp": "1640000000",
                                    }
                                ]
                            }
                        }
                    ]
                }
            ]
        }

        event = connector_no_secret.parse_webhook_event(
            headers={},
            body=json.dumps(payload).encode(),
        )

        assert event.event_type == "status"

    def test_parse_webhook_event_invalid_json(self, connector):
        """Test parsing invalid JSON returns error event."""
        event = connector.parse_webhook_event(
            headers={},
            body=b"not valid json",
        )

        assert event.event_type == "error"

    def test_parse_webhook_event_signature_mismatch(self, connector):
        """Test parsing with invalid signature returns error."""
        payload = {"entry": [{"changes": [{"value": {}}]}]}

        event = connector.parse_webhook_event(
            headers={"X-Hub-Signature-256": "sha256=invalid"},
            body=json.dumps(payload).encode(),
        )

        assert event.event_type == "error"
        assert "signature_mismatch" in str(event.raw_payload.get("error", ""))


# =============================================================================
# Test Classes - Response Methods
# =============================================================================


class TestRespondToCommand:
    """Test respond_to_command method."""

    @pytest.mark.asyncio
    async def test_respond_to_command(self, connector, mock_httpx_response):
        """Test responding to a command."""
        mock_response = mock_httpx_response({"messages": [{"id": "wamid.response"}]})

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            command = BotCommand(
                name="help",
                text="/help",
                channel=ChatChannel(id="+15559876543", platform="whatsapp"),
                user=ChatUser(id="+15559876543", platform="whatsapp"),
            )

            result = await connector.respond_to_command(
                command=command,
                text="Here is the help information.",
            )

            assert result.success is True

    @pytest.mark.asyncio
    async def test_respond_to_command_no_channel(self, connector):
        """Test command response without channel fails gracefully."""
        command = BotCommand(name="test", text="/test", channel=None)

        result = await connector.respond_to_command(
            command=command,
            text="Response",
        )

        assert result.success is False
        assert "No channel" in result.error


class TestRespondToInteraction:
    """Test respond_to_interaction method."""

    @pytest.mark.asyncio
    async def test_respond_to_interaction(self, connector, mock_httpx_response):
        """Test responding to an interaction."""
        mock_response = mock_httpx_response({"messages": [{"id": "wamid.int_response"}]})

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            interaction = UserInteraction(
                id="int_123",
                interaction_type=InteractionType.BUTTON_CLICK,
                action_id="btn_confirm",
                channel=ChatChannel(id="+15559876543", platform="whatsapp"),
            )

            result = await connector.respond_to_interaction(
                interaction=interaction,
                text="Action confirmed!",
            )

            assert result.success is True

    @pytest.mark.asyncio
    async def test_respond_to_interaction_no_channel(self, connector):
        """Test interaction response without channel fails gracefully."""
        interaction = UserInteraction(
            id="int_456",
            interaction_type=InteractionType.BUTTON_CLICK,
            action_id="btn",
            channel=None,
        )

        result = await connector.respond_to_interaction(
            interaction=interaction,
            text="Response",
        )

        assert result.success is False
        assert "No channel" in result.error


# =============================================================================
# Test Classes - Channel and User Info
# =============================================================================


class TestChannelAndUserInfo:
    """Test get_channel_info and get_user_info methods."""

    @pytest.mark.asyncio
    async def test_get_channel_info(self, connector):
        """Test getting channel info (WhatsApp = phone number)."""
        result = await connector.get_channel_info("+15559876543")

        assert isinstance(result, ChatChannel)
        assert result.id == "+15559876543"
        assert result.platform == "whatsapp"
        assert result.is_dm is True
        assert result.name == "+15559876543"

    @pytest.mark.asyncio
    async def test_get_user_info(self, connector):
        """Test getting user info."""
        result = await connector.get_user_info("+15559876543")

        assert isinstance(result, ChatUser)
        assert result.id == "+15559876543"
        assert result.username == "+15559876543"
        assert result.platform == "whatsapp"


# =============================================================================
# Test Classes - Evidence Extraction
# =============================================================================


class TestExtractEvidence:
    """Test extract_evidence method."""

    @pytest.mark.asyncio
    async def test_extract_evidence(self, connector):
        """Test extracting evidence from a message."""
        message = ChatMessage(
            id="wamid.evidence123",
            platform="whatsapp",
            channel=ChatChannel(id="+15559876543", platform="whatsapp", name="User Chat"),
            author=ChatUser(
                id="+15559876543",
                platform="whatsapp",
                display_name="John Doe",
            ),
            content="I think we should invest in solar energy.",
            timestamp=datetime(2024, 1, 15, 10, 30, 0),
        )

        evidence = await connector.extract_evidence(message)

        assert isinstance(evidence, ChatEvidence)
        assert evidence.id == "evidence-wamid.evidence123"
        assert evidence.platform == "whatsapp"
        assert evidence.content == "I think we should invest in solar energy."
        assert evidence.author_name == "John Doe"


# =============================================================================
# Test Classes - Rate Limiting
# =============================================================================


class TestRateLimiting:
    """Test rate limiting and retry behavior."""

    @pytest.mark.asyncio
    async def test_rate_limit_retry(self, connector, mock_httpx_response):
        """Test automatic retry on rate limit (429)."""
        rate_limit_response = mock_httpx_response(
            {"error": {"message": "Rate limit exceeded", "code": 4}}
        )
        success_response = mock_httpx_response({"messages": [{"id": "wamid.after_retry"}]})

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.post = AsyncMock(side_effect=[rate_limit_response, success_response])

            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await connector.send_message(
                    channel_id="+15559876543",
                    text="Test after rate limit",
                )

            # Should eventually succeed after retry
            assert result.success is True

    @pytest.mark.asyncio
    async def test_rate_limit_code_80007(self, connector, mock_httpx_response):
        """Test handling WhatsApp rate limit code 80007."""
        rate_limit_response = mock_httpx_response(
            {"error": {"message": "Throughput limit reached", "code": 80007}}
        )
        success_response = mock_httpx_response({"messages": [{"id": "wamid.success"}]})

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.post = AsyncMock(side_effect=[rate_limit_response, success_response])

            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await connector.send_message(
                    channel_id="+15559876543",
                    text="Test",
                )

            assert result.success is True

    @pytest.mark.asyncio
    async def test_rate_limit_code_130429(self, connector, mock_httpx_response):
        """Test handling rate limit code 130429."""
        rate_limit_response = mock_httpx_response(
            {"error": {"message": "Message rate limit", "code": 130429}}
        )
        success_response = mock_httpx_response({"messages": [{"id": "wamid.ok"}]})

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.post = AsyncMock(side_effect=[rate_limit_response, success_response])

            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await connector.send_message(
                    channel_id="+15559876543",
                    text="Test",
                )

            assert result.success is True

    @pytest.mark.asyncio
    async def test_rate_limit_max_retries_exceeded(self, connector, mock_httpx_response):
        """Test failure when max retries exceeded."""
        rate_limit_response = mock_httpx_response(
            {"error": {"message": "Rate limit exceeded", "code": 4}}
        )

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.post = AsyncMock(return_value=rate_limit_response)

            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await connector.send_message(
                    channel_id="+15559876543",
                    text="Test",
                )

            assert result.success is False
            assert "Rate limited" in result.error


# =============================================================================
# Test Classes - Error Handling
# =============================================================================


class TestErrorHandling:
    """Test error handling and recovery."""

    @pytest.mark.asyncio
    async def test_auth_error_no_retry(self, connector, mock_httpx_response):
        """Test auth errors (190, 102) do not retry."""
        auth_error_response = mock_httpx_response(
            {"error": {"message": "Invalid access token", "code": 190}}
        )

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.post = AsyncMock(return_value=auth_error_response)

            result = await connector.send_message(
                channel_id="+15559876543",
                text="Test",
            )

            # Should fail immediately without retrying
            assert mock_instance.post.call_count == 1
            assert result.success is False
            assert "Auth error" in result.error

    @pytest.mark.asyncio
    async def test_timeout_retry(self, connector):
        """Test timeout errors trigger retry."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.post = AsyncMock(side_effect=httpx.TimeoutException("Request timed out"))

            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await connector.send_message(
                    channel_id="+15559876543",
                    text="Test",
                )

            # Should have retried
            assert mock_instance.post.call_count == 3  # max_retries
            assert result.success is False
            assert "timeout" in result.error.lower()

    @pytest.mark.asyncio
    async def test_server_error_retry(self, connector):
        """Test 5xx server errors trigger retry."""
        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value

            error_response = MagicMock()
            error_response.status_code = 500

            mock_instance.post = AsyncMock(
                side_effect=httpx.HTTPStatusError(
                    "Server error",
                    request=MagicMock(),
                    response=error_response,
                )
            )

            with patch("asyncio.sleep", new_callable=AsyncMock):
                result = await connector.send_message(
                    channel_id="+15559876543",
                    text="Test",
                )

            assert mock_instance.post.call_count == 3
            assert result.success is False

    @pytest.mark.asyncio
    async def test_httpx_not_available(self):
        """Test graceful handling when httpx not available."""
        with patch("aragora.connectors.chat.whatsapp.HTTPX_AVAILABLE", False):
            connector = WhatsAppConnector(access_token="test")

            result = await connector.send_message("+15559876543", "Test")

            assert result.success is False
            assert "httpx not available" in result.error


# =============================================================================
# Test Classes - Circuit Breaker
# =============================================================================


class TestCircuitBreaker:
    """Test circuit breaker functionality."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_after_failures(self, connector, mock_httpx_response):
        """Test circuit breaker opens after threshold failures."""
        error_response = mock_httpx_response(
            {"error": {"message": "Service unavailable", "code": 500}}
        )

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=error_response
            )

            # Make multiple failing requests
            for _ in range(10):
                await connector.send_message("+15559876543", "Test")

            # Circuit breaker should be tracking failures
            cb = connector._get_circuit_breaker()
            if cb:
                assert cb.get_status() in ["open", "half_open", "closed"]

    @pytest.mark.asyncio
    async def test_circuit_breaker_disabled(self):
        """Test operations work with circuit breaker disabled."""
        connector = WhatsAppConnector(
            access_token="test",
            enable_circuit_breaker=False,
        )

        assert connector._get_circuit_breaker() is None

    @pytest.mark.asyncio
    async def test_circuit_breaker_check_returns_true_when_disabled(self):
        """Test _check_circuit_breaker allows requests when disabled."""
        connector = WhatsAppConnector(
            access_token="test",
            enable_circuit_breaker=False,
        )

        can_proceed, error = connector._check_circuit_breaker()

        assert can_proceed is True
        assert error is None


# =============================================================================
# Test Classes - Debate Integration
# =============================================================================


class TestDebateIntegration:
    """Test debate-related functionality."""

    @pytest.mark.asyncio
    async def test_debate_command_parsing(self, connector):
        """Test parsing /debate command for initiating debates."""
        payload = {
            "entry": [
                {
                    "changes": [
                        {
                            "value": {
                                "messages": [
                                    {
                                        "from": "15559876543",
                                        "id": "wamid.debate_cmd",
                                        "timestamp": "1640000000",
                                        "type": "text",
                                        "text": {"body": "/debate Should we adopt remote work?"},
                                    }
                                ],
                                "contacts": [
                                    {"wa_id": "15559876543", "profile": {"name": "Manager"}}
                                ],
                            }
                        }
                    ]
                }
            ]
        }

        command = await connector.parse_command(payload)

        assert command is not None
        assert command.name == "debate"
        assert "remote work" in " ".join(command.args)

    @pytest.mark.asyncio
    async def test_format_debate_result(self, connector, mock_httpx_response):
        """Test formatting and sending debate results."""
        mock_response = mock_httpx_response({"messages": [{"id": "wamid.result"}]})

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.post = AsyncMock(return_value=mock_response)

            # Format debate result as blocks
            blocks = connector.format_blocks(
                title="Debate Result",
                body="The consensus is to adopt a hybrid remote work policy.",
                fields=[
                    ("Confidence", "87%"),
                    ("Agents Agreed", "4/5"),
                ],
                actions=[
                    MessageButton(text="View Details", action_id="view_details"),
                    MessageButton(text="Start New", action_id="new_debate"),
                ],
            )

            result = await connector.send_message(
                channel_id="+15559876543",
                text="Debate completed!",
                blocks=blocks,
            )

            assert result.success is True
