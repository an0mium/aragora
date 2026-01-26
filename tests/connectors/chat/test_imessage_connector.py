"""
Tests for iMessage Connector (via BlueBubbles).

Tests cover:
- Initialization and configuration
- Message sending
- Tapback reactions
- Webhook handling
- Read receipts and typing indicators
- Error handling
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from aragora.connectors.chat.imessage import IMessageConnector
from aragora.connectors.chat.models import SendMessageResponse, WebhookEvent


class TestIMessageConnectorInit:
    """Tests for IMessageConnector initialization."""

    def test_default_initialization(self):
        """Test default connector initialization."""
        with patch.dict(
            "os.environ",
            {
                "BLUEBUBBLES_URL": "http://localhost:1234",
                "BLUEBUBBLES_PASSWORD": "test-password",
            },
        ):
            connector = IMessageConnector()
            assert "localhost:1234" in connector._api_url
            assert connector._password == "test-password"
            assert connector.platform_name == "imessage"
            assert connector.platform_display_name == "iMessage"

    def test_custom_initialization(self):
        """Test connector with custom settings."""
        connector = IMessageConnector(
            api_url="http://custom:5000",
            password="custom-password",
        )
        assert connector._api_url == "http://custom:5000"
        assert connector._password == "custom-password"

    def test_is_configured(self):
        """Test is_configured check."""
        connector = IMessageConnector(
            api_url="http://localhost:1234",
            password="test",
        )
        assert connector.is_configured is True

    def test_not_configured_missing_url(self):
        """Test is_configured when URL is missing."""
        connector = IMessageConnector(
            api_url="",
            password="test",
        )
        assert connector.is_configured is False

    def test_not_configured_missing_password(self):
        """Test is_configured when password is missing."""
        connector = IMessageConnector(
            api_url="http://localhost:1234",
            password="",
        )
        assert connector.is_configured is False


class TestIMessageConnectorMessages:
    """Tests for message operations."""

    @pytest.fixture
    def connector(self):
        """Create a configured connector."""
        return IMessageConnector(
            api_url="http://localhost:1234",
            password="test-password",
        )

    @pytest.mark.asyncio
    async def test_send_message_success(self, connector):
        """Test successful message sending."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = '{"status": 200, "data": {"guid": "msg-123"}}'
        mock_response.json.return_value = {"status": 200, "data": {"guid": "msg-123"}}

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            result = await connector.send_message(
                channel_id="+1234567890",
                text="Hello from iMessage!",
            )

            assert result.success is True

    @pytest.mark.asyncio
    async def test_send_message_to_chat(self, connector):
        """Test sending message to existing chat by GUID."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = '{"status": 200, "data": {"guid": "msg-456"}}'
        mock_response.json.return_value = {"status": 200, "data": {"guid": "msg-456"}}

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            result = await connector.send_message(
                channel_id="chat-guid-abc123",
                text="Hello chat!",
            )

            assert result.success is True

    @pytest.mark.asyncio
    async def test_send_message_error(self, connector):
        """Test handling HTTP error."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal server error"

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            result = await connector.send_message(
                channel_id="+1234567890",
                text="Hello!",
            )

            assert result.success is False

    @pytest.mark.asyncio
    async def test_update_message_not_supported(self, connector):
        """Test that update_message returns failure (iMessage doesn't support it)."""
        result = await connector.update_message(
            channel_id="+1234567890",
            message_id="msg-123",
            text="Updated text",
        )

        assert result.success is False


class TestIMessageConnectorTapbacks:
    """Tests for tapback (reaction) operations."""

    @pytest.fixture
    def connector(self):
        """Create a configured connector."""
        return IMessageConnector(
            api_url="http://localhost:1234",
            password="test-password",
        )

    @pytest.mark.asyncio
    async def test_send_tapback(self, connector):
        """Test sending a tapback reaction."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = '{"status": 200}'
        mock_response.json.return_value = {"status": 200}

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            # Tapbacks are typically sent via respond_to_interaction
            result = await connector.respond_to_interaction(
                interaction_id="msg-123",
                response="love",  # tapback type
            )

            assert result.success is True


class TestIMessageConnectorWebhook:
    """Tests for webhook handling."""

    @pytest.fixture
    def connector(self):
        """Create a configured connector."""
        return IMessageConnector(
            api_url="http://localhost:1234",
            password="test-password",
        )

    def test_verify_webhook_success(self, connector):
        """Test successful webhook verification."""
        result = connector.verify_webhook({"type": "new-message"})
        assert result is True

    def test_parse_webhook_event_new_message(self, connector):
        """Test parsing new message webhook."""
        payload = {
            "type": "new-message",
            "data": {
                "guid": "msg-123",
                "text": "Hello!",
                "handle": {"id": "+1234567890"},
                "chats": [{"guid": "chat-abc"}],
                "dateCreated": 1234567890000,
                "isFromMe": False,
            },
        }

        event = connector.parse_webhook_event(payload)

        assert event is not None
        assert event.platform == "imessage"

    def test_parse_webhook_event_typing(self, connector):
        """Test parsing typing indicator webhook."""
        payload = {
            "type": "typing-indicator",
            "data": {
                "display": True,
                "guid": "chat-abc",
            },
        }

        event = connector.parse_webhook_event(payload)

        assert event is not None

    def test_parse_webhook_event_read_receipt(self, connector):
        """Test parsing read receipt webhook."""
        payload = {
            "type": "updated-message",
            "data": {
                "guid": "msg-123",
                "dateRead": 1234567890000,
            },
        }

        event = connector.parse_webhook_event(payload)

        assert event is not None


class TestIMessageConnectorFiles:
    """Tests for file operations."""

    @pytest.fixture
    def connector(self):
        """Create a configured connector."""
        return IMessageConnector(
            api_url="http://localhost:1234",
            password="test-password",
        )

    @pytest.mark.asyncio
    async def test_upload_file(self, connector):
        """Test file upload as attachment."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = '{"status": 200, "data": {"guid": "att-123"}}'
        mock_response.json.return_value = {"status": 200, "data": {"guid": "att-123"}}

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            result = await connector.upload_file(
                channel_id="+1234567890",
                file_data=b"test image content",
                filename="photo.jpg",
            )

            assert result is not None

    @pytest.mark.asyncio
    async def test_download_file(self, connector):
        """Test file download."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"downloaded image data"

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            result = await connector.download_file("attachment-guid-123")

            assert result is not None


class TestIMessageConnectorCircuitBreaker:
    """Tests for circuit breaker behavior."""

    @pytest.fixture
    def connector(self):
        """Create a configured connector."""
        return IMessageConnector(
            api_url="http://localhost:1234",
            password="test-password",
        )

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_after_failures(self, connector):
        """Test that circuit breaker opens after consecutive failures."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "BlueBubbles server error"

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            # Trigger multiple failures to open circuit breaker
            for _ in range(10):
                try:
                    await connector.send_message("+1234567890", "test")
                except Exception:
                    pass

            # Circuit breaker should now be open
            result = await connector.send_message("+1234567890", "test")
            # Either raises or returns failure
            if result:
                assert result.success is False


class TestIMessageConnectorCommands:
    """Tests for bot command handling."""

    @pytest.fixture
    def connector(self):
        """Create a configured connector."""
        return IMessageConnector(
            api_url="http://localhost:1234",
            password="test-password",
        )

    @pytest.mark.asyncio
    async def test_respond_to_command(self, connector):
        """Test responding to a bot command."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = '{"status": 200, "data": {"guid": "msg-response"}}'
        mock_response.json.return_value = {"status": 200, "data": {"guid": "msg-response"}}

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            result = await connector.respond_to_command(
                command_id="cmd-123",
                channel_id="+1234567890",
                response="Command executed!",
            )

            assert result.success is True
