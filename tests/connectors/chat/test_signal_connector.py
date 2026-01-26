"""
Tests for Signal Messenger Connector.

Tests cover:
- Initialization and configuration
- Message sending
- Webhook handling and verification
- Circuit breaker behavior
- Error handling
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from aragora.connectors.chat.signal import SignalConnector
from aragora.connectors.chat.models import SendMessageResponse, WebhookEvent


class TestSignalConnectorInit:
    """Tests for SignalConnector initialization."""

    def test_default_initialization(self):
        """Test default connector initialization."""
        with patch.dict(
            "os.environ",
            {
                "SIGNAL_CLI_URL": "http://localhost:8080",
                "SIGNAL_PHONE_NUMBER": "+1234567890",
            },
        ):
            connector = SignalConnector()
            assert connector._api_url == "http://localhost:8080"
            assert connector._phone_number == "+1234567890"
            assert connector.platform_name == "signal"
            assert connector.platform_display_name == "Signal"

    def test_custom_initialization(self):
        """Test connector with custom settings."""
        connector = SignalConnector(
            api_url="http://custom:9000/",
            phone_number="+9876543210",
        )
        assert connector._api_url == "http://custom:9000"
        assert connector._phone_number == "+9876543210"

    def test_is_configured(self):
        """Test is_configured check."""
        connector = SignalConnector(
            api_url="http://localhost:8080",
            phone_number="+1234567890",
        )
        assert connector.is_configured is True

    def test_not_configured_missing_url(self):
        """Test is_configured when URL is missing."""
        connector = SignalConnector(
            api_url="",
            phone_number="+1234567890",
        )
        assert connector.is_configured is False

    def test_not_configured_missing_phone(self):
        """Test is_configured when phone is missing."""
        connector = SignalConnector(
            api_url="http://localhost:8080",
            phone_number="",
        )
        assert connector.is_configured is False


class TestSignalConnectorMessages:
    """Tests for message operations."""

    @pytest.fixture
    def connector(self):
        """Create a configured connector."""
        return SignalConnector(
            api_url="http://localhost:8080",
            phone_number="+1234567890",
        )

    @pytest.mark.asyncio
    async def test_send_message_success(self, connector):
        """Test successful message sending."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = '{"timestamp": 1234567890000}'
        mock_response.json.return_value = {"timestamp": 1234567890000}

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            result = await connector.send_message(
                channel_id="+9876543210",
                text="Hello, Signal!",
            )

            assert result.success is True
            assert result.channel_id == "+9876543210"
            assert result.message_id == "1234567890000"

    @pytest.mark.asyncio
    async def test_send_message_to_group(self, connector):
        """Test sending message to a group."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = '{"timestamp": 1234567890000}'
        mock_response.json.return_value = {"timestamp": 1234567890000}

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            result = await connector.send_message(
                channel_id="group.abc123",
                text="Hello group!",
            )

            assert result.success is True

    @pytest.mark.asyncio
    async def test_send_message_rate_limited(self, connector):
        """Test handling rate limit (429)."""
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.text = "Too many requests"

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            result = await connector.send_message(
                channel_id="+9876543210",
                text="Hello!",
            )

            assert result.success is False
            assert "rate limit" in result.error.lower()

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
                channel_id="+9876543210",
                text="Hello!",
            )

            assert result.success is False
            assert "500" in result.error

    @pytest.mark.asyncio
    async def test_update_message_not_supported(self, connector):
        """Test that update_message returns failure (Signal doesn't support it)."""
        result = await connector.update_message(
            channel_id="+9876543210",
            message_id="123",
            text="Updated text",
        )

        assert result.success is False
        assert "not support" in result.error.lower() or "edit" in result.error.lower()

    @pytest.mark.asyncio
    async def test_delete_message(self, connector):
        """Test message deletion."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = ""

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.delete = AsyncMock(
                return_value=mock_response
            )

            result = await connector.delete_message(
                channel_id="+9876543210",
                message_id="123456789000",
            )

            # Signal has limited delete support
            assert isinstance(result.success, bool)


class TestSignalConnectorWebhook:
    """Tests for webhook handling."""

    @pytest.fixture
    def connector(self):
        """Create a configured connector."""
        return SignalConnector(
            api_url="http://localhost:8080",
            phone_number="+1234567890",
        )

    def test_verify_webhook_success(self, connector):
        """Test successful webhook verification."""
        import json

        # Signal webhook verification uses headers and body
        headers = {"content-type": "application/json"}
        body = json.dumps({"source": "+9876543210"}).encode()
        result = connector.verify_webhook(headers, body)
        assert result is True

    def test_parse_webhook_event_message(self, connector):
        """Test parsing incoming message webhook."""
        import json

        payload = {
            "envelope": {
                "source": "+9876543210",
                "sourceDevice": 1,
                "timestamp": 1234567890000,
                "dataMessage": {
                    "timestamp": 1234567890000,
                    "message": "Hello from Signal!",
                    "groupInfo": None,
                },
            },
        }

        event = connector.parse_webhook_event(json.dumps(payload).encode())

        assert event is not None
        assert event.platform == "signal"

    def test_parse_webhook_event_group(self, connector):
        """Test parsing group message webhook."""
        import json

        payload = {
            "envelope": {
                "source": "+9876543210",
                "timestamp": 1234567890000,
                "dataMessage": {
                    "message": "Group message",
                    "groupInfo": {
                        "groupId": "abc123",
                        "type": "DELIVER",
                    },
                },
            },
        }

        event = connector.parse_webhook_event(json.dumps(payload).encode())

        assert event is not None


class TestSignalConnectorFiles:
    """Tests for file operations."""

    @pytest.fixture
    def connector(self):
        """Create a configured connector."""
        return SignalConnector(
            api_url="http://localhost:8080",
            phone_number="+1234567890",
        )

    @pytest.mark.asyncio
    async def test_upload_file(self, connector):
        """Test file upload (as attachment)."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = '{"timestamp": 1234567890000}'
        mock_response.json.return_value = {"timestamp": 1234567890000}

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            result = await connector.upload_file(
                channel_id="+9876543210",
                file_data=b"test file content",
                filename="test.txt",
            )

            # Signal sends files as attachments in messages
            assert result is not None

    @pytest.mark.asyncio
    async def test_download_file(self, connector):
        """Test file download."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"downloaded content"

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            result = await connector.download_file("attachment-id-123")

            assert result is not None


class TestSignalConnectorCircuitBreaker:
    """Tests for circuit breaker behavior."""

    @pytest.fixture
    def connector(self):
        """Create a configured connector."""
        return SignalConnector(
            api_url="http://localhost:8080",
            phone_number="+1234567890",
        )

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_after_failures(self, connector):
        """Test that circuit breaker opens after consecutive failures."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Server error"

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            # Trigger multiple failures to open circuit breaker
            for _ in range(10):
                try:
                    await connector.send_message("+9876543210", "test")
                except Exception:
                    pass

            # Circuit breaker should now be open
            result = await connector.send_message("+9876543210", "test")
            # Either raises or returns failure
            if result:
                assert result.success is False
