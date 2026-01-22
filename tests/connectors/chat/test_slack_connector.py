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

        connector = SlackConnector(bot_token="xoxb-test", use_circuit_breaker=False)

        mock_response = MagicMock()
        mock_response.status_code = 200
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

        connector = SlackConnector(bot_token="xoxb-test", use_circuit_breaker=False)

        mock_response = MagicMock()
        mock_response.status_code = 200
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

        connector = SlackConnector(bot_token="xoxb-test", use_circuit_breaker=False)

        mock_response = MagicMock()
        mock_response.status_code = 200
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

        connector = SlackConnector(bot_token="xoxb-test", use_circuit_breaker=False)

        mock_response = MagicMock()
        mock_response.status_code = 400
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

        connector = SlackConnector(bot_token="xoxb-test", use_circuit_breaker=False, max_retries=1)

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

        connector = SlackConnector(bot_token="xoxb-test", use_circuit_breaker=False)

        mock_response = MagicMock()
        mock_response.status_code = 200
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

        connector = SlackConnector(bot_token="xoxb-test", use_circuit_breaker=False)

        mock_response = MagicMock()
        mock_response.status_code = 400
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

        connector = SlackConnector(bot_token="xoxb-test", use_circuit_breaker=False)

        mock_response = MagicMock()
        mock_response.status_code = 200
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

        connector = SlackConnector(bot_token="xoxb-test", use_circuit_breaker=False)

        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"ok": False, "error": "message_not_found"}

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

        connector = SlackConnector(bot_token="xoxb-test", use_circuit_breaker=False)

        mock_response = MagicMock()
        mock_response.status_code = 200
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


class TestSlackResilienceFeatures:
    """Tests for circuit breaker, retry logic, and timeout handling."""

    def test_init_with_circuit_breaker_disabled(self):
        """Should allow disabling circuit breaker."""
        from aragora.connectors.chat.slack import SlackConnector

        connector = SlackConnector(bot_token="xoxb-test", use_circuit_breaker=False)
        assert connector._circuit_breaker is None

    def test_init_with_custom_timeout(self):
        """Should accept custom timeout."""
        from aragora.connectors.chat.slack import SlackConnector

        connector = SlackConnector(bot_token="xoxb-test", timeout=60.0)
        assert connector._timeout == 60.0

    def test_init_with_custom_retries(self):
        """Should accept custom max_retries."""
        from aragora.connectors.chat.slack import SlackConnector

        connector = SlackConnector(bot_token="xoxb-test", max_retries=5)
        assert connector._max_retries == 5

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_on_failures(self):
        """Circuit breaker should open after threshold failures."""
        from aragora.connectors.chat.slack import SlackConnector
        from aragora.resilience import reset_all_circuit_breakers

        reset_all_circuit_breakers()

        connector = SlackConnector(
            bot_token="xoxb-test",
            use_circuit_breaker=True,
            max_retries=1,  # No retries for faster test
        )

        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"ok": False, "error": "invalid_auth"}

        # Trigger multiple failures to open circuit
        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                return_value=mock_response
            )

            for _ in range(6):  # More than threshold (5)
                await connector.send_message(channel_id="C123", text="Test")

        # Circuit should be open now
        result = await connector.send_message(channel_id="C123", text="Test")
        assert result.success is False
        assert "Circuit breaker open" in result.error

    @pytest.mark.asyncio
    async def test_retry_on_rate_limit(self):
        """Should retry on 429 rate limit errors."""
        from aragora.connectors.chat.slack import SlackConnector

        connector = SlackConnector(
            bot_token="xoxb-test",
            use_circuit_breaker=False,
            max_retries=3,
        )

        # First two calls return rate limit, third succeeds
        rate_limit_response = MagicMock()
        rate_limit_response.status_code = 429
        rate_limit_response.json.return_value = {"ok": False, "error": "rate_limited"}

        success_response = MagicMock()
        success_response.status_code = 200
        success_response.json.return_value = {"ok": True, "ts": "123", "channel": "C1"}

        call_count = 0

        async def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return rate_limit_response
            return success_response

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = mock_post

            # Patch sleep to avoid slow test
            with patch("aragora.connectors.chat.slack._exponential_backoff"):
                result = await connector.send_message(
                    channel_id="C12345",
                    text="Test",
                )

        assert result.success is True
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_on_server_error(self):
        """Should retry on 5xx server errors."""
        from aragora.connectors.chat.slack import SlackConnector

        connector = SlackConnector(
            bot_token="xoxb-test",
            use_circuit_breaker=False,
            max_retries=2,
        )

        server_error_response = MagicMock()
        server_error_response.status_code = 500
        server_error_response.json.return_value = {"ok": False, "error": "internal_error"}

        success_response = MagicMock()
        success_response.status_code = 200
        success_response.json.return_value = {"ok": True, "ts": "123", "channel": "C1"}

        call_count = 0

        async def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return server_error_response
            return success_response

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = mock_post

            with patch("aragora.connectors.chat.slack._exponential_backoff"):
                result = await connector.send_message(
                    channel_id="C12345",
                    text="Test",
                )

        assert result.success is True
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_no_retry_on_client_error(self):
        """Should not retry on 4xx client errors (except 429)."""
        from aragora.connectors.chat.slack import SlackConnector

        connector = SlackConnector(
            bot_token="xoxb-test",
            use_circuit_breaker=False,
            max_retries=3,
        )

        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"ok": False, "error": "channel_not_found"}

        call_count = 0

        async def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return mock_response

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = mock_post

            result = await connector.send_message(
                channel_id="invalid",
                text="Test",
            )

        assert result.success is False
        assert call_count == 1  # No retries

    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Should handle timeout exceptions with retry."""
        from aragora.connectors.chat.slack import SlackConnector
        import httpx

        connector = SlackConnector(
            bot_token="xoxb-test",
            use_circuit_breaker=False,
            max_retries=2,
        )

        success_response = MagicMock()
        success_response.status_code = 200
        success_response.json.return_value = {"ok": True, "ts": "123", "channel": "C1"}

        call_count = 0

        async def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise httpx.TimeoutException("Connection timeout")
            return success_response

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = mock_post

            with patch("aragora.connectors.chat.slack._exponential_backoff"):
                result = await connector.send_message(
                    channel_id="C12345",
                    text="Test",
                )

        assert result.success is True
        assert call_count == 2


class TestSlackChannelAndUserInfo:
    """Tests for get_channel_info and get_user_info methods."""

    @pytest.mark.asyncio
    async def test_get_channel_info_success(self):
        """Should get channel info successfully."""
        from aragora.connectors.chat.slack import SlackConnector

        connector = SlackConnector(bot_token="xoxb-test", use_circuit_breaker=False)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "ok": True,
            "channel": {
                "id": "C12345",
                "name": "general",
                "is_private": False,
                "context_team_id": "T123",
                "topic": {"value": "General discussion"},
                "purpose": {"value": "Team chat"},
                "num_members": 50,
            },
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            result = await connector.get_channel_info("C12345")

        assert result is not None
        assert result.id == "C12345"
        assert result.name == "general"
        assert result.is_private is False
        assert result.team_id == "T123"
        assert result.metadata["topic"] == "General discussion"

    @pytest.mark.asyncio
    async def test_get_channel_info_not_found(self):
        """Should return None when channel not found."""
        from aragora.connectors.chat.slack import SlackConnector

        connector = SlackConnector(bot_token="xoxb-test", use_circuit_breaker=False)

        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"ok": False, "error": "channel_not_found"}

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            result = await connector.get_channel_info("invalid")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_user_info_success(self):
        """Should get user info successfully."""
        from aragora.connectors.chat.slack import SlackConnector

        connector = SlackConnector(bot_token="xoxb-test", use_circuit_breaker=False)

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "ok": True,
            "user": {
                "id": "U12345",
                "name": "john.doe",
                "is_bot": False,
                "team_id": "T123",
                "tz": "America/Los_Angeles",
                "profile": {
                    "display_name": "John Doe",
                    "real_name": "John Doe",
                    "email": "john@example.com",
                    "title": "Engineer",
                },
            },
        }

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            result = await connector.get_user_info("U12345")

        assert result is not None
        assert result.id == "U12345"
        assert result.username == "john.doe"
        assert result.display_name == "John Doe"
        assert result.email == "john@example.com"
        assert result.is_bot is False

    @pytest.mark.asyncio
    async def test_get_user_info_not_found(self):
        """Should return None when user not found."""
        from aragora.connectors.chat.slack import SlackConnector

        connector = SlackConnector(bot_token="xoxb-test", use_circuit_breaker=False)

        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"ok": False, "error": "user_not_found"}

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(
                return_value=mock_response
            )

            result = await connector.get_user_info("invalid")

        assert result is None


class TestSlackWebhookParsing:
    """Tests for webhook verification and parsing."""

    def test_verify_webhook_valid_signature(self):
        """Should verify valid webhook signature."""
        from aragora.connectors.chat.slack import SlackConnector
        import time

        connector = SlackConnector(signing_secret="test-secret")

        timestamp = str(int(time.time()))
        body = b'{"type":"url_verification","challenge":"abc123"}'

        # Compute expected signature
        import hmac
        import hashlib

        sig_basestring = f"v0:{timestamp}:{body.decode('utf-8')}"
        expected_sig = (
            "v0="
            + hmac.new(
                b"test-secret",
                sig_basestring.encode(),
                hashlib.sha256,
            ).hexdigest()
        )

        headers = {
            "X-Slack-Request-Timestamp": timestamp,
            "X-Slack-Signature": expected_sig,
        }

        result = connector.verify_webhook(headers, body)
        assert result is True

    def test_verify_webhook_invalid_signature(self):
        """Should reject invalid webhook signature."""
        from aragora.connectors.chat.slack import SlackConnector
        import time

        connector = SlackConnector(signing_secret="test-secret")

        headers = {
            "X-Slack-Request-Timestamp": str(int(time.time())),
            "X-Slack-Signature": "v0=invalid_signature",
        }

        result = connector.verify_webhook(headers, b"test body")
        assert result is False

    def test_verify_webhook_expired_timestamp(self):
        """Should reject expired timestamps (replay attack protection)."""
        from aragora.connectors.chat.slack import SlackConnector
        import time

        connector = SlackConnector(signing_secret="test-secret")

        # Timestamp from 10 minutes ago
        old_timestamp = str(int(time.time()) - 600)

        headers = {
            "X-Slack-Request-Timestamp": old_timestamp,
            "X-Slack-Signature": "v0=any_signature",
        }

        result = connector.verify_webhook(headers, b"test body")
        assert result is False

    def test_parse_slash_command(self):
        """Should parse slash command webhook."""
        from aragora.connectors.chat.slack import SlackConnector
        from urllib.parse import urlencode

        connector = SlackConnector()

        body = urlencode(
            {
                "command": "/debate",
                "text": "topic argument",
                "user_id": "U123",
                "user_name": "testuser",
                "channel_id": "C456",
                "channel_name": "general",
                "team_id": "T789",
                "response_url": "https://hooks.slack.com/response/...",
                "trigger_id": "trigger123",
            }
        ).encode()

        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        event = connector.parse_webhook_event(headers, body)

        assert event.event_type == "slash_command"
        assert event.command is not None
        assert event.command.name == "debate"
        assert event.command.args == ["topic", "argument"]
        assert event.command.user.id == "U123"
        assert event.command.channel.id == "C456"

    def test_parse_button_click_interaction(self):
        """Should parse button click interaction."""
        from aragora.connectors.chat.slack import SlackConnector
        from urllib.parse import urlencode

        connector = SlackConnector()

        payload = {
            "type": "block_actions",
            "trigger_id": "trigger123",
            "user": {"id": "U123", "username": "testuser"},
            "channel": {"id": "C456", "name": "general"},
            "team": {"id": "T789"},
            "actions": [
                {
                    "type": "button",
                    "action_id": "approve_btn",
                    "value": "approved",
                }
            ],
            "message": {"ts": "1234567890.123456"},
            "response_url": "https://hooks.slack.com/response/...",
        }

        body = urlencode({"payload": json.dumps(payload)}).encode()
        headers = {"Content-Type": "application/x-www-form-urlencoded"}

        event = connector.parse_webhook_event(headers, body)

        assert event.event_type == "block_actions"
        assert event.interaction is not None
        assert event.interaction.action_id == "approve_btn"
        assert event.interaction.value == "approved"
        assert event.interaction.user.id == "U123"

    def test_parse_url_verification(self):
        """Should parse URL verification challenge."""
        from aragora.connectors.chat.slack import SlackConnector

        connector = SlackConnector()

        body = json.dumps(
            {
                "type": "url_verification",
                "challenge": "abc123xyz",
            }
        ).encode()

        headers = {"Content-Type": "application/json"}

        event = connector.parse_webhook_event(headers, body)

        assert event.event_type == "url_verification"
        assert event.challenge == "abc123xyz"

    def test_parse_message_event(self):
        """Should parse message event callback."""
        from aragora.connectors.chat.slack import SlackConnector

        connector = SlackConnector()

        body = json.dumps(
            {
                "type": "event_callback",
                "team_id": "T789",
                "event": {
                    "type": "message",
                    "user": "U123",
                    "channel": "C456",
                    "text": "Hello world",
                    "ts": "1234567890.123456",
                    "thread_ts": "1234567890.000001",
                },
            }
        ).encode()

        headers = {"Content-Type": "application/json"}

        event = connector.parse_webhook_event(headers, body)

        assert event.event_type == "message"
        assert event.message is not None
        assert event.message.content == "Hello world"
        assert event.message.author.id == "U123"
        assert event.message.thread_id == "1234567890.000001"


class TestSlackBlockKitFormatting:
    """Tests for Block Kit formatting methods."""

    def test_format_blocks_with_title(self):
        """Should format blocks with header title."""
        from aragora.connectors.chat.slack import SlackConnector

        connector = SlackConnector()

        blocks = connector.format_blocks(title="Important Alert")

        assert len(blocks) == 1
        assert blocks[0]["type"] == "header"
        assert blocks[0]["text"]["text"] == "Important Alert"

    def test_format_blocks_with_body(self):
        """Should format blocks with body section."""
        from aragora.connectors.chat.slack import SlackConnector

        connector = SlackConnector()

        blocks = connector.format_blocks(body="This is the message body")

        assert len(blocks) == 1
        assert blocks[0]["type"] == "section"
        assert blocks[0]["text"]["text"] == "This is the message body"
        assert blocks[0]["text"]["type"] == "mrkdwn"

    def test_format_blocks_with_fields(self):
        """Should format blocks with fields."""
        from aragora.connectors.chat.slack import SlackConnector

        connector = SlackConnector()

        blocks = connector.format_blocks(
            fields=[
                ("Status", "Active"),
                ("Priority", "High"),
            ]
        )

        assert len(blocks) == 1
        assert blocks[0]["type"] == "section"
        assert len(blocks[0]["fields"]) == 2
        assert "*Status*\nActive" in blocks[0]["fields"][0]["text"]

    def test_format_blocks_complete(self):
        """Should format complete block with all elements."""
        from aragora.connectors.chat.slack import SlackConnector
        from aragora.connectors.chat.models import MessageButton

        connector = SlackConnector()

        blocks = connector.format_blocks(
            title="Alert",
            body="Something happened",
            fields=[("Field", "Value")],
            actions=[
                MessageButton(text="Approve", action_id="approve", value="yes", style="primary"),
                MessageButton(text="Reject", action_id="reject", value="no", style="danger"),
            ],
        )

        assert len(blocks) == 4
        assert blocks[0]["type"] == "header"
        assert blocks[1]["type"] == "section"
        assert blocks[2]["type"] == "section"
        assert blocks[3]["type"] == "actions"
        assert len(blocks[3]["elements"]) == 2

    def test_format_button_primary(self):
        """Should format primary style button."""
        from aragora.connectors.chat.slack import SlackConnector

        connector = SlackConnector()

        button = connector.format_button(
            text="Submit",
            action_id="submit_btn",
            value="submitted",
            style="primary",
        )

        assert button["type"] == "button"
        assert button["text"]["text"] == "Submit"
        assert button["action_id"] == "submit_btn"
        assert button["value"] == "submitted"
        assert button["style"] == "primary"

    def test_format_button_with_url(self):
        """Should format URL button."""
        from aragora.connectors.chat.slack import SlackConnector

        connector = SlackConnector()

        button = connector.format_button(
            text="View Details",
            action_id="view_btn",
            url="https://example.com/details",
        )

        assert button["type"] == "button"
        assert button["url"] == "https://example.com/details"
        assert "action_id" not in button  # URL buttons don't have action_id


class TestSlackReactions:
    """Tests for Slack emoji reaction operations."""

    @pytest.mark.asyncio
    async def test_add_reaction_success(self):
        """Should add reaction to message."""
        from aragora.connectors.chat.slack import SlackConnector

        connector = SlackConnector(bot_token="xoxb-test")

        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": True}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value.__aenter__.return_value = mock_client

            result = await connector.add_reaction(
                channel_id="C123",
                message_id="1234567890.123456",
                emoji="thumbsup",
            )

            assert result is True
            mock_client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_reaction_already_reacted(self):
        """Should return True if already reacted."""
        from aragora.connectors.chat.slack import SlackConnector

        connector = SlackConnector(bot_token="xoxb-test")

        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": False, "error": "already_reacted"}
        mock_response.status_code = 200

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value.__aenter__.return_value = mock_client

            result = await connector.add_reaction(
                channel_id="C123",
                message_id="1234567890.123456",
                emoji=":thumbsup:",  # Test colon stripping
            )

            assert result is True

    @pytest.mark.asyncio
    async def test_remove_reaction_success(self):
        """Should remove reaction from message."""
        from aragora.connectors.chat.slack import SlackConnector

        connector = SlackConnector(bot_token="xoxb-test")

        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": True}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value.__aenter__.return_value = mock_client

            result = await connector.remove_reaction(
                channel_id="C123",
                message_id="1234567890.123456",
                emoji="thumbsup",
            )

            assert result is True


class TestSlackChannelUserDiscovery:
    """Tests for channel and user discovery."""

    @pytest.mark.asyncio
    async def test_list_channels_success(self):
        """Should list workspace channels."""
        from aragora.connectors.chat.slack import SlackConnector

        connector = SlackConnector(bot_token="xoxb-test")

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "ok": True,
            "channels": [
                {
                    "id": "C123",
                    "name": "general",
                    "is_private": False,
                    "is_archived": False,
                    "is_member": True,
                    "num_members": 42,
                    "topic": {"value": "General discussions"},
                    "purpose": {"value": "Company-wide announcements"},
                },
                {
                    "id": "C456",
                    "name": "random",
                    "is_private": False,
                    "is_archived": False,
                    "is_member": True,
                    "num_members": 35,
                },
            ],
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_class.return_value.__aenter__.return_value = mock_client

            channels = await connector.list_channels()

            assert len(channels) == 2
            assert channels[0].id == "C123"
            assert channels[0].name == "general"
            assert channels[0].metadata["num_members"] == 42

    @pytest.mark.asyncio
    async def test_list_users_success(self):
        """Should list workspace users."""
        from aragora.connectors.chat.slack import SlackConnector

        connector = SlackConnector(bot_token="xoxb-test")

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "ok": True,
            "members": [
                {
                    "id": "U123",
                    "name": "john.doe",
                    "is_bot": False,
                    "deleted": False,
                    "is_admin": True,
                    "is_owner": False,
                    "tz": "America/New_York",
                    "profile": {
                        "display_name": "John Doe",
                        "real_name": "John Doe",
                        "email": "john@example.com",
                        "title": "Engineer",
                        "image_72": "https://example.com/avatar.jpg",
                    },
                },
                {
                    "id": "UBOT",
                    "name": "slackbot",
                    "is_bot": True,
                    "deleted": False,
                    "profile": {},
                },
            ],
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_class.return_value.__aenter__.return_value = mock_client

            users = await connector.list_users(include_bots=False)

            # Should exclude bots
            assert len(users) == 1
            assert users[0].id == "U123"
            assert users[0].username == "john.doe"
            assert users[0].display_name == "John Doe"
            assert users[0].metadata["email"] == "john@example.com"

    @pytest.mark.asyncio
    async def test_list_users_include_bots(self):
        """Should include bots when requested."""
        from aragora.connectors.chat.slack import SlackConnector

        connector = SlackConnector(bot_token="xoxb-test")

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "ok": True,
            "members": [
                {"id": "U123", "name": "user", "is_bot": False, "deleted": False, "profile": {}},
                {"id": "UBOT", "name": "bot", "is_bot": True, "deleted": False, "profile": {}},
            ],
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_class.return_value.__aenter__.return_value = mock_client

            users = await connector.list_users(include_bots=True)

            assert len(users) == 2


class TestSlackMentionFormatting:
    """Tests for mention formatting helpers."""

    def test_format_user_mention(self):
        """Should format user mention correctly."""
        from aragora.connectors.chat.slack import SlackConnector

        mention = SlackConnector.format_user_mention("U123ABC")
        assert mention == "<@U123ABC>"

    def test_format_channel_mention(self):
        """Should format channel mention correctly."""
        from aragora.connectors.chat.slack import SlackConnector

        mention = SlackConnector.format_channel_mention("C456DEF")
        assert mention == "<#C456DEF>"


class TestSlackModalOperations:
    """Tests for modal/view operations."""

    @pytest.mark.asyncio
    async def test_open_modal_success(self):
        """Should open modal and return view ID."""
        from aragora.connectors.chat.slack import SlackConnector

        connector = SlackConnector(bot_token="xoxb-test")

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "ok": True,
            "view": {"id": "V123ABC"},
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value.__aenter__.return_value = mock_client

            view_id = await connector.open_modal(
                trigger_id="trigger123",
                view={
                    "type": "modal",
                    "title": {"type": "plain_text", "text": "Test Modal"},
                    "blocks": [],
                },
            )

            assert view_id == "V123ABC"

    @pytest.mark.asyncio
    async def test_update_modal_success(self):
        """Should update existing modal."""
        from aragora.connectors.chat.slack import SlackConnector

        connector = SlackConnector(bot_token="xoxb-test")

        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": True}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value.__aenter__.return_value = mock_client

            success = await connector.update_modal(
                view_id="V123ABC",
                view={"type": "modal", "blocks": []},
            )

            assert success is True


class TestSlackPinnedMessages:
    """Tests for pinned message operations."""

    @pytest.mark.asyncio
    async def test_pin_message_success(self):
        """Should pin message to channel."""
        from aragora.connectors.chat.slack import SlackConnector

        connector = SlackConnector(bot_token="xoxb-test")

        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": True}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value.__aenter__.return_value = mock_client

            result = await connector.pin_message(
                channel_id="C123",
                message_id="1234567890.123456",
            )

            assert result is True

    @pytest.mark.asyncio
    async def test_unpin_message_success(self):
        """Should unpin message from channel."""
        from aragora.connectors.chat.slack import SlackConnector

        connector = SlackConnector(bot_token="xoxb-test")

        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": True}

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = AsyncMock(return_value=mock_response)
            mock_client_class.return_value.__aenter__.return_value = mock_client

            result = await connector.unpin_message(
                channel_id="C123",
                message_id="1234567890.123456",
            )

            assert result is True

    @pytest.mark.asyncio
    async def test_get_pinned_messages_success(self):
        """Should get pinned messages in channel."""
        from aragora.connectors.chat.slack import SlackConnector

        connector = SlackConnector(bot_token="xoxb-test")

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "ok": True,
            "items": [
                {
                    "type": "message",
                    "message": {
                        "ts": "1234567890.123456",
                        "user": "U123",
                        "text": "This is pinned!",
                    },
                },
            ],
        }

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_class.return_value.__aenter__.return_value = mock_client

            messages = await connector.get_pinned_messages(channel_id="C123")

            assert len(messages) == 1
            assert messages[0].content == "This is pinned!"
            assert messages[0].metadata.get("pinned") is True
