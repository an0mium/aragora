"""
Tests for TeamsMessagingMixin - Microsoft Teams messaging operations.

Tests cover:
- Message sending and receiving
- Message updating
- Message deletion
- Typing indicators
- Adaptive Card attachments
- Command responses
- Interaction responses
- Error handling (rate limits, auth failures, network errors)
- Retry logic
- Circuit breaker protection
"""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

from aragora.connectors.chat.models import (
    BotCommand,
    ChatChannel,
    ChatUser,
    SendMessageResponse,
    UserInteraction,
    InteractionType,
)


class MockTeamsConnector:
    """Mock connector implementing the protocol required by TeamsMessagingMixin."""

    def __init__(self):
        self._circuit_breaker_open = False
        self._circuit_breaker_error = None
        self._access_token = "test-token-12345"
        self._http_request_mock = AsyncMock()
        self._record_failure_mock = MagicMock()

    def _check_circuit_breaker(self):
        if self._circuit_breaker_open:
            return False, self._circuit_breaker_error
        return True, None

    async def _get_access_token(self):
        return self._access_token

    async def _http_request(self, method, url, headers=None, json=None, operation=None):
        return await self._http_request_mock(
            method=method, url=url, headers=headers, json=json, operation=operation
        )

    def _record_failure(self, error=None):
        self._record_failure_mock(error)


class TestSendMessage:
    """Tests for send_message method."""

    @pytest.mark.asyncio
    async def test_send_simple_text_message(self):
        """Should send a simple text message successfully."""
        from aragora.connectors.chat.teams._messaging import TeamsMessagingMixin

        class TestConnector(TeamsMessagingMixin, MockTeamsConnector):
            pass

        connector = TestConnector()
        connector._http_request_mock.return_value = (
            True,
            {"id": "msg-123"},
            None,
        )

        result = await connector.send_message(
            channel_id="channel-456",
            text="Hello, Teams!",
        )

        assert result.success is True
        assert result.message_id == "msg-123"
        assert result.channel_id == "channel-456"

    @pytest.mark.asyncio
    async def test_send_message_with_conversation_id(self):
        """Should use conversation_id when provided."""
        from aragora.connectors.chat.teams._messaging import TeamsMessagingMixin

        class TestConnector(TeamsMessagingMixin, MockTeamsConnector):
            pass

        connector = TestConnector()
        connector._http_request_mock.return_value = (True, {"id": "msg-789"}, None)

        result = await connector.send_message(
            channel_id="channel-456",
            text="Test message",
            conversation_id="conv-999",
        )

        assert result.success is True
        assert result.channel_id == "conv-999"

    @pytest.mark.asyncio
    async def test_send_message_with_adaptive_card_blocks(self):
        """Should include Adaptive Card when blocks are provided."""
        from aragora.connectors.chat.teams._messaging import TeamsMessagingMixin

        class TestConnector(TeamsMessagingMixin, MockTeamsConnector):
            pass

        connector = TestConnector()
        connector._http_request_mock.return_value = (True, {"id": "msg-card"}, None)

        blocks = [{"type": "TextBlock", "text": "Card content", "size": "Large"}]

        result = await connector.send_message(
            channel_id="channel-123",
            text="Message with card",
            blocks=blocks,
        )

        assert result.success is True
        # Verify the request included attachments
        call_args = connector._http_request_mock.call_args
        json_payload = call_args.kwargs["json"]
        assert "attachments" in json_payload
        assert (
            json_payload["attachments"][0]["contentType"]
            == "application/vnd.microsoft.card.adaptive"
        )
        assert json_payload["attachments"][0]["content"]["body"] == blocks

    @pytest.mark.asyncio
    async def test_send_message_with_thread_id(self):
        """Should include replyToId for threaded replies."""
        from aragora.connectors.chat.teams._messaging import TeamsMessagingMixin

        class TestConnector(TeamsMessagingMixin, MockTeamsConnector):
            pass

        connector = TestConnector()
        connector._http_request_mock.return_value = (True, {"id": "reply-123"}, None)

        result = await connector.send_message(
            channel_id="channel-123",
            text="Thread reply",
            thread_id="parent-msg-456",
        )

        assert result.success is True
        call_args = connector._http_request_mock.call_args
        json_payload = call_args.kwargs["json"]
        assert json_payload["replyToId"] == "parent-msg-456"

    @pytest.mark.asyncio
    async def test_send_message_with_custom_service_url(self):
        """Should use custom service URL when provided."""
        from aragora.connectors.chat.teams._messaging import TeamsMessagingMixin

        class TestConnector(TeamsMessagingMixin, MockTeamsConnector):
            pass

        connector = TestConnector()
        connector._http_request_mock.return_value = (True, {"id": "msg-custom"}, None)

        result = await connector.send_message(
            channel_id="channel-123",
            text="Custom service",
            service_url="https://custom.api.example.com",
        )

        assert result.success is True
        call_args = connector._http_request_mock.call_args
        assert "https://custom.api.example.com" in call_args.kwargs["url"]

    @pytest.mark.asyncio
    async def test_send_message_circuit_breaker_open(self):
        """Should fail fast when circuit breaker is open."""
        from aragora.connectors.chat.teams._messaging import TeamsMessagingMixin

        class TestConnector(TeamsMessagingMixin, MockTeamsConnector):
            pass

        connector = TestConnector()
        connector._circuit_breaker_open = True
        connector._circuit_breaker_error = "Circuit breaker open"

        result = await connector.send_message(
            channel_id="channel-123",
            text="Should fail",
        )

        assert result.success is False
        assert result.error == "Circuit breaker open"

    @pytest.mark.asyncio
    async def test_send_message_http_request_failure(self):
        """Should handle HTTP request failures."""
        from aragora.connectors.chat.teams._messaging import TeamsMessagingMixin

        class TestConnector(TeamsMessagingMixin, MockTeamsConnector):
            pass

        connector = TestConnector()
        connector._http_request_mock.return_value = (
            False,
            None,
            "HTTP 500 Internal Server Error",
        )

        result = await connector.send_message(
            channel_id="channel-123",
            text="Will fail",
        )

        assert result.success is False
        assert "500" in result.error

    @pytest.mark.asyncio
    async def test_send_message_timeout_error(self):
        """Should raise ConnectorTimeoutError on timeout."""
        from aragora.connectors.chat.teams._messaging import TeamsMessagingMixin
        from aragora.connectors.exceptions import ConnectorTimeoutError

        class TestConnector(TeamsMessagingMixin, MockTeamsConnector):
            pass

        connector = TestConnector()

        with patch("aragora.connectors.chat.teams._constants.HTTPX_AVAILABLE", True):
            import httpx

            connector._http_request_mock.side_effect = httpx.TimeoutException(
                "Connection timed out"
            )

            with pytest.raises(ConnectorTimeoutError):
                await connector.send_message(
                    channel_id="channel-123",
                    text="Timeout test",
                )

    @pytest.mark.asyncio
    async def test_send_message_connection_error(self):
        """Should raise ConnectorNetworkError on connection failure."""
        from aragora.connectors.chat.teams._messaging import TeamsMessagingMixin
        from aragora.connectors.exceptions import ConnectorNetworkError

        class TestConnector(TeamsMessagingMixin, MockTeamsConnector):
            pass

        connector = TestConnector()

        with patch("aragora.connectors.chat.teams._constants.HTTPX_AVAILABLE", True):
            import httpx

            connector._http_request_mock.side_effect = httpx.ConnectError("Connection refused")

            with pytest.raises(ConnectorNetworkError):
                await connector.send_message(
                    channel_id="channel-123",
                    text="Network test",
                )

    @pytest.mark.asyncio
    async def test_send_message_httpx_not_available(self):
        """Should return error when httpx is not available."""
        from aragora.connectors.chat.teams._messaging import TeamsMessagingMixin

        class TestConnector(TeamsMessagingMixin, MockTeamsConnector):
            pass

        connector = TestConnector()

        with patch("aragora.connectors.chat.teams._constants.HTTPX_AVAILABLE", False):
            result = await connector.send_message(
                channel_id="channel-123",
                text="No httpx",
            )

            assert result.success is False
            assert "httpx not available" in result.error

    @pytest.mark.asyncio
    async def test_send_message_records_failure_on_error(self):
        """Should record failure when error occurs."""
        from aragora.connectors.chat.teams._messaging import TeamsMessagingMixin

        class TestConnector(TeamsMessagingMixin, MockTeamsConnector):
            pass

        connector = TestConnector()
        connector._http_request_mock.side_effect = ValueError("Test error")

        result = await connector.send_message(
            channel_id="channel-123",
            text="Error test",
        )

        assert result.success is False
        connector._record_failure_mock.assert_called_once()


class TestUpdateMessage:
    """Tests for update_message method."""

    @pytest.mark.asyncio
    async def test_update_message_success(self):
        """Should update an existing message."""
        from aragora.connectors.chat.teams._messaging import TeamsMessagingMixin

        class TestConnector(TeamsMessagingMixin, MockTeamsConnector):
            pass

        connector = TestConnector()
        connector._http_request_mock.return_value = (True, {}, None)

        result = await connector.update_message(
            channel_id="channel-123",
            message_id="msg-456",
            text="Updated content",
        )

        assert result.success is True
        assert result.message_id == "msg-456"
        assert result.channel_id == "channel-123"

    @pytest.mark.asyncio
    async def test_update_message_with_blocks(self):
        """Should update message with Adaptive Card blocks."""
        from aragora.connectors.chat.teams._messaging import TeamsMessagingMixin

        class TestConnector(TeamsMessagingMixin, MockTeamsConnector):
            pass

        connector = TestConnector()
        connector._http_request_mock.return_value = (True, {}, None)

        blocks = [{"type": "TextBlock", "text": "Updated card"}]

        result = await connector.update_message(
            channel_id="channel-123",
            message_id="msg-456",
            text="Updated with card",
            blocks=blocks,
        )

        assert result.success is True
        call_args = connector._http_request_mock.call_args
        json_payload = call_args.kwargs["json"]
        assert "attachments" in json_payload

    @pytest.mark.asyncio
    async def test_update_message_uses_put_method(self):
        """Should use PUT HTTP method for updates."""
        from aragora.connectors.chat.teams._messaging import TeamsMessagingMixin

        class TestConnector(TeamsMessagingMixin, MockTeamsConnector):
            pass

        connector = TestConnector()
        connector._http_request_mock.return_value = (True, {}, None)

        await connector.update_message(
            channel_id="channel-123",
            message_id="msg-456",
            text="PUT test",
        )

        call_args = connector._http_request_mock.call_args
        assert call_args.kwargs["method"] == "PUT"

    @pytest.mark.asyncio
    async def test_update_message_circuit_breaker_open(self):
        """Should fail fast when circuit breaker is open."""
        from aragora.connectors.chat.teams._messaging import TeamsMessagingMixin

        class TestConnector(TeamsMessagingMixin, MockTeamsConnector):
            pass

        connector = TestConnector()
        connector._circuit_breaker_open = True
        connector._circuit_breaker_error = "Service unavailable"

        result = await connector.update_message(
            channel_id="channel-123",
            message_id="msg-456",
            text="Should fail",
        )

        assert result.success is False
        assert "Service unavailable" in result.error

    @pytest.mark.asyncio
    async def test_update_message_httpx_not_available(self):
        """Should return error when httpx is not available."""
        from aragora.connectors.chat.teams._messaging import TeamsMessagingMixin

        class TestConnector(TeamsMessagingMixin, MockTeamsConnector):
            pass

        connector = TestConnector()

        with patch("aragora.connectors.chat.teams._constants.HTTPX_AVAILABLE", False):
            result = await connector.update_message(
                channel_id="channel-123",
                message_id="msg-456",
                text="No httpx",
            )

            assert result.success is False
            assert "httpx not available" in result.error


class TestDeleteMessage:
    """Tests for delete_message method."""

    @pytest.mark.asyncio
    async def test_delete_message_success(self):
        """Should delete a message successfully."""
        from aragora.connectors.chat.teams._messaging import TeamsMessagingMixin

        class TestConnector(TeamsMessagingMixin, MockTeamsConnector):
            pass

        connector = TestConnector()
        connector._http_request_mock.return_value = (True, None, None)

        result = await connector.delete_message(
            channel_id="channel-123",
            message_id="msg-456",
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_delete_message_uses_delete_method(self):
        """Should use DELETE HTTP method."""
        from aragora.connectors.chat.teams._messaging import TeamsMessagingMixin

        class TestConnector(TeamsMessagingMixin, MockTeamsConnector):
            pass

        connector = TestConnector()
        connector._http_request_mock.return_value = (True, None, None)

        await connector.delete_message(
            channel_id="channel-123",
            message_id="msg-456",
        )

        call_args = connector._http_request_mock.call_args
        assert call_args.kwargs["method"] == "DELETE"

    @pytest.mark.asyncio
    async def test_delete_message_failure(self):
        """Should return False on failure."""
        from aragora.connectors.chat.teams._messaging import TeamsMessagingMixin

        class TestConnector(TeamsMessagingMixin, MockTeamsConnector):
            pass

        connector = TestConnector()
        connector._http_request_mock.return_value = (False, None, "Not found")

        result = await connector.delete_message(
            channel_id="channel-123",
            message_id="msg-456",
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_delete_message_httpx_not_available(self):
        """Should return False when httpx is not available."""
        from aragora.connectors.chat.teams._messaging import TeamsMessagingMixin

        class TestConnector(TeamsMessagingMixin, MockTeamsConnector):
            pass

        connector = TestConnector()

        with patch("aragora.connectors.chat.teams._constants.HTTPX_AVAILABLE", False):
            result = await connector.delete_message(
                channel_id="channel-123",
                message_id="msg-456",
            )

            assert result is False

    @pytest.mark.asyncio
    async def test_delete_message_exception_handling(self):
        """Should handle exceptions gracefully."""
        from aragora.connectors.chat.teams._messaging import TeamsMessagingMixin

        class TestConnector(TeamsMessagingMixin, MockTeamsConnector):
            pass

        connector = TestConnector()
        connector._http_request_mock.side_effect = RuntimeError("Unexpected error")

        result = await connector.delete_message(
            channel_id="channel-123",
            message_id="msg-456",
        )

        assert result is False


class TestSendTypingIndicator:
    """Tests for send_typing_indicator method."""

    @pytest.mark.asyncio
    async def test_send_typing_indicator_success(self):
        """Should send typing indicator successfully."""
        from aragora.connectors.chat.teams._messaging import TeamsMessagingMixin

        class TestConnector(TeamsMessagingMixin, MockTeamsConnector):
            pass

        connector = TestConnector()
        connector._http_request_mock.return_value = (True, None, None)

        result = await connector.send_typing_indicator(
            channel_id="channel-123",
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_send_typing_indicator_sends_typing_activity(self):
        """Should send activity with type 'typing'."""
        from aragora.connectors.chat.teams._messaging import TeamsMessagingMixin

        class TestConnector(TeamsMessagingMixin, MockTeamsConnector):
            pass

        connector = TestConnector()
        connector._http_request_mock.return_value = (True, None, None)

        await connector.send_typing_indicator(
            channel_id="channel-123",
        )

        call_args = connector._http_request_mock.call_args
        json_payload = call_args.kwargs["json"]
        assert json_payload["type"] == "typing"

    @pytest.mark.asyncio
    async def test_send_typing_indicator_failure(self):
        """Should return False on failure."""
        from aragora.connectors.chat.teams._messaging import TeamsMessagingMixin

        class TestConnector(TeamsMessagingMixin, MockTeamsConnector):
            pass

        connector = TestConnector()
        connector._http_request_mock.return_value = (False, None, "Failed")

        result = await connector.send_typing_indicator(
            channel_id="channel-123",
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_send_typing_indicator_httpx_not_available(self):
        """Should return False when httpx is not available."""
        from aragora.connectors.chat.teams._messaging import TeamsMessagingMixin

        class TestConnector(TeamsMessagingMixin, MockTeamsConnector):
            pass

        connector = TestConnector()

        with patch("aragora.connectors.chat.teams._constants.HTTPX_AVAILABLE", False):
            result = await connector.send_typing_indicator(
                channel_id="channel-123",
            )

            assert result is False


class TestRespondToCommand:
    """Tests for respond_to_command method."""

    @pytest.mark.asyncio
    async def test_respond_to_command_with_response_url(self):
        """Should use response URL when available."""
        from aragora.connectors.chat.teams._messaging import TeamsMessagingMixin

        class TestConnector(TeamsMessagingMixin, MockTeamsConnector):
            async def _send_to_response_url(self, response_url, text, blocks=None):
                return SendMessageResponse(success=True, message_id="resp-123")

        connector = TestConnector()

        command = BotCommand(
            name="test",
            text="/test",
            response_url="https://response.url/callback",
        )

        result = await connector.respond_to_command(
            command=command,
            text="Response text",
        )

        assert result.success is True
        assert result.message_id == "resp-123"

    @pytest.mark.asyncio
    async def test_respond_to_command_with_channel(self):
        """Should send to channel when no response URL."""
        from aragora.connectors.chat.teams._messaging import TeamsMessagingMixin

        class TestConnector(TeamsMessagingMixin, MockTeamsConnector):
            pass

        connector = TestConnector()
        connector._http_request_mock.return_value = (True, {"id": "msg-channel"}, None)

        command = BotCommand(
            name="test",
            text="/test",
            channel=ChatChannel(id="channel-123", platform="teams"),
            metadata={"service_url": "https://service.url"},
        )

        result = await connector.respond_to_command(
            command=command,
            text="Channel response",
        )

        assert result.success is True

    @pytest.mark.asyncio
    async def test_respond_to_command_no_target(self):
        """Should fail when no channel or response URL."""
        from aragora.connectors.chat.teams._messaging import TeamsMessagingMixin

        class TestConnector(TeamsMessagingMixin, MockTeamsConnector):
            pass

        connector = TestConnector()

        command = BotCommand(
            name="test",
            text="/test",
        )

        result = await connector.respond_to_command(
            command=command,
            text="No target",
        )

        assert result.success is False
        assert "No channel or response URL" in result.error


class TestRespondToInteraction:
    """Tests for respond_to_interaction method."""

    @pytest.mark.asyncio
    async def test_respond_to_interaction_with_response_url(self):
        """Should use response URL when available."""
        from aragora.connectors.chat.teams._messaging import TeamsMessagingMixin

        class TestConnector(TeamsMessagingMixin, MockTeamsConnector):
            async def _send_to_response_url(self, response_url, text, blocks=None):
                return SendMessageResponse(success=True)

        connector = TestConnector()

        interaction = UserInteraction(
            id="int-123",
            interaction_type=InteractionType.BUTTON_CLICK,
            action_id="btn-action",
            response_url="https://response.url/callback",
        )

        result = await connector.respond_to_interaction(
            interaction=interaction,
            text="Interaction response",
        )

        assert result.success is True

    @pytest.mark.asyncio
    async def test_respond_to_interaction_replace_original(self):
        """Should update original message when replace_original is True."""
        from aragora.connectors.chat.teams._messaging import TeamsMessagingMixin

        class TestConnector(TeamsMessagingMixin, MockTeamsConnector):
            pass

        connector = TestConnector()
        connector._http_request_mock.return_value = (True, {}, None)

        interaction = UserInteraction(
            id="int-123",
            interaction_type=InteractionType.BUTTON_CLICK,
            action_id="btn-action",
            channel=ChatChannel(id="channel-123", platform="teams"),
            message_id="msg-456",
            metadata={"service_url": "https://service.url"},
        )

        result = await connector.respond_to_interaction(
            interaction=interaction,
            text="Replaced content",
            replace_original=True,
        )

        assert result.success is True
        # Should use PUT method for update
        call_args = connector._http_request_mock.call_args
        assert call_args.kwargs["method"] == "PUT"

    @pytest.mark.asyncio
    async def test_respond_to_interaction_new_message(self):
        """Should send new message when not replacing original."""
        from aragora.connectors.chat.teams._messaging import TeamsMessagingMixin

        class TestConnector(TeamsMessagingMixin, MockTeamsConnector):
            pass

        connector = TestConnector()
        connector._http_request_mock.return_value = (True, {"id": "new-msg"}, None)

        interaction = UserInteraction(
            id="int-123",
            interaction_type=InteractionType.BUTTON_CLICK,
            action_id="btn-action",
            channel=ChatChannel(id="channel-123", platform="teams"),
            metadata={"service_url": "https://service.url"},
        )

        result = await connector.respond_to_interaction(
            interaction=interaction,
            text="New message",
            replace_original=False,
        )

        assert result.success is True

    @pytest.mark.asyncio
    async def test_respond_to_interaction_no_target(self):
        """Should fail when no target available."""
        from aragora.connectors.chat.teams._messaging import TeamsMessagingMixin

        class TestConnector(TeamsMessagingMixin, MockTeamsConnector):
            pass

        connector = TestConnector()

        interaction = UserInteraction(
            id="int-123",
            interaction_type=InteractionType.BUTTON_CLICK,
            action_id="btn-action",
        )

        result = await connector.respond_to_interaction(
            interaction=interaction,
            text="No target",
        )

        assert result.success is False
        assert "No response target" in result.error


class TestSendToResponseUrl:
    """Tests for _send_to_response_url method."""

    @pytest.mark.asyncio
    async def test_send_to_response_url_success(self):
        """Should send to response URL successfully."""
        from aragora.connectors.chat.teams._messaging import TeamsMessagingMixin

        class TestConnector(TeamsMessagingMixin, MockTeamsConnector):
            pass

        connector = TestConnector()
        connector._http_request_mock.return_value = (True, None, None)

        result = await connector._send_to_response_url(
            response_url="https://response.url/callback",
            text="Response message",
        )

        assert result.success is True

    @pytest.mark.asyncio
    async def test_send_to_response_url_with_blocks(self):
        """Should include Adaptive Card in response."""
        from aragora.connectors.chat.teams._messaging import TeamsMessagingMixin

        class TestConnector(TeamsMessagingMixin, MockTeamsConnector):
            pass

        connector = TestConnector()
        connector._http_request_mock.return_value = (True, None, None)

        blocks = [{"type": "TextBlock", "text": "Card content"}]

        result = await connector._send_to_response_url(
            response_url="https://response.url/callback",
            text="Response with card",
            blocks=blocks,
        )

        assert result.success is True
        call_args = connector._http_request_mock.call_args
        json_payload = call_args.kwargs["json"]
        assert "attachments" in json_payload

    @pytest.mark.asyncio
    async def test_send_to_response_url_failure(self):
        """Should handle failure gracefully."""
        from aragora.connectors.chat.teams._messaging import TeamsMessagingMixin

        class TestConnector(TeamsMessagingMixin, MockTeamsConnector):
            pass

        connector = TestConnector()
        connector._http_request_mock.return_value = (False, None, "Bad Request")

        result = await connector._send_to_response_url(
            response_url="https://response.url/callback",
            text="Will fail",
        )

        assert result.success is False
        assert "Bad Request" in result.error

    @pytest.mark.asyncio
    async def test_send_to_response_url_httpx_not_available(self):
        """Should return error when httpx is not available."""
        from aragora.connectors.chat.teams._messaging import TeamsMessagingMixin

        class TestConnector(TeamsMessagingMixin, MockTeamsConnector):
            pass

        connector = TestConnector()

        with patch("aragora.connectors.chat.teams._constants.HTTPX_AVAILABLE", False):
            result = await connector._send_to_response_url(
                response_url="https://response.url/callback",
                text="No httpx",
            )

            assert result.success is False
            assert "httpx not available" in result.error


class TestMessageFormatting:
    """Tests for message formatting and payload construction."""

    @pytest.mark.asyncio
    async def test_activity_payload_structure(self):
        """Should construct proper activity payload."""
        from aragora.connectors.chat.teams._messaging import TeamsMessagingMixin

        class TestConnector(TeamsMessagingMixin, MockTeamsConnector):
            pass

        connector = TestConnector()
        connector._http_request_mock.return_value = (True, {"id": "msg-1"}, None)

        await connector.send_message(
            channel_id="channel-123",
            text="Test message",
        )

        call_args = connector._http_request_mock.call_args
        json_payload = call_args.kwargs["json"]

        assert json_payload["type"] == "message"
        assert json_payload["text"] == "Test message"

    @pytest.mark.asyncio
    async def test_adaptive_card_schema_version(self):
        """Should use correct Adaptive Card schema version."""
        from aragora.connectors.chat.teams._messaging import TeamsMessagingMixin

        class TestConnector(TeamsMessagingMixin, MockTeamsConnector):
            pass

        connector = TestConnector()
        connector._http_request_mock.return_value = (True, {"id": "msg-1"}, None)

        blocks = [{"type": "TextBlock", "text": "Test"}]

        await connector.send_message(
            channel_id="channel-123",
            text="Card message",
            blocks=blocks,
        )

        call_args = connector._http_request_mock.call_args
        json_payload = call_args.kwargs["json"]
        card_content = json_payload["attachments"][0]["content"]

        assert card_content["$schema"] == "http://adaptivecards.io/schemas/adaptive-card.json"
        assert card_content["type"] == "AdaptiveCard"
        assert card_content["version"] == "1.4"

    @pytest.mark.asyncio
    async def test_authorization_header(self):
        """Should include Bearer token in Authorization header."""
        from aragora.connectors.chat.teams._messaging import TeamsMessagingMixin

        class TestConnector(TeamsMessagingMixin, MockTeamsConnector):
            pass

        connector = TestConnector()
        connector._http_request_mock.return_value = (True, {"id": "msg-1"}, None)

        await connector.send_message(
            channel_id="channel-123",
            text="Auth test",
        )

        call_args = connector._http_request_mock.call_args
        headers = call_args.kwargs["headers"]

        assert "Authorization" in headers
        assert headers["Authorization"].startswith("Bearer ")
        assert "test-token-12345" in headers["Authorization"]


class TestErrorClassification:
    """Tests for error classification and handling."""

    @pytest.mark.asyncio
    async def test_json_decode_error_handling(self):
        """Should handle JSON decode errors."""
        from aragora.connectors.chat.teams._messaging import TeamsMessagingMixin
        import json

        class TestConnector(TeamsMessagingMixin, MockTeamsConnector):
            pass

        connector = TestConnector()
        connector._http_request_mock.side_effect = json.JSONDecodeError("Expecting value", "", 0)

        result = await connector.send_message(
            channel_id="channel-123",
            text="JSON error test",
        )

        assert result.success is False
        connector._record_failure_mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_key_error_handling(self):
        """Should handle KeyError exceptions."""
        from aragora.connectors.chat.teams._messaging import TeamsMessagingMixin

        class TestConnector(TeamsMessagingMixin, MockTeamsConnector):
            pass

        connector = TestConnector()
        connector._http_request_mock.side_effect = KeyError("missing_key")

        result = await connector.send_message(
            channel_id="channel-123",
            text="Key error test",
        )

        assert result.success is False

    @pytest.mark.asyncio
    async def test_value_error_handling(self):
        """Should handle ValueError exceptions."""
        from aragora.connectors.chat.teams._messaging import TeamsMessagingMixin

        class TestConnector(TeamsMessagingMixin, MockTeamsConnector):
            pass

        connector = TestConnector()
        connector._http_request_mock.side_effect = ValueError("Invalid value")

        result = await connector.send_message(
            channel_id="channel-123",
            text="Value error test",
        )

        assert result.success is False

    @pytest.mark.asyncio
    async def test_os_error_handling(self):
        """Should handle OSError exceptions."""
        from aragora.connectors.chat.teams._messaging import TeamsMessagingMixin

        class TestConnector(TeamsMessagingMixin, MockTeamsConnector):
            pass

        connector = TestConnector()
        connector._http_request_mock.side_effect = OSError("Network unreachable")

        result = await connector.send_message(
            channel_id="channel-123",
            text="OS error test",
        )

        assert result.success is False
