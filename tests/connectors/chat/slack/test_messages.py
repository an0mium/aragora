"""
Tests for Slack messages module - message sending, formatting, and file operations.

Tests cover:
1. Message composition (text, blocks, attachments)
2. Threading (reply_to, thread_ts handling)
3. Message formatting (markdown, mentions, links)
4. Block elements (buttons, selects, inputs)
5. Message sending (post_message, update_message, delete_message)
6. Message reactions (add_reaction, remove_reaction)
7. Ephemeral messages
8. Error handling (invalid channel, rate limits, permission errors)
9. File attachments (upload, download)
10. Channel and user info
11. Modal/view operations
12. Pinned messages
13. Evidence collection
"""

from __future__ import annotations

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_slack_connector():
    """Create a mock connector with SlackMessagesMixin methods."""
    from aragora.connectors.chat.slack.messages import SlackMessagesMixin

    class MockConnector(SlackMessagesMixin):
        def __init__(self):
            self.bot_token = "xoxb-test-token"
            self._circuit_breaker = None
            self._max_retries = 3
            self._timeout = 30.0
            self._slack_api_request = AsyncMock()
            self._http_request = AsyncMock()

        @property
        def platform_name(self) -> str:
            return "slack"

        def _get_headers(self) -> dict[str, str]:
            return {"Authorization": f"Bearer {self.bot_token}"}

        def _compute_message_relevance(
            self,
            message: Any,
            query: str | None = None,
        ) -> float:
            return 1.0

    return MockConnector()


@pytest.fixture
def mock_connector_with_circuit_breaker(mock_slack_connector):
    """Create a mock connector with circuit breaker enabled."""
    mock_slack_connector._circuit_breaker = MagicMock()
    mock_slack_connector._circuit_breaker.can_proceed.return_value = True
    mock_slack_connector._circuit_breaker.record_success = MagicMock()
    mock_slack_connector._circuit_breaker.record_failure = MagicMock()
    return mock_slack_connector


# ---------------------------------------------------------------------------
# Send Message Tests
# ---------------------------------------------------------------------------


class TestSendMessage:
    """Tests for send_message method."""

    @pytest.mark.asyncio
    async def test_send_message_success(self, mock_slack_connector):
        """Should send message and return success response."""
        mock_slack_connector._slack_api_request.return_value = (
            True,
            {"ts": "1704067200.123456", "channel": "C12345"},
            None,
        )

        response = await mock_slack_connector.send_message(
            channel_id="C12345",
            text="Hello, world!",
        )

        assert response.success is True
        assert response.message_id == "1704067200.123456"
        assert response.channel_id == "C12345"
        assert response.timestamp == "1704067200.123456"

    @pytest.mark.asyncio
    async def test_send_message_with_blocks(self, mock_slack_connector):
        """Should include blocks in message payload."""
        mock_slack_connector._slack_api_request.return_value = (
            True,
            {"ts": "1.0", "channel": "C1"},
            None,
        )

        blocks = [{"type": "section", "text": {"type": "mrkdwn", "text": "Bold"}}]
        await mock_slack_connector.send_message(
            channel_id="C12345",
            text="Fallback text",
            blocks=blocks,
        )

        call_args = mock_slack_connector._slack_api_request.call_args
        payload = call_args[0][1]
        assert payload["blocks"] == blocks

    @pytest.mark.asyncio
    async def test_send_message_with_thread_id(self, mock_slack_connector):
        """Should include thread_ts for threaded replies."""
        mock_slack_connector._slack_api_request.return_value = (
            True,
            {"ts": "1.0", "channel": "C1"},
            None,
        )

        await mock_slack_connector.send_message(
            channel_id="C12345",
            text="Reply in thread",
            thread_id="1704067200.000001",
        )

        call_args = mock_slack_connector._slack_api_request.call_args
        payload = call_args[0][1]
        assert payload["thread_ts"] == "1704067200.000001"

    @pytest.mark.asyncio
    async def test_send_message_with_unfurl_options(self, mock_slack_connector):
        """Should pass unfurl_links and unfurl_media options."""
        mock_slack_connector._slack_api_request.return_value = (
            True,
            {"ts": "1.0", "channel": "C1"},
            None,
        )

        await mock_slack_connector.send_message(
            channel_id="C12345",
            text="Check this link",
            unfurl_links=True,
            unfurl_media=False,
        )

        call_args = mock_slack_connector._slack_api_request.call_args
        payload = call_args[0][1]
        assert payload["unfurl_links"] is True
        assert payload["unfurl_media"] is False

    @pytest.mark.asyncio
    async def test_send_message_failure(self, mock_slack_connector):
        """Should return failure response on API error."""
        mock_slack_connector._slack_api_request.return_value = (
            False,
            None,
            "channel_not_found",
        )

        response = await mock_slack_connector.send_message(
            channel_id="C_INVALID",
            text="Test",
        )

        assert response.success is False
        assert response.error == "channel_not_found"

    @pytest.mark.asyncio
    async def test_send_message_calls_correct_endpoint(self, mock_slack_connector):
        """Should call chat.postMessage endpoint."""
        mock_slack_connector._slack_api_request.return_value = (
            True,
            {"ts": "1.0", "channel": "C1"},
            None,
        )

        await mock_slack_connector.send_message(
            channel_id="C12345",
            text="Test",
        )

        call_args = mock_slack_connector._slack_api_request.call_args
        assert call_args[0][0] == "chat.postMessage"
        assert call_args[0][2] == "send_message"


# ---------------------------------------------------------------------------
# Update Message Tests
# ---------------------------------------------------------------------------


class TestUpdateMessage:
    """Tests for update_message method."""

    @pytest.mark.asyncio
    async def test_update_message_success(self, mock_slack_connector):
        """Should update message and return success response."""
        mock_slack_connector._slack_api_request.return_value = (
            True,
            {"ts": "1704067200.123456", "channel": "C12345"},
            None,
        )

        response = await mock_slack_connector.update_message(
            channel_id="C12345",
            message_id="1704067200.123456",
            text="Updated text",
        )

        assert response.success is True
        assert response.message_id == "1704067200.123456"
        assert response.channel_id == "C12345"

    @pytest.mark.asyncio
    async def test_update_message_with_blocks(self, mock_slack_connector):
        """Should include blocks in update payload."""
        mock_slack_connector._slack_api_request.return_value = (
            True,
            {"ts": "1.0", "channel": "C1"},
            None,
        )

        blocks = [{"type": "section", "text": {"type": "plain_text", "text": "New"}}]
        await mock_slack_connector.update_message(
            channel_id="C12345",
            message_id="1.0",
            text="Updated",
            blocks=blocks,
        )

        call_args = mock_slack_connector._slack_api_request.call_args
        payload = call_args[0][1]
        assert payload["blocks"] == blocks

    @pytest.mark.asyncio
    async def test_update_message_includes_ts(self, mock_slack_connector):
        """Should include message ts in payload."""
        mock_slack_connector._slack_api_request.return_value = (
            True,
            {"ts": "1.0", "channel": "C1"},
            None,
        )

        await mock_slack_connector.update_message(
            channel_id="C12345",
            message_id="1704067200.123456",
            text="Updated",
        )

        call_args = mock_slack_connector._slack_api_request.call_args
        payload = call_args[0][1]
        assert payload["ts"] == "1704067200.123456"

    @pytest.mark.asyncio
    async def test_update_message_failure(self, mock_slack_connector):
        """Should return failure on API error."""
        mock_slack_connector._slack_api_request.return_value = (
            False,
            None,
            "message_not_found",
        )

        response = await mock_slack_connector.update_message(
            channel_id="C12345",
            message_id="invalid",
            text="Updated",
        )

        assert response.success is False
        assert response.error == "message_not_found"

    @pytest.mark.asyncio
    async def test_update_message_calls_correct_endpoint(self, mock_slack_connector):
        """Should call chat.update endpoint."""
        mock_slack_connector._slack_api_request.return_value = (
            True,
            {"ts": "1.0", "channel": "C1"},
            None,
        )

        await mock_slack_connector.update_message(
            channel_id="C12345",
            message_id="1.0",
            text="Test",
        )

        call_args = mock_slack_connector._slack_api_request.call_args
        assert call_args[0][0] == "chat.update"


# ---------------------------------------------------------------------------
# Delete Message Tests
# ---------------------------------------------------------------------------


class TestDeleteMessage:
    """Tests for delete_message method."""

    @pytest.mark.asyncio
    async def test_delete_message_success(self, mock_slack_connector):
        """Should delete message and return True."""
        mock_slack_connector._slack_api_request.return_value = (
            True,
            {"ok": True},
            None,
        )

        result = await mock_slack_connector.delete_message(
            channel_id="C12345",
            message_id="1704067200.123456",
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_delete_message_failure(self, mock_slack_connector):
        """Should return False on API error."""
        mock_slack_connector._slack_api_request.return_value = (
            False,
            None,
            "message_not_found",
        )

        result = await mock_slack_connector.delete_message(
            channel_id="C12345",
            message_id="invalid",
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_delete_message_calls_correct_endpoint(self, mock_slack_connector):
        """Should call chat.delete endpoint."""
        mock_slack_connector._slack_api_request.return_value = (
            True,
            {"ok": True},
            None,
        )

        await mock_slack_connector.delete_message(
            channel_id="C12345",
            message_id="1.0",
        )

        call_args = mock_slack_connector._slack_api_request.call_args
        assert call_args[0][0] == "chat.delete"

    @pytest.mark.asyncio
    async def test_delete_message_includes_channel_and_ts(self, mock_slack_connector):
        """Should include channel and ts in payload."""
        mock_slack_connector._slack_api_request.return_value = (
            True,
            {"ok": True},
            None,
        )

        await mock_slack_connector.delete_message(
            channel_id="C12345",
            message_id="1704067200.123456",
        )

        call_args = mock_slack_connector._slack_api_request.call_args
        payload = call_args[0][1]
        assert payload["channel"] == "C12345"
        assert payload["ts"] == "1704067200.123456"


# ---------------------------------------------------------------------------
# Ephemeral Message Tests
# ---------------------------------------------------------------------------


class TestSendEphemeral:
    """Tests for send_ephemeral method."""

    @pytest.mark.asyncio
    async def test_send_ephemeral_httpx_not_available(self, mock_slack_connector):
        """Should return failure when httpx not available."""
        with patch("aragora.connectors.chat.slack.messages.HTTPX_AVAILABLE", False):
            response = await mock_slack_connector.send_ephemeral(
                channel_id="C12345",
                user_id="U12345",
                text="Secret message",
            )

        assert response.success is False
        assert "httpx" in response.error.lower()

    @pytest.mark.asyncio
    async def test_send_ephemeral_circuit_breaker_open(self, mock_connector_with_circuit_breaker):
        """Should return failure when circuit breaker is open."""
        mock_connector_with_circuit_breaker._circuit_breaker.can_proceed.return_value = False

        response = await mock_connector_with_circuit_breaker.send_ephemeral(
            channel_id="C12345",
            user_id="U12345",
            text="Secret message",
        )

        assert response.success is False
        assert "circuit breaker" in response.error.lower()

    @pytest.mark.asyncio
    async def test_send_ephemeral_success(self, mock_connector_with_circuit_breaker):
        """Should send ephemeral message successfully."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": True}

        with patch("aragora.connectors.chat.slack.messages.HTTPX_AVAILABLE", True):
            with patch("aragora.connectors.chat.slack.messages.httpx") as mock_httpx:
                mock_client = AsyncMock()
                mock_client.post.return_value = mock_response
                mock_httpx.AsyncClient.return_value.__aenter__.return_value = mock_client

                response = await mock_connector_with_circuit_breaker.send_ephemeral(
                    channel_id="C12345",
                    user_id="U12345",
                    text="Secret message",
                )

        assert response.success is True
        mock_connector_with_circuit_breaker._circuit_breaker.record_success.assert_called()

    @pytest.mark.asyncio
    async def test_send_ephemeral_with_blocks(self, mock_connector_with_circuit_breaker):
        """Should include blocks in ephemeral message."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": True}

        with patch("aragora.connectors.chat.slack.messages.HTTPX_AVAILABLE", True):
            with patch("aragora.connectors.chat.slack.messages.httpx") as mock_httpx:
                mock_client = AsyncMock()
                mock_client.post.return_value = mock_response
                mock_httpx.AsyncClient.return_value.__aenter__.return_value = mock_client

                blocks = [{"type": "section", "text": {"type": "mrkdwn", "text": "Secret"}}]
                await mock_connector_with_circuit_breaker.send_ephemeral(
                    channel_id="C12345",
                    user_id="U12345",
                    text="Secret",
                    blocks=blocks,
                )

                call_args = mock_client.post.call_args
                assert call_args[1]["json"]["blocks"] == blocks

    @pytest.mark.asyncio
    async def test_send_ephemeral_api_error_records_failure(
        self, mock_connector_with_circuit_breaker
    ):
        """Should record circuit breaker failure on API error."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": False, "error": "user_not_found"}
        mock_response.status_code = 200

        with patch("aragora.connectors.chat.slack.messages.HTTPX_AVAILABLE", True):
            with patch("aragora.connectors.chat.slack.messages.httpx") as mock_httpx:
                mock_client = AsyncMock()
                mock_client.post.return_value = mock_response
                mock_httpx.AsyncClient.return_value.__aenter__.return_value = mock_client

                response = await mock_connector_with_circuit_breaker.send_ephemeral(
                    channel_id="C12345",
                    user_id="U_INVALID",
                    text="Test",
                )

        assert response.success is False
        assert response.error == "user_not_found"
        mock_connector_with_circuit_breaker._circuit_breaker.record_failure.assert_called()

    @pytest.mark.asyncio
    async def test_send_ephemeral_timeout_retries(self, mock_connector_with_circuit_breaker):
        """Should retry on timeout and eventually fail."""
        import httpx as real_httpx

        with patch("aragora.connectors.chat.slack.messages.HTTPX_AVAILABLE", True):
            with patch("aragora.connectors.chat.slack.messages.httpx") as mock_httpx:
                with patch(
                    "aragora.connectors.chat.slack.messages._exponential_backoff",
                    new_callable=AsyncMock,
                ):
                    mock_client = AsyncMock()
                    # Use the real TimeoutException from httpx
                    mock_httpx.TimeoutException = real_httpx.TimeoutException
                    mock_httpx.ConnectError = real_httpx.ConnectError
                    mock_client.post.side_effect = real_httpx.TimeoutException("timeout")
                    mock_httpx.AsyncClient.return_value.__aenter__.return_value = mock_client

                    response = await mock_connector_with_circuit_breaker.send_ephemeral(
                        channel_id="C12345",
                        user_id="U12345",
                        text="Test",
                    )

        assert response.success is False
        assert "timeout" in response.error.lower()


# ---------------------------------------------------------------------------
# Respond to Command Tests
# ---------------------------------------------------------------------------


class TestRespondToCommand:
    """Tests for respond_to_command method."""

    @pytest.mark.asyncio
    async def test_respond_to_command_with_response_url(self, mock_slack_connector):
        """Should use response_url when available."""
        mock_slack_connector._send_to_response_url = AsyncMock()
        mock_slack_connector._send_to_response_url.return_value = MagicMock(success=True)

        command = MagicMock()
        command.response_url = "https://hooks.slack.com/response/123"
        command.channel = MagicMock(id="C12345")
        command.user = MagicMock(id="U12345")

        response = await mock_slack_connector.respond_to_command(
            command=command,
            text="Response text",
            ephemeral=True,
        )

        mock_slack_connector._send_to_response_url.assert_called_once()
        call_args = mock_slack_connector._send_to_response_url.call_args
        assert call_args[0][0] == "https://hooks.slack.com/response/123"
        assert call_args[1]["response_type"] == "ephemeral"

    @pytest.mark.asyncio
    async def test_respond_to_command_in_channel(self, mock_slack_connector):
        """Should respond in channel when not ephemeral."""
        mock_slack_connector._send_to_response_url = AsyncMock()
        mock_slack_connector._send_to_response_url.return_value = MagicMock(success=True)

        command = MagicMock()
        command.response_url = "https://hooks.slack.com/response/123"
        command.channel = MagicMock(id="C12345")
        command.user = MagicMock(id="U12345")

        await mock_slack_connector.respond_to_command(
            command=command,
            text="Public response",
            ephemeral=False,
        )

        call_args = mock_slack_connector._send_to_response_url.call_args
        assert call_args[1]["response_type"] == "in_channel"

    @pytest.mark.asyncio
    async def test_respond_to_command_fallback_to_ephemeral(self, mock_slack_connector):
        """Should fallback to send_ephemeral without response_url."""
        mock_slack_connector.send_ephemeral = AsyncMock()
        mock_slack_connector.send_ephemeral.return_value = MagicMock(success=True)

        command = MagicMock()
        command.response_url = None
        command.channel = MagicMock(id="C12345")
        command.user = MagicMock(id="U12345")

        await mock_slack_connector.respond_to_command(
            command=command,
            text="Ephemeral fallback",
            ephemeral=True,
        )

        mock_slack_connector.send_ephemeral.assert_called_once()
        call_args = mock_slack_connector.send_ephemeral.call_args
        assert call_args[0][0] == "C12345"
        assert call_args[0][1] == "U12345"

    @pytest.mark.asyncio
    async def test_respond_to_command_no_target(self, mock_slack_connector):
        """Should return error when no response target available."""
        command = MagicMock()
        command.response_url = None
        command.channel = None
        command.user = None

        response = await mock_slack_connector.respond_to_command(
            command=command,
            text="Test",
        )

        assert response.success is False
        assert "no response target" in response.error.lower()


# ---------------------------------------------------------------------------
# Respond to Interaction Tests
# ---------------------------------------------------------------------------


class TestRespondToInteraction:
    """Tests for respond_to_interaction method."""

    @pytest.mark.asyncio
    async def test_respond_to_interaction_with_response_url(self, mock_slack_connector):
        """Should use response_url when available."""
        mock_slack_connector._send_to_response_url = AsyncMock()
        mock_slack_connector._send_to_response_url.return_value = MagicMock(success=True)

        interaction = MagicMock()
        interaction.response_url = "https://hooks.slack.com/response/456"
        interaction.channel = MagicMock(id="C12345")
        interaction.message_id = "1.0"

        await mock_slack_connector.respond_to_interaction(
            interaction=interaction,
            text="Interaction response",
        )

        mock_slack_connector._send_to_response_url.assert_called_once()

    @pytest.mark.asyncio
    async def test_respond_to_interaction_replace_original(self, mock_slack_connector):
        """Should update original message when replace_original is True."""
        mock_slack_connector._send_to_response_url = AsyncMock()
        mock_slack_connector._send_to_response_url.return_value = MagicMock(success=True)

        interaction = MagicMock()
        interaction.response_url = "https://hooks.slack.com/response/456"
        interaction.channel = MagicMock(id="C12345")
        interaction.message_id = "1.0"

        await mock_slack_connector.respond_to_interaction(
            interaction=interaction,
            text="Replaced content",
            replace_original=True,
        )

        call_args = mock_slack_connector._send_to_response_url.call_args
        assert call_args[1]["replace_original"] is True

    @pytest.mark.asyncio
    async def test_respond_to_interaction_fallback_update(self, mock_slack_connector):
        """Should fallback to update_message for replace_original."""
        mock_slack_connector.update_message = AsyncMock()
        mock_slack_connector.update_message.return_value = MagicMock(success=True)

        interaction = MagicMock()
        interaction.response_url = None
        interaction.channel = MagicMock(id="C12345")
        interaction.message_id = "1704067200.123456"

        await mock_slack_connector.respond_to_interaction(
            interaction=interaction,
            text="Updated",
            replace_original=True,
        )

        mock_slack_connector.update_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_respond_to_interaction_fallback_send(self, mock_slack_connector):
        """Should fallback to send_message when not replacing."""
        mock_slack_connector.send_message = AsyncMock()
        mock_slack_connector.send_message.return_value = MagicMock(success=True)

        interaction = MagicMock()
        interaction.response_url = None
        interaction.channel = MagicMock(id="C12345")
        interaction.message_id = None

        await mock_slack_connector.respond_to_interaction(
            interaction=interaction,
            text="New message",
            replace_original=False,
        )

        mock_slack_connector.send_message.assert_called_once()


# ---------------------------------------------------------------------------
# Format Blocks Tests
# ---------------------------------------------------------------------------


class TestFormatBlocks:
    """Tests for format_blocks method."""

    def test_format_blocks_with_title(self, mock_slack_connector):
        """Should create header block for title."""
        blocks = mock_slack_connector.format_blocks(title="My Title")

        assert len(blocks) == 1
        assert blocks[0]["type"] == "header"
        assert blocks[0]["text"]["text"] == "My Title"

    def test_format_blocks_with_body(self, mock_slack_connector):
        """Should create section block for body."""
        blocks = mock_slack_connector.format_blocks(body="This is the body text")

        assert len(blocks) == 1
        assert blocks[0]["type"] == "section"
        assert blocks[0]["text"]["type"] == "mrkdwn"
        assert blocks[0]["text"]["text"] == "This is the body text"

    def test_format_blocks_with_fields(self, mock_slack_connector):
        """Should create fields section for field tuples."""
        fields = [("Label 1", "Value 1"), ("Label 2", "Value 2")]
        blocks = mock_slack_connector.format_blocks(fields=fields)

        assert len(blocks) == 1
        assert blocks[0]["type"] == "section"
        assert len(blocks[0]["fields"]) == 2
        assert "*Label 1*\nValue 1" in blocks[0]["fields"][0]["text"]

    def test_format_blocks_with_actions(self, mock_slack_connector):
        """Should create actions block for buttons."""
        from aragora.connectors.chat.models import MessageButton

        buttons = [
            MessageButton(text="Approve", action_id="approve", style="primary"),
            MessageButton(text="Reject", action_id="reject", style="danger"),
        ]
        blocks = mock_slack_connector.format_blocks(actions=buttons)

        assert len(blocks) == 1
        assert blocks[0]["type"] == "actions"
        assert len(blocks[0]["elements"]) == 2

    def test_format_blocks_combined(self, mock_slack_connector):
        """Should combine multiple block types."""
        from aragora.connectors.chat.models import MessageButton

        blocks = mock_slack_connector.format_blocks(
            title="Report",
            body="Summary of findings",
            fields=[("Status", "Complete")],
            actions=[MessageButton(text="View", action_id="view")],
        )

        assert len(blocks) == 4
        assert blocks[0]["type"] == "header"
        assert blocks[1]["type"] == "section"
        assert blocks[2]["type"] == "section"
        assert blocks[3]["type"] == "actions"

    def test_format_blocks_empty(self, mock_slack_connector):
        """Should return empty list when no content provided."""
        blocks = mock_slack_connector.format_blocks()

        assert blocks == []


# ---------------------------------------------------------------------------
# Format Button Tests
# ---------------------------------------------------------------------------


class TestFormatButton:
    """Tests for format_button method."""

    def test_format_button_basic(self, mock_slack_connector):
        """Should create basic button element."""
        button = mock_slack_connector.format_button(
            text="Click Me",
            action_id="click_action",
        )

        assert button["type"] == "button"
        assert button["text"]["text"] == "Click Me"
        assert button["action_id"] == "click_action"

    def test_format_button_with_value(self, mock_slack_connector):
        """Should include custom value."""
        button = mock_slack_connector.format_button(
            text="Submit",
            action_id="submit",
            value="form_data_123",
        )

        assert button["value"] == "form_data_123"

    def test_format_button_default_value(self, mock_slack_connector):
        """Should use action_id as default value."""
        button = mock_slack_connector.format_button(
            text="Submit",
            action_id="submit_action",
        )

        assert button["value"] == "submit_action"

    def test_format_button_primary_style(self, mock_slack_connector):
        """Should set primary button style."""
        button = mock_slack_connector.format_button(
            text="Confirm",
            action_id="confirm",
            style="primary",
        )

        assert button["style"] == "primary"

    def test_format_button_danger_style(self, mock_slack_connector):
        """Should set danger button style."""
        button = mock_slack_connector.format_button(
            text="Delete",
            action_id="delete",
            style="danger",
        )

        assert button["style"] == "danger"

    def test_format_button_default_style_no_style_key(self, mock_slack_connector):
        """Should not include style key for default style."""
        button = mock_slack_connector.format_button(
            text="Normal",
            action_id="normal",
            style="default",
        )

        assert "style" not in button

    def test_format_button_with_url(self, mock_slack_connector):
        """Should create link button with URL."""
        button = mock_slack_connector.format_button(
            text="Visit Site",
            action_id="visit",
            url="https://example.com",
        )

        assert button["url"] == "https://example.com"
        assert "action_id" not in button  # URL buttons don't have action_id


# ---------------------------------------------------------------------------
# Reaction Tests
# ---------------------------------------------------------------------------


class TestAddReaction:
    """Tests for add_reaction method."""

    @pytest.mark.asyncio
    async def test_add_reaction_success(self, mock_slack_connector):
        """Should add reaction successfully."""
        mock_slack_connector._slack_api_request.return_value = (
            True,
            {"ok": True},
            None,
        )

        result = await mock_slack_connector.add_reaction(
            channel_id="C12345",
            message_id="1704067200.123456",
            emoji="thumbsup",
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_add_reaction_strips_colons(self, mock_slack_connector):
        """Should strip colons from emoji name."""
        mock_slack_connector._slack_api_request.return_value = (
            True,
            {"ok": True},
            None,
        )

        await mock_slack_connector.add_reaction(
            channel_id="C12345",
            message_id="1.0",
            emoji=":thumbsup:",
        )

        call_args = mock_slack_connector._slack_api_request.call_args
        json_data = call_args[1]["json_data"]
        assert json_data["name"] == "thumbsup"

    @pytest.mark.asyncio
    async def test_add_reaction_already_reacted(self, mock_slack_connector):
        """Should return True for already_reacted error."""
        mock_slack_connector._slack_api_request.return_value = (
            False,
            None,
            "already_reacted",
        )

        result = await mock_slack_connector.add_reaction(
            channel_id="C12345",
            message_id="1.0",
            emoji="thumbsup",
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_add_reaction_failure(self, mock_slack_connector):
        """Should return False on API error."""
        mock_slack_connector._slack_api_request.return_value = (
            False,
            None,
            "message_not_found",
        )

        result = await mock_slack_connector.add_reaction(
            channel_id="C12345",
            message_id="invalid",
            emoji="thumbsup",
        )

        assert result is False


class TestRemoveReaction:
    """Tests for remove_reaction method."""

    @pytest.mark.asyncio
    async def test_remove_reaction_success(self, mock_slack_connector):
        """Should remove reaction successfully."""
        mock_slack_connector._slack_api_request.return_value = (
            True,
            {"ok": True},
            None,
        )

        result = await mock_slack_connector.remove_reaction(
            channel_id="C12345",
            message_id="1704067200.123456",
            reaction="thumbsup",
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_remove_reaction_no_reaction(self, mock_slack_connector):
        """Should return True for no_reaction error."""
        mock_slack_connector._slack_api_request.return_value = (
            False,
            None,
            "no_reaction",
        )

        result = await mock_slack_connector.remove_reaction(
            channel_id="C12345",
            message_id="1.0",
            reaction="thumbsup",
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_remove_reaction_strips_colons(self, mock_slack_connector):
        """Should strip colons from reaction name."""
        mock_slack_connector._slack_api_request.return_value = (
            True,
            {"ok": True},
            None,
        )

        await mock_slack_connector.remove_reaction(
            channel_id="C12345",
            message_id="1.0",
            reaction=":heart:",
        )

        call_args = mock_slack_connector._slack_api_request.call_args
        json_data = call_args[1]["json_data"]
        assert json_data["name"] == "heart"


# ---------------------------------------------------------------------------
# File Upload Tests
# ---------------------------------------------------------------------------


class TestUploadFile:
    """Tests for upload_file method."""

    @pytest.mark.asyncio
    async def test_upload_file_success(self, mock_slack_connector):
        """Should upload file and return FileAttachment."""
        mock_slack_connector._slack_api_request.return_value = (
            True,
            {
                "file": {
                    "id": "F12345",
                    "name": "document.pdf",
                    "mimetype": "application/pdf",
                    "size": 1024,
                    "url_private": "https://files.slack.com/...",
                }
            },
            None,
        )

        attachment = await mock_slack_connector.upload_file(
            channel_id="C12345",
            content=b"PDF content here",
            filename="document.pdf",
            content_type="application/pdf",
        )

        assert attachment.id == "F12345"
        assert attachment.filename == "document.pdf"
        assert attachment.content_type == "application/pdf"
        assert attachment.size == 1024
        assert attachment.url == "https://files.slack.com/..."

    @pytest.mark.asyncio
    async def test_upload_file_with_title(self, mock_slack_connector):
        """Should include title in upload."""
        mock_slack_connector._slack_api_request.return_value = (
            True,
            {"file": {"id": "F1", "name": "test.txt", "mimetype": "text/plain", "size": 100}},
            None,
        )

        await mock_slack_connector.upload_file(
            channel_id="C12345",
            content=b"Test content",
            filename="test.txt",
            title="Test Document",
        )

        call_args = mock_slack_connector._slack_api_request.call_args
        form_data = call_args[1]["form_data"]
        assert form_data["title"] == "Test Document"

    @pytest.mark.asyncio
    async def test_upload_file_in_thread(self, mock_slack_connector):
        """Should include thread_ts for thread uploads."""
        mock_slack_connector._slack_api_request.return_value = (
            True,
            {"file": {"id": "F1", "name": "test.txt", "mimetype": "text/plain", "size": 100}},
            None,
        )

        await mock_slack_connector.upload_file(
            channel_id="C12345",
            content=b"Test",
            filename="test.txt",
            thread_id="1704067200.000001",
        )

        call_args = mock_slack_connector._slack_api_request.call_args
        form_data = call_args[1]["form_data"]
        assert form_data["thread_ts"] == "1704067200.000001"

    @pytest.mark.asyncio
    async def test_upload_file_failure(self, mock_slack_connector):
        """Should return minimal FileAttachment on failure."""
        mock_slack_connector._slack_api_request.return_value = (
            False,
            None,
            "file_upload_failed",
        )

        attachment = await mock_slack_connector.upload_file(
            channel_id="C12345",
            content=b"Content",
            filename="failed.txt",
        )

        assert attachment.id == ""
        assert attachment.filename == "failed.txt"
        assert attachment.url is None


# ---------------------------------------------------------------------------
# File Download Tests
# ---------------------------------------------------------------------------


class TestDownloadFile:
    """Tests for download_file method."""

    @pytest.mark.asyncio
    async def test_download_file_success(self, mock_slack_connector):
        """Should download file and return FileAttachment with content."""
        # First call: file info
        mock_slack_connector._slack_api_request.return_value = (
            True,
            {
                "file": {
                    "id": "F12345",
                    "name": "document.pdf",
                    "mimetype": "application/pdf",
                    "size": 1024,
                    "url_private_download": "https://files.slack.com/download/...",
                }
            },
            None,
        )
        # Second call: binary download
        mock_slack_connector._http_request.return_value = (
            True,
            b"PDF binary content",
            None,
        )

        attachment = await mock_slack_connector.download_file(file_id="F12345")

        assert attachment.id == "F12345"
        assert attachment.filename == "document.pdf"
        assert attachment.content == b"PDF binary content"

    @pytest.mark.asyncio
    async def test_download_file_no_url(self, mock_slack_connector):
        """Should return FileAttachment without content when no URL."""
        mock_slack_connector._slack_api_request.return_value = (
            True,
            {
                "file": {
                    "id": "F12345",
                    "name": "document.pdf",
                    "mimetype": "application/pdf",
                    "size": 1024,
                }
            },
            None,
        )

        attachment = await mock_slack_connector.download_file(file_id="F12345")

        assert attachment.id == "F12345"
        assert attachment.content is None

    @pytest.mark.asyncio
    async def test_download_file_info_failure(self, mock_slack_connector):
        """Should return minimal FileAttachment on file info failure."""
        mock_slack_connector._slack_api_request.return_value = (
            False,
            None,
            "file_not_found",
        )

        attachment = await mock_slack_connector.download_file(file_id="F_INVALID")

        assert attachment.id == "F_INVALID"
        assert attachment.filename == ""


# ---------------------------------------------------------------------------
# Channel Info Tests
# ---------------------------------------------------------------------------


class TestGetChannelInfo:
    """Tests for get_channel_info method."""

    @pytest.mark.asyncio
    async def test_get_channel_info_success(self, mock_slack_connector):
        """Should return ChatChannel with channel data."""
        mock_slack_connector._slack_api_request.return_value = (
            True,
            {
                "channel": {
                    "id": "C12345",
                    "name": "general",
                    "context_team_id": "T12345",
                    "is_private": False,
                    "topic": {"value": "General discussion"},
                    "purpose": {"value": "Company-wide announcements"},
                    "num_members": 100,
                }
            },
            None,
        )

        channel = await mock_slack_connector.get_channel_info("C12345")

        assert channel.id == "C12345"
        assert channel.name == "general"
        assert channel.team_id == "T12345"
        assert channel.is_private is False
        assert channel.metadata["topic"] == "General discussion"
        assert channel.metadata["num_members"] == 100

    @pytest.mark.asyncio
    async def test_get_channel_info_not_found(self, mock_slack_connector):
        """Should return None when channel not found."""
        mock_slack_connector._slack_api_request.return_value = (
            False,
            None,
            "channel_not_found",
        )

        channel = await mock_slack_connector.get_channel_info("C_INVALID")

        assert channel is None


# ---------------------------------------------------------------------------
# User Info Tests
# ---------------------------------------------------------------------------


class TestGetUserInfo:
    """Tests for get_user_info method."""

    @pytest.mark.asyncio
    async def test_get_user_info_success(self, mock_slack_connector):
        """Should return ChatUser with user data."""
        mock_slack_connector._slack_api_request.return_value = (
            True,
            {
                "user": {
                    "id": "U12345",
                    "name": "johndoe",
                    "team_id": "T12345",
                    "is_bot": False,
                    "tz": "America/New_York",
                    "profile": {
                        "display_name": "John Doe",
                        "real_name": "John Doe",
                        "email": "john@example.com",
                        "title": "Engineer",
                    },
                }
            },
            None,
        )

        user = await mock_slack_connector.get_user_info("U12345")

        assert user.id == "U12345"
        assert user.username == "johndoe"
        assert user.display_name == "John Doe"
        assert user.email == "john@example.com"
        assert user.is_bot is False
        assert user.metadata["tz"] == "America/New_York"

    @pytest.mark.asyncio
    async def test_get_user_info_not_found(self, mock_slack_connector):
        """Should return None when user not found."""
        mock_slack_connector._slack_api_request.return_value = (
            False,
            None,
            "user_not_found",
        )

        user = await mock_slack_connector.get_user_info("U_INVALID")

        assert user is None

    @pytest.mark.asyncio
    async def test_get_user_info_display_name_fallback(self, mock_slack_connector):
        """Should fallback to real_name when display_name empty."""
        mock_slack_connector._slack_api_request.return_value = (
            True,
            {
                "user": {
                    "id": "U12345",
                    "name": "johndoe",
                    "profile": {
                        "display_name": "",
                        "real_name": "John Doe",
                    },
                }
            },
            None,
        )

        user = await mock_slack_connector.get_user_info("U12345")

        assert user.display_name == "John Doe"


# ---------------------------------------------------------------------------
# List Channels Tests
# ---------------------------------------------------------------------------


class TestListChannels:
    """Tests for list_channels method."""

    @pytest.mark.asyncio
    async def test_list_channels_success(self, mock_slack_connector):
        """Should return list of ChatChannel objects."""
        mock_slack_connector._slack_api_request.return_value = (
            True,
            {
                "channels": [
                    {
                        "id": "C12345",
                        "name": "general",
                        "is_private": False,
                        "is_archived": False,
                        "is_member": True,
                        "num_members": 50,
                        "topic": {"value": "General chat"},
                        "purpose": {"value": "For general discussion"},
                    },
                    {
                        "id": "C67890",
                        "name": "random",
                        "is_private": False,
                        "is_archived": False,
                        "is_member": True,
                        "num_members": 45,
                        "topic": {"value": "Random stuff"},
                    },
                ]
            },
            None,
        )

        channels = await mock_slack_connector.list_channels()

        assert len(channels) == 2
        assert channels[0].id == "C12345"
        assert channels[0].name == "general"
        assert channels[0].metadata["is_member"] is True

    @pytest.mark.asyncio
    async def test_list_channels_api_failure(self, mock_slack_connector):
        """Should return empty list on API failure."""
        mock_slack_connector._slack_api_request.return_value = (
            False,
            None,
            "not_authed",
        )

        channels = await mock_slack_connector.list_channels()

        assert channels == []


# ---------------------------------------------------------------------------
# List Users Tests
# ---------------------------------------------------------------------------


class TestListUsers:
    """Tests for list_users method."""

    @pytest.mark.asyncio
    async def test_list_users_workspace(self, mock_slack_connector):
        """Should list workspace users."""
        mock_slack_connector._slack_api_request.return_value = (
            True,
            {
                "members": [
                    {
                        "id": "U12345",
                        "name": "johndoe",
                        "deleted": False,
                        "is_bot": False,
                        "is_admin": True,
                        "tz": "America/New_York",
                        "profile": {
                            "display_name": "John",
                            "real_name": "John Doe",
                            "image_72": "https://avatars.slack.com/...",
                            "email": "john@example.com",
                        },
                    },
                ],
                "response_metadata": {"next_cursor": ""},
            },
            None,
        )

        users, cursor = await mock_slack_connector.list_users()

        assert len(users) == 1
        assert users[0].id == "U12345"
        assert users[0].username == "johndoe"
        assert cursor is None

    @pytest.mark.asyncio
    async def test_list_users_excludes_deleted(self, mock_slack_connector):
        """Should exclude deleted users."""
        mock_slack_connector._slack_api_request.return_value = (
            True,
            {
                "members": [
                    {"id": "U1", "name": "active", "deleted": False, "is_bot": False},
                    {"id": "U2", "name": "deleted", "deleted": True, "is_bot": False},
                ],
                "response_metadata": {"next_cursor": ""},
            },
            None,
        )

        users, _ = await mock_slack_connector.list_users()

        assert len(users) == 1
        assert users[0].id == "U1"

    @pytest.mark.asyncio
    async def test_list_users_excludes_bots_by_default(self, mock_slack_connector):
        """Should exclude bots by default."""
        mock_slack_connector._slack_api_request.return_value = (
            True,
            {
                "members": [
                    {"id": "U1", "name": "human", "deleted": False, "is_bot": False},
                    {"id": "B1", "name": "bot", "deleted": False, "is_bot": True},
                ],
                "response_metadata": {"next_cursor": ""},
            },
            None,
        )

        users, _ = await mock_slack_connector.list_users()

        assert len(users) == 1
        assert users[0].id == "U1"

    @pytest.mark.asyncio
    async def test_list_users_include_bots(self, mock_slack_connector):
        """Should include bots when requested."""
        mock_slack_connector._slack_api_request.return_value = (
            True,
            {
                "members": [
                    {"id": "U1", "name": "human", "deleted": False, "is_bot": False},
                    {"id": "B1", "name": "bot", "deleted": False, "is_bot": True},
                ],
                "response_metadata": {"next_cursor": ""},
            },
            None,
        )

        users, _ = await mock_slack_connector.list_users(include_bots=True)

        assert len(users) == 2

    @pytest.mark.asyncio
    async def test_list_users_with_pagination(self, mock_slack_connector):
        """Should return pagination cursor."""
        mock_slack_connector._slack_api_request.return_value = (
            True,
            {
                "members": [
                    {"id": "U1", "name": "user1", "deleted": False, "is_bot": False},
                ],
                "response_metadata": {"next_cursor": "next_page_token"},
            },
            None,
        )

        users, cursor = await mock_slack_connector.list_users()

        assert cursor == "next_page_token"


# ---------------------------------------------------------------------------
# User Mention Helpers Tests
# ---------------------------------------------------------------------------


class TestUserMentionHelpers:
    """Tests for static mention formatting methods."""

    def test_format_user_mention(self, mock_slack_connector):
        """Should format user mention correctly."""
        mention = mock_slack_connector.format_user_mention("U12345ABC")

        assert mention == "<@U12345ABC>"

    def test_format_channel_mention(self, mock_slack_connector):
        """Should format channel mention correctly."""
        mention = mock_slack_connector.format_channel_mention("C12345ABC")

        assert mention == "<#C12345ABC>"


# ---------------------------------------------------------------------------
# Modal Tests
# ---------------------------------------------------------------------------


class TestOpenModal:
    """Tests for open_modal method."""

    @pytest.mark.asyncio
    async def test_open_modal_success(self, mock_slack_connector):
        """Should open modal and return view ID."""
        mock_slack_connector._slack_api_request.return_value = (
            True,
            {"view": {"id": "V12345"}},
            None,
        )

        view = {
            "type": "modal",
            "title": {"type": "plain_text", "text": "My Modal"},
            "blocks": [],
        }

        view_id = await mock_slack_connector.open_modal(
            trigger_id="trigger123",
            view=view,
        )

        assert view_id == "V12345"

    @pytest.mark.asyncio
    async def test_open_modal_failure(self, mock_slack_connector):
        """Should return None on failure."""
        mock_slack_connector._slack_api_request.return_value = (
            False,
            None,
            "trigger_expired",
        )

        view_id = await mock_slack_connector.open_modal(
            trigger_id="expired_trigger",
            view={},
        )

        assert view_id is None


class TestUpdateModal:
    """Tests for update_modal method."""

    @pytest.mark.asyncio
    async def test_update_modal_success(self, mock_slack_connector):
        """Should update modal and return True."""
        mock_slack_connector._slack_api_request.return_value = (
            True,
            {"view": {"id": "V12345"}},
            None,
        )

        result = await mock_slack_connector.update_modal(
            view_id="V12345",
            view={"type": "modal", "blocks": []},
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_update_modal_with_hash(self, mock_slack_connector):
        """Should include view_hash for optimistic locking."""
        mock_slack_connector._slack_api_request.return_value = (
            True,
            {"view": {"id": "V12345"}},
            None,
        )

        await mock_slack_connector.update_modal(
            view_id="V12345",
            view={},
            view_hash="hash123",
        )

        call_args = mock_slack_connector._slack_api_request.call_args
        json_data = call_args[1]["json_data"]
        assert json_data["hash"] == "hash123"

    @pytest.mark.asyncio
    async def test_update_modal_failure(self, mock_slack_connector):
        """Should return False on failure."""
        mock_slack_connector._slack_api_request.return_value = (
            False,
            None,
            "view_not_found",
        )

        result = await mock_slack_connector.update_modal(
            view_id="V_INVALID",
            view={},
        )

        assert result is False


# ---------------------------------------------------------------------------
# Pinned Messages Tests
# ---------------------------------------------------------------------------


class TestPinMessage:
    """Tests for pin_message method."""

    @pytest.mark.asyncio
    async def test_pin_message_success(self, mock_slack_connector):
        """Should pin message successfully."""
        mock_slack_connector._slack_api_request.return_value = (
            True,
            {"ok": True},
            None,
        )

        result = await mock_slack_connector.pin_message(
            channel_id="C12345",
            message_id="1704067200.123456",
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_pin_message_already_pinned(self, mock_slack_connector):
        """Should return True for already_pinned error."""
        mock_slack_connector._slack_api_request.return_value = (
            False,
            None,
            "already_pinned",
        )

        result = await mock_slack_connector.pin_message(
            channel_id="C12345",
            message_id="1.0",
        )

        assert result is True


class TestUnpinMessage:
    """Tests for unpin_message method."""

    @pytest.mark.asyncio
    async def test_unpin_message_success(self, mock_slack_connector):
        """Should unpin message successfully."""
        mock_slack_connector._slack_api_request.return_value = (
            True,
            {"ok": True},
            None,
        )

        result = await mock_slack_connector.unpin_message(
            channel_id="C12345",
            message_id="1704067200.123456",
        )

        assert result is True

    @pytest.mark.asyncio
    async def test_unpin_message_no_pin(self, mock_slack_connector):
        """Should return True for no_pin error."""
        mock_slack_connector._slack_api_request.return_value = (
            False,
            None,
            "no_pin",
        )

        result = await mock_slack_connector.unpin_message(
            channel_id="C12345",
            message_id="1.0",
        )

        assert result is True


class TestGetPinnedMessages:
    """Tests for get_pinned_messages method."""

    @pytest.mark.asyncio
    async def test_get_pinned_messages_success(self, mock_slack_connector):
        """Should return list of pinned messages."""
        mock_slack_connector._slack_api_request.return_value = (
            True,
            {
                "items": [
                    {
                        "type": "message",
                        "message": {
                            "ts": "1704067200.123456",
                            "user": "U12345",
                            "text": "Important announcement",
                        },
                    },
                    {
                        "type": "message",
                        "message": {
                            "ts": "1704153600.123456",
                            "user": "U67890",
                            "text": "Another pinned message",
                        },
                    },
                ]
            },
            None,
        )

        messages = await mock_slack_connector.get_pinned_messages("C12345")

        assert len(messages) == 2
        assert messages[0].id == "1704067200.123456"
        assert messages[0].content == "Important announcement"
        assert messages[0].metadata["pinned"] is True

    @pytest.mark.asyncio
    async def test_get_pinned_messages_empty(self, mock_slack_connector):
        """Should return empty list when no pinned messages."""
        mock_slack_connector._slack_api_request.return_value = (
            True,
            {"items": []},
            None,
        )

        messages = await mock_slack_connector.get_pinned_messages("C12345")

        assert messages == []

    @pytest.mark.asyncio
    async def test_get_pinned_messages_filters_non_messages(self, mock_slack_connector):
        """Should filter out non-message items (files, etc.)."""
        mock_slack_connector._slack_api_request.return_value = (
            True,
            {
                "items": [
                    {"type": "file", "file": {"id": "F12345"}},
                    {
                        "type": "message",
                        "message": {"ts": "1.0", "user": "U1", "text": "Only message"},
                    },
                ]
            },
            None,
        )

        messages = await mock_slack_connector.get_pinned_messages("C12345")

        assert len(messages) == 1


# ---------------------------------------------------------------------------
# Channel History Tests
# ---------------------------------------------------------------------------


class TestGetChannelHistory:
    """Tests for get_channel_history method."""

    @pytest.mark.asyncio
    async def test_get_channel_history_success(self, mock_slack_connector):
        """Should return list of ChatMessage objects."""
        mock_slack_connector._slack_api_request.return_value = (
            True,
            {
                "messages": [
                    {"ts": "1704067200.123456", "user": "U12345", "text": "First message"},
                    {"ts": "1704067260.123456", "user": "U67890", "text": "Second message"},
                ]
            },
            None,
        )

        messages = await mock_slack_connector.get_channel_history("C12345", limit=10)

        assert len(messages) == 2
        assert messages[0].id == "1704067200.123456"
        assert messages[0].content == "First message"

    @pytest.mark.asyncio
    async def test_get_channel_history_skips_bots(self, mock_slack_connector):
        """Should skip bot messages by default."""
        mock_slack_connector._slack_api_request.return_value = (
            True,
            {
                "messages": [
                    {"ts": "1.0", "user": "U1", "text": "Human message"},
                    {"ts": "2.0", "bot_id": "B1", "text": "Bot message"},
                ]
            },
            None,
        )

        messages = await mock_slack_connector.get_channel_history("C12345")

        assert len(messages) == 1
        assert messages[0].author.is_bot is False

    @pytest.mark.asyncio
    async def test_get_channel_history_includes_thread_info(self, mock_slack_connector):
        """Should include thread_ts as thread_id."""
        mock_slack_connector._slack_api_request.return_value = (
            True,
            {
                "messages": [
                    {
                        "ts": "1704067260.000001",
                        "user": "U1",
                        "text": "Reply",
                        "thread_ts": "1704067200.000001",
                    },
                ]
            },
            None,
        )

        messages = await mock_slack_connector.get_channel_history("C12345")

        assert messages[0].thread_id == "1704067200.000001"

    @pytest.mark.asyncio
    async def test_get_channel_history_limits_to_1000(self, mock_slack_connector):
        """Should cap limit at 1000."""
        mock_slack_connector._slack_api_request.return_value = (
            True,
            {"messages": []},
            None,
        )

        await mock_slack_connector.get_channel_history("C12345", limit=5000)

        # First call should be to conversations.history
        call_args = mock_slack_connector._slack_api_request.call_args_list[0]
        params = call_args[1]["params"]
        assert params["limit"] == 1000


# ---------------------------------------------------------------------------
# Evidence Collection Tests
# ---------------------------------------------------------------------------


class TestCollectEvidence:
    """Tests for collect_evidence method."""

    @pytest.mark.asyncio
    async def test_collect_evidence_success(self, mock_slack_connector):
        """Should collect evidence from channel history."""
        mock_slack_connector.get_channel_history = AsyncMock()
        mock_slack_connector.get_channel_history.return_value = [
            MagicMock(
                id="1.0",
                platform="slack",
                channel=MagicMock(id="C12345", name="general"),
                author=MagicMock(id="U1", display_name="John", username="john", is_bot=False),
                content="Relevant message about topic",
                timestamp=datetime.now(),
                thread_id=None,
                metadata={"reply_count": 0},
            ),
        ]
        mock_slack_connector._enrich_with_threads = AsyncMock()

        evidence = await mock_slack_connector.collect_evidence(
            channel_id="C12345",
            query="topic",
            limit=10,
        )

        assert len(evidence) == 1
        assert evidence[0].content == "Relevant message about topic"

    @pytest.mark.asyncio
    async def test_collect_evidence_filters_by_relevance(self, mock_slack_connector):
        """Should filter messages below min_relevance threshold."""
        mock_slack_connector.get_channel_history = AsyncMock()
        mock_slack_connector.get_channel_history.return_value = [
            MagicMock(
                id="1.0",
                platform="slack",
                channel=MagicMock(id="C12345", name="general"),
                author=MagicMock(id="U1", display_name="John", username="john", is_bot=False),
                content="Message",
                timestamp=datetime.now(),
                thread_id=None,
                metadata={},
            ),
        ]
        mock_slack_connector._compute_message_relevance = MagicMock(return_value=0.3)
        mock_slack_connector._enrich_with_threads = AsyncMock()

        evidence = await mock_slack_connector.collect_evidence(
            channel_id="C12345",
            min_relevance=0.5,
        )

        assert len(evidence) == 0


# ---------------------------------------------------------------------------
# Search Messages Tests
# ---------------------------------------------------------------------------


class TestSearchMessages:
    """Tests for search_messages method."""

    @pytest.mark.asyncio
    async def test_search_messages_success(self, mock_slack_connector):
        """Should search messages and return evidence."""
        mock_slack_connector._slack_api_request.return_value = (
            True,
            {
                "messages": {
                    "matches": [
                        {
                            "ts": "1704067200.123456",
                            "user": "U12345",
                            "username": "johndoe",
                            "text": "Found this important message",
                            "channel": {"id": "C12345", "name": "general"},
                            "permalink": "https://slack.com/archives/...",
                            "score": 85.5,
                        },
                    ]
                }
            },
            None,
        )

        evidence = await mock_slack_connector.search_messages(
            query="important",
            limit=10,
        )

        assert len(evidence) == 1
        assert evidence[0].content == "Found this important message"
        assert evidence[0].metadata["permalink"] == "https://slack.com/archives/..."

    @pytest.mark.asyncio
    async def test_search_messages_with_channel_filter(self, mock_slack_connector):
        """Should include channel filter in search query."""
        mock_slack_connector._slack_api_request.return_value = (
            True,
            {"messages": {"matches": []}},
            None,
        )

        await mock_slack_connector.search_messages(
            query="test",
            channel_id="C12345",
        )

        call_args = mock_slack_connector._slack_api_request.call_args
        params = call_args[1]["params"]
        assert "in:<#C12345>" in params["query"]

    @pytest.mark.asyncio
    async def test_search_messages_empty_results(self, mock_slack_connector):
        """Should return empty list when no matches."""
        mock_slack_connector._slack_api_request.return_value = (
            True,
            {"messages": {"matches": []}},
            None,
        )

        evidence = await mock_slack_connector.search_messages(query="nonexistent")

        assert evidence == []


# ---------------------------------------------------------------------------
# Timestamp Formatting Tests
# ---------------------------------------------------------------------------


class TestFormatTimestampForApi:
    """Tests for _format_timestamp_for_api method."""

    def test_format_timestamp_datetime(self, mock_slack_connector):
        """Should format datetime to Unix timestamp string."""
        dt = datetime(2024, 1, 1, 0, 0, 0)
        result = mock_slack_connector._format_timestamp_for_api(dt)

        assert result == str(dt.timestamp())

    def test_format_timestamp_string_passthrough(self, mock_slack_connector):
        """Should pass through string timestamps."""
        result = mock_slack_connector._format_timestamp_for_api("1704067200.0")

        assert result == "1704067200.0"

    def test_format_timestamp_none(self, mock_slack_connector):
        """Should return None for None input."""
        result = mock_slack_connector._format_timestamp_for_api(None)

        assert result is None


# ---------------------------------------------------------------------------
# Send to Response URL Tests
# ---------------------------------------------------------------------------


class TestSendToResponseUrl:
    """Tests for _send_to_response_url method."""

    @pytest.mark.asyncio
    async def test_send_to_response_url_httpx_not_available(self, mock_slack_connector):
        """Should return failure when httpx not available."""
        with patch("aragora.connectors.chat.slack.messages.HTTPX_AVAILABLE", False):
            response = await mock_slack_connector._send_to_response_url(
                response_url="https://hooks.slack.com/...",
                text="Response",
            )

        assert response.success is False
        assert "httpx" in response.error.lower()

    @pytest.mark.asyncio
    async def test_send_to_response_url_success(self, mock_slack_connector):
        """Should send to response URL successfully."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("aragora.connectors.chat.slack.messages.HTTPX_AVAILABLE", True):
            with patch("aragora.connectors.chat.slack.messages.httpx") as mock_httpx:
                mock_client = AsyncMock()
                mock_client.post.return_value = mock_response
                mock_httpx.AsyncClient.return_value.__aenter__.return_value = mock_client

                response = await mock_slack_connector._send_to_response_url(
                    response_url="https://hooks.slack.com/response/123",
                    text="Response text",
                )

        assert response.success is True

    @pytest.mark.asyncio
    async def test_send_to_response_url_with_replace_original(self, mock_slack_connector):
        """Should include replace_original in payload."""
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("aragora.connectors.chat.slack.messages.HTTPX_AVAILABLE", True):
            with patch("aragora.connectors.chat.slack.messages.httpx") as mock_httpx:
                mock_client = AsyncMock()
                mock_client.post.return_value = mock_response
                mock_httpx.AsyncClient.return_value.__aenter__.return_value = mock_client

                await mock_slack_connector._send_to_response_url(
                    response_url="https://hooks.slack.com/...",
                    text="Updated",
                    replace_original=True,
                )

                call_args = mock_client.post.call_args
                assert call_args[1]["json"]["replace_original"] is True

    @pytest.mark.asyncio
    async def test_send_to_response_url_http_error(self, mock_slack_connector):
        """Should return failure on HTTP error."""
        mock_response = MagicMock()
        mock_response.status_code = 404

        with patch("aragora.connectors.chat.slack.messages.HTTPX_AVAILABLE", True):
            with patch("aragora.connectors.chat.slack.messages.httpx") as mock_httpx:
                mock_client = AsyncMock()
                mock_client.post.return_value = mock_response
                mock_httpx.AsyncClient.return_value.__aenter__.return_value = mock_client

                response = await mock_slack_connector._send_to_response_url(
                    response_url="https://hooks.slack.com/...",
                    text="Test",
                )

        assert response.success is False
        assert "404" in response.error
