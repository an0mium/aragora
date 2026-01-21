"""
Tests for TeamsConnector - Microsoft Teams chat platform integration.

Tests cover:
- Message operations (send, update, delete)
- Adaptive Card formatting
- Bot Framework activities
- Error handling
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import json


class TestTeamsConnectorInit:
    """Tests for TeamsConnector initialization."""

    def test_default_init(self):
        """Should initialize with default values."""
        from aragora.connectors.chat.teams import TeamsConnector

        connector = TeamsConnector()

        assert connector.platform_name == "teams"
        assert connector.platform_display_name == "Microsoft Teams"

    def test_init_with_app_credentials(self):
        """Should accept app credentials."""
        from aragora.connectors.chat.teams import TeamsConnector

        connector = TeamsConnector(
            app_id="app-123",
            app_password="secret-pass",
        )

        assert connector.app_id == "app-123"
        assert connector.app_password == "secret-pass"

    def test_init_with_tenant(self):
        """Should accept tenant ID for single-tenant apps."""
        from aragora.connectors.chat.teams import TeamsConnector

        connector = TeamsConnector(tenant_id="tenant-456")

        assert connector.tenant_id == "tenant-456"


class TestTeamsAuthentication:
    """Tests for OAuth authentication."""

    @pytest.mark.asyncio
    async def test_get_access_token(self):
        """Should obtain access token from Azure AD."""
        from aragora.connectors.chat.teams import TeamsConnector

        connector = TeamsConnector(
            app_id="app-123",
            app_password="secret-pass",
        )

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "access_token": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...",
            "expires_in": 3600,
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.post = AsyncMock(return_value=mock_response)

            token = await connector._get_access_token()

        assert token is not None
        assert token.startswith("eyJ")

    @pytest.mark.asyncio
    async def test_token_caching(self):
        """Should cache access token."""
        from aragora.connectors.chat.teams import TeamsConnector

        connector = TeamsConnector(
            app_id="app-123",
            app_password="secret-pass",
        )

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "access_token": "cached-token",
            "expires_in": 3600,
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.post = AsyncMock(return_value=mock_response)

            # First call
            token1 = await connector._get_access_token()
            # Second call (should use cache)
            token2 = await connector._get_access_token()

        # Should only call API once
        assert mock_instance.post.call_count == 1
        assert token1 == token2


class TestTeamsSendMessage:
    """Tests for send_message method."""

    @pytest.mark.asyncio
    async def test_send_simple_message(self):
        """Should send simple text message."""
        from aragora.connectors.chat.teams import TeamsConnector

        connector = TeamsConnector(app_id="app-123", app_password="secret")

        # Mock token retrieval
        connector._access_token = "test-token"
        connector._token_expires = 9999999999

        mock_response = MagicMock()
        mock_response.json.return_value = {"id": "msg-123"}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.post = AsyncMock(return_value=mock_response)

            result = await connector.send_message(
                channel_id="conv-456",
                text="Hello, Teams!",
            )

        assert result.success is True
        assert result.message_id == "msg-123"

    @pytest.mark.asyncio
    async def test_send_message_with_adaptive_card(self):
        """Should send message with Adaptive Card blocks."""
        from aragora.connectors.chat.teams import TeamsConnector

        connector = TeamsConnector(app_id="app-123", app_password="secret")
        connector._access_token = "test-token"
        connector._token_expires = 9999999999

        mock_response = MagicMock()
        mock_response.json.return_value = {"id": "msg-123"}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.post = AsyncMock(return_value=mock_response)

            blocks = [
                {"type": "TextBlock", "text": "Debate Result", "weight": "Bolder"}
            ]

            result = await connector.send_message(
                channel_id="conv-456",
                text="Fallback",
                blocks=blocks,
            )

            # Verify adaptive card was included
            call_kwargs = mock_instance.post.call_args[1]
            payload = call_kwargs["json"]
            assert "attachments" in payload

        assert result.success is True

    @pytest.mark.asyncio
    async def test_send_threaded_reply(self):
        """Should send reply in conversation thread."""
        from aragora.connectors.chat.teams import TeamsConnector

        connector = TeamsConnector(app_id="app-123", app_password="secret")
        connector._access_token = "test-token"
        connector._token_expires = 9999999999

        mock_response = MagicMock()
        mock_response.json.return_value = {"id": "msg-789"}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.post = AsyncMock(return_value=mock_response)

            result = await connector.send_message(
                channel_id="conv-456",
                text="Thread reply",
                thread_id="msg-parent",
            )

            # Verify replyToId was included
            call_kwargs = mock_instance.post.call_args[1]
            payload = call_kwargs["json"]
            assert payload.get("replyToId") == "msg-parent"

        assert result.success is True

    @pytest.mark.asyncio
    async def test_send_message_error(self):
        """Should handle API errors gracefully."""
        from aragora.connectors.chat.teams import TeamsConnector

        connector = TeamsConnector(app_id="app-123", app_password="secret")
        connector._access_token = "test-token"
        connector._token_expires = 9999999999

        with patch("httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                side_effect=Exception("Conversation not found")
            )

            result = await connector.send_message(
                channel_id="invalid",
                text="Test",
            )

        assert result.success is False
        assert "Conversation not found" in result.error


class TestTeamsUpdateMessage:
    """Tests for update_message method."""

    @pytest.mark.asyncio
    async def test_update_message(self):
        """Should update existing message."""
        from aragora.connectors.chat.teams import TeamsConnector

        connector = TeamsConnector(app_id="app-123", app_password="secret")
        connector._access_token = "test-token"
        connector._token_expires = 9999999999

        mock_response = MagicMock()
        mock_response.json.return_value = {"id": "msg-123"}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.put = AsyncMock(return_value=mock_response)

            result = await connector.update_message(
                channel_id="conv-456",
                message_id="msg-123",
                text="Updated text",
            )

            # Verify PUT request
            assert mock_instance.put.called

        assert result.success is True


class TestTeamsDeleteMessage:
    """Tests for delete_message method."""

    @pytest.mark.asyncio
    async def test_delete_message(self):
        """Should delete message."""
        from aragora.connectors.chat.teams import TeamsConnector

        connector = TeamsConnector(app_id="app-123", app_password="secret")
        connector._access_token = "test-token"
        connector._token_expires = 9999999999

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.delete = AsyncMock(return_value=mock_response)

            result = await connector.delete_message(
                channel_id="conv-456",
                message_id="msg-123",
            )

        assert result is True


class TestTeamsWebhookParsing:
    """Tests for Bot Framework activity parsing."""

    def test_parse_message_activity(self):
        """Should parse incoming message activity."""
        from aragora.connectors.chat.teams import TeamsConnector

        connector = TeamsConnector()

        activity = {
            "type": "message",
            "id": "act-123",
            "text": "Hello bot!",
            "from": {
                "id": "user-789",
                "name": "Test User",
            },
            "conversation": {
                "id": "conv-456",
                "conversationType": "personal",
            },
            "channelId": "msteams",
            "serviceUrl": "https://smba.trafficmanager.net/...",
        }

        event = connector.parse_webhook_event(
            headers={},
            body=json.dumps(activity).encode("utf-8"),
        )

        assert event is not None
        assert event.event_type == "message"
        assert event.message.author.id == "user-789"
        assert "Hello bot" in event.message.content

    def test_parse_invoke_activity(self):
        """Should parse invoke activity (button click)."""
        from aragora.connectors.chat.teams import TeamsConnector

        connector = TeamsConnector()

        activity = {
            "type": "invoke",
            "name": "adaptiveCard/action",
            "id": "act-456",
            "value": {
                "action": "vote",
                "value": "agree",
            },
            "from": {
                "id": "user-789",
                "name": "Test User",
            },
            "conversation": {"id": "conv-456"},
            "serviceUrl": "https://smba.trafficmanager.net/...",
        }

        event = connector.parse_webhook_event(
            headers={},
            body=json.dumps(activity).encode("utf-8"),
        )

        assert event is not None
        assert event.event_type == "invoke"
        assert event.interaction is not None
        assert event.interaction.action_id == "vote"


class TestTeamsAdaptiveCardHelpers:
    """Tests for Adaptive Card formatting helpers."""

    def test_format_blocks_with_title(self):
        """Should format title as TextBlock."""
        from aragora.connectors.chat.teams import TeamsConnector

        connector = TeamsConnector()

        blocks = connector.format_blocks(
            title="Debate Result",
        )

        assert len(blocks) > 0
        assert blocks[0]["type"] == "TextBlock"
        assert blocks[0]["text"] == "Debate Result"

    def test_format_blocks_with_body(self):
        """Should format body as TextBlock."""
        from aragora.connectors.chat.teams import TeamsConnector

        connector = TeamsConnector()

        blocks = connector.format_blocks(
            body="The conclusion text here",
        )

        assert len(blocks) > 0
        assert any(b["type"] == "TextBlock" and "conclusion" in b["text"] for b in blocks)

    def test_format_blocks_with_fields(self):
        """Should format fields as FactSet."""
        from aragora.connectors.chat.teams import TeamsConnector

        connector = TeamsConnector()

        blocks = connector.format_blocks(
            fields=[("Confidence", "88%"), ("Rounds", "3")],
        )

        assert len(blocks) > 0
        fact_set = next((b for b in blocks if b.get("type") == "FactSet"), None)
        assert fact_set is not None
        assert len(fact_set["facts"]) == 2

    def test_format_button_submit(self):
        """Should format submit button."""
        from aragora.connectors.chat.teams import TeamsConnector

        connector = TeamsConnector()

        button = connector.format_button(
            text="Vote Yes",
            action_id="vote_yes",
            value="yes",
        )

        assert button["type"] == "Action.Submit"
        assert button["title"] == "Vote Yes"
        assert button["data"]["action"] == "vote_yes"

    def test_format_button_url(self):
        """Should format URL button."""
        from aragora.connectors.chat.teams import TeamsConnector

        connector = TeamsConnector()

        button = connector.format_button(
            text="View Details",
            action_id="view",
            url="https://example.com/debate/123",
        )

        assert button["type"] == "Action.OpenUrl"
        assert button["url"] == "https://example.com/debate/123"


class TestTeamsWithoutHttpx:
    """Tests for behavior when httpx is not available."""

    @pytest.mark.asyncio
    async def test_send_without_httpx(self):
        """Should return error when httpx not available."""
        from aragora.connectors.chat.teams import TeamsConnector

        connector = TeamsConnector(app_id="app-123", app_password="secret")

        with patch("aragora.connectors.chat.teams.HTTPX_AVAILABLE", False):
            # Need to reimport to get patched value
            connector_module = __import__(
                "aragora.connectors.chat.teams", fromlist=["TeamsConnector"]
            )
            patched_connector = connector_module.TeamsConnector(
                app_id="app-123", app_password="secret"
            )

            result = await patched_connector.send_message(
                channel_id="conv-456",
                text="Test",
            )

            # When httpx not available, should fail gracefully
            assert result.success is False or result.error is not None
