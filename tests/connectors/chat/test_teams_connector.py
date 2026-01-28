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
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...",
            "expires_in": 3600,
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            # Mock request() as _http_request uses generic request method
            mock_instance.request = AsyncMock(return_value=mock_response)

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
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": "cached-token",
            "expires_in": 3600,
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.request = AsyncMock(return_value=mock_response)

            # First call
            token1 = await connector._get_access_token()
            # Second call (should use cache)
            token2 = await connector._get_access_token()

        # Should only call API once
        assert mock_instance.request.call_count == 1
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
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": "msg-123"}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.request = AsyncMock(return_value=mock_response)

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
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": "msg-123"}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.request = AsyncMock(return_value=mock_response)

            blocks = [{"type": "TextBlock", "text": "Debate Result", "weight": "Bolder"}]

            result = await connector.send_message(
                channel_id="conv-456",
                text="Fallback",
                blocks=blocks,
            )

            # Verify adaptive card was included
            call_kwargs = mock_instance.request.call_args[1]
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
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": "msg-789"}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.request = AsyncMock(return_value=mock_response)

            result = await connector.send_message(
                channel_id="conv-456",
                text="Thread reply",
                thread_id="msg-parent",
            )

            # Verify replyToId was included
            call_kwargs = mock_instance.request.call_args[1]
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
            mock_client.return_value.__aenter__.return_value.request = AsyncMock(
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
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": "msg-123"}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.request = AsyncMock(return_value=mock_response)

            result = await connector.update_message(
                channel_id="conv-456",
                message_id="msg-123",
                text="Updated text",
            )

            # Verify request was called with PUT method
            assert mock_instance.request.called
            call_kwargs = mock_instance.request.call_args[1]
            assert call_kwargs["method"] == "PUT"

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
        mock_response.json.return_value = {"status": "ok"}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.request = AsyncMock(return_value=mock_response)

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


class TestTeamsGraphAPI:
    """Tests for Microsoft Graph API integration."""

    @pytest.mark.asyncio
    async def test_get_graph_token(self):
        """Should obtain Graph API access token."""
        from aragora.connectors.chat.teams import TeamsConnector

        connector = TeamsConnector(
            app_id="app-123",
            app_password="secret-pass",
            tenant_id="tenant-456",
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "access_token": "graph-token-xyz",
            "expires_in": 3600,
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client:
            mock_instance = mock_client.return_value.__aenter__.return_value
            mock_instance.request = AsyncMock(return_value=mock_response)

            token = await connector._get_graph_token()

        assert token == "graph-token-xyz"

    @pytest.mark.asyncio
    async def test_graph_token_requires_tenant_id(self):
        """Should require tenant ID for Graph API."""
        from aragora.connectors.chat.teams import TeamsConnector

        connector = TeamsConnector(
            app_id="app-123",
            app_password="secret-pass",
            # No tenant_id
        )

        with pytest.raises(RuntimeError, match="Tenant ID required"):
            await connector._get_graph_token()


class TestTeamsFileOperations:
    """Tests for Teams file upload/download via Graph API."""

    @pytest.mark.asyncio
    async def test_upload_file_success(self):
        """Should upload file via Graph API."""
        from aragora.connectors.chat.teams import TeamsConnector

        connector = TeamsConnector(
            app_id="app-123",
            app_password="secret-pass",
            tenant_id="tenant-456",
        )
        connector._graph_token = "graph-token"
        connector._graph_token_expires = 9999999999

        # Mock Graph API responses
        folder_response = {
            "id": "folder-id",
            "parentReference": {"driveId": "drive-id"},
        }
        upload_response = {
            "id": "file-id-123",
            "webUrl": "https://example.sharepoint.com/file",
        }

        with patch.object(connector, "_graph_api_request", new_callable=AsyncMock) as mock_req:
            # First call: get files folder, second call: upload
            mock_req.side_effect = [
                (True, folder_response, None),
                (True, upload_response, None),
            ]

            result = await connector.upload_file(
                channel_id="channel-123",
                content=b"test content",
                filename="test.txt",
                team_id="team-456",
            )

        assert result.id == "file-id-123"
        assert result.filename == "test.txt"
        assert "sharepoint" in result.url

    @pytest.mark.asyncio
    async def test_upload_file_requires_team_id(self):
        """Should require team_id for file upload."""
        from aragora.connectors.chat.teams import TeamsConnector

        connector = TeamsConnector(
            app_id="app-123",
            app_password="secret-pass",
        )

        result = await connector.upload_file(
            channel_id="channel-123",
            content=b"test content",
            filename="test.txt",
            # No team_id
        )

        assert result.id == ""
        assert "team_id required" in result.metadata.get("error", "")

    @pytest.mark.asyncio
    async def test_download_file_success(self):
        """Should download file via Graph API."""
        from aragora.connectors.chat.teams import TeamsConnector

        connector = TeamsConnector(
            app_id="app-123",
            app_password="secret-pass",
            tenant_id="tenant-456",
        )
        connector._graph_token = "graph-token"
        connector._graph_token_expires = 9999999999

        # Mock metadata response
        meta_response = {
            "name": "document.pdf",
            "size": 1024,
            "file": {"mimeType": "application/pdf"},
            "@microsoft.graph.downloadUrl": "https://download.example.com/file",
            "webUrl": "https://view.example.com/file",
        }

        # Mock download
        mock_download_response = MagicMock()
        mock_download_response.status_code = 200
        mock_download_response.content = b"PDF content here"
        mock_download_response.raise_for_status = MagicMock()

        with patch.object(connector, "_graph_api_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = (True, meta_response, None)

            with patch("httpx.AsyncClient") as mock_client:
                mock_instance = mock_client.return_value.__aenter__.return_value
                mock_instance.get = AsyncMock(return_value=mock_download_response)

                result = await connector.download_file(
                    file_id="file-id-123",
                    drive_id="drive-id",
                )

        assert result.filename == "document.pdf"
        assert result.content_type == "application/pdf"
        assert result.content == b"PDF content here"


class TestTeamsChannelHistory:
    """Tests for Teams channel history retrieval."""

    @pytest.mark.asyncio
    async def test_get_channel_history(self):
        """Should retrieve channel messages via Graph API."""
        from aragora.connectors.chat.teams import TeamsConnector

        connector = TeamsConnector(
            app_id="app-123",
            app_password="secret-pass",
            tenant_id="tenant-456",
        )
        connector._graph_token = "graph-token"
        connector._graph_token_expires = 9999999999

        # Mock Graph API response
        messages_response = {
            "value": [
                {
                    "id": "msg-1",
                    "createdDateTime": "2024-01-15T10:30:00Z",
                    "from": {"user": {"id": "user-1", "displayName": "Alice"}},
                    "body": {"content": "Hello team!", "contentType": "text"},
                },
                {
                    "id": "msg-2",
                    "createdDateTime": "2024-01-15T10:35:00Z",
                    "from": {"user": {"id": "user-2", "displayName": "Bob"}},
                    "body": {"content": "Hi Alice!", "contentType": "text"},
                },
            ],
        }

        with patch.object(connector, "_graph_api_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = (True, messages_response, None)

            messages = await connector.get_channel_history(
                channel_id="channel-123",
                team_id="team-456",
                limit=50,
            )

        assert len(messages) == 2
        assert messages[0].id == "msg-1"
        assert "Hello team" in messages[0].content
        assert messages[0].author.display_name == "Alice"

    @pytest.mark.asyncio
    async def test_get_channel_history_requires_team_id(self):
        """Should require team_id for channel history."""
        from aragora.connectors.chat.teams import TeamsConnector

        connector = TeamsConnector(
            app_id="app-123",
            app_password="secret-pass",
        )

        messages = await connector.get_channel_history(
            channel_id="channel-123",
            # No team_id
        )

        assert messages == []


class TestTeamsEvidenceCollection:
    """Tests for Teams evidence collection."""

    @pytest.mark.asyncio
    async def test_collect_evidence(self):
        """Should collect evidence from channel messages."""
        from aragora.connectors.chat.teams import TeamsConnector
        from aragora.connectors.chat.models import ChatMessage, ChatChannel, ChatUser
        from datetime import datetime

        connector = TeamsConnector(
            app_id="app-123",
            app_password="secret-pass",
            tenant_id="tenant-456",
        )

        # Create mock messages
        channel = ChatChannel(id="channel-123", platform="teams")
        user = ChatUser(id="user-1", platform="teams", display_name="Alice")

        mock_messages = [
            ChatMessage(
                id="msg-1",
                platform="teams",
                channel=channel,
                author=user,
                content="This is about database optimization",
                timestamp=datetime.utcnow(),
            ),
            ChatMessage(
                id="msg-2",
                platform="teams",
                channel=channel,
                author=user,
                content="Random unrelated message",
                timestamp=datetime.utcnow(),
            ),
        ]

        with patch.object(connector, "get_channel_history", new_callable=AsyncMock) as mock_history:
            mock_history.return_value = mock_messages

            evidence = await connector.collect_evidence(
                channel_id="channel-123",
                query="database",
                team_id="team-456",
                min_relevance=0.0,
            )

        assert len(evidence) == 2
        # First should be the one with "database" in content (higher relevance)
        assert "database" in evidence[0].content.lower()


class TestTeamsMetadataLookups:
    """Tests for Teams channel and user info lookups."""

    @pytest.mark.asyncio
    async def test_get_channel_info(self):
        """Should retrieve channel info via Graph API."""
        from aragora.connectors.chat.teams import TeamsConnector

        connector = TeamsConnector(
            app_id="app-123",
            app_password="secret-pass",
            tenant_id="tenant-456",
        )
        connector._graph_token = "graph-token"
        connector._graph_token_expires = 9999999999

        channel_response = {
            "id": "channel-123",
            "displayName": "General",
            "description": "Main channel",
            "membershipType": "standard",
            "webUrl": "https://teams.microsoft.com/channel/...",
        }

        with patch.object(connector, "_graph_api_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = (True, channel_response, None)

            channel = await connector.get_channel_info(
                channel_id="channel-123",
                team_id="team-456",
            )

        assert channel is not None
        assert channel.name == "General"
        assert channel.team_id == "team-456"

    @pytest.mark.asyncio
    async def test_get_user_info(self):
        """Should retrieve user info via Graph API."""
        from aragora.connectors.chat.teams import TeamsConnector

        connector = TeamsConnector(
            app_id="app-123",
            app_password="secret-pass",
            tenant_id="tenant-456",
        )
        connector._graph_token = "graph-token"
        connector._graph_token_expires = 9999999999

        user_response = {
            "id": "user-123",
            "displayName": "John Doe",
            "userPrincipalName": "john.doe@example.com",
            "mail": "john.doe@example.com",
            "jobTitle": "Engineer",
        }

        with patch.object(connector, "_graph_api_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = (True, user_response, None)

            user = await connector.get_user_info(user_id="user-123")

        assert user is not None
        assert user.display_name == "John Doe"
        assert user.email == "john.doe@example.com"
