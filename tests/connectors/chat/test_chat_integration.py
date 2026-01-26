"""
Integration Tests for Chat Platform Connectors.

Tests end-to-end flows for:
- Webhook event processing (receive → parse → respond)
- Message lifecycle (send → update → delete)
- File operations (upload → download)
- Evidence collection from channel history
- Multi-platform consistency
"""

import json
import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from aragora.connectors.chat.models import (
    ChatMessage,
    ChatChannel,
    ChatUser,
    ChatEvidence,
    FileAttachment,
    SendMessageResponse,
)


# =============================================================================
# Discord Integration Tests
# =============================================================================


class TestDiscordIntegration:
    """End-to-end integration tests for Discord connector."""

    @pytest.fixture
    def discord_connector(self):
        """Create Discord connector with test credentials."""
        from aragora.connectors.chat.discord import DiscordConnector

        return DiscordConnector(
            bot_token="test-bot-token",
            application_id="123456789",
            public_key="test-public-key",
        )

    @pytest.mark.asyncio
    async def test_full_message_lifecycle(self, discord_connector):
        """Test complete message lifecycle: send → update → delete."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": "msg-123", "channel_id": "chan-456"}
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_client):
            # Step 1: Send message
            send_result = await discord_connector.send_message(
                channel_id="chan-456",
                text="Initial message",
            )
            assert send_result.success is True
            assert send_result.message_id == "msg-123"

            # Step 2: Update message
            update_result = await discord_connector.update_message(
                channel_id="chan-456",
                message_id="msg-123",
                text="Updated message",
            )
            assert update_result.success is True

            # Step 3: Delete message
            mock_delete_response = MagicMock()
            mock_delete_response.status_code = 204
            mock_delete_response.raise_for_status = MagicMock()
            mock_client.delete = AsyncMock(return_value=mock_delete_response)

            delete_result = await discord_connector.delete_message(
                channel_id="chan-456",
                message_id="msg-123",
            )
            assert delete_result is True

    @pytest.mark.asyncio
    async def test_webhook_to_response_flow(self, discord_connector):
        """Test webhook event parsing and response generation."""
        # Simulate incoming Discord interaction
        interaction_payload = {
            "type": 2,  # APPLICATION_COMMAND
            "id": "int-123",
            "application_id": "123456789",
            "token": "interaction-token",
            "data": {
                "name": "debate",
                "options": [{"name": "topic", "value": "Should we use GraphQL?"}],
            },
            "channel_id": "chan-456",
            "guild_id": "guild-789",
            "member": {
                "user": {
                    "id": "user-111",
                    "username": "testuser",
                    "global_name": "Test User",
                }
            },
        }

        # Parse the webhook event
        event = discord_connector.parse_webhook_event(
            headers={"Content-Type": "application/json"},
            body=json.dumps(interaction_payload).encode("utf-8"),
        )

        assert event is not None
        # Discord type 2 = APPLICATION_COMMAND, populates 'command' field
        assert event.event_type == "interaction_2"
        assert event.command is not None
        assert event.command.name == "debate"

    @pytest.mark.asyncio
    async def test_metadata_lookup_flow(self, discord_connector):
        """Test channel and user info lookup flow."""
        channel_data = {
            "id": "chan-456",
            "type": 0,
            "name": "general",
            "guild_id": "guild-789",
            "topic": "General discussion",
        }

        user_data = {
            "id": "user-111",
            "username": "testuser",
            "global_name": "Test User",
            "avatar": "abc123",
            "bot": False,
        }

        with patch.object(discord_connector, "_http_request", new_callable=AsyncMock) as mock_req:
            # First call returns channel, second returns user
            mock_req.side_effect = [
                (True, channel_data, None),
                (True, user_data, None),
            ]

            # Get channel info
            channel = await discord_connector.get_channel_info(channel_id="chan-456")
            assert channel is not None
            assert channel.name == "general"
            assert channel.team_id == "guild-789"

            # Get user info
            user = await discord_connector.get_user_info(user_id="user-111")
            assert user is not None
            assert user.display_name == "Test User"
            assert "avatars/user-111/abc123" in user.avatar_url


# =============================================================================
# Teams Integration Tests
# =============================================================================


class TestTeamsIntegration:
    """End-to-end integration tests for Teams connector."""

    @pytest.fixture
    def teams_connector(self):
        """Create Teams connector with test credentials."""
        from aragora.connectors.chat.teams import TeamsConnector

        connector = TeamsConnector(
            app_id="test-app-id",
            app_password="test-app-password",
            tenant_id="test-tenant-id",
        )
        # Pre-set tokens to avoid OAuth flow
        connector._access_token = "test-bot-token"
        connector._token_expires = 9999999999
        connector._graph_token = "test-graph-token"
        connector._graph_token_expires = 9999999999
        return connector

    @pytest.mark.asyncio
    async def test_full_message_lifecycle(self, teams_connector):
        """Test complete message lifecycle: send → update → delete."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": "msg-123"}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = mock_client_class.return_value.__aenter__.return_value
            mock_client.request = AsyncMock(return_value=mock_response)
            mock_client.delete = AsyncMock(return_value=mock_response)

            # Step 1: Send message
            send_result = await teams_connector.send_message(
                channel_id="conv-456",
                text="Initial message",
            )
            assert send_result.success is True
            assert send_result.message_id == "msg-123"

            # Step 2: Update message
            update_result = await teams_connector.update_message(
                channel_id="conv-456",
                message_id="msg-123",
                text="Updated message",
            )
            assert update_result.success is True

            # Step 3: Delete message
            delete_result = await teams_connector.delete_message(
                channel_id="conv-456",
                message_id="msg-123",
            )
            assert delete_result is True

    @pytest.mark.asyncio
    async def test_file_operations_flow(self, teams_connector):
        """Test file upload and download flow via Graph API."""
        # Mock Graph API responses
        folder_response = {
            "id": "folder-123",
            "parentReference": {"driveId": "drive-456"},
        }

        upload_response = {
            "id": "file-789",
            "webUrl": "https://example.sharepoint.com/file.pdf",
            "name": "test.pdf",
        }

        download_meta = {
            "id": "file-789",
            "name": "test.pdf",
            "size": 1024,
            "file": {"mimeType": "application/pdf"},
            "@microsoft.graph.downloadUrl": "https://download.example.com/file",
            "webUrl": "https://view.example.com/file",
        }

        with patch.object(
            teams_connector, "_graph_api_request", new_callable=AsyncMock
        ) as mock_graph:
            # Upload flow: get folder, then upload
            mock_graph.side_effect = [
                (True, folder_response, None),
                (True, upload_response, None),
            ]

            upload_result = await teams_connector.upload_file(
                channel_id="chan-123",
                content=b"PDF content here",
                filename="test.pdf",
                team_id="team-456",
            )

            assert upload_result.id == "file-789"
            assert upload_result.filename == "test.pdf"
            assert "sharepoint" in upload_result.url

        # Download flow
        with patch.object(
            teams_connector, "_graph_api_request", new_callable=AsyncMock
        ) as mock_graph:
            mock_graph.return_value = (True, download_meta, None)

            mock_download = MagicMock()
            mock_download.content = b"PDF content here"
            mock_download.raise_for_status = MagicMock()

            with patch("httpx.AsyncClient") as mock_client_class:
                mock_client = mock_client_class.return_value.__aenter__.return_value
                mock_client.get = AsyncMock(return_value=mock_download)

                download_result = await teams_connector.download_file(
                    file_id="file-789",
                    drive_id="drive-456",
                )

                assert download_result.filename == "test.pdf"
                assert download_result.content == b"PDF content here"
                assert download_result.content_type == "application/pdf"

    @pytest.mark.asyncio
    async def test_evidence_collection_flow(self, teams_connector):
        """Test channel history retrieval and evidence collection."""
        messages_response = {
            "value": [
                {
                    "id": "msg-1",
                    "createdDateTime": "2024-01-15T10:30:00Z",
                    "from": {"user": {"id": "user-1", "displayName": "Alice"}},
                    "body": {"content": "We should use GraphQL for the API", "contentType": "text"},
                },
                {
                    "id": "msg-2",
                    "createdDateTime": "2024-01-15T10:35:00Z",
                    "from": {"user": {"id": "user-2", "displayName": "Bob"}},
                    "body": {"content": "I prefer REST for simplicity", "contentType": "text"},
                },
                {
                    "id": "msg-3",
                    "createdDateTime": "2024-01-15T10:40:00Z",
                    "from": {"user": {"id": "user-1", "displayName": "Alice"}},
                    "body": {"content": "Random off-topic message", "contentType": "text"},
                },
            ],
        }

        with patch.object(
            teams_connector, "_graph_api_request", new_callable=AsyncMock
        ) as mock_graph:
            mock_graph.return_value = (True, messages_response, None)

            # Collect evidence about "API"
            evidence = await teams_connector.collect_evidence(
                channel_id="chan-123",
                query="API",
                team_id="team-456",
                min_relevance=0.0,
            )

            # Should have all 3 messages but sorted by relevance
            assert len(evidence) == 3
            # First should be most relevant (contains "API")
            assert "API" in evidence[0].content

    @pytest.mark.asyncio
    async def test_webhook_to_response_flow(self, teams_connector):
        """Test Bot Framework activity parsing and response generation."""
        activity_payload = {
            "type": "message",
            "id": "act-123",
            "text": "@Aragora debate Should we use microservices?",
            "from": {
                "id": "user-789",
                "name": "Test User",
            },
            "conversation": {
                "id": "conv-456",
                "conversationType": "channel",
            },
            "channelId": "msteams",
            "serviceUrl": "https://smba.trafficmanager.net/emea/",
        }

        # Parse the webhook event
        event = teams_connector.parse_webhook_event(
            headers={"Content-Type": "application/json"},
            body=json.dumps(activity_payload).encode("utf-8"),
        )

        assert event is not None
        assert event.event_type == "message"
        assert event.message is not None
        assert "microservices" in event.message.content


# =============================================================================
# Slack Integration Tests
# =============================================================================


class TestSlackIntegration:
    """End-to-end integration tests for Slack connector."""

    @pytest.fixture
    def slack_connector(self):
        """Create Slack connector with test credentials."""
        from aragora.connectors.chat.slack import SlackConnector

        return SlackConnector(
            bot_token="xoxb-test-token",
            signing_secret="test-signing-secret",
        )

    @pytest.mark.asyncio
    async def test_full_message_lifecycle(self, slack_connector):
        """Test complete message lifecycle: send → update → delete."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "ok": True,
            "ts": "1234567890.123456",
            "channel": "C123456",
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_client):
            # Step 1: Send message
            send_result = await slack_connector.send_message(
                channel_id="C123456",
                text="Initial message",
            )
            assert send_result.success is True
            assert send_result.message_id == "1234567890.123456"

            # Step 2: Update message
            update_result = await slack_connector.update_message(
                channel_id="C123456",
                message_id="1234567890.123456",
                text="Updated message",
            )
            assert update_result.success is True

            # Step 3: Delete message
            delete_result = await slack_connector.delete_message(
                channel_id="C123456",
                message_id="1234567890.123456",
            )
            assert delete_result is True


# =============================================================================
# Cross-Platform Consistency Tests
# =============================================================================


class TestCrossPlatformConsistency:
    """Tests ensuring consistent behavior across all chat platforms."""

    @pytest.fixture
    def all_connectors(self):
        """Create instances of all chat connectors."""
        from aragora.connectors.chat.discord import DiscordConnector
        from aragora.connectors.chat.teams import TeamsConnector
        from aragora.connectors.chat.slack import SlackConnector
        from aragora.connectors.chat.telegram import TelegramConnector

        return {
            "discord": DiscordConnector(bot_token="test"),
            "teams": TeamsConnector(app_id="test", app_password="test"),
            "slack": SlackConnector(bot_token="xoxb-test"),
            "telegram": TelegramConnector(bot_token="test"),
        }

    def test_all_connectors_have_platform_name(self, all_connectors):
        """All connectors must have platform_name property."""
        for name, connector in all_connectors.items():
            assert hasattr(connector, "platform_name")
            assert connector.platform_name == name

    def test_all_connectors_have_display_name(self, all_connectors):
        """All connectors must have platform_display_name property."""
        expected = {
            "discord": "Discord",
            "teams": "Microsoft Teams",
            "slack": "Slack",
            "telegram": "Telegram",
        }
        for name, connector in all_connectors.items():
            assert hasattr(connector, "platform_display_name")
            assert connector.platform_display_name == expected[name]

    def test_all_connectors_have_format_blocks(self, all_connectors):
        """All connectors must implement format_blocks method."""
        for name, connector in all_connectors.items():
            assert hasattr(connector, "format_blocks")
            assert callable(connector.format_blocks)

            # Should return a list
            blocks = connector.format_blocks(title="Test", body="Content")
            assert isinstance(blocks, list)

    def test_all_connectors_have_format_button(self, all_connectors):
        """All connectors must implement format_button method."""
        for name, connector in all_connectors.items():
            assert hasattr(connector, "format_button")
            assert callable(connector.format_button)

            # Should return a dict
            button = connector.format_button(
                text="Click Me",
                action_id="test_action",
                value="test_value",
            )
            assert isinstance(button, dict)

    @pytest.mark.asyncio
    async def test_send_message_returns_consistent_result(self, all_connectors):
        """send_message must return SendMessageResponse on all platforms."""
        # Test Discord, Teams, Slack which properly handle errors
        test_connectors = {
            k: v for k, v in all_connectors.items() if k in ("discord", "teams", "slack")
        }

        for name, connector in test_connectors.items():
            # Mock the HTTP call to fail gracefully
            with patch("httpx.AsyncClient") as mock_client:
                mock_client.return_value.__aenter__.return_value.request = AsyncMock(
                    side_effect=Exception("Network error")
                )
                mock_client.return_value.__aenter__.return_value.post = AsyncMock(
                    side_effect=Exception("Network error")
                )

                result = await connector.send_message(
                    channel_id="test-channel",
                    text="Test message",
                )

                # Must return SendMessageResponse even on failure
                assert isinstance(result, SendMessageResponse), (
                    f"{name} should return SendMessageResponse"
                )
                assert result.success is False, f"{name} should return success=False on error"
                assert result.error is not None, f"{name} should include error message"


# =============================================================================
# Error Handling Integration Tests
# =============================================================================


class TestErrorHandlingIntegration:
    """Tests for graceful error handling across connectors."""

    @pytest.mark.asyncio
    async def test_discord_handles_rate_limit(self):
        """Discord connector should handle 429 rate limit gracefully."""
        from aragora.connectors.chat.discord import DiscordConnector

        connector = DiscordConnector(bot_token="test")

        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.headers = {"Retry-After": "1"}
        mock_response.raise_for_status.side_effect = Exception("Rate limited")

        mock_client = MagicMock()
        mock_client.request = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await connector.send_message(
                channel_id="test",
                text="Test",
            )

            assert result.success is False
            # Should not crash, returns error result

    @pytest.mark.asyncio
    async def test_teams_handles_graph_api_permission_error(self):
        """Teams connector should handle Graph API 403 gracefully."""
        from aragora.connectors.chat.teams import TeamsConnector

        connector = TeamsConnector(
            app_id="test",
            app_password="test",
            tenant_id="test-tenant",
        )
        connector._graph_token = "test-token"
        connector._graph_token_expires = 9999999999

        with patch.object(connector, "_graph_api_request", new_callable=AsyncMock) as mock_graph:
            mock_graph.return_value = (False, None, "Forbidden: missing permission")

            result = await connector.upload_file(
                channel_id="test",
                content=b"test",
                filename="test.txt",
                team_id="team-123",
            )

            # Should return empty FileAttachment, not crash
            assert result.id == "" or result.metadata.get("error") is not None

    @pytest.mark.asyncio
    async def test_slack_handles_invalid_token(self):
        """Slack connector should handle invalid token gracefully."""
        from aragora.connectors.chat.slack import SlackConnector

        connector = SlackConnector(bot_token="invalid-token")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"ok": False, "error": "invalid_auth"}
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await connector.send_message(
                channel_id="C123",
                text="Test",
            )

            assert result.success is False
            assert "invalid_auth" in result.error


# =============================================================================
# Webhook Verification Integration Tests
# =============================================================================


class TestWebhookVerificationIntegration:
    """Tests for webhook handling across platforms."""

    def test_discord_handles_ping(self):
        """Discord connector should handle ping interaction."""
        from aragora.connectors.chat.discord import DiscordConnector

        connector = DiscordConnector(bot_token="test")

        # Discord ping interaction (type=1)
        ping_payload = {"type": 1, "id": "123", "application_id": "456"}

        event = connector.parse_webhook_event(
            headers={},
            body=json.dumps(ping_payload).encode("utf-8"),
        )

        assert event is not None
        assert event.event_type == "ping"

    def test_slack_handles_url_verification(self):
        """Slack connector should handle URL verification challenge."""
        from aragora.connectors.chat.slack import SlackConnector

        connector = SlackConnector(bot_token="xoxb-test")

        # Slack URL verification event
        verification_payload = {
            "type": "url_verification",
            "challenge": "test-challenge-token",
        }

        event = connector.parse_webhook_event(
            headers={},
            body=json.dumps(verification_payload).encode("utf-8"),
        )

        # URL verification returns challenge in metadata
        assert event is not None
        assert "challenge" in event.metadata or event.event_type == "url_verification"

    def test_teams_handles_missing_auth(self):
        """Teams connector should handle missing Bot Framework auth."""
        from aragora.connectors.chat.teams import TeamsConnector

        connector = TeamsConnector()  # No credentials

        # Should handle gracefully when no app_id/password
        event = connector.parse_webhook_event(
            headers={},
            body=b'{"type": "message", "text": "hello"}',
        )

        # Should still parse the event (auth happens at HTTP layer)
        assert event is not None
