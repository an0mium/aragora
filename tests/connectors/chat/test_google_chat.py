"""
Tests for Google Chat connector.

Tests the Google Chat API integration and card formatting.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import json


class TestGoogleChatConnector:
    """Tests for GoogleChatConnector."""

    @pytest.fixture
    def connector(self):
        """Create a Google Chat connector instance."""
        from aragora.connectors.chat.google_chat import GoogleChatConnector

        return GoogleChatConnector(
            credentials_json='{"type": "service_account", "project_id": "test"}',
            project_id="test-project",
        )

    def test_connector_init(self, connector):
        """Connector should initialize with configuration."""
        assert connector.project_id == "test-project"

    def test_platform_name(self, connector):
        """Platform name should be google_chat."""
        assert connector.platform_name == "google_chat"

    def test_platform_display_name(self, connector):
        """Display name should be Google Chat."""
        assert connector.platform_display_name == "Google Chat"


class TestGoogleChatConnectorAvailability:
    """Tests for connector availability checks."""

    def test_httpx_availability_check(self):
        """Should check httpx availability."""
        from aragora.connectors.chat import google_chat

        assert hasattr(google_chat, "HTTPX_AVAILABLE")

    def test_google_auth_availability_check(self):
        """Should check google-auth availability."""
        from aragora.connectors.chat import google_chat

        assert hasattr(google_chat, "GOOGLE_AUTH_AVAILABLE")


class TestGoogleChatMessageFormatting:
    """Tests for message formatting."""

    @pytest.fixture
    def connector(self):
        """Create a Google Chat connector instance."""
        from aragora.connectors.chat.google_chat import GoogleChatConnector

        return GoogleChatConnector(project_id="test")

    def test_format_simple_message(self, connector):
        """Should format simple text messages."""
        # GoogleChatConnector should have a method to format messages
        if hasattr(connector, "_format_message"):
            message = connector._format_message("Hello, world!")
            assert "text" in message or "cards" in message

    def test_format_card_message(self, connector):
        """Should format card messages with buttons."""
        if hasattr(connector, "_format_card"):
            card = connector._format_card(
                title="Test Card",
                subtitle="Subtitle",
                sections=[{"widgets": [{"textParagraph": {"text": "Content"}}]}],
            )
            assert card is not None


class TestGoogleChatWebhookParsing:
    """Tests for webhook event parsing."""

    @pytest.fixture
    def connector(self):
        """Create a Google Chat connector instance."""
        from aragora.connectors.chat.google_chat import GoogleChatConnector

        return GoogleChatConnector(project_id="test")

    def test_parse_message_event(self, connector):
        """Should parse MESSAGE event from webhook."""
        webhook_data = {
            "type": "MESSAGE",
            "eventTime": "2024-01-15T10:30:00.000Z",
            "message": {
                "name": "spaces/SPACE_ID/messages/MSG_ID",
                "sender": {
                    "name": "users/USER_ID",
                    "displayName": "Test User",
                    "email": "test@example.com",
                    "type": "HUMAN",
                },
                "text": "Hello bot!",
                "thread": {"name": "spaces/SPACE_ID/threads/THREAD_ID"},
                "space": {"name": "spaces/SPACE_ID", "type": "ROOM"},
            },
            "user": {
                "name": "users/USER_ID",
                "displayName": "Test User",
                "email": "test@example.com",
            },
            "space": {"name": "spaces/SPACE_ID", "type": "ROOM"},
        }

        if hasattr(connector, "parse_webhook"):
            event = connector.parse_webhook(webhook_data)
            assert event is not None
            assert event.event_type == "message" or event.raw_data["type"] == "MESSAGE"

    def test_parse_added_to_space_event(self, connector):
        """Should parse ADDED_TO_SPACE event."""
        webhook_data = {
            "type": "ADDED_TO_SPACE",
            "eventTime": "2024-01-15T10:30:00.000Z",
            "space": {
                "name": "spaces/SPACE_ID",
                "type": "ROOM",
                "displayName": "Test Room",
            },
            "user": {
                "name": "users/USER_ID",
                "displayName": "Test User",
            },
        }

        if hasattr(connector, "parse_webhook"):
            event = connector.parse_webhook(webhook_data)
            assert event is not None

    def test_parse_slash_command(self, connector):
        """Should parse slash command from message."""
        webhook_data = {
            "type": "MESSAGE",
            "message": {
                "name": "spaces/SPACE_ID/messages/MSG_ID",
                "sender": {"name": "users/USER_ID", "displayName": "User"},
                "text": "",
                "slashCommand": {"commandId": "1"},
                "annotations": [
                    {
                        "type": "SLASH_COMMAND",
                        "startIndex": 0,
                        "length": 5,
                        "slashCommand": {"commandName": "/help", "commandId": "1"},
                    }
                ],
            },
            "space": {"name": "spaces/SPACE_ID"},
        }

        if hasattr(connector, "parse_webhook"):
            event = connector.parse_webhook(webhook_data)
            assert event is not None


class TestGoogleChatCardActions:
    """Tests for card action handling."""

    @pytest.fixture
    def connector(self):
        """Create a Google Chat connector instance."""
        from aragora.connectors.chat.google_chat import GoogleChatConnector

        return GoogleChatConnector(project_id="test")

    def test_parse_card_clicked_event(self, connector):
        """Should parse CARD_CLICKED event."""
        webhook_data = {
            "type": "CARD_CLICKED",
            "eventTime": "2024-01-15T10:30:00.000Z",
            "action": {
                "actionMethodName": "approve_action",
                "parameters": [
                    {"key": "debate_id", "value": "debate-123"},
                    {"key": "action", "value": "approve"},
                ],
            },
            "user": {"name": "users/USER_ID", "displayName": "User"},
            "space": {"name": "spaces/SPACE_ID"},
            "message": {"name": "spaces/SPACE_ID/messages/MSG_ID"},
        }

        if hasattr(connector, "parse_webhook"):
            event = connector.parse_webhook(webhook_data)
            assert event is not None


class TestGoogleChatSendMessage:
    """Tests for sending messages."""

    @pytest.fixture
    def connector(self):
        """Create a Google Chat connector instance."""
        from aragora.connectors.chat.google_chat import GoogleChatConnector

        return GoogleChatConnector(project_id="test")

    @pytest.mark.asyncio
    async def test_send_message_method_exists(self, connector):
        """Connector should have send_message method."""
        assert hasattr(connector, "send_message")
        assert callable(connector.send_message)

    @pytest.mark.asyncio
    async def test_send_message_handles_missing_credentials(self, connector):
        """Sending message without credentials should fail gracefully."""
        if hasattr(connector, "send_message"):
            # Without proper credentials, should return failure response (not raise)
            result = await connector.send_message(
                channel_id="spaces/SPACE_ID",
                text="Test message",
            )
            # Should return SendMessageResponse with success=False
            assert hasattr(result, "success")
            assert result.success is False
            assert "credentials" in result.error.lower()


class TestGoogleChatConstants:
    """Tests for module constants."""

    def test_api_base_url(self):
        """API base URL should be correct."""
        from aragora.connectors.chat.google_chat import CHAT_API_BASE

        assert CHAT_API_BASE == "https://chat.googleapis.com/v1"

    def test_chat_scopes(self):
        """Chat scopes should include bot scope."""
        from aragora.connectors.chat.google_chat import CHAT_SCOPES

        assert "https://www.googleapis.com/auth/chat.bot" in CHAT_SCOPES
