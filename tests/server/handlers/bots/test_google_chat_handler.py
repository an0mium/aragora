"""Tests for Google Chat bot handler."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.bots.google_chat import (
    GoogleChatHandler,
    get_google_chat_handler,
    get_google_chat_connector,
)


# =============================================================================
# Test Handler Initialization
# =============================================================================


class TestGoogleChatHandlerInit:
    """Tests for Google Chat handler initialization."""

    def test_handler_routes(self):
        """Should define correct routes."""
        handler = GoogleChatHandler({})
        assert "/api/v1/bots/google-chat/webhook" in handler.ROUTES
        assert "/api/v1/bots/google-chat/status" in handler.ROUTES

    def test_can_handle_webhook_route(self):
        """Should handle webhook route."""
        handler = GoogleChatHandler({})
        assert handler.can_handle("/api/v1/bots/google-chat/webhook") is True

    def test_can_handle_status_route(self):
        """Should handle status route."""
        handler = GoogleChatHandler({})
        assert handler.can_handle("/api/v1/bots/google-chat/status") is True

    def test_cannot_handle_unknown_route(self):
        """Should not handle unknown routes."""
        handler = GoogleChatHandler({})
        assert handler.can_handle("/api/v1/bots/unknown") is False


# =============================================================================
# Test Factory Functions
# =============================================================================


class TestFactoryFunctions:
    """Tests for factory functions."""

    def test_get_google_chat_handler(self):
        """Should return GoogleChatHandler instance."""
        # Reset singleton for testing
        import aragora.server.handlers.bots.google_chat as gchat_module

        gchat_module._google_chat_handler = None

        handler = get_google_chat_handler()
        assert isinstance(handler, GoogleChatHandler)

    def test_get_google_chat_connector_no_credentials(self):
        """Should return None when no credentials configured."""
        # Reset singleton for testing
        import aragora.server.handlers.bots.google_chat as gchat_module

        gchat_module._google_chat_connector = None

        with patch("aragora.server.handlers.bots.google_chat.GOOGLE_CHAT_CREDENTIALS", ""):
            connector = get_google_chat_connector()
        assert connector is None


# =============================================================================
# Test Status Endpoint
# =============================================================================


class TestGoogleChatStatus:
    """Tests for Google Chat status endpoint."""

    @pytest.mark.asyncio
    async def test_get_status(self):
        """Should return status information."""
        handler = GoogleChatHandler({})

        with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
            mock_auth.return_value = MagicMock(permissions=["bots:read"])
            with patch.object(handler, "check_permission"):
                mock_handler = MagicMock()
                result = await handler.handle("/api/v1/bots/google-chat/status", {}, mock_handler)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        assert body["platform"] == "google_chat"
        assert "enabled" in body
        assert "credentials_configured" in body
        assert "project_id_configured" in body

    @pytest.mark.asyncio
    async def test_get_status_requires_auth(self):
        """Should require authentication for status endpoint."""
        from aragora.server.handlers.secure import UnauthorizedError

        handler = GoogleChatHandler({})

        with patch.object(handler, "get_auth_context", new_callable=AsyncMock) as mock_auth:
            mock_auth.side_effect = UnauthorizedError("No auth")
            mock_handler = MagicMock()
            result = await handler.handle("/api/v1/bots/google-chat/status", {}, mock_handler)

        assert result is not None
        assert result.status_code == 401


# =============================================================================
# Test Webhook Events
# =============================================================================


class TestGoogleChatWebhook:
    """Tests for Google Chat webhook event handling."""

    def test_handle_message_event(self):
        """Should handle MESSAGE event."""
        handler = GoogleChatHandler({})

        event = {
            "type": "MESSAGE",
            "message": {
                "text": "Hello bot",
                "sender": {"displayName": "Test User"},
            },
            "space": {"name": "spaces/123", "type": "DM"},
            "user": {"displayName": "Test User"},
        }

        mock_request = MagicMock()
        mock_request.headers = {"Content-Length": str(len(json.dumps(event)))}
        mock_request.rfile.read.return_value = json.dumps(event).encode()

        result = handler.handle_post("/api/v1/bots/google-chat/webhook", {}, mock_request)

        assert result is not None
        assert result.status_code == 200

    def test_handle_added_to_space_event(self):
        """Should handle ADDED_TO_SPACE event."""
        handler = GoogleChatHandler({})

        event = {
            "type": "ADDED_TO_SPACE",
            "space": {"name": "spaces/123", "displayName": "Test Space"},
        }

        mock_request = MagicMock()
        mock_request.headers = {"Content-Length": str(len(json.dumps(event)))}
        mock_request.rfile.read.return_value = json.dumps(event).encode()

        result = handler.handle_post("/api/v1/bots/google-chat/webhook", {}, mock_request)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        # Should return welcome card
        assert "cardsV2" in body or "text" in body

    def test_handle_removed_from_space_event(self):
        """Should handle REMOVED_FROM_SPACE event."""
        handler = GoogleChatHandler({})

        event = {
            "type": "REMOVED_FROM_SPACE",
            "space": {"name": "spaces/123"},
        }

        mock_request = MagicMock()
        mock_request.headers = {"Content-Length": str(len(json.dumps(event)))}
        mock_request.rfile.read.return_value = json.dumps(event).encode()

        result = handler.handle_post("/api/v1/bots/google-chat/webhook", {}, mock_request)

        assert result is not None
        assert result.status_code == 200

    def test_handle_card_clicked_event(self):
        """Should handle CARD_CLICKED event."""
        handler = GoogleChatHandler({})

        event = {
            "type": "CARD_CLICKED",
            "action": {
                "actionMethodName": "vote_agree",
                "parameters": [{"key": "debate_id", "value": "debate123"}],
            },
            "user": {"name": "users/123", "displayName": "Test User"},
            "space": {"name": "spaces/456"},
        }

        mock_request = MagicMock()
        mock_request.headers = {"Content-Length": str(len(json.dumps(event)))}
        mock_request.rfile.read.return_value = json.dumps(event).encode()

        result = handler.handle_post("/api/v1/bots/google-chat/webhook", {}, mock_request)

        assert result is not None
        assert result.status_code == 200

    def test_handle_invalid_json(self):
        """Should handle invalid JSON gracefully."""
        handler = GoogleChatHandler({})

        mock_request = MagicMock()
        mock_request.headers = {"Content-Length": "15"}
        mock_request.rfile.read.return_value = b"not valid json"

        result = handler.handle_post("/api/v1/bots/google-chat/webhook", {}, mock_request)

        assert result is not None
        assert result.status_code == 400

    def test_handle_unknown_event_type(self):
        """Should handle unknown event types gracefully."""
        handler = GoogleChatHandler({})

        event = {"type": "UNKNOWN_EVENT"}

        mock_request = MagicMock()
        mock_request.headers = {"Content-Length": str(len(json.dumps(event)))}
        mock_request.rfile.read.return_value = json.dumps(event).encode()

        result = handler.handle_post("/api/v1/bots/google-chat/webhook", {}, mock_request)

        assert result is not None
        assert result.status_code == 200


# =============================================================================
# Test Slash Commands
# =============================================================================


class TestGoogleChatSlashCommands:
    """Tests for Google Chat slash command handling."""

    def test_handle_help_command(self):
        """Should handle /help slash command."""
        handler = GoogleChatHandler({})

        event = {
            "type": "MESSAGE",
            "message": {
                "text": "/help",
                "slashCommand": {"commandName": "/help"},
            },
            "space": {"name": "spaces/123"},
            "user": {"displayName": "Test User", "name": "users/456"},
        }

        mock_request = MagicMock()
        mock_request.headers = {"Content-Length": str(len(json.dumps(event)))}
        mock_request.rfile.read.return_value = json.dumps(event).encode()

        result = handler.handle_post("/api/v1/bots/google-chat/webhook", {}, mock_request)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        # Should return help card
        assert "cardsV2" in body

    def test_handle_status_command(self):
        """Should handle /status slash command."""
        handler = GoogleChatHandler({})

        event = {
            "type": "MESSAGE",
            "message": {
                "text": "/status",
                "slashCommand": {"commandName": "/status"},
            },
            "space": {"name": "spaces/123"},
            "user": {"displayName": "Test User", "name": "users/456"},
        }

        mock_request = MagicMock()
        mock_request.headers = {"Content-Length": str(len(json.dumps(event)))}
        mock_request.rfile.read.return_value = json.dumps(event).encode()

        result = handler.handle_post("/api/v1/bots/google-chat/webhook", {}, mock_request)

        assert result is not None
        assert result.status_code == 200

    def test_handle_debate_command_empty_topic(self):
        """Should handle /debate with empty topic."""
        handler = GoogleChatHandler({})

        event = {
            "type": "MESSAGE",
            "message": {
                "text": "/debate",
                "slashCommand": {"commandName": "/debate"},
                "argumentText": "",
            },
            "space": {"name": "spaces/123"},
            "user": {"displayName": "Test User", "name": "users/456"},
        }

        mock_request = MagicMock()
        mock_request.headers = {"Content-Length": str(len(json.dumps(event)))}
        mock_request.rfile.read.return_value = json.dumps(event).encode()

        result = handler.handle_post("/api/v1/bots/google-chat/webhook", {}, mock_request)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        # Should return error about empty topic
        assert "cardsV2" in body or "text" in body

    def test_handle_unknown_command(self):
        """Should handle unknown slash commands."""
        handler = GoogleChatHandler({})

        event = {
            "type": "MESSAGE",
            "message": {
                "text": "/unknowncmd",
                "slashCommand": {"commandName": "/unknowncmd"},
            },
            "space": {"name": "spaces/123"},
            "user": {"displayName": "Test User", "name": "users/456"},
        }

        mock_request = MagicMock()
        mock_request.headers = {"Content-Length": str(len(json.dumps(event)))}
        mock_request.rfile.read.return_value = json.dumps(event).encode()

        result = handler.handle_post("/api/v1/bots/google-chat/webhook", {}, mock_request)

        assert result is not None
        assert result.status_code == 200


# =============================================================================
# Test Card Response Builder
# =============================================================================


class TestCardResponseBuilder:
    """Tests for Google Chat card response builder."""

    def test_card_response_with_title(self):
        """Should build card response with title."""
        handler = GoogleChatHandler({})

        result = handler._card_response(title="Test Title", body="Test body")

        assert result.status_code == 200
        body = json.loads(result.body)
        assert "cardsV2" in body
        card = body["cardsV2"][0]["card"]
        assert len(card["sections"]) >= 2  # header + body

    def test_card_response_with_fields(self):
        """Should build card response with fields."""
        handler = GoogleChatHandler({})

        result = handler._card_response(
            title="Test",
            fields=[("Field1", "Value1"), ("Field2", "Value2")],
        )

        assert result.status_code == 200
        body = json.loads(result.body)
        assert "cardsV2" in body

    def test_card_response_body_only(self):
        """Should build simple text response when no title."""
        handler = GoogleChatHandler({})

        result = handler._card_response(body="Simple text message")

        assert result.status_code == 200
        body = json.loads(result.body)
        # May be card or text depending on implementation
        assert "cardsV2" in body or "text" in body


# =============================================================================
# Test Input Validation
# =============================================================================


class TestInputValidation:
    """Tests for input validation."""

    def test_debate_topic_too_short(self):
        """Should reject debate topic that is too short."""
        handler = GoogleChatHandler({})

        event = {
            "type": "MESSAGE",
            "message": {
                "text": "/debate hi",
                "slashCommand": {"commandName": "/debate"},
                "argumentText": "hi",
            },
            "space": {"name": "spaces/123"},
            "user": {"displayName": "Test User", "name": "users/456"},
        }

        mock_request = MagicMock()
        mock_request.headers = {"Content-Length": str(len(json.dumps(event)))}
        mock_request.rfile.read.return_value = json.dumps(event).encode()

        result = handler.handle_post("/api/v1/bots/google-chat/webhook", {}, mock_request)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        # Should contain error message about topic being too short
        sections = body.get("cardsV2", [{}])[0].get("card", {}).get("sections", [])
        section_texts = [
            s.get("widgets", [{}])[0].get("textParagraph", {}).get("text", "")
            for s in sections
            if s.get("widgets")
        ]
        assert any("too short" in t.lower() for t in section_texts if t)
