"""
Tests for aragora.server.handlers.social.teams - Teams Integration Handler.

Tests cover:
- Routing and method handling
- Status endpoint
- Command handling (debate, status, help, cancel)
- Interactive components (votes, view receipt)
- Notification sending
- Error handling
"""

from __future__ import annotations

import json
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.social.teams import (
    COMMAND_PATTERN,
    TeamsIntegrationHandler,
    get_teams_connector,
)

from .conftest import MockHandler, get_json, get_status_code, parse_result


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def handler(mock_server_context):
    """Create a TeamsIntegrationHandler instance."""
    return TeamsIntegrationHandler(mock_server_context)


@pytest.fixture
def mock_teams_connector():
    """Create a mock Teams connector."""
    connector = MagicMock()
    connector.send_message = AsyncMock(
        return_value=MagicMock(success=True, message_id="msg123", error=None)
    )
    return connector


# ===========================================================================
# Routing Tests
# ===========================================================================


class TestRouting:
    """Tests for route handling."""

    def test_can_handle_commands(self, handler):
        """Test handler recognizes commands endpoint."""
        assert handler.can_handle("/api/v1/integrations/teams/commands") is True

    def test_can_handle_interactive(self, handler):
        """Test handler recognizes interactive endpoint."""
        assert handler.can_handle("/api/v1/integrations/teams/interactive") is True

    def test_can_handle_status(self, handler):
        """Test handler recognizes status endpoint."""
        assert handler.can_handle("/api/v1/integrations/teams/status") is True

    def test_can_handle_notify(self, handler):
        """Test handler recognizes notify endpoint."""
        assert handler.can_handle("/api/v1/integrations/teams/notify") is True

    def test_cannot_handle_unknown(self, handler):
        """Test handler rejects unknown endpoints."""
        assert handler.can_handle("/api/v1/integrations/teams/unknown") is False
        assert handler.can_handle("/api/v1/integrations/slack/commands") is False
        assert handler.can_handle("/api/v1/other/endpoint") is False

    def test_routes_defined(self, handler):
        """Test handler has ROUTES defined."""
        assert hasattr(handler, "ROUTES")
        assert len(handler.ROUTES) >= 4

    def test_routes_include_all_endpoints(self, handler):
        """Verify all expected routes are defined."""
        expected = [
            "/api/v1/integrations/teams/commands",
            "/api/v1/integrations/teams/interactive",
            "/api/v1/integrations/teams/status",
            "/api/v1/integrations/teams/notify",
        ]
        for route in expected:
            assert route in handler.ROUTES


# ===========================================================================
# Command Pattern Tests
# ===========================================================================


class TestCommandPattern:
    """Tests for command parsing regex."""

    def test_simple_command(self):
        """Simple command should be parsed."""
        match = COMMAND_PATTERN.match("help")
        assert match is not None
        assert match.group(1) == "help"

    def test_command_with_args(self):
        """Command with arguments should be parsed."""
        match = COMMAND_PATTERN.match("debate This is the topic")
        assert match is not None
        assert match.group(1) == "debate"
        assert match.group(2) == "This is the topic"

    def test_command_with_at_mention(self):
        """Command with @mention should be parsed."""
        match = COMMAND_PATTERN.match("@aragora status")
        assert match is not None
        assert match.group(1).lower() in ["aragora", "status"]

    def test_case_insensitive(self):
        """Command matching should be case insensitive."""
        match_upper = COMMAND_PATTERN.match("HELP")
        match_lower = COMMAND_PATTERN.match("help")
        assert match_upper is not None
        assert match_lower is not None


# ===========================================================================
# Status Endpoint Tests
# ===========================================================================


class TestStatusEndpoint:
    """Tests for GET /api/integrations/teams/status."""

    def test_get_status_without_config(self, handler):
        """Status without config shows disabled."""
        mock_http = MockHandler(
            headers={"Content-Type": "application/json"},
            path="/api/v1/integrations/teams/status",
            method="GET",
        )

        with (
            patch("aragora.server.handlers.social.teams.TEAMS_APP_ID", ""),
            patch("aragora.server.handlers.social.teams.TEAMS_APP_PASSWORD", ""),
        ):
            result = handler.handle("/api/v1/integrations/teams/status", {}, mock_http)

        assert result is not None
        status_code, body = parse_result(result)
        assert status_code == 200
        assert "enabled" in body
        assert body.get("enabled") is False

    def test_get_status_with_config(self, handler):
        """Status with credentials configured shows enabled."""
        mock_http = MockHandler(
            headers={"Content-Type": "application/json"},
            path="/api/v1/integrations/teams/status",
            method="GET",
        )

        with (
            patch("aragora.server.handlers.social.teams.TEAMS_APP_ID", "app-id-123"),
            patch("aragora.server.handlers.social.teams.TEAMS_APP_PASSWORD", "password123"),
            patch("aragora.server.handlers.social.teams.get_teams_connector") as mock_connector,
        ):
            mock_connector.return_value = MagicMock()
            result = handler.handle("/api/v1/integrations/teams/status", {}, mock_http)

        status_code, body = parse_result(result)
        assert status_code == 200
        assert body.get("app_id_configured") is True
        assert body.get("password_configured") is True

    def test_get_status_shows_connector_ready(self, handler, mock_teams_connector):
        """Status should show connector readiness."""
        mock_http = MockHandler(
            headers={"Content-Type": "application/json"},
            path="/api/v1/integrations/teams/status",
            method="GET",
        )

        with patch(
            "aragora.server.handlers.social.teams.get_teams_connector",
            return_value=mock_teams_connector,
        ):
            result = handler.handle("/api/v1/integrations/teams/status", {}, mock_http)

        body = get_json(result)
        assert body.get("connector_ready") is True


# ===========================================================================
# Command Endpoint Tests
# ===========================================================================


class TestCommandEndpoint:
    """Tests for POST /api/integrations/teams/commands."""

    def test_help_command(self, handler, mock_teams_connector):
        """Help command should return help text."""
        body = {
            "text": "help",
            "conversation": {"id": "conv123"},
            "serviceUrl": "https://smba.trafficmanager.net/amer/",
            "from": {"id": "user123", "name": "Test User"},
        }

        mock_http = MockHandler.with_json_body(body, path="/api/v1/integrations/teams/commands")

        with patch(
            "aragora.server.handlers.social.teams.get_teams_connector",
            return_value=mock_teams_connector,
        ):
            result = handler.handle_post("/api/v1/integrations/teams/commands", {}, mock_http)

        assert result is not None
        assert get_status_code(result) == 200

    def test_status_command_no_debate(self, handler):
        """Status command with no active debate."""
        body = {
            "text": "status",
            "conversation": {"id": "conv123"},
            "serviceUrl": "https://smba.trafficmanager.net/amer/",
            "from": {"id": "user123"},
        }

        mock_http = MockHandler.with_json_body(body, path="/api/v1/integrations/teams/commands")

        result = handler.handle_post("/api/v1/integrations/teams/commands", {}, mock_http)

        assert result is not None
        response_body = get_json(result)
        assert response_body.get("active") is False

    def test_status_command_with_debate(self, handler):
        """Status command with active debate."""
        # Add an active debate
        handler._active_debates["conv123"] = {
            "topic": "Test topic",
            "status": "running",
        }

        body = {
            "text": "status",
            "conversation": {"id": "conv123"},
            "serviceUrl": "https://smba.trafficmanager.net/amer/",
            "from": {"id": "user123"},
        }

        mock_http = MockHandler.with_json_body(body, path="/api/v1/integrations/teams/commands")

        result = handler.handle_post("/api/v1/integrations/teams/commands", {}, mock_http)

        assert result is not None
        response_body = get_json(result)
        assert response_body.get("active") is True
        assert response_body.get("topic") == "Test topic"

    def test_cancel_command_with_debate(self, handler):
        """Cancel command should remove active debate."""
        # Add an active debate
        handler._active_debates["conv123"] = {
            "topic": "Test topic",
            "status": "running",
        }

        body = {
            "text": "cancel",
            "conversation": {"id": "conv123"},
            "serviceUrl": "https://smba.trafficmanager.net/amer/",
            "from": {"id": "user123"},
        }

        mock_http = MockHandler.with_json_body(body, path="/api/v1/integrations/teams/commands")

        result = handler.handle_post("/api/v1/integrations/teams/commands", {}, mock_http)

        assert result is not None
        assert "conv123" not in handler._active_debates

    def test_cancel_command_no_debate(self, handler):
        """Cancel command with no active debate."""
        body = {
            "text": "cancel",
            "conversation": {"id": "conv123"},
            "serviceUrl": "https://smba.trafficmanager.net/amer/",
            "from": {"id": "user123"},
        }

        mock_http = MockHandler.with_json_body(body, path="/api/v1/integrations/teams/commands")

        result = handler.handle_post("/api/v1/integrations/teams/commands", {}, mock_http)

        assert result is not None
        # Should handle gracefully

    def test_debate_command_no_topic(self, handler, mock_teams_connector):
        """Debate command without topic should show error."""
        body = {
            "text": "debate",
            "conversation": {"id": "conv123"},
            "serviceUrl": "https://smba.trafficmanager.net/amer/",
            "from": {"id": "user123"},
        }

        mock_http = MockHandler.with_json_body(body, path="/api/v1/integrations/teams/commands")

        with patch(
            "aragora.server.handlers.social.teams.get_teams_connector",
            return_value=mock_teams_connector,
        ):
            result = handler.handle_post("/api/v1/integrations/teams/commands", {}, mock_http)

        assert result is not None
        # Should request topic

    def test_debate_command_with_topic(self, handler, mock_teams_connector):
        """Debate command with topic should start debate."""
        body = {
            "text": "debate Should AI be regulated?",
            "conversation": {"id": "conv123"},
            "serviceUrl": "https://smba.trafficmanager.net/amer/",
            "from": {"id": "user123", "name": "Test User"},
        }

        mock_http = MockHandler.with_json_body(body, path="/api/v1/integrations/teams/commands")

        with (
            patch(
                "aragora.server.handlers.social.teams.get_teams_connector",
                return_value=mock_teams_connector,
            ),
            patch("aragora.server.handlers.social.teams.create_tracked_task"),
        ):
            result = handler.handle_post("/api/v1/integrations/teams/commands", {}, mock_http)

        assert result is not None
        response_body = get_json(result)
        assert response_body.get("success") is True
        assert "conv123" in handler._active_debates

    def test_debate_command_already_running(self, handler, mock_teams_connector):
        """Debate command when one is already running should error."""
        # Add an active debate
        handler._active_debates["conv123"] = {
            "topic": "Existing topic",
            "status": "running",
        }

        body = {
            "text": "debate New topic",
            "conversation": {"id": "conv123"},
            "serviceUrl": "https://smba.trafficmanager.net/amer/",
            "from": {"id": "user123"},
        }

        mock_http = MockHandler.with_json_body(body, path="/api/v1/integrations/teams/commands")

        with patch(
            "aragora.server.handlers.social.teams.get_teams_connector",
            return_value=mock_teams_connector,
        ):
            result = handler.handle_post("/api/v1/integrations/teams/commands", {}, mock_http)

        assert result is not None
        # Should indicate debate already running

    def test_unknown_command(self, handler, mock_teams_connector):
        """Unknown command should return help or error."""
        body = {
            "text": "foobar",
            "conversation": {"id": "conv123"},
            "serviceUrl": "https://smba.trafficmanager.net/amer/",
            "from": {"id": "user123"},
        }

        mock_http = MockHandler.with_json_body(body, path="/api/v1/integrations/teams/commands")

        with patch(
            "aragora.server.handlers.social.teams.get_teams_connector",
            return_value=mock_teams_connector,
        ):
            result = handler.handle_post("/api/v1/integrations/teams/commands", {}, mock_http)

        assert result is not None

    def test_command_with_at_mention_removed(self, handler, mock_teams_connector):
        """Bot @mention should be stripped from command."""
        body = {
            "text": "<at>Aragora</at> help",
            "conversation": {"id": "conv123"},
            "serviceUrl": "https://smba.trafficmanager.net/amer/",
            "from": {"id": "user123"},
        }

        mock_http = MockHandler.with_json_body(body, path="/api/v1/integrations/teams/commands")

        with patch(
            "aragora.server.handlers.social.teams.get_teams_connector",
            return_value=mock_teams_connector,
        ):
            result = handler.handle_post("/api/v1/integrations/teams/commands", {}, mock_http)

        assert result is not None
        assert get_status_code(result) == 200

    def test_invalid_json_body(self, handler):
        """Invalid JSON should return 400."""
        mock_http = MockHandler(
            headers={"Content-Type": "application/json", "Content-Length": "12"},
            body=b"not valid json",
            path="/api/v1/integrations/teams/commands",
            method="POST",
        )

        result = handler.handle_post("/api/v1/integrations/teams/commands", {}, mock_http)

        assert result is not None
        assert get_status_code(result) == 400


# ===========================================================================
# Interactive Endpoint Tests
# ===========================================================================


class TestInteractiveEndpoint:
    """Tests for POST /api/integrations/teams/interactive."""

    def test_vote_action(self, handler):
        """Vote action should be processed."""
        body = {
            "value": {"action": "vote", "debate_id": "debate123", "choice": "for"},
            "conversation": {"id": "conv123"},
            "serviceUrl": "https://smba.trafficmanager.net/amer/",
            "from": {"id": "user123", "name": "Test User"},
        }

        mock_http = MockHandler.with_json_body(body, path="/api/v1/integrations/teams/interactive")

        result = handler.handle_post("/api/v1/integrations/teams/interactive", {}, mock_http)

        assert result is not None

    def test_cancel_debate_action(self, handler):
        """Cancel debate action should remove debate."""
        # Add an active debate
        handler._active_debates["conv123"] = {
            "topic": "Test topic",
            "status": "running",
        }

        body = {
            "value": {"action": "cancel_debate"},
            "conversation": {"id": "conv123"},
            "serviceUrl": "https://smba.trafficmanager.net/amer/",
        }

        mock_http = MockHandler.with_json_body(body, path="/api/v1/integrations/teams/interactive")

        result = handler.handle_post("/api/v1/integrations/teams/interactive", {}, mock_http)

        assert result is not None
        assert "conv123" not in handler._active_debates

    def test_view_receipt_action(self, handler):
        """View receipt action should be handled."""
        body = {
            "value": {"action": "view_receipt", "receipt_id": "receipt123"},
            "conversation": {"id": "conv123"},
            "serviceUrl": "https://smba.trafficmanager.net/amer/",
        }

        mock_http = MockHandler.with_json_body(body, path="/api/v1/integrations/teams/interactive")

        result = handler.handle_post("/api/v1/integrations/teams/interactive", {}, mock_http)

        assert result is not None

    def test_unknown_action(self, handler):
        """Unknown action should be logged but not fail."""
        body = {
            "value": {"action": "unknown_action"},
            "conversation": {"id": "conv123"},
            "serviceUrl": "https://smba.trafficmanager.net/amer/",
        }

        mock_http = MockHandler.with_json_body(body, path="/api/v1/integrations/teams/interactive")

        result = handler.handle_post("/api/v1/integrations/teams/interactive", {}, mock_http)

        assert result is not None
        response_body = get_json(result)
        assert response_body.get("status") == "unknown_action"

    def test_invalid_body(self, handler):
        """Invalid body should return 400."""
        mock_http = MockHandler(
            headers={"Content-Type": "application/json", "Content-Length": "5"},
            body=b"null",
            path="/api/v1/integrations/teams/interactive",
            method="POST",
        )

        result = handler.handle_post("/api/v1/integrations/teams/interactive", {}, mock_http)

        assert result is not None
        assert get_status_code(result) == 400


# ===========================================================================
# Notify Endpoint Tests
# ===========================================================================


class TestNotifyEndpoint:
    """Tests for POST /api/integrations/teams/notify."""

    def test_notify_success(self, handler, mock_teams_connector):
        """Notify with valid params should send message."""
        body = {
            "conversation_id": "conv123",
            "service_url": "https://smba.trafficmanager.net/amer/",
            "message": "Test notification",
        }

        mock_http = MockHandler.with_json_body(body, path="/api/v1/integrations/teams/notify")

        with patch(
            "aragora.server.handlers.social.teams.get_teams_connector",
            return_value=mock_teams_connector,
        ):
            result = handler.handle_post("/api/v1/integrations/teams/notify", {}, mock_http)

        assert result is not None
        response_body = get_json(result)
        assert response_body.get("success") is True

    def test_notify_missing_conversation_id(self, handler):
        """Notify without conversation_id should fail."""
        body = {
            "service_url": "https://smba.trafficmanager.net/amer/",
            "message": "Test notification",
        }

        mock_http = MockHandler.with_json_body(body, path="/api/v1/integrations/teams/notify")

        result = handler.handle_post("/api/v1/integrations/teams/notify", {}, mock_http)

        assert result is not None
        assert get_status_code(result) == 400

    def test_notify_missing_service_url(self, handler):
        """Notify without service_url should fail."""
        body = {
            "conversation_id": "conv123",
            "message": "Test notification",
        }

        mock_http = MockHandler.with_json_body(body, path="/api/v1/integrations/teams/notify")

        result = handler.handle_post("/api/v1/integrations/teams/notify", {}, mock_http)

        assert result is not None
        assert get_status_code(result) == 400

    def test_notify_connector_not_configured(self, handler):
        """Notify without connector should return 503."""
        body = {
            "conversation_id": "conv123",
            "service_url": "https://smba.trafficmanager.net/amer/",
            "message": "Test notification",
        }

        mock_http = MockHandler.with_json_body(body, path="/api/v1/integrations/teams/notify")

        with patch(
            "aragora.server.handlers.social.teams.get_teams_connector",
            return_value=None,
        ):
            result = handler.handle_post("/api/v1/integrations/teams/notify", {}, mock_http)

        assert result is not None
        assert get_status_code(result) == 503

    def test_notify_with_blocks(self, handler, mock_teams_connector):
        """Notify with blocks should include them."""
        blocks = [{"type": "TextBlock", "text": "Test block"}]
        body = {
            "conversation_id": "conv123",
            "service_url": "https://smba.trafficmanager.net/amer/",
            "message": "Test notification",
            "blocks": blocks,
        }

        mock_http = MockHandler.with_json_body(body, path="/api/v1/integrations/teams/notify")

        with patch(
            "aragora.server.handlers.social.teams.get_teams_connector",
            return_value=mock_teams_connector,
        ):
            result = handler.handle_post("/api/v1/integrations/teams/notify", {}, mock_http)

        assert result is not None
        mock_teams_connector.send_message.assert_called_once()
        call_kwargs = mock_teams_connector.send_message.call_args[1]
        assert call_kwargs.get("blocks") == blocks


# ===========================================================================
# Connector Tests
# ===========================================================================


class TestTeamsConnector:
    """Tests for Teams connector factory."""

    def test_get_connector_without_credentials(self):
        """Connector should not be created without credentials."""
        import aragora.server.handlers.social.teams as teams_module

        teams_module._teams_connector = None

        with (
            patch("aragora.server.handlers.social.teams.TEAMS_APP_ID", ""),
            patch("aragora.server.handlers.social.teams.TEAMS_APP_PASSWORD", ""),
        ):
            connector = get_teams_connector()

        assert connector is None

    def test_get_connector_with_credentials(self):
        """Connector should be created with credentials."""
        import aragora.server.handlers.social.teams as teams_module

        teams_module._teams_connector = None

        with (
            patch("aragora.server.handlers.social.teams.TEAMS_APP_ID", "app123"),
            patch("aragora.server.handlers.social.teams.TEAMS_APP_PASSWORD", "pass123"),
            patch("aragora.connectors.chat.teams.TeamsConnector") as mock_connector_class,
        ):
            mock_connector_class.return_value = MagicMock()
            connector = get_teams_connector()

        assert connector is not None


# ===========================================================================
# Error Handling Tests
# ===========================================================================


class TestErrorHandling:
    """Tests for error handling."""

    def test_command_exception_handled(self, handler):
        """Exceptions in command handling should be caught."""
        body = {
            "text": "debate Test topic",
            "conversation": {"id": "conv123"},
            "serviceUrl": "https://smba.trafficmanager.net/amer/",
            "from": {"id": "user123"},
        }

        mock_http = MockHandler.with_json_body(body, path="/api/v1/integrations/teams/commands")

        with patch("aragora.server.handlers.social.teams.get_teams_connector") as mock_connector:
            mock_connector.side_effect = Exception("Test error")
            result = handler.handle_post("/api/v1/integrations/teams/commands", {}, mock_http)

        assert result is not None
        assert get_status_code(result) == 500

    def test_interactive_exception_handled(self, handler):
        """Exceptions in interactive handling should be caught."""
        body = {
            "value": {"action": "vote"},
            "conversation": {"id": "conv123"},
        }

        mock_http = MockHandler.with_json_body(body, path="/api/v1/integrations/teams/interactive")

        with patch.object(handler, "_handle_vote", side_effect=Exception("Test error")):
            result = handler.handle_post("/api/v1/integrations/teams/interactive", {}, mock_http)

        assert result is not None
        assert get_status_code(result) == 500

    def test_notify_exception_handled(self, handler, mock_teams_connector):
        """Exceptions in notify should be caught."""
        body = {
            "conversation_id": "conv123",
            "service_url": "https://smba.trafficmanager.net/amer/",
            "message": "Test",
        }

        mock_http = MockHandler.with_json_body(body, path="/api/v1/integrations/teams/notify")

        mock_teams_connector.send_message = AsyncMock(side_effect=Exception("Send failed"))

        with patch(
            "aragora.server.handlers.social.teams.get_teams_connector",
            return_value=mock_teams_connector,
        ):
            result = handler.handle_post("/api/v1/integrations/teams/notify", {}, mock_http)

        assert result is not None
        assert get_status_code(result) == 500


# ===========================================================================
# Active Debates State Tests
# ===========================================================================


class TestActiveDebatesState:
    """Tests for active debates tracking."""

    def test_active_debates_initialized(self, handler):
        """Active debates dict should be initialized."""
        assert hasattr(handler, "_active_debates")
        assert isinstance(handler._active_debates, dict)

    def test_debate_added_on_start(self, handler, mock_teams_connector):
        """Starting a debate should add to active debates."""
        assert "conv456" not in handler._active_debates

        body = {
            "text": "debate New topic here",
            "conversation": {"id": "conv456"},
            "serviceUrl": "https://smba.trafficmanager.net/amer/",
            "from": {"id": "user123", "name": "Test"},
        }

        mock_http = MockHandler.with_json_body(body, path="/api/v1/integrations/teams/commands")

        with (
            patch(
                "aragora.server.handlers.social.teams.get_teams_connector",
                return_value=mock_teams_connector,
            ),
            patch("aragora.server.handlers.social.teams.create_tracked_task"),
        ):
            handler.handle_post("/api/v1/integrations/teams/commands", {}, mock_http)

        assert "conv456" in handler._active_debates
        assert handler._active_debates["conv456"]["topic"] == "New topic here"

    def test_debate_removed_on_cancel(self, handler):
        """Cancelling a debate should remove from active debates."""
        handler._active_debates["conv789"] = {"topic": "Will be cancelled"}

        body = {
            "text": "cancel",
            "conversation": {"id": "conv789"},
            "serviceUrl": "https://smba.trafficmanager.net/amer/",
            "from": {"id": "user123"},
        }

        mock_http = MockHandler.with_json_body(body, path="/api/v1/integrations/teams/commands")

        handler.handle_post("/api/v1/integrations/teams/commands", {}, mock_http)

        assert "conv789" not in handler._active_debates
