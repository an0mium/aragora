"""
Tests for Teams handler.

Tests cover:
- Request routing (can_handle)
- Bot enabled detection
- Status endpoint
- Message handling
- RBAC permission constants
- Input validation
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.bots.teams.handler import TeamsHandler


@pytest.fixture
def handler():
    """Create a TeamsHandler instance."""
    with patch.dict(
        "os.environ",
        {"TEAMS_APP_ID": "test-app-id", "TEAMS_APP_PASSWORD": "test-password"},
    ):
        return TeamsHandler({})


class TestRouting:
    """Tests for request routing."""

    def test_can_handle_teams_status(self, handler):
        """Test can_handle for status endpoint."""
        assert handler.can_handle("/api/v1/bots/teams/status")

    def test_can_handle_teams_messages(self, handler):
        """Test can_handle for messages endpoint."""
        assert handler.can_handle("/api/v1/bots/teams/messages")

    def test_cannot_handle_other_paths(self, handler):
        """Test can_handle rejects non-teams paths."""
        assert not handler.can_handle("/api/v1/bots/slack/status")
        assert not handler.can_handle("/api/v1/payments/charge")

    def test_routes_list(self, handler):
        """Test ROUTES list contains expected endpoints."""
        expected_routes = [
            "/api/v1/bots/teams/status",
            "/api/v1/bots/teams/messages",
        ]
        for route in expected_routes:
            assert route in handler.ROUTES


class TestStatusEndpoint:
    """Tests for GET /status endpoint."""

    @pytest.mark.asyncio
    async def test_status_endpoint_returns_json(self, handler):
        """Test status endpoint returns valid JSON."""
        mock_handler = MagicMock()
        mock_handler.command = "GET"

        result = await handler.handle("/api/v1/bots/teams/status", {}, mock_handler)

        assert result is not None
        assert result.status_code == 200

        body = json.loads(result.body)
        assert "configured" in body or "enabled" in body or "platform" in body


class TestPermissionConstants:
    """Tests for RBAC permission constants."""

    def test_permission_constants_defined(self):
        """Test RBAC permission constants are defined."""
        from aragora.server.handlers.bots.teams.handler import (
            PERM_TEAMS_ADMIN,
            PERM_TEAMS_CARDS_RESPOND,
            PERM_TEAMS_DEBATES_CREATE,
            PERM_TEAMS_DEBATES_VOTE,
            PERM_TEAMS_MESSAGES_READ,
            PERM_TEAMS_MESSAGES_SEND,
        )

        assert PERM_TEAMS_ADMIN is not None
        assert PERM_TEAMS_CARDS_RESPOND is not None
        assert PERM_TEAMS_DEBATES_CREATE is not None
        assert PERM_TEAMS_DEBATES_VOTE is not None
        assert PERM_TEAMS_MESSAGES_READ is not None
        assert PERM_TEAMS_MESSAGES_SEND is not None

    def test_permissions_have_correct_prefix(self):
        """Test permissions have teams prefix."""
        from aragora.server.handlers.bots.teams.handler import (
            PERM_TEAMS_ADMIN,
            PERM_TEAMS_DEBATES_CREATE,
            PERM_TEAMS_MESSAGES_READ,
        )

        assert "teams" in PERM_TEAMS_ADMIN.lower()
        assert "teams" in PERM_TEAMS_DEBATES_CREATE.lower()
        assert "teams" in PERM_TEAMS_MESSAGES_READ.lower()


class TestAgentDisplayNames:
    """Tests for agent display names."""

    def test_agent_display_names_defined(self):
        """Test agent display names are defined."""
        from aragora.server.handlers.bots.teams.handler import AGENT_DISPLAY_NAMES

        assert AGENT_DISPLAY_NAMES is not None
        assert isinstance(AGENT_DISPLAY_NAMES, dict)
        assert len(AGENT_DISPLAY_NAMES) > 0

    def test_common_agents_have_names(self):
        """Test common agents have display names."""
        from aragora.server.handlers.bots.teams.handler import AGENT_DISPLAY_NAMES

        # Check for common agents
        common_agents = ["claude", "gpt4", "gemini"]
        for agent in common_agents:
            assert agent in AGENT_DISPLAY_NAMES


class TestTeamsUtils:
    """Tests for Teams utilities."""

    def test_active_debates_dict_exists(self):
        """Test active debates dict is accessible."""
        from aragora.server.handlers.bots.teams_utils import _active_debates

        assert isinstance(_active_debates, dict)

    def test_conversation_references_dict_exists(self):
        """Test conversation references dict is accessible."""
        from aragora.server.handlers.bots.teams_utils import _conversation_references

        assert isinstance(_conversation_references, dict)

    def test_get_conversation_reference_none_for_unknown(self):
        """Test get_conversation_reference returns None for unknown ID."""
        from aragora.server.handlers.bots.teams_utils import get_conversation_reference

        result = get_conversation_reference("nonexistent-conversation")

        assert result is None


class TestTokenVerification:
    """Tests for Teams token verification."""

    def test_verify_teams_token_function_exists(self):
        """Test _verify_teams_token function is importable."""
        from aragora.server.handlers.bots.teams_utils import _verify_teams_token

        assert callable(_verify_teams_token)

    @pytest.mark.asyncio
    async def test_verify_teams_token_rejects_empty(self):
        """Test token verification rejects empty token."""
        from aragora.server.handlers.bots.teams_utils import _verify_teams_token

        result = await _verify_teams_token("", app_id="test-app-id")

        # Should reject empty token
        assert result is False or result is None


class TestBotFrameworkAvailability:
    """Tests for Bot Framework availability checks."""

    def test_check_botframework_available_function_exists(self):
        """Test _check_botframework_available function is importable."""
        from aragora.server.handlers.bots.teams_utils import _check_botframework_available

        assert callable(_check_botframework_available)

    def test_check_connector_available_function_exists(self):
        """Test _check_connector_available function is importable."""
        from aragora.server.handlers.bots.teams_utils import _check_connector_available

        assert callable(_check_connector_available)


class TestEnvironmentVariables:
    """Tests for environment variable handling."""

    def test_teams_app_id_variable_handling(self):
        """Test TEAMS_APP_ID environment variable handling."""
        import os

        from aragora.server.handlers.bots import teams

        # Check that module loads without crashing
        assert hasattr(teams, "TeamsHandler")

    def test_teams_credentials_handling(self):
        """Test Teams credentials from environment."""
        from aragora.server.handlers.bots.teams.handler import (
            TEAMS_APP_ID,
            TEAMS_APP_PASSWORD,
        )

        # These might be None in test environment, but shouldn't raise
        assert TEAMS_APP_ID is None or isinstance(TEAMS_APP_ID, str)
        assert TEAMS_APP_PASSWORD is None or isinstance(TEAMS_APP_PASSWORD, str)
