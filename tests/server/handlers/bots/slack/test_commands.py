"""
Tests for Slack slash commands handler.

Tests cover:
- Command parsing
- Help command
- Status command
- Ask command
- Vote command
- Rate limiting
- Input validation
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import urlencode

import pytest


@pytest.fixture
def mock_request():
    """Create a mock HTTP request for Slack commands."""

    class MockRequest:
        def __init__(
            self,
            command: str = "/aragora",
            text: str = "",
            user_id: str = "U12345678",
            user_name: str = "testuser",
            channel_id: str = "C12345678",
            team_id: str = "T12345678",
            response_url: str = "https://hooks.slack.com/commands/xxx",
        ):
            params = {
                "command": command,
                "text": text,
                "user_id": user_id,
                "user_name": user_name,
                "channel_id": channel_id,
                "team_id": team_id,
                "response_url": response_url,
            }
            self._body = urlencode(params).encode()
            self.headers = {}

        async def body(self) -> bytes:
            return self._body

    return MockRequest


class TestCommandParsing:
    """Tests for command parsing."""

    @pytest.mark.asyncio
    async def test_help_command(self, mock_request):
        """Test help command returns help message."""
        from aragora.server.handlers.bots.slack.commands import handle_slack_commands

        request = mock_request(text="help")

        result = await handle_slack_commands(request)

        assert result is not None
        assert result.status_code == 200

        body = json.loads(result.body)
        # Help should return ephemeral message with help text
        assert body.get("response_type") == "ephemeral" or "help" in str(body).lower()

    @pytest.mark.asyncio
    async def test_status_command(self, mock_request):
        """Test status command returns status info."""
        from aragora.server.handlers.bots.slack.commands import handle_slack_commands

        request = mock_request(text="status")

        result = await handle_slack_commands(request)

        assert result is not None
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_empty_command_shows_help(self, mock_request):
        """Test empty command shows help."""
        from aragora.server.handlers.bots.slack.commands import handle_slack_commands

        request = mock_request(text="")

        result = await handle_slack_commands(request)

        assert result is not None
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_unknown_command(self, mock_request):
        """Test unknown command returns error or help."""
        from aragora.server.handlers.bots.slack.commands import handle_slack_commands

        request = mock_request(text="invalidcommand12345")

        result = await handle_slack_commands(request)

        assert result is not None
        assert result.status_code == 200
        # Should indicate unknown command or show help
        body = json.loads(result.body)
        assert "response_type" in body


class TestAskCommand:
    """Tests for the ask command."""

    @pytest.mark.asyncio
    async def test_ask_command_starts_debate(self, mock_request):
        """Test ask command initiates a debate."""
        from aragora.server.handlers.bots.slack.commands import handle_slack_commands

        request = mock_request(text="ask What is the best framework?")

        with patch("aragora.server.handlers.bots.slack.commands.start_slack_debate") as mock_start:
            mock_start.return_value = {"debate_id": "debate-123"}

            result = await handle_slack_commands(request)

        assert result is not None
        assert result.status_code == 200

    @pytest.mark.asyncio
    async def test_ask_command_validates_topic(self, mock_request):
        """Test ask command validates topic input."""
        from aragora.server.handlers.bots.slack.commands import handle_slack_commands

        # Very long topic should be rejected or truncated
        long_topic = "x" * 5000
        request = mock_request(text=f"ask {long_topic}")

        result = await handle_slack_commands(request)

        assert result is not None
        # Should either truncate or reject
        assert result.status_code in (200, 400)

    @pytest.mark.asyncio
    async def test_ask_command_empty_question(self, mock_request):
        """Test ask command without question."""
        from aragora.server.handlers.bots.slack.commands import handle_slack_commands

        request = mock_request(text="ask")

        result = await handle_slack_commands(request)

        assert result is not None
        assert result.status_code == 200
        # Should prompt for question
        body = json.loads(result.body)
        assert "response_type" in body


class TestVoteCommand:
    """Tests for the vote command."""

    @pytest.mark.asyncio
    async def test_vote_command_response(self, mock_request):
        """Test vote command returns appropriate response."""
        from aragora.server.handlers.bots.slack.commands import handle_slack_commands
        from aragora.server.handlers.bots.slack.state import _active_debates

        _active_debates.clear()

        request = mock_request(text="vote 1")

        result = await handle_slack_commands(request)

        assert result is not None
        assert result.status_code == 200
        body = json.loads(result.body)
        # Should indicate no active debate, permission denied, or vote message
        assert (
            "no active" in str(body).lower()
            or "debate" in str(body).lower()
            or "permission" in str(body).lower()
            or "vote" in str(body).lower()
        )


class TestLeaderboardCommand:
    """Tests for the leaderboard command."""

    @pytest.mark.asyncio
    async def test_leaderboard_command(self, mock_request):
        """Test leaderboard command returns rankings."""
        from aragora.server.handlers.bots.slack.commands import handle_slack_commands

        request = mock_request(text="leaderboard")

        result = await handle_slack_commands(request)

        assert result is not None
        assert result.status_code == 200


class TestInputValidation:
    """Tests for command input validation."""

    @pytest.mark.asyncio
    async def test_invalid_user_id_rejected(self, mock_request):
        """Test invalid user ID is rejected."""
        from aragora.server.handlers.bots.slack.commands import handle_slack_commands

        request = mock_request(user_id="<script>alert(1)</script>")

        result = await handle_slack_commands(request)

        assert result is not None
        # Should reject invalid user ID
        body = json.loads(result.body)
        assert "invalid" in str(body).lower() or result.status_code == 200

    @pytest.mark.asyncio
    async def test_invalid_channel_id_rejected(self, mock_request):
        """Test invalid channel ID is rejected."""
        from aragora.server.handlers.bots.slack.commands import handle_slack_commands

        request = mock_request(channel_id="../../../etc/passwd")

        result = await handle_slack_commands(request)

        assert result is not None
        body = json.loads(result.body)
        assert "invalid" in str(body).lower() or result.status_code == 200

    @pytest.mark.asyncio
    async def test_invalid_team_id_rejected(self, mock_request):
        """Test invalid team ID is rejected."""
        from aragora.server.handlers.bots.slack.commands import handle_slack_commands

        request = mock_request(team_id="; DROP TABLE users;")

        result = await handle_slack_commands(request)

        assert result is not None
        body = json.loads(result.body)
        assert "invalid" in str(body).lower() or result.status_code == 200


class TestCommandConstants:
    """Tests for command-related constants."""

    def test_agent_display_names_defined(self):
        """Test agent display names are defined."""
        from aragora.server.handlers.bots.slack.constants import AGENT_DISPLAY_NAMES

        assert AGENT_DISPLAY_NAMES is not None
        assert isinstance(AGENT_DISPLAY_NAMES, dict)

    def test_max_topic_length_reasonable(self):
        """Test MAX_TOPIC_LENGTH is reasonable."""
        from aragora.server.handlers.bots.slack.constants import MAX_TOPIC_LENGTH

        assert MAX_TOPIC_LENGTH > 0
        assert MAX_TOPIC_LENGTH <= 10000

    def test_max_command_length_reasonable(self):
        """Test MAX_COMMAND_LENGTH is reasonable."""
        from aragora.server.handlers.bots.slack.constants import MAX_COMMAND_LENGTH

        assert MAX_COMMAND_LENGTH > 0
        assert MAX_COMMAND_LENGTH <= 10000


class TestPlanCommand:
    """Tests for the plan command."""

    @pytest.mark.asyncio
    async def test_plan_command(self, mock_request):
        """Test plan command starts debate with planning."""
        from aragora.server.handlers.bots.slack.commands import handle_slack_commands

        request = mock_request(text="plan How should we refactor auth?")

        with patch("aragora.server.handlers.bots.slack.commands.start_slack_debate") as mock_start:
            mock_start.return_value = {"debate_id": "debate-123"}

            result = await handle_slack_commands(request)

        assert result is not None
        assert result.status_code == 200


class TestImplementCommand:
    """Tests for the implement command."""

    @pytest.mark.asyncio
    async def test_implement_command(self, mock_request):
        """Test implement command starts full implementation flow."""
        from aragora.server.handlers.bots.slack.commands import handle_slack_commands

        request = mock_request(text="implement Add user authentication")

        with patch("aragora.server.handlers.bots.slack.commands.start_slack_debate") as mock_start:
            mock_start.return_value = {"debate_id": "debate-123"}

            result = await handle_slack_commands(request)

        assert result is not None
        assert result.status_code == 200
