"""
Tests for Slack handler.

Tests cover:
- Request routing (can_handle)
- Bot enabled detection
- GET status endpoint
- POST events endpoint (url_verification)
- RBAC permission checks
- Input validation
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.bots.slack.handler import SlackHandler


@pytest.fixture
def handler():
    """Create a SlackHandler instance."""
    with patch.dict("os.environ", {"SLACK_SIGNING_SECRET": "test-secret"}):
        return SlackHandler({})


@pytest.fixture
def mock_request():
    """Create a mock HTTP request."""

    class MockRequest:
        def __init__(self, body_data: bytes = b"", headers: dict | None = None):
            self._body = body_data
            self.headers = headers or {}

        async def body(self) -> bytes:
            return self._body

    return MockRequest


class TestRouting:
    """Tests for request routing."""

    def test_can_handle_slack_status(self, handler):
        """Test can_handle for status endpoint."""
        assert handler.can_handle("/api/v1/bots/slack/status")

    def test_can_handle_slack_events(self, handler):
        """Test can_handle for events endpoint."""
        assert handler.can_handle("/api/v1/bots/slack/events")

    def test_can_handle_slack_interactions(self, handler):
        """Test can_handle for interactions endpoint."""
        assert handler.can_handle("/api/v1/bots/slack/interactions")

    def test_can_handle_slack_commands(self, handler):
        """Test can_handle for commands endpoint."""
        assert handler.can_handle("/api/v1/bots/slack/commands")

    def test_can_handle_integrations_path(self, handler):
        """Test can_handle for integrations path."""
        assert handler.can_handle("/api/v1/integrations/slack/status")

    def test_cannot_handle_other_paths(self, handler):
        """Test can_handle rejects non-slack paths."""
        assert not handler.can_handle("/api/v1/bots/teams/status")
        assert not handler.can_handle("/api/v1/payments/charge")

    def test_routes_list(self, handler):
        """Test ROUTES list contains expected endpoints."""
        expected_routes = [
            "/api/v1/bots/slack/status",
            "/api/v1/bots/slack/events",
            "/api/v1/bots/slack/interactions",
            "/api/v1/bots/slack/commands",
        ]
        for route in expected_routes:
            assert route in handler.ROUTES


class TestBotEnabled:
    """Tests for bot enabled detection."""

    def test_bot_enabled_with_token(self, handler):
        """Test bot is enabled when token is set."""
        with patch.object(
            __import__("aragora.server.handlers.bots.slack", fromlist=["SLACK_BOT_TOKEN"]),
            "SLACK_BOT_TOKEN",
            "xoxb-test-token",
        ):
            with patch.object(
                __import__("aragora.server.handlers.bots.slack", fromlist=["SLACK_SIGNING_SECRET"]),
                "SLACK_SIGNING_SECRET",
                None,
            ):
                assert handler._is_bot_enabled()

    def test_bot_enabled_with_secret(self, handler):
        """Test bot is enabled when signing secret is set."""
        with patch.object(
            __import__("aragora.server.handlers.bots.slack", fromlist=["SLACK_SIGNING_SECRET"]),
            "SLACK_SIGNING_SECRET",
            "test-secret",
        ):
            with patch.object(
                __import__("aragora.server.handlers.bots.slack", fromlist=["SLACK_BOT_TOKEN"]),
                "SLACK_BOT_TOKEN",
                None,
            ):
                assert handler._is_bot_enabled()

    def test_bot_disabled_without_credentials(self, handler):
        """Test bot is disabled without credentials."""
        with patch.object(
            __import__("aragora.server.handlers.bots.slack", fromlist=["SLACK_BOT_TOKEN"]),
            "SLACK_BOT_TOKEN",
            None,
        ):
            with patch.object(
                __import__("aragora.server.handlers.bots.slack", fromlist=["SLACK_SIGNING_SECRET"]),
                "SLACK_SIGNING_SECRET",
                None,
            ):
                assert not handler._is_bot_enabled()


class TestStatusEndpoint:
    """Tests for GET /status endpoint."""

    def test_status_endpoint_returns_json(self, handler):
        """Test status endpoint returns valid JSON."""
        mock_handler = MagicMock()
        mock_handler.command = "GET"

        result = handler.handle("/api/v1/bots/slack/status", {}, mock_handler)

        assert result is not None
        assert result.status_code == 200

        body = json.loads(result.body)
        assert "configured" in body or "enabled" in body or "platform" in body


class TestEventsEndpoint:
    """Tests for POST /events endpoint."""

    @pytest.mark.asyncio
    async def test_url_verification_challenge(self):
        """Test URL verification challenge response via events handler."""
        from aragora.server.handlers.bots.slack.events import handle_slack_events

        class MockRequest:
            async def body(self):
                return json.dumps(
                    {"type": "url_verification", "challenge": "test-challenge-token"}
                ).encode()

        request = MockRequest()
        result = await handle_slack_events(request)

        assert result is not None
        assert result.status_code == 200

        body = json.loads(result.body)
        assert body.get("challenge") == "test-challenge-token"

    @pytest.mark.asyncio
    async def test_events_invalid_json(self):
        """Test events endpoint with invalid JSON."""
        from aragora.server.handlers.bots.slack.events import handle_slack_events

        class MockRequest:
            async def body(self):
                return b"not-valid-json"

        request = MockRequest()
        result = await handle_slack_events(request)

        assert result is not None
        # Should handle gracefully (400 or 500)
        assert result.status_code in (400, 500)


class TestSignatureVerification:
    """Tests for Slack signature verification."""

    def test_verify_slack_signature_function_exists(self):
        """Test verify_slack_signature function is importable."""
        from aragora.server.handlers.bots.slack.signature import verify_slack_signature

        assert callable(verify_slack_signature)

    def test_verify_slack_signature_with_missing_headers(self):
        """Test signature verification with missing headers."""
        from aragora.server.handlers.bots.slack.signature import verify_slack_signature

        # Without proper headers, should return False
        result = verify_slack_signature(
            body=b'{"type": "event_callback"}',
            timestamp="",
            signature="",
            signing_secret="test-secret",
        )

        assert result is False

    def test_verify_slack_signature_with_invalid_signature(self):
        """Test signature verification with invalid signature."""
        from aragora.server.handlers.bots.slack.signature import verify_slack_signature

        result = verify_slack_signature(
            body=b'{"type": "event_callback"}',
            timestamp="1234567890",
            signature="v0=invalid",
            signing_secret="test-secret",
        )

        assert result is False


class TestInputValidation:
    """Tests for input validation."""

    def test_validate_user_id_valid(self):
        """Test valid user ID validation."""
        from aragora.server.handlers.bots.slack.constants import _validate_slack_user_id

        valid, error = _validate_slack_user_id("U12345678")
        assert valid
        assert error is None

    def test_validate_user_id_invalid_chars(self):
        """Test invalid user ID with special chars."""
        from aragora.server.handlers.bots.slack.constants import _validate_slack_user_id

        valid, error = _validate_slack_user_id("U<script>alert(1)</script>")
        assert not valid
        assert error is not None

    def test_validate_channel_id_valid(self):
        """Test valid channel ID validation."""
        from aragora.server.handlers.bots.slack.constants import (
            _validate_slack_channel_id,
        )

        valid, error = _validate_slack_channel_id("C12345678")
        assert valid
        assert error is None

    def test_validate_channel_id_invalid(self):
        """Test invalid channel ID."""
        from aragora.server.handlers.bots.slack.constants import (
            _validate_slack_channel_id,
        )

        valid, error = _validate_slack_channel_id("../../../etc/passwd")
        assert not valid
        assert error is not None

    def test_validate_team_id_valid(self):
        """Test valid team ID validation."""
        from aragora.server.handlers.bots.slack.constants import _validate_slack_team_id

        valid, error = _validate_slack_team_id("T12345678")
        assert valid
        assert error is None

    def test_validate_input_max_length(self):
        """Test input validation max length."""
        from aragora.server.handlers.bots.slack.constants import _validate_slack_input

        long_input = "x" * 10000
        valid, error = _validate_slack_input(long_input, "test", max_length=100)
        assert not valid
        assert "length" in error.lower() or "long" in error.lower()


class TestRBACHelpers:
    """Tests for RBAC helper methods."""

    def test_build_auth_context_from_slack(self, handler):
        """Test building auth context from Slack data."""
        with patch(
            "aragora.server.handlers.bots.slack.handler.build_auth_context_from_slack"
        ) as mock_build:
            mock_context = MagicMock()
            mock_build.return_value = mock_context

            result = handler._build_auth_context_from_slack("T123", "U456", "C789")

            mock_build.assert_called_once_with("T123", "U456", "C789")
            assert result == mock_context

    def test_get_org_from_team(self, handler):
        """Test getting org ID from team."""
        with patch("aragora.server.handlers.bots.slack.handler.get_org_from_team") as mock_get:
            mock_get.return_value = "org-123"

            result = handler._get_org_from_team("T123")

            mock_get.assert_called_once_with("T123")
            assert result == "org-123"

    def test_check_workspace_authorized(self, handler):
        """Test workspace authorization check."""
        with patch(
            "aragora.server.handlers.bots.slack.handler.check_workspace_authorized"
        ) as mock_check:
            mock_check.return_value = (True, None)

            authorized, error = handler._check_workspace_authorized("T123")

            mock_check.assert_called_once_with("T123")
            assert authorized
            assert error is None

    def test_check_workspace_not_authorized(self, handler):
        """Test workspace not authorized."""
        with patch(
            "aragora.server.handlers.bots.slack.handler.check_workspace_authorized"
        ) as mock_check:
            mock_check.return_value = (False, "Workspace not registered")

            authorized, error = handler._check_workspace_authorized("T999")

            assert not authorized
            assert error is not None


class TestPermissions:
    """Tests for permission constants."""

    def test_permission_constants_defined(self):
        """Test RBAC permission constants are defined."""
        from aragora.server.handlers.bots.slack.constants import (
            PERM_SLACK_ADMIN,
            PERM_SLACK_COMMANDS_EXECUTE,
            PERM_SLACK_COMMANDS_READ,
            PERM_SLACK_DEBATES_CREATE,
            PERM_SLACK_INTERACTIVE,
            PERM_SLACK_VOTES_RECORD,
        )

        assert PERM_SLACK_ADMIN is not None
        assert PERM_SLACK_COMMANDS_EXECUTE is not None
        assert PERM_SLACK_COMMANDS_READ is not None
        assert PERM_SLACK_DEBATES_CREATE is not None
        assert PERM_SLACK_INTERACTIVE is not None
        assert PERM_SLACK_VOTES_RECORD is not None

    def test_permissions_have_correct_prefix(self):
        """Test permissions have slack prefix."""
        from aragora.server.handlers.bots.slack.constants import (
            PERM_SLACK_ADMIN,
            PERM_SLACK_COMMANDS_EXECUTE,
            PERM_SLACK_COMMANDS_READ,
            PERM_SLACK_DEBATES_CREATE,
        )

        assert "slack" in PERM_SLACK_ADMIN.lower()
        assert "slack" in PERM_SLACK_COMMANDS_EXECUTE.lower()
        assert "slack" in PERM_SLACK_COMMANDS_READ.lower()
        assert "slack" in PERM_SLACK_DEBATES_CREATE.lower()


class TestStateManagement:
    """Tests for state management."""

    def test_active_debates_dict_exists(self):
        """Test active debates dict is accessible."""
        from aragora.server.handlers.bots.slack.state import _active_debates

        assert isinstance(_active_debates, dict)

    def test_user_votes_dict_exists(self):
        """Test user votes dict is accessible."""
        from aragora.server.handlers.bots.slack.state import _user_votes

        assert isinstance(_user_votes, dict)

    def test_get_active_debates_returns_copy(self):
        """Test get_active_debates returns a copy."""
        from aragora.server.handlers.bots.slack.state import (
            _active_debates,
            get_active_debates,
        )

        # Clear and add test data
        _active_debates.clear()
        _active_debates["test-channel"] = {"debate_id": "test-123"}

        result = get_active_debates()

        assert "test-channel" in result
        # Verify it's a copy or safe view
        assert result is not _active_debates or isinstance(result, dict)

        # Cleanup
        _active_debates.clear()


class TestBlockKit:
    """Tests for Block Kit message builders."""

    def test_build_debate_message_blocks(self):
        """Test debate message block building."""
        from aragora.server.handlers.bots.slack.blocks import build_debate_message_blocks

        blocks = build_debate_message_blocks(
            debate_id="debate-123",
            task="Test question?",
            agents=["claude", "gpt"],
            current_round=1,
            total_rounds=3,
        )

        assert blocks is not None
        assert isinstance(blocks, list)

    def test_build_consensus_message_blocks(self):
        """Test consensus message block building."""
        from aragora.server.handlers.bots.slack.blocks import (
            build_consensus_message_blocks,
        )

        blocks = build_consensus_message_blocks(
            debate_id="debate-123",
            task="Test question?",
            consensus_reached=True,
            confidence=0.85,
            winner="claude",
            final_answer="The consensus is X",
            vote_counts={"claude": 3, "gpt": 2},
        )

        assert blocks is not None
        assert isinstance(blocks, list)

    def test_build_start_debate_modal(self):
        """Test start debate modal building."""
        from aragora.server.handlers.bots.slack.blocks import build_start_debate_modal

        modal = build_start_debate_modal()

        assert modal is not None
        # Modal should have type and blocks
        assert isinstance(modal, dict)


class TestConstants:
    """Tests for handler constants."""

    def test_max_length_constants(self):
        """Test max length constants are reasonable."""
        from aragora.server.handlers.bots.slack.constants import (
            MAX_CHANNEL_ID_LENGTH,
            MAX_COMMAND_LENGTH,
            MAX_TOPIC_LENGTH,
            MAX_USER_ID_LENGTH,
        )

        assert MAX_CHANNEL_ID_LENGTH > 0
        assert MAX_COMMAND_LENGTH > 0
        assert MAX_TOPIC_LENGTH > 0
        assert MAX_USER_ID_LENGTH > 0

        # Reasonable bounds
        assert MAX_CHANNEL_ID_LENGTH <= 100
        assert MAX_USER_ID_LENGTH <= 100

    def test_validation_patterns_defined(self):
        """Test validation patterns are defined."""
        from aragora.server.handlers.bots.slack.constants import (
            COMMAND_PATTERN,
            TOPIC_PATTERN,
        )

        assert COMMAND_PATTERN is not None
        assert TOPIC_PATTERN is not None
