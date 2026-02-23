"""Tests for aragora.server.handlers.social.chat_events - Chat Events Handler."""

import sys
import types as _types_mod

# Pre-stub Slack modules to prevent import chain failures
_SLACK_ATTRS = [
    "SlackHandler",
    "get_slack_handler",
    "get_slack_integration",
    "get_workspace_store",
    "resolve_workspace",
    "create_tracked_task",
    "_validate_slack_url",
    "SLACK_SIGNING_SECRET",
    "SLACK_BOT_TOKEN",
    "SLACK_WEBHOOK_URL",
    "SLACK_ALLOWED_DOMAINS",
    "SignatureVerifierMixin",
    "CommandsMixin",
    "EventsMixin",
    "init_slack_handler",
]
for _mod_name in (
    "aragora.server.handlers.social.slack.handler",
    "aragora.server.handlers.social.slack",
    "aragora.server.handlers.social._slack_impl",
):
    if _mod_name not in sys.modules:
        _m = _types_mod.ModuleType(_mod_name)
        for _a in _SLACK_ATTRS:
            setattr(_m, _a, None)
        sys.modules[_mod_name] = _m

import time
from unittest.mock import MagicMock, patch

import pytest

from aragora.server.handlers.social import chat_events


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def mock_dispatch_event():
    """Mock the dispatch_event function."""
    with patch("aragora.server.handlers.social.chat_events.dispatch_event") as mock:
        yield mock


@pytest.fixture(autouse=True)
def patch_dispatch():
    """Patch the dispatch function for all tests."""
    with patch("aragora.server.handlers.social.chat_events._dispatch_chat_event") as mock:
        mock.return_value = None
        yield mock


# ===========================================================================
# Message Received Tests
# ===========================================================================


class TestMessageReceived:
    """Tests for emit_message_received function."""

    def test_emit_message_received_basic(self, patch_dispatch):
        """Test basic message received event."""
        from aragora.server.handlers.social.chat_events import emit_message_received

        emit_message_received(
            platform="telegram",
            chat_id="123456",
            user_id="user-123",
            username="testuser",
            message_text="Hello, world!",
        )

        patch_dispatch.assert_called_once()
        args = patch_dispatch.call_args
        assert args[0][0] == "chat.message_received"
        assert args[0][1]["platform"] == "telegram"
        assert args[0][1]["chat_id"] == "123456"
        assert args[0][1]["user_id"] == "user-123"
        assert args[0][1]["username"] == "testuser"

    def test_emit_message_received_with_type(self, patch_dispatch):
        """Test message received with type."""
        from aragora.server.handlers.social.chat_events import emit_message_received

        emit_message_received(
            platform="whatsapp",
            chat_id="789",
            user_id="user-456",
            username="otheruser",
            message_text="Command text",
            message_type="command",
        )

        patch_dispatch.assert_called_once()
        args = patch_dispatch.call_args
        assert args[0][1]["message_type"] == "command"

    def test_emit_message_received_truncates_preview(self, patch_dispatch):
        """Test message preview is truncated."""
        from aragora.server.handlers.social.chat_events import emit_message_received

        long_message = "x" * 200
        emit_message_received(
            platform="telegram",
            chat_id="123",
            user_id="user-1",
            username="user",
            message_text=long_message,
        )

        args = patch_dispatch.call_args
        assert len(args[0][1]["message_preview"]) == 100


# ===========================================================================
# Command Received Tests
# ===========================================================================


class TestCommandReceived:
    """Tests for emit_command_received function."""

    def test_emit_command_received_basic(self, patch_dispatch):
        """Test basic command received event."""
        from aragora.server.handlers.social.chat_events import emit_command_received

        emit_command_received(
            platform="telegram",
            chat_id="123456",
            user_id="user-123",
            username="testuser",
            command="debate",
        )

        patch_dispatch.assert_called_once()
        args = patch_dispatch.call_args
        assert args[0][0] == "chat.command_received"
        assert args[0][1]["command"] == "debate"

    def test_emit_command_received_with_args(self, patch_dispatch):
        """Test command received with arguments."""
        from aragora.server.handlers.social.chat_events import emit_command_received

        emit_command_received(
            platform="telegram",
            chat_id="123",
            user_id="user-1",
            username="user",
            command="debate",
            args="should we use microservices?",
        )

        args = patch_dispatch.call_args
        assert "args_preview" in args[0][1]


# ===========================================================================
# Debate Started Tests
# ===========================================================================


class TestDebateStarted:
    """Tests for emit_debate_started function."""

    def test_emit_debate_started_basic(self, patch_dispatch):
        """Test basic debate started event."""
        from aragora.server.handlers.social.chat_events import emit_debate_started

        emit_debate_started(
            platform="telegram",
            chat_id="123456",
            user_id="user-123",
            username="testuser",
            topic="Should we adopt microservices?",
        )

        patch_dispatch.assert_called_once()
        args = patch_dispatch.call_args
        assert args[0][0] == "chat.debate_started"
        assert args[0][1]["topic"] == "Should we adopt microservices?"

    def test_emit_debate_started_with_id(self, patch_dispatch):
        """Test debate started with debate ID."""
        from aragora.server.handlers.social.chat_events import emit_debate_started

        emit_debate_started(
            platform="whatsapp",
            chat_id="789",
            user_id="user-456",
            username="otheruser",
            topic="Test topic",
            debate_id="debate-123",
        )

        args = patch_dispatch.call_args
        assert args[0][1]["debate_id"] == "debate-123"


# ===========================================================================
# Debate Completed Tests
# ===========================================================================


class TestDebateCompleted:
    """Tests for emit_debate_completed function."""

    def test_emit_debate_completed_consensus(self, patch_dispatch):
        """Test debate completed with consensus."""
        from aragora.server.handlers.social.chat_events import emit_debate_completed

        emit_debate_completed(
            platform="telegram",
            chat_id="123456",
            debate_id="debate-123",
            topic="Test topic",
            consensus_reached=True,
            confidence=0.95,
            rounds_used=3,
        )

        patch_dispatch.assert_called_once()
        args = patch_dispatch.call_args
        assert args[0][0] == "chat.debate_completed"
        assert args[0][1]["consensus_reached"] is True
        assert args[0][1]["confidence"] == 0.95

    def test_emit_debate_completed_with_answer(self, patch_dispatch):
        """Test debate completed with final answer."""
        from aragora.server.handlers.social.chat_events import emit_debate_completed

        emit_debate_completed(
            platform="telegram",
            chat_id="123",
            debate_id="debate-456",
            topic="Topic",
            consensus_reached=False,
            confidence=0.6,
            rounds_used=5,
            final_answer="No consensus was reached, but here are the key points...",
        )

        args = patch_dispatch.call_args
        assert "final_answer_preview" in args[0][1]


# ===========================================================================
# Gauntlet Started Tests
# ===========================================================================


class TestGauntletStarted:
    """Tests for emit_gauntlet_started function."""

    def test_emit_gauntlet_started_basic(self, patch_dispatch):
        """Test basic gauntlet started event."""
        from aragora.server.handlers.social.chat_events import emit_gauntlet_started

        emit_gauntlet_started(
            platform="telegram",
            chat_id="123456",
            user_id="user-123",
            username="testuser",
            statement="The earth is round",
        )

        patch_dispatch.assert_called_once()
        args = patch_dispatch.call_args
        assert args[0][0] == "chat.gauntlet_started"
        assert args[0][1]["statement"] == "The earth is round"


# ===========================================================================
# Gauntlet Completed Tests
# ===========================================================================


class TestGauntletCompleted:
    """Tests for emit_gauntlet_completed function."""

    def test_emit_gauntlet_completed_passed(self, patch_dispatch):
        """Test gauntlet completed with passed result."""
        from aragora.server.handlers.social.chat_events import emit_gauntlet_completed

        emit_gauntlet_completed(
            platform="telegram",
            chat_id="123456",
            gauntlet_id="gauntlet-123",
            statement="Test statement",
            verdict="PASSED",
            confidence=0.9,
            challenges_passed=5,
            challenges_total=5,
        )

        patch_dispatch.assert_called_once()
        args = patch_dispatch.call_args
        assert args[0][0] == "chat.gauntlet_completed"
        assert args[0][1]["verdict"] == "PASSED"
        assert args[0][1]["challenges_passed"] == 5


# ===========================================================================
# Vote Received Tests
# ===========================================================================


class TestVoteReceived:
    """Tests for emit_vote_received function."""

    def test_emit_vote_received_agree(self, patch_dispatch):
        """Test vote received with agree."""
        from aragora.server.handlers.social.chat_events import emit_vote_received

        emit_vote_received(
            platform="telegram",
            chat_id="123456",
            user_id="user-123",
            username="testuser",
            debate_id="debate-123",
            vote="agree",
        )

        patch_dispatch.assert_called_once()
        args = patch_dispatch.call_args
        assert args[0][0] == "chat.vote_received"
        assert args[0][1]["vote"] == "agree"

    def test_emit_vote_received_disagree(self, patch_dispatch):
        """Test vote received with disagree."""
        from aragora.server.handlers.social.chat_events import emit_vote_received

        emit_vote_received(
            platform="whatsapp",
            chat_id="789",
            user_id="user-456",
            username="otheruser",
            debate_id="debate-456",
            vote="disagree",
        )

        args = patch_dispatch.call_args
        assert args[0][1]["vote"] == "disagree"


# ===========================================================================
# Module Exports Tests
# ===========================================================================


class TestModuleExports:
    """Tests for module exports."""

    def test_all_functions_exported(self):
        """Test all expected functions are exported."""

        expected = [
            "emit_message_received",
            "emit_command_received",
            "emit_debate_started",
            "emit_debate_completed",
            "emit_gauntlet_started",
            "emit_gauntlet_completed",
            "emit_vote_received",
        ]
        for func_name in expected:
            assert hasattr(chat_events, func_name)
            assert callable(getattr(chat_events, func_name))
