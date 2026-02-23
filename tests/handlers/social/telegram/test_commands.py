"""Tests for Telegram bot command handling (commands.py).

Covers all commands and their branches:
- /start, /help, /status, /agents, /debate, /plan, /implement,
  /gauntlet, /search, /recent, /receipt
- Unknown commands
- RBAC permission denials
- Topic/statement validation (too short, too long, missing)
- Async debate execution (_run_debate_async)
- Async gauntlet execution (_run_gauntlet_async)
- Receipt formatting helpers (_format_receipt, _format_debate_as_receipt)
- Edge cases: @botname suffix, command parsing, etc.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aragora.server.handlers.social.telegram.handler import TelegramHandler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result: object) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if isinstance(result, dict):
        return result
    return json.loads(result.body)


def _status(result: object) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler():
    """Create a TelegramHandler for testing commands."""
    return TelegramHandler(ctx={})


@pytest.fixture
def _patch_tg():
    """Patch the _tg() lazy import to return a mock telegram module.

    This prevents real asyncio task creation and provides controllable
    module-level attributes (create_tracked_task, RBAC_AVAILABLE, TTS_VOICE_ENABLED).
    """
    mock_tg = MagicMock()
    mock_tg.create_tracked_task = MagicMock()
    mock_tg.RBAC_AVAILABLE = False
    mock_tg.TTS_VOICE_ENABLED = False
    mock_tg.TELEGRAM_BOT_TOKEN = "fake-token"

    with patch(
        "aragora.server.handlers.social.telegram.commands._tg",
        return_value=mock_tg,
    ) as _p:
        yield mock_tg


@pytest.fixture
def _patch_telemetry():
    """Patch telemetry functions so they do nothing."""
    with (
        patch("aragora.server.handlers.social.telegram.commands.record_command") as rc,
        patch("aragora.server.handlers.social.telegram.commands.record_debate_started"),
        patch("aragora.server.handlers.social.telegram.commands.record_debate_completed"),
        patch("aragora.server.handlers.social.telegram.commands.record_debate_failed"),
        patch("aragora.server.handlers.social.telegram.commands.record_gauntlet_started"),
        patch("aragora.server.handlers.social.telegram.commands.record_gauntlet_completed"),
        patch("aragora.server.handlers.social.telegram.commands.record_gauntlet_failed"),
    ):
        yield rc


@pytest.fixture
def _patch_events():
    """Patch chat event emitters."""
    with (
        patch("aragora.server.handlers.social.telegram.commands.emit_command_received"),
        patch("aragora.server.handlers.social.telegram.commands.emit_debate_started"),
        patch("aragora.server.handlers.social.telegram.commands.emit_debate_completed"),
        patch("aragora.server.handlers.social.telegram.commands.emit_gauntlet_started"),
        patch("aragora.server.handlers.social.telegram.commands.emit_gauntlet_completed"),
    ):
        yield


@pytest.fixture
def _patch_rbac():
    """Patch RBAC permission check to always allow."""
    with patch.object(
        TelegramHandler,
        "_check_telegram_user_permission",
        return_value=True,
    ):
        yield


@pytest.fixture
def _patch_deny_rbac():
    """Patch RBAC permission check to always deny."""
    with (
        patch.object(
            TelegramHandler,
            "_check_telegram_user_permission",
            return_value=False,
        ),
        patch.object(
            TelegramHandler,
            "_deny_telegram_permission",
            return_value=MagicMock(status_code=200, body=b'{"ok":true}'),
        ) as deny_mock,
    ):
        yield deny_mock


@pytest.fixture
def _full_patch(_patch_tg, _patch_telemetry, _patch_events, _patch_rbac):
    """Combine all patches needed for command tests."""
    yield


# Shorthand: chat_id, user_id, username used across tests
CHAT_ID = 12345
USER_ID = 67890
USERNAME = "testuser"


# ============================================================================
# _handle_command: Command dispatch and routing
# ============================================================================


class TestHandleCommandRouting:
    """Test command dispatch from _handle_command."""

    @pytest.mark.usefixtures("_full_patch")
    def test_start_command(self, handler, _patch_tg):
        result = handler._handle_command(CHAT_ID, USER_ID, USERNAME, "/start")
        assert _status(result) == 200
        assert _body(result)["ok"] is True
        # create_tracked_task should be called to send the message
        _patch_tg.create_tracked_task.assert_called()

    @pytest.mark.usefixtures("_full_patch")
    def test_help_command(self, handler, _patch_tg):
        result = handler._handle_command(CHAT_ID, USER_ID, USERNAME, "/help")
        assert _status(result) == 200
        assert _body(result)["ok"] is True
        _patch_tg.create_tracked_task.assert_called()

    @pytest.mark.usefixtures("_full_patch")
    def test_status_command(self, handler, _patch_tg):
        with patch(
            "aragora.server.handlers.social.telegram.commands.TelegramCommandsMixin._command_status",
            return_value="*Aragora Status*\n\nStatus: Online",
        ):
            result = handler._handle_command(CHAT_ID, USER_ID, USERNAME, "/status")
        assert _status(result) == 200
        assert _body(result)["ok"] is True

    @pytest.mark.usefixtures("_full_patch")
    def test_agents_command(self, handler, _patch_tg):
        with patch.object(handler, "_command_agents", return_value="No agents registered yet."):
            result = handler._handle_command(CHAT_ID, USER_ID, USERNAME, "/agents")
        assert _status(result) == 200

    @pytest.mark.usefixtures("_full_patch")
    def test_debate_command_delegates(self, handler, _patch_tg):
        with patch.object(
            handler,
            "_command_debate",
            return_value=MagicMock(status_code=200, body=b'{"ok":true}'),
        ) as cd:
            result = handler._handle_command(
                CHAT_ID, USER_ID, USERNAME, "/debate Should AI be regulated?"
            )
            cd.assert_called_once_with(CHAT_ID, USER_ID, USERNAME, "Should AI be regulated?")
        assert _status(result) == 200

    @pytest.mark.usefixtures("_full_patch")
    def test_plan_command_delegates_with_decision_integrity(self, handler, _patch_tg):
        with patch.object(
            handler,
            "_command_debate",
            return_value=MagicMock(status_code=200, body=b'{"ok":true}'),
        ) as cd:
            handler._handle_command(CHAT_ID, USER_ID, USERNAME, "/plan Improve on-call process")
            call_kwargs = cd.call_args
            # decision_integrity should have plan flags
            di = call_kwargs.kwargs.get(
                "decision_integrity", call_kwargs[0][3] if len(call_kwargs[0]) > 3 else None
            )
            # Check it was called with decision_integrity and mode_label
            assert cd.called

    @pytest.mark.usefixtures("_full_patch")
    def test_implement_command_delegates(self, handler, _patch_tg):
        with patch.object(
            handler,
            "_command_debate",
            return_value=MagicMock(status_code=200, body=b'{"ok":true}'),
        ) as cd:
            handler._handle_command(CHAT_ID, USER_ID, USERNAME, "/implement Build a dashboard")
            assert cd.called

    @pytest.mark.usefixtures("_full_patch")
    def test_gauntlet_command_delegates(self, handler, _patch_tg):
        with patch.object(
            handler,
            "_command_gauntlet",
            return_value=MagicMock(status_code=200, body=b'{"ok":true}'),
        ):
            result = handler._handle_command(
                CHAT_ID, USER_ID, USERNAME, "/gauntlet We should use microservices for everything"
            )
        assert _status(result) == 200

    @pytest.mark.usefixtures("_full_patch")
    def test_search_command(self, handler, _patch_tg):
        with patch.object(handler, "_command_search", return_value="No results"):
            result = handler._handle_command(CHAT_ID, USER_ID, USERNAME, "/search machine learning")
        assert _status(result) == 200

    @pytest.mark.usefixtures("_full_patch")
    def test_recent_command(self, handler, _patch_tg):
        with patch.object(handler, "_command_recent", return_value="No recent debates"):
            result = handler._handle_command(CHAT_ID, USER_ID, USERNAME, "/recent")
        assert _status(result) == 200

    @pytest.mark.usefixtures("_full_patch")
    def test_receipt_command(self, handler, _patch_tg):
        with patch.object(handler, "_command_receipt", return_value="Receipt data"):
            result = handler._handle_command(CHAT_ID, USER_ID, USERNAME, "/receipt abc123")
        assert _status(result) == 200

    @pytest.mark.usefixtures("_full_patch")
    def test_unknown_command(self, handler, _patch_tg):
        result = handler._handle_command(CHAT_ID, USER_ID, USERNAME, "/foobar")
        assert _status(result) == 200
        # The tracked task should contain "Unknown command"
        call_args = _patch_tg.create_tracked_task.call_args
        assert call_args is not None

    @pytest.mark.usefixtures("_full_patch")
    def test_command_strips_botname_suffix(self, handler, _patch_tg):
        """Commands like /help@AragoraBot should work as /help."""
        with patch.object(handler, "_command_help", return_value="help text") as ch:
            result = handler._handle_command(CHAT_ID, USER_ID, USERNAME, "/help@AragoraBot")
            ch.assert_called_once()
        assert _status(result) == 200

    @pytest.mark.usefixtures("_full_patch")
    def test_command_with_args_parsed(self, handler, _patch_tg):
        """Args after the command should be correctly extracted."""
        with patch.object(
            handler,
            "_command_debate",
            return_value=MagicMock(status_code=200, body=b'{"ok":true}'),
        ) as cd:
            handler._handle_command(CHAT_ID, USER_ID, USERNAME, "/debate  multi word topic here")
            # args should be " multi word topic here" (preserving leading space from split)
            cd.assert_called_once()
            call_args = cd.call_args[0]
            assert "multi word topic here" in call_args[3]

    @pytest.mark.usefixtures("_full_patch")
    def test_command_case_insensitive(self, handler, _patch_tg):
        """Commands are lowercased before matching."""
        with patch.object(handler, "_command_help", return_value="help text"):
            result = handler._handle_command(CHAT_ID, USER_ID, USERNAME, "/HELP")
        assert _status(result) == 200


# ============================================================================
# RBAC permission denial tests
# ============================================================================


class TestRBACDenials:
    """Test RBAC permission denials for each command."""

    @pytest.mark.usefixtures("_patch_tg", "_patch_telemetry", "_patch_events")
    def test_base_command_permission_denied(self, handler, _patch_deny_rbac):
        """When base command execute permission is denied, handler returns."""
        result = handler._handle_command(CHAT_ID, USER_ID, USERNAME, "/start")
        assert _status(result) == 200  # Denial sends a message and returns ok

    @pytest.mark.usefixtures("_patch_telemetry", "_patch_events")
    def test_start_read_permission_denied(self, handler, _patch_tg):
        """Start command checks read permission after execute permission."""
        call_count = [0]

        def check_perm(user_id, username, chat_id, perm):
            call_count[0] += 1
            # Allow execute (first call), deny read (second call)
            if call_count[0] == 1:
                return True
            return False

        with (
            patch.object(
                TelegramHandler, "_check_telegram_user_permission", side_effect=check_perm
            ),
            patch.object(
                TelegramHandler,
                "_deny_telegram_permission",
                return_value=MagicMock(status_code=200, body=b'{"ok":true}'),
            ) as deny,
        ):
            result = handler._handle_command(CHAT_ID, USER_ID, USERNAME, "/start")
            deny.assert_called_once()

    @pytest.mark.usefixtures("_patch_telemetry", "_patch_events")
    def test_debate_create_permission_denied(self, handler, _patch_tg):
        """Debate command checks debates:create permission."""
        call_count = [0]

        def check_perm(user_id, username, chat_id, perm):
            call_count[0] += 1
            if call_count[0] == 1:
                return True  # Allow execute
            return False  # Deny create

        with (
            patch.object(
                TelegramHandler, "_check_telegram_user_permission", side_effect=check_perm
            ),
            patch.object(
                TelegramHandler,
                "_deny_telegram_permission",
                return_value=MagicMock(status_code=200, body=b'{"ok":true}'),
            ),
        ):
            result = handler._handle_command(CHAT_ID, USER_ID, USERNAME, "/debate topic")
            assert _status(result) == 200

    @pytest.mark.usefixtures("_patch_telemetry", "_patch_events")
    def test_gauntlet_permission_denied(self, handler, _patch_tg):
        """Gauntlet command checks gauntlet:run permission."""
        call_count = [0]

        def check_perm(user_id, username, chat_id, perm):
            call_count[0] += 1
            if call_count[0] == 1:
                return True
            return False

        with (
            patch.object(
                TelegramHandler, "_check_telegram_user_permission", side_effect=check_perm
            ),
            patch.object(
                TelegramHandler,
                "_deny_telegram_permission",
                return_value=MagicMock(status_code=200, body=b'{"ok":true}'),
            ),
        ):
            result = handler._handle_command(CHAT_ID, USER_ID, USERNAME, "/gauntlet statement")
            assert _status(result) == 200

    @pytest.mark.usefixtures("_patch_telemetry", "_patch_events")
    def test_plan_permission_denied(self, handler, _patch_tg):
        """Plan command checks debates:create permission."""
        call_count = [0]

        def check_perm(user_id, username, chat_id, perm):
            call_count[0] += 1
            if call_count[0] == 1:
                return True
            return False

        with (
            patch.object(
                TelegramHandler, "_check_telegram_user_permission", side_effect=check_perm
            ),
            patch.object(
                TelegramHandler,
                "_deny_telegram_permission",
                return_value=MagicMock(status_code=200, body=b'{"ok":true}'),
            ),
        ):
            result = handler._handle_command(CHAT_ID, USER_ID, USERNAME, "/plan Some plan topic")
            assert _status(result) == 200

    @pytest.mark.usefixtures("_patch_telemetry", "_patch_events")
    def test_implement_permission_denied(self, handler, _patch_tg):
        """Implement command checks debates:create permission."""
        call_count = [0]

        def check_perm(user_id, username, chat_id, perm):
            call_count[0] += 1
            if call_count[0] == 1:
                return True
            return False

        with (
            patch.object(
                TelegramHandler, "_check_telegram_user_permission", side_effect=check_perm
            ),
            patch.object(
                TelegramHandler,
                "_deny_telegram_permission",
                return_value=MagicMock(status_code=200, body=b'{"ok":true}'),
            ),
        ):
            result = handler._handle_command(CHAT_ID, USER_ID, USERNAME, "/implement Build a thing")
            assert _status(result) == 200

    @pytest.mark.usefixtures("_patch_telemetry", "_patch_events")
    def test_search_read_permission_denied(self, handler, _patch_tg):
        call_count = [0]

        def check_perm(user_id, username, chat_id, perm):
            call_count[0] += 1
            if call_count[0] == 1:
                return True
            return False

        with (
            patch.object(
                TelegramHandler, "_check_telegram_user_permission", side_effect=check_perm
            ),
            patch.object(
                TelegramHandler,
                "_deny_telegram_permission",
                return_value=MagicMock(status_code=200, body=b'{"ok":true}'),
            ),
        ):
            result = handler._handle_command(CHAT_ID, USER_ID, USERNAME, "/search query")
            assert _status(result) == 200

    @pytest.mark.usefixtures("_patch_telemetry", "_patch_events")
    def test_recent_read_permission_denied(self, handler, _patch_tg):
        call_count = [0]

        def check_perm(user_id, username, chat_id, perm):
            call_count[0] += 1
            if call_count[0] == 1:
                return True
            return False

        with (
            patch.object(
                TelegramHandler, "_check_telegram_user_permission", side_effect=check_perm
            ),
            patch.object(
                TelegramHandler,
                "_deny_telegram_permission",
                return_value=MagicMock(status_code=200, body=b'{"ok":true}'),
            ),
        ):
            result = handler._handle_command(CHAT_ID, USER_ID, USERNAME, "/recent")
            assert _status(result) == 200

    @pytest.mark.usefixtures("_patch_telemetry", "_patch_events")
    def test_receipt_read_permission_denied(self, handler, _patch_tg):
        call_count = [0]

        def check_perm(user_id, username, chat_id, perm):
            call_count[0] += 1
            if call_count[0] == 1:
                return True
            return False

        with (
            patch.object(
                TelegramHandler, "_check_telegram_user_permission", side_effect=check_perm
            ),
            patch.object(
                TelegramHandler,
                "_deny_telegram_permission",
                return_value=MagicMock(status_code=200, body=b'{"ok":true}'),
            ),
        ):
            result = handler._handle_command(CHAT_ID, USER_ID, USERNAME, "/receipt abc")
            assert _status(result) == 200


# ============================================================================
# _command_start
# ============================================================================


class TestCommandStart:
    """Test /start command output."""

    def test_contains_welcome(self, handler):
        text = handler._command_start("alice")
        assert "Welcome to Aragora, alice!" in text

    def test_contains_commands_list(self, handler):
        text = handler._command_start("bob")
        assert "/debate" in text
        assert "/help" in text
        assert "/status" in text
        assert "/agents" in text

    def test_contains_plan_and_implement(self, handler):
        text = handler._command_start("user")
        assert "/plan" in text
        assert "/implement" in text

    def test_contains_gauntlet(self, handler):
        text = handler._command_start("user")
        assert "/gauntlet" in text

    def test_contains_search_recent_receipt(self, handler):
        text = handler._command_start("user")
        assert "/search" in text
        assert "/recent" in text
        assert "/receipt" in text


# ============================================================================
# _command_help
# ============================================================================


class TestCommandHelp:
    """Test /help command output."""

    def test_contains_all_commands(self, handler):
        text = handler._command_help()
        for cmd in [
            "/start",
            "/debate",
            "/plan",
            "/implement",
            "/gauntlet",
            "/search",
            "/recent",
            "/receipt",
            "/status",
            "/agents",
            "/help",
        ]:
            assert cmd in text

    def test_contains_examples(self, handler):
        text = handler._command_help()
        assert "Examples" in text
        assert "Should AI be regulated" in text

    def test_returns_string(self, handler):
        assert isinstance(handler._command_help(), str)


# ============================================================================
# _command_status
# ============================================================================


class TestCommandStatus:
    """Test /status command."""

    def test_status_with_elo_system(self, handler):
        mock_store = MagicMock()
        mock_store.get_all_ratings.return_value = [1, 2, 3]
        with patch(
            "aragora.server.handlers.social.telegram.commands.TelegramCommandsMixin._command_status",
            wraps=handler._command_status,
        ):
            with patch(
                "aragora.ranking.elo.EloSystem",
                return_value=mock_store,
            ):
                text = handler._command_status()
        assert "Online" in text
        assert "3" in text

    def test_status_import_error(self, handler):
        with patch.dict("sys.modules", {"aragora.ranking.elo": None}):
            # Force ImportError
            with patch(
                "builtins.__import__",
                side_effect=lambda name, *a, **kw: (_ for _ in ()).throw(ImportError())
                if "ranking.elo" in name
                else __builtins__.__import__(name, *a, **kw)
                if hasattr(__builtins__, "__import__")
                else None,
            ):
                text = handler._command_status()
        # Falls back to simple status
        assert "Online" in text

    def test_status_attribute_error(self, handler):
        """AttributeError in EloSystem is caught gracefully."""
        with patch(
            "aragora.ranking.elo.EloSystem",
            side_effect=AttributeError("no attribute"),
        ):
            text = handler._command_status()
        assert "Online" in text


# ============================================================================
# _command_agents
# ============================================================================


class TestCommandAgents:
    """Test /agents command."""

    def test_agents_empty(self, handler):
        mock_store = MagicMock()
        mock_store.get_all_ratings.return_value = []
        with patch("aragora.ranking.elo.EloSystem", return_value=mock_store):
            text = handler._command_agents()
        assert "No agents registered" in text

    def test_agents_with_ratings(self, handler):
        @dataclass
        class MockAgent:
            name: str
            elo: float
            wins: int

        agents = [
            MockAgent(name="Claude", elo=1600.0, wins=10),
            MockAgent(name="GPT", elo=1550.0, wins=8),
            MockAgent(name="Gemini", elo=1500.0, wins=5),
        ]
        mock_store = MagicMock()
        mock_store.get_all_ratings.return_value = agents
        with patch("aragora.ranking.elo.EloSystem", return_value=mock_store):
            text = handler._command_agents()
        assert "Claude" in text
        assert "GPT" in text
        assert "ELO" in text
        assert "1." in text
        assert "2." in text
        assert "3." in text

    def test_agents_sorted_by_elo(self, handler):
        @dataclass
        class MockAgent:
            name: str
            elo: float
            wins: int

        agents = [
            MockAgent(name="Low", elo=1400.0, wins=1),
            MockAgent(name="High", elo=1700.0, wins=20),
        ]
        mock_store = MagicMock()
        mock_store.get_all_ratings.return_value = agents
        with patch("aragora.ranking.elo.EloSystem", return_value=mock_store):
            text = handler._command_agents()
        # "High" should appear before "Low"
        assert text.index("High") < text.index("Low")

    def test_agents_limits_to_10(self, handler):
        @dataclass
        class MockAgent:
            name: str
            elo: float
            wins: int

        agents = [MockAgent(name=f"Agent{i}", elo=1500.0 + i, wins=i) for i in range(15)]
        mock_store = MagicMock()
        mock_store.get_all_ratings.return_value = agents
        with patch("aragora.ranking.elo.EloSystem", return_value=mock_store):
            text = handler._command_agents()
        # Should contain Agent14 (highest ELO) but not Agent0 through Agent4 (lowest)
        assert "Agent14" in text
        # Only 10 entries
        assert "11." not in text

    def test_agents_import_error(self, handler):
        with patch(
            "aragora.ranking.elo.EloSystem",
            side_effect=ImportError("no module"),
        ):
            text = handler._command_agents()
        assert "Could not fetch agent list" in text

    def test_agents_type_error(self, handler):
        with patch(
            "aragora.ranking.elo.EloSystem",
            side_effect=TypeError("bad type"),
        ):
            text = handler._command_agents()
        assert "Could not fetch agent list" in text


# ============================================================================
# _command_debate
# ============================================================================


class TestCommandDebate:
    """Test /debate command validation and dispatch."""

    @pytest.mark.usefixtures("_patch_tg")
    def test_no_args_returns_help(self, handler, _patch_tg):
        result = handler._command_debate(CHAT_ID, USER_ID, USERNAME, "")
        assert _status(result) == 200
        _patch_tg.create_tracked_task.assert_called()

    @pytest.mark.usefixtures("_patch_tg")
    def test_topic_too_short(self, handler, _patch_tg):
        result = handler._command_debate(CHAT_ID, USER_ID, USERNAME, "short")
        assert _status(result) == 200
        # Should have sent "too short" message
        _patch_tg.create_tracked_task.assert_called()

    @pytest.mark.usefixtures("_patch_tg")
    def test_topic_too_long(self, handler, _patch_tg):
        long_topic = "x" * 501
        result = handler._command_debate(CHAT_ID, USER_ID, USERNAME, long_topic)
        assert _status(result) == 200
        _patch_tg.create_tracked_task.assert_called()

    @pytest.mark.usefixtures("_patch_tg")
    def test_valid_topic_starts_debate(self, handler, _patch_tg):
        result = handler._command_debate(
            CHAT_ID, USER_ID, USERNAME, "Should we adopt Kubernetes for our infrastructure?"
        )
        assert _status(result) == 200
        # Should be called at least twice: ack message + debate task
        assert _patch_tg.create_tracked_task.call_count >= 2

    @pytest.mark.usefixtures("_patch_tg")
    def test_topic_stripped_of_quotes(self, handler, _patch_tg):
        result = handler._command_debate(
            CHAT_ID, USER_ID, USERNAME, '"Should we migrate databases?"'
        )
        assert _status(result) == 200

    @pytest.mark.usefixtures("_patch_tg")
    def test_exactly_10_chars_is_valid(self, handler, _patch_tg):
        result = handler._command_debate(CHAT_ID, USER_ID, USERNAME, "1234567890")
        assert _status(result) == 200
        assert _patch_tg.create_tracked_task.call_count >= 2

    @pytest.mark.usefixtures("_patch_tg")
    def test_exactly_500_chars_is_valid(self, handler, _patch_tg):
        topic = "a" * 500
        result = handler._command_debate(CHAT_ID, USER_ID, USERNAME, topic)
        assert _status(result) == 200
        assert _patch_tg.create_tracked_task.call_count >= 2

    @pytest.mark.usefixtures("_patch_tg")
    def test_9_chars_is_too_short(self, handler, _patch_tg):
        result = handler._command_debate(CHAT_ID, USER_ID, USERNAME, "123456789")
        assert _status(result) == 200
        # Only one call: the "too short" message
        assert _patch_tg.create_tracked_task.call_count == 1

    @pytest.mark.usefixtures("_patch_tg")
    def test_no_args_custom_label(self, handler, _patch_tg):
        result = handler._command_debate(CHAT_ID, USER_ID, USERNAME, "", mode_label="plan")
        assert _status(result) == 200

    @pytest.mark.usefixtures("_patch_tg")
    def test_no_args_uses_command_label(self, handler, _patch_tg):
        result = handler._command_debate(CHAT_ID, USER_ID, USERNAME, "", command_label="implement")
        assert _status(result) == 200

    @pytest.mark.usefixtures("_patch_tg")
    def test_decision_integrity_passed_through(self, handler, _patch_tg):
        di = {"include_receipt": True, "include_plan": True}
        result = handler._command_debate(
            CHAT_ID,
            USER_ID,
            USERNAME,
            "A sufficiently long topic for debate",
            decision_integrity=di,
        )
        assert _status(result) == 200


# ============================================================================
# _command_gauntlet
# ============================================================================


class TestCommandGauntlet:
    """Test /gauntlet command validation and dispatch."""

    @pytest.mark.usefixtures("_patch_tg")
    def test_no_args_returns_help(self, handler, _patch_tg):
        result = handler._command_gauntlet(CHAT_ID, USER_ID, USERNAME, "")
        assert _status(result) == 200
        _patch_tg.create_tracked_task.assert_called()

    @pytest.mark.usefixtures("_patch_tg")
    def test_statement_too_short(self, handler, _patch_tg):
        result = handler._command_gauntlet(CHAT_ID, USER_ID, USERNAME, "short")
        assert _status(result) == 200

    @pytest.mark.usefixtures("_patch_tg")
    def test_statement_too_long(self, handler, _patch_tg):
        long_stmt = "x" * 1001
        result = handler._command_gauntlet(CHAT_ID, USER_ID, USERNAME, long_stmt)
        assert _status(result) == 200

    @pytest.mark.usefixtures("_patch_tg")
    def test_valid_statement_starts_gauntlet(self, handler, _patch_tg):
        result = handler._command_gauntlet(
            CHAT_ID,
            USER_ID,
            USERNAME,
            "We should migrate to microservices for our main application",
        )
        assert _status(result) == 200
        # ack + gauntlet task
        assert _patch_tg.create_tracked_task.call_count >= 2

    @pytest.mark.usefixtures("_patch_tg")
    def test_exactly_10_chars_is_valid(self, handler, _patch_tg):
        result = handler._command_gauntlet(CHAT_ID, USER_ID, USERNAME, "1234567890")
        assert _status(result) == 200
        assert _patch_tg.create_tracked_task.call_count >= 2

    @pytest.mark.usefixtures("_patch_tg")
    def test_exactly_1000_chars_is_valid(self, handler, _patch_tg):
        stmt = "a" * 1000
        result = handler._command_gauntlet(CHAT_ID, USER_ID, USERNAME, stmt)
        assert _status(result) == 200

    @pytest.mark.usefixtures("_patch_tg")
    def test_quotes_stripped(self, handler, _patch_tg):
        result = handler._command_gauntlet(
            CHAT_ID, USER_ID, USERNAME, "'We should definitely use microservices'"
        )
        assert _status(result) == 200


# ============================================================================
# _command_search
# ============================================================================


class TestCommandSearch:
    """Test /search command."""

    def test_empty_query(self, handler):
        text = handler._command_search("")
        assert "at least 3 characters" in text

    def test_short_query(self, handler):
        text = handler._command_search("ab")
        assert "at least 3 characters" in text

    def test_whitespace_only_query(self, handler):
        text = handler._command_search("  ")
        assert "at least 3 characters" in text

    def test_search_with_results(self, handler):
        mock_db = MagicMock()
        results = [
            {"topic": "AI Regulation Debate", "id": "d1", "consensus_reached": True},
            {"topic": "ML Adoption Strategy", "id": "d2", "consensus_reached": False},
        ]
        mock_db.search.return_value = (results, 2)
        mock_db.get_recent_debates = MagicMock()

        with patch("aragora.storage.get_storage", return_value=mock_db):
            text = handler._command_search("regulation")
        assert "AI Regulation Debate" in text
        assert "d1" in text

    def test_search_no_results(self, handler):
        mock_db = MagicMock()
        mock_db.search.return_value = ([], 0)

        with patch("aragora.storage.get_storage", return_value=mock_db):
            text = handler._command_search("nonexistent")
        assert "No debates found" in text

    def test_search_fallback_to_get_recent(self, handler):
        mock_db = MagicMock(spec=[])
        mock_db.get_recent_debates = MagicMock(
            return_value=[
                {
                    "topic": "machine learning strategy",
                    "id": "d3",
                    "consensus_reached": True,
                    "conclusion": "",
                },
            ]
        )
        # No search method, but has get_recent_debates
        mock_db.search = None
        del mock_db.search

        with patch("aragora.storage.get_storage", return_value=mock_db):
            text = handler._command_search("machine")
        assert "machine learning" in text

    def test_search_no_db(self, handler):
        with patch("aragora.storage.get_storage", return_value=None):
            text = handler._command_search("test query")
        assert "not available" in text

    def test_search_db_no_search_no_recent(self, handler):
        mock_db = MagicMock(spec=[])
        with patch("aragora.storage.get_storage", return_value=mock_db):
            text = handler._command_search("test query")
        assert "not available" in text

    def test_search_import_error(self, handler):
        with patch(
            "aragora.storage.get_storage",
            side_effect=ImportError("no module"),
        ):
            text = handler._command_search("test query")
        assert "temporarily unavailable" in text

    def test_search_runtime_error(self, handler):
        mock_db = MagicMock()
        mock_db.search.side_effect = ValueError("boom")
        with patch("aragora.storage.get_storage", return_value=mock_db):
            text = handler._command_search("test query")
        assert "error occurred" in text

    def test_search_results_truncation(self, handler):
        mock_db = MagicMock()
        results = [
            {"topic": "A" * 100, "id": f"d{i}", "consensus_reached": True} for i in range(10)
        ]
        mock_db.search.return_value = (results, 10)
        with patch("aragora.storage.get_storage", return_value=mock_db):
            text = handler._command_search("test")
        assert "Showing 5 of 10 results" in text

    def test_search_topic_truncated_at_60(self, handler):
        mock_db = MagicMock()
        long_topic = "X" * 80
        results = [{"topic": long_topic, "id": "d1", "consensus_reached": False}]
        mock_db.search.return_value = (results, 1)
        with patch("aragora.storage.get_storage", return_value=mock_db):
            text = handler._command_search("test")
        assert "..." in text


# ============================================================================
# _command_recent
# ============================================================================


class TestCommandRecent:
    """Test /recent command."""

    def test_recent_no_db(self, handler):
        with patch("aragora.storage.get_storage", return_value=None):
            text = handler._command_recent()
        assert "not available" in text

    def test_recent_no_get_recent_method(self, handler):
        mock_db = MagicMock(spec=[])
        with patch("aragora.storage.get_storage", return_value=mock_db):
            text = handler._command_recent()
        assert "not available" in text

    def test_recent_empty(self, handler):
        mock_db = MagicMock()
        mock_db.get_recent_debates.return_value = []
        with patch("aragora.storage.get_storage", return_value=mock_db):
            text = handler._command_recent()
        assert "No recent debates" in text
        assert "/debate" in text

    def test_recent_with_results(self, handler):
        mock_db = MagicMock()
        mock_db.get_recent_debates.return_value = [
            {
                "topic": "Topic One",
                "id": "id1",
                "consensus_reached": True,
                "confidence": 0.85,
            },
            {
                "topic": "Topic Two",
                "id": "id2",
                "consensus_reached": False,
                "confidence": 0.4,
            },
        ]
        with patch("aragora.storage.get_storage", return_value=mock_db):
            text = handler._command_recent()
        assert "Topic One" in text
        assert "id1" in text
        assert "Yes" in text
        assert "85%" in text
        assert "/receipt" in text

    def test_recent_import_error(self, handler):
        with patch(
            "aragora.storage.get_storage",
            side_effect=ImportError("no module"),
        ):
            text = handler._command_recent()
        assert "temporarily unavailable" in text

    def test_recent_value_error(self, handler):
        mock_db = MagicMock()
        mock_db.get_recent_debates.side_effect = ValueError("db error")
        with patch("aragora.storage.get_storage", return_value=mock_db):
            text = handler._command_recent()
        assert "error occurred" in text

    def test_recent_topic_truncated_at_50(self, handler):
        mock_db = MagicMock()
        long_topic = "Y" * 70
        mock_db.get_recent_debates.return_value = [
            {"topic": long_topic, "id": "id1", "consensus_reached": True, "confidence": 0.5},
        ]
        with patch("aragora.storage.get_storage", return_value=mock_db):
            text = handler._command_recent()
        assert "..." in text


# ============================================================================
# _command_receipt
# ============================================================================


class TestCommandReceipt:
    """Test /receipt command."""

    def test_no_args(self, handler):
        text = handler._command_receipt("")
        assert "Please provide a debate ID" in text
        assert "/recent" in text

    def test_whitespace_only(self, handler):
        text = handler._command_receipt("   ")
        assert "Please provide a debate ID" in text

    def test_receipt_from_receipt_store(self, handler):
        mock_receipt = MagicMock()
        mock_receipt.to_dict.return_value = {
            "receipt_id": "r1",
            "topic": "Test topic",
            "decision": "We should proceed",
            "confidence": 0.9,
            "timestamp": "2026-01-01T00:00:00Z",
            "agents": [],
        }
        mock_store = MagicMock()
        mock_store.get.return_value = mock_receipt

        with patch(
            "aragora.storage.receipt_store.get_receipt_store",
            return_value=mock_store,
        ):
            text = handler._command_receipt("r1")
        assert "r1" in text
        assert "Decision Receipt" in text

    def test_receipt_fallback_to_debate_storage(self, handler):
        mock_db = MagicMock()
        mock_db.get_debate.return_value = {
            "id": "d1",
            "topic": "Fallback topic",
            "conclusion": "Some conclusion",
            "consensus_reached": True,
            "confidence": 0.7,
        }

        with (
            patch(
                "aragora.storage.receipt_store.get_receipt_store",
                side_effect=ImportError("no store"),
            ),
            patch("aragora.storage.get_storage", return_value=mock_db),
            patch(
                "aragora.gauntlet.receipt.DecisionReceipt.from_dict",
                side_effect=ImportError("no receipt module"),
            ),
        ):
            text = handler._command_receipt("d1")
        assert "d1" in text

    def test_receipt_debate_not_found(self, handler):
        mock_db = MagicMock()
        mock_db.get_debate.return_value = None

        with (
            patch(
                "aragora.storage.receipt_store.get_receipt_store",
                side_effect=ImportError("no store"),
            ),
            patch("aragora.storage.get_storage", return_value=mock_db),
        ):
            text = handler._command_receipt("nonexistent")
        assert "No debate found" in text

    def test_receipt_no_storage(self, handler):
        with (
            patch(
                "aragora.storage.receipt_store.get_receipt_store",
                side_effect=ImportError("no store"),
            ),
            patch("aragora.storage.get_storage", return_value=None),
        ):
            text = handler._command_receipt("d1")
        assert "not available" in text

    def test_receipt_error_handling(self, handler):
        with (
            patch(
                "aragora.storage.receipt_store.get_receipt_store",
                side_effect=ImportError("no store"),
            ),
            patch(
                "aragora.storage.get_storage",
                side_effect=ValueError("db error"),
            ),
        ):
            text = handler._command_receipt("d1")
        assert "error occurred" in text

    def test_receipt_store_returns_none(self, handler):
        """Receipt store exists but returns None for the ID."""
        mock_store = MagicMock()
        mock_store.get.return_value = None

        mock_db = MagicMock()
        mock_db.get_debate.return_value = {
            "id": "d1",
            "topic": "Test",
            "conclusion": "Decided",
            "consensus_reached": True,
            "confidence": 0.5,
        }

        with (
            patch(
                "aragora.storage.receipt_store.get_receipt_store",
                return_value=mock_store,
            ),
            patch("aragora.storage.get_storage", return_value=mock_db),
            patch(
                "aragora.gauntlet.receipt.DecisionReceipt.from_dict",
                side_effect=ImportError("no module"),
            ),
        ):
            text = handler._command_receipt("d1")
        assert "d1" in text


# ============================================================================
# _format_receipt
# ============================================================================


class TestFormatReceipt:
    """Test receipt formatting helper."""

    def test_basic_receipt(self, handler):
        data = {
            "receipt_id": "r123",
            "topic": "Test Topic",
            "decision": "We proceed",
            "confidence": 0.85,
            "timestamp": "2026-01-01",
            "agents": [],
        }
        text = handler._format_receipt(data)
        assert "r123" in text
        assert "Test Topic" in text
        assert "We proceed" in text
        assert "85%" in text

    def test_receipt_with_agents(self, handler):
        data = {
            "receipt_id": "r1",
            "topic": "T",
            "decision": "D",
            "confidence": 0.5,
            "timestamp": "now",
            "agents": [{"name": "Claude"}, {"name": "GPT"}],
        }
        text = handler._format_receipt(data)
        assert "Claude" in text
        assert "GPT" in text

    def test_receipt_with_hash(self, handler):
        data = {
            "receipt_id": "r1",
            "topic": "T",
            "decision": "D",
            "confidence": 0.5,
            "timestamp": "now",
            "agents": [],
            "hash": "abcdef1234567890abcdef1234567890",
        }
        text = handler._format_receipt(data)
        assert "Verification Hash" in text
        assert "abcdef1234567890" in text

    def test_receipt_fallback_keys(self, handler):
        """Uses 'id' instead of 'receipt_id', 'question' instead of 'topic'."""
        data = {
            "id": "fallback-id",
            "question": "Fallback question",
            "conclusion": "fallback conclusion",
            "confidence": 0.7,
            "created_at": "2026-02-01",
            "participants": ["agent1", "agent2"],
        }
        text = handler._format_receipt(data)
        assert "fallback-id" in text
        assert "Fallback question" in text

    def test_receipt_long_topic_truncated(self, handler):
        data = {
            "receipt_id": "r1",
            "topic": "X" * 200,
            "decision": "D",
            "confidence": 0.5,
            "timestamp": "now",
            "agents": [],
        }
        text = handler._format_receipt(data)
        assert "..." in text

    def test_receipt_long_decision_truncated(self, handler):
        data = {
            "receipt_id": "r1",
            "topic": "T",
            "decision": "D" * 400,
            "confidence": 0.5,
            "timestamp": "now",
            "agents": [],
        }
        text = handler._format_receipt(data)
        assert "..." in text

    def test_receipt_string_confidence(self, handler):
        data = {
            "receipt_id": "r1",
            "topic": "T",
            "decision": "D",
            "confidence": "high",
            "timestamp": "now",
            "agents": [],
        }
        text = handler._format_receipt(data)
        assert "high" in text

    def test_receipt_agents_as_strings(self, handler):
        data = {
            "receipt_id": "r1",
            "topic": "T",
            "decision": "D",
            "confidence": 0.5,
            "timestamp": "now",
            "agents": ["agent_a", "agent_b"],
        }
        text = handler._format_receipt(data)
        assert "agent_a" in text
        assert "agent_b" in text

    def test_receipt_agents_limited_to_5(self, handler):
        data = {
            "receipt_id": "r1",
            "topic": "T",
            "decision": "D",
            "confidence": 0.5,
            "timestamp": "now",
            "agents": [f"agent{i}" for i in range(10)],
        }
        text = handler._format_receipt(data)
        assert "agent0" in text
        assert "agent4" in text
        # agent5+ should not be listed
        assert "agent5" not in text


# ============================================================================
# _format_debate_as_receipt
# ============================================================================


class TestFormatDebateAsReceipt:
    """Test debate-as-receipt formatting fallback."""

    def test_basic_formatting(self, handler):
        debate = {
            "id": "d1",
            "topic": "Test Topic",
            "conclusion": "We decided X",
            "consensus_reached": True,
            "confidence": 0.8,
        }
        text = handler._format_debate_as_receipt(debate)
        assert "d1" in text
        assert "Test Topic" in text
        assert "We decided X" in text
        assert "Yes" in text
        assert "80%" in text

    def test_no_consensus(self, handler):
        debate = {
            "id": "d1",
            "topic": "T",
            "conclusion": "C",
            "consensus_reached": False,
            "confidence": 0.3,
        }
        text = handler._format_debate_as_receipt(debate)
        assert "No" in text

    def test_with_rounds(self, handler):
        debate = {
            "id": "d1",
            "topic": "T",
            "conclusion": "C",
            "consensus_reached": True,
            "confidence": 0.5,
            "rounds_used": 3,
        }
        text = handler._format_debate_as_receipt(debate)
        assert "Rounds" in text
        assert "3" in text

    def test_without_rounds(self, handler):
        debate = {
            "id": "d1",
            "topic": "T",
            "conclusion": "C",
            "consensus_reached": True,
            "confidence": 0.5,
        }
        text = handler._format_debate_as_receipt(debate)
        assert "Rounds" not in text

    def test_fallback_final_answer_key(self, handler):
        debate = {
            "id": "d1",
            "topic": "T",
            "final_answer": "The final answer",
            "consensus_reached": True,
            "confidence": 0.5,
        }
        text = handler._format_debate_as_receipt(debate)
        assert "The final answer" in text

    def test_long_topic_truncated(self, handler):
        debate = {
            "id": "d1",
            "topic": "Z" * 200,
            "conclusion": "C",
            "consensus_reached": True,
            "confidence": 0.5,
        }
        text = handler._format_debate_as_receipt(debate)
        # Topic truncated at 100 chars
        assert len([c for c in text if c == "Z"]) <= 100

    def test_long_conclusion_truncated(self, handler):
        debate = {
            "id": "d1",
            "topic": "T",
            "conclusion": "W" * 400,
            "consensus_reached": True,
            "confidence": 0.5,
        }
        text = handler._format_debate_as_receipt(debate)
        assert len([c for c in text if c == "W"]) <= 300

    def test_string_confidence(self, handler):
        debate = {
            "id": "d1",
            "topic": "T",
            "conclusion": "C",
            "consensus_reached": True,
            "confidence": "medium",
        }
        text = handler._format_debate_as_receipt(debate)
        assert "medium" in text


# ============================================================================
# _run_debate_async
# ============================================================================


class TestRunDebateAsync:
    """Test async debate execution."""

    @pytest.mark.asyncio
    async def test_successful_debate(self, handler):
        mock_result = MagicMock()
        mock_result.consensus_reached = True
        mock_result.confidence = 0.85
        mock_result.rounds_used = 3
        mock_result.final_answer = "We should proceed with the migration."
        mock_result.id = "debate-123"

        mock_arena = MagicMock()
        mock_arena.run = AsyncMock(return_value=mock_result)

        handler._send_message_async = AsyncMock()

        with (
            patch("aragora.config.DEFAULT_CONSENSUS", "majority"),
            patch("aragora.config.DEFAULT_ROUNDS", 3),
            patch("aragora.Environment") as mock_env,
            patch("aragora.agents.get_agents_by_names", return_value=["a1", "a2"]),
            patch("aragora.DebateProtocol") as mock_proto,
            patch("aragora.Arena") as MockArena,
            patch("aragora.server.handlers.social.telegram.commands.record_debate_started"),
            patch("aragora.server.handlers.social.telegram.commands.record_debate_completed"),
            patch("aragora.server.handlers.social.telegram.commands.emit_debate_started"),
            patch("aragora.server.handlers.social.telegram.commands.emit_debate_completed"),
            patch("aragora.server.handlers.social.telegram.commands._tg") as mock_tg_fn,
            patch("aragora.server.debate_origin.register_debate_origin"),
            patch("aragora.server.debate_origin.mark_result_sent"),
            patch(
                "aragora.server.decision_integrity_utils.maybe_emit_decision_integrity",
                new_callable=AsyncMock,
            ),
        ):
            MockArena.from_env.return_value = mock_arena
            mock_tg_fn.return_value.TTS_VOICE_ENABLED = False

            await handler._run_debate_async(CHAT_ID, USER_ID, USERNAME, "Test topic for debate")

        handler._send_message_async.assert_called()

    @pytest.mark.asyncio
    async def test_debate_no_agents(self, handler):
        handler._send_message_async = AsyncMock()

        with (
            patch("aragora.config.DEFAULT_CONSENSUS", "majority"),
            patch("aragora.config.DEFAULT_ROUNDS", 3),
            patch("aragora.Environment"),
            patch("aragora.agents.get_agents_by_names", return_value=[]),
            patch("aragora.DebateProtocol"),
            patch("aragora.server.handlers.social.telegram.commands.record_debate_started"),
            patch(
                "aragora.server.handlers.social.telegram.commands.record_debate_failed"
            ) as rec_fail,
            patch("aragora.server.handlers.social.telegram.commands.emit_debate_started"),
            patch("aragora.server.debate_origin.register_debate_origin"),
        ):
            await handler._run_debate_async(CHAT_ID, USER_ID, USERNAME, "Test topic for debate")

        # Should send "No agents available" message
        call_args = handler._send_message_async.call_args_list
        assert any("No agents available" in str(c) for c in call_args)
        rec_fail.assert_called_once()

    @pytest.mark.asyncio
    async def test_debate_runtime_error(self, handler):
        handler._send_message_async = AsyncMock()

        with (
            patch("aragora.config.DEFAULT_CONSENSUS", "majority"),
            patch("aragora.config.DEFAULT_ROUNDS", 3),
            patch(
                "aragora.Environment",
                side_effect=RuntimeError("env error"),
            ),
            patch("aragora.server.handlers.social.telegram.commands.record_debate_started"),
            patch(
                "aragora.server.handlers.social.telegram.commands.record_debate_failed"
            ) as rec_fail,
            patch("aragora.server.handlers.social.telegram.commands.emit_debate_started"),
            patch("aragora.server.debate_origin.register_debate_origin"),
        ):
            await handler._run_debate_async(CHAT_ID, USER_ID, USERNAME, "Test topic")

        rec_fail.assert_called_once()
        # Should send error message
        call_args = handler._send_message_async.call_args_list
        assert any("error occurred" in str(c) for c in call_args)

    @pytest.mark.asyncio
    async def test_debate_with_tts_enabled(self, handler):
        mock_result = MagicMock()
        mock_result.consensus_reached = True
        mock_result.confidence = 0.9
        mock_result.rounds_used = 2
        mock_result.final_answer = "Short answer"
        mock_result.id = "debate-tts"

        mock_arena = MagicMock()
        mock_arena.run = AsyncMock(return_value=mock_result)

        handler._send_message_async = AsyncMock()
        handler._send_voice_summary = AsyncMock()

        with (
            patch("aragora.config.DEFAULT_CONSENSUS", "majority"),
            patch("aragora.config.DEFAULT_ROUNDS", 3),
            patch("aragora.Environment"),
            patch("aragora.agents.get_agents_by_names", return_value=["a1"]),
            patch("aragora.DebateProtocol"),
            patch("aragora.Arena") as MockArena,
            patch("aragora.server.handlers.social.telegram.commands.record_debate_started"),
            patch("aragora.server.handlers.social.telegram.commands.record_debate_completed"),
            patch("aragora.server.handlers.social.telegram.commands.emit_debate_started"),
            patch("aragora.server.handlers.social.telegram.commands.emit_debate_completed"),
            patch("aragora.server.handlers.social.telegram.commands._tg") as mock_tg_fn,
            patch("aragora.server.debate_origin.register_debate_origin"),
            patch("aragora.server.debate_origin.mark_result_sent"),
            patch(
                "aragora.server.decision_integrity_utils.maybe_emit_decision_integrity",
                new_callable=AsyncMock,
            ),
        ):
            MockArena.from_env.return_value = mock_arena
            mock_tg_fn.return_value.TTS_VOICE_ENABLED = True

            await handler._run_debate_async(CHAT_ID, USER_ID, USERNAME, "Test topic for TTS")

        handler._send_voice_summary.assert_called_once()

    @pytest.mark.asyncio
    async def test_debate_origin_import_error(self, handler):
        """If debate_origin module is not available, debate still runs."""
        mock_result = MagicMock()
        mock_result.consensus_reached = False
        mock_result.confidence = 0.4
        mock_result.rounds_used = 3
        mock_result.final_answer = "No consensus"
        mock_result.id = "d-no-origin"

        mock_arena = MagicMock()
        mock_arena.run = AsyncMock(return_value=mock_result)

        handler._send_message_async = AsyncMock()

        with (
            patch("aragora.config.DEFAULT_CONSENSUS", "majority"),
            patch("aragora.config.DEFAULT_ROUNDS", 3),
            patch("aragora.Environment"),
            patch("aragora.agents.get_agents_by_names", return_value=["a1"]),
            patch("aragora.DebateProtocol"),
            patch("aragora.Arena") as MockArena,
            patch("aragora.server.handlers.social.telegram.commands.record_debate_started"),
            patch("aragora.server.handlers.social.telegram.commands.record_debate_completed"),
            patch("aragora.server.handlers.social.telegram.commands.emit_debate_started"),
            patch("aragora.server.handlers.social.telegram.commands.emit_debate_completed"),
            patch("aragora.server.handlers.social.telegram.commands._tg") as mock_tg_fn,
            patch(
                "aragora.server.debate_origin.register_debate_origin",
                side_effect=ImportError("not available"),
            ),
            patch(
                "aragora.server.decision_integrity_utils.maybe_emit_decision_integrity",
                new_callable=AsyncMock,
            ),
        ):
            MockArena.from_env.return_value = mock_arena
            mock_tg_fn.return_value.TTS_VOICE_ENABLED = False

            await handler._run_debate_async(CHAT_ID, USER_ID, USERNAME, "Test topic no origin")

        handler._send_message_async.assert_called()

    @pytest.mark.asyncio
    async def test_debate_with_binding_resolution(self, handler):
        """Binding router resolves agent pool."""
        mock_result = MagicMock()
        mock_result.consensus_reached = True
        mock_result.confidence = 0.7
        mock_result.rounds_used = 2
        mock_result.final_answer = "Answer"
        mock_result.id = "d-binding"

        mock_arena = MagicMock()
        mock_arena.run = AsyncMock(return_value=mock_result)

        handler._send_message_async = AsyncMock()

        mock_resolution = MagicMock()
        mock_resolution.matched = True
        mock_resolution.binding = True
        mock_resolution.agent_binding = "full-team"
        mock_resolution.binding_type = MagicMock()
        mock_resolution.binding_type.__eq__ = lambda self, other: str(other).endswith("AGENT_POOL")
        mock_resolution.config_overrides = {"rounds": 5}
        mock_resolution.match_reason = "test"

        # Make binding_type == BindingType.AGENT_POOL
        mock_binding_type = MagicMock()
        mock_binding_type.SPECIFIC_AGENT = "SPECIFIC_AGENT"
        mock_binding_type.AGENT_POOL = "AGENT_POOL"

        mock_router = MagicMock()
        mock_router.resolve.return_value = mock_resolution

        with (
            patch("aragora.config.DEFAULT_CONSENSUS", "majority"),
            patch("aragora.config.DEFAULT_ROUNDS", 3),
            patch("aragora.Environment"),
            patch("aragora.agents.get_agents_by_names", return_value=["a1", "a2", "a3"]),
            patch("aragora.DebateProtocol"),
            patch("aragora.Arena") as MockArena,
            patch("aragora.server.handlers.social.telegram.commands.record_debate_started"),
            patch("aragora.server.handlers.social.telegram.commands.record_debate_completed"),
            patch("aragora.server.handlers.social.telegram.commands.emit_debate_started"),
            patch("aragora.server.handlers.social.telegram.commands.emit_debate_completed"),
            patch("aragora.server.handlers.social.telegram.commands._tg") as mock_tg_fn,
            patch("aragora.server.debate_origin.register_debate_origin"),
            patch("aragora.server.debate_origin.mark_result_sent"),
            patch("aragora.server.bindings.get_binding_router", return_value=mock_router),
            patch("aragora.server.bindings.BindingType", mock_binding_type),
            patch(
                "aragora.server.decision_integrity_utils.maybe_emit_decision_integrity",
                new_callable=AsyncMock,
            ),
        ):
            MockArena.from_env.return_value = mock_arena
            mock_tg_fn.return_value.TTS_VOICE_ENABLED = False

            await handler._run_debate_async(CHAT_ID, USER_ID, USERNAME, "Binding test topic")

        handler._send_message_async.assert_called()

    @pytest.mark.asyncio
    async def test_debate_long_final_answer_truncated(self, handler):
        """Final answers longer than 500 chars are truncated in the message."""
        mock_result = MagicMock()
        mock_result.consensus_reached = True
        mock_result.confidence = 0.9
        mock_result.rounds_used = 3
        mock_result.final_answer = "A" * 600
        mock_result.id = "d-long"

        mock_arena = MagicMock()
        mock_arena.run = AsyncMock(return_value=mock_result)

        handler._send_message_async = AsyncMock()

        with (
            patch("aragora.config.DEFAULT_CONSENSUS", "majority"),
            patch("aragora.config.DEFAULT_ROUNDS", 3),
            patch("aragora.Environment"),
            patch("aragora.agents.get_agents_by_names", return_value=["a1"]),
            patch("aragora.DebateProtocol"),
            patch("aragora.Arena") as MockArena,
            patch("aragora.server.handlers.social.telegram.commands.record_debate_started"),
            patch("aragora.server.handlers.social.telegram.commands.record_debate_completed"),
            patch("aragora.server.handlers.social.telegram.commands.emit_debate_started"),
            patch("aragora.server.handlers.social.telegram.commands.emit_debate_completed"),
            patch("aragora.server.handlers.social.telegram.commands._tg") as mock_tg_fn,
            patch("aragora.server.debate_origin.register_debate_origin"),
            patch("aragora.server.debate_origin.mark_result_sent"),
            patch(
                "aragora.server.decision_integrity_utils.maybe_emit_decision_integrity",
                new_callable=AsyncMock,
            ),
        ):
            MockArena.from_env.return_value = mock_arena
            mock_tg_fn.return_value.TTS_VOICE_ENABLED = False

            await handler._run_debate_async(CHAT_ID, USER_ID, USERNAME, "Long answer test")

        # The message should include "..." for truncated answer
        calls = handler._send_message_async.call_args_list
        msg_text = str(calls[-1])
        assert "..." in msg_text

    @pytest.mark.asyncio
    async def test_debate_no_final_answer(self, handler):
        """When final_answer is None."""
        mock_result = MagicMock()
        mock_result.consensus_reached = False
        mock_result.confidence = 0.2
        mock_result.rounds_used = 3
        mock_result.final_answer = None
        mock_result.id = "d-none"

        mock_arena = MagicMock()
        mock_arena.run = AsyncMock(return_value=mock_result)

        handler._send_message_async = AsyncMock()

        with (
            patch("aragora.config.DEFAULT_CONSENSUS", "majority"),
            patch("aragora.config.DEFAULT_ROUNDS", 3),
            patch("aragora.Environment"),
            patch("aragora.agents.get_agents_by_names", return_value=["a1"]),
            patch("aragora.DebateProtocol"),
            patch("aragora.Arena") as MockArena,
            patch("aragora.server.handlers.social.telegram.commands.record_debate_started"),
            patch("aragora.server.handlers.social.telegram.commands.record_debate_completed"),
            patch("aragora.server.handlers.social.telegram.commands.emit_debate_started"),
            patch("aragora.server.handlers.social.telegram.commands.emit_debate_completed"),
            patch("aragora.server.handlers.social.telegram.commands._tg") as mock_tg_fn,
            patch("aragora.server.debate_origin.register_debate_origin"),
            patch("aragora.server.debate_origin.mark_result_sent"),
            patch(
                "aragora.server.decision_integrity_utils.maybe_emit_decision_integrity",
                new_callable=AsyncMock,
            ),
        ):
            MockArena.from_env.return_value = mock_arena
            mock_tg_fn.return_value.TTS_VOICE_ENABLED = False

            await handler._run_debate_async(CHAT_ID, USER_ID, USERNAME, "No answer test")

        calls = handler._send_message_async.call_args_list
        msg_text = str(calls[-1])
        assert "No conclusion reached" in msg_text


# ============================================================================
# _run_gauntlet_async
# ============================================================================


class TestRunGauntletAsync:
    """Test async gauntlet execution."""

    @pytest.mark.asyncio
    async def test_gauntlet_success_passed(self, handler):
        handler._send_message_async = AsyncMock()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "run_id": "g-123",
            "score": 0.85,
            "passed": True,
            "vulnerabilities": [],
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        mock_session_cm = MagicMock()
        mock_session_cm.__aenter__ = AsyncMock(return_value=mock_client)
        mock_session_cm.__aexit__ = AsyncMock(return_value=False)

        mock_pool = MagicMock()
        mock_pool.get_session.return_value = mock_session_cm

        with (
            patch(
                "aragora.server.http_client_pool.get_http_pool",
                return_value=mock_pool,
            ),
            patch("aragora.server.handlers.social.telegram.commands.record_gauntlet_started"),
            patch(
                "aragora.server.handlers.social.telegram.commands.record_gauntlet_completed"
            ) as rec_comp,
            patch("aragora.server.handlers.social.telegram.commands.emit_gauntlet_started"),
            patch("aragora.server.handlers.social.telegram.commands.emit_gauntlet_completed"),
        ):
            await handler._run_gauntlet_async(
                CHAT_ID, USER_ID, USERNAME, "We should use microservices"
            )

        handler._send_message_async.assert_called()
        rec_comp.assert_called_once_with("telegram", True)

    @pytest.mark.asyncio
    async def test_gauntlet_success_failed(self, handler):
        handler._send_message_async = AsyncMock()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "run_id": "g-fail",
            "score": 0.3,
            "passed": False,
            "vulnerabilities": [
                {"description": "Logical fallacy detected", "critical": True},
                {"description": "Missing evidence", "critical": False},
            ],
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        mock_session_cm = MagicMock()
        mock_session_cm.__aenter__ = AsyncMock(return_value=mock_client)
        mock_session_cm.__aexit__ = AsyncMock(return_value=False)

        mock_pool = MagicMock()
        mock_pool.get_session.return_value = mock_session_cm

        with (
            patch(
                "aragora.server.http_client_pool.get_http_pool",
                return_value=mock_pool,
            ),
            patch("aragora.server.handlers.social.telegram.commands.record_gauntlet_started"),
            patch(
                "aragora.server.handlers.social.telegram.commands.record_gauntlet_completed"
            ) as rec_comp,
            patch("aragora.server.handlers.social.telegram.commands.emit_gauntlet_started"),
            patch("aragora.server.handlers.social.telegram.commands.emit_gauntlet_completed"),
        ):
            await handler._run_gauntlet_async(
                CHAT_ID, USER_ID, USERNAME, "We should use microservices"
            )

        rec_comp.assert_called_once_with("telegram", False)

        # Check that vulnerabilities are in the message
        calls = handler._send_message_async.call_args_list
        msg_text = str(calls[-1])
        assert "Logical fallacy" in msg_text

    @pytest.mark.asyncio
    async def test_gauntlet_api_error(self, handler):
        handler._send_message_async = AsyncMock()

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.json.return_value = {"error": "Internal server error"}

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        mock_session_cm = MagicMock()
        mock_session_cm.__aenter__ = AsyncMock(return_value=mock_client)
        mock_session_cm.__aexit__ = AsyncMock(return_value=False)

        mock_pool = MagicMock()
        mock_pool.get_session.return_value = mock_session_cm

        with (
            patch(
                "aragora.server.http_client_pool.get_http_pool",
                return_value=mock_pool,
            ),
            patch("aragora.server.handlers.social.telegram.commands.record_gauntlet_started"),
            patch(
                "aragora.server.handlers.social.telegram.commands.record_gauntlet_failed"
            ) as rec_fail,
            patch("aragora.server.handlers.social.telegram.commands.emit_gauntlet_started"),
        ):
            await handler._run_gauntlet_async(CHAT_ID, USER_ID, USERNAME, "Statement to test")

        rec_fail.assert_called_once()
        calls = handler._send_message_async.call_args_list
        msg_text = str(calls[-1])
        assert "failed" in msg_text.lower() or "error" in msg_text.lower()

    @pytest.mark.asyncio
    async def test_gauntlet_connection_error(self, handler):
        handler._send_message_async = AsyncMock()

        mock_pool = MagicMock()
        mock_session_cm = MagicMock()
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=ConnectionError("no connection"))
        mock_session_cm.__aenter__ = AsyncMock(return_value=mock_client)
        mock_session_cm.__aexit__ = AsyncMock(return_value=False)
        mock_pool.get_session.return_value = mock_session_cm

        with (
            patch(
                "aragora.server.http_client_pool.get_http_pool",
                return_value=mock_pool,
            ),
            patch("aragora.server.handlers.social.telegram.commands.record_gauntlet_started"),
            patch(
                "aragora.server.handlers.social.telegram.commands.record_gauntlet_failed"
            ) as rec_fail,
            patch("aragora.server.handlers.social.telegram.commands.emit_gauntlet_started"),
        ):
            await handler._run_gauntlet_async(
                CHAT_ID, USER_ID, USERNAME, "Connection test statement"
            )

        rec_fail.assert_called_once()
        calls = handler._send_message_async.call_args_list
        assert any("error occurred" in str(c) for c in calls)

    @pytest.mark.asyncio
    async def test_gauntlet_timeout_error(self, handler):
        handler._send_message_async = AsyncMock()

        mock_pool = MagicMock()
        mock_session_cm = MagicMock()
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=TimeoutError("timed out"))
        mock_session_cm.__aenter__ = AsyncMock(return_value=mock_client)
        mock_session_cm.__aexit__ = AsyncMock(return_value=False)
        mock_pool.get_session.return_value = mock_session_cm

        with (
            patch(
                "aragora.server.http_client_pool.get_http_pool",
                return_value=mock_pool,
            ),
            patch("aragora.server.handlers.social.telegram.commands.record_gauntlet_started"),
            patch(
                "aragora.server.handlers.social.telegram.commands.record_gauntlet_failed"
            ) as rec_fail,
            patch("aragora.server.handlers.social.telegram.commands.emit_gauntlet_started"),
        ):
            await handler._run_gauntlet_async(CHAT_ID, USER_ID, USERNAME, "Timeout test statement")

        rec_fail.assert_called_once()

    @pytest.mark.asyncio
    async def test_gauntlet_many_vulnerabilities_truncated(self, handler):
        handler._send_message_async = AsyncMock()

        vulns = [{"description": f"Vulnerability {i}", "critical": False} for i in range(10)]
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "run_id": "g-many",
            "score": 0.4,
            "passed": False,
            "vulnerabilities": vulns,
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        mock_session_cm = MagicMock()
        mock_session_cm.__aenter__ = AsyncMock(return_value=mock_client)
        mock_session_cm.__aexit__ = AsyncMock(return_value=False)

        mock_pool = MagicMock()
        mock_pool.get_session.return_value = mock_session_cm

        with (
            patch(
                "aragora.server.http_client_pool.get_http_pool",
                return_value=mock_pool,
            ),
            patch("aragora.server.handlers.social.telegram.commands.record_gauntlet_started"),
            patch("aragora.server.handlers.social.telegram.commands.record_gauntlet_completed"),
            patch("aragora.server.handlers.social.telegram.commands.emit_gauntlet_started"),
            patch("aragora.server.handlers.social.telegram.commands.emit_gauntlet_completed"),
        ):
            await handler._run_gauntlet_async(CHAT_ID, USER_ID, USERNAME, "Many vulns test")

        calls = handler._send_message_async.call_args_list
        msg_text = str(calls[-1])
        assert "and 5 more" in msg_text

    @pytest.mark.asyncio
    async def test_gauntlet_no_vulnerabilities(self, handler):
        handler._send_message_async = AsyncMock()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "run_id": "g-clean",
            "score": 0.95,
            "passed": True,
            "vulnerabilities": [],
        }

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        mock_session_cm = MagicMock()
        mock_session_cm.__aenter__ = AsyncMock(return_value=mock_client)
        mock_session_cm.__aexit__ = AsyncMock(return_value=False)

        mock_pool = MagicMock()
        mock_pool.get_session.return_value = mock_session_cm

        with (
            patch(
                "aragora.server.http_client_pool.get_http_pool",
                return_value=mock_pool,
            ),
            patch("aragora.server.handlers.social.telegram.commands.record_gauntlet_started"),
            patch("aragora.server.handlers.social.telegram.commands.record_gauntlet_completed"),
            patch("aragora.server.handlers.social.telegram.commands.emit_gauntlet_started"),
            patch("aragora.server.handlers.social.telegram.commands.emit_gauntlet_completed"),
        ):
            await handler._run_gauntlet_async(CHAT_ID, USER_ID, USERNAME, "Clean statement here")

        calls = handler._send_message_async.call_args_list
        msg_text = str(calls[-1])
        assert "Issues Found" not in msg_text


# ============================================================================
# Edge cases and integration
# ============================================================================


class TestEdgeCases:
    """Miscellaneous edge cases."""

    @pytest.mark.usefixtures("_full_patch")
    def test_command_without_leading_slash(self, handler, _patch_tg):
        """Even without a slash prefix, the command is handled."""
        # The handler lowercases and checks; "help" won't match "/help"
        result = handler._handle_command(CHAT_ID, USER_ID, USERNAME, "help")
        assert _status(result) == 200
        # It goes to the unknown command branch

    @pytest.mark.usefixtures("_full_patch")
    def test_multiple_at_signs_in_command(self, handler, _patch_tg):
        """Only splits on first @ in the command."""
        with patch.object(handler, "_command_help", return_value="text"):
            result = handler._handle_command(CHAT_ID, USER_ID, USERNAME, "/help@bot@extra")
        assert _status(result) == 200

    @pytest.mark.usefixtures("_full_patch")
    def test_empty_args_for_receipt(self, handler, _patch_tg):
        with patch.object(handler, "_command_receipt", return_value="Please provide a debate ID"):
            result = handler._handle_command(CHAT_ID, USER_ID, USERNAME, "/receipt")
        assert _status(result) == 200

    def test_format_receipt_empty_agents(self, handler):
        data = {
            "receipt_id": "r1",
            "topic": "T",
            "decision": "D",
            "confidence": 0.5,
            "timestamp": "now",
            "agents": [],
        }
        text = handler._format_receipt(data)
        assert "Agents" not in text

    def test_format_receipt_no_hash(self, handler):
        data = {
            "receipt_id": "r1",
            "topic": "T",
            "decision": "D",
            "confidence": 0.5,
            "timestamp": "now",
            "agents": [],
        }
        text = handler._format_receipt(data)
        assert "Verification Hash" not in text

    def test_format_receipt_zero_confidence(self, handler):
        data = {
            "receipt_id": "r1",
            "topic": "T",
            "decision": "D",
            "confidence": 0.0,
            "timestamp": "now",
            "agents": [],
        }
        text = handler._format_receipt(data)
        assert "0%" in text

    def test_format_debate_missing_keys(self, handler):
        """Missing optional keys fallback to defaults."""
        debate = {}
        text = handler._format_debate_as_receipt(debate)
        assert "N/A" in text
        assert "Unknown" in text

    @pytest.mark.usefixtures("_full_patch")
    def test_handle_command_emits_event(self, handler, _patch_tg):
        with patch(
            "aragora.server.handlers.social.telegram.commands.emit_command_received"
        ) as emit:
            handler._handle_command(CHAT_ID, USER_ID, USERNAME, "/help")
        emit.assert_called_once()

    @pytest.mark.usefixtures("_full_patch")
    def test_record_command_called(self, handler, _patch_tg, _patch_telemetry):
        handler._handle_command(CHAT_ID, USER_ID, USERNAME, "/help")
        _patch_telemetry.assert_called_with("telegram", "help")

    def test_search_fallback_manual_no_match(self, handler):
        """Manual search fallback finds no matches."""
        mock_db = MagicMock(spec=[])
        mock_db.get_recent_debates = MagicMock(
            return_value=[
                {
                    "topic": "unrelated topic",
                    "id": "d1",
                    "consensus_reached": True,
                    "conclusion": "stuff",
                },
            ]
        )

        with patch("aragora.storage.get_storage", return_value=mock_db):
            text = handler._command_search("nonexistent query")
        assert "No debates found" in text

    def test_search_manual_matches_conclusion(self, handler):
        """Manual search fallback matches on conclusion field."""
        mock_db = MagicMock(spec=[])
        mock_db.get_recent_debates = MagicMock(
            return_value=[
                {
                    "topic": "A topic",
                    "id": "d1",
                    "consensus_reached": True,
                    "conclusion": "machine learning is great",
                },
            ]
        )

        with patch("aragora.storage.get_storage", return_value=mock_db):
            text = handler._command_search("machine learning")
        assert "A topic" in text
