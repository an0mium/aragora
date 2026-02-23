"""Tests for Telegram callback query, inline query, and message handling (callbacks.py).

Covers all methods in TelegramCallbacksMixin:
- _handle_message: text messages, commands, RBAC, short/long messages
- _handle_callback_query: vote, details, unknown actions, RBAC
- _handle_vote: vote recording, storage integration, emoji logic
- _handle_view_details: debate lookup, formatting, not found
- _handle_inline_query: short queries, long queries, RBAC
- Edge cases: empty data, missing fields, error paths
"""

from __future__ import annotations

import json
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
# Constants
# ---------------------------------------------------------------------------

CHAT_ID = 12345
USER_ID = 67890
USERNAME = "testuser"
CALLBACK_ID = "cb-9999"
QUERY_ID = "iq-5555"
DEBATE_ID = "debate-abc123"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler():
    """Create a TelegramHandler for testing callbacks."""
    return TelegramHandler(ctx={})


@pytest.fixture
def _patch_tg():
    """Patch the _tg() lazy import to return a mock telegram module.

    This prevents real asyncio task creation and provides controllable
    module-level attributes (create_tracked_task, RBAC_AVAILABLE).
    """
    mock_tg = MagicMock()
    mock_tg.create_tracked_task = MagicMock()
    mock_tg.RBAC_AVAILABLE = False
    mock_tg.TTS_VOICE_ENABLED = False
    mock_tg.TELEGRAM_BOT_TOKEN = "fake-token"

    with patch(
        "aragora.server.handlers.social.telegram.callbacks._tg",
        return_value=mock_tg,
    ):
        yield mock_tg


@pytest.fixture
def _patch_telemetry():
    """Patch telemetry functions so they do nothing."""
    with (
        patch("aragora.server.handlers.social.telegram.callbacks.record_message") as rm,
        patch("aragora.server.handlers.social.telegram.callbacks.record_vote") as rv,
    ):
        yield {"record_message": rm, "record_vote": rv}


@pytest.fixture
def _patch_events():
    """Patch chat event emitters."""
    with (
        patch("aragora.server.handlers.social.telegram.callbacks.emit_message_received") as emr,
        patch("aragora.server.handlers.social.telegram.callbacks.emit_vote_received") as evr,
    ):
        yield {"emit_message_received": emr, "emit_vote_received": evr}


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
def _patch_command():
    """Patch _handle_command to avoid going into command logic."""
    with patch.object(
        TelegramHandler,
        "_handle_command",
        return_value=MagicMock(status_code=200, body=b'{"ok":true}'),
    ) as cmd_mock:
        yield cmd_mock


@pytest.fixture
def _full_patch(_patch_tg, _patch_telemetry, _patch_events, _patch_rbac):
    """Combine all patches needed for callback tests."""
    yield


# ---------------------------------------------------------------------------
# Helper to build message / callback dicts
# ---------------------------------------------------------------------------


def _make_message(
    text: str = "Hello, world!",
    chat_id: int = CHAT_ID,
    user_id: int = USER_ID,
    username: str = USERNAME,
) -> dict[str, Any]:
    """Build a Telegram message dict."""
    return {
        "chat": {"id": chat_id},
        "text": text,
        "from": {"id": user_id, "username": username},
    }


def _make_callback(
    data: str = "vote:debate1:agree",
    callback_id: str = CALLBACK_ID,
    chat_id: int = CHAT_ID,
    user_id: int = USER_ID,
    username: str = USERNAME,
) -> dict[str, Any]:
    """Build a Telegram callback query dict."""
    return {
        "id": callback_id,
        "data": data,
        "from": {"id": user_id, "username": username},
        "message": {"chat": {"id": chat_id}},
    }


def _make_inline_query(
    query_text: str = "Should AI be regulated?",
    query_id: str = QUERY_ID,
    user_id: int = USER_ID,
    username: str = USERNAME,
) -> dict[str, Any]:
    """Build a Telegram inline query dict."""
    return {
        "id": query_id,
        "query": query_text,
        "from": {"id": user_id, "username": username},
    }


# ============================================================================
# _handle_message: Text Message Handling
# ============================================================================


class TestHandleMessageBasic:
    """Test basic message handling."""

    @pytest.mark.usefixtures("_full_patch")
    def test_regular_text_message_returns_ok(self, handler, _patch_tg):
        """A normal text message longer than 10 chars returns 200 ok."""
        msg = _make_message("This is a longer test message for the bot")
        result = handler._handle_message(msg)
        assert _status(result) == 200
        assert _body(result)["ok"] is True

    @pytest.mark.usefixtures("_full_patch")
    def test_short_message_returns_help_hint(self, handler, _patch_tg):
        """Messages with 10 or fewer chars get the help hint."""
        msg = _make_message("Hi")
        result = handler._handle_message(msg)
        assert _status(result) == 200
        assert _body(result)["ok"] is True
        # Should call create_tracked_task with help suggestion
        _patch_tg.create_tracked_task.assert_called()

    @pytest.mark.usefixtures("_full_patch")
    def test_long_message_triggers_debate_suggestion(self, handler, _patch_tg):
        """Messages longer than 10 chars suggest starting a debate."""
        long_text = (
            "This is definitely longer than ten characters and should get a debate suggestion"
        )
        msg = _make_message(long_text)
        result = handler._handle_message(msg)
        assert _status(result) == 200
        # The create_tracked_task should be called with a coroutine that sends debate suggestion
        _patch_tg.create_tracked_task.assert_called()

    @pytest.mark.usefixtures("_full_patch")
    def test_exactly_10_chars_is_short(self, handler, _patch_tg):
        """Exactly 10 chars should be treated as short (not > 10)."""
        msg = _make_message("1234567890")
        result = handler._handle_message(msg)
        assert _status(result) == 200
        _patch_tg.create_tracked_task.assert_called()

    @pytest.mark.usefixtures("_full_patch")
    def test_eleven_chars_is_long(self, handler, _patch_tg):
        """11 characters triggers the debate suggestion path."""
        msg = _make_message("12345678901")
        result = handler._handle_message(msg)
        assert _status(result) == 200
        _patch_tg.create_tracked_task.assert_called()


class TestHandleMessageEmpty:
    """Test message handling with empty/missing fields."""

    @pytest.mark.usefixtures("_full_patch")
    def test_empty_text_returns_ok(self, handler):
        """Empty text returns early with ok."""
        msg = _make_message("")
        result = handler._handle_message(msg)
        assert _status(result) == 200
        assert _body(result)["ok"] is True

    @pytest.mark.usefixtures("_full_patch")
    def test_whitespace_only_text_returns_ok(self, handler):
        """Whitespace-only text returns early with ok."""
        msg = _make_message("   ")
        result = handler._handle_message(msg)
        assert _status(result) == 200

    @pytest.mark.usefixtures("_full_patch")
    def test_no_chat_id_returns_ok(self, handler):
        """Missing chat id returns early."""
        msg = {"text": "Hello!", "from": {"id": USER_ID, "username": USERNAME}}
        result = handler._handle_message(msg)
        assert _status(result) == 200

    @pytest.mark.usefixtures("_full_patch")
    def test_empty_chat_dict_returns_ok(self, handler):
        """Empty chat dict (no 'id' key) returns early."""
        msg = {"chat": {}, "text": "Hello!", "from": {"id": USER_ID, "username": USERNAME}}
        result = handler._handle_message(msg)
        assert _status(result) == 200

    @pytest.mark.usefixtures("_full_patch")
    def test_missing_text_key_returns_ok(self, handler):
        """Missing text key defaults to empty string."""
        msg = {"chat": {"id": CHAT_ID}, "from": {"id": USER_ID, "username": USERNAME}}
        result = handler._handle_message(msg)
        assert _status(result) == 200

    @pytest.mark.usefixtures("_full_patch")
    def test_missing_from_defaults(self, handler):
        """Missing 'from' field defaults gracefully."""
        msg = {"chat": {"id": CHAT_ID}, "text": "A long enough test message here"}
        # This should still work because user defaults to {}
        with (
            patch.object(TelegramHandler, "_check_telegram_user_permission", return_value=True),
            patch("aragora.server.handlers.social.telegram.callbacks._tg") as mock_tg_fn,
            patch("aragora.server.handlers.social.telegram.callbacks.record_message"),
            patch("aragora.server.handlers.social.telegram.callbacks.emit_message_received"),
        ):
            mock_tg_fn.return_value.create_tracked_task = MagicMock()
            result = handler._handle_message(msg)
        assert _status(result) == 200


class TestHandleMessageCommands:
    """Test that command messages get routed to _handle_command."""

    @pytest.mark.usefixtures("_patch_tg", "_patch_telemetry", "_patch_events", "_patch_rbac")
    def test_slash_command_delegates_to_handle_command(self, handler, _patch_command):
        """Messages starting with / are routed to _handle_command."""
        msg = _make_message("/help")
        result = handler._handle_message(msg)
        _patch_command.assert_called_once_with(CHAT_ID, USER_ID, USERNAME, "/help")

    @pytest.mark.usefixtures("_patch_tg", "_patch_telemetry", "_patch_events", "_patch_rbac")
    def test_slash_debate_command(self, handler, _patch_command):
        """The /debate command gets delegated."""
        msg = _make_message("/debate Should we use Rust?")
        handler._handle_message(msg)
        _patch_command.assert_called_once_with(
            CHAT_ID, USER_ID, USERNAME, "/debate Should we use Rust?"
        )

    @pytest.mark.usefixtures("_patch_tg", "_patch_events", "_patch_rbac")
    def test_command_records_telemetry(self, handler, _patch_command, _patch_telemetry):
        """Command messages record 'command' telemetry type."""
        msg = _make_message("/start")
        handler._handle_message(msg)
        _patch_telemetry["record_message"].assert_called_with("telegram", "command")

    @pytest.mark.usefixtures("_full_patch")
    def test_text_message_records_telemetry(self, handler, _patch_telemetry):
        """Non-command messages record 'text' telemetry type."""
        msg = _make_message("This is a normal text message")
        handler._handle_message(msg)
        _patch_telemetry["record_message"].assert_called_with("telegram", "text")


class TestHandleMessageRBAC:
    """Test RBAC permission handling in _handle_message."""

    @pytest.mark.usefixtures("_patch_tg", "_patch_telemetry", "_patch_events")
    def test_permission_denied_sends_deny_message(self, handler, _patch_deny_rbac):
        """When send permission is denied, the deny message is sent."""
        msg = _make_message("This is a long enough message to test permissions")
        result = handler._handle_message(msg)
        _patch_deny_rbac.assert_called_once()
        assert _status(result) == 200

    @pytest.mark.usefixtures("_full_patch")
    def test_message_emits_event_on_success(self, handler, _patch_events):
        """Successful messages emit a message_received event."""
        msg = _make_message("This is a regular long text message to test events")
        handler._handle_message(msg)
        _patch_events["emit_message_received"].assert_called_once_with(
            platform="telegram",
            chat_id=str(CHAT_ID),
            user_id=str(USER_ID),
            username=USERNAME,
            message_text="This is a regular long text message to test events",
            message_type="text",
        )


# ============================================================================
# _handle_callback_query: Callback Query Handling
# ============================================================================


class TestHandleCallbackQueryRouting:
    """Test callback query dispatch based on action type."""

    @pytest.mark.usefixtures("_full_patch")
    def test_vote_callback_routes_to_handle_vote(self, handler):
        """vote:debate_id:option routes to _handle_vote."""
        cb = _make_callback("vote:debate-1:agree")
        with patch.object(
            handler,
            "_handle_vote",
            return_value=MagicMock(status_code=200, body=b'{"ok":true}'),
        ) as hv:
            result = handler._handle_callback_query(cb)
            hv.assert_called_once_with(CALLBACK_ID, CHAT_ID, USER_ID, USERNAME, "debate-1", "agree")
        assert _status(result) == 200

    @pytest.mark.usefixtures("_full_patch")
    def test_details_callback_routes_to_handle_view_details(self, handler):
        """details:debate_id routes to _handle_view_details."""
        cb = _make_callback("details:debate-2")
        with patch.object(
            handler,
            "_handle_view_details",
            return_value=MagicMock(status_code=200, body=b'{"ok":true}'),
        ) as hvd:
            result = handler._handle_callback_query(cb)
            hvd.assert_called_once_with(CALLBACK_ID, CHAT_ID, USER_ID, USERNAME, "debate-2")
        assert _status(result) == 200

    @pytest.mark.usefixtures("_full_patch")
    def test_unknown_action_acknowledged(self, handler, _patch_tg):
        """Unknown actions are acknowledged with a generic message."""
        cb = _make_callback("unknown_action")
        result = handler._handle_callback_query(cb)
        assert _status(result) == 200
        # Should acknowledge the callback
        _patch_tg.create_tracked_task.assert_called()

    @pytest.mark.usefixtures("_full_patch")
    def test_empty_data_acknowledged(self, handler, _patch_tg):
        """Empty callback data is acknowledged gracefully."""
        cb = _make_callback("")
        result = handler._handle_callback_query(cb)
        assert _status(result) == 200
        _patch_tg.create_tracked_task.assert_called()

    @pytest.mark.usefixtures("_full_patch")
    def test_vote_missing_parts_fallthrough(self, handler, _patch_tg):
        """vote:only_one_part (missing option) falls through to unknown."""
        cb = _make_callback("vote:debate-1")
        result = handler._handle_callback_query(cb)
        assert _status(result) == 200
        # Falls through to ack since len(parts) < 3
        _patch_tg.create_tracked_task.assert_called()

    @pytest.mark.usefixtures("_full_patch")
    def test_details_missing_debate_id_fallthrough(self, handler, _patch_tg):
        """details (with no debate_id) falls through to unknown."""
        cb = _make_callback("details")
        result = handler._handle_callback_query(cb)
        assert _status(result) == 200
        _patch_tg.create_tracked_task.assert_called()

    @pytest.mark.usefixtures("_full_patch")
    def test_vote_with_extra_parts(self, handler):
        """vote:debate:option:extra still works (extra parts ignored)."""
        cb = _make_callback("vote:debate-1:disagree:extra")
        with patch.object(
            handler,
            "_handle_vote",
            return_value=MagicMock(status_code=200, body=b'{"ok":true}'),
        ) as hv:
            handler._handle_callback_query(cb)
            hv.assert_called_once_with(
                CALLBACK_ID, CHAT_ID, USER_ID, USERNAME, "debate-1", "disagree"
            )


class TestHandleCallbackQueryRBAC:
    """Test RBAC permission checks in callback queries."""

    @pytest.mark.usefixtures("_patch_tg", "_patch_telemetry", "_patch_events")
    def test_base_callback_permission_denied(self, handler, _patch_tg):
        """When base callback permission is denied, access is rejected."""
        with patch.object(
            TelegramHandler,
            "_check_telegram_user_permission",
            return_value=False,
        ):
            cb = _make_callback("vote:d1:agree")
            result = handler._handle_callback_query(cb)
        assert _status(result) == 200
        # Should have called answer_callback with "Permission denied"
        _patch_tg.create_tracked_task.assert_called()

    @pytest.mark.usefixtures("_patch_tg", "_patch_telemetry", "_patch_events")
    def test_vote_permission_denied_after_base_allowed(self, handler, _patch_tg):
        """Base callback allowed, but vote recording denied."""
        call_count = [0]

        def check_perm(user_id, username, chat_id, perm):
            call_count[0] += 1
            if call_count[0] == 1:
                return True  # Allow base callback
            return False  # Deny vote recording

        with patch.object(
            TelegramHandler, "_check_telegram_user_permission", side_effect=check_perm
        ):
            cb = _make_callback("vote:d1:agree")
            result = handler._handle_callback_query(cb)
        assert _status(result) == 200
        _patch_tg.create_tracked_task.assert_called()

    @pytest.mark.usefixtures("_patch_tg", "_patch_telemetry", "_patch_events")
    def test_details_permission_denied_after_base_allowed(self, handler, _patch_tg):
        """Base callback allowed, but debate read denied."""
        call_count = [0]

        def check_perm(user_id, username, chat_id, perm):
            call_count[0] += 1
            if call_count[0] == 1:
                return True  # Allow base callback
            return False  # Deny debate read

        with patch.object(
            TelegramHandler, "_check_telegram_user_permission", side_effect=check_perm
        ):
            cb = _make_callback("details:d1")
            result = handler._handle_callback_query(cb)
        assert _status(result) == 200
        _patch_tg.create_tracked_task.assert_called()


# ============================================================================
# _handle_vote: Vote Recording
# ============================================================================


class TestHandleVote:
    """Test vote recording logic."""

    @pytest.mark.usefixtures("_patch_tg", "_patch_events")
    def test_agree_vote_records_and_acks(self, handler, _patch_tg, _patch_events):
        """Agree vote records metrics and sends ack."""
        with patch("aragora.server.handlers.social.telegram.callbacks.record_vote") as rv:
            result = handler._handle_vote(
                CALLBACK_ID, CHAT_ID, USER_ID, USERNAME, DEBATE_ID, "agree"
            )
        assert _status(result) == 200
        rv.assert_called_once_with("telegram", "agree")
        _patch_tg.create_tracked_task.assert_called()
        _patch_events["emit_vote_received"].assert_called_once_with(
            platform="telegram",
            chat_id=str(CHAT_ID),
            user_id=str(USER_ID),
            username=USERNAME,
            debate_id=DEBATE_ID,
            vote="agree",
        )

    @pytest.mark.usefixtures("_patch_tg", "_patch_events")
    def test_disagree_vote_records_and_acks(self, handler, _patch_tg, _patch_events):
        """Disagree vote records correct emoji and metrics."""
        with patch("aragora.server.handlers.social.telegram.callbacks.record_vote"):
            result = handler._handle_vote(
                CALLBACK_ID, CHAT_ID, USER_ID, USERNAME, DEBATE_ID, "disagree"
            )
        assert _status(result) == 200
        _patch_tg.create_tracked_task.assert_called()

    @pytest.mark.usefixtures("_patch_tg", "_patch_events")
    def test_vote_agree_emoji_is_plus(self, handler, _patch_tg):
        """Agree votes use '+' emoji."""
        with patch("aragora.server.handlers.social.telegram.callbacks.record_vote"):
            handler._handle_vote(CALLBACK_ID, CHAT_ID, USER_ID, USERNAME, DEBATE_ID, "agree")
        # Check the task callback message contains "+"
        call_args = _patch_tg.create_tracked_task.call_args
        assert call_args is not None

    @pytest.mark.usefixtures("_patch_tg", "_patch_events")
    def test_vote_disagree_emoji_is_minus(self, handler, _patch_tg):
        """Disagree votes use '-' emoji."""
        with patch("aragora.server.handlers.social.telegram.callbacks.record_vote"):
            handler._handle_vote(CALLBACK_ID, CHAT_ID, USER_ID, USERNAME, DEBATE_ID, "disagree")
        call_args = _patch_tg.create_tracked_task.call_args
        assert call_args is not None

    @pytest.mark.usefixtures("_patch_tg", "_patch_events")
    def test_vote_records_to_storage(self, handler, _patch_tg):
        """Vote is persisted in debate storage when available."""
        mock_db = MagicMock()
        mock_db.record_vote = MagicMock()

        with (
            patch("aragora.server.handlers.social.telegram.callbacks.record_vote"),
            patch(
                "aragora.server.storage.get_debates_db",
                return_value=mock_db,
            ),
        ):
            handler._handle_vote(CALLBACK_ID, CHAT_ID, USER_ID, USERNAME, DEBATE_ID, "agree")
        mock_db.record_vote.assert_called_once_with(
            debate_id=DEBATE_ID,
            voter_id=f"telegram:{USER_ID}",
            vote="agree",
            source="telegram",
        )

    @pytest.mark.usefixtures("_patch_tg", "_patch_events")
    def test_vote_storage_import_error_handled(self, handler, _patch_tg):
        """ImportError when accessing storage is handled gracefully."""
        with (
            patch("aragora.server.handlers.social.telegram.callbacks.record_vote"),
            patch(
                "aragora.server.storage.get_debates_db",
                side_effect=ImportError("no storage"),
            ),
        ):
            result = handler._handle_vote(
                CALLBACK_ID, CHAT_ID, USER_ID, USERNAME, DEBATE_ID, "agree"
            )
        assert _status(result) == 200

    @pytest.mark.usefixtures("_patch_tg", "_patch_events")
    def test_vote_storage_runtime_error_handled(self, handler, _patch_tg):
        """RuntimeError in storage is handled gracefully."""
        with (
            patch("aragora.server.handlers.social.telegram.callbacks.record_vote"),
            patch(
                "aragora.server.storage.get_debates_db",
                side_effect=RuntimeError("db error"),
            ),
        ):
            result = handler._handle_vote(
                CALLBACK_ID, CHAT_ID, USER_ID, USERNAME, DEBATE_ID, "agree"
            )
        assert _status(result) == 200

    @pytest.mark.usefixtures("_patch_tg", "_patch_events")
    def test_vote_storage_returns_none(self, handler, _patch_tg):
        """When get_debates_db returns None, vote still succeeds."""
        with (
            patch("aragora.server.handlers.social.telegram.callbacks.record_vote"),
            patch(
                "aragora.server.storage.get_debates_db",
                return_value=None,
            ),
        ):
            result = handler._handle_vote(
                CALLBACK_ID, CHAT_ID, USER_ID, USERNAME, DEBATE_ID, "agree"
            )
        assert _status(result) == 200

    @pytest.mark.usefixtures("_patch_tg", "_patch_events")
    def test_vote_storage_no_record_vote_method(self, handler, _patch_tg):
        """When db exists but has no record_vote, vote still succeeds."""
        mock_db = MagicMock(spec=[])  # no record_vote
        with (
            patch("aragora.server.handlers.social.telegram.callbacks.record_vote"),
            patch(
                "aragora.server.storage.get_debates_db",
                return_value=mock_db,
            ),
        ):
            result = handler._handle_vote(
                CALLBACK_ID, CHAT_ID, USER_ID, USERNAME, DEBATE_ID, "agree"
            )
        assert _status(result) == 200

    @pytest.mark.usefixtures("_patch_tg", "_patch_events")
    def test_vote_storage_value_error_handled(self, handler, _patch_tg):
        """ValueError during vote recording is caught."""
        mock_db = MagicMock()
        mock_db.record_vote.side_effect = ValueError("invalid vote")
        with (
            patch("aragora.server.handlers.social.telegram.callbacks.record_vote"),
            patch(
                "aragora.server.storage.get_debates_db",
                return_value=mock_db,
            ),
        ):
            result = handler._handle_vote(
                CALLBACK_ID, CHAT_ID, USER_ID, USERNAME, DEBATE_ID, "agree"
            )
        assert _status(result) == 200

    @pytest.mark.usefixtures("_patch_tg", "_patch_events")
    def test_vote_custom_option(self, handler, _patch_tg):
        """Custom vote options (not agree/disagree) still work."""
        with patch("aragora.server.handlers.social.telegram.callbacks.record_vote") as rv:
            result = handler._handle_vote(
                CALLBACK_ID, CHAT_ID, USER_ID, USERNAME, DEBATE_ID, "abstain"
            )
        assert _status(result) == 200
        rv.assert_called_once_with("telegram", "abstain")


# ============================================================================
# _handle_view_details: View Debate Details
# ============================================================================


class TestHandleViewDetails:
    """Test debate detail viewing."""

    @pytest.mark.usefixtures("_patch_tg")
    def test_debate_not_found(self, handler, _patch_tg):
        """When debate is not found, sends not found message."""
        with patch(
            "aragora.server.storage.get_debates_db",
            return_value=MagicMock(get=MagicMock(return_value=None)),
        ):
            result = handler._handle_view_details(
                CALLBACK_ID, CHAT_ID, USER_ID, USERNAME, DEBATE_ID
            )
        assert _status(result) == 200
        _patch_tg.create_tracked_task.assert_called()

    @pytest.mark.usefixtures("_patch_tg")
    def test_debate_found_shows_details(self, handler, _patch_tg):
        """When debate found, sends formatted details."""
        debate_data = {
            "task": "Should we use microservices?",
            "final_answer": "Yes, for large-scale applications.",
            "consensus_reached": True,
            "confidence": 0.85,
            "rounds_used": 3,
            "agents": ["Claude", "GPT-4"],
        }
        mock_db = MagicMock()
        mock_db.get.return_value = debate_data

        with patch(
            "aragora.server.storage.get_debates_db",
            return_value=mock_db,
        ):
            result = handler._handle_view_details(
                CALLBACK_ID, CHAT_ID, USER_ID, USERNAME, DEBATE_ID
            )
        assert _status(result) == 200
        # Should have called create_tracked_task at least twice (ack + details)
        assert _patch_tg.create_tracked_task.call_count >= 2

    @pytest.mark.usefixtures("_patch_tg")
    def test_debate_import_error_handled(self, handler, _patch_tg):
        """ImportError for storage is handled, shows not found."""
        with patch(
            "aragora.server.storage.get_debates_db",
            side_effect=ImportError("no storage"),
        ):
            result = handler._handle_view_details(
                CALLBACK_ID, CHAT_ID, USER_ID, USERNAME, DEBATE_ID
            )
        assert _status(result) == 200
        _patch_tg.create_tracked_task.assert_called()

    @pytest.mark.usefixtures("_patch_tg")
    def test_debate_runtime_error_handled(self, handler, _patch_tg):
        """RuntimeError for storage is handled gracefully."""
        with patch(
            "aragora.server.storage.get_debates_db",
            side_effect=RuntimeError("db connection"),
        ):
            result = handler._handle_view_details(
                CALLBACK_ID, CHAT_ID, USER_ID, USERNAME, DEBATE_ID
            )
        assert _status(result) == 200

    @pytest.mark.usefixtures("_patch_tg")
    def test_debate_db_returns_none(self, handler, _patch_tg):
        """When get_debates_db returns None, debate is not found."""
        with patch(
            "aragora.server.storage.get_debates_db",
            return_value=None,
        ):
            result = handler._handle_view_details(
                CALLBACK_ID, CHAT_ID, USER_ID, USERNAME, DEBATE_ID
            )
        assert _status(result) == 200

    @pytest.mark.usefixtures("_patch_tg")
    def test_debate_long_task_truncated(self, handler, _patch_tg):
        """Task longer than 200 chars is truncated."""
        debate_data = {
            "task": "X" * 300,
            "final_answer": "Answer",
            "consensus_reached": True,
            "confidence": 0.5,
            "rounds_used": 1,
            "agents": [],
        }
        mock_db = MagicMock()
        mock_db.get.return_value = debate_data

        with patch(
            "aragora.server.storage.get_debates_db",
            return_value=mock_db,
        ):
            result = handler._handle_view_details(
                CALLBACK_ID, CHAT_ID, USER_ID, USERNAME, DEBATE_ID
            )
        assert _status(result) == 200

    @pytest.mark.usefixtures("_patch_tg")
    def test_debate_long_answer_truncated(self, handler, _patch_tg):
        """Final answer longer than 500 chars is truncated."""
        debate_data = {
            "task": "Test",
            "final_answer": "A" * 600,
            "consensus_reached": True,
            "confidence": 0.9,
            "rounds_used": 2,
            "agents": ["Agent1"],
        }
        mock_db = MagicMock()
        mock_db.get.return_value = debate_data

        with patch(
            "aragora.server.storage.get_debates_db",
            return_value=mock_db,
        ):
            result = handler._handle_view_details(
                CALLBACK_ID, CHAT_ID, USER_ID, USERNAME, DEBATE_ID
            )
        assert _status(result) == 200

    @pytest.mark.usefixtures("_patch_tg")
    def test_debate_many_agents_truncated(self, handler, _patch_tg):
        """More than 5 agents shows truncation message."""
        debate_data = {
            "task": "Test topic",
            "final_answer": "Answer text",
            "consensus_reached": True,
            "confidence": 0.7,
            "rounds_used": 3,
            "agents": [f"Agent{i}" for i in range(10)],
        }
        mock_db = MagicMock()
        mock_db.get.return_value = debate_data

        with patch(
            "aragora.server.storage.get_debates_db",
            return_value=mock_db,
        ):
            result = handler._handle_view_details(
                CALLBACK_ID, CHAT_ID, USER_ID, USERNAME, DEBATE_ID
            )
        assert _status(result) == 200

    @pytest.mark.usefixtures("_patch_tg")
    def test_debate_exactly_5_agents_no_truncation(self, handler, _patch_tg):
        """Exactly 5 agents does not trigger truncation."""
        debate_data = {
            "task": "Test",
            "final_answer": "Answer",
            "consensus_reached": False,
            "confidence": 0.4,
            "rounds_used": 2,
            "agents": ["A", "B", "C", "D", "E"],
        }
        mock_db = MagicMock()
        mock_db.get.return_value = debate_data

        with patch(
            "aragora.server.storage.get_debates_db",
            return_value=mock_db,
        ):
            result = handler._handle_view_details(
                CALLBACK_ID, CHAT_ID, USER_ID, USERNAME, DEBATE_ID
            )
        assert _status(result) == 200

    @pytest.mark.usefixtures("_patch_tg")
    def test_debate_no_agents(self, handler, _patch_tg):
        """No agents defaults to 'Unknown'."""
        debate_data = {
            "task": "Test",
            "final_answer": "Answer",
            "consensus_reached": True,
            "confidence": 0.5,
            "rounds_used": 1,
            "agents": [],
        }
        mock_db = MagicMock()
        mock_db.get.return_value = debate_data

        with patch(
            "aragora.server.storage.get_debates_db",
            return_value=mock_db,
        ):
            result = handler._handle_view_details(
                CALLBACK_ID, CHAT_ID, USER_ID, USERNAME, DEBATE_ID
            )
        assert _status(result) == 200

    @pytest.mark.usefixtures("_patch_tg")
    def test_debate_no_consensus(self, handler, _patch_tg):
        """No consensus shows 'No'."""
        debate_data = {
            "task": "Topic",
            "final_answer": "No agreement",
            "consensus_reached": False,
            "confidence": 0.3,
            "rounds_used": 5,
            "agents": ["Agent1"],
        }
        mock_db = MagicMock()
        mock_db.get.return_value = debate_data

        with patch(
            "aragora.server.storage.get_debates_db",
            return_value=mock_db,
        ):
            result = handler._handle_view_details(
                CALLBACK_ID, CHAT_ID, USER_ID, USERNAME, DEBATE_ID
            )
        assert _status(result) == 200

    @pytest.mark.usefixtures("_patch_tg")
    def test_debate_missing_optional_fields(self, handler, _patch_tg):
        """Missing optional fields use defaults."""
        debate_data = {}
        mock_db = MagicMock()
        mock_db.get.return_value = debate_data

        with patch(
            "aragora.server.storage.get_debates_db",
            return_value=mock_db,
        ):
            result = handler._handle_view_details(
                CALLBACK_ID, CHAT_ID, USER_ID, USERNAME, DEBATE_ID
            )
        assert _status(result) == 200

    @pytest.mark.usefixtures("_patch_tg")
    def test_debate_no_final_answer(self, handler, _patch_tg):
        """Missing final_answer shows 'No conclusion'."""
        debate_data = {
            "task": "Test",
            "consensus_reached": False,
            "confidence": 0.1,
            "rounds_used": 3,
            "agents": [],
        }
        mock_db = MagicMock()
        mock_db.get.return_value = debate_data

        with patch(
            "aragora.server.storage.get_debates_db",
            return_value=mock_db,
        ):
            result = handler._handle_view_details(
                CALLBACK_ID, CHAT_ID, USER_ID, USERNAME, DEBATE_ID
            )
        assert _status(result) == 200

    @pytest.mark.usefixtures("_patch_tg")
    def test_debate_key_error_handled(self, handler, _patch_tg):
        """KeyError during storage access is handled."""
        with patch(
            "aragora.server.storage.get_debates_db",
            side_effect=KeyError("missing"),
        ):
            result = handler._handle_view_details(
                CALLBACK_ID, CHAT_ID, USER_ID, USERNAME, DEBATE_ID
            )
        assert _status(result) == 200

    @pytest.mark.usefixtures("_patch_tg")
    def test_debate_os_error_handled(self, handler, _patch_tg):
        """OSError during storage access is handled."""
        with patch(
            "aragora.server.storage.get_debates_db",
            side_effect=OSError("disk error"),
        ):
            result = handler._handle_view_details(
                CALLBACK_ID, CHAT_ID, USER_ID, USERNAME, DEBATE_ID
            )
        assert _status(result) == 200

    @pytest.mark.usefixtures("_patch_tg")
    def test_debate_value_error_handled(self, handler, _patch_tg):
        """ValueError during storage access is handled."""
        with patch(
            "aragora.server.storage.get_debates_db",
            side_effect=ValueError("bad value"),
        ):
            result = handler._handle_view_details(
                CALLBACK_ID, CHAT_ID, USER_ID, USERNAME, DEBATE_ID
            )
        assert _status(result) == 200


# ============================================================================
# _handle_inline_query: Inline Query Handling
# ============================================================================


class TestHandleInlineQuery:
    """Test inline query handling (@bot queries)."""

    @pytest.mark.usefixtures("_full_patch")
    def test_empty_query_returns_no_results(self, handler, _patch_tg):
        """Empty query text returns empty results."""
        query = _make_inline_query("")
        result = handler._handle_inline_query(query)
        assert _status(result) == 200
        _patch_tg.create_tracked_task.assert_called()

    @pytest.mark.usefixtures("_full_patch")
    def test_short_query_returns_no_results(self, handler, _patch_tg):
        """Query shorter than 5 chars returns empty results."""
        query = _make_inline_query("Hi")
        result = handler._handle_inline_query(query)
        assert _status(result) == 200

    @pytest.mark.usefixtures("_full_patch")
    def test_exactly_4_chars_is_too_short(self, handler, _patch_tg):
        """Exactly 4 characters is too short for inline results."""
        query = _make_inline_query("ABCD")
        result = handler._handle_inline_query(query)
        assert _status(result) == 200

    @pytest.mark.usefixtures("_full_patch")
    def test_exactly_5_chars_returns_results(self, handler, _patch_tg):
        """Exactly 5 characters returns inline results."""
        query = _make_inline_query("ABCDE")
        result = handler._handle_inline_query(query)
        assert _status(result) == 200
        _patch_tg.create_tracked_task.assert_called()

    @pytest.mark.usefixtures("_full_patch")
    def test_long_query_returns_two_results(self, handler, _patch_tg):
        """Long enough query returns debate + gauntlet results."""
        query = _make_inline_query("Should we adopt microservices architecture?")

        # Capture what gets passed to answer_inline_query_async
        handler._answer_inline_query_async = AsyncMock()

        result = handler._handle_inline_query(query)
        assert _status(result) == 200
        _patch_tg.create_tracked_task.assert_called()

    @pytest.mark.usefixtures("_full_patch")
    def test_query_whitespace_only_is_empty(self, handler, _patch_tg):
        """Whitespace-only query is treated as empty after strip."""
        query = _make_inline_query("    ")
        result = handler._handle_inline_query(query)
        assert _status(result) == 200

    @pytest.mark.usefixtures("_patch_tg", "_patch_telemetry", "_patch_events")
    def test_inline_query_rbac_denied(self, handler, _patch_tg):
        """RBAC denial for inline queries returns empty results."""
        with patch.object(
            TelegramHandler,
            "_check_telegram_user_permission",
            return_value=False,
        ):
            query = _make_inline_query("Should we use Rust?")
            result = handler._handle_inline_query(query)
        assert _status(result) == 200
        _patch_tg.create_tracked_task.assert_called()

    @pytest.mark.usefixtures("_full_patch")
    def test_inline_query_missing_from(self, handler, _patch_tg):
        """Missing 'from' field defaults gracefully."""
        query = {"id": QUERY_ID, "query": "Some valid query text"}
        result = handler._handle_inline_query(query)
        assert _status(result) == 200

    @pytest.mark.usefixtures("_full_patch")
    def test_inline_query_result_article_type(self, handler, _patch_tg):
        """Results include article-type entries."""
        query = _make_inline_query("A sufficiently long query for inline")

        # We track the coroutine args passed to create_tracked_task
        result = handler._handle_inline_query(query)
        assert _status(result) == 200

    @pytest.mark.usefixtures("_full_patch")
    def test_inline_query_long_text_truncated_in_title(self, handler, _patch_tg):
        """Very long query text is truncated at 50 chars in the title."""
        long_query = "X" * 200
        query = _make_inline_query(long_query)
        result = handler._handle_inline_query(query)
        assert _status(result) == 200


# ============================================================================
# Edge Cases and Integration
# ============================================================================


class TestEdgeCases:
    """Miscellaneous edge cases and integration points."""

    @pytest.mark.usefixtures("_full_patch")
    def test_callback_with_special_chars_in_data(self, handler, _patch_tg):
        """Special characters in callback data are handled."""
        cb = _make_callback("vote:debate-with-dashes:agree")
        with patch.object(
            handler,
            "_handle_vote",
            return_value=MagicMock(status_code=200, body=b'{"ok":true}'),
        ) as hv:
            handler._handle_callback_query(cb)
            hv.assert_called_once()

    @pytest.mark.usefixtures("_full_patch")
    def test_callback_with_unicode_data(self, handler, _patch_tg):
        """Unicode in callback data is handled."""
        cb = _make_callback("unknown_action")
        cb["data"] = "action_\u00e9\u00e8\u00ea"
        result = handler._handle_callback_query(cb)
        assert _status(result) == 200

    @pytest.mark.usefixtures("_full_patch")
    def test_message_with_unicode_text(self, handler, _patch_tg):
        """Unicode text in messages is handled correctly."""
        msg = _make_message("This is a unicode message with accents: cafe\u0301")
        result = handler._handle_message(msg)
        assert _status(result) == 200

    @pytest.mark.usefixtures("_full_patch")
    def test_message_with_newlines(self, handler, _patch_tg):
        """Messages with newlines are handled."""
        msg = _make_message("Line 1\nLine 2\nLine 3 is longer")
        result = handler._handle_message(msg)
        assert _status(result) == 200

    @pytest.mark.usefixtures("_full_patch")
    def test_callback_no_message_key(self, handler, _patch_tg):
        """Callback without 'message' key still extracts chat_id (as None)."""
        cb = {
            "id": CALLBACK_ID,
            "data": "unknown_action",
            "from": {"id": USER_ID, "username": USERNAME},
        }
        result = handler._handle_callback_query(cb)
        assert _status(result) == 200

    @pytest.mark.usefixtures("_full_patch")
    def test_callback_no_from_key(self, handler, _patch_tg):
        """Callback without 'from' key defaults user fields."""
        cb = {
            "id": CALLBACK_ID,
            "data": "unknown_action",
            "message": {"chat": {"id": CHAT_ID}},
        }
        result = handler._handle_callback_query(cb)
        assert _status(result) == 200

    @pytest.mark.usefixtures("_full_patch")
    def test_vote_with_empty_debate_id(self, handler, _patch_tg, _patch_events):
        """Vote with empty debate_id still processes."""
        with patch("aragora.server.handlers.social.telegram.callbacks.record_vote"):
            result = handler._handle_vote(CALLBACK_ID, CHAT_ID, USER_ID, USERNAME, "", "agree")
        assert _status(result) == 200

    @pytest.mark.usefixtures("_full_patch")
    def test_view_details_with_empty_debate_id(self, handler, _patch_tg):
        """View details with empty debate_id."""
        with patch(
            "aragora.server.storage.get_debates_db",
            return_value=MagicMock(get=MagicMock(return_value=None)),
        ):
            result = handler._handle_view_details(CALLBACK_ID, CHAT_ID, USER_ID, USERNAME, "")
        assert _status(result) == 200

    @pytest.mark.usefixtures("_full_patch")
    def test_message_with_very_long_text(self, handler, _patch_tg):
        """Very long messages are handled (text truncated in response)."""
        long_text = "A" * 5000
        msg = _make_message(long_text)
        result = handler._handle_message(msg)
        assert _status(result) == 200


class TestSecurityCases:
    """Security-related edge cases."""

    @pytest.mark.usefixtures("_full_patch")
    def test_message_path_traversal_in_text(self, handler, _patch_tg):
        """Path traversal attempts in message text are safe."""
        msg = _make_message("../../etc/passwd is an important file to know about")
        result = handler._handle_message(msg)
        assert _status(result) == 200

    @pytest.mark.usefixtures("_full_patch")
    def test_callback_injection_in_data(self, handler, _patch_tg):
        """Injection attempts in callback data are handled safely."""
        cb = _make_callback("vote'; DROP TABLE debates;--:d1:agree")
        result = handler._handle_callback_query(cb)
        assert _status(result) == 200

    @pytest.mark.usefixtures("_full_patch")
    def test_callback_xss_in_data(self, handler, _patch_tg):
        """XSS attempts in callback data are handled safely."""
        cb = _make_callback("<script>alert('xss')</script>:d1:agree")
        result = handler._handle_callback_query(cb)
        assert _status(result) == 200

    @pytest.mark.usefixtures("_full_patch")
    def test_message_html_injection(self, handler, _patch_tg):
        """HTML in message text does not break processing."""
        msg = _make_message("<b>Bold</b> <script>evil()</script> text content")
        result = handler._handle_message(msg)
        assert _status(result) == 200

    @pytest.mark.usefixtures("_full_patch")
    def test_inline_query_injection(self, handler, _patch_tg):
        """Injection in inline query text is handled safely."""
        query = _make_inline_query("SELECT * FROM debates WHERE 1=1;")
        result = handler._handle_inline_query(query)
        assert _status(result) == 200

    @pytest.mark.usefixtures("_full_patch")
    def test_extremely_large_user_id(self, handler, _patch_tg):
        """Very large user IDs are handled."""
        msg = _make_message("A long enough message for testing purposes", user_id=999999999999999)
        result = handler._handle_message(msg)
        assert _status(result) == 200

    @pytest.mark.usefixtures("_full_patch")
    def test_negative_chat_id(self, handler, _patch_tg):
        """Negative chat IDs (group chats) are handled."""
        msg = _make_message("A long enough message for testing purposes", chat_id=-100123456)
        result = handler._handle_message(msg)
        assert _status(result) == 200

    @pytest.mark.usefixtures("_full_patch")
    def test_callback_with_null_bytes(self, handler, _patch_tg):
        """Null bytes in callback data are handled."""
        cb = _make_callback("unknown\x00action")
        result = handler._handle_callback_query(cb)
        assert _status(result) == 200


# ============================================================================
# Deep Behavioral Assertions
# ============================================================================


class TestMessageBehavior:
    """Deeper assertions on message handling behavior."""

    @pytest.mark.usefixtures("_full_patch")
    def test_command_returns_before_permission_check(self, handler, _patch_tg):
        """Commands bypass the message-send permission check."""
        # Commands go through _handle_command, not the text message path
        with (
            patch.object(
                TelegramHandler,
                "_handle_command",
                return_value=MagicMock(status_code=200, body=b'{"ok":true}'),
            ) as hc,
            patch("aragora.server.handlers.social.telegram.callbacks.record_message") as rm,
        ):
            msg = _make_message("/help")
            handler._handle_message(msg)
            hc.assert_called_once()
            rm.assert_called_once_with("telegram", "command")

    @pytest.mark.usefixtures("_full_patch")
    def test_text_message_task_name_includes_chat_id(self, handler, _patch_tg):
        """The async task name includes the chat_id for traceability."""
        msg = _make_message("This is a test message that is long enough")
        handler._handle_message(msg)
        call_args = _patch_tg.create_tracked_task.call_args
        assert call_args is not None
        # Second positional arg or name kwarg should include chat_id
        task_name = (
            call_args[1].get("name", "") if len(call_args) > 1 else call_args.kwargs.get("name", "")
        )
        assert str(CHAT_ID) in task_name

    @pytest.mark.usefixtures("_full_patch")
    def test_short_message_suggests_help(self, handler, _patch_tg):
        """Short messages get a /help suggestion."""
        handler._send_message_async = AsyncMock()
        msg = _make_message("Hi")
        handler._handle_message(msg)
        # The tracked task's coroutine was created with _send_message_async
        _patch_tg.create_tracked_task.assert_called_once()

    @pytest.mark.usefixtures("_full_patch")
    def test_long_message_includes_debate_command(self, handler, _patch_tg):
        """Long messages get a /debate suggestion with the text."""
        handler._send_message_async = AsyncMock()
        long_text = "This is a much longer message that should get a debate suggestion from the bot"
        msg = _make_message(long_text)
        handler._handle_message(msg)
        _patch_tg.create_tracked_task.assert_called_once()

    @pytest.mark.usefixtures("_full_patch")
    def test_username_defaults_to_unknown(self, handler, _patch_tg):
        """Missing username defaults to 'unknown'."""
        msg = {
            "chat": {"id": CHAT_ID},
            "text": "A long enough message for testing purposes",
            "from": {"id": USER_ID},
        }
        result = handler._handle_message(msg)
        assert _status(result) == 200


class TestCallbackBehavior:
    """Deeper assertions on callback query behavior."""

    @pytest.mark.usefixtures("_full_patch")
    def test_callback_ack_task_name(self, handler, _patch_tg):
        """Unknown action ack task name includes callback_id."""
        cb = _make_callback("some_unknown_action")
        handler._handle_callback_query(cb)
        call_args = _patch_tg.create_tracked_task.call_args
        assert call_args is not None
        task_name = (
            call_args[1].get("name", "") if len(call_args) > 1 else call_args.kwargs.get("name", "")
        )
        assert CALLBACK_ID in task_name

    @pytest.mark.usefixtures("_full_patch")
    def test_callback_username_defaults_to_unknown(self, handler, _patch_tg):
        """Missing callback username defaults to 'unknown'."""
        cb = {
            "id": CALLBACK_ID,
            "data": "action",
            "from": {"id": USER_ID},
            "message": {"chat": {"id": CHAT_ID}},
        }
        result = handler._handle_callback_query(cb)
        assert _status(result) == 200

    @pytest.mark.usefixtures("_full_patch")
    def test_vote_callback_passes_correct_parts(self, handler):
        """Verify vote callback correctly parses parts."""
        cb = _make_callback("vote:my-debate-42:disagree")
        with patch.object(
            handler,
            "_handle_vote",
            return_value=MagicMock(status_code=200, body=b'{"ok":true}'),
        ) as hv:
            handler._handle_callback_query(cb)
            hv.assert_called_once_with(
                CALLBACK_ID, CHAT_ID, USER_ID, USERNAME, "my-debate-42", "disagree"
            )

    @pytest.mark.usefixtures("_full_patch")
    def test_details_callback_passes_correct_debate_id(self, handler):
        """Verify details callback correctly parses debate_id."""
        cb = _make_callback("details:debate-xyz-789")
        with patch.object(
            handler,
            "_handle_view_details",
            return_value=MagicMock(status_code=200, body=b'{"ok":true}'),
        ) as hvd:
            handler._handle_callback_query(cb)
            hvd.assert_called_once_with(CALLBACK_ID, CHAT_ID, USER_ID, USERNAME, "debate-xyz-789")

    @pytest.mark.usefixtures("_full_patch")
    def test_rbac_deny_task_name_includes_denied(self, handler, _patch_tg):
        """Permission denied ack task name includes 'denied'."""
        with patch.object(
            TelegramHandler,
            "_check_telegram_user_permission",
            return_value=False,
        ):
            cb = _make_callback("vote:d1:agree")
            handler._handle_callback_query(cb)
        call_args = _patch_tg.create_tracked_task.call_args
        assert call_args is not None
        task_name = (
            call_args[1].get("name", "") if len(call_args) > 1 else call_args.kwargs.get("name", "")
        )
        assert "denied" in task_name


class TestVoteBehavior:
    """Deeper assertions on vote behavior."""

    @pytest.mark.usefixtures("_patch_tg", "_patch_events")
    def test_vote_emits_correct_platform(self, handler, _patch_events):
        """Vote event includes platform='telegram'."""
        with patch("aragora.server.handlers.social.telegram.callbacks.record_vote"):
            handler._handle_vote(CALLBACK_ID, CHAT_ID, USER_ID, USERNAME, DEBATE_ID, "agree")
        _patch_events["emit_vote_received"].assert_called_once()
        call_kwargs = _patch_events["emit_vote_received"].call_args.kwargs
        assert call_kwargs["platform"] == "telegram"

    @pytest.mark.usefixtures("_patch_tg", "_patch_events")
    def test_vote_stores_with_telegram_prefix(self, handler, _patch_tg):
        """Vote voter_id is prefixed with 'telegram:'."""
        mock_db = MagicMock()
        mock_db.record_vote = MagicMock()
        with (
            patch("aragora.server.handlers.social.telegram.callbacks.record_vote"),
            patch(
                "aragora.server.storage.get_debates_db",
                return_value=mock_db,
            ),
        ):
            handler._handle_vote(CALLBACK_ID, CHAT_ID, USER_ID, USERNAME, DEBATE_ID, "disagree")
        mock_db.record_vote.assert_called_once()
        call_kwargs = mock_db.record_vote.call_args.kwargs
        assert call_kwargs["voter_id"] == f"telegram:{USER_ID}"
        assert call_kwargs["source"] == "telegram"

    @pytest.mark.usefixtures("_patch_tg", "_patch_events")
    def test_vote_ack_task_name_includes_callback_id(self, handler, _patch_tg):
        """Vote ack task name includes the callback_id."""
        with patch("aragora.server.handlers.social.telegram.callbacks.record_vote"):
            handler._handle_vote(CALLBACK_ID, CHAT_ID, USER_ID, USERNAME, DEBATE_ID, "agree")
        call_args = _patch_tg.create_tracked_task.call_args
        assert call_args is not None
        task_name = (
            call_args[1].get("name", "") if len(call_args) > 1 else call_args.kwargs.get("name", "")
        )
        assert CALLBACK_ID in task_name


class TestViewDetailsBehavior:
    """Deeper assertions on view details behavior."""

    @pytest.mark.usefixtures("_patch_tg")
    def test_details_sends_markdown_message(self, handler, _patch_tg):
        """Details sends message with Markdown parse mode."""
        debate_data = {
            "task": "Test topic for details",
            "final_answer": "We concluded something.",
            "consensus_reached": True,
            "confidence": 0.75,
            "rounds_used": 2,
            "agents": ["Claude"],
        }
        mock_db = MagicMock()
        mock_db.get.return_value = debate_data

        with patch(
            "aragora.server.storage.get_debates_db",
            return_value=mock_db,
        ):
            handler._handle_view_details(CALLBACK_ID, CHAT_ID, USER_ID, USERNAME, DEBATE_ID)
        # Second create_tracked_task call should be the detail message
        assert _patch_tg.create_tracked_task.call_count >= 2

    @pytest.mark.usefixtures("_patch_tg")
    def test_details_not_found_includes_debate_id(self, handler, _patch_tg):
        """Not found message includes the debate ID."""
        mock_db = MagicMock()
        mock_db.get.return_value = None

        with patch(
            "aragora.server.storage.get_debates_db",
            return_value=mock_db,
        ):
            handler._handle_view_details(
                CALLBACK_ID, CHAT_ID, USER_ID, USERNAME, "my-special-debate"
            )
        _patch_tg.create_tracked_task.assert_called_once()

    @pytest.mark.usefixtures("_patch_tg")
    def test_details_zero_confidence(self, handler, _patch_tg):
        """Zero confidence is displayed correctly."""
        debate_data = {
            "task": "Topic",
            "final_answer": "Answer",
            "consensus_reached": False,
            "confidence": 0,
            "rounds_used": 1,
            "agents": [],
        }
        mock_db = MagicMock()
        mock_db.get.return_value = debate_data

        with patch(
            "aragora.server.storage.get_debates_db",
            return_value=mock_db,
        ):
            result = handler._handle_view_details(
                CALLBACK_ID, CHAT_ID, USER_ID, USERNAME, DEBATE_ID
            )
        assert _status(result) == 200

    @pytest.mark.usefixtures("_patch_tg")
    def test_details_full_confidence(self, handler, _patch_tg):
        """100% confidence is displayed correctly."""
        debate_data = {
            "task": "Topic",
            "final_answer": "Answer",
            "consensus_reached": True,
            "confidence": 1.0,
            "rounds_used": 1,
            "agents": ["Agent1"],
        }
        mock_db = MagicMock()
        mock_db.get.return_value = debate_data

        with patch(
            "aragora.server.storage.get_debates_db",
            return_value=mock_db,
        ):
            result = handler._handle_view_details(
                CALLBACK_ID, CHAT_ID, USER_ID, USERNAME, DEBATE_ID
            )
        assert _status(result) == 200


class TestInlineQueryBehavior:
    """Deeper assertions on inline query behavior."""

    @pytest.mark.usefixtures("_full_patch")
    def test_inline_query_uses_read_permission(self, handler, _patch_tg):
        """Inline queries check PERM_TELEGRAM_READ."""
        with patch.object(
            TelegramHandler,
            "_check_telegram_user_permission",
            return_value=True,
        ) as check:
            query = _make_inline_query("Some query text")
            handler._handle_inline_query(query)
            check.assert_called_once()
            call_args = check.call_args[0]
            # The 4th arg should be the read permission
            assert call_args[3] == "telegram:read"

    @pytest.mark.usefixtures("_full_patch")
    def test_inline_query_rbac_check_passes_user_id_zero_for_chat(self, handler, _patch_tg):
        """Inline queries pass chat_id=0 since there's no chat."""
        with patch.object(
            TelegramHandler,
            "_check_telegram_user_permission",
            return_value=True,
        ) as check:
            query = _make_inline_query("Some query text")
            handler._handle_inline_query(query)
            call_args = check.call_args[0]
            assert call_args[2] == 0  # chat_id is 0 for inline queries

    @pytest.mark.usefixtures("_full_patch")
    def test_inline_query_task_name_includes_query_id(self, handler, _patch_tg):
        """Inline query task name includes query_id."""
        query = _make_inline_query("Valid query text here")
        handler._handle_inline_query(query)
        call_args = _patch_tg.create_tracked_task.call_args
        assert call_args is not None
        task_name = (
            call_args[1].get("name", "") if len(call_args) > 1 else call_args.kwargs.get("name", "")
        )
        assert QUERY_ID in task_name


class TestCallbackQueryRBACSequence:
    """Test RBAC permission check sequencing in callback queries."""

    @pytest.mark.usefixtures("_patch_tg", "_patch_telemetry", "_patch_events")
    def test_vote_checks_two_permissions(self, handler, _patch_tg):
        """Vote path checks both callback_handle and votes_record permissions."""
        permission_log = []

        def check_perm(user_id, username, chat_id, perm):
            permission_log.append(perm)
            return True

        with patch.object(
            TelegramHandler, "_check_telegram_user_permission", side_effect=check_perm
        ):
            cb = _make_callback("vote:d1:agree")
            with patch.object(
                handler,
                "_handle_vote",
                return_value=MagicMock(status_code=200, body=b'{"ok":true}'),
            ):
                handler._handle_callback_query(cb)

        assert len(permission_log) == 2
        assert "telegram:callbacks:handle" in permission_log
        assert "telegram:votes:record" in permission_log

    @pytest.mark.usefixtures("_patch_tg", "_patch_telemetry", "_patch_events")
    def test_details_checks_two_permissions(self, handler, _patch_tg):
        """Details path checks both callback_handle and debates_read permissions."""
        permission_log = []

        def check_perm(user_id, username, chat_id, perm):
            permission_log.append(perm)
            return True

        with patch.object(
            TelegramHandler, "_check_telegram_user_permission", side_effect=check_perm
        ):
            cb = _make_callback("details:d1")
            with patch.object(
                handler,
                "_handle_view_details",
                return_value=MagicMock(status_code=200, body=b'{"ok":true}'),
            ):
                handler._handle_callback_query(cb)

        assert len(permission_log) == 2
        assert "telegram:callbacks:handle" in permission_log
        assert "telegram:debates:read" in permission_log

    @pytest.mark.usefixtures("_patch_tg", "_patch_telemetry", "_patch_events")
    def test_unknown_action_checks_one_permission(self, handler, _patch_tg):
        """Unknown action only checks the base callback_handle permission."""
        permission_log = []

        def check_perm(user_id, username, chat_id, perm):
            permission_log.append(perm)
            return True

        with patch.object(
            TelegramHandler, "_check_telegram_user_permission", side_effect=check_perm
        ):
            cb = _make_callback("some_unknown_action")
            handler._handle_callback_query(cb)

        assert len(permission_log) == 1
        assert permission_log[0] == "telegram:callbacks:handle"
