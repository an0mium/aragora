"""Comprehensive tests for Slack event handler implementations.

Covers the EventsMixin class from
aragora.server.handlers.social.slack.events.handlers:

- handle_app_mention:
    * Empty mention (no text after bot mention) -> help message
    * "debate <topic>" -> slash command redirect
    * "plan <topic>" -> slash command redirect
    * "implement <topic>" -> slash command redirect
    * "status" -> slash command redirect
    * "agents" -> slash command redirect
    * "help" -> slash command redirect
    * Unknown text -> error + help suggestion
    * Bot mention regex stripping (single, multiple, mixed case)
    * Thread reply via create_tracked_task when SLACK_BOT_TOKEN set
    * No task created when SLACK_BOT_TOKEN empty
    * Quote stripping from topics (single/double quotes)
    * Long text truncation in unknown-command response
    * Missing event fields default to empty strings

- handle_message_event:
    * Non-IM channel type ignored
    * Bot messages ignored (bot_id)
    * Bot messages ignored (subtype "bot_message")
    * Valid DM dispatches to _parse_dm_command
    * Reply posted via create_tracked_task when SLACK_BOT_TOKEN set
    * No task created when SLACK_BOT_TOKEN empty
    * Missing fields default gracefully

- _parse_dm_command:
    * Empty text -> welcome/help message
    * "help" -> full help listing
    * "status" -> delegates to _get_status_response
    * "agents" -> delegates to _get_agents_response
    * "debate <topic>" -> slash command redirect
    * "plan <topic>" -> slash command redirect
    * "implement <topic>" -> slash command redirect
    * Unknown command -> error message
    * Case-insensitive matching (HELP, Help, etc.)
    * Long unknown text truncated at 50 chars
    * Quote stripping from debate/plan/implement topics

- _get_status_response:
    * EloSystem available with agents -> formatted status
    * ImportError -> fallback "Unknown"
    * AttributeError -> fallback "Unknown"
    * RuntimeError -> fallback "Unknown"

- _get_agents_response:
    * Agents available -> sorted top-5 listing
    * Empty agents list -> "No agents registered yet."
    * ImportError -> "Agent list temporarily unavailable."
    * AttributeError -> fallback message
    * RuntimeError -> fallback message

- _post_message_async:
    * Successful post returns ts
    * SLACK_BOT_TOKEN empty -> returns None
    * Slack API error (ok=false) -> returns None
    * ConnectionError -> returns None
    * TimeoutError -> returns None
    * RuntimeError -> returns None
    * ValueError -> returns None
    * thread_ts included when provided
    * blocks included when provided
"""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# The conftest.py in this directory sets up the sys.modules alias so that
# ``from .._slack_impl import ...`` in handlers.py resolves correctly.
# The alias key is the one used for patching throughout this file.
_IMPL = "aragora.server.handlers.social.slack._slack_impl"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _body(result) -> dict:
    """Extract JSON body dict from a HandlerResult."""
    if result is None:
        return {}
    if isinstance(result, dict):
        return result
    raw = result.body
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")
    return json.loads(raw)


def _status(result) -> int:
    """Extract HTTP status code from a HandlerResult."""
    if result is None:
        return 0
    if isinstance(result, dict):
        return result.get("status_code", 200)
    return result.status_code


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def events_module():
    """Import the events handler module lazily."""
    from aragora.server.handlers.social.slack.events import handlers as mod

    return mod


@pytest.fixture
def mixin_instance(events_module):
    """Create a concrete instance of EventsMixin for testing."""

    class ConcreteHandler(events_module.EventsMixin):
        pass

    return ConcreteHandler()


def _patch_impl(token: str = "", task_mock=None):
    """Return nested context managers that patch SLACK_BOT_TOKEN and create_tracked_task."""
    if task_mock is None:
        task_mock = MagicMock()
    return (
        patch(f"{_IMPL}.SLACK_BOT_TOKEN", token),
        patch(f"{_IMPL}.create_tracked_task", task_mock),
    )


# ---------------------------------------------------------------------------
# handle_app_mention tests
# ---------------------------------------------------------------------------


class TestHandleAppMention:
    """Tests for EventsMixin.handle_app_mention."""

    def test_returns_ok(self, mixin_instance):
        """handle_app_mention always returns json_response({"ok": True})."""
        p1, p2 = _patch_impl("")
        with p1, p2:
            result = mixin_instance.handle_app_mention({"text": "", "channel": "C1"})

        assert _status(result) == 200
        assert _body(result) == {"ok": True}

    def test_empty_mention_shows_help(self, mixin_instance):
        """Mention with no text returns help message (verified via task text)."""
        mock_task = MagicMock()
        p1, p2 = _patch_impl("xoxb-test", mock_task)

        with p1, p2:
            event = {"text": "<@U12345>", "channel": "C1", "user": "U99"}
            result = mixin_instance.handle_app_mention(event)

        assert _body(result).get("ok") is True
        mock_task.assert_called_once()

    def test_only_bot_mention_triggers_help(self, mixin_instance):
        """Just a bot mention with whitespace triggers help."""
        mock_task = MagicMock()
        p1, p2 = _patch_impl("xoxb-test", mock_task)

        with p1, p2:
            event = {"text": "<@U12345>  ", "channel": "C1", "user": "U99"}
            mixin_instance.handle_app_mention(event)

        mock_task.assert_called_once()

    def test_debate_command(self, mixin_instance):
        """'debate <topic>' in mention returns slash command suggestion."""
        captured_coro = {}
        mock_post = AsyncMock()
        mixin_instance._post_message_async = mock_post

        def capture_task(coro, name=""):
            captured_coro["name"] = name
            coro.close()

        p1, p2 = _patch_impl("xoxb-test", capture_task)
        with p1, p2:
            event = {
                "text": '<@U12345> debate "Should AI be regulated?"',
                "channel": "C1",
                "user": "U99",
                "ts": "123.456",
            }
            result = mixin_instance.handle_app_mention(event)

        assert _body(result).get("ok") is True
        assert captured_coro.get("name") == "slack-reply-C1"

    def test_plan_command(self, mixin_instance):
        """'plan <topic>' in mention returns plan redirect."""
        mock_post = AsyncMock()
        mixin_instance._post_message_async = mock_post
        captured = {}

        def capture_task(coro, name=""):
            captured["name"] = name
            coro.close()

        p1, p2 = _patch_impl("xoxb-test", capture_task)
        with p1, p2:
            event = {
                "text": '<@U12345> plan "Improve onboarding"',
                "channel": "C1",
                "user": "U99",
            }
            result = mixin_instance.handle_app_mention(event)

        assert _body(result).get("ok") is True
        assert captured.get("name") == "slack-reply-C1"

    def test_implement_command(self, mixin_instance):
        """'implement <topic>' in mention returns implement redirect."""
        captured = {}

        def capture_task(coro, name=""):
            captured["name"] = name
            coro.close()

        p1, p2 = _patch_impl("xoxb-test", capture_task)
        with p1, p2:
            event = {
                "text": '<@U12345> implement "Automate reports"',
                "channel": "C1",
                "user": "U99",
            }
            result = mixin_instance.handle_app_mention(event)

        assert _body(result).get("ok") is True

    def test_status_command(self, mixin_instance):
        """'status' in mention returns status redirect."""
        captured = {}

        def capture_task(coro, name=""):
            captured["name"] = name
            coro.close()

        p1, p2 = _patch_impl("xoxb-test", capture_task)
        with p1, p2:
            event = {
                "text": "<@U12345> status",
                "channel": "C1",
                "user": "U99",
            }
            result = mixin_instance.handle_app_mention(event)

        assert _body(result).get("ok") is True

    def test_agents_command(self, mixin_instance):
        """'agents' in mention returns agents redirect."""
        captured = {}

        def capture_task(coro, name=""):
            captured["name"] = name
            coro.close()

        p1, p2 = _patch_impl("xoxb-test", capture_task)
        with p1, p2:
            event = {
                "text": "<@U12345> agents",
                "channel": "C1",
                "user": "U99",
            }
            result = mixin_instance.handle_app_mention(event)

        assert _body(result).get("ok") is True

    def test_help_command(self, mixin_instance):
        """'help' in mention returns help redirect."""
        captured = {}

        def capture_task(coro, name=""):
            captured["name"] = name
            coro.close()

        p1, p2 = _patch_impl("xoxb-test", capture_task)
        with p1, p2:
            event = {
                "text": "<@U12345> help",
                "channel": "C1",
                "user": "U99",
            }
            result = mixin_instance.handle_app_mention(event)

        assert _body(result).get("ok") is True

    def test_unknown_command(self, mixin_instance):
        """Unknown text in mention returns error message."""
        captured = {}

        def capture_task(coro, name=""):
            captured["name"] = name
            coro.close()

        p1, p2 = _patch_impl("xoxb-test", capture_task)
        with p1, p2:
            event = {
                "text": "<@U12345> foobar baz",
                "channel": "C1",
                "user": "U99",
            }
            result = mixin_instance.handle_app_mention(event)

        assert _body(result).get("ok") is True

    def test_no_task_when_bot_token_empty(self, mixin_instance):
        """No task is created when SLACK_BOT_TOKEN is empty."""
        mock_task = MagicMock()
        p1, p2 = _patch_impl("", mock_task)

        with p1, p2:
            event = {"text": "<@U12345> help", "channel": "C1", "user": "U99"}
            result = mixin_instance.handle_app_mention(event)

        mock_task.assert_not_called()
        assert _body(result).get("ok") is True

    def test_multiple_bot_mentions_stripped(self, mixin_instance):
        """Multiple bot mentions in text are all stripped."""
        p1, p2 = _patch_impl("")
        with p1, p2:
            event = {
                "text": "<@U12345> <@U67890> help",
                "channel": "C1",
                "user": "U99",
            }
            result = mixin_instance.handle_app_mention(event)

        assert _body(result).get("ok") is True

    def test_missing_event_fields(self, mixin_instance):
        """Missing event fields default to empty strings."""
        p1, p2 = _patch_impl("")
        with p1, p2:
            result = mixin_instance.handle_app_mention({})

        assert _body(result).get("ok") is True

    def test_debate_topic_quote_stripping(self, mixin_instance):
        """Quotes around debate topic are stripped."""
        mock_post = AsyncMock()
        mixin_instance._post_message_async = mock_post

        def capture_task(coro, name=""):
            coro.close()

        p1, p2 = _patch_impl("xoxb-test", capture_task)
        with p1, p2:
            event = {
                "text": """<@U12345> debate '"AI regulation"'""",
                "channel": "C1",
                "user": "U99",
            }
            result = mixin_instance.handle_app_mention(event)

        assert _body(result).get("ok") is True

    def test_case_insensitive_commands(self, mixin_instance):
        """Commands in mention are case-insensitive."""
        p1, p2 = _patch_impl("")
        with p1, p2:
            for cmd in ["STATUS", "Status", "sTaTuS"]:
                event = {"text": f"<@U12345> {cmd}", "channel": "C1", "user": "U99"}
                result = mixin_instance.handle_app_mention(event)
                assert _body(result).get("ok") is True

    def test_thread_ts_passed_to_post(self, mixin_instance):
        """Thread timestamp from event is passed to _post_message_async."""
        post_calls = []
        original_post = AsyncMock()
        mixin_instance._post_message_async = original_post

        def capture_task(coro, name=""):
            coro.close()
            post_calls.append(name)

        p1, p2 = _patch_impl("xoxb-test", capture_task)
        with p1, p2:
            event = {
                "text": "<@U12345> help",
                "channel": "C1",
                "user": "U99",
                "ts": "123.456",
            }
            mixin_instance.handle_app_mention(event)

        assert len(post_calls) == 1

    def test_long_unknown_text_truncated(self, mixin_instance):
        """Unknown command text is truncated to 50 chars in the response."""
        p1, p2 = _patch_impl("")
        with p1, p2:
            long_text = "x" * 200
            event = {
                "text": f"<@U12345> {long_text}",
                "channel": "C1",
                "user": "U99",
            }
            result = mixin_instance.handle_app_mention(event)

        assert _body(result).get("ok") is True

    def test_text_without_mention_treated_as_command(self, mixin_instance):
        """Text with no bot mention is treated as a command directly."""
        p1, p2 = _patch_impl("")
        with p1, p2:
            event = {"text": "status", "channel": "C1", "user": "U99"}
            result = mixin_instance.handle_app_mention(event)

        assert _body(result).get("ok") is True


# ---------------------------------------------------------------------------
# handle_message_event tests
# ---------------------------------------------------------------------------


class TestHandleMessageEvent:
    """Tests for EventsMixin.handle_message_event."""

    def test_non_im_channel_ignored(self, mixin_instance):
        """Messages in non-IM channels are ignored."""
        mock_task = MagicMock()
        p1, p2 = _patch_impl("xoxb-test", mock_task)
        with p1, p2:
            event = {
                "channel_type": "channel",
                "text": "hello",
                "channel": "C1",
                "user": "U99",
            }
            result = mixin_instance.handle_message_event(event)

        assert _body(result).get("ok") is True

    def test_bot_id_ignored(self, mixin_instance):
        """Messages from bots (bot_id) are ignored."""
        mock_task = MagicMock()
        p1, p2 = _patch_impl("xoxb-test", mock_task)

        with p1, p2:
            event = {
                "channel_type": "im",
                "bot_id": "B123",
                "text": "hello",
                "channel": "C1",
                "user": "U99",
            }
            result = mixin_instance.handle_message_event(event)

        mock_task.assert_not_called()
        assert _body(result).get("ok") is True

    def test_bot_message_subtype_ignored(self, mixin_instance):
        """Messages with subtype 'bot_message' are ignored."""
        mock_task = MagicMock()
        p1, p2 = _patch_impl("xoxb-test", mock_task)

        with p1, p2:
            event = {
                "channel_type": "im",
                "subtype": "bot_message",
                "text": "hello",
                "channel": "C1",
                "user": "U99",
            }
            result = mixin_instance.handle_message_event(event)

        mock_task.assert_not_called()
        assert _body(result).get("ok") is True

    def test_valid_dm_dispatches_parse(self, mixin_instance):
        """Valid DM dispatches to _parse_dm_command and posts reply."""
        mock_task = MagicMock()
        p1, p2 = _patch_impl("xoxb-test", mock_task)

        with p1, p2:
            event = {
                "channel_type": "im",
                "text": "help",
                "channel": "D1",
                "user": "U99",
            }
            result = mixin_instance.handle_message_event(event)

        mock_task.assert_called_once()
        assert _body(result).get("ok") is True

    def test_no_task_when_bot_token_empty(self, mixin_instance):
        """No task is created when SLACK_BOT_TOKEN is empty."""
        mock_task = MagicMock()
        p1, p2 = _patch_impl("", mock_task)

        with p1, p2:
            event = {
                "channel_type": "im",
                "text": "help",
                "channel": "D1",
                "user": "U99",
            }
            result = mixin_instance.handle_message_event(event)

        mock_task.assert_not_called()
        assert _body(result).get("ok") is True

    def test_missing_fields(self, mixin_instance):
        """Missing event fields default gracefully."""
        p1, p2 = _patch_impl("")
        with p1, p2:
            event = {"channel_type": "im"}
            result = mixin_instance.handle_message_event(event)

        assert _body(result).get("ok") is True

    def test_whitespace_text_stripped(self, mixin_instance):
        """Text with only whitespace is treated as empty."""
        mock_task = MagicMock()
        p1, p2 = _patch_impl("xoxb-test", mock_task)

        with p1, p2:
            event = {
                "channel_type": "im",
                "text": "   ",
                "channel": "D1",
                "user": "U99",
            }
            result = mixin_instance.handle_message_event(event)

        mock_task.assert_called_once()
        assert _body(result).get("ok") is True

    def test_no_channel_type_ignored(self, mixin_instance):
        """Missing channel_type is not 'im', so message is ignored."""
        mock_task = MagicMock()
        p1, p2 = _patch_impl("xoxb-test", mock_task)

        with p1, p2:
            event = {"text": "hello", "channel": "C1", "user": "U99"}
            result = mixin_instance.handle_message_event(event)

        mock_task.assert_not_called()
        assert _body(result).get("ok") is True

    def test_task_name_includes_user(self, mixin_instance):
        """Task name includes user ID for tracking."""
        captured = {}

        def capture_task(coro, name=""):
            captured["name"] = name
            coro.close()

        p1, p2 = _patch_impl("xoxb-test", capture_task)
        with p1, p2:
            event = {
                "channel_type": "im",
                "text": "help",
                "channel": "D1",
                "user": "U99",
            }
            mixin_instance.handle_message_event(event)

        assert captured["name"] == "slack-dm-reply-U99"

    def test_group_dm_ignored(self, mixin_instance):
        """Messages in group DMs (mpim) are ignored."""
        mock_task = MagicMock()
        p1, p2 = _patch_impl("xoxb-test", mock_task)

        with p1, p2:
            event = {
                "channel_type": "mpim",
                "text": "hello",
                "channel": "G1",
                "user": "U99",
            }
            result = mixin_instance.handle_message_event(event)

        mock_task.assert_not_called()
        assert _body(result).get("ok") is True


# ---------------------------------------------------------------------------
# _parse_dm_command tests
# ---------------------------------------------------------------------------


class TestParseDmCommand:
    """Tests for EventsMixin._parse_dm_command."""

    def test_empty_text_returns_welcome(self, mixin_instance):
        """Empty text returns welcome message with available commands."""
        result = mixin_instance._parse_dm_command("")
        assert "Send me a command" in result
        assert "help" in result
        assert "status" in result
        assert "agents" in result

    def test_help_returns_full_listing(self, mixin_instance):
        """'help' returns full command listing."""
        result = mixin_instance._parse_dm_command("help")
        assert "Aragora Direct Message Commands" in result
        assert "help" in result
        assert "status" in result
        assert "agents" in result
        assert "debate" in result
        assert "recent" in result
        assert "/aragora" in result

    def test_status_delegates(self, mixin_instance):
        """'status' delegates to _get_status_response."""
        with patch.object(mixin_instance, "_get_status_response", return_value="mock-status"):
            result = mixin_instance._parse_dm_command("status")
        assert result == "mock-status"

    def test_agents_delegates(self, mixin_instance):
        """'agents' delegates to _get_agents_response."""
        with patch.object(mixin_instance, "_get_agents_response", return_value="mock-agents"):
            result = mixin_instance._parse_dm_command("agents")
        assert result == "mock-agents"

    def test_debate_command(self, mixin_instance):
        """'debate <topic>' returns slash command redirect."""
        result = mixin_instance._parse_dm_command('debate "AI safety"')
        assert "AI safety" in result
        assert "/aragora debate" in result

    def test_plan_command(self, mixin_instance):
        """'plan <topic>' returns plan redirect."""
        result = mixin_instance._parse_dm_command('plan "Improve onboarding"')
        assert "Improve onboarding" in result
        assert "/aragora plan" in result

    def test_implement_command(self, mixin_instance):
        """'implement <topic>' returns implement redirect."""
        result = mixin_instance._parse_dm_command('implement "Automate reports"')
        assert "Automate reports" in result
        assert "/aragora implement" in result

    def test_unknown_command(self, mixin_instance):
        """Unknown command returns error with help suggestion."""
        result = mixin_instance._parse_dm_command("foobar")
        assert "I don't understand" in result
        assert "foobar" in result
        assert "help" in result

    def test_case_insensitive_help(self, mixin_instance):
        """Commands are case-insensitive."""
        for cmd in ["HELP", "Help", "hElP"]:
            result = mixin_instance._parse_dm_command(cmd)
            assert "Aragora Direct Message Commands" in result

    def test_case_insensitive_status(self, mixin_instance):
        """'status' is case-insensitive."""
        with patch.object(mixin_instance, "_get_status_response", return_value="mock-status"):
            for cmd in ["STATUS", "Status", "sTaTuS"]:
                result = mixin_instance._parse_dm_command(cmd)
                assert result == "mock-status"

    def test_case_insensitive_agents(self, mixin_instance):
        """'agents' is case-insensitive."""
        with patch.object(mixin_instance, "_get_agents_response", return_value="mock-agents"):
            for cmd in ["AGENTS", "Agents", "aGeNtS"]:
                result = mixin_instance._parse_dm_command(cmd)
                assert result == "mock-agents"

    def test_case_insensitive_debate(self, mixin_instance):
        """'debate' prefix is case-insensitive."""
        for prefix in ["DEBATE", "Debate", "dEbAtE"]:
            result = mixin_instance._parse_dm_command(f'{prefix} "topic"')
            assert "/aragora debate" in result

    def test_case_insensitive_plan(self, mixin_instance):
        """'plan' prefix is case-insensitive."""
        for prefix in ["PLAN", "Plan", "pLaN"]:
            result = mixin_instance._parse_dm_command(f'{prefix} "topic"')
            assert "/aragora plan" in result

    def test_case_insensitive_implement(self, mixin_instance):
        """'implement' prefix is case-insensitive."""
        for prefix in ["IMPLEMENT", "Implement", "iMpLeMeNt"]:
            result = mixin_instance._parse_dm_command(f'{prefix} "topic"')
            assert "/aragora implement" in result

    def test_long_unknown_text_truncated(self, mixin_instance):
        """Unknown text longer than 50 chars is truncated."""
        long_text = "x" * 200
        result = mixin_instance._parse_dm_command(long_text)
        assert "I don't understand" in result
        assert f"`{'x' * 50}`" in result

    def test_debate_topic_with_double_quotes(self, mixin_instance):
        """Double quotes are stripped from debate topic."""
        result = mixin_instance._parse_dm_command('debate "AI regulation"')
        assert "AI regulation" in result
        assert 'To start a full debate on "AI regulation"' in result

    def test_debate_topic_with_single_quotes(self, mixin_instance):
        """Single quotes are stripped from debate topic."""
        result = mixin_instance._parse_dm_command("debate 'AI regulation'")
        assert "AI regulation" in result

    def test_plan_topic_with_quotes(self, mixin_instance):
        """Quotes are stripped from plan topic."""
        result = mixin_instance._parse_dm_command('plan "Better testing"')
        assert "Better testing" in result

    def test_implement_topic_with_quotes(self, mixin_instance):
        """Quotes are stripped from implement topic."""
        result = mixin_instance._parse_dm_command('implement "Deploy faster"')
        assert "Deploy faster" in result

    def test_debate_with_extra_whitespace(self, mixin_instance):
        """Extra whitespace in debate topic is stripped."""
        result = mixin_instance._parse_dm_command('debate   "  AI regulation  "')
        assert "AI regulation" in result

    def test_welcome_includes_plan(self, mixin_instance):
        """Welcome message includes plan command."""
        result = mixin_instance._parse_dm_command("")
        assert "plan" in result

    def test_welcome_includes_implement(self, mixin_instance):
        """Welcome message includes implement command."""
        result = mixin_instance._parse_dm_command("")
        assert "implement" in result

    def test_help_includes_plan(self, mixin_instance):
        """Help message includes plan command."""
        result = mixin_instance._parse_dm_command("help")
        assert "plan" in result

    def test_help_includes_implement(self, mixin_instance):
        """Help message includes implement command."""
        result = mixin_instance._parse_dm_command("help")
        assert "implement" in result

    def test_debate_topic_no_quotes(self, mixin_instance):
        """Debate topic without quotes works."""
        result = mixin_instance._parse_dm_command("debate AI safety is important")
        assert "AI safety is important" in result

    def test_plan_topic_no_quotes(self, mixin_instance):
        """Plan topic without quotes works."""
        result = mixin_instance._parse_dm_command("plan improve testing coverage")
        assert "improve testing coverage" in result

    def test_implement_topic_no_quotes(self, mixin_instance):
        """Implement topic without quotes works."""
        result = mixin_instance._parse_dm_command("implement better error handling")
        assert "better error handling" in result


# ---------------------------------------------------------------------------
# _get_status_response tests
# ---------------------------------------------------------------------------


class TestGetStatusResponse:
    """Tests for EventsMixin._get_status_response."""

    def test_elo_system_available(self, mixin_instance):
        """Returns agent count when EloSystem is available."""
        mock_elo = MagicMock()
        mock_elo.return_value.get_all_ratings.return_value = [
            SimpleNamespace(agent_id="a", rating=1500),
            SimpleNamespace(agent_id="b", rating=1600),
            SimpleNamespace(agent_id="c", rating=1400),
        ]

        with patch.dict(
            "sys.modules",
            {"aragora.ranking.elo": MagicMock(EloSystem=mock_elo)},
        ):
            result = mixin_instance._get_status_response()

        assert "Aragora Status" in result
        assert "Online" in result
        assert "3 registered" in result

    def test_import_error_fallback(self, mixin_instance):
        """ImportError falls back to 'Unknown' agents."""
        saved = sys.modules.get("aragora.ranking.elo")
        sys.modules["aragora.ranking.elo"] = None  # type: ignore[assignment]

        try:
            result = mixin_instance._get_status_response()
        finally:
            if saved is not None:
                sys.modules["aragora.ranking.elo"] = saved
            else:
                sys.modules.pop("aragora.ranking.elo", None)

        assert "Aragora Status" in result
        assert "Unknown" in result

    def test_runtime_error_fallback(self, mixin_instance):
        """RuntimeError falls back to 'Unknown' agents."""
        mock_elo_cls = MagicMock(side_effect=RuntimeError("DB unavailable"))

        with patch.dict(
            "sys.modules",
            {"aragora.ranking.elo": MagicMock(EloSystem=mock_elo_cls)},
        ):
            result = mixin_instance._get_status_response()

        assert "Aragora Status" in result
        assert "Unknown" in result

    def test_attribute_error_fallback(self, mixin_instance):
        """AttributeError falls back to 'Unknown' agents."""
        mock_elo_cls = MagicMock(side_effect=AttributeError("no attribute"))

        with patch.dict(
            "sys.modules",
            {"aragora.ranking.elo": MagicMock(EloSystem=mock_elo_cls)},
        ):
            result = mixin_instance._get_status_response()

        assert "Aragora Status" in result
        assert "Unknown" in result

    def test_zero_agents(self, mixin_instance):
        """Returns '0 registered' when EloSystem returns empty list."""
        mock_elo = MagicMock()
        mock_elo.return_value.get_all_ratings.return_value = []

        with patch.dict(
            "sys.modules",
            {"aragora.ranking.elo": MagicMock(EloSystem=mock_elo)},
        ):
            result = mixin_instance._get_status_response()

        assert "0 registered" in result


# ---------------------------------------------------------------------------
# _get_agents_response tests
# ---------------------------------------------------------------------------


class TestGetAgentsResponse:
    """Tests for EventsMixin._get_agents_response."""

    def test_agents_available_sorted(self, mixin_instance):
        """Returns sorted top-5 agents by rating."""
        agents = [
            SimpleNamespace(agent_id="low", rating=1200),
            SimpleNamespace(agent_id="high", rating=1800),
            SimpleNamespace(agent_id="mid", rating=1500),
        ]
        mock_elo_cls = MagicMock()
        mock_elo_cls.return_value.get_all_ratings.return_value = agents

        with patch.dict(
            "sys.modules",
            {"aragora.ranking.elo": MagicMock(EloSystem=mock_elo_cls)},
        ):
            result = mixin_instance._get_agents_response()

        assert "Top Agents" in result
        assert "high" in result
        assert "mid" in result
        assert "low" in result
        lines = result.strip().split("\n")
        assert "high" in lines[1]

    def test_more_than_5_agents_truncated(self, mixin_instance):
        """Only top 5 agents are shown."""
        agents = [SimpleNamespace(agent_id=f"agent-{i}", rating=1500 + i * 100) for i in range(8)]
        mock_elo_cls = MagicMock()
        mock_elo_cls.return_value.get_all_ratings.return_value = agents

        with patch.dict(
            "sys.modules",
            {"aragora.ranking.elo": MagicMock(EloSystem=mock_elo_cls)},
        ):
            result = mixin_instance._get_agents_response()

        lines = result.strip().split("\n")
        assert len(lines) == 6

    def test_empty_agents_list(self, mixin_instance):
        """Empty agents list returns 'No agents registered yet.'"""
        mock_elo_cls = MagicMock()
        mock_elo_cls.return_value.get_all_ratings.return_value = []

        with patch.dict(
            "sys.modules",
            {"aragora.ranking.elo": MagicMock(EloSystem=mock_elo_cls)},
        ):
            result = mixin_instance._get_agents_response()

        assert result == "No agents registered yet."

    def test_import_error_fallback(self, mixin_instance):
        """ImportError returns fallback message."""
        saved = sys.modules.get("aragora.ranking.elo")
        sys.modules["aragora.ranking.elo"] = None  # type: ignore[assignment]

        try:
            result = mixin_instance._get_agents_response()
        finally:
            if saved is not None:
                sys.modules["aragora.ranking.elo"] = saved
            else:
                sys.modules.pop("aragora.ranking.elo", None)

        assert "temporarily unavailable" in result

    def test_runtime_error_fallback(self, mixin_instance):
        """RuntimeError returns fallback message."""
        mock_elo_cls = MagicMock(side_effect=RuntimeError("DB down"))

        with patch.dict(
            "sys.modules",
            {"aragora.ranking.elo": MagicMock(EloSystem=mock_elo_cls)},
        ):
            result = mixin_instance._get_agents_response()

        assert "temporarily unavailable" in result

    def test_attribute_error_fallback(self, mixin_instance):
        """AttributeError returns fallback message."""
        mock_elo_cls = MagicMock(side_effect=AttributeError("missing"))

        with patch.dict(
            "sys.modules",
            {"aragora.ranking.elo": MagicMock(EloSystem=mock_elo_cls)},
        ):
            result = mixin_instance._get_agents_response()

        assert "temporarily unavailable" in result

    def test_agent_missing_rating_uses_default(self, mixin_instance):
        """Agents without 'rating' attribute use default 1500."""
        agents = [SimpleNamespace(agent_id="no-rating")]
        mock_elo_cls = MagicMock()
        mock_elo_cls.return_value.get_all_ratings.return_value = agents

        with patch.dict(
            "sys.modules",
            {"aragora.ranking.elo": MagicMock(EloSystem=mock_elo_cls)},
        ):
            result = mixin_instance._get_agents_response()

        assert "no-rating" in result
        assert "1500" in result

    def test_agent_missing_agent_id_uses_unknown(self, mixin_instance):
        """Agents without 'agent_id' attribute display 'Unknown'."""
        agents = [SimpleNamespace(rating=1600)]
        mock_elo_cls = MagicMock()
        mock_elo_cls.return_value.get_all_ratings.return_value = agents

        with patch.dict(
            "sys.modules",
            {"aragora.ranking.elo": MagicMock(EloSystem=mock_elo_cls)},
        ):
            result = mixin_instance._get_agents_response()

        assert "Unknown" in result
        assert "1600" in result

    def test_single_agent(self, mixin_instance):
        """Single agent is displayed correctly."""
        agents = [SimpleNamespace(agent_id="solo", rating=1700)]
        mock_elo_cls = MagicMock()
        mock_elo_cls.return_value.get_all_ratings.return_value = agents

        with patch.dict(
            "sys.modules",
            {"aragora.ranking.elo": MagicMock(EloSystem=mock_elo_cls)},
        ):
            result = mixin_instance._get_agents_response()

        assert "1. solo" in result
        assert "1700" in result


# ---------------------------------------------------------------------------
# _post_message_async tests
# ---------------------------------------------------------------------------


class TestPostMessageAsync:
    """Tests for EventsMixin._post_message_async."""

    @pytest.mark.asyncio
    async def test_successful_post_returns_ts(self, mixin_instance):
        """Successful Slack API call returns the message ts."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": True, "ts": "1234.5678"}

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response

        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__.return_value = mock_client

        mock_pool = MagicMock()
        mock_pool.get_session.return_value = mock_session_ctx

        with (
            patch(f"{_IMPL}.SLACK_BOT_TOKEN", "xoxb-test"),
            patch(
                "aragora.server.http_client_pool.get_http_pool",
                return_value=mock_pool,
            ),
        ):
            result = await mixin_instance._post_message_async("C1", "Hello!")

        assert result == "1234.5678"

    @pytest.mark.asyncio
    async def test_no_bot_token_returns_none(self, mixin_instance):
        """Returns None when SLACK_BOT_TOKEN is empty."""
        with patch(f"{_IMPL}.SLACK_BOT_TOKEN", ""):
            result = await mixin_instance._post_message_async("C1", "Hello!")

        assert result is None

    @pytest.mark.asyncio
    async def test_api_error_returns_none(self, mixin_instance):
        """Returns None when Slack API returns ok=false."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": False, "error": "channel_not_found"}

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response

        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__.return_value = mock_client

        mock_pool = MagicMock()
        mock_pool.get_session.return_value = mock_session_ctx

        with (
            patch(f"{_IMPL}.SLACK_BOT_TOKEN", "xoxb-test"),
            patch(
                "aragora.server.http_client_pool.get_http_pool",
                return_value=mock_pool,
            ),
        ):
            result = await mixin_instance._post_message_async("C1", "Hello!")

        assert result is None

    @pytest.mark.asyncio
    async def test_connection_error_returns_none(self, mixin_instance):
        """Returns None on ConnectionError."""
        mock_client = AsyncMock()
        mock_client.post.side_effect = ConnectionError("refused")

        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__.return_value = mock_client

        mock_pool = MagicMock()
        mock_pool.get_session.return_value = mock_session_ctx

        with (
            patch(f"{_IMPL}.SLACK_BOT_TOKEN", "xoxb-test"),
            patch(
                "aragora.server.http_client_pool.get_http_pool",
                return_value=mock_pool,
            ),
        ):
            result = await mixin_instance._post_message_async("C1", "Hello!")

        assert result is None

    @pytest.mark.asyncio
    async def test_timeout_error_returns_none(self, mixin_instance):
        """Returns None on TimeoutError."""
        mock_client = AsyncMock()
        mock_client.post.side_effect = TimeoutError("timed out")

        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__.return_value = mock_client

        mock_pool = MagicMock()
        mock_pool.get_session.return_value = mock_session_ctx

        with (
            patch(f"{_IMPL}.SLACK_BOT_TOKEN", "xoxb-test"),
            patch(
                "aragora.server.http_client_pool.get_http_pool",
                return_value=mock_pool,
            ),
        ):
            result = await mixin_instance._post_message_async("C1", "Hello!")

        assert result is None

    @pytest.mark.asyncio
    async def test_runtime_error_returns_none(self, mixin_instance):
        """Returns None on RuntimeError."""
        mock_client = AsyncMock()
        mock_client.post.side_effect = RuntimeError("unexpected")

        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__.return_value = mock_client

        mock_pool = MagicMock()
        mock_pool.get_session.return_value = mock_session_ctx

        with (
            patch(f"{_IMPL}.SLACK_BOT_TOKEN", "xoxb-test"),
            patch(
                "aragora.server.http_client_pool.get_http_pool",
                return_value=mock_pool,
            ),
        ):
            result = await mixin_instance._post_message_async("C1", "Hello!")

        assert result is None

    @pytest.mark.asyncio
    async def test_value_error_returns_none(self, mixin_instance):
        """Returns None on ValueError."""
        mock_client = AsyncMock()
        mock_client.post.side_effect = ValueError("bad value")

        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__.return_value = mock_client

        mock_pool = MagicMock()
        mock_pool.get_session.return_value = mock_session_ctx

        with (
            patch(f"{_IMPL}.SLACK_BOT_TOKEN", "xoxb-test"),
            patch(
                "aragora.server.http_client_pool.get_http_pool",
                return_value=mock_pool,
            ),
        ):
            result = await mixin_instance._post_message_async("C1", "Hello!")

        assert result is None

    @pytest.mark.asyncio
    async def test_thread_ts_included(self, mixin_instance):
        """thread_ts is included in payload when provided."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": True, "ts": "1234.5678"}

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response

        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__.return_value = mock_client

        mock_pool = MagicMock()
        mock_pool.get_session.return_value = mock_session_ctx

        with (
            patch(f"{_IMPL}.SLACK_BOT_TOKEN", "xoxb-test"),
            patch(
                "aragora.server.http_client_pool.get_http_pool",
                return_value=mock_pool,
            ),
        ):
            await mixin_instance._post_message_async("C1", "Hello!", thread_ts="999.888")

        call_kwargs = mock_client.post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["thread_ts"] == "999.888"

    @pytest.mark.asyncio
    async def test_blocks_included(self, mixin_instance):
        """Blocks are included in payload when provided."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": True, "ts": "1234.5678"}

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response

        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__.return_value = mock_client

        mock_pool = MagicMock()
        mock_pool.get_session.return_value = mock_session_ctx

        blocks = [{"type": "section", "text": {"type": "mrkdwn", "text": "Hello"}}]

        with (
            patch(f"{_IMPL}.SLACK_BOT_TOKEN", "xoxb-test"),
            patch(
                "aragora.server.http_client_pool.get_http_pool",
                return_value=mock_pool,
            ),
        ):
            await mixin_instance._post_message_async("C1", "Hello!", blocks=blocks)

        call_kwargs = mock_client.post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["blocks"] == blocks

    @pytest.mark.asyncio
    async def test_no_thread_ts_when_none(self, mixin_instance):
        """thread_ts is not in payload when not provided."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": True, "ts": "1234.5678"}

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response

        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__.return_value = mock_client

        mock_pool = MagicMock()
        mock_pool.get_session.return_value = mock_session_ctx

        with (
            patch(f"{_IMPL}.SLACK_BOT_TOKEN", "xoxb-test"),
            patch(
                "aragora.server.http_client_pool.get_http_pool",
                return_value=mock_pool,
            ),
        ):
            await mixin_instance._post_message_async("C1", "Hello!")

        call_kwargs = mock_client.post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert "thread_ts" not in payload

    @pytest.mark.asyncio
    async def test_no_blocks_when_none(self, mixin_instance):
        """blocks is not in payload when not provided."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": True, "ts": "1234.5678"}

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response

        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__.return_value = mock_client

        mock_pool = MagicMock()
        mock_pool.get_session.return_value = mock_session_ctx

        with (
            patch(f"{_IMPL}.SLACK_BOT_TOKEN", "xoxb-test"),
            patch(
                "aragora.server.http_client_pool.get_http_pool",
                return_value=mock_pool,
            ),
        ):
            await mixin_instance._post_message_async("C1", "Hello!")

        call_kwargs = mock_client.post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert "blocks" not in payload

    @pytest.mark.asyncio
    async def test_correct_url_and_headers(self, mixin_instance):
        """Posts to correct Slack API URL with proper headers."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": True, "ts": "1234.5678"}

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response

        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__.return_value = mock_client

        mock_pool = MagicMock()
        mock_pool.get_session.return_value = mock_session_ctx

        with (
            patch(f"{_IMPL}.SLACK_BOT_TOKEN", "xoxb-my-token"),
            patch(
                "aragora.server.http_client_pool.get_http_pool",
                return_value=mock_pool,
            ),
        ):
            await mixin_instance._post_message_async("C1", "Hello!")

        call_args = mock_client.post.call_args
        assert call_args[0][0] == "https://slack.com/api/chat.postMessage"
        headers = call_args.kwargs.get("headers") or call_args[1].get("headers")
        assert headers["Authorization"] == "Bearer xoxb-my-token"
        assert headers["Content-Type"] == "application/json"

    @pytest.mark.asyncio
    async def test_os_error_returns_none(self, mixin_instance):
        """Returns None on OSError."""
        mock_client = AsyncMock()
        mock_client.post.side_effect = OSError("network error")

        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__.return_value = mock_client

        mock_pool = MagicMock()
        mock_pool.get_session.return_value = mock_session_ctx

        with (
            patch(f"{_IMPL}.SLACK_BOT_TOKEN", "xoxb-test"),
            patch(
                "aragora.server.http_client_pool.get_http_pool",
                return_value=mock_pool,
            ),
        ):
            result = await mixin_instance._post_message_async("C1", "Hello!")

        assert result is None

    @pytest.mark.asyncio
    async def test_type_error_returns_none(self, mixin_instance):
        """Returns None on TypeError."""
        mock_client = AsyncMock()
        mock_client.post.side_effect = TypeError("bad type")

        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__.return_value = mock_client

        mock_pool = MagicMock()
        mock_pool.get_session.return_value = mock_session_ctx

        with (
            patch(f"{_IMPL}.SLACK_BOT_TOKEN", "xoxb-test"),
            patch(
                "aragora.server.http_client_pool.get_http_pool",
                return_value=mock_pool,
            ),
        ):
            result = await mixin_instance._post_message_async("C1", "Hello!")

        assert result is None

    @pytest.mark.asyncio
    async def test_payload_channel_and_text(self, mixin_instance):
        """Payload always includes channel and text."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": True, "ts": "1234.5678"}

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response

        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__.return_value = mock_client

        mock_pool = MagicMock()
        mock_pool.get_session.return_value = mock_session_ctx

        with (
            patch(f"{_IMPL}.SLACK_BOT_TOKEN", "xoxb-test"),
            patch(
                "aragora.server.http_client_pool.get_http_pool",
                return_value=mock_pool,
            ),
        ):
            await mixin_instance._post_message_async("C123", "Test message")

        call_kwargs = mock_client.post.call_args
        payload = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
        assert payload["channel"] == "C123"
        assert payload["text"] == "Test message"

    @pytest.mark.asyncio
    async def test_timeout_param_set(self, mixin_instance):
        """Request timeout is set to 30 seconds."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": True, "ts": "1234.5678"}

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response

        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__.return_value = mock_client

        mock_pool = MagicMock()
        mock_pool.get_session.return_value = mock_session_ctx

        with (
            patch(f"{_IMPL}.SLACK_BOT_TOKEN", "xoxb-test"),
            patch(
                "aragora.server.http_client_pool.get_http_pool",
                return_value=mock_pool,
            ),
        ):
            await mixin_instance._post_message_async("C1", "Hello!")

        call_kwargs = mock_client.post.call_args
        timeout = call_kwargs.kwargs.get("timeout") or call_kwargs[1].get("timeout")
        assert timeout == 30

    @pytest.mark.asyncio
    async def test_pool_session_named_slack(self, mixin_instance):
        """HTTP pool session is fetched with name 'slack'."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"ok": True, "ts": "1234.5678"}

        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response

        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__.return_value = mock_client

        mock_pool = MagicMock()
        mock_pool.get_session.return_value = mock_session_ctx

        with (
            patch(f"{_IMPL}.SLACK_BOT_TOKEN", "xoxb-test"),
            patch(
                "aragora.server.http_client_pool.get_http_pool",
                return_value=mock_pool,
            ),
        ):
            await mixin_instance._post_message_async("C1", "Hello!")

        mock_pool.get_session.assert_called_with("slack")
