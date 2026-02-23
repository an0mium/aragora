"""Comprehensive tests for the CommandsMixin in _slack_impl/commands.py.

Covers every public/internal method of CommandsMixin:
- _handle_slash_command dispatch (all subcommands, unknown, empty)
- Per-workspace and per-user rate limiting
- Audit logging on success, error, and rate-limit events
- _command_help (text content, all commands mentioned, examples)
- _command_status (success with ELO, ImportError, data error, connection error)
- _command_agents (sorted, empty, top-10 limit, medals, import/attribute errors)
- _command_ask (validation, quotes, async task queueing, boundary lengths)
- _command_search (db.search, db.list fallback, no results, object format, limits, errors)
- _command_leaderboard (table, empty, agent count, import/attribute errors)
- _command_recent (dict+object formats, buttons, truncation, empty, errors)
- _command_debate (validation, plan/implement modes, boundary lengths, decision_integrity)
- _command_gauntlet (validation, boundary lengths, async queueing, quotes)
- _answer_question_async (happy path, debate fallback, quick-answer error, network error)
- _run_gauntlet_async (happy path, vulnerabilities, API error, network error)
- _create_debate_async (starting message, no agents, error, origin tracking)
- _update_debate_status (store available, unavailable, with error)
- Error handling for all caught exception types in _handle_slash_command
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import urlencode

import pytest


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


def _make_form_body(**kwargs) -> str:
    """Build a URL-encoded form body string from kwargs."""
    return urlencode(kwargs)


def _make_agent(name: str, elo: float = 1500, wins: int = 0, losses: int = 0):
    """Create a mock agent object with ELO attributes."""
    return SimpleNamespace(name=name, elo=elo, wins=wins, losses=losses)


def _make_handler_with_body(
    text: str = "",
    command: str = "/aragora",
    user_id: str = "U123",
    channel_id: str = "C456",
    team_id: str = "T789",
    response_url: str = "https://hooks.slack.com/resp/123",
):
    """Build a mock HTTP handler with Slack form body attributes."""
    h = MagicMock()
    form: dict[str, str] = {
        "command": command,
        "text": text,
        "user_id": user_id,
        "channel_id": channel_id,
        "response_url": response_url,
    }
    if team_id:
        form["team_id"] = team_id
    h._slack_body = urlencode(form)
    h._slack_workspace = None
    h._slack_team_id = team_id if team_id else None
    h.command = "POST"
    return h


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def handler_module():
    """Import the handler module lazily (after conftest patches)."""
    from aragora.server.handlers.social._slack_impl import handler as mod

    return mod


@pytest.fixture
def commands_module():
    """Import the commands module lazily."""
    from aragora.server.handlers.social._slack_impl import commands as mod

    return mod


@pytest.fixture
def config_module():
    """Import the config module lazily."""
    from aragora.server.handlers.social._slack_impl import config as mod

    return mod


@pytest.fixture
def slack_handler(handler_module):
    """Create a SlackHandler with empty context."""
    return handler_module.SlackHandler(ctx={})


@pytest.fixture(autouse=True)
def _reset_config_singletons(config_module, monkeypatch):
    """Reset module-level singletons between tests."""
    monkeypatch.setattr(config_module, "_slack_audit", None)
    monkeypatch.setattr(config_module, "_slack_user_limiter", None)
    monkeypatch.setattr(config_module, "_slack_workspace_limiter", None)
    monkeypatch.setattr(config_module, "_slack_integration", None)
    yield


@pytest.fixture(autouse=True)
def _disable_rate_limit_decorator(monkeypatch):
    """Disable the @rate_limit decorator so it does not interfere with tests."""
    try:
        from aragora.server.handlers.utils import rate_limit as rl_mod

        monkeypatch.setattr(rl_mod, "_RATE_LIMIT_DISABLED", True, raising=False)
    except (ImportError, AttributeError):
        pass
    yield


def _bypass_limiters_and_audit(commands_module, monkeypatch):
    """Helper to disable both rate limiters and audit logger."""
    monkeypatch.setattr(commands_module, "_get_workspace_rate_limiter", lambda: None)
    monkeypatch.setattr(commands_module, "_get_user_rate_limiter", lambda: None)
    monkeypatch.setattr(commands_module, "_get_audit_logger", lambda: None)


# ---------------------------------------------------------------------------
# _command_help
# ---------------------------------------------------------------------------


class TestCommandHelp:
    """Tests for the help subcommand."""

    def test_help_returns_ephemeral(self, slack_handler):
        body = _body(slack_handler._command_help())
        assert body["response_type"] == "ephemeral"

    def test_help_contains_header(self, slack_handler):
        body = _body(slack_handler._command_help())
        assert "Aragora Slash Commands" in body["text"]

    def test_help_mentions_all_commands(self, slack_handler):
        body = _body(slack_handler._command_help())
        expected = [
            "debate", "plan", "implement", "ask", "gauntlet",
            "search", "recent", "leaderboard", "agents", "status", "help",
        ]
        for cmd in expected:
            assert cmd in body["text"], f"Help text should mention '{cmd}'"

    def test_help_includes_examples(self, slack_handler):
        body = _body(slack_handler._command_help())
        assert "Examples:" in body["text"]

    def test_help_mentions_search_example(self, slack_handler):
        body = _body(slack_handler._command_help())
        assert "machine learning" in body["text"]


# ---------------------------------------------------------------------------
# _command_status
# ---------------------------------------------------------------------------


class TestCommandStatus:
    """Tests for the status subcommand."""

    def test_status_returns_blocks(self, slack_handler):
        mock_elo_mod = MagicMock()
        mock_elo_mod.EloSystem.return_value.get_all_ratings.return_value = [
            _make_agent("a1"), _make_agent("a2"),
        ]
        with patch.dict("sys.modules", {"aragora.ranking.elo": mock_elo_mod}):
            result = slack_handler._command_status()
            body = _body(result)
            assert _status(result) == 200
            assert "blocks" in body

    def test_status_shows_agent_count_in_blocks(self, slack_handler):
        mock_elo_mod = MagicMock()
        mock_elo_mod.EloSystem.return_value.get_all_ratings.return_value = [
            _make_agent("a1"), _make_agent("a2"), _make_agent("a3"),
        ]
        with patch.dict("sys.modules", {"aragora.ranking.elo": mock_elo_mod}):
            result = slack_handler._command_status()
            body = _body(result)
            blocks_text = json.dumps(body.get("blocks", []))
            assert "3" in blocks_text

    def test_status_import_error(self, slack_handler):
        with patch.dict("sys.modules", {"aragora.ranking.elo": None}):
            result = slack_handler._command_status()
            body = _body(result)
            assert "unavailable" in body["text"].lower()

    def test_status_data_error_type_error(self, slack_handler):
        mock_elo_mod = MagicMock()
        mock_elo_mod.EloSystem.return_value.get_all_ratings.side_effect = TypeError("bad")
        with patch.dict("sys.modules", {"aragora.ranking.elo": mock_elo_mod}):
            result = slack_handler._command_status()
            body = _body(result)
            assert "error" in body["text"].lower()

    def test_status_data_error_key_error(self, slack_handler):
        mock_elo_mod = MagicMock()
        mock_elo_mod.EloSystem.return_value.get_all_ratings.side_effect = KeyError("x")
        with patch.dict("sys.modules", {"aragora.ranking.elo": mock_elo_mod}):
            result = slack_handler._command_status()
            body = _body(result)
            assert "error" in body["text"].lower()

    def test_status_data_error_attribute_error(self, slack_handler):
        mock_elo_mod = MagicMock()
        mock_elo_mod.EloSystem.return_value.get_all_ratings.side_effect = AttributeError("no attr")
        with patch.dict("sys.modules", {"aragora.ranking.elo": mock_elo_mod}):
            result = slack_handler._command_status()
            body = _body(result)
            assert "error" in body["text"].lower()

    def test_status_connection_error(self, slack_handler):
        mock_elo_mod = MagicMock()
        mock_elo_mod.EloSystem.return_value.get_all_ratings.side_effect = ConnectionError("down")
        with patch.dict("sys.modules", {"aragora.ranking.elo": mock_elo_mod}):
            result = slack_handler._command_status()
            body = _body(result)
            assert "error" in body["text"].lower()

    def test_status_timeout_error(self, slack_handler):
        mock_elo_mod = MagicMock()
        mock_elo_mod.EloSystem.return_value.get_all_ratings.side_effect = TimeoutError("slow")
        with patch.dict("sys.modules", {"aragora.ranking.elo": mock_elo_mod}):
            result = slack_handler._command_status()
            body = _body(result)
            assert "error" in body["text"].lower()

    def test_status_response_type_is_ephemeral(self, slack_handler):
        mock_elo_mod = MagicMock()
        mock_elo_mod.EloSystem.return_value.get_all_ratings.return_value = [_make_agent("x")]
        with patch.dict("sys.modules", {"aragora.ranking.elo": mock_elo_mod}):
            result = slack_handler._command_status()
            body = _body(result)
            assert body["response_type"] == "ephemeral"


# ---------------------------------------------------------------------------
# _command_agents
# ---------------------------------------------------------------------------


class TestCommandAgents:
    """Tests for the agents subcommand."""

    def test_agents_empty_returns_message(self, slack_handler):
        mock_elo_mod = MagicMock()
        mock_elo_mod.EloSystem.return_value.get_all_ratings.return_value = []
        with patch.dict("sys.modules", {"aragora.ranking.elo": mock_elo_mod}):
            body = _body(slack_handler._command_agents())
            assert "no agents" in body["text"].lower()

    def test_agents_sorted_descending_by_elo(self, slack_handler):
        agents = [
            _make_agent("low", elo=1200, wins=2),
            _make_agent("high", elo=1800, wins=10),
            _make_agent("mid", elo=1500, wins=5),
        ]
        mock_elo_mod = MagicMock()
        mock_elo_mod.EloSystem.return_value.get_all_ratings.return_value = agents
        with patch.dict("sys.modules", {"aragora.ranking.elo": mock_elo_mod}):
            body = _body(slack_handler._command_agents())
            text = body["text"]
            assert text.index("high") < text.index("mid") < text.index("low")

    def test_agents_limits_to_top_10(self, slack_handler):
        agents = [_make_agent(f"agent{i}", elo=2000 - i * 10) for i in range(15)]
        mock_elo_mod = MagicMock()
        mock_elo_mod.EloSystem.return_value.get_all_ratings.return_value = agents
        with patch.dict("sys.modules", {"aragora.ranking.elo": mock_elo_mod}):
            body = _body(slack_handler._command_agents())
            assert "agent14" not in body["text"]
            assert "agent0" in body["text"]

    def test_agents_medals_for_top_3(self, slack_handler):
        agents = [
            _make_agent("gold", elo=2000, wins=20),
            _make_agent("silver", elo=1900, wins=15),
            _make_agent("bronze", elo=1800, wins=10),
            _make_agent("fourth", elo=1700, wins=5),
        ]
        mock_elo_mod = MagicMock()
        mock_elo_mod.EloSystem.return_value.get_all_ratings.return_value = agents
        with patch.dict("sys.modules", {"aragora.ranking.elo": mock_elo_mod}):
            body = _body(slack_handler._command_agents())
            assert "\U0001f947" in body["text"]  # gold medal
            assert "\U0001f948" in body["text"]  # silver medal
            assert "\U0001f949" in body["text"]  # bronze medal

    def test_agents_shows_wins(self, slack_handler):
        agents = [_make_agent("winner", elo=1800, wins=42)]
        mock_elo_mod = MagicMock()
        mock_elo_mod.EloSystem.return_value.get_all_ratings.return_value = agents
        with patch.dict("sys.modules", {"aragora.ranking.elo": mock_elo_mod}):
            body = _body(slack_handler._command_agents())
            assert "42" in body["text"]

    def test_agents_import_error(self, slack_handler):
        with patch.dict("sys.modules", {"aragora.ranking.elo": None}):
            body = _body(slack_handler._command_agents())
            assert "unavailable" in body["text"].lower()

    def test_agents_attribute_error(self, slack_handler):
        mock_elo_mod = MagicMock()
        mock_elo_mod.EloSystem.return_value.get_all_ratings.side_effect = AttributeError("no attr")
        with patch.dict("sys.modules", {"aragora.ranking.elo": mock_elo_mod}):
            body = _body(slack_handler._command_agents())
            assert "error" in body["text"].lower()

    def test_agents_connection_error(self, slack_handler):
        mock_elo_mod = MagicMock()
        mock_elo_mod.EloSystem.return_value.get_all_ratings.side_effect = ConnectionError("down")
        with patch.dict("sys.modules", {"aragora.ranking.elo": mock_elo_mod}):
            body = _body(slack_handler._command_agents())
            assert "error" in body["text"].lower()

    def test_agents_single_agent(self, slack_handler):
        agents = [_make_agent("solo_agent", elo=1600, wins=3)]
        mock_elo_mod = MagicMock()
        mock_elo_mod.EloSystem.return_value.get_all_ratings.return_value = agents
        with patch.dict("sys.modules", {"aragora.ranking.elo": mock_elo_mod}):
            body = _body(slack_handler._command_agents())
            assert "solo_agent" in body["text"]


# ---------------------------------------------------------------------------
# _command_ask
# ---------------------------------------------------------------------------


class TestCommandAsk:
    """Tests for the ask subcommand."""

    def test_ask_no_args(self, slack_handler):
        body = _body(slack_handler._command_ask("", "U1", "C1", "https://hooks.slack.com/r"))
        assert "provide a question" in body["text"].lower()

    def test_ask_too_short_4_chars(self, slack_handler):
        body = _body(slack_handler._command_ask("Hiya", "U1", "C1", "https://hooks.slack.com/r"))
        assert "too short" in body["text"].lower()

    def test_ask_exactly_5_chars_accepted(self, slack_handler, commands_module, monkeypatch):
        monkeypatch.setattr(commands_module, "create_tracked_task", MagicMock())
        body = _body(slack_handler._command_ask("Hello", "U1", "C1", "https://hooks.slack.com/r"))
        assert "too short" not in body.get("text", "").lower()

    def test_ask_too_long_501_chars(self, slack_handler):
        body = _body(slack_handler._command_ask("x" * 501, "U1", "C1", "https://hooks.slack.com/r"))
        assert "too long" in body["text"].lower()

    def test_ask_exactly_500_chars_accepted(self, slack_handler, commands_module, monkeypatch):
        monkeypatch.setattr(commands_module, "create_tracked_task", MagicMock())
        body = _body(slack_handler._command_ask("x" * 500, "U1", "C1", "https://hooks.slack.com/r"))
        assert "too long" not in body.get("text", "").lower()

    def test_ask_strips_double_quotes(self, slack_handler, commands_module, monkeypatch):
        monkeypatch.setattr(commands_module, "create_tracked_task", MagicMock())
        result = slack_handler._command_ask(
            '"What is the meaning of life?"', "U1", "C1", "https://hooks.slack.com/r"
        )
        body = _body(result)
        assert body["response_type"] == "in_channel"
        assert "blocks" in body

    def test_ask_strips_single_quotes(self, slack_handler, commands_module, monkeypatch):
        monkeypatch.setattr(commands_module, "create_tracked_task", MagicMock())
        result = slack_handler._command_ask(
            "'What is quantum computing all about?'", "U1", "C1", "https://hooks.slack.com/r"
        )
        body = _body(result)
        assert body["response_type"] == "in_channel"

    def test_ask_queues_async_task(self, slack_handler, commands_module, monkeypatch):
        mock_create = MagicMock()
        monkeypatch.setattr(commands_module, "create_tracked_task", mock_create)
        slack_handler._command_ask(
            "What is quantum computing used for?", "U1", "C1", "https://hooks.slack.com/r"
        )
        mock_create.assert_called_once()

    def test_ask_no_response_url_skips_task(self, slack_handler, commands_module, monkeypatch):
        mock_create = MagicMock()
        monkeypatch.setattr(commands_module, "create_tracked_task", mock_create)
        slack_handler._command_ask(
            "What is quantum computing used for?", "U1", "C1", ""
        )
        mock_create.assert_not_called()

    def test_ask_response_contains_user_id(self, slack_handler, commands_module, monkeypatch):
        monkeypatch.setattr(commands_module, "create_tracked_task", MagicMock())
        result = slack_handler._command_ask(
            "Explain artificial intelligence in detail",
            "UXYZ99",
            "C1",
            "https://hooks.slack.com/x",
        )
        body = _body(result)
        blocks_text = json.dumps(body.get("blocks", []))
        assert "UXYZ99" in blocks_text

    def test_ask_response_is_in_channel(self, slack_handler, commands_module, monkeypatch):
        monkeypatch.setattr(commands_module, "create_tracked_task", MagicMock())
        body = _body(slack_handler._command_ask(
            "Why is the sky blue in the daytime?", "U1", "C1", "https://hooks.slack.com/r"
        ))
        assert body["response_type"] == "in_channel"

    def test_ask_truncates_long_question_in_blocks(self, slack_handler, commands_module, monkeypatch):
        monkeypatch.setattr(commands_module, "create_tracked_task", MagicMock())
        long_q = "W" * 250
        result = slack_handler._command_ask(long_q, "U1", "C1", "https://hooks.slack.com/r")
        body = _body(result)
        blocks_text = json.dumps(body.get("blocks", []))
        assert "..." in blocks_text


# ---------------------------------------------------------------------------
# _command_search
# ---------------------------------------------------------------------------


class TestCommandSearch:
    """Tests for the search subcommand."""

    def test_search_no_args(self, slack_handler):
        body = _body(slack_handler._command_search(""))
        assert "provide a search query" in body["text"].lower()

    def test_search_too_short_1_char(self, slack_handler):
        body = _body(slack_handler._command_search("a"))
        assert "too short" in body["text"].lower()

    def test_search_2_chars_accepted(self, slack_handler, commands_module, monkeypatch):
        mock_db = MagicMock()
        mock_db.search.return_value = ([], 0)
        monkeypatch.setattr(commands_module, "get_debates_db", MagicMock(return_value=mock_db))
        body = _body(slack_handler._command_search("AI"))
        assert "too short" not in body.get("text", "").lower()

    def test_search_with_db_search(self, slack_handler, commands_module, monkeypatch):
        mock_db = MagicMock()
        mock_db.search.return_value = (
            [{"task": "AI regulation", "consensus_reached": True, "id": "abc123", "confidence": 0.85}],
            1,
        )
        monkeypatch.setattr(commands_module, "get_debates_db", MagicMock(return_value=mock_db))
        result = slack_handler._command_search("AI regulation")
        body = _body(result)
        assert "blocks" in body
        assert "1 result" in body["text"]

    def test_search_with_db_list_fallback(self, slack_handler, commands_module, monkeypatch):
        mock_db = MagicMock(spec=["list"])
        mock_db.list.return_value = [
            {
                "task": "Machine learning basics",
                "final_answer": "ML is...",
                "id": "xyz",
                "consensus_reached": False,
                "confidence": 0.5,
            },
        ]
        monkeypatch.setattr(commands_module, "get_debates_db", MagicMock(return_value=mock_db))
        result = slack_handler._command_search("machine")
        body = _body(result)
        assert "blocks" in body

    def test_search_list_fallback_no_match(self, slack_handler, commands_module, monkeypatch):
        mock_db = MagicMock(spec=["list"])
        mock_db.list.return_value = [
            {
                "task": "Unrelated topic",
                "final_answer": "Something else",
                "id": "x",
                "consensus_reached": False,
                "confidence": 0,
            },
        ]
        monkeypatch.setattr(commands_module, "get_debates_db", MagicMock(return_value=mock_db))
        body = _body(slack_handler._command_search("quantum"))
        assert "no results" in body["text"].lower()

    def test_search_no_results(self, slack_handler, commands_module, monkeypatch):
        mock_db = MagicMock()
        mock_db.search.return_value = ([], 0)
        monkeypatch.setattr(commands_module, "get_debates_db", MagicMock(return_value=mock_db))
        body = _body(slack_handler._command_search("nonexistent"))
        assert "no results" in body["text"].lower()

    def test_search_db_none_attr(self, slack_handler, commands_module, monkeypatch):
        """get_debates_db is None (not importable)."""
        monkeypatch.setattr(commands_module, "get_debates_db", None)
        body = _body(slack_handler._command_search("some query"))
        assert "failed" in body["text"].lower() or "unavailable" in body["text"].lower()

    def test_search_db_returns_none(self, slack_handler, commands_module, monkeypatch):
        monkeypatch.setattr(commands_module, "get_debates_db", MagicMock(return_value=None))
        body = _body(slack_handler._command_search("some query"))
        assert "no results" in body["text"].lower()

    def test_search_object_format(self, slack_handler, commands_module, monkeypatch):
        item = SimpleNamespace(
            task="Object topic", consensus_reached=True, id="obj123", confidence=0.9
        )
        mock_db = MagicMock()
        mock_db.search.return_value = ([item], 1)
        monkeypatch.setattr(commands_module, "get_debates_db", MagicMock(return_value=mock_db))
        body = _body(slack_handler._command_search("object"))
        assert "blocks" in body

    def test_search_limits_to_5(self, slack_handler, commands_module, monkeypatch):
        items = [
            {"task": f"Topic {i}", "consensus_reached": True, "id": f"id{i}", "confidence": 0.5}
            for i in range(10)
        ]
        mock_db = MagicMock()
        mock_db.search.return_value = (items, 10)
        monkeypatch.setattr(commands_module, "get_debates_db", MagicMock(return_value=mock_db))
        body = _body(slack_handler._command_search("Topic"))
        section_blocks = [b for b in body.get("blocks", []) if b.get("type") == "section"]
        assert len(section_blocks) <= 5

    def test_search_strips_quotes(self, slack_handler, commands_module, monkeypatch):
        mock_db = MagicMock()
        mock_db.search.return_value = ([], 0)
        monkeypatch.setattr(commands_module, "get_debates_db", MagicMock(return_value=mock_db))
        slack_handler._command_search('"quoted query"')
        mock_db.search.assert_called_once_with("quoted query", limit=5)

    def test_search_value_error(self, slack_handler, commands_module, monkeypatch):
        mock_db = MagicMock()
        mock_db.search.side_effect = ValueError("bad query")
        monkeypatch.setattr(commands_module, "get_debates_db", MagicMock(return_value=mock_db))
        body = _body(slack_handler._command_search("valid query"))
        assert "failed" in body["text"].lower()

    def test_search_runtime_error(self, slack_handler, commands_module, monkeypatch):
        mock_db = MagicMock()
        mock_db.search.side_effect = RuntimeError("db unavailable")
        monkeypatch.setattr(commands_module, "get_debates_db", MagicMock(return_value=mock_db))
        body = _body(slack_handler._command_search("any query"))
        assert "failed" in body["text"].lower()

    def test_search_result_count_in_context_block(self, slack_handler, commands_module, monkeypatch):
        items = [
            {"task": "Alpha", "consensus_reached": True, "id": "a1", "confidence": 0.9},
            {"task": "Beta", "consensus_reached": False, "id": "b2", "confidence": 0.4},
        ]
        mock_db = MagicMock()
        mock_db.search.return_value = (items, 2)
        monkeypatch.setattr(commands_module, "get_debates_db", MagicMock(return_value=mock_db))
        body = _body(slack_handler._command_search("test"))
        blocks_text = json.dumps(body.get("blocks", []))
        assert "2 result" in blocks_text


# ---------------------------------------------------------------------------
# _command_leaderboard
# ---------------------------------------------------------------------------


class TestCommandLeaderboard:
    """Tests for the leaderboard subcommand."""

    def test_leaderboard_empty(self, slack_handler):
        mock_elo_mod = MagicMock()
        mock_elo_mod.EloSystem.return_value.get_all_ratings.return_value = []
        with patch.dict("sys.modules", {"aragora.ranking.elo": mock_elo_mod}):
            body = _body(slack_handler._command_leaderboard())
            assert "no agents ranked" in body["text"].lower()

    def test_leaderboard_renders_table(self, slack_handler):
        agents = [
            _make_agent("agent_a", elo=1800, wins=10, losses=2),
            _make_agent("agent_b", elo=1600, wins=7, losses=5),
        ]
        mock_elo_mod = MagicMock()
        mock_elo_mod.EloSystem.return_value.get_all_ratings.return_value = agents
        with patch.dict("sys.modules", {"aragora.ranking.elo": mock_elo_mod}):
            result = slack_handler._command_leaderboard()
            body = _body(result)
            assert "blocks" in body
            assert body["response_type"] == "in_channel"

    def test_leaderboard_shows_agent_count(self, slack_handler):
        agents = [_make_agent(f"a{i}", elo=1500 + i * 10) for i in range(5)]
        mock_elo_mod = MagicMock()
        mock_elo_mod.EloSystem.return_value.get_all_ratings.return_value = agents
        with patch.dict("sys.modules", {"aragora.ranking.elo": mock_elo_mod}):
            body = _body(slack_handler._command_leaderboard())
            blocks_text = json.dumps(body.get("blocks", []))
            assert "5" in blocks_text

    def test_leaderboard_includes_code_block(self, slack_handler):
        agents = [_make_agent("best", elo=2000, wins=50, losses=5)]
        mock_elo_mod = MagicMock()
        mock_elo_mod.EloSystem.return_value.get_all_ratings.return_value = agents
        with patch.dict("sys.modules", {"aragora.ranking.elo": mock_elo_mod}):
            body = _body(slack_handler._command_leaderboard())
            blocks_text = json.dumps(body.get("blocks", []))
            assert "```" in blocks_text

    def test_leaderboard_shows_win_loss(self, slack_handler):
        agents = [_make_agent("fighter", elo=1700, wins=8, losses=3)]
        mock_elo_mod = MagicMock()
        mock_elo_mod.EloSystem.return_value.get_all_ratings.return_value = agents
        with patch.dict("sys.modules", {"aragora.ranking.elo": mock_elo_mod}):
            body = _body(slack_handler._command_leaderboard())
            blocks_text = json.dumps(body.get("blocks", []))
            assert "8/3" in blocks_text

    def test_leaderboard_import_error(self, slack_handler):
        with patch.dict("sys.modules", {"aragora.ranking.elo": None}):
            body = _body(slack_handler._command_leaderboard())
            assert "unavailable" in body["text"].lower()

    def test_leaderboard_attribute_error(self, slack_handler):
        mock_elo_mod = MagicMock()
        mock_elo_mod.EloSystem.return_value.get_all_ratings.side_effect = AttributeError("bad")
        with patch.dict("sys.modules", {"aragora.ranking.elo": mock_elo_mod}):
            body = _body(slack_handler._command_leaderboard())
            assert "failed" in body["text"].lower()

    def test_leaderboard_value_error(self, slack_handler):
        mock_elo_mod = MagicMock()
        mock_elo_mod.EloSystem.return_value.get_all_ratings.side_effect = ValueError("bad data")
        with patch.dict("sys.modules", {"aragora.ranking.elo": mock_elo_mod}):
            body = _body(slack_handler._command_leaderboard())
            assert "failed" in body["text"].lower()


# ---------------------------------------------------------------------------
# _command_recent
# ---------------------------------------------------------------------------


class TestCommandRecent:
    """Tests for the recent subcommand."""

    def test_recent_db_unavailable(self, slack_handler, commands_module, monkeypatch):
        monkeypatch.setattr(commands_module, "get_debates_db", None)
        body = _body(slack_handler._command_recent())
        assert "failed" in body["text"].lower() or "unavailable" in body["text"].lower()

    def test_recent_db_returns_none(self, slack_handler, commands_module, monkeypatch):
        monkeypatch.setattr(commands_module, "get_debates_db", MagicMock(return_value=None))
        body = _body(slack_handler._command_recent())
        assert "not available" in body["text"].lower()

    def test_recent_db_no_list_method(self, slack_handler, commands_module, monkeypatch):
        mock_db = MagicMock(spec=[])
        monkeypatch.setattr(commands_module, "get_debates_db", MagicMock(return_value=mock_db))
        body = _body(slack_handler._command_recent())
        assert "not available" in body["text"].lower()

    def test_recent_empty(self, slack_handler, commands_module, monkeypatch):
        mock_db = MagicMock()
        mock_db.list.return_value = []
        monkeypatch.setattr(commands_module, "get_debates_db", MagicMock(return_value=mock_db))
        body = _body(slack_handler._command_recent())
        assert "no recent debates" in body["text"].lower()

    def test_recent_dict_format(self, slack_handler, commands_module, monkeypatch):
        debates = [
            {
                "task": "Recent topic one",
                "consensus_reached": True,
                "confidence": 0.92,
                "id": "d001",
                "created_at": "2026-01-15T10:00:00",
            },
        ]
        mock_db = MagicMock()
        mock_db.list.return_value = debates
        monkeypatch.setattr(commands_module, "get_debates_db", MagicMock(return_value=mock_db))
        result = slack_handler._command_recent()
        body = _body(result)
        assert "blocks" in body
        assert body["response_type"] == "ephemeral"

    def test_recent_object_format(self, slack_handler, commands_module, monkeypatch):
        obj = SimpleNamespace(
            task="Object debate", consensus_reached=False, confidence=0.4,
            id="obj-d1", created_at="2026-02-01T12:00:00",
        )
        mock_db = MagicMock()
        mock_db.list.return_value = [obj]
        monkeypatch.setattr(commands_module, "get_debates_db", MagicMock(return_value=mock_db))
        body = _body(slack_handler._command_recent())
        assert "blocks" in body

    def test_recent_includes_details_button(self, slack_handler, commands_module, monkeypatch):
        debates = [
            {
                "task": "Button test debate topic",
                "consensus_reached": True,
                "confidence": 0.8,
                "id": "btn123",
                "created_at": "2026-01-01",
            },
        ]
        mock_db = MagicMock()
        mock_db.list.return_value = debates
        monkeypatch.setattr(commands_module, "get_debates_db", MagicMock(return_value=mock_db))
        body = _body(slack_handler._command_recent())
        found = False
        for b in body.get("blocks", []):
            acc = b.get("accessory", {})
            if acc.get("action_id") == "view_details":
                found = True
                assert acc["value"] == "btn123"
        assert found

    def test_recent_truncates_long_topic(self, slack_handler, commands_module, monkeypatch):
        debates = [
            {
                "task": "A" * 100,
                "consensus_reached": True,
                "confidence": 0.5,
                "id": "trunc1",
                "created_at": "2026-01-01",
            },
        ]
        mock_db = MagicMock()
        mock_db.list.return_value = debates
        monkeypatch.setattr(commands_module, "get_debates_db", MagicMock(return_value=mock_db))
        body = _body(slack_handler._command_recent())
        blocks_text = json.dumps(body.get("blocks", []))
        assert "..." in blocks_text

    def test_recent_short_topic_no_ellipsis(self, slack_handler, commands_module, monkeypatch):
        debates = [
            {
                "task": "Short topic here",
                "consensus_reached": True,
                "confidence": 0.7,
                "id": "short1",
                "created_at": "2026-02-20",
            },
        ]
        mock_db = MagicMock()
        mock_db.list.return_value = debates
        monkeypatch.setattr(commands_module, "get_debates_db", MagicMock(return_value=mock_db))
        body = _body(slack_handler._command_recent())
        # Find the section with the topic -- should NOT have ...
        for b in body.get("blocks", []):
            if b.get("type") == "section" and "Short topic here" in b.get("text", {}).get("text", ""):
                assert "..." not in b["text"]["text"]

    def test_recent_runtime_error(self, slack_handler, commands_module, monkeypatch):
        mock_db = MagicMock()
        mock_db.list.side_effect = RuntimeError("db down")
        monkeypatch.setattr(commands_module, "get_debates_db", MagicMock(return_value=mock_db))
        body = _body(slack_handler._command_recent())
        assert "failed" in body["text"].lower()

    def test_recent_shows_debate_count(self, slack_handler, commands_module, monkeypatch):
        debates = [
            {"task": f"Topic {i}", "consensus_reached": True, "confidence": 0.5,
             "id": f"id{i}", "created_at": "2026-01-01"}
            for i in range(3)
        ]
        mock_db = MagicMock()
        mock_db.list.return_value = debates
        monkeypatch.setattr(commands_module, "get_debates_db", MagicMock(return_value=mock_db))
        body = _body(slack_handler._command_recent())
        blocks_text = json.dumps(body.get("blocks", []))
        assert "3" in blocks_text


# ---------------------------------------------------------------------------
# _command_debate
# ---------------------------------------------------------------------------


class TestCommandDebate:
    """Tests for the debate subcommand."""

    def test_debate_no_args(self, slack_handler):
        body = _body(slack_handler._command_debate("", "U1", "C1", "https://hooks.slack.com/x"))
        assert "provide a topic" in body["text"].lower()

    def test_debate_no_args_uses_mode_label(self, slack_handler):
        body = _body(slack_handler._command_debate(
            "", "U1", "C1", "https://hooks.slack.com/x",
            command_label="implement",
        ))
        assert "implement" in body["text"].lower()

    def test_debate_topic_too_short_9_chars(self, slack_handler):
        body = _body(slack_handler._command_debate("123456789", "U1", "C1", "https://hooks.slack.com/x"))
        assert "too short" in body["text"].lower()

    def test_debate_exactly_10_chars_accepted(self, slack_handler, commands_module, monkeypatch):
        monkeypatch.setattr(commands_module, "create_tracked_task", MagicMock())
        body = _body(slack_handler._command_debate("0123456789", "U1", "C1", "https://hooks.slack.com/x"))
        assert "too short" not in body.get("text", "").lower()

    def test_debate_topic_too_long_501_chars(self, slack_handler):
        body = _body(slack_handler._command_debate("x" * 501, "U1", "C1", "https://hooks.slack.com/x"))
        assert "too long" in body["text"].lower()

    def test_debate_exactly_500_chars_accepted(self, slack_handler, commands_module, monkeypatch):
        monkeypatch.setattr(commands_module, "create_tracked_task", MagicMock())
        body = _body(slack_handler._command_debate("x" * 500, "U1", "C1", "https://hooks.slack.com/x"))
        assert "too long" not in body.get("text", "").lower()

    def test_debate_valid_topic(self, slack_handler, commands_module, monkeypatch):
        monkeypatch.setattr(commands_module, "create_tracked_task", MagicMock())
        result = slack_handler._command_debate(
            "Should AI be regulated by governments?", "U1", "C1", "https://hooks.slack.com/x"
        )
        body = _body(result)
        assert body["response_type"] == "in_channel"
        assert "blocks" in body

    def test_debate_queues_async_task(self, slack_handler, commands_module, monkeypatch):
        mock_create = MagicMock()
        monkeypatch.setattr(commands_module, "create_tracked_task", mock_create)
        slack_handler._command_debate(
            "Should AI be regulated by governments?", "U1", "C1", "https://hooks.slack.com/x"
        )
        mock_create.assert_called_once()

    def test_debate_no_response_url_skips_task(self, slack_handler, commands_module, monkeypatch):
        mock_create = MagicMock()
        monkeypatch.setattr(commands_module, "create_tracked_task", mock_create)
        slack_handler._command_debate(
            "Should AI be regulated by governments?", "U1", "C1", ""
        )
        mock_create.assert_not_called()

    def test_debate_strips_quotes(self, slack_handler, commands_module, monkeypatch):
        monkeypatch.setattr(commands_module, "create_tracked_task", MagicMock())
        body = _body(slack_handler._command_debate(
            '"Should we adopt microservices architecture?"',
            "U1", "C1", "https://hooks.slack.com/x",
        ))
        blocks_text = json.dumps(body.get("blocks", []))
        assert "microservices" in blocks_text

    def test_debate_plan_mode_label(self, slack_handler, commands_module, monkeypatch):
        monkeypatch.setattr(commands_module, "create_tracked_task", MagicMock())
        body = _body(slack_handler._command_debate(
            "Refactor the entire backend codebase project",
            "U1", "C1", "https://hooks.slack.com/x",
            mode_label="plan",
        ))
        blocks_text = json.dumps(body.get("blocks", []))
        assert "plan" in blocks_text

    def test_debate_implement_mode_label(self, slack_handler, commands_module, monkeypatch):
        monkeypatch.setattr(commands_module, "create_tracked_task", MagicMock())
        body = _body(slack_handler._command_debate(
            "Build a new authentication system from scratch",
            "U1", "C1", "https://hooks.slack.com/x",
            mode_label="implementation plan",
        ))
        blocks_text = json.dumps(body.get("blocks", []))
        assert "implementation plan" in blocks_text

    def test_debate_includes_user_id_in_blocks(self, slack_handler, commands_module, monkeypatch):
        monkeypatch.setattr(commands_module, "create_tracked_task", MagicMock())
        body = _body(slack_handler._command_debate(
            "Important topic for discussion today",
            "UABCDEF", "C1", "https://hooks.slack.com/x",
        ))
        blocks_text = json.dumps(body.get("blocks", []))
        assert "UABCDEF" in blocks_text


# ---------------------------------------------------------------------------
# _command_gauntlet
# ---------------------------------------------------------------------------


class TestCommandGauntlet:
    """Tests for the gauntlet subcommand."""

    def test_gauntlet_no_args(self, slack_handler):
        body = _body(slack_handler._command_gauntlet("", "U1", "C1", "https://hooks.slack.com/x"))
        assert "provide a statement" in body["text"].lower()

    def test_gauntlet_too_short_9_chars(self, slack_handler):
        body = _body(slack_handler._command_gauntlet("123456789", "U1", "C1", "https://hooks.slack.com/x"))
        assert "too short" in body["text"].lower()

    def test_gauntlet_exactly_10_chars_accepted(self, slack_handler, commands_module, monkeypatch):
        monkeypatch.setattr(commands_module, "create_tracked_task", MagicMock())
        body = _body(slack_handler._command_gauntlet("0123456789", "U1", "C1", "https://hooks.slack.com/x"))
        assert "too short" not in body.get("text", "").lower()

    def test_gauntlet_too_long_1001_chars(self, slack_handler):
        body = _body(slack_handler._command_gauntlet("x" * 1001, "U1", "C1", "https://hooks.slack.com/x"))
        assert "too long" in body["text"].lower()

    def test_gauntlet_exactly_1000_chars_accepted(self, slack_handler, commands_module, monkeypatch):
        monkeypatch.setattr(commands_module, "create_tracked_task", MagicMock())
        body = _body(slack_handler._command_gauntlet("x" * 1000, "U1", "C1", "https://hooks.slack.com/x"))
        assert "too long" not in body.get("text", "").lower()

    def test_gauntlet_valid_statement(self, slack_handler, commands_module, monkeypatch):
        monkeypatch.setattr(commands_module, "create_tracked_task", MagicMock())
        result = slack_handler._command_gauntlet(
            "We should migrate to microservices architecture",
            "U1", "C1", "https://hooks.slack.com/x",
        )
        body = _body(result)
        assert body["response_type"] == "in_channel"
        assert "blocks" in body

    def test_gauntlet_queues_async_task(self, slack_handler, commands_module, monkeypatch):
        mock_create = MagicMock()
        monkeypatch.setattr(commands_module, "create_tracked_task", mock_create)
        slack_handler._command_gauntlet(
            "We should migrate to microservices architecture",
            "U1", "C1", "https://hooks.slack.com/resp",
        )
        mock_create.assert_called_once()

    def test_gauntlet_no_response_url_skips_task(self, slack_handler, commands_module, monkeypatch):
        mock_create = MagicMock()
        monkeypatch.setattr(commands_module, "create_tracked_task", mock_create)
        slack_handler._command_gauntlet(
            "We should migrate to microservices architecture",
            "U1", "C1", "",
        )
        mock_create.assert_not_called()

    def test_gauntlet_strips_quotes(self, slack_handler, commands_module, monkeypatch):
        monkeypatch.setattr(commands_module, "create_tracked_task", MagicMock())
        body = _body(slack_handler._command_gauntlet(
            '"We should adopt a monorepo strategy for our codebase"',
            "U1", "C1", "https://hooks.slack.com/x",
        ))
        blocks_text = json.dumps(body.get("blocks", []))
        assert "monorepo" in blocks_text

    def test_gauntlet_includes_user_id_in_blocks(self, slack_handler, commands_module, monkeypatch):
        monkeypatch.setattr(commands_module, "create_tracked_task", MagicMock())
        body = _body(slack_handler._command_gauntlet(
            "This organization should invest in AI safety research",
            "UTEST88", "C1", "https://hooks.slack.com/x",
        ))
        blocks_text = json.dumps(body.get("blocks", []))
        assert "UTEST88" in blocks_text


# ---------------------------------------------------------------------------
# _handle_slash_command dispatch
# ---------------------------------------------------------------------------


class TestHandleSlashCommandDispatch:
    """Tests for top-level slash command dispatcher routing."""

    def test_dispatch_no_text_shows_help(self, slack_handler, commands_module, monkeypatch):
        _bypass_limiters_and_audit(commands_module, monkeypatch)
        h = _make_handler_with_body("")
        body = _body(slack_handler._handle_slash_command(h))
        assert "Aragora Slash Commands" in body["text"]

    def test_dispatch_help(self, slack_handler, commands_module, monkeypatch):
        _bypass_limiters_and_audit(commands_module, monkeypatch)
        h = _make_handler_with_body("help")
        body = _body(slack_handler._handle_slash_command(h))
        assert "Aragora Slash Commands" in body["text"]

    def test_dispatch_status(self, slack_handler, commands_module, monkeypatch):
        _bypass_limiters_and_audit(commands_module, monkeypatch)
        slack_handler._command_status = MagicMock(
            return_value=slack_handler._slack_response("Status OK")
        )
        h = _make_handler_with_body("status")
        slack_handler._handle_slash_command(h)
        slack_handler._command_status.assert_called_once()

    def test_dispatch_debate(self, slack_handler, commands_module, monkeypatch):
        _bypass_limiters_and_audit(commands_module, monkeypatch)
        monkeypatch.setattr(commands_module, "create_tracked_task", MagicMock())
        h = _make_handler_with_body('debate "Should AI be regulated?"')
        body = _body(slack_handler._handle_slash_command(h))
        assert body["response_type"] == "in_channel"

    def test_dispatch_plan(self, slack_handler, commands_module, monkeypatch):
        _bypass_limiters_and_audit(commands_module, monkeypatch)
        monkeypatch.setattr(commands_module, "create_tracked_task", MagicMock())
        h = _make_handler_with_body('plan "Refactor the entire backend codebase"')
        body = _body(slack_handler._handle_slash_command(h))
        assert body["response_type"] == "in_channel"

    def test_dispatch_implement(self, slack_handler, commands_module, monkeypatch):
        _bypass_limiters_and_audit(commands_module, monkeypatch)
        monkeypatch.setattr(commands_module, "create_tracked_task", MagicMock())
        h = _make_handler_with_body('implement "Build a new authentication system"')
        body = _body(slack_handler._handle_slash_command(h))
        assert body["response_type"] == "in_channel"

    def test_dispatch_agents(self, slack_handler, commands_module, monkeypatch):
        _bypass_limiters_and_audit(commands_module, monkeypatch)
        slack_handler._command_agents = MagicMock(
            return_value=slack_handler._slack_response("Agents list")
        )
        h = _make_handler_with_body("agents")
        slack_handler._handle_slash_command(h)
        slack_handler._command_agents.assert_called_once()

    def test_dispatch_ask(self, slack_handler, commands_module, monkeypatch):
        _bypass_limiters_and_audit(commands_module, monkeypatch)
        monkeypatch.setattr(commands_module, "create_tracked_task", MagicMock())
        h = _make_handler_with_body('ask "What is the capital of France?"')
        body = _body(slack_handler._handle_slash_command(h))
        assert body["response_type"] == "in_channel"

    def test_dispatch_search(self, slack_handler, commands_module, monkeypatch):
        _bypass_limiters_and_audit(commands_module, monkeypatch)
        slack_handler._command_search = MagicMock(
            return_value=slack_handler._slack_response("Results")
        )
        h = _make_handler_with_body("search machine learning")
        slack_handler._handle_slash_command(h)
        slack_handler._command_search.assert_called_once_with("machine learning")

    def test_dispatch_leaderboard(self, slack_handler, commands_module, monkeypatch):
        _bypass_limiters_and_audit(commands_module, monkeypatch)
        slack_handler._command_leaderboard = MagicMock(
            return_value=slack_handler._slack_response("Leaderboard")
        )
        h = _make_handler_with_body("leaderboard")
        slack_handler._handle_slash_command(h)
        slack_handler._command_leaderboard.assert_called_once()

    def test_dispatch_recent(self, slack_handler, commands_module, monkeypatch):
        _bypass_limiters_and_audit(commands_module, monkeypatch)
        slack_handler._command_recent = MagicMock(
            return_value=slack_handler._slack_response("Recent")
        )
        h = _make_handler_with_body("recent")
        slack_handler._handle_slash_command(h)
        slack_handler._command_recent.assert_called_once()

    def test_dispatch_gauntlet(self, slack_handler, commands_module, monkeypatch):
        _bypass_limiters_and_audit(commands_module, monkeypatch)
        monkeypatch.setattr(commands_module, "create_tracked_task", MagicMock())
        h = _make_handler_with_body('gauntlet "We should adopt microservices"')
        body = _body(slack_handler._handle_slash_command(h))
        assert body["response_type"] == "in_channel"

    def test_dispatch_unknown_command(self, slack_handler, commands_module, monkeypatch):
        _bypass_limiters_and_audit(commands_module, monkeypatch)
        h = _make_handler_with_body("foobar")
        body = _body(slack_handler._handle_slash_command(h))
        assert "unknown command" in body["text"].lower()
        assert "foobar" in body["text"].lower()

    def test_dispatch_case_insensitive(self, slack_handler, commands_module, monkeypatch):
        _bypass_limiters_and_audit(commands_module, monkeypatch)
        h = _make_handler_with_body("HELP")
        body = _body(slack_handler._handle_slash_command(h))
        assert "Aragora Slash Commands" in body["text"]

    def test_dispatch_mixed_case(self, slack_handler, commands_module, monkeypatch):
        _bypass_limiters_and_audit(commands_module, monkeypatch)
        h = _make_handler_with_body("Status")
        slack_handler._command_status = MagicMock(
            return_value=slack_handler._slack_response("Status OK")
        )
        slack_handler._handle_slash_command(h)
        slack_handler._command_status.assert_called_once()

    def test_dispatch_with_extra_whitespace(self, slack_handler, commands_module, monkeypatch):
        _bypass_limiters_and_audit(commands_module, monkeypatch)
        h = MagicMock()
        form = {
            "command": "/aragora",
            "text": "  help  ",
            "user_id": "U1",
            "channel_id": "C1",
            "response_url": "",
        }
        h._slack_body = urlencode(form)
        h._slack_workspace = None
        h._slack_team_id = None
        body = _body(slack_handler._handle_slash_command(h))
        assert "Aragora Slash Commands" in body["text"]


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


class TestRateLimiting:
    """Tests for workspace and user rate limiting."""

    def test_workspace_rate_limited(self, slack_handler, commands_module, monkeypatch):
        ws_limiter = MagicMock()
        ws_limiter.allow.return_value = SimpleNamespace(allowed=False, retry_after=30)
        monkeypatch.setattr(commands_module, "_get_workspace_rate_limiter", lambda: ws_limiter)
        monkeypatch.setattr(commands_module, "_get_user_rate_limiter", lambda: None)
        monkeypatch.setattr(commands_module, "_get_audit_logger", lambda: None)
        h = _make_handler_with_body("help")
        body = _body(slack_handler._handle_slash_command(h))
        assert "too quickly" in body["text"].lower()
        assert "30" in body["text"]

    def test_user_rate_limited(self, slack_handler, commands_module, monkeypatch):
        user_limiter = MagicMock()
        user_limiter.allow.return_value = SimpleNamespace(allowed=False, retry_after=15)
        monkeypatch.setattr(commands_module, "_get_workspace_rate_limiter", lambda: None)
        monkeypatch.setattr(commands_module, "_get_user_rate_limiter", lambda: user_limiter)
        monkeypatch.setattr(commands_module, "_get_audit_logger", lambda: None)
        h = _make_handler_with_body("help")
        body = _body(slack_handler._handle_slash_command(h))
        assert "too quickly" in body["text"].lower()
        assert "15" in body["text"]

    def test_workspace_rate_limit_audit_logged(self, slack_handler, commands_module, monkeypatch):
        ws_limiter = MagicMock()
        ws_limiter.allow.return_value = SimpleNamespace(allowed=False, retry_after=10)
        mock_audit = MagicMock()
        monkeypatch.setattr(commands_module, "_get_workspace_rate_limiter", lambda: ws_limiter)
        monkeypatch.setattr(commands_module, "_get_user_rate_limiter", lambda: None)
        monkeypatch.setattr(commands_module, "_get_audit_logger", lambda: mock_audit)
        h = _make_handler_with_body("help")
        slack_handler._handle_slash_command(h)
        mock_audit.log_rate_limit.assert_called_once()
        assert mock_audit.log_rate_limit.call_args.kwargs["limit_type"] == "workspace"

    def test_user_rate_limit_audit_logged(self, slack_handler, commands_module, monkeypatch):
        user_limiter = MagicMock()
        user_limiter.allow.return_value = SimpleNamespace(allowed=False, retry_after=5)
        mock_audit = MagicMock()
        monkeypatch.setattr(commands_module, "_get_workspace_rate_limiter", lambda: None)
        monkeypatch.setattr(commands_module, "_get_user_rate_limiter", lambda: user_limiter)
        monkeypatch.setattr(commands_module, "_get_audit_logger", lambda: mock_audit)
        h = _make_handler_with_body("help")
        slack_handler._handle_slash_command(h)
        mock_audit.log_rate_limit.assert_called_once()
        assert mock_audit.log_rate_limit.call_args.kwargs["limit_type"] == "user"

    def test_both_limiters_allow_passes_through(self, slack_handler, commands_module, monkeypatch):
        ws_limiter = MagicMock()
        ws_limiter.allow.return_value = SimpleNamespace(allowed=True, retry_after=0)
        user_limiter = MagicMock()
        user_limiter.allow.return_value = SimpleNamespace(allowed=True, retry_after=0)
        monkeypatch.setattr(commands_module, "_get_workspace_rate_limiter", lambda: ws_limiter)
        monkeypatch.setattr(commands_module, "_get_user_rate_limiter", lambda: user_limiter)
        monkeypatch.setattr(commands_module, "_get_audit_logger", lambda: None)
        h = _make_handler_with_body("help")
        body = _body(slack_handler._handle_slash_command(h))
        assert "Aragora Slash Commands" in body["text"]

    def test_no_team_id_skips_workspace_rate_limit(self, slack_handler, commands_module, monkeypatch):
        ws_limiter = MagicMock()
        monkeypatch.setattr(commands_module, "_get_workspace_rate_limiter", lambda: ws_limiter)
        monkeypatch.setattr(commands_module, "_get_user_rate_limiter", lambda: None)
        monkeypatch.setattr(commands_module, "_get_audit_logger", lambda: None)
        h = _make_handler_with_body("help", team_id="")
        h._slack_team_id = None
        slack_handler._handle_slash_command(h)
        ws_limiter.allow.assert_not_called()

    def test_workspace_rate_limit_response_is_ephemeral(self, slack_handler, commands_module, monkeypatch):
        ws_limiter = MagicMock()
        ws_limiter.allow.return_value = SimpleNamespace(allowed=False, retry_after=5)
        monkeypatch.setattr(commands_module, "_get_workspace_rate_limiter", lambda: ws_limiter)
        monkeypatch.setattr(commands_module, "_get_user_rate_limiter", lambda: None)
        monkeypatch.setattr(commands_module, "_get_audit_logger", lambda: None)
        h = _make_handler_with_body("help")
        body = _body(slack_handler._handle_slash_command(h))
        assert body["response_type"] == "ephemeral"

    def test_user_rate_limit_response_is_ephemeral(self, slack_handler, commands_module, monkeypatch):
        user_limiter = MagicMock()
        user_limiter.allow.return_value = SimpleNamespace(allowed=False, retry_after=5)
        monkeypatch.setattr(commands_module, "_get_workspace_rate_limiter", lambda: None)
        monkeypatch.setattr(commands_module, "_get_user_rate_limiter", lambda: user_limiter)
        monkeypatch.setattr(commands_module, "_get_audit_logger", lambda: None)
        h = _make_handler_with_body("help")
        body = _body(slack_handler._handle_slash_command(h))
        assert body["response_type"] == "ephemeral"


# ---------------------------------------------------------------------------
# Audit logging
# ---------------------------------------------------------------------------


class TestAuditLogging:
    """Tests for audit logging on command success and error."""

    def test_audit_logged_on_success(self, slack_handler, commands_module, monkeypatch):
        mock_audit = MagicMock()
        monkeypatch.setattr(commands_module, "_get_workspace_rate_limiter", lambda: None)
        monkeypatch.setattr(commands_module, "_get_user_rate_limiter", lambda: None)
        monkeypatch.setattr(commands_module, "_get_audit_logger", lambda: mock_audit)
        h = _make_handler_with_body("help")
        slack_handler._handle_slash_command(h)
        mock_audit.log_command.assert_called_once()
        assert mock_audit.log_command.call_args.kwargs["result"] == "success"

    def test_audit_logged_on_error(self, slack_handler, commands_module, monkeypatch):
        mock_audit = MagicMock()
        monkeypatch.setattr(commands_module, "_get_workspace_rate_limiter", lambda: None)
        monkeypatch.setattr(commands_module, "_get_user_rate_limiter", lambda: None)
        monkeypatch.setattr(commands_module, "_get_audit_logger", lambda: mock_audit)
        slack_handler._command_status = MagicMock(side_effect=ValueError("test error"))
        h = _make_handler_with_body("status")
        slack_handler._handle_slash_command(h)
        error_calls = [
            c for c in mock_audit.log_command.call_args_list
            if c.kwargs.get("result") == "error"
        ]
        assert len(error_calls) == 1

    def test_no_audit_when_logger_none(self, slack_handler, commands_module, monkeypatch):
        monkeypatch.setattr(commands_module, "_get_workspace_rate_limiter", lambda: None)
        monkeypatch.setattr(commands_module, "_get_user_rate_limiter", lambda: None)
        monkeypatch.setattr(commands_module, "_get_audit_logger", lambda: None)
        h = _make_handler_with_body("help")
        result = slack_handler._handle_slash_command(h)
        assert _status(result) == 200

    def test_audit_includes_response_time(self, slack_handler, commands_module, monkeypatch):
        mock_audit = MagicMock()
        monkeypatch.setattr(commands_module, "_get_workspace_rate_limiter", lambda: None)
        monkeypatch.setattr(commands_module, "_get_user_rate_limiter", lambda: None)
        monkeypatch.setattr(commands_module, "_get_audit_logger", lambda: mock_audit)
        h = _make_handler_with_body("help")
        slack_handler._handle_slash_command(h)
        kw = mock_audit.log_command.call_args.kwargs
        assert "response_time_ms" in kw
        assert kw["response_time_ms"] >= 0

    def test_audit_includes_channel_id(self, slack_handler, commands_module, monkeypatch):
        mock_audit = MagicMock()
        monkeypatch.setattr(commands_module, "_get_workspace_rate_limiter", lambda: None)
        monkeypatch.setattr(commands_module, "_get_user_rate_limiter", lambda: None)
        monkeypatch.setattr(commands_module, "_get_audit_logger", lambda: mock_audit)
        h = _make_handler_with_body("help", channel_id="CTEST")
        slack_handler._handle_slash_command(h)
        kw = mock_audit.log_command.call_args.kwargs
        assert kw["channel_id"] == "CTEST"

    def test_audit_error_includes_error_message(self, slack_handler, commands_module, monkeypatch):
        mock_audit = MagicMock()
        monkeypatch.setattr(commands_module, "_get_workspace_rate_limiter", lambda: None)
        monkeypatch.setattr(commands_module, "_get_user_rate_limiter", lambda: None)
        monkeypatch.setattr(commands_module, "_get_audit_logger", lambda: mock_audit)
        slack_handler._command_status = MagicMock(side_effect=ValueError("boom"))
        h = _make_handler_with_body("status")
        slack_handler._handle_slash_command(h)
        error_calls = [c for c in mock_audit.log_command.call_args_list if c.kwargs.get("result") == "error"]
        assert error_calls[0].kwargs["error"] == "Command execution failed"


# ---------------------------------------------------------------------------
# Plan and implement modes via dispatcher
# ---------------------------------------------------------------------------


class TestPlanAndImplementModes:
    """Tests for plan and implement subcommands via the dispatcher."""

    def test_plan_passes_decision_integrity(self, slack_handler, commands_module, monkeypatch):
        _bypass_limiters_and_audit(commands_module, monkeypatch)
        monkeypatch.setattr(commands_module, "create_tracked_task", MagicMock())

        captured = {}
        original_debate = slack_handler._command_debate

        def spy_debate(*args, **kwargs):
            captured.update(kwargs)
            return original_debate(*args, **kwargs)

        slack_handler._command_debate = spy_debate
        h = _make_handler_with_body('plan "Refactor the backend to be much better"')
        slack_handler._handle_slash_command(h)
        di = captured.get("decision_integrity")
        assert di is not None
        assert di["include_plan"] is True
        assert di["include_context"] is False
        assert di["include_receipt"] is True
        assert captured["mode_label"] == "plan"

    def test_implement_passes_execution_mode(self, slack_handler, commands_module, monkeypatch):
        _bypass_limiters_and_audit(commands_module, monkeypatch)
        monkeypatch.setattr(commands_module, "create_tracked_task", MagicMock())

        captured = {}
        original_debate = slack_handler._command_debate

        def spy_debate(*args, **kwargs):
            captured.update(kwargs)
            return original_debate(*args, **kwargs)

        slack_handler._command_debate = spy_debate
        h = _make_handler_with_body('implement "Build new auth system for the app"')
        slack_handler._handle_slash_command(h)
        di = captured.get("decision_integrity", {})
        assert di.get("execution_mode") == "execute"
        assert di.get("execution_engine") == "hybrid"
        assert di.get("include_context") is True
        assert captured.get("mode_label") == "implementation plan"
        assert captured.get("command_label") == "implement"

    def test_plan_decision_integrity_requested_by(self, slack_handler, commands_module, monkeypatch):
        _bypass_limiters_and_audit(commands_module, monkeypatch)
        monkeypatch.setattr(commands_module, "create_tracked_task", MagicMock())

        captured = {}
        original_debate = slack_handler._command_debate

        def spy_debate(*args, **kwargs):
            captured.update(kwargs)
            return original_debate(*args, **kwargs)

        slack_handler._command_debate = spy_debate
        h = _make_handler_with_body(
            'plan "Plan some important project improvement"',
            user_id="UPLANNER",
        )
        slack_handler._handle_slash_command(h)
        di = captured.get("decision_integrity", {})
        assert di.get("requested_by") == "slack:UPLANNER"


# ---------------------------------------------------------------------------
# Error handling in _handle_slash_command
# ---------------------------------------------------------------------------


class TestSlashCommandErrorHandling:
    """Tests for error handling in the top-level slash command dispatcher."""

    @pytest.mark.parametrize("exc_cls", [
        ValueError, KeyError, TypeError, RuntimeError, OSError, ConnectionError,
    ])
    def test_caught_exceptions(self, slack_handler, commands_module, monkeypatch, exc_cls):
        _bypass_limiters_and_audit(commands_module, monkeypatch)
        slack_handler._command_status = MagicMock(side_effect=exc_cls("test"))
        h = _make_handler_with_body("status")
        result = slack_handler._handle_slash_command(h)
        body = _body(result)
        assert "error" in body["text"].lower()

    def test_error_response_is_ephemeral(self, slack_handler, commands_module, monkeypatch):
        _bypass_limiters_and_audit(commands_module, monkeypatch)
        slack_handler._command_status = MagicMock(side_effect=ValueError("fail"))
        h = _make_handler_with_body("status")
        body = _body(slack_handler._handle_slash_command(h))
        assert body["response_type"] == "ephemeral"

    def test_error_audit_includes_response_time(self, slack_handler, commands_module, monkeypatch):
        mock_audit = MagicMock()
        monkeypatch.setattr(commands_module, "_get_workspace_rate_limiter", lambda: None)
        monkeypatch.setattr(commands_module, "_get_user_rate_limiter", lambda: None)
        monkeypatch.setattr(commands_module, "_get_audit_logger", lambda: mock_audit)
        slack_handler._command_status = MagicMock(side_effect=ValueError("fail"))
        h = _make_handler_with_body("status")
        slack_handler._handle_slash_command(h)
        kw = mock_audit.log_command.call_args.kwargs
        assert "response_time_ms" in kw
        assert kw["response_time_ms"] >= 0


# ---------------------------------------------------------------------------
# _answer_question_async
# ---------------------------------------------------------------------------


class TestAnswerQuestionAsync:
    """Tests for the async question answering flow."""

    def _make_mock_pool(self, session):
        mock_pool = MagicMock()
        mock_pool.get_session.return_value.__aenter__ = AsyncMock(return_value=session)
        mock_pool.get_session.return_value.__aexit__ = AsyncMock(return_value=None)
        return mock_pool

    @pytest.mark.asyncio
    async def test_happy_path(self, slack_handler):
        mock_session = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"answer": "Paris is the capital of France"}
        mock_session.post.return_value = mock_response

        with patch(
            "aragora.server.http_client_pool.get_http_pool",
            return_value=self._make_mock_pool(mock_session),
            create=True,
        ):
            slack_handler._post_to_response_url = AsyncMock()
            await slack_handler._answer_question_async(
                "What is the capital of France?",
                "https://hooks.slack.com/resp",
                "U1", "C1",
            )
            slack_handler._post_to_response_url.assert_called_once()
            payload = slack_handler._post_to_response_url.call_args[0][1]
            assert "Paris" in json.dumps(payload.get("blocks", []))

    @pytest.mark.asyncio
    async def test_fallback_to_debate(self, slack_handler):
        mock_session = AsyncMock()
        resp1 = MagicMock()
        resp1.status_code = 200
        resp1.json.return_value = {"answer": None}
        resp2 = MagicMock()
        resp2.status_code = 200
        resp2.json.return_value = {"final_answer": "Debate answer here"}
        mock_session.post.side_effect = [resp1, resp2]

        with patch(
            "aragora.server.http_client_pool.get_http_pool",
            return_value=self._make_mock_pool(mock_session),
            create=True,
        ):
            slack_handler._post_to_response_url = AsyncMock()
            await slack_handler._answer_question_async(
                "Explain deep learning", "https://hooks.slack.com/resp", "U1", "C1"
            )
            slack_handler._post_to_response_url.assert_called_once()
            payload = slack_handler._post_to_response_url.call_args[0][1]
            assert "Debate answer" in json.dumps(payload.get("blocks", []))

    @pytest.mark.asyncio
    async def test_quick_answer_non_200_falls_back(self, slack_handler):
        mock_session = AsyncMock()
        resp1 = MagicMock()
        resp1.status_code = 500
        resp1.json.return_value = {"error": "service down"}
        resp2 = MagicMock()
        resp2.status_code = 201
        resp2.json.return_value = {"final_answer": "Fallback debate answer"}
        mock_session.post.side_effect = [resp1, resp2]

        with patch(
            "aragora.server.http_client_pool.get_http_pool",
            return_value=self._make_mock_pool(mock_session),
            create=True,
        ):
            slack_handler._post_to_response_url = AsyncMock()
            await slack_handler._answer_question_async(
                "Some important question here", "https://hooks.slack.com/resp", "U1", "C1"
            )
            slack_handler._post_to_response_url.assert_called_once()

    @pytest.mark.asyncio
    async def test_network_error_posts_failure(self, slack_handler):
        mock_pool = MagicMock()
        mock_pool.get_session.return_value.__aenter__ = AsyncMock(
            side_effect=OSError("network error")
        )
        mock_pool.get_session.return_value.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "aragora.server.http_client_pool.get_http_pool",
            return_value=mock_pool,
            create=True,
        ):
            slack_handler._post_to_response_url = AsyncMock()
            await slack_handler._answer_question_async(
                "Some question here", "https://hooks.slack.com/resp", "U1", "C1"
            )
            payload = slack_handler._post_to_response_url.call_args[0][1]
            assert "failed" in payload["text"].lower()

    @pytest.mark.asyncio
    async def test_debate_fallback_also_fails(self, slack_handler):
        """When both quick-answer and debate API fail, posts generic answer."""
        mock_session = AsyncMock()
        resp1 = MagicMock()
        resp1.status_code = 200
        resp1.json.return_value = {"answer": None}
        resp2 = MagicMock()
        resp2.status_code = 500
        resp2.json.return_value = {"error": "internal"}
        mock_session.post.side_effect = [resp1, resp2]

        with patch(
            "aragora.server.http_client_pool.get_http_pool",
            return_value=self._make_mock_pool(mock_session),
            create=True,
        ):
            slack_handler._post_to_response_url = AsyncMock()
            await slack_handler._answer_question_async(
                "Some question here", "https://hooks.slack.com/resp", "U1", "C1"
            )
            payload = slack_handler._post_to_response_url.call_args[0][1]
            blocks_text = json.dumps(payload.get("blocks", []))
            assert "Unable to generate answer" in blocks_text


# ---------------------------------------------------------------------------
# _run_gauntlet_async
# ---------------------------------------------------------------------------


class TestRunGauntletAsync:
    """Tests for the async gauntlet execution flow."""

    def _make_mock_pool(self, session):
        mock_pool = MagicMock()
        mock_pool.get_session.return_value.__aenter__ = AsyncMock(return_value=session)
        mock_pool.get_session.return_value.__aexit__ = AsyncMock(return_value=None)
        return mock_pool

    @pytest.mark.asyncio
    async def test_happy_path_passed(self, slack_handler):
        mock_session = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "run_id": "gauntlet-001",
            "score": 0.8,
            "passed": True,
            "vulnerabilities": [],
        }
        mock_session.post.return_value = mock_response

        with patch(
            "aragora.server.http_client_pool.get_http_pool",
            return_value=self._make_mock_pool(mock_session),
            create=True,
        ):
            slack_handler._post_to_response_url = AsyncMock()
            await slack_handler._run_gauntlet_async(
                "Test statement for validation",
                "https://hooks.slack.com/resp", "U1", "C1",
            )
            payload = slack_handler._post_to_response_url.call_args[0][1]
            assert "blocks" in payload

    @pytest.mark.asyncio
    async def test_with_vulnerabilities(self, slack_handler):
        mock_session = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "run_id": "gauntlet-002",
            "score": 0.3,
            "passed": False,
            "vulnerabilities": [
                {"description": "Lacks evidence"},
                {"description": "Logical fallacy detected"},
            ],
        }
        mock_session.post.return_value = mock_response

        with patch(
            "aragora.server.http_client_pool.get_http_pool",
            return_value=self._make_mock_pool(mock_session),
            create=True,
        ):
            slack_handler._post_to_response_url = AsyncMock()
            await slack_handler._run_gauntlet_async(
                "Weak statement here!", "https://hooks.slack.com/resp", "U1", "C1",
            )
            payload = slack_handler._post_to_response_url.call_args[0][1]
            blocks_text = json.dumps(payload.get("blocks", []))
            assert "Lacks evidence" in blocks_text
            assert "Logical fallacy" in blocks_text

    @pytest.mark.asyncio
    async def test_more_than_5_vulnerabilities_truncated(self, slack_handler):
        mock_session = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        vulns = [{"description": f"Issue {i}"} for i in range(8)]
        mock_response.json.return_value = {
            "run_id": "gauntlet-003",
            "score": 0.1,
            "passed": False,
            "vulnerabilities": vulns,
        }
        mock_session.post.return_value = mock_response

        with patch(
            "aragora.server.http_client_pool.get_http_pool",
            return_value=self._make_mock_pool(mock_session),
            create=True,
        ):
            slack_handler._post_to_response_url = AsyncMock()
            await slack_handler._run_gauntlet_async(
                "Statement with many issues",
                "https://hooks.slack.com/resp", "U1", "C1",
            )
            payload = slack_handler._post_to_response_url.call_args[0][1]
            blocks_text = json.dumps(payload.get("blocks", []))
            assert "3 more" in blocks_text

    @pytest.mark.asyncio
    async def test_api_error(self, slack_handler):
        mock_session = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.json.return_value = {"error": "Internal server error"}
        mock_session.post.return_value = mock_response

        with patch(
            "aragora.server.http_client_pool.get_http_pool",
            return_value=self._make_mock_pool(mock_session),
            create=True,
        ):
            slack_handler._post_to_response_url = AsyncMock()
            await slack_handler._run_gauntlet_async(
                "Some statement here",
                "https://hooks.slack.com/resp", "U1", "C1",
            )
            payload = slack_handler._post_to_response_url.call_args[0][1]
            assert "failed" in payload["text"].lower()

    @pytest.mark.asyncio
    async def test_network_error(self, slack_handler):
        mock_pool = MagicMock()
        mock_pool.get_session.return_value.__aenter__ = AsyncMock(
            side_effect=OSError("connection refused")
        )
        mock_pool.get_session.return_value.__aexit__ = AsyncMock(return_value=None)

        with patch(
            "aragora.server.http_client_pool.get_http_pool",
            return_value=mock_pool,
            create=True,
        ):
            slack_handler._post_to_response_url = AsyncMock()
            await slack_handler._run_gauntlet_async(
                "Some statement here",
                "https://hooks.slack.com/resp", "U1", "C1",
            )
            payload = slack_handler._post_to_response_url.call_args[0][1]
            assert "failed" in payload["text"].lower()

    @pytest.mark.asyncio
    async def test_includes_run_id_in_blocks(self, slack_handler):
        mock_session = AsyncMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "run_id": "run-xyz-456",
            "score": 0.9,
            "passed": True,
            "vulnerabilities": [],
        }
        mock_session.post.return_value = mock_response

        with patch(
            "aragora.server.http_client_pool.get_http_pool",
            return_value=self._make_mock_pool(mock_session),
            create=True,
        ):
            slack_handler._post_to_response_url = AsyncMock()
            await slack_handler._run_gauntlet_async(
                "Valid statement for gauntlet run",
                "https://hooks.slack.com/resp", "U1", "C1",
            )
            payload = slack_handler._post_to_response_url.call_args[0][1]
            blocks_text = json.dumps(payload.get("blocks", []))
            assert "run-xyz-456" in blocks_text


# ---------------------------------------------------------------------------
# _create_debate_async
# ---------------------------------------------------------------------------


class TestCreateDebateAsync:
    """Tests for the async debate creation flow."""

    @pytest.mark.asyncio
    async def test_posts_starting_message_without_bot_token(
        self, slack_handler, commands_module, monkeypatch
    ):
        monkeypatch.setattr(commands_module, "SLACK_BOT_TOKEN", None)
        monkeypatch.setattr(commands_module, "create_tracked_task", MagicMock())

        try:
            from aragora.server.handlers.social._slack_impl import blocks as blocks_mod
            monkeypatch.setattr(blocks_mod, "SLACK_BOT_TOKEN", None)
        except (ImportError, AttributeError):
            pass

        mock_result = MagicMock()
        mock_result.consensus_reached = True
        mock_result.confidence = 0.9
        mock_result.rounds_used = 3
        mock_result.participants = ["anthropic-api", "openai-api"]
        mock_result.final_answer = "The agents concluded that..."
        mock_result.winner = None
        mock_result.id = "test-debate-id"

        mock_arena = MagicMock()
        mock_arena.run = AsyncMock(return_value=mock_result)

        slack_handler._post_to_response_url = AsyncMock()
        slack_handler._update_debate_status = MagicMock()

        with patch(
            "aragora.server.handlers.social._slack_impl.commands.register_debate_origin",
            create=True,
        ):
            with patch("aragora.Environment", MagicMock(), create=True):
                with patch(
                    "aragora.DebateProtocol",
                    return_value=MagicMock(rounds=3),
                    create=True,
                ):
                    with patch("aragora.Arena", create=True) as mock_arena_cls:
                        mock_arena_cls.from_env.return_value = mock_arena
                        with patch(
                            "aragora.agents.get_agents_by_names",
                            return_value=["a1", "a2"],
                            create=True,
                        ):
                            with patch(
                                "aragora.server.handlers.social._slack_impl.commands.maybe_emit_decision_integrity",
                                new_callable=AsyncMock,
                                create=True,
                            ):
                                await slack_handler._create_debate_async(
                                    "Should AI be regulated?",
                                    "https://hooks.slack.com/resp",
                                    "U1", "C1", "WS1",
                                )

        assert slack_handler._post_to_response_url.call_count >= 1

    @pytest.mark.asyncio
    async def test_no_agents_posts_failure(self, slack_handler, commands_module, monkeypatch):
        monkeypatch.setattr(commands_module, "SLACK_BOT_TOKEN", None)
        monkeypatch.setattr(commands_module, "create_tracked_task", MagicMock())

        try:
            from aragora.server.handlers.social._slack_impl import blocks as blocks_mod
            monkeypatch.setattr(blocks_mod, "SLACK_BOT_TOKEN", None)
        except (ImportError, AttributeError):
            pass

        slack_handler._post_to_response_url = AsyncMock()
        slack_handler._update_debate_status = MagicMock()

        with patch(
            "aragora.server.handlers.social._slack_impl.commands.register_debate_origin",
            create=True,
        ):
            with patch("aragora.Environment", MagicMock(), create=True):
                with patch(
                    "aragora.DebateProtocol",
                    return_value=MagicMock(rounds=3),
                    create=True,
                ):
                    with patch("aragora.Arena", create=True):
                        with patch(
                            "aragora.agents.get_agents_by_names",
                            return_value=[],
                            create=True,
                        ):
                            await slack_handler._create_debate_async(
                                "Should AI be regulated?",
                                "https://hooks.slack.com/resp",
                                "U1", "C1", "WS1",
                            )

        # Should have posted "No agents available" error
        found_no_agents = False
        for call in slack_handler._post_to_response_url.call_args_list:
            payload = call[0][1]
            if "no agents" in payload.get("text", "").lower():
                found_no_agents = True
        assert found_no_agents
        slack_handler._update_debate_status.assert_called()

    @pytest.mark.asyncio
    async def test_arena_error_posts_failure(self, slack_handler, commands_module, monkeypatch):
        monkeypatch.setattr(commands_module, "SLACK_BOT_TOKEN", None)
        monkeypatch.setattr(commands_module, "create_tracked_task", MagicMock())

        try:
            from aragora.server.handlers.social._slack_impl import blocks as blocks_mod
            monkeypatch.setattr(blocks_mod, "SLACK_BOT_TOKEN", None)
        except (ImportError, AttributeError):
            pass

        mock_arena = MagicMock()
        mock_arena.run = AsyncMock(side_effect=RuntimeError("debate engine crashed"))

        slack_handler._post_to_response_url = AsyncMock()
        slack_handler._update_debate_status = MagicMock()

        with patch(
            "aragora.server.handlers.social._slack_impl.commands.register_debate_origin",
            create=True,
        ):
            with patch("aragora.Environment", MagicMock(), create=True):
                with patch(
                    "aragora.DebateProtocol",
                    return_value=MagicMock(rounds=3),
                    create=True,
                ):
                    with patch("aragora.Arena", create=True) as mock_arena_cls:
                        mock_arena_cls.from_env.return_value = mock_arena
                        with patch(
                            "aragora.agents.get_agents_by_names",
                            return_value=["a1", "a2"],
                            create=True,
                        ):
                            await slack_handler._create_debate_async(
                                "Should AI be regulated?",
                                "https://hooks.slack.com/resp",
                                "U1", "C1", "WS1",
                            )

        # Should have posted error message
        found_error = False
        for call in slack_handler._post_to_response_url.call_args_list:
            payload = call[0][1]
            if "failed" in payload.get("text", "").lower():
                found_error = True
        assert found_error
        # Should update debate status to failed
        status_calls = [
            c for c in slack_handler._update_debate_status.call_args_list
            if c[0][1] == "failed"
        ]
        assert len(status_calls) >= 1


# ---------------------------------------------------------------------------
# _update_debate_status
# ---------------------------------------------------------------------------


class TestUpdateDebateStatus:
    """Tests for _update_debate_status."""

    def test_store_available(self, slack_handler):
        mock_store = MagicMock()
        with patch(
            "aragora.storage.slack_debate_store.get_slack_debate_store",
            return_value=mock_store,
            create=True,
        ):
            slack_handler._update_debate_status("d1", "completed", receipt_id="r1")
            mock_store.update_status.assert_called_once_with(
                "d1", "completed", receipt_id="r1", error_message=None
            )

    def test_store_unavailable(self, slack_handler):
        with patch.dict("sys.modules", {"aragora.storage.slack_debate_store": None}):
            slack_handler._update_debate_status("d1", "failed", error="some error")

    def test_store_with_error_message(self, slack_handler):
        mock_store = MagicMock()
        with patch(
            "aragora.storage.slack_debate_store.get_slack_debate_store",
            return_value=mock_store,
            create=True,
        ):
            slack_handler._update_debate_status("d2", "failed", error="timeout")
            mock_store.update_status.assert_called_once_with(
                "d2", "failed", receipt_id=None, error_message="timeout"
            )

    def test_store_with_both_receipt_and_error(self, slack_handler):
        mock_store = MagicMock()
        with patch(
            "aragora.storage.slack_debate_store.get_slack_debate_store",
            return_value=mock_store,
            create=True,
        ):
            slack_handler._update_debate_status("d3", "completed", receipt_id="r3", error="warn")
            mock_store.update_status.assert_called_once_with(
                "d3", "completed", receipt_id="r3", error_message="warn"
            )


# ---------------------------------------------------------------------------
# Edge cases and integration
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases across multiple commands."""

    def test_all_help_responses_ephemeral(self, slack_handler):
        body = _body(slack_handler._command_help())
        assert body["response_type"] == "ephemeral"

    def test_search_import_error_via_none_module(self, slack_handler, commands_module, monkeypatch):
        """Search handles get_debates_db being None."""
        monkeypatch.setattr(commands_module, "get_debates_db", None)
        body = _body(slack_handler._command_search("query"))
        assert _status(slack_handler._command_search("query")) == 200

    def test_search_db_no_search_no_list(self, slack_handler, commands_module, monkeypatch):
        """DB with neither search nor list returns no results."""
        mock_db = MagicMock(spec=[])
        monkeypatch.setattr(commands_module, "get_debates_db", MagicMock(return_value=mock_db))
        body = _body(slack_handler._command_search("query"))
        assert "no results" in body["text"].lower()

    def test_debate_with_decision_integrity_dict(self, slack_handler, commands_module, monkeypatch):
        monkeypatch.setattr(commands_module, "create_tracked_task", MagicMock())
        result = slack_handler._command_debate(
            "Important decision topic for the team",
            "U1", "C1", "https://hooks.slack.com/x",
            decision_integrity={"include_receipt": True, "include_plan": False},
        )
        body = _body(result)
        assert body["response_type"] == "in_channel"

    def test_gauntlet_with_workspace_and_team(self, slack_handler, commands_module, monkeypatch):
        monkeypatch.setattr(commands_module, "create_tracked_task", MagicMock())
        result = slack_handler._command_gauntlet(
            "This company should invest heavily in quantum computing",
            "U1", "C1", "https://hooks.slack.com/x",
            workspace=MagicMock(), team_id="T001",
        )
        body = _body(result)
        assert body["response_type"] == "in_channel"

    def test_ask_with_workspace_and_team(self, slack_handler, commands_module, monkeypatch):
        monkeypatch.setattr(commands_module, "create_tracked_task", MagicMock())
        result = slack_handler._command_ask(
            "What are best practices for scaling databases?",
            "U1", "C1", "https://hooks.slack.com/x",
            workspace=MagicMock(), team_id="T001",
        )
        body = _body(result)
        assert body["response_type"] == "in_channel"
